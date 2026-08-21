"""The activation gate: exact hard TopK forward, LapSum Top(K+J) surrogate backward.

Forward is exactly ``K``-sparse, always::

    a_hat_i = a_i * 1[i in TopK(r)],    r = a  (topk)  or  |a|  (abs_topk)

Backward additionally lets the next ``J`` candidates move, through a soft LapSum
mask that is numerically inert in the forward pass::

    m = m_hard + lambda * (p - stopgrad(p))

so ``m == m_hard`` numerically while ``dm/dr == lambda * dp/dr``.  Everything
outside Top(K+J) gets exactly zero gradient from this module.

One ``torch.topk`` per call supplies the sorted candidate pool that the hard
mask, the temperature solve, the barrier solve, the probabilities, the backward
and the diagnostics all share -- nothing is sorted twice.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .lapsum import lapsum_barrier_sorted, lapsum_budget, lapsum_probs
from .temperature import (
    STATUS_OK,
    gradient_count,
    score_softmax_count,
    solve_joint_temperature,
    solve_reference_temperature,
    solve_score_softmax_temperature,
    status_fractions,
)

_DTYPES = {"float32": torch.float32, "float64": torch.float64}


class AdaptiveLapSumTopKGate(nn.Module):
    """Hard TopK / AbsTopK with an adaptive-bandwidth LapSum surrogate gradient."""

    def __init__(
        self,
        n_features: int,
        k: int,
        j: int,
        n_eff: float,
        selection_mode: str = "abs_topk",
        effective_count_metric: str = "ess",
        boundary_mode: str = "outside_only",
        one_sided_weight_mode: str = "score_softmax",
        surrogate_mode: str = "lapsum_adaptive",
        surrogate_grad_scale: float = 1.0,
        fixed_temperature: float = 1.0,
        temperature_scale_mode: str = "relative",
        temperature_solver_tol: float = 1e-5,
        temperature_solver_max_iters: int = 12,
        barrier_solver_tol: float = 1e-6,
        solver_dtype: str = "float32",
        log_diagnostics: bool = True,
        hard_inference: bool = True,
    ):
        super().__init__()
        validate_gate_shapes(n_features, k, j, n_eff, boundary_mode)
        if selection_mode not in ("topk", "abs_topk", "gated_topk"):
            raise ValueError(
                f"unknown selection_mode: {selection_mode!r} (topk | abs_topk | gated_topk)"
            )
        if effective_count_metric not in ("ess", "entropy"):
            raise ValueError(
                f"unknown effective_count_metric: {effective_count_metric!r} (ess | entropy)"
            )
        if boundary_mode not in ("outside_only", "both_sides"):
            raise ValueError(
                f"unknown boundary_mode: {boundary_mode!r} (outside_only | both_sides)"
            )
        if one_sided_weight_mode not in ("score_softmax", "true_gradient"):
            raise ValueError(
                f"unknown one_sided_weight_mode: {one_sided_weight_mode!r} "
                "(score_softmax | true_gradient)"
            )
        if surrogate_mode not in (
            "lapsum_adaptive", "lapsum_scheduled", "lapsum_fixed", "hard"
        ):
            raise ValueError(
                f"unknown surrogate_mode: {surrogate_mode!r} "
                "(lapsum_adaptive | lapsum_scheduled | lapsum_fixed | hard)"
            )
        if temperature_scale_mode not in ("relative", "absolute"):
            raise ValueError(
                f"unknown temperature_scale_mode: {temperature_scale_mode!r} "
                "(relative | absolute)"
            )
        if solver_dtype not in _DTYPES:
            raise ValueError(f"unknown solver_dtype: {solver_dtype!r} (float32 | float64)")

        self.n_features = int(n_features)
        self.k = int(k)
        self.j = int(j)
        self.m = self.k + self.j
        self.n_eff = float(n_eff)
        self.selection_mode = selection_mode
        self.effective_count_metric = effective_count_metric
        self.boundary_mode = boundary_mode
        self.one_sided_weight_mode = one_sided_weight_mode
        self.surrogate_mode = surrogate_mode
        self.surrogate_grad_scale = float(surrogate_grad_scale)
        self.fixed_temperature = float(fixed_temperature)
        self.temperature_scale_mode = temperature_scale_mode
        self.temperature_solver_tol = float(temperature_solver_tol)
        self.temperature_solver_max_iters = int(temperature_solver_max_iters)
        self.barrier_solver_tol = float(barrier_solver_tol)
        self.solver_dtype = _DTYPES[solver_dtype]
        self.log_diagnostics = bool(log_diagnostics)
        self.hard_inference = bool(hard_inference)

        # Set by the controller from the schedule each optimiser step; a
        # buffer so it travels with .to(device), non-persistent because it is a
        # pure function of the step count.
        self.register_buffer(
            "scheduled_temperature",
            torch.tensor(float(fixed_temperature), dtype=torch.float32),
            persistent=False,
        )
        # EMA of how often each of the N features is selected, for the
        # dead-feature diagnostics below.  Zero-initialised and bias-corrected
        # on read (as in Adam): seeding it at the uniform rate instead would
        # take ~460 steps to decay past the dead threshold, so a fully collapsed
        # bottleneck would report 0% dead for the whole early phase.
        self.register_buffer(
            "usage_ema", torch.zeros(self.n_features, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "usage_steps", torch.zeros((), dtype=torch.float32), persistent=False
        )
        self._forward_diag: Dict[str, torch.Tensor] = {}
        self._usage_diag: Dict[str, torch.Tensor] = {}
        self._grad_sink: Dict[str, torch.Tensor] = {}

    @property
    def diagnostics(self) -> Dict[str, torch.Tensor]:
        """Forward-pass diagnostics merged with the latest backward-pass ones.

        Merged on read rather than snapshotted in ``forward`` because the
        gradient magnitudes are only known once ``backward`` has run.
        """
        return {**self._forward_diag, **self._usage_diag, **self._grad_sink}

    # ---- selection --------------------------------------------------------- #
    @property
    def gated(self) -> bool:
        """Independent score and value branches (``selection_mode='gated_topk'``)."""
        return self.selection_mode == "gated_topk"

    def scores_of(self, a: torch.Tensor) -> torch.Tensor:
        """Ranking score.  Autograd carries ``dr/da`` (1, or sign(a)) for free."""
        return a.abs() if self.selection_mode == "abs_topk" else a

    def surrogate_active(self) -> bool:
        if self.surrogate_mode == "hard":
            return False
        if not torch.is_grad_enabled():
            return False  # the surrogate term is identically zero without grad
        if self.hard_inference and not self.training:
            return False
        return True

    # ---- solve ------------------------------------------------------------- #
    @property
    def calibration(self) -> slice:
        """Which candidates enter the effective-count equation.

        Inactive candidates only for one-sided calibration, every candidate for
        two-sided.  ``F1 = sum p - K`` always runs over the whole pool.
        """
        return slice(self.k, self.m) if self.boundary_mode == "outside_only" else slice(0, self.m)

    @property
    def exact_calibration(self) -> bool:
        """True when N_eff is calibrated on the actual LapSum gradient weights."""
        return self.boundary_mode == "both_sides" or self.one_sided_weight_mode == "true_gradient"

    def temperature_scale(self, candidates: torch.Tensor) -> torch.Tensor:
        """Per-row score scale a prescribed temperature is measured in.

        ``relative`` uses the standard deviation of the Top-(K+J) scores, so the
        schedule value is exactly the ``temperature_rel`` that gets logged and
        stays meaningful as the activation scale drifts.  Degenerate rows fall
        back to 1 rather than collapsing ``t`` to 0.
        """
        if self.temperature_scale_mode == "absolute":
            return torch.ones_like(candidates[..., 0])
        std = candidates.std(-1)
        return torch.where(std > 0, std, torch.ones_like(std))

    def prescribed_temperature(self, candidates: torch.Tensor) -> torch.Tensor:
        """``t`` for the fixed / scheduled modes (no root-find involved)."""
        return self.scheduled_temperature.to(candidates.dtype) * self.temperature_scale(
            candidates
        )

    def solve(self, candidates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """``(b, t)`` for a batch of **detached, sorted** candidate rows."""
        diag: Dict[str, torch.Tensor] = {}
        span = (candidates[..., :1] - candidates[..., -1:]).squeeze(-1).clamp_min(0.0)
        scale = torch.where(span > 0, span, torch.ones_like(span))

        if self.surrogate_mode in ("lapsum_fixed", "lapsum_scheduled"):
            # No solve at all: the temperature is prescribed, and the barrier
            # still follows in closed form so the budget stays exactly K.
            t = self.prescribed_temperature(candidates)
            diag = {"temperature_scheduled": self.scheduled_temperature.detach()}
            return lapsum_barrier_sorted(candidates, self.k, t), t, diag

        # Every mode starts from the cheap decoupled solve: for score_softmax it
        # is the answer, for the exact modes it is the Newton initialiser.
        t, diag = solve_score_softmax_temperature(
            candidates[..., self.k :],
            self.n_eff,
            self.effective_count_metric,
            tol=self.temperature_solver_tol,
            max_iters=self.temperature_solver_max_iters,
            fallback_scale=scale,
        )
        b = lapsum_barrier_sorted(candidates, self.k, t)
        if not self.exact_calibration:
            return b, t, diag

        b, t, ok, nd = solve_joint_temperature(
            candidates,
            self.k,
            self.n_eff,
            b,
            t,
            self.effective_count_metric,
            calibration=self.calibration,
            tol=self.temperature_solver_tol,
            budget_tol=self.barrier_solver_tol,
            max_iters=self.temperature_solver_max_iters,
        )
        diag = dict(diag)
        diag.update(nd)
        if not bool(ok.all()):
            # Never silently return the last Newton iterate: unconverged rows go
            # to the reference root search, which also reports whether the
            # target was attainable at all for that row.
            b_ref, t_ref, status = solve_reference_temperature(
                candidates,
                self.k,
                self.n_eff,
                self.effective_count_metric,
                calibration=self.calibration,
            )
            b = torch.where(ok, b, b_ref)
            t = torch.where(ok, t, t_ref)
            diag["temp_status"] = torch.where(ok, diag["temp_status"], status)
        return b, t, diag

    # ---- forward ------------------------------------------------------------ #
    def forward(self, a: torch.Tensor, values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``values * mask``, where the mask is hard TopK over the *scores*.

        With ``gated_topk`` the two arguments are independent branches: ``a`` is
        the score ``s`` that decides the support, ``values`` is the value ``v``
        that is carried.  The gradients then separate exactly as intended --
        ``dL/dv = m * g`` because the mask is numerically hard, and
        ``dL/ds`` is the constrained LapSum VJP applied to ``u = g * v``, since
        that is what autograd hands the custom Function.  For the other modes
        the value *is* the score tensor, which is the original behaviour.
        """
        if self.gated:
            if values is None:
                raise ValueError("selection_mode='gated_topk' requires a value branch")
            scores, value = a, values
        else:
            if values is not None:
                raise ValueError(
                    f"selection_mode={self.selection_mode!r} takes a single tensor; "
                    "a separate value branch is only used by gated_topk"
                )
            scores, value = self.scores_of(a), a

        cand_scores, cand_idx = torch.topk(
            scores, self.m, dim=-1, largest=True, sorted=True
        )
        hard_mask = torch.zeros_like(scores).scatter(-1, cand_idx[..., : self.k], 1.0)
        if self.log_diagnostics and self.training:
            self._record_usage(hard_mask)

        if not self.surrogate_active():
            if self.log_diagnostics and self.training:
                # Keep the hard baseline comparable with the surrogate runs: the
                # forward-side statistics are all still meaningful, and the
                # gradient on the J candidates is exactly zero by construction
                # rather than merely absent.
                self._record_hard(cand_scores.to(self.solver_dtype).detach())
            return value * hard_mask

        cand = cand_scores.to(self.solver_dtype)
        detached = cand.detach()
        b, t, solver_diag = self.solve(detached)

        sink = self._grad_sink if self.log_diagnostics else None
        # Evaluate the probabilities about r_K.  Shifting by a detached constant
        # leaves dz/dr -- and so the whole VJP -- untouched.  Note this is inert
        # for *precision*: (s-c)-(b-c) loses the same mantissa as s-b, since the
        # damage is done representing s itself (measured: a 1e4 offset perturbs
        # s-c by ~9e-4 in float32, centred or not).  Where centring genuinely
        # pays is the barrier and Newton solves.  Kept because it costs nothing
        # and keeps the exponent small if b ever drifts far from the scores.
        centre = detached[..., self.k - 1 : self.k]
        p = lapsum_probs(cand - centre, b - centre.squeeze(-1), t, self.k, sink)
        p_full = (
            torch.zeros_like(scores, dtype=p.dtype)
            .scatter(-1, cand_idx, p)
            .to(value.dtype)
        )
        mask = hard_mask + self.surrogate_grad_scale * (p_full - p_full.detach())

        if self.log_diagnostics:
            self._record(detached, b, t, p.detach(), solver_diag)
        return value * mask

    # ---- diagnostics --------------------------------------------------------- #
    @torch.no_grad()
    def feature_usage(self) -> torch.Tensor:
        """Per-feature selection rate, bias-corrected, as a length-N vector.

        The same quantity the ``feature_*`` scalars are reduced from, exposed so
        the distribution itself can be logged rather than only its summaries.
        """
        bias = (1.0 - 0.99**self.usage_steps).clamp_min(torch.finfo(torch.float32).eps)
        return (self.usage_ema / bias).detach()

    @torch.no_grad()
    def _record_usage(self, hard_mask: torch.Tensor, decay: float = 0.99) -> None:
        """How evenly the K slots are spread over the N features.

        The characteristic failure of a TopK activation bottleneck is feature
        collapse: a subset of features wins every token and the rest are never
        selected, so their ``W_in``/``W_out`` columns stop receiving gradient
        entirely and the effective width is far below N.  Nothing else logged
        here would show that -- the loss, the budget and N_eff all look healthy
        while it happens -- so it is tracked over an EMA window rather than a
        single batch, which a small batch would make far too noisy.
        """
        rate = hard_mask.reshape(-1, self.n_features).mean(0).float()
        self.usage_ema.mul_(decay).add_(rate, alpha=1.0 - decay)
        self.usage_steps.add_(1.0)
        bias = 1.0 - decay**self.usage_steps
        usage = self.usage_ema / bias.clamp_min(torch.finfo(rate.dtype).eps)
        uniform = self.k / self.n_features
        total = usage.sum().clamp_min(torch.finfo(usage.dtype).tiny)
        p = usage / total
        self._usage_diag = {
            # selected less than 1% as often as a uniform allocation would
            "feature_dead_frac": (usage < 0.01 * uniform).float().mean(),
            # exp(H) / N: 1.0 is perfectly even usage, ->0 is total collapse
            "feature_usage_entropy": torch.exp(-torch.xlogy(p, p).sum()) / self.n_features,
            "feature_usage_max": (usage.max() / uniform),
        }

    @torch.no_grad()
    def _record_hard(self, cand) -> None:
        """Forward-only diagnostics for the no-surrogate baseline."""
        zero = cand.new_zeros(())
        self._forward_diag = {
            "score_gap": (cand[..., self.k - 1] - cand[..., self.k]).mean(),
            "score_span": (cand[..., self.k - 1] - cand[..., -1]).mean(),
            "grad_inactive": zero,
            "grad_active": zero,
        }
        self._grad_sink.clear()

    @torch.no_grad()
    def _record(self, cand, b, t, p, solver_diag) -> None:
        k = self.k
        gap = cand[..., k - 1] - cand[..., k]
        span = cand[..., k - 1] - cand[..., -1]
        budget = p.sum(-1) - k
        std = cand.std(-1).clamp_min(torch.finfo(cand.dtype).tiny)

        # Is the cheap approximation exact here?  softmax(r/t) over the inactive
        # tail is proportional to the true kappa weights exactly when every
        # outside candidate sits below the barrier; r_{K+1} is the highest of
        # them, so r_{K+1} - b < 0 certifies it for the whole row.
        first_inactive = cand[..., k]
        barrier_gap = first_inactive - b
        n_eff_score = score_softmax_count(cand[..., k:], t, self.effective_count_metric)
        n_eff_true = gradient_count(
            cand, b, t, self.effective_count_metric, slice(k, self.m)
        )
        realized = (
            gradient_count(cand, b, t, self.effective_count_metric, self.calibration)
            if self.exact_calibration
            else n_eff_score
        )
        d = {
            "temperature": t.mean(),
            "temperature_rel": (t / std).mean(),
            "barrier": b.mean(),
            "n_eff_realized": realized.mean(),
            "n_eff_error": (realized - self.n_eff).mean(),
            "n_eff_abs_error": (realized - self.n_eff).abs().mean(),
            # the approximation-quality probes
            "barrier_gap": barrier_gap.mean(),
            "barrier_gap_rel": (barrier_gap / t).mean(),
            "frac_above_barrier": (first_inactive > b).float().mean(),
            "n_eff_score": n_eff_score.mean(),
            "n_eff_true_gradient": n_eff_true.mean(),
            "n_eff_gap": (n_eff_true - n_eff_score).mean(),
            "budget_residual": budget.abs().mean(),
            "barrier_failures": (budget.abs() > self._budget_tolerance(cand, t)).float().mean(),
            "score_gap": gap.mean(),
            "score_span": span.mean(),
        }
        for key in ("temp_iters", "newton_iters", "newton_failed", "temperature_scheduled"):
            if key in solver_diag:
                d[key] = solver_diag[key].mean()
        if "temp_status" in solver_diag:
            d.update(status_fractions(solver_diag["temp_status"]))
        self._forward_diag = {key: value.detach() for key, value in d.items()}

    def _budget_tolerance(self, cand: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """What ``|sum p - K|`` can actually reach in the solver dtype.

        Scores arrive already rounded, so ``(r_i - b)`` carries an absolute
        error of about ``eps * |r|``, which dividing by ``t`` amplifies into the
        exponent.  Only the candidates within a few ``t`` of the barrier have an
        appreciable ``dF/dz``, so roughly ``n_eff`` of them contribute at up to
        ``1/4`` each.  Comparing against a flat ``barrier_solver_tol * K`` would
        report a failure on every batch of offset activations even though the
        solver is exact -- the limit is the input representation.
        """
        eps = torch.finfo(cand.dtype).eps
        floor = 0.25 * self.n_eff * eps * cand.abs().amax(-1) / t.clamp_min(
            torch.finfo(cand.dtype).tiny
        )
        return floor.clamp_min(self.barrier_solver_tol * max(self.k, 1))

    def extra_repr(self) -> str:
        return (
            f"n_features={self.n_features}, k={self.k}, j={self.j}, "
            f"n_eff={self.n_eff:g}, mode={self.selection_mode}, "
            f"metric={self.effective_count_metric}, boundary={self.boundary_mode}, "
            f"weights={self.one_sided_weight_mode}, "
            f"surrogate={self.surrogate_mode}"
            + (
                f", t_scale={self.temperature_scale_mode}"
                if self.surrogate_mode in ("lapsum_scheduled", "lapsum_fixed")
                else ""
            )
        )


def validate_gate_shapes(
    n_features: int, k: int, j: int, n_eff: float, boundary_mode: str
) -> None:
    """Section 23 of the spec, enforced up front rather than as runtime NaNs."""
    if not 1 <= k < n_features:
        raise ValueError(f"require 1 <= k < n_features, got k={k}, n_features={n_features}")
    if j < 1:
        raise ValueError(f"require j >= 1, got j={j}")
    if k + j > n_features:
        raise ValueError(
            f"require k + j <= n_features, got k={k}, j={j}, n_features={n_features}"
        )
    if boundary_mode == "outside_only":
        if not 1.0 < n_eff < j:
            raise ValueError(
                f"one-sided calibration requires 1 < n_eff < j, got n_eff={n_eff}, j={j}"
            )
    elif not 1.0 < n_eff < k + j:
        raise ValueError(
            f"two-sided calibration requires 1 < n_eff < k + j, "
            f"got n_eff={n_eff}, k+j={k + j}"
        )
