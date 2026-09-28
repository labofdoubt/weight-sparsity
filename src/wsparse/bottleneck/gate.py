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

from .lapsum import lapsum_barrier_sorted, lapsum_budget, lapsum_probs, laplace_cdf
from .rblapsum import GRAD_MODES, VALUE_GRAD_MODES, rblapsum_gate, rblapsum_sf_gate
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
        rblapsum_boundary_grad_mode: Optional[str] = None,
        rblapsum_boundary_floor=None,
        rblapsum_support_scale: float = 1.0,
        rblapsum_temperature: float = 1.0,
        rblapsum_kernel: str = "exponential",
        rblapsum_temperature_mode: str = "fixed",
        rblapsum_chi_target: float = 45.0,
        rblapsum_window_floor: float = 16.0,
        rblapsum_t_min: float = 0.25,
        rblapsum_t_max: float = 8.0,
        rblapsum_servo_rate: float = 0.02,
        rblapsum_sf_value_grad: str = "pool",
        rblapsum_rho_random_perm_prob_grad: float = 0.0,
        surrogate_grad_scale: float = 1.0,
        inactive_grad_scale: float = 1.0,
        project_scale_gradient: bool = False,
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
        validate_gate_shapes(n_features, k, j, n_eff, boundary_mode, surrogate_mode)
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
            "lapsum_adaptive", "lapsum_scheduled", "lapsum_fixed",
            "rblapsum", "rblapsum_sf", "hard"
        ):
            raise ValueError(
                f"unknown surrogate_mode: {surrogate_mode!r} "
                "(lapsum_adaptive | lapsum_scheduled | lapsum_fixed "
                "| rblapsum | rblapsum_sf | hard)"
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
        # all features active: no boundary exists, so no surrogate can apply
        self.trivial = self.k >= self.n_features
        self.n_eff = float(n_eff)
        self.selection_mode = selection_mode
        self.effective_count_metric = effective_count_metric
        self.boundary_mode = boundary_mode
        self.one_sided_weight_mode = one_sided_weight_mode
        self.surrogate_mode = surrogate_mode
        # None mirrors the config default: detach for the hard forward,
        # through_rank_kappa for the soft forward (see ActivationBottleneckConfig)
        self.rblapsum_boundary_grad_mode = (
            ("through_rank_kappa" if surrogate_mode == "rblapsum_sf" else "detach")
            if rblapsum_boundary_grad_mode is None else rblapsum_boundary_grad_mode)
        # None -> 0.0: no floor by default (config resolves this too, but a
        # directly-constructed gate should get the same default)
        self.rblapsum_boundary_floor = (
            0.0 if rblapsum_boundary_floor is None else float(rblapsum_boundary_floor)
        )
        self.rblapsum_temperature = float(rblapsum_temperature)
        self.rblapsum_kernel = rblapsum_kernel
        self.rblapsum_temperature_mode = rblapsum_temperature_mode
        self.rblapsum_chi_target = float(rblapsum_chi_target)
        self.rblapsum_window_floor = float(rblapsum_window_floor)
        self.rblapsum_t_min = float(rblapsum_t_min)
        self.rblapsum_t_max = float(rblapsum_t_max)
        self.rblapsum_servo_rate = float(rblapsum_servo_rate)
        self.rblapsum_sf_value_grad = rblapsum_sf_value_grad
        self.rblapsum_rho_random_perm_prob_grad = float(
            rblapsum_rho_random_perm_prob_grad)
        # Experiment-only knobs, set programmatically (analysis/scale_dynamics.py),
        # deliberately not config fields.  support_scale multiplies the surrogate
        # support gradient g_s in the backward (0 = pure hard task path).
        # view_scale alpha rescales the gate's VIEW of the activations
        # (z -> alpha z at the gate input, output divided by alpha): the forward
        # function and the task gradient are exactly unchanged, while the
        # surrogate sees scores, boundary and spacings scaled by alpha at fixed
        # kernel width T -- a function-preserving emulation of an
        # activation-scale excursion.  Setting view_scale = c together with
        # temperature * c is an exact score-units reparameterization (all
        # gradients invariant).
        # config-settable (see ActivationBottleneckConfig.rblapsum_support_scale);
        # analysis/scale_dynamics.py still overwrites it in place for its
        # counterfactual branches, which is why it stays a plain attribute.
        self.rblapsum_support_scale = float(rblapsum_support_scale)
        self.rblapsum_view_scale = 1.0
        if surrogate_mode in ("rblapsum", "rblapsum_sf"):
            if selection_mode not in ("topk", "abs_topk"):
                raise ValueError(
                    f"surrogate_mode={surrogate_mode!r} requires selection_mode "
                    "'topk' or 'abs_topk' (not gated_topk)"
                )
            if self.rblapsum_boundary_grad_mode not in GRAD_MODES:
                raise ValueError(
                    f"unknown rblapsum_boundary_grad_mode: "
                    f"{self.rblapsum_boundary_grad_mode!r} ({' | '.join(GRAD_MODES)})"
                )
            if rblapsum_kernel != "exponential":
                raise ValueError(
                    f"rblapsum_kernel={rblapsum_kernel!r}: only 'exponential' is "
                    "implemented"
                )
            if self.rblapsum_temperature <= 0:
                raise ValueError("rblapsum_temperature must be positive")
            if not 0.0 <= float(rblapsum_rho_random_perm_prob_grad) <= 1.0:
                raise ValueError(
                    "rblapsum_rho_random_perm_prob_grad must be in [0, 1], "
                    f"got {rblapsum_rho_random_perm_prob_grad!r}")
            if (float(rblapsum_rho_random_perm_prob_grad) != 0.0
                    and surrogate_mode != "rblapsum"):
                raise ValueError(
                    "rblapsum_rho_random_perm_prob_grad is implemented for the "
                    "hard-forward surrogate_mode='rblapsum' only; leave it at 0.0 "
                    f"under {surrogate_mode!r}")
            if rblapsum_sf_value_grad not in VALUE_GRAD_MODES:
                raise ValueError(
                    f"unknown rblapsum_sf_value_grad: {rblapsum_sf_value_grad!r} "
                    f"({' | '.join(VALUE_GRAD_MODES)})"
                )
            if (rblapsum_sf_value_grad != "pool"
                    and surrogate_mode != "rblapsum_sf"):
                raise ValueError(
                    "rblapsum_sf_value_grad is a soft-forward knob and is not "
                    f"applied by surrogate_mode={surrogate_mode!r}; leave it at 'pool'"
                )
            if rblapsum_temperature_mode not in ("fixed", "servo"):
                raise ValueError(
                    f"rblapsum_temperature_mode must be fixed | servo, "
                    f"got {rblapsum_temperature_mode!r}")
            if rblapsum_temperature_mode == "servo":
                if not (0 < self.rblapsum_t_min <= self.rblapsum_temperature
                        <= self.rblapsum_t_max):
                    raise ValueError(
                        "servo needs 0 < rblapsum_t_min <= rblapsum_temperature"
                        " <= rblapsum_t_max")
                if self.rblapsum_chi_target <= 0:
                    raise ValueError("rblapsum_chi_target must be positive")
                if self.rblapsum_window_floor < 1:
                    raise ValueError("rblapsum_window_floor must be >= 1")
                if not 0 < self.rblapsum_servo_rate <= 0.2:
                    raise ValueError("rblapsum_servo_rate must be in (0, 0.2]")
                if self.k <= 4 or self.j < 8:
                    raise ValueError(
                        "servo measures the rank spacing over the 4 ranks each"
                        " side of the boundary: needs k > 4 and j >= 8")
                self.register_buffer(
                    "rb_temp", torch.tensor(float(self.rblapsum_temperature)))
                self.register_buffer("rb_b_ema", torch.zeros(()))
                self.register_buffer("rb_delta_ema", torch.zeros(()))
            # the (K+1)-st score is the rank boundary, so J>=1 (already required
            # by validate_gate_shapes for every non-hard mode) guarantees it
            if self.j < 1:
                raise ValueError("surrogate_mode='rblapsum' needs j >= 1 (the K+1 boundary)")
        self.surrogate_grad_scale = float(surrogate_grad_scale)
        self.inactive_grad_scale = float(inactive_grad_scale)
        self.project_scale_gradient = bool(project_scale_gradient)
        self.fixed_temperature = float(fixed_temperature)
        self.temperature_scale_mode = temperature_scale_mode
        self.temperature_solver_tol = float(temperature_solver_tol)
        self.temperature_solver_max_iters = int(temperature_solver_max_iters)
        self.barrier_solver_tol = float(barrier_solver_tol)
        self.solver_dtype = _DTYPES[solver_dtype]
        self.log_diagnostics = bool(log_diagnostics)
        self.hard_inference = bool(hard_inference)

        # Set by the controller from the schedule each optimizer step; a
        # buffer so it travels with .to(device), non-persistent because it is a
        # pure function of the step count.
        self.register_buffer(
            "scheduled_temperature",
            torch.tensor(float(fixed_temperature), dtype=torch.float32),
            persistent=False,
        )
        # EMA of how often each of the N features is selected, for the
        # dead-feature diagnostics below.  Zero-initialized and bias-corrected
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
        # is the answer, for the exact modes it is the Newton initializer.
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

        if self.trivial:
            # k == n_features: every feature is active, so the mask is all ones
            # and topk would be a full sort for nothing.  Kept as a control run
            # rather than an optimization of a real configuration.
            hard_mask = torch.ones_like(scores)
            if self.log_diagnostics and self.training:
                self._record_usage(hard_mask)
            return value * hard_mask

        if self.surrogate_mode in ("rblapsum", "rblapsum_sf"):
            return self._rblapsum(scores, value)

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
        # Optional analysis capture: the LapSum support gradient in SELECTION-
        # SCORE space (d~L/ds over the K+J pool, |z|-space under abs_topk,
        # BEFORE the sign chain back to z).  Enabled by assigning a dict to
        # `_score_grad_capture`; costs nothing when None (the default).
        if getattr(self, "_score_grad_capture", None) is not None:
            _cap = self._score_grad_capture
            _cap["idx"] = cand_idx.detach()
            cand.register_hook(lambda g, _cap=_cap: _cap.__setitem__("grad", g.detach()))
        detached = cand.detach()
        sink = self._grad_sink if self.log_diagnostics else None

        b, t, solver_diag = self.solve(detached)
        # Evaluate the probabilities about r_K.  Shifting by a detached constant
        # leaves dz/dr -- and so the whole VJP -- untouched.  Note this is inert
        # for *precision*: (s-c)-(b-c) loses the same mantissa as s-b, since the
        # damage is done representing s itself (measured: a 1e4 offset perturbs
        # s-c by ~9e-4 in float32, centred or not).  Where centring genuinely
        # pays is the barrier and Newton solves.  Kept because it costs nothing
        # and keeps the exponent small if b ever drifts far from the scores.
        centre = detached[..., self.k - 1 : self.k]
        p = lapsum_probs(
            cand - centre, b - centre.squeeze(-1), t, self.k, sink,
            inactive_scale=self.inactive_grad_scale,
            project_scale=self.project_scale_gradient,
        )
        p_full = (
            torch.zeros_like(scores, dtype=p.dtype)
            .scatter(-1, cand_idx, p)
            .to(value.dtype)
        )
        mask = hard_mask + self.surrogate_grad_scale * (p_full - p_full.detach())

        if self.log_diagnostics:
            self._record(detached, b, t, p.detach(), solver_diag)
        return value * mask

    # ---- rblapsum ------------------------------------------------------------- #
    def _rblapsum(self, scores: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Hard TopK forward with the rank-boundary local support gradient.

        ONE Top(K+J) is reused for the candidates, the first-K hard forward and
        the (K+1)-st rank boundary.  The hard support is ``TopK AND s > b0`` --
        never more than K, fewer where fewer than K clear the floor -- and the
        three grad modes share this forward exactly (see :mod:`.rblapsum`).
        """
        b0 = self.rblapsum_boundary_floor
        alpha = float(self.rblapsum_view_scale)
        if alpha != 1.0:
            scores = scores * alpha
            value = value * alpha
        cand_scores, cand_idx = torch.topk(scores, self.m, dim=-1, largest=True, sorted=True)
        value_c = torch.gather(value, -1, cand_idx)          # signed z, carries grad
        score_c = cand_scores.detach()                       # s = |z| or z, sorted desc
        b_rank = score_c[..., self.k:self.k + 1]             # the (K+1)-st score
        cap_active = b_rank > b0                             # rank cap binds?
        b = torch.clamp(b_rank, min=b0)                      # b = max(b0, b_rank)
        # active = TopK AND s > b0, using TopK's own tie-breaking (position < k)
        active_c = torch.zeros_like(score_c)
        active_c[..., :self.k] = (score_c[..., :self.k] > b0).to(score_c.dtype)
        sign_c = (value_c.sign() if self.selection_mode == "abs_topk"
                  else torch.ones_like(value_c))

        servo = self.rblapsum_temperature_mode == "servo"
        if servo and self.training:
            self._rb_servo_update(score_c, b)
        t = float(self.rb_temp) if servo else self.rblapsum_temperature
        sink = self._grad_sink if self.log_diagnostics and self.training else None
        # rblapsum_sf: the probabilities are IN the forward, y = z * p over the
        # whole pool.  In eval, hard_inference=True (the default) falls back to
        # the hard Top-K forward below, so the base validation metric measures
        # the hardened model and stays name-comparable with the other regimes;
        # flip hard_inference off around a second eval pass for the soft CE.
        soft = (self.surrogate_mode == "rblapsum_sf"
                and (self.training or not self.hard_inference))
        if soft:
            p_c = laplace_cdf((score_c - b) / t)
            y_c = rblapsum_sf_gate(value_c, p_c, active_c, score_c, sign_c,
                                   b, t, self.rblapsum_boundary_grad_mode,
                                   self.k, cap_active, sink,
                                   supp_scale=float(self.rblapsum_support_scale),
                                   value_grad=self.rblapsum_sf_value_grad)
        else:
            y_c = rblapsum_gate(value_c, active_c, score_c, sign_c, b, t,
                                self.rblapsum_boundary_grad_mode, self.k, cap_active, sink,
                                supp_scale=float(self.rblapsum_support_scale),
                                perm_rho=(self.rblapsum_rho_random_perm_prob_grad
                                          if self.training else 0.0))
        y = torch.zeros_like(value).scatter(-1, cand_idx, y_c.to(value.dtype))
        if alpha != 1.0:
            y = y / alpha

        if self.log_diagnostics and self.training:
            with torch.no_grad():
                mask_full = torch.zeros_like(scores).scatter(
                    -1, cand_idx, active_c.to(scores.dtype))
                self._record_usage(mask_full)
                self._record_rblapsum(active_c, cap_active, b_rank, b,
                                      p_c if soft else None)
        return y

    @torch.no_grad()
    def _rb_servo_update(self, score_c: torch.Tensor, b: torch.Tensor) -> None:
        """One step of the kernel-temperature servo (training forwards only).

        Two controls on the per-layer scalar ``rb_temp``
        (docs/rblapsum-kappa-stability.md has the evidence):

        - population floor: if fewer than ``rblapsum_window_floor`` candidates
          sit inside the kernel window this batch, T grows 10% immediately.
          The divergence cliff is the window emptying -- the kappa zero-sum
          then concentrates onto one feature (through_rank's point sink) and
          the boundary runs away -- so the window is never allowed to empty.
        - geometric-pressure trim (at most 2%/step): hold

              chi_geo = EMA(b) / (2 * T * EMA(delta(b)))

          at ``rblapsum_chi_target``, where delta(b) is the local score
          spacing per rank at the boundary.  chi_geo is dimensionless pure
          forward geometry -- no gradient units, so no dependence on batch
          size or loss normalisation (a kick-based trim was tried first and
          falsified by its own telemetry: gradient-unit targets do not
          transfer across batch sizes).  Measured chi_geo of the campaign:
          every run that stayed <= ~55 survived; every death carried >= ~58
          somewhere; >= ~110 is seed-roulette territory.

        The trim is PROTECTIVE-ONLY: it never sharpens T below the
        configured ``rblapsum_temperature``.  Holding the chi target
        symmetrically was falsified live: safe geometry relaxes over
        training (the stable k32/T1 flagship runs at chi_geo 39 -> 30), so
        a symmetric trim walks T down into the proven-marginal band --
        the first k32 null test destabilised exactly that way at ~4k.
        The servo therefore rises above the baseline when geometry demands
        (larger K, bursts, the early transient) and relaxes back TO the
        baseline, never below it.  Set the baseline to the value you would
        run fixed (1.0).

        Uses local batches only -- under DDP each rank servos its own copy.
        """
        t = float(self.rb_temp)
        m_sp = 4
        delta = (score_c[..., self.k - 1 - m_sp]
                 - score_c[..., self.k - 1 + m_sp]) / (2 * m_sp)
        delta_b = delta.mean()
        b_mean = b.mean()
        n_win = ((score_c - b).abs() < t).sum(-1).to(torch.float32).mean()
        ema = 0.95
        # init-on-first-use, then EMA
        if float(self.rb_delta_ema) == 0:
            self.rb_delta_ema.fill_(float(delta_b))
            self.rb_b_ema.fill_(float(b_mean))
        else:
            self.rb_delta_ema.mul_(ema).add_((1 - ema) * delta_b)
            self.rb_b_ema.mul_(ema).add_((1 - ema) * b_mean)
        chi_geo = float(self.rb_b_ema) / max(
            2 * t * float(self.rb_delta_ema), 1e-30)
        if float(n_win) < self.rblapsum_window_floor:
            self.rb_temp.mul_(1.10)
        else:
            step = self.rblapsum_servo_rate * math.log(
                max(chi_geo, 1e-30) / self.rblapsum_chi_target)
            self.rb_temp.mul_(math.exp(max(-0.0198, min(0.0198, step))))
        self.rb_temp.clamp_(
            max(self.rblapsum_t_min, self.rblapsum_temperature),
            self.rblapsum_t_max)
        diag = {
            "rb_temp": self.rb_temp.detach().clone(),
            "rb_win_count": n_win.detach(),
            "rb_chi": delta_b.new_tensor(chi_geo),
        }
        kick = self._grad_sink.get("rb_kick_win")
        if kick is not None:
            diag["rb_kick"] = kick
        self._rb_servo_diag = diag

    @torch.no_grad()
    def _record_rblapsum(self, active_c, cap_active, b_rank, b,
                         p_c=None) -> None:
        """Forward diagnostics for rblapsum.

        ``active_count`` stays the HARD support size (rank <= K and s > b0) in
        every mode, so the L0 log line and its TB series remain comparable
        across regimes; the sf forward's dense pool shows up as ``rb_soft_mass``
        (mean over tokens of sum_i p_i) instead.
        """
        d = {
            "active_count": active_c.sum(-1).mean(),
            "active_count_max": active_c.sum(-1).max(),
            "rb_cap_active_frac": cap_active.float().mean(),
            "rb_b_rank": b_rank.mean(),
            "rb_boundary": b.mean(),
        }
        if p_c is not None:
            d["rb_soft_mass"] = p_c.sum(-1).mean()
        if getattr(self, "_rb_servo_diag", None):
            d.update(self._rb_servo_diag)
        self._forward_diag = {key: value.detach() for key, value in d.items()}

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
        """Forward-only diagnostics for the no-surrogate baseline.

        With ``j = 0`` there is no candidate beyond the support, so the
        boundary gap does not exist; the span degenerates to within-support
        spread and the gap is recorded as 0 rather than indexing off the end.
        """
        zero = cand.new_zeros(())
        gap = ((cand[..., self.k - 1] - cand[..., self.k]).mean()
               if cand.shape[-1] > self.k else zero)
        self._forward_diag = {
            "score_gap": gap,
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
    n_features: int,
    k: int,
    j: int,
    n_eff: float,
    boundary_mode: str,
    surrogate_mode: str = "lapsum_adaptive",
) -> None:
    """Section 23 of the spec, enforced up front rather than as runtime NaNs.

    A hard mask never solves a barrier, so two geometries that are degenerate
    for the surrogate are meaningful there: ``j = 0`` (no candidates beyond the
    support) and ``k = n_features`` (every feature active -- a *trivial*
    bottleneck, i.e. the bare ``W_in -> W_out`` projection pair, which isolates
    the cost of the extra parameters from the cost of the sparsity).  ``n_eff``
    is inert without a surrogate, so it is left unchecked.
    """
    if surrogate_mode == "hard":
        if not 1 <= k <= n_features:
            raise ValueError(
                f"require 1 <= k <= n_features, got k={k}, n_features={n_features}"
            )
        if j < 0:
            raise ValueError(f"require j >= 0, got j={j}")
        if k + j > n_features:
            raise ValueError(
                f"require k + j <= n_features, got k={k}, j={j}, n_features={n_features}"
            )
        return
    if not 1 <= k < n_features:
        raise ValueError(f"require 1 <= k < n_features, got k={k}, n_features={n_features}")
    if j < 1:
        raise ValueError(f"require j >= 1, got j={j}")
    if k + j > n_features:
        raise ValueError(
            f"require k + j <= n_features, got k={k}, j={j}, n_features={n_features}"
        )
    if surrogate_mode in ("rblapsum", "rblapsum_sf"):
        # same pool geometry as the LapSum modes, but n_eff is inert: these
        # modes replace the effective-count calibration rather than aiming at it
        return
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
