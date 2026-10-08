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

from .lapsum import lapsum_barrier_sorted, lapsum_probs, laplace_cdf
from .laplace_policy import (ESTIMATORS, RB_SCOPES, WIDTH_GRADIENTS, ProjectScaleFree,
                             RaoBlackwellSelect,
                             SUPPORT_SCALE_MODES, TEMPERATURE_MODES, PolicyRecord,
                             at_least_float32, effective_width, laplace_log_density,
                             sample_laplace, support_multiplier)
from .rblapsum import (GRAD_MODES, SURROGATE_SCOPES, VALUE_GRAD_MODES, carry_scope,
                       kernel_width, rblapsum_carry_gate, rblapsum_gate, rblapsum_sf_gate,
                       relative_kernel_width, strength_scale)

_DTYPES = {"float32": torch.float32, "float64": torch.float64}


class AdaptiveLapSumTopKGate(nn.Module):
    """Hard TopK / AbsTopK with LapSum / RBLapSum surrogate gradients."""

    def __init__(
        self,
        n_features: int,
        k: int,
        j: int,
        selection_mode: str = "abs_topk",
        surrogate_mode: str = "hard",
        rblapsum_boundary_grad_mode: Optional[str] = None,
        rblapsum_boundary_floor=None,
        rblapsum_support_scale: float = 1.0,
        temperature: float = 1.0,
        rblapsum_sf_value_grad: str = "pool",
        rblapsum_rho_random_perm_prob_grad: float = 0.0,
        rblapsum_surrogate_scope: str = "pool",
        rblapsum_relative_temperature: bool = False,
        rblapsum_kernel_width: str = "fixed",
        rblapsum_support_strength=None,
        rblapsum_radial_project: bool = False,
        rblapsum_center_tokens: bool = False,
        rblapsum_carry_decay: float = 1.0,
        barrier_solver_tol: float = 1e-6,
        solver_dtype: str = "float32",
        log_diagnostics: bool = True,
        hard_inference: bool = True,
        value_shift: str = "none",
        value_shift_lambda: float = 0.0,
        stochastic_width: str = "none",
        stochastic_width_param: float = 0.5,
        policy_temperature_mode: str = "absolute",
        policy_min_temperature: float = 1e-6,
        policy_center_scores: bool = True,
        policy_support_scale: float = 1.0,
        policy_support_scale_mode: str = "constant",
        policy_support_temperature_ref: float = 1.0,
        policy_support_scale_max=None,
        policy_estimator: str = "likelihood_ratio",
        policy_rb_scope: str = "full",
        policy_width_gradient: str = "frozen",
        policy_rb_samples: int = 1,
    ):
        super().__init__()
        validate_gate_shapes(n_features, k, j, surrogate_mode)
        # see ActivationBottleneckConfig.stochastic_width
        if stochastic_width not in ("none", "uniform", "two_point", "geometric"):
            raise ValueError(f"unknown stochastic_width: {stochastic_width!r}")
        if stochastic_width != "none" and (surrogate_mode != "hard" or j < 1):
            raise ValueError("stochastic_width needs surrogate_mode='hard' and j >= 1")
        self.stochastic_width = stochastic_width
        self.stochastic_width_param = float(stochastic_width_param)
        if value_shift not in ("none", "fixed", "energy"):
            raise ValueError(
                f"unknown value_shift: {value_shift!r} (none | fixed | energy)")
        if value_shift != "none" and (selection_mode != "abs_topk"
                                      or surrogate_mode != "hard"):
            raise ValueError(
                "value_shift needs selection_mode='abs_topk' and "
                f"surrogate_mode='hard', got {selection_mode!r} / {surrogate_mode!r}")
        # See ActivationBottleneckConfig.value_shift.  Plain attributes: the
        # shift is a fixed rule, not a parameter, so state_dicts are unchanged.
        self.value_shift = value_shift
        self.value_shift_lambda = float(value_shift_lambda)
        if selection_mode not in ("topk", "abs_topk", "gated_topk"):
            raise ValueError(
                f"unknown selection_mode: {selection_mode!r} (topk | abs_topk | gated_topk)"
            )
        if surrogate_mode not in ("lapsum", "rblapsum", "rblapsum_sf", "hard", "soft_ste",
                                  "laplace_policy"):
            raise ValueError(
                f"unknown surrogate_mode: {surrogate_mode!r} "
                "(lapsum | rblapsum | rblapsum_sf | hard | soft_ste | laplace_policy)"
            )
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if solver_dtype not in _DTYPES:
            raise ValueError(f"unknown solver_dtype: {solver_dtype!r} (float32 | float64)")

        self.n_features = int(n_features)
        self.k = int(k)
        self.j = int(j)
        self.m = self.k + self.j
        # all features active: no boundary exists, so no surrogate can apply
        self.trivial = self.k >= self.n_features
        self.selection_mode = selection_mode
        self.surrogate_mode = surrogate_mode
        self.temperature = float(temperature)
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
        self.rblapsum_sf_value_grad = rblapsum_sf_value_grad
        self.rblapsum_rho_random_perm_prob_grad = float(
            rblapsum_rho_random_perm_prob_grad)
        # see ActivationBottleneckConfig.rblapsum_surrogate_scope; the
        # "update*" routing is the model's (TransformerLM._code_residual_stack),
        # the gate itself only restricts the members
        if rblapsum_surrogate_scope not in SURROGATE_SCOPES:
            raise ValueError(
                f"unknown rblapsum_surrogate_scope: {rblapsum_surrogate_scope!r}")
        self.rblapsum_surrogate_scope = rblapsum_surrogate_scope
        # carry scopes: set by TransformerLM._code_residual_stack for the
        # duration of the gate call (the chain of the current forward and this
        # gate's index in the stack); plain attributes, not state
        self._carry_chain = None
        self._carry_index = -1
        # see ActivationBottleneckConfig.rblapsum_carry_decay
        if not 0.0 <= float(rblapsum_carry_decay) <= 1.0:
            raise ValueError("rblapsum_carry_decay must be in [0, 1]")
        self.rblapsum_carry_decay = float(rblapsum_carry_decay)
        # see ActivationBottleneckConfig.rblapsum_relative_temperature: the
        # kernel width is temperature * b per row
        if rblapsum_relative_temperature and surrogate_mode != "rblapsum":
            raise ValueError(
                "rblapsum_relative_temperature applies to surrogate_mode='rblapsum' only")
        self.rblapsum_relative_temperature = bool(rblapsum_relative_temperature)
        # see ActivationBottleneckConfig.rblapsum_kernel_width / _support_strength
        if rblapsum_kernel_width not in ("fixed", "relative_b", "relative_span"):
            raise ValueError(f"unknown rblapsum_kernel_width: {rblapsum_kernel_width!r}")
        if self.rblapsum_relative_temperature:
            rblapsum_kernel_width = "relative_b"
        self.rblapsum_kernel_width = rblapsum_kernel_width
        self.rblapsum_support_strength = (None if rblapsum_support_strength is None
                                          else float(rblapsum_support_strength))
        if (self.rblapsum_kernel_width != "fixed" or self.rblapsum_support_strength is not None) \
                and surrogate_mode != "rblapsum":
            raise ValueError("rblapsum_kernel_width / rblapsum_support_strength apply to "
                             "surrogate_mode='rblapsum' only")
        # see ActivationBottleneckConfig.rblapsum_radial_project
        self.rblapsum_radial_project = bool(rblapsum_radial_project)
        if self.rblapsum_radial_project and surrogate_mode != "rblapsum":
            raise ValueError("rblapsum_radial_project applies to surrogate_mode='rblapsum' only")
        # see ActivationBottleneckConfig.rblapsum_center_tokens
        if rblapsum_center_tokens and surrogate_mode != "rblapsum":
            raise ValueError(
                "rblapsum_center_tokens applies to surrogate_mode='rblapsum' only")
        self.rblapsum_center_tokens = bool(rblapsum_center_tokens)
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
            # the (K+1)-st score is the rank boundary, so J>=1 (already required
            # by validate_gate_shapes for every non-hard mode) guarantees it
            if self.j < 1:
                raise ValueError("surrogate_mode='rblapsum' needs j >= 1 (the K+1 boundary)")
        self.barrier_solver_tol = float(barrier_solver_tol)
        self.solver_dtype = _DTYPES[solver_dtype]
        self.log_diagnostics = bool(log_diagnostics)
        self.hard_inference = bool(hard_inference)

        # ---- laplace_policy (see .laplace_policy) ---------------------------- #
        # The configured `temperature` is tau_0; `policy_tau` is the RUNTIME
        # scheduled value the controller sets per step (a plain attribute, so
        # the dumped config keeps the original schedule).  `_policy_settings`
        # is the temporary per-forward request (sample override, collector,
        # generator) that TransformerLM.forward installs and restores; None
        # means "sample iff training, record nothing".
        if policy_temperature_mode not in TEMPERATURE_MODES:
            raise ValueError(f"unknown policy_temperature_mode: {policy_temperature_mode!r}")
        if policy_support_scale_mode not in SUPPORT_SCALE_MODES:
            raise ValueError(
                f"unknown policy_support_scale_mode: {policy_support_scale_mode!r}")
        self.policy_temperature_mode = policy_temperature_mode
        self.policy_min_temperature = float(policy_min_temperature)
        self.policy_center_scores = bool(policy_center_scores)
        self.policy_support_scale = float(policy_support_scale)
        self.policy_support_scale_mode = policy_support_scale_mode
        self.policy_support_temperature_ref = float(policy_support_temperature_ref)
        self.policy_support_scale_max = (None if policy_support_scale_max is None
                                         else float(policy_support_scale_max))
        self.policy_tau = float(temperature)
        self._policy_settings = None
        if policy_estimator not in ESTIMATORS:
            raise ValueError(f"unknown policy_estimator: {policy_estimator!r}")
        if policy_rb_scope not in RB_SCOPES:
            raise ValueError(f"unknown policy_rb_scope: {policy_rb_scope!r}")
        if policy_width_gradient not in WIDTH_GRADIENTS:
            raise ValueError(f"unknown policy_width_gradient: {policy_width_gradient!r}")
        self.policy_estimator = policy_estimator
        self.policy_rb_scope = policy_rb_scope
        self.policy_width_gradient = policy_width_gradient
        if int(policy_rb_samples) < 1:
            raise ValueError("policy_rb_samples must be >= 1")
        self.policy_rb_samples = int(policy_rb_samples)
        # j = 0 or k = n_features: no exchange is possible, so the gate is the
        # ordinary hard Top-K with no noise and no density term
        self.policy_fixed = self.j == 0 or self.trivial
        if surrogate_mode == "laplace_policy":
            if selection_mode not in ("topk", "abs_topk"):
                raise ValueError("surrogate_mode='laplace_policy' supports selection_mode "
                                 "'topk' and 'abs_topk' only")
            if not self.policy_min_temperature > 0:
                raise ValueError("policy_min_temperature must be positive")
            if not self.policy_support_scale >= 0:
                raise ValueError("policy_support_scale must be >= 0")
            if not self.policy_support_temperature_ref > 0:
                raise ValueError("policy_support_temperature_ref must be positive")
            if not self.policy_fixed:
                if policy_temperature_mode == "relative_b" and selection_mode != "abs_topk":
                    raise ValueError("policy_temperature_mode='relative_b' needs abs_topk "
                                     "(a nonnegative rank boundary)")
                if policy_temperature_mode == "relative_span" and self.j < 2:
                    raise ValueError("policy_temperature_mode='relative_span' needs j >= 2 "
                                     "(for j=1 the span s_(K+1) - s_(K+J) is zero)")

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
    def solve(self, candidates: torch.Tensor):
        """``(b, t, diag)`` for a batch of **detached, sorted** candidate rows.

        The temperature is the constant ``self.temperature`` (one shared field
        for the lapsum and rblapsum modes since the 2026-09-28 cleanup); the
        barrier ``b`` follows in closed form so the soft budget is exactly K.
        Matches the pre-cleanup ``lapsum_fixed`` mode with
        ``temperature_scale_mode="absolute"`` bit for bit: ``t`` is built as
        the same per-row tensor product it always was.
        """
        t = self.temperature * torch.ones_like(candidates[..., 0])
        return lapsum_barrier_sorted(candidates, self.k, t), t, {}

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

        if self.surrogate_mode in ("rblapsum", "rblapsum_sf", "soft_ste"):
            return self._rblapsum(scores, value)
        if self.surrogate_mode == "laplace_policy":
            return self._laplace_policy(scores, value)

        cand_scores, cand_idx = torch.topk(
            scores, self.m, dim=-1, largest=True, sorted=True
        )
        if self.stochastic_width != "none" and self.training:
            # per-token width K' in [K, K+J]: the sorted pool keeps ranks < K'
            kprime = self._sample_width(scores)
            keep = (torch.arange(self.m, device=scores.device) < kprime).to(scores.dtype)
            hard_mask = torch.zeros_like(scores).scatter(-1, cand_idx, keep)
            if self.log_diagnostics:
                self._forward_diag["active_count"] = kprime.to(torch.float32).mean().detach()
        else:
            hard_mask = torch.zeros_like(scores).scatter(-1, cand_idx[..., : self.k], 1.0)
        if self.log_diagnostics and self.training:
            self._record_usage(hard_mask)

        if not self.surrogate_active():
            if self.log_diagnostics and self.training:
                # Keep the hard baseline comparable with the surrogate runs: the
                # forward-side statistics are all still meaningful, and the
                # gradient on the J candidates is exactly zero by construction
                # rather than merely absent.
                width = self._forward_diag.get("active_count")
                self._record_hard(cand_scores.to(self.solver_dtype).detach())
                if width is not None:
                    self._forward_diag["active_count"] = width
            if self.value_shift != "none":
                return self._shifted(value, cand_idx[..., : self.k])
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
        p = lapsum_probs(cand - centre, b - centre.squeeze(-1), t, self.k, sink)
        p_full = (
            torch.zeros_like(scores, dtype=p.dtype)
            .scatter(-1, cand_idx, p)
            .to(value.dtype)
        )
        mask = hard_mask + (p_full - p_full.detach())

        if self.log_diagnostics:
            self._record(detached, b, t, p.detach(), solver_diag)
        return value * mask

    # ---- laplace_policy -------------------------------------------------------- #
    def policy_sampling(self) -> bool:
        """Whether this forward samples the support (laplace_policy).

        The per-forward settings' ``sample`` wins when given; otherwise the
        gate samples iff it is in ``train()`` mode.  Deliberately independent
        of ``torch.is_grad_enabled()``: a stochastic validation runs in
        ``eval()`` under ``no_grad()`` with ``sample=True``.
        """
        if self.policy_fixed:
            return False
        settings = self._policy_settings
        if settings is not None and settings.sample is not None:
            return bool(settings.sample)
        return bool(self.training)

    def _laplace_policy(self, scores: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Sampled exactly-K support over the clean Top(K+J) pool (laplace_policy).

        Ordinary gather / mask / scatter autograd for the values (the sampled
        hard-mask gradient), differentiable gathered scores for the density
        term, which goes to the forward's collector when one is installed and
        grad is enabled.  Nothing here calls the RBLapSum machinery.
        """
        settings = self._policy_settings
        cand_scores, cand_idx = torch.topk(scores, self.m, dim=-1, largest=True, sorted=True)
        if not self.policy_sampling():
            # clean deterministic Top-K: evaluation, generation, the probes,
            # and the fixed geometries (j = 0, k = n)
            hard_mask = torch.zeros_like(scores).scatter(-1, cand_idx[..., : self.k], 1.0)
            if self.log_diagnostics and self.training:
                self._record_usage(hard_mask)
                self._record_hard(cand_scores.to(self.solver_dtype).detach())
            return value * hard_mask

        k, j = self.k, self.j
        # scores in >= float32 WITH their autograd path (the pool indices are
        # discrete; the gathered scores are what the density differentiates)
        score_c = at_least_float32(cand_scores)
        value_c = torch.gather(value, -1, cand_idx)
        with torch.no_grad():
            t_row, floor_binding, raw_scale = effective_width(
                self.policy_temperature_mode, self.policy_tau, score_c, k, j,
                self.policy_min_temperature)
        if self.policy_width_gradient == "project" and self.policy_temperature_mode != "absolute":
            # the selection path's score gradient loses its component along the
            # centred scores (ProjectScaleFree): no push on the scale the width follows
            centred = (score_c - score_c.mean(-1, keepdim=True)).detach()
            score_c = ProjectScaleFree.apply(score_c, centred)
        u = score_c - score_c.mean(-1, keepdim=True) if self.policy_center_scores else score_c
        t_score = t_row  # the width in score units, for the diagnostics
        if self.policy_width_gradient == "through" and self.policy_temperature_mode != "absolute":
            # Exactly scale-free selection: the noise of width tau acts on the
            # scores divided by the activation scale a (differentiable), the
            # same forward as T = tau * a, but the gradient now includes the
            # derivative through a, so a common rescaling of the scores has no
            # selection gradient (the frozen-scale rule pushes it).
            if self.policy_temperature_mode == "relative_b":
                a = score_c[..., k:k + 1]
            else:
                a = score_c[..., k:k + 1] - score_c[..., k + j - 1:k + j]
            a = a.clamp_min(self.policy_min_temperature)
            u = u / a
            t_row = torch.full_like(t_row, max(self.policy_min_temperature, self.policy_tau))
        generator = settings.generator if settings is not None else None
        eps = sample_laplace(u.shape, t_row, generator=generator, dtype=u.dtype,
                             device=u.device)
        # the realized noisy scores: detached, and used BOTH for the ranking
        # and for the density (all K+J coordinates, selected or not)
        r = (u.detach() + eps).detach()
        selected = torch.topk(r, k, dim=-1, largest=True, sorted=False).indices
        mask_c = torch.zeros_like(r).scatter(-1, selected, 1.0)
        collector = settings.collector if settings is not None else None
        gamma = support_multiplier(self.policy_support_scale_mode, self.policy_support_scale,
                                   t_score, self.policy_tau, self.policy_support_temperature_ref,
                                   self.policy_support_scale_max)
        rao_blackwell = (self.policy_estimator == "rao_blackwell" and torch.is_grad_enabled()
                         and collector is not None)
        if rao_blackwell:
            # the selection gradient comes from the Rao-Blackwellized backward of
            # this Function; the density record below is kept (diagnostics, the
            # trainer's contract) with zero weight
            rb_seed = 0
            if self.policy_rb_samples > 1:
                # the backward's further draws: a seed taken from the training
                # generator, so they are fresh every forward and reproducible
                rb_seed = int(torch.randint(0, 2 ** 62, (1,), generator=generator,
                                            device=generator.device if generator is not None
                                            else u.device).item())
            y_c = RaoBlackwellSelect.apply(u, value_c, mask_c, r, t_row, gamma,
                                           self.policy_rb_scope == "first_order", self,
                                           self.policy_rb_samples, rb_seed)
        else:
            y_c = value_c * mask_c.to(value_c.dtype)
        y = torch.zeros_like(value).scatter(-1, cand_idx, y_c)
        diag: Dict[str, torch.Tensor] = {}
        if self.log_diagnostics:
            with torch.no_grad():
                diag = self._policy_diag(score_c.detach(), r, u.detach(), t_row, floor_binding,
                                         raw_scale, gamma, mask_c, y,
                                         laplace_log_density(r, u.detach(), t_row),
                                         t_score=t_score)
            if self.training:
                mask_full = torch.zeros_like(scores).scatter(
                    -1, cand_idx, mask_c.to(scores.dtype))
                self._record_usage(mask_full)
                self._forward_diag = diag
        if collector is not None and torch.is_grad_enabled():
            log_prob = laplace_log_density(r, u, t_row)  # shape = scores.shape[:-1]
            weight = gamma.squeeze(-1) * (0.0 if rao_blackwell else 1.0)
            collector.add(PolicyRecord(
                gate=self, log_prob=weight * log_prob, diag=diag,
                r=r if collector.keep_samples else None,
                cand_idx=cand_idx.detach() if collector.keep_samples else None))
        return y

    @torch.no_grad()
    def _policy_diag(self, score_c, r, u, t_row, floor_binding, raw_scale, gamma, mask_c, y,
                     log_prob, t_score=None) -> Dict[str, torch.Tensor]:
        """Detached per-gate statistics of one sampled forward (laplace_policy).

        Everything is computed from what was actually sampled -- no CDF
        probabilities borrowed from RBLapSum -- and the location-gradient
        statistics use the analytic row formula ``sign(r - u) / T`` (projected
        onto the zero-sum subspace when the scores are centred), so no second
        backward is run for them.  Score-space signals, not parameter gradients.
        """
        k = self.k
        h = torch.sign(r - u) / t_row                       # raw location gradient
        h_used = h - h.mean(-1, keepdim=True) if self.policy_center_scores else h
        g_used = gamma * h_used                            # what the loss actually scales
        in_clean_topk = mask_c[..., :k]                     # sampled members at clean rank < K
        exchanged = k - in_clean_topk.sum(-1)
        span = score_c[..., k:k + 1] - score_c[..., k + self.j - 1:k + self.j]
        finite = (torch.isfinite(score_c).all() & torch.isfinite(t_row).all()
                  & torch.isfinite(log_prob).all())
        d = {
            "active_count": mask_c.sum(-1).mean(),
            "output_nonzero_count": (y != 0).sum(-1).to(torch.float32).mean(),
            "candidate_count": torch.tensor(float(self.m), device=r.device),
            "policy_tau": torch.tensor(float(self.policy_tau), device=r.device),
            "policy_t_mean": (t_row if t_score is None else t_score).mean(),
            "policy_t_min": (t_row if t_score is None else t_score).min(),
            "policy_t_max": (t_row if t_score is None else t_score).max(),
            "policy_t_floor_frac": floor_binding.to(torch.float32).mean(),
            "policy_scale_mean": raw_scale.mean(),
            "policy_span_zero_frac": (span == 0).to(torch.float32).mean(),
            "policy_gamma_mean": gamma.mean(), "policy_gamma_min": gamma.min(),
            "policy_gamma_max": gamma.max(),
            "policy_exchange_frac": exchanged.mean() / k,
            "policy_overlap": in_clean_topk.sum(-1).mean() / k,
            "score_gap": (score_c[..., k - 1] - score_c[..., k]).mean(),
            "score_span": (score_c[..., k - 1] - score_c[..., -1]).mean(),
            "policy_pool_span": span.mean(),
            "policy_logp_mean": log_prob.mean(),
            "policy_collapsed_frac": (r == u).to(torch.float32).mean(),
            "policy_score_grad_rms_raw": h.pow(2).mean().sqrt(),
            "policy_score_grad_rms": g_used.pow(2).mean().sqrt(),
            "policy_zero_sum_residual": h_used.sum(-1).abs().mean(),
            "policy_nonfinite": (~finite).to(torch.float32),
        }
        return {key: val.detach() for key, val in d.items()}

    # ---- stochastic support width --------------------------------------------- #
    def _sample_width(self, scores: torch.Tensor) -> torch.Tensor:
        """Per-token K' in [K, K+J] (see ActivationBottleneckConfig.stochastic_width).

        Returns an integer tensor of shape ``scores.shape[:-1] + (1,)``.
        """
        shape = scores.shape[:-1] + (1,)
        dev = scores.device
        k, j, q = self.k, self.j, self.stochastic_width_param
        if self.stochastic_width == "uniform":
            extra = torch.randint(0, j + 1, shape, device=dev)
        elif self.stochastic_width == "two_point":
            extra = (torch.rand(shape, device=dev) < q).long() * j
        else:  # geometric with mean q * J, capped at J
            mean = max(q * j, 1e-6)
            u = torch.rand(shape, device=dev).clamp_min(1e-12)
            extra = torch.floor(torch.log(u) / math.log(mean / (1.0 + mean))).long().clamp_(0, j)
        return k + extra

    # ---- value shift (hard forward) -------------------------------------------- #
    def _shifted(self, value: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """``sign(z) * max(|z| - delta, 0)`` on the TopK support, zero elsewhere.

        ``delta`` is per token: ``value_shift_lambda * RMS(z)`` ("fixed"), or the
        root of ``mean_S (|z| - delta)^2 = mean_all z^2`` ("energy"), i.e.
        ``delta = m1 - sqrt(m1^2 - m2 + sigma^2)`` with ``m1``, ``m2`` the mean
        and mean square of the kept magnitudes and ``sigma^2`` the mean square
        of all N.  ``m2 >= sigma^2`` (the kept are the largest) makes the root
        at most ``m1``, so ``delta >= 0``; when the kept magnitudes are so spread
        that no shift gets down to ``sigma^2``, the discriminant is floored and
        ``delta ~ m1``, the shift of least energy.  Gradients go through
        everything (delta included): this is the forward.

        ``idx`` holds the K kept positions, so everything but ``sigma^2`` works
        on K gathered values rather than N (in at least float32; bf16/fp16 are
        promoted, float64 stays float64), and the result is scattered back.
        """
        ft = torch.promote_types(value.dtype, torch.float32)
        vk = value.gather(-1, idx).to(ft)
        sig2 = torch.linalg.vector_norm(value, dim=-1, keepdim=True,
                                        dtype=ft).pow(2) / value.shape[-1]
        a = vk.abs()
        if self.value_shift == "fixed":
            delta = self.value_shift_lambda * sig2.sqrt()
        else:
            m1 = a.mean(-1, keepdim=True)
            m2 = (a * a).mean(-1, keepdim=True)
            # floored away from 0: sqrt's slope is unbounded there
            disc = (m1 * m1 - m2 + sig2).clamp_min(1e-6 * sig2.detach()
                                                   + torch.finfo(ft).tiny)
            delta = m1 - disc.sqrt()
        yk = vk.sign() * torch.relu(a - delta)
        if self.log_diagnostics and self.training:
            with torch.no_grad():
                sig = sig2.sqrt().clamp_min(torch.finfo(ft).tiny)
                self._forward_diag["shift_rel"] = (delta / sig).mean()
                # kept by TopK but below delta, so output as zero
                self._forward_diag["shift_clamped_frac"] = (a <= delta).float().mean()
        return torch.zeros_like(value).scatter(-1, idx, yk.to(value.dtype))

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

        t = float(self.temperature)
        sink = self._grad_sink if self.log_diagnostics and self.training else None
        # rblapsum_sf: the probabilities are IN the forward, y = z * p over the
        # whole pool.  In eval, hard_inference=True (the default) falls back to
        # the hard Top-K forward below, so the base validation metric measures
        # the hardened model and stays name-comparable with the other regimes;
        # flip hard_inference off around a second eval pass for the soft CE.
        soft = (self.surrogate_mode == "rblapsum_sf"
                and (self.training or not self.hard_inference))
        if self.surrogate_mode == "soft_ste":
            # hard forward; backward M + (1 - M) p over the pool: a candidate
            # receives the gradient it would receive if present, times its
            # kernel probability.  The soft part is numerically zero in the
            # forward and carries only its gradient.
            p_c = laplace_cdf((score_c - b) / t)
            soft_part = value_c * (p_c * (1.0 - active_c)).to(value_c.dtype)
            y_c = value_c * active_c + (soft_part - soft_part.detach())
        elif soft:
            p_c = laplace_cdf((score_c - b) / t)
            y_c = rblapsum_sf_gate(value_c, p_c, active_c, score_c, sign_c,
                                   b, t, self.rblapsum_boundary_grad_mode,
                                   self.k, cap_active, sink,
                                   supp_scale=float(self.rblapsum_support_scale),
                                   value_grad=self.rblapsum_sf_value_grad)
        elif carry_scope(self.rblapsum_surrogate_scope) and torch.is_grad_enabled():
            if self._carry_chain is None:
                raise RuntimeError(
                    f"rblapsum_surrogate_scope={self.rblapsum_surrogate_scope!r} needs "
                    "the code-residual stack to split the gate's output into its carry "
                    "and read copies (TransformerLM._code_residual_stack)")
            y_c = rblapsum_carry_gate(value_c, active_c, score_c, sign_c, b, t,
                                      self.rblapsum_boundary_grad_mode, self.k,
                                      cap_active, sink,
                                      float(self.rblapsum_support_scale),
                                      SURROGATE_SCOPES[self.rblapsum_surrogate_scope],
                                      self.rblapsum_relative_temperature, cand_idx,
                                      self.n_features, self._carry_chain,
                                      self._carry_index, self.rblapsum_surrogate_scope,
                                      self.rblapsum_carry_decay)
        else:
            # per-row kernel width and support scale (temperature mode, strength);
            # the legacy relative flag is handled here too, so relative_t is
            # never set on the Function
            t_row = kernel_width(self.rblapsum_kernel_width, t, score_c, b, self.k, self.j)
            scale = float(self.rblapsum_support_scale)
            if self.rblapsum_support_strength is not None:
                scale = strength_scale(self.rblapsum_support_strength, t_row, b)
            y_c = rblapsum_gate(value_c, active_c, score_c, sign_c, b, t_row,
                                self.rblapsum_boundary_grad_mode, self.k, cap_active, sink,
                                supp_scale=scale,
                                perm_rho=(self.rblapsum_rho_random_perm_prob_grad
                                          if self.training else 0.0),
                                members=SURROGATE_SCOPES[self.rblapsum_surrogate_scope],
                                relative_t=False,
                                cand_idx=(cand_idx if self.rblapsum_center_tokens
                                          and self.training else None),
                                n_features=self.n_features,
                                first_order=self.rblapsum_surrogate_scope.startswith(
                                    "first_order"),
                                radial_project=self.rblapsum_radial_project)
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
                if self.rblapsum_kernel_width != "fixed":
                    self._forward_diag["rb_temperature"] = (
                        kernel_width(self.rblapsum_kernel_width, t, score_c, b,
                                     self.k, self.j).mean().detach())
        return y

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
        here would show that -- the loss and the budget both look healthy while
        it happens -- so it is tracked over an EMA window rather than a
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
        first_inactive = cand[..., k]
        barrier_gap = first_inactive - b
        d = {
            "temperature": t.mean(),
            "barrier": b.mean(),
            "barrier_gap": barrier_gap.mean(),
            "frac_above_barrier": (first_inactive > b).float().mean(),
            "budget_residual": budget.abs().mean(),
            "barrier_failures": (budget.abs() > self._budget_tolerance(cand, t)).float().mean(),
            "score_gap": gap.mean(),
            "score_span": span.mean(),
        }
        self._forward_diag = {key: value.detach() for key, value in d.items()}

    def _budget_tolerance(self, cand: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """What ``|sum p - K|`` can actually reach in the solver dtype.

        Scores arrive already rounded, so ``(r_i - b)`` carries an absolute
        error of about ``eps * |r|``, which dividing by ``t`` amplifies into the
        exponent.  Only the candidates within a few ``t`` of the barrier have an
        appreciable ``dF/dz``, so only an effective few of them contribute, at
        up to ``1/4`` each.  Comparing against a flat ``barrier_solver_tol * K`` would
        report a failure on every batch of offset activations even though the
        solver is exact -- the limit is the input representation.
        """
        eps = torch.finfo(cand.dtype).eps
        floor = 0.25 * self.m * eps * cand.abs().amax(-1) / t.clamp_min(
            torch.finfo(cand.dtype).tiny
        )
        return floor.clamp_min(self.barrier_solver_tol * max(self.k, 1))

    def extra_repr(self) -> str:
        shift = ""
        if self.value_shift == "fixed":
            shift = f", value_shift=fixed({self.value_shift_lambda:.4g})"
        elif self.value_shift == "energy":
            shift = ", value_shift=energy"
        policy = ""
        if self.surrogate_mode == "laplace_policy":
            policy = (f", policy=({self.policy_temperature_mode}, "
                      f"center={self.policy_center_scores}, "
                      f"gamma={self.policy_support_scale:g}/{self.policy_support_scale_mode})")
        return (
            f"n_features={self.n_features}, k={self.k}, j={self.j}, "
            f"mode={self.selection_mode}, surrogate={self.surrogate_mode}, "
            f"temperature={self.temperature:g}{shift}{policy}"
        )


def validate_gate_shapes(
    n_features: int,
    k: int,
    j: int,
    surrogate_mode: str = "hard",
) -> None:
    """Shape rules, enforced up front rather than as runtime NaNs.

    A hard mask never solves a barrier, so two geometries that are degenerate
    for the surrogates are meaningful there: ``j = 0`` (no candidates beyond
    the support) and ``k = n_features`` (every feature active -- a *trivial*
    bottleneck isolating the projection pair's parameter cost).  laplace_policy
    shares the hard rules: with no possible exchange it bypasses noise and
    density and IS the hard gate.
    """
    if surrogate_mode in ("hard", "laplace_policy"):
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
