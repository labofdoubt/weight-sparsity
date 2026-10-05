"""The dense-in / sparse-gate / dense-out bottleneck placed before an MLP."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model import RMSNorm
from .gate import AdaptiveLapSumTopKGate

INIT_MODES = (
    "default",
    "sqrt_k",
    "sqrt_k_selection_corrected",
    "unit_norm_dictionary",
)
#: renamed options, mapped so an old config fails with a useful message
_RENAMED_INIT_MODES = {"unit_scale_output": "sqrt_k"}


def selection_gain(k: int, n_features: int) -> float:
    """``E[|z| | z survives TopK]`` for standard-normal pre-activations.

    TopK keeps the ``k`` largest of ``n`` by magnitude, so the survivors are
    tail order statistics, not typical draws: at k=32 of 2048 their mean
    magnitude is ~2.75, against ~0.80 for an unselected coefficient.  Any
    decoder scale derived from ``k`` alone therefore overshoots by this factor.

    Closed form via the inverse Mills ratio, ``phi(t) / (1 - Phi(t))`` at the
    threshold ``t`` with ``P(|Z| > t) = k/n``.  Checked against simulation to
    within 0.2% for k/n between 1/64 and 1/2, so no sampling is needed.
    """
    p = min(1.0, max(1e-12, k / max(1, n_features)))
    # t = Phi^-1(1 - p/2), written through erfinv to avoid a scipy dependency
    t = math.sqrt(2.0) * float(
        torch.erfinv(torch.tensor(1.0 - p, dtype=torch.float64))
    )
    return 2.0 * math.exp(-0.5 * t * t) / (math.sqrt(2.0 * math.pi) * p)


def selection_energy_gain(k: int, n_features: int) -> float:
    """``s^2 = E[z^2 | z survives abs-TopK]`` for standard-normal coefficients.

    The ratio of the kept coefficients' mean square to that of a typical one:
    ``1 + 2 t phi(t) / rho`` with ``P(|Z| > t) = rho = k/n``.  It is the factor
    by which a stream bottleneck's forward energy gain exceeds its backward one
    at initialization (4.02 at k/n = 1/8, 8.90 at 1/128), and no rescaling of
    the bottleneck changes it.
    """
    p = min(1.0, max(1e-12, k / max(1, n_features)))
    if p >= 1.0:
        return 1.0
    t = math.sqrt(2.0) * float(
        torch.erfinv(torch.tensor(1.0 - p, dtype=torch.float64))
    )
    phi = math.exp(-0.5 * t * t) / math.sqrt(2.0 * math.pi)
    return 1.0 + 2.0 * t * phi / p


def critical_shift(k: int, n_features: int) -> float:
    """The ``lambda`` at which shifted abs-TopK carries no selection gain.

    Solves ``E[(|Z| - lambda)^2 | |Z| > t] = 1`` for standard-normal ``Z``:
    with ``m1 = E[|Z| | kept] = selection_gain(k, n)`` and ``m2 =
    selection_energy_gain(k, n)`` it is ``m1 - sqrt(m1^2 - m2 + 1)``.  1.044 at
    k/n = 1/8, 0.400 at 1/2, 2.011 at 1/128; 0 when nothing is dropped.
    """
    if k >= n_features:
        return 0.0
    m1 = selection_gain(k, n_features)
    m2 = selection_energy_gain(k, n_features)
    return m1 - math.sqrt(max(0.0, m1 * m1 - m2 + 1.0))


def effective_backward_support(cfg) -> float:
    """How many features per token carry gradient out of the gate, ``K_eff``.

    The forward support is always ``K``.  Under a surrogate the ``J`` extra
    candidates also receive gradient, and this returns the simplest estimate
    that accounts for them, ``K + J`` -- which assumes their gradients have
    comparable RMS to the active ones.  That is the approximation to replace
    when a measured version is wanted (e.g. ``K + sum_j lambda_j^2`` over the
    kernel weights); every caller goes through this function, so nothing else
    has to change.
    """
    k, j = int(cfg.k), int(cfg.j)
    if cfg.surrogate_mode == "hard":
        return float(k)  # j is inert in the backward, diagnostics only
    return float(k + j)


class TiedDecoder(nn.Module):
    """Decoder whose weight *is* the encoder's, transposed.

    Implemented as a view rather than a copy, so the two never drift and the
    tie costs no parameters.  ``.weight`` still reads as the ``(d_model,
    n_features)`` decoder matrix, which is what the calibration pass and the
    interpretability tooling expect.
    """

    def __init__(self, encoder: nn.Linear, bias: bool = True):
        super().__init__()
        self.encoder = [encoder]  # in a list: not a submodule, so not double-counted
        self.bias = nn.Parameter(torch.zeros(encoder.in_features)) if bias else None

    @property
    def weight(self) -> torch.Tensor:
        return self.encoder[0].weight.t()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class SparseTopKBottleneck(nn.Module):
    """``x -> W_in -> TopK/AbsTopK -> W_out -> (original MLP)``.

    All projections are ordinary dense ``nn.Linear`` layers trained by the
    model's ordinary objective -- there is no reconstruction loss, no weight
    mask and no pruning anywhere in this module.  ``in_proj``/``out_proj``
    rather than down/up because ``n_features`` may be larger or smaller than
    ``d_model``.

    ``selection_mode="gated_topk"`` adds a third projection: the support is
    ranked by an independent score branch ``s = W_s x + b_s`` while ``in_proj``
    supplies the value ``v``.  That splits the two roles the single projection
    otherwise plays, at the cost of ``d_model * n_features`` more parameters
    per layer.  No nonlinearity is applied to ``v``, so values stay signed.

    ``share_from`` makes this module reuse another one's projections instead of
    creating its own: ``in_proj``, ``score_proj`` and ``out_proj`` become the
    *same* ``nn.Linear`` objects, nothing is re-initialized, and only the gate
    (with its usage buffers), the optional post-norm and ``decoder_scale`` are
    this module's own.  The controller passes the first bottleneck it built
    when ``cfg.share_projections`` is set.
    """

    def __init__(self, d_model: int, cfg, bias: bool = True,
                 post_norm: bool = False, norm_eps: float = 1e-6,
                 share_from: Optional["SparseTopKBottleneck"] = None):
        super().__init__()
        self.d_model = int(d_model)
        self.n_features = int(cfg.n_features)
        self.gated = cfg.selection_mode == "gated_topk"
        self.tied = bool(getattr(cfg, "tie_encoder_decoder", False))
        # The config flag, true for every bottleneck built under it -- the one
        # the others adopt from included; `share_from` is the mechanism.
        self.shared = bool(getattr(cfg, "share_projections", False))
        if share_from is not None:
            self._adopt_projections(share_from, bias)
        else:
            # in_proj is the value branch; score_proj (gated_topk only) ranks.
            self.in_proj = nn.Linear(self.d_model, self.n_features, bias=bias)
            self.score_proj = (
                nn.Linear(self.d_model, self.n_features, bias=bias) if self.gated else None
            )
        self.gate = AdaptiveLapSumTopKGate(
            n_features=self.n_features,
            k=cfg.k,
            j=cfg.j,
            selection_mode=cfg.selection_mode,
            surrogate_mode=cfg.surrogate_mode,
            rblapsum_boundary_grad_mode=cfg.rblapsum_boundary_grad_mode,
            rblapsum_boundary_floor=cfg.rblapsum_boundary_floor,
            rblapsum_support_scale=getattr(cfg, 'rblapsum_support_scale', 1.0),
            temperature=cfg.temperature,
            rblapsum_sf_value_grad=getattr(cfg, "rblapsum_sf_value_grad", "pool"),
            rblapsum_rho_random_perm_prob_grad=getattr(
                cfg, "rblapsum_rho_random_perm_prob_grad", 0.0),
            rblapsum_surrogate_scope=getattr(cfg, "rblapsum_surrogate_scope", "pool"),
            rblapsum_relative_temperature=getattr(
                cfg, "rblapsum_relative_temperature", False),
            rblapsum_center_tokens=getattr(cfg, "rblapsum_center_tokens", False),
            rblapsum_carry_decay=getattr(cfg, "rblapsum_carry_decay", 1.0),
            barrier_solver_tol=cfg.barrier_solver_tol,
            solver_dtype=cfg.solver_dtype,
            log_diagnostics=cfg.log_diagnostics,
            hard_inference=cfg.hard_inference,
            value_shift=getattr(cfg, "value_shift", "none"),
            value_shift_lambda=(
                getattr(cfg, "value_shift_lambda", None)
                if getattr(cfg, "value_shift_lambda", None) is not None
                else critical_shift(int(cfg.k), self.n_features)),
        )
        self.init_mode = getattr(cfg, "init_mode", "default")
        if self.init_mode in _RENAMED_INIT_MODES:
            raise ValueError(
                f"init_mode={self.init_mode!r} was renamed to "
                f"{_RENAMED_INIT_MODES[self.init_mode]!r}"
            )
        if self.init_mode not in INIT_MODES:
            raise ValueError(
                f"unknown bottleneck init_mode: {self.init_mode!r} ({' | '.join(INIT_MODES)})"
            )
        if self.tied and self.init_mode != "unit_norm_dictionary":
            raise ValueError(
                "tie_encoder_decoder requires init_mode='unit_norm_dictionary': "
                "tying only makes sense when both sides share a scale"
            )
        if share_from is None:  # an adopter took the source's, initialized once
            if self.tied:
                self.out_proj = TiedDecoder(self.in_proj, bias=bias)
            else:
                self.out_proj = nn.Linear(self.n_features, self.d_model, bias=bias)
            self._init_projections(int(cfg.k))
        # An optional RMSNorm on this bottleneck's output.  It lives inside the
        # module so the gate hooks and `_PLACEMENT_ATTR` lookups are unchanged,
        # and Identity costs nothing when the option is off.
        self.post_norm = (RMSNorm(self.d_model, norm_eps) if post_norm
                          else nn.Identity())
        # Fixed global decoder scale g_D, part of the parameterization
        #     W_D = g_D * diag(g_row) W_hat_D diag(g_col).
        # A plain float (like the model's embed_scale), so state_dicts are
        # unchanged and a rebuild from the config restores it; md_init_ sets it
        # from model.bottleneck_decoder_scale.  1.0 is the identity and takes
        # the untouched forward path below.
        self.decoder_scale = 1.0

    def _adopt_projections(self, source: "SparseTopKBottleneck", bias: bool) -> None:
        """Take ``source``'s projections as this module's own (share_projections).

        Assigning a module that another module already holds registers it here
        too, so ``parameters()``, ``state_dict()`` and ``.to(device)`` all see
        the shared objects; ``nn.Module`` deduplicates the parameters and
        ``.to`` is idempotent on an already-moved tensor.
        """
        want = (self.d_model, self.n_features, self.gated, self.tied, bool(bias))
        have = (source.d_model, source.n_features, source.gated, source.tied,
                source.in_proj.bias is not None)
        if want != have:
            raise ValueError(
                "share_from: cannot share projections between bottlenecks of "
                "different geometry -- (d_model, n_features, gated, tied, bias) "
                f"is {want} here but {have} in the source"
            )
        self.in_proj = source.in_proj
        self.score_proj = source.score_proj
        self.out_proj = source.out_proj

    def _init_projections(self, k: int) -> None:
        """Re-initialize the projections; ``default`` leaves PyTorch's alone.

        ``sqrt_k``  encoder std 1/sqrt(d_model), decoder std 1/sqrt(k).
            The decoder's fan-in is ``n_features``, but only ``k`` of those
            coefficients are ever non-zero, so scaling by ``n_features`` -- as
            PyTorch's default does -- under-scales the output by sqrt(n/k).
            Correcting the fan-in to ``k`` overshoots in the other direction,
            because the surviving coefficients are the largest ones.

        ``sqrt_k_selection_corrected``  the same, divided by the mean magnitude
            of a surviving coefficient (see ``selection_gain``).  This is the
            one that actually lands at unit output scale.

        ``unit_norm_dictionary``  both std 1/sqrt(d_model), so every decoder
            column has expected unit norm: a dictionary of unit atoms, which is
            the usual convention for a sparse code and what makes the decoder
            directions comparable to one another.

        Biases are zeroed in both, matching the rest of the model; PyTorch's
        default leaves them uniformly random.
        """
        if self.init_mode == "default":
            return
        enc_std = 1.0 / math.sqrt(self.d_model)
        for proj in (self.in_proj, self.score_proj):
            if proj is not None:
                nn.init.normal_(proj.weight, mean=0.0, std=enc_std)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)
        if not self.tied:
            if self.init_mode == "sqrt_k":
                dec_std = 1.0 / math.sqrt(max(1, k))
            elif self.init_mode == "sqrt_k_selection_corrected":
                dec_std = 1.0 / (
                    math.sqrt(max(1, k)) * selection_gain(k, self.n_features)
                )
            else:  # unit_norm_dictionary
                dec_std = enc_std
            nn.init.normal_(self.out_proj.weight, mean=0.0, std=dec_std)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    @property
    def value_proj(self) -> nn.Linear:
        """Alias: ``in_proj`` is the value branch."""
        return self.in_proj

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        """``out_proj`` with the fixed global decoder scale folded in.

        ``g_D`` multiplies the decoder *weight*, not its bias, and it has to sit
        inside the differentiable graph: the MD optimizer reads ``p.grad`` of the
        fused weight and splits it into direction and gain gradients, so a
        forward that carries ``g_D`` is what puts the same ``g_D`` factor into
        all three (see decouple.DecoupledAdamW.step).  Scaling the linear's
        output rather than a copy of the matrix is the same function and the
        same gradients, without materializing ``g_D * W``.
        """
        if self.decoder_scale == 1.0:
            return self.out_proj(code)
        out = F.linear(code, self.out_proj.weight) * self.decoder_scale
        bias = self.out_proj.bias
        return out if bias is None else out + bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.in_proj(x)
        if self.gated:
            y = self.decode(self.gate(self.score_proj(x), value))
        else:
            y = self.decode(self.gate(value))
        # Identity unless post_norm was requested; Identity holds no parameters
        # or buffers, so state_dicts are unchanged when the option is off.
        y = self.post_norm(y)
        return y

    @property
    def diagnostics(self):
        return self.gate.diagnostics

    def extra_repr(self) -> str:
        branches = "score+value" if self.gated else "single"
        scale = "" if self.decoder_scale == 1.0 else f", g_D={self.decoder_scale:.4g}"
        shared = ", projections shared" if self.shared else ""
        return (
            f"d_model={self.d_model}, n_features={self.n_features}, "
            f"branches={branches}{scale}{shared}"
        )
