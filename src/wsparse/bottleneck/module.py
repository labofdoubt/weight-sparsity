"""The dense-in / sparse-gate / dense-out bottleneck placed before an MLP."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """

    def __init__(self, d_model: int, cfg, bias: bool = True):
        super().__init__()
        self.d_model = int(d_model)
        self.n_features = int(cfg.n_features)
        self.gated = cfg.selection_mode == "gated_topk"
        # in_proj is the value branch; score_proj (gated_topk only) ranks.
        self.in_proj = nn.Linear(self.d_model, self.n_features, bias=bias)
        self.score_proj = (
            nn.Linear(self.d_model, self.n_features, bias=bias) if self.gated else None
        )
        self.gate = AdaptiveLapSumTopKGate(
            n_features=self.n_features,
            k=cfg.k,
            j=cfg.j,
            n_eff=cfg.n_eff,
            selection_mode=cfg.selection_mode,
            effective_count_metric=cfg.effective_count_metric,
            boundary_mode=cfg.boundary_mode,
            one_sided_weight_mode=cfg.one_sided_weight_mode,
            surrogate_mode=cfg.surrogate_mode,
            surrogate_grad_scale=cfg.surrogate_grad_scale,
            inactive_grad_scale=cfg.inactive_grad_scale,
            fixed_temperature=cfg.fixed_temperature,
            temperature_scale_mode=cfg.temperature_scale_mode,
            temperature_solver_tol=cfg.temperature_solver_tol,
            temperature_solver_max_iters=cfg.temperature_solver_max_iters,
            barrier_solver_tol=cfg.barrier_solver_tol,
            solver_dtype=cfg.solver_dtype,
            log_diagnostics=cfg.log_diagnostics,
            hard_inference=cfg.hard_inference,
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
        self.tied = bool(getattr(cfg, "tie_encoder_decoder", False))
        if self.tied and self.init_mode != "unit_norm_dictionary":
            raise ValueError(
                "tie_encoder_decoder requires init_mode='unit_norm_dictionary': "
                "tying only makes sense when both sides share a scale"
            )
        if self.tied:
            self.out_proj = TiedDecoder(self.in_proj, bias=bias)
        else:
            self.out_proj = nn.Linear(self.n_features, self.d_model, bias=bias)
        self._init_projections(int(cfg.k))
        # Non-trainable output gain, fitted once before training so the block's
        # output variance matches its input's.  Registered only when enabled, so
        # checkpoints written without calibration still load.
        if cfg.calibrate_output:
            self.register_buffer("output_scale", torch.ones((), dtype=torch.float32))
        else:
            self.output_scale = None
        self.reconstruction_coef = float(cfg.reconstruction_coef)
        self.reconstruction_normalize = bool(cfg.reconstruction_normalize)
        self._reconstruction = None

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.in_proj(x)
        if self.gated:
            y = self.out_proj(self.gate(self.score_proj(x), value))
        else:
            y = self.out_proj(self.gate(value))
        if self.output_scale is not None:
            y = y * self.output_scale.to(y.dtype)
        if self.reconstruction_coef and self.training and torch.is_grad_enabled():
            # Held for the controller to collect after this forward.  It has to
            # be produced here rather than recomputed later: the term depends on
            # the activations of *this* micro-batch, unlike the weight-sparsity
            # penalty, which is a function of parameters alone.
            diff = (y.float() - x.float()).pow(2).sum(-1)
            if self.reconstruction_normalize:
                # relative error, so the coefficient means the same thing at any
                # activation scale and across placements
                diff = diff / (x.float().pow(2).sum(-1) + 1e-8)
            self._reconstruction = diff.mean()
        return y

    def take_reconstruction(self):
        """Pop the term recorded by the last forward (None if disabled)."""
        term, self._reconstruction = self._reconstruction, None
        return term

    @property
    def diagnostics(self):
        return self.gate.diagnostics

    def extra_repr(self) -> str:
        branches = "score+value" if self.gated else "single"
        return (
            f"d_model={self.d_model}, n_features={self.n_features}, branches={branches}"
        )
