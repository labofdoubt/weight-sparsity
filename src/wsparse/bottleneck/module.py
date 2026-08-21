"""The dense-in / sparse-gate / dense-out bottleneck placed before an MLP."""

from __future__ import annotations

import torch
import torch.nn as nn

from .gate import AdaptiveLapSumTopKGate


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
        self.out_proj = nn.Linear(self.n_features, self.d_model, bias=bias)
        # Non-trainable output gain, fitted once before training so the block's
        # output variance matches its input's.  Registered only when enabled, so
        # checkpoints written without calibration still load.
        if cfg.calibrate_output:
            self.register_buffer("output_scale", torch.ones((), dtype=torch.float32))
        else:
            self.output_scale = None

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
        return y

    @property
    def diagnostics(self):
        return self.gate.diagnostics

    def extra_repr(self) -> str:
        branches = "score+value" if self.gated else "single"
        return (
            f"d_model={self.d_model}, n_features={self.n_features}, branches={branches}"
        )
