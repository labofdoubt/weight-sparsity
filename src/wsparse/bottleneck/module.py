"""The dense-in / sparse-gate / dense-out bottleneck placed before an MLP."""

from __future__ import annotations

import torch
import torch.nn as nn

from .gate import AdaptiveLapSumTopKGate


class SparseTopKBottleneck(nn.Module):
    """``x -> W_in -> TopK/AbsTopK -> W_out -> (original MLP)``.

    Both projections are ordinary dense ``nn.Linear`` layers trained by the
    model's ordinary objective -- there is no reconstruction loss, no weight
    mask and no pruning anywhere in this module.  ``in_proj``/``out_proj``
    rather than down/up because ``n_features`` may be larger or smaller than
    ``d_model``.
    """

    def __init__(self, d_model: int, cfg, bias: bool = True):
        super().__init__()
        self.d_model = int(d_model)
        self.n_features = int(cfg.n_features)
        self.in_proj = nn.Linear(self.d_model, self.n_features, bias=bias)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.gate(self.in_proj(x)))

    @property
    def diagnostics(self):
        return self.gate.diagnostics

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, n_features={self.n_features}"
