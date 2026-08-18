"""Soft-masked linear layers.

Two masking parameterisations of the form ``v = w * m`` with
``m = sigmoid(beta * z)``:

``LTPLinear``  (Learned Threshold Pruning, arXiv:2003.00075)
    ``z = w**2 - tau`` with a single learnable scalar threshold ``tau`` per
    layer.  ``beta = 1 / T`` is the inverse temperature and comes purely from
    the schedule -- we deliberately do *not* rescale it by the weight variance
    the way eq. (15) of the paper does.

``CSLinear``   (Continuous Sparsification, arXiv:1912.04427)
    ``z = s`` with a free auxiliary parameter ``s`` of the same shape as ``w``.

The smooth L0 of a layer is ``sum(m)`` in both cases -- for LTP this is eq. (6)
of that paper, for CS it is ``||sigmoid(beta * s)||_1`` from eq. (4).

A third method, ``TopKSoftGateLinear`` (``topk.py``), keeps the same interface
but its forward support is a hard TopK over ``s`` rather than a threshold on
``z``, and its backward support is deliberately wider than its forward support.

Masks are always computed in float32 (``beta`` is often >= 1e4 and ``w**2`` is
~1e-4, so bf16 would be hopeless), and cast back to the weight dtype.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import SparsityConfig


class SparseLinear(nn.Module):
    """Base class: adopts the parameters of an existing ``nn.Linear``."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight  # reuse the very same Parameter
        self.bias = linear.bias
        # beta lives in a buffer so that it travels with .to(device) / state_dict
        # and does not trigger a torch.compile recompilation every step.
        self.register_buffer("beta", torch.ones((), dtype=torch.float32))
        self.hard_mask = False  # flipped by `hard_mask_mode` during evaluation

    # ---- to be provided by subclasses ------------------------------------ #
    def logits(self) -> torch.Tensor:
        """The pre-sigmoid score ``z`` (mask = sigmoid(beta * z))."""
        raise NotImplementedError

    def mask_parameters(self):
        """The parameters that define the mask (trained with ``mask_lr``)."""
        raise NotImplementedError

    def extra_penalty(self) -> Optional[torch.Tensor]:
        """A method-specific sparsity penalty, or ``None`` if there is none.

        Added to the loss by the controller on top of the shared ``l0_coef`` /
        ``target_density_coef`` terms.
        """
        return None

    # ---- shared ----------------------------------------------------------- #
    def hard_mask_tensor(self) -> torch.Tensor:
        """The binary mask the soft one anneals to as ``beta -> inf``."""
        return (self.logits() > 0).to(torch.float32)

    def mask(self) -> torch.Tensor:
        if self.hard_mask:
            return self.hard_mask_tensor()
        return torch.sigmoid(self.beta * self.logits())

    def effective_weight(self) -> torch.Tensor:
        return self.weight * self.mask().to(self.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.bias)

    # ---- sparsity statistics / penalties ---------------------------------- #
    @property
    def mask_numel(self) -> int:
        return self.weight.numel()

    def soft_l0(self) -> torch.Tensor:
        """Differentiable ``sum_k sigmoid(beta * z_k)``."""
        return torch.sigmoid(self.beta * self.logits()).sum()

    @torch.no_grad()
    def hard_l0(self) -> torch.Tensor:
        """Number of weights that survive hard pruning (``z > 0``)."""
        return (self.logits() > 0).sum()

    @torch.no_grad()
    def transition_fraction(self) -> torch.Tensor:
        """Fraction of weights inside the sigmoid transition region.

        ``|beta * z| < 4`` covers the band where the mask is meaningfully
        between 0 and 1; when this collapses to ~0 the mask has hardened and
        the thresholds/gates stop receiving gradient (the "premature
        termination" failure mode of LTP section 3.2).
        """
        return (torch.abs(self.beta * self.logits()) < 4.0).float().mean()

    @torch.no_grad()
    def apply_hard_mask_(self) -> None:
        """Permanently zero out the pruned weights (for export/eval)."""
        self.weight.mul_(self.hard_mask_tensor().to(self.weight.dtype))

    def to_linear(self) -> nn.Linear:
        """Materialise a dense ``nn.Linear`` holding the hard-pruned weights."""
        linear = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        with torch.no_grad():
            linear.weight.copy_(self.weight * self.hard_mask_tensor().to(self.weight.dtype))
            if self.bias is not None:
                linear.bias.copy_(self.bias)
        return linear


class LTPLinear(SparseLinear):
    """``m = sigmoid(beta * (w**2 - tau))`` with a learnable per-layer ``tau``."""

    def __init__(
        self,
        linear: nn.Linear,
        threshold_init: float = 0.0,
        grad_through_mask: bool = True,
    ):
        super().__init__(linear)
        self.grad_through_mask = grad_through_mask
        self.threshold = nn.Parameter(
            torch.tensor(float(threshold_init), dtype=torch.float32)
        )

    def logits(self) -> torch.Tensor:
        w2 = self.weight.float() ** 2
        if not self.grad_through_mask:
            # eq. (14) of the LTP paper: treat the sigmoid as a constant w.r.t.
            # w in the backward pass (so dv/dw = m), while tau keeps its full
            # gradient.  This is what prevents pruning from stalling.
            w2 = w2.detach()
        return w2 - self.threshold

    def mask_parameters(self):
        return [self.threshold]

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"method=ltp, grad_through_mask={self.grad_through_mask}"
        )


class CSLinear(SparseLinear):
    """``m = sigmoid(beta * s)`` with an auxiliary parameter ``s`` per weight."""

    def __init__(self, linear: nn.Linear, s_init: float = 0.05):
        super().__init__(linear)
        self.s = nn.Parameter(torch.full_like(linear.weight, float(s_init), dtype=torch.float32))

    def logits(self) -> torch.Tensor:
        return self.s.float()

    def mask_parameters(self):
        return [self.s]

    @torch.no_grad()
    def rescale_(self, beta: float, s_init: float) -> None:
        """CS round reset: ``s <- min(beta * s, s_init)`` (Algorithm 2, step 4)."""
        self.s.copy_(torch.clamp(beta * self.s, max=float(s_init)))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, method=cs"
        )


def make_sparse_linear(linear: nn.Linear, cfg: SparsityConfig) -> SparseLinear:
    """Wrap ``linear`` in the masked layer selected by ``cfg.method``."""
    if cfg.method == "ltp":
        return LTPLinear(
            linear,
            threshold_init=cfg.threshold_init,
            grad_through_mask=cfg.grad_through_mask,
        )
    if cfg.method == "cs":
        return CSLinear(linear, s_init=cfg.s_init)
    if cfg.method == "topk":
        # imported here: topk.py needs SparseLinear from this module
        from .topk import TopKSoftGateLinear

        return TopKSoftGateLinear(
            linear,
            k=cfg.k,
            j=cfg.j,
            s_init=cfg.s_init,
            s_init_mode=cfg.s_init_mode,
            groups=cfg.topk_groups,
            block_size=cfg.topk_block_size,
            w_grad_support=cfg.w_grad_support,
            soft_l0_enabled=cfg.soft_l0_enabled,
            soft_l0_lambda_topk=cfg.soft_l0_lambda_topk,
            soft_l0_lambda_explore=cfg.soft_l0_lambda_explore,
            track_turnover=cfg.topk_track_turnover,
        )
    raise ValueError(f"unknown sparsity method: {cfg.method}")


class hard_mask_mode:
    """Context manager: evaluate the model with binary masks (``z > 0``)."""

    def __init__(self, model: nn.Module, enabled: bool = True):
        self.modules = [m for m in model.modules() if isinstance(m, SparseLinear)]
        self.enabled = enabled
        self.previous: Optional[list] = None

    def __enter__(self):
        self.previous = [m.hard_mask for m in self.modules]
        for m in self.modules:
            m.hard_mask = self.enabled
        return self

    def __exit__(self, *exc):
        for m, prev in zip(self.modules, self.previous or []):
            m.hard_mask = prev
        return False
