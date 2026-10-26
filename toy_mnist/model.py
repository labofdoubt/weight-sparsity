"""The toy model: ``z = W2 relu(W1 x')`` with ``x'`` the optionally rescaled,
linearly mapped and gated input,

    v  = exp(g) * x                 (rescale.enabled; g trainable, init 0)
    x' = AbsTopK_k(v)               (topk.enabled, no W0)
    x' = W0 AbsTopK_k(v)            (model.pre_width > 0, topk.site = input)
    x' = AbsTopK_k(W0 v)            (model.pre_width > 0, topk.site = pre_linear)

with ``W0`` a bias-free linear layer 784 -> pre_width and the gate's hard mask
passing the backward through the active entries only.  ``gate_input`` and
``forward_from_gate`` split the forward at the gate input ``u`` (``v`` or
``W0 v``) so that a DUST step can rerun the tail from a perturbed ``u``.

``model.pre_sparsity = S > 0`` makes ``W0`` row-sparse by projection
(``project_pre``): every row keeps its ``S`` entries of largest magnitude.
The trainer applies it once at initialization and after every optimizer
step; the gradient and the update themselves stay dense, so an entry at 0 can
re-enter a row's support when its dense update is large enough.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config, N_CLASSES, N_PIXELS


class DiagonalRescale(nn.Module):
    """``x -> exp(g) * x``; ``g`` is the parameter, zero at initialization."""

    def __init__(self, n: int):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.exp(self.log_scale)

    def extra_repr(self) -> str:
        return f"n={self.log_scale.numel()}"


class AbsTopKGate(nn.Module):
    """Hard AbsTopK over the last axis: the ``k`` entries of largest ``|u|``
    pass, the rest are zero.  Numerically hard mask ``m``; the backward is
    ``dL/du = m * dL/dx'`` (no gradient to the inactive entries).

    ``m = min(k + j, n)`` is the width of the sorted candidate pool a DUST step
    perturbs; the first ``k`` of it are the support.  ``usage`` is an EMA of
    the per-feature activation rate over training forwards, ``seen`` the
    features active at least once since the trainer last cleared it.
    """

    def __init__(self, n_features: int, k: int, j: int = 0, usage_decay: float = 0.99):
        super().__init__()
        if k < 1 or j < 0:
            raise ValueError("AbsTopKGate needs k >= 1 and j >= 0")
        self.n_features = int(n_features)
        self.k = int(min(k, n_features))
        self.j = int(j)
        self.m = int(min(k + j, n_features))
        self.trivial = self.k >= self.n_features
        self.usage_decay = float(usage_decay)
        self.register_buffer("usage", torch.zeros(n_features))
        self.register_buffer("usage_n", torch.zeros((), dtype=torch.long))
        self.register_buffer("seen", torch.zeros(n_features, dtype=torch.bool))
        self._dust_capture: Optional[List[torch.Tensor]] = None

    @contextmanager
    def capture(self, records: List[torch.Tensor]) -> Iterator[None]:
        """Append the gate input of every forward inside the block to ``records``."""
        prior = self._dust_capture
        self._dust_capture = records
        try:
            yield
        finally:
            self._dust_capture = prior

    def pool(self, u: torch.Tensor) -> torch.Tensor:
        """Indices of the Top-``m`` pool by ``|u|``, sorted, ``[B, m]``."""
        return torch.topk(u.detach().abs(), self.m, dim=-1, largest=True, sorted=True).indices

    def support(self, u: torch.Tensor) -> torch.Tensor:
        """Indices of the ``k`` entries of largest ``|u|`` per row, ``[B, k]``."""
        return torch.topk(u.detach().abs(), self.k, dim=-1, largest=True, sorted=False).indices

    def hard_mask(self, u: torch.Tensor) -> torch.Tensor:
        if self.trivial:
            return torch.ones_like(u)
        return torch.zeros_like(u).scatter(-1, self.support(u), 1.0)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        if self._dust_capture is not None:
            self._dust_capture.append(u)
        mask = self.hard_mask(u)
        if self.training:
            self._record_usage(mask)
        return u * mask

    @torch.no_grad()
    def _record_usage(self, mask: torch.Tensor) -> None:
        rate = mask.reshape(-1, self.n_features).float().mean(0)
        if int(self.usage_n) == 0:
            self.usage.copy_(rate)
        else:
            self.usage.mul_(self.usage_decay).add_(rate, alpha=1.0 - self.usage_decay)
        self.usage_n += 1
        self.seen |= rate > 0

    def extra_repr(self) -> str:
        return f"n={self.n_features}, k={self.k}, j={self.j}, pool={self.m}"


class ToyMLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        n = N_PIXELS
        self.rescale = DiagonalRescale(n) if cfg.rescale.enabled else None
        self.pre = (nn.Linear(n, cfg.model.pre_width, bias=False)
                    if cfg.model.pre_width > 0 else None)
        self.pre_sparsity = int(cfg.model.pre_sparsity)
        if self.pre_sparsity > 0 and self.pre is None:
            raise ValueError("model.pre_sparsity needs model.pre_width > 0")
        self.gate_site = cfg.topk.site
        if self.pre is None and self.gate_site != "input":
            raise ValueError("topk.site='pre_linear' needs model.pre_width > 0")
        n_gate = cfg.model.pre_width if self.gate_site == "pre_linear" else n
        self.gate = (AbsTopKGate(n_gate, cfg.topk.k, cfg.topk.resolved_j(n_gate))
                     if cfg.topk.enabled else None)
        self.fc1 = nn.Linear(cfg.model.pre_width if self.pre is not None else n,
                             cfg.model.hidden, bias=cfg.model.bias)
        self.fc2 = nn.Linear(cfg.model.hidden, N_CLASSES, bias=cfg.model.bias)

    @torch.no_grad()
    def project_pre(self) -> Optional[torch.Tensor]:
        """PerRowAbsTopS on ``W0``: every row keeps its ``pre_sparsity``
        entries of largest magnitude, the rest become 0.  Returns the boolean
        support mask (``None`` when ``W0`` is absent or dense)."""
        if self.pre is None or self.pre_sparsity <= 0:
            return None
        w = self.pre.weight
        idx = torch.topk(w.abs(), self.pre_sparsity, dim=1, largest=True, sorted=False).indices
        mask = torch.zeros_like(w, dtype=torch.bool).scatter_(1, idx, True)
        w.mul_(mask)
        return mask

    def gate_input(self, x: torch.Tensor) -> torch.Tensor:
        """The vector the gate acts on: ``exp(g) * x`` (or ``x``), mapped by
        ``W0`` first when the gate sits after it."""
        v = self.rescale(x) if self.rescale is not None else x
        if self.pre is not None and self.gate_site == "pre_linear":
            return self.pre(v)
        return v

    def forward_from_gate(self, u: torch.Tensor) -> torch.Tensor:
        """The tail from the gate input: gate (if any), ``W0`` when the gate
        sits on the input, the MLP; returns logits."""
        h = self.gate(u) if self.gate is not None else u
        if self.pre is not None and self.gate_site == "input":
            h = self.pre(h)
        return self.fc2(F.relu(self.fc1(h)))

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        logits = self.forward_from_gate(self.gate_input(x))
        ce = F.cross_entropy(logits.float(), y) if y is not None else None
        return logits, ce
