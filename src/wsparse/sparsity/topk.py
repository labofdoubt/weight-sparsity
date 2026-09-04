"""TopK + soft gate: a hard TopK forward support with a *wider* backward support.

Every masked tensor carries a second parameter ``s`` of the same shape, whose
soft gate is ``p = sigmoid(beta * s)``.  Two index sets are derived from ``s``
alone (equivalently from ``p``, the sigmoid being monotone):

    A = TopK(s)          the forward support, |A| = k
    B = Top(K+J)(s)      the backward / exploration support, A subset of B

The layer computes

    w_tilde = M_A * w * p                                          (forward)

and the gradients are *defined* -- not autograd-derived -- as

    dL/dw = M_B * p * G
    dL/ds = M_B * (G * w + Lambda) * beta * p * (1 - p)

with ``G = dL/dw_tilde`` and ``Lambda`` the optional soft-L0 coefficient
(``lambda_topk`` on A, ``lambda_explore`` on B\\A).  The hard TopK selection is
therefore never differentiated through; instead the backward pass pretends the
forward mask was the larger ``M_B``, which is what lets the ``j`` currently
inactive candidates accumulate weight *and* score updates and enter TopK on a
later step -- all from a single forward/backward pass.

``G`` is generally nonzero where ``w_tilde`` is zero (it is the gradient w.r.t.
the effective weight, not w.r.t. the layer output), so those positions do carry
a real learning signal.  Positions outside ``B`` get exactly zero.

TopK runs either over the whole tensor, per output row, or inside fixed-size
blocks (``groups``), which is what makes an N:M-style budget expressible as
``groups=block, block_size=M, k=N``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .masks import SparseLinear

# ``|beta * s| < TRANSITION_BAND`` is the band in which the sigmoid is still
# meaningfully between 0 and 1 (same convention as the other methods).
TRANSITION_BAND = 4.0


# --------------------------------------------------------------------------- #
# grouping helpers
# --------------------------------------------------------------------------- #


def group_shape(shape: Tuple[int, int], groups: str, block_size: int) -> Tuple[int, int]:
    """The ``(n_groups, group_size)`` view a weight tensor is TopK'd in.

    ``tensor`` -- one global TopK; ``row`` -- one per output row; ``block`` --
    one per contiguous run of ``block_size`` weights (row-major, so blocks run
    along the input dimension: ``block_size=4, k=2`` is 2:4 sparsity).
    """
    out_features, in_features = shape
    numel = out_features * in_features
    if groups == "tensor":
        return (1, numel)
    if groups == "row":
        return (out_features, in_features)
    if groups == "block":
        if block_size <= 0:
            raise ValueError(f"topk_block_size must be positive, got {block_size}")
        if numel % block_size:
            raise ValueError(
                f"topk_block_size={block_size} does not divide a "
                f"{out_features}x{in_features} weight ({numel} elements)"
            )
        return (numel // block_size, block_size)
    raise ValueError(f"unknown topk grouping: {groups!r} (tensor | row | block)")


def resolve_count(value: float, group_size: int, minimum: int = 0) -> int:
    """``0 < value < 1`` is a fraction of the group, ``value >= 1`` a count."""
    if value < 0:
        raise ValueError(f"expected a non-negative count or fraction, got {value}")
    n = int(round(value * group_size)) if value < 1.0 else int(round(value))
    return max(minimum, min(group_size, n))


def topk_masks(
    s: torch.Tensor, k: int, j: int, shape: Tuple[int, int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(M_A, M_B)`` as boolean tensors shaped like ``s``.

    One sorted ``topk(k + j)`` gives both sets: its first ``k`` columns are
    exactly TopK(s), the whole thing is Top(K+J)(s).
    """
    view = s.detach().reshape(shape)
    idx = view.topk(k + j, dim=1, sorted=True).indices
    mask_b = torch.zeros_like(view, dtype=torch.bool).scatter_(1, idx, True)
    mask_a = (
        mask_b
        if j == 0
        else torch.zeros_like(view, dtype=torch.bool).scatter_(1, idx[:, :k], True)
    )
    return mask_a.reshape(s.shape), mask_b.reshape(s.shape)


# --------------------------------------------------------------------------- #
# the autograd function
# --------------------------------------------------------------------------- #


class TopKSoftGate(torch.autograd.Function):
    """``w_tilde = M_A * w * sigmoid(beta*s)`` with the backward support ``M_B``.

    The masks are passed in (rather than computed here) so that the module can
    share one TopK per optimizer step between the forward pass, the soft-L0
    penalty and the logged statistics.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        weight: torch.Tensor,
        s: torch.Tensor,
        beta: torch.Tensor,
        mask_a: torch.Tensor,
        mask_b: torch.Tensor,
        w_grad_explore: bool,
    ) -> torch.Tensor:
        p = torch.sigmoid(beta * s.float())
        ctx.save_for_backward(weight, p, mask_a, mask_b, beta)
        ctx.w_grad_explore = w_grad_explore
        return (weight.float() * p * mask_a).to(weight.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        weight, p, mask_a, mask_b, beta = ctx.saved_tensors
        g = grad_out.float()
        grad_w = grad_s = None
        if ctx.needs_input_grad[0]:
            # M_{K+J} * p * G  (or M_K * p * G when exploration is off for w)
            support = mask_b if ctx.w_grad_explore else mask_a
            grad_w = (g * p * support).to(weight.dtype)
        if ctx.needs_input_grad[1]:
            # M_{K+J} * G * w * beta*p*(1-p)   -- the sigmoid is *not* detached
            grad_s = g * weight.float() * (beta * p * (1.0 - p)) * mask_b
        return grad_w, grad_s, None, None, None, None


# --------------------------------------------------------------------------- #
# the layer
# --------------------------------------------------------------------------- #


class TopKSoftGateLinear(SparseLinear):
    """``v = TopK_k(s) * w * sigmoid(beta * s)``, backward support ``Top_{k+j}(s)``."""

    def __init__(
        self,
        linear: nn.Linear,
        k: float,
        j: float = 0.0,
        s_init: float = 1.0,
        s_init_mode: str = "magnitude",
        groups: str = "tensor",
        block_size: int = 4,
        w_grad_support: str = "topk_j",
        soft_l0_enabled: bool = False,
        soft_l0_lambda_topk: float = 0.0,
        soft_l0_lambda_explore: float = 0.0,
        track_turnover: bool = True,
    ):
        super().__init__(linear)
        if w_grad_support not in ("topk_j", "topk"):
            raise ValueError(f"unknown w_grad_support: {w_grad_support!r} (topk_j | topk)")
        self.groups = groups
        self.group_shape = group_shape(
            (self.out_features, self.in_features), groups, block_size
        )
        group_size = self.group_shape[1]
        self.k = resolve_count(k, group_size, minimum=1)
        self.j = min(resolve_count(j, group_size, minimum=0), group_size - self.k)
        self.w_grad_explore = w_grad_support == "topk_j"
        self.soft_l0_enabled = soft_l0_enabled
        self.lambda_topk = float(soft_l0_lambda_topk)
        self.lambda_explore = float(soft_l0_lambda_explore)

        self.s = nn.Parameter(torch.empty_like(linear.weight, dtype=torch.float32))
        self.init_s_(float(s_init), s_init_mode)

        # TopK is recomputed whenever ``s`` changes (its version counter is
        # bumped by the optimizer step), and reused by every micro-batch of a
        # gradient-accumulation step, by the penalty and by the statistics.
        self._support: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._support_version = -1

        self.track_turnover = track_turnover
        self._previous_a: Optional[torch.Tensor] = None
        # fraction of TopK that changed at the last re-selection; a 0-dim buffer
        # so that updating it never forces a device sync.
        self.register_buffer("turnover", torch.zeros((), dtype=torch.float32), persistent=False)

    # ---- initialization --------------------------------------------------- #
    @torch.no_grad()
    def init_s_(self, scale: float, mode: str) -> None:
        """``constant`` | ``uniform`` | ``normal`` | ``magnitude``.

        ``magnitude`` (the default) makes the initial TopK the top-k weights by
        ``|w|`` and centres ``s`` on the selection boundary, so that ``s > 0``
        holds on exactly A at step 0: the hard mask, the soft gate and the TopK
        support all start out agreeing.  ``constant`` leaves the selection to
        index order and is only useful for tests.
        """
        if mode == "constant":
            self.s.fill_(scale)
            return
        if mode == "uniform":
            self.s.uniform_(-scale, scale)
            return
        if mode == "normal":
            self.s.normal_(0.0, max(scale, 1e-12))
            return
        if mode != "magnitude":
            raise ValueError(
                f"unknown s_init_mode: {mode!r} (constant | uniform | normal | magnitude)"
            )
        a = self.weight.detach().float().abs().reshape(self.group_shape)
        n = self.group_shape[1]
        values = a.topk(min(self.k + 1, n), dim=1, sorted=True).values
        if self.k < n:  # midpoint between the k-th and (k+1)-th largest |w|
            boundary = 0.5 * (values[:, self.k - 1 : self.k] + values[:, self.k : self.k + 1])
        else:
            boundary = values[:, -1:] * 0.5
        spread = a.std(dim=1, keepdim=True).clamp_min(1e-12)
        self.s.copy_((scale * (a - boundary) / spread).reshape(self.s.shape))

    # ---- support ----------------------------------------------------------- #
    @torch.compiler.disable
    def supports(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(M_K, M_{K+J})`` for the current ``s``, cached within a step.

        Opaque to ``torch.compile``: the cache is keyed on the version counter
        of ``s``, which Dynamo can only treat as data-dependent -- it would
        graph-break on the comparison anyway and then recompile on every
        optimizer step until it hit the recompile limit.  Selection is a topk
        plus a scatter, so there is nothing here for the compiler to fuse; the
        matmuls around it still compile.
        """
        version = self.s._version
        if self._support is not None and self._support_version == version:
            return self._support
        mask_a, mask_b = topk_masks(self.s, self.k, self.j, self.group_shape)
        if self.track_turnover:
            if self._previous_a is not None:
                entered = (mask_a & ~self._previous_a).sum()
                self.turnover.copy_(entered / max(1, self.k * self.group_shape[0]))
            self._previous_a = mask_a
        self._support = (mask_a, mask_b)
        self._support_version = version
        return self._support

    def gate(self) -> torch.Tensor:
        """``p = sigmoid(beta * s)`` (or its ``beta -> inf`` limit ``1[s > 0]``)."""
        if self.hard_mask:
            return (self.s > 0).to(torch.float32)
        return torch.sigmoid(self.beta * self.s.float())

    # ---- SparseLinear API --------------------------------------------------- #
    def logits(self) -> torch.Tensor:
        return self.s.float()

    def mask_parameters(self):
        return [self.s]

    def mask(self) -> torch.Tensor:
        return self.supports()[0].to(torch.float32) * self.gate()

    def hard_mask_tensor(self) -> torch.Tensor:
        return (self.supports()[0] & (self.s > 0)).to(torch.float32)

    def effective_weight(self) -> torch.Tensor:
        if self.hard_mask:
            return self.weight * self.mask().to(self.weight.dtype)
        mask_a, mask_b = self.supports()
        return TopKSoftGate.apply(
            self.weight, self.s, self.beta, mask_a, mask_b, self.w_grad_explore
        )

    def soft_l0(self) -> torch.Tensor:
        """``sum_{A} p`` -- the soft L0 of the *forward* support.

        Bounded above by ``k * n_groups``: the TopK budget is the hard FLOP
        budget, and the gates can only shrink the effective count below it.
        """
        return (self.supports()[0] * torch.sigmoid(self.beta * self.s.float())).sum()

    @torch.no_grad()
    def hard_l0(self) -> torch.Tensor:
        return (self.supports()[0] & (self.s > 0)).sum()

    @torch.no_grad()
    def transition_fraction(self) -> torch.Tensor:
        """Fraction of the Top-(K+J) support still inside the sigmoid's band.

        Measured over B rather than the whole tensor: positions outside B get no
        gradient at all, so their gates are irrelevant.
        """
        mask_b = self.supports()[1]
        inside = ((self.beta * self.s.float()).abs() < TRANSITION_BAND) & mask_b
        return inside.sum() / mask_b.sum().clamp_min(1)

    # ---- soft L0 penalty ------------------------------------------------------ #
    def penalty_coefficients(self) -> torch.Tensor:
        """``Lambda``: ``lambda_topk`` on A, ``lambda_explore`` on B\\A, 0 outside."""
        mask_a, mask_b = self.supports()
        return mask_b * (
            self.lambda_explore + (self.lambda_topk - self.lambda_explore) * mask_a
        )

    def extra_penalty(self) -> Optional[torch.Tensor]:
        """``sum_ij M_{K+J} * Lambda * p``.

        Computed over B only -- never the dense tensor -- and differentiated
        with ``Lambda`` and the masks held constant, so its contribution to the
        score gradient is exactly ``M_{K+J} * Lambda * beta*p*(1-p)``.
        """
        if not self.soft_l0_enabled:
            return None
        if self.lambda_topk == 0.0 and self.lambda_explore == 0.0:
            return None
        return (self.penalty_coefficients() * torch.sigmoid(self.beta * self.s.float())).sum()

    # ---- statistics ------------------------------------------------------------ #
    @property
    def topk_numel(self) -> int:
        """``|A|``: the hard FLOP budget of this layer."""
        return self.k * self.group_shape[0]

    @property
    def explore_numel(self) -> int:
        """``|B \\ A|``: positions that only ever get gradients."""
        return self.j * self.group_shape[0]

    @torch.no_grad()
    def stats(self) -> Dict[str, torch.Tensor]:
        mask_a, mask_b = self.supports()
        p = torch.sigmoid(self.beta * self.s.float())
        return {
            "gate_mean_topk": (mask_a * p).sum() / max(1, self.topk_numel),
            "gate_mean_explore": (
                ((mask_b & ~mask_a) * p).sum() / max(1, self.explore_numel)
                if self.j
                else torch.zeros((), device=p.device)
            ),
            "turnover": self.turnover,
        }

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"method=topk, k={self.k}, j={self.j}, groups={self.groups}"
            f"({self.group_shape[0]}x{self.group_shape[1]}), "
            f"w_grad={'topk+j' if self.w_grad_explore else 'topk'}"
        )
