"""Independent-threshold JumpReLU gate: hard forward, kernel gradient to theta only.

Each feature owns a trainable threshold ``theta_i`` (parameterized as
``log_theta`` so thresholds stay positive -- scores are ``|a|`` here).  The
margin ``m_i = score_i - theta_i`` ranks the Top(K+J) candidate pool, and for
candidates the forward is completely hard:

    h_i = H(m_i)                     (strict: m > 0)
    y_i = value_i * h_i

Everything outside the pool contributes zero and receives zero gradient of any
kind from this gate.

The backward rules are deliberately asymmetric (the whole point of the method):

    dy/dvalue_i   = h_i                                   (plain hard mask)
    ~dy/dtheta_i  = -theta_i * kappa_T(m_i)               (KDE pseudo-derivative)
    dy/dscore_i   = 0                                     (no boundary STE to z)

using the exact distributional identity ``z * delta(z - theta) = theta *
delta(z - theta)`` -- the value at the boundary is ``theta``, so the kernel
estimate multiplies ``theta``, not ``z``.  ``kappa_T`` is a rectangle of
ONE-SIDED width ``T``:

    kappa_T(m) = 1/(2T) * 1{|m| < T}

(the JumpReLU paper's ``(1/eps) K(m/eps)`` with a rectangle on (-1/2, 1/2) has
one-sided width eps/2; here ``T`` *is* the one-sided width, total window 2T).

The hard active count gets the matching pseudo-derivative so a target-K loss
``lambda * (K - L0)^2`` can steer the thresholds:

    L0 = sum_i h_i        ~dL0/dtheta_i = -kappa_T(m_i)       dL0/dscore = 0

There is no shared boundary, no mass constraint and no pairwise coupling: the
only interactions are the Top(K+J) pool selection and the scalar (K - L0)^2.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


def rect_kernel(margin: torch.Tensor, width: float) -> torch.Tensor:
    """``1/(2T) * 1{|m| < T}`` -- the rectangular KDE kernel, one-sided width T."""
    return (margin.abs() < width).to(margin.dtype) / (2.0 * width)


def default_log_theta(k: int, n_features: int) -> float:
    """``log`` of the expected k-th largest ``|N(0,1)|`` score of ``n`` draws.

    The rectangle gives theta a gradient only within ``T`` of the boundary, so
    thresholds must start where the scores actually are.  Under the repo's
    init conventions the pre-activation scores are approximately unit-scale
    half-normals, so the natural starting boundary is the magnitude the k-th
    survivor sits at: ``P(|Z| > t) = k/n`` -- the same inverse-Mills geometry
    ``selection_gain`` uses, and scipy-free via erfinv.
    """
    p = min(1.0 - 1e-9, max(1e-9, k / max(1, n_features)))
    t = math.sqrt(2.0) * float(
        torch.erfinv(torch.tensor(1.0 - p, dtype=torch.float64))
    )
    return math.log(max(t, 1e-6))


class _JumpReLUGate(torch.autograd.Function):
    """``y = value * H(score - theta)`` with the theta-only boundary gradient.

    ``score`` arrives detached (the caller guarantees it), so the only inputs
    that carry gradient are ``value`` (plain hard mask) and ``theta`` (kernel
    pseudo-derivative).  Nothing here differentiates the pool selection.
    """

    @staticmethod
    def forward(ctx, value, theta, score, width):  # type: ignore[override]
        margin = score - theta.detach()
        h = (margin > 0).to(value.dtype)
        kappa = rect_kernel(margin, float(width))
        ctx.save_for_backward(h, kappa, theta.detach())
        return value * h

    @staticmethod
    def backward(ctx, grad_y):  # type: ignore[override]
        h, kappa, theta = ctx.saved_tensors
        grad_value = grad_y * h
        grad_theta = grad_y * (-theta) * kappa.to(grad_y.dtype)
        return grad_value, grad_theta, None, None


class _JumpReLUCount(torch.autograd.Function):
    """Hard per-row active count with the theta-only pseudo-derivative.

    forward:  L0 = sum_i H(score_i - theta_i)          (the *hard* mask)
    backward: ~dL0/dtheta_i = -kappa_T(score_i - theta_i),  dL0/dscore = 0
    """

    @staticmethod
    def forward(ctx, theta, score, width):  # type: ignore[override]
        margin = score - theta.detach()
        kappa = rect_kernel(margin, float(width))
        ctx.save_for_backward(kappa)
        return (margin > 0).to(theta.dtype).sum(-1)

    @staticmethod
    def backward(ctx, grad_l0):  # type: ignore[override]
        (kappa,) = ctx.saved_tensors
        return grad_l0.unsqueeze(-1) * (-kappa), None, None


def jumprelu_forward(
    value_c: torch.Tensor,
    theta_c: torch.Tensor,
    score_c: torch.Tensor,
    width: float,
) -> torch.Tensor:
    """Candidate outputs ``value * H(score - theta)``; see :class:`_JumpReLUGate`."""
    return _JumpReLUGate.apply(value_c, theta_c, score_c.detach(), width)


def jumprelu_count(
    theta_c: torch.Tensor, score_c: torch.Tensor, width: float
) -> torch.Tensor:
    """Per-row hard L0 over the candidate pool; see :class:`_JumpReLUCount`."""
    return _JumpReLUCount.apply(theta_c, score_c.detach(), width)
