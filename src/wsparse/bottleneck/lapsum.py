"""LapSum: a Laplace-CDF soft TopK over an already-sorted candidate pool.

For candidate scores ``r`` and a barrier ``b`` at temperature ``t > 0``::

    p_i = F((r_i - b) / t),      F(z) = 0.5 e^z          (z <= 0)
                                       1 - 0.5 e^{-z}    (z >  0)

with ``b`` chosen so that ``sum_i p_i = K``.  As ``t -> 0`` this converges to
the indicator of the largest ``K`` scores, which is the sign convention we want.

Nothing here is used numerically in the forward pass of the bottleneck -- the
forward is exact hard TopK.  These probabilities exist only to define the
backward surrogate, and the barrier is what couples the candidates to each
other (raising one score pushes the barrier up and squeezes everyone else).

The barrier solve exploits the piecewise-exponential structure of the Laplace
CDF: given scores sorted descending, one prefix scan and one suffix scan locate
the interval containing ``b``, and a single quadratic is solved in closed form.
No second sort, no iteration.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

LOG2 = math.log(2.0)
# |u| above this and asinh(u) is log(2u) to well past float32 precision; the
# switch also keeps exp(L) from overflowing when the score gaps dwarf t.
_ASINH_LINEAR = 30.0
# F(z) saturates for |z| >> 1: F = 1 - 0.5e^-60 and 0.5e^-60 ~ 1e-26 are both
# exact in float32 even after summing over M candidates, so clamping the scan
# this far from the barrier changes nothing numerically.  It does bound the
# prefix/suffix cancellation, which is otherwise fatal: a score sitting 1e8
# temperatures from the centre leaves log A with no significant digits at all
# (1e8 - 1e8 in float32).  The clamp is centred on r_K rather than r_max
# because that is where the barrier lives -- and whenever the span/t ratio is
# large enough for the clamp to bite, sum p = K forces b into the r_K/r_K+1
# gap, so a symmetric window around r_K provably contains it.
_SCAN_CLAMP = {torch.float32: 60.0, torch.float64: 300.0}


def laplace_cdf(z: torch.Tensor) -> torch.Tensor:
    """``F(z)``, evaluated without ever exponentiating a positive number."""
    return torch.where(
        z <= 0,
        0.5 * torch.exp(z.clamp(max=0.0)),
        1.0 - 0.5 * torch.exp(-z.clamp(min=0.0)),
    )


def laplace_pdf(z: torch.Tensor) -> torch.Tensor:
    """``F'(z) = 0.5 e^{-|z|}``."""
    return 0.5 * torch.exp(-z.abs())


def lapsum_probs_at(scores: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """``p_i = F((r_i - b) / t)`` for a batch of rows (no barrier solve)."""
    return laplace_cdf((scores - b.unsqueeze(-1)) / t.unsqueeze(-1))


def lapsum_budget(scores: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """``sum_i p_i`` -- the quantity the barrier drives to ``K``."""
    return lapsum_probs_at(scores, b, t).sum(-1)


# --------------------------------------------------------------------------- #
# closed-form barrier
# --------------------------------------------------------------------------- #


def lapsum_barrier_sorted(
    candidate_scores: torch.Tensor,
    k: float,
    t: torch.Tensor,
) -> torch.Tensor:
    """Solve ``sum_i F((r_i - b)/t) = k`` in closed form.  Returns ``b``.

    ``candidate_scores`` is ``(..., M)`` sorted **descending**; ``t`` is
    ``(...)``.  Runs under ``no_grad``: the gradient of ``b`` w.r.t. the scores
    is supplied analytically by :class:`_LapSumProbs`, never by autograd.

    On the interval ``r_j >= b >= r_{j+1}`` the budget is

        sum_i p_i = j - (1/2) e^{b/t} A_j + (1/2) e^{-b/t} B_j

    with ``A_j = sum_{i<=j} e^{-r_i/t}`` and ``B_j = sum_{i>j} e^{r_i/t}``, so
    ``y = e^{b/t}`` solves ``A_j y^2 + 2(k - j) y - B_j = 0``.  Everything is
    carried relative to the pivot score ``r_j``, which keeps ``A``/``B`` in
    ``[0, log M]`` in log space, and the positive root is taken through
    ``asinh`` -- which is exactly the branch-free form of the root and removes
    the cancellation that ``-B + sqrt(B^2 + 4AC)`` would suffer for ``j < k``.
    """
    with torch.no_grad():
        r = candidate_scores
        m = r.shape[-1]
        t_col = t.unsqueeze(-1)

        # Work in rho = (r - r_max) / t throughout: bounds every exponent by the
        # candidate *span* over t rather than the absolute score over t, makes
        # the solve exactly translation invariant, and lets the saturation
        # clamp below be applied consistently to the scan and the pivot.
        centre = min(max(int(k) - 1, 0), m - 1)
        ref = r[..., centre : centre + 1]
        clamp = _SCAN_CLAMP.get(r.dtype, 60.0)
        rho = ((r - ref) / t_col).clamp(-clamp, clamp)

        # logA'_j = logsumexp_{i<=j} (rho_j - rho_i)  in [0, log j]
        prefix = torch.logcumsumexp(-rho, dim=-1)
        log_a = prefix + rho
        # logB'_j = logsumexp_{i>j} (rho_i - rho_j)   in (-inf, log(M-j)]
        suffix = torch.flip(torch.logcumsumexp(torch.flip(rho, [-1]), dim=-1), [-1])
        neg_inf = torch.full_like(suffix[..., :1], float("-inf"))
        log_b = torch.cat([suffix[..., 1:], neg_inf], dim=-1) - rho

        # Budget at each knot b = r_j.  Increasing in j, so the number of knots
        # whose budget is <= k is exactly the index of the interval holding b.
        idx = torch.arange(1, m + 1, device=r.device, dtype=r.dtype)
        knot = idx - 0.5 * torch.exp(log_a) + 0.5 * torch.exp(log_b)
        j_star = (knot <= k).sum(-1)  # 0 .. M

        # Extend to j = 0 (b above every score: A = 0, pivot is r_max, rho = 0).
        log_a_all = torch.cat([torch.full_like(neg_inf, float("-inf")), log_a], dim=-1)
        log_b_all = torch.cat([suffix[..., :1] - rho[..., :1], log_b], dim=-1)
        pivot_all = torch.cat([rho[..., :1], rho], dim=-1)

        gather_at = j_star.unsqueeze(-1)
        la = torch.gather(log_a_all, -1, gather_at).squeeze(-1)
        lb = torch.gather(log_b_all, -1, gather_at).squeeze(-1)
        pivot = torch.gather(pivot_all, -1, gather_at).squeeze(-1)

        lo_edge = j_star == 0  # A = 0: linear, y = B / (2k)
        hi_edge = j_star == m  # B = 0: linear, y = 2(M - k) / A
        la_s = torch.where(lo_edge, torch.zeros_like(la), la)
        lb_s = torch.where(hi_edge, torch.zeros_like(lb), lb)

        d = j_star.to(r.dtype) - k
        half_sum = 0.5 * (la_s + lb_s)
        half_diff = 0.5 * (lb_s - la_s)
        # asinh(d * exp(-half_sum)), taken in log space so that a huge |u| --
        # which is where the A=0 / B=0 limits live -- cannot overflow
        log_u = torch.log(d.abs()) - half_sum
        big = log_u > _ASINH_LINEAR
        u = torch.exp(torch.where(big, torch.zeros_like(log_u), log_u))
        asinh = torch.where(big, log_u + LOG2, torch.asinh(u))
        asinh = torch.sign(d) * torch.nan_to_num(asinh, nan=0.0, neginf=0.0)
        log_w = half_diff + asinh

        log_w = torch.where(lo_edge, lb - math.log(2.0 * k), log_w)
        log_w = torch.where(hi_edge, math.log(2.0 * (m - k)) - la, log_w)
        return ref.squeeze(-1) + t * (pivot + log_w)


def lapsum_barrier_bisect(
    candidate_scores: torch.Tensor,
    k: float,
    t: torch.Tensor,
    tol: float = 1e-7,
    max_iters: int = 200,
) -> torch.Tensor:
    """Reference barrier solver: plain bisection on ``sum_i p_i - k = 0``.

    Only for tests, for validating :func:`lapsum_barrier_sorted`, and as a
    fallback.  Does not require sorted input.
    """
    with torch.no_grad():
        r = candidate_scores
        t_ = t
        lo = r.amin(-1) - 4.0 * t_
        hi = r.amax(-1) + 4.0 * t_
        # sum_i p is decreasing in b: grow the bracket until it straddles k
        for _ in range(60):
            need_lo = lapsum_budget(r, lo, t_) < k
            need_hi = lapsum_budget(r, hi, t_) > k
            if not (need_lo.any() or need_hi.any()):
                break
            span = (hi - lo).clamp_min(torch.finfo(r.dtype).tiny)
            lo = torch.where(need_lo, lo - span, lo)
            hi = torch.where(need_hi, hi + span, hi)
        for _ in range(max_iters):
            mid = 0.5 * (lo + hi)
            over = lapsum_budget(r, mid, t_) > k
            lo = torch.where(over, mid, lo)
            hi = torch.where(over, hi, mid)
            if bool((hi - lo).max() <= tol):
                break
        return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# exact fixed-temperature VJP
# --------------------------------------------------------------------------- #


class _LapSumProbs(torch.autograd.Function):
    """``p = F((r - b)/t)`` with the exact fixed-``t`` implicit-barrier VJP.

    ``b`` is a solved constant here, but it is a *function of every candidate
    score* through ``sum_i p_i = k``.  Differentiating that constraint gives
    ``db/dr_l = kappa_l / sum_j kappa_j``, hence

        dL/dr_i = kappa_i * (u_i - <q_budget, u>),    q_budget = kappa / sum kappa

    Using ``u_i kappa_i`` alone -- i.e. forgetting that the barrier moves --
    would let the surrogate inflate the budget instead of trading candidates
    against each other.
    """

    @staticmethod
    def forward(ctx, scores, b, t, k_active, sink, inactive_scale=1.0):  # type: ignore[override]
        z = (scores - b.unsqueeze(-1)) / t.unsqueeze(-1)
        # |z| and t are saved rather than kappa itself: the normalised budget
        # weights are a softmax of -|z| with the 1/2t prefactor cancelled, which
        # stays exact even when every kappa has underflowed (tiny t, wide gaps),
        # where kappa / sum(kappa) would be 0/0.
        ctx.save_for_backward(z.abs(), t)
        ctx.k_active = int(k_active)
        ctx.sink = sink
        ctx.inactive_scale = float(inactive_scale)
        return laplace_cdf(z)

    @staticmethod
    def backward(ctx, grad_p):  # type: ignore[override]
        abs_z, t = ctx.saved_tensors
        kappa = 0.5 * torch.exp(-abs_z) / t.unsqueeze(-1)  # gradient magnitude
        q_budget = torch.softmax(-abs_z, dim=-1)           # db/dr, always finite
        shared = (q_budget * grad_p).sum(-1, keepdim=True)
        grad_scores = kappa * (grad_p - shared)
        if ctx.inactive_scale != 1.0:
            # Reweight only the J candidates outside the forward support.  Note
            # this deliberately breaks the zero-sum property: sum_i grad_i is 0
            # only when the whole vector is scaled uniformly, so a scale != 1
            # lets the surrogate move the budget rather than purely redistribute
            # it.  That is the point of the knob, but it is a real change of
            # character, not just a magnitude.
            k = ctx.k_active
            grad_scores = torch.cat(
                [grad_scores[..., :k], grad_scores[..., k:] * ctx.inactive_scale], dim=-1
            )
        sink = ctx.sink
        if sink is not None:
            with torch.no_grad():
                k = ctx.k_active
                mag = grad_scores.abs()
                flat = mag.reshape(-1, mag.shape[-1])
                sink["grad_active"] = flat[:, :k].mean().detach()
                sink["grad_inactive"] = flat[:, k:].mean().detach()
                bins = min(8, flat.shape[-1])
                edges = torch.linspace(
                    0, flat.shape[-1], bins + 1, device=flat.device
                ).long()
                sink["grad_by_rank"] = torch.stack(
                    [flat[:, edges[i] : edges[i + 1]].mean() for i in range(bins)]
                ).detach()
        return grad_scores, None, None, None, None, None


def lapsum_probs(
    candidate_scores: torch.Tensor,
    b: torch.Tensor,
    t: torch.Tensor,
    k_active: int,
    sink: Optional[dict] = None,
    inactive_scale: float = 1.0,
) -> torch.Tensor:
    """Differentiable LapSum probabilities at a **detached** ``(b, t)``.

    ``inactive_scale`` reweights the gradient reaching the ``J`` candidates
    outside the forward support (ranks ``k:``), leaving the active ones alone.
    """
    return _LapSumProbs.apply(candidate_scores, b, t, k_active, sink, inactive_scale)
