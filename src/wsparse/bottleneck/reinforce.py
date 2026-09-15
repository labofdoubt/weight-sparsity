"""Exact-K stochastic support samplers for the REINFORCE bottleneck.

Two distributions over K-subsets of the q = K+J candidates, behind one
interface (:func:`sample_exact_k`):

``gumbel_pl``
    Gumbel-TopK: perturb logits with Gumbel noise, take the ordered top-K.
    The induced *ordered* law is Plackett-Luce; using its ordered log-prob in
    REINFORCE is unbiased for the set objective (the order is just an extra
    latent), at the cost of order-only variance.  Score gradient in O(q+K):
    with w = exp(a - max a), Z_t the pre-step-t remaining mass and
    c_t = sum_{u<=t} 1/Z_u,

        d log P / d a_j = 1 - w_j c_{r_j}   (selected at step r_j)
                        =   - w_j c_K       (not selected).

``conditional_bernoulli``
    Independent Bernoullis *conditioned on exactly K successes*:
    P(S) = exp(sum_{i in S} a_i) / Z_K with Z_K the degree-K elementary
    symmetric polynomial of exp(a).  Set-symmetric (no latent order), and

        d log P / d a_i = 1[i in S] - mu_i,      mu_i = P(i in S),

    with an O(qK) log-space prefix DP for Z_K, exact backward sampling, and a
    prefix x rolling-suffix pass for all marginals.

Everything here runs under ``no_grad`` in fp32: samples, score gradients and
log-probs are *coefficients*, never autograd nodes.  The caller builds the
gradient-only surrogate ``(a * score_grad).sum(-1)``, whose derivative w.r.t.
the live logits is exactly ``d log pi(sample) / d a``.  Both distributions are
invariant to a constant logit shift, so ``score_grad.sum(-1) ~= 0`` up to
numerics -- no common-mode projection is needed or applied.

``q == K`` is degenerate for both (the set is forced): the samplers return the
full mask with ``score_grad = 0`` and ``log_prob = 0`` so the latent PL order
cannot inject pure-variance gradients.
"""

from __future__ import annotations

from typing import Dict

import torch

DISTRIBUTIONS = ("gumbel_pl", "conditional_bernoulli")
_EPS = 1e-9
_NEG = float("-inf")


def _deterministic_result(a: torch.Tensor, k: int) -> Dict[str, torch.Tensor]:
    """q <= k: every candidate is selected, nothing is learnable."""
    mask = torch.ones_like(a)
    return dict(
        selected_mask=mask,
        score_grad=torch.zeros_like(a),
        log_prob=torch.zeros(a.shape[:-1], dtype=a.dtype, device=a.device),
        forced=True,
    )


# --------------------------------------------------------------------------- #
# Gumbel-TopK / Plackett-Luce
# --------------------------------------------------------------------------- #


def pl_score_from_order(a: torch.Tensor, order: torch.Tensor):
    """Analytic ``d log P(order) / d a`` and ``log P(order)`` in O(q + K).

    ``a``: (..., q) fp32 logits (already detached).  ``order``: (..., K) the
    sampled selection order (indices into the candidate axis, first = first
    chosen).  Exposed separately so tests can compare it against autograd on
    the sequential-logsumexp definition.
    """
    k = order.shape[-1]
    m = a.max(dim=-1, keepdim=True).values
    w = (a - m).exp()                                   # shift-invariant weights
    w_sel = torch.gather(w, -1, order)                  # (..., k)
    a_sel = torch.gather(a, -1, order)
    remaining = w.sum(-1, keepdim=True)                 # Z_1
    cum_inv = torch.zeros_like(remaining)
    cum_at_sel = torch.zeros_like(w_sel)
    log_prob = torch.zeros(a.shape[:-1], dtype=a.dtype, device=a.device)
    clipped = False
    for t in range(k):
        z_t = remaining.clamp_min(_EPS)
        clipped = clipped or bool((remaining < _EPS).any())
        cum_inv = cum_inv + 1.0 / z_t
        cum_at_sel[..., t] = cum_inv.squeeze(-1)
        log_prob = log_prob + (a_sel[..., t] - m.squeeze(-1)) - z_t.log().squeeze(-1)
        remaining = remaining - w_sel[..., t : t + 1]
    horizon = cum_inv.expand_as(w).clone()              # c_K everywhere ...
    horizon.scatter_(-1, order, cum_at_sel)             # ... c_{r_j} where selected
    mask = torch.zeros_like(a).scatter(-1, order, 1.0)
    score_grad = mask - w * horizon
    return mask, score_grad, log_prob, clipped


@torch.no_grad()
def gumbel_pl_sample(a: torch.Tensor, k: int) -> Dict[str, torch.Tensor]:
    """Ordered Gumbel-TopK sample with its Plackett-Luce score gradient."""
    q = a.shape[-1]
    if k >= q:
        return _deterministic_result(a, k)
    u = torch.rand_like(a).clamp_(_EPS, 1.0 - _EPS)
    gumbel = -torch.log(-torch.log(u))
    order = (a + gumbel).topk(k, dim=-1, sorted=True).indices
    mask, score_grad, log_prob, clipped = pl_score_from_order(a, order)
    return dict(selected_mask=mask, score_grad=score_grad, log_prob=log_prob,
                denom_clipped=clipped)


# --------------------------------------------------------------------------- #
# exact-K conditional Bernoulli
# --------------------------------------------------------------------------- #


def cb_prefix_table(a: torch.Tensor, k: int) -> torch.Tensor:
    """``logF[m, i, r]``: log total weight of r-subsets of candidates [0, i).

    Shape (M, q+1, k+1); ``logF[:, q, k]`` is ``log Z_K``.  Log-space with
    -inf for impossible states.
    """
    m_rows, q = a.shape
    logf = a.new_full((m_rows, q + 1, k + 1), _NEG)
    logf[:, :, 0] = 0.0
    for i in range(q):
        logf[:, i + 1, 1:] = torch.logaddexp(
            logf[:, i, 1:], a[:, i : i + 1] + logf[:, i, :-1]
        )
    return logf


@torch.no_grad()
def conditional_bernoulli_sample(a: torch.Tensor, k: int) -> Dict[str, torch.Tensor]:
    """Exact sample, marginals and score gradient for the exact-K law.

    ``a``: (..., q) fp32 detached logits.  O(qK) time, one (M, q+1, K+1)
    prefix table plus an O(K) rolling suffix.
    """
    q = a.shape[-1]
    if k >= q:
        return _deterministic_result(a, k)
    lead = a.shape[:-1]
    a2 = (a - a.max(dim=-1, keepdim=True).values).reshape(-1, q)  # shift-invariant
    m_rows = a2.shape[0]
    logf = cb_prefix_table(a2, k)
    logz = logf[:, q, k]

    # ---- exact backward sampling ------------------------------------------ #
    mask = a2.new_zeros(m_rows, q)
    k_rem = torch.full((m_rows,), k, dtype=torch.long, device=a2.device)
    for i in range(q - 1, -1, -1):
        num = a2[:, i] + logf[:, i, :].gather(1, (k_rem - 1).clamp_min(0)[:, None])[:, 0]
        den = logf[:, i + 1, :].gather(1, k_rem[:, None])[:, 0]
        p = (num - den).exp()
        p = torch.where(k_rem == 0, torch.zeros_like(p), p)
        p = torch.where(k_rem == i + 1, torch.ones_like(p), p)   # must take the rest
        inc = torch.rand_like(p) < p
        mask[:, i] = inc.to(mask.dtype)
        k_rem = k_rem - inc.long()
    # every row must have placed exactly k (k_rem == 0 by construction)

    # ---- marginals: prefix x rolling suffix -------------------------------- #
    mu = a2.new_zeros(m_rows, q)
    logb = a2.new_full((m_rows, k + 1), _NEG)
    logb[:, 0] = 0.0
    for i in range(q - 1, -1, -1):
        terms = logf[:, i, :k] + logb[:, :k].flip(-1)            # r + (K-1-r)
        mu[:, i] = (a2[:, i] + terms.logsumexp(-1) - logz).exp()
        newb = logb.clone()
        newb[:, 1:] = torch.logaddexp(logb[:, 1:], a2[:, i : i + 1] + logb[:, :-1])
        logb = newb

    score_grad = mask - mu
    log_prob = (mask * a2).sum(-1) - logz
    return dict(
        selected_mask=mask.reshape(*lead, q),
        score_grad=score_grad.reshape(*lead, q),
        log_prob=log_prob.reshape(lead),
        mu=mu.reshape(*lead, q),
        mu_sum_error=(mu.sum(-1) - float(k)).abs().max(),
    )


@torch.no_grad()
def sample_exact_k(a: torch.Tensor, k: int, distribution: str) -> Dict[str, torch.Tensor]:
    """Common interface: fp32 detached logits in, detached sample/score out."""
    a = a.detach().float()
    if distribution == "gumbel_pl":
        return gumbel_pl_sample(a, k)
    if distribution == "conditional_bernoulli":
        return conditional_bernoulli_sample(a, k)
    raise ValueError(
        f"unknown reinforce_distribution: {distribution!r} ({' | '.join(DISTRIBUTIONS)})"
    )
