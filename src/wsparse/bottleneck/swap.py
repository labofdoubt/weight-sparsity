"""Local one-swap Gibbs surrogate: exact hard TopK forward, swap-model backward.

The forward mask is exactly hard TopK, always.  The backward pretends the
support is a random variable over the current support ``S = {a_1..a_K}`` and
its ``K*J`` one-swap neighbours ``S - a_i + c_j``, where the candidates
``c_1..c_J`` are the next ``J`` ranks of the same Top(K+J) pool the LapSum
surrogate uses.  Every alternative support is still exactly ``K``-sparse.

Relative to the current support, the swap ``i -> j`` carries unnormalized weight

    w(S - a_i + c_j) = rho * exp((s_cj - s_ai) / T),        w(S) = 1,

so the weight factorizes over ``(i, j)`` and nothing here ever materializes a
``K x J`` tensor.  With ``b = s_aK`` (a purely numerical centre -- it cancels
exactly in the swap probability and shifts nothing),

    alpha = softmax(-(s_a - b) / T)          over the K active features
    beta  = softmax( (s_c - b) / T)          over the J candidates
    R     = sigmoid(log rho + logsumexp(-(s_a - b)/T) + logsumexp((s_c - b)/T))

``R`` is the total probability of swapping at all, ``q_ij = R alpha_i beta_j``
the probability of the particular swap, and the soft inclusion marginals are

    p_{a_i} = 1 - R alpha_i,        p_{c_j} = R beta_j,

which always sum to ``K``.  The custom Function below returns the hard pool
mask ``[1]*K + [0]*J`` in forward and the exact VJP of these marginals in
backward -- O(K + J) compute and saved state.

``rho`` comes from the ``swap_lambda`` config: ``"default"`` is ``rho = 1``
(the original construction, equivalent to ``lambda = KJ / (KJ + 1)``); a
numerical ``0 <= lambda < 1`` is the prior mass allowed off the current
support, ``rho = lambda / (KJ (1 - lambda))``, and bounds ``R <= lambda``.

``T`` is whatever the gate's shared prescribed-temperature path resolves (the
controller's schedule times the gate's ``temperature_scale_mode`` row scale);
nothing here re-derives, re-schedules or rescales it.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def swap_log_rho(swap_lambda, k: int, j: int) -> float:
    """``log rho`` from the ``swap_lambda`` config value.  Validates it too.

    ``"default"`` -> 0.0 (rho = 1); numerical ``lambda`` in [0, 1) ->
    ``log(lambda) - log(KJ) - log1p(-lambda)``; ``lambda = 0`` -> ``-inf``,
    which makes ``R`` exactly zero and the surrogate gradient vanish.
    """
    if isinstance(swap_lambda, str):
        if swap_lambda == "default":
            return 0.0
        raise ValueError(
            f"swap_lambda must be 'default' or a number in [0, 1), got {swap_lambda!r}"
        )
    lam = float(swap_lambda)
    if not 0.0 <= lam < 1.0:
        raise ValueError(f"swap_lambda must be 'default' or in [0, 1), got {lam}")
    if lam == 0.0:
        return float("-inf")
    return math.log(lam) - math.log(k * j) - math.log1p(-lam)


def swap_weights(
    cand_scores: torch.Tensor, t: torch.Tensor, k: int, log_rho: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(alpha, beta, R)`` for a sorted-descending ``(..., K+J)`` pool.

    Centred at ``s_aK`` so no raw score is ever exponentiated; the centre
    cancels identically in ``R`` (and softmaxes are shift-invariant), so it
    changes neither the value nor the gradient.
    """
    active, cand = cand_scores[..., :k], cand_scores[..., k:]
    centre = active[..., -1:].detach()
    t_col = t.unsqueeze(-1)
    log_u = -(active - centre) / t_col
    log_v = (cand - centre) / t_col
    alpha = torch.softmax(log_u, dim=-1)
    beta = torch.softmax(log_v, dim=-1)
    log_w = log_rho + log_u.logsumexp(-1) + log_v.logsumexp(-1)
    return alpha, beta, torch.sigmoid(log_w)


def swap_probs(
    cand_scores: torch.Tensor, t: torch.Tensor, k: int, log_rho: float
) -> torch.Tensor:
    """Soft inclusion marginals ``[1 - R alpha, R beta]`` over the pool.

    Differentiable through plain autograd: the reference implementation, used
    as ``hard_mask + p - p.detach()`` to verify the custom VJP.  ``sum p = K``
    by construction.
    """
    alpha, beta, r = swap_weights(cand_scores, t, k, log_rho)
    r_col = r.unsqueeze(-1)
    return torch.cat([1.0 - r_col * alpha, r_col * beta], dim=-1)


class _SwapGibbsMask(torch.autograd.Function):
    """Hard pool mask forward, exact one-swap-marginal VJP backward.

    With ``gA/gC`` the upstream gradient on the active/candidate mask entries
    and ``gbarA = <alpha, gA>``, ``gbarC = <beta, gC>``, ``D = R (gbarC - gbarA)``:

        dL/ds_ai = (R alpha_i / T) (gA_i - gbarC + D)
        dL/ds_cj = (R beta_j  / T) (gC_j - gbarA - D)

    which is the exact Jacobian-transpose of ``p`` above (through ``alpha``,
    ``beta`` *and* ``R``), sums to zero over the pool, and never forms ``q_ij``.
    ``T`` is saved from the forward, so backward reuses the exact effective
    temperature this forward resolved even if the schedule has since moved.
    """

    @staticmethod
    def forward(ctx, cand_scores, t, k_active, log_rho, sink):  # type: ignore[override]
        k = int(k_active)
        alpha, beta, r = swap_weights(cand_scores, t, k, log_rho)
        ctx.save_for_backward(alpha, beta, r, t)
        ctx.k_active = k
        ctx.sink = sink
        mask = torch.zeros_like(cand_scores)
        mask[..., :k] = 1.0
        return mask

    @staticmethod
    def backward(ctx, grad_m):  # type: ignore[override]
        alpha, beta, r, t = ctx.saved_tensors
        k = ctx.k_active
        g_a, g_c = grad_m[..., :k], grad_m[..., k:]
        gbar_a = (alpha * g_a).sum(-1, keepdim=True)
        gbar_c = (beta * g_c).sum(-1, keepdim=True)
        r_col, t_col = r.unsqueeze(-1), t.unsqueeze(-1)
        d = r_col * (gbar_c - gbar_a)
        grad_scores = torch.cat(
            [
                r_col * alpha / t_col * (g_a - gbar_c + d),
                r_col * beta / t_col * (g_c - gbar_a - d),
            ],
            dim=-1,
        )
        sink = ctx.sink
        if sink is not None:
            with torch.no_grad():
                mag = grad_scores.abs().reshape(-1, grad_scores.shape[-1])
                sink["grad_active"] = mag[:, :k].mean().detach()
                sink["grad_inactive"] = mag[:, k:].mean().detach()
                bins = min(8, mag.shape[-1])
                edges = torch.linspace(
                    0, mag.shape[-1], bins + 1, device=mag.device
                ).long()
                sink["grad_by_rank"] = torch.stack(
                    [mag[:, edges[i] : edges[i + 1]].mean() for i in range(bins)]
                ).detach()
        return grad_scores, None, None, None, None


def swap_gibbs_mask(
    cand_scores: torch.Tensor,
    t: torch.Tensor,
    k_active: int,
    log_rho: float,
    sink: Optional[dict] = None,
) -> torch.Tensor:
    """The hard ``[1]*K + [0]*J`` pool mask with the swap-model backward.

    ``t`` must already be the resolved effective temperature (detached, per
    row) from the gate's shared prescribed-temperature path.
    """
    return _SwapGibbsMask.apply(cand_scores, t, k_active, log_rho, sink)
