"""Learned-score Top-K trained with Gumbel policy gradients.

``selection_mode="learned_score_topk"`` ranks the code dimensions by an
independent linear scorer of the residual-stream state ``h`` at the
bottleneck, ``s = W_s h + b_s`` (one score per code dimension), and keeps the
values unscaled on a binary support::

    no code skip   y = W_D ((W_E h) * m)
    code skip      c' = (c + W_E Delta) * m           (code_residual)

``m`` depends on ``s`` only.  With ``gumbel_policy`` on, a training forward
draws the support per sequence, position and bottleneck from the Gumbel-Top-K
(Plackett-Luce) policy at temperature ``tau``::

    (i_1, ..., i_K) = TopK_K(s / tau + G),   G_j ~ Gumbel(0, 1),

ordered by descending perturbed score; evaluation uses the deterministic
``TopK_K(s)``.  The ordered selection has the differentiable log-probability

    log p(i_1..i_K | h) = sum_r [ s_{i_r} / tau - logsumexp_{j not in {i_1..i_{r-1}}} s_j / tau ],

computed with the stable identity

    logsumexp_{j not in {i_1..i_{r-1}}} z_j = logaddexp( logsumexp_{j not in S} z_j,
                                                         logsumexp_{q >= r} z_{i_q} )

(the remaining set at step ``r`` is the never-selected features plus the
selected ones from rank ``r`` on), so the cost is one masked ``logsumexp``
over the N features and one reverse ``logcumsumexp`` over the K selected ones,
with autograd reaching every score, selected or not.  The sampled indices and
the noise carry no gradient.

``gumbel_pool="top_kj"`` restricts the draw to the deterministic candidate
pool ``C = TopIndices_{K+J}(s)``: ``S = TopK_K(s_C / tau + G_C)`` (the same
order as ``s_C + tau G_C``), and the ordered log-probability normalizes over
the remaining members of ``C`` at each step.  ``C`` is treated as fixed in
the backward, so only its members' scores receive the policy gradient; the
estimator is the policy gradient of the pool-restricted policy (the
dependence of ``C`` itself on ``s`` is not differentiated).  Evaluation is the
deterministic ``TopK_K(s)`` over all N in both cases.

The hard mask passes no ordinary gradient to ``s``: the scorer learns from the
policy term alone.  Per micro-batch of ``B`` sequences with mean CE ``l_b``,
valid-target weight ``w_b = n_b / N_valid`` (``1/B`` for full windows) and
``P_b`` the sum of the ordered log-probabilities over the valid target
positions and all bottlenecks of sequence ``b``::

    beta_b = (1 / (B - 1)) sum_{a != b} l_a                    (leave-one-out)
    L      = sum_b w_b l_b + lambda sum_b w_b stopgrad(l_b - beta_b) P_b

and the trainer backpropagates ``L_CE + [T - stopgrad(T)]`` with ``T`` the
policy term, so the loss VALUE is the sampled CE while the gradient is that of
``L``.  ``beta_b`` uses the other sequences of the same micro-batch (their
supports are drawn independently of sequence ``b``'s), so with ``lambda = 1``
the policy term is an unbiased estimator of the selection part of the gradient
of the expected sampled CE.  The ordered log-probability (not the
unordered-subset one) keeps that property: the loss depends on the set only,
and the order is marginalized by the expectation.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Tuple

import torch

#: version of the optional ``gumbel_state`` checkpoint payload
GUMBEL_STATE_VERSION = 1


# --------------------------------------------------------------------------- #
# sampling and log-probability
# --------------------------------------------------------------------------- #


def sample_gumbel(shape, generator: Optional[torch.Generator] = None,
                  device: Optional[torch.device] = None,
                  dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Standard Gumbel noise ``-log(-log U)``, ``U ~ Uniform(0, 1)``.

    ``torch.rand`` returns values in ``[0, 1)`` on a grid of ``2^-24`` in
    float32; ``U`` is clamped to ``[tiny, 1 - eps]``, so ``G`` is finite.
    That cuts the lower tail at ``P(G < -2.8) ~ 6e-8`` and the upper one at
    ``G ~ 16.6``: a machine-precision truncation, not a modelling choice.
    """
    u = torch.rand(shape, generator=generator, dtype=dtype, device=device)
    info = torch.finfo(dtype)
    u = u.clamp(info.tiny, 1.0 - info.eps)
    return -torch.log(-torch.log(u))


def ordered_log_prob(logits: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Plackett-Luce log-probability of the ordered selection ``idx``.

    ``logits`` is ``s / tau`` (``[..., N]``, float32 or wider, differentiable);
    ``idx`` (``[..., K]``, long) the selected features in sampled order.
    Returns ``[...]``::

        sum_{r=1..K} [ z_{i_r} - logsumexp_{j not in {i_1, ..., i_{r-1}}} z_j ].

    The denominator at step ``r`` is ``logaddexp`` of the never-selected
    features' ``logsumexp`` and the ``logsumexp`` of the selected ones from
    rank ``r`` on (a reverse ``logcumsumexp``); both are stable, nothing is
    subtracted in linear space, and the gradient reaches every logit.
    """
    n, k = logits.shape[-1], idx.shape[-1]
    if not 1 <= k <= n:
        raise ValueError(f"ordered_log_prob: need 1 <= K <= N, got K={k}, N={n}")
    z_sel = logits.gather(-1, idx)
    suffix = torch.logcumsumexp(z_sel.flip(-1), dim=-1).flip(-1)
    if k < n:
        selected = torch.zeros_like(logits, dtype=torch.bool).scatter(-1, idx, True)
        rest = torch.logsumexp(logits.masked_fill(selected, float("-inf")), dim=-1,
                               keepdim=True)
        denom = torch.logaddexp(suffix, rest)
    else:  # every feature selected: the remaining set is the selected suffix
        denom = suffix
    return (z_sel - denom).sum(-1)


#: where the Gumbel-Top-K draw takes its K members (gumbel_pool)
POOLS = ("all", "top_kj")


class GumbelDraw(NamedTuple):
    """One Gumbel-Top-K draw.

    ``idx`` (``[..., K]``): the selected features, indices into all N, in
    sampled (descending perturbed) order.  ``logits`` (``[..., N]``, float32,
    differentiable): ``scores / tau``.  ``pool_logits`` (``[..., M]``) and
    ``pos`` (``[..., K]``, positions in the pool) are what
    :func:`ordered_log_prob` normalizes over: all N features (``M = N``,
    ``pos = idx``) or the deterministic Top-(K+J) pool (``M = K + J``).
    """

    idx: torch.Tensor
    logits: torch.Tensor
    pool_logits: torch.Tensor
    pos: torch.Tensor


def gumbel_topk(scores: torch.Tensor, k: int, tau: float,
                generator: Optional[torch.Generator] = None,
                pool: Optional[int] = None) -> GumbelDraw:
    """A Gumbel-Top-K draw of K features, ordered by descending perturbed score.

    ``pool=None``: over all N, ``TopK_K(scores / tau + G)``, a draw of the
    ordered Plackett-Luce selection with weights ``exp(scores / tau)``.
    ``pool=M`` (``K + J``): first the deterministic candidate pool
    ``C = TopIndices_M(scores)``, then ``TopK_K(s_C / tau + G_C)`` with one
    Gumbel variate per candidate -- the same order as ``s_C + tau G_C`` -- a
    Plackett-Luce draw normalized over ``C`` only.  The pool membership is
    discrete (indices from the clean scores); the gathered pool logits keep
    their autograd path, so features outside ``C`` receive no policy
    gradient.  The perturbation uses detached logits.
    """
    k = int(k)
    with torch.autocast(device_type=scores.device.type, enabled=False):
        logits = scores.float() / float(tau)
        if pool is None or int(pool) >= logits.shape[-1]:
            noise = sample_gumbel(logits.shape, generator=generator, device=logits.device)
            idx = torch.topk(logits.detach() + noise, k, dim=-1, largest=True,
                             sorted=True).indices
            return GumbelDraw(idx, logits, logits, idx)
        if not k <= int(pool):
            raise ValueError(f"gumbel_topk: the pool ({pool}) is smaller than K ({k})")
        cand_idx = torch.topk(logits.detach(), int(pool), dim=-1, largest=True,
                              sorted=True).indices
        pool_logits = logits.gather(-1, cand_idx)
        noise = sample_gumbel(pool_logits.shape, generator=generator, device=logits.device)
        pos = torch.topk(pool_logits.detach() + noise, k, dim=-1, largest=True,
                         sorted=True).indices
        return GumbelDraw(cand_idx.gather(-1, pos), logits, pool_logits, pos)


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #


def loo_baseline(seq_ce: torch.Tensor, seq_valid: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(beta, has)``: the leave-one-out baseline per sequence, detached.

    ``beta_b = (sum_{a != b} l_a) / (B_eff - 1)`` over the sequences ``a`` of
    the micro-batch that have at least one valid target (``has``); a sequence
    without one has no loss, is nobody's peer and gets ``beta = 0`` (its
    weight in the loss is zero).  Fewer than two such sequences leave the
    baseline undefined: that raises instead of using a sequence's own loss.
    """
    has = seq_valid > 0
    n = int(has.sum())
    if n < 2:
        raise ValueError(
            "gumbel_policy: the leave-one-out baseline needs at least two sequences with "
            f"valid targets in the micro-batch, got {n} (set train.micro_batch_size >= 2)")
    c = torch.where(has, seq_ce.detach().double(), torch.zeros((), dtype=torch.float64,
                                                               device=seq_ce.device))
    beta = torch.where(has, (c.sum() - c) / float(n - 1), torch.zeros_like(c))
    return beta, has


def gumbel_policy_loss(ce: torch.Tensor, seq_ce: torch.Tensor, seq_valid: torch.Tensor,
                       seq_log_prob: torch.Tensor, coef: float
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(loss, term, advantage, beta)`` for one micro-batch.

    ``term = lambda sum_b w_b stopgrad(l_b - beta_b) P_b`` keeps its graph
    (the trainer's gradient-split diagnostic differentiates it alone);
    ``loss = ce + (term - stopgrad(term))`` has the VALUE of the sampled CE
    and the gradient of ``ce + term``.  ``advantage`` (``[B]``, detached, 0
    for sequences without valid targets) and ``beta`` are returned for the
    statistics.  ``w_b = n_b / N_valid`` is the weight the CE reduction
    already gives sequence ``b`` (``1/B`` for full windows), so the policy
    term is normalized exactly like the loss it estimates the gradient of.
    """
    beta, has = loo_baseline(seq_ce, seq_valid)
    adv = torch.where(has, seq_ce.detach().double() - beta, torch.zeros_like(beta))
    adv = adv.to(torch.float32)
    w = seq_valid.to(torch.float32) / seq_valid.sum().to(torch.float32)
    term = float(coef) * (w * adv * seq_log_prob.to(torch.float32)).sum()
    loss = ce + (term - term.detach())
    return loss, term, adv, beta.to(torch.float32)


# --------------------------------------------------------------------------- #
# trainer-side state: the noise RNG
# --------------------------------------------------------------------------- #


class GumbelTrainingState:
    """The per-rank training noise generator of a gumbel_policy run.

    The leave-one-out baseline is formed inside each micro-batch and keeps no
    state, so the generator is everything a resume needs beyond the weights
    and the optimizer.
    """

    def __init__(self, device: torch.device, seed: int):
        from .laplace_policy import make_generator
        self.generator = make_generator(device, seed)

    def state_dict(self, generator_states: Optional[List[torch.Tensor]] = None) -> Dict:
        states = (generator_states if generator_states is not None
                  else [self.generator.get_state().cpu()])
        return {"version": GUMBEL_STATE_VERSION, "world_size": len(states),
                "generator_states": [s.cpu() for s in states]}

    def load_state_dict(self, payload: Dict, rank: int, world: int) -> None:
        if int(payload.get("version", -1)) != GUMBEL_STATE_VERSION:
            raise ValueError(f"gumbel_state version {payload.get('version')!r} is not "
                             f"{GUMBEL_STATE_VERSION}")
        if int(payload["world_size"]) != int(world):
            raise ValueError(
                f"gumbel_state holds {payload['world_size']} per-rank noise generator states "
                f"but this run has world size {world}; resuming a gumbel_policy run with a "
                "different number of processes is not supported")
        self.generator.set_state(payload["generator_states"][rank].cpu())
