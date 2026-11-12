"""learned_score_topk and abs_topk with gumbel_policy (wsparse.bottleneck.gumbel_policy).

The ordered Plackett-Luce log-probability against a direct masked-logsumexp
loop and against normalization; the Gumbel-Top-K sampling frequencies (all N
and the Top(K+J) pool) against the Plackett-Luce law; the unbiasedness of the
score-function gradient on an enumerable objective; the gate's exact-K forward,
deterministic evaluation and gradient routing (the support gradient
sign(v) dL/dy to the selected scores through the forward, policy gradient to
every score -- or every pool member); the same under abs_topk; the
leave-one-out baseline and the loss; scoring from the residual stream in both
stacks; the config rules; and a training run with a resume.
"""
import itertools
import math

import numpy as np
import pytest
import torch

from wsparse.bottleneck import apply_activation_bottleneck
from wsparse.bottleneck.gate import AdaptiveLapSumTopKGate
from wsparse.bottleneck.gumbel_policy import (gumbel_policy_loss, gumbel_topk, loo_baseline,
                                              ordered_log_prob, sample_gumbel)
from wsparse.bottleneck.laplace_policy import (PolicyCollector, PolicyForwardSettings,
                                               PolicyRecord, make_generator, policy_forward)
from wsparse.config import ActivationBottleneckConfig, Config, ModelConfig
from wsparse.model import build_model
from wsparse.train import train

CPU = torch.device("cpu")


def gate_of(**kw):
    cfg = dict(n_features=16, k=3, j=5, selection_mode="learned_score_topk",
               surrogate_mode="hard", gumbel_policy=True, gumbel_temperature=0.5)
    cfg.update(kw)
    g = AdaptiveLapSumTopKGate(**cfg)
    g.train()
    return g


def sampled(gate, s, v, seed=0, collector=True):
    col = PolicyCollector() if collector else None
    with policy_forward([gate], PolicyForwardSettings(sample=True, collector=col,
                                                       generator=make_generator(CPU, seed))):
        y = gate(s, v)
    return y, col


def naive_log_prob(z, order):
    """The definition: sum_r [z_{i_r} - logsumexp over the not-yet-selected]."""
    remaining = list(range(z.shape[-1]))
    total = z.new_zeros(())
    for i in order:
        total = total + z[i] - torch.logsumexp(z[remaining], dim=0)
        remaining.remove(i)
    return total


# --------------------------------------------------------------------------- #
# log-probability
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scale", [1.0, 1e3])
def test_ordered_log_prob_matches_the_masked_logsumexp_definition(scale):
    torch.manual_seed(0)
    z = (torch.randn(5, 12, dtype=torch.float64) * scale).requires_grad_()
    idx = torch.stack([torch.randperm(12)[:4] for _ in range(5)])
    fast = ordered_log_prob(z, idx)
    ref = torch.stack([naive_log_prob(z[r], idx[r].tolist()) for r in range(5)])
    assert torch.isfinite(fast).all()
    torch.testing.assert_close(fast, ref, rtol=1e-10, atol=1e-8 * scale)
    g_fast, = torch.autograd.grad(fast.sum(), z)
    g_ref, = torch.autograd.grad(ref.sum(), z)
    torch.testing.assert_close(g_fast, g_ref, rtol=1e-9, atol=1e-9)


def test_ordered_log_prob_is_a_distribution_over_ordered_selections():
    torch.manual_seed(1)
    z = torch.randn(6, dtype=torch.float64)
    for k in (1, 2, 3, 6):
        orders = torch.tensor(list(itertools.permutations(range(6), k)))
        lp = ordered_log_prob(z.expand(len(orders), 6), orders)
        assert abs(float(lp.exp().sum()) - 1.0) < 1e-12


def test_log_prob_is_finite_and_differentiable_at_extreme_scores_in_float32():
    s = torch.tensor([[3e4, -3e4, 1e4, 0.0, 2e4, -1e4, 5.0, 7.0]], requires_grad=True)
    draw = gumbel_topk(s, 3, 1e-3, make_generator(CPU, 0))
    lp = ordered_log_prob(draw.pool_logits, draw.pos)
    lp.sum().backward()
    assert torch.isfinite(lp).all() and torch.isfinite(s.grad).all()


def test_gumbel_noise_is_finite():
    g = sample_gumbel((200_000,), make_generator(CPU, 3))
    assert torch.isfinite(g).all()
    assert abs(float(g.mean()) - 0.5772) < 0.01            # Euler-Mascheroni
    assert abs(float(g.var()) - math.pi ** 2 / 6) < 0.03


# --------------------------------------------------------------------------- #
# sampling law
# --------------------------------------------------------------------------- #


def _frequency_check(draw_idx, probs_of, n_draws):
    counts = {}
    for row in draw_idx.tolist():
        counts[tuple(row)] = counts.get(tuple(row), 0) + 1
    for order, p in probs_of.items():
        f = counts.get(order, 0) / n_draws
        se = math.sqrt(p * (1 - p) / n_draws)
        assert abs(f - p) < 5 * se + 1e-12, (order, f, p)
    assert set(counts) <= set(probs_of)


def test_gumbel_topk_samples_the_ordered_plackett_luce_law():
    s = torch.tensor([0.3, -0.4, 1.1, 0.0])
    tau, n_draws = 0.7, 200_000
    draw = gumbel_topk(s.expand(n_draws, 4), 2, tau, make_generator(CPU, 11))
    orders = list(itertools.permutations(range(4), 2))
    lp = ordered_log_prob((s / tau).double().expand(len(orders), 4), torch.tensor(orders))
    _frequency_check(draw.idx, dict(zip(orders, lp.exp().tolist())), n_draws)


def test_top_kj_pool_samples_plackett_luce_restricted_to_the_pool():
    s = torch.tensor([0.3, -0.4, 1.1, 0.0, -2.0, 0.6])
    tau, n_draws, k, pool = 0.7, 200_000, 2, 4
    draw = gumbel_topk(s.expand(n_draws, 6), k, tau, make_generator(CPU, 12), pool=pool)
    cand = torch.topk(s, pool).indices.tolist()                  # [2, 5, 0, 3]
    assert set(draw.idx.flatten().tolist()) <= set(cand)
    orders = list(itertools.permutations(range(pool), k))
    lp = ordered_log_prob((s[cand] / tau).double().expand(len(orders), pool),
                          torch.tensor(orders))
    probs = {tuple(cand[i] for i in o): p for o, p in zip(orders, lp.exp().tolist())}
    _frequency_check(draw.idx, probs, n_draws)


@pytest.mark.parametrize("pool", [None, 4])
def test_score_function_gradient_is_unbiased_on_an_enumerable_objective(pool):
    """E[L(S) grad log p(order)] = grad sum_S P(S) L(S), with L a fixed table of sets
    (the pool held fixed, as the estimator assumes)."""
    torch.manual_seed(5)
    n, k, tau = 6, 2, 0.8
    s0 = torch.randn(n, dtype=torch.float64)
    table = {frozenset(c): float(v) for c, v in zip(itertools.combinations(range(n), k),
                                                   torch.randn(15).tolist())}
    cand = (torch.topk(s0, pool).indices if pool else torch.arange(n)).tolist()
    s = s0.clone().requires_grad_()
    orders = torch.tensor(list(itertools.permutations(range(len(cand)), k)))
    lp = ordered_log_prob((s[cand] / tau).expand(len(orders), len(cand)), orders)
    loss = [table[frozenset(cand[i] for i in o)] for o in orders.tolist()]
    exact, = torch.autograd.grad((lp.exp() * torch.tensor(loss, dtype=torch.float64)).sum(), s)

    n_draws = 400_000
    sf = s0.float().clone().requires_grad_()
    draw = gumbel_topk(sf.expand(n_draws, n), k, tau, make_generator(CPU, 21), pool=pool)
    lp_d = ordered_log_prob(draw.pool_logits, draw.pos)
    rew = torch.tensor([table[frozenset(r)] for r in draw.idx.tolist()])
    per_draw = []
    for chunk in torch.arange(n_draws).chunk(8):   # standard error from 8 batch means
        g, = torch.autograd.grad((rew[chunk] * lp_d[chunk]).sum() / len(chunk), sf,
                                 retain_graph=True)
        per_draw.append(g.double())
    est = torch.stack(per_draw)
    se = est.std(0) / math.sqrt(len(per_draw))
    assert ((est.mean(0) - exact).abs() <= 5 * se + 1e-4).all(), (est.mean(0), exact, se)
    if pool:
        outside = [i for i in range(n) if i not in cand]
        assert (est[:, outside] == 0).all()


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("pool", ["all", "top_kj"])
def test_gate_exact_k_and_deterministic_eval(pool):
    g = gate_of(gumbel_pool=pool)
    torch.manual_seed(0)
    s, v = torch.randn(4, 7, 16), torch.randn(4, 7, 16)
    y, _ = sampled(g, s, v)
    mask = y != 0
    assert (mask.sum(-1) == 3).all()
    assert torch.equal(y, torch.where(mask, v, torch.zeros_like(v)))      # values unscaled
    if pool == "top_kj":
        cand = torch.zeros_like(mask).scatter(-1, torch.topk(s, 8, dim=-1).indices, True)
        assert not (mask & ~cand).any()
    g.eval()
    clean = torch.zeros_like(mask).scatter(-1, torch.topk(s, 3, dim=-1).indices, True)
    for _ in range(3):
        assert torch.equal(g(s, v) != 0, clean)
    g.train()
    with policy_forward([g], PolicyForwardSettings(sample=False)):
        assert torch.equal(g(s, v) != 0, clean)
    # a sampled draw differs from the clean support somewhere
    assert not torch.equal(mask, clean)


@pytest.mark.parametrize("support_grad", [True, False])
@pytest.mark.parametrize("pool", ["all", "top_kj"])
def test_gradient_routing(pool, support_grad):
    """The forward gives the values m * g and the selected scores sign(v) g
    (nothing with the support gradient off); the recorded log-probability
    reaches every score (all N, or every pool member)."""
    g = gate_of(gumbel_pool=pool, score_support_grad=support_grad)
    torch.manual_seed(2)
    s = torch.randn(3, 5, 16, requires_grad=True)
    v = torch.randn(3, 5, 16, requires_grad=True)
    y, col = sampled(g, s, v)
    mask = y != 0
    assert torch.equal(y, torch.where(mask, v, torch.zeros_like(v)))
    up = torch.randn_like(y)
    gs, gv = torch.autograd.grad((y * up).sum(), (s, v), allow_unused=True, retain_graph=True)
    if support_grad:
        torch.testing.assert_close(gs, up * v.detach().sign() * mask)
    else:
        assert gs is None or (gs == 0).all()
    torch.testing.assert_close(gv, up * mask)
    assert len(col.records) == 1 and col.records[0].valid_only
    lp, = torch.autograd.grad(col.records[0].log_prob.sum(), s)
    if pool == "all":
        assert (lp != 0).all()
    else:
        cand = torch.zeros_like(s, dtype=torch.bool).scatter(-1, torch.topk(s, 8, -1).indices, True)
        assert (lp[cand] != 0).all() and (lp[~cand] == 0).all()


@pytest.mark.parametrize("support_grad", [True, False])
def test_deterministic_learned_score_support_gradient(support_grad):
    """gumbel_policy off: TopK(s) in training, forward v * m, and the selected
    scores get sign(v) dL/dy (the abs_topk signal); unselected ones nothing."""
    g = gate_of(gumbel_policy=False, gumbel_temperature=1.0, score_support_grad=support_grad)
    torch.manual_seed(4)
    s = torch.randn(3, 5, 16, requires_grad=True)
    v = torch.randn(3, 5, 16, requires_grad=True)
    y = g(s, v)
    clean = torch.zeros_like(s, dtype=torch.bool).scatter(-1, torch.topk(s, 3, -1).indices, True)
    assert torch.equal(y, torch.where(clean, v, torch.zeros_like(v)))
    up = torch.randn_like(y)
    gs, gv = torch.autograd.grad((y * up).sum(), (s, v), allow_unused=True)
    torch.testing.assert_close(gv, up * clean)
    if support_grad:
        torch.testing.assert_close(gs, up * v.detach().sign() * clean)
    else:
        assert gs is None or (gs == 0).all()


@pytest.mark.parametrize("pool", ["all", "top_kj"])
def test_abs_topk_gate_samples_the_magnitudes(pool):
    """abs_topk: the policy is over |a|, exactly K kept unscaled, the
    log-probability reaches a through |a|, evaluation is TopK(|a|)."""
    g = AdaptiveLapSumTopKGate(n_features=16, k=3, j=5, selection_mode="abs_topk",
                               surrogate_mode="hard", gumbel_policy=True,
                               gumbel_temperature=0.5, gumbel_pool=pool)
    g.train()
    torch.manual_seed(5)
    a = torch.randn(3, 5, 16, requires_grad=True)
    col = PolicyCollector()
    with policy_forward([g], PolicyForwardSettings(sample=True, collector=col,
                                                   generator=make_generator(CPU, 0))):
        y = g(a)
    mask = y != 0
    assert (mask.sum(-1) == 3).all()
    assert torch.equal(y, torch.where(mask, a, torch.zeros_like(a)))
    if pool == "top_kj":
        cand = torch.zeros_like(mask).scatter(-1, torch.topk(a.abs(), 8, -1).indices, True)
        assert not (mask & ~cand).any()
    up = torch.randn_like(y)
    ga, = torch.autograd.grad((y * up).sum(), a, retain_graph=True)
    torch.testing.assert_close(ga, up * mask)                  # the plain hard-mask gradient
    lp, = torch.autograd.grad(col.records[0].log_prob.sum(), a)
    if pool == "all":
        assert (lp != 0).all()
    else:
        assert (lp[cand] != 0).all() and (lp[~cand] == 0).all()
    # through |a|: the same draw from the magnitudes as a leaf, d/da = sign(a) d/d|a|
    m = a.detach().abs().requires_grad_()
    col2 = PolicyCollector()
    with policy_forward([g], PolicyForwardSettings(sample=True, collector=col2,
                                                   generator=make_generator(CPU, 0))):
        assert torch.equal(g(m) != 0, mask)
    lp_m, = torch.autograd.grad(col2.records[0].log_prob.sum(), m)
    torch.testing.assert_close(lp, a.detach().sign() * lp_m)
    g.eval()
    clean = torch.zeros_like(mask).scatter(-1, torch.topk(a.abs(), 3, -1).indices, True)
    assert torch.equal(g(a) != 0, clean)


def test_no_record_without_collector_or_grad():
    g = gate_of()
    s, v = torch.randn(2, 3, 16), torch.randn(2, 3, 16)
    _, col = sampled(g, s, v, collector=False)
    assert col is None
    col = PolicyCollector()
    with torch.no_grad(), policy_forward([g], PolicyForwardSettings(sample=True, collector=col)):
        g(s, v)
    assert col.records == []


def test_valid_mask_in_the_sequence_sum():
    lp = torch.arange(6.0).reshape(2, 3)
    valid = torch.tensor([[True, False, True], [True, True, False]])
    col = PolicyCollector()
    col.add(PolicyRecord(gate=None, log_prob=lp, valid_only=True))
    torch.testing.assert_close(col.sequence_log_prob(valid), torch.tensor([2.0, 7.0]))
    with pytest.raises(ValueError, match="target mask"):
        col.sequence_log_prob()
    lap = PolicyCollector()  # laplace_policy records keep every position
    lap.add(PolicyRecord(gate=None, log_prob=lp))
    torch.testing.assert_close(lap.sequence_log_prob(valid), torch.tensor([3.0, 12.0]))


# --------------------------------------------------------------------------- #
# baseline and loss
# --------------------------------------------------------------------------- #


def test_leave_one_out_baseline():
    seq_ce = torch.tensor([1.0, 2.0, 4.0, 9.0])
    beta, has = loo_baseline(seq_ce, torch.tensor([5, 5, 5, 5]))
    torch.testing.assert_close(beta.float(), torch.tensor([5.0, 14 / 3, 4.0, 7 / 3]))
    # a sequence without valid targets is nobody's peer and has no baseline
    beta, has = loo_baseline(seq_ce, torch.tensor([5, 0, 5, 5]))
    torch.testing.assert_close(beta.float(), torch.tensor([6.5, 0.0, 5.0, 2.5]))
    assert has.tolist() == [True, False, True, True]
    with pytest.raises(ValueError, match="at least two"):
        loo_baseline(torch.tensor([3.0]), torch.tensor([7]))
    with pytest.raises(ValueError, match="at least two"):
        loo_baseline(torch.tensor([3.0, 1.0]), torch.tensor([7, 0]))


def test_policy_loss_value_and_gradient():
    seq_ce = torch.tensor([1.0, 2.0, 4.0], requires_grad=True)
    seq_valid = torch.tensor([4, 4, 4])
    ce = seq_ce.mean()
    lp = torch.tensor([-3.0, -1.0, -2.0], requires_grad=True)
    loss, term, adv, beta = gumbel_policy_loss(ce, seq_ce, seq_valid, lp, coef=0.1)
    assert float(loss.detach()) == pytest.approx(float(ce.detach()))  # value = sampled CE
    torch.testing.assert_close(adv, seq_ce.detach() - torch.tensor([3.0, 2.5, 1.5]))
    g_ce, g_lp = torch.autograd.grad(loss, (seq_ce, lp))
    torch.testing.assert_close(g_ce, torch.full((3,), 1 / 3))  # no gradient through A_b
    torch.testing.assert_close(g_lp, 0.1 * adv / 3)
    assert float(term) == pytest.approx(float(0.1 * (adv * lp).sum() / 3))


# --------------------------------------------------------------------------- #
# model: both stacks score from the residual stream
# --------------------------------------------------------------------------- #


def model_of(code_residual, n_layers=3, **bn):
    torch.manual_seed(0)
    kw = dict(enabled=True, n_features=64, k=8, j=8, layers="all", placement="residual_out",
              selection_mode="learned_score_topk", surrogate_mode="hard", gumbel_policy=True,
              gumbel_temperature=0.5, code_residual=code_residual)
    kw.update(bn)
    model = build_model(ModelConfig(vocab_size=61, max_seq_len=16, n_layers=n_layers,
                                    d_model=32, n_heads=4))
    ctl = apply_activation_bottleneck(model, ActivationBottleneckConfig(**kw))
    return model, ctl


def topk_mask(s, k):
    return torch.zeros_like(s).scatter(-1, torch.topk(s, k, dim=-1).indices, 1.0)


@torch.no_grad()
def reference_logits(model, idx, code_residual, k):
    """The eval forward written out: support = TopK(W_s h + b_s) of the stream state."""
    x = model.tok_emb(idx) * model.embed_scale
    if model.pos_emb is not None:
        x = x + model.pos_emb(torch.arange(idx.shape[1]))[None]
    if code_residual:
        code = None
        for block in model.blocks:
            bot = block.residual_out_bottleneck
            if code is None:
                h = block.body(x)
                code = bot.in_proj(h) * topk_mask(bot.score_proj(h), k)
            else:
                delta = block.branches(x)
                code = (code + bot.in_proj(delta)) * topk_mask(bot.score_proj(x + delta), k)
            x = bot.post_norm(bot.decode(code))
    else:
        for block in model.blocks:
            h = block.body(x)
            bot = block.residual_out_bottleneck
            x = bot.post_norm(bot.decode(bot.in_proj(h) * topk_mask(bot.score_proj(h), k)))
    return model.lm_head(model.norm_f(x)) * model.logit_mult


@pytest.mark.parametrize("code_residual", [False, True])
def test_eval_forward_scores_from_the_residual_stream(code_residual):
    model, _ = model_of(code_residual)
    model.eval()
    idx = torch.randint(0, 61, (2, 16))
    logits, _ = model(idx)
    torch.testing.assert_close(logits, reference_logits(model, idx, code_residual, 8))


@pytest.mark.parametrize("code_residual", [False, True])
@pytest.mark.parametrize("pool", ["all", "top_kj"])
def test_training_forward_records_one_term_per_gate(code_residual, pool):
    model, ctl = model_of(code_residual, gumbel_pool=pool)
    assert ctl.gumbel_active and len(ctl.gumbel_gates) == 3
    model.train()
    idx = torch.randint(0, 61, (4, 16))
    y = idx.roll(-1, 1)
    y[0, 10:] = -100  # ignored targets: their positions leave the sum
    out = model(idx, y, return_loss_details=True,
                policy=PolicyForwardSettings(sample=True, generator=make_generator(CPU, 1)))
    assert len(out.policy_records) == 3
    manual = sum(torch.where(out.valid, r.log_prob, torch.zeros(())).sum(-1)
                 for r in out.policy_records)
    torch.testing.assert_close(out.policy_log_prob, manual)
    for _, layer in ctl.layers:
        assert layer.gate.diagnostics["active_count"] == 8.0
    loss, *_ = gumbel_policy_loss(out.ce, out.seq_ce, out.seq_valid, out.policy_log_prob, 1.0)
    loss.backward()
    for p in ctl.scorer_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0


@pytest.mark.parametrize("code_residual", [False, True])
def test_abs_topk_training_forward_reaches_the_encoders(code_residual):
    """abs_topk + gumbel_policy: one record per gate, and the policy term alone
    reaches every encoder (under code_residual also through the carry)."""
    model, ctl = model_of(code_residual, selection_mode="abs_topk", gumbel_pool="top_kj")
    assert ctl.gumbel_active and ctl.scorer_parameters() == []
    model.train()
    idx = torch.randint(0, 61, (4, 16))
    out = model(idx, idx.roll(-1, 1), return_loss_details=True,
                policy=PolicyForwardSettings(sample=True, generator=make_generator(CPU, 1)))
    assert len(out.policy_records) == 3
    for _, layer in ctl.layers:
        assert layer.gate.diagnostics["active_count"] == 8.0
    encoders = [layer.in_proj.weight for _, layer in ctl.layers]
    grads = torch.autograd.grad(out.policy_log_prob.sum(), encoders)
    assert all(gr.abs().sum() > 0 for gr in grads)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def ab(**kw):
    base = dict(enabled=True, n_features=64, k=8, j=8, selection_mode="learned_score_topk",
                surrogate_mode="hard")
    base.update(kw)
    return ActivationBottleneckConfig(**base)


@pytest.mark.parametrize("kw, match", [
    (dict(surrogate_mode="rblapsum"), "selection_mode 'topk' or 'abs_topk'"),
    (dict(surrogate_mode="lapsum"), "surrogate_mode='hard'"),
    (dict(selection_mode="topk", gumbel_policy=True), "'learned_score_topk' or 'abs_topk'"),
    (dict(selection_mode="abs_topk", surrogate_mode="lapsum", gumbel_policy=True),
     "surrogate_mode='hard'"),
    (dict(selection_mode="abs_topk", learned_score_support_grad=False),
     "learned_score_topk' only"),
    (dict(selection_mode="abs_topk", gumbel_policy=True, stochastic_width="uniform"),
     "stochastic_width"),
    (dict(gumbel_temperature=0.3), "gumbel_policy: true only"),
    (dict(gumbel_policy_coef=0.01), "gumbel_policy: true only"),
    (dict(gumbel_pool="top_kj"), "gumbel_policy: true only"),
    (dict(gumbel_policy=True, gumbel_temperature=0.0), "positive"),
    (dict(gumbel_policy=True, gumbel_policy_coef=-1.0), ">= 0"),
    (dict(gumbel_policy=True, gumbel_pool="top_kj", j=0), "j >= 1"),
    (dict(gumbel_policy=True, gumbel_pool="nearest"), "unknown gumbel_pool"),
    (dict(k=64, j=0), "k < n_features"),
    (dict(tie_encoder_decoder=True, init_mode="unit_norm_dictionary"), "separate matrices"),
    (dict(hard_k_annealing=True), "hard_k_annealing"),
])
def test_config_rejects(kw, match):
    with pytest.raises(ValueError, match=match):
        ab(**kw)


def test_config_accepts_and_defaults_are_unchanged():
    ab(gumbel_policy=True, gumbel_pool="top_kj", code_residual=True,
       placement="residual_out", gumbel_policy_coef=0.0)
    ab(selection_mode="abs_topk", gumbel_policy=True, gumbel_pool="top_kj",
       code_residual=True, placement="residual_out")
    ab(learned_score_support_grad=False)
    d = ActivationBottleneckConfig()
    assert (d.selection_mode, d.gumbel_policy, d.gumbel_pool) == ("abs_topk", False, "all")
    assert d.learned_score_support_grad is True


def test_config_needs_two_sequences_per_micro_batch():
    cfg = Config()
    cfg.train.batch_size, cfg.train.micro_batch_size = 4, 1
    cfg.activation_bottleneck = ab(gumbel_policy=True)
    with pytest.raises(ValueError, match="micro_batch_size >= 2"):
        cfg.__post_init__()


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #


def make_data(tmp_path, vocab=64):
    import json
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for name, n in (("train", 60_000), ("val", 8_000)):
        arr = np.memmap(d / f"{name}.bin", dtype=np.uint16, mode="w+", shape=(n,))
        arr[:] = rng.integers(0, vocab, size=n).astype(np.uint16)
        arr.flush()
        del arr
    with open(d / "meta.json", "w") as f:
        json.dump({"vocab_size": vocab, "eos_id": 0, "tokenizer": "fake",
                   "train_tokens": 60_000}, f)
    return str(d)


def train_cfg(tmp_path, name, max_steps=4, **bn):
    cfg = Config()
    cfg.data.data_dir = make_data(tmp_path)
    cfg.data.seq_len = 32
    cfg.model = ModelConfig(vocab_size=64, max_seq_len=32, n_layers=2, d_model=32, n_heads=4)
    cfg.train.batch_size = 8
    cfg.train.micro_batch_size = 4
    cfg.train.max_steps = max_steps
    cfg.train.warmup_steps = 2
    cfg.train.log_every_steps = 2
    cfg.train.validate_every_steps = 2
    cfg.train.val_batches = 2
    cfg.train.checkpoint_every_steps = 2
    cfg.train.dtype = "float32"
    cfg.train.device = "cpu"
    cfg.train.tensorboard = False
    cfg.train.out_dir = str(tmp_path / "runs")
    cfg.train.run_name = name
    kw = dict(enabled=True, n_features=128, k=16, j=48, layers="all", placement="residual_out",
              selection_mode="learned_score_topk", surrogate_mode="hard", gumbel_policy=True,
              gumbel_temperature=0.5, gumbel_policy_coef=0.1)
    kw.update(bn)
    cfg.activation_bottleneck = ActivationBottleneckConfig(**kw)
    return cfg


@pytest.mark.parametrize("code_residual, pool, mode", [
    (False, "all", "learned_score_topk"), (True, "top_kj", "learned_score_topk"),
    (True, "top_kj", "abs_topk")])
def test_training_run_and_resume(tmp_path, code_residual, pool, mode):
    name = f"g_{code_residual}_{mode}"
    cfg = train_cfg(tmp_path, name, code_residual=code_residual, gumbel_pool=pool,
                    selection_mode=mode)
    summary = train(cfg)
    assert math.isfinite(summary["val/ce"]) and math.isfinite(summary["train/ce"])
    assert summary["bottleneck/active_count"] == 16.0
    assert summary["gumbel/nonfinite"] == 0.0
    assert summary["gumbel/grad_split/policy_norm"] > 0
    if mode == "learned_score_topk":
        # the scorer gets both the support gradient and the policy term
        assert summary["gumbel/grad_split/ce_norm_scorer"] > 0
        assert summary["gumbel/grad_split/policy_norm_scorer"] > 0
        assert summary["gumbel/scorer_grad_norm"] > 0
    else:
        # no scorer: the policy term goes to the encoders and the trunk
        assert summary["gumbel/grad_split/policy_norm_scorer"] == 0
        assert summary["gumbel/grad_split/policy_norm_rest"] > 0
    assert "val_stochastic/ce" in summary
    ckpt = torch.load(f"{cfg.train.out_dir}/{cfg.train.run_name}/latest.pt",
                      weights_only=False)
    assert ckpt["gumbel_state"]["world_size"] == 1
    # resume to step 6 from the step-4 checkpoint
    cfg2 = train_cfg(tmp_path, name, max_steps=6, code_residual=code_residual,
                     gumbel_pool=pool, selection_mode=mode)
    cfg2.train.resume = "auto"
    summary2 = train(cfg2)
    assert math.isfinite(summary2["val/ce"])


def test_resume_without_gumbel_state_is_rejected(tmp_path):
    cfg = train_cfg(tmp_path, "nostate")
    train(cfg)
    path = f"{cfg.train.out_dir}/{cfg.train.run_name}/latest.pt"
    payload = torch.load(path, weights_only=False)
    del payload["gumbel_state"]
    torch.save(payload, path)
    cfg2 = train_cfg(tmp_path, "nostate", max_steps=6)
    cfg2.train.resume = "auto"
    with pytest.raises(ValueError, match="gumbel_state"):
        train(cfg2)


@pytest.mark.parametrize("support_grad", [True, False])
def test_coefficient_zero_leaves_the_support_gradient(tmp_path, support_grad):
    """lambda = 0: the scorer trains through the support gradient alone, and
    not at all with that off too."""
    cfg = train_cfg(tmp_path, f"coef0_{support_grad}", gumbel_policy_coef=0.0,
                    learned_score_support_grad=support_grad)
    summary = train(cfg)
    if support_grad:
        assert summary["gumbel/scorer_grad_norm"] > 0
    else:
        assert summary["gumbel/scorer_grad_norm"] == 0.0
    assert math.isfinite(summary["val/ce"])


@pytest.mark.parametrize("inp, grad", [("raw", True), ("rmsnorm", True), ("raw", False),
                                       ("rmsnorm", False)])
def test_scorer_input_options(inp, grad):
    """learned_score_input: s = W_s h + b_s or W_s RMSNorm(h) + b_s; with
    learned_score_input_grad false nothing reaches h through the scorer."""
    model, ctl = model_of(False, gumbel_policy=False, gumbel_temperature=1.0,
                          learned_score_input=inp, learned_score_input_grad=grad)
    bot = model.blocks[0].residual_out_bottleneck
    h = torch.randn(2, 5, 32, requires_grad=True)
    s = bot.score(h)
    ref = torch.nn.functional.rms_norm(h, (32,), eps=1e-6) if inp == "rmsnorm" else h
    torch.testing.assert_close(s, bot.score_proj(ref))
    if grad:
        g, = torch.autograd.grad(s.sum(), h)
        assert g.abs().sum() > 0
    else:
        assert not s.requires_grad or s.grad_fn is not None
        g = torch.autograd.grad(s.sum(), h, allow_unused=True)[0]
        assert g is None
    with pytest.raises(ValueError, match="learned_score_topk' only"):
        ab(selection_mode="abs_topk", learned_score_input="rmsnorm")


def test_only_the_listed_gates_sample():
    """PolicyForwardSettings.only (gumbel_gates_per_forward): the listed gates
    draw and record, the others take their deterministic Top-K."""
    model, ctl = model_of(True, selection_mode="abs_topk", gumbel_pool="top_kj", n_layers=4)
    model.train()
    gates = [layer.gate for _, layer in ctl.layers]
    idx = torch.randint(0, 61, (4, 16))
    only = frozenset({id(gates[1]), id(gates[3])})
    out = model(idx, idx.roll(-1, 1), return_loss_details=True,
                policy=PolicyForwardSettings(sample=True, generator=make_generator(CPU, 1),
                                             only=only))
    assert [r.gate for r in out.policy_records] == [gates[1], gates[3]]


def test_gate_subset_is_uniform_and_exact():
    from wsparse.train import gate_subset
    gates = [object() for _ in range(8)]
    gen = torch.Generator().manual_seed(0)
    counts = {id(g): 0 for g in gates}
    for _ in range(4000):
        sub = gate_subset(gates, 3, gen)
        assert len(sub) == 3
        for i in sub:
            counts[i] += 1
    freq = np.array(list(counts.values())) / 4000
    assert np.allclose(freq, 3 / 8, atol=0.03)


def test_training_with_gates_per_forward_and_resume(tmp_path):
    cfg = train_cfg(tmp_path, "gpf", code_residual=True, gumbel_pool="top_kj",
                    selection_mode="abs_topk", gumbel_gates_per_forward=1)
    summary = train(cfg)
    assert math.isfinite(summary["val/ce"]) and "val_stochastic/ce" in summary
    cfg2 = train_cfg(tmp_path, "gpf", max_steps=6, code_residual=True, gumbel_pool="top_kj",
                     selection_mode="abs_topk", gumbel_gates_per_forward=1)
    cfg2.train.resume = "auto"
    assert math.isfinite(train(cfg2)["val/ce"])
    bad = train_cfg(tmp_path, "gpf_bad", selection_mode="abs_topk", gumbel_gates_per_forward=2)
    with pytest.raises(ValueError, match="must be below"):
        train(bad)


def test_gates_per_forward_config_rules():
    with pytest.raises(ValueError, match="gumbel_policy: true only"):
        ab(gumbel_gates_per_forward=3)
    with pytest.raises(ValueError, match=">= 0"):
        ab(gumbel_policy=True, gumbel_gates_per_forward=-1)
    ab(gumbel_policy=True, gumbel_gates_per_forward=3)
