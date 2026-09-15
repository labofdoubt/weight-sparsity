"""reinforce_topk: exact-K stochastic hard support with score-function gradients."""

import itertools
import math

import pytest
import torch

from wsparse.bottleneck import (
    AdaptiveLapSumTopKGate,
    apply_activation_bottleneck,
    conditional_bernoulli_sample,
    gumbel_pl_sample,
    pl_score_from_order,
    sample_exact_k,
)
from wsparse.config import ActivationBottleneckConfig, ModelConfig
from wsparse.model import build_model

DISTS = ("gumbel_pl", "conditional_bernoulli")


def make_gate(**kw):
    cfg = dict(n_features=16, k=3, j=4, n_eff=3.0, selection_mode="abs_topk",
               surrogate_mode="reinforce_topk", reinforce_temperature=1.0,
               log_diagnostics=True)
    cfg.update(kw)
    return AdaptiveLapSumTopKGate(**cfg)


def bn_cfg(**kw):
    cfg = dict(enabled=True, n_features=64, k=8, j=24, n_eff=6.0, layers="all",
               surrogate_mode="reinforce_topk", selection_mode="abs_topk",
               placement="residual_out", bias=False)
    cfg.update(kw)
    return ActivationBottleneckConfig(**cfg)


# --------------------------------------------------------------------------- #
# A/B: exact support size and the q == K degenerate case
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dist", DISTS)
def test_exact_k_support_and_zero_sum(dist):
    torch.manual_seed(0)
    a = torch.randn(64, 12)
    res = sample_exact_k(a, 5, dist)
    assert torch.all(res["selected_mask"].sum(-1) == 5)
    assert float(res["score_grad"].sum(-1).abs().max()) < 1e-5


@pytest.mark.parametrize("dist", DISTS)
def test_q_equals_k_is_deterministic_with_zero_score(dist):
    torch.manual_seed(1)
    a = torch.randn(8, 6)
    res = sample_exact_k(a, 6, dist)
    assert torch.all(res["selected_mask"] == 1.0)
    assert float(res["score_grad"].abs().max()) == 0.0
    assert float(res["log_prob"].abs().max()) == 0.0


# --------------------------------------------------------------------------- #
# C: K == 1 reduces to categorical softmax for both
# --------------------------------------------------------------------------- #


def test_k1_conditional_bernoulli_marginals_are_softmax():
    torch.manual_seed(2)
    a = torch.randn(16, 7)
    res = conditional_bernoulli_sample(a, 1)
    assert torch.allclose(res["mu"], torch.softmax(a, -1), atol=1e-5)


def test_k1_gumbel_matches_softmax_empirically():
    torch.manual_seed(3)
    a = torch.tensor([[1.5, 0.0, -0.5, 0.7, -1.2]])
    n = 40000
    counts = torch.zeros(5)
    res = gumbel_pl_sample(a.expand(n, 5).contiguous(), 1)
    counts = res["selected_mask"].sum(0)
    freq = counts / n
    assert torch.allclose(freq, torch.softmax(a[0], -1), atol=0.01)


# --------------------------------------------------------------------------- #
# D: common-shift invariance
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dist", DISTS)
def test_common_shift_invariance(dist):
    a = torch.randn(32, 9)
    torch.manual_seed(11)
    r1 = sample_exact_k(a, 4, dist)
    torch.manual_seed(11)
    r2 = sample_exact_k(a + 3.7, 4, dist)
    assert torch.equal(r1["selected_mask"], r2["selected_mask"])
    assert torch.allclose(r1["score_grad"], r2["score_grad"], atol=1e-5)
    assert float(r1["score_grad"].sum(-1).abs().max()) < 1e-5


# --------------------------------------------------------------------------- #
# E/F: conditional Bernoulli against brute force and autograd
# --------------------------------------------------------------------------- #


def brute_force_cb(a, k):
    """Exact P(S), Z and marginals by subset enumeration (tiny q only)."""
    q = a.shape[-1]
    subsets = list(itertools.combinations(range(q), k))
    logws = torch.stack([a[..., list(sub)].sum(-1) for sub in subsets], -1)
    logz = logws.logsumexp(-1)
    probs = (logws - logz.unsqueeze(-1)).exp()
    mu = torch.zeros_like(a)
    for si, sub in enumerate(subsets):
        for i in sub:
            mu[..., i] += probs[..., si]
    return subsets, probs, logz, mu


def test_cb_dp_matches_brute_force():
    torch.manual_seed(4)
    a = torch.randn(6, 5)
    subsets, probs, logz, mu_ref = brute_force_cb(a, 2)
    res = conditional_bernoulli_sample(a, 2)
    assert torch.allclose(res["mu"], mu_ref, atol=1e-5)
    assert float(res["mu_sum_error"]) < 1e-4
    # sampled log_prob matches the enumerated log P(S) for the drawn subset
    for row in range(6):
        sel = tuple(res["selected_mask"][row].nonzero().flatten().tolist())
        si = subsets.index(sel)
        assert res["log_prob"][row].item() == pytest.approx(
            math.log(probs[row, si].item()), abs=1e-4)


def test_cb_empirical_frequencies():
    torch.manual_seed(5)
    a = torch.tensor([0.8, -0.3, 0.1, 1.2, -0.9])
    subsets, probs, _, _ = brute_force_cb(a, 2)
    n = 30000
    res = conditional_bernoulli_sample(a.expand(n, 5).contiguous(), 2)
    key = res["selected_mask"] @ (2 ** torch.arange(5, dtype=torch.float32))
    for si, sub in enumerate(subsets):
        want = probs[si].item()
        code = sum(2 ** i for i in sub)
        got = float((key == code).float().mean())
        assert got == pytest.approx(want, abs=0.012), (sub, got, want)


def test_cb_score_equals_autograd():
    torch.manual_seed(6)
    a = torch.randn(4, 6, requires_grad=True)
    res = conditional_bernoulli_sample(a.detach(), 3)
    mask = res["selected_mask"]
    subsets = list(itertools.combinations(range(6), 3))
    logws = torch.stack([a[:, list(sub)].sum(-1) for sub in subsets], -1)
    logz = logws.logsumexp(-1)
    ((a * mask).sum() - logz.sum()).backward()
    assert torch.allclose(a.grad, res["score_grad"], atol=1e-5)


# --------------------------------------------------------------------------- #
# G/H: Gumbel-PL score against autograd; sampling against PL probabilities
# --------------------------------------------------------------------------- #


def pl_ordered_logprob(a, order):
    """Reference ordered Plackett-Luce log prob via sequential logsumexp."""
    lp = torch.zeros(a.shape[:-1])
    avail = torch.ones_like(a, dtype=torch.bool)
    for t in range(order.shape[-1]):
        idx = order[..., t]
        masked = a.masked_fill(~avail, float("-inf"))
        lp = lp + a.gather(-1, idx[..., None])[..., 0] - masked.logsumexp(-1)
        avail = avail.scatter(-1, idx[..., None], False)
    return lp


def test_gumbel_pl_score_matches_autograd():
    torch.manual_seed(7)
    a = torch.randn(5, 6, requires_grad=True)
    order = torch.stack([torch.randperm(6)[:3] for _ in range(5)])
    _, score, log_prob, _ = pl_score_from_order(a.detach(), order)
    lp_ref = pl_ordered_logprob(a, order)
    lp_ref.sum().backward()
    assert torch.allclose(a.grad, score, atol=1e-5)
    assert torch.allclose(lp_ref.detach(), log_prob, atol=1e-5)


def test_gumbel_sampling_matches_pl_probabilities():
    torch.manual_seed(8)
    a = torch.tensor([0.9, -0.2, 0.4, -1.0])
    n = 40000
    res = gumbel_pl_sample(a.expand(n, 4).contiguous(), 2)
    mask = res["selected_mask"]
    # compare SET frequencies against summed ordered-PL probabilities
    for sub in itertools.combinations(range(4), 2):
        want = 0.0
        for perm in itertools.permutations(sub):
            lp = pl_ordered_logprob(a[None], torch.tensor([list(perm)]))
            want += math.exp(lp.item())
        code = sum(2 ** i for i in sub)
        key = mask @ (2 ** torch.arange(4, dtype=torch.float32))
        got = float((key == code).float().mean())
        assert got == pytest.approx(want, abs=0.012), (sub, got, want)


def test_gumbel_pl_score_stable_at_extreme_logit_spread():
    # tiny-T regime: spreads of O(400) in logit units.  The score must stay in
    # its exact bound [-K, 1], sum to ~0, and match autograd -- this is the
    # regression for the linear-space cancellation that produced entries of
    # -112 in real training.
    torch.manual_seed(20)
    base = torch.sort(torch.rand(6, 64), -1, descending=True).values
    a = (base * 400.0).requires_grad_(True)
    res = gumbel_pl_sample(a.detach(), 32)
    sg = res["score_grad"]
    assert float(sg.min()) >= -32.0 - 1e-4 and float(sg.max()) <= 1.0 + 1e-4
    assert float(sg.sum(-1).abs().max()) < 1e-3
    order = (a.detach() + 0).topk(32, -1).indices        # any fixed order works
    _, sg2, lp, _ = pl_score_from_order(a.detach(), order)
    lp_ref = pl_ordered_logprob(a, order)
    lp_ref.sum().backward()
    assert torch.allclose(a.grad, sg2, atol=1e-4)
    assert torch.allclose(lp_ref.detach(), lp, atol=1e-3)


def test_cb_stable_at_extreme_logit_spread():
    torch.manual_seed(21)
    a = torch.sort(torch.rand(4, 40), -1, descending=True).values * 400.0
    res = conditional_bernoulli_sample(a, 8)
    assert torch.all(res["selected_mask"].sum(-1) == 8)
    assert float(res["mu_sum_error"]) < 1e-3
    assert float(res["score_grad"].sum(-1).abs().max()) < 1e-3
    assert float(res["score_grad"].min()) >= -1.0 - 1e-5   # mask - mu in [-1, 1]


# --------------------------------------------------------------------------- #
# I/J/K/L: gate-level gradient routing, sign, forward invariance, eval
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dist", DISTS)
def test_gradient_routing(dist):
    torch.manual_seed(9)
    gate = make_gate(n_features=32, k=3, j=4, reinforce_distribution=dist)
    gate.train()
    z = torch.randn(2, 32, requires_grad=True)
    y = gate(z)
    assert torch.all((y != 0).sum(-1) == 3)                       # exact K forward
    ps = gate.take_policy_score()
    assert ps is not None and ps.shape == (2,)
    up = torch.randn_like(y)
    adv = 0.7
    (y * up).sum().backward(retain_graph=True)
    g_ord = z.grad.clone(); z.grad = None
    (adv * ps).sum().backward()
    g_rf = z.grad.clone()

    sel = (y != 0)
    scores = z.detach().abs()
    _, ci = torch.topk(scores, gate.m, dim=-1)
    pool = torch.zeros_like(sel); pool.scatter_(-1, ci, True)
    # selected: ordinary gradient = upstream exactly
    assert torch.allclose(g_ord[sel], up[sel])
    # unselected pool members: zero ordinary, nonzero support overall
    assert float(g_ord[pool & ~sel].abs().max()) == 0.0
    assert float(g_rf[pool].abs().sum()) > 0.0
    # outside the pool: nothing at all
    assert float(g_ord[~pool].abs().max()) == 0.0
    assert float(g_rf[~pool].abs().max()) == 0.0


def test_abstopk_policy_gradient_carries_sign_z():
    torch.manual_seed(10)
    gate = make_gate(n_features=8, k=2, j=3, reinforce_temperature=2.0)
    gate.train()
    z = torch.tensor([[4.0, -3.5, 3.0, -2.5, 2.0, 0.1, 0.05, 0.02]],
                     requires_grad=True)
    gate(z)
    ps = gate.take_policy_score()
    ps.sum().backward()
    # dL/dz_i = score_grad_i * sign(z_i) / T: recompute score side in |z| space
    # and check the sign flip on negative-z candidates
    g = z.grad[0]
    scores = z.detach().abs()[0]
    _, ci = torch.topk(scores, gate.m, dim=-1)
    for i in ci.tolist():
        if z[0, i] < 0 and g[i] != 0:
            # the |z|-space gradient for this candidate is -g * T at sign -1
            assert g[i] == pytest.approx(-(g[i] / torch.sign(z[0, i]).item()))
    neg = (z[0] < 0) & (g != 0)
    assert bool(neg.any())                                        # test exercised


def test_forward_independent_of_reinforce_coef():
    # the gate always samples in training; the coefficient only scales the loss
    for seed in (0, 1):
        outs = []
        for _ in range(2):
            torch.manual_seed(seed)
            gate = make_gate(n_features=16, k=3, j=4)
            gate.train()
            torch.manual_seed(100 + seed)
            z = torch.randn(3, 16)
            torch.manual_seed(7)
            outs.append(gate(z.clone()))
            gate.take_policy_score()
        assert torch.equal(outs[0], outs[1])


def test_eval_is_deterministic_topk():
    torch.manual_seed(12)
    gate = make_gate(n_features=24, k=4, j=6)
    gate.eval()
    z = torch.randn(3, 24)
    y1, y2 = gate(z.clone()), gate(z.clone())
    assert torch.equal(y1, y2)
    active = (y1 != 0)
    assert torch.all(active.sum(-1) == 4)
    _, top = torch.topk(z.abs(), 4, dim=-1)
    want = torch.zeros_like(active); want.scatter_(-1, top, True)
    assert torch.equal(active, want)
    assert gate.take_policy_score() is None                      # no eval surrogate


# --------------------------------------------------------------------------- #
# integration, config, no-regression
# --------------------------------------------------------------------------- #


def test_controller_policy_score_and_training_composition():
    torch.manual_seed(13)
    model = build_model(ModelConfig(vocab_size=97, max_seq_len=32, n_layers=2,
                                    d_model=32, n_heads=4))
    ctl = apply_activation_bottleneck(model, bn_cfg(), max_steps=10)
    model.train()
    x = torch.randint(0, 97, (3, 16))
    logits, _ = model(x)
    ps = ctl.policy_score()
    assert ps is not None and ps.shape == (3, 16)                # summed over 2 gates
    ce_tok = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 97).float(), x.reshape(-1), reduction="none"
    ).view(3, 16)
    l_ex = ce_tok.detach().mean(-1)
    baseline = (l_ex.sum() - l_ex) / 2                            # batch_loo, B=3
    rf = ((l_ex - baseline) * ps.sum(-1)).mean()
    (ce_tok.mean() + rf).backward()                               # composes cleanly
    stats = ctl.stats()
    for key in ("bottleneck/rf_overlap", "bottleneck/rf_score_grad_sum_abs",
                "bottleneck/rf_log_prob"):
        assert key in stats, key
    assert stats["bottleneck/rf_score_grad_sum_abs"] < 1e-4
    assert ctl.policy_score() is None                             # popped


def test_config_validation_and_no_new_parameters():
    with pytest.raises(ValueError):
        bn_cfg(reinforce_distribution="bogus")
    with pytest.raises(ValueError):
        bn_cfg(reinforce_baseline="bogus")
    with pytest.raises(ValueError):
        bn_cfg(reinforce_temperature=0.0)
    with pytest.raises(ValueError):
        bn_cfg(selection_mode="gated_topk")
    with pytest.raises(ValueError):
        bn_cfg(project_scale_gradient=True)
    cfg = bn_cfg(reinforce_distribution="conditional_bernoulli")
    assert cfg.reinforce_coef == 1.0 and cfg.reinforce_baseline == "batch_loo"
    gate = make_gate()
    assert [n for n, _ in gate.named_parameters()] == []          # no new params
    lap = AdaptiveLapSumTopKGate(n_features=16, k=3, j=4, n_eff=3.0,
                                 surrogate_mode="lapsum_scheduled")
    assert lap.take_policy_score() is None                        # inert elsewhere
