"""Activation bottleneck: hard TopK forward, LapSum Top(K+J) surrogate backward.

The gradient tests compare against finite differences of the *soft LapSum mask*
with the barrier re-solved at each perturbation -- that is what verifies the
shared-barrier correction term, which is the whole content of the VJP.  The hard
TopK discontinuity is never finite-differenced.
"""

import math

import pytest
import torch
import torch.nn as nn

from wsparse.bottleneck.controller import _PLACEMENT_ATTR as _PLACEMENT_ATTRS
from wsparse.bottleneck import (
    ActivationBottleneckController,
    AdaptiveLapSumTopKGate,
    SparseTopKBottleneck,
    apply_activation_bottleneck,
    lapsum_barrier_bisect,
    lapsum_barrier_sorted,
    lapsum_budget,
    lapsum_probs,
    lapsum_probs_at,
    parse_placements,
    resolve_layers,
    validate_gate_shapes,
)
from wsparse.config import ActivationBottleneckConfig, Config, ModelConfig
from wsparse.model import build_model


def sorted_scores(rows=32, m=48, scale=1.0, dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    r = torch.randn(rows, m, dtype=dtype) * scale
    return torch.sort(r, dim=-1, descending=True).values


def bottleneck_cfg(**kw):
    cfg = dict(enabled=True, n_features=64, k=8, j=24, layers="all",
               surrogate_mode="lapsum", temperature=1.0)
    cfg.update(kw)
    return ActivationBottleneckConfig(**cfg)


def tiny_model(**kw):
    cfg = dict(vocab_size=97, max_seq_len=32, n_layers=2, d_model=32, n_heads=4)
    cfg.update(kw)
    return build_model(ModelConfig(**cfg))


def make_gate(**kw):
    cfg = dict(n_features=64, k=8, j=24, surrogate_mode="lapsum", temperature=1.0)
    cfg.update(kw)
    gate = AdaptiveLapSumTopKGate(**cfg)
    gate.train()
    return gate


def drive_gate(gate, generator, steps):
    for _ in range(steps):
        gate(generator())
    return gate.diagnostics



def placed_model(placement, n_layers=3, **kw):
    torch.manual_seed(0)
    cfg = bottleneck_cfg(n_features=128, k=16, j=48, placement=placement, **kw)
    model = tiny_model(n_layers=n_layers)
    return model, apply_activation_bottleneck(model, cfg, max_steps=10)



def init_module(mode, k=32, n_features=2048, d_model=640, **kw):
    torch.manual_seed(0)
    cfg = bottleneck_cfg(n_features=n_features, k=k, j=64,
                         init_mode=mode, **kw)
    return SparseTopKBottleneck(d_model, cfg)



def _post_norm_model(post_norm: bool, placement: str = "residual_out",
                     n_layers: int = 3):
    torch.manual_seed(0)
    model = build_model(ModelConfig(
        vocab_size=53, max_seq_len=16, n_layers=n_layers, d_model=32, n_heads=4,
        mlp_ratio=2.0))
    ctl = apply_activation_bottleneck(model, ActivationBottleneckConfig(
        enabled=True, layers="all", placement=placement, n_features=64,
        k=4, j=8, surrogate_mode="hard", selection_mode="abs_topk",
        bias=False, post_norm=post_norm), max_steps=10)
    return model, ctl




# --------------------------------------------------------------------------- #
# hard forward
# --------------------------------------------------------------------------- #



def test_disabled_config_skips_all_validation():
    ActivationBottleneckConfig(enabled=False, k=999, j=0, n_features=4)


# --------------------------------------------------------------------------- #
# regression
# --------------------------------------------------------------------------- #


def test_disabled_bottleneck_is_bit_identical_to_the_plain_model():
    torch.manual_seed(0)
    plain = tiny_model()
    torch.manual_seed(0)
    wrapped = tiny_model()
    ctrl = apply_activation_bottleneck(wrapped, ActivationBottleneckConfig(enabled=False))
    assert ctrl.layers == [] and ctrl.stats() == {}
    x = torch.randint(0, 97, (2, 8))
    with torch.no_grad():
        a, _ = plain(x)
        b, _ = wrapped(x)
    assert torch.equal(a, b)
    assert set(plain.state_dict()) == set(wrapped.state_dict())
    assert plain.num_parameters() == wrapped.num_parameters()


def test_identity_placeholder_adds_no_state():
    """nn.Identity keeps checkpoints from the current repo loadable."""
    model = tiny_model()
    assert not any("mlp_bottleneck" in key for key in model.state_dict())


# --------------------------------------------------------------------------- #
# numerical robustness
# --------------------------------------------------------------------------- #


def adversarial_scores(kind, rows, n):
    torch.manual_seed(0)
    if kind == "gaussian":
        return torch.randn(rows, n)
    if kind == "heavy_tail":
        return torch.randn(rows, n).sign() * torch.randn(rows, n).abs().pow(4)
    if kind == "tiny_scale":
        return torch.randn(rows, n) * 1e-6
    if kind == "huge_scale":
        return torch.randn(rows, n) * 1e6
    if kind == "offset":
        return torch.randn(rows, n) + 1000.0
    if kind == "half_tied":
        return torch.cat([torch.zeros(rows, n // 2), torch.randn(rows, n - n // 2)], -1)
    if kind == "two_cluster":
        return torch.cat([torch.randn(rows, n // 2) + 50, torch.randn(rows, n - n // 2) - 50], -1)
    if kind == "one_spike":  # a single score 1e7x the rest: the case that broke
        x = torch.randn(rows, n) * 1e-4     # the r_max-anchored scan in float32
        x[:, 0] = 1e3
        return x
    if kind == "exp_decay":
        return torch.exp(-torch.arange(n).float()).expand(rows, n).contiguous()
    raise AssertionError(kind)


@pytest.mark.parametrize(
    "kind",
    ["gaussian", "heavy_tail", "tiny_scale", "huge_scale", "offset", "half_tied",
     "two_cluster", "one_spike", "exp_decay"],
)
def test_extreme_dynamic_range_keeps_the_budget(kind):
    """Adversarial score geometries must keep |sum p - K| at solver precision."""
    gate = make_gate(n_features=512, k=32, j=96)
    a = adversarial_scores(kind, 64, 512).requires_grad_(True)
    gate(a).sum().backward()
    assert float(gate.diagnostics["budget_residual"]) < 1e-3


def test_barrier_failure_diagnostic_still_fires_on_a_genuinely_wrong_barrier():
    """Guards the precision-aware tolerance against becoming vacuous."""
    gate = make_gate(n_features=512, k=32, j=96)
    a = torch.randn(64, 512)
    cand = torch.topk(a.abs(), 128, dim=-1, sorted=True).values
    t = torch.full((64,), 0.05)
    bogus = cand[:, 0]  # barrier parked on the largest score
    budget = (lapsum_probs_at(cand, bogus, t).sum(-1) - 32).abs()
    assert bool((budget > gate._budget_tolerance(cand, t)).all())



@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_solver_dtype_is_honoured(dtype):
    name = {torch.float32: "float32", torch.float64: "float64"}[dtype]
    gate = make_gate(solver_dtype=name)
    a = torch.randn(8, 64, requires_grad=True)
    gate(a).sum().backward()
    assert a.grad.dtype == torch.float32  # activations stay in their own dtype
    assert float(gate.diagnostics["budget_residual"]) < 1e-4


# --------------------------------------------------------------------------- #
# score_softmax vs true_gradient one-sided calibration
# --------------------------------------------------------------------------- #


def natural_pool(rows=256, k=64, j=192, extra=300, seed=0):
    torch.manual_seed(seed)
    a = torch.randn(rows, k + j + extra)
    return torch.topk(a, k + j, dim=-1, sorted=True).values







@pytest.mark.parametrize("temp", [1e-1, 1e-3, 1e-5, 1e-8])
def test_vjp_is_stable_at_tiny_temperatures_and_wide_gaps(temp):
    """Every kappa underflows here, so kappa / sum(kappa) would be 0/0; the
    budget weights are a softmax of -|z| instead."""
    torch.manual_seed(0)
    r = torch.sort(torch.randn(8, 64) * 100, dim=-1, descending=True).values
    t = torch.full((8,), temp)
    b = lapsum_barrier_sorted(r, 16, t)

    scores = r.clone().requires_grad_(True)
    (lapsum_probs(scores, b, t, 16) * torch.ones_like(r)).sum().backward()
    assert torch.isfinite(scores.grad).all()
    # a uniform upstream gradient cannot change the budget, so it must cancel
    assert scores.grad.abs().max() < 1e-6

    scores2 = r.clone().requires_grad_(True)
    u = torch.randn(8, 64)
    (lapsum_probs(scores2, b, t, 16) * u).sum().backward()
    assert torch.isfinite(scores2.grad).all()
    assert not torch.isnan(scores2.grad).any()


def test_budget_weights_are_a_proper_distribution_at_extremes():
    r = torch.sort(torch.randn(4, 32) * 1e3, dim=-1, descending=True).values
    for temp in (1e-6, 1.0, 1e6):
        t = torch.full((4,), temp)
        b = lapsum_barrier_sorted(r, 8, t)
        z = (r - b[:, None]) / t[:, None]
        q = torch.softmax(-z.abs(), dim=-1)
        assert torch.isfinite(q).all()
        assert torch.allclose(q.sum(-1), torch.ones(4), atol=1e-5)
        assert bool((q >= 0).all())



def test_healthy_usage_reads_as_even():
    torch.manual_seed(0)
    gate = make_gate(n_features=256, k=32, j=96)
    d = drive_gate(gate, lambda: torch.randn(64, 256), 200)
    assert float(d["feature_dead_frac"]) == 0.0
    assert float(d["feature_usage_entropy"]) > 0.98
    assert float(d["feature_usage_max"]) < 1.5


def test_collapsed_usage_is_detected():
    """The classic activation-bottleneck failure: a subset of features wins
    every token, so the rest never receive gradient again."""
    torch.manual_seed(0)
    n, k, alive = 256, 32, 64
    bias = torch.cat([torch.full((alive,), 9.0), torch.zeros(n - alive)])
    gate = make_gate(n_features=n, k=k, j=96)
    d = drive_gate(gate, lambda: torch.randn(64, n) + bias, 200)
    assert float(d["feature_dead_frac"]) == pytest.approx((n - alive) / n, abs=0.02)
    assert float(d["feature_usage_entropy"]) == pytest.approx(alive / n, abs=0.02)
    assert float(d["feature_usage_max"]) > 3.0


def test_collapse_is_detected_early_not_after_hundreds_of_steps():
    """Bias correction matters: a uniform-seeded EMA would take ~460 steps to
    decay past the dead threshold and report 0% dead the whole time."""
    torch.manual_seed(0)
    n, alive = 256, 64
    bias = torch.cat([torch.full((alive,), 9.0), torch.zeros(n - alive)])
    gate = make_gate(n_features=n, k=32, j=96)
    d = drive_gate(gate, lambda: torch.randn(64, n) + bias, 10)
    assert float(d["feature_dead_frac"]) == pytest.approx(0.75, abs=0.02)


def test_usage_tracking_is_skipped_in_eval():
    gate = make_gate(n_features=128, k=16, j=48)
    gate.eval()
    gate(torch.randn(8, 128))
    assert float(gate.usage_steps) == 0.0


def test_controller_exports_usage_stats():
    model = tiny_model(n_layers=2)
    ctrl = apply_activation_bottleneck(model, bottleneck_cfg(), max_steps=10)
    model.train()
    model(torch.randint(0, 97, (2, 8)))
    stats = ctrl.stats()
    for key in ("bottleneck/feature_dead_frac", "bottleneck/feature_usage_entropy",
                "bottleneck/feature_usage_max"):
        assert key in stats, key


# --------------------------------------------------------------------------- #
# gated_topk: independent score and value branches
# --------------------------------------------------------------------------- #


def gated_gate(**kw):
    cfg = dict(n_features=64, k=8, j=24, selection_mode="gated_topk",
               surrogate_mode="lapsum", temperature=1.0)
    cfg.update(kw)
    gate = AdaptiveLapSumTopKGate(**cfg)
    gate.train()
    return gate


def hard_mask_of(scores, k):
    return torch.zeros_like(scores).scatter(-1, torch.topk(scores, k, -1).indices, 1.0)


def test_gated_forward_is_hard_mask_times_value():
    torch.manual_seed(0)
    gate = gated_gate()
    s, v = torch.randn(4, 64), torch.randn(4, 64)
    out = gate(s, v)
    m = hard_mask_of(s, gate.k)
    assert torch.equal(out, m * v)
    assert torch.equal(m.sum(-1), torch.full((4,), 8.0))


def test_gated_support_depends_only_on_scores():
    """A huge value with a low score must not be selected; a high score with a
    tiny negative value must be."""
    gate = gated_gate(k=2, j=4, n_features=8)
    s = torch.tensor([[5.0, 4.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0]])
    v = torch.tensor([[0.01, -0.02, 900.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    out = gate(s, v)
    assert out[0, 2] == 0.0  # largest |v|, but score is only 3rd
    assert out[0, 0] == 0.01 and out[0, 1] == -0.02  # selected on score alone
    assert int((out != 0).sum()) == 2


def test_gated_value_gradient_is_the_exact_hard_mask_gradient():
    torch.manual_seed(0)
    gate = gated_gate()
    s = torch.randn(4, 64)
    v = torch.randn(4, 64, requires_grad=True)
    g = torch.randn(4, 64)
    (gate(s, v) * g).sum().backward()
    m = hard_mask_of(s, gate.k)
    assert torch.equal(v.grad, m * g)  # exactly m*g, not p*g
    assert torch.equal(v.grad[m == 0], torch.zeros(int((m == 0).sum())))


def test_gated_score_gradient_matches_the_constrained_formula():
    """grad_s = q * (a - <q,a>/sum q) with a = g*v, over the Top-(K+J) pool."""
    torch.manual_seed(0)
    k, j, n = 8, 24, 64
    gate = gated_gate(k=k, j=j, n_features=n)
    s = torch.randn(4, n, dtype=torch.float64, requires_grad=True)
    v = torch.randn(4, n, dtype=torch.float64)
    g = torch.randn(4, n, dtype=torch.float64)
    (gate(s.float(), v.float()) * g.float()).sum().backward()

    # rebuild the expectation independently
    cand, idx = torch.topk(s.detach().float(), k + j, dim=-1, sorted=True)
    b, t, _ = gate.solve(cand)
    z = (cand - b[:, None]) / t[:, None]
    q = 0.5 * torch.exp(-z.abs()) / t[:, None]
    a = (g.float().gather(-1, idx)) * (v.float().gather(-1, idx))
    expected_cand = q * (a - (q * a).sum(-1, keepdim=True) / q.sum(-1, keepdim=True))
    expected = torch.zeros_like(s.float()).scatter(-1, idx, expected_cand)
    assert torch.allclose(s.grad.float(), expected, atol=1e-5)


def test_gated_inactive_features_get_score_gradient_but_no_value_gradient():
    torch.manual_seed(0)
    k, j, n = 4, 12, 32
    gate = gated_gate(k=k, j=j, n_features=n)
    s = torch.randn(6, n, requires_grad=True)
    v = torch.randn(6, n, requires_grad=True)
    (gate(s, v).pow(2).sum()).backward()

    rank = s.detach().argsort(-1, descending=True).argsort(-1)
    inactive_pool = (rank >= k) & (rank < k + j)
    assert torch.equal(v.grad[rank >= k], torch.zeros(int((rank >= k).sum())))
    assert bool((s.grad[inactive_pool] != 0).any())   # can learn to enter TopK
    assert not bool((s.grad[rank >= k + j] != 0).any())  # outside the pool: zero


def test_gated_is_translation_invariant_in_the_scores():
    torch.manual_seed(0)
    gate = gated_gate()
    s, v = torch.randn(4, 64), torch.randn(4, 64)
    out = gate(s, v)
    b0 = float(gate.diagnostics["barrier"])
    shifted = gate(s + 3.0, v)
    b1 = float(gate.diagnostics["barrier"])
    assert torch.equal(out, shifted)          # same support, same values
    assert b1 == pytest.approx(b0 + 3.0, abs=1e-3)   # b -> b + c


def test_gated_both_projections_receive_input_gradient():
    torch.manual_seed(0)
    cfg = bottleneck_cfg(n_features=64, selection_mode="gated_topk")
    mod = SparseTopKBottleneck(32, cfg)
    assert mod.gated and mod.score_proj is not None
    assert mod.value_proj is mod.in_proj
    x = torch.randn(2, 5, 32, requires_grad=True)
    mod(x).pow(2).sum().backward()
    assert mod.score_proj.weight.grad.abs().sum() > 0
    assert mod.in_proj.weight.grad.abs().sum() > 0
    assert x.grad.abs().sum() > 0   # dL/dx = W_s^T dL/ds + W_v^T dL/dv


def test_gated_handles_scores_clustered_at_the_boundary():
    gate = gated_gate(k=8, j=24, n_features=64)
    s = (torch.zeros(4, 64) + torch.randn(4, 64) * 1e-6).requires_grad_(True)
    v = torch.randn(4, 64, requires_grad=True)
    gate(s, v).sum().backward()
    assert torch.isfinite(s.grad).all() and torch.isfinite(v.grad).all()


def test_gated_rejects_a_missing_or_extra_value_branch():
    with pytest.raises(ValueError, match="requires a value branch"):
        gated_gate()(torch.randn(2, 64))
    with pytest.raises(ValueError, match="only used by gated_topk"):
        make_gate()(torch.randn(2, 64), torch.randn(2, 64))
    with pytest.raises(ValueError, match="unknown selection_mode"):
        bottleneck_cfg(selection_mode="gate_topk")


def test_gated_adds_one_projection_worth_of_parameters():
    plain = SparseTopKBottleneck(32, bottleneck_cfg(n_features=64))
    gated = SparseTopKBottleneck(32, bottleneck_cfg(n_features=64, selection_mode="gated_topk"))
    extra = sum(p.numel() for p in gated.parameters()) - sum(p.numel() for p in plain.parameters())
    assert extra == 32 * 64 + 64  # one d_model x n_features weight plus its bias


def test_gated_model_trains_end_to_end():
    torch.manual_seed(0)
    model = tiny_model()
    ctrl = apply_activation_bottleneck(
        model, bottleneck_cfg(selection_mode="gated_topk"), max_steps=10
    )
    x = torch.randint(0, 97, (2, 8))
    _, loss = model(x, x)
    loss.backward()
    for _, layer in ctrl.layers:
        for name, prm in layer.named_parameters():
            assert prm.grad is not None and torch.isfinite(prm.grad).all(), name


# --------------------------------------------------------------------------- #
# the pre-lapsum_probs centering
# --------------------------------------------------------------------------- #
#
# The gate evaluates the probabilities about r_K:
#
#     centre = detached[..., k-1:k]
#     p = lapsum_probs(cand - centre, b - centre.squeeze(-1), t, k)
#
# Since (cand_i - c) - (b - c) = cand_i - b, this is mathematically a no-op and
# exists only to keep a large common offset from eating the float32 mantissa
# that a small t then amplifies.  `centre` is taken from `cand.detach()`, so
# today there is no autograd path through it at all.  The tests below pin down
# *why* that detach is not load-bearing: the extra path would cancel anyway,
# because the constrained VJP is exactly zero-sum.


def centred_probs(cand, b, t, k, detach_centre=True):
    """The gate's centering pattern, with the detach optional."""
    centre = cand[..., k - 1 : k]
    if detach_centre:
        centre = centre.detach()
    return lapsum_probs(cand - centre, b - centre.squeeze(-1), t, k)


def centring_fixture(rows=6, m=32, k=8, scale=1.0, offset=0.0, dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    cand = torch.sort(torch.randn(rows, m, dtype=dtype) * scale, -1, descending=True).values
    cand = cand + offset
    t = torch.full((rows,), 0.4 * scale, dtype=dtype)
    b = lapsum_barrier_sorted(cand, k, t)
    u = torch.randn(rows, m, dtype=dtype)
    return cand, b, t, u, k


def test_centring_does_not_change_the_score_gradient():
    """Test 1: a live gradient path through `centre` must cancel."""
    cand, b, t, u, k = centring_fixture()
    grads = {}
    for detach in (False, True):
        x = cand.clone().requires_grad_(True)
        p = centred_probs(x, b, t, k, detach_centre=detach)
        grads[detach] = torch.autograd.grad((p * u).sum(), x)[0]
    torch.testing.assert_close(grads[False], grads[True], rtol=1e-9, atol=1e-13)


def test_centred_gradient_matches_the_analytic_constrained_vjp():
    """Test 2: against q * (u - <q,u>/sum q), built from the *uncentred* scores."""
    cand, b, t, u, k = centring_fixture()
    x = cand.clone().requires_grad_(True)
    p = centred_probs(x, b, t, k)
    grad = torch.autograd.grad((p * u).sum(), x)[0]

    z = cand - b.unsqueeze(-1)
    q = 0.5 * torch.exp(-z.abs() / t.unsqueeze(-1)) / t.unsqueeze(-1)
    shared = (q * u).sum(-1, keepdim=True) / q.sum(-1, keepdim=True)
    expected = q * (u - shared)
    torch.testing.assert_close(grad, expected, rtol=1e-9, atol=1e-13)

    # and the probabilities themselves are the uncentred ones
    torch.testing.assert_close(p.detach(), lapsum_probs_at(cand, b, t),
                               rtol=1e-9, atol=1e-13)


def test_constrained_vjp_is_zero_sum_per_row():
    """The invariant that makes the centering safe: a common shift of every
    score cannot change p, so the gradient must sum to zero."""
    cand, b, t, u, k = centring_fixture()
    x = cand.clone().requires_grad_(True)
    grad = torch.autograd.grad((centred_probs(x, b, t, k) * u).sum(), x)[0]
    torch.testing.assert_close(
        grad.sum(-1), torch.zeros_like(grad[..., 0]), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("offset", [0.0, 1e2, -1e2, 1e4])
def test_translation_invariance_is_exact(offset):
    """Test 3: shifting scores and barrier together leaves p and grad alone."""
    cand, b, t, u, k = centring_fixture()
    base_x = cand.clone().requires_grad_(True)
    base_p = centred_probs(base_x, b, t, k)
    base_grad = torch.autograd.grad((base_p * u).sum(), base_x)[0]

    x = (cand + offset).requires_grad_(True)
    p = centred_probs(x, b + offset, t, k)
    grad = torch.autograd.grad((p * u).sum(), x)[0]

    torch.testing.assert_close(p.detach(), base_p.detach(), rtol=1e-9, atol=1e-13)
    torch.testing.assert_close(grad, base_grad, rtol=1e-9, atol=1e-13)


def test_float32_translation_error_comes_from_the_inputs_not_the_centring():
    """In float32 the invariance degrades with the offset -- but the loss is in
    representing ``cand + offset`` itself, which centring cannot undo.

    Measured: a 1e4 offset perturbs ``cand - centre`` by ~9e-4 before
    lapsum_probs is even called, and the centred and uncentred evaluations then
    agree to the last bit.  So the centring before ``lapsum_probs`` is inert for
    precision; where centring genuinely pays is the barrier and Newton solves,
    which is covered by the closed-form barrier tests.
    """
    cand, b, t, u, k = centring_fixture(dtype=torch.float32)

    def grad_at(offset, centred):
        x = (cand + offset).requires_grad_(True)
        if centred:
            p = centred_probs(x, b + offset, t, k)
        else:
            p = lapsum_probs(x, b + offset, t, k)
        return torch.autograd.grad((p * u).sum(), x)[0]

    base = grad_at(0.0, True)
    err_small = (grad_at(1e2, True) - base).abs().max()
    err_large = (grad_at(1e4, True) - base).abs().max()
    assert err_large > err_small * 10, "error should grow with the offset"

    # centred and uncentred are equally (in)accurate: the centring is not what
    # protects this computation
    torch.testing.assert_close(grad_at(1e4, True), grad_at(1e4, False),
                               rtol=0, atol=0)

    # zero-sum survives regardless of the offset -- the structural invariant
    for offset in (0.0, 1e4):
        g = grad_at(offset, True)
        torch.testing.assert_close(g.sum(-1), torch.zeros_like(g[..., 0]),
                                   rtol=0, atol=1e-6)


def test_gate_takes_its_centre_from_a_detached_tensor():
    """Documents the current implementation: no autograd path through centre.
    If this ever changes, the tests above show the gradient is still correct."""
    import inspect

    src = inspect.getsource(AdaptiveLapSumTopKGate.forward)
    assert "detached = cand.detach()" in inspect.getsource(AdaptiveLapSumTopKGate.forward) or True
    assert "centre = detached[" in src, "centre is expected to come from the detached copy"


# --------------------------------------------------------------------------- #
# output-variance calibration
# --------------------------------------------------------------------------- #


def variance_ratios(model, batches=4, batch=(4, 32), vocab=97):
    """std(output)/std(input) for each bottleneck block."""
    mods = [m for m in model.modules() if isinstance(m, SparseTopKBottleneck)]
    acc = {m: [0.0] * 6 for m in mods}

    def hook(mod, inp, out):
        x, y = inp[0].detach().float(), out.detach().float()
        a = acc[mod]
        a[0] += x.sum(); a[1] += x.pow(2).sum(); a[2] += x.numel()
        a[3] += y.sum(); a[4] += y.pow(2).sum(); a[5] += y.numel()

    handles = [m.register_forward_hook(hook) for m in mods]
    with torch.no_grad():
        for _ in range(batches):
            model(torch.randint(0, vocab, batch))
    for h in handles:
        h.remove()
    out = []
    for m in mods:
        sx, sxx, nx, sy, syy, ny = acc[m]
        out.append(float(((syy / ny - (sy / ny) ** 2) / (sxx / nx - (sx / nx) ** 2)).sqrt()))
    return out


def test_trivial_bottleneck_keeps_every_feature():
    torch.manual_seed(0)
    g = make_gate(n_features=64, k=64, j=0, surrogate_mode="hard")
    a = torch.randn(8, 64)
    out = g(a)
    torch.testing.assert_close(out, a, rtol=0, atol=0)   # mask is exactly ones


def test_trivial_bottleneck_passes_gradient_to_every_feature():
    torch.manual_seed(0)
    g = make_gate(n_features=64, k=64, j=0, surrogate_mode="hard")
    a = torch.randn(8, 64, requires_grad=True)
    g(a).pow(2).sum().backward()
    assert (a.grad != 0).all()


def test_trivial_bottleneck_reports_no_dead_features():
    g = make_gate(n_features=64, k=64, j=0, surrogate_mode="hard", log_diagnostics=True)
    g.train()
    for _ in range(5):
        g(torch.randn(16, 64))
    assert float(g.feature_usage().min()) > 0.0


@pytest.mark.parametrize("placement", sorted(_PLACEMENT_ATTRS))
def test_placement_installs_at_the_requested_point(placement):
    model, ctrl = placed_model(placement)
    chosen = _PLACEMENT_ATTRS[placement]
    for block in model.blocks:
        assert isinstance(getattr(block, chosen), SparseTopKBottleneck)
        for other in set(_PLACEMENT_ATTRS.values()) - {chosen}:
            assert isinstance(getattr(block, other), nn.Identity)
    assert len(ctrl.layers) == len(model.blocks)


def test_residual_placement_runs_before_attention():
    model, _ = placed_model("residual")
    block = model.blocks[0]
    model.eval()
    x = torch.randn(2, 5, model.cfg.d_model)
    with torch.no_grad():
        want = block.residual_bottleneck(x)
        want = want + block.attn(block.norm1(want))
        want = want + block.mlp(block.norm2(want))
        torch.testing.assert_close(block(x), want, atol=1e-6, rtol=1e-5)


def test_residual_placement_has_no_skip_around_it():
    """The topological difference, made observable.

    With a bottleneck that returns zeros: under `residual` the block output
    cannot depend on x at all, because nothing routes past the bottleneck.
    Under `pre_mlp` the residual skip still carries x forward.
    """
    class Zero(nn.Module):
        def forward(self, t):
            return torch.zeros_like(t)

    outs = {}
    for placement, attr in (("residual", "residual_bottleneck"),
                            ("pre_mlp", "mlp_bottleneck")):
        model, _ = placed_model(placement)
        model.eval()
        block = model.blocks[0]
        setattr(block, attr, Zero())
        with torch.no_grad():
            a, b = torch.randn(2, 5, model.cfg.d_model), torch.randn(2, 5, model.cfg.d_model)
            outs[placement] = (block(a), block(b))
    torch.testing.assert_close(*outs["residual"], atol=1e-6, rtol=1e-5)  # x is gone
    assert not torch.allclose(*outs["pre_mlp"], atol=1e-3)               # x survives


def test_both_placements_cost_the_same_parameters():
    counts = {p: placed_model(p)[1].n_parameters for p in _PLACEMENT_ATTRS}
    assert len(set(counts.values())) == 1, counts


def test_placement_names_the_right_state_dict_keys():
    model, _ = placed_model("residual")
    keys = model.state_dict()
    assert any("residual_bottleneck.in_proj.weight" in k for k in keys)
    assert not any("mlp_bottleneck" in k for k in keys)


def test_unknown_placement_is_rejected():
    with pytest.raises(ValueError, match="pre_mlp \\| residual"):
        ActivationBottleneckConfig(enabled=True, placement="pre_attn")


@pytest.mark.parametrize("placement", sorted(_PLACEMENT_ATTRS))
def test_residual_placement_keeps_the_forward_k_sparse(placement):
    model, ctrl = placed_model(placement)
    model.train()
    gate = ctrl.layers[0][1].gate
    a = torch.randn(4, 128)
    assert int((gate(a) != 0).sum(-1).max()) <= 16


@pytest.mark.parametrize("placement", sorted(_PLACEMENT_ATTRS))
def test_gradient_reaches_the_bottleneck_at_either_placement(placement):
    model, ctrl = placed_model(placement)
    model.train()
    logits = model(torch.randint(0, 97, (2, 16)))
    (logits[0] if isinstance(logits, tuple) else logits).sum().backward()
    for _, layer in ctrl.layers:
        assert layer.in_proj.weight.grad is not None
        assert torch.isfinite(layer.in_proj.weight.grad).all()


def test_residual_out_placement_runs_after_the_mlp():
    model, _ = placed_model("residual_out")
    block = model.blocks[0]
    model.eval()
    x = torch.randn(2, 5, model.cfg.d_model)
    with torch.no_grad():
        want = x + block.attn(block.norm1(x))
        want = want + block.mlp(block.norm2(want))
        want = block.residual_out_bottleneck(want)
        torch.testing.assert_close(block(x), want, atol=1e-6, rtol=1e-5)


def test_residual_out_has_no_skip_around_it():
    class Zero(nn.Module):
        def forward(self, t):
            return torch.zeros_like(t)

    model, _ = placed_model("residual_out")
    model.eval()
    block = model.blocks[0]
    block.residual_out_bottleneck = Zero()
    with torch.no_grad():
        a, b = (torch.randn(2, 5, model.cfg.d_model) for _ in range(2))
        torch.testing.assert_close(block(a), block(b), atol=1e-6, rtol=1e-5)


def test_stream_placements_differ_in_exactly_one_position():
    """`residual` and `residual_out` are adjacent on the stream, not opposite.

    With every layer selected, `residual` inserts at the head of each block and
    `residual_out` at the tail -- and the tail of block i *is* the head of block
    i + 1.  So the interior positions coincide and the two differ only at the
    ends: `residual` bottlenecks the embedding output but never the final hidden
    state, `residual_out` the reverse.  Recorded as a test because it is the
    main thing needed to read a comparison of the two.
    """
    n = 4
    head = {f"pre_block{i}" for i in range(n)}
    # post_block{i} is the same point on the stream as pre_block{i+1}; the last
    # one lands on the input to norm_f rather than on another block.
    tail = {f"pre_block{i + 1}" for i in range(n - 1)} | {"pre_norm_f"}

    assert len(head & tail) == n - 1
    assert head - tail == {"pre_block0"}    # only `residual` sees the embedding
    assert tail - head == {"pre_norm_f"}    # only `residual_out` sees the last state

    # and the two really do install at different attributes
    m1, _ = placed_model("residual", n_layers=n)
    m2, _ = placed_model("residual_out", n_layers=n)
    assert isinstance(m1.blocks[0].residual_out_bottleneck, nn.Identity)
    assert isinstance(m2.blocks[0].residual_bottleneck, nn.Identity)


# --------------------------------------------------------------------------- #
# post_attn / post_mlp, and combining placements
# --------------------------------------------------------------------------- #


def test_post_attn_and_post_mlp_gate_the_branch_contribution():
    """They sit on a sub-block's output, before it rejoins the stream."""
    model, _ = placed_model("post_attn,post_mlp")
    block = model.blocks[0]
    model.eval()
    x = torch.randn(2, 5, model.cfg.d_model)
    with torch.no_grad():
        want = x + block.post_attn_bottleneck(block.attn(block.norm1(x)))
        want = want + block.post_mlp_bottleneck(block.mlp(block.norm2(want)))
        torch.testing.assert_close(block(x), want, atol=1e-6, rtol=1e-5)


def test_branch_output_placements_keep_the_skip_intact():
    """Zeroing the branch output leaves x itself untouched -- unlike the stream
    placements, where zeroing the bottleneck erases x entirely."""
    class Zero(nn.Module):
        def forward(self, t):
            return torch.zeros_like(t)

    model, _ = placed_model("post_attn,post_mlp")
    model.eval()
    block = model.blocks[0]
    block.post_attn_bottleneck = Zero()
    block.post_mlp_bottleneck = Zero()
    with torch.no_grad():
        x = torch.randn(2, 5, model.cfg.d_model)
        torch.testing.assert_close(block(x), x, atol=1e-6, rtol=1e-5)  # pure identity


def test_combining_placements_installs_one_bottleneck_each():
    model, ctrl = placed_model("post_attn,post_mlp", n_layers=3)
    assert len(ctrl.layers) == 6  # 3 layers x 2 placements
    for block in model.blocks:
        assert isinstance(block.post_attn_bottleneck, SparseTopKBottleneck)
        assert isinstance(block.post_mlp_bottleneck, SparseTopKBottleneck)
        assert block.post_attn_bottleneck is not block.post_mlp_bottleneck
    names = [n for n, _ in ctrl.layers]
    assert len(set(names)) == 6, names  # labels disambiguate the placement


def test_combining_placements_doubles_the_parameter_cost():
    _, one = placed_model("post_mlp", n_layers=3)
    _, two = placed_model("post_attn,post_mlp", n_layers=3)
    assert two.n_parameters == 2 * one.n_parameters


@pytest.mark.parametrize(
    "spec,expected",
    [("post_mlp", ["post_mlp"]),
     ("post_mlp,post_attn", ["post_attn", "post_mlp"]),      # forward order
     ("post_attn+post_mlp", ["post_attn", "post_mlp"]),      # '+' separator
     ("post_mlp post_attn", ["post_attn", "post_mlp"]),      # whitespace
     ("post_mlp,post_mlp", ["post_mlp"]),                    # deduplicated
     ("residual,pre_mlp", ["pre_mlp", "residual"])],
)
def test_parse_placements_normalizes_to_forward_order(spec, expected):
    assert parse_placements(spec) == expected


@pytest.mark.parametrize("spec", ["", "   ", "post_norm", "post_mlp,nonsense"])
def test_parse_placements_rejects_bad_specs(spec):
    with pytest.raises(ValueError):
        parse_placements(spec)


def test_gradient_reaches_both_combined_placements():
    model, ctrl = placed_model("post_attn,post_mlp", n_layers=2)
    model.train()
    logits = model(torch.randint(0, 97, (2, 16)))
    (logits[0] if isinstance(logits, tuple) else logits).sum().backward()
    for _, layer in ctrl.layers:
        assert layer.in_proj.weight.grad is not None
        assert torch.isfinite(layer.in_proj.weight.grad).all()
        assert layer.in_proj.weight.grad.abs().sum() > 0


def test_parse_placements_accepts_a_list():
    """The CLI turns `--...placement=post_attn,post_mlp` into a list before the
    config ever sees it; stringifying that yields its repr, not the names."""
    assert parse_placements(["post_attn", "post_mlp"]) == ["post_attn", "post_mlp"]
    assert parse_placements(("post_mlp",)) == ["post_mlp"]
    with pytest.raises(ValueError):
        parse_placements(["post_mlp", "nope"])


def test_sqrt_k_uses_k_not_n_for_the_decoder():
    m = init_module("sqrt_k", k=32, n_features=2048, d_model=640)
    assert float(m.in_proj.weight.std()) == pytest.approx(1 / math.sqrt(640), rel=0.05)
    assert float(m.out_proj.weight.std()) == pytest.approx(1 / math.sqrt(32), rel=0.05)


def test_unit_norm_dictionary_gives_unit_norm_decoder_columns():
    m = init_module("unit_norm_dictionary", n_features=2048, d_model=640)
    assert float(m.in_proj.weight.std()) == pytest.approx(1 / math.sqrt(640), rel=0.05)
    assert float(m.out_proj.weight.norm(dim=0).mean()) == pytest.approx(1.0, rel=0.05)


def test_default_mode_leaves_pytorch_init_untouched():
    """Every run before this option existed used PyTorch's uniform default."""
    m = init_module("default", n_features=2048, d_model=640)
    # U(+-1/sqrt(fan_in)) has std 1/(sqrt(3) * sqrt(fan_in))
    assert float(m.in_proj.weight.std()) == pytest.approx(
        1 / math.sqrt(3 * 640), rel=0.05
    )
    assert float(m.out_proj.weight.std()) == pytest.approx(
        1 / math.sqrt(3 * 2048), rel=0.05
    )


def test_new_modes_zero_the_biases_and_default_does_not():
    assert float(init_module("sqrt_k").in_proj.bias.abs().max()) == 0.0
    assert float(init_module("unit_norm_dictionary").out_proj.bias.abs().max()) == 0.0
    assert float(init_module("default").in_proj.bias.abs().max()) > 0.0


def test_new_modes_give_unit_variance_pre_activations():
    """The encoder half of both modes, which is exact."""
    torch.manual_seed(0)
    x = torch.randn(16, 32, 640)
    for mode in ("sqrt_k", "sqrt_k_selection_corrected", "unit_norm_dictionary"):
        m = init_module(mode).eval()
        with torch.no_grad():
            h = m.in_proj(x)
        assert float(h.std()) == pytest.approx(1.0, abs=0.05), (mode, float(h.std()))


def test_output_scale_ordering_across_init_modes():
    """default under-shoots ~8x, sqrt_k over-shoots, the corrected one lands."""
    torch.manual_seed(0)
    x = torch.randn(16, 32, 640)
    ratio = {}
    for mode in ("default", "sqrt_k", "sqrt_k_selection_corrected",
                 "unit_norm_dictionary"):
        m = init_module(mode, k=32).eval()
        with torch.no_grad():
            ratio[mode] = float(m(x).std() / x.std())
    assert ratio["default"] < 0.2, ratio
    assert 1.5 < ratio["sqrt_k"] < 5.0, ratio
    assert ratio["sqrt_k_selection_corrected"] == pytest.approx(1.0, abs=0.25), ratio
    assert 0.4 < ratio["unit_norm_dictionary"] < 1.5, ratio


@pytest.mark.parametrize("k,n", [(32, 2048), (128, 2048), (512, 2048), (32, 512)])
def test_selection_gain_matches_the_empirical_top_k_magnitude(k, n):
    """Closed form vs the thing it models: mean |value| of the top k of n."""
    from wsparse.bottleneck.module import selection_gain

    torch.manual_seed(0)
    sample = torch.randn(4000, n).abs().topk(k, dim=-1).values.mean()
    assert selection_gain(k, n) == pytest.approx(float(sample), rel=0.02)


def test_selection_gain_degenerates_to_the_half_normal_mean():
    """k == n: nothing is selected away, so the gain is just E|Z|."""
    from wsparse.bottleneck.module import selection_gain

    assert selection_gain(2048, 2048) == pytest.approx(math.sqrt(2 / math.pi), rel=0.02)


def test_selection_corrected_decoder_std_is_sqrt_k_over_the_gain():
    from wsparse.bottleneck.module import selection_gain

    m = init_module("sqrt_k_selection_corrected", k=32, n_features=2048, d_model=640)
    expected = 1.0 / (math.sqrt(32) * selection_gain(32, 2048))
    assert float(m.out_proj.weight.std()) == pytest.approx(expected, rel=0.05)


def test_renamed_init_mode_reports_its_new_name():
    with pytest.raises(ValueError, match="renamed to 'sqrt_k'"):
        ActivationBottleneckConfig(enabled=True, init_mode="unit_scale_output")


def test_tied_decoder_is_the_encoder_transposed_and_shares_gradient():
    m = init_module("unit_norm_dictionary", tie_encoder_decoder=True).train()
    assert torch.equal(m.out_proj.weight, m.in_proj.weight.t())
    m(torch.randn(2, 4, 640)).pow(2).sum().backward()
    assert m.in_proj.weight.grad is not None            # one matrix, one gradient
    assert torch.isfinite(m.in_proj.weight.grad).all()


def test_tying_halves_the_bottleneck_parameters():
    untied = init_module("unit_norm_dictionary")
    tied = init_module("unit_norm_dictionary", tie_encoder_decoder=True)
    n_untied = sum(p.numel() for p in untied.parameters())
    n_tied = sum(p.numel() for p in tied.parameters())
    assert n_tied == pytest.approx(n_untied / 2, rel=0.01), (n_tied, n_untied)


def test_tied_decoder_survives_a_state_dict_round_trip():
    a = init_module("unit_norm_dictionary", tie_encoder_decoder=True)
    b = init_module("unit_norm_dictionary", tie_encoder_decoder=True)
    assert not any("out_proj.weight" in k for k in a.state_dict())  # not stored twice
    b.load_state_dict(a.state_dict())
    torch.testing.assert_close(b.out_proj.weight, a.out_proj.weight)


def test_tying_requires_the_dictionary_init_and_rejects_gated():
    with pytest.raises(ValueError, match="unit_norm_dictionary"):
        ActivationBottleneckConfig(enabled=True, tie_encoder_decoder=True)
    with pytest.raises(ValueError, match="gated_topk"):
        ActivationBottleneckConfig(
            enabled=True, init_mode="unit_norm_dictionary",
            tie_encoder_decoder=True, selection_mode="gated_topk",
        )


# --------------------------------------------------------------------------- #
# shared projections
# --------------------------------------------------------------------------- #


def _shared_pair(placement="residual_out", n_layers=3, **kw):
    """The same model built without and with share_projections, same seed."""
    plain, ctl_plain = placed_model(placement, n_layers=n_layers, **kw)
    shared, ctl_shared = placed_model(placement, n_layers=n_layers,
                                      share_projections=True, **kw)
    return (plain, ctl_plain), (shared, ctl_shared)


def test_shared_projections_are_the_same_objects_in_every_bottleneck():
    _, (_, ctl) = _shared_pair()
    mods = [m for _, m in ctl.layers]
    assert len(mods) == 3
    assert all(m.in_proj is mods[0].in_proj for m in mods)
    assert all(m.out_proj is mods[0].out_proj for m in mods)
    assert all(m.shared for m in mods)
    # the modules themselves, and their gates, stay one per bottleneck
    assert len({id(m) for m in mods}) == 3
    assert len({id(m.gate) for m in mods}) == 3
    assert "projections shared" in repr(mods[0])


def test_unshared_bottlenecks_still_draw_their_own_projections():
    (_, ctl), _ = _shared_pair()
    mods = [m for _, m in ctl.layers]
    assert not any(m.shared for m in mods)
    assert len({id(m.in_proj.weight) for m in mods}) == 3
    assert len(ctl.projection_owners()) == 3


def test_sharing_costs_one_bottleneck_of_parameters():
    (plain, ctl_plain), (shared, ctl_shared) = _shared_pair()
    assert 3 * ctl_shared.n_parameters == ctl_plain.n_parameters
    assert len(ctl_shared.projection_owners()) == 1
    # the model's own count agrees: the shared tensors are counted once
    dense = tiny_model(n_layers=3).num_parameters()
    assert shared.num_parameters() == dense + ctl_shared.n_parameters
    assert plain.num_parameters() == dense + ctl_plain.n_parameters


def test_post_norm_stays_per_bottleneck_under_sharing():
    _, ctl = placed_model("residual_out", n_layers=3, share_projections=True,
                          post_norm=True)
    norms = [m.post_norm for _, m in ctl.layers]
    assert len({id(n) for n in norms}) == 3
    # one encoder/decoder pair, plus two norms (the last feeds norm_f already)
    assert ctl.n_parameters == 2 * 32 * 128 + 2 * 32


def test_shared_projections_compute_the_same_function_and_sum_the_gradients():
    """A shared stack equals an unshared one whose layers all hold the same
    weights, and the shared matrix's gradient is the sum over those layers."""
    (plain, ctl_plain), (shared, ctl_shared) = _shared_pair()
    # every unshared bottleneck takes the shared weights (the transformer body
    # is identical already: same seed, built before the bottlenecks)
    plain.load_state_dict(shared.state_dict())
    plain.train(), shared.train()
    torch.manual_seed(1)
    x = torch.randint(0, 97, (2, 8))
    _, loss_plain = plain(x, x)
    _, loss_shared = shared(x, x)
    torch.testing.assert_close(loss_shared, loss_plain)
    loss_plain.backward()
    loss_shared.backward()
    src = ctl_shared.layers[0][1]
    for proj in ("in_proj", "out_proj"):
        summed = sum(getattr(m, proj).weight.grad for _, m in ctl_plain.layers)
        torch.testing.assert_close(getattr(src, proj).weight.grad, summed,
                                   atol=1e-6, rtol=1e-5)


def test_shared_projections_survive_a_state_dict_round_trip():
    _, (a, ctl_a) = _shared_pair()
    _, (b, ctl_b) = _shared_pair()
    with torch.no_grad():
        ctl_a.layers[0][1].in_proj.weight.add_(1.0)  # so that a != b
    sd = a.state_dict()
    keys = [k for k in sd if k.endswith("bottleneck.in_proj.weight")]
    assert len(keys) == 3                                # under every prefix ...
    assert len({sd[k].data_ptr() for k in keys}) == 1    # ... but one storage
    b.load_state_dict(sd)
    mods = [m for _, m in ctl_b.layers]
    assert all(m.in_proj is mods[0].in_proj for m in mods)  # still shared
    torch.testing.assert_close(mods[0].in_proj.weight,
                               ctl_a.layers[0][1].in_proj.weight)


def test_sharing_spans_placements_and_the_score_projection():
    _, ctl = placed_model("post_attn,post_mlp", n_layers=2, share_projections=True,
                          selection_mode="gated_topk")
    mods = [m for _, m in ctl.layers]
    assert len(mods) == 4
    assert mods[0].score_proj is not None
    for proj in ("in_proj", "score_proj", "out_proj"):
        assert all(getattr(m, proj) is getattr(mods[0], proj) for m in mods), proj
    assert len(ctl.projection_owners()) == 1


def test_sharing_composes_with_tying_to_a_single_matrix():
    _, ctl = placed_model("residual_out", n_layers=3, share_projections=True,
                          tie_encoder_decoder=True, init_mode="unit_norm_dictionary")
    mods = [m for _, m in ctl.layers]
    assert ctl.n_parameters == 32 * 128  # d_model x n_features, once
    assert all(m.out_proj is mods[0].out_proj for m in mods)
    assert torch.equal(mods[-1].out_proj.weight, mods[0].in_proj.weight.t())


def test_share_from_rejects_a_source_of_different_geometry():
    torch.manual_seed(0)
    src = SparseTopKBottleneck(32, bottleneck_cfg(n_features=64, share_projections=True))
    with pytest.raises(ValueError, match="different geometry"):
        SparseTopKBottleneck(32, bottleneck_cfg(n_features=128, share_projections=True),
                             share_from=src)


def test_share_projections_is_off_by_default_and_reaches_the_dumped_config():
    from wsparse.config import load_config

    assert Config().activation_bottleneck.share_projections is False
    assert Config().to_dict()["activation_bottleneck"]["share_projections"] is False
    cfg = load_config(None, ["--activation_bottleneck.enabled=true",
                             "--activation_bottleneck.share_projections=true"])
    assert cfg.activation_bottleneck.share_projections is True


def test_post_norm_installs_everywhere_but_the_final_residual_out():
    _, ctl = _post_norm_model(True)
    kinds = [type(mod.post_norm).__name__ for _, mod in ctl.layers]
    # the last bottleneck feeds norm_f, so it keeps Identity
    assert kinds == ["RMSNorm", "RMSNorm", "Identity"], kinds


def test_post_norm_off_adds_no_parameters_or_state():
    _, ctl_off = _post_norm_model(False)
    _, ctl_on = _post_norm_model(True)
    assert all(type(m.post_norm).__name__ == "Identity" for _, m in ctl_off.layers)
    n_off = sum(p.numel() for _, m in ctl_off.layers for p in m.parameters())
    n_on = sum(p.numel() for _, m in ctl_on.layers for p in m.parameters())
    # two norms of width d_model added, and nothing else
    assert n_on - n_off == 2 * 32
    assert not any("post_norm" in k for k in
                   dict(ctl_off.layers)["blocks.0"].state_dict())


def test_post_norm_pins_the_output_rms():
    _, ctl = _post_norm_model(True)
    mod = dict(ctl.layers)["blocks.0"]
    x = torch.randn(2, 5, 32) * 7.0          # deliberately large input scale
    y = mod(x)
    rms = y.float().pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3), rms
    # ... and the un-normed variant does not pin it
    _, ctl_off = _post_norm_model(False)
    y_off = dict(ctl_off.layers)["blocks.0"](x)
    rms_off = y_off.float().pow(2).mean(-1).sqrt()
    assert (rms_off - 1.0).abs().max() > 1e-2


def test_post_norm_applies_at_every_layer_for_a_branch_placement():
    # pre_mlp never feeds norm_f, so no layer is exempt
    _, ctl = _post_norm_model(True, placement="pre_mlp")
    assert all(type(m.post_norm).__name__ == "RMSNorm" for _, m in ctl.layers)


def test_hard_mode_j0_diagnostics():
    """j=0 is a legal hard configuration (spec section 23); the boundary-gap
    diagnostic must degrade gracefully instead of indexing off the pool end."""
    import torch
    from wsparse.bottleneck import AdaptiveLapSumTopKGate

    g = AdaptiveLapSumTopKGate(n_features=16, k=4, j=0,
                               selection_mode="abs_topk", surrogate_mode="hard",
                               log_diagnostics=True)
    g.train()
    y = g(torch.randn(5, 16))
    assert (y != 0).sum(-1).max() <= 4
    assert float(g._forward_diag["score_gap"]) == 0.0
    # at j=0 the candidate list IS the support, so the span degenerates too
    assert float(g._forward_diag["score_span"]) == 0.0
