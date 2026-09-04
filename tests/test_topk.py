"""TopK + soft gate: forward support, the hand-written backward, the soft-L0
penalty and the wiring (config, controller, optimizer, one training step).

The gradient tests all compare against the boxed formulas by hand rather than
against autograd, which is the point of the method: autograd through the hard
TopK would give the *narrow* backward support, and here it must be the wide one.
"""

import pytest
import torch
import torch.nn as nn

from wsparse.config import Config, ModelConfig, SparsityConfig
from wsparse.model import build_model
from wsparse.optim import build_optimizer, set_lr
from wsparse.sparsity import (
    SparseLinear,
    TopKSoftGateLinear,
    apply_sparsity,
    group_shape,
    resolve_count,
    topk_masks,
)


def make_layer(out_features=4, in_features=5, beta=2.0, seed=0, **kw):
    torch.manual_seed(seed)
    linear = nn.Linear(in_features, out_features, bias=False)
    kw.setdefault("k", 6)
    kw.setdefault("j", 4)
    kw.setdefault("s_init_mode", "normal")
    kw.setdefault("s_init", 1.0)
    layer = TopKSoftGateLinear(linear, **kw)
    layer.beta.fill_(beta)
    return layer


def reference(layer):
    """(mask_a, mask_b, p) recomputed from scratch, independently of the layer."""
    s = layer.s.detach().float()
    flat = s.reshape(layer.group_shape)
    order = flat.argsort(dim=1, descending=True)
    mask_a = torch.zeros_like(flat, dtype=torch.bool).scatter_(1, order[:, : layer.k], True)
    mask_b = torch.zeros_like(flat, dtype=torch.bool).scatter_(
        1, order[:, : layer.k + layer.j], True
    )
    p = torch.sigmoid(float(layer.beta) * s)
    return mask_a.reshape(s.shape), mask_b.reshape(s.shape), p


def tiny_model(**kw):
    cfg = dict(vocab_size=97, max_seq_len=32, n_layers=2, d_model=32, n_heads=4, mlp_ratio=4.0)
    cfg.update(kw)
    return build_model(ModelConfig(**cfg))


def topk_config(**kw):
    # beta = 2 by default: the SparsityConfig default schedule (1e3 -> 1e6) is
    # on the LTP scale, where z = w**2, and would saturate every gate here.
    cfg = dict(
        enabled=True,
        method="topk",
        k=0.25,
        j=0.25,
        s_init_mode="normal",
        s_init=1.0,
        inverse_temperature=2.0,
    )
    cfg.update(kw)
    return SparsityConfig(**cfg)


# --------------------------------------------------------------------------- #
# grouping and count resolution
# --------------------------------------------------------------------------- #


def test_group_shapes():
    assert group_shape((4, 6), "tensor", 4) == (1, 24)
    assert group_shape((4, 6), "row", 4) == (4, 6)
    assert group_shape((4, 6), "block", 4) == (6, 4)
    with pytest.raises(ValueError, match="does not divide"):
        group_shape((4, 6), "block", 5)
    with pytest.raises(ValueError, match="unknown topk grouping"):
        group_shape((4, 6), "diagonal", 4)


def test_counts_are_fractions_below_one_and_absolute_above():
    assert resolve_count(0.1, 100) == 10
    assert resolve_count(10, 100) == 10
    assert resolve_count(1.0, 100) == 1  # exactly 1 is a count, not "all of it"
    assert resolve_count(0.0, 100) == 0  # j = 0: no exploration
    assert resolve_count(0.001, 100, minimum=1) == 1  # k never rounds down to 0
    assert resolve_count(500, 100) == 100  # clamped to the group


def test_k_and_j_are_resolved_per_group():
    layer = make_layer(out_features=4, in_features=8, k=0.25, j=0.125, groups="row")
    assert layer.group_shape == (4, 8)
    assert (layer.k, layer.j) == (2, 1)
    assert layer.topk_numel == 8 and layer.explore_numel == 4

    flat = make_layer(out_features=4, in_features=8, k=0.25, j=0.125, groups="tensor")
    assert flat.group_shape == (1, 32)
    assert (flat.k, flat.j) == (8, 4)


def test_j_cannot_overflow_the_group():
    layer = make_layer(out_features=2, in_features=4, k=6, j=9)
    assert (layer.k, layer.j) == (6, 2)  # k + j == group size


# --------------------------------------------------------------------------- #
# forward pass
# --------------------------------------------------------------------------- #


def test_forward_is_topk_times_weight_times_gate():
    layer = make_layer()
    mask_a, _, p = reference(layer)
    expected = mask_a * layer.weight.detach() * p
    assert torch.allclose(layer.effective_weight(), expected, atol=1e-6)
    assert int(mask_a.sum()) == layer.k
    # everything outside TopK is *exactly* zero, not just small
    assert torch.equal(layer.effective_weight()[~mask_a], torch.zeros(int((~mask_a).sum())))


def test_selection_uses_s_and_not_the_weights():
    layer = make_layer(out_features=2, in_features=3, k=2, j=0)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[9.0, 9.0, 9.0], [0.01, 0.01, 0.01]]))
        layer.s.copy_(torch.tensor([[-1.0, -2.0, -3.0], [3.0, 2.0, 1.0]]))
    mask_a, _ = layer.supports()
    # the two largest s are both in the small-weight row; |w| and w*p would
    # both have picked the other row
    assert mask_a.tolist() == [[False, False, False], [True, True, False]]


def test_gate_attenuates_inside_topk():
    layer = make_layer(beta=0.5)
    mask_a, _, p = reference(layer)
    active = layer.effective_weight()[mask_a] / layer.weight.detach()[mask_a]
    assert torch.allclose(active, p[mask_a], atol=1e-6)
    assert (active > 0).all() and (active < 1).all()  # a genuine soft gate


def test_forward_uses_the_custom_autograd_function():
    layer = make_layer()
    assert type(layer.effective_weight().grad_fn).__name__.startswith("TopKSoftGate")


def test_layer_forward_matches_a_masked_linear():
    layer = make_layer(out_features=4, in_features=5)
    x = torch.randn(2, 3, 5)
    assert torch.allclose(layer(x), x @ layer.effective_weight().T, atol=1e-6)


# --------------------------------------------------------------------------- #
# backward pass
# --------------------------------------------------------------------------- #


def test_weight_gradient_is_over_the_wide_support():
    layer = make_layer()
    mask_a, mask_b, p = reference(layer)
    g = torch.randn_like(layer.weight)

    (layer.effective_weight() * g).sum().backward()

    assert torch.allclose(layer.weight.grad, mask_b * p * g, atol=1e-6)
    # the J exploratory positions are inactive in the forward pass but do learn
    explore = mask_b & ~mask_a
    assert explore.sum() == layer.j
    assert (layer.weight.grad[explore] != 0).all()
    assert torch.equal(layer.weight.grad[~mask_b], torch.zeros(int((~mask_b).sum())))


def test_score_gradient_matches_the_boxed_formula():
    layer = make_layer()
    beta = float(layer.beta)
    mask_a, mask_b, p = reference(layer)
    g = torch.randn_like(layer.weight)

    (layer.effective_weight() * g).sum().backward()

    expected = mask_b * g * layer.weight.detach() * (beta * p * (1 - p))
    assert torch.allclose(layer.s.grad, expected, atol=1e-6)
    assert (layer.s.grad[mask_b & ~mask_a] != 0).all()
    assert torch.equal(layer.s.grad[~mask_b], torch.zeros(int((~mask_b).sum())))


def test_backward_support_is_wider_than_autograd_would_give():
    """Autograd through `w * topk_mask * p` would zero the B\\A gradients."""
    layer = make_layer()
    mask_a, mask_b, _ = reference(layer)
    g = torch.randn_like(layer.weight)
    (layer.effective_weight() * g).sum().backward()

    explore = mask_b & ~mask_a
    naive_w = mask_a * torch.sigmoid(layer.beta * layer.s.detach()) * g
    assert torch.equal(naive_w[explore], torch.zeros(int(explore.sum())))
    assert (layer.weight.grad[explore] != 0).all()


def test_w_grad_support_can_be_restricted_to_topk():
    layer = make_layer(w_grad_support="topk")
    mask_a, mask_b, p = reference(layer)
    g = torch.randn_like(layer.weight)

    (layer.effective_weight() * g).sum().backward()

    assert torch.allclose(layer.weight.grad, mask_a * p * g, atol=1e-6)
    # s still explores over the whole of B
    assert (layer.s.grad[mask_b & ~mask_a] != 0).all()


def test_zero_j_makes_both_supports_equal():
    layer = make_layer(j=0)
    mask_a, mask_b, p = reference(layer)
    assert torch.equal(mask_a, mask_b)
    g = torch.randn_like(layer.weight)
    (layer.effective_weight() * g).sum().backward()
    assert torch.allclose(layer.weight.grad, mask_a * p * g, atol=1e-6)


def test_gradients_survive_a_full_layer_backward():
    layer = make_layer(out_features=4, in_features=5)
    x = torch.randn(3, 5)
    layer(x).pow(2).sum().backward()
    _, mask_b, _ = reference(layer)
    assert layer.weight.grad is not None and layer.s.grad is not None
    assert torch.equal(layer.s.grad[~mask_b], torch.zeros(int((~mask_b).sum())))


def test_exploratory_position_can_be_promoted_into_topk():
    layer = make_layer(out_features=1, in_features=4, k=1, j=1)
    with torch.no_grad():
        layer.s.copy_(torch.tensor([[0.5, 0.4, 0.1, 0.0]]))
    assert layer.supports()[0].tolist() == [[True, False, False, False]]
    with torch.no_grad():
        layer.s[0, 1] += 0.3  # the candidate outgrows the incumbent
    mask_a, mask_b = layer.supports()
    assert mask_a.tolist() == [[False, True, False, False]]
    assert mask_b.tolist() == [[True, True, False, False]]


# --------------------------------------------------------------------------- #
# soft L0 penalty
# --------------------------------------------------------------------------- #


def test_soft_l0_penalty_value_is_over_the_wide_support_only():
    layer = make_layer(soft_l0_enabled=True, soft_l0_lambda_topk=0.5, soft_l0_lambda_explore=0.25)
    mask_a, mask_b, p = reference(layer)
    expected = 0.5 * p[mask_a].sum() + 0.25 * p[mask_b & ~mask_a].sum()
    assert layer.extra_penalty().item() == pytest.approx(expected.item(), rel=1e-5)


def test_soft_l0_penalty_gradient():
    layer = make_layer(soft_l0_enabled=True, soft_l0_lambda_topk=0.5, soft_l0_lambda_explore=0.25)
    beta = float(layer.beta)
    mask_a, mask_b, p = reference(layer)

    layer.extra_penalty().backward()

    lam = torch.where(mask_a, 0.5, torch.where(mask_b, 0.25, 0.0))
    assert torch.allclose(layer.s.grad, lam * (beta * p * (1 - p)), atol=1e-6)
    assert layer.weight.grad is None  # the penalty does not touch w


def test_penalty_and_task_gradients_add_up():
    layer = make_layer(soft_l0_enabled=True, soft_l0_lambda_topk=0.5, soft_l0_lambda_explore=0.25)
    beta = float(layer.beta)
    mask_a, mask_b, p = reference(layer)
    g = torch.randn_like(layer.weight)

    (layer.effective_weight() * g).sum().backward()
    layer.extra_penalty().backward()

    lam = torch.where(mask_a, 0.5, torch.where(mask_b, 0.25, 0.0))
    expected = mask_b * (g * layer.weight.detach() + lam) * (beta * p * (1 - p))
    assert torch.allclose(layer.s.grad, expected, atol=1e-6)


def test_penalty_is_off_by_default_and_when_lambdas_vanish():
    assert make_layer().extra_penalty() is None
    assert make_layer(soft_l0_enabled=True).extra_penalty() is None
    assert make_layer(soft_l0_lambda_topk=1.0).extra_penalty() is None  # not enabled


def test_penalty_pushes_the_gates_down():
    layer = make_layer(soft_l0_enabled=True, soft_l0_lambda_topk=1.0, soft_l0_lambda_explore=1.0)
    _, mask_b, _ = reference(layer)
    layer.extra_penalty().backward()
    assert (layer.s.grad[mask_b] > 0).all()  # gradient descent lowers s -> lowers p


# --------------------------------------------------------------------------- #
# initialization of s
# --------------------------------------------------------------------------- #


def test_magnitude_init_selects_the_largest_weights():
    layer = make_layer(out_features=4, in_features=6, k=5, j=2, s_init_mode="magnitude")
    mask_a, _ = layer.supports()
    by_magnitude = layer.weight.detach().abs().flatten().topk(5).indices
    assert set(mask_a.flatten().nonzero().flatten().tolist()) == set(by_magnitude.tolist())
    # ... and s > 0 holds on exactly the TopK support, so the hard mask,
    # the soft gate and TopK all agree at step 0
    assert torch.equal(layer.s > 0, mask_a)


def test_magnitude_init_respects_groups():
    layer = make_layer(out_features=3, in_features=8, k=2, j=1, groups="row", s_init_mode="magnitude")
    mask_a, _ = layer.supports()
    assert mask_a.sum(dim=1).tolist() == [2, 2, 2]
    assert torch.equal(layer.s > 0, mask_a)


@pytest.mark.parametrize("mode", ["constant", "uniform", "normal", "magnitude"])
def test_init_modes_produce_finite_scores(mode):
    layer = make_layer(s_init_mode=mode, s_init=0.5)
    assert layer.s.shape == layer.weight.shape
    assert torch.isfinite(layer.s).all()
    if mode == "constant":
        assert (layer.s == 0.5).all()


def test_unknown_init_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown s_init_mode"):
        make_layer(s_init_mode="magic")


# --------------------------------------------------------------------------- #
# grouping in the forward pass
# --------------------------------------------------------------------------- #


def test_row_grouping_keeps_k_per_row():
    layer = make_layer(out_features=3, in_features=8, k=2, j=2, groups="row")
    mask_a, mask_b = layer.supports()
    assert mask_a.sum(dim=1).tolist() == [2, 2, 2]
    assert mask_b.sum(dim=1).tolist() == [4, 4, 4]
    assert (layer.effective_weight() != 0).sum(dim=1).tolist() == [2, 2, 2]


def test_block_grouping_gives_n_to_m_sparsity():
    layer = make_layer(out_features=2, in_features=8, k=2, j=1, groups="block", block_size=4)
    assert layer.group_shape == (4, 4)
    nonzero = (layer.effective_weight() != 0).reshape(4, 4).sum(dim=1)
    assert nonzero.tolist() == [2, 2, 2, 2]  # 2:4


def test_topk_masks_helper_matches_a_sort():
    s = torch.randn(3, 7)
    mask_a, mask_b = topk_masks(s, k=2, j=3, shape=(3, 7))
    order = s.argsort(dim=1, descending=True)
    assert torch.equal(mask_a, torch.zeros_like(s, dtype=torch.bool).scatter_(1, order[:, :2], True))
    assert torch.equal(mask_b, torch.zeros_like(s, dtype=torch.bool).scatter_(1, order[:, :5], True))
    assert bool((mask_a & mask_b).eq(mask_a).all())  # A subset of B


# --------------------------------------------------------------------------- #
# support caching / recomputation
# --------------------------------------------------------------------------- #


def test_support_is_recomputed_after_s_changes_and_reused_otherwise():
    layer = make_layer()
    first = layer.supports()
    assert layer.supports()[0] is first[0]  # reused within a step
    with torch.no_grad():
        layer.s.add_(0.0)  # any in-place update invalidates the cache
    assert layer.supports()[0] is not first[0]


def test_support_follows_the_optimizer():
    layer = make_layer(out_features=1, in_features=4, k=1, j=3)
    opt = torch.optim.SGD([layer.s], lr=1.0)
    with torch.no_grad():
        layer.s.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert layer.supports()[0].tolist() == [[True, False, False, False]]
    layer.s.grad = torch.tensor([[10.0, -1.0, 0.0, 0.0]])  # push s0 down, s1 up
    opt.step()
    assert layer.supports()[0].tolist() == [[False, True, False, False]]


def test_turnover_tracks_topk_churn():
    layer = make_layer(out_features=1, in_features=4, k=2, j=2)
    with torch.no_grad():
        layer.s.copy_(torch.tensor([[4.0, 3.0, 2.0, 1.0]]))
    layer.supports()
    assert float(layer.turnover) == pytest.approx(0.0)
    with torch.no_grad():
        layer.s.copy_(torch.tensor([[4.0, 1.0, 3.0, 2.0]]))  # index 2 replaces index 1
    layer.supports()
    assert float(layer.turnover) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# statistics and hard-mask evaluation
# --------------------------------------------------------------------------- #


def test_soft_l0_counts_only_the_topk_support():
    layer = make_layer()
    mask_a, _, p = reference(layer)
    assert layer.soft_l0().item() == pytest.approx(p[mask_a].sum().item(), rel=1e-5)
    assert layer.soft_l0().item() <= layer.topk_numel


def test_hard_l0_is_topk_intersected_with_positive_scores():
    layer = make_layer(out_features=2, in_features=4, k=3, j=1)
    with torch.no_grad():
        layer.s.copy_(torch.tensor([[2.0, 1.0, -1.0, -2.0], [-3.0, -4.0, -5.0, -6.0]]))
    # TopK(3) = {2.0, 1.0, -1.0}; only two of those have s > 0
    assert int(layer.hard_l0()) == 2


def test_hard_mask_mode_is_binary_within_topk():
    layer = make_layer()
    layer.hard_mask = True
    mask = layer.mask()
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    mask_a, _, _ = reference(layer)
    assert torch.equal(mask.bool(), mask_a & (layer.s > 0))
    assert torch.equal(layer.effective_weight(), layer.weight * mask)


def test_large_beta_makes_the_gate_binary():
    layer = make_layer(beta=1e4, s_init_mode="normal", s_init=1.0)
    soft = layer.effective_weight()
    layer.hard_mask = True
    assert torch.allclose(soft, layer.effective_weight(), atol=1e-6)


def test_apply_hard_mask_and_export():
    layer = make_layer(out_features=3, in_features=4, k=5, j=2, s_init_mode="magnitude")
    mask_a, _ = layer.supports()
    dense = layer.to_linear()
    assert isinstance(dense, nn.Linear)
    assert torch.equal(dense.weight != 0, mask_a & (layer.s > 0))
    layer.apply_hard_mask_()
    assert torch.equal(layer.weight != 0, mask_a & (layer.s > 0))


def test_transition_fraction_is_measured_over_the_backward_support():
    layer = make_layer(out_features=2, in_features=6, k=3, j=3, beta=1.0)
    with torch.no_grad():
        layer.s.copy_(torch.tensor([[9.0, 8.0, 7.0, 1.0, 2.0, 3.0], [0.0] * 6]))
    # B is the six largest scores: 9, 8, 7, 3, 2, 1 (row 0) -- three of them
    # sit inside |beta*s| < 4
    assert float(layer.transition_fraction()) == pytest.approx(3 / 6)


def test_layer_stats_report_gates_and_turnover():
    layer = make_layer()
    stats = layer.stats()
    assert set(stats) == {"gate_mean_topk", "gate_mean_explore", "turnover"}
    mask_a, mask_b, p = reference(layer)
    assert float(stats["gate_mean_topk"]) == pytest.approx(p[mask_a].mean().item(), rel=1e-5)
    assert float(stats["gate_mean_explore"]) == pytest.approx(
        p[mask_b & ~mask_a].mean().item(), rel=1e-5
    )


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


def test_constant_inverse_temperature():
    cfg = topk_config(inverse_temperature=4.0)
    assert cfg.beta_schedule == "constant"
    assert cfg.beta_start == 4.0 and cfg.beta_end == 4.0
    ctrl = apply_sparsity(tiny_model(), cfg, max_steps=100)
    assert ctrl.set_step(0) == pytest.approx(4.0)
    assert ctrl.set_step(100) == pytest.approx(4.0)


def test_scheduled_inverse_temperature_uses_the_library_schedule():
    cfg = topk_config(
        inverse_temperature=1.0,
        inverse_temperature_schedule="exponential",
        beta_end=1e4,
    )
    assert cfg.beta_schedule == "exponential" and cfg.beta_start == 1.0
    ctrl = apply_sparsity(tiny_model(), cfg, max_steps=100)
    assert ctrl.set_step(50) == pytest.approx(100.0)  # sqrt(1e4)
    assert ctrl.set_step(100) == pytest.approx(1e4)
    for _, layer in ctrl.layers:
        assert float(layer.beta) == pytest.approx(1e4)


def test_topk_config_validation():
    with pytest.raises(ValueError, match="requires k > 0"):
        SparsityConfig(enabled=True, method="topk")
    with pytest.raises(ValueError, match="j must be >= 0"):
        topk_config(j=-1.0)
    with pytest.raises(ValueError, match="unknown topk_groups"):
        topk_config(topk_groups="rows")
    with pytest.raises(ValueError, match="unknown w_grad_support"):
        topk_config(w_grad_support="all")
    with pytest.raises(ValueError, match="both be counts"):
        topk_config(k=8, j=0.5)
    with pytest.raises(ValueError, match="target_density"):
        topk_config(target_density=0.1, target_density_coef=1.0)
    with pytest.raises(ValueError, match="only used by method=topk"):
        SparsityConfig(enabled=True, method="cs", s_init_mode="magnitude")


def test_unknown_method_is_still_rejected():
    with pytest.raises(ValueError, match="unknown sparsity method"):
        SparsityConfig(enabled=True, method="topj")


# --------------------------------------------------------------------------- #
# controller / optimizer wiring
# --------------------------------------------------------------------------- #


def test_controller_installs_topk_layers():
    model = tiny_model()
    ctrl = apply_sparsity(model, topk_config(k=0.1, j=0.05), max_steps=10)
    assert len(ctrl.layers) == 4
    assert isinstance(model.blocks[0].mlp.fc1, TopKSoftGateLinear)
    assert isinstance(model.blocks[0].mlp.fc1, SparseLinear)
    layer = model.blocks[0].mlp.fc1
    assert layer.k == round(0.1 * layer.weight.numel())
    assert sum(p.numel() for p in ctrl.mask_parameters()) == ctrl.total_maskable


def test_controller_stats_include_the_topk_budget():
    ctrl = apply_sparsity(tiny_model(), topk_config(k=0.25, j=0.25), max_steps=10)
    stats = ctrl.stats()
    assert stats["sparsity/density_topk"] == pytest.approx(0.25, abs=1e-3)
    assert stats["sparsity/density_soft"] <= stats["sparsity/density_topk"] + 1e-6
    assert stats["sparsity/density_hard"] <= stats["sparsity/density_topk"] + 1e-6
    for key in ("gate_mean_topk", "gate_mean_explore", "turnover", "beta", "transition_frac"):
        assert f"sparsity/{key}" in stats


def test_controller_adds_the_soft_l0_penalty():
    cfg = topk_config(
        soft_l0_enabled=True, soft_l0_lambda_topk=1e-3, soft_l0_lambda_explore=1e-4
    )
    ctrl = apply_sparsity(tiny_model(), cfg, max_steps=10)
    penalty, logs = ctrl.penalty()
    expected = sum(float(layer.extra_penalty().detach()) for _, layer in ctrl.layers)
    assert penalty.item() == pytest.approx(expected, rel=1e-5)
    assert logs["sparsity/soft_l0_penalty"] == pytest.approx(expected, rel=1e-5)

    penalty.backward()
    for _, layer in ctrl.layers:
        _, mask_b = layer.supports()
        assert (layer.s.grad[mask_b] > 0).all()
        assert torch.equal(layer.s.grad[~mask_b], torch.zeros(int((~mask_b).sum())))


def test_controller_penalty_is_empty_when_everything_is_off():
    ctrl = apply_sparsity(tiny_model(), topk_config(), max_steps=10)
    penalty, logs = ctrl.penalty()
    assert penalty.item() == 0.0 and penalty.requires_grad is False and logs == {}


def test_mask_lr_mult_follows_the_lr_schedule():
    model = tiny_model()
    cfg = Config()
    cfg.sparsity = topk_config(mask_lr=1e-9, mask_lr_mult=10.0)
    cfg.train.lr = 1e-3
    ctrl = apply_sparsity(model, cfg.sparsity, max_steps=10)
    opt = build_optimizer(model, cfg.train, cfg.sparsity, ctrl.mask_parameter_ids())
    groups = {g["name"]: g for g in opt.param_groups}
    assert groups["mask"]["lr"] == pytest.approx(1e-2)
    set_lr(opt, 5e-4)
    assert groups["decay"]["lr"] == pytest.approx(5e-4)
    assert groups["mask"]["lr"] == pytest.approx(5e-3)


def test_plain_mask_lr_is_still_unscheduled():
    model = tiny_model()
    cfg = Config()
    cfg.sparsity = topk_config(mask_lr=7e-3)
    ctrl = apply_sparsity(model, cfg.sparsity, max_steps=10)
    opt = build_optimizer(model, cfg.train, cfg.sparsity, ctrl.mask_parameter_ids())
    set_lr(opt, 5e-4)
    assert {g["name"]: g["lr"] for g in opt.param_groups}["mask"] == pytest.approx(7e-3)


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #


def test_model_trains_with_one_forward_and_one_backward():
    torch.manual_seed(0)
    model = tiny_model()
    cfg = topk_config(k=0.2, j=0.1, soft_l0_enabled=True, soft_l0_lambda_topk=1e-5)
    ctrl = apply_sparsity(model, cfg, max_steps=10)
    train_cfg = Config().train
    opt = build_optimizer(model, train_cfg, cfg, ctrl.mask_parameter_ids())

    x = torch.randint(0, 97, (2, 8))
    opt.zero_grad()
    _, ce = model(x, x)
    ce.backward()
    penalty, _ = ctrl.penalty()
    penalty.backward()

    # one forward, one backward: every masked layer already has both gradients,
    # confined to the Top-(k+j) support the single forward pass selected
    for _, layer in ctrl.layers:
        assert layer.weight.grad is not None and layer.s.grad is not None
        _, mask_b = layer.supports()
        assert (layer.s.grad[~mask_b] == 0).all()
        assert (layer.weight.grad[~mask_b] == 0).all()
        assert (layer.s.grad[mask_b] != 0).any()

    opt.step()  # only now may the support change
    assert all(torch.isfinite(layer.s).all() for _, layer in ctrl.layers)


def test_short_run_moves_scores_and_keeps_the_topk_budget():
    torch.manual_seed(0)
    model = tiny_model()
    cfg = topk_config(k=0.2, j=0.2, mask_lr=0.05, inverse_temperature=4.0)
    ctrl = apply_sparsity(model, cfg, max_steps=20)
    opt = build_optimizer(model, Config().train, cfg, ctrl.mask_parameter_ids())

    layer = ctrl.layers[0][1]
    start = layer.s.detach().clone()
    for _ in range(20):
        opt.zero_grad()
        x = torch.randint(0, 97, (2, 8))
        _, ce = model(x, x)
        ce.backward()
        opt.step()

    assert not torch.allclose(start, layer.s.detach())
    # the forward density is a hard budget: exactly k per group, always
    assert int((layer.effective_weight() != 0).sum()) == layer.topk_numel
    # k = round(0.2 * numel) per group, hence the rounding slack
    assert ctrl.stats()["sparsity/density_topk"] == pytest.approx(0.2, abs=1e-3)


def test_hard_mask_evaluation_runs_on_the_model():
    model = tiny_model()
    ctrl = apply_sparsity(model, topk_config(k=0.5, j=0.1), max_steps=10)
    with ctrl.hard_mask():
        logits, _ = model(torch.randint(0, 97, (1, 4)))
        assert torch.isfinite(logits).all()
    assert not any(layer.hard_mask for _, layer in ctrl.layers)


def test_compiled_model_matches_eager():
    """`train.compile=true` must not change the hand-written backward.

    TopK selection is opaque to Dynamo (`supports` is `torch.compiler.disable`d)
    precisely so that the version-counter cache does not force a recompile on
    every optimizer step.
    """
    def build():
        torch.manual_seed(0)
        model = tiny_model()
        return model, apply_sparsity(model, topk_config(k=0.2, j=0.1), max_steps=10)

    x = torch.randint(0, 97, (2, 8))
    eager, eager_ctrl = build()
    _, loss = eager(x, x)
    loss.backward()

    compiled_model, compiled_ctrl = build()
    try:
        wrapped = torch.compile(compiled_model)
        _, compiled_loss = wrapped(x, x)
    except Exception as exc:  # no working backend on this platform
        pytest.skip(f"torch.compile unavailable: {exc}")
    compiled_loss.backward()

    assert torch.allclose(loss, compiled_loss, atol=1e-6)
    for (_, a), (_, b) in zip(eager_ctrl.layers, compiled_ctrl.layers):
        assert torch.allclose(a.weight.grad, b.weight.grad, atol=1e-6)
        assert torch.allclose(a.s.grad, b.s.grad, atol=1e-6)
        _, mask_b = a.supports()
        assert (b.s.grad[~mask_b] == 0).all()


def test_checkpoint_round_trip_preserves_the_support():
    model = tiny_model()
    ctrl = apply_sparsity(model, topk_config(), max_steps=10)
    state = model.state_dict()

    clone = tiny_model()
    ctrl2 = apply_sparsity(clone, topk_config(), max_steps=10)
    clone.load_state_dict(state)

    for (_, a), (_, b) in zip(ctrl.layers, ctrl2.layers):
        assert torch.equal(a.s, b.s)
        assert torch.equal(a.supports()[0], b.supports()[0])
        assert torch.equal(a.effective_weight(), b.effective_weight())
