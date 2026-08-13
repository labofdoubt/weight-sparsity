import pytest
import torch
import torch.nn as nn

from wsparse.config import Config, ModelConfig, SparsityConfig
from wsparse.model import build_model
from wsparse.optim import build_optimizer
from wsparse.sparsity import CSLinear, LTPLinear, SparseLinear, apply_sparsity, build_beta_schedule


def tiny_model(**kw):
    cfg = dict(vocab_size=97, max_seq_len=32, n_layers=2, d_model=32, n_heads=4, mlp_ratio=4.0)
    cfg.update(kw)
    return build_model(ModelConfig(**cfg))


# --------------------------------------------------------------------------- #
# beta schedules
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", ["constant", "linear", "exponential", "cosine", "polynomial"])
def test_schedule_endpoints_and_monotonicity(kind):
    sched = build_beta_schedule(kind, 1.0, 100.0, warmup_steps=10, max_steps=110)
    assert sched(0) == pytest.approx(1.0)
    assert sched(10) == pytest.approx(1.0)
    if kind == "constant":
        assert sched(110) == pytest.approx(1.0)
        return
    assert sched(110) == pytest.approx(100.0)
    assert sched(500) == pytest.approx(100.0)  # clamped after annealing
    values = [sched(s) for s in range(0, 120, 5)]
    assert all(b <= a + 1e-9 for a, b in zip(values, values[1:])) is False  # increasing
    assert all(a <= b + 1e-9 for a, b in zip(values, values[1:]))


def test_exponential_is_geometric():
    sched = build_beta_schedule("exponential", 1.0, 1e4, max_steps=100)
    assert sched(50) == pytest.approx(100.0)  # sqrt(1e4)


def test_default_anneal_covers_remaining_steps():
    sched = build_beta_schedule("linear", 1.0, 11.0, warmup_steps=100, max_steps=1100)
    assert sched.anneal_steps == 1000
    assert sched(600) == pytest.approx(6.0)


# --------------------------------------------------------------------------- #
# masks
# --------------------------------------------------------------------------- #


def test_ltp_mask_formula():
    linear = nn.Linear(4, 3, bias=False)
    layer = LTPLinear(linear, threshold_init=0.01)
    layer.beta.fill_(50.0)
    expected = torch.sigmoid(50.0 * (linear.weight.float() ** 2 - 0.01))
    assert torch.allclose(layer.mask(), expected)
    assert layer.soft_l0().item() == pytest.approx(expected.sum().item(), rel=1e-5)


def test_cs_mask_formula():
    linear = nn.Linear(4, 3, bias=False)
    layer = CSLinear(linear, s_init=0.05)
    layer.beta.fill_(7.0)
    assert torch.allclose(layer.mask(), torch.sigmoid(torch.tensor(7.0 * 0.05)).expand(3, 4))
    assert layer.s.shape == linear.weight.shape


def test_hard_mask_is_binary_and_matches_sign():
    linear = nn.Linear(6, 5, bias=False)
    layer = CSLinear(linear, s_init=0.0)
    with torch.no_grad():
        layer.s.normal_()
    layer.hard_mask = True
    mask = layer.mask()
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    assert torch.equal(mask.bool(), layer.s > 0)
    assert layer.hard_l0().item() == mask.sum().item()


def test_large_beta_makes_soft_and_hard_agree():
    linear = nn.Linear(8, 8, bias=False)
    layer = CSLinear(linear, s_init=0.0)
    with torch.no_grad():
        layer.s.normal_(std=1.0)
    layer.beta.fill_(1e4)
    soft = layer.mask()
    layer.hard_mask = True
    assert torch.allclose(soft, layer.mask(), atol=1e-6)


def test_ltp_gradient_through_mask():
    """dv/dw = m + w * dm/dw when the mask is differentiable w.r.t. w."""
    torch.manual_seed(0)
    linear = nn.Linear(4, 3, bias=False)
    layer = LTPLinear(linear, threshold_init=0.001, grad_through_mask=True)
    layer.beta.fill_(100.0)
    g = torch.randn_like(layer.weight)

    (layer.effective_weight() * g).sum().backward()

    w = layer.weight.detach()
    m = torch.sigmoid(100.0 * (w**2 - 0.001))
    dm_dw = 100.0 * 2 * w * m * (1 - m)
    assert torch.allclose(layer.weight.grad, g * (m + w * dm_dw), atol=1e-4)


def test_ltp_gradient_not_through_mask():
    """With grad_through_mask=False the weight sees dv/dw = m (eq. 14),
    but the threshold still receives its full gradient."""
    torch.manual_seed(0)
    linear = nn.Linear(4, 3, bias=False)
    layer = LTPLinear(linear, threshold_init=0.001, grad_through_mask=False)
    layer.beta.fill_(100.0)
    g = torch.randn_like(layer.weight)

    (layer.effective_weight() * g).sum().backward()

    w = layer.weight.detach()
    m = torch.sigmoid(100.0 * (w**2 - 0.001))
    assert torch.allclose(layer.weight.grad, g * m, atol=1e-6)

    dm_dtau = -100.0 * m * (1 - m)
    assert layer.threshold.grad is not None
    assert layer.threshold.grad.item() == pytest.approx((g * w * dm_dtau).sum().item(), rel=1e-4)


def test_cs_auxiliary_parameter_receives_gradient():
    linear = nn.Linear(4, 3, bias=False)
    layer = CSLinear(linear, s_init=0.05)
    layer.beta.fill_(2.0)
    layer.effective_weight().sum().backward()
    assert layer.s.grad is not None and layer.s.grad.abs().sum() > 0
    assert layer.weight.grad is not None


def test_forward_equals_masked_linear():
    linear = nn.Linear(5, 4, bias=False)
    layer = LTPLinear(linear, threshold_init=0.0)
    layer.beta.fill_(10.0)
    x = torch.randn(2, 7, 5)
    assert torch.allclose(layer(x), x @ layer.effective_weight().T, atol=1e-6)


# --------------------------------------------------------------------------- #
# controller
# --------------------------------------------------------------------------- #


def test_only_mlp_is_masked_by_default():
    model = tiny_model()
    ctrl = apply_sparsity(model, SparsityConfig(enabled=True, method="ltp"), max_steps=10)
    names = [n for n, _ in ctrl.layers]
    assert names == [
        "blocks.0.mlp.fc1",
        "blocks.0.mlp.fc2",
        "blocks.1.mlp.fc1",
        "blocks.1.mlp.fc2",
    ]
    assert isinstance(model.blocks[0].mlp.fc1, LTPLinear)
    assert not isinstance(model.blocks[0].attn.qkv, SparseLinear)
    assert not isinstance(model.lm_head, SparseLinear)


def test_attention_can_be_targeted_too():
    model = tiny_model()
    ctrl = apply_sparsity(
        model, SparsityConfig(enabled=True, method="cs", targets=["mlp", "attn"]), max_steps=10
    )
    assert len(ctrl.layers) == 8  # (fc1, fc2, qkv, proj) x 2 blocks
    assert isinstance(model.blocks[0].attn.qkv, CSLinear)


def test_masking_preserves_the_weight_parameter():
    model = tiny_model()
    original = model.blocks[0].mlp.fc1.weight
    apply_sparsity(model, SparsityConfig(enabled=True, method="ltp"), max_steps=10)
    assert model.blocks[0].mlp.fc1.weight is original


def test_disabled_controller_is_a_noop():
    model = tiny_model()
    ctrl = apply_sparsity(model, SparsityConfig(enabled=False), max_steps=10)
    assert ctrl.layers == []
    assert ctrl.penalty()[0].item() == 0.0
    assert ctrl.stats() == {}
    assert isinstance(model.blocks[0].mlp.fc1, nn.Linear)


def test_beta_is_broadcast_to_layers():
    model = tiny_model()
    cfg = SparsityConfig(
        enabled=True, method="cs", beta_schedule="exponential", beta_start=1.0, beta_end=100.0
    )
    ctrl = apply_sparsity(model, cfg, max_steps=100)
    assert ctrl.set_step(0) == pytest.approx(1.0)
    ctrl.set_step(50)
    assert ctrl.beta == pytest.approx(10.0)
    for _, layer in ctrl.layers:
        assert float(layer.beta) == pytest.approx(10.0)


def test_model_still_runs_after_masking():
    model = tiny_model()
    apply_sparsity(model, SparsityConfig(enabled=True, method="cs"), max_steps=10)
    logits, loss = model(torch.randint(0, 97, (2, 8)), torch.randint(0, 97, (2, 8)))
    assert logits.shape == (2, 8, 97)
    loss.backward()
    assert model.blocks[0].mlp.fc1.s.grad is not None


# --------------------------------------------------------------------------- #
# objective terms
# --------------------------------------------------------------------------- #


def test_l0_penalty_normalisation():
    model = tiny_model()
    cfg = SparsityConfig(enabled=True, method="cs", l0_coef=1.0, l0_normalize=True, s_init=0.0)
    ctrl = apply_sparsity(model, cfg, max_steps=10)
    penalty, logs = ctrl.penalty()
    # s = 0 everywhere -> every mask is exactly 0.5 -> normalised L0 = 0.5
    assert penalty.item() == pytest.approx(0.5, rel=1e-5)
    assert logs["sparsity/density_soft"] == pytest.approx(0.5, rel=1e-5)

    cfg_raw = SparsityConfig(enabled=True, method="cs", l0_coef=1.0, l0_normalize=False, s_init=0.0)
    ctrl_raw = apply_sparsity(tiny_model(), cfg_raw, max_steps=10)
    assert ctrl_raw.penalty()[0].item() == pytest.approx(0.5 * ctrl_raw.total_maskable, rel=1e-5)


def test_target_density_penalty_is_zero_at_the_target():
    model = tiny_model()
    cfg = SparsityConfig(
        enabled=True,
        method="cs",
        l0_coef=0.0,
        target_density=0.5,
        target_density_coef=1.0,
        s_init=0.0,  # every mask = 0.5 -> soft L0 is exactly half of numel
    )
    ctrl = apply_sparsity(model, cfg, max_steps=10)
    penalty, logs = ctrl.penalty()
    assert penalty.item() == pytest.approx(0.0, abs=1e-8)
    assert logs["sparsity/density_soft"] == pytest.approx(0.5, rel=1e-5)


def test_target_density_penalty_value_and_gradient_sign():
    model = tiny_model()
    cfg = SparsityConfig(
        enabled=True, method="cs", l0_coef=0.0, target_density=0.25,
        target_density_coef=1.0, s_init=0.0,
    )
    ctrl = apply_sparsity(model, cfg, max_steps=10)
    penalty, _ = ctrl.penalty()
    # L0/D = 0.5 / 0.25 = 2  ->  (2 - 1)**2 = 1
    assert penalty.item() == pytest.approx(1.0, rel=1e-5)
    penalty.backward()
    # too dense -> the penalty pushes every s down
    assert (model.blocks[0].mlp.fc1.s.grad > 0).all()


def test_per_layer_target_density_overrides():
    model = tiny_model()
    cfg = SparsityConfig(
        enabled=True,
        method="cs",
        target_density=0.1,
        target_density_coef=1.0,
        target_density_overrides={"blocks.0.mlp.*": 0.5, "*.mlp.fc2": 0.3},
    )
    ctrl = apply_sparsity(model, cfg, max_steps=10)
    # later patterns win, so fc2 gets 0.3 in both blocks
    assert ctrl.target_density["blocks.0.mlp.fc1"] == 0.5
    assert ctrl.target_density["blocks.0.mlp.fc2"] == 0.3
    assert ctrl.target_density["blocks.1.mlp.fc1"] == 0.1
    assert ctrl.target_density["blocks.1.mlp.fc2"] == 0.3


def test_target_density_uses_actual_layer_sizes():
    model = tiny_model(d_model=32, mlp_ratio=4.0)
    cfg = SparsityConfig(
        enabled=True, method="cs", target_density=0.2, target_density_coef=1.0, s_init=0.0
    )
    ctrl = apply_sparsity(model, cfg, max_steps=10)
    layer = dict(ctrl.layers)["blocks.0.mlp.fc1"]
    assert layer.mask_numel == 128 * 32
    assert ctrl.total_maskable == 4 * 128 * 32


def test_zero_coefficients_give_no_penalty_graph():
    model = tiny_model()
    ctrl = apply_sparsity(model, SparsityConfig(enabled=True, method="cs"), max_steps=10)
    penalty, logs = ctrl.penalty()
    assert penalty.item() == 0.0
    assert penalty.requires_grad is False
    assert logs == {}


# --------------------------------------------------------------------------- #
# statistics / evaluation
# --------------------------------------------------------------------------- #


def test_stats_report_density_and_beta():
    model = tiny_model()
    cfg = SparsityConfig(enabled=True, method="cs", s_init=0.0, beta_start=3.0, beta_end=3.0,
                         beta_schedule="constant")
    ctrl = apply_sparsity(model, cfg, max_steps=10)
    stats = ctrl.stats()
    assert stats["sparsity/beta"] == pytest.approx(3.0)
    assert stats["sparsity/density_soft"] == pytest.approx(0.5, rel=1e-5)
    assert stats["sparsity/density_hard"] == pytest.approx(0.0)  # s == 0 is not > 0
    assert stats["sparsity/maskable_params"] == ctrl.total_maskable
    assert "sparsity/s_mean" in stats
    assert set(ctrl.layer_densities()) == {n for n, _ in ctrl.layers}


def test_hard_mask_context_restores_state():
    model = tiny_model()
    ctrl = apply_sparsity(model, SparsityConfig(enabled=True, method="cs"), max_steps=10)
    with ctrl.hard_mask():
        assert all(layer.hard_mask for _, layer in ctrl.layers)
        logits, _ = model(torch.randint(0, 97, (1, 4)))
        assert torch.isfinite(logits).all()
    assert not any(layer.hard_mask for _, layer in ctrl.layers)


def test_apply_hard_masks_zeroes_weights():
    model = tiny_model()
    ctrl = apply_sparsity(model, SparsityConfig(enabled=True, method="cs", s_init=0.0), max_steps=10)
    layer = ctrl.layers[0][1]
    with torch.no_grad():
        layer.s.copy_(torch.where(torch.rand_like(layer.s) > 0.5, 1.0, -1.0))
    ctrl.apply_hard_masks_()
    assert torch.equal((layer.weight != 0), (layer.s > 0))
    dense = layer.to_linear()
    assert isinstance(dense, nn.Linear) and torch.equal(dense.weight, layer.weight)


def test_transition_fraction_collapses_as_beta_grows():
    model = tiny_model()
    ctrl = apply_sparsity(
        model,
        SparsityConfig(enabled=True, method="cs", beta_start=1.0, beta_end=1e6),
        max_steps=100,
    )
    ctrl.set_step(0)
    early = ctrl.stats()["sparsity/transition_frac"]
    ctrl.set_step(100)
    late = ctrl.stats()["sparsity/transition_frac"]
    assert early == pytest.approx(1.0)
    assert late < 0.01


# --------------------------------------------------------------------------- #
# optimiser wiring
# --------------------------------------------------------------------------- #


def test_mask_parameters_get_their_own_group():
    cfg = Config()
    cfg.model = ModelConfig(vocab_size=97, max_seq_len=32, n_layers=2, d_model=32, n_heads=4)
    cfg.sparsity = SparsityConfig(enabled=True, method="cs", mask_lr=1e-3)
    cfg.train.lr = 0.1
    cfg.train.weight_decay = 0.7
    model = build_model(cfg.model)
    ctrl = apply_sparsity(model, cfg.sparsity, max_steps=10)
    opt = build_optimizer(model, cfg.train, cfg.sparsity, ctrl.mask_parameter_ids())

    groups = {g["name"]: g for g in opt.param_groups}
    assert set(groups) == {"decay", "nodecay", "mask"}
    assert groups["mask"]["lr"] == 1e-3
    assert groups["mask"]["weight_decay"] == 0.0
    assert groups["decay"]["weight_decay"] == 0.7
    mask_ids = ctrl.mask_parameter_ids()
    assert {id(p) for p in groups["mask"]["params"]} == mask_ids
    # no mask parameter leaked into a decayed group
    for name in ("decay", "nodecay"):
        assert not ({id(p) for p in groups[name]["params"]} & mask_ids)


def test_ltp_thresholds_are_one_scalar_per_layer():
    model = tiny_model()
    ctrl = apply_sparsity(model, SparsityConfig(enabled=True, method="ltp"), max_steps=10)
    params = ctrl.mask_parameters()
    assert len(params) == len(ctrl.layers)
    assert all(p.numel() == 1 for p in params)


def test_cs_mask_parameters_match_weight_shapes():
    model = tiny_model()
    ctrl = apply_sparsity(model, SparsityConfig(enabled=True, method="cs"), max_steps=10)
    assert sum(p.numel() for p in ctrl.mask_parameters()) == ctrl.total_maskable


def test_parameter_count_excludes_sparsity_parameters():
    dense = tiny_model()
    dense_count = dense.num_parameters()
    model = tiny_model()
    ctrl = apply_sparsity(model, SparsityConfig(enabled=True, method="cs"), max_steps=10)
    assert model.num_parameters() == dense_count
    assert model.num_parameters(include_mask=True) == dense_count + ctrl.total_maskable


def test_set_lr_leaves_mask_group_alone():
    from wsparse.optim import set_lr

    model = tiny_model()
    cfg = Config()
    cfg.sparsity = SparsityConfig(enabled=True, method="cs", mask_lr=5e-4)
    ctrl = apply_sparsity(model, cfg.sparsity, max_steps=10)
    opt = build_optimizer(model, cfg.train, cfg.sparsity, ctrl.mask_parameter_ids())
    set_lr(opt, 0.123)
    groups = {g["name"]: g for g in opt.param_groups}
    assert groups["decay"]["lr"] == 0.123
    assert groups["mask"]["lr"] == 5e-4


def test_sparsity_requires_matching_layers():
    class Empty(nn.Module):
        def forward(self, x):
            return x

    with pytest.raises(ValueError, match="no layers matched"):
        apply_sparsity(Empty(), SparsityConfig(enabled=True), max_steps=10)


def test_training_step_reduces_density_with_target_objective():
    """A short optimisation run should move density towards the target."""
    torch.manual_seed(0)
    model = tiny_model()
    cfg = SparsityConfig(
        enabled=True,
        method="cs",
        s_init=1.0,
        beta_schedule="constant",
        beta_start=5.0,
        beta_end=5.0,
        mask_lr=0.02,
        target_density=0.2,
        target_density_coef=1.0,
    )
    ctrl = apply_sparsity(model, cfg, max_steps=50)
    train_cfg = Config().train
    opt = build_optimizer(model, train_cfg, cfg, ctrl.mask_parameter_ids())
    start = ctrl.stats()["sparsity/density_soft"]
    for _ in range(120):
        opt.zero_grad()
        penalty, _ = ctrl.penalty()
        penalty.backward()
        opt.step()
    end = ctrl.stats()["sparsity/density_soft"]
    assert end < start
    assert end == pytest.approx(0.2, abs=0.1)
