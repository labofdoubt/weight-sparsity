"""Rank-Boundary LapSum: hard cap forward, three boundary-gradient modes."""

import pytest
import torch

from wsparse.bottleneck import AdaptiveLapSumTopKGate, apply_activation_bottleneck
from wsparse.bottleneck.lapsum import laplace_pdf
from wsparse.config import ActivationBottleneckConfig, ModelConfig
from wsparse.model import build_model


def make_gate(mode="detach", sel="abs_topk", b0=0.0, T=1.0, k=3, j=4, n=16):
    # b0=None exercises the mode-dependent default; a float pins it
    return AdaptiveLapSumTopKGate(
        n_features=n, k=k, j=j, n_eff=3.0, selection_mode=sel,
        surrogate_mode="rblapsum", rblapsum_boundary_grad_mode=mode,
        rblapsum_boundary_floor=b0, rblapsum_temperature=T, log_diagnostics=True)


def bn_cfg(**kw):
    cfg = dict(enabled=True, n_features=64, k=8, j=24, n_eff=6.0, layers="all",
               surrogate_mode="rblapsum", selection_mode="abs_topk",
               placement="residual_out", bias=False)
    cfg.update(kw)
    return ActivationBottleneckConfig(**cfg)


def raw_a(gate, a, upstream):
    """The spec's raw score-space support gradient, computed independently."""
    scores = a.abs() if gate.selection_mode == "abs_topk" else a
    cs, ci = torch.topk(scores, gate.m, dim=-1, largest=True, sorted=True)
    z_c = torch.gather(a, -1, ci)
    up_c = torch.gather(upstream, -1, ci)
    b_rank = cs[..., gate.k:gate.k + 1]
    b = torch.clamp(b_rank, min=gate.rblapsum_boundary_floor)
    kap = laplace_pdf((cs - b) / gate.rblapsum_temperature) / gate.rblapsum_temperature
    return up_c * z_c * kap, ci, (b_rank > gate.rblapsum_boundary_floor)


def z_support_grad(gate, a, upstream):
    """dL/dz from a run, minus the ordinary hard-forward part -> support only."""
    a = a.clone().requires_grad_(True)
    gate.train()
    y = gate(a)
    y.backward(upstream)
    g = a.grad.clone()
    # ordinary part = upstream * hard_mask; subtract it to isolate support
    with torch.no_grad():
        hard = (gate(a.detach()) != 0).to(a.dtype)
    return g - upstream * hard


# --------------------------------------------------------------------------- #
# forward: cap, floor, mode-invariance
# --------------------------------------------------------------------------- #


def test_forward_identical_across_modes():
    torch.manual_seed(0)
    a = torch.randn(4, 16)
    ys = [make_gate(mode=m).eval()(a.clone()) for m in ("detach", "project", "through_rank")]
    assert torch.equal(ys[0], ys[1]) and torch.equal(ys[1], ys[2])


def test_hard_sparsity_at_most_k():
    torch.manual_seed(1)
    g = make_gate(b0=0.0)
    a = torch.randn(8, 16)
    y = g(a)
    assert int((y != 0).sum(-1).max()) <= g.k


def test_under_full_token_keeps_only_scores_above_floor():
    g = make_gate(k=4, j=6, n=16, b0=2.0)
    # exactly 2 scores exceed b0=2.0
    a = torch.tensor([[5.0, 3.0, 1.5, 1.0, 0.5, 0.2] + [0.1] * 10])
    y = g(a)
    active = (y != 0)[0]
    assert int(active.sum()) == 2
    assert active[0] and active[1] and not active[2]


def test_over_full_token_caps_at_k():
    g = make_gate(k=3, j=4, n=16, b0=0.0)
    a = torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0] + [0.0] * 9])
    y = g(a)
    assert int((y != 0).sum()) == 3
    assert torch.equal((y != 0)[0, :3], torch.ones(3, dtype=torch.bool))


def test_candidates_outside_topkj_get_zero_support():
    torch.manual_seed(2)
    g = make_gate(k=3, j=4, n=16)
    a = torch.randn(2, 16, requires_grad=True)
    g.train()
    y = g(a)
    y.backward(torch.randn_like(y))
    scores = a.detach().abs()
    _, ci = torch.topk(scores, g.m, dim=-1)
    outside = torch.ones(2, 16, dtype=torch.bool)
    for r in range(2):
        outside[r, ci[r]] = False
    assert a.grad[outside].abs().sum().item() == 0.0


# --------------------------------------------------------------------------- #
# backward: the three modes in score space
# --------------------------------------------------------------------------- #


def cap_active_example(sel="abs_topk", b0=0.0):
    # positive scores well above b0 -> the rank cap binds for every row
    g = make_gate(mode="detach", sel=sel, b0=b0, k=3, j=4, n=16)
    a = torch.tensor([[6.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5] + [0.1] * 9])
    up = torch.randn(1, 16)
    return g, a, up


def test_detach_mode_support_equals_raw_a():
    for sel in ("abs_topk", "topk"):
        g, a, up = cap_active_example(sel)
        a_ref, ci, cap = raw_a(g, a, up)
        assert bool(cap.all())
        gz = z_support_grad(g, a, up)
        gz_c = torch.gather(gz, -1, ci)
        sign_c = torch.gather(a, -1, ci).sign() if sel == "abs_topk" else 1.0
        assert torch.allclose(gz_c, (sign_c * a_ref if sel == "abs_topk" else a_ref),
                              atol=1e-5), sel


def test_project_mode_cap_active_sums_to_zero():
    g, a, up = cap_active_example()
    g.rblapsum_boundary_grad_mode = "project"
    a_ref, ci, cap = raw_a(g, a, up)
    # reproduce the expected g_s in score space and check zero-sum
    g_s = a_ref - a_ref.mean(-1, keepdim=True)
    assert float(g_s.sum(-1).abs().max()) < 1e-5
    gz = z_support_grad(g, a, up)
    gz_c = torch.gather(gz, -1, ci)
    sign_c = torch.gather(a, -1, ci).sign()
    assert torch.allclose(gz_c, sign_c * g_s, atol=1e-5)


def test_project_mode_floor_active_no_projection():
    # b0 above the K+1 score -> floor wins, cap inactive -> g_s == a
    g = make_gate(mode="project", b0=3.5, k=3, j=4, n=16)
    a = torch.tensor([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5] + [0.1] * 9])
    up = torch.randn(1, 16)
    a_ref, ci, cap = raw_a(g, a, up)
    assert not bool(cap.any())          # floor active everywhere
    gz = z_support_grad(g, a, up)
    gz_c = torch.gather(gz, -1, ci)
    sign_c = torch.gather(a, -1, ci).sign()
    assert torch.allclose(gz_c, sign_c * a_ref, atol=1e-5)   # == a, unprojected


def test_through_rank_cap_active_subtracts_sum_at_boundary():
    g, a, up = cap_active_example()
    g.rblapsum_boundary_grad_mode = "through_rank"
    a_ref, ci, cap = raw_a(g, a, up)
    g_s = a_ref.clone()
    g_s[..., g.k:g.k + 1] -= a_ref.sum(-1, keepdim=True)
    assert float(g_s.sum(-1).abs().max()) < 1e-5
    gz = z_support_grad(g, a, up)
    gz_c = torch.gather(gz, -1, ci)
    sign_c = torch.gather(a, -1, ci).sign()
    assert torch.allclose(gz_c, sign_c * g_s, atol=1e-5)


def test_through_rank_kappa_distributes_the_correction():
    g, a, up = cap_active_example()
    g.rblapsum_boundary_grad_mode = "through_rank_kappa"
    a_ref, ci, cap = raw_a(g, a, up)
    assert bool(cap.all())
    # expected: g_s = a - q * sum(a), q = kappa / sum(kappa) over the pool
    from wsparse.bottleneck.lapsum import laplace_pdf
    scores = a.abs()
    cs, _ = torch.topk(scores, g.m, dim=-1, largest=True, sorted=True)
    b = torch.clamp(cs[..., g.k:g.k + 1], min=g.rblapsum_boundary_floor)
    kap = laplace_pdf((cs - b) / g.rblapsum_temperature) / g.rblapsum_temperature
    q = kap / kap.sum(-1, keepdim=True)
    g_s = a_ref - q.double() * a_ref.sum(-1, keepdim=True)
    assert float(g_s.sum(-1).abs().max()) < 1e-5              # still zero-sum
    gz = z_support_grad(g, a, up)
    gz_c = torch.gather(gz, -1, ci)
    sign_c = torch.gather(a, -1, ci).sign()
    assert torch.allclose(gz_c, (sign_c * g_s).float(), atol=1e-5)
    # and the boundary feature is NOT a point sink: its correction share is
    # proportional to its kappa, like everyone else's
    corr = (a_ref - g_s).abs()
    share = corr / corr.sum(-1, keepdim=True)
    qn = q / q.sum(-1, keepdim=True)
    assert torch.allclose(share.float(), qn.float(), atol=1e-5)


def test_through_rank_kappa_common_shift_invariant():
    d = abs(common_shift_deriv("through_rank_kappa"))
    assert d < 1e-6


def test_through_rank_floor_active_no_correction():
    g = make_gate(mode="through_rank", b0=3.5, k=3, j=4, n=16)
    a = torch.tensor([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5] + [0.1] * 9])
    up = torch.randn(1, 16)
    a_ref, ci, cap = raw_a(g, a, up)
    assert not bool(cap.any())
    gz = z_support_grad(g, a, up)
    gz_c = torch.gather(gz, -1, ci)
    sign_c = torch.gather(a, -1, ci).sign()
    assert torch.allclose(gz_c, sign_c * a_ref, atol=1e-5)   # == a


def test_abstopk_maps_back_with_sign_z():
    # a candidate with negative z: its support gradient must flip sign vs |z|-space
    g = make_gate(mode="detach", sel="abs_topk", k=2, j=3, n=8)
    a = torch.tensor([[5.0, -4.0, 3.0, -1.0, 0.5, 0.2, 0.1, 0.05]])
    up = torch.ones(1, 8)
    a_ref, ci, _ = raw_a(g, a, up)
    gz = z_support_grad(g, a, up)
    gz_c = torch.gather(gz, -1, ci)
    sign_c = torch.gather(a, -1, ci).sign()
    assert torch.allclose(gz_c, sign_c * a_ref, atol=1e-5)
    # concretely: candidate 1 has z=-4, so its z-support has opposite sign to a
    assert (gz_c[0, 1] * a_ref[0, 1]) <= 0


# --------------------------------------------------------------------------- #
# common-shift invariance: the whole point of the ablation
# --------------------------------------------------------------------------- #


def common_shift_deriv(mode):
    """Directional derivative of L along the all-ones score direction."""
    g = make_gate(mode=mode, sel="topk", b0=-100.0, k=3, j=4, n=16)  # b0 low: cap active
    torch.manual_seed(5)
    base = torch.rand(1, 16) + 2.0          # positive, cap binds
    up = torch.randn(1, 16)
    # directional derivative of the loss along the all-ones score shift equals
    # the sum over candidates of the support gradient in score space
    a = base.clone().requires_grad_(True)
    y = g(a); (y * up).sum().backward()
    # dL/ds = dL/dz for topk; project onto all-ones = sum of candidate grads
    scores = a.detach()
    _, ci = torch.topk(scores, g.m, dim=-1)
    with torch.no_grad():
        hard = (g(a.detach()) != 0).to(a.dtype)
    support = a.grad - up * hard
    return float(torch.gather(support, -1, ci).sum())


def test_common_shift_invariance_by_mode():
    d_detach = abs(common_shift_deriv("detach"))
    d_project = abs(common_shift_deriv("project"))
    d_through = abs(common_shift_deriv("through_rank"))
    assert d_detach > 1e-6                 # detached keeps the common mode
    assert d_project < 1e-6                # project removes it
    assert d_through < 1e-6                # through_rank removes it too


# --------------------------------------------------------------------------- #
# integration, diagnostics, config, no-regression
# --------------------------------------------------------------------------- #


def test_controller_diagnostics_and_backward():
    torch.manual_seed(6)
    model = build_model(ModelConfig(vocab_size=97, max_seq_len=32, n_layers=2,
                                    d_model=32, n_heads=4))
    ctl = apply_activation_bottleneck(
        model, bn_cfg(rblapsum_boundary_floor=0.3,
                      rblapsum_boundary_grad_mode="through_rank"), max_steps=10)
    model.train()
    x = torch.randint(0, 97, (2, 16))
    _, loss = model(x, x)
    loss.backward()
    stats = ctl.stats()
    for key in ("bottleneck/active_count", "bottleneck/rb_cap_active_frac",
                "bottleneck/rb_b_rank", "bottleneck/rb_boundary",
                "bottleneck/rb_support_grad_norm"):
        assert key in stats, key
    assert stats["bottleneck/active_count"] <= 8 + 1e-6


def test_boundary_floor_default_is_mode_dependent():
    # abs_topk -> 0.1, topk -> 0.0, explicit value honored either way
    assert bn_cfg(selection_mode="abs_topk").rblapsum_boundary_floor == 0.1
    assert bn_cfg(selection_mode="topk").rblapsum_boundary_floor == 0.0
    assert bn_cfg(selection_mode="abs_topk",
                  rblapsum_boundary_floor=0.5).rblapsum_boundary_floor == 0.5
    assert bn_cfg(selection_mode="abs_topk",
                  rblapsum_boundary_floor=0.0).rblapsum_boundary_floor == 0.0
    # the directly-constructed gate resolves the same way
    assert make_gate(sel="abs_topk", b0=None).rblapsum_boundary_floor == 0.1
    assert make_gate(sel="topk", b0=None).rblapsum_boundary_floor == 0.0


def test_config_validation():
    with pytest.raises(ValueError):
        bn_cfg(selection_mode="gated_topk")
    with pytest.raises(ValueError):
        bn_cfg(rblapsum_boundary_grad_mode="bogus")
    with pytest.raises(ValueError):
        bn_cfg(rblapsum_temperature=0.0)
    with pytest.raises(ValueError):
        bn_cfg(rblapsum_kernel="gaussian")
    with pytest.raises(ValueError):
        bn_cfg(project_scale_gradient=True)
    cfg = bn_cfg(selection_mode="topk")
    assert cfg.rblapsum_boundary_grad_mode == "detach"


def test_lapsum_gate_unaffected():
    lap = AdaptiveLapSumTopKGate(n_features=16, k=3, j=4, n_eff=3.0,
                                 surrogate_mode="lapsum_scheduled")
    assert [n for n, _ in lap.named_parameters()] == []
    torch.manual_seed(7)
    a = torch.randn(2, 16, requires_grad=True)
    lap.train()
    lap.scheduled_temperature.fill_(0.5)
    y = lap(a)
    assert int((y != 0).sum(-1).max()) == 3        # exactly K, LapSum unchanged


# --------------------------------------------------------------------------- #
# temperature servo (rblapsum_temperature_mode="servo")
# --------------------------------------------------------------------------- #


def make_servo_gate(T=1.0, k=8, j=24, n=64, **kw):
    return AdaptiveLapSumTopKGate(
        n_features=n, k=k, j=j, n_eff=3.0, selection_mode="abs_topk",
        surrogate_mode="rblapsum",
        rblapsum_boundary_grad_mode="through_rank_kappa",
        rblapsum_boundary_floor=0.0, rblapsum_temperature=T,
        rblapsum_temperature_mode="servo", log_diagnostics=True, **kw)


def tight_batch(n=64, spacing=1e-3, base=1.0, rows=4):
    """Candidate scores packed within the window: population guard stays off."""
    a = torch.full((rows, n), 1e-4)
    vals = base + spacing * torch.arange(40, dtype=torch.float32)
    a[:, :40] = vals.flip(0)
    return a


def test_servo_fixed_mode_has_no_state():
    g = make_gate(mode="through_rank_kappa", T=1.0)
    assert not hasattr(g, "rb_temp")
    assert "rb_temp" not in g.state_dict()


def test_servo_population_guard_raises_T():
    g = make_servo_gate(T=0.5)
    a = torch.zeros(2, 64)
    a[:, :32] = torch.linspace(3200.0, 100.0, 32)  # spacing 100 >> T
    g.train()
    g(a)
    assert torch.isclose(g.rb_temp, torch.tensor(0.55), atol=1e-6)
    g(a)
    assert torch.isclose(g.rb_temp, torch.tensor(0.605), atol=1e-6)
    g.eval()
    g(a)
    assert torch.isclose(g.rb_temp, torch.tensor(0.605), atol=1e-6)  # eval: frozen


def test_servo_trim_raises_T_under_kernel_pressure():
    torch.manual_seed(0)
    g = make_servo_gate(T=1.0)
    a = tight_batch()
    g.train()
    for _ in range(30):
        x = a.clone().requires_grad_(True)
        y = g(x)
        y.backward(torch.ones_like(y))  # big upstream -> big kick -> chi >> target
    assert 1.2 < float(g.rb_temp) <= 1.02 ** 30 + 1e-6
    assert float(g.rb_temp) <= g.rblapsum_t_max


def test_servo_trim_lowers_T_when_quiet():
    torch.manual_seed(0)
    g = make_servo_gate(T=1.0)
    a = tight_batch()
    g.train()
    for _ in range(30):
        x = a.clone().requires_grad_(True)
        y = g(x)
        y.backward(1e-12 * torch.ones_like(y))  # negligible kick -> chi << target
    assert g.rblapsum_t_min <= float(g.rb_temp) < 0.9


def test_servo_state_roundtrip():
    g = make_servo_gate(T=1.0)
    a = torch.zeros(2, 64)
    a[:, :32] = torch.linspace(3200.0, 100.0, 32)
    g.train()
    g(a)
    sd = g.state_dict()
    assert "rb_temp" in sd and "rb_kick_ema" in sd and "rb_delta_ema" in sd
    g2 = make_servo_gate(T=1.0)
    g2.load_state_dict(sd)
    assert torch.isclose(g2.rb_temp, g.rb_temp)


def test_servo_validation():
    with pytest.raises(ValueError):
        AdaptiveLapSumTopKGate(
            n_features=64, k=8, j=24, n_eff=3.0, selection_mode="abs_topk",
            surrogate_mode="rblapsum",
            rblapsum_boundary_grad_mode="through_rank_kappa",
            rblapsum_temperature_mode="auto")  # unknown mode
    with pytest.raises(ValueError):
        make_servo_gate(T=10.0)                     # T0 above t_max
    with pytest.raises(ValueError):
        make_servo_gate(T=1.0, k=4)                 # spacing needs k > 4
    with pytest.raises(ValueError):
        make_servo_gate(T=1.0, j=4)                 # spacing needs j >= 8
    with pytest.raises(ValueError):
        bn_cfg(rblapsum_temperature_mode="servo", rblapsum_t_min=2.0)
    with pytest.raises(ValueError):
        bn_cfg(rblapsum_temperature_mode="servo", rblapsum_chi_target=0.0)
    # a valid servo config passes end to end
    assert bn_cfg(rblapsum_temperature_mode="servo").rblapsum_temperature_mode == "servo"


def test_servo_through_model_and_stats():
    torch.manual_seed(7)
    model = build_model(ModelConfig(vocab_size=97, max_seq_len=32, n_layers=2,
                                    d_model=32, n_heads=4))
    ctl = apply_activation_bottleneck(
        model, bn_cfg(rblapsum_boundary_grad_mode="through_rank_kappa",
                      rblapsum_temperature_mode="servo"), max_steps=10)
    model.train()
    x = torch.randint(0, 97, (2, 16))
    _, loss = model(x, x)
    loss.backward()
    _, loss = model(x, x)   # second step: servo diag now has kick history
    loss.backward()
    stats = ctl.stats()
    for key in ("bottleneck/rb_temp", "bottleneck/rb_win_count",
                "bottleneck/rb_chi"):
        assert key in stats, key
    assert stats["bottleneck/rb_temp"] > 0
