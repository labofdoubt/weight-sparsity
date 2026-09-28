"""rblapsum_sf: soft forward y = z * p over the Top(K+J) pool.

The decisive test is exactness: with boundary_grad_mode="through_rank" and
support scale 1.0 the custom backward must equal torch autograd's gradient of
the reference forward, since b literally is the (K+1)-st score.  The kappa
mode redistributes the same total; detach drops the boundary term; the support
scale interpolates the boundary term linearly.
"""

import pytest
import torch

from wsparse.bottleneck import AdaptiveLapSumTopKGate
from wsparse.bottleneck.lapsum import laplace_cdf, laplace_pdf
from wsparse.config import ActivationBottleneckConfig


def make_gate(mode=None, b0=0.0, T=1.0, k=3, j=4, n=16, supp=1.0):
    return AdaptiveLapSumTopKGate(
        n_features=n, k=k, j=j, selection_mode="abs_topk",
        surrogate_mode="rblapsum_sf", rblapsum_boundary_grad_mode=mode,
        rblapsum_boundary_floor=b0, temperature=T,
        rblapsum_support_scale=supp, log_diagnostics=True)


def reference_soft(a, k, m, T, b0):
    """The spec's forward, written so autograd routes b through the rank."""
    scores = a.abs()
    _, ci = torch.topk(scores.detach(), m, dim=-1, largest=True, sorted=True)
    z_c = torch.gather(a, -1, ci)
    s_c = z_c.abs()                      # grad-connected scores at the pool
    b = s_c[..., k:k + 1].clamp(min=b0)  # clamp kills grad when the floor wins
    p = laplace_cdf((s_c - b) / T)
    y = torch.zeros_like(a).scatter(-1, ci, z_c * p)
    return y


def grad_of(fn, a, upstream):
    a = a.clone().requires_grad_(True)
    fn(a).backward(upstream)
    return a.grad.clone()


# --------------------------------------------------------------------------- #
# forward
# --------------------------------------------------------------------------- #


def test_forward_is_z_times_p_on_pool_zero_outside():
    torch.manual_seed(0)
    a = torch.randn(5, 16, dtype=torch.float64)
    g = make_gate()
    g.train()
    y = g(a)
    scores = a.abs()
    cs, ci = torch.topk(scores, g.m, dim=-1, largest=True, sorted=True)
    b = cs[..., g.k:g.k + 1].clamp(min=0.0)
    p = laplace_cdf((cs - b) / g.temperature)
    want = torch.zeros_like(a).scatter(-1, ci, torch.gather(a, -1, ci) * p)
    assert torch.allclose(y, want, atol=1e-12)
    # everything outside the pool is exactly zero
    outside = torch.ones_like(a, dtype=torch.bool).scatter(-1, ci, False)
    assert (y[outside] == 0).all()
    # dense within the pool: K+J nonzeros per row (generic scores)
    assert (y != 0).sum(-1).float().mean() == g.m


def test_boundary_feature_outputs_half():
    torch.manual_seed(1)
    a = torch.randn(4, 16, dtype=torch.float64)
    g = make_gate()
    g.train()
    y = g(a)
    scores = a.abs()
    _, ci = torch.topk(scores, g.m, dim=-1, largest=True, sorted=True)
    z_k1 = torch.gather(a, -1, ci[..., g.k:g.k + 1])
    y_k1 = torch.gather(y, -1, ci[..., g.k:g.k + 1])
    assert torch.allclose(y_k1, 0.5 * z_k1, atol=1e-12)


def test_eval_hard_inference_matches_hard_rblapsum():
    torch.manual_seed(2)
    a = torch.randn(6, 16)
    sf = make_gate()
    hard = AdaptiveLapSumTopKGate(
        n_features=16, k=3, j=4, selection_mode="abs_topk",
        surrogate_mode="rblapsum", rblapsum_boundary_floor=0.0)
    sf.eval(), hard.eval()
    with torch.no_grad():
        assert torch.equal(sf(a), hard(a))
    # and the soft eval forward is the training forward
    sf.hard_inference = False
    sf.train()
    with torch.no_grad():
        y_train = sf(a)
    sf.eval()
    with torch.no_grad():
        assert torch.equal(sf(a), y_train)


# --------------------------------------------------------------------------- #
# backward
# --------------------------------------------------------------------------- #


def test_through_rank_scale1_is_exact_autograd():
    torch.manual_seed(3)
    a = torch.randn(8, 16, dtype=torch.float64)
    up = torch.randn(8, 16, dtype=torch.float64)
    g = make_gate(mode="through_rank")
    g.train()
    got = grad_of(g, a, up)
    want = grad_of(lambda x: reference_soft(x, g.k, g.m, 1.0, 0.0), a, up)
    assert torch.allclose(got, want, atol=1e-10)


def test_through_rank_exact_with_floor_binding():
    # b0 large enough to win on some rows: there the boundary is a constant
    # and clamp kills the reference gradient, matching cap_active gating
    torch.manual_seed(4)
    a = torch.randn(64, 16, dtype=torch.float64)
    up = torch.randn(64, 16, dtype=torch.float64)
    b0 = 1.0
    g = make_gate(mode="through_rank", b0=b0)
    g.train()
    got = grad_of(g, a, up)
    want = grad_of(lambda x: reference_soft(x, g.k, g.m, 1.0, b0), a, up)
    assert torch.allclose(got, want, atol=1e-10)
    # the floor must actually bind somewhere for this test to mean anything
    cs, _ = torch.topk(a.abs(), g.m, dim=-1, sorted=True)
    assert (cs[..., g.k] < b0).any()


def _score_grads(gate, a, up):
    """Score-space gradient over the pool: sign(z) * (dL/dz - u * p)."""
    scores = a.abs()
    cs, ci = torch.topk(scores, gate.m, dim=-1, largest=True, sorted=True)
    b = cs[..., gate.k:gate.k + 1].clamp(min=gate.rblapsum_boundary_floor)
    p = laplace_cdf((cs - b) / gate.temperature)
    dz = torch.gather(grad_of(gate, a, up), -1, ci)
    u_c = torch.gather(up, -1, ci)
    z_c = torch.gather(a, -1, ci)
    return z_c.sign() * (dz - u_c * p), cs, b, u_c, z_c


def test_kappa_zero_sum_and_formula():
    torch.manual_seed(5)
    a = torch.randn(8, 16, dtype=torch.float64)
    up = torch.randn(8, 16, dtype=torch.float64)
    g = make_gate(mode="through_rank_kappa")
    g.train()
    g_s, cs, b, u_c, z_c = _score_grads(g, a, up)
    # zero-sum over the pool wherever the cap binds (b0=0 -> everywhere here)
    assert g_s.sum(-1).abs().max() < 1e-10
    # and exactly a - q * sum(a)
    kap = laplace_pdf((cs - b) / 1.0) / 1.0
    a_term = u_c * z_c * kap
    q = kap / kap.sum(-1, keepdim=True)
    want = a_term - q * a_term.sum(-1, keepdim=True)
    assert torch.allclose(g_s, want, atol=1e-10)


def test_support_scale_scales_only_boundary_term():
    torch.manual_seed(6)
    a = torch.randn(8, 16, dtype=torch.float64)
    up = torch.randn(8, 16, dtype=torch.float64)
    full = _score_grads(make_gate(mode="through_rank_kappa", supp=1.0).train(), a, up)[0]
    half = _score_grads(make_gate(mode="through_rank_kappa", supp=0.5).train(), a, up)[0]
    off = _score_grads(make_gate(mode="through_rank_kappa", supp=0.0).train(), a, up)[0]
    detach = _score_grads(make_gate(mode="detach").train(), a, up)[0]
    # scale 0 = detach = the pure direct gradient
    assert torch.allclose(off, detach, atol=1e-12)
    # linear interpolation in the scale
    assert torch.allclose(half, 0.5 * (full + off), atol=1e-10)
    # detach really keeps the direct term (nonzero), not a hard backward
    assert detach.abs().sum() > 0


def test_value_path_reaches_inactive_candidates():
    # supp=0, detach: dL/dz on an inactive pool member is u*p + sign*u*z*kappa,
    # nonzero in general -- the value path exists for the whole pool
    torch.manual_seed(7)
    a = torch.randn(4, 16, dtype=torch.float64)
    up = torch.ones_like(a)
    g = make_gate(mode="detach", supp=0.0)
    g.train()
    dz = grad_of(g, a, up)
    scores = a.abs()
    _, ci = torch.topk(scores, g.m, dim=-1, largest=True, sorted=True)
    inactive = ci[..., g.k + 1:]
    assert torch.gather(dz, -1, inactive).abs().min() > 0


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def test_config_mode_dependent_default():
    sf = ActivationBottleneckConfig(enabled=True, surrogate_mode="rblapsum_sf",
                                    selection_mode="abs_topk")
    assert sf.rblapsum_boundary_grad_mode == "through_rank_kappa"
    hard = ActivationBottleneckConfig(enabled=True, surrogate_mode="rblapsum",
                                      selection_mode="abs_topk")
    assert hard.rblapsum_boundary_grad_mode == "detach"
    explicit = ActivationBottleneckConfig(
        enabled=True, surrogate_mode="rblapsum_sf", selection_mode="abs_topk",
        rblapsum_boundary_grad_mode="detach")
    assert explicit.rblapsum_boundary_grad_mode == "detach"


def test_config_rejects_gated_topk():
    with pytest.raises(ValueError, match="rblapsum_sf"):
        ActivationBottleneckConfig(enabled=True, surrogate_mode="rblapsum_sf",
                                   selection_mode="gated_topk")


# --------------------------------------------------------------------------- #
# value_grad="support": tail values do not train, ranking still does
# --------------------------------------------------------------------------- #


def make_gate_vg(vg, mode=None, b0=0.0, T=1.0, k=3, j=4, n=16, supp=1.0):
    return AdaptiveLapSumTopKGate(
        n_features=n, k=k, j=j, selection_mode="abs_topk",
        surrogate_mode="rblapsum_sf", rblapsum_boundary_grad_mode=mode,
        rblapsum_boundary_floor=b0, temperature=T,
        rblapsum_support_scale=supp, rblapsum_sf_value_grad=vg,
        log_diagnostics=True)


def test_support_masks_exactly_the_tail_value_path():
    torch.manual_seed(8)
    a = torch.randn(8, 16, dtype=torch.float64)
    up = torch.randn(8, 16, dtype=torch.float64)
    g_pool = make_gate_vg("pool"); g_pool.train()
    g_sup = make_gate_vg("support"); g_sup.train()
    # identical forward
    with torch.no_grad():
        assert torch.equal(g_pool(a), g_sup(a))
    d_pool = grad_of(g_pool, a, up)
    d_sup = grad_of(g_sup, a, up)
    # the difference is u * p on the inactive candidates, exactly
    scores = a.abs()
    cs, ci = torch.topk(scores, g_pool.m, dim=-1, largest=True, sorted=True)
    b = cs[..., g_pool.k:g_pool.k + 1].clamp(min=0.0)
    p = laplace_cdf((cs - b) / g_pool.temperature)
    active = torch.zeros_like(cs)
    active[..., :g_pool.k] = 1.0
    want = torch.zeros_like(a).scatter(
        -1, ci, torch.gather(up, -1, ci) * p * (1 - active))
    assert torch.allclose(d_pool - d_sup, want, atol=1e-12)
    # actives bit-identical between the two modes
    act_idx = ci[..., :g_pool.k]
    assert torch.equal(torch.gather(d_pool, -1, act_idx),
                       torch.gather(d_sup, -1, act_idx))
    # the tail still receives its score-path gradient (nonzero in general)
    tail_idx = ci[..., g_pool.k:]
    assert torch.gather(d_sup, -1, tail_idx).abs().sum() > 0


def test_support_tail_gradient_is_score_path_only():
    # with the boundary term off (supp=0, detach) the tail gradient under
    # "support" must equal sign(z) * u * z * kappa -- the direct score term
    torch.manual_seed(9)
    a = torch.randn(6, 16, dtype=torch.float64)
    up = torch.randn(6, 16, dtype=torch.float64)
    g = make_gate_vg("support", mode="detach", supp=0.0); g.train()
    dz = grad_of(g, a, up)
    scores = a.abs()
    cs, ci = torch.topk(scores, g.m, dim=-1, largest=True, sorted=True)
    b = cs[..., g.k:g.k + 1].clamp(min=0.0)
    kap = laplace_pdf((cs - b) / 1.0) / 1.0
    z_c = torch.gather(a, -1, ci)
    u_c = torch.gather(up, -1, ci)
    want_tail = (z_c.sign() * u_c * z_c * kap)[..., g.k:]
    got_tail = torch.gather(dz, -1, ci[..., g.k:])
    assert torch.allclose(got_tail, want_tail, atol=1e-12)


def test_value_grad_config_validation():
    ActivationBottleneckConfig(enabled=True, surrogate_mode="rblapsum_sf",
                               selection_mode="abs_topk",
                               rblapsum_sf_value_grad="support")
    with pytest.raises(ValueError, match="pool | support"):
        ActivationBottleneckConfig(enabled=True, surrogate_mode="rblapsum_sf",
                                   selection_mode="abs_topk",
                                   rblapsum_sf_value_grad="tail")
    with pytest.raises(ValueError, match="soft-forward knob"):
        ActivationBottleneckConfig(enabled=True, surrogate_mode="rblapsum",
                                   selection_mode="abs_topk",
                                   rblapsum_sf_value_grad="support")
