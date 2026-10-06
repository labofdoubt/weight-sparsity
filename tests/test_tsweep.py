"""rblapsum_kernel_width (relative_span / relative_b) and rblapsum_support_strength."""
import torch

from wsparse.bottleneck.gate import AdaptiveLapSumTopKGate
from wsparse.bottleneck.rblapsum import kernel_width, strength_scale
from wsparse.config import ActivationBottleneckConfig


def gate(mode="fixed", s=None, t=1.0, grad_mode="detach", k=8, j=24):
    g = AdaptiveLapSumTopKGate(n_features=64, k=k, j=j, surrogate_mode="rblapsum",
                               rblapsum_boundary_grad_mode=grad_mode, temperature=t,
                               rblapsum_kernel_width=mode, rblapsum_support_strength=s)
    g.train()
    return g


def test_kernel_width_rules():
    torch.manual_seed(0)
    k, j = 8, 24
    sc = -torch.sort(-torch.rand(5, 40, dtype=torch.float64), dim=-1).values   # sorted pool
    b = sc[..., k:k + 1]
    assert kernel_width("fixed", 1.5, sc, b, k, j) == 1.5
    assert torch.allclose(kernel_width("relative_b", 0.5, sc, b, k, j), 0.5 * b)
    span = sc[..., k:k + 1] - sc[..., k + j - 1:k + j]
    assert torch.allclose(kernel_width("relative_span", 0.25, sc, b, k, j), 0.25 * span)
    assert (span > 0).all()


def test_strength_pins_the_boundary_member_to_s_times_upstream():
    # mode "detach": g_z_i = gamma * g_i * u_i * kappa_i * sign(u_i) on the pool, so at the
    # (K+1)-st member (|u| = b, kappa = 1/2T) the support gradient is exactly s * g_i
    torch.manual_seed(1)
    k, j = 8, 24
    for mode, tau in (("relative_span", 0.5), ("relative_b", 0.25), ("fixed", 1.3)):
        for s in (0.1, 1.0):
            g = gate(mode, s, t=tau)
            z = (3 * torch.randn(6, 64, dtype=torch.float64)).requires_grad_(True)
            w = torch.randn(6, 64, dtype=torch.float64)
            y = g(z)
            (y * w).sum().backward()
            order = z.detach().abs().argsort(-1, descending=True)
            bnd = order[:, k]                                   # the (K+1)-st feature per row
            gz = z.grad.gather(1, bnd[:, None]).squeeze(1)
            gw = w.gather(1, bnd[:, None]).squeeze(1)
            assert torch.allclose(gz, s * gw, rtol=1e-9, atol=1e-12), (mode, s)
            # outside the pool nothing; forward is the hard gate
            rest = order[:, k + j:]
            assert (z.grad.gather(1, rest) == 0).all()
            assert torch.equal((y != 0).sum(-1), torch.full((6,), k))


def test_strength_scale_formula_and_legacy_flag():
    b = torch.tensor([[2.0], [0.5]], dtype=torch.float64)
    t = torch.tensor([[1.0], [0.25]], dtype=torch.float64)
    assert torch.allclose(strength_scale(0.5, t, b), torch.tensor([[0.5], [0.5]], dtype=torch.float64))
    g = AdaptiveLapSumTopKGate(n_features=64, k=8, j=8, surrogate_mode="rblapsum",
                               rblapsum_relative_temperature=True, temperature=0.5)
    assert g.rblapsum_kernel_width == "relative_b"


def test_config_validation():
    base = dict(enabled=True, n_features=64, k=8, j=8, placement="residual_out",
                surrogate_mode="rblapsum", rblapsum_boundary_grad_mode="through_rank_kappa")
    c = ActivationBottleneckConfig(**base, rblapsum_kernel_width="relative_span",
                                   rblapsum_support_strength=0.25)
    assert c.rblapsum_support_strength == 0.25
    c = ActivationBottleneckConfig(**base, rblapsum_relative_temperature=True)
    assert c.rblapsum_kernel_width == "relative_b"
    import pytest
    with pytest.raises(ValueError):
        ActivationBottleneckConfig(**base, rblapsum_kernel_width="relative_x")
    with pytest.raises(ValueError):
        ActivationBottleneckConfig(**{**base, "surrogate_mode": "hard"}, rblapsum_support_strength=0.5)
    # first_order shares the pool's member set and the per-token T / gamma code path
    c = ActivationBottleneckConfig(**base, rblapsum_support_strength=0.5,
                                   code_residual=True, rblapsum_surrogate_scope="first_order")
    assert c.rblapsum_support_strength == 0.5
    with pytest.raises(ValueError):
        ActivationBottleneckConfig(**base, rblapsum_support_strength=0.5,
                                   code_residual=True, share_projections=True,
                                   rblapsum_surrogate_scope="carry_mixed")


def test_first_order_scope_uses_the_same_per_token_width_and_strength():
    # outside the hard pass the first_order gate's backward is the pool gate's backward
    torch.manual_seed(2)
    k, j = 8, 24
    for mode in ("relative_span", "relative_b"):
        grads = []
        for scope in ("pool", "first_order"):
            g = AdaptiveLapSumTopKGate(n_features=64, k=k, j=j, surrogate_mode="rblapsum",
                                       rblapsum_boundary_grad_mode="through_rank_kappa",
                                       temperature=0.5, rblapsum_kernel_width=mode,
                                       rblapsum_support_strength=0.5,
                                       rblapsum_surrogate_scope=scope)
            g.train()
            torch.manual_seed(3)
            z = (3 * torch.randn(6, 64, dtype=torch.float64)).requires_grad_(True)
            w = torch.randn(6, 64, dtype=torch.float64)
            (g(z) * w).sum().backward()
            grads.append(z.grad.clone())
        assert torch.allclose(grads[0], grads[1], rtol=1e-12, atol=1e-14), mode
        assert (grads[0] != 0).sum() > 6 * k           # the support term is present
