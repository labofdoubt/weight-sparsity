"""stochastic_width (per-token K' in [K, K+J]) and surrogate_mode="soft_ste"."""
import torch

from wsparse.bottleneck.gate import AdaptiveLapSumTopKGate
from wsparse.bottleneck.lapsum import laplace_cdf
from wsparse.config import ActivationBottleneckConfig
from wsparse.model import ModelConfig, build_model
from wsparse.bottleneck import apply_activation_bottleneck


def test_stochastic_width_keeps_between_k_and_k_plus_j_in_training_and_k_in_eval():
    torch.manual_seed(0)
    for dist in ("uniform", "two_point", "geometric"):
        g = AdaptiveLapSumTopKGate(n_features=64, k=8, j=24, surrogate_mode="hard",
                                   stochastic_width=dist, stochastic_width_param=0.5)
        g.train()
        z = torch.randn(300, 64, requires_grad=True)
        y = g(z)
        n_active = (y != 0).sum(-1)
        assert int(n_active.min()) >= 8 and int(n_active.max()) <= 32
        assert int(n_active.max()) > 8  # some tokens widened
        # the kept entries are the largest by magnitude: a prefix of the sorted pool
        order = z.abs().argsort(-1, descending=True)
        for t in range(10):
            kept = set((y[t] != 0).nonzero().flatten().tolist())
            assert kept == set(order[t, :len(kept)].tolist())
        # hard gradient: active entries pass, inactive get exactly zero
        (y * torch.randn_like(y)).sum().backward()
        assert torch.equal((z.grad != 0), (y != 0))
        g.eval()
        with torch.no_grad():
            assert torch.equal((g(z) != 0).sum(-1), torch.full((300,), 8))


def test_soft_ste_hard_forward_and_soft_backward():
    torch.manual_seed(1)
    g = AdaptiveLapSumTopKGate(n_features=64, k=8, j=24, surrogate_mode="soft_ste",
                               temperature=1.0)
    g.train()
    z = (2 * torch.randn(5, 64, dtype=torch.float64)).requires_grad_(True)
    w = torch.randn(5, 64, dtype=torch.float64)
    y = g(z)
    hard = AdaptiveLapSumTopKGate(n_features=64, k=8, j=24, surrogate_mode="hard")
    assert torch.equal(y.detach(), hard(z.detach()))           # the forward is hard
    (y * w).sum().backward()
    s = z.detach().abs()
    top = s.topk(32, dim=-1)
    b = top.values[:, 8:9]
    p = laplace_cdf((s - b) / 1.0)
    pool = torch.zeros_like(s).scatter(-1, top.indices, 1.0)
    active = torch.zeros_like(s).scatter(-1, top.indices[:, :8], 1.0)
    expected = w * (active + (1 - active) * pool * p)
    assert torch.allclose(z.grad, expected, atol=1e-12)
    assert (z.grad[pool == 0] == 0).all()
    g.eval()
    with torch.no_grad():
        assert torch.equal(g(z), hard(z))


def test_new_modes_build_in_a_code_residual_model(tmp_path):
    for kw in (dict(surrogate_mode="hard", stochastic_width="uniform"),
               dict(surrogate_mode="soft_ste", temperature=2.0)):
        cfg = ActivationBottleneckConfig(enabled=True, n_features=96, k=8, j=24,
                                         placement="residual_out", share_projections=True,
                                         code_residual=True, **kw)
        m = build_model(ModelConfig(vocab_size=61, max_seq_len=16, n_layers=3, d_model=24,
                                    n_heads=4))
        apply_activation_bottleneck(m, cfg)
        m.train()
        idx = torch.randint(0, 61, (2, 12))
        _, loss = m(idx, idx)
        loss.backward()
        assert torch.isfinite(loss)
        m.eval()
        with torch.no_grad():
            m(idx, idx)
