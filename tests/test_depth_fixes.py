"""The selection gain of a stream bottleneck, and the two options that remove it.

``value_shift`` shrinks the kept TopK magnitudes so the forward stops carrying
the selection bias; ``code_residual`` carries the K-sparse code between blocks.
See ActivationBottleneckConfig and docs/stream-bottleneck-depth.tex.
"""

import math

import pytest
import torch

from wsparse.bottleneck import AdaptiveLapSumTopKGate, apply_activation_bottleneck
from wsparse.bottleneck.module import (
    critical_shift,
    selection_energy_gain,
    selection_gain,
)
from wsparse.config import ActivationBottleneckConfig, ModelConfig
from wsparse.model import build_model


def gaussian_topk_stats(k, n, rows=4000, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(rows, n, generator=g, dtype=torch.float64)
    kept = z.abs().topk(k, dim=-1).values
    return z, kept


@pytest.mark.parametrize("k,n", [(512, 4096), (32, 4096), (256, 512), (8, 64)])
def test_selection_energy_gain_matches_simulation(k, n):
    _, kept = gaussian_topk_stats(k, n, rows=2000)
    sim = float((kept ** 2).mean())
    assert selection_energy_gain(k, n) == pytest.approx(sim, rel=0.02)


@pytest.mark.parametrize("k,n", [(512, 4096), (32, 4096), (256, 512)])
def test_critical_shift_removes_the_selection_gain(k, n):
    lam = critical_shift(k, n)
    _, kept = gaussian_topk_stats(k, n, rows=2000)
    assert float(((kept - lam) ** 2).mean()) == pytest.approx(1.0, rel=0.03)
    # the kept magnitudes all clear the shift, so nothing is clamped
    assert float(kept.min()) > lam


def test_critical_shift_reference_values():
    assert selection_energy_gain(512, 4096) == pytest.approx(4.019, abs=2e-3)
    assert critical_shift(512, 4096) == pytest.approx(1.044, abs=2e-3)
    assert critical_shift(4096, 4096) == 0.0
    # the selection gain m1 used inside is the existing closed form
    assert selection_gain(512, 4096) == pytest.approx(1.968, abs=2e-3)


def shift_gate(mode, lam=None, n=256, k=32):
    lam = critical_shift(k, n) if lam is None else lam
    gate = AdaptiveLapSumTopKGate(n_features=n, k=k, j=0, surrogate_mode="hard",
                                  value_shift=mode, value_shift_lambda=lam)
    gate.train()
    return gate


def test_energy_shift_matches_the_unselected_energy_per_token():
    gate = shift_gate("energy")
    torch.manual_seed(0)
    z = torch.randn(64, 256)
    y = gate(z)
    mean_kept = (y ** 2).sum(-1) / gate.k
    assert torch.allclose(mean_kept, (z ** 2).mean(-1), rtol=1e-4)


def test_energy_shift_on_heavy_tails_is_exact_where_a_shift_can_be_exact():
    """A pure shift reaches mean_all z^2 only if the kept magnitudes are not too
    spread (m1^2 - m2 + sigma^2 >= 0) and none falls below delta; elsewhere it
    falls back to the shift of least energy, which is still below plain TopK."""
    gate = shift_gate("energy")
    torch.manual_seed(0)
    # tails from Gaussian (p = 1) to very heavy (p = 3), one exponent per token
    g = torch.randn(256, 256)
    p = torch.linspace(1.0, 3.0, 256).unsqueeze(1)
    z = g.sign() * g.abs() ** p
    y = gate(z)
    kept = z.abs().topk(gate.k, dim=-1).values
    m1, m2 = kept.mean(-1), (kept ** 2).mean(-1)
    sig2 = (z ** 2).mean(-1)
    disc = m1 ** 2 - m2 + sig2
    delta = m1 - disc.clamp_min(0).sqrt()
    exact = (disc > 1e-3 * sig2) & (kept.min(-1).values > delta)
    assert exact.any() and (~exact).any()
    mean_kept = (y ** 2).sum(-1) / gate.k
    assert torch.allclose(mean_kept[exact], sig2[exact], rtol=1e-4)
    assert (mean_kept < m2).all()


def test_fixed_shift_is_the_energy_shift_on_gaussian_codes():
    torch.manual_seed(1)
    z = torch.randn(512, 4096)
    fixed = shift_gate("fixed", n=4096, k=512)(z)
    energy = shift_gate("energy", n=4096, k=512)(z)
    ratio = float((fixed ** 2).sum() / (energy ** 2).sum())
    assert ratio == pytest.approx(1.0, rel=0.02)


def test_shift_keeps_support_and_signs():
    gate = shift_gate("energy")
    torch.manual_seed(2)
    z = torch.randn(16, 256)
    y = gate(z)
    plain = AdaptiveLapSumTopKGate(n_features=256, k=32, j=0, surrogate_mode="hard")
    ref = plain(z)
    on = ref != 0
    # at most K survive, all inside the plain support, none change sign
    assert ((y != 0) & ~on).sum() == 0
    assert (torch.sign(y[y != 0]) == torch.sign(ref[y != 0])).all()
    # every kept magnitude is shrunk by the same per-token delta
    delta = (ref.abs() - y.abs())[on & (y != 0)]
    assert float(delta.min()) > 0


def test_shift_none_is_the_plain_forward():
    torch.manual_seed(3)
    z = torch.randn(8, 256, requires_grad=True)
    a = AdaptiveLapSumTopKGate(n_features=256, k=32, j=0, surrogate_mode="hard")
    b = AdaptiveLapSumTopKGate(n_features=256, k=32, j=0, surrogate_mode="hard",
                               value_shift="none")
    assert torch.equal(a(z), b(z))


def test_shift_gradient_is_finite_and_matches_finite_differences():
    gate = shift_gate("energy", n=64, k=8)
    torch.manual_seed(4)
    z = torch.randn(3, 64, dtype=torch.float64, requires_grad=True)
    w = torch.randn(3, 64, dtype=torch.float64)
    f = lambda t: (gate(t) * w).sum()  # noqa: E731
    f(z).backward()
    g = z.grad.clone()
    assert torch.isfinite(g).all()
    # the support is piecewise constant: finite differences small enough not
    # to cross the boundary check the smooth part, delta's dependence included
    eps = 1e-6
    for idx in [(0, 3), (1, 10), (2, 40)]:
        e = torch.zeros_like(z)
        e[idx] = eps
        num = (f(z.detach() + e) - f(z.detach() - e)) / (2 * eps)
        assert float(num) == pytest.approx(float(g[idx]), rel=1e-5, abs=1e-8)


def test_shift_config_validation():
    base = dict(enabled=True, n_features=64, k=8, j=8, surrogate_mode="hard")
    ActivationBottleneckConfig(**base, value_shift="energy")
    with pytest.raises(ValueError):
        ActivationBottleneckConfig(**base, value_shift="bogus")
    with pytest.raises(ValueError):
        ActivationBottleneckConfig(**{**base, "surrogate_mode": "lapsum"},
                                   value_shift="energy")
    with pytest.raises(ValueError):
        ActivationBottleneckConfig(**base, selection_mode="topk", value_shift="fixed")
    with pytest.raises(ValueError):  # lambda only means something for 'fixed'
        ActivationBottleneckConfig(**base, value_shift="energy", value_shift_lambda=1.0)


def test_shifted_bottleneck_uses_the_critical_lambda_by_default():
    torch.manual_seed(0)
    cfg = ActivationBottleneckConfig(enabled=True, n_features=128, k=16, j=0,
                                     surrogate_mode="hard", value_shift="fixed",
                                     placement="residual_out")
    model = build_model(ModelConfig(vocab_size=97, max_seq_len=32, n_layers=2,
                                    d_model=32, n_heads=4))
    ctl = apply_activation_bottleneck(model, cfg)
    for _, mod in ctl.layers:
        assert mod.gate.value_shift_lambda == pytest.approx(critical_shift(16, 128))


# ---- code residual ---------------------------------------------------------- #

def code_model(n_layers=3, scale=1.0, **kw):
    torch.manual_seed(0)
    cfg = ActivationBottleneckConfig(
        enabled=True, n_features=128, k=16, j=0, surrogate_mode="hard",
        placement="residual_out", share_projections=True, code_residual=True,
        code_residual_scale=scale, init_mode="unit_norm_dictionary", **kw)
    model = build_model(ModelConfig(vocab_size=97, max_seq_len=32,
                                    n_layers=n_layers, d_model=32, n_heads=4))
    ctl = apply_activation_bottleneck(model, cfg)
    return model, ctl


def test_code_residual_config_validation():
    base = dict(enabled=True, n_features=64, k=8, j=8, surrogate_mode="hard",
                placement="residual_out", share_projections=True)
    ActivationBottleneckConfig(**base, code_residual=True)
    for bad in ({"placement": "residual"}, {"share_projections": False},
                {"post_norm": True}, {"code_residual_scale": 0.0}):
        with pytest.raises(ValueError):
            ActivationBottleneckConfig(**{**base, **bad}, code_residual=True)
    with pytest.raises(ValueError):
        ActivationBottleneckConfig(**base, code_residual=True, value_shift="energy")


def test_code_residual_rejects_a_partial_stack():
    cfg = ActivationBottleneckConfig(
        enabled=True, n_features=64, k=8, j=0, surrogate_mode="hard",
        placement="residual_out", share_projections=True, code_residual=True,
        layers="first:1")
    model = build_model(ModelConfig(vocab_size=97, max_seq_len=32, n_layers=2,
                                    d_model=32, n_heads=4))
    with pytest.raises(ValueError):
        apply_activation_bottleneck(model, cfg)


def test_code_residual_adds_an_entry_gate_and_no_parameters():
    model, ctl = code_model()
    assert ctl.layers[0][0] == "entry"
    assert model.code_entry.in_proj is model.blocks[0].residual_out_bottleneck.in_proj
    assert len(ctl.layers) == 1 + len(model.blocks)
    # the same stack with the stream carried: one shared pair either way
    torch.manual_seed(0)
    shared = build_model(ModelConfig(vocab_size=97, max_seq_len=32, n_layers=3,
                                     d_model=32, n_heads=4))
    apply_activation_bottleneck(shared, ActivationBottleneckConfig(
        enabled=True, n_features=128, k=16, j=0, surrogate_mode="hard",
        placement="residual_out", share_projections=True,
        init_mode="unit_norm_dictionary"))
    assert model.num_parameters() == shared.num_parameters()
    assert "code_entry.in_proj.weight" in model.state_dict()


def test_code_residual_with_silent_blocks_is_the_identity_on_the_code():
    model, _ = code_model(n_layers=3)
    with torch.no_grad():
        for blk in model.blocks:  # every block's contribution is zero
            blk.attn.proj.weight.zero_()
            blk.mlp.fc2.weight.zero_()
    model.eval()
    idx = torch.randint(0, 97, (2, 16))
    entry = model.code_entry
    with torch.no_grad():
        x = model.tok_emb(idx) * model.embed_scale
        x = x + model.pos_emb(torch.arange(idx.shape[1]))[None]
        code0 = entry.gate(entry.in_proj(x))
        expect = model.lm_head(model.norm_f(entry.decode(code0))) * model.logit_mult
        got, _ = model(idx)
    assert torch.allclose(got, expect, atol=1e-5)


def test_code_residual_forward_is_k_sparse_between_blocks():
    model, _ = code_model(n_layers=2)
    codes = []
    for blk in model.blocks:
        blk.residual_out_bottleneck.gate.register_forward_hook(
            lambda m, i, o: codes.append(o.detach()))
    model(torch.randint(0, 97, (2, 8)))
    assert len(codes) == 2
    for c in codes:
        assert int((c != 0).sum(-1).max()) <= 16


def test_code_residual_gradient_reaches_the_first_block():
    model, _ = code_model(n_layers=4, scale=0.3)
    model.train()
    idx = torch.randint(0, 97, (2, 16))
    _, loss = model(idx, idx)
    loss.backward()
    first = model.blocks[0].mlp.fc1.weight.grad
    last = model.blocks[-1].mlp.fc1.weight.grad
    assert first is not None and float(first.norm()) > 0
    assert float(first.norm() / last.norm()) > 1e-2


def test_code_residual_checkpoint_roundtrip():
    model, _ = code_model(n_layers=2)
    sd = model.state_dict()
    fresh, _ = code_model(n_layers=2)
    fresh.load_state_dict(sd)
    idx = torch.randint(0, 97, (2, 8))
    model.eval(); fresh.eval()
    assert torch.equal(model(idx)[0], fresh(idx)[0])
