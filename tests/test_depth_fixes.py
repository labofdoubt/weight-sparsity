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

def code_model(n_layers=3, scale=1.0, share=True, **kw):
    torch.manual_seed(0)
    cfg = ActivationBottleneckConfig(
        enabled=True, n_features=128, k=16, j=0, surrogate_mode="hard",
        placement="residual_out", share_projections=share, code_residual=True,
        code_residual_scale=scale, init_mode="unit_norm_dictionary", **kw)
    model = build_model(ModelConfig(vocab_size=97, max_seq_len=32,
                                    n_layers=n_layers, d_model=32, n_heads=4))
    ctl = apply_activation_bottleneck(model, cfg)
    return model, ctl


def test_code_residual_config_validation():
    base = dict(enabled=True, n_features=64, k=8, j=8, surrogate_mode="hard",
                placement="residual_out", share_projections=True)
    ActivationBottleneckConfig(**base, code_residual=True)
    # per-block projections are allowed (each block reads and writes its own)
    ActivationBottleneckConfig(**{**base, "share_projections": False}, code_residual=True)
    for bad in ({"placement": "residual"}, {"post_norm": True},
                {"code_residual_scale": 0.0}):
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


@pytest.mark.parametrize("share", [True, False])
def test_code_residual_with_silent_blocks_is_the_identity_on_the_code(share):
    model, _ = code_model(n_layers=3, share=share)
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


@pytest.mark.parametrize("share", [True, False])
def test_code_residual_gradient_reaches_the_first_block(share):
    model, _ = code_model(n_layers=4, scale=0.3, share=share)
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


def reference_shift(z, k, mode, lam):
    """The dense (N-sized) formulation the gathered implementation must match."""
    v = z.double()
    mask = torch.zeros_like(v).scatter(-1, v.abs().topk(k, dim=-1).indices, 1.0)
    sig2 = (v * v).mean(-1, keepdim=True)
    if mode == "fixed":
        delta = lam * sig2.sqrt()
    else:
        a = v.abs() * mask
        m1, m2 = a.sum(-1, keepdim=True) / k, (a * a).sum(-1, keepdim=True) / k
        delta = m1 - (m1 * m1 - m2 + sig2).clamp_min(1e-6 * sig2).sqrt()
    return v.sign() * torch.relu(v.abs() - delta) * mask


@pytest.mark.parametrize("mode", ["fixed", "energy"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gathered_shift_matches_the_dense_formulation(mode, dtype):
    gate = shift_gate(mode, n=512, k=64)
    torch.manual_seed(5)
    g = torch.randn(32, 512)
    z = (g.sign() * g.abs() ** torch.linspace(1.0, 2.0, 32).unsqueeze(1)).to(dtype)
    got = gate(z).double()
    want = reference_shift(z, 64, mode, gate.value_shift_lambda)
    tol = 1e-5 if dtype == torch.float32 else 1e-2
    assert torch.allclose(got, want, rtol=tol, atol=tol * float(want.abs().max()))


def test_unshared_code_residual_gives_every_block_its_own_pair():
    """Without share_projections block l reads through its own decoder and
    writes through its own encoder; the entry adds one more pair (E_in for c_0,
    D_out for the final readout)."""
    model, ctl = code_model(n_layers=3, share=False)
    mods = [m for _, m in ctl.layers]
    assert len({id(m.in_proj.weight) for m in mods}) == len(mods) == 4
    assert len({id(m.out_proj.weight) for m in mods}) == 4
    plain = build_model(ModelConfig(vocab_size=97, max_seq_len=32, n_layers=3,
                                    d_model=32, n_heads=4))
    pair = 2 * 32 * 128  # biasless in_proj + out_proj
    assert model.num_parameters() - plain.num_parameters() == 4 * pair
    # each block really reads through its own decoder: perturbing block 1's
    # decoder changes the output, and so does perturbing the entry's (readout)
    idx = torch.randint(0, 97, (2, 8))
    model.eval()
    with torch.no_grad():
        base = model(idx)[0].clone()
        for mod in (model.blocks[1].residual_out_bottleneck, model.code_entry):
            w = mod.out_proj.weight
            saved = w.clone()
            w.add_(0.1 * torch.randn_like(w))
            assert not torch.allclose(model(idx)[0], base)
            w.copy_(saved)
        assert torch.equal(model(idx)[0], base)


# ---- rblapsum_surrogate_scope ------------------------------------------------ #

def rbk_gate(scope="pool", mode="through_rank_kappa", n=64, k=8, j=24, t=1.0):
    gate = AdaptiveLapSumTopKGate(n_features=n, k=k, j=j, surrogate_mode="rblapsum",
                                  rblapsum_boundary_grad_mode=mode, temperature=t,
                                  rblapsum_surrogate_scope=scope)
    gate.train()
    return gate


@pytest.mark.parametrize("mode", ["detach", "project", "through_rank", "through_rank_kappa"])
def test_inactive_scope_leaves_active_features_on_the_exact_gradient(mode):
    torch.manual_seed(0)
    z = (3 * torch.randn(5, 64, dtype=torch.float64)).requires_grad_(True)
    w = torch.randn(5, 64, dtype=torch.float64)
    gate = rbk_gate("inactive", mode)
    y = gate(z)
    (y * w).sum().backward()
    active = (y.detach() != 0)
    # active features: exactly dL/dy (the hard mask's gradient), nothing added
    assert torch.equal(z.grad[active], w[active])
    # some inactive pool members do receive the support term
    pool = z.detach().abs().topk(8 + 24, dim=-1).indices
    inactive_pool = torch.zeros_like(z, dtype=torch.bool).scatter(-1, pool, True) & ~active
    assert float(z.grad[inactive_pool].abs().sum()) > 0
    if mode == "through_rank_kappa":  # zero-sum over the inactive members (s-space)
        gs = (z.grad * z.detach().sign())[inactive_pool].reshape(5, 24)
        assert torch.allclose(gs.sum(-1), torch.zeros(5, dtype=torch.float64), atol=1e-12)


def test_pool_scope_is_the_unchanged_backward():
    torch.manual_seed(1)
    z = (3 * torch.randn(4, 64)).requires_grad_(True)
    w = torch.randn(4, 64)
    a = AdaptiveLapSumTopKGate(n_features=64, k=8, j=24, surrogate_mode="rblapsum",
                               rblapsum_boundary_grad_mode="through_rank_kappa")
    b = rbk_gate("pool")
    a.train()
    (a(z) * w).sum().backward()
    g0 = z.grad.clone(); z.grad = None
    (b(z) * w).sum().backward()
    assert torch.equal(g0, z.grad)


def test_update_scope_routes_the_support_term_to_the_update_only():
    from wsparse.model import TransformerLM
    torch.manual_seed(2)
    gate = rbk_gate("update")
    carry = (3 * torch.randn(4, 64)).requires_grad_(True)
    update = torch.randn(4, 64).requires_grad_(True)
    w = torch.randn(4, 64)
    y = TransformerLM._split_surrogate(gate, carry, update)
    # the forward is the gate's own
    ref_in = (carry.detach() + update.detach()).requires_grad_(True)
    y_ref = gate(ref_in)
    assert torch.equal(y.detach(), y_ref.detach())
    (y * w).sum().backward()
    (y_ref * w).sum().backward()
    mask = (y_ref.detach() != 0).to(w.dtype)
    # the carry: exact hard gradient; the update: the gate's full gradient
    assert torch.equal(carry.grad, w * mask)
    assert torch.allclose(update.grad, ref_in.grad)
    assert not torch.allclose(update.grad, w * mask)  # the support term is there


def test_surrogate_scope_config_validation():
    base = dict(enabled=True, n_features=64, k=8, j=8, placement="residual_out",
                share_projections=True)
    ActivationBottleneckConfig(**base, surrogate_mode="rblapsum",
                               rblapsum_surrogate_scope="inactive")
    ActivationBottleneckConfig(**base, surrogate_mode="rblapsum", code_residual=True,
                               rblapsum_surrogate_scope="update")
    with pytest.raises(ValueError):  # unknown value
        ActivationBottleneckConfig(**base, surrogate_mode="rblapsum",
                                   rblapsum_surrogate_scope="bogus")
    with pytest.raises(ValueError):  # hard-forward rblapsum only
        ActivationBottleneckConfig(**base, surrogate_mode="lapsum",
                                   rblapsum_surrogate_scope="inactive")
    with pytest.raises(ValueError):  # the update routing needs a code residual
        ActivationBottleneckConfig(**base, surrogate_mode="rblapsum",
                                   rblapsum_surrogate_scope="update")


def test_update_scope_keeps_the_code_residual_forward():
    torch.manual_seed(0)
    def model(scope):
        torch.manual_seed(0)
        cfg = ActivationBottleneckConfig(
            enabled=True, n_features=128, k=16, j=16, surrogate_mode="rblapsum",
            placement="residual_out", share_projections=True, code_residual=True,
            init_mode="unit_norm_dictionary", rblapsum_surrogate_scope=scope)
        m = build_model(ModelConfig(vocab_size=97, max_seq_len=32, n_layers=3,
                                    d_model=32, n_heads=4))
        apply_activation_bottleneck(m, cfg)
        return m
    idx = torch.randint(0, 97, (2, 8))
    a, b = model("pool"), model("update")
    a.train(); b.train()
    la, lb = a(idx, idx)[1], b(idx, idx)[1]
    assert torch.equal(la.detach(), lb.detach())
    la.backward(); lb.backward()
    # the gradients differ (the carry no longer takes the support term)
    ga = a.blocks[0].mlp.fc1.weight.grad
    gb = b.blocks[0].mlp.fc1.weight.grad
    assert not torch.allclose(ga, gb)


# ---- rblapsum_relative_temperature ------------------------------------------- #

def rel_gate(t, relative, scope="pool", mode="through_rank_kappa"):
    gate = AdaptiveLapSumTopKGate(n_features=64, k=8, j=24, surrogate_mode="rblapsum",
                                  rblapsum_boundary_grad_mode=mode, temperature=t,
                                  rblapsum_surrogate_scope=scope,
                                  rblapsum_relative_temperature=relative)
    gate.train()
    return gate


def gate_grad(gate, z, w):
    z = z.clone().requires_grad_(True)
    (gate(z) * w).sum().backward()
    return z.grad


@pytest.mark.parametrize("scope", ["pool", "inactive"])
@pytest.mark.parametrize("mode", ["detach", "project", "through_rank_kappa"])
def test_relative_temperature_makes_the_backward_scale_equivariant(scope, mode):
    # hard TopK is equivariant to its input scale; with the relative kernel the
    # surrogate is too: for L(z) = <w, gate(c z)> / c every gradient is
    # independent of c.  With an absolute T it is not (the surrogate's strength
    # |u| kappa = b / 2T grows with the scale).
    torch.manual_seed(3)
    z = 3 * torch.randn(6, 64, dtype=torch.float64)
    w = torch.randn(6, 64, dtype=torch.float64)
    for relative in (True, False):
        gate = rel_gate(0.3, relative, scope, mode)
        g1 = gate_grad(gate, z, w)
        g4 = gate_grad(gate, 4 * z, w / 4) * 4          # d/dz of <w, gate(4z)>/4
        assert torch.allclose(g1, g4, atol=1e-12) == relative


def test_relative_temperature_is_the_absolute_kernel_at_width_t_times_b():
    torch.manual_seed(4)
    z = 3 * torch.randn(1, 64, dtype=torch.float64)
    w = torch.randn(1, 64, dtype=torch.float64)
    b = float(z.abs().topk(9, dim=-1).values[0, 8])      # the (K+1)-st score
    g_rel = gate_grad(rel_gate(0.25, True), z, w)
    g_abs = gate_grad(rel_gate(0.25 * b, False), z, w)
    assert torch.allclose(g_rel, g_abs, atol=1e-12)
    # the forward is the hard gate's either way
    assert torch.equal(rel_gate(0.25, True)(z), rel_gate(1.0, False)(z))


def test_relative_temperature_config_validation():
    base = dict(enabled=True, n_features=64, k=8, j=8, placement="residual_out",
                share_projections=True)
    ActivationBottleneckConfig(**base, surrogate_mode="rblapsum",
                               rblapsum_relative_temperature=True)
    for mode in ("hard", "lapsum", "rblapsum_sf"):
        with pytest.raises(ValueError):
            ActivationBottleneckConfig(**base, surrogate_mode=mode,
                                       rblapsum_relative_temperature=True)


@pytest.mark.parametrize("scope,members", [("update_inactive", "inactive"),
                                           ("update_active", "active")])
def test_update_scopes_restrict_the_routed_support_term(scope, members):
    from wsparse.model import TransformerLM
    torch.manual_seed(5)
    carry = (3 * torch.randn(4, 64, dtype=torch.float64)).requires_grad_(True)
    update = torch.randn(4, 64, dtype=torch.float64).requires_grad_(True)
    w = torch.randn(4, 64, dtype=torch.float64)
    y = TransformerLM._split_surrogate(rbk_gate(scope), carry, update)
    (y * w).sum().backward()
    active = (y.detach() != 0)
    mask = active.to(w.dtype)
    assert torch.equal(carry.grad, w * mask)                 # the carry: exact
    diff = update.grad - w * mask                            # the routed support term
    pool = (carry + update).detach().abs().topk(8 + 24, dim=-1).indices
    in_pool = torch.zeros_like(active).scatter(-1, pool, True)
    side = active if members == "active" else (in_pool & ~active)
    assert float(diff[side].abs().sum()) > 0
    assert torch.equal(diff[~side], torch.zeros_like(diff[~side]))
    # zero-sum over the chosen side, in score space (through_rank_kappa)
    gs = (diff * (carry + update).detach().sign())[side].reshape(4, -1)
    assert torch.allclose(gs.sum(-1), torch.zeros(4, dtype=torch.float64), atol=1e-12)


def test_update_scope_is_the_sum_of_its_two_halves_under_detach():
    # without a boundary correction the support term is diagonal, so the two
    # restricted scopes partition it exactly
    from wsparse.model import TransformerLM
    torch.manual_seed(6)
    base_c = 3 * torch.randn(3, 64, dtype=torch.float64)
    base_u = torch.randn(3, 64, dtype=torch.float64)
    w = torch.randn(3, 64, dtype=torch.float64)
    grads = {}
    for scope in ("update", "update_inactive", "update_active"):
        carry, update = base_c.clone().requires_grad_(True), base_u.clone().requires_grad_(True)
        y = TransformerLM._split_surrogate(rbk_gate(scope, mode="detach"), carry, update)
        (y * w).sum().backward()
        grads[scope] = update.grad - w * (y.detach() != 0).to(w.dtype)
    assert torch.allclose(grads["update"], grads["update_inactive"] + grads["update_active"],
                          atol=1e-12)


def test_restricted_update_scopes_config_validation():
    base = dict(enabled=True, n_features=64, k=8, j=8, placement="residual_out",
                share_projections=True, surrogate_mode="rblapsum")
    for scope in ("update_inactive", "update_active"):
        ActivationBottleneckConfig(**base, code_residual=True, rblapsum_surrogate_scope=scope,
                                   rblapsum_boundary_grad_mode="through_rank_kappa")
        with pytest.raises(ValueError):  # needs the code residual
            ActivationBottleneckConfig(**base, rblapsum_surrogate_scope=scope)
    with pytest.raises(ValueError):      # the point-mass correction is on an inactive feature
        ActivationBottleneckConfig(**base, code_residual=True,
                                   rblapsum_surrogate_scope="update_active",
                                   rblapsum_boundary_grad_mode="through_rank")


# ---- rblapsum_center_tokens ---------------------------------------------------- #

def center_gate(scope="pool", center=True, mode="through_rank_kappa"):
    gate = AdaptiveLapSumTopKGate(n_features=64, k=8, j=24, surrogate_mode="rblapsum",
                                  rblapsum_boundary_grad_mode=mode, temperature=1.0,
                                  rblapsum_surrogate_scope=scope,
                                  rblapsum_center_tokens=center)
    gate.train()
    return gate


@pytest.mark.parametrize("scope", ["pool", "inactive", "update_active"])
def test_center_tokens_zeroes_every_features_token_sum(scope):
    torch.manual_seed(7)
    z = 3 * torch.randn(2, 40, 64, dtype=torch.float64) + 2 * torch.randn(64, dtype=torch.float64)
    w = torch.randn(2, 40, 64, dtype=torch.float64)
    g = gate_grad(center_gate(scope), z, w)
    g0 = gate_grad(center_gate(scope, center=False), z, w)
    y = center_gate(scope)(z)
    mask = (y != 0).to(w.dtype)
    sign = z.sign()
    supp = (g - w * mask) * sign                       # the support term, score space
    supp0 = (g0 - w * mask) * sign
    # every feature's support term sums to zero over the tokens ...
    assert torch.allclose(supp.reshape(-1, 64).sum(0), torch.zeros(64, dtype=torch.float64),
                          atol=1e-12)
    assert supp0.reshape(-1, 64).sum(0).abs().max() > 1e-6   # ... which it did not before
    # only pool members carry it, and the forward is unchanged
    pool = z.abs().topk(32, dim=-1).indices
    in_pool = torch.zeros_like(z, dtype=torch.bool).scatter(-1, pool, True)
    assert torch.equal(supp[~in_pool], torch.zeros_like(supp[~in_pool]))
    assert torch.equal(y, center_gate(scope, center=False)(z))


def test_center_tokens_config_validation():
    base = dict(enabled=True, n_features=64, k=8, j=8, placement="residual_out",
                share_projections=True)
    ActivationBottleneckConfig(**base, surrogate_mode="rblapsum", rblapsum_center_tokens=True)
    for mode in ("hard", "lapsum", "rblapsum_sf"):
        with pytest.raises(ValueError):
            ActivationBottleneckConfig(**base, surrogate_mode=mode, rblapsum_center_tokens=True)


# ---- rblapsum_surrogate_scope="first_order" ------------------------------------ #

def vjp(gate, z, up):
    z = z.detach().clone().requires_grad_(True)
    (gate(z) * up).sum().backward()
    return z.grad


def test_first_order_scope_takes_one_support_term_per_path():
    # two gates in a pure carry, y = G2(G1(x)): the first-order gradient is the
    # hard path, plus gate 2's support term carried down through gate 1's mask,
    # plus gate 1's support term computed from the hard-path upstream m2 * w
    from wsparse.bottleneck.rblapsum import first_order_backward
    torch.manual_seed(8)
    x = (3 * torch.randn(5, 64, dtype=torch.float64)).requires_grad_(True)
    w = torch.randn(5, 64, dtype=torch.float64)
    f1, f2 = rbk_gate("first_order"), rbk_gate("first_order")
    p1, p2 = rbk_gate("pool"), rbk_gate("pool")
    c1 = f1(x)
    y = f2(c1)
    assert torch.equal(y.detach(), p2(p1(x.detach())))           # the forward is the gate's
    first_order_backward((y * w).sum(), x)
    m1 = (p1(x.detach()) != 0).to(w.dtype)
    m2 = (y.detach() != 0).to(w.dtype)
    total_c1 = vjp(p2, c1, w)                                     # m2 w + S2^T w
    hard_c1 = m2 * w
    expected = m1 * total_c1 + (vjp(p1, x, hard_c1) - m1 * hard_c1)
    assert torch.allclose(x.grad, expected, atol=1e-12)
    # the ordinary backward differs: it applies gate 1's support term to the total
    pooled = vjp(p1, x, total_c1)
    assert not torch.allclose(pooled, expected)


def test_first_order_scope_trains_end_to_end(tmp_path, monkeypatch):
    import numpy as np
    import wsparse.train as wt
    from tests.test_train import make_fake_dataset, smoke_config
    calls = []
    real = wt.first_order_backward
    monkeypatch.setattr(wt, "first_order_backward",
                        lambda loss, anchor: (calls.append(1), real(loss, anchor)))
    data_dir = make_fake_dataset(tmp_path)
    cfg = smoke_config(data_dir, str(tmp_path / "out"))
    cfg.activation_bottleneck = ActivationBottleneckConfig(
        enabled=True, placement="residual_out", layers="all", n_features=64, k=8, j=8,
        surrogate_mode="rblapsum", rblapsum_boundary_grad_mode="through_rank_kappa",
        share_projections=True, code_residual=True, rblapsum_surrogate_scope="first_order")
    summary = wt.train(cfg)
    assert np.isfinite(summary["best_val_ce"])
    # two micro-batches per step, each through the two-pass backward
    assert len(calls) == cfg.train.max_steps * cfg.train.grad_accum_steps


def test_first_order_is_the_sum_of_single_gate_relaxations():
    # on a real code-residual stack (blocks read the code through the decoder):
    # grad_first_order = grad_hard + sum_l (grad with only gate l relaxed - grad_hard)
    from wsparse.bottleneck.rblapsum import first_order_backward
    torch.manual_seed(9)

    def model(scope):
        torch.manual_seed(0)
        cfg = ActivationBottleneckConfig(
            enabled=True, n_features=96, k=8, j=24, surrogate_mode="rblapsum",
            rblapsum_boundary_grad_mode="through_rank_kappa", placement="residual_out",
            share_projections=True, code_residual=True, init_mode="unit_norm_dictionary",
            rblapsum_surrogate_scope=scope)
        m = build_model(ModelConfig(vocab_size=61, max_seq_len=16, n_layers=3, d_model=24,
                                    n_heads=4)).double()
        apply_activation_bottleneck(m, cfg)
        m.double().train()
        return m

    idx = torch.randint(0, 61, (2, 12))

    def grads(m):
        return torch.cat([p.grad.reshape(-1) for p in m.parameters() if p.grad is not None])

    fo = model("first_order")
    first_order_backward(fo(idx, idx)[1], fo.tok_emb.weight)
    g_fo = grads(fo)

    def relaxed(only):
        m = model("pool")
        gates = [m.code_entry.gate] + [b.residual_out_bottleneck.gate for b in m.blocks]
        for i, gate in enumerate(gates):
            gate.rblapsum_support_scale = 1.0 if i == only else 0.0
        m(idx, idx)[1].backward()
        return grads(m)

    g_hard = relaxed(-1)
    expected = g_hard + sum(relaxed(i) - g_hard for i in range(4))
    # float64; the reference sums 4 relaxed gradients minus 3 hard ones, so allow round-off
    assert torch.allclose(g_fo, expected, rtol=1e-6, atol=1e-9)
    # and it is not the ordinary backward
    pool = model("pool")
    pool(idx, idx)[1].backward()
    assert not torch.allclose(grads(pool), g_fo)


def test_first_order_inactive_keeps_active_features_exact_and_sums_single_gates():
    # the first-order estimator restricted to the inactive members: equals the hard
    # gradient plus the single-gate "inactive" relaxations, summed over the gates
    from wsparse.bottleneck.rblapsum import first_order_backward
    torch.manual_seed(10)
    x = (3 * torch.randn(4, 64, dtype=torch.float64)).requires_grad_(True)
    w = torch.randn(4, 64, dtype=torch.float64)
    f1, f2 = rbk_gate("first_order_inactive"), rbk_gate("first_order_inactive")
    i1, i2 = rbk_gate("inactive"), rbk_gate("inactive")
    c1 = f1(x)
    y = f2(c1)
    first_order_backward((y * w).sum(), x)
    m1 = (i1(x.detach()) != 0).to(w.dtype)
    m2 = (y.detach() != 0).to(w.dtype)
    total_c1 = vjp(i2, c1, w)
    hard_c1 = m2 * w
    expected = m1 * total_c1 + (vjp(i1, x, hard_c1) - m1 * hard_c1)
    assert torch.allclose(x.grad, expected, atol=1e-12)
    ActivationBottleneckConfig(enabled=True, n_features=64, k=8, j=8, placement="residual_out",
                               share_projections=True, surrogate_mode="rblapsum",
                               code_residual=True, rblapsum_surrogate_scope="first_order_inactive")
