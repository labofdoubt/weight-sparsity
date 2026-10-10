"""wsparse.swap_audit: capture, native swap, suffix, estimator."""

import numpy as np
import pytest
import torch

from wsparse import swap_audit as sa
from wsparse.bottleneck import apply_activation_bottleneck
from wsparse.config import ActivationBottleneckConfig, ModelConfig
from wsparse.model import build_model

VOCAB, T, L, D, N, K, J = 97, 16, 3, 32, 64, 6, 10


def make(scope="pool", code_residual=False, radial=False, center=False, post_norm=True, seed=0):
    torch.manual_seed(seed)
    model = build_model(ModelConfig(vocab_size=VOCAB, max_seq_len=T, n_layers=L, d_model=D, n_heads=4))
    cfg = ActivationBottleneckConfig(enabled=True, n_features=N, k=K, j=J, layers="all",
                                     surrogate_mode="rblapsum", selection_mode="abs_topk",
                                     placement="residual_out", bias=False, post_norm=post_norm,
                                     rblapsum_boundary_grad_mode="through_rank_kappa", temperature=1.0,
                                     rblapsum_surrogate_scope=scope, code_residual=code_residual,
                                     rblapsum_radial_project=radial, rblapsum_center_tokens=center)
    apply_activation_bottleneck(model, cfg, max_steps=10)
    mods = sa.find_bottlenecks(model, type("C", (), {"activation_bottleneck": cfg})())
    full_cfg = type("C", (), {"activation_bottleneck": cfg})()
    idx = torch.randint(0, VOCAB, (3, T)); targets = torch.randint(0, VOCAB, (3, T))
    return model, full_cfg, mods, idx, targets


def test_capture_matches_the_local_decomposition_and_is_inert_when_unused():
    for scope, radial, center in (("pool", False, False), ("first_order", False, False),
                                  ("pool", True, True), ("inactive", True, False)):
        model, cfg, mods, idx, targets = make(scope=scope, radial=radial, center=center)
        # reference gradients without any capture installed
        model.train(); model.zero_grad(set_to_none=True)
        with sa.deterministic_forward(model):
            _, loss = model(idx, targets)
            (sa.first_order_backward(loss, model.tok_emb.weight) if scope.startswith("first_order") else loss.backward())
        ref = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        cap = sa.capture_support(model, cfg, mods, idx, targets, layers=[0, 1, 2])
        for l in (0, 1, 2):
            assert cap[l]["decomp_err"] < 1e-5           # g_z = m*g_y + sign(z)*h, verified inside
            assert cap[l]["h"].shape == (3, T, K + J)
            assert cap[l]["phase"] == "total"
            assert cap[l]["h"].abs().sum() > 0
        assert all(mods[l].gate._support_capture is None for l in range(L))
        # the instrumentation did not change training numerics
        model.train(); model.zero_grad(set_to_none=True)
        with sa.deterministic_forward(model):
            _, loss = model(idx, targets)
            (sa.first_order_backward(loss, model.tok_emb.weight) if scope.startswith("first_order") else loss.backward())
        for n, p in model.named_parameters():
            if p.grad is not None:
                assert torch.equal(p.grad, ref[n]), (scope, n)
        model.zero_grad(set_to_none=True)


def test_first_order_capture_differs_from_pool_scope_on_the_same_weights():
    model_p, cfg_p, mods_p, idx, targets = make(scope="pool", seed=3)
    model_f, cfg_f, mods_f, _, _ = make(scope="first_order", seed=3)
    model_f.load_state_dict(model_p.state_dict())
    hp = sa.capture_support(model_p, cfg_p, mods_p, idx, targets, layers=[0, 1, 2])
    hf = sa.capture_support(model_f, cfg_f, mods_f, idx, targets, layers=[0, 1, 2])
    assert torch.allclose(hp[2]["h"], hf[2]["h"], atol=1e-6)      # the last gate sees only hard paths
    assert not torch.allclose(hp[0]["h"], hf[0]["h"], atol=1e-6)  # earlier gates differ by the products


def test_native_swap_inserts_the_candidate_value_and_keeps_cardinality():
    model, cfg, mods, idx, targets = make()
    z, y, L0 = sa.baseline_codes(model, mods, idx, targets, layers=[1])
    b, t = 1, 7
    order = torch.argsort(z[1][b, t].abs(), descending=True)
    i, j = int(order[K - 1]), int(order[K])
    sw = sa.Swap(b, t, i, j, float(z[1][b, t, i]), float(z[1][b, t, j]))
    y2 = sa.apply_swaps(y[1][[b]], [0], [sw])
    assert y2[0, t, i] == 0 and y2[0, t, j] == z[1][b, t, j]      # removed exactly, the candidate's own signed value
    assert int((y2[0, t] != 0).sum()) == K
    diff = (y2[0] != y[1][b]).nonzero()
    assert set(map(tuple, diff.tolist())) == {(t, i), (t, j)}      # nothing else touched


def test_identity_replay_causal_prefix_batched_and_suffix_agree_with_the_oracle():
    model, cfg, mods, idx, targets = make()
    layer = 1
    z, y, L0 = sa.baseline_codes(model, mods, idx, targets, layers=[layer])
    swaps = []
    for b in range(3):
        for t in (4, 9, 13):
            order = torch.argsort(z[layer][b, t].abs(), descending=True)
            i, j = int(order[K - 2]), int(order[K + 1])
            swaps.append(sa.Swap(b, t, i, j, float(z[layer][b, t, i]), float(z[layer][b, t, j])))
    ident = [sa.Swap(s.seq, s.pos, s.i, s.i, s.z_i, s.z_i) for s in swaps]
    # identity replay reproduces the baseline
    Lid = sa.swap_losses(model, mods, layer, idx, targets, ident, batch_size=4)
    assert np.abs(Lid - L0[[s.seq for s in ident]]).max() < 1e-6
    # oracle, batched vs one at a time
    Lb = sa.swap_losses(model, mods, layer, idx, targets, swaps, batch_size=4)
    L1 = np.array([sa.swap_losses(model, mods, layer, idx, targets, [s], batch_size=1)[0] for s in swaps])
    assert np.abs(Lb - L1).max() < 1e-6
    # the swaps do change the loss
    assert np.abs(Lb - L0[[s.seq for s in swaps]]).max() > 1e-6
    # suffix equals the oracle
    Ls = sa.swap_losses(model, mods, layer, idx, targets, swaps, batch_size=4, suffix=True, baseline_code=y[layer])
    assert np.abs(Ls - Lb).max() < 1e-5
    # causal prefix: per-token CE unchanged before the swap position
    s = swaps[4]
    with torch.no_grad(), sa.deterministic_forward(model):
        model.eval()
        base, _ = model(idx[[s.seq]], targets[[s.seq]])
        h = mods[layer].gate.register_forward_hook(lambda m, i_, o: sa.apply_swaps(o, [0], [s]))
        try:
            after, _ = model(idx[[s.seq]], targets[[s.seq]])
        finally:
            h.remove()
    ce0, ce1 = sa.per_token_ce(base, targets[[s.seq]])[0], sa.per_token_ce(after, targets[[s.seq]])[0]
    assert torch.equal(ce0[: s.pos], ce1[: s.pos]) and not torch.equal(ce0[s.pos:], ce1[s.pos:])


def test_suffix_is_refused_for_code_residual_but_the_oracle_works():
    model, cfg, mods, idx, targets = make(code_residual=True)
    z, y, L0 = sa.baseline_codes(model, mods, idx, targets, layers=[1])
    order = torch.argsort(z[1][0, 5].abs(), descending=True)
    sw = [sa.Swap(0, 5, int(order[K - 1]), int(order[K]), float(z[1][0, 5, order[K - 1]]), float(z[1][0, 5, order[K]]))]
    with pytest.raises(NotImplementedError):
        sa.swap_losses(model, mods, 1, idx, targets, sw, suffix=True, baseline_code=y[1])
    out = sa.swap_losses(model, mods, 1, idx, targets, sw)
    assert np.isfinite(out).all() and abs(out[0] - L0[0]) > 1e-7


def test_pair_sampling_is_seeded_by_token_and_independent_of_the_pool_values():
    pool = torch.arange(100, 100 + K + J)
    a = sa.sample_pairs(pool, K, 3, 4, 5, seed=7, layer=1, seq=2, pos=3)
    b = sa.sample_pairs(pool.flip(0) * 0 + pool, K, 3, 4, 5, seed=7, layer=1, seq=2, pos=3)
    assert [p[:2] for p in a] == [p[:2] for p in b] and len(a) == 5
    assert all(K - 3 <= ri < K and K <= rj < K + 4 for ri, rj, _, _ in a)
    c = sa.sample_pairs(pool, K, 3, 4, 5, seed=7, layer=1, seq=2, pos=4)
    assert [p[:2] for p in a] != [p[:2] for p in c]
    assert len(sa.sample_pairs(pool, K, 3, 4, 5, seed=7, layer=0, seq=0, pos=0, exhaustive=True)) == 12


def test_credit_error_cases():
    r = np.array([1.0, 1.0, -1.0, -1.0]); dL = np.array([-1.0, -1.0, 1.0, 1.0])
    assert sa.credit_error(r, dL, 0.1)["E"] == 0.0                       # perfect agreement
    assert sa.credit_error(-r, dL, 0.1)["E"] == 1.0                      # perfectly wrong
    assert sa.credit_error(np.array([1.0, -1.0, 1.0, -1.0]), dL, 0.1)["E"] == 0.5
    e = sa.credit_error(r, np.array([-1.0, -1.0, 0.05, -0.05]), 0.1)      # no harm class
    assert e["E"] is None and e["n_harm"] == 0 and e["n_improve"] == 2 and e["n_dead"] == 2
    per = {0: {"E": 0.2}, 1: {"E": None}}
    assert sa.aggregate_layers(per, {0: 1.0, 1: 1.0}) is None
    assert sa.aggregate_layers({0: {"E": 0.2}, 1: {"E": 0.4}}, {0: 3.0, 1: 1.0}) == pytest.approx(0.25)


def test_bootstrap_reports_undefined_resamples():
    out = sa.bootstrap_credit(np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0]), np.array([1.0, -1.0, 1.0, -1.0]),
                              np.array([-1.0, 1.0, -1.0, -1.0]), [0], {0: 1.0}, 0.1, n_boot=50, seed=0)
    assert out["aggregate"]["n_defined"] + out["aggregate"]["n_undefined"] == 50
    assert out["aggregate"]["n_undefined"] > 0      # resamples with only seq 1 have no harm class
