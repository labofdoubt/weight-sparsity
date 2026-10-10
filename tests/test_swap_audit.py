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


def test_paired_deltas_match_the_oracle_differences_and_have_zero_replay_noise():
    model, cfg, mods, idx, targets = make()
    layer = 1
    z, y, L0 = sa.baseline_codes(model, mods, idx, targets, layers=[layer])
    swaps = []
    for b in range(3):
        for t in (3, 8, 12):
            order = torch.argsort(z[layer][b, t].abs(), descending=True)
            i, j = int(order[K - 1]), int(order[K])
            swaps.append(sa.Swap(b, t, i, j, float(z[layer][b, t, i]), float(z[layer][b, t, j])))
    d, noise = sa.swap_deltas(model, mods, layer, idx, targets, swaps, batch_size=5, n_identity=2)
    L1 = sa.swap_losses(model, mods, layer, idx, targets, swaps, batch_size=5)
    assert noise.size > 0 and noise.max() == 0.0
    assert np.abs(d - (L1 - L0[[s.seq for s in swaps]])).max() < 1e-6
    ds, _ = sa.swap_deltas(model, mods, layer, idx, targets, swaps, batch_size=5, suffix=True, baseline_code=y[layer])
    assert np.abs(ds - d).max() < 1e-6


# ---- paired downstream-support replay -------------------------------------- #

def _swaps_for(z, layer, pairs=((0, 3), (1, 8), (2, 12)), lo=K - 1, hi=K):
    out = []
    for b, t in pairs:
        order = torch.argsort(z[layer][b, t].abs(), descending=True)
        i, j = int(order[lo]), int(order[hi])
        out.append(sa.Swap(b, t, i, j, float(z[layer][b, t, i]), float(z[layer][b, t, j])))
    return out


def test_mask_sink_records_the_actual_selection_on_both_gate_paths():
    for mode in ("rblapsum", "hard"):
        model, cfg, mods, idx, targets = make()
        if mode == "hard":
            for m in mods:
                m.gate.surrogate_mode = "hard"
        sink = {}; mods[1].gate._mask_sink = sink
        with torch.no_grad():
            model.eval(); model(idx)
        mods[1].gate._mask_sink = None
        z, y, _ = sa.baseline_codes(model, mods, idx, targets, layers=[1])
        assert sink["mask"].dtype == torch.bool and sink["mask"].shape == y[1].shape
        assert torch.equal(sink["mask"], y[1] != 0) and int(sink["mask"].sum(-1).min()) == K


def test_unswapped_mask_replay_reproduces_the_baseline_and_only_masks_are_frozen():
    model, cfg, mods, idx, targets = make()
    layer = 1
    z, y, L0 = sa.baseline_codes(model, mods, idx, targets, layers=list(range(L)))
    swaps = _swaps_for(z, layer)
    res = sa.paired_swap_deltas(model, mods, layer, idx, targets, swaps, batch_size=4, verify_replay=10)
    assert res["replay_err"] == 0.0                                  # fixed masks, no swap == baseline
    assert res["valid"].all() and res["later_layers"] == [2]
    # under the fixed condition the later gate keeps M0 but its values move
    with sa._ForwardScope(model, mods):
        _, M0, _ = sa._run_paired_batch(model, mods, layer, idx[[0]], targets[[0]], [], [], "plain", None, [1, 2], torch.float32)
        s = swaps[0]
        logits_f, Mf, outs = sa._run_paired_batch(model, mods, layer, idx[[0]], targets[[0]], [0], [s], "swap",
                                                  {2: M0[2]}, [1, 2], torch.float32, grad=True)
    assert torch.equal((outs[2] != 0) | (outs[2] == 0), torch.ones_like(M0[2]))
    assert torch.equal(outs[2].detach() != 0, M0[2] & (outs[2].detach() != 0))   # support within M0
    assert not torch.equal(outs[2].detach(), y[2][[0]])                   # values changed
    # the forced swap carries the candidate's own signed live value
    y1 = outs[1].detach()[0, s.pos]
    assert y1[s.i] == 0 and torch.isclose(y1[s.j], z[layer][0, s.pos, s.j]) and int((y1 != 0).sum()) == K


def test_last_layer_native_and_fixed_coincide_and_cascade_is_real_elsewhere():
    model, cfg, mods, idx, targets = make()
    z, y, L0 = sa.baseline_codes(model, mods, idx, targets, layers=list(range(L)))
    res_last = sa.paired_swap_deltas(model, mods, L - 1, idx, targets, _swaps_for(z, L - 1), batch_size=4)
    assert np.array_equal(res_last["dL_native"], res_last["dL_fixed"]) and res_last["cascade"].shape[1] == 0
    res0 = sa.paired_swap_deltas(model, mods, 0, idx, targets, _swaps_for(z, 0), batch_size=4)
    assert res0["cascade"].sum() > 0                                   # later supports did change
    assert np.abs(res0["dL_native"] - res0["dL_fixed"]).max() > 0


def test_hard_output_gradient_matches_the_hard_backward_and_the_forward_is_unchanged():
    model, cfg, mods, idx, targets = make()
    g, fwd_err = sa.hard_output_gradients(model, mods, idx, targets, layers=[0, 1, 2])
    assert fwd_err == 0.0
    # reference: the surrogate Function with its support term switched off
    for m in mods:
        m.gate.rblapsum_support_scale = 0.0
    grabbed = {}
    def tap(m_, i, o, l):
        o.register_hook(lambda gg, l=l: grabbed.__setitem__(l, gg.detach()))
    hs = [mods[l].gate.register_forward_hook(lambda m_, i, o, l=l: tap(m_, i, o, l)) for l in (0, 1, 2)]
    model.eval()
    with sa.deterministic_forward(model), torch.enable_grad():
        logits, _ = model(idx, targets)
        sa.sequence_loss(logits, targets).sum().backward()
    for h in hs:
        h.remove()
    model.zero_grad(set_to_none=True)
    for l in (0, 1, 2):
        assert torch.allclose(g[l], grabbed[l], atol=1e-6, rtol=1e-5)


def test_paired_causal_prefix_batched_vs_single_and_restoration_after_failure():
    model, cfg, mods, idx, targets = make()
    layer = 1
    z, y, L0 = sa.baseline_codes(model, mods, idx, targets, layers=list(range(L)))
    swaps = _swaps_for(z, layer, pairs=((0, 5), (1, 7), (2, 9), (0, 11), (1, 3)))
    rb = sa.paired_swap_deltas(model, mods, layer, idx, targets, swaps, batch_size=3)
    rs = [sa.paired_swap_deltas(model, mods, layer, idx, targets, [s], batch_size=1) for s in swaps]
    for k_, r1 in enumerate(rs):
        assert abs(r1["dL_native"][0] - rb["dL_native"][k_]) < 1e-6 and abs(r1["dL_fixed"][0] - rb["dL_fixed"][k_]) < 1e-6
    # causal prefix under the fixed condition
    s = swaps[1]
    with sa._ForwardScope(model, mods):
        lp, M0, _ = sa._run_paired_batch(model, mods, layer, idx[[s.seq]], targets[[s.seq]], [], [], "plain", None, [2], torch.float32)
        lf, _, _ = sa._run_paired_batch(model, mods, layer, idx[[s.seq]], targets[[s.seq]], [0], [s], "swap", {2: M0[2]}, [], torch.float32)
    ce0, ce1 = sa.per_token_ce(lp, targets[[s.seq]])[0], sa.per_token_ce(lf, targets[[s.seq]])[0]
    assert torch.equal(ce0[: s.pos], ce1[: s.pos]) and not torch.equal(ce0[s.pos:], ce1[s.pos:])
    # restoration after a failure inside the scope
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.3
    model.train()
    bad = mods[2].gate.register_forward_hook(lambda m_, i, o: (_ for _ in ()).throw(RuntimeError("boom")))
    try:
        with pytest.raises(RuntimeError):
            sa.paired_swap_deltas(model, mods, layer, idx, targets, swaps[:1], batch_size=1)
    finally:
        bad.remove()
    assert model.training and all(m.gate._mask_sink is None for m in mods)
    assert all(m.p == 0.3 for m in model.modules() if isinstance(m, torch.nn.Dropout))
    assert all(len(m.gate._forward_hooks) == 0 and len(m.gate._forward_pre_hooks) == 0 for m in mods)


def test_transitions_and_paired_scores():
    r = np.array([1, 1, -1, -1, 1, -1], float)
    dn = np.array([-1, 1, 1, -1, -1, 1], float)      # native: right, wrong, right, wrong, right, right
    df = np.array([-1, -1, 1, 1, 0.0, -1], float)    # fixed:  right, right, right, right, dead, wrong
    t = sa.transitions(r, dn, df, 0.1)
    assert t["n_both"] == 5
    assert t["beneficial"] == dict(n=2, repaired=1, broken=0, right_native=1, right_fixed=2)
    assert t["harmful"] == dict(n=3, repaired=1, broken=1, right_native=2, right_fixed=2)
    seq = np.array([0, 0, 1, 1, 2, 2]); layer = np.zeros(6, int)
    out = sa.paired_scores(seq, layer, {"r": r}, dn, df, [0], {0: 1.0}, 0.1, n_boot=20, seed=0)
    assert out["r"]["native"]["E"] == pytest.approx(0.5 * (1 / 3 + 1 / 3))
    assert out["r"]["fixed"]["E"] == pytest.approx(0.5 * (1 / 3 + 0 / 2))   # fixed: improve {0,1,5}, harm {2,3}
    assert set(out["r"]["bootstrap"]) == {"native", "fixed", "diff", "n_boot"}
