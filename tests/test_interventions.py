"""Swap-intervention engine: restart exactness, swap semantics, causality."""

import numpy as np
import pytest
import torch

from wsparse.bottleneck import apply_activation_bottleneck
from wsparse.config import ActivationBottleneckConfig, ModelConfig
from wsparse.interventions import SwapInterventionEngine, per_token_ce
from wsparse.model import build_model


def make(k=3, j=4, n=16, layers="all", sel="abs_topk"):
    torch.manual_seed(0)
    model = build_model(ModelConfig(vocab_size=97, max_seq_len=32, n_layers=3,
                                    d_model=32, n_heads=4))
    bn = ActivationBottleneckConfig(
        enabled=True, n_features=n, k=k, j=j, n_eff=float(k), layers=layers,
        surrogate_mode="hard", selection_mode=sel, placement="residual_out",
        bias=False)

    class Cfg:  # the engine only reads .activation_bottleneck and shapes
        activation_bottleneck = bn

    apply_activation_bottleneck(model, bn, max_steps=10)
    model.eval()
    x = torch.randint(0, 97, (1, 24))
    y = torch.roll(x, -1, dims=1).clone()
    eng = SwapInterventionEngine(model, Cfg, device="cpu")
    state = eng.capture(x, y)
    eng.capture_gradients(x, y, state)
    return eng, state, x, y


# ---- section 28: baseline cache correctness -------------------------------- #
def test_suffix_reproduces_baseline_exactly():
    eng, state, x, y = make()
    diffs = eng.verify_baseline(state, y)
    assert set(diffs) == set(eng.layers)
    for li, d in diffs.items():
        assert d < 1e-5, (li, d)


# ---- section 30: swap tensor semantics -------------------------------------- #
def test_swap_semantics_and_value_modes():
    eng, state, x, y = make()
    li = eng.layers[0]
    t = 7
    sw = eng.build_swap(state, li, t, src_rank=1, tgt_rank=2, kind="random")
    code0 = state.code[li]
    assert code0[t, sw.source] != 0 and code0[t, sw.target] == 0
    codes = eng.apply_swaps(state, li, [sw])
    edited, orig = codes[0], code0
    assert edited[t, sw.source] == 0.0
    assert float(edited[t, sw.target]) == pytest.approx(sw.value)
    assert abs(sw.value) == pytest.approx(abs(float(orig[t, sw.source])))
    # support cardinality preserved at K
    assert int((edited[t] != 0).sum()) == int((orig[t] != 0).sum()) == eng.k
    # everything else untouched: other features at t, all other tokens
    mask = torch.ones_like(orig, dtype=torch.bool)
    mask[t, sw.source] = mask[t, sw.target] = False
    assert torch.equal(edited[mask], orig[mask])
    # abs_topk default: target sign, source magnitude
    zj = float(state.value[li][t, sw.target])
    if zj != 0:
        assert np.sign(sw.value) == np.sign(zj)
    # copy_signed_source mode copies the source value verbatim
    sw2 = eng.build_swap(state, li, t, 1, 2, "random", value_mode="copy_signed_source")
    assert sw2.value == pytest.approx(float(orig[t, sw2.source]))


# ---- section 29: batched == one-at-a-time ----------------------------------- #
def test_batched_matches_single():
    eng, state, x, y = make()
    li = eng.layers[1]
    swaps = [eng.build_swap(state, li, t, r, c, "random")
             for (t, r, c) in [(5, 1, 1), (9, 2, 3), (15, 3, 4)]]
    batched = eng.evaluate(state, li, swaps, y, batch_size=8)
    singles = [eng.evaluate(state, li, [s], y, batch_size=1)[0] for s in swaps]
    for b, s in zip(batched, singles):
        assert b["delta_loss_mean"] == pytest.approx(s["delta_loss_mean"], abs=1e-9)
        assert b["delta_nll_total"] == pytest.approx(s["delta_nll_total"], abs=1e-7)


# ---- section 31: causality --------------------------------------------------- #
def test_causal_prefix_unaffected():
    eng, state, x, y = make()
    for li in eng.layers:
        sw = eng.build_swap(state, li, 12, 1, 1, "random")
        res = eng.evaluate(state, li, [sw], y, batch_size=1)[0]
        assert res["prefix_delta_max_abs"] < 1e-5, (li, res)
        # and the swap does change something at/after t (almost surely)
        assert abs(res["delta_nll_total"]) > 0


# ---- section 32: small-epsilon linearization --------------------------------- #
def test_linearization_at_small_epsilon():
    eng, state, x, y = make()
    li = eng.layers[0]
    t = 10
    g = state.grads[li]
    act, cand = eng.ranks_at(state, li, t)
    i, jf = int(act[0]), int(cand[0])
    eps = 1e-3
    delta = torch.zeros_like(state.code[li])
    delta[t, i] = -float(state.code[li][t, i]) * eps
    delta[t, jf] = 1.0 * eps
    codes = (state.code[li] + delta)[None]
    ce = eng.suffix_ce(li, codes, y)[0]
    valid = y[0] != -100
    fd = float((ce - state.ce)[valid].mean())
    pred = float((g * delta).sum() / valid.sum())
    assert fd == pytest.approx(pred, rel=0.15, abs=1e-7)


# ---- pair selection ----------------------------------------------------------- #
def test_pair_selection_modes_and_seeding():
    eng, state, x, y = make()
    li = eng.layers[0]
    # small K*J -> hybrid auto-upgrades to exhaustive
    ex = eng.select_pairs(state, li, 6, mode="hybrid", exhaustive_threshold=2048)
    assert len(ex) == eng.k * eng.j and all(s.kind == "exhaustive" for s in ex)
    # forced sampled mode: stratified, deterministic in (seed, seq, layer, token)
    s1 = eng.select_pairs(state, li, 6, mode="sampled", random_targets=2, seed=7)
    s2 = eng.select_pairs(state, li, 6, mode="sampled", random_targets=2, seed=7)
    assert [(s.source_rank, s.target_rank) for s in s1] == \
           [(s.source_rank, s.target_rank) for s in s2]
    per_src = {}
    for s in s1:
        per_src.setdefault(s.source_rank, []).append(s.target_rank)
    assert set(per_src) == set(range(1, eng.k + 1))
    assert all(len(v) == 2 and len(set(v)) == 2 for v in per_src.values())
    # hybrid with a tiny threshold: random + labeled tails, no duplicates
    hy = eng.select_pairs(state, li, 6, mode="hybrid", exhaustive_threshold=1,
                          random_targets=2, tail_best=3, tail_worst=3, seed=7)
    kinds = {}
    for s in hy:
        kinds.setdefault(s.kind, 0)
        kinds[s.kind] += 1
    assert kinds["random"] == eng.k * 2
    assert kinds.get("tail_best", 0) == 3 and kinds.get("tail_worst", 0) == 3
    pairs = [(s.source_rank, s.target_rank) for s in hy]
    assert len(pairs) == len(set(pairs))
    # lin present on every built swap when gradients are captured
    assert all(np.isfinite(s.lin) for s in hy)


def test_lin_matrix_matches_build_swap():
    eng, state, x, y = make()
    li = eng.layers[0]
    m = eng.lin_matrix(state, li, 8)
    for (i, j) in [(1, 1), (2, 3), (3, 4)]:
        sw = eng.build_swap(state, li, 8, i, j, "screen")
        assert m[i - 1, j - 1] == pytest.approx(sw.lin, rel=1e-5, abs=1e-9)


def test_non_residual_out_placement_rejected():
    torch.manual_seed(1)
    model = build_model(ModelConfig(vocab_size=97, max_seq_len=16, n_layers=2,
                                    d_model=32, n_heads=4))
    bn = ActivationBottleneckConfig(enabled=True, n_features=16, k=3, j=4,
                                    n_eff=3.0, layers="all", surrogate_mode="hard",
                                    placement="pre_mlp", bias=False)

    class Cfg:
        activation_bottleneck = bn

    apply_activation_bottleneck(model, bn, max_steps=10)
    with pytest.raises(NotImplementedError):
        SwapInterventionEngine(model, Cfg, device="cpu")
