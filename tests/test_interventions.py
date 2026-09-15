"""Swap-intervention engine: restart exactness, swap semantics, causality."""

import os

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


def make_lapsum(sel="abs_topk"):
    torch.manual_seed(3)
    model = build_model(ModelConfig(vocab_size=97, max_seq_len=32, n_layers=2,
                                    d_model=32, n_heads=4))
    bn = ActivationBottleneckConfig(
        enabled=True, n_features=16, k=3, j=4, n_eff=3.0, layers="all",
        surrogate_mode="lapsum_scheduled", selection_mode=sel,
        temperature_schedule="constant", temperature_start=1.0,
        temperature_scale_mode="absolute", placement="residual_out", bias=False)

    class TrainCfg:
        max_steps = 1000

    class Cfg:
        activation_bottleneck = bn
        train = TrainCfg

    apply_activation_bottleneck(model, bn, max_steps=1000)
    model.eval()
    x = torch.randint(0, 97, (1, 20))
    y = torch.roll(x, -1, dims=1).clone()
    eng = SwapInterventionEngine(model, Cfg, device="cpu")
    state = eng.capture(x, y)
    return eng, state, x, y


def test_lapsum_support_gradient_capture():
    eng, state, x, y = make_lapsum()
    ok = eng.capture_lapsum_gradients(x, y, state, step=500)
    assert ok
    t = 9
    for li in eng.layers:
        g = state.ls_grad[li]
        assert g.shape == state.value[li].shape
        act, cand = eng.ranks_at(state, li, t)
        # the hallmark of the surrogate: INACTIVE candidates get support gradient
        assert float(g[t, torch.as_tensor(cand)].abs().sum()) > 0
        # outside the K+J pool: exactly nothing
        outside = torch.ones(16, dtype=torch.bool)
        outside[torch.as_tensor(act)] = False
        outside[torch.as_tensor(cand)] = False
        assert float(g[t, outside].abs().max()) == 0.0
    # Q matrix: exact broadcast identity Q[i,j] = g_j - g_i
    li = eng.layers[0]
    q = eng.q_matrix(state, li, t)
    assert q.shape == (eng.k, eng.j)
    g = state.ls_grad[li]
    act, cand = eng.ranks_at(state, li, t)
    for i in (0, 2):
        for jj in (0, 3):
            want = float(g[t, int(cand[jj])] - g[t, int(act[i])])
            assert q[i, jj] == pytest.approx(want, rel=1e-4, abs=1e-8)
    # forward/eval state restored, no param grads left behind
    assert not eng.model.training
    assert all(p.grad is None for p in eng.model.parameters())


def test_lapsum_capture_refuses_non_lapsum():
    eng, state, x, y = make()
    assert eng.capture_lapsum_gradients(x, y, state, step=100) is False
    assert not state.ls_grad


def test_sign_convention_gradient_descent_direction():
    # g_j < g_i  =>  Q < 0  =>  descent on s (delta s = -eta g) raises s_j - s_i
    g_i, g_j, eta = 0.5, -0.2, 0.1
    q = g_j - g_i
    assert q < 0
    ds_j, ds_i = -eta * g_j, -eta * g_i
    assert (ds_j - ds_i) > 0                      # j rises relative to i


def test_alignment_summary_math_on_synthetic_rows(tmp_path):
    import importlib.util
    import pandas as pd
    spec = importlib.util.spec_from_file_location(
        "swapcli", os.path.join(os.path.dirname(__file__), "..", "analysis",
                                "swap_interventions.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(0)
    n = 200
    q = rng.normal(size=n)
    base = dict(checkpoint_step=100, sequence_id=0, layer_index=0, token_index=5,
                K=4, J=8, sample_kind="random")
    def rows(delta):
        return pd.DataFrame([dict(base, source_rank=int(i % 4) + 1,
                                  target_candidate_rank=int(i % 8) + 1,
                                  delta_loss_mean=float(delta[i]),
                                  delta_nll_total=float(delta[i]),
                                  delta_loss_linearized=float("nan"),
                                  lapsum_swap_score=float(q[i]))
                             for i in range(n)])
    d = tmp_path / "ds"; d.mkdir()
    rows(3.0 * q).to_parquet(d / "rows_step100.parquet")
    mod.summarize(str(d))
    ctx = pd.read_parquet(d / "context_summary.parquet").iloc[0]
    assert ctx.lapsum_swap_spearman == pytest.approx(1.0)
    assert ctx.lapsum_swap_pearson == pytest.approx(1.0)
    assert ctx.lapsum_swap_sign_agreement == pytest.approx(1.0)
    assert ctx.lapsum_swap_beneficial_precision == pytest.approx(1.0)
    rows(-3.0 * q).to_parquet(d / "rows_step100.parquet")
    mod.summarize(str(d))
    ctx = pd.read_parquet(d / "context_summary.parquet").iloc[0]
    assert ctx.lapsum_swap_spearman == pytest.approx(-1.0)
    # degenerate Q -> flagged, correlations absent (NaN), never silently 0
    rows(np.zeros(n) + 1.0).assign(lapsum_swap_score=0.0).to_parquet(
        d / "rows_step100.parquet")
    mod.summarize(str(d))
    ctx = pd.read_parquet(d / "context_summary.parquet").iloc[0]
    assert bool(ctx.lapsum_swap_degenerate)
    assert not np.isfinite(ctx.get("lapsum_swap_spearman", float("nan")))


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
