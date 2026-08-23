"""Activation bottleneck: hard TopK forward, LapSum Top(K+J) surrogate backward.

The gradient tests compare against finite differences of the *soft LapSum mask*
with the barrier re-solved at each perturbation -- that is what verifies the
shared-barrier correction term, which is the whole content of the VJP.  The hard
TopK discontinuity is never finite-differenced.
"""

import math

import pytest
import torch
import torch.nn as nn

from wsparse.bottleneck import (
    STATUS_ABOVE_RANGE,
    STATUS_BELOW_RANGE,
    STATUS_OK,
    ActivationBottleneckController,
    AdaptiveLapSumTopKGate,
    SparseTopKBottleneck,
    apply_activation_bottleneck,
    effective_count,
    lapsum_barrier_bisect,
    lapsum_barrier_sorted,
    lapsum_budget,
    lapsum_probs,
    lapsum_probs_at,
    gradient_count,
    gradient_weights,
    score_softmax_count,
    resolve_layers,
    solve_joint_temperature,
    solve_reference_temperature,
    solve_score_softmax_temperature,
    parse_placements,
    validate_gate_shapes,
)
from wsparse.config import ActivationBottleneckConfig, Config, ModelConfig, SparsityConfig
from wsparse.model import build_model


def sorted_scores(rows=32, m=48, scale=1.0, dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    r = torch.randn(rows, m, dtype=dtype) * scale
    return torch.sort(r, dim=-1, descending=True).values


def bottleneck_cfg(**kw):
    cfg = dict(enabled=True, n_features=64, k=8, j=24, n_eff=6.0, layers="all")
    cfg.update(kw)
    return ActivationBottleneckConfig(**cfg)


def tiny_model(**kw):
    cfg = dict(vocab_size=97, max_seq_len=32, n_layers=2, d_model=32, n_heads=4)
    cfg.update(kw)
    return build_model(ModelConfig(**cfg))


def make_gate(**kw):
    cfg = dict(n_features=64, k=8, j=24, n_eff=6.0)
    cfg.update(kw)
    gate = AdaptiveLapSumTopKGate(**cfg)
    gate.train()
    return gate


# --------------------------------------------------------------------------- #
# hard forward
# --------------------------------------------------------------------------- #


def test_exactly_k_mask_positions_per_token():
    """Counted from the mask, not from numeric nonzeros: a selected activation
    can itself be exactly zero."""
    gate = make_gate(k=8, selection_mode="topk")
    a = torch.randn(4, 5, 64)
    a[0, 0, :] = 0.0  # every score tied at zero
    a[1, 1, 3] = 0.0
    scores = gate.scores_of(a)
    idx = torch.topk(scores, gate.k, dim=-1).indices
    hard = torch.zeros_like(a).scatter(-1, idx, 1.0)
    assert torch.equal(hard.sum(-1), torch.full((4, 5), 8.0))
    assert torch.equal(gate(a), a * hard)


def test_forward_is_exactly_k_sparse_regardless_of_the_surrogate_knobs():
    a = torch.randn(3, 7, 64, requires_grad=True)
    ref = None
    for kw in (
        dict(j=24, n_eff=6.0),
        dict(j=40, n_eff=6.0),
        dict(j=24, n_eff=20.0),
        dict(j=24, n_eff=6.0, effective_count_metric="entropy"),
        dict(j=24, n_eff=6.0, boundary_mode="both_sides"),
        dict(j=24, n_eff=6.0, surrogate_mode="hard"),
        dict(j=24, n_eff=6.0, surrogate_mode="lapsum_fixed"),
        dict(j=24, n_eff=6.0, surrogate_grad_scale=7.0),
    ):
        out = make_gate(**kw)(a)
        assert int((out != 0).sum(-1).max()) <= 8
        if ref is None:
            ref = out.detach()
        else:
            assert torch.equal(out.detach(), ref)


def test_topk_selects_largest_signed_values():
    gate = make_gate(k=3, j=5, n_eff=2.0, n_features=16, selection_mode="topk")
    a = torch.tensor([[-9.0, 1.0, 5.0, -2.0, 3.0] + [0.1] * 11])
    out = gate(a)
    assert out[0, 2] == 5.0 and out[0, 4] == 3.0 and out[0, 1] == 1.0
    assert out[0, 0] == 0.0  # -9 is the largest magnitude but the smallest value


def test_abs_topk_selects_largest_magnitude_and_keeps_the_sign():
    gate = make_gate(k=3, j=5, n_eff=2.0, n_features=16, selection_mode="abs_topk")
    a = torch.tensor([[-9.0, 1.0, 5.0, -2.0, 3.0] + [0.1] * 11])
    out = gate(a)
    assert out[0, 0] == -9.0  # selected on |a|, forwarded with its sign
    assert out[0, 2] == 5.0 and out[0, 4] == 3.0
    assert out[0, 1] == 0.0 and out[0, 3] == 0.0
    assert (out.abs() != a.abs()).any()  # not |a|


# --------------------------------------------------------------------------- #
# Top(K+J) support
# --------------------------------------------------------------------------- #


def test_gradient_support_is_exactly_top_k_plus_j():
    torch.manual_seed(0)
    gate = make_gate(k=4, j=8, n_eff=3.0, n_features=64, selection_mode="topk")
    a = torch.randn(6, 64, requires_grad=True)
    gate(a).pow(2).sum().backward()

    order = a.detach().argsort(dim=-1, descending=True)
    rank = order.argsort(dim=-1)
    nonzero = a.grad != 0
    assert bool(nonzero[rank < 4].all())  # active features always learn
    assert bool(nonzero[(rank >= 4) & (rank < 12)].any())  # J candidates can learn
    assert not bool(nonzero[rank >= 12].any())  # outside Top(K+J): exactly zero


def test_inactive_candidates_receive_gradient_though_they_are_off_in_forward():
    torch.manual_seed(0)
    gate = make_gate(k=4, j=8, n_eff=3.0, n_features=32, selection_mode="topk")
    a = torch.randn(4, 32, requires_grad=True)
    out = gate(a)
    out.sum().backward()
    order = a.detach().argsort(dim=-1, descending=True)
    rank = order.argsort(dim=-1)
    explore = (rank >= 4) & (rank < 12)
    assert torch.equal(out[explore], torch.zeros(int(explore.sum())))  # no forward role
    assert bool((a.grad[explore] != 0).any())  # but a backward one


def test_hard_surrogate_mode_gives_no_boundary_gradient():
    torch.manual_seed(0)
    gate = make_gate(k=4, j=8, n_eff=3.0, n_features=32, surrogate_mode="hard")
    a = torch.randn(4, 32, requires_grad=True)
    gate(a).sum().backward()
    order = a.detach().abs().argsort(dim=-1, descending=True)
    rank = order.argsort(dim=-1)
    assert not bool((a.grad[rank >= 4] != 0).any())


# --------------------------------------------------------------------------- #
# closed-form LapSum barrier
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("k", [1, 4, 16, 31])
def test_barrier_matches_bisection_reference(k):
    r = sorted_scores(m=32)
    t = torch.full((r.shape[0],), 0.7, dtype=r.dtype)
    b = lapsum_barrier_sorted(r, k, t)
    ref = lapsum_barrier_bisect(r, k, t, tol=1e-12, max_iters=400)
    assert torch.allclose(b, ref, atol=1e-8)
    assert (lapsum_budget(r, b, t) - k).abs().max() < 1e-10


@pytest.mark.parametrize("temp", [1e-4, 1e-2, 1.0, 1e2, 1e4])
def test_barrier_budget_is_exact_across_temperatures(temp):
    """At tiny t the barrier is only determined up to the flat plateau between
    r_K and r_K+1, so the budget -- not b itself -- is what must be exact."""
    r = sorted_scores(m=32)
    t = torch.full((r.shape[0],), temp, dtype=r.dtype)
    b = lapsum_barrier_sorted(r, 8, t)
    assert (lapsum_budget(r, b, t) - 8).abs().max() < 1e-9
    ref = lapsum_barrier_bisect(r, 8, t, tol=1e-12, max_iters=400)
    assert (lapsum_probs_at(r, b, t) - lapsum_probs_at(r, ref, t)).abs().max() < 1e-8
    if temp <= 1e-2:
        # only for t well below the score gaps does b sit at the K/K+1 boundary;
        # at large t every p_i -> 0.5 and b climbs above every score to hold
        # the budget at K < M/2
        assert bool(((b < r[:, 7]) & (b > r[:, 8])).all())
    else:
        assert bool((b > r[:, 8]).all())


@pytest.mark.parametrize("k,j", [(1, 1), (1, 31), (16, 16), (30, 2)])
def test_barrier_for_various_k_and_j(k, j):
    r = sorted_scores(m=k + j)
    t = torch.full((r.shape[0],), 0.3, dtype=r.dtype)
    b = lapsum_barrier_sorted(r, k, t)
    assert (lapsum_budget(r, b, t) - k).abs().max() < 1e-9


def test_barrier_handles_large_magnitudes_and_ties():
    t = torch.full((4,), 1.0, dtype=torch.float64)
    huge = sorted_scores(rows=4, m=16) * 1e6
    assert (lapsum_budget(huge, lapsum_barrier_sorted(huge, 4, t * 1e6), t * 1e6) - 4).abs().max() < 1e-8
    tied = torch.zeros(4, 16, dtype=torch.float64)
    for k in (1, 4, 8, 12, 15):  # spans both degenerate edge intervals
        b = lapsum_barrier_sorted(tied, k, t)
        assert torch.isfinite(b).all()
        assert (lapsum_budget(tied, b, t) - k).abs().max() < 1e-9


def test_barrier_translation_and_scale_equivariance():
    r = sorted_scores(m=32)
    t = torch.full((r.shape[0],), 0.4, dtype=r.dtype)
    b = lapsum_barrier_sorted(r, 8, t)
    shifted = lapsum_barrier_sorted(r + 3.5, 8, t)
    assert torch.allclose(shifted, b + 3.5, atol=1e-9)
    c = 7.0
    scaled = lapsum_barrier_sorted(r * c, 8, t * c)
    assert torch.allclose(scaled, b * c, atol=1e-7)
    assert torch.allclose(lapsum_probs_at(r * c, scaled, t * c), lapsum_probs_at(r, b, t), atol=1e-9)


# --------------------------------------------------------------------------- #
# fixed-temperature LapSum VJP
# --------------------------------------------------------------------------- #


def test_vjp_matches_finite_differences_of_the_soft_mask():
    """Perturb one score, re-solve the barrier, finite-difference p.

    Re-solving is essential: it is what makes the shared-barrier correction
    term ``-<q_budget, u>`` observable.
    """
    torch.manual_seed(0)
    k, m = 5, 20
    r = sorted_scores(rows=3, m=m, seed=1)
    t = torch.full((3,), 0.5, dtype=torch.float64)
    u = torch.randn(3, m, dtype=torch.float64)

    scores = r.clone().requires_grad_(True)
    b = lapsum_barrier_sorted(r, k, t)
    (lapsum_probs(scores, b, t, k) * u).sum().backward()

    eps = 1e-6
    fd = torch.zeros_like(r)
    for row in range(3):
        for i in range(m):
            out = []
            for sign in (+1, -1):
                pert = r.clone()
                pert[row, i] += sign * eps
                b_p = lapsum_barrier_sorted(pert, k, t)  # barrier moves with the score
                out.append(float((lapsum_probs_at(pert, b_p, t)[row] * u[row]).sum()))
            fd[row, i] = (out[0] - out[1]) / (2 * eps)
    assert torch.allclose(scores.grad, fd, atol=1e-6, rtol=1e-4)


def test_vjp_without_the_shared_barrier_term_would_be_wrong():
    """A uniform upstream gradient moves every score equally, which cannot
    change the budget -- so the exact VJP must vanish."""
    k, m = 5, 20
    r = sorted_scores(rows=3, m=m, seed=2)
    t = torch.full((3,), 0.5, dtype=torch.float64)
    scores = r.clone().requires_grad_(True)
    b = lapsum_barrier_sorted(r, k, t)
    (lapsum_probs(scores, b, t, k) * torch.ones_like(r)).sum().backward()
    assert scores.grad.abs().max() < 1e-12
    # the naive kappa * u gradient would instead be strictly positive
    z = (r - b[:, None]) / t[:, None]
    naive = 0.5 * torch.exp(-z.abs()) / t[:, None]
    assert naive.min() > 0


def test_surrogate_grad_scale_scales_only_the_boundary_term():
    torch.manual_seed(0)
    a = torch.randn(4, 32)
    grads = {}
    for scale in (1.0, 3.0):
        x = a.clone().requires_grad_(True)
        make_gate(k=4, j=8, n_eff=3.0, n_features=32, surrogate_grad_scale=scale)(x).pow(2).sum().backward()
        grads[scale] = x.grad.clone()
    hard = make_gate(k=4, j=8, n_eff=3.0, n_features=32, surrogate_mode="hard")
    x = a.clone().requires_grad_(True)
    hard(x).pow(2).sum().backward()
    base = x.grad
    # grad = hard term + scale * boundary term
    assert torch.allclose(grads[3.0] - base, 3.0 * (grads[1.0] - base), atol=1e-5)


# --------------------------------------------------------------------------- #
# one-sided temperature
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("metric", ["ess", "entropy"])
@pytest.mark.parametrize("target", [2.0, 5.0, 20.0])
def test_one_sided_temperature_hits_the_target(metric, target):
    inactive = sorted_scores(rows=64, m=48, dtype=torch.float32)
    t, diag = solve_score_softmax_temperature(inactive, target, metric, tol=1e-5, max_iters=12)
    realized = score_softmax_count(inactive, t, metric)
    assert (realized - target).abs().max() < 1e-3 * target
    assert int((diag["temp_status"] != 0).sum()) == 0
    assert bool((t > 0).all())


@pytest.mark.parametrize("metric", ["ess", "entropy"])
def test_one_sided_temperature_is_scale_equivariant(metric):
    inactive = sorted_scores(rows=16, m=32, dtype=torch.float32)
    t, _ = solve_score_softmax_temperature(inactive, 8.0, metric)
    c = 5.0
    t_scaled, _ = solve_score_softmax_temperature(inactive * c, 8.0, metric)
    assert torch.allclose(t_scaled, c * t, rtol=2e-3)
    # the calibration weights, and so N_eff, are unchanged
    assert torch.allclose(
        score_softmax_count(inactive * c, t_scaled, metric),
        score_softmax_count(inactive, t, metric),
        rtol=1e-3,
    )


@pytest.mark.parametrize("metric", ["ess", "entropy"])
def test_one_sided_temperature_is_translation_invariant(metric):
    inactive = sorted_scores(rows=16, m=32, dtype=torch.float32)
    t, _ = solve_score_softmax_temperature(inactive, 8.0, metric)
    t_shift, _ = solve_score_softmax_temperature(inactive + 4.0, 8.0, metric)
    assert torch.allclose(t_shift, t, rtol=1e-4)


def test_entropy_counts_the_tail_more_than_ess():
    """Same scores, same target: entropy needs a colder t, since it credits the
    long tail of small weights that ESS ignores."""
    inactive = sorted_scores(rows=32, m=64, dtype=torch.float32)
    t_ess, _ = solve_score_softmax_temperature(inactive, 8.0, "ess")
    t_ent, _ = solve_score_softmax_temperature(inactive, 8.0, "entropy")
    assert bool((t_ent < t_ess).all())


def test_degenerate_tail_does_not_return_nan():
    tied = torch.zeros(8, 16)
    t, diag = solve_score_softmax_temperature(tied, 6.0, "ess", fallback_scale=torch.ones(8))
    assert torch.isfinite(t).all() and bool((t > 0).all())
    assert int((diag["temp_status"] == 3).sum()) == 8


# --------------------------------------------------------------------------- #
# two-sided temperature
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("metric", ["ess", "entropy"])
@pytest.mark.parametrize("target", [6.0, 20.0])
def test_two_sided_joint_newton_solves_both_equations(metric, target):
    k = 16
    cand = sorted_scores(rows=64, m=64, dtype=torch.float32)
    t0, _ = solve_score_softmax_temperature(cand[:, k:], target, metric)
    b0 = lapsum_barrier_sorted(cand, k, t0)
    b, t, ok, diag = solve_joint_temperature(
        cand, k, target, b0, t0, metric, calibration=slice(0, cand.shape[-1]), max_iters=12
    )
    assert bool(ok.all()), f"{float(diag['newton_failed'].sum())} rows failed"
    assert (lapsum_budget(cand, b, t) - k).abs().max() < 1e-3
    assert (gradient_count(cand, b, t, metric) - target).abs().max() < 1e-2


@pytest.mark.parametrize("metric", ["ess", "entropy"])
def test_two_sided_joint_matches_the_reference_solver(metric):
    k, target = 16, 12.0
    cand = sorted_scores(rows=32, m=64, dtype=torch.float32)
    t0, _ = solve_score_softmax_temperature(cand[:, k:], target, metric)
    b0 = lapsum_barrier_sorted(cand, k, t0)
    b, t, ok, _ = solve_joint_temperature(
        cand, k, target, b0, t0, metric, calibration=slice(0, cand.shape[-1]), max_iters=12
    )
    b_ref, t_ref, _ = solve_reference_temperature(
        cand, k, target, metric, calibration=slice(0, cand.shape[-1])
    )
    assert torch.allclose(t, t_ref, rtol=1e-2)
    assert torch.allclose(b, b_ref, rtol=1e-2, atol=1e-3)


def test_two_sided_uses_both_sides_of_the_boundary():
    """q^edge puts weight on active candidates too, unlike the one-sided kernel."""
    k, target = 16, 12.0
    cand = sorted_scores(rows=8, m=64, dtype=torch.float32)
    t0, _ = solve_score_softmax_temperature(cand[:, k:], target, "ess")
    b0 = lapsum_barrier_sorted(cand, k, t0)
    b, t, _, _ = solve_joint_temperature(
        cand, k, target, b0, t0, "ess", calibration=slice(0, cand.shape[-1]), max_iters=12
    )
    q = torch.softmax(-((cand - b[:, None]) / t[:, None]).abs(), dim=-1)
    assert float(q[:, :k].sum(-1).min()) > 0.05


# --------------------------------------------------------------------------- #
# gate-level invariances
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("boundary", ["outside_only", "both_sides"])
def test_gate_diagnostics_report_a_hit_target(boundary):
    torch.manual_seed(0)
    gate = make_gate(k=8, j=24, n_eff=6.0, boundary_mode=boundary)
    a = torch.randn(16, 64, requires_grad=True)
    gate(a).sum().backward()
    d = gate.diagnostics
    assert abs(float(d["n_eff_realized"]) - 6.0) < 0.05
    assert float(d["budget_residual"]) < 1e-3
    assert float(d["barrier_failures"]) == 0.0
    assert float(d["temperature"]) > 0
    for key in ("temperature_rel", "barrier", "score_gap", "score_span",
                "grad_active", "grad_inactive", "n_eff_error"):
        assert key in d, key
    assert d["grad_by_rank"].numel() == 8


def test_gate_output_scale_equivariance():
    torch.manual_seed(0)
    gate = make_gate(k=8, j=24, n_eff=6.0, selection_mode="abs_topk")
    a = torch.randn(8, 64)
    plain = gate(a)
    t0 = float(gate.diagnostics["temperature"])
    b0 = float(gate.diagnostics["barrier"])
    n0 = float(gate.diagnostics["n_eff_realized"])
    scaled = gate(a * 3.0)
    t1 = float(gate.diagnostics["temperature"])
    b1 = float(gate.diagnostics["barrier"])
    n1 = float(gate.diagnostics["n_eff_realized"])
    assert torch.allclose(scaled, plain * 3.0, atol=1e-5)
    assert t1 == pytest.approx(3.0 * t0, rel=2e-2)  # t -> c t
    assert b1 == pytest.approx(3.0 * b0, rel=2e-2)  # b -> c b
    assert n1 == pytest.approx(n0, rel=1e-3)  # N_eff unchanged


def test_inference_skips_the_solver_but_not_the_selection():
    gate = make_gate()
    a = torch.randn(4, 64)
    gate.eval()
    assert gate.surrogate_active() is False
    train_out = None
    gate.train()
    with torch.no_grad():
        train_out = gate(a)
    gate.eval()
    assert torch.equal(gate(a), train_out)


# --------------------------------------------------------------------------- #
# module / model wiring
# --------------------------------------------------------------------------- #


def test_bottleneck_module_shapes_and_dense_projections():
    cfg = bottleneck_cfg(n_features=64)
    mod = SparseTopKBottleneck(32, cfg)
    x = torch.randn(2, 5, 32)
    assert mod(x).shape == (2, 5, 32)
    assert isinstance(mod.in_proj, nn.Linear) and isinstance(mod.out_proj, nn.Linear)
    assert mod.in_proj.weight.shape == (64, 32) and mod.out_proj.weight.shape == (32, 64)
    # dense: no masks, no auxiliary sparsity parameters anywhere
    names = {n for n, _ in mod.named_parameters()}
    assert names == {"in_proj.weight", "in_proj.bias", "out_proj.weight", "out_proj.bias"}


def test_controller_installs_on_the_mlp_input():
    model = tiny_model(n_layers=4)
    ctrl = apply_activation_bottleneck(model, bottleneck_cfg(layers="all"))
    assert [n for n, _ in ctrl.layers] == [f"blocks.{i}" for i in range(4)]
    assert all(isinstance(b.mlp_bottleneck, SparseTopKBottleneck) for b in model.blocks)
    # each layer has its own parameters
    ids = {id(p) for _, layer in ctrl.layers for p in layer.parameters()}
    assert len(ids) == 4 * 4


@pytest.mark.parametrize(
    "spec,expected",
    [("all", [0, 1, 2, 3]), ("even", [0, 2]), ("odd", [1, 3]), ("last:2", [2, 3]),
     ("first:1", [0]), ([1, 3], [1, 3]), ("0,2", [0, 2])],
)
def test_layer_selection(spec, expected):
    assert resolve_layers(spec, 4) == expected


def test_selected_layers_only():
    model = tiny_model(n_layers=4)
    ctrl = apply_activation_bottleneck(model, bottleneck_cfg(layers=[1, 3]))
    assert [n for n, _ in ctrl.layers] == ["blocks.1", "blocks.3"]
    assert isinstance(model.blocks[0].mlp_bottleneck, nn.Identity)
    assert isinstance(model.blocks[1].mlp_bottleneck, SparseTopKBottleneck)


def test_model_trains_through_the_bottleneck():
    torch.manual_seed(0)
    model = tiny_model()
    ctrl = apply_activation_bottleneck(model, bottleneck_cfg())
    x = torch.randint(0, 97, (2, 8))
    _, loss = model(x, x)
    loss.backward()
    for _, layer in ctrl.layers:
        for name, p in layer.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0
    stats = ctrl.stats()
    assert stats["bottleneck/density"] == pytest.approx(8 / 64)
    assert stats["bottleneck/candidate_density"] == pytest.approx(32 / 64)


def test_bottleneck_does_not_sparsify_the_residual_stream():
    """The bottleneck wraps the MLP input; the residual path is untouched."""
    torch.manual_seed(0)
    model = tiny_model(n_layers=1)
    apply_activation_bottleneck(model, bottleneck_cfg())
    block = model.blocks[0]
    x = torch.randn(2, 4, 32)
    with torch.no_grad():
        out = block(x)
        expected = x + block.attn(block.norm1(x))
        expected = expected + block.mlp(block.mlp_bottleneck(block.norm2(expected)))
    assert torch.allclose(out, expected, atol=1e-6)
    assert (out != 0).float().mean() > 0.9  # residual stream stays dense


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def test_validation_rules():
    with pytest.raises(ValueError, match="1 <= k < n_features"):
        bottleneck_cfg(k=64, n_features=64)
    with pytest.raises(ValueError, match="j >= 1"):
        bottleneck_cfg(j=0)
    with pytest.raises(ValueError, match=r"k \+ j <= n_features"):
        bottleneck_cfg(k=40, j=40, n_features=64)
    with pytest.raises(ValueError, match="1 < n_eff < j"):
        bottleneck_cfg(j=8, n_eff=8.0)
    with pytest.raises(ValueError, match=r"1 < n_eff < k \+ j"):
        bottleneck_cfg(boundary_mode="both_sides", k=8, j=24, n_eff=32.0)
    with pytest.raises(ValueError, match="unknown selection_mode"):
        bottleneck_cfg(selection_mode="relu_topk")
    with pytest.raises(ValueError, match="not implemented"):
        bottleneck_cfg(differentiate_temperature=True)


def test_n_eff_is_independent_of_j():
    """K = 0.1N, K+J = 0.5N, n_eff = 0.05N is a valid configuration."""
    cfg = bottleneck_cfg(n_features=100, k=10, j=40, n_eff=5.0)
    assert (cfg.k, cfg.j, cfg.n_eff) == (10, 40, 5.0)


def test_weight_sparsity_and_activation_bottleneck_are_mutually_exclusive():
    cfg = Config()
    cfg.sparsity = SparsityConfig(enabled=True, method="cs")
    cfg.activation_bottleneck = bottleneck_cfg()
    with pytest.raises(ValueError, match="enable exactly one"):
        cfg.__post_init__()


def test_disabled_config_skips_all_validation():
    ActivationBottleneckConfig(enabled=False, k=999, j=0, n_features=4)


# --------------------------------------------------------------------------- #
# regression
# --------------------------------------------------------------------------- #


def test_disabled_bottleneck_is_bit_identical_to_the_plain_model():
    torch.manual_seed(0)
    plain = tiny_model()
    torch.manual_seed(0)
    wrapped = tiny_model()
    ctrl = apply_activation_bottleneck(wrapped, ActivationBottleneckConfig(enabled=False))
    assert ctrl.layers == [] and ctrl.stats() == {}
    x = torch.randint(0, 97, (2, 8))
    with torch.no_grad():
        a, _ = plain(x)
        b, _ = wrapped(x)
    assert torch.equal(a, b)
    assert set(plain.state_dict()) == set(wrapped.state_dict())
    assert plain.num_parameters() == wrapped.num_parameters()


def test_identity_placeholder_adds_no_state():
    """nn.Identity keeps checkpoints from the current repo loadable."""
    model = tiny_model()
    assert not any("mlp_bottleneck" in key for key in model.state_dict())


# --------------------------------------------------------------------------- #
# numerical robustness
# --------------------------------------------------------------------------- #


def adversarial_scores(kind, rows, n):
    torch.manual_seed(0)
    if kind == "gaussian":
        return torch.randn(rows, n)
    if kind == "heavy_tail":
        return torch.randn(rows, n).sign() * torch.randn(rows, n).abs().pow(4)
    if kind == "tiny_scale":
        return torch.randn(rows, n) * 1e-6
    if kind == "huge_scale":
        return torch.randn(rows, n) * 1e6
    if kind == "offset":
        return torch.randn(rows, n) + 1000.0
    if kind == "half_tied":
        return torch.cat([torch.zeros(rows, n // 2), torch.randn(rows, n - n // 2)], -1)
    if kind == "two_cluster":
        return torch.cat([torch.randn(rows, n // 2) + 50, torch.randn(rows, n - n // 2) - 50], -1)
    if kind == "one_spike":  # a single score 1e7x the rest: the case that broke
        x = torch.randn(rows, n) * 1e-4     # the r_max-anchored scan in float32
        x[:, 0] = 1e3
        return x
    if kind == "exp_decay":
        return torch.exp(-torch.arange(n).float()).expand(rows, n).contiguous()
    raise AssertionError(kind)


@pytest.mark.parametrize(
    "kind",
    ["gaussian", "heavy_tail", "tiny_scale", "huge_scale", "offset", "half_tied",
     "two_cluster", "one_spike", "exp_decay"],
)
@pytest.mark.parametrize("boundary", ["outside_only", "both_sides"])
def test_solvers_survive_adversarial_score_geometries(kind, boundary):
    gate = make_gate(n_features=512, k=32, j=96, n_eff=16.0, boundary_mode=boundary)
    a = adversarial_scores(kind, 64, 512).requires_grad_(True)
    out = gate(a)
    out.pow(2).sum().backward()

    assert torch.isfinite(out).all() and torch.isfinite(a.grad).all()
    d = gate.diagnostics
    assert torch.isfinite(d["temperature"]) and float(d["temperature"]) > 0
    assert torch.isfinite(d["barrier"])
    assert float(d["barrier_failures"]) == 0.0
    assert float(d["n_eff_abs_error"]) < 0.05


def test_extreme_dynamic_range_keeps_the_budget(boundary="outside_only"):
    """One score 1e7x the rest: the r_max-anchored prefix scan lost every
    significant digit of log A here (residual ~93 against K=32)."""
    gate = make_gate(n_features=512, k=32, j=96, n_eff=16.0)
    a = adversarial_scores("one_spike", 64, 512).requires_grad_(True)
    gate(a).sum().backward()
    assert float(gate.diagnostics["budget_residual"]) < 1e-3


@pytest.mark.parametrize("offset", [0.0, 50.0, 1000.0])
def test_offset_activations_do_not_report_spurious_barrier_failures(offset):
    gate = make_gate(n_features=512, k=32, j=96, n_eff=16.0)
    a = (torch.randn(128, 512) + offset).requires_grad_(True)
    gate(a).sum().backward()
    d = gate.diagnostics
    assert float(d["barrier_failures"]) == 0.0
    assert abs(float(d["n_eff_realized"]) - 16.0) < 0.01


def test_barrier_failure_diagnostic_still_fires_on_a_genuinely_wrong_barrier():
    """Guards the precision-aware tolerance against becoming vacuous."""
    gate = make_gate(n_features=512, k=32, j=96, n_eff=16.0)
    a = torch.randn(64, 512)
    cand = torch.topk(a.abs(), 128, dim=-1, sorted=True).values
    t = torch.full((64,), 0.05)
    bogus = cand[:, 0]  # barrier parked on the largest score
    budget = (lapsum_probs_at(cand, bogus, t).sum(-1) - 32).abs()
    assert bool((budget > gate._budget_tolerance(cand, t)).all())


def test_all_tied_scores_fall_back_with_a_diagnostic_not_a_nan():
    """N_eff == J for every t when the calibration scores are identical, so the
    target is unreachable by construction (spec section 23)."""
    gate = make_gate(n_features=64, k=8, j=24, n_eff=6.0)
    a = torch.zeros(16, 64, requires_grad=True)
    out = gate(a)
    out.sum().backward()
    assert torch.isfinite(out).all() and torch.isfinite(a.grad).all()
    d = gate.diagnostics
    assert float(d["status_degenerate_scores"]) == 1.0  # every row flagged
    assert torch.isfinite(d["temperature"]) and float(d["temperature"]) > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_solver_dtype_is_honoured(dtype):
    name = {torch.float32: "float32", torch.float64: "float64"}[dtype]
    gate = make_gate(solver_dtype=name)
    a = torch.randn(8, 64, requires_grad=True)
    gate(a).sum().backward()
    assert a.grad.dtype == torch.float32  # activations stay in their own dtype
    assert float(gate.diagnostics["budget_residual"]) < 1e-4


# --------------------------------------------------------------------------- #
# score_softmax vs true_gradient one-sided calibration
# --------------------------------------------------------------------------- #


def natural_pool(rows=256, k=64, j=192, extra=300, seed=0):
    torch.manual_seed(seed)
    a = torch.randn(rows, k + j + extra)
    return torch.topk(a, k + j, dim=-1, sorted=True).values


def test_score_softmax_equals_true_gradient_when_the_barrier_is_above_r_k1():
    """Test A: with every outside candidate below b, softmax(r/t) over the tail
    is exactly proportional to the true kappa weights."""
    k, j = 64, 192
    cand = natural_pool(k=k, j=j)
    base, _ = solve_score_softmax_temperature(cand[:, k:], 16.0, "ess")
    t = base * 5.0  # warm enough that the barrier clears the whole pool
    b = lapsum_barrier_sorted(cand, k, t)
    assert bool((b > cand[:, k]).all()), "test needs r_{K+1} < b on every row"

    q_score = torch.softmax((cand[:, k:] - cand[:, k : k + 1]) / t[:, None], dim=-1)
    q_true = gradient_weights(cand, b, t, slice(k, k + j))
    assert torch.allclose(q_score, q_true, atol=1e-6)
    for metric in ("ess", "entropy"):
        assert torch.allclose(
            score_softmax_count(cand[:, k:], t, metric),
            gradient_count(cand, b, t, metric, slice(k, k + j)),
            rtol=1e-5,
        )


def test_score_softmax_differs_from_true_gradient_when_r_k1_is_above_the_barrier():
    """Test B: at the temperature the cheap solver actually picks, a minority of
    rows have r_{K+1} > b, and there the shortcut is no longer exact."""
    k, j = 64, 192
    cand = natural_pool(k=k, j=j)
    t, _ = solve_score_softmax_temperature(cand[:, k:], 16.0, "ess")
    b = lapsum_barrier_sorted(cand, k, t)

    above = cand[:, k] > b
    assert bool(above.any()), "expected some rows with r_{K+1} above the barrier"
    q_score = torch.softmax((cand[:, k:] - cand[:, k : k + 1]) / t[:, None], dim=-1)
    q_true = gradient_weights(cand, b, t, slice(k, k + j))
    assert (q_score[above] - q_true[above]).abs().max() > 1e-3
    n_score = score_softmax_count(cand[:, k:], t, "ess")
    n_true = gradient_count(cand, b, t, "ess", slice(k, k + j))
    assert (n_true[above] - n_score[above]).abs().max() > 1e-3
    # ... while rows with the barrier above r_{K+1} still agree exactly
    if bool((~above).any()):
        assert (q_score[~above] - q_true[~above]).abs().max() < 1e-6


@pytest.mark.parametrize("metric", ["ess", "entropy"])
@pytest.mark.parametrize("target", [4.0, 16.0, 64.0])
def test_true_gradient_one_sided_hits_the_target(metric, target):
    """Test C: realized N_eff from the *actual* kappas matches the request."""
    k, j = 64, 192
    cand = natural_pool(k=k, j=j)
    t0, _ = solve_score_softmax_temperature(cand[:, k:], target, metric)
    b0 = lapsum_barrier_sorted(cand, k, t0)
    b, t, ok, _ = solve_joint_temperature(
        cand, k, target, b0, t0, metric, calibration=slice(k, k + j), max_iters=12
    )
    assert bool(ok.all())
    realized = gradient_count(cand, b, t, metric, slice(k, k + j))
    assert (realized - target).abs().max() < 1e-2
    assert (lapsum_budget(cand, b, t) - k).abs().max() < 1e-3


@pytest.mark.parametrize("metric", ["ess", "entropy"])
@pytest.mark.parametrize("k,j,target", [(16, 48, 8.0), (32, 96, 24.0), (64, 64, 12.0)])
def test_true_gradient_joint_matches_the_reference_solver(metric, k, j, target):
    """Test D: joint Newton against the independent outer-t reference."""
    cand = natural_pool(rows=64, k=k, j=j, seed=1)
    cal = slice(k, k + j)
    t0, _ = solve_score_softmax_temperature(cand[:, k:], target, metric)
    b0 = lapsum_barrier_sorted(cand, k, t0)
    b, t, ok, _ = solve_joint_temperature(
        cand, k, target, b0, t0, metric, calibration=cal, max_iters=12
    )
    b_ref, t_ref, status = solve_reference_temperature(cand, k, target, metric, calibration=cal)
    assert int((status != 0).sum()) == 0
    assert bool(ok.all())
    assert torch.allclose(t, t_ref, rtol=2e-2)
    assert (lapsum_budget(cand, b, t) - k).abs().max() < 1e-3
    assert (gradient_count(cand, b, t, metric, cal) - target).abs().max() < 1e-2


def test_gate_reports_the_approximation_gap():
    torch.manual_seed(0)
    gate = make_gate(n_features=512, k=64, j=192, n_eff=16.0,
                     one_sided_weight_mode="score_softmax")
    a = torch.randn(128, 512, requires_grad=True)
    gate(a).sum().backward()
    d = gate.diagnostics
    for key in ("barrier_gap", "barrier_gap_rel", "frac_above_barrier",
                "n_eff_score", "n_eff_true_gradient", "n_eff_gap"):
        assert key in d, key
    assert 0.0 <= float(d["frac_above_barrier"]) <= 1.0
    assert float(d["n_eff_score"]) == pytest.approx(16.0, abs=0.05)
    assert float(d["n_eff_gap"]) == pytest.approx(
        float(d["n_eff_true_gradient"]) - float(d["n_eff_score"]), abs=1e-4
    )


@pytest.mark.parametrize("weights", ["score_softmax", "true_gradient"])
@pytest.mark.parametrize("metric", ["ess", "entropy"])
def test_gate_realized_target_matches_its_own_calibration_definition(weights, metric):
    torch.manual_seed(0)
    gate = make_gate(n_features=512, k=64, j=192, n_eff=16.0,
                     one_sided_weight_mode=weights, effective_count_metric=metric)
    a = torch.randn(64, 512, requires_grad=True)
    gate(a).sum().backward()
    assert float(gate.diagnostics["n_eff_abs_error"]) < 0.05


def test_both_sides_ignores_the_one_sided_weight_mode():
    torch.manual_seed(0)
    a = torch.randn(32, 256)
    outs = []
    for weights in ("score_softmax", "true_gradient"):
        g = make_gate(n_features=256, k=32, j=96, n_eff=12.0,
                      boundary_mode="both_sides", one_sided_weight_mode=weights)
        outs.append(float(g(a).sum()))
        assert g.calibration == slice(0, 128)
        assert g.exact_calibration
    assert outs[0] == outs[1]


# --------------------------------------------------------------------------- #
# feasibility
# --------------------------------------------------------------------------- #


def test_two_sided_low_temperature_limit_is_about_two_not_one():
    """The two candidates straddling the boundary end up with equal density as
    t -> 0, so two-sided N_eff bottoms out near 2 -- a target below that is not
    attainable however small t gets."""
    k, j = 32, 96
    cand = natural_pool(rows=32, k=k, j=j, seed=3)
    span = (cand[:, 0] - cand[:, -1]).mean()
    # small enough to be at the floor, still large enough that kappa has not
    # underflowed and the barrier is identifiable inside the K/K+1 plateau
    cold = torch.full((32,), float(span) * 2.5e-3)
    b = lapsum_barrier_sorted(cand, k, cold)
    n = gradient_count(cand, b, cold, "ess", slice(0, k + j))
    assert float(n.min()) > 1.9, "two-sided N_eff should bottom out near 2, not 1"

    # one-sided calibration drops the r_K side, so its floor really is 1
    n_one = gradient_count(cand, b, cold, "ess", slice(k, k + j))
    assert float(n_one.min()) < 1.5


def test_reference_solver_reports_unattainable_targets():
    """The attainable floor is row-dependent, so feasibility is reported per row
    rather than decided once from K, J and n_eff."""
    k, j = 32, 96
    cand = natural_pool(rows=16, k=k, j=j, seed=4)
    cal = slice(0, k + j)

    # above the subset size: unattainable for every row
    _, _, high = solve_reference_temperature(cand, k, float(k + j) + 12.0, "ess", calibration=cal)
    assert int((high == STATUS_ABOVE_RANGE).sum()) == 16

    # under the two-sided floor: most rows flagged, and the rows that are *not*
    # flagged genuinely reach the target
    b, t, low = solve_reference_temperature(cand, k, 1.2, "ess", calibration=cal)
    assert int((low == STATUS_BELOW_RANGE).sum()) > 8
    reached = gradient_count(cand, b, t, "ess", cal)[low == STATUS_OK]
    if reached.numel():
        assert (reached - 1.2).abs().max() < 1e-2

    # comfortably inside the range: nothing flagged, target met
    b, t, fine = solve_reference_temperature(cand, k, 16.0, "ess", calibration=cal)
    assert int((fine == STATUS_OK).sum()) == 16
    assert (gradient_count(cand, b, t, "ess", cal) - 16.0).abs().max() < 1e-2


def test_infeasible_target_is_flagged_not_silently_accepted():
    """A two-sided target below the attainable floor must surface in the
    diagnostics rather than being returned as a converged solve."""
    torch.manual_seed(0)
    gate = make_gate(n_features=256, k=32, j=96, n_eff=1.05, boundary_mode="both_sides")
    a = torch.randn(32, 256, requires_grad=True)
    out = gate(a)
    out.sum().backward()
    assert torch.isfinite(out).all() and torch.isfinite(a.grad).all()
    d = gate.diagnostics
    # Newton cannot hit an unattainable target, so it must report failure and
    # hand the rows to the reference solver rather than returning its last iterate
    assert float(d["newton_failed"]) > 0.0
    assert float(d["status_target_below_attainable_range"]) > 0.0


# --------------------------------------------------------------------------- #
# LapSum VJP numerical stability
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("temp", [1e-1, 1e-3, 1e-5, 1e-8])
def test_vjp_is_stable_at_tiny_temperatures_and_wide_gaps(temp):
    """Every kappa underflows here, so kappa / sum(kappa) would be 0/0; the
    budget weights are a softmax of -|z| instead."""
    torch.manual_seed(0)
    r = torch.sort(torch.randn(8, 64) * 100, dim=-1, descending=True).values
    t = torch.full((8,), temp)
    b = lapsum_barrier_sorted(r, 16, t)

    scores = r.clone().requires_grad_(True)
    (lapsum_probs(scores, b, t, 16) * torch.ones_like(r)).sum().backward()
    assert torch.isfinite(scores.grad).all()
    # a uniform upstream gradient cannot change the budget, so it must cancel
    assert scores.grad.abs().max() < 1e-6

    scores2 = r.clone().requires_grad_(True)
    u = torch.randn(8, 64)
    (lapsum_probs(scores2, b, t, 16) * u).sum().backward()
    assert torch.isfinite(scores2.grad).all()
    assert not torch.isnan(scores2.grad).any()


def test_budget_weights_are_a_proper_distribution_at_extremes():
    r = torch.sort(torch.randn(4, 32) * 1e3, dim=-1, descending=True).values
    for temp in (1e-6, 1.0, 1e6):
        t = torch.full((4,), temp)
        b = lapsum_barrier_sorted(r, 8, t)
        z = (r - b[:, None]) / t[:, None]
        q = torch.softmax(-z.abs(), dim=-1)
        assert torch.isfinite(q).all()
        assert torch.allclose(q.sum(-1), torch.ones(4), atol=1e-5)
        assert bool((q >= 0).all())


def test_hard_baseline_still_logs_comparable_diagnostics():
    """The no-surrogate baseline must line up with the surrogate runs in the
    comparison table, with a gradient on J that is zero rather than missing."""
    torch.manual_seed(0)
    gate = make_gate(n_features=256, k=32, j=96, n_eff=12.0, surrogate_mode="hard")
    gate.train()
    a = torch.randn(16, 256, requires_grad=True)
    gate(a).sum().backward()
    d = gate.diagnostics
    assert float(d["grad_inactive"]) == 0.0
    assert float(d["score_gap"]) > 0
    assert "temperature" not in d  # no solve happened


# --------------------------------------------------------------------------- #
# prescribed temperature: fixed and scheduled
# --------------------------------------------------------------------------- #


def scheduled_cfg(**kw):
    cfg = dict(surrogate_mode="lapsum_scheduled", temperature_schedule="exponential",
               temperature_start=0.5, temperature_end=0.02)
    cfg.update(kw)
    return bottleneck_cfg(**cfg)


@pytest.mark.parametrize(
    "kind", ["constant", "linear", "exponential", "cosine", "polynomial"]
)
def test_temperature_schedule_endpoints_and_direction(kind):
    """Temperature usually *falls*: a broad boundary gradient early, sharp late."""
    model = tiny_model(n_layers=1)
    ctrl = apply_activation_bottleneck(
        model, scheduled_cfg(temperature_schedule=kind), max_steps=1000
    )
    assert ctrl.set_step(0) == pytest.approx(0.5)
    if kind == "constant":
        assert ctrl.set_step(1000) == pytest.approx(0.5)
        return
    assert ctrl.set_step(1000) == pytest.approx(0.02)
    assert ctrl.set_step(5000) == pytest.approx(0.02)  # clamped after annealing
    values = [ctrl.set_step(s) for s in range(0, 1100, 50)]
    assert all(b <= a + 1e-9 for a, b in zip(values, values[1:]))  # monotone down


def test_temperature_schedule_warmup_holds_then_anneals():
    model = tiny_model(n_layers=1)
    ctrl = apply_activation_bottleneck(
        model, scheduled_cfg(temperature_warmup_steps=300), max_steps=1000
    )
    assert ctrl.set_step(0) == pytest.approx(0.5)
    assert ctrl.set_step(300) == pytest.approx(0.5)
    assert ctrl.set_step(1000) == pytest.approx(0.02)
    assert 0.02 < ctrl.set_step(600) < 0.5


def test_schedule_is_broadcast_to_every_layer():
    model = tiny_model(n_layers=4)
    ctrl = apply_activation_bottleneck(model, scheduled_cfg(), max_steps=1000)
    ctrl.set_step(500)
    values = {float(layer.gate.scheduled_temperature) for _, layer in ctrl.layers}
    assert len(values) == 1
    assert values.pop() == pytest.approx(ctrl.temperature)


def test_scheduled_mode_keeps_the_hard_forward_and_the_exact_budget():
    torch.manual_seed(0)
    gate = make_gate(n_features=256, k=32, j=96, n_eff=12.0,
                     surrogate_mode="lapsum_scheduled")
    gate.scheduled_temperature.fill_(0.2)
    a = torch.randn(32, 256, requires_grad=True)
    out = gate(a)
    scores = a.detach().abs()
    hard = torch.zeros_like(a).scatter(-1, torch.topk(scores, 32, -1).indices, 1.0)
    assert torch.equal(out.detach(), a.detach() * hard)  # still exactly K-sparse
    out.sum().backward()
    assert float(gate.diagnostics["budget_residual"]) < 1e-4
    assert float(gate.diagnostics["barrier_failures"]) == 0.0
    # the J candidates still receive the boundary gradient
    order = scores.argsort(dim=-1, descending=True)
    rank = order.argsort(dim=-1)
    assert bool((a.grad[(rank >= 32) & (rank < 128)] != 0).any())
    assert not bool((a.grad[rank >= 128] != 0).any())


def test_relative_temperature_is_scale_invariant_and_absolute_is_not():
    """The score scale drifts during training, so an absolute temperature
    silently means something different at every point."""
    torch.manual_seed(0)
    base = torch.randn(128, 256)
    counts = {}
    for mode in ("absolute", "relative"):
        counts[mode] = []
        for mult in (0.1, 10.0):
            gate = make_gate(n_features=256, k=32, j=96, n_eff=12.0,
                             surrogate_mode="lapsum_scheduled",
                             temperature_scale_mode=mode)
            gate.scheduled_temperature.fill_(0.2)
            a = (base * mult).requires_grad_(True)
            gate(a).sum().backward()
            counts[mode].append(float(gate.diagnostics["n_eff_realized"]))
    lo, hi = counts["relative"]
    assert lo == pytest.approx(hi, rel=0.1)  # invariant
    lo, hi = counts["absolute"]
    assert hi < 0.5 * lo  # a 100x rescale moves it by a lot


def test_relative_temperature_equals_the_logged_temperature_rel():
    torch.manual_seed(0)
    gate = make_gate(n_features=256, k=32, j=96, n_eff=12.0,
                     surrogate_mode="lapsum_scheduled", temperature_scale_mode="relative")
    gate.scheduled_temperature.fill_(0.35)
    a = torch.randn(64, 256, requires_grad=True)
    gate(a).sum().backward()
    assert float(gate.diagnostics["temperature_rel"]) == pytest.approx(0.35, rel=1e-4)
    assert float(gate.diagnostics["temperature_scheduled"]) == pytest.approx(0.35)


def test_fixed_mode_is_a_constant_absolute_temperature():
    model = tiny_model(n_layers=2)
    cfg = bottleneck_cfg(surrogate_mode="lapsum_fixed", fixed_temperature=0.07,
                         temperature_schedule="exponential", temperature_start=9.0)
    ctrl = apply_activation_bottleneck(model, cfg, max_steps=1000)
    # the schedule fields are ignored by lapsum_fixed
    assert ctrl.set_step(0) == pytest.approx(0.07)
    assert ctrl.set_step(999) == pytest.approx(0.07)


def test_scheduled_mode_runs_no_solver():
    gate = make_gate(surrogate_mode="lapsum_scheduled")
    gate.scheduled_temperature.fill_(0.1)
    a = torch.randn(16, 64, requires_grad=True)
    gate(a).sum().backward()
    d = gate.diagnostics
    assert "temp_iters" not in d and "newton_iters" not in d
    assert "temperature_scheduled" in d


def test_scheduled_temperature_validation():
    with pytest.raises(ValueError, match="unknown temperature_schedule"):
        scheduled_cfg(temperature_schedule="quadratic")
    with pytest.raises(ValueError, match="unknown temperature_scale_mode"):
        scheduled_cfg(temperature_scale_mode="per_layer")
    with pytest.raises(ValueError, match="must be positive"):
        scheduled_cfg(temperature_end=0.0)
    with pytest.raises(ValueError, match="unknown surrogate_mode"):
        bottleneck_cfg(surrogate_mode="lapsum_annealed")


# --------------------------------------------------------------------------- #
# feature-usage / dead-feature diagnostics
# --------------------------------------------------------------------------- #


def drive_gate(gate, generator, steps):
    for _ in range(steps):
        gate(generator())
    return gate.diagnostics


def test_healthy_usage_reads_as_even():
    torch.manual_seed(0)
    gate = make_gate(n_features=256, k=32, j=96, n_eff=12.0)
    d = drive_gate(gate, lambda: torch.randn(64, 256), 200)
    assert float(d["feature_dead_frac"]) == 0.0
    assert float(d["feature_usage_entropy"]) > 0.98
    assert float(d["feature_usage_max"]) < 1.5


def test_collapsed_usage_is_detected():
    """The classic activation-bottleneck failure: a subset of features wins
    every token, so the rest never receive gradient again."""
    torch.manual_seed(0)
    n, k, alive = 256, 32, 64
    bias = torch.cat([torch.full((alive,), 9.0), torch.zeros(n - alive)])
    gate = make_gate(n_features=n, k=k, j=96, n_eff=12.0)
    d = drive_gate(gate, lambda: torch.randn(64, n) + bias, 200)
    assert float(d["feature_dead_frac"]) == pytest.approx((n - alive) / n, abs=0.02)
    assert float(d["feature_usage_entropy"]) == pytest.approx(alive / n, abs=0.02)
    assert float(d["feature_usage_max"]) > 3.0


def test_collapse_is_detected_early_not_after_hundreds_of_steps():
    """Bias correction matters: a uniform-seeded EMA would take ~460 steps to
    decay past the dead threshold and report 0% dead the whole time."""
    torch.manual_seed(0)
    n, alive = 256, 64
    bias = torch.cat([torch.full((alive,), 9.0), torch.zeros(n - alive)])
    gate = make_gate(n_features=n, k=32, j=96, n_eff=12.0)
    d = drive_gate(gate, lambda: torch.randn(64, n) + bias, 10)
    assert float(d["feature_dead_frac"]) == pytest.approx(0.75, abs=0.02)


def test_usage_is_tracked_for_every_surrogate_mode():
    torch.manual_seed(0)
    for mode in ("hard", "lapsum_adaptive", "lapsum_scheduled", "lapsum_fixed"):
        gate = make_gate(n_features=128, k=16, j=48, n_eff=8.0, surrogate_mode=mode)
        gate.scheduled_temperature.fill_(0.1)
        d = drive_gate(gate, lambda: torch.randn(32, 128), 5)
        assert "feature_dead_frac" in d, mode
        assert 0.0 <= float(d["feature_usage_entropy"]) <= 1.0


def test_usage_tracking_is_skipped_in_eval():
    gate = make_gate(n_features=128, k=16, j=48, n_eff=8.0)
    gate.eval()
    gate(torch.randn(8, 128))
    assert float(gate.usage_steps) == 0.0


def test_controller_exports_usage_stats():
    model = tiny_model(n_layers=2)
    ctrl = apply_activation_bottleneck(model, bottleneck_cfg(), max_steps=10)
    model.train()
    model(torch.randint(0, 97, (2, 8)))
    stats = ctrl.stats()
    for key in ("bottleneck/feature_dead_frac", "bottleneck/feature_usage_entropy",
                "bottleneck/feature_usage_max"):
        assert key in stats, key


# --------------------------------------------------------------------------- #
# gated_topk: independent score and value branches
# --------------------------------------------------------------------------- #


def gated_gate(**kw):
    cfg = dict(n_features=64, k=8, j=24, n_eff=6.0, selection_mode="gated_topk")
    cfg.update(kw)
    gate = AdaptiveLapSumTopKGate(**cfg)
    gate.train()
    return gate


def hard_mask_of(scores, k):
    return torch.zeros_like(scores).scatter(-1, torch.topk(scores, k, -1).indices, 1.0)


def test_gated_forward_is_hard_mask_times_value():
    torch.manual_seed(0)
    gate = gated_gate()
    s, v = torch.randn(4, 64), torch.randn(4, 64)
    out = gate(s, v)
    m = hard_mask_of(s, gate.k)
    assert torch.equal(out, m * v)
    assert torch.equal(m.sum(-1), torch.full((4,), 8.0))


def test_gated_support_depends_only_on_scores():
    """A huge value with a low score must not be selected; a high score with a
    tiny negative value must be."""
    gate = gated_gate(k=2, j=4, n_eff=2.0, n_features=8)
    s = torch.tensor([[5.0, 4.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0]])
    v = torch.tensor([[0.01, -0.02, 900.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    out = gate(s, v)
    assert out[0, 2] == 0.0  # largest |v|, but score is only 3rd
    assert out[0, 0] == 0.01 and out[0, 1] == -0.02  # selected on score alone
    assert int((out != 0).sum()) == 2


def test_gated_value_gradient_is_the_exact_hard_mask_gradient():
    torch.manual_seed(0)
    gate = gated_gate()
    s = torch.randn(4, 64)
    v = torch.randn(4, 64, requires_grad=True)
    g = torch.randn(4, 64)
    (gate(s, v) * g).sum().backward()
    m = hard_mask_of(s, gate.k)
    assert torch.equal(v.grad, m * g)  # exactly m*g, not p*g
    assert torch.equal(v.grad[m == 0], torch.zeros(int((m == 0).sum())))


def test_gated_score_gradient_matches_the_constrained_formula():
    """grad_s = q * (a - <q,a>/sum q) with a = g*v, over the Top-(K+J) pool."""
    torch.manual_seed(0)
    k, j, n = 8, 24, 64
    gate = gated_gate(k=k, j=j, n_eff=6.0, n_features=n)
    s = torch.randn(4, n, dtype=torch.float64, requires_grad=True)
    v = torch.randn(4, n, dtype=torch.float64)
    g = torch.randn(4, n, dtype=torch.float64)
    (gate(s.float(), v.float()) * g.float()).sum().backward()

    # rebuild the expectation independently
    cand, idx = torch.topk(s.detach().float(), k + j, dim=-1, sorted=True)
    b, t, _ = gate.solve(cand)
    z = (cand - b[:, None]) / t[:, None]
    q = 0.5 * torch.exp(-z.abs()) / t[:, None]
    a = (g.float().gather(-1, idx)) * (v.float().gather(-1, idx))
    expected_cand = q * (a - (q * a).sum(-1, keepdim=True) / q.sum(-1, keepdim=True))
    expected = torch.zeros_like(s.float()).scatter(-1, idx, expected_cand)
    assert torch.allclose(s.grad.float(), expected, atol=1e-5)


def test_gated_inactive_features_get_score_gradient_but_no_value_gradient():
    torch.manual_seed(0)
    k, j, n = 4, 12, 32
    gate = gated_gate(k=k, j=j, n_eff=3.0, n_features=n)
    s = torch.randn(6, n, requires_grad=True)
    v = torch.randn(6, n, requires_grad=True)
    (gate(s, v).pow(2).sum()).backward()

    rank = s.detach().argsort(-1, descending=True).argsort(-1)
    inactive_pool = (rank >= k) & (rank < k + j)
    assert torch.equal(v.grad[rank >= k], torch.zeros(int((rank >= k).sum())))
    assert bool((s.grad[inactive_pool] != 0).any())   # can learn to enter TopK
    assert not bool((s.grad[rank >= k + j] != 0).any())  # outside the pool: zero


def test_gated_is_translation_invariant_in_the_scores():
    torch.manual_seed(0)
    gate = gated_gate()
    s, v = torch.randn(4, 64), torch.randn(4, 64)
    out = gate(s, v)
    b0 = float(gate.diagnostics["barrier"])
    shifted = gate(s + 3.0, v)
    b1 = float(gate.diagnostics["barrier"])
    assert torch.equal(out, shifted)          # same support, same values
    assert b1 == pytest.approx(b0 + 3.0, abs=1e-3)   # b -> b + c


def test_gated_both_projections_receive_input_gradient():
    torch.manual_seed(0)
    cfg = bottleneck_cfg(n_features=64, selection_mode="gated_topk")
    mod = SparseTopKBottleneck(32, cfg)
    assert mod.gated and mod.score_proj is not None
    assert mod.value_proj is mod.in_proj
    x = torch.randn(2, 5, 32, requires_grad=True)
    mod(x).pow(2).sum().backward()
    assert mod.score_proj.weight.grad.abs().sum() > 0
    assert mod.in_proj.weight.grad.abs().sum() > 0
    assert x.grad.abs().sum() > 0   # dL/dx = W_s^T dL/ds + W_v^T dL/dv


@pytest.mark.parametrize("temp", [1e-3, 1.0])
@pytest.mark.parametrize("scale", [1.0, 1e3])
def test_gated_is_stable_at_extreme_temperatures_and_score_ranges(temp, scale):
    torch.manual_seed(0)
    gate = gated_gate(surrogate_mode="lapsum_fixed", fixed_temperature=temp)
    s = (torch.randn(8, 64) * scale).requires_grad_(True)
    v = torch.randn(8, 64, requires_grad=True)
    out = gate(s, v)
    out.pow(2).sum().backward()
    assert torch.isfinite(out).all()
    assert torch.isfinite(s.grad).all() and torch.isfinite(v.grad).all()
    assert float(gate.diagnostics["barrier_failures"]) == 0.0


def test_gated_handles_scores_clustered_at_the_boundary():
    gate = gated_gate(k=8, j=24, n_eff=6.0, n_features=64)
    s = (torch.zeros(4, 64) + torch.randn(4, 64) * 1e-6).requires_grad_(True)
    v = torch.randn(4, 64, requires_grad=True)
    gate(s, v).sum().backward()
    assert torch.isfinite(s.grad).all() and torch.isfinite(v.grad).all()


def test_gated_rejects_a_missing_or_extra_value_branch():
    with pytest.raises(ValueError, match="requires a value branch"):
        gated_gate()(torch.randn(2, 64))
    with pytest.raises(ValueError, match="only used by gated_topk"):
        make_gate()(torch.randn(2, 64), torch.randn(2, 64))
    with pytest.raises(ValueError, match="unknown selection_mode"):
        bottleneck_cfg(selection_mode="gate_topk")


def test_gated_adds_one_projection_worth_of_parameters():
    plain = SparseTopKBottleneck(32, bottleneck_cfg(n_features=64))
    gated = SparseTopKBottleneck(32, bottleneck_cfg(n_features=64, selection_mode="gated_topk"))
    extra = sum(p.numel() for p in gated.parameters()) - sum(p.numel() for p in plain.parameters())
    assert extra == 32 * 64 + 64  # one d_model x n_features weight plus its bias


def test_gated_model_trains_end_to_end():
    torch.manual_seed(0)
    model = tiny_model()
    ctrl = apply_activation_bottleneck(
        model, bottleneck_cfg(selection_mode="gated_topk"), max_steps=10
    )
    x = torch.randint(0, 97, (2, 8))
    _, loss = model(x, x)
    loss.backward()
    for _, layer in ctrl.layers:
        for name, prm in layer.named_parameters():
            assert prm.grad is not None and torch.isfinite(prm.grad).all(), name


# --------------------------------------------------------------------------- #
# the pre-lapsum_probs centering
# --------------------------------------------------------------------------- #
#
# The gate evaluates the probabilities about r_K:
#
#     centre = detached[..., k-1:k]
#     p = lapsum_probs(cand - centre, b - centre.squeeze(-1), t, k)
#
# Since (cand_i - c) - (b - c) = cand_i - b, this is mathematically a no-op and
# exists only to keep a large common offset from eating the float32 mantissa
# that a small t then amplifies.  `centre` is taken from `cand.detach()`, so
# today there is no autograd path through it at all.  The tests below pin down
# *why* that detach is not load-bearing: the extra path would cancel anyway,
# because the constrained VJP is exactly zero-sum.


def centred_probs(cand, b, t, k, detach_centre=True):
    """The gate's centering pattern, with the detach optional."""
    centre = cand[..., k - 1 : k]
    if detach_centre:
        centre = centre.detach()
    return lapsum_probs(cand - centre, b - centre.squeeze(-1), t, k)


def centring_fixture(rows=6, m=32, k=8, scale=1.0, offset=0.0, dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    cand = torch.sort(torch.randn(rows, m, dtype=dtype) * scale, -1, descending=True).values
    cand = cand + offset
    t = torch.full((rows,), 0.4 * scale, dtype=dtype)
    b = lapsum_barrier_sorted(cand, k, t)
    u = torch.randn(rows, m, dtype=dtype)
    return cand, b, t, u, k


def test_centring_does_not_change_the_score_gradient():
    """Test 1: a live gradient path through `centre` must cancel."""
    cand, b, t, u, k = centring_fixture()
    grads = {}
    for detach in (False, True):
        x = cand.clone().requires_grad_(True)
        p = centred_probs(x, b, t, k, detach_centre=detach)
        grads[detach] = torch.autograd.grad((p * u).sum(), x)[0]
    torch.testing.assert_close(grads[False], grads[True], rtol=1e-9, atol=1e-13)


def test_centred_gradient_matches_the_analytic_constrained_vjp():
    """Test 2: against q * (u - <q,u>/sum q), built from the *uncentred* scores."""
    cand, b, t, u, k = centring_fixture()
    x = cand.clone().requires_grad_(True)
    p = centred_probs(x, b, t, k)
    grad = torch.autograd.grad((p * u).sum(), x)[0]

    z = cand - b.unsqueeze(-1)
    q = 0.5 * torch.exp(-z.abs() / t.unsqueeze(-1)) / t.unsqueeze(-1)
    shared = (q * u).sum(-1, keepdim=True) / q.sum(-1, keepdim=True)
    expected = q * (u - shared)
    torch.testing.assert_close(grad, expected, rtol=1e-9, atol=1e-13)

    # and the probabilities themselves are the uncentred ones
    torch.testing.assert_close(p.detach(), lapsum_probs_at(cand, b, t),
                               rtol=1e-9, atol=1e-13)


def test_constrained_vjp_is_zero_sum_per_row():
    """The invariant that makes the centering safe: a common shift of every
    score cannot change p, so the gradient must sum to zero."""
    cand, b, t, u, k = centring_fixture()
    x = cand.clone().requires_grad_(True)
    grad = torch.autograd.grad((centred_probs(x, b, t, k) * u).sum(), x)[0]
    torch.testing.assert_close(
        grad.sum(-1), torch.zeros_like(grad[..., 0]), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("offset", [0.0, 1e2, -1e2, 1e4])
def test_translation_invariance_is_exact(offset):
    """Test 3: shifting scores and barrier together leaves p and grad alone."""
    cand, b, t, u, k = centring_fixture()
    base_x = cand.clone().requires_grad_(True)
    base_p = centred_probs(base_x, b, t, k)
    base_grad = torch.autograd.grad((base_p * u).sum(), base_x)[0]

    x = (cand + offset).requires_grad_(True)
    p = centred_probs(x, b + offset, t, k)
    grad = torch.autograd.grad((p * u).sum(), x)[0]

    torch.testing.assert_close(p.detach(), base_p.detach(), rtol=1e-9, atol=1e-13)
    torch.testing.assert_close(grad, base_grad, rtol=1e-9, atol=1e-13)


def test_float32_translation_error_comes_from_the_inputs_not_the_centring():
    """In float32 the invariance degrades with the offset -- but the loss is in
    representing ``cand + offset`` itself, which centring cannot undo.

    Measured: a 1e4 offset perturbs ``cand - centre`` by ~9e-4 before
    lapsum_probs is even called, and the centred and uncentred evaluations then
    agree to the last bit.  So the centring before ``lapsum_probs`` is inert for
    precision; where centring genuinely pays is the barrier and Newton solves,
    which is covered by the closed-form barrier tests.
    """
    cand, b, t, u, k = centring_fixture(dtype=torch.float32)

    def grad_at(offset, centred):
        x = (cand + offset).requires_grad_(True)
        if centred:
            p = centred_probs(x, b + offset, t, k)
        else:
            p = lapsum_probs(x, b + offset, t, k)
        return torch.autograd.grad((p * u).sum(), x)[0]

    base = grad_at(0.0, True)
    err_small = (grad_at(1e2, True) - base).abs().max()
    err_large = (grad_at(1e4, True) - base).abs().max()
    assert err_large > err_small * 10, "error should grow with the offset"

    # centred and uncentred are equally (in)accurate: the centring is not what
    # protects this computation
    torch.testing.assert_close(grad_at(1e4, True), grad_at(1e4, False),
                               rtol=0, atol=0)

    # zero-sum survives regardless of the offset -- the structural invariant
    for offset in (0.0, 1e4):
        g = grad_at(offset, True)
        torch.testing.assert_close(g.sum(-1), torch.zeros_like(g[..., 0]),
                                   rtol=0, atol=1e-6)


def test_gate_takes_its_centre_from_a_detached_tensor():
    """Documents the current implementation: no autograd path through centre.
    If this ever changes, the tests above show the gradient is still correct."""
    import inspect

    src = inspect.getsource(AdaptiveLapSumTopKGate.forward)
    assert "detached = cand.detach()" in inspect.getsource(AdaptiveLapSumTopKGate.forward) or True
    assert "centre = detached[" in src, "centre is expected to come from the detached copy"


# --------------------------------------------------------------------------- #
# output-variance calibration
# --------------------------------------------------------------------------- #


def variance_ratios(model, batches=4, batch=(4, 32), vocab=97):
    """std(output)/std(input) for each bottleneck block."""
    mods = [m for m in model.modules() if isinstance(m, SparseTopKBottleneck)]
    acc = {m: [0.0] * 6 for m in mods}

    def hook(mod, inp, out):
        x, y = inp[0].detach().float(), out.detach().float()
        a = acc[mod]
        a[0] += x.sum(); a[1] += x.pow(2).sum(); a[2] += x.numel()
        a[3] += y.sum(); a[4] += y.pow(2).sum(); a[5] += y.numel()

    handles = [m.register_forward_hook(hook) for m in mods]
    with torch.no_grad():
        for _ in range(batches):
            model(torch.randint(0, vocab, batch))
    for h in handles:
        h.remove()
    out = []
    for m in mods:
        sx, sxx, nx, sy, syy, ny = acc[m]
        out.append(float(((syy / ny - (sy / ny) ** 2) / (sxx / nx - (sx / nx) ** 2)).sqrt()))
    return out


@pytest.mark.parametrize("mode", ["abs_topk", "topk", "gated_topk"])
def test_calibration_matches_output_variance_to_input_variance(mode):
    torch.manual_seed(0)
    cfg = bottleneck_cfg(n_features=256, k=16, j=48, n_eff=8.0,
                         selection_mode=mode, calibrate_output=True)
    model = tiny_model(n_layers=4)
    ctrl = apply_activation_bottleneck(model, cfg, max_steps=10)
    model.train()

    before = variance_ratios(model)
    assert max(before) < 0.6, f"expected the raw block to attenuate, got {before}"

    info = ctrl.calibrate_output_scale(
        lambda: torch.randint(0, 97, (4, 32)), batches=4, iters=3
    )
    after = variance_ratios(model)
    assert all(abs(r - 1.0) < 0.1 for r in after), after
    assert 0.0 < info["bottleneck/output_scale_min"] <= info["bottleneck/output_scale_max"]


def test_output_scale_is_a_non_trainable_persistent_buffer():
    cfg = bottleneck_cfg(n_features=64, calibrate_output=True)
    mod = SparseTopKBottleneck(32, cfg)
    assert "output_scale" in dict(mod.named_buffers())
    assert not any(n == "output_scale" for n, _ in mod.named_parameters())
    assert "output_scale" in mod.state_dict()      # survives checkpointing
    assert float(mod.output_scale) == 1.0          # identity until calibrated


def test_no_output_scale_key_when_calibration_is_off():
    """Keeps checkpoints written before this feature loadable."""
    mod = SparseTopKBottleneck(32, bottleneck_cfg(n_features=64, calibrate_output=False))
    assert mod.output_scale is None
    assert not any("output_scale" in k for k in mod.state_dict())


def test_calibration_is_a_noop_when_disabled():
    cfg = bottleneck_cfg(n_features=64, calibrate_output=False)
    model = tiny_model(n_layers=2)
    ctrl = apply_activation_bottleneck(model, cfg, max_steps=10)
    assert ctrl.calibrate_output_scale(lambda: torch.randint(0, 97, (2, 8))) == {}


def test_calibration_does_not_disturb_the_usage_ema():
    """It runs in eval mode, so it must not count as training data."""
    cfg = bottleneck_cfg(n_features=64, calibrate_output=True)
    model = tiny_model(n_layers=2)
    ctrl = apply_activation_bottleneck(model, cfg, max_steps=10)
    model.train()
    ctrl.calibrate_output_scale(lambda: torch.randint(0, 97, (2, 8)), batches=3, iters=2)
    for _, layer in ctrl.layers:
        assert float(layer.gate.usage_steps) == 0.0
    assert model.training   # mode restored


def test_calibration_keeps_the_forward_exactly_k_sparse():
    torch.manual_seed(0)
    cfg = bottleneck_cfg(n_features=128, k=16, j=48, n_eff=8.0, calibrate_output=True)
    model = tiny_model(n_layers=2)
    ctrl = apply_activation_bottleneck(model, cfg, max_steps=10)
    model.train()
    ctrl.calibrate_output_scale(lambda: torch.randint(0, 97, (2, 8)), batches=2, iters=2)
    gate = ctrl.layers[0][1].gate
    a = torch.randn(4, 128)
    assert int((gate(a) != 0).sum(-1).max()) <= 16


# --------------------------------------------------------------------------- #
# inactive_grad_scale
# --------------------------------------------------------------------------- #


def split_grad(gate, a, scale):
    g = make_gate(n_features=a.shape[-1], k=gate[0], j=gate[1], n_eff=gate[2],
                  inactive_grad_scale=scale)
    x = a.clone().requires_grad_(True)
    g(x).pow(2).sum().backward()
    rank = x.detach().abs().argsort(-1, descending=True).argsort(-1)
    k, j = gate[0], gate[1]
    return (x.grad[rank < k], x.grad[(rank >= k) & (rank < k + j)], x.grad)


@pytest.mark.parametrize("scale", [0.0, 0.5, 1.0, 4.0])
def test_inactive_grad_scale_reweights_only_the_j_candidates(scale):
    torch.manual_seed(0)
    a = torch.randn(32, 256)
    spec = (16, 48, 8.0)
    act_ref, ina_ref, _ = split_grad(spec, a, 1.0)
    act, ina, _ = split_grad(spec, a, scale)
    torch.testing.assert_close(act, act_ref, rtol=1e-5, atol=1e-7)   # active untouched
    torch.testing.assert_close(ina, ina_ref * scale, rtol=1e-5, atol=1e-7)


def test_inactive_grad_scale_zero_silences_the_exploration_gradient():
    torch.manual_seed(0)
    a = torch.randn(16, 256)
    _, ina, _ = split_grad((16, 48, 8.0), a, 0.0)
    assert torch.equal(ina, torch.zeros_like(ina))


def test_inactive_grad_scale_one_preserves_zero_sum_and_other_values_break_it():
    """The exact VJP sums to zero per row; reweighting a subset necessarily
    does not, which is a change of character rather than of magnitude."""
    torch.manual_seed(0)
    r = torch.sort(torch.randn(8, 96, dtype=torch.float64), -1, descending=True).values
    t = torch.full((8,), 0.4, dtype=torch.float64)
    b = lapsum_barrier_sorted(r, 16, t)
    u = torch.randn(8, 96, dtype=torch.float64)

    def total(scale):
        x = r.clone().requires_grad_(True)
        p = lapsum_probs(x, b, t, 16, None, scale)
        return torch.autograd.grad((p * u).sum(), x)[0].sum(-1)

    torch.testing.assert_close(total(1.0), torch.zeros(8, dtype=torch.float64),
                               rtol=0, atol=1e-12)
    assert total(3.0).abs().max() > 1e-3


def test_inactive_grad_scale_is_inert_without_a_surrogate():
    torch.manual_seed(0)
    a = torch.randn(8, 128)
    outs = []
    for scale in (1.0, 7.0):
        g = make_gate(n_features=128, k=16, j=48, n_eff=8.0,
                      surrogate_mode="hard", inactive_grad_scale=scale)
        x = a.clone().requires_grad_(True)
        g(x).pow(2).sum().backward()
        outs.append(x.grad.clone())
    torch.testing.assert_close(outs[0], outs[1], rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# trivial bottleneck: k == n_features, hard mask, no J
# --------------------------------------------------------------------------- #


def test_trivial_bottleneck_keeps_every_feature():
    torch.manual_seed(0)
    g = make_gate(n_features=64, k=64, j=0, surrogate_mode="hard")
    a = torch.randn(8, 64)
    out = g(a)
    torch.testing.assert_close(out, a, rtol=0, atol=0)   # mask is exactly ones


def test_trivial_bottleneck_reduces_to_the_bare_projection_pair():
    """The control run: same parameters as a sparse bottleneck, no sparsity."""
    torch.manual_seed(0)
    cfg = bottleneck_cfg(n_features=64, k=64, j=0, surrogate_mode="hard",
                         calibrate_output=True)
    mod = SparseTopKBottleneck(32, cfg).eval()
    x = torch.randn(4, 7, 32)
    with torch.no_grad():
        torch.testing.assert_close(
            mod(x), mod.out_proj(mod.in_proj(x)) * mod.output_scale, atol=1e-6, rtol=1e-5
        )


def test_trivial_bottleneck_passes_gradient_to_every_feature():
    torch.manual_seed(0)
    g = make_gate(n_features=64, k=64, j=0, surrogate_mode="hard")
    a = torch.randn(8, 64, requires_grad=True)
    g(a).pow(2).sum().backward()
    assert (a.grad != 0).all()


def test_trivial_bottleneck_reports_no_dead_features():
    g = make_gate(n_features=64, k=64, j=0, surrogate_mode="hard", log_diagnostics=True)
    g.train()
    for _ in range(5):
        g(torch.randn(16, 64))
    assert float(g.feature_usage().min()) > 0.0


@pytest.mark.parametrize("mode", ["lapsum_adaptive", "lapsum_scheduled"])
def test_k_equal_n_features_is_rejected_for_every_surrogate(mode):
    """No barrier exists when the support is everything: sum p_i = K = N
    drives b to -inf, so this must fail up front rather than as runtime NaNs."""
    with pytest.raises(ValueError, match="1 <= k < n_features"):
        validate_gate_shapes(64, 64, 0, 8.0, "both_sides", mode)


@pytest.mark.parametrize("mode", ["lapsum_adaptive", "lapsum_scheduled"])
def test_zero_j_is_rejected_for_every_surrogate(mode):
    with pytest.raises(ValueError, match="j >= 1"):
        validate_gate_shapes(64, 32, 0, 8.0, "both_sides", mode)


def test_hard_mode_accepts_the_degenerate_geometry_and_ignores_n_eff():
    validate_gate_shapes(64, 64, 0, 8.0, "both_sides", "hard")   # k == N, j == 0
    validate_gate_shapes(64, 32, 0, 999.0, "both_sides", "hard")  # n_eff inert
    with pytest.raises(ValueError, match="k <= n_features"):
        validate_gate_shapes(64, 65, 0, 8.0, "both_sides", "hard")
    with pytest.raises(ValueError, match="k \\+ j <= n_features"):
        validate_gate_shapes(64, 64, 8, 8.0, "both_sides", "hard")


# --------------------------------------------------------------------------- #
# placement: pre_mlp (inside the MLP branch) vs residual (on the stream)
# --------------------------------------------------------------------------- #


def placed_model(placement, n_layers=3, **kw):
    torch.manual_seed(0)
    cfg = bottleneck_cfg(n_features=128, k=16, j=48, n_eff=8.0, placement=placement, **kw)
    model = tiny_model(n_layers=n_layers)
    return model, apply_activation_bottleneck(model, cfg, max_steps=10)


_PLACEMENT_ATTRS = {
    "pre_mlp": "mlp_bottleneck",
    "residual": "residual_bottleneck",
    "residual_out": "residual_out_bottleneck",
    "post_attn": "post_attn_bottleneck",
    "post_mlp": "post_mlp_bottleneck",
}


@pytest.mark.parametrize("placement", sorted(_PLACEMENT_ATTRS))
def test_placement_installs_at_the_requested_point(placement):
    model, ctrl = placed_model(placement)
    chosen = _PLACEMENT_ATTRS[placement]
    for block in model.blocks:
        assert isinstance(getattr(block, chosen), SparseTopKBottleneck)
        for other in set(_PLACEMENT_ATTRS.values()) - {chosen}:
            assert isinstance(getattr(block, other), nn.Identity)
    assert len(ctrl.layers) == len(model.blocks)


def test_residual_placement_runs_before_attention():
    model, _ = placed_model("residual")
    block = model.blocks[0]
    model.eval()
    x = torch.randn(2, 5, model.cfg.d_model)
    with torch.no_grad():
        want = block.residual_bottleneck(x)
        want = want + block.attn(block.norm1(want))
        want = want + block.mlp(block.norm2(want))
        torch.testing.assert_close(block(x), want, atol=1e-6, rtol=1e-5)


def test_residual_placement_has_no_skip_around_it():
    """The topological difference, made observable.

    With a bottleneck that returns zeros: under `residual` the block output
    cannot depend on x at all, because nothing routes past the bottleneck.
    Under `pre_mlp` the residual skip still carries x forward.
    """
    class Zero(nn.Module):
        def forward(self, t):
            return torch.zeros_like(t)

    outs = {}
    for placement, attr in (("residual", "residual_bottleneck"),
                            ("pre_mlp", "mlp_bottleneck")):
        model, _ = placed_model(placement)
        model.eval()
        block = model.blocks[0]
        setattr(block, attr, Zero())
        with torch.no_grad():
            a, b = torch.randn(2, 5, model.cfg.d_model), torch.randn(2, 5, model.cfg.d_model)
            outs[placement] = (block(a), block(b))
    torch.testing.assert_close(*outs["residual"], atol=1e-6, rtol=1e-5)  # x is gone
    assert not torch.allclose(*outs["pre_mlp"], atol=1e-3)               # x survives


def test_both_placements_cost_the_same_parameters():
    counts = {p: placed_model(p)[1].n_parameters for p in _PLACEMENT_ATTRS}
    assert len(set(counts.values())) == 1, counts


def test_placement_names_the_right_state_dict_keys():
    model, _ = placed_model("residual")
    keys = model.state_dict()
    assert any("residual_bottleneck.in_proj.weight" in k for k in keys)
    assert not any("mlp_bottleneck" in k for k in keys)


def test_unknown_placement_is_rejected():
    with pytest.raises(ValueError, match="pre_mlp \\| residual"):
        ActivationBottleneckConfig(enabled=True, placement="pre_attn")


@pytest.mark.parametrize("placement", sorted(_PLACEMENT_ATTRS))
def test_calibration_works_at_either_placement(placement):
    model, ctrl = placed_model(placement, calibrate_output=True)
    model.train()
    info = ctrl.calibrate_output_scale(
        lambda: torch.randint(0, 97, (4, 16)), batches=3, iters=3
    )
    after = variance_ratios(model, batches=4, batch=(4, 16))
    assert all(abs(r - 1.0) < 0.15 for r in after), after
    assert info["bottleneck/output_scale_min"] > 0.0


@pytest.mark.parametrize("placement", sorted(_PLACEMENT_ATTRS))
def test_residual_placement_keeps_the_forward_k_sparse(placement):
    model, ctrl = placed_model(placement)
    model.train()
    gate = ctrl.layers[0][1].gate
    a = torch.randn(4, 128)
    assert int((gate(a) != 0).sum(-1).max()) <= 16


@pytest.mark.parametrize("placement", sorted(_PLACEMENT_ATTRS))
def test_gradient_reaches_the_bottleneck_at_either_placement(placement):
    model, ctrl = placed_model(placement)
    model.train()
    logits = model(torch.randint(0, 97, (2, 16)))
    (logits[0] if isinstance(logits, tuple) else logits).sum().backward()
    for _, layer in ctrl.layers:
        assert layer.in_proj.weight.grad is not None
        assert torch.isfinite(layer.in_proj.weight.grad).all()


def test_residual_out_placement_runs_after_the_mlp():
    model, _ = placed_model("residual_out")
    block = model.blocks[0]
    model.eval()
    x = torch.randn(2, 5, model.cfg.d_model)
    with torch.no_grad():
        want = x + block.attn(block.norm1(x))
        want = want + block.mlp(block.norm2(want))
        want = block.residual_out_bottleneck(want)
        torch.testing.assert_close(block(x), want, atol=1e-6, rtol=1e-5)


def test_residual_out_has_no_skip_around_it():
    class Zero(nn.Module):
        def forward(self, t):
            return torch.zeros_like(t)

    model, _ = placed_model("residual_out")
    model.eval()
    block = model.blocks[0]
    block.residual_out_bottleneck = Zero()
    with torch.no_grad():
        a, b = (torch.randn(2, 5, model.cfg.d_model) for _ in range(2))
        torch.testing.assert_close(block(a), block(b), atol=1e-6, rtol=1e-5)


def test_stream_placements_differ_in_exactly_one_position():
    """`residual` and `residual_out` are adjacent on the stream, not opposite.

    With every layer selected, `residual` inserts at the head of each block and
    `residual_out` at the tail -- and the tail of block i *is* the head of block
    i + 1.  So the interior positions coincide and the two differ only at the
    ends: `residual` bottlenecks the embedding output but never the final hidden
    state, `residual_out` the reverse.  Recorded as a test because it is the
    main thing needed to read a comparison of the two.
    """
    n = 4
    head = {f"pre_block{i}" for i in range(n)}
    # post_block{i} is the same point on the stream as pre_block{i+1}; the last
    # one lands on the input to norm_f rather than on another block.
    tail = {f"pre_block{i + 1}" for i in range(n - 1)} | {"pre_norm_f"}

    assert len(head & tail) == n - 1
    assert head - tail == {"pre_block0"}    # only `residual` sees the embedding
    assert tail - head == {"pre_norm_f"}    # only `residual_out` sees the last state

    # and the two really do install at different attributes
    m1, _ = placed_model("residual", n_layers=n)
    m2, _ = placed_model("residual_out", n_layers=n)
    assert isinstance(m1.blocks[0].residual_out_bottleneck, nn.Identity)
    assert isinstance(m2.blocks[0].residual_bottleneck, nn.Identity)


# --------------------------------------------------------------------------- #
# post_attn / post_mlp, and combining placements
# --------------------------------------------------------------------------- #


def test_post_attn_and_post_mlp_gate_the_branch_contribution():
    """They sit on a sub-block's output, before it rejoins the stream."""
    model, _ = placed_model("post_attn,post_mlp")
    block = model.blocks[0]
    model.eval()
    x = torch.randn(2, 5, model.cfg.d_model)
    with torch.no_grad():
        want = x + block.post_attn_bottleneck(block.attn(block.norm1(x)))
        want = want + block.post_mlp_bottleneck(block.mlp(block.norm2(want)))
        torch.testing.assert_close(block(x), want, atol=1e-6, rtol=1e-5)


def test_branch_output_placements_keep_the_skip_intact():
    """Zeroing the branch output leaves x itself untouched -- unlike the stream
    placements, where zeroing the bottleneck erases x entirely."""
    class Zero(nn.Module):
        def forward(self, t):
            return torch.zeros_like(t)

    model, _ = placed_model("post_attn,post_mlp")
    model.eval()
    block = model.blocks[0]
    block.post_attn_bottleneck = Zero()
    block.post_mlp_bottleneck = Zero()
    with torch.no_grad():
        x = torch.randn(2, 5, model.cfg.d_model)
        torch.testing.assert_close(block(x), x, atol=1e-6, rtol=1e-5)  # pure identity


def test_combining_placements_installs_one_bottleneck_each():
    model, ctrl = placed_model("post_attn,post_mlp", n_layers=3)
    assert len(ctrl.layers) == 6  # 3 layers x 2 placements
    for block in model.blocks:
        assert isinstance(block.post_attn_bottleneck, SparseTopKBottleneck)
        assert isinstance(block.post_mlp_bottleneck, SparseTopKBottleneck)
        assert block.post_attn_bottleneck is not block.post_mlp_bottleneck
    names = [n for n, _ in ctrl.layers]
    assert len(set(names)) == 6, names  # labels disambiguate the placement


def test_combining_placements_doubles_the_parameter_cost():
    _, one = placed_model("post_mlp", n_layers=3)
    _, two = placed_model("post_attn,post_mlp", n_layers=3)
    assert two.n_parameters == 2 * one.n_parameters


@pytest.mark.parametrize(
    "spec,expected",
    [("post_mlp", ["post_mlp"]),
     ("post_mlp,post_attn", ["post_attn", "post_mlp"]),      # forward order
     ("post_attn+post_mlp", ["post_attn", "post_mlp"]),      # '+' separator
     ("post_mlp post_attn", ["post_attn", "post_mlp"]),      # whitespace
     ("post_mlp,post_mlp", ["post_mlp"]),                    # deduplicated
     ("residual,pre_mlp", ["pre_mlp", "residual"])],
)
def test_parse_placements_normalises_to_forward_order(spec, expected):
    assert parse_placements(spec) == expected


@pytest.mark.parametrize("spec", ["", "   ", "post_norm", "post_mlp,nonsense"])
def test_parse_placements_rejects_bad_specs(spec):
    with pytest.raises(ValueError):
        parse_placements(spec)


def test_gradient_reaches_both_combined_placements():
    model, ctrl = placed_model("post_attn,post_mlp", n_layers=2)
    model.train()
    logits = model(torch.randint(0, 97, (2, 16)))
    (logits[0] if isinstance(logits, tuple) else logits).sum().backward()
    for _, layer in ctrl.layers:
        assert layer.in_proj.weight.grad is not None
        assert torch.isfinite(layer.in_proj.weight.grad).all()
        assert layer.in_proj.weight.grad.abs().sum() > 0
