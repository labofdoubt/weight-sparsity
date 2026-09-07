"""swap_gibbs surrogate: hard TopK forward, one-swap Gibbs backward.

The backward is verified against the differentiable reference
``hard_mask + p - p.detach()`` built from :func:`swap_probs` -- the custom VJP
must reproduce plain autograd through the soft marginals exactly.  The hard
TopK discontinuity itself is never finite-differenced.
"""

import math

import pytest
import torch

from wsparse.bottleneck import (
    AdaptiveLapSumTopKGate,
    SparseTopKBottleneck,
    apply_activation_bottleneck,
    swap_gibbs_mask,
    swap_log_rho,
    swap_probs,
    swap_weights,
)
from wsparse.config import ActivationBottleneckConfig, ModelConfig
from wsparse.model import build_model


def sorted_scores(rows=32, m=48, scale=1.0, dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    r = torch.randn(rows, m, dtype=dtype) * scale
    return torch.sort(r, dim=-1, descending=True).values


def make_gate(temperature=0.3, **kw):
    cfg = dict(
        n_features=64, k=8, j=24, n_eff=6.0,
        surrogate_mode="swap_gibbs",
        temperature_scale_mode="absolute",
        solver_dtype="float64",
    )
    cfg.update(kw)
    gate = AdaptiveLapSumTopKGate(**cfg)
    gate.scheduled_temperature.fill_(temperature)
    gate.train()
    return gate


def bottleneck_cfg(**kw):
    cfg = dict(
        enabled=True, n_features=64, k=8, j=24, n_eff=6.0, layers="all",
        surrogate_mode="swap_gibbs", temperature_scale_mode="absolute",
    )
    cfg.update(kw)
    return ActivationBottleneckConfig(**cfg)


def tiny_model(**kw):
    cfg = dict(vocab_size=97, max_seq_len=32, n_layers=2, d_model=32, n_heads=4)
    cfg.update(kw)
    return build_model(ModelConfig(**cfg))


def reference_mask(scores, gate):
    """``hard + p - p.detach()`` with plain autograd through swap_probs."""
    cand_scores, cand_idx = torch.topk(scores, gate.m, dim=-1, largest=True, sorted=True)
    hard = torch.zeros_like(scores).scatter(-1, cand_idx[..., : gate.k], 1.0)
    cand = cand_scores.to(gate.solver_dtype)
    t = gate.prescribed_temperature(cand.detach())
    p = swap_probs(cand, t, gate.k, gate.swap_log_rho)
    p_full = (
        torch.zeros_like(scores, dtype=p.dtype).scatter(-1, cand_idx, p).to(scores.dtype)
    )
    return hard + gate.surrogate_grad_scale * (p_full - p_full.detach())


# --------------------------------------------------------------------------- #
# forward
# --------------------------------------------------------------------------- #


def test_forward_is_exactly_hard_topk():
    torch.manual_seed(0)
    a = torch.randn(4, 5, 64, dtype=torch.float64)
    for mode in ("topk", "abs_topk"):
        swap = make_gate(selection_mode=mode)
        hard = make_gate(selection_mode=mode, surrogate_mode="hard")
        assert torch.equal(swap(a).detach(), hard(a).detach())
        out = swap(a)
        assert int((out != 0).sum(-1).max()) <= swap.k
    # eval mode never even builds the surrogate
    gate = make_gate()
    gate.eval()
    hard = make_gate(surrogate_mode="hard")
    hard.eval()
    assert torch.equal(gate(a), hard(a))


def test_forward_survives_exact_score_ties():
    gate = make_gate(k=4, j=8, n_features=32, selection_mode="topk")
    hard = make_gate(k=4, j=8, n_features=32, selection_mode="topk", surrogate_mode="hard")
    a = torch.zeros(3, 32, dtype=torch.float64)
    a[0, :6] = 1.0  # ties across the K/K+1 boundary; other rows fully tied
    x = a.clone().requires_grad_(True)
    out = gate(x)
    assert torch.equal(out.detach(), hard(a))  # standard torch.topk tie behaviour
    out.pow(2).sum().backward()
    assert torch.isfinite(x.grad).all()


# --------------------------------------------------------------------------- #
# the exact VJP against the differentiable reference
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["topk", "abs_topk"])
@pytest.mark.parametrize("lam", ["default", 0.25])
def test_custom_vjp_matches_the_reference_soft_marginals(mode, lam):
    torch.manual_seed(1)
    gate = make_gate(selection_mode=mode, swap_lambda=lam)
    a = torch.randn(6, 64, dtype=torch.float64)
    u = torch.randn(6, 64, dtype=torch.float64)

    x = a.clone().requires_grad_(True)
    (gate(x) * u).sum().backward()

    x_ref = a.clone().requires_grad_(True)
    scores = gate.scores_of(x_ref)
    value = x_ref
    (value * reference_mask(scores, gate) * u).sum().backward()

    assert torch.allclose(x.grad, x_ref.grad, atol=1e-12, rtol=1e-9)


def test_gated_score_branch_gets_the_pure_swap_gradient():
    torch.manual_seed(2)
    gate = make_gate(selection_mode="gated_topk")
    s = torch.randn(5, 64, dtype=torch.float64, requires_grad=True)
    v = torch.randn(5, 64, dtype=torch.float64, requires_grad=True)
    u = torch.randn(5, 64, dtype=torch.float64)
    (gate(s, v) * u).sum().backward()

    s_ref = s.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    (v_ref * reference_mask(s_ref, gate) * u).sum().backward()
    assert torch.allclose(s.grad, s_ref.grad, atol=1e-12, rtol=1e-9)
    assert torch.allclose(v.grad, v_ref.grad, atol=1e-12, rtol=1e-9)
    # the support-selection gradient is zero-sum per row: every surrogate
    # support has exactly K active features, so total mass cannot change
    assert s.grad.sum(-1).abs().max() < 1e-12


def test_complete_bottleneck_matches_the_reference():
    torch.manual_seed(3)
    cfg = bottleneck_cfg(n_features=48, k=6, j=12)
    bn = SparseTopKBottleneck(24, cfg)
    bn.train()
    bn.gate.scheduled_temperature.fill_(0.4)
    x = torch.randn(3, 7, 24, requires_grad=True)
    u = torch.randn(3, 7, 24)
    (bn(x) * u).sum().backward()
    grads = (x.grad.clone(), bn.in_proj.weight.grad.clone(), bn.out_proj.weight.grad.clone())

    bn.zero_grad()
    x_ref = x.detach().clone().requires_grad_(True)
    value = bn.in_proj(x_ref)
    y_ref = bn.out_proj(value * reference_mask(bn.gate.scores_of(value), bn.gate))
    (y_ref * u).sum().backward()
    assert torch.allclose(grads[0], x_ref.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(grads[1], bn.in_proj.weight.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(grads[2], bn.out_proj.weight.grad, atol=1e-6, rtol=1e-5)


def test_no_kj_state_is_saved_for_backward():
    """Everything saved for backward is O(rows * (K + J)), never rows * K * J."""
    k, j, rows = 64, 64, 4
    gate = make_gate(n_features=256, k=k, j=j)
    a = torch.randn(rows, 256, dtype=torch.float64, requires_grad=True)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: saved.append(t.numel()) or t, lambda t: t
    ):
        out = gate(a)
    out.pow(2).sum().backward()
    assert max(saved) <= rows * 256  # full-width activations, fine
    assert max(saved) < rows * k * j  # the K x J swap matrix, never


# --------------------------------------------------------------------------- #
# the soft marginals
# --------------------------------------------------------------------------- #


def test_soft_marginals_conserve_mass():
    r = sorted_scores(rows=16, m=30)
    t = torch.full((16,), 0.5, dtype=torch.float64)
    for lam in ("default", 0.7, 0.01):
        p = swap_probs(r, t, 10, swap_log_rho(lam, 10, 20))
        assert torch.allclose(p.sum(-1), torch.full((16,), 10.0, dtype=torch.float64))
        assert bool((p >= 0).all()) and bool((p <= 1).all())


def test_score_gradients_are_zero_sum():
    r = sorted_scores(rows=8, m=24).requires_grad_(True)
    t = torch.full((8,), 0.3, dtype=torch.float64)
    m = swap_gibbs_mask(r, t, 6, swap_log_rho("default", 6, 18))
    (m * torch.randn(8, 24, dtype=torch.float64)).sum().backward()
    assert r.grad.sum(-1).abs().max() < 1e-12


def test_score_gradients_are_shift_invariant():
    r = sorted_scores(rows=8, m=24)
    t = torch.full((8,), 0.3, dtype=torch.float64)
    u = torch.randn(8, 24, dtype=torch.float64)
    grads = []
    for shift in (0.0, 137.0):
        x = (r + shift).requires_grad_(True)
        (swap_gibbs_mask(x, t, 6, 0.0) * u).sum().backward()
        grads.append(x.grad)
    assert torch.allclose(grads[0], grads[1], atol=1e-9)


# --------------------------------------------------------------------------- #
# lambda
# --------------------------------------------------------------------------- #


def test_default_lambda_is_rho_one_and_matches_kj_over_kj_plus_one():
    k, j = 8, 24
    assert swap_log_rho("default", k, j) == 0.0
    lam = (k * j) / (k * j + 1)
    assert abs(swap_log_rho(lam, k, j)) < 1e-12
    r = sorted_scores(rows=6, m=k + j)
    t = torch.full((6,), 0.4, dtype=torch.float64)
    assert torch.allclose(
        swap_probs(r, t, k, swap_log_rho("default", k, j)),
        swap_probs(r, t, k, swap_log_rho(lam, k, j)),
        atol=1e-12,
    )


def test_lambda_zero_gives_exactly_the_hard_gradient():
    torch.manual_seed(4)
    a = torch.randn(4, 64, dtype=torch.float64)
    grads = {}
    for kw in (dict(swap_lambda=0.0), dict(surrogate_mode="hard")):
        x = a.clone().requires_grad_(True)
        make_gate(**kw)(x).pow(2).sum().backward()
        grads[str(kw)] = x.grad
    a_grad, b_grad = grads.values()
    assert torch.equal(a_grad, b_grad)


def test_high_temperature_swap_probability_approaches_lambda():
    r = sorted_scores(rows=8, m=32)
    k, j, lam = 8, 24, 0.35
    _, _, big = swap_weights(r, torch.full((8,), 1e8, dtype=torch.float64), k,
                             swap_log_rho(lam, k, j))
    assert torch.allclose(big, torch.full((8,), lam, dtype=torch.float64), atol=1e-6)
    # and rho = 1 reproduces lambda_default = KJ / (KJ + 1)
    _, _, dflt = swap_weights(r, torch.full((8,), 1e8, dtype=torch.float64), k, 0.0)
    lam_default = (k * j) / (k * j + 1)
    assert torch.allclose(dflt, torch.full((8,), lam_default, dtype=torch.float64), atol=1e-6)


def test_swap_probability_is_bounded_by_lambda():
    k, j, lam = 8, 24, 0.35
    log_rho = swap_log_rho(lam, k, j)
    for seed in range(4):
        r = sorted_scores(rows=16, m=k + j, seed=seed)
        for temp in (1e-3, 0.1, 1.0, 10.0, 1e4):
            _, _, rr = swap_weights(r, torch.full((16,), temp, dtype=torch.float64),
                                    k, log_rho)
            assert float(rr.max()) <= lam + 1e-12


def test_low_temperature_swap_probability_vanishes():
    r = sorted_scores(rows=8, m=32)  # continuous scores: strict gaps a.s.
    gap = r[..., 7] - r[..., 8]  # the K/K+1 gap dominates: W <= K J e^{-gap/t}
    _, _, rr = swap_weights(r, gap / 60.0, 8, 0.0)
    assert float(rr.max()) < 1e-12


def test_swap_lambda_validation():
    for bad in (-0.1, 1.0, 1.5, "adaptive"):
        with pytest.raises(ValueError, match="swap_lambda"):
            swap_log_rho(bad, 8, 24)
        with pytest.raises(ValueError, match="swap_lambda"):
            make_gate(swap_lambda=bad)
        with pytest.raises(ValueError, match="swap_lambda"):
            bottleneck_cfg(swap_lambda=bad)
    assert swap_log_rho(0.0, 8, 24) == float("-inf")


def test_lapsum_only_knobs_are_rejected_not_ignored():
    with pytest.raises(ValueError, match="inactive_grad_scale"):
        bottleneck_cfg(inactive_grad_scale=2.0)
    with pytest.raises(ValueError, match="project_scale_gradient"):
        bottleneck_cfg(project_scale_gradient=True, temperature_scale_mode="relative")


# --------------------------------------------------------------------------- #
# batch shapes and dtypes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_low_precision_scores_compute_in_the_solver_dtype(dtype):
    torch.manual_seed(5)
    gate = make_gate(solver_dtype="float32")
    a = torch.randn(2, 3, 64).to(dtype).requires_grad_(True)
    out = gate(a)
    # same pool as the gate: low precision ties, and topk(k) may break them
    # differently from the first k of topk(k+j)
    idx = torch.topk(a.detach().abs(), gate.m, dim=-1, sorted=True).indices[..., : gate.k]
    hard = torch.zeros_like(a).scatter(-1, idx, torch.ones_like(a))
    assert torch.equal(out, a.detach() * hard)
    out.float().pow(2).sum().backward()
    assert a.grad.dtype == dtype
    assert torch.isfinite(a.grad.float()).all()


def test_arbitrary_leading_batch_dimensions():
    gate = make_gate()
    a = torch.randn(2, 3, 4, 64, dtype=torch.float64, requires_grad=True)
    out = gate(a)
    assert out.shape == a.shape
    out.pow(2).sum().backward()
    assert torch.isfinite(a.grad).all()


# --------------------------------------------------------------------------- #
# temperature: shared machinery, not a parallel path
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scale_mode", ["relative", "absolute"])
def test_effective_temperature_equals_the_scheduled_lapsum_one(scale_mode):
    """Same config, same step -> the two prescribed modes resolve the same T."""
    torch.manual_seed(6)
    a = torch.randn(4, 5, 64, dtype=torch.float64)
    temps = {}
    for mode in ("swap_gibbs", "lapsum_scheduled"):
        gate = make_gate(surrogate_mode=mode, temperature_scale_mode=scale_mode,
                         temperature=0.27)
        gate(a)
        temps[mode] = gate.diagnostics["temperature"]
    assert torch.allclose(temps["swap_gibbs"], temps["lapsum_scheduled"], atol=1e-12)


def test_controller_schedule_drives_the_swap_temperature_identically():
    kw = dict(temperature_schedule="exponential", temperature_start=0.5,
              temperature_end=0.02, temperature_anneal_steps=100)
    ctrls = {
        mode: apply_activation_bottleneck(
            tiny_model(), bottleneck_cfg(surrogate_mode=mode, **kw), max_steps=100
        )
        for mode in ("swap_gibbs", "lapsum_scheduled")
    }
    for step in (0, 25, 50, 100):
        values = {mode: ctrl.set_step(step) for mode, ctrl in ctrls.items()}
        assert values["swap_gibbs"] == values["lapsum_scheduled"]
        assert values["swap_gibbs"] == pytest.approx(ctrls["swap_gibbs"].schedule(step))
        for _, layer in ctrls["swap_gibbs"].layers:
            assert float(layer.gate.scheduled_temperature) == pytest.approx(values["swap_gibbs"])


def test_swap_reads_t_through_the_shared_prescribed_helper(monkeypatch):
    """Overriding the one shared helper must move the swap surrogate with it --
    there is no second temperature source to fall back on."""
    gate = make_gate()
    a = torch.randn(3, 64, dtype=torch.float64)
    monkeypatch.setattr(
        gate, "prescribed_temperature",
        lambda candidates: torch.full_like(candidates[..., 0], 0.777),
    )
    gate(a)
    assert float(gate.diagnostics["temperature"]) == pytest.approx(0.777)


def test_backward_uses_the_temperature_saved_at_forward_time():
    torch.manual_seed(7)
    a = torch.randn(3, 64, dtype=torch.float64)
    u = torch.randn(3, 64, dtype=torch.float64)

    gate = make_gate(temperature=0.3)
    x = a.clone().requires_grad_(True)
    out = gate(x)
    gate.scheduled_temperature.fill_(0.03)  # schedule moves between fwd and bwd
    (out * u).sum().backward()

    ref = make_gate(temperature=0.3)  # schedule never moves
    x_ref = a.clone().requires_grad_(True)
    (ref(x_ref) * u).sum().backward()
    assert torch.equal(x.grad, x_ref.grad)


# --------------------------------------------------------------------------- #
# diagnostics and repr
# --------------------------------------------------------------------------- #


def test_diagnostics_expose_temperature_and_swap_probability():
    gate = make_gate()
    a = torch.randn(4, 64, dtype=torch.float64, requires_grad=True)
    gate(a).pow(2).sum().backward()
    diag = gate.diagnostics
    for key in ("temperature", "temperature_rel", "temperature_scheduled",
                "swap_R", "swap_R_median", "swap_R_max", "score_gap", "score_span",
                "grad_active", "grad_inactive"):
        assert key in diag, key
    assert 0.0 <= float(diag["swap_R"]) <= 1.0
    assert float(diag["swap_R_max"]) <= 1.0
    assert "swap_gibbs" in repr(gate) and "swap_lambda=default" in repr(gate)


def test_controller_reports_the_temperature_target():
    ctrl = apply_activation_bottleneck(tiny_model(), bottleneck_cfg(), max_steps=10)
    ctrl.set_step(0)
    x = torch.randint(0, 97, (2, 16))
    ctrl.model.train()
    ctrl.model(x)
    stats = ctrl.stats()
    assert "bottleneck/temperature_target" in stats
    assert "bottleneck/swap_R" in stats


def test_n_eff_is_inert_for_swap_gibbs():
    # out of the LapSum-legal range on purpose: swap_gibbs never reads it
    make_gate(n_eff=0.5)
    make_gate(n_eff=1e9)
