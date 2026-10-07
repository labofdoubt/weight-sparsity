"""surrogate_mode="laplace_policy": sampled exactly-K supports over the clean
Top(K+J) pool, trained with the likelihood-ratio estimator of the sampled CE.

The estimator tests are Monte Carlo against analytic expectations (the
two-candidate derivative, an enumerated two-gate expected loss) with
tolerances set from the measured standard error, never fixed-noise finite
differences of the hard forward.
"""

import copy
import json
import math
import os
import weakref

import numpy as np
import pytest
import torch
import yaml

from wsparse.bottleneck import apply_activation_bottleneck
from wsparse.bottleneck.gate import AdaptiveLapSumTopKGate
from wsparse.bottleneck.laplace_policy import (
    PolicyCollector, PolicyForwardSettings, PolicyTrainingState, effective_width,
    laplace_log_density, make_generator, policy_backward_loss, policy_forward,
    sample_laplace, scheduled_tau, sequence_costs, support_multiplier)
from wsparse.bottleneck.module import effective_backward_support
from wsparse.config import (ActivationBottleneckConfig, Config, ModelConfig, TrainConfig,
                            config_from_dict, load_config)
from wsparse.model import LossDetails, build_model
from wsparse.train import deterministic_probe, evaluate_stochastic, load_for_inference, train

CPU = torch.device("cpu")
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
VOCAB = 64


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def laplace_diff_pdf(d: float, t: float) -> float:
    """Density of eps_1 - eps_2 for iid Laplace(0, t)."""
    a = abs(d) / t
    return (1.0 + a) / (4.0 * t) * math.exp(-a)


def laplace_diff_cdf(d, t):
    """P(eps_1 - eps_2 <= d); tensor or float."""
    if torch.is_tensor(d):
        a = d.abs() / t
        tail = 0.5 * (1 + a / 2) * torch.exp(-a)
        return torch.where(d >= 0, 1 - tail, tail)
    a = abs(d) / t
    tail = 0.5 * (1 + a / 2) * math.exp(-a)
    return 1 - tail if d >= 0 else tail


def make_gate(**kw):
    cfg = dict(n_features=64, k=8, j=24, surrogate_mode="laplace_policy", temperature=0.5)
    cfg.update(kw)
    gate = AdaptiveLapSumTopKGate(**cfg)
    gate.train()
    return gate


def sampled(gate, z, seed=0, collector=None, keep_samples=False, sample=True):
    """One forward with sampling and (optionally) a collector; returns (y, collector)."""
    col = collector if collector is not None else PolicyCollector(keep_samples=keep_samples)
    gen = make_generator(CPU, seed)
    with policy_forward([gate], PolicyForwardSettings(sample=sample, collector=col,
                                                       generator=gen)):
        y = gate(z)
    return y, col


def clean_topk_mask(z, gate):
    scores = gate.scores_of(z.detach())
    idx = torch.topk(scores, gate.k, dim=-1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter(-1, idx, True)


def pool_mask(z, gate):
    scores = gate.scores_of(z.detach())
    idx = torch.topk(scores, gate.m, dim=-1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter(-1, idx, True)


def make_fake_dataset(tmp_path, n_train=60_000, n_val=8_000):
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for name, n in (("train", n_train), ("val", n_val)):
        arr = np.memmap(d / f"{name}.bin", dtype=np.uint16, mode="w+", shape=(n,))
        arr[:] = rng.integers(0, VOCAB, size=n).astype(np.uint16)
        arr.flush()
        del arr
    with open(d / "meta.json", "w") as f:
        json.dump({"vocab_size": VOCAB, "eos_id": 0, "tokenizer": "fake",
                   "train_tokens": n_train}, f)
    return str(d)


def smoke_config(data_dir, out_dir, run_name="pol", **bn):
    cfg = Config()
    cfg.data.data_dir = data_dir
    cfg.data.seq_len = 32
    cfg.model = ModelConfig(vocab_size=VOCAB, max_seq_len=32, n_layers=2, d_model=32, n_heads=4)
    cfg.train.batch_size = 8
    cfg.train.micro_batch_size = 4
    cfg.train.max_steps = 6
    cfg.train.warmup_steps = 2
    cfg.train.log_every_steps = 2
    cfg.train.validate_every_steps = 3
    cfg.train.val_batches = 2
    cfg.train.checkpoint_every_steps = 6
    cfg.train.dtype = "float32"
    cfg.train.device = "cpu"
    cfg.train.tensorboard = False
    cfg.train.out_dir = out_dir
    cfg.train.run_name = run_name
    kw = dict(enabled=True, n_features=128, k=16, j=48, layers="all", placement="residual_out",
              surrogate_mode="laplace_policy", temperature=0.5)
    kw.update(bn)
    cfg.activation_bottleneck = ActivationBottleneckConfig(**kw)
    return cfg


def policy_model(n_layers=3, **bn):
    torch.manual_seed(0)
    kw = dict(enabled=True, n_features=64, k=8, j=8, layers="all", placement="residual_out",
              surrogate_mode="laplace_policy", temperature=0.5)
    kw.update(bn)
    cfg = ActivationBottleneckConfig(**kw)
    model = build_model(ModelConfig(vocab_size=61, max_seq_len=16, n_layers=n_layers,
                                    d_model=32, n_heads=4))
    ctl = apply_activation_bottleneck(model, cfg)
    return model, ctl


# --------------------------------------------------------------------------- #
# 11.1 forward and local autodiff
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("temperature", [0.05, 1.0, 100.0])
def test_exactly_k_sampled_members_inside_the_clean_pool(temperature):
    gate = make_gate(temperature=temperature)
    torch.manual_seed(0)
    z = torch.randn(500, 64)
    y, _ = sampled(gate, z)
    chosen = y != 0
    assert torch.equal(chosen.sum(-1), torch.full((500,), 8))        # exactly K
    assert bool((chosen <= pool_mask(z, gate)).all())                # inside Top(K+J)
    assert float(gate.diagnostics["active_count"]) == 8.0
    exchanged = (chosen & ~clean_topk_mask(z, gate)).sum()
    if temperature >= 1.0:
        assert int(exchanged) > 0                                     # noise does exchange
    assert float(gate.diagnostics["policy_exchange_frac"]) == pytest.approx(
        float(exchanged) / (500 * 8))


def test_transmitted_values_are_the_original_signed_values():
    gate = make_gate(temperature=2.0)
    torch.manual_seed(1)
    z = torch.randn(64, 64) * 3
    y, _ = sampled(gate, z)
    chosen = y != 0
    assert torch.equal(y[chosen], z[chosen])
    assert bool((y[~chosen] == 0).all())


def test_noise_off_output_is_ordinary_hard_topk_with_the_same_tie_convention():
    for mode in ("abs_topk", "topk"):
        gate = make_gate(selection_mode=mode)
        hard = AdaptiveLapSumTopKGate(n_features=64, k=8, j=24, surrogate_mode="hard",
                                      selection_mode=mode)
        torch.manual_seed(2)
        z = torch.randn(16, 64)
        z[:, :20] = torch.tensor([1.0, -1.0, 0.0, 0.5] * 5)  # positive, negative, zero, ties
        gate.eval()
        hard.eval()
        with torch.no_grad():
            assert torch.equal(gate(z), hard(z))                       # eval: noise off
        gate.train()
        y, _ = sampled(gate, z, sample=False)                           # explicit override
        assert torch.equal(y, hard(z))


def test_stochastic_eval_under_no_grad_samples_and_default_eval_does_not():
    gate = make_gate(temperature=1.0)
    gate.eval()
    torch.manual_seed(3)
    z = torch.randn(64, 64)
    with torch.no_grad():
        clean = gate(z)
        assert torch.equal(gate(z), clean)
        a, col = sampled(gate, z, seed=1)
        b, _ = sampled(gate, z, seed=2)
    assert not torch.equal(a, clean) and not torch.equal(a, b)
    assert col.records == []                                            # no graph under no_grad
    assert gate._policy_settings is None                                # restored


def test_settings_are_restored_after_an_exception_and_in_nested_calls():
    gate = make_gate()
    outer = PolicyForwardSettings(sample=False)
    inner = PolicyForwardSettings(sample=True)
    with policy_forward([gate], outer):
        assert gate._policy_settings is outer
        with policy_forward([gate], inner):
            assert gate._policy_settings is inner
        assert gate._policy_settings is outer
    assert gate._policy_settings is None
    with pytest.raises(RuntimeError):
        with policy_forward([gate], inner):
            raise RuntimeError("boom")
    assert gate._policy_settings is None


def test_generation_uses_clean_supports():
    model, _ = policy_model()
    model.train()
    idx = torch.randint(0, 61, (2, 6))
    out1 = model.generate(idx, 4, temperature=0.0)
    out2 = model.generate(idx, 4, temperature=0.0)
    assert torch.equal(out1, out2)                                      # no sampling in generate


@pytest.mark.parametrize("center", [False, True])
def test_density_autodiff_is_sign_over_t_and_centering_projects_to_zero_sum(center):
    torch.manual_seed(4)
    t = torch.tensor([[0.3]], dtype=torch.float64)
    s = torch.randn(1, 10, dtype=torch.float64, requires_grad=True)
    r = (s.detach() + sample_laplace(s.shape, t, dtype=torch.float64)).detach()
    u = s - s.mean(-1, keepdim=True) if center else s
    laplace_log_density(r, u, t).sum().backward()
    h = torch.sign(r - u.detach()) / t
    expected = h - h.mean(-1, keepdim=True) if center else h
    assert torch.allclose(s.grad, expected, atol=1e-12)
    if center:
        assert abs(float(s.grad.sum())) < 1e-12


def test_leaving_the_sample_attached_zeroes_the_location_gradient():
    """The regression the detach rule guards against."""
    torch.manual_seed(5)
    t = torch.tensor([[0.5]], dtype=torch.float64)
    s = torch.randn(1, 6, dtype=torch.float64, requires_grad=True)
    eps = sample_laplace(s.shape, t, dtype=torch.float64)
    r_attached = s + eps                       # WRONG: r carries the u dependence
    laplace_log_density(r_attached, s, t).sum().backward()
    assert torch.allclose(s.grad, torch.zeros_like(s))
    # the production gate detaches r and its location gradient is +-1/T
    gate = make_gate(n_features=6, k=2, j=4, temperature=0.5, policy_center_scores=False,
                     selection_mode="topk")
    z = torch.randn(1, 6, dtype=torch.float64, requires_grad=True)
    y, col = sampled(gate, z, keep_samples=True)
    rec = col.records[0]
    assert not rec.r.requires_grad
    rec.log_prob.sum().backward()
    assert torch.allclose(z.grad.abs(), torch.full_like(z, 2.0))        # 1/T = 2


def test_width_gamma_and_advantage_are_detached_but_the_centering_mean_is_not():
    gate = make_gate(n_features=16, k=3, j=5, temperature=0.4, policy_temperature_mode="relative_b",
                     policy_support_scale=0.7, policy_support_scale_mode="effective_temperature",
                     policy_support_temperature_ref=0.2)
    torch.manual_seed(6)
    z = torch.rand(7, 16, dtype=torch.float64) + 0.5
    z.requires_grad_(True)
    y, col = sampled(gate, z, keep_samples=True)
    rec = col.records[0]
    scores = z.detach().abs()
    cand = torch.topk(scores, 8, dim=-1).values
    t_row = (0.4 * cand[:, 3:4]).clamp_min(1e-6)                         # tau * s_(K+1)
    gamma = 0.7 * t_row / 0.2
    rec.log_prob.sum().backward()
    # frozen-width formula: gamma * sign(r - u)/T projected (centering), back
    # through sign(z) -- no dT/ds or dgamma/ds term anywhere
    u = cand - cand.mean(-1, keepdim=True)
    h = torch.sign(rec.r - u) / t_row
    h = h - h.mean(-1, keepdim=True)
    g_scores = torch.zeros_like(scores).scatter(-1, rec.cand_idx, gamma * h)
    assert torch.allclose(z.grad, g_scores * torch.sign(z.detach()), atol=1e-10)
    assert float(gate.diagnostics["policy_gamma_mean"]) == pytest.approx(
        float(gamma.mean()), rel=1e-6)


def test_loss_detaches_advantage_and_baseline_and_keeps_both_paths():
    seq_ce = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    seq_valid = torch.tensor([4, 0, 12])
    log_prob = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64, requires_grad=True)
    w = seq_valid.double() / seq_valid.sum()
    ce = (w * seq_ce).sum()
    loss, support, adv = policy_backward_loss(ce, seq_ce, seq_valid, log_prob, baseline=1.5)
    assert float(loss) == pytest.approx(float(ce))                      # value is the CE
    loss.backward()
    assert torch.allclose(seq_ce.grad, w)                                # no leak through adv
    assert torch.allclose(log_prob.grad, w * (seq_ce.detach() - 1.5))
    assert not adv.requires_grad and not support.requires_grad
    assert float(support) == pytest.approx(float((w * (seq_ce.detach() - 1.5) * log_prob.detach()).sum()))


def test_fixed_geometries_bypass_noise_and_density():
    for kw in (dict(n_features=64, k=8, j=0), dict(n_features=8, k=8, j=0)):
        gate = make_gate(temperature=5.0, **kw)
        hard = AdaptiveLapSumTopKGate(surrogate_mode="hard", **kw)
        assert gate.policy_fixed
        torch.manual_seed(7)
        z = torch.randn(10, kw["n_features"], requires_grad=True)
        zh = z.detach().clone().requires_grad_(True)
        y, col = sampled(gate, z)
        yh = hard(zh)
        assert torch.equal(y, yh) and col.records == []
        w = torch.randn_like(y)
        (y * w).sum().backward()
        (yh * w).sum().backward()
        assert torch.equal(z.grad, zh.grad)


def test_relative_span_needs_two_candidates_and_relative_b_needs_abs_topk():
    with pytest.raises(ValueError, match="relative_span"):
        make_gate(j=1, policy_temperature_mode="relative_span")
    with pytest.raises(ValueError, match="relative_b"):
        make_gate(selection_mode="topk", policy_temperature_mode="relative_b")
    make_gate(j=0, policy_temperature_mode="relative_span")              # fixed: no exchange
    with pytest.raises(ValueError, match="relative_span"):
        ActivationBottleneckConfig(enabled=True, n_features=64, k=8, j=1,
                                   surrogate_mode="laplace_policy",
                                   policy_temperature_mode="relative_span")
    with pytest.raises(ValueError, match="relative_b"):
        ActivationBottleneckConfig(enabled=True, n_features=64, k=8, j=8,
                                   surrogate_mode="laplace_policy", selection_mode="topk",
                                   policy_temperature_mode="relative_b")


def test_laplace_sampler_moments_signs_and_reproducibility():
    t = torch.tensor([[0.7]], dtype=torch.float64)
    gen = make_generator(CPU, 11)
    eps = sample_laplace((1, 400_000), t, generator=gen, dtype=torch.float64)
    assert torch.isfinite(eps).all()
    assert abs(float(eps.mean())) < 0.01
    assert float(eps.var()) == pytest.approx(2 * 0.7 ** 2, rel=0.03)
    assert float((eps > 0).double().mean()) == pytest.approx(0.5, abs=0.005)
    assert float(eps.abs().max()) <= 0.7 * math.log(1.0 / (2 * torch.finfo(torch.float64).eps)) + 1e-6
    gen2 = make_generator(CPU, 11)
    assert torch.equal(sample_laplace((1, 400_000), t, generator=gen2, dtype=torch.float64), eps)
    # float32 endpoints: no log(0)
    e32 = sample_laplace((2000000,), torch.tensor(1.0), generator=make_generator(CPU, 1))
    assert torch.isfinite(e32).all()


def test_nonfinite_scores_are_flagged_not_repaired():
    gate = make_gate(n_features=16, k=3, j=5)
    z = torch.randn(4, 16)
    z[0, 0] = float("nan")
    y, _ = sampled(gate, z)
    assert float(gate.diagnostics["policy_nonfinite"]) == 1.0
    assert not torch.isfinite(y).all()


# --------------------------------------------------------------------------- #
# 11.2 expected-gradient checks
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("delta,temperature,center,baseline", [
    (0.5, 0.5, True, 0.0), (0.5, 0.5, False, 0.3), (1.0, 0.25, True, 0.5), (0.0, 1.0, False, 0.0),
])
def test_two_candidate_policy_gradient_matches_the_analytic_derivative(
        delta, temperature, center, baseline):
    """K=1 of two candidates, costs C_1 / C_2 of the two supports:
    dJ/ds_1 = (C_1 - C_2) (1 + |delta|/T) / (4T) exp(-|delta|/T) = -dJ/ds_2."""
    n = 300_000
    gate = make_gate(n_features=2, k=1, j=1, selection_mode="topk", temperature=temperature,
                     policy_center_scores=center)
    s1, s2 = 1.0 + delta, 1.0                                              # values nonzero
    z = torch.tensor([[s1, s2]], dtype=torch.float64).repeat(n, 1).requires_grad_(True)
    costs = torch.tensor([0.7, -0.3], dtype=torch.float64)
    y, col = sampled(gate, z, seed=0)
    mask = (y != 0).double()
    cost = (mask * costs).sum(-1)                                          # detached support cost
    support = ((cost - baseline) * col.records[0].log_prob).sum() / n
    support.backward()
    per_row = z.grad * n                                                   # one estimate per row
    mean, se = per_row.mean(0), per_row.std(0) / math.sqrt(n)
    exact = float(costs[0] - costs[1]) * laplace_diff_pdf(delta, temperature)
    assert abs(float(mean[0]) - exact) < 5 * float(se[0])
    assert abs(float(mean[1]) + exact) < 5 * float(se[1])
    assert float(mask[:, 0].mean()) == pytest.approx(laplace_diff_cdf(delta, temperature), abs=0.005)


def test_value_and_policy_paths_combine_to_the_expected_loss_gradient():
    """cost = c_i z_i on the selected i (topk: z is both score and value);
    J = F(delta) c_1 s_1 + (1 - F(delta)) c_2 s_2 and the sampled gradient of
    CE + [support - sg(support)] must estimate dJ/ds."""
    n, t = 300_000, 0.5
    gate = make_gate(n_features=2, k=1, j=1, selection_mode="topk", temperature=t)
    s = torch.tensor([1.3, 0.9], dtype=torch.float64)
    c = torch.tensor([0.8, -0.5], dtype=torch.float64)
    z = s.repeat(n, 1).requires_grad_(True)
    y, col = sampled(gate, z, seed=1)
    cost = (y * c).sum(-1)                                                 # differentiable value path
    adv = (cost - 0.1).detach()
    support = (adv * col.records[0].log_prob).sum() / n
    loss = cost.sum() / n + (support - support.detach())
    loss.backward()
    per_row = z.grad * n
    mean, se = per_row.mean(0), per_row.std(0) / math.sqrt(n)
    sp = torch.tensor(s.tolist(), dtype=torch.float64, requires_grad=True)
    p1 = laplace_diff_cdf(sp[0] - sp[1], t)
    J = p1 * c[0] * sp[0] + (1 - p1) * c[1] * sp[1]
    (exact,) = torch.autograd.grad(J, [sp])
    assert bool(((mean - exact).abs() < 5 * se).all()), (mean, exact, se)


def test_relative_mode_gradient_is_the_frozen_width_partial_derivative():
    """abs_topk, relative_b: the gradient must equal the sampled location
    formula at the realized detached width, not the derivative of a width
    that is recomputed from the scores."""
    gate = make_gate(n_features=12, k=2, j=6, temperature=0.3,
                     policy_temperature_mode="relative_b", policy_center_scores=False)
    torch.manual_seed(8)
    z = (torch.rand(5, 12, dtype=torch.float64) + 0.2).requires_grad_(True)
    y, col = sampled(gate, z, keep_samples=True)
    rec = col.records[0]
    rec.log_prob.sum().backward()
    cand = torch.topk(z.detach().abs(), 8, dim=-1).values
    t_row = 0.3 * cand[:, 2:3]
    h = torch.sign(rec.r - cand) / t_row
    frozen = torch.zeros_like(z).scatter(-1, rec.cand_idx, h) * torch.sign(z.detach())
    assert torch.allclose(z.grad, frozen, atol=1e-10)
    # the boundary score s_(K+1) itself gets only its own location term
    # (its contribution through T is absent by construction)
    assert float(gate.diagnostics["policy_t_mean"]) == pytest.approx(float(t_row.mean()), rel=1e-6)


def test_support_rescaling_changes_the_support_path_only():
    torch.manual_seed(9)
    z0 = torch.randn(6, 32, dtype=torch.float64)
    w = torch.randn(6, 32, dtype=torch.float64)

    def run(**kw):
        gate = make_gate(n_features=32, k=4, j=12, temperature=0.5, **kw)
        z = z0.clone().requires_grad_(True)
        y, col = sampled(gate, z, seed=3)                                  # same noise each time
        (y * w).sum().backward()
        value_grad = z.grad.clone()
        z.grad = None
        col.records[0].log_prob.sum().backward()
        return value_grad, z.grad.clone(), y.detach()

    v1, s1, y1 = run(policy_support_scale=1.0)
    v2, s2, y2 = run(policy_support_scale=0.25)
    v0, s0, y0 = run(policy_support_scale=0.0)
    assert torch.equal(y1, y2) and torch.equal(y1, y0)                     # forward unchanged
    assert torch.equal(v1, v2) and torch.equal(v1, v0)                     # value path unchanged
    assert torch.allclose(s2, 0.25 * s1) and bool((s0 == 0).all())        # support path scaled
    # effective_temperature: gamma T/T_ref cancels the sampled 1/T
    for temp in (0.2, 0.8):
        gate = make_gate(n_features=32, k=4, j=12, temperature=temp, policy_center_scores=False,
                         policy_support_scale=1.0, policy_support_scale_mode="effective_temperature",
                         policy_support_temperature_ref=0.4)
        z = z0.clone().requires_grad_(True)
        y, col = sampled(gate, z, seed=3)
        col.records[0].log_prob.sum().backward()
        pool = pool_mask(z, gate)
        assert torch.allclose(z.grad[pool].abs(), torch.full((int(pool.sum()),), 1 / 0.4,
                                                             dtype=torch.float64))
    # the cap binds from above
    g = support_multiplier("effective_temperature", 1.0, torch.tensor([[0.2], [0.8]]), 0.5, 0.4,
                           gamma_max=1.0)
    assert g.flatten().tolist() == pytest.approx([0.5, 1.0])
    assert support_multiplier("scheduled_temperature", 0.1, torch.ones(2, 1), 0.25, 0.5
                              ).flatten().tolist() == pytest.approx([0.05, 0.05])
    assert support_multiplier("constant", 0.3, torch.ones(2, 1), 0.25, 0.5
                              ).flatten().tolist() == pytest.approx([0.3, 0.3])


# --------------------------------------------------------------------------- #
# 11.3 multi-gate and trainer integration
# --------------------------------------------------------------------------- #


def test_two_gate_chain_matches_the_enumerated_expected_loss_and_needs_the_cross_gate_path():
    """Two K=1-of-2 gates in series; the second gate's scores depend on the
    first gate's transmitted value.  The exact gradient of the enumerated
    expected cost (4 support pairs, Laplace-difference CDFs) is matched by the
    sampled estimator; removing the later density term's path into W1 is not."""
    t1, t2 = 0.5, 0.4
    x = torch.tensor([0.9, -0.4, 0.6], dtype=torch.float64)
    W1 = torch.tensor([[0.8, 0.3, -0.2], [0.1, -0.9, 0.5]], dtype=torch.float64,
                      requires_grad=True)
    A = torch.tensor([[1.0, -0.6], [0.5, 1.3]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([0.15, -0.25], dtype=torch.float64, requires_grad=True)
    c = torch.tensor([1.0, -0.8], dtype=torch.float64)
    baseline = 0.2

    z1 = W1 @ x
    p1 = laplace_diff_cdf(z1[0] - z1[1], t1)
    J = 0.0
    for s1, P1 in ((0, p1), (1, 1 - p1)):
        y1 = torch.where(torch.arange(2) == s1, z1, torch.zeros_like(z1))
        z2 = A @ y1 + b
        p2 = laplace_diff_cdf(z2[0] - z2[1], t2)
        J = J + P1 * (p2 * c[0] * z2[0] + (1 - p2) * c[1] * z2[1])
    exact = dict(zip(("W1", "A", "b"), torch.autograd.grad(J, [W1, A, b])))

    g1 = make_gate(n_features=2, k=1, j=1, selection_mode="topk", temperature=t1)
    g2 = make_gate(n_features=2, k=1, j=1, selection_mode="topk", temperature=t2)
    runs, n = 8, 50_000
    est = {"W1": [], "A": [], "b": []}
    without_cross = []
    for r in range(runs):
        gen = make_generator(CPU, 100 + r)
        col = PolicyCollector()
        with policy_forward([g1, g2], PolicyForwardSettings(sample=True, collector=col,
                                                             generator=gen)):
            y1 = g1(x.expand(n, 3) @ W1.t())
            y2 = g2(y1 @ A.t() + b)
        assert len(col.records) == 2 and col.records[0].gate is g1 and col.records[1].gate is g2
        cost = (y2 * c).sum(-1)
        lp1, lp2 = col.records[0].log_prob, col.records[1].log_prob
        adv = (cost - baseline).detach()
        support = (adv * (lp1 + lp2)).mean()
        loss = cost.mean() + (support - support.detach())
        gw, ga, gb = torch.autograd.grad(loss, [W1, A, b], retain_graph=True)
        est["W1"].append(gw); est["A"].append(ga); est["b"].append(gb)
        # the later density term's gradient into the EARLIER parameters,
        # through the sampled hard value y1 -> z2 -> u2
        (cross,) = torch.autograd.grad((adv * lp2).mean(), [W1])
        assert float(cross.abs().max()) > 0
        without_cross.append(gw - cross)
    for name in ("W1", "A", "b"):
        st = torch.stack(est[name])
        mean, se = st.mean(0), st.std(0) / math.sqrt(runs)
        assert bool(((mean - exact[name]).abs() < 5 * se).all()), (name, mean, exact[name], se)
    st = torch.stack(without_cross)
    mean, se = st.mean(0), st.std(0) / math.sqrt(runs)
    assert float(((mean - exact["W1"]).abs() / se).max()) > 10   # detaching the path fails


@pytest.mark.parametrize("kw,expected", [
    (dict(), 3),
    (dict(placement="post_attn,post_mlp"), 6),
    (dict(share_projections=True), 3),
    (dict(tie_encoder_decoder=True, init_mode="unit_norm_dictionary"), 3),
    (dict(code_residual=True, share_projections=True), 3),
    (dict(code_residual=True, share_projections=False, post_norm=True), 3),
])
def test_records_cover_every_gate_instance_without_stale_entries(kw, expected):
    model, ctl = policy_model(n_layers=3, **kw)
    model.train()
    idx = torch.randint(0, 61, (2, 10))
    for _ in range(2):                                                     # second forward: fresh records
        out = model(idx, idx, return_loss_details=True)
        assert isinstance(out, LossDetails)
        assert len(out.policy_records) == expected
        assert len({id(r) for r in out.policy_records}) == expected
        gates = [r.gate for r in out.policy_records]
        assert len(set(map(id, gates))) == expected                       # one record per gate here
        assert set(map(id, gates)) == set(map(id, ctl.policy_gates))
        assert out.policy_log_prob.shape == (2,)
        assert all(r.log_prob.shape == (2, 10) for r in out.policy_records)
        loss, _, _ = policy_backward_loss(out.ce, out.seq_ce, out.seq_valid,
                                          out.policy_log_prob, 4.0)
        loss.backward()
        assert all(g._policy_settings is None for g in ctl.policy_gates)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    if kw.get("code_residual"):
        # the later density terms reach block 0 through the carried code
        first = model.blocks[0].mlp.fc1.weight.grad
        assert first is not None and float(first.norm()) > 0


def test_code_residual_uses_the_plain_gate_on_the_carried_code():
    model, ctl = policy_model(n_layers=3, code_residual=True, share_projections=True)
    for g in ctl.policy_gates:
        assert g.rblapsum_surrogate_scope == "pool" and g._carry_chain is None
    model.eval()
    idx = torch.randint(0, 61, (2, 8))
    with torch.no_grad():
        a = model(idx)[0]
        b = model(idx)[0]
    assert torch.equal(a, b)
    codes = []
    for blk in model.blocks:
        blk.residual_out_bottleneck.gate.register_forward_hook(
            lambda m, i, o: codes.append(o.detach()))
    model.train()
    out = model(idx, idx, return_loss_details=True)
    assert len(codes) == 3 and all(int((cc != 0).sum(-1).max()) == 8 for cc in codes)
    assert len(out.policy_records) == 3


def test_sequence_costs_weight_sequences_by_valid_targets():
    ce_tokens = torch.tensor([[1.0, 2.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0]])
    valid = torch.tensor([[1, 1, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
    ce, seq_ce, seq_valid = sequence_costs(ce_tokens, valid)
    assert seq_valid.tolist() == [3, 0, 1]
    assert seq_ce.tolist() == pytest.approx([2.0, 0.0, 4.0])
    assert float(ce) == pytest.approx(10.0 / 4)                           # total / N_valid
    lp = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
    loss, _, _ = policy_backward_loss(ce, seq_ce, seq_valid, lp, baseline=1.0)
    (loss / 2).backward()                                                  # accumulation factor
    assert lp.grad.tolist() == pytest.approx([0.75 * 1.0 / 2, 0.0, 0.25 * 3.0 / 2])
    with pytest.raises(ValueError, match="no valid targets"):
        sequence_costs(ce_tokens, torch.zeros_like(valid))


def test_model_details_use_whole_sequence_cost_with_ignored_targets():
    model, _ = policy_model(n_layers=2)
    model.train()
    idx = torch.randint(0, 61, (3, 8))
    tgt = idx.clone()
    tgt[0, :5] = -100
    tgt[1] = -100                                                          # an empty sequence
    out = model(idx, tgt, return_loss_details=True)
    assert out.seq_valid.tolist() == [3, 0, 8]
    assert float(out.seq_ce[1]) == 0.0
    assert float(out.ce) == pytest.approx(float(out.ce_tokens.sum() / 11), rel=1e-6)
    logits, loss = model.eval()(idx, tgt)
    assert float(loss) == pytest.approx(float(torch.nn.functional.cross_entropy(
        logits.view(-1, 61), tgt.view(-1), ignore_index=-100)))
    with pytest.raises(ValueError, match="no valid targets"):
        model.train()(idx, torch.full_like(tgt, -100), return_loss_details=True)
    with pytest.raises(ValueError, match="targets"):
        model(idx, return_loss_details=True)


def test_ema_baseline_holds_within_a_step_and_updates_once():
    st = PolicyTrainingState("ema", 0.9, 4.0, CPU, seed=1)
    assert st.baseline == 4.0 and st.updates == 0
    st.accumulate(2.0 * 10, 10)                                            # micro-batch 1: mean 2
    assert st.baseline == 4.0                                              # fixed within the step
    st.accumulate(4.0 * 30, 30)                                            # micro-batch 2: mean 4
    mean = st.finish_step(world=1)
    assert mean == pytest.approx(3.5)                                      # token-weighted
    assert st.baseline == pytest.approx(0.9 * 4.0 + 0.1 * 3.5) and st.updates == 1
    assert math.isnan(st.finish_step(world=1))                             # nothing accumulated
    assert st.updates == 1                                                 # ... and no update
    none = PolicyTrainingState("none", 0.9, 4.0, CPU, seed=1)
    none.accumulate(3.0, 1)
    none.finish_step()
    assert none.baseline == 0.0 and none.updates == 0


def test_validation_and_probe_leave_training_state_untouched():
    model, ctl = policy_model(n_layers=2)
    state = PolicyTrainingState("ema", 0.99, 4.0, CPU, seed=5)
    before = state.generator.get_state().clone()
    usage_before = [g.usage_steps.clone() for g in ctl.policy_gates]

    class Stream:
        seq_len = 8

        def batch(self, bs, device, deterministic_offset=None):
            g = torch.Generator().manual_seed(int(deterministic_offset))
            x = torch.randint(0, 61, (bs, 8), generator=g)
            return x, x

    model.train()
    sto = evaluate_stochastic(model, Stream(), 2, 3, CPU, torch.float32, samples=3, seed=7)
    sto2 = evaluate_stochastic(model, Stream(), 2, 3, CPU, torch.float32, samples=3, seed=7)
    assert sto == sto2                                                     # fixed eval seed
    assert sto["samples"] == 3 and "mc_std" in sto and sto["mc_std"] > 0
    one = evaluate_stochastic(model, Stream(), 2, 3, CPU, torch.float32, samples=1, seed=7)
    assert "mc_std" not in one
    det = deterministic_probe(model, [Stream().batch(2, CPU, 0)], CPU, torch.float32)
    assert math.isfinite(det)
    assert model.training                                                  # mode restored
    assert torch.equal(state.generator.get_state(), before)
    assert state.baseline == 4.0
    assert all(torch.equal(a, g.usage_steps) for a, g in zip(usage_before, ctl.policy_gates))
    model.eval()
    with torch.no_grad():
        det2 = float(model(*Stream().batch(2, CPU, 0))[1])
    assert det == pytest.approx(det2)                                      # the probe IS the clean forward


def test_training_run_logs_both_ces_checkpoints_state_and_resumes(tmp_path):
    data = make_fake_dataset(tmp_path)
    out = str(tmp_path / "runs")
    cfg = smoke_config(data, out, policy_temperature_schedule="exponential",
                       policy_temperature_final=0.1, policy_temperature_hold_steps=1,
                       policy_temperature_anneal_steps=3)
    cfg.train.policy_val_samples = 2
    cfg.train.final_val_batches = 2
    summary = train(cfg)
    run_dir = tmp_path / "runs" / "pol"
    rows = [json.loads(l) for l in open(run_dir / "metrics.jsonl")]
    tr = [r for r in rows if "train/ce" in r]
    va = [r for r in rows if "val/ce" in r]
    assert tr and va
    for r in tr:
        assert r["train_stochastic/ce"] == r["train/ce"] == r["train/loss"]
        assert "train_deterministic_probe/ce" in r and math.isfinite(r["train_deterministic_probe/ce"])
        assert "policy/support_term_diag" in r and "bottleneck_policy_tau/blocks.0" in r
    assert tr[0]["policy/baseline"] == pytest.approx(math.log(VOCAB))       # step 1 uses B_0
    assert tr[0]["policy/tau"] == 0.5 and tr[-1]["policy/tau"] == pytest.approx(0.1)
    for r in va:
        assert r["val_deterministic/ce"] == r["val/ce"]
        assert r["val_stochastic/samples"] == 2.0 and "val_stochastic/mc_std" in r
        assert r["val_stochastic/gap"] == pytest.approx(r["val_stochastic/ce"] - r["val/ce"])
    assert summary["best_val_ce"] == min(r["val/ce"] for r in va)        # deterministic
    for key in ("val_final/ce", "val_final_deterministic/ce", "val_final_stochastic/ce",
                "val_final_stochastic/mc_std"):
        assert key in summary and math.isfinite(summary[key])
    payload = torch.load(run_dir / "latest.pt", weights_only=False)
    ps = payload["policy_state"]
    assert ps["version"] == 1 and ps["updates"] == 6 and ps["world_size"] == 1
    assert len(ps["generator_states"]) == 1 and math.isfinite(ps["baseline"])
    assert ps["baseline"] != math.log(VOCAB)
    # inference reconstruction: weights + config only, clean supports
    model, loaded, ctl = load_for_inference(str(run_dir / "latest.pt"))
    assert loaded.activation_bottleneck.surrogate_mode == "laplace_policy"
    assert loaded.activation_bottleneck.policy_temperature_final == 0.1
    x = torch.randint(0, VOCAB, (1, 8))
    with torch.no_grad():
        assert torch.equal(model(x)[0], model(x)[0])
    # resume: state restored, schedule recomputed from the step
    cfg2 = smoke_config(data, out, policy_temperature_schedule="exponential",
                        policy_temperature_final=0.1, policy_temperature_hold_steps=1,
                        policy_temperature_anneal_steps=3)
    cfg2.train.max_steps = 9
    cfg2.train.resume = "auto"
    train(cfg2)
    rows = [json.loads(l) for l in open(run_dir / "metrics.jsonl")]
    assert max(r["step"] for r in rows) == 9
    payload2 = torch.load(run_dir / "latest.pt", weights_only=False)
    assert payload2["policy_state"]["updates"] == 9
    # the dumped config keeps the ORIGINAL schedule, not the annealed tau
    dumped = yaml.safe_load(open(run_dir / "config.yaml"))
    assert dumped["activation_bottleneck"]["temperature"] == 0.5


def test_single_validation_draw_reports_no_spread(tmp_path):
    data = make_fake_dataset(tmp_path, 20_000, 4_000)
    cfg = smoke_config(data, str(tmp_path / "runs"), run_name="one")
    cfg.train.policy_val_samples = 1
    cfg.train.policy_train_deterministic_every_steps = -1
    train(cfg)
    rows = [json.loads(l) for l in open(tmp_path / "runs" / "one" / "metrics.jsonl")]
    va = [r for r in rows if "val/ce" in r]
    assert va and all("val_stochastic/mc_std" not in r and r["val_stochastic/samples"] == 1.0
                      for r in va)
    assert all("train_deterministic_probe/ce" not in r for r in rows if "train/ce" in r)


def test_resume_without_policy_state_is_rejected(tmp_path):
    data = make_fake_dataset(tmp_path, 20_000, 4_000)
    out = str(tmp_path / "runs")
    hard = smoke_config(data, out, run_name="hard", surrogate_mode="hard")
    train(hard)
    cfg = smoke_config(data, out, run_name="hard")                         # same run dir, policy mode
    cfg.train.max_steps = 9
    cfg.train.resume = "auto"
    with pytest.raises(ValueError, match="policy_state"):
        train(cfg)


def test_rng_and_baseline_continuation_reproduces_supports_and_gradients():
    gate = make_gate(n_features=32, k=4, j=12, temperature=0.5)
    torch.manual_seed(10)
    z0 = torch.randn(5, 32, dtype=torch.float64)
    st = PolicyTrainingState("ema", 0.9, 4.0, CPU, seed=3)
    st.accumulate(3.0, 1)
    st.finish_step()

    def step(state):
        z = z0.clone().requires_grad_(True)
        col = PolicyCollector()
        with policy_forward([gate], PolicyForwardSettings(sample=True, collector=col,
                                                           generator=state.generator)):
            y = gate(z)
        cost = (y * y).sum(-1)
        loss, _, _ = policy_backward_loss(cost.mean(), cost, torch.ones(5, dtype=torch.long),
                                          col.records[0].log_prob, state.baseline)
        loss.backward()
        return y.detach(), z.grad

    step(st)                                                               # consume some noise
    saved = copy.deepcopy(st.state_dict())
    y_a, g_a = step(st)
    fresh = PolicyTrainingState("ema", 0.9, 0.0, CPU, seed=999)
    fresh.load_state_dict(saved, rank=0, world=1)
    assert fresh.baseline == st.baseline and fresh.updates == st.updates
    y_b, g_b = step(fresh)
    assert torch.equal(y_a, y_b) and torch.equal(g_a, g_b)
    with pytest.raises(ValueError, match="world size"):
        fresh.load_state_dict(saved, rank=0, world=2)
    with pytest.raises(ValueError, match="version"):
        fresh.load_state_dict({**saved, "version": 99}, rank=0, world=1)
    with pytest.raises(ValueError, match="policy_baseline"):
        PolicyTrainingState("none", 0.9, 0.0, CPU, seed=1).load_state_dict(saved, 0, 1)


def test_no_graph_is_retained_across_forwards():
    model, ctl = policy_model(n_layers=2)
    model.train()
    idx = torch.randint(0, 61, (2, 8))
    refs = []
    for _ in range(3):
        out = model(idx, idx, return_loss_details=True)
        refs.append(weakref.ref(out.policy_log_prob))
        loss, _, _ = policy_backward_loss(out.ce, out.seq_ce, out.seq_valid,
                                          out.policy_log_prob, 4.0)
        loss.backward()
        del out, loss
    assert all(r() is None for r in refs)
    for g in ctl.policy_gates:
        assert g._policy_settings is None
        assert all(not v.requires_grad for v in g.diagnostics.values())


def test_effective_backward_support_is_k_and_the_decoder_scale_agrees():
    cfg = ActivationBottleneckConfig(enabled=True, n_features=64, k=8, j=24,
                                     surrogate_mode="laplace_policy", temperature=1.0)
    assert effective_backward_support(cfg) == 8.0
    hard = ActivationBottleneckConfig(enabled=True, n_features=64, k=8, j=24, surrogate_mode="hard")
    lap = ActivationBottleneckConfig(enabled=True, n_features=64, k=8, j=24, surrogate_mode="lapsum")
    assert effective_backward_support(hard) == 8.0 and effective_backward_support(lap) == 32.0
    torch.manual_seed(0)
    mcfg = ModelConfig(vocab_size=61, max_seq_len=16, n_layers=2, d_model=32, n_heads=4,
                       pos_encoding="rope", md_init=True, logit_scale="none",
                       bottleneck_decoder_scale="backward_preserving")
    model = build_model(mcfg)
    ctl = apply_activation_bottleneck(model, ActivationBottleneckConfig(
        enabled=True, n_features=64, k=8, j=24, surrogate_mode="laplace_policy",
        placement="residual_out", temperature=1.0))
    from wsparse.decouple import md_init_
    counts = md_init_(model, "row_col")
    assert counts["k_eff"] == 8.0
    assert all(m.decoder_scale == pytest.approx(math.sqrt(32 / 8)) for _, m in ctl.layers)


# --------------------------------------------------------------------------- #
# schedule, width, floor
# --------------------------------------------------------------------------- #


def test_exponential_schedule_endpoints_hold_and_anneal():
    assert scheduled_tau(0, 0.5, "constant") == 0.5
    assert scheduled_tau(10 ** 6, 0.5, "constant") == 0.5
    args = dict(tau_final=0.05, hold_steps=1000, anneal_steps=17000)
    assert scheduled_tau(0, 0.5, "exponential", **args) == 0.5
    assert scheduled_tau(1000, 0.5, "exponential", **args) == 0.5           # end of the hold
    mid = scheduled_tau(1000 + 8500, 0.5, "exponential", **args)
    assert mid == pytest.approx(math.sqrt(0.5 * 0.05))                      # geometric midpoint
    assert scheduled_tau(18000, 0.5, "exponential", **args) == pytest.approx(0.05)
    assert scheduled_tau(25000, 0.5, "exponential", **args) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        scheduled_tau(0, 0.5, "exponential", tau_final=None, anneal_steps=10)
    with pytest.raises(ValueError):
        scheduled_tau(0, 0.5, "linear")


def test_controller_set_step_writes_the_scheduled_tau_and_keeps_tau0():
    model, ctl = policy_model(n_layers=2, policy_temperature_schedule="exponential",
                              policy_temperature_final=0.05, policy_temperature_hold_steps=10,
                              policy_temperature_anneal_steps=20)
    assert ctl.policy_active
    assert ctl.set_step(0) == 0.5 and all(g.policy_tau == 0.5 for g in ctl.policy_gates)
    assert ctl.set_step(20) == pytest.approx(math.sqrt(0.5 * 0.05))
    assert ctl.set_step(1000) == pytest.approx(0.05)
    assert all(g.policy_tau == pytest.approx(0.05) and g.temperature == 0.5
               for g in ctl.policy_gates)
    assert ctl.cfg.temperature == 0.5
    # every other mode: a constant temperature, set_step returns 0.0 as before
    _, hard = policy_model(n_layers=2, surrogate_mode="hard", temperature=1.0)
    assert hard.set_step(5) == 0.0 and not hard.policy_active and hard.policy_gates == []
    # a resume recomputes tau from the restored step
    _, ctl2 = policy_model(n_layers=2, policy_temperature_schedule="exponential",
                           policy_temperature_final=0.05, policy_temperature_hold_steps=10,
                           policy_temperature_anneal_steps=20)
    assert ctl2.set_step(20) == ctl.set_step(20)


def test_effective_width_modes_floor_and_zero_span():
    cand = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.5], [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]])
    k, j = 2, 4
    t, binding, scale = effective_width("absolute", 0.3, cand, k, j, 1e-6)
    assert t.flatten().tolist() == pytest.approx([0.3, 0.3]) and not binding.any()
    assert scale.tolist() == [[1.0], [1.0]]
    t, binding, scale = effective_width("relative_b", 0.5, cand, k, j, 1e-6)
    assert t.flatten().tolist() == pytest.approx([1.5, 1.0])              # tau * s_(K+1)
    t, binding, scale = effective_width("relative_span", 0.5, cand, k, j, 1e-6)
    assert t[0].item() == pytest.approx(0.5 * (3.0 - 0.5))                  # tau * (s_(K+1) - s_(K+J))
    assert t[1].item() == pytest.approx(1e-6) and bool(binding[1]) and not bool(binding[0])
    assert scale[1].item() == 0.0                                           # the zero span is reported
    assert not t.requires_grad
    t, binding, _ = effective_width("absolute", 1e-9, cand, k, j, 1e-6)
    assert bool(binding.all()) and t.flatten().tolist() == pytest.approx([1e-6, 1e-6])
    with pytest.raises(ValueError):
        effective_width("other", 1.0, cand, k, j, 1e-6)
    gate = make_gate(n_features=8, k=2, j=4, temperature=0.5, policy_temperature_mode="relative_span")
    z = torch.ones(3, 8)                                                   # all tied: zero spans
    y, _ = sampled(gate, z)
    d = gate.diagnostics
    assert d["policy_span_zero_frac"] == 1.0 and d["policy_t_floor_frac"] == 1.0
    assert torch.equal((y != 0).sum(-1), torch.full((3,), 2))


def test_small_width_collapsed_perturbations_are_diagnosed():
    gate = make_gate(n_features=16, k=2, j=6, temperature=1e-12, policy_min_temperature=1e-12,
                     policy_center_scores=False)
    z = torch.randn(50, 16) * 100
    y, _ = sampled(gate, z)
    assert float(gate.diagnostics["policy_collapsed_frac"]) > 0.5           # r rounds back to u


# --------------------------------------------------------------------------- #
# configuration contract
# --------------------------------------------------------------------------- #

POLICY_FIELDS = {
    "policy_temperature_mode": "relative_span", "policy_temperature_schedule": "exponential",
    "policy_temperature_final": 0.05, "policy_temperature_hold_steps": 100,
    "policy_temperature_anneal_steps": 900, "policy_min_temperature": 1e-5,
    "policy_center_scores": False, "policy_baseline": "none", "policy_baseline_decay": 0.9,
    "policy_baseline_initial": 3.5, "policy_support_scale": 0.2,
    "policy_support_scale_mode": "scheduled_temperature",
    "policy_support_temperature_ref": 0.5, "policy_support_scale_max": 0.3,
}
TRAIN_FIELDS = {"policy_val_samples": 3, "policy_val_seed": 42,
                "policy_train_deterministic_every_steps": 7}


def _check(cfg: Config):
    ab = cfg.activation_bottleneck
    assert ab.surrogate_mode == "laplace_policy"
    for name, value in POLICY_FIELDS.items():
        assert getattr(ab, name) == value, name
    for name, value in TRAIN_FIELDS.items():
        assert getattr(cfg.train, name) == value, name


def test_policy_fields_survive_yaml_cli_dict_and_saved_config(tmp_path):
    base = {"activation_bottleneck": {"enabled": True, "n_features": 64, "k": 8, "j": 8,
                                      "surrogate_mode": "laplace_policy", "temperature": 0.5,
                                      **POLICY_FIELDS},
            "train": dict(TRAIN_FIELDS)}
    path = tmp_path / "pol.yaml"
    path.write_text(yaml.safe_dump(base))
    cfg = load_config(str(path))
    _check(cfg)
    overrides = [f"--activation_bottleneck.{k}={v}" for k, v in POLICY_FIELDS.items()]
    overrides += [f"--train.{k}={v}" for k, v in TRAIN_FIELDS.items()]
    overrides += ["--activation_bottleneck.enabled=true", "--activation_bottleneck.n_features=64",
                  "--activation_bottleneck.k=8", "--activation_bottleneck.j=8",
                  "--activation_bottleneck.surrogate_mode=laplace_policy",
                  "--activation_bottleneck.temperature=0.5"]
    _check(load_config(None, overrides))
    _check(config_from_dict(cfg.to_dict()))
    dumped = tmp_path / "dumped.yaml"
    cfg.dump(str(dumped))
    _check(load_config(str(dumped)))
    _check(config_from_dict(json.loads(json.dumps(cfg.to_dict()))))


def test_defaults_and_shipped_configs():
    ab = ActivationBottleneckConfig()
    assert ab.policy_temperature_mode == "absolute" and ab.policy_temperature_schedule == "constant"
    assert ab.policy_temperature_final is None and ab.policy_min_temperature == 1e-6
    assert ab.policy_center_scores is True and ab.policy_baseline == "ema"
    assert ab.policy_baseline_decay == 0.99 and ab.policy_baseline_initial is None
    assert ab.policy_support_scale == 1.0 and ab.policy_support_scale_mode == "constant"
    assert ab.policy_support_temperature_ref == 1.0 and ab.policy_support_scale_max is None
    tr = TrainConfig()
    assert (tr.policy_val_samples, tr.policy_val_seed, tr.policy_train_deterministic_every_steps) \
        == (1, 1337, 0)
    for name in ("bn_laplace_policy_absolute.yaml", "bn_laplace_policy_relative.yaml"):
        cfg = load_config(os.path.join(CONFIG_DIR, name))
        assert cfg.activation_bottleneck.surrogate_mode == "laplace_policy"
        assert cfg.train.compile is False and cfg.train.policy_val_samples == 2
        assert cfg.activation_bottleneck.policy_temperature_hold_steps + \
            cfg.activation_bottleneck.policy_temperature_anneal_steps <= cfg.train.max_steps
    rel = load_config(os.path.join(CONFIG_DIR, "bn_laplace_policy_relative.yaml"))
    assert rel.activation_bottleneck.policy_support_scale_mode == "scheduled_temperature"


@pytest.mark.parametrize("bad", [
    dict(selection_mode="gated_topk"),
    dict(stochastic_width="uniform"),
    dict(value_shift="energy"),
    dict(rblapsum_boundary_floor=0.1),
    dict(hard_inference=False),
    dict(rblapsum_surrogate_scope="first_order"),
    dict(rblapsum_support_scale=0.5),
    dict(rblapsum_center_tokens=True),
    dict(rblapsum_kernel_width="relative_b"),
    dict(rblapsum_boundary_grad_mode="through_rank_kappa"),
    dict(policy_temperature_mode="relative"),
    dict(policy_temperature_schedule="linear"),
    dict(policy_temperature_schedule="exponential"),                       # no final
    dict(policy_temperature_schedule="exponential", policy_temperature_final=0.05),  # no duration
    dict(policy_temperature_schedule="exponential", policy_temperature_final=2.0,
         policy_temperature_anneal_steps=10),                               # final > tau_0
    dict(policy_temperature_schedule="exponential", policy_temperature_final=float("nan"),
         policy_temperature_anneal_steps=10),
    dict(policy_temperature_final=0.1),                                     # constant + endpoint
    dict(policy_temperature_hold_steps=5),
    dict(policy_temperature_anneal_steps=-1, policy_temperature_schedule="exponential",
         policy_temperature_final=0.1),
    dict(policy_min_temperature=0.0),
    dict(policy_min_temperature=float("inf")),
    dict(policy_baseline="batch"),
    dict(policy_baseline_decay=1.0),
    dict(policy_baseline_decay=float("nan")),
    dict(policy_baseline_initial=float("inf")),
    dict(policy_support_scale=-1.0),
    dict(policy_support_scale=float("nan")),
    dict(policy_support_scale_mode="scale_by_T"),
    dict(policy_support_temperature_ref=0.0),
    dict(policy_support_scale_max=0.0),
    dict(temperature=0.0),
])
def test_invalid_policy_configurations_are_rejected(bad):
    kw = dict(enabled=True, n_features=64, k=8, j=8, surrogate_mode="laplace_policy",
              temperature=0.5)
    kw.update(bad)
    with pytest.raises(ValueError):
        ActivationBottleneckConfig(**kw)


def test_policy_fields_under_other_modes_and_bad_train_fields_are_rejected():
    for mode in ("hard", "lapsum", "rblapsum", "rblapsum_sf", "soft_ste"):
        base = dict(enabled=True, n_features=64, k=8, j=8, surrogate_mode=mode, temperature=1.0)
        ActivationBottleneckConfig(**base)                                 # defaults are fine
        with pytest.raises(ValueError, match="policy_"):
            ActivationBottleneckConfig(**base, policy_temperature_schedule="exponential",
                                       policy_temperature_final=0.1,
                                       policy_temperature_anneal_steps=10)
        with pytest.raises(ValueError, match="policy_"):
            ActivationBottleneckConfig(**base, policy_baseline="none")
    with pytest.raises(ValueError):
        TrainConfig(policy_val_samples=0)
    with pytest.raises(ValueError):
        TrainConfig(policy_train_deterministic_every_steps=-2)
    TrainConfig(policy_train_deterministic_every_steps=-1)


def test_laplace_policy_allows_the_hard_mode_shape_rules():
    ActivationBottleneckConfig(enabled=True, n_features=64, k=8, j=0, surrogate_mode="laplace_policy")
    ActivationBottleneckConfig(enabled=True, n_features=8, k=8, j=0, surrogate_mode="laplace_policy")
    with pytest.raises(ValueError):
        ActivationBottleneckConfig(enabled=True, n_features=8, k=8, j=1,
                                   surrogate_mode="laplace_policy")
    with pytest.raises(ValueError):
        ActivationBottleneckConfig(enabled=True, n_features=64, k=8, j=0, surrogate_mode="lapsum")


def test_archived_configs_still_migrate_and_never_become_policy_runs():
    """The removed temperature-schedule names are dropped as before; nothing
    maps onto the policy_* namespace."""
    archived = {"activation_bottleneck": {
        "enabled": True, "n_features": 64, "k": 8, "j": 8, "surrogate_mode": "lapsum_scheduled",
        "temperature_schedule": "constant", "temperature_scale_mode": "absolute",
        "temperature_start": 2.0, "temperature_end": 0.1, "temperature_anneal_steps": 500,
        "differentiate_temperature": True, "n_eff": 4.0},
        "model": {"n_layers": 2, "d_model": 32, "n_heads": 2}}
    cfg = config_from_dict(archived)
    ab = cfg.activation_bottleneck
    assert ab.surrogate_mode == "lapsum" and ab.temperature == 2.0
    assert ab.policy_temperature_schedule == "constant" and ab.policy_temperature_final is None
    with pytest.raises(ValueError, match="removed"):
        config_from_dict({"activation_bottleneck": {"enabled": True, "n_features": 64, "k": 8,
                                                    "j": 8, "surrogate_mode": "reinforce_topk"}})
    # the removed reinforce_* knobs are dropped by the migration as before and
    # never reach the policy_* fields
    dropped = load_config(None, ["activation_bottleneck.reinforce_coef=1.0",
                                 "activation_bottleneck.reinforce_baseline=ema"])
    assert dropped.activation_bottleneck.surrogate_mode == "hard"
    assert dropped.activation_bottleneck.policy_baseline == "ema"       # the default, untouched
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(None, ["activation_bottleneck.policy_temperature=1.0"])
