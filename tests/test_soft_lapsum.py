"""soft_lapsum: LapSum's probabilities in the forward, y = z * p over the
Top(K+J) pool with sum p = K, and the temperature annealing it shares with
rblapsum_sf.

The decisive gate test is exactness: the backward (p on the values, the
implicit-barrier VJP of _LapSumProbs on the scores) must be the gradient of the
forward, checked against finite differences by gradcheck in float64.
"""
import dataclasses
import json
import math
import os

import pytest
import torch

from wsparse.bottleneck import AdaptiveLapSumTopKGate, apply_activation_bottleneck
from wsparse.bottleneck.controller import (anneal_fraction, annealed_k,
                                           annealed_temperature)
from wsparse.bottleneck.lapsum import laplace_cdf, lapsum_barrier_bisect
from wsparse.config import ActivationBottleneckConfig, Config
from wsparse.model import ModelConfig, build_model
from wsparse.train import load_for_inference, train

from test_train import make_fake_dataset, smoke_config


def make_gate(selection="abs_topk", k=3, j=5, n=16, T=0.7, **kw):
    return AdaptiveLapSumTopKGate(
        n_features=n, k=k, j=j, selection_mode=selection, surrogate_mode="soft_lapsum",
        temperature=T, solver_dtype="float64", **kw)


def reference_soft(a, k, m, T, selection="abs_topk"):
    """The spec's forward: p_i = F((s_i - b)/T) with sum_pool p = K, b by bisection."""
    scores = a.abs() if selection == "abs_topk" else a
    cs, ci = torch.topk(scores, m, dim=-1, largest=True, sorted=True)
    b = lapsum_barrier_bisect(cs, k, torch.full_like(cs[..., 0], T), tol=1e-13)
    p = laplace_cdf((cs - b.unsqueeze(-1)) / T)
    return torch.zeros_like(a).scatter(-1, ci, torch.gather(a, -1, ci) * p), p


# --------------------------------------------------------------------------- #
# gate
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("selection", ["abs_topk", "topk"])
def test_forward_is_z_times_p_with_budget_k(selection):
    torch.manual_seed(0)
    a = torch.randn(6, 16, dtype=torch.float64)
    g = make_gate(selection)
    g.train()
    y = g(a)
    want, p = reference_soft(a, g.k, g.m, g.temperature, selection)
    assert torch.allclose(y, want, atol=1e-10)
    assert torch.allclose(p.sum(-1), torch.full((6,), float(g.k), dtype=p.dtype), atol=1e-9)
    assert ((y != 0).sum(-1) == g.m).all()  # dense on the pool, zero outside
    # the tail mass outside the top K equals the shortfall inside it
    tail = p[..., g.k:].sum(-1)
    assert torch.allclose(tail, (1 - p[..., :g.k]).sum(-1), atol=1e-9)
    assert math.isclose(float(g.diagnostics["soft_tail_mass"]), float(tail.mean()),
                        rel_tol=1e-9)


@pytest.mark.parametrize("selection", ["abs_topk", "topk", "gated_topk"])
def test_backward_is_the_gradient_of_the_forward(selection):
    torch.manual_seed(1)
    g = make_gate(selection)
    g.train()
    a = torch.randn(4, 16, dtype=torch.float64, requires_grad=True)
    if selection == "gated_topk":
        v = torch.randn(4, 16, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(lambda s, x: g(s, x), (a, v), eps=1e-6, atol=1e-7)
    else:
        assert torch.autograd.gradcheck(g, (a,), eps=1e-6, atol=1e-7)


def test_score_gradient_is_the_implicit_barrier_vjp():
    torch.manual_seed(2)
    g = make_gate("topk")
    g.train()
    a = torch.randn(3, 16, dtype=torch.float64, requires_grad=True)
    up = torch.randn(3, 16, dtype=torch.float64)
    g(a).backward(up)
    cs, ci = torch.topk(a.detach(), g.m, dim=-1, largest=True, sorted=True)
    _, p = reference_soft(a.detach(), g.k, g.m, g.temperature, "topk")
    b = lapsum_barrier_bisect(cs, g.k, torch.full_like(cs[..., 0], g.temperature), tol=1e-13)
    kappa = torch.exp(-(cs - b.unsqueeze(-1)).abs() / g.temperature) / (2 * g.temperature)
    u = torch.gather(up, -1, ci) * cs                       # g * z on the pool
    q = kappa / kappa.sum(-1, keepdim=True)
    support = kappa * (u - (q * u).sum(-1, keepdim=True))  # zero-sum over the pool
    want = torch.gather(up, -1, ci) * p + support          # value path + score path
    assert torch.allclose(torch.gather(a.grad, -1, ci), want, atol=1e-9)
    outside = torch.ones_like(a, dtype=torch.bool).scatter(-1, ci, False)
    assert (a.grad[outside] == 0).all()


def test_eval_is_hard_topk_unless_hard_inference_is_off():
    torch.manual_seed(3)
    a = torch.randn(5, 16, dtype=torch.float64)
    g = make_gate()
    g.train()
    with torch.no_grad():
        soft_train = g(a)  # train mode without grad is still the soft forward
    g.eval()
    hard = AdaptiveLapSumTopKGate(n_features=16, k=3, j=5, surrogate_mode="hard").eval()
    assert torch.equal(g(a), hard(a))
    g.hard_inference = False
    with torch.no_grad():
        assert torch.allclose(g(a), soft_train, atol=1e-12)


def test_set_temperature_is_a_gate_of_that_temperature():
    torch.manual_seed(4)
    a = torch.randn(5, 16, dtype=torch.float64)
    for mode in ("soft_lapsum", "rblapsum_sf"):
        g = AdaptiveLapSumTopKGate(n_features=16, k=3, j=5, surrogate_mode=mode,
                                   temperature=2.0, solver_dtype="float64").train()
        g.set_temperature(0.3)
        ref = AdaptiveLapSumTopKGate(n_features=16, k=3, j=5, surrogate_mode=mode,
                                     temperature=0.3, solver_dtype="float64").train()
        assert torch.equal(g(a), ref(a))
    with pytest.raises(ValueError):
        g.set_temperature(0.0)
    with pytest.raises(ValueError):
        AdaptiveLapSumTopKGate(n_features=16, k=3, j=5,
                               surrogate_mode="lapsum").set_temperature(0.5)


# --------------------------------------------------------------------------- #
# temperature schedule
# --------------------------------------------------------------------------- #


def test_schedule_endpoints_and_shapes():
    t0, tf, n = 2.0, 0.25, 1000
    for kw in (dict(), dict(schedule="log", half_step=50.0),
               dict(schedule="log", s0=3.0), dict(schedule="power", s0=10.0, gamma=0.4),
               dict(schedule="power", s0=10.0, gamma=-0.5)):
        ts = [annealed_temperature(s, t0, tf, n, **kw) for s in range(n + 50)]
        assert ts[0] == t0 and set(ts[n:]) == {tf}
        assert all(a >= b for a, b in zip(ts, ts[1:]))
        # T(s) = T_f + (T_0 - T_f)(1 - f(s)) with the f(s) that hard_k_annealing
        # quantizes, K'(s) = K + J - min(J, floor((J + 1) f(s)))
        for s in (1, 7, 120, 600):
            f = anneal_fraction(s, n, **kw)
            assert 1 - (ts[s] - tf) / (t0 - tf) == pytest.approx(f, abs=1e-12)
            if kw:
                assert annealed_k(s, 32, 99, n, **kw) == 32 + 99 - min(99, math.floor(100 * f))
    # linear: T(s) = T_f + (T_0 - T_f)(1 - s/S)
    assert annealed_temperature(250, t0, tf, n) == pytest.approx(tf + (t0 - tf) * 0.75)
    # log half step: T(h) midway
    assert annealed_temperature(50, t0, tf, n, "log", half_step=50.0) == pytest.approx(
        0.5 * (t0 + tf))
    # power gamma = 1 is linear; tf > t0 warms up instead
    assert annealed_temperature(300, t0, tf, n, "power", s0=5.0, gamma=1.0) == pytest.approx(
        annealed_temperature(300, t0, tf, n))
    assert annealed_temperature(500, 0.5, 1.5, n) == pytest.approx(1.0)
    assert annealed_temperature(-3, t0, tf, n) == t0


def test_temperature_config_rules():
    base = dict(enabled=True, k=8, j=24, surrogate_mode="soft_lapsum", temperature=2.0,
                temperature_annealing=True, temperature_annealing_final=0.5)
    c = ActivationBottleneckConfig(**base, temperature_annealing_schedule="log",
                                   temperature_annealing_half_step=100,
                                   temperature_annealing_steps=900)
    assert c.temperature_annealing_half_step == 100.0 and c.temperature_annealing_steps == 900
    ActivationBottleneckConfig(**{**base, "surrogate_mode": "rblapsum_sf"})
    for bad in (dict(surrogate_mode="lapsum"), dict(surrogate_mode="rblapsum"),
                dict(surrogate_mode="hard"), dict(temperature_annealing_final=None),
                dict(temperature_annealing_final=0.0),
                dict(temperature_annealing_schedule="cosine"),
                dict(temperature_annealing_schedule="log"),
                dict(temperature_annealing_schedule="power", temperature_annealing_s0=1.0),
                dict(temperature_annealing_schedule="linear", temperature_annealing_s0=1.0),
                dict(temperature_annealing_steps=0)):
        with pytest.raises(ValueError):
            ActivationBottleneckConfig(**{**base, **bad})
    off = dict(enabled=True, k=8, j=24, surrogate_mode="soft_lapsum")
    for stray in (dict(temperature_annealing_final=0.5),
                  dict(temperature_annealing_steps=10),
                  dict(temperature_annealing_schedule="log",
                       temperature_annealing_half_step=5)):
        with pytest.raises(ValueError):
            ActivationBottleneckConfig(**off, **stray)
    cfg = Config()
    cfg.train.max_steps = 100
    for extra in (dict(temperature_annealing_steps=101),
                  dict(temperature_annealing_schedule="log",
                       temperature_annealing_half_step=50)):
        with pytest.raises(ValueError):
            Config(train=cfg.train,
                   activation_bottleneck=ActivationBottleneckConfig(**base, **extra))


def test_controller_sets_every_gate_and_logs_the_temperature():
    for mode in ("soft_lapsum", "rblapsum_sf"):
        m = build_model(ModelConfig(vocab_size=61, max_seq_len=16, n_layers=3, d_model=24,
                                    n_heads=4))
        cfg = ActivationBottleneckConfig(
            enabled=True, n_features=96, k=8, j=24, surrogate_mode=mode, temperature=2.0,
            temperature_annealing=True, temperature_annealing_final=0.5)
        ctl = apply_activation_bottleneck(m, cfg, max_steps=100)
        assert ctl.set_step(0) == 0.0
        assert {layer.gate.temperature for _, layer in ctl.layers} == {2.0}
        ctl.set_step(40)
        want = annealed_temperature(40, 2.0, 0.5, 100)
        assert {layer.gate.temperature for _, layer in ctl.layers} == {want}
        idx = torch.randint(0, 61, (2, 12))
        m.train()
        _, loss = m(idx, idx)
        loss.backward()
        assert torch.isfinite(loss)
        assert ctl.stats()["bottleneck/temperature"] == pytest.approx(want)
        ctl.set_step(100)
        assert ctl.temperature == 0.5
        assert cfg.temperature == 2.0  # the config keeps T_0

    plain = build_model(ModelConfig(vocab_size=61, max_seq_len=16, n_layers=3, d_model=24,
                                    n_heads=4))
    ctl2 = apply_activation_bottleneck(plain, ActivationBottleneckConfig(
        enabled=True, n_features=96, k=8, j=24, surrogate_mode="soft_lapsum",
        temperature=2.0), max_steps=100)
    ctl2.set_step(60)
    assert ctl2.temperature == 2.0


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #


def soft_config(data_dir, out_dir, mode="soft_lapsum", max_steps=12, **anneal):
    cfg = smoke_config(data_dir, out_dir)
    cfg.train.max_steps = max_steps
    cfg.train.checkpoint_every_steps = 6
    cfg.train.keep_last_checkpoints = 3
    cfg.train.log_every_steps = 1
    cfg.train.validate_every_steps = 6
    ab = ActivationBottleneckConfig(
        enabled=True, n_features=128, k=4, j=12, layers="all", placement="pre_mlp",
        surrogate_mode=mode, temperature=1.5, **anneal)
    return Config(model=cfg.model, data=cfg.data, train=cfg.train,
                  activation_bottleneck=ActivationBottleneckConfig(**dataclasses.asdict(ab)))


def test_soft_lapsum_trains_and_logs_both_validations(tmp_path):
    data_dir = make_fake_dataset(tmp_path, 6000, 1200)
    out = str(tmp_path / "runs_sl")
    train(soft_config(data_dir, out, max_steps=6))
    run = os.path.join(out, "test")
    records = [json.loads(line) for line in open(os.path.join(run, "metrics.jsonl"))]
    val = [r for r in records if "val/ce" in r]
    assert val and all("val_soft/ce" in r and math.isfinite(r["val_soft/ce"]) for r in val)
    trained = [r for r in records if "bottleneck/soft_tail_mass" in r]
    assert trained and all(0 < r["bottleneck/soft_tail_mass"] < 4 for r in trained)


@pytest.mark.parametrize("mode", ["soft_lapsum", "rblapsum_sf"])
def test_training_follows_the_temperature_schedule_resumes_and_reloads(tmp_path, mode):
    data_dir = make_fake_dataset(tmp_path, 6000, 1200)
    out = str(tmp_path / f"runs_{mode}")
    anneal = dict(temperature_annealing=True, temperature_annealing_final=0.25,
                  temperature_annealing_schedule="log", temperature_annealing_half_step=2.0,
                  temperature_annealing_steps=10)
    seen = {}

    def record(step, model, bottleneck, optimizer):
        seen[step] = {layer.gate.temperature for _, layer in bottleneck.layers}

    # stop after step 6 (checkpoint), then resume to the end
    train(soft_config(data_dir, out, mode, **anneal), on_step=record,
          should_stop=lambda: "stop" if max(seen) == 5 else None)
    cfg = soft_config(data_dir, out, mode, **anneal)
    cfg.train.resume = "auto"
    train(cfg, on_step=record)
    sched = dict(schedule="log", half_step=2.0)
    assert sorted(seen) == list(range(12))
    for step, temps in seen.items():
        assert temps == {annealed_temperature(step, 1.5, 0.25, 10, **sched)}
    assert seen[0] == {1.5} and seen[10] == seen[11] == {0.25}

    run = os.path.join(out, "test")
    records = [json.loads(line) for line in open(os.path.join(run, "metrics.jsonl"))]
    logged = {r["step"]: r["bottleneck/temperature"] for r in records
              if "bottleneck/temperature" in r}
    # the step logged as s+1 ran at T(s)
    assert logged and all(v == pytest.approx(annealed_temperature(s - 1, 1.5, 0.25, 10, **sched))
                          for s, v in logged.items())
    assert all("val_soft/ce" in r for r in records if "val/ce" in r)

    # a checkpoint reloads at the temperature of its last step
    _, cfg6, ctl = load_for_inference(os.path.join(run, "ckpt_step6.pt"))
    assert ctl.temperature == annealed_temperature(5, 1.5, 0.25, 10, **sched)
    assert cfg6.activation_bottleneck.temperature == 1.5
    _, _, ctl = load_for_inference(os.path.join(run, "ckpt_step12.pt"))
    assert ctl.temperature == 0.25
