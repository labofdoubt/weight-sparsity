"""hard_k_annealing: the hard width falls from K+J to K over the run."""
import dataclasses
import json
import os
from collections import Counter

import pytest
import torch

from wsparse.bottleneck import apply_activation_bottleneck
from wsparse.bottleneck.controller import annealed_k
from wsparse.bottleneck.gate import AdaptiveLapSumTopKGate
from wsparse.config import ActivationBottleneckConfig, Config
from wsparse.model import ModelConfig, build_model
from wsparse.train import load_for_inference, train

from test_train import make_fake_dataset, smoke_config


def test_schedule_starts_at_k_plus_j_ends_at_k_in_equal_plateaus():
    for k, j, steps in ((32, 480, 20_000), (4, 5, 12), (8, 3, 4), (16, 7, 1000)):
        widths = [annealed_k(s, k, j, steps) for s in range(steps)]
        assert widths[0] == k + j and widths[-1] == k
        # one feature at a time, never up
        assert all(a - b in (0, 1) for a, b in zip(widths, widths[1:]))
        # every width K+J..K is visited, each for floor or ceil of steps/(J+1)
        counts = Counter(widths)
        assert sorted(counts) == list(range(k, k + j + 1))
        share = steps / (j + 1)
        assert all(int(share) <= c <= int(share) + 1 for c in counts.values())
    # the example of the request: one feature every 20000/481 ~ 41.6 steps
    assert annealed_k(41, 32, 480, 20_000) == 512
    assert annealed_k(42, 32, 480, 20_000) == 511
    # clamped outside the run
    assert annealed_k(-5, 32, 480, 20_000) == 512
    assert annealed_k(25_000, 32, 480, 20_000) == 32


def test_config_rules():
    ActivationBottleneckConfig(enabled=True, k=8, j=24, hard_k_annealing=True)
    for bad in (dict(surrogate_mode="rblapsum"), dict(j=0),
                dict(stochastic_width="uniform"),
                dict(selection_mode="abs_topk", value_shift="fixed")):
        kw = dict(enabled=True, k=8, j=24, hard_k_annealing=True)
        kw.update(bad)
        with pytest.raises(ValueError):
            ActivationBottleneckConfig(**kw)
    cfg = Config()
    cfg.train.max_steps = 20
    with pytest.raises(ValueError):
        Config(train=cfg.train, activation_bottleneck=ActivationBottleneckConfig(
            enabled=True, k=8, j=24, hard_k_annealing=True))


def test_gate_at_an_annealed_width_is_the_hard_gate_of_that_width():
    torch.manual_seed(0)
    g = AdaptiveLapSumTopKGate(n_features=64, k=8, j=24, surrogate_mode="hard")
    z = torch.randn(50, 64)
    for k_now in (32, 20, 8):
        g.set_active_count(k_now)
        assert (g.k, g.j, g.m) == (k_now, 32 - k_now, 32)
        ref = AdaptiveLapSumTopKGate(n_features=64, k=k_now, j=0, surrogate_mode="hard")
        assert torch.equal(g(z), ref(z))
        assert torch.equal((g(z) != 0).sum(-1), torch.full((50,), k_now))
    with pytest.raises(ValueError):
        g.set_active_count(33)


def test_controller_sets_every_gate_and_leaves_other_runs_alone():
    m = build_model(ModelConfig(vocab_size=61, max_seq_len=16, n_layers=3, d_model=24,
                                n_heads=4))
    cfg = ActivationBottleneckConfig(enabled=True, n_features=96, k=8, j=24,
                                     placement="residual_out", code_residual=True,
                                     post_norm=True, hard_k_annealing=True)
    ctl = apply_activation_bottleneck(m, cfg, max_steps=100)
    assert ctl.set_step(0) == 0.0
    assert {layer.gate.k for _, layer in ctl.layers} == {32}
    ctl.set_step(99)
    assert {layer.gate.k for _, layer in ctl.layers} == {8}
    idx = torch.randint(0, 61, (2, 12))
    m.train()
    _, loss = m(idx, idx)
    loss.backward()
    assert torch.isfinite(loss)
    assert ctl.stats()["bottleneck/active_count"] == 8.0

    plain = build_model(ModelConfig(vocab_size=61, max_seq_len=16, n_layers=3, d_model=24,
                                    n_heads=4))
    ctl2 = apply_activation_bottleneck(
        plain, ActivationBottleneckConfig(enabled=True, n_features=96, k=8, j=24),
        max_steps=100)
    ctl2.set_step(0)
    assert {layer.gate.k for _, layer in ctl2.layers} == {8}


def annealing_config(data_dir, out_dir, max_steps=12):
    cfg = smoke_config(data_dir, out_dir)
    cfg.train.max_steps = max_steps
    cfg.train.checkpoint_every_steps = 6
    cfg.train.keep_last_checkpoints = 3
    cfg.train.log_every_steps = 1
    cfg.train.validate_every_steps = 6
    cfg.activation_bottleneck = ActivationBottleneckConfig(
        enabled=True, n_features=128, k=4, j=5, layers="all", placement="residual_out",
        code_residual=True, post_norm=True, hard_k_annealing=True)
    return Config(model=cfg.model, data=cfg.data, train=cfg.train,
                  activation_bottleneck=cfg.activation_bottleneck)


def test_training_follows_the_schedule_resumes_and_reloads_its_width(tmp_path):
    data_dir = make_fake_dataset(tmp_path, 6000, 1200)
    out = str(tmp_path / "runs_anneal")
    seen = {}

    def record(step, model, bottleneck, optimizer):
        seen[step] = {layer.gate.k for _, layer in bottleneck.layers}

    # stop after step 6 (checkpoint), then resume to the end
    cfg = annealing_config(data_dir, out)
    train(cfg, on_step=record, should_stop=lambda: "stop" if max(seen) == 5 else None)
    cfg = annealing_config(data_dir, out)
    cfg.train.resume = "auto"
    train(cfg, on_step=record)
    assert sorted(seen) == list(range(12))
    for step, widths in seen.items():
        assert widths == {annealed_k(step, 4, 5, 12)}

    run = os.path.join(out, "test")
    records = [json.loads(l) for l in open(os.path.join(run, "metrics.jsonl"))]
    logged = {r["step"]: r["bottleneck/active_count"] for r in records
              if "bottleneck/active_count" in r}
    # the step logged as s+1 ran at K'(s)
    assert logged and all(v == annealed_k(s - 1, 4, 5, 12) for s, v in logged.items())
    assert logged[12] == 4.0

    # a checkpoint reloads at the width of its last step; the final one at K
    model, _, ctl = load_for_inference(os.path.join(run, "ckpt_step6.pt"))
    assert {layer.gate.k for _, layer in ctl.layers} == {annealed_k(5, 4, 5, 12)}
    model, _, ctl = load_for_inference(os.path.join(run, "ckpt_step12.pt"))
    assert {layer.gate.k for _, layer in ctl.layers} == {4}


def test_log_schedule_endpoints_half_step_and_linear_limit():
    k, j, steps = 32, 480, 20_000
    for h in (125, 1000, 8000):
        widths = [annealed_k(s, k, j, steps, "log", h) for s in range(steps)]
        assert widths[0] == k + j and widths[-1] == k
        assert all(a >= b for a, b in zip(widths, widths[1:]))
        # half of the J features removed at s = h (to the floor's rounding)
        assert abs((k + j - widths[h]) - (j + 1) / 2) <= 1
        # more of them removed early than under the linear schedule
        assert widths[h] < annealed_k(h, k, j, steps)
    # h -> max_steps/2 is the linear schedule
    near = [annealed_k(s, k, j, steps, "log", 9990) for s in range(steps)]
    lin = [annealed_k(s, k, j, steps) for s in range(steps)]
    assert max(abs(a - b) for a, b in zip(near, lin)) <= 1


def test_log_schedule_config_rules():
    ok = dict(enabled=True, k=8, j=24, hard_k_annealing=True,
              hard_k_annealing_schedule="log", hard_k_annealing_half_step=100)
    assert ActivationBottleneckConfig(**ok).hard_k_annealing_half_step == 100.0
    for bad in (dict(hard_k_annealing_half_step=None),
                dict(hard_k_annealing_schedule="linear"),
                dict(hard_k_annealing=False),
                dict(hard_k_annealing_schedule="cosine"),
                dict(hard_k_annealing_half_step=0)):
        kw = dict(ok)
        kw.update(bad)
        with pytest.raises(ValueError):
            ActivationBottleneckConfig(**kw)
    cfg = Config()
    cfg.train.max_steps = 200
    with pytest.raises(ValueError):  # h must be below max_steps / 2
        Config(train=cfg.train, activation_bottleneck=ActivationBottleneckConfig(**ok))


def test_training_follows_the_log_schedule(tmp_path):
    data_dir = make_fake_dataset(tmp_path, 6000, 1200)
    cfg = annealing_config(data_dir, str(tmp_path / "runs_log"))
    cfg.activation_bottleneck.hard_k_annealing_schedule = "log"
    cfg.activation_bottleneck.hard_k_annealing_half_step = 2.0
    cfg = Config(model=cfg.model, data=cfg.data, train=cfg.train,
                 activation_bottleneck=ActivationBottleneckConfig(
                     **dataclasses.asdict(cfg.activation_bottleneck)))
    seen = {}

    def record(step, model, bottleneck, optimizer):
        seen[step] = {layer.gate.k for _, layer in bottleneck.layers}

    train(cfg, on_step=record)
    expected = {s: {annealed_k(s, 4, 5, 12, "log", 2.0)} for s in range(12)}
    assert seen == expected
    assert expected[0] == {9} and expected[11] == {4}
    model, _, ctl = load_for_inference(os.path.join(str(tmp_path / "runs_log"), "test",
                                                    "ckpt_step6.pt"))
    assert {layer.gate.k for _, layer in ctl.layers} == expected[5]


def test_power_schedule_endpoints_limits_and_anneal_length():
    k, j, n = 32, 224, 18_000
    for s0, g in ((1.0, 0.1), (10.0, 0.6), (10.0, -0.5)):
        widths = [annealed_k(s, k, j, n, "power", s0=s0, gamma=g) for s in range(20_000)]
        assert widths[0] == k + j and widths[n - 1] == k
        assert all(a >= b for a, b in zip(widths, widths[1:]))
        assert set(widths[n:]) == {k}  # K from step S on
    # gamma = 1 is the linear schedule, gamma -> 0 the log one at the same s_0
    for s0 in (1.0, 10.0):
        lin = [annealed_k(s, k, j, n) for s in range(n)]
        p1 = [annealed_k(s, k, j, n, "power", s0=s0, gamma=1.0) for s in range(n)]
        assert max(abs(a - b) for a, b in zip(p1, lin)) <= 1
        lg = [annealed_k(s, k, j, n, "log", s0=s0) for s in range(n)]
        p0 = [annealed_k(s, k, j, n, "power", s0=s0, gamma=1e-7) for s in range(n)]
        assert p0 == lg
    # smaller gamma removes more features early
    early = [annealed_k(500, k, j, n, "power", s0=10.0, gamma=g) for g in (0.1, 0.4, 0.6)]
    assert early[0] < early[1] < early[2]
    # the log schedule's two parameterizations agree
    h = 250
    assert ([annealed_k(s, k, j, n, "log", half_step=h) for s in range(0, n, 7)]
            == [annealed_k(s, k, j, n, "log", s0=h * h / (n - 2 * h)) for s in range(0, n, 7)])


def test_power_s0_and_steps_config_rules():
    base = dict(enabled=True, k=8, j=24, hard_k_annealing=True)
    ActivationBottleneckConfig(**base, hard_k_annealing_schedule="power",
                               hard_k_annealing_s0=10, hard_k_annealing_gamma=0.2,
                               hard_k_annealing_steps=90)
    ActivationBottleneckConfig(**base, hard_k_annealing_schedule="log", hard_k_annealing_s0=1)
    for bad in (dict(hard_k_annealing_schedule="power", hard_k_annealing_s0=10),
                dict(hard_k_annealing_schedule="power", hard_k_annealing_gamma=0.2),
                dict(hard_k_annealing_schedule="power", hard_k_annealing_s0=10,
                     hard_k_annealing_gamma=0.0),
                dict(hard_k_annealing_schedule="power", hard_k_annealing_s0=10,
                     hard_k_annealing_gamma=0.2, hard_k_annealing_half_step=5),
                dict(hard_k_annealing_schedule="log", hard_k_annealing_s0=1,
                     hard_k_annealing_half_step=5),
                dict(hard_k_annealing_schedule="linear", hard_k_annealing_s0=1),
                dict(hard_k_annealing_steps=0)):
        with pytest.raises(ValueError):
            ActivationBottleneckConfig(**base, **bad)
    with pytest.raises(ValueError):
        ActivationBottleneckConfig(enabled=True, k=8, j=24, hard_k_annealing_steps=50)
    cfg = Config()
    cfg.train.max_steps = 100
    for steps in (101, 20):  # beyond max_steps; below J+1
        with pytest.raises(ValueError):
            Config(train=cfg.train, activation_bottleneck=ActivationBottleneckConfig(
                **base, hard_k_annealing_steps=steps))


def power_config(data_dir, out_dir, validate_every):
    cfg = annealing_config(data_dir, out_dir)
    ab = dataclasses.asdict(cfg.activation_bottleneck)
    ab.update(hard_k_annealing_schedule="power", hard_k_annealing_s0=1.0,
              hard_k_annealing_gamma=0.4, hard_k_annealing_steps=8)
    cfg.train.validate_every_steps = validate_every
    cfg.train.checkpoint_every_steps = 12
    return Config(model=cfg.model, data=cfg.data, train=cfg.train,
                  activation_bottleneck=ActivationBottleneckConfig(**ab))


def test_validation_at_both_ends_is_logged_and_does_not_perturb_training(tmp_path):
    data_dir = make_fake_dataset(tmp_path, 6000, 1200)
    seen = {}

    def record(step, model, bottleneck, optimizer):
        seen[step] = bottleneck.active_count

    train(power_config(data_dir, str(tmp_path / "with_val"), 3), on_step=record)
    assert seen == {s: annealed_k(s, 4, 5, 8, "power", s0=1.0, gamma=0.4) for s in range(12)}
    assert seen[0] == 9 and all(seen[s] == 4 for s in range(8, 12))
    run = os.path.join(str(tmp_path / "with_val"), "test")
    val = [json.loads(l) for l in open(os.path.join(run, "metrics.jsonl"))]
    val = {r["step"]: r for r in val if "val/ce" in r}
    assert sorted(val) == [3, 6, 9, 12]
    for r in val.values():
        assert {"val_topk/ce", "val_topkj/ce"} <= set(r)
    # past S the run is at K, so its own validation IS the Top-K one
    assert val[12]["val_topk/ce"] == val[12]["val/ce"]
    # the extra passes restore K'(s): validating changes nothing in training
    train(power_config(data_dir, str(tmp_path / "no_val"), 0))
    a = torch.load(os.path.join(run, "ckpt_step12.pt"), weights_only=False)["model"]
    b = torch.load(os.path.join(str(tmp_path / "no_val"), "test", "ckpt_step12.pt"),
                   weights_only=False)["model"]
    assert a.keys() == b.keys() and all(torch.equal(a[n], b[n]) for n in a)
