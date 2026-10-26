"""hard_k_annealing: the hard width falls from K+J to K over the run."""
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
