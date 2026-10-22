import json
import os

import pytest
import torch

from toy_mnist.dust import dust_step_kind
from toy_mnist.train import build_optimizer, lr_at, train
from toy_mnist.tests.helpers import small_config, synthetic_data


def _read_metrics(run_dir):
    with open(os.path.join(run_dir, "metrics.jsonl")) as f:
        return [json.loads(line) for line in f]


@pytest.mark.parametrize("name", ["baseline", "rescale", "topk_hard"])
def test_plain_runs_finish_and_write_outputs(tmp_path, name):
    cfg = small_config(tmp_path, name)
    data = synthetic_data()
    summary = train(cfg, data)
    run_dir = summary["run_dir"]
    assert summary["status"] == "finished" and summary["steps"] == 6
    assert os.path.exists(os.path.join(run_dir, "summary.json"))
    assert os.path.exists(os.path.join(run_dir, "config.json"))
    assert os.path.exists(os.path.join(run_dir, "final.pt"))
    recs = _read_metrics(run_dir)
    train_recs = [r for r in recs if "train/ce" in r]
    val_recs = [r for r in recs if "val/ce" in r]
    assert [r["step"] for r in train_recs] == [1, 2, 4, 6]
    assert [r["step"] for r in val_recs] == [3, 6]
    assert summary["final_val_ce"] == val_recs[-1]["val/ce"]
    if name == "topk_hard":
        assert "gate/features_seen" in train_recs[0] and "rescale/g_std" in train_recs[0]
    payload = torch.load(os.path.join(run_dir, "final.pt"), weights_only=False)
    assert payload["step"] == 6 and ("dust_state" in payload) is False


def test_lr_schedule_and_optimizer_groups():
    from toy_mnist.config import load_config
    from toy_mnist.model import ToyMLP
    c = load_config(values={"rescale": {"enabled": True}, "model": {"hidden": 4},
                            "train": {"warmup_steps": 10, "max_steps": 110, "lr": 1.0,
                                      "min_lr_ratio": 0.1, "weight_decay": 0.5}})
    assert lr_at(0, c.train) == pytest.approx(0.1) and lr_at(9, c.train) == pytest.approx(1.0)
    assert lr_at(109, c.train) == pytest.approx(0.1, abs=1e-3)
    opt = build_optimizer(ToyMLP(c), c.train)
    by_name = {g["name"]: g for g in opt.param_groups}
    assert by_name["decay"]["weight_decay"] == 0.5 and by_name["nodecay"]["weight_decay"] == 0.0
    assert sum(p.numel() for p in by_name["decay"]["params"]) == 784 * 4 + 4 * 10
    assert sum(p.numel() for p in by_name["nodecay"]["params"]) == 784 + 4 + 10


@pytest.mark.parametrize("name", ["topk_dust", "topk_dust_backprop"])
def test_dust_runs_alternate_and_move_g_only_on_dust_steps(tmp_path, name):
    cfg = small_config(tmp_path, name, topk={"dust_population": 4, "dust_chunk_draws": 2,
                                             "dust_cosine_every_steps": 1})
    data = synthetic_data()
    snaps = {}

    def on_step(step, model, optimizer):
        snaps[step] = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if k in ("fc1.weight", "rescale.log_scale")}

    summary = train(cfg, data, on_step=on_step)
    assert summary["status"] == "finished" and summary["dust_steps"] == 3
    for step in range(1, 6):
        kind = dust_step_kind(step - 1, cfg.topk.dust_schedule)
        fc1_moved = not torch.equal(snaps[step]["fc1.weight"], snaps[step - 1]["fc1.weight"])
        g_moved = not torch.equal(snaps[step]["rescale.log_scale"],
                                  snaps[step - 1]["rescale.log_scale"])
        assert g_moved, step
        assert fc1_moved == (kind == "hard"), (step, kind)
    recs = _read_metrics(summary["run_dir"])
    train_recs = [r for r in recs if "train/ce" in r]
    # window at step 2 holds steps 1 (hard) and 2 (dust): one DUST step
    assert train_recs[1]["dust/steps"] == 1.0 and "dust/cosine" in train_recs[1]
    assert train_recs[0]["dust/steps"] == 0.0 and "dust/ce_gap" not in train_recs[0]
    assert train_recs[1]["dust/lr"] == pytest.approx(train_recs[1]["train/lr"])
    payload = torch.load(os.path.join(summary["run_dir"], "final.pt"), weights_only=False)
    assert payload["dust_state"]["dust_steps"] == 3


def test_separate_optimizer_and_lr_scale(tmp_path):
    cfg = small_config(tmp_path, "topk_dust", topk={"dust_population": 4, "dust_chunk_draws": 4,
                                                    "dust_optimizer": "separate",
                                                    "dust_lr_scale": 0.25})
    data = synthetic_data()
    summary = train(cfg, data)
    payload = torch.load(os.path.join(summary["run_dir"], "final.pt"), weights_only=False)
    main_steps = {int(s["step"]) for s in payload["optimizer"]["state"].values()}
    dust_steps = {int(s["step"]) for s in payload["dust_optimizer"]["state"].values()}
    assert main_steps == {3}          # the three hard steps
    assert dust_steps == {3}          # the three DUST steps, g only
    assert len(payload["dust_optimizer"]["state"]) == 1
    recs = [r for r in _read_metrics(summary["run_dir"]) if "dust/lr" in r]
    for r in recs:
        assert r["dust/lr"] == pytest.approx(0.25 * r["train/lr"])


def test_cache_is_written_atomically_and_a_damaged_cache_is_rebuilt(tmp_path, monkeypatch):
    import numpy as np
    from toy_mnist import data as D
    rng = np.random.default_rng(0)
    arrays = {"train_x": rng.integers(0, 256, (60000, 28, 28), dtype=np.uint8),
              "train_y": rng.integers(0, 10, 60000).astype(np.uint8),
              "test_x": rng.integers(0, 256, (10000, 28, 28), dtype=np.uint8),
              "test_y": rng.integers(0, 10, 10000).astype(np.uint8)}
    names = {"train_images": "train_x", "train_labels": "train_y",
             "test_images": "test_x", "test_labels": "test_y"}
    monkeypatch.setattr(D, "download", lambda d: {k: k for k in names})
    monkeypatch.setattr(D, "parse_idx", lambda key: arrays[names[key]])
    cache = tmp_path / D.CACHE
    cache.write_bytes(b"PK\x03\x04 partial")  # a half-written file
    out = D.load_arrays(str(tmp_path))
    assert np.array_equal(out["train_y"], arrays["train_y"])
    with np.load(cache) as z:
        assert np.array_equal(z["test_x"], arrays["test_x"])
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
