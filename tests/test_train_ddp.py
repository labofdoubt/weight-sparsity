"""train() after the DDP refactor: single-process behaviour, the should_stop
hook, final_val_batches, the guard's flag path, and a real 2-process gloo run.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from wsparse.config import load_config
from wsparse.train import train

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_data(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    rng = np.random.default_rng(0)
    for name, n in (("train", 20000), ("val", 4000)):
        rng.integers(0, 256, size=n).astype(np.uint16).tofile(d / f"{name}.bin")
    (d / "meta.json").write_text(json.dumps(
        {"dataset": "synthetic", "tokenizer": "gpt_neo", "vocab_size": 256,
         "eos_id": 0, "train_tokens": 20000, "val_tokens": 4000}))
    return str(d)


def tiny_overrides(data_dir, out_dir, **extra):
    ov = {
        "model.n_layers": 2, "model.d_model": 32, "model.n_heads": 2,
        "model.max_seq_len": 16, "model.pos_encoding": "rope",
        "model.decouple": "true", "model.logit_scale": "none",
        "model.bias": "false",
        "data.data_dir": data_dir, "data.seq_len": 16,
        "activation_bottleneck.enabled": "true",
        "activation_bottleneck.n_features": 64,
        "activation_bottleneck.k": 4, "activation_bottleneck.j": 4,
        "activation_bottleneck.selection_mode": "abs_topk",
        "activation_bottleneck.surrogate_mode": "rblapsum",
        "activation_bottleneck.rblapsum_boundary_grad_mode": "through_rank_kappa",
        "activation_bottleneck.post_norm": "true",
        "train.device": "cpu", "train.dtype": "float32",
        "train.batch_size": 4, "train.micro_batch_size": 2,
        "train.max_steps": 6,
        "train.log_every_steps": 2, "train.validate_every_steps": 3,
        "train.val_batches": 2, "train.checkpoint_every_steps": 0,
        "train.sample_every_steps": 0, "train.tensorboard": "false",
        "train.out_dir": out_dir, "train.run_name": "tiny",
    }
    ov.update(extra)
    return [f"--{k}={v}" for k, v in ov.items()]


def test_single_process_end_to_end(tmp_path):
    cfg = load_config(None, tiny_overrides(make_data(tmp_path), str(tmp_path / "runs")))
    summary = train(cfg)
    assert "val/ce" in summary and np.isfinite(summary["val/ce"])
    run_dir = tmp_path / "runs" / "tiny"
    assert (run_dir / "summary.json").exists()
    rows = [json.loads(l) for l in open(run_dir / "metrics.jsonl")]
    assert any("train/ce" in r for r in rows)
    assert "stopped_at" not in summary


def test_should_stop_ends_run_and_records_reason(tmp_path):
    cfg = load_config(None, tiny_overrides(make_data(tmp_path), str(tmp_path / "runs")))
    calls = {"n": 0}

    def stopper():
        calls["n"] += 1
        return "test stop" if calls["n"] >= 3 else None

    summary = train(cfg, should_stop=stopper)
    assert summary["stopped_at"] == 3
    assert summary["stopped_reason"] == "test stop"


def test_final_val_batches(tmp_path):
    cfg = load_config(None, tiny_overrides(
        make_data(tmp_path), str(tmp_path / "runs"),
        **{"train.final_val_batches": 4}))
    summary = train(cfg)
    assert "val_final/ce" in summary and np.isfinite(summary["val_final/ce"])


def test_guard_stop_step_subprocess(tmp_path):
    data = make_data(tmp_path)
    cfg = load_config(None, tiny_overrides(data, str(tmp_path / "runs")))
    cfg_path = str(tmp_path / "tiny.yaml")
    cfg.dump(cfg_path)
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "train_guard.py"),
         "--config", cfg_path, "--stop-step", "3"],
        capture_output=True, text=True, timeout=600,
        env={**os.environ, "PYTHONPATH": os.path.join(REPO, "src")})
    assert r.returncode == 0, r.stderr[-2000:]
    run_dir = tmp_path / "runs" / "tiny"
    assert (run_dir / "stopped.json").exists(), r.stdout[-2000:]
    stopped = json.loads((run_dir / "stopped.json").read_text())
    assert stopped["stop_step"] == 3
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["stopped_at"] <= 5


@pytest.mark.skipif(os.environ.get("WSPARSE_DDP_TEST") != "1",
                    reason="spawns torchrun; set WSPARSE_DDP_TEST=1 (run on the training box)")
@pytest.mark.skipif(not torch.distributed.is_available(), reason="no torch.distributed")
def test_two_process_gloo_ddp(tmp_path):
    data = make_data(tmp_path)
    out = str(tmp_path / "runs")
    runner = tmp_path / "runner.py"
    runner.write_text(
        "import sys\n"
        f"sys.path.insert(0, {os.path.join(REPO, 'src')!r})\n"
        "from wsparse.config import load_config\n"
        "from wsparse.train import train\n"
        f"cfg = load_config(None, {tiny_overrides(data, out)!r})\n"
        "s = train(cfg)\n"
        "print('RANK-DONE', s.get('val/ce'))\n")
    r = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone",
         "--nproc_per_node", "2", str(runner)],
        capture_output=True, text=True, timeout=900,
        env={**os.environ, "PYTHONPATH": os.path.join(REPO, "src")})
    assert r.returncode == 0, (r.stdout[-1500:], r.stderr[-1500:])
    run_dir = tmp_path / "runs" / "tiny"
    summary = json.loads((run_dir / "summary.json").read_text())
    assert np.isfinite(summary["val/ce"])
    # exactly one writer: rank 0's metrics, parseable end to end
    rows = [json.loads(l) for l in open(run_dir / "metrics.jsonl")]
    steps = [row["step"] for row in rows if "train/ce" in row]
    assert steps == sorted(steps) and len(steps) == len(set(steps))
