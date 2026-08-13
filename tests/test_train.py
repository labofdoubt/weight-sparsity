import json
import os

import numpy as np
import pytest
import torch

from wsparse.config import Config, ModelConfig, SparsityConfig
from wsparse.data import TokenStream, load_meta
from wsparse.optim import lr_at
from wsparse.train import evaluate, load_for_inference, train
from wsparse.model import build_model


VOCAB = 64


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
        json.dump(
            {"vocab_size": VOCAB, "eos_id": 0, "tokenizer": "fake", "train_tokens": n_train}, f
        )
    return str(d)


def smoke_config(data_dir, out_dir, **sparsity_kw):
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
    cfg.train.out_dir = out_dir
    cfg.train.run_name = "test"
    if sparsity_kw:
        cfg.sparsity = SparsityConfig(**sparsity_kw)
    return cfg


def test_token_stream_shapes_and_shift(tmp_path):
    data_dir = make_fake_dataset(tmp_path, 5000, 1000)
    stream = TokenStream(os.path.join(data_dir, "train.bin"), seq_len=16, seed=0)
    x, y = stream.batch(4, torch.device("cpu"))
    assert x.shape == (4, 16) and y.shape == (4, 16)
    assert x.dtype == torch.long
    assert torch.equal(x[:, 1:], y[:, :-1])  # y is x shifted by one
    assert load_meta(data_dir)["vocab_size"] == VOCAB


def test_token_stream_deterministic_offsets_are_reproducible(tmp_path):
    data_dir = make_fake_dataset(tmp_path, 5000, 1000)
    stream = TokenStream(os.path.join(data_dir, "val.bin"), seq_len=16, seed=0)
    a, _ = stream.batch(3, torch.device("cpu"), deterministic_offset=7)
    b, _ = stream.batch(3, torch.device("cpu"), deterministic_offset=7)
    assert torch.equal(a, b)


def test_lr_schedule_shape():
    cfg = Config().train
    cfg.lr = 1e-3
    cfg.warmup_steps = 100
    cfg.max_steps = 1000
    cfg.min_lr_ratio = 0.1
    assert lr_at(0, cfg) == pytest.approx(1e-5)
    assert lr_at(99, cfg) == pytest.approx(1e-3)
    assert lr_at(999, cfg) == pytest.approx(1e-4, rel=1e-2)
    assert lr_at(500, cfg) < lr_at(200, cfg)
    cfg.lr_schedule = "constant"
    assert lr_at(999, cfg) == pytest.approx(1e-3)


def test_evaluate_returns_finite_metrics(tmp_path):
    data_dir = make_fake_dataset(tmp_path, 5000, 2000)
    model = build_model(ModelConfig(vocab_size=VOCAB, max_seq_len=32, n_layers=1, d_model=32, n_heads=4))
    stream = TokenStream(os.path.join(data_dir, "val.bin"), seq_len=16)
    out = evaluate(model, stream, 2, 2, torch.device("cpu"), torch.float32)
    assert np.isfinite(out["ce"]) and out["ppl"] > 1


def test_dense_training_runs_and_checkpoints(tmp_path):
    data_dir = make_fake_dataset(tmp_path)
    cfg = smoke_config(data_dir, str(tmp_path / "runs"))
    summary = train(cfg)
    run_dir = tmp_path / "runs" / "test"
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "latest.pt").exists()
    assert np.isfinite(summary["best_val_ce"])
    lines = [json.loads(l) for l in open(run_dir / "metrics.jsonl")]
    assert any("train/ce" in r for r in lines)
    assert any("val/ce" in r for r in lines)


@pytest.mark.parametrize("method", ["ltp", "cs"])
def test_sparse_training_logs_beta_and_density(tmp_path, method):
    data_dir = make_fake_dataset(tmp_path)
    cfg = smoke_config(
        data_dir,
        str(tmp_path / f"runs_{method}"),
        enabled=True,
        method=method,
        beta_start=1.0 if method == "cs" else 1e3,
        beta_end=50.0 if method == "cs" else 1e5,
        l0_coef=0.1,
        mask_lr=1e-3,
    )
    cfg.train.run_name = f"test_{method}"
    train(cfg)
    run_dir = tmp_path / f"runs_{method}" / f"test_{method}"
    records = [json.loads(l) for l in open(run_dir / "metrics.jsonl")]
    train_records = [r for r in records if "sparsity/beta" in r]
    assert train_records, "no sparsity metrics were logged"
    last = train_records[-1]
    assert last["sparsity/beta"] > train_records[0]["sparsity/beta"]
    assert 0.0 <= last["sparsity/density_soft"] <= 1.0
    assert 0.0 <= last["sparsity/density_hard"] <= 1.0
    assert "sparsity/l0_penalty" in last
    val_records = [r for r in records if "val_hard/ce" in r]
    assert val_records and np.isfinite(val_records[-1]["val_hard/ce"])


def test_checkpoint_round_trip_preserves_masks(tmp_path):
    data_dir = make_fake_dataset(tmp_path)
    cfg = smoke_config(
        data_dir,
        str(tmp_path / "runs_ckpt"),
        enabled=True,
        method="cs",
        beta_start=1.0,
        beta_end=20.0,
        l0_coef=0.1,
    )
    cfg.train.run_name = "ckpt"
    train(cfg)
    path = tmp_path / "runs_ckpt" / "ckpt" / "latest.pt"
    model, loaded_cfg, ctrl = load_for_inference(str(path))
    assert loaded_cfg.sparsity.method == "cs"
    assert ctrl.enabled and len(ctrl.layers) == 4
    assert ctrl.beta == pytest.approx(20.0)
    logits, _ = model(torch.randint(0, VOCAB, (1, 8)))
    assert logits.shape == (1, 8, VOCAB)
    assert 0.0 <= ctrl.stats()["sparsity/density_hard"] <= 1.0


def test_resume_continues_from_saved_step(tmp_path):
    data_dir = make_fake_dataset(tmp_path)
    cfg = smoke_config(data_dir, str(tmp_path / "runs_resume"))
    cfg.train.run_name = "resume"
    train(cfg)
    cfg2 = smoke_config(data_dir, str(tmp_path / "runs_resume"))
    cfg2.train.run_name = "resume"
    cfg2.train.max_steps = 9
    cfg2.train.resume = "auto"
    train(cfg2)
    records = [json.loads(l) for l in open(tmp_path / "runs_resume" / "resume" / "metrics.jsonl")]
    assert max(r["step"] for r in records) == 9
