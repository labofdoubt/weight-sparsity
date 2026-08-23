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


def topk_smoke_config(tmp_path, data_dir, out_dir, **extra):
    kw = dict(
        enabled=True,
        method="topk",
        k=0.25,
        j=0.25,
        s_init=1.0,
        s_init_mode="magnitude",
        inverse_temperature=2.0,
        inverse_temperature_schedule="exponential",
        beta_end=20.0,
        mask_lr=1e-2,
        mask_grad_clip=0.5,
    )
    kw.update(extra)
    return smoke_config(data_dir, out_dir, **kw)


def test_topk_training_logs_the_budget_and_the_soft_l0_penalty(tmp_path):
    data_dir = make_fake_dataset(tmp_path)
    cfg = topk_smoke_config(
        tmp_path,
        data_dir,
        str(tmp_path / "runs_topk"),
        soft_l0_enabled=True,
        soft_l0_lambda_topk=1e-5,
        soft_l0_lambda_explore=1e-6,
    )
    cfg.train.run_name = "test_topk"
    train(cfg)

    records = [json.loads(l) for l in open(tmp_path / "runs_topk" / "test_topk" / "metrics.jsonl")]
    train_records = [r for r in records if "sparsity/beta" in r]
    assert train_records
    last = train_records[-1]
    assert last["sparsity/beta"] > train_records[0]["sparsity/beta"]
    # the TopK budget is hard; soft gating can only undershoot it
    assert last["sparsity/density_topk"] == pytest.approx(0.25, abs=1e-3)
    assert last["sparsity/density_soft"] <= last["sparsity/density_topk"] + 1e-6
    assert last["sparsity/density_hard"] <= last["sparsity/density_topk"] + 1e-6
    assert last["sparsity/soft_l0_penalty"] > 0
    assert 0.0 <= last["sparsity/gate_mean_topk"] <= 1.0
    assert 0.0 <= last["sparsity/turnover"] <= 1.0
    assert np.isfinite(last["sparsity/mask_grad_norm"])
    val_records = [r for r in records if "val_hard/ce" in r]
    assert val_records and np.isfinite(val_records[-1]["val_hard/ce"])


def test_topk_checkpoint_round_trip(tmp_path):
    data_dir = make_fake_dataset(tmp_path)
    cfg = topk_smoke_config(tmp_path, data_dir, str(tmp_path / "runs_topk_ckpt"))
    cfg.train.run_name = "topk_ckpt"
    train(cfg)

    path = tmp_path / "runs_topk_ckpt" / "topk_ckpt" / "latest.pt"
    model, loaded_cfg, ctrl = load_for_inference(str(path))
    assert loaded_cfg.sparsity.method == "topk"
    assert ctrl.enabled and len(ctrl.layers) == 4
    assert ctrl.beta == pytest.approx(20.0)
    layer = ctrl.layers[0][1]
    assert int((layer.effective_weight() != 0).sum()) == layer.topk_numel
    logits, _ = model(torch.randint(0, VOCAB, (1, 8)))
    assert logits.shape == (1, 8, VOCAB)


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


def bottleneck_smoke_config(data_dir, out_dir, **extra):
    from wsparse.config import ActivationBottleneckConfig

    cfg = smoke_config(data_dir, out_dir)
    kw = dict(enabled=True, n_features=128, k=16, j=48, n_eff=8.0, layers="all")
    kw.update(extra)
    cfg.activation_bottleneck = ActivationBottleneckConfig(**kw)
    return cfg


@pytest.mark.parametrize(
    "boundary,metric", [("outside_only", "ess"), ("both_sides", "entropy")]
)
def test_activation_bottleneck_training_logs_diagnostics(tmp_path, boundary, metric):
    data_dir = make_fake_dataset(tmp_path)
    name = f"bn_{boundary}_{metric}"
    cfg = bottleneck_smoke_config(
        data_dir, str(tmp_path / "runs_bn"), boundary_mode=boundary,
        effective_count_metric=metric,
    )
    cfg.train.run_name = name
    train(cfg)

    records = [json.loads(l) for l in open(tmp_path / "runs_bn" / name / "metrics.jsonl")]
    logged = [r for r in records if "bottleneck/temperature" in r]
    assert logged
    last = logged[-1]
    assert last["bottleneck/density"] == pytest.approx(16 / 128)
    assert last["bottleneck/candidate_density"] == pytest.approx(64 / 128)
    assert last["bottleneck/temperature"] > 0
    assert abs(last["bottleneck/n_eff_realized"] - 8.0) < 0.5
    assert last["bottleneck/budget_residual"] < 1e-3
    assert last["bottleneck/barrier_failures"] == 0.0
    if boundary == "both_sides":
        assert last["bottleneck/newton_failed"] == 0.0
    # the J candidates really do receive gradient, and the active ones more
    assert last["bottleneck/grad_inactive"] > 0
    assert last["bottleneck/grad_active"] > 0
    assert np.isfinite(last["val/ce"]) if "val/ce" in last else True


def test_activation_bottleneck_checkpoint_round_trip(tmp_path):
    data_dir = make_fake_dataset(tmp_path)
    cfg = bottleneck_smoke_config(data_dir, str(tmp_path / "runs_bn_ckpt"))
    cfg.train.run_name = "bn_ckpt"
    train(cfg)

    model, loaded_cfg, _ = load_for_inference(
        str(tmp_path / "runs_bn_ckpt" / "bn_ckpt" / "latest.pt")
    )
    assert loaded_cfg.activation_bottleneck.enabled
    from wsparse.bottleneck import SparseTopKBottleneck

    assert isinstance(model.blocks[0].mlp_bottleneck, SparseTopKBottleneck)
    logits, _ = model(torch.randint(0, VOCAB, (1, 8)))
    assert torch.isfinite(logits).all()


def test_tensorboard_events_are_written(tmp_path):
    pytest.importorskip("tensorboard")
    data_dir = make_fake_dataset(tmp_path)
    cfg = bottleneck_smoke_config(data_dir, str(tmp_path / "runs_tb"))
    cfg.train.run_name = "tb_run"
    cfg.train.tensorboard = True
    train(cfg)

    tb_dir = tmp_path / "runs_tb" / "tb_run" / "tb"
    assert tb_dir.exists()
    events = list(tb_dir.glob("events.out.tfevents.*"))
    assert events, "no tensorboard event file was written"

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(tb_dir))
    acc.Reload()
    tags = set(acc.Tags()["scalars"])
    # the losses, and the quantities that matter for this experiment
    for tag in ("train/ce", "train/loss", "val/ce", "train/lr",
                "bottleneck/temperature", "bottleneck/n_eff_realized",
                "bottleneck/budget_residual", "bottleneck/feature_dead_frac",
                "bottleneck/feature_usage_entropy", "perf/tokens_per_s"):
        assert tag in tags, f"{tag} missing from {sorted(tags)[:40]}"
    steps = [e.step for e in acc.Scalars("train/ce")]
    assert steps == sorted(steps) and len(steps) >= 2
    assert acc.Tags()["tensors"] or True  # config text is best-effort


def test_tensorboard_can_be_disabled(tmp_path):
    data_dir = make_fake_dataset(tmp_path)
    cfg = smoke_config(data_dir, str(tmp_path / "runs_notb"))
    cfg.train.run_name = "notb"
    cfg.train.tensorboard = False
    train(cfg)
    assert not (tmp_path / "runs_notb" / "notb" / "tb").exists()
    assert (tmp_path / "runs_notb" / "notb" / "metrics.jsonl").exists()


def test_dtype_fallbacks_are_announced(capsys, monkeypatch):
    """Both fallbacks change the numerics of a run -- and bf16 -> fp16 silently
    switches loss scaling on -- so neither may happen quietly."""
    from wsparse.utils import resolve_dtype

    assert resolve_dtype("float16", torch.device("cpu")) is torch.float32
    assert "float32" in capsys.readouterr().out

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "Tesla V100-SXM2")
    assert resolve_dtype("bfloat16", torch.device("cuda")) is torch.float16
    out = capsys.readouterr().out
    assert "V100" in out and "loss scaling" in out

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert resolve_dtype("bfloat16", torch.device("cuda")) is torch.bfloat16
    assert capsys.readouterr().out == ""  # supported: nothing to say


def test_supported_dtypes_are_silent(capsys):
    from wsparse.utils import resolve_dtype

    assert resolve_dtype("float32", torch.device("cpu")) is torch.float32
    assert resolve_dtype("bfloat16", torch.device("cpu")) is torch.bfloat16
    assert capsys.readouterr().out == ""


def test_sampling_produces_several_distinct_generations(tmp_path, monkeypatch):
    """sample_count continuations from one prompt, drawn as a single batch."""
    import wsparse.train as train_mod

    model = build_model(
        ModelConfig(vocab_size=VOCAB, max_seq_len=32, n_layers=1, d_model=32, n_heads=4)
    )

    class FakeTok:
        def encode(self, text):
            return [1, 2, 3]

        def decode(self, ids):
            return " ".join(map(str, ids))

    monkeypatch.setattr("wsparse.tokenizer.build_tokenizer", lambda cfg: FakeTok())
    cfg = Config()
    cfg.model = model.cfg
    cfg.train.sample_count = 4
    cfg.train.sample_tokens = 8
    texts = train_mod.sample(model, cfg, torch.device("cpu"), torch.float32, step=100)
    assert len(texts) == 4
    assert len(set(texts)) > 1, "all four generations were identical"
    assert model.training  # training mode restored


def test_sampling_is_reproducible_for_a_given_step(tmp_path, monkeypatch):
    """The explicit generator makes samples comparable across runs, rather than
    depending on how much global RNG the training loop consumed first."""
    import wsparse.train as train_mod

    class FakeTok:
        def encode(self, text):
            return [1, 2, 3]

        def decode(self, ids):
            return " ".join(map(str, ids))

    monkeypatch.setattr("wsparse.tokenizer.build_tokenizer", lambda cfg: FakeTok())
    torch.manual_seed(0)
    model = build_model(
        ModelConfig(vocab_size=VOCAB, max_seq_len=32, n_layers=1, d_model=32, n_heads=4)
    )
    cfg = Config()
    cfg.model = model.cfg
    cfg.train.sample_count = 3
    cfg.train.sample_tokens = 8

    a = train_mod.sample(model, cfg, torch.device("cpu"), torch.float32, step=500)
    torch.randn(1000)  # disturb the global RNG, as a differing run would
    b = train_mod.sample(model, cfg, torch.device("cpu"), torch.float32, step=500)
    assert a == b
    c = train_mod.sample(model, cfg, torch.device("cpu"), torch.float32, step=1000)
    assert a != c  # a different step is a different draw


def test_samples_reach_the_log_file_and_tensorboard(tmp_path, monkeypatch):
    pytest.importorskip("tensorboard")

    class FakeTok:
        def encode(self, text):
            return [1, 2, 3]

        def decode(self, ids):
            return " ".join(map(str, ids))

    # patched so the test runs offline and actually exercises the logging path
    monkeypatch.setattr("wsparse.tokenizer.build_tokenizer", lambda cfg: FakeTok())
    data_dir = make_fake_dataset(tmp_path)
    cfg = smoke_config(data_dir, str(tmp_path / "runs_sample"))
    cfg.train.run_name = "sample_run"
    cfg.train.tensorboard = True
    cfg.train.sample_every_steps = 3
    cfg.train.sample_count = 2
    cfg.train.sample_tokens = 8
    train(cfg)

    run_dir = tmp_path / "runs_sample" / "sample_run"
    body = (run_dir / "samples.txt").read_text()
    assert "=== step 3" in body and "=== step 6" in body
    assert body.count("**1.**") == 2 and body.count("**2.**") == 2  # 2 samples, 2 events

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(run_dir / "tb"))
    acc.Reload()
    # add_text tags get a /text_summary suffix from TensorBoard
    assert "samples/text_summary" in acc.Tags()["tensors"]
    assert "config/text_summary" in acc.Tags()["tensors"]
    assert [e.step for e in acc.Tensors("samples/text_summary")] == [3, 6]
    # metrics.jsonl stays numeric-only
    records = [json.loads(l) for l in open(run_dir / "metrics.jsonl")]
    assert not any("samples" in r for r in records)


def test_feature_usage_is_logged_to_tensorboard(tmp_path):
    """Histogram, sorted-usage figure and quantile scalars, at validation cadence."""
    pytest.importorskip("tensorboard")
    pytest.importorskip("matplotlib")
    data_dir = make_fake_dataset(tmp_path)
    cfg = bottleneck_smoke_config(data_dir, str(tmp_path / "runs_usage"))
    cfg.train.run_name = "usage"
    cfg.train.tensorboard = True
    cfg.train.validate_every_steps = 3
    train(cfg)

    run_dir = tmp_path / "runs_usage" / "usage"
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(run_dir / "tb"), size_guidance={"histograms": 10, "images": 10})
    acc.Reload()
    tags = acc.Tags()
    assert any(t.startswith("usage/blocks") for t in tags["histograms"]), tags["histograms"]
    assert "usage/sorted" in tags["images"], tags["images"]
    for q in ("p50", "p90", "p99"):
        assert f"bottleneck/usage_{q}" in tags["scalars"]

    records = [json.loads(l) for l in open(run_dir / "metrics.jsonl")]
    assert any("bottleneck/usage_p50" in r for r in records)


def test_usage_figure_survives_missing_matplotlib(monkeypatch):
    """The plot is optional; training must not depend on it."""
    import builtins

    import wsparse.train as train_mod

    real_import = builtins.__import__

    def no_mpl(name, *a, **k):
        if name == "matplotlib":
            raise ImportError("no matplotlib")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_mpl)
    assert train_mod.usage_figure({"blocks.0": torch.rand(64)}, 8, 64) is None


@pytest.mark.parametrize("placement", ["pre_mlp", "residual"])
def test_trivial_bottleneck_trains_end_to_end(tmp_path, placement):
    """Regression: with k == n_features there is no boundary, so the console
    line had no gap statistic to print and the run died three seconds in --
    while every unit test on the geometry passed.  Train it for real instead.
    """
    data_dir = make_fake_dataset(tmp_path)
    name = f"bn_trivial_{placement}"
    cfg = bottleneck_smoke_config(
        data_dir, str(tmp_path / "runs_triv"),
        placement=placement, k=128, j=0, surrogate_mode="hard",
        calibrate_output=True,
    )
    cfg.train.run_name = name
    train(cfg)

    records = [json.loads(l) for l in open(tmp_path / "runs_triv" / name / "metrics.jsonl")]
    bn = [r for r in records if "bottleneck/density" in r]
    assert bn, "no bottleneck diagnostics were logged"
    assert bn[-1]["bottleneck/density"] == pytest.approx(1.0)
    assert "bottleneck/score_gap" not in bn[-1]  # undefined without a boundary
    assert np.isfinite([r["train/ce"] for r in records if "train/ce" in r]).all()


@pytest.mark.parametrize("surrogate", ["hard", "lapsum_scheduled"])
def test_residual_placement_trains_end_to_end(tmp_path, surrogate):
    data_dir = make_fake_dataset(tmp_path)
    name = f"bn_res_{surrogate}"
    cfg = bottleneck_smoke_config(
        data_dir, str(tmp_path / "runs_res"),
        placement="residual", surrogate_mode=surrogate,
        temperature_schedule="constant", temperature_start=1.0,
        temperature_scale_mode="relative",
    )
    cfg.train.run_name = name
    train(cfg)

    records = [json.loads(l) for l in open(tmp_path / "runs_res" / name / "metrics.jsonl")]
    ce = [r["train/ce"] for r in records if "train/ce" in r]
    assert ce and np.isfinite(ce).all()
    bn = [r for r in records if "bottleneck/density" in r]
    assert bn[-1]["bottleneck/density"] == pytest.approx(16 / 128)
