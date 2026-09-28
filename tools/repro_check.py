"""Bit-exactness oracle for the repo cleanup.

Runs a small but real training (tiny transformer + MD + bottleneck on
TinyStories, fp32, single GPU) in one of three modes -- hard | lapsum
(constant T) | rbk (rblapsum through_rank_kappa) -- and digests everything
that defines the computation:

  - the per-step training CE (hex-encoded doubles, log_every=1) and every
    validation CE: any forward/backward/update deviation anywhere in the run
    shows up here and in everything after it,
  - logits + loss on a fixed probe batch at initialization (step 0),
  - after the final step (via the saved checkpoint): every named parameter,
    the MD optimizer's per-matrix state (raw_grow / raw_gcol / c_f / Adam
    moments), and the probe forward again.

Two code versions are equivalent iff the artifacts match bit for bit.
Schema-adaptive (probes the config dataclass), so the same script runs on the
pre- and post-cleanup code.

  python tools/repro_check.py --mode rbk --out ref_rbk.json
  python tools/repro_check.py --compare ref_rbk.json new_rbk.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def hexd(x: float) -> str:
    return struct.pack("<d", float(x)).hex()


def compare(pa: str, pb: str) -> None:
    a, b = json.load(open(pa)), json.load(open(pb))
    skip = {"schema", "config_note"}
    keys = sorted((set(a) | set(b)) - skip)
    bad = [k for k in keys if a.get(k) != b.get(k)]
    if bad:
        print("MISMATCH in:", bad)
        for k in bad[:5]:
            va, vb = a.get(k), b.get(k)
            if isinstance(va, list) and isinstance(vb, list):
                n = 0
                for i, (x, y) in enumerate(zip(va, vb)):
                    if x != y:
                        print(f"  {k}[{i}]: {x} != {y}")
                        n += 1
                        if n >= 2:
                            break
                if len(va) != len(vb):
                    print(f"  {k}: length {len(va)} != {len(vb)}")
            elif isinstance(va, dict) and isinstance(vb, dict):
                for kk in sorted(set(va) | set(vb)):
                    if va.get(kk) != vb.get(kk):
                        print(f"  {k}.{kk}: {str(va.get(kk))[:40]} != "
                              f"{str(vb.get(kk))[:40]}")
                        break
            else:
                print(f"  {k}: {str(va)[:60]} != {str(vb)[:60]}")
        sys.exit(1)
    print(f"IDENTICAL ({len(keys)} fields, modes {a['mode']}/{b['mode']}, "
          f"schemas {a['schema']}/{b['schema']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hard", "lapsum", "rbk"])
    ap.add_argument("--data-dir", default="/workspace/data/tinystories")
    ap.add_argument("--out", default="")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
        return

    from wsparse.config import load_config
    from wsparse.train import train
    from wsparse.data import build_streams

    torch.use_deterministic_algorithms(True, warn_only=True)
    run_dir = f"/tmp/repro/repro_{args.mode}"
    shutil.rmtree(run_dir, ignore_errors=True)

    ov = {
        "model.n_layers": 4, "model.d_model": 256, "model.n_heads": 4,
        "model.max_seq_len": 128, "model.pos_encoding": "rope",
        "model.decouple": "true", "model.logit_scale": "none",
        "model.bias": "false", "model.mlp_activation": "gelu",
        "data.data_dir": args.data_dir, "data.seq_len": 128,
        "activation_bottleneck.enabled": "true",
        "activation_bottleneck.layers": "all",
        "activation_bottleneck.placement": "residual_out",
        "activation_bottleneck.n_features": 512,
        "activation_bottleneck.k": 16, "activation_bottleneck.j": 16,
        "activation_bottleneck.selection_mode": "abs_topk",
        "activation_bottleneck.post_norm": "true",
        "activation_bottleneck.bias": "false",
        "activation_bottleneck.init_mode": "sqrt_k_selection_corrected",
        "activation_bottleneck.log_diagnostics": "true",
        "train.device": "cuda", "train.dtype": "float32",
        "train.seed": 1337, "train.batch_size": 8,
        "train.micro_batch_size": 8, "train.max_steps": args.steps,
        "train.lr": 3e-4, "train.warmup_steps": 10,
        "train.log_every_steps": 1, "train.validate_every_steps": 20,
        "train.val_batches": 4,
        "train.checkpoint_every_steps": args.steps,  # final step only
        "train.keep_last_checkpoints": 1,
        "train.sample_every_steps": 0, "train.tensorboard": "false",
        "train.out_dir": "/tmp/repro", "train.run_name": f"repro_{args.mode}",
    }

    probe_cfg = load_config(None, [])
    new_schema = hasattr(probe_cfg.activation_bottleneck, "temperature")
    if args.mode == "hard":
        ov["activation_bottleneck.surrogate_mode"] = "hard"
    elif args.mode == "lapsum":
        ov["activation_bottleneck.temperature" if new_schema else
           "activation_bottleneck.fixed_temperature"] = 2.0
        ov["activation_bottleneck.surrogate_mode"] = (
            "lapsum" if new_schema else "lapsum_fixed")
        if not new_schema:
            ov["activation_bottleneck.temperature_scale_mode"] = "absolute"
    else:
        ov["activation_bottleneck.surrogate_mode"] = "rblapsum"
        ov["activation_bottleneck.rblapsum_boundary_grad_mode"] = "through_rank_kappa"
        ov["activation_bottleneck.temperature" if new_schema else
           "activation_bottleneck.rblapsum_temperature"] = 1.0

    cfg = load_config(None, [f"--{k}={v}" for k, v in ov.items()])

    losses = []
    import wsparse.utils as U
    orig_log = U.Logger.log

    def tap(self, step, metrics, console=""):
        if "train/ce" in metrics:
            losses.append(f"{step}:{hexd(metrics['train/ce'])}")
        if "val/ce" in metrics:
            losses.append(f"{step}:val:{hexd(metrics['val/ce'])}")
        orig_log(self, step, metrics, console)

    U.Logger.log = tap
    probe = {}

    def on_step(step, model, bottleneck, optimizer):
        if step != 0:
            return
        device = next(model.parameters()).device
        _, vs = build_streams(cfg.data, seed=999)
        x, y = vs.batch(4, device, deterministic_offset=777)
        was = model.training
        model.eval()
        with torch.no_grad():
            logits, l0 = model(x, y)
        model.train(was)
        probe["batch"] = (x.cpu(), y.cpu())
        probe["logits0"] = sha(logits)
        probe["loss0"] = hexd(float(l0))

    try:
        train(cfg, on_step=on_step)
    finally:
        U.Logger.log = orig_log

    # final state from the last-step checkpoint
    from wsparse.config import config_from_dict
    from wsparse.model import build_model
    from wsparse.bottleneck import apply_activation_bottleneck

    payload = torch.load(os.path.join(run_dir, "latest.pt"),
                         map_location="cpu", weights_only=False)
    cfg2 = config_from_dict(payload["config"])
    model = build_model(cfg2.model)
    apply_activation_bottleneck(model, cfg2.activation_bottleneck,
                                max_steps=cfg2.train.max_steps)
    model.load_state_dict(payload["model"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    x, y = (t.to(device) for t in probe["batch"])
    with torch.no_grad():
        logits, lF = model(x, y)

    params = {n: sha(p) for n, p in sorted(model.named_parameters())}
    opt_state = payload["optimizer"]["state"]
    opt_digest = {}
    for idx in sorted(opt_state, key=str):
        st = opt_state[idx]
        for field in sorted(st):
            v = st[field]
            if torch.is_tensor(v):
                opt_digest[f"{idx}.{field}"] = sha(v)

    out = {
        "schema": "new" if new_schema else "old",
        "mode": args.mode,
        "final_step": int(payload["step"]),
        "losses": losses,
        "probe_logits_step0": probe["logits0"],
        "probe_loss_step0": probe["loss0"],
        "probe_logits_final": sha(logits),
        "probe_loss_final": hexd(float(lF)),
        "params": params,
        "optimizer": opt_digest,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", args.out, "| losses:", len(losses),
          "| params:", len(params), "| opt fields:", len(opt_digest))


if __name__ == "__main__":
    main()
