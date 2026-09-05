"""Cache the bottleneck *ranking scores* across checkpoints, for the score explorer.

Unlike ``extract_bottleneck_activations.py``, which caches the post-TopK code
``z`` at one target token, this walks every checkpoint of a run and stores the
**pre-TopK score** the gate ranks on, for every bottleneck, every sequence in a
fixed batch and every token position in a fixed window.

The score is exactly what the gate sorts::

    r = a        (selection_mode="topk")
    r = |a|      (selection_mode="abs_topk")
    r = |s|      (selection_mode="gated_topk", s from the score branch)

so it is captured as a forward *pre*-hook on ``mod.gate``: its first positional
argument is the ranking signal in every selection mode.  Signed values are kept
(the viewer takes the magnitude) because the sign is not recoverable later and
is worth having for analysis.

Bands follow the gate: sorted by ``r`` descending, ranks ``[0, k)`` are the
TopK that survive the forward, ``[k, k+j)`` are the J candidates that receive
surrogate gradient, and the remainder get exactly zero gradient from the gate.

    python analysis/extract_bottleneck_scores.py \
        --ckpt-dir /workspace/ckpt/dc_rout_soft_k32_j32 \
        --data-dir /workspace/data/tinystories \
        --out-dir /workspace/analysis/scores

Output, one per run:

    <out-dir>/<run>.npy     float32 (n_ckpt, n_layer, batch, n_pos, n_features)
    <out-dir>/<run>.json    steps, layer labels, k, j, token ids, val CE

The array is written through ``open_memmap`` so peak memory stays at one
checkpoint's worth, and the viewer reads it back memory-mapped and slices only
the cell it draws.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wsparse.bottleneck.controller import _PLACEMENT_ATTR, parse_placements  # noqa: E402
from wsparse.data import TokenStream  # noqa: E402
from wsparse.train import load_for_inference  # noqa: E402


def checkpoint_step(path: str) -> int:
    m = re.search(r"ckpt_step(\d+)\.pt$", os.path.basename(path))
    if not m:
        raise ValueError(f"cannot read a step number from {path!r}")
    return int(m.group(1))


def find_bottlenecks(model, cfg):
    """``[(label, module), ...]`` in the order the controller installs them.

    Mirrors ``ActivationBottleneckController._install`` rather than trusting the
    run name: a checkpoint rebuilt from its own config is the only reliable
    statement of which placements are actually present.
    """
    placements = parse_placements(cfg.activation_bottleneck.placement)
    found = []
    for i, block in enumerate(model.blocks):
        for name in placements:
            mod = getattr(block, _PLACEMENT_ATTR[name], None)
            if mod is None or isinstance(mod, torch.nn.Identity):
                continue
            label = f"blocks.{i}" if len(placements) == 1 else f"blocks.{i}.{name}"
            found.append((label, mod))
    if not found:
        raise ValueError("this checkpoint has no activation bottleneck installed")
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", required=True, help="a run directory holding ckpt_step*.pt")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch", type=int, default=8, help="sequences in the fixed batch")
    ap.add_argument("--positions", type=int, default=128, help="token positions kept, from 0")
    ap.add_argument("--offset", type=int, default=0, help="deterministic batch offset in val.bin")
    ap.add_argument("--split", default="val", choices=("val", "train"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    run = os.path.basename(args.ckpt_dir.rstrip("/"))
    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "ckpt_step*.pt")), key=checkpoint_step)
    if not ckpts:
        raise SystemExit(f"no ckpt_step*.pt under {args.ckpt_dir}")
    steps = [checkpoint_step(p) for p in ckpts]
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- the fixed batch ---------------------------------------------------- #
    # Built once and reused for every checkpoint and every run, so that a change
    # between two cells of the viewer is a change in the model, never in the
    # data.  deterministic_offset makes it independent of the RNG seed.
    model, cfg, _ = load_for_inference(ckpts[0], device=str(device))
    seq_len = int(cfg.data.seq_len)
    n_pos = min(int(args.positions), seq_len)
    stream = TokenStream(os.path.join(args.data_dir, f"{args.split}.bin"), seq_len, seed=0)
    idx, targets = stream.batch(args.batch, device, deterministic_offset=args.offset)

    bottlenecks = find_bottlenecks(model, cfg)
    labels = [lbl for lbl, _ in bottlenecks]
    n_feat = int(cfg.activation_bottleneck.n_features)
    k = int(cfg.activation_bottleneck.k)
    j = int(cfg.activation_bottleneck.j)
    sel = cfg.activation_bottleneck.selection_mode

    print(f"run={run}  ckpts={len(ckpts)}  bottlenecks={len(labels)}  "
          f"n_features={n_feat}  k={k}  j={j}  selection={sel}")
    print(f"batch={args.batch}  seq_len={seq_len}  positions kept={n_pos}")

    out_path = os.path.join(args.out_dir, f"{run}.npy")
    arr = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(ckpts), len(labels), int(args.batch), n_pos, n_feat),
    )

    val_ce = []
    for ci, path in enumerate(ckpts):
        if ci > 0:  # the first is already loaded
            model, cfg, _ = load_for_inference(path, device=str(device))
            bottlenecks = find_bottlenecks(model, cfg)
        model.eval()

        # The gate's first positional argument is the ranking signal in every
        # selection mode, so this one hook is placement- and mode-agnostic.
        grabbed: dict = {}
        handles = []

        def make_hook(li):
            def hook(module, inputs):
                grabbed[li] = inputs[0].detach()
                return None
            return hook

        for li, (_, mod) in enumerate(bottlenecks):
            handles.append(mod.gate.register_forward_pre_hook(make_hook(li)))

        with torch.no_grad():
            _, loss = model(idx, targets)
        for h in handles:
            h.remove()

        if len(grabbed) != len(bottlenecks):
            raise RuntimeError(
                f"{path}: captured {len(grabbed)} of {len(bottlenecks)} bottlenecks"
            )
        for li in range(len(bottlenecks)):
            a = grabbed[li]
            if a.shape != (args.batch, seq_len, n_feat):
                raise RuntimeError(f"{path} layer {li}: unexpected score shape {tuple(a.shape)}")
            arr[ci, li] = a[:, :n_pos, :].float().cpu().numpy()

        val_ce.append(float(loss))
        print(f"  [{ci + 1}/{len(ckpts)}] step {steps[ci]:>6}  batch CE {float(loss):.4f}")
        grabbed.clear()
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    arr.flush()

    meta = dict(
        run=run,
        steps=steps,
        layer_labels=labels,
        n_features=n_feat,
        k=k,
        j=j,
        selection_mode=sel,
        placement=cfg.activation_bottleneck.placement,
        surrogate_mode=cfg.activation_bottleneck.surrogate_mode,
        # Enough of the temperature configuration to reconstruct the LapSum
        # barrier offline.  The gate's own `scheduled_temperature` buffer is
        # registered persistent=False, so it is *not* in the checkpoint -- the
        # schedule has to be re-evaluated from these fields at the right step.
        temperature=dict(
            scale_mode=cfg.activation_bottleneck.temperature_scale_mode,
            schedule=cfg.activation_bottleneck.temperature_schedule,
            start=cfg.activation_bottleneck.temperature_start,
            end=cfg.activation_bottleneck.temperature_end,
            warmup_steps=cfg.activation_bottleneck.temperature_warmup_steps,
            anneal_steps=cfg.activation_bottleneck.temperature_anneal_steps,
            power=cfg.activation_bottleneck.temperature_power,
            fixed=cfg.activation_bottleneck.fixed_temperature,
            max_steps=int(cfg.train.max_steps),
        ),
        n_layers=int(cfg.model.n_layers),
        batch=int(args.batch),
        n_pos=n_pos,
        seq_len=seq_len,
        split=args.split,
        offset=int(args.offset),
        token_ids=idx[:, :n_pos].cpu().numpy().tolist(),
        batch_ce=val_ce,
        array=os.path.basename(out_path),
        array_shape=list(arr.shape),
        note=(
            "scores are the signed pre-TopK ranking signal; rank by |value| when "
            f"selection_mode={sel!r}. Bands: [0,k) TopK, [k,k+j) J candidates, rest zero-grad."
        ),
    )
    with open(os.path.join(args.out_dir, f"{run}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {out_path} {arr.shape} and {run}.json")


if __name__ == "__main__":
    main()
