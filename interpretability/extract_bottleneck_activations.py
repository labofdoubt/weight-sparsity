"""Cache bottleneck activations at each example's target token (spec section 11).

Captures, for one fixed bottleneck:

    x      the vector entering the bottleneck
    z      the sparse code that actually reaches the decoder
    x_hat  the bottleneck's output, including any output scaling

``z`` is taken from the gate's output, i.e. the coefficients that survive TopK
and participate in the forward pass -- not pre-TopK scores, not encoder logits,
not gradients, not decoder outputs (spec section 2).

    python interpretability/extract_bottleneck_activations.py \
        --checkpoint runs/bn_pmlp_k32j64_rel/latest.pt \
        --data-dir /workspace/data/tinystories \
        --benchmark benchmark_data --out activations/bn_pmlp_k32j64_rel.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wsparse.bottleneck.controller import _PLACEMENT_ATTR, parse_placements  # noqa: E402
from wsparse.train import load_for_inference  # noqa: E402

SIGNED_MODES = {"abs_topk", "gated_topk"}


def choose_layer(cfg, model) -> Dict:
    """The installed bottleneck nearest 50% depth -- fixed, never tuned.

    Chosen before any evaluation runs and independent of benchmark scores, so
    that the same relative layer is used for every model variant.
    """
    bn = cfg.activation_bottleneck
    placements = parse_placements(bn.placement)
    depth = cfg.model.n_layers
    target = depth // 2
    installed = [
        i
        for i, block in enumerate(model.blocks)
        if any(
            not isinstance(getattr(block, _PLACEMENT_ATTR[p]), torch.nn.Identity)
            for p in placements
        )
    ]
    if not installed:
        raise ValueError("this checkpoint has no activation bottleneck installed")
    layer = min(installed, key=lambda i: (abs(i - target), i))
    return dict(
        evaluation_layer=int(layer),
        model_depth=int(depth),
        placements=placements,
        target_depth_fraction=0.5,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--benchmark", default="benchmark_data")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--placement",
        default=None,
        help="which bottleneck to evaluate when a model has several per block",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-stories", type=int, default=16)
    args = ap.parse_args()

    model, cfg, _ = load_for_inference(args.checkpoint, device=args.device)
    model.eval()

    info = choose_layer(cfg, model)
    placements = info.pop("placements")
    if args.placement:
        if args.placement not in placements:
            raise ValueError(f"{args.placement!r} not installed; have {placements}")
        location = args.placement
    elif len(placements) == 1:
        location = placements[0]
    else:
        raise ValueError(
            f"this model has several bottlenecks per block ({placements}); "
            "pass --placement to say which one to evaluate"
        )

    block = model.blocks[info["evaluation_layer"]]
    module = getattr(block, _PLACEMENT_ATTR[location])
    bn = cfg.activation_bottleneck
    info.update(
        bottleneck_location=location,
        bottleneck_width=int(bn.n_features),
        K=int(bn.k),
        J=int(bn.j),
        selection_mode=bn.selection_mode,
        surrogate_mode=bn.surrogate_mode,
        signed=bool(bn.selection_mode in SIGNED_MODES),
        checkpoint=os.path.abspath(args.checkpoint),
        run_name=cfg.train.run_name,
    )
    print(
        f"[extract] layer {info['evaluation_layer']}/{info['model_depth']} "
        f"({location}) N={info['bottleneck_width']} K={info['K']} "
        f"signed={info['signed']}"
    )

    # --- what to evaluate ------------------------------------------------- #
    frames = {
        s: pd.read_parquet(os.path.join(args.benchmark, f"{s}.parquet"))
        for s in ("train", "validation", "test")
    }
    everything = pd.concat(frames.values(), ignore_index=True)
    keys = (
        everything[["story_id", "target_token_index"]]
        .drop_duplicates()
        .sort_values(["story_id", "target_token_index"])
        .reset_index(drop=True)
    )
    key_index = {
        (int(r.story_id), int(r.target_token_index)): i for i, r in keys.iterrows()
    }
    print(f"[extract] {len(everything):,} examples -> {len(keys):,} distinct positions")

    index = json.load(open(os.path.join(args.benchmark, "split_story_ids.json")))
    spans = {int(k): v for k, v in index["story_spans"].items()}
    tokens = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")

    # --- hooks ------------------------------------------------------------- #
    grab: Dict[str, torch.Tensor] = {}
    handles = [
        module.register_forward_pre_hook(lambda m, i: grab.__setitem__("x", i[0].detach())),
        module.register_forward_hook(lambda m, i, o: grab.__setitem__("x_hat", o.detach())),
        module.gate.register_forward_hook(lambda m, i, o: grab.__setitem__("z", o.detach())),
    ]

    d_model, width = cfg.model.d_model, info["bottleneck_width"]
    X = np.zeros((len(keys), d_model), dtype=np.float32)
    Z = np.zeros((len(keys), width), dtype=np.float32)
    XH = np.zeros((len(keys), d_model), dtype=np.float32)

    by_story: Dict[int, List[int]] = {}
    for (sid, pos) in key_index:
        by_story.setdefault(sid, []).append(pos)
    story_ids = sorted(by_story)
    max_len = cfg.model.max_seq_len

    with torch.no_grad():
        for start in range(0, len(story_ids), args.batch_stories):
            chunk = story_ids[start : start + args.batch_stories]
            seqs = []
            for sid in chunk:
                a, b = spans[sid]
                seqs.append(np.asarray(tokens[a:b], dtype=np.int64)[:max_len])
            width_b = max(len(s) for s in seqs)
            batch = np.zeros((len(chunk), width_b), dtype=np.int64)
            for r, s in enumerate(seqs):
                batch[r, : len(s)] = s  # right padding only: causal, so it cannot
                batch[r, len(s) :] = s[-1]  # affect any earlier position
            out = model(torch.from_numpy(batch).to(args.device))
            del out
            for r, sid in enumerate(chunk):
                for pos in by_story[sid]:
                    if pos >= len(seqs[r]):
                        raise ValueError(f"story {sid} position {pos} beyond truncation")
                    row = key_index[(sid, pos)]
                    X[row] = grab["x"][r, pos].float().cpu().numpy()
                    Z[row] = grab["z"][r, pos].float().cpu().numpy()
                    XH[row] = grab["x_hat"][r, pos].float().cpu().numpy()
            if (start // args.batch_stories) % 50 == 0:
                print(f"[extract] {start:>6}/{len(story_ids)} stories", flush=True)

    for h in handles:
        h.remove()

    scale = float(module.output_scale) if module.output_scale is not None else 1.0
    nonzero = (Z != 0).sum(1)
    info.update(
        n_positions=int(len(keys)),
        mean_active=float(nonzero.mean()),
        max_active=int(nonzero.max()),
        output_scale=scale,
        dead_features=int((np.abs(Z).max(0) == 0).sum()),
    )
    print(
        f"[extract] active per token: mean {info['mean_active']:.1f} "
        f"max {info['max_active']} (K={info['K']}); "
        f"{info['dead_features']} features never active"
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        x=X.astype(np.float32),
        z=Z.astype(np.float32),
        x_hat=XH.astype(np.float32),
        story_id=keys.story_id.to_numpy(np.int64),
        target_token_index=keys.target_token_index.to_numpy(np.int64),
        W_dec=module.out_proj.weight.detach().float().cpu().numpy(),
        b_dec=(
            module.out_proj.bias.detach().float().cpu().numpy()
            if module.out_proj.bias is not None
            else np.zeros(d_model, dtype=np.float32)
        ),
        output_scale=np.float32(scale),
        meta=json.dumps(info),
    )
    print(f"[extract] wrote {args.out}")


if __name__ == "__main__":
    main()
