"""Exact neuron-swap intervention scans -> parquet for the Swap Interventions tab.

Two modes over the same fixed probe sequences (TokenStream, deterministic
offset, identical across checkpoints):

  checkpoint mode (default): walk a run's ckpt_step*.pt like
      extract_bottleneck_scores.py and measure DeltaL_{i->j} at each step.
  live mode (--live): the early-training analogue -- train from a config and
      probe the live model every --probe-every steps in [0, --steps], via
      train()'s on_step hook.  The measurement is pure inference plus
      torch.autograd.grad on gate outputs (parameter .grad never touched), so
      the run is not perturbed.

Outputs under <out-dir>/<name>/:
    meta.json                  probe corpus (token ids/strings), config, modes
    rows_step<S>.parquet       one row per exact swap (fields per the spec)
    lin_step<S>.npz            full first-order K x J matrix per context (f16)
    context_summary.parquet    per-context distribution stats + lin quality
    source_summary.parquet     per-source-rank stats over j
Summaries use ONLY random/exhaustive rows; tail rows are stored but labeled.
"""

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import torch

from wsparse.data import TokenStream
from wsparse.interventions import SwapInterventionEngine
from wsparse.train import load_for_inference


def checkpoint_step(path: str) -> int:
    m = re.search(r"ckpt_step(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def token_strings(ids):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        return [tok.decode([t]) for t in ids]
    except Exception:
        return [str(int(t)) for t in ids]


def fixed_positions(seq_len: int, n: int) -> list:
    return sorted(set(np.linspace(8, seq_len - 2, num=n, dtype=int).tolist()))


def scan_model(model, cfg, step, xb, yb, args, store):
    """All contexts for one model snapshot; appends rows/lin to `store`."""
    dev = str(next(model.parameters()).device)
    eng = SwapInterventionEngine(model, cfg, device=dev)
    if args.layers:
        keep = {int(v) for v in str(args.layers).split(",")}
        eng.layers = [l for l in eng.layers if l in keep]
        if not eng.layers:
            raise SystemExit(f"--layers {args.layers} matched no installed bottleneck")
    positions = store["positions"]
    vmode = None if args.value_mode == "auto" else args.value_mode
    for seq in range(xb.shape[0]):
        x, y = xb[seq: seq + 1], yb[seq: seq + 1]
        state = eng.capture(x, y)
        eng.capture_gradients(x, y, state)
        worst = max(eng.verify_baseline(state, y).values())
        if worst > 1e-4:
            print(f"[swap] WARNING step={step} seq={seq}: baseline restart "
                  f"mismatch {worst:.2e} -- results suspect")
        base_mean = float(state.ce[(y[0] != -100)].mean())
        for li in eng.layers:
            for t in positions:
                swaps = eng.select_pairs(
                    state, li, int(t), mode=args.pair_mode,
                    random_targets=args.random_targets, tail_best=args.tail_best,
                    tail_worst=args.tail_worst,
                    exhaustive_threshold=args.exhaustive_threshold,
                    seed=args.seed, seq_id=seq, value_mode=vmode)
                res = eng.evaluate(state, li, swaps, y, batch_size=args.batch_size)
                act, cand = eng.ranks_at(state, li, int(t))
                for s, r in zip(swaps, res):
                    store["rows"].append(dict(
                        checkpoint_step=step, sequence_id=seq,
                        sequence_length=int(xb.shape[1]),
                        bottleneck_id=f"blocks.{li}", layer_index=li,
                        token_index=int(t),
                        token_string=store["tok_strings"][seq][int(t)],
                        source_feature_id=s.source, source_rank=s.source_rank,
                        source_value=float(state.code[li][t, s.source]),
                        source_score=float(state.value[li][t, s.source].abs()
                                           if eng.selection == "abs_topk"
                                           else state.value[li][t, s.source]),
                        target_feature_id=s.target,
                        target_candidate_rank=s.target_rank,
                        target_global_rank=eng.k + s.target_rank,
                        target_score=float(state.value[li][t, s.target].abs()
                                           if eng.selection == "abs_topk"
                                           else state.value[li][t, s.target]),
                        target_original_value=0.0,
                        swap_value=s.value,
                        swap_value_mode=vmode or (
                            "target_sign_source_magnitude"
                            if eng.selection == "abs_topk" else "copy_signed_source"),
                        baseline_loss_mean=base_mean,
                        swapped_loss_mean=r["swapped_loss_mean"],
                        delta_loss_mean=r["delta_loss_mean"],
                        delta_nll_total=r["delta_nll_total"],
                        delta_loss_linearized=s.lin,
                        sample_kind=s.kind,
                        pair_sampling_probability=(
                            1.0 if s.kind == "exhaustive"
                            else min(1.0, args.random_targets / eng.j)),
                        prefix_delta_max_abs=r["prefix_delta_max_abs"],
                        downstream_support_mode="recompute",
                        K=eng.k, J=eng.j,
                    ))
                store["lin"][f"s{seq}_l{li}_t{int(t)}"] = \
                    eng.lin_matrix(state, li, int(t), vmode).astype(np.float16)
        print(f"[swap] step={step} seq={seq}: {len(store['rows'])} rows total")


QS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def summarize(name_dir: str) -> None:
    files = sorted(glob.glob(os.path.join(name_dir, "rows_step*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    unb = df[df.sample_kind.isin(["random", "exhaustive"])]
    keys = ["checkpoint_step", "sequence_id", "layer_index", "token_index"]
    ctx = []
    for kv, g in unb.groupby(keys):
        d = g.delta_loss_mean
        row = dict(zip(keys, kv))
        row.update(n_pairs=len(g), coverage=len(g) / (g.K.iat[0] * g.J.iat[0]),
                   exhaustive=bool((g.sample_kind == "exhaustive").all()),
                   mean=d.mean(), std=d.std(), median=d.median(),
                   mad=(d - d.median()).abs().median(),
                   frac_beneficial=float((d < 0).mean()),
                   min=d.min(), max=d.max())
        for q in QS:
            row[f"q{int(q*100):02d}"] = d.quantile(q)
        fin = g.dropna(subset=["delta_loss_linearized"])
        if len(fin) > 2:
            row["lin_pearson"] = fin.delta_loss_mean.corr(fin.delta_loss_linearized)
            row["lin_spearman"] = fin.delta_loss_mean.corr(
                fin.delta_loss_linearized, method="spearman")
            row["lin_mae"] = float((fin.delta_loss_mean
                                    - fin.delta_loss_linearized).abs().mean())
        ctx.append(row)
    pd.DataFrame(ctx).to_parquet(os.path.join(name_dir, "context_summary.parquet"))
    src = []
    for kv, g in unb.groupby(keys + ["source_rank"]):
        d = g.delta_loss_mean
        row = dict(zip(keys + ["source_rank"], kv))
        row.update(n=len(g), median=d.median(), mean=d.mean(), min=d.min(),
                   q05=d.quantile(0.05), q95=d.quantile(0.95),
                   frac_beneficial=float((d < 0).mean()))
        src.append(row)
    pd.DataFrame(src).to_parquet(os.path.join(name_dir, "source_summary.parquet"))
    print(f"[swap] summaries: {len(ctx)} contexts, {len(src)} source rows")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", help="run dir with ckpt_step*.pt (checkpoint mode)")
    ap.add_argument("--live-config", help="config.json to train from (live mode)")
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--steps", type=int, default=1000, help="live: train this far")
    ap.add_argument("--probe-every", type=int, default=250)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--sequences", type=int, default=2)
    ap.add_argument("--out-dir", default="/workspace/analysis/swaps")
    ap.add_argument("--name", default=None)
    ap.add_argument("--positions", type=int, default=12)
    ap.add_argument("--pair-mode", default="hybrid",
                    choices=["hybrid", "sampled", "exhaustive"])
    ap.add_argument("--random-targets", type=int, default=8)
    ap.add_argument("--tail-best", type=int, default=32)
    ap.add_argument("--tail-worst", type=int, default=32)
    ap.add_argument("--exhaustive-threshold", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--value-mode", default="auto",
                    choices=["auto", "copy_signed_source",
                             "target_sign_source_magnitude"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layers", default=None,
                    help="comma-separated layer indices to scan (default: all)")
    ap.add_argument("--max-ckpts", type=int, default=None)
    args = ap.parse_args()

    if bool(args.ckpt_dir) == bool(args.live_config):
        raise SystemExit("exactly one of --ckpt-dir / --live-config")

    if args.ckpt_dir:
        run = os.path.basename(os.path.normpath(args.ckpt_dir))
    else:
        from wsparse.config import load_config
        cfg_live = load_config(args.live_config, list(args.overrides))
        run = cfg_live.train.run_name
    name = args.name or f"swap_{run}"
    name_dir = os.path.join(args.out_dir, name)
    os.makedirs(name_dir, exist_ok=True)

    def write_step(store, step):
        rows = pd.DataFrame(store["rows"])
        rows.to_parquet(os.path.join(name_dir, f"rows_step{step}.parquet"))
        np.savez_compressed(os.path.join(name_dir, f"lin_step{step}.npz"),
                            **store["lin"])
        store["rows"], store["lin"] = [], {}

    def make_store(cfg, xb):
        seq_len = xb.shape[1]
        positions = fixed_positions(seq_len, args.positions)
        toks = [token_strings(xb[s].tolist()) for s in range(xb.shape[0])]
        meta = dict(run=run, name=name, mode="ckpt" if args.ckpt_dir else "live",
                    k=int(cfg.activation_bottleneck.k),
                    j=int(cfg.activation_bottleneck.j),
                    selection_mode=cfg.activation_bottleneck.selection_mode,
                    positions=[int(p) for p in positions],
                    pair_mode=args.pair_mode, random_targets=args.random_targets,
                    tail_best=args.tail_best, tail_worst=args.tail_worst,
                    seed=args.seed, offset=args.offset, split=args.split,
                    sequences=[dict(sequence_id=s,
                                    token_ids=[int(t) for t in xb[s].tolist()],
                                    token_strings=toks[s])
                               for s in range(xb.shape[0])],
                    downstream_support_mode="recompute")
        with open(os.path.join(name_dir, "meta.json"), "w") as f:
            json.dump(meta, f)
        return dict(rows=[], lin={}, positions=positions, tok_strings=toks)

    device = torch.device(args.device)
    if args.ckpt_dir:
        ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "ckpt_step*.pt")),
                       key=checkpoint_step)
        if args.max_ckpts:
            ckpts = ckpts[:: max(1, len(ckpts) // args.max_ckpts)][: args.max_ckpts]
        if not ckpts:
            raise SystemExit(f"no ckpt_step*.pt under {args.ckpt_dir}")
        model, cfg, _ = load_for_inference(ckpts[0], device=str(device))
        stream = TokenStream(os.path.join(args.data_dir, f"{args.split}.bin"),
                             int(cfg.data.seq_len), seed=0)
        xb, yb = stream.batch(args.sequences, device,
                              deterministic_offset=args.offset)
        store = make_store(cfg, xb.cpu())
        for path in ckpts:
            step = checkpoint_step(path)
            model, cfg, _ = load_for_inference(path, device=str(device))
            scan_model(model, cfg, step, xb, yb, args, store)
            write_step(store, step)
            del model
            torch.cuda.empty_cache()
    else:
        from wsparse.train import train
        cfg_live.train.run_name = f"_swaplive_{run}"
        cfg_live.train.out_dir = os.path.join("/tmp", f"swaplive_{run}")
        cfg_live.train.tensorboard = False
        stream = TokenStream(os.path.join(args.data_dir, f"{args.split}.bin"),
                             int(cfg_live.data.seq_len), seed=0)
        xb, yb = stream.batch(args.sequences, device,
                              deterministic_offset=args.offset)
        store = make_store(cfg_live, xb.cpu())
        probe_at = set(range(0, args.steps + 1, args.probe_every))

        class StopProbing(Exception):
            pass

        def on_step(step, model, bottleneck, optimizer):
            if step in probe_at:
                scan_model(model, cfg_live, step, xb, yb, args, store)
                write_step(store, step)
            if step >= args.steps:
                raise StopProbing()

        try:
            train(cfg_live, on_step=on_step)
        except StopProbing:
            pass
    summarize(name_dir)
    print(f"[swap] wrote {name_dir}")


if __name__ == "__main__":
    main()
