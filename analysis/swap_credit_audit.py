"""Offline counterfactual hard-swap audit of RBLapSum support credit.

At each checkpoint: capture the surrogate's final score-space support term
``h`` on a fixed probe batch (train mode, dropout off, the checkpoint's own
backward), sample active/candidate pairs from rank windows at fixed token
positions, replace feature ``i`` by candidate ``j`` with the candidate's own
signed pre-gate value at that one token and gate, re-run everything
downstream, and compare the replacement pressure ``r_ij = h_i - h_j`` with
the loss change ``dL_ij = L(y^{i->j}) - L(y)``.  The balanced decision error

    E_credit = 1/2 [ P(r <= 0 | dL < -tau) + P(r > 0 | dL > tau) ]

is reported per layer and with fixed layer weights, undefined when a class
is empty, with confidence intervals from resampling whole sequence windows.

    python analysis/swap_credit_audit.py --ckpt <run>/ckpt_step2000.pt --ckpt <run>/ckpt_step20000.pt \
        --data-dir /workspace/data/tinystories --layers 1,3,5 --n-positions 12 \
        --active-window 8 --cand-window 8 --pairs-per-token 16 --out-dir /workspace/analysis/swap_audit/<run>

tau is calibrated from identity replays (every (layer, seq, pos) replayed
through the swap machinery with i = j) as ``--tau-factor`` times the largest
|dL| they produce, unless ``--tau`` is given.  ``--suffix`` evaluates swaps
through the cached stream suffix after checking it against the full-forward
oracle on ``--verify-suffix`` swaps per layer (a mismatch above the tolerance
is an error); code-residual checkpoints always use the oracle.

A control scorer is reported beside the support term on the same swaps and
the same dL: the first-order prediction of the swap from the hard value
gradient alone, ``r_linear = z_i g_i - z_j g_j`` with ``g = dL/dy`` at the
gate output (dense, captured in the same backward); positive again means the
swap is predicted to lower the loss.  ``--rescore`` adds it to a finished
audit from its saved rows (one capture per checkpoint, dL reused).

Outputs under ``<out-dir>/step<S>/``: ``rows.parquet`` (one row per swap),
``meta.json`` (configuration and provenance), ``summary.json`` (class
counts, per-layer and aggregate errors with bootstrap CIs); and
``<out-dir>/credit_vs_step.json`` across the given checkpoints.  With
``--backup <rclone dest>`` every checkpoint's outputs are copied as soon as
they are written.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wsparse.data import TokenStream  # noqa: E402
from wsparse.swap_audit import (Swap, aggregate_layers, baseline_codes, bootstrap_credit,  # noqa: E402
                                capture_support, check_auditable, credit_error, find_bottlenecks,
                                sample_pairs, swap_deltas, swap_losses)
from wsparse.train import load_for_inference  # noqa: E402

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def checkpoint_step(path: str) -> int:
    m = re.search(r"ckpt_step(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def fixed_positions(seq_len: int, n: int) -> list:
    return sorted(set(np.linspace(8, seq_len - 2, num=n, dtype=int).tolist()))


def add_linear_control(df: pd.DataFrame, cap: dict) -> pd.DataFrame:
    """``g_i``, ``g_j`` (dL/dy at the gate output) and ``r_linear = z_i g_i - z_j g_j``."""
    gi = np.zeros(len(df)); gj = np.zeros(len(df))
    for n, r in enumerate(df.itertuples()):
        g = cap[int(r.layer)]["g_y"]
        gi[n] = float(g[int(r.seq), int(r.pos), int(r.i)]); gj[n] = float(g[int(r.seq), int(r.pos), int(r.j)])
    df = df.copy(); df["g_i"] = gi; df["g_j"] = gj
    df["r_linear"] = df["z_i"] * df["g_i"] - df["z_j"] * df["g_j"]
    return df


def score(df: pd.DataFrame, layers, weights, tau: float, n_boot: int, seed: int, key: str) -> dict:
    per_layer = {int(l): credit_error(df.loc[df["layer"] == l, key].to_numpy(),
                                      df.loc[df["layer"] == l, "dL"].to_numpy(), tau) for l in layers}
    E = aggregate_layers(per_layer, weights)
    boot = (bootstrap_credit(df["seq"].to_numpy(), df["layer"].to_numpy(), df[key].to_numpy(),
                             df["dL"].to_numpy(), layers, weights, tau, n_boot, seed) if n_boot else None)
    return dict(pressure=key, per_layer=per_layer, E=E, bootstrap=boot)


def write_outputs(od: str, df: pd.DataFrame, meta: dict, layers, weights, tau: float, n_boot: int, seed: int):
    sup = score(df, layers, weights, tau, n_boot, seed, "r")
    lin = score(df, layers, weights, tau, n_boot, seed, "r_linear") if "r_linear" in df else None
    df.to_parquet(os.path.join(od, "rows.parquet"), index=False)
    with open(os.path.join(od, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    summary = dict(step=meta["step"], tau=tau, n_rows=int(len(df)), classes=df["cls"].value_counts().to_dict(),
                   support=sup, linear=lin,
                   # backward-compatible top-level fields for the support scorer
                   per_layer=sup["per_layer"], E=sup["E"], bootstrap=sup["bootstrap"])
    with open(os.path.join(od, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return sup, lin


def fmt(sc: dict) -> str:
    if sc is None:
        return "n/a"
    b = sc["bootstrap"]
    return (f"{sc['E'] if sc['E'] is None else round(sc['E'], 4)}"
            + (f" CI [{b['aggregate']['lo'] if b['aggregate']['lo'] is None else round(b['aggregate']['lo'], 4)}, "
               f"{b['aggregate']['hi'] if b['aggregate']['hi'] is None else round(b['aggregate']['hi'], 4)}]"
               f" ({b['aggregate']['n_undefined']} undef)" if b else "")
            + "  per layer " + " ".join(f"L{l}:{v['E'] if v['E'] is None else round(v['E'], 3)}" for l, v in sc["per_layer"].items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", required=True, help="checkpoint(s), repeatable")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="val", choices=("val", "train"))
    ap.add_argument("--offset", type=int, default=0, help="deterministic window offset")
    ap.add_argument("--sequences", type=int, default=8, help="probe windows (sequences)")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--positions", default=None, help="comma-separated token positions")
    ap.add_argument("--n-positions", type=int, default=12, help="evenly spaced positions if --positions is absent")
    ap.add_argument("--active-window", type=int, default=8, help="active ranks [K-w, K)")
    ap.add_argument("--cand-window", type=int, default=8, help="candidate ranks [K, K+w)")
    ap.add_argument("--pairs-per-token", type=int, default=16)
    ap.add_argument("--exhaustive", action="store_true", help="every pair of the windows")
    ap.add_argument("--seed", type=int, default=1234, help="pair-sampling and bootstrap seed")
    ap.add_argument("--dtype", default="float32", choices=list(DTYPES))
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--tau-factor", type=float, default=4.0,
                    help="tau = max(tau_factor x paired replay noise, tau_floor)")
    ap.add_argument("--tau-floor", type=float, default=1e-7, help="the fp32 resolution of a sequence-mean CE")
    ap.add_argument("--swap-batch", type=int, default=32)
    ap.add_argument("--layer-weights", default=None, help="comma-separated, one per layer; default uniform")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--suffix", action="store_true")
    ap.add_argument("--verify-suffix", type=int, default=64, help="swaps per layer checked against the oracle")
    ap.add_argument("--suffix-tol", type=float, default=1e-5,
                    help="max |L_suffix - L_oracle| allowed beyond the identity-replay noise (tau_factor x "
                         "the largest identity-replay |dL|): batch-composition rounding can flip a near-tied "
                         "downstream selection in either path, which is what the replays measure")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--backup", default=None, help="rclone destination for the outputs")
    ap.add_argument("--rescore", action="store_true",
                    help="add the linear control to a finished audit in --out-dir: reuse rows.parquet and dL, "
                         "recapture the gradients, rewrite summaries")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    os.makedirs(args.out_dir, exist_ok=True)
    ckpts = sorted(args.ckpt, key=checkpoint_step)
    stream = None
    vs_step = []
    for path in ckpts:
        step = checkpoint_step(path)
        t0 = time.time()
        model, cfg, _ = load_for_inference(path, device=str(device))
        prov = check_auditable(cfg)
        mods = find_bottlenecks(model, cfg)
        K, J = prov["k"], prov["j"]
        layers = list(range(len(mods))) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
        weights = ({l: float(w) for l, w in zip(layers, args.layer_weights.split(","))}
                   if args.layer_weights else {l: 1.0 for l in layers})
        if len(weights) != len(layers):
            raise SystemExit("--layer-weights must have one entry per layer")
        seq_len = int(cfg.data.seq_len)
        if stream is None:
            stream = TokenStream(os.path.join(args.data_dir, f"{args.split}.bin"), seq_len, seed=0)
            idx, targets = stream.batch(args.sequences, device, deterministic_offset=args.offset)
        positions = ([int(p) for p in args.positions.split(",")] if args.positions
                     else fixed_positions(seq_len, args.n_positions))
        use_suffix = args.suffix and not prov["code_residual"]
        if args.suffix and prov["code_residual"]:
            print(f"step {step}: code_residual checkpoint, --suffix ignored, using the oracle")

        # ---- support term (train mode, the checkpoint's own backward) ---- #
        cap = capture_support(model, cfg, mods, idx, targets, layers, dtype=dtype)

        if args.rescore:
            od = os.path.join(args.out_dir, f"step{step}")
            meta = json.load(open(os.path.join(od, "meta.json")))
            df = pd.read_parquet(os.path.join(od, "rows.parquet"))
            if meta["probe"]["positions"] != positions or meta["probe"]["batch_shape"] != list(idx.shape):
                raise SystemExit(f"step {step}: the saved audit used other positions or windows than this invocation")
            for l in layers:
                sub = df[df["layer"] == l]
                h = cap[l]["h"]
                hi = np.array([float(h[int(r.seq), int(r.pos), int(r.rank_i)]) for r in sub.itertuples()])
                if not np.allclose(hi, sub["h_i"].to_numpy(), rtol=1e-4, atol=1e-9):
                    raise SystemExit(f"step {step} layer {l}: the recaptured support term differs from the saved one")
            df = add_linear_control(df, cap)
            meta["linear_control"] = "r_linear = z_i g_i - z_j g_j, g = dL/dy at the gate output, same backward as h"
            tau = float(meta["tau"])
            sup, lin = write_outputs(od, df, meta, layers, weights, tau, args.bootstrap, args.seed)
            vs_step.append(dict(step=step, E=sup["E"], E_linear=lin["E"], ci=sup["bootstrap"]["aggregate"] if sup["bootstrap"] else None,
                                ci_linear=lin["bootstrap"]["aggregate"] if lin["bootstrap"] else None,
                                per_layer={l: v["E"] for l, v in sup["per_layer"].items()},
                                per_layer_linear={l: v["E"] for l, v in lin["per_layer"].items()}, tau=tau))
            with open(os.path.join(args.out_dir, "credit_vs_step.json"), "w") as fh:
                json.dump(vs_step, fh, indent=2)
            print(f"step {step:>6} (rescored): support {fmt(sup)}\n                      linear  {fmt(lin)}")
            if args.backup:
                subprocess.run(["rclone", "copy", args.out_dir, args.backup], check=False)
            del model
            torch.cuda.empty_cache() if device.type == "cuda" else None
            continue
        # ---- baseline codes and losses (eval mode, hook-free) ------------- #
        z_eval, y_eval, L0 = baseline_codes(model, mods, idx, targets, layers, dtype=dtype)
        for l in layers:
            if not torch.equal((y_eval[l] != 0), (cap[l]["y"] != 0)):
                raise RuntimeError(f"step {step} layer {l}: the train-mode capture and the eval-mode "
                                   "baseline disagree on the active support (baseline-restart mismatch)")

        # ---- pairs ------------------------------------------------------- #
        rows, ident = [], []
        for l in layers:
            z, h, cand = cap[l]["z"], cap[l]["h"], cap[l]["cand_idx"]
            for b in range(idx.shape[0]):
                for t in positions:
                    pairs = sample_pairs(cand[b, t], K, args.active_window, args.cand_window,
                                         args.pairs_per_token, args.seed, l, b, t, args.exhaustive)
                    zi_id = int(cand[b, t, K - 1])
                    ident.append((l, Swap(b, t, zi_id, zi_id, float(z[b, t, zi_id]), float(z[b, t, zi_id]))))
                    for ri, rj, fi, fj in pairs:
                        rows.append(dict(step=step, layer=l, seq=b, pos=t, rank_i=ri, rank_j=rj, i=fi, j=fj,
                                         z_i=float(z[b, t, fi]), z_j=float(z[b, t, fj]),
                                         h_i=float(h[b, t, ri]), h_j=float(h[b, t, rj])))
        df = pd.DataFrame(rows)
        df["r"] = df["h_i"] - df["h_j"]

        # ---- paired evaluation: every batch carries unswapped copies of its
        # sequences, dL is row against row, and the two unswapped copies of a
        # sequence in one batch measure the replay noise that is left ------- #
        dL = np.zeros(len(df)); noise_all = []
        for l in layers:
            m = (df["layer"] == l).to_numpy()
            sw = [Swap(int(r.seq), int(r.pos), int(r.i), int(r.j), r.z_i, r.z_j) for r in df[m].itertuples()]
            d, noise = swap_deltas(model, mods, l, idx, targets, sw, args.swap_batch, dtype,
                                   suffix=use_suffix, baseline_code=y_eval[l] if use_suffix else None, n_identity=2)
            dL[m] = d; noise_all.append(noise)
        noise_all = np.concatenate(noise_all) if noise_all else np.zeros(0)
        tau_replay = float(noise_all.max()) if noise_all.size else 0.0
        # cross-batch reference: the unpaired identity replay against the hook-free baseline (reported only)
        tau_src = {}
        for l in layers:
            sw = [s for ll, s in ident if ll == l]
            Lid = swap_losses(model, mods, l, idx, targets, sw, args.swap_batch, dtype,
                              suffix=use_suffix, baseline_code=y_eval[l] if use_suffix else None)
            tau_src[l] = float(np.abs(Lid - L0[[s.seq for s in sw]]).max())
        suffix_check = None
        if use_suffix and args.verify_suffix > 0:
            suffix_check = {}
            for l in layers:
                sub = df[df["layer"] == l].head(args.verify_suffix)
                sw = [Swap(int(r.seq), int(r.pos), int(r.i), int(r.j), r.z_i, r.z_j) for r in sub.itertuples()]
                a, _ = swap_deltas(model, mods, l, idx, targets, sw, args.swap_batch, dtype, suffix=False)
                b, _ = swap_deltas(model, mods, l, idx, targets, sw, args.swap_batch, dtype, suffix=True,
                                   baseline_code=y_eval[l])
                suffix_check[l] = float(np.abs(a - b).max())
                allowed = args.suffix_tol + args.tau_factor * max(tau_replay, max(tau_src.values()))
                if suffix_check[l] > allowed:
                    raise RuntimeError(f"step {step} layer {l}: suffix vs oracle differ by {suffix_check[l]:.3e} > {allowed:.3e}")
        tau = args.tau if args.tau is not None else max(args.tau_factor * tau_replay, args.tau_floor)
        df["dL"] = dL
        df["cls"] = np.where(df["dL"] < -tau, "improve", np.where(df["dL"] > tau, "harm", "dead"))
        df = add_linear_control(df, cap)

        # ---- outputs and the two estimators ------------------------------ #
        od = os.path.join(args.out_dir, f"step{step}")
        os.makedirs(od, exist_ok=True)
        meta = dict(ckpt=os.path.abspath(path), step=step, run=os.path.basename(os.path.dirname(os.path.abspath(path))),
                    provenance=prov, probe=dict(split=args.split, offset=args.offset, sequences=int(idx.shape[0]),
                                                seq_len=seq_len, positions=positions, batch_shape=list(idx.shape)),
                    capture=cap["_meta"], decomp_err={int(l): cap[l]["decomp_err"] for l in layers},
                    capture_phase={int(l): cap[l]["phase"] for l in layers},
                    windows=dict(active=args.active_window, cand=args.cand_window, pairs_per_token=args.pairs_per_token,
                                 exhaustive=args.exhaustive, seed=args.seed),
                    dtype=args.dtype, suffix=use_suffix, suffix_check=suffix_check,
                    tau=tau, tau_replay_paired=tau_replay, n_paired_noise=int(noise_all.size),
                    cross_batch_identity_per_layer=tau_src, tau_factor=args.tau_factor, tau_floor=args.tau_floor,
                    layer_weights={int(l): w for l, w in weights.items()}, loss="mean fp32 CE per sequence, fp64",
                    linear_control="r_linear = z_i g_i - z_j g_j, g = dL/dy at the gate output, same backward as h",
                    seconds=time.time() - t0)
        sup, lin = write_outputs(od, df, meta, layers, weights, tau, args.bootstrap, args.seed)
        per_layer = sup["per_layer"]
        vs_step.append(dict(step=step, E=sup["E"], E_linear=lin["E"], ci=sup["bootstrap"]["aggregate"] if sup["bootstrap"] else None,
                            ci_linear=lin["bootstrap"]["aggregate"] if lin["bootstrap"] else None,
                            per_layer={l: v["E"] for l, v in per_layer.items()},
                            per_layer_linear={l: v["E"] for l, v in lin["per_layer"].items()},
                            n_improve=int(sum(v["n_improve"] for v in per_layer.values())),
                            n_harm=int(sum(v["n_harm"] for v in per_layer.values())), tau=tau))
        with open(os.path.join(args.out_dir, "credit_vs_step.json"), "w") as fh:
            json.dump(vs_step, fh, indent=2)
        print(f"step {step:>6}: tau {tau:.2e}  rows {len(df)}  classes {df['cls'].value_counts().to_dict()}\n"
              f"                      support {fmt(sup)}\n                      linear  {fmt(lin)}")
        if args.backup:
            subprocess.run(["rclone", "copy", args.out_dir, args.backup], check=False)
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None


if __name__ == "__main__":
    main()
