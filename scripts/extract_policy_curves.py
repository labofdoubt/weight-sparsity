"""Curves of laplace_policy runs for the policy_cr note, as one JSON.

For every run directory given (or matched by ``--glob``) reads ``config.json``,
``metrics.jsonl`` and, when present, ``summary.json`` / ``diverged.json`` /
``stopped.json``, and records

  cfg     K, J, width mode, tau_0, schedule, tau_f, hold, anneal, seed
  val     [step, clean val/ce, val_stochastic/ce, mc_std]           (every validation)
  train   [step, sampled train/ce, noise-off probe CE, grad norm, tau,
           baseline, advantage mean, advantage rms]                 (every --every steps)
  blocks  {key: [[step, [value per block]], ...]} for the per-gate keys
          exchange, overlap, t_mean, span, score_grad_rms, score_grad_rms_raw
                                                                    (every --every steps)
  clip    [window end, share of logged steps with grad norm > clip, median norm]
          over consecutive --clip-window step windows (every logged step)
  summary / diverged / stopped   the files' contents

    python scripts/extract_policy_curves.py --glob '/workspace/runs/vi_pol_rs_*' \
        --out curves_policy.json [--every 100]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics

BLOCK_KEYS = {"exchange": "policy_exchange_frac", "overlap": "policy_overlap",
              "t_mean": "policy_t_mean", "span": "policy_pool_span",
              "score_grad_rms": "policy_score_grad_rms",
              "score_grad_rms_raw": "policy_score_grad_rms_raw"}


def blocks(row: dict, key: str) -> list:
    out = []
    for name, v in row.items():
        m = re.fullmatch(rf"bottleneck_{key}/blocks\.(\d+)", name)
        if m:
            out.append((int(m.group(1)), v))
    return [v for _, v in sorted(out)]


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def extract(run_dir: str, every: int, clip: float, window: int) -> dict:
    cfg = load_json(os.path.join(run_dir, "config.json")) or {}
    bn, tr = cfg.get("activation_bottleneck", {}), cfg.get("train", {})
    rec = {"cfg": {"k": bn.get("k"), "j": bn.get("j"),
                   "mode": bn.get("policy_temperature_mode"), "tau0": bn.get("temperature"),
                   "schedule": bn.get("policy_temperature_schedule"),
                   "tau_final": bn.get("policy_temperature_final"),
                   "hold": bn.get("policy_temperature_hold_steps"),
                   "anneal": bn.get("policy_temperature_anneal_steps"),
                   "seed": tr.get("seed"), "max_steps": tr.get("max_steps"),
                   "grad_clip": tr.get("grad_clip")},
           "val": [], "train": [], "blocks": {k: [] for k in BLOCK_KEYS}, "clip": []}
    rows = []
    with open(os.path.join(run_dir, "metrics.jsonl")) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    norms = []
    for r in rows:
        s = r.get("step")
        if "val/ce" in r:
            rec["val"].append([s, r["val/ce"], r.get("val_stochastic/ce"),
                               r.get("val_stochastic/mc_std")])
        if "train/ce" in r:
            norms.append((s, r["train/grad_norm"]))
            if s % every == 0 or s == 1:
                rec["train"].append([s, r["train/ce"], r.get("train_deterministic_probe/ce"),
                                     r["train/grad_norm"], r.get("policy/tau"),
                                     r.get("policy/baseline"), r.get("policy/advantage_mean"),
                                     r.get("policy/advantage_rms")])
                for name, key in BLOCK_KEYS.items():
                    vals = blocks(r, key)
                    if vals:
                        rec["blocks"][name].append([s, vals])
    for end in range(window, (norms[-1][0] if norms else 0) + window, window):
        w = [g for s, g in norms if end - window < s <= end]
        if w:
            rec["clip"].append([end, sum(g > clip for g in w) / len(w), statistics.median(w)])
    for fn in ("summary", "diverged", "stopped"):
        v = load_json(os.path.join(run_dir, f"{fn}.json"))
        if v is not None:
            rec[fn] = v
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--glob", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--clip-window", type=int, default=500)
    args = ap.parse_args()
    dirs = list(args.runs) + (sorted(glob.glob(args.glob)) if args.glob else [])
    out = {}
    for d in dirs:
        if os.path.isfile(os.path.join(d, "metrics.jsonl")):
            out[os.path.basename(d.rstrip("/"))] = extract(d, args.every, args.clip,
                                                           args.clip_window)
    with open(args.out, "w") as f:
        json.dump(out, f)
    for name, rec in out.items():
        v = rec["val"][-1] if rec["val"] else None
        print(f"{name:36s} val points {len(rec['val']):3d}  last {v}")


if __name__ == "__main__":
    main()
