"""Calibration table of the laplace_policy temperature probes (vi_cal_*).

For every probe run directory under ``--runs`` reads ``metrics.jsonl`` and
prints, at the requested steps, the per-block support exchange fraction
q_l = 1 - |S_sampled & S_clean| / K (``bottleneck_policy_exchange_frac/
blocks.l``), its median over blocks, the largest block value, the pool span
s_(K+1) - s_(K+J) (median over blocks), the sampled training CE and the
gradient norm.  ``--json`` also writes everything as one file for the report.

    python scripts/policy_calibration_table.py --runs /workspace/runs \
        --steps 10 20 30 100 200 [--json out.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics


def per_block(row: dict, key: str) -> list:
    out = []
    for name, v in row.items():
        m = re.fullmatch(rf"bottleneck_{key}/blocks\.(\d+)", name)
        if m:
            out.append((int(m.group(1)), v))
    return [v for _, v in sorted(out)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/workspace/runs")
    ap.add_argument("--pattern", default="vi_cal_*")
    ap.add_argument("--steps", nargs="*", type=int, default=[10, 20, 30, 100, 200])
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    table = {}
    for d in sorted(glob.glob(os.path.join(args.runs, args.pattern))):
        path = os.path.join(d, "metrics.jsonl")
        if not os.path.isfile(path):
            continue
        name = os.path.basename(d)
        m = re.search(r"_k(\d+)_j(\d+)_t([0-9p]+)", name)
        k, j, t = int(m.group(1)), int(m.group(2)), float(m.group(3).replace("p", "."))
        rows = {}
        for line in open(path):
            r = json.loads(line)
            if "bottleneck/policy_exchange_frac" in r:
                rows[r["step"]] = r
        entry = {"k": k, "j": j, "T": t, "steps": {}}
        for s in args.steps:
            if s not in rows:
                continue
            r = rows[s]
            q = per_block(r, "policy_exchange_frac")
            span = per_block(r, "policy_pool_span")
            gap = per_block(r, "score_gap")
            entry["steps"][s] = {
                "q": q, "q_median": statistics.median(q), "q_max": max(q), "q_min": min(q),
                "span_median": statistics.median(span) if span else None,
                "gap_median": statistics.median(gap) if gap else None,
                "ce": r.get("train/ce"), "gnorm": r.get("train/grad_norm"),
                "det_ce": r.get("train_deterministic_probe/ce"),
                "adv_mean": r.get("policy/advantage_mean"),
                "adv_rms": r.get("policy/advantage_rms"),
                "score_grad_rms": per_block(r, "policy_score_grad_rms"),
            }
        table[name] = entry
    cells = sorted({(e["k"], e["j"]) for e in table.values()})
    for k, j in cells:
        print(f"\n(K, J) = ({k}, {j})")
        print("   T     step  med q  min q  max q   span   CE      gnorm   per-block q")
        for name, e in sorted(table.items(), key=lambda kv: kv[1]["T"]):
            if (e["k"], e["j"]) != (k, j):
                continue
            for s, v in sorted(e["steps"].items()):
                print(f"  {e['T']:<6g} {s:4d}  {v['q_median']:.3f}  {v['q_min']:.3f}  "
                      f"{v['q_max']:.3f}  {v['span_median']:.3f}  {v['ce']:.3f}  "
                      f"{v['gnorm']:8.0f}  " + " ".join(f"{x:.3f}" for x in v["q"]))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(table, f, indent=1)


if __name__ == "__main__":
    main()
