"""Collect per-run benchmark results into one comparison table.

    python interpretability/summarize_runs.py --results results --out results/summary

Writes ``benchmark_summary.csv`` and ``benchmark_summary.md``.  Localization
columns use the ``_refit`` variants, which are the meaningful ones whenever the
bottleneck is not an autoencoder (see the README); the as-specified columns are
carried alongside so both are on the record.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

COLUMNS = [
    ("run", "run", None),
    ("place", "bottleneck_location", None),
    ("K", "K", None),
    ("N", "bottleneck_width", None),
    ("layer", "evaluation_layer", None),
    ("Top1", "SparseProbeTop1", 3),
    ("CI_lo", "ci_lo", 3),
    ("CI_hi", "ci_hi", 3),
    ("A2", "k2", 3),
    ("A8", "k8", 3),
    ("A16", "k16", 3),
    ("Resid", "ResidualProbeAccuracy", 3),
    ("Miss", "CanonicalMissRate", 3),
    ("Scat*", "ScatterRate_refit", 3),
    ("Retain*", "RetainedElsewhere_refit", 3),
    ("Neff", "MedianEffectiveContributorCount", 1),
    ("Ccan", "MeanCanonicalContributionShare", 3),
    ("Out*", "BottleneckOutputProbeAccuracy_refit", 3),
    ("Scat_spec", "ScatterRate", 3),
    ("Out_spec", "BottleneckOutputProbeAccuracy", 3),
    ("cos", "reconstruction_cosine", 3),
    ("rand", "random_feature_accuracy", 3),
    ("shuf", "shuffled_label_accuracy", 3),
]


def collect(results_dir: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "aggregate_metrics.json"))):
        a = json.load(open(path))
        meta, loc = a["meta"], a.get("localization", {}).get("macro", {})
        ci = a.get("SparseProbeTop1_ci95", [float("nan")] * 2)
        row = dict(
            run=os.path.basename(os.path.dirname(path)),
            ci_lo=ci[0],
            ci_hi=ci[1],
            **{k: meta[k] for k in
               ("bottleneck_location", "K", "bottleneck_width", "evaluation_layer")},
            **{k: a[k] for k in ("SparseProbeTop1", "ResidualProbeAccuracy")},
            **{k: v for k, v in a["k_curve"].items()},
            **{k: a["sanity"][k] for k in
               ("random_feature_accuracy", "shuffled_label_accuracy")},
        )
        # the localization block repeats a few probe columns; keep the originals
        row.update({k: v for k, v in loc.items() if k not in row})
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", required=True)
    ap.add_argument("--order", nargs="*", default=None, help="run names, in report order")
    args = ap.parse_args()

    df = collect(args.results)
    if args.order:
        rank = {n: i for i, n in enumerate(args.order)}
        df = df[df.run.isin(rank)].sort_values("run", key=lambda s: s.map(rank))
    os.makedirs(args.out, exist_ok=True)

    keep = [src for _, src, _ in COLUMNS if src in df.columns]
    table = df[keep].copy()
    table.columns = [label for label, src, _ in COLUMNS if src in df.columns]
    table.to_csv(os.path.join(args.out, "benchmark_summary.csv"), index=False)

    fmt = table.copy()
    for label, src, nd in COLUMNS:
        if nd and label in fmt.columns:
            fmt[label] = fmt[label].map(lambda v: "" if pd.isna(v) else f"{v:.{nd}f}")
    md = ["| " + " | ".join(fmt.columns) + " |",
          "|" + "|".join("---" for _ in fmt.columns) + "|"]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in fmt.itertuples(index=False)]
    with open(os.path.join(args.out, "benchmark_summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    print("\n".join(md))
    print(f"\n[summary] {len(table)} runs -> {args.out}/benchmark_summary.{{csv,md}}")


if __name__ == "__main__":
    main()
