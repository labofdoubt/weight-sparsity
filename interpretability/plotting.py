"""Figures for the interpretability benchmark.

    python interpretability/plotting.py --results results/bn_pmlp_k32j64_rel
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

K_CURVE = (1, 2, 4, 8, 16)


def topk_curve(results: str, out: str) -> None:
    df = pd.read_csv(os.path.join(results, "sparse_probe_per_concept.csv"))
    cols = [f"acc_k{k}" for k in K_CURVE]
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, r in df.iterrows():
        ax.plot(K_CURVE, [r[c] for c in cols], color="0.75", lw=1, zorder=1)
    macro = [df[c].mean() for c in cols]
    ax.plot(K_CURVE, macro, "o-", color="#1f77b4", lw=2.5, ms=7,
            label="macro average", zorder=3)
    ax.axhline(0.5, ls=":", color="0.4", lw=1, label="chance")
    ax.set_xscale("log", base=2)
    ax.set_xticks(K_CURVE)
    ax.set_xticklabels(K_CURVE)
    ax.set_xlabel("k (features selected by train-set mean difference)")
    ax.set_ylabel("test accuracy")
    ax.set_title(
        "Sparse probe accuracy vs k\n"
        f"A1={macro[0]:.3f}  A16={macro[-1]:.3f}  "
        f"({'localized' if macro[-1] - macro[0] < 0.05 else 'distributed'})"
    )
    ax.set_ylim(0.45, 1.0)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def localization(results: str, out: str) -> None:
    path = os.path.join(results, "localization_per_concept.csv")
    df = pd.read_csv(path).sort_values("ScatterRate")
    fig, axes = plt.subplots(1, 3, figsize=(15, 8), sharey=True)
    y = range(len(df))
    panels = [
        ("CanonicalMissRate", "#d62728", "canonical feature inactive\n(lower is better)"),
        ("ScatterRate", "#ff7f0e", "concept survives elsewhere\n(lower is better)"),
        ("MedianEffectiveContributorCount", "#1f77b4",
         "effective contributing features\n(lower is more localized)"),
    ]
    for ax, (col, colour, title) in zip(axes, panels):
        ax.barh(list(y), df[col], color=colour, height=0.7)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.axvline(df[col].mean(), ls="--", lw=1, color="0.3")
    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels(df.concept, fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle("TinyStories Concept Localization (dashed = macro mean)", y=0.99)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    topk_curve(args.results, os.path.join(args.results, "sparse_probe_topk_curve.png"))
    localization(args.results, os.path.join(args.results, "concept_localization.png"))
    print(f"[plot] wrote figures to {args.results}/")


if __name__ == "__main__":
    main()
