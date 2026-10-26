"""Figures of the gate-site study (toy_mnist/configs/site_sweep): per K, the
training CE, test CE and test accuracy of the gate after W0 against the gate on
the input.

    python toy_mnist/plot_site_sweep.py --runs <dir with pre_k*/ and input_k*/> \
        --out toy_mnist/reports/site_sweep
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

from plot_pop_sweep import INK, SURFACE, read_metrics, style_axes  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
VARIANTS = [
    ("pre", "gate after W0: W2 phi(W1 TopK(W0 x))", BLUE),
    ("input", "gate on the input: W2 phi(W1 W0 TopK(x))", ORANGE),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="2,4,8,16")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)
    missing, results = [], {}
    for k in [int(v) for v in args.ks.split(",")]:
        series = []
        for prefix, label, color in VARIANTS:
            m = read_metrics(os.path.join(args.runs, f"{prefix}_k{k}"))
            if m is None:
                missing.append(f"{prefix}_k{k}")
                continue
            series.append((prefix, label, color, m))
        if not series:
            continue
        fig, (ax_ce, ax_vce, ax_acc) = plt.subplots(1, 3, figsize=(10.8, 2.9), dpi=200)
        fig.patch.set_facecolor(SURFACE)
        for prefix, label, color, m in series:
            steps = [r["step"] for r in m["train"]]
            ax_ce.plot(steps, [r["train/ce"] for r in m["train"]], color=color, lw=1.5,
                       label=label, solid_joinstyle="round", solid_capstyle="round")
            vsteps = [r["step"] for r in m["val"]]
            for ax, key in ((ax_vce, "val/ce"), (ax_acc, "val/acc")):
                ax.plot(vsteps, [r[key] for r in m["val"]], color=color, lw=1.5, marker="o",
                        ms=3.5, markeredgecolor=SURFACE, markeredgewidth=0.8, label=label)
            results[f"{prefix}_k{k}"] = {
                "k": k, "site": prefix, "step": m["val"][-1]["step"],
                "train_ce": m["train"][-1]["train/ce"], "val_ce": m["val"][-1]["val/ce"],
                "val_acc": m["val"][-1]["val/acc"]}
        for ax, key, part in ((ax_ce, "train/ce", "train"), (ax_vce, "val/ce", "val")):
            ax.set_yscale("log")
            ces = [r[key] for _, _, _, m in series for r in m[part]]
            ticks = [t for t in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0)
                     if min(ces) * 0.8 <= t <= max(ces) * 1.2]
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{t:g}" for t in ticks])
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        style_axes(ax_ce, "step", "training CE (20-step windows)")
        style_axes(ax_vce, "step", "test CE")
        style_axes(ax_acc, "step", "test accuracy")
        accs = [r["val/acc"] for _, _, _, m in series for r in m["val"][1:]]
        if accs:
            ax_acc.set_ylim(max(0.0, min(accs) - 0.01), 1.0 if max(accs) > 0.97 else max(accs) + 0.01)
        ax_ce.set_title(f"K = {k}: training loss", color=INK, fontsize=10, loc="left")
        ax_vce.set_title(f"K = {k}: test loss", color=INK, fontsize=10, loc="left")
        ax_acc.set_title(f"K = {k}: test accuracy", color=INK, fontsize=10, loc="left")
        ax_ce.legend(fontsize=7, frameon=False, loc="upper right", labelcolor="#52514e")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "figures", f"curves_k{k}.png"), facecolor=SURFACE)
        plt.close(fig)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"{len(results)} runs; figures in {args.out}/figures")
    if missing:
        print("missing (no summary.json):", ", ".join(missing))


if __name__ == "__main__":
    main()
