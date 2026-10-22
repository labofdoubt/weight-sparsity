"""Figures and tables of the population sweep (toy_mnist/configs/pop_sweep).

    python toy_mnist/plot_pop_sweep.py --runs <dir with hard_k*/ and dust_k*_p*/> \
        --out toy_mnist/reports/pop_sweep

Per K one figure (training CE and test accuracy against the step: hard Top-K
and DUST at every population) and one LaTeX table of the final values; also
results.json.  Runs without summary.json are skipped and listed.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# colors: the hard baseline wears the orange categorical slot, the populations
# the blue ordinal ramp (light -> dark with the population); ink and chrome
# from the reference palette
HARD = "#eb6834"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def read_metrics(run_dir: str) -> Optional[Dict[str, List[dict]]]:
    if not os.path.exists(os.path.join(run_dir, "summary.json")):
        return None
    train, val = [], []
    with open(os.path.join(run_dir, "metrics.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            (val if "val/ce" in r else train).append(r)
    return {"train": train, "val": val}


def fmt_pop(p: int) -> str:
    return f"{p // 1000}k" if p % 1000 == 0 else str(p)


def style_axes(ax, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_color(INK2)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="4,8,16,32")
    ap.add_argument("--pops", default="128,512,2000,8000,32000")
    args = ap.parse_args()
    ks = [int(v) for v in args.ks.split(",")]
    pops = [int(v) for v in args.pops.split(",")]
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)
    missing: List[str] = []
    results: Dict[str, dict] = {}
    tables: List[str] = []

    for k in ks:
        series = []  # (label, color, linestyle, metrics)
        m = read_metrics(os.path.join(args.runs, f"hard_k{k}"))
        if m is None:
            missing.append(f"hard_k{k}")
        else:
            series.append((f"hard Top-{k}", HARD, "-", m, "hard", None))
        for color, p in zip(RAMP, pops):
            name = f"dust_k{k}_p{p}"
            m = read_metrics(os.path.join(args.runs, name))
            if m is None:
                missing.append(name)
                continue
            series.append((f"DUST, population {fmt_pop(p)}", color, "-", m, "dust", p))
        if not series:
            continue

        fig, (ax_ce, ax_acc) = plt.subplots(1, 2, figsize=(7.2, 2.9), dpi=200)
        fig.patch.set_facecolor(SURFACE)
        for label, color, ls, m, kind, p in series:
            steps = [r["step"] for r in m["train"]]
            ax_ce.plot(steps, [r["train/ce"] for r in m["train"]], color=color, ls=ls,
                       lw=1.5, label=label, solid_joinstyle="round", solid_capstyle="round")
            vsteps = [r["step"] for r in m["val"]]
            ax_acc.plot(vsteps, [r["val/acc"] for r in m["val"]], color=color, ls=ls, lw=1.5,
                        marker="o", ms=3.5, markeredgecolor=SURFACE, markeredgewidth=0.8,
                        label=label)
            final_train = m["train"][-1]
            final_val = m["val"][-1]
            results[f"k{k}_" + ("hard" if kind == "hard" else f"p{p}")] = {
                "k": k, "method": kind, "population": p, "step": final_val["step"],
                "train_ce": final_train["train/ce"], "val_ce": final_val["val/ce"],
                "val_acc": final_val["val/acc"],
            }
        ax_ce.set_yscale("log")
        ces = [r["train/ce"] for _, _, _, m, _, _ in series for r in m["train"]]
        ticks = [t for t in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0) if min(ces) * 0.9 <= t <= max(ces) * 1.1]
        ax_ce.set_yticks(ticks)
        ax_ce.set_yticklabels([f"{t:g}" for t in ticks])
        ax_ce.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        style_axes(ax_ce, "step", "training CE (20-step windows)")
        style_axes(ax_acc, "step", "test accuracy")
        accs = [r["val/acc"] for _, _, _, m, _, _ in series for r in m["val"][1:]]
        if accs:
            lo = max(0.0, min(accs) - 0.01)
            ax_acc.set_ylim(lo, 1.0 if max(accs) > 0.97 else max(accs) + 0.01)
        ax_ce.set_title(f"K = {k}: training loss", color=INK, fontsize=10, loc="left")
        ax_acc.set_title(f"K = {k}: test accuracy", color=INK, fontsize=10, loc="left")
        leg = ax_acc.legend(fontsize=7, frameon=False, loc="lower right", labelcolor=INK2)
        fig.tight_layout()
        path = os.path.join(args.out, "figures", f"curves_k{k}.png")
        fig.savefig(path, facecolor=SURFACE)
        plt.close(fig)

        rows = []
        for label, color, ls, m, kind, p in series:
            r = results[f"k{k}_" + ("hard" if kind == "hard" else f"p{p}")]
            method = "hard Top-$K$" if kind == "hard" else "DUST"
            pop = "--" if kind == "hard" else f"{p:,}"
            rows.append(f"    {method} & {pop} & {r['train_ce']:.4f} & {r['val_ce']:.4f} & "
                        f"{100 * r['val_acc']:.2f} \\\\")
        tables.append(
            "\\begin{table}[H]\n\\centering\n"
            f"\\caption{{$K = {k}$: values at the final step ({series[0][3]['val'][-1]['step']}). "
            "Training CE is the mean of the last 20-step window of the clean hard forward; "
            "test CE and accuracy are the deterministic hard forward on the 10\\,000 test images.}\n"
            f"\\label{{tab:k{k}}}\n"
            "\\begin{tabular}{llrrr}\n\\toprule\n"
            "    method & population & train CE & test CE & test acc (\\%) \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    with open(os.path.join(args.out, "tables.tex"), "w") as f:
        f.write("\n".join(tables))
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"{len(results)} runs; figures and tables in {args.out}")
    if missing:
        print("missing (no summary.json):", ", ".join(missing))


if __name__ == "__main__":
    main()
