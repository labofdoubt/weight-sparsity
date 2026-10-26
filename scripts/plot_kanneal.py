"""Figures of docs/kanneal: hard_k_annealing (log schedule) against hard Top-K.

    python scripts/plot_kanneal.py --metrics-root DIR   # extract, then plot
    python scripts/plot_kanneal.py                      # plot from the JSON

``--metrics-root`` holds one directory per run (named as on Drive) with its
``metrics.jsonl``; the validation curves and the logged active counts are
written to ``docs/kanneal/data/curves.json``, which the figures are drawn from.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "docs", "kanneal")
DATA = os.path.join(ROOT, "data", "curves.json")
FIG = os.path.join(ROOT, "figures")

HS = (125, 250, 500, 1000, 2000, 4000, 6000, 8000)
KS = (32, 64, 128, 256, 512)
# (key, Drive folder, run name); the hard ladder is the code-carried one of
# docs/kj-vs-hard-topk-code-residual.tex
RUNS = ([(f"hard{k}", "runs_madrid", f"ma_cr_hard_k{k}_pnorm") for k in KS]
        + [("linear", "runs_riga", "ri_cr_hard_kanneal_k32_j480_pnorm")]
        + [(f"h{h}", "runs_tallinn", f"ta_cr_hard_klog_h{h}_k32_j480_pnorm") for h in HS])

# the log runs on a blue ramp, light -> dark with h; the hard ladder dashed on
# an orange ramp, light -> dark with K; the linear schedule in magenta.  The
# same run has the same color in every figure.
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
        "#184f95", "#0d366b"]
HARD_RAMP = ["#f39a6b", "#ee7f48", "#eb6834", "#c4501f", "#8f3a14"]
COLOR = {f"h{h}": c for h, c in zip(HS, RAMP)}
COLOR.update({f"hard{k}": c for k, c in zip(KS, HARD_RAMP)})
COLOR["linear"] = "#d55181"   # magenta
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def extract(metrics_root: str) -> None:
    out = {}
    for key, folder, run in RUNS:
        path = os.path.join(metrics_root, run, "metrics.jsonl")
        val, active = [], []
        for line in open(path):
            r = json.loads(line)
            if "val/ce" in r:
                val.append([r["step"], r["val/ce"]])
            if "bottleneck/active_count" in r:
                active.append([r["step"], r["bottleneck/active_count"]])
        out[key] = {"drive": f"gdrive:weight-sparsity/{folder}/{run}", "run": run,
                    "val_ce": val, "active_count": active}
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    with open(DATA, "w") as f:
        json.dump(out, f)
    print(f"wrote {DATA}")


def label(key: str, d: dict) -> str:
    final = d["val_ce"][-1]
    tail = f"{final[1]:.4f}" if final[0] == 20000 else f"step {final[0]}"
    return f"{name_of(key)}  ({tail})"


def name_of(key: str) -> str:
    if key.startswith("hard"):
        return f"hard Top-{key[4:]}"
    if key == "linear":
        return "linear schedule ($h = S/2$)"
    return f"log, $h = {key[1:]}$"


def style(ax) -> None:
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


MAIN = ["hard32", "hard512"] + [f"h{h}" for h in HS] + ["linear"]
LADDER = [f"hard{k}" for k in KS] + ["h125", "h250", "h500"]


def plot_val(data: dict, keys, fname: str, ncol: int = 3) -> None:
    keys = [k for k in keys if k in data]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.2))
    for ax, (lo, hi), (ylo, yhi) in zip(axes, ((0, 20000), (12000, 20000)),
                                         ((1.3, 2.7), (1.33, 1.72))):
        for key in keys:
            d = data[key]
            xs = [s for s, _ in d["val_ce"] if lo <= s <= hi]
            ys = [v for s, v in d["val_ce"] if lo <= s <= hi]
            base = key.startswith("hard")
            ax.plot(xs, ys, color=COLOR[key], lw=1.6 if base else 1.4,
                    ls="--" if base else "-", label=label(key, d), zorder=2 if base else 3)
            if xs and xs[-1] == 20000 and not base:
                ax.plot(xs[-1], ys[-1], "o", ms=4, color=COLOR[key], zorder=4,
                        markeredgecolor="white", markeredgewidth=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(ylo, yhi)
        ax.set_xlabel("training step")
        style(ax)
    axes[0].set_ylabel("validation CE (nats)")
    axes[0].set_title("whole run", fontsize=8, color=INK)
    axes[1].set_title("steps 12000 to 20000", fontsize=8, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=ncol,
               fontsize=7, frameon=False, columnspacing=1.2, handlelength=2.2,
               title="in parentheses: final validation CE", title_fontsize=7)
    fig.savefig(os.path.join(FIG, fname), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_active(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for key in [k for k in MAIN if k in data]:
        d = data[key]
        if key.startswith("hard"):
            k = int(key[4:])
            ax.axhline(k, color=COLOR[key], lw=1.6, ls="--", label=f"hard Top-{k}")
            continue
        xs = [s for s, _ in d["active_count"]]
        ys = [v for _, v in d["active_count"]]
        name = ("linear schedule ($h = S/2$)" if key == "linear"
                else f"log, $h = {key[1:]}$")
        ax.plot(xs, ys, color=COLOR[key], lw=1.4, label=name)
    ax.set_xlim(0, 20000)
    ax.set_ylim(0, 530)
    ax.set_yticks([0, 32, 128, 256, 384, 512])
    ax.set_xlabel("training step")
    ax.set_ylabel("active count $K'$ per token")
    style(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "active_count.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-root", default="")
    args = ap.parse_args()
    if args.metrics_root:
        extract(args.metrics_root)
    data = json.load(open(DATA))
    os.makedirs(FIG, exist_ok=True)
    plot_val(data, MAIN, "val_ce.png")
    plot_active(data)
    plot_val(data, LADDER, "val_ce_ladder.png", ncol=3)
    print(f"wrote {FIG}/val_ce.png, active_count.png, val_ce_ladder.png")


if __name__ == "__main__":
    main()
