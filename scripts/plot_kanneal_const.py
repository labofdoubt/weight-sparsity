"""Constant-LR K-annealing (docs/kanneal): the gamma family against hard Top-K.

    python scripts/plot_kanneal_const.py --metrics-root DIR   # extract, then plot
    python scripts/plot_kanneal_const.py                      # plot from the JSON

All runs at constant learning rate 6e-4 without warm-up and without post-norm
(configs/hard_k_anneal_const/).  Upper panel: validation CE with the Top-32
forward (val_topk/ce) of the K-annealing runs at J = 224, s_0 = 1 -- the log
schedule (gamma -> 0) and the power schedule at gamma = 0.1 .. 0.6 -- against
the hard Top-K runs (val/ce at their own K).  Lower panel: the logged K'.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "docs", "kanneal")
DATA = os.path.join(ROOT, "data", "curves_const.json")
FIG = os.path.join(ROOT, "figures", "const_gamma_vs_hard.png")

S = 18000  # hard_k_annealing_steps of the annealed runs
KS = (32, 64, 128, 256, 512)
# (key, legend label, Drive folder, run name)
ANNEAL = [("log", r"log ($\gamma \to 0$)", "runs_vilnius", "vi_cr_klog_s1_k32_j224_const")] + [
    (f"g{g}", rf"$\gamma = {g / 10:g}$", "runs_vilnius", f"vi_cr_kpow_g0{g}_s1_k32_j224_const")
    for g in (1, 2, 4, 6)]
HARD = [(f"hard{k}", f"Top-{k}", "runs_tallinn", f"ta_cr_hard_k{k}_const") for k in KS]

# K-annealing: blue ramp, light -> dark with gamma; hard Top-K: orange ramp,
# light -> dark with K (as in the other docs/kanneal figures)
BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
ORANGE = ["#f39a6b", "#ee7f48", "#eb6834", "#c4501f", "#8f3a14"]
COLOR = {k: c for (k, *_), c in zip(ANNEAL, BLUE)}
COLOR.update({k: c for (k, *_), c in zip(HARD, ORANGE)})
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def extract(root: str) -> None:
    out = {}
    for key, _, folder, run in ANNEAL + HARD:
        val, active = [], []
        for line in open(os.path.join(root, run, "metrics.jsonl")):
            r = json.loads(line)
            if "val/ce" in r:
                # the annealed runs: the Top-32 forward; the hard runs: their own K
                val.append([r["step"], r.get("val_topk/ce", r["val/ce"])])
            if "bottleneck/active_count" in r:
                active.append([r["step"], r["bottleneck/active_count"]])
        out[key] = {"drive": f"gdrive:weight-sparsity/{folder}/{run}", "val": val,
                    "active_count": active}
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    with open(DATA, "w") as f:
        json.dump(out, f)
    print(f"wrote {DATA}")


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


def plot(data: dict) -> None:
    fig, (top, bot) = plt.subplots(2, 1, figsize=(7.6, 6.6), sharex=True,
                                   gridspec_kw={"height_ratios": [2.1, 1]})
    hard_lines, anneal_lines = [], []
    for key, label, *_ in HARD:
        xs, ys = zip(*data[key]["val"])
        (ln,) = top.plot(xs, ys, color=COLOR[key], lw=1.5, ls="--",
                         label=f"{label}  ({ys[-1]:.4f})")
        hard_lines.append(ln)
    for key, label, *_ in ANNEAL:
        xs, ys = zip(*data[key]["val"])
        (ln,) = top.plot(xs, ys, color=COLOR[key], lw=1.5, label=f"{label}  ({ys[-1]:.4f})")
        top.plot(xs[-1], ys[-1], "o", ms=3.5, color=COLOR[key], markeredgecolor="white",
                 markeredgewidth=0.7, zorder=4)
        anneal_lines.append(ln)
        xa, ya = zip(*data[key]["active_count"])
        bot.plot(xa, ya, color=COLOR[key], lw=1.5)
    for ax in (top, bot):
        ax.axvline(S, color=MUTED, lw=0.8, ls=":")
        style(ax)
    top.set_ylim(1.42, 2.6)
    top.set_ylabel("validation CE (nats)")
    top.set_title("validation: Top-32 forward for K-annealing, Top-K for hard Top-K",
                  fontsize=8, color=INK)
    leg1 = top.legend(handles=anneal_lines, loc="upper right", bbox_to_anchor=(0.88, 1.0),
                      fontsize=7, frameon=False,
                      title="K-annealing, J = 224, $s_0 = 1$\n(final val CE)", title_fontsize=7)
    top.add_artist(leg1)
    top.legend(handles=hard_lines, loc="upper right", bbox_to_anchor=(0.6, 1.0), fontsize=7,
               frameon=False, title="hard Top-K\n(final val CE)", title_fontsize=7)
    bot.text(S, 262, " S = 18000", fontsize=7, color=MUTED, va="top")
    bot.set_ylim(0, 270)
    bot.set_yticks([32, 64, 128, 192, 256])
    bot.set_xlim(0, 20000)
    bot.set_xlabel("training step")
    bot.set_ylabel("active count $K'$")
    bot.set_title(r"K-annealing schedule, $K' = 256 \to 32$ over the first S steps",
                  fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(FIG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-root", default="")
    args = ap.parse_args()
    if args.metrics_root:
        extract(args.metrics_root)
    plot(json.load(open(DATA)))


if __name__ == "__main__":
    main()
