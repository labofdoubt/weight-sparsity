"""Figures of the 512-draw schedule sweep of the small-model DUST study
(docs/dust/small-dust-schedule-sweep.tex; configs/small_dust).

    rclone copy gdrive:weight-sparsity/runs_warsaw RUNS --include "wa_k4_*/metrics.jsonl"
    python scripts/plot_small_dust_sweep.py --runs-dir RUNS [--out-dir docs/dust/figures]

Every curve is the validation CE (val/ce: 40 batches x 24 windows, the same
windows in every run, every 20 steps).  The difference panels subtract the
hard Top-4 run's value at the same step and smooth the difference with a
centred moving average over SMOOTH validations.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

HARD = "wa_k4_hard"
SMOOTH = 5  # validations (100 steps)

#: schedule -> colour, the same in every figure
COLORS = {
    (1, 1): "#d62728",
    (30, 5): "#ff7f0e",
    (10, 1): "#1f77b4",
    (30, 1): "#2ca02c",
    (90, 1): "#9467bd",
}
LR3_COLOR = "#17becf"

ENCODER = [  # (run, schedule)
    ("wa_k4_dust_p512", (1, 1)),
    ("wa_k4_dust_p512_h30_d5", (30, 5)),
    ("wa_k4_dust_p512_h10", (10, 1)),
    ("wa_k4_dust_p512_h30", (30, 1)),
    ("wa_k4_dust_p512_h90", (90, 1)),
]
BACKPROP = [
    ("wa_k4_dust_p512_bp_h10", (10, 1)),
    ("wa_k4_dust_p512_bp_h30", (30, 1)),
    ("wa_k4_dust_p512_bp_h90", (90, 1)),
]


def val_curve(runs_dir: str, run: str) -> Tuple[np.ndarray, np.ndarray]:
    rows = [json.loads(line) for line in open(os.path.join(runs_dir, run, "metrics.jsonl"))]
    pts = sorted((r["step"], r["val/ce"]) for r in rows if "val/ce" in r)
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def n_dust_steps(schedule: Tuple[int, int], last_step: int) -> int:
    h, d = schedule
    return sum(1 for s in range(last_step) if s % (h + d) >= h)


def gap(runs_dir: str, run: str, hard: Tuple[np.ndarray, np.ndarray]
        ) -> Tuple[np.ndarray, np.ndarray]:
    """``val/ce(run) - val/ce(hard)`` at common steps, centred moving average
    over SMOOTH validations (shorter windows at the ends)."""
    s, v = val_curve(runs_dir, run)
    ref = dict(zip(hard[0].tolist(), hard[1].tolist()))
    keep = [i for i, x in enumerate(s) if int(x) in ref]
    s = s[keep]
    d = v[keep] - np.array([ref[int(x)] for x in s])
    half = SMOOTH // 2
    sm = np.array([d[max(0, i - half): i + half + 1].mean() for i in range(len(d))])
    return s, sm


def label(schedule: Tuple[int, int], s: np.ndarray, stopped: bool) -> str:
    last = int(s[-1])
    txt = f"DUST [{schedule[0]}, {schedule[1]}], {n_dust_steps(schedule, last)} DUST steps"
    if stopped:
        txt += f" (stopped at step {last})"
    return txt


def stopped(runs_dir: str, run: str) -> bool:
    return not os.path.exists(os.path.join(runs_dir, run, "summary.json"))


def style_axes(ax, ylabel: str) -> None:
    ax.set_xlabel("optimizer step (hard and DUST steps)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 10000)
    ax.set_xticks(range(0, 10001, 2000))
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: "0" if x == 0 else f"{x / 1000:g}k"))


def target_figure(runs_dir: str, runs, target: str, out: str, ce_ylim, gap_ylim) -> None:
    hard = val_curve(runs_dir, HARD)
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.0, 5.0), dpi=200)
    a.plot(hard[0], hard[1], color="black", lw=1.6, label="hard Top-4 (no DUST steps)")
    b.axhline(0.0, color="black", lw=1.6, label="hard Top-4 (no DUST steps)")
    for run, sched in runs:
        s, v = val_curve(runs_dir, run)
        st = stopped(runs_dir, run)
        lab = label(sched, s, st)
        a.plot(s, v, color=COLORS[sched], lw=1.4, label=lab)
        gs, gd = gap(runs_dir, run, hard)
        b.plot(gs, gd, color=COLORS[sched], lw=1.8, label=lab)
        late = gd[gs >= 1000]
        if late.size and late.min() > gap_ylim[1]:
            # off the scale of (b): say where it is instead of squeezing the others
            b.annotate(f"DUST [{sched[0]}, {sched[1]}] above this range: "
                       f"+{late.min():.2f} to +{late.max():.2f} from step 1k on",
                       xy=(0.98, 0.97), xycoords="axes fraction", ha="right", va="top",
                       fontsize=9, color=COLORS[sched])
    style_axes(a, "validation CE (nats)")
    a.set_ylim(*ce_ylim)
    a.set_title(f"(a) validation CE, dust_target {target}")
    style_axes(b, "validation CE minus hard Top-4 (nats)")
    b.set_ylim(*gap_ylim)
    b.set_title(f"(b) difference to the hard run, {SMOOTH * 20}-step moving average")
    handles, labels = b.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def pair_figure(runs_dir: str, out: str, ylim) -> None:
    """Encoder vs backprop, one panel per schedule."""
    hard = val_curve(runs_dir, HARD)
    enc = dict((sched, run) for run, sched in ENCODER)
    bp = dict((sched, run) for run, sched in BACKPROP)
    scheds = [(10, 1), (30, 1), (90, 1)]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), dpi=200, sharey=True)
    for ax, sched in zip(axes, scheds):
        ax.axhline(0.0, color="black", lw=1.4, label="hard Top-4")
        for run, ls, name in ((enc[sched], "-", "encoder"), (bp[sched], "--", "backprop")):
            s, d = gap(runs_dir, run, hard)
            ax.plot(s, d, color=COLORS[sched], ls=ls, lw=1.9, label=f"dust_target {name}")
        style_axes(ax, "validation CE minus hard Top-4 (nats)" if sched == scheds[0] else "")
        ax.set_ylim(*ylim)
        ax.set_title(f"DUST [{sched[0]}, {sched[1]}], "
                     f"{n_dust_steps(sched, 10000)} DUST steps")
        ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def lr_figure(runs_dir: str, out: str, ylim) -> None:
    """dust_lr_scale 1 vs 3 at [10, 1], one panel per target."""
    hard = val_curve(runs_dir, HARD)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), dpi=200, sharey=True)
    pairs = (("encoder", "wa_k4_dust_p512_h10", "wa_k4_dust_p512_h10_lr3"),
             ("backprop", "wa_k4_dust_p512_bp_h10", "wa_k4_dust_p512_bp_h10_lr3"))
    for i, (ax, (target, r1, r3)) in enumerate(zip(axes, pairs)):
        ax.axhline(0.0, color="black", lw=1.4, label="hard Top-4")
        for run, color, scale in ((r1, COLORS[(10, 1)], 1), (r3, LR3_COLOR, 3)):
            s, d = gap(runs_dir, run, hard)
            ax.plot(s, d, color=color, lw=1.9,
                    label=f"dust_lr_scale {scale} (DUST lr {6e-4 * scale:.1e})")
        style_axes(ax, "validation CE minus hard Top-4 (nats)" if i == 0 else "")
        ax.set_ylim(*ylim)
        ax.set_title(f"DUST [10, 1], dust_target {target}")
        ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out-dir", default=os.path.join("docs", "dust", "figures"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    o = lambda name: os.path.join(args.out_dir, name)  # noqa: E731
    target_figure(args.runs_dir, ENCODER, "encoder", o("small_sched_encoder.png"),
                  ce_ylim=(2.7, 3.7), gap_ylim=(-0.03, 0.09))
    target_figure(args.runs_dir, BACKPROP, "backprop", o("small_sched_backprop.png"),
                  ce_ylim=(2.7, 3.7), gap_ylim=(-0.03, 0.09))
    pair_figure(args.runs_dir, o("small_sched_enc_vs_bp.png"), ylim=(-0.03, 0.09))
    lr_figure(args.runs_dir, o("small_sched_lr3.png"), ylim=(-0.03, 0.09))
    print("wrote", ", ".join(sorted(f for f in os.listdir(args.out_dir)
                                    if f.startswith("small_sched_"))))


if __name__ == "__main__":
    main()
