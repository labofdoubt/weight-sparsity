"""Animate the viewer's half-line panel across probe steps and write a GIF.

Reproduces `score_explorer.py`'s half-line for one cell of an early-training
probe dataset, one frame per probe step: raw |score| on a log x axis with a
minimum-x display cut, TopK / J-candidate / rest marks in the viewer's colours
and sizes, deterministic per-feature jitter (so a circle moves only when the
model does), rings against the previous probe step (black = was TopK, magenta
= was a J candidate), k / k+j edges at the score-threshold midpoints, and the
mode-appropriate temperature band: violet b ± t for the prescribed LapSum
modes, TopK-boundary ± T for swap_gibbs, none for hard.  The x range is frozen
across frames so motion is training dynamics, not axis rescaling.

    python analysis/halfline_gif.py --run probe_dc_rout_soft_k32_j128 \
        --layer 4 --seq 0 --token 63 --min-x-log -1.0 \
        --out /workspace/plots/halfline_probe_dc_rout_soft_k32_j128_L4_s0_t63.gif

Needs matplotlib + pillow (and torch/wsparse only when a band is drawn).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# the viewer's palette, verbatim
BAND_COLOR = {"topk": "#2a78d6", "cand": "#eb6834", "rest": "#1baf7a"}
INK, INK_MUTED = "#0b0b0b", "#52514e"
SURFACE, GRID = "#fcfcfb", "#e8e7e3"
RING_WAS_TOPK, RING_WAS_CAND = INK, "#d55181"
BARRIER, BARRIER_FILL = "#4a3aa7", (74 / 255, 58 / 255, 167 / 255, 0.10)


def band_at(meta, band_sorted, t_sched):
    """(center, t, label) for one frame's sorted candidate band, or None."""
    mode = meta.get("surrogate_mode", "")
    tc = meta.get("temperature")
    if tc is None or mode == "hard":
        return None
    import torch
    k = int(meta["k"])
    cand = torch.from_numpy(band_sorted[None].copy())
    scale = float(cand.std(-1, unbiased=True)) if tc["scale_mode"] == "relative" else 1.0
    t = t_sched * (scale if scale > 0 else 1.0)
    if mode == "swap_gibbs":
        return (float(band_sorted[k - 1] + band_sorted[k]) / 2.0, t, "K|K+1")
    from wsparse.bottleneck.lapsum import lapsum_barrier_sorted
    b = lapsum_barrier_sorted(cand, k, torch.tensor([t], dtype=cand.dtype))
    return (float(b[0]), t, "b")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="probe dataset name (no extension)")
    ap.add_argument("--probe-dir", default="/workspace/analysis/probe")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--token", type=int, required=True)
    ap.add_argument("--min-x-log", type=float, default=-1.0,
                    help="display cut, log10 of the minimum |score| (viewer slider units)")
    ap.add_argument("--every", type=int, default=1, help="use every Nth probe step")
    ap.add_argument("--duration-ms", type=int, default=120)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    meta = json.load(open(f"{a.probe_dir}/{a.run}.json"))
    k = int(meta["k"])
    j = 0 if meta.get("surrogate_mode") == "hard" else int(meta["j"])
    steps = meta["steps"]
    S = np.load(f"{a.probe_dir}/{a.run}.score.npy", mmap_mode="r")
    N = S.shape[-1]
    cell = np.abs(np.asarray(S[:, a.layer, a.seq, a.token], dtype=np.float64))
    cut = 10.0 ** a.min_x_log
    x_hi = float(cell.max()) * 1.15
    jt = np.random.default_rng(0).uniform(-1.0, 1.0, size=N)

    t_scheds = None
    if meta.get("surrogate_mode") not in ("hard", None) and meta.get("temperature"):
        tc = meta["temperature"]
        if meta["surrogate_mode"] == "lapsum_fixed":
            t_scheds = [float(tc["fixed"])] * len(steps)
        else:
            from wsparse.schedules import build_schedule
            sched = build_schedule(kind=tc["schedule"], start=tc["start"], end=tc["end"],
                                   warmup_steps=tc["warmup_steps"],
                                   anneal_steps=tc["anneal_steps"], power=tc["power"],
                                   max_steps=tc["max_steps"])
            t_scheds = [float(sched(s)) for s in steps]

    frames = []
    prev_topk = prev_cand = None
    for ci in range(0, cell.shape[0], a.every):
        r = cell[ci]
        order = np.argsort(-r)
        topk = np.zeros(N, bool); topk[order[:k]] = True
        cand = np.zeros(N, bool)
        if j:
            cand[order[k:k + j]] = True
        was_t = np.zeros(N, bool) if prev_topk is None else prev_topk
        was_c = np.zeros(N, bool) if prev_cand is None else prev_cand
        keep = r >= cut

        fig, ax = plt.subplots(figsize=(10.0, 2.9), dpi=110)
        fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
        bt = None
        if t_scheds is not None and j:
            bt = band_at(meta, r[order[:k + j]], t_scheds[ci])
        if bt is not None:
            c, t, lab = bt
            lo, hi = max(c - t, cut), c + t
            if hi > lo:
                ax.axvspan(lo, hi, color=BARRIER_FILL, zorder=1)
                for v in (lo, hi):
                    ax.axvline(v, color=BARRIER, lw=1, ls=":", zorder=2)
            if c >= cut:
                ax.axvline(c, color=BARRIER, lw=1.6, zorder=2)
                ax.annotate(lab, (c, -1.38), color=BARRIER, fontsize=9,
                            ha="left", va="bottom")
        for name, sel, size, alpha in (("rest", keep & ~topk & ~cand, 36, 0.45),
                                       ("cand", keep & cand, 100, 1.0),
                                       ("topk", keep & topk, 100, 1.0)):
            idx = np.where(sel)[0]
            if not idx.size:
                continue
            ring = np.where(was_t[idx], RING_WAS_TOPK,
                            np.where(was_c[idx], RING_WAS_CAND, SURFACE))
            ax.scatter(r[idx], jt[idx], s=size, c=BAND_COLOR[name], alpha=alpha,
                       zorder=3, linewidths=np.where(was_t[idx] | was_c[idx], 2.5, 1.0),
                       edgecolors=ring)
        edges = [(k, f"k={k}")] + ([] if not j else [(k + j, f"k+j={k + j}")])
        for n_e, (edge, lab) in enumerate(edges):
            mid = (r[order[edge - 1]] + r[order[min(edge, N - 1)]]) / 2.0
            if mid < cut:
                continue
            ax.axvline(mid, color=INK_MUTED, lw=1, zorder=2)
            ax.annotate(lab, (mid, 1.28), color=INK_MUTED, fontsize=9,
                        ha="left" if n_e == 0 else "right", va="top")
        ax.set_xscale("log"); ax.set_xlim(cut, x_hi); ax.set_ylim(-1.45, 1.45)
        ax.set_yticks([]); ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel("|score| (raw)", fontsize=9, color=INK_MUTED)
        ax.set_title(f"{a.run}  ·  bottleneck {a.layer} · seq {a.seq} · token {a.token}"
                     f"   —   step {steps[ci]:>4}", fontsize=10, color=INK, loc="left")
        note = (f"min x = 10^{a.min_x_log:g}: {int(keep.sum())} of {N} shown · "
                f"black ring = was TopK" + (" · magenta = was J cand" if j else "")
                + " at prev probe")
        ax.annotate(note, xy=(0.995, 0.02), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=8, color=INK_MUTED)
        fig.tight_layout(); fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)
        prev_topk, prev_cand = topk, cand

    out = a.out or (f"/workspace/plots/halfline_{a.run}_L{a.layer}"
                    f"_s{a.seq}_t{a.token}.gif")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:] + [frames[-1]] * 8,
                   duration=a.duration_ms, loop=0, optimize=True)
    print(f"wrote {out} ({os.path.getsize(out) / 2**20:.1f} MiB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
