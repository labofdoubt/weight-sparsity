"""Post-norm Top-(K+J) versus hard Top-K' figures for docs/main.tex.

Two four-panel figures (2 x 2) from the stream-carried (no code skip) K+J
campaign of docs/kj-vs-hard-topk.tex, post-norm runs only:

  kj_active_pnorm.png  one panel per active count K = 32, 64, 128, 256: every
                       candidate window J at that K (solid, blues, light to
                       dark with increasing J) against all five hard Top-K'
                       baselines (dashed, reds), each hard curve labelled with
                       its K' at its right end.
  kj_pool_pnorm.png    one panel per candidate pool K' = K+J = 64, 128, 256,
                       512: the hard Top-K' run (dashed, black, labelled at its
                       right end) against every surrogate cell with that pool.

Usage: python scripts/plot_main_kj.py OUTDIR CURVES.json [CURVES2.json ...]
           [--only active|pool] [--suffix _cr_pnorm_fo] [--code-residual]
CURVES.json maps run name -> {k, j, T, post_norm, surrogate, val: [[step, ce], ...]}
(scripts/extract_val_curves.py; docs/figures/main/data/curves_kj_pnorm.json is
a subset of docs/figures/kj-cr/data/curves_old.json).  --code-residual keeps
only code-residual runs (default: only stream-carried ones); a diverged run
ends in a cross.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ap = argparse.ArgumentParser()
ap.add_argument("outdir")
ap.add_argument("curves", nargs="+")
ap.add_argument("--only", choices=("active", "pool"), default=None)
ap.add_argument("--suffix", default="", help="appended to the figure file names")
ap.add_argument("--code-residual", action="store_true")
ap.add_argument("--ylim", type=float, nargs=2, default=None)
args = ap.parse_args()
outdir = args.outdir
os.makedirs(outdir, exist_ok=True)
raw = {}
for path in args.curves:
    raw.update(json.load(open(path)))

ACTIVE = [32, 64, 128, 256]
POOLS = [64, 128, 256, 512]
YLIM = tuple(args.ylim) if args.ylim else ((1.30, 2.0) if args.code_residual else (1.40, 2.0))
XMAX = 20000
J_CMAP, H_CMAP = matplotlib.colormaps["Blues"], matplotlib.colormaps["Reds"]
J_RANGE, H_RANGE = (0.35, 0.95), (0.35, 0.92)

kappa, hard = {}, {}
for n, r in raw.items():
    if not r.get("post_norm") or not r.get("val"):
        continue
    if bool(r.get("code_residual", False)) != args.code_residual:
        continue
    if r.get("surrogate") == "hard" or "rout_hard" in n:
        hard[r["k"]] = r
    elif r.get("surrogate") == "rblapsum" or "rblapsum_kappa" in n:
        kappa[(r["k"], r["j"])] = r


def shades(vals, cmap, rng):
    lo, hi = rng
    if len(vals) == 1:
        return {vals[0]: cmap(0.5 * (lo + hi))}
    return {v: cmap(lo + (hi - lo) * i / (len(vals) - 1)) for i, v in enumerate(vals)}


def curve(ax, rec, color, ls, lw):
    s = [p[0] for p in rec["val"]]
    v = [p[1] for p in rec["val"]]
    ax.plot(s, v, ls, color=color, lw=lw)
    if "diverged" in rec:
        ax.plot(s[-1], min(v[-1], YLIM[1] - 0.02 * (YLIM[1] - YLIM[0])), "x",
                color=color, ms=10, mew=2.4, clip_on=False)


def end_label(ax, rec, text, color):
    """The hard run's K' just right of its last point (outside the axes)."""
    s, v = rec["val"][-1]
    ax.annotate(text, xy=(s, v), xytext=(4, 0), textcoords="offset points",
                va="center", ha="left", fontsize=9, color=color,
                annotation_clip=False)


def frame(ax, row, col):
    ax.set_xlim(0, XMAX)
    ax.set_ylim(*YLIM)
    ax.grid(alpha=0.3)
    if row == 1:
        ax.set_xlabel("training step")
    if col == 0:
        ax.set_ylabel("validation CE (nats)")


def layout():
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.16, hspace=0.16)
    return fig, axes


hard_ks = sorted(hard)
hcol = shades(hard_ks, H_CMAP, H_RANGE)

# ---- active count -------------------------------------------------------- #
if args.only in (None, "active"):
  fig, axes = layout()
  for n, K in enumerate(ACTIVE):
      ax = axes.flat[n]
      js = sorted(j for (k, j) in kappa if k == K)
      col = shades(js, J_CMAP, J_RANGE)
      for kp in hard_ks:
          curve(ax, hard[kp], hcol[kp], "--", 1.6)
          end_label(ax, hard[kp], f"$K={kp}$", hcol[kp])
      for j in js:
          curve(ax, kappa[(K, j)], col[j], "-", 2.1)
      frame(ax, n // 2, n % 2)
      ax.set_title(f"active count $K={K}$", fontsize=12)
      h = [Line2D([0], [0], color=col[j], lw=2.2, label=f"RBLapSum $K={K}$, $J={j}$") for j in js]
      h.append(Line2D([0], [0], color=hcol[hard_ks[-1]], ls="--", lw=1.6, label="hard Top-$K$"))
      ax.legend(handles=h, fontsize=9, loc="upper right", framealpha=0.93)
  p = os.path.join(outdir, f"kj_active_pnorm{args.suffix}.png")
  fig.savefig(p, dpi=170, bbox_inches="tight")
  print(p)

# ---- candidate pool ------------------------------------------------------ #
if args.only in (None, "pool"):
  fig, axes = layout()
  for n, Kp in enumerate(POOLS):
      ax = axes.flat[n]
      js = sorted(j for (k, j) in kappa if k + j == Kp)
      col = shades(js, J_CMAP, J_RANGE)
      if Kp in hard:
          curve(ax, hard[Kp], "0.15", "--", 1.8)
          end_label(ax, hard[Kp], f"$K={Kp}$", "0.15")
      for j in js:
          curve(ax, kappa[(Kp - j, j)], col[j], "-", 2.1)
      frame(ax, n // 2, n % 2)
      ax.set_title(f"candidate pool $K+J={Kp}$", fontsize=12)
      h = [Line2D([0], [0], color=col[j], lw=2.2, label=f"RBLapSum $K={Kp - j}$, $J={j}$") for j in js]
      h.append(Line2D([0], [0], color="0.15", ls="--", lw=1.8, label=f"hard Top-$K$, $K={Kp}$"))
      ax.legend(handles=h, fontsize=9, loc="upper right", framealpha=0.93)
  p = os.path.join(outdir, f"kj_pool_pnorm{args.suffix}.png")
  fig.savefig(p, dpi=170, bbox_inches="tight")
  print(p)

# ---- final values, for the text ------------------------------------------ #
for kp in hard_ks:
    print(f"hard K={kp}: {hard[kp]['val'][-1][1]:.4f}")
for (k, j) in sorted(kappa):
    r = kappa[(k, j)]
    tag = f"  (diverged at {r['diverged'].get('step')})" if "diverged" in r else ""
    print(f"K={k} J={j}: {r['val'][-1][1]:.4f} at step {r['val'][-1][0]}{tag}")
