"""Top(K+J) with the RBLapSum kappa surrogate versus hard Top-K'.

Two figure families, both 2x2 (left T=2, right post-norm; lower row the same
panels with the vertical axis capped at 2.0 nats):

  by_kprime_K<K'>   one candidate-pool size K' = K+J: the hard Top-K' run
                    (dashed, black) against every kappa run whose pool is K',
                    solid, coloured light-to-dark with increasing J.
  by_k_K<K>         one active count K: every available J for that K (solid,
                    light-to-dark with increasing J) together with all hard
                    Top-K' runs (dashed, a second ramp, light-to-dark with
                    increasing K').

The hard baselines are the post-norm ones throughout.

Usage: python plot_kj.py outdir/ curves.json [curves2.json ...]
"""
import json, os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm

outdir, jsons = sys.argv[1], sys.argv[2:]
os.makedirs(outdir, exist_ok=True)
raw = {}
for p in jsons:
    raw.update(json.load(open(p)))

KS = [32, 64, 128, 256, 512]
YLIM_FULL, YLIM_CAP = (1.40, 3.1), (1.40, 2.0)
J_CMAP, H_CMAP = cm.get_cmap("viridis"), cm.get_cmap("copper")


def classify(name, rec):
    """-> ('kappa', K, J, variant) | ('hard', K', None, 'pnorm') | None"""
    if rec.get("surrogate") == "hard" or "rout_hard" in name:
        if "pnorm" not in name:
            return None                      # only post-norm hard baselines
        return ("hard", rec["k"], None, "pnorm")
    if "rblapsum_kappa" not in name:
        return None
    if any(t in name for t in ("mdinit", "supp03", "lr2e4", "through_rank")):
        return None
    var = None
    if "pnorm" in name:
        var = "pnorm"
    elif rec.get("T") == 2.0:
        var = "t2"
    if var is None:
        return None
    return ("kappa", rec["k"], rec["j"], var)


def priority(name):
    """Which run wins when one cell was trained more than once.

    Explicit and deterministic, best first: the homogeneous b0=0 campaigns
    (stabilization, then this one, then the kappa ablation), and only then an
    older b0=0.1 run.  Without a fixed order the choice falls out of dict
    iteration and two scripts can disagree on the same cell.
    """
    for i, tag in enumerate(("_stab_", "_kj_", "_b00_")):
        if tag in name:
            return i
    return 9


kappa, hard = {}, {}
for n, r in sorted(raw.items()):
    if not r.get("val"):
        continue
    c = classify(n, r)
    if c is None:
        continue
    kind, k, j, var = c
    key = (k, j, var) if kind == "kappa" else (k, var)
    tgt = kappa if kind == "kappa" else hard
    if key not in tgt or priority(n) < priority(tgt[key][0]):
        tgt[key] = (n, r)


def curve(ax, rec, color, ls, lw, ylim):
    s = [p[0] for p in rec["val"]]
    v = [p[1] for p in rec["val"]]
    ax.plot(s, v, ls, color=color, lw=lw)
    if "diverged" in rec:
        span = ylim[1] - ylim[0]
        ax.plot(s[-1], min(v[-1], ylim[1] - 0.02 * span), "x", color=color,
                ms=10, mew=2.4)


def finish(ax, ylim, xlabel, ylabel):
    ax.set_ylim(*ylim)
    ax.set_xlim(0, 20000)
    ax.grid(alpha=0.3)
    if xlabel:
        ax.set_xlabel("training step")
    if ylabel:
        ax.set_ylabel(ylabel)


def shades(vals, cmap, lo=0.78, hi=0.06):
    """Light for the smallest value, dark for the largest (lo > hi inverts the
    ramp), so 'more J' and 'more K-prime' read as 'more intense'."""
    if len(vals) == 1:
        return {vals[0]: cmap(0.45)}
    return {v: cmap(lo + (hi - lo) * i / (len(vals) - 1))
            for i, v in enumerate(vals)}


# ---- family 1: one figure per candidate-pool size K' --------------------- #
for Kp in KS:
    js = sorted({j for (k, j, v) in kappa if k + j == Kp})
    if not js:
        continue
    col = shades(js, J_CMAP)
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4), sharex="col", sharey="row")
    for c, var in enumerate(("t2", "pnorm")):
        for row, ylim in enumerate((YLIM_FULL, YLIM_CAP)):
            ax = axes[row][c]
            if (Kp, "pnorm") in hard:
                curve(ax, hard[(Kp, "pnorm")][1], "0.15", "--", 1.9, ylim)
            for j in js:
                k = Kp - j
                if (k, j, var) in kappa:
                    curve(ax, kappa[(k, j, var)][1], col[j], "-", 2.1, ylim)
            finish(ax, ylim, row == 1,
                   ("validation CE (nats)" if row == 0 else
                    "validation CE (nats), capped at 2.0") if c == 0 else None)
        axes[0][c].set_title("$T=2$" if var == "t2" else "post-norm", fontsize=12)
    h = [Line2D([0], [0], color="0.15", ls="--", lw=1.9,
                label=f"hard Top-$K'$, $K'={Kp}$ (post-norm)")]
    h += [Line2D([0], [0], color=col[j], lw=2.2,
                 label=f"$K={Kp - j}$, $J={j}$") for j in js]
    axes[0][1].legend(handles=h, fontsize=9, loc="upper right", framealpha=0.93)
    fig.suptitle(f"Candidate pool $K+J=K'={Kp}$: RBLapSum kappa vs hard Top-$K'$",
                 fontsize=14, y=0.995)
    fig.tight_layout()
    p = os.path.join(outdir, f"kj_pool_k{Kp}.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    print(p)

# ---- family 2: one figure per active count K ----------------------------- #
hard_ks = sorted({k for (k, v) in hard})
hcol = shades(hard_ks, H_CMAP)
for K in KS:
    js = sorted({j for (k, j, v) in kappa if k == K})
    if not js:
        continue
    col = shades(js, J_CMAP)
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4), sharex="col", sharey="row")
    for c, var in enumerate(("t2", "pnorm")):
        for row, ylim in enumerate((YLIM_FULL, YLIM_CAP)):
            ax = axes[row][c]
            for kp in hard_ks:
                curve(ax, hard[(kp, "pnorm")][1], hcol[kp], "--", 1.7, ylim)
            for j in js:
                if (K, j, var) in kappa:
                    curve(ax, kappa[(K, j, var)][1], col[j], "-", 2.1, ylim)
            finish(ax, ylim, row == 1,
                   ("validation CE (nats)" if row == 0 else
                    "validation CE (nats), capped at 2.0") if c == 0 else None)
        axes[0][c].set_title("$T=2$" if var == "t2" else "post-norm", fontsize=12)
    h = [Line2D([0], [0], color=col[j], lw=2.2, label=f"$K={K}$, $J={j}$")
         for j in js]
    h += [Line2D([0], [0], color=hcol[kp], ls="--", lw=1.7,
                 label=f"hard Top-${kp}$") for kp in hard_ks]
    axes[0][1].legend(handles=h, fontsize=9, loc="upper right", framealpha=0.93,
                      ncol=2)
    fig.suptitle(f"Active count $K={K}$: every candidate window, against all "
                 f"hard Top-$K'$", fontsize=14, y=0.995)
    fig.tight_layout()
    p = os.path.join(outdir, f"kj_active_k{K}.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    print(p)

# ---- the two tables ------------------------------------------------------ #
def cell(entry):
    if entry is None:
        return "--"
    n, r = entry
    if "diverged" in r:
        return "div@%d" % r["diverged"]["step"]
    return "%.4f" % r["val"][-1][1]


print()
print("=== cell provenance ===")
for key in sorted(kappa):
    print("   K=%-4d J=%-4d %-6s <- %s" % (key[0], key[1], key[2], kappa[key][0]))
for key in sorted(hard):
    print("   hard K'=%-4d %-6s <- %s" % (key[0], key[1], hard[key][0]))

for var, lab in (("t2", "T=2"), ("pnorm", "post-norm")):
    print()
    print("=== %s: rows K, columns K'=K+J ===" % lab)
    print("%-8s %s" % ("K", "".join("%12s" % ("K'=%d" % kp) for kp in KS[1:])))
    for K in KS[:-1]:
        row = "".join("%12s" % cell(kappa.get((K, kp - K, var))) for kp in KS[1:])
        print("%-8d %s" % (K, row))
    print("%-8s %s" % ("hard", "".join("%12s" % cell(hard.get((kp, "pnorm")))
                                       for kp in KS[1:])))
