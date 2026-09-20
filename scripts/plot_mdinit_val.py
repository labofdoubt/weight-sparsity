"""Per-family validation-curve figures for the MD-vs-MD-init comparison.

Color encodes the (k,j) pair; line style encodes the training regime
(MD solid, MD-init wd=0.1 dashed, MD-init wd=0.01 dotted).  A cross marks
the last point of a run that the divergence guard stopped.

Usage: python figures.py mdinit_curves.json outdir/
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

curves = json.load(open(sys.argv[1]))
outdir = sys.argv[2]
os.makedirs(outdir, exist_ok=True)

# fixed hue per (k,j,T): colorblind-safe categoricals; the T=2 members of a
# (k,j) pair wear a lighter shade of the T=1 hue
PAIR_COLORS = {
    (32, 32, 1.0):    "#0173B2",
    (32, 480, 1.0):   "#DE8F05",
    (128, 128, 1.0):  "#029E73",
    (128, 384, 1.0):  "#CC3311",
    (128, 128, 2.0):  "#7CCBA2",
    (128, 384, 2.0):  "#F49081",
    (32, 64, 1.0):    "#0173B2",   # hard family (T inert)
    (512, 64, 1.0):   "#DE8F05",
}
STYLES = {  # regime -> (linestyle, linewidth, alpha)
    "md":           ("-",  2.2, 1.0),
    "mdinit_wd0.1": ("--", 1.7, 0.95),
    "mdinit_wd0.01": (":", 1.9, 0.95),
}
REGIME_LABEL = {"md": "MD (decouple)",
                "mdinit_wd0.1": "MD init, AdamW wd=0.1",
                "mdinit_wd0.01": "MD init, AdamW wd=0.01"}

FAMILIES = {
    "soft": ("ca_rout_soft_", "Basic LapSum (rout_soft), constant absolute temperature"),
    "rblapsum_kappa": ("ca_rout_rblapsum_kappa_",
                       "RBLapSum through_rank_kappa, $b_0=0.1$"),
    "hard": ("ca_rout_hard_", "Hard AbsTopK (no surrogate gradient)"),
}

def regime_key(rec):
    r = rec["regime"]
    return {"mdinit_wd0.1": "mdinit_wd0.1", "mdinit_wd0.01": "mdinit_wd0.01"}.get(r, r)

for fam, (prefix, title) in FAMILIES.items():
    runs = {n: r for n, r in curves.items() if n.startswith(prefix)}
    if not runs:
        print("no runs yet for", fam)
        continue
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    deaths, live = [], []
    # the hard family has no surrogate temperature; its config carries an
    # inert schedule field, so pin T in the identity key
    def ident(r):
        return (r["k"], r["j"], 1.0 if fam == "hard" else r.get("T", 1.0))
    pairs = sorted({ident(r) for r in runs.values()})
    finals = []
    for name, rec in sorted(runs.items()):
        if not rec["val"]:
            continue
        kj = ident(rec)
        reg = regime_key(rec)
        ls, lw, al = STYLES[reg]
        s = [p[0] for p in rec["val"]]
        v = [p[1] for p in rec["val"]]
        ax.plot(s, v, ls, color=PAIR_COLORS[kj], lw=lw, alpha=al)
        if "diverged" in rec:
            deaths.append((s[-1], v[-1], PAIR_COLORS[kj]))
        elif "summary" not in rec:  # still training at snapshot time
            live.append((s[-1], v[-1], PAIR_COLORS[kj]))
        finals.append((name, v[-1], s[-1], "diverged" in rec))
    ax.set_xlabel("training step")
    ax.set_ylabel("validation CE (nats)")
    ax.set_title(title)
    lo = min(f[1] for f in finals if not f[3]) if any(not f[3] for f in finals) else 1.4
    hi = min(4.2, max(2.6, lo + 1.6))
    ax.set_ylim(lo - 0.08, hi)
    ax.set_xlim(0, 20000)
    # divergence / in-progress end markers, clamped into the axes so a curve
    # that exits the top still shows where it was stopped
    for sx, vy, col in deaths:
        ax.plot(sx, min(vy, hi - 0.02), "x", color=col, ms=10, mew=2.4)
    for sx, vy, col in live:
        ax.plot(sx, min(vy, hi - 0.02), "o", color=col, ms=7, mfc="white", mew=1.8)
    ax.grid(alpha=0.3)
    def pair_label(p):
        if fam == "hard":
            return f"$K={p[0]}$"
        return f"$K={p[0]}, J={p[1]}, T={p[2]:g}$"
    color_handles = [Line2D([0], [0], color=PAIR_COLORS[p], lw=2.6,
                            label=pair_label(p)) for p in pairs]
    style_handles = [Line2D([0], [0], color="0.25", ls=STYLES[r][0], lw=STYLES[r][1],
                            label=REGIME_LABEL[r]) for r in STYLES]
    if deaths:
        style_handles.append(Line2D([0], [0], color="0.25", ls="", marker="x",
                                    ms=8, mew=2.0, label="stopped as diverged"))
    if any(not f[3] and f[2] < 20000 for f in finals):
        style_handles.append(Line2D([0], [0], color="0.25", ls="", marker="o",
                                    ms=7, mfc="white", mew=1.6,
                                    label="still training"))
    leg1 = ax.legend(handles=color_handles, loc="lower left", fontsize=10,
                     framealpha=0.92)
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc="upper center", fontsize=10,
              framealpha=0.92)
    fig.tight_layout()
    p = os.path.join(outdir, f"mdinit_val_{fam}.png")
    fig.savefig(p, dpi=170)
    print(p)
    for name, v, s, dv in sorted(finals, key=lambda t: t[1]):
        print(f"   {name:52s} last val {v:.4f} @ {s}{' DIVERGED' if dv else ''}")
