"""Per-family validation-curve figures for the MD-vs-MD-init comparison.

The two surrogate families are split into three panels -- (K=32, T=1),
(K=128, T=1), (K=128, T=2) -- with color encoding J within a panel and line
style encoding the training regime (MD solid, MD-init wd=0.1 dashed,
MD-init wd=0.01 dotted).  A cross marks the point where a run was stopped
as diverged (clamped to the top edge if the curve left the range); an open
circle marks the last point of a run still training at snapshot time.
The hard family has no temperature and stays a single panel.

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

J_COLORS = ["#0173B2", "#DE8F05"]          # low J, high J within a panel
STYLES = {  # regime -> (linestyle, linewidth, alpha)
    "md":            ("-",  2.2, 1.0),
    "mdinit_wd0.1":  ("--", 1.7, 0.95),
    "mdinit_wd0.01": (":",  1.9, 0.95),
}
REGIME_LABEL = {"md": "MD (decouple)",
                "mdinit_wd0.1": "MD init, AdamW wd=0.1",
                "mdinit_wd0.01": "MD init, AdamW wd=0.01"}
YLIM = (1.39, 3.1)


def plot_panel(ax, runs, color_key, fam):
    """Draw one panel; returns (finals, any_death, any_live)."""
    finals, deaths, live = [], [], []
    keys = sorted({color_key(r) for r in runs.values()})
    colors = {k: J_COLORS[i % len(J_COLORS)] for i, k in enumerate(keys)}
    for name, rec in sorted(runs.items()):
        if not rec["val"]:
            continue
        col = colors[color_key(rec)]
        ls, lw, al = STYLES[rec["regime"] if rec["regime"] in STYLES else "md"]
        s = [p[0] for p in rec["val"]]
        v = [p[1] for p in rec["val"]]
        ax.plot(s, v, ls, color=col, lw=lw, alpha=al)
        if "diverged" in rec:
            deaths.append((s[-1], v[-1], col))
        elif "summary" not in rec:
            live.append((s[-1], v[-1], col))
        finals.append((name, v[-1], s[-1], "diverged" in rec))
    ax.set_ylim(*YLIM)
    ax.set_xlim(0, 20000)
    hi = YLIM[1]
    for sx, vy, col in deaths:
        ax.plot(sx, min(vy, hi - 0.03), "x", color=col, ms=10, mew=2.4)
    for sx, vy, col in live:
        ax.plot(sx, min(vy, hi - 0.03), "o", color=col, ms=7, mfc="white", mew=1.8)
    ax.grid(alpha=0.3)
    ax.set_xlabel("training step")
    handles = [Line2D([0], [0], color=colors[k], lw=2.6,
                      label=("$K=%d$" % k if fam == "hard" else "$J=%d$" % k))
               for k in keys]
    ax.legend(handles=handles, loc="upper right", fontsize=10, framealpha=0.92)
    return finals, bool(deaths), bool(live)


def style_legend(target, any_death, any_live, **kw):
    handles = [Line2D([0], [0], color="0.25", ls=STYLES[r][0], lw=STYLES[r][1],
                      label=REGIME_LABEL[r]) for r in STYLES]
    if any_death:
        handles.append(Line2D([0], [0], color="0.25", ls="", marker="x", ms=8,
                              mew=2.0, label="stopped as diverged"))
    if any_live:
        handles.append(Line2D([0], [0], color="0.25", ls="", marker="o", ms=7,
                              mfc="white", mew=1.6, label="still training"))
    target.legend(handles=handles, fontsize=10, framealpha=0.92, **kw)


def family_runs(prefix):
    # the _b00_ runs belong to the kappa-ablation campaign, not this comparison
    return {n: r for n, r in curves.items()
            if n.startswith(prefix) and "_b00_" not in n}


all_finals = []

# ---- surrogate families: three panels ------------------------------------ #
for fam, prefix, title in (
        ("soft", "ca_rout_soft_", "Basic LapSum (rout_soft)"),
        ("rblapsum_kappa", "ca_rout_rblapsum_kappa_",
         "RBLapSum through_rank_kappa, $b_0=0.1$")):
    runs = family_runs(prefix)
    if not runs:
        continue
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9), sharey=True)
    panels = [(32, 1.0, "$K=32$, $T=1$"), (128, 1.0, "$K=128$, $T=1$"),
              (128, 2.0, "$K=128$, $T=2$")]
    any_death = any_live = False
    for ax, (K, T, ptitle) in zip(axes, panels):
        sel = {n: r for n, r in runs.items()
               if r["k"] == K and r.get("T", 1.0) == T}
        finals, d, l = plot_panel(ax, sel, lambda r: r["j"], fam)
        all_finals += [(fam,) + f for f in finals]
        any_death |= d
        any_live |= l
        ax.set_title(ptitle, fontsize=12)
    axes[0].set_ylabel("validation CE (nats)")
    style_legend(axes[1], any_death, any_live, loc="center")
    fig.suptitle(title, fontsize=14, y=1.005)
    fig.tight_layout()
    p = os.path.join(outdir, "mdinit_val_%s.png" % fam)
    fig.savefig(p, dpi=170, bbox_inches="tight")
    print(p)

# ---- hard family: single panel ------------------------------------------- #
runs = family_runs("ca_rout_hard_")
if runs:
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    finals, d, l = plot_panel(ax, runs, lambda r: r["k"], "hard")
    all_finals += [("hard",) + f for f in finals]
    ax.set_ylabel("validation CE (nats)")
    ax.set_title("Hard AbsTopK (no surrogate gradient)", fontsize=13)
    style_legend(ax, d, l, loc="upper center")
    fig.tight_layout()
    p = os.path.join(outdir, "mdinit_val_hard.png")
    fig.savefig(p, dpi=170)
    print(p)

for fam, name, v, s, dv in sorted(all_finals, key=lambda t: (t[0], t[2])):
    print("   %-52s last val %.4f @ %-5d%s" % (name, v, s, " DIVERGED" if dv else ""))
