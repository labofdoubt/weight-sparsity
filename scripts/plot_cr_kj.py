"""The K+J grid with the code residual: figures and tables.

Reads the curve JSONs written by scripts/extract_val_curves.py (any number;
later files override earlier ones) and sorts every run into one of two
families by its ``code_residual`` flag -- "stream" (the stream-carried models
of docs/kj-vs-hard-topk.tex) and "cr" (the same cells with the code carried)
-- and into hard Top-K' with the post-norm, or an RBLapSum kappa cell with
its variant (T=2 without output norm, or post-norm at T=1).

Figures (all with the vertical axis capped at 2.0 nats):

  cr_pool_k<K'>.png      code-residual family, one candidate pool K' = K+J:
                         hard Top-K' (dashed, black) against every kappa cell
                         with that pool (solid, blues light-to-dark with J);
                         left T=2, right post-norm.
  cr_active_k<K>.png     code-residual family, one active count K: every J for
                         that K (blues) against all five hard Top-K' (dashed,
                         reds light-to-dark with K').
  cr_vs_stream_hard.png  hard Top-K' with the post-norm, stream-carried
                         (dashed, reds light-to-dark with K') against
                         code-carried (solid, blues light-to-dark with K');
                         right panel the last 12k steps on a tight scale.
  cr_vs_stream_rbk_k<K>.png  one active count K: the stream-carried kappa
                         cells (dashed, reds light-to-dark with J) against the
                         code-carried ones (solid, blues light-to-dark with J);
                         left T=2, right post-norm.

Hue separates the families or the hard/kappa roles, lightness orders K' or J
within each.  Tables of final validation CE for the code-residual family (T=2
and post-norm, rows K, columns K', bottom row hard Top-K'), the per-cell
difference code-carried minus stream-carried, and the run behind every cell
are printed.

The supplementary first-order runs (code residual with
rblapsum_surrogate_scope=first_order, run for the cells whose pool-scope run
diverged) form a third family, "cr_fo": dotted blue in every figure that shows
their cell, and a table of their own.

Usage: python scripts/plot_cr_kj.py OUTDIR curves_old.json curves_new.json
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

outdir, jsons = sys.argv[1], sys.argv[2:]
os.makedirs(outdir, exist_ok=True)
raw = {}
for p in jsons:
    raw.update(json.load(open(p)))

KS = [32, 64, 128, 256, 512]
YCAP = 2.0
BLUES, REDS = cm.get_cmap("Blues"), cm.get_cmap("Reds")
B_RANGE, R_RANGE = (0.35, 0.95), (0.35, 0.92)


def classify(name, rec):
    """-> (family, kind, K, J, variant) or None.

    family: "cr" | "stream" | "cr_fo"; kind: "hard" | "kappa"; variant:
    "pnorm" | "t2" (hard baselines are the post-norm ones, variant "pnorm").
    "cr_fo" is the code residual with the first-order surrogate scope.
    """
    fam = "cr" if rec.get("code_residual") else "stream"
    if rec.get("surrogate") == "hard":
        if not rec.get("post_norm"):
            return None
        return (fam, "hard", rec["k"], None, "pnorm")
    if rec.get("surrogate") != "rblapsum" or rec.get("grad_mode") != "through_rank_kappa":
        return None
    scope = rec.get("scope", "pool")
    if scope != "pool":
        if fam != "cr" or not str(scope).startswith("first_order"):
            return None
        fam = "cr_fo"
    if float(rec.get("b0") or 0.0) != 0.0:
        return None
    if rec.get("post_norm"):
        var = "pnorm"
    elif float(rec.get("T") or 0.0) == 2.0:
        var = "t2"
    else:
        return None
    return (fam, "kappa", rec["k"], rec["j"], var)


def priority(name):
    """Which run wins a cell trained more than once: the homogeneous b0=0
    campaigns of the K+J note first (stabilization, then that note, then the
    kappa ablation), the madrid runs by name, anything else last."""
    for i, tag in enumerate(("_stab_", "_kj_", "_b00_", "ma_cr_")):
        if tag in name:
            return i
    return 9


cells = {}   # (family, kind, K, J, variant) -> (name, rec)
for n, r in sorted(raw.items()):
    if not r.get("val"):
        continue
    c = classify(n, r)
    if c is None:
        continue
    if c not in cells or priority(n) < priority(cells[c][0]):
        cells[c] = (n, r)


def get(fam, kind, k, j, var):
    return cells.get((fam, kind, k, j, var))


def shades(vals, cmap, rng):
    lo, hi = rng
    if len(vals) == 1:
        return {vals[0]: cmap(0.5 * (lo + hi))}
    return {v: cmap(lo + (hi - lo) * i / (len(vals) - 1)) for i, v in enumerate(vals)}


def curve(ax, rec, color, ls, lw, ylim):
    s = [p[0] for p in rec["val"]]
    v = [p[1] for p in rec["val"]]
    ax.plot(s, v, ls, color=color, lw=lw)
    if "diverged" in rec:
        span = ylim[1] - ylim[0]
        ax.plot(s[-1], min(v[-1], ylim[1] - 0.02 * span), "x", color=color, ms=10, mew=2.4)


def ylim_for(recs, cap=YCAP, floor=1.40):
    lo = floor
    for r in recs:
        if r is not None and r["val"]:
            lo = min(lo, min(p[1] for p in r["val"]) - 0.01)
    return (round(lo - 0.005, 2), cap)


def finish(ax, ylim, ylabel):
    ax.set_ylim(*ylim)
    ax.set_xlim(0, 20000)
    ax.grid(alpha=0.3)
    ax.set_xlabel("training step")
    if ylabel:
        ax.set_ylabel(ylabel)


def save(fig, fn):
    fig.tight_layout()
    p = os.path.join(outdir, fn)
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(p)


# ---- A: code-residual family, one figure per candidate pool K' ------------ #
cr_js_by_pool = {}
for (fam, kind, k, j, var) in cells:
    if fam == "cr" and kind == "kappa":
        cr_js_by_pool.setdefault(k + j, set()).add(j)
for Kp in KS:
    js = sorted(cr_js_by_pool.get(Kp, ()))
    if not js:
        continue
    col = shades(js, BLUES, B_RANGE)
    recs = [get("cr", "hard", Kp, None, "pnorm")] + [get("cr", "kappa", Kp - j, j, v)
                                                     for j in js for v in ("t2", "pnorm")]
    ylim = ylim_for([r[1] if r else None for r in recs])
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)
    for c, var in enumerate(("t2", "pnorm")):
        ax = axes[c]
        h = get("cr", "hard", Kp, None, "pnorm")
        if h:
            curve(ax, h[1], "0.15", "--", 1.9, ylim)
        for j in js:
            e = get("cr", "kappa", Kp - j, j, var)
            if e:
                curve(ax, e[1], col[j], "-", 2.1, ylim)
            fo = get("cr_fo", "kappa", Kp - j, j, var)
            if fo:
                curve(ax, fo[1], col[j], ":", 2.4, ylim)
        finish(ax, ylim, "validation CE (nats)" if c == 0 else None)
        ax.set_title("$T=2$" if var == "t2" else "post-norm", fontsize=12)
    hd = [Line2D([0], [0], color="0.15", ls="--", lw=1.9,
                 label=f"hard Top-$K'$, $K'={Kp}$ (post-norm)")]
    hd += [Line2D([0], [0], color=col[j], lw=2.2, label=f"$K={Kp - j}$, $J={j}$") for j in js]
    fo_js = [j for j in js if any(get("cr_fo", "kappa", Kp - j, j, v) for v in ("t2", "pnorm"))]
    hd += [Line2D([0], [0], color=col[j], ls=":", lw=2.4,
                  label=f"$K={Kp - j}$, $J={j}$, first-order scope") for j in fo_js]
    axes[1].legend(handles=hd, fontsize=9, loc="upper right", framealpha=0.93)
    fig.suptitle(f"Code residual, candidate pool $K+J=K'={Kp}$: RBLapSum kappa vs hard Top-$K'$",
                 fontsize=14, y=0.995)
    save(fig, f"cr_pool_k{Kp}.png")

# ---- B: code-residual family, one figure per active count K --------------- #
cr_hard_ks = sorted(k for (fam, kind, k, j, var) in cells if fam == "cr" and kind == "hard")
hcol = shades(cr_hard_ks, REDS, R_RANGE) if cr_hard_ks else {}
for K in KS:
    js = sorted(j for (fam, kind, k, j, var) in cells if fam == "cr" and kind == "kappa" and k == K)
    if not js:
        continue
    col = shades(js, BLUES, B_RANGE)
    recs = [get("cr", "hard", kp, None, "pnorm") for kp in cr_hard_ks]
    recs += [get("cr", "kappa", K, j, v) for j in js for v in ("t2", "pnorm")]
    ylim = ylim_for([r[1] if r else None for r in recs])
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)
    for c, var in enumerate(("t2", "pnorm")):
        ax = axes[c]
        for kp in cr_hard_ks:
            curve(ax, get("cr", "hard", kp, None, "pnorm")[1], hcol[kp], "--", 1.7, ylim)
        for j in js:
            e = get("cr", "kappa", K, j, var)
            if e:
                curve(ax, e[1], col[j], "-", 2.1, ylim)
            fo = get("cr_fo", "kappa", K, j, var)
            if fo:
                curve(ax, fo[1], col[j], ":", 2.4, ylim)
        finish(ax, ylim, "validation CE (nats)" if c == 0 else None)
        ax.set_title("$T=2$" if var == "t2" else "post-norm", fontsize=12)
    hd = [Line2D([0], [0], color=col[j], lw=2.2, label=f"$K={K}$, $J={j}$") for j in js]
    fo_js = [j for j in js if any(get("cr_fo", "kappa", K, j, v) for v in ("t2", "pnorm"))]
    hd += [Line2D([0], [0], color=col[j], ls=":", lw=2.4,
                  label=f"$K={K}$, $J={j}$, first-order scope") for j in fo_js]
    hd += [Line2D([0], [0], color=hcol[kp], ls="--", lw=1.7, label=f"hard Top-${kp}$")
           for kp in cr_hard_ks]
    axes[1].legend(handles=hd, fontsize=9, loc="upper right", framealpha=0.93, ncol=2)
    fig.suptitle(f"Code residual, active count $K={K}$: every candidate window, "
                 f"against all hard Top-$K'$", fontsize=14, y=0.995)
    save(fig, f"cr_active_k{K}.png")

# ---- C: hard Top-K', stream-carried vs code-carried ----------------------- #
hard_ks = sorted({k for (fam, kind, k, j, var) in cells if kind == "hard"})
if hard_ks:
    scol, ccol = shades(hard_ks, REDS, R_RANGE), shades(hard_ks, BLUES, B_RANGE)
    recs = [get(f, "hard", k, None, "pnorm") for k in hard_ks for f in ("stream", "cr")]
    recs = [r[1] for r in recs if r]
    ylim = ylim_for(recs)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    for c, ax in enumerate(axes):
        for k in hard_ks:
            s, r = get("stream", "hard", k, None, "pnorm"), get("cr", "hard", k, None, "pnorm")
            if s:
                curve(ax, s[1], scol[k], "--", 1.8, ylim)
            if r:
                curve(ax, r[1], ccol[k], "-", 2.0, ylim)
        if c == 0:
            finish(ax, ylim, "validation CE (nats)")
            ax.set_title("whole run", fontsize=12)
        else:
            tail = [p[1] for rr in recs for p in rr["val"] if p[0] >= 8000]
            lo, hi = (min(tail) - 0.01, max(tail) + 0.01) if tail else ylim
            ax.set_ylim(lo, hi)
            ax.set_xlim(8000, 20000)
            ax.grid(alpha=0.3)
            ax.set_xlabel("training step")
            ax.set_title("steps 8000-20000", fontsize=12)
    hd = [Line2D([0], [0], color=scol[k], ls="--", lw=1.8, label=f"stream, hard Top-${k}$")
          for k in hard_ks]
    hd += [Line2D([0], [0], color=ccol[k], lw=2.0, label=f"code residual, hard Top-${k}$")
           for k in hard_ks]
    axes[1].legend(handles=hd, fontsize=8.5, loc="upper right", framealpha=0.93, ncol=2)
    fig.suptitle("Hard Top-$K'$ with the post-norm: stream carried (dashed, red) "
                 "vs code carried (solid, blue)", fontsize=14, y=0.995)
    save(fig, "cr_vs_stream_hard.png")

# ---- D: RBLapSum kappa, stream-carried vs code-carried, one figure per K -- #
for K in KS:
    js = sorted({j for (fam, kind, k, j, var) in cells if kind == "kappa" and k == K})
    if not js:
        continue
    scol, ccol = shades(js, REDS, R_RANGE), shades(js, BLUES, B_RANGE)
    recs = [get(f, "kappa", K, j, v) for j in js for v in ("t2", "pnorm")
            for f in ("stream", "cr", "cr_fo")]
    ylim = ylim_for([r[1] if r else None for r in recs])
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)
    for c, var in enumerate(("t2", "pnorm")):
        ax = axes[c]
        for j in js:
            s, r = get("stream", "kappa", K, j, var), get("cr", "kappa", K, j, var)
            fo = get("cr_fo", "kappa", K, j, var)
            if s:
                curve(ax, s[1], scol[j], "--", 1.8, ylim)
            if r:
                curve(ax, r[1], ccol[j], "-", 2.0, ylim)
            if fo:
                curve(ax, fo[1], ccol[j], ":", 2.4, ylim)
        finish(ax, ylim, "validation CE (nats)" if c == 0 else None)
        ax.set_title("$T=2$" if var == "t2" else "post-norm", fontsize=12)
    hd = [Line2D([0], [0], color=scol[j], ls="--", lw=1.8, label=f"stream, $J={j}$") for j in js]
    hd += [Line2D([0], [0], color=ccol[j], lw=2.0, label=f"code residual, $J={j}$") for j in js]
    fo_js = [j for j in js if any(get("cr_fo", "kappa", K, j, v) for v in ("t2", "pnorm"))]
    hd += [Line2D([0], [0], color=ccol[j], ls=":", lw=2.4,
                  label=f"code residual, first-order scope, $J={j}$") for j in fo_js]
    axes[1].legend(handles=hd, fontsize=9, loc="lower left", framealpha=0.93, ncol=2)
    fig.suptitle(f"RBLapSum kappa, $K={K}$: stream carried (dashed, red) vs "
                 f"code carried (solid, blue)", fontsize=14, y=0.995)
    save(fig, f"cr_vs_stream_rbk_k{K}.png")


# ---- tables --------------------------------------------------------------- #
def final(entry):
    if entry is None:
        return None
    n, r = entry
    if "diverged" in r:
        return "div@%d" % r["diverged"]["step"]
    return r["val"][-1][1]


def cell(entry):
    v = final(entry)
    return "--" if v is None else (v if isinstance(v, str) else "%.4f" % v)


def delta(a, b):
    fa, fb = final(a), final(b)
    if not isinstance(fa, float) or not isinstance(fb, float):
        return "--"
    return "%+.4f" % (fa - fb)


print("\n=== cell provenance ===")
for key in sorted(cells, key=lambda t: (t[0], t[1], t[2], t[3] or 0, t[4])):
    fam, kind, k, j, var = key
    print("   %-6s %-5s K=%-4d J=%-4s %-6s <- %s" % (fam, kind, k, j if j else "-", var, cells[key][0]))

for fam in ("cr", "stream"):
    for var, lab in (("t2", "T=2"), ("pnorm", "post-norm")):
        print("\n=== %s family, %s: rows K, columns K'=K+J (final val CE) ===" % (fam, lab))
        print("%-8s %s" % ("K", "".join("%12s" % ("K'=%d" % kp) for kp in KS[1:])))
        for K in KS[:-1]:
            print("%-8d %s" % (K, "".join("%12s" % cell(get(fam, "kappa", K, kp - K, var))
                                          for kp in KS[1:])))
        print("%-8s %s" % ("hard", "".join("%12s" % cell(get(fam, "hard", kp, None, "pnorm"))
                                           for kp in KS[1:])))
    print("   hard Top-32 (%s): %s" % (fam, cell(get(fam, "hard", 32, None, "pnorm"))))

for var, lab in (("t2", "T=2"), ("pnorm", "post-norm")):
    print("\n=== code residual minus stream, %s ===" % lab)
    print("%-8s %s" % ("K", "".join("%12s" % ("K'=%d" % kp) for kp in KS[1:])))
    for K in KS[:-1]:
        print("%-8d %s" % (K, "".join("%12s" % delta(get("cr", "kappa", K, kp - K, var),
                                                    get("stream", "kappa", K, kp - K, var))
                                      for kp in KS[1:])))
    print("%-8s %s" % ("hard", "".join("%12s" % delta(get("cr", "hard", kp, None, "pnorm"),
                                                      get("stream", "hard", kp, None, "pnorm"))
                                       for kp in KS[1:])))
print("   hard Top-32: %s" % delta(get("cr", "hard", 32, None, "pnorm"),
                                   get("stream", "hard", 32, None, "pnorm")))

fo_cells = sorted((k, j, v) for (fam, kind, k, j, v) in cells if fam == "cr_fo")
if fo_cells:
    print("\n=== supplementary: code residual with the first-order scope ===")
    print("%-5s %-5s %-6s %10s %12s %10s %12s" % ("K", "J", "var", "first-ord", "pool (cr)",
                                                 "stream", "hard cr K'"))
    for k, j, v in fo_cells:
        print("%-5d %-5d %-6s %10s %12s %10s %12s" % (
            k, j, v, cell(get("cr_fo", "kappa", k, j, v)), cell(get("cr", "kappa", k, j, v)),
            cell(get("stream", "kappa", k, j, v)), cell(get("cr", "hard", k + j, None, "pnorm"))))


# ---- LaTeX table bodies for the note ---------------------------------------- #
def tex_val(entry, bold=False):
    v = final(entry)
    if v is None:
        return "---"
    if isinstance(v, str):
        return "\\textit{%s}" % v
    s = "%.4f" % v
    return "\\bst{%s}" % s if bold else s


def tex_delta(a, b):
    d = delta(a, b)
    return "---" if d == "--" else "$%s$" % d


print("\n%% ===== LaTeX: code-carried family, final val CE (bold = beats hard in its column) =====")
for var, lab in (("t2", "T=2"), ("pnorm", "post-norm")):
    print("%% %s" % lab)
    for K in KS[:-1]:
        row = []
        for kp in KS[1:]:
            if kp <= K:
                row.append("---")
                continue
            e, h = get("cr", "kappa", K, kp - K, var), get("cr", "hard", kp, None, "pnorm")
            fe, fh = final(e), final(h)
            bold = isinstance(fe, float) and isinstance(fh, float) and fe < fh
            row.append(tex_val(e, bold))
        print("$%d$ & %s \\\\" % (K, " & ".join(row)))
    print("hard Top-$\\Kp$ & %s \\\\" % " & ".join(tex_val(get("cr", "hard", kp, None, "pnorm"))
                                                   for kp in KS[1:]))
print("%% hard Top-32 (cr): %s   (stream): %s" % (tex_val(get("cr", "hard", 32, None, "pnorm")),
                                                   tex_val(get("stream", "hard", 32, None, "pnorm"))))
print("\n%% ===== LaTeX: code carried minus stream carried =====")
for var, lab in (("t2", "T=2"), ("pnorm", "post-norm")):
    print("%% %s" % lab)
    for K in KS[:-1]:
        row = ["---" if kp <= K else tex_delta(get("cr", "kappa", K, kp - K, var),
                                               get("stream", "kappa", K, kp - K, var))
               for kp in KS[1:]]
        print("$%d$ & %s \\\\" % (K, " & ".join(row)))
print("hard Top-$\\Kp$ & %s \\\\" % " & ".join(tex_delta(get("cr", "hard", kp, None, "pnorm"),
                                                      get("stream", "hard", kp, None, "pnorm"))
                                            for kp in KS[1:]))
print("%% hard Top-32: %s" % tex_delta(get("cr", "hard", 32, None, "pnorm"),
                                      get("stream", "hard", 32, None, "pnorm")))
if fo_cells:
    print("\n%% ===== LaTeX: supplementary first-order rows: K & J & variant & first-order "
          "& pool & stream & hard cr K' =====")
    for k, j, v in fo_cells:
        print("$%d$ & $%d$ & %s & %s & %s & %s & %s \\\\" % (
            k, j, "post-norm" if v == "pnorm" else "$T=2$",
            tex_val(get("cr_fo", "kappa", k, j, v)), tex_val(get("cr", "kappa", k, j, v)),
            tex_val(get("stream", "kappa", k, j, v)), tex_val(get("cr", "hard", k + j, None, "pnorm"))))
