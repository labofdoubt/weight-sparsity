"""Streamlit viewer of the per-coordinate DUST SNR (toy_mnist.snr_probe).

    streamlit run toy_mnist/snr_viewer.py -- --root runs_mnist/snr_sweep

Reads every ``<root>/*/snr.npz``; two dropdowns pick K and the population,
a slider the checkpoint; the page shows the distribution of SNR over the 784
input coordinates (histogram on a log axis, fixed across the slider) and the
same values laid out as the 28 x 28 pixel grid.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE = "#2a78d6"
RAMP = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("SNR_ROOT", "runs_mnist/snr_sweep"))
    args, _ = ap.parse_known_args(sys.argv[1:])
    return args


@st.cache_data(show_spinner=False)
def load_all(root: str):
    runs = {}
    for path in sorted(glob.glob(os.path.join(root, "*", "snr.npz"))):
        with np.load(path, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        runs[(int(d["k"]), int(d["population"]))] = d
    return runs


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8, length=0)


def main():
    args = parse_args()
    st.set_page_config(page_title="DUST SNR per coordinate", layout="wide")
    st.title("SNR of the population-averaged DUST estimate, per input coordinate")
    runs = load_all(args.root)
    if not runs:
        st.error(f"no <run>/snr.npz under {args.root}")
        return
    ks = sorted({k for k, _ in runs})
    pops = sorted({p for _, p in runs})  # the union over K: the list never changes with K
    col1, col2, col3, col4 = st.columns([2, 2, 3, 2])
    k = col1.selectbox("K (support width)", ks, index=0, key="k_choice")
    pop = col2.selectbox("population (draws per DUST step)", pops, index=0, key="pop_choice")
    which = col3.selectbox("gradient", ["on u = exp(g) * x (the perturbed vector)",
                                       "on g (what the optimizer applies)"], index=0,
                           key="grad_choice")
    xscale = col4.radio("histogram x-axis", ["log", "linear"], index=0, horizontal=True,
                        key="xscale_choice")
    key = "u" if which.startswith("on u") else "g"
    if (k, pop) not in runs:
        have = ", ".join(str(p) for kk, p in sorted(runs) if kk == k)
        st.warning(f"no run for K = {k} at population {pop}; K = {k} has populations {have}.")
        return
    d = runs[(k, pop)]
    steps = d["steps"]
    def label(n: int) -> str:
        if n == 0:
            return "step 0 (initial state)"
        if n % 2 == 1:
            return f"step {n} (the state DUST step {n + 1} acts on)"
        return f"step {n} (after {n} updates; a hard step follows)"
    labels = [label(int(n)) for n in steps]
    i = st.slider("checkpoint", 0, len(steps) - 1, 0, format="%d", key="ckpt_choice")
    i = min(i, len(steps) - 1)
    st.caption(f"checkpoint {i}: **{labels[i]}** — completed optimizer steps N = {int(steps[i])}; "
               f"an odd N is the state the DUST step N + 1 acts on.")
    snr_all = d[f"snr_{key}"]
    snr = snr_all[i]
    finite = np.isfinite(snr)
    vals = snr[finite]
    pos = snr_all[np.isfinite(snr_all) & (snr_all > 0)]
    lo, hi = (float(pos.min()), float(pos.max())) if pos.size else (1e-3, 1e1)
    lo, hi = max(lo, 1e-4), max(hi, lo * 10)

    c1, c2 = st.columns([3, 2])
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    if xscale == "log":
        bins = np.logspace(np.log10(lo), np.log10(hi), 60)
        centers = np.sqrt(bins[:-1] * bins[1:])
    else:
        bins = np.linspace(0.0, hi, 60)
        centers = 0.5 * (bins[:-1] + bins[1:])
    ax.hist(np.clip(vals, bins[0], hi), bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.6,
            label="measured")
    # pure-noise reference: a coordinate without signal has sqrt(P) |mu| / nu
    # distributed as |N(0, 1)| (median 0.674), independent of P (as in
    # analysis/dust_snr_viewer.py)
    ref = np.abs(np.random.default_rng(0).standard_normal(200_000))
    counts, _ = np.histogram(np.clip(ref, bins[0], hi), bins=bins)
    ax.plot(centers, counts * (vals.size / ref.size), color=MUTED, lw=1.3, ls="--",
            label="pure-noise reference |N(0,1)|")
    ax.axvline(1.0, color=MUTED, linewidth=1.0)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    if xscale == "log":
        ax.set_xscale("log")
    else:
        ax.set_xlim(0.0, hi)
    style(ax)
    ax.set_xlabel(f"SNR = sqrt(P) |mu| / nu  ({xscale} axis, fixed range over the slider; "
                  "the line marks 1)", color=INK2, fontsize=9)
    ax.set_ylabel("coordinates", color=INK2, fontsize=9)
    ax.set_title(f"K = {k}, P = {pop}, {labels[i]}: distribution over {vals.size} coordinates",
                 color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    c1.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 3.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    img = np.where(finite & (snr > 0), snr, np.nan).reshape(28, 28)
    im = ax.imshow(np.nan_to_num(img, nan=lo), cmap=RAMP, norm=LogNorm(vmin=lo, vmax=hi))
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(AXIS)
    ax.set_title("SNR per pixel (log colour scale)", color=INK, fontsize=10, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    fig.tight_layout()
    c2.pyplot(fig)
    plt.close(fig)

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("median SNR", f"{np.median(vals):.3f}" if vals.size else "nan")
    s2.metric("mean SNR", f"{vals.mean():.3f}" if vals.size else "nan")
    n_above = int((vals > 1).sum())
    s3.metric("coordinates with SNR > 1", f"{n_above} / {vals.size}",
              delta=f"{n_above - 0.3173 * vals.size:+.0f} vs pure noise",
              help="For a coordinate without signal, SNR > 1 with probability 0.317; "
                   "the delta is the excess over that expectation.")
    s4.metric("clean CE of the batch", f"{float(d['ce_clean'][i]):.4f}")
    s5.metric("perturbed - clean CE", f"{float(d['ce_perturbed'][i] - d['ce_clean'][i]):+.4f}")
    if (~finite).sum():
        st.caption(f"{int((~finite).sum())} coordinates have nu = 0 (SNR undefined) and are "
                   "left out: under the g weighting these are pixels that are 0 in every "
                   "sample of the batch.")

    fig, ax = plt.subplots(figsize=(11, 2.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    med = np.array([np.nanmedian(r) if np.isfinite(r).any() else np.nan for r in snr_all])
    q25 = np.array([np.nanpercentile(r, 25) if np.isfinite(r).any() else np.nan for r in snr_all])
    q75 = np.array([np.nanpercentile(r, 75) if np.isfinite(r).any() else np.nan for r in snr_all])
    xs = np.maximum(steps, 1)
    ax.fill_between(xs, q25, q75, color=BLUE, alpha=0.12, linewidth=0)
    ax.plot(xs, med, color=BLUE, lw=1.5, marker="o", ms=3, markeredgecolor=SURFACE)
    ax.axvline(max(int(steps[i]), 1), color=MUTED, linewidth=1.0)
    ax.axhline(1.0, color=GRID, linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    style(ax)
    ax.set_xlabel("completed optimizer steps (log; the initial state is drawn at 1)",
                  color=INK2, fontsize=9)
    ax.set_ylabel("median SNR, IQR band", color=INK2, fontsize=9)
    ax.set_title("over the checkpoints", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(f"batch of {int(d['batch_size'])} training images (seed {int(d['batch_seed'])}), "
               f"noise seed {int(d['noise_seed'])}, sigma {float(d['sigma']):g} "
               f"({str(d['sigma_mode'])}), hidden {int(d['hidden'])}; source {args.root}")


if __name__ == "__main__":
    main()
