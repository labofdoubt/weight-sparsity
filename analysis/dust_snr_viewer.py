"""Streamlit viewer of the per-coordinate DUST SNR of the transformer bottlenecks
(analysis/dust_snr_probe.py).

    streamlit run analysis/dust_snr_viewer.py -- --root /workspace/analysis/dust_snr

Reads every ``<root>/<run>/snr_step<N>.npz``.  Two dropdowns pick the Top-K
``K`` and the training population; sliders pick the checkpoint, the
bottleneck, the token position and the sample sequence; the page shows the
distribution over the feature coordinates of ``SNR_K = sqrt(K) |mu| / nu``
(histogram on a log axis, range fixed over the sliders), the token's
summary numbers, and the median over tokens per checkpoint.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE = "#2a78d6"
ORANGE = "#e07b1a"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("DUST_SNR_ROOT", "/workspace/analysis/dust_snr"))
    ap.add_argument("--single-root", default=os.environ.get(
        "DUST_SNR_SINGLE_ROOT", "/workspace/analysis/dust_snr_single"))
    args, _ = ap.parse_known_args(sys.argv[1:])
    return args


def step_of(path: str) -> int:
    return int(re.search(r"snr_step(\d+)", path).group(1))


@st.cache_data(show_spinner=False, ttl=120)
def catalogue(root: str):
    """``{(K, population): {step: path}}`` with each run's meta (re-read every 2 min)."""
    runs, metas = {}, {}
    for path in sorted(glob.glob(os.path.join(root, "*", "snr_step*.npz"))):
        try:
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
        except Exception:
            continue  # being written
        key = (int(meta["topk_k"]), int(meta["population"]))
        runs.setdefault(key, {})[step_of(path)] = path
        metas[key] = meta
    return runs, metas


@st.cache_data(show_spinner=False, max_entries=24)
def load_snr(path: str):
    with np.load(path, allow_pickle=False) as z:
        return z["snr"].astype(np.float32), z["snr_vec"], z["valid"], json.loads(str(z["meta"]))


@st.cache_data(show_spinner=False, max_entries=64)
def summary_over_tokens(path: str, layer: int, sample: int):
    """Per checkpoint: median and quartiles over (valid tokens x pool coordinates)
    of one bottleneck and sample, and the median vector SNR over tokens."""
    snr, snr_vec, valid, meta = load_snr(path)
    v = snr[sample, layer][valid[sample]]
    v = v[v > 0]
    return (float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75)),
            float(np.median(snr_vec[sample, layer][valid[sample]])))


@st.cache_data(show_spinner=False, ttl=120)
def catalogue_single(root: str):
    """Single-block probes: ``{(population, step, block, sigma, draws): path}``."""
    files, metas = {}, {}
    for path in sorted(glob.glob(os.path.join(root, "*", "snr_step*_*.npz"))):
        try:
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
        except Exception:
            continue
        key = (int(meta["population"]), int(meta["step"]), int(meta["layer_indices"][0]),
               float(meta["sigma"]), int(meta["draws"]))
        files[key] = path
        metas[key] = meta
    return files, metas


@st.cache_data(show_spinner=False, max_entries=64)
def single_summary(path: str, sample: int):
    snr, snr_vec, valid, meta = load_snr(path)
    v = snr[sample, 0][valid[sample]]
    v = v[v > 0]
    return float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8, length=0)


def label(n: int) -> str:
    if n == 0:
        return "step 0 (initial state)"
    return f"step {n} (after {n // 2} DUST steps and {n - n // 2} hard steps)"


def histogram_and_metrics(container, row, range_values, title, logx, nbins):
    """The per-coordinate histogram with the pure-noise reference, and the
    summary metrics; ``range_values`` fixes the axis range."""
    vals = row[row > 0]
    pos = range_values[range_values > 0]
    lo, hi = (float(np.percentile(pos, 0.1)), float(pos.max())) if pos.size else (1e-3, 1e1)
    lo, hi = max(lo, 1e-4), max(hi, lo * 10)
    left, _ = container.columns([3, 2])
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    if logx:
        bins = np.logspace(np.log10(lo), np.log10(hi), nbins + 1)
        centers = np.sqrt(bins[:-1] * bins[1:])
    else:
        bins = np.linspace(0.0, hi, nbins + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])
    ax.hist(np.clip(vals, bins[0], hi), bins=bins, color=BLUE, edgecolor=SURFACE,
            linewidth=0.6 if nbins <= 100 else 0.0, label="measured")
    # pure-noise reference: a coordinate without signal has sqrt(K) |mu| / nu
    # distributed as |N(0, 1)| (median 0.674), independent of K
    ref = np.abs(np.random.default_rng(0).standard_normal(200_000))
    counts, _ = np.histogram(np.clip(ref, bins[0], hi), bins=bins)
    ax.plot(centers, counts * (vals.size / ref.size), color=ORANGE, lw=1.6, ls="--",
            label="pure-noise reference |N(0,1)|")
    ax.axvline(1.0, color=MUTED, linewidth=1.0)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2)
    if logx:
        ax.set_xscale("log")
    style(ax)
    ax.set_xlabel("SNR_K = sqrt(K) |mu| / nu per coordinate  "
                  f"({'log' if logx else 'linear'} axis; range fixed over the sliders; "
                  "the line marks 1)", color=INK2, fontsize=9)
    ax.set_ylabel("coordinates", color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    left.pyplot(fig, use_container_width=True)
    plt.close(fig)
    m1, m2, m3 = container.columns(3)
    m1.metric("median SNR_K", f"{np.median(vals):.3f}" if vals.size else "nan")
    m2.metric("mean SNR_K", f"{vals.mean():.3f}" if vals.size else "nan")
    n_above = int((vals > 1).sum())
    m3.metric("coordinates with SNR_K > 1", f"{n_above} / {vals.size}",
              delta=f"{n_above - 0.3173 * vals.size:+.0f} vs pure noise",
              help="For a coordinate without signal, SNR_K > 1 with probability 0.317; "
                   "the delta is the excess over that expectation.")
    return vals


def tab_training(args):
    runs, metas = catalogue(args.root)
    if not runs:
        st.error(f"no <run>/snr_step*.npz under {args.root} yet")
        return
    ks = sorted({k for k, _ in runs})
    pops = sorted({p for _, p in runs})  # the union over K
    c1, c2 = st.columns(2)
    k = c1.selectbox("K (Top-K support width)", ks, index=0, key="k_choice")
    pop = c2.selectbox("population (draws per DUST step and micro-batch)", pops, index=0,
                       key="pop_choice")
    if (k, pop) not in runs:
        have = ", ".join(str(p) for kk, p in sorted(runs) if kk == k)
        st.warning(f"no run for K = {k} at population {pop}; K = {k} has populations {have}.")
        return
    files = runs[(k, pop)]
    steps = sorted(files)
    meta = metas[(k, pop)]
    L, T, S = int(meta["layers"]), int(meta["seq_len"]), int(meta["sequences"])
    labels = [label(n) for n in steps]
    s1, s2, s3, s4, s5, s6 = st.columns([3, 1, 2, 1, 1, 1])
    i = s1.slider("checkpoint", 0, len(steps) - 1, 0, key="ckpt_choice")
    i = min(i, len(steps) - 1)
    layer = s2.slider("bottleneck", 0, L - 1, 0, key="layer_choice")
    tok = s3.slider("token position", 0, T - 1, T // 2, key="tok_choice")
    sample = s4.slider("sample", 0, S - 1, 0, key="sample_choice")
    logx = s5.toggle("log x-axis", value=True, key="logx_choice")
    nbins = s6.slider("bins", 20, 400, 60, step=10, key="bins_choice")
    step = steps[i]
    st.caption(f"checkpoint {i}: **{labels[i]}** — K = {k}, population {pop} "
               f"({meta['draws']} draws per bottleneck in training and in this probe), "
               f"sigma {meta['sigma']:g} ({meta['sigma_mode']}), gamma {meta['gamma']:g}, "
               f"target {meta['target']}, {'MD' if meta['decouple'] else 'no MD'}; "
               f"pool {meta['pool']} of {meta['n_features']} features.")
    snr, snr_vec, valid, _ = load_snr(files[step])
    title = (f"K = {k}, population {pop}, step {step}, bottleneck {layer}, token {tok}, "
             f"sample {sample}: {int((snr[sample, layer, tok] > 0).sum())} coordinates")
    histogram_and_metrics(st, snr[sample, layer, tok], snr, title, logx, nbins)
    m4, m5 = st.columns(2)
    m4.metric("vector SNR_K of this token", f"{float(snr_vec[sample, layer, tok]):.3f}")
    m5.metric("token valid", "yes" if bool(valid[sample, tok]) else "no (ignored target)")

    left2, _ = st.columns([3, 2])  # the same 60% column as the histogram
    fig, ax = plt.subplots(figsize=(7, 2.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    rows = [summary_over_tokens(files[n], layer, sample) for n in steps]
    med = np.array([r[0] for r in rows]); q25 = np.array([r[1] for r in rows])
    q75 = np.array([r[2] for r in rows]); medv = np.array([r[3] for r in rows])
    xs = np.maximum(np.array(steps), 1)
    ax.fill_between(xs, q25, q75, color=BLUE, alpha=0.12, linewidth=0)
    ax.plot(xs, med, color=BLUE, lw=1.5, marker="o", ms=3, markeredgecolor=SURFACE,
            label="median over valid tokens x coordinates (IQR band)")
    ax.plot(xs, medv, color=INK2, lw=1.2, ls="--", marker="s", ms=2.5,
            label="median over tokens of the vector SNR_K")
    ax.axvline(max(step, 1), color=MUTED, linewidth=1.0)
    ax.axhline(1.0, color=GRID, linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    style(ax)
    ax.set_xlabel("completed optimizer steps (log; the initial state is drawn at 1)",
                  color=INK2, fontsize=9)
    ax.set_ylabel("SNR_K", color=INK2, fontsize=9)
    ax.set_title(f"bottleneck {layer}, sample {sample}: over the checkpoints", color=INK,
                 fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    fig.tight_layout()
    left2.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(f"{S} validation windows of {T} tokens (offset {meta['val_offset']}), noise seed "
               f"{meta['seed']}; run {meta['run']}; source {args.root}")


def tab_single(args):
    files, metas = catalogue_single(args.single_root)
    if not files:
        st.info(f"no single-block probes under {args.single_root} yet")
        return
    pops = sorted({k[0] for k in files})
    c0, c1, c2, c3, c4 = st.columns(5)
    pop = c0.selectbox("model (training population)", pops, index=0, key="sg_pop")
    steps = sorted({k[1] for k in files if k[0] == pop})
    step = c1.selectbox("checkpoint step", steps, index=len(steps) - 1, key="sg_step")
    blocks = sorted({k[2] for k in files if k[0] == pop and k[1] == step})
    block = c2.selectbox("perturbed block", blocks, index=0, key="sg_block")
    sigmas = sorted({k[3] for k in files if k[0] == pop and k[1] == step and k[2] == block})
    sigma = c3.selectbox("sigma", sigmas, index=0, key="sg_sigma")
    drawss = sorted({k[4] for k in files if k[:4] == (pop, step, block, sigma)})
    draws = c4.selectbox("draws K", drawss, index=0, key="sg_draws")
    key = (pop, step, block, sigma, draws)
    if key not in files:
        st.warning("no probe for this combination yet")
        return
    meta = metas[key]
    T, S = int(meta["seq_len"]), int(meta["sequences"])
    s3, s4, s5, s6 = st.columns([2, 1, 1, 1])
    tok = s3.slider("token position", 0, T - 1, T // 2, key="sg_tok")
    sample = s4.slider("sample", 0, S - 1, 0, key="sg_sample")
    logx = s5.toggle("log x-axis", value=True, key="sg_logx")
    nbins = s6.slider("bins", 20, 400, 60, step=10, key="sg_bins")
    st.caption(f"model {meta['run']} at step {step} (trained at population {pop}, sigma "
               f"{meta['train_sigma']:g}); this probe perturbs block {block} only, with "
               f"{draws} draws at sigma {sigma:g}; pool {meta['pool']} of {meta['n_features']} "
               f"features.")
    snr, snr_vec, valid, _ = load_snr(files[key])
    title = (f"block {block}, sigma {sigma:g}, K = {draws} draws, step {step}, token {tok}, "
             f"sample {sample}: {int((snr[sample, 0, tok] > 0).sum())} coordinates")
    histogram_and_metrics(st, snr[sample, 0, tok], snr, title, logx, nbins)
    m4, m5 = st.columns(2)
    m4.metric("vector SNR_K of this token", f"{float(snr_vec[sample, 0, tok]):.3f}")
    m5.metric("token valid", "yes" if bool(valid[sample, tok]) else "no (ignored target)")

    # the median over valid tokens x coordinates against the draws, per (block, sigma)
    left2, _ = st.columns([3, 2])
    fig, ax = plt.subplots(figsize=(7, 2.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    combos = sorted({(k[2], k[3]) for k in files if k[0] == pop and k[1] == step})
    palette = [BLUE, "#0d366b", ORANGE, "#8a4a0f", "#4a9c5c", "#7a3f9d"]
    for ci, (b, sg) in enumerate(combos):
        ks_ = sorted(k[4] for k in files if k[:4] == (pop, step, b, sg))
        rows = [single_summary(files[(pop, step, b, sg, kk)], sample) for kk in ks_]
        med = np.array([r[0] for r in rows]); q25 = np.array([r[1] for r in rows])
        q75 = np.array([r[2] for r in rows])
        col = palette[ci % len(palette)]
        ax.fill_between(ks_, q25, q75, color=col, alpha=0.10, linewidth=0)
        ax.plot(ks_, med, color=col, lw=1.5, marker="o", ms=3, markeredgecolor=SURFACE,
                label=f"block {b}, sigma {sg:g}")
    ax.axhline(1.0, color=GRID, linewidth=1.0)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    style(ax)
    ax.set_xlabel("draws K (log)", color=INK2, fontsize=9)
    ax.set_ylabel("median SNR_K, IQR band", color=INK2, fontsize=9)
    ax.set_title(f"sample {sample}: median over valid tokens x coordinates against the draws",
                 color=INK, fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3)
    fig.tight_layout()
    left2.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(f"{S} validation windows of {T} tokens (offset {meta['val_offset']}), noise seed "
               f"{meta['seed']}; source {args.single_root}")


def main():
    args = parse_args()
    st.set_page_config(page_title="DUST SNR per coordinate (bottlenecks)", layout="wide")
    st.title("SNR of the population-averaged DUST estimate, per bottleneck coordinate")
    t1, t2 = st.tabs(["training populations", "single-block probes"])
    with t1:
        tab_training(args)
    with t2:
        tab_single(args)


if __name__ == "__main__":
    main()
