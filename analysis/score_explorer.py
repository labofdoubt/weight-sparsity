"""Interactive explorer for bottleneck ranking scores across checkpoints.

    streamlit run interpretability/score_explorer.py

Serves two kinds of dataset, chosen in the sidebar, both memory-mapped so only
the drawn cell is ever read:

* **Checkpoint ladder** -- what ``extract_bottleneck_scores.py`` cached from
  saved weights: one array per run, shaped
  ``(ckpt, bottleneck, sequence, position, n_features)``, steps 2k..20k.
* **Early training probe** -- what ``probe_early_training.py`` recorded while
  re-running training with the same seed, steps 0..1000 every 10 with no
  checkpoints kept.  Same shape on a probe-step axis, plus two gradient arrays,
  which add the gradient panels at the bottom of the page.

Four linked views of the same cell, in the order they appear:

1. **rank x |score|** -- the readable one.  1536 points never collide, the band
   edges are exact at rank ``k`` and ``k+j``, and the selection margin shows up
   as a kink instead of a sub-pixel gap.
2. **the half-line** -- |score| on one axis, as originally specified, kept as a
   marginal strip with deterministic jitter so overlapping circles stay countable.
3. **the support map** -- one circle per neuron, ordered by *feature index*, so a
   neuron keeps its seat between frames and the support itself can be watched
   moving as a slider is dragged.  Ordered by rank instead it would be the same
   picture every frame (k of one colour, then j of another) and say nothing.
4. **the raw score scale** across steps, the effect normalization removes.

Probe datasets add two more, against |score| on the same axis furniture as the
half-line: **dL/d~z** at the gate's output (dense -- non-zero even where TopK
did not select) and **dL/dz** at its input, after the LapSum surrogate (exactly
zero outside Top(K+J)).  Each reports the pool's sign agreement, which is the
number that answers whether the surrogate drives the pool one way.

Both score-axis views carry the LapSum **barrier** ``b`` and the ``b ± t`` window
(see :func:`barrier_at`) -- the candidates inside it are the ones whose surrogate
gradient is not saturated away.  It is drawn on whichever axis carries the score,
so it runs horizontally on view 1 and vertically on view 2.

Why the normalization control matters more than it looks: the raw score scale
falls roughly 5x over training and varies >10x across layers, and once you
divide by the selection threshold ``r[k-1]`` the profile is nearly invariant
(cross-checkpoint CV 0.462 -> 0.012 on the run measured).  So on raw values the
checkpoint slider mostly shows a global shrink; normalized, it shows the shape
change that is left.  Both are worth looking at, hence the toggle.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st

SCORES_DIR = os.environ.get("SCORES_DIR", "/workspace/analysis/scores")
PROBE_DIR = os.environ.get("PROBE_DIR", "/workspace/analysis/probe")
WNORM_DIR = os.environ.get("WNORM_DIR", "/workspace/analysis/wnorm")

# A dataset is addressed as "<kind>/<name>" so every cached helper below can go
# on taking a single string, and st.cache_data keys stay correct across kinds.
#   ckpt  -- one array per run, the 2k..20k checkpoint ladder
#   probe -- three arrays per run (score + the two gradients), steps 0..1000
GRAD_ARRAYS = ("g_ztilde", "g_z")
GRAD_LABEL = {"g_ztilde": "dL/d\u007ez  (gate output, before the decoder)",
              "g_z": "dL/dz  (gate input, through the LapSum surrogate)"}
GRAD_AXIS = {"g_ztilde": "dL/d\u007ez", "g_z": "dL/dz"}

# Categorical slots 1-3 of the reference palette, validated on the all-pairs
# pairlist in both modes (worst CVD dE 9.2 light / 9.4 dark).  Aqua carries the
# light-mode contrast warning (2.74:1), so it is given to `rest` -- the least
# important band -- and the relief rule is met by the band labels and the table
# view at the bottom of the page.
BAND_COLOR = {"topk": "#2a78d6", "cand": "#eb6834", "rest": "#1baf7a"}
BAND_LABEL = {"topk": "TopK (survives forward)",
              "cand": "next J (surrogate grad)",
              "rest": "rest (zero grad)"}
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e8e7e3"
SURFACE = "#fcfcfb"

# Ring colours encode which band a neuron occupied in the *comparison* frame,
# independent of the fill, which is its band now.  Black for "was TopK";
# magenta (palette slot 5) for "was a J candidate" -- picked over the palette's
# yellow because yellow beside the orange `cand` fill is the one documented
# failing pair, and over violet/gray because neither reads as clearly non-black
# at a 2px stroke.
RING_WAS_TOPK = INK
RING_WAS_CAND = "#d55181"

# The LapSum barrier is an annotation, not a series, so it gets the one unused
# palette hue (violet, slot 7) as a soft wash rather than a fourth mark colour:
# a large low-opacity area does not compete with the point marks the way another
# saturated hue would.
BARRIER = "#4a3aa7"
BARRIER_FILL = "rgba(74, 58, 167, 0.10)"

# Line charts use the *adjacent* pairlist, which the reference palette's full
# eight-slot order passes in both modes -- unlike the all-pairs scatter case,
# which caps at three. So overlaying up to 8 runs as lines is in-spec.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
       "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# Blocks are ordered, so a one-hue ordinal ramp would be the natural choice --
# but it does not survive validation here: the blue ramp's usable range (step
# 250, the lightest that clears the surface, to 700) gives an adjacent dL of
# 0.047 across 8 steps, which the ordinal gate rejects as indistinguishable.
# Eight ordered steps of a single hue are simply not achievable, so blocks use
# the categorical order instead and the legend carries the ordering.

st.set_page_config(page_title="Bottleneck score explorer", layout="wide")


def split_key(key: str):
    kind, name = key.split("/", 1)
    return kind, name


@st.cache_resource(show_spinner=False)
def load_run(key: str):
    """``(score_array, meta, arrays)`` for one dataset, memory-mapped.

    Returns the score array first because every existing panel wants it; the
    dict carries the probe datasets' extra gradient arrays and is empty for a
    checkpoint ladder.
    """
    kind, name = split_key(key)
    if kind == "probe":
        arrays = {"score": np.load(os.path.join(PROBE_DIR, f"{name}.score.npy"),
                                   mmap_mode="r")}
        for g in GRAD_ARRAYS:
            fp = os.path.join(PROBE_DIR, f"{name}.{g}.npy")
            if os.path.exists(fp):
                arrays[g] = np.load(fp, mmap_mode="r")
        meta = json.load(open(os.path.join(PROBE_DIR, f"{name}.json")))
    else:
        arrays = {"score": np.load(os.path.join(SCORES_DIR, f"{name}.npy"),
                                   mmap_mode="r")}
        meta = json.load(open(os.path.join(SCORES_DIR, f"{name}.json")))
    return arrays["score"], meta, arrays


@st.cache_data(show_spinner=False, ttl=30)
def available_runs(kind: str):
    """Run names for one kind, ordered by (k, effective j) rather than by name.

    Alphabetical order puts ``..._j128`` first, which is the worst default for
    the checkpoint ladder: that is the run that diverges, so the app would open
    on a collapsed model.
    """
    if kind == "probe":
        # The .json is written only when a probe run finishes, so it -- not the
        # .npy, which is allocated up front -- is what marks a dataset complete.
        # Listing by array would show a still-training run and then fail to load
        # its metadata.
        names = [os.path.basename(p)[: -len(".score.npy")]
                 for p in glob.glob(os.path.join(PROBE_DIR, "*.score.npy"))
                 if os.path.exists(p[: -len(".score.npy")] + ".json")]
        meta_dir = PROBE_DIR
    else:
        names = [os.path.splitext(os.path.basename(p))[0]
                 for p in glob.glob(os.path.join(SCORES_DIR, "*.npy"))]
        meta_dir = SCORES_DIR

    def sort_key(n):
        try:
            m = json.load(open(os.path.join(meta_dir, f"{n}.json")))
            jj = 0 if m.get("surrogate_mode") == "hard" else int(m["j"])
            return (0, int(m["k"]), jj, n)
        except Exception:
            return (1, 0, 0, n)

    return sorted(names, key=sort_key)


@st.cache_data(show_spinner=False)
def jitter(n: int) -> np.ndarray:
    """Fixed vertical offsets for the half-line, keyed by feature index.

    Deterministic on purpose: jitter redrawn per frame would make every circle
    jump whenever a slider moved, which is exactly the motion the user is trying
    to read as a change in the model.
    """
    return np.random.default_rng(0).uniform(-1.0, 1.0, size=n)


@st.cache_data(show_spinner=False)
def bands_at(run: str, ci: int, li: int, bi: int, ti: int, k: int, j: int):
    """``(topk, cand)`` feature-index sets for one cell.

    Both are needed because the rings distinguish "was TopK" from "was a J
    candidate", so a promotion out of the candidate band is visible as such
    rather than lumped in with a neuron arriving from nowhere.
    """
    arr, _, _ = load_run(run)
    r = np.abs(np.asarray(arr[ci, li, bi, ti], dtype=np.float64))
    order = np.argpartition(-r, k + j - 1)[: k + j]
    order = order[np.argsort(-r[order])]
    return frozenset(order[:k].tolist()), frozenset(order[k:].tolist())


@st.cache_data(show_spinner=False)
def axis_range(run: str, li: int, bi: int, ti: int, norm: str, k: int) -> tuple:
    """Min/max of |score| over *all* checkpoints for this cell.

    Needed so that dragging the checkpoint slider does not silently rescale the
    axis underneath the thing being compared.
    """
    arr, _, _ = load_run(run)
    block = np.abs(np.asarray(arr[:, li, bi, ti], dtype=np.float64))  # (C, N)
    if norm != "raw":
        srt = -np.sort(-block, axis=-1)
        denom = srt[:, k - 1] if norm == "thresh" else srt[:, 0]
        block = block / denom[:, None]
    pos = block[block > 0]
    return float(pos.min()), float(block.max())


@st.cache_data(show_spinner=False)
def run_range(run: str, norm: str, k: int) -> tuple:
    """Min/max of |score| over the whole dataset, not one cell.

    The cut sliders are bounded by this rather than by ``axis_range`` so their
    domain does not move when you change block, sequence, position or step.  A
    per-cell domain meant streamlit clamped the remembered value into the new
    range on every switch -- and because "no cut" is the domain's own minimum, a
    block whose smallest score fell in a different decade could silently turn a
    remembered "no cut" into a real one.

    Strided rather than exhaustive: the arrays run to 1.3 GB, and the axis bound
    does not need every element.
    """
    arr, _, _ = load_run(run)
    C, L, B, T, N = arr.shape
    cs, ts = max(1, C // 12), max(1, T // 16)
    flat = np.abs(np.asarray(arr[::cs, :, :, ::ts, :], dtype=np.float32)).reshape(-1, N)
    if norm != "raw":
        srt = -np.sort(-flat, axis=-1)
        den = srt[:, k - 1] if norm == "thresh" else srt[:, 0]
        flat = flat / np.maximum(den, 1e-30)[:, None]
    pos = flat[flat > 0]
    return float(pos.min()), float(flat.max())


@st.cache_data(show_spinner=False)
def barrier_at(run: str, ci: int, li: int, bi: int, ti: int, k: int, j: int):
    """``(b, t)`` -- the LapSum barrier and temperature for one cell, or None.

    Reconstructed rather than read back: the gate's ``scheduled_temperature``
    buffer is ``persistent=False``, so it is not in the checkpoint.  Follows
    ``AdaptiveLapSumTopKGate.prescribed_temperature`` / ``solve`` exactly --

        t = schedule(step) * std(top-(k+j) scores)     [scale_mode="relative"]
        b  solves  sum_i F((r_i - b)/t) = k            over the k+j pool

    and calls the project's own ``lapsum_barrier_sorted`` rather than
    reimplementing the closed form.  Note the budget is ``k``, not ``k+j``.

    Returns None where no barrier exists: a hard gate never solves one, and the
    adaptive modes derive ``t`` from a Newton solve on the score geometry rather
    than from the schedule, which this does not attempt to reproduce.
    """
    _, meta, _ = load_run(run)
    mode = str(meta.get("surrogate_mode", ""))
    tc = meta.get("temperature")
    if j == 0 or tc is None or mode not in ("lapsum_scheduled", "lapsum_fixed"):
        return None
    import torch  # deferred: keeps app start-up off the torch import path
    from wsparse.bottleneck.lapsum import lapsum_barrier_sorted
    from wsparse.schedules import build_schedule

    arr, _, _ = load_run(run)
    r = np.abs(np.asarray(arr[ci, li, bi, ti], dtype=np.float32))
    cand = torch.from_numpy(np.sort(r)[::-1][: k + j].copy()).unsqueeze(0)
    if mode == "lapsum_fixed":
        t_sched = float(tc["fixed"])
    else:
        t_sched = float(build_schedule(
            kind=tc["schedule"], start=tc["start"], end=tc["end"],
            warmup_steps=tc["warmup_steps"], anneal_steps=tc["anneal_steps"],
            power=tc["power"], max_steps=tc["max_steps"],
        )(meta["steps"][ci]))
    # torch's std is unbiased (ddof=1); numpy's default is not, so go through
    # torch to match the gate bit-for-bit.
    scale = cand.std(-1) if tc["scale_mode"] == "relative" else torch.ones(1)
    t = t_sched * torch.where(scale > 0, scale, torch.ones_like(scale))
    b = lapsum_barrier_sorted(cand, k, t)
    return float(b[0]), float(t[0])


def min_cut_slider(label: str, key: str, lo: float, hi: float, log: bool) -> float:
    """Sidebar slider for a minimum-score cut, in the units on screen.

    Log-scaled when the axis is: the score axis spans several decades, so a
    linear slider would spend its whole travel inside the top one.  The top
    stops at the data max rather than ceil() of it, since a cut above the largest
    score would empty the panel and invert the axis range.

    The chosen value is mirrored into ``st.session_state["_cuts"]`` as an
    absolute score (``None`` for "no cut") and the widget's own key carries its
    domain.  Two things fall out of that:

    * changing run moves the domain, which makes a *new* widget, which takes its
      default from the remembered value -- so the cut carries across runs instead
      of snapping back, and is clamped rather than raising if the new run cannot
      represent it;
    * "no cut" is remembered as ``None`` rather than as the old domain's minimum,
      so it stays "no cut" on a run whose scores start lower.  Storing the number
      instead would silently turn it into a real cut.
    """
    store = st.session_state.setdefault("_cuts", {})
    prev = store.get(key)                       # None => no cut
    help_text = ("Hide neurons whose score falls below this, and start the axis "
                 "there. The domain spans the whole dataset, so the value "
                 "survives changing block, sequence, position, step and run; "
                 "leftmost means no cut.")
    if log:
        e_lo = float(np.floor(np.log10(lo)))
        e_hi = max(float(np.log10(hi)), e_lo + 1.0)   # st.slider needs min < max
        default = e_lo if prev is None else float(np.clip(np.log10(prev), e_lo, e_hi))
        e = st.slider(f"{label} (10^)", e_lo, e_hi, default, step=0.05,
                      key=f"{key}_log_{e_lo:.2f}_{e_hi:.2f}", help=help_text)
        # Leftmost is "no cut", returned as 0 so the panel keeps its per-cell
        # automatic range instead of being padded down to the dataset minimum.
        val = 0.0 if e <= e_lo else float(10.0 ** e)
    else:
        hi_f = max(float(hi), 1e-9)
        default = 0.0 if prev is None else float(np.clip(prev, 0.0, hi_f))
        val = st.slider(label, 0.0, hi_f, default, step=hi_f / 200.0,
                        key=f"{key}_lin_{hi_f:.4g}", help=help_text)
    store[key] = None if val <= 0 else val
    return val


def cut_range(base, cut: float, log: bool, data_max: float):
    """``base`` range with its low edge moved to ``cut`` (log10 units if log)."""
    if cut <= 0:
        return None if base is None else list(base)
    edge = float(np.log10(cut)) if log else float(cut)
    if base is None:
        top = (np.log10(data_max) + 0.04) if log else data_max * 1.04
        return [edge, float(top)]
    out = list(base)
    out[0] = edge
    return out


def ring(idx: np.ndarray, was_topk, was_cand, width: float):
    """Per-point ring colour/width: black = was TopK, magenta = was a J candidate."""
    w = np.where(was_topk[idx] | was_cand[idx], width, 0.0)
    c = np.where(was_topk[idx], RING_WAS_TOPK,
                 np.where(was_cand[idx], RING_WAS_CAND, SURFACE))
    return w, c


def band_of(rank: np.ndarray, k: int, j: int) -> np.ndarray:
    out = np.full(rank.shape, "rest", dtype=object)
    out[rank < k + j] = "cand"
    out[rank < k] = "topk"
    return out


# Title + horizontal legend both live above the plot.  legend.y is measured in
# plot-area units, so a fixed y like 1.06 translates to a different pixel offset
# on every figure height -- on the tall support-map grid it reached past the top
# margin and sat on the title.  Fix: reserve a fixed pixel band (BANNER_T) for
# the two, pin the title to the *container* top, and put the legend just above
# the plot area.  Height-independent, so it holds for any grid size.
BANNER_T = 94


def barrier_band(fig, b: float, t: float, denom: float, axis: str, log: bool,
                 floor: float):
    """Draw b and b±t plus the shaded active-gradient band.

    ``axis="y"`` for the rank panel, where score is the vertical axis, and
    ``axis="x"`` for the half-line, where it is horizontal -- the barrier is a
    *score*, so which way the line runs is decided by where the score axis is.
    Inside b±t the Laplace CDF is unsaturated, so those are the candidates whose
    surrogate gradient is actually non-negligible.
    """
    lo, hi = (b - t) / denom, (b + t) / denom
    if log:                      # a log axis cannot show b-t once it goes <= 0
        lo = max(lo, floor)
    rect = fig.add_hrect if axis == "y" else fig.add_vrect
    line = fig.add_hline if axis == "y" else fig.add_vline
    if hi > lo:
        rect(**{f"{axis}0": lo, f"{axis}1": hi}, fillcolor=BARRIER_FILL,
             line_width=0, layer="below")
    for v, w in ((lo, 1.0), (hi, 1.0)):
        line(**{axis: v}, line=dict(color=BARRIER, width=w, dash="dot"))
    line(**{axis: b / denom}, line=dict(color=BARRIER, width=1.6))


def banner(text: str, height: int, legend: bool = True) -> dict:
    """Layout fragment: a title, and optionally a legend, that never overlap."""
    lay = dict(
        height=height + (BANNER_T - 34),
        margin=dict(l=70, r=20, t=BANNER_T, b=52),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(color=INK, size=12),
        title=dict(text=text, font=dict(size=13), x=0, xanchor="left",
                   yref="container", y=1.0, yanchor="top",
                   pad=dict(t=14, l=2)),
    )
    if legend:
        lay["legend"] = dict(orientation="h", x=0, xanchor="left",
                             y=1.0, yanchor="bottom", yref="paper",
                             font=dict(size=11))
    else:
        lay["showlegend"] = False
    return lay




# --------------------------------------------------------------------------- #
# the weight-norm page  (probe_weight_norms.py datasets)
# --------------------------------------------------------------------------- #
WN_STATS = [
    ("fro", "Frobenius norm  ||W||_F"),
    ("spec", "spectral norm  (largest singular value)"),
    ("max_row", "largest row norm  (one output unit)"),
    ("med_row", "median row norm"),
    ("row_max_over_med", "row max / median  (is it a few rows?)"),
    ("max_col", "largest column norm  (one input unit)"),
    ("med_col", "median column norm"),
    ("col_max_over_med", "column max / median"),
    ("mean_abs", "mean |W_ij|"),
]
WN_ACT = [
    ("mlp_out_max", "max ||mlp out||  (the spike, if any)"),
    ("mlp_out_med", "median ||mlp out||"),
    ("attn_out_max", "max ||attn out||"),
    ("attn_out_med", "median ||attn out||"),
    ("x_norm_max", "max ||x||  (bottleneck input)"),
    ("x_norm_med", "median ||x||"),
    ("gain_med", "median gain  ||x_hat||/||x||"),
    ("gain_max", "max gain"),
    ("score_max", "max |score|"),
]


@st.cache_data(show_spinner=False, ttl=30)
def wnorm_runs():
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(WNORM_DIR, "*.json")))


@st.cache_data(show_spinner=False)
def wnorm_load(name: str):
    """``(steps, matrices, table)`` where table[(layer, matrix, stat)] -> array."""
    m = json.load(open(os.path.join(WNORM_DIR, f"{name}.json")))
    steps = sorted({r["step"] for r in m["rows"]})
    s_idx = {s: i for i, s in enumerate(steps)}
    mats = [x for x in dict.fromkeys(r["matrix"] for r in m["rows"]) if x != "_act"]
    n_l = m["n_layers"]
    table: dict = {}
    for r in m["rows"]:
        for kk, v in r.items():
            if kk in ("step", "layer", "matrix", "shape") or not isinstance(v, (int, float)):
                continue
            key = (r["layer"], r["matrix"], kk)
            if key not in table:
                table[key] = np.full(len(steps), np.nan)
            table[key][s_idx[r["step"]]] = v
    shapes = {r["matrix"]: r.get("shape") for r in m["rows"] if r.get("shape")}
    return np.array(steps), mats, table, m, shapes, n_l


def weight_norm_page():
    names = wnorm_runs()
    if not names:
        st.info(f"No weight-norm datasets in {WNORM_DIR}. Run "
                f"`analysis/probe_weight_norms.py` first.")
        return
    with st.sidebar:
        st.subheader("Weight norms")
        picked = st.multiselect(
            "Runs to overlay", names, default=names[: min(4, len(names))],
            help="All picked runs are drawn on the same axes, so the k / j "
                 "comparison is read off one chart rather than by flipping "
                 "between them.")
        if not picked:
            st.warning("Pick at least one run.")
            return
        steps0, mats0, _, meta0, shapes0, n_l0 = wnorm_load(picked[0])
        matrix = st.selectbox(
            "Weight matrix", mats0, index=mats0.index("mlp.fc2") if "mlp.fc2" in mats0 else 0,
            format_func=lambda mm: {
                "mlp.fc1": "mlp.fc1  (before the nonlinearity)",
                "mlp.fc2": "mlp.fc2  (after the nonlinearity)",
                "attn.qkv": "attn.qkv", "attn.proj": "attn.proj",
                "bn.in_proj": "bn.in_proj  (bottleneck encoder)",
                "bn.out_proj": "bn.out_proj  (bottleneck decoder)",
            }.get(mm, mm)
            + (f"   {tuple(shapes0[mm])}" if shapes0.get(mm) else ""))
        stat = st.selectbox("Statistic", [k for k, _ in WN_STATS],
                            format_func=lambda k: dict(WN_STATS)[k])
        block = st.slider("Block", 0, n_l0 - 1, 0,
                          help="Block 0 is where the MLP spike was found to "
                               "originate; the deeper blocks mostly compound it.")
        all_blocks = st.checkbox("Overlay all blocks (one run only)", value=False)
        log_y = st.checkbox("Log y axis", value=False, key="wn_log")
        st.subheader("Step marker")
        step_i = st.select_slider(
            "Training step", options=list(range(len(steps0))), value=len(steps0) - 1,
            format_func=lambda i: f"{steps0[i]:,}",
            help="Marks the step on the curves and drives the table below.")

    st.markdown(f"### Weight norms · `{matrix}` · {dict(WN_STATS)[stat]}")
    st.caption(
        f"k / j / surrogate per run is in the legend. weight_decay="
        f"{meta0.get('weight_decay')} applies to every matrix here (the optimiser "
        f"groups by `p.dim() >= 2`, with no name-based exemptions), so growth is "
        f"happening against a restoring force.")

    fig = go.Figure()
    if all_blocks:
        run = picked[0]
        steps, mats, tab, meta, shp, n_l = wnorm_load(run)
        for li in range(n_l):
            y = tab.get((li, matrix, stat))
            if y is None:
                continue
            fig.add_trace(go.Scatter(
                x=steps, y=y, mode="lines", name=f"block {li}",
                line=dict(width=2, color=CAT[li % len(CAT)]),
                hovertemplate=f"block {li}<br>step %{{x:,}}<br>%{{y:.4g}}<extra></extra>"))
        title_extra = f" — {run}, all blocks"
    else:
        for n_, run in enumerate(picked):
            steps, mats, tab, meta, shp, n_l = wnorm_load(run)
            y = tab.get((block, matrix, stat))
            if y is None:
                continue
            jj = 0 if meta.get("surrogate_mode") == "hard" else meta["j"]
            fig.add_trace(go.Scatter(
                x=steps, y=y, mode="lines", name=f"{run}  (k={meta['k']} j={jj})",
                line=dict(width=2, color=CAT[n_ % len(CAT)]),
                hovertemplate=f"{run}<br>step %{{x:,}}<br>%{{y:.4g}}<extra></extra>"))
        title_extra = f" — block {block}"
    fig.add_vline(x=float(steps0[step_i]), line=dict(color=INK_MUTED, width=1))
    fig.update_layout(hovermode="x unified",
                      **banner(f"{matrix} · {dict(WN_STATS)[stat]}{title_extra}", 420))
    fig.update_xaxes(title="training step", gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(title=dict(WN_STATS)[stat], type="log" if log_y else "linear",
                     gridcolor=GRID, zeroline=False, linecolor=GRID)
    st.plotly_chart(fig, width="stretch", theme=None)

    # activation context: the weights are only interesting against what they produce
    st.markdown("#### What those weights produce (same block)")
    act = st.selectbox("Activation statistic", [k for k, _ in WN_ACT],
                       format_func=lambda k: dict(WN_ACT)[k], key="wn_act")
    fig2 = go.Figure()
    for n_, run in enumerate(picked):
        steps, mats, tab, meta, shp, n_l = wnorm_load(run)
        y = tab.get((block, "_act", act))
        if y is None:
            continue
        jj = 0 if meta.get("surrogate_mode") == "hard" else meta["j"]
        fig2.add_trace(go.Scatter(
            x=steps, y=y, mode="lines", name=f"{run}  (k={meta['k']} j={jj})",
            line=dict(width=2, color=CAT[n_ % len(CAT)]),
            hovertemplate=f"{run}<br>step %{{x:,}}<br>%{{y:.4g}}<extra></extra>"))
    fig2.add_vline(x=float(steps0[step_i]), line=dict(color=INK_MUTED, width=1))
    fig2.update_layout(hovermode="x unified",
                       **banner(f"{dict(WN_ACT)[act]} — block {block}", 340))
    fig2.update_xaxes(title="training step", gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig2.update_yaxes(title=dict(WN_ACT)[act], type="log", gridcolor=GRID,
                      zeroline=False, linecolor=GRID)
    st.plotly_chart(fig2, width="stretch", theme=None)

    # the step slider's table: every block x every matrix at one step
    st.markdown(f"#### All blocks x all matrices at step {steps0[step_i]:,}")
    for run in picked:
        steps, mats, tab, meta, shp, n_l = wnorm_load(run)
        i = int(np.argmin(np.abs(steps - steps0[step_i])))
        data = {"block": list(range(n_l))}
        for mm in mats:
            data[mm] = [None if tab.get((li, mm, stat)) is None
                        else round(float(tab[(li, mm, stat)][i]), 4)
                        for li in range(n_l)]
        jj = 0 if meta.get("surrogate_mode") == "hard" else meta["j"]
        st.caption(f"`{run}` — k={meta['k']} j={jj} {meta['surrogate_mode']} "
                   f"{meta['placement']} · step {steps[i]:,} · "
                   f"{dict(WN_STATS)[stat]}")
        st.dataframe(data, width="stretch", height=min(40 + 35 * n_l, 340))


# --------------------------------------------------------------------------- #
# page dispatch
# --------------------------------------------------------------------------- #
# st.stop() below means the score pages' code never runs on the weight-norm
# page, so the two do not pay for each other -- which st.tabs would not give,
# since it executes every tab body on every rerun.
_page = st.sidebar.radio(
    "Page", ["Scores & gradients", "Weight norms"], index=0,
    help="Scores & gradients: the per-cell views over checkpoint or probe "
         "datasets. Weight norms: how each block's weight matrices evolve "
         "through training, from probe_weight_norms.py.")
if _page == "Weight norms":
    weight_norm_page()
    st.stop()


# --------------------------------------------------------------------------- #
# controls
# --------------------------------------------------------------------------- #
KIND_LABEL = {"ckpt": "Checkpoint ladder (2k-20k)",
              "probe": "Early training probe (0-1k)"}
kinds = [k for k in ("ckpt", "probe") if available_runs(k)]
if not kinds:
    st.error(f"No datasets found in {SCORES_DIR} or {PROBE_DIR}. Run "
             f"extract_bottleneck_scores.py or probe_early_training.py first.")
    st.stop()

with st.sidebar:
    st.subheader("Dataset")
    kind = st.radio("Analysis", kinds, index=0,
                    format_func=lambda k: KIND_LABEL[k],
                    help="The checkpoint ladder samples every 2000 steps from "
                         "saved weights. The probe re-runs training with the "
                         "same seed and measures every 10 steps without saving "
                         "any, and it additionally carries the two gradients.")
    run = f"{kind}/" + st.selectbox("Run", available_runs(kind), index=0)
    arr, meta, arrays = load_run(run)
    C, L, B, T, N = arr.shape
    k = int(meta["k"])
    # Under surrogate_mode="hard" the J band receives *exactly* zero gradient --
    # the config still carries a j, but it is inert (verified bit-identical for
    # j = 1..1504), so it only sets the width of a diagnostics window. Treating
    # it as a live band here would paint 64 neurons as "surrogate grad" when they
    # get none, so j is forced to 0 and the analysis is TopK vs rest.
    hard_gate = str(meta.get("surrogate_mode", "")) == "hard"
    j_cfg = int(meta["j"])
    j = 0 if hard_gate else j_cfg
    steps = meta["steps"]

    step_word = "probe" if kind == "probe" else "checkpoint"
    ci = st.select_slider(f"{step_word.capitalize()} (step)", options=list(range(C)),
                          value=C - 1, format_func=lambda i: f"{steps[i]:,}")
    li = st.slider("Bottleneck (block)", 0, L - 1, L // 2)
    bi = st.slider("Sequence in batch", 0, B - 1, 0)
    ti = st.slider("Token position", 0, T - 1, min(64, T - 1))

    st.subheader("Scale")
    norm = st.radio(
        "Normalization",
        options=["raw", "thresh", "max"],
        index=0,
        format_func=lambda s: {"thresh": "/ r[k-1]  (selection threshold)",
                               "max": "/ r[0]  (largest score)",
                               "raw": "raw |score|"}[s],
        help="The raw scale falls ~5x over training and varies >10x across "
             "blocks, which swamps everything else. Dividing by the selection "
             "threshold pins the TopK edge at 1.0 so shape changes are visible.",
    )
    log_y = st.checkbox("Log score axis", value=True,
                        help="|score| spans ~4 orders of magnitude within one cell.")
    freeze = st.checkbox(f"Freeze axis across {step_word}s", value=True,
                         help=f"Hold the range fixed over all {step_word}s for "
                              "this cell, so the slider shows change and not "
                              "rescaling.")

    st.subheader("Bands")
    show_rest = st.checkbox("Show 'rest' band", value=True)
    max_rest = st.slider("Max 'rest' points drawn", 100, N, min(600, N), step=100,
                         help="The rest band's internal ordering is about half "
                              "noise between checkpoints (Spearman ~0.50), so a "
                              "subsample shows the same envelope with less ink.")
    carry = st.selectbox(
        "Ring neurons by their band at...",
        options=["none", "prev position", "prev checkpoint"], index=2,
        format_func=lambda o: o.replace("checkpoint", step_word),
        help="The ring shows which band a neuron occupied in the comparison "
             "frame; the fill shows the band it is in now. So a black-ringed "
             "blue circle held its TopK place, a magenta-ringed blue circle was "
             "promoted out of the J band, and an unringed blue circle came from "
             "outside k+j entirely. Answers 'does the TopK persist?' directly, "
             "instead of by hovering 32 circles and memorizing indices.",
    )

    st.subheader("Axis cuts")
    # Bounds span the whole dataset (run_range), not the selected cell, so the
    # sliders keep their meaning as you move through blocks/sequences/positions/
    # steps.  Keyed by run and units, which are the only changes that should
    # reset them.  The two cuts are independent on purpose: it is useful to keep
    # the full decay curve on the rank panel while zooming the half-line onto the
    # boundary, or the reverse.
    _lo, _hi = run_range(run, norm, k)
    # Keyed by units only, not by run: the remembered value is meant to outlive a
    # run change (min_cut_slider clamps it if the new run's range cannot hold
    # it).  Units are excluded from that -- a cut of 0.5 means something quite
    # different in raw scores than in units of r[k-1].
    sr_min = min_cut_slider("Min y on Score by rank", f"sr_{norm}", _lo, _hi, log_y)
    hl_min = min_cut_slider("Min x on the half-line", f"hl_{norm}", _lo, _hi, log_y)
    st.caption(f"cuts: rank-panel y \u2265 {sr_min:.4g} \u00b7 "
               f"half-line x \u2265 {hl_min:.4g}")

    st.subheader("Support map")
    map_layout = st.radio(
        "Layout", options=["grid", "row"], index=0, horizontal=True,
        help="The map is ordered by neuron index, so a neuron keeps its place "
             "between frames and you can watch the support move. 'row' is the "
             "literal single sequence; at n_features=1536 that is under a pixel "
             "per circle, so 'grid' wraps it in reading order instead.",
    )
    # divisors of 1536, so the grid has no ragged final row
    ncols = st.select_slider(
        "Grid width", options=[c for c in (24, 32, 48, 64, 96, 128) if N % c == 0],
        value=48, disabled=map_layout == "row",
    )
    map_rest = st.checkbox("Show 'rest' neurons in the map", value=True,
                           help="Drawn faint. Turn off to see only the k+j "
                                "selected neurons against empty space.")

# --------------------------------------------------------------------------- #
# the cell
# --------------------------------------------------------------------------- #
signed = np.asarray(arr[ci, li, bi, ti], dtype=np.float64)
r = np.abs(signed)
order = np.argsort(-r, kind="stable")
rank = np.empty(N, dtype=np.int64)
rank[order] = np.arange(N)
thresh, rmax = r[order[k - 1]], r[order[0]]
denom = {"thresh": thresh, "max": rmax, "raw": 1.0}[norm]
val = r / denom
bands = band_of(rank, k, j)

was_topk = np.zeros(N, dtype=bool)
was_cand = np.zeros(N, dtype=bool)
if carry == "prev position" and ti > 0:
    pt, pc = bands_at(run, ci, li, bi, ti - 1, k, j)
elif carry == "prev checkpoint" and ci > 0:
    pt, pc = bands_at(run, ci - 1, li, bi, ti, k, j)
else:
    pt, pc = frozenset(), frozenset()
carry_label = carry.replace("checkpoint", step_word)
was_topk[list(pt)] = True
was_cand[list(pc)] = True
carried = was_topk | was_cand          # "had any gradient in the comparison frame"

cur_top = set(order[:k].tolist())
cur_pool = set(order[: k + j].tolist())     # the whole Top(K+J) candidate pool
jac_ck = jac_pool_ck = None
if ci > 0:
    p_top, p_cand = bands_at(run, ci - 1, li, bi, ti, k, j)
    jac_ck = len(cur_top & p_top) / len(cur_top | p_top)
    prev_pool = p_top | p_cand
    jac_pool_ck = len(cur_pool & prev_pool) / len(cur_pool | prev_pool)

st.markdown(f"### `{run}`  ·  step {steps[ci]:,}  ·  block {li}  ·  seq {bi}  ·  position {ti}")
tok = meta.get("token_strs")
st.caption(
    f"k={k}  "
    + (f"j={j_cfg} (inert: hard gate)  " if hard_gate else f"j={j}  ")
    + f"n_features={N}  placement={meta['placement']}  "
    f"selection={meta['selection_mode']}  surrogate={meta['surrogate_mode']}  ·  "
    f"batch CE at this checkpoint {meta['batch_ce'][ci]:.4f}"
    + (f"  ·  token here: {tok[bi][ti]!r}" if tok else "")
)

# A diverged run still has checkpoints and still plots, so say so loudly rather
# than letting a 1e10 score axis read as a property of the method.
ce = meta["batch_ce"]
best_i = int(np.argmin(ce))
if ce[-1] > min(ce) * 1.05:
    st.warning(
        f"**This run degraded.** Batch CE bottoms out at {ce[best_i]:.3f} "
        f"(step {steps[best_i]:,}) and ends at {ce[-1]:.3f} (step {steps[-1]:,}). "
        f"Scores at the last checkpoint reach {np.abs(np.asarray(arr[-1, li, bi, ti])).max():.3g}. "
        f"Read checkpoints after step {steps[best_i]:,} as a collapse, not as training progress."
    )

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("r[0]  (largest)", f"{rmax:,.3f}")
c2.metric("r[k-1]  (threshold)", f"{thresh:,.3f}")
# 3 significant figures, not 2 decimals: at k=64 the margin is ~0.002% of r[0],
# which "%.2f" renders as a flat "0.00" and reads as a missing value rather than
# as the real and rather striking number it is.
margin = r[order[k - 1]] - r[order[k]]
c3.metric("selection margin", f"{margin / rmax * 100:.3g}% of r[0]",
          help=f"r[k-1]-r[k] = {margin:.3g} in absolute terms, against r[0] = "
               f"{rmax:.3g} and a threshold r[k-1] = {thresh:.3g}. This is why "
               "the TopK edge is invisible on a bare number line: the neuron "
               "that just made the cut and the one that just missed are nearly "
               "tied.")
c4.metric(f"TopK ∩ prev {step_word}", "n/a" if jac_ck is None else f"{jac_ck:.3f}",
          help="Jaccard overlap of the two TopK sets: |A∩B| / |A∪B|, so 1.0 is "
               "an identical selection and 0.0 a disjoint one.")
c5.metric(f"Top(K+J) ∩ prev {step_word}",
          "n/a (hard gate)" if j == 0 else
          ("n/a" if jac_pool_ck is None else f"{jac_pool_ck:.3f}"),
          help="The same overlap for the whole k+j candidate pool -- everything "
               "the gate gives any gradient to. Compare it with the TopK figure: "
               "it usually comes out *lower* (0.36 vs 0.49 at the default cell), "
               "i.e. the TopK core is the more stable part. Expected, since rank "
               "k+j sits in the flat tail of the score curve where a tiny score "
               "change reorders membership, while the top ranks are separated by "
               "large margins.")

# ---- which points to draw ------------------------------------------------- #
draw = np.ones(N, dtype=bool)
if not show_rest:
    draw &= bands != "rest"
elif max_rest < (N - k - j):
    keep = order[k + j:][np.linspace(0, N - k - j - 1, max_rest).astype(int)]
    mask = np.zeros(N, dtype=bool)
    mask[order[: k + j]] = True
    mask[keep] = True
    draw &= mask

yr = None
if freeze:
    lo, hi = axis_range(run, li, bi, ti, norm, k)
    pad = (np.log10(hi) - np.log10(max(lo, 1e-12))) * 0.04
    yr = ([np.log10(max(lo, 1e-12)) - pad, np.log10(hi) + pad] if log_y
          else [0, hi * 1.04])

score_label = {"thresh": "|score| / r[k-1]", "max": "|score| / r[0]",
               "raw": "|score|"}[norm]
bt = barrier_at(run, ci, li, bi, ti, k, j)
# floor for the log axis, so a b-t that goes non-positive still has somewhere
# to be drawn instead of silently dropping the whole rect
axis_floor = ((10 ** yr[0]) if (yr is not None and log_y)
              else (val[val > 0].min() if log_y else 0.0))


def hover(i):
    return (f"feature {i}<br>rank {rank[i]}<br>{score_label} {val[i]:.4g}"
            f"<br>raw |score| {r[i]:.4g}<br>signed {signed[i]:+.4g}"
            f"<br>{BAND_LABEL[bands[i]]}" + (f"<br><b>was {'TopK' if was_topk[i] else 'a J candidate'}"
                 f" at the {carry_label}</b>" if carried[i] else ""))


# --------------------------------------------------------------------------- #
# panel 1 -- rank x score
# --------------------------------------------------------------------------- #
fig = go.Figure()
# A display filter, so it is never silent -- the title reports what it removes.
sr_keep = val >= sr_min
sr_shown, sr_total = int((draw & sr_keep).sum()), int(draw.sum())
for b in ("rest", "cand", "topk"):          # rest first so it sits underneath
    sel = draw & sr_keep & (bands == b)
    if not sel.any():
        continue
    idx = np.where(sel)[0]
    idx = idx[np.argsort(rank[idx])]
    fig.add_trace(go.Scattergl(
        x=rank[idx] + 1, y=val[idx], mode="markers", name=BAND_LABEL[b],
        legendrank={"topk": 1, "cand": 2, "rest": 3}[b],
        marker=dict(
            size=9 if b != "rest" else 6,
            color=BAND_COLOR[b],
            opacity=1.0 if b != "rest" else 0.55,
            line=dict(width=np.where(carried[idx], 2.5, 1.0),
                      color=ring(idx, was_topk, was_cand, 2.5)[1]),
        ),
        text=[hover(i) for i in idx], hoverinfo="text",
    ))
# Two plotly log-axis traps here, both hit once:
#   * add_vline's *auto-annotation* takes x in log10 space (the line shape does
#     not), so annotation_text=... at x=64.5 lands at 10^64.5 and stretches the
#     axis by 60-odd decades.  Hence a separate add_annotation at log10(x).
#   * a paper-referenced y puts the label in the title band, where it collides
#     with the title -- so it is anchored inside the plot via "y domain".
edges = [(k, f"k={k}")] + ([] if j == 0 else [(k + j, f"k+j={k + j}")])
for edge, lab in edges:
    fig.add_vline(x=edge + 0.5, line=dict(color=INK_MUTED, width=1))
    fig.add_annotation(x=np.log10(edge + 0.5), y=0.97, yref="y domain",
                       text=lab, showarrow=False, xanchor="left", yanchor="top",
                       font=dict(size=11, color=INK_MUTED))
if norm == "thresh":
    fig.add_hline(y=1.0, line=dict(color=INK_MUTED, width=1))
sr_r = cut_range(yr, sr_min, log_y, float(val.max()))
sr_note = ("" if sr_shown == sr_total else
           f"  ·  showing {sr_shown:,} of {sr_total:,} drawn neurons "
           f"(y \u2265 {sr_min:.4g})")
fig.update_layout(
    hovermode="closest", hoverdistance=12,
    **banner("Score by rank" + sr_note, 460),
)
fig.update_xaxes(type="log", title="rank (1 = largest |score|)",
                 gridcolor=GRID, zeroline=False, linecolor=GRID)
sr_decades = (sr_r[1] - sr_r[0]) if (log_y and sr_r is not None) else 99.0
sr_wide = not log_y or sr_decades >= 1.5
fig.update_yaxes(type="log" if log_y else "linear", title=score_label,
                 range=sr_r, gridcolor=GRID, zeroline=False, linecolor=GRID,
                 dtick=1 if (log_y and sr_wide) else None,
                 tickformat=None if sr_wide else ".3g")
if bt is not None:
    barrier_band(fig, bt[0], bt[1], denom, "y", log_y,
                 max(axis_floor, sr_min) if log_y else axis_floor)
    # Only label b when it is actually on the visible axis; a cut above it would
    # otherwise leave the letter pinned to the bottom edge, pointing at nothing.
    if bt[0] / denom >= sr_min:
        fig.add_annotation(x=0.0, xref="x domain", y=bt[0] / denom, yanchor="bottom",
                           xanchor="left", text="b", showarrow=False,
                           font=dict(size=11, color=BARRIER))
st.plotly_chart(fig, width="stretch", theme=None)
if bt is None and hard_gate:
    st.caption(
        "No barrier band: a hard gate never solves one, so there is no "
        "temperature window to draw. The gradient is the hard mask's, and only "
        f"the k={k} edge exists -- the config's j={j_cfg} is inert."
    )
elif bt is not None:
    st.caption(
        f"Violet band = **b ± t**, the LapSum active-gradient window: "
        f"b = {bt[0]:.4g}, t = {bt[1]:.4g} (raw score units; "
        f"t = schedule({steps[ci]:,}) x std of the top-{k + j} scores, and b solves "
        f"&Sigma; F((r&minus;b)/t) = k over that pool). Candidates inside it have an "
        f"unsaturated Laplace CDF, so they are the ones whose surrogate gradient is "
        f"actually non-negligible."
    )

# --------------------------------------------------------------------------- #
# panel 2 -- the half-line, as specified
# --------------------------------------------------------------------------- #
fig2 = go.Figure()
jt = jitter(N)
# The cut is a *display* filter, so it must never be silent -- the panel title
# below reports how many neurons it removes.
hl_keep = val >= hl_min
n_shown, n_total = int((draw & hl_keep).sum()), int(draw.sum())
for b in ("rest", "cand", "topk"):
    sel = draw & hl_keep & (bands == b)
    if not sel.any():
        continue
    idx = np.where(sel)[0]
    fig2.add_trace(go.Scattergl(
        x=val[idx], y=jt[idx], mode="markers", name=BAND_LABEL[b],
        marker=dict(
            size=10 if b != "rest" else 6,
            color=BAND_COLOR[b],
            opacity=1.0 if b != "rest" else 0.45,
            line=dict(width=np.where(carried[idx], 2.5, 1.0),
                      color=ring(idx, was_topk, was_cand, 2.5)[1]),
        ),
        text=[hover(i) for i in idx], hoverinfo="text", showlegend=False,
        legendrank={"topk": 1, "cand": 2, "rest": 3}[b],
    ))
# The band edges are score *thresholds* here, not ranks, so they go at the
# midpoint between the last member of one band and the first of the next.
# The thresholds can sit a factor of ~1.25 apart (a few dozen pixels on a log
# axis spanning four decades), so the labels are staggered in y and anchored to
# opposite sides -- side-anchoring alone is not enough separation.
for n_e, (edge, lab) in enumerate(edges):
    mid = (r[order[edge - 1]] + r[order[min(edge, N - 1)]]) / 2.0 / denom
    fig2.add_vline(x=mid, line=dict(color=INK_MUTED, width=1))
    fig2.add_annotation(x=np.log10(mid) if log_y else mid,
                        y=1.0 - 0.17 * n_e, yref="y domain",
                        text=lab, showarrow=False, yanchor="top",
                        xanchor="left" if n_e == 0 else "right",
                        font=dict(size=11, color=INK_MUTED))
if bt is not None:
    barrier_band(fig2, bt[0], bt[1], denom, "x", log_y,
                 max(axis_floor, hl_min) if log_y else axis_floor)
# Same right edge as the frozen axis; left edge follows the cut.  Plotly wants
# log10 units for a log axis, which is what `yr` already carries.
hl_r = cut_range(yr, hl_min, log_y, float(val.max()))
cut_note = ("" if n_shown == n_total else
            f"  ·  showing {n_shown:,} of {n_total:,} drawn neurons "
            f"(x \u2265 {hl_min:.4g})")
fig2.update_layout(
    hovermode="closest", hoverdistance=12,
    **banner("The same cell on the half-line (vertical offset is jitter only)"
             + cut_note, 250, legend=False),
)
# Decade-only ticks suit a four-decade axis, but once the left cut narrows the
# view to well under a decade they leave a single labelled tick. Fall back to
# plotly's automatic ticks when the visible span is short.
hl_decades = (hl_r[1] - hl_r[0]) if (log_y and hl_r is not None) else 99.0
hl_wide = not log_y or hl_decades >= 1.5
fig2.update_xaxes(type="log" if log_y else "linear", title=score_label,
                  range=hl_r, gridcolor=GRID, zeroline=False, linecolor=GRID,
                  dtick=1 if (log_y and hl_wide) else None,
                  tickformat=None if hl_wide else ".3g")
fig2.update_yaxes(visible=False, range=[-2.4, 2.4])
st.plotly_chart(fig2, width="stretch", theme=None)

# --------------------------------------------------------------------------- #
# panel 3 -- the support map, ordered by neuron index
# --------------------------------------------------------------------------- #
# Ordered by feature index on purpose.  Ordered by rank this panel would be the
# same picture in every frame -- k circles of one colour then j of another -- and
# would carry no information at all.  Index order gives each neuron a fixed seat,
# so dragging a slider shows the support itself moving.
if map_layout == "row":
    mx, my = np.arange(N), np.zeros(N)
    nrows, cell = 1, 12
    m_height, yr_map = 190, [-1.0, 1.0]
else:
    nrows = int(np.ceil(N / ncols))
    mx, my = np.arange(N) % ncols, -(np.arange(N) // ncols)
    cell = int(min(24, max(6, 620 // nrows)))
    m_height, yr_map = nrows * cell + 96, [-nrows + 0.5, 0.5]
msize = max(3.0, cell * 0.62) if map_layout == "grid" else 7.0

fig_map = go.Figure()
for b in ("rest", "cand", "topk"):
    sel = bands == b
    if b == "rest" and not map_rest:
        continue
    if not sel.any():
        continue
    idx = np.where(sel)[0]
    # The rest band is 1400+ of the 1536 seats; drawn faint it reads as the empty
    # backdrop the selected neurons sit in.  But a *dropped* neuron -- one that
    # was TopK in the comparison frame and is now rest -- is signal, not
    # backdrop, so it is lifted out of the faint layer and ringed in muted ink
    # rather than full ink, which would make it compete with the live support.
    # Ring width scales with the marker: a fixed 2px ring on the ~5.6px circles
    # of a 24-column grid swallows the fill, so retained TopK reads as dark navy
    # instead of ringed blue.
    rw_max = float(np.clip(msize * 0.22, 1.0, 2.5))
    ring_w, ring_c = ring(idx, was_topk, was_cand, rw_max)
    if b == "rest":
        # a neuron that had gradient last frame and is now rest is signal, not
        # backdrop, so it is lifted out of the faint layer
        op = np.where(carried[idx], 0.6, 0.22)
    else:
        op = 1.0
    fig_map.add_trace(go.Scattergl(
        x=mx[idx], y=my[idx], mode="markers", name=BAND_LABEL[b],
        legendrank={"topk": 1, "cand": 2, "rest": 3}[b],
        marker=dict(
            size=msize if b != "rest" else msize * 0.72,
            color=BAND_COLOR[b], opacity=op,
            line=dict(width=ring_w, color=ring_c),
        ),
        text=[hover(i) for i in idx], hoverinfo="text",
    ))

if carry != "none":
    top_now = order[:k]
    n_kept = int(was_topk[top_now].sum())
    n_from_j = int(was_cand[top_now].sum())
    if j == 0:
        sub = (f"  ·  of {k} TopK now: {n_kept} held from the {carry}, "
               f"{k - n_kept} new")
    else:
        sub = (f"  ·  of {k} TopK now: {n_kept} held from the {carry}, "
               f"{n_from_j} promoted from J, {k - n_kept - n_from_j} from outside k+j")
else:
    sub = ""
fig_map.update_layout(
    hovermode="closest", hoverdistance=12,
    **banner(f"Support over neuron index "
             f"({'single row' if map_layout == 'row' else f'{ncols} per row, reading order'})"
             + sub, m_height),
)
fig_map.update_xaxes(
    title="neuron index" if map_layout == "row" else f"neuron index mod {ncols}",
    gridcolor=GRID, zeroline=False, linecolor=GRID,
    range=[-0.5, (N if map_layout == "row" else ncols) - 0.5],
)
fig_map.update_yaxes(
    visible=map_layout == "grid", range=yr_map, gridcolor=GRID, zeroline=False,
    linecolor=GRID,
    title="index // %d" % ncols if map_layout == "grid" else None,
    tickmode="array",
    tickvals=list(range(0, -nrows, -max(1, nrows // 8))) if map_layout == "grid" else [],
    ticktext=[str(-v * ncols) for v in range(0, -nrows, -max(1, nrows // 8))]
    if map_layout == "grid" else [],
)
st.plotly_chart(fig_map, width="stretch", theme=None)
if carry != "none":
    # The plotly legend can only describe the fills, so the ring encoding is
    # spelled out here -- this doubles as the visible-label relief the light-mode
    # contrast warning on the aqua fill requires.
    bits = [f"<span style='color:{RING_WAS_TOPK}'>&#9679;</span> black ring = was TopK"]
    if j:
        bits.append(f"<span style='color:{RING_WAS_CAND}'>&#9679;</span> magenta ring "
                    f"= was a J candidate")
    bits.append(f"no ring = was outside {'k+j' if j else 'TopK'}")
    st.caption(
        f"Fill = band **now**. Ring = band at the **{carry}**: " + " · ".join(bits)
        + ". A ringed faint circle lost its gradient; an unringed coloured circle "
          "just gained one.",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
# panel 4 -- the scale trajectory this cell sits in
# --------------------------------------------------------------------------- #
blk = np.abs(np.asarray(arr[:, li, bi, ti], dtype=np.float64))
srt = -np.sort(-blk, axis=-1)
fig3 = go.Figure()
traces = [(0, "r[0]", BAND_COLOR["topk"]), (k - 1, "r[k-1]", BAND_COLOR["cand"])]
traces += ([(k + j - 1, "r[k+j-1]", BAND_COLOR["rest"])] if j else
           [(min(2 * k - 1, N - 1), f"r[{min(2 * k, N)}-1]", BAND_COLOR["rest"])])
for rk, nm, col in traces:
    fig3.add_trace(go.Scatter(
        x=steps, y=srt[:, rk], mode="lines+markers", name=nm,
        line=dict(width=2, color=col), marker=dict(size=7, color=col),
        hovertemplate=f"{nm}<br>step %{{x:,}}<br>%{{y:.4g}}<extra></extra>",
    ))
fig3.add_vline(x=steps[ci], line=dict(color=INK_MUTED, width=1))
fig3.update_layout(
    hovermode="x unified",
    **banner("Raw score scale for this cell across training "
             "(the effect normalization removes)", 250),
)
fig3.update_xaxes(title="step", gridcolor=GRID, zeroline=False, linecolor=GRID)
fig3.update_yaxes(type="log", title="raw |score|", gridcolor=GRID,
                  zeroline=False, linecolor=GRID, dtick=1)
st.plotly_chart(fig3, width="stretch", theme=None)

# --------------------------------------------------------------------------- #
# panel 5 -- gradients against |score|   (probe datasets only)
# --------------------------------------------------------------------------- #
# The question these answer: does the candidate pool receive gradients that
# point the same way, i.e. is the surrogate driving the whole pool in one
# direction, or is it pushing members against each other?
#
#   dL/d~z  is the gradient at the gate's *output*.  It does not pass through
#           the mask, so it is generally non-zero even for a neuron TopK did not
#           select -- that is why it is worth plotting outside TopK at all.
#   dL/dz   is the gradient at the gate's *input*, after the LapSum surrogate.
#           It is exactly zero outside Top(K+J) by construction, so the "rest"
#           band lying flat on y=0 is the encoding working, not missing data.
grad_present = [g for g in GRAD_ARRAYS if g in arrays]
if grad_present:
    st.markdown("#### Gradients at this cell")
    gcol1, gcol2 = st.columns([3, 2])
    with gcol1:
        grad_signed = st.radio(
            "Gradient axis", ["signed", "magnitude"], index=0, horizontal=True,
            help="Signed on a linear axis answers the alignment question and is "
                 "the default. Magnitude on a log axis shows the spread across "
                 "orders of magnitude but throws the direction away.",
        )
    with gcol2:
        grad_rest = st.checkbox(
            "include 'rest' band", value=False,
            help="Off by default: the ask is the k+j pool. Worth turning on for "
                 "dL/d~z, which is non-zero outside the pool, and as a check on "
                 "dL/dz, which must be exactly zero there.",
        )

    if hard_gate:
        st.caption(
            f"Hard gate: the pool is just the k={k} TopK. dL/dz is exactly zero "
            f"outside it (the J band gets *nothing*, not a small amount), which "
            f"is what \"include 'rest' band\" lets you confirm -- those points "
            f"land on y=0. dL/d\u007ez is still dense. **The two panels below "
            f"are identical on the pool, and that is correct**: the mask is 1 "
            f"on TopK, so dL/dz = dL/d\u007ez there exactly (measured 0.0e+00 "
            f"across probes and blocks). They differ only off the pool, where "
            f"dL/dz is zeroed and dL/d\u007ez is not."
        )
    for gname in grad_present:
        gv_full = np.asarray(arrays[gname][ci, li, bi, ti], dtype=np.float64)
        pool = rank < (k + j if j else k)
        gsel = pool | grad_rest
        yv = gv_full if grad_signed == "signed" else np.abs(gv_full)
        figg = go.Figure()
        for bnd in ("rest", "cand", "topk"):
            m_ = gsel & (bands == bnd)
            if not m_.any():
                continue
            idx = np.where(m_)[0]
            figg.add_trace(go.Scattergl(
                x=val[idx], y=yv[idx], mode="markers", name=BAND_LABEL[bnd],
                legendrank={"topk": 1, "cand": 2, "rest": 3}[bnd],
                marker=dict(size=9 if bnd != "rest" else 5,
                            color=BAND_COLOR[bnd],
                            opacity=1.0 if bnd != "rest" else 0.45,
                            line=dict(width=np.where(carried[idx], 2.0, 0.0),
                                      color=ring(idx, was_topk, was_cand, 2.0)[1])),
                text=[f"feature {i}<br>rank {rank[i]}<br>{score_label} {val[i]:.4g}"
                      f"<br>{gname} {gv_full[i]:+.4g}<br>{BAND_LABEL[bands[i]]}"
                      for i in idx],
                hoverinfo="text",
            ))
        # score-axis furniture, exactly as on the half-line
        for n_e, (edge, lab) in enumerate(edges):
            mid = (r[order[edge - 1]] + r[order[min(edge, N - 1)]]) / 2.0 / denom
            figg.add_vline(x=mid, line=dict(color=INK_MUTED, width=1))
            figg.add_annotation(x=np.log10(mid) if log_y else mid,
                                y=1.0 - 0.17 * n_e, yref="y domain", text=lab,
                                showarrow=False, yanchor="top",
                                xanchor="left" if n_e == 0 else "right",
                                font=dict(size=11, color=INK_MUTED))
        if bt is not None:
            barrier_band(figg, bt[0], bt[1], denom, "x", log_y,
                         max(axis_floor, hl_min) if log_y else axis_floor)
        if grad_signed == "signed":
            # the reference the whole panel is read against
            figg.add_hline(y=0.0, line=dict(color=INK_MUTED, width=1))
        title = GRAD_LABEL[gname]
        if gname == "g_z" and hard_gate:
            title = "dL/dz  (gate input, straight through the hard mask)"
        figg.update_layout(hovermode="closest", hoverdistance=12,
                           **banner(title, 300))
        figg.update_xaxes(type="log" if log_y else "linear", title=score_label,
                          range=hl_r, gridcolor=GRID, zeroline=False,
                          linecolor=GRID,
                          dtick=1 if (log_y and hl_wide) else None,
                          tickformat=None if hl_wide else ".3g")
        figg.update_yaxes(type="log" if grad_signed == "magnitude" else "linear",
                          title=(GRAD_AXIS[gname] if grad_signed == "signed"
                                 else f"|{GRAD_AXIS[gname]}|"),
                          gridcolor=GRID, zeroline=False,
                          linecolor=GRID, exponentformat="e")
        st.plotly_chart(figg, width="stretch", theme=None)

        # The alignment question, as a number rather than an eyeball judgement.
        gp = gv_full[pool]
        nz = gp[gp != 0]
        if nz.size:
            frac_pos = float((nz > 0).mean())
            agree = max(frac_pos, 1.0 - frac_pos)
            gk = gv_full[rank < k]
            gj = gv_full[(rank >= k) & (rank < k + j)] if j else np.array([])
            bits = [f"pool sign agreement **{agree:.1%}** "
                    f"({frac_pos:.1%} positive of {nz.size} non-zero)",
                    f"mean **{gp.mean():+.3g}**",
                    f"mean |.| **{np.abs(gp).mean():.3g}**",
                    f"|mean|/mean|.| **{abs(gp.mean()) / max(np.abs(gp).mean(), 1e-30):.3f}**"]
            if gj.size:
                bits.append(f"TopK mean **{gk.mean():+.3g}** vs J mean **{gj.mean():+.3g}**")
            st.caption("  \u00b7  ".join(bits)
                       + ".  |mean|/mean|.| near 1 means the pool is pushed one way; "
                         "near 0 means the contributions cancel.")

# ---- table view (relief for the light-mode contrast warning) -------------- #
with st.expander(f"Table: the top {k + j if j else k} neurons in this cell"):
    idx = order[: k + j]
    st.dataframe(
        {
            "rank": rank[idx],
            "feature": idx,
            "band": [BAND_LABEL[b] for b in bands[idx]],
            score_label: np.round(val[idx], 5),
            "raw |score|": np.round(r[idx], 5),
            "signed": np.round(signed[idx], 5),
            f"band at {carry_label}": np.where(
                was_topk[idx], "TopK",
                np.where(was_cand[idx], "J candidate", "-")),
        },
        width="stretch", height=340,
    )
