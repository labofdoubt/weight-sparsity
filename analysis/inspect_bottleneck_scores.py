"""Characterize the cached score tensor before choosing how to draw it.

Answers the questions that decide the plot format:

* how wide is the dynamic range of |score|, and does the tail collapse onto zero
  (i.e. would 1536 circles on a linear axis be readable at all)?
* how visible is the TopK/J boundary -- the gap between rank k-1 and rank k,
  relative to the spread within each band?
* how much of the drawn ink is the "rest" band, and does it carry any signal?
* do the TopK *identities* persist across neighbouring token positions and
  across checkpoints (the thing hover-for-index is meant to answer)?

    python analysis/inspect_bottleneck_scores.py \
        --scores /workspace/analysis/scores/dc_rout_soft_k32_j32.npy
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def band_edges(k: int, j: int, n: int):
    bands = [("topk", 0, k)]
    if j:
        bands.append(("cand", k, k + j))
    return bands + [("rest", k + j, n)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--meta", default=None)
    args = ap.parse_args()

    meta_path = args.meta or args.scores.replace(".npy", ".json")
    meta = json.load(open(meta_path))
    a = np.load(args.scores, mmap_mode="r")
    C, L, B, T, N = a.shape
    n_show = N
    k = int(meta["k"])
    # j is inert under a hard gate (the J ranks get exactly zero gradient, same
    # as the rest), so reporting a "cand" band for those runs would invent a
    # distinction the model does not make.
    hard_gate = str(meta.get("surrogate_mode", "")) == "hard"
    j = 0 if hard_gate else int(meta["j"])
    steps = meta["steps"]
    print(f"{os.path.basename(args.scores)}  shape={a.shape}  k={k} j={j} N={N}")
    print(f"steps: {steps}")
    print(f"selection_mode={meta['selection_mode']}  placement={meta['placement']}  "
          f"surrogate={meta.get('surrogate_mode')}"
          + (f"  (j={meta['j']} in config but INERT)" if hard_gate else ""))
    print()

    # ---- 1. dynamic range, per band, at a few (ckpt, layer) cells ----------- #
    print("=" * 78)
    print("1. |score| dynamic range by band   (median [p1..p99] over the cell)")
    print("=" * 78)
    hdr = "".join(f"{nm:>26}" for nm, _, _ in band_edges(k, j, N))
    print(f"{'ckpt':>6} {'layer':>5} |{hdr}")
    for ci in (0, C // 2, C - 1):
        for li in (0, L // 2, L - 1):
            r = np.abs(np.asarray(a[ci, li], dtype=np.float64))  # (B,T,N)
            srt = -np.sort(-r, axis=-1)
            row = f"{steps[ci]:>6} {li:>5} |"
            for _, lo, hi in band_edges(k, j, N):
                seg = srt[..., lo:hi]
                row += (f" {np.median(seg):8.4f}"
                        f"[{np.percentile(seg, 1):.4f}..{np.percentile(seg, 99):.4f}]")
            print(row)
    print()

    # ---- 2. how readable is the boundary ----------------------------------- #
    print("=" * 78)
    print("2. TopK/J boundary visibility")
    print("=" * 78)
    print("   gap = r[k-1]-r[k]  (the selection margin)")
    print("   span_topk = r[0]-r[k-1];  frac = gap / span_topk")
    print(f"{'ckpt':>6} {'layer':>5} {'r[0]':>9} {'r[k-1]':>9} {'r[k]':>9} "
          f"{('r[k+j]' if j else 'r[k+j] n/a'):>9} {'gap':>9} {'gap/span':>9}")
    for ci in (0, C // 2, C - 1):
        for li in (0, L // 2, L - 1):
            r = np.abs(np.asarray(a[ci, li], dtype=np.float64))
            srt = -np.sort(-r, axis=-1)
            r0 = srt[..., 0].mean()
            rk1 = srt[..., k - 1].mean()
            rk = srt[..., k].mean()
            rkj = srt[..., min(k + j, n_show - 1)].mean() if j else float("nan")
            gap = (srt[..., k - 1] - srt[..., k]).mean()
            span = max(r0 - rk1, 1e-12)
            print(f"{steps[ci]:>6} {li:>5} {r0:9.4f} {rk1:9.4f} {rk:9.4f} "
                  f"{rkj:9.4f} {gap:9.5f} {gap / span:9.4f}")
    print()

    # ---- 3. what fraction of the line the bands occupy --------------------- #
    print("=" * 78)
    print("3. Where the mass sits on a LINEAR |score| axis")
    print("=" * 78)
    print("   fraction of the axis [0, r_max] spanned by each band, and how many")
    print("   'rest' points fall inside the first 1% of the axis")
    for ci in (0, C - 1):
        for li in (0, L // 2):
            r = np.abs(np.asarray(a[ci, li], dtype=np.float64))
            srt = -np.sort(-r, axis=-1)
            rmax = srt[..., 0]
            out = [f"step {steps[ci]:>6} layer {li}:"]
            for nm, lo, hi in band_edges(k, j, N):
                width = (srt[..., lo] - srt[..., hi - 1]) / rmax
                out.append(f"{nm} spans {width.mean() * 100:5.1f}% of axis")
            rest = srt[..., k + j:]
            frac_tiny = (rest < 0.01 * rmax[..., None]).mean()
            out.append(f"| {frac_tiny * 100:.1f}% of 'rest' in bottom 1% of axis")
            print("   " + "  ".join(out))
    print()

    # ---- 4. identity persistence ------------------------------------------ #
    print("=" * 78)
    print("4. Do TopK identities persist?   (mean Jaccard of the TopK sets)")
    print("=" * 78)
    for li in (0, L // 2, L - 1):
        # neighbouring token positions, at the last checkpoint
        r = np.abs(np.asarray(a[C - 1, li], dtype=np.float64))
        top = np.argsort(-r, axis=-1)[..., :k]              # (B,T,k)
        sets = [[set(top[b, t].tolist()) for t in range(T)] for b in range(B)]
        adj = np.mean([len(sets[b][t] & sets[b][t + 1]) / len(sets[b][t] | sets[b][t + 1])
                       for b in range(B) for t in range(T - 1)])
        far = np.mean([len(sets[b][t] & sets[b][(t + 37) % T]) / len(sets[b][t] | sets[b][(t + 37) % T])
                       for b in range(B) for t in range(T)])
        # same position, consecutive checkpoints
        r0 = np.abs(np.asarray(a[C - 2, li], dtype=np.float64))
        top0 = np.argsort(-r0, axis=-1)[..., :k]
        ck = np.mean([len(set(top0[b, t].tolist()) & set(top[b, t].tolist()))
                      / len(set(top0[b, t].tolist()) | set(top[b, t].tolist()))
                      for b in range(B) for t in range(T)])
        # across sequences at the same position
        cross = np.mean([len(sets[0][t] & sets[b][t]) / len(sets[0][t] | sets[b][t])
                         for b in range(1, B) for t in range(T)])
        print(f"   layer {li}: adjacent pos {adj:.3f} | pos+37 {far:.3f} | "
              f"ckpt {steps[C - 2]}->{steps[C - 1]} {ck:.3f} | across sequences {cross:.3f}")
    print()

    # ---- 5. support: how many distinct features ever win ------------------- #
    print("=" * 78)
    print("5. Support of the TopK across the cached batch (last checkpoint)")
    print("=" * 78)
    for li in range(L):
        r = np.abs(np.asarray(a[C - 1, li], dtype=np.float64))
        top = np.argsort(-r, axis=-1)[..., :k].ravel()
        cnt = np.bincount(top, minlength=N)
        used = int((cnt > 0).sum())
        # share of all TopK slots taken by the busiest 5% of features
        order = -np.sort(-cnt)
        top5 = order[: max(1, N // 20)].sum() / cnt.sum()
        print(f"   layer {li}: {used:>5}/{N} features ever in TopK "
              f"({used / N * 100:4.1f}%) | busiest 5% of features take "
              f"{top5 * 100:4.1f}% of slots")


if __name__ == "__main__":
    main()
