"""The amplification diagnostics from docs/activation-amplification.md, reusable.

These started as a dozen throwaway scripts while chasing why activations blow up
in stream-placed bottlenecks. Collected here because each one answers a question
that will come up again, and because two of them overturned an explanation that
looked solid -- worth being able to re-run rather than re-derive.

    python analysis/diagnose_amplification.py tail      --scores-dir /workspace/analysis/scores
    python analysis/diagnose_amplification.py onset     --probe-dir  /workspace/analysis/probe
    python analysis/diagnose_amplification.py depth     --probe-dir  /workspace/analysis/probe
    python analysis/diagnose_amplification.py grad-mult --probe-dir  /workspace/analysis/probe
    python analysis/diagnose_amplification.py invariant --probe-dir  /workspace/analysis/probe
    python analysis/diagnose_amplification.py t-vs-j    --probe-dir  /workspace/analysis/probe
    python analysis/diagnose_amplification.py rows      --ckpt-dir   /workspace/ckpt --run <run>
    python analysis/diagnose_amplification.py seed      --ckpt-dir   /workspace/ckpt --run <run>

What each answers:

``tail``       median vs max score per token across a checkpoint ladder, and how
               many tokens sit above 100x the median. This is what separated
               "the whole run is inflated" (it is not) from "a few tokens have
               run away" (they have).
``onset``      the same over the probe's 0..1000 window, to date the onset.
``depth``      max/median by block -- is the tail born at L0 or made later?
``grad-mult``  ``g_z / g_ztilde`` on the TopK. Exactly 1 for a hard gate; ~1.65
               for a lapsum one. Shows the surrogate's extra term exists -- but
               NOT that it drives growth (see ``invariant``).
``invariant``  ``|g_z| * |z|`` over training. Roughly conserved in *every* run,
               so it does not discriminate healthy from runaway: the reason the
               "gradient proportional to z" story was wrong.
``t-vs-j``     ``t = std(top-(k+j))`` relative to the top score, as a function of
               j, on fixed score vectors. Pure geometry, no training.
``rows``       per-row / per-column norms of the bottleneck weights. Rules out
               "a few blown-up rows hiding inside ||W||F" -- they are not there.
``seed``       decomposes a block's residual stream into emb / attn / mlp at the
               worst token. This is what found the MLP as the source.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _eff_j(meta) -> int:
    """j is inert under a hard gate: those ranks get exactly zero gradient."""
    return 0 if meta.get("surrogate_mode") == "hard" else int(meta["j"])


def _load(d, name, arr="score"):
    base = os.path.join(d, name)
    meta = json.load(open(f"{base}.json"))
    p = f"{base}.{arr}.npy" if os.path.exists(f"{base}.{arr}.npy") else f"{base}.npy"
    return np.load(p, mmap_mode="r"), meta


def _names(d, suffix=".json"):
    return sorted(os.path.basename(p)[: -len(suffix)]
                  for p in glob.glob(os.path.join(d, f"*{suffix}")))


# --------------------------------------------------------------------------- #
def cmd_tail(a):
    for run in _names(a.scores_dir):
        arr, m = _load(a.scores_dir, run)
        C, L, B, T, N = arr.shape
        print(f"### {run}  (k={m['k']} j={_eff_j(m)} {m['surrogate_mode']})")
        print(f"{'step':>6} {'CE':>7} | {'L7 median':>11} {'L7 p99':>11} {'L7 MAX':>11} "
              f"| {'max/med':>10} | {'#>100x med':>10}")
        for ci in range(C):
            pc = np.abs(np.asarray(arr[ci, L - 1], dtype=np.float64)).max(-1).ravel()
            med = np.median(pc)
            print(f"{m['steps'][ci]:>6} {m['batch_ce'][ci]:>7.3f} | {med:11.4g} "
                  f"{np.percentile(pc, 99):11.4g} {pc.max():11.4g} | {pc.max()/med:10.4g} "
                  f"| {int((pc > 100*med).sum()):>10}")
        print()


def cmd_onset(a):
    for run in _names(a.probe_dir):
        arr, m = _load(a.probe_dir, run)
        P, L, B, T, N = arr.shape
        print(f"### {run}  (k={m['k']} j={_eff_j(m)})")
        print(f"{'step':>6} {'CE':>7} | {'median':>10} {'MAX':>11} {'max/med':>10} "
              f"| {'#>100x med':>10}")
        for p in range(0, P, max(1, P // 10)):
            pc = np.abs(np.asarray(arr[p, L - 1], dtype=np.float64)).max(-1).ravel()
            med = np.median(pc)
            print(f"{m['steps'][p]:>6} {m['batch_ce'][p]:>7.3f} | {med:10.4g} "
                  f"{pc.max():11.4g} {pc.max()/med:10.4g} | {int((pc > 100*med).sum()):>10}")
        print()


def cmd_depth(a):
    print("max/median of max|score| per token, BY BLOCK (last probe)")
    for run in _names(a.probe_dir):
        arr, m = _load(a.probe_dir, run)
        P, L = arr.shape[0], arr.shape[1]
        row = []
        for l in range(L):
            pc = np.abs(np.asarray(arr[P - 1, l], dtype=np.float64)).max(-1).ravel()
            row.append(pc.max() / np.median(pc))
        print(f"{run:>34} | " + " ".join(f"{v:9.3g}" for v in row))


def cmd_grad_mult(a):
    print("g_z / g_ztilde on the TopK.  Exactly 1 <=> no multiplicative term.")
    print(f"{'run':>34} {'j':>4} | {'median':>10} {'p1':>9} {'p99':>9} "
          f"| {'frac |r-1|>0.1':>15}")
    for run in _names(a.probe_dir):
        sc, m = _load(a.probe_dir, run, "score")
        gz, _ = _load(a.probe_dir, run, "g_z")
        gt, _ = _load(a.probe_dir, run, "g_ztilde")
        k, P, L = m["k"], sc.shape[0], sc.shape[1]
        li = L // 2
        r = np.abs(np.asarray(sc[P - 1, li], dtype=np.float64))
        top = np.argsort(-r, axis=-1)[..., :k]
        A = np.take_along_axis(np.asarray(gt[P - 1, li], dtype=np.float64), top, -1)
        Z = np.take_along_axis(np.asarray(gz[P - 1, li], dtype=np.float64), top, -1)
        ok = np.abs(A) > 0
        q = Z[ok] / A[ok]
        print(f"{run:>34} {_eff_j(m):>4} | {np.median(q):10.6f} "
              f"{np.percentile(q,1):9.4f} {np.percentile(q,99):9.4f} "
              f"| {np.mean(np.abs(q-1) > 0.1):15.4f}")


def cmd_invariant(a):
    print("|g_z|*|z| at the worst token.  Conserved in every run, so it is NOT")
    print("what separates a healthy run from a runaway one.")
    for run in _names(a.probe_dir):
        sc, m = _load(a.probe_dir, run, "score")
        gz, _ = _load(a.probe_dir, run, "g_z")
        P, L = sc.shape[0], sc.shape[1]
        ser, rate = [], []
        prev = None
        for p in range(0, P, 5):
            zl = np.abs(np.asarray(sc[p, L - 1], dtype=np.float64))
            gl = np.abs(np.asarray(gz[p, L - 1], dtype=np.float64))
            pc = zl.max(-1)
            wi = np.unravel_index(int(np.argmax(pc)), pc.shape)
            z, g = zl[wi].max(), gl[wi].max()
            ser.append(z * g)
            if prev:
                rate.append((z / prev) ** (1.0 / (5 * m["probe_every"])))
            prev = z
        ser = np.array(ser)
        print(f"  {run:>34}: g*z early {ser[2:6].mean():.4g} late {ser[-4:].mean():.4g} "
              f"ratio {ser[-4:].mean()/ser[2:6].mean():.3f} | median growth/step "
              f"{np.median(rate):.5f}")


def cmd_t_vs_j(a):
    sc, m = _load(a.probe_dir, a.run or _names(a.probe_dir)[0], "score")
    k = m["k"]
    P = sc.shape[0]
    r = np.abs(np.asarray(sc[P // 2, sc.shape[1] // 2], dtype=np.float64)).reshape(-1, m["n_features"])
    srt = -np.sort(-r, axis=-1)
    print(f"t = std(top-(k+j)) on fixed scores from {m['run']}, k={k}: pure geometry")
    print(f"{'j':>5} {'k+j':>5} | {'t':>12} {'t/r[0]':>9} | {'kappa~0.5/t':>12} {'rel':>7}")
    base = None
    for j in (0, 16, 32, 64, 128, 256, 512):
        if k + j > srt.shape[1]:
            continue
        t = float(np.median(srt[:, : k + j].std(-1, ddof=1)))
        kap = 0.5 / t
        base = base or (kap if j == 32 else None)
        print(f"{j:>5} {k+j:>5} | {t:12.5g} {t/np.median(srt[:,0]):9.4f} "
              f"| {kap:12.5g} {(kap/base if base else float('nan')):7.3f}")


def cmd_rows(a):
    import torch
    from wsparse.bottleneck.controller import _PLACEMENT_ATTR, parse_placements
    from wsparse.train import load_for_inference
    paths = sorted(glob.glob(os.path.join(a.ckpt_dir, a.run, "ckpt_step*.pt")),
                   key=lambda p: int(re.search(r"step(\d+)", p).group(1)))
    print(f"### {a.run}: bottleneck weight rows/cols at block {a.layer}")
    print(f"{'step':>6} | {'||Win||F':>9} {'row med':>9} {'row MAX':>9} {'max/med':>8} "
          f"| {'||Wout||F':>10} {'col med':>9} {'col MAX':>9} {'max/med':>8} | {'#rows>10x':>9}")
    for p in paths[:: max(1, len(paths) // 4)] + [paths[-1]]:
        step = int(re.search(r"step(\d+)", p).group(1))
        model, cfg, _ = load_for_inference(p, device="cpu")
        nm = parse_placements(cfg.activation_bottleneck.placement)[0]
        mod = getattr(model.blocks[a.layer], _PLACEMENT_ATTR[nm])
        Wi, Wo = mod.in_proj.weight.detach(), mod.out_proj.weight.detach()
        rn, cn = Wi.norm(dim=1), Wo.norm(dim=0)
        print(f"{step:>6} | {Wi.norm():9.3f} {rn.median():9.4g} {rn.max():9.4g} "
              f"{rn.max()/rn.median():8.3g} | {Wo.norm():10.3f} {cn.median():9.4g} "
              f"{cn.max():9.4g} {cn.max()/cn.median():8.3g} "
              f"| {int((rn > 10*rn.median()).sum()):>9}")
        del model


def cmd_seed(a):
    import torch
    from wsparse.bottleneck.controller import _PLACEMENT_ATTR, parse_placements
    from wsparse.data import TokenStream
    from wsparse.train import load_for_inference
    ck = os.path.join(a.ckpt_dir, a.run, f"ckpt_step{a.step}.pt")
    model, cfg, _ = load_for_inference(ck, device=a.device)
    model.eval()
    nm = parse_placements(cfg.activation_bottleneck.placement)[0]
    cap, hs = {}, []
    for li, b in enumerate(model.blocks):
        hs += [b.mlp.register_forward_hook(
                   lambda m, i, o, li=li: cap.__setitem__(("mlp", li), o.detach().float())),
               b.attn.register_forward_hook(
                   lambda m, i, o, li=li: cap.__setitem__(("attn", li), o.detach().float())),
               getattr(b, _PLACEMENT_ATTR[nm]).register_forward_pre_hook(
                   lambda m, i, li=li: cap.__setitem__(("x", li), i[0].detach().float()))]
    stream = TokenStream(os.path.join(a.data_dir, "val.bin"), int(cfg.data.seq_len), seed=0)
    xi, yi = stream.batch(a.batch, torch.device(a.device), deterministic_offset=0)
    with torch.no_grad():
        model(xi, yi)
    for h in hs:
        h.remove()
    T = a.positions
    L = len(model.blocks)
    xl = cap[("x", L - 1)][:, :T].norm(dim=-1)
    wi = np.unravel_index(int(torch.argmax(xl).cpu()), xl.shape)
    print(f"### {a.run} @ step {a.step}: worst token at the last block is "
          f"seq {wi[0]} pos {wi[1]}")
    print(f"  If mlp/x = 1 at some block and 0 after, the MLP seeded the spike")
    print(f"  and the bottleneck only compounded it.")
    print(f"  {'L':>2} {'||x||':>12} {'||mlp||':>11} {'||attn||':>10} {'mlp/x':>7} "
          f"{'x_l/x_l-1':>10}")
    prev = None
    for li in range(L):
        xa = float(cap[("x", li)][wi].norm())
        ma = float(cap[("mlp", li)][wi].norm())
        aa = float(cap[("attn", li)][wi].norm())
        print(f"  {li:>2} {xa:12.4g} {ma:11.4g} {aa:10.4g} {ma/max(xa,1e-30):7.3f} "
              + ("" if prev is None else f"{xa/prev:10.3f}"))
        prev = xa


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("tail", "onset", "depth", "grad-mult", "invariant", "t-vs-j",
                 "rows", "seed"):
        p = sub.add_parser(name)
        p.add_argument("--scores-dir", default="/workspace/analysis/scores")
        p.add_argument("--probe-dir", default="/workspace/analysis/probe")
        p.add_argument("--ckpt-dir", default="/workspace/ckpt")
        p.add_argument("--data-dir", default="/workspace/data/tinystories")
        p.add_argument("--run", default=None)
        p.add_argument("--layer", type=int, default=7)
        p.add_argument("--step", type=int, default=2000)
        p.add_argument("--batch", type=int, default=4)
        p.add_argument("--positions", type=int, default=64)
        p.add_argument("--device", default="cuda")
    a = ap.parse_args()
    a.scores_dir = a.scores_dir
    a.probe_dir = a.probe_dir
    {"tail": cmd_tail, "onset": cmd_onset, "depth": cmd_depth,
     "grad-mult": cmd_grad_mult, "invariant": cmd_invariant, "t-vs-j": cmd_t_vs_j,
     "rows": cmd_rows, "seed": cmd_seed}[a.cmd](a)


if __name__ == "__main__":
    main()
