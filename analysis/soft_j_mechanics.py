"""Why j doesn't help soft top(k+j): coverage, conservation, selection agreement.

All from existing probe snapshots (real training scores + real gradients).
"""
import json
import sys
sys.path.insert(0, "/workspace/weight-sparsity/src")
import numpy as np
import torch
from wsparse.bottleneck.lapsum import lapsum_barrier_sorted, laplace_pdf

P = "/workspace/analysis/probe"
K = 32

def cells(name, ci, n=1500, seed=0):
    S = np.load(f"{P}/{name}.score.npy", mmap_mode="r")
    a = np.asarray(S[ci], dtype=np.float64)          # (L,B,T,N)
    r = np.abs(a).reshape(-1, a.shape[-1])
    rng = np.random.default_rng(seed)
    idx = rng.choice(r.shape[0], size=min(n, r.shape[0]), replace=False)
    return np.sort(r[idx], axis=-1)[:, ::-1].copy()  # sorted desc

# ---- A. counterfactual coverage & conservation over (t, pool) -------------- #
name = "probe_uk_rout_soft_k32_j32_md_abs"
meta = json.load(open(f"{P}/{name}.json"))
steps = meta["steps"]
ci = min(range(len(steps)), key=lambda i: abs(steps[i] - 1000))
r = torch.from_numpy(cells(name, ci))
print(f"== A: conservation/coverage at step {steps[ci]} ({r.shape[0]} tokens)")
print(f"{'t':>5} {'pool':>5} | {'sum kappa(inact)':>16} {'N_eff(inact)':>12} {'p_gap(k..k+j)':>13}")
for t_val in (0.25, 1.0, 2.0, 4.0, 8.0):
    for pool in (64, 128, 512, 1536):
        cand = r[:, :pool]
        t = torch.full((r.shape[0],), t_val, dtype=torch.float64)
        b = lapsum_barrier_sorted(cand, K, t)
        kap = laplace_pdf((cand - b[:, None]) / t[:, None]) / t[:, None]
        ki = kap[:, K:]                                  # inactive candidates
        tot = ki.sum(-1)
        neff = tot.pow(2) / ki.pow(2).sum(-1).clamp_min(1e-30)
        # soft mass sitting on the inactive band (how much "selection pressure")
        from wsparse.bottleneck.lapsum import laplace_cdf
        p = laplace_cdf((cand - b[:, None]) / t[:, None])
        print(f"{t_val:>5} {pool:>5} | {float(tot.mean()):>16.4f} {float(neff.mean()):>12.1f} "
              f"{float(p[:, K:].sum(-1).mean()):>13.3f}")
# density at the boundary (kernel-free reference): +-0.25 window around b at t=1, pool 1536
cand = r
t = torch.ones(r.shape[0], dtype=torch.float64)
b = lapsum_barrier_sorted(cand, K, t)
rho = ((cand - b[:, None]).abs() < 0.25).double().sum(-1) / 0.5
print(f"score density near boundary rho(b) ~ {float(rho.mean()):.3f} per unit score")

# ---- B. does j change WHICH support is learned? (same seed, j32 vs j480) --- #
print("\n== B: selection agreement across j at matched steps (same seed/init)")
for st in (500, 1000):
    sets = {}
    for jn in (32, 96, 480):
        nm = f"probe_uk_rout_soft_k32_j{jn}_md_abs"
        m = json.load(open(f"{P}/{nm}.json"))
        cj = min(range(len(m["steps"])), key=lambda i: abs(m["steps"][i] - st))
        S = np.load(f"{P}/{nm}.score.npy", mmap_mode="r")
        a = np.abs(np.asarray(S[cj], dtype=np.float32)).reshape(-1, 1536)
        sets[jn] = np.argsort(-a, axis=-1)[:, :K]
    def iou(x, y):
        inter = np.array([len(np.intersect1d(x[i], y[i], assume_unique=True))
                          for i in range(0, x.shape[0], 7)])
        return inter.mean() / K
    print(f" step~{st}: IoU(top32: j32 vs j96) = {iou(sets[32], sets[96]):.3f}   "
          f"(j32 vs j480) = {iou(sets[32], sets[480]):.3f}")

# ---- C. REAL measured inactive-band gradients vs j ------------------------- #
print("\n== C: measured |g_z| on the inactive band, per run (step ~1000)")
print(f"{'run':>8} | {'sum|g| inact(pool)':>18} {'mean|g| per cand':>16}")
for jn in (32, 96, 480):
    nm = f"probe_uk_rout_soft_k32_j{jn}_md_abs"
    m = json.load(open(f"{P}/{nm}.json"))
    cj = min(range(len(m["steps"])), key=lambda i: abs(m["steps"][i] - 1000))
    S = np.load(f"{P}/{nm}.score.npy", mmap_mode="r")
    G = np.load(f"{P}/{nm}.g_z.npy", mmap_mode="r")
    a = np.abs(np.asarray(S[cj], dtype=np.float64)).reshape(-1, 1536)
    g = np.abs(np.asarray(G[cj], dtype=np.float64)).reshape(-1, 1536)
    order = np.argsort(-a, axis=-1)
    gs = np.take_along_axis(g, order, -1)
    band = gs[:, K:K + jn]
    print(f"{'j'+str(jn):>8} | {band.sum(-1).mean():>18.3e} {band.mean():>16.3e}")
