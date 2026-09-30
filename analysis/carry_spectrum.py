"""Singular values of the carry-only Jacobian dx_L/dx_0 (branches off), exact autograd.

Three carries through L fresh Gaussian bottlenecks (MD init statistics, bp scale):
  shift    value shift (fixed lambda*), no norm       x -> B_shift(x)
  pnorm    plain TopK + post-norm (unit RMS)           x -> sqrt(d) B(x)/|B(x)|
  plain    plain TopK, no norm                          x -> B(x)
The code residual's carry with silent blocks is the identity (all s = 1).
Per depth: effective rank (sum s^2)^2/sum s^4, mean and max s^2, and the median
s^2 over the K largest (the Jacobian has rank <= K < d, so the plain median sits
on the null space), for one random input per trial (3 trials averaged).

  python analysis/carry_spectrum.py out.json      (needs a GPU; ~1 min)
"""
import json, math, sys
import torch
from torch.func import jacfwd
dev = "cuda"
d, N, K = 1024, 4096, 512
gD = math.sqrt(d / K)
lam = 1.0440235492202625  # critical_shift(512, 4096)

def layer(x, WE, WD, mode):
    z = WE @ x
    idx = z.abs().topk(K).indices
    zk = z[idx]
    if mode == "shift":
        sig = z.pow(2).mean().sqrt()
        zk = zk.sign() * torch.relu(zk.abs() - lam * sig)
    y = gD * WD[:, idx] @ zk
    if mode == "pnorm":
        y = y * math.sqrt(d) / y.norm()
    return y

res = {}
for mode in ("shift", "pnorm", "plain"):
    for trial in range(3):
        g = torch.Generator(device=dev).manual_seed(trial)
        Ws = [(torch.randn(N, d, device=dev, generator=g) / math.sqrt(d),
               torch.randn(d, N, device=dev, generator=g) / math.sqrt(d)) for _ in range(24)]
        x0 = torch.randn(d, device=dev, generator=g)
        for L in (1, 4, 8, 16, 24):
            def _f(x, Ws=Ws[:L]):
                for WE, WD in Ws:
                    x = layer(x, WE, WD, mode)
                return x
            J = jacfwd(_f)(x0).double()
            s2 = torch.linalg.svdvals(J) ** 2
            med = s2.sort(descending=True).values[:K].median()
            rec = res.setdefault(mode, {}).setdefault(L, [])
            rec.append({"eff_rank": float(s2.sum() ** 2 / (s2 ** 2).sum()),
                        "median_topK_s2": float(med), "mean_s2": float(s2.mean()),
                        "max_s2": float(s2.max())})
for mode, byL in res.items():
    for L, r in sorted(byL.items()):
        print(mode, L, " ".join(f"{k}={sum(x[k] for x in r)/len(r):.3g}" for k in r[0]))
json.dump(res, open(sys.argv[1], "w"))
