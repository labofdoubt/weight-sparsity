import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open(sys.argv[1]))
L = d["layers"]
li = [r["layer"] for r in L]
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))

ax = axes[0][0]
ax.plot(li, [r["topk_mean"] for r in L], "-o", ms=4, label="mean |z| of top-K")
ax.plot(li, [r["topk_med"] for r in L], "-s", ms=4, label="median |z| of top-K")
ax.set_yscale("log"); ax.legend(fontsize=9)
ax.set_title(f"bottleneck top-{d['k']} activations")

ax = axes[0][1]
ax.plot(li, [r["stream_sq"] for r in L], "-o", ms=4, color="#CC3311")
ax.set_yscale("log")
ax.set_title("residual stream mean $x^2$ entering the bottleneck")

ax = axes[1][0]
ax.plot(li, [r["gz_all"] for r in L], "-o", ms=4, label="mean |dL/dz|, all N")
ax.plot(li, [r["gz_act"] for r in L], "-s", ms=4, label="mean |dL/dz|, K active")
ax.set_yscale("log"); ax.legend(fontsize=9)
ax.set_title("gradient to bottleneck raw activations")

ax = axes[1][1]
ax.plot(li, [r["gmlp"] for r in L], "-o", ms=4, color="#117733")
ax.set_yscale("log")
ax.set_title("gradient to MLP hidden activations, mean |dL/dh|")

for ax in axes.ravel():
    ax.grid(alpha=0.3); ax.set_xlabel("layer")
title = sys.argv[3] if len(sys.argv) > 3 else "layer diagnostics"
fig.suptitle(f"{title} @ step {d['step']}  (batch CE {d['ce']:.3f})", fontsize=13)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=160, bbox_inches="tight")
print(sys.argv[2])
