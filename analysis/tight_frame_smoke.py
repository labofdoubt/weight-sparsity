"""Initial forward and backward gain of the bottleneck, per init configuration.

Runs the three configurations of the tight-frame experiment on a tiny MD model
and measures, at initialization, what one bottleneck does to a vector in each
direction:

    R_back = || J^T g ||^2 / || g ||^2      for random output cotangents g
    R_fwd  = || y ||^2 / || x ||^2          for random inputs x

Both are averaged over several random batches, and the backward one uses random
cotangents rather than a real loss gradient on purpose: it measures the module's
Jacobian, not the trajectory the model happens to be on.

    python analysis/tight_frame_smoke.py [--d-model 64] [--n-features 256]
                                         [--batches 8] [--tokens 64]

Expected, and what to read the table against:

* orthogonal, g_D = 1              backward gain near K_eff / d_model
* orthogonal, backward_preserving  backward gain near 1, forward amplified
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wsparse.bottleneck import apply_activation_bottleneck  # noqa: E402
from wsparse.bottleneck.module import effective_backward_support  # noqa: E402
from wsparse.config import ActivationBottleneckConfig, ModelConfig  # noqa: E402
from wsparse.decouple import md_init_  # noqa: E402
from wsparse.model import build_model  # noqa: E402

CONFIGS = (
    ("A  standard", "standard", "none"),
    ("B  orthogonal", "orthogonal", "none"),
    ("C  orthogonal + bp", "orthogonal", "backward_preserving"),
)


def build(bn_init, dec_scale, surrogate, k, j, d_model, n_features, seed=0):
    model_cfg = ModelConfig(
        vocab_size=97, max_seq_len=64, n_layers=2, d_model=d_model, n_heads=4,
        pos_encoding="rope", decouple=True, logit_scale="none", bias=False,
        bottleneck_init=bn_init, bottleneck_decoder_scale=dec_scale)
    bn_cfg = ActivationBottleneckConfig(
        enabled=True, layers="all", placement="residual_out",
        n_features=n_features, k=k, j=j, surrogate_mode=surrogate,
        temperature=1.0, bias=False,
        rblapsum_boundary_grad_mode=(None if surrogate == "hard"
                                     else "through_rank_kappa"))
    torch.manual_seed(seed)
    model = build_model(model_cfg)
    ctl = apply_activation_bottleneck(model, bn_cfg, max_steps=10)
    md_init_(model, model_cfg.decouple_gains)
    return model, ctl, bn_cfg


def gains(mod, d_model, batches, tokens, seed=1234):
    """``(R_fwd, R_back)`` for one bottleneck module, averaged over batches."""
    g = torch.Generator().manual_seed(seed)
    mod.train()  # the surrogate path is off in eval / under no_grad
    fwd, back = [], []
    for _ in range(batches):
        x = torch.randn(tokens, d_model, generator=g, requires_grad=True)
        y = mod(x)
        cot = torch.randn_like(y)
        (gx,) = torch.autograd.grad(y, x, grad_outputs=cot)
        fwd.append(float((y.detach() ** 2).sum() / (x.detach() ** 2).sum()))
        back.append(float((gx ** 2).sum() / (cot ** 2).sum()))
    return sum(fwd) / len(fwd), sum(back) / len(back)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-features", type=int, default=256)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--j", type=int, default=48)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=64)
    args = ap.parse_args()

    d, n = args.d_model, args.n_features
    print(f"d_model={d}  d_bottleneck={n}  K={args.k}  J={args.j}  "
          f"{args.batches} batches x {args.tokens} tokens, random cotangents")
    for surrogate in ("hard", "rblapsum"):
        k_eff = effective_backward_support(
            ActivationBottleneckConfig(enabled=True, n_features=n, k=args.k,
                                       j=args.j, surrogate_mode=surrogate,
                                       temperature=1.0))
        print(f"\n--- surrogate={surrogate}  K_eff={k_eff:g}  "
              f"K_eff/d_model={k_eff / d:.4f}")
        print(f"{'configuration':22s} {'g_D':>7s} {'R_fwd':>9s} {'R_back':>9s} "
              f"{'R_back/(K_eff/d)':>17s}")
        for label, bn_init, dec_scale in CONFIGS:
            model, ctl, _ = build(bn_init, dec_scale, surrogate, args.k,
                                  args.j, d, n)
            mod = ctl.layers[0][1]
            r_fwd, r_back = gains(mod, d, args.batches, args.tokens)
            print(f"{label:22s} {mod.decoder_scale:7.4f} {r_fwd:9.4f} "
                  f"{r_back:9.4f} {r_back / (k_eff / d):17.4f}")


if __name__ == "__main__":
    main()
