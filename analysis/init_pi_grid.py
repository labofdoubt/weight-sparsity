"""Measure the surrogate gain Pi at initialization over a (K, J, T) grid.

Family: rout_rblapsum_kappa (through_rank_kappa, abs_topk, residual_out, MD
initialization).  Pi is defined in docs/vastai-agent-guide.md 9b:

    Pi = ||L_kappa D_z||_F^2,   L_kappa = diag(kappa) - kappa kappa^T / Z

At initialization the weights do not depend on (K, J, T), so one init state
dict serves the whole grid.  The *forward* does depend on K, however: under
``residual_out`` each block replaces the residual stream, so the scores seen
by a deep block depend on how many features the earlier blocks kept.  J and T
enter only the backward kernel.  Hence: one forward per K, reusing one set of
weights, and all (J, T) combinations evaluated from the captured scores.

Usage:
    python analysis/init_pi_grid.py --config configs/mdinit/rbk.yaml \
        --data-dir /workspace/data/tinystories \
        --out /workspace/analysis/init_pi_grid.json
    python scripts/plot_init_pi.py init_pi_grid.json out.pdf [png_prefix]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from wsparse.bottleneck import apply_activation_bottleneck
from wsparse.config import load_config
from wsparse.data import build_streams, load_meta
from wsparse.decouple import md_init_
from wsparse.model import build_model
from wsparse.utils import autocast_context, resolve_device, resolve_dtype, set_seed


M_SP = 4  # half-width of the rank window used for delta, as in
          # analysis/scale_dynamics.py: delta = (s_(K-4) - s_(K+4)) / 8


def boundary_geometry(ss: np.ndarray, K: int, b0: float) -> dict:
    """b and the local rank spacing delta -- both independent of J and T."""
    b = np.maximum(b0, ss[:, K])
    delta = (ss[:, K - 1 - M_SP] - ss[:, K - 1 + M_SP]) / (2.0 * M_SP)
    return {"b_mean": float(b.mean()), "b_med": float(np.median(b)),
            "delta_mean": float(delta.mean()),
            "delta_med": float(np.median(delta)),
            "delta_p10": float(np.quantile(delta, 0.1)),
            "b_over_delta": float((b / np.maximum(delta, 1e-12)).mean())}


def approx_geometry(ss: np.ndarray, K: int, T: float, b0: float) -> dict:
    """Pi_approx = b^2/(4 T delta), the dense-boundary proxy; J-independent."""
    b = np.maximum(b0, ss[:, K])
    delta = np.maximum((ss[:, K - 1 - M_SP] - ss[:, K - 1 + M_SP]) / (2.0 * M_SP),
                       1e-12)
    pa = b ** 2 / (4.0 * T * delta)
    return {"Pi_approx_mean": float(pa.mean()),
            "Pi_approx_med": float(np.median(pa))}


def pi_stats(ss: np.ndarray, K: int, J: int, T: float, b0: float) -> dict:
    """Exact Pi (and n_eff) per token from scores sorted descending."""
    b = np.maximum(b0, ss[:, K])[:, None]
    sc = ss[:, :K + J]
    kap = np.exp(-np.abs(sc - b) / T) / (2.0 * T)
    Z = np.maximum(kap.sum(1, dtype=np.float64), 1e-30)[:, None]
    S2 = (kap.astype(np.float64) ** 2).sum(1)[:, None]
    colj = ((sc * kap).astype(np.float64) ** 2) * (
        (1.0 - kap / Z) ** 2 + (S2 - kap.astype(np.float64) ** 2) / Z ** 2)
    Pi = colj.sum(1)
    n_eff = (Z[:, 0] ** 2) / np.maximum(S2[:, 0], 1e-30)
    return {"Pi_mean": float(Pi.mean()), "Pi_med": float(np.median(Pi)),
            "Pi_p90": float(np.quantile(Pi, 0.9)),
            "n_eff_mean": float(n_eff.mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mdinit/rbk.yaml")
    ap.add_argument("--data-dir", default="/workspace/data/tinystories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", type=int, nargs="*", default=[0, 4, 7])
    ap.add_argument("--k-values", type=int, nargs="*",
                    default=list(range(32, 513, 32)))
    ap.add_argument("--j-values", type=int, nargs="*",
                    default=list(range(32, 513, 32)))
    ap.add_argument("--temps", type=float, nargs="*", default=[1.0, 2.0, 3.0])
    ap.add_argument("--seqs", type=int, default=16, help="sequences in the batch")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--b0", type=float, default=0.0,
                    help="boundary floor; the current default is 0.0")
    ap.add_argument("--set", dest="sets", action="append", default=[],
                    metavar="a.b=c",
                    help="extra config override, repeatable "
                         "(e.g. --set activation_bottleneck.post_norm=true)")
    args = ap.parse_args()

    common = [f"data.data_dir={args.data_dir}", "model.decouple=true",
              "train.run_name=init_pi"] + list(args.sets)
    base = load_config(args.config, list(common))
    base.model.vocab_size = int(load_meta(base.data.data_dir)["vocab_size"])
    device = resolve_device(base.train.device)
    dtype = resolve_dtype(base.train.dtype, device)
    n_features = base.activation_bottleneck.n_features

    def build(k: int, j: int):
        """Fresh model with this gate geometry; weights come from the state dict."""
        set_seed(args.seed)
        cfg = load_config(args.config, list(common) + [
            f"activation_bottleneck.k={k}", f"activation_bottleneck.j={j}"])
        cfg.model.vocab_size = base.model.vocab_size
        m = build_model(cfg.model)
        bctl = apply_activation_bottleneck(m, cfg.activation_bottleneck,
                                           max_steps=cfg.train.max_steps)
        m.to(device)
        md_init_(m, cfg.model.decouple_gains)
        return m, bctl, cfg

    # one init: built at the reference geometry, then reused for every K
    k0 = args.k_values[0]
    ref_model, _, _ = build(k0, min(args.j_values[-1], n_features - k0))
    init_sd = {k: v.detach().clone() for k, v in ref_model.state_dict().items()}
    del ref_model

    # a fixed batch, identical for every configuration
    train_stream, _ = build_streams(base.data, seed=args.seed)
    x, y = train_stream.batch(args.seqs, device, deterministic_offset=777)

    out = {"meta": {"config": args.config, "seed": args.seed,
                    "blocks": args.blocks, "k_values": args.k_values,
                    "j_values": args.j_values, "temps": args.temps,
                    "b0": args.b0, "n_features": n_features,
                    "tokens": int(x.numel()), "d_model": base.model.d_model,
                    "n_layers": base.model.n_layers,
                    "surrogate": "through_rank_kappa",
                    "post_norm": bool(base.activation_bottleneck.post_norm),
                    "overrides": list(args.sets)},
           "grid": {}, "boundary": {}, "approx": {}, "score_scale": {}}

    t0 = time.time()
    for K in args.k_values:
        jmax = max(j for j in args.j_values if K + j <= n_features)
        model, bctl, cfg = build(K, jmax)
        missing = model.load_state_dict(init_sd, strict=True)
        model.train()  # the forward the optimizer would see at step 0

        grab: dict = {}
        handles = []
        for li, (lbl, mod) in enumerate(bctl.layers):
            if li in args.blocks:
                handles.append(mod.gate.register_forward_pre_hook(
                    lambda m, inp, li=li: grab.__setitem__(li, inp[0].detach().float())))
        with torch.no_grad(), autocast_context(device, dtype):
            model(x, y)
        for h in handles:
            h.remove()

        for li in args.blocks:
            z = grab[li].reshape(-1, n_features).cpu().numpy()
            s = np.abs(z).astype(np.float32)
            ss = -np.sort(-s, axis=1)
            out["score_scale"][f"K{K}_blk{li}"] = {
                "rms": float(np.sqrt((s ** 2).mean())),
                "s_K1_mean": float(ss[:, K].mean()),
                "s_1_mean": float(ss[:, 0].mean()),
            }
            # b and delta depend on K only (not on J or T)
            out["boundary"][f"K{K}_blk{li}"] = boundary_geometry(ss, K, args.b0)
            for T in args.temps:
                out["approx"][f"K{K}_T{T:g}_blk{li}"] = approx_geometry(
                    ss, K, T, args.b0)
            for J in args.j_values:
                if K + J > n_features:
                    continue
                for T in args.temps:
                    key = f"K{K}_J{J}_T{T:g}_blk{li}"
                    out["grid"][key] = pi_stats(ss, K, J, T, args.b0)
        del model, bctl, grab
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[init_pi] K={K} done ({time.time() - t0:.0f}s, "
              f"{len(out['grid'])} cells)", flush=True)

    json.dump(out, open(args.out, "w"))
    print(f"[init_pi] wrote {args.out}: {len(out['grid'])} cells")


if __name__ == "__main__":
    main()
