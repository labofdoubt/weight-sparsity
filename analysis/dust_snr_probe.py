"""Per-coordinate SNR of the DUST estimate at saved checkpoints.

    python analysis/dust_snr_probe.py --run-dir /workspace/runs/<run> --out /workspace/analysis/dust_snr/<run> \
        [--steps 0,2,...] [--sequences 4] [--draws K] [--chunk 8] [--device cuda] [--seed 0]

For every checkpoint ``ckpt_step<N>.pt`` of the run (``--steps`` restricts
them): rebuild the model (``load_for_inference``), take ``S`` fixed validation
windows, run the clean forward once with the DUST capture to get every
bottleneck's gate input ``z_l`` (the pre-activation the noise perturbs), and
for each bottleneck draw ``K`` independent perturbations of its clean
Top(K+J) pool with the run's own ``dust_sigma`` (``K`` defaults to
``dust_population / L``, the draws that bottleneck received per micro-batch
in training), rerunning the forward from that gate (``forward_from_gate``,
head in float32) and scoring every draw by the per-token CE.  With the
draw-mean baseline, one draw's estimate at token ``t`` is

    z_t^(i) = -r_t^(i) a_t^(i) / sigma,     r_t^(i) = R~_t - R_t^(i)

and per coordinate ``c`` (streaming sums, no draw stored)

    mu_c   = (1/K) sum_i z_c^(i)
    nu_c^2 = (1/(K-1)) sum_i (z_c^(i) - mu_c)^2
    SNR_K  = sqrt(K) |mu_c| / nu_c                 (0 where nu_c = 0)

plus the per-token vector version ``sqrt(K) ||mu|| / sqrt(sum_c nu_c^2)``.
Output: ``snr_step<N>.npz`` with ``snr`` ``[S, L, T, N_features]`` float16 in
FEATURE order (coordinates outside the pool are 0), ``snr_vec`` ``[S, L, T]``,
``mu_abs`` and ``nu`` (float16, same shape as ``snr``), and a JSON ``meta``.
``--selftest`` checks the streaming formulas against stored draws on a
synthetic loss and exits.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import time
from typing import Callable, Dict, List

import numpy as np
import torch

from wsparse.bottleneck.dust import (dust_capture, make_generator, noise_scale, pool_indices,
                                     token_losses)
from wsparse.data import build_streams
from wsparse.train import load_for_inference
from wsparse.utils import autocast_context


def accumulate_snr(z: torch.Tensor, cand_idx: torch.Tensor, sigma_t: torch.Tensor, draws: int,
                   loss_fn: Callable[[torch.Tensor], torch.Tensor], generator, chunk: int
                   ) -> Dict[str, torch.Tensor]:
    """Per-coordinate mu, nu and SNR_K on the pool, from streaming sums.

    ``z`` ``[S, T, N]`` (detached), ``cand_idx`` ``[S, T, M]``, ``sigma_t``
    ``[S, T, 1]`` float32; ``loss_fn`` maps a perturbed ``[c * S, T, N]`` tensor
    to per-token losses ``[c * S, T]``.  Returns pool-ordered ``mu`` (signed),
    ``nu``, ``snr`` ``[S, T, M]`` and ``snr_vec`` ``[S, T]``.
    """
    z = z.detach()
    S, T, N = z.shape
    M = cand_idx.shape[-1]
    dev, f32 = z.device, torch.float32
    s_a = torch.zeros(S, T, M, dtype=f32, device=dev)
    s_ra = torch.zeros(S, T, M, dtype=f32, device=dev)
    s_a2 = torch.zeros(S, T, M, dtype=f32, device=dev)
    s_ra2 = torch.zeros(S, T, M, dtype=f32, device=dev)
    s_r2a2 = torch.zeros(S, T, M, dtype=f32, device=dev)
    r_sum = torch.zeros(S, T, dtype=f32, device=dev)
    zf = z.to(f32).unsqueeze(0)
    done = 0
    while done < draws:
        c = min(chunk, draws - done)
        a = torch.randn((c, S, T, M), generator=generator, dtype=f32, device=dev)
        noise = torch.zeros((c, S, T, N), dtype=f32, device=dev)
        noise.scatter_(-1, cand_idx.unsqueeze(0).expand(c, S, T, M), sigma_t.unsqueeze(0) * a)
        zp = (zf + noise).to(z.dtype).reshape(c * S, T, N)
        del noise
        r = loss_fn(zp).reshape(c, S, T).to(f32)     # gamma = 0: R = ell
        del zp
        ru = r.unsqueeze(-1)
        s_a += a.sum(0)
        s_ra += (ru * a).sum(0)
        a2 = a * a
        s_a2 += a2.sum(0)
        s_ra2 += (ru * a2).sum(0)
        s_r2a2 += (ru * ru * a2).sum(0)
        r_sum += r.sum(0)
        done += c
        del a, a2, r, ru
    K = float(draws)
    r_bar = (r_sum / K).unsqueeze(-1)
    sig = sigma_t.to(f32)
    mu = (s_ra - r_bar * s_a) / (K * sig)                         # (1/K) sum_i z_i
    z2 = (s_r2a2 - 2.0 * r_bar * s_ra2 + r_bar * r_bar * s_a2) / (sig * sig)   # sum_i z_i^2
    nu2 = (z2 - K * mu * mu).clamp_min(0.0) / max(K - 1.0, 1.0)
    nu = nu2.sqrt()
    tiny = torch.finfo(f32).tiny
    snr = torch.where(nu2 > 0, math.sqrt(K) * mu.abs() / nu.clamp_min(tiny), torch.zeros_like(mu))
    nu_vec = nu2.sum(-1).sqrt()
    snr_vec = torch.where(nu_vec > 0, math.sqrt(K) * mu.norm(dim=-1) / nu_vec.clamp_min(tiny),
                          torch.zeros_like(nu_vec))
    return {"mu": mu, "nu": nu, "snr": snr, "snr_vec": snr_vec}


def selftest() -> None:
    """The streaming formulas against the definition on stored draws."""
    torch.manual_seed(0)
    S, T, N, M, K = 2, 3, 10, 6, 40
    z = torch.randn(S, T, N)
    w = torch.randn(N)
    cand = pool_indices(z, "abs_topk", M)
    sigma_t = noise_scale("absolute", 0.2, z, cand)
    loss_fn = lambda zp: torch.cumsum((zp * w).sum(-1), -1) + 0.3 * (zp ** 2).sum(-1)  # noqa: E731
    out = accumulate_snr(z, cand, sigma_t, K, loss_fn, torch.Generator().manual_seed(1), chunk=7)
    # the same draws by the same generator calls (chunks 7,7,7,7,7,5), scored directly
    gen = torch.Generator().manual_seed(1)
    draws = []
    done = 0
    while done < K:
        c = min(7, K - done)
        draws.append(torch.randn((c, S, T, M), generator=gen))
        done += c
    a = torch.cat(draws, 0)
    noise = torch.zeros(K, S, T, N).scatter_(-1, cand.unsqueeze(0).expand(K, S, T, M),
                                           sigma_t.unsqueeze(0) * a)
    ell = loss_fn((z.unsqueeze(0) + noise).reshape(K * S, T, N)).reshape(K, S, T)
    r = ell.mean(0, keepdim=True) - ell
    zi = -r.unsqueeze(-1) * a / sigma_t.unsqueeze(0)
    mu = zi.mean(0)
    nu = ((zi - mu) ** 2).sum(0).div(K - 1).sqrt()
    snr = math.sqrt(K) * mu.abs() / nu
    snr_vec = math.sqrt(K) * mu.norm(dim=-1) / (nu ** 2).sum(-1).sqrt()
    for name, a_, b_ in (("mu", out["mu"], mu), ("nu", out["nu"], nu), ("snr", out["snr"], snr),
                         ("snr_vec", out["snr_vec"], snr_vec)):
        assert torch.allclose(a_, b_, rtol=1e-3, atol=1e-4), (name, (a_ - b_).abs().max())
    print("selftest ok: streaming mu, nu, SNR_K and the vector SNR match the stored-draw definition")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir")
    ap.add_argument("--out")
    ap.add_argument("--steps", default=None, help="comma-separated checkpoint steps (default all)")
    ap.add_argument("--sequences", type=int, default=4)
    ap.add_argument("--draws", type=int, default=None, help="per bottleneck (default population / L)")
    ap.add_argument("--chunk", type=int, default=8, help="draws per tail forward")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.run_dir or not args.out:
        ap.error("--run-dir and --out are required")
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    ckpts = sorted(glob.glob(os.path.join(args.run_dir, "ckpt_step*.pt")),
                   key=lambda p: int(re.search(r"ckpt_step(\d+)", p).group(1)))
    if args.steps:
        wanted = {int(s) for s in args.steps.split(",")}
        ckpts = [p for p in ckpts if int(re.search(r"ckpt_step(\d+)", p).group(1)) in wanted]
    if not ckpts:
        raise SystemExit(f"no checkpoints in {args.run_dir}")
    for path in ckpts:
        step = int(re.search(r"ckpt_step(\d+)", path).group(1))
        out_path = os.path.join(args.out, f"snr_step{step}.npz")
        if os.path.exists(out_path):
            print(f"[probe] {out_path} exists, skipping")
            continue
        t0 = time.time()
        model, cfg, ctl = load_for_inference(path, device=str(device))
        model.eval()
        ab = cfg.activation_bottleneck
        if ab.surrogate_mode != "dust":
            raise SystemExit(f"{path}: not a dust run")
        if args.data_dir:
            cfg.data.data_dir = args.data_dir
        _, val = build_streams(cfg.data, seed=cfg.train.seed)
        x, y = val.batch(args.sequences, device, deterministic_offset=0)
        layers = list(ctl.layers)
        L = len(layers)
        draws = args.draws or int(ab.dust_population) // L
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        records: List = []
        with torch.no_grad(), dust_capture([m for _, m in layers], records), \
                autocast_context(device, dtype):
            logits, _ = model(x, y)
        ell_clean, valid = token_losses(logits.detach(), y)
        del logits
        S, T = y.shape
        N = int(ab.n_features)
        snr = np.zeros((S, L, T, N), dtype=np.float16)
        mu_abs = np.zeros((S, L, T, N), dtype=np.float16)
        nu = np.zeros((S, L, T, N), dtype=np.float16)
        snr_vec = np.zeros((S, L, T), dtype=np.float32)
        gen = make_generator(device, args.seed + 1000 * step)
        for li, rec in enumerate(records):
            z = rec.z.detach()
            block_index, placement = rec.module._site
            cand = pool_indices(z, ab.selection_mode, rec.module.gate.m)
            sigma_t = noise_scale(ab.dust_sigma_mode, float(ab.dust_sigma), z, cand)

            def loss_fn(zp, _b=block_index, _p=placement):
                c = zp.shape[0] // S
                with torch.no_grad(), autocast_context(device, dtype):
                    lg = model.forward_from_gate(_b, _p, zp, head_dtype=torch.float32)
                ell, _ = token_losses(lg, y.repeat(c, 1))
                return ell

            with torch.no_grad():
                out = accumulate_snr(z, cand, sigma_t, draws, loss_fn, gen, args.chunk)
                full = lambda t: torch.zeros(S, T, N, dtype=t.dtype, device=device).scatter_(  # noqa: E731
                    -1, cand, t)
                snr[:, li] = full(out["snr"]).cpu().numpy().astype(np.float16)
                mu_abs[:, li] = full(out["mu"].abs()).cpu().numpy().astype(np.float16)
                nu[:, li] = full(out["nu"]).cpu().numpy().astype(np.float16)
                snr_vec[:, li] = out["snr_vec"].cpu().numpy()
            print(f"[probe] step {step} layer {li}/{L}: draws {draws}, median SNR_K over valid "
                  f"tokens and pool {float(np.median(snr[:, li][valid.cpu().numpy()][:, cand[0, 0].cpu().numpy()])):.3f}, "
                  f"{time.time() - t0:.0f} s elapsed", flush=True)
        meta = {"run": cfg.train.run_name, "step": step, "population": int(ab.dust_population),
                "draws": draws, "layers": L, "topk_k": int(ab.k), "j": int(ab.j),
                "pool": int(rec.module.gate.m), "n_features": N, "sigma": float(ab.dust_sigma),
                "sigma_mode": ab.dust_sigma_mode, "gamma": float(ab.dust_gamma),
                "target": ab.dust_target, "decouple": bool(cfg.model.decouple),
                "sequences": S, "seq_len": T, "val_offset": 0, "seed": args.seed,
                "clean_ce_tokens_mean": float(ell_clean[valid].mean()),
                "layer_labels": [lab for lab, _ in layers], "checkpoint": os.path.basename(path)}
        # written to a temporary name and renamed: the viewer and the Drive
        # watcher must never see a half-written file
        tmp_path = out_path + ".tmp.npz"
        np.savez_compressed(tmp_path, snr=snr, mu_abs=mu_abs, nu=nu, snr_vec=snr_vec,
                            valid=valid.cpu().numpy(), tokens=x.cpu().numpy(),
                            meta=json.dumps(meta))
        os.replace(tmp_path, out_path)
        print(f"[probe] wrote {out_path} ({os.path.getsize(out_path) / 1e6:.0f} MB) in "
              f"{time.time() - t0:.0f} s", flush=True)
        del model, ctl, records


if __name__ == "__main__":
    main()
