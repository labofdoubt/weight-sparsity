"""Per-coordinate signal-to-noise ratio of the population-averaged DUST
estimate at a run's checkpoints.

    python -m toy_mnist.snr_probe --run runs_mnist/snr_sweep/dust_k4_p512 \
        --data-dir data/mnist [--population 512] [--out <run>/snr.npz]

One fixed batch ``(x, y)`` of ``B`` training images (a seeded draw, the same
for every run and checkpoint) and, per checkpoint, ``P`` independent draws
that perturb every coordinate of the gate input ``u = exp(g) * x``::

    u'^(i) = u + sigma a^(i),     a^(i) ~ N(0, I)                      [B, N]
    R_t^(i) = CE of the tail at u'^(i)_t,   R~_t = (1/P) sum_i R_t^(i),
    r_t^(i) = R~_t - R_t^(i)
    z_{t,n}^(i) = -r_t^(i) a_{t,n}^(i) / sigma_t        the reward-weighted noise

The sample average over the batch gives one value per draw and coordinate,
in two weightings::

    z_n^(i)   = (1/B) sum_t z_{t,n}^(i)                      "u": dL/du_n, the
                                                             perturbed vector
    z_n^(i,g) = (1/B) sum_t z_{t,n}^(i) exp(g_n) x_{t,n}     "g": dL/dg_n, what
                                                             the optimizer gets

and for each weighting, per coordinate ``n``::

    mu_n  = (1/P) sum_i z_n^(i)
    nu_n^2 = (1/(P - 1)) sum_i (z_n^(i) - mu_n)^2
    SNR_n = sqrt(P) |mu_n| / nu_n          (the SNR of the population mean)

``P * B`` draws are never stored: pass 1 runs the tails and keeps the
``[P, B]`` losses (for the baseline), pass 2 regenerates the same noise from
the same generator state and accumulates the sums and sums of squares of
``z``.  The tails run as in training (``eval`` mode, TF32 on CUDA).  A
coordinate with ``nu_n = 0`` (every draw gives 0, e.g. a pixel that is 0 in
every sample of the batch under the "g" weighting) gets ``SNR = nan``.

Output ``snr.npz``: ``steps`` [C] (completed optimizer steps of each
checkpoint; under the [1, 1] schedule an odd N is the state the DUST step
N + 1 acts on, 0 the initial state), ``snr_u``, ``mu_u``, ``nu_u``, ``snr_g``,
``mu_g``, ``nu_g`` [C, N], ``ce_clean``, ``ce_perturbed`` [C], and the
settings (``k``, ``population``, ``sigma``, ``batch_size``, ``batch_seed``,
``noise_seed``, ``batch_indices``).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import N_PIXELS, load_config
from .data import MNISTData
from .dust import TAIL_TF32, make_generator, noise_scale
from .model import ToyMLP
from .train import resolve_device


def fixed_batch(data: MNISTData, batch_size: int, seed: int) -> torch.Tensor:
    """Indices of a seeded batch of training images (CPU generator)."""
    gen = torch.Generator().manual_seed(int(seed))
    return torch.randperm(data.n_train, generator=gen)[:batch_size].to(data.device)


def list_checkpoints(run_dir: str, max_steps: int) -> List[Tuple[int, str]]:
    """``(steps_completed, path)`` for every ckpt_step*.pt and final.pt, sorted."""
    out = []
    for path in glob.glob(os.path.join(run_dir, "ckpt_step*.pt")):
        m = re.match(r"ckpt_step(\d+)\.pt$", os.path.basename(path))
        if m:
            out.append((int(m.group(1)), path))
    final = os.path.join(run_dir, "final.pt")
    if os.path.exists(final) and all(n != max_steps for n, _ in out):
        out.append((int(max_steps), final))
    return sorted(out)


@torch.no_grad()
def snr_at_state(model: ToyMLP, x: torch.Tensor, y: torch.Tensor, population: int,
                 sigma: float, sigma_mode: str, noise_seed: int, chunk: int = 512
                 ) -> Dict[str, torch.Tensor]:
    """The arrays of the module docstring for the model's current parameters."""
    device = x.device
    f32 = torch.float32
    was_training = model.training
    model.eval()
    tf32_prior = torch.backends.cuda.matmul.allow_tf32
    if TAIL_TF32 and x.is_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
    try:
        u = model.gate_input(x).to(f32)
        B, N = u.shape
        P = int(population)
        if P < 2:
            raise ValueError("population must be >= 2")
        chunk = max(1, min(int(chunk), P))
        ident = torch.arange(N, device=device).expand(B, N)
        sigma_t = noise_scale(sigma_mode, sigma, u, ident)  # [B, 1]
        weight_g = (torch.exp(model.rescale.log_scale.detach().to(f32)) * x.to(f32))  # [B, N]
        logits_clean = model.forward_from_gate(u)
        ce_clean = F.cross_entropy(logits_clean.float(), y)

        # pass 1: the losses of every draw
        gen = make_generator(device, noise_seed)
        losses = torch.empty(P, B, dtype=f32, device=device)
        done = 0
        while done < P:
            c = min(chunk, P - done)
            a = torch.randn((c, B, N), generator=gen, dtype=f32, device=device)
            up = (u.unsqueeze(0) + sigma_t.unsqueeze(0) * a).reshape(c * B, N)
            lg = model.forward_from_gate(up)
            losses[done:done + c] = F.cross_entropy(lg.float(), y.repeat(c),
                                                    reduction="none").reshape(c, B)
            done += c
            del a, up, lg
        r_bar = losses.mean(0, keepdim=True)         # [1, B]
        d = (losses - r_bar) / sigma_t.T             # [P, B]: (R - R~) / sigma = -r / sigma

        # pass 2: the same noise again, the per-draw sample-averaged z and its square
        gen = make_generator(device, noise_seed)
        s1_u = torch.zeros(N, dtype=f32, device=device)
        s2_u = torch.zeros(N, dtype=f32, device=device)
        s1_g = torch.zeros(N, dtype=f32, device=device)
        s2_g = torch.zeros(N, dtype=f32, device=device)
        done = 0
        while done < P:
            c = min(chunk, P - done)
            a = torch.randn((c, B, N), generator=gen, dtype=f32, device=device)
            dc = d[done:done + c]                                     # [c, B]
            z_u = torch.einsum("cb,cbn->cn", dc, a) / B               # [c, N]
            z_g = torch.einsum("cb,cbn,bn->cn", dc, a, weight_g) / B
            s1_u += z_u.sum(0)
            s2_u += (z_u * z_u).sum(0)
            s1_g += z_g.sum(0)
            s2_g += (z_g * z_g).sum(0)
            done += c
            del a, z_u, z_g

        def finish(s1: torch.Tensor, s2: torch.Tensor):
            mu = s1 / P
            var = ((s2 - P * mu * mu) / (P - 1)).clamp_min(0.0)
            nu = var.sqrt()
            snr = torch.where(nu > 0, P ** 0.5 * mu.abs() / nu, torch.full_like(nu, float("nan")))
            return mu, nu, snr

        mu_u, nu_u, snr_u = finish(s1_u, s2_u)
        mu_g, nu_g, snr_g = finish(s1_g, s2_g)
        return {"mu_u": mu_u, "nu_u": nu_u, "snr_u": snr_u, "mu_g": mu_g, "nu_g": nu_g,
                "snr_g": snr_g, "ce_clean": ce_clean.detach().float(),
                "ce_perturbed": losses.mean()}
    finally:
        model.train(was_training)
        torch.backends.cuda.matmul.allow_tf32 = tf32_prior


def main(argv=None) -> str:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="run directory with config.json and ckpt_step*.pt")
    ap.add_argument("--data-dir", default=None, help="MNIST directory (default: the run's)")
    ap.add_argument("--population", type=int, default=None,
                    help="draws per checkpoint (default: the run's dust_population)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--batch-seed", type=int, default=0)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="default: <run>/snr.npz")
    args = ap.parse_args(argv)

    with open(os.path.join(args.run, "config.json")) as f:
        cfg_dict = json.load(f)
    if args.data_dir:
        cfg_dict.setdefault("data", {})["data_dir"] = args.data_dir
    cfg = load_config(values=cfg_dict)
    if not (cfg.rescale.enabled and cfg.topk.enabled):
        raise ValueError("the probe needs the rescaling and the gate (the DUST setting)")
    device = resolve_device(args.device)
    data = MNISTData.from_config(cfg.data, device, cfg.train.seed)
    idx = fixed_batch(data, args.batch_size, args.batch_seed)
    x, y = data.train_x[idx], data.train_y[idx]
    population = int(args.population or cfg.topk.dust_population)
    model = ToyMLP(cfg).to(device)
    ckpts = list_checkpoints(args.run, cfg.train.max_steps)
    if not ckpts:
        raise FileNotFoundError(f"no ckpt_step*.pt or final.pt under {args.run}")
    print(f"[snr] {args.run}: {len(ckpts)} checkpoints, population {population}, "
          f"batch {args.batch_size} (seed {args.batch_seed}), sigma {cfg.topk.dust_sigma:g} "
          f"({cfg.topk.dust_sigma_mode}), k {cfg.topk.k}")
    rows: Dict[str, List] = {k: [] for k in ("mu_u", "nu_u", "snr_u", "mu_g", "nu_g", "snr_g",
                                             "ce_clean", "ce_perturbed")}
    steps = []
    t0 = time.time()
    for n, path in ckpts:
        payload = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        out = snr_at_state(model, x, y, population, cfg.topk.dust_sigma, cfg.topk.dust_sigma_mode,
                           args.noise_seed, args.chunk)
        steps.append(n)
        for k in rows:
            rows[k].append(out[k].cpu().numpy())
        s = out["snr_u"]
        print(f"[snr] step {n:5d}: ce {float(out['ce_clean']):.4f} gap "
              f"{float(out['ce_perturbed'] - out['ce_clean']):+.4f} | SNR(u) median "
              f"{float(torch.nanmedian(s)):.3f} max {float(torch.nan_to_num(s, nan=0).max()):.2f}"
              f" | SNR(g) median {float(torch.nanmedian(out['snr_g'])):.3f}")
    out_path = args.out or os.path.join(args.run, "snr.npz")
    np.savez(out_path, steps=np.array(steps), **{k: np.stack(v) for k, v in rows.items()},
             k=int(cfg.topk.k), population=population, sigma=float(cfg.topk.dust_sigma),
             sigma_mode=str(cfg.topk.dust_sigma_mode), batch_size=int(args.batch_size),
             batch_seed=int(args.batch_seed), noise_seed=int(args.noise_seed),
             batch_indices=idx.cpu().numpy(), run_name=str(cfg.train.run_name),
             hidden=int(cfg.model.hidden))
    print(f"[snr] wrote {out_path} in {time.time() - t0:.1f} s")
    return out_path


if __name__ == "__main__":
    main()
