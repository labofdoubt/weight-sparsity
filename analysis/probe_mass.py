"""Surrogate mass and per-candidate p_i over training, for RBLapSum runs.

The LapSum soft assignment is

    p_i = F((s_i - b) / T),        F = Laplace CDF  (wsparse.bottleneck.lapsum)

and ``sum_i p_i`` is the quantity LapSum's barrier solve drives to ``K``.
RBLapSum does not solve for the barrier: it pins ``b`` to the rank boundary,
``b = max(b0, s_(K+1))``, so the mass is free to drift.  This probe measures
whether it does, and how the profile ``p_i`` over the candidate pool evolves.

Two modes:

  probe   re-run training from a run's own config with the same seed and
          measure every ``--every`` steps through ``train()``'s ``on_step``
          hook, saving no checkpoints.  This is the only way to see the first
          couple of thousand steps of a run that died before its first
          checkpoint.  Follows the probe invariants of
          analysis/probe_early_training.py: the measurement is forward-only
          under ``no_grad``, and the gate buffers and RNG state it would
          perturb are snapshotted and restored.

  ckpt    measure saved checkpoints instead (no training).

Usage:
    python analysis/probe_mass.py probe --run-config cfg.json --stop-step 2500 \
        --every 50 --out mass_<run>.npz
    python analysis/probe_mass.py ckpt --ckpt a.pt --ckpt b.pt --out mass_ck.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from wsparse.bottleneck.lapsum import laplace_cdf            # noqa: E402
from wsparse.config import config_from_dict                  # noqa: E402
from wsparse.data import build_streams                       # noqa: E402
from wsparse.train import train, load_for_inference          # noqa: E402


class StopProbing(Exception):
    pass


def gate_geometry(z, k, j, t, b0):
    """(scores, p, mass) for the top-(k+j) candidates of each token.

    ``z`` is the gate input, shape (batch, seq, n_features); scores are |z|
    (abs_topk).  Returns sorted-descending scores and the matching p_i.
    """
    s = z.abs().reshape(-1, z.shape[-1])
    q = min(k + j, s.shape[-1])
    sc = s.topk(q, dim=-1, sorted=True).values
    b = sc[:, k].clamp_min(b0) if k < q else sc[:, -1].clamp_min(b0)
    p = laplace_cdf((sc - b[:, None]) / t)
    return sc, p, p.sum(-1)


class MassProbe:
    def __init__(self, cfg, every, stop_step, positions, prof_layers, prof_tokens):
        self.cfg = cfg
        self.every = int(every)
        self.stop_step = int(stop_step)
        self.positions = list(positions)
        self.prof_layers = list(prof_layers)
        self.prof_tokens = list(prof_tokens)
        b = cfg.activation_bottleneck
        self.k, self.j = int(b.k), int(b.j)
        self.t = float(b.rblapsum_temperature)
        self.b0 = float(b.rblapsum_boundary_floor)
        self.x = self.y = None
        self.steps, self.mass, self.prof_s, self.prof_p, self.ce = [], [], [], [], []

    def _batch(self, device):
        tr, _ = build_streams(self.cfg.data, seed=self.cfg.train.seed)
        # a fixed batch, identical for every run compared
        return tr.batch(4, device, deterministic_offset=4242)

    def __call__(self, step, model, bottleneck, optimizer):
        if step > self.stop_step:
            raise StopProbing
        if step % self.every:
            return
        layers = bottleneck.layers
        device = next(model.parameters()).device
        if self.x is None:
            self.x, self.y = self._batch(device)

        # ---- snapshot what the measurement would perturb ------------------ #
        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        snaps = []
        for _, mod in layers:
            g = mod.gate
            snaps.append((g.usage_ema.clone(), g.usage_steps.clone(),
                          dict(g._forward_diag), dict(g._usage_diag),
                          dict(g._grad_sink), mod._reconstruction))
        was_training = model.training

        cap = {}
        handles = []
        for li, (_, mod) in enumerate(layers):
            handles.append(mod.gate.register_forward_pre_hook(
                lambda m, inp, li=li: cap.__setitem__(li, inp[0].detach().float())))
        with torch.no_grad():
            _, loss = model(self.x, self.y)
        for h in handles:
            h.remove()

        mass = np.zeros((len(layers), len(self.positions)), dtype=np.float32)
        prof_s, prof_p = [], []
        for li in range(len(layers)):
            z = cap[li]
            sc, p, m = gate_geometry(z, self.k, self.j, self.t, self.b0)
            seq = z.shape[1]
            # token index p of sequence 0 -> flat row p
            rows = [t_ for t_ in self.positions if t_ < seq]
            mass[li, :len(rows)] = m[rows].cpu().numpy()
            if li in self.prof_layers:
                trows = [t_ for t_ in self.prof_tokens if t_ < seq]
                prof_s.append(sc[trows].cpu().numpy())
                prof_p.append(p[trows].cpu().numpy())
        self.steps.append(int(step))
        self.mass.append(mass)
        self.ce.append(float(loss))
        if prof_s:
            self.prof_s.append(np.stack(prof_s))
            self.prof_p.append(np.stack(prof_p))

        # ---- restore ------------------------------------------------------- #
        for (_, mod), sn in zip(layers, snaps):
            g = mod.gate
            g.usage_ema.copy_(sn[0]); g.usage_steps.copy_(sn[1])
            g._forward_diag, g._usage_diag, g._grad_sink = sn[2], sn[3], sn[4]
            mod._reconstruction = sn[5]
        model.train(was_training)
        torch.set_rng_state(rng_cpu)
        if rng_cuda is not None:
            torch.cuda.set_rng_state_all(rng_cuda)
        cap.clear()
        print(f"[mass] step {step:>6}  CE {float(loss):.4f}  "
              f"mass(mean over layers/tokens) {mass.mean():.2f}  "
              f"mass(block4,tok0) {mass[min(4, len(layers)-1), 0]:.2f}", flush=True)

    def save(self, path, extra):
        np.savez_compressed(
            path,
            steps=np.array(self.steps),
            mass=np.stack(self.mass) if self.mass else np.zeros(0),
            ce=np.array(self.ce),
            prof_s=np.stack(self.prof_s) if self.prof_s else np.zeros(0),
            prof_p=np.stack(self.prof_p) if self.prof_p else np.zeros(0),
            meta=json.dumps({**extra, "k": self.k, "j": self.j, "T": self.t,
                             "b0": self.b0, "positions": self.positions,
                             "prof_layers": self.prof_layers,
                             "prof_tokens": self.prof_tokens}),
        )
        print(f"[mass] wrote {path}: {len(self.steps)} measurement steps")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("probe", "ckpt"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
        s.add_argument("--positions", type=int, nargs="*", default=[0, 7, 23, 61])
        s.add_argument("--prof-layers", type=int, nargs="*", default=[4, 7])
        s.add_argument("--prof-tokens", type=int, nargs="*", default=[0, 7])
        s.add_argument("--data-dir", default="/workspace/data/tinystories")
        if name == "probe":
            s.add_argument("--run-config", required=True,
                           help="a run's dumped config.json")
            s.add_argument("--stop-step", type=int, default=2500)
            s.add_argument("--every", type=int, default=50)
        else:
            s.add_argument("--ckpt", action="append", required=True)
    args = ap.parse_args()

    if args.cmd == "probe":
        cfg = config_from_dict(json.load(open(args.run_config)))
        cfg.data.data_dir = args.data_dir
        cfg.train.resume = ""            # a fresh trajectory, not a resume
        cfg.train.run_name = "mass_probe_tmp"
        cfg.train.checkpoint_every_steps = 10 ** 9   # save nothing
        cfg.train.sample_every_steps = 0
        cfg.train.tensorboard = False
        probe = MassProbe(cfg, args.every, args.stop_step, args.positions,
                          args.prof_layers, args.prof_tokens)
        try:
            train(cfg, on_step=probe)
        except StopProbing:
            print(f"[mass] stopped at {args.stop_step} as planned")
        probe.save(args.out, {"mode": "probe", "config": args.run_config})
    else:
        # measure saved checkpoints; the config travels inside each one
        steps, mass, prof_s, prof_p, ce = [], [], [], [], []
        meta = {}
        for path in args.ckpt:
            model, cfg, bctl = load_for_inference(path, device="cuda")
            cfg.data.data_dir = args.data_dir
            b = cfg.activation_bottleneck
            k, j = int(b.k), int(b.j)
            t, b0 = float(b.rblapsum_temperature), float(b.rblapsum_boundary_floor)
            tr, _ = build_streams(cfg.data, seed=cfg.train.seed)
            device = next(model.parameters()).device
            x, y = tr.batch(4, device, deterministic_offset=4242)
            model.train()        # same mode the probe measures in
            cap, handles = {}, []
            for li, (_, mod) in enumerate(bctl.layers):
                handles.append(mod.gate.register_forward_pre_hook(
                    lambda m, inp, li=li: cap.__setitem__(li, inp[0].detach().float())))
            with torch.no_grad():
                _, loss = model(x, y)
            for h in handles:
                h.remove()
            m_all = np.zeros((len(bctl.layers), len(args.positions)), dtype=np.float32)
            ps, pp = [], []
            for li in range(len(bctl.layers)):
                sc, p, m = gate_geometry(cap[li], k, j, t, b0)
                m_all[li, :] = m[args.positions].cpu().numpy()
                if li in args.prof_layers:
                    ps.append(sc[args.prof_tokens].cpu().numpy())
                    pp.append(p[args.prof_tokens].cpu().numpy())
            step = int(torch.load(path, map_location="cpu",
                                  weights_only=False).get("step", -1))
            steps.append(step); mass.append(m_all); ce.append(float(loss))
            prof_s.append(np.stack(ps)); prof_p.append(np.stack(pp))
            meta = {"k": k, "j": j, "T": t, "b0": b0, "mode": "ckpt",
                    "positions": args.positions, "prof_layers": args.prof_layers,
                    "prof_tokens": args.prof_tokens}
            print(f"[mass] {os.path.basename(path)} step {step}: CE {float(loss):.4f} "
                  f"mass(block4,tok0) {m_all[min(4, len(bctl.layers)-1), 0]:.2f}",
                  flush=True)
            del model
            torch.cuda.empty_cache()
        order = np.argsort(steps)
        np.savez_compressed(
            args.out, steps=np.array(steps)[order],
            mass=np.stack(mass)[order], ce=np.array(ce)[order],
            prof_s=np.stack(prof_s)[order], prof_p=np.stack(prof_p)[order],
            meta=json.dumps(meta))
        print(f"[mass] wrote {args.out}: {len(steps)} checkpoints")


if __name__ == "__main__":
    main()
