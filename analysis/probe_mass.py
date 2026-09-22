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


def gate_gradients(z, u, k, j, t, b0, mode):
    """(scores, dL/dp, dL/ds) at the top-(k+j) candidates, per token.

    Follows the backward of :class:`wsparse.bottleneck.rblapsum._RBLapSumGate`
    exactly.  With ``y_i = p_i z_i`` and ``u_i = dL/dy_i``,

        dL/dp_i = u_i z_i,
        a_i     = (dL/dp_i) * kappa_i          (kappa_i = dp_i/ds_i)

    and the score gradient is ``a`` after the mode correction: for
    ``through_rank`` the common mode is dropped as a point mass on the
    boundary feature, for ``through_rank_kappa`` it is spread over the pool
    kappa-weighted (``a - q sum_i a_i``, ``q = kappa / sum kappa``).
    """
    F = z.shape[-1]
    zf = z.reshape(-1, F)
    uf = u.reshape(-1, F)
    sraw = zf.abs()
    q_n = min(k + j, F)
    order = sraw.argsort(dim=-1, descending=True)[:, :q_n]
    sc = sraw.gather(1, order)
    z_c = zf.gather(1, order)
    u_c = uf.gather(1, order)
    b = sc[:, k].clamp_min(b0) if k < q_n else sc[:, -1].clamp_min(b0)
    cap_active = (sc[:, k] > b0)[:, None] if k < q_n else torch.ones_like(b)[:, None]
    kap = torch.exp(-(sc - b[:, None]).abs() / t) / (2.0 * t)
    dLdp = u_c * z_c
    a = dLdp * kap
    if mode == "through_rank_kappa":
        qw = kap / kap.sum(1, keepdim=True).clamp_min(1e-30)
        g_s = torch.where(cap_active, a - qw * a.sum(1, keepdim=True), a)
    elif mode == "through_rank":
        corr = torch.zeros_like(a)
        total = a.sum(1, keepdim=True)
        corr[:, k:k + 1] = torch.where(cap_active, total, torch.zeros_like(total))
        g_s = a - corr
    else:
        g_s = a
    return sc, dLdp, g_s


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


class GradProbe:
    """Same cadence as MassProbe, but runs a backward so u_i is available.

    Gradient probes must run in train() mode with grad enabled:
    ``surrogate_active()`` is False in eval and under ``no_grad``, so an
    eval-mode probe would measure the hard mask instead of the surrogate.
    The probe's own gradients are discarded -- the hook fires before the
    step's ``zero_grad`` -- and the gate buffers and RNG it perturbs are
    restored.
    """

    def __init__(self, cfg, every, stop_step, prof_layers, prof_tokens):
        self.cfg = cfg
        self.every, self.stop_step = int(every), int(stop_step)
        self.prof_layers, self.prof_tokens = list(prof_layers), list(prof_tokens)
        b = cfg.activation_bottleneck
        self.k, self.j = int(b.k), int(b.j)
        self.t = float(b.rblapsum_temperature)
        self.b0 = float(b.rblapsum_boundary_floor)
        self.mode = str(b.rblapsum_boundary_grad_mode)
        self.x = self.y = None
        self.steps, self.s, self.dLdp, self.dLds, self.ce = [], [], [], [], []

    def _batch(self, device):
        tr, _ = build_streams(self.cfg.data, seed=self.cfg.train.seed)
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

        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        snaps = []
        for _, mod in layers:
            g = mod.gate
            snaps.append((g.usage_ema.clone(), g.usage_steps.clone(),
                          dict(g._forward_diag), dict(g._usage_diag),
                          dict(g._grad_sink), mod._reconstruction))
        was_training = model.training
        model.train()

        cap, handles = {}, []
        for li, (_, mod) in enumerate(layers):
            if li not in self.prof_layers:
                continue

            def pre(m, inputs, li=li):
                cap[("z", li)] = inputs[0].detach().float()

            def post(m, inputs, output, li=li):
                if output.requires_grad:
                    output.register_hook(
                        lambda g, li=li: cap.__setitem__(("u", li), g.detach().float()))

            handles.append(mod.gate.register_forward_pre_hook(pre))
            handles.append(mod.gate.register_forward_hook(post))

        _, loss = model(self.x, self.y)
        loss.backward()
        for h in handles:
            h.remove()

        ss, gp, gs = [], [], []
        for li in self.prof_layers:
            z, u = cap[("z", li)], cap[("u", li)]
            sc, dLdp, g_s = gate_gradients(z, u, self.k, self.j, self.t,
                                           self.b0, self.mode)
            seq = z.shape[1]
            rows = [t_ for t_ in self.prof_tokens if t_ < seq]
            ss.append(sc[rows].cpu().numpy())
            gp.append(dLdp[rows].cpu().numpy())
            gs.append(g_s[rows].cpu().numpy())
        self.steps.append(int(step))
        self.s.append(np.stack(ss)); self.dLdp.append(np.stack(gp))
        self.dLds.append(np.stack(gs)); self.ce.append(float(loss.detach()))

        optimizer.zero_grad(set_to_none=True)
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
        gp0 = self.dLdp[-1][0, 0]
        print(f"[grad] step {step:>6}  CE {float(loss.detach()):.4f}  "
              f"max|dL/dp| {np.abs(gp0).max():.3g}  "
              f"max|dL/ds| {np.abs(self.dLds[-1][0, 0]).max():.3g}", flush=True)

    def save(self, path, extra):
        np.savez_compressed(
            path, steps=np.array(self.steps), s=np.stack(self.s),
            dLdp=np.stack(self.dLdp), dLds=np.stack(self.dLds),
            ce=np.array(self.ce),
            meta=json.dumps({**extra, "k": self.k, "j": self.j, "T": self.t,
                             "b0": self.b0, "mode": self.mode,
                             "prof_layers": self.prof_layers,
                             "prof_tokens": self.prof_tokens}))
        print(f"[grad] wrote {path}: {len(self.steps)} measurement steps")


class GainProbe:
    """Magnitude-direction gains and spectral concentration of the bottleneck
    projections over training.

    Under ``decouple=True`` the gains are optimizer state, not model
    parameters: per matrix the optimizer holds ``raw_grow`` / ``raw_gcol``
    (softplus-raw, initialized so every gain is exactly 1) and the sphere
    radius ``c_f``.  The direction is pinned at ``||W_hat||_F = c_f``, so a
    growing score scale has to come either from the gains or from the
    direction concentrating its fixed norm onto fewer singular directions --
    this probe measures both.

    No forward pass is needed, so the measurement cannot perturb the run at
    all; it only reads parameters and optimizer state.
    """

    def __init__(self, cfg, every, stop_step):
        self.cfg = cfg
        self.every, self.stop_step = int(every), int(stop_step)
        self.steps, self.rows = [], []

    def __call__(self, step, model, bottleneck, optimizer):
        if step > self.stop_step:
            raise StopProbing
        if step % self.every:
            return
        import torch.nn.functional as Fn

        rec = []
        for lbl, mod in bottleneck.layers:
            per_mat = {}
            for which, W in (("enc", mod.in_proj.weight),
                             ("dec", mod.out_proj.weight)):
                st = optimizer.state.get(W, {})
                grow = (Fn.softplus(st["raw_grow"]) if "raw_grow" in st
                        else torch.ones(W.shape[0], device=W.device))
                gcol = (Fn.softplus(st["raw_gcol"]) if "raw_gcol" in st
                        else torch.ones(W.shape[1], device=W.device))
                with torch.no_grad():
                    w_hat = W.detach().float()
                    w_hat = w_hat / grow.float().unsqueeze(1)
                    w_hat = w_hat / gcol.float().unsqueeze(0)
                    fro_hat = float(w_hat.norm())
                    smax = float(torch.linalg.matrix_norm(w_hat, ord=2))
                    per_mat[which] = {
                        "grow_mean": float(grow.mean()),
                        "grow_max": float(grow.max()),
                        "grow_min": float(grow.min()),
                        "grow_p90": float(grow.float().quantile(0.9)),
                        "gcol_mean": float(gcol.mean()),
                        "gcol_max": float(gcol.max()),
                        "fused_fro": float(W.detach().float().norm()),
                        "dir_fro": fro_hat,
                        "c_f": float(st["c_f"]) if "c_f" in st else fro_hat,
                        "sigma_max": smax,
                        "sigma_over_fro": smax / max(fro_hat, 1e-30),
                    }
            rec.append(per_mat)
        self.steps.append(int(step))
        self.rows.append(rec)
        e = rec[4]["enc"]
        print(f"[gain] step {step:>6}  enc(block4): grow mean {e['grow_mean']:.4f} "
              f"max {e['grow_max']:.4f} | gcol mean {e['gcol_mean']:.4f} | "
              f"fused ||W|| {e['fused_fro']:.1f} (c_F {e['c_f']:.1f}) | "
              f"sigma_max/||W_hat|| {e['sigma_over_fro']:.4f}", flush=True)

    def save(self, path, extra):
        with open(path, "w") as f:
            json.dump({"meta": extra, "steps": self.steps, "rows": self.rows}, f)
        print(f"[gain] wrote {path}: {len(self.steps)} measurement steps")


class RowProbe:
    """Encoder rows of the top-(K+J) candidates, and the score decomposition.

    For one block and one token the score of candidate i is

        s_i = |w_i . x| = ||w_i|| ||x|| |cos(w_i, x)|,

    with ``w_i`` the encoder row (row of ``in_proj.weight``) that produces the
    feature and ``x`` the block's bottleneck input.  Recording the three
    factors separately says which of them carries a change in the score, and
    recording them per candidate -- ordered by score -- says whether the pool
    rows behave differently from the other 1536 - (K+J).

    Under ``decouple=True`` each row also has a learnable gain
    ``softplus(raw_grow)_i`` held in the optimizer state; the fused row is the
    gain times the on-sphere direction row.
    """

    def __init__(self, cfg, every, stop_step, layer, token):
        self.cfg = cfg
        self.every, self.stop_step = int(every), int(stop_step)
        self.layer, self.token = int(layer), int(token)
        b = cfg.activation_bottleneck
        self.k, self.j = int(b.k), int(b.j)
        self.t = float(b.rblapsum_temperature)
        self.b0 = float(b.rblapsum_boundary_floor)
        self.x = self.y = None
        self.out = {n: [] for n in ("steps", "x_norm", "s", "row_fused",
                                    "row_dir", "row_gain", "cos", "idx",
                                    "all_row_fused_mean", "ce")}

    def _batch(self, device):
        tr, _ = build_streams(self.cfg.data, seed=self.cfg.train.seed)
        return tr.batch(4, device, deterministic_offset=4242)

    def __call__(self, step, model, bottleneck, optimizer):
        if step > self.stop_step:
            raise StopProbing
        if step % self.every:
            return
        import torch.nn.functional as Fn

        layers = bottleneck.layers
        device = next(model.parameters()).device
        if self.x is None:
            self.x, self.y = self._batch(device)
        lbl, mod = layers[self.layer]

        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        snaps = []
        for _, m_ in layers:
            g = m_.gate
            snaps.append((g.usage_ema.clone(), g.usage_steps.clone(),
                          dict(g._forward_diag), dict(g._usage_diag),
                          dict(g._grad_sink), m_._reconstruction))
        was_training = model.training

        cap = {}
        h1 = mod.register_forward_pre_hook(
            lambda m_, inp: cap.__setitem__("x", inp[0].detach().float()))
        h2 = mod.gate.register_forward_pre_hook(
            lambda m_, inp: cap.__setitem__("z", inp[0].detach().float()))
        with torch.no_grad():
            _, loss = model(self.x, self.y)
        h1.remove(); h2.remove()

        with torch.no_grad():
            xrow = cap["x"].reshape(-1, cap["x"].shape[-1])[self.token]
            zrow = cap["z"].reshape(-1, cap["z"].shape[-1])[self.token]
            q = min(self.k + self.j, zrow.shape[-1])
            sc, idx = zrow.abs().topk(q, sorted=True)
            W = mod.in_proj.weight.detach().float()          # (n_features, d_model)
            st = optimizer.state.get(mod.in_proj.weight, {})
            grow = (Fn.softplus(st["raw_grow"]).float() if "raw_grow" in st
                    else torch.ones(W.shape[0], device=W.device))
            gcol = (Fn.softplus(st["raw_gcol"]).float() if "raw_gcol" in st
                    else torch.ones(W.shape[1], device=W.device))
            rows = W[idx]
            row_fused = rows.norm(dim=-1)
            row_dir = (rows / grow[idx].unsqueeze(1) / gcol.unsqueeze(0)).norm(dim=-1)
            xn = float(xrow.norm())
            cos = (sc / (row_fused * max(xn, 1e-30))).clamp(max=1.0)
            self.out["steps"].append(int(step))
            self.out["x_norm"].append(xn)
            self.out["s"].append(sc.cpu().numpy())
            self.out["row_fused"].append(row_fused.cpu().numpy())
            self.out["row_dir"].append(row_dir.cpu().numpy())
            self.out["row_gain"].append(grow[idx].cpu().numpy())
            self.out["cos"].append(cos.cpu().numpy())
            self.out["idx"].append(idx.cpu().numpy())
            self.out["all_row_fused_mean"].append(float(W.norm(dim=-1).mean()))
            self.out["ce"].append(float(loss))

        for (_, m_), sn in zip(layers, snaps):
            g = m_.gate
            g.usage_ema.copy_(sn[0]); g.usage_steps.copy_(sn[1])
            g._forward_diag, g._usage_diag, g._grad_sink = sn[2], sn[3], sn[4]
            m_._reconstruction = sn[5]
        model.train(was_training)
        torch.set_rng_state(rng_cpu)
        if rng_cuda is not None:
            torch.cuda.set_rng_state_all(rng_cuda)
        cap.clear()
        K = self.k
        print(f"[row] step {step:>6}  ||x|| {xn:.3g}  s_1 {float(sc[0]):.3g}  "
              f"pool row||w|| mean {float(row_fused.mean()):.4f} "
              f"(all rows {self.out['all_row_fused_mean'][-1]:.4f})  "
              f"gain pool mean {float(grow[idx].mean()):.4f}  "
              f"|cos| mean {float(cos.mean()):.4f}", flush=True)

    def save(self, path, extra):
        np.savez_compressed(
            path,
            meta=json.dumps({**extra, "k": self.k, "j": self.j, "T": self.t,
                             "b0": self.b0, "layer": self.layer,
                             "token": self.token}),
            **{n: np.array(v) for n, v in self.out.items()})
        print(f"[row] wrote {path}: {len(self.out['steps'])} measurement steps")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("probe", "ckpt", "gradprobe", "gainprobe", "rowprobe"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
        s.add_argument("--positions", type=int, nargs="*", default=[0, 7, 23, 61])
        s.add_argument("--prof-layers", type=int, nargs="*", default=[4, 7])
        s.add_argument("--prof-tokens", type=int, nargs="*", default=[0, 7])
        s.add_argument("--data-dir", default="/workspace/data/tinystories")
        if name in ("probe", "gradprobe", "gainprobe", "rowprobe"):
            s.add_argument("--run-config", required=True,
                           help="a run's dumped config.json")
            s.add_argument("--stop-step", type=int, default=2500)
            s.add_argument("--every", type=int, default=50)
        else:
            s.add_argument("--ckpt", action="append", required=True)
    args = ap.parse_args()

    if args.cmd == "rowprobe":
        cfg = config_from_dict(json.load(open(args.run_config)))
        cfg.data.data_dir = args.data_dir
        cfg.train.resume = ""
        cfg.train.run_name = "row_probe_tmp"
        cfg.train.checkpoint_every_steps = 10 ** 9
        cfg.train.sample_every_steps = 0
        cfg.train.tensorboard = False
        probe = RowProbe(cfg, args.every, args.stop_step,
                         args.prof_layers[0], args.prof_tokens[0])
        try:
            train(cfg, on_step=probe)
        except StopProbing:
            print(f"[row] stopped at {args.stop_step} as planned")
        probe.save(args.out, {"mode_cmd": "rowprobe", "config": args.run_config})
    elif args.cmd == "gainprobe":
        cfg = config_from_dict(json.load(open(args.run_config)))
        cfg.data.data_dir = args.data_dir
        cfg.train.resume = ""
        cfg.train.run_name = "gain_probe_tmp"
        cfg.train.checkpoint_every_steps = 10 ** 9
        cfg.train.sample_every_steps = 0
        cfg.train.tensorboard = False
        probe = GainProbe(cfg, args.every, args.stop_step)
        try:
            train(cfg, on_step=probe)
        except StopProbing:
            print(f"[gain] stopped at {args.stop_step} as planned")
        probe.save(args.out, {"mode_cmd": "gainprobe", "config": args.run_config})
    elif args.cmd == "gradprobe":
        cfg = config_from_dict(json.load(open(args.run_config)))
        cfg.data.data_dir = args.data_dir
        cfg.train.resume = ""
        cfg.train.run_name = "grad_probe_tmp"
        cfg.train.checkpoint_every_steps = 10 ** 9
        cfg.train.sample_every_steps = 0
        cfg.train.tensorboard = False
        probe = GradProbe(cfg, args.every, args.stop_step,
                          args.prof_layers, args.prof_tokens)
        try:
            train(cfg, on_step=probe)
        except StopProbing:
            print(f"[grad] stopped at {args.stop_step} as planned")
        probe.save(args.out, {"mode_cmd": "gradprobe", "config": args.run_config})
    elif args.cmd == "probe":
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
