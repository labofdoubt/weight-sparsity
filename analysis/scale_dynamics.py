"""One-step score-scale dynamics of the RBLapSum kappa gate.

Measures, at saved checkpoints, whether and how the training update moves the
score scale -- the quantities that the burst-escape investigation needs
(docs/rblapsum-kappa-stability.md, and the follow-up protocol):

``drift``     At one checkpoint (model + optimizer state), estimate the
              one-step change of the per-layer score scale x = log R under
              counterfactual optimizer steps computed from the same state:
              the full gradient, the task gradient alone (surrogate support
              gradient disabled), and the support gradient alone (the exact
              raw-gradient difference; optimizer steps are nonlinear in the
              gradient, so branch updates need not add).  Also records, per
              layer: the exact surrogate gain Pi = ||L_kappa D_z||_F^2, its
              dense-boundary approximation b^2/(4 T delta), chi_geo,
              n_eff, the empirical gain ||g_s||^2/||u||^2, the radial
              projection of the surrogate score update, and a permutation
              null for the radial projection (w shuffled among candidates).

``rescale``   The same drift measurement swept over the gate view scale
              alpha (z -> alpha z at the gate input, output / alpha): the
              forward function and the task gradient are exactly unchanged,
              the surrogate sees an alpha-times score excursion at fixed T.
              Produces F(x) = E[dx] and D(x) as a function of the imposed
              scale offset log(alpha).

``reparam``   Exact score-units reparameterization check on a real
              checkpoint: alpha = c together with temperature * c must leave
              every parameter gradient invariant while chi_geo scales by 1/c.

``continue``  Resume training from a checkpoint for a fixed number of steps
              with optional interventions (support gradient off, temperature
              multiplied), applied at the start or when a boundary-inflation
              trigger fires; records per-layer R and b over the continuation.

All measurements run per layer.  Scale statistics: R_all = RMS over all
features of s = |z|; R_K = mean of the top-K scores; b = s_(K+1).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from wsparse.config import config_from_dict  # noqa: E402
from wsparse.data import TokenStream  # noqa: E402
from wsparse.model import build_model  # noqa: E402
from wsparse.sparsity import apply_sparsity  # noqa: E402
from wsparse.bottleneck import apply_activation_bottleneck  # noqa: E402
from wsparse.train import lr_at, set_lr, train  # noqa: E402
from wsparse.utils import autocast_context, resolve_device, resolve_dtype  # noqa: E402


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_ckpt(path: str, device: str):
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_dict(payload["config"])
    model = build_model(cfg.model)
    sctl = apply_sparsity(model, cfg.sparsity, max_steps=cfg.train.max_steps)
    bctl = apply_activation_bottleneck(
        model, cfg.activation_bottleneck, max_steps=cfg.train.max_steps)
    model.load_state_dict(payload["model"])
    model.to(device)
    step = int(payload.get("step", 0))
    sctl.set_step(step)
    bctl.set_step(step)
    assert cfg.model.decouple, "scale_dynamics expects the MD/decouple optimizer"
    from wsparse.decouple import build_decoupled_optimizer
    opt = build_decoupled_optimizer(
        model, cfg.train, gain_mode=cfg.model.decouple_gains,
        mask_param_ids=sctl.mask_parameter_ids() or None)
    opt.load_state_dict(payload["optimizer"])
    lr = lr_at(step, cfg.train)
    set_lr(opt, lr)
    return model, cfg, bctl, opt, step, lr


def gates_of(bctl):
    return [(lbl, mod.gate) for lbl, mod in bctl.layers]


def set_knob(bctl, name: str, value: float):
    for _, g in gates_of(bctl):
        setattr(g, name, float(value))


# --------------------------------------------------------------------------- #
# capture
# --------------------------------------------------------------------------- #

class Capture:
    """Collect each gate's input z (pre-hook) and output gradient u (hook)."""

    def __init__(self, bctl, want_u: bool):
        self.layers = bctl.layers
        self.want_u = want_u
        self.z: dict = {}
        self.u: dict = {}
        self.handles = []

    def __enter__(self):
        for li, (_, mod) in enumerate(self.layers):
            def pre(m, inputs, li=li):
                a = inputs[0]
                self.z[li] = a.detach().float()
            self.handles.append(mod.gate.register_forward_pre_hook(pre))
            if self.want_u:
                def post(m, inputs, out, li=li):
                    if out.requires_grad:
                        out.register_hook(
                            lambda g, li=li: self.u.__setitem__(li, g.detach().float()))
                self.handles.append(mod.gate.register_forward_hook(post))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def eval_scale(model, bctl, x, y):
    """Per-layer scale statistics on a fixed batch (eval mode, no grad)."""
    was = model.training
    model.eval()
    with Capture(bctl, want_u=False) as cap:
        model(x, y)
    model.train(was)
    out = []
    for li in range(len(bctl.layers)):
        z = cap.z[li]
        s = z.abs().reshape(-1, z.shape[-1])
        K = bctl.layers[li][1].gate.k
        top = s.topk(K + 1, dim=-1, sorted=True).values
        out.append({
            "R_all": float((s ** 2).mean().sqrt()),
            "R_K": float(top[:, :K].mean()),
            "b": float(top[:, K].mean()),
        })
    return out


# --------------------------------------------------------------------------- #
# per-layer analytics from (z, u)
# --------------------------------------------------------------------------- #

def layer_state_metrics(z, gate, alpha: float) -> dict:
    """Geometry of the gate view (scores scaled by alpha, fixed T)."""
    K, J = gate.k, gate.j
    T = float(gate.rblapsum_temperature)
    b0 = float(gate.rblapsum_boundary_floor)
    s = z.abs().reshape(-1, z.shape[-1]) * alpha
    q = K + J
    sc = s.topk(q, dim=-1, sorted=True).values
    m_sp = 4
    delta = (sc[:, K - 1 - m_sp] - sc[:, K - 1 + m_sp]) / (2 * m_sp)
    b = sc[:, K].clamp_min(b0)
    kap = torch.exp(-(sc - b[:, None]).abs() / T) / (2 * T)
    Z = kap.sum(1)
    S2 = (kap ** 2).sum(1)
    n_eff = Z ** 2 / S2.clamp_min(1e-30)
    chi_geo = b / (2 * T * delta.clamp_min(1e-12))
    # exact Pi = ||L_kappa D_z||_F^2 via the diagonal-minus-rank-one structure:
    # column j contributes z_j^2 kap_j^2 [ (1 - kap_j/Z)^2 + (S2 - kap_j^2)/Z^2 ]
    zc = sc  # |z| at candidates (score = |z| for abs_topk); signs square away
    colj = (zc * kap) ** 2 * ((1 - kap / Z[:, None]) ** 2
                              + (S2[:, None] - kap ** 2) / (Z[:, None] ** 2))
    Pi = colj.sum(1)
    Pi_approx = b ** 2 / (4 * T * delta.clamp_min(1e-12))
    def stat(v):
        return {"mean": float(v.mean()), "p90": float(v.quantile(0.9))}
    return {"b": stat(b), "delta": stat(delta), "n_eff": stat(n_eff),
            "chi_geo": stat(chi_geo), "Pi": stat(Pi), "Pi_approx": stat(Pi_approx),
            "sum_kappa": stat(Z), "T": T, "K": K, "J": J, "alpha": alpha}


def layer_grad_metrics(z, u, gate, alpha: float, n_perm: int) -> dict:
    """Surrogate force analytics; the reconstruction was validated against the
    captured dL/dz earlier in the investigation (relative error ~0)."""
    K, J = gate.k, gate.j
    T = float(gate.rblapsum_temperature)
    b0 = float(gate.rblapsum_boundary_floor)
    F = z.shape[-1]
    zf = z.reshape(-1, F)
    uf = u.reshape(-1, F)
    s_raw = zf.abs()
    s = s_raw * alpha
    q = K + J
    order = s.argsort(dim=-1, descending=True)[:, :q]
    s_c = s.gather(1, order)
    z_c = zf.gather(1, order) * alpha          # gate-view value
    u_c = uf.gather(1, order) / alpha          # upstream after the /alpha output
    b = s_c[:, K].clamp_min(b0)
    cap_active = (s_c[:, K] > b0)[:, None]
    kap = torch.exp(-(s_c - b[:, None]).abs() / T) / (2 * T)
    a = u_c * z_c * kap
    qw = kap / kap.sum(1, keepdim=True).clamp_min(1e-30)
    g_s = torch.where(cap_active, a - qw * a.sum(1, keepdim=True), a)
    g_s_applied = alpha * g_s                  # what reaches raw z (score space)
    # empirical gain
    G_emp = float(((g_s ** 2).sum(1) / (u_c ** 2).sum(1).clamp_min(1e-30)).mean())
    # radial projection of the surrogate score update (descent: ds ~ -g_s)
    s_c_raw = s_c / alpha
    denom = (s_c_raw ** 2).sum(1).clamp_min(1e-30)
    r_supp = (s_c_raw * (-g_s_applied)).sum(1) / denom
    # boundary-centred projection and Cov_q
    d = s_c - b[:, None]
    w = -(u_c * z_c)
    radial_Lw = float(((d / alpha) * (-g_s_applied)).sum(1).mean())
    dq = (qw * d).sum(1, keepdim=True)
    wq = (qw * w).sum(1, keepdim=True)
    cov_q = float(((qw * (d - dq) * (w - wq)).sum(1)).mean())
    # permutation null: shuffle w among candidates within each token
    null = []
    for _ in range(n_perm):
        idx = torch.argsort(torch.rand_like(w), dim=-1)
        wp = w.gather(1, idx)
        ap = -wp * kap
        gp = torch.where(cap_active, ap - qw * ap.sum(1, keepdim=True), ap)
        rp = (s_c_raw * (-alpha * gp)).sum(1) / denom
        null.append(float(rp.mean()))
    null = np.array(null) if null else np.array([0.0])
    r_real = float(r_supp.mean())
    return {"G_emp": G_emp, "r_supp": r_real,
            "r_supp_null_mean": float(null.mean()),
            "r_supp_null_std": float(null.std() + 1e-30),
            "r_supp_null_z": float((r_real - null.mean()) / (null.std() + 1e-30)),
            "radial_Lw": radial_Lw, "cov_q_d_w": cov_q,
            "u_rms": float(u_c.pow(2).mean().sqrt()),
            "gs_rms": float(g_s.pow(2).mean().sqrt())}


# --------------------------------------------------------------------------- #
# counterfactual one-step drift
# --------------------------------------------------------------------------- #

def named_grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()
            if p.grad is not None}


def assign_grads(model, G):
    for n, p in model.named_parameters():
        if n in G:
            p.grad = G[n].clone()
        else:
            p.grad = None


def run_drift(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, bctl, opt, step, lr = load_ckpt(args.ckpt, device)
    dtype = resolve_dtype(cfg.train.dtype, device)
    if args.lr_mult != 1.0:
        set_lr(opt, lr * args.lr_mult)
    set_knob(bctl, "rblapsum_view_scale", args.alpha)
    if args.temp_mult != 1.0:
        for _, g in gates_of(bctl):
            g.rblapsum_temperature = float(g.rblapsum_temperature) * args.temp_mult

    seq = int(cfg.data.seq_len)
    train_stream = TokenStream(os.path.join(args.data_dir, "train.bin"), seq, seed=1234)
    val_stream = TokenStream(os.path.join(args.data_dir, "val.bin"), seq, seed=0)
    ex, ey = val_stream.batch(args.eval_batch, device, deterministic_offset=0)

    accum = int(cfg.train.grad_accum_steps) if args.accum < 0 else args.accum
    micro = max(1, int(round(int(cfg.train.micro_batch_size) * args.batch_mult)))

    # ---- state + force analytics on one training batch ---------------------
    model.train()
    with Capture(bctl, want_u=True) as cap:
        x, y = train_stream.batch(micro, device, deterministic_offset=777)
        with autocast_context(device, dtype):
            _, loss = model(x, y)
        loss.backward()
    opt.zero_grad(set_to_none=True)
    layers_out = []
    for li, (lbl, mod) in enumerate(bctl.layers):
        g = mod.gate
        rec = {"layer": lbl,
               "state": layer_state_metrics(cap.z[li], g, args.alpha),
               "force": layer_grad_metrics(cap.z[li], cap.u[li], g, args.alpha,
                                           args.perm_null)}
        layers_out.append(rec)

    # ---- counterfactual optimizer steps ------------------------------------
    R0 = eval_scale(model, bctl, ex, ey)
    snap_m = {k: v.detach().clone() for k, v in model.state_dict().items()}
    snap_o = copy.deepcopy(opt.state_dict())
    stats = ("R_all", "R_K", "b")
    branches = ("full", "task", "supp", "nfull", "ntask")
    dx = {br: [[] for _ in bctl.layers] for br in branches}
    clip = float(cfg.train.grad_clip)

    # null branch: a zero-gradient optimizer step.  The saved optimizer state
    # carries momentum from the original run, so every counterfactual step
    # includes a momentum-replay component shared by all branches; the null
    # step measures exactly that baseline.  Deterministic, so computed once.
    G_zero = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    assign_grads(model, G_zero)
    opt.step()
    R1 = eval_scale(model, bctl, ex, ey)
    dx_null = [{st: math.log(max(R1[li][st], 1e-30))
                    - math.log(max(R0[li][st], 1e-30)) for st in stats}
               for li in range(len(bctl.layers))]
    model.load_state_dict(snap_m)
    opt.load_state_dict(copy.deepcopy(snap_o))
    set_lr(opt, lr * args.lr_mult)

    for bi in range(args.batches):
        micros = [train_stream.batch(micro, device, deterministic_offset=1000 + bi * accum + a)
                  for a in range(accum)]
        G = {}
        for supp in (0.0, 1.0):
            set_knob(bctl, "rblapsum_support_scale", supp)
            opt.zero_grad(set_to_none=True)
            model.train()
            # identical RNG for both passes so any stochastic layer produces
            # the same masks and G_full - G_task is exactly the surrogate term
            torch.manual_seed(9000 + bi)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(9000 + bi)
            for (x, y) in micros:
                with autocast_context(device, dtype):
                    _, loss = model(x, y)
                (loss / accum).backward()
            G[supp] = named_grads(model)
        set_knob(bctl, "rblapsum_support_scale", 1.0)
        G_task, G_full = G[0.0], G[1.0]
        G_supp = {n: G_full[n] - G_task.get(n, torch.zeros_like(G_full[n]))
                  for n in G_full}
        G_negf = {n: -g for n, g in G_full.items()}
        G_negt = {n: -g for n, g in G_task.items()}
        for br, Gb in (("full", G_full), ("task", G_task), ("supp", G_supp),
                       ("nfull", G_negf), ("ntask", G_negt)):
            assign_grads(model, Gb)
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            R1 = eval_scale(model, bctl, ex, ey)
            for li in range(len(bctl.layers)):
                dx[br][li].append({st: math.log(max(R1[li][st], 1e-30))
                                       - math.log(max(R0[li][st], 1e-30))
                                   for st in stats})
            model.load_state_dict(snap_m)
            opt.load_state_dict(copy.deepcopy(snap_o))
            set_lr(opt, lr * args.lr_mult)
        opt.zero_grad(set_to_none=True)

    for li, rec in enumerate(layers_out):
        rec["drift"] = {"null": {st: dx_null[li][st] for st in stats}}
        for br in branches:
            rec["drift"][br] = {}
            for st in stats:
                v = np.array([d[st] for d in dx[br][li]])
                rec["drift"][br][st] = {
                    "mu": float(v.mean()),
                    "D": float(0.5 * v.var()),
                    "n": int(v.size),
                    "samples": [float(t) for t in v]}
        # paired effects: task_effect = task - null (per batch, null constant);
        # supp_effect = full - task (per batch, cancels shared batch noise).
        # Any parameter perturbation inflates R quadratically (heating), so the
        # signed first-order drift is the antithetic ODD part
        # (Delta(+G) - Delta(-G)) / 2, and the quadratic heating is the EVEN
        # part (Delta(+G) + Delta(-G)) / 2 - Delta(null).
        rec["drift"]["task_effect"] = {}
        rec["drift"]["supp_effect"] = {}
        rec["drift"]["task_odd"] = {}
        rec["drift"]["full_odd"] = {}
        rec["drift"]["supp_odd"] = {}
        rec["drift"]["task_even"] = {}
        rec["drift"]["full_even"] = {}
        for st in stats:
            t = np.array([d[st] for d in dx["task"][li]]) - dx_null[li][st]
            f = np.array([d[st] for d in dx["full"][li]])
            k = np.array([d[st] for d in dx["task"][li]])
            nf = np.array([d[st] for d in dx["nfull"][li]])
            nt = np.array([d[st] for d in dx["ntask"][li]])
            e = f - k
            t_odd = 0.5 * (k - nt)
            f_odd = 0.5 * (f - nf)
            s_odd = f_odd - t_odd
            t_even = 0.5 * (k + nt) - dx_null[li][st]
            f_even = 0.5 * (f + nf) - dx_null[li][st]
            def pack(v, keep=False):
                d = {"mu": float(v.mean()), "D": float(0.5 * v.var()),
                     "n": int(v.size)}
                if keep:
                    d["samples"] = [float(x) for x in v]
                return d
            rec["drift"]["task_effect"][st] = pack(t)
            rec["drift"]["supp_effect"][st] = pack(e, keep=True)
            rec["drift"]["task_odd"][st] = pack(t_odd)
            rec["drift"]["full_odd"][st] = pack(f_odd)
            rec["drift"]["supp_odd"][st] = pack(s_odd, keep=True)
            rec["drift"]["task_even"][st] = pack(t_even)
            rec["drift"]["full_even"][st] = pack(f_even)

    out = {"ckpt": args.ckpt, "step": step, "lr": lr, "lr_mult": args.lr_mult,
           "alpha": args.alpha, "temp_mult": args.temp_mult,
           "batches": args.batches, "accum": accum, "micro": micro,
           "batch_mult": args.batch_mult,
           "k": cfg.activation_bottleneck.k, "j": cfg.activation_bottleneck.j,
           "T": cfg.activation_bottleneck.rblapsum_temperature,
           "R0": R0, "layers": layers_out}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    mu_to = np.mean([r["drift"]["task_odd"]["R_all"]["mu"] for r in layers_out])
    mu_so = np.mean([r["drift"]["supp_odd"]["R_all"]["mu"] for r in layers_out])
    mu_fe = np.mean([r["drift"]["full_even"]["R_all"]["mu"] for r in layers_out])
    print(f"[drift] {os.path.basename(args.ckpt)} alpha={args.alpha} "
          f"task_odd={mu_to:+.3e} supp_odd={mu_so:+.3e} heat={mu_fe:+.3e} "
          f"-> {args.out}")


# --------------------------------------------------------------------------- #
# reparameterization check
# --------------------------------------------------------------------------- #

def run_reparam(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for c in (1.0, args.c):
        model, cfg, bctl, opt, step, lr = load_ckpt(args.ckpt, device)
        dtype = resolve_dtype(cfg.train.dtype, device)
        set_knob(bctl, "rblapsum_view_scale", c)
        for _, g in gates_of(bctl):
            g.rblapsum_temperature = float(g.rblapsum_temperature) * c
        stream = TokenStream(os.path.join(args.data_dir, "train.bin"),
                             int(cfg.data.seq_len), seed=1234)
        x, y = stream.batch(int(cfg.train.micro_batch_size), device,
                            deterministic_offset=777)
        model.train()
        opt.zero_grad(set_to_none=True)
        with Capture(bctl, want_u=False) as cap:
            with autocast_context(device, dtype):
                _, loss = model(x, y)
            loss.backward()
        chi = []
        for li, (_, mod) in enumerate(bctl.layers):
            chi.append(layer_state_metrics(cap.z[li], mod.gate, c)["chi_geo"]["mean"])
        results[c] = {"loss": float(loss), "grads": named_grads(model), "chi": chi}
    g1, gc = results[1.0]["grads"], results[args.c]["grads"]
    rels = {}
    for n in g1:
        a, b = g1[n], gc[n]
        rels[n] = float((a - b).norm() / a.norm().clamp_min(1e-30))
    worst = max(rels.values())
    print(f"[reparam] c={args.c}: worst relative gradient difference = {worst:.3e}")
    print(f"[reparam] chi_geo layer means at c=1: "
          f"{[round(v,1) for v in results[1.0]['chi']]}")
    print(f"[reparam] chi_geo layer means at c={args.c}: "
          f"{[round(v,1) for v in results[args.c]['chi']]}")
    json.dump({"c": args.c, "worst_rel_grad_diff": worst,
               "rel_by_param": rels,
               "chi_1": results[1.0]["chi"], "chi_c": results[args.c]["chi"],
               "loss_1": results[1.0]["loss"], "loss_c": results[args.c]["loss"]},
              open(args.out, "w"), indent=1)


# --------------------------------------------------------------------------- #
# continuation with interventions
# --------------------------------------------------------------------------- #

class StopContinue(Exception):
    pass


def run_continue(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = config_from_dict(payload["config"])
    start = int(payload["step"])
    cfg.train.resume = args.ckpt
    cfg.train.run_name = args.tag
    cfg.train.out_dir = args.scratch
    cfg.train.checkpoint_every_steps = 10 ** 9
    cfg.train.sample_every_steps = 10 ** 9
    cfg.train.wandb_project = ""
    if args.seed:
        cfg.train.seed = args.seed

    seq = int(cfg.data.seq_len)
    val_stream = TokenStream(os.path.join(args.data_dir, "val.bin"), seq, seed=0)
    rec = {"steps": [], "ce": [], "R_all": [], "b": [], "trigger_step": None,
           "applied": False}
    state = {"ex": None, "bctl": None, "medbuf": []}

    def apply_knobs(bctl):
        set_knob(bctl, "rblapsum_support_scale", args.supp_scale)
        if args.temp_mult != 1.0:
            for _, g in gates_of(bctl):
                g.rblapsum_temperature = float(g.rblapsum_temperature) * args.temp_mult
        rec["applied"] = True

    def hook(step, model, bctl, optimizer):
        if step >= start + args.steps:
            raise StopContinue
        state["bctl"] = bctl
        if state["ex"] is None:
            state["ex"] = val_stream.batch(args.eval_batch,
                                           next(model.parameters()).device,
                                           deterministic_offset=0)
        if args.when == "start" and not rec["applied"] and step >= start:
            apply_knobs(bctl)
        if step % args.every:
            return
        ex, ey = state["ex"]
        was = model.training
        model.eval()
        with torch.no_grad(), Capture(bctl, want_u=False) as cap:
            _, ce = model(ex, ey)
        model.train(was)
        Rs, bs = [], []
        for li, (_, mod) in enumerate(bctl.layers):
            z = cap.z[li]
            s = z.abs().reshape(-1, z.shape[-1])
            K = mod.gate.k
            top = s.topk(K + 1, dim=-1, sorted=True).values
            Rs.append(float((s ** 2).mean().sqrt()))
            bs.append(float(top[:, K].mean()))
        rec["steps"].append(int(step))
        rec["ce"].append(float(ce))
        rec["R_all"].append(Rs)
        rec["b"].append(bs)
        pooled_b = float(np.mean(bs))
        buf = state["medbuf"]
        if args.when.startswith("trigger") and not rec["applied"] and len(buf) >= 10:
            thresh = float(args.when.split(":")[1]) if ":" in args.when else 3.0
            if pooled_b > thresh * float(np.median(buf)):
                rec["trigger_step"] = int(step)
                apply_knobs(bctl)
        buf.append(pooled_b)
        if len(buf) > 50:
            buf.pop(0)

    try:
        train(cfg, on_step=hook)
    except StopContinue:
        pass
    rec.update({"ckpt": args.ckpt, "start": start, "steps_run": args.steps,
                "supp_scale": args.supp_scale, "temp_mult": args.temp_mult,
                "when": args.when})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(rec, open(args.out, "w"), indent=1)
    print(f"[continue] {args.tag}: {len(rec['steps'])} records, "
          f"trigger={rec['trigger_step']} -> {args.out}")


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("drift")
    d.add_argument("--ckpt", required=True)
    d.add_argument("--data-dir", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--batches", type=int, default=16)
    d.add_argument("--eval-batch", type=int, default=8)
    d.add_argument("--alpha", type=float, default=1.0)
    d.add_argument("--temp-mult", type=float, default=1.0)
    d.add_argument("--lr-mult", type=float, default=1.0)
    d.add_argument("--batch-mult", type=float, default=1.0)
    d.add_argument("--accum", type=int, default=-1)
    d.add_argument("--perm-null", type=int, default=48)
    d.set_defaults(fn=run_drift)

    r = sub.add_parser("reparam")
    r.add_argument("--ckpt", required=True)
    r.add_argument("--data-dir", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--c", type=float, default=3.0)
    r.set_defaults(fn=run_reparam)

    c = sub.add_parser("continue")
    c.add_argument("--ckpt", required=True)
    c.add_argument("--data-dir", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--tag", required=True)
    c.add_argument("--scratch", default="/workspace/sd_runs")
    c.add_argument("--steps", type=int, default=1500)
    c.add_argument("--every", type=int, default=10)
    c.add_argument("--eval-batch", type=int, default=8)
    c.add_argument("--supp-scale", type=float, default=1.0)
    c.add_argument("--temp-mult", type=float, default=1.0)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--when", default="start",
                   help="start | trigger[:X] (pooled b above X times rolling median) | never")
    c.set_defaults(fn=run_continue)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
