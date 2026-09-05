"""Track every block's weight-matrix norms while training, every N steps.

Written for the amplification investigation (`docs/activation-amplification.md`),
where the tail turned out to be seeded by the MLP rather than by the bottleneck:
block 0's `mlp.fc2` Frobenius norm reached 560 in the j=128 run against ~56-60 in
the healthy ones. That was measured post hoc from checkpoints, which only exist
every 2000 steps -- this records it every 10.

Uses the same `on_step` hook and the same non-perturbation guarantees as
`probe_early_training.py`: called before the step's `zero_grad`, no backward of
its own, RNG and gate buffers restored.

Per probe step, for every block and every matrix in

    attn.qkv      d_model -> 3*d_model
    attn.proj     d_model -> d_model
    mlp.fc1       d_model -> d_mlp     (before the nonlinearity)
    mlp.fc2       d_mlp   -> d_model   (after the nonlinearity)
    bn.in_proj    d_model -> n_features    (the bottleneck encoder)
    bn.out_proj   n_features -> d_model    (the bottleneck decoder)

it records `fro`, `spec` (largest singular value -- the honest "how much can
this amplify" number), and the max/median row and column norms, which is what
distinguishes "the whole matrix grew" from "a few rows blew up". A little
activation context comes along per block (`x_norm`, `gain`, `score_max`) so the
weights can be read against what they produce.

    python analysis/probe_weight_norms.py \\
        --config /workspace/ckpt/dc_rout_soft_k32_j64/config.json \\
        --data-dir /workspace/data/tinystories \\
        --out /workspace/analysis/wnorm --steps 1000 --probe-every 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wsparse.bottleneck.controller import _PLACEMENT_ATTR, parse_placements  # noqa: E402
from wsparse.config import config_from_dict, load_config  # noqa: E402
from wsparse.data import TokenStream  # noqa: E402
from wsparse.train import train  # noqa: E402


class StopProbing(Exception):
    pass


# label -> how to reach the module from a block.  Order is the forward order,
# which is also how the viewer lists them.
MATRICES = (
    ("attn.qkv", lambda b: b.attn.qkv),
    ("attn.proj", lambda b: b.attn.proj),
    ("mlp.fc1", lambda b: b.mlp.fc1),
    ("mlp.fc2", lambda b: b.mlp.fc2),
)


@torch.no_grad()
def matrix_stats(W: torch.Tensor, spectral: bool) -> dict:
    """Norms of one weight matrix, shaped (out_features, in_features)."""
    Wf = W.detach().float()
    rows = Wf.norm(dim=1)          # one per output unit
    cols = Wf.norm(dim=0)          # one per input unit
    out = dict(
        fro=float(Wf.norm()),
        mean_abs=float(Wf.abs().mean()),
        max_row=float(rows.max()), med_row=float(rows.median()),
        max_col=float(cols.max()), med_col=float(cols.median()),
        shape=list(Wf.shape),
    )
    # A few enormous rows barely move the Frobenius norm of a big matrix, so
    # these ratios are the ones that separate "grew everywhere" from "grew in a
    # handful of directions".
    out["row_max_over_med"] = out["max_row"] / max(out["med_row"], 1e-30)
    out["col_max_over_med"] = out["max_col"] / max(out["med_col"], 1e-30)
    out["spec"] = float(torch.linalg.matrix_norm(Wf, ord=2)) if spectral else float("nan")
    return out


class WeightProbe:
    def __init__(self, steps, every, batch, positions, data_dir, split, offset,
                 seq_len, spectral):
        self.steps, self.every = int(steps), int(every)
        self.batch, self.positions = int(batch), int(positions)
        self.data_dir, self.split, self.offset = data_dir, split, int(offset)
        self.seq_len, self.spectral = int(seq_len), bool(spectral)
        self.rows, self.globals = [], []
        self.x = self.y = None
        self.t0 = time.time()

    def __call__(self, step, model, bottleneck, optimizer):
        if step > self.steps:
            raise StopProbing
        if step % self.every:
            return
        device = next(model.parameters()).device
        if self.x is None:
            stream = TokenStream(os.path.join(self.data_dir, f"{self.split}.bin"),
                                 self.seq_len, seed=0)
            self.x, self.y = stream.batch(self.batch, device,
                                          deterministic_offset=self.offset)

        layers = dict(bottleneck.layers) if bottleneck.enabled else {}
        placements = (parse_placements(bottleneck.cfg.placement)
                      if bottleneck.enabled else [])

        # ---- activation context, forward only ------------------------------ #
        snaps = [(m.gate.usage_ema.clone(), m.gate.usage_steps.clone(),
                  dict(m.gate._forward_diag), dict(m.gate._usage_diag),
                  dict(m.gate._grad_sink), m._reconstruction)
                 for _, m in bottleneck.layers] if bottleneck.enabled else []
        was_training = model.training
        model.eval()
        cap, handles = {}, []
        for li, blk in enumerate(model.blocks):
            handles.append(blk.mlp.register_forward_hook(
                lambda m, i, o, li=li: cap.__setitem__(("mlp", li), o.detach().float())))
            handles.append(blk.attn.register_forward_hook(
                lambda m, i, o, li=li: cap.__setitem__(("attn", li), o.detach().float())))
            if placements:
                mod = getattr(blk, _PLACEMENT_ATTR[placements[0]], None)
                if mod is not None and not isinstance(mod, torch.nn.Identity):
                    handles.append(mod.register_forward_pre_hook(
                        lambda m, i, li=li: cap.__setitem__(("x", li), i[0].detach().float())))
                    handles.append(mod.register_forward_hook(
                        lambda m, i, o, li=li: cap.__setitem__(("xh", li), o.detach().float())))
                    handles.append(mod.gate.register_forward_pre_hook(
                        lambda m, i, li=li: cap.__setitem__(("z", li), i[0].detach().float())))
        with torch.no_grad():
            _, loss = model(self.x, self.y)
        for h in handles:
            h.remove()

        T = self.positions
        for li, blk in enumerate(model.blocks):
            mats = list(MATRICES)
            if placements:
                mod = getattr(blk, _PLACEMENT_ATTR[placements[0]], None)
                if mod is not None and not isinstance(mod, torch.nn.Identity):
                    mats += [("bn.in_proj", lambda b, m=mod: m.in_proj),
                             ("bn.out_proj", lambda b, m=mod: m.out_proj)]
            for label, get in mats:
                st = matrix_stats(get(blk).weight, self.spectral)
                st.update(step=int(step), layer=li, matrix=label, ce=float(loss))
                self.rows.append(st)

            ctx = dict(step=int(step), layer=li, matrix="_act", ce=float(loss))
            mm = cap.get(("mlp", li)); aa = cap.get(("attn", li))
            if mm is not None:
                n = mm[:, :T].norm(dim=-1)
                ctx.update(mlp_out_med=float(n.median()), mlp_out_max=float(n.max()))
            if aa is not None:
                n = aa[:, :T].norm(dim=-1)
                ctx.update(attn_out_med=float(n.median()), attn_out_max=float(n.max()))
            if ("x", li) in cap:
                nx = cap[("x", li)][:, :T].norm(dim=-1).clamp_min(1e-30)
                ctx.update(x_norm_med=float(nx.median()), x_norm_max=float(nx.max()))
                if ("xh", li) in cap:
                    g = cap[("xh", li)][:, :T].norm(dim=-1) / nx
                    ctx.update(gain_med=float(g.median()), gain_max=float(g.max()))
                if ("z", li) in cap:
                    ctx.update(score_max=float(cap[("z", li)][:, :T].abs().max()))
            self.rows.append(ctx)

        self.globals.append(dict(
            step=int(step), ce=float(loss),
            tok_emb_fro=float(model.tok_emb.weight.detach().float().norm()),
            # None under pos_encoding="rope" -- there is no position table then
            pos_emb_fro=(float(model.pos_emb.weight.detach().float().norm())
                         if model.pos_emb is not None else float("nan")),
            norm_f=float(model.norm_f.weight.detach().float().norm()),
        ))

        for (_, mod), sn in zip(bottleneck.layers if bottleneck.enabled else [], snaps):
            g_ = mod.gate
            g_.usage_ema.copy_(sn[0]); g_.usage_steps.copy_(sn[1])
            g_._forward_diag, g_._usage_diag, g_._grad_sink = sn[2], sn[3], sn[4]
            mod._reconstruction = sn[5]
        model.train(was_training)
        cap.clear()
        if (step // self.every) % 20 == 0:
            f0 = [r for r in self.rows if r["step"] == step and r["layer"] == 0
                  and r["matrix"] == "mlp.fc2"]
            print(f"[wnorm] step {step:>5}  CE {float(loss):.4f}  "
                  f"L0 mlp.fc2 fro {f0[0]['fro'] if f0 else float('nan'):.2f}  "
                  f"[{time.time() - self.t0:.0f}s]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--probe-every", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--positions", type=int, default=64)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--split", default="val")
    ap.add_argument("--no-spectral", action="store_true",
                    help="skip the largest-singular-value computation (an SVD "
                         "per matrix per probe); everything else is cheap")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args()

    cfg = (config_from_dict(json.load(open(args.config)))
           if args.config.endswith(".json") else load_config(args.config))
    src = cfg.train.run_name
    for spec in args.overrides:
        path, raw = spec.split("=", 1)
        obj = cfg
        parts = path.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        cur = getattr(obj, parts[-1])
        new = (raw.lower() in ("1", "true", "yes") if isinstance(cur, bool)
               else int(raw) if isinstance(cur, int)
               else float(raw) if isinstance(cur, float) else raw)
        setattr(obj, parts[-1], new)
        print(f"[wnorm] override {path}: {cur!r} -> {new!r}")

    # Default to the original run's identity, like probe_early_training does.
    # A config-derived name (mode/k/j) loses the run name and its markers (md,
    # selcorr, machine) and collides across sweeps -- learned the hard way when
    # five uk_* datasets landed as "wnorm_hard_k32_j64"-style names.
    original = cfg.train.run_name or os.path.basename(
        os.path.dirname(os.path.abspath(args.config)))
    name = args.name or f"wnorm_{original}"
    # max_steps untouched: lr and temperature schedules are defined over it.
    cfg.train.run_name = name
    cfg.train.out_dir = os.path.join(args.out, "tb_runs")
    cfg.train.resume = ""
    cfg.train.sample_every_steps = 10 ** 9
    cfg.train.validate_every_steps = 10 ** 9
    cfg.train.tensorboard = False
    cfg.train.wandb_project = ""

    probe = WeightProbe(args.steps, args.probe_every, args.batch, args.positions,
                        args.data_dir, args.split, args.offset, cfg.data.seq_len,
                        spectral=not args.no_spectral)
    print(f"[wnorm] {name}: k={bn.k} j={bn.j} {bn.surrogate_mode} {bn.placement} "
          f"steps 0..{args.steps} every {args.probe_every} "
          f"spectral={not args.no_spectral}")
    try:
        train(cfg, on_step=probe)
    except StopProbing:
        pass

    os.makedirs(args.out, exist_ok=True)
    out = dict(
        name=name, kind="wnorm", source_run=src, overrides=list(args.overrides),
        k=int(bn.k), j=int(bn.j), surrogate_mode=bn.surrogate_mode,
        placement=bn.placement, init_mode=bn.init_mode,
        n_features=int(bn.n_features), n_layers=int(cfg.model.n_layers),
        d_model=int(cfg.model.d_model), d_mlp=int(cfg.model.d_mlp),
        weight_decay=float(cfg.train.weight_decay),
        steps=int(args.steps), probe_every=int(args.probe_every),
        seed=int(cfg.train.seed), max_steps=int(cfg.train.max_steps),
        rows=probe.rows, globals=probe.globals,
        command=" ".join(sys.argv),
    )
    with open(os.path.join(args.out, f"{name}.json"), "w") as f:
        json.dump(out, f)
    print(f"[wnorm] wrote {name}.json ({len(probe.rows)} rows)")


if __name__ == "__main__":
    main()
