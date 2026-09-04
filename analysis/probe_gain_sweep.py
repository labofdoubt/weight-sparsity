"""Measure the bottleneck's gain while training, for a k / j sweep.

Companion to ``probe_early_training.py``: same ``on_step`` hook, same
non-perturbation guarantees, but it records a handful of scalars per bottleneck
instead of the full score tensor, so a whole sweep costs megabytes and a few
minutes per config rather than gigabytes.

What it records, per probe step and per bottleneck, over one fixed held-out
batch:

``gain``            median and p99 of ``||x_hat|| / ||x||`` -- the multiplier the
                    residual stream picks up at this bottleneck.  With a stream
                    placement (``residual`` / ``residual_out``) nothing routes
                    around it, so the stream norm grows as the product of these.
``enc``             ``||z|| / ||x||``  -- how much norm the encoder produces.
``keep``            ``||z_m|| / ||z||`` -- the fraction TopK lets through.
``cos``             ``cos(x, x_hat)``  -- whether the map preserves direction or
                    substitutes a new one; a low value with a high gain means the
                    bottleneck is injecting norm rather than reconstructing.
``x_norm``          median ``||x||``, i.e. the stream itself.
``w_in`` / ``w_out``  Frobenius norms, to separate "the weights grew" from
                    "the selection geometry changed".

    python interpretability/probe_gain_sweep.py \
        --config /workspace/ckpt/dc_rout_hard_k32_selcorr/config.json \
        --data-dir /workspace/data/tinystories --out /workspace/analysis/gain \
        --steps 300 --probe-every 10 --set activation_bottleneck.k=64
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

from wsparse.config import config_from_dict, load_config  # noqa: E402
from wsparse.data import TokenStream  # noqa: E402
from wsparse.train import train  # noqa: E402


class StopProbing(Exception):
    pass


class GainProbe:
    def __init__(self, steps, every, batch, positions, data_dir, split, offset, seq_len):
        self.steps, self.every = int(steps), int(every)
        self.batch, self.positions = int(batch), int(positions)
        self.data_dir, self.split, self.offset = data_dir, split, int(offset)
        self.seq_len = int(seq_len)
        self.rows = []
        self.x = self.y = None
        self.t0 = time.time()

    def __call__(self, step, model, bottleneck, optimizer):
        if step > self.steps:
            raise StopProbing
        if step % self.every:
            return
        layers = bottleneck.layers if bottleneck.enabled else []
        device = next(model.parameters()).device
        if self.x is None:
            stream = TokenStream(os.path.join(self.data_dir, f"{self.split}.bin"),
                                 self.seq_len, seed=0)
            self.x, self.y = stream.batch(self.batch, device,
                                          deterministic_offset=self.offset)

        # Forward only, under no_grad: the surrogate is irrelevant to the gain,
        # and no gradient means nothing to leak into the optimiser.  Buffers the
        # gate mutates are still restored, so the run is unaffected either way.
        snaps = [(m.gate.usage_ema.clone(), m.gate.usage_steps.clone(),
                  dict(m.gate._forward_diag), dict(m.gate._usage_diag),
                  dict(m.gate._grad_sink), m._reconstruction)
                 for _, m in layers]
        was_training = model.training
        model.eval()

        cap, handles = {}, []
        for li, (_, mod) in enumerate(layers):
            handles += [
                mod.register_forward_pre_hook(
                    lambda m, i, li=li: cap.__setitem__(("x", li), i[0].detach().float())),
                mod.register_forward_hook(
                    lambda m, i, o, li=li: cap.__setitem__(("xh", li), o.detach().float())),
                mod.gate.register_forward_pre_hook(
                    lambda m, i, li=li: cap.__setitem__(("z", li), i[0].detach().float())),
                mod.gate.register_forward_hook(
                    lambda m, i, o, li=li: cap.__setitem__(("zm", li), o.detach().float())),
            ]
        with torch.no_grad():
            _, loss = model(self.x, self.y)
        for h in handles:
            h.remove()

        if not layers:
            # No bottleneck installed (activation_bottleneck.enabled=false):
            # there is no gain to measure, but the CE trajectory is still the
            # point of such a run, so record that.
            self.rows.append(dict(step=int(step), layer=-1, label="none",
                                  ce=float(loss)))
            model.train(was_training)
            if (step // self.every) % 10 == 0:
                print(f"[gain] step {step:>5}  CE {float(loss):.4f}  (no bottleneck)  "
                      f"[{time.time()-self.t0:.0f}s]")
            return

        T = self.positions
        for li, (label, mod) in enumerate(layers):
            x_, xh, z_, zm = (cap[(t, li)][:, :T] for t in ("x", "xh", "z", "zm"))
            nx = x_.norm(dim=-1).clamp_min(1e-30)
            nz = z_.norm(dim=-1).clamp_min(1e-30)
            g = (xh.norm(dim=-1) / nx).flatten()
            cos = torch.nn.functional.cosine_similarity(
                x_.reshape(-1, x_.shape[-1]), xh.reshape(-1, xh.shape[-1]), dim=-1)
            self.rows.append(dict(
                step=int(step), layer=li, label=label,
                ce=float(loss),
                gain=float(g.median()), gain_p99=float(torch.quantile(g, 0.99)),
                enc=float((nz / nx).median()),
                keep=float((zm.norm(dim=-1) / nz).median()),
                cos=float(cos.median()),
                x_norm=float(nx.median()),
                score_max=float(z_.abs().max()),
                w_in=float(mod.in_proj.weight.norm()),
                w_out=float(mod.out_proj.weight.norm()),
            ))

        for (_, mod), s in zip(layers, snaps):
            g_ = mod.gate
            g_.usage_ema.copy_(s[0]); g_.usage_steps.copy_(s[1])
            g_._forward_diag, g_._usage_diag, g_._grad_sink = s[2], s[3], s[4]
            mod._reconstruction = s[5]
        model.train(was_training)
        cap.clear()
        if (step // self.every) % 10 == 0:
            print(f"[gain] step {step:>5}  CE {float(loss):.4f}  "
                  f"gain L0 {self.rows[-len(layers)]['gain']:.3f} "
                  f"L{len(layers)-1} {self.rows[-1]['gain']:.3f}  "
                  f"stream ||x|| L{len(layers)-1} {self.rows[-1]['x_norm']:.4g}  "
                  f"[{time.time()-self.t0:.0f}s]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True, help="directory for <name>.json")
    ap.add_argument("--name", default=None)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--probe-every", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--positions", type=int, default=64)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--split", default="val")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--tb-dir", default=None,
                    help="if set, train() writes TensorBoard logs there (e.g. "
                         "/workspace/runs/lens_1) so the run shows up in TB; "
                         "default keeps the sweep out of the TB index")
    ap.add_argument("--init-only", action="store_true",
                    help="measure at initialisation and stop, no training at all")
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
        print(f"[gain] override {path}: {cur!r} -> {new!r}")

    bn = cfg.activation_bottleneck
    name = args.name or (f"{'init' if args.init_only else 'gain'}_"
                         f"{bn.surrogate_mode}_k{bn.k}_j{bn.j}")
    steps = 0 if args.init_only else args.steps
    # max_steps untouched: lr and temperature schedules are defined over it.
    cfg.train.run_name = f"gainsweep_{name}"
    cfg.train.out_dir = args.tb_dir or os.path.join(args.out, "tb_runs")
    cfg.train.resume = ""
    cfg.train.sample_every_steps = 10 ** 9
    cfg.train.validate_every_steps = 10 ** 9      # nothing here needs val
    cfg.train.tensorboard = bool(args.tb_dir)
    cfg.train.wandb_project = ""

    probe = GainProbe(steps, args.probe_every, args.batch, args.positions,
                      args.data_dir, args.split, args.offset, cfg.data.seq_len)
    print(f"[gain] {name}: k={bn.k} j={bn.j} {bn.surrogate_mode} {bn.placement} "
          f"init={bn.init_mode} steps 0..{steps}")
    try:
        train(cfg, on_step=probe)
    except StopProbing:
        pass

    os.makedirs(args.out, exist_ok=True)
    out = dict(name=name, source_run=src, overrides=list(args.overrides),
               k=int(bn.k), j=int(bn.j), surrogate_mode=bn.surrogate_mode,
               placement=bn.placement, init_mode=bn.init_mode,
               n_features=int(bn.n_features), n_layers=int(cfg.model.n_layers),
               steps=steps, probe_every=args.probe_every,
               seed=int(cfg.train.seed), rows=probe.rows,
               command=" ".join(sys.argv))
    with open(os.path.join(args.out, f"{name}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[gain] wrote {name}.json ({len(probe.rows)} rows)")


if __name__ == "__main__":
    main()
