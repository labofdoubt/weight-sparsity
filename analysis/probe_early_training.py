"""Train a run from scratch and measure the bottleneck every few steps.

The early phase has no checkpoints -- ``checkpoint_every_steps`` is 2000 -- and
keeping 100 of them would cost ~140 GB per run.  So instead of saving weights
this re-runs training with the *same config and seed* and probes the live model
on a fixed held-out batch every ``--probe-every`` steps, keeping only the
measurements.  With the same seed and the same config the trajectory is the same
one the original run took, so step 250 here is step 250 there.

Three tensors are recorded per probe, all shaped
``(probe, bottleneck, sequence, position, n_features)``:

``score``     the pre-TopK ranking signal ``r`` the gate sorts on (signed; the
              viewer takes the magnitude), i.e. the gate's *input*.
``g_ztilde``  ``dL/d~z`` -- the gradient arriving at the gate's *output*, the
              sparse code that reaches the decoder.  Dense: it is generally
              non-zero even for a neuron TopK did not select, because it is
              ``W_out^T dL/dy`` and does not pass through the mask.
``g_z``       ``dL/dz`` -- the gradient at the gate's *input*, i.e. after the
              LapSum surrogate backward.  Exactly zero outside Top(K+J); inside
              the J band it is the surrogate's doing.

Comparing the two answers the question the probe exists for: whether the pool's
gradients point the same way, and whether the surrogate passes that alignment
through to the encoder or destroys it.

Two things this is careful about, both of which would quietly invalidate the
result:

* **The probe must not train the model.**  It runs its own backward, so it is
  invoked from ``train``'s ``on_step`` hook *before* that step's
  ``optimizer.zero_grad(set_to_none=True)``, which is then what guarantees the
  probe's gradients can never reach the optimizer.  It also zeroes them itself,
  restores the RNG state, and restores every buffer the gate mutates while
  measuring (the usage EMA and the diagnostics dicts), so training proceeds
  exactly as it would have.
* **The schedules must not be rescaled.**  ``lr_at`` decays cosine over
  ``train.max_steps`` and the bottleneck temperature anneals over
  ``max_steps - warmup``, so lowering ``max_steps`` to stop early would compress
  the whole decay into the probed window and reproduce nothing.  ``max_steps``
  is therefore left at the original value and the run is stopped by raising
  :class:`StopProbing` from the hook.

    python analysis/probe_early_training.py \
        --config /workspace/ckpt/dc_rout_soft_k32_j64/config.json \
        --data-dir /workspace/data/tinystories \
        --out-dir /workspace/analysis/probe \
        --steps 1000 --probe-every 10 --batch 4 --positions 64
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
    """Raised from the hook to end training without rescaling any schedule."""


class Probe:
    """``on_step`` hook: measures scores and both gradients, writes to memmaps."""

    def __init__(self, out_dir, run, steps, every, batch, positions, data_dir,
                 split, offset, seq_len, n_features, k, j, cfg_tree):
        self.out_dir, self.run = out_dir, run
        self.steps, self.every = int(steps), int(every)
        self.batch, self.positions = int(batch), int(positions)
        self.n_probes = self.steps // self.every + 1
        self.n_features, self.k, self.j = int(n_features), int(k), int(j)
        self.cfg_tree = cfg_tree
        self.split, self.offset, self.seq_len = split, int(offset), int(seq_len)
        self.data_dir = data_dir
        self.arrays = None
        self.probe_steps, self.probe_ce = [], []
        self.labels = None
        self.x = self.y = None
        self.t0 = time.time()

    # ---- setup ------------------------------------------------------------- #
    def _batch(self, device):
        """The one held-out batch, identical at every probe and every run.

        deterministic_offset makes it independent of the RNG, and it is drawn
        from a stream this class owns, so it never advances the training stream.
        """
        stream = TokenStream(
            os.path.join(self.data_dir, f"{self.split}.bin"), self.seq_len, seed=0
        )
        return stream.batch(self.batch, device, deterministic_offset=self.offset)

    def _alloc(self, n_layers):
        os.makedirs(self.out_dir, exist_ok=True)
        shape = (self.n_probes, n_layers, self.batch, self.positions, self.n_features)
        # float32, not float16: the gradients run to ~1e-8, and float16's
        # smallest normal is ~6e-5, so half precision would flush them to zero.
        self.arrays = {
            name: np.lib.format.open_memmap(
                os.path.join(self.out_dir, f"{self.run}.{name}.npy"),
                mode="w+", dtype=np.float32, shape=shape,
            )
            for name in ("score", "g_ztilde", "g_z")
        }
        gb = 3 * np.prod(shape) * 4 / 2**30
        print(f"[probe] allocated 3 x {shape} float32 ({gb:.2f} GiB total)")

    # ---- the hook ---------------------------------------------------------- #
    def __call__(self, step, model, bottleneck, optimizer):
        if step > self.steps:
            raise StopProbing
        if step % self.every:
            return
        layers = bottleneck.layers
        if self.arrays is None:
            self.labels = [lbl for lbl, _ in layers]
            self._alloc(len(layers))
        device = next(model.parameters()).device
        if self.x is None:
            self.x, self.y = self._batch(device)

        # ---- snapshot everything the measurement would perturb ------------- #
        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        snaps = []
        for _, mod in layers:
            g = mod.gate
            snaps.append((
                g.usage_ema.clone(), g.usage_steps.clone(),
                dict(g._forward_diag), dict(g._usage_diag), dict(g._grad_sink),
                mod._reconstruction,
            ))
        was_training = model.training

        # surrogate_active() is False in eval and under no_grad, so a probe run
        # in eval mode would measure the *hard* mask's gradient, not the
        # surrogate's -- the whole point of g_z would be lost.
        model.train()
        cap: dict = {}
        handles = []

        def mk_pre(li):
            def pre(m, inputs):
                a = inputs[0]
                cap[("score", li)] = a.detach().float()
                if a.requires_grad:
                    a.register_hook(
                        lambda g, li=li: cap.__setitem__(("g_z", li), g.detach().float())
                    )
                return None
            return pre

        def mk_post(li):
            def post(m, inputs, output):
                if output.requires_grad:
                    output.register_hook(
                        lambda g, li=li: cap.__setitem__(("g_ztilde", li), g.detach().float())
                    )
                return None
            return post

        for li, (_, mod) in enumerate(layers):
            handles.append(mod.gate.register_forward_pre_hook(mk_pre(li)))
            handles.append(mod.gate.register_forward_hook(mk_post(li)))

        # No autocast: training applies these in bfloat16, but the measurement
        # is reported in float32 so small gradients are not quantized away.
        _, loss = model(self.x, self.y)
        loss.backward()
        for h in handles:
            h.remove()

        p = step // self.every
        T, N = self.positions, self.n_features
        for name in ("score", "g_ztilde", "g_z"):
            for li in range(len(layers)):
                key = (name, li)
                if key not in cap:
                    raise RuntimeError(
                        f"step {step}: {name} never captured for bottleneck {li} -- "
                        "the backward did not reach it"
                    )
                v = cap[key]
                if v.shape != (self.batch, self.seq_len, N):
                    raise RuntimeError(
                        f"step {step}: {name}[{li}] has shape {tuple(v.shape)}, "
                        f"expected {(self.batch, self.seq_len, N)}"
                    )
                self.arrays[name][p, li] = v[:, :T, :].cpu().numpy()
        self.probe_steps.append(int(step))
        self.probe_ce.append(float(loss.detach()))

        # ---- restore ------------------------------------------------------- #
        optimizer.zero_grad(set_to_none=True)   # belt; train() also braces
        for (_, mod), s in zip(layers, snaps):
            g = mod.gate
            g.usage_ema.copy_(s[0])
            g.usage_steps.copy_(s[1])
            g._forward_diag, g._usage_diag, g._grad_sink = s[2], s[3], s[4]
            mod._reconstruction = s[5]
        model.train(was_training)
        torch.set_rng_state(rng_cpu)
        if rng_cuda is not None:
            torch.cuda.set_rng_state_all(rng_cuda)
        cap.clear()

        if p % 10 == 0 or step == 0:
            print(f"[probe] step {step:>5} ({p + 1}/{self.n_probes})  "
                  f"held-out CE {float(loss.detach()):.4f}  "
                  f"[{time.time() - self.t0:.0f}s]")

    # ---- output ------------------------------------------------------------ #
    def finish(self, token_ids, extra):
        for a in (self.arrays or {}).values():
            a.flush()
        meta = dict(
            run=self.run, kind="probe",
            probe_steps=self.probe_steps, probe_ce=self.probe_ce,
            steps=self.probe_steps,          # the viewer's generic step axis
            batch_ce=self.probe_ce,          # ... and its generic loss series
            layer_labels=self.labels,
            n_features=self.n_features, k=self.k, j=self.j,
            batch=self.batch, n_pos=self.positions, seq_len=self.seq_len,
            split=self.split, offset=self.offset,
            probe_every=self.every, probe_max_step=self.steps,
            token_ids=token_ids,
            arrays={n: f"{self.run}.{n}.npy" for n in ("score", "g_ztilde", "g_z")},
            **extra,
        )
        with open(os.path.join(self.out_dir, f"{self.run}.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[probe] wrote {self.run}.json with {len(self.probe_steps)} probes")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True,
                    help="a run's config.json (or a .yaml config)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-name", default=None, help="default: probe_<original run>")
    ap.add_argument("--steps", type=int, default=1000, help="last step probed")
    ap.add_argument("--probe-every", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4, help="held-out sequences")
    ap.add_argument("--positions", type=int, default=64)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--split", default="val", choices=("val", "train"))
    ap.add_argument("--tb-dir", default=None,
                    help="where train() writes its own logs (default <out-dir>/tb_runs)")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="a.b.c=VALUE",
                    help="override a config field before training, e.g. "
                         "--set activation_bottleneck.k=64. Repeatable. Lets a "
                         "variant that was never trained be probed from an "
                         "existing run's config without editing anything.")
    args = ap.parse_args()

    if args.config.endswith(".json"):
        cfg = config_from_dict(json.load(open(args.config)))
    else:
        cfg = load_config(args.config)
    original = cfg.train.run_name

    # ---- config overrides, applied before anything reads the config -------- #
    for spec in args.overrides:
        if "=" not in spec:
            raise SystemExit(f"--set expects a.b.c=VALUE, got {spec!r}")
        path, raw = spec.split("=", 1)
        obj = cfg
        parts = path.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        leaf = parts[-1]
        if not hasattr(obj, leaf):
            raise SystemExit(f"--set: no config field {path!r}")
        cur = getattr(obj, leaf)
        # Coerce to the existing field's type so a string never silently
        # replaces an int and changes behaviour downstream.
        if isinstance(cur, bool):
            new = raw.lower() in ("1", "true", "yes")
        elif isinstance(cur, int):
            new = int(raw)
        elif isinstance(cur, float):
            new = float(raw)
        else:
            new = raw
        setattr(obj, leaf, new)
        print(f"[probe] override {path}: {cur!r} -> {new!r}")

    run = args.run_name or f"probe_{original}"

    # max_steps is deliberately NOT touched: lr_at decays cosine over it and the
    # temperature anneals over it, so shortening it would change the trajectory
    # being reproduced.  Training stops via StopProbing instead.
    cfg.data.data_dir = args.data_dir
    cfg.train.run_name = run
    cfg.train.out_dir = args.tb_dir or os.path.join(args.out_dir, "tb_runs")
    cfg.train.resume = ""            # never resume: this must start from init
    cfg.train.sample_every_steps = 10 ** 9   # sampling consumes RNG
    cfg.train.wandb_project = ""

    if cfg.model.dropout or cfg.model.attn_dropout:
        print(f"[probe] WARNING dropout={cfg.model.dropout} "
              f"attn_dropout={cfg.model.attn_dropout}: the probe forward runs in "
              f"train() mode (the surrogate needs it), so its measurement is "
              f"stochastic and the restored RNG will not undo the difference.")

    print(f"[probe] {original} -> {run}: steps 0..{args.steps} every "
          f"{args.probe_every}, batch {args.batch} x {args.positions} pos from "
          f"{args.split}.bin, max_steps left at {cfg.train.max_steps}")

    probe = Probe(
        out_dir=args.out_dir, run=run, steps=args.steps, every=args.probe_every,
        batch=args.batch, positions=args.positions, data_dir=args.data_dir,
        split=args.split, offset=args.offset, seq_len=int(cfg.data.seq_len),
        n_features=int(cfg.activation_bottleneck.n_features),
        k=int(cfg.activation_bottleneck.k), j=int(cfg.activation_bottleneck.j),
        cfg_tree=cfg.to_dict(),
    )
    try:
        train(cfg, on_step=probe)
    except StopProbing:
        print(f"[probe] stopped at step {args.steps} as planned")

    bn = cfg.activation_bottleneck
    probe.finish(
        token_ids=probe.x[:, : args.positions].cpu().numpy().tolist(),
        extra=dict(
            source_run=original, source_config=args.config,
            selection_mode=bn.selection_mode, placement=bn.placement,
            surrogate_mode=bn.surrogate_mode, n_layers=int(cfg.model.n_layers),
            seed=int(cfg.train.seed), max_steps=int(cfg.train.max_steps),
            temperature=dict(
                scale_mode=bn.temperature_scale_mode, schedule=bn.temperature_schedule,
                start=bn.temperature_start, end=bn.temperature_end,
                warmup_steps=bn.temperature_warmup_steps,
                anneal_steps=bn.temperature_anneal_steps, power=bn.temperature_power,
                fixed=bn.fixed_temperature, max_steps=int(cfg.train.max_steps),
            ),
            torch_version=torch.__version__,
            command=" ".join(sys.argv),
            overrides=list(args.overrides),
            k_config=int(bn.k), j_config=int(bn.j),
        ),
    )


if __name__ == "__main__":
    main()
