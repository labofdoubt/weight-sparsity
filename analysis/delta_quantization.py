"""Is the Pi_approx roughness in K a real feature or a bf16 artifact?

delta = (s_(K-4) - s_(K+4)) / 8 is a difference of nearby scores.  In bf16 the
score itself carries ~8 mantissa bits, so a difference of two nearby values is
quantized to multiples of the bf16 spacing at that magnitude -- and the spacing
doubles at every binade (power of two).  Pi_approx divides by delta, so it
inherits that lattice; exact Pi sums over K+J candidates and does not.

Runs the same forward in bf16 and in float32 and reports, per K:
  the number of distinct delta values, the lattice step, the bf16 spacing at b,
  and both Pi_approx and exact Pi.
"""
import argparse
import numpy as np
import torch

from wsparse.bottleneck import apply_activation_bottleneck
from wsparse.config import load_config
from wsparse.data import build_streams, load_meta
from wsparse.decouple import md_init_
from wsparse.model import build_model
from wsparse.utils import autocast_context, resolve_device, resolve_dtype, set_seed

M_SP = 4

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/mdinit/rbk.yaml")
ap.add_argument("--data-dir", default="/workspace/data/tinystories")
ap.add_argument("--block", type=int, default=7)
ap.add_argument("--k-values", type=int, nargs="*",
                default=[32, 128, 256, 320, 352, 384, 416, 448, 512])
ap.add_argument("--seqs", type=int, default=16)
ap.add_argument("--post-norm", action="store_true")
args = ap.parse_args()

common = [f"data.data_dir={args.data_dir}", "model.decouple=true",
          "train.run_name=dq"]
if args.post_norm:
    common.append("activation_bottleneck.post_norm=true")

base = load_config(args.config, list(common))
base.model.vocab_size = int(load_meta(base.data.data_dir)["vocab_size"])
device = resolve_device(base.train.device)
N = base.activation_bottleneck.n_features


def build(k, j, dtype_name):
    set_seed(1337)
    cfg = load_config(args.config, list(common) + [
        f"activation_bottleneck.k={k}", f"activation_bottleneck.j={j}",
        f"train.dtype={dtype_name}"])
    cfg.model.vocab_size = base.model.vocab_size
    m = build_model(cfg.model)
    bctl = apply_activation_bottleneck(m, cfg.activation_bottleneck,
                                       max_steps=cfg.train.max_steps)
    m.to(device)
    md_init_(m, cfg.model.decouple_gains)
    return m, bctl, cfg


ref, _, _ = build(32, 512, "bfloat16")
init_sd = {k: v.detach().clone() for k, v in ref.state_dict().items()}
del ref
train_stream, _ = build_streams(base.data, seed=1337)
x, y = train_stream.batch(args.seqs, device, deterministic_offset=777)

print(f"block {args.block}, post_norm={args.post_norm}, {x.numel()} tokens")
print(f"{'K':>5} {'dtype':>9} {'b_med':>8} {'delta_med':>10} {'uniq_d':>7} "
      f"{'d_step':>9} {'eps(b)':>9} {'Pi_apx':>8} {'Pi':>8}")
for K in args.k_values:
    for dt in ("bfloat16", "float32"):
        model, bctl, cfg = build(K, 512, dt)
        model.load_state_dict(init_sd, strict=True)
        model.train()
        grab = {}
        h = bctl.layers[args.block][1].gate.register_forward_pre_hook(
            lambda m, inp: grab.__setitem__("z", inp[0].detach().float()))
        with torch.no_grad(), autocast_context(device, resolve_dtype(dt, device)):
            model(x, y)
        h.remove()
        z = grab["z"].reshape(-1, N).cpu().numpy()
        ss = -np.sort(-np.abs(z).astype(np.float32), axis=1)
        b = ss[:, K]
        delta = (ss[:, K - 1 - M_SP] - ss[:, K - 1 + M_SP]) / (2.0 * M_SP)
        pos = delta[delta > 0]
        uniq = np.unique(np.round(pos, 9))
        step = float(np.min(np.diff(uniq))) if uniq.size > 1 else float("nan")
        bm = float(np.median(b))
        # bf16 spacing at |b|: 8 explicit mantissa bits
        eps_b = 2.0 ** (np.floor(np.log2(bm)) - 8)
        pa = float(np.median(b ** 2 / (4.0 * 1.0 * np.maximum(delta, 1e-12))))
        sc = ss[:, :K + 512] if K + 512 <= N else ss[:, :N]
        kap = np.exp(-np.abs(sc - b[:, None]) / 1.0) / 2.0
        Z = np.maximum(kap.sum(1, dtype=np.float64), 1e-30)[:, None]
        S2 = (kap.astype(np.float64) ** 2).sum(1)[:, None]
        Pi = (((sc * kap).astype(np.float64) ** 2) *
              ((1 - kap / Z) ** 2 + (S2 - kap.astype(np.float64) ** 2) / Z ** 2)).sum(1)
        print(f"{K:>5} {dt:>9} {bm:>8.4f} {float(np.median(delta)):>10.6f} "
              f"{uniq.size:>7} {step:>9.6f} {eps_b:>9.6f} {pa:>8.1f} "
              f"{float(np.median(Pi)):>8.1f}")
        del model, bctl
        if device.type == "cuda":
            torch.cuda.empty_cache()
