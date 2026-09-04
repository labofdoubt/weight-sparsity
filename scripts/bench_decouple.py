"""Overhead of magnitude-direction decoupling: full training-step timing.

The paper stores fused weights, so the forward/backward is unchanged by
construction -- the entire cost sits in the optimizer step (unfuse, split the
gradient, project, refuse; all elementwise).  So unlike bench_pos_encoding.py
this times the whole step: forward + backward + grad-clip + optimizer.step.

    python scripts/bench_decouple.py \
        --config /workspace/ckpt/dc_rout_soft_k32_j64/config.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wsparse.bottleneck import apply_activation_bottleneck  # noqa: E402
from wsparse.config import config_from_dict, load_config  # noqa: E402
from wsparse.decouple import build_decoupled_optimizer, md_init_  # noqa: E402
from wsparse.model import build_model  # noqa: E402
from wsparse.optim import build_optimizer  # noqa: E402
from wsparse.utils import autocast_context, resolve_device, resolve_dtype  # noqa: E402


def make(cfg, decouple: bool, device):
    c = copy.deepcopy(cfg)
    c.model.pos_encoding = "rope"        # both sides on rope: isolate decoupling
    c.model.decouple = decouple
    torch.manual_seed(0)
    model = build_model(c.model).to(device)
    if c.activation_bottleneck.enabled:
        apply_activation_bottleneck(model, c.activation_bottleneck,
                                    max_steps=c.train.max_steps)
        model.to(device)
    if decouple:
        md_init_(model, c.model.decouple_gains)
        opt = build_decoupled_optimizer(model, c.train,
                                        gain_mode=c.model.decouple_gains)
    else:
        opt = build_optimizer(model, c.train, c.sparsity, mask_param_ids=set())
    return model, opt, c


def bench(model, opt, cfg, x, y, device, dtype, iters, warmup):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    step_t, opt_t = [], []
    o0 = torch.cuda.Event(enable_timing=True)
    for i in range(warmup + iters):
        opt.zero_grad(set_to_none=True)
        start.record()
        with autocast_context(device, dtype):
            _, loss = model(x, y)
        loss.backward()
        if cfg.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
        o0.record()
        opt.step()
        end.record()
        torch.cuda.synchronize()
        if i >= warmup:
            step_t.append(start.elapsed_time(end))
            opt_t.append(o0.elapsed_time(end))
    st, ot = torch.tensor(step_t), torch.tensor(opt_t)
    return float(st.median()), float(ot.median())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--no-bottleneck", action="store_true")
    a = ap.parse_args()
    cfg = (config_from_dict(json.load(open(a.config)))
           if a.config.endswith(".json") else load_config(a.config))
    if a.no_bottleneck:
        cfg.activation_bottleneck.enabled = False
    device = resolve_device(cfg.train.device)
    dtype = resolve_dtype(cfg.train.dtype, device)
    B, T = int(cfg.train.micro_batch_size), int(cfg.data.seq_len)
    x = torch.randint(0, cfg.model.vocab_size, (B, T), device=device)
    y = torch.randint(0, cfg.model.vocab_size, (B, T), device=device)
    print(f"shape {B}x{T}, {dtype}, rope both sides, "
          f"bottleneck={'on' if cfg.activation_bottleneck.enabled else 'off'}")
    print(f"{'optimizer':>14} | {'full step med ms':>16} | {'opt.step med ms':>15}")
    out = {}
    for decouple in (False, True):
        model, opt, c = make(cfg, decouple, device)
        s, o = bench(model, opt, c, x, y, device, dtype, a.iters, a.warmup)
        out[decouple] = (s, o)
        name = "AdamW+MD" if decouple else "AdamW"
        print(f"{name:>14} | {s:16.3f} | {o:15.3f}")
        del model, opt
        torch.cuda.empty_cache()
    print(f"\nMD / baseline: full step {out[True][0]/out[False][0]:.4f}x   "
          f"optimizer step {out[True][1]/out[False][1]:.4f}x")


if __name__ == "__main__":
    main()
