"""GPU preflight of a laplace_policy config, run before a campaign is launched.

Builds the model of ``--config`` exactly as ``train()`` does (seed, bottleneck,
MD re-initialization), reads real training windows and checks the sampled
training forward -- ``train()`` mode, the trainer's collector and loss:

  1. every bottleneck gate records a density term in the forward (all blocks);
  2. the forward is exactly K-sparse per token and the kept indices are the K
     largest noisy scores of the clean Top(K+J) pool (from the records' pool
     indices and realized noisy scores);
  3. the CE, the support term and every parameter gradient are finite, and
     the selection term alone reaches the encoder of every block;
  4. a PolicyTrainingState saved after some noise was consumed and restored
     into a fresh state (other seed) reproduces the next sampled supports bit
     for bit on the same inputs, with the baseline and update count restored
     (the resume guarantee of docs/laplace-policy-topk.md section 6, on the
     device rather than on CPU).

Prints one line per check and exits non-zero on the first failure.

    python scripts/policy_preflight.py --config configs/policy_cr/<run>.yaml \
        [--data.data_dir=/workspace/data/tinystories]
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wsparse.bottleneck import apply_activation_bottleneck  # noqa: E402
from wsparse.bottleneck.laplace_policy import (PolicyCollector,  # noqa: E402
                                               PolicyForwardSettings, PolicyTrainingState,
                                               policy_backward_loss, policy_gates)
from wsparse.config import load_config  # noqa: E402
from wsparse.data import build_streams, load_meta  # noqa: E402
from wsparse.model import build_model  # noqa: E402
from wsparse.utils import autocast_context, resolve_dtype, set_seed  # noqa: E402


def fail(msg: str) -> None:
    print(f"FAIL  {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"ok    {msg}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--micro", type=int, default=0, help="batch rows (0: the config's micro batch)")
    args, unknown = ap.parse_known_args()
    cfg = load_config(args.config, list(unknown))
    bn = cfg.activation_bottleneck
    if bn.surrogate_mode != "laplace_policy":
        fail(f"{args.config} is not a laplace_policy config ({bn.surrogate_mode})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(cfg.train.dtype, device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    set_seed(cfg.train.seed)
    meta = load_meta(cfg.data.data_dir)
    cfg.model.vocab_size = int(meta["vocab_size"])
    train_stream, _ = build_streams(cfg.data, seed=cfg.train.seed)
    model = build_model(cfg.model).to(device)
    ctl = apply_activation_bottleneck(model, bn, max_steps=cfg.train.max_steps)
    model.to(device)
    if cfg.model.decouple:
        from wsparse.decouple import md_init_
        md_init_(model, cfg.model.decouple_gains)
    model.train()
    ctl.set_step(0)
    gates = policy_gates(model)
    names = {id(layer.gate): name for name, layer in ctl.layers}
    k, j = bn.k, bn.j
    print(f"[preflight] {cfg.train.run_name}: K={k} J={j} T0={bn.temperature:g} "
          f"({bn.policy_temperature_mode}/{bn.policy_temperature_schedule}) "
          f"code_residual={bn.code_residual} post_norm={bn.post_norm} gates={len(gates)} "
          f"device={device}")
    if len(gates) != cfg.model.n_layers:
        fail(f"{len(gates)} laplace_policy gates for {cfg.model.n_layers} blocks")

    outputs = {}

    def hook(mod, inp, out):
        outputs[id(mod)] = out.detach()

    handles = [g.register_forward_hook(hook) for g in gates]
    micro = args.micro or cfg.train.micro_batch_size
    x, y = train_stream.batch(micro, device)

    def sampled_forward(state: PolicyTrainingState):
        outputs.clear()
        col = PolicyCollector(keep_samples=True)
        settings = PolicyForwardSettings(sample=True, collector=col, generator=state.generator)
        with autocast_context(device, dtype):
            out = model(x, y, return_loss_details=True, policy=settings)
        return out

    # ---- 1-2: records, exact K-sparsity, kept = top-K noisy scores of the pool ----
    st = PolicyTrainingState(bn.policy_baseline, bn.policy_baseline_decay,
                             math.log(cfg.model.vocab_size), device, seed=cfg.train.seed + 1)
    out = sampled_forward(st)
    recs = out.policy_records
    by_gate = {}
    for rec in recs:
        by_gate.setdefault(id(rec.gate), []).append(rec)
    missing = [names.get(id(g), "?") for g in gates if id(g) not in by_gate]
    if missing or len(recs) != len(gates):
        fail(f"{len(recs)} records for {len(gates)} gates; missing {missing}")
    ok(f"records: one per gate per forward ({len(recs)} records, {len(gates)} gates)")
    for g in gates:
        rec = by_gate[id(g)][0]
        yo = outputs[id(g)]
        nnz = (yo != 0).sum(-1)
        if not bool((nnz == k).all()):
            fail(f"{names[id(g)]}: nonzeros per token min {int(nnz.min())} max {int(nnz.max())}, K={k}")
        sel = torch.topk(rec.r, k, dim=-1).indices
        kept = rec.cand_idx.gather(-1, sel).sort(-1).values
        nz = torch.nonzero(yo.reshape(-1, yo.shape[-1]) != 0)[:, 1].reshape(-1, k)
        if not torch.equal(kept.reshape(-1, k), nz.sort(-1).values):
            fail(f"{names[id(g)]}: kept indices are not the top-K noisy scores of the pool")
        if rec.cand_idx.shape[-1] != k + j:
            fail(f"{names[id(g)]}: pool size {rec.cand_idx.shape[-1]} != K+J={k + j}")
    xchg = []
    for g in gates:
        rec = by_gate[id(g)][0]
        sel = torch.topk(rec.r, k, dim=-1).indices
        xchg.append(float((sel >= k).float().sum(-1).mean()) / k)
    ok(f"forward exactly {k}-sparse per token in every gate, kept set = top-{k} noisy "
       f"scores of the clean Top-{k + j} pool")
    print("      exchange fraction per gate: " + " ".join(f"{q:.3f}" for q in xchg))

    # ---- 3: finite loss and gradients; the selection term reaches every encoder ----
    loss, support, adv = policy_backward_loss(out.ce, out.seq_ce, out.seq_valid,
                                              out.policy_log_prob, st.baseline)
    if not (math.isfinite(float(out.ce.detach())) and math.isfinite(float(support))):
        fail(f"non-finite CE {float(out.ce.detach())} or support {float(support)}")
    enc = {}
    for name, layer in ctl.layers:
        ps = [p for n, p in layer.named_parameters()
              if p.requires_grad and ("enc" in n or "in_proj" in n) and p.dim() >= 1]
        if not ps:
            fail(f"{name}: no encoder parameter found")
        enc[name] = ps
    w = out.seq_valid.float() / out.seq_valid.sum().float()
    sup_live = (w * adv * out.policy_log_prob.float()).sum()
    flat = [p for ps in enc.values() for p in ps]
    g_sup = torch.autograd.grad(sup_live, flat, retain_graph=True, allow_unused=True)
    i = 0
    norms = []
    for name, ps in enc.items():
        gn = 0.0
        for _ in ps:
            g = g_sup[i]
            i += 1
            if g is not None:
                if not bool(torch.isfinite(g).all()):
                    fail(f"{name}: non-finite selection gradient")
                gn += float(g.float().pow(2).sum())
        norms.append(math.sqrt(gn))
        if gn == 0.0:
            fail(f"{name}: the selection term does not reach the encoder")
    ok("selection term alone reaches every encoder; |grad| per block: "
       + " ".join(f"{v:.2e}" for v in norms))
    model.zero_grad(set_to_none=True)
    loss.backward()
    bad = [n for n, p in model.named_parameters()
           if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
    if bad:
        fail(f"non-finite gradients in {bad[:5]}")
    total = math.sqrt(sum(float(p.grad.float().pow(2).sum())
                          for p in model.parameters() if p.grad is not None))
    ok(f"CE {float(out.ce.detach()):.4f} (sampled), support term {float(support):.4e}, "
       f"advantage mean {float(adv.mean()):+.4f}; all gradients finite, total norm {total:.3e}")
    del out, loss, support, adv, sup_live, g_sup
    model.zero_grad(set_to_none=True)

    # ---- 4: state round trip reproduces the next supports on the same inputs ----
    st.accumulate(3.0 * 100, 100)
    st.finish_step()
    saved = copy.deepcopy(st.state_dict())

    def supports_and_grads(state):
        model.zero_grad(set_to_none=True)
        o = sampled_forward(state)
        masks = {gid: (outputs[gid] != 0) for gid in outputs}
        lo, _, _ = policy_backward_loss(o.ce, o.seq_ce, o.seq_valid, o.policy_log_prob,
                                        state.baseline)
        lo.backward()
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                 if p.grad is not None}
        return masks, grads, float(o.ce)

    m_a, g_a, ce_a = supports_and_grads(st)
    fresh = PolicyTrainingState(bn.policy_baseline, bn.policy_baseline_decay, 0.0, device,
                                seed=cfg.train.seed + 999)
    fresh.load_state_dict(saved, rank=0, world=1)
    if fresh.baseline != saved["baseline"] or fresh.updates != saved["updates"]:
        fail("baseline / update count not restored")
    m_b, g_b, ce_b = supports_and_grads(fresh)
    same = all(torch.equal(m_a[gid], m_b[gid]) for gid in m_a)
    if not same:
        fail("restored generator does not reproduce the sampled supports")
    dmax = max(float((g_a[n] - g_b[n]).abs().max()) for n in g_a)
    gmax = max(float(g_a[n].abs().max()) for n in g_a)
    ok(f"restored state (baseline {fresh.baseline:.4f}, {fresh.updates} update) reproduces "
       f"all {len(m_a)} gates' supports bit for bit; CE {ce_a:.6f} vs {ce_b:.6f}; "
       f"max |grad diff| {dmax:.2e} (max |grad| {gmax:.2e})")
    for h in handles:
        h.remove()
    print("PREFLIGHT_OK")


if __name__ == "__main__":
    main()
