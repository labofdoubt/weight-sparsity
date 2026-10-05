"""Run statistics of the carried code: where a feature enters, how long it stays.

For a ``code_residual`` model (from a checkpoint or at initialization), a few
validation batches without gradient, recording the hard support of every
gate's output ``c_{l+1}`` and, per token and feature, the runs of consecutive
gates in the support:

  entry_hist[l]      fraction of entry events (0 -> 1, or active at gate 0)
                     that happen at gate l
  final_entry[l]     fraction of the LAST code's support that entered at gate
                     l and stayed since
  survival[d]        P(still in the support d gates after entering | entered
                     at a gate that has d more gates after it), the quantity
                     the "persistent" driver of the carry scopes sets to 1
  survive_to_end     P(a run started at gate l reaches the last gate), per l
  evict_frac[l]      fraction of gate l-1's support evicted at gate l
  new_frac[l]        fraction of gate l's support that entered at gate l
  new_energy_frac[l] share of the code's energy at gate l in the new entries
  reentry_frac       fraction of entry events whose feature had an earlier run
                     at this token

  python analysis/code_runs.py --ckpt latest.pt --out x.json
  python analysis/code_runs.py --init-config cfg.yaml --out x.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from surrogate_probe import build, load_tree  # noqa: E402
from wsparse.config import apply_overrides  # noqa: E402
from wsparse.data import build_streams  # noqa: E402
from wsparse.utils import autocast_context, resolve_dtype  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-config", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--data-dir", default="/workspace/data/tinystories")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--batches", type=int, default=2)
    ap.add_argument("--offset", type=int, default=4242)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = None
    if args.ckpt:
        payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        tree = copy.deepcopy(payload["config"])
    else:
        tree = load_tree(args.init_config)
    apply_overrides(tree, args.set)
    cfg, model, bn = build(tree, dev, payload)
    cfg.data.data_dir = args.data_dir
    model.eval()
    assert getattr(model, "code_residual", False), "needs a code_residual model"
    gates = [b.residual_out_bottleneck.gate for b in model.blocks]
    L = len(gates)
    codes = {}
    hooks = [g.register_forward_hook(lambda m, i, o, gi=gi: codes.__setitem__(gi, o.detach()))
             for gi, g in enumerate(gates)]
    _, val = build_streams(cfg.data, seed=cfg.train.seed)
    dtype = resolve_dtype(cfg.train.dtype, dev)

    entry = torch.zeros(L, dtype=torch.float64)
    final_entry = torch.zeros(L, dtype=torch.float64)
    surv_num = torch.zeros(L, dtype=torch.float64)   # runs alive d gates after entry
    surv_den = torch.zeros(L, dtype=torch.float64)   # runs with >= d gates after entry
    to_end_num = torch.zeros(L, dtype=torch.float64)
    to_end_den = torch.zeros(L, dtype=torch.float64)
    evict = torch.zeros(L, dtype=torch.float64)
    new_frac = torch.zeros(L, dtype=torch.float64)
    new_energy = torch.zeros(L, dtype=torch.float64)
    n_tokens = 0
    reentry = 0.0
    n_entries = 0.0
    for bi in range(args.batches):
        x, y = val.batch(args.batch, dev, deterministic_offset=args.offset + bi * 100003)
        with torch.no_grad(), autocast_context(dev, dtype):
            model(x, y)
        masks = torch.stack([(codes[l] != 0) for l in range(L)])          # L, B, T, N
        energy = torch.stack([codes[l].float().pow(2) for l in range(L)])  # L, B, T, N
        prev = torch.zeros_like(masks[0])
        run_start = torch.full(masks[0].shape, -1, dtype=torch.long, device=dev)
        seen = torch.zeros_like(masks[0])
        for l in range(L):
            m = masks[l]
            new = m & ~prev
            gone = prev & ~m
            entry[l] += float(new.sum())
            reentry += float((new & seen).sum())
            n_entries += float(new.sum())
            seen |= m
            if l > 0:
                evict[l] += float(gone.sum()) / max(1.0, float(prev.sum()))
            new_frac[l] += float(new.sum()) / max(1.0, float(m.sum()))
            new_energy[l] += float((energy[l] * new).sum()) / max(1e-30, float((energy[l] * m).sum()))
            run_start = torch.where(new, torch.full_like(run_start, l), run_start)
            prev = m
        # survival and entry-gate statistics need the run table: recompute runs explicitly
        mk = masks.permute(1, 2, 3, 0).reshape(-1, L).cpu()            # rows: (token, feature)
        has = mk.any(1)
        mk = mk[has]
        padded = torch.cat([torch.zeros(mk.shape[0], 1, dtype=torch.bool), mk], 1)
        starts = mk & ~padded[:, :-1]                                   # entry events
        for s in range(L):
            rows = starts[:, s]
            if not bool(rows.any()):
                continue
            sub = mk[rows]
            # alive d gates after entry (consecutively): cumulative AND
            alive = torch.ones(sub.shape[0], dtype=torch.bool)
            for d in range(0, L - s):
                alive = alive & sub[:, s + d]
                surv_num[d] += float(alive.sum())
                surv_den[d] += float(sub.shape[0])
            to_end_num[s] += float(alive.sum())
            to_end_den[s] += float(sub.shape[0])
            # final code's support by entry gate: runs alive at the last gate
            final_entry[s] += float(alive.sum())
        n_tokens += x.shape[0] * x.shape[1]
    for h in hooks:
        h.remove()
    nb = float(args.batches)
    out = {
        "source": args.ckpt or args.init_config, "step": int(payload["step"]) if payload else 0,
        "k": int(cfg.activation_bottleneck.k), "n_layers": L, "tokens": n_tokens,
        "entry_hist": (entry / entry.sum()).tolist(),
        "final_entry": (final_entry / final_entry.sum()).tolist(),
        "survival": (surv_num / surv_den.clamp_min(1)).tolist(),
        "survive_to_end": (to_end_num / to_end_den.clamp_min(1)).tolist(),
        "evict_frac": (evict / nb).tolist(),
        "new_frac": (new_frac / nb).tolist(),
        "new_energy_frac": (new_energy / nb).tolist(),
        "reentry_frac": reentry / max(1.0, n_entries),
        "entries_per_token": n_entries / n_tokens,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[code_runs] step {out['step']} K={out['k']}: final code by entry gate "
          f"{[round(v, 2) for v in out['final_entry']]}; survival by d "
          f"{[round(v, 2) for v in out['survival']]}; to end by entry gate "
          f"{[round(v, 2) for v in out['survive_to_end']]}; evicted per gate "
          f"{[round(v, 3) for v in out['evict_frac']]} -> {args.out}")


if __name__ == "__main__":
    main()
