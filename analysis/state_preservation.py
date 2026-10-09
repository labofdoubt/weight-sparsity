"""State preservation across stream-carried bottlenecks: linear reconstruction.

How much of the code a bottleneck emits survives the re-encoding of the
bottlenecks after it?  For a stream-carried model (``residual_out`` placement,
no code residual) the code ``c`` of bottleneck ``s`` is transported through
bottlenecks ``s+1 .. e`` with the block updates disabled (``Delta_l = 0``):

    x_{l}   = N_{l-1}(D_{l-1} c_{l-1})         (decode, output norm)
    c_{l}   = TopK(E_l x_l)                    (encode, hard gate)

so ``c~ = c_e`` is a deterministic function of ``c = c_s`` alone -- the model's
own carry, with nothing added.  A ridge-regression map ``R(c~) = A c~ + b``
(``A`` a full N x N matrix: the active indices change from token to token, so
a map on the active coordinates only would discard coordinate identity) is
fitted on training tokens to predict ``c`` from ``c~``, ``lambda`` is chosen
on validation tokens, and the normalized held-out error

    eps = E ||R(c~) - c||^2 / E ||c - E[c]||^2

is reported on test tokens, with the training mean as ``E[c]``, so the
constant predictor scores exactly 1.  Predictions are dense: no TopK is
applied to ``R(c~)`` and the error counts every coordinate, false non-zeros
included.  Both codes are K-sparse; the map accommodates a change of basis.

Tokens come from fixed windows of ``val.bin`` (``TokenStream.batch`` with a
deterministic offset), split by *sequence* into training / validation / test
so that no window is shared; the same windows serve every model, so a
difference between two models is a difference in the models.  The map has
N^2 = 2.36M parameters, and the fit is data-limited below ~10^5 training
tokens: at 32k tokens the (32, 224) RBLapSum model scored 0.27 against 0.16
for hard Top-32, at 229k tokens both are at 0.13 with train, validation and
test errors within 0.01 of each other.  The defaults (448 / 32 / 32 windows
of 512 tokens, about a minute on one GPU) are the converged size.  Under
``code_residual`` the transport is the identity on the code and the question
is empty, so such checkpoints are refused.

    python analysis/state_preservation.py --ckpt /workspace/ckpt/<run>/ckpt_step20000.pt \
        --start 2 --end 6 --data-dir /workspace/data/tinystories \
        --out /workspace/analysis/state_pres/<run>_s2_e6.npz

Output: ``<out>.npz`` with the per-token test errors (``err_ridge``,
``err_identity``: normalized squared error per token; ``support_overlap``:
share of the start code's support still active at the end) and
``<out>.json`` with the split sizes, the lambda grid and its validation
errors, the chosen lambda, and the summary errors (ridge / identity /
constant, on train, validation and test).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wsparse.bottleneck.controller import _PLACEMENT_ATTR, parse_placements  # noqa: E402
from wsparse.data import TokenStream  # noqa: E402
from wsparse.train import load_for_inference  # noqa: E402


def find_bottlenecks(model, cfg):
    placements = parse_placements(cfg.activation_bottleneck.placement)
    if list(placements) != ["residual_out"]:
        raise SystemExit("state_preservation needs the single residual_out placement, "
                         f"got {cfg.activation_bottleneck.placement!r}")
    mods = []
    for block in model.blocks:
        mod = getattr(block, _PLACEMENT_ATTR["residual_out"], None)
        if mod is None or isinstance(mod, torch.nn.Identity):
            raise SystemExit("every block must carry a residual_out bottleneck")
        mods.append(mod)
    return mods


@torch.no_grad()
def transport(mods, c: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """``c_e`` from ``c_s`` through bottlenecks ``s+1 .. e`` with ``Delta = 0``."""
    for l in range(start + 1, end + 1):
        prev, cur = mods[l - 1], mods[l]
        x = prev.post_norm(prev.decode(c))          # what block l reads, Delta_l = 0
        if getattr(cur, "gated", False):
            raise SystemExit("gated_topk bottlenecks are not supported")
        c = cur.gate(cur.in_proj(x))                # hard TopK in eval mode
    return c


@torch.no_grad()
def collect_codes(model, mods, idx: torch.Tensor, start: int, end: int, chunk: int = 16):
    """``(c_start, c_end_transported)`` for every token of ``idx``, flattened."""
    ys, xs = [], []
    for i in range(0, idx.shape[0], chunk):
        grabbed = {}
        h = mods[start].gate.register_forward_hook(
            lambda m, inp, out: grabbed.__setitem__("c", out.detach()))
        model(idx[i:i + chunk])
        h.remove()
        c = grabbed["c"]
        ct = transport(mods, c, start, end)
        ys.append(c.reshape(-1, c.shape[-1]).float())
        xs.append(ct.reshape(-1, ct.shape[-1]).float())
    return torch.cat(ys), torch.cat(xs)


def fit_ridge(x_tr, y_tr, x_va, y_va, lambdas_rel):
    """Ridge map ``y ~ x W + b`` with ``lambda`` chosen on validation.

    Returns ``(W, mean_x, mean_y, lam, table)`` with ``table`` the list of
    ``(lambda, train_err, val_err)``; ``lambdas_rel`` are relative to the mean
    eigenvalue of the training Gram matrix.  Everything in float64.
    """
    x_tr, y_tr, x_va, y_va = (t.double() for t in (x_tr, y_tr, x_va, y_va))
    mx, my = x_tr.mean(0), y_tr.mean(0)
    xc, yc = x_tr - mx, y_tr - my
    gram = xc.T @ xc
    cross = xc.T @ yc
    evals, evecs = torch.linalg.eigh(gram)
    evals = evals.clamp_min(0.0)
    proj = evecs.T @ cross                               # (N, N)
    scale = float(evals.mean())
    den_tr = float((yc * yc).sum())
    den_va = float(((y_va - my) ** 2).sum())
    table, best = [], None
    for lr in lambdas_rel:
        lam = lr * scale
        W = evecs @ (proj / (evals + lam)[:, None])
        e_tr = float(((xc @ W - yc) ** 2).sum()) / den_tr
        e_va = float((((x_va - mx) @ W + my - y_va) ** 2).sum()) / den_va
        table.append((lam, e_tr, e_va))
        if best is None or e_va < best[1]:
            best = (W, e_va, lam)
    return best[0], mx, my, best[2], table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--start", type=int, required=True, help="bottleneck whose code is the target")
    ap.add_argument("--end", type=int, required=True, help="bottleneck whose transported code is the input")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="val", choices=("val", "train"))
    ap.add_argument("--offset", type=int, default=0, help="deterministic window offset")
    ap.add_argument("--train-seqs", type=int, default=448)
    ap.add_argument("--val-seqs", type=int, default=32)
    ap.add_argument("--test-seqs", type=int, default=32)
    ap.add_argument("--lambdas", default="1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,10",
                    help="ridge lambdas relative to the mean eigenvalue of the training Gram")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if not args.start < args.end:
        raise SystemExit("--start must be below --end")

    device = torch.device(args.device)
    model, cfg, _ = load_for_inference(args.ckpt, device=str(device))
    model.eval()
    if bool(getattr(cfg.activation_bottleneck, "code_residual", False)):
        raise SystemExit("code_residual models carry the code unchanged with Delta = 0; "
                         "this measurement is for stream-carried models")
    mods = find_bottlenecks(model, cfg)
    if not args.end < len(mods):
        raise SystemExit(f"--end {args.end} beyond the last bottleneck {len(mods) - 1}")
    k, j, n_feat = int(cfg.activation_bottleneck.k), int(cfg.activation_bottleneck.j), int(cfg.activation_bottleneck.n_features)
    seq_len = int(cfg.data.seq_len)
    n_tr, n_va, n_te = args.train_seqs, args.val_seqs, args.test_seqs
    stream = TokenStream(os.path.join(args.data_dir, f"{args.split}.bin"), seq_len, seed=0)
    idx, _ = stream.batch(n_tr + n_va + n_te, device, deterministic_offset=args.offset)

    y_all, x_all = collect_codes(model, mods, idx, args.start, args.end)
    T = seq_len
    sl = {"train": slice(0, n_tr * T), "val": slice(n_tr * T, (n_tr + n_va) * T),
          "test": slice((n_tr + n_va) * T, (n_tr + n_va + n_te) * T)}
    x = {s: x_all[v] for s, v in sl.items()}
    y = {s: y_all[v] for s, v in sl.items()}
    lambdas = [float(v) for v in args.lambdas.split(",")]
    W, mx, my, lam, table = fit_ridge(x["train"], y["train"], x["val"], y["val"], lambdas)

    def errors(xs, ys):
        xs, ys = xs.double(), ys.double()
        den = float(((ys - my) ** 2).sum(1).mean())
        e_ridge = ((xs - mx) @ W + my - ys).pow(2).sum(1) / den
        e_ident = (xs - ys).pow(2).sum(1) / den
        e_const = ((ys - my) ** 2).sum(1) / den
        sup_y, sup_x = ys != 0, xs != 0
        overlap = (sup_y & sup_x).sum(1).double() / sup_y.sum(1).clamp_min(1).double()
        return e_ridge, e_ident, e_const, overlap, den

    summary = {"run": os.path.basename(os.path.dirname(os.path.abspath(args.ckpt))),
               "ckpt": os.path.abspath(args.ckpt), "start": args.start, "end": args.end,
               "k": k, "j": j, "n_features": n_feat, "seq_len": seq_len, "split_source": args.split,
               "offset": args.offset, "seqs": {"train": n_tr, "val": n_va, "test": n_te},
               "tokens": {s: int(x[s].shape[0]) for s in x},
               "surrogate_mode": cfg.activation_bottleneck.surrogate_mode,
               "post_norm": bool(getattr(cfg.activation_bottleneck, "post_norm", False)),
               "lambda_table": [{"lambda": a, "train_err": b, "val_err": c} for a, b, c in table],
               "lambda": lam, "errors": {}}
    arrays = {}
    for s in ("train", "val", "test"):
        e_r, e_i, e_c, ov, den = errors(x[s], y[s])
        summary["errors"][s] = {"ridge": float(e_r.mean()), "identity": float(e_i.mean()),
                                "constant": float(e_c.mean()), "support_overlap": float(ov.mean()),
                                "denominator_per_token": den}
        if s == "test":
            arrays = {"err_ridge": e_r.float().cpu().numpy(), "err_identity": e_i.float().cpu().numpy(),
                      "support_overlap": ov.float().cpu().numpy()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
        json.dump(summary, fh, indent=2)
    e = summary["errors"]
    print(f"{summary['run']}  s={args.start} e={args.end}  K={k} J={j}  "
          f"tokens train/val/test {n_tr * T}/{n_va * T}/{n_te * T}  lambda {lam:.3g}")
    print("  normalized error   ridge      identity   constant   support overlap")
    for s in ("train", "val", "test"):
        print(f"  {s:5s}            {e[s]['ridge']:.4f}     {e[s]['identity']:.4f}     "
              f"{e[s]['constant']:.4f}     {e[s]['support_overlap']:.3f}")
    print(f"wrote {args.out} and {os.path.splitext(args.out)[0]}.json")


if __name__ == "__main__":
    main()
