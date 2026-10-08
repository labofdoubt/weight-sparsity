"""Cache the bottleneck *ranking scores* across checkpoints, for the score explorer.

Unlike ``extract_bottleneck_activations.py``, which caches the post-TopK code
``z`` at one target token, this walks every checkpoint of a run and stores the
**pre-TopK score** the gate ranks on, for every bottleneck, every sequence in a
fixed batch and every token position in a fixed window.

The score is exactly what the gate sorts::

    r = a        (selection_mode="topk")
    r = |a|      (selection_mode="abs_topk")
    r = |s|      (selection_mode="gated_topk", s from the score branch)

so it is captured as a forward *pre*-hook on ``mod.gate``: its first positional
argument is the ranking signal in every selection mode.  Signed values are kept
(the viewer takes the magnitude) because the sign is not recoverable later and
is worth having for analysis.

Bands follow the gate: sorted by ``r`` descending, ranks ``[0, k)`` are the
TopK that survive the forward, ``[k, k+j)`` are the J candidates that receive
surrogate gradient, and the remainder get exactly zero gradient from the gate.

    python analysis/extract_bottleneck_scores.py \
        --ckpt-dir /workspace/ckpt/dc_rout_soft_k32_j32 \
        --data-dir /workspace/data/tinystories \
        --out-dir /workspace/analysis/scores

Output, one per run:

    <out-dir>/<run>.npy           float32 (n_ckpt, n_layer, batch, n_pos, n_features)
    <out-dir>/<run>.g_ztilde.npy  dL/d~z at the gate's output (same shape; --grads)
    <out-dir>/<run>.g_z.npy       dL/dz at the gate's input, through the surrogate
    <out-dir>/<run>.json          steps, layer labels, k, j, token ids, val CE

Under ``code_residual`` a fourth array is written, ``<run>.code_residual.npy``: the
K-sparse code that gate ``l`` received from gate ``l-1`` (block 0 carries
nothing, so its slice is zero).  The score at gate ``l >= 1`` is
``carry + alpha * E_l Delta_l``, so ``score - carry`` is the block's own
contribution and the viewer can put the two on one axis.

``--code-residual-only`` adds that array to a dataset extracted before it existed: it
reads ``<out-dir>/<run>.json``, checks that the checkpoint list and the fixed
batch are the ones the dataset was made from, runs an eval forward per
checkpoint, writes ``<run>.code_residual.npy`` and lists it in the JSON.  The scores
and gradients already on disk are not touched.

With ``--grads`` (the default since 2026-10-05) every checkpoint also runs one
forward + backward on the fixed batch in ``train()`` mode and float32 -- the
same measurement as ``probe_early_training.py`` -- so the viewer's gradient
panel is available on the checkpoint ladder too.  Under
``rblapsum_surrogate_scope=first_order*`` the backward is the two-pass
``first_order_backward`` the run trained with; otherwise ``loss.backward()``.
The score itself is taken from the same pass: the forward is hard in both
modes and there is no dropout, so it equals the eval-mode score.

The array is written through ``open_memmap`` so peak memory stays at one
checkpoint's worth, and the viewer reads it back memory-mapped and slices only
the cell it draws.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wsparse.bottleneck.controller import _PLACEMENT_ATTR, parse_placements  # noqa: E402
from wsparse.bottleneck.rblapsum import first_order_backward  # noqa: E402
from wsparse.data import TokenStream  # noqa: E402
from wsparse.train import load_for_inference  # noqa: E402

GRAD_NAMES = ("g_ztilde", "g_z")


def checkpoint_step(path: str) -> int:
    m = re.search(r"ckpt_step(\d+)\.pt$", os.path.basename(path))
    if not m:
        raise ValueError(f"cannot read a step number from {path!r}")
    return int(m.group(1))


def find_bottlenecks(model, cfg):
    """``[(label, module), ...]`` in the order the controller installs them.

    Mirrors ``ActivationBottleneckController._install`` rather than trusting the
    run name: a checkpoint rebuilt from its own config is the only reliable
    statement of which placements are actually present.
    """
    placements = parse_placements(cfg.activation_bottleneck.placement)
    found = []
    for i, block in enumerate(model.blocks):
        for name in placements:
            mod = getattr(block, _PLACEMENT_ATTR[name], None)
            if mod is None or isinstance(mod, torch.nn.Identity):
                continue
            label = f"blocks.{i}" if len(placements) == 1 else f"blocks.{i}.{name}"
            found.append((label, mod))
    if not found:
        raise ValueError("this checkpoint has no activation bottleneck installed")
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", required=True, help="a run directory holding ckpt_step*.pt")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch", type=int, default=8, help="sequences in the fixed batch")
    ap.add_argument("--positions", type=int, default=128, help="token positions kept, from 0")
    ap.add_argument("--offset", type=int, default=0, help="deterministic batch offset in val.bin")
    ap.add_argument("--split", default="val", choices=("val", "train"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--grads", dest="grads", action="store_true", default=True,
                    help="also record dL/d~z and dL/dz at every checkpoint (default)")
    ap.add_argument("--no-grads", dest="grads", action="store_false")
    ap.add_argument("--code-residual-only", action="store_true",
                    help="only add <run>.code_residual.npy to an existing dataset in --out-dir")
    args = ap.parse_args()
    if args.code_residual_only:
        code_residual_only(args)
        return

    run = os.path.basename(args.ckpt_dir.rstrip("/"))
    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "ckpt_step*.pt")), key=checkpoint_step)
    if not ckpts:
        raise SystemExit(f"no ckpt_step*.pt under {args.ckpt_dir}")
    steps = [checkpoint_step(p) for p in ckpts]
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- the fixed batch ---------------------------------------------------- #
    # Built once and reused for every checkpoint and every run, so that a change
    # between two cells of the viewer is a change in the model, never in the
    # data.  deterministic_offset makes it independent of the RNG seed.
    model, cfg, _ = load_for_inference(ckpts[0], device=str(device))
    code_res = bool(getattr(cfg.activation_bottleneck, "code_residual", False))
    seq_len = int(cfg.data.seq_len)
    n_pos = min(int(args.positions), seq_len)
    stream = TokenStream(os.path.join(args.data_dir, f"{args.split}.bin"), seq_len, seed=0)
    idx, targets = stream.batch(args.batch, device, deterministic_offset=args.offset)

    bottlenecks = find_bottlenecks(model, cfg)
    labels = [lbl for lbl, _ in bottlenecks]
    n_feat = int(cfg.activation_bottleneck.n_features)
    k = int(cfg.activation_bottleneck.k)
    j = int(cfg.activation_bottleneck.j)
    sel = cfg.activation_bottleneck.selection_mode

    print(f"run={run}  ckpts={len(ckpts)}  bottlenecks={len(labels)}  "
          f"n_features={n_feat}  k={k}  j={j}  selection={sel}")
    print(f"batch={args.batch}  seq_len={seq_len}  positions kept={n_pos}")

    out_path = os.path.join(args.out_dir, f"{run}.npy")
    arr = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(ckpts), len(labels), int(args.batch), n_pos, n_feat),
    )

    grad_arrs = {}
    if args.grads:
        for g in GRAD_NAMES:
            grad_arrs[g] = np.lib.format.open_memmap(
                os.path.join(args.out_dir, f"{run}.{g}.npy"), mode="w+",
                dtype=np.float32, shape=arr.shape)

    carry_arr = None
    if code_res:
        carry_arr = np.lib.format.open_memmap(
            os.path.join(args.out_dir, f"{run}.code_residual.npy"), mode="w+",
            dtype=np.float32, shape=arr.shape)

    val_ce = []
    for ci, path in enumerate(ckpts):
        if ci > 0:  # the first is already loaded
            model, cfg, _ = load_for_inference(path, device=str(device))
            bottlenecks = find_bottlenecks(model, cfg)
        # The gate's first positional argument is the ranking signal in every
        # selection mode, so this one hook is placement- and mode-agnostic.
        grabbed: dict = {}
        handles = []

        def make_hook(li):
            def hook(module, inputs):
                a = inputs[0]
                grabbed[("score", li)] = a.detach()
                if args.grads and a.requires_grad:
                    a.register_hook(lambda g, li=li: grabbed.__setitem__(("g_z", li), g.detach()))
                return None
            return hook

        def make_post(li):
            def post(module, inputs, output):
                # the gate's output is the K-sparse code; under code_residual
                # it is what the next gate receives as its carry
                grabbed[("code", li)] = output.detach()
                if args.grads and output.requires_grad:
                    output.register_hook(
                        lambda g, li=li: grabbed.__setitem__(("g_ztilde", li), g.detach()))
                return None
            return post

        for li, (_, mod) in enumerate(bottlenecks):
            handles.append(mod.gate.register_forward_pre_hook(make_hook(li)))
            handles.append(mod.gate.register_forward_hook(make_post(li)))

        if args.grads:
            # train() mode: surrogate_active() is False in eval and under
            # no_grad, so an eval-mode backward would measure the hard mask's
            # gradient instead of the surrogate's.  No autocast: the gradients
            # are reported in float32.  The forward is hard in both modes and
            # there is no dropout, so the score is the eval-mode score.
            model.train()
            _, loss = model(idx, targets)
            scope = str(getattr(cfg.activation_bottleneck, "rblapsum_surrogate_scope", "pool"))
            if scope.startswith("first_order"):
                first_order_backward(loss, model.tok_emb.weight)
            else:
                loss.backward()
            loss = loss.detach()
        else:
            model.eval()
            with torch.no_grad():
                _, loss = model(idx, targets)
        for h in handles:
            h.remove()

        names = ("score",) + (GRAD_NAMES if args.grads else ())
        for name in names:
            for li in range(len(bottlenecks)):
                a = grabbed.get((name, li))
                if a is None:
                    raise RuntimeError(f"{path}: {name} never captured for bottleneck {li}")
                if a.shape != (args.batch, seq_len, n_feat):
                    raise RuntimeError(f"{path} layer {li}: unexpected {name} shape {tuple(a.shape)}")
                dst = arr if name == "score" else grad_arrs[name]
                dst[ci, li] = a[:, :n_pos, :].float().cpu().numpy()
        if carry_arr is not None:
            # gate l's carry is gate l-1's output (model._code_residual_stack:
            # code = gate(code + alpha * E_l Delta_l)); block 0 has none
            carry_arr[ci, 0] = 0.0
            for li in range(1, len(bottlenecks)):
                a = grabbed.get(("code", li - 1))
                if a is None:
                    raise RuntimeError(f"{path}: code never captured for bottleneck {li - 1}")
                if a.shape != (args.batch, seq_len, n_feat):
                    raise RuntimeError(f"{path} layer {li - 1}: unexpected code shape {tuple(a.shape)}")
                carry_arr[ci, li] = a[:, :n_pos, :].float().cpu().numpy()
        model.zero_grad(set_to_none=True)

        val_ce.append(float(loss))
        print(f"  [{ci + 1}/{len(ckpts)}] step {steps[ci]:>6}  batch CE {float(loss):.4f}")
        grabbed.clear()
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    arr.flush()
    for g in grad_arrs.values():
        g.flush()
    if carry_arr is not None:
        carry_arr.flush()

    meta = dict(
        run=run,
        steps=steps,
        layer_labels=labels,
        n_features=n_feat,
        k=k,
        j=j,
        selection_mode=sel,
        placement=cfg.activation_bottleneck.placement,
        surrogate_mode=cfg.activation_bottleneck.surrogate_mode,
        rblapsum_boundary_floor=cfg.activation_bottleneck.rblapsum_boundary_floor,
        rblapsum_boundary_grad_mode=cfg.activation_bottleneck.rblapsum_boundary_grad_mode,
        # The kernel/barrier temperature is one constant shared by the lapsum
        # and rblapsum modes, so the offline barrier reconstruction in
        # score_explorer.py needs nothing step-dependent.
        temperature=float(cfg.activation_bottleneck.temperature),
        n_layers=int(cfg.model.n_layers),
        batch=int(args.batch),
        n_pos=n_pos,
        seq_len=seq_len,
        split=args.split,
        offset=int(args.offset),
        token_ids=idx[:, :n_pos].cpu().numpy().tolist(),
        batch_ce=val_ce,
        array=os.path.basename(out_path),
        array_shape=list(arr.shape),
        arrays={"score": os.path.basename(out_path),
                **({g: f"{run}.{g}.npy" for g in GRAD_NAMES} if args.grads else {}),
                **({"code_residual": f"{run}.code_residual.npy"} if code_res else {})},
        rblapsum_surrogate_scope=str(getattr(cfg.activation_bottleneck,
                                             "rblapsum_surrogate_scope", "pool")),
        code_residual=code_res,
        code_residual_scale=float(getattr(cfg.activation_bottleneck, "code_residual_scale", 1.0)),
        code_residual_note=("code_residual[c, l] is the K-sparse code gate l received from gate l-1 "
                    "(zero at l = 0); score - carry is the block's own encoded "
                    "contribution alpha * E_l Delta_l" if code_res else None),
        grads_mode=("train-mode float32 forward+backward on the fixed batch; "
                    "first_order_backward for first_order scopes" if args.grads else None),
        note=(
            "scores are the signed pre-TopK ranking signal; rank by |value| when "
            f"selection_mode={sel!r}. Bands: [0,k) TopK, [k,k+j) J candidates, rest zero-grad."
        ),
    )
    with open(os.path.join(args.out_dir, f"{run}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {out_path} {arr.shape} and {run}.json")


def code_residual_only(args) -> None:
    """Add the carried-code array to a dataset that was extracted without it."""
    run = os.path.basename(args.ckpt_dir.rstrip("/"))
    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "ckpt_step*.pt")), key=checkpoint_step)
    if not ckpts:
        raise SystemExit(f"no ckpt_step*.pt under {args.ckpt_dir}")
    steps = [checkpoint_step(p) for p in ckpts]
    meta_path = os.path.join(args.out_dir, f"{run}.json")
    if not os.path.exists(meta_path):
        raise SystemExit(f"--code-residual-only needs an existing dataset: {meta_path} not found")
    meta = json.load(open(meta_path))
    device = torch.device(args.device)

    model, cfg, _ = load_for_inference(ckpts[0], device=str(device))
    if not bool(getattr(cfg.activation_bottleneck, "code_residual", False)):
        raise SystemExit(f"{run} is not a code_residual run; nothing to add")
    seq_len = int(cfg.data.seq_len)
    n_pos = min(int(args.positions), seq_len)
    stream = TokenStream(os.path.join(args.data_dir, f"{args.split}.bin"), seq_len, seed=0)
    idx, targets = stream.batch(args.batch, device, deterministic_offset=args.offset)
    # the array must line up cell for cell with the scores already on disk
    for key, want in (("steps", steps), ("batch", int(args.batch)), ("n_pos", n_pos),
                      ("split", args.split), ("offset", int(args.offset)),
                      ("token_ids", idx[:, :n_pos].cpu().numpy().tolist())):
        if meta.get(key) != want:
            raise SystemExit(f"{run}: dataset {key} differs from this invocation "
                             f"({str(meta.get(key))[:80]} vs {str(want)[:80]})")
    bottlenecks = find_bottlenecks(model, cfg)
    n_feat = int(cfg.activation_bottleneck.n_features)
    if meta.get("layer_labels") != [lbl for lbl, _ in bottlenecks]:
        raise SystemExit(f"{run}: bottleneck labels differ from the dataset's")
    print(f"run={run}  ckpts={len(ckpts)}  code_residual only")

    out_path = os.path.join(args.out_dir, f"{run}.code_residual.npy")
    carry_arr = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float32,
        shape=(len(ckpts), len(bottlenecks), int(args.batch), n_pos, n_feat))
    for ci, path in enumerate(ckpts):
        if ci > 0:
            model, cfg, _ = load_for_inference(path, device=str(device))
            bottlenecks = find_bottlenecks(model, cfg)
        codes: dict = {}
        handles = [mod.gate.register_forward_hook(
            lambda module, inputs, output, li=li: codes.__setitem__(li, output.detach()))
            for li, (_, mod) in enumerate(bottlenecks)]
        # the forward is hard in both modes, so the eval-mode code is the one the
        # training forward carried; no backward is needed for the carry
        model.eval()
        with torch.no_grad():
            model(idx, targets)
        for h in handles:
            h.remove()
        carry_arr[ci, 0] = 0.0
        for li in range(1, len(bottlenecks)):
            a = codes.get(li - 1)
            if a is None or a.shape != (args.batch, seq_len, n_feat):
                raise RuntimeError(f"{path}: code of bottleneck {li - 1} missing or misshaped")
            carry_arr[ci, li] = a[:, :n_pos, :].float().cpu().numpy()
        print(f"  [{ci + 1}/{len(ckpts)}] step {steps[ci]:>6}")
        codes.clear()
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None
    carry_arr.flush()

    meta["arrays"] = {**meta.get("arrays", {"score": meta.get("array")}),
                      "code_residual": os.path.basename(out_path)}
    meta["code_residual"] = True
    meta["code_residual_scale"] = float(getattr(cfg.activation_bottleneck, "code_residual_scale", 1.0))
    meta["code_residual_note"] = ("code_residual[c, l] is the K-sparse code gate l received from gate l-1 "
                          "(zero at l = 0); score - carry is the block's own encoded "
                          "contribution alpha * E_l Delta_l; added with --code-residual-only")
    tmp = meta_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, meta_path)
    print(f"wrote {out_path} {carry_arr.shape} and updated {run}.json")


if __name__ == "__main__":
    main()
