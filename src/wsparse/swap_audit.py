"""Counterfactual hard-swap audit of RBLapSum support credit.

An offline diagnostic over saved checkpoints: how well does the surrogate's
score-space support term predict the loss effect of a finite feature
replacement?  Nothing here updates parameters or optimizer state.

Pieces, each usable on its own:

* :class:`SupportCapture` -- installed on a gate for one backward, it receives
  the *final* score-space support term the RBLapSum backward applies (after
  scale, member restriction, token centering and the radial projection) in
  the sorted order of the gate's candidate pool, plus the candidate feature
  ids recorded in the forward.  ``h = alpha * g_s`` is the term in the units
  of the unscaled scores (``rblapsum_view_scale`` = alpha).  The ordinary hard
  value gradient is never part of it, and it is not obtained by subtracting
  two backward passes.  For the first-order scope the gate's own two-pass
  backward runs and the term of the total pass is the one captured.
* :func:`capture_support` -- one deterministic train-mode forward + backward
  on a probe batch with dropout off, returning per layer the gate input ``z``,
  output ``y``, pool ids, ``h`` on the pool, and the two gradients, and
  verifying the local decomposition for abs-TopK,
  ``g_z = m * g_y + sign(z) * h``.
* :func:`swap_losses` -- the exact native-value swap at one token of one gate,
  ``y' = y - z_i e_i + z_j e_j`` with ``z_j`` the candidate's own signed
  pre-gate value, evaluated by a full forward with a gate-output hook (the
  oracle) or by a cached suffix for stream-carried models, batched; returns
  the per-sequence mean CE (fp32 CE, fp64 reduction).
* :func:`sample_pairs` -- rank-window pair sampling, uniform or exhaustive,
  seeded by (seed, layer, sequence, position) so the same rank pairs are
  drawn for every checkpoint; nothing in it looks at a gradient.
* :func:`credit_error` / :func:`bootstrap_credit` -- the balanced decision
  error with the dead band ``|dL| <= tau``, per layer and with fixed layer
  weights; undefined (``None``) when a class is empty, never 0.5.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bottleneck.controller import _PLACEMENT_ATTR, parse_placements
from .bottleneck.rblapsum import first_order_backward

AUDITABLE_SCOPES = ("pool", "inactive", "first_order", "first_order_inactive")


# --------------------------------------------------------------------------- #
# capture
# --------------------------------------------------------------------------- #
class SupportCapture:
    """One gate's final support term from one backward.

    ``record_forward`` is called by the gate (candidate ids, view scale);
    ``record_backward`` by the autograd Function with the phase of the
    first-order machinery (``"hard"`` never reaches it: that pass returns
    before the term exists) and the term in sorted-pool order.
    """

    def __init__(self) -> None:
        self.cand_idx: Optional[torch.Tensor] = None
        self.alpha: float = 1.0
        self.terms: List[Tuple[str, torch.Tensor]] = []

    def record_forward(self, cand_idx: torch.Tensor, alpha: float) -> None:
        self.cand_idx = cand_idx
        self.alpha = float(alpha)

    def record_backward(self, phase: str, g_s: torch.Tensor) -> None:
        self.terms.append((phase, g_s))

    @property
    def h(self) -> torch.Tensor:
        """The support term in unscaled-score units, sorted-pool order."""
        if len(self.terms) != 1:
            raise RuntimeError(f"expected exactly one captured support term, got {len(self.terms)}")
        return self.terms[0][1] * self.alpha


def find_bottlenecks(model, cfg) -> list:
    placements = parse_placements(cfg.activation_bottleneck.placement)
    if list(placements) != ["residual_out"]:
        raise ValueError("the audit supports the single residual_out placement, "
                         f"got {cfg.activation_bottleneck.placement!r}")
    mods = []
    for block in model.blocks:
        mod = getattr(block, _PLACEMENT_ATTR["residual_out"], None)
        if mod is None or isinstance(mod, nn.Identity):
            raise ValueError("every block must carry a residual_out bottleneck")
        mods.append(mod)
    return mods


def check_auditable(cfg) -> dict:
    """Refuse configurations the capture does not cover; return provenance."""
    ab = cfg.activation_bottleneck
    if ab.surrogate_mode != "rblapsum":
        raise ValueError(f"the audit needs surrogate_mode='rblapsum' (hard-forward RBLapSum), "
                         f"got {ab.surrogate_mode!r}")
    scope = str(getattr(ab, "rblapsum_surrogate_scope", "pool"))
    if scope not in AUDITABLE_SCOPES:
        raise ValueError(f"rblapsum_surrogate_scope={scope!r} is not instrumented "
                         f"(carry and update scopes use other gate paths); auditable: {AUDITABLE_SCOPES}")
    if ab.selection_mode != "abs_topk":
        raise ValueError(f"the decomposition check assumes selection_mode='abs_topk', got {ab.selection_mode!r}")
    if getattr(ab, "gated", False) or ab.selection_mode == "gated_topk":
        raise ValueError("gated_topk is not supported")
    return dict(
        surrogate_mode=ab.surrogate_mode, scope=scope, k=int(ab.k), j=int(ab.j),
        n_features=int(ab.n_features), temperature=float(ab.temperature),
        kernel_width=str(getattr(ab, "rblapsum_kernel_width", "fixed")),
        support_strength=getattr(ab, "rblapsum_support_strength", None),
        support_scale=float(ab.rblapsum_support_scale),
        boundary_grad_mode=str(ab.rblapsum_boundary_grad_mode),
        boundary_floor=float(ab.rblapsum_boundary_floor),
        center_tokens=bool(getattr(ab, "rblapsum_center_tokens", False)),
        radial_project=bool(getattr(ab, "rblapsum_radial_project", False)),
        view_scale=float(getattr(ab, "rblapsum_view_scale", 1.0)),
        post_norm=bool(getattr(ab, "post_norm", False)),
        code_residual=bool(getattr(ab, "code_residual", False)),
        share_projections=bool(getattr(ab, "share_projections", False)),
    )


@contextlib.contextmanager
def deterministic_forward(model: nn.Module):
    """Dropout off everywhere (``nn.Dropout`` and the functional attention
    dropout), RNG state saved; everything restored on exit.  Yields a record
    of what was changed."""
    record = {"dropout_modules": [], "attn_dropout": []}
    for name, m in model.named_modules():
        if isinstance(m, nn.Dropout) and m.p != 0.0:
            record["dropout_modules"].append((name, m.p)); m.p = 0.0
        if hasattr(m, "attn_dropout") and getattr(m, "attn_dropout") != 0.0:
            record["attn_dropout"].append((name, m.attn_dropout)); m.attn_dropout = 0.0
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield record
    finally:
        for name, p in record["dropout_modules"]:
            model.get_submodule(name).p = p
        for name, p in record["attn_dropout"]:
            model.get_submodule(name).attn_dropout = p
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


@contextlib.contextmanager
def _autocast(device: torch.device, dtype: torch.dtype):
    if dtype == torch.float32 or device.type != "cuda":
        yield
    else:
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield


def capture_support(model, cfg, mods, idx: torch.Tensor, targets: torch.Tensor,
                    layers: Sequence[int], dtype: torch.dtype = torch.float32,
                    check_tol: float = 1e-4) -> Dict[int, dict]:
    """Train-mode forward + backward on the probe batch; per layer:
    ``z`` (B,T,N) gate input, ``y`` gate output, ``cand_idx`` (B,T,K+J),
    ``h`` (B,T,K+J) in pool order, ``g_z``, ``g_y`` (B,T,N), ``mask`` (B,T,N),
    ``decomp_err`` the relative error of ``g_z - m*g_y = sign(z)*scatter(h)``.
    Parameters are left as they were (grads cleared, no optimizer)."""
    ab = cfg.activation_bottleneck
    scope = str(getattr(ab, "rblapsum_surrogate_scope", "pool"))
    device = idx.device
    was_training = model.training
    caps: Dict[int, SupportCapture] = {}
    grabbed: dict = {}
    handles = []
    out: Dict[int, dict] = {}
    try:
        model.train()
        with deterministic_forward(model) as det:
            for l in layers:
                gate = mods[l].gate
                caps[l] = SupportCapture()
                gate._support_capture = caps[l]

                def pre(module, inputs, l=l):
                    a = inputs[0]
                    grabbed[("z", l)] = a.detach()
                    if a.requires_grad:
                        a.register_hook(lambda g, l=l: grabbed.__setitem__(("g_z", l), g.detach()))

                def post(module, inputs, output, l=l):
                    grabbed[("y", l)] = output.detach()
                    if output.requires_grad:
                        output.register_hook(lambda g, l=l: grabbed.__setitem__(("g_y", l), g.detach()))

                handles += [gate.register_forward_pre_hook(pre), gate.register_forward_hook(post)]
            model.zero_grad(set_to_none=True)
            with _autocast(device, dtype):
                _, loss = model(idx, targets)
            if scope.startswith("first_order"):
                first_order_backward(loss, model.tok_emb.weight)
            else:
                loss.backward()
            for l in layers:
                z, y = grabbed[("z", l)], grabbed[("y", l)]
                g_z, g_y = grabbed[("g_z", l)], grabbed[("g_y", l)]
                cap = caps[l]
                h = cap.h
                mask = (y != 0).to(g_y.dtype)
                recon = torch.zeros_like(z, dtype=h.dtype).scatter_(-1, cap.cand_idx, h)
                lhs = (g_z - mask * g_y).double()
                rhs = (torch.sign(z).double() * recon.double())
                err = float((lhs - rhs).norm() / lhs.norm().clamp_min(1e-30))
                if err > check_tol:
                    raise RuntimeError(f"layer {l}: local decomposition g_z = m*g_y + sign(z)*h fails, "
                                       f"relative error {err:.3e} > {check_tol}")
                out[l] = dict(z=z, y=y, cand_idx=cap.cand_idx, h=h, g_z=g_z, g_y=g_y, mask=mask,
                              decomp_err=err, phase=cap.terms[0][0], alpha=cap.alpha)
            out["_meta"] = dict(loss=float(loss.detach()), dropout=det, scope=scope,
                                batch_shape=list(idx.shape), dtype=str(dtype))
    finally:
        for h_ in handles:
            h_.remove()
        for l in caps:
            mods[l].gate._support_capture = None
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    return out


# --------------------------------------------------------------------------- #
# the swap
# --------------------------------------------------------------------------- #
@dataclass
class Swap:
    seq: int          # row of the probe batch
    pos: int          # token position
    i: int            # active feature replaced
    j: int            # candidate inserted
    z_i: float        # y_i = z_i (removed exactly)
    z_j: float        # the candidate's own signed pre-gate value (inserted)


def per_token_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1),
                           ignore_index=-100, reduction="none").view(targets.shape)


def sequence_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean fp32 per-token CE over the valid positions of each row, in fp64."""
    ce = per_token_ce(logits, targets).double()
    valid = (targets != -100).double()
    return (ce * valid).sum(1) / valid.sum(1).clamp_min(1.0)


def apply_swaps(y: torch.Tensor, rows: Sequence[int], swaps: Sequence[Swap],
                z_live: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``y' = y - z_i e_i + z_j e_j`` on the given rows of a (B,T,N) code.

    The hard gate emits ``y_i = z_i`` exactly, so the removal is written as
    setting coordinate ``i`` to zero: subtracting a ``z_i`` stored from
    another forward would leave a kernel-rounding residue (a 33rd non-zero)
    whenever the batch composition differs.  With ``z_live`` (the gate input
    of the same forward) the inserted value is the live ``z_j``; otherwise
    the stored one.
    """
    out = y.clone()
    for r, s in zip(rows, swaps):
        out[r, s.pos, s.i] = 0.0
        out[r, s.pos, s.j] = (z_live[r, s.pos, s.j] if z_live is not None else s.z_j)
    return out


def _logits_from_code(model, mods, layer: int, y: torch.Tensor) -> torch.Tensor:
    """Stream-carried suffix: decode + post-norm of bottleneck ``layer``, the
    blocks after it, the final norm and the head."""
    mod = mods[layer]
    x = mod.post_norm(mod.decode(y))
    for blk in model.blocks[layer + 1:]:
        x = blk(x)
    x = model.norm_f(x)
    logits = model.lm_head(x)
    if model.logit_mult != 1.0:
        logits = logits * model.logit_mult
    return logits


@torch.no_grad()
def swap_losses(model, mods, layer: int, idx: torch.Tensor, targets: torch.Tensor,
                swaps: Sequence[Swap], batch_size: int = 32, dtype: torch.dtype = torch.float32,
                suffix: bool = False, baseline_code: Optional[torch.Tensor] = None) -> np.ndarray:
    """Per-swap sequence loss (fp64), evaluated in batches.

    ``suffix=False``: full forward with a hook replacing the gate output (the
    oracle; valid for every architecture).  ``suffix=True``: from the cached
    baseline code ``baseline_code`` (B,T,N) of this gate through
    :func:`_logits_from_code`; stream-carried models only.  Each batch row is
    a copy of the swap's sequence with its own swap applied, so batched and
    single evaluation differ only by kernel batching.
    """
    cr = bool(getattr(model, "code_residual", False))
    if suffix and cr:
        raise NotImplementedError("the cached suffix is not valid for code_residual models "
                                  "(the carried code is not decoded by the suffix); use the oracle")
    if suffix and baseline_code is None:
        raise ValueError("suffix=True needs baseline_code")
    was_training = model.training
    model.eval()
    device = idx.device
    out = np.zeros(len(swaps), dtype=np.float64)
    try:
        with deterministic_forward(model):
            for start in range(0, len(swaps), batch_size):
                chunk = list(swaps[start:start + batch_size])
                rows = list(range(len(chunk)))
                seqs = torch.tensor([s.seq for s in chunk], device=device)
                xb, tb = idx[seqs], targets[seqs]
                if suffix:
                    yb = apply_swaps(baseline_code[seqs], rows, chunk)
                    with _autocast(device, dtype):
                        logits = _logits_from_code(model, mods, layer, yb)
                else:
                    live = {}
                    def pre(module, inputs):
                        live["z"] = inputs[0].detach()
                    def hook(module, inputs, output, rows=rows, chunk=chunk):
                        return apply_swaps(output, rows, chunk, z_live=live["z"])
                    hs = [mods[layer].gate.register_forward_pre_hook(pre),
                          mods[layer].gate.register_forward_hook(hook)]
                    try:
                        with _autocast(device, dtype):
                            logits, _ = model(xb, tb)
                    finally:
                        for h in hs:
                            h.remove()
                out[start:start + len(chunk)] = sequence_loss(logits, tb).cpu().numpy()
    finally:
        model.train(was_training)
    return out


@torch.no_grad()
def baseline_codes(model, mods, idx: torch.Tensor, targets: torch.Tensor, layers: Sequence[int],
                   dtype: torch.dtype = torch.float32) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], np.ndarray]:
    """Eval-mode gate inputs ``z`` and outputs ``y`` per layer, and the
    per-sequence baseline loss, from one hook-free forward."""
    was_training = model.training
    model.eval()
    grabbed: dict = {}
    handles = []
    try:
        with deterministic_forward(model):
            for l in layers:
                handles.append(mods[l].gate.register_forward_pre_hook(
                    lambda m, inp, l=l: grabbed.__setitem__(("z", l), inp[0].detach())))
                handles.append(mods[l].gate.register_forward_hook(
                    lambda m, inp, out, l=l: grabbed.__setitem__(("y", l), out.detach())))
            with _autocast(idx.device, dtype):
                logits, _ = model(idx, targets)
            loss = sequence_loss(logits, targets).cpu().numpy()
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return {l: grabbed[("z", l)] for l in layers}, {l: grabbed[("y", l)] for l in layers}, loss


# --------------------------------------------------------------------------- #
# pairs
# --------------------------------------------------------------------------- #
def sample_pairs(cand_idx_bt: torch.Tensor, k: int, active_window: int, cand_window: int,
                 n_pairs: int, seed: int, layer: int, seq: int, pos: int,
                 exhaustive: bool = False) -> List[Tuple[int, int, int, int]]:
    """``(rank_i, rank_j, feature_i, feature_j)`` for one token: active ranks
    ``[k - active_window, k)`` against candidate ranks ``[k, k + cand_window)``
    of the sorted pool; uniform without replacement, or every pair.  The RNG
    is seeded from (seed, layer, seq, pos) only."""
    pool = cand_idx_bt.tolist()
    m = len(pool)
    a_lo, a_hi = max(0, k - active_window), k
    c_lo, c_hi = k, min(m, k + cand_window)
    pairs = [(ri, rj) for ri in range(a_lo, a_hi) for rj in range(c_lo, c_hi)]
    if not exhaustive and n_pairs < len(pairs):
        rng = np.random.default_rng([seed, layer, seq, pos])
        sel = rng.choice(len(pairs), size=n_pairs, replace=False)
        pairs = [pairs[int(s)] for s in sorted(sel)]
    return [(ri, rj, pool[ri], pool[rj]) for ri, rj in pairs]


# --------------------------------------------------------------------------- #
# the estimator
# --------------------------------------------------------------------------- #
def credit_error(r: np.ndarray, dL: np.ndarray, tau: float) -> dict:
    """Balanced decision error with the dead band ``|dL| <= tau``.

    improve: ``dL < -tau`` (the swap lowers the loss, j should be promoted:
    error if ``r <= 0``); harm: ``dL > tau`` (error if ``r > 0``).  ``E`` is
    None when either class is empty."""
    improve, harm = dL < -tau, dL > tau
    n_imp, n_harm = int(improve.sum()), int(harm.sum())
    p_miss = float((r[improve] <= 0).mean()) if n_imp else None
    p_false = float((r[harm] > 0).mean()) if n_harm else None
    E = 0.5 * (p_miss + p_false) if (p_miss is not None and p_false is not None) else None
    return dict(n_improve=n_imp, n_harm=n_harm, n_dead=int(len(dL) - n_imp - n_harm),
                p_miss_promote=p_miss, p_false_promote=p_false, E=E)


def aggregate_layers(per_layer: Dict[int, dict], weights: Dict[int, float]) -> Optional[float]:
    """Fixed-weight aggregate; None if any weighted layer is undefined."""
    tot = sum(weights[l] for l in weights)
    acc = 0.0
    for l, w in weights.items():
        E = per_layer.get(l, {}).get("E")
        if E is None:
            return None
        acc += w * E
    return acc / tot


def bootstrap_credit(seq: np.ndarray, layer: np.ndarray, r: np.ndarray, dL: np.ndarray,
                     layers: Sequence[int], weights: Dict[int, float], tau: float,
                     n_boot: int, seed: int) -> dict:
    """Resample whole sequence windows; per-layer and aggregate percentile CIs.
    All four arrays are per swap (numpy); resamples whose estimator is
    undefined are counted, not dropped or filled."""
    seq, layer, r, dL = (np.asarray(a) for a in (seq, layer, r, dL))
    seqs = np.array(sorted(set(seq.tolist())))
    rng = np.random.default_rng(seed)
    agg, per = [], {l: [] for l in layers}
    rows_of = {s: np.where(seq == s)[0] for s in seqs}
    for _ in range(n_boot):
        draw = rng.choice(seqs, size=len(seqs), replace=True)
        sel = np.concatenate([rows_of[s] for s in draw])
        pl = {l: credit_error(r[sel][layer[sel] == l], dL[sel][layer[sel] == l], tau) for l in layers}
        for l in layers:
            per[l].append(pl[l]["E"])
        agg.append(aggregate_layers(pl, weights))

    def ci(vals):
        v = np.array([x for x in vals if x is not None], dtype=float)
        return dict(n_defined=int(v.size), n_undefined=int(len(vals) - v.size),
                    lo=float(np.percentile(v, 2.5)) if v.size else None,
                    hi=float(np.percentile(v, 97.5)) if v.size else None)
    return dict(aggregate=ci(agg), per_layer={int(l): ci(per[l]) for l in layers}, n_boot=n_boot)
