"""Exact neuron-swap interventions on a bottleneck's sparse code.

The primary object is the counterfactual

    DeltaL_{i->j} = L(swap i->j at one token, one bottleneck) - L(baseline),

measured EXACTLY by editing the gate output (the N-dim sparse code) and
re-running the model from that point (negative = the swap improves the loss).
One baseline forward caches, per installed bottleneck, the tensor entering the
module and its code; a *suffix forward* then resumes computation from the edited
code -- out_proj (+ output_scale), the remaining blocks, final norm and head --
so a batch of interventions costs one truncated forward, not a full one.
Downstream bottlenecks recompute their own supports from the perturbed stream
("recompute" semantics).  Only the ``residual_out`` placement is supported: it
is the last op of its block, so the suffix is exactly ``blocks[l+1:]``.

Everything runs deterministically in ``model.eval()`` under ``torch.no_grad()``
in fp32 (the analysis convention: probes measure
without autocast); per-token CE is computed unreduced in fp32 and differenced
before any reduction.  The first-order screen

    DeltaL_hat(i, j) = -g_i y_i + g_j v_swap,      g = dL/d(code)

uses gradients of the same mean-CE reduction, captured with
``torch.autograd.grad`` so model parameter ``.grad`` fields are never touched
(safe to run mid-training from an ``on_step`` hook).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .bottleneck.controller import _PLACEMENT_ATTR

SAMPLE_KINDS = ("random", "tail_best", "tail_worst", "exhaustive")
VALUE_MODES = ("target_sign_source_magnitude", "copy_signed_source")


def per_token_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Unreduced fp32 CE with the model's ignore_index convention, (B, T)."""
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1),
        ignore_index=-100, reduction="none",
    ).view(targets.shape)


@dataclass
class Swap:
    token: int
    source: int            # feature id, active in baseline
    target: int            # feature id, inactive in baseline
    value: float           # v_swap written at (token, target)
    source_rank: int       # 1..K
    target_rank: int       # 1..J (candidate rank)
    kind: str = "random"
    lin: float = float("nan")   # first-order DeltaL_hat


@dataclass
class BaselineState:
    x_in: Dict[int, torch.Tensor]      # layer -> (T, d) tensor entering the module
    code: Dict[int, torch.Tensor]      # layer -> (T, N) gate output (sparse)
    value: Dict[int, torch.Tensor]     # layer -> (T, N) pre-gate value z
    ce: torch.Tensor                   # (T,) baseline per-token CE, fp32
    grads: Dict[int, torch.Tensor] = field(default_factory=dict)  # dL/dcode


class SwapInterventionEngine:
    """Baseline capture + batched exact suffix forwards for one model."""

    def __init__(self, model, cfg, device: str = "cuda"):
        placement = str(cfg.activation_bottleneck.placement)
        if placement != "residual_out":
            raise NotImplementedError(
                f"swap interventions support placement='residual_out' (the "
                f"bottleneck is its block's final op, so the suffix restart is "
                f"exact); got {placement!r}"
            )
        self.model = model
        self.cfg = cfg
        self.device = torch.device(device)
        self.k = int(cfg.activation_bottleneck.k)
        self.j = int(cfg.activation_bottleneck.j)
        self.selection = str(cfg.activation_bottleneck.selection_mode)
        self.layers: List[int] = []
        self.mods: Dict[int, torch.nn.Module] = {}
        for li, block in enumerate(model.blocks):
            mod = getattr(block, _PLACEMENT_ATTR[placement])
            if not isinstance(mod, torch.nn.Identity):
                self.layers.append(li)
                self.mods[li] = mod
        if not self.layers:
            raise ValueError("no installed bottlenecks found")

    # ---- baseline ---------------------------------------------------------- #
    @torch.no_grad()
    def capture(self, x_ids: torch.Tensor, targets: torch.Tensor) -> BaselineState:
        """One deterministic forward; caches restart state per bottleneck."""
        assert x_ids.shape[0] == 1, "capture works on a single sequence"
        was_training = self.model.training
        self.model.eval()
        x_in: Dict[int, torch.Tensor] = {}
        code: Dict[int, torch.Tensor] = {}
        value: Dict[int, torch.Tensor] = {}
        hooks = []
        for li in self.layers:
            mod = self.mods[li]

            def pre(m, inp, li=li):
                x_in[li] = inp[0].detach()[0].float().clone()

            def post_gate(m, inp, out, li=li):
                value[li] = inp[0].detach()[0].float().clone()
                code[li] = out.detach()[0].float().clone()

            hooks.append(mod.register_forward_pre_hook(pre))
            hooks.append(mod.gate.register_forward_hook(post_gate))
        try:
            logits, _ = self.model(x_ids)
            ce = per_token_ce(logits, targets)[0]
        finally:
            for h in hooks:
                h.remove()
            self.model.train(was_training)
        return BaselineState(x_in=x_in, code=code, value=value, ce=ce)

    def capture_gradients(self, x_ids: torch.Tensor, targets: torch.Tensor,
                          state: BaselineState) -> None:
        """dL/dcode for every bottleneck, without touching parameter .grad.

        eval mode keeps the gate hard (``surrogate_active()`` is False), so the
        gradient is exactly the deterministic model's -- the right linearization
        for the deterministic counterfactual.
        """
        was_training = self.model.training
        self.model.eval()
        codes: Dict[int, torch.Tensor] = {}
        hooks = []
        for li in self.layers:
            def post_gate(m, inp, out, li=li):
                codes[li] = out
                return out

            hooks.append(self.mods[li].gate.register_forward_hook(post_gate))
        try:
            with torch.enable_grad():
                logits, _ = self.model(x_ids)
                ce = per_token_ce(logits, targets)
                loss = ce[targets != -100].mean()
                grads = torch.autograd.grad(loss, [codes[li] for li in self.layers],
                                            allow_unused=False)
            for li, g in zip(self.layers, grads):
                state.grads[li] = g.detach()[0].float().clone()
        finally:
            for h in hooks:
                h.remove()
            self.model.train(was_training)

    # ---- suffix forward ----------------------------------------------------- #
    @torch.no_grad()
    def suffix_ce(self, layer: int, codes: torch.Tensor, targets: torch.Tensor
                  ) -> torch.Tensor:
        """Per-token CE (B, T) resuming from edited codes at `layer`."""
        was_training = self.model.training
        self.model.eval()
        try:
            mod = self.mods[layer]
            y = mod.out_proj(codes.to(next(mod.out_proj.parameters()).dtype))
            if getattr(mod, "output_scale", None) is not None:
                y = y * mod.output_scale.to(y.dtype)
            x = y                                    # residual_out REPLACES the stream
            for blk in self.model.blocks[layer + 1:]:
                x = blk(x)
            x = self.model.norm_f(x)
            logits = self.model.lm_head(x)
            if self.model.logit_mult != 1.0:
                logits = logits * self.model.logit_mult
            tgt = targets.expand(codes.shape[0], -1)
            return per_token_ce(logits, tgt)
        finally:
            self.model.train(was_training)

    def verify_baseline(self, state: BaselineState, targets: torch.Tensor
                        ) -> Dict[int, float]:
        """max |suffix CE - full CE| per layer with NO intervention (must be ~0)."""
        out = {}
        for li in self.layers:
            ce = self.suffix_ce(li, state.code[li][None], targets)[0]
            out[li] = float((ce - state.ce).abs().max())
        return out

    # ---- swap construction --------------------------------------------------- #
    def ranks_at(self, state: BaselineState, layer: int, token: int
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """(active_ids[K], candidate_ids[J]) by the gate's own ranking score."""
        z = state.value[layer][token]
        s = z.abs() if self.selection == "abs_topk" else z
        order = torch.argsort(s, descending=True)
        return (order[: self.k].cpu().numpy(),
                order[self.k: self.k + self.j].cpu().numpy())

    def build_swap(self, state: BaselineState, layer: int, token: int,
                   src_rank: int, tgt_rank: int, kind: str,
                   value_mode: Optional[str] = None) -> Swap:
        """One swap from (source rank in 1..K, target candidate rank in 1..J)."""
        act, cand = self.ranks_at(state, layer, token)
        i = int(act[src_rank - 1]); jf = int(cand[tgt_rank - 1])
        code = state.code[layer]
        yi = float(code[token, i])
        assert yi != 0.0, "source must be active in the baseline"
        assert float(code[token, jf]) == 0.0, "target must be inactive in the baseline"
        mode = value_mode or (
            "target_sign_source_magnitude" if self.selection == "abs_topk"
            else "copy_signed_source")
        if mode == "copy_signed_source":
            v = yi
        else:
            sj = float(torch.sign(state.value[layer][token, jf]))
            v = (sj if sj != 0.0 else float(np.sign(yi))) * abs(yi)
        g = state.grads.get(layer)
        lin = float("nan")
        if g is not None:
            lin = float(-g[token, i] * yi + g[token, jf] * v)
        return Swap(token=token, source=i, target=jf, value=v,
                    source_rank=src_rank, target_rank=tgt_rank, kind=kind, lin=lin)

    def apply_swaps(self, state: BaselineState, layer: int, swaps: Sequence[Swap]
                    ) -> torch.Tensor:
        """(B, T, N) codes, one edited copy per swap."""
        code0 = state.code[layer]
        codes = code0[None].repeat(len(swaps), 1, 1)
        for b, s in enumerate(swaps):
            codes[b, s.token, s.source] = 0.0
            codes[b, s.token, s.target] = s.value
        return codes

    # ---- pair selection ------------------------------------------------------ #
    def select_pairs(self, state: BaselineState, layer: int, token: int, *,
                     mode: str = "hybrid", random_targets: int = 8,
                     tail_best: int = 32, tail_worst: int = 32,
                     exhaustive_threshold: int = 2048, seed: int = 1234,
                     seq_id: int = 0, value_mode: Optional[str] = None
                     ) -> List[Swap]:
        """Deterministic pair list for one (layer, token) context.

        The random stratum is seeded by (seed, seq, layer, token) -- NOT the
        checkpoint step -- so the same target RANKS recur across checkpoints.
        Tail pairs are chosen by the first-order screen over all K*J and are a
        BIASED sample, labeled as such; they never duplicate a random pair.
        """
        kj = self.k * self.j
        if mode == "exhaustive" or (mode == "hybrid" and kj <= exhaustive_threshold):
            return [self.build_swap(state, layer, token, i, j, "exhaustive", value_mode)
                    for i in range(1, self.k + 1) for j in range(1, self.j + 1)]
        key = f"{seed}:{seq_id}:{layer}:{token}".encode()
        rng = np.random.default_rng(int(hashlib.sha256(key).hexdigest()[:15], 16))
        chosen: Dict[Tuple[int, int], str] = {}
        for i in range(1, self.k + 1):
            for j in rng.choice(self.j, size=min(random_targets, self.j),
                                replace=False):
                chosen[(i, int(j) + 1)] = "random"
        if mode == "hybrid" and state.grads.get(layer) is not None:
            lin = self.lin_matrix(state, layer, token, value_mode)
            flat = np.argsort(lin, axis=None)
            picked = 0
            for f in flat:                                   # most negative first
                pair = (int(f // self.j) + 1, int(f % self.j) + 1)
                if pair not in chosen:
                    chosen[pair] = "tail_best"; picked += 1
                if picked >= tail_best: break
            picked = 0
            for f in flat[::-1]:                             # most positive first
                pair = (int(f // self.j) + 1, int(f % self.j) + 1)
                if pair not in chosen:
                    chosen[pair] = "tail_worst"; picked += 1
                if picked >= tail_worst: break
        return [self.build_swap(state, layer, token, i, j, kind, value_mode)
                for (i, j), kind in sorted(chosen.items())]

    def lin_matrix(self, state: BaselineState, layer: int, token: int,
                   value_mode: Optional[str] = None) -> np.ndarray:
        """Full (K, J) first-order DeltaL_hat, vectorized.

        lin[i, j] = -g_i y_i + g_j v(i, j) with v per swap_value_mode.
        """
        act, cand = self.ranks_at(state, layer, token)
        g = state.grads[layer][token]
        y = state.code[layer][token]
        z = state.value[layer][token]
        ai = torch.as_tensor(act, device=g.device)
        cj = torch.as_tensor(cand, device=g.device)
        gi, yi, gj = g[ai].double(), y[ai].double(), g[cj].double()
        mode = value_mode or (
            "target_sign_source_magnitude" if self.selection == "abs_topk"
            else "copy_signed_source")
        if mode == "copy_signed_source":
            v = yi[:, None].expand(self.k, self.j)
        else:
            sj = torch.sign(z[cj].double())[None, :].expand(self.k, self.j)
            si = torch.sign(yi)[:, None].expand(self.k, self.j)
            v = torch.where(sj == 0, si, sj) * yi.abs()[:, None]
        lin = (-gi * yi)[:, None] + gj[None, :] * v
        return lin.float().cpu().numpy()

    # ---- batched exact evaluation -------------------------------------------- #
    def evaluate(self, state: BaselineState, layer: int, swaps: Sequence[Swap],
                 targets: torch.Tensor, batch_size: int = 64
                 ) -> List[Dict[str, float]]:
        """Exact DeltaL for each swap; OOM-halving batches; fp32 accumulation."""
        valid = (targets[0] != -100)
        n_valid = int(valid.sum())
        results: List[Dict[str, float]] = []
        pos = 0
        bs = max(1, batch_size)
        base = state.ce
        while pos < len(swaps):
            chunk = list(swaps[pos: pos + bs])
            try:
                codes = self.apply_swaps(state, layer, chunk).to(self.device)
                ce = self.suffix_ce(layer, codes, targets).double()
            except torch.cuda.OutOfMemoryError:
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                torch.cuda.empty_cache()
                continue
            d = (ce - base.double()[None])                      # (B, T)
            d = torch.where(valid[None], d, torch.zeros_like(d))
            for b, s in enumerate(chunk):
                prefix = float(d[b, : s.token].abs().max()) if s.token > 0 else 0.0
                results.append(dict(
                    delta_loss_mean=float(d[b].sum() / n_valid),
                    delta_nll_total=float(d[b].sum()),
                    swapped_loss_mean=float(ce[b][valid].mean()),
                    prefix_delta_max_abs=prefix,
                ))
            pos += len(chunk)
        return results
