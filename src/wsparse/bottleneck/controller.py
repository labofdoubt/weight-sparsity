"""Installs activation bottlenecks into a model and aggregates their diagnostics.

The bottleneck's two projections are dense parameters trained normally; what
is sparse is the code between them.  Each module keeps its own diagnostics and
the controller only aggregates them.  Under ``share_projections`` every
installed module holds the same projection objects (see ``projection_owners``);
the gates stay per module regardless.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..config import ActivationBottleneckConfig
from .module import SparseTopKBottleneck


def resolve_layers(spec, n_layers: int) -> List[int]:
    """``"all"``, ``"even"``, ``"odd"``, ``"last:n"``, ``"first:n"`` or indices."""
    if isinstance(spec, str):
        text = spec.strip().lower()
        if text == "all":
            return list(range(n_layers))
        if text == "even":
            return [i for i in range(n_layers) if i % 2 == 0]
        if text == "odd":
            return [i for i in range(n_layers) if i % 2 == 1]
        if text.startswith("last:"):
            return list(range(max(0, n_layers - int(text[5:])), n_layers))
        if text.startswith("first:"):
            return list(range(min(n_layers, int(text[6:]))))
        spec = [int(part) for part in text.replace(",", " ").split()]
    idx = sorted({int(i) for i in spec})
    bad = [i for i in idx if not 0 <= i < n_layers]
    if bad:
        raise ValueError(f"bottleneck layers {bad} out of range for {n_layers} layers")
    return idx


#: where in a block the bottleneck is spliced in -> the attribute it replaces.
#: ``pre_mlp`` sits inside the MLP branch, so the residual skip routes around
#: it; the two stream placements replace the stream itself, so nothing does --
#: ``residual`` at the head of the block, ``residual_out`` at the tail.
#: ``post_attn`` / ``post_mlp`` sit on a sub-block output before the residual
#: add: inside a branch like ``pre_mlp``, but constraining what the branch may
#: contribute rather than what it may read.
_PLACEMENT_ATTR = {
    "pre_mlp": "mlp_bottleneck",
    "residual": "residual_bottleneck",
    "residual_out": "residual_out_bottleneck",
    "post_attn": "post_attn_bottleneck",
    "post_mlp": "post_mlp_bottleneck",
}


def parse_placements(spec) -> List[str]:
    """``"post_mlp"`` or ``"post_mlp,post_attn"`` -> a list of placement names.

    Accepts a list as readily as a string: the CLI parser turns a bare ``a,b``
    into ``["a", "b"]`` before this ever sees it, and YAML may give a sequence
    too.  Stringifying a list here would silently produce its repr.

    Several placements may be active at once; each installs its own bottleneck
    with its own parameters, so the parameter cost scales with how many are
    named.  Order is normalized to the order they occur in a block's forward.
    """
    if isinstance(spec, (list, tuple, set)):
        names = [str(t).strip() for t in spec if str(t).strip()]
    else:
        names = [t for t in str(spec).replace("+", ",").replace(" ", ",").split(",") if t]
    if not names:
        raise ValueError("bottleneck placement is empty")
    unknown = [n for n in names if n not in _PLACEMENT_ATTR]
    if unknown:
        raise ValueError(
            f"unknown bottleneck placement: {', '.join(repr(u) for u in unknown)} "
            f"({' | '.join(_PLACEMENT_ATTR)})"
        )
    seen = list(dict.fromkeys(names))
    return [n for n in _PLACEMENT_ATTR if n in seen]


class ActivationBottleneckController:
    """Owns the installed bottlenecks and the metrics they export."""

    def __init__(
        self,
        model: nn.Module,
        cfg: ActivationBottleneckConfig,
        max_steps: int = 1,
    ):
        self.cfg = cfg
        self.model = model
        self.enabled = cfg.enabled
        self.layers: List[Tuple[str, SparseTopKBottleneck]] = []
        if self.enabled:
            self._install()

    def set_step(self, step: int) -> float:
        """No-op kept for probe compatibility: temperatures are constants now."""
        del step
        return 0.0


    def _install(self) -> None:
        cfg = self.cfg
        placements = parse_placements(cfg.placement)
        blocks = getattr(self.model, "blocks", None)
        if blocks is None:
            raise ValueError("activation bottleneck requires a model with .blocks")
        indices = resolve_layers(cfg.layers, len(blocks))
        if not indices:
            raise ValueError(f"bottleneck is enabled but layers={cfg.layers!r} matched nothing")
        d_model = self.model.cfg.d_model
        norm_eps = float(getattr(self.model.cfg, "norm_eps", 1e-6))
        last = len(blocks) - 1
        # Under share_projections the first module built owns the one encoder
        # / decoder pair and every later module adopts it (all layers, all
        # placements); otherwise each module draws its own.
        share = bool(getattr(cfg, "share_projections", False))
        source: Optional[SparseTopKBottleneck] = None
        for i in indices:
            block = blocks[i]
            for name in placements:
                # The final residual_out bottleneck already feeds norm_f, so a
                # post-norm there would be two norms in a row.
                already_normed = (name == "residual_out" and i == last)
                bottleneck = SparseTopKBottleneck(
                    d_model, cfg, bias=cfg.bias,
                    post_norm=bool(cfg.post_norm) and not already_normed,
                    norm_eps=norm_eps, share_from=source)
                if share and source is None:
                    source = bottleneck
                # each selected layer *and* placement gets its own module -- and
                # its own parameters, unless they are shared
                setattr(block, _PLACEMENT_ATTR[name], bottleneck)
                label = f"blocks.{i}" if len(placements) == 1 else f"blocks.{i}.{name}"
                self.layers.append((label, bottleneck))
        if getattr(cfg, "code_residual", False):
            self._install_code_residual(indices, len(blocks))
        self._restore_decoder_scale(d_model)

    def _restore_decoder_scale(self, d_model: int) -> None:
        """Set ``g_D`` from the model config at install time.

        ``g_D`` is a plain float, not state, so a checkpoint does not carry it;
        md_init_ sets it for a fresh run, but every loader that rebuilds a model
        from a saved config (load_for_inference, the analysis --ckpt paths) runs
        build_model + apply_activation_bottleneck + load_state_dict and never
        md_init_.  Setting the same value here makes those rebuilds exact;
        md_init_ still sets it again (same value), so training is unchanged.
        """
        mode = str(getattr(getattr(self.model, "cfg", None),
                           "bottleneck_decoder_scale", "none"))
        if mode == "none":
            return
        from .module import effective_backward_support
        for _, layer in self.layers:
            layer.decoder_scale = math.sqrt(
                d_model / effective_backward_support(layer.gate))

    def _install_code_residual(self, indices, n_blocks: int) -> None:
        """The model-side switch for the code-residual forward.

        TransformerLM._code_residual_stack carries the code between the
        installed residual_out bottlenecks, so every block must have one.  No
        module is added: the first bottleneck encodes block 0's whole output
        and the last one's decoder is the readout, as in the stream-carried
        stack.  Plain attributes (like ``embed_scale``), so the state_dict and
        the parameter count are the stream-carried model's.
        """
        cfg = self.cfg
        if list(indices) != list(range(n_blocks)):
            raise ValueError(
                "code_residual carries the code through EVERY block: "
                f"layers must be 'all', got {cfg.layers!r}")
        self.model.code_residual = True
        self.model.code_residual_scale = float(cfg.code_residual_scale)


    def parameters(self) -> List[nn.Parameter]:
        """Every bottleneck parameter once: shared projections are not repeated."""
        params: List[nn.Parameter] = []
        seen = set()
        for _, layer in self.layers:
            for p in layer.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    params.append(p)
        return params

    def projection_owners(self) -> List[Tuple[str, SparseTopKBottleneck]]:
        """``layers`` restricted to one module per distinct set of projections.

        Without ``share_projections`` that is every bottleneck; with it, only
        the first.  Anything that scales or re-initializes projection weights
        in place has to loop over this rather than over ``layers``, or a shared
        matrix takes the change once per module that holds it.
        """
        owners: List[Tuple[str, SparseTopKBottleneck]] = []
        seen = set()
        for name, layer in self.layers:
            key = id(layer.in_proj.weight)
            if key not in seen:
                seen.add(key)
                owners.append((name, layer))
        return owners

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


    @torch.no_grad()
    def usage_vectors(self) -> Dict[str, torch.Tensor]:
        """``{layer_name: per-feature selection rate}`` for the usage plots."""
        if not self.enabled:
            return {}
        return {
            name: layer.gate.feature_usage()
            for name, layer in self.layers
            if float(layer.gate.usage_steps) > 0
        }

    # ---- diagnostics ---------------------------------------------------------- #
    @torch.no_grad()
    def stats(self, per_layer: bool = False) -> Dict[str, float]:
        if not self.enabled or not self.layers:
            return {}
        pooled: Dict[str, List[float]] = {}
        out: Dict[str, float] = {}
        for name, layer in self.layers:
            diag = layer.diagnostics
            for key, value in diag.items():
                if key == "grad_by_rank":
                    for b, v in enumerate(value.tolist()):
                        pooled.setdefault(f"grad_rank_bin{b}", []).append(v)
                    continue
                pooled.setdefault(key, []).append(float(value))
                if per_layer:
                    out[f"bottleneck_{key}/{name}"] = float(value)
        for key, values in pooled.items():
            out[f"bottleneck/{key}"] = sum(values) / len(values)
        if out:
            out["bottleneck/layers"] = float(len(self.layers))

            out["bottleneck/density"] = self.cfg.k / self.cfg.n_features
            out["bottleneck/candidate_density"] = (
                self.cfg.k + self.cfg.j
            ) / self.cfg.n_features
        return out


def apply_activation_bottleneck(
    model: nn.Module, cfg: ActivationBottleneckConfig, max_steps: int = 1
) -> ActivationBottleneckController:
    return ActivationBottleneckController(model, cfg, max_steps=max_steps)
