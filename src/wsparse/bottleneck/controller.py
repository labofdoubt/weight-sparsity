"""Installs activation bottlenecks into a model and aggregates their diagnostics.

Deliberately independent of ``wsparse.sparsity``: this is an activation-sparsity
experiment, and its two projections are dense parameters trained normally.  The
two experiments are mutually exclusive (see ``Config.__post_init__``).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from ..config import ActivationBottleneckConfig
from ..schedules import Schedule, build_schedule
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
_PLACEMENT_ATTR = {
    "pre_mlp": "mlp_bottleneck",
    "residual": "residual_bottleneck",
    "residual_out": "residual_out_bottleneck",
}


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
        self.schedule: Schedule = build_schedule(
            kind=cfg.temperature_schedule,
            start=cfg.temperature_start,
            end=cfg.temperature_end,
            warmup_steps=cfg.temperature_warmup_steps,
            anneal_steps=cfg.temperature_anneal_steps,
            power=cfg.temperature_power,
            max_steps=max_steps,
        )
        self._temperature = float(cfg.temperature_start)
        if self.enabled:
            self._install()
            self.set_step(0)

    # ---- temperature schedule ------------------------------------------------ #
    @property
    def temperature(self) -> float:
        """The current scheduled temperature (before any per-row scaling)."""
        return self._temperature

    def set_step(self, step: int) -> float:
        """Update the prescribed temperature for this optimiser step.

        A no-op for the adaptive and hard modes, which never read it -- the
        adaptive solvers derive ``t`` from the score geometry instead.
        """
        if not self.enabled:
            return 0.0
        if self.cfg.surrogate_mode == "lapsum_fixed":
            self._temperature = float(self.cfg.fixed_temperature)
        else:
            self._temperature = float(self.schedule(step))
        for _, layer in self.layers:
            layer.gate.scheduled_temperature.fill_(self._temperature)
        return self._temperature

    def _install(self) -> None:
        cfg = self.cfg
        attr = _PLACEMENT_ATTR.get(cfg.placement)
        if attr is None:
            raise ValueError(
                f"unknown bottleneck placement: {cfg.placement!r} "
                f"({' | '.join(_PLACEMENT_ATTR)})"
            )
        blocks = getattr(self.model, "blocks", None)
        if blocks is None:
            raise ValueError("activation bottleneck requires a model with .blocks")
        indices = resolve_layers(cfg.layers, len(blocks))
        if not indices:
            raise ValueError(f"bottleneck is enabled but layers={cfg.layers!r} matched nothing")
        d_model = self.model.cfg.d_model
        for i in indices:
            block = blocks[i]
            bottleneck = SparseTopKBottleneck(d_model, cfg, bias=cfg.bias)
            # each selected layer gets its own parameters
            setattr(block, attr, bottleneck)
            self.layers.append((f"blocks.{i}", bottleneck))

    # ---- parameters ---------------------------------------------------------- #
    def parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for _, layer in self.layers:
            params.extend(layer.parameters())
        return params

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def calibrate_output_scale(self, next_batch, batches: int = 4, iters: int = 3) -> Dict[str, float]:
        """Fit each bottleneck's output gain so ``var(y) ~ var(x)`` at init.

        The block is a projection into a K-sparse code and back, so its output
        variance need not resemble its input's -- and it is inserted in front of
        an MLP that was initialised expecting the latter.  This measures both
        with forward hooks over a handful of batches and sets a fixed scalar
        ``sqrt(var_in / var_out)`` after ``out_proj``.

        Applied multiplicatively over ``iters`` passes because the layers are
        sequential: rescaling layer *l* changes what layer *l+1* sees.  Runs in
        eval mode under no_grad so it neither trains anything nor pollutes the
        feature-usage EMA.
        """
        if not self.enabled or not self.cfg.calibrate_output:
            return {}
        mods = [layer for _, layer in self.layers]
        was_training = self.model.training
        self.model.eval()
        acc: Dict[nn.Module, List[torch.Tensor]] = {}

        def hook(mod, inp, out):
            x = inp[0].detach().float()
            y = out.detach().float()
            a = acc[mod]
            a[0] += x.sum(); a[1] += x.pow(2).sum(); a[2] += x.numel()
            a[3] += y.sum(); a[4] += y.pow(2).sum(); a[5] += y.numel()

        for _ in range(max(1, iters)):
            acc = {m: [0.0] * 6 for m in mods}
            handles = [m.register_forward_hook(hook) for m in mods]
            try:
                for _ in range(max(1, batches)):
                    self.model(next_batch())
            finally:
                for h in handles:
                    h.remove()
            for m in mods:
                sx, sxx, nx, sy, syy, ny = acc[m]
                var_in = sxx / nx - (sx / nx) ** 2
                var_out = syy / ny - (sy / ny) ** 2
                if float(var_out) <= 0 or float(var_in) <= 0:
                    continue
                m.output_scale.mul_((var_in / var_out).sqrt())

        self.model.train(was_training)
        scales = [float(m.output_scale) for m in mods]
        return {
            "bottleneck/output_scale": sum(scales) / len(scales),
            "bottleneck/output_scale_min": min(scales),
            "bottleneck/output_scale_max": max(scales),
        }

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
            if self.cfg.surrogate_mode in ("lapsum_scheduled", "lapsum_fixed"):
                out["bottleneck/temperature_target"] = self._temperature
            out["bottleneck/density"] = self.cfg.k / self.cfg.n_features
            out["bottleneck/candidate_density"] = (
                self.cfg.k + self.cfg.j
            ) / self.cfg.n_features
        return out


def apply_activation_bottleneck(
    model: nn.Module, cfg: ActivationBottleneckConfig, max_steps: int = 1
) -> ActivationBottleneckController:
    return ActivationBottleneckController(model, cfg, max_steps=max_steps)
