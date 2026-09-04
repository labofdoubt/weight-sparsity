"""Wires the sparsity config into a model: layer selection, beta schedule,
penalty terms and the statistics that get logged during training.
"""

from __future__ import annotations

import fnmatch
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

from ..config import SparsityConfig
from ..model import MLP, CausalSelfAttention
from .masks import CSLinear, LTPLinear, SparseLinear, hard_mask_mode, make_sparse_linear
from .schedules import BetaSchedule, build_beta_schedule
from .topk import TopKSoftGateLinear

_TARGET_PARENTS = {"mlp": MLP, "attn": CausalSelfAttention}


def _iter_target_linears(model: nn.Module, targets: Iterable[str]):
    """Yield ``(parent_module, attr_name, full_name, linear)`` for every linear
    layer inside a targeted parent block."""
    parent_types = tuple(_TARGET_PARENTS[t] for t in targets)
    if not parent_types:
        return
    for parent_name, parent in model.named_modules():
        if not isinstance(parent, parent_types):
            continue
        for attr, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                full = f"{parent_name}.{attr}" if parent_name else attr
                yield parent, attr, full, child


class SparsityController:
    """Owns the masked layers, the beta schedule and the sparsity objective."""

    def __init__(self, model: nn.Module, cfg: SparsityConfig, max_steps: int = 1):
        self.cfg = cfg
        self.model = model
        self.enabled = cfg.enabled
        self.layers: List[Tuple[str, SparseLinear]] = []
        self.target_density: Dict[str, float] = {}
        self.schedule: BetaSchedule = build_beta_schedule(
            kind=cfg.beta_schedule,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
            warmup_steps=cfg.beta_warmup_steps,
            anneal_steps=cfg.beta_anneal_steps,
            power=cfg.beta_power,
            max_steps=max_steps,
        )
        self._beta = float(cfg.beta_start)
        if self.enabled:
            self._install()
            self._resolve_target_densities()
            self.set_step(0)

    # ---- construction ----------------------------------------------------- #
    def _install(self) -> None:
        cfg = self.cfg
        replacements = list(_iter_target_linears(self.model, cfg.targets))
        if not replacements:
            raise ValueError(
                f"sparsity is enabled but no layers matched targets={cfg.targets}"
            )
        for parent, attr, full_name, linear in replacements:
            sparse = make_sparse_linear(linear, cfg)
            setattr(parent, attr, sparse)
            self.layers.append((full_name, sparse))

    def _resolve_target_densities(self) -> None:
        cfg = self.cfg
        if cfg.target_density is None:
            return
        for name, _ in self.layers:
            density = cfg.target_density
            for pattern, value in cfg.target_density_overrides.items():
                if fnmatch.fnmatch(name, pattern):
                    density = float(value)
            if not 0.0 < density <= 1.0:
                raise ValueError(f"target density for {name} must be in (0, 1], got {density}")
            self.target_density[name] = density

    # ---- schedule --------------------------------------------------------- #
    @property
    def beta(self) -> float:
        return self._beta

    def set_step(self, step: int) -> float:
        """Update beta for the given optimizer step; returns the new beta."""
        if not self.enabled:
            return 0.0
        self._beta = float(self.schedule(step))
        for _, layer in self.layers:
            layer.beta.fill_(self._beta)
        return self._beta

    # ---- parameters -------------------------------------------------------- #
    def mask_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for _, layer in self.layers:
            params.extend(layer.mask_parameters())
        return params

    def mask_parameter_ids(self) -> set:
        return {id(p) for p in self.mask_parameters()}

    @property
    def total_maskable(self) -> int:
        return sum(layer.mask_numel for _, layer in self.layers)

    # ---- objective ---------------------------------------------------------- #
    def penalty(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Sparsity part of the loss plus the differentiable stats to log.

        Returns a scalar tensor (0 when sparsity is off or both coefficients
        are zero) and a dict of floats for logging.
        """
        device = next(self.model.parameters()).device
        zero = torch.zeros((), device=device, dtype=torch.float32)
        if not self.enabled or not self.layers:
            return zero, {}

        cfg = self.cfg
        need_l0 = cfg.l0_coef != 0.0
        need_target = cfg.target_density_coef != 0.0
        # method-specific penalties (the topk soft-L0 over Top-(k+j))
        extra = (layer.extra_penalty() for _, layer in self.layers)
        extra_terms = [t for t in extra if t is not None]
        if not (need_l0 or need_target or extra_terms):
            return zero, {}

        l0_total = zero
        target_terms: List[torch.Tensor] = []
        for name, layer in self.layers:
            l0 = layer.soft_l0()
            l0_total = l0_total + l0
            if need_target:
                d = self.target_density[name] * layer.mask_numel
                target_terms.append((l0 / d - 1.0) ** 2)

        loss = zero
        logs: Dict[str, float] = {}
        if need_l0:
            term = l0_total / self.total_maskable if cfg.l0_normalize else l0_total
            loss = loss + cfg.l0_coef * term
            logs["sparsity/l0_penalty"] = float(cfg.l0_coef * term.detach())
        if need_target:
            term = torch.stack(target_terms).mean()
            loss = loss + cfg.target_density_coef * term
            logs["sparsity/target_penalty"] = float(cfg.target_density_coef * term.detach())
        if extra_terms:
            term = torch.stack(extra_terms).sum()
            loss = loss + term
            logs["sparsity/soft_l0_penalty"] = float(term.detach())
        logs["sparsity/density_soft"] = float(l0_total.detach()) / self.total_maskable
        return loss, logs

    # ---- statistics ---------------------------------------------------------- #
    @torch.no_grad()
    def stats(self, per_layer: bool = False) -> Dict[str, float]:
        if not self.enabled or not self.layers:
            return {}
        soft = 0.0
        hard = 0.0
        transition = []
        mask_values = []
        topk_stats: Dict[str, List[float]] = {}
        topk_budget = 0
        out: Dict[str, float] = {}
        for name, layer in self.layers:
            l0 = float(layer.soft_l0())
            h = float(layer.hard_l0())
            soft += l0
            hard += h
            transition.append(float(layer.transition_fraction()))
            if isinstance(layer, LTPLinear):
                mask_values.append(float(layer.threshold))
            elif isinstance(layer, TopKSoftGateLinear):
                topk_budget += layer.topk_numel
                for key, value in layer.stats().items():
                    topk_stats.setdefault(key, []).append(float(value))
            elif isinstance(layer, CSLinear):
                mask_values.append(float(layer.s.mean()))
            if per_layer:
                out[f"density_hard/{name}"] = h / layer.mask_numel
                out[f"density_soft/{name}"] = l0 / layer.mask_numel
        total = self.total_maskable
        out["sparsity/beta"] = self._beta
        out["sparsity/density_soft"] = soft / total
        out["sparsity/density_hard"] = hard / total
        out["sparsity/transition_frac"] = sum(transition) / len(transition)
        if mask_values:
            key = "sparsity/threshold_mean" if self.cfg.method == "ltp" else "sparsity/s_mean"
            out[key] = sum(mask_values) / len(mask_values)
        for key, values in topk_stats.items():
            out[f"sparsity/{key}"] = sum(values) / len(values)
        if topk_budget:
            # the hard FLOP budget: |A| / N, which soft gating can only undershoot
            out["sparsity/density_topk"] = topk_budget / total
        out["sparsity/maskable_params"] = float(total)
        out["sparsity/pruned_params"] = float(total - hard)
        return out

    @torch.no_grad()
    def layer_densities(self) -> Dict[str, float]:
        return {
            name: float(layer.hard_l0()) / layer.mask_numel for name, layer in self.layers
        }

    # ---- export -------------------------------------------------------------- #
    def hard_mask(self, enabled: bool = True) -> hard_mask_mode:
        return hard_mask_mode(self.model, enabled=enabled)

    @torch.no_grad()
    def apply_hard_masks_(self) -> None:
        """Zero out pruned weights in place (irreversible)."""
        for _, layer in self.layers:
            layer.apply_hard_mask_()


def apply_sparsity(
    model: nn.Module, cfg: SparsityConfig, max_steps: int = 1
) -> SparsityController:
    return SparsityController(model, cfg, max_steps=max_steps)
