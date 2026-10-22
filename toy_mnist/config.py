"""Configuration of the toy MNIST study.

Nested dataclasses with YAML files (``_base_`` includes, resolved relative to
the including file) and dotted command-line overrides (``train.lr=1e-3`` or
``--train.lr=1e-3``).  The ``topk.dust_*`` fields are the DUST fields of the
main repository (``wsparse.config.ActivationBottleneckConfig``), with the same
names, defaults and validation, so a setting carries over by name.
"""

import copy
import dataclasses
import math
import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml

N_PIXELS = 28 * 28
N_CLASSES = 10

NORMALIZE_MODES = ("none", "global")
SURROGATE_MODES = ("hard", "dust")
SIGMA_MODES = ("absolute", "relative_rms")
TARGETS = ("encoder", "backprop")
OPTIMIZERS = ("shared", "separate")
LR_SCHEDULES = ("cosine", "linear", "constant")


@dataclass
class DataConfig:
    data_dir: str = "data/mnist"
    # none:   pixels in [0, 1] (the raw intensities / 255)
    # global: (pixel - mean) / std with the scalar mean and std of the training
    #         pixels, so the background is a negative constant and AbsTopK
    #         ranks by distance from the mean intensity
    normalize: str = "none"


@dataclass
class ModelConfig:
    # z = W2 relu(W1 x'), x' the (rescaled, gated) input
    hidden: int = 256
    bias: bool = True


@dataclass
class RescaleConfig:
    # the diagonal rescaling right after the input: x -> exp(g) * x, g a
    # trainable vector of N_PIXELS entries initialised at 0 (the identity)
    enabled: bool = False


@dataclass
class TopKConfig:
    # AbsTopK on the (rescaled) input: keeps the k entries of largest |u| per
    # sample, zero elsewhere.  j is the J of the Top(K+J) pool that receives
    # the DUST noise (as in the main repository); the hard backward never
    # reads it.  null (the default) resolves to the whole input under
    # surrogate_mode: dust (j = 784 - k, every pixel is perturbed) and to 0
    # under hard; an integer is used as given.
    enabled: bool = False
    k: int = 64
    j: Optional[int] = None
    # hard: the gradient passes the hard mask (active entries only)
    # dust: the hard gate, with ordinary hard steps alternating with DUST
    #       steps that estimate dL/du on the pool by reward-weighted noise
    #       (toy_mnist.dust); needs rescale.enabled, the only parameter
    #       before the gate
    surrogate_mode: str = "hard"
    # ---- dust (surrogate_mode: dust): the main repository's fields --------- #
    # a non-default value under surrogate_mode: hard is an error
    dust_population: int = 32          # draws per batch (one gate, so K_l = population)
    dust_sigma: float = 0.1
    dust_sigma_mode: str = "absolute"  # absolute | relative_rms
    dust_gamma: float = 0.0            # must stay 0: a sample has no later tokens
    dust_schedule: List[int] = field(default_factory=lambda: [1, 1])  # [n_hard, n_dust]
    dust_target: str = "encoder"       # encoder | backprop (coincide here, see README)
    dust_grad_scale: float = 1.0
    dust_optimizer: str = "shared"     # shared | separate
    dust_lr_scale: float = 1.0
    dust_chunk_draws: int = 1
    dust_cosine_every_steps: int = 0

    def resolved_j(self) -> int:
        """The pool width beyond k: ``j`` as given, else the whole input under
        dust and 0 under hard."""
        if self.j is not None:
            return int(self.j)
        return max(0, N_PIXELS - int(self.k)) if self.surrogate_mode == "dust" else 0


@dataclass
class TrainConfig:
    optimizer: str = "adamw"
    lr: float = 1.0e-3
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    eps: float = 1.0e-8
    weight_decay: float = 0.0   # >= 2-D weights only; biases and g never decay
    grad_clip: float = 0.0      # 0 = no clipping (train/grad_norm is still logged)
    lr_schedule: str = "cosine"
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    batch_size: int = 128
    max_steps: int = 3000
    log_every_steps: int = 20
    validate_every_steps: int = 200
    val_batch_size: int = 1000
    seed: int = 1337
    device: str = "auto"
    out_dir: str = "runs_mnist"
    run_name: str = "default"
    tensorboard: bool = True
    save_checkpoint: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    rescale: RescaleConfig = field(default_factory=RescaleConfig)
    topk: TopKConfig = field(default_factory=TopKConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # the dust fields and their defaults (a non-default value under
    # surrogate_mode: hard is rejected, as in the main repository)
    DUST_DEFAULTS = {
        "dust_population": 32,
        "dust_sigma": 0.1,
        "dust_sigma_mode": "absolute",
        "dust_gamma": 0.0,
        "dust_schedule": [1, 1],
        "dust_target": "encoder",
        "dust_grad_scale": 1.0,
        "dust_optimizer": "shared",
        "dust_lr_scale": 1.0,
        "dust_chunk_draws": 1,
        "dust_cosine_every_steps": 0,
    }

    def validate(self) -> "Config":
        d, m, r, t, tr = self.data, self.model, self.rescale, self.topk, self.train
        if d.normalize not in NORMALIZE_MODES:
            raise ValueError(f"data.normalize must be one of {NORMALIZE_MODES}, got {d.normalize!r}")
        if m.hidden < 1:
            raise ValueError("model.hidden must be >= 1")
        if t.surrogate_mode not in SURROGATE_MODES:
            raise ValueError(f"topk.surrogate_mode must be one of {SURROGATE_MODES}, "
                             f"got {t.surrogate_mode!r}")
        if t.enabled:
            if t.k < 1 or t.k > N_PIXELS:
                raise ValueError(f"topk.k must be in [1, {N_PIXELS}], got {t.k}")
            if t.j is not None and t.j < 0:
                raise ValueError("topk.j must be >= 0 (or null: the whole input under dust)")
        changed = [k for k, v in self.DUST_DEFAULTS.items() if getattr(t, k) != v]
        if t.surrogate_mode != "dust":
            if changed:
                raise ValueError(f"{', '.join(changed)}: the dust_* fields apply to "
                                 f"surrogate_mode='dust' only, got {t.surrogate_mode!r}")
        else:
            if not t.enabled:
                raise ValueError("surrogate_mode='dust' needs topk.enabled")
            if not r.enabled:
                raise ValueError("surrogate_mode='dust' updates the parameters before the gate, "
                                 "which here is the diagonal rescaling: needs rescale.enabled")
            if int(t.dust_population) < 2:
                raise ValueError("dust_population must be >= 2 (one gate; the baseline is "
                                 "the mean over its own draws)")
            sigma = float(t.dust_sigma)
            if not (math.isfinite(sigma) and sigma > 0):
                raise ValueError(f"dust_sigma must be positive and finite, got {t.dust_sigma!r}")
            if t.dust_sigma_mode not in SIGMA_MODES:
                raise ValueError(f"unknown dust_sigma_mode: {t.dust_sigma_mode!r} "
                                 f"(one of {SIGMA_MODES})")
            if float(t.dust_gamma) != 0.0:
                raise ValueError("dust_gamma must be 0 on MNIST: a sample has no later tokens "
                                 "whose losses could be credited to its noise")
            sched = t.dust_schedule
            if (not isinstance(sched, (list, tuple)) or len(sched) != 2
                    or any(int(v) != v or int(v) < 0 for v in sched)):
                raise ValueError("dust_schedule must be [n_hard, n_dust] with non-negative "
                                 f"integers, got {sched!r}")
            t.dust_schedule = [int(v) for v in sched]
            if t.dust_schedule[1] < 1:
                raise ValueError("dust_schedule needs n_dust >= 1 (with no DUST steps use "
                                 "surrogate_mode: hard)")
            if t.dust_target not in TARGETS:
                raise ValueError(f"unknown dust_target: {t.dust_target!r} (one of {TARGETS})")
            for name in ("dust_grad_scale", "dust_lr_scale"):
                v = float(getattr(t, name))
                if not (math.isfinite(v) and v > 0):
                    raise ValueError(f"{name} must be positive and finite, got {v!r}")
            if t.dust_optimizer not in OPTIMIZERS:
                raise ValueError(f"unknown dust_optimizer: {t.dust_optimizer!r} "
                                 f"(one of {OPTIMIZERS})")
            if int(t.dust_chunk_draws) < 1:
                raise ValueError("dust_chunk_draws must be >= 1")
            if int(t.dust_cosine_every_steps) < 0:
                raise ValueError("dust_cosine_every_steps must be >= 0")
        if tr.optimizer.lower() != "adamw":
            raise ValueError("train.optimizer: only adamw")
        if tr.lr_schedule not in LR_SCHEDULES:
            raise ValueError(f"train.lr_schedule must be one of {LR_SCHEDULES}")
        if len(tr.betas) != 2:
            raise ValueError("train.betas must have two entries")
        if tr.max_steps < 1 or tr.batch_size < 1 or tr.val_batch_size < 1:
            raise ValueError("train.max_steps, batch_size and val_batch_size must be >= 1")
        if tr.log_every_steps < 1 or tr.validate_every_steps < 0:
            raise ValueError("train.log_every_steps >= 1 and validate_every_steps >= 0")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def _deep_merge(base: Dict, over: Dict) -> Dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml_with_bases(path: str) -> Dict[str, Any]:
    """The file's mapping with its ``_base_`` files (a path or a list, relative
    to the file's directory) merged underneath, earlier bases first."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    bases = raw.pop("_base_", None) or []
    if isinstance(bases, str):
        bases = [bases]
    merged: Dict[str, Any] = {}
    for b in bases:
        bp = b if os.path.isabs(b) else os.path.join(os.path.dirname(os.path.abspath(path)), b)
        merged = _deep_merge(merged, load_yaml_with_bases(bp))
    return _deep_merge(merged, raw)


def _coerce(value: Any, typ: Any, where: str) -> Any:
    origin = getattr(typ, "__origin__", None)
    if origin is Union:  # Optional[T]
        args = [a for a in getattr(typ, "__args__", ()) if a is not type(None)]
        if value is None or (isinstance(value, str) and value.strip().lower() in ("none", "null", "")):
            return None
        return _coerce(value, args[0], where) if len(args) == 1 else value
    if origin in (list, List):
        if isinstance(value, str):
            value = yaml.safe_load(value)
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{where}: expected a list, got {value!r}")
        args = getattr(typ, "__args__", ())
        inner = args[0] if args else None
        return [_coerce(v, inner, where) if inner else v for v in value]
    if typ is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "1", "yes", "on"):
            return True
        if isinstance(value, str) and value.strip().lower() in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"{where}: expected a boolean, got {value!r}")
    if typ is int:
        if isinstance(value, bool):
            raise ValueError(f"{where}: expected an integer, got {value!r}")
        if isinstance(value, int):
            return value
        fv = float(value)
        if fv != int(fv):
            raise ValueError(f"{where}: expected an integer, got {value!r}")
        return int(fv)
    if typ is float:
        if isinstance(value, bool):
            raise ValueError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if typ is str:
        return str(value)
    return value


def _apply(obj: Any, values: Dict[str, Any], prefix: str = "") -> None:
    names = {f.name: f for f in fields(obj)}
    for key, value in values.items():
        if key not in names:
            raise ValueError(f"unknown config key {prefix}{key!r}; valid: {sorted(names)}")
        f = names[key]
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"{prefix}{key}: expected a mapping")
            _apply(current, value, prefix=f"{prefix}{key}.")
        else:
            setattr(obj, key, _coerce(value, f.type, f"{prefix}{key}"))


def _parse_overrides(overrides: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in overrides:
        s = item[2:] if item.startswith("--") else item
        if "=" not in s:
            raise ValueError(f"override {item!r} is not of the form section.key=value")
        key, val = s.split("=", 1)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return out


def load_config(path: Optional[str] = None, overrides: Sequence[str] = (),
                values: Optional[Dict[str, Any]] = None) -> Config:
    """Defaults, then the YAML file, then ``values`` (a mapping), then the
    command-line overrides; validated."""
    cfg = Config()
    if path is not None:
        _apply(cfg, load_yaml_with_bases(path))
    if values:
        _apply(cfg, values)
    if overrides:
        _apply(cfg, _parse_overrides(overrides))
    return cfg.validate()
