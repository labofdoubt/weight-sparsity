"""Synthetic MNIST-shaped data and small configs for the tests (no download)."""

import numpy as np
import torch

from toy_mnist.config import load_config
from toy_mnist.data import MNISTData

CONFIG_DIR = __import__("os").path.join(
    __import__("os").path.dirname(__file__), "..", "configs")


def synthetic_data(n_train=512, n_test=200, seed=0, device="cpu", normalize="none"):
    rng = np.random.default_rng(seed)
    tx = rng.integers(0, 256, size=(n_train, 28, 28), dtype=np.uint8)
    vx = rng.integers(0, 256, size=(n_test, 28, 28), dtype=np.uint8)
    ty = rng.integers(0, 10, size=n_train).astype(np.uint8)
    vy = rng.integers(0, 10, size=n_test).astype(np.uint8)
    return MNISTData(tx, ty, vx, vy, normalize, torch.device(device), seed)


def small_config(tmp_path, name, **values):
    """A shipped config with a tiny training budget and the run dir under tmp."""
    base = {"train": {"max_steps": 6, "log_every_steps": 2, "validate_every_steps": 3,
                      "warmup_steps": 2, "batch_size": 32, "val_batch_size": 100,
                      "out_dir": str(tmp_path), "tensorboard": False, "device": "cpu",
                      "run_name": name},
            "model": {"hidden": 16}}
    for section, vals in values.items():
        base.setdefault(section, {}).update(vals)
    import os
    return load_config(os.path.join(CONFIG_DIR, f"{name}.yaml"), values=base)
