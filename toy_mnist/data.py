"""MNIST as flat float tensors on the training device.

The four IDX files are downloaded from a mirror list, verified by MD5, parsed
and cached as ``mnist.npz`` in ``data.data_dir``.  Everything (60 000 train,
10 000 test images of 28 x 28) lives on the device as float32 ``[N, 784]``.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import struct
import urllib.request
import zipfile
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import torch

from .config import DataConfig, N_PIXELS

MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)
#: file name -> MD5 of the gzip (the values torchvision checks against)
FILES = {
    "train_images": ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    "train_labels": ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    "test_images": ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    "test_labels": ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
}
CACHE = "mnist.npz"


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(data_dir: str) -> Dict[str, str]:
    """Fetch any missing file into ``data_dir``; verify every MD5."""
    os.makedirs(data_dir, exist_ok=True)
    paths = {}
    for key, (name, md5) in FILES.items():
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            last: Optional[Exception] = None
            for mirror in MIRRORS:
                try:
                    urllib.request.urlretrieve(mirror + name, path + ".part")
                    os.replace(path + ".part", path)
                    last = None
                    break
                except Exception as exc:  # pragma: no cover - network
                    last = exc
            if last is not None:
                raise RuntimeError(f"could not download {name} from {MIRRORS}: {last}")
        got = _md5(path)
        if got != md5:
            raise RuntimeError(f"{path}: MD5 {got} != expected {md5}; delete it and retry")
        paths[key] = path
    return paths


def parse_idx(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    zero, dtype_code, ndim = data[0], data[2], data[3]
    if zero != 0 or dtype_code != 0x08:
        raise ValueError(f"{path}: not an unsigned-byte IDX file")
    dims = struct.unpack(">" + "I" * ndim, data[4:4 + 4 * ndim])
    arr = np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim)
    if arr.size != int(np.prod(dims)):
        raise ValueError(f"{path}: {arr.size} bytes for shape {dims}")
    return arr.reshape(dims)


def load_arrays(data_dir: str) -> Dict[str, np.ndarray]:
    """``train_x [60000, 28, 28] uint8, train_y [60000] uint8, test_x, test_y``;
    cached as ``mnist.npz`` after the first parse."""
    cache = os.path.join(data_dir, CACHE)
    arrays = None
    if os.path.exists(cache):
        try:
            with np.load(cache) as z:
                arrays = {k: z[k] for k in ("train_x", "train_y", "test_x", "test_y")}
        except (zipfile.BadZipFile, EOFError, KeyError, ValueError, OSError):
            arrays = None  # another process is still writing it, or it is damaged
    if arrays is None:
        paths = download(data_dir)
        arrays = {
            "train_x": parse_idx(paths["train_images"]),
            "train_y": parse_idx(paths["train_labels"]),
            "test_x": parse_idx(paths["test_images"]),
            "test_y": parse_idx(paths["test_labels"]),
        }
        # atomic: concurrent runs starting together must never read a partial file
        tmp = f"{cache}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
        os.replace(tmp, cache)
    expect = {"train_x": (60000, 28, 28), "train_y": (60000,),
              "test_x": (10000, 28, 28), "test_y": (10000,)}
    for k, shape in expect.items():
        if tuple(arrays[k].shape) != shape:
            raise ValueError(f"{k}: shape {arrays[k].shape} != {shape}")
    return arrays


class MNISTData:
    """Flat float32 images on ``device``; an epoch-permutation training
    sampler (every sample once per epoch, the last partial batch dropped) and
    sequential validation batches over the test set."""

    def __init__(self, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
                 test_y: np.ndarray, normalize: str, device: torch.device, seed: int):
        def flat(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(a)).reshape(a.shape[0], -1).float() / 255.0

        tx, vx = flat(train_x), flat(test_x)
        if tx.shape[1] != N_PIXELS:
            raise ValueError(f"expected {N_PIXELS} pixels per image, got {tx.shape[1]}")
        self.mean, self.std = 0.0, 1.0
        if normalize == "global":
            self.mean, self.std = float(tx.mean()), float(tx.std())
            tx = (tx - self.mean) / self.std
            vx = (vx - self.mean) / self.std
        elif normalize != "none":
            raise ValueError(f"unknown normalize mode {normalize!r}")
        self.train_x = tx.to(device)
        self.train_y = torch.from_numpy(np.asarray(train_y)).long().to(device)
        self.test_x = vx.to(device)
        self.test_y = torch.from_numpy(np.asarray(test_y)).long().to(device)
        self.device = device
        self.n_train = self.train_x.shape[0]
        self.generator = torch.Generator().manual_seed(int(seed) + 1)
        self._perm: Optional[torch.Tensor] = None
        self._pos = 0
        self.epoch = 0

    @classmethod
    def from_config(cls, cfg: DataConfig, device: torch.device, seed: int) -> "MNISTData":
        a = load_arrays(cfg.data_dir)
        return cls(a["train_x"], a["train_y"], a["test_x"], a["test_y"], cfg.normalize,
                   device, seed)

    def train_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._perm is None or self._pos + batch_size > self.n_train:
            self._perm = torch.randperm(self.n_train, generator=self.generator).to(self.device)
            self._pos = 0
            self.epoch += 1  # permutations drawn so far
        idx = self._perm[self._pos:self._pos + batch_size]
        self._pos += batch_size
        return self.train_x[idx], self.train_y[idx]

    def val_batches(self, batch_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for i in range(0, self.test_x.shape[0], batch_size):
            yield self.test_x[i:i + batch_size], self.test_y[i:i + batch_size]
