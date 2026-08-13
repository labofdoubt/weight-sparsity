"""Inverse-temperature (beta) schedules.

All schedules move ``beta`` from ``beta_start`` up to ``beta_end``:

* ``constant``    -- beta_start forever.
* ``exponential`` -- beta_start * (beta_end / beta_start) ** p  (geometric; this
  is the schedule used by Continuous Sparsification, arXiv:1912.04427).
* ``linear``      -- beta_start + (beta_end - beta_start) * p.
* ``polynomial``  -- beta_start + (beta_end - beta_start) * p ** power
  (power > 1 keeps beta small for longer, power < 1 sharpens early).
* ``cosine``      -- beta_start + (beta_end - beta_start) * (1 - cos(pi p)) / 2
  (slow at both ends, fast in the middle).

``p`` is the annealing progress: beta is held at ``beta_start`` for
``warmup_steps`` optimiser steps, then annealed over ``anneal_steps`` steps and
held at ``beta_end`` afterwards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class BetaSchedule:
    kind: str = "exponential"
    beta_start: float = 1.0
    beta_end: float = 1.0
    warmup_steps: int = 0
    anneal_steps: int = 1
    power: float = 2.0

    def __post_init__(self) -> None:
        self.anneal_steps = max(1, int(self.anneal_steps))
        self.warmup_steps = max(0, int(self.warmup_steps))

    def progress(self, step: int) -> float:
        p = (step - self.warmup_steps) / self.anneal_steps
        return min(1.0, max(0.0, p))

    def __call__(self, step: int) -> float:
        p = self.progress(step)
        b0, b1 = self.beta_start, self.beta_end
        if self.kind == "constant":
            return b0
        if self.kind == "linear":
            return b0 + (b1 - b0) * p
        if self.kind == "exponential":
            if b0 <= 0 or b1 <= 0:
                raise ValueError("exponential schedule requires positive beta_start/beta_end")
            return b0 * (b1 / b0) ** p
        if self.kind == "cosine":
            return b0 + (b1 - b0) * (1.0 - math.cos(math.pi * p)) / 2.0
        if self.kind == "polynomial":
            return b0 + (b1 - b0) * (p**self.power)
        raise ValueError(f"unknown beta schedule: {self.kind}")


def build_beta_schedule(
    kind: str,
    beta_start: float,
    beta_end: float,
    warmup_steps: int = 0,
    anneal_steps: Optional[int] = None,
    power: float = 2.0,
    max_steps: int = 1,
) -> BetaSchedule:
    """``anneal_steps=None`` anneals over everything left after the warmup."""
    if anneal_steps is None:
        anneal_steps = max(1, max_steps - warmup_steps)
    return BetaSchedule(
        kind=kind,
        beta_start=beta_start,
        beta_end=beta_end,
        warmup_steps=warmup_steps,
        anneal_steps=anneal_steps,
        power=power,
    )
