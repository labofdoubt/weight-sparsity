"""Monotone parameter schedules, shared by both experiment families.

Used for the weight-sparsity inverse temperature ``beta`` (which rises) and for
the activation-bottleneck LapSum temperature ``t`` (which usually falls).  The
interpolation is direction-agnostic -- nothing here assumes ``end > start``.

All schedules move a value from ``start`` to ``end``:

* ``constant``    -- ``start`` forever.
* ``exponential`` -- start * (end / start) ** p  (geometric, i.e. linear in log
  space -- the natural interpolation for a temperature; this
  is the schedule used by Continuous Sparsification, arXiv:1912.04427).
* ``linear``      -- start + (end - start) * p.
* ``polynomial``  -- start + (end - start) * p ** power
  (power > 1 keeps beta small for longer, power < 1 sharpens early).
* ``cosine``      -- start + (end - start) * (1 - cos(pi p)) / 2
  (slow at both ends, fast in the middle).

``p`` is the annealing progress: the value is held at ``beta_start`` for
``warmup_steps`` optimizer steps, then annealed over ``anneal_steps`` steps and
held at ``beta_end`` afterwards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class Schedule:
    kind: str = "exponential"
    start: float = 1.0
    end: float = 1.0
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
        b0, b1 = self.start, self.end
        if self.kind == "constant":
            return b0
        if self.kind == "linear":
            return b0 + (b1 - b0) * p
        if self.kind == "exponential":
            if b0 <= 0 or b1 <= 0:
                raise ValueError("exponential schedule requires positive start/end")
            return b0 * (b1 / b0) ** p
        if self.kind == "cosine":
            return b0 + (b1 - b0) * (1.0 - math.cos(math.pi * p)) / 2.0
        if self.kind == "polynomial":
            return b0 + (b1 - b0) * (p**self.power)
        raise ValueError(f"unknown schedule: {self.kind}")


def build_schedule(
    kind: str,
    start: float,
    end: float,
    warmup_steps: int = 0,
    anneal_steps: Optional[int] = None,
    power: float = 2.0,
    max_steps: int = 1,
) -> Schedule:
    """``anneal_steps=None`` anneals over everything left after the warmup."""
    if anneal_steps is None:
        anneal_steps = max(1, max_steps - warmup_steps)
    return Schedule(
        kind=kind,
        start=start,
        end=end,
        warmup_steps=warmup_steps,
        anneal_steps=anneal_steps,
        power=power,
    )


SCHEDULE_KINDS = ("constant", "linear", "exponential", "cosine", "polynomial")

# The weight-sparsity side spells the endpoints beta_start / beta_end.
BetaSchedule = Schedule


def build_beta_schedule(
    kind: str,
    beta_start: float,
    beta_end: float,
    warmup_steps: int = 0,
    anneal_steps: Optional[int] = None,
    power: float = 2.0,
    max_steps: int = 1,
) -> Schedule:
    """``build_schedule`` under the weight-sparsity spelling of the endpoints."""
    return build_schedule(
        kind, beta_start, beta_end, warmup_steps, anneal_steps, power, max_steps
    )
