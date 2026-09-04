"""Stop training runs whose bottleneck has collapsed.

A Top-(K+J) run with a large J can lose almost every feature and then sit at a
plateau for the rest of its schedule, burning a GPU to confirm a failure that
was already decided. ``bottleneck/feature_dead_frac`` pinned near 1 is the
signature; the loss and the budget residual look healthy throughout, so nothing
else flags it.

    python scripts/dead_feature_watchdog.py --runs /workspace/runs

Polls each live run's ``metrics.jsonl`` and sends SIGTERM to a run whose last
``--consecutive`` logged values all exceed ``--threshold``. Requires a minimum
step first, so the transient collapse many runs pass through early is not
mistaken for the permanent one.

Only ever kills the training process for that run -- the queue runner sees a
non-zero exit and moves on to the next job, so a wave keeps its cards busy.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import time
from typing import Dict, List, Optional

RUN_NAME = re.compile(r"--train\.run_name=([A-Za-z0-9_.-]+)")


def live_runs() -> Dict[str, int]:
    """``{run_name: pid}`` for every training process currently running."""
    out: Dict[str, int] = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if "wsparse.train" not in cmd:
            continue
        m = RUN_NAME.search(cmd)
        if m:
            out[m.group(1)] = int(pid)
    return out


def tail_dead_frac(path: str, n: int) -> List[tuple]:
    """The last ``n`` ``(step, dead_frac)`` pairs, cheaply."""
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        block = min(size, 400_000)          # plenty for a few hundred records
        f.seek(size - block)
        lines = f.read().decode("utf-8", "replace").splitlines()[1:]
    pts = []
    for line in lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "bottleneck/feature_dead_frac" in r:
            pts.append((r.get("step", 0), r["bottleneck/feature_dead_frac"]))
    return pts[-n:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="/workspace/runs")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--consecutive", type=int, default=5)
    ap.add_argument("--min-step", type=int, default=1000)
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"[watchdog] dead_frac > {args.threshold} for {args.consecutive} "
          f"consecutive logs after step {args.min_step}; polling every {args.interval}s",
          flush=True)
    killed: set = set()
    while True:
        for name, pid in live_runs().items():
            if name in killed:
                continue
            pts = tail_dead_frac(os.path.join(args.runs, name, "metrics.jsonl"),
                                 args.consecutive)
            if len(pts) < args.consecutive:
                continue
            step = pts[-1][0]
            if step < args.min_step:
                continue
            if all(v > args.threshold for _, v in pts):
                vals = ", ".join(f"{v:.3f}" for _, v in pts)
                print(f"[watchdog] {name} collapsed at step {step}: "
                      f"dead_frac = [{vals}] -> {'would kill' if args.dry_run else 'SIGTERM'} "
                      f"pid {pid}", flush=True)
                if not args.dry_run:
                    try:
                        os.kill(pid, signal.SIGTERM)
                        killed.add(name)
                    except OSError as exc:
                        print(f"[watchdog] could not signal {pid}: {exc}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
