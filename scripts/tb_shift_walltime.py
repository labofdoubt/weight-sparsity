"""Rewrite a TensorBoard event file so the run reports a chosen start time.

TensorBoard orders runs by ``run.start_time`` -- the wall_time of the FIRST
event inside the event file -- ascending, with the run name only as a
tiebreaker (tensorboard/plugins/core/core_plugin.py, ``_serve_runs``).  Run
names, file mtimes and directory order do not affect it.

So to place mirrored runs after a still-training box's own runs, and in a
chosen order among themselves, rewrite their event streams with every wall_time
shifted by a constant.  Relative timing inside a run is preserved (so the
wall-time axis still looks right), and step-indexed charts are untouched.

    python tb_shift_walltime.py --src RUN/tb --dst OUT/tb --start-epoch 1793...

Skips work when the source is unchanged since the last rewrite (a marker file
records the source's size and mtime).
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from tensorboard.backend.event_processing import event_file_loader
from tensorboard.compat.proto import event_pb2
from tensorboard.summary.writer import record_writer


def rewrite(src_dir: str, dst_dir: str, start_epoch: float) -> int:
    srcs = sorted(glob.glob(os.path.join(src_dir, "events.out.tfevents*")))
    if not srcs:
        return 0
    os.makedirs(dst_dir, exist_ok=True)
    marker = os.path.join(dst_dir, ".shift_state.json")
    state = {os.path.basename(f): [os.path.getsize(f), os.path.getmtime(f)]
             for f in srcs}
    state["_start"] = start_epoch
    try:
        if json.load(open(marker)) == state:
            return 0                      # nothing changed since last rewrite
    except (OSError, ValueError):
        pass

    # the offset comes from the earliest wall_time across the run's files, so
    # every file of the run shifts by the same amount
    first = None
    for f in srcs:
        for raw in event_file_loader.RawEventFileLoader(f).Load():
            ev = event_pb2.Event.FromString(raw)
            if ev.wall_time:
                first = ev.wall_time if first is None else min(first, ev.wall_time)
            break
    if first is None:
        return 0
    delta = start_epoch - first

    n = 0
    for f in srcs:
        out = os.path.join(dst_dir, os.path.basename(f))
        with open(out, "wb") as fh:
            w = record_writer.RecordWriter(fh)
            for raw in event_file_loader.RawEventFileLoader(f).Load():
                ev = event_pb2.Event.FromString(raw)
                if ev.wall_time:
                    ev.wall_time += delta
                w.write(ev.SerializeToString())
                n += 1
            w.flush()
    json.dump(state, open(marker, "w"))
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source tb/ directory")
    ap.add_argument("--dst", required=True, help="destination tb/ directory")
    ap.add_argument("--start-epoch", type=float, required=True,
                    help="wall_time the rewritten run should start at")
    args = ap.parse_args()
    n = rewrite(args.src, args.dst, args.start_epoch)
    print(f"[shift] {args.src} -> {args.dst}: {n} event(s) rewritten")


if __name__ == "__main__":
    main()
