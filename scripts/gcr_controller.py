"""Controller of the Gumbel Top-(K+J) campaign on one box (one score type).

Runs beside ``gpu_queue.sh --hold CLOSE`` and appends the 20000-step Gumbel
jobs to its queue as the (tau, lambda) search finishes:

* a searched (K, J) cell is complete when every one of its search runs has
  ended (stopped.json / diverged.json in the run directory, or a "DONE <run>"
  line in the queue log); its best setting is the lowest deterministic
  ``val/ce`` at step 2500, and that run is resumed from its own step-2500
  checkpoint (latest.pt) to 20000;
* a cell without a search (make_gumbel_cr_configs.BORROW) starts from scratch
  with the best setting of its source cell, or, with two sources, the setting
  with the lowest mean val/ce over both.

Every decision (all candidates' val/ce) goes to the selection JSON, and the
CLOSE file is created after the last job is appended.

    python scripts/gcr_controller.py --scores abs --queue /workspace/gcr/jobs.txt \\
        --queue-log /workspace/gcr/queue.log --close /workspace/gcr/queue.closed \\
        --selection /workspace/analysis/gcr_selection_abs.json
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_gumbel_cr_configs as M  # noqa: E402


def val_at(run_dir: str, step: int) -> float:
    """Deterministic val/ce logged at ``step`` (inf when missing or diverged)."""
    if os.path.exists(os.path.join(run_dir, "diverged.json")):
        return math.inf
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return math.inf
    best = math.inf
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("step") == step and "val/ce" in rec:
                v = float(rec["val/ce"])
                best = v if math.isfinite(v) else math.inf
    return best


def ended(run_dir: str, name: str, log_text: str) -> bool:
    return (os.path.exists(os.path.join(run_dir, "stopped.json"))
            or os.path.exists(os.path.join(run_dir, "diverged.json"))
            or f" DONE  {name} " in log_text)


def append(queue: str, line: str, lock: str) -> None:
    """Append under gpu_queue's own lock ($OUT/.queue.lock): its pop is
    ``head -1; sed -i 1d``, which rewrites the file, so an unlocked append
    could land between the read and the rename and be lost."""
    with open(lock, "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        with open(queue, "a") as f:
            f.write(line.rstrip("\n") + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", choices=list(M.MODE), required=True)
    ap.add_argument("--runs", default="/workspace/runs")
    ap.add_argument("--queue", required=True)
    ap.add_argument("--queue-log", required=True)
    ap.add_argument("--close", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--poll", type=int, default=60)
    args = ap.parse_args()
    sc = args.scores
    lock = os.path.join(args.runs, ".queue.lock")
    settings = [(t, l) for t in M.TAUS[sc] for l in M.LAMBDAS]
    table = {}        # (k, j) -> {(tau, lam): val}
    promoted = set()  # searched cells whose best run was resumed
    started = set()   # borrowed cells started
    log = {"scores": sc, "stop_step": M.SEARCH_STOP, "cells": {}, "borrowed": {}}

    def save():
        os.makedirs(os.path.dirname(args.selection), exist_ok=True)
        with open(args.selection + ".tmp", "w") as f:
            json.dump(log, f, indent=1)
        os.replace(args.selection + ".tmp", args.selection)

    while True:
        log_text = open(args.queue_log).read() if os.path.exists(args.queue_log) else ""
        for (k, j) in M.SEARCH:
            if (k, j) in promoted:
                continue
            names = {s: M.gum_name(sc, k, j, *s) for s in settings}
            dirs = {s: os.path.join(args.runs, n) for s, n in names.items()}
            if not all(ended(dirs[s], names[s], log_text) for s in settings):
                continue
            vals = {s: val_at(dirs[s], M.SEARCH_STOP) for s in settings}
            table[(k, j)] = vals
            best = min(settings, key=lambda s: vals[s])
            entry = {"candidates": {names[s]: vals[s] for s in settings},
                     "best": names[best], "tau": best[0], "lambda": best[1],
                     "val_ce": vals[best], "time": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                  time.gmtime())}
            if not math.isfinite(vals[best]):
                entry["error"] = "no finite val/ce at the stop step: nothing resumed"
                print(f"[ctl] {sc} ({k},{j}): {entry['error']}", flush=True)
            else:
                name = names[best]
                latest = os.path.join(dirs[best], "latest.pt")
                if not os.path.exists(latest):
                    entry["error"] = f"{latest} missing: not resumed"
                    print(f"[ctl] {sc} ({k},{j}): {entry['error']}", flush=True)
                else:
                    # keep the search phase's console log (the resume rewrites NAME.log)
                    lg = os.path.join(args.runs, f"{name}.log")
                    if os.path.exists(lg):
                        shutil.copyfile(lg, os.path.join(args.runs, f"{name}.search.log"))
                    cfg = M.rel(M.search_path(sc, k, j, *best))
                    append(args.queue, f"{name}|{cfg}|--train.resume={latest}", lock)
                    entry["resumed_from"] = latest
                    print(f"[ctl] {sc} ({k},{j}): best {name} val {vals[best]:.4f}; "
                          "resumed to step 20000", flush=True)
            log["cells"][f"{k},{j}"] = entry
            promoted.add((k, j))
            save()
        for (k, j), sources in M.BORROW.items():
            if (k, j) in started or not all(s in table for s in sources):
                continue
            mean = {s: sum(table[c][s] for c in sources) / len(sources) for s in settings}
            best = min(settings, key=lambda s: mean[s])
            src = sources[0]
            name = M.gum_name(sc, k, j, *best)
            cfg = M.rel(M.search_path(sc, src[0], src[1], *best))
            append(args.queue, f"{name}|{cfg}|--activation_bottleneck.k={k} "
                               f"--activation_bottleneck.j={j}", lock)
            log["borrowed"][f"{k},{j}"] = {
                "sources": [f"{a},{b}" for a, b in sources], "run": name, "tau": best[0],
                "lambda": best[1], "mean_source_val_ce": mean[best],
                "config": cfg, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            started.add((k, j))
            print(f"[ctl] {sc} ({k},{j}): started {name} (from {sources})", flush=True)
            save()
        if len(promoted) == len(M.SEARCH) and len(started) == len(M.BORROW):
            open(args.close, "w").close()
            print("[ctl] all 20000-step jobs appended; queue closed", flush=True)
            save()
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
