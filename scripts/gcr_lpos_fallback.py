"""Fallback of the lambda > 0 continuation of the Gumbel-CR campaign.

The lambda = 1e-4 runs are resumed from their step-2500 search checkpoints to
20000 by ``gpu_queue.sh`` (jobs file JOBS, full list ALL).  When one of them
collapses (train_guard writes ``diverged.json``), the run with the same cell,
variant and tau at the next lambda down (default 3e-5) is resumed the same
way from its own step-2500 search checkpoint:

* while the queue file still holds jobs, the replacement goes to its head
  (under gpu_queue's lock, $RUNS/.queue.lock), so a live worker takes it next;
* once the queue file is empty (the workers exit when they find it empty), the
  replacement is started directly on a GPU with no process, in its own tmux
  session, with the same entry point and shared arguments as the queue.

START / DONE lines of the direct starts go to the queue log as well; every
decision goes to STATE (JSON).  The script exits, and creates DONE_FILE, once
the queue has drained, no replacement is pending and no direct start runs.

    python scripts/gcr_lpos_fallback.py --all /workspace/gcr/jobs_lpos.all.txt \\
        --queue /workspace/gcr/jobs_lpos.txt --queue-log /workspace/gcr/queue_lpos.log \\
        --state /workspace/gcr/lpos_fallback.json --done-file /workspace/gcr/lpos_fallback.done
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_gumbel_cr_configs as M  # noqa: E402

NAME_RE = re.compile(r"^gcr_gum(3f|3t)?_abs_k(\d+)_j(\d+)_t([\d.]+)_l(.+)$")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def max_step(run_dir: str) -> int:
    mx = 0
    path = os.path.join(run_dir, "metrics.jsonl")
    if os.path.exists(path):
        for line in open(path):
            try:
                mx = max(mx, int(json.loads(line).get("step", 0)))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
    return mx


def preflight(run_dir: str) -> str | None:
    """None when run_dir is a finished step-2500 search run that can be resumed."""
    if not os.path.exists(os.path.join(run_dir, "latest.pt")):
        return "no latest.pt"
    if os.path.exists(os.path.join(run_dir, "diverged.json")):
        return "diverged.json"
    try:
        step = json.load(open(os.path.join(run_dir, "stopped.json")))["step"]
    except (OSError, KeyError, json.JSONDecodeError):
        return "no stopped.json"
    if step != M.SEARCH_STOP:
        return f"stopped at {step}"
    if max_step(run_dir) != M.SEARCH_STOP:
        return f"metrics reach step {max_step(run_dir)} (already resumed?)"
    return None


def free_gpus(gpus, recent):
    """GPUs with no compute process (and none started by us in the last 5 min)."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    used = {int(a): float(b) for a, b in (ln.split(",") for ln in out.strip().splitlines())}
    return [g for g in gpus if used.get(g, 1e9) < 1000 and time.time() - recent.get(g, 0) > 300]


def tmux_alive(session: str) -> bool:
    return subprocess.run(["tmux", "has-session", f"-t={session}"],
                          capture_output=True).returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", required=True, help="every job line of the lambda > 0 queue")
    ap.add_argument("--queue", required=True, help="the live job file of gpu_queue.sh")
    ap.add_argument("--queue-log", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--done-file", required=True)
    ap.add_argument("--runs", default="/workspace/runs")
    ap.add_argument("--repo", default="/workspace/weight-sparsity")
    ap.add_argument("--from-lam", type=float, default=1e-4)
    ap.add_argument("--to-lam", type=float, default=3e-5)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--python", default="/venv/main/bin/python")
    ap.add_argument("--entry", default="scripts/train_guard.py")
    ap.add_argument("--shared", default="--data.data_dir=/workspace/data/tinystories "
                                        "--train.out_dir=/workspace/runs")
    ap.add_argument("--poll", type=int, default=60)
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    lock = os.path.join(args.runs, ".queue.lock")
    state = json.load(open(args.state)) if os.path.exists(args.state) else {"runs": {}}
    recent: dict = {}

    def save():
        with open(args.state + ".tmp", "w") as f:
            json.dump(state, f, indent=1)
        os.replace(args.state + ".tmp", args.state)

    def qlog(msg):
        with open(args.queue_log, "a") as f:
            f.write(msg + "\n")

    sources = [ln.split("|", 1)[0] for ln in open(args.all) if ln.strip()]
    while True:
        log_text = open(args.queue_log).read() if os.path.exists(args.queue_log) else ""
        # new collapses
        for src in sources:
            m = NAME_RE.match(src)
            if not m or float(m.group(5)) != args.from_lam or src in state["runs"]:
                continue
            if not os.path.exists(os.path.join(args.runs, src, "diverged.json")):
                continue
            V, k, j, tau = m.group(1) or "", int(m.group(2)), int(m.group(3)), float(m.group(4))
            name = M.gum_name("abs", k, j, tau, args.to_lam, V)
            cfg = M.rel(M.search_path("abs", k, j, tau, args.to_lam, V))
            entry = {"source": src, "run": name, "config": cfg, "detected": now(),
                     "source_diverged": json.load(open(os.path.join(args.runs, src,
                                                                    "diverged.json")))}
            err = preflight(os.path.join(args.runs, name))
            if err or not os.path.exists(os.path.join(args.repo, cfg)):
                entry["status"] = "error"
                entry["error"] = err or f"no config {cfg}"
            else:
                entry["status"] = "pending"
            print(f"[fb] {src} diverged -> {name}: {entry['status']} {entry.get('error', '')}",
                  flush=True)
            state["runs"][src] = entry
            save()
        # place pending replacements
        for src, e in state["runs"].items():
            if e["status"] != "pending":
                continue
            name = e["run"]
            resume = os.path.join(args.runs, name, "latest.pt")
            lg = os.path.join(args.runs, f"{name}.log")
            if os.path.exists(lg) and not os.path.exists(os.path.join(args.runs,
                                                                       f"{name}.search.log")):
                shutil.copyfile(lg, os.path.join(args.runs, f"{name}.search.log"))
            line = f"{name}|{e['config']}|--train.resume={resume}"
            with open(lock, "a") as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                rest = open(args.queue).read() if os.path.exists(args.queue) else ""
                if rest.strip():
                    with open(args.queue, "w") as f:
                        f.write(line + "\n" + rest)
                    e["status"], e["placed"] = "queued", now()
                    print(f"[fb] {name}: queued at the head", flush=True)
            if e["status"] == "pending":
                free = free_gpus(gpus, recent)
                if not free:
                    continue
                g = free[0]
                sess = f"lposfb_{name.replace('.', 'p')}"[:60]
                env = (f"CUDA_VISIBLE_DEVICES={g} OMP_NUM_THREADS={args.threads} "
                       f"MKL_NUM_THREADS={args.threads} OPENBLAS_NUM_THREADS={args.threads} "
                       "PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false")
                cmd = (f"cd {args.repo}; {env} {args.python} {args.entry} --config {e['config']} "
                       f"{args.shared} --train.resume={resume} --train.run_name={name} "
                       f"> {lg} 2>&1; echo \"[gpu{g}] DONE  {name} rc=$? $(date -Is) (direct)\" "
                       f">> {args.queue_log}")
                subprocess.run(["tmux", "new", "-d", "-s", sess, cmd], check=True)
                qlog(f"[gpu{g}] START {name}  {now()} (direct)")
                recent[g] = time.time()
                e["status"], e["placed"], e["gpu"], e["session"] = "direct", now(), g, sess
                print(f"[fb] {name}: started directly on gpu{g} ({sess})", flush=True)
            save()
        drained = "queue drained" in log_text
        pending = any(e["status"] == "pending" for e in state["runs"].values())
        running = any(e["status"] == "direct" and tmux_alive(e["session"])
                      for e in state["runs"].values())
        if drained and not pending and not running:
            open(args.done_file, "w").close()
            print(f"[fb] done {now()}", flush=True)
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
