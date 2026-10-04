"""Validation curves and run metadata from run directories, one compact JSON.

For every ``<runs_dir>/<pattern>/`` that holds a ``config.json`` (archived or
current schema): the bottleneck's K, J, temperature, surrogate, post_norm and
code_residual flags, the val/ce series from ``metrics.jsonl`` as
``[[step, ce], ...]``, ``diverged.json`` and ``summary.json`` when present.
The plot scripts (``scripts/plot_kj_comparison.py``, ``scripts/plot_cr_kj.py``)
read this format.

  python scripts/extract_val_curves.py RUNS_DIR OUT.json [GLOB ...]

GLOB defaults to ``*``; several may be given.  Runs without a val/ce point are
skipped.  Runs on the box, or locally on a ``rclone copy`` of the run folders
restricted to ``{config.json,metrics.jsonl,summary.json,diverged.json}``.
"""
import glob
import json
import os
import sys


def read_run(run_dir):
    cfg_p = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_p):
        return None
    cfg = json.load(open(cfg_p))
    b, m, t = cfg["activation_bottleneck"], cfg["model"], cfg["train"]
    rec = {
        "k": b["k"], "j": b["j"], "surrogate": b["surrogate_mode"],
        # current schema: one `temperature`; archived: the per-mode keys
        "T": b.get("temperature", b.get("rblapsum_temperature")
                   if b["surrogate_mode"] == "rblapsum" else b.get("temperature_start")),
        "post_norm": bool(b.get("post_norm", False)),
        "code_residual": bool(b.get("code_residual", False)),
        "grad_mode": b.get("rblapsum_boundary_grad_mode"),
        "b0": b.get("rblapsum_boundary_floor"),
        "n_layers": m["n_layers"], "d_model": m["d_model"], "decouple": bool(m.get("decouple")),
        "seed": t["seed"], "max_steps": t["max_steps"],
        "val": [],
    }
    mj = os.path.join(run_dir, "metrics.jsonl")
    if os.path.exists(mj):
        with open(mj) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn tail line of a live run
                if "val/ce" in r:
                    rec["val"].append([r["step"], r["val/ce"]])
    for key, fn in (("diverged", "diverged.json"), ("summary", "summary.json")):
        p = os.path.join(run_dir, fn)
        if os.path.exists(p):
            rec[key] = json.load(open(p))
    return rec


def main():
    runs_dir, dst = sys.argv[1], sys.argv[2]
    patterns = sys.argv[3:] or ["*"]
    out = {}
    for pat in patterns:
        for run_dir in sorted(glob.glob(os.path.join(runs_dir, pat))):
            if not os.path.isdir(run_dir):
                continue
            rec = read_run(run_dir)
            if rec is None or not rec["val"]:
                continue
            out[os.path.basename(run_dir)] = rec
    json.dump(out, open(dst, "w"))
    print("wrote", dst, "runs:", len(out),
          "val points:", sum(len(v["val"]) for v in out.values()))


if __name__ == "__main__":
    main()
