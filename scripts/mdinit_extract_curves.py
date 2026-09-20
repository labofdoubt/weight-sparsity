"""Collect val/ce + train/ce curves and run metadata for the mdinit campaign.

Runs on the training box; writes one compact JSON to stdout-named file.
Usage: python extract_remote.py /workspace/runs /workspace/mdinit_curves.json
"""
import glob, json, os, sys

out = {}
runs_dir, dst = sys.argv[1], sys.argv[2]
for run_dir in sorted(glob.glob(os.path.join(runs_dir, "ca_rout_*"))):
    if not os.path.isdir(run_dir):
        continue
    name = os.path.basename(run_dir)
    rec = {"val": [], "train": []}
    cfg_p = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_p):
        cfg = json.load(open(cfg_p))
        b, m, t = cfg["activation_bottleneck"], cfg["model"], cfg["train"]
        rec["k"], rec["j"] = b["k"], b["j"]
        rec["surrogate"] = b["surrogate_mode"]
        rec["T"] = (b["rblapsum_temperature"] if b["surrogate_mode"] == "rblapsum"
                    else b["temperature_start"])
        rec["regime"] = ("md" if m.get("decouple") else
                         "mdinit_wd%g" % t["weight_decay"] if m.get("md_init") else "plain")
        rec["weight_decay"] = t["weight_decay"]
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
                if "train/ce" in r and r["step"] % 100 == 0:
                    rec["train"].append([r["step"], r["train/ce"]])
    dv = os.path.join(run_dir, "diverged.json")
    if os.path.exists(dv):
        rec["diverged"] = json.load(open(dv))
    sm = os.path.join(run_dir, "summary.json")
    if os.path.exists(sm):
        rec["summary"] = json.load(open(sm))
    out[name] = rec
json.dump(out, open(dst, "w"))
print("wrote", dst, "runs:", len(out),
      "val points:", sum(len(v["val"]) for v in out.values()))
