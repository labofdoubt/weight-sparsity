"""Training-diagnostic summaries from metrics.jsonl files, one JSON for the plots.

For every ``<runs_dir>/<pattern>/metrics.jsonl``: gradient-norm spikes (logged
training steps after ``--from-step`` with train/grad_norm above 1, their count
and maximum), the median gradient norm, feature-usage entropy, dead-feature
fraction and rank boundary over ``--window`` (a step range), and the val/ce
series.  Several runs directories may be given (box copies, Drive copies).

  python scripts/extract_run_diag.py OUT.json RUNS_DIR[:GLOB] ...
"""
import glob
import json
import os
import statistics
import sys


def summarize(path, from_step=200, window=(2000, 3000)):
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    def ser(key):
        return [(r["step"], r[key]) for r in rows if key in r]
    def med(key):
        v = [x for s, x in ser(key) if window[0] <= s <= window[1]]
        return statistics.median(v) if v else None
    gn = [(s, g) for s, g in ser("train/grad_norm") if s > from_step]
    spikes = [(s, g) for s, g in gn if g > 1.0]
    return {
        "last_step": rows[-1]["step"] if rows else 0,
        "spikes": len(spikes), "spike_max": max([g for _, g in spikes], default=0.0),
        "spike_steps": [s for s, _ in spikes][:40],
        "grad_norm_med": med("train/grad_norm"),
        "usage_entropy": med("bottleneck/feature_usage_entropy"),
        "usage_max": med("bottleneck/feature_usage_max"),
        "dead_frac": med("bottleneck/feature_dead_frac"),
        "boundary": med("bottleneck/rb_boundary"),
        "val": [[s, v] for s, v in ser("val/ce")],
    }


def main():
    dst = sys.argv[1]
    out = {}
    for spec in sys.argv[2:]:
        runs_dir, _, pat = spec.partition(":")
        for mf in sorted(glob.glob(os.path.join(runs_dir, pat or "*", "metrics.jsonl"))):
            out[os.path.basename(os.path.dirname(mf))] = summarize(mf)
    json.dump(out, open(dst, "w"))
    for name, d in out.items():
        print(f"{name:34s} to {d['last_step']:5d}  spikes>1: {d['spikes']:3d} (max {d['spike_max']:.1f})  "
              f"gnorm {d['grad_norm_med']}  H_usage {d['usage_entropy']}  dead {d['dead_frac']}  b {d['boundary']}")


if __name__ == "__main__":
    main()
