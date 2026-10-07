"""Per-block monitor of laplace_policy campaign runs at a given step.

For every run directory matching ``--pattern`` under ``--runs`` reads
``metrics.jsonl`` and prints, at the last logged step <= ``--step``:

  - clean (deterministic Top-K) and stochastic validation CE and their gap
    (val_stochastic/ce - val/ce) at the last validation <= step;
  - the sampled training CE, the noise-off probe on the same inputs, their gap;
  - per block l: exchange fraction q_l, overlap 1 - q_l, effective width T_l
    (mean over rows), pool span s_(K+1) - s_(K+J), score-gradient RMS (gamma *
    centred sign(r - u) / T, the analytic location gradient);
  - the total gradient norm at the step and the clipping frequency: the share
    of logged steps in (previous report step, step] whose norm exceeds
    train.grad_clip (logged every log_every_steps, so a sample of the steps);
  - advantage mean and variance (rms^2 - mean^2) over the step's log window;
  - non-finite flags: any policy/nonfinite or per-block policy_nonfinite seen
    up to the step, and any non-finite CE.

    python scripts/policy_monitor.py --runs /workspace/runs --pattern 'vi_pol_*' \
        --step 500 [--since 0] [--json out.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re


def blocks(row: dict, key: str) -> list:
    out = []
    for name, v in row.items():
        m = re.fullmatch(rf"bottleneck_{key}/blocks\.(\d+)", name)
        if m:
            out.append((int(m.group(1)), v))
    return [v for _, v in sorted(out)]


def read(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a line being written
    return rows


def report(run_dir: str, step: int, since: int, clip: float) -> dict:
    rows = read(os.path.join(run_dir, "metrics.jsonl"))
    train = [r for r in rows if "train/ce" in r and r["step"] <= step]
    val = [r for r in rows if "val/ce" in r and r["step"] <= step]
    if not train:
        return {}
    t = train[-1]
    window = [r for r in train if since < r["step"] <= step]
    gn = [r["train/grad_norm"] for r in window]
    out = {
        "step": t["step"], "train_ce": t["train/ce"],
        "det_probe_ce": t.get("train_deterministic_probe/ce"),
        "probe_gap": t.get("train_deterministic_probe/gap"),
        "tau": t.get("policy/tau"), "baseline": t.get("policy/baseline"),
        "grad_norm": t["train/grad_norm"],
        "clip_frac": (sum(g > clip for g in gn) / len(gn)) if gn else None,
        "grad_norm_median": sorted(gn)[len(gn) // 2] if gn else None,
        "adv_mean": t.get("policy/advantage_mean"), "adv_rms": t.get("policy/advantage_rms"),
        "q": blocks(t, "policy_exchange_frac"), "overlap": blocks(t, "policy_overlap"),
        "t_mean": blocks(t, "policy_t_mean"), "span": blocks(t, "policy_pool_span"),
        "score_grad_rms": blocks(t, "policy_score_grad_rms"),
        "nonfinite_any": any(r.get("policy/nonfinite", 0) or any(blocks(r, "policy_nonfinite"))
                             for r in train),
        "ce_nonfinite": any(not math.isfinite(r["train/ce"]) for r in train),
    }
    if out["adv_mean"] is not None and out["adv_rms"] is not None:
        out["adv_var"] = max(0.0, out["adv_rms"] ** 2 - out["adv_mean"] ** 2)
    if val:
        v = val[-1]
        out.update({"val_step": v["step"], "val_ce": v["val/ce"],
                    "val_sto_ce": v.get("val_stochastic/ce"),
                    "val_sto_std": v.get("val_stochastic/mc_std"),
                    "val_gap": (v.get("val_stochastic/ce") - v["val/ce"]
                                if v.get("val_stochastic/ce") is not None else None)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/workspace/runs")
    ap.add_argument("--pattern", default="vi_pol_*")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--since", type=int, default=0, help="start of the clipping window")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    allrep = {}
    for d in sorted(glob.glob(os.path.join(args.runs, args.pattern))):
        if not os.path.isfile(os.path.join(d, "metrics.jsonl")):
            continue
        rep = report(d, args.step, args.since, args.clip)
        if not rep:
            continue
        name = os.path.basename(d)
        allrep[name] = rep
        f = lambda x, p=4: "-" if x is None else f"{x:.{p}f}"  # noqa: E731
        print(f"\n{name}  step {rep['step']}  tau {f(rep['tau'], 4)}  B {f(rep['baseline'], 3)}")
        print(f"  val@{rep.get('val_step', '-')}: clean {f(rep.get('val_ce'))}  stochastic "
              f"{f(rep.get('val_sto_ce'))} (+-{f(rep.get('val_sto_std'))})  gap "
              f"{f(rep.get('val_gap'))}")
        print(f"  train: sampled {f(rep['train_ce'])}  noise-off probe {f(rep['det_probe_ce'])}  "
              f"gap {f(rep['probe_gap'])}")
        print(f"  grad norm {rep['grad_norm']:.3g} (median {f(rep['grad_norm_median'], 3)} over "
              f"({args.since}, {args.step}]), clipped share {f(rep['clip_frac'], 3)};  advantage "
              f"mean {f(rep['adv_mean'])} var {f(rep.get('adv_var'), 5)};  nonfinite "
              f"policy={rep['nonfinite_any']} ce={rep['ce_nonfinite']}")
        print("  block      " + " ".join(f"{i:>7d}" for i in range(len(rep["q"]))))
        for key, lab in (("q", "exchange"), ("overlap", "overlap"), ("t_mean", "T mean"),
                         ("span", "span"), ("score_grad_rms", "sgrad rms")):
            print(f"  {lab:10s} " + " ".join(f"{x:7.3g}" for x in rep[key]))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(allrep, fh, indent=1)


if __name__ == "__main__":
    main()
