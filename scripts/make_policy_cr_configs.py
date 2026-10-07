"""Configs for the laplace_policy temperature campaign under the code residual.

Every config written here is one of the four stable code-carried RBLapSum
cells of docs/kj-vs-hard-topk-code-residual.tex -- ``configs/cr_kj/
ma_cr_rbk_k<K>_j<J>_t2.yaml`` (code_residual: true, no output norm, T=2) --
with the gate's training rule replaced by ``surrogate_mode: laplace_policy``:

  activation_bottleneck.surrogate_mode            rblapsum -> laplace_policy
  activation_bottleneck.rblapsum_boundary_grad_mode  through_rank_kappa -> detach
      (the RBLapSum knob back at its default; laplace_policy rejects it otherwise)
  activation_bottleneck.temperature               2.0 -> tau_0 (absolute, score units)
  activation_bottleneck.policy_*                  the schedule below
  train.policy_val_samples / _val_seed / _train_deterministic_every_steps
  train.run_name

and nothing else: the script reloads every file and refuses to write one in
which any other field moved.  Two families:

  --calibration [T ...]   constant-T probes vi_cal_k<K>_j<J>_t<T> for every cell
                          (default T in 0.05 0.1 0.2 0.5), configs/policy_cr/calibration/
  --campaign K:J:T0 ...   the constant / annealed pair of each cell,
                          vi_pol_k<K>_j<J>_t<T0>_{const,anneal}, configs/policy_cr/

The annealed arm is exponential from tau_0 to 0.2 tau_0, held for 1000 steps
and annealed over 17000 (tau_f is reached at step 18000 of 20000); both arms
keep gamma = 1 (constant), centred scores, the EMA baseline (decay 0.99) and
two stochastic validation draws.

    python scripts/make_policy_cr_configs.py --calibration
    python scripts/make_policy_cr_configs.py --campaign 32:32:0.1 32:480:0.2 ...
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wsparse.config import config_from_dict, load_config  # noqa: E402

CELLS = ((32, 32), (32, 480), (128, 128), (256, 256))
CAL_T = (0.05, 0.1, 0.2, 0.5)
ROOT = os.path.join(os.path.dirname(__file__), "..")
BASE = "configs/cr_kj/ma_cr_rbk_k{k}_j{j}_t2.yaml"
OUT = "configs/policy_cr"

COMMON_BN = {
    "surrogate_mode": "laplace_policy",
    "rblapsum_boundary_grad_mode": "detach",
    "policy_temperature_mode": "absolute",
    "policy_min_temperature": 1.0e-6,
    "policy_center_scores": True,
    "policy_baseline": "ema",
    "policy_baseline_decay": 0.99,
    "policy_baseline_initial": None,
    "policy_support_scale": 1.0,
    "policy_support_scale_mode": "constant",
    "policy_support_temperature_ref": 1.0,
    "policy_support_scale_max": None,
}
CONSTANT = {"policy_temperature_schedule": "constant", "policy_temperature_final": None,
            "policy_temperature_hold_steps": 0, "policy_temperature_anneal_steps": 0}
COMMON_TRAIN = {"policy_val_samples": 2, "policy_val_seed": 1337,
                "policy_train_deterministic_every_steps": 0}
ALLOWED = ({f"activation_bottleneck.{k}" for k in COMMON_BN}
           | {f"activation_bottleneck.{k}" for k in CONSTANT}
           | {"activation_bottleneck.temperature"}
           | {f"train.{k}" for k in COMMON_TRAIN} | {"train.run_name"})


def tcode(t: float) -> str:
    """0.05 -> '0p05' (run names carry tau_0 without a dot)."""
    return f"{t:g}".replace(".", "p")


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, prefix + k + "."))
        else:
            out[prefix + k] = list(v) if isinstance(v, tuple) else v
    return out


HEADER = """# {name}: laplace_policy on the code-carried cell ({k}, {j}).
# Base: {base} (code_residual: true, post_norm: false, the stable T=2 arm of
# docs/kj-vs-hard-topk-code-residual.tex), with the gate's training rule
# replaced -- every other field is the base file's (checked field by field by
# scripts/make_policy_cr_configs.py, which wrote this file):
{changes}
# {purpose}
"""


def write(k: int, j: int, tau0: float, schedule: str, name: str, out_dir: str,
          purpose: str, mode: str = "absolute") -> str:
    base = BASE.format(k=k, j=j)
    base_cfg = load_config(os.path.join(ROOT, base))
    b = base_cfg.activation_bottleneck
    assert (b.code_residual, b.post_norm, b.surrogate_mode, b.temperature,
            b.rblapsum_surrogate_scope) == (True, False, "rblapsum", 2.0, "pool"), base
    tree0 = base_cfg.to_dict()
    tree = copy.deepcopy(tree0)
    bn = tree["activation_bottleneck"]
    bn.update(COMMON_BN)
    bn["policy_temperature_mode"] = mode
    bn["temperature"] = float(tau0)
    if schedule == "constant":
        bn.update(CONSTANT)
    else:
        bn.update({"policy_temperature_schedule": "exponential",
                   "policy_temperature_final": round(0.2 * float(tau0), 10),
                   "policy_temperature_hold_steps": 1000,
                   "policy_temperature_anneal_steps": 17000})
    tree["train"].update(COMMON_TRAIN)
    tree["train"]["run_name"] = name
    final = config_from_dict(copy.deepcopy(tree)).to_dict()     # validates
    f0, f1 = flat(tree0), flat(final)
    moved = sorted(key for key in set(f0) | set(f1) if f0.get(key) != f1.get(key))
    stray = [key for key in moved if key not in ALLOWED]
    if stray:
        sys.exit(f"{name}: unexpected changes {stray}")
    changes = "\n".join(f"#   {key}: {f0.get(key)!r} -> {f1.get(key)!r}" for key in moved)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name + ".yaml")
    with open(path, "w") as f:
        f.write(HEADER.format(name=name, k=k, j=j, base=base, changes=changes,
                              purpose=purpose))
        yaml.safe_dump(final, f, sort_keys=False)
    if load_config(path).to_dict() != final:
        sys.exit(f"{path}: does not reload to what was written")
    c = load_config(path).activation_bottleneck
    print(f"{name:34s} K={c.k:<4d} J={c.j:<4d} {c.policy_temperature_mode:13s} tau0={c.temperature:<6g} "
          f"{c.policy_temperature_schedule:11s} tau_f={c.policy_temperature_final} "
          f"hold={c.policy_temperature_hold_steps} anneal={c.policy_temperature_anneal_steps} "
          f"code_residual={c.code_residual} post_norm={c.post_norm}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", nargs="*", type=float, default=None)
    ap.add_argument("--campaign", nargs="*", default=None, help="K:J:T0 per cell")
    ap.add_argument("--cells", nargs="*", default=None, help="K:J subset for --calibration")
    ap.add_argument("--out", default=os.path.join(ROOT, OUT))
    ap.add_argument("--mode", default="absolute",
                    choices=("absolute", "relative_span", "relative_b"))
    args = ap.parse_args()
    # run-name tag of the width mode: none for absolute (the first probes), rs / rb
    tag = {"absolute": "", "relative_span": "rs_", "relative_b": "rb_"}[args.mode]
    if args.calibration is not None:
        temps = tuple(args.calibration) or CAL_T
        cells = [tuple(int(x) for x in c.split(":")) for c in args.cells] if args.cells else CELLS
        for k, j in cells:
            for t in temps:
                write(k, j, t, "constant", f"vi_cal_{tag}k{k}_j{j}_t{tcode(t)}",
                      os.path.join(args.out, "calibration"),
                      "Calibration probe: constant width, run with train_guard.py --stop-step.",
                      mode=args.mode)
    if args.campaign:
        for spec in args.campaign:
            k, j, t0 = spec.split(":")
            k, j, t0 = int(k), int(j), float(t0)
            if (k, j) not in CELLS:
                sys.exit(f"({k}, {j}) is not a campaign cell")
            for schedule, arm, purpose in (
                    ("constant", "const", "Constant arm: tau = tau_0 for all 20k steps."),
                    ("exponential", "anneal",
                     "Annealed arm: tau_0 held 1000 steps, then exponential to 0.2 tau_0 at "
                     "step 18000.")):
                write(k, j, t0, schedule, f"vi_pol_{tag}k{k}_j{j}_t{tcode(t0)}_{arm}", args.out,
                      purpose + " tau_0 calibrated by the probes (configs/policy_cr/"
                      "calibration).", mode=args.mode)


if __name__ == "__main__":
    main()
