"""Configs for the code-residual K+J campaign (docs/kj-vs-hard-topk-code-residual.tex).

Every config written here is the archived ``config.json`` of one cell of the
K+J campaign (docs/kj-vs-hard-topk.tex: the 25 runs in ``RUNS``), migrated
through ``wsparse.config.config_from_dict`` onto the current schema and changed
in exactly one functional field, ``activation_bottleneck.code_residual: true``,
plus a new ``train.run_name``.  For every run the script prints

  - the fields the migration dropped (the removed subsystems of the 2026-09-28
    cleanup, guide section 9d; all inert in these runs),
  - the fields added since, with the default they take,
  - the fields whose value changed, which must be ``train.run_name`` only,

and refuses to write anything if some other value moved.  Each written file is
reloaded with ``load_config`` and checked to reproduce the migrated archived
config up to those two fields.

  python scripts/make_cr_kj_configs.py --archive DIR --out configs/cr_kj

``DIR`` holds one subdirectory per archived run with its ``config.json``
(``rclone copy gdrive:weight-sparsity/runs_<box>/<run> DIR/<run> --include config.json``).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wsparse.config import config_from_dict, load_config  # noqa: E402

#: archived run -> (Drive folder, new run name).  The new name keeps K, J and
#: the variant; `cr` marks the code residual, `ma` the madrid box.
RUNS = {
    # rblapsum through_rank_kappa, T = 2, no output norm
    "ca_rout_rblapsum_kappa_k32_j32_stab_t2_md_abs":    ("runs_california", "ma_cr_rbk_k32_j32_t2"),
    "ca_rout_rblapsum_kappa_k32_j96_kj_t2_md_abs":      ("runs_california", "ma_cr_rbk_k32_j96_t2"),
    "ca_rout_rblapsum_kappa_k32_j224_kj_t2_md_abs":     ("runs_california", "ma_cr_rbk_k32_j224_t2"),
    "ca_rout_rblapsum_kappa_k32_j480_stab_t2_md_abs":   ("runs_california", "ma_cr_rbk_k32_j480_t2"),
    "ca_rout_rblapsum_kappa_k64_j64_stab_t2_md_abs":    ("runs_california", "ma_cr_rbk_k64_j64_t2"),
    "ca_rout_rblapsum_kappa_k64_j192_kj_t2_md_abs":     ("runs_california", "ma_cr_rbk_k64_j192_t2"),
    "ca_rout_rblapsum_kappa_k64_j448_stab_t2_md_abs":   ("runs_california", "ma_cr_rbk_k64_j448_t2"),
    "ca_rout_rblapsum_kappa_k128_j128_stab_t2_md_abs":  ("runs_california", "ma_cr_rbk_k128_j128_t2"),
    "ca_rout_rblapsum_kappa_k128_j384_stab_t2_md_abs":  ("runs_california", "ma_cr_rbk_k128_j384_t2"),
    "ca_rout_rblapsum_kappa_k256_j256_kj_t2_md_abs":    ("runs_california", "ma_cr_rbk_k256_j256_t2"),
    # rblapsum through_rank_kappa, T = 1, RMSNorm on each bottleneck output
    "ca_rout_rblapsum_kappa_k32_j32_stab_pnorm_md_abs":   ("runs_california", "ma_cr_rbk_k32_j32_pnorm"),
    "ca_rout_rblapsum_kappa_k32_j96_kj_pnorm_md_abs":     ("runs_california", "ma_cr_rbk_k32_j96_pnorm"),
    "ca_rout_rblapsum_kappa_k32_j224_kj_pnorm_md_abs":    ("runs_california", "ma_cr_rbk_k32_j224_pnorm"),
    "ca_rout_rblapsum_kappa_k32_j480_stab_pnorm_md_abs":  ("runs_california", "ma_cr_rbk_k32_j480_pnorm"),
    "ca_rout_rblapsum_kappa_k64_j64_stab_pnorm_md_abs":   ("runs_california", "ma_cr_rbk_k64_j64_pnorm"),
    "ca_rout_rblapsum_kappa_k64_j192_kj_pnorm_md_abs":    ("runs_california", "ma_cr_rbk_k64_j192_pnorm"),
    "ca_rout_rblapsum_kappa_k64_j448_stab_pnorm_md_abs":  ("runs_california", "ma_cr_rbk_k64_j448_pnorm"),
    "ca_rout_rblapsum_kappa_k128_j128_stab_pnorm_md_abs": ("runs_california", "ma_cr_rbk_k128_j128_pnorm"),
    "ca_rout_rblapsum_kappa_k128_j384_stab_pnorm_md_abs": ("runs_california", "ma_cr_rbk_k128_j384_pnorm"),
    "ca_rout_rblapsum_kappa_k256_j256_kj_pnorm_md_abs":   ("runs_california", "ma_cr_rbk_k256_j256_pnorm"),
    # hard Top-K' with the post-norm
    "ko_rout_hard_k32_pnorm_md":     ("runs_korea",      "ma_cr_hard_k32_pnorm"),
    "ca_rout_hard_k64_kj_pnorm_md":  ("runs_california", "ma_cr_hard_k64_pnorm"),
    "ca_rout_hard_k128_kj_pnorm_md": ("runs_california", "ma_cr_hard_k128_pnorm"),
    "ca_rout_hard_k256_kj_pnorm_md": ("runs_california", "ma_cr_hard_k256_pnorm"),
    "ko_rout_hard_k512_pnorm_md":    ("runs_korea",      "ma_cr_hard_k512_pnorm"),
}

CHANGED_ALLOWED = {"train.run_name"}
#: `rblapsum_temperature` became `temperature`: reported as a rename, not a drop
RENAMED = {"activation_bottleneck.rblapsum_temperature": "activation_bottleneck.temperature"}


def flat(d, prefix=""):
    """``{"a.b": value}``; tuples read as lists (JSON has no tuples)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, prefix + k + "."))
        else:
            out[prefix + k] = list(v) if isinstance(v, tuple) else v
    return out


def header(old_name, folder, new_name, dropped, added, renamed):
    lines = [
        f"# {new_name}: the K+J campaign cell {old_name}",
        f"# (gdrive:weight-sparsity/{folder}/{old_name}, docs/kj-vs-hard-topk.tex)",
        "# with activation_bottleneck.code_residual: true as the ONLY functional change.",
        "#",
        "# Written by scripts/make_cr_kj_configs.py: the archived config.json migrated",
        "# through wsparse.config.config_from_dict onto the current schema.  Every",
        "# field the cleanup kept has the archived value; the fields it dropped were",
        "# inert in that run (removed subsystems, docs/vastai-agent-guide.md 9d):",
    ]
    lines += [f"#   {k} = {v!r}" for k, v in dropped]
    if renamed:
        lines.append("# renamed:")
        lines += [f"#   {a} -> {b} = {v!r}" for a, b, v in renamed]
    lines.append("# fields added since, at their defaults (code_residual is the change):")
    lines += [f"#   {k} = {v!r}" for k, v in added]
    lines += [
        "#",
        "# Self-contained on purpose: it pins what the archived run used, not today's",
        "# defaults (init_mode default under decouple, weight_decay ignored under",
        "# decouple, j inert under surrogate_mode hard).",
        "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plans = []
    for old_name, (folder, new_name) in RUNS.items():
        archived = json.load(open(os.path.join(args.archive, old_name, "config.json")))
        migrated = config_from_dict(copy.deepcopy(archived)).to_dict()
        fa, fm = flat(archived), flat(migrated)
        renamed = [(a, b, fa[a]) for a, b in RENAMED.items() if a in fa and b in fm and fa[a] == fm[b]]
        dropped = sorted((k, v) for k, v in fa.items() if k not in fm and k not in RENAMED)
        added = sorted((k, v) for k, v in fm.items() if k not in fa and k not in RENAMED.values())
        changed = sorted(k for k in fa if k in fm and fa[k] != fm[k])
        if changed:
            sys.exit(f"{old_name}: migration changed {changed}")
        # the one functional change, and the new name
        tree = copy.deepcopy(migrated)
        tree["activation_bottleneck"]["code_residual"] = True
        tree["train"]["run_name"] = new_name
        cfg = config_from_dict(copy.deepcopy(tree))     # validates (post_norm + code_residual allowed)
        final = cfg.to_dict()
        ff = flat(final)
        moved = sorted(k for k in fm if ff.get(k) != fm[k])
        if set(moved) != {"train.run_name", "activation_bottleneck.code_residual"}:
            sys.exit(f"{old_name}: unexpected changes {moved}")
        added = [(k, (True if k == "activation_bottleneck.code_residual" else v)) for k, v in added]
        plans.append((old_name, folder, new_name, dropped, added, renamed, final))

    for old_name, folder, new_name, dropped, added, renamed, final in plans:
        path = os.path.join(args.out, new_name + ".yaml")
        with open(path, "w") as f:
            f.write(header(old_name, folder, new_name, dropped, added, renamed))
            yaml.safe_dump(final, f, sort_keys=False)
        back = load_config(path).to_dict()
        if back != final:
            sys.exit(f"{path}: does not reload to what was written")
        b = final["activation_bottleneck"]
        print(f"{new_name:28s} <- {old_name:52s} {b['surrogate_mode']:8s} k={b['k']:<4d} j={b['j']:<4d} "
              f"T={b['temperature']:<4g} post_norm={b['post_norm']!s:5s} code_residual={b['code_residual']}")

    # the diff is the same for every run of a family; print it once per family
    seen = set()
    for old_name, folder, new_name, dropped, added, renamed, final in plans:
        key = (tuple(k for k, _ in dropped), tuple(added), tuple(renamed))
        if key in seen:
            continue
        seen.add(key)
        print(f"\n=== field diff archived -> written, as for {old_name} ===")
        print("dropped:")
        for k, v in dropped:
            print(f"   {k} = {v!r}")
        print("renamed:")
        for a, b, v in renamed:
            print(f"   {a} -> {b} = {v!r}")
        print("added (defaults; code_residual is the change):")
        for k, v in added:
            print(f"   {k} = {v!r}")


if __name__ == "__main__":
    main()
