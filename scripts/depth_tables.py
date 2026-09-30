"""Tables for docs/stream-bottleneck-depth*.tex, printed as LaTeX rows.

Reads docs/figures/depth/data/curves.json (the curve extract) so the numbers in
the notes come from the data, not from transcription.

  python scripts/depth_tables.py
"""

from __future__ import annotations

import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
curves = json.load(open(os.path.join(ROOT, "docs", "figures", "depth", "data", "curves.json")))

ROWS = [
    ("TopK, post-norm (\\code{al\\_500m\\_hard\\_k512\\_pnorm\\_bp\\_std})", "al_500m_hard_k512_pnorm_bp_std"),
    ("TopK, no norm (\\code{al\\_500m\\_hard\\_k512\\_nopnorm\\_bp\\_std})", "al_500m_hard_k512_nopnorm_bp_std"),
    ("value shift energy, post-norm", "li_24L_k512_pnorm_shift_energy"),
    ("\\(K=N\\) (no selection), post-norm", "li_24L_kN_pnorm"),
    ("\\(K=N\\) (no selection), no norm", "li_24L_kN_nopnorm"),
    ("value shift energy, no norm", "li_24L_k512_nopnorm_shift_energy"),
    ("value shift fixed, no norm", "li_24L_k512_nopnorm_shift_fixed"),
    ("code residual", "li_24L_k512_coderes_a1"),
    ("code residual, \\(\\alpha=0.3\\)", "li_24L_k512_coderes_a03"),
    ("code residual, \\(K=32\\)", "li_24L_k32_coderes_a1"),
    ("dense, no bottleneck", "li_24L_dense"),
    ("8 layers: TopK, no norm (\\code{al\\_8L\\_hard\\_k512\\_nopnorm\\_bp\\_std})", "al_8L_hard_k512_nopnorm_bp_std"),
    ("8 layers: code residual", "li_8L_k512_coderes_a1"),
]


def val_at(run, step):
    c = curves.get(run)
    if not c:
        return None
    return dict(c["val"]).get(step)


def fmt(v):
    return "--" if v is None else f"{v:.3f}"


def last(run):
    c = curves.get(run)
    if not c or not c["val"]:
        return None, None
    s, v = c["val"][-1]
    return s, v


print("% run & 500 & 2000 & 10000 & 20000 & last")
for label, run in ROWS:
    s, v = last(run)
    cells = [fmt(val_at(run, st)) for st in (500, 2000, 10000, 20000)]
    tail = "--" if s is None else f"{v:.3f} @{s}"
    print(f"{label} & " + " & ".join(cells) + f" & {tail} \\\\")

print("\n% depth sweep, val CE at step 2000")
for fam, runs in (
    ("TopK, post-norm", ["li_8L_k512_pnorm", "li_12L_k512_pnorm", "li_16L_k512_pnorm",
                         "al_500m_hard_k512_pnorm_bp_std"]),
    ("TopK, no norm", ["al_8L_hard_k512_nopnorm_bp_std", "li_12L_k512_nopnorm",
                       "li_16L_k512_nopnorm", "al_500m_hard_k512_nopnorm_bp_std"]),
    ("code residual", ["li_8L_k512_coderes_a1", "li_12L_k512_coderes_a1",
                       "li_16L_k512_coderes_a1", "li_24L_k512_coderes_a1"]),
):
    print(f"{fam} & " + " & ".join(fmt(val_at(r, 2000)) for r in runs) + " \\\\")
