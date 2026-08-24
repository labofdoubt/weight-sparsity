"""Benchmark 3: TinyStories Concept Localization (absorption proxy).

Not an implementation of the published SAEBench Feature Absorption metric --
it asks a narrower question:

    when the residual stream demonstrably carries a concept, does the canonical
    bottleneck feature carry it, or does the concept survive the bottleneck
    through some other combination of features?

Only concepts whose residual probe clears a globally fixed accuracy floor are
included: if the layer does not represent the concept, there is nothing whose
localization could be measured.

    python interpretability/concept_localization.py \
        --activations activations/bn_pmlp_k32j64_rel.npz \
        --benchmark benchmark_data --results results/bn_pmlp_k32j64_rel
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from sparse_probe import fit_l2

RESIDUAL_PROBE_MIN_ACCURACY = 0.80
GT_PRESENT_MIN_PROBABILITY = 0.80   # "the concept is clearly there"
SURVIVES_MIN_PROBABILITY = 0.50     # "it is still there after the bottleneck"
CONTRIBUTION_FLOOR = 1e-6           # below this the shares are meaningless


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -60, 60)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", required=True)
    ap.add_argument("--benchmark", default="benchmark_data")
    ap.add_argument("--results", required=True, help="output of sparse_probe.py")
    ap.add_argument("--identity-samples", type=int, default=256)
    args = ap.parse_args()

    blob = np.load(args.activations, allow_pickle=True)
    meta = json.loads(str(blob["meta"]))
    X, Z, XH = blob["x"], blob["z"], blob["x_hat"]
    W_dec, b_dec = blob["W_dec"], blob["b_dec"]      # (d_model, m), (d_model,)
    scale = float(blob["output_scale"])
    signed = bool(meta.get("signed_unfolded", meta["signed"]))

    picked = json.load(open(os.path.join(args.results, "selected_features.json")))
    probe = pd.read_csv(os.path.join(args.results, "sparse_probe_per_concept.csv"))
    probe = probe.set_index("concept")

    key = {
        (int(s), int(p)): i
        for i, (s, p) in enumerate(zip(blob["story_id"], blob["target_token_index"]))
    }
    frames = {
        sp: pd.read_parquet(os.path.join(args.benchmark, f"{sp}.parquet"))
        for sp in ("train", "validation", "test")
    }
    test = frames["test"]

    # --- decoder contribution identity (spec section 24) -------------------- #
    rng = np.random.default_rng(0)
    probe_dir = np.array(picked["directions"][sorted(picked["directions"])[0]]["w"])
    wbar = probe_dir / np.linalg.norm(probe_dir)
    sample = rng.choice(len(X), size=min(args.identity_samples, len(X)), replace=False)
    lhs = (XH[sample] - scale * b_dec) @ wbar
    rhs = scale * (Z[sample] * (W_dec.T @ wbar)).sum(1)
    identity_err = float(np.abs(lhs - rhs).max() / (np.abs(lhs).max() + 1e-9))
    print(f"[local] decoder identity max relative error: {identity_err:.2e}")
    if identity_err > 1e-3:
        raise SystemExit(
            f"decoder contribution identity failed ({identity_err:.2e}); "
            "the reconstruction model is wrong for this checkpoint"
        )

    rows = []
    for name in sorted(picked["selected"]):
        info = picked["selected"][name]
        direction = picked["directions"][name]
        w = np.array(direction["w"])
        b = float(direction["b"])
        residual_acc = float(probe.loc[name, "residual_accuracy"])
        if residual_acc < RESIDUAL_PROBE_MIN_ACCURACY:
            continue

        def part(split):
            f = frames[split]
            f = f[f.concept == name]
            ix = np.array([key[(int(a), int(c))] for a, c in
                           zip(f.story_id, f.target_token_index)])
            return ix, f.label.to_numpy(int)

        itr, ytr = part("train")
        iva, yva = part("validation")
        idx, y = part("test")
        x, z, xh = X[idx], Z[idx], XH[idx]

        s_before, s_after = x @ w + b, xh @ w + b
        acc_before = float(accuracy_score(y, (s_before > 0).astype(int)))
        acc_after = float(accuracy_score(y, (s_after > 0).astype(int)))
        # accuracy alone cannot separate "the bottleneck destroyed the concept"
        # from "the decision threshold no longer suits the output scale"; AUROC
        # is invariant to any monotone rescaling, so the pair distinguishes them
        auc_before = float(roc_auc_score(y, s_before))
        auc_after = float(roc_auc_score(y, s_after))
        norm_ratio = float(
            np.linalg.norm(xh, axis=1).mean() / (np.linalg.norm(x, axis=1).mean() + 1e-9)
        )
        cos = float(
            np.mean(
                (x * xh).sum(1)
                / (np.linalg.norm(x, axis=1) * np.linalg.norm(xh, axis=1) + 1e-9)
            )
        )

        # --- canonical feature activity ------------------------------------ #
        raw = int(info["raw_index"])
        zc = z[:, raw]
        if signed and info["sign"] == "-":
            active = zc < 0
        elif signed:
            active = zc > 0
        else:
            active = zc != 0

        # A probe fitted *on the bottleneck output*.  The spec applies w_c, which
        # lives in input coordinates, directly to x_hat -- correct for an
        # autoencoder, but this bottleneck has no reconstruction objective, so
        # x_hat is a learned transformation of x rather than an estimate of it
        # (measured cosine ~ 0).  In that regime "did the concept survive?" has
        # to be asked in the output's own coordinates, or the answer is chance
        # by construction.  Both are reported; _refit is the meaningful one
        # whenever reconstruction_cosine is near zero.
        out_probe, w_out, b_out = fit_l2(XH[itr], ytr, XH[iva], yva, xh, y)
        s_after_refit = xh @ w_out + b_out

        gt = (y == 1) & (sigmoid(s_before) >= GT_PRESENT_MIN_PROBABILITY)
        n_gt = int(gt.sum())
        missing = gt & ~active
        miss_rate = float(missing.sum() / n_gt) if n_gt else float("nan")

        def scatter_stats(scores):
            survives = sigmoid(scores) >= SURVIVES_MIN_PROBABILITY
            hits = missing & survives
            rate = float(hits.sum() / n_gt) if n_gt else float("nan")
            kept = float(hits.sum() / missing.sum()) if missing.sum() else float("nan")
            return rate, kept

        scatter_rate, retained = scatter_stats(s_after)
        scatter_rate_refit, retained_refit = scatter_stats(s_after_refit)

        # --- how the concept direction is split across active features ------ #
        wbar = w / np.linalg.norm(w)
        overlap = W_dec.T @ wbar                       # <d_j, wbar> per feature
        pos_mask = y == 1
        q = scale * z[pos_mask] * overlap              # signed contributions
        p = np.maximum(q, 0.0)
        total = p.sum(1)
        ok = total > CONTRIBUTION_FLOOR
        if ok.any():
            pk = p[ok]
            tot = total[ok]
            c1 = pk.max(1) / tot
            ccan = pk[:, raw] / tot
            neff = tot ** 2 / (pk ** 2).sum(1)
        else:
            c1 = ccan = neff = np.array([np.nan])

        rows.append(
            dict(
                concept=name,
                group=probe.loc[name, "group"],
                canonical_feature=info["feature_label"],
                ResidualProbeAccuracy=residual_acc,
                BottleneckOutputProbeAccuracy=acc_after,
                BottleneckOutputProbeAUROC=auc_after,
                probe_accuracy_before=acc_before,
                probe_auroc_before=auc_before,
                reconstruction_norm_ratio=norm_ratio,
                reconstruction_cosine=cos,
                Top1SparseProbeAccuracy=float(probe.loc[name, "top1_accuracy"]),
                n_gt_present=n_gt,
                CanonicalMissRate=miss_rate,
                ScatterRate=scatter_rate,
                RetainedElsewhere=retained,
                BottleneckOutputProbeAccuracy_refit=out_probe["accuracy"],
                BottleneckOutputProbeAUROC_refit=out_probe["auroc"],
                ScatterRate_refit=scatter_rate_refit,
                RetainedElsewhere_refit=retained_refit,
                MeanTop1ContributionShare=float(np.nanmean(c1)),
                MeanCanonicalContributionShare=float(np.nanmean(ccan)),
                MedianEffectiveContributorCount=float(np.nanmedian(neff)),
                MeanEffectiveContributorCount=float(np.nanmean(neff)),
            )
        )
        print(
            f"[local] {name:22} miss {miss_rate:.3f}  scatter {scatter_rate:.3f}"
            f"/{scatter_rate_refit:.3f}  "
            f"out {out_probe['accuracy']:.3f}  C1 {np.nanmean(c1):.3f}  "
            f"Ccan {np.nanmean(ccan):.3f}  Neff {np.nanmedian(neff):.1f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.results, "localization_per_concept.csv"), index=False)

    agg_path = os.path.join(args.results, "aggregate_metrics.json")
    agg = json.load(open(agg_path))
    numeric = [c for c in df.columns if df[c].dtype.kind == "f"]
    agg["localization"] = dict(
        n_concepts_evaluated=int(len(df)),
        residual_probe_threshold=RESIDUAL_PROBE_MIN_ACCURACY,
        gt_present_min_probability=GT_PRESENT_MIN_PROBABILITY,
        survives_min_probability=SURVIVES_MIN_PROBABILITY,
        decoder_identity_max_rel_error=identity_err,
        macro={c: float(df[c].mean()) for c in numeric},
        by_group={
            g: {c: float(sub[c].mean()) for c in numeric}
            for g, sub in df.groupby("group")
        },
    )
    json.dump(agg, open(agg_path, "w"), indent=2)

    print(
        f"\n[local] {len(df)} concepts passed the residual-probe floor "
        f"({RESIDUAL_PROBE_MIN_ACCURACY})"
        f"\n[local] probe accuracy   before {df.probe_accuracy_before.mean():.3f} "
        f"-> after {df.BottleneckOutputProbeAccuracy.mean():.3f}"
        f"\n[local] probe AUROC      before {df.probe_auroc_before.mean():.3f} "
        f"-> after {df.BottleneckOutputProbeAUROC.mean():.3f}"
        f"   (reconstruction cos {df.reconstruction_cosine.mean():.3f}, "
        f"norm ratio {df.reconstruction_norm_ratio.mean():.2f})"
        f"\n[local] CanonicalMissRate {df.CanonicalMissRate.mean():.3f}"
        f"\n[local] output probe refit acc {df.BottleneckOutputProbeAccuracy_refit.mean():.3f}"
        f"\n[local] ScatterRate       {df.ScatterRate.mean():.3f} "
        f"(refit {df.ScatterRate_refit.mean():.3f})"
        f"\n[local] RetainedElsewhere {df.RetainedElsewhere.mean():.3f} "
        f"(refit {df.RetainedElsewhere_refit.mean():.3f})"
        f"\n[local] Neff (median)     {df.MedianEffectiveContributorCount.mean():.2f}"
    )


if __name__ == "__main__":
    main()
