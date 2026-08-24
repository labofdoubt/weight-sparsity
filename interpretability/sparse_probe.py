"""Benchmark 1: top-1 sparse probe, plus the full-residual baseline.

For each concept:

  * pick the single bottleneck feature with the largest train-set mean
    difference between classes (selection touches training data only);
  * fit a one-dimensional logistic regression on that feature;
  * fit an L2 logistic regression on the full residual vector entering the
    bottleneck, as the "is this concept even here" baseline;
  * sweep k = 1,2,4,8,16 features to show how concentrated the concept is.

Also runs the sanity checks from spec section 24.

    python interpretability/sparse_probe.py \
        --activations activations/bn_pmlp_k32j64_rel.npz \
        --benchmark benchmark_data --out results/bn_pmlp_k32j64_rel
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

K_CURVE = (1, 2, 4, 8, 16)
C_GRID = (0.01, 0.1, 1.0, 10.0)
RESIDUAL_PROBE_MIN_ACCURACY = 0.80  # fixed globally, never tuned per model


def unfold_signed(z: np.ndarray) -> np.ndarray:
    """``m`` signed features -> ``2m`` virtual features (spec section 2).

    Positive and negative excursions of one index may mean different things, so
    they are scored separately.  Virtual feature ``j`` is ``z_j+`` for j < m and
    ``z_{j-m}-`` above.
    """
    return np.concatenate([np.maximum(z, 0.0), np.maximum(-z, 0.0)], axis=1)


def feature_label(j: int, m: int, signed: bool) -> str:
    if not signed:
        return str(j)
    return f"{j % m}{'+' if j < m else '-'}"


def metrics(y, pred, score) -> Dict[str, float]:
    return dict(
        accuracy=float(accuracy_score(y, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        auroc=float(roc_auc_score(y, score)) if len(set(y)) > 1 else float("nan"),
        f1=float(f1_score(y, pred, zero_division=0)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
    )


def fit_1d(train_x, train_y, test_x, test_y) -> Dict[str, float]:
    if train_x.std() == 0:
        return dict(accuracy=0.5, balanced_accuracy=0.5, auroc=0.5, f1=0.0,
                    precision=0.0, recall=0.0)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_x.reshape(-1, 1), train_y)
    s = clf.decision_function(test_x.reshape(-1, 1))
    return metrics(test_y, (s > 0).astype(int), s)


def fit_l2(train_x, train_y, val_x, val_y, test_x, test_y, standardize=True):
    """L2 logistic regression, C chosen on validation.

    Returns test metrics plus the weight vector expressed in the *original*
    coordinate system, so it can be used as a direction in residual space.
    """
    mu = train_x.mean(0) if standardize else np.zeros(train_x.shape[1])
    sd = train_x.std(0) + 1e-8 if standardize else np.ones(train_x.shape[1])
    tr, va, te = (train_x - mu) / sd, (val_x - mu) / sd, (test_x - mu) / sd

    best, best_acc = None, -1.0
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(tr, train_y)
        acc = accuracy_score(val_y, clf.predict(va))
        if acc > best_acc:
            best, best_acc, best_C = clf, acc, C

    s = best.decision_function(te)
    out = metrics(test_y, (s > 0).astype(int), s)
    out["C"] = float(best_C)
    # undo standardisation so w lives in residual-stream coordinates
    w = best.coef_[0] / sd
    b = float(best.intercept_[0] - np.dot(best.coef_[0], mu / sd))
    return out, w, b


def select_features(z_train, y_train, top: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Rank by train-set mean difference; never select an always-zero feature."""
    delta = z_train[y_train == 1].mean(0) - z_train[y_train == 0].mean(0)
    alive = np.abs(z_train).max(0) > 0
    delta = np.where(alive, delta, -np.inf)
    order = np.argsort(-delta)
    return order[:top], delta


def bootstrap_ci(values, n=10000, seed=0):
    if len(values) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    draws = rng.choice(v, size=(n, len(v)), replace=True).mean(1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", required=True)
    ap.add_argument("--benchmark", default="benchmark_data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--signed", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--random-features", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    blob = np.load(args.activations, allow_pickle=True)
    meta = json.loads(str(blob["meta"]))
    m = int(meta["bottleneck_width"])
    signed = meta["signed"] if args.signed == "auto" else args.signed == "yes"
    meta["signed_unfolded"] = bool(signed)

    key = {
        (int(s), int(p)): i
        for i, (s, p) in enumerate(zip(blob["story_id"], blob["target_token_index"]))
    }
    X, Zraw, = blob["x"], blob["z"]
    Z = unfold_signed(Zraw) if signed else Zraw
    print(
        f"[probe] {meta['run_name']} layer {meta['evaluation_layer']} "
        f"({meta['bottleneck_location']}) -> {Z.shape[1]} "
        f"{'virtual ' if signed else ''}features"
    )

    frames = {
        s: pd.read_parquet(os.path.join(args.benchmark, f"{s}.parquet"))
        for s in ("train", "validation", "test")
    }
    stats = json.load(open(os.path.join(args.benchmark, "dataset_stats.json")))

    # spec section 24: the splits must be disjoint at the story level
    sids = {s: set(f.story_id) for s, f in frames.items()}
    assert not (sids["train"] & sids["validation"]), "train/val story leakage"
    assert not (sids["train"] & sids["test"]), "train/test story leakage"
    assert not (sids["validation"] & sids["test"]), "val/test story leakage"

    rng = np.random.default_rng(args.seed)
    rows, selected, directions = [], {}, {}

    concepts = sorted(set(frames["train"].concept))
    for name in concepts:
        def part(split):
            d = frames[split]
            d = d[d.concept == name]
            idx = np.array([key[(int(a), int(b))] for a, b in
                            zip(d.story_id, d.target_token_index)])
            return idx, d.label.to_numpy(int)

        itr, ytr = part("train")
        iva, yva = part("validation")
        ite, yte = part("test")
        ztr, zte = Z[itr], Z[ite]

        top, delta = select_features(ztr, ytr)
        j = int(top[0])
        one = fit_1d(ztr[:, j], ytr, zte[:, j], yte)

        res, w, b = fit_l2(X[itr], ytr, X[iva], yva, X[ite], yte)

        curve = {}
        for k in K_CURVE:
            cols = top[:k]
            clf = LogisticRegression(max_iter=2000)
            clf.fit(ztr[:, cols], ytr)
            curve[k] = float(accuracy_score(yte, clf.predict(zte[:, cols])))

        # --- sanity checks -------------------------------------------------- #
        shuffled = rng.permutation(ytr)
        s_top, _ = select_features(ztr, shuffled)
        shuffle_acc = fit_1d(
            ztr[:, s_top[0]], shuffled, zte[:, s_top[0]], rng.permutation(yte)
        )["accuracy"]
        alive = np.flatnonzero(np.abs(ztr).max(0) > 0)
        picks = rng.choice(alive, size=min(args.random_features, len(alive)), replace=False)
        rand_acc = float(
            np.mean([fit_1d(ztr[:, r], ytr, zte[:, r], yte)["accuracy"] for r in picks])
        )

        selected[name] = dict(
            feature_index=j,
            feature_label=feature_label(j, m, signed),
            raw_index=int(j % m) if signed else j,
            sign="+" if (not signed or j < m) else "-",
            top10=[feature_label(int(t), m, signed) for t in top],
            top10_index=[int(t) for t in top],
            top10_delta=[float(delta[int(t)]) for t in top],
        )
        directions[name] = dict(w=w.tolist(), b=b, residual_accuracy=res["accuracy"])

        rows.append(
            dict(
                concept=name,
                group=stats["concepts"][name]["group"],
                core=stats["concepts"][name]["core"],
                best_feature=selected[name]["feature_label"],
                n_test=len(yte),
                top1_accuracy=one["accuracy"],
                top1_balanced_accuracy=one["balanced_accuracy"],
                top1_auroc=one["auroc"],
                top1_f1=one["f1"],
                top1_precision=one["precision"],
                top1_recall=one["recall"],
                residual_accuracy=res["accuracy"],
                residual_auroc=res["auroc"],
                residual_C=res["C"],
                **{f"acc_k{k}": v for k, v in curve.items()},
                shuffled_label_accuracy=shuffle_acc,
                random_feature_accuracy=rand_acc,
            )
        )
        print(
            f"[probe] {name:22} top1 {one['accuracy']:.3f}  "
            f"residual {res['accuracy']:.3f}  k16 {curve[16]:.3f}  "
            f"rand {rand_acc:.3f}  shuffled {shuffle_acc:.3f}"
        )

    df = pd.DataFrame(rows).sort_values("concept")
    df.to_csv(os.path.join(args.out, "sparse_probe_per_concept.csv"), index=False)

    core = df[df.core]
    agg = dict(
        meta=meta,
        n_concepts=int(len(df)),
        n_core_concepts=int(len(core)),
        SparseProbeTop1=float(core.top1_accuracy.mean()),
        SparseProbeTop1_ci95=bootstrap_ci(core.top1_accuracy),
        ResidualProbeAccuracy=float(core.residual_accuracy.mean()),
        ResidualProbeAccuracy_ci95=bootstrap_ci(core.residual_accuracy),
        k_curve={f"k{k}": float(core[f"acc_k{k}"].mean()) for k in K_CURVE},
        by_group={
            g: dict(
                n=int(len(sub)),
                SparseProbeTop1=float(sub.top1_accuracy.mean()),
                ResidualProbeAccuracy=float(sub.residual_accuracy.mean()),
            )
            for g, sub in core.groupby("group")
        },
        sanity=dict(
            shuffled_label_accuracy=float(core.shuffled_label_accuracy.mean()),
            random_feature_accuracy=float(core.random_feature_accuracy.mean()),
            top1_minus_random=float(
                (core.top1_accuracy - core.random_feature_accuracy).mean()
            ),
            splits_disjoint=True,
            residual_probe_threshold=RESIDUAL_PROBE_MIN_ACCURACY,
            n_concepts_passing_residual_threshold=int(
                (core.residual_accuracy >= RESIDUAL_PROBE_MIN_ACCURACY).sum()
            ),
        ),
    )
    json.dump(agg, open(os.path.join(args.out, "aggregate_metrics.json"), "w"), indent=2)
    json.dump(
        dict(meta=meta, selected=selected, directions=directions),
        open(os.path.join(args.out, "selected_features.json"), "w"),
    )

    print(
        f"\n[probe] SparseProbeTop1 = {agg['SparseProbeTop1']:.4f} "
        f"(95% CI {agg['SparseProbeTop1_ci95'][0]:.3f}-{agg['SparseProbeTop1_ci95'][1]:.3f})"
        f"\n[probe] ResidualProbe   = {agg['ResidualProbeAccuracy']:.4f}"
        f"\n[probe] sanity: shuffled {agg['sanity']['shuffled_label_accuracy']:.3f} "
        f"| random feature {agg['sanity']['random_feature_accuracy']:.3f} "
        f"| top1-random +{agg['sanity']['top1_minus_random']:.3f}"
    )


if __name__ == "__main__":
    main()
