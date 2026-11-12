"""Figure of docs/separate-score-failure.md.

  python scripts/plot_separate_score_failure.py extract OUT.json NAME=metrics.jsonl ...
  python scripts/plot_separate_score_failure.py plot CURVES.json OVERLAP.json OUT.png

``extract`` keeps, per run, the logged training CE and gradient norm and the
validation CE up to step 2500.  ``plot`` draws three panels:

  (a) support overlap |S_l n S_{l+1}| / K between consecutive bottlenecks at
      initialization (analysis/code_carry_overlap.py): abs-TopK, separate
      scores, separate scores with the carry bonus gamma = 5;
  (b) training CE of hard Top-16 in the six configurations of the note;
  (c) the gradient norm before clipping of four of them (log scale).
"""
import json
import sys

STOP = 2500
RUNS = [  # name, label, color, linestyle
    ("gcr_hard_abs_k16", "abs-TopK", "black", "-"),
    ("gcr_hard_sep_k16", "sep., as configured", "tab:red", "-"),
    ("sx_sep_k16_cb0_detach", "sep., input detached", "tab:orange", "--"),
    ("sx_sep_k16_cb5", r"sep., bonus $\gamma$=5", "tab:purple", "--"),
    ("sx_sep_k16_cb5_detach", r"sep., bonus $\gamma$=5 + detached", "tab:blue", "-"),
    ("sx_sep_k16_cb1e4_detach", "sep., fixed support + detached", "tab:green", "-"),
]
GNORM = ["gcr_hard_abs_k16", "gcr_hard_sep_k16", "sx_sep_k16_cb5", "sx_sep_k16_cb5_detach"]


def extract(out, pairs):
    data = {}
    for item in pairs:
        name, path = item.split("=", 1)
        tr, val = [], []
        for line in open(path):
            r = json.loads(line)
            s = r.get("step", 0)
            if s > STOP:
                continue
            if "val/ce" in r:
                val.append([s, r["val/ce"]])
            elif "train/ce" in r:
                tr.append([s, r["train/ce"], r.get("train/grad_norm")])
        data[name] = {"train": sorted(tr), "val": sorted(val)}
    json.dump(data, open(out, "w"))
    print(f"{len(data)} runs -> {out}")


def plot(curves_path, overlap_path, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = json.load(open(curves_path))
    overlap = json.load(open(overlap_path))
    plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11, "legend.fontsize": 8.5})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))

    ax = axes[0]
    colors = {"abs-TopK": "black", "separate": "tab:red", "separate+bonus5": "tab:blue"}
    names = {"abs-TopK": "abs-TopK", "separate": "separate scores",
             "separate+bonus5": r"separate, bonus $\gamma$=5"}
    for row in overlap:
        xs = list(range(len(row["overlap"])))
        ax.plot(xs, row["overlap"], "o-", color=colors[row["label"]], label=names[row["label"]])
    ax.set_xticks(range(7), [rf"{i}$\to${i + 1}" for i in range(7)], fontsize=9)
    ax.set_xlabel(r"consecutive bottlenecks $l \to l+1$")
    ax.set_ylabel(r"support overlap  $|S_l \cap S_{l+1}|\,/\,K$")
    ax.set_ylim(-0.02, 1.0)
    ax.set_title("(a) carried coordinates kept, init, K=16")
    ax.grid(alpha=0.3)
    ax.legend(loc="center right")

    ax = axes[1]
    for name, label, color, ls in RUNS:
        tr = curves[name]["train"]
        ax.plot([t[0] for t in tr], [t[1] for t in tr], color=color, ls=ls, lw=1.4, label=label)
    ax.axhline(5.99, color="gray", lw=0.8, ls=":")
    ax.text(2480, 6.1, "unigram level", fontsize=8.5, color="gray", ha="right")
    ax.set_ylim(1.8, 8.5)
    ax.set_xlim(0, STOP)
    ax.set_xlabel("training step")
    ax.set_ylabel("training CE (nats)")
    ax.set_title("(b) training CE, hard Top-16")
    ax.grid(alpha=0.3)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.40))

    ax = axes[2]
    for name, label, color, ls in RUNS:
        if name not in GNORM:
            continue
        tr = [t for t in curves[name]["train"] if t[2] is not None]
        ax.plot([t[0] for t in tr], [t[2] for t in tr], color=color, ls=ls, lw=1.2, label=label)
    ax.set_yscale("log")
    ax.set_xlim(0, STOP)
    ax.set_xlabel("training step")
    ax.set_ylabel("gradient norm")
    ax.set_title("(c) gradient norm before clipping")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(sys.argv[2], sys.argv[3:])
    elif sys.argv[1] == "plot":
        plot(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print(__doc__)
