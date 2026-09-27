"""Figure 3 -- Precision@10 versus Degree/PageRank/Closeness EG@10 (Section 7.1).

All Table 4 and Table 7 points (109 method-dataset rows), mean +- population SD,
with the two-objective Pareto frontier of every panel.  Values come from the
same per-run results as Tables 4 and 7 (``_main_rows``).
"""
import math

import matplotlib.pyplot as plt
import pandas as pd

from tables._main_rows import load_rows, summaries, input_files  # noqa: F401
from tables._plotting import pareto_style as style
from tables._common import DATASET_LABEL, DATASET_ORDER

ID = "F03"
# Historic method names and ordering keep the published colour assignment.
HISTORIC = {"Random": "random", "HL-GNN": "HLGNN", "HL-GNN+RA": "HLGNN+RA", "Cross-LinkBias": "Cross-Link Bias",
            "TailSTEAK": "TailSteak"}
ORDER = ["AA", "CAFE-In", "CAFE-In-B", "CAFE-P", "CAFE-Pre", "CAFE-Pre-B", "CI-In", "CI-Pre", "CN",
         "Cross-Link Bias", "FairAdj", "HLGNN", "HLGNN+RA", "LBA", "LTLP", "LightGCN", "MB-Post", "MORAL",
         "PA", "RA", "SSNet", "TailSteak", "random"]
DISPLAY = {"random": "Random", "HLGNN": "HL-GNN", "HLGNN+RA": "HL-GNN+RA", "TailSteak": "TailSTEAK",
           "Best per metric": "Axis maximum*"}
LABELS = {"prec": "Precision", "deg_eg": "Degree EG@10", "pr_eg": "PageRank EG@10", "cl_eg": "Closeness EG@10"}


def points():
    rows = []
    for (ds, method), spec in load_rows().items():
        item = {"Dataset": ds, "Method": HISTORIC.get(method, method), "n": int(spec["n"])}
        for col, s in summaries(spec).items():
            item[f"{LABELS[col]}_Mean"], item[f"{LABELS[col]}_Std"] = s["mean"], s["sd"]
        rows.append(item)
    return pd.DataFrame(rows)


def build(out):
    df = points()
    assert len(df) == 109 and all(math.isfinite(v) for v in df.filter(regex="_(Mean|Std)$").to_numpy().ravel())
    cmap = plt.get_cmap("tab20")
    colors = {m: cmap(i % 20) for i, m in enumerate(m for m in ORDER if m not in style.FIXED_COLORS)}
    colors.update(style.FIXED_COLORS)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(5, 3, figsize=(15.6, 20.2))
    fig.subplots_adjust(left=.105, right=.81, bottom=.09, top=.955, wspace=.24, hspace=.19)
    fronts = []
    for i, ds in enumerate(DATASET_ORDER):
        block = df[df.Dataset == ds].reset_index(drop=True)
        best = style.find_best_per_column(block)
        for j, metric in enumerate(style.EG_METRICS):
            ax = axes[i, j]
            style.plot_single_panel(ax, block, metric, colors, best, show_ylabel=False, show_xlabel=i == 4,
                                    label_fontsize=12, tick_fontsize=9, annotation_fontsize=7)
            for annotation in list(ax.texts):
                annotation.remove()
            for method in dict.fromkeys([best["Precision"], best[metric]]):
                p = block[block.Method == method].iloc[0]
                right = method == best["Precision"]
                ax.annotate(DISPLAY.get(method, method), (p.Precision_Mean, p[metric + "_Mean"]),
                            xytext=(-6, 7) if right else (6, 7), textcoords="offset points",
                            ha="right" if right else "left", fontsize=8, color="#333333")
            ax.margins(x=.08, y=.13)
            if i == 0:
                ax.set_title(metric, fontsize=13, fontweight="bold", pad=11)
            if j == 0:
                ax.set_ylabel(DATASET_LABEL[ds], fontsize=13, fontweight="bold", labelpad=15)
            xs, ys = block.Precision_Mean.to_numpy(), block[metric + "_Mean"].to_numpy()
            fronts += [dict(dataset=ds, metric=metric, method=block.iloc[a].Method, precision=float(xs[a]),
                            eg=float(ys[a])) for a in style._pareto_front_2d(xs, ys)]
    handles = style._build_legend_handles(ORDER, colors)
    for h in handles:
        h.set_label(DISPLAY.get(h.get_label(), h.get_label()))
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(.825, .53), fontsize=10, frameon=True,
               labelspacing=.65, handletextpad=.65)
    fig.suptitle("Precision@10 versus Equality Gain@10", fontsize=17, fontweight="bold", y=.989)
    fig.text(.105, .022, "Fixed-split mean ± SD; n=10, except deterministic heuristics (n=1) and nine Pokec-n "
             "baselines (n=8).\nExact corrected Closeness for Brightkite/Pokec-n. Dashed lines: two-objective "
             "Pareto frontiers.\n*Diamond: highest precision or EG; CAFE variants retain their own marker.",
             fontsize=10, va="bottom")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{ID}.pdf")
    fig.savefig(out / f"{ID}.png", dpi=140)
    plt.close(fig)
    df.to_csv(out / f"{ID}_points.csv", index=False)
    pd.DataFrame(fronts).to_csv(out / f"{ID}_frontiers.csv", index=False)
    return None
