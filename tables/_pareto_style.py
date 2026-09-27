"""Plot helpers for Figure 3 (precision versus Equality Gain Pareto panels).

Markers: CAFE variants keep their own marker; a diamond marks the method with
the highest precision or the highest EG in a panel; circles for the rest.
Dashed lines are the two-objective Pareto frontier of each panel.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASETS = ["brightkite", "ego-facebook", "oklahoma", "pokec_n", "unc"]
EG_METRICS = ["Degree EG@10", "PageRank EG@10", "Closeness EG@10"]

# Methods to exclude from plots. Names here must match what appears in
# table1_precision_equality_gain.csv, which stores already-renamed short names
# (see METHOD_RENAME in generate_final_tables.py).
EXCLUDE_METHODS = {
    "pure_gnn",
    "node2vec",
    "deepwalk",
    # These historical ablations are not defined in the paper tables/text.
    "MB-Pre",
    "MB-In",
    "CI-Post",
    "WeakTie",
    "WeakTie (no re-rank)",
}

# Weak-tie diversification rows emitted by generate_final_tables.py
# (WEAK_TIE_GROUP / WEAK_TIE_CONTROL_GROUP there).  --no_weak_tie drops both.
WEAK_TIE_METHODS = {"WeakTie", "WeakTie (no re-rank)"}

# ASF weights — identical to generate_final_tables.py
ASF_WEIGHTS = np.array([0.5, 0.3, 0.1, 0.1])

# Methods with a fixed colour; all others take tab20 colours in order.
FIXED_COLORS = {
    "HLGNN+RA": "#555555",  # fusion-only control (HL-GNN + beta*RA)
}

# The four objectives used in the ASF (column names after wrangling)
ASF_OBJ_COLS = [
    "Precision_Mean",
    "Degree EG@10_Mean",
    "PageRank EG@10_Mean",
    "Closeness EG@10_Mean",
]


# ---------------------------------------------------------------------------
# Parsing helpers  (identical to original)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------


def find_best_per_column(df_ds):
    """
    For one dataset, find the method with the highest value in each column:
      - "Precision" → highest Precision_Mean
      - each EG metric → highest <EG>_Mean

    Returns dict  {column_label: method_name}.
    """
    best = {}

    prec_valid = df_ds.dropna(subset=["Precision_Mean"])
    if not prec_valid.empty:
        best["Precision"] = prec_valid.loc[
            prec_valid["Precision_Mean"].idxmax(), "Method"
        ]

    for eg in EG_METRICS:
        col = f"{eg}_Mean"
        eg_valid = df_ds.dropna(subset=[col])
        if not eg_valid.empty:
            best[eg] = eg_valid.loc[eg_valid[col].idxmax(), "Method"]

    return best


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

CAFE_METHODS = {"CAFE-In-B", "CAFE-Pre-B", "CAFE-In", "CAFE-Pre"}
CAFE_MARKERS = {
    "CAFE-In-B":  "^",   # triangle up
    "CAFE-Pre-B": "v",   # triangle down
    "CAFE-In":    "s",   # square
    "CAFE-Pre":   "p",   # pentagon
}


def _pareto_front_2d(xs, ys):
    """
    Return indices of the 2-D Pareto front (both objectives maximised).
    Points are returned sorted by x for easy line drawing.
    """
    n = len(xs)
    pts = np.column_stack([xs, ys])
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if pts[j, 0] >= pts[i, 0] and pts[j, 1] >= pts[i, 1] and \
               (pts[j, 0] > pts[i, 0] or pts[j, 1] > pts[i, 1]):
                is_dominated[i] = True
                break
    front_idx = np.where(~is_dominated)[0]
    # sort by x
    order = np.argsort(pts[front_idx, 0])
    return front_idx[order]


def plot_single_panel(ax, df_ds, eg_metric, color_map,
                      best_per_col,
                      show_ylabel=True, show_xlabel=True,
                      label_fontsize=14, tick_fontsize=11,
                      annotation_fontsize=9):
    """
    Scatter + error bars for one (dataset, eg_metric) panel.

    Markers:
    triangle — CAFE-In / CAFE-Pre  (emphasised)
    diamond  — best-per-column  (best Precision OR best this-EG)
    circle   — everything else

    Pareto frontier drawn with dashed line.
    """
    mean_y_col = f"{eg_metric}_Mean"
    std_y_col = f"{eg_metric}_Std"

    df_valid = df_ds.dropna(subset=["Precision_Mean", mean_y_col]).copy()
    if df_valid.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=label_fontsize, color="gray")
        if show_ylabel:
            ax.set_ylabel(eg_metric, fontsize=label_fontsize)
        if show_xlabel:
            ax.set_xlabel("Precision@10", fontsize=label_fontsize)
        return

    # Determine which methods get special markers in *this* subplot
    diamond_methods = set()
    best_prec = best_per_col.get("Precision")
    best_eg = best_per_col.get(eg_metric)
    if best_prec is not None:
        diamond_methods.add(best_prec)
    if best_eg is not None:
        diamond_methods.add(best_eg)

    # --- Draw Pareto frontier (dashed) ---
    xs_all = df_valid["Precision_Mean"].values
    ys_all = df_valid[mean_y_col].values
    pf_idx = _pareto_front_2d(xs_all, ys_all)
    pareto_methods = set(df_valid.iloc[pf_idx]["Method"].tolist())
    if len(pf_idx) >= 2:
        ax.plot(xs_all[pf_idx], ys_all[pf_idx],
                linestyle="--", linewidth=1.2, color="gray", alpha=0.7,
                zorder=1, label="Pareto frontier")
    elif len(pf_idx) == 1:
        ax.plot(xs_all[pf_idx], ys_all[pf_idx],
                marker="x", markersize=6, color="gray", alpha=0.7,
                linestyle="None", zorder=1)

    for _, row in df_valid.iterrows():
        method = row["Method"]
        x = row["Precision_Mean"]
        y = row[mean_y_col]
        xerr = row["Precision_Std"] if pd.notna(row["Precision_Std"]) else 0.0
        yerr = row[std_y_col] if pd.notna(row[std_y_col]) else 0.0
        c = color_map.get(method, "#4C72B0")

        if method in CAFE_METHODS:
            # CAFE variants — unique markers, own colour from color_map
            cafe_marker = CAFE_MARKERS.get(method, "^")
            ax.errorbar(
                x, y, xerr=xerr, yerr=yerr,
                fmt=cafe_marker, markersize=13, capsize=2, elinewidth=0.8,
                color=c, ecolor="#AAAAAA",
                markeredgecolor="black", markeredgewidth=0.9,
                zorder=4,
            )
            if method in pareto_methods:
                ax.annotate(
                    method, (x, y),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=annotation_fontsize, color=c,
                    fontweight="bold", zorder=5,
                )
        elif method in diamond_methods:
            # Diamond — best per-column
            ax.errorbar(
                x, y, xerr=xerr, yerr=yerr,
                fmt="D", markersize=10, capsize=2, elinewidth=0.8,
                color=c, ecolor="#AAAAAA",
                markeredgecolor="black", markeredgewidth=0.7,
                zorder=3,
            )
            if method in pareto_methods:
                ax.annotate(
                    method, (x, y),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=annotation_fontsize, color=c,
                    fontweight="bold", zorder=5,
                )
        else:
            # Circle — regular
            ax.errorbar(
                x, y, xerr=xerr, yerr=yerr,
                fmt="o", markersize=7, capsize=2, elinewidth=0.8,
                color=c, ecolor="#AAAAAA",
                markeredgecolor="black", markeredgewidth=0.5,
                zorder=2,
            )

    if show_ylabel:
        ax.set_ylabel(eg_metric, fontsize=label_fontsize)
    if show_xlabel:
        ax.set_xlabel("Precision@10", fontsize=label_fontsize)

    ax.tick_params(labelsize=tick_fontsize)
    ax.grid(True, alpha=0.3, linewidth=0.5)


def _build_legend_handles(methods_in_data, color_map, legend_fontsize=12):
    """Build legend handles including special marker explanations."""
    handles = []
    for m in methods_in_data:
        c = color_map.get(m, "#4C72B0")
        if m in CAFE_METHODS:
            marker = CAFE_MARKERS.get(m, "^")
            ms = 11
        else:
            marker, ms = "o", 9
        handles.append(
            mlines.Line2D([], [], color=c, marker=marker,
                          linestyle="None", markersize=ms, label=m,
                          markeredgecolor="black", markeredgewidth=0.5)
        )
    # Special markers
    handles.append(
        mlines.Line2D([], [], color="gray", marker="D", linestyle="None",
                      markersize=10, markeredgecolor="black",
                      markeredgewidth=0.7,
                      label="Best per metric")
    )
    handles.append(
        mlines.Line2D([], [], color="gray", linestyle="--", linewidth=1.2,
                      label="Pareto frontier")
    )
    return handles


