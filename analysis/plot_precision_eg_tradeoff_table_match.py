"""
Precision vs Equality-Gain Trade-off — Table-Matched Highlights
================================================================
  - Star    : best overall 4D trade-off method per dataset (ASF-based,
                   same weights as generate_final_tables.py)
  - Diamond : best-per-column methods (highest Precision@10 or highest
                   EG metric for that subplot)
  - Circle  : all other methods

Output:
  - full_comparison_matrix_table_match.pdf   (grid, all datasets × EG metrics)
  - <DatasetName>_summary_table_match.pdf    (1×3 per dataset)
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ---------------------------------------------------------------------------
# Configuration  (mirrors plot_precision_eg_tradeoff.py)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "table1_precision_equality_gain.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots")

DATASETS = ["brightkite", "ego-facebook", "oklahoma", "pokec_n", "unc"]
EG_METRICS = ["Degree EG@10", "PageRank EG@10", "Closeness EG@10"]

# Methods to exclude from plots. Names here must match what appears in
# table1_precision_equality_gain.csv, which stores already-renamed short names
# (see METHOD_RENAME in generate_final_tables.py).
EXCLUDE_METHODS = {"pure_gnn", "node2vec", "deepwalk"}

# Weak-tie diversification rows emitted by generate_final_tables.py
# (WEAK_TIE_GROUP / WEAK_TIE_CONTROL_GROUP there).  --no_weak_tie drops both.
WEAK_TIE_METHODS = {"WeakTie", "WeakTie (no re-rank)"}

# ASF weights — identical to generate_final_tables.py
ASF_WEIGHTS = np.array([0.5, 0.3, 0.1, 0.1])

# Methods added after the RecSys submission. They get FIXED colors so the
# tab20 assignment of all previously published methods does not shift
# (reviewers compare against the old figures).
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

def parse_mean_std(cell):
    """Parse a cell string into (mean, std).  Returns (NaN, NaN) for '-'."""
    if not isinstance(cell, str):
        try:
            return float(cell), 0.0
        except (ValueError, TypeError):
            return np.nan, np.nan

    cell = cell.strip()
    if cell == "-":
        return np.nan, np.nan

    if "±" in cell:
        parts = cell.split("±")
        return float(parts[0].strip()), float(parts[1].strip())

    return float(cell), 0.0


def load_and_wrangle(csv_path):
    """
    Return a DataFrame with columns:
        Method, Dataset, Precision_Mean, Precision_Std,
        <EG_metric>_Mean, <EG_metric>_Std   (for each EG metric)
    One row per (Method, Dataset) pair.
    """
    raw = pd.read_csv(csv_path)
    methods = raw["Method"].unique()
    available_datasets = [d for d in DATASETS if d in raw.columns]

    records = []
    for method in methods:
        method_rows = raw[raw["Method"] == method]

        prec_row = method_rows[method_rows["Metric"] == "Precision@10"]
        if prec_row.empty:
            continue

        for dataset in available_datasets:
            prec_mean, prec_std = parse_mean_std(prec_row.iloc[0][dataset])
            if np.isnan(prec_mean):
                continue

            rec = {
                "Method": method,
                "Dataset": dataset,
                "Precision_Mean": prec_mean,
                "Precision_Std": prec_std,
            }

            for eg in EG_METRICS:
                eg_row = method_rows[method_rows["Metric"] == eg]
                if eg_row.empty:
                    eg_mean, eg_std = np.nan, np.nan
                else:
                    eg_mean, eg_std = parse_mean_std(eg_row.iloc[0][dataset])
                rec[f"{eg}_Mean"] = eg_mean
                rec[f"{eg}_Std"] = eg_std

            records.append(rec)

    df = pd.DataFrame(records)
    df = df[~df["Method"].isin(EXCLUDE_METHODS)].copy()
    return df


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------

def _non_dominated_front0(F):
    """
    Return indices of non-dominated points (Pareto front 0).
    F is for *minimisation* (lower is better).
    """
    n = len(F)
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j <= i on all and j < i on at least one
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                is_dominated[i] = True
                break
    return np.where(~is_dominated)[0]


def find_best_asf_method(df_ds):
    """
    Among all methods for one dataset, find the best overall trade-off
    using 4-objective Pareto + ASF (same logic as generate_final_tables.py).

    Returns the method name (str) or None.
    """
    df_valid = df_ds.dropna(subset=ASF_OBJ_COLS).copy()
    if df_valid.empty:
        return None
    if len(df_valid) == 1:
        return df_valid.iloc[0]["Method"]

    F_max = df_valid[ASF_OBJ_COLS].values  # all objectives to maximise

    # Normalise to [0, 1]
    obj_min = F_max.min(axis=0)
    obj_max = F_max.max(axis=0)
    obj_range = obj_max - obj_min
    obj_range[obj_range == 0] = 1.0
    F_norm = (F_max - obj_min) / obj_range

    # Negate for minimisation
    F_neg = -F_norm

    # Non-dominated sorting — keep front 0
    pareto_idx = _non_dominated_front0(F_neg)

    if len(pareto_idx) == 1:
        return df_valid.iloc[pareto_idx[0]]["Method"]

    # ASF on Pareto front  (mirrors generate_final_tables.py exactly)
    pf = F_neg[pareto_idx]
    pf_min = pf.min(axis=0)
    pf_max = pf.max(axis=0)
    pf_range = pf_max - pf_min
    pf_range[pf_range == 0] = 1.0
    pf_norm = (pf - pf_min) / pf_range

    # ASF value = max_j( pf_norm_j / (1/W_j) ) = max_j( pf_norm_j * W_j )
    asf_values = np.max(pf_norm * ASF_WEIGHTS, axis=1)
    best_in_pf = asf_values.argmin()

    return df_valid.iloc[pareto_idx[best_in_pf]]["Method"]


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


def plot_grid(df, output_path, color_map, col_bests, datasets):
    """Create the full grid figure (datasets × EG metrics)."""
    n_rows = len(datasets)
    if n_rows == 0:
        print("  Skipping grid: no datasets with data")
        return

    fig, axes = plt.subplots(
        n_rows, len(EG_METRICS),
        figsize=(16, max(5.5, 4.4 * n_rows + 1.2)),
        constrained_layout=True,
        squeeze=False,
    )

    for i, dataset in enumerate(datasets):
        df_ds = df[df["Dataset"] == dataset]
        for j, eg in enumerate(EG_METRICS):
            ax = axes[i, j]
            is_bottom = (i == n_rows - 1)
            is_left = (j == 0)
            plot_single_panel(
                ax, df_ds, eg, color_map,
                best_per_col=col_bests.get(dataset, {}),
                show_ylabel=is_left, show_xlabel=is_bottom,
            )
            if i == 0:
                ax.set_title(eg, fontsize=14, fontweight="bold", pad=8)

        axes[i, 0].annotate(
            dataset, xy=(-0.35, 0.5), xycoords="axes fraction",
            fontsize=14, fontweight="bold", rotation=90,
            ha="center", va="center",
        )

    handles = _build_legend_handles(df["Method"].unique(), color_map)
    fig.legend(handles=handles, loc="center right",
               bbox_to_anchor=(1.18, 0.5), fontsize=12, ncol=1)

    fig.suptitle(
        "Precision@10 vs Equality Gain Trade-off  (table-matched highlights)",
        fontsize=18, fontweight="bold",
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_single_dataset(df, dataset, output_path, color_map, best_per_col):
    """Create a 1×3 figure for a single dataset."""
    df_ds = df[df["Dataset"] == dataset]

    fig, axes = plt.subplots(1, len(EG_METRICS), figsize=(18, 4.5),
                             constrained_layout=True)

    for j, eg in enumerate(EG_METRICS):
        ax = axes[j]
        plot_single_panel(
            ax, df_ds, eg, color_map,
            best_per_col=best_per_col,
            show_ylabel=True, show_xlabel=True,
        )
        ax.set_title(eg, fontsize=14, fontweight="bold")

    handles = _build_legend_handles(df_ds["Method"].unique(), color_map)
    fig.legend(handles=handles, loc="center right",
               bbox_to_anchor=(1.18, 0.5), fontsize=10, ncol=2, columnspacing=0.8)

    fig.suptitle(
        f"{dataset} — Precision@10 vs Equality Gain  (table-matched)",
        fontsize=17, fontweight="bold",
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot the Precision@10 vs Equality Gain Pareto figures")
    parser.add_argument("--csv", type=str, default=CSV_PATH,
                        help="table1_precision_equality_gain.csv to plot "
                             "(default: the copy in analysis/)")
    parser.add_argument("--out_dir", type=str, default=OUTPUT_DIR,
                        help="Directory for the output PDFs (default: analysis/plots)")
    parser.add_argument("--exclude", type=str, default="",
                        help="Comma-separated method names to drop, as spelled in "
                             "the CSV (e.g. 'WeakTie,WeakTie (no re-rank)'). "
                             "Excluded methods are removed before the Pareto "
                             "frontier and best-per-metric diamonds are computed.")
    parser.add_argument("--no_weak_tie", action="store_true",
                        help=f"Shorthand for --exclude {','.join(sorted(WEAK_TIE_METHODS))}")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix appended to the output filenames, e.g. "
                             "'_no_weaktie', so existing PDFs are not overwritten")
    cli = parser.parse_args()
    csv_path = os.path.abspath(cli.csv)
    out_dir = os.path.abspath(cli.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    excluded = {m.strip() for m in cli.exclude.split(",") if m.strip()}
    if cli.no_weak_tie:
        excluded |= WEAK_TIE_METHODS

    print("Loading data...")
    print(f"  csv    : {csv_path}")
    print(f"  out_dir: {out_dir}")
    print(f"  suffix : {cli.suffix or '(none)'}")
    df_all = load_and_wrangle(csv_path)
    print(f"  {len(df_all)} (Method, Dataset) entries loaded")

    # Colour map (same deterministic assignment as original).
    # Post-submission methods (FIXED_COLORS) are excluded from the tab20
    # enumeration so the colours of all pre-existing methods are unchanged.
    # Built from the *unfiltered* method list so that --exclude does not shift
    # the colours of the methods that stay in the figure.
    methods = df_all["Method"].unique()
    cmap = plt.get_cmap("tab20")
    color_map = {m: cmap(i % 20)
                 for i, m in enumerate(m for m in methods if m not in FIXED_COLORS)}
    for m, c in FIXED_COLORS.items():
        if m in methods:
            color_map[m] = c

    # ---- Apply --exclude / --no_weak_tie ----
    df = df_all
    if excluded:
        unknown = sorted(excluded - set(methods))
        if unknown:
            print(f"  WARNING: --exclude names not present in CSV: {unknown}")
        dropped = sorted(excluded & set(methods))
        df = df_all[~df_all["Method"].isin(excluded)].copy()
        print(f"  Excluded methods: {dropped or '(none matched)'}  "
              f"→ {len(df)} entries remain")

    # ---- Pre-compute highlights per dataset ----
    col_bests = {}   # dataset → {column_label: method name}

    available_datasets = [d for d in DATASETS if d in df["Dataset"].unique()]
    if not available_datasets:
        print("  No datasets with data in CSV — nothing to plot.")
        return

    print("\nComputing table-matched highlights per dataset...")
    for dataset in available_datasets:
        df_ds = df[df["Dataset"] == dataset]
        per_col = find_best_per_column(df_ds)
        col_bests[dataset] = per_col
        print(f"  {dataset:15s}  "
              f"Best Prec: {per_col.get('Precision', '-'):12s}  |  "
              f"Best Deg EG: {per_col.get('Degree EG@10', '-'):12s}  |  "
              f"Best PR EG: {per_col.get('PageRank EG@10', '-'):12s}  |  "
              f"Best Cl EG: {per_col.get('Closeness EG@10', '-'):12s}")

    # ---- Plot ----
    print("\nPlotting full grid (table-matched)...")
    plot_grid(df,
              os.path.join(out_dir,
                           f"full_comparison_matrix_table_match{cli.suffix}.pdf"),
              color_map, col_bests, available_datasets)

    print("\nPlotting per-dataset summaries (table-matched)...")
    for dataset in available_datasets:
        fname = f"{dataset}_summary_table_match{cli.suffix}.pdf"
        plot_single_dataset(
            df, dataset,
            os.path.join(out_dir, fname),
            color_map,
            best_per_col=col_bests[dataset],
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
