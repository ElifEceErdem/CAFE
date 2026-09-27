"""Plotting helpers for the coreness-stratified diagnosis (Figures 1 and 4).

Split out of ``run_precision_by_coreness.py`` so the figures can be rebuilt
from saved per-run results without importing the training stack.  Used by
``run_precision_by_coreness.py``, ``run_stratified_cafe.py`` and
``tables/f01_diagnosis.py`` / ``tables/f04_stratified.py``.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_COLORS = {
    'CN': '#1abc9c',
    'RA': '#9b59b6',
    'AA': '#3498db',
    'LightGCN': '#e74c3c',
    'HL-GNN': '#e67e22',
    'Hybrid': '#2ecc71',
}

MODEL_MARKERS = {
    'CN': 'o',
    'RA': 's',
    'AA': '^',
    'LightGCN': 'D',
    'HL-GNN': 'p',
    'Hybrid': 'X',
}


def assign_coreness_bins(df, num_bins):
    """Assign quantile-based coreness bins."""
    try:
        df['coreness_bin'], bin_edges = pd.qcut(
            df['node_coreness'], q=num_bins, retbins=True,
            labels=False, duplicates='drop'
        )
    except ValueError:
        df['coreness_bin'], bin_edges = pd.cut(
            df['node_coreness'], bins=num_bins, retbins=True, labels=False
        )

    # Human-readable labels
    actual_bins = sorted(df['coreness_bin'].unique())
    bin_labels = {}
    for b in actual_bins:
        mask = df['coreness_bin'] == b
        lo = df.loc[mask, 'node_coreness'].min()
        hi = df.loc[mask, 'node_coreness'].max()
        bin_labels[b] = f"[{lo:.0f}-{hi:.0f}]"

    df['coreness_bin_label'] = df['coreness_bin'].map(bin_labels)
    return df, bin_labels


def assign_quantile_bins(series, num_bins, label_prefix=''):
    """
    Assign quantile-based bins to a pandas Series.
    Returns (bin_series, bin_label_map) where bin_label_map maps bin_index -> label string.
    Falls back to equal-width bins if quantile fails due to duplicates.
    """
    try:
        bin_series, bin_edges = pd.qcut(
            series, q=num_bins, retbins=True, labels=False, duplicates='drop'
        )
    except ValueError:
        bin_series, bin_edges = pd.cut(
            series, bins=num_bins, retbins=True, labels=False
        )

    actual_bins = sorted(bin_series.dropna().unique())
    bin_label_map = {}
    for b in actual_bins:
        mask = bin_series == b
        lo = series[mask].min()
        hi = series[mask].max()
        bin_label_map[b] = f"{label_prefix}[{lo:.0f}–{hi:.0f}]"

    return bin_series, bin_label_map


def plot_2d_heatmap(df, model_name, dataset_name, k,
                    num_degree_bins, num_coreness_bins, save_dir):
    """
    2D heatmap of mean Precision@K across (degree_bin, coreness_bin).

    If coreness matters independently, precision should increase left→right
    within every row of the heatmap.
    """
    df = df.copy()

    df['degree_bin'], deg_label_map = assign_quantile_bins(
        df['degree'], num_degree_bins, label_prefix='deg '
    )
    df['coreness_bin'], cor_label_map = assign_quantile_bins(
        df['coreness'], num_coreness_bins
    )

    degree_groups = sorted(df['degree_bin'].dropna().unique())
    coreness_groups = sorted(df['coreness_bin'].dropna().unique())

    # Build matrix: rows = degree (ascending), cols = coreness (ascending)
    matrix = np.full((len(degree_groups), len(coreness_groups)), np.nan)
    cell_counts = np.zeros_like(matrix, dtype=int)

    for i, dg in enumerate(degree_groups):
        for j, cg in enumerate(coreness_groups):
            cell = df[(df['degree_bin'] == dg) & (df['coreness_bin'] == cg)]
            if len(cell) >= 3:
                matrix[i, j] = cell['precision'].mean()
                cell_counts[i, j] = len(cell)

    fig, ax = plt.subplots(figsize=(max(6, len(coreness_groups) * 1.1),
                                     max(4, len(degree_groups) * 0.9)))
    ax.grid(False)
    ax.set_axisbelow(False)
    ax.tick_params(which='both', length=0)

    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn',
                   vmin=np.nanmin(matrix) if not np.all(np.isnan(matrix)) else 0,
                   vmax=np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 1)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f'Mean Precision@{k}', fontsize=11)

    # Annotate cells
    for i in range(len(degree_groups)):
        for j in range(len(coreness_groups)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f'{matrix[i, j]:.3f}\n(n={cell_counts[i,j]})',
                        ha='center', va='center', fontsize=7, color='black')
            else:
                ax.text(j, i, 'n/a', ha='center', va='center',
                        fontsize=7, color='gray')

    ax.set_xticks(range(len(coreness_groups)))
    ax.set_xticklabels(
        [cor_label_map.get(cg, str(cg)) for cg in coreness_groups],
        rotation=30, ha='right', fontsize=9
    )
    ax.set_yticks(range(len(degree_groups)))
    ax.set_yticklabels(
        [deg_label_map.get(dg, str(dg)) for dg in degree_groups],
        fontsize=9
    )

    ax.set_xlabel('Coreness (k-core number) →', fontsize=11)
    ax.set_ylabel('← Degree group', fontsize=11)
    ax.set_title(
        f'Precision@{k}: Degree × Coreness Heatmap\n'
        f'{model_name} — {dataset_name}\n'
        f'(If hypothesis holds: precision increases left→right within each row)',
        fontsize=11, fontweight='bold'
    )

    plt.tight_layout()
    fname = f'heatmap_{model_name.replace(" ", "_")}_{dataset_name}_k{k}.pdf'
    path = os.path.join(save_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path




def aggregate_precision_within_runs(prec_df):
    """Aggregate node precision to one value per run/model/coreness bin.

    Run-level replication is the experimental unit.  Computing whiskers from
    the raw node rows would treat thousands of users as independent reruns
    (pseudo-replication) and dramatically understate uncertainty.
    """
    required = {
        'model', 'coreness_bin', 'coreness_bin_label', 'precision',
    }
    missing = required - set(prec_df.columns)
    if missing:
        raise ValueError(
            f"Precision dataframe lacks columns: {sorted(missing)}"
        )

    working = prec_df.copy()
    if 'run' not in working.columns:
        working['run'] = 0

    return (
        working
        .groupby(
            ['run', 'model', 'coreness_bin', 'coreness_bin_label'],
            as_index=False,
        )
        .agg(
            precision=('precision', 'mean'),
            n_nodes=('node_id', 'nunique')
            if 'node_id' in working.columns
            else ('precision', 'count'),
        )
    )


def _ordered_models(values, requested_order=None):
    available = list(dict.fromkeys(values))
    if requested_order is None:
        return available
    ordered = [name for name in requested_order if name in available]
    ordered.extend(name for name in available if name not in ordered)
    return ordered


def plot_stacked_barchart(
    prec_df,
    eg_df,
    save_dir,
    dataset_name,
    K,
    model_order=None,
    display_name=None,
    filename_tag=None,
    figure_size=(6, 6),
    font_size=10,
    legend_inside=False,
):
    """
    Compact vertically-stacked grouped bar chart for paper figures.

    Two rows, one column, shared x-axis:
      Top:    Precision@K  (grouped bars + SEM error bars)
      Bottom: Equality Gain (%) (grouped bars + SEM error bars if `eg_sem`
              column is present in `eg_df`)

    X-tick labels only on the bottom panel.
    Output: stacked_barchart_{dataset_name}.pdf
    """
    os.makedirs(save_dir, exist_ok=True)

    models_prec = _ordered_models(prec_df['model'].unique(), model_order)
    models_eg = _ordered_models(eg_df['model'].unique(), model_order)
    bin_order = sorted(prec_df['coreness_bin'].unique())

    # Build x-tick labels from precision df
    label_map = (prec_df[['coreness_bin', 'coreness_bin_label']]
                 .drop_duplicates()
                 .set_index('coreness_bin'))
    xlabels = [label_map.loc[b, 'coreness_bin_label']
               if b in label_map.index else str(b)
               for b in bin_order]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figure_size, sharex=True,
        gridspec_kw={'hspace': 0.08}
    )

    x = np.arange(len(bin_order))

    # ------------------------------------------------------------------
    # Top panel: Precision@K grouped bars
    # ------------------------------------------------------------------
    num_models_prec = len(models_prec)
    width_prec = 0.7 / max(num_models_prec, 1)
    precision_by_run = aggregate_precision_within_runs(prec_df)

    for i, model_name in enumerate(models_prec):
        model_df = precision_by_run[
            precision_by_run['model'] == model_name
        ]
        grouped = model_df.groupby('coreness_bin')['precision'].agg(
            ['mean', 'std', 'count']
        )

        means = [grouped.loc[b, 'mean'] if b in grouped.index else 0
                 for b in bin_order]
        stds = [grouped.loc[b, 'std'] if b in grouped.index else 0
                for b in bin_order]
        counts = [grouped.loc[b, 'count'] if b in grouped.index else 0
                  for b in bin_order]
        sems = [s / np.sqrt(c) if c > 1 else 0
                for s, c in zip(stds, counts)]

        offset = (i - (num_models_prec - 1) / 2) * width_prec
        color = MODEL_COLORS.get(model_name, f'C{i}')
        ax_top.bar(x + offset, means, width_prec, yerr=sems,
                   label=model_name, color=color, alpha=0.85,
                   capsize=2, edgecolor='white', linewidth=0.5)

    tick_font_size = max(font_size - 2, 8)
    legend_font_size = max(font_size - 3, 8)

    ax_top.set_ylabel(f'Precision@{K}', fontsize=font_size)
    ax_top.tick_params(axis='y', labelsize=tick_font_size)
    ax_top.grid(axis='y', alpha=0.3, linewidth=0.5)

    # ------------------------------------------------------------------
    # Bottom panel: Equality Gain grouped bars
    # ------------------------------------------------------------------
    num_models_eg = len(models_eg)
    width_eg = 0.7 / max(num_models_eg, 1)
    eg_bins_available = set(eg_df['coreness_bin'].unique())
    has_eg_sem = 'eg_sem' in eg_df.columns

    for i, model_name in enumerate(models_eg):
        model_df = eg_df[eg_df['model'] == model_name]

        egs = []
        sems = []
        for b in bin_order:
            if b in eg_bins_available:
                match = model_df.loc[model_df['coreness_bin'] == b]
                if len(match) > 0:
                    egs.append(float(match['equality_gain'].values[0]))
                    if has_eg_sem:
                        sem_val = match['eg_sem'].values[0]
                        sems.append(float(sem_val) if not pd.isna(sem_val) else 0.0)
                    else:
                        sems.append(0.0)
                else:
                    egs.append(0.0)
                    sems.append(0.0)
            else:
                egs.append(0.0)
                sems.append(0.0)

        offset = (i - (num_models_eg - 1) / 2) * width_eg
        color = MODEL_COLORS.get(model_name, f'C{i}')
        ax_bot.bar(x + offset, egs, width_eg,
                   yerr=sems if has_eg_sem else None,
                   label=model_name,
                   color=color, alpha=0.85, capsize=2,
                   edgecolor='white', linewidth=0.5)

    ax_bot.axhline(y=0, color='gray', linestyle='--', alpha=0.5,
                   linewidth=0.8)
    ax_bot.set_ylabel('Equality Gain (%)', fontsize=font_size)
    ax_bot.set_xlabel('Node Coreness Group', fontsize=font_size)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(xlabels, rotation=45, ha='right',
                           fontsize=tick_font_size)
    ax_bot.tick_params(axis='y', labelsize=tick_font_size)
    ax_bot.grid(axis='y', alpha=0.3, linewidth=0.5)

    # ------------------------------------------------------------------
    # Shared legend at the top
    # ------------------------------------------------------------------
    handles, labels = ax_top.get_legend_handles_labels()
    bot_handles, bot_labels = ax_bot.get_legend_handles_labels()
    for h, l in zip(bot_handles, bot_labels):
        if l not in labels:
            handles.append(h)
            labels.append(l)

    if legend_inside:
        ax_top.legend(
            handles,
            labels,
            loc='upper left',
            ncol=min(3, len(labels)),
            fontsize=legend_font_size,
            frameon=True,
            framealpha=0.92,
            edgecolor='0.8',
            handletextpad=0.3,
            columnspacing=0.8,
        )
        title_y = 0.99
    else:
        fig.legend(handles, labels, loc='upper center',
                   ncol=len(labels), fontsize=legend_font_size, frameon=False,
                   bbox_to_anchor=(0.5, 1.02),
                   handletextpad=0.3, columnspacing=1.0)
        title_y = 1.06

    title = display_name if display_name is not None else dataset_name
    fig.suptitle(f'{title} (K={K})', fontsize=font_size + 1,
                 fontweight='bold', y=title_y)

    output_tag = filename_tag if filename_tag is not None else dataset_name
    path = os.path.join(save_dir,
                        f'stacked_barchart_{output_tag}.pdf')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")





# ========================================================================
# Main
# ========================================================================
