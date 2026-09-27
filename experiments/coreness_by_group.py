"""
Experiment (f): group-wise coreness distributions.

Converts the "demographic fairness as a byproduct" claim into a mechanism:
if a demographic group is over-represented in the periphery (low coreness),
then a coreness-targeting intervention mechanically transfers utility to
that group. This script quantifies the first link of that chain, and
(optionally) the second one using the per-node output of experiment (a).

For each dataset with sensitive attributes (oklahoma, unc: gender;
pokec_n: region):

  1. Coreness distribution per group: CDF + violin plot, medians/means.
  2. Association statistics: Mann-Whitney U, KS test, Cliff's delta,
     point-biserial correlation, per-group share in the bottom/top
     coreness quintile.
  3. If --stratified_csv_dir points at experiment (a)'s output, the
     per-node CAFE-vs-HL-GNN precision uplift is joined with group
     membership: mean uplift per group, and per (group x coreness tercile),
     directly testing "CAFE helps group X because group X is peripheral".

CPU-only; no training involved (coreness is computed on the train split).

Usage:
    python coreness_by_group.py --datasets oklahoma,unc,pokec_n \
        --stratified_csv_dir plots/stratified_cafe --save_dir plots/coreness_by_group
"""

import os
import sys
import argparse
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
for path in [parent_dir, os.path.join(parent_dir, 'backbone')]:
    if path not in sys.path:
        sys.path.insert(0, path)

from core import set_seed, load_dataset  # noqa: E402
from core.data_loader import load_sensitive_attr  # noqa: E402
from metrics.coreness import compute_coreness  # noqa: E402

GROUP_LABELS = {
    'oklahoma': {0: 'group 0 (gender)', 1: 'group 1 (gender)'},
    'unc': {0: 'group 0 (gender)', 1: 'group 1 (gender)'},
    'pokec_n': {0: 'group 0 (region)', 1: 'group 1 (region)'},
}


def cliffs_delta(x, y):
    """Cliff's delta via the Mann-Whitney U statistic (efficient)."""
    nx, ny = len(x), len(y)
    u, _ = stats.mannwhitneyu(x, y, alternative='two-sided')
    return 2.0 * u / (nx * ny) - 1.0


def analyze_dataset(data_name, args):
    print(f"\n{'=' * 80}\nDATASET: {data_name}\n{'=' * 80}")

    load_args = SimpleNamespace(
        data_name=data_name, data_path=args.data_path,
        val_ratio=0.05, test_ratio=0.1,
        encoder='HLGNN', device=0, split_file=None,
        data_split_seed=args.split_seed,
    )
    dataset, split_edge, data, degrees = load_dataset(load_args)
    num_nodes = dataset._data.num_nodes if hasattr(dataset, '_data') else dataset.data.num_nodes
    train_edges = split_edge['train']['edge']

    coreness = compute_coreness(train_edges, num_nodes).cpu().numpy()

    try:
        sens = load_sensitive_attr(data_name, num_nodes, root=args.data_path)
    except ValueError as e:
        print(f"SKIP {data_name}: {e}")
        return None, None
    sens = np.asarray(sens.cpu() if torch.is_tensor(sens) else sens).astype(int)

    valid = sens >= 0  # guard against unknown/-1 encodings
    c0 = coreness[valid & (sens == 0)]
    c1 = coreness[valid & (sens == 1)]
    labels = GROUP_LABELS.get(data_name, {0: 'group 0', 1: 'group 1'})
    print(f"n0={len(c0)}, n1={len(c1)}")

    # --- Association statistics ---
    u_stat, p_mw = stats.mannwhitneyu(c0, c1, alternative='two-sided')
    ks_stat, p_ks = stats.ks_2samp(c0, c1)
    delta = cliffs_delta(c0, c1)
    r_pb, p_pb = stats.pointbiserialr(sens[valid], coreness[valid])

    q20, q80 = np.percentile(coreness[valid], [20, 80])
    stats_row = {
        'dataset': data_name,
        'n_group0': len(c0), 'n_group1': len(c1),
        'coreness_mean_g0': c0.mean(), 'coreness_mean_g1': c1.mean(),
        'coreness_median_g0': float(np.median(c0)),
        'coreness_median_g1': float(np.median(c1)),
        'mannwhitney_p': p_mw, 'ks_p': p_ks,
        'cliffs_delta_g0_vs_g1': delta,
        'pointbiserial_r': r_pb, 'pointbiserial_p': p_pb,
        'share_bottom_quintile_g0': float((c0 <= q20).mean()),
        'share_bottom_quintile_g1': float((c1 <= q20).mean()),
        'share_top_quintile_g0': float((c0 >= q80).mean()),
        'share_top_quintile_g1': float((c1 >= q80).mean()),
    }
    print(pd.Series(stats_row).to_string())

    # --- Plot: CDF + violin ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    for cvals, lbl, color in [(c0, labels[0], '#1f77b4'), (c1, labels[1], '#d62728')]:
        xs = np.sort(cvals)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax1.step(xs, ys, where='post', label=lbl, color=color, lw=1.5)
    ax1.set_xlabel('Coreness (k-core number)')
    ax1.set_ylabel('CDF')
    ax1.legend(fontsize=8, frameon=False)
    ax1.grid(alpha=0.3, lw=0.5)

    parts = ax2.violinplot([c0, c1], showmedians=True)
    for pc, color in zip(parts['bodies'], ['#1f77b4', '#d62728']):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels([labels[0], labels[1]], fontsize=8)
    ax2.set_ylabel('Coreness')
    ax2.grid(axis='y', alpha=0.3, lw=0.5)

    fig.suptitle(f'Coreness distribution by demographic group — {data_name}\n'
                 f"Cliff's $\\delta$={delta:.3f}, Mann-Whitney p={p_mw:.2e}",
                 fontsize=10)
    fig.tight_layout()
    out = os.path.join(args.save_dir, f'coreness_by_group_{data_name}.pdf')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")

    # --- Optional: join with experiment (a) per-node uplift ---
    uplift_rows = None
    strat_csv = os.path.join(args.stratified_csv_dir,
                             f'stratified_cafe_{data_name}.csv') \
        if args.stratified_csv_dir else None
    if strat_csv and os.path.exists(strat_csv):
        print(f"\nJoining per-node uplift from {strat_csv}")
        sdf = pd.read_csv(strat_csv)
        # mean per-node precision across runs, per model
        per_node = (sdf.groupby(['node_id', 'model'])['precision']
                    .mean().unstack('model'))
        uplift_rows = []
        for cafe_variant in ['CAFE-In', 'CAFE-Pre', 'Fusion']:
            if cafe_variant not in per_node.columns or 'HL-GNN' not in per_node.columns:
                continue
            uplift = (per_node[cafe_variant] - per_node['HL-GNN']).dropna()
            node_ids = uplift.index.astype(int).values
            in_range = node_ids < num_nodes
            node_ids, uvals = node_ids[in_range], uplift.values[in_range]
            g = sens[node_ids]
            c = coreness[node_ids]
            terciles = np.percentile(c, [33.3, 66.7])
            tier = np.digitize(c, terciles)  # 0=periphery, 1=mid, 2=core
            for grp in [0, 1]:
                for t, tname in enumerate(['periphery', 'mid', 'core']):
                    mask = (g == grp) & (tier == t)
                    if mask.sum() > 0:
                        uplift_rows.append({
                            'dataset': data_name,
                            'cafe_variant': cafe_variant,
                            'group': grp,
                            'coreness_tier': tname,
                            'n_nodes': int(mask.sum()),
                            'mean_uplift': float(uvals[mask].mean()),
                        })
                mask_g = g == grp
                uplift_rows.append({
                    'dataset': data_name,
                    'cafe_variant': cafe_variant,
                    'group': grp,
                    'coreness_tier': 'ALL',
                    'n_nodes': int(mask_g.sum()),
                    'mean_uplift': float(uvals[mask_g].mean()),
                })
        udf = pd.DataFrame(uplift_rows)
        print(udf.to_string(index=False))
    elif strat_csv:
        print(f"\n(No per-node CSV at {strat_csv} — run experiment (a) first "
              f"to get the uplift-by-group analysis.)")

    return stats_row, uplift_rows


def main():
    parser = argparse.ArgumentParser(description='Group-wise coreness distributions (experiment f)')
    parser.add_argument('--datasets', type=str, default='oklahoma,unc,pokec_n')
    parser.add_argument('--data_path', type=str, default='../data/SNAP')
    parser.add_argument('--split_seed', type=int, default=42,
                        help='data split seed (42 = same split as main experiments)')
    parser.add_argument('--stratified_csv_dir', type=str,
                        default='plots/stratified_cafe',
                        help="experiment (a)'s save_dir, for the uplift join")
    parser.add_argument('--save_dir', type=str, default='plots/coreness_by_group')
    args = parser.parse_args()

    set_seed(args.split_seed)
    os.makedirs(args.save_dir, exist_ok=True)

    stats_rows, uplift_all = [], []
    for ds in [d.strip() for d in args.datasets.split(',')]:
        srow, urows = analyze_dataset(ds, args)
        if srow:
            stats_rows.append(srow)
        if urows:
            uplift_all.extend(urows)

    if stats_rows:
        out = os.path.join(args.save_dir, 'coreness_group_association.csv')
        pd.DataFrame(stats_rows).to_csv(out, index=False)
        print(f"\nAssociation table saved to {out}")

    if uplift_all:
        out = os.path.join(args.save_dir, 'uplift_by_group_and_tier.csv')
        pd.DataFrame(uplift_all).to_csv(out, index=False)
        print(f"Uplift table saved to {out}")

    print("\nGroup-wise coreness analysis complete!")


if __name__ == '__main__':
    main()
