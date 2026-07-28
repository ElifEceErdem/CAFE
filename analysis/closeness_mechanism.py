"""
Experiment (g): why does RA fusion regress Closeness EG on Oklahoma?

Mechanistic analysis promoted from the rebuttal argument. The structural
hypothesis is testable: RA(u,v) > 0 only when u and v share a neighbor,
i.e. RA can only recommend DISTANCE-2 pairs. Fusing RA therefore pulls the
recommendation mass toward short-range, locally dense pairs and suppresses
the long-range shortcuts that flatten the closeness distribution. This
script measures every link of that chain, before vs. after RA fusion, on
the same trained backbone (so the ONLY difference is the beta term):

  Methods: HL-GNN (beta=0) | +RA (beta=beta*) | +RA+boost | pure RA

  1. Distance spectrum: train-graph shortest-path distance of each
     recommended pair (d=2, 3, 4, >=5, unreachable). The hypothesis
     predicts a collapse onto d=2 after fusion.
  2. Predicted-graph path statistics: mean shortest path length and
     reachable-pair fraction of (train + recommended edges).
  3. Closeness EG replication + per-coreness-bin closeness change:
     who wins/loses closeness when RA is fused.
  4. Target profile: mean degree / coreness of recommended targets.

Exact APSP is used (feasible for Oklahoma/UNC-scale graphs; do not run
this script on brightkite).

Usage:
    python closeness_mechanism.py --data_name oklahoma --beta 5.0 --lam 0.5 --runs 3

With --vary_split true, run r uses split seed r and training seed r
(mirroring experiments/run_significance.py), so the runs are paired-protocol
replications rather than same-split seed-42 reruns. --lam_grid adds extra
inference-time boost arms at other lambda_mob values (no extra training).
"""

import os
import sys
import argparse
import gc

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
experiments_dir = os.path.join(parent_dir, 'experiments')
for path in [parent_dir, current_dir, experiments_dir,
             os.path.join(parent_dir, 'backbone')]:
    if path not in sys.path:
        sys.path.insert(0, path)

from core import set_seed, load_dataset  # noqa: E402
from metrics.coreness import compute_coreness  # noqa: E402
from metrics.centrality_equality_gain import (  # noqa: E402
    build_control_centrality, build_predicted_centrality,
    compute_gini_coefficient,
)
from cafe_methods import (  # noqa: E402
    add_common_args, build_train_adj_csr, build_ra_sparse,
    mobility_multiplier_vector, train_cafe_model,
    build_cafe_recs, build_heuristic_recs, get_eval_nodes,
)

DIST_BINS = ['d=2', 'd=3', 'd=4', 'd>=5', 'unreachable']


def bin_distance(d):
    if np.isinf(d):
        return 'unreachable'
    if d <= 2:
        return 'd=2'
    if d == 3:
        return 'd=3'
    if d == 4:
        return 'd=4'
    return 'd>=5'


def distance_spectrum(recs_by_u, D):
    """Fraction of recommended pairs per train-graph distance bin."""
    counts = {b: 0 for b in DIST_BINS}
    total = 0
    for u, vs in recs_by_u.items():
        for v in vs:
            counts[bin_distance(D[u, v])] += 1
            total += 1
    return {b: counts[b] / total for b in DIST_BINS}, total


def predicted_graph_stats(train_adj_csr, recs_by_u, num_nodes):
    """Mean shortest path + reachable fraction of train + rec edges."""
    rows, cols = [], []
    for u, vs in recs_by_u.items():
        for v in vs:
            rows.append(u)
            cols.append(v)
    rec_adj = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(num_nodes, num_nodes))
    pred_adj = train_adj_csr + rec_adj + rec_adj.T
    pred_adj.data = np.ones_like(pred_adj.data)
    # Unique undirected edge budget after reciprocal-recommendation dedup:
    # symmetric scorers collapse more reciprocal pairs than asymmetric ones,
    # so equal K does not guarantee equal added-edge counts across arms.
    n_unique_new = int((pred_adj.nnz - train_adj_csr.nnz) // 2)

    D = shortest_path(pred_adj, method='D', unweighted=True, directed=False)
    finite = np.isfinite(D) & (D > 0)
    return {
        'mean_path_length': float(D[finite].mean()),
        'reachable_pair_fraction': float(finite.sum() / (num_nodes * (num_nodes - 1))),
        'diameter': float(D[finite].max()),
        'n_unique_new_edges': n_unique_new,
        'n_directed_recs': len(rows),
    }


def closeness_by_bin(train_edges_cpu, test_edges_cpu, recs_by_u, num_nodes,
                     K, coreness_np, num_bins=5):
    """Closeness EG + mean predicted closeness per coreness quintile."""
    cent_ctrl = build_control_centrality(
        train_edges_cpu, test_edges_cpu, num_nodes, 'closeness',
        use_sampling=False)
    cent_pred = build_predicted_centrality(
        train_edges_cpu, recs_by_u, num_nodes, K, 'closeness',
        use_sampling=False)
    g_ctrl = compute_gini_coefficient(cent_ctrl)
    g_pred = compute_gini_coefficient(cent_pred)
    eg = ((g_ctrl - g_pred) / g_ctrl * 100.0) if g_ctrl > 0 else 0.0

    pred_np = np.asarray(cent_pred)
    ctrl_np = np.asarray(cent_ctrl)
    # float64 edges and a CLOSED final interval: the previous float32
    # `edges[-1] += 1e-9` was a no-op at coreness scale, silently dropping
    # every maximum-coreness node from the top bin.
    edges = np.percentile(np.asarray(coreness_np, dtype=np.float64),
                          np.linspace(0, 100, num_bins + 1))
    per_bin = []
    for i in range(num_bins):
        last = (i == num_bins - 1)
        upper = (coreness_np <= edges[i + 1]) if last \
            else (coreness_np < edges[i + 1])
        mask = (coreness_np >= edges[i]) & upper
        if mask.sum() > 0:
            per_bin.append({
                'bin': i,
                'coreness_range': f'[{edges[i]:.0f},{edges[i + 1]:.0f}'
                                  + (']' if last else ')'),
                'n_nodes': int(mask.sum()),
                'mean_closeness_pred': float(pred_np[mask].mean()),
                'mean_closeness_ctrl': float(ctrl_np[mask].mean()),
            })
    return eg, g_ctrl, g_pred, per_bin


def target_profile(recs_by_u, degrees_np, coreness_np):
    tgt_deg, tgt_cor = [], []
    for u, vs in recs_by_u.items():
        for v in vs:
            tgt_deg.append(degrees_np[v])
            tgt_cor.append(coreness_np[v])
    return {
        'mean_target_degree': float(np.mean(tgt_deg)),
        'mean_target_coreness': float(np.mean(tgt_cor)),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Closeness EG mechanism analysis (experiment g)')
    parser = add_common_args(parser)
    parser.add_argument('--beta', type=float, required=True,
                        help='RA fusion weight (best config for this dataset)')
    parser.add_argument('--lam', type=float, required=True,
                        help='mobility boost strength (best config)')
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--vary_split', type=lambda x: x.lower() == 'true',
                        default=False,
                        help='run r uses split seed r and training seed r, '
                             'matching the paired protocol of '
                             'run_significance.py (default: legacy same-split '
                             'seed-42 reruns)')
    parser.add_argument('--lam_grid', type=str, default='',
                        help='comma-separated extra lambda_mob values; each '
                             'adds an inference-time +boost arm on the same '
                             'trained backbone')
    parser.add_argument('--replot', type=lambda x: x.lower() == 'true',
                        default=False,
                        help='skip all runs and rebuild the figure from the '
                             'CSVs already in save_dir')
    parser.add_argument('--save_dir', type=str, default='plots/closeness_mechanism')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    print('=' * 80)
    print(f'CLOSENESS MECHANISM ANALYSIS — {args.data_name} '
          f'(beta={args.beta}, lam={args.lam}, K={args.K}, runs={args.runs})')
    print('=' * 80)

    spectrum_rows, stats_rows, bin_rows = [], [], []

    lam_extra = [float(x) for x in args.lam_grid.split(',') if x.strip()]

    for run in ([] if args.replot else range(args.runs)):
        split_seed = run if args.vary_split else 42
        print(f"\n{'=' * 80}\nRUN {run + 1}/{args.runs} "
              f"(split_seed={split_seed}, vary_split={args.vary_split})"
              f"\n{'=' * 80}")

        if args.vary_split:
            # Mirror run_significance.py: seed the RNG and resample the
            # split with this run's seed.
            set_seed(run)
            args.data_split_seed = run
        dataset, split_edge, data, degrees = load_dataset(args)
        num_nodes = dataset._data.num_nodes if hasattr(dataset, '_data') else dataset.data.num_nodes
        if num_nodes > 30000:
            raise SystemExit(f'{args.data_name} has {num_nodes} nodes; exact '
                             f'APSP is infeasible — this analysis targets '
                             f'Oklahoma/UNC-scale graphs.')
        train_edges = split_edge['train']['edge'].to(device)
        pos_test_edge = split_edge['test']['edge'].to(device)
        train_edges_cpu = train_edges.cpu()
        test_edges_cpu = pos_test_edge.cpu()

        coreness = compute_coreness(train_edges, num_nodes).to(device)
        data.coreness = coreness
        c_np = coreness.cpu().numpy()
        degrees_np = degrees.cpu().numpy()

        adj_sp = build_train_adj_csr(train_edges, num_nodes)
        ra_sparse = build_ra_sparse(adj_sp)
        eval_nodes = get_eval_nodes(pos_test_edge)
        print(f'{len(eval_nodes)} eval nodes')

        # Exact APSP on the TRAIN graph (for the distance spectrum)
        print('Computing train-graph APSP...')
        D_train = shortest_path(adj_sp, method='D', unweighted=True,
                                directed=False)

        # One backbone; fusion/boost are inference-time deltas
        print('\nTraining vanilla HL-GNN backbone...')
        if args.vary_split:
            set_seed(run)  # identical RNG state before training, as in run_significance
        vanilla = train_cafe_model(data, split_edge, args, degrees, coreness,
                                   device, mode='none')
        mult = mobility_multiplier_vector(c_np, args.lam, args.coreness_alpha)

        method_recs = {
            'HL-GNN': build_cafe_recs(
                vanilla['h_emb'], vanilla['score_fn'], eval_nodes, num_nodes,
                device, adj_sp, beta=0.0, k=args.K, verbose=False),
            '+RA': build_cafe_recs(
                vanilla['h_emb'], vanilla['score_fn'], eval_nodes, num_nodes,
                device, adj_sp, ra_sparse=ra_sparse, beta=args.beta,
                k=args.K, verbose=False),
            '+RA+boost': build_cafe_recs(
                vanilla['h_emb'], vanilla['score_fn'], eval_nodes, num_nodes,
                device, adj_sp, ra_sparse=ra_sparse, beta=args.beta,
                mult_vec=mult, k=args.K, verbose=False),
            'RA only': build_heuristic_recs(
                ra_sparse, adj_sp, eval_nodes, args.K, verbose=False),
        }
        for lam in lam_extra:
            if lam == args.lam:
                continue
            mult_l = mobility_multiplier_vector(c_np, lam, args.coreness_alpha)
            method_recs[f'+boost(lam={lam:g})'] = build_cafe_recs(
                vanilla['h_emb'], vanilla['score_fn'], eval_nodes, num_nodes,
                device, adj_sp, ra_sparse=ra_sparse, beta=args.beta,
                mult_vec=mult_l, k=args.K, verbose=False)
        del vanilla
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        for method, recs in method_recs.items():
            print(f'\n--- {method} ---')
            spec, n_pairs = distance_spectrum(recs, D_train)
            print('Distance spectrum of recommended pairs: '
                  + ', '.join(f'{b}: {v:.3f}' for b, v in spec.items()))
            spectrum_rows.append({'run': run, 'split_seed': split_seed,
                                  'method': method, 'n_pairs': n_pairs, **spec})

            pstats = predicted_graph_stats(adj_sp, recs, num_nodes)
            eg, g_ctrl, g_pred, per_bin = closeness_by_bin(
                train_edges_cpu, test_edges_cpu, recs, num_nodes,
                args.K, c_np)
            tprof = target_profile(recs, degrees_np, c_np)
            print(f'Closeness EG@{args.K} = {eg:.2f}% '
                  f'(Gini ctrl {g_ctrl:.4f} -> pred {g_pred:.4f}) | '
                  f'mean path {pstats["mean_path_length"]:.3f} | '
                  f'target degree {tprof["mean_target_degree"]:.1f}')
            stats_rows.append({'run': run, 'split_seed': split_seed,
                               'method': method, 'closeness_eg': eg,
                               'gini_ctrl': g_ctrl, 'gini_pred': g_pred,
                               **pstats, **tprof})
            for pb in per_bin:
                bin_rows.append({'run': run, 'split_seed': split_seed,
                                 'method': method, **pb})

        del D_train
        gc.collect()

    # ------------------------------------------------------------------
    # Save + aggregate
    # ------------------------------------------------------------------
    if args.replot:
        spec_df = pd.read_csv(os.path.join(
            args.save_dir, f'distance_spectrum_{args.data_name}.csv'))
        stats_df = pd.read_csv(os.path.join(
            args.save_dir, f'path_stats_{args.data_name}.csv'))
    else:
        spec_df = pd.DataFrame(spectrum_rows)
        stats_df = pd.DataFrame(stats_rows)
        bins_df = pd.DataFrame(bin_rows)
        spec_df.to_csv(os.path.join(args.save_dir,
                                    f'distance_spectrum_{args.data_name}.csv'), index=False)
        stats_df.to_csv(os.path.join(args.save_dir,
                                     f'path_stats_{args.data_name}.csv'), index=False)
        bins_df.to_csv(os.path.join(args.save_dir,
                                    f'closeness_by_bin_{args.data_name}.csv'), index=False)

    spec_mean = spec_df.groupby('method')[DIST_BINS].mean()
    stats_mean = stats_df.groupby('method').mean(numeric_only=True)

    print('\n' + '=' * 80)
    print('AGGREGATED RESULTS (mean over runs)')
    print('=' * 80)
    print('\nDistance spectrum of recommended pairs:')
    print(spec_mean.round(4).to_string())
    print('\nPath / closeness statistics:')
    cols = ['closeness_eg', 'gini_pred', 'mean_path_length',
            'reachable_pair_fraction', 'mean_target_degree',
            'mean_target_coreness']
    print(stats_mean[cols].round(4).to_string())

    # ------------------------------------------------------------------
    # Figure: stacked distance spectrum + closeness EG per method
    # ------------------------------------------------------------------
    def method_order(m):
        """Backbone, +RA, boost arms by increasing lambda_mob, RA only."""
        if m == 'HL-GNN':
            return (0, 0.0)
        if m == '+RA':
            return (1, 0.0)
        if m == '+RA+boost':
            return (2, args.lam)
        if m.startswith('+boost(lam='):
            return (2, float(m[len('+boost(lam='):-1]))
        if m == 'RA only':
            return (3, 0.0)
        return (4, 0.0)

    methods = sorted(spec_mean.index, key=method_order)
    spec_mean = spec_mean.loc[methods]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))

    bottom = np.zeros(len(methods))
    colors = ['#08519c', '#3182bd', '#6baed6', '#bdd7e7', '#bbbbbb']
    for b, color in zip(DIST_BINS, colors):
        vals = spec_mean[b].values
        ax1.bar(methods, vals, bottom=bottom, label=b, color=color,
                edgecolor='white', linewidth=0.5)
        bottom += vals
    ax1.set_ylabel('Fraction of recommended pairs')
    ax1.set_title('Train-graph distance of recommended pairs', fontsize=10)
    ax1.legend(fontsize=7, frameon=False, ncol=2)
    ax1.tick_params(axis='x', rotation=20, labelsize=8)

    eg_mean = stats_df.groupby('method')['closeness_eg'].mean()
    eg_sem = stats_df.groupby('method')['closeness_eg'].sem()
    ax2.bar(methods, [eg_mean[m] for m in methods],
            yerr=[eg_sem[m] if not np.isnan(eg_sem[m]) else 0 for m in methods],
            color='#e6550d', alpha=0.85, capsize=3,
            edgecolor='white', linewidth=0.5)
    ax2.axhline(0, color='gray', ls='--', lw=0.8)
    ax2.set_ylabel(f'Closeness EG@{args.K} (%)')
    ax2.set_title('Closeness Equality Gain', fontsize=10)
    ax2.tick_params(axis='x', rotation=20, labelsize=8)

    fig.suptitle(f'{args.data_name}: RA fusion and the closeness trade-off',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    out = os.path.join(args.save_dir, f'closeness_mechanism_{args.data_name}.pdf')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nFigure saved: {out}')
    print('\nCloseness mechanism analysis complete!')


if __name__ == '__main__':
    main()
