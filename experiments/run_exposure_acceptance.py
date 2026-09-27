"""
Experiments (h) + exposure mechanism: acceptance-robust EG variants and
exposure-by-group analysis, from one shared training pass.

Recommendations are not stored by the main pipeline, so both analyses
need models retrained at the best configurations. This runner trains
each method ONCE per run and derives both outputs:

  1. Acceptance-robust EG (RQ8, ToRS §7.6; all datasets):
     Degree/PageRank/Closeness EG under full / TP / PW acceptance
     (metrics/acceptance_eg.py, spec in ToRS §3.4) for
     RA, HL-GNN, HL-GNN+RA (fusion), CAFE-In, CAFE-Pre.
     -> results/{ds}/acceptance_eg/acceptance_eg_{ds}.csv
        (long format: run, method, metric, value; includes per-rank
        precision rho_r rows)

  2. Exposure by group and coreness tier (ToRS §8.3; datasets with
     sensitive attributes): per-node exposure
     Exp(v) = (1/|U|) * sum_u 1[v in Rec_K(u)] per method; mean
     exposure and Delta vs HL-GNN per (group x tier); Delta_SP.
     -> {exposure_dir}/exposure_by_group_and_tier_{ds}.csv   (summary)
     -> {exposure_dir}/exposure_per_node_{ds}.csv            (raw)

Protocol matches the stratified/uplift analyses: fixed split (seed 42),
--runs repetitions varying training stochasticity.

Usage:
    python run_exposure_acceptance.py --data_name oklahoma \
        --cafe_in_lambda 0.3 --cafe_in_beta 5.0 --cafe_in_lam 0.5 \
        --cafe_pre_alpha 1.0 --cafe_pre_beta 0.5 --cafe_pre_lam 0.5 \
        --runs 3 \
        --acceptance_csv ../results/oklahoma/acceptance_eg/acceptance_eg_oklahoma.csv \
        --exposure_dir ../outputs/coreness_by_group
"""

import os
import sys
import argparse
import gc

import numpy as np
import pandas as pd
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
for path in [parent_dir, current_dir, os.path.join(parent_dir, 'backbone')]:
    if path not in sys.path:
        sys.path.insert(0, path)

from core import set_seed, load_dataset, str2bool  # noqa: E402
from core.data_loader import load_sensitive_attr  # noqa: E402
from metrics.coreness import compute_coreness  # noqa: E402
from metrics.acceptance_eg import compute_acceptance_robust_eg  # noqa: E402
from cafe_methods import (  # noqa: E402
    add_common_args, build_train_adj_csr, build_ra_sparse,
    mobility_multiplier_vector, train_cafe_model,
    build_cafe_recs, build_heuristic_recs, get_eval_nodes,
)


def argument():
    parser = argparse.ArgumentParser(
        description='Acceptance-robust EG + exposure-by-group (shared training)')
    parser = add_common_args(parser)

    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--K', type=int, default=10)

    # Best configs (same conventions as run_significance.py)
    parser.add_argument('--cafe_in_lambda', type=float, required=True)
    parser.add_argument('--cafe_in_beta', type=float, required=True)
    parser.add_argument('--cafe_in_lam', type=float, required=True)
    parser.add_argument('--cafe_pre_alpha', type=float, required=True)
    parser.add_argument('--cafe_pre_beta', type=float, required=True)
    parser.add_argument('--cafe_pre_lam', type=float, required=True)
    parser.add_argument('--fusion_beta', type=float, default=None)

    parser.add_argument('--acceptance_csv', type=str, required=True)
    parser.add_argument('--exposure_dir', type=str, required=True,
                        help='output dir for exposure CSVs (skipped if the '
                             'dataset has no sensitive attributes)')
    parser.add_argument('--use_sampling_closeness', type=str2bool, default=True)
    parser.add_argument('--tercile_closure', choices=['left', 'right'],
                        default='left',
                        help="target-coreness tercile boundaries. 'left' "
                             "(default) reproduces the published audit; "
                             "'right' puts a boundary-valued tie in the LOWER "
                             "tier, which is what keeps Brightkite's large "
                             "coreness-1 group peripheral.")
    return parser.parse_args()


def compute_exposure(recs_by_u, num_nodes):
    """Exp(v) = (1/|U|) sum_u 1[v in Rec_K(u)] over eval users U."""
    exp = np.zeros(num_nodes, dtype=np.float64)
    for u, vs in recs_by_u.items():
        for v in vs:
            if 0 <= v < num_nodes:
                exp[v] += 1.0
    return exp / max(len(recs_by_u), 1)


def main():
    args = argument()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    os.makedirs(os.path.dirname(os.path.abspath(args.acceptance_csv)), exist_ok=True)
    os.makedirs(args.exposure_dir, exist_ok=True)
    fusion_beta = args.fusion_beta if args.fusion_beta is not None else args.cafe_in_beta

    print("=" * 80)
    print("ACCEPTANCE-ROBUST EG + EXPOSURE-BY-GROUP")
    print(f"Dataset: {args.data_name} | K={args.K} | runs={args.runs}")
    print(f"CAFE-In : lambda={args.cafe_in_lambda}, beta={args.cafe_in_beta}, lam={args.cafe_in_lam}")
    print(f"CAFE-Pre: alpha={args.cafe_pre_alpha}, beta={args.cafe_pre_beta}, lam={args.cafe_pre_lam}")
    print(f"Fusion  : beta={fusion_beta}")
    print("=" * 80)

    acceptance_rows = []
    exposure_node_rows = []
    has_attributes = False  # set per run; read after the loop for Delta_SP

    for run in range(args.runs):
        print(f"\n{'=' * 80}\nRUN {run + 1}/{args.runs}\n{'=' * 80}")

        dataset, split_edge, data, degrees = load_dataset(args)
        num_nodes = dataset._data.num_nodes if hasattr(dataset, '_data') else dataset.data.num_nodes
        train_edges = split_edge['train']['edge'].to(device)
        pos_test_edge = split_edge['test']['edge'].to(device)

        sens = None
        try:
            sens_t = load_sensitive_attr(args.data_name, num_nodes, root=args.data_path)
            sens = np.asarray(sens_t.cpu() if torch.is_tensor(sens_t) else sens_t).astype(int)
            print(f"Sensitive attributes loaded (n0={(sens == 0).sum()}, n1={(sens == 1).sum()})")
        except ValueError as e:
            print(f"No sensitive attributes ({e}); exposure analysis will be skipped.")

        coreness = compute_coreness(train_edges, num_nodes).to(device)
        data.coreness = coreness
        c_np = coreness.cpu().numpy()

        adj_sp = build_train_adj_csr(train_edges, num_nodes)
        ra_sparse = build_ra_sparse(adj_sp)
        eval_nodes = get_eval_nodes(pos_test_edge)
        print(f"Eval nodes: {len(eval_nodes)}")

        # ---- Recommendations per method (one training per model) ----
        method_recs = {}

        print("\n[RA]")
        method_recs['RA'] = build_heuristic_recs(
            ra_sparse, adj_sp, eval_nodes, args.K, verbose=False)

        print("[HL-GNN] training...")
        vanilla = train_cafe_model(data, split_edge, args, degrees, coreness,
                                   device, mode='none')
        method_recs['HL-GNN'] = build_cafe_recs(
            vanilla['h_emb'], vanilla['score_fn'], eval_nodes, num_nodes,
            device, adj_sp, beta=0.0, k=args.K, verbose=False)
        method_recs['HL-GNN+RA'] = build_cafe_recs(
            vanilla['h_emb'], vanilla['score_fn'], eval_nodes, num_nodes,
            device, adj_sp, ra_sparse=ra_sparse, beta=fusion_beta,
            k=args.K, verbose=False)
        del vanilla
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print("[CAFE-In] training...")
        cafe_in = train_cafe_model(data, split_edge, args, degrees, coreness,
                                   device, mode='inproc', pen_val=args.cafe_in_lambda)
        mult = mobility_multiplier_vector(c_np, args.cafe_in_lam, args.coreness_alpha)
        method_recs['CAFE-In'] = build_cafe_recs(
            cafe_in['h_emb'], cafe_in['score_fn'], eval_nodes, num_nodes,
            device, adj_sp, ra_sparse=ra_sparse, beta=args.cafe_in_beta,
            mult_vec=mult, k=args.K, verbose=False)
        del cafe_in
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print("[CAFE-Pre] training...")
        cafe_pre = train_cafe_model(data, split_edge, args, degrees, coreness,
                                    device, mode='preproc', pen_val=args.cafe_pre_alpha)
        mult = mobility_multiplier_vector(c_np, args.cafe_pre_lam, args.coreness_alpha)
        method_recs['CAFE-Pre'] = build_cafe_recs(
            cafe_pre['h_emb'], cafe_pre['score_fn'], eval_nodes, num_nodes,
            device, adj_sp, ra_sparse=ra_sparse, beta=args.cafe_pre_beta,
            mult_vec=mult, k=args.K, verbose=False)
        del cafe_pre
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # ---- 1. Acceptance-robust EG per method ----
        train_cpu = train_edges.cpu()
        test_cpu = pos_test_edge.cpu()
        for method, recs in method_recs.items():
            print(f"\n[acceptance EG] {method}")
            res, rho = compute_acceptance_robust_eg(
                recs, train_cpu, test_cpu, num_nodes, k=args.K,
                use_sampling_closeness=args.use_sampling_closeness)
            for metric, value in res.items():
                acceptance_rows.append({
                    'dataset': args.data_name, 'run': run, 'method': method,
                    'metric': metric, 'value': float(value)})
            for r, val in enumerate(rho):
                acceptance_rows.append({
                    'dataset': args.data_name, 'run': run, 'method': method,
                    'metric': f'rho_rank_{r + 1}', 'value': float(val)})
            summary = {m: round(res[m], 2) for m in res if m.startswith(('Degree', 'Pagerank', 'Closeness'))}
            print(f"  {summary}")

        # ---- 2. Exposure per node (ALL datasets) ----
        # The target-side audit of ToRS Section 7.2 pools the demographic
        # groups and therefore needs no sensitive attribute; only the
        # Delta_SP column further down does. Gating this whole block on
        # `sens` is what left Ego-FB and Brightkite without an audit.
        # Datasets without attributes get group = -1 and no Delta_SP.
        has_attributes = sens is not None
        valid = (sens >= 0) if sens is not None else np.ones(num_nodes, dtype=bool)
        terciles = np.percentile(c_np[valid], [100 / 3, 200 / 3])
        tier_idx = np.digitize(c_np, terciles,
                               right=(args.tercile_closure == 'right'))
        tier_names = np.array(['periphery', 'mid', 'core'])

        # Coreness is integer-valued and heavily tied, so a tercile boundary
        # can coincide with a large tie and starve a tier. Report the sizes
        # rather than letting a degenerate split pass unnoticed.
        sizes = {name: int((tier_idx[valid] == i).sum())
                 for i, name in enumerate(tier_names)}
        print(f"Target-coreness terciles at {terciles} "
              f"({args.tercile_closure}-closed): {sizes}")
        if min(sizes.values()) < 0.05 * int(valid.sum()):
            print("  WARNING: a tercile holds <5% of nodes; a coreness tie "
                  "straddles a boundary. Consider --tercile_closure right.")

        for method, recs in method_recs.items():
            exp = compute_exposure(recs, num_nodes)
            for v in np.where(valid)[0]:
                exposure_node_rows.append({
                    'dataset': args.data_name, 'run': run, 'method': method,
                    'node_id': int(v),
                    'group': int(sens[v]) if sens is not None else -1,
                    'coreness': float(c_np[v]),
                    'tier': tier_names[tier_idx[v]],
                    'exposure': float(exp[v])})

    # ======================================================================
    # Save outputs
    # ======================================================================
    acc_df = pd.DataFrame(acceptance_rows)
    acc_df.to_csv(args.acceptance_csv, index=False)
    print(f"\nAcceptance-robust EG rows saved to {args.acceptance_csv}")

    # Console summary: mean over runs, wide by variant
    pivot = (acc_df[acc_df['metric'].str.contains('_EG@')]
             .groupby(['method', 'metric'])['value'].mean().unstack('metric'))
    print("\nAcceptance-robust EG (mean over runs):")
    print(pivot.round(2).to_string())

    if exposure_node_rows:
        node_df = pd.DataFrame(exposure_node_rows)
        node_path = os.path.join(args.exposure_dir,
                                 f'exposure_per_node_{args.data_name}.csv')
        node_df.to_csv(node_path, index=False)
        print(f"\nPer-node exposure saved to {node_path}")

        # Summary: mean exposure per (method, group, tier) per run,
        # then mean/std across runs, plus Delta vs HL-GNN and Delta_SP.
        per_run = (node_df.groupby(['run', 'method', 'group', 'tier'])['exposure']
                   .mean().reset_index())
        base = per_run[per_run['method'] == 'HL-GNN'] \
            .rename(columns={'exposure': 'exposure_base'}) \
            .drop(columns='method')
        per_run = per_run.merge(base, on=['run', 'group', 'tier'])
        per_run['delta_vs_hlgnn'] = per_run['exposure'] - per_run['exposure_base']

        summary = (per_run.groupby(['method', 'group', 'tier'])
                   .agg(exposure_mean=('exposure', 'mean'),
                        exposure_std=('exposure', 'std'),
                        delta_vs_hlgnn_mean=('delta_vs_hlgnn', 'mean'),
                        delta_vs_hlgnn_std=('delta_vs_hlgnn', 'std'))
                   .reset_index())
        summary.insert(0, 'dataset', args.data_name)

        # Delta_SP per method (over ALL nodes with attributes, matching §3.4).
        # Undefined without a sensitive attribute; the tercile audit above is
        # not, which is the whole point of decoupling them.
        if has_attributes:
            sp_rows = []
            for (r, m), grp in node_df.groupby(['run', 'method']):
                g0 = grp[grp['group'] == 0]['exposure'].mean()
                g1 = grp[grp['group'] == 1]['exposure'].mean()
                sp_rows.append({'run': r, 'method': m, 'delta_sp': abs(g0 - g1)})
            sp_df = pd.DataFrame(sp_rows).groupby('method')['delta_sp'] \
                .agg(['mean', 'std']).reset_index() \
                .rename(columns={'mean': 'delta_sp_mean', 'std': 'delta_sp_std'})
            summary = summary.merge(sp_df, on='method', how='left')
        else:
            summary['delta_sp_mean'] = float('nan')
            summary['delta_sp_std'] = float('nan')
            print("No sensitive attributes: Delta_SP columns left empty; "
                  "the target-coreness audit above is unaffected.")

        sum_path = os.path.join(args.exposure_dir,
                                f'exposure_by_group_and_tier_{args.data_name}.csv')
        summary.to_csv(sum_path, index=False)
        print(f"Exposure summary saved to {sum_path}")
        print("\nDelta exposure vs HL-GNN by (method, group, tier):")
        print(summary.pivot_table(index=['group', 'tier'], columns='method',
                                  values='delta_vs_hlgnn_mean')
              .round(6).to_string())
    else:
        print("\nNo sensitive attributes — exposure analysis skipped.")

    print("\nDone.")


if __name__ == '__main__':
    main()
