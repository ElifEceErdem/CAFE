"""
Experiment (a): Coreness-stratified evaluation of CAFE itself.

Reproduces the Figure 1-2 analysis (Precision@K and Equality Gain per
coreness bin) with CAFE-In / CAFE-Pre overlaid on the diagnostic methods,
so the paper shows the stratified effect of the *cure*, not just the disease.

Methods evaluated (all on the same split within each run):
  RA        : pure Resource Allocation heuristic
  HL-GNN    : vanilla backbone (trained with the CAFE pipeline, lambda=0, beta=0)
  Fusion    : HL-GNN + beta*RA (fusion-only control, no penalty, no boost)
  CAFE-In   : in-processing penalty + RA fusion + mobility boost
  CAFE-Pre  : pre-processing penalty + RA fusion + mobility boost

Training and hybrid scoring go through experiments/cafe_methods.py, which
mirrors experiments/run_cafe.py exactly.

Usage (best configs from analysis/best_configs.csv):
    python run_stratified_cafe.py --data_name oklahoma \
        --cafe_in_lambda 0.3 --cafe_in_beta 5.0 --cafe_in_lam 0.5 \
        --cafe_pre_alpha 1.0 --cafe_pre_beta 0.5 --cafe_pre_lam 0.5 \
        --runs 3
"""

import os
import sys
import argparse
import gc
import hashlib
import json

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
experiments_dir = os.path.join(parent_dir, 'experiments')
for path in [parent_dir, current_dir, experiments_dir,
             os.path.join(parent_dir, 'backbone')]:
    if path not in sys.path:
        sys.path.insert(0, path)

from core import set_seed, load_dataset  # noqa: E402
from metrics.coreness import compute_coreness  # noqa: E402
from cafe_methods import (  # noqa: E402
    add_common_args, build_train_adj_csr, build_ra_sparse,
    mobility_multiplier_vector, train_cafe_model,
    build_cafe_recs, build_heuristic_recs,
    validate_recommendations,
)
from metrics.graph_utils import canonicalize_undirected_edges  # noqa: E402

# Reuse figure-1/2 machinery so plots are directly comparable
import run_precision_by_coreness as rpbc  # noqa: E402

# Colors/markers for the new overlay methods
rpbc.MODEL_COLORS.update({
    'Fusion': '#7f8c8d',
    'CAFE-In': '#2ecc71',
    'CAFE-Pre': '#16a085',
})
rpbc.MODEL_MARKERS.update({
    'Fusion': 'v',
    'CAFE-In': 'X',
    'CAFE-Pre': '*',
})

MODEL_ORDER = ['RA', 'HL-GNN', 'Fusion', 'CAFE-In', 'CAFE-Pre']
DATASET_LABELS = {
    'ego-facebook': 'Ego-FB',
    'oklahoma': 'Oklahoma',
    'unc': 'UNC',
    'pokec_n': 'Pokec-n',
    'brightkite': 'Brightkite',
}


def argument():
    parser = argparse.ArgumentParser(
        description='Coreness-stratified evaluation of CAFE (experiment a)'
    )
    parser = add_common_args(parser)

    # Analysis
    parser.add_argument('--K', type=int, default=10, help='K for Precision@K and EG@K')
    parser.add_argument('--num_bins', type=int, default=10,
                        help='Number of coreness quantile bins')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--save_dir', type=str, default='plots/stratified_cafe')

    # Best CAFE-In config for this dataset (from best_configs.csv / paper Table)
    parser.add_argument('--cafe_in_lambda', type=float, default=None)
    parser.add_argument('--cafe_in_beta', type=float, default=None)
    parser.add_argument('--cafe_in_lam', type=float, default=None)

    # Best CAFE-Pre config for this dataset
    parser.add_argument('--cafe_pre_alpha', type=float, default=None)
    parser.add_argument('--cafe_pre_beta', type=float, default=None)
    parser.add_argument('--cafe_pre_lam', type=float, default=None)
    parser.add_argument('--selected_configs', type=str, default=None,
                        help='Frozen egfix selected_configs.csv')

    # Fusion-only control: HL-GNN + beta*RA, no penalty, no boost.
    # Default: reuse the CAFE-In beta so the control isolates penalty+boost.
    parser.add_argument('--fusion_beta', type=float, default=None)
    parser.add_argument('--include_fusion', type=lambda x: x.lower() == 'true',
                        default=True)
    parser.add_argument('--save_recommendations', type=lambda x: x.lower() == 'true',
                        default=False,
                        help='Save compressed top-K recommendation matrices')
    parser.add_argument(
        '--split_seed', type=int, default=42,
        help='Random split seed; ignored for the canonical Brightkite temporal split',
    )
    parser.add_argument(
        '--allow_brightkite_random_control',
        type=lambda x: x.lower() == 'true',
        default=False,
        help='Explicit opt-in for the Brightkite random-split protocol control',
    )

    return parser.parse_args()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_dataset(value):
    value = str(value).strip().lower().replace('_', '-')
    aliases = {
        'ego-facebook': 'ego-facebook',
        'ego-fb': 'ego-facebook',
        'pokec-n': 'pokec-n',
        'oklahoma': 'oklahoma',
        'unc': 'unc',
        'brightkite': 'brightkite',
    }
    if value not in aliases:
        raise ValueError(f"Unknown dataset name: {value!r}")
    return aliases[value]


def _resolve_selected_configs(args):
    selected_sha256 = None
    if args.selected_configs is not None:
        if not os.path.isfile(args.selected_configs):
            raise FileNotFoundError(
                f"Selected-config manifest not found: {args.selected_configs}"
            )
        selected_sha256 = _sha256_file(args.selected_configs)
        frame = pd.read_csv(args.selected_configs)
        required = {
            'dataset', 'variant', 'lambda_fair', 'preproc_alpha',
            'beta', 'lambda_mobility',
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"Selected-config manifest lacks columns: {sorted(missing)}"
            )
        wanted = _normalise_dataset(args.data_name)
        rows = frame.loc[
            frame['dataset'].map(_normalise_dataset) == wanted
        ].copy()
        if len(rows) != 2 or set(rows['variant']) != {'cafe_in', 'cafe_pre'}:
            raise ValueError(
                f"Expected exactly cafe_in/cafe_pre rows for {args.data_name}"
            )
        row_in = rows.loc[rows['variant'] == 'cafe_in'].iloc[0]
        row_pre = rows.loc[rows['variant'] == 'cafe_pre'].iloc[0]
        args.cafe_in_lambda = float(row_in['lambda_fair'])
        args.cafe_in_beta = float(row_in['beta'])
        args.cafe_in_lam = float(row_in['lambda_mobility'])
        args.cafe_pre_alpha = float(row_pre['preproc_alpha'])
        args.cafe_pre_beta = float(row_pre['beta'])
        args.cafe_pre_lam = float(row_pre['lambda_mobility'])
        if 'penalty_reduction' in rows.columns:
            reductions = set(rows['penalty_reduction'].dropna().astype(str))
            if reductions and reductions != {args.penalty_reduction}:
                raise ValueError(
                    f"Penalty reduction mismatch: {reductions} vs "
                    f"{args.penalty_reduction}"
                )

    values = [
        args.cafe_in_lambda, args.cafe_in_beta, args.cafe_in_lam,
        args.cafe_pre_alpha, args.cafe_pre_beta, args.cafe_pre_lam,
    ]
    if any(value is None for value in values):
        raise ValueError(
            "Supply --selected_configs or all six CAFE parameters"
        )
    return selected_sha256


def _split_fingerprint(split_edge, num_nodes):
    digest = hashlib.sha256()
    digest.update(f"num_nodes={int(num_nodes)}\n".encode())
    for split_name in ('train', 'test'):
        edges = canonicalize_undirected_edges(
            split_edge[split_name]['edge'].cpu(), num_nodes
        )
        digest.update(f"{split_name}:{len(edges)}\n".encode())
        digest.update(
            edges.to(dtype=torch.int64).contiguous().numpy().astype(
                '<i8', copy=False
            ).tobytes()
        )
    return digest.hexdigest()


def _save_recommendation_artifact(path, all_recs, eval_users, metadata):
    sources = np.asarray(eval_users, dtype=np.int64)
    payload = {
        'sources': sources,
        'metadata_json': np.asarray(
            json.dumps(metadata, sort_keys=True), dtype=np.str_
        ),
    }
    for method, recs in all_recs.items():
        key = 'targets_' + method.lower().replace('-', '_')
        payload[key] = np.asarray(
            [recs[int(u)] for u in sources], dtype=np.int64
        )
    np.savez_compressed(path, **payload)


def main():
    args = argument()
    selected_configs_sha256 = _resolve_selected_configs(args)
    if not args.include_fusion:
        raise ValueError("The egfix stratified campaign requires Fusion")
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if args.closeness_sample_size < 0:
        raise ValueError("--closeness_sample_size must be 0 or positive")

    source_manifest_sha256 = None
    if args.source_manifest is not None:
        if not os.path.isfile(args.source_manifest):
            raise FileNotFoundError(
                f"Source manifest not found: {args.source_manifest}"
            )
        source_manifest_sha256 = _sha256_file(args.source_manifest)

    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError(
            'val_ratio and test_ratio must be non-negative and sum to less than 1'
        )

    split_sha256 = None
    force_random_brightkite = (
        str(args.data_name).lower() == 'brightkite'
        and isinstance(args.split_file, str)
        and args.split_file.strip().lower() == 'random'
    )
    if str(args.data_name).lower() == 'brightkite':
        split_path = args.split_file or os.path.join(
            args.data_path, 'brightkite', 'temporal_split.pt'
        )
        if force_random_brightkite and not args.allow_brightkite_random_control:
            raise ValueError(
                'A random Brightkite split is permitted only with '
                '--allow_brightkite_random_control True'
            )
        if not force_random_brightkite and not os.path.isfile(split_path):
            raise FileNotFoundError(
                f"Brightkite temporal split not found: {split_path}"
            )
        if not force_random_brightkite:
            split_sha256 = _sha256_file(split_path)

    # The source-stratified experiment holds the graph split fixed and varies
    # only model initialization/sampling.
    args.data_split_seed = int(args.split_seed) if force_random_brightkite else 42
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    fusion_beta = args.fusion_beta if args.fusion_beta is not None else args.cafe_in_beta
    if (
        selected_configs_sha256 is not None
        and not np.isclose(fusion_beta, args.cafe_in_beta)
    ):
        raise ValueError(
            "The egfix stratified control must match CAFE-In beta"
        )

    print("=" * 80)
    print("CORENESS-STRATIFIED EVALUATION OF CAFE (experiment a)")
    print(f"Dataset: {args.data_name} | K={args.K} | bins={args.num_bins} | runs={args.runs}")
    print(f"CAFE-In : lambda={args.cafe_in_lambda}, beta={args.cafe_in_beta}, lam={args.cafe_in_lam}")
    print(f"CAFE-Pre: alpha={args.cafe_pre_alpha}, beta={args.cafe_pre_beta}, lam={args.cafe_pre_lam}")
    if args.include_fusion:
        print(f"Fusion  : beta={fusion_beta} (no penalty, no boost)")
    print(f"Training seed base: {args.seed}")
    split_protocol = 'brightkite_random_control' if force_random_brightkite else (
        'brightkite_proxy_temporal' if str(args.data_name).lower() == 'brightkite'
        else 'fixed_random'
    )
    print(f"Split protocol: {split_protocol} | split seed: {args.data_split_seed}")
    print(f"Penalty reduction: {args.penalty_reduction}")
    print("=" * 80)

    all_dfs = []
    per_run_eg_inputs = []
    run_metadata = []

    for run in range(args.runs):
        print(f"\n{'=' * 80}\nRUN {run + 1}/{args.runs}\n{'=' * 80}")

        # --- Data (same fixed split-seed convention as the main experiments) ---
        dataset, split_edge, data, degrees = load_dataset(args)
        num_nodes = dataset._data.num_nodes if hasattr(dataset, '_data') else dataset.data.num_nodes
        train_edges = split_edge['train']['edge'].to(device)
        pos_test_edge = split_edge['test']['edge'].to(device)
        split_fingerprint = _split_fingerprint(split_edge, num_nodes)
        split_edge_counts = {
            name: int(len(canonicalize_undirected_edges(
                split_edge[name]['edge'].cpu(), num_nodes
            )))
            for name in ('train', 'valid', 'test')
        }
        split_edge_total = sum(split_edge_counts.values())
        split_edge_fractions = {
            name: count / split_edge_total
            for name, count in split_edge_counts.items()
        }
        training_seed = int(args.seed + run)
        print(f"Training seed: {training_seed}")

        coreness = compute_coreness(train_edges, num_nodes).to(device)
        data.coreness = coreness
        c_np = coreness.cpu().numpy()

        adj_sp = build_train_adj_csr(train_edges, num_nodes)
        ra_sparse = build_ra_sparse(adj_sp)

        # Same eval-population convention as Figures 1-2: every node with a test edge
        test_edge_dict = rpbc.build_test_edge_dict(pos_test_edge)
        eval_users = sorted(test_edge_dict.keys())
        print(f"Evaluation nodes: {len(eval_users)}/{num_nodes}")

        all_recs = {}

        # --- RA heuristic ---
        print("\n[RA] generating recommendations...")
        all_recs['RA'] = build_heuristic_recs(ra_sparse, adj_sp, eval_users, args.K)
        validate_recommendations(
            all_recs['RA'], eval_users, num_nodes, args.K, adj_sp
        )

        # --- Vanilla HL-GNN (also used for Fusion) ---
        print("\n[HL-GNN] training vanilla backbone...")
        set_seed(training_seed)
        vanilla = train_cafe_model(data, split_edge, args, degrees, coreness,
                                   device, mode='none')
        print("[HL-GNN] generating recommendations...")
        all_recs['HL-GNN'] = build_cafe_recs(
            vanilla['h_emb'], vanilla['score_fn'], eval_users, num_nodes, device,
            adj_sp, ra_sparse=None, beta=0.0, mult_vec=None, k=args.K)
        validate_recommendations(
            all_recs['HL-GNN'], eval_users, num_nodes, args.K, adj_sp
        )

        if args.include_fusion:
            print(f"\n[Fusion] generating recommendations (beta={fusion_beta})...")
            all_recs['Fusion'] = build_cafe_recs(
                vanilla['h_emb'], vanilla['score_fn'], eval_users, num_nodes, device,
                adj_sp, ra_sparse=ra_sparse, beta=fusion_beta,
                mult_vec=None, k=args.K)
            validate_recommendations(
                all_recs['Fusion'], eval_users, num_nodes, args.K, adj_sp
            )

        del vanilla
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # --- CAFE-In ---
        print(f"\n[CAFE-In] training (lambda={args.cafe_in_lambda})...")
        set_seed(training_seed)
        cafe_in = train_cafe_model(data, split_edge, args, degrees, coreness,
                                   device, mode='inproc', pen_val=args.cafe_in_lambda)
        mult_in = mobility_multiplier_vector(c_np, args.cafe_in_lam, args.coreness_alpha)
        print("[CAFE-In] generating recommendations...")
        all_recs['CAFE-In'] = build_cafe_recs(
            cafe_in['h_emb'], cafe_in['score_fn'], eval_users, num_nodes, device,
            adj_sp, ra_sparse=ra_sparse, beta=args.cafe_in_beta,
            mult_vec=mult_in, k=args.K)
        validate_recommendations(
            all_recs['CAFE-In'], eval_users, num_nodes, args.K, adj_sp
        )

        del cafe_in
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # --- CAFE-Pre ---
        print(f"\n[CAFE-Pre] training (alpha={args.cafe_pre_alpha})...")
        set_seed(training_seed)
        cafe_pre = train_cafe_model(data, split_edge, args, degrees, coreness,
                                    device, mode='preproc', pen_val=args.cafe_pre_alpha)
        mult_pre = mobility_multiplier_vector(c_np, args.cafe_pre_lam, args.coreness_alpha)
        print("[CAFE-Pre] generating recommendations...")
        all_recs['CAFE-Pre'] = build_cafe_recs(
            cafe_pre['h_emb'], cafe_pre['score_fn'], eval_users, num_nodes, device,
            adj_sp, ra_sparse=ra_sparse, beta=args.cafe_pre_beta,
            mult_vec=mult_pre, k=args.K)
        validate_recommendations(
            all_recs['CAFE-Pre'], eval_users, num_nodes, args.K, adj_sp
        )

        del cafe_pre
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # --- Per-node precision dataframes ---
        degrees_np = degrees.cpu().numpy()

        def build_df(precisions, model_name):
            rows = [{
                'node_id': u,
                'model': model_name,
                'precision': p,
                'node_coreness': c_np[u],
                'degree': float(degrees_np[u]),
                'run': training_seed,
                'training_seed': training_seed,
                'split_seed': int(args.data_split_seed),
                'split_fingerprint': split_fingerprint,
            } for u, p in precisions.items()]
            return pd.DataFrame(rows)

        run_dfs = []
        for model_name, recs in all_recs.items():
            prec = rpbc.compute_per_node_precision(recs, test_edge_dict, args.K)
            run_dfs.append(build_df(prec, model_name))
        all_dfs.append(pd.concat(run_dfs, ignore_index=True))

        per_run_eg_inputs.append({
            'run': training_seed,
            'train_edges': train_edges.cpu(),
            'test_edges': pos_test_edge.cpu(),
            'num_nodes': num_nodes,
            'coreness_np': c_np.copy(),
            'recs': all_recs,
        })

        metadata = {
            'status': 'complete',
            'dataset': args.data_name,
            'training_seed': training_seed,
            'split_seed': int(args.data_split_seed),
            'split_protocol': split_protocol,
            'split_fingerprint': split_fingerprint,
            'split_sha256': split_sha256,
            'split_edge_counts': split_edge_counts,
            'split_edge_fractions': split_edge_fractions,
            'num_nodes': int(num_nodes),
            'num_eval_users': len(eval_users),
            'k': int(args.K),
            'num_bins_requested': int(args.num_bins),
            'methods': list(all_recs),
            'eg_graph_semantics': 'simple_undirected_set_union',
            'evaluation_split': 'test',
            'penalty_reduction': args.penalty_reduction,
            'selected_configs_sha256': selected_configs_sha256,
            'source_manifest_sha256': source_manifest_sha256,
            'fusion_beta': float(fusion_beta),
            'cafe_in': {
                'lambda_fair': float(args.cafe_in_lambda),
                'beta': float(args.cafe_in_beta),
                'lambda_mobility': float(args.cafe_in_lam),
            },
            'cafe_pre': {
                'preproc_alpha': float(args.cafe_pre_alpha),
                'beta': float(args.cafe_pre_beta),
                'lambda_mobility': float(args.cafe_pre_lam),
            },
            'recommendation_validation_passed': True,
        }
        run_metadata.append(metadata)
        if args.save_recommendations:
            rec_path = os.path.join(
                args.save_dir,
                f"recommendations_{args.data_name}_seed{training_seed}.npz",
            )
            _save_recommendation_artifact(
                rec_path, all_recs, eval_users, metadata
            )
            metadata['recommendations_file'] = os.path.basename(rec_path)
            metadata['recommendations_sha256'] = _sha256_file(rec_path)

    # =====================================================================
    # Aggregation, EG per coreness bin, plotting (same flow as Figures 1-2)
    # =====================================================================
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df, bin_labels = rpbc.assign_coreness_bins(full_df, args.num_bins)

    csv_path = os.path.join(args.save_dir, f'stratified_cafe_{args.data_name}.csv')
    full_df.to_csv(csv_path, index=False)
    print(f"\nPer-node precision data saved to {csv_path}")

    print("\nComputing Equality Gain by coreness bin (per run)...")
    per_run_eg_dfs = []
    for run_position, rd in enumerate(per_run_eg_inputs, start=1):
        print(
            f"\n--- EG run {run_position}/{len(per_run_eg_inputs)} "
            f"(training seed {rd['run']}) ---"
        )
        run_df = full_df[full_df['run'] == rd['run']]
        run_bin_assignments = {}
        for nid, b in zip(run_df['node_id'].astype(int),
                          run_df['coreness_bin'].astype(int)):
            if nid not in run_bin_assignments:
                run_bin_assignments[nid] = b

        eg_df_run = rpbc.compute_eg_by_coreness_bin(
            rd['recs'], rd['coreness_np'], run_bin_assignments, bin_labels,
            rd['train_edges'], rd['test_edges'], rd['num_nodes'], args.K
        )
        eg_df_run['run'] = rd['run']
        eg_df_run['training_seed'] = rd['run']
        eg_df_run['split_seed'] = int(args.data_split_seed)
        per_run_eg_dfs.append(eg_df_run)

    eg_df_all_runs = pd.concat(per_run_eg_dfs, ignore_index=True)
    eg_agg = (
        eg_df_all_runs
        .groupby(['model', 'coreness_bin', 'coreness_bin_label'], as_index=False)
        .agg(equality_gain=('equality_gain', 'mean'),
             eg_std=('equality_gain', 'std'),
             eg_count=('equality_gain', 'count'))
    )
    eg_agg['eg_sem'] = np.where(
        eg_agg['eg_count'] > 1,
        eg_agg['eg_std'].fillna(0.0) / np.sqrt(eg_agg['eg_count'].clip(lower=1)),
        0.0,
    )

    eg_runs_csv = os.path.join(args.save_dir,
                               f'eg_by_coreness_cafe_{args.data_name}_per_run.csv')
    eg_df_all_runs.to_csv(eg_runs_csv, index=False)
    eg_csv = os.path.join(args.save_dir, f'eg_by_coreness_cafe_{args.data_name}.csv')
    eg_agg.to_csv(eg_csv, index=False)
    print(f"EG data saved to {eg_csv}")

    print("\nGenerating stacked bar chart (Fig 1-2 style, CAFE overlaid)...")
    rpbc.plot_stacked_barchart(
        full_df,
        eg_agg,
        args.save_dir,
        args.data_name,
        args.K,
        model_order=MODEL_ORDER,
        display_name=DATASET_LABELS.get(args.data_name, args.data_name),
        filename_tag=args.data_name,
    )

    # Summary tables for the paper text
    print("\n" + "=" * 80)
    print(f"SUMMARY: Mean Precision@{args.K} by coreness bin")
    print("=" * 80)
    per_run_precision = rpbc.aggregate_precision_within_runs(full_df)
    summary_df = (
        per_run_precision
        .groupby(
            ['model', 'coreness_bin', 'coreness_bin_label'],
            as_index=False,
        )
        .agg(
            precision_mean=('precision', 'mean'),
            precision_std=('precision', 'std'),
            run_count=('precision', 'count'),
            n_nodes=('n_nodes', 'max'),
        )
    )
    summary_df['precision_std'] = summary_df['precision_std'].fillna(0.0)
    summary_df['precision_sem'] = np.where(
        summary_df['run_count'] > 1,
        summary_df['precision_std']
        / np.sqrt(summary_df['run_count'].clip(lower=1)),
        0.0,
    )
    summary_csv = os.path.join(args.save_dir,
                               f'summary_precision_by_bin_{args.data_name}.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(summary_df.to_string(index=False))

    # The headline number for the rebuttal: where does CAFE's lift come from?
    print("\n" + "=" * 80)
    print("PERIPHERY vs CORE LIFT (CAFE vs HL-GNN)")
    print("=" * 80)
    pivot = summary_df.pivot_table(index='coreness_bin', columns='model',
                                   values='precision_mean', sort=False)
    for cafe_variant in ['CAFE-In', 'CAFE-Pre', 'Fusion']:
        if cafe_variant in pivot.columns and 'HL-GNN' in pivot.columns:
            lift = pivot[cafe_variant] - pivot['HL-GNN']
            print(f"\nDelta Precision@{args.K} ({cafe_variant} - HL-GNN) per bin:")
            print(lift.to_string())

    metadata_path = os.path.join(
        args.save_dir, f'metadata_{args.data_name}.json'
    )
    metadata_output = {
        'status': 'complete',
        'dataset': args.data_name,
        'runs': run_metadata,
        'actual_bins': int(full_df['coreness_bin'].nunique()),
        'precision_rows': int(len(full_df)),
        'eg_rows': int(len(eg_df_all_runs)),
        'methods': MODEL_ORDER,
        'selected_configs_sha256': selected_configs_sha256,
        'source_manifest_sha256': source_manifest_sha256,
        'eg_graph_semantics': 'simple_undirected_set_union',
        'penalty_reduction': args.penalty_reduction,
        'split_protocol': split_protocol,
        'split_seed': int(args.data_split_seed),
    }
    temp_metadata = f"{metadata_path}.tmp.{os.getpid()}"
    with open(temp_metadata, 'w') as handle:
        json.dump(metadata_output, handle, indent=2, sort_keys=True)
    os.replace(temp_metadata, metadata_path)

    print("\nStratified CAFE evaluation complete!")
    print("EGFIX_RUN_COMPLETE")


if __name__ == '__main__':
    main()
