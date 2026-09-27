"""
Experiment (c): per-seed paired runs for statistical significance.

For each seed s:
  1. set_seed(s); draw the train/val/test split with data_split_seed = s
     (so split variance is included; disable with --vary_split False)
  2. On that SAME split, evaluate:
       RA        : pure Resource Allocation heuristic
       HL-GNN    : vanilla backbone
       Fusion    : HL-GNN + beta*RA        (no penalty, no boost)
       CAFE-In   : penalty + RA + boost    (best per-dataset config)
       CAFE-Pre  : preproc + RA + boost    (best per-dataset config)
  3. Append one row per (seed, method, metric) to a long-format CSV.

Because every method within a seed shares the split, per-seed differences
are properly paired; analysis/significance_tests.py consumes the CSV and
runs paired t-tests / Wilcoxon with Holm correction and effect sizes.

The CSV is written incrementally after each seed, and seeds already present
in the output file are skipped, so jobs can be resumed / extended (e.g. from
10 to 30 seeds) without recomputation.

Usage:
    python run_significance.py --data_name oklahoma \
        --cafe_in_lambda 0.3 --cafe_in_beta 5.0 --cafe_in_lam 0.5 \
        --cafe_pre_alpha 1.0 --cafe_pre_beta 0.5 --cafe_pre_lam 0.5 \
        --seed_start 0 --num_seeds 20 \
        --out_csv ../results/oklahoma/significance/per_seed_metrics_oklahoma.csv
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

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
for path in [parent_dir, current_dir, os.path.join(parent_dir, 'backbone')]:
    if path not in sys.path:
        sys.path.insert(0, path)

from core import set_seed, load_dataset, str2bool  # noqa: E402
from core.data_loader import load_sensitive_attr  # noqa: E402
from core.evaluation import compute_metrics_from_recs, compute_equality_gain_metrics  # noqa: E402
from core.statistics import _eg_semantics_marker  # noqa: E402  (label = metric code in use)
from metrics.coreness import compute_coreness  # noqa: E402
from metrics.centrality_equality_gain import compute_all_centrality_equality_gains  # noqa: E402
from metrics.graph_utils import canonicalize_undirected_edges  # noqa: E402
from metrics.demographics import (  # noqa: E402
    compute_demographic_metrics, compute_demographic_parity_ratio_change,
)
from cafe_methods import (  # noqa: E402
    add_common_args, build_train_adj_csr, build_ra_sparse,
    mobility_multiplier_vector, train_cafe_model,
    build_cafe_recs, build_heuristic_recs, get_eval_nodes,
    validate_recommendations,
)


def argument():
    parser = argparse.ArgumentParser(
        description='Per-seed paired significance runs (experiment c)'
    )
    parser = add_common_args(parser)

    # Seeds
    parser.add_argument('--seed_start', type=int, default=0)
    parser.add_argument('--num_seeds', type=int, default=20)
    parser.add_argument('--vary_split', type=str2bool, default=True,
                        help='True: split seed = run seed (captures split variance). '
                             'False: fixed split seed 42 (training variance only).')

    # Best CAFE configs for this dataset
    parser.add_argument('--cafe_in_lambda', type=float, default=None)
    parser.add_argument('--cafe_in_beta', type=float, default=None)
    parser.add_argument('--cafe_in_lam', type=float, default=None)
    parser.add_argument('--cafe_pre_alpha', type=float, default=None)
    parser.add_argument('--cafe_pre_beta', type=float, default=None)
    parser.add_argument('--cafe_pre_lam', type=float, default=None)
    parser.add_argument('--fusion_beta', type=float, default=None,
                        help='beta for the fusion-only control (default: cafe_in_beta)')
    parser.add_argument('--selected_configs', type=str, default=None,
                        help='Frozen egfix selected_configs.csv; when supplied, '
                             'all six CAFE parameters are read from it')
    parser.add_argument('--protocol', type=str, default='paired',
                        choices=['headline', 'paired'],
                        help='Provenance label for fixed-split headline or '
                             'varying-split paired runs')

    # Metrics
    parser.add_argument('--k_values', type=str, default='10',
                        help='comma-separated K values for EG/centrality metrics')
    # Same contract as run_cafe.py / run_coreness_inverse.py: a LIST on args,
    # so downstream consumers reading args.centrality_types do not iterate a
    # bare string character by character.
    parser.add_argument('--centrality_types',
                        type=lambda s: (['pagerank', 'closeness'] if s is None
                                        else [x.strip() for x in s.split(',')]),
                        default='pagerank,closeness')
    parser.add_argument('--use_demographics', type=str2bool, default=True)

    parser.add_argument('--out_csv', type=str, required=True)
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
        raise ValueError(f"Unknown dataset name in selected config: {value!r}")
    return aliases[value]


def _resolve_selected_configs(args):
    """Populate runner parameters from the frozen selector output."""
    selected_sha256 = None
    if args.selected_configs is not None:
        if not os.path.isfile(args.selected_configs):
            raise FileNotFoundError(
                f"Selected-config manifest not found: {args.selected_configs}"
            )
        selected_sha256 = _sha256_file(args.selected_configs)
        frame = pd.read_csv(args.selected_configs)
        required_columns = {
            'dataset', 'variant', 'lambda_fair', 'preproc_alpha',
            'beta', 'lambda_mobility',
        }
        missing = required_columns - set(frame.columns)
        if missing:
            raise ValueError(
                f"Selected-config manifest lacks columns: {sorted(missing)}"
            )
        wanted = _normalise_dataset(args.data_name)
        dataset_norm = frame['dataset'].map(_normalise_dataset)
        rows = frame.loc[dataset_norm == wanted].copy()
        if set(rows['variant']) != {'cafe_in', 'cafe_pre'} or len(rows) != 2:
            raise ValueError(
                f"Expected exactly cafe_in/cafe_pre rows for {args.data_name}; "
                f"found {rows[['dataset', 'variant']].to_dict('records')}"
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
                    "Selected configuration reduction does not match runner: "
                    f"{reductions} vs {args.penalty_reduction}"
                )

    values = {
        'cafe_in_lambda': args.cafe_in_lambda,
        'cafe_in_beta': args.cafe_in_beta,
        'cafe_in_lam': args.cafe_in_lam,
        'cafe_pre_alpha': args.cafe_pre_alpha,
        'cafe_pre_beta': args.cafe_pre_beta,
        'cafe_pre_lam': args.cafe_pre_lam,
    }
    missing_values = [name for name, value in values.items() if value is None]
    if missing_values:
        raise ValueError(
            "Supply --selected_configs or all six CAFE parameters; missing "
            + ', '.join(missing_values)
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


def flatten_metrics(res):
    """Flatten a metric dict (values may be (0, val) tuples or scalars)."""
    flat = {}
    for name, val in res.items():
        if name.startswith('_'):
            continue
        if isinstance(val, (tuple, list)) and len(val) >= 2:
            val = val[1]
        if hasattr(val, 'item'):
            val = val.item()
        try:
            flat[name] = float(val)
        except (TypeError, ValueError):
            pass  # skip non-scalar entries
    return flat


def evaluate_method(recs_by_u, train_edges_cpu, test_edges_cpu, num_nodes, args,
                    k_values, centrality_types, sens_attr=None, h_emb=None,
                    centrality_control_cache=None):
    """Full metric battery for one method's recommendations."""
    res = compute_metrics_from_recs(recs_by_u, test_edges_cpu, num_nodes,
                                    args.fairness_mode, k_values=k_values)
    res.update(compute_equality_gain_metrics(
        recs_by_u, train_edges_cpu, test_edges_cpu, num_nodes, k_values=k_values))
    res.update(compute_all_centrality_equality_gains(
        recs_by_u, train_edges_cpu, test_edges_cpu, num_nodes,
        k_values=k_values, centrality_types=centrality_types,
        use_sampling_closeness=True,
        closeness_random_seed=args.closeness_seed,
        closeness_sample_size=(
            None if args.closeness_sample_size == 0
            else args.closeness_sample_size
        ),
        closeness_epsilon=args.closeness_epsilon,
        control_cache=centrality_control_cache,
    ))

    if sens_attr is not None:
        demo = compute_demographic_metrics(
            recs_by_u=recs_by_u,
            train_edges=train_edges_cpu,
            test_edges=test_edges_cpu,
            num_nodes=num_nodes,
            sens=sens_attr,
            h_emb=h_emb.cpu() if h_emb is not None else None,
            features=None,
            adj=None,
            k=10,
        )
        res.update(demo)
        res['Delta_DPR'] = compute_demographic_parity_ratio_change(
            train_edges_cpu, recs_by_u, num_nodes, sens_attr, k=10)

    return flatten_metrics(res)


def load_done_seeds(out_csv):
    if not os.path.exists(out_csv):
        return set()
    try:
        df = pd.read_csv(out_csv)
        # A seed counts as done only when the four campaign objectives are
        # present for all five methods.  This prevents an interrupted partial
        # CSV from being mistaken for a completed shard.
        required_methods = {'RA', 'HL-GNN', 'Fusion', 'CAFE-In', 'CAFE-Pre'}
        required_metrics = {
            'Precision@10',
            'Equality_Gain@10',
            'Pagerank_EG@10',
            'Closeness_EG@10',
        }
        done = set()
        for s, grp in df.groupby('seed'):
            methods = set(grp['method'])
            complete = methods == required_methods
            for method in required_methods:
                metrics = set(grp.loc[grp['method'] == method, 'metric'])
                complete = complete and required_metrics.issubset(metrics)
            if complete:
                done.add(int(s))
        return done
    except Exception:
        return set()


def write_seed_rows(out_csv, rows):
    """Atomically upsert one seed, making interrupted shards resumable."""
    fresh = pd.DataFrame(rows)
    if os.path.exists(out_csv):
        existing = pd.read_csv(out_csv)
        seed = int(fresh['seed'].iloc[0])
        dataset = str(fresh['dataset'].iloc[0])
        keep = ~(
            (existing['seed'].astype(int) == seed)
            & (existing['dataset'].astype(str) == dataset)
        )
        fresh = pd.concat([existing.loc[keep], fresh], ignore_index=True)

    temp_path = f"{out_csv}.tmp.{os.getpid()}"
    fresh.to_csv(temp_path, index=False)
    os.replace(temp_path, out_csv)


def main():
    args = argument()
    selected_configs_sha256 = _resolve_selected_configs(args)
    if args.protocol == 'headline' and args.vary_split:
        raise ValueError(
            "Headline protocol requires --vary_split False"
        )
    if args.closeness_sample_size < 0:
        raise ValueError("--closeness_sample_size must be 0 or positive")

    source_manifest_sha256 = None
    if args.source_manifest is not None:
        if not os.path.isfile(args.source_manifest):
            raise FileNotFoundError(
                f"Source manifest not found: {args.source_manifest}"
            )
        source_manifest_sha256 = _sha256_file(args.source_manifest)

    split_sha256 = None
    if str(args.data_name).lower() == 'brightkite':
        split_path = args.split_file or os.path.join(
            args.data_path, 'brightkite', 'temporal_split.pt'
        )
        if str(split_path).lower() == 'random':
            raise ValueError(
                "The egfix campaign does not permit a random Brightkite split"
            )
        if not os.path.isfile(split_path):
            raise FileNotFoundError(
                f"Brightkite temporal split not found: {split_path}"
            )
        split_sha256 = _sha256_file(split_path)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)

    fusion_beta = args.fusion_beta if args.fusion_beta is not None else args.cafe_in_beta
    if (
        selected_configs_sha256 is not None
        and not np.isclose(fusion_beta, args.cafe_in_beta)
    ):
        raise ValueError(
            "The egfix aggregate requires matched-beta fusion: "
            f"fusion={fusion_beta}, CAFE-In={args.cafe_in_beta}"
        )
    k_values = [int(x) for x in args.k_values.split(',')]
    if 10 not in k_values:
        raise ValueError("--k_values must include 10")
    rec_k = max(max(k_values), 10)
    centrality_types = list(args.centrality_types)
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    done = load_done_seeds(args.out_csv)

    print("=" * 80)
    print("PER-SEED PAIRED SIGNIFICANCE RUNS (experiment c)")
    print(f"Dataset: {args.data_name} | seeds: {seeds}")
    print(f"Already done (skipped): {sorted(done & set(seeds))}")
    print(f"Vary split per seed: {args.vary_split}")
    print(f"Protocol: {args.protocol}")
    print(f"Penalty reduction: {args.penalty_reduction}")
    print(f"Approximate closeness seed: {args.closeness_seed}")
    print(f"CAFE-In : lambda={args.cafe_in_lambda}, beta={args.cafe_in_beta}, lam={args.cafe_in_lam}")
    print(f"CAFE-Pre: alpha={args.cafe_pre_alpha}, beta={args.cafe_pre_beta}, lam={args.cafe_pre_lam}")
    print(f"Fusion  : beta={fusion_beta}")
    print(f"Output: {args.out_csv}")
    print("=" * 80)

    for seed in seeds:
        if seed in done:
            continue
        print(f"\n{'=' * 80}\nSEED {seed}\n{'=' * 80}")

        # --- Split (shared by every method within this seed) ---
        set_seed(seed)
        if str(args.data_name).lower() == 'brightkite':
            # The canonical temporal artifact is fixed across repetitions.
            args.data_split_seed = 42
        else:
            args.data_split_seed = seed if args.vary_split else 42
        dataset, split_edge, data, degrees = load_dataset(args)
        num_nodes = dataset._data.num_nodes if hasattr(dataset, '_data') else dataset.data.num_nodes
        train_edges = split_edge['train']['edge'].to(device)
        pos_test_edge = split_edge['test']['edge'].to(device)
        train_edges_cpu = train_edges.cpu()
        test_edges_cpu = pos_test_edge.cpu()
        split_fingerprint = _split_fingerprint(split_edge, num_nodes)

        sens_attr = None
        if args.use_demographics:
            try:
                sens_attr = load_sensitive_attr(args.data_name, num_nodes,
                                                root=args.data_path)
                data.sens = torch.tensor(sens_attr, device=device) \
                    if not torch.is_tensor(sens_attr) else sens_attr.to(device)
            except ValueError as e:
                print(f"WARNING: no sensitive attributes ({e}); "
                      f"skipping demographic metrics.")

        coreness = compute_coreness(train_edges, num_nodes).to(device)
        data.coreness = coreness
        c_np = coreness.cpu().numpy()

        adj_sp = build_train_adj_csr(train_edges, num_nodes)
        ra_sparse = build_ra_sparse(adj_sp)
        eval_nodes = get_eval_nodes(pos_test_edge)
        print(f"Evaluating {len(eval_nodes)} test nodes")

        method_metrics = {}
        centrality_control_cache = {}

        # --- RA heuristic ---
        print("\n[RA]")
        recs = build_heuristic_recs(ra_sparse, adj_sp, eval_nodes, rec_k)
        validate_recommendations(
            recs, eval_nodes, num_nodes, rec_k, adj_sp
        )
        method_metrics['RA'] = evaluate_method(
            recs, train_edges_cpu, test_edges_cpu, num_nodes, args,
            k_values, centrality_types, sens_attr, h_emb=None,
            centrality_control_cache=centrality_control_cache)

        # --- Vanilla HL-GNN (shared by HL-GNN and Fusion rows) ---
        print("\n[HL-GNN] training...")
        set_seed(seed)  # identical RNG state for every trained method
        vanilla = train_cafe_model(data, split_edge, args, degrees, coreness,
                                   device, mode='none')
        recs = build_cafe_recs(vanilla['h_emb'], vanilla['score_fn'], eval_nodes,
                               num_nodes, device, adj_sp, beta=0.0, k=rec_k)
        validate_recommendations(
            recs, eval_nodes, num_nodes, rec_k, adj_sp
        )
        method_metrics['HL-GNN'] = evaluate_method(
            recs, train_edges_cpu, test_edges_cpu, num_nodes, args,
            k_values, centrality_types, sens_attr, h_emb=vanilla['h_emb'],
            centrality_control_cache=centrality_control_cache)

        print(f"\n[Fusion] beta={fusion_beta}")
        recs = build_cafe_recs(vanilla['h_emb'], vanilla['score_fn'], eval_nodes,
                               num_nodes, device, adj_sp, ra_sparse=ra_sparse,
                               beta=fusion_beta, k=rec_k)
        validate_recommendations(
            recs, eval_nodes, num_nodes, rec_k, adj_sp
        )
        method_metrics['Fusion'] = evaluate_method(
            recs, train_edges_cpu, test_edges_cpu, num_nodes, args,
            k_values, centrality_types, sens_attr, h_emb=vanilla['h_emb'],
            centrality_control_cache=centrality_control_cache)
        del vanilla, recs
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # --- CAFE-In ---
        print(f"\n[CAFE-In] training (lambda={args.cafe_in_lambda})...")
        set_seed(seed)
        cafe_in = train_cafe_model(data, split_edge, args, degrees, coreness,
                                   device, mode='inproc',
                                   pen_val=args.cafe_in_lambda)
        mult = mobility_multiplier_vector(c_np, args.cafe_in_lam, args.coreness_alpha)
        recs = build_cafe_recs(cafe_in['h_emb'], cafe_in['score_fn'], eval_nodes,
                               num_nodes, device, adj_sp, ra_sparse=ra_sparse,
                               beta=args.cafe_in_beta, mult_vec=mult, k=rec_k)
        validate_recommendations(
            recs, eval_nodes, num_nodes, rec_k, adj_sp
        )
        method_metrics['CAFE-In'] = evaluate_method(
            recs, train_edges_cpu, test_edges_cpu, num_nodes, args,
            k_values, centrality_types, sens_attr, h_emb=cafe_in['h_emb'],
            centrality_control_cache=centrality_control_cache)
        del cafe_in, recs
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # --- CAFE-Pre ---
        print(f"\n[CAFE-Pre] training (alpha={args.cafe_pre_alpha})...")
        set_seed(seed)
        cafe_pre = train_cafe_model(data, split_edge, args, degrees, coreness,
                                    device, mode='preproc',
                                    pen_val=args.cafe_pre_alpha)
        mult = mobility_multiplier_vector(c_np, args.cafe_pre_lam, args.coreness_alpha)
        recs = build_cafe_recs(cafe_pre['h_emb'], cafe_pre['score_fn'], eval_nodes,
                               num_nodes, device, adj_sp, ra_sparse=ra_sparse,
                               beta=args.cafe_pre_beta, mult_vec=mult, k=rec_k)
        validate_recommendations(
            recs, eval_nodes, num_nodes, rec_k, adj_sp
        )
        method_metrics['CAFE-Pre'] = evaluate_method(
            recs, train_edges_cpu, test_edges_cpu, num_nodes, args,
            k_values, centrality_types, sens_attr, h_emb=cafe_pre['h_emb'],
            centrality_control_cache=centrality_control_cache)
        del cafe_pre, recs
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # --- Persist this seed ---
        rows = []
        for method, metrics in method_metrics.items():
            for metric, value in metrics.items():
                rows.append({
                    'dataset': args.data_name,
                    'seed': seed,
                    'split_seed': args.data_split_seed,
                    'method': method,
                    'metric': metric,
                    'value': value,
                    'protocol': args.protocol,
                    'evaluation_split': 'test',
                    'eg_graph_semantics': _eg_semantics_marker(),
                    'penalty_reduction': args.penalty_reduction,
                    'closeness_random_seed': args.closeness_seed,
                    'closeness_sample_size': args.closeness_sample_size,
                    'closeness_epsilon': args.closeness_epsilon,
                    'split_fingerprint': split_fingerprint,
                    'split_sha256': split_sha256,
                    'selected_configs_sha256': selected_configs_sha256,
                    'source_manifest_sha256': source_manifest_sha256,
                    'cafe_in_lambda': args.cafe_in_lambda,
                    'cafe_in_beta': args.cafe_in_beta,
                    'cafe_in_lam': args.cafe_in_lam,
                    'cafe_pre_alpha': args.cafe_pre_alpha,
                    'cafe_pre_beta': args.cafe_pre_beta,
                    'cafe_pre_lam': args.cafe_pre_lam,
                    'fusion_beta': fusion_beta,
                })
        write_seed_rows(args.out_csv, rows)
        print(f"\nSeed {seed} written to {args.out_csv} ({len(rows)} rows)")

    if not os.path.isfile(args.out_csv):
        raise RuntimeError("No aggregate metrics were written")
    complete = {
        'status': 'complete',
        'dataset': args.data_name,
        'protocol': args.protocol,
        'seeds': seeds,
        'csv_sha256': _sha256_file(args.out_csv),
        'selected_configs_sha256': selected_configs_sha256,
        'source_manifest_sha256': source_manifest_sha256,
        'eg_graph_semantics': _eg_semantics_marker(),
        'penalty_reduction': args.penalty_reduction,
    }
    marker_path = f"{args.out_csv}.complete.json"
    temp_marker = f"{marker_path}.tmp.{os.getpid()}"
    with open(temp_marker, 'w') as handle:
        json.dump(complete, handle, indent=2, sort_keys=True)
    os.replace(temp_marker, marker_path)
    print("\nAll seeds complete.")
    print("EGFIX_RUN_COMPLETE")


if __name__ == '__main__':
    main()
