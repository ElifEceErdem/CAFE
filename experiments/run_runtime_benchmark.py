"""
Experiment (e): empirical runtime measurements vs. vanilla HL-GNN.

Complements the asymptotic complexity table with wall-clock numbers per
dataset (mean +/- std over --runs repetitions on the same GPU):

  Preprocessing:
    coreness_time      : k-core decomposition of the training graph
    ra_time            : sparse RA matrix construction + normalization
    adj_mod_time       : coreness-inverse adjacency modification (CAFE-Pre)

  Training (to early-stopping):
    train_vanilla      : HL-GNN, no penalty
    train_cafe_in      : loss-penalty training (best lambda_fair)
    train_cafe_pre     : pre-processed-adjacency training (best alpha_pre)

  Inference (all test nodes, top-100):
    infer_vanilla      : GNN scoring only
    infer_cafe         : GNN + beta*RA + mobility boost scoring

Outputs one JSON + a flat CSV row per (dataset, run, phase).

Usage:
    python run_runtime_benchmark.py --data_name oklahoma \
        --cafe_in_lambda 0.3 --cafe_pre_alpha 1.0 \
        --cafe_beta 5.0 --cafe_lam 0.5 --runs 3 \
        --out_json ../outputs/runtime/runtime_oklahoma.json
"""

import os
import sys
import argparse
import gc
import json
import platform
import time

import numpy as np
import pandas as pd
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
for path in [parent_dir, current_dir, os.path.join(parent_dir, 'backbone')]:
    if path not in sys.path:
        sys.path.insert(0, path)

from core import set_seed, load_dataset  # noqa: E402
from metrics.coreness import compute_coreness  # noqa: E402
from fairness_methods.inprocessing import apply_coreness_inverse_aggregation  # noqa: E402
from cafe_methods import (  # noqa: E402
    add_common_args, build_train_adj_csr, build_ra_sparse,
    mobility_multiplier_vector, train_cafe_model,
    build_cafe_recs, get_eval_nodes,
)


class Timer:
    """Wall-clock timer with CUDA synchronization."""
    def __init__(self):
        self.t0 = None

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.t0


def argument():
    parser = argparse.ArgumentParser(description='CAFE runtime benchmark (experiment e)')
    parser = add_common_args(parser)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--cafe_in_lambda', type=float, required=True)
    parser.add_argument('--cafe_pre_alpha', type=float, required=True)
    parser.add_argument('--cafe_beta', type=float, required=True,
                        help='beta for hybrid inference timing')
    parser.add_argument('--cafe_lam', type=float, required=True,
                        help='lambda_mobility for hybrid inference timing')
    parser.add_argument('--rec_k', type=int, default=100)
    parser.add_argument('--out_json', type=str, required=True)
    return parser.parse_args()


def main():
    args = argument()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)

    env = {
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'torch': torch.__version__,
        'python': platform.python_version(),
        'hostname': platform.node(),
    }
    print(f"Environment: {env}")

    all_rows = []
    per_run = []

    for run in range(args.runs):
        print(f"\n{'=' * 80}\nRUN {run + 1}/{args.runs}\n{'=' * 80}")
        timings = {}

        dataset, split_edge, data, degrees = load_dataset(args)
        num_nodes = dataset._data.num_nodes if hasattr(dataset, '_data') else dataset.data.num_nodes
        train_edges = split_edge['train']['edge'].to(device)
        pos_test_edge = split_edge['test']['edge'].to(device)
        num_train_edges = train_edges.shape[0]

        # --- Preprocessing costs ---
        with Timer() as t:
            coreness = compute_coreness(train_edges, num_nodes).to(device)
        timings['coreness_time'] = t.elapsed
        data.coreness = coreness
        c_np = coreness.cpu().numpy()

        with Timer() as t:
            adj_sp = build_train_adj_csr(train_edges, num_nodes)
            ra_sparse = build_ra_sparse(adj_sp)
        timings['ra_time'] = t.elapsed

        with Timer() as t:
            _ = apply_coreness_inverse_aggregation(
                data.adj_t, coreness, alpha=args.cafe_pre_alpha)
        timings['adj_mod_time'] = t.elapsed

        eval_nodes = get_eval_nodes(pos_test_edge)

        # --- Training costs (each to early stopping; epochs also recorded) ---
        print("\n[vanilla HL-GNN]")
        vanilla = train_cafe_model(data, split_edge, args, degrees, coreness,
                                   device, mode='none')
        timings['train_vanilla'] = vanilla['train_time']
        timings['epochs_vanilla'] = vanilla['best_epoch']

        # --- Inference costs (measured with the vanilla model so the GNN
        #     forward cost is identical; difference isolates the CAFE overhead) ---
        mult = mobility_multiplier_vector(c_np, args.cafe_lam, args.coreness_alpha)

        with Timer() as t:
            _ = build_cafe_recs(vanilla['h_emb'], vanilla['score_fn'], eval_nodes,
                                num_nodes, device, adj_sp, beta=0.0,
                                k=args.rec_k, verbose=False)
        timings['infer_vanilla'] = t.elapsed

        with Timer() as t:
            _ = build_cafe_recs(vanilla['h_emb'], vanilla['score_fn'], eval_nodes,
                                num_nodes, device, adj_sp, ra_sparse=ra_sparse,
                                beta=args.cafe_beta, mult_vec=mult,
                                k=args.rec_k, verbose=False)
        timings['infer_cafe'] = t.elapsed

        del vanilla
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print(f"\n[CAFE-In] lambda={args.cafe_in_lambda}")
        cafe_in = train_cafe_model(data, split_edge, args, degrees, coreness,
                                   device, mode='inproc',
                                   pen_val=args.cafe_in_lambda)
        timings['train_cafe_in'] = cafe_in['train_time']
        timings['epochs_cafe_in'] = cafe_in['best_epoch']
        del cafe_in
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print(f"\n[CAFE-Pre] alpha={args.cafe_pre_alpha}")
        cafe_pre = train_cafe_model(data, split_edge, args, degrees, coreness,
                                    device, mode='preproc',
                                    pen_val=args.cafe_pre_alpha)
        timings['train_cafe_pre'] = cafe_pre['train_time']
        timings['epochs_cafe_pre'] = cafe_pre['best_epoch']
        del cafe_pre
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Derived: per-epoch normalization (removes early-stopping noise)
        for variant in ['vanilla', 'cafe_in', 'cafe_pre']:
            ep = max(int(timings[f'epochs_{variant}']), 1)
            timings[f'per_epoch_{variant}'] = timings[f'train_{variant}'] / ep

        timings['num_nodes'] = num_nodes
        timings['num_train_edges'] = int(num_train_edges)
        timings['num_eval_nodes'] = len(eval_nodes)
        per_run.append(timings)

        for phase, val in timings.items():
            all_rows.append({'dataset': args.data_name, 'run': run,
                             'phase': phase, 'value': float(val)})

        print(f"\nRun {run + 1} timings: "
              f"{json.dumps({k: round(float(v), 2) for k, v in timings.items()}, indent=2)}")

    # --- Aggregate & save ---
    agg = {}
    for phase in per_run[0]:
        vals = [float(r[phase]) for r in per_run]
        agg[phase] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                      'values': vals}

    output = {
        'experiment': 'runtime_benchmark',
        'dataset': args.data_name,
        'environment': env,
        'config': {
            'cafe_in_lambda': args.cafe_in_lambda,
            'cafe_pre_alpha': args.cafe_pre_alpha,
            'cafe_beta': args.cafe_beta,
            'cafe_lam': args.cafe_lam,
            'rec_k': args.rec_k,
            'runs': args.runs,
            'epochs_max': args.epochs,
            'patience': args.patience,
        },
        'timings': agg,
    }
    with open(args.out_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON saved to {args.out_json}")

    csv_path = os.path.splitext(args.out_json)[0] + '.csv'
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"CSV saved to {csv_path}")

    # Console summary for the paper table
    print("\n" + "=" * 70)
    print(f"RUNTIME SUMMARY: {args.data_name} "
          f"(N={agg['num_nodes']['mean']:.0f}, |E|={agg['num_train_edges']['mean']:.0f})")
    print("=" * 70)
    print(f"{'phase':<22}{'mean (s)':>12}{'std':>10}")
    for phase in ['coreness_time', 'ra_time', 'adj_mod_time',
                  'train_vanilla', 'train_cafe_in', 'train_cafe_pre',
                  'per_epoch_vanilla', 'per_epoch_cafe_in', 'per_epoch_cafe_pre',
                  'infer_vanilla', 'infer_cafe']:
        print(f"{phase:<22}{agg[phase]['mean']:>12.2f}{agg[phase]['std']:>10.2f}")
    ratio_in = agg['train_cafe_in']['mean'] / max(agg['train_vanilla']['mean'], 1e-9)
    ratio_pre = agg['train_cafe_pre']['mean'] / max(agg['train_vanilla']['mean'], 1e-9)
    ratio_inf = agg['infer_cafe']['mean'] / max(agg['infer_vanilla']['mean'], 1e-9)
    print(f"\nTraining overhead: CAFE-In {ratio_in:.2f}x, CAFE-Pre {ratio_pre:.2f}x vanilla")
    print(f"Inference overhead (hybrid scoring): {ratio_inf:.2f}x vanilla")


if __name__ == '__main__':
    main()
