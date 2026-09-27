"""
Baseline experiment runner for the paper's comparison tables (Tables 4 & 5).

Runs the non-CAFE methods through baselines.BaselineWrapper and writes
result JSONs in the directory layout expected by
experiments/result_extraction.py:

  results/<dataset>/baselines/          random, PA, CN, AA, RA   (one file, keyed by method)
  results/<dataset>/lightgcn/           LightGCN
  results/<dataset>/ssnet/              SSNet
  results/<dataset>/ltlp/               LTLP
  results/<dataset>/tailsteak/          TailSteak
  results/<dataset>/cross_link_bias/    Cross-Link Bias
  results/<dataset>/fairadj/            FairAdj   (attribute datasets only)
  results/<dataset>/lba/                LBA       (attribute datasets only)
  results/<dataset>/moral/              MORAL     (attribute datasets only)

Attribute-dependent hyperparameters follow Table 8 of the supplementary
material (identity features, official-implementation values). Trained
baselines run for --epochs (default 300); each method keeps its own default
learning rate (MORAL: 3e-4 per Table 8).

Usage:
    python run_baselines.py --data_name oklahoma --data_path ../data/SNAP \
        --methods random,pa,cn,aa,ra --runs 10 \
        --results_dir ../results
"""

import os
import sys
import gc
import json
import argparse
import collections
import traceback
from datetime import datetime

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
snap_dir = os.path.abspath(os.path.join(current_dir, '../backbone'))
fairness_dir = os.path.abspath(os.path.join(current_dir, '../fairness_methods'))
for path in [parent_dir, snap_dir, fairness_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from core import set_seed, str2bool, load_dataset, compute_statistics, create_experiment_info  # noqa: E402
from core.data_loader import load_sensitive_attr  # noqa: E402
from core.statistics import _serialize_stats  # noqa: E402
from baselines.baseline_wrapper import BaselineWrapper  # noqa: E402


# method key -> (output dirname, JSON/table key, wrapper baseline_type,
#                needs sensitive attributes, wrapper kwargs)
METHOD_REGISTRY = {
    'random':          ('baselines', 'random', 'random', False, {}),
    'pa':              ('baselines', 'PA', 'preferential_attachment', False, {}),
    'cn':              ('baselines', 'CN', 'common_neighbors', False, {}),
    'aa':              ('baselines', 'AA', 'adamic_adar', False, {}),
    'ra':              ('baselines', 'RA', 'resource_allocation', False, {}),
    'lightgcn':        ('lightgcn', 'LightGCN', 'lightgcn', False,
                        {'embedding_dim': 64, 'gnn_num_layers': 2}),
    'ssnet':           ('ssnet', 'SSNet', 'ssnet', False,
                        {'gnn_hidden_channels': 64, 'gnn_num_layers': 1}),
    'ltlp':            ('ltlp', 'LTLP', 'ltlp', False,
                        {'ltlp_backbone': 'pure_gnn', 'gnn_hidden_channels': 256,
                         'gnn_num_layers': 2, 'dropout': 0.3}),
    'tailsteak':       ('tailsteak', 'TailSteak', 'tailsteak', False,
                        {'embedding_dim': 128, 'gnn_hidden_channels': 32,
                         'gnn_num_layers': 1, 'heads': 4}),
    'cross_link_bias': ('cross_link_bias', 'Cross-Link Bias', 'cross_link_bias', False,
                        {'embedding_dim': 64, 'gnn_num_layers': 2}),
    # Attribute-dependent baselines -- hyperparameters per supp. Table 8
    'fairadj':         ('fairadj', 'FairAdj', 'fairadj', True,
                        {'gnn_hidden_channels': 32, 'embedding_dim': 16,
                         'dropout': 0.0, 'fairadj_T1': 50, 'fairadj_T2': 20,
                         'fairadj_eta': 0.2, 'fairadj_outer_epochs': 4,
                         'use_node_feats': False}),
    'lba':             ('lba', 'LBA', 'lba', True,
                        {'gnn_hidden_channels': 128, 'embedding_dim': 128,
                         'gnn_num_layers': 4, 'conv_type': 'sym',
                         'reg_weight': 1.0, 'use_node_feats': False}),
    'moral':           ('moral', 'MORAL', 'moral', True,
                        {'gnn_hidden_channels': 128, 'lr': 3e-4,
                         'weight_decay': 0.0, 'encoder': 'gcn'}),
}


def argument():
    parser = argparse.ArgumentParser(description='Baseline runner (Tables 4 & 5)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--data_name', type=str, default='ego-facebook')
    parser.add_argument('--data_path', type=str, default='../data/SNAP')
    parser.add_argument('--val_ratio', type=float, default=0.05)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--split_file', type=str, default=None,
                        help='Pre-computed split (brightkite temporal split)')
    parser.add_argument('--encoder', type=str, default='HLGNN',
                        help='Only used by load_dataset for adjacency normalization; '
                             'HLGNN keeps adj_t raw (correct for these baselines)')

    parser.add_argument('--methods', type=str, required=True,
                        help=f"Comma-separated subset of: {','.join(METHOD_REGISTRY)}")
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=16384)
    parser.add_argument('--moral_epochs', type=int, default=300)
    parser.add_argument('--moral_batch_size', type=int, default=1024)

    parser.add_argument('--fairness_mode', type=str, default='indegree')
    parser.add_argument('--centrality_types', type=str, default='pagerank,closeness')
    parser.add_argument('--use_equality_gain', type=str2bool, default=True)
    parser.add_argument('--use_centrality_gain', type=str2bool, default=True)

    parser.add_argument('--results_dir', type=str, default='../results')
    parser.add_argument('--acceptance_probe_jsonl', default=None,
                        help='Optional scalar-only RA acceptance pilot output; no predictions saved')
    return parser.parse_args()


def run_one_method(method, data, split_edge, num_nodes, sens_attr, args, run_seed):
    dirname, key, baseline_type, needs_sens, kwargs = METHOD_REGISTRY[method]
    device_str = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    # Learning rate is deliberately NOT injected: each method uses its own
    # wrapper/class default (MORAL's 3e-4 is pinned in the registry).
    wrapper_kwargs = dict(kwargs)
    wrapper_kwargs.setdefault('batch_size', args.batch_size)
    wrapper_kwargs['seed'] = run_seed  # only consumed by the random baseline

    epochs = args.moral_epochs if method == 'moral' else args.epochs
    batch_size = args.moral_batch_size if method == 'moral' else args.batch_size

    wrapper = BaselineWrapper(
        baseline_type=baseline_type,
        num_nodes=num_nodes,
        device=device_str,
        **wrapper_kwargs,
    )
    wrapper.fit(data, split_edge, epochs=epochs, batch_size=batch_size,
                sens_attr=sens_attr if needs_sens else None)

    acceptance_probe = None
    if args.acceptance_probe_jsonl and method == 'ra':
        from core.acceptance_probe import record_acceptance
        def acceptance_probe(recs, train_edges, test_edges, n, metrics):
            record_acceptance(
                args.acceptance_probe_jsonl, dataset=args.data_name, method='RA',
                run=run_seed - args.seed, config='resource_allocation', recs=recs,
                train_edges=train_edges, test_edges=test_edges, num_nodes=n,
                main_metrics=metrics)

    res = wrapper.test(
        data, split_edge, batch_size=batch_size, evaluator=None,
        eval_metric='all', fairness_mode=args.fairness_mode,
        use_equality_gain=args.use_equality_gain,
        use_centrality_gain=args.use_centrality_gain,
        centrality_types=[x.strip() for x in args.centrality_types.split(',')],
        acceptance_probe=acceptance_probe,
    )

    del wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def save_results(per_method_runs, args, ts):
    """Write one JSON per output directory from whatever has finished so far.

    Called after every run, not just at the end: a method that crashes (or a
    wall-clock kill) used to discard every seed the task had already completed,
    including those of the methods that ran fine.
    """
    by_dir = collections.defaultdict(dict)
    runs_by_dir = collections.defaultdict(dict)
    for method, run_list in per_method_runs.items():
        if not run_list:
            continue
        dirname, key = METHOD_REGISTRY[method][0], METHOD_REGISTRY[method][1]
        by_dir[dirname][key] = _serialize_stats(compute_statistics(run_list))
        runs_by_dir[dirname][key] = len(run_list)

    for dirname, results in by_dir.items():
        out_dir = os.path.join(args.results_dir, args.data_name, dirname)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'results_{dirname}_{args.data_name}_{ts}.json')
        payload = {
            'experiment_info': create_experiment_info(args, f'baseline_{dirname}'),
            # experiment_info['num_runs'] is the *requested* count; this records
            # what each method actually completed, so a method that died partway
            # cannot be mistaken for a full 10-seed result.
            'runs_completed': runs_by_dir[dirname],
            'results': results,
        }
        # Write-then-rename: a kill during the dump must not leave the previous
        # (valid) JSON truncated. The .tmp suffix keeps it out of the *.json
        # globs in experiments/result_extraction.py.
        tmp_path = out_path + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, out_path)
        print(f"Saved {list(results)} -> {out_path}")


def main():
    args = argument()
    methods = [m.strip().lower() for m in args.methods.split(',')]
    unknown = [m for m in methods if m not in METHOD_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; valid: {list(METHOD_REGISTRY)}")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 80)
    print(f"BASELINE RUNNER | dataset={args.data_name} | methods={methods} | runs={args.runs}")
    print("=" * 80)

    # method key -> list of per-run metric dicts
    per_method_runs = collections.defaultdict(list)
    # method key -> error string, for methods dropped mid-task
    failures = {}

    for run in range(args.runs):
        run_seed = args.seed + run
        print(f"\n{'=' * 80}\nRUN {run + 1}/{args.runs} (seed {run_seed})\n{'=' * 80}")
        set_seed(run_seed)

        # Split is fixed (seed 42 inside get_edge_split / temporal file), matching
        # the CAFE runners; only training/sampling randomness varies across runs.
        dataset, split_edge, data, degrees = load_dataset(args)
        num_nodes = dataset._data.num_nodes if hasattr(dataset, '_data') else dataset.data.num_nodes

        sens_attr = None
        try:
            sens_attr = load_sensitive_attr(args.data_name, num_nodes, root=args.data_path)
            data.sens = sens_attr if torch.is_tensor(sens_attr) else torch.tensor(sens_attr)
            data.sens = data.sens.to(data.edge_index.device)
            print(f"Loaded sensitive attributes "
                  f"(g0={int((data.sens == 0).sum())}, g1={int((data.sens == 1).sum())})")
        except ValueError as e:
            print(f"NOTE: {e} -- demographic metrics skipped.")

        for method in list(methods):
            needs_sens = METHOD_REGISTRY[method][3]
            if needs_sens and sens_attr is None:
                print(f"SKIP {method}: requires sensitive attributes, "
                      f"none available for {args.data_name}")
                continue
            print(f"\n----- [{method}] run {run + 1} -----")
            try:
                res = run_one_method(method, data, split_edge, num_nodes,
                                     sens_attr, args, run_seed)
            except Exception as exc:
                # Drop this method, keep the seeds it already finished, and let
                # the rest of the task continue. Previously one failure aborted
                # the whole array task before anything was written to disk.
                print(f"FAILED [{method}] run {run + 1}: "
                      f"{type(exc).__name__}: {exc}")
                traceback.print_exc()
                failures[method] = f"run {run + 1}: {type(exc).__name__}: {exc}"
                methods.remove(method)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            per_method_runs[method].append(res)

        save_results(per_method_runs, args, ts)

        # Heuristics are deterministic on the fixed split: one run is enough.
        deterministic = {'pa', 'cn', 'aa', 'ra'}
        methods = [m for m in methods if not (m in deterministic and run == 0)]
        if not methods:
            print("\nNo methods left to run (deterministic or failed) -- stopping.")
            break

        del dataset, split_edge, data, degrees
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_results(per_method_runs, args, ts)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for method in [m.strip().lower() for m in args.methods.split(',')]:
        n = len(per_method_runs.get(method, []))
        note = f"  <-- FAILED: {failures[method]}" if method in failures else ""
        print(f"  {method:16s} {n}/{args.runs} runs completed{note}")

    if failures:
        print(f"\n{len(failures)} method(s) did not complete; "
              f"partial results were still written.")
        return 1
    print("\nAll baselines complete.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
