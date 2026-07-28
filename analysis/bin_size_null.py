"""Size-matched null for the source-coreness stratified Degree EG.

Reviewer objection (Section 7.2): the stratified Degree-EG gradient across
source-coreness bins could be a cross-bin *perturbation-size* gradient, because
the EG of bin ``g`` is computed on ``G_hat_g = G_train union Rec(B_g)`` and the
bins hold unequal numbers of users.

This script answers the objection directly rather than by proxy. For every
dataset it

  1. rebuilds the fixed split used by the stratified campaign and asserts that
     its fingerprint matches the one recorded in the run metadata;
  2. recomputes the published per-bin EG from the saved recommendation lists
     and asserts agreement, so the null below is measured on the same scale;
  3. reports EG for an empty recommendation set -- the zero-perturbation
     reference every bin's EG converges to as |Rec(B_g)| -> 0;
  4. draws, for each bin, ``--draws`` random source sets of exactly the same
     size from the same evaluation population, ignoring coreness, and reports
     the resulting EG distribution.

The null isolates size: if the coreness gradient were a size artifact, each
observed EG_g would sit inside its own size-matched null band. Nothing is
retrained and no model is rescored.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (_parent, os.path.join(_parent, 'backbone')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from core.data_loader import load_dataset  # noqa: E402
from metrics.centrality_equality_gain import compute_gini_coefficient  # noqa: E402
from metrics.graph_utils import (  # noqa: E402
    build_control_edge_set,
    canonicalize_undirected_edges,
    degree_from_simple_undirected_edges,
)

STRATIFIED = 'results_egfix/20260724_degree_eg_sum_in_v1/todo3/stratified'
FINALIZED = 'results_egfix/20260724_degree_eg_sum_in_v1/todo3/finalized'

METHOD_KEYS = {
    'targets_ra': 'RA',
    'targets_hl_gnn': 'HL-GNN',
    'targets_fusion': 'Fusion',
    'targets_cafe_in': 'CAFE-In',
    'targets_cafe_pre': 'CAFE-Pre',
}


class Args:
    """Minimal namespace for core.data_loader.load_dataset."""

    def __init__(self, data_name, data_path):
        self.data_name = data_name
        self.data_path = data_path
        self.val_ratio = 0.05
        self.test_ratio = 0.1
        self.data_split_seed = 42
        self.split_file = None
        # Only the split is used here; the encoder choice merely selects which
        # normalized adjacency load_dataset attaches to `data`.
        self.encoder = 'HLGNN'
        self.device = 'cpu'


def argument():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_path', default='data/SNAP')
    parser.add_argument('--stratified', default=STRATIFIED)
    parser.add_argument('--finalized', default=FINALIZED)
    parser.add_argument('--out_dir', default='analysis/plots/bin_perturbation')
    parser.add_argument('--draws', type=int, default=200,
                        help='random size-matched source sets per bin')
    parser.add_argument('--method', default='CAFE-In',
                        help='method whose recommendation lists define the null')
    parser.add_argument('--datasets', default=None,
                        help='comma-separated subset; default: all found')
    parser.add_argument('--seed', type=int, default=20260727)
    return parser.parse_args()


def split_fingerprint(split_edge, num_nodes):
    """Byte-identical to analysis/run_stratified_cafe.py::_split_fingerprint."""
    import hashlib
    digest = hashlib.sha256()
    digest.update(f"num_nodes={int(num_nodes)}\n".encode())
    for name in ('train', 'test'):
        edges = canonicalize_undirected_edges(
            split_edge[name]['edge'].cpu(), num_nodes)
        digest.update(f"{name}:{len(edges)}\n".encode())
        digest.update(
            edges.to(dtype=torch.int64).contiguous().numpy()
            .astype('<i8', copy=False).tobytes())
    return digest.hexdigest()


class EGCalculator:
    """Degree EG for an arbitrary set of recommendation sources.

    Mirrors metrics.centrality_equality_gain.
    compute_degree_equality_gain_by_source_bins_for_models: the control graph,
    canonical training graph, and training degrees are built once; per source
    set only the novel recommendation edges are added.
    """

    def __init__(self, train_edges, test_edges, num_nodes):
        self.num_nodes = int(num_nodes)
        train_simple = canonicalize_undirected_edges(train_edges, num_nodes)
        self.train_degree = degree_from_simple_undirected_edges(
            train_simple, num_nodes)
        control_simple = build_control_edge_set(
            train_simple, test_edges, num_nodes)
        self.gini_control = compute_gini_coefficient(
            degree_from_simple_undirected_edges(control_simple, num_nodes))
        # Sorted keys of the observed training pairs, for novelty filtering.
        tr = train_simple.numpy().astype(np.int64)
        self.train_keys = np.unique(tr[:, 0] * self.num_nodes + tr[:, 1])

    def eg(self, src, dst):
        """src, dst: flat int64 arrays of directed recommendation events."""
        degree = self.train_degree.clone()
        if src.size:
            lo = np.minimum(src, dst)
            hi = np.maximum(src, dst)
            keys = np.unique(lo * self.num_nodes + hi)
            novel = keys[~np.isin(keys, self.train_keys, assume_unique=True)]
            if novel.size:
                endpoints = np.concatenate(
                    [novel // self.num_nodes, novel % self.num_nodes])
                degree += torch.bincount(
                    torch.from_numpy(endpoints), minlength=self.num_nodes
                ).to(dtype=torch.float32)
        gini_pred = compute_gini_coefficient(degree)
        if self.gini_control == 0:
            return 0.0
        return (self.gini_control - gini_pred) / self.gini_control * 100.0


def main():
    args = argument()
    os.makedirs(args.out_dir, exist_ok=True)

    published = pd.read_csv(
        os.path.join(args.finalized, 'stratified_degree_eg_summary.csv'))
    raw = pd.read_csv(
        os.path.join(args.finalized, 'stratified_precision_raw.csv'),
        usecols=['dataset', 'node_id', 'coreness_bin', 'coreness_bin_label'])
    bin_map = raw.drop_duplicates(['dataset', 'node_id'])

    npz_paths = sorted(glob.glob(
        os.path.join(args.stratified, '*', 'seed_*', 'recommendations_*.npz')))
    by_dataset = {}
    for path in npz_paths:
        meta = json.loads(str(np.load(path)['metadata_json']))
        by_dataset.setdefault(meta['dataset'], []).append((path, meta))

    wanted = (set(args.datasets.split(',')) if args.datasets
              else set(by_dataset))

    obs_rows, null_rows = [], []
    for dataset in sorted(wanted):
        runs = by_dataset[dataset]
        print(f"\n{'=' * 70}\n{dataset}: {len(runs)} runs\n{'=' * 70}")

        ds_args = Args(dataset, args.data_path)
        dataset_obj, split_edge, _, _ = load_dataset(ds_args)
        num_nodes = (dataset_obj._data.num_nodes
                     if hasattr(dataset_obj, '_data')
                     else dataset_obj.data.num_nodes)
        fingerprint = split_fingerprint(split_edge, num_nodes)
        expected = {meta['split_fingerprint'] for _, meta in runs}
        if expected != {fingerprint}:
            raise ValueError(
                f"{dataset}: rebuilt split fingerprint {fingerprint[:12]} does "
                f"not match the campaign's {sorted(e[:12] for e in expected)}")
        print(f"  split fingerprint verified: {fingerprint[:16]}")

        calc = EGCalculator(split_edge['train']['edge'].cpu(),
                            split_edge['test']['edge'].cpu(), num_nodes)
        empty = calc.eg(np.array([], dtype=np.int64),
                        np.array([], dtype=np.int64))
        print(f"  Gini(control)={calc.gini_control:.6f}  "
              f"EG(empty recommendation set)={empty:+.4f}")

        bins = (bin_map.loc[bin_map['dataset'] == dataset]
                       .set_index('node_id')['coreness_bin'])
        labels = (bin_map.loc[bin_map['dataset'] == dataset]
                         .drop_duplicates('coreness_bin')
                         .set_index('coreness_bin')['coreness_bin_label'])

        rng = np.random.default_rng(args.seed)
        key = [k for k, v in METHOD_KEYS.items() if v == args.method][0]

        for path, meta in runs:
            payload = np.load(path)
            sources = payload['sources'].astype(np.int64)
            targets = payload[key].astype(np.int64)
            k = int(meta['k'])
            source_bin = bins.loc[sources].to_numpy()
            seed = int(meta['training_seed'])

            for b in np.unique(source_bin):
                mask = source_bin == b
                src = np.repeat(sources[mask], k)
                dst = targets[mask].reshape(-1)
                observed = calc.eg(src, dst)
                ref = published[(published['dataset'] == dataset)
                                & (published['model'] == args.method)
                                & (published['coreness_bin'] == b)]
                obs_rows.append({
                    'dataset': dataset, 'method': args.method,
                    'coreness_bin': int(b),
                    'coreness_bin_label': labels.loc[b],
                    'training_seed': seed,
                    'n_users': int(mask.sum()),
                    'eg_observed': observed,
                    'eg_published_mean': (float(ref['equality_gain'].iloc[0])
                                          if len(ref) else np.nan),
                    'eg_empty_recset': empty,
                })

                # Size-matched null: same number of sources, coreness ignored.
                draws = np.empty(args.draws, dtype=float)
                n_g = int(mask.sum())
                for d in range(args.draws):
                    pick = rng.choice(sources.size, size=n_g, replace=False)
                    draws[d] = calc.eg(np.repeat(sources[pick], k),
                                       targets[pick].reshape(-1))
                null_rows.append({
                    'dataset': dataset, 'method': args.method,
                    'coreness_bin': int(b),
                    'coreness_bin_label': labels.loc[b],
                    'training_seed': seed,
                    'n_users': n_g,
                    'eg_observed': observed,
                    'null_mean': draws.mean(),
                    'null_std': draws.std(ddof=1),
                    'null_lo': np.percentile(draws, 2.5),
                    'null_hi': np.percentile(draws, 97.5),
                    'null_pct_of_draws_below_observed':
                        float((draws < observed).mean()),
                    'z_vs_null': ((observed - draws.mean()) / draws.std(ddof=1)
                                  if draws.std(ddof=1) > 0 else np.nan),
                    'draws': args.draws,
                })
            print(f"  seed {seed}: {len(np.unique(source_bin))} bins done")

    obs = pd.DataFrame(obs_rows)
    obs['reproduction_error'] = (obs.groupby(
        ['dataset', 'method', 'coreness_bin'])['eg_observed']
        .transform('mean') - obs['eg_published_mean'])
    obs.to_csv(os.path.join(args.out_dir, 'bin_eg_reproduction.csv'),
               index=False)

    null = pd.DataFrame(null_rows)
    null.to_csv(os.path.join(args.out_dir, 'bin_eg_size_matched_null_per_run.csv'),
                index=False)
    null_summary = (null
                    .groupby(['dataset', 'method', 'coreness_bin',
                              'coreness_bin_label', 'n_users'], as_index=False)
                    .agg(eg_observed=('eg_observed', 'mean'),
                         null_mean=('null_mean', 'mean'),
                         null_lo=('null_lo', 'mean'),
                         null_hi=('null_hi', 'mean'),
                         z_vs_null=('z_vs_null', 'mean'),
                         runs=('eg_observed', 'size')))
    null_summary['outside_null_band'] = (
        (null_summary['eg_observed'] < null_summary['null_lo'])
        | (null_summary['eg_observed'] > null_summary['null_hi']))
    null_summary.to_csv(
        os.path.join(args.out_dir, 'bin_eg_size_matched_null.csv'), index=False)

    with pd.option_context('display.width', 220, 'display.max_rows', 200):
        print("\n=== published-EG reproduction (max abs error per dataset) ===")
        print(obs.groupby('dataset')['reproduction_error']
                 .apply(lambda s: s.abs().max()).round(8).to_string())
        print("\n=== size-matched null ===")
        print(null_summary.round(3).to_string(index=False))

    print(f"\nWrote 3 CSVs to {args.out_dir}")


if __name__ == '__main__':
    main()
