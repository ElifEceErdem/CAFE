"""Per-bin perturbation size for the source-coreness strata (Section 7.2).

Reviewer objection: the stratified Degree-EG gradient across source-coreness
bins could be a cross-bin *perturbation-size* gradient, because the EG of bin
``g`` is computed on ``G_hat_g = G_train union Rec(B_g)`` and the bins hold
unequal numbers of evaluated users (roughly a factor of two on four of the five
networks, 7.7x on Brightkite).

This script answers it from the artifacts that already back Section 7.2 --
``analysis/plots/stratified_cafe/`` -- with no retraining and no rescoring:

  * per-bin evaluated-user counts |B_g| and issued recommendations K|B_g|,
    read from the per-node stratified files (authoritative: the campaign's
    ``summary_precision_by_bin_*.csv`` misaligns ``n_nodes`` with
    ``coreness_bin``, so it is deliberately not used), together with the
    bins' mean node degree;
  * whether |B_g| is monotone in coreness (Spearman rho, sign changes);
  * Spearman rho of EG against coreness rank versus against |Rec(B_g)|,
    plus the partial Spearman of EG against coreness rank controlling for
    bin size and for mean bin degree (the degree control answers the
    objection that the equality gradient is a degree gradient);
  * size-matched bin pairs -- pairs whose perturbations differ by at most
    ``--size_tol`` -- and the EG gap between them.

``|Rec(B_g)| = K|B_g|`` counts the recommendation events bin ``g`` issues. The
distinct undirected edges it adds to ``G_hat_g`` is this minus the within-bin
reciprocal pairs (u recommends v and v recommends u, both in bin g); every
recommended pair is otherwise novel with respect to ``G_train``, which
``validate_recommendations`` checks per run. K|B_g| is therefore an exact upper
bound, and the exact deduplicated count can be recovered with ``--npz_dir``
when a campaign was run with ``--save_recommendations True``.

Outputs (to --out_dir):
  bin_perturbation_summary.csv   per dataset/bin: size, K|B_g|, coreness range,
                                 mean degree, EG
  bin_size_monotonicity.csv      per dataset/model: rho(size), rho(EG,.), the
                                 partial Spearman of EG with coreness rank
                                 controlling for bin size and for mean bin
                                 degree, pairs
  bin_size_matched_pairs.csv     every size-matched bin pair and its EG gap
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

STRATIFIED_DIR = 'analysis/plots/stratified_cafe'

METHOD_KEYS = {
    'targets_ra': 'RA',
    'targets_hl_gnn': 'HL-GNN',
    'targets_fusion': 'Fusion',
    'targets_cafe_in': 'CAFE-In',
    'targets_cafe_pre': 'CAFE-Pre',
}


def partial_spearman(x, y, z):
    """Partial Spearman of x and y controlling for z (Pearson on ranks)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)

    def pearson(a, b):
        a = a - a.mean()
        b = b - b.mean()
        denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
        return float(np.sum(a * b) / denom) if denom > 0 else 0.0

    r_xy = pearson(rx, ry)
    r_xz = pearson(rx, rz)
    r_yz = pearson(ry, rz)
    denom = np.sqrt((1.0 - r_xz ** 2) * (1.0 - r_yz ** 2))
    return float((r_xy - r_xz * r_yz) / denom) if denom > 0 else np.nan


def argument():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stratified_dir', default=STRATIFIED_DIR,
                        help='directory holding stratified_cafe_*.csv and '
                             'eg_by_coreness_cafe_*.csv')
    parser.add_argument('--npz_dir', default=None,
                        help='optional campaign root with '
                             '<dataset>/seed_*/recommendations_*.npz; enables '
                             'exact deduplicated |Rec(B_g)|')
    parser.add_argument('--out_dir', default='analysis/plots/bin_perturbation')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--focus_model', default='CAFE-In')
    parser.add_argument('--size_tol', type=float, default=0.05,
                        help='max relative |Rec| difference for a size-matched pair')
    return parser.parse_args()


def load_bins(stratified_dir):
    """Per-bin sizes and coreness ranges from the per-node stratified files."""
    bins = {}
    paths = sorted(glob.glob(os.path.join(stratified_dir, 'stratified_cafe_*.csv')))
    if not paths:
        raise FileNotFoundError(f"No stratified_cafe_*.csv under {stratified_dir}")
    for path in paths:
        dataset = os.path.basename(path)[len('stratified_cafe_'):-len('.csv')]
        frame = pd.read_csv(path, usecols=['node_id', 'model', 'run', 'degree',
                                           'coreness_bin', 'coreness_bin_label',
                                           'node_coreness'])
        # One model and one run is enough: the split is fixed, so the binning
        # is shared. Assert that rather than assume it.
        counts = (frame.groupby(['model', 'run'])['coreness_bin']
                       .apply(lambda s: tuple(s.value_counts().sort_index())))
        if counts.nunique() != 1:
            raise ValueError(f"{dataset}: bin sizes differ across model/run")
        first = frame[(frame['model'] == frame['model'].iloc[0])
                      & (frame['run'] == frame['run'].iloc[0])]
        grouped = (first.groupby(['coreness_bin', 'coreness_bin_label'])
                        .agg(n_users=('node_id', 'nunique'),
                             coreness_min=('node_coreness', 'min'),
                             coreness_max=('node_coreness', 'max'),
                             degree_mean=('degree', 'mean'))
                        .reset_index()
                        .sort_values('coreness_bin')
                        .reset_index(drop=True))
        lo = grouped['coreness_min'].to_numpy()
        hi = grouped['coreness_max'].to_numpy()
        if not np.all(lo[1:] > hi[:-1]):
            raise ValueError(f"{dataset}: coreness_bin is not in coreness order")
        bins[dataset] = grouped
    return bins


def load_eg(stratified_dir, dataset):
    path = os.path.join(stratified_dir, f'eg_by_coreness_cafe_{dataset}.csv')
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def exact_rec_edges(npz_dir, dataset, bin_of_node, method):
    """Distinct undirected edges added per bin, from saved recommendation lists."""
    paths = sorted(glob.glob(os.path.join(
        npz_dir, dataset, 'seed_*', 'recommendations_*.npz')))
    if not paths:
        return None
    key = [k for k, v in METHOD_KEYS.items() if v == method]
    if not key:
        return None
    per_run = {}
    for path in paths:
        payload = np.load(path)
        meta = json.loads(str(payload['metadata_json']))
        if not meta.get('recommendation_validation_passed', False):
            raise ValueError(f"{path}: recommendations failed validation")
        num_nodes = int(meta['num_nodes'])
        sources = payload['sources'].astype(np.int64)
        targets = payload[key[0]].astype(np.int64)
        source_bin = np.array([bin_of_node[int(u)] for u in sources])
        for b in np.unique(source_bin):
            mask = source_bin == b
            src = np.repeat(sources[mask], targets.shape[1])
            dst = targets[mask].reshape(-1)
            keys = (np.minimum(src, dst) * np.int64(num_nodes)
                    + np.maximum(src, dst))
            per_run.setdefault(int(b), []).append(int(np.unique(keys).size))
    return {b: float(np.mean(v)) for b, v in per_run.items()}


def main():
    args = argument()
    os.makedirs(args.out_dir, exist_ok=True)
    bins = load_bins(args.stratified_dir)

    summary_rows, mono_rows, pair_rows = [], [], []

    for dataset, grouped in bins.items():
        sizes = grouped['n_users'].to_numpy(float)
        rec = sizes * args.k
        degrees = grouped['degree_mean'].to_numpy(float)
        labels = grouped['coreness_bin_label'].tolist()
        rank = np.arange(len(sizes), dtype=float)

        exact = None
        if args.npz_dir:
            node_bin = {}
            frame = pd.read_csv(
                os.path.join(args.stratified_dir,
                             f'stratified_cafe_{dataset}.csv'),
                usecols=['node_id', 'coreness_bin'])
            for nid, b in frame.drop_duplicates('node_id').itertuples(index=False):
                node_bin[int(nid)] = int(b)
            exact = exact_rec_edges(args.npz_dir, dataset, node_bin,
                                    args.focus_model)

        eg = load_eg(args.stratified_dir, dataset)
        eg_by_model = {}
        if eg is not None:
            for model, g in eg.groupby('model'):
                g = g.sort_values('coreness_bin')
                if len(g) != len(sizes):
                    raise ValueError(f"{dataset}/{model}: bin count mismatch")
                eg_by_model[model] = g['equality_gain'].to_numpy(float)

        for i, row in grouped.iterrows():
            entry = {
                'dataset': dataset,
                'coreness_bin': int(row['coreness_bin']),
                'coreness_bin_label': row['coreness_bin_label'],
                'coreness_min': int(row['coreness_min']),
                'coreness_max': int(row['coreness_max']),
                'n_users': int(row['n_users']),
                'rec_events': int(rec[i]),
                'degree_mean': float(row['degree_mean']),
            }
            if exact is not None:
                entry['rec_edges_unique'] = exact.get(int(row['coreness_bin']))
            for model, values in eg_by_model.items():
                entry[f'eg_{model}'] = values[i]
            summary_rows.append(entry)

        rho_size, p_size = spearmanr(rank, sizes)
        diffs = np.diff(sizes)
        base = {
            'dataset': dataset,
            'n_bins': len(sizes),
            'eval_users': int(sizes.sum()),
            'size_min': int(sizes.min()),
            'size_max': int(sizes.max()),
            'size_max_over_min': sizes.max() / sizes.min(),
            'rec_min': int(rec.min()),
            'rec_max': int(rec.max()),
            'rho_size_vs_coreness': rho_size,
            'p_size_vs_coreness': p_size,
            'size_strictly_increasing': bool(np.all(diffs > 0)),
            'size_strictly_decreasing': bool(np.all(diffs < 0)),
            'size_sign_changes': int(np.sum(np.sign(diffs[:-1])
                                            != np.sign(diffs[1:]))),
        }
        for model, values in eg_by_model.items():
            rho_rank, _ = spearmanr(rank, values)
            rho_sz, _ = spearmanr(sizes, values)
            mono_rows.append(dict(
                base, model=model,
                rho_eg_vs_coreness=rho_rank,
                rho_eg_vs_size=rho_sz,
                rho_eg_vs_coreness_partial_size=partial_spearman(
                    rank, values, sizes),
                rho_eg_vs_coreness_partial_degree=partial_spearman(
                    rank, values, degrees),
                eg_strictly_decreasing=bool(np.all(np.diff(values) < 0)),
                eg_upward_steps=int(np.sum(np.diff(values) > 0)),
                eg_min=values.min(), eg_max=values.max(),
                eg_span=values.max() - values.min(),
            ))

        # Size-matched pairs: comparable perturbation, different coreness.
        focus = eg_by_model.get(args.focus_model)
        if focus is not None:
            for i in range(len(sizes)):
                for j in range(i + 1, len(sizes)):
                    rel = abs(rec[i] - rec[j]) / max(rec[i], rec[j])
                    if rel <= args.size_tol:
                        pair_rows.append({
                            'dataset': dataset,
                            'model': args.focus_model,
                            'bin_a': i, 'label_a': labels[i],
                            'bin_b': j, 'label_b': labels[j],
                            'rec_a': int(rec[i]), 'rec_b': int(rec[j]),
                            'rec_rel_diff': rel,
                            'eg_a': focus[i], 'eg_b': focus[j],
                            'eg_gap': abs(focus[i] - focus[j]),
                        })

    summary = pd.DataFrame(summary_rows)
    mono = pd.DataFrame(mono_rows)
    pairs = pd.DataFrame(pair_rows)
    summary.to_csv(os.path.join(args.out_dir, 'bin_perturbation_summary.csv'),
                   index=False)
    mono.to_csv(os.path.join(args.out_dir, 'bin_size_monotonicity.csv'),
                index=False)
    pairs.to_csv(os.path.join(args.out_dir, 'bin_size_matched_pairs.csv'),
                 index=False)

    with pd.option_context('display.width', 220, 'display.max_rows', 300):
        cols = ['dataset', 'n_bins', 'eval_users', 'size_min', 'size_max',
                'size_max_over_min', 'rec_min', 'rec_max',
                'rho_size_vs_coreness', 'size_strictly_increasing',
                'size_strictly_decreasing', 'size_sign_changes']
        print("\n=== perturbation size per bin ===")
        print(mono.drop_duplicates('dataset')[cols].round(3).to_string(index=False))
        print(f"\n=== EG gradient vs size ({args.focus_model}) ===")
        focus = mono[mono['model'] == args.focus_model]
        print(focus[['dataset', 'rho_eg_vs_coreness', 'rho_eg_vs_size',
                     'rho_eg_vs_coreness_partial_size',
                     'rho_eg_vs_coreness_partial_degree',
                     'rho_size_vs_coreness', 'eg_strictly_decreasing',
                     'eg_upward_steps', 'eg_span']].round(3).to_string(index=False))
        if len(pairs):
            print(f"\n=== widest size-matched pair per dataset "
                  f"(<= {args.size_tol:.0%} apart in |Rec|) ===")
            widest = pairs.loc[pairs.groupby('dataset')['eg_gap'].idxmax()]
            print(widest[['dataset', 'label_a', 'label_b', 'rec_a', 'rec_b',
                          'rec_rel_diff', 'eg_a', 'eg_b',
                          'eg_gap']].round(3).to_string(index=False))

    print(f"\nWrote 3 CSVs to {args.out_dir}")


if __name__ == '__main__':
    main()
