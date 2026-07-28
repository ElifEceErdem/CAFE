"""
Experiment (d), analysis stage: sensitivity of the model-selection procedure
to the ASF weights (default 0.5, 0.3, 0.1, 0.1).

Pareto DOMINANCE is weight-independent; what the weights influence is which
point ON the Pareto front is reported per method. This script quantifies
that influence directly:

  1. Re-runs the full config-selection pipeline from generate_final_tables.py
     under (i) a systematic grid over the 4-simplex and (ii) Dirichlet draws
     centred on the default weights.
  2. For each (method, dataset): reports how often the selected config
     changes vs the default, and the min/max range each headline metric
     takes across all weight settings.

If the selected configs are stable and the metric ranges are tight, the
weights are demonstrably not doing hidden work. No GPU needed -- this is a
pure re-analysis of the existing sweep JSONs in results/.

Usage:
    python asf_weight_sensitivity.py --n_dirichlet 200 --out_prefix asf_sensitivity
"""

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_final_tables as gft  # noqa: E402

HEADLINE = ['Precision@10', 'Equality_Gain@10', 'Pagerank_EG@10', 'Closeness_EG@10']

# Methods whose selection procedure we care about (config-swept methods)
CAFE_METHODS = [
    'Hybrid Penalized D (Inproc)',   # CAFE-In
    'Hybrid Penalized D (Preproc)',  # CAFE-Pre
    'Hybrid Penalized C (Inproc)',   # CAFE-In-B
    'Hybrid Penalized C (Preproc)',  # CAFE-Pre-B
]


def simplex_grid(step=0.1, min_w=0.05):
    """All weight vectors on the 4-simplex with the given step, floor min_w."""
    vals = np.arange(min_w, 1.0, step)
    grid = []
    for w in itertools.product(vals, repeat=4):
        s = sum(w)
        if s > 0:
            wn = np.array(w) / s
            if (wn >= min_w - 1e-9).all():
                grid.append(np.round(wn, 4))
    # Deduplicate
    return list({tuple(w): w for w in grid}.values())


def select_with_weights(df, weights):
    """Run gft's selection with a given weight vector; return best-config df."""
    gft.WEIGHTS = np.asarray(weights, dtype=float)
    return gft.select_best_configs(df)


def main():
    parser = argparse.ArgumentParser(description='ASF weight sensitivity (experiment d)')
    parser.add_argument('--base_dir', type=str, default=gft.BASE_DIR)
    parser.add_argument('--grid_step', type=float, default=0.15)
    parser.add_argument('--n_dirichlet', type=int, default=200,
                        help='Dirichlet draws centred on the default weights')
    parser.add_argument('--dirichlet_conc', type=float, default=50.0,
                        help='Concentration (higher = tighter around default)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_prefix', type=str, default='asf_sensitivity')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    default_weights = np.array([0.5, 0.3, 0.1, 0.1])

    print(f'Loading all sweep results from {args.base_dir} ...')
    df = gft.extract_all_results(args.base_dir)
    df = df[df['Method_Group'].isin(CAFE_METHODS) & df['Has_Config']]
    if df.empty:
        raise SystemExit('No config-swept CAFE results found under base_dir.')
    print(f'{len(df)} config rows across '
          f'{df["Dataset"].nunique()} datasets, {df["Method_Group"].nunique()} methods')

    # Weight sets: default + simplex grid + Dirichlet perturbations
    weight_sets = [('default', default_weights)]
    for i, w in enumerate(simplex_grid(step=args.grid_step)):
        weight_sets.append((f'grid_{i}', np.array(w)))
    draws = rng.dirichlet(default_weights * args.dirichlet_conc,
                          size=args.n_dirichlet)
    for i, w in enumerate(draws):
        weight_sets.append((f'dirichlet_{i}', w))
    print(f'Total weight settings evaluated: {len(weight_sets)}')

    records = []
    for name, w in weight_sets:
        best = select_with_weights(df, w)
        for _, row in best.iterrows():
            rec = {
                'weight_set': name,
                'w1': w[0], 'w2': w[1], 'w3': w[2], 'w4': w[3],
                'method': row['Method_Group'],
                'dataset': row['Dataset'],
                'config': row.get('Method_Detail', ''),
            }
            for m in HEADLINE:
                rec[m] = row.get(f'{m}_Mean', np.nan)
            records.append(rec)

    res = pd.DataFrame(records)
    raw_csv = f'{args.out_prefix}_raw.csv'
    res.to_csv(raw_csv, index=False)
    print(f'Raw selections written to {raw_csv}')

    # ------------------------------------------------------------------
    # Stability summary per (method, dataset)
    # ------------------------------------------------------------------
    default_sel = res[res['weight_set'] == 'default'] \
        .set_index(['method', 'dataset'])['config']

    summary_rows = []
    for (method, dataset), grp in res.groupby(['method', 'dataset']):
        default_config = default_sel.get((method, dataset), None)
        frac_same = (grp['config'] == default_config).mean()
        row = {
            'method': method,
            'dataset': dataset,
            'default_config': default_config,
            'n_weight_sets': len(grp),
            'n_distinct_configs': grp['config'].nunique(),
            'frac_same_as_default': frac_same,
        }
        for m in HEADLINE:
            vals = grp[m].dropna()
            if len(vals):
                row[f'{m}_min'] = vals.min()
                row[f'{m}_max'] = vals.max()
                row[f'{m}_range'] = vals.max() - vals.min()
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(['method', 'dataset'])
    out_csv = f'{args.out_prefix}_summary.csv'
    summary.to_csv(out_csv, index=False)

    pd.set_option('display.width', 220)
    print('\n' + '=' * 100)
    print('ASF WEIGHT SENSITIVITY SUMMARY')
    print('(frac_same_as_default = share of weight settings selecting the '
          'default config;\n metric ranges = spread of the reported value '
          'across all weight settings)')
    print('=' * 100)
    cols = ['method', 'dataset', 'n_distinct_configs', 'frac_same_as_default'] + \
           [f'{m}_range' for m in HEADLINE if f'{m}_range' in summary.columns]
    print(summary[cols].round(4).to_string(index=False))
    print(f'\nSummary written to {out_csv}')


if __name__ == '__main__':
    main()
