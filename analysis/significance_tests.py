"""
Experiment (c), analysis stage: paired significance tests with multiple-
comparison correction and effect sizes.

Consumes the long-format per-seed CSVs written by experiments/run_significance.py
(columns: dataset, seed, split_seed, method, metric, value) and produces, for
each (comparison, dataset, metric):

  - mean +/- std for both methods (across seeds)
  - mean paired difference and its 95% CI (t-based)
  - paired t-test p-value and Wilcoxon signed-rank p-value
  - Holm-Bonferroni corrected p-values (family = all datasets x metrics
    within one comparison, the conservative choice a reviewer will expect)
  - Cohen's d_z (paired) with Hedges' small-sample correction

Outputs a CSV and two LaTeX fragments: the full long-format table
({out_prefix}_table.tex) and a compact three-panel version for the paper
appendix ({out_prefix}_table_compact.tex), one panel per comparator with
datasets x metrics cells of "mean difference +/- 95% CI half-width" plus
Holm significance stars.

Usage:
    python significance_tests.py \
        --input_glob '../results/*/significance/per_seed_metrics_*.csv' \
        --metrics 'Precision@10,Equality_Gain@10,Pagerank_EG@10,Closeness_EG@10' \
        --comparisons 'CAFE-In:HL-GNN,CAFE-Pre:HL-GNN,CAFE-In:Fusion,CAFE-Pre:Fusion' \
        --out_prefix significance
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats


def holm_correction(pvals):
    """Holm-Bonferroni step-down correction. Returns adjusted p-values."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def cohens_dz(diffs):
    """Paired Cohen's d_z with Hedges' correction for small n."""
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    sd = diffs.std(ddof=1)
    if sd == 0:
        return np.nan
    dz = diffs.mean() / sd
    # Hedges' g correction factor
    j = 1.0 - 3.0 / (4.0 * (n - 1) - 1.0) if n > 2 else 1.0
    return dz * j


def paired_stats(a, b):
    """
    a, b: aligned per-seed arrays (method_a, method_b).
    Returns dict of paired statistics for a - b.
    """
    diffs = a - b
    n = len(diffs)
    mean_diff = diffs.mean()
    sd_diff = diffs.std(ddof=1)
    se = sd_diff / np.sqrt(n) if n > 1 else np.nan

    if n > 1 and sd_diff > 0:
        t_stat, p_t = stats.ttest_rel(a, b)
        try:
            w_stat, p_w = stats.wilcoxon(a, b)
        except ValueError:  # all differences zero
            w_stat, p_w = np.nan, 1.0
        tcrit = stats.t.ppf(0.975, df=n - 1)
        ci_lo, ci_hi = mean_diff - tcrit * se, mean_diff + tcrit * se
    else:
        t_stat = p_t = w_stat = p_w = ci_lo = ci_hi = np.nan

    return {
        'n_seeds': n,
        'mean_diff': mean_diff,
        'ci95_lo': ci_lo,
        'ci95_hi': ci_hi,
        't_stat': t_stat,
        'p_ttest': p_t,
        'p_wilcoxon': p_w,
        'cohens_dz': cohens_dz(diffs),
    }


def main():
    parser = argparse.ArgumentParser(description='Paired significance tests (experiment c)')
    parser.add_argument('--input_glob', type=str,
                        default='../results/*/significance/per_seed_metrics_*.csv')
    parser.add_argument('--metrics', type=str,
                        default='Precision@10,Equality_Gain@10,Pagerank_EG@10,Closeness_EG@10')
    parser.add_argument('--comparisons', type=str,
                        default='CAFE-In:HL-GNN,CAFE-Pre:HL-GNN,CAFE-In:Fusion,CAFE-Pre:Fusion,CAFE-In:RA,CAFE-Pre:RA')
    parser.add_argument('--correction_scope', type=str, default='comparison',
                        choices=['comparison', 'global'],
                        help="Holm family: per comparison (datasets x metrics) or "
                             "one global family over everything")
    parser.add_argument('--out_prefix', type=str, default='significance')
    args = parser.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise SystemExit(f"No input files matched: {args.input_glob}")
    print(f"Loading {len(files)} per-seed CSV(s):")
    for f in files:
        print(f"  {f}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset=['dataset', 'seed', 'method', 'metric'], keep='last')

    metrics = [m.strip() for m in args.metrics.split(',')]
    comparisons = [tuple(c.split(':')) for c in args.comparisons.split(',')]

    rows = []
    for method_a, method_b in comparisons:
        for dataset in sorted(df['dataset'].unique()):
            for metric in metrics:
                sub = df[(df['dataset'] == dataset) & (df['metric'] == metric)]
                a = sub[sub['method'] == method_a].set_index('seed')['value']
                b = sub[sub['method'] == method_b].set_index('seed')['value']
                common = a.index.intersection(b.index)
                if len(common) < 2:
                    continue
                a_al, b_al = a.loc[common].values, b.loc[common].values

                row = {
                    'comparison': f'{method_a} vs {method_b}',
                    'dataset': dataset,
                    'metric': metric,
                    f'{method_a}_mean': a_al.mean(),
                    f'{method_a}_std': a_al.std(ddof=1),
                    f'{method_b}_mean': b_al.mean(),
                    f'{method_b}_std': b_al.std(ddof=1),
                    'mean_a': a_al.mean(),
                    'std_a': a_al.std(ddof=1),
                    'mean_b': b_al.mean(),
                    'std_b': b_al.std(ddof=1),
                }
                row.update(paired_stats(a_al, b_al))
                rows.append(row)

    if not rows:
        raise SystemExit("No comparable (comparison, dataset, metric) cells found. "
                         "Check method/metric names against the CSV contents.")

    res = pd.DataFrame(rows)

    # Holm correction
    if args.correction_scope == 'global':
        res['p_ttest_holm'] = holm_correction(res['p_ttest'].fillna(1.0).values)
        res['p_wilcoxon_holm'] = holm_correction(res['p_wilcoxon'].fillna(1.0).values)
    else:
        res['p_ttest_holm'] = np.nan
        res['p_wilcoxon_holm'] = np.nan
        for comp, idx in res.groupby('comparison').groups.items():
            res.loc[idx, 'p_ttest_holm'] = holm_correction(
                res.loc[idx, 'p_ttest'].fillna(1.0).values)
            res.loc[idx, 'p_wilcoxon_holm'] = holm_correction(
                res.loc[idx, 'p_wilcoxon'].fillna(1.0).values)

    res['significant_0.05'] = res['p_ttest_holm'] < 0.05

    out_csv = f'{args.out_prefix}_tests.csv'
    res.to_csv(out_csv, index=False)
    print(f"\nFull results written to {out_csv}")

    # ------------------------------------------------------------------
    # Console summary + LaTeX fragment
    # ------------------------------------------------------------------
    def stars(p):
        if np.isnan(p):
            return ''
        if p < 0.001:
            return '***'
        if p < 0.01:
            return '**'
        if p < 0.05:
            return '*'
        return ''

    pd.set_option('display.width', 200)
    print("\nSummary (Holm-corrected paired t-test):")
    summary_cols = ['comparison', 'dataset', 'metric', 'mean_a', 'mean_b',
                    'mean_diff', 'ci95_lo', 'ci95_hi', 'p_ttest', 'p_ttest_holm',
                    'p_wilcoxon_holm', 'cohens_dz', 'n_seeds']
    print(res[summary_cols].round(4).to_string(index=False))

    def tex_escape(s):
        return s.replace('_', '\\_')

    def fmt_p(p):
        return '$<$0.001' if p < 0.001 else f'{p:.3f}'

    latex_lines = [
        '% Auto-generated by analysis/significance_tests.py',
        '\\begin{tabular}{llrrrrrr}',
        '\\toprule',
        'Comparison & Dataset & Metric & $\\Delta$ & 95\\% CI & '
        '$p_{\\mathrm{Holm}}$ & $d_z$ & $n$ \\\\',
        '\\midrule',
    ]
    for _, r in res.iterrows():
        ci = f"[{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}]"
        latex_lines.append(
            f"{tex_escape(r['comparison'])} & {tex_escape(r['dataset'])} & "
            f"{tex_escape(r['metric'])} & "
            f"{r['mean_diff']:.4f}{stars(r['p_ttest_holm'])} & {ci} & "
            f"{fmt_p(r['p_ttest_holm'])} & {r['cohens_dz']:.2f} & {int(r['n_seeds'])} \\\\"
        )
    latex_lines += ['\\bottomrule', '\\end{tabular}',
                    '% *: p<0.05, **: p<0.01, ***: p<0.001 (Holm-corrected paired t-test)']

    out_tex = f'{args.out_prefix}_table.tex'
    with open(out_tex, 'w') as f:
        f.write('\n'.join(latex_lines) + '\n')
    print(f"\nLaTeX table written to {out_tex}")

    # ------------------------------------------------------------------
    # Compact three-panel fragment for the paper appendix. Display names
    # fall back to the escaped raw strings for datasets/metrics/methods
    # outside the paper's headline set.
    # ------------------------------------------------------------------
    DATASET_TEX = {'brightkite': 'Brightkite', 'ego-facebook': 'Ego-FB',
                   'oklahoma': 'Oklahoma', 'pokec_n': 'Pokec-n', 'unc': 'UNC'}
    METRIC_TEX = {'Precision@10': 'Prec@10', 'Equality_Gain@10': 'Deg EG@10',
                  'Pagerank_EG@10': 'PR EG@10', 'Closeness_EG@10': 'Cl.\\ EG@10'}
    COMPARATOR_TEX = {'HL-GNN': 'HL-GNN backbone', 'Fusion': 'fusion-only control'}

    by_key = {(r['comparison'], r['dataset'], r['metric']): r
              for _, r in res.iterrows()}
    variants = list(dict.fromkeys(a for a, _ in comparisons))
    comparators = list(dict.fromkeys(b for _, b in comparisons))
    datasets = sorted(res['dataset'].unique())

    def cell(comp, dataset, metric):
        r = by_key.get((comp, dataset, metric))
        if r is None:
            return '--'
        half = (r['ci95_hi'] - r['ci95_lo']) / 2.0
        dec = 4 if metric == 'Precision@10' else 2
        return f"${r['mean_diff']:+.{dec}f} \\pm {half:.{dec}f}$" \
               f"{stars(r['p_ttest_holm'])}"

    header = ' & '.join(['Variant', 'Dataset'] +
                        [METRIC_TEX.get(m, tex_escape(m)) for m in metrics])
    compact = ['% Auto-generated by analysis/significance_tests.py '
               '-- do not edit by hand.']
    for i, comparator in enumerate(comparators):
        label = COMPARATOR_TEX.get(comparator, tex_escape(comparator))
        compact += [f"\\subcaption*{{({chr(ord('a') + i)}) vs.\\ {label}}}",
                    f"\\begin{{tabular}}{{ll{'r' * len(metrics)}}}",
                    '\\toprule', f'{header} \\\\', '\\midrule']
        for j, variant in enumerate(variants):
            if j:
                compact.append('\\midrule')
            comp = f'{variant} vs {comparator}'
            for k, dataset in enumerate(datasets):
                lead = variant if k == 0 else ''
                cells = ' & '.join(cell(comp, dataset, m) for m in metrics)
                compact.append(
                    f"{lead} & {DATASET_TEX.get(dataset, tex_escape(dataset))} "
                    f"& {cells} \\\\")
        compact += ['\\bottomrule', '\\end{tabular}', '\\vspace{6pt}', '']

    t_sig = res['p_ttest_holm'] < 0.05
    w_sig = res['p_wilcoxon_holm'] < 0.05
    agree = int((t_sig == w_sig).sum())
    compact += [f'% Wilcoxon signed-rank agreement with the paired t-test at '
                f'alpha=0.05 (both Holm-corrected): {agree}/{len(res)} contrasts.',
                '% *: p<0.05, **: p<0.01, ***: p<0.001 (Holm-corrected paired '
                't-test); cells are mean paired difference +/- 95% CI half-width.']

    out_compact = f'{args.out_prefix}_table_compact.tex'
    with open(out_compact, 'w') as f:
        f.write('\n'.join(compact) + '\n')
    print(f"Compact LaTeX table written to {out_compact}")

    disagree = res[t_sig != w_sig]
    if len(disagree):
        print("\nWilcoxon/t-test disagreements (Holm-corrected, alpha=0.05):")
        print(disagree[['comparison', 'dataset', 'metric', 'p_ttest_holm',
                        'p_wilcoxon_holm']].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
