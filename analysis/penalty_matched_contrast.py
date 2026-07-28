"""Matched-parameter test of the CAFE in-processing coreness penalty.

Table~7 tunes every row separately, so its CAFE-In-minus-CAFE$^{-P}$ gap
confounds the penalty with different selected $(\\beta, \\lambda_{mob})$: on the
paper's configurations the two rows never share $\\beta$, and differ in
$\\lambda_{mob}$ on three of five datasets. This script isolates the penalty.

``results_penalty_reduction/*/cafe_in/*pen-mean*.json`` sweeps
$\\lambda_{fair}$ with $\\beta$ and $\\lambda_{mob}$ held at CAFE-In's selected
values, one fixed split, ten training seeds (``model_seed = 42 + run_index``),
under the ``mean`` penalty reduction the headline runs use. The
$\\lambda_{fair}=0$ arm is therefore exactly CAFE$^{-P}$ evaluated at CAFE-In's
own $(\\beta, \\lambda_{mob})$ -- the contrast the reviewer asks for -- and the
arms are paired by training seed.

Emits the paired contrast (t-test and Wilcoxon, Holm-corrected within the
dataset x measure family), the full dose curve, and their LaTeX.
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

PEN_DIR = 'results_penalty_reduction'
BEST_CONFIGS = 'results_repro/tables/best_configs.csv'

METRICS = [
    ('Precision@10', 'Prec@10', 4),
    ('Equality_Gain@10', 'Deg EG@10', 2),
    ('Pagerank_EG@10', 'PR EG@10', 2),
    ('Closeness_EG@10', 'Cl.\\ EG@10', 2),
]
DATASET_LABEL = {'ego-facebook': 'Ego-FB', 'oklahoma': 'Oklahoma', 'unc': 'UNC'}


def argument():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pen_dir', default=PEN_DIR)
    parser.add_argument('--best_configs', default=BEST_CONFIGS)
    parser.add_argument('--out_dir', default='results_repro/tables')
    parser.add_argument('--reduction', default='mean',
                        choices=['mean', 'sum'],
                        help="'mean' is what the headline runs use")
    return parser.parse_args()


def selected_lambda(best_configs):
    """CAFE-In's selected (lambda_fair, beta, lambda_mob) per dataset."""
    frame = pd.read_csv(best_configs)
    rows = frame[frame['Method'] == 'CAFE-In']
    out = {}
    for _, row in rows.iterrows():
        # Config_Key looks like 'lambda_fair_0.3_beta_5.0_lam_0.5'
        parts = str(row['Config_Key']).split('_')
        out[row['Dataset']] = {
            'lambda_fair': float(parts[parts.index('fair') + 1]),
            'beta': float(parts[parts.index('beta') + 1]),
            'lambda_mob': float(parts[parts.index('lam') + 1]),
        }
    return out


def load_runs(path):
    """{config_key: {metric: array over runs}} from one penalty-sweep JSON."""
    with open(path) as handle:
        payload = json.load(handle)
    info = payload['experiment_info']
    runs = payload['individual_runs']
    out = {}
    for key in payload['lambda_results']:
        per_metric = {}
        for metric, _, _ in METRICS:
            values = []
            for run in runs:
                if key not in run or metric not in run[key]:
                    values.append(np.nan)
                    continue
                raw = run[key][metric]
                # metrics are stored as [unused, value]
                values.append(float(raw[1] if isinstance(raw, list) else raw))
            per_metric[metric] = np.asarray(values, dtype=float)
        out[key] = per_metric
    return out, info


def holm(pvals):
    order = np.argsort(pvals)
    n = len(pvals)
    adjusted = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * pvals[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def main():
    args = argument()
    os.makedirs(args.out_dir, exist_ok=True)
    selected = selected_lambda(args.best_configs)

    contrast_rows, dose_rows = [], []
    for path in sorted(glob.glob(os.path.join(
            args.pen_dir, '*', 'cafe_in', f'*pen-{args.reduction}*.json'))):
        if path.endswith('.partial.json'):
            continue
        dataset = path.split(os.sep)[1]
        arms, info = load_runs(path)
        if info.get('penalty_reduction') != args.reduction:
            raise ValueError(f"{path}: penalty_reduction mismatch")
        cfg = selected[dataset]

        # Match arms numerically: the stored keys carry Python float reprs
        # ('lambda_fair_0.0_beta_1.0_lam_0.5'), so string formatting is fragile.
        by_lambda = {}
        for key in arms:
            parts = key.split('_')
            lam = float(parts[parts.index('fair') + 1])
            beta = float(parts[parts.index('beta') + 1])
            mob = float(parts[parts.index('lam') + 1])
            if not (np.isclose(beta, cfg['beta'])
                    and np.isclose(mob, cfg['lambda_mob'])):
                raise ValueError(
                    f"{dataset}: arm {key} is not at CAFE-In's selected "
                    f"(beta={cfg['beta']}, lambda_mob={cfg['lambda_mob']})")
            by_lambda[lam] = key

        def key_for(lam):
            for candidate, key in by_lambda.items():
                if np.isclose(candidate, lam):
                    return key
            raise KeyError(f"{dataset}: no arm at lambda_fair={lam}; "
                           f"have {sorted(by_lambda)}")

        zero_key = key_for(0.0)
        sel_key = key_for(cfg['lambda_fair'])

        for key, per_metric in sorted(arms.items()):
            lam = float(key.split('_beta_')[0].rsplit('_', 1)[1])
            for metric, _, _ in METRICS:
                values = per_metric[metric]
                dose_rows.append({
                    'dataset': dataset, 'reduction': args.reduction,
                    'lambda_fair': lam, 'beta': cfg['beta'],
                    'lambda_mob': cfg['lambda_mob'], 'metric': metric,
                    'n_runs': int(np.isfinite(values).sum()),
                    'mean': np.nanmean(values), 'std': np.nanstd(values, ddof=1),
                })

        for metric, _, _ in METRICS:
            a = arms[sel_key][metric]      # CAFE-In at its selected lambda
            b = arms[zero_key][metric]     # CAFE^-P at the same beta, lambda_mob
            mask = np.isfinite(a) & np.isfinite(b)
            a, b = a[mask], b[mask]
            diff = a - b
            n = len(diff)
            t_stat, p_t = ttest_rel(a, b)
            try:
                _, p_w = wilcoxon(a, b)
            except ValueError:      # all-zero differences
                p_w = 1.0
            se = diff.std(ddof=1) / np.sqrt(n)
            # 95% t-based half-width
            from scipy.stats import t as tdist
            half = tdist.ppf(0.975, n - 1) * se
            contrast_rows.append({
                'dataset': dataset, 'reduction': args.reduction,
                'lambda_fair': cfg['lambda_fair'], 'beta': cfg['beta'],
                'lambda_mob': cfg['lambda_mob'], 'metric': metric,
                'n_seeds': n,
                'cafe_in_mean': a.mean(), 'cafe_minus_p_mean': b.mean(),
                'mean_diff': diff.mean(),
                'ci95_half_width': half,
                't_stat': t_stat, 'p_ttest': p_t, 'p_wilcoxon': p_w,
                'cohens_dz': diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) else np.nan,
            })

    contrasts = pd.DataFrame(contrast_rows)
    contrasts['p_ttest_holm'] = holm(contrasts['p_ttest'].to_numpy())
    contrasts['p_wilcoxon_holm'] = holm(contrasts['p_wilcoxon'].to_numpy())
    contrasts['significant_0.05'] = contrasts['p_ttest_holm'] < 0.05
    dose = pd.DataFrame(dose_rows)

    contrasts.to_csv(os.path.join(args.out_dir,
                                  'penalty_matched_contrast.csv'), index=False)
    dose.to_csv(os.path.join(args.out_dir,
                             'penalty_dose_curve.csv'), index=False)

    with pd.option_context('display.width', 220):
        print(f"=== CAFE-In minus CAFE^-P at matched (beta, lambda_mob), "
              f"pen={args.reduction} ===")
        print(contrasts[['dataset', 'lambda_fair', 'beta', 'lambda_mob',
                         'metric', 'n_seeds', 'cafe_minus_p_mean',
                         'cafe_in_mean', 'mean_diff', 'ci95_half_width',
                         'p_ttest_holm', 'significant_0.05']]
              .round(4).to_string(index=False))

    write_latex(contrasts, dose, args)


def write_latex(contrasts, dose, args):
    lines = [
        r'% Auto-generated by analysis/penalty_matched_contrast.py'
        r' -- do not edit by hand.',
        r'\begin{table*}[t]',
        r'\caption{The in-processing coreness penalty at matched fusion and'
        r' boost parameters (family F6). Each dataset holds $\beta$ and'
        r" $\lambda_{\mathrm{mob}}$ at CAFE-In's selected values and varies"
        r' only $\lambda_{\mathrm{fair}}$, so the $\lambda_{\mathrm{fair}}=0$'
        r" arm is CAFE$^{-P}$ at CAFE-In's own operating point. One fixed"
        r' split, $n=10$ training seeds paired by seed'
        r' ($\mathrm{seed}=42+\mathrm{run}$), ' + args.reduction +
        r'-reduced penalty as in the headline runs. Panel~(a) gives the paired'
        r' CAFE-In-minus-CAFE$^{-P}$ difference $\pm$ the half-width of its'
        r' $95\%$ $t$-based confidence interval; asterisks mark two-sided'
        r' paired $t$-tests Holm-corrected within the'
        r' datasets$\times$measures family ($^{*}p<0.05$, $^{**}p<0.01$,'
        r' $^{***}p<0.001$). Panel~(b) gives the dose curve. Brightkite and'
        r' Pokec-n have no CAFE-In penalty-sweep arm and are omitted.}',
        r'\label{tab:penalty-matched}',
        r'\centering\small',
        r'\subcaption*{(a) Paired CAFE-In minus CAFE$^{-P}$ at matched'
        r' $(\beta,\lambda_{\mathrm{mob}})$}',
        r'\begin{tabular}{lrrr' + 'r' * len(METRICS) + r'}',
        r'\toprule',
        r'Dataset & $\beta$ & $\lambda_{\mathrm{mob}}$ &'
        r' $\lambda_{\mathrm{fair}}$ & '
        + ' & '.join(short for _, short, _ in METRICS) + r' \\',
        r'\midrule',
    ]

    def stars(p):
        # Rendered inside the same math group as the value, so no `$$` pair.
        return ('^{***}' if p < 0.001 else '^{**}' if p < 0.01
                else '^{*}' if p < 0.05 else '')

    for dataset in ['ego-facebook', 'oklahoma', 'unc']:
        block = contrasts[contrasts['dataset'] == dataset]
        if block.empty:
            continue
        first = block.iloc[0]
        cells = []
        for metric, _, digits in METRICS:
            row = block[block['metric'] == metric].iloc[0]
            cells.append(f"${row['mean_diff']:+.{digits}f} \\pm "
                         f"{row['ci95_half_width']:.{digits}f}"
                         + stars(row['p_ttest_holm']) + '$')
        lines.append(
            f"{DATASET_LABEL[dataset]} & {first['beta']:g} & "
            f"{first['lambda_mob']:g} & {first['lambda_fair']:g} & "
            + ' & '.join(cells) + r' \\')

    lines += [
        r'\bottomrule', r'\end{tabular}', r'\vspace{8pt}',
        r'\subcaption*{(b) Dose curve: absolute values as'
        r' $\lambda_{\mathrm{fair}}$ varies, $(\beta,\lambda_{\mathrm{mob}})$'
        r' fixed}',
        r'\begin{tabular}{llrrrrrrr}',
        r'\toprule',
        r'Dataset & Measure & ' + ' & '.join(
            f"$\\lambda{{=}}{lam:g}$" for lam in
            sorted(dose['lambda_fair'].unique())) + r' \\',
        r'\midrule',
    ]
    lams = sorted(dose['lambda_fair'].unique())
    for dataset in ['ego-facebook', 'oklahoma', 'unc']:
        block = dose[dose['dataset'] == dataset]
        if block.empty:
            continue
        for i, (metric, short, digits) in enumerate(METRICS):
            sub = block[block['metric'] == metric].set_index('lambda_fair')
            cells = [f"{sub.loc[lam, 'mean']:.{digits}f}" for lam in lams]
            label = DATASET_LABEL[dataset] if i == 0 else ''
            lines.append(f"{label} & {short} & " + ' & '.join(cells) + r' \\')
        if dataset != 'unc':
            lines.append(r'\addlinespace[2pt]')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}', '']

    path = os.path.join(args.out_dir, 'table_penalty_matched.tex')
    with open(path, 'w') as handle:
        handle.write('\n'.join(lines))
    print(f"\nwrote {path}")


if __name__ == '__main__':
    main()
