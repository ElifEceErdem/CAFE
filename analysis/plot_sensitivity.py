"""
Experiment (d), analysis stage: hyperparameter sensitivity curves for CAFE.

Consumes the sweep JSONs produced by experiments/run_cafe.py (the
sensitivity jobs sweep lambda_fair / preproc_alpha x beta_ra x
lambda_mobility on a dense grid) and produces one-at-a-time sensitivity
curves: each CAFE hyperparameter is varied while the other two are held at
the best configuration, for each requested metric.

Also produces a beta x lambda_mobility heatmap at the best penalty value.

Usage:
    python plot_sensitivity.py \
        --results_json ../results/oklahoma/sensitivity_in/results_sensitivity_in_oklahoma_*.json \
        --best "lambda_fair=0.3,beta=5.0,lam=0.5" \
        --metrics "Precision@10,Equality_Gain@10,Pagerank_EG@10,Closeness_EG@10" \
        --save_dir plots/sensitivity
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

KEY_RE = re.compile(
    r'(?P<pen_label>lambda_fair|preproc_alpha)_(?P<pen>[\d.]+)'
    r'_beta_(?P<beta>[\d.]+)_lam_(?P<lam>[\d.]+)'
)

PEN_AXIS_LABEL = {
    'lambda_fair': r'$\lambda_{fair}$ (in-processing penalty)',
    'preproc_alpha': r'$\alpha_{pre}$ (pre-processing strength)',
}


def parse_results(path):
    """Return (pen_label, {(pen, beta, lam): {metric: (mean, std)}})."""
    with open(path) as f:
        data = json.load(f)

    configs = {}
    pen_label = None
    for key, entry in data.get('lambda_results', {}).items():
        m = KEY_RE.match(key)
        if not m:
            continue
        pen_label = m.group('pen_label')
        combo = (float(m.group('pen')), float(m.group('beta')), float(m.group('lam')))
        stats = entry.get('statistics', {})
        metrics = {}
        for metric, s in stats.items():
            if isinstance(s, dict) and 'mean' in s:
                metrics[metric] = (float(s['mean']), float(s.get('std', 0.0)))
        configs[combo] = metrics
    return pen_label, configs, data.get('experiment_info', {})


def parse_best(best_str):
    """Parse 'lambda_fair=0.3,beta=5.0,lam=0.5' -> dict."""
    out = {}
    for part in best_str.split(','):
        k, v = part.split('=')
        out[k.strip()] = float(v)
    return out


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


def main():
    parser = argparse.ArgumentParser(description='CAFE hyperparameter sensitivity plots')
    parser.add_argument('--results_json', type=str, required=True,
                        help='Path or glob to run_cafe.py sweep JSON(s)')
    parser.add_argument('--best', type=str, required=True,
                        help='Best config, e.g. "lambda_fair=0.3,beta=5.0,lam=0.5" '
                             'or "preproc_alpha=1.0,beta=0.5,lam=0.5"')
    parser.add_argument('--metrics', type=str,
                        default='Precision@10,Equality_Gain@10,Pagerank_EG@10,Closeness_EG@10')
    parser.add_argument('--save_dir', type=str, default='plots/sensitivity')
    parser.add_argument('--tag', type=str, default=None,
                        help='Filename tag (default: dataset + config name from JSON)')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    metrics = [m.strip() for m in args.metrics.split(',')]
    best = parse_best(args.best)

    paths = sorted(glob.glob(args.results_json))
    if not paths:
        raise SystemExit(f'No files matched {args.results_json}')

    # Merge configs across files (e.g. sweep split over several jobs)
    all_configs = {}
    pen_label, exp_info = None, {}
    for p in paths:
        pl, configs, info = parse_results(p)
        pen_label = pen_label or pl
        exp_info = exp_info or info
        all_configs.update(configs)
    if pen_label is None:
        raise SystemExit('No sweep keys found in the JSON(s).')

    pen_best = best.get(pen_label)
    beta_best, lam_best = best.get('beta'), best.get('lam')
    if pen_best is None:
        raise SystemExit(f"--best must contain '{pen_label}=' for this sweep")

    dataset = exp_info.get('dataset', 'unknown')
    tag = args.tag or f"{dataset}_{pen_label}"

    print(f'Dataset: {dataset} | sweep: {pen_label} x beta x lam '
          f'| {len(all_configs)} configs')
    print(f'Best config: {pen_label}={pen_best}, beta={beta_best}, lam={lam_best}')

    # ------------------------------------------------------------------
    # One-at-a-time curves: 3 hyperparameters x len(metrics) panels
    # ------------------------------------------------------------------
    sweeps = [
        (pen_label, 0, lambda c: close(c[1], beta_best) and close(c[2], lam_best)),
        ('beta_ra', 1, lambda c: close(c[0], pen_best) and close(c[2], lam_best)),
        ('lambda_mobility', 2, lambda c: close(c[0], pen_best) and close(c[1], beta_best)),
    ]
    axis_labels = {
        pen_label: PEN_AXIS_LABEL.get(pen_label, pen_label),
        'beta_ra': r'$\beta$ (RA fusion weight)',
        'lambda_mobility': r'$\lambda_{mob}$ (mobility boost)',
    }
    best_vals = {pen_label: pen_best, 'beta_ra': beta_best,
                 'lambda_mobility': lam_best}

    fig, axes = plt.subplots(len(metrics), len(sweeps),
                             figsize=(4.2 * len(sweeps), 2.9 * len(metrics)),
                             squeeze=False)

    for col, (sweep_name, dim, mask_fn) in enumerate(sweeps):
        combos = sorted([c for c in all_configs if mask_fn(c)], key=lambda c: c[dim])
        xs = [c[dim] for c in combos]
        if not combos:
            print(f'WARNING: no configs found for {sweep_name} sweep '
                  f'(others fixed at best); check the grid.')
        for row, metric in enumerate(metrics):
            ax = axes[row][col]
            means = [all_configs[c].get(metric, (np.nan, np.nan))[0] for c in combos]
            stds = [all_configs[c].get(metric, (np.nan, np.nan))[1] for c in combos]
            ax.errorbar(xs, means, yerr=stds, marker='o', ms=4, capsize=3,
                        lw=1.2, color='#2c7fb8')
            bv = best_vals[sweep_name]
            if bv is not None:
                ax.axvline(bv, color='#d95f02', ls='--', lw=1, alpha=0.7)
            if row == len(metrics) - 1:
                ax.set_xlabel(axis_labels[sweep_name], fontsize=10)
            if col == 0:
                ax.set_ylabel(metric, fontsize=9)
            ax.grid(alpha=0.3, lw=0.5)
            ax.tick_params(labelsize=8)

    fig.suptitle(f'CAFE hyperparameter sensitivity — {dataset} '
                 f'(others fixed at best config)', fontsize=12, y=1.005)
    fig.tight_layout()
    out = os.path.join(args.save_dir, f'sensitivity_{tag}.pdf')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

    # ------------------------------------------------------------------
    # beta x lambda_mobility heatmap at the best penalty value
    # ------------------------------------------------------------------
    combos_2d = [c for c in all_configs if close(c[0], pen_best)]
    betas = sorted({c[1] for c in combos_2d})
    lams = sorted({c[2] for c in combos_2d})
    if len(betas) > 1 and len(lams) > 1:
        for metric in metrics:
            M = np.full((len(lams), len(betas)), np.nan)
            for c in combos_2d:
                i, j = lams.index(c[2]), betas.index(c[1])
                M[i, j] = all_configs[c].get(metric, (np.nan,))[0]
            fig, ax = plt.subplots(figsize=(1.1 * len(betas) + 2,
                                            0.8 * len(lams) + 2))
            im = ax.imshow(M, aspect='auto', cmap='viridis', origin='lower')
            fig.colorbar(im, ax=ax, label=metric)
            ax.set_xticks(range(len(betas)))
            ax.set_xticklabels(betas)
            ax.set_yticks(range(len(lams)))
            ax.set_yticklabels(lams)
            ax.set_xlabel(r'$\beta$')
            ax.set_ylabel(r'$\lambda_{mob}$')
            for i in range(len(lams)):
                for j in range(len(betas)):
                    if not np.isnan(M[i, j]):
                        ax.text(j, i, f'{M[i, j]:.3f}', ha='center', va='center',
                                fontsize=7, color='white')
            ax.set_title(f'{metric} — {dataset}, {pen_label}={pen_best}',
                         fontsize=10)
            fig.tight_layout()
            safe_metric = metric.replace('@', '').replace('/', '_')
            out = os.path.join(args.save_dir,
                               f'heatmap_{tag}_{safe_metric}.pdf')
            fig.savefig(out, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved: {out}')

    print('Sensitivity plots complete.')


if __name__ == '__main__':
    main()
