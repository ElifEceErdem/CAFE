"""Build Tables 11/15 from completed validation CSVs, without running experiments.

Run from the repository root: python analysis/generate_table11_table15_updates.py
Requires pandas, numpy, scipy and pdftotext. Only comparison_tables/ is written.
Table 15's published cells are extracted directly from PDF page 37.
"""

from pathlib import Path
import hashlib
import json
import re
import statistics
import subprocess

import numpy as np
import pandas as pd

from generate_updated_tex_tables import DATASET_LABELS, TABLE11_ORIGINAL, red
from generate_validation_comparisons import (
    T11_METRICS, TABLE11_OKLAHOMA, _comparison_row, write_csv, write_markdown,
)
from significance_tests import paired_stats, holm_correction

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'comparison_tables'
DATASETS = list(DATASET_LABELS)
METHODS = ['RA', 'HL-GNN', 'Fusion', 'CAFE-In', 'CAFE-Pre']
METRICS = ['Precision@10', 'Equality_Gain@10', 'Pagerank_EG@10', 'Closeness_EG@10']
REFERENCES = ['HL-GNN', 'Fusion', 'RA']
SOURCES = []


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path, role):
    SOURCES.append(dict(path=str(path.relative_to(ROOT)), role=role,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


def validate_grid(frame, dataset, methods, metrics, repetition):
    require(set(frame.dataset) == {dataset}, f'{dataset}: incorrect dataset')
    require(not frame.duplicated([repetition, 'method', 'metric']).any(),
            f'{dataset}: duplicate result rows')
    chosen = frame[frame.metric.isin(metrics)]
    expected = {(r, m, k) for r in range(10) for m in methods for k in metrics}
    actual = set(chosen[[repetition, 'method', 'metric']].itertuples(index=False, name=None))
    require(actual == expected, f'{dataset}: missing/unexpected result cells')
    require(np.isfinite(chosen.value).all(), f'{dataset}: non-finite result values')
    return chosen


def acceptance_rows():
    rows = []
    for ds in DATASETS:
        path = (Path(TABLE11_OKLAHOMA) if ds == 'oklahoma' else
                ROOT / 'results/validation/20260902_table11_corrected/table11' /
                ds / 'acceptance_eg' / f'acceptance_eg_{ds}.csv')
        frame = pd.read_csv(path)
        methods = list(TABLE11_ORIGINAL[DATASET_LABELS[ds]])
        frame = validate_grid(frame, ds, methods, [m[0] for m in T11_METRICS], 'run')
        record(path, 'Table 11 corrected acceptance rerun')
        for method in methods:
            old = TABLE11_ORIGINAL[DATASET_LABELS[ds]][method]
            for (metric, label, decimals), (mean, std) in zip(T11_METRICS, old):
                values = frame.loc[(frame.method == method) & (frame.metric == metric), 'value'].tolist()
                current = dict(
                    stats={metric: dict(mean=statistics.mean(values), std=statistics.stdev(values))},
                    status='complete', count=10, expected=10,
                    semantics='legacy target-indegree events for Degree full/PW; '
                              'simple-undirected set union for PR/Cl full and Cl TP; '
                              'probability-union weighted adjacency for PR PW',
                    source=str(path.relative_to(ROOT)))
                rows.append(_comparison_row('11', ds, method, metric, label, decimals,
                                            mean, std, current))
    return rows


def acceptance_tex(rows):
    lookup = {(r['dataset'], r['method'], r['metric']): r for r in rows}
    lines = [r'% Requires \usepackage{xcolor,booktabs}.',
             '% All five datasets have complete corrected acceptance evidence.',
             r'\begin{table*}[t]', r'\caption{EG under acceptance models. '
             r'Degree and PageRank EG use full and precision-weighted acceptance; '
             r'Closeness EG uses full and true-positive acceptance '
             + red('because the implemented closeness estimator uses an unweighted distance graph. ')
             + r'Mean $\pm$ SD over ' + red('10 training repetitions on a fixed split.')
             + ' ' + red(r'Degree uses target-indegree recommendation events under full and PW '
                         r'acceptance. PageRank and Closeness full/TP use simple-undirected '
                         r'distinct-pair graphs; PageRank PW uses reciprocal-probability-union '
                         r'edge weights. HL-GNN+RA uses the fusion coefficient of CAFE-In '
                         r'(Pokec-n: $\beta=1$, unlike the $\beta=5$ control in Table~15).') + '}',
             r'\label{tab:acceptance_models}', r'\centering\small',
             r'\begin{tabular}{llrrrrrr}', r'\toprule',
             r' & & \multicolumn{2}{c}{Deg EG@10} & \multicolumn{2}{c}{PR EG@10} & \multicolumn{2}{c}{Cl.\ EG@10} \\',
             r'\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}',
             r'Dataset & Method & Full & PW & Full & PW & Full & TP \\', r'\midrule']
    for i, ds in enumerate(DATASETS):
        if i:
            lines.append(r'\midrule')
        for j, method in enumerate(TABLE11_ORIGINAL[DATASET_LABELS[ds]]):
            cells = []
            for _, metric, _ in T11_METRICS:
                r = lookup[ds, method, metric]
                cell = f"{r['reproduced_mean']:.2f} $\\pm$ {r['reproduced_std']:.2f}"
                cells.append(red(cell) if r['recommended_action'] == 'UPDATE' else cell)
            lines.append(f'{DATASET_LABELS[ds] if j == 0 else ""} & {method} & '
                         + ' & '.join(cells) + r' \\')
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabular}', r'\end{table*}', ''])


def published_paired_cells():
    pdf = ROOT / 'paper/CAFE_ToRS_EceErdem.pdf'
    text = subprocess.check_output(['pdftotext', '-f', '37', '-l', '37', '-layout', str(pdf), '-'], text=True)
    require('Table 15. Paired contrasts' in text, 'PDF Table 15 moved; verify its page before updating')
    pattern = re.compile(r'([+−-]\d+\.\d+)\s*±\s*(\d+\.\d+)(\*{0,3})')
    table_lines = [line for line in text.splitlines() if len(pattern.findall(line)) == 4]
    require(len(table_lines) == 30, 'Cannot extract all 30 published Table 15 rows')
    output = {}
    for line, (ref, method, ds) in zip(table_lines, (
            (ref, method, ds) for ref in REFERENCES for method in METHODS[-2:] for ds in DATASETS)):
        require(DATASET_LABELS[ds] in line, f'Unexpected published row order: {line}')
        for metric, (mean, width, stars) in zip(METRICS, pattern.findall(line)):
            output[ref, method, ds, metric] = (mean.replace('−', '-'), width, stars)
    record(pdf, 'Published Table 15 cells: PDF page 37 only')
    return output


def stars(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''


def paired_rows():
    old = published_paired_cells()
    output = []
    for ds in DATASETS:
        path = ROOT / 'results/validation/20260902_table15_rerun/table15' / ds / 'significance' / f'per_seed_metrics_{ds}.csv'
        marker = Path(str(path) + '.complete.json')
        meta = json.loads(marker.read_text())
        require(meta['status'] == 'complete' and meta['seeds'] == list(range(10)), f'{ds}: incomplete marker')
        require(meta['csv_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest(), f'{ds}: result hash mismatch')
        frame = validate_grid(pd.read_csv(path), ds, METHODS, METRICS, 'seed')
        require((frame.groupby('seed').split_fingerprint.nunique() == 1).all(), f'{ds}: unpaired splits')
        for seed, group in frame.groupby('seed'):
            require(set(group.split_seed) == {42 if ds == 'brightkite' else seed}, f'{ds}: unexpected split seed')
        require(set(frame.protocol) == {'paired'} and set(frame.evaluation_split) == {'test'}, f'{ds}: wrong protocol')
        job = ROOT / 'experiments/jobs' / ('t15_paired_validation_pokec.job' if ds == 'pokec_n' else 't15_paired_validation.job')
        config = ROOT / 'experiments/configs/legacy_best_configs_selected.csv'
        for column, source in [('source_manifest_sha256', job), ('selected_configs_sha256', config)]:
            if ds == 'pokec_n' and column == 'selected_configs_sha256':
                # The Pokec wrapper intentionally omits the optional manifest,
                # whose matched-beta guard rejects the separately tuned control.
                require(frame[column].isna().all() and meta[column] is None,
                        'Pokec-n: unexpected optional config provenance')
                continue
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            require(set(frame[column]) == {digest} and meta[column] == digest, f'{ds}: {column} mismatch')
        selected = pd.read_csv(config)
        selected = selected[selected.dataset == ds].set_index('variant')
        for variant, fields in {
                'cafe_in': {'lambda': 'lambda_fair', 'beta': 'beta', 'lam': 'lambda_mobility'},
                'cafe_pre': {'alpha': 'preproc_alpha', 'beta': 'beta', 'lam': 'lambda_mobility'}}.items():
            for suffix, field in fields.items():
                require(set(frame[variant + '_' + suffix]) == {float(selected.loc[variant, field])},
                        f'{ds}: explicit {variant}_{suffix} disagrees with frozen configuration')
        require(set(frame.fusion_beta) == {1.0 if ds == 'ego-facebook' else 5.0},
                f'{ds}: unexpected separately selected fusion coefficient')
        record(path, 'Table 15 completed paired rerun')
        record(marker, 'Table 15 completion/hash evidence')
        record(job, 'Table 15 granular metric semantics and launch protocol')
        record(config, 'Frozen selected configurations')
        for ref in REFERENCES:
            for method in METHODS[-2:]:
                for metric in METRICS:
                    pair = frame[frame.metric == metric].pivot(index='seed', columns='method', values='value').sort_index()
                    a, b = pair[method].to_numpy(), pair[ref].to_numpy()
                    result = paired_stats(a, b)
                    require(all(np.isfinite(result[k]) for k in ['mean_diff', 'ci95_lo', 'ci95_hi', 'p_ttest', 'p_wilcoxon']),
                            f'{ds}/{method}/{ref}/{metric}: undefined paired statistics')
                    pm, pw, ps = old[ref, method, ds, metric]
                    output.append(dict(dataset=ds, method=method, reference=ref, metric=metric,
                                       family=f'F{REFERENCES.index(ref)+1}', family_size=40,
                                       paper_mean_diff=float(pm), paper_ci95_halfwidth=float(pw), paper_stars=ps,
                                       paper_display=f'{pm} ± {pw}{ps}',
                                       mean_method=a.mean(), std_method=a.std(ddof=1),
                                       mean_reference=b.mean(), std_reference=b.std(ddof=1),
                                       **result, source_file=str(path.relative_to(ROOT))))
    result = pd.DataFrame(output)
    for _, ids in result.groupby('reference', sort=False).groups.items():
        require(len(ids) == 40, 'Holm family does not contain 40 contrasts')
        for p in ['p_ttest', 'p_wilcoxon']:
            result.loc[ids, p + '_holm'] = holm_correction(result.loc[ids, p].to_numpy())
    # Sensitivity columns expose the old analysis grouping; not used for new stars.
    for _, ids in result.groupby(['reference', 'method'], sort=False).groups.items():
        result.loc[ids, 'p_ttest_holm_legacy20'] = holm_correction(result.loc[ids, 'p_ttest'].to_numpy())
    result['ci95_halfwidth'] = (result.ci95_hi - result.ci95_lo) / 2
    result['reproduced_stars'] = result.p_ttest_holm.map(stars)
    result['stars_legacy20'] = result.p_ttest_holm_legacy20.map(stars)
    result['stars_changed_by_family_size'] = result.reproduced_stars != result.stars_legacy20
    result['reproduced_display'] = result.apply(
        lambda r: f"{r.mean_diff:+.{4 if r.metric == METRICS[0] else 2}f} ± "
                  f"{r.ci95_halfwidth:.{4 if r.metric == METRICS[0] else 2}f}{r.reproduced_stars}", axis=1)
    result['recommended_action'] = np.where(result.paper_display == result.reproduced_display, 'KEEP', 'UPDATE')
    result['evidence_status'] = 'complete'
    return result


def paired_tex(frame):
    lines = [r'% Requires \usepackage{xcolor,booktabs,subcaption}.',
             '% Stars use the paper-declared families F1/F2/F3 (40 contrasts each).',
             r'\begin{table*}[t]',
             r'\caption{Paired contrasts for CAFE against (a) HL-GNN, (b) HL-GNN+RA, and (c) RA. '
             r'Entries are mean CAFE-minus-reference differences $\pm$ 95\% paired $t$-interval '
             + red('half-widths') + r' over $n=10$ seeds; positive values favor CAFE. Asterisks denote '
             r'Holm-corrected two-sided paired $t$-tests within each comparison family '
             r'(${}^{*}p<0.05$, ${}^{**}p<0.01$, ${}^{***}p<0.001$). '
             + red(r'Each panel is one family of 40 contrasts: both CAFE variants '
                   r'$\times$ five datasets $\times$ four metrics (F1--F3).')
             + r' Each seed draws a split on the four random-split datasets; Brightkite retains '
             r'its fixed temporal split, so means may differ from Table~4. '
             + red(r'Degree EG retains target-indegree recommendation-event semantics; '
                   r'PageRank and Closeness EG use simple-undirected distinct-pair graphs. '
                   r'The fusion-only control retains its separately selected coefficient '
                   r'(Pokec-n: $\beta=5$ versus $\beta=1$ in CAFE-In).') + '}',
             r'\label{tab:paired_contrasts}', r'\centering\small']
    titles = ['(a) vs. HL-GNN backbone', '(b) vs. fusion-only control', '(c) vs. RA']
    for ref, title in zip(REFERENCES, titles):
        if ref != REFERENCES[0]:
            lines.append(r'\vspace{6pt}')
        lines.extend([r'\subcaption*{' + title + '}', r'\begin{tabular}{llrrrr}', r'\toprule',
                      r'Variant & Dataset & Prec@10 & Deg EG@10 & PR EG@10 & Cl.\ EG@10 \\', r'\midrule'])
        for i, method in enumerate(METHODS[-2:]):
            if i:
                lines.append(r'\midrule')
            for j, ds in enumerate(DATASETS):
                cells = []
                for metric in METRICS:
                    row = frame[(frame.reference == ref) & (frame.method == method)
                                & (frame.dataset == ds) & (frame.metric == metric)].iloc[0]
                    decimals = 4 if metric == METRICS[0] else 2
                    body = f'{row.mean_diff:+.{decimals}f} \\pm {row.ci95_halfwidth:.{decimals}f}'
                    if row.reproduced_stars:
                        body += '^{' + row.reproduced_stars + '}'
                    cell = '$' + body + '$'
                    cells.append(red(cell) if row.recommended_action == 'UPDATE' else cell)
                lines.append(f'{method if j == 0 else ""} & {DATASET_LABELS[ds]} & ' + ' & '.join(cells) + r' \\')
        lines.extend([r'\bottomrule', r'\end{tabular}'])
    return '\n'.join(lines + [r'\end{table*}', ''])


def main():
    # Validate every input before replacing any output.
    acceptance = acceptance_rows()
    paired = paired_rows()
    t11_tex, t15_tex = acceptance_tex(acceptance), paired_tex(paired)
    OUT.mkdir(exist_ok=True)
    write_csv(OUT / 'table11_paper_vs_reproduced.csv', acceptance)
    write_markdown(OUT / 'table11_paper_vs_reproduced.md', 'Table 11: paper vs. corrected reruns (all datasets)',
                   acceptance, [m[1] for m in T11_METRICS])
    (OUT / 'table11_acceptance_updated_red.tex').write_text(t11_tex)
    paired.to_csv(OUT / 'table15_paper_vs_reproduced.csv', index=False)
    (OUT / 'table15_updated_red.tex').write_text(t15_tex)
    lines = ['# Table 15: paper vs. corrected paired reruns', '',
             'Mean paired difference ± 95% t-interval half-width, not SD. All cells use 10 paired seeds. '
             'Stars use Holm correction across 40 contrasts per panel; see table11_table15_update_notes.md.', '',
             '| Reference | Variant | Dataset | Metric | Paper | Reproduced | Action |',
             '|---|---|---|---|---|---|---|']
    for row in paired.itertuples():
        lines.append(f'| {row.reference} | {row.method} | {row.dataset} | {row.metric} | '
                     f'`{row.paper_display}` | `{row.reproduced_display}` | {row.recommended_action} |')
    (OUT / 'table15_paper_vs_reproduced.md').write_text('\n'.join(lines) + '\n')
    record(ROOT / 'analysis/generate_updated_tex_tables.py', 'Transcribed original Table 11 values')
    record(Path(__file__).resolve(), 'Reporting generator')
    record(ROOT / 'analysis/significance_tests.py', 'Reused paired t/CI and Holm arithmetic')
    sources = list({s['path']: s for s in SOURCES}.values())
    (OUT / 'table11_table15_sources.json').write_text(json.dumps(sources, indent=2) + '\n')
    print('Table 11:', pd.Series([r['recommended_action'] for r in acceptance]).value_counts().to_dict())
    print('Table 15:', paired.recommended_action.value_counts().to_dict())
    print('New stars affected solely by 40-vs-20 family size:')
    print(paired.loc[paired.stars_changed_by_family_size,
                     ['dataset', 'method', 'reference', 'metric', 'stars_legacy20', 'reproduced_stars']].to_string(index=False))


if __name__ == '__main__':
    main()
