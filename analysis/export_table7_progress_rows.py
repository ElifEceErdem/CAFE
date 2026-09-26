"""Validate available Table 7 reruns; print additive row/CSV/source artifacts.

No jobs, scientific-code changes, original-file overwrites or partial-result
averaging. Incomplete rows retain the manuscript's numbers in red. Rank only a
fully completed dataset panel, using unrounded means and distinct-value ranks.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import re
import statistics
import sys

import export_exact_closeness_rows as evidence

ROOT = evidence.ROOT
sys.path.insert(0, str(ROOT/'experiments'))
import complete_closeness_v1 as missing

ORIGINAL = ROOT/'results/validation/20260913_135426_table7_exact_v1_missing'
RESTART = ROOT/'results/validation/20260913_135828_table7_defq_restart'
ORDER = ['HL-GNN', 'CI-In', 'CI-Pre', 'MB-Post', 'Fusion',
         'CAFE-In-B', 'CAFE-Pre-B', 'CAFE-P', 'CAFE-In', 'CAFE-Pre']
LABEL = {'Fusion':'HL-GNN+RA', 'CAFE-In-B':r'CAFE-In$^{-B}$',
         'CAFE-Pre-B':r'CAFE-Pre$^{-B}$', 'CAFE-P':r'CAFE$^{-P}$',
         'CAFE-In':r'CAFE-In $\star$', 'CAFE-Pre':r'CAFE-Pre $\star$'}
DATASETS = {'brightkite':'Brightkite', 'pokec_n':'Pokec-n'}


def csv_text(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def manuscript_rows(path):
    result, dataset = {}, None
    reverse = {v:k for k,v in DATASETS.items()}
    methods = {LABEL.get(m,m):m for m in ORDER}
    for line in path.read_text().splitlines():
        if line.lstrip().startswith('%'):
            continue
        cells = line.split('&')
        if len(cells) != 8:
            continue
        first = cells[0].strip()
        if first:
            dataset = reverse.get(first)
        label = cells[2].strip()
        if dataset and label in methods:
            key = dataset, methods[label]
            if key in result:
                raise ValueError('Duplicate active manuscript row: '+str(key))
            numbers = evidence.CELL.findall(line)
            delta = re.search(r'([+-]?\d+)\\%', cells[4])
            if len(numbers) != 4:
                raise ValueError('Expected four mean/SD cells')
            result[key] = dict(cells=numbers, delta=delta.group(1)+r'\%' if delta else '--')
    if len(result) != 20:
        raise ValueError('Expected exactly ten rows per affected dataset')
    return result


def collect():
    shared = evidence.acceptance()
    rows = {}
    for ds in DATASETS:
        for method in ['HL-GNN','Fusion','CAFE-In','CAFE-Pre']:
            item = shared[ds,method]
            config = item['config']
            observations = item['rows']
            if method == 'Fusion':
                config = 'lambda_fair_0.0_beta_5.0_lam_0.0'
                observations = [json.loads(s) for s in evidence.g.read(item['folder']/'acceptance.jsonl').splitlines()]
                observations = [r for r in observations if r['config'] == config]
            if len(observations) != 10 or {r['run'] for r in observations} != set(range(10)):
                raise ValueError('Incomplete shared row')
            raw = evidence.g.js(item['folder']/'main_results.json')
            for r in observations:
                if any(r['main'][m] != raw['individual_runs'][r['run']][config][m][1]
                       for m in evidence.g.METRICS):
                    raise ValueError('Shared source/probe mismatch')
            rows[ds,method] = dict(status='complete', config=config,
                source=str(item['folder'].relative_to(ROOT)),
                metrics={m:evidence.summ([r['main'][m] for r in observations]) for m in evidence.g.METRICS})
    manifests = {p:evidence.g.js(p/'manifest.json') for p in [ORIGINAL,RESTART]}
    for manifest in manifests.values():
        missing.v1.t11.verify(manifest)
    dataset_splits = {}
    for index in range(12):
        root = RESTART if index < 4 else ORIGINAL
        task = manifests[root]['tasks'][index]
        folder = missing.task_folder(root, task)
        marker = root/f'complete_{index}.json'
        row = dict(status='pending', config=task['config'],
                   source=str(folder.relative_to(ROOT)),
                   job=f"{'107626' if index < 4 else '107614'}_{index}")
        if marker.is_file():
            if evidence.g.js(marker)['task'] != task:
                raise ValueError('Task completion mismatch')
            evidence.completed(folder)
            evidence.g.js(folder/'main_results.json')
            report = missing.validate_ablation(folder, task)
            if report != evidence.g.js(folder/'validation_report.json'):
                raise ValueError('Raw observations disagree with saved validation report')
            split = report['split_evidence']
            previous_split = dataset_splits.setdefault(task['dataset'], split)
            if split != previous_split:
                raise ValueError('Inconsistent fixed-split hashes across ablations')
            row.update(status='complete', metrics=report['metrics'],
                       noncloseness_exact=report['noncloseness_exact'])
        rows[task['dataset'],task['method']] = row
    return rows


def export(manuscript, date):
    old = manuscript_rows(manuscript)
    rows = collect()
    lines = [r'% Paste-ready ROWS ONLY; requires xcolor and booktabs.',
             '% Snapshot '+date+'. Replace the Brightkite and Pokec-n dataset blocks.',
             '% Black: completed exact-v1 results; mean +/- population SD over ten repetitions.',
             '% Red: unchanged manuscript values awaiting completed reruns; NOT interim averages.',
             '% Delta-Precision uses the unrounded corrected HL-GNN mean for completed rows.',
             '% Red Delta-Precision values remain old placeholders, not recalculated estimates.',
             '% Bold/underline: highest/second distinct UNROUNDED mean, only for complete panels.',
             '% Rankings are descriptive, not significance tests; rounded equal means need not tie.',
             '% Pokec-n emphasis is deferred while its CI-In/CI-Pre rows remain pending.',
             '% Fusion keeps Table 7 beta=5 on both datasets (not Pokec-n Table 11 beta=1).',
             '% Leave the Ego-FB, Oklahoma and UNC blocks unchanged.']
    flat = []
    for ds, dslabel in DATASETS.items():
        complete = all(rows[ds,m]['status'] == 'complete' for m in ORDER)
        ranks = {}
        if complete:
            for metric in evidence.g.METRICS:
                ranks[metric] = sorted({rows[ds,m]['metrics'][metric]['mean'] for m in ORDER}, reverse=True)
        lines += ['', '% '+dslabel+': '+('complete panel; rankings updated.' if complete else 'incomplete; no best/second-best emphasis.')]
        baseline = rows[ds,'HL-GNN']['metrics']['Precision@10']['mean']
        for i, method in enumerate(ORDER):
            row = rows[ds,method]
            pending = row['status'] != 'complete'
            if i in [1,5,8]:
                lines.append(r'\addlinespace[2pt]')
            category = {0:'Backbone',1:'Single comp.',5:'Two-stage',8:'Full pipeline'}.get(i,'')
            if pending:
                formatted = [evidence.g.red(f'{a} $\\pm$ {b}') for a,b in old[ds,method]['cells']]
                delta = evidence.g.red(old[ds,method]['delta'])
                lines.append('% PENDING '+row['job']+': '+method+'; all displayed values retained.')
            else:
                formatted = []
                for j, metric in enumerate(evidence.g.METRICS):
                    s = row['metrics'][metric]
                    body = evidence.fmt(s, 4 if j == 0 else 2)
                    if complete:
                        rank = ranks[metric].index(s['mean'])
                        if rank < 2:
                            body = (r'\textbf{' if rank == 0 else r'\underline{')+body+'}'
                    formatted.append(body)
                delta = '--' if method == 'HL-GNN' else f"{100*(row['metrics']['Precision@10']['mean']/baseline-1):+.0f}"+r'\%'
            cells = [dslabel if i == 0 else '', category, LABEL.get(method,method),
                     formatted[0], delta, *formatted[1:]]
            lines.append(' & '.join(cells)+r' \\')
            for j, metric in enumerate(evidence.g.METRICS):
                s = row.get('metrics',{}).get(metric,{})
                flat.append(dict(dataset=ds,method=method,status=row['status'],metric=metric,
                                 mean=s.get('mean',''),std=s.get('std',''),n=10 if not pending else '',
                                 displayed=(evidence.fmt(s,4 if j==0 else 2) if not pending else
                                            ' $\\pm$ '.join(old[ds,method]['cells'][j])),
                                 source=row['source'],config=row['config']))
    pending = [dict(dataset=ds,method=m,job=r['job']) for (ds,m),r in rows.items() if r['status']=='pending']
    provenance = dict(date=date, input_manuscript=str(manuscript),
                      input_sha256=missing.v1.t11.sha(manuscript), pending=pending,
                      rows=[dict(dataset=ds,method=m,**r) for (ds,m),r in rows.items()],
                      source_sha256=evidence.g.SOURCES,
                      mean_sd='population SD; n=10 completed repetitions',
                      ranking='highest/second distinct unrounded mean; deferred for incomplete panels',
                      no_partial_observations_used=True)
    return {f'table7_updated_rows_{date}.tex':'\n'.join(lines)+'\n',
            f'table7_updated_rows_{date}.csv':csv_text(flat),
            f'table7_updated_rows_{date}_sources.json':json.dumps(provenance,indent=2,allow_nan=False)+'\n'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manuscript', required=True, type=Path)
    parser.add_argument('--date', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'\d{8}',args.date):
        parser.error('--date must be YYYYMMDD')
    print(json.dumps(export(args.manuscript,args.date),allow_nan=False))
