"""Print an apply_patch patch: recover five-decimal TP cells from snapshot sources.

No experiment is run and no source/result file is written by this script.
Retains the dated snapshot sample counts; refuses changed source files.
"""
import csv
import difflib
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'comparison_tables'
TARGET = OUT / 'table11_aligned_20260907_red.tex'
META = OUT / 'aligned_sources_20260907.json'
LABELS = {'Brightkite': 'brightkite', 'Ego-FB': 'ego-facebook',
          'Oklahoma': 'oklahoma', 'Pokec-n': 'pokec_n', 'UNC': 'unc'}
BETA = {'brightkite': 5, 'ego-facebook': 1, 'oklahoma': 5, 'pokec_n': 1, 'unc': 5}
METRICS = ['Degree_EG@10_full', 'Degree_EG@10_pw', 'Pagerank_EG@10_full',
           'Pagerank_EG@10_pw', 'Closeness_EG@10_full', 'Closeness_EG@10_tp']
CELL = re.compile(r'(-?\d+\.\d+)\s*\$\\pm\$\s*(-?\d+\.\d+)')


def source(ds, arm):
    base = ROOT / 'results/validation'
    if ds != 'oklahoma':
        return base / '20260905_163304_table11_table4_aligned_v3' / ds / arm.lower() / 'acceptance.jsonl'
    if arm == 'RA':
        return base / '20260905_110459_table4_acceptance_pilot/ra/acceptance.jsonl'
    if arm == 'CAFE-Pre':
        return base / '20260905_141132_cafe_pre_ten_run_diagnostic/on_10/acceptance.jsonl'
    return base / '20260905_112222_table4_acceptance_replay' / arm.lower() / 'acceptance.jsonl'


def patch(path, old, new):
    if old == new:
        return
    print('*** Update File: ' + str(path))
    diff = list(difflib.unified_diff(old.splitlines(), new.splitlines(), n=2))
    for line in diff[2:]:
        print('@@' if line.startswith('@@') else line)


def main():
    metadata = json.loads(META.read_text())
    with (OUT / 'table11_aligned_20260907_status.csv').open() as handle:
        counts = {(r['dataset'], r['method']): int(r['n']) for r in csv.DictReader(handle)}
    old = TARGET.read_text()
    lines = old.splitlines()
    dataset = None
    checked = 0
    for i, line in enumerate(lines):
        cells = CELL.findall(line)
        if len(cells) != 6:
            continue
        columns = line.split(' & ')
        if columns[0].strip():
            dataset = LABELS[columns[0].strip()]
        arm = columns[1].split('$')[0].split(' (')[0].strip()
        arm = 'Fusion' if arm == 'HL-GNN+RA' else arm
        path = source(dataset, arm)
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == metadata['sources'][str(path.relative_to(ROOT))], path
        rows = [json.loads(line) for line in raw.decode().splitlines()]
        if arm == 'Fusion':
            rows = [r for r in rows if float(r['config'].split('_beta_')[1].split('_')[0]) == BETA[dataset]]
        assert len(rows) == counts[dataset, arm]
        assert len({r['run'] for r in rows}) == len(rows)
        values = [[r['acceptance'][metric] for r in rows] for metric in METRICS]
        for j in range(5):
            assert cells[j] == (f'{statistics.mean(values[j]):.2f}', f'{statistics.pstdev(values[j]):.2f}')
        mean, sd = statistics.mean(values[5]), statistics.pstdev(values[5])
        assert tuple(float(v) for v in cells[5]) in [(float(f'{mean:.2f}'), float(f'{sd:.2f}')),
                                                      (float(f'{mean:.5f}'), float(f'{sd:.5f}'))]
        lines[i] = ' & '.join(columns[:-1]) + f' & {mean:.5f} $\\pm$ {sd:.5f}' + r' \\'
        checked += 1
    assert checked == 25
    old_caption = (r'\textbf{Bold} and \underline{underline} mark the highest and second-highest '
                   r'displayed mean in each dataset/column; ties share rank.')
    new_caption = (r'\textbf{Bold} and \underline{underline} mark the highest and second-highest '
                   r'displayed means in columns other than Closeness TP; shared ranks reflect rounding, not established equality. '
                   r'Closeness TP is shown to five decimal places without ranking emphasis. '
                   r'These extra digits describe the stored outputs, not validated estimator accuracy; '
                   r'large-graph closeness requires a disconnected-graph approximation diagnostic.')
    new = '\n'.join(lines).replace(old_caption, new_caption) + '\n'
    metadata['output_sha256'][TARGET.name] = hashlib.sha256(new.encode()).hexdigest()
    metadata['table11_presentation'] = ('Full coloring against Table 4 unchanged. Closeness TP means/population SD '
        'recovered from hash-verified snapshot JSONL at five decimals, without ranking emphasis. '
        'Other columns and sample counts unchanged. Formatter: analysis/format_table11_tp_precision.py')
    print('*** Begin Patch')
    patch(TARGET, old, new)
    patch(META, META.read_text(), json.dumps(metadata, indent=2) + '\n')
    print('*** End Patch')


if __name__ == '__main__':
    main()
