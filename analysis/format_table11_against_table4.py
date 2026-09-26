"""Print an apply_patch patch for Table 11 presentation; never change numbers.

Red means a displayed Full mean/SD differs from Table 4. PW/TP and the fusion
control have no Table 4 counterparts and are not colored. Rankings use displayed
means within each dataset/column; ties share rank. Reads the dated snapshot only.
"""
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'comparison_tables'
TARGET = OUT / 'table11_aligned_20260907_red.tex'
LABELS = {'Brightkite': 'brightkite', 'Ego-FB': 'ego-facebook', 'Oklahoma': 'oklahoma',
          'Pokec-n': 'pokec_n', 'UNC': 'unc'}
FULL = {0: 'Equality_Gain@10', 2: 'Pagerank_EG@10', 4: 'Closeness_EG@10'}
CELL = re.compile(r'(-?\d+\.\d+)\s*\$\\pm\$\s*(-?\d+\.\d+)')


def uncolor(s):
    prefix = r'\textcolor{red}{'
    while prefix in s:
        start = s.index(prefix)
        pos, depth = start + len(prefix), 1
        end = pos
        while depth:
            if s[end] == '{': depth += 1
            if s[end] == '}': depth -= 1
            end += 1
        s = s[:start] + s[pos:end-1] + s[end:]
    return s


def render():
    old = TARGET.read_text()
    with (OUT / 'aligned_vs_table4_20260907.csv').open() as handle:
        reference = {(r['dataset'], r['method'], r['metric']): r
                     for r in csv.DictReader(handle) if r['table'] == '11'}
    lines = old.splitlines()
    grouped = {}
    dataset = None
    for i, line in enumerate(lines):
        numbers = CELL.findall(line)
        if len(numbers) != 6: continue
        columns = line.split(' & ')
        label = columns[0].strip()
        if label: dataset = LABELS[label]
        method = uncolor(columns[1])
        name = method.split('$')[0].split(' (')[0].strip()
        grouped.setdefault(dataset, []).append((i, columns, name, numbers))
    assert len(grouped) == 5 and all(len(rows) == 5 for rows in grouped.values())
    colored = 0
    for dataset, rows in grouped.items():
        ranks = [sorted({Decimal(row[3][j][0]) for row in rows}, reverse=True) for j in range(6)]
        for i, columns, method, numbers in rows:
            cells = []
            for j, (mean, sd) in enumerate(numbers):
                body = f'{mean} $\\pm$ {sd}'
                rank = ranks[j].index(Decimal(mean))
                if rank == 0: body = r'\textbf{' + body + '}'
                elif rank == 1: body = r'\underline{' + body + '}'
                if j in FULL and method != 'HL-GNN+RA':
                    ref = reference[dataset, method, FULL[j]]
                    assert (mean, sd) == tuple(ref['new_display'].split(' ± '))
                    if (Decimal(mean), Decimal(sd)) != tuple(Decimal(v) for v in ref['table4_display'].split(' ± ')):
                        body = r'\textcolor{red}{' + body + '}'
                        colored += 1
                cells.append(body)
            lines[i] = columns[0] + ' & ' + uncolor(columns[1]) + ' & ' + ' & '.join(cells) + r' \\'
    for i, line in enumerate(lines):
        if line.startswith('% Changed numeric'):
            lines[i] = '% Red: displayed Full mean/SD differs from Table 4; PW/TP and fusion are not directly comparable.'
        elif line.startswith(r'\caption{'):
            line = uncolor(line)
            lines[i] = line[:-1] + (
                r' Red entries differ from Table~4 at displayed precision (mean or SD). '
                r'PW/TP columns and HL-GNN+RA have no direct Table~4 counterpart and are not colored. '
                r'\textbf{Bold} and \underline{underline} mark the highest and second-highest '
                r'displayed mean in each dataset/column; ties share rank.}'
            )
    new = '\n'.join(lines) + '\n'
    assert CELL.findall(new) == CELL.findall(old), 'Numeric values must remain unchanged'
    assert new.count(r'\textcolor{red}{') == colored
    return old, new


if __name__ == '__main__':
    old, new = render()
    print('*** Begin Patch')
    print('*** Update File: ' + str(TARGET))
    print('@@')
    print('\n'.join('-' + line for line in old.splitlines()))
    print('\n'.join('+' + line for line in new.splitlines()))
    provenance = OUT / 'aligned_sources_20260907.json'
    metadata = json.loads(provenance.read_text())
    key = TARGET.name
    print('*** Update File: ' + str(provenance))
    print('@@')
    print('-    ' + json.dumps(key) + ': ' + json.dumps(metadata['output_sha256'][key]) + ',')
    print('+    ' + json.dumps(key) + ': ' + json.dumps(hashlib.sha256(new.encode()).hexdigest()) + ',')
    print('@@')
    print('   "provisional": true,')
    print('+  "table11_presentation": "User-requested Table 4 displayed-value coloring and within-dataset/column ranking; numeric snapshot unchanged. Formatting: analysis/format_table11_against_table4.py",')
    print('*** End Patch')
