"""Validate the completed paired campaign and print scoped apply_patch updates.

Keeps the 40-contrast families; flags large-graph closeness as unvalidated rather
than interpreting its computed significance. Never runs experiments or writes files.
"""
import argparse
import csv
from datetime import datetime, timezone
import difflib
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats
import generate_aligned_snapshot as g


def affected(row):
    return row['dataset'] in ['brightkite', 'pokec_n'] and row['metric'] == 'Closeness_EG@10'


def render(rows):
    lines = [r'% Completed seed grid, 2026-09-07. Requires \usepackage{xcolor,booktabs,subcaption}.',
        r'% Red: change from original paper transcription. Dagger: unvalidated large-graph closeness.',
        r'\begin{table*}[t]',
        r'\caption{Paired contrasts for CAFE against (a) HL-GNN, (b) HL-GNN+RA, and (c) RA. '
        r'Entries are mean CAFE-minus-reference differences $\pm$ 95\% paired $t$-interval half-widths; '
        r'positive values favor CAFE. '+g.red(
        r'All contrasts use $n=10$ paired seeds. Asterisks report Holm-adjusted two-sided paired '
        r'$t$-tests within each predeclared 40-contrast panel '
        r'(${}^{*}p<0.05$, ${}^{**}p<0.01$, ${}^{***}p<0.001$).')+
        r' Seeds 0--9 resample splits on Ego-FB, Oklahoma, Pokec-n, and UNC; Brightkite retains '
        r'its temporal split. Each pair shares its split and seed. These are not differences '
        r'of the fixed-split means in Table~4. '+g.red(
        r'Table~4 model and ranking paths are retained. Degree uses target-indegree recommendation '
        r'events; PageRank and Closeness use simple-undirected distinct pairs. The separately '
        r'selected fusion coefficient is retained (Pokec-n: $\beta=5$ versus $\beta=1$ in CAFE-In). '
        r'$\dagger$ marks Brightkite/Pokec-n Closeness outputs: the current approximation is '
        r'applied to disconnected graphs; these values and intervals are diagnostic only, and '
        r'their stars are withheld, not evidence of nonsignificance. All family corrections '
        r'must be recomputed after metric repair; family membership is unchanged.')+'}',
        r'\label{tab:paired_contrasts}', r'\centering\small']
    titles = ['(a) vs. HL-GNN backbone', '(b) vs. fusion-only control', '(c) vs. RA']
    for ref, title in zip(g.REFS, titles):
        if ref != g.REFS[0]:
            lines.append(r'\vspace{6pt}')
        lines += [r'\subcaption*{'+title+'}', r'\begin{tabular}{llrrrr}', r'\toprule',
                  r'Variant & Dataset & Prec@10 & Deg EG@10 & PR EG@10 & Cl. EG@10 \\', r'\midrule']
        for i, arm in enumerate(['CAFE-In', 'CAFE-Pre']):
            if i:
                lines.append(r'\midrule')
            for j, ds in enumerate(g.DATASETS):
                cells = []
                for metric in g.METRICS:
                    row = next(r for r in rows if (r['reference'], r['method'], r['dataset'], r['metric']) == (ref, arm, ds, metric))
                    dec = 4 if metric == g.METRICS[0] else 2
                    marker = r'\dagger' if affected(row) else row['display_stars']
                    body = f"${row['mean_diff']:+.{dec}f} \\pm {row['ci95_halfwidth']:.{dec}f}"
                    body += ('^{' + marker + '}' if marker else '') + '$'
                    changed = affected(row) or (f"{row['mean_diff']:+.{dec}f}", f"{row['ci95_halfwidth']:.{dec}f}", row['display_stars']) != (
                        f"{row['paper_mean']:+.{dec}f}", f"{row['paper_halfwidth']:.{dec}f}", row['paper_stars'])
                    cells.append(g.red(body) if changed else body)
                lines.append(f'{arm if j == 0 else ""} & {g.DATASET_LABELS[ds]} & '+' & '.join(cells)+r' \\')
        lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines + [r'\end{table*}', ''])


def validate(obs, rows):
    assert len(rows) == 120 and all(row['n'] == 10 for row in rows)
    for row in rows:
        a = np.array([obs[row['dataset']][row['method']][s][row['metric']] for s in range(10)])
        b = np.array([obs[row['dataset']][row['reference']][s][row['metric']] for s in range(10)])
        result = stats.ttest_rel(a, b)
        ci = stats.t.interval(.95, 9, loc=(a-b).mean(), scale=stats.sem(a-b))
        assert math.isclose(row['p_raw'], float(result.pvalue), rel_tol=1e-10, abs_tol=1e-15)
        assert math.isclose(row['mean_diff'], float(a.mean()-b.mean()), rel_tol=1e-9, abs_tol=1e-12)
        assert math.isclose(row['ci95_halfwidth'], float((ci[1]-ci[0])/2), rel_tol=1e-10, abs_tol=1e-12)
    for ref in g.REFS:
        family = sorted([row for row in rows if row['reference'] == ref], key=lambda row: row['p_raw'])
        running = 0
        for i, row in enumerate(family):
            running = max(running, min(1, (40-i)*row['p_raw']))
            assert math.isclose(running, row['p_holm'], rel_tol=1e-12, abs_tol=1e-15)


def build():
    manifest = g.js(g.T15 / 'authority.json')
    g.verify(manifest)
    obs, comparison, counts = g.load_paired(manifest)
    rows, complete = g.contrasts(obs)
    assert complete and sum(r['n'] for r in counts) == 250
    validate(obs, rows)
    for row in rows:
        row['display_stars'] = '' if affected(row) else row['stars']
        row['metric_status'] = 'unvalidated_disconnected_graph_approximation' if affected(row) else 'not_implicated_by_closeness_diagnostic'
        row['inference_status'] = 'metric_unvalidated_stars_withheld' if affected(row) else 'complete_grid_family_correction_pending_closeness_repair'
    outputs = {
        'table15_aligned_20260907_red.tex': render(rows),
        'table15_aligned_20260907_counts.csv': g.csvtext(counts),
        'table15_aligned_20260907_contrasts.csv': g.csvtext(rows),
        'table15_complete_vs_table4_20260907.csv': g.csvtext(comparison),
    }
    # Preserve the previous TeX while leaving the annotated consistency note intact.
    backup = 'archive/table15_aligned_20260907_n9_red.tex'
    outputs[backup] = ((g.OUT / backup) if (g.OUT / backup).exists()
                       else (g.OUT / 'table15_aligned_20260907_red.tex')).read_text()
    previous = list(csv.DictReader(io.StringIO((g.OUT / 'aligned_vs_table4_20260907.csv').read_text())))
    outputs['aligned_vs_table4_20260907.csv'] = g.csvtext([r for r in previous if r['table'] == '11'] + comparison)
    # A separate current interpretation avoids overwriting the user's annotations.
    lines = ['# Completed Table 15: consistency and interpretation — 2026-09-07', '',
        'All **250/250 method–seed observations** are complete: five datasets × five methods × ten seeds. '
        'Pokec-n CAFE-Pre seed 8 is now included. All 120 contrasts have ten matched pairs; no values were imputed.', '',
        'Updated table: [table15_aligned_20260907_red.tex](table15_aligned_20260907_red.tex). '
        'The previous nine-seed draft is preserved in [archive](archive/table15_aligned_20260907_n9_red.tex).', '',
        '## Are Tables 4 and 15 inconsistent?', '',
        '**Different numerical means are not, by themselves, an inconsistency.** The actual manifest says '
        '`table4_paths_table15_split_resampled`. “Aligned” describes the model/ranking paths and selected configurations, '
        'not identical Table 4 data splits or training sequences.', '',
        '| Table | Split protocol | What is summarized |', '|---|---|---|',
        '| 4 | One fixed split per dataset | Method mean ± population SD across training repetitions; deterministic heuristics once |',
        '| 15: Ego-FB, Oklahoma, Pokec-n, UNC | Ten distinct splits, seeds 0–9 | Within-seed CAFE-minus-reference difference; mean ± 95% paired t-interval half-width |',
        '| 15: Brightkite | One fixed temporal split | Paired seed repetitions; **not** temporal split resampling |', '',
        'For every completed Table 15 dataset/seed, all five methods have identical stored split evidence. '
        'Four datasets have ten distinct split fingerprints; Brightkite has one. Launch seeds, selected configurations, '
        'original runner outputs, and frozen inputs passed validation. The seed sets/fresh-process training sequences '
        'differ from Table 4; Brightkite deviations cannot be attributed to split resampling. RA is not trained, but '
        'its fallback recommendation filling can consume seed-dependent randomness.', '',
        'For each metric, Table 15 computes mean_s[CAFE(s) − reference(s)] within its own matched runs. '
        'This equals the difference of those Table 15 method means, not the difference of Table 4 means. '
        'The intervals describe paired contrasts, not method SDs, and are pointwise 95% intervals rather than '
        'Holm-adjusted simultaneous intervals. A pointwise interval excluding zero can coexist with no Holm star.', '',
        'That validates the pairing/protocol; it does **not** prove every discrepancy is caused solely by split variation '
        'or that conclusions are unchanged across protocols. Table 11 is a different case: it targets the fixed-split '
        'Table 4 replay, so its previously flagged mismatches remain unresolved.', '',
        '## Separate metric-validity issue', '',
        'The [closeness diagnostic](closeness_tp_diagnostic_20260907.md) found NaNs and extreme approximate control '
        'centralities on disconnected Brightkite/Pokec-n graphs. That concern affects Full closeness as well as TP, '
        'and therefore these Table 15 closeness contrasts. Resampling is not an explanation or a fix for invalid '
        'centrality inputs. Table 4 remains untouched, but its affected Closeness values also need review.', '',
        'The 12 affected Table 15 Closeness cells retain their computed values/intervals for traceability and carry '
        'a dagger; their stars are withheld. The CSV preserves computed p-values/stars separately from display stars '
        'and explicitly records metric status. The three original 40-contrast families have **not** been redefined. '
        'Their Holm corrections must be recomputed after any approved metric repair, including stars on other cells '
        'in the same families. The completed table is not yet an unqualified final publication table.', '',
        '## Updated interpretation (excluding the flagged large-graph Closeness cells)', '',
        '- Against HL-GNN, both variants improve Precision and PageRank EG in every dataset in these paired runs '
        '(current Holm p < 0.01). Degree EG improves in four datasets but **decreases on Ego-FB** for both variants.',
        '- CAFE-In versus the fusion-only control: Precision is significantly higher on Ego-FB, lower on Pokec-n, '
        'and not significant after Holm on Brightkite/Oklahoma/UNC. Degree EG is significantly higher on all five. '
        'PageRank EG is significantly higher except on Brightkite.',
        '- CAFE-Pre versus fusion: Precision is significantly lower on four datasets and not significant on Brightkite. '
        'PageRank EG is significantly higher on all five; Degree EG is lower on Pokec-n, higher on Ego-FB/Oklahoma/UNC, '
        'and not significant on Brightkite.',
        '- Against RA, neither variant supports a general claim of improving all fairness axes. On Pokec-n both '
        'variants improve Precision but reduce Degree EG; CAFE-Pre also reduces PageRank EG. On Brightkite neither '
        'variant has a Holm-significant Degree/PageRank advantage over RA; CAFE-Pre has lower Precision.',
        '- On the exact-closeness datasets, both variants reduce Closeness EG versus HL-GNN on Ego-FB. '
        'CAFE-In has no Holm-significant Closeness advantage over HL-GNN on Oklahoma, while CAFE-Pre does. '
        'Both variants improve Closeness versus HL-GNN on UNC.', '',
        'These significance summaries use the currently computed 40-contrast families and remain subject to '
        'recomputation after the flagged closeness metrics are repaired. Nonsignificance is not proof of equality.', '',
        '### Direction changes versus differences of Table 4 means', '',
        '| Dataset | Contrast | Metric | Table 4 mean difference | Completed Table 15 paired mean | Current Holm p | Interpretation |',
        '|---|---|---|---:|---:|---:|---|']
    lookup = {(r['dataset'], r['method'], r['metric']): r for r in comparison}
    for r in rows:
        if r['reference'] == 'Fusion':
            continue
        before = lookup[r['dataset'], r['method'], r['metric']]['table4_mean'] - lookup[r['dataset'], r['reference'], r['metric']]['table4_mean']
        if before * r['mean_diff'] < 0:
            flag = 'Metric unvalidated; do not interpret' if affected(r) else ('Not Holm-significant; no robust reversal established' if r['p_holm'] >= .05 else 'Protocol-dependent significant direction; discuss explicitly')
            lines.append(f"| {g.DATASET_LABELS[r['dataset']]} | {r['method']} vs {r['reference']} | {r['metric']} | {before:+.5f} | {r['mean_diff']:+.5f} | {r['p_holm']:.5g} | {flag} |")
    lines += ['', '### Newly completed Pokec-n CAFE-Pre contrasts', '',
        '| Reference | Precision | Degree EG | PageRank EG | Closeness EG (unvalidated) |', '|---|---:|---:|---:|---:|']
    for ref in g.REFS:
        selected = [next(r for r in rows if (r['dataset'], r['method'], r['reference'], r['metric']) == ('pokec_n', 'CAFE-Pre', ref, metric)) for metric in g.METRICS]
        cells = [f"{r['mean_diff']:+.{4 if i == 0 else 2}f} ± {r['ci95_halfwidth']:.{4 if i == 0 else 2}f}" for i, r in enumerate(selected)]
        lines.append('| ' + ref + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Suggested manuscript wording', '',
        '> Table 4 reports performance across training repetitions on a fixed dataset split. Table 15 separately '
        'evaluates paired CAFE-minus-reference contrasts across ten seed-specific splits on Ego-FB, Oklahoma, '
        'Pokec-n, and UNC, while Brightkite retains its fixed temporal split. Methods within each pair share '
        'the split and seed. Consequently, the Table 15 contrasts need not equal differences of Table 4 means. '
        'The paired results show dataset- and metric-dependent trade-offs rather than uniform dominance.', '',
        'Use that protocol wording now; finalize statements about Brightkite/Pokec-n Closeness only after '
        'the metric diagnostic and an approved correction. Do not launch new training merely to force equality '
        'between different-protocol means.', '',
        '## Evidence and validation', '',
        '- [All 120 contrasts, CIs, p-values and metric flags](table15_aligned_20260907_contrasts.csv)',
        '- [Complete method/seed counts](table15_aligned_20260907_counts.csv)',
        '- [Table 15 underlying means versus Table 4](table15_complete_vs_table4_20260907.csv)',
        '- `experiments/prepare_table15_aligned.py` records shared split evidence; '
        '`results/validation/20260905_163304_table15_aligned_prepared/authority.json` defines the protocol.',
        '- All 120 mean differences and t-intervals were cross-checked; paired t-test p-values and the '
        'three 40-member Holm corrections were independently checked. No publication significance is '
        'assigned to the flagged large-graph closeness cells.',
        '- The older annotated `aligned_consistency_20260907.md` is preserved, but its 249/250 count and '
        'pending-seed Table 15 section are superseded by this report. Its stored snapshot hash already '
        'differs from its annotated content; that user-edited file was not rewritten.', '']
    outputs['table15_complete_interpretation_20260907.md'] = '\n'.join(lines)
    outputs = {name: body.replace('\r\n', '\n') for name, body in outputs.items()}
    metadata = json.loads((g.OUT / 'aligned_sources_20260907.json').read_text())
    g.read(Path(__file__))
    metadata['sources'].update(g.SOURCES)
    # apply_patch preserves CRLF on untouched context lines in existing CSVs.
    # When the semantic output is already installed, hash its actual bytes.
    metadata['output_sha256'].update({name: hashlib.sha256(
        (g.OUT / name).read_bytes() if (g.OUT / name).exists() and (g.OUT / name).read_text() == body
        else body.encode()).hexdigest() for name, body in outputs.items()})
    metadata['table15_completion'] = dict(updated_utc=datetime.now(timezone.utc).isoformat(), observations=250,
        paired_contrasts=120, seeds_per_contrast=10, families=3, contrasts_per_family=40,
        metric_validation='Brightkite/Pokec-n Closeness unresolved; 12 cells flagged and stars withheld',
        protocol='split-resampled on four datasets; fixed temporal split for Brightkite')
    outputs['aligned_sources_20260907.json'] = json.dumps(metadata, indent=2) + '\n'
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=['tables', 'contrasts', 'provenance'], required=True)
    args = parser.parse_args()
    outputs = build()
    print('*** Begin Patch')
    for name, new in outputs.items():
        group = 'provenance' if name.endswith('.json') else 'contrasts' if name.endswith('_contrasts.csv') else 'tables'
        if group != args.group:
            continue
        path = g.OUT / name
        if not path.exists():
            print('*** Add File: ' + str(path))
            print('\n'.join('+' + line for line in new.splitlines()))
        else:
            old = path.read_text()
            if old == new:
                continue
            print('*** Update File: ' + str(path))
            for line in list(difflib.unified_diff(old.splitlines(), new.splitlines(), n=2))[2:]:
                print('@@' if line.startswith('@@') else line)
    print('*** End Patch')


if __name__ == '__main__':
    main()
