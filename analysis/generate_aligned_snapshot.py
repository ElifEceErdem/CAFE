"""Read-only experiment snapshot; dated draft tables and comparison evidence.

Never reads supplementary materials or reruns experiments. Original paper values
come from existing transcriptions/comparison CSVs. Incomplete observations are
not imputed; all significance stars are withheld until the paired grid is full.
"""
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from replay_table11_aligned import METRICS, ARMS, verify
from generate_updated_tex_tables import DATASET_LABELS, TABLE11_ORIGINAL
from significance_tests import holm_correction

OUT = ROOT / 'comparison_tables'
T11 = ROOT / 'results/validation/20260905_163304_table11_table4_aligned_v3'
T15 = ROOT / 'results/validation/20260905_163304_table15_aligned_prepared'
PILOT = ROOT / 'results/validation/20260905_110459_table4_acceptance_pilot'
REPLAY = ROOT / 'results/validation/20260905_112222_table4_acceptance_replay'
DIAG = ROOT / 'results/validation/20260905_141132_cafe_pre_ten_run_diagnostic'
DATASETS = list(DATASET_LABELS)
ACCEPT = ['Degree_EG@10_full', 'Degree_EG@10_pw', 'Pagerank_EG@10_full',
          'Pagerank_EG@10_pw', 'Closeness_EG@10_full', 'Closeness_EG@10_tp']
REFS = ['HL-GNN', 'Fusion', 'RA']
SOURCES = {}
DATE = '20260907'


def read(path):
    path = Path(path)
    data = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return data.decode()


def js(path):
    return json.loads(read(path))


def red(s):
    return r'\textcolor{red}{' + s + '}'


def summary(values):
    return dict(mean=statistics.mean(values), std=statistics.pstdev(values), n=len(values))


def display(s, decimals=2):
    return f"{s['mean']:.{decimals}f} ± {s['std']:.{decimals}f}"


def historical(manifest, ds, arm):
    ref = manifest['references'][ds][arm]
    raw = js(ROOT / ref['source_file'])
    if arm == 'RA':
        rows = [{m: float(raw['results']['RA'][m]['values'][0]) for m in METRICS}]
    else:
        rows = [{m: float(r[ref['config']][m][1]) for m in METRICS} for r in raw['individual_runs']]
    return rows


def load_acceptance(manifest, paired_manifest):
    result, comparisons, gates = {}, [], []
    for ds in DATASETS:
        split_hashes = set()
        for arm in ARMS:
            folder = T11 / ds / arm.lower()
            if ds == 'oklahoma':
                folder = (PILOT / 'ra' if arm == 'RA' else DIAG / 'on_10' if arm == 'CAFE-Pre'
                          else REPLAY / arm.lower())
            records = [json.loads(s) for s in read(folder / 'acceptance.jsonl').splitlines()]
            expected_n = 1 if arm == 'RA' else 10
            config = (f'lambda_fair_0.0_beta_{float(paired_manifest["configs"][ds]["cafe_in"]["beta"])}_lam_0.0'
                      if arm == 'Fusion' else paired_manifest['references'][ds][arm]['config'])
            seen = set()
            for r in records:
                key = (r['run'], r['config'])
                assert key not in seen and 0 <= r['run'] < expected_n
                seen.add(key)
                assert r['dataset'] == ds and r['method'] == arm and r['k'] == 10
                assert r['full_matches_main'] and not r['predictions_saved']
                assert all(math.isfinite(v) for v in [*r['main'].values(), *r['acceptance'].values()])
                for a, b in zip(ACCEPT[::2], METRICS[1:]):
                    assert math.isclose(r['acceptance'][a], r['main'][b], rel_tol=1e-8, abs_tol=1e-6)
                for path, digest in r['source_sha256'].items():
                    assert manifest['frozen_sha256'][path] == digest
                split_hashes.add(r['ordered_train_test_sha256'])
            chosen = sorted((r for r in records if r['config'] == config), key=lambda r: r['run'])
            assert chosen and [r['run'] for r in chosen] == list(range(len(chosen)))
            # Original output independently confirms completed observations.
            if arm == 'RA':
                files = list((folder / 'main' / ds / 'baselines').glob('results_baselines_*.json'))
                assert len(files) == 1
                raw = js(files[0])
                assert all(chosen[0]['main'][m] == raw['results']['RA'][m]['values'][0] for m in METRICS)
            elif (folder / 'main_results.json').exists():
                raw = js(folder / 'main_results.json')
                for r in chosen:
                    assert all(r['main'][m] == raw['individual_runs'][r['run']][config][m][1] for m in METRICS)
            else:
                assert len(chosen) < expected_n, 'Complete probe without original output needs review'
            n = len(chosen)
            gate = 'control_no_table4_row'
            if arm != 'Fusion':
                old = historical(paired_manifest, ds, arm)
                exact = all(r['main'] == old[r['run']] for r in chosen)
                gate = ('exact' if exact else 'mismatch') if n == expected_n else ('partial_exact_so_far' if exact else 'partial_mismatch')
                for metric in METRICS:
                    before = summary([r[metric] for r in old])
                    after = summary([r['main'][metric] for r in chosen])
                    decimals = 4 if metric == METRICS[0] else 2
                    comparisons.append(dict(table=11, dataset=ds, method=arm, metric=metric,
                        table4_mean=before['mean'], table4_std=before['std'], new_mean=after['mean'],
                        new_std=after['std'], n=n, expected_n=expected_n, status=gate,
                        mean_delta=after['mean']-before['mean'],
                        max_raw_delta=max(abs(r['main'][metric]-old[r['run']][metric]) for r in chosen),
                        display_equal=display(before, decimals)==display(after, decimals),
                        table4_display=display(before, decimals), new_display=display(after, decimals)))
            gates.append(dict(dataset=ds, method=arm, n=n, expected_n=expected_n, status=gate))
            result[ds, arm] = dict(metrics={a: summary([r['acceptance'][a] for r in chosen]) for a in ACCEPT},
                                  n=n, expected_n=expected_n, status=gate, config=config)
        assert len(split_hashes) == 1, f'Table 11 cross-arm split mismatch: {ds}'
    return result, comparisons, gates


def load_paired(manifest):
    observations, comparisons, counts = {}, [], []
    for ds in DATASETS:
        observations[ds] = {arm: {} for arm in ARMS}
        distinct = set()
        for seed in range(10):
            evidence = []
            for arm in ARMS:
                folder = T15 / ds / f'seed_{seed}' / arm.lower()
                if not (folder / 'observation.json').exists():
                    continue
                r = js(folder / 'observation.json')
                split = js(folder / 'split_evidence.json')
                assert r['split_evidence'] == split and r['method'] == arm
                assert split['dataset'] == ds and split['training_seed'] == seed
                assert split['split_seed'] == (42 if ds == 'brightkite' else seed)
                assert {'train/edge', 'valid/edge', 'test/edge'} <= set(split['splits'])
                evidence.append(split)
                config = manifest['fusion_configs'][ds] if arm == 'Fusion' else manifest['references'][ds][arm]['config']
                assert r['config'] == config and all(math.isfinite(v) for v in r['metrics'].values())
                if arm == 'RA':
                    files = list((folder / 'main' / ds / 'baselines').glob('results_baselines_*.json'))
                    assert len(files) == 1
                    raw = js(files[0])
                    assert raw['runs_completed']['RA'] == 1
                    assert all(r['metrics'][k] == raw['results']['RA'][k]['values'][0] for k in METRICS)
                else:
                    raw = js(folder / 'main_results.json')
                    assert len(raw['individual_runs']) == 1
                    assert all(r['metrics'][k] == raw['individual_runs'][0][config][k][1] for k in METRICS)
                argv = js(folder / 'launch.json')['argv']
                assert argv[argv.index('--seed')+1] == str(seed)
                assert argv[argv.index('--epochs')+1] == ('300' if arm == 'RA' else '500')
                observations[ds][arm][seed] = r['metrics']
            assert evidence and all(e == evidence[0] for e in evidence)
            distinct.add(json.dumps(evidence[0]['splits'], sort_keys=True))
        assert len(distinct) == (1 if ds == 'brightkite' else 10)
        for arm in ARMS:
            current = list(observations[ds][arm].values())
            counts.append(dict(dataset=ds, method=arm, n=len(current), expected_n=10,
                               missing_seeds=','.join(str(s) for s in range(10) if s not in observations[ds][arm])))
            if arm == 'Fusion':
                continue
            old = historical(manifest, ds, arm)
            for metric in METRICS:
                before, after = summary([r[metric] for r in old]), summary([r[metric] for r in current])
                decimals = 4 if metric == METRICS[0] else 2
                comparisons.append(dict(table=15, dataset=ds, method=arm, metric=metric,
                    table4_mean=before['mean'], table4_std=before['std'], new_mean=after['mean'],
                    new_std=after['std'], n=len(current), expected_n=10,
                    status='different_split_protocol' if len(current)==10 else 'partial_different_split_protocol',
                    mean_delta=after['mean']-before['mean'], max_raw_delta='',
                    display_equal=display(before,decimals)==display(after,decimals),
                    table4_display=display(before,decimals), new_display=display(after,decimals)))
    return observations, comparisons, counts


def contrasts(obs):
    rows = []
    oldrows = list(csv.DictReader(io.StringIO(read(OUT / 'table15_paper_vs_reproduced.csv'))))
    original = {(r['reference'],r['method'],r['dataset'],r['metric']):r for r in oldrows}
    for ref in REFS:
        for arm in ['CAFE-In','CAFE-Pre']:
            for ds in DATASETS:
                seeds = sorted(set(obs[ds][arm]) & set(obs[ds][ref]))
                assert len(seeds) >= 2
                for metric in METRICS:
                    diffs = np.array([obs[ds][arm][s][metric]-obs[ds][ref][s][metric] for s in seeds])
                    mean, sd, n = float(diffs.mean()), float(diffs.std(ddof=1)), len(seeds)
                    width = float(stats.t.ppf(.975,n-1)*sd/math.sqrt(n))
                    p = float(stats.ttest_1samp(diffs,0).pvalue) if sd else (1.0 if mean==0 else 0.0)
                    assert math.isfinite(width) and math.isfinite(p)
                    old = original[ref,arm,ds,metric]
                    rows.append(dict(reference=ref,method=arm,dataset=ds,metric=metric,n=n,
                        seeds=','.join(map(str,seeds)),mean_diff=mean,ci95_halfwidth=width,p_raw=p,
                        paper_mean=float(old['paper_mean_diff']),paper_halfwidth=float(old['paper_ci95_halfwidth']),
                        paper_stars=old['paper_stars'],family_size=40))
    complete = all(len(obs[ds][arm])==10 for ds in DATASETS for arm in ARMS)
    for ref in REFS:
        family = [r for r in rows if r['reference']==ref]
        assert len(family)==40
        adjusted = holm_correction(np.array([r['p_raw'] for r in family])) if complete else [None]*40
        for r,p in zip(family,adjusted):
            r['p_holm'] = float(p) if p is not None else ''
            r['stars'] = ('***' if p<.001 else '**' if p<.01 else '*' if p<.05 else '') if p is not None else ''
            r['inference_status'] = 'complete' if complete else 'provisional_stars_withheld_all_families'
    return rows,complete


def tex11(data):
    lines = [r'% Snapshot 2026-09-07. Requires \usepackage{xcolor,booktabs}.',
             r'% Changed numeric values are red relative to the original paper transcription.',
             r'\begin{table*}[t]',r'\caption{EG under acceptance models. '+red(
             r'Draft from available Table-4-aligned reruns. Mean $\pm$ population SD; '
             r'$n=10$ training repetitions on a fixed split, except RA ($n=1$) and '
             r'Pokec-n HL-GNN ($n=9$, provisional). $\dagger$ marks completed rows whose '
             r'main results do not exactly reproduce Table~4; do not treat these as reconciled. '
             r'Full and PW/TP entries always come from the same recommendations. Degree retains '
             r'target-indegree recommendation events. PageRank and Closeness Full use '
             r'simple-undirected distinct-pair graphs; PageRank PW uses probability-union '
             r'edge weights; Closeness TP uses the unweighted true-positive graph. '
             r'HL-GNN+RA uses the fusion coefficient of CAFE-In (Pokec-n: $\beta=1$).')+'}',
             r'\label{tab:acceptance_models}',r'\centering\small',r'\begin{tabular}{llrrrrrr}',r'\toprule',
             r' & & \multicolumn{2}{c}{Deg EG@10} & \multicolumn{2}{c}{PR EG@10} & \multicolumn{2}{c}{Cl. EG@10} \\',
             r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}',
             r'Dataset & Method & Full & PW & Full & PW & Full & TP \\',r'\midrule']
    for i,ds in enumerate(DATASETS):
        if i: lines.append(r'\midrule')
        for j,arm in enumerate(ARMS):
            row=data[ds,arm]; name='HL-GNN+RA' if arm=='Fusion' else arm
            old=TABLE11_ORIGINAL[DATASET_LABELS[ds]][name]
            suffix=red(r'$^{\dagger}$') if row['status']=='mismatch' else ''
            if row['n']<row['expected_n']: suffix+=red(f" ($n={row['n']}$)")
            cells=[]
            for metric,(om,os) in zip(ACCEPT,old):
                s=row['metrics'][metric]; body=f"{s['mean']:.2f} $\\pm$ {s['std']:.2f}"
                cells.append(red(body) if (f'{om:.2f}',f'{os:.2f}')!=(f"{s['mean']:.2f}",f"{s['std']:.2f}") else body)
            lines.append(f'{DATASET_LABELS[ds] if j==0 else ""} & {name}{suffix} & '+' & '.join(cells)+r' \\')
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabular}',r'\end{table*}',''])


def tex15(rows,complete):
    lines=[r'% Snapshot 2026-09-07. Requires \usepackage{xcolor,booktabs,subcaption}.',
           r'\begin{table*}[t]',r'\caption{Paired contrasts for CAFE against (a) HL-GNN, '
           r'(b) HL-GNN+RA, and (c) RA. Entries are mean CAFE-minus-reference differences '
           r'$\pm$ 95\% paired $t$-interval half-widths; positive values favor CAFE. '+red(
           r'Draft: $n=10$ paired seeds except Pokec-n CAFE-Pre ($n=9$; seed 8 pending). '
           r'All significance stars are withheld until the full grid is complete; their absence '
           r'does not indicate nonsignificance. Final Holm correction uses three families of '
           r'40 contrasts each (one family per panel).')+r' Seeds 0--9 resample splits on the '
           r'four random-split datasets; Brightkite retains its temporal split. These are '
           r'not differences of the fixed-split means in Table~4. '+red(
           r'Table~4 model and ranking paths are retained. Degree uses target-indegree '
           r'recommendation events; PageRank and Closeness use simple-undirected distinct pairs. '
           r'The separately selected fusion coefficient is retained (Pokec-n: $\beta=5$ '
           r'versus $\beta=1$ in CAFE-In).')+'}',r'\label{tab:paired_contrasts}',r'\centering\small']
    assert not complete, 'Update complete caption before publishing a final snapshot'
    titles=['(a) vs. HL-GNN backbone','(b) vs. fusion-only control','(c) vs. RA']
    for ref,title in zip(REFS,titles):
        if ref!=REFS[0]: lines.append(r'\vspace{6pt}')
        lines += [r'\subcaption*{'+title+'}',r'\begin{tabular}{llrrrr}',r'\toprule',
                  r'Variant & Dataset & Prec@10 & Deg EG@10 & PR EG@10 & Cl. EG@10 \\',r'\midrule']
        for i,arm in enumerate(['CAFE-In','CAFE-Pre']):
            if i: lines.append(r'\midrule')
            for j,ds in enumerate(DATASETS):
                chosen=[next(r for r in rows if (r['reference'],r['method'],r['dataset'],r['metric'])==(ref,arm,ds,m)) for m in METRICS]
                cells=[]
                for r in chosen:
                    dec=4 if r['metric']==METRICS[0] else 2
                    body=f"${r['mean_diff']:+.{dec}f} \\pm {r['ci95_halfwidth']:.{dec}f}$"
                    changed=(f"{r['mean_diff']:+.{dec}f}",f"{r['ci95_halfwidth']:.{dec}f}",'')!=(f"{r['paper_mean']:+.{dec}f}",f"{r['paper_halfwidth']:.{dec}f}",r['paper_stars'])
                    cells.append(red(body) if changed else body)
                label=DATASET_LABELS[ds]+(red(f" ($n={chosen[0]['n']}$)") if chosen[0]['n']<10 else '')
                lines.append(f'{arm if j==0 else ""} & {label} & '+' & '.join(cells)+r' \\')
        lines += [r'\bottomrule',r'\end{tabular}']
    return '\n'.join(lines+[r'\end{table*}',''])


def csvtext(rows):
    out=io.StringIO(); writer=csv.DictWriter(out,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return out.getvalue()


def main():
    manifest,pm=js(T11/'authority.json'),js(T15/'authority.json')
    verify(manifest);verify(pm)
    t11,comparison11,gates=load_acceptance(manifest,pm)
    obs,comparison15,counts=load_paired(pm)
    paired,complete=contrasts(obs)
    outputs={f'table11_aligned_{DATE}_red.tex':tex11(t11),f'table15_aligned_{DATE}_red.tex':tex15(paired,complete),
             f'table11_aligned_{DATE}_status.csv':csvtext(gates),
             f'table15_aligned_{DATE}_counts.csv':csvtext(counts),
             f'table15_aligned_{DATE}_contrasts.csv':csvtext(paired),
             f'aligned_vs_table4_{DATE}.csv':csvtext(comparison11+comparison15)}
    lines=['# Aligned Tables 11 and 15: snapshot 2026-09-07','',
           'Draft, not final paper tables. No imputation or substitution of Table 4 Full values into a different acceptance run.',
           '', '## Table 11 consistency','', '| Dataset | RA | HL-GNN | CAFE-In | CAFE-Pre |', '|---|---|---|---|---|']
    for ds in DATASETS:
        lines.append('| '+DATASET_LABELS[ds]+' | '+' | '.join(t11[ds,a]['status'] for a in ['RA','HL-GNN','CAFE-In','CAFE-Pre'])+' |')
    lines += ['', 'Exact means all available raw metrics match the corresponding Table 4 observations; partial status does not certify the missing run. Fusion has no direct Table 4 row.',
              'All acceptance Full/main checks and cross-arm fixed-split fingerprints passed. The cause of completed replay mismatches is not established.',
              '', '### Table 11 differences that change displayed means or SDs','',
              '| Dataset | Method | Metric | Table 4 | New rerun | n |','|---|---|---|---|---|---|']
    for r in comparison11:
        if not r['display_equal']:
            lines.append(f"| {DATASET_LABELS[r['dataset']]} | {r['method']} | {r['metric']} | {r['table4_display']} | {r['new_display']} | {r['n']} |")
    lines += ['', '## Table 15 consistency','',
              'All available observations were checked against their raw runner outputs, launch seeds, selected configurations, and within-seed split evidence. Each random-split dataset has ten distinct splits; Brightkite has one fixed split.',
              'The current sample has 249/250 method-seed observations. Pokec-n CAFE-Pre seed 8 is missing; its three reference comparisons each use nine matched seeds. Other contrasts use ten. No unpaired subtraction or imputation was used.',
              'Method means need not equal Table 4: splits and training sequences differ. Matching configuration/metric paths does not prove that every numerical difference is caused solely by split variation.',
              'The paired table reports 95% t-interval half-widths using sample SD of differences. All Holm-adjusted p-values/stars are withheld until completion because each family includes all five datasets. Completing one seed can change stars beyond its own row.',
              '', '### Table 15 underlying method means versus Table 4 (not paired contrasts)','',
              '| Dataset | Method | Prec: T4 → T15 | Degree: T4 → T15 | PR: T4 → T15 | Closeness: T4 → T15 | n |',
              '|---|---|---|---|---|---|---|']
    for ds in DATASETS:
        for arm in ['RA','HL-GNN','CAFE-In','CAFE-Pre']:
            entries=[next(r for r in comparison15 if (r['dataset'],r['method'],r['metric'])==(ds,arm,k)) for k in METRICS]
            cells=[f"{r['table4_mean']:.{4 if i==0 else 2}f} → {r['new_mean']:.{4 if i==0 else 2}f}" for i,r in enumerate(entries)]
            lines.append('| '+DATASET_LABELS[ds]+' | '+arm+' | '+' | '.join(cells)+f" | {entries[0]['n']} |")
    lines += ['', '### Table 15 mean contrasts that change sign relative to Table 4','',
              'These are changes in mean direction, not significance claims. They must not be described as universally unchanged conclusions.',
              '', '| Dataset | Contrast | Metric | Difference of Table 4 means | Table 15 paired mean |',
              '|---|---|---|---|---|']
    lookup={(r['dataset'],r['method'],r['metric']):r for r in comparison15}
    for r in paired:
        if r['reference']=='Fusion': continue
        a=lookup[r['dataset'],r['method'],r['metric']]
        b=lookup[r['dataset'],r['reference'],r['metric']]
        before=a['table4_mean']-b['table4_mean']
        if before*r['mean_diff']<0:
            lines.append(f"| {DATASET_LABELS[r['dataset']]} | {r['method']} vs {r['reference']} | {r['metric']} | {before:+.5f} | {r['mean_diff']:+.5f} |")
    lines += ['', 'TeX row/column and brace checks passed; all 120 paired confidence intervals were independently checked. A TeX compiler is not installed here, so PDF compilation/layout was not tested.',
              '', '## Decision','',
              'Keep Table 4 authoritative. Use the new Table 11 TeX as a flagged working draft, not a fully reconciled replacement. Diagnose mismatching method/dataset arms separately rather than mixing Full from one run with PW/TP from another.',
              'Retain the Table 15 paired campaign; different fixed-split means alone are not a rerun criterion. Wait for the final observation, recompute intervals and all 40-contrast Holm families, then finalize the table.',
              'Previous table TeX/CSV files and all experiment evidence are preserved. Red numeric cells are compared with the original paper values in existing transcriptions, not with a newly read PDF.', '']
    outputs[f'aligned_consistency_{DATE}.md']='\n'.join(lines)
    for name in outputs:
        if (OUT/name).exists(): raise FileExistsError(f'Refusing to overwrite snapshot: {name}')
    read(Path(__file__))
    for name,content in outputs.items():
        with (OUT/name).open('x') as f:f.write(content)
    provenance=dict(captured_utc=datetime.now(timezone.utc).isoformat(),sources=SOURCES,
                    output_sha256={name:hashlib.sha256(content.encode()).hexdigest() for name,content in outputs.items()},
                    provisional=not complete, no_predictions_saved=True)
    with (OUT/f'aligned_sources_{DATE}.json').open('x') as f:json.dump(provenance,f,indent=2)
    print('Wrote:',*outputs,sep='\n')
    print('Table 11 gates:',gates)


if __name__=='__main__': main()
