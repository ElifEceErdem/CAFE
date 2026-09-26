"""Validate scalar evidence and print an additive LaTeX/CSV apply_patch.

No experiment execution, original-file rewriting, or supplementary materials.
Withholds incomplete-provenance baseline rows instead of silently certifying them.
"""
import csv
import io
import json
import math
from pathlib import Path
import re
import statistics as st
import sys

import generate_aligned_snapshot as g
import update_table15_complete as u15
import generate_updated_tex_tables as tex
import generate_validation_comparisons as comparisons

sys.path.insert(0, str(g.ROOT/'experiments'))
import reevaluate_closeness_v1 as v1
import recover_closeness_v1 as recovery

ROOT, OUT = g.ROOT, g.OUT
BASE = ROOT/'results/validation/20260907_181811_closeness_exact_v1'
REC = ROOT/'results/validation/20260910_061800_closeness_v1_recovery'
DS = ['brightkite', 'pokec_n']
CELL = re.compile(r'(-?\d+\.\d+)\s*\$\\pm\$\s*(-?\d+\.\d+)')
CHANGES = []


def csvtext(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def summ(values):
    assert values and all(math.isfinite(v) for v in values)
    return dict(mean=st.mean(values), std=st.pstdev(values), n=len(values))


def fmt(s, dec=2, deterministic=False):
    return f"{s['mean']:.{dec}f}" if deterministic else f"{s['mean']:.{dec}f} $\\pm$ {s['std']:.{dec}f}"


def completed(folder):
    record = g.js(folder/'exact_process_complete.json')
    assert record['metric_version']==v1.VERSION and record['calls']>0 and not record['invalid_metric_errors']
    calls = [json.loads(l) for l in g.read(folder/'exact_metric_calls.jsonl').splitlines()]
    assert calls and all(r['finite'] and 0<=r['minimum']<=r['maximum']<=1.0000001 for r in calls)


def acceptance():
    manifest = g.js(BASE/'table11/authority.json')
    g.verify(manifest)
    data = {}
    for ds in DS:
        splits = set()
        for arm in g.ARMS:
            folder = (REC if ds=='pokec_n' and arm in ['Fusion','CAFE-In','CAFE-Pre'] else BASE)/'table11'/ds/arm.lower()
            completed(folder)
            report = v1.t11.summarize(manifest, ds, arm, folder)
            stored = g.js(folder/'replay_report.json')
            assert report['aggregate_main']==stored['aggregate_main']
            assert report['aggregate_acceptance']==stored['aggregate_acceptance']
            cfg = report.get('table11_config') or manifest['references'][ds][arm]['config']
            records = [json.loads(l) for l in g.read(folder/'acceptance.jsonl').splitlines()]
            rows = sorted((r for r in records if r['config']==cfg), key=lambda r:r['run'])
            assert len(rows)==(1 if arm=='RA' else 10)
            splits.update(r['ordered_train_test_sha256'] for r in rows)
            data[ds,arm] = dict(folder=folder, rows=rows, config=cfg,
                metrics={m:summ([r['acceptance'][m] for r in rows]) for m in g.ACCEPT},
                main={m:summ([r['main'][m] for r in rows]) for m in g.METRICS})
        assert len(splits)==1
    return data


def table11(data):
    before = {}
    ds = None
    for line in g.read(OUT/'table11_aligned_20260907_red.tex').splitlines():
        numbers = CELL.findall(line)
        if len(numbers)!=6:
            continue
        cells=line.split(' & ')
        if cells[0].strip():
            ds=next(k for k,v in g.DATASET_LABELS.items() if v==cells[0].strip())
        name=cells[1].split('$')[0].split(' (')[0].strip()
        before[ds,'Fusion' if name=='HL-GNN+RA' else name]=numbers
    ref = g.js(BASE/'table11/authority.json')['references']
    lines = [r'% Replace ONLY the Brightkite and Pokec-n blocks in Table 11.',
        r'% Full six-column rows: do not mix acceptance values with historical Full values.',
        r'% Requires xcolor, booktabs. TP uses five decimals and no ranking emphasis.',
        r'% Red: Full mean/SD differs from the existing authoritative Table 4 at displayed precision.',
        r'% Bold/underline: highest/second distinct unrounded mean, descriptive only.',
        r'% Caption: exact generalized Closeness on Brightkite/Pokec-n; mean +/- population SD;',
        r'% n=10 fixed-split repetitions, RA n=1. Full and PW/TP use identical recommendations.',
        r'% Control is train union test; Gini retains positive-support convention.',
        r'% Remove the n=9 and invalid-approximation warnings for THESE two blocks only.',
        r'% Existing Ego-FB CAFE-In / UNC CAFE-Pre consistency issues are not resolved here.']
    for ds in DS:
        lines += ['', '% '+g.DATASET_LABELS[ds], r'\midrule']
        for i,arm in enumerate(g.ARMS):
            cells=[]
            for j,m in enumerate(g.ACCEPT):
                s=data[ds,arm]['metrics'][m];dec=5 if j==5 else 2
                body=fmt(s,dec)
                rank=sorted({data[ds,a]['metrics'][m]['mean'] for a in g.ARMS},reverse=True).index(s['mean'])
                if j!=5 and rank<2:
                    body=(r'\textbf{' if rank==0 else r'\underline{')+body+'}'
                if j in [0,2,4] and arm!='Fusion':
                    old=summ([r[g.METRICS[1+j//2]] for r in ref[ds][arm]['historical_runs']])
                    if fmt(s)!=fmt(old):body=g.red(body)
                old=before[ds,arm][j]
                CHANGES.append(dict(table='11',dataset=ds,method=arm,reference='',metric=m,
                    previous=f'{old[0]} +/- {old[1]}',replacement=fmt(s,dec),
                    display_changed=(f'{s["mean"]:.{dec}f}',f'{s["std"]:.{dec}f}')!=old,
                    source=str(data[ds,arm]['folder'].relative_to(ROOT))))
                cells.append(body)
            name='HL-GNN+RA' if arm=='Fusion' else arm
            lines.append(f'{g.DATASET_LABELS[ds] if i==0 else ""} & {name} & '+' & '.join(cells)+r' \\')
    return '\n'.join(lines)+'\n'


def baseline_load(folder):
    values={}
    for path in sorted((folder/'main').glob('*/*/*.json')):
        raw=g.js(path)
        assert raw['experiment_info']['closeness_metric_version']==v1.VERSION
        for method, metrics in raw['results'].items():
            assert method not in values
            vals={m:metrics[m]['values'] for m in g.METRICS}
            assert all(len(v)==raw['runs_completed'][method] and all(math.isfinite(x) for x in v) for v in vals.values())
            values[method]=(vals,path)
    assert values
    return values


def table4(data):
    before=list(csv.DictReader(io.StringIO(g.read(OUT/'table4_paper_vs_reproduced.csv'))))
    by={(r['dataset'],r['method'],r['metric']):r for r in before}
    results={}; sources={};blocked=[]
    for ds in DS:
        for arm in ['RA','HL-GNN','CAFE-In','CAFE-Pre']:
            name='HLGNN' if arm=='HL-GNN' else arm
            results[ds,name]=data[ds,arm]['main'];sources[ds,name]=[data[ds,arm]['folder']]
    folder=BASE/'table4_baselines/brightkite/seed_42'
    completed(folder)
    for method,(vals,path) in baseline_load(folder).items():
        assert all(len(v)==(1 if method.lower() in recovery.DETERMINISTIC else 10) for v in vals.values())
        if method=='RA':
            assert all(summ(v)==results['brightkite','RA'][m] for m,v in vals.items())
        results['brightkite',method]={m:summ(v) for m,v in vals.items()};sources['brightkite',method]=[path]
    first=BASE/'table4_baselines/pokec_n/seed_42';completed(first)
    for method,(vals,path) in baseline_load(first).items():
        if method.lower() in recovery.DETERMINISTIC:
            results['pokec_n',method]={m:summ(v) for m,v in vals.items()};sources['pokec_n',method]=[path]
    # These methods all require seed 49 from a shard lacking its success marker.
    suspect=BASE/'table4_baselines/pokec_n/seed_48'
    assert not (suspect/'exact_process_complete.json').exists(), 'Completion evidence changed; re-audit instead of retaining this hold'
    saved=baseline_load(suspect)
    assert all(len(vals['Precision@10'])==(1 if method.lower() in recovery.DETERMINISTIC else 2) for method,(vals,_) in saved.items())
    blocked=[r['method'] for r in before if r['dataset']=='pokec_n' and r['metric']==tex.METRICS[0] and ('pokec_n',r['method']) not in results]
    lines=[r'% Table 4 replacement ROWS; existing files are not modified.',
        r'% Replace every displayed metric in each supplied row from its coherent rerun.',
        r'% Red compares displayed numbers with table4_paper_vs_reproduced.csv (your screenshot may differ).',
        r'% Brightkite: complete 13-row block; regenerate ranking from new means.',
        r'% Pokec-n: only seven validated rows below; nine baseline rows WITHHELD.',
        r'% Pokec-n ranking emphasis intentionally withheld until all rows are certified.',
        r'% Add to caption: exact generalized Closeness for Brightkite/Pokec-n;',
        r'% other datasets retain their original exact implementation; Gini uses positive support.']
    for ds in DS:
        lines+=['','% '+g.DATASET_LABELS[ds],r'\midrule']
        for category,methods in tex.TABLE4_GROUPS:
            present=[m for m in methods if (ds,m) in results]
            if not present:continue
            for i,method in enumerate(present):
                vals=results[ds,method];cells=[]
                for j,m in enumerate(g.METRICS):
                    s=vals[m];old=by[ds,method,tex.METRICS[j]];dec=4 if j==0 else 2
                    deterministic=method.lower() in recovery.DETERMINISTIC
                    raw=fmt(s,dec,deterministic);body=raw
                    if ds=='brightkite':
                        rank=sorted({v[m]['mean'] for (d,_),v in results.items() if d==ds},reverse=True).index(s['mean'])
                        if rank<2:body=(r'\textbf{' if rank==0 else r'\underline{')+body+'}'
                    oldtext=fmt(dict(mean=float(old['reproduced_mean']),std=float(old['reproduced_std'] or 0)),dec,deterministic)
                    changed=raw!=oldtext
                    if changed:body=g.red(body)
                    CHANGES.append(dict(table='4',dataset=ds,method=method,reference='',metric=m,
                        previous=oldtext,replacement=raw,display_changed=changed,
                        source=';'.join(str(p.relative_to(ROOT)) for p in sources[ds,method])))
                    cells.append(body)
                lines.append(f'{category if i==0 else ""} & {tex.METHOD_LABELS.get(method,method)} & '+' & '.join(cells)+r' \\')
        if ds=='pokec_n':
            lines.append('% WITHHELD (do not retain their old closeness as corrected): '+', '.join(blocked))
    return '\n'.join(lines)+'\n',blocked


def paired():
    manifest=g.js(g.T15/'authority.json');g.verify(manifest)
    obs,_,counts=g.load_paired(manifest)
    exact_manifest=g.js(BASE/'table15/authority.json');g.verify(exact_manifest)
    for ds in DS:
        distinct=set()
        for seed in range(10):
            block=(REC if ds=='pokec_n' and seed>=5 else BASE)/'table15'/ds/f'seed_{seed}'
            pack=g.js(block/'paired_observations.json');assert pack['pairing_verified']
            assert pack['task']['seed']==seed and pack['task']['dataset']==ds
            evidence=[]
            for arm in g.ARMS:
                folder=Path(pack.get('sources',{}).get(arm,str(block/arm.lower())))
                completed(folder)
                r=v1.t15.collect(exact_manifest,pack['task'],arm,folder)
                assert r==g.js(folder/'observation.json')
                assert r==next(x for x in pack['methods'] if x['method']==arm)
                oldsplit=g.js(g.T15/ds/f'seed_{seed}'/arm.lower()/'split_evidence.json')
                assert r['split_evidence']==oldsplit
                g.read(folder/'main_results.json') if arm!='RA' else [g.read(p) for p in (folder/'main'/ds/'baselines').glob('*.json')]
                cfg=manifest['fusion_configs'][ds] if arm=='Fusion' else manifest['references'][ds][arm]['config']
                assert r['config']==cfg
                obs[ds][arm][seed]=r['metrics'];evidence.append(r['split_evidence'])
            assert all(e==evidence[0] for e in evidence)
            distinct.add(json.dumps(evidence[0]['splits'],sort_keys=True))
        assert len(distinct)==(1 if ds=='brightkite' else 10)
    rows,complete=g.contrasts(obs);assert complete
    u15.validate(obs,rows)
    oldrows=list(csv.DictReader(io.StringIO(g.read(OUT/'table15_aligned_20260907_contrasts.csv'))))
    old={(r['dataset'],r['method'],r['reference'],r['metric']):r for r in oldrows}
    for row in rows:
        prior=old[row['dataset'],row['method'],row['reference'],row['metric']]
        dec=4 if row['metric']==g.METRICS[0] else 2
        previous=(f"{float(prior['mean_diff']):+.{dec}f}",f"{float(prior['ci95_halfwidth']):.{dec}f}",prior['display_stars'])
        after=(f"{row['mean_diff']:+.{dec}f}",f"{row['ci95_halfwidth']:.{dec}f}",row['stars'])
        row['display_changed']=previous!=after or (row['dataset'] in DS and row['metric']==g.METRICS[3])
        row['previous_stars']=prior['display_stars']
        CHANGES.append(dict(table='15',dataset=row['dataset'],method=row['method'],reference=row['reference'],metric=row['metric'],
            previous=' +/- '.join(previous[:2])+previous[2],replacement=' +/- '.join(after[:2])+after[2],
            display_changed=row['display_changed'],source='exact campaign + recovery for Brightkite/Pokec-n; original paired campaign for other datasets'))
    lines=[r'% Complete replacement Table 15, 2026-09-13. Requires xcolor,booktabs,subcaption.',
        r'% Red: changes from table15_aligned_20260907_red.tex; all 120 Holm tests recomputed.',
        r'\begin{table*}[t]',r'\caption{Paired contrasts for CAFE against (a) HL-GNN, (b) HL-GNN+RA, and (c) RA. '
        r'Entries are mean CAFE-minus-reference differences $\pm$ pointwise 95\% paired $t$-interval half-widths over $n=10$ seeds. '
        r'Positive values favor CAFE. Asterisks report Holm-adjusted two-sided paired $t$-tests within each predeclared '
        r'40-contrast panel (${}^{*}p<0.05$, ${}^{**}p<0.01$, ${}^{***}p<0.001$). '
        r'Seeds 0--9 resample splits on Ego-FB, Oklahoma, Pokec-n and UNC; Brightkite retains its temporal split. '
        r'Each pair shares its split and seed. These are not differences of fixed-split means in Table~4. '+g.red(
        r'Brightkite and Pokec-n use the corrected exact generalized Closeness evaluation; all three Holm families have been recomputed. ')+
        r'The selected model/ranking paths are retained. Degree uses target-indegree recommendation events; PageRank and Closeness use '
        r'simple-undirected distinct pairs. The separately selected fusion coefficient is retained '
        r'(Pokec-n: $\beta=5$ versus $\beta=1$ in CAFE-In).}',r'\label{tab:paired_contrasts}',r'\centering\small']
    for ref,title in zip(g.REFS,['(a) vs. HL-GNN backbone','(b) vs. fusion-only control','(c) vs. RA']):
        lines += [r'\subcaption*{'+title+'}',r'\begin{tabular}{llrrrr}',r'\toprule',
            r'Variant & Dataset & Prec@10 & Deg EG@10 & PR EG@10 & Cl. EG@10 \\',r'\midrule']
        for i,arm in enumerate(['CAFE-In','CAFE-Pre']):
            if i:lines.append(r'\midrule')
            for j,ds in enumerate(g.DATASETS):
                cells=[]
                for m in g.METRICS:
                    r=next(x for x in rows if (x['dataset'],x['method'],x['reference'],x['metric'])==(ds,arm,ref,m))
                    dec=4 if m==g.METRICS[0] else 2
                    body=f"${r['mean_diff']:+.{dec}f} \\pm {r['ci95_halfwidth']:.{dec}f}"
                    body+=('^{'+r['stars']+'}' if r['stars'] else '')+'$'
                    cells.append(g.red(body) if r['display_changed'] else body)
                lines.append(f'{arm if j==0 else ""} & {g.DATASET_LABELS[ds]} & '+' & '.join(cells)+r' \\')
        lines += [r'\bottomrule',r'\end{tabular}',r'\vspace{6pt}']
    lines.append(r'\end{table*}')
    return '\n'.join(lines)+'\n',rows


def main():
    g.verify(g.js(REC/'recovery.json'))
    data=acceptance()
    a=table11(data);b,blocked=table4(data);c,contrasts=paired()
    outputs={'table11_exact_20260913_rows.tex':a,'table4_exact_20260913_rows.tex':b,
             'table15_exact_20260913_red.tex':c,'exact_row_changes_20260913.csv':csvtext(CHANGES),
             'table15_exact_20260913_contrasts.csv':csvtext(contrasts)}
    changed_other=[r for r in contrasts if r['dataset'] not in DS and r['display_changed']]
    summary=dict(table11_replacement_rows=10,table4_validated_rows=20,
        table4_withheld_pokec_methods=blocked,table15_observations=250,table15_contrasts=120,
        table15_changed_other_dataset_cells=changed_other,
        table7_fusion_betas={d:comparisons.FUSION_BETA[d] for d in DS},
        sources=g.SOURCES,originals_untouched=True)
    outputs['exact_row_sources_20260913.json']=json.dumps(summary,indent=2)+'\n'
    print('*** Begin Patch')
    for name,body in outputs.items():
        path=OUT/name
        if path.exists():
            assert path.read_text()==body,'Refusing to overwrite differing output: '+name
            continue
        print('*** Add File: '+str(path))
        print('\n'.join('+'+line for line in body.splitlines()))
    print('*** End Patch')


if __name__=='__main__':main()
