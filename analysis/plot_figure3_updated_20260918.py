"""Additive Figure 3 export from validated, coherent Tables 4/7 observations.

Run with the hl_gnn environment; no training, GPU, prediction cache, or old
result modification. Existing output directories are refused. Numerical values
come from raw result files, not the rounded historical plotting matrix.
"""
import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile

os.environ.setdefault('MPLBACKEND', 'Agg')
_cache = tempfile.TemporaryDirectory(prefix='cafe-figure3-')
os.environ.setdefault('MPLCONFIGDIR', _cache.name)

import matplotlib.pyplot as plt
import pandas as pd
import export_table7_progress_rows as t7
import plot_precision_eg_tradeoff_table_match as style

e = t7.evidence
g = e.g
c = e.comparisons
ROOT = e.ROOT
METRICS = [m[0] for m in c.METRICS]
LABELS = [m[1] for m in c.METRICS]
DISPLAY = {'brightkite':'Brightkite', 'ego-facebook':'Ego-FB',
           'oklahoma':'Oklahoma', 'pokec_n':'Pokec-n', 'unc':'UNC'}
ALIASES = {'HL-GNN':'HLGNN', 'Fusion':'HLGNN+RA'}
SEEDS8 = [42, 43, 44, 45, 46, 47, 50, 51]


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-11)


def assemble():
    rows = {}

    def add(ds, method, metrics, sources, config='', seeds=None):
        method = ALIASES.get(method, method)
        assert method not in style.EXCLUDE_METHODS
        counts = {s['n'] for s in metrics.values()}
        assert len(counts) == 1 and counts <= {1, 8, 10}
        assert set(metrics) == set(METRICS)
        key = ds, method
        if key in rows:
            for m in METRICS:
                assert metrics[m]['n'] == rows[key]['metrics'][m]['n']
                assert all(close(metrics[m][s], rows[key]['metrics'][m][s]) for s in ['mean','std']), key
            rows[key]['sources'].extend(sources)
            return
        rows[key] = dict(dataset=ds, method=method, metrics=metrics,
                         sources=sources, config=config, seeds=seeds)

    # Small-graph panels: use explicit frozen comparison sources, not newest-file
    # selection. Recompute population summaries from the underlying observations.
    configs = c._best_configs()
    g.read(ROOT/'configs/best_configs.csv')
    for table, specs in [(4,c.TABLE4_SPECS),(7,c.TABLE7_SPECS)]:
        path = ROOT/f'comparison_tables/table{table}_paper_vs_reproduced.csv'
        records = list(csv.DictReader(io.StringIO(g.read(path))))
        specmap = {s[0]:s for s in specs}
        grouped = {}
        for r in records:
            if r['dataset'] not in ['ego-facebook','oklahoma','unc']:
                continue
            grouped.setdefault((r['dataset'], r['method']), []).append(r)
        for (ds, method), block in grouped.items():
            assert len(block) == 4 and {r['metric'] for r in block} == set(LABELS)
            assert all(r['evidence_status'] == 'complete' for r in block)
            assert len({r['source_file'] for r in block}) == 1
            _, _, result_key, marker, expected = specmap[method]
            config = c._resolve_config(configs, method, ds, marker)
            source = ROOT/block[0]['source_file']
            g.read(source)
            extracted = c._extract_stats(source, result_key, config)
            assert extracted and extracted['count'] == expected
            metrics = {}
            for m, label in zip(METRICS, LABELS):
                stat = extracted['stats'][m]
                summary = e.summ(stat['values'])
                assert summary['n'] == expected
                record = next(r for r in block if r['metric'] == label)
                for s in ['mean','std']:
                    assert close(summary[s], stat[s]), (ds,method,m,s)
                    assert close(summary[s], float(record['reproduced_'+s])), (ds,method,m,s)
                metrics[m] = summary
            add(ds, method, metrics, [str(source.relative_to(ROOT))], config or '')

    # Corrected large-graph main methods and ALL completed ablation configurations.
    for (ds, method), row in t7.collect().items():
        assert row['status'] == 'complete', (ds, method)
        add(ds, method, {m:{**row['metrics'][m], 'n':10} for m in METRICS},
            [row['source']], row['config'])

    folder = e.BASE/'table4_baselines/brightkite/seed_42'
    e.completed(folder)
    for method, (values, path) in e.baseline_load(folder).items():
        expected = 1 if method.lower() in e.recovery.DETERMINISTIC else 10
        assert all(len(v) == expected for v in values.values())
        add('brightkite',method,{m:e.summ(v) for m,v in values.items()},
            [str(path.relative_to(ROOT))])

    # Uniform n=8 baseline subset already used by the current Table 4 draft.
    # Failed 48/49 shard is excluded in full, not selected per method/value.
    manifest = g.js(e.REC/'recovery.json')
    e.recovery.v1.t11.verify(manifest)
    parts = []
    folder = e.BASE/'table4_baselines/pokec_n/seed_42'
    e.completed(folder)
    parts.append((e.baseline_load(folder), [42,43]))
    for index in [8,9,10]:
        task = manifest['tasks'][index]
        assert g.js(e.REC/f'complete_{index}.json')['task'] == task
        if task.get('reuse_seed') is not None:
            reused = e.BASE/f"table4_baselines/pokec_n/seed_{task['reuse_seed']}"
            assert e.recovery.baseline_values(reused,1) == task['reused_results']
            parts.append((e.baseline_load(reused), [task['reuse_seed']]))
        folder = e.REC/f"table4_baselines/pokec_n/seed_{task['seed']}"
        e.completed(folder)
        report = g.js(folder/'recovery_baseline_report.json')
        assert report['task'] == task
        assert report['results'] == e.recovery.baseline_values(
            folder, task['runs'], task.get('reuse_seed') is None)
        parts.append((e.baseline_load(folder), list(range(task['seed'],task['seed']+task['runs']))))
    for method in sorted(e.recovery.METHOD_KEYS):
        deterministic = method.lower() in e.recovery.DETERMINISTIC
        selected = parts[:1] if deterministic else parts
        values = {m:[] for m in METRICS}
        sources, seeds = [], []
        for part, partseeds in selected:
            vals, path = part[method]
            assert all(len(v) == (1 if deterministic else len(partseeds)) for v in vals.values())
            for m in METRICS:
                values[m].extend(vals[m])
            sources.append(str(path.relative_to(ROOT)))
            seeds.extend(partseeds[:1] if deterministic else partseeds)
        assert seeds == ([42] if deterministic else SEEDS8)
        add('pokec_n', method, {m:e.summ(v) for m,v in values.items()}, sources, seeds=seeds)

    # Check both updated Table 4 blocks at their actual displayed precision.
    draft = g.read(ROOT/'comparison_tables/table4_paste_ready_20260913.tex')
    ds, checked = None, 0
    for line in draft.splitlines():
        if '% (a) BRIGHTKITE' in line:
            ds = 'brightkite'
        if '% (d) POKEC-N' in line:
            ds = 'pokec_n'
        cells = line.split('&')
        if line.lstrip().startswith('%') or len(cells) != 6:
            continue
        name = cells[1].strip().split('$')[0]
        method = {'Random':'random','HL-GNN':'HLGNN','Cross-LinkBias':'Cross-Link Bias',
                  'Tail-STEAK':'TailSteak','TailSTEAK':'TailSteak'}.get(name,name)
        row = rows[ds,method]
        for j, (cell, m) in enumerate(zip(cells[2:],METRICS)):
            nums = re.findall(r'-?\d+\.\d+',cell)
            stat = row['metrics'][m]
            expected = [f"{stat['mean']:.{4 if j==0 else 2}f}"]
            if stat['n'] > 1:
                expected.append(f"{stat['std']:.{4 if j==0 else 2}f}")
            assert nums == expected, (ds,method,m,nums,expected)
        checked += 1
    assert checked == 29
    assert len(rows) == 109
    assert {ds:sum(d==ds for d,m in rows) for ds in DISPLAY} == {
        'brightkite':20,'ego-facebook':20,'oklahoma':23,'pokec_n':23,'unc':23}
    assert sum(r['metrics'][METRICS[0]]['n']==8 for r in rows.values()) == 9
    return rows


def render(rows, out):
    # Historical matrix is read ONLY for stable method ordering and colours.
    historic = pd.read_csv(io.StringIO(g.read(ROOT/'results_repro/tables/table1_precision_equality_gain.csv')))
    methods = [m for m in historic.Method.unique() if m not in style.EXCLUDE_METHODS]
    assert set(methods) == {m for ds,m in rows} and len(methods) == 23
    cmap = plt.get_cmap('tab20')
    colors = {m:cmap(i%20) for i,m in enumerate(m for m in methods if m not in style.FIXED_COLORS)}
    colors.update(style.FIXED_COLORS)
    flat = []
    for ds in DISPLAY:
        for method in methods:
            if (ds,method) not in rows:
                continue
            r = rows[ds,method]
            item = dict(Dataset=ds,Method=method,n=r['metrics'][METRICS[0]]['n'],
                        sources=';'.join(r['sources']),config=r['config'])
            for m,label in zip(METRICS,['Precision',*style.EG_METRICS]):
                for stat in ['mean','std']:
                    item[label+'_'+stat.title()] = r['metrics'][m][stat]
            flat.append(item)
    df = pd.DataFrame(flat)
    assert all(math.isfinite(v) for col in df if col.endswith(('_Mean','_Std')) for v in df[col])
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'DejaVu Sans'})
    fig, axes = plt.subplots(5,3,figsize=(15.6,20.2))
    fig.subplots_adjust(left=.105,right=.81,bottom=.09,top=.955,wspace=.24,hspace=.19)
    fronts = []
    for i,ds in enumerate(DISPLAY):
        block = df[df.Dataset==ds].reset_index(drop=True)
        best = style.find_best_per_column(block)
        for j,metric in enumerate(style.EG_METRICS):
            ax = axes[i,j]
            style.plot_single_panel(ax,block,metric,colors,best,
                show_ylabel=False,show_xlabel=i==4,label_fontsize=12,
                tick_fontsize=9,annotation_fontsize=7)
            # Keep old marker styles but avoid a stack of tiny overlapping labels;
            # the shared legend identifies every method. Label only axis maxima.
            for annotation in list(ax.texts):
                annotation.remove()
            for method in dict.fromkeys([best['Precision'],best[metric]]):
                point = block[block.Method==method].iloc[0]
                label = {'random':'Random','HLGNN':'HL-GNN'}.get(method,method)
                right = method == best['Precision']
                ax.annotate(label,(point.Precision_Mean,point[metric+'_Mean']),
                    xytext=(-6,7) if right else (6,7),textcoords='offset points',
                    ha='right' if right else 'left',fontsize=8,color='#333333')
            ax.margins(x=.08,y=.13)
            if i==0:
                ax.set_title(metric,fontsize=13,fontweight='bold',pad=11)
            if j==0:
                ax.set_ylabel(DISPLAY[ds],fontsize=13,fontweight='bold',labelpad=15)
            xs,ys = block.Precision_Mean.to_numpy(),block[metric+'_Mean'].to_numpy()
            selected = style._pareto_front_2d(xs,ys)
            independent = {a for a in range(len(block)) if not any(
                xs[b]>=xs[a] and ys[b]>=ys[a] and (xs[b]>xs[a] or ys[b]>ys[a])
                for b in range(len(block)) if b!=a)}
            assert set(selected)==independent
            fronts.extend(dict(dataset=ds,metric=metric,method=block.iloc[a].Method,
                               precision=float(xs[a]),eg=float(ys[a])) for a in selected)
    handles = style._build_legend_handles(methods,colors)
    for handle in handles:
        handle.set_label({'random':'Random','HLGNN':'HL-GNN','HLGNN+RA':'HL-GNN+RA',
                          'TailSteak':'TailSTEAK','Best per metric':'Axis maximum*'}.get(handle.get_label(),handle.get_label()))
    fig.legend(handles=handles,loc='center left',bbox_to_anchor=(.825,.53),fontsize=10,
               frameon=True,labelspacing=.65,handletextpad=.65)
    fig.suptitle('Precision@10 versus Equality Gain@10',fontsize=17,fontweight='bold',y=.989)
    fig.text(.105,.022,'Fixed-split mean ± SD; n=10, except deterministic heuristics (n=1) and nine Pokec-n baselines (n=8).\n'
             'Exact corrected Closeness for Brightkite/Pokec-n. Dashed lines: two-objective Pareto frontiers.\n'
             '*Diamond: highest precision or EG; CAFE variants retain their own marker.',fontsize=10,va='bottom')
    out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'figure3_updated.pdf',metadata={'Title':'Figure 3 — updated Tables 4 and 7','Subject':'Fixed-split precision / EG, exact repaired large-graph closeness'})
    fig.savefig(out/'figure3_updated.png',dpi=140)
    plt.close(fig)
    df.to_csv(out/'figure3_points.csv',index=False)
    pd.DataFrame(fronts).to_csv(out/'figure3_frontiers.csv',index=False)
    caption = r'''% Replace the existing Figure 3 caption. No significance is implied by frontier membership.
\caption{Precision@10 versus Degree, PageRank, and Closeness EG@10 for all compared methods and CAFE configurations included in Tables~4 and~7. Points and error bars show fixed-split means $\pm$ population SD over 10 repetitions, except deterministic heuristics (one run) and nine Pokec-n baselines (eight repetitions: seeds 42--47 and 50--51). These nine baselines are Random, LightGCN, SSNet, Cross-Link Bias, FairAdj, LBA, MORAL, LTLP, and TailSTEAK. Brightkite and Pokec-n use the corrected exact generalized Closeness evaluation with positive-support Gini. Dashed curves connect two-objective Pareto frontiers in each panel, computed from unrounded means; they are not a single four-objective frontier. Diamonds mark methods attaining the highest precision or EG in a panel; highlighted CAFE variants retain their distinct markers.}
'''
    (out/'figure3_caption.tex').write_text(caption)
    # Include the exporter and plotting helper in the recorded source hashes.
    for path in [Path(__file__),Path(style.__file__),Path(t7.__file__),Path(e.__file__)]:
        g.read(path)
    manifest = dict(snapshot='2026-09-18',rows=list(rows.values()),
        source_sha256=g.SOURCES,panels=15,points_per_metric=109,total_points=327,
        excluded_methods=sorted(style.EXCLUDE_METHODS),
        baseline_pokec_seeds=SEEDS8,baseline_pokec_excluded_seeds=[48,49],
        degree_semantics='target-indegree directed recommendation events',
        pagerank_and_closeness_full_graph='simple-undirected distinct pairs',
        closeness_version_large_graphs=e.v1.VERSION,
        fusion_beta={ds:c.FUSION_BETA[ds] for ds in DISPLAY},
        verification='Raw observations, hash manifests, completion reports, n, means, population SD; 29 current Table 4 rows match at displayed precision; shared rows deduplicated only after equality check; independent 2D Pareto check.',
        render_command='PYTHONDONTWRITEBYTECODE=1 /var/scratch/eerdem/miniconda3/envs/hl_gnn/bin/python analysis/plot_figure3_updated_20260918.py --out-dir NEW_DIRECTORY')
    (out/'figure3_sources.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(out),method_dataset_rows=len(df),panels=15,
                         methods=len(methods),n8_rows=int((df.n==8).sum())),indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=ROOT/'comparison_tables/figure3_updated_20260918')
    parser.add_argument('--replace-generated', action='store_true',
                        help='Replace only this exporter\'s own six generated artifacts')
    args = parser.parse_args()
    if args.out_dir.exists():
        expected = {'figure3_updated.pdf','figure3_updated.png','figure3_points.csv',
                    'figure3_frontiers.csv','figure3_caption.tex','figure3_sources.json'}
        if not args.replace_generated or {p.name for p in args.out_dir.iterdir()} != expected:
            raise FileExistsError('Refusing to overwrite existing figure artifacts: '+str(args.out_dir))
    render(assemble(),args.out_dir)
