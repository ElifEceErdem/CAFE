"""Versioned, isolated Tables 4/11/15 exact-closeness reevaluation campaign.

Original runner code, order, selected configurations and split protocols are
retained. A fresh process activates only the versioned closeness dispatcher.
No automatic table export, model selection, checkpoint or prediction saving.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'backbone'), str(ROOT)]
import replay_table11_aligned as t11
import prepare_table15_aligned as t15

OLD11 = ROOT / 'results/validation/20260905_163304_table11_table4_aligned_v3'
OLD15 = ROOT / 'results/validation/20260905_163304_table15_aligned_prepared'
DATASETS = ['brightkite', 'pokec_n']
VERSION = 'exact_generalized_v1_float32_positive_support'
NONCL = ['Precision@10', 'Equality_Gain@10', 'Pagerank_EG@10']
SELF = Path(__file__).resolve()


def write(path, data):
    t11.write_new(path, data)


def replace(argv, flag, value):
    argv[argv.index(flag) + 1] = str(value)


def wrapped(argv, folder):
    return [sys.executable, '-u', str(SELF), '--child-output', str(folder), '--', *argv]


def prepare(out):
    if subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() != 'repo-cleanup':
        raise ValueError('Use repo-cleanup')
    a = json.loads((OLD11/'authority.json').read_text())
    b = json.loads((OLD15/'authority.json').read_text())
    t11.verify(a)
    t11.verify(b)
    pilot = ROOT/'comparison_tables/closeness_exact_pilot_20260907'
    for ds in DATASETS:
        rows = [json.loads(s) for s in (pilot/ds/'ra_exact.jsonl').read_text().splitlines()]
        if rows[-1]['kind'] != 'complete' or not any(r.get('hashes_and_legacy_metrics_exact') for r in rows):
            raise ValueError('Exact pilot is not validated: '+ds)
    additions = [SELF, ROOT/'metrics/closeness_exact_v1.py', ROOT/'tests/test_closeness_exact_v1.py',
                 ROOT/'experiments/jobs/closeness_v1_gate.job', ROOT/'experiments/jobs/closeness_v1_gpu.job',
                 ROOT/'comparison_tables/table4_paper_vs_reproduced.csv']
    additions += [ROOT/'tests/test_closeness_v1_campaign.py',ROOT/'tests/fixtures/closeness_v1_runner.py']
    rows = list(csv.DictReader((ROOT/'comparison_tables/table4_paper_vs_reproduced.csv').open()))
    additions += [ROOT/p for p in {r['source_file'] for r in rows if r['dataset'] in DATASETS}]
    for manifest in [a, b]:
        manifest['frozen_sha256'].update({str(p.relative_to(ROOT)):t11.sha(p) for p in additions})
        manifest['closeness_metric_version'] = VERSION
        manifest['publication_update'] = False
        manifest['no_predictions_saved'] = True
    a['tasks'] = [dict(index=i*5+j, dataset=ds, arm=arm, runs=1 if arm=='RA' else 10)
                  for i,ds in enumerate(DATASETS) for j,arm in enumerate(t11.ARMS)]
    a['protocol'] = 'table4_fixed_split_exact_closeness_v1_with_acceptance'
    b['tasks'] = [dict(index=i*10+s,dataset=ds,seed=s,split_seed=42 if ds=='brightkite' else s)
                  for i,ds in enumerate(DATASETS) for s in range(10)]
    b['protocol'] = 'table4_paths_table15_split_resampled_exact_closeness_v1'
    baseline_tasks = [dict(index=0,dataset='brightkite',seed=42,runs=10)]
    baseline_tasks += [dict(index=i+1,dataset='pokec_n',seed=42+2*i,runs=2) for i in range(5)]
    out.mkdir(parents=True, exist_ok=False)
    for part in ['table11','table15','table4_baselines','gates']:
        (out/part).mkdir()
    write(out/'table11/authority.json', a)
    write(out/'table15/authority.json', b)
    write(out/'campaign.json', dict(metric_version=VERSION, datasets=DATASETS,
        baseline_tasks=baseline_tasks, frozen_sha256=a['frozen_sha256'],
        gate='RA same recommendations, non-Closeness parity, exact Full/main agreement and exact-pilot TP match',
        note='Other fixed-split numerical mismatches are flagged, not silently reconciled.',
        baseline_order='original complete per-dataset order retained within every seed shard',
        table15_protocol=b['protocol'], table15_families=3, contrasts_per_family=40,
        other_datasets='reuse only after provenance review; their exact metric path is unchanged',
        no_automatic_export=True, no_predictions_saved=True))
    plans=[]
    for task in a['tasks']:
        folder=out/'table11'/task['dataset']/task['arm'].lower()
        plans.append(dict(group='table11',task=task,argv=wrapped(t11.command(a,task['dataset'],task['arm'],folder),folder)))
    for task in b['tasks']:
        for arm in t11.ARMS:
            folder=out/'table15'/task['dataset']/f"seed_{task['seed']}"/arm.lower()
            plans.append(dict(group='table15',task=task,arm=arm,argv=wrapped(t15.child_command(b,task,arm,folder),folder)))
    for task in baseline_tasks:
        folder=out/'table4_baselines'/task['dataset']/f"seed_{task['seed']}"
        plans.append(dict(group='table4_baselines',task=task,argv=wrapped(baseline_command(a,task,folder),folder)))
    write(out/'planned_commands.json',plans)
    print(f'PREPARED {out}: 2 gates, 8 trained Table11 arms, 20 paired blocks, 6 baseline tasks; NOT submitted')


def baseline_command(manifest, task, folder):
    argv=t11.command(manifest,task['dataset'],'RA',folder)
    methods='random,pa,cn,aa,ra,lightgcn,ssnet,cross_link_bias,ltlp,tailsteak'
    if task['dataset']=='pokec_n':methods+=',fairadj,lba,moral'
    replace(argv,'--methods',methods)
    replace(argv,'--runs',task['runs'])
    replace(argv,'--seed',task['seed'])
    index=argv.index('--acceptance_probe_jsonl')
    del argv[index:index+2]
    return argv


def child(folder, argv):
    import torch
    import networkit as nk
    import core
    from metrics.closeness_exact_v1 import activated
    torch.set_num_threads(4)
    nk.setNumberOfThreads(4)
    folder.mkdir(parents=True,exist_ok=True)
    version=dict(metric_version=VERSION,networkit=nk.__version__,torch=torch.__version__,
        metric_source_sha256=t11.sha(ROOT/'metrics/closeness_exact_v1.py'),
        legacy_source_sha256=t11.sha(ROOT/'metrics/centrality_equality_gain.py'),
        no_predictions_saved=True, isolated_process=True, argv=argv)
    write(folder/'exact_version.json',version)
    original_info=core.create_experiment_info
    def info(*args,**kwargs):
        result=original_info(*args,**kwargs)
        result.update(closeness_metric_version=VERSION,closeness_sampling_active=False)
        return result
    core.create_experiment_info=info
    count=[0,0]
    def observer(record):
        count[0]+=1
        if not record['cache_hit']:
            count[1]+=1
            with (folder/'exact_metric_calls.jsonl').open('a') as handle:
                handle.write(json.dumps(record,allow_nan=False)+'\n')
    script=Path(argv[2])
    if not script.is_absolute():script=ROOT/'experiments'/script
    sys.argv=[str(script),*argv[3:]]
    os.chdir(ROOT/'experiments')
    with activated(observer):
        try:runpy.run_path(str(script),run_name='__main__')
        except SystemExit as exc:
            if exc.code not in [None,0]:raise
    if not count[0]:raise ValueError('Runner performed no corrected closeness evaluations')
    write(folder/'exact_process_complete.json',dict(metric_version=VERSION,calls=count[0],
        uncached_calls=count[1],invalid_metric_errors=False))


def require_gate(out, ds):
    gate=json.loads((out/'gates'/f'{ds}.json').read_text())
    if not gate['passed'] or gate['metric_version']!=VERSION:
        raise ValueError('Corrected Full/TP gate did not pass')


def table11_run(out,index):
    manifest=json.loads((out/'table11/authority.json').read_text())
    task=manifest['tasks'][index]
    folder=out/'table11'/task['dataset']/task['arm'].lower()
    original_execute=t11.execute
    original_summary=t11.summarize
    def execute(argv):return original_execute(wrapped(argv,folder))
    def summarize(*args):
        report=original_summary(*args)
        deltas=report.get('per_run_delta_vs_table4',[])
        report['closeness_metric_version']=VERSION
        report['non_closeness_exact_vs_table4']=all(d[m]==0 for d in deltas for m in NONCL) if deltas else None
        report['status']='exact_closeness_complete_noncloseness_review_required' if deltas and not report['non_closeness_exact_vs_table4'] else 'exact_closeness_complete'
        report['old_closeness_equality_not_required']=True
        return report
    t11.execute=execute
    t11.summarize=summarize
    try:t11.run(out/'table11',index)
    finally:
        t11.execute=original_execute
        t11.summarize=original_summary
    return folder


def gate(out,index):
    ds=DATASETS[index]
    folder=table11_run(out,index*5)
    actual=json.loads((folder/'acceptance.jsonl').read_text())
    reference=json.loads((OLD11/ds/'ra/acceptance.jsonl').read_text())
    assert actual['recommendations_sha256']==reference['recommendations_sha256']
    assert actual['ordered_train_test_sha256']==reference['ordered_train_test_sha256']
    assert all(actual['main'][m]==reference['main'][m] for m in NONCL)
    assert all(v==reference['acceptance'][m] for m,v in actual['acceptance'].items() if not m.startswith('Closeness'))
    assert actual['full_matches_main']
    pilot=[json.loads(s) for s in (ROOT/'comparison_tables/closeness_exact_pilot_20260907'/ds/'ra_exact.jsonl').read_text().splitlines()]
    expected=next(r['exact_ra_tp_eg'] for r in pilot if r['kind']=='pilot_eg')
    assert abs(actual['acceptance']['Closeness_EG@10_tp']-expected)<1e-8
    write(out/'gates'/f'{ds}.json',dict(passed=True,metric_version=VERSION,
        same_recommendations=True,noncloseness_exact=True,full_main_match=True,
        exact_pilot_tp_match=True,ra_results_reused_for_table11=True))


def paired_run(out,index):
    manifest=json.loads((out/'table15/authority.json').read_text())
    task=manifest['tasks'][index];ds=task['dataset']
    require_gate(out,ds)
    original=t15.execute
    def execute(argv):
        folder=Path(argv[argv.index('--evidence-dir')+1])
        return original(wrapped(argv,folder))
    t15.execute=execute
    try:t15.run(out/'table15',index)
    finally:t15.execute=original
    block=out/'table15'/ds/f"seed_{task['seed']}"
    comparison=[]
    for arm in t11.ARMS:
        current=json.loads((block/arm.lower()/'observation.json').read_text())
        previous=json.loads((OLD15/ds/f"seed_{task['seed']}"/arm.lower()/'observation.json').read_text())
        assert current['split_evidence']==previous['split_evidence']
        delta={m:current['metrics'][m]-previous['metrics'][m] for m in NONCL}
        comparison.append(dict(method=arm,noncloseness_delta=delta,exact=all(v==0 for v in delta.values())))
    write(block/'noncloseness_comparison.json',dict(metric_version=VERSION,rows=comparison,
        all_exact=all(r['exact'] for r in comparison),auto_export=False))


def baseline_run(out,index):
    campaign=json.loads((out/'campaign.json').read_text())
    task=campaign['baseline_tasks'][index];ds=task['dataset']
    require_gate(out,ds)
    folder=out/'table4_baselines'/ds/f"seed_{task['seed']}"
    folder.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((out/'table11/authority.json').read_text())
    argv=baseline_command(manifest,task,folder)
    write(folder/'launch.json',dict(task=task,argv=wrapped(argv,folder),metric_version=VERSION))
    rc=t11.execute(wrapped(argv,folder))
    if rc:raise SystemExit(rc)
    source_rows=list(csv.DictReader((ROOT/'comparison_tables/table4_paper_vs_reproduced.csv').open()))
    def norm(s):return ''.join(c for c in s.lower() if c.isalnum())
    mapping={norm(r['method']):r['source_file'] for r in source_rows if r['dataset']==ds}
    report=[]
    files=list((folder/'main'/ds).glob('*/*.json'))
    if not files:raise ValueError('No baseline output')
    for path in files:
        raw=json.loads(path.read_text())
        for method,metrics in raw['results'].items():
            expected_n=1 if method in ['PA','CN','AA','RA'] else task['runs']
            assert raw['runs_completed'][method]==expected_n
            previous=json.loads((ROOT/mapping[norm(method)]).read_text())['results'][method]
            comparable=expected_n!=1 or task['seed']==42 or method not in ['PA','CN','AA','RA']
            delta={}
            for metric in NONCL:
                values=metrics[metric]['values']
                if not all(__import__('math').isfinite(v) for v in values):raise ValueError('Non-finite baseline metrics')
                old=previous[metric]['values']
                start=0 if method in ['PA','CN','AA','RA'] else task['seed']-42
                if comparable:
                    assert len(old[start:start+expected_n])==len(values)==expected_n
                    delta[metric]=[a-b for a,b in zip(values,old[start:start+expected_n])]
            report.append(dict(method=method,source=mapping[norm(method)],comparable=comparable,
                noncloseness_delta=delta,exact=all(v==0 for d in delta.values() for v in d) if comparable else None))
    write(folder/'noncloseness_comparison.json',dict(rows=report,metric_version=VERSION,auto_export=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--group',choices=['gate','table11','table15','baselines'])
    parser.add_argument('--index',type=int)
    parser.add_argument('--child-output',type=Path)
    parser.add_argument('argv',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if args.child_output:
        argv=args.argv[1:] if args.argv[:1]==['--'] else args.argv
        child(args.child_output,argv)
        return
    if args.output_root is None:parser.error('--output-root required')
    out=args.output_root.resolve()
    if not out.is_relative_to(ROOT/'results/validation'):raise ValueError('Use results/validation')
    if args.prepare:prepare(out);return
    if args.index is None or args.group is None:parser.error('--group and --index required')
    campaign=json.loads((out/'campaign.json').read_text());t11.verify(campaign)
    if args.group=='gate':gate(out,args.index)
    else:
        import torch
        if not torch.cuda.is_available():raise RuntimeError('No provisioned GPU; refusing CPU training')
        if args.group=='table11':
            task_index=[1,2,3,4,6,7,8,9][args.index]
            require_gate(out,DATASETS[task_index//5])
            table11_run(out,task_index)
        elif args.group=='table15':paired_run(out,args.index)
        else:baseline_run(out,args.index)
    t11.verify(campaign)
    print(f'COMPLETE exact-v1 {args.group} index={args.index}; review before publication',flush=True)


if __name__=='__main__':main()
