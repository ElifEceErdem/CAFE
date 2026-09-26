"""Quota recovery in a new directory; never overwrite old campaign evidence.

Table 15 resumes between independent methods. Baselines resume between seeded
repetitions, retaining their full stochastic-method order. Table 11 run_cafe
has continuous RNG state and no checkpoint, so failed arms restart in full.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import reevaluate_closeness_v1 as v1

ROOT = v1.ROOT
OLD = ROOT/'results/validation/20260907_181811_closeness_exact_v1'
METRICS = v1.t11.METRICS
DETERMINISTIC = {'pa', 'cn', 'aa', 'ra'}
METHOD_KEYS = {'random', 'PA', 'CN', 'AA', 'RA', 'LightGCN', 'SSNet',
               'Cross-Link Bias', 'LTLP', 'TailSteak', 'FairAdj', 'LBA', 'MORAL'}


def read(path):
    return json.loads(path.read_text())


def complete(folder):
    r = read(folder/'exact_process_complete.json')
    if r['metric_version'] != v1.VERSION or r['invalid_metric_errors'] or r['calls'] <= 0:
        raise ValueError('Invalid completed process: '+str(folder))


def baseline_values(folder, runs, deterministic=True):
    found = {}
    for path in sorted((folder/'main/pokec_n').glob('*/*.json')):
        raw = read(path)
        if raw['experiment_info']['closeness_metric_version'] != v1.VERSION:
            raise ValueError('Baseline metric version mismatch')
        for method, metrics in raw['results'].items():
            expected = 1 if method.lower() in DETERMINISTIC else runs
            if method in found or raw['runs_completed'][method] != expected:
                raise ValueError('Duplicate/incomplete baseline: '+method)
            for m in METRICS:
                values = metrics[m]['values']
                if len(values) != expected or not all(math.isfinite(x) for x in values):
                    raise ValueError('Invalid baseline values')
            found[method] = str(path)
    expected_methods = METHOD_KEYS if deterministic else METHOD_KEYS-{'PA','CN','AA','RA'}
    if set(found) != expected_methods:
        raise ValueError('Missing/unexpected baseline methods: '+str(set(found)^expected_methods))
    return found


def baseline_command(manifest, task, folder):
    argv = v1.baseline_command(manifest, dict(dataset='pokec_n', seed=task['seed'], runs=task['runs']), folder)
    if task.get('reuse_seed') is not None:
        methods = argv[argv.index('--methods')+1].split(',')
        v1.replace(argv, '--methods', ','.join(m for m in methods if m not in DETERMINISTIC))
    return argv


def prepare(out):
    if out.exists():
        raise FileExistsError(out)
    a = read(OLD/'table11/authority.json')
    b = read(OLD/'table15/authority.json')
    v1.t11.verify(a)
    v1.t11.verify(b)
    for ds in v1.DATASETS:
        v1.require_gate(OLD, ds)
    tasks = []
    for index in [7,8,9]:
        arm = a['tasks'][index]['arm']
        if (OLD/'table11/pokec_n'/arm.lower()/'exact_process_complete.json').exists():
            raise ValueError('Table 11 task completed since inventory; re-plan')
        tasks.append(dict(group='table11', original_index=index, arm=arm,
            reason='Full arm restart: continuous training RNG state was not checkpointed'))
    reused_files = []
    for seed in range(5,10):
        original_task = b['tasks'][10+seed]
        sources = {}
        for arm in v1.t11.ARMS:
            folder = OLD/'table15/pokec_n'/f'seed_{seed}'/arm.lower()
            if (folder/'exact_process_complete.json').exists():
                complete(folder)
                observation = v1.t15.collect(b, original_task, arm, folder)
                if observation != read(folder/'observation.json'):
                    raise ValueError('Saved observation does not match original output')
                sources[arm] = str(folder)
                reused_files.extend(p for p in folder.rglob('*') if p.is_file())
        if len(sources) == 5:
            raise ValueError('Table 15 block completed since inventory; re-plan')
        tasks.append(dict(group='table15', seed=seed, original_index=10+seed, reuse=sources))
    for start in [44,46]:
        folder = OLD/f'table4_baselines/pokec_n/seed_{start}'
        values = baseline_values(folder, 1)
        reused_files.extend(Path(p) for p in set(values.values()))
        reused_files.extend([folder/'launch.json', folder/'exact_version.json'])
        tasks.append(dict(group='baselines', seed=start+1, runs=1, reuse_seed=start,
                          reused_results=values,
                          reason='First seeded repetition saved; continue next seed in original remaining-method order'))
    if (OLD/'table4_baselines/pokec_n/seed_50').exists():
        raise ValueError('Seed-50 task has appeared; inspect before recovery')
    tasks.append(dict(group='baselines', seed=50, runs=2, reuse_seed=None))
    frozen = {**a['frozen_sha256'], **b['frozen_sha256']}
    for path in [Path(__file__).resolve(), ROOT/'experiments/jobs/closeness_v1_recovery.job',
                 ROOT/'tests/test_closeness_v1_recovery.py', *reused_files]:
        frozen[str(path.relative_to(ROOT))] = v1.t11.sha(path)
    manifest = dict(original_campaign=str(OLD), metric_version=v1.VERSION, tasks=tasks,
                    frozen_sha256=frozen, untouched_active_job='107251_4',
                    publication_update=False, predictions_saved=False)
    out.mkdir(parents=True)
    (out/'table11').mkdir()
    v1.write(out/'table11/authority.json', a)
    v1.write(out/'recovery.json', manifest)
    print(json.dumps(tasks, indent=2))


def execute(argv, folder):
    v1.write(folder/'recovery_launch.json', dict(argv=argv, metric_version=v1.VERSION,
        job_id=os.environ.get('SLURM_JOB_ID'), array_id=os.environ.get('SLURM_ARRAY_TASK_ID')))
    rc = v1.t11.execute(v1.wrapped(argv, folder))
    if rc:
        raise SystemExit(rc)
    complete(folder)


def paired(out, task):
    manifest = read(OLD/'table15/authority.json')
    original_task = manifest['tasks'][task['original_index']]
    block = out/'table15/pokec_n'/f"seed_{task['seed']}"
    block.mkdir(parents=True, exist_ok=False)
    observations, sources = [], {}
    for arm in v1.t11.ARMS:
        if arm in task['reuse']:
            folder = Path(task['reuse'][arm])
            complete(folder)
            print('REUSE completed '+str(folder), flush=True)
        else:
            folder = block/arm.lower()
            folder.mkdir()
            argv = v1.t15.child_command(manifest, original_task, arm, folder)
            print(f'RUN missing Table 15 seed={task["seed"]} {arm}', flush=True)
            execute(argv, folder)
        observation = v1.t15.collect(manifest, original_task, arm, folder)
        if arm not in task['reuse']:
            v1.write(folder/'observation.json', observation)
        observations.append(observation)
        sources[arm] = str(folder)
    if not all(o['split_evidence'] == observations[0]['split_evidence'] for o in observations):
        raise ValueError('Recovered methods have inconsistent splits')
    comparisons = []
    for obs in observations:
        old = read(v1.OLD15/'pokec_n'/f"seed_{task['seed']}"/obs['method'].lower()/'observation.json')
        if obs['split_evidence'] != old['split_evidence']:
            raise ValueError('Historical split mismatch')
        delta = {m:obs['metrics'][m]-old['metrics'][m] for m in v1.NONCL}
        comparisons.append(dict(method=obs['method'], noncloseness_delta=delta,
                                exact=all(x==0 for x in delta.values())))
    v1.write(block/'paired_observations.json', dict(task=original_task, methods=observations,
        sources=sources, pairing_verified=True, protocol=manifest['protocol']))
    v1.write(block/'noncloseness_comparison.json', dict(rows=comparisons,
        all_exact=all(r['exact'] for r in comparisons), metric_version=v1.VERSION, auto_export=False))


def run(out, index):
    manifest = read(out/'recovery.json')
    v1.t11.verify(manifest)
    v1.require_gate(OLD, 'pokec_n')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Allocated GPU required')
    # A write+fsync probe tests the actual user quota, not just df free space.
    with tempfile.TemporaryFile(dir=out) as handle:
        handle.write(b'quota recovery preflight\n'*1024)
        handle.flush()
        os.fsync(handle.fileno())
    task = manifest['tasks'][index]
    if (out/f'complete_{index}.json').exists():
        raise FileExistsError('Recovery task already complete')
    if task['group'] == 'table11':
        folder = v1.table11_run(out, task['original_index'])
        complete(folder)
    elif task['group'] == 'table15':
        paired(out, task)
    else:
        folder = out/'table4_baselines/pokec_n'/f"seed_{task['seed']}"
        folder.mkdir(parents=True, exist_ok=False)
        argv = baseline_command(read(OLD/'table11/authority.json'), task, folder)
        execute(argv, folder)
        values = baseline_values(folder, task['runs'], task.get('reuse_seed') is None)
        v1.write(folder/'recovery_baseline_report.json', dict(task=task, results=values,
            metric_version=v1.VERSION, original_seed_offset=task['seed']-42,
            auto_export=False, note='Combine only with the explicitly preserved seed sources; review against Table 4'))
    v1.t11.verify(manifest)
    v1.write(out/f'complete_{index}.json', dict(task=task, metric_version=v1.VERSION, auto_export=False))
    print(f'COMPLETE recovery task {index}; review before publication', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    out = args.output_root.resolve()
    if not out.is_relative_to(ROOT/'results/validation'):
        raise ValueError('Use isolated results/validation directory')
    if args.prepare:
        prepare(out)
    elif args.index is not None:
        run(out, args.index)
    else:
        parser.error('--prepare or --index required')
