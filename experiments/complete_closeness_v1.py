"""Finish the missing exact-v1 Table 7 arms and Table 4 seed-49 evidence.

Runs existing scientific entry points in fresh processes. Only exact Closeness
activation, metadata/fingerprint instrumentation, and post-run plot suppression
are added. Do not split the evolving ten-run ablation RNG stream across GPUs.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import tempfile

import recover_closeness_v1 as recovery
import reevaluate_closeness_v1 as v1

ROOT = v1.ROOT
SELF = Path(__file__).resolve()
METHODS = ['CI-In', 'CI-Pre', 'MB-Post', 'CAFE-In-B', 'CAFE-Pre-B', 'CAFE-P']
METRICS = v1.t11.METRICS


def read(path):
    return json.loads(Path(path).read_text())


def scalar(value):
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError('Unexpected metric pair')
        value = value[1]
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Non-finite metric')
    return value


def observations(raw, config, runs=10):
    rows = raw['individual_runs']
    if len(rows) != runs or set(raw['lambda_results']) != {config}:
        raise ValueError('Incomplete or unexpected configuration grid')
    values = []
    for index, row in enumerate(rows):
        if set(row)-{'run', 'baseline'} != {config}:
            raise ValueError('Unexpected per-run configuration')
        if 'run' in row and row['run'] != index+1:
            raise ValueError('Incorrect repetition index')
        values.append({m: scalar(row[config][m]) for m in METRICS})
    return values


def ablation_command(task, folder):
    ds, method, key = task['dataset'], task['method'], task['config']
    script = ('run_coreness_inverse.py' if method in ['CI-In', 'CI-Pre'] else
              'run_mobility_boost.py' if method == 'MB-Post' else
              'run_cafe_p.py' if method == 'CAFE-P' else 'run_cafe.py')
    argv = [sys.executable, '-u', str(ROOT/'experiments'/script),
            '--data_name', ds, '--data_path', str(ROOT/'data/SNAP'),
            '--encoder', 'HLGNN', '--predictor', 'MLP', '--epochs', '500',
            '--eval_steps', '10', '--patience', '50', '--batch_size', '16384',
            '--lr', '0.001', '--runs', '10', '--seed', '42',
            '--coreness_alpha', '1.0', '--use_equality_gain', 'True',
            '--use_centrality_gain', 'True', '--results_file', str(folder/'main_results.json')]
    if method != 'CAFE-P':
        argv += ['--centrality_types', 'pagerank,closeness']
    if method in ['CI-In', 'CI-Pre', 'MB-Post']:
        match = re.fullmatch(r'lambda_([0-9.]+)', key)
        if not match:
            raise ValueError('Bad CI/MB configuration: '+key)
        lam = match.group(1)
        argv += ['--use_stratified', 'False', '--use_pa_aware', 'False',
                 '--use_ndkl', 'False', '--use_degree_plots', 'False']
        if method == 'MB-Post':
            argv += ['--reranking_approach', 'boost', '--boosting_mode', 'continuous',
                     '--lambda_mobility', lam, '--lambda_bridge', '0.0']
        else:
            argv += ['--fairness_method', 'coreness_inverse_'+('inproc' if method == 'CI-In' else 'preproc'),
                     '--lambda_fair', lam]
    elif method == 'CAFE-P':
        match = re.fullmatch(r'beta_([0-9.]+)_lambda_([0-9.]+)', key)
        if not match:
            raise ValueError('Bad CAFE-P configuration: '+key)
        argv += ['--beta_ra', match.group(1), '--lambda_mobility', match.group(2)]
    else:
        prefix = 'lambda_fair' if method == 'CAFE-In-B' else 'preproc_alpha'
        match = re.fullmatch(prefix+r'_([0-9.]+)_beta_([0-9.]+)_lam_(0\.0)', key)
        if not match:
            raise ValueError('Bad minus-B configuration: '+key)
        argv += ['--penalization_mode', 'inproc' if method == 'CAFE-In-B' else 'preproc',
                 '--'+prefix, match.group(1), '--beta_ra', match.group(2),
                 '--lambda_mobility', '0.0', '--use_mobility_boost', 'False',
                 '--eval_skip_test', 'True', '--eval_max_nodes', '5000',
                 '--eval_steps', '25', '--patience', '6']
    if ds == 'brightkite':
        argv += ['--split_file', str(ROOT/'data/SNAP/brightkite/temporal_split.pt')]
    elif ds != 'pokec_n':
        raise ValueError('Only the two affected datasets are in scope')
    return argv


def wrapped(argv, folder):
    return [sys.executable, '-u', str(SELF), '--child-output', str(folder), '--', *argv]


def child(folder, argv):
    import core
    import core.statistics as stats
    import torch
    original_load = core.load_dataset
    original_info = stats.create_experiment_info
    original_core_info = core.create_experiment_info

    # CI/MB serializers resolve this name in core.statistics, whereas CAFE
    # imports it from core. Route both through the existing versioned hook.
    def info(*args, **kwargs):
        return core.create_experiment_info(*args, **kwargs)

    def load(args):
        result = original_load(args)
        evidence = dict(dataset=args.data_name,
                        split_seed=getattr(args, 'data_split_seed', 42), tensors={})
        for part in ['train', 'valid', 'test']:
            for name in ['edge', 'edge_neg']:
                tensor = result[1].get(part, {}).get(name)
                if torch.is_tensor(tensor):
                    array = tensor.detach().cpu().contiguous().numpy()
                    digest = hashlib.sha256(str((str(array.dtype), array.shape)).encode())
                    digest.update(array.tobytes())
                    evidence['tensors'][part+'/'+name] = digest.hexdigest()
        if not {'train/edge', 'test/edge'} <= set(evidence['tensors']):
            raise ValueError('Missing train/test split evidence')
        path = folder/'split_evidence.json'
        if path.exists():
            if read(path) != evidence:
                raise ValueError('Fixed split changed between repetitions')
        else:
            v1.write(path, evidence)
        return result

    core.load_dataset = load
    stats.create_experiment_info = info
    try:
        v1.child(folder, argv)
    finally:
        core.load_dataset = original_load
        stats.create_experiment_info = original_info
        core.create_experiment_info = original_core_info


def prepare(out):
    if subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() != 'repo-cleanup':
        raise ValueError('Use repo-cleanup')
    if out.exists():
        raise FileExistsError(out)
    authority = read(recovery.OLD/'table11/authority.json')
    v1.t11.verify(authority)
    for ds in v1.DATASETS:
        v1.require_gate(recovery.OLD, ds)
    with (ROOT/'analysis/best_configs.csv').open() as handle:
        configs = list(csv.DictReader(handle))
    with (ROOT/'comparison_tables/table7_paper_vs_reproduced.csv').open() as handle:
        comparison = list(csv.DictReader(handle))
    tasks, paths = [], []
    for ds in v1.DATASETS:
        for method in METHODS:
            selected = [r for r in configs if r['Dataset'] == ds and r['Method'] == method]
            sources = {r['source_file'] for r in comparison if r['dataset'] == ds and r['method'] == method}
            if len(selected) != 1 or len(sources) != 1:
                raise ValueError('Ambiguous/missing selected configuration or original results')
            source = ROOT/sources.pop()
            raw = read(source)
            if raw['experiment_info'].get('closeness_metric_version') == v1.VERSION:
                raise ValueError('Selected ablation already corrected; re-plan')
            config = selected[0]['Config_Key']
            observations(raw, config)
            tasks.append(dict(index=len(tasks), group='table7', dataset=ds, method=method,
                              config=config, runs=10, reference=str(source.relative_to(ROOT))))
            paths.append(source)
    # Seed 48/49 has scalar outputs but no process-completion certificate.
    # Preserve its first repetition as in the earlier seed-44/46 recovery;
    # rerun only seed 49 in its original stochastic-method order. Its original
    # second repetition is retained for comparison, never counted twice.
    old_folder = recovery.OLD/'table4_baselines/pokec_n/seed_48'
    if (old_folder/'exact_process_complete.json').exists():
        raise ValueError('Baseline completion evidence has appeared; re-plan')
    saved = recovery.baseline_values(old_folder, 2)
    paths += [Path(p) for p in set(saved.values())]
    paths += [old_folder/'launch.json', old_folder/'exact_version.json', old_folder/'exact_metric_calls.jsonl']
    tasks.append(dict(index=12, group='baselines', dataset='pokec_n', seed=49, runs=1,
                      reuse_seed=48, reused_results=saved, reuse_value_index=0,
                      reason='Uncertified final repetition; retain original files and independently rerun seed 49'))
    paths += [SELF, ROOT/'experiments/jobs/closeness_v1_missing.job',
              ROOT/'tests/test_closeness_v1_missing.py', ROOT/'tests/fixtures/closeness_v1_ablation_serializer.py',
              ROOT/'experiments/recover_closeness_v1.py',
              ROOT/'analysis/best_configs.csv', ROOT/'comparison_tables/table7_paper_vs_reproduced.csv',
              recovery.OLD/'table11/authority.json']
    paths += [ROOT/'experiments'/name for name in
              ['run_coreness_inverse.py', 'run_mobility_boost.py', 'run_cafe.py', 'run_cafe_p.py']]
    frozen = dict(authority['frozen_sha256'])
    for path in paths:
        frozen[str(path.relative_to(ROOT))] = v1.t11.sha(path)
    manifest = dict(metric_version=v1.VERSION, tasks=tasks, frozen_sha256=frozen,
                    protocol='Table 7 fixed split, original seed-42 evolving RNG, ten repetitions per arm',
                    no_predictions_saved=True, no_checkpoints_saved=True, automatic_table_update=False,
                    plot_policy='Disable optional post-run CI/MB plots; no training/evaluation setting changed',
                    baseline_policy='Reuse seed-48 index 0 only; replace seed-49 index 1 with separately validated recovery',
                    publication_gate='Review all metrics, split fingerprints and baseline recovery evidence before export',
                    already_complete='Tables 11/15 and shared Table 7 HL/Fusion/CAFE rows are not relaunched; Table 7 Fusion beta=5')
    out.mkdir(parents=True, exist_ok=False)
    v1.write(out/'manifest.json', manifest)
    planned = []
    for task in tasks:
        folder = task_folder(out, task)
        argv = ablation_command(task, folder) if task['group'] == 'table7' else recovery.baseline_command(authority, task, folder)
        planned.append(dict(task=task, output=str(folder), argv=wrapped(argv, folder)))
    v1.write(out/'planned_commands.json', planned)
    print(f'PREPARED {out}: 12 ablation jobs plus targeted baseline seed-49 recovery; NOT submitted')


def task_folder(out, task):
    if task['group'] == 'table7':
        return out/'table7'/task['dataset']/task['method'].lower()
    return out/'table4_baselines/pokec_n/seed_49'


def validate_ablation(folder, task):
    raw = read(folder/'main_results.json')
    if raw['experiment_info'].get('closeness_metric_version') != v1.VERSION:
        raise ValueError('Missing exact version in original result metadata')
    if raw['experiment_info']['dataset'] != task['dataset'] or raw['experiment_info']['num_runs'] != 10:
        raise ValueError('Original result dataset/run metadata mismatch')
    values = observations(raw, task['config'])
    old_raw = read(ROOT/task['reference'])
    previous = observations(old_raw, task['config'])
    old_fingerprint = old_raw['experiment_info'].get('split_fingerprint')
    if old_fingerprint and raw['experiment_info'].get('split_fingerprint') != old_fingerprint:
        raise ValueError('Original runner split fingerprint differs from Table 7')
    if task['method'] in ['CI-In', 'CI-Pre', 'MB-Post']:
        for row in raw['individual_runs']:
            for metric in METRICS:
                scalar(row['baseline'][metric])
    summary = {}
    for metric in METRICS:
        nums = [r[metric] for r in values]
        summary[metric] = dict(mean=statistics.mean(nums), std=statistics.pstdev(nums), values=nums)
        reported = raw['lambda_results'][task['config']]['statistics'][metric]
        for field in ['mean', 'std']:
            if not math.isclose(summary[metric][field], scalar(reported[field]), abs_tol=1e-8, rel_tol=1e-6):
                raise ValueError('Per-run values disagree with aggregate '+metric+'/'+field)
    deltas = [{m: row[m]-old[m] for m in v1.NONCL} for row, old in zip(values, previous)]
    return dict(task=task, metrics=summary, noncloseness_per_run_delta=deltas,
                noncloseness_exact=all(v == 0 for row in deltas for v in row.values()),
                split_evidence=read(folder/'split_evidence.json'),
                metric_version=v1.VERSION, auto_export=False,
                note='Keep each whole observation together; changed non-Closeness metrics need review')


def run(out, index):
    manifest = read(out/'manifest.json')
    v1.t11.verify(manifest)
    task = manifest['tasks'][index]
    v1.require_gate(recovery.OLD, task['dataset'])
    if (out/f'complete_{index}.json').exists():
        raise FileExistsError('Task already complete; do not repeat')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Allocated GPU required')
    with tempfile.TemporaryFile(dir=out) as handle:
        handle.write(b'quota preflight\n'*1024)
        handle.flush()
        os.fsync(handle.fileno())
    folder = task_folder(out, task)
    folder.mkdir(parents=True, exist_ok=False)
    if task['group'] == 'table7':
        argv = ablation_command(task, folder)
    else:
        argv = recovery.baseline_command(read(recovery.OLD/'table11/authority.json'), task, folder)
    v1.write(folder/'launch.json', dict(task=task, argv=wrapped(argv, folder),
        manifest_sha256=v1.t11.sha(out/'manifest.json'), node=socket.gethostname(),
        job_id=os.environ.get('SLURM_JOB_ID'), array_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
        gpu=torch.cuda.get_device_name(0)))
    print(f'START missing task {index}: {task}; GPU={torch.cuda.get_device_name(0)}', flush=True)
    rc = v1.t11.execute(wrapped(argv, folder))
    if rc:
        raise SystemExit(rc)
    recovery.complete(folder)
    if task['group'] == 'table7':
        report = validate_ablation(folder, task)
    else:
        values = recovery.baseline_values(folder, 1, deterministic=False)
        comparison = {}
        for method, path in values.items():
            old = read(task['reused_results'][method])['results'][method]
            new = read(path)['results'][method]
            comparison[method] = {m:new[m]['values'][0]-old[m]['values'][1] for m in METRICS}
        report = dict(task=task, results=values, seed49_delta_vs_uncertified_saved=comparison,
                      metric_version=v1.VERSION, auto_export=False,
                      note='Seed 48 saved index 0 is a reuse candidate, not a retroactive process-completion certificate')
    v1.write(folder/'validation_report.json', report)
    v1.t11.verify(manifest)
    v1.write(out/f'complete_{index}.json', dict(task=task, metric_version=v1.VERSION, auto_export=False))
    print(f'COMPLETE missing task {index}; review before publication', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--index', type=int, choices=range(13))
    parser.add_argument('--child-output', type=Path)
    args, remaining = parser.parse_known_args()
    if args.child_output:
        if not remaining or remaining[0] != '--':
            parser.error('Child command must follow --')
        child(args.child_output.resolve(), remaining[1:])
    else:
        if remaining or args.output_root is None:
            parser.error('Provide --output-root without extra arguments')
        out = args.output_root.resolve()
        if not out.is_relative_to(ROOT/'results/validation'):
            raise ValueError('Use a new results/validation directory')
        if args.prepare:
            prepare(out)
        elif args.index is not None:
            run(out, args.index)
        else:
            parser.error('Provide --prepare or --index')
