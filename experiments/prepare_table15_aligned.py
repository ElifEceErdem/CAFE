"""Prepare/run a separately reviewed, split-resampled Table 15 campaign.

No jobs are submitted by this program. Each task is a dataset/seed block,
running the five Table 4 model paths in fresh processes on one allocated GPU.
The only split override is injected into the existing loader in-process;
research source files are never rewritten. No acceptance PW/TP work is needed.
"""
import argparse
import csv
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys

from replay_table11_aligned import ROOT, METRICS, ARMS, command, execute, sha, verify, write_new

DATASETS = ['ego-facebook', 'oklahoma', 'unc', 'pokec_n', 'brightkite']
TABLE11 = ROOT / 'results/validation/20260905_163304_table11_table4_aligned_v3'


def prepare(out):
    manifest = json.loads((TABLE11 / 'authority.json').read_text())
    verify(manifest)
    oklahoma = json.loads((ROOT / 'results/validation/20260905_141132_cafe_pre_ten_run_diagnostic/authority.json').read_text())
    verify(oklahoma)
    with (ROOT / 'experiments/configs/legacy_best_configs_selected.csv').open() as handle:
        selected = list(csv.DictReader(handle))
    manifest['configs']['oklahoma'] = {r['variant']: r for r in selected if r['dataset'] == 'oklahoma'}
    manifest['references']['oklahoma'] = oklahoma['references']
    # Retain the coefficients used by the previous Table 15 job, without
    # executing _common.sh (which activates environments and creates settings).
    common = (ROOT / 'experiments/jobs/_common.sh').read_text()
    fusion = {}
    for ds in DATASETS:
        matches = re.findall(r'^\s*' + re.escape(ds) + r'\)(.*?) ;;', common, re.M | re.S)
        if len(matches) != 1:
            raise ValueError(f'Cannot resolve separately selected fusion control: {ds}')
        beta = re.findall(r'FUSION_BETA=([0-9.]+)', matches[0])
        if len(beta) != 1:
            raise ValueError(f'Ambiguous fusion coefficient: {ds}')
        fusion[ds] = f'lambda_fair_0.0_beta_{float(beta[0])}_lam_0.0'
    manifest.update(protocol='table4_paths_table15_split_resampled', datasets=DATASETS,
                    fusion_control='Separately selected coefficient; Pokec-n beta=5, unlike Table 11 beta=1.',
                    fusion_configs=fusion, training_seeds=list(range(10)),
                    split_rule='seed s for random-split datasets; canonical temporal split for Brightkite',
                    launch_status='prepared_not_submitted_requires_separate_review',
                    tasks=[dict(index=i * 10 + s, dataset=ds, seed=s,
                                split_seed=42 if ds == 'brightkite' else s)
                           for i, ds in enumerate(DATASETS) for s in range(10)],
                    statistics_plan=dict(contrasts='CAFE-minus-reference per matching dataset/seed',
                                         families=['vs HL-GNN', 'vs Fusion', 'vs RA'],
                                         contrasts_per_family=40,
                                         interval='two-sided paired t, 95 percent; sample SD of differences',
                                         correction='Holm within each prespecified 40-contrast family'),
                    no_exact_table4_mean_requirement=True)
    manifest.pop('oklahoma_omitted', None)
    paths = [Path(__file__).resolve(), ROOT / 'experiments/jobs/t15_table4_aligned.job',
             TABLE11 / 'authority.json', ROOT / 'analysis/best_configs.csv',
             ROOT / 'experiments/jobs/t15_paired_validation.job',
             ROOT / 'experiments/jobs/t15_paired_validation_pokec.job',
             ROOT / 'data/SNAP/oklahoma/processed/Oklahoma97_user_sen.pkl',
             ROOT / 'data/SNAP/oklahoma/Oklahoma97/Oklahoma97_user_sen.pkl',
             ROOT / 'data/SNAP/oklahoma/Oklahoma97/Oklahoma97_train_items.pkl',
             ROOT / 'data/SNAP/oklahoma/Oklahoma97/Oklahoma97_test_set.pkl']
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest['frozen_sha256'][str(path.relative_to(ROOT))] = sha(path)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / 'authority.json', manifest)
    # Record every intended command before any job is authorized or run.
    plans = []
    for task in manifest['tasks']:
        for arm in ARMS:
            folder = out / task['dataset'] / f'seed_{task["seed"]}' / arm.lower()
            plans.append(dict(task=task, arm=arm, argv=child_command(manifest, task, arm, folder)))
    write_new(out / 'planned_commands.json', plans)
    print(f'Prepared 50 dataset/seed blocks (250 fresh method processes): {out}; NOT submitted')


def child_command(manifest, task, arm, folder):
    base = command(manifest, task['dataset'], arm, folder)
    runner = 'run_cafe' if arm == 'CAFE-Pre' else Path(base[2]).stem
    args = base[base.index('--') + 1:] if arm == 'CAFE-Pre' else base[3:]
    args[args.index('--runs') + 1] = '1'
    args[args.index('--seed') + 1] = str(task['seed'])
    for flag in ['--acceptance_probe_jsonl', '--acceptance_probe_method']:
        if flag in args:
            idx = args.index(flag)
            del args[idx:idx + 2]
    return [sys.executable, '-u', str(Path(__file__).resolve()), '--child',
            '--runner', runner, '--split-seed', str(task['split_seed']),
            '--evidence-dir', str(folder), '--', *args]


def child(runner_name, split_seed, folder, runner_args):
    sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'backbone'),
                    str(ROOT / 'fairness_methods'), str(ROOT)]
    import hashlib
    import platform
    import torch
    from core.acceptance_probe import preserve_training_rng
    if not torch.cuda.is_available():
        raise RuntimeError('No allocated GPU; refusing CPU fallback')
    runner = importlib.import_module(runner_name)
    original = runner.load_dataset

    def load(args):
        args.data_split_seed = split_seed
        result = original(args)
        dataset, split, data, degrees = result
        with preserve_training_rng():
            hashes = {}
            for name in ['train', 'valid', 'test']:
                for key in ['edge', 'edge_neg']:
                    tensor = split[name].get(key)
                    if tensor is not None:
                        array = tensor.detach().cpu().contiguous().numpy()
                        digest = hashlib.sha256(str((str(array.dtype), array.shape)).encode())
                        digest.update(array.tobytes())
                        hashes[f'{name}/{key}'] = digest.hexdigest()
            evidence = dict(dataset=args.data_name, training_seed=args.seed,
                            split_seed=split_seed, num_nodes=int(data.num_nodes), splits=hashes,
                            temporal=args.data_name == 'brightkite')
            write_new(folder / 'split_evidence.json', evidence)
        return result
    runner.load_dataset = load
    write_new(folder / 'environment.json', dict(hostname=platform.node(), torch=torch.__version__,
              cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
              threads=torch.get_num_threads(), deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
    if runner_name == 'run_cafe' and '--penalization_mode' in runner_args and runner_args[runner_args.index('--penalization_mode') + 1] == 'preproc':
        # Retain the validated Pre fingerprint instrumentation, without PW/TP.
        from diagnose_cafe_pre import child as diagnostic_child
        diagnostic_child(folder / 'trace.jsonl', runner_args)
    else:
        sys.argv = [runner_name + '.py', *runner_args]
        runner.main()


def collect(manifest, task, arm, folder):
    ds = task['dataset']
    if arm == 'RA':
        paths = list((folder / 'main' / ds / 'baselines').glob('results_baselines_*.json'))
        if len(paths) != 1:
            raise ValueError('Missing/ambiguous completed RA output')
        raw = json.loads(paths[0].read_text())
        if raw['runs_completed']['RA'] != 1:
            raise ValueError('Incomplete RA observation')
        values = {m: float(raw['results']['RA'][m]['values'][0]) for m in METRICS}
        config = 'resource_allocation'
    else:
        config = manifest['fusion_configs'][ds] if arm == 'Fusion' else manifest['references'][ds][arm]['config']
        raw = json.loads((folder / 'main_results.json').read_text())
        if len(raw['individual_runs']) != 1:
            raise ValueError('Expected a single training repetition')
        values = {m: float(raw['individual_runs'][0][config][m][1]) for m in METRICS}
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError('Non-finite main metrics')
    evidence = json.loads((folder / 'split_evidence.json').read_text())
    if (evidence['dataset'] != ds or evidence['training_seed'] != task['seed']
            or evidence['split_seed'] != task['split_seed']):
        raise ValueError('Seed/split provenance mismatch')
    return dict(method=arm, config=config, metrics=values, split_evidence=evidence)


def run(out, index):
    manifest = json.loads((out / 'authority.json').read_text())
    verify(manifest)
    task = manifest['tasks'][index]
    block = out / task['dataset'] / f'seed_{task["seed"]}'
    block.mkdir(parents=True, exist_ok=False)
    results = []
    for arm in ARMS:
        verify(manifest)
        folder = block / arm.lower()
        folder.mkdir()
        argv = child_command(manifest, task, arm, folder)
        write_new(folder / 'launch.json', dict(argv=argv, task=task, arm=arm,
                  job_id=os.environ.get('SLURM_JOB_ID'), authority_sha256=sha(out / 'authority.json')))
        print(f'START Table 15 aligned: {task["dataset"]} seed={task["seed"]} {arm}', flush=True)
        rc = execute(argv)
        if rc:
            raise SystemExit(rc)
        result = collect(manifest, task, arm, folder)
        write_new(folder / 'observation.json', result)
        results.append(result)
    verify(manifest)
    if not all(r['split_evidence'] == results[0]['split_evidence'] for r in results):
        write_new(block / 'pairing_failed.json', dict(task=task, reason='Method splits differ; do not form paired contrasts'))
        raise ValueError('Methods do not share identical split evidence')
    write_new(block / 'paired_observations.json', dict(task=task, methods=results,
              pairing_verified=True, protocol=manifest['protocol']))
    print('FINISHED: matching split evidence verified; statistics require all 50 complete blocks', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--task-index', type=int, choices=range(50))
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--runner', choices=['run_baselines', 'run_cafe_p', 'run_cafe'])
    parser.add_argument('--split-seed', type=int)
    parser.add_argument('--evidence-dir', type=Path)
    parser.add_argument('runner_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.child:
        if args.runner is None or args.split_seed is None or args.evidence_dir is None:
            parser.error('Child requires runner, split seed, and evidence directory')
        child(args.runner, args.split_seed, args.evidence_dir,
              args.runner_args[1:] if args.runner_args[:1] == ['--'] else args.runner_args)
    else:
        if args.output_root is None:
            parser.error('--output-root required')
        out = args.output_root.resolve()
        if not out.is_relative_to(ROOT / 'results/validation'):
            raise ValueError('Use an isolated results/validation directory')
        if args.prepare:
            prepare(out)
        elif args.task_index is not None:
            run(out, args.task_index)
        else:
            parser.error('Specify --prepare or --task-index')
