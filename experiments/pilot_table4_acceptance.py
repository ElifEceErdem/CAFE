"""Prepare/run an isolated one-repetition Oklahoma replay; scalar outputs only.

This is a gate before any multi-dataset rerun, NOT a replacement Table 11 or 15.
Table 15's split-resampled protocol is not changed by this fixed-split pilot.
"""
import argparse
from collections import deque
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ARMS = ['RA', 'HL-GNN', 'Fusion', 'CAFE-In', 'CAFE-Pre']
METRICS = ['Precision@10', 'Equality_Gain@10', 'Pagerank_EG@10', 'Closeness_EG@10']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def prepare(out):
    with (ROOT / 'comparison_tables/table4_paper_vs_reproduced.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    references = {}
    paths = ['comparison_tables/table4_updated_red.tex',
             'comparison_tables/table4_paper_vs_reproduced.csv',
             'experiments/configs/legacy_best_configs_selected.csv',
             'experiments/run_cafe.py', 'experiments/run_cafe_p.py',
             'experiments/run_baselines.py', 'baselines/baseline_wrapper.py',
             'baselines/baseline_models.py', 'core/acceptance_probe.py',
             'core/data_loader.py', 'core/training_inproc.py', 'core/training.py',
             'core/evaluation.py', 'core/utils.py', 'fairness_methods/inprocessing.py',
             'backbone/snap_dataset.py',
             'metrics/acceptance_eg.py', 'metrics/equality_gain.py',
             'metrics/centrality_equality_gain.py',
             'experiments/pilot_table4_acceptance.py',
             'experiments/jobs/t04_acceptance_pilot.job',
             'data/SNAP/oklahoma/processed/data.pt']
    for method, key in [('RA', 'RA'), ('HL-GNN', 'HLGNN'),
                        ('CAFE-In', 'CAFE-In'), ('CAFE-Pre', 'CAFE-Pre')]:
        source_rows = [r for r in rows if r['dataset'] == 'oklahoma' and r['method'] == key]
        sources = {r['source_file'] for r in source_rows}
        if len(sources) != 1 or len(source_rows) != 4:
            raise ValueError(f'Unexpected authoritative sources for {method}')
        source = sources.pop()
        paths.append(source)
        data = json.loads((ROOT / source).read_text())
        if method == 'RA':
            first = {m: float(data['results']['RA'][m]['values'][0]) for m in METRICS}
            config = 'resource_allocation'
        else:
            raw = data['individual_runs'][0]
            configs = [k for k in raw if k != 'baseline']
            if len(configs) != 1:
                raise ValueError(f'Cannot infer selected configuration for {method}')
            config = configs[0]
            first = {m: float(raw[config][m][1]) for m in METRICS}
        references[method] = dict(source_file=source, config=config, first_run=first,
                                  table4_aggregate={r['metric']: r['reproduced_display'] for r in source_rows})
    missing = [p for p in paths if not (ROOT / p).is_file()]
    if missing:
        raise FileNotFoundError(missing)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / 'authority.json', dict(dataset='oklahoma', runs=1,
              purpose='gate only; do not replace paper tables from a one-run pilot',
              references=references, frozen_sha256={p: sha(ROOT / p) for p in paths}))
    print(f'Prepared isolated pilot: {out}')


def verify(manifest):
    for name, expected in manifest['frozen_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'Frozen pilot input changed: {name}')


def command(arm, folder):
    data = str(ROOT / 'data/SNAP')
    common = ['--data_name', 'oklahoma', '--data_path', data, '--encoder', 'HLGNN',
              '--predictor', 'MLP', '--epochs', '500', '--eval_steps', '10',
              '--patience', '50', '--batch_size', '16384', '--lr', '0.001',
              '--runs', '1', '--seed', '42', '--use_equality_gain', 'True',
              '--use_centrality_gain', 'True', '--coreness_alpha', '1.0',
              '--results_file', str(folder / 'main_results.json'),
              '--acceptance_probe_jsonl', str(folder / 'acceptance.jsonl')]
    if arm == 'RA':
        # Preserve the original preceding heuristic order and RNG consumption.
        return [sys.executable, '-u', 'run_baselines.py', '--data_name', 'oklahoma',
                '--data_path', data, '--encoder', 'HLGNN', '--methods', 'random,pa,cn,aa,ra',
                '--runs', '1', '--seed', '42', '--epochs', '300', '--batch_size', '16384',
                '--use_equality_gain', 'True', '--use_centrality_gain', 'True',
                '--centrality_types', 'pagerank,closeness', '--results_dir', str(folder / 'main'),
                '--acceptance_probe_jsonl', str(folder / 'acceptance.jsonl')]
    if arm == 'HL-GNN':
        return [sys.executable, '-u', 'run_cafe_p.py', *common,
                '--beta_ra', '0.0', '--lambda_mobility', '0.0']
    options = {
        'Fusion': ['--penalization_mode', 'inproc', '--lambda_fair', '0.0',
                   '--beta_ra', '0.1,0.5,1.0,2.0,5.0', '--lambda_mobility', '0.0',
                   '--use_mobility_boost', 'False'],
        'CAFE-In': ['--penalization_mode', 'inproc', '--lambda_fair', '0.3',
                    '--beta_ra', '5.0', '--lambda_mobility', '0.5', '--use_mobility_boost', 'True'],
        'CAFE-Pre': ['--penalization_mode', 'preproc', '--preproc_alpha', '1.0',
                     '--beta_ra', '0.5', '--lambda_mobility', '0.5', '--use_mobility_boost', 'True'],
    }
    return [sys.executable, '-u', 'run_cafe.py', *common, *options[arm],
            '--centrality_types', 'pagerank,closeness', '--acceptance_probe_method', arm]


def run(out, arm_index):
    manifest = json.loads((out / 'authority.json').read_text())
    verify(manifest)
    arm = ARMS[arm_index]
    folder = out / arm.lower()
    folder.mkdir(exist_ok=False)  # Never overwrite or silently replay a partial arm.
    argv = command(arm, folder)
    write_new(folder / 'launch.json', dict(argv=argv, job_id=os.environ.get('SLURM_JOB_ID'),
               array_task=os.environ.get('SLURM_ARRAY_TASK_ID'), executable=sys.executable,
               authority_sha256=sha(out / 'authority.json')))
    print(f'PILOT START: {arm}; one Table 4 repetition; no predictions/checkpoints saved', flush=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    proc = subprocess.Popen(argv, cwd=ROOT / 'experiments', env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = deque(maxlen=40)
    budget = 48000
    for line in proc.stdout:
        line = line.rstrip()[:1200]
        tail.append(line)
        if any(token in line for token in ['RUN ', '[acceptance pilot]', 'Saved ', 'Error', 'Traceback', 'FATAL']):
            if budget > len(line) + 1:
                print(line, flush=True)
                budget -= len(line) + 1
    rc = proc.wait()
    if rc:
        print('PILOT FAILED; last bounded output:', flush=True)
        print('\n'.join(tail), flush=True)
        raise SystemExit(rc)
    verify(manifest)
    records = [json.loads(s) for s in (folder / 'acceptance.jsonl').read_text().splitlines()]
    expected_configs = ({f'lambda_fair_0.0_beta_{b}_lam_0.0' for b in ['0.1', '0.5', '1.0', '2.0', '5.0']}
                        if arm == 'Fusion' else {manifest['references'][arm]['config']})
    if {r['config'] for r in records} != expected_configs or len(records) != len(expected_configs):
        raise ValueError('Missing/unexpected pilot configurations')
    if not all(r['full_matches_main'] and r['run'] == 0 and r['method'] == arm for r in records):
        raise ValueError('Pilot full/main identity or record provenance failed')
    report = dict(status='calculation_complete', full_matches_main=True,
                  historical_replay_checked=(arm in manifest['references']),
                  not_a_paper_table=True)
    if arm in manifest['references']:
        reference = manifest['references'][arm]['first_run']
        report['historical_first_run'] = reference
        report['replayed_first_run'] = records[0]['main']
        report['replay_minus_historical'] = {m: records[0]['main'][m] - reference[m] for m in METRICS}
        report['matches_historical_display'] = all(
            f"{records[0]['main'][m]:.{4 if m == 'Precision@10' else 2}f}" ==
            f"{reference[m]:.{4 if m == 'Precision@10' else 2}f}" for m in METRICS)
    write_new(folder / 'pilot_report.json', report)
    print(json.dumps(report), flush=True)
    print('PILOT FINISHED. Review historical deltas before authorizing full reruns.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--arm-index', type=int, choices=range(5))
    args = parser.parse_args()
    out = args.output_root.resolve()
    if not out.is_relative_to(ROOT / 'results/validation'):
        raise ValueError('Pilot output must be isolated under results/validation')
    if args.prepare:
        prepare(out)
    elif args.arm_index is not None:
        run(out, args.arm_index)
    else:
        parser.error('Specify --prepare or --arm-index')
