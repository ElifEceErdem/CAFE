"""Scalar-only Table 11 replay through the authoritative Table 4 runners.

Four datasets, five independent arms each. Oklahoma is already validated.
CAFE-Pre reuses the successful fingerprint wrapper without changing research
code. Fusion retains the original five-beta evaluation order; Table 11 selects
the beta of CAFE-In, not the independently selected Table 15 control.
"""
import argparse
from collections import deque
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

from pilot_table4_acceptance import ROOT, METRICS, command as pilot_command, write_new

DATASETS = ['ego-facebook', 'unc', 'pokec_n', 'brightkite']
ARMS = ['RA', 'HL-GNN', 'Fusion', 'CAFE-In', 'CAFE-Pre']
BETAS = ['0.1', '0.5', '1.0', '2.0', '5.0']
VALIDATED = ROOT / 'results/validation/20260905_141132_cafe_pre_ten_run_diagnostic'
CHILD = ROOT / 'experiments/diagnose_cafe_pre.py'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(manifest):
    for path, expected in manifest['frozen_sha256'].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f'Frozen input changed: {path}')


def prepare(out):
    previous = json.loads((VALIDATED / 'authority.json').read_text())
    verify(previous)
    report = json.loads((VALIDATED / 'diagnostic_report.json').read_text())
    if (not report['off_vs_on']['shared_traces_identical'] or
            not all(r['all_raw_metrics_exact'] for r in report['trials'].values())):
        raise ValueError('Oklahoma diagnostic gate failed')
    with (ROOT / 'comparison_tables/table4_paper_vs_reproduced.csv').open() as handle:
        comparison = list(csv.DictReader(handle))
    with (ROOT / 'experiments/configs/legacy_best_configs_selected.csv').open() as handle:
        configs = list(csv.DictReader(handle))
    manifest = dict(protocol='table4_aligned_table11_fixed_split',
                    datasets=DATASETS, arms=ARMS, tasks=[], references={}, configs={},
                    descriptive_sd_ddof=0, predictions_saved=False,
                    oklahoma_omitted='Validated pilot/replay/diagnostic outputs are reused at export, not rerun.',
                    fusion_control='Table 11 beta matches CAFE-In; original five-beta sweep retained.',
                    frozen_sha256=dict(previous['frozen_sha256']))
    paths = [Path(__file__).resolve(), ROOT / 'experiments/jobs/t11_table4_aligned.job',
             ROOT / 'experiments/jobs/t04_cafe.job', ROOT / 'experiments/jobs/_common.sh',
             VALIDATED / 'authority.json', VALIDATED / 'diagnostic_report.json',
             ROOT / 'core/statistics.py']
    # Freeze Python implementation dependencies, not mutable logs or other jobs.
    for folder in ['core', 'backbone', 'baselines', 'metrics', 'fairness_methods']:
        paths.extend((ROOT / folder).rglob('*.py'))
    for ds in DATASETS:
        ds_configs = {r['variant']: r for r in configs if r['dataset'] == ds}
        if set(ds_configs) != {'cafe_in', 'cafe_pre'}:
            raise ValueError(f'Missing/ambiguous selected configs: {ds}')
        manifest['configs'][ds] = ds_configs
        references = {}
        for arm, method in [('RA', 'RA'), ('HL-GNN', 'HLGNN'), ('CAFE-In', 'CAFE-In'), ('CAFE-Pre', 'CAFE-Pre')]:
            rows = [r for r in comparison if r['dataset'] == ds and r['method'] == method]
            sources = {r['source_file'] for r in rows}
            if len(rows) != 4 or len(sources) != 1:
                raise ValueError(f'Ambiguous Table 4 reference: {ds}/{arm}')
            path = ROOT / sources.pop()
            paths.append(path)
            raw = json.loads(path.read_text())
            if arm == 'RA':
                config = 'resource_allocation'
                historical = [{m: float(raw['results']['RA'][m]['values'][0]) for m in METRICS}]
            else:
                keys = set(raw['individual_runs'][0]) - {'baseline'}
                if len(keys) != 1 or len(raw['individual_runs']) != 10:
                    raise ValueError(f'Expected one configuration and ten runs: {ds}/{arm}')
                config = keys.pop()
                historical = [{m: float(r[config][m][1]) for m in METRICS} for r in raw['individual_runs']]
                if arm in ['CAFE-In', 'CAFE-Pre']:
                    selected = ds_configs['cafe_in' if arm == 'CAFE-In' else 'cafe_pre']
                    if selected['config_key'] != config:
                        raise ValueError(f'Selected config disagrees with Table 4: {ds}/{arm}')
            references[arm] = dict(source_file=str(path.relative_to(ROOT)), config=config,
                                   historical_runs=historical,
                                   split_fingerprint=raw['experiment_info'].get('split_fingerprint'))
        manifest['references'][ds] = references
        data = ROOT / 'data/SNAP' / ds
        paths.append(data / 'processed/data.pt')
        if ds == 'brightkite':
            paths.append(data / 'temporal_split.pt')
            expected = json.loads((ROOT / references['CAFE-In']['source_file']).read_text())['experiment_info']['split_sha256']
            if sha(data / 'temporal_split.pt') != expected:
                raise ValueError('Brightkite split differs from Table 4')
        if ds == 'pokec_n':
            table = data / 'region_job_2.csv'
            paths.append(table if table.is_file() else data / 'region_job_2.csv.gz')
        if ds == 'unc':
            paths.extend([data / 'processed/UNC28_user_sen.pkl',
                          data / 'UNC28/UNC28_train_items.pkl', data / 'UNC28/UNC28_test_set.pkl'])
        for arm in ARMS:
            manifest['tasks'].append(dict(index=len(manifest['tasks']), dataset=ds, arm=arm,
                                           runs=1 if arm == 'RA' else 10))
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest['frozen_sha256'][str(path.relative_to(ROOT))] = sha(path)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / 'authority.json', manifest)
    print(f'Prepared {len(manifest["tasks"])} tasks: {out}; Oklahoma omitted')


def command(manifest, ds, arm, folder):
    argv = pilot_command(arm, folder)
    argv[argv.index('--data_name') + 1] = ds
    argv[argv.index('--runs') + 1] = '1' if arm == 'RA' else '10'
    if arm in ['CAFE-In', 'CAFE-Pre']:
        row = manifest['configs'][ds]['cafe_in' if arm == 'CAFE-In' else 'cafe_pre']
        penalty = '--lambda_fair' if arm == 'CAFE-In' else '--preproc_alpha'
        for flag, value in [(penalty, row[penalty[2:]]), ('--beta_ra', row['beta']),
                            ('--lambda_mobility', row['lambda_mobility'])]:
            argv[argv.index(flag) + 1] = value
    if ds in ['pokec_n', 'brightkite'] and arm in ['Fusion', 'CAFE-In', 'CAFE-Pre']:
        argv += ['--eval_skip_test', 'True', '--eval_max_nodes', '5000',
                 '--eval_steps', '25', '--patience', '6']
    if ds == 'brightkite':
        argv += ['--split_file', str(ROOT / 'data/SNAP/brightkite/temporal_split.pt')]
    if arm == 'CAFE-Pre':
        argv = [sys.executable, '-u', str(CHILD), '--child', '--trace',
                str(folder / 'trace.jsonl'), '--', *argv[3:]]
    return argv


def summarize(manifest, ds, arm, folder):
    records = [json.loads(s) for s in (folder / 'acceptance.jsonl').read_text().splitlines()]
    reference = manifest['references'][ds].get(arm)
    configs = {f'lambda_fair_0.0_beta_{b}_lam_0.0' for b in BETAS} if arm == 'Fusion' else {reference['config']}
    n = 1 if arm == 'RA' else 10
    expected = {(r, c) for r in range(n) for c in configs}
    if len(records) != len(expected) or {(r['run'], r['config']) for r in records} != expected:
        raise ValueError('Incomplete/duplicate/unexpected run/config grid')
    for r in records:
        if r['dataset'] != ds or r['method'] != arm or r['k'] != 10 or r['predictions_saved']:
            raise ValueError('Acceptance provenance mismatch')
        if not r['full_matches_main'] or not all(math.isfinite(v) for v in [*r['main'].values(), *r['acceptance'].values()]):
            raise ValueError('Invalid Full/main or non-finite metrics')
        for a, b in [('Degree_EG@10_full', 'Equality_Gain@10'),
                     ('Pagerank_EG@10_full', 'Pagerank_EG@10'), ('Closeness_EG@10_full', 'Closeness_EG@10')]:
            if not math.isclose(r['acceptance'][a], r['main'][b], abs_tol=1e-6, rel_tol=1e-8):
                raise ValueError('Full/main numeric mismatch')
        if any(manifest['frozen_sha256'][p] != h for p, h in r['source_sha256'].items()):
            raise ValueError('Per-record source hash mismatch')
    if len({r['ordered_train_test_sha256'] for r in records}) != 1:
        raise ValueError('Split changed within fixed-split replay')
    # Check the original runner outputs independently of the acceptance record.
    if arm == 'RA':
        candidates = list((folder / 'main' / ds / 'baselines').glob('results_baselines_*.json'))
        if len(candidates) != 1:
            raise ValueError('Expected one completed baseline output')
        raw = json.loads(candidates[0].read_text())
        if any(records[0]['main'][m] != raw['results']['RA'][m]['values'][0] for m in METRICS):
            raise ValueError('RA original output differs from probe')
    else:
        raw = json.loads((folder / 'main_results.json').read_text())
        if len(raw['individual_runs']) != n:
            raise ValueError('Original runner output is incomplete')
        for r in records:
            if any(r['main'][m] != raw['individual_runs'][r['run']][r['config']][m][1] for m in METRICS):
                raise ValueError('Original output differs from probe')
        if reference and reference['split_fingerprint']:
            if raw['experiment_info'].get('split_fingerprint') != reference['split_fingerprint']:
                raise ValueError('Split fingerprint differs from authoritative Table 4')
    report = dict(dataset=ds, arm=arm, n=n, full_matches_main=True, descriptive_sd_ddof=0,
                  ordered_train_test_sha256=records[0]['ordered_train_test_sha256'],
                  aggregate_main={}, aggregate_acceptance={},
                  historical_replay_checked=reference is not None,
                  acceptance_sha256=sha(folder / 'acceptance.jsonl'))
    for config in sorted(configs):
        rows = sorted((r for r in records if r['config'] == config), key=lambda r: r['run'])
        for key in ['main', 'acceptance']:
            report['aggregate_' + key][config] = {
                m: dict(mean=statistics.mean(r[key][m] for r in rows),
                        std=statistics.pstdev(r[key][m] for r in rows), n=n) for m in rows[0][key]}
    if reference:
        ordered = sorted(records, key=lambda r: r['run'])
        deltas = [{m: r['main'][m] - h[m] for m in METRICS}
                  for r, h in zip(ordered, reference['historical_runs'])]
        report['per_run_delta_vs_table4'] = deltas
        report['all_raw_metrics_exact'] = all(v == 0 for r in deltas for v in r.values())
        report['status'] = 'exact_table4_replay' if report['all_raw_metrics_exact'] else 'needs_review_table4_mismatch'
    else:
        report['table11_config'] = f'lambda_fair_0.0_beta_{float(manifest["configs"][ds]["cafe_in"]["beta"])}_lam_0.0'
        report['status'] = 'complete_fusion_control_no_table4_row'
    if arm == 'CAFE-Pre':
        trace = [json.loads(s) for s in (folder / 'trace.jsonl').read_text().splitlines()]
        recs = [r for r in trace if r['stage'] == 'recommendations']
        before = [r['rng'] for r in trace if r['stage'] == 'acceptance_before']
        after = [r['rng'] for r in trace if r['stage'] == 'acceptance_after']
        if trace[-1]['stage'] != 'finished' or len(recs) != n or len(before) != n or before != after:
            raise ValueError('CAFE-Pre fingerprint/RNG validation failed')
        if any(t['top10_sha256'] != r['recommendations_sha256'] for t, r in zip(recs, sorted(records, key=lambda r:r['run']))):
            raise ValueError('CAFE-Pre recommendation fingerprint mismatch')
        report['acceptance_rng_unchanged'] = True
    return report


def execute(argv):
    proc = subprocess.Popen(argv, cwd=ROOT / 'experiments', stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    tail, budget = deque(maxlen=45), 64000
    for line in proc.stdout:
        line = line.rstrip()[:1200]
        tail.append(line)
        if any(s in line for s in ['RUN ', '[acceptance pilot]', 'Training complete', 'Error', 'Traceback', 'FATAL']):
            if budget > len(line) + 1:
                print(line, flush=True)
                budget -= len(line) + 1
    rc = proc.wait()
    if rc:
        print('FAILED; bounded tail:\n' + '\n'.join(tail), flush=True)
    return rc


def run(out, task_index):
    manifest = json.loads((out / 'authority.json').read_text())
    verify(manifest)
    task = manifest['tasks'][task_index]
    ds, arm = task['dataset'], task['arm']
    folder = out / ds / arm.lower()
    folder.mkdir(parents=True, exist_ok=False)
    argv = command(manifest, ds, arm, folder)
    write_new(folder / 'launch.json', dict(argv=argv, task=task,
              job_id=os.environ.get('SLURM_JOB_ID'), array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
              authority_sha256=sha(out / 'authority.json')))
    print(f'START: Table 11 aligned {ds}/{arm}; runs={task["runs"]}; no predictions/checkpoints', flush=True)
    rc = execute(argv)
    verify(manifest)
    report = dict(returncode=rc, dataset=ds, arm=arm, status='execution_failed')
    if rc == 0:
        try:
            report.update(summarize(manifest, ds, arm, folder))
        except (ValueError, KeyError, OSError) as exc:
            report.update(status='validation_failed', error=str(exc))
            rc = 1
    write_new(folder / 'replay_report.json', report)
    print(f'FINISHED: {ds}/{arm}: {report["status"]}; review before table export', flush=True)
    if rc:
        raise SystemExit(rc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--task-index', type=int, choices=range(20))
    args = parser.parse_args()
    out = args.output_root.resolve()
    if not out.is_relative_to(ROOT / 'results/validation'):
        raise ValueError('Use an isolated results/validation directory')
    if args.prepare:
        prepare(out)
    elif args.task_index is not None:
        run(out, args.task_index)
    else:
        parser.error('Specify --prepare or --task-index')
