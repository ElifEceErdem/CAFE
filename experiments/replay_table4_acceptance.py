"""Ten-repetition Oklahoma acceptance replay using the validated pilot path.

Reuses the completed RA measurement (n=1, as in authoritative Table 4).
No predictions or checkpoints; the research runners and pilot remain unchanged.
"""
import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

from pilot_table4_acceptance import ROOT, ARMS, METRICS, command as pilot_command
from pilot_table4_acceptance import sha, verify, write_new

PILOT = ROOT / 'results/validation/20260905_110459_table4_acceptance_pilot'


def command(arm, folder):
    if arm == 'RA':
        raise ValueError('RA is reused, not resubmitted or duplicated into ten observations')
    argv = pilot_command(arm, folder)
    argv[argv.index('--runs') + 1] = '10'
    return argv


def prepare(out):
    manifest = json.loads((PILOT / 'authority.json').read_text())
    verify(manifest)
    for arm in ARMS:
        report_path = PILOT / arm.lower() / 'pilot_report.json'
        report = json.loads(report_path.read_text())
        if not report['full_matches_main'] or (arm != 'Fusion' and not report['matches_historical_display']):
            raise ValueError(f'Pilot gate failed: {arm}')
        manifest['frozen_sha256'][str(report_path.relative_to(ROOT))] = sha(report_path)
    manifest['runs'] = 10
    manifest['purpose'] = 'Oklahoma Table 11 replay; compare ten raw runs and mean/SD with Table 4'
    manifest['protocol'] = 'table4_fixed_split_replay'
    manifest['n_by_method'] = {arm: 1 if arm == 'RA' else 10 for arm in ARMS}
    ra_path = PILOT / 'ra/acceptance.jsonl'
    ra_records = [json.loads(s) for s in ra_path.read_text().splitlines()]
    if len(ra_records) != 1 or ra_records[0]['main'] != manifest['references']['RA']['first_run']:
        raise ValueError('Reused RA evidence does not match the authoritative measurement')
    manifest['reused_ra_file'] = str(ra_path.relative_to(ROOT))
    manifest['ordered_train_test_sha256'] = ra_records[0]['ordered_train_test_sha256']
    for arm in ['HL-GNN', 'CAFE-In', 'CAFE-Pre']:
        ref = manifest['references'][arm]
        source = json.loads((ROOT / ref['source_file']).read_text())
        runs = source['individual_runs']
        if len(runs) != 10:
            raise ValueError(f'Expected ten authoritative runs: {arm}')
        ref['historical_runs'] = [{m: float(r[ref['config']][m][1]) for m in METRICS} for r in runs]
    for path in [PILOT / 'authority.json', ra_path, Path(__file__).resolve(),
                 ROOT / 'experiments/jobs/t04_acceptance_replay.job']:
        manifest['frozen_sha256'][str(path.relative_to(ROOT))] = sha(path)
    verify(manifest)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / 'authority.json', manifest)
    print(f'Prepared ten-repetition replay: {out}; RA reused once')


def validate_records(records, manifest, arm):
    configs = ({f'lambda_fair_0.0_beta_{b}_lam_0.0' for b in ['0.1', '0.5', '1.0', '2.0', '5.0']}
               if arm == 'Fusion' else {manifest['references'][arm]['config']})
    expected = {(r, c) for r in range(10) for c in configs}
    if len(records) != len(expected) or {(r['run'], r['config']) for r in records} != expected:
        raise ValueError('Missing, duplicate or unexpected run/config records')
    for row in records:
        if (row['dataset'] != 'oklahoma' or row['method'] != arm or row['predictions_saved']
                or not row['full_matches_main'] or row['k'] != 10
                or row['ordered_train_test_sha256'] != manifest['ordered_train_test_sha256']):
            raise ValueError('Record provenance or Full/main check failed')
        for key, digest in row['source_sha256'].items():
            if manifest['frozen_sha256'].get(key) != digest:
                raise ValueError(f'Per-record source changed: {key}')
        if not all(math.isfinite(v) for v in [*row['main'].values(), *row['acceptance'].values()]):
            raise ValueError('Non-finite metrics')
        for a, b in [('Degree_EG@10_full', 'Equality_Gain@10'),
                     ('Pagerank_EG@10_full', 'Pagerank_EG@10'),
                     ('Closeness_EG@10_full', 'Closeness_EG@10')]:
            if not math.isclose(row['acceptance'][a], row['main'][b], rel_tol=1e-8, abs_tol=1e-6):
                raise ValueError('Full/main values differ')


def summarize(records, manifest, arm):
    validate_records(records, manifest, arm)
    report = dict(status='calculation_complete', method=arm, runs=10,
                  full_matches_main=True, historical_replay_checked=arm != 'Fusion',
                  protocol='table4_fixed_split_replay', predictions_saved=False,
                  aggregate_main={}, aggregate_acceptance={})
    for config in sorted({r['config'] for r in records}):
        selected = sorted([r for r in records if r['config'] == config], key=lambda r: r['run'])
        for src, dest in [('main', 'aggregate_main'), ('acceptance', 'aggregate_acceptance')]:
            report[dest][config] = {metric: {
                'mean': statistics.mean(r[src][metric] for r in selected),
                'std': statistics.stdev(r[src][metric] for r in selected), 'n': 10}
                for metric in selected[0][src]}
    if arm != 'Fusion':
        reference = manifest['references'][arm]
        ordered = sorted(records, key=lambda r: r['run'])
        report['per_run_delta_vs_table4'] = [
            {m: r['main'][m] - h[m] for m in METRICS}
            for r, h in zip(ordered, reference['historical_runs'])]
        report['aggregate_comparison'] = {}
        for m in METRICS:
            values = [h[m] for h in reference['historical_runs']]
            historical = dict(mean=statistics.mean(values), std=statistics.stdev(values))
            current = report['aggregate_main'][reference['config']][m]
            decimals = 4 if m == 'Precision@10' else 2
            report['aggregate_comparison'][m] = dict(
                historical=historical, replayed=current,
                matches_display=all(f'{historical[k]:.{decimals}f}' == f'{current[k]:.{decimals}f}'
                                    for k in ['mean', 'std']))
        report['matches_table4_mean_and_sd_display'] = all(
            c['matches_display'] for c in report['aggregate_comparison'].values())
        report['all_raw_metrics_exact'] = all(v == 0 for r in report['per_run_delta_vs_table4'] for v in r.values())
    return report


def run(out, arm_index):
    manifest = json.loads((out / 'authority.json').read_text())
    verify(manifest)
    arm = ARMS[arm_index]
    folder = out / arm.lower()
    folder.mkdir(exist_ok=False)
    argv = command(arm, folder)
    write_new(folder / 'launch.json', dict(argv=argv, job_id=os.environ.get('SLURM_JOB_ID'),
              array_task=os.environ.get('SLURM_ARRAY_TASK_ID'), authority_sha256=sha(out / 'authority.json')))
    print(f'REPLAY START: {arm}; Oklahoma; ten repetitions from original RNG start', flush=True)
    proc = subprocess.Popen(argv, cwd=ROOT / 'experiments', env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail, budget = deque(maxlen=40), 48000
    for line in proc.stdout:
        line = line.rstrip()[:1200]
        tail.append(line)
        if any(token in line for token in ['RUN ', '[acceptance pilot]', 'Saved ', 'Error', 'Traceback', 'FATAL']):
            if budget > len(line) + 1:
                print(line, flush=True)
                budget -= len(line) + 1
    rc = proc.wait()
    if rc:
        print('REPLAY FAILED; last bounded output:\n' + '\n'.join(tail), flush=True)
        raise SystemExit(rc)
    verify(manifest)
    records = [json.loads(s) for s in (folder / 'acceptance.jsonl').read_text().splitlines()]
    report = summarize(records, manifest, arm)
    report['acceptance_jsonl_sha256'] = sha(folder / 'acceptance.jsonl')
    write_new(folder / 'replay_report.json', report)
    print('REPLAY FINISHED: ' + json.dumps({k: v for k, v in report.items() if isinstance(v, (str, bool, int))}), flush=True)
    print('Review replay_report.json before updating tables or submitting other datasets.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--arm-index', type=int, choices=range(1, 5))
    args = parser.parse_args()
    out = args.output_root.resolve()
    if not out.is_relative_to(ROOT / 'results/validation'):
        raise ValueError('Use an isolated results/validation directory')
    if args.prepare:
        prepare(out)
    elif args.arm_index is not None:
        run(out, args.arm_index)
    else:
        parser.error('Specify --prepare or --arm-index')
