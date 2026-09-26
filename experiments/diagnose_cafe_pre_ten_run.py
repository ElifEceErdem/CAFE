"""Controlled Oklahoma CAFE-Pre ten-run probe-off/on diagnostic.

Reuse the frozen one-run fingerprint wrapper unchanged. Two fresh processes
share one SLURM GPU allocation. Preserve every result, whether it agrees or not.
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

from pilot_table4_acceptance import ROOT, METRICS, command, sha, verify, write_new
from replay_table4_acceptance import validate_records

PREVIOUS = ROOT / 'results/validation/20260905_134051_cafe_pre_diagnostic'
CHILD = ROOT / 'experiments/diagnose_cafe_pre.py'
TRIALS = [('off_10', False), ('on_10', True)]
STAGES = ['before_preproc', 'after_preproc', 'train_entry', 'train_exit', 'recommendations']


def prepare(out):
    manifest = json.loads((PREVIOUS / 'authority.json').read_text())
    verify(manifest)
    previous = json.loads((PREVIOUS / 'diagnostic_report.json').read_text())
    if len(previous['trials']) != 4 or any(
        r['returncode'] or any(r['delta_vs_table4_first_run'].values())
        for r in previous['trials'].values()
    ):
        raise ValueError('The preceding four-trial diagnostic did not pass')
    paths = [Path(__file__).resolve(), ROOT / 'experiments/jobs/cafe_pre_ten_run_diagnostic.job',
             PREVIOUS / 'authority.json', PREVIOUS / 'diagnostic_report.json']
    manifest['frozen_sha256'].update({str(p.relative_to(ROOT)): sha(p) for p in paths})
    manifest.update(runs=10, diagnostic_trials=TRIALS,
                    protocol='cafe_pre_ten_run_probe_diagnostic',
                    purpose='Two controlled ten-run processes; compare every run; no table updates',
                    descriptive_sd_ddof=0,
                    interpretation='No retries based on agreement. GPU fingerprinting can affect timing.')
    ref = manifest['references']['CAFE-Pre']
    raw = json.loads((ROOT / ref['source_file']).read_text())
    historical = [{m: float(r[ref['config']][m][1]) for m in METRICS}
                  for r in raw['individual_runs']]
    if len(historical) != 10 or historical != ref['historical_runs']:
        raise ValueError('Historical ten-run reference does not match its source')
    verify(manifest)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / 'authority.json', manifest)
    print(f'Prepared: {out}')


def runner_args(folder, enabled):
    args = command('CAFE-Pre', folder)[3:]
    args[args.index('--runs') + 1] = '10'
    if not enabled:
        idx = args.index('--acceptance_probe_jsonl')
        del args[idx:idx + 2]
    return args


def summarize(folder, manifest, enabled):
    ref = manifest['references']['CAFE-Pre']
    raw = json.loads((folder / 'main_results.json').read_text())
    runs = [{m: float(r[ref['config']][m][1]) for m in METRICS}
            for r in raw['individual_runs']]
    if len(runs) != 10 or not all(math.isfinite(v) for r in runs for v in r.values()):
        raise ValueError('Expected ten finite main-metric observations')
    info = raw['experiment_info']
    if (info['num_runs'] != 10 or info['model_seeds'] != list(range(42, 52))
            or info['dataset'] != 'oklahoma' or info['split_seed'] != 42):
        raise ValueError('Unexpected run/seed/dataset protocol')
    traces = [json.loads(s) for s in (folder / 'trace.jsonl').read_text().splitlines()]
    if not traces or traces[-1]['stage'] != 'finished':
        raise ValueError('Incomplete trace')
    expected_stages = ['environment', 'gpu_inventory']
    for _ in range(10):
        expected_stages += STAGES + (['acceptance_before', 'acceptance_after'] if enabled else [])
    expected_stages += ['finished']
    if [r['stage'] for r in traces] != expected_stages:
        raise ValueError('Unexpected trace sequence; cannot safely align repetitions')
    deltas = [{m: r[m] - h[m] for m in METRICS}
              for r, h in zip(runs, ref['historical_runs'])]
    report = dict(main=runs, per_run_delta_vs_table4=deltas,
                  all_raw_metrics_exact=all(v == 0 for r in deltas for v in r.values()),
                  trace_sha256=sha(folder / 'trace.jsonl'), descriptive_sd_ddof=0,
                  aggregate_main={m: dict(mean=statistics.mean(r[m] for r in runs),
                                          std=statistics.pstdev(r[m] for r in runs), n=10)
                                  for m in METRICS})
    if enabled:
        records = [json.loads(s) for s in (folder / 'acceptance.jsonl').read_text().splitlines()]
        validate_records(records, manifest, 'CAFE-Pre')
        records.sort(key=lambda r: r['run'])
        rec_traces = [r for r in traces if r['stage'] == 'recommendations']
        for r, main, trace in zip(records, runs, rec_traces):
            if r['main'] != main or r['recommendations_sha256'] != trace['top10_sha256']:
                raise ValueError('Acceptance/main/trace observations are not aligned')
        before = [r['rng'] for r in traces if r['stage'] == 'acceptance_before']
        after = [r['rng'] for r in traces if r['stage'] == 'acceptance_after']
        report['acceptance_rng_unchanged'] = before == after
        report['full_matches_main'] = True
        report['aggregate_acceptance'] = {
            m: dict(mean=statistics.mean(r['acceptance'][m] for r in records),
                    std=statistics.pstdev(r['acceptance'][m] for r in records), n=10)
            for m in records[0]['acceptance']}
    return report


def compare_traces(left, right):
    def shared(path):
        return [json.loads(s) for s in path.read_text().splitlines()
                if not json.loads(s)['stage'].startswith('acceptance_')]
    a, b = shared(left), shared(right)
    differences = [dict(shared_event_index=i, left_stage=x['stage'], right_stage=y['stage'],
                        differing_fields=sorted(k for k in x.keys() | y.keys() if x.get(k) != y.get(k)))
                   for i, (x, y) in enumerate(zip(a, b)) if x != y]
    return dict(shared_event_counts=[len(a), len(b)],
                shared_traces_identical=a == b, differences=differences,
                first_difference=differences[0] if differences else None)


def launch(out):
    manifest = json.loads((out / 'authority.json').read_text())
    verify(manifest)
    reports = {}
    for name, enabled in TRIALS:
        verify(manifest)
        folder = out / name
        folder.mkdir(exist_ok=False)
        argv = [sys.executable, '-u', str(CHILD), '--child', '--trace',
                str(folder / 'trace.jsonl'), '--', *runner_args(folder, enabled)]
        write_new(folder / 'launch.json', dict(argv=argv, probe_enabled=enabled,
                  job_id=os.environ.get('SLURM_JOB_ID'), authority_sha256=sha(out / 'authority.json')))
        print(f'DIAGNOSTIC START: {name}; ten Oklahoma CAFE-Pre repetitions', flush=True)
        proc = subprocess.Popen(argv, cwd=ROOT / 'experiments', stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        tail, budget = deque(maxlen=40), 48000
        for line in proc.stdout:
            line = line.rstrip()[:1200]
            tail.append(line)
            if any(s in line for s in ['RUN ', '[acceptance pilot]', 'Training complete', 'Error', 'Traceback']):
                if budget > len(line) + 1:
                    print(line, flush=True)
                    budget -= len(line) + 1
        rc = proc.wait()
        report = dict(returncode=rc, probe_enabled=enabled)
        if rc:
            print('TRIAL FAILED:\n' + '\n'.join(tail), flush=True)
        else:
            verify(manifest)
            try:
                report.update(summarize(folder, manifest, enabled))
            except (ValueError, KeyError, OSError) as exc:
                report['validation_error'] = str(exc)
        reports[name] = report
        write_new(folder / 'diagnostic_report.json', report)
        print(f'DIAGNOSTIC FINISHED: {name}; returncode={rc}; '
              f'exact_table4={report.get("all_raw_metrics_exact")}', flush=True)
    verify(manifest)
    result = dict(trials=reports, predictions_saved=False, checkpoints_saved=False,
                  caution='Instrumented comparison only; mismatches are retained, not retried. No table updates.')
    valid = all(not r['returncode'] and 'validation_error' not in r for r in reports.values())
    if valid:
        result['off_vs_on'] = compare_traces(out / 'off_10/trace.jsonl', out / 'on_10/trace.jsonl')
        result['off_vs_on']['per_run_main_deltas'] = [
            {m: b[m] - a[m] for m in METRICS}
            for a, b in zip(reports['off_10']['main'], reports['on_10']['main'])]
    write_new(out / 'diagnostic_report.json', result)
    print(f'Diagnostic complete: {out / "diagnostic_report.json"}', flush=True)
    if not valid:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    out = args.output_root.resolve()
    if not out.is_relative_to(ROOT / 'results/validation'):
        raise ValueError('Use an isolated results/validation directory')
    prepare(out) if args.prepare else launch(out)
