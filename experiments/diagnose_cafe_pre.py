"""Low-disk CAFE-Pre probe-on/off reproducibility diagnostic, Oklahoma only.

Four fresh processes on one allocated GPU: off, on, off, on. Each runs one
original Table 4 repetition. Wrappers record hashes, not tensors or RNG states.
Original research files and frozen pilot/replay inputs are not edited.
"""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from pilot_table4_acceptance import ROOT, METRICS, command, sha, verify, write_new

REPLAY = ROOT / 'results/validation/20260905_112222_table4_acceptance_replay'
TRIALS = [('off_1', False), ('on_1', True), ('off_2', False), ('on_2', True)]


def prepare(out):
    manifest = json.loads((REPLAY / 'authority.json').read_text())
    verify(manifest)
    paths = [Path(__file__).resolve(), ROOT / 'experiments/jobs/cafe_pre_diagnostic.job',
             REPLAY / 'authority.json', REPLAY / 'cafe-pre/replay_report.json']
    manifest['frozen_sha256'].update({str(p.relative_to(ROOT)): sha(p) for p in paths})
    manifest['purpose'] = 'CAFE-Pre diagnostic: four independent one-run processes; no table updates'
    manifest['runs'] = 1
    manifest['protocol'] = 'cafe_pre_one_run_probe_diagnostic'
    manifest.pop('n_by_method', None)
    manifest['diagnostic_trials'] = TRIALS
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / 'authority.json', manifest)
    print(f'Prepared diagnostic: {out}')


def child(trace_path, runner_args):
    # Match the original runner's import setup in both diagnostic conditions.
    sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'backbone'),
                    str(ROOT / 'fairness_methods'), str(ROOT)]
    import pickle
    import random
    import platform
    import importlib.metadata
    import numpy as np
    import torch
    import run_cafe as runner
    import core.acceptance_probe as probe

    def tensor_hash(tensor):
        if tensor is None:
            return None
        tensor = tensor.detach().cpu().contiguous()
        digest = hashlib.sha256(str((str(tensor.dtype), list(tensor.shape))).encode())
        digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    def rng_hashes():
        return dict(python=hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
                    numpy=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
                    torch_cpu=tensor_hash(torch.get_rng_state()),
                    torch_cuda=[tensor_hash(s) for s in torch.cuda.get_rng_state_all()]
                    if torch.cuda.is_initialized() else [])

    def adjacency_hash(adj):
        row, col, val = adj.coo()
        return dict(row=tensor_hash(row), col=tensor_hash(col), value=tensor_hash(val))

    def weights_hash(model):
        digest = hashlib.sha256()
        # FairnessAwareModel is a BaseModel wrapper, not torch.nn.Module.
        # Hash the same components saved/restored by train_inprocessing_model.
        for component in ['encoder', 'predictor', 'emb']:
            module = getattr(model, component)
            digest.update(component.encode())
            if module is None:
                digest.update(b'None')
                continue
            for name, tensor in sorted(module.state_dict().items()):
                digest.update(name.encode())
                digest.update(tensor_hash(tensor).encode())
        return digest.hexdigest()

    def emit(stage, **fields):
        with trace_path.open('a') as handle:
            handle.write(json.dumps(dict(stage=stage, **fields), allow_nan=False) + '\n')

    if trace_path.exists():
        raise ValueError('Refusing to overwrite an existing trace')
    if not torch.cuda.is_available():
        raise RuntimeError('Diagnostic requires the SLURM-allocated GPU; no silent CPU fallback')
    packages = {}
    for package in ['torch', 'numpy', 'scipy', 'networkit', 'torch-geometric', 'torch-sparse']:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = 'not found'
    emit('environment', hostname=platform.node(), python=sys.version, packages=packages,
         cuda_build=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
         cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
         cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
         omp_num_threads=os.environ.get('OMP_NUM_THREADS'),
         torch_threads=torch.get_num_threads(),
         deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
         cudnn_deterministic=torch.backends.cudnn.deterministic,
         cudnn_benchmark=torch.backends.cudnn.benchmark,
         matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
         gpu_name=torch.cuda.get_device_name(0),
         gpu_properties=str(torch.cuda.get_device_properties(0)))
    gpu_tool = shutil.which('nvidia-smi')
    if gpu_tool:
        try:
            gpu = subprocess.run([gpu_tool, '--query-gpu=index,name,uuid,driver_version',
                                  '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
            emit('gpu_inventory', available=True, returncode=gpu.returncode, inventory=gpu.stdout[:4096])
        except (OSError, subprocess.TimeoutExpired) as exc:
            emit('gpu_inventory', available=False, reason=str(exc))
    else:
        emit('gpu_inventory', available=False,
             reason='nvidia-smi is not on PATH; Torch GPU properties and allocation identifiers recorded above')

    original_preproc = runner.apply_coreness_inverse_aggregation
    def preproc(adj, coreness, *args, **kwargs):
        with probe.preserve_training_rng():
            emit('before_preproc', rng=rng_hashes(), adjacency=adjacency_hash(adj),
                 coreness=tensor_hash(coreness))
        result = original_preproc(adj, coreness, *args, **kwargs)
        with probe.preserve_training_rng():
            emit('after_preproc', rng=rng_hashes(), adjacency=adjacency_hash(result))
        return result
    runner.apply_coreness_inverse_aggregation = preproc

    original_train = runner.train_inprocessing_model
    def train(model, data, split_edge, args, degrees, **kwargs):
        with probe.preserve_training_rng():
            emit('train_entry', rng=rng_hashes(), weights=weights_hash(model),
                 adjacency=adjacency_hash(data.adj_t),
                 train_edges=tensor_hash(split_edge['train']['edge']),
                 valid_edges=tensor_hash(split_edge['valid']['edge']),
                 test_edges=tensor_hash(split_edge['test']['edge']))
        result = original_train(model, data, split_edge, args, degrees, **kwargs)
        with probe.preserve_training_rng():
            emit('train_exit', rng=rng_hashes(), weights=weights_hash(result[0]),
                 best_epoch=int(result[1]))
        return result
    runner.train_inprocessing_model = train

    original_metrics = runner.compute_metrics_from_recs
    def metrics(recs, *args, **kwargs):
        result = original_metrics(recs, *args, **kwargs)
        with probe.preserve_training_rng():
            digest = hashlib.sha256()
            for u, vs in sorted(recs.items()):
                digest.update(np.asarray([u, *vs[:10]], dtype='<i8').tobytes())
            emit('recommendations', rng=rng_hashes(), top10_sha256=digest.hexdigest(),
                 precision=float(result['Precision@10'][1]))
        return result
    runner.compute_metrics_from_recs = metrics

    original_probe = probe.record_acceptance
    def acceptance(*args, **kwargs):
        before = rng_hashes()
        emit('acceptance_before', rng=before)
        try:
            return original_probe(*args, **kwargs)
        finally:
            after = rng_hashes()
            emit('acceptance_after', rng=after, training_rng_unchanged=(before == after))
    probe.record_acceptance = acceptance
    sys.argv = ['run_cafe.py', *runner_args]
    runner.main()
    emit('finished')


def launch(out):
    manifest = json.loads((out / 'authority.json').read_text())
    verify(manifest)
    reference = manifest['references']['CAFE-Pre']
    reports = {}
    for name, enabled in TRIALS:
        verify(manifest)
        folder = out / name
        folder.mkdir(exist_ok=False)
        args = command('CAFE-Pre', folder)[3:]
        if not enabled:
            idx = args.index('--acceptance_probe_jsonl')
            del args[idx:idx+2]
        argv = [sys.executable, '-u', str(Path(__file__).resolve()), '--child',
                '--trace', str(folder / 'trace.jsonl'), '--', *args]
        write_new(folder / 'launch.json', dict(argv=argv, probe_enabled=enabled,
                  job_id=os.environ.get('SLURM_JOB_ID'), authority_sha256=sha(out / 'authority.json')))
        print(f'DIAGNOSTIC START: {name}; one Oklahoma CAFE-Pre repetition', flush=True)
        proc = subprocess.Popen(argv, cwd=ROOT / 'experiments', stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        tail, budget = deque(maxlen=30), 16000
        for line in proc.stdout:
            line = line.rstrip()[:1000]
            tail.append(line)
            if any(s in line for s in ['RUN ', '[acceptance pilot]', 'Training complete', 'Error', 'Traceback']):
                if budget > len(line)+1:
                    print(line, flush=True)
                    budget -= len(line)+1
        rc = proc.wait()
        reports[name] = dict(returncode=rc, probe_enabled=enabled)
        if rc:
            print('TRIAL FAILED:\n' + '\n'.join(tail), flush=True)
            continue
        verify(manifest)
        raw = json.loads((folder / 'main_results.json').read_text())
        values = raw['individual_runs'][0][reference['config']]
        reports[name]['main'] = {m: float(values[m][1]) for m in METRICS}
        reports[name]['delta_vs_table4_first_run'] = {
            m: float(values[m][1]) - reference['first_run'][m] for m in METRICS}
        reports[name]['trace_sha256'] = sha(folder / 'trace.jsonl')
        print(f'DIAGNOSTIC FINISHED: {name}', flush=True)
    verify(manifest)
    write_new(out / 'diagnostic_report.json', dict(trials=reports,
              caution='Hashes synchronize GPU execution; compare within this diagnostic before drawing conclusions. No checkpoints or predictions saved.'))
    print(json.dumps(reports), flush=True)
    if any(r['returncode'] for r in reports.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--trace', type=Path)
    parser.add_argument('runner_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.child:
        if args.trace is None:
            parser.error('--child requires --trace')
        child(args.trace, args.runner_args[1:] if args.runner_args[:1] == ['--'] else args.runner_args)
    else:
        if args.output_root is None:
            parser.error('--output-root is required')
        out = args.output_root.resolve()
        if not out.is_relative_to(ROOT / 'results/validation'):
            raise ValueError('Use an isolated results/validation directory')
        prepare(out) if args.prepare else launch(out)
