"""Bounded, diagnostic-only exact generalized-closeness validation.

Calls the existing exact path explicitly; never patches production metrics.
Control is a 20-minute gate; RA replay is capped at six hours. Stores scalars,
quantiles and hashes only, not recommendations, node scores, or checkpoints.
"""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'analysis'), str(ROOT / 'experiments'), str(ROOT / 'backbone'), str(ROOT)]
import networkit as nk
import numpy as np
import torch
from diagnose_closeness_tp import append, distribution
from metrics.centrality_equality_gain import _compute_closeness_networkit, _build_networkit_graph
from metrics.graph_utils import build_control_edge_set, build_predicted_edge_set
from metrics.equality_gain import compute_gini_coefficient

DATASETS = ['brightkite', 'pokec_n']
REFERENCE = ROOT / 'results/validation/20260905_163304_table11_table4_aligned_v3'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def split_sha(train, test, n):
    h = hashlib.sha256(f'num_nodes={n}\n'.encode())
    for label, edges in [('train', train), ('test', test)]:
        h.update(label.encode())
        h.update(edges.detach().cpu().numpy().astype('<i8').tobytes())
    return h.hexdigest()


def graph_sha(edges, n):
    h = hashlib.sha256(f'num_nodes={n}\n'.encode())
    h.update(edges.detach().cpu().numpy().astype('<i8').tobytes())
    return h.hexdigest()


def self_test():
    cases = [('path', 3, [(0, 1), (1, 2)]),
             ('disconnected_with_isolate', 6, [(0, 1), (1, 2), (3, 4)]),
             ('all_isolates', 4, []), ('single_node', 1, []),
             ('unequal_components', 7, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 6)]),
             ('duplicate_reciprocal_loop', 4, [(0, 1), (1, 0), (0, 1), (1, 2), (2, 2)])]
    results = []
    for label, n, pairs in cases:
        edges = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)
        adjacency = [set() for _ in range(n)]
        for u, v in pairs:
            if u != v:
                adjacency[u].add(v)
                adjacency[v].add(u)
        expected = []
        for u in range(n):
            distances = {u: 0}
            queue = deque([u])
            while queue:
                v = queue.popleft()
                for w in adjacency[v]:
                    if w not in distances:
                        distances[w] = distances[v] + 1
                        queue.append(w)
            reachable = len(distances) - 1
            expected.append(reachable**2 / ((n-1)*sum(distances.values())) if reachable else 0.)
        actual = _compute_closeness_networkit(edges, n, use_sampling=False).numpy()
        assert np.isfinite(actual).all() and np.all((actual >= 0) & (actual <= 1))
        assert np.allclose(actual, expected, rtol=1e-6, atol=2e-7), label
        positive = np.asarray(expected)[np.asarray(expected) > 0]
        pairwise_gini = (np.abs(positive[:, None] - positive[None, :]).sum() /
                         (2 * len(positive) * positive.sum())) if len(positive) else 0.
        gini = compute_gini_coefficient(torch.tensor(actual))
        assert abs(gini - pairwise_gini) < 3e-7, (label, gini, pairwise_gini)
        results.append(dict(case=label, nodes=n, score_max_abs_error=float(np.max(np.abs(actual-expected))),
                            gini_abs_error=float(abs(gini-pairwise_gini)), passed=True))
    return results


def exact_summary(edges, n, graph_name, output):
    append(output, dict(kind='exact_started', graph=graph_name, nodes=n, edges=len(edges)))
    start = time.monotonic()
    # This is the unchanged repository path, including its float32 score output.
    scores = _compute_closeness_networkit(edges, n, use_sampling=False)
    seconds = time.monotonic() - start
    values = scores.numpy()
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1 + 1e-7)):
        raise ValueError(f'Invalid exact scores on {graph_name}; refusing interpretation')
    graph = _build_networkit_graph(edges, n)
    cc = nk.components.ConnectedComponents(graph).run()
    components = cc.getComponents()
    largest = max(components, key=len)
    mask = np.zeros(n, dtype=bool)
    mask[largest] = True
    isolates = sum(len(c) == 1 for c in components)
    assert int((values == 0).sum()) == isolates
    # Independent single-source BFS spot checks, including small components.
    chosen = {c[0] for c in sorted(components, key=len, reverse=True)[:8]}
    chosen.update(int(v) for v in np.random.default_rng(9217).choice(n, min(n, 8), replace=False))
    errors = []
    for node in sorted(chosen):
        distances = np.asarray(nk.distance.BFS(graph, node, storePaths=False).run().getDistances())
        finite_distances = distances[(distances > 0) & (distances < n)]
        reachable = len(finite_distances)
        expected = reachable**2 / ((n-1)*finite_distances.sum()) if reachable else 0.
        errors.append(abs(float(values[node]) - expected))
    assert max(errors, default=0) < 2e-7
    g32 = compute_gini_coefficient(scores)
    g64 = compute_gini_coefficient(scores.to(torch.float64))
    assert np.isfinite([g32, g64]).all() and -.000001 <= g32 <= 1.000001
    record = dict(kind='exact_scores', graph=graph_name, graph_sha256=graph_sha(edges, n),
                  metric='existing exact generalized closeness; float32 scores', seconds=seconds,
                  nodes=n, edges=len(edges), components=len(components), isolates=isolates,
                  formula='r(u)^2 / ((n-1) * sum_reachable_distances); isolated nodes = 0',
                  gini_support='positive scores, unchanged production policy',
                  gini_float32=g32, gini_float64_on_float32_scores=g64,
                  all_nodes=distribution(values), largest_component=distribution(values[mask]),
                  other_components=distribution(values[~mask]), bfs_checks=len(chosen),
                  bfs_max_abs_error=max(errors, default=0), passed=True)
    append(output, record)
    return record


def verify(manifest):
    if nk.__version__ != manifest['networkit_version'] or torch.__version__ != manifest['torch_version']:
        raise ValueError('Library version changed since preparation')
    for name, expected in manifest['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'Frozen pilot input changed: {name}')


def control(ds, folder):
    from core import set_seed
    from core.data_loader import load_dataset
    set_seed(42)
    _, split, data, _ = load_dataset(SimpleNamespace(data_name=ds, data_path=str(ROOT/'data/SNAP'),
        encoder='HLGNN', device='cpu', val_ratio=.05, test_ratio=.1, split_file=None, data_split_seed=42))
    train, test, n = split['train']['edge'], split['test']['edge'], data.num_nodes
    reference = json.loads((REFERENCE / ds / 'ra/acceptance.jsonl').read_text())
    fingerprint = split_sha(train, test, n)
    assert fingerprint == reference['ordered_train_test_sha256']
    output = folder / 'control.jsonl'
    append(output, dict(kind='split_identity', ordered_train_test_sha256=fingerprint, matches_table11=True))
    exact_summary(build_control_edge_set(train, test, n), n, 'control', output)
    append(output, dict(kind='complete', phase='control'))


def replay(ds, folder):
    import core.acceptance_probe as probe
    import run_baselines
    from replay_table11_aligned import command
    from metrics.acceptance_eg import build_test_pair_set, filter_recs_tp
    control_rows = [json.loads(s) for s in (folder/'control.jsonl').read_text().splitlines()]
    assert control_rows[-1]['kind'] == 'complete'
    ctrl = next(r for r in control_rows if r['kind'] == 'exact_scores')
    reference = json.loads((REFERENCE / ds / 'ra/acceptance.jsonl').read_text())
    original = probe.record_acceptance
    seen = []
    output = folder / 'ra_exact.jsonl'

    def capture(path, **kw):
        original(path, **kw)  # Unchanged legacy evaluation: identity check only.
        actual = json.loads(Path(path).read_text().splitlines()[-1])
        for key in ['recommendations_sha256', 'ordered_train_test_sha256', 'main', 'acceptance']:
            if actual[key] != reference[key]:
                raise ValueError(f'RA replay differs from certified reference: {key}')
        append(output, dict(kind='ra_identity', hashes_and_legacy_metrics_exact=True,
            recommendations_sha256=actual['recommendations_sha256'],
            ordered_train_test_sha256=actual['ordered_train_test_sha256']))
        train, test, n = kw['train_edges'], kw['test_edges'], kw['num_nodes']
        edges_ctrl = build_control_edge_set(train, test, n)
        assert graph_sha(edges_ctrl, n) == ctrl['graph_sha256']
        tp = filter_recs_tp(kw['recs'], build_test_pair_set(test), 10)
        result = exact_summary(build_predicted_edge_set(train, tp, n, 10), n, 'RA_TP', output)
        if ctrl['gini_float32'] <= 0 or ctrl['gini_float64_on_float32_scores'] <= 0:
            raise ValueError('Zero control Gini; EG undefined')
        eg = 100 * (ctrl['gini_float32']-result['gini_float32']) / ctrl['gini_float32']
        eg64 = 100 * (ctrl['gini_float64_on_float32_scores']-result['gini_float64_on_float32_scores']) / ctrl['gini_float64_on_float32_scores']
        append(output, dict(kind='pilot_eg', exact_ra_tp_eg=eg, float64_gini_ra_tp_eg=eg64,
            legacy_approximate_ra_tp_eg=reference['acceptance']['Closeness_EG@10_tp'],
            production_change_approved=False, paper_replacement=False, predictions_saved=False))
        seen.append(True)
    probe.record_acceptance = capture
    argv = command({}, ds, 'RA', folder / 'legacy_replay')
    sys.argv = [argv[2], *argv[3:]]
    os.chdir(ROOT / 'experiments')
    code = run_baselines.main()
    if code or len(seen) != 1:
        raise ValueError('RA exact pilot did not complete')
    append(output, dict(kind='complete', phase='ra_exact'))


def prepare(out):
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    if branch != 'repo-cleanup':
        raise ValueError('Prepare only on repo-cleanup')
    tests = self_test()
    # Reuse the prior frozen Table 11 authority for original scientific dependencies.
    previous = json.loads((REFERENCE/'authority.json').read_text())
    from replay_table11_aligned import verify as verify_previous
    verify_previous(previous)
    frozen = dict(previous['frozen_sha256'])
    for name in ['analysis/pilot_exact_closeness.py', 'analysis/diagnose_closeness_tp.py',
                 'experiments/jobs/closeness_exact_pilot.job']:
        frozen[name] = sha(ROOT/name)
    for ds in DATASETS:
        name = str((REFERENCE/ds/'ra/acceptance.jsonl').relative_to(ROOT))
        frozen[name] = sha(ROOT/name)
    manifest = dict(protocol='exact_closeness_runtime_pilot_only', datasets=DATASETS,
        branch=branch, networkit_version=nk.__version__, torch_version=torch.__version__,
        control_timeout_seconds=1200, replay_timeout_seconds=21600, threads=4,
        predictions_saved=False, production_code_modified=False, self_tests=tests, source_sha256=frozen)
    out.mkdir(parents=True, exist_ok=False)
    with (out/'manifest.json').open('x') as handle:
        json.dump(manifest, handle, indent=2)
    print(f'PREPARED {out}: six independent toy/BFS/Gini tests passed; no jobs submitted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--dataset', choices=DATASETS)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test', action='store_true')
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--child', choices=['control', 'ra'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    nk.setNumberOfThreads(4)
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return
    if args.output_root is None:
        parser.error('--output-root required')
    out = args.output_root.resolve()
    if args.prepare:
        prepare(out)
        return
    if args.dataset is None:
        parser.error('--dataset required')
    manifest = json.loads((out/'manifest.json').read_text())
    verify(manifest)
    folder = out / args.dataset
    if args.child:
        (control if args.child == 'control' else replay)(args.dataset, folder)
        return
    folder.mkdir(exist_ok=False)
    status = folder / 'status.jsonl'
    append(status, dict(kind='self_tests', results=self_test()))
    for stage, limit in [('control', manifest['control_timeout_seconds']), ('ra', manifest['replay_timeout_seconds'])]:
        started = time.monotonic()
        append(status, dict(kind='stage_start', stage=stage, timeout_seconds=limit))
        argv = [sys.executable, '-u', str(Path(__file__).resolve()), '--output-root', str(out),
                '--dataset', args.dataset, '--child', stage]
        try:
            result = subprocess.run(argv, cwd=ROOT, timeout=limit)
        except subprocess.TimeoutExpired:
            append(status, dict(kind='timeout', stage=stage, seconds=time.monotonic()-started,
                                further_stages_skipped=True))
            raise SystemExit(124)
        if result.returncode:
            append(status, dict(kind='failed', stage=stage, exit_code=result.returncode))
            raise SystemExit(result.returncode)
        append(status, dict(kind='stage_complete', stage=stage, seconds=time.monotonic()-started))
    append(status, dict(kind='complete', dataset=args.dataset, publication_update=False))


if __name__ == '__main__':
    main()
