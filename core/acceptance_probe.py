"""Opt-in, low-disk acceptance instrumentation for the original Table 4 runners.

Never writes recommendations, embeddings, model weights or RNG states. One
JSONL record per evaluated model stores scalar metrics and provenance hashes.
Acceptance evaluation runs after the original metric battery and restores the
Python/NumPy/Torch training RNG streams even if evaluation fails. NetworKit
closeness evaluation sets its own seed in the existing centrality implementation.
"""

import contextlib
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch


@contextlib.contextmanager
def preserve_training_rng():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def scalar(value):
    if isinstance(value, (list, tuple)):
        value = value[1]
    return float(value)


def record_acceptance(path, *, dataset, method, run, config, recs,
                      train_edges, test_edges, num_nodes, main_metrics):
    """Record identical-input full/main checks and acceptance metrics at K=10.

The snapshot copies only the first ten integer targets per source in memory;
no prediction artifact is persisted. Duplicate run/config keys fail closed.
"""
    with preserve_training_rng():
        from metrics.acceptance_eg import compute_acceptance_robust_eg

        top10 = {int(u): [int(v) for v in vs[:10]] for u, vs in recs.items()}
        if not top10 or any(len(vs) != 10 for vs in top10.values()):
            raise ValueError('Acceptance pilot requires exactly ten recommendations per source')
        with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink):
            acceptance, rho = compute_acceptance_robust_eg(
                top10, train_edges.cpu(), test_edges.cpu(), num_nodes,
                k=10, use_sampling_closeness=True)
        names = [('Degree_EG@10_full', 'Equality_Gain@10'),
                 ('Pagerank_EG@10_full', 'Pagerank_EG@10'),
                 ('Closeness_EG@10_full', 'Closeness_EG@10')]
        differences = {a: scalar(acceptance[a]) - scalar(main_metrics[b]) for a, b in names}
        checks = {a: math.isfinite(scalar(acceptance[a])) and
                  math.isclose(scalar(acceptance[a]), scalar(main_metrics[b]),
                               rel_tol=1e-8, abs_tol=1e-6) for a, b in names}
        digest = hashlib.sha256()
        for u, vs in sorted(top10.items()):
            digest.update(np.asarray([u, *vs], dtype='<i8').tobytes())
        graph_digest = hashlib.sha256(f'num_nodes={num_nodes}\n'.encode())
        # Ordered-edge fingerprint deliberately retains legacy Degree orientation.
        for name, edges in [('train', train_edges), ('test', test_edges)]:
            graph_digest.update(name.encode())
            graph_digest.update(edges.detach().cpu().numpy().astype('<i8').tobytes())
        root = Path(__file__).resolve().parents[1]
        record = dict(dataset=dataset, method=method, run=int(run), config=config,
                      protocol='table4_fixed_split_replay_pilot', k=10,
                      n_sources=len(top10), predictions_saved=False,
                      recommendations_sha256=digest.hexdigest(),
                      ordered_train_test_sha256=graph_digest.hexdigest(),
                      main={k: scalar(main_metrics[k]) for k in
                            ['Precision@10', 'Equality_Gain@10', 'Pagerank_EG@10', 'Closeness_EG@10']},
                      acceptance={k: scalar(v) for k, v in acceptance.items()},
                      rho_rank=[float(v) for v in rho], full_minus_main=differences,
                      full_matches_main=all(checks.values()),
                      source_sha256={str(p): hashlib.sha256((root / p).read_bytes()).hexdigest()
                                     for p in ['core/acceptance_probe.py', 'metrics/acceptance_eg.py',
                                               'metrics/equality_gain.py', 'metrics/centrality_equality_gain.py']})
        target = Path(path)
        if target.exists():
            for line in target.read_text().splitlines():
                previous = json.loads(line)
                if (previous['dataset'], previous['method'], previous['run'], previous['config']) == (
                        dataset, method, int(run), config):
                    raise ValueError('Duplicate acceptance record: use an isolated output path')
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('a') as handle:
            handle.write(json.dumps(record, allow_nan=False, separators=(',', ':')) + '\n')
        if not all(checks.values()):
            raise ValueError(f'Acceptance Full differs from main metrics: {differences}')
    print(f'[acceptance pilot] {method} run={run}: Full/main PASS; scalar record saved')
