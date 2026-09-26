"""Opt-in exact generalized closeness v1; legacy entry points stay unchanged.

Activate once around a fresh experiment process, before creating caller caches.
Returns float32 scores and retains the existing positive-support Gini policy.
Only the closeness dispatcher is replaced; Degree/PageRank code is untouched.
"""
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import time

import networkit as nk
import numpy as np
import torch
from metrics import centrality_equality_gain as legacy
from metrics.graph_utils import canonicalize_undirected_edges

VERSION = 'exact_generalized_v1_float32_positive_support'


def validate_scores(scores, n):
    if scores.shape != (n,) or not torch.isfinite(scores).all():
        raise ValueError('Exact closeness: invalid shape or non-finite scores')
    if torch.any(scores < 0) or torch.any(scores > 1 + 1e-7):
        raise ValueError('Exact generalized closeness outside [0,1]')


@contextmanager
def activated(observer=None, cache_entries=4):
    """Use v1 for all public centrality/acceptance evaluations in this process.

The cache holds at most four score vectors, never predictions or graph objects.
Do not share external control caches across legacy/exact contexts. Campaign
children are fresh processes and never do that. Restores legacy on exit.
"""
    original = legacy.compute_closeness_centrality
    if getattr(original, 'metric_version', None):
        raise RuntimeError('Nested closeness version activation is not supported')
    cache = OrderedDict()
    failures = []

    def calculate(edges, num_nodes, use_sampling=True, sample_size=None, epsilon=.1, random_seed=42):
        canonical = canonicalize_undirected_edges(edges, num_nodes)
        digest = hashlib.sha256(canonical.numpy().astype('<i8', copy=False).tobytes()).hexdigest()
        key = (num_nodes, digest)
        cached = key in cache
        started = time.monotonic()
        if cached:
            scores = cache.pop(key)
        else:
            scores = legacy._compute_closeness_networkit(canonical, num_nodes, use_sampling=False)
        validate_scores(scores, num_nodes)
        if cache_entries:
            cache[key] = scores
            while len(cache) > cache_entries:
                cache.popitem(last=False)
        if observer:
            observer(dict(metric_version=VERSION, nodes=num_nodes, edges=len(canonical),
                graph_sha256=digest, cache_hit=cached, seconds=time.monotonic()-started,
                finite=True, minimum=float(scores.min()) if num_nodes else None,
                maximum=float(scores.max()) if num_nodes else None,
                zero_count=int((scores == 0).sum()), predictions_saved=False))
        return scores.clone()

    def exact(*args, **kwargs):
        try:
            return calculate(*args, **kwargs)
        except Exception as exc:
            # Some legacy aggregators catch metric errors and continue. Such a
            # run must still fail closed instead of publishing fallback zeros.
            failures.append(repr(exc))
            raise

    exact.metric_version = VERSION
    legacy.compute_closeness_centrality = exact
    try:
        yield
        if failures:
            raise RuntimeError('Exact closeness failed inside a legacy aggregator: ' + failures[0])
    finally:
        legacy.compute_closeness_centrality = original
        cache.clear()
