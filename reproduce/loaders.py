"""Readers for the result formats written by the experiment runners.

Every summary in the paper is recomputed here from per-run values, never
from a cached mean.  Two JSON layouts exist:

* ``results[<method>][<metric>]["values"]`` -- ``experiments/run_baselines.py``;
* ``lambda_results[<config>]["statistics"][<metric>]["values"]`` -- the CAFE,
  ablation and fusion runners (``run_cafe.py``, ``run_cafe_p.py``,
  ``run_coreness_inverse.py``, ``run_mobility_boost.py``).

A source may also be a run directory from the September validation campaigns,
in which case its ``main_results.json`` is read.
"""
import json
import math
import statistics

from reproduce.paths import resolve

METRICS = {"prec": "Precision@10", "deg_eg": "Equality_Gain@10",
           "pr_eg": "Pagerank_EG@10", "cl_eg": "Closeness_EG@10"}


def _json(rel):
    with open(resolve(rel)) as fh:
        return json.load(fh)


def run_file(rel):
    path = resolve(rel)
    return f"{rel}/main_results.json" if path.is_dir() else rel


def run_values(rel, *, config=None, key=None, metric):
    """Per-run values of ``metric`` from one result file or run directory."""
    data = _json(run_file(rel))
    if config:
        block = data["lambda_results"][config]
        stats = block.get("statistics", block)
    else:
        stats = data["results"][key]
    values = stats[metric]["values"]
    assert values and all(math.isfinite(v) for v in values), (rel, metric)
    return list(values)


def summarize(values):
    """Mean and population SD, the convention of every fixed-split table."""
    return {"mean": statistics.fmean(values), "sd": statistics.pstdev(values), "n": len(values)}


def pooled(sources, *, config=None, key=None, metric, n=None):
    """Concatenate per-run values across shards, keeping the first ``n``.

    Deterministic heuristics are stored once per shard; ``n=1`` keeps a single
    evaluation instead of repeated copies.
    """
    values = []
    for rel in sources:
        values.extend(run_values(rel, config=config, key=key, metric=metric))
    if n is not None:
        if len(values) < n:
            raise ValueError(f"{metric}: {len(values)} runs found, {n} required ({sources})")
        values = values[:n]
    return values


def jsonl(rel):
    with open(resolve(rel)) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_json(rel):
    return _json(rel)
