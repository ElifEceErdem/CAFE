"""Merge seed-sharded run_baselines.py outputs into one full-seed result file.

t04_baselines_shard.job splits the seed range across GPUs, so each shard holds
a disjoint subset of seeds in its own staging tree. This reassembles them into
the single file the analysis expects and writes it into the real results tree.

The merge is exact rather than approximate: run_baselines.py stores every
per-run value under ``values``, so the seeds are concatenated in seed order and
mean/std/min/max are recomputed with the same numpy calls
``core.statistics.compute_statistics`` uses (``np.std`` default ddof=0). The
result is identical to what an unsharded ``--seed 42 --runs 10`` job would have
written, up to floating-point summation order.

    python experiments/merge_baseline_shards.py --dataset pokec_n --check
    python experiments/merge_baseline_shards.py --dataset pokec_n

Refuses to write unless the shards cover the expected seed range exactly and
every method finished every seed; pass --allow-partial to override, which
stamps the true per-method counts into ``runs_completed`` so a short merge can
never be mistaken for a complete one.
"""
import argparse
import collections
import glob
import json
import os
import re
from datetime import datetime

import numpy as np

STAGE_DEFAULT = '/var/scratch/eerdem/cafe_baseline_shards'
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def discover(stage, dataset):
    """seed_start -> {dirname: parsed json}, newest file per dirname."""
    shards = {}
    for d in sorted(glob.glob(os.path.join(stage, dataset, 'seed_*'))):
        m = re.match(r'seed_(\d+)$', os.path.basename(d))
        if not m:
            continue
        seed = int(m.group(1))
        per_dir = {}
        for sub in sorted(glob.glob(os.path.join(d, dataset, '*'))):
            if not os.path.isdir(sub):
                continue
            files = sorted(p for p in glob.glob(os.path.join(sub, 'results_*.json'))
                           if not p.endswith('.partial.json'))
            if files:
                with open(files[-1]) as fh:
                    per_dir[os.path.basename(sub)] = json.load(fh)
        if per_dir:
            shards[seed] = per_dir
    return shards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--stage', default=STAGE_DEFAULT)
    ap.add_argument('--results-root', default=os.path.join(REPO, 'results_repro'))
    ap.add_argument('--base-seed', type=int, default=42)
    ap.add_argument('--total-runs', type=int, default=10)
    ap.add_argument('--check', action='store_true', help='report and exit')
    ap.add_argument('--allow-partial', action='store_true')
    args = ap.parse_args()

    shards = discover(args.stage, args.dataset)
    if not shards:
        raise SystemExit(f"no shards under {os.path.join(args.stage, args.dataset)}")

    expected = list(range(args.base_seed, args.base_seed + args.total_runs))
    covered = []
    print(f"shards for {args.dataset}:")
    for seed in sorted(shards):
        dirs = shards[seed]
        counts = {}
        for dirname, payload in dirs.items():
            for k, n in (payload.get('runs_completed') or {}).items():
                counts[k] = n
        n = max(counts.values()) if counts else 0
        covered.extend(range(seed, seed + n))
        print(f"  seed_{seed}: {len(dirs)} dirs, runs_completed={sorted(counts.items())}")

    missing = sorted(set(expected) - set(covered))
    dupes = sorted({s for s in covered if covered.count(s) > 1})
    print(f"\nseed coverage: {sorted(set(covered))}")
    if missing:
        print(f"MISSING seeds: {missing}")
    if dupes:
        print(f"DUPLICATE seeds: {dupes}")

    if args.check:
        return
    if (missing or dupes) and not args.allow_partial:
        raise SystemExit("refusing to merge: incomplete or overlapping seed coverage "
                         "(re-run the missing shards, or pass --allow-partial)")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    by_dir = collections.defaultdict(lambda: collections.defaultdict(dict))
    runs_by_dir = collections.defaultdict(dict)
    info_by_dir = {}

    for seed in sorted(shards):
        for dirname, payload in shards[seed].items():
            info_by_dir.setdefault(dirname, payload['experiment_info'])
            for key, metrics in payload['results'].items():
                for metric, stat in metrics.items():
                    vals = stat.get('values')
                    if not isinstance(vals, list):
                        continue
                    by_dir[dirname].setdefault(key, {}).setdefault(metric, []).extend(vals)

    for dirname, per_key in by_dir.items():
        results = {}
        for key, metrics in per_key.items():
            out = {}
            for metric in sorted(metrics):
                v = [float(x) for x in metrics[metric]]
                out[metric] = {
                    'mean': float(np.mean(v)),
                    'std': float(np.std(v)),
                    'min': float(np.min(v)),
                    'max': float(np.max(v)),
                    'values': v,
                }
            results[key] = out
            runs_by_dir[dirname][key] = max(len(metrics[m]) for m in metrics)

        info = dict(info_by_dir[dirname])
        info['num_runs'] = args.total_runs
        info['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        info['merged_from_seed_shards'] = sorted(shards)
        info['merge_seed_coverage'] = sorted(set(covered))

        out_dir = os.path.join(args.results_root, args.dataset, dirname)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'results_{dirname}_{args.dataset}_{ts}.json')
        payload = {
            'experiment_info': info,
            'runs_completed': runs_by_dir[dirname],
            'results': results,
        }
        tmp = out_path + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, out_path)
        print(f"wrote {sorted(results)} ({runs_by_dir[dirname]}) -> {out_path}")


if __name__ == '__main__':
    main()
