"""Diagnostic-only closeness inspection; original metric/model code is untouched.

--graphs-only inspects train/control without claiming these are model TP graphs.
--replay-ra retains the original heuristic order and intercepts RA recommendations
in RAM, checks their hash against Table 11, then inspects the actual RA TP graph.
Only scalar summaries, hashes and quantiles are saved; never predictions/scores.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'backbone'), str(ROOT)]
import networkit as nk
import numpy as np
import torch
from metrics.centrality_equality_gain import _build_networkit_graph, _eppstein_wang_sample_size
from metrics.equality_gain import compute_gini_coefficient
from metrics.graph_utils import build_control_edge_set, build_predicted_edge_set


def append(path, record):
    with path.open('a') as handle:
        handle.write(json.dumps(record, allow_nan=False) + '\n')
    print(json.dumps(record, allow_nan=False), flush=True)


def finite(value):
    return float(value) if np.isfinite(value) else None


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    good = values[np.isfinite(values)]
    return dict(n=len(values), nonfinite=int((~np.isfinite(values)).sum()),
                nan=int(np.isnan(values).sum()), positive_inf=int(np.isposinf(values).sum()),
                negative_inf=int(np.isneginf(values).sum()),
                zeros=int((values == 0).sum()), negative=int((values < 0).sum()),
                above_one=int((values > 1).sum()),
                quantiles=dict(zip(['min', 'p01', 'p25', 'p50', 'p75', 'p99', 'max'],
                                   map(float, np.quantile(good, [0, .01, .25, .5, .75, .99, 1])))) if len(good) else {})


def inspect(graphs, output, dataset, phase, reference_tp=None):
    n = next(iter(graphs.values())).numberOfNodes()
    samples = _eppstein_wang_sample_size(n, .1)
    groups = {}
    for name, graph in graphs.items():
        cc = nk.components.ConnectedComponents(graph).run()
        ids = np.asarray(cc.getPartition().getVector())
        sizes = cc.getComponentSizes()
        largest = max(sizes, key=sizes.get)
        groups[name] = ids == largest
        append(output, dict(kind='graph', dataset=dataset, phase=phase, graph=name,
                            nodes=n, edges=graph.numberOfEdges(), components=len(sizes),
                            largest_component_nodes=sizes[largest],
                            isolates=sum(size == 1 for size in sizes.values()),
                            connected_input_requirement_satisfied=len(sizes) == 1))
    settings = [(samples, seed, .1) for seed in [42, 43, 44]]
    settings += [(min(n, samples * factor), 42, .1) for factor in [2, 4]]
    settings += [(samples, 42, .05)]
    for count, seed, epsilon in settings:
        ginis = {}
        for name, graph in graphs.items():
            started = time.monotonic()
            nk.setSeed(seed, False)
            estimator = nk.centrality.ApproxCloseness(
                graph, nSamples=count, epsilon=epsilon, normalized=True,
                type=nk.centrality.ClosenessType.OUTBOUND).run()
            scores = np.asarray(estimator.scores(), dtype=np.float64)
            g32 = finite(compute_gini_coefficient(torch.tensor(scores, dtype=torch.float32)))
            g64 = finite(compute_gini_coefficient(torch.tensor(scores, dtype=torch.float64)))
            ginis[name] = (g32, g64)
            mask = groups[name]
            append(output, dict(kind='scores', dataset=dataset, phase=phase, graph=name,
                samples=count, seed=seed, library_epsilon=epsilon,
                seconds=time.monotonic() - started,
                gini_production_float32=g32, gini_float64_diagnostic=g64,
                all_nodes=distribution(scores), largest_component=distribution(scores[mask]),
                other_components=distribution(scores[~mask]),
                other_components_score_mass_fraction=finite(scores[~mask].sum() / scores.sum())))
        for name in graphs:
            if name == 'control':
                continue
            eg = []
            for index in [0, 1]:
                control, pred = ginis['control'][index], ginis[name][index]
                eg.append((control - pred) / control * 100 if control and pred is not None else None)
            record = dict(kind='eg', dataset=dataset, phase=phase, graph=name,
                          samples=count, seed=seed, library_epsilon=epsilon,
                          eg_production_float32=eg[0], eg_float64_diagnostic=eg[1])
            if name == 'RA_TP' and count == samples and seed == 42 and epsilon == .1:
                record['reference_table11_tp'] = reference_tp
                record['matches_reference_within_1e_8'] = bool(np.isclose(eg[0], reference_tp, rtol=0, atol=1e-8))
            append(output, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['brightkite', 'pokec_n'], required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--graphs-only', action='store_true')
    mode.add_argument('--replay-ra', action='store_true')
    args = parser.parse_args()
    nk.setNumberOfThreads(4)
    torch.set_num_threads(4)
    phase = 'graphs_only' if args.graphs_only else 'ra_replay'
    folder = args.output_root.resolve() / args.dataset / phase
    folder.mkdir(parents=True, exist_ok=False)
    output = folder / 'diagnostic.jsonl'
    paths = ['metrics/centrality_equality_gain.py', 'metrics/equality_gain.py',
             'metrics/acceptance_eg.py', 'metrics/graph_utils.py', 'analysis/diagnose_closeness_tp.py']
    append(output, dict(kind='provenance', dataset=args.dataset, phase=phase,
                       networkit_version=nk.__version__, torch_version=torch.__version__,
                       predictions_saved=False, node_scores_saved=False,
                       approximation_precondition='connected graph; unsupported on disconnected inputs',
                       sensitivity_is_not_a_metric_fix=True,
                       source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}))
    if args.graphs_only:
        from core import set_seed
        from core.data_loader import load_dataset
        set_seed(42)
        loader = SimpleNamespace(data_name=args.dataset, data_path=str(ROOT / 'data/SNAP'),
            encoder='HLGNN', device='cpu', val_ratio=.05, test_ratio=.1,
            split_file=None, data_split_seed=42)
        _, split, data, _ = load_dataset(loader)
        train, test = split['train']['edge'], split['test']['edge']
        inspect({'train': _build_networkit_graph(train, data.num_nodes),
                 'control': _build_networkit_graph(build_control_edge_set(train, test, data.num_nodes), data.num_nodes)},
                output, args.dataset, phase)
    else:
        # Keep original preceding methods, including RNG consumption for random fills.
        import core.acceptance_probe as probe
        import run_baselines
        from replay_table11_aligned import command
        from metrics.acceptance_eg import build_test_pair_set, filter_recs_tp
        authority = ROOT / 'results/validation/20260905_163304_table11_table4_aligned_v3'
        reference_path = authority / args.dataset / 'ra/acceptance.jsonl'
        reference = [json.loads(line) for line in reference_path.read_text().splitlines()]
        assert len(reference) == 1
        reference = reference[0]
        original = probe.record_acceptance
        observed = []

        def capture(path, **kw):
            original(path, **kw)
            actual = json.loads(Path(path).read_text().splitlines()[-1])
            for key in ['recommendations_sha256', 'ordered_train_test_sha256']:
                if actual[key] != reference[key]:
                    raise ValueError(f'RA reconstruction does not match Table 11: {key}')
            append(output, dict(kind='ra_identity', source=str(reference_path.relative_to(ROOT)),
                source_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                recommendations_sha256=actual['recommendations_sha256'],
                ordered_train_test_sha256=actual['ordered_train_test_sha256'],
                main_exact=actual['main'] == reference['main'], hashes_match=True))
            train, test, n = kw['train_edges'], kw['test_edges'], kw['num_nodes']
            recs_tp = filter_recs_tp(kw['recs'], build_test_pair_set(test), 10)
            inspect({'control': _build_networkit_graph(build_control_edge_set(train, test, n), n),
                     'RA_TP': _build_networkit_graph(build_predicted_edge_set(train, recs_tp, n, 10), n)},
                    output, args.dataset, phase, reference['acceptance']['Closeness_EG@10_tp'])
            observed.append(True)

        probe.record_acceptance = capture
        argv = command({}, args.dataset, 'RA', folder)
        sys.argv = [argv[2], *argv[3:]]
        os.chdir(ROOT / 'experiments')
        code = run_baselines.main()
        if code or len(observed) != 1:
            raise RuntimeError('RA metric diagnostic did not complete')
    append(output, dict(kind='complete', dataset=args.dataset, phase=phase))


if __name__ == '__main__':
    main()
