"""Real-dataset smoke check for corrected Equality Gain graph semantics.

This CPU-only check generates deterministic Resource Allocation
recommendations on one fixed split and verifies that aggregate and
source-stratified entry points share the same simple-undirected graph
semantics.  It is intentionally cheaper than a CAFE training rerun.
"""

import argparse
import json
import os
import sys

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
experiments_dir = os.path.join(parent_dir, "experiments")
for path in (parent_dir, experiments_dir, os.path.join(parent_dir, "backbone")):
    if path not in sys.path:
        sys.path.insert(0, path)

from core import load_dataset, set_seed  # noqa: E402
from cafe_methods import (  # noqa: E402
    add_common_args,
    build_heuristic_recs,
    build_ra_sparse,
    build_train_adj_csr,
    get_eval_nodes,
)
from metrics.centrality_equality_gain import (  # noqa: E402
    compute_centrality_equality_gain,
    compute_centrality_equality_gain_by_source_bins,
    compute_degree_equality_gain_by_source_bins_for_models,
)
from metrics.equality_gain import compute_equality_gain  # noqa: E402
from metrics.graph_utils import (  # noqa: E402
    build_control_edge_set,
    build_predicted_edge_set,
    canonicalize_undirected_edges,
    recommendations_to_edges,
)


def argument():
    parser = argparse.ArgumentParser(
        description="Smoke-check simple-undirected EG semantics on real data"
    )
    parser = add_common_args(parser)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


def main():
    args = argument()
    set_seed(args.seed)
    dataset, split_edge, _, _ = load_dataset(args)
    num_nodes = (
        dataset._data.num_nodes
        if hasattr(dataset, "_data")
        else dataset.data.num_nodes
    )
    train_edges = split_edge["train"]["edge"].cpu()
    test_edges = split_edge["test"]["edge"].cpu()

    train_adj = build_train_adj_csr(train_edges, num_nodes)
    ra_sparse = build_ra_sparse(train_adj)
    eval_nodes = get_eval_nodes(test_edges)
    recs = build_heuristic_recs(
        ra_sparse, train_adj, eval_nodes, args.K, verbose=False
    )

    degree_main = compute_equality_gain(
        recs, train_edges, test_edges, num_nodes, args.K
    )
    degree_centrality = compute_centrality_equality_gain(
        recs, train_edges, test_edges, num_nodes, args.K, "degree"
    )
    pagerank = compute_centrality_equality_gain(
        recs, train_edges, test_edges, num_nodes, args.K, "pagerank"
    )

    for left, right in zip(degree_main, degree_centrality):
        if abs(left - right) > 1e-7:
            raise AssertionError(
                f"Degree entry points disagree: {degree_main} vs "
                f"{degree_centrality}"
            )

    all_sources = {int(u): "all" for u in recs}
    stratified_degree = compute_centrality_equality_gain_by_source_bins(
        recs,
        all_sources,
        train_edges,
        test_edges,
        num_nodes,
        args.K,
        "degree",
    )["all"]
    cached_stratified_degree = (
        compute_degree_equality_gain_by_source_bins_for_models(
            {"RA": recs},
            all_sources,
            train_edges,
            test_edges,
            num_nodes,
            args.K,
        )["RA"]["all"]
    )
    if abs(stratified_degree[0] - degree_main[0]) > 1e-7:
        raise AssertionError(
            f"All-source stratum disagrees with aggregate EG: "
            f"{stratified_degree[0]} vs {degree_main[0]}"
        )
    if abs(cached_stratified_degree[0] - degree_main[0]) > 1e-7:
        raise AssertionError(
            "Cached source-stratified Degree EG disagrees with aggregate "
            f"EG: {cached_stratified_degree[0]} vs {degree_main[0]}"
        )

    audit_bins = {int(u): int(u) % 3 for u in recs}
    reference_bins = compute_centrality_equality_gain_by_source_bins(
        recs,
        audit_bins,
        train_edges,
        test_edges,
        num_nodes,
        args.K,
        "degree",
        bin_ids=[0, 1, 2],
    )
    cached_bins = compute_degree_equality_gain_by_source_bins_for_models(
        {"RA": recs},
        audit_bins,
        train_edges,
        test_edges,
        num_nodes,
        args.K,
        bin_ids=[0, 1, 2],
    )["RA"]
    cached_multibin_max_abs_diff = max(
        abs(cached_bins[bin_id][field] - reference_bins[bin_id][field])
        for bin_id in (0, 1, 2)
        for field in range(3)
    )
    if cached_multibin_max_abs_diff > 1e-7:
        raise AssertionError(
            "Cached multi-bin Degree EG disagrees with the reference path: "
            f"max abs diff={cached_multibin_max_abs_diff}"
        )

    reversed_degree = compute_equality_gain(
        recs,
        train_edges.flip(1),
        test_edges.flip(1),
        num_nodes,
        args.K,
    )
    for left, right in zip(degree_main, reversed_degree):
        if abs(left - right) > 1e-7:
            raise AssertionError(
                f"Degree EG depends on stored edge orientation: "
                f"{degree_main} vs {reversed_degree}"
            )

    observed_edges = build_control_edge_set(
        train_edges, test_edges, num_nodes
    )
    predicted_edges = build_predicted_edge_set(
        train_edges, recs, num_nodes, args.K
    )
    rec_events = recommendations_to_edges(recs, num_nodes, args.K)
    rec_unique = canonicalize_undirected_edges(rec_events, num_nodes)

    summary = {
        "dataset": args.data_name,
        "seed": args.seed,
        "K": args.K,
        "num_nodes": int(num_nodes),
        "num_eval_users": len(eval_nodes),
        "num_train_edges_simple": len(
            canonicalize_undirected_edges(train_edges, num_nodes)
        ),
        "num_observed_edges_simple": len(observed_edges),
        "num_predicted_edges_simple": len(predicted_edges),
        "num_recommendation_events": len(rec_events),
        "num_recommendation_edges_simple": len(rec_unique),
        "degree_eg": degree_main[0],
        "degree_gini_control": degree_main[1],
        "degree_gini_predicted": degree_main[2],
        "pagerank_eg": pagerank[0],
        "pagerank_gini_control": pagerank[1],
        "pagerank_gini_predicted": pagerank[2],
        "aggregate_stratum_degree_abs_diff": abs(
            stratified_degree[0] - degree_main[0]
        ),
        "cached_stratum_degree_abs_diff": abs(
            cached_stratified_degree[0] - degree_main[0]
        ),
        "cached_multibin_max_abs_diff": cached_multibin_max_abs_diff,
        "orientation_degree_abs_diff": abs(
            reversed_degree[0] - degree_main[0]
        ),
        "graph_semantics": "simple_undirected_set_union",
    }

    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
        print(f"Saved smoke-check summary to {args.out}")


if __name__ == "__main__":
    main()
