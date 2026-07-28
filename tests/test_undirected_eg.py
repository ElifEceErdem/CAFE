"""Regression tests for the simple-undirected Equality Gain contract."""

import unittest
from unittest import mock

import networkx as nx
import numpy as np
import torch

from metrics.acceptance_eg import (
    _build_expected_predicted_graph,
    compute_acceptance_robust_eg,
    weighted_predicted_degree,
)
from metrics.centrality_equality_gain import (
    build_control_centrality,
    build_predicted_centrality,
    compute_degree_centrality,
    compute_all_centrality_equality_gains,
    compute_centrality_equality_gain,
    compute_centrality_equality_gain_by_source_bins,
    compute_degree_equality_gain_by_source_bins_for_models,
    _compute_closeness_networkit,
)
from metrics.fairness import distribution_fairness_from_recs
from metrics.equality_gain import (
    build_control_degree_distribution,
    build_predicted_degree_distribution,
    compute_equality_gain,
    compute_equality_gain_metrics,
)
from metrics.graph_utils import (
    build_control_edge_set,
    build_predicted_edge_set,
    canonicalize_undirected_edges,
    degree_from_undirected_edges,
)


class UndirectedEqualityGainTests(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 5
        self.train = torch.tensor(
            [[0, 1], [1, 2], [2, 3]], dtype=torch.long
        )
        self.test = torch.tensor(
            [[0, 2], [3, 4]], dtype=torch.long
        )
        self.exact_recs = {0: [2], 2: [0], 3: [4], 4: [3]}
        self.expected_edges = torch.tensor(
            [[0, 1], [0, 2], [1, 2], [2, 3], [3, 4]],
            dtype=torch.long,
        )

    def assertTensorClose(self, actual, expected, atol=1e-7):
        self.assertTrue(
            torch.allclose(actual, expected, atol=atol, rtol=0),
            msg=f"\nactual={actual}\nexpected={expected}",
        )

    def test_canonical_edge_set_contract(self):
        noisy = torch.tensor(
            [
                [1, 0],
                [0, 1],
                [0, 2],
                [2, 0],
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 3],
                [2, 2],
            ],
            dtype=torch.long,
        )
        canonical = canonicalize_undirected_edges(noisy, self.num_nodes)
        self.assertTrue(torch.equal(canonical, self.expected_edges))
        self.assertTrue(torch.all(canonical[:, 0] < canonical[:, 1]))
        self.assertEqual(len(canonical), len(torch.unique(canonical, dim=0)))

        control = build_control_edge_set(
            torch.cat([self.train, self.train.flip(1)]),
            torch.cat([self.test, self.test.flip(1)]),
            self.num_nodes,
        )
        predicted = build_predicted_edge_set(
            self.train, self.exact_recs, self.num_nodes, 1
        )
        self.assertTrue(torch.equal(control, self.expected_edges))
        self.assertTrue(torch.equal(predicted, self.expected_edges))

        with self.assertRaises(ValueError):
            canonicalize_undirected_edges(
                torch.tensor([[0, self.num_nodes]]), self.num_nodes
            )

    def test_degree_is_simple_undirected(self):
        degree = degree_from_undirected_edges(
            self.expected_edges, self.num_nodes
        )
        expected = torch.tensor([2, 2, 3, 2, 1], dtype=torch.float32)
        self.assertTrue(torch.equal(degree, expected))
        self.assertEqual(float(degree.sum()), 2.0 * len(self.expected_edges))

        graph = nx.Graph()
        graph.add_nodes_from(range(self.num_nodes))
        graph.add_edges_from(self.expected_edges.tolist())
        oracle = torch.tensor(
            [graph.degree(node) for node in range(self.num_nodes)],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(degree, oracle))

    def test_exact_reproduction_has_zero_eg_all_centralities(self):
        for centrality in ("degree", "pagerank", "closeness"):
            kwargs = {"use_sampling": False} if centrality == "closeness" else {}
            control = build_control_centrality(
                self.train,
                self.test,
                self.num_nodes,
                centrality,
                **kwargs,
            )
            predicted = build_predicted_centrality(
                self.train,
                self.exact_recs,
                self.num_nodes,
                1,
                centrality,
                **kwargs,
            )
            tolerance = 2e-6 if centrality == "pagerank" else 1e-7
            self.assertTensorClose(control, predicted, atol=tolerance)
            eg, gini_control, gini_pred = compute_centrality_equality_gain(
                self.exact_recs,
                self.train,
                self.test,
                self.num_nodes,
                1,
                centrality,
                **kwargs,
            )
            self.assertAlmostEqual(gini_control, gini_pred, places=7)
            self.assertAlmostEqual(eg, 0.0, places=7)

        degree_eg, _, _ = compute_equality_gain(
            self.exact_recs,
            self.train,
            self.test,
            self.num_nodes,
            1,
        )
        self.assertAlmostEqual(degree_eg, 0.0, places=7)

    def test_approximate_closeness_resets_networkit_seed(self):
        fake_algorithm = mock.Mock()
        fake_algorithm.scores.return_value = [0.0] * 5001
        with mock.patch(
            'metrics.centrality_equality_gain._build_networkit_graph',
            return_value=object(),
        ), mock.patch(
            'metrics.centrality_equality_gain.nk.setSeed'
        ) as set_seed, mock.patch(
            'metrics.centrality_equality_gain.nk.centrality.ApproxCloseness',
            return_value=fake_algorithm,
        ) as approx:
            result = _compute_closeness_networkit(
                torch.empty((0, 2), dtype=torch.long),
                5001,
                use_sampling=True,
                sample_size=7,
                epsilon=0.1,
                random_seed=17,
            )

        set_seed.assert_called_once_with(17, False)
        approx.assert_called_once()
        fake_algorithm.run.assert_called_once_with()
        self.assertEqual(tuple(result.shape), (5001,))

    def test_reciprocal_recommendations_are_idempotent(self):
        base = {0: [2, 3], 2: [4]}
        reciprocal = {0: [2, 3], 2: [0, 4], 3: [0], 4: [2]}

        for centrality in ("degree", "pagerank", "closeness"):
            kwargs = {"use_sampling": False} if centrality == "closeness" else {}
            base_values = build_predicted_centrality(
                self.train, base, self.num_nodes, 3, centrality, **kwargs
            )
            reciprocal_values = build_predicted_centrality(
                self.train,
                reciprocal,
                self.num_nodes,
                3,
                centrality,
                **kwargs,
            )
            tolerance = 2e-6 if centrality == "pagerank" else 1e-7
            self.assertTensorClose(
                base_values, reciprocal_values, atol=tolerance
            )

            base_eg = compute_centrality_equality_gain(
                base,
                self.train,
                self.test,
                self.num_nodes,
                3,
                centrality,
                **kwargs,
            )[0]
            reciprocal_eg = compute_centrality_equality_gain(
                reciprocal,
                self.train,
                self.test,
                self.num_nodes,
                3,
                centrality,
                **kwargs,
            )[0]
            self.assertAlmostEqual(base_eg, reciprocal_eg, places=6)

    def test_degree_public_entrypoints_agree(self):
        recs = {0: [3], 4: [1]}
        direct = compute_equality_gain(
            recs, self.train, self.test, self.num_nodes, 1
        )
        centrality = compute_centrality_equality_gain(
            recs, self.train, self.test, self.num_nodes, 1, "degree"
        )
        for actual, expected in zip(direct, centrality):
            self.assertAlmostEqual(actual, expected, places=7)

        legacy_batch = compute_equality_gain_metrics(
            recs, self.train, self.test, self.num_nodes, [1]
        )["Equality_Gain@1"][1]
        centrality_batch = compute_all_centrality_equality_gains(
            recs,
            self.train,
            self.test,
            self.num_nodes,
            [1],
            ["degree"],
        )["Degree_EG@1"][1]
        self.assertAlmostEqual(legacy_batch, centrality_batch, places=7)

        legacy_control = build_control_degree_distribution(
            self.train, self.test, self.num_nodes
        )
        legacy_predicted = build_predicted_degree_distribution(
            self.train, recs, self.num_nodes, 1
        )
        central_control = build_control_centrality(
            self.train, self.test, self.num_nodes, "degree"
        )
        central_predicted = build_predicted_centrality(
            self.train, recs, self.num_nodes, 1, "degree"
        )
        self.assertTrue(torch.equal(legacy_control, central_control))
        self.assertTrue(torch.equal(legacy_predicted, central_predicted))

    def test_directed_modes_are_rejected_at_eg_boundary(self):
        recs = {0: [3], 4: [1]}
        for mode in ("indegree", "outdegree", "total"):
            with self.subTest(api="aggregate", mode=mode):
                with self.assertRaises(ValueError):
                    compute_equality_gain(
                        recs,
                        self.train,
                        self.test,
                        self.num_nodes,
                        1,
                        mode,
                    )
            with self.subTest(api="aggregate_batch", mode=mode):
                with self.assertRaises(ValueError):
                    compute_equality_gain_metrics(
                        recs,
                        self.train,
                        self.test,
                        self.num_nodes,
                        [1],
                        mode,
                    )
            with self.subTest(api="centrality", mode=mode):
                with self.assertRaises(ValueError):
                    compute_centrality_equality_gain(
                        recs,
                        self.train,
                        self.test,
                        self.num_nodes,
                        1,
                        mode,
                    )
            with self.subTest(api="centrality_batch", mode=mode):
                with self.assertRaises(ValueError):
                    compute_all_centrality_equality_gains(
                        recs,
                        self.train,
                        self.test,
                        self.num_nodes,
                        [1],
                        [mode],
                    )

    def test_directed_exposure_modes_remain_separate(self):
        directed = torch.tensor(
            [[0, 1], [0, 2], [2, 1]], dtype=torch.long
        )
        self.assertTrue(
            torch.equal(
                compute_degree_centrality(directed, 3, "indegree"),
                torch.tensor([0.0, 2.0, 1.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                compute_degree_centrality(directed, 3, "outdegree"),
                torch.tensor([2.0, 0.0, 1.0]),
            )
        )

        recs = {0: [2, 3], 1: [2], 2: [3]}
        test_edges = torch.tensor(
            [[0, 1], [1, 2], [2, 3]], dtype=torch.long
        )
        indegree_fairness = distribution_fairness_from_recs(
            recs, test_edges, 2, 4, "indegree"
        )
        outdegree_fairness = distribution_fairness_from_recs(
            recs, test_edges, 2, 4, "outdegree"
        )
        self.assertNotAlmostEqual(
            indegree_fairness, outdegree_fairness, places=7
        )

    def test_node_relabeling_invariance(self):
        recs = {0: [3], 4: [1]}
        permutation = {0: 4, 1: 1, 2: 3, 3: 0, 4: 2}

        def permute_edges(edges):
            return torch.tensor(
                [[permutation[int(u)], permutation[int(v)]] for u, v in edges],
                dtype=torch.long,
            )

        permuted_recs = {
            permutation[u]: [permutation[v] for v in values]
            for u, values in recs.items()
        }
        permuted_train = permute_edges(self.train)
        permuted_test = permute_edges(self.test)

        for centrality in ("degree", "pagerank", "closeness"):
            kwargs = {"use_sampling": False} if centrality == "closeness" else {}
            original_eg = compute_centrality_equality_gain(
                recs,
                self.train,
                self.test,
                self.num_nodes,
                1,
                centrality,
                **kwargs,
            )[0]
            permuted_eg = compute_centrality_equality_gain(
                permuted_recs,
                permuted_train,
                permuted_test,
                self.num_nodes,
                1,
                centrality,
                **kwargs,
            )[0]
            self.assertAlmostEqual(original_eg, permuted_eg, places=5)

            original = build_predicted_centrality(
                self.train, recs, self.num_nodes, 1, centrality, **kwargs
            )
            permuted = build_predicted_centrality(
                permuted_train,
                permuted_recs,
                self.num_nodes,
                1,
                centrality,
                **kwargs,
            )
            restored = torch.tensor(
                [permuted[permutation[node]] for node in range(self.num_nodes)]
            )
            tolerance = 2e-6 if centrality == "pagerank" else 1e-7
            self.assertTensorClose(original, restored, atol=tolerance)

    def test_source_bin_counterfactuals_match_direct_eg(self):
        recs = {0: [3], 2: [4], 4: [1]}

        for centrality in ("degree", "pagerank", "closeness"):
            kwargs = {"use_sampling": False} if centrality == "closeness" else {}
            all_one_bin = {u: 0 for u in recs}
            one_bin = compute_centrality_equality_gain_by_source_bins(
                recs,
                all_one_bin,
                self.train,
                self.test,
                self.num_nodes,
                1,
                centrality,
                **kwargs,
            )[0][0]
            direct = compute_centrality_equality_gain(
                recs,
                self.train,
                self.test,
                self.num_nodes,
                1,
                centrality,
                **kwargs,
            )[0]
            self.assertAlmostEqual(one_bin, direct, places=6)

            split_bins = {0: "low", 2: "low", 4: "high"}
            split = compute_centrality_equality_gain_by_source_bins(
                recs,
                split_bins,
                self.train,
                self.test,
                self.num_nodes,
                1,
                centrality,
                **kwargs,
            )
            for bin_id in ("low", "high"):
                filtered = {
                    u: values
                    for u, values in recs.items()
                    if split_bins[u] == bin_id
                }
                expected = compute_centrality_equality_gain(
                    filtered,
                    self.train,
                    self.test,
                    self.num_nodes,
                    1,
                    centrality,
                    **kwargs,
                )[0]
                self.assertAlmostEqual(split[bin_id][0], expected, places=6)

        cached_degree = compute_degree_equality_gain_by_source_bins_for_models(
            {"method_a": recs, "method_b": dict(reversed(list(recs.items())))},
            split_bins,
            self.train,
            self.test,
            self.num_nodes,
            1,
            bin_ids=["low", "high"],
        )
        reference_degree = compute_centrality_equality_gain_by_source_bins(
            recs,
            split_bins,
            self.train,
            self.test,
            self.num_nodes,
            1,
            "degree",
            bin_ids=["low", "high"],
        )
        for method_results in cached_degree.values():
            for bin_id in ("low", "high"):
                for actual, expected in zip(
                    method_results[bin_id], reference_degree[bin_id]
                ):
                    self.assertAlmostEqual(actual, expected, places=7)

    def test_pw_reciprocal_probabilities_use_probability_union(self):
        train = torch.tensor([[0, 1]], dtype=torch.long)
        recs = {0: [1, 2], 2: [0, 3]}
        rho = np.array([0.5, 0.2])

        edges, weights = _build_expected_predicted_graph(
            train, recs, 4, 2, rho
        )
        expected_edges = torch.tensor(
            [[0, 1], [0, 2], [2, 3]], dtype=torch.long
        )
        expected_weights = torch.tensor(
            [1.0, 0.6, 0.2], dtype=torch.float64
        )
        self.assertTrue(torch.equal(edges, expected_edges))
        self.assertTensorClose(weights, expected_weights)
        self.assertLessEqual(float(weights.max()), 1.0)

        degree = weighted_predicted_degree(train, recs, 4, 2, rho)
        self.assertTensorClose(
            degree, torch.tensor([1.6, 1.0, 0.8, 0.2], dtype=torch.float32)
        )
        self.assertAlmostEqual(
            float(degree.sum()), 2.0 * float(weights.sum()), places=6
        )

        reversed_recs = {2: [3, 0], 0: [2, 1]}
        reversed_edges, reversed_weights = _build_expected_predicted_graph(
            train.flip(1), reversed_recs, 4, 2, rho[::-1].copy()
        )
        self.assertTrue(torch.equal(reversed_edges, expected_edges))
        self.assertTensorClose(reversed_weights, expected_weights)

        degree_zero = weighted_predicted_degree(
            train, recs, 4, 2, np.zeros(2)
        )
        self.assertTensorClose(
            degree_zero, torch.tensor([1.0, 1.0, 0.0, 0.0])
        )
        degree_one = weighted_predicted_degree(
            train, recs, 4, 2, np.ones(2)
        )
        full_degree = build_predicted_degree_distribution(
            train, recs, 4, 2
        )
        self.assertTensorClose(degree_one, full_degree)

    def test_acceptance_full_matches_main_pipeline(self):
        results, _ = compute_acceptance_robust_eg(
            self.exact_recs,
            self.train,
            self.test,
            self.num_nodes,
            k=1,
            use_sampling_closeness=False,
        )
        degree = compute_equality_gain(
            self.exact_recs,
            self.train,
            self.test,
            self.num_nodes,
            1,
        )[0]
        pagerank = compute_centrality_equality_gain(
            self.exact_recs,
            self.train,
            self.test,
            self.num_nodes,
            1,
            "pagerank",
        )[0]
        closeness = compute_centrality_equality_gain(
            self.exact_recs,
            self.train,
            self.test,
            self.num_nodes,
            1,
            "closeness",
            use_sampling=False,
            epsilon=0.1,
        )[0]
        self.assertAlmostEqual(results["Degree_EG@1_full"], degree, places=7)
        self.assertAlmostEqual(
            results["Pagerank_EG@1_full"], pagerank, places=7
        )
        self.assertAlmostEqual(
            results["Closeness_EG@1_full"], closeness, places=7
        )


if __name__ == "__main__":
    unittest.main()
