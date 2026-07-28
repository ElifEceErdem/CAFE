"""Tests for run-level uncertainty in source-stratified precision."""

import unittest

import pandas as pd

from analysis.run_precision_by_coreness import (
    _ordered_models,
    aggregate_precision_within_runs,
)


class StratifiedAggregationTests(unittest.TestCase):
    def test_nodes_are_aggregated_before_runs(self):
        frame = pd.DataFrame(
            [
                {
                    'run': 42,
                    'model': 'CAFE-In',
                    'coreness_bin': 0,
                    'coreness_bin_label': '[1-2]',
                    'node_id': 0,
                    'precision': 0.0,
                },
                {
                    'run': 42,
                    'model': 'CAFE-In',
                    'coreness_bin': 0,
                    'coreness_bin_label': '[1-2]',
                    'node_id': 1,
                    'precision': 0.2,
                },
                {
                    'run': 43,
                    'model': 'CAFE-In',
                    'coreness_bin': 0,
                    'coreness_bin_label': '[1-2]',
                    'node_id': 0,
                    'precision': 0.8,
                },
                {
                    'run': 43,
                    'model': 'CAFE-In',
                    'coreness_bin': 0,
                    'coreness_bin_label': '[1-2]',
                    'node_id': 1,
                    'precision': 1.0,
                },
            ]
        )

        aggregated = aggregate_precision_within_runs(frame)
        self.assertEqual(len(aggregated), 2)
        self.assertEqual(aggregated['run'].tolist(), [42, 43])
        self.assertEqual(aggregated['n_nodes'].tolist(), [2, 2])
        self.assertAlmostEqual(aggregated['precision'].iloc[0], 0.1)
        self.assertAlmostEqual(aggregated['precision'].iloc[1], 0.9)

    def test_requested_publication_order_is_enforced(self):
        actual = _ordered_models(
            ['CAFE-Pre', 'RA', 'Fusion', 'HL-GNN', 'CAFE-In'],
            ['RA', 'HL-GNN', 'Fusion', 'CAFE-In', 'CAFE-Pre'],
        )
        self.assertEqual(
            actual,
            ['RA', 'HL-GNN', 'Fusion', 'CAFE-In', 'CAFE-Pre'],
        )


if __name__ == '__main__':
    unittest.main()
