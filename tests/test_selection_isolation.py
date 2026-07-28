"""Regression tests for validation-only configuration selection."""

import types
import unittest
from unittest import mock
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKBONE_ROOT = REPO_ROOT / 'backbone'
if str(BACKBONE_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKBONE_ROOT))

from backbone.model import BaseModel
from core.training_inproc import train_inprocessing_model


class SelectionIsolationTests(unittest.TestCase):
    def test_backbone_validation_fast_path_never_requests_test_edges(self):
        dummy = types.SimpleNamespace()
        dummy.device = torch.device('cpu')
        dummy.encoder = mock.Mock()
        dummy.encoder.eval = mock.Mock()
        dummy.encoder.return_value = torch.ones((3, 2))
        dummy.predictor = mock.Mock()
        dummy.predictor.eval = mock.Mock()
        dummy.predictor.side_effect = (
            lambda left, right: (left * right).sum(dim=1, keepdim=True)
        )
        dummy.create_input_feat = mock.Mock(return_value=torch.ones((3, 2)))

        data = types.SimpleNamespace(adj_t=object(), edge_weight=None)
        split_edge = {
            'train': {'edge': torch.tensor([[0, 1]])},
            'valid': {},
        }

        def fake_get_pos_neg_edges(split_name, _split_edge):
            self.assertEqual(split_name, 'valid')
            return (
                torch.tensor([[0, 2]], dtype=torch.long),
                torch.tensor([[1, 2]], dtype=torch.long),
            )

        with mock.patch(
            'backbone.model.get_pos_neg_edges',
            side_effect=fake_get_pos_neg_edges,
        ) as edge_loader:
            result = BaseModel.test(
                dummy,
                data,
                split_edge,
                batch_size=2,
                evaluator=object(),
                eval_metric='precision10',
                skip_test_eval=True,
            )

        self.assertEqual(edge_loader.call_count, 1)
        self.assertGreaterEqual(result['Precision@10'][0], 0.0)
        self.assertTrue(torch.isnan(torch.tensor(result['Precision@10'][1])))

    def test_training_can_skip_discarded_comprehensive_test_battery(self):
        model = types.SimpleNamespace(
            fairness_method='none',
            fairness_loss=None,
        )
        args = types.SimpleNamespace(
            epochs=0,
            eval_metric='all',
            skip_final_comprehensive_eval=True,
        )

        with mock.patch(
            'core.training_inproc.compute_baseline_metrics'
        ) as comprehensive:
            returned_model, best_epoch, results = train_inprocessing_model(
                model,
                data=object(),
                split_edge={},
                args=args,
                degrees=torch.empty(0),
                patience=1,
            )

        self.assertIs(returned_model, model)
        self.assertEqual(best_epoch, 0)
        self.assertEqual(results, {})
        comprehensive.assert_not_called()


if __name__ == '__main__':
    unittest.main()
