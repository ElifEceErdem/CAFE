"""Tests for the corrected-EG rerun finalization gate."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.egfix_finalize_reruns import (
    AGGREGATE_LONG_FILENAME,
    AGGREGATE_SEEDS,
    COMPLETION_MARKER,
    FINAL_DIRNAME,
    INTEGRITY_FILENAME,
    MODEL_ORDER,
    STRATIFIED_EG_SUMMARY_FILENAME,
    STRATIFIED_RAW_FILENAME,
    STRATIFIED_SEEDS,
    RerunValidationError,
    finalize_reruns,
)
from analysis.egfix_preflight import ensure_source_manifest, sha256_file
from experiments.egfix_protocol import (
    DATASETS,
    EG_GRAPH_SEMANTICS,
    OBJECTIVES,
    PENALTY_REDUCTION,
)


class EgfixRerunFinalizerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.campaign = self.root / "campaign"
        self.repo.mkdir()
        self.campaign.mkdir()
        os.environ.setdefault(
            "MPLCONFIGDIR", str(self.root / "matplotlib-cache")
        )
        os.environ.setdefault(
            "XDG_CACHE_HOME", str(self.root / "xdg-cache")
        )

        (self.repo / "source.py").write_text(
            "CORRECTED_EG = True\n", encoding="utf-8"
        )
        (self.repo / "split.pt").write_bytes(b"synthetic temporal split")
        self.split_sha = sha256_file(self.repo / "split.pt")
        self.source_manifest = self.campaign / "source_manifest.json"
        manifest = ensure_source_manifest(
            self.source_manifest,
            self.repo,
            source_paths=("source.py",),
            source_globs=(),
            split_relative_path="split.pt",
            expected_split_sha256=self.split_sha,
        )
        self.source_file_sha = sha256_file(self.source_manifest)
        self.source_content_sha = manifest["manifest_sha256"]

        self.selected_configs = self.campaign / "selected_configs.csv"
        self.params = self._write_selected_configs()
        self.selected_sha = sha256_file(self.selected_configs)
        self._write_aggregate_matrix()
        self._write_stratified_matrix()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_selected_configs(self):
        fields = (
            "dataset",
            "variant",
            "lambda_fair",
            "preproc_alpha",
            "beta",
            "lambda_mobility",
            "source_manifest_sha256",
            "source_manifest_content_sha256",
            "penalty_reduction",
            "evaluation_split",
            "eg_graph_semantics",
            "selection_runs",
        )
        params = {}
        with self.selected_configs.open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for dataset_index, dataset in enumerate(DATASETS):
                in_beta = 0.5 + 0.1 * dataset_index
                pre_beta = 1.0 + 0.1 * dataset_index
                params[dataset] = {
                    "cafe_in": {
                        "lambda_fair": 0.1,
                        "beta": in_beta,
                        "lambda_mobility": 0.5,
                    },
                    "cafe_pre": {
                        "preproc_alpha": 0.3,
                        "beta": pre_beta,
                        "lambda_mobility": 0.1,
                    },
                }
                common = {
                    "dataset": dataset,
                    "source_manifest_sha256": self.source_file_sha,
                    "source_manifest_content_sha256": (
                        self.source_content_sha
                    ),
                    "penalty_reduction": PENALTY_REDUCTION,
                    "evaluation_split": "valid",
                    "eg_graph_semantics": EG_GRAPH_SEMANTICS,
                    "selection_runs": 3,
                }
                writer.writerow(
                    {
                        **common,
                        "variant": "cafe_in",
                        "lambda_fair": 0.1,
                        "preproc_alpha": "",
                        "beta": in_beta,
                        "lambda_mobility": 0.5,
                    }
                )
                writer.writerow(
                    {
                        **common,
                        "variant": "cafe_pre",
                        "lambda_fair": "",
                        "preproc_alpha": 0.3,
                        "beta": pre_beta,
                        "lambda_mobility": 0.1,
                    }
                )
        return params

    @staticmethod
    def _fingerprint(dataset):
        return hashlib.sha256(dataset.encode("utf-8")).hexdigest()

    def _aggregate_rows(self, dataset, seed):
        params = self.params[dataset]
        rows = []
        for method_index, method in enumerate(MODEL_ORDER):
            for metric_index, metric in enumerate(OBJECTIVES):
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "split_seed": 42,
                        "method": method,
                        "metric": metric,
                        "value": (
                            method_index
                            + metric_index / 10
                            + (seed - 42) / 100
                        ),
                        "protocol": "headline",
                        "evaluation_split": "test",
                        "eg_graph_semantics": EG_GRAPH_SEMANTICS,
                        "penalty_reduction": PENALTY_REDUCTION,
                        "closeness_random_seed": 42,
                        "closeness_sample_size": 0,
                        "closeness_epsilon": 0.1,
                        "split_fingerprint": self._fingerprint(dataset),
                        "split_sha256": (
                            self.split_sha
                            if dataset == "brightkite"
                            else ""
                        ),
                        "selected_configs_sha256": self.selected_sha,
                        "source_manifest_sha256": self.source_file_sha,
                        "cafe_in_lambda": params["cafe_in"][
                            "lambda_fair"
                        ],
                        "cafe_in_beta": params["cafe_in"]["beta"],
                        "cafe_in_lam": params["cafe_in"][
                            "lambda_mobility"
                        ],
                        "cafe_pre_alpha": params["cafe_pre"][
                            "preproc_alpha"
                        ],
                        "cafe_pre_beta": params["cafe_pre"]["beta"],
                        "cafe_pre_lam": params["cafe_pre"][
                            "lambda_mobility"
                        ],
                        "fusion_beta": params["cafe_in"]["beta"],
                    }
                )
        return rows

    def _write_aggregate_matrix(self):
        for dataset in DATASETS:
            for seed in AGGREGATE_SEEDS:
                task = (
                    self.campaign
                    / "aggregate"
                    / dataset
                    / f"seed_{seed}"
                )
                task.mkdir(parents=True)
                csv_path = task / "per_seed_metrics.csv"
                rows = self._aggregate_rows(dataset, seed)
                with csv_path.open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=list(rows[0])
                    )
                    writer.writeheader()
                    writer.writerows(rows)
                marker = {
                    "status": "complete",
                    "dataset": dataset,
                    "protocol": "headline",
                    "seeds": [seed],
                    "csv_sha256": sha256_file(csv_path),
                    "selected_configs_sha256": self.selected_sha,
                    "source_manifest_sha256": self.source_file_sha,
                    "eg_graph_semantics": EG_GRAPH_SEMANTICS,
                    "penalty_reduction": PENALTY_REDUCTION,
                }
                (task / "per_seed_metrics.csv.complete.json").write_text(
                    json.dumps(marker, sort_keys=True), encoding="utf-8"
                )
                (task / "_SUCCESS").touch()

    @staticmethod
    def _targets(sources, num_nodes):
        return np.asarray(
            [
                [(int(source) + offset) % num_nodes for offset in range(1, 11)]
                for source in sources
            ],
            dtype=np.int64,
        )

    def _run_metadata(self, dataset, seed, num_nodes, num_users):
        params = self.params[dataset]
        return {
            "status": "complete",
            "dataset": dataset,
            "training_seed": seed,
            "split_seed": 42,
            "split_fingerprint": self._fingerprint(dataset),
            "split_sha256": (
                self.split_sha if dataset == "brightkite" else None
            ),
            "num_nodes": num_nodes,
            "num_eval_users": num_users,
            "k": 10,
            "num_bins_requested": 10,
            "methods": list(MODEL_ORDER),
            "eg_graph_semantics": EG_GRAPH_SEMANTICS,
            "evaluation_split": "test",
            "penalty_reduction": PENALTY_REDUCTION,
            "selected_configs_sha256": self.selected_sha,
            "source_manifest_sha256": self.source_file_sha,
            "fusion_beta": params["cafe_in"]["beta"],
            "cafe_in": params["cafe_in"],
            "cafe_pre": params["cafe_pre"],
            "recommendation_validation_passed": True,
        }

    def _write_stratified_matrix(self):
        sources = np.asarray([0, 1, 2, 3], dtype=np.int64)
        num_nodes = 20
        targets = self._targets(sources, num_nodes)
        for dataset in DATASETS:
            for seed in STRATIFIED_SEEDS:
                task = (
                    self.campaign
                    / "stratified"
                    / dataset
                    / f"seed_{seed}"
                )
                task.mkdir(parents=True)
                run = self._run_metadata(
                    dataset, seed, num_nodes, len(sources)
                )
                rec_path = (
                    task / f"recommendations_{dataset}_seed{seed}.npz"
                )
                payload = {
                    "sources": sources,
                    "metadata_json": np.asarray(
                        json.dumps(run, sort_keys=True), dtype=np.str_
                    ),
                }
                for key in (
                    "targets_ra",
                    "targets_hl_gnn",
                    "targets_fusion",
                    "targets_cafe_in",
                    "targets_cafe_pre",
                ):
                    payload[key] = targets
                np.savez_compressed(rec_path, **payload)
                run["recommendations_file"] = rec_path.name
                run["recommendations_sha256"] = sha256_file(rec_path)

                precision_path = task / f"stratified_cafe_{dataset}.csv"
                precision_rows = []
                for model_index, model in enumerate(MODEL_ORDER):
                    for source in sources:
                        bin_index = 0 if int(source) < 2 else 1
                        precision_rows.append(
                            {
                                "node_id": int(source),
                                "model": model,
                                "precision": (
                                    (model_index + bin_index + seed) % 8
                                )
                                / 10,
                                "node_coreness": bin_index + 1,
                                "degree": int(source) + 2,
                                "run": seed,
                                "training_seed": seed,
                                "split_seed": 42,
                                "split_fingerprint": (
                                    self._fingerprint(dataset)
                                ),
                                "coreness_bin": bin_index,
                                "coreness_bin_label": (
                                    "[1-1]"
                                    if bin_index == 0
                                    else "[2-2]"
                                ),
                            }
                        )
                with precision_path.open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=list(precision_rows[0])
                    )
                    writer.writeheader()
                    writer.writerows(precision_rows)

                eg_path = (
                    task
                    / f"eg_by_coreness_cafe_{dataset}_per_run.csv"
                )
                eg_rows = []
                for model_index, model in enumerate(MODEL_ORDER):
                    for bin_index in range(2):
                        eg_rows.append(
                            {
                                "model": model,
                                "coreness_bin": bin_index,
                                "coreness_bin_label": (
                                    "[1-1]"
                                    if bin_index == 0
                                    else "[2-2]"
                                ),
                                "equality_gain": (
                                    model_index
                                    + bin_index
                                    + (seed - 42) / 10
                                ),
                                "run": seed,
                                "training_seed": seed,
                                "split_seed": 42,
                            }
                        )
                with eg_path.open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=list(eg_rows[0])
                    )
                    writer.writeheader()
                    writer.writerows(eg_rows)

                top = {
                    "status": "complete",
                    "dataset": dataset,
                    "runs": [run],
                    "actual_bins": 2,
                    "precision_rows": len(precision_rows),
                    "eg_rows": len(eg_rows),
                    "methods": list(MODEL_ORDER),
                    "selected_configs_sha256": self.selected_sha,
                    "source_manifest_sha256": self.source_file_sha,
                    "eg_graph_semantics": EG_GRAPH_SEMANTICS,
                    "penalty_reduction": PENALTY_REDUCTION,
                }
                (task / f"metadata_{dataset}.json").write_text(
                    json.dumps(top, sort_keys=True), encoding="utf-8"
                )
                (task / "_SUCCESS").touch()

    def _finalize(self):
        return finalize_reruns(
            campaign_root=self.campaign,
            selected_configs_path=self.selected_configs,
            source_manifest_path=self.source_manifest,
            repo_root=self.repo,
            manifest_source_paths=("source.py",),
            manifest_source_globs=(),
            split_relative_path="split.pt",
            expected_brightkite_split_sha256=self.split_sha,
        )

    def test_complete_campaign_is_merged_with_run_level_uncertainty(self):
        result = self._finalize()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["aggregate_objective_rows"], 1000)
        final_dir = self.campaign / FINAL_DIRNAME
        self.assertTrue((self.campaign / COMPLETION_MARKER).is_file())
        self.assertTrue((final_dir / INTEGRITY_FILENAME).is_file())
        self.assertTrue((final_dir / AGGREGATE_LONG_FILENAME).is_file())
        self.assertTrue((final_dir / STRATIFIED_RAW_FILENAME).is_file())
        self.assertTrue(
            (final_dir / STRATIFIED_EG_SUMMARY_FILENAME).is_file()
        )
        pdfs = sorted((final_dir / "figures").glob("*.pdf"))
        self.assertEqual(len(pdfs), 5)
        self.assertTrue(all(path.read_bytes().startswith(b"%PDF-") for path in pdfs))

        with (final_dir / AGGREGATE_LONG_FILENAME).open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        first_cell = [
            row
            for row in rows
            if row["dataset"] == DATASETS[0]
            and row["seed"] == "42"
            and row["metric"] == OBJECTIVES[0]
        ]
        self.assertEqual(
            [row["method"] for row in first_cell], list(MODEL_ORDER)
        )
        integrity = json.loads(
            (final_dir / INTEGRITY_FILENAME).read_text(encoding="utf-8")
        )
        self.assertEqual(integrity["status"], "complete")
        self.assertEqual(
            integrity["degree_eg_crosscheck"]["status"], "skipped"
        )
        self.assertEqual(len(integrity["publication_pdfs"]), 5)

    def test_missing_shard_success_marker_is_rejected(self):
        marker = (
            self.campaign
            / "aggregate"
            / DATASETS[-1]
            / f"seed_{AGGREGATE_SEEDS[-1]}"
            / "_SUCCESS"
        )
        marker.unlink()
        with self.assertRaisesRegex(
            RerunValidationError, "success-marker matrix"
        ):
            self._finalize()
        self.assertFalse((self.campaign / COMPLETION_MARKER).exists())
        self.assertFalse((self.campaign / FINAL_DIRNAME).exists())

    def test_nonfinite_required_objective_is_rejected(self):
        task = (
            self.campaign
            / "aggregate"
            / DATASETS[0]
            / f"seed_{AGGREGATE_SEEDS[0]}"
        )
        csv_path = task / "per_seed_metrics.csv"
        with csv_path.open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
            fields = list(rows[0])
        rows[0]["value"] = "nan"
        with csv_path.open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        marker_path = task / "per_seed_metrics.csv.complete.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["csv_sha256"] = sha256_file(csv_path)
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
        with self.assertRaisesRegex(RerunValidationError, "must be finite"):
            self._finalize()

    def test_duplicate_recommendation_target_is_rejected(self):
        dataset = DATASETS[0]
        seed = STRATIFIED_SEEDS[0]
        task = (
            self.campaign / "stratified" / dataset / f"seed_{seed}"
        )
        rec_path = task / f"recommendations_{dataset}_seed{seed}.npz"
        with np.load(rec_path, allow_pickle=False) as archive:
            payload = {key: np.asarray(archive[key]) for key in archive.files}
        changed = payload["targets_ra"].copy()
        changed[0, 1] = changed[0, 0]
        payload["targets_ra"] = changed
        np.savez_compressed(rec_path, **payload)
        metadata_path = task / f"metadata_{dataset}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["runs"][0]["recommendations_sha256"] = sha256_file(rec_path)
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(
            RerunValidationError, "duplicate target"
        ):
            self._finalize()

    def test_unmatched_fusion_beta_is_rejected(self):
        task = (
            self.campaign
            / "aggregate"
            / DATASETS[0]
            / f"seed_{AGGREGATE_SEEDS[0]}"
        )
        csv_path = task / "per_seed_metrics.csv"
        with csv_path.open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
            fields = list(rows[0])
        rows[0]["fusion_beta"] = "99"
        with csv_path.open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        marker_path = task / "per_seed_metrics.csv.complete.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["csv_sha256"] = sha256_file(csv_path)
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
        with self.assertRaisesRegex(
            RerunValidationError, "fusion_beta must equal selected"
        ):
            self._finalize()


if __name__ == "__main__":
    unittest.main()
