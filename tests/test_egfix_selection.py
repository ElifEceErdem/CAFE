"""Focused tests for the corrected-EG selection campaign."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from analysis.egfix_finalize_selection import (
    SELECTED_COLUMNS,
    SelectionValidationError,
    finalize_selection,
    pareto_asf_select,
)
from analysis.egfix_preflight import (
    PreflightError,
    ensure_source_manifest,
    sha256_file,
    verify_source_manifest,
)
from experiments.egfix_protocol import (
    BETA_VALUES,
    BRIGHTKITE_TEMPORAL_SPLIT,
    BRIGHTKITE_TEMPORAL_SPLIT_SHA256,
    CLOSENESS_EPSILON,
    CLOSENESS_RANDOM_SEED,
    CLOSENESS_SAMPLE_SIZE,
    CONFIGS_PER_PENALTY,
    CONFIGS_PER_VARIANT,
    DATASETS,
    EG_GRAPH_SEMANTICS,
    EVALUATION_SPLIT,
    K_VALUES,
    MOBILITY_VALUES,
    OBJECTIVES,
    PENALTY_REDUCTION,
    SEED_BASE,
    SEED_RULE,
    SELECTION_RUNS,
    SELECTION_TASK_COUNT,
    SPLIT_SEED,
    USE_DEMOGRAPHICS,
    VARIANTS,
    expected_config_keys,
    expected_penalty_config_keys,
    make_config_key,
    parse_config_key,
    selection_tasks,
    variant_spec,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class EgfixProtocolTests(unittest.TestCase):
    def test_definitive_grid_and_task_matrix(self):
        self.assertEqual(SELECTION_TASK_COUNT, 70)
        self.assertEqual(len(selection_tasks()), 70)
        self.assertEqual(
            len(
                {
                    (
                        task["dataset"],
                        task["variant"],
                        task["penalty_value"],
                    )
                    for task in selection_tasks()
                }
            ),
            70,
        )
        for variant in VARIANTS:
            keys = expected_config_keys(variant)
            self.assertEqual(len(keys), CONFIGS_PER_VARIANT)
            self.assertEqual(len(set(keys)), CONFIGS_PER_VARIANT)
            for key in keys:
                parsed = parse_config_key(key)
                self.assertEqual(parsed["variant"], variant)
                self.assertEqual(
                    make_config_key(
                        variant,
                        parsed["penalty_value"],
                        parsed["beta_ra"],
                        parsed["lambda_mobility"],
                    ),
                    key,
                )

    def test_brightkite_temporal_split_is_frozen(self):
        path = REPO_ROOT / BRIGHTKITE_TEMPORAL_SPLIT
        self.assertTrue(path.is_file())
        self.assertEqual(
            sha256_file(path), BRIGHTKITE_TEMPORAL_SPLIT_SHA256
        )

    def test_asf_tie_break_is_lexicographic(self):
        rows = []
        for config_key in ("z_config", "a_config"):
            row = {"config_key": config_key}
            for objective in OBJECTIVES:
                row[f"{objective}_mean"] = 1.0
            rows.append(row)
        selected = pareto_asf_select(rows)
        self.assertEqual(selected["config_key"], "a_config")
        self.assertTrue(all(row["is_pareto"] for row in rows))
        self.assertEqual(sum(bool(row["selected"]) for row in rows), 1)


class EgfixPreflightTests(unittest.TestCase):
    def test_manifest_is_immutable_and_detects_source_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.py"
            split = root / "split.pt"
            source.write_text("answer = 42\n", encoding="utf-8")
            split.write_bytes(b"fixed temporal split")
            split_sha = sha256_file(split)
            manifest_path = root / "manifest.json"

            created = ensure_source_manifest(
                manifest_path,
                root,
                source_paths=("source.py",),
                source_globs=(),
                split_relative_path="split.pt",
                expected_split_sha256=split_sha,
            )
            verified = verify_source_manifest(
                manifest_path,
                root,
                source_paths=("source.py",),
                source_globs=(),
                split_relative_path="split.pt",
                expected_split_sha256=split_sha,
            )
            self.assertEqual(created, verified)
            self.assertEqual(
                created["brightkite_temporal_split"]["sha256"], split_sha
            )

            source.write_text("answer = 43\n", encoding="utf-8")
            with self.assertRaisesRegex(
                PreflightError, "no longer matches"
            ):
                verify_source_manifest(
                    manifest_path,
                    root,
                    source_paths=("source.py",),
                    source_globs=(),
                    split_relative_path="split.pt",
                    expected_split_sha256=split_sha,
                )


class EgfixFinalizerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.selection = self.root / "selection"
        self.output = self.root / "output"
        self.repo.mkdir()
        self.selection.mkdir()

        (self.repo / "source.py").write_text(
            "CAMPAIGN_SOURCE = True\n", encoding="utf-8"
        )
        (self.repo / "split.pt").write_bytes(b"synthetic fixed split")
        self.split_sha = sha256_file(self.repo / "split.pt")
        self.source_manifest_path = self.root / "source_manifest.json"
        ensure_source_manifest(
            self.source_manifest_path,
            self.repo,
            source_paths=("source.py",),
            source_globs=(),
            split_relative_path="split.pt",
            expected_split_sha256=self.split_sha,
        )
        self.source_manifest_file_sha = sha256_file(
            self.source_manifest_path
        )
        self._write_complete_grid()

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _statistics(values):
        mean = math.fsum(values) / len(values)
        std = math.sqrt(
            math.fsum((value - mean) ** 2 for value in values)
            / len(values)
        )
        return {
            "mean": mean,
            "std": std,
            "min": min(values),
            "max": max(values),
            "values": values,
        }

    def _metadata(self, dataset, variant, penalty):
        spec = variant_spec(variant)
        return {
            "dataset": dataset,
            "penalization_mode": spec["penalization_mode"],
            "use_mobility_boost": True,
            "selection_mode": True,
            "evaluation_split": EVALUATION_SPLIT,
            "metric_split": EVALUATION_SPLIT,
            "test_metrics_evaluated": False,
            "skip_final_comprehensive_eval": True,
            "eg_graph_semantics": EG_GRAPH_SEMANTICS,
            "penalty_reduction": PENALTY_REDUCTION,
            "num_runs": SELECTION_RUNS,
            "reset_seed_per_model": True,
            "seed_base": SEED_BASE,
            "seed_rule": SEED_RULE,
            "split_seed": SPLIT_SEED,
            "split_fingerprint": hashlib.sha256(
                dataset.encode("utf-8")
            ).hexdigest(),
            "split_sha256": (
                self.split_sha if dataset == "brightkite" else None
            ),
            "k_values": list(K_VALUES),
            "use_demographics": USE_DEMOGRAPHICS,
            "centrality_types": ["pagerank", "closeness"],
            "closeness_random_seed": CLOSENESS_RANDOM_SEED,
            "closeness_sample_size": CLOSENESS_SAMPLE_SIZE,
            "closeness_epsilon": CLOSENESS_EPSILON,
            "source_manifest_sha256": self.source_manifest_file_sha,
            "model_seeds": [
                SEED_BASE + index for index in range(SELECTION_RUNS)
            ],
            "grid": {
                spec["penalty_name"]: [float(penalty)],
                "beta": list(BETA_VALUES),
                "lambda_mobility": list(MOBILITY_VALUES),
            },
        }

    def _write_complete_grid(self):
        config_indices = {
            variant: {
                key: index
                for index, key in enumerate(expected_config_keys(variant))
            }
            for variant in VARIANTS
        }
        for dataset in DATASETS:
            for variant in VARIANTS:
                spec = variant_spec(variant)
                for penalty in spec["penalties"]:
                    keys = expected_penalty_config_keys(variant, penalty)
                    individual_runs = [
                        {"baseline": {}} for _ in range(SELECTION_RUNS)
                    ]
                    lambda_results = {}
                    for key in keys:
                        # Identical objective ordering makes the final grid
                        # point uniquely dominate, while the small offsets
                        # exercise three-run mean/std validation.
                        base = (
                            config_indices[variant][key] + 1
                        ) / CONFIGS_PER_VARIANT
                        values = [base - 0.001, base, base + 0.001]
                        metrics = {
                            objective: list(values)
                            for objective in OBJECTIVES
                        }
                        lambda_results[key] = {
                            "statistics": {
                                objective: self._statistics(values)
                                for objective in OBJECTIVES
                            }
                        }
                        for run_index in range(SELECTION_RUNS):
                            individual_runs[run_index][key] = {
                                objective: metrics[objective][run_index]
                                for objective in OBJECTIVES
                            }
                    directory = self.selection / dataset / variant
                    directory.mkdir(parents=True, exist_ok=True)
                    path = directory / (
                        f"{spec['penalty_name']}_{float(penalty)}.json"
                    )
                    path.write_text(
                        json.dumps(
                            {
                                "experiment_info": self._metadata(
                                    dataset, variant, penalty
                                ),
                                "lambda_results": lambda_results,
                                "individual_runs": individual_runs,
                            },
                            allow_nan=False,
                        ),
                        encoding="utf-8",
                    )

    def _finalize(self, output):
        return finalize_selection(
            selection_root=self.selection,
            source_manifest_path=self.source_manifest_path,
            output_dir=output,
            repo_root=self.repo,
            manifest_source_paths=("source.py",),
            manifest_source_globs=(),
            split_relative_path="split.pt",
            expected_brightkite_split_sha256=self.split_sha,
        )

    def test_full_grid_is_validated_selected_and_exported(self):
        summary = self._finalize(self.output)
        self.assertEqual(summary["selection_artifacts"], 70)
        self.assertEqual(
            summary["candidates"],
            len(DATASETS) * len(VARIANTS) * CONFIGS_PER_VARIANT,
        )
        self.assertEqual(
            summary["run_rows"],
            len(DATASETS)
            * len(VARIANTS)
            * CONFIGS_PER_VARIANT
            * SELECTION_RUNS,
        )
        self.assertEqual(summary["selected"], 10)

        with (self.output / "selected_configs.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            self.assertEqual(tuple(reader.fieldnames), SELECTED_COLUMNS)
        self.assertEqual(len(rows), 10)
        self.assertEqual(
            {(row["dataset"], row["variant"]) for row in rows},
            {
                (dataset, variant)
                for dataset in DATASETS
                for variant in VARIANTS
            },
        )
        self.assertTrue(
            all(
                row["source_manifest_sha256"]
                == self.source_manifest_file_sha
                for row in rows
            )
        )

        with (self.output / "selection_candidates.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            candidates = list(csv.DictReader(handle))
        self.assertEqual(len(candidates), 2100)
        self.assertEqual(
            sum(row["selected"] == "True" for row in candidates), 10
        )

        with (self.output / "selection_runs.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            runs = list(csv.DictReader(handle))
        self.assertEqual(len(runs), 6300)

    def test_test_metric_metadata_is_rejected(self):
        path = next(self.selection.rglob("*.json"))
        document = json.loads(path.read_text(encoding="utf-8"))
        document["experiment_info"]["test_metrics_evaluated"] = True
        path.write_text(
            json.dumps(document, allow_nan=False), encoding="utf-8"
        )
        with self.assertRaisesRegex(
            SelectionValidationError, "test_metrics_evaluated"
        ):
            self._finalize(self.root / "rejected")

    def test_partial_artifact_is_rejected(self):
        (self.selection / "orphan.partial.json").write_text(
            "{}", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            SelectionValidationError, "Partial selection artifacts"
        ):
            self._finalize(self.root / "rejected")

    def test_incomplete_penalty_arm_is_rejected(self):
        path = next(self.selection.rglob("*.json"))
        document = json.loads(path.read_text(encoding="utf-8"))
        removed = next(iter(document["lambda_results"]))
        del document["lambda_results"][removed]
        for run in document["individual_runs"]:
            del run[removed]
        path.write_text(
            json.dumps(document, allow_nan=False), encoding="utf-8"
        )
        with self.assertRaisesRegex(
            SelectionValidationError, "exactly 30 configurations"
        ):
            self._finalize(self.root / "rejected")


if __name__ == "__main__":
    unittest.main()
