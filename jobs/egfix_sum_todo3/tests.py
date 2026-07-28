"""Static, unit, and live-lineage tests for the isolated TODO3 workflow."""

from __future__ import annotations

import csv
import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from jobs.egfix_sum_todo3 import finalize_reruns as finalizer
from jobs.egfix_sum_todo3.compose_selected_configs import (
    CompositeConfigError,
    write_composite_configs,
)
from jobs.egfix_sum_todo3.protocol import (
    AGGREGATE_TASK_COUNT,
    K,
    METHODS,
    PENALTY_REDUCTION,
    PROFILE,
    STRATIFIED_TASK_COUNT,
    TARGET_EXPOSURE_SEMANTICS,
    protocol_document,
)
from jobs.egfix_sum_todo3.provenance import (
    REPO_ROOT,
    TODO3_REQUIRED_FILES,
    ensure_source_manifest,
    verify_source_manifest,
)
from jobs.egfix_sum_todo3.target_exposure import (
    TargetExposureError,
    _summaries,
    assign_target_tiers,
    target_event_profile,
)


JOB_ROOT = REPO_ROOT / "jobs" / "egfix_sum_todo3"
DEFAULT_SUM_CAMPAIGN = (
    REPO_ROOT / "results_egfix" / "20260724_degree_eg_sum_in_v1"
)
DEFAULT_PRE_CAMPAIGN = (
    REPO_ROOT / "results_egfix" / "20260723_degree_eg_v1"
)


class Todo3ProtocolTests(unittest.TestCase):
    def test_protocol_is_exactly_the_requested_matrix(self):
        document = protocol_document()
        self.assertEqual(PROFILE, "rq3_degree_eg_sum_todo3_v1")
        self.assertEqual(PENALTY_REDUCTION, "sum")
        self.assertEqual(AGGREGATE_TASK_COUNT, 50)
        self.assertEqual(STRATIFIED_TASK_COUNT, 15)
        self.assertEqual(document["aggregate_seeds"], list(range(42, 52)))
        self.assertEqual(
            document["stratified_seeds"], list(range(42, 45))
        )
        self.assertEqual(document["loss_func"], "AUC")
        self.assertEqual(document["num_neg"], 1)
        self.assertFalse(document["manuscript_writes"])

    def test_target_semantics_are_not_undirected_edge_semantics(self):
        target = protocol_document()["target_exposure"]
        self.assertEqual(
            TARGET_EXPOSURE_SEMANTICS,
            "directed_recommendation_events",
        )
        self.assertFalse(target["reciprocal_events_deduplicated"])
        self.assertEqual(target["percentile_method"], "linear")
        self.assertEqual(target["boundary"], "right_closed")


class TargetExposureUnitTests(unittest.TestCase):
    def test_right_closed_tiers_keep_large_minimum_tie_in_periphery(self):
        values = np.asarray([1] * 5 + [2] * 3 + [3] * 4, dtype=float)
        tiers, thresholds, counts = assign_target_tiers(values)
        self.assertEqual(float(thresholds[0]), 1.0)
        self.assertTrue(np.all(tiers[:5] == 0))
        self.assertEqual(counts.tolist(), [5, 3, 4])

    def test_empty_tier_is_rejected_instead_of_rank_splitting_ties(self):
        with self.assertRaisesRegex(TargetExposureError, "empty target tier"):
            assign_target_tiers(np.ones(12))

    def test_event_profile_includes_zero_exposure_nodes_and_conserves_mass(self):
        num_nodes = 24
        num_users = 2
        tiers = np.repeat(np.arange(3), 8)
        tier_counts = np.asarray([8, 8, 8])
        targets = np.asarray(
            [
                [0, 1, 2, 3, 4, 5, 8, 9, 16, 17],
                [0, 1, 2, 3, 8, 9, 10, 11, 16, 17],
            ],
            dtype=np.int64,
        )
        profile = target_event_profile(
            targets,
            num_nodes=num_nodes,
            num_eval_users=num_users,
            tiers=tiers,
            tier_counts=tier_counts,
        )
        self.assertEqual(sum(row["event_count"] for row in profile), 20)
        self.assertAlmostEqual(
            sum(row["event_share"] for row in profile), 1.0
        )
        self.assertAlmostEqual(
            sum(row["exposure_mass"] for row in profile), K
        )
        # Only 14 of 24 targets occur; the tier means still divide by all
        # eight tier nodes, so zero-exposure nodes are included.
        self.assertAlmostEqual(
            profile[0]["mean_exposure_per_node"],
            profile[0]["event_count"] / (num_users * 8),
        )

    @staticmethod
    def _synthetic_by_run(fusion_value: float = 1.0) -> pd.DataFrame:
        rows = []
        datasets = (
            "ego-facebook",
            "oklahoma",
            "unc",
            "pokec_n",
            "brightkite",
        )
        for dataset in datasets:
            for seed in (42, 43, 44):
                for method in METHODS:
                    for tier in range(3):
                        base = fusion_value + tier * 0.1
                        value = (
                            base + 0.2
                            if method == "CAFE-In"
                            else base
                        )
                        rows.append(
                            {
                                "dataset": dataset,
                                "training_seed": seed,
                                "split_seed": 42,
                                "split_fingerprint": "a" * 64,
                                "method": method,
                                "tier_index": tier,
                                "target_tier": (
                                    "periphery",
                                    "middle",
                                    "core",
                                )[tier],
                                "q_1_3": 1.0,
                                "q_2_3": 2.0,
                                "tier_node_count": 10,
                                "tier_node_share": 1 / 3,
                                "num_nodes": 30,
                                "num_eval_users": 10,
                                "k": 10,
                                "event_count": int(value * 100),
                                "event_share": value / 10,
                                "exposure_mass": value * 3,
                                "mean_exposure_per_node": value,
                                "recommendations_path": (
                                    f"stratified/{dataset}/seed_{seed}/x.npz"
                                ),
                                "recommendations_sha256": "b" * 64,
                            }
                        )
        return pd.DataFrame(rows)

    def test_paired_contrast_uses_runs_and_reports_ratio_of_means(self):
        summary, contrast, contrast_summary = _summaries(
            self._synthetic_by_run()
        )
        self.assertEqual(len(summary), 75)
        self.assertEqual(len(contrast), 45)
        self.assertEqual(len(contrast_summary), 15)
        first = contrast_summary.iloc[0]
        self.assertEqual(int(first["run_count"]), 3)
        self.assertAlmostEqual(first["absolute_delta_mean"], 0.2)
        self.assertEqual(int(first["positive_runs"]), 3)
        self.assertEqual(int(first["negative_runs"]), 0)

    def test_zero_fusion_denominator_is_rejected(self):
        with self.assertRaisesRegex(
            TargetExposureError, "relative change is undefined"
        ):
            _summaries(self._synthetic_by_run(fusion_value=0.0))


class Todo3JobStaticTests(unittest.TestCase):
    def test_all_required_sources_exist_and_shell_parses(self):
        paths = [REPO_ROOT / relative for relative in TODO3_REQUIRED_FILES]
        self.assertTrue(all(path.is_file() for path in paths))
        shell_paths = [
            path for path in paths if path.suffix in {".job", ".sh"}
        ]
        subprocess.run(
            ["bash", "-n", *(str(path) for path in shell_paths)],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_gpu_arrays_and_sum_validity_tuple_are_pinned(self):
        aggregate = (JOB_ROOT / "aggregate.job").read_text(
            encoding="utf-8"
        )
        stratified = (JOB_ROOT / "stratified.job").read_text(
            encoding="utf-8"
        )
        self.assertIn("#SBATCH --array=0-49%10", aggregate)
        self.assertIn("#SBATCH --array=0-14%5", stratified)
        for text in (aggregate, stratified):
            self.assertIn("--penalty_reduction sum", text)
            self.assertIn("--loss_func AUC", text)
            self.assertIn("--num_neg 1", text)
            self.assertIn("--batch_size 16384", text)
            self.assertIn("--eval_skip_test True", text)
            self.assertNotIn("--penalty_reduction mean", text)
        self.assertIn("--save_recommendations True", stratified)

    def test_launcher_has_guarded_afterok_dag(self):
        launcher = (JOB_ROOT / "submit_reruns.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("--confirm-selected-configs", launcher)
        self.assertIn('dependency="afterok:${PREFLIGHT_ID}"', launcher)
        self.assertIn(
            'dependency="afterok:${AGGREGATE_ID}:${STRATIFIED_ID}"',
            launcher,
        )
        self.assertIn('if [[ -e "${CAMPAIGN}" ]]', launcher)
        self.assertNotIn("ToRS_paper", launcher)

    def test_base_finalizer_adapter_is_process_local_and_restores_globals(self):
        original_reduction = finalizer.base.PENALTY_REDUCTION
        original_verifier = finalizer.base.verify_source_manifest
        original_dir = finalizer.base.FINAL_DIRNAME
        with finalizer._sum_base_adapter():
            self.assertEqual(
                finalizer.base.PENALTY_REDUCTION, "sum"
            )
            self.assertEqual(
                finalizer.base.FINAL_DIRNAME,
                finalizer.CORE_DIRNAME,
            )
        self.assertEqual(
            finalizer.base.PENALTY_REDUCTION, original_reduction
        )
        self.assertIs(
            finalizer.base.verify_source_manifest, original_verifier
        )
        self.assertEqual(finalizer.base.FINAL_DIRNAME, original_dir)

    def test_frozen_base_finalizer_hash_matches_reviewed_source(self):
        from jobs.egfix_sum_todo3.provenance import sha256_file

        self.assertEqual(
            sha256_file(REPO_ROOT / "analysis/egfix_finalize_reruns.py"),
            finalizer.EXPECTED_BASE_FINALIZER_SHA256,
        )

    def test_promoted_target_integrity_uses_a_stable_relative_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary) / "campaign"
            staging = campaign / ".staging"
            final_dir = campaign / finalizer.FINAL_DIRNAME
            staging.mkdir(parents=True)
            source_manifest = campaign / "source_manifest.json"
            selected_configs = campaign / "selected_configs.csv"
            source_manifest.write_text("source\n", encoding="utf-8")
            selected_configs.write_text("selected\n", encoding="utf-8")
            (staging / finalizer.base.INTEGRITY_FILENAME).write_text(
                "core\n", encoding="utf-8"
            )

            target_rows = {
                finalizer.TARGET_BY_RUN_FILENAME: 225,
                finalizer.TARGET_SUMMARY_FILENAME: 75,
                finalizer.TARGET_CONTRAST_BY_RUN_FILENAME: 45,
                finalizer.TARGET_CONTRAST_SUMMARY_FILENAME: 15,
            }
            target_outputs = []
            for filename, rows in target_rows.items():
                path = staging / filename
                path.write_text(f"{filename}\n", encoding="utf-8")
                target_outputs.append(
                    {
                        "path": filename,
                        "sha256": finalizer.sha256_file(path),
                        "size_bytes": path.stat().st_size,
                        "rows": rows,
                    }
                )
            target_integrity = {
                "status": "complete",
                "datasets": list(finalizer.DATASETS),
                "training_seeds": list(finalizer.STRATIFIED_SEEDS),
                "methods": list(finalizer.METHODS),
                "core_integrity_path": (
                    finalizer.base.INTEGRITY_FILENAME
                ),
                "core_integrity_sha256": finalizer.sha256_file(
                    staging / finalizer.base.INTEGRITY_FILENAME
                ),
                "outputs": target_outputs,
            }
            target_integrity["integrity_payload_sha256"] = hashlib.sha256(
                finalizer.canonical_json_bytes(target_integrity)
            ).hexdigest()
            finalizer._atomic_write_json(
                staging / finalizer.TARGET_INTEGRITY_FILENAME,
                target_integrity,
            )

            task_records = [{"stage": "synthetic"}]
            outer = {
                "status": "complete",
                "profile": finalizer.PROFILE,
                "campaign_root": str(campaign.resolve()),
                "provenance": {
                    "source_manifest_sha256": finalizer.sha256_file(
                        source_manifest
                    ),
                    "selected_configs_sha256": finalizer.sha256_file(
                        selected_configs
                    ),
                },
                "task_provenance": {
                    "record_count": len(task_records),
                    "records": task_records,
                },
                "target_exposure": {
                    "status": "complete",
                    "by_run_rows": 225,
                    "summary_rows": 75,
                    "contrast_by_run_rows": 45,
                    "contrast_summary_rows": 15,
                    "integrity_file": (
                        finalizer.TARGET_INTEGRITY_FILENAME
                    ),
                    "integrity_sha256": finalizer.sha256_file(
                        staging / finalizer.TARGET_INTEGRITY_FILENAME
                    ),
                },
                "outputs": finalizer._output_records(staging),
                "manuscript_writes": False,
            }
            outer["integrity_payload_sha256"] = hashlib.sha256(
                finalizer.canonical_json_bytes(outer)
            ).hexdigest()
            finalizer._atomic_write_json(
                staging / finalizer.OUTER_INTEGRITY_FILENAME,
                outer,
            )
            os.replace(staging, final_dir)

            validated = finalizer._validate_finished_directory(
                final_dir,
                campaign,
                source_manifest=source_manifest,
                selected_configs=selected_configs,
                task_records=task_records,
            )
            self.assertEqual(
                validated["target_exposure"]["integrity_file"],
                finalizer.TARGET_INTEGRITY_FILENAME,
            )


class Todo3LiveLineageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sum_path = Path(
            os.environ.get(
                "EGFIX_TODO3_SUM_CAMPAIGN",
                str(DEFAULT_SUM_CAMPAIGN),
            )
        )
        pre_path = Path(
            os.environ.get(
                "EGFIX_TODO3_PRE_CAMPAIGN",
                str(DEFAULT_PRE_CAMPAIGN),
            )
        )
        if not sum_path.is_dir() or not pre_path.is_dir():
            raise unittest.SkipTest(
                "Frozen TODO3 upstream campaigns are unavailable"
            )
        cls.sum_campaign = sum_path
        cls.pre_campaign = pre_path
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.source_manifest = cls.root / "source_manifest.json"
        ensure_source_manifest(
            cls.source_manifest,
            sum_campaign=cls.sum_campaign,
            pre_campaign=cls.pre_campaign,
            repo_root=REPO_ROOT,
        )
        cls.config_summary = write_composite_configs(
            output_dir=cls.root,
            source_manifest_path=cls.source_manifest,
            sum_campaign=cls.sum_campaign,
            pre_campaign=cls.pre_campaign,
            repo_root=REPO_ROOT,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "temporary"):
            cls.temporary.cleanup()

    def test_dual_selection_manifest_proves_91_shared_sources(self):
        document = verify_source_manifest(
            self.source_manifest,
            sum_campaign=self.sum_campaign,
            pre_campaign=self.pre_campaign,
            repo_root=REPO_ROOT,
        )
        self.assertEqual(document["upstream"]["shared_source_files"], 91)
        self.assertEqual(
            set(document["upstream"]["fixed_split_fingerprints"]),
            {
                "ego-facebook",
                "oklahoma",
                "unc",
                "pokec_n",
                "brightkite",
            },
        )

    def test_composite_contains_new_in_and_old_pre_only(self):
        selected = self.root / "selected_configs.csv"
        with selected.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 10)
        expected = {
            ("ego-facebook", "cafe_in"): ("0.0", "0.1", "0.5"),
            ("oklahoma", "cafe_in"): ("0.0", "0.5", "1.0"),
            ("unc", "cafe_in"): ("0.0", "5.0", "1.0"),
            ("pokec_n", "cafe_in"): ("0.0", "5.0", "1.0"),
            ("brightkite", "cafe_in"): ("1.0", "5.0", "1.0"),
            ("ego-facebook", "cafe_pre"): ("1.0", "0.1", "1.0"),
            ("oklahoma", "cafe_pre"): ("1.0", "0.5", "1.0"),
            ("unc", "cafe_pre"): ("1.0", "2.0", "1.0"),
            ("pokec_n", "cafe_pre"): ("0.01", "5.0", "1.0"),
            ("brightkite", "cafe_pre"): ("2.0", "5.0", "1.0"),
        }
        for row in rows:
            key = (row["dataset"], row["variant"])
            penalty = (
                row["lambda_fair"]
                if row["variant"] == "cafe_in"
                else row["preproc_alpha"]
            )
            self.assertEqual(
                (penalty, row["beta"], row["lambda_mobility"]),
                expected[key],
            )
            self.assertEqual(row["penalty_reduction"], "sum")
            self.assertEqual(
                row["selection_penalty_reduction"],
                "sum" if row["variant"] == "cafe_in" else "mean",
            )
        old_in = next(
            row
            for row in rows
            if row["dataset"] == "pokec_n"
            and row["variant"] == "cafe_in"
        )
        self.assertEqual(old_in["lambda_fair"], "0.0")
        self.assertNotEqual(old_in["beta"], "0.1")

    def test_tampered_composite_is_rejected(self):
        selected = self.root / "selected_configs.csv"
        original = selected.read_bytes()
        try:
            selected.write_bytes(original.replace(b",sum,", b",mean,", 1))
            with self.assertRaisesRegex(
                CompositeConfigError, "differs from exact recomposition"
            ):
                write_composite_configs(
                    output_dir=self.root,
                    source_manifest_path=self.source_manifest,
                    sum_campaign=self.sum_campaign,
                    pre_campaign=self.pre_campaign,
                    repo_root=REPO_ROOT,
                )
        finally:
            selected.write_bytes(original)

    def test_interrupted_composite_write_resumes_only_exact_artifacts(self):
        partial = self.root / "partial_composite"
        write_composite_configs(
            output_dir=partial,
            source_manifest_path=self.source_manifest,
            sum_campaign=self.sum_campaign,
            pre_campaign=self.pre_campaign,
            repo_root=REPO_ROOT,
        )
        (partial / "selected_configs_lineage.json").unlink()
        summary = write_composite_configs(
            output_dir=partial,
            source_manifest_path=self.source_manifest,
            sum_campaign=self.sum_campaign,
            pre_campaign=self.pre_campaign,
            repo_root=REPO_ROOT,
        )
        self.assertEqual(summary["status"], "verified")
        self.assertTrue(
            (partial / "selected_configs_lineage.json").is_file()
        )


if __name__ == "__main__":
    unittest.main()
