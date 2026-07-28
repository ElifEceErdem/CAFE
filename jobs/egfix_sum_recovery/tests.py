"""Static and live-state checks for the CAFE-In/sum recovery chain."""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

from jobs.egfix_sum_recovery.audit_completed import (
    ADOPT_TASK_IDS,
    EXISTING_TASK_IDS,
    RERUN_TASK_IDS,
)
from jobs.egfix_sum_recovery.provenance import (
    ALLOWED_NONEXECUTION_DRIFT_PATHS,
    RECOVERY_REQUIRED_FILES,
    verify_original_execution_state,
    verify_recovery_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class RecoveryStaticTests(unittest.TestCase):
    def test_recovery_task_partition_is_exact(self):
        self.assertEqual(EXISTING_TASK_IDS, tuple(range(23)))
        self.assertEqual(
            ADOPT_TASK_IDS, (13, 17, 18, 19, 20, 21, 22)
        )
        self.assertEqual(RERUN_TASK_IDS, tuple(range(23, 35)))
        self.assertEqual(
            set(EXISTING_TASK_IDS) | set(RERUN_TASK_IDS),
            set(range(35)),
        )
        self.assertFalse(
            set(EXISTING_TASK_IDS) & set(RERUN_TASK_IDS)
        )

    def test_drift_waiver_is_narrow_and_explicit(self):
        self.assertEqual(
            ALLOWED_NONEXECUTION_DRIFT_PATHS,
            (
                "analysis/generate_acceptance_table.py",
                "analysis/generate_closeness_stratum_table.py",
                "analysis/generate_demographic_appendix_tables.py",
            ),
        )

    def test_recovery_sources_and_shell_syntax(self):
        paths = [REPO_ROOT / path for path in RECOVERY_REQUIRED_FILES]
        self.assertTrue(all(path.is_file() for path in paths))
        shell_paths = [
            path for path in paths if path.suffix in {".job", ".sh"}
        ]
        subprocess.run(
            ["bash", "-n", *(str(path) for path in shell_paths)],
            check=True,
        )

    def test_launcher_stops_before_todo2_and_todo3(self):
        selection = (
            REPO_ROOT / "jobs/egfix_sum_recovery/selection.job"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=23-34%10", selection)
        self.assertIn("--penalty_reduction sum", selection)
        self.assertIn('--source_manifest "${SOURCE_MANIFEST}"', selection)
        submit = (
            REPO_ROOT / "jobs/egfix_sum_recovery/submit_recovery.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("submit_reruns.sh", submit)


class RecoveryLiveStateTests(unittest.TestCase):
    def test_live_campaign_when_launched(self):
        run_tag = os.environ.get("EGFIX_SUM_RUN_TAG")
        if not run_tag:
            self.skipTest("EGFIX_SUM_RUN_TAG is not set")
        campaign = REPO_ROOT / "results_egfix" / run_tag
        state = verify_original_execution_state(campaign, REPO_ROOT)
        self.assertGreaterEqual(
            int(state["verified_original_source_files"]), 80
        )
        recovery_manifest = campaign / "recovery_manifest.json"
        if recovery_manifest.is_file():
            verified = verify_recovery_manifest(
                campaign,
                REPO_ROOT,
                manifest_path=recovery_manifest,
            )
            self.assertEqual(
                verified["recovery_plan"]["rerun_task_ids"],
                list(range(23, 35)),
            )


if __name__ == "__main__":
    unittest.main()
