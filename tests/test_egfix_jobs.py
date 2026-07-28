"""Static integration checks for the corrected-EG Slurm campaign."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
JOB_ROOT = REPO_ROOT / "jobs" / "egfix"


class EgfixJobWiringTests(unittest.TestCase):
    def _read(self, name: str) -> str:
        return (JOB_ROOT / name).read_text(encoding="utf-8")

    def test_all_shell_entrypoints_parse(self):
        paths = sorted(JOB_ROOT.glob("*.job")) + sorted(
            JOB_ROOT.glob("*.sh")
        )
        self.assertEqual(len(paths), 8)
        for path in paths:
            with self.subTest(path=path.name):
                subprocess.run(
                    ["bash", "-n", str(path)],
                    cwd=REPO_ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )

    def test_every_stage_uses_immutable_json_manifest(self):
        names = (
            "preflight.job",
            "selection.job",
            "finalize_selection.job",
            "aggregate.job",
            "stratified.job",
            "finalize_reruns.job",
            "submit_selection.sh",
            "submit_reruns.sh",
        )
        for name in names:
            text = self._read(name)
            with self.subTest(name=name):
                self.assertIn("source_manifest.json", text)
                self.assertNotIn("source_manifest.sha256", text)

        for name in names:
            if name == "submit_selection.sh":
                continue
            with self.subTest(verification=name):
                self.assertIn("--verify-only", self._read(name))

    def test_selection_and_rerun_phases_are_separate(self):
        selection = self._read("submit_selection.sh")
        reruns = self._read("submit_reruns.sh")
        self.assertNotIn('${SCRIPT_DIR}/aggregate.job', selection)
        self.assertNotIn('${SCRIPT_DIR}/stratified.job', selection)
        self.assertIn("--confirm-selected-configs", reruns)
        self.assertIn('${SCRIPT_DIR}/aggregate.job', reruns)
        self.assertIn('${SCRIPT_DIR}/stratified.job', reruns)

    def test_array_protocols_are_pinned(self):
        selection = self._read("selection.job")
        aggregate = self._read("aggregate.job")
        stratified = self._read("stratified.job")
        self.assertIn("#SBATCH --array=0-69%10", selection)
        self.assertIn("experiments/egfix_protocol.py", selection)
        self.assertIn("--evaluation_split valid", selection)
        self.assertIn("--selection_mode True", selection)
        self.assertIn("--skip_final_comprehensive_eval True", selection)
        self.assertIn("#SBATCH --array=0-49%10", aggregate)
        self.assertIn("--protocol headline", aggregate)
        self.assertIn("--vary_split False", aggregate)
        self.assertIn("#SBATCH --array=0-14%5", stratified)
        self.assertIn("--save_recommendations True", stratified)


if __name__ == "__main__":
    unittest.main()
