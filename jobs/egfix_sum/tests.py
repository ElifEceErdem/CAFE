"""Focused integration tests for the isolated CAFE-In/sum workflow."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

from analysis.egfix_finalize_selection import SelectionValidationError
from jobs.egfix_sum.finalize_selection import (
    CANDIDATES_FILENAME,
    RUNS_FILENAME,
    RUNTIME_ENV_DIGEST_FILENAME,
    RUNTIME_ENV_FILENAME,
    SELECTED_COLUMNS,
    SELECTED_CSV_FILENAME,
    SELECTED_JSON_FILENAME,
    finalize_selection,
)
from jobs.egfix_sum.protocol import (
    BATCH_SIZE,
    BETA_VALUES,
    CLOSENESS_EPSILON,
    CLOSENESS_RANDOM_SEED,
    CLOSENESS_SAMPLE_SIZE,
    CONFIGS_PER_DATASET,
    CONFIGS_PER_PENALTY,
    DATASETS,
    EG_GRAPH_SEMANTICS,
    EPOCHS,
    EVALUATION_SPLIT,
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_RUN_ROW_COUNT,
    EXPECTED_SELECTION_COUNT,
    K_VALUES,
    LAMBDA_FAIR_VALUES,
    LEARNING_RATE,
    LOSS_FUNC,
    MOBILITY_VALUES,
    NUM_NEG,
    OBJECTIVES,
    PENALIZATION_MODE,
    PENALTY_NAME,
    PENALTY_REDUCTION,
    PROFILE,
    SEED_BASE,
    SEED_RULE,
    SELECTION_RUNS,
    SELECTION_TASK_COUNT,
    SPLIT_SEED,
    USE_DEMOGRAPHICS,
    VARIANT,
    expected_config_keys,
    expected_penalty_config_keys,
    parse_config_key,
    selection_tasks,
)
from jobs.egfix_sum.provenance import (
    ProvenanceError,
    ensure_source_manifest,
    sha256_file,
    verify_source_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class ProtocolTests(unittest.TestCase):
    def test_profile_and_task_matrix_are_exact(self):
        self.assertEqual(PROFILE, "cafe_in_sum_v1")
        self.assertEqual(PENALTY_REDUCTION, "sum")
        self.assertEqual(SELECTION_TASK_COUNT, 35)
        self.assertEqual(CONFIGS_PER_PENALTY, 30)
        self.assertEqual(CONFIGS_PER_DATASET, 210)
        self.assertEqual(EXPECTED_CANDIDATE_COUNT, 1050)
        self.assertEqual(EXPECTED_RUN_ROW_COUNT, 3150)
        self.assertEqual(EXPECTED_SELECTION_COUNT, 5)
        tasks = selection_tasks()
        self.assertEqual(len(tasks), SELECTION_TASK_COUNT)
        self.assertEqual(
            {
                (
                    task["dataset"],
                    task["variant"],
                    task["penalty_value"],
                )
                for task in tasks
            },
            {
                (dataset, VARIANT, float(penalty))
                for dataset in DATASETS
                for penalty in LAMBDA_FAIR_VALUES
            },
        )
        keys = expected_config_keys()
        self.assertEqual(len(keys), CONFIGS_PER_DATASET)
        self.assertEqual(len(set(keys)), CONFIGS_PER_DATASET)
        self.assertTrue(
            all(parse_config_key(key)["variant"] == VARIANT for key in keys)
        )

    def test_job_scripts_are_syntactically_valid_and_pinned(self):
        paths = [
            REPO_ROOT / "jobs/egfix_sum/preflight.job",
            REPO_ROOT / "jobs/egfix_sum/selection.job",
            REPO_ROOT / "jobs/egfix_sum/finalize_selection.job",
            REPO_ROOT / "jobs/egfix_sum/submit_selection.sh",
        ]
        subprocess.run(
            ["bash", "-n", *(str(path) for path in paths)],
            check=True,
        )
        selection = paths[1].read_text(encoding="utf-8")
        self.assertIn("--array=0-34%10", selection)
        self.assertIn("--penalty_reduction sum", selection)
        self.assertIn("--loss_func AUC", selection)
        self.assertIn("--num_neg 1", selection)
        submit = paths[3].read_text(encoding="utf-8")
        self.assertNotIn("submit_reruns.sh", submit)


class ProvenanceTests(unittest.TestCase):
    def test_manifest_detects_source_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.py"
            split = root / "split.pt"
            graph = root / "graph.pt"
            source.write_text("answer = 42\n", encoding="utf-8")
            split.write_bytes(b"fixed split")
            graph.write_bytes(b"fixed processed graph")
            split_sha = sha256_file(split)
            data_paths = {"synthetic": "graph.pt"}
            manifest_path = root / "source_manifest.json"
            created = ensure_source_manifest(
                manifest_path,
                root,
                source_paths=("source.py",),
                source_globs=(),
                data_artifact_paths=data_paths,
                split_relative_path="split.pt",
                expected_split_sha256=split_sha,
            )
            verified = verify_source_manifest(
                manifest_path,
                root,
                source_paths=("source.py",),
                source_globs=(),
                data_artifact_paths=data_paths,
                split_relative_path="split.pt",
                expected_split_sha256=split_sha,
            )
            self.assertEqual(created, verified)

            graph.write_bytes(b"changed processed graph")
            with self.assertRaisesRegex(
                ProvenanceError, "Dataset manifest"
            ):
                verify_source_manifest(
                    manifest_path,
                    root,
                    source_paths=("source.py",),
                    source_globs=(),
                    data_artifact_paths=data_paths,
                    split_relative_path="split.pt",
                    expected_split_sha256=split_sha,
                )
            graph.write_bytes(b"fixed processed graph")

            source.write_text("answer = 43\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ProvenanceError, "no longer matches"
            ):
                verify_source_manifest(
                    manifest_path,
                    root,
                    source_paths=("source.py",),
                    source_globs=(),
                    data_artifact_paths=data_paths,
                    split_relative_path="split.pt",
                    expected_split_sha256=split_sha,
                )


class FinalizerTests(unittest.TestCase):
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
            data_artifact_paths={},
            split_relative_path="split.pt",
            expected_split_sha256=self.split_sha,
        )
        self.source_manifest_file_sha = sha256_file(
            self.source_manifest_path
        )
        runtime_path = self.root / RUNTIME_ENV_FILENAME
        runtime_path.write_text(
            "@EXPLICIT\nsynthetic-package\n", encoding="utf-8"
        )
        self.runtime_environment_sha = sha256_file(runtime_path)
        (self.root / RUNTIME_ENV_DIGEST_FILENAME).write_text(
            f"{self.runtime_environment_sha}  {RUNTIME_ENV_FILENAME}\n",
            encoding="utf-8",
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

    def _metadata(self, dataset, penalty):
        return {
            "dataset": dataset,
            "encoder": "HLGNN",
            "predictor": "MLP",
            "penalization_mode": PENALIZATION_MODE,
            "use_mobility_boost": True,
            "selection_mode": True,
            "evaluation_split": EVALUATION_SPLIT,
            "metric_split": EVALUATION_SPLIT,
            "test_metrics_evaluated": False,
            "skip_final_comprehensive_eval": True,
            "eg_graph_semantics": EG_GRAPH_SEMANTICS,
            "penalty_reduction": PENALTY_REDUCTION,
            "num_neg": NUM_NEG,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LEARNING_RATE,
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
                PENALTY_NAME: [float(penalty)],
                "beta": list(BETA_VALUES),
                "lambda_mobility": list(MOBILITY_VALUES),
            },
        }

    def _write_complete_grid(self):
        config_indices = {
            key: index for index, key in enumerate(expected_config_keys())
        }
        task_lookup = {
            (task["dataset"], float(task["penalty_value"])): int(
                task["task_index"]
            )
            for task in selection_tasks()
        }
        for dataset in DATASETS:
            for penalty in LAMBDA_FAIR_VALUES:
                task_id = task_lookup[(dataset, float(penalty))]
                keys = expected_penalty_config_keys(penalty)
                individual_runs = [
                    {"baseline": {}} for _ in range(SELECTION_RUNS)
                ]
                lambda_results = {}
                for key in keys:
                    base = (config_indices[key] + 1) / CONFIGS_PER_DATASET
                    values = [base - 0.001, base, base + 0.001]
                    lambda_results[key] = {
                        "statistics": {
                            objective: self._statistics(values)
                            for objective in OBJECTIVES
                        }
                    }
                    for run_index in range(SELECTION_RUNS):
                        individual_runs[run_index][key] = {
                            objective: values[run_index]
                            for objective in OBJECTIVES
                        }
                task_dir = (
                    self.selection
                    / dataset
                    / VARIANT
                    / f"task_{task_id}"
                )
                task_dir.mkdir(parents=True)
                results_path = task_dir / "results.json"
                results_path.write_text(
                    json.dumps(
                        {
                            "experiment_info": self._metadata(
                                dataset, penalty
                            ),
                            "lambda_results": lambda_results,
                            "individual_runs": individual_runs,
                        },
                        allow_nan=False,
                    ),
                    encoding="utf-8",
                )
                (task_dir / "task.env").write_text(
                    "\n".join(
                        (
                            "run_tag=synthetic",
                            "slurm_job_id=synthetic",
                            f"array_task_id={task_id}",
                            f"dataset={dataset}",
                            f"method={VARIANT}",
                            f"penalization_mode={PENALIZATION_MODE}",
                            f"penalty_name={PENALTY_NAME}",
                            f"penalty_value={float(penalty)}",
                            f"evaluation_split={EVALUATION_SPLIT}",
                            f"penalty_reduction={PENALTY_REDUCTION}",
                            f"loss_func={LOSS_FUNC}",
                            f"num_neg={NUM_NEG}",
                            f"batch_size={BATCH_SIZE}",
                            "source_manifest_sha256="
                            f"{self.source_manifest_file_sha}",
                            "runtime_environment_sha256="
                            f"{self.runtime_environment_sha}",
                        )
                    )
                    + "\n",
                    encoding="utf-8",
                )
                large_dataset = dataset in {"pokec_n", "brightkite"}
                command_args = {
                    "--data_name": dataset,
                    "--data_path": str(
                        (self.repo / "data/SNAP").resolve()
                    ),
                    "--encoder": "HLGNN",
                    "--predictor": "MLP",
                    "--epochs": str(EPOCHS),
                    "--eval_steps": "25" if large_dataset else "10",
                    "--patience": "6" if large_dataset else "50",
                    "--batch_size": str(BATCH_SIZE),
                    "--lr": str(LEARNING_RATE),
                    "--loss_func": LOSS_FUNC,
                    "--num_neg": str(NUM_NEG),
                    "--runs": str(SELECTION_RUNS),
                    "--seed": str(SEED_BASE),
                    "--penalization_mode": PENALIZATION_MODE,
                    "--use_mobility_boost": "True",
                    "--lambda_fair": str(float(penalty)),
                    "--beta_ra": ",".join(
                        str(value) for value in BETA_VALUES
                    ),
                    "--lambda_mobility": ",".join(
                        str(value) for value in MOBILITY_VALUES
                    ),
                    "--coreness_alpha": "1.0",
                    "--penalty_reduction": PENALTY_REDUCTION,
                    "--reset_seed_per_model": "True",
                    "--evaluation_split": EVALUATION_SPLIT,
                    "--selection_mode": "True",
                    "--skip_final_comprehensive_eval": "True",
                    "--k_values": "10",
                    "--use_demographics": "False",
                    "--use_equality_gain": "True",
                    "--use_centrality_gain": "True",
                    "--centrality_types": "pagerank,closeness",
                    "--closeness_seed": str(CLOSENESS_RANDOM_SEED),
                    "--closeness_sample_size": str(
                        CLOSENESS_SAMPLE_SIZE
                    ),
                    "--closeness_epsilon": str(CLOSENESS_EPSILON),
                    "--source_manifest": str(
                        self.source_manifest_path.resolve()
                    ),
                    "--results_file": str(results_path.resolve()),
                }
                if large_dataset:
                    command_args.update(
                        {
                            "--eval_skip_test": "True",
                            "--eval_max_nodes": "5000",
                        }
                    )
                if dataset == "brightkite":
                    command_args["--split_file"] = str(
                        (self.repo / "split.pt").resolve()
                    )
                command_tokens = [
                    "python",
                    "-u",
                    "experiments/run_cafe.py",
                ]
                for flag, value in command_args.items():
                    command_tokens.extend((flag, value))
                (task_dir / "command.sh").write_text(
                    shlex.join(command_tokens) + "\n",
                    encoding="utf-8",
                )
                (task_dir / "results.sha256").write_text(
                    f"{sha256_file(results_path)}  results.json\n",
                    encoding="utf-8",
                )
                (task_dir / "_SUCCESS").touch()

    def _finalize(self, output):
        return finalize_selection(
            selection_root=self.selection,
            source_manifest_path=self.source_manifest_path,
            output_dir=output,
            repo_root=self.repo,
            manifest_source_paths=("source.py",),
            manifest_source_globs=(),
            manifest_data_artifact_paths={},
            split_relative_path="split.pt",
            expected_brightkite_split_sha256=self.split_sha,
        )

    def test_complete_grid_is_selected_and_exported(self):
        summary = self._finalize(self.output)
        self.assertEqual(summary["selection_artifacts"], 35)
        self.assertEqual(summary["candidates"], 1050)
        self.assertEqual(summary["run_rows"], 3150)
        self.assertEqual(summary["selected"], 5)
        with (self.output / SELECTED_CSV_FILENAME).open(
            newline="", encoding="utf-8"
        ) as handle:
            reader = csv.DictReader(handle)
            selected = list(reader)
            self.assertEqual(tuple(reader.fieldnames), SELECTED_COLUMNS)
        self.assertEqual(len(selected), 5)
        self.assertTrue(
            all(row["penalty_reduction"] == "sum" for row in selected)
        )
        with (self.output / CANDIDATES_FILENAME).open(
            newline="", encoding="utf-8"
        ) as handle:
            candidates = list(csv.DictReader(handle))
        self.assertEqual(len(candidates), 1050)
        self.assertEqual(
            sum(row["selected"] == "True" for row in candidates), 5
        )
        with (self.output / RUNS_FILENAME).open(
            newline="", encoding="utf-8"
        ) as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 3150)
        (self.output / SELECTED_JSON_FILENAME).unlink()
        resumed = self._finalize(self.output)
        self.assertEqual(resumed["selected"], 5)
        self.assertTrue((self.output / SELECTED_JSON_FILENAME).is_file())

    def test_mean_reduction_is_rejected(self):
        path = next(self.selection.rglob("results.json"))
        document = json.loads(path.read_text(encoding="utf-8"))
        document["experiment_info"]["penalty_reduction"] = "mean"
        path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(
            SelectionValidationError, "penalty_reduction"
        ):
            self._finalize(self.root / "rejected")

    def test_wrong_loss_command_is_rejected(self):
        path = next(self.selection.rglob("command.sh"))
        command = path.read_text(encoding="utf-8")
        path.write_text(
            command.replace("--loss_func AUC", "--loss_func BCE"),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            SelectionValidationError, "command does not match"
        ):
            self._finalize(self.root / "rejected")


if __name__ == "__main__":
    unittest.main()
