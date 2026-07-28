"""Strictly validate and freeze the CAFE-In/sum validation selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

from analysis.egfix_finalize_selection import (
    SelectionValidationError,
    _atomic_write_csv,
    _atomic_write_json,
    _read_json,
    _validate_stats_and_runs,
    pareto_asf_select,
)
from jobs.egfix_sum.protocol import (
    ASF_WEIGHTS,
    BATCH_SIZE,
    BETA_VALUES,
    BRIGHTKITE_TEMPORAL_SPLIT,
    BRIGHTKITE_TEMPORAL_SPLIT_SHA256,
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
    protocol_document,
    selection_task,
)
from jobs.egfix_sum.provenance import (
    canonical_json_bytes,
    sha256_file,
    verify_source_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

CANDIDATES_FILENAME = "cafe_in_selection_candidates.csv"
RUNS_FILENAME = "cafe_in_selection_runs.csv"
SELECTED_CSV_FILENAME = "selected_cafe_in_configs.csv"
SELECTED_JSON_FILENAME = "selected_cafe_in_configs.json"
INPUT_MANIFEST_FILENAME = "cafe_in_selection_input_manifest.json"
RUNTIME_ENV_FILENAME = "hl_gnn.conda-explicit.txt"
RUNTIME_ENV_DIGEST_FILENAME = "hl_gnn.conda-explicit.sha256"
OUTPUT_FILENAMES: Tuple[str, ...] = (
    CANDIDATES_FILENAME,
    RUNS_FILENAME,
    SELECTED_CSV_FILENAME,
    SELECTED_JSON_FILENAME,
    INPUT_MANIFEST_FILENAME,
)

SELECTED_COLUMNS: Tuple[str, ...] = (
    "dataset",
    "variant",
    "penalization_mode",
    "lambda_fair",
    "preproc_alpha",
    "beta",
    "beta_ra",
    "lambda_mobility",
    "config_key",
    "Precision@10_mean",
    "Equality_Gain@10_mean",
    "Pagerank_EG@10_mean",
    "Closeness_EG@10_mean",
    "asf_score",
    "source_manifest_sha256",
    "source_manifest_content_sha256",
    "runtime_environment_sha256",
    "selection_input_manifest_sha256",
    "selection_profile",
    "penalty_reduction",
    "loss_func",
    "num_neg",
    "batch_size",
    "evaluation_split",
    "eg_graph_semantics",
    "selection_runs",
)


def _read_task_env(path: Path) -> Mapping[str, str]:
    values: Dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SelectionValidationError(
            f"Cannot read task provenance {path}: {exc}"
        ) from exc
    for line_number, line in enumerate(lines, start=1):
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise SelectionValidationError(
                f"{path}:{line_number}: expected key=value"
            )
        key, value = line.split("=", 1)
        if not key or key in values:
            raise SelectionValidationError(
                f"{path}:{line_number}: duplicate or empty key {key!r}"
            )
        values[key] = value
    return values


def _runtime_environment_digest(source_manifest_path: Path) -> str:
    campaign_root = source_manifest_path.resolve().parent
    snapshot_path = campaign_root / RUNTIME_ENV_FILENAME
    digest_path = campaign_root / RUNTIME_ENV_DIGEST_FILENAME
    if not snapshot_path.is_file() or not digest_path.is_file():
        raise SelectionValidationError(
            "Runtime-environment attestation is incomplete"
        )
    fields = digest_path.read_text(encoding="utf-8").strip().split()
    actual = sha256_file(snapshot_path)
    if (
        len(fields) != 2
        or fields[1] != RUNTIME_ENV_FILENAME
        or fields[0] != actual
    ):
        raise SelectionValidationError(
            f"{digest_path}: runtime-environment digest mismatch"
        )
    return actual


def _read_command_args(path: Path) -> Mapping[str, str]:
    try:
        tokens = shlex.split(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SelectionValidationError(
            f"Cannot parse recorded command {path}: {exc}"
        ) from exc
    if tokens[:3] != ["python", "-u", "experiments/run_cafe.py"]:
        raise SelectionValidationError(
            f"{path}: unexpected command prefix {tokens[:3]!r}"
        )
    arguments = tokens[3:]
    if len(arguments) % 2:
        raise SelectionValidationError(
            f"{path}: runner arguments must be --flag value pairs"
        )
    parsed: Dict[str, str] = {}
    for index in range(0, len(arguments), 2):
        flag, value = arguments[index : index + 2]
        if not flag.startswith("--") or flag in parsed:
            raise SelectionValidationError(
                f"{path}: invalid or duplicate command flag {flag!r}"
            )
        parsed[flag] = value
    return parsed


def _require_task_provenance(
    result_path: Path,
    *,
    dataset: str,
    penalty: float,
    source_manifest_file_sha256: str,
    runtime_environment_sha256: str,
    source_manifest_path: Path,
    repo_root: Path,
    split_relative_path: str,
) -> Tuple[Path, Path, Path, int]:
    task_dir = result_path.parent
    task_env_path = task_dir / "task.env"
    command_path = task_dir / "command.sh"
    result_digest_path = task_dir / "results.sha256"
    success_path = task_dir / "_SUCCESS"
    for required_path in (
        task_env_path,
        command_path,
        result_digest_path,
        success_path,
    ):
        if not required_path.is_file():
            raise SelectionValidationError(
                f"{result_path}: missing task artifact {required_path.name}"
            )
    task_env = _read_task_env(task_env_path)
    try:
        task_id = int(task_env.get("array_task_id", ""))
        expected = selection_task(task_id)
    except (ValueError, TypeError) as exc:
        raise SelectionValidationError(
            f"{task_env_path}: invalid array_task_id"
        ) from exc
    required = {
        "dataset": dataset,
        "method": VARIANT,
        "penalization_mode": PENALIZATION_MODE,
        "penalty_name": PENALTY_NAME,
        "penalty_value": str(float(penalty)),
        "evaluation_split": EVALUATION_SPLIT,
        "penalty_reduction": PENALTY_REDUCTION,
        "loss_func": LOSS_FUNC,
        "num_neg": str(NUM_NEG),
        "batch_size": str(BATCH_SIZE),
        "source_manifest_sha256": source_manifest_file_sha256,
        "runtime_environment_sha256": runtime_environment_sha256,
    }
    for field, expected_value in required.items():
        actual = task_env.get(field)
        if actual != expected_value:
            raise SelectionValidationError(
                f"{task_env_path}: {field} must equal "
                f"{expected_value!r}, got {actual!r}"
            )
    if (
        expected["dataset"] != dataset
        or expected["variant"] != VARIANT
        or not math.isclose(
            float(expected["penalty_value"]),
            penalty,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise SelectionValidationError(
            f"{task_env_path}: task {task_id} does not map to "
            f"{dataset}/{penalty}"
        )
    large_dataset = dataset in {"pokec_n", "brightkite"}
    expected_command = {
        "--data_name": dataset,
        "--data_path": str((repo_root / "data/SNAP").resolve()),
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
        "--beta_ra": ",".join(str(value) for value in BETA_VALUES),
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
        "--closeness_sample_size": str(CLOSENESS_SAMPLE_SIZE),
        "--closeness_epsilon": str(CLOSENESS_EPSILON),
        "--source_manifest": str(source_manifest_path.resolve()),
        "--results_file": str(result_path.resolve()),
    }
    if large_dataset:
        expected_command.update(
            {
                "--eval_skip_test": "True",
                "--eval_max_nodes": "5000",
            }
        )
    if dataset == "brightkite":
        expected_command["--split_file"] = str(
            (repo_root / split_relative_path).resolve()
        )
    actual_command = _read_command_args(command_path)
    if actual_command != expected_command:
        missing = sorted(set(expected_command) - set(actual_command))
        extra = sorted(set(actual_command) - set(expected_command))
        mismatched = sorted(
            flag
            for flag in set(expected_command) & set(actual_command)
            if expected_command[flag] != actual_command[flag]
        )
        raise SelectionValidationError(
            f"{command_path}: command does not match the frozen protocol; "
            f"missing={missing}, extra={extra}, mismatched={mismatched}"
        )
    digest_fields = result_digest_path.read_text(
        encoding="utf-8"
    ).strip().split()
    if (
        len(digest_fields) != 2
        or digest_fields[1] != "results.json"
        or digest_fields[0] != sha256_file(result_path)
    ):
        raise SelectionValidationError(
            f"{result_digest_path}: results.json digest mismatch"
        )
    return task_env_path, command_path, result_digest_path, task_id


def _require_metadata(
    info: Mapping[str, object],
    *,
    path: Path,
    source_manifest_file_sha256: str,
    expected_brightkite_split_sha256: str,
) -> str:
    dataset = info.get("dataset")
    if dataset not in DATASETS:
        raise SelectionValidationError(
            f"{path}: dataset must be one of {DATASETS}, got {dataset!r}"
        )
    required = {
        "encoder": "HLGNN",
        "predictor": "MLP",
        "penalization_mode": PENALIZATION_MODE,
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
        "k_values": list(K_VALUES),
        "use_demographics": USE_DEMOGRAPHICS,
        "centrality_types": ["pagerank", "closeness"],
        "closeness_random_seed": CLOSENESS_RANDOM_SEED,
        "closeness_sample_size": CLOSENESS_SAMPLE_SIZE,
        "closeness_epsilon": CLOSENESS_EPSILON,
        "source_manifest_sha256": source_manifest_file_sha256,
        "model_seeds": [
            SEED_BASE + index for index in range(SELECTION_RUNS)
        ],
        "use_mobility_boost": True,
    }
    for field, expected in required.items():
        actual = info.get(field)
        if actual != expected:
            raise SelectionValidationError(
                f"{path}: experiment_info.{field} must equal "
                f"{expected!r}, got {actual!r}"
            )

    split_fingerprint = info.get("split_fingerprint")
    if not (
        isinstance(split_fingerprint, str)
        and len(split_fingerprint) == 64
        and all(character in "0123456789abcdef" for character in split_fingerprint)
    ):
        raise SelectionValidationError(
            f"{path}: experiment_info.split_fingerprint must be a SHA256"
        )
    split_sha256 = info.get("split_sha256")
    if dataset == "brightkite":
        if split_sha256 != expected_brightkite_split_sha256:
            raise SelectionValidationError(
                f"{path}: Brightkite split SHA256 must be "
                f"{expected_brightkite_split_sha256}, got {split_sha256!r}"
            )
    elif split_sha256 not in (None, ""):
        raise SelectionValidationError(
            f"{path}: non-Brightkite split_sha256 must be null"
        )
    return str(dataset)


def _result_paths(selection_root: Path) -> List[Path]:
    partials = sorted(selection_root.rglob("*.partial.json"))
    if partials:
        raise SelectionValidationError(
            "Partial selection artifacts are present; refusing to finalize: "
            + ", ".join(str(path) for path in partials[:5])
        )
    paths = sorted(selection_root.rglob("results.json"))
    if not paths:
        raise SelectionValidationError(
            f"No results.json artifacts found below {selection_root}"
        )
    return paths


def finalize_selection(
    *,
    selection_root: Path,
    source_manifest_path: Path,
    output_dir: Path,
    repo_root: Path = REPO_ROOT,
    manifest_source_paths: Sequence[str] | None = None,
    manifest_source_globs: Sequence[str] | None = None,
    manifest_data_artifact_paths: Mapping[str, str] | None = None,
    split_relative_path: str = BRIGHTKITE_TEMPORAL_SPLIT,
    expected_brightkite_split_sha256: str = (
        BRIGHTKITE_TEMPORAL_SPLIT_SHA256
    ),
) -> Dict[str, object]:
    selection_root = selection_root.resolve()
    output_dir = output_dir.resolve()
    if not selection_root.is_dir():
        raise SelectionValidationError(
            f"Selection root is not a directory: {selection_root}"
        )
    try:
        output_dir.relative_to(selection_root)
    except ValueError:
        pass
    else:
        raise SelectionValidationError(
            "Output directory must not be inside the selection input tree"
        )

    try:
        source_manifest = verify_source_manifest(
            source_manifest_path,
            repo_root,
            source_paths=manifest_source_paths,
            source_globs=manifest_source_globs,
            data_artifact_paths=manifest_data_artifact_paths,
            split_relative_path=split_relative_path,
            expected_split_sha256=expected_brightkite_split_sha256,
        )
    except Exception as exc:
        raise SelectionValidationError(
            f"Source-manifest verification failed: {exc}"
        ) from exc
    source_manifest_file_sha256 = sha256_file(
        source_manifest_path.resolve()
    )
    source_manifest_content_sha256 = str(
        source_manifest["manifest_sha256"]
    )
    runtime_environment_sha256 = _runtime_environment_digest(
        source_manifest_path
    )

    result_paths = _result_paths(selection_root)
    if len(result_paths) != SELECTION_TASK_COUNT:
        raise SelectionValidationError(
            f"Expected exactly {SELECTION_TASK_COUNT} isolated CAFE-In "
            f"selection artifacts, found {len(result_paths)}"
        )

    dataset_order = {dataset: index for index, dataset in enumerate(DATASETS)}
    config_order = {
        key: index for index, key in enumerate(expected_config_keys())
    }
    candidates: List[Dict[str, object]] = []
    run_output: List[Dict[str, object]] = []
    input_records: List[Dict[str, object]] = []
    seen_arms = set()
    seen_configs = set()
    seen_task_ids = set()
    split_fingerprints: Dict[str, set] = {
        dataset: set() for dataset in DATASETS
    }

    for path in result_paths:
        data = _read_json(path)
        info = data.get("experiment_info")
        if not isinstance(info, dict):
            raise SelectionValidationError(
                f"{path}: experiment_info must be an object"
            )
        dataset = _require_metadata(
            info,
            path=path,
            source_manifest_file_sha256=source_manifest_file_sha256,
            expected_brightkite_split_sha256=(
                expected_brightkite_split_sha256
            ),
        )
        split_fingerprints[dataset].add(info["split_fingerprint"])

        lambda_results = data.get("lambda_results")
        if not isinstance(lambda_results, dict):
            raise SelectionValidationError(
                f"{path}: lambda_results must be an object"
            )
        if len(lambda_results) != CONFIGS_PER_PENALTY:
            raise SelectionValidationError(
                f"{path}: isolated task must contain exactly "
                f"{CONFIGS_PER_PENALTY} configurations, found "
                f"{len(lambda_results)}"
            )

        parsed_configs = {}
        for config_key in lambda_results:
            try:
                parsed_configs[config_key] = parse_config_key(config_key)
            except ValueError as exc:
                raise SelectionValidationError(f"{path}: {exc}") from exc
        penalties = {
            float(parsed["penalty_value"])
            for parsed in parsed_configs.values()
        }
        if len(penalties) != 1:
            raise SelectionValidationError(
                f"{path}: isolated artifact spans penalties {penalties}"
            )
        penalty = next(iter(penalties))
        expected_arm_keys = set(expected_penalty_config_keys(penalty))
        if set(lambda_results) != expected_arm_keys:
            missing = sorted(expected_arm_keys - set(lambda_results))
            extra = sorted(set(lambda_results) - expected_arm_keys)
            raise SelectionValidationError(
                f"{path}: incomplete penalty arm; missing={missing[:3]}, "
                f"extra={extra[:3]}"
            )
        expected_grid = {
            PENALTY_NAME: [float(penalty)],
            "beta": list(BETA_VALUES),
            "lambda_mobility": list(MOBILITY_VALUES),
        }
        if info.get("grid") != expected_grid:
            raise SelectionValidationError(
                f"{path}: experiment_info.grid must equal {expected_grid!r}, "
                f"got {info.get('grid')!r}"
            )

        (
            task_env_path,
            command_path,
            result_digest_path,
            task_id,
        ) = _require_task_provenance(
            path,
            dataset=dataset,
            penalty=penalty,
            source_manifest_file_sha256=source_manifest_file_sha256,
            runtime_environment_sha256=runtime_environment_sha256,
            source_manifest_path=source_manifest_path,
            repo_root=repo_root.resolve(),
            split_relative_path=split_relative_path,
        )
        if task_id in seen_task_ids:
            raise SelectionValidationError(
                f"Duplicate array task identity {task_id}"
            )
        seen_task_ids.add(task_id)

        arm = (dataset, penalty)
        if arm in seen_arms:
            raise SelectionValidationError(
                f"Duplicate isolated selection arm {arm}"
            )
        seen_arms.add(arm)

        individual_runs = data.get("individual_runs")
        if (
            not isinstance(individual_runs, list)
            or len(individual_runs) != SELECTION_RUNS
            or not all(isinstance(run, dict) for run in individual_runs)
        ):
            raise SelectionValidationError(
                f"{path}: individual_runs must contain exactly "
                f"{SELECTION_RUNS} objects"
            )
        expected_run_keys = expected_arm_keys | {"baseline"}
        for run_index, run in enumerate(individual_runs):
            if set(run) != expected_run_keys:
                missing = sorted(expected_run_keys - set(run))
                extra = sorted(set(run) - expected_run_keys)
                raise SelectionValidationError(
                    f"{path}: run {run_index} configuration set differs; "
                    f"missing={missing[:3]}, extra={extra[:3]}"
                )

        relative_path = path.relative_to(selection_root).as_posix()
        artifact_sha256 = sha256_file(path)
        input_records.append(
            {
                "path": relative_path,
                "sha256": artifact_sha256,
                "task_env_path": task_env_path.relative_to(
                    selection_root
                ).as_posix(),
                "task_env_sha256": sha256_file(task_env_path),
                "command_path": command_path.relative_to(
                    selection_root
                ).as_posix(),
                "command_sha256": sha256_file(command_path),
                "result_digest_path": result_digest_path.relative_to(
                    selection_root
                ).as_posix(),
                "result_digest_sha256": sha256_file(result_digest_path),
                "array_task_id": task_id,
                "dataset": dataset,
                "variant": VARIANT,
                "penalty_name": PENALTY_NAME,
                "penalty_value": penalty,
                "split_fingerprint": info["split_fingerprint"],
            }
        )

        for config_key in expected_penalty_config_keys(penalty):
            identity = (dataset, config_key)
            if identity in seen_configs:
                raise SelectionValidationError(
                    f"Duplicate configuration {identity}"
                )
            seen_configs.add(identity)
            parsed = parsed_configs[config_key]
            result = lambda_results[config_key]
            if not isinstance(result, dict):
                raise SelectionValidationError(
                    f"{path}: result for {config_key} must be an object"
                )
            summary, per_run = _validate_stats_and_runs(
                path=path,
                config_key=config_key,
                result=result,
                individual_runs=individual_runs,
            )
            candidate: Dict[str, object] = {
                "dataset": dataset,
                "variant": VARIANT,
                "penalization_mode": PENALIZATION_MODE,
                "penalty_name": PENALTY_NAME,
                "penalty_value": parsed["penalty_value"],
                "lambda_fair": parsed["lambda_fair"],
                "preproc_alpha": None,
                "beta": parsed["beta_ra"],
                "beta_ra": parsed["beta_ra"],
                "lambda_mobility": parsed["lambda_mobility"],
                "config_key": config_key,
                "source_file": relative_path,
                "source_file_sha256": artifact_sha256,
                "source_manifest_sha256": source_manifest_file_sha256,
                "source_manifest_content_sha256": (
                    source_manifest_content_sha256
                ),
                "runtime_environment_sha256": (
                    runtime_environment_sha256
                ),
                "selection_profile": PROFILE,
                "penalty_reduction": PENALTY_REDUCTION,
                "loss_func": LOSS_FUNC,
                "num_neg": NUM_NEG,
                "batch_size": BATCH_SIZE,
                "evaluation_split": EVALUATION_SPLIT,
                "eg_graph_semantics": EG_GRAPH_SEMANTICS,
                "selection_runs": SELECTION_RUNS,
            }
            candidate.update(summary)
            candidates.append(candidate)

            for run_index, run_metrics in enumerate(per_run):
                run_row: Dict[str, object] = {
                    "dataset": dataset,
                    "variant": VARIANT,
                    "penalization_mode": PENALIZATION_MODE,
                    "run_index": run_index,
                    "model_seed": SEED_BASE + run_index,
                    "lambda_fair": parsed["lambda_fair"],
                    "preproc_alpha": None,
                    "beta": parsed["beta_ra"],
                    "beta_ra": parsed["beta_ra"],
                    "lambda_mobility": parsed["lambda_mobility"],
                    "config_key": config_key,
                    "source_file": relative_path,
                }
                run_row.update(run_metrics)
                run_output.append(run_row)

    expected_arms = {
        (dataset, float(penalty))
        for dataset in DATASETS
        for penalty in LAMBDA_FAIR_VALUES
    }
    if seen_arms != expected_arms:
        raise SelectionValidationError(
            "Selection arm set is incomplete; "
            f"missing={sorted(expected_arms - seen_arms)}, "
            f"extra={sorted(seen_arms - expected_arms)}"
        )
    if seen_task_ids != set(range(SELECTION_TASK_COUNT)):
        raise SelectionValidationError(
            "Array task IDs are incomplete or out of range"
        )
    expected_identities = {
        (dataset, config_key)
        for dataset in DATASETS
        for config_key in expected_config_keys()
    }
    if seen_configs != expected_identities:
        raise SelectionValidationError(
            "The combined grid is not exactly five datasets × "
            f"{CONFIGS_PER_DATASET} CAFE-In configurations"
        )
    for dataset, fingerprints in split_fingerprints.items():
        if len(fingerprints) != 1:
            raise SelectionValidationError(
                f"{dataset}: selection arms used multiple split fingerprints: "
                f"{sorted(fingerprints)}"
            )

    candidates.sort(
        key=lambda row: (
            dataset_order[str(row["dataset"])],
            config_order[str(row["config_key"])],
        )
    )
    run_output.sort(
        key=lambda row: (
            dataset_order[str(row["dataset"])],
            config_order[str(row["config_key"])],
            int(row["run_index"]),
        )
    )
    input_records.sort(key=lambda row: int(row["array_task_id"]))

    selected: List[MutableMapping[str, object]] = []
    for dataset in DATASETS:
        group = [
            row for row in candidates if row["dataset"] == dataset
        ]
        if len(group) != CONFIGS_PER_DATASET:
            raise SelectionValidationError(
                f"{dataset}: expected {CONFIGS_PER_DATASET} candidates, "
                f"found {len(group)}"
            )
        selected.append(pareto_asf_select(group))

    if (
        len(candidates) != EXPECTED_CANDIDATE_COUNT
        or len(run_output) != EXPECTED_RUN_ROW_COUNT
        or len(selected) != EXPECTED_SELECTION_COUNT
    ):
        raise SelectionValidationError(
            "Validated output counts do not match the frozen protocol"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_columns = (
        "dataset",
        "variant",
        "penalization_mode",
        "penalty_name",
        "penalty_value",
        "lambda_fair",
        "preproc_alpha",
        "beta",
        "beta_ra",
        "lambda_mobility",
        "config_key",
    ) + tuple(
        column
        for objective in OBJECTIVES
        for column in (f"{objective}_mean", f"{objective}_std")
    ) + (
        "is_pareto",
        "asf_score",
        "selected",
        "source_file",
        "source_file_sha256",
        "source_manifest_sha256",
        "source_manifest_content_sha256",
        "runtime_environment_sha256",
        "selection_profile",
        "penalty_reduction",
        "loss_func",
        "num_neg",
        "batch_size",
        "evaluation_split",
        "eg_graph_semantics",
        "selection_runs",
    )
    run_columns = (
        "dataset",
        "variant",
        "penalization_mode",
        "run_index",
        "model_seed",
        "lambda_fair",
        "preproc_alpha",
        "beta",
        "beta_ra",
        "lambda_mobility",
        "config_key",
    ) + OBJECTIVES + ("source_file",)

    input_manifest = {
        "schema_version": 1,
        "manifest_kind": "cafe_in_sum_selection_inputs",
        "protocol": protocol_document(),
        "source_manifest_path": str(source_manifest_path.resolve()),
        "source_manifest_sha256": source_manifest_file_sha256,
        "source_manifest_content_sha256": source_manifest_content_sha256,
        "runtime_environment": {
            "path": str(
                (source_manifest_path.resolve().parent / RUNTIME_ENV_FILENAME)
            ),
            "sha256": runtime_environment_sha256,
        },
        "artifacts": input_records,
    }
    input_manifest["input_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(input_manifest)
    ).hexdigest()
    for row in selected:
        row["selection_input_manifest_sha256"] = input_manifest[
            "input_manifest_sha256"
        ]
    selected_rows = [
        {column: row.get(column) for column in SELECTED_COLUMNS}
        for row in selected
    ]
    selected_json = {
        "schema_version": 1,
        "profile": PROFILE,
        "selection_rule": {
            "objectives": list(OBJECTIVES),
            "weights": list(ASF_WEIGHTS),
            "direction": "maximize",
            "tie_break": "lexicographic config_key",
        },
        "source_manifest_sha256": source_manifest_file_sha256,
        "source_manifest_content_sha256": source_manifest_content_sha256,
        "selection_input_manifest_sha256": input_manifest[
            "input_manifest_sha256"
        ],
        "selections": selected_rows,
        "by_dataset": {
            dataset: next(
                row for row in selected_rows if row["dataset"] == dataset
            )
            for dataset in DATASETS
        },
    }

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".cafe_in_sum_selection.", dir=str(output_dir)
        )
    )
    try:
        _atomic_write_csv(
            staging_dir / CANDIDATES_FILENAME,
            candidates,
            candidate_columns,
        )
        _atomic_write_csv(
            staging_dir / RUNS_FILENAME, run_output, run_columns
        )
        _atomic_write_csv(
            staging_dir / SELECTED_CSV_FILENAME,
            selected_rows,
            SELECTED_COLUMNS,
        )
        _atomic_write_json(
            staging_dir / SELECTED_JSON_FILENAME, selected_json
        )
        _atomic_write_json(
            staging_dir / INPUT_MANIFEST_FILENAME, input_manifest
        )

        # Publishing several root-level files cannot be one filesystem rename.
        # Make the operation safely resumable: an existing final file is
        # accepted only when it is byte-identical to the freshly reconstructed
        # artifact; missing files are atomically moved into place.
        for filename in OUTPUT_FILENAMES:
            staged_path = staging_dir / filename
            final_path = output_dir / filename
            if final_path.exists():
                if (
                    not final_path.is_file()
                    or sha256_file(final_path) != sha256_file(staged_path)
                ):
                    raise SelectionValidationError(
                        "Existing partial selection output does not match "
                        f"the validated reconstruction: {final_path}"
                    )
                staged_path.unlink()
            else:
                os.replace(staged_path, final_path)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    return {
        "selection_artifacts": len(result_paths),
        "candidates": len(candidates),
        "run_rows": len(run_output),
        "selected": len(selected_rows),
        "profile": PROFILE,
        "source_manifest_sha256": source_manifest_file_sha256,
        "output_dir": str(output_dir),
    }


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate 35 CAFE-In/sum validation jobs and freeze one "
            "Pareto+ASF configuration per dataset"
        )
    )
    parser.add_argument(
        "--campaign-root",
        "--campaign_root",
        dest="campaign_root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--source-manifest",
        "--source_manifest",
        dest="source_manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()

    summary = finalize_selection(
        selection_root=args.campaign_root / "selection",
        source_manifest_path=args.source_manifest,
        output_dir=args.campaign_root,
        repo_root=args.repo_root,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    _main()
