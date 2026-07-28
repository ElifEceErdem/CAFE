"""Validate and freeze the corrected-EG CAFE configuration selection.

The finalizer is intentionally strict.  It accepts exactly one isolated result
artifact for every dataset × CAFE variant × outer-penalty value, checks that
all 210 configurations and all three validation runs are present, then applies
the paper's deterministic Pareto+ASF rule.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.egfix_preflight import (  # noqa: E402
    canonical_json_bytes,
    sha256_file,
    verify_source_manifest,
)
from experiments.egfix_protocol import (  # noqa: E402
    ASF_WEIGHTS,
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
    parse_config_key,
    protocol_document,
    variant_from_mode,
    variant_spec,
)


OUTPUT_FILENAMES: Tuple[str, ...] = (
    "selection_candidates.csv",
    "selection_runs.csv",
    "selected_configs.csv",
    "selected_configs.json",
    "selection_input_manifest.json",
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
    "penalty_reduction",
    "evaluation_split",
    "eg_graph_semantics",
    "selection_runs",
)


class SelectionValidationError(RuntimeError):
    """Raised when a selection artifact is incomplete or untrustworthy."""


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SelectionValidationError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=_strict_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectionValidationError(
            f"Cannot read selection artifact {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise SelectionValidationError(
            f"Selection artifact {path} must be a JSON object"
        )
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectionValidationError(
            f"{label} must be a finite number, got {value!r}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise SelectionValidationError(
            f"{label} must be finite, got {value!r}"
        )
    return number


def _metric_scalar(value: object, label: str) -> float:
    # Some legacy metric helpers return (display_name, scalar).  The corrected
    # runner serializes scalars, but accepting this shape lets us produce a
    # precise validation error later if one helper still uses the old wrapper.
    if isinstance(value, list) and len(value) >= 2:
        value = value[1]
    return _finite_number(value, label)


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)


def _require_metadata(
    info: Mapping[str, object],
    *,
    path: Path,
    source_manifest_file_sha256: str,
    expected_brightkite_split_sha256: str,
) -> Tuple[str, str]:
    dataset = info.get("dataset")
    if dataset not in DATASETS:
        raise SelectionValidationError(
            f"{path}: dataset must be one of {DATASETS}, got {dataset!r}"
        )
    try:
        variant = variant_from_mode(str(info.get("penalization_mode")))
    except ValueError as exc:
        raise SelectionValidationError(f"{path}: {exc}") from exc

    required = {
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
    if (
        not isinstance(split_fingerprint, str)
        or len(split_fingerprint) != 64
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

    return str(dataset), variant


def _validate_grid_metadata(
    info: Mapping[str, object],
    *,
    path: Path,
    variant: str,
    penalty: float,
) -> None:
    spec = variant_spec(variant)
    expected = {
        str(spec["penalty_name"]): [float(penalty)],
        "beta": list(BETA_VALUES),
        "lambda_mobility": list(MOBILITY_VALUES),
    }
    if info.get("grid") != expected:
        raise SelectionValidationError(
            f"{path}: experiment_info.grid must equal {expected!r}, "
            f"got {info.get('grid')!r}"
        )


def _validate_stats_and_runs(
    *,
    path: Path,
    config_key: str,
    result: Mapping[str, object],
    individual_runs: Sequence[Mapping[str, object]],
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    stats = result.get("statistics")
    if not isinstance(stats, dict):
        raise SelectionValidationError(
            f"{path}: {config_key}.statistics must be an object"
        )

    objective_summary: Dict[str, float] = {}
    run_rows: List[Dict[str, float]] = [
        {} for _ in range(SELECTION_RUNS)
    ]
    for objective in OBJECTIVES:
        entry = stats.get(objective)
        if not isinstance(entry, dict):
            raise SelectionValidationError(
                f"{path}: {config_key} lacks statistics for {objective}"
            )
        mean = _finite_number(
            entry.get("mean"), f"{path}:{config_key}:{objective}:mean"
        )
        std = _finite_number(
            entry.get("std"), f"{path}:{config_key}:{objective}:std"
        )
        values = entry.get("values")
        if not isinstance(values, list) or len(values) != SELECTION_RUNS:
            raise SelectionValidationError(
                f"{path}: {config_key}:{objective}.values must contain "
                f"exactly {SELECTION_RUNS} runs"
            )
        numeric_values = [
            _finite_number(
                value,
                f"{path}:{config_key}:{objective}:values[{run_index}]",
            )
            for run_index, value in enumerate(values)
        ]
        recomputed_mean = math.fsum(numeric_values) / SELECTION_RUNS
        recomputed_std = math.sqrt(
            math.fsum(
                (value - recomputed_mean) ** 2 for value in numeric_values
            )
            / SELECTION_RUNS
        )
        if not _close(mean, recomputed_mean):
            raise SelectionValidationError(
                f"{path}: {config_key}:{objective} mean {mean} does not "
                f"match its values ({recomputed_mean})"
            )
        if not _close(std, recomputed_std):
            raise SelectionValidationError(
                f"{path}: {config_key}:{objective} std {std} does not "
                f"match its values ({recomputed_std})"
            )

        for run_index, run in enumerate(individual_runs):
            config_result = run.get(config_key)
            if not isinstance(config_result, dict):
                raise SelectionValidationError(
                    f"{path}: run {run_index} lacks {config_key}"
                )
            run_value = _metric_scalar(
                config_result.get(objective),
                f"{path}:run {run_index}:{config_key}:{objective}",
            )
            if not _close(run_value, numeric_values[run_index]):
                raise SelectionValidationError(
                    f"{path}: run {run_index} value for "
                    f"{config_key}:{objective} ({run_value}) disagrees with "
                    f"statistics.values ({numeric_values[run_index]})"
                )
            run_rows[run_index][objective] = run_value

        objective_summary[f"{objective}_mean"] = mean
        objective_summary[f"{objective}_std"] = std

    return objective_summary, run_rows


def _dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(a >= b for a, b in zip(left, right)) and any(
        a > b for a, b in zip(left, right)
    )


def pareto_asf_select(
    candidates: Sequence[MutableMapping[str, object]],
) -> MutableMapping[str, object]:
    """Mark the Pareto front and choose one deterministic ASF compromise.

    All objectives are maximized.  The normalization and ASF convention match
    ``analysis/generate_final_tables.py``: normalize the full candidate set,
    negate it for minimization, normalize the Pareto front again, and minimize
    ``max(front_loss * (0.5, 0.3, 0.1, 0.1))``.  Exact score ties are resolved
    by lexicographically smallest ``config_key``.
    """

    if not candidates:
        raise SelectionValidationError("Cannot select from an empty grid")

    vectors: List[Tuple[float, ...]] = []
    for row_index, row in enumerate(candidates):
        vector = tuple(
            _finite_number(
                row.get(f"{objective}_mean"),
                f"candidate {row_index}:{objective}_mean",
            )
            for objective in OBJECTIVES
        )
        vectors.append(vector)

    pareto_indices = [
        index
        for index, vector in enumerate(vectors)
        if not any(
            other_index != index
            and _dominates(other_vector, vector)
            for other_index, other_vector in enumerate(vectors)
        )
    ]
    if not pareto_indices:
        raise SelectionValidationError("Pareto-front computation returned empty")

    minima = [
        min(vector[column] for vector in vectors)
        for column in range(len(OBJECTIVES))
    ]
    maxima = [
        max(vector[column] for vector in vectors)
        for column in range(len(OBJECTIVES))
    ]
    normalized = []
    for vector in vectors:
        normalized.append(
            tuple(
                (vector[column] - minima[column])
                / (
                    maxima[column] - minima[column]
                    if maxima[column] != minima[column]
                    else 1.0
                )
                for column in range(len(OBJECTIVES))
            )
        )

    front_losses = [
        tuple(-value for value in normalized[index])
        for index in pareto_indices
    ]
    front_minima = [
        min(vector[column] for vector in front_losses)
        for column in range(len(OBJECTIVES))
    ]
    front_maxima = [
        max(vector[column] for vector in front_losses)
        for column in range(len(OBJECTIVES))
    ]

    scores: Dict[int, float] = {}
    for index, losses in zip(pareto_indices, front_losses):
        normalized_losses = tuple(
            (losses[column] - front_minima[column])
            / (
                front_maxima[column] - front_minima[column]
                if front_maxima[column] != front_minima[column]
                else 1.0
            )
            for column in range(len(OBJECTIVES))
        )
        scores[index] = max(
            normalized_losses[column] * ASF_WEIGHTS[column]
            for column in range(len(OBJECTIVES))
        )

    for index, row in enumerate(candidates):
        row["is_pareto"] = index in scores
        row["asf_score"] = scores.get(index)
        row["selected"] = False

    best_index = min(
        pareto_indices,
        key=lambda index: (
            scores[index],
            str(candidates[index]["config_key"]),
        ),
    )
    candidates[best_index]["selected"] = True
    return candidates[best_index]


def _atomic_write_json(path: Path, value: object) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_csv(
    path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(columns), extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _result_paths(selection_root: Path) -> List[Path]:
    partials = sorted(selection_root.rglob("*.partial.json"))
    if partials:
        rendered = ", ".join(str(path) for path in partials[:5])
        raise SelectionValidationError(
            "Partial selection artifacts are present; refusing to finalize: "
            f"{rendered}"
        )
    paths = sorted(selection_root.rglob("*.json"))
    if not paths:
        raise SelectionValidationError(
            f"No JSON artifacts found below {selection_root}"
        )
    return paths


def finalize_selection(
    *,
    selection_root: Path,
    source_manifest_path: Path,
    output_dir: Path,
    repo_root: Path = REPO_ROOT,
    manifest_source_paths: Optional[Sequence[str]] = None,
    manifest_source_globs: Optional[Sequence[str]] = None,
    split_relative_path: str = BRIGHTKITE_TEMPORAL_SPLIT,
    expected_brightkite_split_sha256: str = (
        BRIGHTKITE_TEMPORAL_SPLIT_SHA256
    ),
) -> Dict[str, object]:
    """Validate all selection tasks and write immutable selected configs."""

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

    result_paths = _result_paths(selection_root)
    if len(result_paths) != SELECTION_TASK_COUNT:
        raise SelectionValidationError(
            f"Expected exactly {SELECTION_TASK_COUNT} isolated selection "
            f"artifacts, found {len(result_paths)}"
        )

    dataset_order = {dataset: index for index, dataset in enumerate(DATASETS)}
    variant_order = {variant: index for index, variant in enumerate(VARIANTS)}
    candidates: List[Dict[str, object]] = []
    run_output: List[Dict[str, object]] = []
    input_records: List[Dict[str, object]] = []
    seen_arms = set()
    seen_configs = set()
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
        dataset, variant = _require_metadata(
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
                parsed = parse_config_key(config_key)
            except ValueError as exc:
                raise SelectionValidationError(f"{path}: {exc}") from exc
            if parsed["variant"] != variant:
                raise SelectionValidationError(
                    f"{path}: {config_key} belongs to "
                    f"{parsed['variant']}, not {variant}"
                )
            parsed_configs[config_key] = parsed
        penalties = {
            parsed["penalty_value"] for parsed in parsed_configs.values()
        }
        if len(penalties) != 1:
            raise SelectionValidationError(
                f"{path}: isolated artifact spans penalties {penalties}"
            )
        penalty = float(next(iter(penalties)))
        expected_arm_keys = set(
            expected_penalty_config_keys(variant, penalty)
        )
        if set(lambda_results) != expected_arm_keys:
            missing = sorted(expected_arm_keys - set(lambda_results))
            extra = sorted(set(lambda_results) - expected_arm_keys)
            raise SelectionValidationError(
                f"{path}: incomplete penalty arm; missing={missing[:3]}, "
                f"extra={extra[:3]}"
            )
        _validate_grid_metadata(
            info, path=path, variant=variant, penalty=penalty
        )

        arm = (dataset, variant, penalty)
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
                "dataset": dataset,
                "variant": variant,
                "penalty_name": variant_spec(variant)["penalty_name"],
                "penalty_value": penalty,
                "split_fingerprint": info["split_fingerprint"],
            }
        )

        for config_key in expected_penalty_config_keys(variant, penalty):
            config_identity = (dataset, variant, config_key)
            if config_identity in seen_configs:
                raise SelectionValidationError(
                    f"Duplicate configuration {config_identity}"
                )
            seen_configs.add(config_identity)
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
                "variant": variant,
                "penalization_mode": parsed["penalization_mode"],
                "penalty_name": parsed["penalty_name"],
                "penalty_value": parsed["penalty_value"],
                "lambda_fair": parsed["lambda_fair"],
                "preproc_alpha": parsed["preproc_alpha"],
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
                "penalty_reduction": PENALTY_REDUCTION,
                "evaluation_split": EVALUATION_SPLIT,
                "eg_graph_semantics": EG_GRAPH_SEMANTICS,
                "selection_runs": SELECTION_RUNS,
            }
            candidate.update(summary)
            candidates.append(candidate)

            for run_index, run_metrics in enumerate(per_run):
                run_row: Dict[str, object] = {
                    "dataset": dataset,
                    "variant": variant,
                    "penalization_mode": parsed["penalization_mode"],
                    "run_index": run_index,
                    "model_seed": SEED_BASE + run_index,
                    "lambda_fair": parsed["lambda_fair"],
                    "preproc_alpha": parsed["preproc_alpha"],
                    "beta": parsed["beta_ra"],
                    "beta_ra": parsed["beta_ra"],
                    "lambda_mobility": parsed["lambda_mobility"],
                    "config_key": config_key,
                    "source_file": relative_path,
                }
                run_row.update(run_metrics)
                run_output.append(run_row)

    expected_arms = {
        (dataset, variant, float(penalty))
        for dataset in DATASETS
        for variant in VARIANTS
        for penalty in variant_spec(variant)["penalties"]
    }
    if seen_arms != expected_arms:
        missing = sorted(expected_arms - seen_arms)
        extra = sorted(seen_arms - expected_arms)
        raise SelectionValidationError(
            f"Selection arm set is incomplete; missing={missing}, extra={extra}"
        )

    expected_config_identities = {
        (dataset, variant, config_key)
        for dataset in DATASETS
        for variant in VARIANTS
        for config_key in expected_config_keys(variant)
    }
    if seen_configs != expected_config_identities:
        raise SelectionValidationError(
            "The combined selection grid is not exactly "
            "5 datasets × 2 variants × 210 configurations"
        )
    for dataset, fingerprints in split_fingerprints.items():
        if len(fingerprints) != 1:
            raise SelectionValidationError(
                f"{dataset}: selection arms used multiple split fingerprints: "
                f"{sorted(fingerprints)}"
            )

    config_order = {
        variant: {
            config_key: index
            for index, config_key in enumerate(expected_config_keys(variant))
        }
        for variant in VARIANTS
    }
    candidates.sort(
        key=lambda row: (
            dataset_order[str(row["dataset"])],
            variant_order[str(row["variant"])],
            config_order[str(row["variant"])][str(row["config_key"])],
        )
    )
    run_output.sort(
        key=lambda row: (
            dataset_order[str(row["dataset"])],
            variant_order[str(row["variant"])],
            config_order[str(row["variant"])][str(row["config_key"])],
            int(row["run_index"]),
        )
    )
    input_records.sort(
        key=lambda row: (
            dataset_order[str(row["dataset"])],
            variant_order[str(row["variant"])],
            float(row["penalty_value"]),
        )
    )

    selected: List[MutableMapping[str, object]] = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            group = [
                row
                for row in candidates
                if row["dataset"] == dataset and row["variant"] == variant
            ]
            if len(group) != CONFIGS_PER_VARIANT:
                raise SelectionValidationError(
                    f"{dataset}/{variant}: expected {CONFIGS_PER_VARIANT} "
                    f"candidates, found {len(group)}"
                )
            selected.append(pareto_asf_select(group))

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        output_dir / filename
        for filename in OUTPUT_FILENAMES
        if (output_dir / filename).exists()
    ]
    if existing:
        raise SelectionValidationError(
            "Refusing to overwrite frozen selection output(s): "
            + ", ".join(str(path) for path in existing)
        )

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
        "penalty_reduction",
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

    selected_rows = [
        {column: row.get(column) for column in SELECTED_COLUMNS}
        for row in selected
    ]
    input_manifest = {
        "schema_version": 1,
        "manifest_kind": "egfix_selection_inputs",
        "protocol": protocol_document(),
        "source_manifest_path": str(source_manifest_path.resolve()),
        "source_manifest_sha256": source_manifest_file_sha256,
        "source_manifest_content_sha256": source_manifest_content_sha256,
        "artifacts": input_records,
    }
    input_manifest["input_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(input_manifest)
    ).hexdigest()

    selected_json = {
        "schema_version": 1,
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
            dataset: {
                variant: next(
                    row
                    for row in selected_rows
                    if row["dataset"] == dataset
                    and row["variant"] == variant
                )
                for variant in VARIANTS
            }
            for dataset in DATASETS
        },
    }

    _atomic_write_csv(
        output_dir / "selection_candidates.csv",
        candidates,
        candidate_columns,
    )
    _atomic_write_csv(
        output_dir / "selection_runs.csv", run_output, run_columns
    )
    _atomic_write_csv(
        output_dir / "selected_configs.csv",
        selected_rows,
        SELECTED_COLUMNS,
    )
    _atomic_write_json(
        output_dir / "selected_configs.json", selected_json
    )
    _atomic_write_json(
        output_dir / "selection_input_manifest.json", input_manifest
    )

    return {
        "selection_artifacts": len(result_paths),
        "candidates": len(candidates),
        "run_rows": len(run_output),
        "selected": len(selected_rows),
        "source_manifest_sha256": source_manifest_file_sha256,
        "output_dir": str(output_dir),
    }


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly validate the 70 validation-only CAFE grid jobs and "
            "freeze one Pareto+ASF configuration per dataset and variant"
        )
    )
    parser.add_argument(
        "--campaign-root",
        "--campaign_root",
        dest="campaign_root",
        type=Path,
        help=(
            "Campaign root; defaults inputs to ROOT/selection and writes "
            "selection outputs directly below ROOT"
        ),
    )
    parser.add_argument(
        "--selection-root",
        "--selection_root",
        dest="selection_root",
        type=Path,
    )
    parser.add_argument(
        "--source-manifest",
        "--source_manifest",
        dest="source_manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir", "--output_dir", dest="output_dir", type=Path
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()

    if args.selection_root is None:
        if args.campaign_root is None:
            parser.error(
                "one of --selection_root or --campaign_root is required"
            )
        args.selection_root = args.campaign_root / "selection"
    if args.output_dir is None:
        if args.campaign_root is None:
            parser.error(
                "one of --output_dir or --campaign_root is required"
            )
        args.output_dir = args.campaign_root

    summary = finalize_selection(
        selection_root=args.selection_root,
        source_manifest_path=args.source_manifest,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    _main()
