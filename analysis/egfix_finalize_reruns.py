"""Strictly validate and finalize the corrected-EG rerun campaign.

The rerun jobs deliberately write one private shard per dataset and training
seed.  This module is the only promotion step: it validates the complete
5x10 aggregate and 5x3 source-stratified matrices, merges their canonical raw
records, computes uncertainty over runs (not users), renders the five paper
figures, and writes a completion marker only after every check succeeds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.egfix_preflight import (  # noqa: E402
    canonical_json_bytes,
    sha256_file,
    verify_source_manifest,
)
from experiments.egfix_protocol import (  # noqa: E402
    BRIGHTKITE_TEMPORAL_SPLIT,
    BRIGHTKITE_TEMPORAL_SPLIT_SHA256,
    CLOSENESS_EPSILON,
    CLOSENESS_RANDOM_SEED,
    CLOSENESS_SAMPLE_SIZE,
    DATASETS,
    EG_GRAPH_SEMANTICS,
    OBJECTIVES,
    PENALTY_REDUCTION,
    SEED_BASE,
    SPLIT_SEED,
    VARIANTS,
)


MODEL_ORDER: Tuple[str, ...] = (
    "RA",
    "HL-GNN",
    "Fusion",
    "CAFE-In",
    "CAFE-Pre",
)
AGGREGATE_SEEDS: Tuple[int, ...] = tuple(range(SEED_BASE, SEED_BASE + 10))
STRATIFIED_SEEDS: Tuple[int, ...] = tuple(range(SEED_BASE, SEED_BASE + 3))
AGGREGATE_PROTOCOL = "headline"
EVALUATION_SPLIT = "test"
K = 10
NUM_BINS_REQUESTED = 10
FINAL_DIRNAME = "finalized"
INTEGRITY_FILENAME = "rerun_integrity.json"
COMPLETION_MARKER = "_RERUNS_COMPLETE"

AGGREGATE_LONG_FILENAME = "aggregate_metrics_long.csv"
AGGREGATE_SUMMARY_FILENAME = "aggregate_metrics_summary.csv"
STRATIFIED_RAW_FILENAME = "stratified_precision_raw.csv"
STRATIFIED_RUN_FILENAME = "stratified_precision_by_run.csv"
STRATIFIED_SUMMARY_FILENAME = "stratified_precision_summary.csv"
STRATIFIED_EG_RUN_FILENAME = "stratified_degree_eg_per_run.csv"
STRATIFIED_EG_SUMMARY_FILENAME = "stratified_degree_eg_summary.csv"

DATASET_LABELS = {
    "ego-facebook": "Ego-FB",
    "oklahoma": "Oklahoma",
    "unc": "UNC",
    "pokec_n": "Pokec-n",
    "brightkite": "Brightkite",
}
TARGET_KEYS = {
    "RA": "targets_ra",
    "HL-GNN": "targets_hl_gnn",
    "Fusion": "targets_fusion",
    "CAFE-In": "targets_cafe_in",
    "CAFE-Pre": "targets_cafe_pre",
}


class RerunValidationError(RuntimeError):
    """Raised when rerun artifacts cannot be safely promoted."""


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RerunValidationError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=_strict_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise RerunValidationError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RerunValidationError(f"{path} must contain a JSON object")
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV without pandas' duplicate-header or ragged-row coercions."""

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            if not header or any(not name for name in header):
                raise RerunValidationError(f"{path}: empty CSV header")
            if len(header) != len(set(header)):
                raise RerunValidationError(f"{path}: duplicate CSV column")
            rows = []
            for line_number, row in enumerate(reader, start=2):
                if len(row) != len(header):
                    raise RerunValidationError(
                        f"{path}:{line_number}: expected {len(header)} "
                        f"fields, found {len(row)}"
                    )
                rows.append(row)
    except (OSError, UnicodeDecodeError, csv.Error, StopIteration) as exc:
        raise RerunValidationError(f"Cannot read CSV {path}: {exc}") from exc
    if not rows:
        raise RerunValidationError(f"{path}: CSV has no data rows")
    return pd.DataFrame(rows, columns=header, dtype=object)


def _require_columns(
    frame: pd.DataFrame, required: Sequence[str], path: Path
) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise RerunValidationError(
            f"{path}: missing columns {sorted(missing)}"
        )


def _text(value: object, label: str) -> str:
    if value is None:
        raise RerunValidationError(f"{label} must not be null")
    rendered = str(value)
    if not rendered:
        raise RerunValidationError(f"{label} must not be empty")
    return rendered


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise RerunValidationError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RerunValidationError(
            f"{label} must be a finite number, got {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise RerunValidationError(
            f"{label} must be finite, got {value!r}"
        )
    return number


def _integer(value: object, label: str) -> int:
    number = _finite(value, label)
    if not number.is_integer():
        raise RerunValidationError(
            f"{label} must be an integer, got {value!r}"
        )
    return int(number)


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)


def _require_sha(value: object, label: str) -> str:
    rendered = _text(value, label)
    if len(rendered) != 64:
        raise RerunValidationError(f"{label} must be a SHA256 digest")
    try:
        int(rendered, 16)
    except ValueError as exc:
        raise RerunValidationError(f"{label} must be a SHA256 digest") from exc
    return rendered.lower()


def _is_blank(value: object) -> bool:
    return value is None or str(value).strip() == ""


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise RerunValidationError(
            f"{label} must equal {expected!r}, got {actual!r}"
        )


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise RerunValidationError(
            f"Artifact escapes campaign root: {path}"
        ) from exc


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _atomic_write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _record_artifact(
    records: List[Dict[str, object]], path: Path, campaign_root: Path
) -> None:
    records.append(
        {
            "path": _safe_relative(path, campaign_root),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    )


def _assert_exact_success_matrix(
    root: Path, expected_task_dirs: Sequence[Path], label: str
) -> None:
    if not root.is_dir():
        raise RerunValidationError(f"Missing {label} root: {root}")
    expected = {(path / "_SUCCESS").resolve() for path in expected_task_dirs}
    actual = {path.resolve() for path in root.rglob("_SUCCESS") if path.is_file()}
    if actual != expected:
        missing = sorted(str(path) for path in expected - actual)
        extra = sorted(str(path) for path in actual - expected)
        raise RerunValidationError(
            f"{label}: success-marker matrix is not exact; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    partials = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and (
            ".partial" in path.name
            or ".tmp." in path.name
            or path.name.endswith(".tmp")
        )
    )
    if partials:
        raise RerunValidationError(
            f"{label}: partial artifacts remain: {partials[:5]}"
        )


def _validate_selected_configs(
    path: Path,
    *,
    source_manifest_file_sha256: str,
    source_manifest_content_sha256: str,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    frame = _read_csv(path)
    required = (
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
    _require_columns(frame, required, path)
    if len(frame) != len(DATASETS) * len(VARIANTS):
        raise RerunValidationError(
            f"{path}: expected exactly 10 selected rows, found {len(frame)}"
        )
    keys = list(zip(frame["dataset"], frame["variant"]))
    expected_keys = {
        (dataset, variant)
        for dataset in DATASETS
        for variant in VARIANTS
    }
    if set(keys) != expected_keys or len(keys) != len(set(keys)):
        raise RerunValidationError(
            f"{path}: selected rows are not exactly dataset x variant"
        )

    result: Dict[str, Dict[str, Dict[str, float]]] = {
        dataset: {} for dataset in DATASETS
    }
    for row_index, row in frame.iterrows():
        label = f"{path}:row {row_index + 2}"
        dataset = _text(row["dataset"], f"{label}:dataset")
        variant = _text(row["variant"], f"{label}:variant")
        _require_equal(
            _text(row["source_manifest_sha256"], f"{label}:source hash"),
            source_manifest_file_sha256,
            f"{label}:source_manifest_sha256",
        )
        _require_equal(
            _text(
                row["source_manifest_content_sha256"],
                f"{label}:source content hash",
            ),
            source_manifest_content_sha256,
            f"{label}:source_manifest_content_sha256",
        )
        _require_equal(
            row["penalty_reduction"],
            PENALTY_REDUCTION,
            f"{label}:penalty_reduction",
        )
        _require_equal(
            row["evaluation_split"], "valid", f"{label}:evaluation_split"
        )
        _require_equal(
            row["eg_graph_semantics"],
            EG_GRAPH_SEMANTICS,
            f"{label}:eg_graph_semantics",
        )
        _require_equal(
            _integer(row["selection_runs"], f"{label}:selection_runs"),
            3,
            f"{label}:selection_runs",
        )
        beta = _finite(row["beta"], f"{label}:beta")
        mobility = _finite(
            row["lambda_mobility"], f"{label}:lambda_mobility"
        )
        if variant == "cafe_in":
            penalty = _finite(row["lambda_fair"], f"{label}:lambda_fair")
            if not _is_blank(row["preproc_alpha"]):
                raise RerunValidationError(
                    f"{label}: cafe_in preproc_alpha must be blank"
                )
            result[dataset][variant] = {
                "lambda_fair": penalty,
                "beta": beta,
                "lambda_mobility": mobility,
            }
        elif variant == "cafe_pre":
            penalty = _finite(
                row["preproc_alpha"], f"{label}:preproc_alpha"
            )
            if not _is_blank(row["lambda_fair"]):
                raise RerunValidationError(
                    f"{label}: cafe_pre lambda_fair must be blank"
                )
            result[dataset][variant] = {
                "preproc_alpha": penalty,
                "beta": beta,
                "lambda_mobility": mobility,
            }
        else:
            raise RerunValidationError(f"{label}: unknown variant {variant!r}")
    return result


def _validate_run_parameters(
    row: Mapping[str, object],
    *,
    label: str,
    selected: Mapping[str, Mapping[str, float]],
) -> None:
    expected = {
        "cafe_in_lambda": selected["cafe_in"]["lambda_fair"],
        "cafe_in_beta": selected["cafe_in"]["beta"],
        "cafe_in_lam": selected["cafe_in"]["lambda_mobility"],
        "cafe_pre_alpha": selected["cafe_pre"]["preproc_alpha"],
        "cafe_pre_beta": selected["cafe_pre"]["beta"],
        "cafe_pre_lam": selected["cafe_pre"]["lambda_mobility"],
        "fusion_beta": selected["cafe_in"]["beta"],
    }
    for field, wanted in expected.items():
        actual = _finite(row[field], f"{label}:{field}")
        if not _close(actual, wanted):
            raise RerunValidationError(
                f"{label}:{field} must equal selected value {wanted}, "
                f"got {actual}"
            )
    fusion = _finite(row["fusion_beta"], f"{label}:fusion_beta")
    cafe_in_beta = _finite(row["cafe_in_beta"], f"{label}:cafe_in_beta")
    if not _close(fusion, cafe_in_beta):
        raise RerunValidationError(
            f"{label}: Fusion beta is not matched to CAFE-In beta"
        )


def _validate_aggregate(
    campaign_root: Path,
    *,
    selected_configs_sha256: str,
    source_manifest_sha256: str,
    selected: Mapping[str, Mapping[str, Mapping[str, float]]],
    expected_brightkite_split_sha256: str,
    input_artifacts: List[Dict[str, object]],
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, object]]:
    root = campaign_root / "aggregate"
    task_dirs = [
        root / dataset / f"seed_{seed}"
        for dataset in DATASETS
        for seed in AGGREGATE_SEEDS
    ]
    _assert_exact_success_matrix(root, task_dirs, "aggregate")

    required_columns = (
        "dataset",
        "seed",
        "split_seed",
        "method",
        "metric",
        "value",
        "protocol",
        "evaluation_split",
        "eg_graph_semantics",
        "penalty_reduction",
        "closeness_random_seed",
        "closeness_sample_size",
        "closeness_epsilon",
        "split_fingerprint",
        "split_sha256",
        "selected_configs_sha256",
        "source_manifest_sha256",
        "cafe_in_lambda",
        "cafe_in_beta",
        "cafe_in_lam",
        "cafe_pre_alpha",
        "cafe_pre_beta",
        "cafe_pre_lam",
        "fusion_beta",
    )
    output_rows: List[Dict[str, object]] = []
    fingerprints: Dict[str, set] = {dataset: set() for dataset in DATASETS}
    shard_records = []

    for dataset in DATASETS:
        for seed in AGGREGATE_SEEDS:
            task_dir = root / dataset / f"seed_{seed}"
            csv_path = task_dir / "per_seed_metrics.csv"
            marker_path = task_dir / "per_seed_metrics.csv.complete.json"
            success_path = task_dir / "_SUCCESS"
            for required_path in (csv_path, marker_path, success_path):
                if not required_path.is_file():
                    raise RerunValidationError(
                        f"Missing aggregate artifact: {required_path}"
                    )
                _record_artifact(
                    input_artifacts, required_path, campaign_root
                )

            frame = _read_csv(csv_path)
            _require_columns(frame, required_columns, csv_path)
            if set(frame["method"]) != set(MODEL_ORDER):
                raise RerunValidationError(
                    f"{csv_path}: method set must equal {MODEL_ORDER}"
                )
            identities = list(zip(frame["method"], frame["metric"]))
            if len(identities) != len(set(identities)):
                raise RerunValidationError(
                    f"{csv_path}: duplicate method/metric rows"
                )
            for method in MODEL_ORDER:
                metrics = set(frame.loc[frame["method"] == method, "metric"])
                missing = set(OBJECTIVES) - metrics
                if missing:
                    raise RerunValidationError(
                        f"{csv_path}: {method} lacks objectives "
                        f"{sorted(missing)}"
                    )

            shard_fingerprints = set()
            for row_index, row in frame.iterrows():
                label = f"{csv_path}:row {row_index + 2}"
                _require_equal(row["dataset"], dataset, f"{label}:dataset")
                _require_equal(
                    _integer(row["seed"], f"{label}:seed"),
                    seed,
                    f"{label}:seed",
                )
                _require_equal(
                    _integer(row["split_seed"], f"{label}:split_seed"),
                    SPLIT_SEED,
                    f"{label}:split_seed",
                )
                _require_equal(
                    row["protocol"],
                    AGGREGATE_PROTOCOL,
                    f"{label}:protocol",
                )
                _require_equal(
                    row["evaluation_split"],
                    EVALUATION_SPLIT,
                    f"{label}:evaluation_split",
                )
                _require_equal(
                    row["eg_graph_semantics"],
                    EG_GRAPH_SEMANTICS,
                    f"{label}:eg_graph_semantics",
                )
                _require_equal(
                    row["penalty_reduction"],
                    PENALTY_REDUCTION,
                    f"{label}:penalty_reduction",
                )
                _require_equal(
                    _integer(
                        row["closeness_random_seed"],
                        f"{label}:closeness_random_seed",
                    ),
                    CLOSENESS_RANDOM_SEED,
                    f"{label}:closeness_random_seed",
                )
                _require_equal(
                    _integer(
                        row["closeness_sample_size"],
                        f"{label}:closeness_sample_size",
                    ),
                    CLOSENESS_SAMPLE_SIZE,
                    f"{label}:closeness_sample_size",
                )
                closeness_epsilon = _finite(
                    row["closeness_epsilon"],
                    f"{label}:closeness_epsilon",
                )
                if not _close(closeness_epsilon, CLOSENESS_EPSILON):
                    raise RerunValidationError(
                        f"{label}: closeness epsilon must be "
                        f"{CLOSENESS_EPSILON}"
                    )
                fingerprint = _require_sha(
                    row["split_fingerprint"],
                    f"{label}:split_fingerprint",
                )
                shard_fingerprints.add(fingerprint)
                _require_equal(
                    row["selected_configs_sha256"],
                    selected_configs_sha256,
                    f"{label}:selected_configs_sha256",
                )
                _require_equal(
                    row["source_manifest_sha256"],
                    source_manifest_sha256,
                    f"{label}:source_manifest_sha256",
                )
                if dataset == "brightkite":
                    _require_equal(
                        row["split_sha256"],
                        expected_brightkite_split_sha256,
                        f"{label}:split_sha256",
                    )
                elif not _is_blank(row["split_sha256"]):
                    raise RerunValidationError(
                        f"{label}: non-Brightkite split_sha256 must be blank"
                    )
                _validate_run_parameters(
                    row, label=label, selected=selected[dataset]
                )

            if len(shard_fingerprints) != 1:
                raise RerunValidationError(
                    f"{csv_path}: multiple split fingerprints"
                )
            fingerprint = next(iter(shard_fingerprints))
            fingerprints[dataset].add(fingerprint)

            marker = _read_json(marker_path)
            expected_marker = {
                "status": "complete",
                "dataset": dataset,
                "protocol": AGGREGATE_PROTOCOL,
                "seeds": [seed],
                "csv_sha256": sha256_file(csv_path),
                "selected_configs_sha256": selected_configs_sha256,
                "source_manifest_sha256": source_manifest_sha256,
                "eg_graph_semantics": EG_GRAPH_SEMANTICS,
                "penalty_reduction": PENALTY_REDUCTION,
            }
            for field, wanted in expected_marker.items():
                _require_equal(
                    marker.get(field), wanted, f"{marker_path}:{field}"
                )

            objective_frame = frame.loc[
                frame["metric"].isin(OBJECTIVES)
            ].copy()
            if len(objective_frame) != len(MODEL_ORDER) * len(OBJECTIVES):
                raise RerunValidationError(
                    f"{csv_path}: expected exactly 20 objective rows"
                )
            for row_index, row in objective_frame.iterrows():
                value = _finite(
                    row["value"],
                    f"{csv_path}:row {row_index + 2}:value",
                )
                output_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "split_seed": SPLIT_SEED,
                        "method": str(row["method"]),
                        "metric": str(row["metric"]),
                        "value": value,
                        "split_fingerprint": fingerprint,
                        "protocol": AGGREGATE_PROTOCOL,
                        "evaluation_split": EVALUATION_SPLIT,
                        "eg_graph_semantics": EG_GRAPH_SEMANTICS,
                        "penalty_reduction": PENALTY_REDUCTION,
                        "source_shard": _safe_relative(
                            csv_path, campaign_root
                        ),
                        "source_shard_sha256": sha256_file(csv_path),
                    }
                )
            shard_records.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "rows": len(frame),
                    "objective_rows": len(objective_frame),
                    "split_fingerprint": fingerprint,
                }
            )

    for dataset, values in fingerprints.items():
        if len(values) != 1:
            raise RerunValidationError(
                f"{dataset}: aggregate fixed-split shards have "
                f"{len(values)} fingerprints"
            )
    if len(output_rows) != 5 * 10 * 5 * 4:
        raise RerunValidationError(
            f"Aggregate merge expected 1000 objective rows, "
            f"found {len(output_rows)}"
        )
    long_frame = pd.DataFrame(output_rows)
    long_frame = _sort_frame(long_frame)
    return (
        long_frame,
        {dataset: next(iter(values)) for dataset, values in fingerprints.items()},
        {"shards": shard_records, "objective_rows": len(long_frame)},
    )


def _validate_metadata_parameters(
    metadata: Mapping[str, object],
    *,
    label: str,
    selected: Mapping[str, Mapping[str, float]],
) -> None:
    cafe_in = metadata.get("cafe_in")
    cafe_pre = metadata.get("cafe_pre")
    if not isinstance(cafe_in, dict) or not isinstance(cafe_pre, dict):
        raise RerunValidationError(
            f"{label}: cafe_in and cafe_pre must be objects"
        )
    checks = (
        (
            cafe_in.get("lambda_fair"),
            selected["cafe_in"]["lambda_fair"],
            "cafe_in.lambda_fair",
        ),
        (
            cafe_in.get("beta"),
            selected["cafe_in"]["beta"],
            "cafe_in.beta",
        ),
        (
            cafe_in.get("lambda_mobility"),
            selected["cafe_in"]["lambda_mobility"],
            "cafe_in.lambda_mobility",
        ),
        (
            cafe_pre.get("preproc_alpha"),
            selected["cafe_pre"]["preproc_alpha"],
            "cafe_pre.preproc_alpha",
        ),
        (
            cafe_pre.get("beta"),
            selected["cafe_pre"]["beta"],
            "cafe_pre.beta",
        ),
        (
            cafe_pre.get("lambda_mobility"),
            selected["cafe_pre"]["lambda_mobility"],
            "cafe_pre.lambda_mobility",
        ),
        (
            metadata.get("fusion_beta"),
            selected["cafe_in"]["beta"],
            "fusion_beta",
        ),
    )
    for raw, expected, field in checks:
        actual = _finite(raw, f"{label}:{field}")
        if not _close(actual, expected):
            raise RerunValidationError(
                f"{label}:{field} must equal {expected}, got {actual}"
            )
    if not _close(
        _finite(metadata.get("fusion_beta"), f"{label}:fusion_beta"),
        _finite(cafe_in.get("beta"), f"{label}:cafe_in.beta"),
    ):
        raise RerunValidationError(
            f"{label}: Fusion beta is not matched to CAFE-In beta"
        )


def _parse_npz_metadata(raw: np.ndarray, path: Path) -> Mapping[str, object]:
    if raw.shape != ():
        raise RerunValidationError(
            f"{path}: metadata_json must be a scalar string"
        )
    try:
        rendered = str(raw.item())
        value = json.loads(rendered, object_pairs_hook=_strict_object)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RerunValidationError(
            f"{path}: invalid metadata_json: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise RerunValidationError(
            f"{path}: metadata_json must decode to an object"
        )
    return value


def _validate_recommendations(
    path: Path,
    *,
    dataset: str,
    seed: int,
    run_metadata: Mapping[str, object],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Mapping[str, object]]:
    expected_keys = {"sources", "metadata_json", *TARGET_KEYS.values()}
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected_keys:
                raise RerunValidationError(
                    f"{path}: recommendation keys must equal "
                    f"{sorted(expected_keys)}, got {sorted(archive.files)}"
                )
            sources = np.asarray(archive["sources"])
            metadata = _parse_npz_metadata(
                np.asarray(archive["metadata_json"]), path
            )
            targets = {
                method: np.asarray(archive[key])
                for method, key in TARGET_KEYS.items()
            }
    except (OSError, ValueError) as exc:
        if isinstance(exc, RerunValidationError):
            raise
        raise RerunValidationError(
            f"Cannot read recommendation artifact {path}: {exc}"
        ) from exc

    num_nodes = _integer(
        run_metadata.get("num_nodes"), f"{path}:metadata:num_nodes"
    )
    num_users = _integer(
        run_metadata.get("num_eval_users"),
        f"{path}:metadata:num_eval_users",
    )
    if sources.shape != (num_users,):
        raise RerunValidationError(
            f"{path}: sources shape must be ({num_users},), "
            f"got {sources.shape}"
        )
    if not np.issubdtype(sources.dtype, np.integer):
        raise RerunValidationError(f"{path}: sources must have integer dtype")
    sources = sources.astype(np.int64, copy=False)
    if num_users <= 0 or len(np.unique(sources)) != num_users:
        raise RerunValidationError(
            f"{path}: sources must be nonempty and unique"
        )
    if np.any(sources < 0) or np.any(sources >= num_nodes):
        raise RerunValidationError(f"{path}: source ID outside node range")
    if not np.array_equal(sources, np.sort(sources)):
        raise RerunValidationError(f"{path}: sources must be sorted")

    for method, matrix in targets.items():
        if matrix.shape != (num_users, K):
            raise RerunValidationError(
                f"{path}:{method}: target shape must be "
                f"({num_users}, {K}), got {matrix.shape}"
            )
        if not np.issubdtype(matrix.dtype, np.integer):
            raise RerunValidationError(
                f"{path}:{method}: targets must have integer dtype"
            )
        matrix = matrix.astype(np.int64, copy=False)
        targets[method] = matrix
        if np.any(matrix < 0) or np.any(matrix >= num_nodes):
            raise RerunValidationError(
                f"{path}:{method}: target ID outside node range"
            )
        if np.any(matrix == sources[:, None]):
            raise RerunValidationError(
                f"{path}:{method}: self-recommendation found"
            )
        if np.any(np.diff(np.sort(matrix, axis=1), axis=1) == 0):
            raise RerunValidationError(
                f"{path}:{method}: duplicate target within a top-{K} row"
            )

    core_fields = (
        "status",
        "dataset",
        "training_seed",
        "split_seed",
        "split_fingerprint",
        "split_sha256",
        "num_nodes",
        "num_eval_users",
        "k",
        "num_bins_requested",
        "methods",
        "eg_graph_semantics",
        "evaluation_split",
        "penalty_reduction",
        "selected_configs_sha256",
        "source_manifest_sha256",
        "fusion_beta",
        "cafe_in",
        "cafe_pre",
        "recommendation_validation_passed",
    )
    for field in core_fields:
        _require_equal(
            metadata.get(field),
            run_metadata.get(field),
            f"{path}:metadata_json:{field}",
        )
    _require_equal(metadata.get("dataset"), dataset, f"{path}:dataset")
    _require_equal(
        _integer(metadata.get("training_seed"), f"{path}:training_seed"),
        seed,
        f"{path}:training_seed",
    )
    return sources, targets, metadata


def _typed_precision_frame(
    frame: pd.DataFrame,
    *,
    path: Path,
    campaign_root: Path,
    dataset: str,
    seed: int,
    fingerprint: str,
    sources: np.ndarray,
    num_nodes: int,
    actual_bins: int,
) -> pd.DataFrame:
    required = (
        "node_id",
        "model",
        "precision",
        "node_coreness",
        "degree",
        "run",
        "training_seed",
        "split_seed",
        "split_fingerprint",
        "coreness_bin",
        "coreness_bin_label",
    )
    _require_columns(frame, required, path)
    expected_rows = len(sources) * len(MODEL_ORDER)
    if len(frame) != expected_rows:
        raise RerunValidationError(
            f"{path}: expected {expected_rows} precision rows, "
            f"found {len(frame)}"
        )
    typed = frame.copy()
    for column in (
        "node_id",
        "run",
        "training_seed",
        "split_seed",
        "coreness_bin",
    ):
        typed[column] = [
            _integer(value, f"{path}:row {index + 2}:{column}")
            for index, value in enumerate(typed[column])
        ]
    for column in ("precision", "node_coreness", "degree"):
        typed[column] = [
            _finite(value, f"{path}:row {index + 2}:{column}")
            for index, value in enumerate(typed[column])
        ]
    if set(typed["model"]) != set(MODEL_ORDER):
        raise RerunValidationError(
            f"{path}: method set must equal {MODEL_ORDER}"
        )
    if typed[["model", "node_id"]].duplicated().any():
        raise RerunValidationError(f"{path}: duplicate model/node precision row")
    if not typed["run"].eq(seed).all():
        raise RerunValidationError(f"{path}: run must equal seed {seed}")
    if not typed["training_seed"].eq(seed).all():
        raise RerunValidationError(
            f"{path}: training_seed must equal {seed}"
        )
    if not typed["split_seed"].eq(SPLIT_SEED).all():
        raise RerunValidationError(
            f"{path}: split_seed must equal {SPLIT_SEED}"
        )
    if set(typed["split_fingerprint"]) != {fingerprint}:
        raise RerunValidationError(
            f"{path}: split fingerprint does not match metadata"
        )
    if (typed["node_id"] < 0).any() or (typed["node_id"] >= num_nodes).any():
        raise RerunValidationError(f"{path}: node ID outside range")
    if (typed["precision"] < 0).any() or (typed["precision"] > 1).any():
        raise RerunValidationError(f"{path}: precision outside [0, 1]")
    if (typed["degree"] < 0).any() or (typed["node_coreness"] < 0).any():
        raise RerunValidationError(
            f"{path}: degree and coreness must be nonnegative"
        )
    rounded_hits = np.rint(typed["precision"].to_numpy(float) * K)
    if not np.allclose(
        typed["precision"].to_numpy(float) * K,
        rounded_hits,
        rtol=0,
        atol=1e-8,
    ):
        raise RerunValidationError(
            f"{path}: per-user Precision@{K} is not a hit count divided by K"
        )
    expected_sources = set(int(value) for value in sources)
    for method in MODEL_ORDER:
        method_nodes = set(
            typed.loc[typed["model"] == method, "node_id"].astype(int)
        )
        if method_nodes != expected_sources:
            raise RerunValidationError(
                f"{path}:{method}: evaluated source set differs from NPZ"
            )
    invariant_columns = (
        "node_coreness",
        "degree",
        "coreness_bin",
        "coreness_bin_label",
    )
    for column in invariant_columns:
        if typed.groupby("node_id")[column].nunique(dropna=False).max() != 1:
            raise RerunValidationError(
                f"{path}: {column} differs across methods for one source"
            )
    bins = sorted(set(typed["coreness_bin"]))
    if bins != list(range(actual_bins)):
        raise RerunValidationError(
            f"{path}: expected contiguous bins 0..{actual_bins - 1}, "
            f"got {bins}"
        )
    label_pairs = typed[
        ["coreness_bin", "coreness_bin_label"]
    ].drop_duplicates()
    if len(label_pairs) != actual_bins:
        raise RerunValidationError(
            f"{path}: coreness bin labels are not one-to-one"
        )
    typed.insert(0, "dataset", dataset)
    typed["source_shard"] = _safe_relative(path, campaign_root)
    return typed


def _typed_eg_frame(
    frame: pd.DataFrame,
    *,
    path: Path,
    dataset: str,
    seed: int,
    actual_bins: int,
    campaign_root: Path,
) -> pd.DataFrame:
    required = (
        "model",
        "coreness_bin",
        "coreness_bin_label",
        "equality_gain",
        "run",
        "training_seed",
        "split_seed",
    )
    _require_columns(frame, required, path)
    expected_rows = actual_bins * len(MODEL_ORDER)
    if len(frame) != expected_rows:
        raise RerunValidationError(
            f"{path}: expected {expected_rows} Degree-EG rows, "
            f"found {len(frame)}"
        )
    typed = frame.copy()
    for column in (
        "coreness_bin",
        "run",
        "training_seed",
        "split_seed",
    ):
        typed[column] = [
            _integer(value, f"{path}:row {index + 2}:{column}")
            for index, value in enumerate(typed[column])
        ]
    typed["equality_gain"] = [
        _finite(value, f"{path}:row {index + 2}:equality_gain")
        for index, value in enumerate(typed["equality_gain"])
    ]
    if set(typed["model"]) != set(MODEL_ORDER):
        raise RerunValidationError(
            f"{path}: method set must equal {MODEL_ORDER}"
        )
    if typed[["model", "coreness_bin"]].duplicated().any():
        raise RerunValidationError(
            f"{path}: duplicate method/coreness-bin Degree-EG row"
        )
    if set(typed["coreness_bin"]) != set(range(actual_bins)):
        raise RerunValidationError(f"{path}: incomplete coreness-bin set")
    if not typed["run"].eq(seed).all():
        raise RerunValidationError(f"{path}: run must equal seed {seed}")
    if not typed["training_seed"].eq(seed).all():
        raise RerunValidationError(
            f"{path}: training_seed must equal {seed}"
        )
    if not typed["split_seed"].eq(SPLIT_SEED).all():
        raise RerunValidationError(
            f"{path}: split_seed must equal {SPLIT_SEED}"
        )
    label_pairs = typed[
        ["coreness_bin", "coreness_bin_label"]
    ].drop_duplicates()
    if len(label_pairs) != actual_bins:
        raise RerunValidationError(
            f"{path}: Degree-EG bin labels are not one-to-one"
        )
    typed.insert(0, "dataset", dataset)
    typed["source_shard"] = _safe_relative(path, campaign_root)
    return typed


def _validate_stratified(
    campaign_root: Path,
    *,
    selected_configs_sha256: str,
    source_manifest_sha256: str,
    selected: Mapping[str, Mapping[str, Mapping[str, float]]],
    expected_brightkite_split_sha256: str,
    aggregate_fingerprints: Mapping[str, str],
    input_artifacts: List[Dict[str, object]],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, object]], Dict[str, object]]:
    root = campaign_root / "stratified"
    task_dirs = [
        root / dataset / f"seed_{seed}"
        for dataset in DATASETS
        for seed in STRATIFIED_SEEDS
    ]
    _assert_exact_success_matrix(root, task_dirs, "stratified")

    precision_frames = []
    eg_frames = []
    recommendation_records: List[Dict[str, object]] = []
    dataset_profiles: Dict[str, Dict[str, object]] = {}
    shard_records = []

    for dataset in DATASETS:
        for seed in STRATIFIED_SEEDS:
            task_dir = root / dataset / f"seed_{seed}"
            precision_path = task_dir / f"stratified_cafe_{dataset}.csv"
            eg_path = (
                task_dir
                / f"eg_by_coreness_cafe_{dataset}_per_run.csv"
            )
            metadata_path = task_dir / f"metadata_{dataset}.json"
            rec_path = task_dir / f"recommendations_{dataset}_seed{seed}.npz"
            success_path = task_dir / "_SUCCESS"
            required_paths = (
                precision_path,
                eg_path,
                metadata_path,
                rec_path,
                success_path,
            )
            for required_path in required_paths:
                if not required_path.is_file():
                    raise RerunValidationError(
                        f"Missing stratified artifact: {required_path}"
                    )
                _record_artifact(
                    input_artifacts, required_path, campaign_root
                )

            top = _read_json(metadata_path)
            required_top = {
                "status": "complete",
                "dataset": dataset,
                "methods": list(MODEL_ORDER),
                "selected_configs_sha256": selected_configs_sha256,
                "source_manifest_sha256": source_manifest_sha256,
                "eg_graph_semantics": EG_GRAPH_SEMANTICS,
                "penalty_reduction": PENALTY_REDUCTION,
            }
            for field, expected in required_top.items():
                _require_equal(
                    top.get(field), expected, f"{metadata_path}:{field}"
                )
            runs = top.get("runs")
            if (
                not isinstance(runs, list)
                or len(runs) != 1
                or not isinstance(runs[0], dict)
            ):
                raise RerunValidationError(
                    f"{metadata_path}: runs must contain exactly one object"
                )
            run = runs[0]
            required_run = {
                "status": "complete",
                "dataset": dataset,
                "training_seed": seed,
                "split_seed": SPLIT_SEED,
                "k": K,
                "num_bins_requested": NUM_BINS_REQUESTED,
                "methods": list(MODEL_ORDER),
                "eg_graph_semantics": EG_GRAPH_SEMANTICS,
                "evaluation_split": EVALUATION_SPLIT,
                "penalty_reduction": PENALTY_REDUCTION,
                "selected_configs_sha256": selected_configs_sha256,
                "source_manifest_sha256": source_manifest_sha256,
                "recommendation_validation_passed": True,
                "recommendations_file": rec_path.name,
                "recommendations_sha256": sha256_file(rec_path),
            }
            for field, expected in required_run.items():
                _require_equal(
                    run.get(field),
                    expected,
                    f"{metadata_path}:runs[0].{field}",
                )
            fingerprint = _require_sha(
                run.get("split_fingerprint"),
                f"{metadata_path}:runs[0].split_fingerprint",
            )
            _require_equal(
                fingerprint,
                aggregate_fingerprints[dataset],
                f"{metadata_path}:aggregate/stratified split fingerprint",
            )
            split_sha = run.get("split_sha256")
            if dataset == "brightkite":
                _require_equal(
                    split_sha,
                    expected_brightkite_split_sha256,
                    f"{metadata_path}:runs[0].split_sha256",
                )
            elif split_sha is not None and split_sha != "":
                raise RerunValidationError(
                    f"{metadata_path}: non-Brightkite split_sha256 "
                    "must be null"
                )
            num_nodes = _integer(
                run.get("num_nodes"),
                f"{metadata_path}:runs[0].num_nodes",
            )
            num_users = _integer(
                run.get("num_eval_users"),
                f"{metadata_path}:runs[0].num_eval_users",
            )
            actual_bins = _integer(
                top.get("actual_bins"), f"{metadata_path}:actual_bins"
            )
            if num_nodes <= 0 or num_users <= 0:
                raise RerunValidationError(
                    f"{metadata_path}: node/user counts must be positive"
                )
            if not 1 <= actual_bins <= NUM_BINS_REQUESTED:
                raise RerunValidationError(
                    f"{metadata_path}: actual_bins outside 1..10"
                )
            _require_equal(
                _integer(
                    top.get("precision_rows"),
                    f"{metadata_path}:precision_rows",
                ),
                num_users * len(MODEL_ORDER),
                f"{metadata_path}:precision_rows",
            )
            _require_equal(
                _integer(top.get("eg_rows"), f"{metadata_path}:eg_rows"),
                actual_bins * len(MODEL_ORDER),
                f"{metadata_path}:eg_rows",
            )
            _validate_metadata_parameters(
                run,
                label=f"{metadata_path}:runs[0]",
                selected=selected[dataset],
            )

            sources, targets, _ = _validate_recommendations(
                rec_path,
                dataset=dataset,
                seed=seed,
                run_metadata=run,
            )
            precision = _typed_precision_frame(
                _read_csv(precision_path),
                path=precision_path,
                campaign_root=campaign_root,
                dataset=dataset,
                seed=seed,
                fingerprint=fingerprint,
                sources=sources,
                num_nodes=num_nodes,
                actual_bins=actual_bins,
            )
            eg = _typed_eg_frame(
                _read_csv(eg_path),
                path=eg_path,
                dataset=dataset,
                seed=seed,
                actual_bins=actual_bins,
                campaign_root=campaign_root,
            )
            precision_labels = set(
                zip(
                    precision["coreness_bin"],
                    precision["coreness_bin_label"],
                )
            )
            eg_labels = set(
                zip(eg["coreness_bin"], eg["coreness_bin_label"])
            )
            if precision_labels != eg_labels:
                raise RerunValidationError(
                    f"{task_dir}: precision and Degree-EG bin labels differ"
                )

            source_bins = (
                precision.loc[
                    precision["model"] == MODEL_ORDER[0],
                    ["node_id", "coreness_bin", "coreness_bin_label"],
                ]
                .sort_values("node_id")
                .reset_index(drop=True)
            )
            profile = {
                "num_nodes": num_nodes,
                "num_users": num_users,
                "actual_bins": actual_bins,
                "fingerprint": fingerprint,
                "sources": sources.tolist(),
                "source_bins": source_bins.to_dict("records"),
            }
            previous = dataset_profiles.get(dataset)
            if previous is None:
                dataset_profiles[dataset] = profile
            elif previous != profile:
                raise RerunValidationError(
                    f"{dataset}: fixed-split stratified shards disagree on "
                    "users, bins, counts, or split"
                )

            precision_frames.append(precision)
            eg_frames.append(eg)
            recommendation_records.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "path": rec_path,
                    "num_nodes": num_nodes,
                    "sources": sources,
                    "targets": targets,
                    "split_fingerprint": fingerprint,
                }
            )
            shard_records.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "num_nodes": num_nodes,
                    "num_users": num_users,
                    "actual_bins": actual_bins,
                    "precision_rows": len(precision),
                    "eg_rows": len(eg),
                    "split_fingerprint": fingerprint,
                }
            )

    raw = pd.concat(precision_frames, ignore_index=True)
    eg_raw = pd.concat(eg_frames, ignore_index=True)
    expected_precision_rows = sum(
        profile["num_users"] * len(MODEL_ORDER) * len(STRATIFIED_SEEDS)
        for profile in dataset_profiles.values()
    )
    expected_eg_rows = sum(
        profile["actual_bins"] * len(MODEL_ORDER) * len(STRATIFIED_SEEDS)
        for profile in dataset_profiles.values()
    )
    if len(raw) != expected_precision_rows or len(eg_raw) != expected_eg_rows:
        raise RerunValidationError(
            "Merged stratified row counts do not match dataset profiles"
        )
    raw = _sort_frame(raw, bin_column="coreness_bin")
    eg_raw = _sort_frame(eg_raw, bin_column="coreness_bin")
    return (
        raw,
        eg_raw,
        recommendation_records,
        {
            "shards": shard_records,
            "dataset_profiles": {
                dataset: {
                    key: value
                    for key, value in profile.items()
                    if key not in {"sources", "source_bins"}
                }
                for dataset, profile in dataset_profiles.items()
            },
            "precision_rows": len(raw),
            "eg_rows": len(eg_raw),
        },
    )


def _sort_frame(
    frame: pd.DataFrame, *, bin_column: Optional[str] = None
) -> pd.DataFrame:
    result = frame.copy()
    dataset_rank = {name: index for index, name in enumerate(DATASETS)}
    model_rank = {name: index for index, name in enumerate(MODEL_ORDER)}
    objective_rank = {
        name: index for index, name in enumerate(OBJECTIVES)
    }
    result["_dataset_order"] = result["dataset"].map(dataset_rank)
    sort_columns = ["_dataset_order"]
    if "seed" in result:
        sort_columns.append("seed")
    elif "training_seed" in result:
        sort_columns.append("training_seed")
    if bin_column is not None and bin_column in result:
        sort_columns.append(bin_column)
    if "model" in result:
        result["_model_order"] = result["model"].map(model_rank)
        sort_columns.append("_model_order")
    elif "method" in result:
        result["_model_order"] = result["method"].map(model_rank)
        sort_columns.append("_model_order")
    if "metric" in result:
        result["_metric_order"] = result["metric"].map(objective_rank)
        sort_columns.append("_metric_order")
    result = result.sort_values(sort_columns, kind="stable").reset_index(
        drop=True
    )
    return result.drop(
        columns=[
            name
            for name in (
                "_dataset_order",
                "_model_order",
                "_metric_order",
            )
            if name in result
        ]
    )


def _aggregate_metric_summary(long_frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        long_frame.groupby(["dataset", "method", "metric"], as_index=False)
        .agg(
            mean=("value", "mean"),
            std=("value", "std"),
            run_count=("value", "count"),
        )
    )
    if not summary["run_count"].eq(len(AGGREGATE_SEEDS)).all():
        raise RerunValidationError(
            "Aggregate uncertainty does not contain ten runs per cell"
        )
    summary["std"] = summary["std"].fillna(0.0)
    summary["sem"] = summary["std"] / np.sqrt(summary["run_count"])
    return _sort_frame(summary)


def _precision_summaries(
    raw: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from analysis import run_precision_by_coreness as rpbc
    except Exception as exc:
        raise RerunValidationError(
            "Cannot import run_precision_by_coreness for run-level "
            f"precision aggregation: {exc}"
        ) from exc

    run_frames = []
    for dataset in DATASETS:
        dataset_raw = raw.loc[raw["dataset"] == dataset].copy()
        per_run = rpbc.aggregate_precision_within_runs(dataset_raw)
        per_run.insert(0, "dataset", dataset)
        per_run.rename(columns={"run": "training_seed"}, inplace=True)
        run_frames.append(per_run)
    per_run = pd.concat(run_frames, ignore_index=True)
    per_run = _sort_frame(per_run, bin_column="coreness_bin")
    if (
        per_run[
            [
                "dataset",
                "training_seed",
                "model",
                "coreness_bin",
            ]
        ]
        .duplicated()
        .any()
    ):
        raise RerunValidationError("Duplicate run-level precision cell")

    summary = (
        per_run.groupby(
            [
                "dataset",
                "model",
                "coreness_bin",
                "coreness_bin_label",
            ],
            as_index=False,
        )
        .agg(
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            run_count=("precision", "count"),
            n_users=("n_nodes", "max"),
        )
    )
    if not summary["run_count"].eq(len(STRATIFIED_SEEDS)).all():
        raise RerunValidationError(
            "Precision uncertainty does not contain three runs per cell"
        )
    summary["precision_std"] = summary["precision_std"].fillna(0.0)
    summary["precision_sem"] = (
        summary["precision_std"] / np.sqrt(summary["run_count"])
    )
    return per_run, _sort_frame(summary, bin_column="coreness_bin")


def _eg_summary(eg_raw: pd.DataFrame) -> pd.DataFrame:
    summary = (
        eg_raw.groupby(
            [
                "dataset",
                "model",
                "coreness_bin",
                "coreness_bin_label",
            ],
            as_index=False,
        )
        .agg(
            equality_gain=("equality_gain", "mean"),
            eg_std=("equality_gain", "std"),
            eg_count=("equality_gain", "count"),
        )
    )
    if not summary["eg_count"].eq(len(STRATIFIED_SEEDS)).all():
        raise RerunValidationError(
            "Degree-EG uncertainty does not contain three runs per cell"
        )
    summary["eg_std"] = summary["eg_std"].fillna(0.0)
    summary["eg_sem"] = summary["eg_std"] / np.sqrt(summary["eg_count"])
    return _sort_frame(summary, bin_column="coreness_bin")


def _split_fingerprint(split_edge, num_nodes: int) -> str:
    import torch

    from metrics.graph_utils import canonicalize_undirected_edges

    digest = hashlib.sha256()
    digest.update(f"num_nodes={int(num_nodes)}\n".encode())
    for split_name in ("train", "test"):
        edges = canonicalize_undirected_edges(
            split_edge[split_name]["edge"].cpu(), num_nodes
        )
        digest.update(f"{split_name}:{len(edges)}\n".encode())
        digest.update(
            edges.to(dtype=torch.int64)
            .contiguous()
            .numpy()
            .astype("<i8", copy=False)
            .tobytes()
        )
    return digest.hexdigest()


def _crosscheck_degree_eg(
    *,
    repo_root: Path,
    recommendation_records: Sequence[Mapping[str, object]],
    aggregate_long: pd.DataFrame,
    aggregate_fingerprints: Mapping[str, str],
) -> Dict[str, object]:
    """Recompute whole-graph Degree EG from stratified NPZ files.

    The aggregate and stratified jobs train independently.  RA is deterministic
    and therefore provides an exact cross-path oracle.  Learned-model
    differences are recorded but are not treated as semantic failures because
    nondeterministic accelerator kernels can change their recommendation sets.
    """

    data_root = repo_root / "data" / "SNAP"
    required_dirs = [data_root / dataset for dataset in DATASETS]
    if not data_root.is_dir() or not all(path.is_dir() for path in required_dirs):
        return {
            "status": "skipped",
            "reason": "dataset files are not available below repo_root/data/SNAP",
            "comparisons": 0,
        }

    backbone_dir = repo_root / "backbone"
    if str(backbone_dir) not in sys.path:
        sys.path.insert(0, str(backbone_dir))
    try:
        from core.data_loader import load_dataset
        from metrics.equality_gain import compute_equality_gain
    except Exception as exc:
        raise RerunValidationError(
            f"Degree-EG artifact cross-check imports failed: {exc}"
        ) from exc

    records_by_dataset = {
        dataset: [
            record
            for record in recommendation_records
            if record["dataset"] == dataset
        ]
        for dataset in DATASETS
    }
    comparisons = []
    for dataset in DATASETS:
        args = SimpleNamespace(
            data_name=dataset,
            data_path=str(data_root),
            val_ratio=0.05,
            test_ratio=0.1,
            encoder="HLGNN",
            device="cpu",
            data_split_seed=SPLIT_SEED,
            split_file=(
                str(repo_root / BRIGHTKITE_TEMPORAL_SPLIT)
                if dataset == "brightkite"
                else None
            ),
        )
        try:
            loaded, split_edge, _, _ = load_dataset(args)
        except Exception as exc:
            raise RerunValidationError(
                f"{dataset}: cannot reconstruct fixed split for Degree-EG "
                f"cross-check: {exc}"
            ) from exc
        num_nodes = (
            loaded._data.num_nodes
            if hasattr(loaded, "_data")
            else loaded.data.num_nodes
        )
        fingerprint = _split_fingerprint(split_edge, num_nodes)
        _require_equal(
            fingerprint,
            aggregate_fingerprints[dataset],
            f"{dataset}: reconstructed split fingerprint",
        )
        train_edges = split_edge["train"]["edge"].cpu()
        test_edges = split_edge["test"]["edge"].cpu()
        expected_sources = np.sort(
            np.unique(test_edges.reshape(-1).numpy().astype(np.int64))
        )
        train_np = train_edges.numpy().astype(np.int64, copy=False)
        train_keys = np.unique(
            np.concatenate(
                (
                    train_np[:, 0] * int(num_nodes) + train_np[:, 1],
                    train_np[:, 1] * int(num_nodes) + train_np[:, 0],
                )
            )
        )

        for record in records_by_dataset[dataset]:
            if int(record["num_nodes"]) != int(num_nodes):
                raise RerunValidationError(
                    f"{record['path']}: num_nodes differs from loaded dataset"
                )
            sources = np.asarray(record["sources"])
            if not np.array_equal(sources, expected_sources):
                raise RerunValidationError(
                    f"{record['path']}: recommendation sources are not "
                    "exactly the fixed test split's incident users"
                )
            for method in MODEL_ORDER:
                targets = np.asarray(record["targets"][method])
                candidate_keys = (
                    sources[:, None] * int(num_nodes) + targets
                ).reshape(-1)
                positions = np.searchsorted(train_keys, candidate_keys)
                within = positions < len(train_keys)
                if np.any(
                    train_keys[positions[within]]
                    == candidate_keys[within]
                ):
                    raise RerunValidationError(
                        f"{record['path']}:{method}: recommendation contains "
                        "an observed training neighbor"
                    )
                recs = {
                    int(source): targets[index].astype(int).tolist()
                    for index, source in enumerate(sources)
                }
                recomputed, _, _ = compute_equality_gain(
                    recs,
                    train_edges,
                    test_edges,
                    int(num_nodes),
                    K,
                    graph_semantics="simple_undirected",
                )
                if not math.isfinite(recomputed):
                    raise RerunValidationError(
                        f"{record['path']}:{method}: recomputed Degree EG "
                        "is non-finite"
                    )
                match = aggregate_long.loc[
                    (aggregate_long["dataset"] == dataset)
                    & (aggregate_long["seed"] == int(record["seed"]))
                    & (aggregate_long["method"] == method)
                    & (aggregate_long["metric"] == "Equality_Gain@10"),
                    "value",
                ]
                if len(match) != 1:
                    raise RerunValidationError(
                        f"{dataset}/{record['seed']}/{method}: no unique "
                        "aggregate Degree-EG reference"
                    )
                aggregate_value = float(match.iloc[0])
                difference = recomputed - aggregate_value
                if method == "RA" and not math.isclose(
                    recomputed,
                    aggregate_value,
                    rel_tol=1e-9,
                    abs_tol=1e-8,
                ):
                    raise RerunValidationError(
                        f"{dataset}/seed {record['seed']}: deterministic RA "
                        "Degree EG disagrees across aggregate and saved-"
                        f"recommendation paths ({aggregate_value} vs "
                        f"{recomputed})"
                    )
                comparisons.append(
                    {
                        "dataset": dataset,
                        "seed": int(record["seed"]),
                        "method": method,
                        "aggregate_degree_eg": aggregate_value,
                        "artifact_degree_eg": recomputed,
                        "difference": difference,
                        "exact_oracle": method == "RA",
                    }
                )

    model_summary = {}
    for method in MODEL_ORDER:
        values = [
            abs(row["difference"])
            for row in comparisons
            if row["method"] == method
        ]
        model_summary[method] = {
            "comparisons": len(values),
            "mean_absolute_difference": math.fsum(values) / len(values),
            "max_absolute_difference": max(values),
        }
    return {
        "status": "passed",
        "comparison_note": (
            "RA is an exact cross-path oracle. Learned models are independently "
            "trained in the aggregate and stratified jobs, so their numeric "
            "differences are reported descriptively."
        ),
        "comparisons": len(comparisons),
        "model_summary": model_summary,
        "records": comparisons,
    }


def _render_figures(
    *,
    raw: pd.DataFrame,
    eg_summary: pd.DataFrame,
    figure_dir: Path,
) -> List[Path]:
    try:
        from analysis import run_precision_by_coreness as rpbc
    except Exception as exc:
        raise RerunValidationError(
            f"Cannot import run_precision_by_coreness for plotting: {exc}"
        ) from exc

    figure_dir.mkdir(parents=True, exist_ok=False)
    outputs = []
    for dataset in DATASETS:
        dataset_raw = raw.loc[raw["dataset"] == dataset].copy()
        dataset_eg = eg_summary.loc[
            eg_summary["dataset"] == dataset
        ].copy()
        rpbc.plot_stacked_barchart(
            dataset_raw,
            dataset_eg,
            str(figure_dir),
            dataset,
            K,
            model_order=list(MODEL_ORDER),
            display_name=DATASET_LABELS[dataset],
            filename_tag=dataset,
        )
        path = figure_dir / f"stacked_barchart_{dataset}.pdf"
        if not path.is_file() or path.stat().st_size == 0:
            raise RerunValidationError(
                f"Plotter did not create a nonempty PDF: {path}"
            )
        with path.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise RerunValidationError(f"Malformed PDF output: {path}")
        outputs.append(path)
    return outputs


def _output_records(root: Path) -> List[Dict[str, object]]:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != INTEGRITY_FILENAME:
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return records


def finalize_reruns(
    *,
    campaign_root: Path,
    selected_configs_path: Path,
    source_manifest_path: Path,
    repo_root: Path = REPO_ROOT,
    manifest_source_paths: Optional[Sequence[str]] = None,
    manifest_source_globs: Optional[Sequence[str]] = None,
    split_relative_path: str = BRIGHTKITE_TEMPORAL_SPLIT,
    expected_brightkite_split_sha256: str = (
        BRIGHTKITE_TEMPORAL_SPLIT_SHA256
    ),
    enable_graph_crosscheck: bool = True,
) -> Dict[str, object]:
    """Validate, merge, summarize, plot, and atomically finalize a campaign."""

    campaign_root = campaign_root.resolve()
    repo_root = repo_root.resolve()
    selected_configs_path = selected_configs_path.resolve()
    source_manifest_path = source_manifest_path.resolve()
    if not campaign_root.is_dir():
        raise RerunValidationError(
            f"Campaign root is not a directory: {campaign_root}"
        )
    for path, label in (
        (selected_configs_path, "selected configs"),
        (source_manifest_path, "source manifest"),
    ):
        if not path.is_file():
            raise RerunValidationError(f"Missing {label}: {path}")
    final_dir = campaign_root / FINAL_DIRNAME
    marker_path = campaign_root / COMPLETION_MARKER
    if final_dir.exists() or marker_path.exists():
        raise RerunValidationError(
            "Refusing to overwrite an already-finalized campaign"
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
        raise RerunValidationError(
            f"Source-manifest verification failed: {exc}"
        ) from exc
    source_manifest_sha256 = sha256_file(source_manifest_path)
    source_manifest_content_sha256 = _require_sha(
        source_manifest.get("manifest_sha256"),
        "source manifest content digest",
    )
    selected_configs_sha256 = sha256_file(selected_configs_path)
    selected = _validate_selected_configs(
        selected_configs_path,
        source_manifest_file_sha256=source_manifest_sha256,
        source_manifest_content_sha256=source_manifest_content_sha256,
    )

    input_artifacts: List[Dict[str, object]] = []
    _record_artifact(
        input_artifacts, selected_configs_path, campaign_root
    )
    _record_artifact(input_artifacts, source_manifest_path, campaign_root)
    aggregate_long, fingerprints, aggregate_info = _validate_aggregate(
        campaign_root,
        selected_configs_sha256=selected_configs_sha256,
        source_manifest_sha256=source_manifest_sha256,
        selected=selected,
        expected_brightkite_split_sha256=(
            expected_brightkite_split_sha256
        ),
        input_artifacts=input_artifacts,
    )
    (
        stratified_raw,
        stratified_eg,
        recommendation_records,
        stratified_info,
    ) = _validate_stratified(
        campaign_root,
        selected_configs_sha256=selected_configs_sha256,
        source_manifest_sha256=source_manifest_sha256,
        selected=selected,
        expected_brightkite_split_sha256=(
            expected_brightkite_split_sha256
        ),
        aggregate_fingerprints=fingerprints,
        input_artifacts=input_artifacts,
    )

    aggregate_summary = _aggregate_metric_summary(aggregate_long)
    precision_run, precision_summary = _precision_summaries(stratified_raw)
    eg_summary = _eg_summary(stratified_eg)
    crosscheck = (
        _crosscheck_degree_eg(
            repo_root=repo_root,
            recommendation_records=recommendation_records,
            aggregate_long=aggregate_long,
            aggregate_fingerprints=fingerprints,
        )
        if enable_graph_crosscheck
        else {
            "status": "disabled",
            "reason": "disabled by caller",
            "comparisons": 0,
        }
    )

    stage = Path(
        tempfile.mkdtemp(prefix=".egfix_finalized.", dir=str(campaign_root))
    )
    try:
        output_frames = (
            (AGGREGATE_LONG_FILENAME, aggregate_long),
            (AGGREGATE_SUMMARY_FILENAME, aggregate_summary),
            (STRATIFIED_RAW_FILENAME, stratified_raw),
            (STRATIFIED_RUN_FILENAME, precision_run),
            (STRATIFIED_SUMMARY_FILENAME, precision_summary),
            (STRATIFIED_EG_RUN_FILENAME, stratified_eg),
            (STRATIFIED_EG_SUMMARY_FILENAME, eg_summary),
        )
        for filename, frame in output_frames:
            _atomic_write_frame(stage / filename, frame)
        figure_paths = _render_figures(
            raw=stratified_raw,
            eg_summary=eg_summary,
            figure_dir=stage / "figures",
        )

        input_artifacts.sort(key=lambda row: str(row["path"]))
        integrity: Dict[str, object] = {
            "schema_version": 1,
            "status": "complete",
            "campaign_root": str(campaign_root),
            "protocol": {
                "datasets": list(DATASETS),
                "aggregate_seeds": list(AGGREGATE_SEEDS),
                "stratified_seeds": list(STRATIFIED_SEEDS),
                "methods": list(MODEL_ORDER),
                "objectives": list(OBJECTIVES),
                "k": K,
                "split_seed": SPLIT_SEED,
                "aggregate_protocol": AGGREGATE_PROTOCOL,
                "evaluation_split": EVALUATION_SPLIT,
                "eg_graph_semantics": EG_GRAPH_SEMANTICS,
                "penalty_reduction": PENALTY_REDUCTION,
                "brightkite_temporal_split_sha256": (
                    expected_brightkite_split_sha256
                ),
            },
            "provenance": {
                "source_manifest_path": str(source_manifest_path),
                "source_manifest_sha256": source_manifest_sha256,
                "source_manifest_content_sha256": (
                    source_manifest_content_sha256
                ),
                "selected_configs_path": str(selected_configs_path),
                "selected_configs_sha256": selected_configs_sha256,
            },
            "aggregate": aggregate_info,
            "stratified": stratified_info,
            "degree_eg_crosscheck": crosscheck,
            "input_artifacts": input_artifacts,
            "outputs": _output_records(stage),
            "publication_pdfs": [
                path.relative_to(stage).as_posix() for path in figure_paths
            ],
        }
        integrity["integrity_payload_sha256"] = hashlib.sha256(
            canonical_json_bytes(integrity)
        ).hexdigest()
        _atomic_write_json(stage / INTEGRITY_FILENAME, integrity)
        os.replace(stage, final_dir)

        integrity_path = final_dir / INTEGRITY_FILENAME
        completion = {
            "schema_version": 1,
            "status": "complete",
            "finalized_directory": FINAL_DIRNAME,
            "integrity_file": (
                f"{FINAL_DIRNAME}/{INTEGRITY_FILENAME}"
            ),
            "integrity_file_sha256": sha256_file(integrity_path),
            "selected_configs_sha256": selected_configs_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "aggregate_shards": 50,
            "stratified_shards": 15,
            "publication_pdfs": 5,
        }
        _atomic_write_json(marker_path, completion)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise

    return {
        "status": "complete",
        "campaign_root": str(campaign_root),
        "finalized_directory": str(final_dir),
        "aggregate_shards": 50,
        "aggregate_objective_rows": len(aggregate_long),
        "stratified_shards": 15,
        "stratified_precision_rows": len(stratified_raw),
        "stratified_degree_eg_rows": len(stratified_eg),
        "publication_pdfs": 5,
        "degree_eg_crosscheck": crosscheck["status"],
        "completion_marker": str(marker_path),
    }


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly finalize the corrected-EG aggregate and source-"
            "stratified reruns"
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
        "--selected-configs",
        "--selected_configs",
        dest="selected_configs",
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
    summary = finalize_reruns(
        campaign_root=args.campaign_root,
        selected_configs_path=args.selected_configs,
        source_manifest_path=args.source_manifest,
        repo_root=args.repo_root,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    _main()
