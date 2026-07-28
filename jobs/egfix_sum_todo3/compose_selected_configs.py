"""Compose and verify the ten-row execution configuration for TODO3.

The two upstream CSVs remain immutable.  Five CAFE-In rows are taken from the
reviewed sum-reduction selection and five CAFE-Pre rows from the earlier
corrected-EG selection.  The resulting execution CSV uses ``sum`` uniformly
because the existing runners load both rows for a dataset and enforce one
reduction value.  Row-level origin fields retain the original reduction and
complete selection lineage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from experiments.egfix_protocol import (
    BETA_VALUES,
    DATASETS,
    EG_GRAPH_SEMANTICS,
    INPROC_PENALTIES,
    MOBILITY_VALUES,
    PREPROC_PENALTIES,
)
from jobs.egfix_sum_todo3.protocol import (
    BATCH_SIZE,
    LOSS_FUNC,
    NUM_NEG,
    PENALTY_REDUCTION,
    PROFILE,
    SELECTION_SPLIT,
    VARIANTS,
    protocol_sha256,
)
from jobs.egfix_sum_todo3.provenance import (
    REPO_ROOT,
    Todo3ProvenanceError,
    canonical_json_bytes,
    read_json,
    sha256_file,
    verify_source_manifest,
)


SELECTED_FILENAME = "selected_configs.csv"
SELECTED_DIGEST_FILENAME = "selected_configs.sha256"
LINEAGE_FILENAME = "selected_configs_lineage.json"
SCHEMA_VERSION = 1

OUTPUT_FIELDS: Tuple[str, ...] = (
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
    "selection_profile",
    "selection_penalty_reduction",
    "execution_penalty_reduction",
    "reduction_operational_for_variant",
    "loss_func",
    "num_neg",
    "batch_size",
    "runtime_environment_sha256",
    "selection_runtime_environment_sha256",
    "selection_input_manifest_sha256",
    "origin_role",
    "origin_campaign_path",
    "origin_selected_configs_path",
    "origin_selected_configs_sha256",
    "origin_selected_configs_json_path",
    "origin_selected_configs_json_sha256",
    "origin_source_manifest_path",
    "origin_source_manifest_file_sha256",
    "origin_source_manifest_content_sha256",
    "origin_selection_input_manifest_path",
    "origin_selection_input_manifest_file_sha256",
    "origin_selection_input_manifest_content_sha256",
)

NUMERIC_RESULT_FIELDS: Tuple[str, ...] = (
    "Precision@10_mean",
    "Equality_Gain@10_mean",
    "Pagerank_EG@10_mean",
    "Closeness_EG@10_mean",
    "asf_score",
)


class CompositeConfigError(RuntimeError):
    """Raised when the two frozen selections cannot be composed exactly."""


def _strict_csv(path: Path) -> List[Dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            if not header or len(header) != len(set(header)):
                raise CompositeConfigError(
                    f"{path}: missing or duplicate CSV header"
                )
            if any(not field for field in header):
                raise CompositeConfigError(f"{path}: empty CSV column name")
            rows = []
            for line_number, values in enumerate(reader, start=2):
                if len(values) != len(header):
                    raise CompositeConfigError(
                        f"{path}:{line_number}: expected {len(header)} "
                        f"fields, found {len(values)}"
                    )
                rows.append(dict(zip(header, values)))
    except (OSError, UnicodeDecodeError, csv.Error, StopIteration) as exc:
        if isinstance(exc, CompositeConfigError):
            raise
        raise CompositeConfigError(f"Cannot read CSV {path}: {exc}") from exc
    if not rows:
        raise CompositeConfigError(f"{path}: no data rows")
    return rows


def _finite(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CompositeConfigError(
            f"{label} must be a finite number, got {value!r}"
        ) from exc
    if not math.isfinite(result):
        raise CompositeConfigError(
            f"{label} must be finite, got {value!r}"
        )
    return result


def _close_to_grid(value: float, grid: Sequence[float]) -> bool:
    return any(
        math.isclose(value, float(item), rel_tol=0.0, abs_tol=1e-12)
        for item in grid
    )


def _require_fields(
    rows: Sequence[Mapping[str, str]],
    required: Sequence[str],
    path: Path,
) -> None:
    missing = set(required) - set(rows[0])
    if missing:
        raise CompositeConfigError(
            f"{path}: missing columns {sorted(missing)}"
        )


def _selection_map(
    rows: Sequence[Mapping[str, str]],
    *,
    path: Path,
    expected_variants: Sequence[str],
) -> Dict[Tuple[str, str], Mapping[str, str]]:
    expected = {
        (dataset, variant)
        for dataset in DATASETS
        for variant in expected_variants
    }
    keys = [(row.get("dataset"), row.get("variant")) for row in rows]
    if set(keys) != expected or len(keys) != len(set(keys)):
        raise CompositeConfigError(
            f"{path}: rows are not exactly dataset x "
            f"{tuple(expected_variants)}"
        )
    return {
        (str(row["dataset"]), str(row["variant"])): row for row in rows
    }


def _validate_origin_row(
    row: Mapping[str, str],
    *,
    path: Path,
    dataset: str,
    variant: str,
    expected_reduction: str,
    expected_source_file_sha: str,
    expected_source_content_sha: str,
) -> None:
    label = f"{path}:{dataset}/{variant}"
    required_literals = {
        "dataset": dataset,
        "variant": variant,
        "penalization_mode": (
            "inproc" if variant == "cafe_in" else "preproc"
        ),
        "penalty_reduction": expected_reduction,
        "evaluation_split": SELECTION_SPLIT,
        "eg_graph_semantics": EG_GRAPH_SEMANTICS,
        "selection_runs": "3",
        "source_manifest_sha256": expected_source_file_sha,
        "source_manifest_content_sha256": expected_source_content_sha,
    }
    for field, expected in required_literals.items():
        if str(row.get(field, "")) != expected:
            raise CompositeConfigError(
                f"{label}: {field} must equal {expected!r}, "
                f"got {row.get(field)!r}"
            )

    beta = _finite(row.get("beta"), f"{label}:beta")
    mobility = _finite(
        row.get("lambda_mobility"), f"{label}:lambda_mobility"
    )
    if not _close_to_grid(beta, BETA_VALUES):
        raise CompositeConfigError(f"{label}: beta is outside frozen grid")
    if not _close_to_grid(mobility, MOBILITY_VALUES):
        raise CompositeConfigError(
            f"{label}: mobility is outside frozen grid"
        )
    if "beta_ra" in row and row["beta_ra"]:
        beta_ra = _finite(row["beta_ra"], f"{label}:beta_ra")
        if not math.isclose(beta, beta_ra, rel_tol=0.0, abs_tol=1e-12):
            raise CompositeConfigError(
                f"{label}: beta and beta_ra do not match"
            )
    if variant == "cafe_in":
        penalty = _finite(
            row.get("lambda_fair"), f"{label}:lambda_fair"
        )
        if not _close_to_grid(penalty, INPROC_PENALTIES):
            raise CompositeConfigError(
                f"{label}: lambda_fair is outside frozen grid"
            )
        if str(row.get("preproc_alpha", "")).strip():
            raise CompositeConfigError(
                f"{label}: preproc_alpha must be blank"
            )
    else:
        penalty = _finite(
            row.get("preproc_alpha"), f"{label}:preproc_alpha"
        )
        if not _close_to_grid(penalty, PREPROC_PENALTIES):
            raise CompositeConfigError(
                f"{label}: preproc_alpha is outside frozen grid"
            )
        if str(row.get("lambda_fair", "")).strip():
            raise CompositeConfigError(
                f"{label}: lambda_fair must be blank"
            )
    for field in NUMERIC_RESULT_FIELDS:
        _finite(row.get(field), f"{label}:{field}")
    if not str(row.get("config_key", "")):
        raise CompositeConfigError(f"{label}: config_key must not be blank")


def _json_selection_map(
    document: Mapping[str, object], path: Path
) -> Dict[Tuple[str, str], Mapping[str, object]]:
    raw = document.get("selections")
    if not isinstance(raw, list) or not all(
        isinstance(item, dict) for item in raw
    ):
        raise CompositeConfigError(
            f"{path}: selections must be a list of objects"
        )
    keys = [(item.get("dataset"), item.get("variant")) for item in raw]
    if len(keys) != len(set(keys)):
        raise CompositeConfigError(f"{path}: duplicate selected JSON row")
    return {
        (str(item["dataset"]), str(item["variant"])): item for item in raw
    }


def _validate_csv_json_agreement(
    row: Mapping[str, str],
    json_row: Mapping[str, object],
    *,
    label: str,
) -> None:
    text_fields = (
        "dataset",
        "variant",
        "penalization_mode",
        "config_key",
        "penalty_reduction",
        "evaluation_split",
        "eg_graph_semantics",
    )
    for field in text_fields:
        if str(row.get(field, "")) != str(json_row.get(field, "")):
            raise CompositeConfigError(
                f"{label}: CSV/JSON mismatch in {field}"
            )
    numeric_fields = (
        "lambda_fair",
        "preproc_alpha",
        "beta",
        "beta_ra",
        "lambda_mobility",
        *NUMERIC_RESULT_FIELDS,
        "selection_runs",
    )
    for field in numeric_fields:
        csv_value = row.get(field, "")
        json_value = json_row.get(field)
        if (csv_value is None or str(csv_value).strip() == "") and (
            json_value is None
        ):
            continue
        actual = _finite(csv_value, f"{label}:CSV:{field}")
        expected = _finite(json_value, f"{label}:JSON:{field}")
        if not math.isclose(
            actual, expected, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise CompositeConfigError(
                f"{label}: CSV/JSON mismatch in {field}"
            )


def _record_sha(record: Mapping[str, object], label: str) -> str:
    value = record.get("sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise CompositeConfigError(f"{label}: invalid artifact hash")
    return value


def _origin_path(record: Mapping[str, object], label: str) -> str:
    value = record.get("path")
    if not isinstance(value, str) or not value:
        raise CompositeConfigError(f"{label}: invalid artifact path")
    return value


def compose_rows(
    *,
    source_manifest_path: Path,
    sum_campaign: Path,
    pre_campaign: Path,
    repo_root: Path = REPO_ROOT,
) -> Tuple[List[Dict[str, str]], Mapping[str, object]]:
    """Return the exact ten execution rows and verified source manifest."""

    try:
        manifest = verify_source_manifest(
            source_manifest_path,
            sum_campaign=sum_campaign,
            pre_campaign=pre_campaign,
            repo_root=repo_root,
        )
    except Todo3ProvenanceError as exc:
        raise CompositeConfigError(str(exc)) from exc
    source_file_sha = sha256_file(source_manifest_path)
    source_content_sha = str(manifest["manifest_sha256"])
    upstream = manifest["upstream"]
    runtime_sha = str(upstream["runtime_environment_sha256"])

    origins = {
        "cafe_in": upstream["cafe_in"],
        "cafe_pre": upstream["cafe_pre"],
    }
    sum_csv = Path(
        str(origins["cafe_in"]["selected_configs"]["path"])
    )
    pre_csv = Path(
        str(origins["cafe_pre"]["selected_configs"]["path"])
    )
    sum_rows = _strict_csv(sum_csv)
    pre_rows = _strict_csv(pre_csv)
    required = (
        "dataset",
        "variant",
        "penalization_mode",
        "lambda_fair",
        "preproc_alpha",
        "beta",
        "lambda_mobility",
        "config_key",
        *NUMERIC_RESULT_FIELDS,
        "source_manifest_sha256",
        "source_manifest_content_sha256",
        "penalty_reduction",
        "evaluation_split",
        "eg_graph_semantics",
        "selection_runs",
    )
    _require_fields(sum_rows, required, sum_csv)
    _require_fields(pre_rows, required, pre_csv)
    sum_map = _selection_map(
        sum_rows,
        path=sum_csv,
        expected_variants=("cafe_in",),
    )
    pre_map = _selection_map(
        pre_rows,
        path=pre_csv,
        expected_variants=("cafe_in", "cafe_pre"),
    )

    sum_json_path = Path(
        str(origins["cafe_in"]["selected_configs_json"]["path"])
    )
    pre_json_path = Path(
        str(origins["cafe_pre"]["selected_configs_json"]["path"])
    )
    sum_json_map = _json_selection_map(
        read_json(sum_json_path), sum_json_path
    )
    pre_json_map = _json_selection_map(
        read_json(pre_json_path), pre_json_path
    )

    output = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            origin = origins[variant]
            input_row = (
                sum_map[(dataset, variant)]
                if variant == "cafe_in"
                else pre_map[(dataset, variant)]
            )
            input_csv = sum_csv if variant == "cafe_in" else pre_csv
            input_json_map = (
                sum_json_map if variant == "cafe_in" else pre_json_map
            )
            selection_reduction = (
                "sum" if variant == "cafe_in" else "mean"
            )
            origin_source_file_sha = _record_sha(
                origin["source_manifest"],
                f"{variant}:source manifest",
            )
            origin_source_content_sha = str(
                origin["source_manifest_content_sha256"]
            )
            _validate_origin_row(
                input_row,
                path=input_csv,
                dataset=dataset,
                variant=variant,
                expected_reduction=selection_reduction,
                expected_source_file_sha=origin_source_file_sha,
                expected_source_content_sha=origin_source_content_sha,
            )
            _validate_csv_json_agreement(
                input_row,
                input_json_map[(dataset, variant)],
                label=f"{dataset}/{variant}",
            )

            input_manifest_record = origin[
                "selection_input_manifest"
            ]
            input_manifest_content_sha = str(
                origin["selection_input_manifest_content_sha256"]
            )
            selected_json_record = origin["selected_configs_json"]
            row = {
                "dataset": dataset,
                "variant": variant,
                "penalization_mode": str(
                    input_row["penalization_mode"]
                ),
                "lambda_fair": str(input_row["lambda_fair"]),
                "preproc_alpha": str(input_row["preproc_alpha"]),
                "beta": str(input_row["beta"]),
                "beta_ra": str(
                    input_row.get("beta_ra") or input_row["beta"]
                ),
                "lambda_mobility": str(
                    input_row["lambda_mobility"]
                ),
                "config_key": str(input_row["config_key"]),
                **{
                    field: str(input_row[field])
                    for field in NUMERIC_RESULT_FIELDS
                },
                "source_manifest_sha256": source_file_sha,
                "source_manifest_content_sha256": source_content_sha,
                "penalty_reduction": PENALTY_REDUCTION,
                "evaluation_split": SELECTION_SPLIT,
                "eg_graph_semantics": EG_GRAPH_SEMANTICS,
                "selection_runs": "3",
                "selection_profile": str(origin["selection_profile"]),
                "selection_penalty_reduction": selection_reduction,
                "execution_penalty_reduction": PENALTY_REDUCTION,
                "reduction_operational_for_variant": (
                    "true" if variant == "cafe_in" else "false"
                ),
                "loss_func": LOSS_FUNC,
                "num_neg": str(NUM_NEG),
                "batch_size": str(BATCH_SIZE),
                "runtime_environment_sha256": runtime_sha,
                "selection_runtime_environment_sha256": (
                    runtime_sha if variant == "cafe_in" else ""
                ),
                "selection_input_manifest_sha256": (
                    input_manifest_content_sha
                ),
                "origin_role": (
                    "corrected_sum_cafe_in"
                    if variant == "cafe_in"
                    else "frozen_cafe_pre"
                ),
                "origin_campaign_path": str(
                    upstream[
                        "sum_campaign"
                        if variant == "cafe_in"
                        else "pre_campaign"
                    ]
                ),
                "origin_selected_configs_path": _origin_path(
                    origin["selected_configs"],
                    f"{variant}:selected configs",
                ),
                "origin_selected_configs_sha256": _record_sha(
                    origin["selected_configs"],
                    f"{variant}:selected configs",
                ),
                "origin_selected_configs_json_path": _origin_path(
                    selected_json_record,
                    f"{variant}:selected JSON",
                ),
                "origin_selected_configs_json_sha256": _record_sha(
                    selected_json_record,
                    f"{variant}:selected JSON",
                ),
                "origin_source_manifest_path": _origin_path(
                    origin["source_manifest"],
                    f"{variant}:source manifest",
                ),
                "origin_source_manifest_file_sha256": (
                    origin_source_file_sha
                ),
                "origin_source_manifest_content_sha256": (
                    origin_source_content_sha
                ),
                "origin_selection_input_manifest_path": _origin_path(
                    input_manifest_record,
                    f"{variant}:selection input",
                ),
                "origin_selection_input_manifest_file_sha256": (
                    _record_sha(
                        input_manifest_record,
                        f"{variant}:selection input",
                    )
                ),
                "origin_selection_input_manifest_content_sha256": (
                    input_manifest_content_sha
                ),
            }
            if set(row) != set(OUTPUT_FIELDS):
                raise CompositeConfigError(
                    f"Internal output schema mismatch for {dataset}/{variant}"
                )
            output.append(row)

    if len(output) != len(DATASETS) * len(VARIANTS):
        raise CompositeConfigError("Composite row count is not exactly ten")
    return output, manifest


def _csv_bytes(rows: Sequence[Mapping[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(OUTPUT_FIELDS),
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _lineage_document(
    *,
    output_dir: Path,
    selected_path: Path,
    selected_sha: str,
    source_manifest_path: Path,
    source_manifest: Mapping[str, object],
    rows: Sequence[Mapping[str, str]],
) -> Dict[str, object]:
    upstream = source_manifest["upstream"]
    document: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_kind": "rq3_todo3_composite_selected_configs",
        "profile": PROFILE,
        "protocol_sha256": protocol_sha256(),
        "output_directory": str(output_dir.resolve()),
        "source_manifest_path": str(source_manifest_path.resolve()),
        "source_manifest_file_sha256": sha256_file(
            source_manifest_path
        ),
        "source_manifest_content_sha256": source_manifest[
            "manifest_sha256"
        ],
        "selected_configs_path": str(selected_path.resolve()),
        "selected_configs_sha256": selected_sha,
        "row_count": len(rows),
        "dataset_order": list(DATASETS),
        "variant_order": list(VARIANTS),
        "execution_penalty_reduction": PENALTY_REDUCTION,
        "runtime_environment_sha256": upstream[
            "runtime_environment_sha256"
        ],
        "origins": {
            "cafe_in": upstream["cafe_in"],
            "cafe_pre": upstream["cafe_pre"],
        },
        "compatibility_evidence": {
            "shared_source_files": upstream["shared_source_files"],
            "sum_only_source_files": upstream[
                "sum_only_source_files"
            ],
            "fixed_split_fingerprints": upstream[
                "fixed_split_fingerprints"
            ],
            "cafe_pre_execution_reduction_is_nonoperational": True,
            "historical_phase_two_artifacts_reused": False,
        },
        "rows": [
            {
                "dataset": row["dataset"],
                "variant": row["variant"],
                "config_key": row["config_key"],
                "selection_profile": row["selection_profile"],
                "selection_penalty_reduction": row[
                    "selection_penalty_reduction"
                ],
                "origin_selected_configs_sha256": row[
                    "origin_selected_configs_sha256"
                ],
            }
            for row in rows
        ],
    }
    document["lineage_sha256"] = hashlib.sha256(
        canonical_json_bytes(document)
    ).hexdigest()
    return document


def write_composite_configs(
    *,
    output_dir: Path,
    source_manifest_path: Path,
    sum_campaign: Path,
    pre_campaign: Path,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, object]:
    root = output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    selected_path = root / SELECTED_FILENAME
    digest_path = root / SELECTED_DIGEST_FILENAME
    lineage_path = root / LINEAGE_FILENAME

    rows, manifest = compose_rows(
        source_manifest_path=source_manifest_path,
        sum_campaign=sum_campaign,
        pre_campaign=pre_campaign,
        repo_root=repo_root,
    )
    payload = _csv_bytes(rows)
    selected_sha = hashlib.sha256(payload).hexdigest()
    digest_payload = (
        f"{selected_sha}  {SELECTED_FILENAME}\n".encode("utf-8")
    )
    lineage = _lineage_document(
        output_dir=root,
        selected_path=selected_path,
        selected_sha=selected_sha,
        source_manifest_path=source_manifest_path,
        source_manifest=manifest,
        rows=rows,
    )
    lineage_payload = (
        json.dumps(
            lineage,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    # The three artifacts are independently atomic, but a process can still
    # stop between them.  Resume only when every artifact already present is
    # byte-for-byte the deterministic value; otherwise preserve it and fail.
    # This makes the launcher's pre-submission preparation safely retryable
    # without silently replacing an ambiguous file.
    for path, expected in (
        (selected_path, payload),
        (digest_path, digest_payload),
        (lineage_path, lineage_payload),
    ):
        if path.is_symlink():
            raise CompositeConfigError(
                f"Refusing symlinked composite artifact: {path}"
            )
        if path.exists():
            if not path.is_file() or path.read_bytes() != expected:
                raise CompositeConfigError(
                    f"Existing composite artifact differs from exact "
                    f"recomposition: {path}"
                )
        else:
            _atomic_write_bytes(path, expected)

    return verify_composite_configs(
        output_dir=root,
        source_manifest_path=source_manifest_path,
        sum_campaign=sum_campaign,
        pre_campaign=pre_campaign,
        repo_root=repo_root,
    )


def verify_composite_configs(
    *,
    output_dir: Path,
    source_manifest_path: Path,
    sum_campaign: Path,
    pre_campaign: Path,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, object]:
    root = output_dir.resolve()
    selected_path = root / SELECTED_FILENAME
    digest_path = root / SELECTED_DIGEST_FILENAME
    lineage_path = root / LINEAGE_FILENAME
    for path in (selected_path, digest_path, lineage_path):
        if not path.is_file():
            raise CompositeConfigError(
                f"Composite configuration artifact is missing: {path}"
            )

    rows, manifest = compose_rows(
        source_manifest_path=source_manifest_path,
        sum_campaign=sum_campaign,
        pre_campaign=pre_campaign,
        repo_root=repo_root,
    )
    expected_payload = _csv_bytes(rows)
    actual_payload = selected_path.read_bytes()
    if actual_payload != expected_payload:
        raise CompositeConfigError(
            "Composite selected_configs.csv differs from exact "
            "recomposition of its two frozen origins"
        )
    selected_sha = hashlib.sha256(actual_payload).hexdigest()
    if digest_path.read_text(encoding="utf-8") != (
        f"{selected_sha}  {SELECTED_FILENAME}\n"
    ):
        raise CompositeConfigError(
            f"Invalid composite checksum sidecar: {digest_path}"
        )

    expected_lineage = _lineage_document(
        output_dir=root,
        selected_path=selected_path,
        selected_sha=selected_sha,
        source_manifest_path=source_manifest_path,
        source_manifest=manifest,
        rows=rows,
    )
    if read_json(lineage_path) != expected_lineage:
        raise CompositeConfigError(
            "Composite selection lineage differs from exact recomputation"
        )
    return {
        "status": "verified",
        "selected_configs_path": str(selected_path),
        "selected_configs_sha256": selected_sha,
        "lineage_path": str(lineage_path),
        "lineage_file_sha256": sha256_file(lineage_path),
        "lineage_content_sha256": expected_lineage["lineage_sha256"],
        "rows": len(rows),
        "cafe_in_rows": 5,
        "cafe_pre_rows": 5,
        "execution_penalty_reduction": PENALTY_REDUCTION,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Compose or verify the dual-origin TODO3 configs"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--sum-campaign", type=Path, required=True)
    parser.add_argument("--pre-campaign", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.create:
        summary = write_composite_configs(
            output_dir=args.output_dir,
            source_manifest_path=args.source_manifest,
            sum_campaign=args.sum_campaign,
            pre_campaign=args.pre_campaign,
            repo_root=args.repo_root,
        )
    else:
        summary = verify_composite_configs(
            output_dir=args.output_dir,
            source_manifest_path=args.source_manifest,
            sum_campaign=args.sum_campaign,
            pre_campaign=args.pre_campaign,
            repo_root=args.repo_root,
        )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    _main()
