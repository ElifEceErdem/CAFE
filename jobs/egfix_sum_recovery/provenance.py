"""Provenance bridge for resuming the interrupted CAFE-In/sum campaign.

The original campaign intentionally stopped when broad ``analysis/**/*.py``
provenance changed.  This bridge does not rewrite that immutable manifest.
Instead, it verifies every originally recorded source except three explicitly
identified, non-execution analysis utilities, verifies all data/runtime
artifacts, and separately freezes the additive recovery orchestration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

from jobs.egfix_sum import provenance as original_provenance
from jobs.egfix_sum.finalize_selection import (
    RUNTIME_ENV_DIGEST_FILENAME,
    RUNTIME_ENV_FILENAME,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RECOVERY_SCHEMA_VERSION = 1
RECOVERY_MANIFEST_FILENAME = "recovery_manifest.json"

# These are the exact paths that caused the false-positive campaign stop.
# None is imported by the runner, metric implementation, protocol, selection
# finalizer, or Slurm orchestration.
ALLOWED_NONEXECUTION_DRIFT_PATHS: Tuple[str, ...] = (
    "analysis/generate_acceptance_table.py",
    "analysis/generate_closeness_stratum_table.py",
    "analysis/generate_demographic_appendix_tables.py",
)

RECOVERY_REQUIRED_FILES: Tuple[str, ...] = (
    "jobs/egfix_sum_recovery/__init__.py",
    "jobs/egfix_sum_recovery/provenance.py",
    "jobs/egfix_sum_recovery/audit_completed.py",
    "jobs/egfix_sum_recovery/finalize_selection.py",
    "jobs/egfix_sum_recovery/tests.py",
    "jobs/egfix_sum_recovery/preflight.job",
    "jobs/egfix_sum_recovery/selection.job",
    "jobs/egfix_sum_recovery/finalize_selection.job",
    "jobs/egfix_sum_recovery/submit_recovery.sh",
)


class RecoveryProvenanceError(RuntimeError):
    """Raised when the interrupted campaign cannot be resumed safely."""


def _strict_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise RecoveryProvenanceError(f"Duplicate JSON key {key!r}")
        value[key] = item
    return value


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=_strict_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryProvenanceError(
            f"Cannot read JSON document {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise RecoveryProvenanceError(f"{path} must contain a JSON object")
    return value


def _manifest_digest(document: Mapping[str, object], field: str) -> str:
    recorded = document.get(field)
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise RecoveryProvenanceError(f"Missing or invalid {field}")
    unsigned = dict(document)
    del unsigned[field]
    actual = hashlib.sha256(
        original_provenance.canonical_json_bytes(unsigned)
    ).hexdigest()
    if actual != recorded:
        raise RecoveryProvenanceError(
            f"{field} mismatch: expected {recorded}, got {actual}"
        )
    return recorded


def _safe_repo_file(repo_root: Path, relative_path: str) -> Path:
    root = repo_root.resolve()
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise RecoveryProvenanceError(
            f"Unsafe repository-relative path {relative_path!r}"
        )
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RecoveryProvenanceError(
            f"Path escapes repository: {relative_path!r}"
        ) from exc
    if not resolved.is_file():
        raise RecoveryProvenanceError(
            f"Required recovery file is missing: {relative_path}"
        )
    return resolved


def _original_manifest_path(campaign_root: Path) -> Path:
    return campaign_root.resolve() / "source_manifest.json"


def _load_original_manifest(campaign_root: Path) -> Dict[str, object]:
    path = _original_manifest_path(campaign_root)
    manifest = dict(_read_json(path))
    try:
        original_provenance._validate_manifest_digest(manifest)
    except Exception as exc:
        raise RecoveryProvenanceError(
            f"Original source manifest is invalid: {exc}"
        ) from exc
    if manifest.get("manifest_kind") != "cafe_in_sum_source_manifest":
        raise RecoveryProvenanceError(
            "Original manifest is not a CAFE-In/sum manifest"
        )
    protocol = manifest.get("protocol")
    if (
        not isinstance(protocol, dict)
        or protocol.get("profile") != "cafe_in_sum_v1"
        or protocol.get("penalty_reduction") != "sum"
        or protocol.get("selection_task_count") != 35
    ):
        raise RecoveryProvenanceError(
            "Original manifest has the wrong selection protocol"
        )
    return manifest


def _current_original_source_set(repo_root: Path) -> Tuple[str, ...]:
    try:
        return original_provenance.resolve_source_paths(repo_root)
    except Exception as exc:
        raise RecoveryProvenanceError(
            f"Cannot resolve original source set: {exc}"
        ) from exc


def verify_original_execution_state(
    campaign_root: Path, repo_root: Path = REPO_ROOT
) -> Dict[str, object]:
    """Verify all original inputs except the explicit non-execution allowlist."""

    root = repo_root.resolve()
    manifest = _load_original_manifest(campaign_root)
    recorded_files = manifest.get("files")
    if not isinstance(recorded_files, dict):
        raise RecoveryProvenanceError(
            "Original manifest files field is invalid"
        )
    recorded_paths = set(recorded_files)
    current_paths = set(_current_original_source_set(root))
    allowed = set(ALLOWED_NONEXECUTION_DRIFT_PATHS)

    unexpected_added = sorted((current_paths - recorded_paths) - allowed)
    unexpected_removed = sorted((recorded_paths - current_paths) - allowed)
    if unexpected_added or unexpected_removed:
        raise RecoveryProvenanceError(
            "Unexpected source-set drift outside the recovery allowlist; "
            f"added={unexpected_added}, removed={unexpected_removed}"
        )

    changed = []
    verified_count = 0
    for relative_path in sorted((recorded_paths & current_paths) - allowed):
        path = _safe_repo_file(root, relative_path)
        record = recorded_files[relative_path]
        if not isinstance(record, dict):
            raise RecoveryProvenanceError(
                f"Malformed original source record for {relative_path}"
            )
        current_sha = original_provenance.sha256_file(path)
        current_size = path.stat().st_size
        if (
            current_sha != record.get("sha256")
            or current_size != record.get("size_bytes")
        ):
            changed.append(relative_path)
        verified_count += 1
    if changed:
        raise RecoveryProvenanceError(
            "Execution-relevant source changed after the original manifest: "
            + ", ".join(changed)
        )

    data_artifacts = manifest.get("data_artifacts")
    if not isinstance(data_artifacts, dict) or set(data_artifacts) != {
        "ego-facebook",
        "oklahoma",
        "unc",
        "pokec_n",
        "brightkite",
    }:
        raise RecoveryProvenanceError(
            "Original data-artifact manifest is incomplete"
        )
    for dataset, record in sorted(data_artifacts.items()):
        if not isinstance(record, dict):
            raise RecoveryProvenanceError(
                f"Malformed data record for {dataset}"
            )
        path = _safe_repo_file(root, str(record.get("path")))
        if (
            original_provenance.sha256_file(path) != record.get("sha256")
            or path.stat().st_size != record.get("size_bytes")
        ):
            raise RecoveryProvenanceError(
                f"Processed dataset changed: {dataset}"
            )

    split = manifest.get("brightkite_temporal_split")
    if not isinstance(split, dict):
        raise RecoveryProvenanceError(
            "Brightkite split record is missing"
        )
    split_path = _safe_repo_file(root, str(split.get("path")))
    if (
        original_provenance.sha256_file(split_path) != split.get("sha256")
        or split_path.stat().st_size != split.get("size_bytes")
    ):
        raise RecoveryProvenanceError(
            "Brightkite temporal split changed"
        )

    campaign = campaign_root.resolve()
    runtime_path = campaign / RUNTIME_ENV_FILENAME
    runtime_digest_path = campaign / RUNTIME_ENV_DIGEST_FILENAME
    if not runtime_path.is_file() or not runtime_digest_path.is_file():
        raise RecoveryProvenanceError(
            "Runtime-environment attestation is missing"
        )
    fields = runtime_digest_path.read_text(
        encoding="utf-8"
    ).strip().split()
    runtime_sha = original_provenance.sha256_file(runtime_path)
    if (
        len(fields) != 2
        or fields[1] != RUNTIME_ENV_FILENAME
        or fields[0] != runtime_sha
    ):
        raise RecoveryProvenanceError(
            "Runtime-environment attestation is invalid"
        )

    ignored_state = {}
    for relative_path in ALLOWED_NONEXECUTION_DRIFT_PATHS:
        path = root / relative_path
        recorded = recorded_files.get(relative_path)
        if path.is_file():
            ignored_state[relative_path] = {
                "current_status": "present",
                "current_sha256": original_provenance.sha256_file(path),
                "current_size_bytes": path.stat().st_size,
                "recorded_sha256": (
                    recorded.get("sha256")
                    if isinstance(recorded, dict)
                    else None
                ),
            }
        else:
            ignored_state[relative_path] = {
                "current_status": "absent",
                "recorded_sha256": (
                    recorded.get("sha256")
                    if isinstance(recorded, dict)
                    else None
                ),
            }

    return {
        "original_manifest": manifest,
        "original_manifest_path": str(
            _original_manifest_path(campaign).resolve()
        ),
        "original_manifest_file_sha256": (
            original_provenance.sha256_file(
                _original_manifest_path(campaign)
            )
        ),
        "original_manifest_content_sha256": manifest["manifest_sha256"],
        "verified_original_source_files": verified_count,
        "runtime_environment_sha256": runtime_sha,
        "ignored_nonexecution_drift": ignored_state,
    }


def _recovery_file_records(repo_root: Path) -> Dict[str, object]:
    records = {}
    for relative_path in RECOVERY_REQUIRED_FILES:
        path = _safe_repo_file(repo_root, relative_path)
        records[relative_path] = {
            "sha256": original_provenance.sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return records


def build_recovery_manifest(
    campaign_root: Path, repo_root: Path = REPO_ROOT
) -> Dict[str, object]:
    state = verify_original_execution_state(campaign_root, repo_root)
    payload: Dict[str, object] = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "manifest_kind": "cafe_in_sum_recovery_manifest",
        "original_source_manifest": {
            "path": state["original_manifest_path"],
            "file_sha256": state["original_manifest_file_sha256"],
            "content_sha256": state[
                "original_manifest_content_sha256"
            ],
        },
        "allowed_nonexecution_drift_paths": list(
            ALLOWED_NONEXECUTION_DRIFT_PATHS
        ),
        "ignored_nonexecution_drift_at_creation": state[
            "ignored_nonexecution_drift"
        ],
        "verified_original_source_files": state[
            "verified_original_source_files"
        ],
        "runtime_environment_sha256": state[
            "runtime_environment_sha256"
        ],
        "recovery_files": _recovery_file_records(repo_root),
        "recovery_plan": {
            "validated_existing_task_ids": list(range(23)),
            "adopt_task_ids": [13, 17, 18, 19, 20, 21, 22],
            "rerun_task_ids": list(range(23, 35)),
            "final_selection_count": 5,
            "phase_two_submitted": False,
        },
    }
    payload["recovery_manifest_sha256"] = hashlib.sha256(
        original_provenance.canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
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


def ensure_recovery_manifest(
    campaign_root: Path,
    repo_root: Path = REPO_ROOT,
    manifest_path: Path | None = None,
) -> Dict[str, object]:
    path = (
        campaign_root.resolve() / RECOVERY_MANIFEST_FILENAME
        if manifest_path is None
        else manifest_path.resolve()
    )
    if path.exists():
        return verify_recovery_manifest(
            campaign_root, repo_root, manifest_path=path
        )
    document = build_recovery_manifest(campaign_root, repo_root)
    _atomic_write_json(path, document)
    return verify_recovery_manifest(
        campaign_root, repo_root, manifest_path=path
    )


def verify_recovery_manifest(
    campaign_root: Path,
    repo_root: Path = REPO_ROOT,
    manifest_path: Path | None = None,
) -> Dict[str, object]:
    campaign = campaign_root.resolve()
    path = (
        campaign / RECOVERY_MANIFEST_FILENAME
        if manifest_path is None
        else manifest_path.resolve()
    )
    document = dict(_read_json(path))
    _manifest_digest(document, "recovery_manifest_sha256")
    if document.get("manifest_kind") != "cafe_in_sum_recovery_manifest":
        raise RecoveryProvenanceError(
            "Not a CAFE-In/sum recovery manifest"
        )
    if document.get("allowed_nonexecution_drift_paths") != list(
        ALLOWED_NONEXECUTION_DRIFT_PATHS
    ):
        raise RecoveryProvenanceError(
            "Recovery drift allowlist differs from the reviewed policy"
        )

    state = verify_original_execution_state(campaign, repo_root)
    original_record = document.get("original_source_manifest")
    if not isinstance(original_record, dict):
        raise RecoveryProvenanceError(
            "Recovery manifest lacks original-source lineage"
        )
    expected_original = {
        "path": state["original_manifest_path"],
        "file_sha256": state["original_manifest_file_sha256"],
        "content_sha256": state[
            "original_manifest_content_sha256"
        ],
    }
    if original_record != expected_original:
        raise RecoveryProvenanceError(
            "Original source-manifest lineage changed"
        )
    if document.get("runtime_environment_sha256") != state[
        "runtime_environment_sha256"
    ]:
        raise RecoveryProvenanceError(
            "Runtime environment differs from recovery manifest"
        )

    recorded_recovery_files = document.get("recovery_files")
    current_recovery_files = _recovery_file_records(repo_root)
    if recorded_recovery_files != current_recovery_files:
        raise RecoveryProvenanceError(
            "Recovery orchestration source changed after authorization"
        )
    return document


def verify_original_manifest_for_finalizer(
    manifest_path: Path,
    repo_root: Path,
    **_: object,
) -> Dict[str, object]:
    """Adapter matching the strict selection finalizer's verifier signature."""

    campaign_root = manifest_path.resolve().parent
    verify_recovery_manifest(campaign_root, repo_root)
    expected_path = _original_manifest_path(campaign_root).resolve()
    if manifest_path.resolve() != expected_path:
        raise RecoveryProvenanceError(
            f"Finalizer must use original manifest {expected_path}"
        )
    return _load_original_manifest(campaign_root)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Create or verify the sum-selection recovery manifest"
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--manifest-path", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.check_only:
        document = build_recovery_manifest(
            args.campaign_root, args.repo_root
        )
        action = "prospective recovery state validated"
    elif args.create:
        document = ensure_recovery_manifest(
            args.campaign_root,
            args.repo_root,
            manifest_path=args.manifest_path,
        )
        action = "recovery manifest created"
    else:
        document = verify_recovery_manifest(
            args.campaign_root,
            args.repo_root,
            manifest_path=args.manifest_path,
        )
        action = "recovery manifest verified"
    print(
        json.dumps(
            {
                "status": action,
                "recovery_manifest_sha256": document[
                    "recovery_manifest_sha256"
                ],
                "plan": document["recovery_plan"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    _main()
