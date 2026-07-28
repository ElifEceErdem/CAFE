"""Audit and explicitly adopt completed artifacts stopped by provenance drift."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple
from unittest.mock import patch

from analysis.egfix_finalize_selection import SelectionValidationError
from jobs.egfix_sum import finalize_selection as base_finalizer
from jobs.egfix_sum.protocol import (
    BETA_VALUES,
    CONFIGS_PER_PENALTY,
    DATASETS,
    LAMBDA_FAIR_VALUES,
    MOBILITY_VALUES,
    PENALTY_NAME,
    SELECTION_RUNS,
    expected_penalty_config_keys,
    parse_config_key,
    selection_task,
)
from jobs.egfix_sum.provenance import canonical_json_bytes, sha256_file
from jobs.egfix_sum_recovery.provenance import (
    RECOVERY_MANIFEST_FILENAME,
    verify_recovery_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EXISTING_TASK_IDS: Tuple[int, ...] = tuple(range(23))
ADOPT_TASK_IDS: Tuple[int, ...] = (13, 17, 18, 19, 20, 21, 22)
RERUN_TASK_IDS: Tuple[int, ...] = tuple(range(23, 35))
ADOPTION_MANIFEST_FILENAME = "recovery_adoption_manifest.json"
ADOPTION_SCHEMA_VERSION = 1


class RecoveryAuditError(RuntimeError):
    """Raised when a pre-existing result cannot be adopted exactly."""


def _task_directory(campaign_root: Path, task_id: int) -> Path:
    task = selection_task(task_id)
    return (
        campaign_root.resolve()
        / "selection"
        / str(task["dataset"])
        / str(task["variant"])
        / f"task_{task_id}"
    )


def _selection_job_id(campaign_root: Path) -> str:
    path = campaign_root.resolve() / "job_ids.selection.tsv"
    try:
        rows = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RecoveryAuditError(f"Cannot read {path}: {exc}") from exc
    matches = []
    for line in rows[1:]:
        fields = line.split("\t")
        if len(fields) >= 2 and fields[0] == "selection":
            matches.append(fields[1])
    if len(matches) != 1 or not matches[0].isdigit():
        raise RecoveryAuditError(
            f"{path}: expected one numeric selection job ID"
        )
    return matches[0]


def _call_task_provenance_without_success(
    result_path: Path, **kwargs: object
):
    success_path = result_path.parent / "_SUCCESS"
    original_is_file = Path.is_file

    def audit_is_file(path: Path) -> bool:
        if path == success_path:
            return True
        return original_is_file(path)

    with patch.object(Path, "is_file", audit_is_file):
        return base_finalizer._require_task_provenance(
            result_path, **kwargs
        )


def _validate_result_artifact(
    campaign_root: Path,
    task_id: int,
    *,
    repo_root: Path,
    source_manifest_path: Path,
    source_manifest_file_sha256: str,
    runtime_environment_sha256: str,
) -> Dict[str, object]:
    task = selection_task(task_id)
    task_dir = _task_directory(campaign_root, task_id)
    result_path = task_dir / "results.json"
    if not result_path.is_file():
        raise RecoveryAuditError(
            f"Task {task_id} is missing {result_path}"
        )
    partials = sorted(task_dir.glob("*.partial.json"))
    if partials:
        raise RecoveryAuditError(
            f"Task {task_id} retains partial artifacts: {partials}"
        )

    try:
        document = base_finalizer._read_json(result_path)
        info = document.get("experiment_info")
        if not isinstance(info, dict):
            raise SelectionValidationError(
                f"{result_path}: experiment_info must be an object"
            )
        dataset = base_finalizer._require_metadata(
            info,
            path=result_path,
            source_manifest_file_sha256=source_manifest_file_sha256,
            expected_brightkite_split_sha256=(
                base_finalizer.BRIGHTKITE_TEMPORAL_SPLIT_SHA256
            ),
        )
        if dataset != task["dataset"]:
            raise SelectionValidationError(
                f"{result_path}: task maps to {task['dataset']}, got {dataset}"
            )

        lambda_results = document.get("lambda_results")
        if not isinstance(lambda_results, dict):
            raise SelectionValidationError(
                f"{result_path}: lambda_results must be an object"
            )
        if len(lambda_results) != CONFIGS_PER_PENALTY:
            raise SelectionValidationError(
                f"{result_path}: expected {CONFIGS_PER_PENALTY} configs"
            )
        parsed = {}
        for config_key in lambda_results:
            parsed[config_key] = parse_config_key(config_key)
        penalties = {
            float(config["penalty_value"]) for config in parsed.values()
        }
        if penalties != {float(task["penalty_value"])}:
            raise SelectionValidationError(
                f"{result_path}: penalty set {penalties} differs from task"
            )
        penalty = float(task["penalty_value"])
        expected_keys = set(expected_penalty_config_keys(penalty))
        if set(lambda_results) != expected_keys:
            raise SelectionValidationError(
                f"{result_path}: configuration keys are incomplete"
            )
        expected_grid = {
            PENALTY_NAME: [penalty],
            "beta": list(BETA_VALUES),
            "lambda_mobility": list(MOBILITY_VALUES),
        }
        if info.get("grid") != expected_grid:
            raise SelectionValidationError(
                f"{result_path}: grid metadata differs from {expected_grid}"
            )

        individual_runs = document.get("individual_runs")
        if (
            not isinstance(individual_runs, list)
            or len(individual_runs) != SELECTION_RUNS
            or not all(isinstance(run, dict) for run in individual_runs)
        ):
            raise SelectionValidationError(
                f"{result_path}: expected {SELECTION_RUNS} run objects"
            )
        expected_run_keys = expected_keys | {"baseline"}
        for run_index, run in enumerate(individual_runs):
            if set(run) != expected_run_keys:
                raise SelectionValidationError(
                    f"{result_path}: run {run_index} keys are incomplete"
                )
        for config_key in expected_penalty_config_keys(penalty):
            result = lambda_results[config_key]
            if not isinstance(result, dict):
                raise SelectionValidationError(
                    f"{result_path}: invalid result for {config_key}"
                )
            base_finalizer._validate_stats_and_runs(
                path=result_path,
                config_key=config_key,
                result=result,
                individual_runs=individual_runs,
            )

        provenance_kwargs = {
            "dataset": dataset,
            "penalty": penalty,
            "source_manifest_file_sha256": (
                source_manifest_file_sha256
            ),
            "runtime_environment_sha256": (
                runtime_environment_sha256
            ),
            "source_manifest_path": source_manifest_path,
            "repo_root": repo_root.resolve(),
            "split_relative_path": (
                base_finalizer.BRIGHTKITE_TEMPORAL_SPLIT
            ),
        }
        success_path = task_dir / "_SUCCESS"
        if success_path.is_file():
            provenance = base_finalizer._require_task_provenance(
                result_path, **provenance_kwargs
            )
        else:
            provenance = _call_task_provenance_without_success(
                result_path, **provenance_kwargs
            )
        if int(provenance[-1]) != task_id:
            raise SelectionValidationError(
                f"{result_path}: provenance task ID mismatch"
            )
    except (SelectionValidationError, ValueError, OSError) as exc:
        raise RecoveryAuditError(
            f"Task {task_id} failed strict artifact validation: {exc}"
        ) from exc

    return {
        "task_id": task_id,
        "dataset": task["dataset"],
        "penalty_value": float(task["penalty_value"]),
        "result_path": str(result_path.resolve()),
        "result_sha256": sha256_file(result_path),
        "result_size_bytes": result_path.stat().st_size,
        "task_env_sha256": sha256_file(task_dir / "task.env"),
        "command_sha256": sha256_file(task_dir / "command.sh"),
        "result_digest_sha256": sha256_file(
            task_dir / "results.sha256"
        ),
        "split_fingerprint": info["split_fingerprint"],
        # This records the reviewed pre-recovery state, not the marker's
        # current state.  Keeping it invariant makes adoption safely
        # resumable after some or all recovery markers have been written.
        "original_success_marker": task_id not in ADOPT_TASK_IDS,
    }


def _atomic_write_text(path: Path, text: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _adoption_document(
    *,
    campaign_root: Path,
    recovery_manifest: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    document: Dict[str, object] = {
        "schema_version": ADOPTION_SCHEMA_VERSION,
        "manifest_kind": "cafe_in_sum_recovery_adoption",
        "campaign_root": str(campaign_root.resolve()),
        "recovery_manifest_sha256": recovery_manifest[
            "recovery_manifest_sha256"
        ],
        "validated_existing_task_ids": list(EXISTING_TASK_IDS),
        "adopted_task_ids": list(ADOPT_TASK_IDS),
        "rerun_task_ids": list(RERUN_TASK_IDS),
        "artifacts": list(records),
    }
    document["adoption_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(document)
    ).hexdigest()
    return document


def _verify_adoption_document(
    path: Path, expected: Mapping[str, object]
) -> None:
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryAuditError(
            f"Cannot read adoption manifest {path}: {exc}"
        ) from exc
    if actual != expected:
        raise RecoveryAuditError(
            f"Existing adoption manifest differs from strict re-audit: {path}"
        )


def audit_completed(
    campaign_root: Path,
    *,
    repo_root: Path = REPO_ROOT,
    recovery_manifest_path: Path | None = None,
    adopt: bool = False,
) -> Dict[str, object]:
    campaign = campaign_root.resolve()
    recovery_manifest = verify_recovery_manifest(
        campaign,
        repo_root,
        manifest_path=recovery_manifest_path,
    )
    source_manifest_path = campaign / "source_manifest.json"
    source_manifest_file_sha256 = sha256_file(source_manifest_path)
    runtime_environment_sha256 = (
        base_finalizer._runtime_environment_digest(source_manifest_path)
    )

    result_ids = set()
    for path in (campaign / "selection").rglob("results.json"):
        if not path.parent.name.startswith("task_"):
            raise RecoveryAuditError(f"Unexpected result path {path}")
        try:
            result_ids.add(int(path.parent.name.removeprefix("task_")))
        except ValueError as exc:
            raise RecoveryAuditError(
                f"Malformed task directory {path.parent}"
            ) from exc
    expected_result_ids = (
        set(range(35)) if adopt else set(EXISTING_TASK_IDS)
    )
    if result_ids != expected_result_ids:
        raise RecoveryAuditError(
            f"Expected result tasks {sorted(expected_result_ids)} exactly; "
            f"found={sorted(result_ids)}"
        )

    job_id = _selection_job_id(campaign)
    records = []
    for task_id in EXISTING_TASK_IDS:
        success = (_task_directory(campaign, task_id) / "_SUCCESS").is_file()
        if task_id in ADOPT_TASK_IDS:
            recovered_marker = (
                _task_directory(campaign, task_id)
                / "_RECOVERED_SUCCESS"
            )
            if success and not recovered_marker.is_file():
                raise RecoveryAuditError(
                    f"Task {task_id} has an unexplained success marker"
                )
            if not success:
                log_path = (
                    campaign
                    / "logs"
                    / f"selection_{job_id}_{task_id}.out"
                )
                try:
                    log_text = log_path.read_text(
                        encoding="utf-8", errors="replace"
                    )
                except OSError as exc:
                    raise RecoveryAuditError(
                        f"Cannot read original task log {log_path}: {exc}"
                    ) from exc
                required_log_text = (
                    "EGFIX_RUN_COMPLETE",
                    "Source manifest no longer matches the repository",
                )
                if not all(item in log_text for item in required_log_text):
                    raise RecoveryAuditError(
                        f"Task {task_id} did not stop solely at the "
                        "post-run broad provenance check"
                    )
        elif not success:
            raise RecoveryAuditError(
                f"Originally successful task {task_id} lacks _SUCCESS"
            )
        records.append(
            _validate_result_artifact(
                campaign,
                task_id,
                repo_root=repo_root,
                source_manifest_path=source_manifest_path,
                source_manifest_file_sha256=(
                    source_manifest_file_sha256
                ),
                runtime_environment_sha256=(
                    runtime_environment_sha256
                ),
            )
        )

    adoption_document = _adoption_document(
        campaign_root=campaign,
        recovery_manifest=recovery_manifest,
        records=records,
    )
    adoption_path = campaign / ADOPTION_MANIFEST_FILENAME
    if adopt:
        if adoption_path.exists():
            _verify_adoption_document(
                adoption_path, adoption_document
            )
        else:
            original_provenance = __import__(
                "jobs.egfix_sum.provenance",
                fromlist=["_atomic_write_json"],
            )
            original_provenance._atomic_write_json(
                adoption_path, adoption_document
            )
        marker_text = (
            "adoption_manifest_sha256="
            f"{adoption_document['adoption_manifest_sha256']}\n"
        )
        for task_id in ADOPT_TASK_IDS:
            task_dir = _task_directory(campaign, task_id)
            recovered_path = task_dir / "_RECOVERED_SUCCESS"
            success_path = task_dir / "_SUCCESS"
            if recovered_path.exists():
                if recovered_path.read_text(
                    encoding="utf-8"
                ) != marker_text:
                    raise RecoveryAuditError(
                        f"Recovery marker differs for task {task_id}"
                    )
            else:
                _atomic_write_text(recovered_path, marker_text)
            if not success_path.exists():
                _atomic_write_text(success_path, marker_text)

    return {
        "validated_existing_results": len(records),
        "original_successes": len(EXISTING_TASK_IDS) - len(ADOPT_TASK_IDS),
        "adopted_results": len(ADOPT_TASK_IDS),
        "rerun_task_ids": list(RERUN_TASK_IDS),
        "adoption_manifest_sha256": adoption_document[
            "adoption_manifest_sha256"
        ],
        "adopted": adopt,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Strictly audit completed artifacts for recovery"
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--recovery-manifest", type=Path)
    parser.add_argument("--adopt", action="store_true")
    args = parser.parse_args()
    summary = audit_completed(
        args.campaign_root,
        repo_root=args.repo_root,
        recovery_manifest_path=args.recovery_manifest,
        adopt=args.adopt,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    _main()
