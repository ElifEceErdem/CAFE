"""Run the strict CAFE-In finalizer through the reviewed recovery bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping

from jobs.egfix_sum import finalize_selection as base_finalizer
from jobs.egfix_sum.provenance import (
    canonical_json_bytes,
    sha256_file,
)
from jobs.egfix_sum_recovery.audit_completed import (
    ADOPTION_MANIFEST_FILENAME,
    ADOPT_TASK_IDS,
    EXISTING_TASK_IDS,
    RERUN_TASK_IDS,
)
from jobs.egfix_sum_recovery.provenance import (
    RECOVERY_MANIFEST_FILENAME,
    verify_original_manifest_for_finalizer,
    verify_recovery_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
LINEAGE_FILENAME = "cafe_in_sum_recovery_lineage.json"


class RecoveryFinalizationError(RuntimeError):
    """Raised when recovered selection lineage is incomplete."""


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryFinalizationError(
            f"Cannot read JSON file {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise RecoveryFinalizationError(f"{path} must contain an object")
    return value


def _verify_embedded_digest(
    document: Mapping[str, object], field: str, path: Path
) -> str:
    recorded = document.get(field)
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise RecoveryFinalizationError(
            f"{path}: missing or invalid {field}"
        )
    unsigned = dict(document)
    del unsigned[field]
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != recorded:
        raise RecoveryFinalizationError(
            f"{path}: {field} mismatch"
        )
    return recorded


def finalize_recovery(
    campaign_root: Path, repo_root: Path = REPO_ROOT
) -> Dict[str, object]:
    campaign = campaign_root.resolve()
    recovery_manifest_path = campaign / RECOVERY_MANIFEST_FILENAME
    recovery_manifest = verify_recovery_manifest(
        campaign, repo_root, manifest_path=recovery_manifest_path
    )

    adoption_path = campaign / ADOPTION_MANIFEST_FILENAME
    adoption = _read_json(adoption_path)
    adoption_sha = _verify_embedded_digest(
        adoption, "adoption_manifest_sha256", adoption_path
    )
    if adoption.get("recovery_manifest_sha256") != recovery_manifest[
        "recovery_manifest_sha256"
    ]:
        raise RecoveryFinalizationError(
            "Adoption manifest belongs to a different recovery manifest"
        )
    if adoption.get("validated_existing_task_ids") != list(
        EXISTING_TASK_IDS
    ):
        raise RecoveryFinalizationError(
            "Adoption manifest does not cover tasks 0..22"
        )
    if adoption.get("adopted_task_ids") != list(ADOPT_TASK_IDS):
        raise RecoveryFinalizationError(
            "Adoption manifest has the wrong adopted task set"
        )

    success_count = sum(
        1
        for _ in (campaign / "selection").rglob("_SUCCESS")
    )
    if success_count != 35:
        raise RecoveryFinalizationError(
            f"Expected 35 success markers, found {success_count}"
        )

    source_manifest_path = campaign / "source_manifest.json"
    original_verifier = base_finalizer.verify_source_manifest
    base_finalizer.verify_source_manifest = (
        verify_original_manifest_for_finalizer
    )
    try:
        selection_summary = base_finalizer.finalize_selection(
            selection_root=campaign / "selection",
            source_manifest_path=source_manifest_path,
            output_dir=campaign,
            repo_root=repo_root,
        )
    finally:
        base_finalizer.verify_source_manifest = original_verifier

    selected_path = campaign / base_finalizer.SELECTED_CSV_FILENAME
    input_manifest_path = campaign / base_finalizer.INPUT_MANIFEST_FILENAME
    selected_json_path = campaign / base_finalizer.SELECTED_JSON_FILENAME
    for path in (
        selected_path,
        input_manifest_path,
        selected_json_path,
    ):
        if not path.is_file():
            raise RecoveryFinalizationError(
                f"Strict finalizer did not create {path}"
            )

    rerun_artifacts = []
    for task_id in RERUN_TASK_IDS:
        task = base_finalizer.selection_task(task_id)
        result_path = (
            campaign
            / "selection"
            / str(task["dataset"])
            / str(task["variant"])
            / f"task_{task_id}"
            / "results.json"
        )
        rerun_artifacts.append(
            {
                "task_id": task_id,
                "dataset": task["dataset"],
                "penalty_value": task["penalty_value"],
                "result_path": str(result_path.resolve()),
                "result_sha256": sha256_file(result_path),
            }
        )

    lineage: Dict[str, object] = {
        "schema_version": 1,
        "manifest_kind": "cafe_in_sum_recovery_lineage",
        "recovery_manifest_path": str(recovery_manifest_path.resolve()),
        "recovery_manifest_sha256": recovery_manifest[
            "recovery_manifest_sha256"
        ],
        "adoption_manifest_path": str(adoption_path.resolve()),
        "adoption_manifest_sha256": adoption_sha,
        "original_source_manifest_path": str(
            source_manifest_path.resolve()
        ),
        "original_source_manifest_file_sha256": sha256_file(
            source_manifest_path
        ),
        "reused_task_ids": list(EXISTING_TASK_IDS),
        "adopted_task_ids": list(ADOPT_TASK_IDS),
        "rerun_task_ids": list(RERUN_TASK_IDS),
        "rerun_artifacts": rerun_artifacts,
        "selection_input_manifest_path": str(
            input_manifest_path.resolve()
        ),
        "selection_input_manifest_file_sha256": sha256_file(
            input_manifest_path
        ),
        "selected_configs_path": str(selected_path.resolve()),
        "selected_configs_sha256": sha256_file(selected_path),
        "selected_configs_json_sha256": sha256_file(selected_json_path),
        "selection_summary": selection_summary,
        "phase_two_submitted": False,
    }
    lineage["lineage_sha256"] = hashlib.sha256(
        canonical_json_bytes(lineage)
    ).hexdigest()
    lineage_path = campaign / LINEAGE_FILENAME
    if lineage_path.exists():
        if _read_json(lineage_path) != lineage:
            raise RecoveryFinalizationError(
                f"Existing recovery lineage differs: {lineage_path}"
            )
    else:
        base_finalizer._atomic_write_json(lineage_path, lineage)

    return {
        **selection_summary,
        "recovery_lineage": str(lineage_path),
        "recovery_lineage_sha256": lineage["lineage_sha256"],
    }


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize the recovered CAFE-In/sum selection"
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    print(
        json.dumps(
            finalize_recovery(args.campaign_root, args.repo_root),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    _main()
