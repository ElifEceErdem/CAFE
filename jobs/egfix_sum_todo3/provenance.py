"""Immutable dual-selection provenance for the isolated TODO3 reruns.

The phase combines two already-frozen configuration selections.  This module
does not rewrite either one.  It verifies their complete lineage, proves that
their shared scientific source files and fixed splits agree, freezes the
additive TODO3 orchestration, and exposes the narrow verifier adapter used by
the unchanged strict rerun finalizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from jobs.egfix_sum import provenance as sum_provenance
from jobs.egfix_sum_recovery.provenance import (
    verify_recovery_manifest,
)
from jobs.egfix_sum_todo3.protocol import (
    PROFILE,
    SCHEMA_VERSION,
    protocol_document,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST_FILENAME = "source_manifest.json"

TODO3_REQUIRED_FILES: Tuple[str, ...] = (
    "jobs/egfix_sum_todo3/__init__.py",
    "jobs/egfix_sum_todo3/protocol.py",
    "jobs/egfix_sum_todo3/provenance.py",
    "jobs/egfix_sum_todo3/compose_selected_configs.py",
    "jobs/egfix_sum_todo3/target_exposure.py",
    "jobs/egfix_sum_todo3/finalize_reruns.py",
    "jobs/egfix_sum_todo3/tests.py",
    "jobs/egfix_sum_todo3/preflight.job",
    "jobs/egfix_sum_todo3/aggregate.job",
    "jobs/egfix_sum_todo3/stratified.job",
    "jobs/egfix_sum_todo3/finalize_reruns.job",
    "jobs/egfix_sum_todo3/submit_reruns.sh",
    "jobs/egfix_sum_todo3/README.md",
)

DATASETS: Tuple[str, ...] = (
    "ego-facebook",
    "oklahoma",
    "unc",
    "pokec_n",
    "brightkite",
)


class Todo3ProvenanceError(RuntimeError):
    """Raised when TODO3 source or upstream selection lineage is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _strict_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise Todo3ProvenanceError(f"Duplicate JSON key {key!r}")
        value[key] = item
    return value


def read_json(path: Path) -> Mapping[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=_strict_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise Todo3ProvenanceError(
            f"Cannot read JSON document {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise Todo3ProvenanceError(f"{path} must contain a JSON object")
    return value


def _require_sha(value: object, label: str) -> str:
    rendered = str(value)
    if len(rendered) != 64:
        raise Todo3ProvenanceError(f"{label} must be a SHA256 digest")
    try:
        int(rendered, 16)
    except ValueError as exc:
        raise Todo3ProvenanceError(
            f"{label} must be a SHA256 digest"
        ) from exc
    return rendered.lower()


def verify_embedded_digest(
    document: Mapping[str, object], field: str, path: Path
) -> str:
    recorded = _require_sha(document.get(field), f"{path}:{field}")
    unsigned = dict(document)
    del unsigned[field]
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != recorded:
        raise Todo3ProvenanceError(
            f"{path}: {field} mismatch: expected {recorded}, got {actual}"
        )
    return recorded


def _safe_repo_file(repo_root: Path, relative_path: str) -> Path:
    root = repo_root.resolve()
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise Todo3ProvenanceError(
            f"Unsafe repository-relative path: {relative_path!r}"
        )
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise Todo3ProvenanceError(
            f"Path escapes repository root: {relative_path!r}"
        ) from exc
    if not path.is_file():
        raise Todo3ProvenanceError(
            f"Required TODO3 source is missing: {relative_path}"
        )
    return path


def _safe_campaign(
    repo_root: Path, campaign_root: Path, label: str
) -> Path:
    repo = repo_root.resolve()
    expected_parent = (repo / "results_egfix").resolve()
    campaign = campaign_root.resolve()
    try:
        relative = campaign.relative_to(expected_parent)
    except ValueError as exc:
        raise Todo3ProvenanceError(
            f"{label} must be below {expected_parent}: {campaign}"
        ) from exc
    if len(relative.parts) != 1 or not campaign.is_dir():
        raise Todo3ProvenanceError(
            f"{label} must identify one existing campaign directory"
        )
    return campaign


def _file_record(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise Todo3ProvenanceError(f"Required artifact is missing: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _verify_sha_sidecar(
    sidecar: Path, artifact: Path, expected_name: str
) -> str:
    try:
        fields = sidecar.read_text(encoding="utf-8").strip().split()
    except OSError as exc:
        raise Todo3ProvenanceError(
            f"Cannot read checksum sidecar {sidecar}: {exc}"
        ) from exc
    actual = sha256_file(artifact)
    if (
        len(fields) != 2
        or fields[0] != actual
        or fields[1] != expected_name
    ):
        raise Todo3ProvenanceError(
            f"Invalid checksum sidecar {sidecar} for {artifact}"
        )
    return actual


def _selection_fingerprints(
    document: Mapping[str, object], path: Path
) -> Dict[str, str]:
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list):
        raise Todo3ProvenanceError(f"{path}: artifacts must be a list")
    by_dataset: Dict[str, set] = {dataset: set() for dataset in DATASETS}
    for index, record in enumerate(artifacts):
        if not isinstance(record, dict):
            raise Todo3ProvenanceError(
                f"{path}: artifact {index} must be an object"
            )
        dataset = record.get("dataset")
        if dataset not in by_dataset:
            raise Todo3ProvenanceError(
                f"{path}: unknown artifact dataset {dataset!r}"
            )
        by_dataset[str(dataset)].add(
            _require_sha(
                record.get("split_fingerprint"),
                f"{path}:artifact {index}:split_fingerprint",
            )
        )
    result = {}
    for dataset, values in by_dataset.items():
        if len(values) != 1:
            raise Todo3ProvenanceError(
                f"{path}: {dataset} has {len(values)} split fingerprints"
            )
        result[dataset] = next(iter(values))
    return result


def _validate_selection_input_manifest(
    path: Path,
    *,
    expected_kind: str,
    expected_source_file_sha256: str,
    expected_source_content_sha256: str,
    expected_reduction: str,
) -> Tuple[Mapping[str, object], str, Dict[str, str]]:
    document = read_json(path)
    digest = verify_embedded_digest(
        document, "input_manifest_sha256", path
    )
    if document.get("manifest_kind") != expected_kind:
        raise Todo3ProvenanceError(
            f"{path}: wrong selection-input manifest kind"
        )
    if document.get("source_manifest_sha256") != (
        expected_source_file_sha256
    ):
        raise Todo3ProvenanceError(
            f"{path}: source-manifest file hash mismatch"
        )
    if document.get("source_manifest_content_sha256") != (
        expected_source_content_sha256
    ):
        raise Todo3ProvenanceError(
            f"{path}: source-manifest content hash mismatch"
        )
    protocol = document.get("protocol")
    if not isinstance(protocol, dict):
        raise Todo3ProvenanceError(f"{path}: protocol must be an object")
    required = {
        "penalty_reduction": expected_reduction,
        "evaluation_split": "valid",
        "selection_runs": 3,
        "split_seed": 42,
        "eg_graph_semantics": "simple_undirected_set_union",
        "test_metrics_evaluated": False,
    }
    for field, expected in required.items():
        if protocol.get(field) != expected:
            raise Todo3ProvenanceError(
                f"{path}: protocol {field} must equal {expected!r}"
            )
    if protocol.get("datasets") != list(DATASETS):
        raise Todo3ProvenanceError(
            f"{path}: protocol dataset order is not frozen"
        )
    return document, digest, _selection_fingerprints(document, path)


def _validate_selected_json(
    path: Path,
    *,
    expected_rows: int,
    expected_variants: Sequence[str],
    expected_source_file_sha256: str,
    expected_source_content_sha256: str,
    expected_input_digest: str,
) -> Mapping[str, object]:
    document = read_json(path)
    selections = document.get("selections")
    if not isinstance(selections, list) or len(selections) != expected_rows:
        raise Todo3ProvenanceError(
            f"{path}: expected {expected_rows} selected rows"
        )
    variants = [record.get("variant") for record in selections]
    if set(variants) != set(expected_variants):
        raise Todo3ProvenanceError(
            f"{path}: selected variant set is invalid"
        )
    if document.get("source_manifest_sha256") != (
        expected_source_file_sha256
    ):
        raise Todo3ProvenanceError(
            f"{path}: selected JSON source file hash mismatch"
        )
    if document.get("source_manifest_content_sha256") != (
        expected_source_content_sha256
    ):
        raise Todo3ProvenanceError(
            f"{path}: selected JSON source content hash mismatch"
        )
    if document.get("selection_input_manifest_sha256") != (
        expected_input_digest
    ):
        raise Todo3ProvenanceError(
            f"{path}: selected JSON input-manifest digest mismatch"
        )
    return document


def _source_manifest_digest(
    path: Path, *, expected_kind: str
) -> Tuple[Mapping[str, object], str, str]:
    document = read_json(path)
    content_digest = verify_embedded_digest(
        document, "manifest_sha256", path
    )
    if document.get("manifest_kind") != expected_kind:
        raise Todo3ProvenanceError(
            f"{path}: expected manifest kind {expected_kind!r}"
        )
    return document, sha256_file(path), content_digest


def verify_upstream_state(
    sum_campaign: Path,
    pre_campaign: Path,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, object]:
    """Strictly verify both frozen selections and their compatibility."""

    repo = repo_root.resolve()
    sum_root = _safe_campaign(repo, sum_campaign, "CAFE-In campaign")
    pre_root = _safe_campaign(repo, pre_campaign, "CAFE-Pre campaign")
    if sum_root == pre_root:
        raise Todo3ProvenanceError(
            "CAFE-In and CAFE-Pre selections must be separate campaigns"
        )

    for marker in (
        sum_root / "_CAFE_IN_SUM_SELECTION_COMPLETE",
        pre_root / "_SELECTION_COMPLETE",
    ):
        if not marker.is_file():
            raise Todo3ProvenanceError(
                f"Selection completion marker is missing: {marker}"
            )

    recovery_path = sum_root / "recovery_manifest.json"
    recovery = verify_recovery_manifest(
        sum_root, repo, manifest_path=recovery_path
    )
    recovery_file_sha = sha256_file(recovery_path)
    recovery_content_sha = _require_sha(
        recovery.get("recovery_manifest_sha256"),
        f"{recovery_path}:recovery_manifest_sha256",
    )

    sum_source_path = sum_root / "source_manifest.json"
    sum_source, sum_source_file_sha, sum_source_content_sha = (
        _source_manifest_digest(
            sum_source_path,
            expected_kind="cafe_in_sum_source_manifest",
        )
    )
    original_record = recovery.get("original_source_manifest")
    if not isinstance(original_record, dict) or (
        original_record.get("file_sha256") != sum_source_file_sha
        or original_record.get("content_sha256")
        != sum_source_content_sha
    ):
        raise Todo3ProvenanceError(
            "Recovery manifest does not bind the current sum source manifest"
        )

    sum_selected_path = sum_root / "selected_cafe_in_configs.csv"
    sum_selected_sidecar = sum_root / "selected_cafe_in_configs.sha256"
    sum_selected_sha = _verify_sha_sidecar(
        sum_selected_sidecar,
        sum_selected_path,
        "selected_cafe_in_configs.csv",
    )
    sum_input_path = sum_root / "cafe_in_selection_input_manifest.json"
    (
        sum_input,
        sum_input_content_sha,
        sum_fingerprints,
    ) = _validate_selection_input_manifest(
        sum_input_path,
        expected_kind="cafe_in_sum_selection_inputs",
        expected_source_file_sha256=sum_source_file_sha,
        expected_source_content_sha256=sum_source_content_sha,
        expected_reduction="sum",
    )
    sum_selected_json_path = sum_root / "selected_cafe_in_configs.json"
    _validate_selected_json(
        sum_selected_json_path,
        expected_rows=5,
        expected_variants=("cafe_in",),
        expected_source_file_sha256=sum_source_file_sha,
        expected_source_content_sha256=sum_source_content_sha,
        expected_input_digest=sum_input_content_sha,
    )

    lineage_path = sum_root / "cafe_in_sum_recovery_lineage.json"
    lineage = read_json(lineage_path)
    lineage_content_sha = verify_embedded_digest(
        lineage, "lineage_sha256", lineage_path
    )
    required_lineage = {
        "manifest_kind": "cafe_in_sum_recovery_lineage",
        "recovery_manifest_sha256": recovery_content_sha,
        "original_source_manifest_file_sha256": sum_source_file_sha,
        "selected_configs_sha256": sum_selected_sha,
        "selected_configs_json_sha256": sha256_file(
            sum_selected_json_path
        ),
        "selection_input_manifest_file_sha256": sha256_file(
            sum_input_path
        ),
        "phase_two_submitted": False,
    }
    for field, expected in required_lineage.items():
        if lineage.get(field) != expected:
            raise Todo3ProvenanceError(
                f"{lineage_path}: {field} does not match reviewed lineage"
            )

    adoption_path = sum_root / "recovery_adoption_manifest.json"
    adoption = read_json(adoption_path)
    adoption_content_sha = verify_embedded_digest(
        adoption, "adoption_manifest_sha256", adoption_path
    )
    if (
        lineage.get("adoption_manifest_sha256") != adoption_content_sha
        or adoption.get("recovery_manifest_sha256")
        != recovery_content_sha
    ):
        raise Todo3ProvenanceError(
            "Recovery adoption and final selection lineage are disconnected"
        )

    pre_source_path = pre_root / "source_manifest.json"
    pre_source, pre_source_file_sha, pre_source_content_sha = (
        _source_manifest_digest(
            pre_source_path, expected_kind="egfix_source_manifest"
        )
    )
    pre_selected_path = pre_root / "selected_configs.csv"
    pre_selected_sidecar = pre_root / "selected_configs.sha256"
    pre_selected_sha = _verify_sha_sidecar(
        pre_selected_sidecar, pre_selected_path, "selected_configs.csv"
    )
    pre_input_path = pre_root / "selection_input_manifest.json"
    (
        pre_input,
        pre_input_content_sha,
        pre_fingerprints,
    ) = _validate_selection_input_manifest(
        pre_input_path,
        expected_kind="egfix_selection_inputs",
        expected_source_file_sha256=pre_source_file_sha,
        expected_source_content_sha256=pre_source_content_sha,
        expected_reduction="mean",
    )
    pre_selected_json_path = pre_root / "selected_configs.json"
    _validate_selected_json(
        pre_selected_json_path,
        expected_rows=10,
        expected_variants=("cafe_in", "cafe_pre"),
        expected_source_file_sha256=pre_source_file_sha,
        expected_source_content_sha256=pre_source_content_sha,
        expected_input_digest=pre_input_content_sha,
    )

    sum_files = sum_source.get("files")
    pre_files = pre_source.get("files")
    if not isinstance(sum_files, dict) or not isinstance(pre_files, dict):
        raise Todo3ProvenanceError(
            "Upstream source manifests have malformed file records"
        )
    if not set(pre_files).issubset(set(sum_files)):
        raise Todo3ProvenanceError(
            "The CAFE-Pre source set is not a subset of the reviewed sum set"
        )
    changed_shared = [
        name
        for name, record in pre_files.items()
        if sum_files.get(name) != record
    ]
    if changed_shared:
        raise Todo3ProvenanceError(
            "Shared selection source files differ: "
            + ", ".join(changed_shared[:10])
        )
    if (
        pre_source.get("brightkite_temporal_split")
        != sum_source.get("brightkite_temporal_split")
    ):
        raise Todo3ProvenanceError(
            "The two selections used different Brightkite split artifacts"
        )
    if pre_fingerprints != sum_fingerprints:
        raise Todo3ProvenanceError(
            "The two selections have different fixed-split fingerprints"
        )

    runtime_path = sum_root / "hl_gnn.conda-explicit.txt"
    runtime_sidecar = sum_root / "hl_gnn.conda-explicit.sha256"
    runtime_sha = _verify_sha_sidecar(
        runtime_sidecar, runtime_path, "hl_gnn.conda-explicit.txt"
    )
    if recovery.get("runtime_environment_sha256") != runtime_sha:
        raise Todo3ProvenanceError(
            "Recovery manifest runtime digest does not match its attestation"
        )

    data_records = sum_source.get("data_artifacts")
    if not isinstance(data_records, dict) or set(data_records) != set(
        DATASETS
    ):
        raise Todo3ProvenanceError(
            "The sum source manifest lacks the five processed datasets"
        )

    return {
        "sum_campaign": str(sum_root),
        "pre_campaign": str(pre_root),
        "fixed_split_fingerprints": sum_fingerprints,
        "shared_source_files": len(pre_files),
        "sum_only_source_files": sorted(set(sum_files) - set(pre_files)),
        "runtime_environment_sha256": runtime_sha,
        "runtime_environment_path": str(runtime_path.resolve()),
        "data_artifacts": data_records,
        "brightkite_temporal_split": sum_source[
            "brightkite_temporal_split"
        ],
        "cafe_in": {
            "selection_profile": "cafe_in_sum_v1",
            "selection_penalty_reduction": "sum",
            "source_manifest": _file_record(sum_source_path),
            "source_manifest_content_sha256": sum_source_content_sha,
            "selected_configs": _file_record(sum_selected_path),
            "selected_configs_sidecar": _file_record(
                sum_selected_sidecar
            ),
            "selected_configs_json": _file_record(
                sum_selected_json_path
            ),
            "selection_input_manifest": _file_record(sum_input_path),
            "selection_input_manifest_content_sha256": (
                sum_input_content_sha
            ),
            "recovery_manifest": _file_record(recovery_path),
            "recovery_manifest_content_sha256": recovery_content_sha,
            "recovery_adoption_manifest": _file_record(adoption_path),
            "recovery_adoption_manifest_content_sha256": (
                adoption_content_sha
            ),
            "recovery_lineage": _file_record(lineage_path),
            "recovery_lineage_content_sha256": lineage_content_sha,
            "completion_marker": _file_record(
                sum_root / "_CAFE_IN_SUM_SELECTION_COMPLETE"
            ),
        },
        "cafe_pre": {
            "selection_profile": "corrected_eg_mean_v1",
            "selection_penalty_reduction": "mean",
            "selection_runtime_attestation": None,
            "selection_runtime_limitation": (
                "The earlier selection did not freeze a conda-explicit "
                "snapshot; TODO3 execution uses the attested sum runtime."
            ),
            "source_manifest": _file_record(pre_source_path),
            "source_manifest_content_sha256": pre_source_content_sha,
            "selected_configs": _file_record(pre_selected_path),
            "selected_configs_sidecar": _file_record(
                pre_selected_sidecar
            ),
            "selected_configs_json": _file_record(
                pre_selected_json_path
            ),
            "selection_input_manifest": _file_record(pre_input_path),
            "selection_input_manifest_content_sha256": (
                pre_input_content_sha
            ),
            "completion_marker": _file_record(
                pre_root / "_SELECTION_COMPLETE"
            ),
            "historical_phase_two_artifacts_reused": False,
        },
    }


def _todo3_file_records(repo_root: Path) -> Dict[str, object]:
    records = {}
    for relative_path in TODO3_REQUIRED_FILES:
        path = _safe_repo_file(repo_root, relative_path)
        records[relative_path] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return records


def build_source_manifest(
    *,
    sum_campaign: Path,
    pre_campaign: Path,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, object]:
    """Build the prospective TODO3 source manifest."""

    upstream = verify_upstream_state(
        sum_campaign, pre_campaign, repo_root
    )
    protocol = protocol_document()
    payload: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_kind": "rq3_degree_eg_sum_todo3_source_manifest",
        "profile": PROFILE,
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(
            canonical_json_bytes(protocol)
        ).hexdigest(),
        "todo3_files": _todo3_file_records(repo_root.resolve()),
        "upstream": upstream,
        "runtime_environment": {
            "path": upstream["runtime_environment_path"],
            "sha256": upstream["runtime_environment_sha256"],
            "source": "CAFE-In/sum selection campaign",
        },
        "data_artifacts": upstream["data_artifacts"],
        "brightkite_temporal_split": upstream[
            "brightkite_temporal_split"
        ],
    }
    payload["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
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


def ensure_source_manifest(
    manifest_path: Path,
    *,
    sum_campaign: Path,
    pre_campaign: Path,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, object]:
    path = manifest_path.resolve()
    if path.exists():
        return verify_source_manifest(
            path,
            sum_campaign=sum_campaign,
            pre_campaign=pre_campaign,
            repo_root=repo_root,
        )
    document = build_source_manifest(
        sum_campaign=sum_campaign,
        pre_campaign=pre_campaign,
        repo_root=repo_root,
    )
    _atomic_write_json(path, document)
    return verify_source_manifest(
        path,
        sum_campaign=sum_campaign,
        pre_campaign=pre_campaign,
        repo_root=repo_root,
    )


def verify_source_manifest(
    manifest_path: Path,
    *,
    sum_campaign: Optional[Path] = None,
    pre_campaign: Optional[Path] = None,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, object]:
    path = manifest_path.resolve()
    document = dict(read_json(path))
    verify_embedded_digest(document, "manifest_sha256", path)
    upstream = document.get("upstream")
    if not isinstance(upstream, dict):
        raise Todo3ProvenanceError(
            f"{path}: upstream provenance must be an object"
        )
    recorded_sum = Path(str(upstream.get("sum_campaign")))
    recorded_pre = Path(str(upstream.get("pre_campaign")))
    if sum_campaign is not None and recorded_sum.resolve() != (
        sum_campaign.resolve()
    ):
        raise Todo3ProvenanceError(
            f"{path}: CAFE-In campaign path differs from request"
        )
    if pre_campaign is not None and recorded_pre.resolve() != (
        pre_campaign.resolve()
    ):
        raise Todo3ProvenanceError(
            f"{path}: CAFE-Pre campaign path differs from request"
        )
    expected = build_source_manifest(
        sum_campaign=recorded_sum,
        pre_campaign=recorded_pre,
        repo_root=repo_root,
    )
    if document != expected:
        if document.get("todo3_files") != expected["todo3_files"]:
            raise Todo3ProvenanceError(
                "TODO3 orchestration changed after the source manifest "
                "was created"
            )
        if document.get("upstream") != expected["upstream"]:
            raise Todo3ProvenanceError(
                "An upstream selection, recovery record, data artifact, "
                "runtime attestation, or split changed"
            )
        if document.get("protocol") != expected["protocol"]:
            raise Todo3ProvenanceError(
                "The frozen TODO3 protocol changed"
            )
        raise Todo3ProvenanceError(
            "TODO3 source manifest differs from its exact recomputation"
        )
    return expected


def verify_manifest_for_base_finalizer(
    manifest_path: Path,
    repo_root: Path,
    **_: object,
) -> Dict[str, object]:
    """Adapter for ``analysis.egfix_finalize_reruns``.

    The base finalizer accepts a verifier with the older preflight signature.
    TODO3 records both upstream paths inside its own immutable manifest, so no
    caller-supplied lineage can replace them here.
    """

    return verify_source_manifest(
        manifest_path, repo_root=repo_root
    )


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Create or verify the isolated TODO3 source manifest"
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--sum-campaign", type=Path, required=True)
    parser.add_argument("--pre-campaign", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.check_only:
        document = build_source_manifest(
            sum_campaign=args.sum_campaign,
            pre_campaign=args.pre_campaign,
            repo_root=args.repo_root,
        )
        action = "prospective state validated"
    elif args.create:
        document = ensure_source_manifest(
            args.source_manifest,
            sum_campaign=args.sum_campaign,
            pre_campaign=args.pre_campaign,
            repo_root=args.repo_root,
        )
        action = "created"
    else:
        document = verify_source_manifest(
            args.source_manifest,
            sum_campaign=args.sum_campaign,
            pre_campaign=args.pre_campaign,
            repo_root=args.repo_root,
        )
        action = "verified"
    print(
        json.dumps(
            {
                "status": action,
                "profile": document["profile"],
                "manifest_sha256": document["manifest_sha256"],
                "shared_source_files": document["upstream"][
                    "shared_source_files"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    _main()
