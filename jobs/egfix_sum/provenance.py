"""Create and verify an immutable source manifest for CAFE-In/sum selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from jobs.egfix_sum.protocol import (
    BRIGHTKITE_TEMPORAL_SPLIT,
    BRIGHTKITE_TEMPORAL_SPLIT_SHA256,
    SCHEMA_VERSION,
    protocol_document,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_REQUIRED_SOURCE_PATHS: Tuple[str, ...] = (
    "environment.yml",
    "experiments/run_cafe.py",
    "experiments/cafe_methods.py",
    "experiments/egfix_protocol.py",
    "analysis/egfix_finalize_selection.py",
    "core/data_loader.py",
    "core/evaluation.py",
    "core/statistics.py",
    "core/training_inproc.py",
    "fairness_methods/inprocessing.py",
    "metrics/centrality_equality_gain.py",
    "metrics/equality_gain.py",
    "metrics/graph_utils.py",
    "jobs/egfix_sum/protocol.py",
    "jobs/egfix_sum/provenance.py",
    "jobs/egfix_sum/finalize_selection.py",
    "jobs/egfix_sum/tests.py",
    "jobs/egfix_sum/preflight.job",
    "jobs/egfix_sum/selection.job",
    "jobs/egfix_sum/finalize_selection.job",
    "jobs/egfix_sum/submit_selection.sh",
)

DEFAULT_DATA_ARTIFACT_PATHS: Mapping[str, str] = {
    "ego-facebook": "data/SNAP/ego-facebook/processed/data.pt",
    "oklahoma": "data/SNAP/oklahoma/processed/data.pt",
    "unc": "data/SNAP/unc/processed/data.pt",
    "pokec_n": "data/SNAP/pokec_n/processed/data.pt",
    "brightkite": "data/SNAP/brightkite/processed/data.pt",
}

DEFAULT_SOURCE_GLOBS: Tuple[str, ...] = (
    "analysis/**/*.py",
    "backbone/**/*.py",
    "baselines/**/*.py",
    "core/**/*.py",
    "experiments/**/*.py",
    "fairness_methods/**/*.py",
    "metrics/**/*.py",
    "tests/**/*.py",
    "jobs/egfix/*.job",
    "jobs/egfix/*.sh",
    "jobs/egfix_sum/*.py",
    "jobs/egfix_sum/*.job",
    "jobs/egfix_sum/*.sh",
)


class ProvenanceError(RuntimeError):
    """Raised when the campaign source state cannot be proven immutable."""


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
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProvenanceError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=_strict_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"Cannot read JSON manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProvenanceError(f"Manifest {path} must contain a JSON object")
    return value


def _safe_repo_file(repo_root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ProvenanceError(
            f"Source paths must be repository-relative: {relative_path!r}"
        )
    root = repo_root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ProvenanceError(
            f"Source path escapes repository root: {relative_path!r}"
        ) from exc
    if not resolved.is_file():
        raise ProvenanceError(f"Required file is missing: {relative_path}")
    return resolved


def resolve_source_paths(
    repo_root: Path,
    source_paths: Optional[Sequence[str]] = None,
    source_globs: Optional[Sequence[str]] = None,
) -> Tuple[str, ...]:
    root = repo_root.resolve()
    required = (
        DEFAULT_REQUIRED_SOURCE_PATHS if source_paths is None else source_paths
    )
    globs = DEFAULT_SOURCE_GLOBS if source_globs is None else source_globs
    selected = {Path(item).as_posix() for item in required}

    for pattern in globs:
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ProvenanceError(f"Unsafe source glob: {pattern!r}")
        for path in root.glob(pattern):
            if path.is_file():
                selected.add(path.relative_to(root).as_posix())

    for relative_path in selected:
        _safe_repo_file(root, relative_path)
    if not selected:
        raise ProvenanceError("The source hash set must not be empty")
    return tuple(sorted(selected))


def _protocol_for_split(
    split_relative_path: str, expected_split_sha256: str
) -> Dict[str, object]:
    protocol = protocol_document()
    protocol["brightkite_temporal_split"] = split_relative_path
    protocol["brightkite_temporal_split_sha256"] = expected_split_sha256
    return protocol


def build_source_manifest(
    repo_root: Path,
    source_paths: Optional[Sequence[str]] = None,
    source_globs: Optional[Sequence[str]] = None,
    data_artifact_paths: Optional[Mapping[str, str]] = None,
    split_relative_path: str = BRIGHTKITE_TEMPORAL_SPLIT,
    expected_split_sha256: str = BRIGHTKITE_TEMPORAL_SPLIT_SHA256,
) -> Dict[str, object]:
    root = repo_root.resolve()
    relative_sources = resolve_source_paths(root, source_paths, source_globs)
    data_paths = (
        DEFAULT_DATA_ARTIFACT_PATHS
        if data_artifact_paths is None
        else data_artifact_paths
    )
    split_path = _safe_repo_file(root, split_relative_path)
    split_sha256 = sha256_file(split_path)
    if split_sha256 != expected_split_sha256:
        raise ProvenanceError(
            "Brightkite temporal split SHA256 mismatch: "
            f"expected {expected_split_sha256}, got {split_sha256}"
        )

    files = {
        relative_path: {
            "sha256": sha256_file(_safe_repo_file(root, relative_path)),
            "size_bytes": _safe_repo_file(root, relative_path).stat().st_size,
        }
        for relative_path in relative_sources
    }
    data_artifacts = {}
    for dataset, relative_path in sorted(data_paths.items()):
        artifact_path = _safe_repo_file(root, relative_path)
        data_artifacts[dataset] = {
            "path": relative_path,
            "sha256": sha256_file(artifact_path),
            "size_bytes": artifact_path.stat().st_size,
        }
    protocol = _protocol_for_split(
        split_relative_path, expected_split_sha256
    )
    payload: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_kind": "cafe_in_sum_source_manifest",
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(
            canonical_json_bytes(protocol)
        ).hexdigest(),
        "files": files,
        "data_artifacts": data_artifacts,
        "brightkite_temporal_split": {
            "path": split_relative_path,
            "sha256": split_sha256,
            "size_bytes": split_path.stat().st_size,
        },
    }
    payload["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def _validate_manifest_digest(manifest: Mapping[str, object]) -> str:
    recorded = manifest.get("manifest_sha256")
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise ProvenanceError("Manifest has no valid manifest_sha256")
    unsigned = dict(manifest)
    del unsigned["manifest_sha256"]
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != recorded:
        raise ProvenanceError(
            f"Manifest self-digest mismatch: expected {recorded}, got {actual}"
        )
    return recorded


def verify_source_manifest(
    manifest_path: Path,
    repo_root: Path = REPO_ROOT,
    source_paths: Optional[Sequence[str]] = None,
    source_globs: Optional[Sequence[str]] = None,
    data_artifact_paths: Optional[Mapping[str, str]] = None,
    split_relative_path: str = BRIGHTKITE_TEMPORAL_SPLIT,
    expected_split_sha256: str = BRIGHTKITE_TEMPORAL_SPLIT_SHA256,
) -> Dict[str, object]:
    manifest = dict(_read_json(manifest_path.resolve()))
    _validate_manifest_digest(manifest)
    expected = build_source_manifest(
        repo_root,
        source_paths=source_paths,
        source_globs=source_globs,
        data_artifact_paths=data_artifact_paths,
        split_relative_path=split_relative_path,
        expected_split_sha256=expected_split_sha256,
    )
    if manifest != expected:
        if manifest.get("files") != expected["files"]:
            raise ProvenanceError(
                "Source manifest no longer matches the repository; "
                "a source file was added, removed, or changed"
            )
        if manifest.get("data_artifacts") != expected["data_artifacts"]:
            raise ProvenanceError(
                "Dataset manifest no longer matches the repository; "
                "a processed graph artifact was changed"
            )
        if manifest.get("protocol") != expected["protocol"]:
            raise ProvenanceError(
                "Source manifest protocol does not match the frozen protocol"
            )
        raise ProvenanceError(
            "Source manifest content does not match the recomputed manifest"
        )
    return expected


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
    repo_root: Path = REPO_ROOT,
    source_paths: Optional[Sequence[str]] = None,
    source_globs: Optional[Sequence[str]] = None,
    data_artifact_paths: Optional[Mapping[str, str]] = None,
    split_relative_path: str = BRIGHTKITE_TEMPORAL_SPLIT,
    expected_split_sha256: str = BRIGHTKITE_TEMPORAL_SPLIT_SHA256,
) -> Dict[str, object]:
    path = manifest_path.resolve()
    if path.exists():
        return verify_source_manifest(
            path,
            repo_root,
            source_paths=source_paths,
            source_globs=source_globs,
            data_artifact_paths=data_artifact_paths,
            split_relative_path=split_relative_path,
            expected_split_sha256=expected_split_sha256,
        )
    manifest = build_source_manifest(
        repo_root,
        source_paths=source_paths,
        source_globs=source_globs,
        data_artifact_paths=data_artifact_paths,
        split_relative_path=split_relative_path,
        expected_split_sha256=expected_split_sha256,
    )
    _atomic_write_json(path, manifest)
    return verify_source_manifest(
        path,
        repo_root,
        source_paths=source_paths,
        source_globs=source_globs,
        data_artifact_paths=data_artifact_paths,
        split_relative_path=split_relative_path,
        expected_split_sha256=expected_split_sha256,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Create or verify the CAFE-In/sum source manifest"
    )
    parser.add_argument(
        "--source-manifest",
        "--source_manifest",
        dest="manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        if not args.manifest.is_file():
            raise SystemExit(
                f"Preflight failed: manifest does not exist: {args.manifest}"
            )
        manifest = verify_source_manifest(args.manifest, args.repo_root)
        action = "verified"
    else:
        manifest = ensure_source_manifest(args.manifest, args.repo_root)
        action = "created"
    print(
        f"CAFE-In/sum source manifest {action}: "
        f"{manifest['manifest_sha256']}"
    )


if __name__ == "__main__":
    _main()
