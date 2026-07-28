"""Create or verify the corrected-EG campaign provenance manifest.

The manifest is immutable: the first invocation creates it, and later
invocations verify both its self-digest and every hashed source file.  A source
change therefore stops a dependent job instead of silently mixing code
versions in one campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.egfix_protocol import (  # noqa: E402
    BRIGHTKITE_TEMPORAL_SPLIT,
    BRIGHTKITE_TEMPORAL_SPLIT_SHA256,
    SCHEMA_VERSION,
    protocol_document,
    protocol_sha256,
)


DEFAULT_REQUIRED_SOURCE_PATHS: Tuple[str, ...] = (
    "experiments/run_cafe.py",
    "experiments/run_significance.py",
    "analysis/run_stratified_cafe.py",
    "experiments/cafe_methods.py",
    "experiments/egfix_protocol.py",
    "analysis/egfix_preflight.py",
    "analysis/egfix_finalize_selection.py",
    "analysis/run_precision_by_coreness.py",
    "core/data_loader.py",
    "core/evaluation.py",
    "core/statistics.py",
    "core/training_inproc.py",
    "fairness_methods/inprocessing.py",
    "metrics/centrality_equality_gain.py",
    "metrics/equality_gain.py",
    "metrics/graph_utils.py",
    "metrics/stratified.py",
)

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
)


class PreflightError(RuntimeError):
    """Raised when campaign provenance cannot be established."""


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
            raise PreflightError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=_strict_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"Cannot read JSON manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"Manifest {path} must contain a JSON object")
    return value


def _safe_repo_file(repo_root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute():
        raise PreflightError(
            f"Source paths must be repository-relative: {relative_path!r}"
        )
    root = repo_root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PreflightError(
            f"Source path escapes repository root: {relative_path!r}"
        ) from exc
    if not resolved.is_file():
        raise PreflightError(f"Required file is missing: {relative_path}")
    return resolved


def resolve_source_paths(
    repo_root: Path,
    source_paths: Optional[Sequence[str]] = None,
    source_globs: Optional[Sequence[str]] = None,
) -> Tuple[str, ...]:
    """Resolve, deduplicate, and validate the source files to hash."""

    root = repo_root.resolve()
    required = (
        DEFAULT_REQUIRED_SOURCE_PATHS if source_paths is None else source_paths
    )
    globs = DEFAULT_SOURCE_GLOBS if source_globs is None else source_globs
    selected = {Path(item).as_posix() for item in required}

    for pattern in globs:
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise PreflightError(f"Unsafe source glob: {pattern!r}")
        for path in root.glob(pattern):
            if path.is_file():
                selected.add(path.relative_to(root).as_posix())

    for relative_path in selected:
        _safe_repo_file(root, relative_path)
    if not selected:
        raise PreflightError("The source hash set must not be empty")
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
    split_relative_path: str = BRIGHTKITE_TEMPORAL_SPLIT,
    expected_split_sha256: str = BRIGHTKITE_TEMPORAL_SPLIT_SHA256,
) -> Dict[str, object]:
    """Build a manifest and fail if the frozen Brightkite split is not exact."""

    root = repo_root.resolve()
    relative_sources = resolve_source_paths(root, source_paths, source_globs)
    split_path = _safe_repo_file(root, split_relative_path)
    split_sha256 = sha256_file(split_path)
    if split_sha256 != expected_split_sha256:
        raise PreflightError(
            "Brightkite temporal split SHA256 mismatch: "
            f"expected {expected_split_sha256}, got {split_sha256} "
            f"for {split_relative_path}"
        )

    files = {}
    for relative_path in relative_sources:
        path = _safe_repo_file(root, relative_path)
        files[relative_path] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    protocol = _protocol_for_split(
        split_relative_path, expected_split_sha256
    )
    payload: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_kind": "egfix_source_manifest",
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(
            canonical_json_bytes(protocol)
        ).hexdigest(),
        "files": files,
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
        raise PreflightError("Manifest has no valid manifest_sha256")
    unsigned = dict(manifest)
    del unsigned["manifest_sha256"]
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != recorded:
        raise PreflightError(
            f"Manifest self-digest mismatch: expected {recorded}, got {actual}"
        )
    return recorded


def verify_source_manifest(
    manifest_path: Path,
    repo_root: Path,
    source_paths: Optional[Sequence[str]] = None,
    source_globs: Optional[Sequence[str]] = None,
    split_relative_path: str = BRIGHTKITE_TEMPORAL_SPLIT,
    expected_split_sha256: str = BRIGHTKITE_TEMPORAL_SPLIT_SHA256,
) -> Dict[str, object]:
    """Verify the manifest, protocol, split, and current source tree."""

    path = manifest_path.resolve()
    manifest = dict(_read_json(path))
    _validate_manifest_digest(manifest)

    expected = build_source_manifest(
        repo_root=repo_root,
        source_paths=source_paths,
        source_globs=source_globs,
        split_relative_path=split_relative_path,
        expected_split_sha256=expected_split_sha256,
    )

    if manifest != expected:
        recorded_files = manifest.get("files")
        expected_files = expected["files"]
        if recorded_files != expected_files:
            raise PreflightError(
                "Source manifest no longer matches the repository; "
                "a source file was added, removed, or changed"
            )
        if manifest.get("protocol") != expected["protocol"]:
            raise PreflightError(
                "Source manifest protocol does not match the frozen protocol"
            )
        raise PreflightError(
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
    repo_root: Path,
    source_paths: Optional[Sequence[str]] = None,
    source_globs: Optional[Sequence[str]] = None,
    split_relative_path: str = BRIGHTKITE_TEMPORAL_SPLIT,
    expected_split_sha256: str = BRIGHTKITE_TEMPORAL_SPLIT_SHA256,
) -> Dict[str, object]:
    """Create a missing manifest, or verify an existing immutable manifest."""

    path = manifest_path.resolve()
    if path.exists():
        return verify_source_manifest(
            path,
            repo_root,
            source_paths=source_paths,
            source_globs=source_globs,
            split_relative_path=split_relative_path,
            expected_split_sha256=expected_split_sha256,
        )

    manifest = build_source_manifest(
        repo_root,
        source_paths=source_paths,
        source_globs=source_globs,
        split_relative_path=split_relative_path,
        expected_split_sha256=expected_split_sha256,
    )
    _atomic_write_json(path, manifest)
    return verify_source_manifest(
        path,
        repo_root,
        source_paths=source_paths,
        source_globs=source_globs,
        split_relative_path=split_relative_path,
        expected_split_sha256=expected_split_sha256,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create or verify the immutable source manifest for an EG-fix "
            "rerun campaign"
        )
    )
    parser.add_argument(
        "--manifest",
        "--source-manifest",
        "--source_manifest",
        dest="manifest",
        type=Path,
    )
    parser.add_argument(
        "--campaign-root",
        "--campaign_root",
        dest="campaign_root",
        type=Path,
        help=(
            "Campaign directory; when --source_manifest is omitted, the "
            "manifest is written to ROOT/source_manifest.json"
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Additional repository-relative source file to hash",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Fail if the manifest does not already exist",
    )
    args = parser.parse_args()

    if args.manifest is None:
        if args.campaign_root is None:
            parser.error(
                "one of --source_manifest or --campaign_root is required"
            )
        args.manifest = args.campaign_root / "source_manifest.json"

    source_paths = tuple(DEFAULT_REQUIRED_SOURCE_PATHS) + tuple(args.source)
    if args.verify_only:
        if not args.manifest.is_file():
            raise SystemExit(
                f"Preflight failed: manifest does not exist: {args.manifest}"
            )
        action = "verified"
        manifest = verify_source_manifest(
            args.manifest,
            args.repo_root,
            source_paths=source_paths,
        )
    else:
        action = "verified" if args.manifest.exists() else "created"
        manifest = ensure_source_manifest(
            args.manifest,
            args.repo_root,
            source_paths=source_paths,
        )

    print(
        json.dumps(
            {
                "status": action,
                "manifest": str(args.manifest.resolve()),
                "manifest_sha256": manifest["manifest_sha256"],
                "protocol_sha256": manifest["protocol_sha256"],
                "source_file_count": len(manifest["files"]),
                "brightkite_temporal_split_sha256": manifest[
                    "brightkite_temporal_split"
                ]["sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    _main()
