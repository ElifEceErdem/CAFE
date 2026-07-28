"""Strict sum-aware TODO3 finalization without modifying the base finalizer.

The existing 2,061-line rerun finalizer is reused byte-for-byte through a
narrow, process-local adapter.  Its expected reduction and source verifier are
temporarily changed to the TODO3 values, and its outputs are first promoted to
a private core directory.  Only after command provenance and the all-five
target-exposure audit pass is that directory atomically renamed to
``finalized`` and the authoritative ``_TODO3_COMPLETE`` marker written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from analysis import egfix_finalize_reruns as base
from jobs.egfix_sum_todo3.compose_selected_configs import (
    LINEAGE_FILENAME,
    SELECTED_FILENAME,
    verify_composite_configs,
)
from jobs.egfix_sum_todo3.protocol import (
    AGGREGATE_SEEDS,
    BATCH_SIZE,
    DATASETS,
    K,
    LOSS_FUNC,
    METHODS,
    NUM_NEG,
    PENALTY_REDUCTION,
    PROFILE,
    STRATIFIED_SEEDS,
    protocol_document,
)
from jobs.egfix_sum_todo3.provenance import (
    REPO_ROOT,
    canonical_json_bytes,
    read_json,
    sha256_file,
    verify_manifest_for_base_finalizer,
    verify_source_manifest,
)
from jobs.egfix_sum_todo3.target_exposure import (
    BY_RUN_FILENAME as TARGET_BY_RUN_FILENAME,
    CONTRAST_BY_RUN_FILENAME as TARGET_CONTRAST_BY_RUN_FILENAME,
    CONTRAST_SUMMARY_FILENAME as TARGET_CONTRAST_SUMMARY_FILENAME,
    INTEGRITY_FILENAME as TARGET_INTEGRITY_FILENAME,
    SUMMARY_FILENAME as TARGET_SUMMARY_FILENAME,
    generate_target_exposure,
)


EXPECTED_BASE_FINALIZER_SHA256 = (
    "0b0ba402c4e12e1bfed8d6bde6373b911bf5b2b8845d14bdb8a4f737d1f03e73"
)
CORE_DIRNAME = ".todo3_core_finalized"
CORE_MARKER = "_TODO3_CORE_RERUNS_COMPLETE"
FINAL_DIRNAME = "finalized"
COMPLETION_MARKER = "_TODO3_COMPLETE"
OUTER_INTEGRITY_FILENAME = "todo3_integrity.json"
TASK_DIGEST_FILENAME = "task_artifacts.sha256"


class Todo3FinalizationError(RuntimeError):
    """Raised when TODO3 artifacts are not safe to promote."""


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


def _safe_relative(path: Path, campaign: Path) -> str:
    try:
        return path.resolve().relative_to(campaign.resolve()).as_posix()
    except ValueError as exc:
        raise Todo3FinalizationError(
            f"Artifact escapes TODO3 campaign: {path}"
        ) from exc


def _strict_env(path: Path) -> Dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise Todo3FinalizationError(
            f"Cannot read task environment {path}: {exc}"
        ) from exc
    result = {}
    for line_number, line in enumerate(lines, start=1):
        if not line or "=" not in line:
            raise Todo3FinalizationError(
                f"{path}:{line_number}: expected key=value"
            )
        key, value = line.split("=", 1)
        if not key or key in result:
            raise Todo3FinalizationError(
                f"{path}:{line_number}: empty or duplicate key"
            )
        result[key] = value
    return result


def _require_env(
    actual: Mapping[str, str],
    expected: Mapping[str, str],
    path: Path,
) -> None:
    missing = set(expected) - set(actual)
    if missing:
        raise Todo3FinalizationError(
            f"{path}: missing task fields {sorted(missing)}"
        )
    for field, wanted in expected.items():
        if actual[field] != wanted:
            raise Todo3FinalizationError(
                f"{path}: {field} must equal {wanted!r}, "
                f"got {actual[field]!r}"
            )


def _parse_command(path: Path) -> List[str]:
    try:
        text = path.read_text(encoding="utf-8")
        tokens = shlex.split(text, posix=True)
    except (OSError, ValueError) as exc:
        raise Todo3FinalizationError(
            f"Cannot parse command record {path}: {exc}"
        ) from exc
    if not tokens:
        raise Todo3FinalizationError(f"{path}: empty command")
    return tokens


def _command_flags(tokens: Sequence[str], path: Path) -> Dict[str, str]:
    if any(token in {";", "&&", "||", "|", ">", ">>"} for token in tokens):
        raise Todo3FinalizationError(
            f"{path}: shell operators are not permitted"
        )
    flags = {}
    index = 3
    while index < len(tokens):
        flag = tokens[index]
        if not flag.startswith("--") or index + 1 >= len(tokens):
            raise Todo3FinalizationError(
                f"{path}: malformed command token at {flag!r}"
            )
        if flag in flags:
            raise Todo3FinalizationError(
                f"{path}: duplicate command flag {flag}"
            )
        flags[flag] = tokens[index + 1]
        index += 2
    return flags


def _validate_command(
    path: Path,
    *,
    stage: str,
    dataset: str,
    seed: int,
    selected_configs: Path,
    source_manifest: Path,
    repo_root: Path,
) -> None:
    tokens = _parse_command(path)
    expected_prefix = (
        ["python", "-u", "experiments/run_significance.py"]
        if stage == "aggregate"
        else ["python", "-u", "analysis/run_stratified_cafe.py"]
    )
    if tokens[:3] != expected_prefix:
        raise Todo3FinalizationError(
            f"{path}: runner prefix differs from {expected_prefix}"
        )
    flags = _command_flags(tokens, path)
    common = {
        "--data_name": dataset,
        "--data_path": str(repo_root / "data" / "SNAP"),
        "--encoder": "HLGNN",
        "--predictor": "MLP",
        "--epochs": "500",
        "--batch_size": str(BATCH_SIZE),
        "--lr": "0.001",
        "--loss_func": LOSS_FUNC,
        "--num_neg": str(NUM_NEG),
        "--selected_configs": str(selected_configs),
        "--penalty_reduction": PENALTY_REDUCTION,
        "--skip_final_comprehensive_eval": "True",
        "--eval_skip_test": "True",
        "--closeness_seed": "42",
        "--closeness_sample_size": "0",
        "--closeness_epsilon": "0.1",
        "--source_manifest": str(source_manifest),
    }
    if stage == "aggregate":
        common.update(
            {
                "--seed_start": str(seed),
                "--num_seeds": "1",
                "--vary_split": "False",
                "--protocol": "headline",
                "--k_values": "10",
                "--centrality_types": "pagerank,closeness",
                "--use_demographics": "False",
                "--out_csv": str(path.parent / "per_seed_metrics.csv"),
            }
        )
    else:
        common.update(
            {
                "--K": str(K),
                "--num_bins": "10",
                "--runs": "1",
                "--seed": str(seed),
                "--include_fusion": "True",
                "--save_recommendations": "True",
                "--save_dir": str(path.parent),
            }
        )
    if dataset in {"pokec_n", "brightkite"}:
        common.update(
            {
                "--eval_max_nodes": "5000",
                "--eval_steps": "25",
                "--patience": "6",
            }
        )
    else:
        common.update({"--eval_steps": "10", "--patience": "50"})
    if dataset == "brightkite":
        common["--split_file"] = str(
            repo_root
            / "data"
            / "SNAP"
            / "brightkite"
            / "temporal_split.pt"
        )
    for flag, expected in common.items():
        if flags.get(flag) != expected:
            raise Todo3FinalizationError(
                f"{path}: {flag} must equal {expected!r}, "
                f"got {flags.get(flag)!r}"
            )
    if set(flags) != set(common):
        raise Todo3FinalizationError(
            f"{path}: command flags differ; "
            f"missing={sorted(set(common) - set(flags))}, "
            f"extra={sorted(set(flags) - set(common))}"
        )
    if "mean" in tokens:
        raise Todo3FinalizationError(
            f"{path}: historical mean reduction leaked into TODO3"
        )


def _validate_digest_sidecar(
    task_dir: Path, expected_names: Sequence[str]
) -> Dict[str, str]:
    path = task_dir / TASK_DIGEST_FILENAME
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise Todo3FinalizationError(
            f"Cannot read task digest sidecar {path}: {exc}"
        ) from exc
    records = {}
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 2 or len(fields[0]) != 64:
            raise Todo3FinalizationError(
                f"{path}:{line_number}: malformed SHA256 record"
            )
        digest, name = fields
        if name.startswith("*"):
            name = name[1:]
        if name in records or Path(name).name != name:
            raise Todo3FinalizationError(
                f"{path}:{line_number}: duplicate or unsafe filename"
            )
        records[name] = digest
    if set(records) != set(expected_names):
        raise Todo3FinalizationError(
            f"{path}: digest names differ; expected={sorted(expected_names)}, "
            f"found={sorted(records)}"
        )
    for name, expected_sha in records.items():
        artifact = task_dir / name
        if not artifact.is_file() or sha256_file(artifact) != expected_sha:
            raise Todo3FinalizationError(
                f"{path}: hash mismatch for {artifact}"
            )
    return records


def validate_task_provenance(
    *,
    campaign_root: Path,
    selected_configs: Path,
    source_manifest: Path,
    repo_root: Path,
    sum_run_tag: str,
    pre_run_tag: str,
    runtime_sha256: str,
) -> List[Dict[str, object]]:
    records = []
    selected_sha = sha256_file(selected_configs)
    source_sha = sha256_file(source_manifest)
    for stage, seeds in (
        ("aggregate", AGGREGATE_SEEDS),
        ("stratified", STRATIFIED_SEEDS),
    ):
        task_id = 0
        for dataset in DATASETS:
            for seed in seeds:
                task_dir = (
                    campaign_root / stage / dataset / f"seed_{seed}"
                )
                success = task_dir / "_SUCCESS"
                env_path = task_dir / "task.env"
                command_path = task_dir / "command.sh"
                for required in (success, env_path, command_path):
                    if not required.is_file():
                        raise Todo3FinalizationError(
                            f"Missing {stage} task artifact: {required}"
                        )
                expected_env = {
                    "stage": stage,
                    "sum_run_tag": sum_run_tag,
                    "pre_run_tag": pre_run_tag,
                    "array_task_id": str(task_id),
                    "dataset": dataset,
                    "training_seed": str(seed),
                    "split_seed": "42",
                    "evaluation_split": "test",
                    "penalty_reduction": PENALTY_REDUCTION,
                    "loss_func": LOSS_FUNC,
                    "num_neg": str(NUM_NEG),
                    "batch_size": str(BATCH_SIZE),
                    "eval_skip_test": "True",
                    "selected_configs_sha256": selected_sha,
                    "source_manifest_sha256": source_sha,
                    "runtime_environment_sha256": runtime_sha256,
                }
                if stage == "stratified":
                    expected_env.update(
                        {
                            "K": str(K),
                            "num_bins": "10",
                            "save_recommendations": "True",
                        }
                    )
                _require_env(
                    _strict_env(env_path), expected_env, env_path
                )
                _validate_command(
                    command_path,
                    stage=stage,
                    dataset=dataset,
                    seed=seed,
                    selected_configs=selected_configs,
                    source_manifest=source_manifest,
                    repo_root=repo_root,
                )
                if stage == "aggregate":
                    names = (
                        "task.env",
                        "command.sh",
                        "per_seed_metrics.csv",
                        "per_seed_metrics.csv.complete.json",
                    )
                else:
                    names = (
                        "task.env",
                        "command.sh",
                        f"stratified_cafe_{dataset}.csv",
                        (
                            f"eg_by_coreness_cafe_{dataset}"
                            "_per_run.csv"
                        ),
                        f"metadata_{dataset}.json",
                        f"recommendations_{dataset}_seed{seed}.npz",
                    )
                digests = _validate_digest_sidecar(task_dir, names)
                records.append(
                    {
                        "stage": stage,
                        "task_id": task_id,
                        "dataset": dataset,
                        "training_seed": seed,
                        "task_directory": _safe_relative(
                            task_dir, campaign_root
                        ),
                        "task_env_sha256": digests["task.env"],
                        "command_sha256": digests["command.sh"],
                        "task_digest_path": _safe_relative(
                            task_dir / TASK_DIGEST_FILENAME,
                            campaign_root,
                        ),
                        "task_digest_sha256": sha256_file(
                            task_dir / TASK_DIGEST_FILENAME
                        ),
                        "success_marker_sha256": sha256_file(success),
                    }
                )
                task_id += 1
    if len(records) != 65:
        raise Todo3FinalizationError(
            f"Expected 65 task provenance records, found {len(records)}"
        )
    return records


@contextmanager
def _sum_base_adapter():
    original = {
        "PENALTY_REDUCTION": base.PENALTY_REDUCTION,
        "verify_source_manifest": base.verify_source_manifest,
        "FINAL_DIRNAME": base.FINAL_DIRNAME,
        "COMPLETION_MARKER": base.COMPLETION_MARKER,
    }
    base.PENALTY_REDUCTION = PENALTY_REDUCTION
    base.verify_source_manifest = verify_manifest_for_base_finalizer
    base.FINAL_DIRNAME = CORE_DIRNAME
    base.COMPLETION_MARKER = CORE_MARKER
    try:
        yield
    finally:
        for field, value in original.items():
            setattr(base, field, value)


def _validate_payload_digest(
    document: Mapping[str, object], field: str, path: Path
) -> None:
    recorded = document.get(field)
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise Todo3FinalizationError(f"{path}: missing {field}")
    unsigned = dict(document)
    del unsigned[field]
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != recorded:
        raise Todo3FinalizationError(
            f"{path}: embedded digest {field} is invalid"
        )


def _validate_core_stage(
    *,
    campaign_root: Path,
    selected_configs: Path,
    source_manifest: Path,
) -> Mapping[str, object]:
    core_dir = campaign_root / CORE_DIRNAME
    marker_path = campaign_root / CORE_MARKER
    integrity_path = core_dir / base.INTEGRITY_FILENAME
    if not core_dir.is_dir() or not marker_path.is_file():
        raise Todo3FinalizationError(
            "Core finalizer did not create both private completion artifacts"
        )
    marker = read_json(marker_path)
    if (
        marker.get("status") != "complete"
        or marker.get("selected_configs_sha256")
        != sha256_file(selected_configs)
        or marker.get("source_manifest_sha256")
        != sha256_file(source_manifest)
        or marker.get("aggregate_shards") != 50
        or marker.get("stratified_shards") != 15
    ):
        raise Todo3FinalizationError(
            f"{marker_path}: invalid core completion record"
        )
    if marker.get("integrity_file_sha256") != sha256_file(integrity_path):
        raise Todo3FinalizationError(
            f"{marker_path}: core integrity hash mismatch"
        )
    integrity = read_json(integrity_path)
    _validate_payload_digest(
        integrity, "integrity_payload_sha256", integrity_path
    )
    protocol = integrity.get("protocol")
    provenance = integrity.get("provenance")
    crosscheck = integrity.get("degree_eg_crosscheck")
    if (
        integrity.get("status") != "complete"
        or not isinstance(protocol, dict)
        or protocol.get("penalty_reduction") != PENALTY_REDUCTION
        or not isinstance(provenance, dict)
        or provenance.get("selected_configs_sha256")
        != sha256_file(selected_configs)
        or provenance.get("source_manifest_sha256")
        != sha256_file(source_manifest)
        or not isinstance(crosscheck, dict)
        or crosscheck.get("status") != "passed"
    ):
        raise Todo3FinalizationError(
            f"{integrity_path}: core validation is incomplete or not sum"
        )
    return integrity


def _run_or_resume_core(
    *,
    campaign_root: Path,
    selected_configs: Path,
    source_manifest: Path,
    repo_root: Path,
) -> Mapping[str, object]:
    core_dir = campaign_root / CORE_DIRNAME
    core_marker = campaign_root / CORE_MARKER
    if core_dir.exists() != core_marker.exists():
        raise Todo3FinalizationError(
            "Only one private core-finalizer artifact exists; inspect the "
            "interrupted finalizer before retrying"
        )
    if not core_dir.exists():
        base_path = repo_root / "analysis" / "egfix_finalize_reruns.py"
        if sha256_file(base_path) != EXPECTED_BASE_FINALIZER_SHA256:
            raise Todo3FinalizationError(
                "The frozen base rerun finalizer differs from the reviewed "
                f"SHA256 {EXPECTED_BASE_FINALIZER_SHA256}"
            )
        with _sum_base_adapter():
            result = base.finalize_reruns(
                campaign_root=campaign_root,
                selected_configs_path=selected_configs,
                source_manifest_path=source_manifest,
                repo_root=repo_root,
                enable_graph_crosscheck=True,
            )
        if result.get("degree_eg_crosscheck") != "passed":
            raise Todo3FinalizationError(
                "Production Degree-EG cross-path validation did not pass"
            )
    return _validate_core_stage(
        campaign_root=campaign_root,
        selected_configs=selected_configs,
        source_manifest=source_manifest,
    )


def _output_records(root: Path) -> List[Dict[str, object]]:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != OUTER_INTEGRITY_FILENAME:
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return records


def _build_outer_integrity(
    *,
    campaign_root: Path,
    core_dir: Path,
    source_manifest: Path,
    selected_configs: Path,
    config_summary: Mapping[str, object],
    core_integrity: Mapping[str, object],
    target_summary: Mapping[str, object],
    task_records: Sequence[Mapping[str, object]],
    sum_run_tag: str,
    pre_run_tag: str,
) -> Dict[str, object]:
    lineage_path = campaign_root / LINEAGE_FILENAME
    lineage = read_json(lineage_path)
    _validate_payload_digest(
        lineage, "lineage_sha256", lineage_path
    )
    document: Dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "profile": PROFILE,
        "campaign_root": str(campaign_root.resolve()),
        "protocol": protocol_document(),
        "upstream_run_tags": {
            "cafe_in_sum": sum_run_tag,
            "cafe_pre": pre_run_tag,
        },
        "provenance": {
            "source_manifest_path": str(source_manifest.resolve()),
            "source_manifest_sha256": sha256_file(source_manifest),
            "source_manifest_content_sha256": read_json(
                source_manifest
            )["manifest_sha256"],
            "selected_configs_path": str(selected_configs.resolve()),
            "selected_configs_sha256": sha256_file(selected_configs),
            "selected_configs_lineage_path": str(
                lineage_path.resolve()
            ),
            "selected_configs_lineage_file_sha256": sha256_file(
                lineage_path
            ),
            "selected_configs_lineage_content_sha256": lineage[
                "lineage_sha256"
            ],
            "configuration_summary": dict(config_summary),
            "base_finalizer_sha256": EXPECTED_BASE_FINALIZER_SHA256,
        },
        "core_reruns": {
            "integrity_file": base.INTEGRITY_FILENAME,
            "integrity_sha256": sha256_file(
                core_dir / base.INTEGRITY_FILENAME
            ),
            "aggregate_shards": 50,
            "stratified_shards": 15,
            "degree_eg_crosscheck": core_integrity[
                "degree_eg_crosscheck"
            ]["status"],
        },
        "target_exposure": dict(target_summary),
        "task_provenance": {
            "records": list(task_records),
            "record_count": len(task_records),
            "sum_validity_tuple": {
                "loss_func": LOSS_FUNC,
                "num_neg": NUM_NEG,
                "batch_size": BATCH_SIZE,
                "penalty_reduction": PENALTY_REDUCTION,
            },
        },
        "outputs": _output_records(core_dir),
        "publication_pdfs": core_integrity["publication_pdfs"],
        "manuscript_writes": False,
    }
    document["integrity_payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(document)
    ).hexdigest()
    return document


def _validate_target_integrity(
    final_dir: Path,
) -> Mapping[str, object]:
    integrity_path = final_dir / TARGET_INTEGRITY_FILENAME
    if not integrity_path.is_file() or integrity_path.is_symlink():
        raise Todo3FinalizationError(
            f"Final target-exposure integrity is missing: {integrity_path}"
        )
    integrity = read_json(integrity_path)
    _validate_payload_digest(
        integrity, "integrity_payload_sha256", integrity_path
    )
    if (
        integrity.get("status") != "complete"
        or integrity.get("datasets") != list(DATASETS)
        or integrity.get("training_seeds") != list(STRATIFIED_SEEDS)
        or integrity.get("methods") != list(METHODS)
        or integrity.get("core_integrity_path")
        != base.INTEGRITY_FILENAME
        or integrity.get("core_integrity_sha256")
        != sha256_file(final_dir / base.INTEGRITY_FILENAME)
    ):
        raise Todo3FinalizationError(
            f"{integrity_path}: invalid target-exposure completion state"
        )

    expected_rows = {
        TARGET_BY_RUN_FILENAME: 225,
        TARGET_SUMMARY_FILENAME: 75,
        TARGET_CONTRAST_BY_RUN_FILENAME: 45,
        TARGET_CONTRAST_SUMMARY_FILENAME: 15,
    }
    records = integrity.get("outputs")
    if not isinstance(records, list) or len(records) != len(expected_rows):
        raise Todo3FinalizationError(
            f"{integrity_path}: target output records are incomplete"
        )
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise Todo3FinalizationError(
                f"{integrity_path}: malformed target output record"
            )
        relative = record.get("path")
        if (
            not isinstance(relative, str)
            or relative not in expected_rows
            or relative in seen
            or Path(relative).name != relative
        ):
            raise Todo3FinalizationError(
                f"{integrity_path}: invalid target output path {relative!r}"
            )
        seen.add(relative)
        path = final_dir / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != record.get("sha256")
            or path.stat().st_size != record.get("size_bytes")
            or record.get("rows") != expected_rows[relative]
        ):
            raise Todo3FinalizationError(
                f"{integrity_path}: changed target output {path}"
            )
    if seen != set(expected_rows):
        raise Todo3FinalizationError(
            f"{integrity_path}: target output set is incomplete"
        )
    return integrity


def _validate_finished_directory(
    final_dir: Path,
    campaign_root: Path,
    *,
    source_manifest: Path,
    selected_configs: Path,
    task_records: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    integrity_path = final_dir / OUTER_INTEGRITY_FILENAME
    if not integrity_path.is_file() or integrity_path.is_symlink():
        raise Todo3FinalizationError(
            f"Interrupted final directory lacks {integrity_path}"
        )
    integrity = read_json(integrity_path)
    _validate_payload_digest(
        integrity, "integrity_payload_sha256", integrity_path
    )
    if (
        integrity.get("status") != "complete"
        or integrity.get("profile") != PROFILE
        or integrity.get("campaign_root") != str(campaign_root.resolve())
        or integrity.get("manuscript_writes") is not False
    ):
        raise Todo3FinalizationError(
            f"{integrity_path}: invalid outer completion state"
        )

    provenance = integrity.get("provenance")
    task_provenance = integrity.get("task_provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("source_manifest_sha256")
        != sha256_file(source_manifest)
        or provenance.get("selected_configs_sha256")
        != sha256_file(selected_configs)
        or not isinstance(task_provenance, dict)
        or task_provenance.get("record_count") != len(task_records)
        or task_provenance.get("records") != list(task_records)
    ):
        raise Todo3FinalizationError(
            f"{integrity_path}: promoted provenance differs from current "
            "immutable campaign state"
        )

    records = integrity.get("outputs")
    if not isinstance(records, list) or not records:
        raise Todo3FinalizationError(
            f"{integrity_path}: output records must be a nonempty list"
        )
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise Todo3FinalizationError(
                f"{integrity_path}: malformed output record"
            )
        relative = record.get("path")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in seen
        ):
            raise Todo3FinalizationError(
                f"{integrity_path}: unsafe or duplicate output path "
                f"{relative!r}"
            )
        seen.add(relative)
        path = final_dir / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != record.get("sha256")
            or path.stat().st_size != record.get("size_bytes")
        ):
            raise Todo3FinalizationError(
                f"{integrity_path}: promoted output changed: {path}"
            )

    actual = {
        path.relative_to(final_dir).as_posix()
        for path in final_dir.rglob("*")
        if path.is_file() and path.name != OUTER_INTEGRITY_FILENAME
    }
    if actual != seen:
        raise Todo3FinalizationError(
            f"{integrity_path}: promoted output set differs; "
            f"missing={sorted(seen - actual)}, extra={sorted(actual - seen)}"
        )

    target_integrity = _validate_target_integrity(final_dir)
    target_summary = integrity.get("target_exposure")
    if (
        not isinstance(target_summary, dict)
        or target_summary.get("status") != "complete"
        or target_summary.get("integrity_file")
        != TARGET_INTEGRITY_FILENAME
        or target_summary.get("integrity_sha256")
        != sha256_file(final_dir / TARGET_INTEGRITY_FILENAME)
        or target_summary.get("by_run_rows") != 225
        or target_summary.get("summary_rows") != 75
        or target_summary.get("contrast_by_run_rows") != 45
        or target_summary.get("contrast_summary_rows") != 15
        or target_integrity.get("status") != "complete"
    ):
        raise Todo3FinalizationError(
            f"{integrity_path}: target-exposure lineage is invalid"
        )
    return integrity


def _completion_document(
    *,
    campaign_root: Path,
    final_dir: Path,
    integrity: Mapping[str, object],
) -> Dict[str, object]:
    integrity_path = final_dir / OUTER_INTEGRITY_FILENAME
    return {
        "schema_version": 1,
        "status": "complete",
        "profile": PROFILE,
        "campaign_root": str(campaign_root.resolve()),
        "finalized_directory": FINAL_DIRNAME,
        "integrity_file": (
            f"{FINAL_DIRNAME}/{OUTER_INTEGRITY_FILENAME}"
        ),
        "integrity_file_sha256": sha256_file(integrity_path),
        "integrity_payload_sha256": integrity[
            "integrity_payload_sha256"
        ],
        "aggregate_shards": 50,
        "stratified_shards": 15,
        "target_exposure_by_run_rows": 225,
        "publication_pdfs": 5,
        "manuscript_writes": False,
    }


def finalize_todo3(
    *,
    campaign_root: Path,
    sum_campaign: Path,
    pre_campaign: Path,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, object]:
    campaign = campaign_root.resolve()
    repo = repo_root.resolve()
    final_dir = campaign / FINAL_DIRNAME
    official_marker = campaign / COMPLETION_MARKER
    core_marker = campaign / CORE_MARKER
    if official_marker.exists() and not final_dir.is_dir():
        raise Todo3FinalizationError(
            "TODO3 completion marker exists without finalized directory"
        )
    if official_marker.exists() and final_dir.is_dir():
        raise Todo3FinalizationError(
            "Refusing to overwrite an already-completed TODO3 campaign"
        )

    # Recheck the complete immutable execution state even when resuming after
    # the atomic core->finalized promotion.  A missing official marker alone
    # must never allow changed sources, configs, tasks, or upstream lineage to
    # be blessed as complete.
    source_manifest = campaign / "source_manifest.json"
    selected_configs = campaign / SELECTED_FILENAME
    verify_source_manifest(
        source_manifest,
        sum_campaign=sum_campaign,
        pre_campaign=pre_campaign,
        repo_root=repo,
    )
    config_summary = verify_composite_configs(
        output_dir=campaign,
        source_manifest_path=source_manifest,
        sum_campaign=sum_campaign,
        pre_campaign=pre_campaign,
        repo_root=repo,
    )
    source_document = read_json(source_manifest)
    runtime_sha = str(
        source_document["upstream"]["runtime_environment_sha256"]
    )
    task_records = validate_task_provenance(
        campaign_root=campaign,
        selected_configs=selected_configs,
        source_manifest=source_manifest,
        repo_root=repo,
        sum_run_tag=sum_campaign.name,
        pre_run_tag=pre_campaign.name,
        runtime_sha256=runtime_sha,
    )

    if final_dir.is_dir():
        integrity = _validate_finished_directory(
            final_dir,
            campaign,
            source_manifest=source_manifest,
            selected_configs=selected_configs,
            task_records=task_records,
        )
        if core_marker.exists():
            core_marker.unlink()
        completion = _completion_document(
            campaign_root=campaign,
            final_dir=final_dir,
            integrity=integrity,
        )
        _atomic_write_json(official_marker, completion)
        return {
            "status": "complete",
            "resumed_after_atomic_promotion": True,
            "finalized_directory": str(final_dir),
            "completion_marker": str(official_marker),
        }
    core_integrity = _run_or_resume_core(
        campaign_root=campaign,
        selected_configs=selected_configs,
        source_manifest=source_manifest,
        repo_root=repo,
    )
    core_dir = campaign / CORE_DIRNAME
    target_summary = generate_target_exposure(
        campaign_root=campaign,
        core_finalized_dir=core_dir,
        repo_root=repo,
    )
    outer = _build_outer_integrity(
        campaign_root=campaign,
        core_dir=core_dir,
        source_manifest=source_manifest,
        selected_configs=selected_configs,
        config_summary=config_summary,
        core_integrity=core_integrity,
        target_summary=target_summary,
        task_records=task_records,
        sum_run_tag=sum_campaign.name,
        pre_run_tag=pre_campaign.name,
    )
    _atomic_write_json(core_dir / OUTER_INTEGRITY_FILENAME, outer)
    if final_dir.exists():
        raise Todo3FinalizationError(
            f"Refusing to replace existing final directory: {final_dir}"
        )
    os.replace(core_dir, final_dir)
    if core_marker.exists():
        core_marker.unlink()
    promoted = _validate_finished_directory(
        final_dir,
        campaign,
        source_manifest=source_manifest,
        selected_configs=selected_configs,
        task_records=task_records,
    )
    completion = _completion_document(
        campaign_root=campaign,
        final_dir=final_dir,
        integrity=promoted,
    )
    _atomic_write_json(official_marker, completion)
    return {
        "status": "complete",
        "profile": PROFILE,
        "finalized_directory": str(final_dir),
        "completion_marker": str(official_marker),
        "aggregate_shards": 50,
        "stratified_shards": 15,
        "target_exposure_by_run_rows": 225,
        "target_exposure_summary_rows": 75,
        "target_contrast_summary_rows": 15,
        "publication_pdfs": 5,
        "degree_eg_crosscheck": "passed",
        "manuscript_writes": False,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Strictly finalize the isolated TODO3 reruns"
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--sum-campaign", type=Path, required=True)
    parser.add_argument("--pre-campaign", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    print(
        json.dumps(
            finalize_todo3(
                campaign_root=args.campaign_root,
                sum_campaign=args.sum_campaign,
                pre_campaign=args.pre_campaign,
                repo_root=args.repo_root,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    _main()
