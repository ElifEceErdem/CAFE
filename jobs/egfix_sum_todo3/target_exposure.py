"""All-five target-coreness exposure audit from validated TODO3 NPZ files.

Exposure is deliberately an incoming *directed recommendation-event* measure,
not a graph edge-counting measure.  Reciprocal recommendation events are
therefore retained separately even though Degree EG canonicalizes the
post-recommendation graph as a simple undirected edge set.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from jobs.egfix_sum_todo3.protocol import (
    K,
    METHODS,
    PENALTY_REDUCTION,
    STRATIFIED_SEEDS,
    TARGET_EXPOSURE_SEMANTICS,
    TARGET_TIER_NAMES,
    TARGET_TIER_PERCENTILES,
)
from jobs.egfix_sum_todo3.provenance import (
    canonical_json_bytes,
    read_json,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
for source_path in (REPO_ROOT, REPO_ROOT / "backbone"):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from core.data_loader import load_dataset  # noqa: E402
from metrics.coreness import compute_coreness  # noqa: E402
from metrics.graph_utils import canonicalize_undirected_edges  # noqa: E402


BY_RUN_FILENAME = "target_exposure_by_run.csv"
SUMMARY_FILENAME = "target_exposure_summary.csv"
CONTRAST_BY_RUN_FILENAME = (
    "target_exposure_cafe_in_vs_fusion_by_run.csv"
)
CONTRAST_SUMMARY_FILENAME = (
    "target_exposure_cafe_in_vs_fusion_summary.csv"
)
INTEGRITY_FILENAME = "target_exposure_integrity.json"

DATASETS: Tuple[str, ...] = (
    "ego-facebook",
    "oklahoma",
    "unc",
    "pokec_n",
    "brightkite",
)
TARGET_KEYS = {
    "RA": "targets_ra",
    "HL-GNN": "targets_hl_gnn",
    "Fusion": "targets_fusion",
    "CAFE-In": "targets_cafe_in",
    "CAFE-Pre": "targets_cafe_pre",
}


class TargetExposureError(RuntimeError):
    """Raised when target exposure cannot be derived without ambiguity."""


def _split_fingerprint(split_edge, num_nodes: int) -> str:
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


def assign_target_tiers(
    coreness: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign right-closed coreness terciles over all graph nodes.

    The right-closed rule is explicit because Brightkite has a large
    coreness-one tie: the default left-closed ``numpy.digitize`` rule would
    create an empty peripheral tier.  Equal-coreness nodes are never split.
    """

    values = np.asarray(coreness, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise TargetExposureError(
            "Coreness must be a nonempty one-dimensional array"
        )
    if not np.isfinite(values).all() or np.any(values < 0):
        raise TargetExposureError(
            "Coreness values must be finite and nonnegative"
        )
    thresholds = np.percentile(
        values,
        TARGET_TIER_PERCENTILES,
        method="linear",
    ).astype(np.float64)
    tiers = np.digitize(values, thresholds, right=True).astype(
        np.int64, copy=False
    )
    counts = np.bincount(tiers, minlength=3).astype(np.int64)
    if len(counts) != 3 or np.any(counts <= 0):
        raise TargetExposureError(
            "Right-closed coreness percentiles produced an empty target "
            f"tier: thresholds={thresholds.tolist()}, counts={counts.tolist()}"
        )
    if int(counts.sum()) != len(values):
        raise TargetExposureError("Target-tier node counts do not sum to N")
    return tiers, thresholds, counts


def target_event_profile(
    targets: np.ndarray,
    *,
    num_nodes: int,
    num_eval_users: int,
    tiers: np.ndarray,
    tier_counts: np.ndarray,
) -> List[Dict[str, object]]:
    """Return the three target-tier event/exposure records for one method."""

    matrix = np.asarray(targets)
    if matrix.shape != (num_eval_users, K):
        raise TargetExposureError(
            f"Target matrix must have shape ({num_eval_users}, {K}), "
            f"got {matrix.shape}"
        )
    if not np.issubdtype(matrix.dtype, np.integer):
        raise TargetExposureError("Target matrix must have integer dtype")
    matrix = matrix.astype(np.int64, copy=False)
    if np.any(matrix < 0) or np.any(matrix >= num_nodes):
        raise TargetExposureError("Target ID lies outside the graph")
    if tiers.shape != (num_nodes,) or tier_counts.shape != (3,):
        raise TargetExposureError("Target-tier profile has invalid shape")

    target_events = num_eval_users * K
    event_counts = np.bincount(
        tiers[matrix.reshape(-1)], minlength=3
    ).astype(np.int64)
    if len(event_counts) != 3:
        raise TargetExposureError("Unexpected target-tier event dimension")
    if int(event_counts.sum()) != target_events:
        raise TargetExposureError(
            "Target-tier events do not sum to |U|K"
        )

    rows = []
    for tier_index, tier_name in enumerate(TARGET_TIER_NAMES):
        node_count = int(tier_counts[tier_index])
        event_count = int(event_counts[tier_index])
        event_share = event_count / target_events
        exposure_mass = event_count / num_eval_users
        mean_exposure = exposure_mass / node_count
        rows.append(
            {
                "tier_index": tier_index,
                "target_tier": tier_name,
                "tier_node_count": node_count,
                "tier_node_share": node_count / num_nodes,
                "event_count": event_count,
                "event_share": event_share,
                "exposure_mass": exposure_mass,
                "mean_exposure_per_node": mean_exposure,
            }
        )

    if not math.isclose(
        math.fsum(float(row["event_share"]) for row in rows),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise TargetExposureError("Target event shares do not sum to one")
    if not math.isclose(
        math.fsum(float(row["exposure_mass"]) for row in rows),
        K,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise TargetExposureError("Total incoming exposure mass is not K")
    weighted = math.fsum(
        float(row["mean_exposure_per_node"])
        * int(row["tier_node_count"])
        for row in rows
    )
    if not math.isclose(weighted, K, rel_tol=0.0, abs_tol=1e-12):
        raise TargetExposureError(
            "Tier-node-weighted mean exposure does not equal K"
        )
    return rows


def _validate_integrity_digest(
    document: Mapping[str, object], path: Path
) -> None:
    recorded = document.get("integrity_payload_sha256")
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise TargetExposureError(
            f"{path}: missing integrity_payload_sha256"
        )
    unsigned = dict(document)
    del unsigned["integrity_payload_sha256"]
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != recorded:
        raise TargetExposureError(
            f"{path}: core integrity self-digest mismatch"
        )


def _core_npz_hashes(
    core_integrity: Mapping[str, object],
) -> Dict[str, str]:
    records = core_integrity.get("input_artifacts")
    if not isinstance(records, list):
        raise TargetExposureError(
            "Core integrity input_artifacts must be a list"
        )
    result = {}
    for record in records:
        if not isinstance(record, dict):
            raise TargetExposureError(
                "Malformed core input-artifact record"
            )
        relative = record.get("path")
        digest = record.get("sha256")
        if (
            isinstance(relative, str)
            and relative.endswith(".npz")
            and isinstance(digest, str)
        ):
            if relative in result:
                raise TargetExposureError(
                    f"Duplicate NPZ in core integrity: {relative}"
                )
            result[relative] = digest
    if len(result) != len(DATASETS) * len(STRATIFIED_SEEDS):
        raise TargetExposureError(
            f"Core integrity must bind exactly 15 NPZs, found {len(result)}"
        )
    return result


def _expected_dataset_fingerprints(
    core_integrity: Mapping[str, object],
) -> Dict[str, str]:
    stratified = core_integrity.get("stratified")
    if not isinstance(stratified, dict):
        raise TargetExposureError(
            "Core integrity lacks stratified validation details"
        )
    shards = stratified.get("shards")
    if not isinstance(shards, list):
        raise TargetExposureError(
            "Core stratified shard records must be a list"
        )
    by_dataset = {dataset: set() for dataset in DATASETS}
    for record in shards:
        if not isinstance(record, dict):
            raise TargetExposureError("Malformed stratified shard record")
        dataset = record.get("dataset")
        fingerprint = record.get("split_fingerprint")
        if dataset not in by_dataset or not isinstance(
            fingerprint, str
        ):
            raise TargetExposureError(
                "Invalid stratified shard dataset/fingerprint"
            )
        by_dataset[str(dataset)].add(fingerprint)
    result = {}
    for dataset, fingerprints in by_dataset.items():
        if len(fingerprints) != 1:
            raise TargetExposureError(
                f"{dataset}: core integrity has inconsistent split hashes"
            )
        result[dataset] = next(iter(fingerprints))
    return result


def _load_recommendations(
    path: Path,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Mapping[str, object]]:
    expected_keys = {"sources", "metadata_json", *TARGET_KEYS.values()}
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected_keys:
                raise TargetExposureError(
                    f"{path}: unexpected recommendation keys"
                )
            sources = np.asarray(archive["sources"]).astype(
                np.int64, copy=False
            )
            raw_metadata = np.asarray(archive["metadata_json"])
            if raw_metadata.shape != ():
                raise TargetExposureError(
                    f"{path}: metadata_json must be scalar"
                )
            metadata = json.loads(str(raw_metadata.item()))
            if not isinstance(metadata, dict):
                raise TargetExposureError(
                    f"{path}: metadata_json must decode to an object"
                )
            targets = {
                method: np.asarray(archive[key]).astype(
                    np.int64, copy=False
                )
                for method, key in TARGET_KEYS.items()
            }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, TargetExposureError):
            raise
        raise TargetExposureError(
            f"Cannot read recommendation artifact {path}: {exc}"
        ) from exc
    return sources, targets, metadata


def _load_fixed_split(repo_root: Path, dataset: str):
    args = SimpleNamespace(
        data_name=dataset,
        data_path=str(repo_root / "data" / "SNAP"),
        val_ratio=0.05,
        test_ratio=0.1,
        encoder="HLGNN",
        device="cpu",
        data_split_seed=42,
        split_file=(
            str(
                repo_root
                / "data"
                / "SNAP"
                / "brightkite"
                / "temporal_split.pt"
            )
            if dataset == "brightkite"
            else None
        ),
    )
    try:
        loaded, split_edge, _, _ = load_dataset(args)
    except Exception as exc:
        raise TargetExposureError(
            f"{dataset}: cannot reconstruct fixed split: {exc}"
        ) from exc
    num_nodes = (
        loaded._data.num_nodes
        if hasattr(loaded, "_data")
        else loaded.data.num_nodes
    )
    return int(num_nodes), split_edge


def _sample_std(values: pd.Series) -> float:
    result = float(values.std(ddof=1))
    return 0.0 if not math.isfinite(result) else result


def _direction_count(values: pd.Series, direction: str) -> int:
    array = np.asarray(values, dtype=float)
    zero = np.isclose(array, 0.0, rtol=0.0, atol=1e-15)
    if direction == "positive":
        return int(np.sum((array > 0) & ~zero))
    if direction == "negative":
        return int(np.sum((array < 0) & ~zero))
    if direction == "zero":
        return int(np.sum(zero))
    raise TargetExposureError(f"Unknown direction {direction!r}")


def _summaries(
    by_run: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_fields = [
        "dataset",
        "method",
        "tier_index",
        "target_tier",
        "q_1_3",
        "q_2_3",
        "tier_node_count",
        "tier_node_share",
        "num_nodes",
        "num_eval_users",
        "k",
    ]
    summary = (
        by_run.groupby(group_fields, as_index=False, dropna=False)
        .agg(
            run_count=("training_seed", "count"),
            event_count_mean=("event_count", "mean"),
            event_count_std=("event_count", _sample_std),
            event_share_mean=("event_share", "mean"),
            event_share_std=("event_share", _sample_std),
            exposure_mass_mean=("exposure_mass", "mean"),
            exposure_mass_std=("exposure_mass", _sample_std),
            mean_exposure_per_node_mean=(
                "mean_exposure_per_node",
                "mean",
            ),
            mean_exposure_per_node_std=(
                "mean_exposure_per_node",
                _sample_std,
            ),
        )
    )
    if not summary["run_count"].eq(len(STRATIFIED_SEEDS)).all():
        raise TargetExposureError(
            "Target exposure summary lacks three runs per cell"
        )
    for base in (
        "event_count",
        "event_share",
        "exposure_mass",
        "mean_exposure_per_node",
    ):
        summary[f"{base}_sem"] = summary[f"{base}_std"] / np.sqrt(
            summary["run_count"]
        )

    cafe = by_run.loc[
        by_run["method"] == "CAFE-In"
    ].copy()
    fusion = by_run.loc[
        by_run["method"] == "Fusion"
    ].copy()
    keys = [
        "dataset",
        "training_seed",
        "split_seed",
        "split_fingerprint",
        "tier_index",
        "target_tier",
        "q_1_3",
        "q_2_3",
        "tier_node_count",
        "tier_node_share",
        "num_nodes",
        "num_eval_users",
        "k",
    ]
    cafe = cafe.rename(
        columns={
            "event_count": "cafe_in_event_count",
            "event_share": "cafe_in_event_share",
            "exposure_mass": "cafe_in_exposure_mass",
            "mean_exposure_per_node": (
                "cafe_in_mean_exposure_per_node"
            ),
            "recommendations_path": "cafe_in_recommendations_path",
            "recommendations_sha256": (
                "cafe_in_recommendations_sha256"
            ),
        }
    )
    fusion = fusion.rename(
        columns={
            "event_count": "fusion_event_count",
            "event_share": "fusion_event_share",
            "exposure_mass": "fusion_exposure_mass",
            "mean_exposure_per_node": (
                "fusion_mean_exposure_per_node"
            ),
            "recommendations_path": "fusion_recommendations_path",
            "recommendations_sha256": (
                "fusion_recommendations_sha256"
            ),
        }
    )
    value_fields = [
        "cafe_in_event_count",
        "cafe_in_event_share",
        "cafe_in_exposure_mass",
        "cafe_in_mean_exposure_per_node",
        "cafe_in_recommendations_path",
        "cafe_in_recommendations_sha256",
    ]
    fusion_fields = [
        "fusion_event_count",
        "fusion_event_share",
        "fusion_exposure_mass",
        "fusion_mean_exposure_per_node",
        "fusion_recommendations_path",
        "fusion_recommendations_sha256",
    ]
    contrast = cafe[keys + value_fields].merge(
        fusion[keys + fusion_fields],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if len(contrast) != len(DATASETS) * len(STRATIFIED_SEEDS) * 3:
        raise TargetExposureError(
            "CAFE-In/Fusion paired target matrix is incomplete"
        )
    if (
        contrast["cafe_in_recommendations_path"]
        != contrast["fusion_recommendations_path"]
    ).any() or (
        contrast["cafe_in_recommendations_sha256"]
        != contrast["fusion_recommendations_sha256"]
    ).any():
        raise TargetExposureError(
            "Paired CAFE-In/Fusion rows do not share one NPZ artifact"
        )
    denominator = contrast["fusion_mean_exposure_per_node"].astype(float)
    if (denominator <= 0).any():
        raise TargetExposureError(
            "Fusion has zero target-tier exposure; relative change is "
            "undefined and will not be epsilon-smoothed"
        )
    contrast["event_count_delta"] = (
        contrast["cafe_in_event_count"]
        - contrast["fusion_event_count"]
    )
    contrast["event_share_delta"] = (
        contrast["cafe_in_event_share"]
        - contrast["fusion_event_share"]
    )
    contrast["exposure_mass_delta"] = (
        contrast["cafe_in_exposure_mass"]
        - contrast["fusion_exposure_mass"]
    )
    contrast["mean_exposure_per_node_delta"] = (
        contrast["cafe_in_mean_exposure_per_node"]
        - contrast["fusion_mean_exposure_per_node"]
    )
    contrast["relative_change_percent"] = 100.0 * (
        contrast["cafe_in_mean_exposure_per_node"] / denominator - 1.0
    )

    contrast_groups = [
        "dataset",
        "tier_index",
        "target_tier",
        "q_1_3",
        "q_2_3",
        "tier_node_count",
        "tier_node_share",
        "num_nodes",
        "num_eval_users",
        "k",
    ]
    contrast_summary = (
        contrast.groupby(
            contrast_groups, as_index=False, dropna=False
        )
        .agg(
            run_count=("training_seed", "count"),
            cafe_in_mean_exposure=(
                "cafe_in_mean_exposure_per_node",
                "mean",
            ),
            fusion_mean_exposure=(
                "fusion_mean_exposure_per_node",
                "mean",
            ),
            absolute_delta_mean=(
                "mean_exposure_per_node_delta",
                "mean",
            ),
            absolute_delta_std=(
                "mean_exposure_per_node_delta",
                _sample_std,
            ),
            paired_relative_change_percent_mean=(
                "relative_change_percent",
                "mean",
            ),
            paired_relative_change_percent_std=(
                "relative_change_percent",
                _sample_std,
            ),
            event_share_delta_mean=("event_share_delta", "mean"),
            event_share_delta_std=(
                "event_share_delta",
                _sample_std,
            ),
            positive_runs=(
                "mean_exposure_per_node_delta",
                lambda values: _direction_count(values, "positive"),
            ),
            zero_runs=(
                "mean_exposure_per_node_delta",
                lambda values: _direction_count(values, "zero"),
            ),
            negative_runs=(
                "mean_exposure_per_node_delta",
                lambda values: _direction_count(values, "negative"),
            ),
        )
    )
    if not contrast_summary["run_count"].eq(
        len(STRATIFIED_SEEDS)
    ).all():
        raise TargetExposureError(
            "Target contrast summary lacks three paired runs per cell"
        )
    contrast_summary["absolute_delta_sem"] = (
        contrast_summary["absolute_delta_std"]
        / np.sqrt(contrast_summary["run_count"])
    )
    contrast_summary["paired_relative_change_percent_sem"] = (
        contrast_summary["paired_relative_change_percent_std"]
        / np.sqrt(contrast_summary["run_count"])
    )
    contrast_summary["event_share_delta_sem"] = (
        contrast_summary["event_share_delta_std"]
        / np.sqrt(contrast_summary["run_count"])
    )
    contrast_summary["ratio_of_run_means_percent"] = 100.0 * (
        contrast_summary["cafe_in_mean_exposure"]
        / contrast_summary["fusion_mean_exposure"]
        - 1.0
    )
    if (
        contrast_summary["positive_runs"]
        + contrast_summary["zero_runs"]
        + contrast_summary["negative_runs"]
        != contrast_summary["run_count"]
    ).any():
        raise TargetExposureError(
            "Target contrast direction counts do not sum to run count"
        )
    return summary, contrast, contrast_summary


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


def _atomic_write_json(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                document,
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


def _sort(
    frame: pd.DataFrame, *, include_method: bool = True
) -> pd.DataFrame:
    result = frame.copy()
    dataset_order = {name: index for index, name in enumerate(DATASETS)}
    method_order = {name: index for index, name in enumerate(METHODS)}
    result["_dataset_order"] = result["dataset"].map(dataset_order)
    columns = ["_dataset_order"]
    if "training_seed" in result:
        columns.append("training_seed")
    columns.append("tier_index")
    if include_method and "method" in result:
        result["_method_order"] = result["method"].map(method_order)
        columns.append("_method_order")
    result = result.sort_values(columns, kind="stable").reset_index(
        drop=True
    )
    return result.drop(
        columns=[
            column
            for column in ("_dataset_order", "_method_order")
            if column in result
        ]
    )


def generate_target_exposure(
    *,
    campaign_root: Path,
    core_finalized_dir: Path,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, object]:
    """Derive and persist the all-five target exposure audit."""

    campaign = campaign_root.resolve()
    output_dir = core_finalized_dir.resolve()
    core_integrity_path = output_dir / "rerun_integrity.json"
    if not core_integrity_path.is_file():
        raise TargetExposureError(
            f"Core rerun integrity is missing: {core_integrity_path}"
        )
    core_integrity = read_json(core_integrity_path)
    _validate_integrity_digest(core_integrity, core_integrity_path)
    protocol = core_integrity.get("protocol")
    if not isinstance(protocol, dict) or protocol.get(
        "penalty_reduction"
    ) != PENALTY_REDUCTION:
        raise TargetExposureError(
            "Core finalizer did not validate sum-reduction reruns"
        )
    crosscheck = core_integrity.get("degree_eg_crosscheck")
    if not isinstance(crosscheck, dict) or crosscheck.get(
        "status"
    ) != "passed":
        raise TargetExposureError(
            "Corrected Degree-EG cross-path validation did not pass"
        )
    npz_hashes = _core_npz_hashes(core_integrity)
    expected_fingerprints = _expected_dataset_fingerprints(
        core_integrity
    )

    rows: List[Dict[str, object]] = []
    tier_profiles = {}
    ra_reference = {}
    for dataset in DATASETS:
        num_nodes, split_edge = _load_fixed_split(repo_root, dataset)
        fingerprint = _split_fingerprint(split_edge, num_nodes)
        if fingerprint != expected_fingerprints[dataset]:
            raise TargetExposureError(
                f"{dataset}: reconstructed split fingerprint differs "
                "from the core rerun"
            )
        test_edges = split_edge["test"]["edge"].cpu()
        expected_sources = np.sort(
            np.unique(
                test_edges.reshape(-1).numpy().astype(np.int64)
            )
        )
        coreness = (
            compute_coreness(
                split_edge["train"]["edge"].cpu(), num_nodes
            )
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        tiers, thresholds, tier_counts = assign_target_tiers(coreness)
        tier_profiles[dataset] = {
            "num_nodes": num_nodes,
            "q_1_3": float(thresholds[0]),
            "q_2_3": float(thresholds[1]),
            "tier_counts": {
                TARGET_TIER_NAMES[index]: int(tier_counts[index])
                for index in range(3)
            },
            "split_fingerprint": fingerprint,
        }

        for seed in STRATIFIED_SEEDS:
            rec_path = (
                campaign
                / "stratified"
                / dataset
                / f"seed_{seed}"
                / f"recommendations_{dataset}_seed{seed}.npz"
            )
            try:
                relative = rec_path.resolve().relative_to(
                    campaign
                ).as_posix()
            except ValueError as exc:
                raise TargetExposureError(
                    f"Recommendation artifact escapes campaign: {rec_path}"
                ) from exc
            recorded_sha = npz_hashes.get(relative)
            actual_sha = sha256_file(rec_path)
            if recorded_sha != actual_sha:
                raise TargetExposureError(
                    f"{rec_path}: hash is absent from or differs from "
                    "core integrity"
                )
            sources, target_matrices, metadata = _load_recommendations(
                rec_path
            )
            if not np.array_equal(sources, expected_sources):
                raise TargetExposureError(
                    f"{rec_path}: sources are not exactly fixed-test-active "
                    "users"
                )
            num_users = len(sources)
            required_metadata = {
                "dataset": dataset,
                "training_seed": seed,
                "split_seed": 42,
                "split_fingerprint": fingerprint,
                "num_nodes": num_nodes,
                "num_eval_users": num_users,
                "k": K,
                "penalty_reduction": PENALTY_REDUCTION,
            }
            for field, expected in required_metadata.items():
                if metadata.get(field) != expected:
                    raise TargetExposureError(
                        f"{rec_path}: metadata {field} mismatch"
                    )

            if dataset not in ra_reference:
                ra_reference[dataset] = target_matrices["RA"].copy()
            elif not np.array_equal(
                ra_reference[dataset], target_matrices["RA"]
            ):
                raise TargetExposureError(
                    f"{dataset}: deterministic RA target matrix differs "
                    "across fixed-split training seeds"
                )

            for method in METHODS:
                profile = target_event_profile(
                    target_matrices[method],
                    num_nodes=num_nodes,
                    num_eval_users=num_users,
                    tiers=tiers,
                    tier_counts=tier_counts,
                )
                for record in profile:
                    rows.append(
                        {
                            "dataset": dataset,
                            "training_seed": seed,
                            "split_seed": 42,
                            "split_fingerprint": fingerprint,
                            "method": method,
                            "target_exposure_semantics": (
                                TARGET_EXPOSURE_SEMANTICS
                            ),
                            "tier_boundary": "right_closed",
                            "percentile_method": "linear",
                            "q_1_3": float(thresholds[0]),
                            "q_2_3": float(thresholds[1]),
                            "num_nodes": num_nodes,
                            "num_eval_users": num_users,
                            "k": K,
                            "target_events": num_users * K,
                            **record,
                            "recommendations_path": relative,
                            "recommendations_sha256": actual_sha,
                        }
                    )

    by_run = pd.DataFrame(rows)
    if len(by_run) != (
        len(DATASETS) * len(STRATIFIED_SEEDS) * len(METHODS) * 3
    ):
        raise TargetExposureError(
            f"Expected 225 target-by-run rows, found {len(by_run)}"
        )
    identity = [
        "dataset",
        "training_seed",
        "method",
        "tier_index",
    ]
    if by_run[identity].duplicated().any():
        raise TargetExposureError("Duplicate target exposure run cell")
    numeric = by_run.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise TargetExposureError(
            "Target exposure output contains a non-finite number"
        )
    if (
        by_run[
            [
                "tier_node_count",
                "tier_node_share",
                "event_count",
                "event_share",
                "exposure_mass",
                "mean_exposure_per_node",
            ]
        ]
        < 0
    ).any().any():
        raise TargetExposureError(
            "Target exposure output contains a negative value"
        )

    by_run = _sort(by_run)
    summary, contrast, contrast_summary = _summaries(by_run)
    summary = _sort(summary)
    contrast = _sort(contrast, include_method=False)
    contrast_summary = _sort(
        contrast_summary, include_method=False
    )
    if len(summary) != 75 or len(contrast) != 45 or len(
        contrast_summary
    ) != 15:
        raise TargetExposureError(
            "Target exposure summary row counts are not 75/45/15"
        )

    output_frames = (
        (BY_RUN_FILENAME, by_run),
        (SUMMARY_FILENAME, summary),
        (CONTRAST_BY_RUN_FILENAME, contrast),
        (CONTRAST_SUMMARY_FILENAME, contrast_summary),
    )
    for filename, frame in output_frames:
        _atomic_write_frame(output_dir / filename, frame)

    artifacts = [
        {
            "path": filename,
            "sha256": sha256_file(output_dir / filename),
            "size_bytes": (output_dir / filename).stat().st_size,
            "rows": len(frame),
        }
        for filename, frame in output_frames
    ]
    integrity: Dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "target_exposure_semantics": TARGET_EXPOSURE_SEMANTICS,
        "reciprocal_recommendation_events_deduplicated": False,
        "population": "all graph nodes including zero-exposure nodes",
        "normalization": "incoming target events divided by eval-user count",
        "tier_rule": {
            "percentiles": list(TARGET_TIER_PERCENTILES),
            "percentile_method": "linear",
            "boundary": "right_closed",
            "periphery": "coreness <= q_1_3",
            "middle": "q_1_3 < coreness <= q_2_3",
            "core": "coreness > q_2_3",
        },
        "datasets": list(DATASETS),
        "training_seeds": list(STRATIFIED_SEEDS),
        "methods": list(METHODS),
        "contrast": "CAFE-In minus matched-beta Fusion",
        "uncertainty_unit": "training seed",
        "core_integrity_path": "rerun_integrity.json",
        "core_integrity_sha256": sha256_file(core_integrity_path),
        "recommendation_artifact_count": len(npz_hashes),
        "tier_profiles": tier_profiles,
        "row_counts": {
            BY_RUN_FILENAME: len(by_run),
            SUMMARY_FILENAME: len(summary),
            CONTRAST_BY_RUN_FILENAME: len(contrast),
            CONTRAST_SUMMARY_FILENAME: len(contrast_summary),
        },
        "invariants": {
            "all_nodes_included": True,
            "zero_exposure_nodes_included": True,
            "events_per_method_run": "|U| * K",
            "event_shares_sum_to_one": True,
            "exposure_mass_sums_to_k": True,
            "fixed_tiers_across_seeds": True,
            "ra_targets_identical_across_seeds": True,
            "fusion_relative_denominators_positive": True,
        },
        "outputs": artifacts,
    }
    integrity["integrity_payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(integrity)
    ).hexdigest()
    _atomic_write_json(output_dir / INTEGRITY_FILENAME, integrity)
    return {
        "status": "complete",
        "by_run_rows": len(by_run),
        "summary_rows": len(summary),
        "contrast_by_run_rows": len(contrast),
        "contrast_summary_rows": len(contrast_summary),
        # Keep this path relative to the finalized artifact directory.  The
        # caller atomically renames its private staging directory after this
        # function returns.
        "integrity_file": INTEGRITY_FILENAME,
        "integrity_sha256": sha256_file(
            output_dir / INTEGRITY_FILENAME
        ),
        "tier_profiles": tier_profiles,
    }
