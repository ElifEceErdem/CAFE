#!/usr/bin/env python3
"""Run and finalize the fixed-configuration K-sensitivity experiment.

The experiment evaluates K in {5, 20} as prefixes of one top-20 ranking.
Internally K=10 is evaluated as well because ``run_significance.py`` uses it
for its resumability contract; K=10 is not included in the paper table.

Each ``run`` invocation owns one (dataset, training-seed) shard and delegates
model training and metric computation to the existing paired five-method
runner.  ``finalize`` strictly requires the complete five-dataset by ten-seed
matrix before producing per-seed, summary, paired-contrast, and LaTeX outputs.

Examples
--------
Run one shard (normally done by ``reproduce/jobs/t20_k_sensitivity.job``)::

    python -u experiments/run_k_sensitivity.py run \
        --dataset unc --seed 42 \
        --campaign-dir results_k_sensitivity/corrected_k5_k20_v1 \
        --selected-configs results_egfix/20260724_degree_eg_sum_in_v1/todo3/selected_configs.csv

Finalize only after all shards exist::

    python -u experiments/run_k_sensitivity.py finalize \
        --campaign-dir results_k_sensitivity/corrected_k5_k20_v1

Scientific safety
-----------------
The current checkout has a known Eq. (8) shape defect: an MLP prediction of
shape [B,1] is multiplied by a coreness vector of shape [B], broadcasting to
[B,B].  By default this script tests the edgewise identity and stops before
training if it fails.  ``--allow-nonedgewise-cafe-in`` exists only to produce
an explicitly labelled legacy-implementation sensitivity artifact; such an
artifact is not evidence for Eq. (8) as written in the manuscript.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
PROFILE = "fixed_config_k5_k20_v1"
SCHEMA_VERSION = 1
DATASETS = ("ego-facebook", "oklahoma", "unc", "pokec_n", "brightkite")
SEEDS = tuple(range(42, 52))
METHODS = ("RA", "HL-GNN", "Fusion", "CAFE-In", "CAFE-Pre")
EVALUATED_K = (5, 10, 20)
REPORTED_K = (5, 20)


def _current_eg_semantics() -> str:
    """Equality-Gain definition of the metric code in use (see core/statistics.py)."""
    from metrics import equality_gain as _eg
    if hasattr(_eg, "SIMPLE_UNDIRECTED"):
        return "simple_undirected_set_union"
    return "legacy_indegree_reconstructed"


EG_SEMANTICS = _current_eg_semantics()
# Graphs on which --exact-closeness replaces the sampled (ApproxCloseness)
# evaluation of the legacy code; smaller graphs are already evaluated exactly.
EXACT_CLOSENESS_DATASETS = ("pokec_n", "brightkite")
EXACT_CLOSENESS_VERSION = "exact_generalized_v1_float32_positive_support"
DEFAULT_PENALTY_REDUCTION = "sum"
KNOWN_NONEDGEWISE_CONFIG_SHA256 = (
    "57b80781dbd15cc367c8c857d92ff535d63084a20ed2110438185f4e49fb13df"
)

METRIC_PREFIXES = {
    "precision": "Precision",
    "degree_eg": "Equality_Gain",
    "pagerank_eg": "Pagerank_EG",
    "closeness_eg": "Closeness_EG",
}

CODE_FILES = (
    "experiments/run_k_sensitivity.py",
    "reproduce/jobs/t20_k_sensitivity.job",
    "experiments/run_with.py",
    "metrics/closeness_exact_v1.py",
    "experiments/configs/verify_legacy_best_configs.py",
    "experiments/run_significance.py",
    "experiments/cafe_methods.py",
    "fairness_methods/inprocessing.py",
    "backbone/model.py",
    "backbone/layer.py",
    "backbone/loss.py",
    "backbone/snap_dataset.py",
    "core/data_loader.py",
    "metrics/precision.py",
    "metrics/equality_gain.py",
    "metrics/centrality_equality_gain.py",
    "metrics/graph_utils.py",
)

DISPLAY_DATASET = {
    "ego-facebook": "Ego-FB",
    "oklahoma": "Oklahoma",
    "unc": "UNC",
    "pokec_n": "Pokec-n",
    "brightkite": "Brightkite",
}


class KSensitivityError(RuntimeError):
    """Raised when a shard or final table violates the frozen protocol."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    _atomic_write_bytes(path, _canonical_json(value))


def _atomic_write_text(path: Path, value: str) -> None:
    _atomic_write_bytes(path, value.encode("utf-8"))


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        with open(temporary, "rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _relative_or_absolute(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def _normalise_dataset(value: str) -> str:
    normal = value.strip().lower().replace("-", "_")
    aliases = {
        "ego_facebook": "ego-facebook",
        "ego_fb": "ego-facebook",
        "oklahoma": "oklahoma",
        "unc": "unc",
        "pokec_n": "pokec_n",
        "brightkite": "brightkite",
    }
    try:
        return aliases[normal]
    except KeyError as exc:
        raise KSensitivityError(f"Unsupported dataset: {value!r}") from exc


def _required_metric_names(k_values: Sequence[int] = EVALUATED_K) -> set[str]:
    return {
        f"{prefix}@{k}"
        for prefix in METRIC_PREFIXES.values()
        for k in k_values
    }


def _read_selected_configs(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise KSensitivityError(f"Selected-config file not found: {path}")
    frame = pd.read_csv(path)
    required = {
        "dataset",
        "variant",
        "lambda_fair",
        "preproc_alpha",
        "beta",
        "lambda_mobility",
    }
    missing = required - set(frame.columns)
    if missing:
        raise KSensitivityError(
            f"Selected-config file lacks columns: {sorted(missing)}"
        )
    normal = frame["dataset"].map(_normalise_dataset)
    for dataset in DATASETS:
        rows = frame.loc[normal == dataset]
        if len(rows) != 2 or set(rows["variant"]) != {"cafe_in", "cafe_pre"}:
            raise KSensitivityError(
                f"Expected one cafe_in and one cafe_pre row for {dataset}; "
                f"found {rows[['dataset', 'variant']].to_dict('records')}"
            )
    return frame


def _lineage_selected_config_hashes(value: object) -> set[str]:
    """Collect selected-config hashes from a JSON lineage document."""

    hashes: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "selected_configs_sha256" and isinstance(child, str):
                hashes.add(child.lower())
            hashes.update(_lineage_selected_config_hashes(child))
    elif isinstance(value, list):
        for child in value:
            hashes.update(_lineage_selected_config_hashes(child))
    return hashes


def _validate_selection_lineage(path: Path, selected_sha256: str) -> None:
    """Require a JSON lineage artifact tied to the supplied configurations."""

    if not path.is_file():
        raise KSensitivityError(f"Selection-lineage artifact not found: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise KSensitivityError(
            f"Cannot read selection-lineage JSON {path}: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise KSensitivityError(f"Selection lineage must be a JSON object: {path}")
    recorded = _lineage_selected_config_hashes(document)
    if selected_sha256.lower() not in recorded:
        raise KSensitivityError(
            f"{path} does not bind the supplied selected-config SHA256 "
            f"{selected_sha256}"
        )


def _edgewise_penalty_preflight() -> bool:
    """Return True iff the implemented CAFE-In penalty is edgewise.

    The deliberately non-constant scores and weights distinguish the intended
    matched-vector mean from the erroneous broadcasted matrix mean.
    """

    import torch

    from fairness_methods.inprocessing import CorenessInverseLoss

    penalty = CorenessInverseLoss(
        base_loss_name="AUC",
        lambda_fair=1.0,
        alpha=1.0,
        penalty_reduction="mean",
    )
    positive_edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    negative_edges = torch.tensor([[[0, 2]], [[2, 0]]], dtype=torch.long)
    positive_scores = torch.tensor([[-1.0], [2.0]], dtype=torch.float32)
    negative_scores = torch.zeros((2, 1), dtype=torch.float32)
    coreness = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32)

    actual = penalty.compute_fairness_penalty(
        positive_edges,
        negative_edges,
        positive_scores,
        negative_scores,
        coreness=coreness,
    )
    weights = (
        coreness[positive_edges[:, 0]] * coreness[positive_edges[:, 1]]
    ) / (coreness.max() * coreness.max())
    expected = (torch.sigmoid(positive_scores.squeeze(-1)) * weights).mean()
    return bool(torch.allclose(actual, expected, rtol=1e-6, atol=1e-7))


def _code_hashes() -> Dict[str, str]:
    result = {}
    for relative in CODE_FILES:
        path = REPO / relative
        if not path.is_file():
            raise KSensitivityError(f"Protocol source file is absent: {path}")
        result[relative] = _sha256(path)
    return result


def _protocol_document(
    *,
    selected_configs: Path,
    upstream_lineage: Path | None,
    runtime_manifest: Path | None,
    edgewise_penalty: bool,
    allow_nonedgewise: bool,
    penalty_reduction: str,
    configuration_origin: str,
    exact_closeness: bool = False,
) -> Dict[str, object]:
    if configuration_origin == "legacy_test_selected":
        configuration_policy = (
            "freeze the historical K=10 test-selected operating points; "
            "do not reselect by evaluation cutoff"
        )
    else:
        configuration_policy = (
            "freeze the K=10 validation-selected configurations; "
            "do not reselect by evaluation cutoff"
        )
    document: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "profile": PROFILE,
        "scope": "fixed-configuration evaluation-cutoff sensitivity",
        "datasets": list(DATASETS),
        "training_seeds": list(SEEDS),
        "split_seed": 42,
        "brightkite_split": "data/SNAP/brightkite/temporal_split.pt",
        "methods": list(METHODS),
        "evaluated_k_values": list(EVALUATED_K),
        "reported_k_values": list(REPORTED_K),
        "selection_k": 10,
        "configuration_origin": configuration_origin,
        "configuration_policy": configuration_policy,
        "metrics": list(METRIC_PREFIXES),
        "evaluation_split": "test",
        "training_time_test_evaluation": False,
        "eg_graph_semantics": EG_SEMANTICS,
        "penalty_reduction": penalty_reduction,
        "loss_func": "AUC",
        "num_neg": 1,
        "batch_size": 16384,
        "epochs": 500,
        "learning_rate": 0.001,
        "closeness": {
            "random_seed": 42,
            "sample_size": 0,
            "epsilon": 0.1,
            "exact_generalized_datasets": (
                list(EXACT_CLOSENESS_DATASETS) if exact_closeness else []
            ),
            "exact_generalized_version": (
                EXACT_CLOSENESS_VERSION if exact_closeness else None
            ),
        },
        "large_graph_training_validation": {
            "datasets": ["pokec_n", "brightkite"],
            "eval_max_nodes": 5000,
            "eval_steps": 25,
            "patience": 6,
        },
        "selected_configs": {
            "path": _relative_or_absolute(selected_configs),
            "sha256": _sha256(selected_configs),
        },
        "eq8_edgewise_penalty_preflight": edgewise_penalty,
        "allow_nonedgewise_cafe_in": allow_nonedgewise,
        "selected_configs_known_nonedgewise": (
            _sha256(selected_configs) == KNOWN_NONEDGEWISE_CONFIG_SHA256
        ),
        "publishable": (
            edgewise_penalty
            and not allow_nonedgewise
            and _sha256(selected_configs) != KNOWN_NONEDGEWISE_CONFIG_SHA256
        ),
        "code_sha256": _code_hashes(),
    }
    if upstream_lineage is not None:
        if not upstream_lineage.is_file():
            raise KSensitivityError(
                f"Upstream-lineage artifact not found: {upstream_lineage}"
            )
        document["upstream_lineage"] = {
            "path": _relative_or_absolute(upstream_lineage),
            "sha256": _sha256(upstream_lineage),
            "role": "configuration lineage only; not the K-sensitivity protocol",
        }
    if runtime_manifest is not None:
        if not runtime_manifest.is_file():
            raise KSensitivityError(
                f"Runtime manifest not found: {runtime_manifest}"
            )
        document["runtime_environment_sha256"] = _sha256(runtime_manifest)
    return document


def _create_or_verify_protocol(campaign: Path, document: Mapping[str, object]) -> Path:
    campaign.mkdir(parents=True, exist_ok=True)
    path = campaign / "k_sensitivity_protocol.json"
    payload = _canonical_json(document)
    lock_path = campaign / ".protocol.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            if path.read_bytes() != payload:
                raise KSensitivityError(
                    "Campaign protocol differs from this task. Refusing to mix "
                    f"code, configuration, runtime, or safety policies in {campaign}."
                )
        else:
            _atomic_write_bytes(path, payload)
    return path


def _task_dir(campaign: Path, dataset: str, seed: int) -> Path:
    return campaign / "shards" / dataset / f"seed_{seed}"


def _unique_strings(frame: pd.DataFrame, column: str) -> set[str]:
    values = frame[column].dropna().astype(str)
    return set(values)


def _validate_shard(
    *,
    csv_path: Path,
    dataset: str,
    seed: int,
    selected_configs_sha256: str,
    protocol_sha256: str,
    penalty_reduction: str,
) -> Dict[str, object]:
    if not csv_path.is_file():
        raise KSensitivityError(f"Shard CSV not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    required_columns = {
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
        "split_fingerprint",
        "selected_configs_sha256",
        "source_manifest_sha256",
    }
    missing = required_columns - set(frame.columns)
    if missing:
        raise KSensitivityError(f"{csv_path}: missing columns {sorted(missing)}")
    if frame.empty:
        raise KSensitivityError(f"{csv_path}: empty shard")
    if set(frame["seed"].astype(int)) != {seed}:
        raise KSensitivityError(f"{csv_path}: seed is not exactly {seed}")
    if set(frame["split_seed"].astype(int)) != {42}:
        raise KSensitivityError(f"{csv_path}: split seed is not exactly 42")
    observed_datasets = {_normalise_dataset(x) for x in frame["dataset"].astype(str)}
    if observed_datasets != {dataset}:
        raise KSensitivityError(
            f"{csv_path}: dataset mismatch {sorted(observed_datasets)}"
        )
    if set(frame["method"].astype(str)) != set(METHODS):
        raise KSensitivityError(
            f"{csv_path}: method set differs from {list(METHODS)}"
        )
    expected_singletons = {
        "protocol": "headline",
        "evaluation_split": "test",
        "eg_graph_semantics": EG_SEMANTICS,
        "penalty_reduction": penalty_reduction,
        "selected_configs_sha256": selected_configs_sha256,
        "source_manifest_sha256": protocol_sha256,
    }
    for column, wanted in expected_singletons.items():
        actual = _unique_strings(frame, column)
        if actual != {str(wanted)}:
            raise KSensitivityError(
                f"{csv_path}: {column} must be {wanted!r}, got {sorted(actual)}"
            )
    fingerprints = _unique_strings(frame, "split_fingerprint")
    if len(fingerprints) != 1 or not next(iter(fingerprints), ""):
        raise KSensitivityError(f"{csv_path}: invalid split fingerprint")

    required_metrics = _required_metric_names()
    required_rows = frame.loc[frame["metric"].isin(required_metrics)].copy()
    duplicates = required_rows.duplicated(subset=["method", "metric"], keep=False)
    if duplicates.any():
        bad = required_rows.loc[duplicates, ["method", "metric"]].to_dict("records")
        raise KSensitivityError(f"{csv_path}: duplicate required rows: {bad}")
    for method in METHODS:
        present = set(required_rows.loc[required_rows["method"] == method, "metric"])
        if present != required_metrics:
            raise KSensitivityError(
                f"{csv_path}: {method} metrics differ; missing "
                f"{sorted(required_metrics - present)}, extra {sorted(present - required_metrics)}"
            )
    values = pd.to_numeric(required_rows["value"], errors="coerce").to_numpy()
    if not np.isfinite(values).all():
        raise KSensitivityError(f"{csv_path}: required metric contains NaN/Inf")

    return {
        "dataset": dataset,
        "seed": seed,
        "split_seed": 42,
        "split_fingerprint": next(iter(fingerprints)),
        "csv_sha256": _sha256(csv_path),
        "row_count": int(len(frame)),
        "required_metric_row_count": int(len(required_rows)),
    }


def _stream_subprocess(command: Sequence[str], log_path: Path) -> tuple[int, list[str]]:
    failure_lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=str(REPO),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            if (
                "Failed to compute control centrality" in line
                or "Failed to compute (" in line
            ):
                failure_lines.append(line.strip())
        return process.wait(), failure_lines


def _runner_command(
    *,
    dataset: str,
    seed: int,
    data_path: Path,
    selected_configs: Path,
    protocol_path: Path,
    out_csv: Path,
    penalty_reduction: str,
    exact_closeness: bool = False,
    record_dir: Path | None = None,
) -> list[str]:
    launcher = []
    if exact_closeness and dataset in EXACT_CLOSENESS_DATASETS:
        # experiments/run_with.py activates metrics/closeness_exact_v1.py for
        # the runner process and fails if no closeness evaluation used it.
        launcher = [str(REPO / "experiments" / "run_with.py"), "--exact-closeness"]
        if record_dir is not None:
            launcher += ["--record-dir", str(record_dir)]
    command = [
        sys.executable,
        "-u",
        *launcher,
        str(REPO / "experiments" / "run_significance.py"),
        "--data_name",
        dataset,
        "--data_path",
        str(data_path),
        "--encoder",
        "HLGNN",
        "--predictor",
        "MLP",
        "--epochs",
        "500",
        "--batch_size",
        "16384",
        "--lr",
        "0.001",
        "--loss_func",
        "AUC",
        "--num_neg",
        "1",
        "--seed_start",
        str(seed),
        "--num_seeds",
        "1",
        "--vary_split",
        "False",
        "--protocol",
        "headline",
        "--selected_configs",
        str(selected_configs),
        "--penalty_reduction",
        penalty_reduction,
        "--skip_final_comprehensive_eval",
        "True",
        "--eval_skip_test",
        "True",
        "--k_values",
        ",".join(str(k) for k in EVALUATED_K),
        "--centrality_types",
        "pagerank,closeness",
        "--closeness_seed",
        "42",
        "--closeness_sample_size",
        "0",
        "--closeness_epsilon",
        "0.1",
        "--use_demographics",
        "False",
        "--source_manifest",
        str(protocol_path),
        "--out_csv",
        str(out_csv),
    ]
    if dataset in {"pokec_n", "brightkite"}:
        command += [
            "--eval_max_nodes",
            "5000",
            "--eval_steps",
            "25",
            "--patience",
            "6",
        ]
    else:
        command += ["--eval_steps", "10", "--patience", "50"]
    if dataset == "brightkite":
        command += [
            "--split_file",
            str(data_path / "brightkite" / "temporal_split.pt"),
        ]
    return command


def run_shard(args: argparse.Namespace) -> int:
    dataset = _normalise_dataset(args.dataset)
    seed = int(args.seed)
    if seed not in SEEDS:
        raise KSensitivityError(
            f"Seed {seed} is outside the frozen set {list(SEEDS)}"
        )
    campaign = Path(args.campaign_dir).resolve()
    selected_configs = Path(args.selected_configs).resolve()
    upstream_lineage = (
        Path(args.upstream_lineage).resolve() if args.upstream_lineage else None
    )
    runtime_manifest = (
        Path(args.runtime_manifest).resolve() if args.runtime_manifest else None
    )
    data_path = Path(args.data_path).resolve()
    if args.penalty_reduction not in {"mean", "sum"}:
        raise KSensitivityError("--penalty-reduction must be mean or sum")
    _read_selected_configs(selected_configs)
    selected_sha256 = _sha256(selected_configs)
    if upstream_lineage is None:
        raise KSensitivityError(
            "--upstream-lineage is required so the frozen configurations are "
            "cryptographically tied to their corrected selection campaign"
        )
    _validate_selection_lineage(upstream_lineage, selected_sha256)
    if (
        selected_sha256 == KNOWN_NONEDGEWISE_CONFIG_SHA256
        and not args.allow_nonedgewise_cafe_in
    ):
        raise KSensitivityError(
            "The supplied selected configurations were produced under the "
            "known non-edgewise Eq. (8) implementation. Rerun configuration "
            "selection after fixing Eq. (8), or explicitly pass "
            "--allow-nonedgewise-cafe-in for a labelled invalid legacy run."
        )
    if not data_path.is_dir():
        raise KSensitivityError(f"Data directory not found: {data_path}")

    edgewise = _edgewise_penalty_preflight()
    if not edgewise and not args.allow_nonedgewise_cafe_in:
        raise KSensitivityError(
            "Eq. (8) edgewise preflight FAILED: the current CAFE-In loss "
            "broadcasts [B,1] predictions against [B] weights. No model was "
            "trained. Fix the loss and rerun CAFE-In configuration selection, "
            "or explicitly pass --allow-nonedgewise-cafe-in to obtain a "
            "legacy-implementation artifact that must not be presented as "
            "evidence for Eq. (8)."
        )

    protocol_document = _protocol_document(
        selected_configs=selected_configs,
        upstream_lineage=upstream_lineage,
        runtime_manifest=runtime_manifest,
        edgewise_penalty=edgewise,
        allow_nonedgewise=args.allow_nonedgewise_cafe_in,
        penalty_reduction=args.penalty_reduction,
        configuration_origin=args.configuration_origin,
        exact_closeness=args.exact_closeness,
    )
    protocol_path = _create_or_verify_protocol(campaign, protocol_document)
    protocol_sha256 = _sha256(protocol_path)
    selected_sha256 = _sha256(selected_configs)
    task_dir = _task_dir(campaign, dataset, seed)
    task_dir.mkdir(parents=True, exist_ok=True)
    success_path = task_dir / "_SUCCESS"
    out_csv = task_dir / "per_seed_metrics.csv"
    runner_marker = Path(f"{out_csv}.complete.json")

    if success_path.exists():
        _validate_shard(
            csv_path=out_csv,
            dataset=dataset,
            seed=seed,
            selected_configs_sha256=selected_sha256,
            protocol_sha256=protocol_sha256,
            penalty_reduction=args.penalty_reduction,
        )
        print(f"Shard already complete: {dataset}, seed {seed}")
        return 0
    if out_csv.exists():
        raise KSensitivityError(
            f"Incomplete task directory already contains {out_csv}. Move the "
            "task directory aside before retrying; it is not safe to trust the "
            "underlying runner's K=10-only resume check for this campaign."
        )

    command = _runner_command(
        dataset=dataset,
        seed=seed,
        data_path=data_path,
        selected_configs=selected_configs,
        protocol_path=protocol_path,
        out_csv=out_csv,
        penalty_reduction=args.penalty_reduction,
        exact_closeness=args.exact_closeness,
        record_dir=task_dir,
    )
    _atomic_write_text(task_dir / "command.sh", shlex.join(command) + "\n")
    _atomic_write_json(task_dir / "command.json", {"argv": command})
    if args.dry_run:
        print(shlex.join(command))
        return 0

    return_code, metric_failures = _stream_subprocess(
        command, task_dir / "runner.log"
    )
    if return_code != 0:
        raise KSensitivityError(
            f"Underlying runner exited with status {return_code}; see "
            f"{task_dir / 'runner.log'}"
        )
    if metric_failures:
        raise KSensitivityError(
            "A centrality computation was converted to a zero by the underlying "
            "runner. This campaign treats that as a failure: "
            + " | ".join(metric_failures)
        )
    if not runner_marker.is_file():
        raise KSensitivityError(
            f"Underlying completion marker missing: {runner_marker}"
        )
    if args.exact_closeness and dataset in EXACT_CLOSENESS_DATASETS:
        exact_marker = task_dir / "exact_process_complete.json"
        if not exact_marker.is_file():
            raise KSensitivityError(
                f"Exact closeness was requested but {exact_marker} is missing"
            )

    validation = _validate_shard(
        csv_path=out_csv,
        dataset=dataset,
        seed=seed,
        selected_configs_sha256=selected_sha256,
        protocol_sha256=protocol_sha256,
        penalty_reduction=args.penalty_reduction,
    )
    supporting = {}
    for name in (
        "command.sh",
        "command.json",
        "runner.log",
        "exact_process_complete.json",
        "exact_metric_calls.jsonl",
        "task.env",
        "runtime.conda-explicit.txt",
        "wrapper_command.sh",
    ):
        path = task_dir / name
        if path.is_file():
            supporting[name] = _sha256(path)
    task_manifest = {
        "schema_version": SCHEMA_VERSION,
        "profile": PROFILE,
        "status": "complete",
        **validation,
        "evaluated_k_values": list(EVALUATED_K),
        "reported_k_values": list(REPORTED_K),
        "protocol_sha256": protocol_sha256,
        "selected_configs_sha256": selected_sha256,
        "runner_completion_sha256": _sha256(runner_marker),
        "eq8_edgewise_penalty_preflight": edgewise,
        "allow_nonedgewise_cafe_in": args.allow_nonedgewise_cafe_in,
        "penalty_reduction": args.penalty_reduction,
        "configuration_origin": args.configuration_origin,
        "supporting_artifacts_sha256": supporting,
    }
    _atomic_write_json(task_dir / "task_manifest.json", task_manifest)
    _atomic_write_text(success_path, "complete\n")
    print(f"K-sensitivity shard complete: {dataset}, seed {seed}")
    return 0


def _load_protocol(campaign: Path) -> tuple[dict, str]:
    path = campaign / "k_sensitivity_protocol.json"
    if not path.is_file():
        raise KSensitivityError(f"Campaign protocol not found: {path}")
    try:
        protocol = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise KSensitivityError(f"Cannot read protocol {path}: {exc}") from exc
    if protocol.get("profile") != PROFILE:
        raise KSensitivityError(f"Unexpected campaign profile in {path}")
    return protocol, _sha256(path)


def _collect_shards(
    campaign: Path,
    protocol: Mapping[str, object],
    protocol_sha256: str,
    *,
    if_complete: bool,
) -> tuple[pd.DataFrame | None, Dict[str, str]]:
    selected_sha = str(protocol["selected_configs"]["sha256"])
    penalty_reduction = str(protocol.get("penalty_reduction", ""))
    if penalty_reduction not in {"mean", "sum"}:
        raise KSensitivityError("Campaign protocol has invalid penalty reduction")
    frames = []
    manifests: Dict[str, str] = {}
    missing = []
    for dataset in DATASETS:
        for seed in SEEDS:
            task_dir = _task_dir(campaign, dataset, seed)
            success = task_dir / "_SUCCESS"
            manifest_path = task_dir / "task_manifest.json"
            csv_path = task_dir / "per_seed_metrics.csv"
            if (
                not success.is_file()
                or not manifest_path.is_file()
                or not csv_path.is_file()
            ):
                missing.append(f"{dataset}/seed_{seed}")
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise KSensitivityError(
                    f"Cannot read task manifest {manifest_path}: {exc}"
                ) from exc
            if manifest.get("protocol_sha256") != protocol_sha256:
                raise KSensitivityError(
                    f"{manifest_path}: protocol hash mismatch"
                )
            if manifest.get("csv_sha256") != _sha256(csv_path):
                raise KSensitivityError(f"{csv_path}: hash differs from task manifest")
            if bool(manifest.get("eq8_edgewise_penalty_preflight")) != bool(
                protocol.get("eq8_edgewise_penalty_preflight")
            ):
                raise KSensitivityError(
                    f"{manifest_path}: Eq. (8) preflight status mismatch"
                )
            _validate_shard(
                csv_path=csv_path,
                dataset=dataset,
                seed=seed,
                selected_configs_sha256=selected_sha,
                protocol_sha256=protocol_sha256,
                penalty_reduction=penalty_reduction,
            )
            frame = pd.read_csv(csv_path)
            frame["dataset"] = dataset
            frames.append(frame)
            manifests[f"{dataset}/seed_{seed}"] = _sha256(manifest_path)

    if missing:
        message = (
            f"K-sensitivity campaign is missing {len(missing)} shard(s): "
            + ", ".join(missing)
        )
        if if_complete:
            print(message)
            return None, manifests
        raise KSensitivityError(message)
    return pd.concat(frames, ignore_index=True), manifests


def _wide_per_seed(frame: pd.DataFrame, k_values: Iterable[int]) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for method in METHODS:
                subset = frame.loc[
                    (frame["dataset"] == dataset)
                    & (frame["seed"].astype(int) == seed)
                    & (frame["method"] == method)
                ]
                values = dict(zip(subset["metric"], subset["value"]))
                for k in k_values:
                    row = {
                        "dataset": dataset,
                        "seed": seed,
                        "method": method,
                        "k": k,
                    }
                    for output_name, prefix in METRIC_PREFIXES.items():
                        key = f"{prefix}@{k}"
                        if key not in values:
                            raise KSensitivityError(
                                f"Missing {key} for {dataset}, seed {seed}, {method}"
                            )
                        row[output_name] = float(values[key])
                    rows.append(row)
    return pd.DataFrame(rows)


def _summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_columns = list(METRIC_PREFIXES)
    grouped = per_seed.groupby(["dataset", "method", "k"], sort=False)
    for (dataset, method, k), group in grouped:
        if set(group["seed"].astype(int)) != set(SEEDS):
            raise KSensitivityError(
                f"Incomplete seed set for {dataset}, {method}, K={k}"
            )
        row = {
            "dataset": dataset,
            "method": method,
            "k": int(k),
            "run_count": int(len(group)),
        }
        for metric in metric_columns:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        rows.append(row)
    result = pd.DataFrame(rows)
    dataset_rank = {value: index for index, value in enumerate(DATASETS)}
    method_rank = {value: index for index, value in enumerate(METHODS)}
    result["_dataset_rank"] = result["dataset"].map(dataset_rank)
    result["_method_rank"] = result["method"].map(method_rank)
    result = result.sort_values(["k", "_dataset_rank", "_method_rank"])
    return result.drop(columns=["_dataset_rank", "_method_rank"]).reset_index(drop=True)


def _paired_contrasts(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for k in REPORTED_K:
            block = per_seed.loc[
                (per_seed["dataset"] == dataset) & (per_seed["k"] == k)
            ].set_index(["method", "seed"])
            for variant in ("CAFE-In", "CAFE-Pre"):
                for reference in ("HL-GNN", "Fusion"):
                    row = {
                        "dataset": dataset,
                        "k": k,
                        "variant": variant,
                        "reference": reference,
                        "run_count": len(SEEDS),
                    }
                    for metric in METRIC_PREFIXES:
                        left = block.loc[variant, metric].sort_index()
                        right = block.loc[reference, metric].sort_index()
                        if not left.index.equals(right.index):
                            raise KSensitivityError(
                                f"Unpaired seeds for {dataset}, K={k}, "
                                f"{variant} vs {reference}"
                            )
                        difference = left - right
                        row[f"{metric}_difference_mean"] = float(difference.mean())
                        row[f"{metric}_difference_std"] = float(
                            difference.std(ddof=1)
                        )
                    rows.append(row)
    return pd.DataFrame(rows)


def _latex_cell(mean: float, std: float, digits: int, bold: bool) -> str:
    rendered = f"${mean:.{digits}f} \\pm {std:.{digits}f}$"
    return f"\\textbf{{{rendered}}}" if bold else rendered


def _latex_table(summary: pd.DataFrame, protocol: Mapping[str, object]) -> str:
    lines = [
        "% Auto-generated by experiments/run_k_sensitivity.py.",
        "% Configurations were selected at K=10 and frozen; no per-K reselection.",
    ]
    edgewise = bool(protocol.get("eq8_edgewise_penalty_preflight"))
    if not edgewise:
        lines += [
            "% WARNING: Eq. (8) edgewise preflight FAILED.",
            "% This is a legacy-implementation artifact and must not be cited as Eq. (8).",
        ]

    headers = {
        "precision": "Precision",
        "degree_eg": "Degree EG",
        "pagerank_eg": "PageRank EG",
        "closeness_eg": "Closeness EG",
    }
    for k in REPORTED_K:
        block = summary.loc[summary["k"] == k]
        warning = (
            " The CAFE-In loss failed the edgewise Eq.~(8) preflight; these "
            "rows describe the legacy implementation."
            if not edgewise
            else ""
        )
        lines += [
            "\\begin{table*}[t]",
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{4pt}",
            "\\caption{Fixed-configuration sensitivity at "
            f"$K={k}$. Values are mean $\\pm$ sample standard deviation over "
            "the frozen training seeds; configurations selected at $K=10$ are "
            "held fixed, and the same top-20 ranking supplies each cutoff prefix."
            f"{warning}}}",
            f"\\label{{tab:k-sensitivity-k{k}}}",
            "\\begin{tabular}{llrrrr}",
            "\\toprule",
            f"Dataset & Method & {headers['precision']}@{k} & "
            f"{headers['degree_eg']}@{k} & {headers['pagerank_eg']}@{k} & "
            f"{headers['closeness_eg']}@{k} \\\\",
            "\\midrule",
        ]
        for dataset_index, dataset in enumerate(DATASETS):
            dataset_block = block.loc[block["dataset"] == dataset]
            maxima = {
                metric: float(dataset_block[f"{metric}_mean"].max())
                for metric in METRIC_PREFIXES
            }
            for method_index, method in enumerate(METHODS):
                row = dataset_block.loc[dataset_block["method"] == method]
                if len(row) != 1:
                    raise KSensitivityError(
                        f"LaTeX source lacks one row for {dataset}, {method}, K={k}"
                    )
                values = row.iloc[0]
                dataset_label = DISPLAY_DATASET[dataset] if method_index == 0 else ""
                cells = []
                for metric in METRIC_PREFIXES:
                    mean = float(values[f"{metric}_mean"])
                    std = float(values[f"{metric}_std"])
                    digits = 4 if metric == "precision" else 2
                    cells.append(
                        _latex_cell(
                            mean,
                            std,
                            digits,
                            math.isclose(mean, maxima[metric], rel_tol=0.0, abs_tol=1e-12),
                        )
                    )
                lines.append(
                    f"{dataset_label} & {method} & " + " & ".join(cells) + " \\\\"
                )
            if dataset_index != len(DATASETS) - 1:
                lines.append("\\addlinespace")
        lines += [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
            "",
        ]
    return "\n".join(lines)


def finalize(args: argparse.Namespace) -> int:
    campaign = Path(args.campaign_dir).resolve()
    campaign.mkdir(parents=True, exist_ok=True)
    lock_path = campaign / ".finalize.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        protocol, protocol_sha256 = _load_protocol(campaign)
        frame, task_manifests = _collect_shards(
            campaign,
            protocol,
            protocol_sha256,
            if_complete=args.if_complete,
        )
        if frame is None:
            return 0

        output_dir = (
            Path(args.output_dir).resolve()
            if args.output_dir
            else campaign / "finalized"
        )
        marker = output_dir / "_COMPLETE"
        if marker.exists() and not args.rebuild:
            print(f"Finalized K-sensitivity artifacts already exist: {output_dir}")
            return 0

        reported_per_seed = _wide_per_seed(frame, REPORTED_K)
        internal_reference = _wide_per_seed(frame, (10,))
        summary = _summary(reported_per_seed)
        reference_summary = _summary(internal_reference)
        contrasts = _paired_contrasts(reported_per_seed)
        latex = _latex_table(summary, protocol)

        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "k_sensitivity_per_seed.csv": reported_per_seed,
            "k_sensitivity_summary.csv": summary,
            "k_sensitivity_paired_contrasts.csv": contrasts,
            "k10_internal_reference.csv": reference_summary,
        }
        for name, output_frame in outputs.items():
            _atomic_write_csv(output_dir / name, output_frame)
        _atomic_write_text(output_dir / "k_sensitivity_table.tex", latex)

        output_hashes = {
            name: _sha256(output_dir / name)
            for name in [*outputs, "k_sensitivity_table.tex"]
        }
        integrity = {
            "schema_version": SCHEMA_VERSION,
            "profile": PROFILE,
            "status": "complete",
            "protocol_sha256": protocol_sha256,
            "datasets": list(DATASETS),
            "training_seeds": list(SEEDS),
            "reported_k_values": list(REPORTED_K),
            "internal_reference_k": 10,
            "methods": list(METHODS),
            "eq8_edgewise_penalty_preflight": bool(
                protocol.get("eq8_edgewise_penalty_preflight")
            ),
            "allow_nonedgewise_cafe_in": bool(
                protocol.get("allow_nonedgewise_cafe_in")
            ),
            "penalty_reduction": protocol.get("penalty_reduction"),
            "configuration_origin": protocol.get("configuration_origin"),
            "task_manifest_sha256": task_manifests,
            "outputs_sha256": output_hashes,
        }
        _atomic_write_json(output_dir / "k_sensitivity_integrity.json", integrity)
        _atomic_write_text(marker, "complete\n")
        print(f"Finalized K-sensitivity table: {output_dir}")
        return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    run = subparsers.add_parser("run", help="run one dataset/seed shard")
    run.add_argument("--dataset", required=True)
    run.add_argument("--seed", required=True, type=int)
    run.add_argument("--campaign-dir", required=True)
    run.add_argument("--selected-configs", required=True)
    run.add_argument("--data-path", default=str(REPO / "data" / "SNAP"))
    run.add_argument(
        "--upstream-lineage",
        default=None,
        help="optional upstream integrity artifact; recorded as lineage only",
    )
    run.add_argument(
        "--runtime-manifest",
        default=None,
        help="optional conda explicit manifest whose hash is frozen in protocol",
    )
    run.add_argument(
        "--allow-nonedgewise-cafe-in",
        action="store_true",
        help="DANGEROUS: permit the known broadcasted CAFE-In implementation",
    )
    run.add_argument(
        "--penalty-reduction",
        choices=("mean", "sum"),
        default=DEFAULT_PENALTY_REDUCTION,
        help="loss-reduction semantics frozen for this campaign",
    )
    run.add_argument(
        "--configuration-origin",
        choices=("validation_selected", "legacy_test_selected"),
        default="validation_selected",
        help="honest provenance label for the frozen operating points",
    )
    run.add_argument(
        "--exact-closeness",
        action="store_true",
        help="evaluate Closeness EG exactly on "
        + ", ".join(EXACT_CLOSENESS_DATASETS)
        + " (metrics/closeness_exact_v1.py), as in Tables 4, 7, 11 and 15",
    )
    run.add_argument("--dry-run", action="store_true")

    finish = subparsers.add_parser(
        "finalize", help="strictly aggregate all dataset/seed shards"
    )
    finish.add_argument("--campaign-dir", required=True)
    finish.add_argument("--output-dir", default=None)
    finish.add_argument(
        "--if-complete",
        action="store_true",
        help="return success without output while shards are still missing",
    )
    finish.add_argument(
        "--rebuild",
        action="store_true",
        help="atomically regenerate final outputs from the same verified shards",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.stage == "run":
            return run_shard(args)
        return finalize(args)
    except KSensitivityError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
