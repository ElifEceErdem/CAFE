"""Frozen protocol for the CAFE-In ``penalty_reduction=sum`` selection.

This campaign is intentionally separate from ``jobs/egfix``.  The latter
records the historical mean-reduced, two-variant selection; this profile
reselects only CAFE-In after correcting the penalty scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Dict, Tuple

from experiments.egfix_protocol import (
    ASF_WEIGHTS,
    BETA_VALUES,
    BRIGHTKITE_TEMPORAL_SPLIT,
    BRIGHTKITE_TEMPORAL_SPLIT_SHA256,
    CLOSENESS_EPSILON,
    CLOSENESS_RANDOM_SEED,
    CLOSENESS_SAMPLE_SIZE,
    DATASETS,
    EG_GRAPH_SEMANTICS,
    INPROC_PENALTIES,
    K_VALUES,
    MOBILITY_VALUES,
    OBJECTIVES,
    SEED_BASE,
    SEED_RULE,
    SELECTION_RUNS,
    SPLIT_SEED,
    USE_DEMOGRAPHICS,
    expected_penalty_config_keys as _base_expected_penalty_config_keys,
    make_config_key as _base_make_config_key,
    parse_config_key as _base_parse_config_key,
)


SCHEMA_VERSION = 1
PROFILE = "cafe_in_sum_v1"
VARIANT = "cafe_in"
PENALIZATION_MODE = "inproc"
PENALTY_NAME = "lambda_fair"
PENALTY_REDUCTION = "sum"
EVALUATION_SPLIT = "valid"
LOSS_FUNC = "AUC"
NUM_NEG = 1
BATCH_SIZE = 16384
EPOCHS = 500
LEARNING_RATE = 0.001

LAMBDA_FAIR_VALUES: Tuple[float, ...] = tuple(INPROC_PENALTIES)
CONFIGS_PER_PENALTY = len(BETA_VALUES) * len(MOBILITY_VALUES)
CONFIGS_PER_DATASET = (
    len(LAMBDA_FAIR_VALUES) * CONFIGS_PER_PENALTY
)
SELECTION_TASK_COUNT = len(DATASETS) * len(LAMBDA_FAIR_VALUES)
EXPECTED_CANDIDATE_COUNT = len(DATASETS) * CONFIGS_PER_DATASET
EXPECTED_RUN_ROW_COUNT = EXPECTED_CANDIDATE_COUNT * SELECTION_RUNS
EXPECTED_SELECTION_COUNT = len(DATASETS)


class ProtocolError(ValueError):
    """Raised when a value falls outside the frozen sum-selection protocol."""


def make_config_key(
    penalty: float, beta_ra: float, lambda_mobility: float
) -> str:
    return _base_make_config_key(
        VARIANT, penalty, beta_ra, lambda_mobility
    )


def parse_config_key(config_key: str) -> Dict[str, object]:
    try:
        parsed = _base_parse_config_key(config_key)
    except ValueError as exc:
        raise ProtocolError(str(exc)) from exc
    if parsed["variant"] != VARIANT:
        raise ProtocolError(
            f"{config_key!r} is not a {VARIANT} configuration"
        )
    return parsed


def expected_penalty_config_keys(penalty: float) -> Tuple[str, ...]:
    try:
        return _base_expected_penalty_config_keys(VARIANT, penalty)
    except ValueError as exc:
        raise ProtocolError(str(exc)) from exc


def expected_config_keys() -> Tuple[str, ...]:
    return tuple(
        key
        for penalty in LAMBDA_FAIR_VALUES
        for key in expected_penalty_config_keys(penalty)
    )


def selection_tasks() -> Tuple[Dict[str, object], ...]:
    tasks = []
    for dataset in DATASETS:
        for penalty in LAMBDA_FAIR_VALUES:
            tasks.append(
                {
                    "task_index": len(tasks),
                    "dataset": dataset,
                    "variant": VARIANT,
                    "penalization_mode": PENALIZATION_MODE,
                    "penalty_name": PENALTY_NAME,
                    "penalty_value": float(penalty),
                    "configs": CONFIGS_PER_PENALTY,
                }
            )
    return tuple(tasks)


def selection_task(index: int) -> Dict[str, object]:
    tasks = selection_tasks()
    if index < 0 or index >= len(tasks):
        raise ProtocolError(
            f"Task index {index} outside [0, {len(tasks) - 1}]"
        )
    return dict(tasks[index])


def protocol_document() -> Dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": PROFILE,
        "scope": "TODO1 validation-only configuration selection",
        "datasets": list(DATASETS),
        "variants": [VARIANT],
        "penalization_mode": PENALIZATION_MODE,
        "penalty_reduction": PENALTY_REDUCTION,
        "loss_func": LOSS_FUNC,
        "num_neg": NUM_NEG,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "grid": {
            PENALTY_NAME: list(LAMBDA_FAIR_VALUES),
            "beta_ra": list(BETA_VALUES),
            "lambda_mobility": list(MOBILITY_VALUES),
            "config_count_per_penalty": CONFIGS_PER_PENALTY,
            "config_count_per_dataset": CONFIGS_PER_DATASET,
        },
        "selection_runs": SELECTION_RUNS,
        "seed_base": SEED_BASE,
        "seed_rule": SEED_RULE,
        "split_seed": SPLIT_SEED,
        "evaluation_split": EVALUATION_SPLIT,
        "test_metrics_evaluated": False,
        "k_values": list(K_VALUES),
        "use_demographics": USE_DEMOGRAPHICS,
        "eg_graph_semantics": EG_GRAPH_SEMANTICS,
        "closeness_random_seed": CLOSENESS_RANDOM_SEED,
        "closeness_sample_size": CLOSENESS_SAMPLE_SIZE,
        "closeness_epsilon": CLOSENESS_EPSILON,
        "objectives": list(OBJECTIVES),
        "asf_weights": list(ASF_WEIGHTS),
        "asf_tie_break": "lexicographic config_key",
        "brightkite_temporal_split": BRIGHTKITE_TEMPORAL_SPLIT,
        "brightkite_temporal_split_sha256": (
            BRIGHTKITE_TEMPORAL_SPLIT_SHA256
        ),
        "selection_task_count": SELECTION_TASK_COUNT,
        "expected_candidate_count": EXPECTED_CANDIDATE_COUNT,
        "expected_run_row_count": EXPECTED_RUN_ROW_COUNT,
        "expected_selection_count": EXPECTED_SELECTION_COUNT,
        "large_dataset_training_eval": {
            "datasets": ["pokec_n", "brightkite"],
            "eval_skip_test": True,
            "eval_max_nodes": 5000,
            "eval_steps": 25,
            "patience": 6,
        },
    }


def protocol_sha256() -> str:
    payload = json.dumps(
        protocol_document(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the isolated CAFE-In/sum protocol or one task"
    )
    parser.add_argument("--task-index", type=int)
    parser.add_argument(
        "--format", choices=("json", "tsv"), default="json"
    )
    args = parser.parse_args()

    if args.task_index is None:
        print(json.dumps(protocol_document(), sort_keys=True, indent=2))
        return

    task = selection_task(args.task_index)
    if args.format == "json":
        print(json.dumps(task, sort_keys=True))
    else:
        print(
            "\t".join(
                str(task[field])
                for field in (
                    "dataset",
                    "variant",
                    "penalization_mode",
                    "penalty_name",
                    "penalty_value",
                )
            )
        )


if __name__ == "__main__":
    _main()
