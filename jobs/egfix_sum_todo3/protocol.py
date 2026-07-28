"""Frozen protocol for the corrected RQ3 aggregate/stratified reruns.

This protocol is deliberately additive.  It does not alter either upstream
selection campaign:

* CAFE-In comes from the reviewed ``penalty_reduction=sum`` selection.
* CAFE-Pre comes from the earlier corrected-EG selection.  The execution-wide
  reduction is recorded as ``sum`` because both existing runners require one
  value for the two selected rows; that reduction is not consulted by the
  preprocessing training path.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Tuple

from experiments.egfix_protocol import (
    BRIGHTKITE_TEMPORAL_SPLIT,
    BRIGHTKITE_TEMPORAL_SPLIT_SHA256,
    CLOSENESS_EPSILON,
    CLOSENESS_RANDOM_SEED,
    CLOSENESS_SAMPLE_SIZE,
    DATASETS,
    EG_GRAPH_SEMANTICS,
    OBJECTIVES,
    SEED_BASE,
    SPLIT_SEED,
)


SCHEMA_VERSION = 1
PROFILE = "rq3_degree_eg_sum_todo3_v1"
SCOPE = "RQ3 aggregate and five-dataset source/target-stratified reruns"

VARIANTS: Tuple[str, ...] = ("cafe_in", "cafe_pre")
METHODS: Tuple[str, ...] = (
    "RA",
    "HL-GNN",
    "Fusion",
    "CAFE-In",
    "CAFE-Pre",
)
PENALTY_REDUCTION = "sum"
LOSS_FUNC = "AUC"
NUM_NEG = 1
BATCH_SIZE = 16384
EPOCHS = 500
LEARNING_RATE = 0.001
K = 10
NUM_SOURCE_BINS = 10
EVALUATION_SPLIT = "test"
SELECTION_SPLIT = "valid"

AGGREGATE_SEEDS: Tuple[int, ...] = tuple(range(SEED_BASE, SEED_BASE + 10))
STRATIFIED_SEEDS: Tuple[int, ...] = tuple(
    range(SEED_BASE, SEED_BASE + 3)
)
AGGREGATE_TASK_COUNT = len(DATASETS) * len(AGGREGATE_SEEDS)
STRATIFIED_TASK_COUNT = len(DATASETS) * len(STRATIFIED_SEEDS)

TARGET_TIER_NAMES: Tuple[str, ...] = (
    "periphery",
    "middle",
    "core",
)
TARGET_TIER_PERCENTILES: Tuple[float, ...] = (100.0 / 3.0, 200.0 / 3.0)
TARGET_TIER_BOUNDARY = "right_closed"
TARGET_EXPOSURE_SEMANTICS = "directed_recommendation_events"


def protocol_document() -> Dict[str, object]:
    """Return the complete, JSON-serializable TODO3 protocol."""

    return {
        "schema_version": SCHEMA_VERSION,
        "profile": PROFILE,
        "scope": SCOPE,
        "datasets": list(DATASETS),
        "variants": list(VARIANTS),
        "methods": list(METHODS),
        "objectives": list(OBJECTIVES),
        "k": K,
        "source_bins_requested": NUM_SOURCE_BINS,
        "aggregate_seeds": list(AGGREGATE_SEEDS),
        "stratified_seeds": list(STRATIFIED_SEEDS),
        "aggregate_task_count": AGGREGATE_TASK_COUNT,
        "stratified_task_count": STRATIFIED_TASK_COUNT,
        "split_seed": SPLIT_SEED,
        "evaluation_split": EVALUATION_SPLIT,
        "selection_split": SELECTION_SPLIT,
        "eg_graph_semantics": EG_GRAPH_SEMANTICS,
        "penalty_reduction": PENALTY_REDUCTION,
        "loss_func": LOSS_FUNC,
        "num_neg": NUM_NEG,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "closeness_random_seed": CLOSENESS_RANDOM_SEED,
        "closeness_sample_size": CLOSENESS_SAMPLE_SIZE,
        "closeness_epsilon": CLOSENESS_EPSILON,
        "brightkite_temporal_split": BRIGHTKITE_TEMPORAL_SPLIT,
        "brightkite_temporal_split_sha256": (
            BRIGHTKITE_TEMPORAL_SPLIT_SHA256
        ),
        "training_time_test_evaluation": False,
        "large_dataset_training_validation": {
            "datasets": ["pokec_n", "brightkite"],
            "eval_max_nodes": 5000,
            "eval_steps": 25,
            "patience": 6,
        },
        "configuration_origins": {
            "cafe_in": {
                "selection_profile": "cafe_in_sum_v1",
                "selection_penalty_reduction": "sum",
            },
            "cafe_pre": {
                "selection_profile": "corrected_eg_mean_v1",
                "selection_penalty_reduction": "mean",
                "execution_reduction_note": (
                    "penalty_reduction is not used by preprocessing mode"
                ),
            },
        },
        "fusion_control": {
            "matched_to": "CAFE-In beta",
            "no_penalty": True,
            "no_mobility_boost": True,
        },
        "target_exposure": {
            "semantics": TARGET_EXPOSURE_SEMANTICS,
            "population": "all graph nodes, including zero-exposure nodes",
            "normalization": "incoming target events divided by eval-user count",
            "percentiles": list(TARGET_TIER_PERCENTILES),
            "percentile_method": "linear",
            "boundary": TARGET_TIER_BOUNDARY,
            "tiers": {
                "periphery": "coreness <= q_1_3",
                "middle": "q_1_3 < coreness <= q_2_3",
                "core": "coreness > q_2_3",
            },
            "reciprocal_events_deduplicated": False,
            "uncertainty_unit": "training seed",
            "contrast": "CAFE-In minus matched-beta Fusion",
        },
        "manuscript_writes": False,
    }


def protocol_sha256() -> str:
    """Return a stable digest of :func:`protocol_document`."""

    payload = json.dumps(
        protocol_document(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

