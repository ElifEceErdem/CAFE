"""Immutable protocol constants for the corrected-EG rerun campaign.

This module deliberately has no dependencies outside the Python standard
library.  The Slurm launchers, preflight checker, and result finalizers import
the same constants so that a grid or provenance change cannot be made in only
one stage of the campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from itertools import product
from typing import Dict, Iterator, Mapping, Sequence, Tuple


SCHEMA_VERSION = 1

DATASETS: Tuple[str, ...] = (
    "ego-facebook",
    "oklahoma",
    "unc",
    "pokec_n",
    "brightkite",
)

VARIANTS: Tuple[str, ...] = ("cafe_in", "cafe_pre")

INPROC_PENALTIES: Tuple[float, ...] = (
    0.0,
    0.01,
    0.05,
    0.1,
    0.3,
    0.5,
    1.0,
)
PREPROC_PENALTIES: Tuple[float, ...] = (
    0.01,
    0.05,
    0.1,
    0.3,
    0.5,
    1.0,
    2.0,
)
BETA_VALUES: Tuple[float, ...] = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0)
MOBILITY_VALUES: Tuple[float, ...] = (0.0, 0.1, 0.3, 0.5, 1.0)

VARIANT_SPECS: Mapping[str, Mapping[str, object]] = {
    "cafe_in": {
        "penalization_mode": "inproc",
        "penalty_name": "lambda_fair",
        "penalties": INPROC_PENALTIES,
    },
    "cafe_pre": {
        "penalization_mode": "preproc",
        "penalty_name": "preproc_alpha",
        "penalties": PREPROC_PENALTIES,
    },
}

SELECTION_RUNS = 3
SEED_BASE = 42
SEED_RULE = "model_seed = seed_base + run_index"
SPLIT_SEED = 42
K_VALUES: Tuple[int, ...] = (10,)
EVALUATION_SPLIT = "valid"
EG_GRAPH_SEMANTICS = "simple_undirected_set_union"
PENALTY_REDUCTION = "mean"
CLOSENESS_RANDOM_SEED = 42
CLOSENESS_SAMPLE_SIZE = 0
CLOSENESS_EPSILON = 0.1
USE_DEMOGRAPHICS = False

OBJECTIVES: Tuple[str, ...] = (
    "Precision@10",
    "Equality_Gain@10",
    "Pagerank_EG@10",
    "Closeness_EG@10",
)
ASF_WEIGHTS: Tuple[float, ...] = (0.5, 0.3, 0.1, 0.1)

BRIGHTKITE_TEMPORAL_SPLIT = "data/SNAP/brightkite/temporal_split.pt"
BRIGHTKITE_TEMPORAL_SPLIT_SHA256 = (
    "9c85751eb089129bf3f1d889e797d515a653a2213317332b5be4f1731d820118"
)

CONFIGS_PER_PENALTY = len(BETA_VALUES) * len(MOBILITY_VALUES)
CONFIGS_PER_VARIANT = 7 * CONFIGS_PER_PENALTY
SELECTION_TASK_COUNT = len(DATASETS) * len(VARIANTS) * 7

_NUMBER = r"(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?"
_CONFIG_RE = re.compile(
    rf"^(?P<penalty_name>lambda_fair|preproc_alpha)_"
    rf"(?P<penalty>{_NUMBER})_beta_(?P<beta>{_NUMBER})_"
    rf"lam_(?P<mobility>{_NUMBER})$"
)


class ProtocolError(ValueError):
    """Raised when a campaign value violates the frozen protocol."""


def format_grid_value(value: float) -> str:
    """Return the exact spelling used by ``run_cafe.py`` configuration keys."""

    return str(float(value))


def variant_spec(variant: str) -> Mapping[str, object]:
    try:
        return VARIANT_SPECS[variant]
    except KeyError as exc:
        raise ProtocolError(
            f"Unknown variant {variant!r}; expected one of {VARIANTS}"
        ) from exc


def variant_from_mode(penalization_mode: str) -> str:
    for variant, spec in VARIANT_SPECS.items():
        if spec["penalization_mode"] == penalization_mode:
            return variant
    raise ProtocolError(
        f"Unknown penalization mode {penalization_mode!r}; "
        "expected 'inproc' or 'preproc'"
    )


def make_config_key(
    variant: str,
    penalty: float,
    beta_ra: float,
    lambda_mobility: float,
) -> str:
    spec = variant_spec(variant)
    return (
        f"{spec['penalty_name']}_{format_grid_value(penalty)}"
        f"_beta_{format_grid_value(beta_ra)}"
        f"_lam_{format_grid_value(lambda_mobility)}"
    )


def parse_config_key(config_key: str) -> Dict[str, object]:
    """Parse and validate one key from the definitive CAFE grid."""

    match = _CONFIG_RE.fullmatch(config_key)
    if match is None:
        raise ProtocolError(f"Malformed CAFE configuration key: {config_key!r}")

    penalty_name = match.group("penalty_name")
    variant = "cafe_in" if penalty_name == "lambda_fair" else "cafe_pre"
    penalty = float(match.group("penalty"))
    beta_ra = float(match.group("beta"))
    lambda_mobility = float(match.group("mobility"))
    spec = variant_spec(variant)

    if penalty not in spec["penalties"]:
        raise ProtocolError(
            f"{config_key!r} has {penalty_name}={penalty}, outside the "
            f"definitive grid {tuple(spec['penalties'])}"
        )
    if beta_ra not in BETA_VALUES:
        raise ProtocolError(
            f"{config_key!r} has beta_ra={beta_ra}, outside {BETA_VALUES}"
        )
    if lambda_mobility not in MOBILITY_VALUES:
        raise ProtocolError(
            f"{config_key!r} has lambda_mobility={lambda_mobility}, "
            f"outside {MOBILITY_VALUES}"
        )

    canonical = make_config_key(variant, penalty, beta_ra, lambda_mobility)
    if config_key != canonical:
        raise ProtocolError(
            f"Non-canonical configuration key {config_key!r}; "
            f"expected {canonical!r}"
        )

    return {
        "variant": variant,
        "penalization_mode": spec["penalization_mode"],
        "penalty_name": penalty_name,
        "penalty_value": penalty,
        "lambda_fair": penalty if variant == "cafe_in" else None,
        "preproc_alpha": penalty if variant == "cafe_pre" else None,
        "beta_ra": beta_ra,
        "lambda_mobility": lambda_mobility,
    }


def iter_variant_configs(variant: str) -> Iterator[Dict[str, object]]:
    """Yield all 210 configurations for one CAFE variant in fixed order."""

    spec = variant_spec(variant)
    for penalty, beta_ra, mobility in product(
        spec["penalties"], BETA_VALUES, MOBILITY_VALUES
    ):
        key = make_config_key(variant, penalty, beta_ra, mobility)
        parsed = parse_config_key(key)
        parsed["config_key"] = key
        yield parsed


def expected_config_keys(variant: str) -> Tuple[str, ...]:
    return tuple(row["config_key"] for row in iter_variant_configs(variant))


def expected_penalty_config_keys(
    variant: str, penalty: float
) -> Tuple[str, ...]:
    spec = variant_spec(variant)
    if float(penalty) not in spec["penalties"]:
        raise ProtocolError(
            f"Penalty {penalty} is outside the {variant} grid"
        )
    return tuple(
        make_config_key(variant, float(penalty), beta, mobility)
        for beta, mobility in product(BETA_VALUES, MOBILITY_VALUES)
    )


def selection_tasks() -> Tuple[Dict[str, object], ...]:
    """Return the immutable 70-task dataset × variant × penalty matrix."""

    tasks = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            spec = variant_spec(variant)
            for penalty in spec["penalties"]:
                tasks.append(
                    {
                        "task_index": len(tasks),
                        "dataset": dataset,
                        "variant": variant,
                        "penalization_mode": spec["penalization_mode"],
                        "penalty_name": spec["penalty_name"],
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
    """Return a JSON-serializable, deterministic statement of the protocol."""

    grids = {}
    for variant in VARIANTS:
        spec = variant_spec(variant)
        grids[variant] = {
            "penalization_mode": spec["penalization_mode"],
            "penalty_name": spec["penalty_name"],
            "penalties": list(spec["penalties"]),
            "beta_ra": list(BETA_VALUES),
            "lambda_mobility": list(MOBILITY_VALUES),
            "config_count": CONFIGS_PER_VARIANT,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "datasets": list(DATASETS),
        "variants": list(VARIANTS),
        "grids": grids,
        "selection_runs": SELECTION_RUNS,
        "seed_base": SEED_BASE,
        "seed_rule": SEED_RULE,
        "split_seed": SPLIT_SEED,
        "evaluation_split": EVALUATION_SPLIT,
        "test_metrics_evaluated": False,
        "k_values": list(K_VALUES),
        "use_demographics": USE_DEMOGRAPHICS,
        "eg_graph_semantics": EG_GRAPH_SEMANTICS,
        "penalty_reduction": PENALTY_REDUCTION,
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
    }


def protocol_sha256() -> str:
    payload = json.dumps(
        protocol_document(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the corrected-EG protocol or one selection task"
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
