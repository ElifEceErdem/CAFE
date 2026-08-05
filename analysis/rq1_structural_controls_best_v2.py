#!/usr/bin/env python3
"""Best-config RQ1 structural controls with size-scaled logit diagnostics.

This is an intentionally thin wrapper around ``rq1_structural_controls``.  It
does not change the fitted objective, optimizer, coefficients, covariance
matrix, confidence intervals, tables, or aggregation.  It changes only the
post-fit convergence decision: the analytic score is divided by the number of
observations before applying a fixed tolerance, and optimizer success is also
required.  The v1 summed-score decision is retained in every diagnostics row
for auditability.

The wrapper has a distinct profile and code manifest, so v1 and v2 shards
cannot be combined by the strict finalizer.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis import rq1_structural_controls as base  # noqa: E402


PROFILE = "rq1_conditional_hit_and_clustering_axis_best_configs_v2"
SCHEMA_VERSION = 2
SCORE_TOLERANCE_PER_OBSERVATION = 1e-7
LEGACY_BEST_CONFIGS_SELECTED_SHA256 = (
    "a6162e8bd89893b38bf41b5782a7996a34ea6364be8731b23690fb69f945ebef"
)
CODE_FILES = (
    "analysis/rq1_structural_controls_best_v2.py",
    "analysis/rq1_structural_controls.py",
    "metrics/centrality_equality_gain.py",
    "metrics/equality_gain.py",
    "metrics/graph_utils.py",
    "metrics/coreness.py",
    "core/data_loader.py",
    "backbone/snap_dataset.py",
    "fairness_methods/inprocessing.py",
)


_fit_logit_v1 = base.fit_logit


def fit_logit_best_v2(
    frame: pd.DataFrame,
    predictors: Sequence[str],
    *,
    cluster_column: str = "node_id",
) -> tuple[pd.DataFrame, dict]:
    """Return the unchanged v1 fit with an observation-scaled score check."""

    coefficients, diagnostics = _fit_logit_v1(
        frame, predictors, cluster_column=cluster_column
    )
    n_rows = int(diagnostics["n_rows"])
    if n_rows <= 0:
        raise base.StructuralControlError(
            f"Invalid logistic observation count: {n_rows}"
        )

    summed_score = float(diagnostics["max_absolute_gradient"])
    score_per_observation = summed_score / n_rows
    optimizer_success = bool(diagnostics["optimizer_success"])
    v1_converged = bool(diagnostics["converged"])
    scaled_score_passed = bool(
        score_per_observation <= SCORE_TOLERANCE_PER_OBSERVATION
    )

    diagnostics.update({
        "converged_v1_summed_score_rule": v1_converged,
        "v1_summed_score_tolerance": 1e-4,
        "max_absolute_score_sum": summed_score,
        "max_absolute_score_per_observation": score_per_observation,
        "score_tolerance_per_observation": SCORE_TOLERANCE_PER_OBSERVATION,
        "scaled_score_passed": scaled_score_passed,
        "convergence_rule": (
            "optimizer_success AND "
            "max_absolute_score_per_observation <= 1e-7"
        ),
        "coefficient_and_covariance_implementation": (
            "unchanged from analysis/rq1_structural_controls.py"
        ),
        "converged": bool(optimizer_success and scaled_score_passed),
    })
    return coefficients, diagnostics


def configure_base() -> None:
    """Install the isolated v2 profile and diagnostic rule in the base module."""

    base.PROFILE = PROFILE
    base.SCHEMA_VERSION = SCHEMA_VERSION
    base.CODE_FILES = CODE_FILES
    base.KNOWN_NONEDGEWISE_SELECTED_CONFIG_SHA256 = (
        LEGACY_BEST_CONFIGS_SELECTED_SHA256
    )
    base.fit_logit = fit_logit_best_v2


def main() -> int:
    configure_base()
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
