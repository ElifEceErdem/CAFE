#!/usr/bin/env python3
"""Fail closed unless the frozen legacy configuration adapter is unchanged."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "analysis" / "best_configs.csv"
SELECTED = REPO / "experiments" / "configs" / "legacy_best_configs_selected.csv"
LINEAGE = REPO / "experiments" / "configs" / "legacy_best_configs_lineage.json"

SOURCE_SHA256 = "44ea8142fa6e953dc2c79fbe3d5b6ae5e4eb77ad08a3a72636db31942e9c4ade"
SELECTED_SHA256 = "a6162e8bd89893b38bf41b5782a7996a34ea6364be8731b23690fb69f945ebef"

SOURCE_HEADER = ("Method", "Dataset", "Config_Key", "Method_Detail")
SELECTED_HEADER = (
    "dataset",
    "variant",
    "penalization_mode",
    "lambda_fair",
    "preproc_alpha",
    "beta",
    "beta_ra",
    "lambda_mobility",
    "config_key",
    "penalty_reduction",
    "selection_evaluation_split",
    "selection_k",
    "source_method",
    "source_best_configs_sha256",
)

# dataset, variant, source method, config key, lambda_fair, preproc_alpha,
# beta, lambda_mobility
EXPECTED = (
    ("ego-facebook", "cafe_in", "CAFE-In", "lambda_fair_0.3_beta_1.0_lam_0.5", "0.3", "", "1.0", "0.5"),
    ("ego-facebook", "cafe_pre", "CAFE-Pre", "preproc_alpha_0.05_beta_0.5_lam_0.3", "", "0.05", "0.5", "0.3"),
    ("oklahoma", "cafe_in", "CAFE-In", "lambda_fair_0.3_beta_5.0_lam_0.5", "0.3", "", "5.0", "0.5"),
    ("oklahoma", "cafe_pre", "CAFE-Pre", "preproc_alpha_1.0_beta_0.5_lam_0.5", "", "1.0", "0.5", "0.5"),
    ("unc", "cafe_in", "CAFE-In", "lambda_fair_1.0_beta_5.0_lam_0.5", "1.0", "", "5.0", "0.5"),
    ("unc", "cafe_pre", "CAFE-Pre", "preproc_alpha_1.0_beta_1.0_lam_0.5", "", "1.0", "1.0", "0.5"),
    ("pokec_n", "cafe_in", "CAFE-In", "lambda_fair_0.05_beta_1.0_lam_0.1", "0.05", "", "1.0", "0.1"),
    ("pokec_n", "cafe_pre", "CAFE-Pre", "preproc_alpha_0.05_beta_2.0_lam_0.5", "", "0.05", "2.0", "0.5"),
    ("brightkite", "cafe_in", "CAFE-In", "lambda_fair_0.05_beta_5.0_lam_0.1", "0.05", "", "5.0", "0.1"),
    ("brightkite", "cafe_pre", "CAFE-Pre", "preproc_alpha_0.3_beta_2.0_lam_0.5", "", "0.3", "2.0", "0.5"),
)


class VerificationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    if not path.is_file():
        raise VerificationError(f"required file is absent: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_csv(path: Path, header: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != header:
                raise VerificationError(
                    f"{path}: header differs from {list(header)}"
                )
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise VerificationError(f"cannot read {path}: {exc}") from exc
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise VerificationError(f"{path}: malformed CSV row")
    return rows


def verify_source() -> None:
    actual = sha256(SOURCE)
    if actual != SOURCE_SHA256:
        raise VerificationError(
            f"{SOURCE}: SHA256 changed: expected {SOURCE_SHA256}, got {actual}"
        )
    rows = strict_csv(SOURCE, SOURCE_HEADER)
    lookup: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (row["Dataset"], row["Method"])
        if key in lookup:
            raise VerificationError(f"{SOURCE}: duplicate source row {key}")
        lookup[key] = row["Config_Key"]
    for dataset, _variant, method, config, *_rest in EXPECTED:
        actual_config = lookup.get((dataset, method))
        if actual_config != config:
            raise VerificationError(
                f"{SOURCE}: {dataset}/{method} must be {config}, got {actual_config}"
            )


def verify_selected() -> None:
    actual = sha256(SELECTED)
    if actual != SELECTED_SHA256:
        raise VerificationError(
            f"{SELECTED}: SHA256 changed: expected {SELECTED_SHA256}, got {actual}"
        )
    rows = strict_csv(SELECTED, SELECTED_HEADER)
    if len(rows) != len(EXPECTED):
        raise VerificationError(
            f"{SELECTED}: expected {len(EXPECTED)} rows, found {len(rows)}"
        )
    for row, expected in zip(rows, EXPECTED):
        dataset, variant, method, config, lambda_fair, preproc_alpha, beta, mobility = expected
        wanted = {
            "dataset": dataset,
            "variant": variant,
            "penalization_mode": "inproc" if variant == "cafe_in" else "preproc",
            "lambda_fair": lambda_fair,
            "preproc_alpha": preproc_alpha,
            "beta": beta,
            "beta_ra": beta,
            "lambda_mobility": mobility,
            "config_key": config,
            "penalty_reduction": "mean",
            "selection_evaluation_split": "test",
            "selection_k": "10",
            "source_method": method,
            "source_best_configs_sha256": SOURCE_SHA256,
        }
        if row != wanted:
            differences = {
                key: {"expected": wanted[key], "actual": row.get(key)}
                for key in wanted
                if row.get(key) != wanted[key]
            }
            raise VerificationError(
                f"{SELECTED}: adapted row {dataset}/{variant} differs: {differences}"
            )


def verify_lineage() -> None:
    try:
        document = json.loads(LINEAGE.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read {LINEAGE}: {exc}") from exc
    checks = {
        "source.path": (document.get("source") or {}).get("path"),
        "source.sha256": (document.get("source") or {}).get("sha256"),
        "selected_configs.path": (document.get("selected_configs") or {}).get("path"),
        "selected_configs.selected_configs_sha256": (
            document.get("selected_configs") or {}
        ).get("selected_configs_sha256"),
        "selected_configs.row_count": (
            document.get("selected_configs") or {}
        ).get("row_count"),
        "configuration_selection.evaluation_split": (
            document.get("configuration_selection") or {}
        ).get("evaluation_split"),
        "configuration_selection.evaluation_cutoff": (
            document.get("configuration_selection") or {}
        ).get("evaluation_cutoff"),
        "execution_semantics.penalty_reduction": (
            document.get("execution_semantics") or {}
        ).get("penalty_reduction"),
        "execution_semantics.eq8_edgewise_penalty": (
            document.get("execution_semantics") or {}
        ).get("eq8_edgewise_penalty"),
        "execution_semantics.legacy_nonedgewise_cafe_in": (
            document.get("execution_semantics") or {}
        ).get("legacy_nonedgewise_cafe_in"),
        "execution_semantics.publishable_as_evidence_for_eq8": (
            document.get("execution_semantics") or {}
        ).get("publishable_as_evidence_for_eq8"),
    }
    wanted = {
        "source.path": str(SOURCE),
        "source.sha256": SOURCE_SHA256,
        "selected_configs.path": str(SELECTED),
        "selected_configs.selected_configs_sha256": SELECTED_SHA256,
        "selected_configs.row_count": len(EXPECTED),
        "configuration_selection.evaluation_split": "test",
        "configuration_selection.evaluation_cutoff": 10,
        "execution_semantics.penalty_reduction": "mean",
        "execution_semantics.eq8_edgewise_penalty": False,
        "execution_semantics.legacy_nonedgewise_cafe_in": True,
        "execution_semantics.publishable_as_evidence_for_eq8": False,
    }
    differences = {
        key: {"expected": wanted[key], "actual": checks[key]}
        for key in wanted
        if checks[key] != wanted[key]
    }
    if differences:
        raise VerificationError(f"{LINEAGE}: lineage differs: {differences}")


def main() -> int:
    verify_source()
    verify_selected()
    verify_lineage()
    print(f"verified source SHA256:   {SOURCE_SHA256}")
    print(f"verified selected SHA256: {SELECTED_SHA256}")
    print(f"selected configurations:  {SELECTED}")
    print(f"lineage:                  {LINEAGE}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
