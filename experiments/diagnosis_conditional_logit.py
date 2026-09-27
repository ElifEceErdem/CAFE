#!/usr/bin/env python3
"""Fit the RQ1 diagnosis-cohort conditional hit@10 regressions.

The input is produced by run_precision_by_coreness.py and contains the five
original diagnosis scorers: CN, RA, AA, LightGCN, and HL-GNN. This module fits
both the requested degree+coreness specification and the local-clustering
control, with a held-out-opportunity sensitivity. Covariance is clustered by
source user so repeated model seeds are not treated as independent users.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.rq1_structural_controls import (  # noqa: E402
    StructuralControlError,
    atomic_bytes,
    atomic_csv,
    atomic_json,
    fit_logit,
    holm_adjust,
    sha256,
    standardize,
    vif_diagnostics,
)


DATASETS = ("ego-facebook", "oklahoma", "unc", "pokec_n", "brightkite")
METHODS = ("CN", "RA", "AA", "LightGCN", "HL-GNN")
SEEDS = (42, 43, 44)
PROFILE = "rq1_diagnosis_conditional_hit_v1"
CODE_FILES = (
    "experiments/run_precision_by_coreness.py",
    "experiments/diagnosis_conditional_logit.py",
    "experiments/rq1_structural_controls.py",
    "metrics/centrality_equality_gain.py",
    "metrics/graph_utils.py",
    "core/data_loader.py",
    "backbone/snap_dataset.py",
)
SPECS = {
    "degree_coreness": ("z_log_degree", "z_coreness"),
    "degree_coreness_clustering": (
        "z_log_degree", "z_coreness", "z_clustering",
    ),
    "degree_coreness_clustering_opportunity": (
        "z_log_degree", "z_coreness", "z_clustering",
        "z_log1p_test_targets",
    ),
}


def normalize_dataset(value: str) -> str:
    normal = value.strip().lower().replace("-", "_")
    aliases = {
        "ego_facebook": "ego-facebook",
        "oklahoma": "oklahoma",
        "unc": "unc",
        "pokec_n": "pokec_n",
        "brightkite": "brightkite",
    }
    if normal not in aliases:
        raise StructuralControlError(f"Unsupported dataset: {value!r}")
    return aliases[normal]


def code_hashes() -> dict[str, str]:
    result = {}
    for relative in CODE_FILES:
        path = REPO / relative
        if not path.is_file():
            raise StructuralControlError(f"Code input missing: {path}")
        result[relative] = sha256(path)
    return result


def prepare_input(raw: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, dict]:
    required = {
        "node_id", "model", "precision", "node_coreness", "degree",
        "clustering_coefficient", "n_test_targets", "training_seed",
        "split_seed", "split_fingerprint",
    }
    missing = required - set(raw)
    if missing:
        raise StructuralControlError(f"Diagnosis input lacks {sorted(missing)}")
    if set(raw["model"].astype(str)) != set(METHODS):
        raise StructuralControlError("Diagnosis method set mismatch")
    if set(raw["training_seed"].astype(int)) != set(SEEDS):
        raise StructuralControlError("Diagnosis training-seed set mismatch")
    if set(raw["split_seed"].astype(int)) != {42}:
        raise StructuralControlError("Diagnosis split seed is not fixed at 42")
    fingerprints = set(raw["split_fingerprint"].dropna().astype(str))
    if len(fingerprints) != 1:
        raise StructuralControlError("Diagnosis input mixes graph splits")
    if raw.duplicated(["model", "training_seed", "node_id"]).any():
        raise StructuralControlError("Duplicate diagnosis method/seed/user rows")
    counts = raw.groupby(["model", "training_seed"])["node_id"].nunique()
    if counts.nunique() != 1:
        raise StructuralControlError("Diagnosis methods/seeds use different user sets")

    frame = raw.rename(columns={
        "model": "method",
        "precision": "precision_at_10",
        "node_coreness": "coreness",
        "clustering_coefficient": "clustering",
    }).copy()
    frame.insert(0, "dataset", dataset)
    precision = pd.to_numeric(frame["precision_at_10"], errors="coerce")
    if not np.isfinite(precision).all() or ((precision < 0) | (precision > 1)).any():
        raise StructuralControlError("Invalid per-user precision")
    if (frame["degree"].astype(float) <= 0).any():
        raise StructuralControlError("log(degree) is undefined for a diagnosis user")
    if (frame["n_test_targets"].astype(int) <= 0).any():
        raise StructuralControlError("Evaluation user lacks held-out targets")
    frame["hit_at_10"] = (precision > 0).astype(int)
    frame["hit_count_at_10"] = np.rint(precision * 10).astype(int)
    if not np.allclose(frame["hit_count_at_10"] / 10.0, precision, atol=1e-12):
        raise StructuralControlError("Precision is not an exact top-10 hit count")
    frame["log_degree"] = np.log(frame["degree"].astype(float))
    frame["log1p_test_targets"] = np.log1p(
        frame["n_test_targets"].astype(float)
    )
    frame, scales = standardize(
        frame,
        ("log_degree", "coreness", "clustering", "log1p_test_targets"),
    )
    return frame, {
        "split_fingerprint": next(iter(fingerprints)),
        "feature_standardization": scales,
    }


def fit_all(frame: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficient_frames = []
    diagnostics = []
    for method in METHODS:
        method_frame = frame.loc[frame["method"] == method]
        deterministic = method in {"CN", "RA", "AA"}
        for specification, predictors in SPECS.items():
            coefficients, diagnostic = fit_logit(method_frame, predictors)
            coefficients.insert(0, "specification", specification)
            coefficients.insert(0, "method", method)
            coefficients.insert(0, "dataset", dataset)
            coefficients["fit_scope"] = "pooled_training_seeds_user_clustered"
            coefficients["deterministic_scorer"] = deterministic
            coefficient_frames.append(coefficients)
            diagnostics.append({
                "dataset": dataset,
                "method": method,
                "specification": specification,
                "fit_scope": "pooled_training_seeds_user_clustered",
                "deterministic_scorer": deterministic,
                **diagnostic,
                "vif_json": json.dumps(
                    vif_diagnostics(method_frame, predictors), sort_keys=True
                ),
            })
            for seed in SEEDS:
                seed_frame = method_frame.loc[
                    method_frame["training_seed"].astype(int) == seed
                ]
                seed_coefficients, seed_diagnostic = fit_logit(
                    seed_frame, predictors
                )
                seed_coefficients.insert(0, "training_seed", seed)
                seed_coefficients.insert(0, "specification", specification)
                seed_coefficients.insert(0, "method", method)
                seed_coefficients.insert(0, "dataset", dataset)
                seed_coefficients["fit_scope"] = "single_training_seed"
                seed_coefficients["deterministic_scorer"] = deterministic
                coefficient_frames.append(seed_coefficients)
                diagnostics.append({
                    "dataset": dataset,
                    "method": method,
                    "specification": specification,
                    "fit_scope": "single_training_seed",
                    "training_seed": seed,
                    "deterministic_scorer": deterministic,
                    **seed_diagnostic,
                    "vif_json": json.dumps(
                        vif_diagnostics(seed_frame, predictors), sort_keys=True
                    ),
                })
    coefficient_frame = pd.concat(coefficient_frames, ignore_index=True)
    diagnostic_frame = pd.DataFrame(diagnostics)
    if not diagnostic_frame["converged"].all():
        bad = diagnostic_frame.loc[
            ~diagnostic_frame["converged"],
            ["method", "specification", "fit_scope", "training_seed",
             "max_absolute_gradient"],
        ].to_dict("records")
        raise StructuralControlError(f"Diagnosis logistic fit failed: {bad}")
    return coefficient_frame, diagnostic_frame


def run_dataset(
    dataset: str,
    input_csv: Path,
    input_metadata: Path,
    output_dir: Path,
) -> int:
    dataset = normalize_dataset(dataset)
    if (output_dir / "_SUCCESS").is_file():
        print(f"Diagnosis conditional-logit shard already complete: {dataset}")
        return 0
    if not input_csv.is_file() or not input_metadata.is_file():
        raise StructuralControlError("Diagnosis runner output is incomplete")
    metadata = json.loads(input_metadata.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or metadata.get("dataset") != dataset:
        raise StructuralControlError("Diagnosis metadata status/dataset mismatch")
    if metadata.get("methods") != list(METHODS):
        raise StructuralControlError("Diagnosis metadata method order mismatch")
    if metadata.get("training_seeds") != list(SEEDS):
        raise StructuralControlError("Diagnosis metadata seed set mismatch")

    frame, provenance = prepare_input(pd.read_csv(input_csv), dataset)
    if metadata.get("split_fingerprint") != provenance["split_fingerprint"]:
        raise StructuralControlError("Diagnosis metadata fingerprint mismatch")
    coefficients, diagnostics = fit_all(frame, dataset)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "diagnosis_per_user_hits.csv": frame,
        "diagnosis_logit_coefficients.csv": coefficients,
        "diagnosis_logit_diagnostics.csv": diagnostics,
    }
    for name, output in outputs.items():
        atomic_csv(output_dir / name, output)
    manifest = {
        "schema_version": 1,
        "profile": PROFILE,
        "status": "complete",
        "dataset": dataset,
        "methods": list(METHODS),
        "training_seeds": list(SEEDS),
        "split_seed": 42,
        **provenance,
        "inference": (
            "pooled fixed-split model-seed rows with source-user-clustered "
            "sandwich covariance"
        ),
        "deterministic_scorers": ["CN", "RA", "AA"],
        "input_sha256": {
            input_csv.name: sha256(input_csv),
            input_metadata.name: sha256(input_metadata),
        },
        "code_sha256": code_hashes(),
        "outputs_sha256": {
            name: sha256(output_dir / name) for name in outputs
        },
    }
    atomic_json(output_dir / "diagnosis_logit_manifest.json", manifest)
    atomic_bytes(output_dir / "_SUCCESS", b"complete\n")
    print(f"Completed diagnosis conditional logits: {dataset}")
    return 0


def make_table(coefficients: pd.DataFrame) -> str:
    wanted = coefficients.loc[
        (coefficients["fit_scope"] == "pooled_training_seeds_user_clustered")
        & (coefficients["coefficient"] == "z_coreness")
    ]
    lines = [
        "% First line: degree+coreness. Second line: +local clustering.",
        "\\begin{table*}[t]", "\\centering", "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\caption{Conditional coreness association for the original RQ1 "
        "diagnosis scorers. Each cell reports the standardized coreness "
        "coefficient and source-user-clustered 95\\% Wald interval for "
        "$\\log d+C$ (first line) and after adding local clustering "
        "(second line).}",
        "\\label{tab:rq1-diagnosis-conditional-hit}",
        "\\begin{tabular}{lrrrrr}", "\\toprule",
        "Dataset & CN & RA & AA & LightGCN & HL-GNN \\\\", "\\midrule",
    ]
    labels = {
        "ego-facebook": "Ego-FB", "oklahoma": "Oklahoma", "unc": "UNC",
        "pokec_n": "Pokec-n", "brightkite": "Brightkite",
    }
    for dataset in DATASETS:
        cells = []
        for method in METHODS:
            rows = wanted.loc[
                (wanted["dataset"] == dataset) & (wanted["method"] == method)
            ].set_index("specification")
            base = rows.loc["degree_coreness"]
            adjusted = rows.loc["degree_coreness_clustering"]
            cells.append(
                "\\shortstack{"
                f"{base['beta']:+.2f} [{base['ci95_low']:+.2f}, "
                f"{base['ci95_high']:+.2f}]\\\\"
                f"{adjusted['beta']:+.2f} [{adjusted['ci95_low']:+.2f}, "
                f"{adjusted['ci95_high']:+.2f}]"
                "}"
            )
        lines.append(f"{labels[dataset]} & " + " & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule", "\\end{tabular}",
        "\\par\\footnotesize Coefficients are log-odds changes per one-SD "
        "coreness increase, conditional on the observed graph. CN, RA and AA "
        "are deterministic on the fixed split; duplicating their rows does not "
        "create seed-level uncertainty because covariance is clustered by user.",
        "\\end{table*}", "",
    ]
    return "\n".join(lines)


def finalize(campaign: Path, if_complete: bool, rebuild: bool) -> int:
    campaign.mkdir(parents=True, exist_ok=True)
    with (campaign / ".diagnosis_finalize.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        missing = []
        manifests = {}
        code_documents = set()
        coefficient_frames = []
        diagnostic_frames = []
        user_frames = []
        for dataset in DATASETS:
            directory = campaign / "shards" / dataset
            manifest_path = directory / "diagnosis_logit_manifest.json"
            if not (directory / "_SUCCESS").is_file() or not manifest_path.is_file():
                missing.append(dataset)
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("profile") != PROFILE or manifest.get("dataset") != dataset:
                raise StructuralControlError(f"{manifest_path}: profile mismatch")
            code_documents.add(json.dumps(manifest.get("code_sha256"), sort_keys=True))
            paths = {
                "diagnosis_per_user_hits.csv": directory / "diagnosis_per_user_hits.csv",
                "diagnosis_logit_coefficients.csv": directory / "diagnosis_logit_coefficients.csv",
                "diagnosis_logit_diagnostics.csv": directory / "diagnosis_logit_diagnostics.csv",
            }
            for name, path in paths.items():
                if manifest["outputs_sha256"].get(name) != sha256(path):
                    raise StructuralControlError(f"{path}: hash mismatch")
            user_frames.append(pd.read_csv(paths["diagnosis_per_user_hits.csv"]))
            coefficient_frames.append(
                pd.read_csv(paths["diagnosis_logit_coefficients.csv"])
            )
            diagnostic_frames.append(
                pd.read_csv(paths["diagnosis_logit_diagnostics.csv"])
            )
            manifests[dataset] = sha256(manifest_path)
        if missing:
            message = f"Missing diagnosis conditional-logit datasets: {missing}"
            if if_complete:
                print(message)
                return 0
            raise StructuralControlError(message)
        if len(code_documents) != 1:
            raise StructuralControlError("Diagnosis shards used different code")

        output_dir = campaign / "finalized"
        if (output_dir / "_COMPLETE").is_file() and not rebuild:
            print(f"Diagnosis conditional logits already finalized: {output_dir}")
            return 0
        users = pd.concat(user_frames, ignore_index=True)
        coefficients = pd.concat(coefficient_frames, ignore_index=True)
        diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
        primary = coefficients.loc[
            (coefficients["fit_scope"] == "pooled_training_seeds_user_clustered")
            & (coefficients["coefficient"] == "z_coreness")
        ].copy()
        primary["p_value_holm_25"] = np.nan
        for specification, indices in primary.groupby("specification").groups.items():
            primary.loc[indices, "p_value_holm_25"] = holm_adjust(
                primary.loc[indices, "p_value"].to_numpy()
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "diagnosis_per_user_hits_all.csv": users,
            "diagnosis_logit_coefficients_all.csv": coefficients,
            "diagnosis_logit_diagnostics_all.csv": diagnostics,
            "diagnosis_coreness_coefficients_holm.csv": primary,
        }
        for name, frame in outputs.items():
            atomic_csv(output_dir / name, frame)
        table_name = "diagnosis_conditional_hit_table.tex"
        atomic_bytes(output_dir / table_name, make_table(coefficients).encode("utf-8"))
        counts = {}
        for specification in ("degree_coreness", "degree_coreness_clustering"):
            block = primary.loc[primary["specification"] == specification]
            counts[specification] = {
                "positive_beta": int((block["beta"] > 0).sum()),
                "positive_ci_excludes_zero": int((block["ci95_low"] > 0).sum()),
                "positive_holm_0_05": int(
                    ((block["beta"] > 0) & (block["p_value_holm_25"] < 0.05)).sum()
                ),
                "cells": int(len(block)),
            }
        atomic_json(output_dir / "diagnosis_conditional_counts.json", counts)
        output_names = [
            *outputs, table_name, "diagnosis_conditional_counts.json",
        ]
        atomic_json(output_dir / "diagnosis_conditional_integrity.json", {
            "schema_version": 1,
            "profile": PROFILE,
            "status": "complete",
            "datasets": list(DATASETS),
            "methods": list(METHODS),
            "training_seeds": list(SEEDS),
            "split_seed": 42,
            "task_manifest_sha256": manifests,
            "code_sha256": json.loads(next(iter(code_documents))),
            "outputs_sha256": {
                name: sha256(output_dir / name) for name in output_names
            },
        })
        atomic_bytes(output_dir / "_COMPLETE", b"complete\n")
        print(f"Finalized diagnosis conditional logits: {output_dir}")
        return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--dataset", required=True)
    run.add_argument("--input-csv", required=True)
    run.add_argument("--input-metadata", required=True)
    run.add_argument("--output-dir", required=True)
    finish = commands.add_parser("finalize")
    finish.add_argument("--campaign-dir", required=True)
    finish.add_argument("--if-complete", action="store_true")
    finish.add_argument("--rebuild", action="store_true")
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "run":
            return run_dataset(
                args.dataset,
                Path(args.input_csv).resolve(),
                Path(args.input_metadata).resolve(),
                Path(args.output_dir).resolve(),
            )
        return finalize(
            Path(args.campaign_dir).resolve(), args.if_complete, args.rebuild
        )
    except (
        StructuralControlError,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        np.linalg.LinAlgError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
