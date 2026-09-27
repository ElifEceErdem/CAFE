#!/usr/bin/env python3
"""Validate and aggregate the Brightkite random-split protocol control.

Each GPU shard is produced by ``analysis/run_stratified_cafe.py`` for one
connectivity-preserving random split and one matched training seed.  This
module refuses partial or mixed campaigns and emits the exact numbers needed
for the RQ1 protocol-control sentence: lowest/highest-coreness Precision@10
and the source-bin Degree-EG Spearman gradient for every Fig. 4 scorer.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import stats


REPO = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(42, 52))
METHODS = ("RA", "HL-GNN", "Fusion", "CAFE-In", "CAFE-Pre")
PROFILE = "brightkite_connectivity_preserving_random_85_5_10_v1"
SCHEMA_VERSION = 1
CODE_FILES = (
    "reproduce/jobs/t17_brightkite_random.job",
    "experiments/configs/verify_legacy_best_configs.py",
    "analysis/run_stratified_cafe.py",
    "analysis/run_precision_by_coreness.py",
    "analysis/finalize_brightkite_random_control.py",
    "experiments/cafe_methods.py",
    "fairness_methods/inprocessing.py",
    "metrics/centrality_equality_gain.py",
    "metrics/equality_gain.py",
    "metrics/graph_utils.py",
    "core/data_loader.py",
    "backbone/snap_dataset.py",
)


class ControlError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_hashes() -> dict[str, str]:
    result = {}
    for relative in CODE_FILES:
        path = REPO / relative
        if not path.is_file():
            raise ControlError(f"Code input absent: {path}")
        result[relative] = sha256(path)
    return result


def atomic_bytes(path: Path, payload: bytes) -> None:
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


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    atomic_bytes(path, payload)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
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


def task_dir(campaign: Path, seed: int) -> Path:
    return campaign / "shards" / f"seed_{seed}"


def expected_paths(directory: Path) -> dict[str, Path]:
    return {
        "per_user": directory / "stratified_cafe_brightkite.csv",
        "per_bin_eg": directory / "eg_by_coreness_cafe_brightkite_per_run.csv",
        "precision_summary": directory / "summary_precision_by_bin_brightkite.csv",
        "metadata": directory / "metadata_brightkite.json",
    }


def _one_value(frame: pd.DataFrame, column: str, wanted: object, source: Path) -> None:
    if column not in frame:
        raise ControlError(f"{source}: missing column {column}")
    values = set(frame[column].dropna().astype(str))
    if values != {str(wanted)}:
        raise ControlError(
            f"{source}: {column} must be {wanted!r}, found {sorted(values)}"
        )


def validate_task(
    directory: Path,
    seed: int,
    selected_configs: Path,
    edgewise_preflight: bool,
    legacy_override: bool,
    penalty_reduction: str,
    configuration_origin: str,
) -> dict[str, object]:
    if seed not in SEEDS:
        raise ControlError(f"Seed {seed} is outside {list(SEEDS)}")
    if not edgewise_preflight and not legacy_override:
        raise ControlError("Eq. (8) edgewise preflight failed; refusing this shard")
    if not selected_configs.is_file():
        raise ControlError(f"Selected-config manifest absent: {selected_configs}")

    paths = expected_paths(directory)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ControlError(f"{directory}: missing artifacts {missing}")

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or metadata.get("dataset") != "brightkite":
        raise ControlError(f"{paths['metadata']}: invalid status or dataset")
    if metadata.get("split_protocol") != "brightkite_random_control":
        raise ControlError(f"{paths['metadata']}: not a random-control artifact")
    if metadata.get("penalty_reduction") != penalty_reduction:
        raise ControlError(
            f"{paths['metadata']}: penalty reduction is not {penalty_reduction}"
        )
    runs = metadata.get("runs")
    if not isinstance(runs, list) or len(runs) != 1:
        raise ControlError(f"{paths['metadata']}: expected exactly one run")
    run = runs[0]
    for key in ("training_seed", "split_seed"):
        if int(run.get(key, -1)) != seed:
            raise ControlError(f"{paths['metadata']}: {key} does not equal {seed}")
    if run.get("split_protocol") != "brightkite_random_control":
        raise ControlError(f"{paths['metadata']}: run protocol mismatch")
    if run.get("split_sha256") is not None:
        raise ControlError(f"{paths['metadata']}: random split must not have split_sha256")
    if int(run.get("k", -1)) != 10:
        raise ControlError(f"{paths['metadata']}: K is not 10")
    if set(run.get("methods", [])) != set(METHODS):
        raise ControlError(f"{paths['metadata']}: method set mismatch")
    if run.get("eg_graph_semantics") != "simple_undirected_set_union":
        raise ControlError(f"{paths['metadata']}: incorrect EG semantics")
    edge_counts = run.get("split_edge_counts")
    edge_fractions = run.get("split_edge_fractions")
    if not isinstance(edge_counts, dict) or set(edge_counts) != {"train", "valid", "test"}:
        raise ControlError(f"{paths['metadata']}: missing split edge counts")
    if not isinstance(edge_fractions, dict) or set(edge_fractions) != {"train", "valid", "test"}:
        raise ControlError(f"{paths['metadata']}: missing split edge fractions")
    total_edges = sum(int(value) for value in edge_counts.values())
    if total_edges <= 0:
        raise ControlError(f"{paths['metadata']}: invalid split edge total")
    expected = {"train": 0.85, "valid": 0.05, "test": 0.10}
    for name, wanted in expected.items():
        actual_count_fraction = int(edge_counts[name]) / total_edges
        recorded_fraction = float(edge_fractions[name])
        if not math.isclose(actual_count_fraction, recorded_fraction, abs_tol=1e-12):
            raise ControlError(f"{paths['metadata']}: inconsistent {name} fraction")
        # Tolerance is a few edges, not one: the connectivity-preserving
        # splitter rounds each partition's target edge count independently,
        # which lands up to ~2 edges off the ideal fraction on every seed
        # (observed: +1.7/-0.9/-0.8 edges on 214078 total).  Eight edges is
        # still 0.004% and rejects any genuinely wrong ratio.
        if abs(recorded_fraction - wanted) > 8.0 / total_edges + 1e-12:
            raise ControlError(
                f"{paths['metadata']}: {name} fraction {recorded_fraction} is not {wanted}"
            )

    selected_hash = sha256(selected_configs)
    if metadata.get("selected_configs_sha256") != selected_hash:
        raise ControlError(f"{paths['metadata']}: selected-config hash mismatch")

    per_user = pd.read_csv(paths["per_user"])
    required_user = {
        "node_id", "model", "precision", "node_coreness", "degree",
        "run", "training_seed", "split_seed", "split_fingerprint",
        "coreness_bin", "coreness_bin_label",
    }
    if required_user - set(per_user):
        raise ControlError(
            f"{paths['per_user']}: missing {sorted(required_user - set(per_user))}"
        )
    if set(per_user["model"].astype(str)) != set(METHODS):
        raise ControlError(f"{paths['per_user']}: method set mismatch")
    _one_value(per_user, "training_seed", seed, paths["per_user"])
    _one_value(per_user, "split_seed", seed, paths["per_user"])
    fingerprints = set(per_user["split_fingerprint"].dropna().astype(str))
    if fingerprints != {str(run.get("split_fingerprint"))}:
        raise ControlError(f"{paths['per_user']}: split fingerprint mismatch")
    precision = pd.to_numeric(per_user["precision"], errors="coerce").to_numpy()
    if not np.isfinite(precision).all() or ((precision < 0) | (precision > 1)).any():
        raise ControlError(f"{paths['per_user']}: invalid precision values")
    counts = per_user.groupby("model")["node_id"].nunique()
    if len(counts) != len(METHODS) or counts.nunique() != 1:
        raise ControlError(f"{paths['per_user']}: methods have different user sets")
    if per_user.duplicated(["model", "node_id"]).any():
        raise ControlError(f"{paths['per_user']}: duplicate method/user rows")

    eg = pd.read_csv(paths["per_bin_eg"])
    required_eg = {
        "model", "coreness_bin", "coreness_bin_label", "equality_gain",
        "gini_control", "gini_pred", "n_sources", "run", "training_seed",
        "split_seed",
    }
    if required_eg - set(eg):
        raise ControlError(
            f"{paths['per_bin_eg']}: missing {sorted(required_eg - set(eg))}"
        )
    if set(eg["model"].astype(str)) != set(METHODS):
        raise ControlError(f"{paths['per_bin_eg']}: method set mismatch")
    _one_value(eg, "training_seed", seed, paths["per_bin_eg"])
    _one_value(eg, "split_seed", seed, paths["per_bin_eg"])
    numeric = eg[["equality_gain", "gini_control", "gini_pred"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy()).all():
        raise ControlError(f"{paths['per_bin_eg']}: non-finite EG/Gini value")
    bins_by_method = {
        method: tuple(sorted(group["coreness_bin"].astype(int).unique()))
        for method, group in eg.groupby("model")
    }
    if len(set(bins_by_method.values())) != 1:
        raise ControlError(f"{paths['per_bin_eg']}: methods use different bins")
    if len(next(iter(bins_by_method.values()))) < 3:
        raise ControlError(f"{paths['per_bin_eg']}: fewer than three realized bins")

    recommendation = directory / f"recommendations_brightkite_seed{seed}.npz"
    if not recommendation.is_file():
        raise ControlError(f"Recommendation artifact missing: {recommendation}")
    with np.load(recommendation, allow_pickle=False) as payload:
        required_keys = {
            "sources", "targets_ra", "targets_hl_gnn", "targets_fusion",
            "targets_cafe_in", "targets_cafe_pre", "metadata_json",
        }
        if required_keys - set(payload.files):
            raise ControlError(
                f"{recommendation}: missing {sorted(required_keys - set(payload.files))}"
            )
        sources = payload["sources"]
        if len(sources) != int(counts.iloc[0]):
            raise ControlError(f"{recommendation}: source count mismatch")
        for key in required_keys - {"sources", "metadata_json"}:
            if payload[key].shape != (len(sources), 10):
                raise ControlError(f"{recommendation}: {key} has wrong shape")

    hashes = {name: sha256(path) for name, path in paths.items()}
    hashes[recommendation.name] = sha256(recommendation)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "profile": PROFILE,
        "status": "complete",
        "dataset": "brightkite",
        "training_seed": seed,
        "split_seed": seed,
        "split_protocol": "connectivity_preserving_random",
        "requested_fractions": {"train": 0.85, "validation": 0.05, "test": 0.10},
        "split_fingerprint": run.get("split_fingerprint"),
        "split_edge_counts": edge_counts,
        "split_edge_fractions": edge_fractions,
        "methods": list(METHODS),
        "selected_configs_sha256": selected_hash,
        "eq8_edgewise_penalty_preflight": edgewise_preflight,
        "legacy_nonedgewise_override": legacy_override,
        "penalty_reduction": penalty_reduction,
        "configuration_origin": configuration_origin,
        "code_sha256": code_hashes(),
        "artifacts_sha256": hashes,
    }
    atomic_json(directory / "task_manifest.json", manifest)
    atomic_bytes(directory / "_SUCCESS", b"complete\n")
    return manifest


def mean_interval(values: pd.Series) -> tuple[float, float, float, float]:
    array = values.astype(float).to_numpy()
    if len(array) != len(SEEDS) or not np.isfinite(array).all():
        raise ControlError("A summary cell does not contain ten finite split values")
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    half = float(stats.t.ppf(0.975, len(array) - 1) * sd / math.sqrt(len(array)))
    return mean, sd, mean - half, mean + half


def build_outputs(per_user: pd.DataFrame, eg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile_rows = []
    rho_rows = []
    for seed in SEEDS:
        for method in METHODS:
            users = per_user.loc[
                (per_user["training_seed"].astype(int) == seed)
                & (per_user["model"] == method)
            ]
            by_bin = users.groupby("coreness_bin", as_index=False)["precision"].mean()
            by_bin = by_bin.sort_values("coreness_bin")
            low = float(by_bin.iloc[0]["precision"])
            high = float(by_bin.iloc[-1]["precision"])
            profile_rows.append({
                "method": method,
                "seed": seed,
                "lowest_bin": int(by_bin.iloc[0]["coreness_bin"]),
                "highest_bin": int(by_bin.iloc[-1]["coreness_bin"]),
                "lowest_bin_precision": low,
                "highest_bin_precision": high,
                "high_minus_low": high - low,
                "high_greater_than_low": bool(high > low),
            })

            eg_block = eg.loc[
                (eg["training_seed"].astype(int) == seed)
                & (eg["model"] == method)
            ].sort_values("coreness_bin")
            rho, p_value = stats.spearmanr(
                eg_block["coreness_bin"].astype(float),
                eg_block["equality_gain"].astype(float),
            )
            if not np.isfinite(rho) or not np.isfinite(p_value):
                raise ControlError(f"Undefined EG Spearman correlation: {method}, {seed}")
            rho_rows.append({
                "method": method,
                "seed": seed,
                "realized_bins": int(len(eg_block)),
                "eg_coreness_spearman_rho": float(rho),
                "spearman_p_value": float(p_value),
                "rho_is_negative": bool(rho < 0),
            })

    profiles = pd.DataFrame(profile_rows)
    rhos = pd.DataFrame(rho_rows)
    summary_rows = []
    for method in METHODS:
        p = profiles.loc[profiles["method"] == method]
        r = rhos.loc[rhos["method"] == method]
        row: dict[str, object] = {
            "method": method,
            "split_count": len(SEEDS),
            "high_greater_than_low_count": int(p["high_greater_than_low"].sum()),
            "negative_eg_rho_count": int(r["rho_is_negative"].sum()),
        }
        for prefix, values in (
            ("lowest_bin_precision", p["lowest_bin_precision"]),
            ("highest_bin_precision", p["highest_bin_precision"]),
            ("high_minus_low", p["high_minus_low"]),
            ("eg_coreness_spearman_rho", r["eg_coreness_spearman_rho"]),
        ):
            mean, sd, low, high = mean_interval(values)
            row[f"{prefix}_mean"] = mean
            row[f"{prefix}_sd"] = sd
            row[f"{prefix}_ci95_low"] = low
            row[f"{prefix}_ci95_high"] = high
        summary_rows.append(row)
    return pd.concat(
        [profiles.assign(record_type="precision_profile"),
         rhos.assign(record_type="eg_gradient")],
        ignore_index=True,
        sort=False,
    ), pd.DataFrame(summary_rows)


def latex_table(summary: pd.DataFrame, edgewise: bool) -> str:
    lines = [
        "% Auto-generated Brightkite connectivity-preserving random-split control.",
    ]
    if not edgewise:
        lines.append(
            "% INVALID FOR EQ. (8): CAFE-In used the non-edgewise legacy implementation."
        )
    lines += [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Brightkite random-split protocol control (ten paired "
        "connectivity-preserving 85/5/10 splits). Values are means across "
        "splits; brackets give paired 95\\% confidence intervals.}",
        "\\label{tab:brightkite-random-control}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Method & Lowest P@10 & Highest P@10 & High$-$Low [95\\% CI] & "
        "$\\rho_S$(Degree EG) \\\\",
        "\\midrule",
    ]
    for method in METHODS:
        row = summary.loc[summary["method"] == method].iloc[0]
        invalid = method == "CAFE-In" and not edgewise
        label = f"{method}*" if invalid else method
        lines.append(
            f"{label} & {row['lowest_bin_precision_mean']:.4f} & "
            f"{row['highest_bin_precision_mean']:.4f} & "
            f"{row['high_minus_low_mean']:+.4f} "
            f"[{row['high_minus_low_ci95_low']:+.4f}, "
            f"{row['high_minus_low_ci95_high']:+.4f}] & "
            f"{row['eg_coreness_spearman_rho_mean']:+.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    if not edgewise:
        lines.append(
            "\\par\\footnotesize *Invalid for manuscript use until Eq.~(8) "
            "and CAFE-In configuration selection are corrected."
        )
    lines += ["\\end{table}", ""]
    return "\n".join(lines)


def finalize(campaign: Path, if_complete: bool, rebuild: bool) -> int:
    campaign.mkdir(parents=True, exist_ok=True)
    with (campaign / ".finalize.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        missing = []
        user_frames = []
        eg_frames = []
        manifests = {}
        selected_hashes = set()
        edgewise_values = set()
        legacy_values = set()
        penalty_reductions = set()
        configuration_origins = set()
        code_documents = set()
        for seed in SEEDS:
            directory = task_dir(campaign, seed)
            marker = directory / "_SUCCESS"
            manifest_path = directory / "task_manifest.json"
            if not marker.is_file() or not manifest_path.is_file():
                missing.append(seed)
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("profile") != PROFILE or int(manifest.get("training_seed", -1)) != seed:
                raise ControlError(f"{manifest_path}: profile/seed mismatch")
            selected_hashes.add(manifest.get("selected_configs_sha256"))
            edgewise_values.add(bool(manifest.get("eq8_edgewise_penalty_preflight")))
            legacy_values.add(bool(manifest.get("legacy_nonedgewise_override")))
            penalty_reductions.add(str(manifest.get("penalty_reduction")))
            configuration_origins.add(str(manifest.get("configuration_origin")))
            code_documents.add(json.dumps(manifest.get("code_sha256"), sort_keys=True))
            paths = expected_paths(directory)
            for name, path in paths.items():
                recorded = manifest["artifacts_sha256"].get(name)
                if recorded != sha256(path):
                    raise ControlError(f"{path}: hash mismatch")
            user_frames.append(pd.read_csv(paths["per_user"]))
            eg_frames.append(pd.read_csv(paths["per_bin_eg"]))
            manifests[f"seed_{seed}"] = sha256(manifest_path)
        if missing:
            message = f"Missing Brightkite random-control seeds: {missing}"
            if if_complete:
                print(message)
                return 0
            raise ControlError(message)
        if (
            len(selected_hashes) != 1
            or len(edgewise_values) != 1
            or len(legacy_values) != 1
            or len(penalty_reductions) != 1
            or len(configuration_origins) != 1
            or len(code_documents) != 1
        ):
            raise ControlError("Campaign mixes configurations or Eq. (8) safety states")

        output_dir = campaign / "finalized"
        complete = output_dir / "_COMPLETE"
        if complete.exists() and not rebuild:
            print(f"Already finalized: {output_dir}")
            return 0
        per_user = pd.concat(user_frames, ignore_index=True)
        per_bin_eg = pd.concat(eg_frames, ignore_index=True)
        per_seed, summary = build_outputs(per_user, per_bin_eg)
        edgewise = next(iter(edgewise_values))

        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "brightkite_random_per_user.csv": per_user,
            "brightkite_random_degree_eg_per_bin.csv": per_bin_eg,
            "brightkite_random_per_seed_controls.csv": per_seed,
            "brightkite_random_control_summary.csv": summary,
        }
        for name, frame in outputs.items():
            atomic_csv(output_dir / name, frame)
        table_path = output_dir / "brightkite_random_control_table.tex"
        atomic_bytes(table_path, latex_table(summary, edgewise).encode("utf-8"))

        exact_numbers = {}
        for method in METHODS:
            row = summary.loc[summary["method"] == method].iloc[0]
            exact_numbers[method] = {
                key: (int(value) if key.endswith("_count") else float(value))
                for key, value in row.items()
                if key != "method"
            }
        atomic_json(output_dir / "brightkite_random_main_text_numbers.json", exact_numbers)

        output_names = [
            *outputs,
            table_path.name,
            "brightkite_random_main_text_numbers.json",
        ]
        integrity = {
            "schema_version": SCHEMA_VERSION,
            "profile": PROFILE,
            "status": "complete",
            "split_seeds": list(SEEDS),
            "training_seeds": list(SEEDS),
            "methods": list(METHODS),
            "requested_fractions": {"train": 0.85, "validation": 0.05, "test": 0.10},
            "split_algorithm": "connectivity-preserving random split",
            "selected_configs_sha256": next(iter(selected_hashes)),
            "eq8_edgewise_penalty_preflight": edgewise,
            "legacy_nonedgewise_override": next(iter(legacy_values)),
            "penalty_reduction": next(iter(penalty_reductions)),
            "configuration_origin": next(iter(configuration_origins)),
            "code_sha256": json.loads(next(iter(code_documents))),
            "task_manifest_sha256": manifests,
            "outputs_sha256": {
                name: sha256(output_dir / name) for name in output_names
            },
        }
        atomic_json(output_dir / "brightkite_random_integrity.json", integrity)
        atomic_bytes(complete, b"complete\n")
        print(f"Finalized Brightkite random control: {output_dir}")
        return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument("--task-dir", required=True)
    check.add_argument("--seed", required=True, type=int)
    check.add_argument("--selected-configs", required=True)
    check.add_argument("--edgewise-preflight", required=True, choices=("true", "false"))
    check.add_argument("--legacy-override", action="store_true")
    check.add_argument("--penalty-reduction", required=True, choices=("mean", "sum"))
    check.add_argument(
        "--configuration-origin",
        required=True,
        choices=("validation_selected", "legacy_test_selected"),
    )
    finish = commands.add_parser("finalize")
    finish.add_argument("--campaign-dir", required=True)
    finish.add_argument("--if-complete", action="store_true")
    finish.add_argument("--rebuild", action="store_true")
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "check":
            validate_task(
                Path(args.task_dir).resolve(),
                args.seed,
                Path(args.selected_configs).resolve(),
                args.edgewise_preflight == "true",
                args.legacy_override,
                args.penalty_reduction,
                args.configuration_origin,
            )
            print(f"Validated Brightkite random-control seed {args.seed}")
            return 0
        return finalize(
            Path(args.campaign_dir).resolve(), args.if_complete, args.rebuild
        )
    except (ControlError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
