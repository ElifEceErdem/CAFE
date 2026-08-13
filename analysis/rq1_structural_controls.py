#!/usr/bin/env python3
"""Conditional-hit and alternative-axis controls for RQ1.

This is a read-only reanalysis of the corrected stratified recommendation
artifacts.  For one dataset, ``run`` reconstructs the frozen training split,
verifies every recommendation artifact, computes exact training-graph local
clustering, fits per-user logistic models with source-user-clustered sandwich
covariance, and re-bins the same recommendations along three alternative
stratifiers (coreness, degree, local clustering) plus a nested
degree-band x coreness grid.

The three marginal axes and the nested grid answer different objections.  A
marginal degree axis does NOT settle the "your equality gradient is a degree
gradient" objection, because degree and coreness are correlated and the
degree-ranked profile is expected to decline too.  Nor can a bin-level
partial correlation settle it: mean bin degree is perfectly rank-correlated
with coreness-bin rank (rho_S = +1.0 on all five networks), so that partial
is undefined.  What settles it is the nested grid -- whether the coreness
gradient survives WITHIN a degree band, which is the Equality-Gain twin of
the degree x coreness precision heatmap in Figure 1(a).

``finalize`` requires all five datasets and writes the full coefficient grid,
the CAFE-In columns to append to Table 18(b), all-method diagnostics, and
integrity metadata.  No model is trained by this program.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Mapping, Sequence

import networkit as nk
import numpy as np
import pandas as pd
import torch
from scipy import optimize, stats


REPO = Path(__file__).resolve().parents[1]
for import_root in (REPO, REPO / "backbone", REPO / "experiments"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from core.data_loader import load_dataset  # noqa: E402
from metrics.centrality_equality_gain import (  # noqa: E402
    compute_degree_equality_gain_by_source_bins_for_models,
)
from metrics.coreness import compute_coreness  # noqa: E402
from metrics.graph_utils import canonicalize_undirected_edges  # noqa: E402


DATASETS = ("ego-facebook", "oklahoma", "unc", "pokec_n", "brightkite")
SEEDS = (42, 43, 44)
METHODS = ("RA", "HL-GNN", "Fusion", "CAFE-In", "CAFE-Pre")
NPZ_KEYS = {
    "RA": "targets_ra",
    "HL-GNN": "targets_hl_gnn",
    "Fusion": "targets_fusion",
    "CAFE-In": "targets_cafe_in",
    "CAFE-Pre": "targets_cafe_pre",
}
PROFILE = "rq1_conditional_hit_and_alternative_axes_v2"
SCHEMA_VERSION = 2
K = 10
NUM_BINS = 10
# Marginal alternative-axis controls for the stratified Degree-EG gradient.
# "degree" answers the mechanical confound (coreness is bounded by degree),
# "clustering" the conceptual competitor.  Both re-bin the SAME
# recommendations, so any difference is the stratifier, not the scorer.
AXES = ("coreness", "degree", "clustering")
# The nested control.  A marginal degree axis cannot settle the degree
# objection on its own, because degree and coreness are correlated: a
# degree-ranked stratification is expected to show a gradient too.  What
# settles it is whether the coreness gradient survives WITHIN a degree band.
# The band/bin counts match Figure 1(a) (--num_degree_bins 4,
# --num_coreness_bins 5), so the EG panel is that figure's twin.
NESTED_DEGREE_BANDS = 4
NESTED_CORENESS_BINS = 5
# A Spearman correlation over fewer than three bins is not interpretable.
MIN_BINS_PER_BAND = 3
KNOWN_NONEDGEWISE_SELECTED_CONFIG_SHA256 = (
    "57b80781dbd15cc367c8c857d92ff535d63084a20ed2110438185f4e49fb13df"
)
SCORE_TOLERANCE_PER_OBSERVATION = 1e-7
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
CODE_FILES = (
    "analysis/rq1_structural_controls.py",
    "metrics/centrality_equality_gain.py",
    "metrics/equality_gain.py",
    "metrics/graph_utils.py",
    "metrics/coreness.py",
    "core/data_loader.py",
    "backbone/snap_dataset.py",
    "fairness_methods/inprocessing.py",
)


class StructuralControlError(RuntimeError):
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
            raise StructuralControlError(f"Code input missing: {path}")
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
    atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )


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


def normalize_dataset(value: str) -> str:
    normal = value.strip().lower().replace("-", "_")
    aliases = {
        "ego_facebook": "ego-facebook",
        "oklahoma": "oklahoma",
        "unc": "unc",
        "pokec_n": "pokec_n",
        "brightkite": "brightkite",
    }
    try:
        return aliases[normal]
    except KeyError as exc:
        raise StructuralControlError(f"Unsupported dataset {value!r}") from exc


def split_fingerprint(split_edge: Mapping[str, object], num_nodes: int) -> str:
    digest = hashlib.sha256()
    digest.update(f"num_nodes={num_nodes}\n".encode())
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


def load_frozen_split(dataset: str, data_path: Path):
    split_file = (
        str(data_path / "brightkite" / "temporal_split.pt")
        if dataset == "brightkite"
        else None
    )
    args = SimpleNamespace(
        data_name=dataset,
        data_path=str(data_path),
        val_ratio=0.05,
        test_ratio=0.10,
        encoder="HLGNN",
        device="cpu",
        split_file=split_file,
        data_split_seed=42,
    )
    loaded, split_edge, data, degrees = load_dataset(args)
    num_nodes = (
        loaded._data.num_nodes if hasattr(loaded, "_data") else loaded.data.num_nodes
    )
    return split_edge, data, degrees.cpu().numpy().astype(float), int(num_nodes)


def local_clustering(train_edges: torch.Tensor, num_nodes: int) -> np.ndarray:
    edges = canonicalize_undirected_edges(train_edges.cpu(), num_nodes)
    array = edges.numpy().astype(np.int64, copy=False)
    graph = nk.Graph(num_nodes, weighted=False, directed=False)
    if len(array):
        graph.addEdges(
            (
                np.ascontiguousarray(array[:, 0]),
                np.ascontiguousarray(array[:, 1]),
            )
        )
    if graph.numberOfSelfLoops() != 0:
        raise StructuralControlError("Simple training graph contains self-loops")
    scores = np.asarray(
        nk.centrality.LocalClusteringCoefficient(graph, turbo=True).run().scores(),
        dtype=float,
    )
    if scores.shape != (num_nodes,) or not np.isfinite(scores).all():
        raise StructuralControlError("Invalid local clustering coefficients")
    if (scores < -1e-12).any() or (scores > 1 + 1e-12).any():
        raise StructuralControlError("Local clustering coefficient outside [0,1]")
    return np.clip(scores, 0.0, 1.0)


def test_targets(test_edges: torch.Tensor) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    for raw_u, raw_v in test_edges.cpu().numpy():
        u, v = int(raw_u), int(raw_v)
        result.setdefault(u, set()).add(v)
        result.setdefault(v, set()).add(u)
    return result


def quantile_bins(
    values: pd.Series,
    requested: int = NUM_BINS,
    minimum: int = MIN_BINS_PER_BAND,
) -> pd.Series:
    """Rank-bin ``values`` into contiguous bins numbered from zero.

    ``minimum`` guards the marginal axes, where too few realized bins make the
    Spearman gradient meaningless and the whole dataset should fail loudly.
    Pass ``minimum=0`` for the nested bands, where a thin band is a fact about
    the network (Brightkite's coreness collapses) and must be recorded rather
    than abort the four bands that do resolve.
    """
    try:
        result = pd.qcut(values, q=requested, labels=False, duplicates="drop")
    except ValueError:
        result = pd.cut(values, bins=requested, labels=False, duplicates="drop")
    if result.isna().any():
        raise StructuralControlError("Axis binning left users unassigned")
    observed = sorted(int(value) for value in result.unique())
    remap = {old: new for new, old in enumerate(observed)}
    normalized = result.astype(int).map(remap).astype(int)
    if minimum and normalized.nunique() < minimum:
        raise StructuralControlError(
            f"Alternative axis realizes fewer than {minimum} bins"
        )
    return normalized


def nested_degree_coreness(
    *,
    sources: Sequence[int],
    features: pd.DataFrame,
    coreness: np.ndarray,
    degrees,
    recs: Mapping[str, Mapping[int, Sequence[int]]],
    seed_users: pd.DataFrame,
    train_edges: torch.Tensor,
    test_edges: torch.Tensor,
    num_nodes: int,
    dataset: str,
    seed: int,
) -> pd.DataFrame:
    """Stratified Degree EG on the degree x coreness grid, for one seed.

    Band users by degree quantile, then re-bin each band by coreness, and
    evaluate EG on every cell.  Every cell is a distinct source set, so one
    call per model covers the whole grid: the cell id is the pair (band, bin)
    flattened, and the caller recovers the two coordinates from the columns.

    A band whose coreness realizes fewer than ``MIN_BINS_PER_BAND`` bins is
    still emitted -- its rows carry ``resolved = False`` -- because dropping
    it silently would let a network with collapsed coreness (Brightkite) look
    like a network where the control simply passed.
    """
    band_series = quantile_bins(
        features.loc[sources, "degree"].reset_index(drop=True),
        requested=NESTED_DEGREE_BANDS,
        minimum=MIN_BINS_PER_BAND,
    )
    bands = {user: int(value) for user, value in zip(sources, band_series)}

    cell_of_user: dict[int, int] = {}
    cell_meta: dict[int, dict] = {}
    for band in sorted(set(bands.values())):
        members = [user for user in sources if bands[user] == band]
        band_coreness = pd.Series(
            [float(coreness[user]) for user in members], index=range(len(members))
        )
        # minimum=0: a thin band is data, not a failure -- see the docstring.
        band_bins = quantile_bins(
            band_coreness, requested=NESTED_CORENESS_BINS, minimum=0
        )
        resolved = bool(band_bins.nunique() >= MIN_BINS_PER_BAND)
        for user, coreness_bin in zip(members, band_bins):
            cell = band * NESTED_CORENESS_BINS + int(coreness_bin)
            cell_of_user[user] = cell
            cell_meta.setdefault(cell, {
                "degree_band": band,
                "coreness_bin": int(coreness_bin),
                "resolved": resolved,
                "members": [],
            })["members"].append(user)

    cell_ids = sorted(cell_meta)
    eg_results = compute_degree_equality_gain_by_source_bins_for_models(
        recs, cell_of_user, train_edges, test_edges, num_nodes, K,
        bin_ids=cell_ids,
    )

    rows = []
    for method in METHODS:
        method_users = seed_users.loc[seed_users["method"] == method]
        for cell in cell_ids:
            meta = cell_meta[cell]
            members = meta["members"]
            eg_value, gini_control, gini_pred = eg_results[method][cell]
            precision_values = method_users.loc[
                method_users["node_id"].isin(members), "precision_at_10"
            ]
            rows.append({
                "dataset": dataset,
                "training_seed": seed,
                "method": method,
                "degree_band": meta["degree_band"],
                "coreness_bin": meta["coreness_bin"],
                "resolved": meta["resolved"],
                "n_sources": len(members),
                "degree_min": float(min(degrees[user] for user in members)),
                "degree_max": float(max(degrees[user] for user in members)),
                "coreness_min": float(min(coreness[user] for user in members)),
                "coreness_max": float(max(coreness[user] for user in members)),
                "precision_at_10": float(precision_values.mean()),
                "degree_eg_at_10": float(eg_value),
                "gini_control": float(gini_control),
                "gini_pred": float(gini_pred),
            })
    return pd.DataFrame(rows)


def edgewise_preflight() -> bool:
    from experiments.run_k_sensitivity import _edgewise_penalty_preflight

    return bool(_edgewise_penalty_preflight())


def load_seed_artifact(
    upstream_root: Path,
    dataset: str,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, dict[int, list[int]]], dict]:
    directory = upstream_root / "stratified" / dataset / f"seed_{seed}"
    if not (directory / "_SUCCESS").is_file():
        raise StructuralControlError(f"Incomplete upstream shard: {directory}")
    csv_path = directory / f"stratified_cafe_{dataset}.csv"
    npz_path = directory / f"recommendations_{dataset}_seed{seed}.npz"
    metadata_path = directory / f"metadata_{dataset}.json"
    for path in (csv_path, npz_path, metadata_path):
        if not path.is_file():
            raise StructuralControlError(f"Upstream artifact missing: {path}")
    frame = pd.read_csv(csv_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or metadata.get("dataset") != dataset:
        raise StructuralControlError(f"Bad upstream metadata: {metadata_path}")
    runs = metadata.get("runs")
    if not isinstance(runs, list) or len(runs) != 1:
        raise StructuralControlError(f"Expected one run in {metadata_path}")
    run = runs[0]
    if int(run.get("training_seed", -1)) != seed or int(run.get("split_seed", -1)) != 42:
        raise StructuralControlError(f"Seed mismatch in {metadata_path}")
    if set(frame["model"].astype(str)) != set(METHODS):
        raise StructuralControlError(f"Method mismatch in {csv_path}")
    if frame.duplicated(["model", "node_id"]).any():
        raise StructuralControlError(f"Duplicate method/user rows in {csv_path}")

    recs: dict[str, dict[int, list[int]]] = {}
    with np.load(npz_path, allow_pickle=False) as payload:
        embedded = json.loads(str(payload["metadata_json"]))
        if embedded.get("split_fingerprint") != run.get("split_fingerprint"):
            raise StructuralControlError(f"NPZ fingerprint mismatch: {npz_path}")
        sources = payload["sources"].astype(np.int64)
        if len(sources) != frame["node_id"].nunique():
            raise StructuralControlError(f"NPZ source count mismatch: {npz_path}")
        for method, key in NPZ_KEYS.items():
            targets = payload[key].astype(np.int64)
            if targets.shape != (len(sources), K):
                raise StructuralControlError(f"{npz_path}: {key} shape {targets.shape}")
            recs[method] = {
                int(user): row.tolist() for user, row in zip(sources, targets)
            }
    run["artifact_sha256"] = {
        csv_path.name: sha256(csv_path),
        npz_path.name: sha256(npz_path),
        metadata_path.name: sha256(metadata_path),
    }
    return frame, recs, run


def standardize(frame: pd.DataFrame, columns: Iterable[str]) -> tuple[pd.DataFrame, dict]:
    result = frame.copy()
    scales = {}
    unique = result.drop_duplicates("node_id")
    for column in columns:
        mean = float(unique[column].mean())
        std = float(unique[column].std(ddof=0))
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise StructuralControlError(f"Cannot standardize {column}: mean={mean}, sd={std}")
        result[f"z_{column}"] = (result[column].astype(float) - mean) / std
        scales[column] = {"mean": mean, "population_sd": std}
    return result, scales


def fit_logit(
    frame: pd.DataFrame,
    predictors: Sequence[str],
    *,
    cluster_column: str = "node_id",
) -> tuple[pd.DataFrame, dict]:
    names = ("intercept", *predictors)
    x = np.column_stack(
        [np.ones(len(frame)), *(frame[name].astype(float).to_numpy() for name in predictors)]
    )
    y = frame["hit_at_10"].astype(float).to_numpy()
    clusters = frame[cluster_column].astype(int).to_numpy()
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise StructuralControlError("Non-finite logistic-regression input")
    if set(np.unique(y)) != {0.0, 1.0}:
        raise StructuralControlError("Logistic outcome lacks both classes")
    rank = int(np.linalg.matrix_rank(x))
    if rank != x.shape[1]:
        raise StructuralControlError(
            f"Rank-deficient logistic design ({rank} < {x.shape[1]})"
        )

    def objective(beta: np.ndarray) -> float:
        eta = x @ beta
        return float(np.logaddexp(0.0, eta).sum() - y @ eta)

    def gradient(beta: np.ndarray) -> np.ndarray:
        eta = x @ beta
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(eta, -40, 40)))
        return x.T @ (probabilities - y)

    fitted = optimize.minimize(
        objective,
        np.zeros(x.shape[1], dtype=float),
        jac=gradient,
        method="L-BFGS-B",
        options={"gtol": 1e-10, "ftol": 1e-13, "maxiter": 2000, "maxls": 50},
    )
    beta = fitted.x
    if (
        np.max(np.abs(gradient(beta))) / len(y)
        > SCORE_TOLERANCE_PER_OBSERVATION
    ):
        # L-BFGS-B can stall at the float64 precision floor
        # (REL_REDUCTION_OF_F_<=_FACTR*EPSMCH) with the score just above the
        # per-observation tolerance even though the optimizer reports success.
        # The logistic objective is convex and its analytic Hessian is cheap
        # here, so polish with Newton steps (quadratic convergence) instead of
        # weakening the tolerance.  Fits that already pass are untouched.
        for _ in range(10):
            eta_polish = x @ beta
            p_polish = 1.0 / (1.0 + np.exp(-np.clip(eta_polish, -40, 40)))
            w_polish = p_polish * (1.0 - p_polish)
            h_polish = x.T @ (w_polish[:, None] * x)
            step = np.linalg.solve(h_polish, x.T @ (p_polish - y))
            beta_next = beta - step
            beta = beta_next
            if np.max(np.abs(step)) < 1e-12:
                break
    eta = x @ beta
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(eta, -40, 40)))
    weights = probabilities * (1.0 - probabilities)
    hessian = x.T @ (weights[:, None] * x)
    hessian_rank = int(np.linalg.matrix_rank(hessian))
    condition = float(np.linalg.cond(hessian))
    max_gradient = float(np.max(np.abs(gradient(beta))))
    separated = bool(np.max(np.abs(beta)) > 20 or weights.min() < 1e-14)
    if hessian_rank != x.shape[1] or not np.isfinite(condition):
        raise StructuralControlError("Singular logistic information matrix")
    bread = np.linalg.inv(hessian)

    score_rows = x * (y - probabilities)[:, None]
    unique_clusters, inverse = np.unique(clusters, return_inverse=True)
    cluster_scores = np.zeros((len(unique_clusters), x.shape[1]), dtype=float)
    np.add.at(cluster_scores, inverse, score_rows)
    meat = cluster_scores.T @ cluster_scores
    n, p, groups = len(y), x.shape[1], len(unique_clusters)
    correction = (groups / (groups - 1.0)) * ((n - 1.0) / (n - p))
    covariance = correction * bread @ meat @ bread
    variances = np.diag(covariance)
    if (variances < -1e-12).any():
        raise StructuralControlError("Negative sandwich variance")
    standard_errors = np.sqrt(np.clip(variances, 0.0, None))
    z_values = np.divide(
        beta,
        standard_errors,
        out=np.full_like(beta, np.nan),
        where=standard_errors > 0,
    )
    p_values = 2.0 * stats.norm.sf(np.abs(z_values))
    rows = pd.DataFrame({
        "coefficient": names,
        "beta": beta,
        "cluster_robust_se": standard_errors,
        "ci95_low": beta - 1.959963984540054 * standard_errors,
        "ci95_high": beta + 1.959963984540054 * standard_errors,
        "wald_z": z_values,
        "p_value": p_values,
    })
    diagnostics = {
        # Convergence requires both optimizer success and a small analytic
        # score per observation.  A tolerance on the summed score cannot work:
        # its scale grows with the number of observations, so large datasets
        # fail even at an excellent score-equation solution.
        "converged": bool(fitted.success) and bool(
            max_gradient / n <= SCORE_TOLERANCE_PER_OBSERVATION
        ),
        "optimizer_success": bool(fitted.success),
        "optimizer_message": str(fitted.message),
        "iterations": int(fitted.nit),
        "n_rows": n,
        "n_source_users": groups,
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "design_rank": rank,
        "information_rank": hessian_rank,
        "information_condition_number": condition,
        "max_absolute_gradient": max_gradient,
        "separation_warning": separated,
        "log_likelihood": -objective(beta),
        "cluster_covariance": "source-user sandwich with finite-sample correction",
    }
    return rows, diagnostics


def vif_diagnostics(frame: pd.DataFrame, predictors: Sequence[str]) -> dict[str, float]:
    unique = frame.drop_duplicates("node_id")
    matrix = unique[list(predictors)].astype(float).to_numpy()
    correlation = np.corrcoef(matrix, rowvar=False)
    inverse = np.linalg.inv(correlation)
    return {name: float(inverse[i, i]) for i, name in enumerate(predictors)}


def analyze_dataset(
    dataset: str,
    data_path: Path,
    upstream_root: Path,
    output_dir: Path,
    threads: int,
) -> int:
    dataset = normalize_dataset(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "_SUCCESS").exists():
        print(f"Structural-control shard already complete: {dataset}")
        return 0
    if (output_dir / "per_user_hits.csv").exists():
        raise StructuralControlError(
            f"Incomplete output exists in {output_dir}; move it aside before retrying"
        )
    nk.setNumberOfThreads(max(1, threads))

    split_edge, data, degrees, num_nodes = load_frozen_split(dataset, data_path)
    train_edges = split_edge["train"]["edge"].cpu()
    test_edges = split_edge["test"]["edge"].cpu()
    fingerprint = split_fingerprint(split_edge, num_nodes)
    coreness = compute_coreness(train_edges, num_nodes).cpu().numpy().astype(float)
    clustering = local_clustering(train_edges, num_nodes)
    targets_by_user = test_targets(test_edges)
    if not targets_by_user:
        raise StructuralControlError("Test split has no evaluation users")

    all_user_frames = []
    axis_frames = []
    nested_frames = []
    upstream_hashes = {}
    selected_hashes = set()
    lambda_values = set()
    for seed in SEEDS:
        raw, recs, run_metadata = load_seed_artifact(
            upstream_root, dataset, seed
        )
        if run_metadata.get("split_fingerprint") != fingerprint:
            raise StructuralControlError(
                f"{dataset}/seed_{seed}: reconstructed split fingerprint mismatch"
            )
        selected_hashes.add(run_metadata.get("selected_configs_sha256"))
        lambda_values.add(float(run_metadata["cafe_in"]["lambda_fair"]))
        upstream_hashes[f"seed_{seed}"] = run_metadata["artifact_sha256"]
        sources = sorted(recs["RA"])
        if set(sources) != set(targets_by_user):
            raise StructuralControlError(
                f"{dataset}/seed_{seed}: evaluation-user set differs from test endpoints"
            )

        feature_frame = pd.DataFrame({
            "node_id": sources,
            "degree": [float(degrees[user]) for user in sources],
            "coreness": [float(coreness[user]) for user in sources],
            "clustering": [float(clustering[user]) for user in sources],
            "n_test_targets": [len(targets_by_user[user]) for user in sources],
        })
        if (feature_frame["degree"] <= 0).any():
            bad = feature_frame.loc[feature_frame["degree"] <= 0, "node_id"].tolist()[:10]
            raise StructuralControlError(
                f"log(degree) undefined for evaluation users {bad}"
            )
        feature_frame["log_degree"] = np.log(feature_frame["degree"])
        feature_frame["log1p_test_targets"] = np.log1p(
            feature_frame["n_test_targets"]
        )

        raw_maps = {
            method: group.set_index("node_id")
            for method, group in raw.groupby("model")
        }
        seed_user_rows = []
        for method in METHODS:
            for row in feature_frame.itertuples(index=False):
                recommendations = recs[method][int(row.node_id)]
                hits = len(set(recommendations[:K]) & targets_by_user[int(row.node_id)])
                precision = hits / K
                recorded = float(raw_maps[method].loc[int(row.node_id), "precision"])
                if not math.isclose(precision, recorded, rel_tol=0.0, abs_tol=1e-12):
                    raise StructuralControlError(
                        f"{dataset}/seed_{seed}/{method}/{row.node_id}: "
                        "recommendation-derived precision mismatch"
                    )
                seed_user_rows.append({
                    "dataset": dataset,
                    "training_seed": seed,
                    "split_seed": 42,
                    "split_fingerprint": fingerprint,
                    "method": method,
                    "node_id": int(row.node_id),
                    "hit_at_10": int(hits > 0),
                    "hit_count_at_10": int(hits),
                    "precision_at_10": precision,
                    "n_test_targets": int(row.n_test_targets),
                    "degree": row.degree,
                    "log_degree": row.log_degree,
                    "coreness": row.coreness,
                    "clustering": row.clustering,
                    "log1p_test_targets": row.log1p_test_targets,
                })
        seed_users = pd.DataFrame(seed_user_rows)
        all_user_frames.append(seed_users)

        unique_features = seed_users.drop_duplicates("node_id").set_index("node_id")
        raw_bin_map = (
            raw.drop_duplicates("node_id")
            .set_index("node_id")["coreness_bin"]
            .astype(int)
            .to_dict()
        )
        if set(raw_bin_map) != set(sources):
            raise StructuralControlError("Upstream coreness bins omit evaluation users")
        clustering_bins = quantile_bins(
            unique_features.loc[sources, "clustering"].reset_index(drop=True)
        )
        clustering_bin_map = {
            user: int(value) for user, value in zip(sources, clustering_bins)
        }
        degree_bins = quantile_bins(
            unique_features.loc[sources, "degree"].reset_index(drop=True)
        )
        degree_bin_map = {
            user: int(value) for user, value in zip(sources, degree_bins)
        }
        axis_maps = {
            "coreness": raw_bin_map,
            "degree": degree_bin_map,
            "clustering": clustering_bin_map,
        }
        axis_values = {
            "coreness": {user: float(coreness[user]) for user in sources},
            "degree": {user: float(degrees[user]) for user in sources},
            "clustering": {user: float(clustering[user]) for user in sources},
        }
        if set(axis_maps) != set(AXES):
            raise StructuralControlError(
                f"Axis set drifted from AXES: {sorted(axis_maps)}"
            )
        for axis, assignments in axis_maps.items():
            bin_ids = sorted(set(assignments.values()))
            eg_results = compute_degree_equality_gain_by_source_bins_for_models(
                recs,
                assignments,
                train_edges,
                test_edges,
                num_nodes,
                K,
                bin_ids=bin_ids,
            )
            for method in METHODS:
                method_users = seed_users.loc[seed_users["method"] == method]
                for bin_id in bin_ids:
                    members = [
                        user for user, assigned in assignments.items() if assigned == bin_id
                    ]
                    eg_value, gini_control, gini_pred = eg_results[method][bin_id]
                    precision_values = method_users.loc[
                        method_users["node_id"].isin(members), "precision_at_10"
                    ]
                    values = [axis_values[axis][user] for user in members]
                    axis_frames.append(pd.DataFrame([{
                        "dataset": dataset,
                        "training_seed": seed,
                        "method": method,
                        "axis": axis,
                        "axis_bin": int(bin_id),
                        "axis_min": float(min(values)),
                        "axis_max": float(max(values)),
                        "n_sources": int(len(members)),
                        "precision_at_10": float(precision_values.mean()),
                        "degree_eg_at_10": float(eg_value),
                        "gini_control": float(gini_control),
                        "gini_pred": float(gini_pred),
                    }]))

        nested_frames.append(nested_degree_coreness(
            sources=sources,
            features=unique_features,
            coreness=coreness,
            degrees=degrees,
            recs=recs,
            seed_users=seed_users,
            train_edges=train_edges,
            test_edges=test_edges,
            num_nodes=num_nodes,
            dataset=dataset,
            seed=seed,
        ))

    per_user = pd.concat(all_user_frames, ignore_index=True)
    per_user, scales = standardize(
        per_user,
        ("log_degree", "coreness", "clustering", "log1p_test_targets"),
    )
    per_bin = pd.concat(axis_frames, ignore_index=True)
    per_nested = pd.concat(nested_frames, ignore_index=True)

    coefficient_frames = []
    diagnostic_rows = []
    for method in METHODS:
        method_frame = per_user.loc[per_user["method"] == method]
        for spec, predictors in SPECS.items():
            coefficients, diagnostics = fit_logit(method_frame, predictors)
            coefficients.insert(0, "specification", spec)
            coefficients.insert(0, "method", method)
            coefficients.insert(0, "dataset", dataset)
            coefficients["fit_scope"] = "pooled_training_seeds_user_clustered"
            coefficient_frames.append(coefficients)
            diagnostic_rows.append({
                "dataset": dataset,
                "method": method,
                "specification": spec,
                "fit_scope": "pooled_training_seeds_user_clustered",
                **diagnostics,
                "vif_json": json.dumps(vif_diagnostics(method_frame, predictors), sort_keys=True),
            })
            for seed in SEEDS:
                seed_coefficients, seed_diagnostics = fit_logit(
                    method_frame.loc[method_frame["training_seed"] == seed], predictors
                )
                seed_coefficients.insert(0, "training_seed", seed)
                seed_coefficients.insert(0, "specification", spec)
                seed_coefficients.insert(0, "method", method)
                seed_coefficients.insert(0, "dataset", dataset)
                seed_coefficients["fit_scope"] = "single_training_seed"
                coefficient_frames.append(seed_coefficients)
                diagnostic_rows.append({
                    "dataset": dataset,
                    "method": method,
                    "specification": spec,
                    "fit_scope": "single_training_seed",
                    "training_seed": seed,
                    **seed_diagnostics,
                    "vif_json": json.dumps(
                        vif_diagnostics(
                            method_frame.loc[method_frame["training_seed"] == seed],
                            predictors,
                        ),
                        sort_keys=True,
                    ),
                })
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    diagnostics = pd.DataFrame(diagnostic_rows)
    if not diagnostics["converged"].all():
        bad = diagnostics.loc[~diagnostics["converged"], [
            "method", "specification", "fit_scope", "training_seed",
            "optimizer_message", "max_absolute_gradient",
        ]].to_dict("records")
        raise StructuralControlError(f"Logistic fit did not converge: {bad}")

    edgewise = edgewise_preflight()
    cafe_lambda = next(iter(lambda_values)) if len(lambda_values) == 1 else None
    if cafe_lambda is None:
        raise StructuralControlError("CAFE-In lambda differs across upstream seeds")
    if len(selected_hashes) != 1:
        raise StructuralControlError("Selected-config hash differs across upstream seeds")
    selected_hash = str(next(iter(selected_hashes)))
    known_nonedgewise_upstream = (
        selected_hash == KNOWN_NONEDGEWISE_SELECTED_CONFIG_SHA256
    )
    upstream_edgewise_provenance = bool(edgewise and not known_nonedgewise_upstream)
    execution_valid = bool(upstream_edgewise_provenance or cafe_lambda == 0.0)
    selection_valid = upstream_edgewise_provenance
    validity_rows = []
    for method in METHODS:
        validity_rows.append({
            "dataset": dataset,
            "method": method,
            "execution_valid_under_eq8": bool(method != "CAFE-In" or execution_valid),
            "selection_provenance_valid_under_eq8": bool(
                method != "CAFE-In" or selection_valid
            ),
            "note": (
                "CAFE-In selected under non-edgewise loss"
                if method == "CAFE-In" and not selection_valid
                else ""
            ),
        })
    validity = pd.DataFrame(validity_rows)

    outputs = {
        "per_user_hits.csv": per_user,
        "axis_bins_per_seed.csv": per_bin,
        "nested_degree_coreness_per_seed.csv": per_nested,
        "logit_coefficients.csv": coefficients,
        "logit_diagnostics.csv": diagnostics,
        "method_validity.csv": validity,
    }
    for name, frame in outputs.items():
        atomic_csv(output_dir / name, frame)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "profile": PROFILE,
        "status": "complete",
        "dataset": dataset,
        "training_seeds": list(SEEDS),
        "split_seed": 42,
        "split_fingerprint": fingerprint,
        "methods": list(METHODS),
        "k": K,
        "requested_bins": NUM_BINS,
        "axes": list(AXES),
        "nested_degree_bands": NESTED_DEGREE_BANDS,
        "nested_coreness_bins": NESTED_CORENESS_BINS,
        "min_bins_per_band": MIN_BINS_PER_BAND,
        "features": {
            "degree": "simple undirected training-graph degree",
            "log_degree": "natural logarithm of positive training degree",
            "coreness": "training-graph k-core number",
            "clustering": "exact training-graph local clustering coefficient",
            "outcome": "indicator of at least one held-out neighbor in top 10",
            "opportunity": "log(1 + number of held-out neighbors)",
        },
        "feature_standardization": scales,
        "inference": "pooled model-seed rows with source-user-clustered sandwich covariance",
        "conditional_inference_scope": "descriptive, conditional on the observed graph",
        "selected_configs_sha256": sorted(str(value) for value in selected_hashes),
        "eq8_edgewise_penalty_preflight": edgewise,
        "known_nonedgewise_upstream": known_nonedgewise_upstream,
        "upstream_edgewise_provenance": upstream_edgewise_provenance,
        "cafe_in_lambda_fair": cafe_lambda,
        "cafe_in_execution_valid_under_eq8": execution_valid,
        "cafe_in_selection_provenance_valid_under_eq8": selection_valid,
        "upstream_artifacts_sha256": upstream_hashes,
        "code_sha256": code_hashes(),
        "outputs_sha256": {name: sha256(output_dir / name) for name in outputs},
    }
    atomic_json(output_dir / "task_manifest.json", manifest)
    atomic_bytes(output_dir / "_SUCCESS", b"complete\n")
    print(f"Completed RQ1 structural controls for {dataset}: {output_dir}")
    return 0


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = (total - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def aggregate_axis_correlations(per_bin: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed_rows = []
    aggregate_rows = []
    for dataset in DATASETS:
        for method in METHODS:
            for axis in AXES:
                block = per_bin.loc[
                    (per_bin["dataset"] == dataset)
                    & (per_bin["method"] == method)
                    & (per_bin["axis"] == axis)
                ]
                for seed in SEEDS:
                    seed_block = block.loc[
                        block["training_seed"].astype(int) == seed
                    ].sort_values("axis_bin")
                    rho, p_value = stats.spearmanr(
                        seed_block["axis_bin"], seed_block["degree_eg_at_10"]
                    )
                    if not np.isfinite(rho):
                        raise StructuralControlError(
                            f"Undefined axis correlation: {dataset}/{method}/{axis}/{seed}"
                        )
                    per_seed_rows.append({
                        "dataset": dataset,
                        "method": method,
                        "axis": axis,
                        "training_seed": seed,
                        "realized_bins": len(seed_block),
                        "spearman_rho": float(rho),
                        "spearman_p_value": float(p_value),
                    })
                aggregated = (
                    block.groupby("axis_bin", as_index=False)["degree_eg_at_10"].mean()
                    .sort_values("axis_bin")
                )
                rho, p_value = stats.spearmanr(
                    aggregated["axis_bin"], aggregated["degree_eg_at_10"]
                )
                aggregate_rows.append({
                    "dataset": dataset,
                    "method": method,
                    "axis": axis,
                    "realized_bins": len(aggregated),
                    "spearman_rho_on_mean_eg": float(rho),
                    "spearman_p_value_on_mean_eg": float(p_value),
                })
    return pd.DataFrame(per_seed_rows), pd.DataFrame(aggregate_rows)


def aggregate_nested_correlations(
    nested: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coreness--EG Spearman within each degree band, and its per-cell summary.

    Returns (per_band, summary).  ``per_band`` carries one row per
    dataset/method/band; ``summary`` reduces each dataset/method to the range
    of band correlations, which is what Table 18(b) prints.  Bands that did
    not resolve enough coreness bins are excluded from the summary and
    counted, so a partial control can never read as a clean one.
    """
    band_rows = []
    summary_rows = []
    for dataset in DATASETS:
        for method in METHODS:
            block = nested.loc[
                (nested["dataset"] == dataset) & (nested["method"] == method)
            ]
            if block.empty:
                raise StructuralControlError(
                    f"Nested control missing cells: {dataset}/{method}"
                )
            correlations = []
            for band in sorted(block["degree_band"].unique()):
                band_block = block.loc[block["degree_band"] == band]
                # Average the seeds first, exactly as the marginal axes do.
                averaged = (
                    band_block.groupby("coreness_bin", as_index=False)
                    ["degree_eg_at_10"].mean().sort_values("coreness_bin")
                )
                resolved = bool(band_block["resolved"].all()
                                and len(averaged) >= MIN_BINS_PER_BAND)
                if resolved:
                    rho, p_value = stats.spearmanr(
                        averaged["coreness_bin"], averaged["degree_eg_at_10"]
                    )
                else:
                    rho, p_value = np.nan, np.nan
                band_rows.append({
                    "dataset": dataset,
                    "method": method,
                    "degree_band": int(band),
                    "resolved": resolved,
                    "realized_bins": int(len(averaged)),
                    "n_sources": int(band_block["n_sources"].sum() / len(SEEDS)),
                    "degree_min": float(band_block["degree_min"].min()),
                    "degree_max": float(band_block["degree_max"].max()),
                    "spearman_rho_on_mean_eg": float(rho),
                    "spearman_p_value_on_mean_eg": float(p_value),
                    "eg_span": float(averaged["degree_eg_at_10"].max()
                                     - averaged["degree_eg_at_10"].min()),
                })
                if resolved and np.isfinite(rho):
                    correlations.append(float(rho))
            if not correlations:
                raise StructuralControlError(
                    f"No degree band resolved for {dataset}/{method}"
                )
            summary_rows.append({
                "dataset": dataset,
                "method": method,
                "bands_total": int(block["degree_band"].nunique()),
                "bands_resolved": len(correlations),
                "rho_min": min(correlations),
                "rho_max": max(correlations),
                "rho_mean": float(np.mean(correlations)),
                "all_negative": bool(all(value < 0 for value in correlations)),
            })
    return pd.DataFrame(band_rows), pd.DataFrame(summary_rows)


def conditional_table(coefficients: pd.DataFrame, validity: pd.DataFrame) -> str:
    wanted = coefficients.loc[
        (coefficients["fit_scope"] == "pooled_training_seeds_user_clustered")
        & (coefficients["coefficient"] == "z_coreness")
    ].copy()
    lines = [
        "% Standardized coreness coefficient; source-user-clustered 95% Wald CI.",
        "% First line: degree+coreness. Second line: +local clustering.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\caption{Conditional association of training-graph coreness with "
        "top-10 recommendation success. Each cell gives the standardized coreness "
        "coefficient and source-user-clustered 95\\% Wald interval for "
        "$\\log d+C$ (first line) and after adding local clustering (second line).}",
        "\\label{tab:rq1-conditional-hit}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Dataset & RA & HL-GNN & Fusion & CAFE-In & CAFE-Pre \\\\",
        "\\midrule",
    ]
    display = {
        "ego-facebook": "Ego-FB", "oklahoma": "Oklahoma", "unc": "UNC",
        "pokec_n": "Pokec-n", "brightkite": "Brightkite",
    }
    for dataset in DATASETS:
        cells = []
        for method in METHODS:
            base = wanted.loc[
                (wanted["dataset"] == dataset) & (wanted["method"] == method)
                & (wanted["specification"] == "degree_coreness")
            ]
            augmented = wanted.loc[
                (wanted["dataset"] == dataset) & (wanted["method"] == method)
                & (wanted["specification"] == "degree_coreness_clustering")
            ]
            if len(base) != 1 or len(augmented) != 1:
                raise StructuralControlError(
                    f"Missing conditional coefficient: {dataset}/{method}"
                )
            base_value = base.iloc[0]
            augmented_value = augmented.iloc[0]
            valid = validity.loc[
                (validity["dataset"] == dataset) & (validity["method"] == method),
                "selection_provenance_valid_under_eq8",
            ].iloc[0]
            star = "*" if not bool(valid) else ""
            cells.append(
                "\\shortstack{"
                f"{base_value['beta']:+.2f} [{base_value['ci95_low']:+.2f}, "
                f"{base_value['ci95_high']:+.2f}]\\\\"
                f"{augmented_value['beta']:+.2f} "
                f"[{augmented_value['ci95_low']:+.2f}, "
                f"{augmented_value['ci95_high']:+.2f}]{star}"
                "}"
            )
        lines.append(f"{display[dataset]} & " + " & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule", "\\end{tabular}",
        "\\par\\footnotesize The output bundle also reports an "
        "opportunity-adjusted sensitivity including the number of held-out "
        "neighbors. *CAFE-In configuration selection predates the Eq.~(8) shape fix.",
        "\\end{table*}", "",
    ]
    return "\n".join(lines)


def table18_extension(
    axis_summary: pd.DataFrame,
    nested_summary: pd.DataFrame,
    validity: pd.DataFrame,
) -> str:
    """Columns to append to Table 18(b) -- no new table in the paper.

    Three marginal stratifiers plus the nested within-degree range.  The
    marginal degree column is expected to be negative too (degree and coreness
    are correlated); the column that answers the objection is the last one.
    """
    lines = [
        "% Columns to APPEND to Table 18(b); computed from the same",
        "% recommendation artifacts as the coreness column.",
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{Alternative-axis controls for CAFE-In stratified Degree "
        "EG@10.  The first three columns re-bin the same recommendations by a "
        "different stratifier; each entry is Spearman's $\\rho_S$ between bin "
        "rank and mean EG across the three frozen training seeds.  The last "
        "column stratifies by coreness \\emph{within} each degree quartile and "
        "reports the range of the four within-band correlations.}",
        "\\label{tab:table18-alternative-axes}",
        "\\begin{tabular}{lrrrr}", "\\toprule",
        " & \\multicolumn{3}{c}{Marginal stratifier}"
        " & Within degree band \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-5}",
        "Dataset & Coreness rank & Degree rank & Local-clustering rank"
        " & Coreness $\\rho_S$ range \\\\", "\\midrule",
    ]
    display = {
        "ego-facebook": "Ego-FB", "oklahoma": "Oklahoma", "unc": "UNC",
        "pokec_n": "Pokec-n", "brightkite": "Brightkite",
    }
    partial_bands = False
    for dataset in DATASETS:
        rows = axis_summary.loc[
            (axis_summary["dataset"] == dataset)
            & (axis_summary["method"] == "CAFE-In")
        ].set_index("axis")
        nested = nested_summary.loc[
            (nested_summary["dataset"] == dataset)
            & (nested_summary["method"] == "CAFE-In")
        ].iloc[0]
        invalid = not bool(validity.loc[
            (validity["dataset"] == dataset) & (validity["method"] == "CAFE-In"),
            "selection_provenance_valid_under_eq8",
        ].iloc[0])
        star = "*" if invalid else ""
        band_note = ""
        if int(nested["bands_resolved"]) != int(nested["bands_total"]):
            partial_bands = True
            band_note = (f"$^{{\\dagger}}$ ({int(nested['bands_resolved'])}"
                         f"/{int(nested['bands_total'])})")
        lines.append(
            f"{display[dataset]} & "
            f"{rows.loc['coreness', 'spearman_rho_on_mean_eg']:+.2f}{star} & "
            f"{rows.loc['degree', 'spearman_rho_on_mean_eg']:+.2f}{star} & "
            f"{rows.loc['clustering', 'spearman_rho_on_mean_eg']:+.2f}{star} & "
            f"[{nested['rho_min']:+.2f}, {nested['rho_max']:+.2f}]"
            f"{band_note} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    if partial_bands:
        lines.append(
            "\\par\\footnotesize $^{\\dagger}$Degree bands whose coreness "
            f"realizes fewer than {MIN_BINS_PER_BAND} bins are excluded; the "
            "fraction gives resolved bands out of the bands the degree "
            f"quantiles actually realize (at most {NESTED_DEGREE_BANDS})."
        )
    lines += [
        "\\par\\footnotesize *Upstream CAFE-In configuration selection predates "
        "the Eq.~(8) shape fix; recompute from the corrected campaign before use.",
        "\\end{table}", "",
    ]
    return "\n".join(lines)


def finalize(campaign: Path, if_complete: bool, rebuild: bool) -> int:
    campaign.mkdir(parents=True, exist_ok=True)
    with (campaign / ".finalize.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        missing = []
        manifests = {}
        code_documents = set()
        frames: dict[str, list[pd.DataFrame]] = {
            "per_user_hits.csv": [],
            "axis_bins_per_seed.csv": [],
            "nested_degree_coreness_per_seed.csv": [],
            "logit_coefficients.csv": [],
            "logit_diagnostics.csv": [],
            "method_validity.csv": [],
        }
        for dataset in DATASETS:
            directory = campaign / "shards" / dataset
            manifest_path = directory / "task_manifest.json"
            if not (directory / "_SUCCESS").is_file() or not manifest_path.is_file():
                missing.append(dataset)
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("profile") != PROFILE or manifest.get("dataset") != dataset:
                raise StructuralControlError(f"{manifest_path}: profile/dataset mismatch")
            code_documents.add(json.dumps(manifest.get("code_sha256"), sort_keys=True))
            for name in frames:
                path = directory / name
                if manifest["outputs_sha256"].get(name) != sha256(path):
                    raise StructuralControlError(f"{path}: hash mismatch")
                frames[name].append(pd.read_csv(path))
            manifests[dataset] = sha256(manifest_path)
        if missing:
            message = f"Missing RQ1 structural-control datasets: {missing}"
            if if_complete:
                print(message)
                return 0
            raise StructuralControlError(message)
        if len(code_documents) != 1:
            raise StructuralControlError("Dataset shards were analyzed with different code")

        output_dir = campaign / "finalized"
        if (output_dir / "_COMPLETE").exists() and not rebuild:
            print(f"Already finalized: {output_dir}")
            return 0
        combined = {name: pd.concat(parts, ignore_index=True) for name, parts in frames.items()}
        coefficients = combined["logit_coefficients.csv"]
        validity = combined["method_validity.csv"]

        primary = coefficients.loc[
            (coefficients["fit_scope"] == "pooled_training_seeds_user_clustered")
            & (coefficients["coefficient"] == "z_coreness")
        ].copy()
        primary["holm_family"] = primary["specification"]
        primary["p_value_holm_25"] = np.nan
        for spec, indices in primary.groupby("specification").groups.items():
            primary.loc[indices, "p_value_holm_25"] = holm_adjust(
                primary.loc[indices, "p_value"].to_numpy()
            )

        per_seed_axis, aggregate_axis = aggregate_axis_correlations(
            combined["axis_bins_per_seed.csv"]
        )
        per_band_nested, nested_summary = aggregate_nested_correlations(
            combined["nested_degree_coreness_per_seed.csv"]
        )
        table_conditional = conditional_table(coefficients, validity)
        table_axis = table18_extension(aggregate_axis, nested_summary, validity)

        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "rq1_per_user_hits.csv": combined["per_user_hits.csv"],
            "rq1_axis_bins_per_seed.csv": combined["axis_bins_per_seed.csv"],
            "rq1_logit_coefficients_all.csv": coefficients,
            "rq1_logit_diagnostics_all.csv": combined["logit_diagnostics.csv"],
            "rq1_coreness_coefficients_holm.csv": primary,
            "rq1_method_validity.csv": validity,
            "rq1_axis_correlations_per_seed.csv": per_seed_axis,
            "rq1_axis_correlations_on_mean_eg.csv": aggregate_axis,
            "rq1_nested_cells_per_seed.csv":
                combined["nested_degree_coreness_per_seed.csv"],
            "rq1_nested_within_degree_band.csv": per_band_nested,
            "rq1_nested_within_degree_summary.csv": nested_summary,
        }
        for name, frame in outputs.items():
            atomic_csv(output_dir / name, frame)
        atomic_bytes(
            output_dir / "rq1_conditional_hit_table.tex", table_conditional.encode("utf-8")
        )
        atomic_bytes(
            output_dir / "table18_alternative_axes_extension.tex",
            table_axis.encode("utf-8"),
        )

        exact = {
            "inference_scope": "descriptive and conditional on each observed graph",
            "specifications": {},
        }
        for spec in ("degree_coreness", "degree_coreness_clustering"):
            block = primary.loc[primary["specification"] == spec]
            exact["specifications"][spec] = {
                "positive_coreness_beta_cells": int((block["beta"] > 0).sum()),
                "ci_excludes_zero_positive_cells": int((block["ci95_low"] > 0).sum()),
                "holm_positive_cells_0_05": int(
                    ((block["beta"] > 0) & (block["p_value_holm_25"] < 0.05)).sum()
                ),
                "total_cells": int(len(block)),
            }
        # The degree control's answer, stated as a count rather than left for
        # the reader to reconstruct from the table.
        cafe_in_bands = per_band_nested.loc[
            (per_band_nested["method"] == "CAFE-In") & per_band_nested["resolved"]
        ]
        exceptions = cafe_in_bands.loc[cafe_in_bands["spearman_rho_on_mean_eg"] >= 0]
        exact["within_degree_band_coreness_gradient"] = {
            "resolved_bands": int(len(cafe_in_bands)),
            "negative_rho_bands": int((cafe_in_bands["spearman_rho_on_mean_eg"] < 0).sum()),
            # "all resolved bands", not "all bands": a dataset whose degree
            # quantiles collapse (Brightkite) can qualify on a subset.
            "datasets_all_resolved_bands_negative": sorted(
                nested_summary.loc[
                    (nested_summary["method"] == "CAFE-In")
                    & nested_summary["all_negative"], "dataset"
                ].tolist()
            ),
            "datasets_with_unresolved_bands": sorted(
                nested_summary.loc[
                    (nested_summary["method"] == "CAFE-In")
                    & (nested_summary["bands_resolved"]
                       < nested_summary["bands_total"]), "dataset"
                ].tolist()
            ),
            "non_negative_bands": [
                {
                    "dataset": row.dataset,
                    "degree_band": int(row.degree_band),
                    "degree_range": [row.degree_min, row.degree_max],
                    "realized_bins": int(row.realized_bins),
                    "rho": float(row.spearman_rho_on_mean_eg),
                    "p_value": float(row.spearman_p_value_on_mean_eg),
                }
                for row in exceptions.itertuples(index=False)
            ],
            "bands_p_below_0_05": int(
                (cafe_in_bands["spearman_p_value_on_mean_eg"] < 0.05).sum()
            ),
            "degree_bands_requested": NESTED_DEGREE_BANDS,
            "coreness_bins_per_band_requested": NESTED_CORENESS_BINS,
        }
        atomic_json(output_dir / "rq1_structural_control_counts.json", exact)

        output_names = [
            *outputs,
            "rq1_conditional_hit_table.tex",
            "table18_alternative_axes_extension.tex",
            "rq1_structural_control_counts.json",
        ]
        integrity = {
            "schema_version": SCHEMA_VERSION,
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
            "warning": (
                "Current upstream CAFE-In configurations were selected before "
                "the Eq. (8) edgewise shape fix; invalid cells are marked."
            ),
        }
        atomic_json(output_dir / "rq1_structural_controls_integrity.json", integrity)
        atomic_bytes(output_dir / "_COMPLETE", b"complete\n")
        print(f"Finalized RQ1 structural controls: {output_dir}")
        return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--dataset", required=True)
    run.add_argument("--data-path", required=True)
    run.add_argument("--upstream-root", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--threads", type=int, default=8)
    finish = commands.add_parser("finalize")
    finish.add_argument("--campaign-dir", required=True)
    finish.add_argument("--if-complete", action="store_true")
    finish.add_argument("--rebuild", action="store_true")
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "run":
            return analyze_dataset(
                args.dataset,
                Path(args.data_path).resolve(),
                Path(args.upstream_root).resolve(),
                Path(args.output_dir).resolve(),
                args.threads,
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
