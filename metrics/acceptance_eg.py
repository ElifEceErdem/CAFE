"""
Acceptance-robust Equality Gain variants (experiment (h), RQ8).

Implements the two robustness variants specified in the ToRS paper
(§3.4, "Acceptance-robust EG variants"):

  EG-full : the standard EG (every recommended link materializes) —
            computed via the SAME code paths as the main pipeline
            (metrics/equality_gain.py, metrics/centrality_equality_gain.py)
            so the 'full' column reproduces the corrected main metric.
  EG-TP   : oracle acceptance — only recommended edges that appear in
            E_test are added to the predicted graph.
  EG-PW   : precision-weighted acceptance — each recommended edge at
            rank r carries weight rho(r), the empirical fraction of
            rank-r recommendations (across users) appearing in E_test.
            Degree becomes weighted (expected) degree; PageRank runs
            on the weighted adjacency. Closeness is reported under
            full and TP only (it requires an unweighted distance
            structure).

Graph conventions:
  - full and TP variants use the same simple undirected set-union graph
    constructor as the main EG pipeline
  - reciprocal PW recommendation events with probabilities p_i collapse
    to one expected edge of weight 1 - prod_i(1 - p_i)
  - PageRank uses the resulting symmetric weighted adjacency, damping 0.85,
    and power iteration with per-step normalization
  - Gini over the nonzero support (compute_gini_coefficient)
  - control graph = train ∪ test, identical for all variants
"""

from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from metrics.equality_gain import (
    compute_gini_coefficient,
    build_control_degree_distribution,
    build_predicted_degree_distribution,
)
from metrics.centrality_equality_gain import (
    build_control_centrality,
    build_predicted_centrality,
)
from metrics.graph_utils import (
    canonicalize_undirected_edges,
)


# ---------------------------------------------------------------------------
# Acceptance models
# ---------------------------------------------------------------------------

def build_test_pair_set(test_edges: torch.Tensor) -> set:
    """Undirected set of test pairs {(u,v), (v,u)}."""
    pairs = set()
    for u, v in test_edges.cpu().tolist():
        pairs.add((u, v))
        pairs.add((v, u))
    return pairs


def compute_rank_precision(recs_by_u: Dict[int, List[int]],
                           test_pairs: set, k: int) -> np.ndarray:
    """
    rho(r): fraction of rank-r recommendations (r = 0..k-1, across all
    users with at least r+1 recommendations) appearing in E_test.
    """
    hits = np.zeros(k, dtype=np.float64)
    counts = np.zeros(k, dtype=np.float64)
    for u, vs in recs_by_u.items():
        for r, v in enumerate(vs[:k]):
            counts[r] += 1
            if (u, v) in test_pairs:
                hits[r] += 1
    rho = np.divide(hits, counts, out=np.zeros_like(hits), where=counts > 0)
    return rho


def filter_recs_tp(recs_by_u: Dict[int, List[int]],
                   test_pairs: set, k: int) -> Dict[int, List[int]]:
    """Keep only recommendations that are true positives (in E_test)."""
    out = {}
    for u, vs in recs_by_u.items():
        kept = [v for v in vs[:k] if (u, v) in test_pairs]
        if kept:
            out[u] = kept
    return out


# ---------------------------------------------------------------------------
# Weighted (precision-weighted) centralities
# ---------------------------------------------------------------------------

def weighted_predicted_degree(train_edges: torch.Tensor,
                              recs_by_u: Dict[int, List[int]],
                              num_nodes: int, k: int,
                              rho: np.ndarray,
                              mode: str = 'undirected') -> torch.Tensor:
    """
    Expected degree under independent rank-weighted acceptance.

    The compatibility ``mode`` argument is ignored: structural EG is
    evaluated on the paper's simple undirected graph.
    """
    edges, weights = _build_expected_predicted_graph(
        train_edges, recs_by_u, num_nodes, k, rho
    )
    degrees = torch.zeros(num_nodes, dtype=torch.float32)
    if edges.numel() > 0:
        weights_f = weights.to(dtype=torch.float32)
        degrees.scatter_add_(0, edges[:, 0], weights_f)
        degrees.scatter_add_(0, edges[:, 1], weights_f)
    return degrees


def _symmetric_binary_adj(edges: torch.Tensor, num_nodes: int) -> sp.csr_matrix:
    e = canonicalize_undirected_edges(edges, num_nodes).numpy()
    if len(e) == 0:
        return sp.csr_matrix((num_nodes, num_nodes), dtype=np.float64)
    row = np.concatenate([e[:, 0], e[:, 1]])
    col = np.concatenate([e[:, 1], e[:, 0]])
    adj = sp.csr_matrix((np.ones_like(row, dtype=np.float64), (row, col)),
                        shape=(num_nodes, num_nodes))
    adj.data = np.ones_like(adj.data)  # binarize duplicates
    return adj


def _build_expected_predicted_graph(
    train_edges: torch.Tensor,
    recs_by_u: Dict[int, List[int]],
    num_nodes: int,
    k: int,
    rho: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return canonical edges and expected weights for the PW graph.

    Training edges have weight one.  For a recommended undirected pair with
    directional events at ranks carrying probabilities ``p_i``, the edge
    materializes with probability ``1 - prod_i(1 - p_i)``.  Recommendation
    events overlapping training edges cannot raise their weight above one.
    """
    rho = np.asarray(rho, dtype=np.float64)
    if len(rho) < k:
        raise ValueError(f"rho has length {len(rho)} but k={k}")
    if np.any(~np.isfinite(rho[:k])) or np.any((rho[:k] < 0) | (rho[:k] > 1)):
        raise ValueError("Acceptance probabilities must be finite and in [0, 1]")

    train_simple = canonicalize_undirected_edges(train_edges, num_nodes)
    train_pairs = {tuple(edge) for edge in train_simple.tolist()}
    survival = {}

    for raw_u, values in recs_by_u.items():
        u = int(raw_u)
        if not 0 <= u < num_nodes:
            raise ValueError(f"Recommendation source outside [0, {num_nodes}): {u}")
        for rank, raw_v in enumerate(values[:k]):
            v = int(raw_v)
            if not 0 <= v < num_nodes:
                raise ValueError(
                    f"Recommendation target outside [0, {num_nodes}): {v}"
                )
            if u == v:
                continue
            pair = (min(u, v), max(u, v))
            if pair in train_pairs:
                continue
            survival[pair] = survival.get(pair, 1.0) * (1.0 - float(rho[rank]))

    rec_pairs = sorted(pair for pair, q in survival.items() if q < 1.0)
    if rec_pairs:
        rec_edges = torch.tensor(rec_pairs, dtype=torch.long)
        rec_weights = torch.tensor(
            [1.0 - survival[pair] for pair in rec_pairs], dtype=torch.float64
        )
        edges = torch.cat([train_simple, rec_edges], dim=0)
        weights = torch.cat([
            torch.ones(len(train_simple), dtype=torch.float64),
            rec_weights,
        ])
    else:
        edges = train_simple
        weights = torch.ones(len(train_simple), dtype=torch.float64)
    return edges, weights


def weighted_predicted_pagerank(train_edges: torch.Tensor,
                                recs_by_u: Dict[int, List[int]],
                                num_nodes: int, k: int,
                                rho: np.ndarray,
                                alpha: float = 0.85,
                                max_iter: int = 100,
                                tol: float = 1e-6) -> torch.Tensor:
    """
    PageRank on the expected simple undirected PW graph.
    """
    edges, weights = _build_expected_predicted_graph(
        train_edges, recs_by_u, num_nodes, k, rho
    )
    if edges.numel() == 0:
        adj = sp.csr_matrix((num_nodes, num_nodes), dtype=np.float64)
    else:
        e = edges.numpy()
        w = weights.numpy()
        rows = np.concatenate([e[:, 0], e[:, 1]])
        cols = np.concatenate([e[:, 1], e[:, 0]])
        vals = np.concatenate([w, w])
        adj = sp.csr_matrix(
            (vals, (rows, cols)), shape=(num_nodes, num_nodes)
        )

    out_deg = np.array(adj.sum(axis=1)).flatten()
    out_deg[out_deg == 0] = 1.0
    transition = adj.multiply((1.0 / out_deg)[:, np.newaxis]).T.tocsr()

    x = np.ones(num_nodes, dtype=np.float64) / num_nodes
    teleport = (1.0 - alpha) / num_nodes
    for _ in range(max_iter):
        x_new = alpha * transition.dot(x) + teleport
        x_new = x_new / x_new.sum()
        diff = np.linalg.norm(x_new - x)
        x = x_new
        if diff < tol:
            break
    return torch.from_numpy(x.astype(np.float32))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _eg(gini_control: float, gini_pred: float) -> float:
    if gini_control == 0:
        return 0.0
    return (gini_control - gini_pred) / gini_control * 100.0


def compute_acceptance_robust_eg(
    recs_by_u: Dict[int, List[int]],
    train_edges: torch.Tensor,
    test_edges: torch.Tensor,
    num_nodes: int,
    k: int = 10,
    use_sampling_closeness: bool = True,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Compute Degree/PageRank/Closeness EG under the full, TP, and PW
    acceptance variants.

    Returns:
        results: dict with keys f'{Metric}_EG@{k}_{variant}' where
                 Metric in {Degree, Pagerank, Closeness} and variant in
                 {full, tp, pw} (Closeness: full and tp only), plus
                 'n_recs@{k}' and 'n_tp@{k}'.
        rho:     per-rank precision array [k]
    """
    train_cpu = train_edges.cpu()
    test_cpu = test_edges.cpu()
    test_pairs = build_test_pair_set(test_cpu)

    rho = compute_rank_precision(recs_by_u, test_pairs, k)
    recs_tp = filter_recs_tp(recs_by_u, test_pairs, k)
    n_recs = sum(len(vs[:k]) for vs in recs_by_u.values())
    n_tp = sum(len(vs) for vs in recs_tp.values())

    results = {f'n_recs@{k}': float(n_recs), f'n_tp@{k}': float(n_tp)}
    cent_kwargs_closeness = {'use_sampling': use_sampling_closeness,
                             'epsilon': 0.1}

    # ---- Degree on the same simple undirected graph as the main pipeline ----
    deg_ctrl = build_control_degree_distribution(
        train_cpu, test_cpu, num_nodes)
    g_ctrl_deg = compute_gini_coefficient(deg_ctrl)

    deg_full = build_predicted_degree_distribution(
        train_cpu, recs_by_u, num_nodes, k)
    deg_tp = build_predicted_degree_distribution(
        train_cpu, recs_tp, num_nodes, k)
    deg_pw = weighted_predicted_degree(
        train_cpu, recs_by_u, num_nodes, k, rho)

    results[f'Degree_EG@{k}_full'] = _eg(g_ctrl_deg, compute_gini_coefficient(deg_full))
    results[f'Degree_EG@{k}_tp'] = _eg(g_ctrl_deg, compute_gini_coefficient(deg_tp))
    results[f'Degree_EG@{k}_pw'] = _eg(g_ctrl_deg, compute_gini_coefficient(deg_pw))

    # ---- PageRank ----
    pr_ctrl = build_control_centrality(train_cpu, test_cpu, num_nodes, 'pagerank')
    g_ctrl_pr = compute_gini_coefficient(pr_ctrl)

    pr_full = build_predicted_centrality(train_cpu, recs_by_u, num_nodes, k, 'pagerank')
    pr_tp = build_predicted_centrality(train_cpu, recs_tp, num_nodes, k, 'pagerank')
    pr_pw = weighted_predicted_pagerank(train_cpu, recs_by_u, num_nodes, k, rho)

    results[f'Pagerank_EG@{k}_full'] = _eg(g_ctrl_pr, compute_gini_coefficient(pr_full))
    results[f'Pagerank_EG@{k}_tp'] = _eg(g_ctrl_pr, compute_gini_coefficient(pr_tp))
    results[f'Pagerank_EG@{k}_pw'] = _eg(g_ctrl_pr, compute_gini_coefficient(pr_pw))

    # ---- Closeness (full + TP only; PW needs a distance structure) ----
    cl_ctrl = build_control_centrality(
        train_cpu, test_cpu, num_nodes, 'closeness', **cent_kwargs_closeness)
    g_ctrl_cl = compute_gini_coefficient(cl_ctrl)

    cl_full = build_predicted_centrality(
        train_cpu, recs_by_u, num_nodes, k, 'closeness', **cent_kwargs_closeness)
    cl_tp = build_predicted_centrality(
        train_cpu, recs_tp, num_nodes, k, 'closeness', **cent_kwargs_closeness)

    results[f'Closeness_EG@{k}_full'] = _eg(g_ctrl_cl, compute_gini_coefficient(cl_full))
    results[f'Closeness_EG@{k}_tp'] = _eg(g_ctrl_cl, compute_gini_coefficient(cl_tp))

    return results, rho
