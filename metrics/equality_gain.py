"""
Equality Gain Metrics

Measures fairness improvement by comparing Gini coefficients of 
predicted vs control (observed) graphs.
"""

import torch
from typing import Dict, List, Tuple

from metrics.graph_utils import (
    build_control_edge_set,
    build_predicted_edge_set,
    degree_from_simple_undirected_edges,
)


SIMPLE_UNDIRECTED = 'simple_undirected'


def _require_simple_undirected(graph_semantics: str) -> None:
    """Reject exposure-oriented directed modes at the structural-EG API."""
    if graph_semantics != SIMPLE_UNDIRECTED:
        raise ValueError(
            "Degree Equality Gain is defined on a simple undirected graph; "
            f"got graph_semantics={graph_semantics!r}. Directed indegree/"
            "outdegree belongs to the separate exposure metrics."
        )


def compute_gini_coefficient(degrees: torch.Tensor) -> float:
    """
    Compute Gini coefficient for a degree distribution.
    
    Args:
        degrees: Node degrees [num_nodes]
    
    Returns:
        float: Gini coefficient in [0, 1]
    """
    # Filter out zero degrees for meaningful Gini
    nonzero_deg = degrees[degrees > 0]
    
    if nonzero_deg.numel() == 0:
        return 0.0
    
    # Sort degrees
    x_sorted, _ = torch.sort(nonzero_deg.view(-1))
    n = x_sorted.numel()
    s = x_sorted.sum()
    
    if s == 0:
        return 0.0
    
    # Compute Gini using formula: G = 1 + (1/n) - (2/n*s) * sum((n+1-i) * x_i)
    idx = torch.arange(1, n + 1, dtype=x_sorted.dtype, device=x_sorted.device)
    gini = 1.0 + (1.0 / n) - 2.0 * (torch.sum((n + 1 - idx) * x_sorted) / (n * s))
    
    return float(gini)


def build_predicted_degree_distribution(
    train_edges: torch.Tensor,
    recs_by_u: Dict[int, List[int]],
    num_nodes: int,
    k: int,
    graph_semantics: str = SIMPLE_UNDIRECTED,
) -> torch.Tensor:
    """
    Build degree on the simple undirected graph ``E_train ∪ Rec_K``.

    Structural Degree EG is distinct from the directed target-exposure
    metrics. Passing a directed mode is rejected rather than silently
    reinterpreted.
    """
    _require_simple_undirected(graph_semantics)
    predicted_edges = build_predicted_edge_set(
        train_edges, recs_by_u, num_nodes, k
    )
    return degree_from_simple_undirected_edges(predicted_edges, num_nodes)


def build_control_degree_distribution(
    train_edges: torch.Tensor,
    test_edges: torch.Tensor,
    num_nodes: int,
    graph_semantics: str = SIMPLE_UNDIRECTED,
) -> torch.Tensor:
    """
    Build degree on the simple undirected graph ``E_train ∪ E_test``.

    See :func:`build_predicted_degree_distribution`.
    """
    _require_simple_undirected(graph_semantics)
    control_edges = build_control_edge_set(train_edges, test_edges, num_nodes)
    return degree_from_simple_undirected_edges(control_edges, num_nodes)


def compute_equality_gain(
    recs_by_u: Dict[int, List[int]],
    train_edges: torch.Tensor,
    test_edges: torch.Tensor,
    num_nodes: int,
    k: int,
    graph_semantics: str = SIMPLE_UNDIRECTED,
) -> Tuple[float, float, float]:
    """
    Compute Equality Gain metric.
    
    Equality Gain = (Gini_control - Gini_pred) / Gini_control * 100
    
    Args:
        recs_by_u: Recommendations {node_id -> list}
        train_edges: Training edges
        test_edges: Test edges
        num_nodes: Total number of nodes
        k: Number of recommendations to use
        graph_semantics: Must be ``"simple_undirected"``. Directed modes are
            intentionally rejected because they describe exposure, not the
            post-recommendation graph in the paper.
    
    Returns:
        (equality_gain, gini_control, gini_pred)
    """
    _require_simple_undirected(graph_semantics)
    # Build control distribution (train + observed test)
    deg_control = build_control_degree_distribution(
        train_edges, test_edges, num_nodes, graph_semantics
    )
    gini_control = compute_gini_coefficient(deg_control)
    
    # Build predicted distribution (train + recommendations)
    deg_pred = build_predicted_degree_distribution(
        train_edges, recs_by_u, num_nodes, k, graph_semantics
    )
    gini_pred = compute_gini_coefficient(deg_pred)
    
    # Compute equality gain
    if gini_control == 0:
        equality_gain = 0.0
    else:
        equality_gain = ((gini_control - gini_pred) / gini_control) * 100.0
    
    return equality_gain, gini_control, gini_pred


def compute_equality_gain_metrics(
    recs_by_u: Dict[int, List[int]],
    train_edges: torch.Tensor,
    test_edges: torch.Tensor,
    num_nodes: int,
    k_values: List[int] = None,
    graph_semantics: str = SIMPLE_UNDIRECTED,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute Equality Gain for multiple K values.
    
    Args:
        recs_by_u: Recommendations
        train_edges: Training edges
        test_edges: Test edges
        num_nodes: Total number of nodes
        k_values: List of K values
        graph_semantics: Must be ``"simple_undirected"``.
    
    Returns:
        dict: {f'Equality_Gain@{k}': (0.0, value)}
    """
    if k_values is None:
        k_values = [10, 20, 50, 100]
    _require_simple_undirected(graph_semantics)
    
    results = {}
    
    print("\n" + "="*80)
    print("COMPUTING EQUALITY GAIN METRICS")
    print("="*80)
    
    # Cache control distribution (same for all k values)
    deg_control = build_control_degree_distribution(
        train_edges, test_edges, num_nodes
    )
    gini_ctrl = compute_gini_coefficient(deg_control)

    for k in k_values:
        deg_pred = build_predicted_degree_distribution(
            train_edges, recs_by_u, num_nodes, k
        )
        gini_pred = compute_gini_coefficient(deg_pred)

        if gini_ctrl == 0:
            eq_gain = 0.0
        else:
            eq_gain = ((gini_ctrl - gini_pred) / gini_ctrl) * 100.0

        results[f'Equality_Gain@{k}'] = (0.0, eq_gain)

        print(f"K={k:>3}: Equality Gain = {eq_gain:>7.2f}%")
        print(f"       Gini_control = {gini_ctrl:.4f}, Gini_pred = {gini_pred:.4f}")
    
    print("="*80)
    
    return results
