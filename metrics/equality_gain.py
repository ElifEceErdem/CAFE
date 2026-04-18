"""
Equality Gain Metrics

Measures fairness improvement by comparing Gini coefficients of 
predicted vs control (observed) graphs.
"""

import torch
from typing import Dict, List, Tuple


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


def _vectorized_degree_count(edges: torch.Tensor, num_nodes: int, mode: str = 'indegree') -> torch.Tensor:
    """Vectorized degree counting from edge tensor."""
    edges_cpu = edges.cpu()
    degrees = torch.zeros(num_nodes, dtype=torch.float32)

    if mode in ['indegree', 'total']:
        dst = edges_cpu[:, 1].long()
        valid = (dst >= 0) & (dst < num_nodes)
        degrees.scatter_add_(0, dst[valid], torch.ones(valid.sum(), dtype=torch.float32))

    if mode in ['outdegree', 'total']:
        src = edges_cpu[:, 0].long()
        valid = (src >= 0) & (src < num_nodes)
        degrees.scatter_add_(0, src[valid], torch.ones(valid.sum(), dtype=torch.float32))

    return degrees


def build_predicted_degree_distribution(
    train_edges: torch.Tensor,
    recs_by_u: Dict[int, List[int]],
    num_nodes: int,
    k: int,
    mode: str = 'indegree'
) -> torch.Tensor:
    """
    Build degree distribution for predicted graph (training + recommendations).
    Vectorized implementation.
    """
    # Training edge degrees (vectorized)
    degrees = _vectorized_degree_count(train_edges, num_nodes, mode)

    # Recommendation edges -- build tensor then vectorize
    rec_src, rec_dst = [], []
    for u, vs in recs_by_u.items():
        topk = vs[:k]
        rec_src.extend([u] * len(topk))
        rec_dst.extend(topk)

    if rec_src:
        rec_edges = torch.tensor([rec_src, rec_dst], dtype=torch.long).t()
        degrees += _vectorized_degree_count(rec_edges, num_nodes, mode)

    return degrees


def build_control_degree_distribution(
    train_edges: torch.Tensor,
    test_edges: torch.Tensor,
    num_nodes: int,
    mode: str = 'indegree'
) -> torch.Tensor:
    """
    Build degree distribution for control graph (training + observed test edges).
    Vectorized implementation.
    """
    all_edges = torch.cat([train_edges.cpu(), test_edges.cpu()], dim=0)
    return _vectorized_degree_count(all_edges, num_nodes, mode)


def compute_equality_gain(
    recs_by_u: Dict[int, List[int]],
    train_edges: torch.Tensor,
    test_edges: torch.Tensor,
    num_nodes: int,
    k: int,
    mode: str = 'indegree'
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
        mode: Degree counting mode
    
    Returns:
        (equality_gain, gini_control, gini_pred)
    """
    # Build control distribution (train + observed test)
    deg_control = build_control_degree_distribution(
        train_edges, test_edges, num_nodes, mode
    )
    gini_control = compute_gini_coefficient(deg_control)
    
    # Build predicted distribution (train + recommendations)
    deg_pred = build_predicted_degree_distribution(
        train_edges, recs_by_u, num_nodes, k, mode
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
    fairness_mode: str = 'indegree'
) -> Dict[str, Tuple[float, float]]:
    """
    Compute Equality Gain for multiple K values.
    
    Args:
        recs_by_u: Recommendations
        train_edges: Training edges
        test_edges: Test edges
        num_nodes: Total number of nodes
        k_values: List of K values
        fairness_mode: Degree counting mode
    
    Returns:
        dict: {f'Equality_Gain@{k}': (0.0, value)}
    """
    if k_values is None:
        k_values = [10, 20, 50, 100]
    
    results = {}
    
    print("\n" + "="*80)
    print("COMPUTING EQUALITY GAIN METRICS")
    print("="*80)
    
    # Cache control distribution (same for all k values)
    deg_control = build_control_degree_distribution(
        train_edges, test_edges, num_nodes, fairness_mode
    )
    gini_ctrl = compute_gini_coefficient(deg_control)

    for k in k_values:
        deg_pred = build_predicted_degree_distribution(
            train_edges, recs_by_u, num_nodes, k, fairness_mode
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