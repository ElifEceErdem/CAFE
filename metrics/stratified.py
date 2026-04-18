"""
Stratified Metrics by Degree

Computes precision and equality gain separately for low-degree and high-degree nodes.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from metrics.equality_gain import (
    compute_gini_coefficient,
    build_predicted_degree_distribution,
    build_control_degree_distribution
)

def compute_stratified_equality_gain_by_source(
    recs_by_u: Dict[int, List[int]],
    train_edges: torch.Tensor,
    test_edges: torch.Tensor,
    low_degree_nodes: List[int],
    high_degree_nodes: List[int],
    num_nodes: int,
    k: int,
    mode: str = 'indegree'
) -> Tuple[float, float]:
    """
    Compute Equality Gain for recommendations made BY low/high-degree SOURCE nodes.
    
    Question: "When low-degree sources make recommendations, do they create
               more equality in the target distribution compared to high-degree sources?"
    """
    
    # === LOW-DEGREE SOURCES ===
    low_set = set(low_degree_nodes)
    
    # Build target degree distributions for edges FROM low-degree sources
    deg_ctrl_targets = torch.zeros(num_nodes, dtype=torch.float32)
    deg_pred_targets = torch.zeros(num_nodes, dtype=torch.float32)
    
    # Control: count targets of test edges FROM low-degree sources
    for u, v in test_edges.tolist():
        if u in low_set:  # u is a low-degree SOURCE
            if 0 <= v < num_nodes:
                deg_ctrl_targets[v] += 1
    
    # Predicted: count targets of recommendations FROM low-degree sources
    for u, vs in recs_by_u.items():
        if u in low_set:  # u is a low-degree SOURCE
            for v in vs[:k]:
                if 0 <= v < num_nodes:
                    deg_pred_targets[v] += 1
    
    # Compute Gini on target distributions
    gini_ctrl_low = compute_gini_coefficient(deg_ctrl_targets)
    gini_pred_low = compute_gini_coefficient(deg_pred_targets)
    
    eq_gain_low = ((gini_ctrl_low - gini_pred_low) / gini_ctrl_low * 100.0 
                   if gini_ctrl_low > 0 else 0.0)
    
    # Debug
    print(f"\n[DEBUG] Low-degree SOURCES ({len(low_degree_nodes)} nodes):")
    print(f"  Target dist control: sum={deg_ctrl_targets.sum():.0f}, Gini={gini_ctrl_low:.4f}")
    print(f"  Target dist predicted: sum={deg_pred_targets.sum():.0f}, Gini={gini_pred_low:.4f}")
    print(f"  -> Equality Gain = {eq_gain_low:.2f}%")
    
    # === HIGH-DEGREE SOURCES ===
    high_set = set(high_degree_nodes)
    
    deg_ctrl_targets_high = torch.zeros(num_nodes, dtype=torch.float32)
    deg_pred_targets_high = torch.zeros(num_nodes, dtype=torch.float32)
    
    # Control: count targets of test edges FROM high-degree sources
    for u, v in test_edges.tolist():
        if u in high_set:  # u is a high-degree SOURCE
            if 0 <= v < num_nodes:
                deg_ctrl_targets_high[v] += 1
    
    # Predicted: count targets of recommendations FROM high-degree sources
    for u, vs in recs_by_u.items():
        if u in high_set:  # u is a high-degree SOURCE
            for v in vs[:k]:
                if 0 <= v < num_nodes:
                    deg_pred_targets_high[v] += 1
    
    gini_ctrl_high = compute_gini_coefficient(deg_ctrl_targets_high)
    gini_pred_high = compute_gini_coefficient(deg_pred_targets_high)
    
    eq_gain_high = ((gini_ctrl_high - gini_pred_high) / gini_ctrl_high * 100.0
                    if gini_ctrl_high > 0 else 0.0)
    
    # Debug
    print(f"\n[DEBUG] High-degree SOURCES ({len(high_degree_nodes)} nodes):")
    print(f"  Target dist control: sum={deg_ctrl_targets_high.sum():.0f}, Gini={gini_ctrl_high:.4f}")
    print(f"  Target dist predicted: sum={deg_pred_targets_high.sum():.0f}, Gini={gini_pred_high:.4f}")
    print(f"  -> Equality Gain = {eq_gain_high:.2f}%")
    
    return eq_gain_low, eq_gain_high
    
def compute_stratified_metrics(
    recs_by_u: Dict[int, List[int]],
    train_edges: torch.Tensor,
    test_edges: torch.Tensor,
    degrees: torch.Tensor,
    num_nodes: int,
    k: int = 10,
    low_percentile: float = 20.0,
    high_percentile: float = 80.0,
    fairness_mode: str = 'indegree',
    stratification_mode: str = 'source'  
) -> Dict[str, Tuple[float, float]]:
    """
    Compute stratified precision and equality gain metrics.
    
    Args:
        recs_by_u: Recommendations
        train_edges: Training edges
        test_edges: Test edges
        degrees: Node degrees (from training graph)
        num_nodes: Total number of nodes
        k: Cutoff value (default: 10)
        low_percentile: Threshold for low-degree nodes (default: 20%)
        high_percentile: Threshold for high-degree nodes (default: 80%)
        fairness_mode: Degree counting mode
        stratification_mode: 'source', 'target', or 'mixed' (default: 'source')
    
    Returns:
        dict with stratified metrics
    """
    print("\n" + "="*80)
    print(f"COMPUTING STRATIFIED METRICS @ K={k} (Mode: {stratification_mode.upper()})")
    print("="*80)
    
    # Stratify nodes
    low_nodes, high_nodes = stratify_nodes_by_degree(
        degrees, low_percentile, high_percentile
    )
    
    # Compute stratified precision (always by SOURCE)
    prec_low, prec_high = compute_stratified_precision(
        recs_by_u, test_edges, low_nodes, high_nodes, k
    )
    
    print(f"\nStratified Precision@{k} (by SOURCE):")
    print(f"  Low-degree:  {prec_low:.4f} ({prec_low*100:.2f}%)")
    print(f"  High-degree: {prec_high:.4f} ({prec_high*100:.2f}%)")
    
    # Compute stratified equality gain based on mode
    if stratification_mode == 'source':
        eq_gain_low, eq_gain_high = compute_stratified_equality_gain_by_source(
            recs_by_u, train_edges, test_edges, low_nodes, high_nodes,
            num_nodes, k, fairness_mode
        )
        mode_label = "SOURCE"
    elif stratification_mode == 'target':
        eq_gain_low, eq_gain_high = compute_stratified_equality_gain_by_target(
            recs_by_u, train_edges, test_edges, low_nodes, high_nodes,
            num_nodes, k, fairness_mode
        )
        mode_label = "TARGET"
    elif stratification_mode == 'mixed':
        # Use old function (renamed)
        eq_gain_low, eq_gain_high = compute_stratified_equality_gain(
            recs_by_u, train_edges, test_edges, low_nodes, high_nodes,
            num_nodes, k, fairness_mode
        )
        mode_label = "MIXED (deprecated)"
    else:
        raise ValueError(f"Invalid stratification_mode: {stratification_mode}")
    
    print(f"\nStratified Equality Gain@{k} (by {mode_label}):")
    print(f"  Low-degree:  {eq_gain_low:>7.2f}%")
    print(f"  High-degree: {eq_gain_high:>7.2f}%")
    print("="*80)
    
    return {
        f'Stratified_Precision_Low@{k}': (0.0, prec_low),
        f'Stratified_Precision_High@{k}': (0.0, prec_high),
        f'Stratified_EqualityGain_Low@{k}_{stratification_mode}': (0.0, eq_gain_low),
        f'Stratified_EqualityGain_High@{k}_{stratification_mode}': (0.0, eq_gain_high),
        '_low_degree_nodes': low_nodes,
        '_high_degree_nodes': high_nodes,
        '_stratification_mode': stratification_mode
    }


def stratify_nodes_by_degree(
    degrees: torch.Tensor,
    low_percentile: float = 20.0,
    high_percentile: float = 80.0
) -> Tuple[List[int], List[int]]:
    """
    Stratify nodes into low-degree and high-degree groups.
    
    Args:
        degrees: Node degrees [num_nodes]
        low_percentile: Percentile threshold for low-degree (default: 20%)
        high_percentile: Percentile threshold for high-degree (default: 80%)
    
    Returns:
        (low_degree_nodes, high_degree_nodes)
    """
    # Compute thresholds
    low_threshold = np.percentile(degrees.cpu().numpy(), low_percentile)
    high_threshold = np.percentile(degrees.cpu().numpy(), high_percentile)
    
    # Stratify
    low_degree_nodes = [
        i for i, deg in enumerate(degrees.tolist())
        if deg <= low_threshold
    ]
    
    high_degree_nodes = [
        i for i, deg in enumerate(degrees.tolist())
        if deg >= high_threshold
    ]
    
    print(f"\nNode Stratification (by degree):")
    print(f"  Low-degree (<= {low_threshold:.1f}): {len(low_degree_nodes)} nodes")
    print(f"  High-degree (>= {high_threshold:.1f}): {len(high_degree_nodes)} nodes")
    
    return low_degree_nodes, high_degree_nodes


def compute_stratified_precision(
    recs_by_u: Dict[int, List[int]],
    test_edges: torch.Tensor,
    low_degree_nodes: List[int],
    high_degree_nodes: List[int],
    k: int
) -> Tuple[float, float]:
    """
    Compute precision@K separately for low-degree and high-degree nodes.
    
    Args:
        recs_by_u: Recommendations
        test_edges: Test edges
        low_degree_nodes: List of low-degree node IDs
        high_degree_nodes: List of high-degree node IDs
        k: Cutoff value
    
    Returns:
        (precision_low, precision_high)
    """
    # Build test edge set (undirected)
    test_edge_set = set()
    for u, v in test_edges.tolist():
        test_edge_set.add((u, v))
        test_edge_set.add((v, u))
    
    # Compute precision for low-degree nodes
    total_prec_low = 0.0
    count_low = 0
    for u in low_degree_nodes:
        if u not in recs_by_u:
            continue
        topk = recs_by_u[u][:k]
        if not topk:
            continue
        hits = sum(1 for v in topk if (u, v) in test_edge_set)
        total_prec_low += hits / len(topk)
        count_low += 1
    
    prec_low = total_prec_low / count_low if count_low > 0 else 0.0
    
    # Compute precision for high-degree nodes
    total_prec_high = 0.0
    count_high = 0
    for u in high_degree_nodes:
        if u not in recs_by_u:
            continue
        topk = recs_by_u[u][:k]
        if not topk:
            continue
        hits = sum(1 for v in topk if (u, v) in test_edge_set)
        total_prec_high += hits / len(topk)
        count_high += 1
    
    prec_high = total_prec_high / count_high if count_high > 0 else 0.0
    
    return prec_low, prec_high


def compute_stratified_equality_gain(
    recs_by_u: Dict[int, List[int]],
    train_edges: torch.Tensor,
    test_edges: torch.Tensor,
    low_degree_nodes: List[int],
    high_degree_nodes: List[int],
    num_nodes: int,
    k: int,
    mode: str = 'indegree'
) -> Tuple[float, float]:
    """
    Compute Equality Gain separately for low-degree and high-degree subgraphs.
    
    Args:
        recs_by_u: Recommendations
        train_edges: Training edges
        test_edges: Test edges
        low_degree_nodes: Low-degree node IDs
        high_degree_nodes: High-degree node IDs
        num_nodes: Total number of nodes
        k: Cutoff value
        mode: Degree counting mode
    
    Returns:
        (eq_gain_low, eq_gain_high)
    """
    # Filter edges involving low-degree nodes
    low_set = set(low_degree_nodes)
    train_low = train_edges[[u in low_set or v in low_set 
                             for u, v in train_edges.tolist()]]
    test_low = test_edges[[u in low_set or v in low_set 
                           for u, v in test_edges.tolist()]]
    recs_low = {u: vs for u, vs in recs_by_u.items() if u in low_set}
    
    # Compute for low-degree subgraph
    deg_ctrl_low = build_control_degree_distribution(
        train_low, test_low, num_nodes, mode
    )
    gini_ctrl_low = compute_gini_coefficient(deg_ctrl_low)
    
    deg_pred_low = build_predicted_degree_distribution(
        train_low, recs_low, num_nodes, k, mode
    )
    gini_pred_low = compute_gini_coefficient(deg_pred_low)
    
    if gini_ctrl_low > 0:
        eq_gain_low = ((gini_ctrl_low - gini_pred_low) / gini_ctrl_low) * 100.0
    else:
        eq_gain_low = 0.0
    
    # Filter edges involving high-degree nodes
    high_set = set(high_degree_nodes)
    train_high = train_edges[[u in high_set or v in high_set 
                              for u, v in train_edges.tolist()]]
    test_high = test_edges[[u in high_set or v in high_set 
                            for u, v in test_edges.tolist()]]
    recs_high = {u: vs for u, vs in recs_by_u.items() if u in high_set}
    
    # Compute for high-degree subgraph
    deg_ctrl_high = build_control_degree_distribution(
        train_high, test_high, num_nodes, mode
    )
    gini_ctrl_high = compute_gini_coefficient(deg_ctrl_high)
    
    deg_pred_high = build_predicted_degree_distribution(
        train_high, recs_high, num_nodes, k, mode
    )
    gini_pred_high = compute_gini_coefficient(deg_pred_high)
    
    if gini_ctrl_high > 0:
        eq_gain_high = ((gini_ctrl_high - gini_pred_high) / gini_ctrl_high) * 100.0
    else:
        eq_gain_high = 0.0
    
    return eq_gain_low, eq_gain_high