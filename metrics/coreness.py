"""
Coreness (k-core) computation for fairness re-ranking.

Implements iterative H-index to measure node embeddedness.
"""

import torch
import numpy as np
from typing import Dict


def compute_coreness(edges: torch.Tensor, num_nodes: int, 
                     max_iter: int = 100) -> torch.Tensor:
    """
    Compute coreness (k-core index) using iterative H-index.
    
    Coreness measures embeddedness: how deep a node sits in a 
    densely connected core.
    
    Algorithm:
    1. Start with H^(0) = degree
    2. Iterate: H^(t+1)(v) = H-index of {H^(t)(u) : u \in N(v)}
    3. Converges to coreness (k-core membership)
    
    Args:
        edges: Edge list [num_edges, 2]
        num_nodes: Number of nodes
        max_iter: Maximum iterations
    
    Returns:
        coreness: Coreness scores [num_nodes]
    """
    # Build adjacency list
    adj_list = {i: set() for i in range(num_nodes)}
    for u, v in edges.tolist():
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adj_list[u].add(v)
            adj_list[v].add(u)
    
    # Initialize with degree
    h_values = np.array([len(adj_list[i]) for i in range(num_nodes)], 
                        dtype=np.float32)
    
    print(f"  Computing coreness (iterative H-index)...")
    print(f"  Initial: min={h_values.min():.1f}, max={h_values.max():.1f}, "
          f"mean={h_values.mean():.1f}")
    
    # Iterative H-index
    for iteration in range(max_iter):
        h_new = np.zeros(num_nodes, dtype=np.float32)
        
        for v in range(num_nodes):
            neighbors = list(adj_list[v])
            if not neighbors:
                h_new[v] = 0.0
                continue
            
            # Get H-values of neighbors and compute H-index
            neighbor_h = h_values[neighbors]
            h_new[v] = compute_h_index(neighbor_h)
        
        # Check convergence
        diff = np.abs(h_new - h_values).max()
        h_values = h_new
        
        if iteration % 10 == 0 or diff < 1e-6:
            print(f"  Iteration {iteration}: max_change={diff:.4f}, "
                  f"mean_coreness={h_values.mean():.2f}")
        
        if diff < 1e-6:
            print(f"  Converged at iteration {iteration}")
            break
    
    return torch.from_numpy(h_values)


def compute_h_index(values: np.ndarray) -> float:
    """
    Compute H-index of a list of values.
    
    Args:
        values: Array of values
    
    Returns:
        h_index: H-index (integer)
    """
    if len(values) == 0:
        return 0.0
    
    # Sort in descending order
    sorted_vals = np.sort(values)[::-1]
    
    # Find largest h where sorted_vals[h-1] >= h
    h = 0
    for i, val in enumerate(sorted_vals, start=1):
        if val >= i:
            h = i
        else:
            break
    
    return float(h)


def compute_common_neighbors(u: int, v: int, adj_list: Dict[int, set]) -> int:
    """
    Compute number of common neighbors between u and v.
    
    Args:
        u: Source node
        v: Target node
        adj_list: Adjacency list
    
    Returns:
        Number of common neighbors
    """
    return len(adj_list[u] & adj_list[v])