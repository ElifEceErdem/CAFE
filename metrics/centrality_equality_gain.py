"""
Equality Gain Metrics for Multiple Centrality Measures

Extends equality_gain.py to support:
- Degree centrality (original)
- Katz centrality
- PageRank centrality
- Eigenvector centrality
- Closeness centrality

All metrics use the same Equality Gain formula:
    EG = (Gini_control - Gini_pred) / Gini_control * 100
"""

import torch
import numpy as np
import math
import networkx as nx
from typing import Dict, Hashable, List, Optional, Tuple
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import networkit as nk

from metrics.graph_utils import (
    build_control_edge_set,
    build_predicted_edge_set,
    canonicalize_undirected_edges,
    degree_from_undirected_edges,
    degree_from_simple_undirected_edges,
    edge_tensor_2col,
    novel_edges_against_simple_undirected_base,
    recommendations_to_edges,
)


_EG_CENTRALITY_TYPES = {
    'degree',
    'katz',
    'pagerank',
    'eigenvector',
    'closeness',
    'harmonic',
}


def _validate_eg_centrality_type(centrality_type: str) -> None:
    """Reject directed degree modes at the undirected structural-EG boundary."""
    if centrality_type not in _EG_CENTRALITY_TYPES:
        if centrality_type in {'indegree', 'outdegree', 'total'}:
            raise ValueError(
                f"{centrality_type!r} is a directed degree mode and is not "
                "valid for Equality Gain on a simple undirected graph. Use "
                "'degree'; use the exposure metrics for directed counts."
            )
        raise ValueError(f"Unknown centrality type: {centrality_type}")


def compute_gini_coefficient(values: torch.Tensor) -> float:
    """
    Compute Gini coefficient for a distribution.
    
    Args:
        values: Distribution values [num_nodes]
    
    Returns:
        float: Gini coefficient in [0, 1]
    """
    # Filter out zero values
    nonzero = values[values > 0]
    
    if nonzero.numel() == 0:
        return 0.0
    
    # Sort values
    x_sorted, _ = torch.sort(nonzero.view(-1))
    n = x_sorted.numel()
    s = x_sorted.sum()
    
    if s == 0:
        return 0.0
    
    # Gini formula: G = 1 + (1/n) - (2/n*s) * sum((n+1-i) * x_i)
    idx = torch.arange(1, n + 1, dtype=x_sorted.dtype, device=x_sorted.device)
    gini = 1.0 + (1.0 / n) - 2.0 * (torch.sum((n + 1 - idx) * x_sorted) / (n * s))
    
    return float(gini)


def edges_to_adjacency_matrix(edges: torch.Tensor, num_nodes: int, directed: bool = False):
    """
    Convert edge list to scipy sparse adjacency matrix.
    
    Args:
        edges: Edge list [num_edges, 2]
        num_nodes: Number of nodes
        directed: Whether graph is directed
    
    Returns:
        scipy.sparse.csr_matrix
    """
    if directed:
        edge_tensor = edge_tensor_2col(edges)
    else:
        edge_tensor = canonicalize_undirected_edges(edges, num_nodes)
    edges_np = edge_tensor.numpy()
    
    # Create COO format data
    row = edges_np[:, 0]
    col = edges_np[:, 1]
    data = np.ones(len(row))
    
    if not directed:
        # Add reverse edges for undirected graph
        row = np.concatenate([row, col])
        col = np.concatenate([col, edges_np[:, 0]])
        data = np.concatenate([data, data[:len(edges_np)]])
    
    # Create sparse matrix
    adj = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    if not directed:
        # Defense in depth: centrality on the paper's simple graph must never
        # interpret duplicate rows as edge weights.
        adj.data = np.ones_like(adj.data)
    
    return adj


def compute_degree_centrality(
    edges: torch.Tensor,
    num_nodes: int,
    mode: str = 'undirected',
) -> torch.Tensor:
    """
    Compute degree centrality (vectorized).

    ``undirected``/``degree`` canonicalizes a simple graph. The directed
    ``indegree``, ``outdegree``, and ``total`` modes operate on the supplied
    orientation and remain available to non-EG diagnostics.
    """
    if mode in ['degree', 'undirected']:
        return degree_from_undirected_edges(edges, num_nodes)
    if mode not in ['indegree', 'outdegree', 'total']:
        raise ValueError(f"Unknown degree mode: {mode}")

    edges_cpu = edge_tensor_2col(edges)
    degrees = torch.zeros(num_nodes, dtype=torch.float32)

    if mode in ['indegree', 'total']:
        dst = edges_cpu[:, 1]
        valid = (dst >= 0) & (dst < num_nodes)
        degrees.scatter_add_(0, dst[valid].long(), torch.ones(valid.sum(), dtype=torch.float32))

    if mode in ['outdegree', 'total']:
        src = edges_cpu[:, 0]
        valid = (src >= 0) & (src < num_nodes)
        degrees.scatter_add_(0, src[valid].long(), torch.ones(valid.sum(), dtype=torch.float32))

    return degrees


def compute_katz_centrality(edges: torch.Tensor, num_nodes: int, 
                            alpha: float = 0.1, beta: float = 1.0,
                            max_iter: int = 1000, tol: float = 1e-6) -> torch.Tensor:
    """
    Compute Katz centrality using power iteration.
    
    Katz centrality: x = beta * (I - alpha * A^T)^-1 * 1
    
    We use power iteration: x_{k+1} = alpha * A^T * x_k + beta
    
    Args:
        edges: Edge list [num_edges, 2]
        num_nodes: Number of nodes
        alpha: Attenuation factor (must be < 1/lambda_max)
        beta: Weight of exogenous sources
        max_iter: Maximum iterations
        tol: Convergence tolerance
    
    Returns:
        katz: Katz centrality scores [num_nodes]
    """
    # Build adjacency matrix
    adj = edges_to_adjacency_matrix(edges, num_nodes, directed=False)
    
    # Auto-adjust alpha if needed (use 0.85 / spectral_radius)
    try:
        # Compute largest eigenvalue (spectral radius)
        eigenvalues = eigs(adj.T, k=1, which='LM', return_eigenvectors=False, maxiter=100)
        spectral_radius = np.abs(eigenvalues[0])
        
        # Ensure alpha < 1/spectral_radius for convergence
        if spectral_radius > 0:
            alpha_max = 0.99 / spectral_radius
            if alpha >= alpha_max:
                alpha = 0.85 / spectral_radius
                print(f"  [Katz] Auto-adjusted alpha to {alpha:.6f} for convergence")
    except:
        # If eigenvalue computation fails, use conservative alpha
        alpha = 0.01
        print(f"  [Katz] Using conservative alpha={alpha}")
    
    # Power iteration
    x = np.ones(num_nodes, dtype=np.float32) * beta
    
    for i in range(max_iter):
        x_new = alpha * adj.T.dot(x) + beta
        
        # Check convergence
        diff = np.linalg.norm(x_new - x)
        x = x_new
        
        if diff < tol:
            break
    
    return torch.from_numpy(x)


def compute_pagerank_centrality(edges: torch.Tensor, num_nodes: int,
                                alpha: float = 0.85, max_iter: int = 100,
                                tol: float = 1e-6) -> torch.Tensor:
    """
    Compute PageRank centrality using power iteration.
    
    PageRank: x = (1-alpha)/N + alpha * A^T * D^-1 * x
    
    Args:
        edges: Edge list [num_edges, 2]
        num_nodes: Number of nodes
        alpha: Damping factor (typically 0.85)
        max_iter: Maximum iterations
        tol: Convergence tolerance
    
    Returns:
        pagerank: PageRank scores [num_nodes]
    """
    # Build adjacency matrix
    adj = edges_to_adjacency_matrix(edges, num_nodes, directed=False)
    
    # Compute out-degrees
    out_deg = np.array(adj.sum(axis=1)).flatten()
    out_deg[out_deg == 0] = 1  # Avoid division by zero
    
    # Create transition matrix: D^-1 * A
    inv_deg = 1.0 / out_deg
    transition = adj.multiply(inv_deg[:, np.newaxis]).T.tocsr()
    
    # Power iteration
    x = np.ones(num_nodes, dtype=np.float32) / num_nodes
    teleport = (1 - alpha) / num_nodes
    
    for i in range(max_iter):
        x_new = alpha * transition.dot(x) + teleport
        
        # Normalize to sum to 1
        x_new = x_new / x_new.sum()
        
        # Check convergence
        diff = np.linalg.norm(x_new - x)
        x = x_new
        
        if diff < tol:
            break
    
    return torch.from_numpy(x)


def compute_eigenvector_centrality(edges: torch.Tensor, num_nodes: int,
                                   max_iter: int = 1000, tol: float = 1e-6) -> torch.Tensor:
    """
    Compute eigenvector centrality using power iteration.
    
    Eigenvector centrality: x = A * x / ||A * x||
    
    Args:
        edges: Edge list [num_edges, 2]
        num_nodes: Number of nodes
        max_iter: Maximum iterations
        tol: Convergence tolerance
    
    Returns:
        eigenvec: Eigenvector centrality scores [num_nodes]
    """
    # Build adjacency matrix
    adj = edges_to_adjacency_matrix(edges, num_nodes, directed=False)
    
    # Power iteration
    x = np.ones(num_nodes, dtype=np.float32)
    x = x / np.linalg.norm(x)
    
    for i in range(max_iter):
        x_new = adj.dot(x)
        
        # Normalize
        norm = np.linalg.norm(x_new)
        if norm == 0:
            # Graph might be disconnected, return zeros
            return torch.zeros(num_nodes, dtype=torch.float32)
        
        x_new = x_new / norm
        
        # Check convergence
        diff = np.linalg.norm(x_new - x)
        x = x_new
        
        if diff < tol:
            break
    
    return torch.from_numpy(x)


def _eppstein_wang_sample_size(num_nodes: int, epsilon: float = 0.1) -> int:
    """
    Compute sample size using Eppstein-Wang (2004) formula.
    
    k = ceil(ln(n) / epsilon^2)
    
    This gives an additive error of epsilon * Diameter with high probability.
    
    Args:
        num_nodes: Number of nodes in the graph
        epsilon: Desired error bound (default: 0.1)
    
    Returns:
        int: Number of pivot samples to use
    """
    k = math.ceil(math.log(num_nodes) / (epsilon ** 2))
    # Ensure at least 50 samples and at most num_nodes
    return max(50, min(k, num_nodes))


def _build_networkit_graph(edges: torch.Tensor, num_nodes: int):
    """
    Build a NetworKit graph from a PyTorch edge tensor (vectorized).
    """

    unique_edges = canonicalize_undirected_edges(edges, num_nodes).numpy()

    G = nk.Graph(num_nodes, weighted=False, directed=False)
    for i in range(len(unique_edges)):
        G.addEdge(int(unique_edges[i, 0]), int(unique_edges[i, 1]))
    return G


def compute_closeness_centrality(edges: torch.Tensor, num_nodes: int,
                                 use_sampling: bool = True,
                                 sample_size: int = None,
                                 epsilon: float = 0.1,
                                 random_seed: int = 42) -> torch.Tensor:
    """
    Compute closeness centrality using NetworKit.
    
    For small graphs (n < 5000), uses exact computation.
    For larger graphs, uses RAND approximation (Eppstein & Wang, 2004)
    with adaptive sample size k = ceil(ln(n) / epsilon^2).
    
    Args:
        edges: Edge list [num_edges, 2]
        num_nodes: Number of nodes
        use_sampling: Whether to use sampling for large graphs
        sample_size: Override sample size (if None, uses EW formula)
        epsilon: Error bound for EW formula (default: 0.1)
        random_seed: NetworKit RNG seed used by approximate closeness
    
    Returns:
        closeness: Closeness centrality scores [num_nodes]
    """
    try:
        return _compute_closeness_networkit(
            edges, num_nodes, use_sampling, sample_size, epsilon, random_seed
        )
    except ImportError:
        print("  [Closeness] NetworKit not available, falling back to NetworkX")
        return _compute_closeness_networkx(edges, num_nodes)


def _compute_closeness_networkit(edges: torch.Tensor, num_nodes: int,
                                  use_sampling: bool = True,
                                  sample_size: int = None,
                                  epsilon: float = 0.1,
                                  random_seed: int = 42) -> torch.Tensor:
    """
    Compute closeness centrality using NetworKit (C++ backend).
    
    Uses exact computation for small graphs, RAND approximation for large ones.
    """    
    G = _build_networkit_graph(edges, num_nodes)
    
    EXACT_THRESHOLD = 5000
    
    if not use_sampling or num_nodes <= EXACT_THRESHOLD:
        # Exact computation via NetworKit (much faster than NetworkX)
        print(f"  [Closeness] Computing exact closeness for {num_nodes} nodes (NetworKit C++)")
        cc = nk.centrality.Closeness(G, False, nk.centrality.ClosenessVariant.Generalized)
        cc.run()
    else:
        # Approximate using RAND algorithm (Eppstein & Wang, 2004)
        if sample_size is None:
            sample_size = _eppstein_wang_sample_size(num_nodes, epsilon)
        
        print(f"  [Closeness] Approximating closeness for {num_nodes} nodes "
              f"(k={sample_size} samples, ε={epsilon}, seed={random_seed}, "
              f"EW formula)")
        # ApproxCloseness samples pivots internally.  Reset NetworKit's RNG
        # immediately before every invocation so the observed/control and
        # recommendation graphs use common, reproducible random pivots.
        nk.setSeed(int(random_seed), False)
        cc = nk.centrality.ApproxCloseness(G, nSamples=sample_size, 
                                            normalized=True, 
                                            type=nk.centrality.ClosenessType.OUTBOUND)
        cc.run()
    
    # Extract scores into tensor
    scores = cc.scores()
    closeness = torch.tensor(scores, dtype=torch.float32)
    
    return closeness


def _compute_closeness_networkx(edges: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Fallback: exact closeness computation using NetworkX."""
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges_list = edges.cpu().numpy().tolist()
    G.add_edges_from(edges_list)
    
    closeness_dict = nx.closeness_centrality(G)
    
    closeness = torch.zeros(num_nodes, dtype=torch.float32)
    for node, value in closeness_dict.items():
        if node < num_nodes:
            closeness[node] = value
    
    return closeness


def compute_harmonic_centrality(edges: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Compute harmonic centrality: H(v) = sum_{u!=v} 1/d(v,u).
    
    Advantages over closeness:
    - Handles disconnected graphs naturally (1/inf = 0)
    - Preferred in modern social network analysis
    
    Args:
        edges: Edge list [num_edges, 2]
        num_nodes: Number of nodes
    
    Returns:
        harmonic: Harmonic centrality scores [num_nodes]
    """
    try:
        G = _build_networkit_graph(edges, num_nodes)
        hc = nk.centrality.HarmonicCloseness(G, normalized=True)
        hc.run()
        return torch.tensor(hc.scores(), dtype=torch.float32)
    except ImportError:
        # Fallback to NetworkX
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edges.cpu().numpy().tolist())
        harmonic_dict = nx.harmonic_centrality(G)
        harmonic = torch.zeros(num_nodes, dtype=torch.float32)
        for node, value in harmonic_dict.items():
            if node < num_nodes:
                harmonic[node] = value
        # Normalize
        if harmonic.max() > 0:
            harmonic = harmonic / harmonic.max()
        return harmonic


def build_centrality_distribution(
    edges: torch.Tensor,
    num_nodes: int,
    centrality_type: str,
    **kwargs
) -> torch.Tensor:
    """
    Build centrality distribution for a graph.
    
    Args:
        edges: Edge list [num_edges, 2]
        num_nodes: Number of nodes
        centrality_type: Type of centrality
            - 'degree' / 'indegree' / 'outdegree' / 'total'
            - 'katz'
            - 'pagerank'
            - 'eigenvector'
            - 'closeness'
        **kwargs: Additional arguments for specific centrality measures
    
    Returns:
        centrality: Centrality scores [num_nodes]
    """
    if centrality_type in ['degree', 'indegree', 'outdegree', 'total']:
        mode = 'undirected' if centrality_type == 'degree' else centrality_type
        return compute_degree_centrality(edges, num_nodes, mode)
    
    elif centrality_type == 'katz':
        return compute_katz_centrality(edges, num_nodes, **kwargs)
    
    elif centrality_type == 'pagerank':
        return compute_pagerank_centrality(edges, num_nodes, **kwargs)
    
    elif centrality_type == 'eigenvector':
        return compute_eigenvector_centrality(edges, num_nodes, **kwargs)
    
    elif centrality_type == 'closeness':
        return compute_closeness_centrality(edges, num_nodes, **kwargs)
    
    elif centrality_type == 'harmonic':
        return compute_harmonic_centrality(edges, num_nodes)
    
    else:
        raise ValueError(f"Unknown centrality_type: {centrality_type}")


def build_predicted_centrality(
    train_edges: torch.Tensor,
    recs_by_u: Dict[int, List[int]],
    num_nodes: int,
    k: int,
    centrality_type: str,
    **kwargs
) -> torch.Tensor:
    """
    Build centrality distribution for predicted graph (train + recommendations).
    """
    _validate_eg_centrality_type(centrality_type)
    predicted_edges = build_predicted_edge_set(
        train_edges, recs_by_u, num_nodes, k
    )
    if centrality_type == 'degree':
        return degree_from_simple_undirected_edges(predicted_edges, num_nodes)
    return build_centrality_distribution(
        predicted_edges, num_nodes, centrality_type, **kwargs
    )


def build_control_centrality(
    train_edges: torch.Tensor,
    test_edges: torch.Tensor,
    num_nodes: int,
    centrality_type: str,
    **kwargs
) -> torch.Tensor:
    """
    Build centrality distribution for control graph (train + test).
    """
    _validate_eg_centrality_type(centrality_type)
    control_edges = build_control_edge_set(train_edges, test_edges, num_nodes)
    if centrality_type == 'degree':
        return degree_from_simple_undirected_edges(control_edges, num_nodes)
    return build_centrality_distribution(
        control_edges, num_nodes, centrality_type, **kwargs
    )


def compute_centrality_equality_gain(
    recs_by_u: Dict[int, List[int]],
    train_edges: torch.Tensor,
    test_edges: torch.Tensor,
    num_nodes: int,
    k: int,
    centrality_type: str,
    **kwargs
) -> Tuple[float, float, float]:
    """
    Compute Equality Gain for a specific centrality measure.
    
    Returns:
        (equality_gain, gini_control, gini_pred)
    """
    # Build control centrality
    cent_control = build_control_centrality(
        train_edges, test_edges, num_nodes, centrality_type, **kwargs
    )
    gini_control = compute_gini_coefficient(cent_control)
    
    # Build predicted centrality
    cent_pred = build_predicted_centrality(
        train_edges, recs_by_u, num_nodes, k, centrality_type, **kwargs
    )
    gini_pred = compute_gini_coefficient(cent_pred)
    
    # Compute equality gain
    if gini_control == 0:
        equality_gain = 0.0
    else:
        equality_gain = ((gini_control - gini_pred) / gini_control) * 100.0
    
    return equality_gain, gini_control, gini_pred


def compute_centrality_equality_gain_by_source_bins(
    recs_by_u: Dict[int, List[int]],
    source_bins: Dict[int, Hashable],
    train_edges: torch.Tensor,
    test_edges: torch.Tensor,
    num_nodes: int,
    k: int,
    centrality_type: str,
    bin_ids: Optional[List[Hashable]] = None,
    **kwargs,
) -> Dict[Hashable, Tuple[float, float, float]]:
    """Compute EG counterfactuals for source-defined recommendation bins.

    Each bin uses the full observed control graph and a predicted graph in
    which only recommendations issued by sources assigned to that bin
    materialize.  The bin values are counterfactuals and are not additive.
    """
    if bin_ids is None:
        bin_ids = sorted(set(source_bins.values()))

    control = build_control_centrality(
        train_edges, test_edges, num_nodes, centrality_type, **kwargs
    )
    gini_control = compute_gini_coefficient(control)
    results = {}

    for bin_id in bin_ids:
        filtered_recs = {
            u: values
            for u, values in recs_by_u.items()
            if source_bins.get(int(u)) == bin_id
        }
        predicted = build_predicted_centrality(
            train_edges, filtered_recs, num_nodes, k, centrality_type, **kwargs
        )
        gini_pred = compute_gini_coefficient(predicted)
        equality_gain = (
            0.0
            if gini_control == 0
            else ((gini_control - gini_pred) / gini_control) * 100.0
        )
        results[bin_id] = (equality_gain, gini_control, gini_pred)

    return results


def compute_degree_equality_gain_by_source_bins_for_models(
    recs_by_model: Dict[str, Dict[int, List[int]]],
    source_bins: Dict[int, Hashable],
    train_edges: torch.Tensor,
    test_edges: torch.Tensor,
    num_nodes: int,
    k: int,
    bin_ids: Optional[List[Hashable]] = None,
) -> Dict[str, Dict[Hashable, Tuple[float, float, float]]]:
    """Compute source-bin Degree EG for multiple models with shared caches.

    The observed control, canonical training graph, and training degrees are
    model-independent and are built once. Per bin, only the recommendation
    events are canonicalized; train overlaps are removed by searching the
    cached sorted training-edge keys. This is numerically equivalent to
    :func:`compute_centrality_equality_gain_by_source_bins` with
    ``centrality_type="degree"``.
    """
    if bin_ids is None:
        bin_ids = sorted(set(source_bins.values()))

    train_simple = canonicalize_undirected_edges(train_edges, num_nodes)
    train_degree = degree_from_simple_undirected_edges(
        train_simple, num_nodes
    )
    control_simple = build_control_edge_set(
        train_simple, test_edges, num_nodes
    )
    control_degree = degree_from_simple_undirected_edges(
        control_simple, num_nodes
    )
    gini_control = compute_gini_coefficient(control_degree)

    all_results = {}
    for model_name, recs_by_u in recs_by_model.items():
        grouped_recs = {bin_id: {} for bin_id in bin_ids}
        for raw_u, values in recs_by_u.items():
            bin_id = source_bins.get(int(raw_u))
            if bin_id in grouped_recs:
                grouped_recs[bin_id][raw_u] = values

        model_results = {}
        for bin_id in bin_ids:
            rec_edges = recommendations_to_edges(
                grouped_recs[bin_id], num_nodes, k
            )
            novel_edges = novel_edges_against_simple_undirected_base(
                train_simple, rec_edges, num_nodes
            )
            predicted_degree = train_degree.clone()
            if novel_edges.numel() > 0:
                predicted_degree += torch.bincount(
                    novel_edges.reshape(-1), minlength=num_nodes
                ).to(dtype=torch.float32)
            gini_pred = compute_gini_coefficient(predicted_degree)
            equality_gain = (
                0.0
                if gini_control == 0
                else ((gini_control - gini_pred) / gini_control) * 100.0
            )
            model_results[bin_id] = (
                equality_gain,
                gini_control,
                gini_pred,
            )
        all_results[model_name] = model_results

    return all_results


def compute_all_centrality_equality_gains(
    recs_by_u: Dict[int, List[int]],
    train_edges: torch.Tensor,
    test_edges: torch.Tensor,
    num_nodes: int,
    k_values: List[int] = None,
    centrality_types: List[str] = None,
    use_sampling_closeness: bool = True,
    closeness_random_seed: int = 42,
    closeness_sample_size: int = None,
    closeness_epsilon: float = 0.1,
    control_cache: Optional[Dict[Tuple[object, ...], Tuple[torch.Tensor, float]]] = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute Equality Gain for multiple centrality measures and K values.
    
    Args:
        recs_by_u: Recommendations
        train_edges: Training edges
        test_edges: Test edges
        num_nodes: Total number of nodes
        k_values: List of K values
        centrality_types: List of centrality types to compute
        use_sampling_closeness: Use sampling for closeness (large graphs)
        closeness_random_seed: NetworKit seed for approximate closeness
        closeness_sample_size: Optional fixed number of approximation pivots
        closeness_epsilon: Eppstein--Wang error parameter when sample size is
            selected adaptively
        control_cache: Optional caller-owned cache for control centralities.
            Reusing it is exact when train/test edges and centrality options
            are unchanged.
    
    Returns:
        dict: {f'{centrality}_EG@{k}': (0.0, value)}
    """
    if k_values is None:
        k_values = [10, 20, 50, 100]
    
    if centrality_types is None:
        centrality_types = ['degree', 'katz', 'pagerank', 'eigenvector', 'closeness']
    for centrality_type in centrality_types:
        _validate_eg_centrality_type(centrality_type)

    # ADD THIS: Move to CPU for centrality computations
    train_edges = train_edges.cpu()
    test_edges = test_edges.cpu()
    
    results = {}
    
    print("\n" + "="*80)
    print("COMPUTING CENTRALITY-BASED EQUALITY GAIN METRICS")
    print("="*80)
    
    for cent_type in centrality_types:
        print(f"\n{cent_type.upper()} Centrality:")
        print("-" * 60)

        cent_kwargs = {}
        if cent_type == 'closeness':
            cent_kwargs['use_sampling'] = use_sampling_closeness
            cent_kwargs['epsilon'] = closeness_epsilon
            cent_kwargs['sample_size'] = closeness_sample_size
            cent_kwargs['random_seed'] = closeness_random_seed

        cache_key = (
            cent_type,
            int(num_nodes),
            bool(use_sampling_closeness),
            int(closeness_random_seed),
            closeness_sample_size,
            float(closeness_epsilon),
        )

        # Control centrality is shared across K and, in configuration sweeps,
        # across every inference-only beta/boost combination.
        try:
            cached = (
                control_cache.get(cache_key)
                if control_cache is not None else None
            )
            if cached is None:
                cent_control = build_control_centrality(
                    train_edges, test_edges, num_nodes, cent_type,
                    **cent_kwargs
                )
                gini_ctrl = compute_gini_coefficient(cent_control)
                if control_cache is not None:
                    control_cache[cache_key] = (cent_control, gini_ctrl)
            else:
                cent_control, gini_ctrl = cached
        except Exception as e:
            print(f"  Failed to compute control centrality ({str(e)})")
            for k in k_values:
                results[f'{cent_type.capitalize()}_EG@{k}'] = (0.0, 0.0)
            continue

        for k in k_values:
            try:
                cent_pred = build_predicted_centrality(
                    train_edges, recs_by_u, num_nodes, k, cent_type, **cent_kwargs
                )
                gini_pred = compute_gini_coefficient(cent_pred)

                if gini_ctrl == 0:
                    eq_gain = 0.0
                else:
                    eq_gain = ((gini_ctrl - gini_pred) / gini_ctrl) * 100.0

                metric_name = f'{cent_type.capitalize()}_EG@{k}'
                results[metric_name] = (0.0, eq_gain)

                print(f"  K={k:>3}: EG = {eq_gain:>7.2f}% "
                      f"(Gini_ctrl={gini_ctrl:.4f}, Gini_pred={gini_pred:.4f})")

            except Exception as e:
                print(f"  K={k:>3}: Failed to compute ({str(e)})")
                results[f'{cent_type.capitalize()}_EG@{k}'] = (0.0, 0.0)

    print("="*80)

    return results
