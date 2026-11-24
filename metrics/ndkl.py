"""
NDKL (Normalized Discounted KL-divergence) Fairness Metrics

Measures fairness by comparing recommendation distribution against
target degree distribution.

Refactored to include necessary helpers
"""

import torch
import math
import numpy as np

# ============================================================================
# HELPER FUNCTIONS (Previously in ndkl_helper.py)
# ============================================================================

def categorize_nodes_by_degree(degrees, method='cumulative', num_bins=4):
    """
    Categorize nodes into groups based on degree.
    """
    if method == 'cumulative':
        # Sort nodes by degree
        sorted_indices = torch.argsort(degrees)
        num_nodes = len(degrees)
        bin_size = num_nodes // num_bins
        
        categories = torch.zeros(num_nodes, dtype=torch.long)
        thresholds = []
        
        for i in range(num_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < num_bins - 1 else num_nodes
            
            nodes_in_bin = sorted_indices[start:end]
            categories[nodes_in_bin] = i
            
            # Record threshold (max degree in this bin)
            thresholds.append(degrees[nodes_in_bin].max().item())
            
        return categories, thresholds
    else:
        raise ValueError(f"Unknown method: {method}")

def compute_target_distribution(train_edges, node_categories):
    """
    Compute target distribution P(G) based on training edges.
    P(G) = (Edges connected to group G) / (Total Edges)
    """
    # Simply count how many ends of edges belong to each group
    # Each edge (u, v) contributes to group of u and group of v
    
    u_list, v_list = train_edges
    if torch.is_tensor(u_list):
        u_list = u_list.cpu()
        v_list = v_list.cpu()
        
    cat_u = node_categories[u_list]
    cat_v = node_categories[v_list]
    
    num_groups = node_categories.max().item() + 1
    counts = torch.zeros(num_groups)
    
    for g in range(num_groups):
        c = (cat_u == g).sum() + (cat_v == g).sum()
        counts[g] = c.float()
        
    total = counts.sum()
    dist = counts / total
    
    return dist, counts

def get_evaluation_user_set(split_edge, num_nodes, mode='test_users'):
    """
    Get consistent user set for evaluation across all methods.
    """
    if mode == 'all_nodes':
        user_set = list(range(num_nodes))
        print(f"\nEvaluation user set: ALL {len(user_set)} nodes")
        return user_set
    
    elif mode == 'test_users':
        pos_test_edge = split_edge['test']['edge']
        test_users = set(pos_test_edge[:, 0].tolist()) | \
                     set(pos_test_edge[:, 1].tolist())
        test_users = sorted([u for u in test_users if u < num_nodes])
        
        print(f"\nEvaluation user set: {len(test_users)} test users "
              f"(out of {num_nodes} total nodes)")
        return test_users
    
    else:
        raise ValueError(f"Unknown mode: {mode}")

def print_ndkl_results(ndkl_results, k_values, mode='per_user'):
    print("\nNDKL Results:")
    for k in k_values:
        if k in ndkl_results['ndkl_scores']:
            print(f"  NDKL@{k}: {ndkl_results['ndkl_scores'][k]:.4f}")


# ============================================================================
# CORE NDKL LOGIC (User Implementation 1)
# ============================================================================

def compute_ndkl(
    ranking,
    pi,
    *,
    k: int | None = None,
    num_groups: int | None = None,
    eps: float = 1e-12,
) -> float:
    """
    Compute NDKL for a ranked list. ranking can be either
    the dict returned by reconstruct_ranked_edges (expects key 'group'), 
    or a 1D tensor/list of group IDs per rank position.

    Definition:
      NDKL = (1/Z) * sum_{k=1..K} [ 1/log2(k+1) * D_KL(pi_hat_k || pi) ]
      Z    = sum_{k=1..K} 1/log2(k+1)
      pi is the target/global group distribution (length num_groups).
    """
                                        # get ranking
    if isinstance(ranking, dict):
        groups = ranking.get("group")
        if groups is None:
            raise ValueError("ranking dict must contain a 'group' key.")
    else:
        groups = ranking
                                        # make tensor of groups
    groups_t = torch.as_tensor(groups, dtype=torch.long).view(-1)
    if groups_t.numel() == 0:
        # raise ValueError("ranking/groups is empty.")
        return 0.0 # Handle empty ranking gracefully
                                        # set k, k = complete ranking if not defined
    if k is None:
        k_eff = int(groups_t.numel())
    else:
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        k_eff = int(min(k, groups_t.numel()))
    
    if k_eff == 0:
        return 0.0
        
                                        # limit ranking up to k
    groups_t = groups_t[:k_eff]
                                        # determine number of groups in pi distribution
    pi_t = torch.as_tensor(pi, dtype=torch.float).view(-1)
    if num_groups is None:
        num_groups = int(pi_t.numel())
    else:
        num_groups = int(num_groups)
        
    if int(pi_t.numel()) != int(num_groups):
        raise ValueError(f"pi must have length {num_groups}, got {pi_t.numel()}.")
                                        # smooth pi so we never divide by 0 
                                        # in case a group is absent globally
    if eps and eps > 0.0:
        pi_t = pi_t.clamp_min(eps)
        pi_t = pi_t / pi_t.sum()
    else:
                                        # target distribution pi must be strictly positive 
                                        # wherever pi_hat can put mass, otherwise KL is infinite
        if bool((pi_t <= 0).any()):
            raise ValueError(
                "pi contains zeros/negatives; KL(pi_hat||pi) is undefined/infinite without smoothing. "
                "Pass eps>0 to smooth, or ensure pi>0 for all groups."
            )
        pi_t = pi_t / pi_t.sum()
                                        # empirical prefix distributions pi_hat_k 
                                        # for all k via cumulative one-hot counts
    one_hot = torch.zeros((k_eff, num_groups), dtype=torch.float)
    
    # Handle device mismatch if any
    if groups_t.device != one_hot.device:
        groups_t = groups_t.to(one_hot.device)

    valid = (groups_t >= 0) & (groups_t < num_groups)
    if not bool(valid.all()):
        bad = groups_t[~valid].unique().tolist()
        raise ValueError(f"Found out-of-range group ids: {bad}. Expected 0..{num_groups-1}.")
    
    one_hot.scatter_(1, groups_t.view(-1, 1), 1.0)

    cum_counts = one_hot.cumsum(dim=0)
    denom = torch.arange(1, k_eff + 1, dtype=torch.float).view(-1, 1)
                                        # determine input pi
    pi_hat = cum_counts / denom
                                        # KL per prefix: sum_i p_i * log(p_i / q_i)
                                        # using natural log
    q = pi_t.view(1, -1).expand_as(pi_hat)
    p = pi_hat
    kl_terms = torch.where(p > 0, p * (p / q).log(), torch.zeros_like(p))
    kl_per_k = kl_terms.sum(dim=1)

                                        # discount weights: 1/log2(k+1) for k=1..K.
    weights = 1.0 / torch.log2(torch.arange(2, k_eff + 2, dtype=torch.float))
    z = float(weights.sum().item())
    if z == 0.0 or math.isinf(z) or math.isnan(z):
        raise ValueError("Invalid normalization constant Z.")
                                        # return final NDKL calculation
    return float((weights * kl_per_k).sum().item() / z)


# ============================================================================
# COMPATIBILITY WRAPPERS
# ============================================================================

def build_edge_scores_for_ndkl(h_emb, recs_by_u, predictor=None, device=None, mode='per_user'):
    """
    Build edge scores (or rankings) for NDKL.
    Same logic as before, kept for compatibility if needed.
    """
    if device is None:
        device = h_emb.device
    
    edge_scores = {}
    
    if mode == 'per_user':
        # Just rank 1..K
        for u, vs in recs_by_u.items():
            for rank, v in enumerate(vs):
                edge_scores[(int(u), int(v))] = 1.0 / (rank + 1)
        return edge_scores
    

    return {} 

def compute_ndkl_from_recommendations(recs_by_u, node_categories, target_dist, k_values, edge_scores=None, mode='per_user'):
    """
    Compute NDKL for multiple K values from recommendations.
    """
    scores = {}
    
    if mode == 'per_user':
        ndkl_sums = {k: 0.0 for k in k_values}
        counts = {k: 0 for k in k_values}
        
        for u, vs in recs_by_u.items():
            if not vs:
                continue
                
            # Convert recommended nodes to their group IDs
            group_ids = [node_categories[v].item() for v in vs if v < len(node_categories)]
            
            for k in k_values:
                val = compute_ndkl(
                    group_ids, 
                    target_dist, 
                    k=k, 
                    num_groups=len(target_dist)
                )
                ndkl_sums[k] += val
                counts[k] += 1
                
        # Average
        for k in k_values:
            if counts[k] > 0:
                scores[k] = ndkl_sums[k] / counts[k]
            else:
                scores[k] = 0.0
                
    elif mode == 'global':
        # Need a global ranking of ALL edges
        # Use edge_scores to sort all (u, v) pairs
        if not edge_scores:
            raise ValueError("Global mode requires edge_scores")
            
        sorted_edges = sorted(edge_scores.items(), key=lambda x: x[1], reverse=True)
        # Extract v's groups
        all_group_ids = [node_categories[v].item() for (u, v), s in sorted_edges]
        
        for k in k_values:
            scores[k] = compute_ndkl(
                all_group_ids,
                target_dist,
                k=k,
                num_groups=len(target_dist)
            )
            
    return {'ndkl_scores': scores}


def compute_ndkl_metrics(
    recs_by_u,
    degrees,
    train_edges,
    h_emb,
    predictor=None,
    device=None,
    k_values=None,
    ndkl_mode='per_user',
    node_categories=None,
    target_dist=None
):
    """
    Main entry point for NDKL evaluation.
    """
    if k_values is None:
        k_values = [10, 20, 50, 100]
    
    # Categorize nodes
    if node_categories is None:
        node_categories, _ = categorize_nodes_by_degree(degrees, method='cumulative')
    
    # Target dist
    if target_dist is None:
        target_dist, _ = compute_target_distribution(train_edges, node_categories)
        
    # Build edge scores ONLY if global mode
    edge_scores = None
    if ndkl_mode == 'global':
        edge_scores = {}
        pass

    ndkl_results = compute_ndkl_from_recommendations(
        recs_by_u, node_categories, target_dist,
        k_values=k_values,
        edge_scores=edge_scores,
        mode=ndkl_mode
    )
    
    print_ndkl_results(ndkl_results, k_values, mode=ndkl_mode)
    return ndkl_results
