import torch

def distribution_fairness_from_recs(recs_by_u, test_edges, k, num_nodes, fairness_mode):
    """Calculate fairness (1 - Gini coefficient) from recommendation distributions."""
    test_nodes = set(test_edges[:, 0].tolist()) | set(test_edges[:, 1].tolist())
    test_nodes = [u for u in test_nodes if u < num_nodes]
    
    unique_edges = set()
    for u in test_nodes:
        if u not in recs_by_u: continue
        for v in recs_by_u[u][:k]:
            a, b = (u, v) if u < v else (v, u)
            unique_edges.add((a, b))
    
    if not unique_edges: return 1.0
    
    nodes_in_recs = {n for edge in unique_edges for n in edge}
    max_id = min(num_nodes - 1, max(nodes_in_recs))
    degree = torch.zeros(max_id + 1, dtype=torch.float)
    
    if fairness_mode == 'indegree':
        for u in test_nodes:
            if u in recs_by_u:
                for v in recs_by_u[u][:k]:
                    if v <= max_id: degree[v] += 1
    elif fairness_mode == 'outdegree':
        for u in test_nodes:
            if u in recs_by_u and u <= max_id:
                degree[u] += len(recs_by_u[u][:k])
    else:  # 'total'
        for a, b in unique_edges:
            if a <= max_id: degree[a] += 1
            if b <= max_id: degree[b] += 1
    
    nonzero_deg = degree[degree > 0]
    if nonzero_deg.numel() == 0: return 1.0
    
    x_sorted, _ = torch.sort(nonzero_deg.view(-1))
    n = x_sorted.numel()
    s = x_sorted.sum()
    idx = torch.arange(1, n + 1, dtype=x_sorted.dtype, device=x_sorted.device)
    gini = 1.0 + (1.0 / n) - 2.0 * (torch.sum((n + 1 - idx) * x_sorted) / (n * s))
    return float(1.0 - gini)