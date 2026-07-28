from metrics.fairness import distribution_fairness_from_recs


def compute_metrics_from_recs(
    recs_by_u, test_edges, num_nodes, fairness_mode, k_values=None
):
    """Compute Precision@K and Fairness@K from recommendations."""
    if k_values is None:
        k_values = [10, 20, 50, 100]

    test_edge_set = set()
    for u, v in test_edges.tolist():
        test_edge_set.add((u, v))
        test_edge_set.add((v, u))  # Undirected
    
    results = {}
    for K in k_values:
        total_precision = 0.0
        total_users = 0
        for u, vs in recs_by_u.items():
            topk = vs[:K]
            if not topk: continue
            hits = sum(1 for v in topk if (u, v) in test_edge_set)
            total_precision += hits / len(topk)
            total_users += 1
        
        prec = total_precision / total_users if total_users > 0 else 0.0
        results[f'Precision@{K}'] = (0.0, prec)
        
        # Call the fairness metric from metrics/fairness.py
        fairness = distribution_fairness_from_recs(
            recs_by_u, test_edges, K, num_nodes, fairness_mode
        )
        results[f'Distribution_Fairness@{K}'] = (0.0, fairness)
    return results
