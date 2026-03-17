"""
PA-Aware Fairness Metrics

Computes preferential attachment (PA) bias metrics by comparing
recommendations with PA-based recommendations.
"""

import torch
import numpy as np


def generate_pa_recommendations(degrees, train_adj, test_nodes, k, num_nodes):
    """
    Generate recommendations purely based on preferential attachment.

    Args:
        degrees: Node degrees [num_nodes]
        train_adj: Training adjacency list (or sparse CSR matrix)
        test_nodes: Nodes to generate recommendations for
        k: Top-k recommendations
        num_nodes: Total number of nodes

    Returns:
        dict: {node_id -> list of top-k nodes by PA score}
    """
    import scipy.sparse as sp
    degrees_cpu = degrees.cpu().float()
    recs_pa = {}

    # Build sparse CSR mask if train_adj is a list (legacy format)
    if isinstance(train_adj, list):
        import numpy as np
        from core.data_loader import build_train_adjacency
        # Convert list-of-sets to CSR - need original edges
        # Build CSR from list-of-sets directly
        rows_list, cols_list = [], []
        for u, neighbors in enumerate(train_adj):
            for v in neighbors:
                rows_list.append(u)
                cols_list.append(v)
        if rows_list:
            train_mask_csr = sp.csr_matrix(
                (np.ones(len(rows_list), dtype=np.float32),
                 (np.array(rows_list), np.array(cols_list))),
                shape=(num_nodes, num_nodes)
            )
        else:
            train_mask_csr = sp.csr_matrix((num_nodes, num_nodes))
    else:
        train_mask_csr = train_adj  # Already CSR

    from tqdm import tqdm
    chunk_size = 1024
    num_chunks = (len(test_nodes) + chunk_size - 1) // chunk_size

    for chunk_start in tqdm(range(0, len(test_nodes), chunk_size), total=num_chunks,
                            desc="PA recommendations", unit="chunk"):
        chunk_end = min(chunk_start + chunk_size, len(test_nodes))
        chunk_nodes = test_nodes[chunk_start:chunk_end]

        # Vectorized: compute PA scores for entire chunk at once
        chunk_tensor = torch.tensor(chunk_nodes, dtype=torch.long)
        # PA score: deg(u) * deg(v) for all v
        chunk_pa_scores = degrees_cpu[chunk_tensor].unsqueeze(1) * degrees_cpu.unsqueeze(0)  # [chunk, N]

        # Vectorized masking
        import numpy as np
        chunk_arr = np.array(chunk_nodes)
        chunk_pa_scores[range(len(chunk_nodes)), chunk_arr] = float('-inf')
        mask_block = train_mask_csr[chunk_arr]
        mask_rows, mask_cols = mask_block.nonzero()
        chunk_pa_scores[mask_rows, mask_cols] = float('-inf')

        # Batch top-K
        topk_scores, topk_indices = torch.topk(chunk_pa_scores, min(k, num_nodes), dim=1)

        for idx, u in enumerate(chunk_nodes):
            valid_mask = torch.isfinite(topk_scores[idx])
            recs_pa[u] = topk_indices[idx][valid_mask][:k].tolist()

        pass  # tqdm handles progress

    return recs_pa


def compute_overlap_at_k(recs_1, recs_2, k, test_nodes):
    """
    Compute average overlap between two recommendation sets at cutoff k.
    
    Args:
        recs_1: First recommendation dict {node_id -> list}
        recs_2: Second recommendation dict {node_id -> list}
        k: Cutoff value
        test_nodes: Nodes to compute overlap for
    
    Returns:
        float: Average overlap ratio (0 to 1)
    """
    total_overlap = 0.0
    total_nodes = 0
    
    for u in test_nodes:
        if u not in recs_1 or u not in recs_2:
            continue
        
        # Get top-k from each
        topk_1 = set(recs_1[u][:k])
        topk_2 = set(recs_2[u][:k])
        
        # Compute overlap
        if len(topk_1) == 0:
            continue
        
        overlap = len(topk_1 & topk_2)
        overlap_ratio = overlap / k if k > 0 else 0.0
        
        total_overlap += overlap_ratio
        total_nodes += 1
    
    return total_overlap / total_nodes if total_nodes > 0 else 0.0


def compute_pa_fairness_from_overlap(
    recs_baseline,
    recs_postprocessed,
    degrees,
    train_adj,
    test_nodes,
    num_nodes,
    k_values=None,
    normalization='auc'
):
    """
    Compute PA fairness score by comparing overlap with PA recommendations.
    
    Args:
        recs_baseline: Baseline recommendations {node_id -> list}
        recs_postprocessed: Post-processed recommendations {node_id -> list}
        degrees: Node degrees tensor [num_nodes]
        train_adj: Training adjacency list
        test_nodes: Nodes to evaluate
        num_nodes: Total number of nodes
        k_values: List of k values for overlap curve (default: [10, 20, 50, 100])
        normalization: 'auc', 'relative', or 'deviation'
    
    Returns:
        dict: {
            'pa_fairness_score': float in [-1, 1],
            'pa_bias_reduction': float (can be negative if bias increased),
            'overlap_baseline': list of overlap values,
            'overlap_postproc': list of overlap values,
            'k_values': list of k values used,
            'reduction_per_k': list of reduction values,
            'mean_overlap_baseline': float,
            'mean_overlap_postproc': float
        }
    """
    if k_values is None:
        k_values = [10, 20, 50, 100]
    
    print("\n" + "="*80)
    print("COMPUTING PA FAIRNESS VIA OVERLAP CURVES")
    print("="*80)
    
    # Step 1: Generate PA recommendations
    print("Generating PA recommendations...")
    max_k = max(k_values)
    recs_pa = generate_pa_recommendations(
        degrees, train_adj, test_nodes, max_k, num_nodes
    )
    
    # Step 2: Compute overlap curves
    print("\nComputing overlap curves...")
    overlap_baseline = []
    overlap_postproc = []
    
    for k in k_values:
        # Baseline vs PA
        overlap_b = compute_overlap_at_k(recs_pa, recs_baseline, k, test_nodes)
        overlap_baseline.append(overlap_b)
        
        # Post-processed vs PA
        overlap_p = compute_overlap_at_k(recs_pa, recs_postprocessed, k, test_nodes)
        overlap_postproc.append(overlap_p)
        
        reduction = overlap_b - overlap_p
        reduction_pct = (reduction / overlap_b * 100) if overlap_b > 0 else 0.0
        
        print(f"  k={k:>3}: Baseline={overlap_b:.3f}, "
              f"Post-proc={overlap_p:.3f}, "
              f"Reduction={reduction:+.3f} ({reduction_pct:+.1f}%)")
    
    # Step 3: Normalize based on chosen method
    print(f"\nNormalization method: {normalization}")
    
    if normalization == 'auc':
        k_array = np.array(k_values, dtype=float)
        log_k = np.log10(k_array)
        
        overlap_b_array = np.array(overlap_baseline)
        overlap_p_array = np.array(overlap_postproc)
        
        # Expected random overlap: k / num_nodes
        expected_overlap = k_array / num_nodes
        
        # Compute AUCs
        auc_baseline = np.trapz(overlap_b_array, log_k)
        auc_postproc = np.trapz(overlap_p_array, log_k)
        auc_random = np.trapz(expected_overlap, log_k)
        
        print(f"\nAUC Statistics:")
        print(f"  Baseline AUC:      {auc_baseline:.3f}")
        print(f"  Post-proc AUC:     {auc_postproc:.3f}")
        print(f"  Random AUC:        {auc_random:.3f}")
        
        # Compute actual reduction (can be negative)
        actual_reduction = auc_baseline - auc_postproc
        
        if auc_baseline > auc_random:
            max_possible_reduction = auc_baseline - auc_random
            pa_bias_reduction = actual_reduction / max_possible_reduction
            pa_fairness_score = float(np.clip(pa_bias_reduction, -1.0, 1.0))
        else:
            # Baseline already at or below random (rare case)
            pa_bias_reduction = 0.0
            pa_fairness_score = 0.0
    
    elif normalization == 'relative':
        reduction_values = [
            overlap_baseline[i] - overlap_postproc[i] 
            for i in range(len(k_values))
        ]
        mean_reduction = np.mean(reduction_values)
        mean_baseline = np.mean(overlap_baseline)
        
        pa_bias_reduction = mean_reduction / mean_baseline if mean_baseline > 0 else 0.0
        pa_fairness_score = float(np.clip(pa_bias_reduction, -1.0, 1.0))
    
    elif normalization == 'deviation':
        expected_overlaps = [k / num_nodes for k in k_values]
        
        baseline_deviations = [
            overlap_baseline[i] - expected_overlaps[i]
            for i in range(len(k_values))
        ]
        postproc_deviations = [
            overlap_postproc[i] - expected_overlaps[i]
            for i in range(len(k_values))
        ]
        
        reductions = []
        for i in range(len(k_values)):
            if baseline_deviations[i] > 0:
                reduction = (baseline_deviations[i] - postproc_deviations[i]) / baseline_deviations[i]
                reductions.append(reduction)
        
        pa_bias_reduction = np.mean(reductions) if reductions else 0.0
        pa_fairness_score = float(np.clip(pa_bias_reduction, -1.0, 1.0))
    
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")
    
    print(f"\n{'='*80}")
    print(f"PA FAIRNESS SCORE: {pa_fairness_score:.4f}")
    print(f"  (>0 = reduced PA bias, 0 = no change, <0 = increased PA bias)")
    print(f"PA BIAS REDUCTION: {pa_bias_reduction:+.4f}")
    print(f"  (positive = reduced PA bias, negative = increased PA bias)")
    print(f"{'='*80}")
    
    # Compute per-k reductions for detailed analysis
    reduction_per_k = [
        overlap_baseline[i] - overlap_postproc[i]
        for i in range(len(k_values))
    ]
    
    return {
        'pa_fairness_score': float(pa_fairness_score),
        'pa_bias_reduction': float(pa_bias_reduction),
        'overlap_baseline': overlap_baseline,
        'overlap_postproc': overlap_postproc,
        'k_values': k_values,
        'reduction_per_k': reduction_per_k,
        'mean_overlap_baseline': float(np.mean(overlap_baseline)),
        'mean_overlap_postproc': float(np.mean(overlap_postproc)),
    }