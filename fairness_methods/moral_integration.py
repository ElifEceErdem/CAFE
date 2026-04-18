"""
MORAL Integration Module for Fairness Experiments

Integrates the faithful MORAL model with the paper's greedy KL-divergence
ranking aggregation (Algorithm 1, Appendix F).

Core pipeline (matching /var/scratch/eerdem/MORAL/main.py):
  1. Train per-group encoder/predictor pairs via MORAL.fit()
  2. Score all test edges via MORAL.predict()
  3. Compute target distribution phi from training graph edge-type proportions
  4. Generate position->group assignments via greedy DKL minimisation
  5. For each group, sort test edges by score (descending)
  6. Interleave top edges into the final ranking by position assignments

Naming convention:
  - NDKL   = sensitive-attribute-based NDKL (core, used for reranking)
  - deg_NDKL = degree-based NDKL (secondary metric, kept for reporting)
"""

import numpy as np
import torch
from typing import Dict, Tuple
from baselines.fairness_aware_baseline_models import train_moral
from core.evaluation import compute_hits_mrr_from_scores
from metrics.precision import compute_metrics_from_recs


# ─────────────────────────────────────────────────────────────────────
# Greedy KL-divergence position assignment (Algorithm 1)
# ─────────────────────────────────────────────────────────────────────

def generate_array_greedy_dkl(n: int, distribution: np.ndarray) -> torch.Tensor:
    """Generate a length-n sequence of group indices whose prefix
    proportions approximate *distribution* at every prefix.

    At each position the group with the largest deficit
    (desired − actual count) is chosen, which greedily minimises
    the KL divergence between the empirical and target distribution
    for every prefix.

    Args:
        n: length of the output sequence
        distribution: target probability vector (sums to 1, length G)

    Returns:
        Tensor of shape [n] with values in {0, …, G-1}
    """
    actual_counts = np.zeros_like(distribution)
    result = []

    for i in range(n):
        if i == 0:
            choice = int(np.argmax(distribution))
        else:
            desired = distribution * i
            deficit = desired - actual_counts
            choice = int(np.argmax(deficit))
        result.append(choice)
        actual_counts[choice] += 1

    return torch.tensor(result)


# ─────────────────────────────────────────────────────────────────────
# Target distribution from training graph
# ─────────────────────────────────────────────────────────────────────

_NUM_SENS_GROUPS = 3  # sens[u]+sens[v] 


def compute_sens_target_distribution(
    edge_index: torch.Tensor,
    sens: torch.Tensor,
) -> np.ndarray:
    """Compute phi -- the proportion of each edge-type in the training graph.

    Edge type = sens[src] + sens[dst]  

    Args:
        edge_index: [2, E] training edge index
        sens: [N] sensitive attribute (binary)

    Returns:
        np.ndarray of shape [3] summing to 1
    """
    src, dst = edge_index
    edge_groups = (sens[src] + sens[dst]).long()
    counts = torch.bincount(edge_groups, minlength=_NUM_SENS_GROUPS).float()
    pi = (counts / counts.sum().clamp(min=1.0)).cpu().numpy()
    return pi


# ─────────────────────────────────────────────────────────────────────
# Greedy KL Reranking  (main.py:140-167)
# ─────────────────────────────────────────────────────────────────────

def greedy_kl_rerank(
    scores: torch.Tensor,
    test_edges: torch.Tensor,
    test_labels: torch.Tensor,
    sens: torch.Tensor,
    pi: np.ndarray,
    K: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply greedy KL-divergence reranking to test-edge scores.

    For each of the K positions:
      - The position is assigned a group via generate_array_greedy_dkl
      - The highest-scoring remaining edge from that group fills the slot

    Args:
        scores: [E] logits from MORAL.predict()
        test_edges: [E, 2] test edge node pairs
        test_labels: [E] binary labels (1=positive, 0=negative)
        sens: [N] sensitive attributes
        pi: [3] target distribution from compute_sens_target_distribution
        K: length of the final ranking

    Returns:
        (final_scores [K], final_labels [K])
    """
    scores = scores.cpu()
    test_labels = test_labels.cpu()
    sens_cpu = sens.cpu()

    edge_sens_groups = sens_cpu[test_edges.cpu()].sum(dim=1).long()  # [E]
    output_positions = generate_array_greedy_dkl(K, pi)             # [K]

    final_scores = torch.zeros(K)
    final_labels = torch.zeros(K)

    for sens_value in range(_NUM_SENS_GROUPS):
        pos_mask = output_positions == sens_value          # positions for this group
        if pos_mask.sum() == 0:
            continue

        group_mask = edge_sens_groups == sens_value         # edges in this group
        if group_mask.sum() == 0:
            continue

        group_scores, indices = scores[group_mask].sort(descending=True)
        group_labels = test_labels[group_mask][indices]

        n_fill = min(int(pos_mask.sum()), len(group_scores))
        # Use explicit index assignment -- chained indexing
        # (tensor[bool_mask][:n] = ...) writes to a copy, not the original.
        pos_indices = torch.where(pos_mask)[0][:n_fill]
        final_scores[pos_indices] = group_scores[:n_fill]
        final_labels[pos_indices] = group_labels[:n_fill]

    return final_scores, final_labels


# ─────────────────────────────────────────────────────────────────────
# Sens-based NDKL (reported as "NDKL")
# ─────────────────────────────────────────────────────────────────────

def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) with zero-safe guards."""
    p = np.clip(p, 1e-10, None)
    q = np.clip(q, 1e-10, None)
    return float(np.sum(p * np.log(p / q)))


def compute_sens_ndkl(
    ranked_edges: torch.Tensor,
    sens: torch.Tensor,
    pi: np.ndarray,
    k_values: list = None,
) -> Dict[str, float]:
    """Compute NDKL -- the normalised discounted KL divergence between
    the prefix-distribution of edge groups in a ranking and the target phi.

    This is the sens-attr-based version (the paper's core metric).

    Args:
        ranked_edges: [K, 2] edges in ranking order
        sens: [N] sensitive attribute
        pi: [3] target distribution
        k_values: list of k cutoffs

    Returns:
        dict  e.g. {"NDKL@10": 0.04, "NDKL@50": 0.02, ...}
    """
    if k_values is None:
        k_values = [10, 20, 50, 100]

    sens_cpu = sens.cpu()
    edge_groups = sens_cpu[ranked_edges.cpu()].sum(dim=1).long().numpy()

    results = {}
    for k in k_values:
        k_eff = min(k, len(edge_groups))
        if k_eff == 0:
            results[f"NDKL@{k}"] = 0.0
            continue

        # Discounted KL at each prefix position
        cumcounts = np.zeros(_NUM_SENS_GROUPS)
        total_dkl = 0.0
        Z = 0.0

        for i in range(k_eff):
            g = edge_groups[i]
            cumcounts[g] += 1
            prefix_dist = cumcounts / cumcounts.sum()
            kl = _kl_divergence(prefix_dist, pi)
            discount = 1.0 / np.log2(i + 2)  # i+2 because positions are 1-indexed
            total_dkl += discount * kl
            Z += discount

        results[f"NDKL@{k}"] = total_dkl / Z if Z > 0 else 0.0

    return results


# ─────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────

def compute_moral_metrics(base_model, data, split_edge, args):
    """Train MORAL, apply greedy KL reranking, and evaluate.

    Returns a dict of metrics compatible with the rest of the scaffold:
      { metric_name: (val_score, test_score) }

    Metrics reported:
      - Hits@K, MRR (from raw MORAL scores on pos/neg test edges)
      - NDKL@K (sens-based, core metric)
      - deg_NDKL@K (degree-based, secondary)
      - Precision@K, Distribution_Fairness@K (from per-user recs)
    """
    print("\n" + "=" * 80)
    print("COMPUTING MORAL METRICS")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Train MORAL
    # ------------------------------------------------------------------
    moral, _ = train_moral(
        base_model=base_model,
        data=data,
        split_edge=split_edge,
        args=args,
    )

    # ------------------------------------------------------------------
    # 2. Raw test-edge scoring (for Hits@K / MRR)
    # ------------------------------------------------------------------
    print("\n[MORAL] Scoring test edges...")
    pos_test_edge = split_edge['test']['edge'].to(moral.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(moral.device)

    pos_pred = moral.predict_scores(pos_test_edge).cpu()
    neg_pred = moral.predict_scores(neg_test_edge.view(-1, 2)).cpu()

    hits_mrr = compute_hits_mrr_from_scores(pos_pred, neg_pred)
    results = {}
    results.update(hits_mrr)

    # ------------------------------------------------------------------
    # 3. Greedy KL reranking (the paper's core contribution)
    # ------------------------------------------------------------------
    print("[MORAL] Computing greedy KL-divergence reranking...")

    # Build edge index for phi computation
    train_edge_index = data.edge_index   # [2, E_train]
    sens = moral.sens_cpu

    pi = compute_sens_target_distribution(train_edge_index, sens)
    print(f"  Target distribution phi = {pi}")

    # Get all test edges + labels + scores
    all_test_outputs = moral.predict().cpu()  # uses stored test edges

    test_split = split_edge['test']
    test_edges_all = torch.cat(
        [test_split['edge'], test_split['edge_neg']], dim=0
    ).long()
    test_labels_all = torch.cat([
        torch.ones(test_split['edge'].size(0)),
        torch.zeros(test_split['edge_neg'].size(0)),
    ], dim=0)

    K = min(1000, len(test_labels_all))
    final_scores, final_labels = greedy_kl_rerank(
        all_test_outputs, test_edges_all, test_labels_all, sens, pi, K=K
    )

    # ------------------------------------------------------------------
    # 4. NDKL (sens-based -> "NDKL")
    # ------------------------------------------------------------------
    # Build the reranked edge list for NDKL measurement
    # Sort all test edges by the reranked order
    edge_sens_groups = sens[test_edges_all.cpu()].sum(dim=1).long()
    output_positions = generate_array_greedy_dkl(K, pi)

    # Reconstruct the reranked edge list
    reranked_edges = torch.zeros(K, 2, dtype=torch.long)
    for sens_value in range(_NUM_SENS_GROUPS):
        pos_mask = output_positions == sens_value
        if pos_mask.sum() == 0:
            continue
        group_mask = edge_sens_groups == sens_value
        if group_mask.sum() == 0:
            continue
        group_scores, indices = all_test_outputs[group_mask].sort(descending=True)
        group_edges = test_edges_all[group_mask][indices]
        n_fill = min(int(pos_mask.sum()), len(group_scores))
        pos_indices = torch.where(pos_mask)[0][:n_fill]
        reranked_edges[pos_indices] = group_edges[:n_fill]

    k_values = [10, 20, 50, 100]
    sens_ndkl = compute_sens_ndkl(reranked_edges, sens, pi, k_values=k_values)
    for metric_name, value in sens_ndkl.items():
        results[metric_name] = (0.0, value)
    print(f"  Sens-based NDKL: {sens_ndkl}")

    # ------------------------------------------------------------------
    # 5. Per-user recommendations (computed once, reused for all metrics)
    # ------------------------------------------------------------------
    print("[MORAL] Generating per-user recommendations...")
    num_nodes = data.num_nodes
    test_nodes_set = (set(pos_test_edge[:, 0].cpu().tolist())
                      | set(pos_test_edge[:, 1].cpu().tolist()))
    test_nodes = sorted([u for u in test_nodes_set if u < num_nodes])
    train_edges_dev = split_edge['train']['edge'].to(moral.device)

    recs_by_u = moral.predict_topk(train_edges_dev, test_nodes, k=100)

    # ------------------------------------------------------------------
    # 6. Precision@K, Distribution_Fairness@K (from per-user recs)
    # ------------------------------------------------------------------
    fairness_mode = getattr(args, 'fairness_mode', 'indegree')
    rec_results = compute_metrics_from_recs(
        recs_by_u, pos_test_edge.cpu(), num_nodes, fairness_mode
    )
    results.update(rec_results)

    # ------------------------------------------------------------------
    # 7. deg_NDKL (degree-based, secondary -- reuses recs_by_u)
    # ------------------------------------------------------------------
    try:
        from ndkl_helper import (
            categorize_nodes_by_degree,
            compute_target_distribution as compute_deg_target_dist,
            compute_ndkl_from_recommendations,
        )

        train_edges = split_edge['train']['edge']

        deg = torch.zeros(num_nodes, dtype=torch.float32)
        for u, v in train_edges.cpu().tolist():
            if 0 <= u < num_nodes:
                deg[u] += 1.0
            if 0 <= v < num_nodes:
                deg[v] += 1.0

        node_cat_method = getattr(args, 'node_categorization', 'cumulative')
        node_categories, _ = categorize_nodes_by_degree(deg, method=node_cat_method)
        deg_target_dist, _ = compute_deg_target_dist(train_edges, node_categories)

        # Rank-based scores for degree NDKL
        edge_scores = {}
        for u in recs_by_u:
            for rank, v in enumerate(recs_by_u[u]):
                edge_scores[(u, v)] = 1.0 / (rank + 1)

        deg_ndkl_results = compute_ndkl_from_recommendations(
            recs_by_u, node_categories, deg_target_dist,
            k_values=k_values, edge_scores=edge_scores, mode='per_user'
        )

        if 'ndkl_scores' in deg_ndkl_results:
            for k in k_values:
                if k in deg_ndkl_results['ndkl_scores']:
                    results[f'deg_NDKL@{k}'] = (0.0, deg_ndkl_results['ndkl_scores'][k])
            print(f"  Degree-based NDKL: {deg_ndkl_results.get('ndkl_scores', {})}")

    except ImportError:
        print("  [MORAL] ndkl_helper not available -- skipping deg_NDKL")

    print("\n MORAL metrics computed")
    return results