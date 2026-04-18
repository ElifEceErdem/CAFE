"""
Networked Inequality: Preferential Attachment Bias in GNN Link Prediction
=========================================================================

Implementation of the within-group fairness metric and Taylor representation
from:

    Subramonian, Sagun, Sun. "Networked Inequality: Preferential Attachment
    Bias in Graph Neural Network Link Prediction." ICML 2024 (PMLR 235).

Reference implementation: https://github.com/ArjunSubramonian/link_bias_amplification
"""

import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from torch_geometric.utils import (
        add_remaining_self_loops, subgraph, to_dense_adj, coalesce
    )


def within_group_unfairness(social_group_id, edge_index, scores,
                            social_groups, sens_attr):
    """
    Compute within-group unfairness Delta^(b) for a single social group.
    (Eq. 17-19 of the paper)

    Measures the absolute difference in mean post-sigmoid LP scores between
    sensitive subgroups (T^(1) vs T^(2)) within a social group S^(b).

    Args:
        social_group_id: The social group index b.
        edge_index: Edge indices [2, num_edges]. edge_index[0] = source,
                    edge_index[1] = target.
        scores: Pre-sigmoid (raw logit) LP scores [num_edges].
        social_groups: Social group labels per node [num_nodes]. (data.y)
        sens_attr: Binary sensitive attribute per node [num_nodes]. (data.sens)

    Returns:
        Tensor scalar: |mean(sigma(scores for T^(1)->S^(b)))
                       - mean(sigma(scores for T^(2)->S^(b)))|
        or None if either subgroup has no qualifying edges.
    """
    # Nodes in social group b
    c_mask = (social_groups == social_group_id)

    # Nodes in social group b AND sensitive group 0 or 1
    c0_mask = c_mask & (sens_attr == 0)
    c1_mask = c_mask & (sens_attr == 1)

    # Edges where source is in (S^(b) ∩ T^(d)) and target is in S^(b)
    c0a_mask = c0_mask[edge_index[0]] & c_mask[edge_index[1]]
    c1a_mask = c1_mask[edge_index[0]] & c_mask[edge_index[1]]

    if scores[c0a_mask].size(0) == 0 or scores[c1a_mask].size(0) == 0:
        return None

    total_0 = torch.sigmoid(scores[c0a_mask]).mean()
    total_1 = torch.sigmoid(scores[c1a_mask]).mean()

    return torch.abs(total_0 - total_1)


def compute_avg_within_group_unfairness(edge_index, scores,
                                        social_groups, sens_attr):
    """
    Compute the average within-group unfairness across all social groups.
    (1/B) * sum_{b in [B]} Delta^(b)  -- used in the fairness regularizer (Eq. 29).

    Args:
        edge_index: Edge indices [2, num_edges].
        scores: Pre-sigmoid LP scores [num_edges].
        social_groups: Social group labels per node [num_nodes].
        sens_attr: Binary sensitive attribute per node [num_nodes].

    Returns:
        Tensor scalar: average Delta^(b) across valid social groups.
        Returns tensor(0.0) if no valid groups exist.
    """
    num_groups = int(social_groups.max().item()) + 1
    total_unfairness = torch.tensor(0.0, device=scores.device)
    valid_groups = 0

    for b in range(num_groups):
        delta_b = within_group_unfairness(
            b, edge_index, scores, social_groups, sens_attr
        )
        if delta_b is not None:
            total_unfairness = total_unfairness + delta_b
            valid_groups += 1

    if valid_groups == 0:
        return torch.tensor(0.0, device=scores.device)

    return total_unfairness / valid_groups


def compute_within_group_unfairness_metric(model, data, split_edge,
                                           social_groups, sens_attr,
                                           batch_size=65536):
    """
    Compute the within-group unfairness metric on test edges for evaluation.
    (Section 6.2 of the paper)

    Uses post-sigmoid LP scores over the sampled (positive and negative) test
    edges, following the paper's experimental setup.

    Args:
        model: Trained link prediction model with encoder/predictor.
        data: PyG data object (with adj_t, edge_weight, etc.).
        split_edge: Dictionary with train/test edges.
        social_groups: Social group labels [num_nodes].
        sens_attr: Binary sensitive attribute [num_nodes].
        batch_size: Batch size for score computation.

    Returns:
        dict: {
            'WithinGroupUnfairness': (0.0, avg_delta),
            'WithinGroupUnfairness_per_group': {b: delta_b, ...},
            'num_valid_groups': int
        }
    """
    model.encoder.eval()
    model.predictor.eval()

    device = model.device

    # Ensure social_groups and sens_attr are on the same device
    if torch.is_tensor(social_groups):
        social_groups = social_groups.to(device)
    else:
        social_groups = torch.tensor(social_groups, device=device)

    if torch.is_tensor(sens_attr):
        sens_attr = sens_attr.to(device)
    else:
        sens_attr = torch.tensor(sens_attr, dtype=torch.float32, device=device)

    with torch.no_grad():
        # Get embeddings
        input_feat = model.create_input_feat(data)
        h = model.encoder(input_feat, data.adj_t, data.edge_weight)
        mean_h = torch.mean(h, dim=0, keepdim=True)
        h_with_mean = torch.cat([h, mean_h], dim=0)

        # Combine positive and negative test edges
        pos_test_edge = split_edge['test']['edge'].to(device)
        neg_test_edge = split_edge['test']['edge_neg'].to(device)

        all_edges = torch.cat([pos_test_edge, neg_test_edge.view(-1, 2)], dim=0)

        # Compute raw (pre-sigmoid) scores in batches
        all_scores = []
        for start in range(0, all_edges.size(0), batch_size):
            end = min(start + batch_size, all_edges.size(0))
            batch_edges = all_edges[start:end]
            src, dst = batch_edges[:, 0], batch_edges[:, 1]
            # Use predictor to get raw scores (pre-sigmoid)
            score = model.predictor(h_with_mean[src], h_with_mean[dst]).squeeze()
            all_scores.append(score)

        all_scores = torch.cat(all_scores, dim=0)

        # Build edge_index in [2, E] format
        edge_index = all_edges.t()  # [2, num_edges]

        # Compute per-group unfairness
        num_groups = int(social_groups.max().item()) + 1
        per_group = {}
        total_unfairness = 0.0
        valid_groups = 0

        for b in range(num_groups):
            delta_b = within_group_unfairness(
                b, edge_index, all_scores, social_groups, sens_attr
            )
            if delta_b is not None:
                per_group[b] = delta_b.item()
                total_unfairness += delta_b.item()
                valid_groups += 1

        avg_delta = total_unfairness / valid_groups if valid_groups > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"WITHIN-GROUP UNFAIRNESS (Networked Inequality)")
    print(f"{'='*60}")
    print(f"  Valid social groups: {valid_groups}/{num_groups}")
    for b, delta in per_group.items():
        print(f"  Group {b}: Delta^({b}) = {delta:.6f}")
    print(f"  Average Delta = {avg_delta:.6f}")
    print(f"  (lower = fairer, 0 = perfect parity)")
    print(f"{'='*60}")

    return {
        'WithinGroupUnfairness': (0.0, avg_delta),
        'WithinGroupUnfairness_per_group': per_group,
        'num_valid_groups': valid_groups,
    }


# ============================================================================
# Taylor Representation (Section 4.1, 6.1)
# Only meaningful for GCN encoders with inner-product decoders.
# ============================================================================

def compute_taylor_representation(model, data, social_groups, conv_type='sym'):
    """
    Compute the Taylor (theoretic) link prediction representation.
    (Section 4.1 / 6.1 of the paper)

    For GCN with symmetric normalization (Theorem 4.3):
        E[f_LP(s_i, s_j)] ~ C_0 * sqrt(D_hat_ii * D_hat_jj)

    The Taylor representation approximates this by computing, for each
    social group c:
        taylor_rep[c_mask] = eigv_i @ eigv_j^T @ z[c_mask]

    where eigv_i, eigv_j depend on within-group degrees and the
    normalization type.

    NOTE: This analysis is specific to vanilla GCN (GCNConv) with
    inner-product decoder. For other architectures (HL-GNN, SAGE, GAT, MLP
    predictors), results may not be meaningful.

    Args:
        model: Trained GCN model. Must have model.encoder with convolution
               layers that have a `lin.weight` attribute (standard PyG GCNConv).
        data: PyG data object with x, edge_index.
        social_groups: Social group labels [num_nodes]. (data.y)
        conv_type: 'sym' for symmetric normalized or 'rw' for random walk.

    Returns:
        dict: {
            'taylor_rep': Tensor [num_nodes, out_dim],
            'actual_rep': Tensor [num_nodes, out_dim],
            'pcc_per_group': {group_id: pcc_value},
            'nrmse_per_group': {group_id: nrmse_value},
            'overall_pcc': float,
            'overall_nrmse': float
        }
        or None if the model architecture is incompatible.
    """
    model.encoder.eval()
    device = model.device

    if torch.is_tensor(social_groups):
        social_groups = social_groups.to(device)

    # --- Extract GCN linear weights ---
    # Try to find convolution layers with lin.weight
    conv_layers = []
    encoder = model.encoder

    # Try different common attribute patterns for GNN layers
    if hasattr(encoder, 'convs'):
        conv_layers = list(encoder.convs)
    elif hasattr(encoder, 'layers'):
        conv_layers = list(encoder.layers)
    else:
        print("WARNING: Cannot find convolution layers in encoder. "
              "Taylor representation requires a GCN encoder with "
              "accessible conv layers (encoder.convs or encoder.layers).")
        return None

    # Check that layers have lin.weight
    for i, conv in enumerate(conv_layers):
        if not (hasattr(conv, 'lin') and hasattr(conv.lin, 'weight')):
            print(f"WARNING: Conv layer {i} does not have lin.weight. "
                  f"Taylor representation requires standard GCNConv layers.")
            return None

    with torch.no_grad():
        # Step 1: Compute z = x @ W_1^T @ W_2^T @ ... @ W_L^T
        # (linear transformation without message passing or nonlinearities)
        x = data.x.to(device) if data.x is not None else None
        if x is None:
            print("WARNING: Taylor representation requires node features (data.x).")
            return None

        z = x
        for conv in conv_layers:
            z = z @ conv.lin.weight.detach().t()

        # Step 2: Build the Taylor representation per social group
        taylor_rep = torch.zeros_like(z)

        # Edge index with self-loops (matching GCN preprocessing)
        edge_index_sl = add_remaining_self_loops(
            data.edge_index.to(device), num_nodes=data.num_nodes
        )[0]

        num_groups = int(social_groups.max().item()) + 1

        for c in range(num_groups):
            c_mask = torch.nonzero(social_groups.flatten() == c).flatten()

            if c_mask.size(0) == 0:
                continue

            # Extract within-group subgraph (relabel nodes)
            c_edge_index = subgraph(
                c_mask, edge_index_sl,
                relabel_nodes=True, num_nodes=data.num_nodes
            )[0]
            c_edge_index = coalesce(c_edge_index)
            c_section = to_dense_adj(c_edge_index)[0]

            deg_i = c_section.sum(dim=1)  # row sums
            deg_j = c_section.sum(dim=0)  # col sums

            if conv_type == 'sym':
                # Theorem 4.3: eigv = sqrt(deg / sum(deg))
                eigv_i = torch.sqrt(deg_i / deg_i.sum())
                eigv_j = torch.sqrt(deg_j / deg_j.sum())
            else:
                # Theorem 4.4 (random walk): eigv_i = 1, eigv_j = deg/sum
                eigv_i = torch.ones_like(deg_i)
                eigv_j = deg_j / deg_j.sum()

            # Rank-1 approximation: eigv_i @ eigv_j^T @ z[c_mask]
            taylor_rep[c_mask] = (
                eigv_i.reshape(-1, 1) @ eigv_j.reshape(1, -1) @ z[c_mask]
            )

        # Step 3: Compute actual GCN representations
        input_feat = model.create_input_feat(data)
        actual_rep = model.encoder(input_feat, data.adj_t, data.edge_weight)

    return {
        'taylor_rep': taylor_rep,
        'actual_rep': actual_rep,
    }


def validate_taylor_approximation(model, data, split_edge, social_groups,
                                  conv_type='sym'):
    """
    Validate the Taylor approximation against actual GCN predictions.
    (Section 6.1 of the paper)

    For each social group, computes PCC and NRMSE between actual LP scores
    and theoretic (Taylor) LP scores on within-group test edges.

    Args:
        model: Trained GCN model.
        data: PyG data object.
        split_edge: Edge split dictionary.
        social_groups: Social group labels [num_nodes].
        conv_type: 'sym' or 'rw'.

    Returns:
        dict with overall and per-group PCC/NRMSE, or None if incompatible.
    """
    taylor_result = compute_taylor_representation(
        model, data, social_groups, conv_type
    )
    if taylor_result is None:
        return None

    taylor_rep = taylor_result['taylor_rep']
    actual_rep = taylor_result['actual_rep']
    device = model.device

    if torch.is_tensor(social_groups):
        social_groups = social_groups.to(device)

    # Use test edge_label_index (positive + negative)
    pos_test_edge = split_edge['test']['edge'].to(device)
    neg_test_edge = split_edge['test']['edge_neg'].to(device).view(-1, 2)
    all_test_edges = torch.cat([pos_test_edge, neg_test_edge], dim=0)
    edge_label_index = all_test_edges.t()  # [2, E]

    num_groups = int(social_groups.max().item()) + 1
    pcc_per_group = {}
    nrmse_per_group = {}
    all_x, all_y = [], []

    with torch.no_grad():
        for c in range(num_groups):
            c_mask = (social_groups == c)

            # Filter edges where both endpoints are in social group c
            src_in_c = c_mask[edge_label_index[0]]
            dst_in_c = c_mask[edge_label_index[1]]
            c_edge_mask = src_in_c & dst_in_c
            c_edge_label_index = edge_label_index[:, c_edge_mask]

            if c_edge_label_index.size(1) <= 1:
                continue

            # Actual LP scores (inner product)
            x = (actual_rep[c_edge_label_index[0]]
                 * actual_rep[c_edge_label_index[1]]).sum(dim=-1)
            x = x.detach().cpu().numpy()

            # Taylor LP scores (inner product)
            y = (taylor_rep[c_edge_label_index[0]]
                 * taylor_rep[c_edge_label_index[1]]).sum(dim=-1)
            y = y.detach().cpu().numpy()

            if len(x) <= 1:
                continue

            # Scale Taylor scores via least-squares (as in original code)
            rho, _, _, _ = np.linalg.lstsq(
                y[:, np.newaxis], x, rcond=None
            )
            y = y * rho[0]

            all_x.extend(x.tolist())
            all_y.extend(y.tolist())

            # Per-group metrics
            pcc_val, _ = pearsonr(x, y)
            pcc_per_group[c] = pcc_val

            rmse_val = mean_squared_error(x, y, squared=False)
            x_range = x.max() - x.min()
            nrmse_val = rmse_val / x_range if x_range > 0 else 0.0
            nrmse_per_group[c] = nrmse_val

    # Overall metrics
    if len(all_x) > 1:
        overall_pcc, _ = pearsonr(all_x, all_y)
        overall_rmse = mean_squared_error(all_x, all_y, squared=False)
        x_range = max(all_x) - min(all_x)
        overall_nrmse = overall_rmse / x_range if x_range > 0 else 0.0
    else:
        overall_pcc = 0.0
        overall_nrmse = 0.0

    print(f"\n{'='*60}")
    print("TAYLOR APPROXIMATION VALIDATION")
    print(f"{'='*60}")
    for c in sorted(pcc_per_group.keys()):
        print(f"  Group {c}: PCC={pcc_per_group[c]:.4f}, "
              f"NRMSE={nrmse_per_group[c]:.4f}")
    print(f"  Overall:  PCC={overall_pcc:.4f}, NRMSE={overall_nrmse:.4f}")
    print(f"{'='*60}")

    return {
        'Taylor_PCC': (0.0, overall_pcc),
        'Taylor_NRMSE': (0.0, overall_nrmse),
        'pcc_per_group': pcc_per_group,
        'nrmse_per_group': nrmse_per_group,
    }
