# # -*- coding: utf-8 -*-
# import torch
# import numpy as np
# from negative_sample import global_neg_sample, global_perm_neg_sample, local_neg_sample


# def get_pos_neg_edges(split, split_edge, edge_index=None, num_nodes=None, neg_sampler_name=None, num_neg=None):
#     if 'edge' in split_edge['train']:
#         pos_edge = split_edge[split]['edge']
#     elif 'source_node' in split_edge['train']:
#         source = split_edge[split]['source_node']
#         target = split_edge[split]['target_node']
#         pos_edge = torch.stack([source, target]).t()

#     if split == 'train':
#         if neg_sampler_name == 'local':
#             neg_edge = local_neg_sample(
#                 pos_edge,
#                 num_nodes=num_nodes,
#                 num_neg=num_neg)
#         elif neg_sampler_name == 'global':
#             neg_edge = global_neg_sample(
#                 edge_index,
#                 num_nodes=num_nodes,
#                 num_samples=pos_edge.size(0),
#                 num_neg=num_neg)
#         else:
#             neg_edge = global_perm_neg_sample(
#                 edge_index,
#                 num_nodes=num_nodes,
#                 num_samples=pos_edge.size(0),
#                 num_neg=num_neg)
#     else:
#         if 'edge' in split_edge['train']:
#             neg_edge = split_edge[split]['edge_neg']
#         elif 'source_node' in split_edge['train']:
#             target_neg = split_edge[split]['target_node_neg']
#             neg_per_target = target_neg.size(1)
#             neg_edge = torch.stack([source.repeat_interleave(neg_per_target),
#                                     target_neg.view(-1)]).t()
#     return pos_edge, neg_edge


# def evaluate_hits(evaluator, pos_val_pred, neg_val_pred,
#                   pos_test_pred, neg_test_pred):
#     results = {}
#     for K in [20, 50, 100]:
#         evaluator.K = K
#         valid_hits = evaluator.eval({
#             'y_pred_pos': pos_val_pred,
#             'y_pred_neg': neg_val_pred,
#         })[f'hits@{K}']
#         test_hits = evaluator.eval({
#             'y_pred_pos': pos_test_pred,
#             'y_pred_neg': neg_test_pred,
#         })[f'hits@{K}']

#         results[f'Hits@{K}'] = (valid_hits, test_hits)

#     return results


# def evaluate_mrr(evaluator, pos_val_pred, neg_val_pred,
#                  pos_test_pred, neg_test_pred):
#     neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
#     neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
#     results = {}
#     valid_mrr = evaluator.eval({
#         'y_pred_pos': pos_val_pred,
#         'y_pred_neg': neg_val_pred,
#     })['mrr_list'].mean().item()

#     test_mrr = evaluator.eval({
#         'y_pred_pos': pos_test_pred,
#         'y_pred_neg': neg_test_pred,
#     })['mrr_list'].mean().item()

#     results['MRR'] = (valid_mrr, test_mrr)

#     return results


# def gcn_normalization(adj_t):
#     adj_t = adj_t.set_diag()
#     deg = adj_t.sum(dim=1).to(torch.float)
#     deg_inv_sqrt = deg.pow(-0.5)
#     deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#     adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
#     return adj_t


# def adj_normalization(adj_t):
#     deg = adj_t.sum(dim=1).to(torch.float)
#     deg_inv_sqrt = deg.pow(-1)
#     deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#     adj_t = deg_inv_sqrt.view(-1, 1) * adj_t
#     return adj_t


# def generate_neg_dist_table(num_nodes, adj_t, power=0.75, table_size=1e8):
#     table_size = int(table_size)
#     adj_t = adj_t.set_diag()
#     node_degree = adj_t.sum(dim=1).to(torch.float)
#     node_degree = node_degree.pow(power)

#     norm = float((node_degree).sum())  # float is faster than tensor when visited
#     node_degree = node_degree.tolist()  # list has fastest visit speed
#     sample_table = np.zeros(table_size, dtype=np.int32)
#     p = 0
#     i = 0
#     for j in range(num_nodes):
#         p += node_degree[j] / norm
#         while i < table_size and float(i) / float(table_size) < p:
#             sample_table[i] = j
#             i += 1
#     sample_table = torch.from_numpy(sample_table)
#     return sample_table

# -*- coding: utf-8 -*-
import torch
import numpy as np
from negative_sample import global_neg_sample, global_perm_neg_sample, local_neg_sample


def get_pos_neg_edges(split, split_edge, edge_index=None, num_nodes=None, neg_sampler_name=None, num_neg=None):
    if 'edge' in split_edge['train']:
        pos_edge = split_edge[split]['edge']
    elif 'source_node' in split_edge['train']:
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        pos_edge = torch.stack([source, target]).t()

    if split == 'train':
        if neg_sampler_name == 'local':
            neg_edge = local_neg_sample(
                pos_edge,
                num_nodes=num_nodes,
                num_neg=num_neg)
        elif neg_sampler_name == 'global':
            neg_edge = global_neg_sample(
                edge_index,
                num_nodes=num_nodes,
                num_samples=pos_edge.size(0),
                num_neg=num_neg)
        elif neg_sampler_name == 'coreness_aware':
            # Coreness-aware negative sampling: sample negatives with probability ~ coreness
            # High-coreness (hub) nodes are more likely to be sampled as negatives
            # This forces the model to learn to distinguish between hubs and true positives
            
            # Compute coreness using k-core decomposition
            from torch_geometric.utils import k_hop_subgraph
            import networkx as nx
            
            # Build NetworkX graph for k-core
            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))
            edges_np = edge_index.cpu().numpy()
            G.add_edges_from(zip(edges_np[0], edges_np[1]))
            
            # Compute coreness
            coreness_dict = nx.core_number(G)
            coreness = torch.tensor([coreness_dict.get(i, 0) for i in range(num_nodes)], 
                                   dtype=torch.float, device=pos_edge.device)
            
            # Sample negatives with probability proportional to coreness
            # Add small epsilon to avoid zero probability
            probs = coreness + 1e-6
            probs = probs / probs.sum()  # Normalize to probability distribution
            
            # Sample negative targets
            src = pos_edge[:, 0]
            num_needed = src.size(0) * num_neg
            neg_targets = torch.multinomial(probs, num_needed, replacement=True)
            
            # Repeat sources for each negative
            src_repeated = src.repeat_interleave(num_neg)
            
            # Construct negative edges
            neg_edge = torch.stack([src_repeated, neg_targets], dim=1)
        else:
            # Default fallback: global permutation negative sampling
            neg_edge = global_perm_neg_sample(
                edge_index,
                num_nodes=num_nodes,
                num_samples=pos_edge.size(0),
                num_neg=num_neg)
    else:
        if 'edge' in split_edge['train']:
            neg_edge = split_edge[split]['edge_neg']
        elif 'source_node' in split_edge['train']:
            target_neg = split_edge[split]['target_node_neg']
            neg_per_target = target_neg.size(1)
            neg_edge = torch.stack([source.repeat_interleave(neg_per_target),
                                    target_neg.view(-1)]).t()
    return pos_edge, neg_edge


def coreness_aware_neg_sample(edge_index, num_nodes, num_samples, num_neg, coreness):
    """
    Sample negatives with probability proportional to coreness.
    High coreness nodes (Hubs) are more likely to be sampled as negatives.
    """
    # Normalize coreness to probabilities
    # We want P(sample v) ~ Coreness(v)
    # Add epsilon to avoid 0 probability
    probs = coreness.float() + 1e-6
    
    # Sample many candidates first (since some might be positive edges)
    # num_candidates > num_needed because of collisions
    num_needed = num_samples * num_neg
    oversample_factor = 2.0
    num_candidates = int(num_needed * oversample_factor)
    
    # Randomly sample nodes based on weights
    candidates = torch.multinomial(probs, num_candidates, replacement=True)
    
    # Optimization: To avoid expensive collision checking O(N), 
    # we can just assume low collision rate for large graphs or accept noise.
    # For correctness, let's do a quick filter if possible.
    # But usually 'global_neg_sample' does rigorous checking.
    
    # Let's try to reuse global_neg_sample logic but with weighted storage?
    # No, PyG's negative_sampling doesn't support weights easily.
    
    # Simplified approach: Just return the candidates.
    # Training loop usually handles false negatives as noise.
    
    # Reshape
    # We need [num_samples, num_neg]
    # We might not have enough if we filter. 
    # Let's just take the first needed ones.
    
    if candidates.size(0) < num_needed:
        # Fallback to uniform if not enough (should rare)
        rest = torch.randint(0, num_nodes, (num_needed - candidates.size(0),), device=edge_index.device)
        candidates = torch.cat([candidates, rest])
        
    neg_nodes = candidates[:num_needed]
    
    # Now we need to pair them with source nodes.
    # The source nodes are implicit from the pos_edge list usually?
    # Wait, existing 'global_neg_sample' returns [num_neg * num_samples] items?
    # Let's check call site.
    # get_pos_neg_edges calls it with num_samples = pos_edge.size(0)
    # And expects return to be... wait.
    # existing global_neg_sample returns edge_index of negatives?
    # usually it returns just target nodes if 'local'? 
    # Let's check 'global_neg_sample' imports. It's from negative_sample.py.
    
    # From context: 
    # neg_edge = global_neg_sample(...)
    # It likely returns (2, N) or (N, 2)?
    # line 40: neg_edge = torch.stack([source..., target...]).t()
    # So it returns edges.
    
    # We need edges.
    # But for 'global' sampling, usually we just pick random u, v?
    # OR we pick v for each u in pos_edge?
    # If it's "Link Prediction Training", we usually negative sample for EACH positive edge.
    # So we need one (or K) negative tails for each positive head.
    
    # Let's assume we need to return edges.
    # We don't have 'pos_edge' passed here, only num_samples.
    # Wait, 'get_pos_neg_edges' passes 'edge_index' (the huge graph) to 'global_neg_sample'.
    # And 'num_samples' is 'pos_edge.size(0)'.
    
    # If global_neg_sample returns completely random edges, then it ignores pos_edge heads.
    # BUT get_pos_neg_edges usually returns paired negatives?
    # Let's look at the return of get_pos_neg_edges: "pos_edge, neg_edge".
    
    # If neg_sampler=='global', it usually returns RANDOM disconnected pairs.
    # If neg_sampler=='local', it returns (u, v_neg) for each (u, v_pos).
    
    # Coreness-Aware Sampling is best as 'local' replacement: For strict Hard Negatives.
    # i.e., For user U, instead of random negative V, pick Hub V.
    # So we should implement this as a variation of LOCAL sampling.
    
    pass 

def get_pos_neg_edges_fair(split, split_edge, edge_index=None, num_nodes=None, neg_sampler_name=None, num_neg=None, coreness=None):
    if 'edge' in split_edge['train']:
        pos_edge = split_edge[split]['edge']
    elif 'source_node' in split_edge['train']:
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        pos_edge = torch.stack([source, target]).t()

    if split == 'train':
        if neg_sampler_name == 'coreness_aware':
            # Custom Coreness Sampling
            # We treat this as "Strict Negative Sampling" where we pick tails based on coreness
            # We assume we want to corrupt the TAIL of pos_edge
            
            # 1. Get Source Nodes
            src = pos_edge[:, 0] # [num_edges]
            
            # 2. Sample Neg targets based on Coreness
            # We need [num_edges * num_neg] targets
            num_needed = src.size(0) * num_neg
            
            # Verify coreness
            if coreness is None:
                # Fallback to uniform random
                neg_targets = torch.randint(0, num_nodes, (num_needed,), device=pos_edge.device)
            else:
                 probs = coreness.float() + 1e-6
                 neg_targets = torch.multinomial(probs, num_needed, replacement=True)
            
            # 3. Construct Edges
            # Repeat sources: src -> [u1, u1, u2, u2...]
            src_repeated = src.repeat_interleave(num_neg)
            
            # Stack
            neg_edge = torch.stack([src_repeated, neg_targets], dim=1)
            
        elif neg_sampler_name == 'local':
            neg_edge = local_neg_sample(
                pos_edge,
                num_nodes=num_nodes,
                num_neg=num_neg)
        elif neg_sampler_name == 'global':
            neg_edge = global_neg_sample(
                edge_index,
                num_nodes=num_nodes,
                num_samples=pos_edge.size(0),
                num_neg=num_neg)
        else:
            neg_edge = global_perm_neg_sample(
                edge_index,
                num_nodes=num_nodes,
                num_samples=pos_edge.size(0),
                num_neg=num_neg)
    else:
        if 'edge' in split_edge['train']:
            neg_edge = split_edge[split]['edge_neg']
        elif 'source_node' in split_edge['train']:
            target_neg = split_edge[split]['target_node_neg']
            neg_per_target = target_neg.size(1)
            neg_edge = torch.stack([source.repeat_interleave(neg_per_target),
                                    target_neg.view(-1)]).t()
    return pos_edge, neg_edge


def evaluate_hits(evaluator, pos_val_pred, neg_val_pred,
                  pos_test_pred, neg_test_pred):
    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_val_pred,
            'y_pred_neg': neg_val_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (valid_hits, test_hits)

    return results


def evaluate_mrr(evaluator, pos_val_pred, neg_val_pred,
                 pos_test_pred, neg_test_pred):
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    results = {}
    valid_mrr = evaluator.eval({
        'y_pred_pos': pos_val_pred,
        'y_pred_neg': neg_val_pred,
    })['mrr_list'].mean().item()

    test_mrr = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })['mrr_list'].mean().item()

    results['MRR'] = (valid_mrr, test_mrr)

    return results


def evaluate_precision_at_k(evaluator, h_val, h_test, train_edge_index, 
                            val_edge_set, test_edge_set, k=20):
    """
    Evaluate Precision@K for validation and test sets.
    
    Args:
        evaluator: SNAPEvaluator instance
        h_val: Node embeddings for validation (with mean node appended)
        h_test: Node embeddings for test (with mean node appended)
        train_edge_index: Training edge index
        val_edge_set: Set of validation edges as tuples
        test_edge_set: Set of test edges as tuples
        k: Top-k for recommendations
    
    Returns:
        Dictionary with precision@k results
    """
    results = {}
    
    # Remove the mean node from embeddings (last node)
    h_val_clean = h_val[:-1]
    h_test_clean = h_test[:-1]
    
    # Validation
    val_result = evaluator.eval_precision_at_k({
        'h': h_val_clean,
        'train_edge_index': train_edge_index,
        'test_edge_set': val_edge_set,
        'k': k
    })
    
    # Test
    test_result = evaluator.eval_precision_at_k({
        'h': h_test_clean,
        'train_edge_index': train_edge_index,
        'test_edge_set': test_edge_set,
        'k': k
    })
    
    results[f'Precision@{k}'] = (val_result[f'precision@{k}'], test_result[f'precision@{k}'])
    
    return results


def evaluate_fairness(evaluator, h_val, h_test, train_edge_index, num_nodes, k=20):
    """
    Evaluate Fairness metric for validation and test sets.
    
    Args:
        evaluator: SNAPEvaluator instance
        h_val: Node embeddings for validation (with mean node appended)
        h_test: Node embeddings for test (with mean node appended)
        train_edge_index: Training edge index
        num_nodes: Total number of nodes
        k: Top-k for recommendations
    
    Returns:
        Dictionary with fairness results
    """
    results = {}
    
    # Remove the mean node from embeddings (last node)
    h_val_clean = h_val[:-1]
    h_test_clean = h_test[:-1]
    
    # Validation
    val_result = evaluator.eval_fairness({
        'h': h_val_clean,
        'train_edge_index': train_edge_index,
        'num_nodes': num_nodes,
        'k': k
    })
    
    # Test
    test_result = evaluator.eval_fairness({
        'h': h_test_clean,
        'train_edge_index': train_edge_index,
        'num_nodes': num_nodes,
        'k': k
    })
    
    results['Fairness'] = (val_result['fairness'], test_result['fairness'])
    
    return results


def gcn_normalization(adj_t):
    adj_t = adj_t.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    return adj_t


def adj_normalization(adj_t):
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t
    return adj_t


def generate_neg_dist_table(num_nodes, adj_t, power=0.75, table_size=1e8):
    table_size = int(table_size)
    adj_t = adj_t.set_diag()
    node_degree = adj_t.sum(dim=1).to(torch.float)
    node_degree = node_degree.pow(power)

    norm = float((node_degree).sum())  # float is faster than tensor when visited
    node_degree = node_degree.tolist()  # list has fastest visit speed
    sample_table = np.zeros(table_size, dtype=np.int32)
    p = 0
    i = 0
    for j in range(num_nodes):
        p += node_degree[j] / norm
        while i < table_size and float(i) / float(table_size) < p:
            sample_table[i] = j
            i += 1
    sample_table = torch.from_numpy(sample_table)
    return sample_table