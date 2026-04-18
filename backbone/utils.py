import torch
from negative_sample import global_neg_sample, global_perm_neg_sample, local_neg_sample
import networkx as nx

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
        elif neg_sampler_name == 'coreness_aware': # Note: Tried for better results, didnt work 
            # Coreness-aware negative sampling: sample negatives with probability ~ coreness
            # High-coreness (hub) nodes are more likely to be sampled as negatives
            # This forces the model to learn to distinguish between hubs and true positives
            
            # Compute coreness using k-core decomposition
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

