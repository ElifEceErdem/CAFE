"""
RA-Guided Negative Sampling Module.

Provides functions to sample "hard negatives" (non-edges with high RA scores)
to improve GNN precision.
"""
import torch
from negative_sample import global_neg_sample

def ra_guided_neg_sample(pos_edge, edge_index, num_nodes, num_neg, ra_candidates, easy_ratio=0.5):
    """
    Sample negatives with a mix of "easy" (random) and "hard" (high RA) negatives.
    
    Args:
        pos_edge: Positive training edges [E, 2]
        edge_index: Full graph edge index [2, Num_Edges] (for collision checking in easy sampling)
        num_nodes: Number of nodes
        num_neg: Number of negatives per positive edge
        ra_candidates: Tensor of shape [N_cand, 2] containing RA hard negative candidates 
                       (pairs with RA > 0 but no edge)
        easy_ratio: Fraction of negatives that should be "easy" (random)
        
    Returns:
        neg_edge: Negative edges [E * num_neg, 2]
    """
    num_samples = pos_edge.size(0)
    total_neg_needed = num_samples * num_neg
    
    num_easy = int(total_neg_needed * easy_ratio)
    num_hard = total_neg_needed - num_easy
    
    # === 1. Sample Easy Negatives ===
    # We use global_neg_sample which handles collision checking.
    # It usually takes (num_samples, num_neg) and produces num_samples * num_neg outputs.
    # We can fake it by asking for num_easy samples directly.
    # But global_neg_sample signature is: (edge_index, num_nodes, num_samples, num_neg)
    # It returns [num_samples, num_neg, 2] reshaped to [num_samples * num_neg, 2]
    # To get exactly 'num_easy' samples, we can ask for (num_easy, 1).
    if num_easy > 0:
        easy_neg = global_neg_sample(
            edge_index,
            num_nodes=num_nodes,
            num_samples=num_easy,
            num_neg=1,
            method='sparse'
        ) # Returns [num_easy, 2]
        
        # Ensure it's [num_easy, 2]
        easy_neg = easy_neg.view(-1, 2)
    else:
        easy_neg = torch.empty((0, 2), dtype=torch.long, device=pos_edge.device)

    # === 2. Sample Hard Negatives ===
    if num_hard > 0:
        if len(ra_candidates) == 0:
             # Fallback if no candidates
             print("WARNING: No RA candidates found. Falling back to random sampling.")
             hard_neg = global_neg_sample(
                edge_index,
                num_nodes=num_nodes,
                num_samples=num_hard,
                num_neg=1, 
                method='sparse'
            ).view(-1, 2)
        else:
            # Randomly sample from candidates with replacement
            indices = torch.randint(0, len(ra_candidates), (num_hard,), device=ra_candidates.device)
            hard_neg = ra_candidates[indices]
            
            # Note: ra_candidates are already verified to be non-edges by construction
    else:
        hard_neg = torch.empty((0, 2), dtype=torch.long, device=pos_edge.device)
        
    # === 3. Combine ===
    all_neg = torch.cat([easy_neg, hard_neg], dim=0)
    
    # Shuffle
    perm = torch.randperm(all_neg.size(0), device=all_neg.device)
    all_neg = all_neg[perm]
    
    return all_neg
