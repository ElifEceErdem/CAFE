"""
Exposure-Based Reranking for Fairness

Implements a dynamic fairness re-ranking approach that penalizes nodes
based on their cumulative exposure (recommendation count) during the evaluation.
"""

import torch
import numpy as np
from collections import defaultdict
from core.data_loader import build_train_adjacency

class ExposureReRanker:
    """
    Exposure-based fairness re-ranker.
    
    Key Idea:
    - Track global exposure of items across users.
    - Penalize items with high exposure to encourage diversity.
    - Penalty = lambda * log(1 + exposure_count)
    """
    
    def __init__(self, lambda_exposure=0.1):
        """
        Args:
            lambda_exposure: Strength of exposure penalty.
        """
        self.lambda_exposure = lambda_exposure
        # Track global exposure counts: node_id -> count
        self.exposure_counts = defaultdict(int)
        
    def rerank_topk(self, h_emb, train_edges, test_nodes, k, degrees=None, score_fn=None):
        """
        Rerank recommendations based on dynamic exposure penalty.
        
        Args:
            h_emb: Node embeddings [N, D]
            train_edges: Training edges
            test_nodes: Nodes to generate recommendations for
            k: Number of recommendations per node
            degrees: Unused (kept for API compatibility)
            score_fn: Optional scoring function (e.g., from make_predictor_score_fn()).
                      If None, uses dot product.
        
        Returns:
            recs_by_u: dict mapping node_id -> list of top-k recommended nodes
        """
        device = h_emb.device
        n = h_emb.size(0)
        
        # Build adjacency list for common neighbor computation / masking
        train_adj = build_train_adjacency(train_edges, n)
        
        recs_by_u = {}
        
        # We must process sequentially (or semi-sequentially) to update exposure
        # Process in smaller chunks to allow exposure updates to propagate
        chunk_size = 256 
        
        scoring_method = "trained predictor" if score_fn is not None else "dot product"
        print(f"[Exposure] Processing {len(test_nodes)} nodes "
              f"(λ_exposure={self.lambda_exposure}, scoring={scoring_method})...")
        
        for chunk_start in range(0, len(test_nodes), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(test_nodes))
            chunk_test_nodes = test_nodes[chunk_start:chunk_end]
            
            # Compute relevance scores for chunk
            with torch.no_grad():
                test_nodes_tensor = torch.tensor(chunk_test_nodes, dtype=torch.long, device=device)
                h_test_chunk = h_emb[test_nodes_tensor]
                if score_fn is not None:
                    chunk_scores = score_fn(h_test_chunk, h_emb).cpu().numpy()
                else:
                    chunk_scores = torch.sigmoid(h_test_chunk @ h_emb.t()).cpu().numpy()
                del h_test_chunk
            
            # Process users in chunk
            for i, u in enumerate(chunk_test_nodes):
                scores = chunk_scores[i].copy()
                
                # Mask invalid candidates (train edges & self)
                scores[u] = -1e9
                for v in train_adj[u]:
                    if v < n:
                        scores[v] = -1e9
                
                # Filter to top candidates for efficiency before applying penalty
                # 2000 candidates should be enough
                candidate_k = 2000
                if len(scores) > candidate_k:
                    # argpartition to get indices of top K scores
                    candidate_indices = np.argpartition(scores, -candidate_k)[-candidate_k:]
                else:
                    candidate_indices = np.arange(len(scores))
                
                c_scores = scores[candidate_indices]
                c_nodes = candidate_indices
                
                # Calculate Exposure Penalty
                # Vectorized lookup of exposure counts
                current_exposures = np.array([self.exposure_counts[v] for v in c_nodes])
                
                # Penalty = λ * log(1 + exposure)
                # log1p is natural log(1+x)
                penalties = self.lambda_exposure * np.log1p(current_exposures)
                
                final_scores = c_scores - penalties
                
                # Select Top-K
                topk_local_indices = np.argsort(final_scores)[::-1][:k]
                selected = c_nodes[topk_local_indices]
                
                # Store recommendations
                recs_by_u[u] = selected.tolist()
                
                # Update Global Exposure for selected items
                for v in selected:
                    self.exposure_counts[v] += 1
            
            if (chunk_end % 5000 == 0) or (chunk_end == len(test_nodes)):
                print(f"  Processed {chunk_end}/{len(test_nodes)} nodes...")
                
        print(f"[Exposure] Reranking complete")
        return recs_by_u
