"""
Dynamic PA-based Re-ranking for Fairness

This module implements the Dynamic PA re-ranker from fairness_pa_bias_clean.py
integrated into the modular framework.
"""

from typing import Dict, List
import torch
import numpy as np
from collections import defaultdict
from core.data_loader import build_train_adjacency


class DynamicPAReRanker:
    """
    Dynamic PA-based reranking that accumulates degree bias penalties.
    
    Core Idea:
    ----------
    Instead of tracking "how many times node v was recommended" (exposure),
    we track "cumulative degree product of recommendations for user u".
    
    This penalizes users who keep recommending high-degree nodes, encouraging
    diversity in the degree distribution of their recommendations.
    
    Modes:
    ------
    1. 'cumulative': Track total PA score accumulated per user
       - Penalizes users who favor hubs repeatedly
       - Fair per-user recommendation diversity
    
    2. 'global': Track total PA score across all users (like MMR exposure)
       - Penalizes globally popular high-degree pairs
       - Fair global exposure
    
    3. 'hybrid': Combination of per-user and global tracking
       - Best of both approaches
    """
    
    def __init__(self, lambda_pa=0.3, accumulation_mode='cumulative', 
                 pa_aggregation='product', temperature=1.0):
        """
        Args:
            lambda_pa: Strength of PA-based penalty (0 = no penalty, 1 = maximum penalty)
            accumulation_mode: How to track PA accumulation
                - 'cumulative': Per-user accumulation
                - 'global': Global accumulation (like MMR)
                - 'hybrid': Both per-user and global
            pa_aggregation: How to compute PA score
                - 'product': sqrt(deg_u * deg_v)
                - 'sum': deg_u + deg_v
                - 'max': max(deg_u, deg_v)
                - 'log_product': log(deg_u * deg_v + 1)
            temperature: Softening parameter for PA scores (higher = softer penalties)
        """
        self.lambda_pa = lambda_pa
        self.accumulation_mode = accumulation_mode
        self.pa_aggregation = pa_aggregation
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = float(temperature)
        
        # Normalization parameters (fitted during reranking)
        self.max_pa_score = None
        
    def _compute_pa_score(self, deg_u, deg_v):
        """
        Compute Preferential Attachment score for edge (u, v).
        
        Args:
            deg_u: Degree of node u (can be scalar or array)
            deg_v: Degree of node v (can be scalar or array)
        
        Returns:
            PA score (normalized to [0, 1])
        """
        if self.pa_aggregation == 'product':
            pa = np.sqrt(deg_u * deg_v)
        elif self.pa_aggregation == 'sum':
            pa = deg_u + deg_v
        elif self.pa_aggregation == 'max':
            pa = np.maximum(deg_u, deg_v)
        elif self.pa_aggregation == 'log_product':
            pa = np.log1p(deg_u * deg_v)
        else:
            raise ValueError(f"Unknown pa_aggregation: {self.pa_aggregation}")
        
        return pa
    
    def rerank_topk(self, h_emb, train_edges, test_nodes, k, degrees, score_callback=None, score_fn=None):
        """
        Dynamic PA-based reranking.
        
        Args:
            h_emb: Node embeddings [N, D] (can be None if score_callback is used)
            train_edges: Training edges to exclude
            test_nodes: Nodes to generate recommendations for
            k: Number of recommendations per node
            degrees: Node degrees [N]
            score_callback: Optional function(users_indices) -> scores_matrix [len(users), N]
                            Used to compute utility scores for non-embedding models.
            score_fn: Optional scoring function (e.g., from make_predictor_score_fn()).
                      If None, uses dot product. Ignored if score_callback is provided.
        
        Returns:
            recs_by_u: dict mapping node_id -> list of top-k recommended nodes
        """
        if h_emb is not None:
            device = h_emb.device
            n = h_emb.size(0)
        else:
            # If h_emb is None, we rely on degrees/score_callback to infer N or device
            if torch.is_tensor(degrees):
                device = degrees.device
                n = degrees.size(0)
            else:
                device = torch.device('cpu')
                n = len(degrees)
        
        # Convert degrees to numpy for faster computation
        if torch.is_tensor(degrees):
            degrees_np = degrees.cpu().numpy()
        else:
            degrees_np = np.array(degrees)
        
        # Compute normalization factor
        self.max_pa_score = self._compute_pa_score(
            degrees_np.max(), degrees_np.max()
        )
        
        # Build sparse CSR mask for vectorized masking
        import scipy.sparse as sp
        edges_cpu = train_edges.cpu().numpy()
        rows_e = np.concatenate([edges_cpu[0], edges_cpu[1]])
        cols_e = np.concatenate([edges_cpu[1], edges_cpu[0]])
        valid_e = (rows_e < n) & (cols_e < n)
        rows_e, cols_e = rows_e[valid_e], cols_e[valid_e]
        train_mask_csr = sp.csr_matrix(
            (np.ones(len(rows_e), dtype=np.float32), (rows_e, cols_e)),
            shape=(n, n)
        )

        # Initialize accumulators based on mode
        if self.accumulation_mode == 'cumulative':
            # Track per-user accumulated PA
            user_pa_accumulated = defaultdict(float)
        elif self.accumulation_mode == 'global':
            # Track global accumulated PA per node (like MMR exposure)
            global_pa_accumulated = np.zeros(n, dtype=np.float32)
        elif self.accumulation_mode == 'hybrid':
            # Track both
            user_pa_accumulated = defaultdict(float)
            global_pa_accumulated = np.zeros(n, dtype=np.float32)
        else:
            raise ValueError(f"Unknown accumulation_mode: {self.accumulation_mode}")
        
        recs_by_u = {}
        
        # Process in chunks to avoid OOM
        import sys
        from tqdm import tqdm
        chunk_size = 1024
        num_chunks = (len(test_nodes) + chunk_size - 1) // chunk_size
        print(f"[DynamicPA] λ={self.lambda_pa}, mode={self.accumulation_mode}, aggregation={self.pa_aggregation}")

        for chunk_start in tqdm(range(0, len(test_nodes), chunk_size), total=num_chunks,
                                desc="PA rerank", unit="chunk", file=sys.stdout):
            chunk_end = min(chunk_start + chunk_size, len(test_nodes))
            chunk_test_nodes = test_nodes[chunk_start:chunk_end]
            
            # Compute relevance scores
            if score_callback is not None:
                # Use callback to get raw scores [Chunk, N]
                chunk_scores = score_callback(chunk_test_nodes, n)
                
                # Robust Normalization to [0, 1] per user
                # This ensures compatibility with FA penalty [0, 1]
                # We use min-max scaling per row
                min_vals = chunk_scores.min(axis=1, keepdims=True)
                max_vals = chunk_scores.max(axis=1, keepdims=True)
                denom = max_vals - min_vals
                denom[denom == 0] = 1.0  # Avoid division by zero
                chunk_scores = (chunk_scores - min_vals) / denom
                
            else:
                # Standard embedding scoring
                test_nodes_tensor = torch.tensor(chunk_test_nodes, dtype=torch.long, device=device)
                h_test_chunk = h_emb[test_nodes_tensor]
                if score_fn is not None:
                    # Use trained predictor (MLP)
                    chunk_scores = score_fn(h_test_chunk, h_emb).cpu().numpy()
                else:
                    # Compute raw dot products (no sigmoid)
                    chunk_scores = (h_test_chunk @ h_emb.t()).cpu().numpy()
                del h_test_chunk  # Free GPU memory
                
                # Apply Min-Max normalization to [0, 1] per user
                # This matches the logic used for score_callback above
                min_vals = chunk_scores.min(axis=1, keepdims=True)
                max_vals = chunk_scores.max(axis=1, keepdims=True)
                
                # DEBUG: Print score stats (embeddings)
                if chunk_start == 0:
                    scoring_method = "trained predictor" if score_fn is not None else "dot product"
                    print(f"[DEBUG] Scoring method: {scoring_method}")
                    print(f"[DEBUG] Scores (Raw) - Min: {min_vals.flatten()[:5]}, Max: {max_vals.flatten()[:5]}")
                    print(f"[DEBUG] Scores (Raw) - Mean: {chunk_scores.mean(axis=1)[:5]}")
                
                denom = max_vals - min_vals
                denom[denom == 0] = 1.0
                chunk_scores = (chunk_scores - min_vals) / denom
            
            # Vectorized masking for entire chunk
            chunk_arr = np.array(chunk_test_nodes)
            chunk_scores[range(len(chunk_test_nodes)), chunk_arr] = -1e9
            mask_block = train_mask_csr[chunk_arr]
            mask_rows, mask_cols = mask_block.nonzero()
            chunk_scores[mask_rows, mask_cols] = -1e9

            # Process each user in the chunk
            for i, u in enumerate(chunk_test_nodes):
                scores = chunk_scores[i].copy()
                
                # Greedy selection with dynamic PA penalty
                selected = []
                deg_u = degrees_np[u]

                # Precompute PA scores for this user once
                deg_candidates = degrees_np
                pa_scores = self._compute_pa_score(deg_u, deg_candidates)
                pa_scores_norm = pa_scores / (self.max_pa_score + 1e-8)

                mean_pa = np.mean(pa_scores_norm)  # used in cumulative / hybrid
                if mean_pa < 1e-12:
                    mean_pa = 1e-12  # avoid tiny expected_accumulated
                
                for step in range(k):
                    # Check if any valid candidates remain
                    valid_mask = scores > -1e8
                    if not valid_mask.any():
                        break
                    
                    # Compute penalty based on accumulation mode
                    if self.accumulation_mode == 'cumulative':
                        # Penalty grows with user's accumulated PA
                        current_accumulated = user_pa_accumulated[u]
                        # Normalize by expected accumulation at this step
                        expected_accumulated = (step + 1) * mean_pa
                        accumulation_weight = current_accumulated / (expected_accumulated + 1e-8)
                        
                        # Dynamic penalty: base PA × accumulation weight
                        pa_penalty = pa_scores_norm * (1.0 + accumulation_weight)
                        
                    elif self.accumulation_mode == 'global':
                        # Penalty grows with global accumulated PA (like MMR)
                        global_weight = global_pa_accumulated / (global_pa_accumulated.max() + 1e-8)
                        pa_penalty = pa_scores_norm * (1.0 + global_weight)
                        
                    elif self.accumulation_mode == 'hybrid':
                        # Combine per-user and global penalties
                        current_accumulated = user_pa_accumulated[u]
                        expected_accumulated = (step + 1) * mean_pa
                        user_weight = current_accumulated / (expected_accumulated + 1e-8)
                        
                        global_weight = global_pa_accumulated / (global_pa_accumulated.max() + 1e-8)
                        
                        # Weighted combination (50-50)
                        combined_weight = 0.5 * user_weight + 0.5 * global_weight
                        pa_penalty = pa_scores_norm * (1.0 + combined_weight)
                    
                    # Apply temperature softening
                    pa_penalty = pa_penalty / self.temperature
                    
                    # ===== COMPUTE FINAL SCORES =====
                    # Score = Relevance - λ × PA_penalty
                    final_scores = scores - self.lambda_pa * pa_penalty
                    
                    # Select best candidate
                    best_idx = np.argmax(final_scores)
                    
                    if scores[best_idx] <= -1e8:  # Invalid candidate
                        break
                    
                    # Add to recommendations
                    selected.append(int(best_idx))
                    
                    # ===== UPDATE ACCUMULATORS =====
                    pa_score_selected = pa_scores_norm[best_idx]
                    
                    if self.accumulation_mode in ['cumulative', 'hybrid']:
                        user_pa_accumulated[u] += pa_score_selected
                    
                    if self.accumulation_mode in ['global', 'hybrid']:
                        global_pa_accumulated[best_idx] += pa_score_selected
                    
                    # Mask selected candidate
                    scores[best_idx] = -1e9
                
                recs_by_u[u] = selected
            
            pass  # tqdm handles progress
        
        # Print final statistics
        if self.accumulation_mode in ['cumulative', 'hybrid']:
            user_accums = list(user_pa_accumulated.values())
            if user_accums:
                print(f"[DynamicPA] User PA accumulation: mean={np.mean(user_accums):.3f}, "
                      f"std={np.std(user_accums):.3f}")
        
        if self.accumulation_mode in ['global', 'hybrid']:
            nonzero_global = global_pa_accumulated[global_pa_accumulated > 0]
            if len(nonzero_global) > 0:
                print(f"[DynamicPA] Global PA accumulation: {len(nonzero_global)} nodes, "
                      f"mean={nonzero_global.mean():.3f}, max={nonzero_global.max():.3f}")
        
        return recs_by_u