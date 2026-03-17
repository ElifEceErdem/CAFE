"""
Coreness-based Re-ranking for Structural Fairness

Implements the coreness fairness approach:
- Mobility Boost: Helps low-coreness (periphery) nodes
- Bridge Bonus: Protects cross-community connections

Score = Relevance + λ₁ * (1/Core(v)^α) - λ₂ * CN(u,v)/√(deg_u * deg_v) * I[both_core]
"""

import torch
import numpy as np
from typing import Dict, List
from collections import defaultdict
from scipy import stats
from core.data_loader import build_train_adjacency
from metrics.coreness import compute_coreness


class CorenessReRanker:
    """
    Coreness-based fairness re-ranker.
    
    Key Ideas:
    1. Mobility Boost: Prioritize low-coreness nodes (periphery -> core)
       - Uses Log-Damped, Percentile-Normalized Coreness for stability.
    2. Bridge Bonus: (Replaces Penalty) - Currently Disabled to recover Precision.
    """
    
    def __init__(self, lambda_mobility=0.3, lambda_bridge=0.0, 
                 alpha=1.0, core_threshold=None, score_window=None,
                 use_adaptive=False, adaptive_threshold=0.3,
                 periphery_percentile=0.25,
                 use_precision_aware=False, score_gap_threshold=0.1,
                 boosting_mode='continuous',
                 reranking_approach='boost'):
        """
        Args:
            lambda_mobility: Weight for mobility boost (helping weak nodes)
            lambda_bridge: Weight for bridge penalty (Disabled, default 0.0)
            alpha: Exponent for coreness penalty (higher = more aggressive)
            core_threshold: Threshold for "core" membership (default: median)
            score_window: If set, limits the boosted score to be at most (original + window)
            use_adaptive: Enable per-query adaptive lambda
            adaptive_threshold: Threshold for periphery ratio to disable fairness
            periphery_percentile: Bottom percentile of coreness to consider as periphery (default: 0.25 = bottom 25%)
            use_precision_aware: Enable precision-aware adaptive (skip boosting if clear winner)
            score_gap_threshold: Score gap threshold δ for precision-aware logic (default: 0.1)
            boosting_mode: Boosting mode: 'continuous' (default) or 'tiered'
            reranking_approach: 'boost' (promote periphery, default) or 'penalty' (suppress core)
        """
        self.lambda_mobility = lambda_mobility
        self.lambda_bridge = lambda_bridge
        self.alpha = alpha
        self.core_threshold = core_threshold
        self.score_window = score_window
        self.use_adaptive = use_adaptive
        self.adaptive_threshold = adaptive_threshold
        self.periphery_percentile = periphery_percentile
        self.use_precision_aware = use_precision_aware
        self.score_gap_threshold = score_gap_threshold
        self.boosting_mode = boosting_mode
        self.reranking_approach = reranking_approach
        
    def rerank_topk(self, h_emb, train_edges, test_nodes, k, degrees, score_fn=None):
        """
        Coreness-based re-ranking.
        
        Args:
            h_emb: Node embeddings [N, D]
            train_edges: Training edges
            test_nodes: Nodes to generate recommendations for
            k: Number of recommendations per node
            degrees: Node degrees [N] (used for bridge detection)
            score_fn: Optional scoring function (e.g., from make_predictor_score_fn()).
                      If None, uses dot product.
        
        Returns:
            recs_by_u: dict mapping node_id -> list of top-k recommended nodes
        """
        device = h_emb.device
        n = h_emb.size(0)
        
        # Convert degrees to numpy
        if torch.is_tensor(degrees):
            degrees_np = degrees.cpu().numpy()
        else:
            degrees_np = np.array(degrees)
        
        print(f"\n[Coreness] Computing coreness scores...")
        # Compute coreness
        coreness = compute_coreness(train_edges, n)
        coreness_np = coreness.cpu().numpy()
        
        # --- Percentile Normalization ---
        # Rank coreness scores (0 to N-1)
        # dense strategy assigns same rank to ties, but we want percentiles
        # average strategy is good.
        try:
            ranks = stats.rankdata(coreness_np, method='average')
            # Normalize to [0, 1]
            coreness_percentile = ranks / len(coreness_np)
        except Exception as e:
            print(f"[Coreness] Warning: Scipy stats rankdata failed ({e}), falling back to min-max norm")
            c_min, c_max = coreness_np.min(), coreness_np.max()
            if c_max > c_min:
                coreness_percentile = (coreness_np - c_min) / (c_max - c_min)
            else:
                coreness_percentile = np.zeros_like(coreness_np)
        
        # Set core threshold (median if not specified)
        if self.core_threshold is None:
            self.core_threshold = np.median(coreness_np)
            print(f"[Coreness] Using median threshold: τ = {self.core_threshold:.2f}")
        
        # Build sparse CSR mask for vectorized masking
        import scipy.sparse as sp
        edges_cpu = train_edges.cpu().numpy()
        rows = np.concatenate([edges_cpu[0], edges_cpu[1]])
        cols = np.concatenate([edges_cpu[1], edges_cpu[0]])
        valid_mask = (rows < n) & (cols < n)
        rows, cols = rows[valid_mask], cols[valid_mask]
        train_mask_csr = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(n, n)
        )

        recs_by_u = {}

        # INCREASED from 1024 to speed up large batch processing if memory allows, but sticking to safe default
        import sys
        from tqdm import tqdm
        chunk_size = 1024
        scoring_method = "trained predictor" if score_fn is not None else "dot product"
        num_chunks = (len(test_nodes) + chunk_size - 1) // chunk_size
        print(f"[Coreness] λ_mob={self.lambda_mobility}, λ_bridge={self.lambda_bridge}, α={self.alpha}, scoring={scoring_method}")

        for chunk_start in tqdm(range(0, len(test_nodes), chunk_size), total=num_chunks,
                                desc="Coreness rerank", unit="chunk", file=sys.stdout):
            chunk_end = min(chunk_start + chunk_size, len(test_nodes))
            chunk_test_nodes = test_nodes[chunk_start:chunk_end]

            # Compute relevance scores
            # Use torch.no_grad() for inference to save memory
            with torch.no_grad():
                test_nodes_tensor = torch.tensor(chunk_test_nodes, dtype=torch.long, device=device)
                h_test_chunk = h_emb[test_nodes_tensor]
                if score_fn is not None:
                    chunk_scores = score_fn(h_test_chunk, h_emb).cpu().numpy()
                else:
                    chunk_scores = torch.sigmoid(h_test_chunk @ h_emb.t()).cpu().numpy()
                del h_test_chunk

            # Vectorized masking for entire chunk
            chunk_arr = np.array(chunk_test_nodes)
            chunk_scores[range(len(chunk_test_nodes)), chunk_arr] = -1e9
            mask_block = train_mask_csr[chunk_arr]
            mask_rows, mask_cols = mask_block.nonzero()
            chunk_scores[mask_rows, mask_cols] = -1e9

            for i, u in enumerate(chunk_test_nodes):
                scores = chunk_scores[i].copy()
                
                # Compute fairness adjustments
                # Candidate Filtering: Increased to 2000 to avoid filtering top-K items
                # This fixes the lambda=0 discrepancy with baseline
                candidate_k = 2000  
                
                if len(scores) > candidate_k:
                    # argpartition is linear time
                    candidate_indices = np.argpartition(scores, -candidate_k)[-candidate_k:]
                else:
                    candidate_indices = np.arange(len(scores))
                
                # Robustness: Check if we have valid candidates
                valid_mask = scores[candidate_indices] > -1e8
                if not np.any(valid_mask):
                    recs_by_u[u] = []
                    continue
                    
                # Filter candidates
                # Filter candidates
                c_scores = scores[candidate_indices]
                c_nodes = candidate_indices
                
                # --- PRECISION-AWARE LOGIC ---
                skip_fairness = False
                score_gap = 0.0
                
                if self.use_precision_aware:
                    # Sort candidate scores to find top items
                    sorted_scores = np.sort(c_scores)[::-1]  # Descending
                    if len(sorted_scores) >= 2:
                        # Gap between #1 and middle item (or #2 for small pools)
                        mid_idx = min(len(sorted_scores) // 2, len(sorted_scores) - 1)
                        score_gap = sorted_scores[0] - sorted_scores[mid_idx]
                        
                        if score_gap > self.score_gap_threshold:
                            skip_fairness = True
                
                # --- ADAPTIVE FAIRNESS LOGIC ---
                effective_lambda = self.lambda_mobility
                periphery_ratio = 0.0  # Default for logging
                
                if self.use_adaptive:
                    # Calculate periphery ratio in candidate pool
                    # Using percentile-based definition: bottom X% of coreness are periphery
                    c_core_vals = coreness_np[c_nodes]
                    periphery_coreness_threshold = np.percentile(coreness_np, self.periphery_percentile * 100)
                    periphery_count = np.sum(c_core_vals <= periphery_coreness_threshold)
                    periphery_ratio = periphery_count / len(c_nodes)
                    
                    if periphery_ratio > self.adaptive_threshold:
                        # Already diverse -> No fairness needed
                        skip_fairness = True
                    else:
                        # Scale lambda: factor goes from 1.0 (at ratio=0) to 0.0 (at ratio=threshold)
                        factor = max(0.0, 1.0 - (periphery_ratio / self.adaptive_threshold))
                        effective_lambda = self.lambda_mobility * factor
                
                # Apply skip if precision-aware or adaptive triggered it
                if skip_fairness:
                    effective_lambda = 0.0
                
                # --- Mobility Boost ---
                mobility_boost = None
                
                if self.boosting_mode == 'tiered':
                    # Tiered Boosting Logic
                    # Coreness <= 1: Boost 1.5
                    # Coreness [2, 3]: Boost 1.2
                    # Coreness [4, 5]: Boost 1.1
                    # Coreness > 5: Boost 1.0 (no change)
                    
                    # We need raw coreness values for this, not percentiles
                    c_vals = coreness_np[c_nodes]
                    mobility_boost = np.ones_like(c_vals, dtype=np.float32)
                    
                    mobility_boost[c_vals <= 1] = 1.5
                    mobility_boost[(c_vals >= 2) & (c_vals <= 3)] = 1.2
                    mobility_boost[(c_vals >= 4) & (c_vals <= 5)] = 1.1
                    
                    # For tiered, the lambda_mobility acts as a global on/off or scaler?
                    # "Multiplier = 1 + lambda * Boost" structure suggests Boost is the "magnitude".
                    # But the tiered values (1.5, 1.2) look like Multipliers themselves.
                    # If tiered values are Multipliers, then we should probably set lambda=1.0 effectively,
                    # OR we treat them as (1 + boost_val).
                    # Let's assume the tiered values are the TARGET MULTIPLIERS.
                    # So Boost = (TierValue - 1).
                    # Then Multiplier = 1 + lambda * (TierValue - 1).
                    # If lambda=1, we get TierValue. If lambda=0, we get 1.
                    
                    # Adjusting to be compatible with the formula: multiplier = (1.0 + effective_lambda * mobility_boost)
                    # So mobility_boost should be (TierValue - 1.0)
                    mobility_boost[c_vals <= 1] = 0.5   # Target 1.5
                    mobility_boost[(c_vals >= 2) & (c_vals <= 3)] = 0.2 # Target 1.2
                    mobility_boost[(c_vals >= 4) & (c_vals <= 5)] = 0.1 # Target 1.1
                    mobility_boost[c_vals > 5] = 0.0    # Target 1.0
                    
                else:
                    # Continuous (Log-Damped + Normalized)
                    # M(v) = 1 / log(percentile * scaling_factor + e)^alpha
                    scaling_factor = 100.0 
                    c_percentiles = coreness_percentile[c_nodes]
                    
                    # Log-damping: ln(x + e) ensures min value is 1 (log(e)=1), avoiding div by zero or negative
                    denom = np.log(c_percentiles * scaling_factor + np.e)
                    mobility_boost = 1.0 / (denom ** self.alpha)
                
                # --- Bridge Penalty Removed ---
                # Set to 0.0 to recover precision
                bridge_penalty = 0.0
                
                # --- Final Score Computation ---
                if self.reranking_approach == 'penalty':
                    # PENALTY mode: suppress scores for core nodes
                    # final_score = score / (1 + λ * core_penalty)
                    # where core_penalty = (Core(v))^α / max_core^α
                    c_vals = coreness_np[c_nodes]
                    max_core = max(coreness_np.max(), 1.0)
                    core_penalty = (c_vals / max_core) ** self.alpha
                    divisor = (1.0 + effective_lambda * core_penalty)
                    divisor = np.clip(divisor, 0.2, 5.0)  # Safety
                    final_scores = c_scores / divisor
                else:
                    # BOOST mode (default): promote periphery nodes
                    # Final score: Multiplicative Boosting
                    # Score * (1 + lambda_mob * Boost)
                    multiplier = (1.0 + effective_lambda * mobility_boost)
                    
                    # Safety: Clip multiplier to prevent explosion
                    multiplier = np.clip(multiplier, 1.0, 5.0) 
                    
                    final_scores = c_scores * multiplier
                
                # --- Score Window Constraint ---
                if self.score_window is not None:
                    # Constraint: New score cannot exceed original score + window
                    # Ensures we don't displace items that are much more relevant
                    final_scores = np.minimum(final_scores, c_scores + self.score_window)
                
                # Diagnostics (First user only)
                if chunk_start == 0 and i == 0:
                    print(f"\n[Coreness Debug] User {u} (Coreness={coreness_np[u]:.1f}, Pctl={coreness_percentile[u]:.2f}):")
                    print(f"  Candidates: {len(c_nodes)}")
                    if mobility_boost is not None:
                        print(f"  Mobility Boost: min={mobility_boost.min():.3f}, max={mobility_boost.max():.3f}")
                    if self.reranking_approach == 'penalty':
                        print(f"  Divisor: min={divisor.min():.3f}, max={divisor.max():.3f}")
                    else:
                        print(f"  Multiplier: min={multiplier.min():.3f}, max={multiplier.max():.3f}")
                
                # Select Top-K from Candidates
                topk_local_indices = np.argsort(final_scores)[::-1]
                
                selected = []
                for l_idx in topk_local_indices:
                    global_idx = c_nodes[l_idx]
                    if scores[global_idx] <= -1e8:
                        break
                    selected.append(int(global_idx))
                    if len(selected) >= k:
                        break
                
                recs_by_u[u] = selected
            
            pass  # tqdm handles progress
        
        print(f"[Coreness] Re-ranking complete")
        
        return recs_by_u