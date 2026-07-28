"""
Coreness-based Re-ranking for Structural Fairness

Implements two re-ranking approaches:
- Boost mode (default): Promotes low-coreness (periphery) nodes via multiplicative boosting
- Penalty mode: Suppresses high-coreness (core) nodes via score division

Score (boost):   final = score * (1 + λ_mob * M(v))
Score (penalty): final = score / (1 + λ_mob * P(v))

where M(v) = 1 / log(percentile(v) * 100 + e)^α  (log-damped mobility boost)
      P(v) = (core(v) / max_core)^α                (normalized core penalty)
"""

import torch
import numpy as np
from scipy import stats
from metrics.coreness import compute_coreness
import scipy.sparse as sp
import sys
from tqdm import tqdm


class CorenessReRanker:
    """
    Coreness-based fairness re-ranker.

    Mobility Boost: Prioritize low-coreness nodes (periphery -> core)
    using Log-Damped, Percentile-Normalized Coreness for stability.
    """

    def __init__(self, lambda_mobility=0.3, lambda_bridge=0.0,
                 alpha=1.0,
                 boosting_mode='continuous',
                 reranking_approach='boost'):
        """
        Args:
            lambda_mobility: Weight for mobility boost (helping weak nodes)
            lambda_bridge: Weight for bridge bonus (currently disabled, kept for interface compatibility)
            alpha: Exponent for coreness boost/penalty (higher = more aggressive)
            boosting_mode: 'continuous' (log-damped, default)
            reranking_approach: 'boost' (promote periphery) or 'penalty' (suppress core)
        """
        self.lambda_mobility = lambda_mobility
        self.lambda_bridge = lambda_bridge
        self.alpha = alpha
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
            degrees: Node degrees [N]
            score_fn: Optional scoring function (e.g., from make_predictor_score_fn()).
                      If None, uses dot product.

        Returns:
            recs_by_u: dict mapping node_id -> list of top-k recommended nodes
        """
        device = h_emb.device
        n = h_emb.size(0)

        print(f"\n[Coreness] Computing coreness scores...")
        coreness = compute_coreness(train_edges, n)
        coreness_np = coreness.cpu().numpy()

        # Percentile normalization
        try:
            ranks = stats.rankdata(coreness_np, method='average')
            coreness_percentile = ranks / len(coreness_np)
        except Exception as e:
            print(f"[Coreness] Warning: rankdata failed ({e}), falling back to min-max norm")
            c_min, c_max = coreness_np.min(), coreness_np.max()
            if c_max > c_min:
                coreness_percentile = (coreness_np - c_min) / (c_max - c_min)
            else:
                coreness_percentile = np.zeros_like(coreness_np)

        # Build sparse CSR mask for vectorized masking
        from core.evaluation import canon_edges_2col
        edges_cpu = canon_edges_2col(train_edges).numpy()
        rows = np.concatenate([edges_cpu[:, 0], edges_cpu[:, 1]])
        cols = np.concatenate([edges_cpu[:, 1], edges_cpu[:, 0]])
        valid_mask = (rows < n) & (cols < n)
        rows, cols = rows[valid_mask], cols[valid_mask]
        train_mask_csr = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(n, n)
        )

        recs_by_u = {}
        chunk_size = 1024
        scoring_method = "trained predictor" if score_fn is not None else "dot product"
        num_chunks = (len(test_nodes) + chunk_size - 1) // chunk_size
        print(f"[Coreness] λ_mob={self.lambda_mobility}, λ_bridge={self.lambda_bridge}, α={self.alpha}, scoring={scoring_method}")

        for chunk_start in tqdm(range(0, len(test_nodes), chunk_size), total=num_chunks,
                                desc="Coreness rerank", unit="chunk", file=sys.stdout):
            chunk_end = min(chunk_start + chunk_size, len(test_nodes))
            chunk_test_nodes = test_nodes[chunk_start:chunk_end]

            # Compute relevance scores
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

                # Candidate filtering (top-2000 by relevance)
                candidate_k = 2000
                if len(scores) > candidate_k:
                    candidate_indices = np.argpartition(scores, -candidate_k)[-candidate_k:]
                else:
                    candidate_indices = np.arange(len(scores))

                # Check for valid candidates
                valid_mask = scores[candidate_indices] > -1e8
                if not np.any(valid_mask):
                    recs_by_u[u] = []
                    continue

                c_scores = scores[candidate_indices]
                c_nodes = candidate_indices

                # Final score computation
                if self.reranking_approach == 'penalty':
                    # Penalty mode: suppress scores for core nodes
                    # final_score = score / (1 + λ * core_penalty)
                    c_vals = coreness_np[c_nodes]
                    max_core = max(coreness_np.max(), 1.0)
                    core_penalty = (c_vals / max_core) ** self.alpha
                    divisor = (1.0 + self.lambda_mobility * core_penalty)
                    divisor = np.clip(divisor, 0.2, 5.0)
                    final_scores = c_scores / divisor
                else:
                    # Boost mode (default): promote periphery nodes
                    # M(v) = 1 / log(percentile * 100 + e)^alpha
                    scaling_factor = 100.0
                    c_percentiles = coreness_percentile[c_nodes]
                    denom = np.log(c_percentiles * scaling_factor + np.e)
                    mobility_boost = 1.0 / (denom ** self.alpha)

                    multiplier = (1.0 + self.lambda_mobility * mobility_boost)
                    multiplier = np.clip(multiplier, 1.0, 5.0)
                    final_scores = c_scores * multiplier

                # Diagnostics (first user only)
                if chunk_start == 0 and i == 0:
                    print(f"\n[Coreness Debug] User {u} (Coreness={coreness_np[u]:.1f}, Pctl={coreness_percentile[u]:.2f}):")
                    print(f"  Candidates: {len(c_nodes)}")
                    if self.reranking_approach == 'penalty':
                        print(f"  Divisor: min={divisor.min():.3f}, max={divisor.max():.3f}")
                    else:
                        print(f"  Mobility Boost: min={mobility_boost.min():.3f}, max={mobility_boost.max():.3f}")
                        print(f"  Multiplier: min={multiplier.min():.3f}, max={multiplier.max():.3f}")

                # Select top-K from candidates
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

        print(f"[Coreness] Re-ranking complete")

        return recs_by_u
