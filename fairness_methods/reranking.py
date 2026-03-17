# from typing import Dict, List, Optional
# import torch
# import sys
# from collections import defaultdict
# from tqdm import tqdm
# import os

# # Path Fix: Navigate to SNAP folder to find model.py
# current_dir = os.path.dirname(os.path.abspath(__file__))
# snap_path = os.path.abspath(os.path.join(current_dir, '../../SNAP'))
# if snap_path not in sys.path:
#     sys.path.append(snap_path)


# class DiversityReRanker:
#     """Re-rank recommendations for fairness while maintaining precision."""
    
#     def __init__(self, method='mmr', lambda_div=0.3, exposure_cap=None):
#         """
#         Args:
#             method: 'mmr' or 'exposure'
#             lambda_div: Diversity strength (0 = no diversity, 1 = maximum diversity)
#             exposure_cap: Maximum recommendations per node (for exposure method)
#         """
#         self.method = method
#         self.lambda_div = lambda_div
#         self.exposure_cap = exposure_cap
    
#     def rerank_topk(self, h, train_edges, test_nodes, k, degrees):
#         """
#         Re-rank recommendations for fairness.
        
#         Args:
#             h: Node embeddings [N, D]
#             train_edges: Training edges to exclude
#             test_nodes: Nodes to generate recommendations for
#             k: Number of recommendations per node
#             degrees: Node degrees (currently unused, kept for API consistency)
        
#         Returns:
#             recs_by_u: dict mapping node_id -> list of top-k recommended nodes
#         """
#         device = h.device
#         n = h.size(0)
#         train_adj = self._build_adjacency(train_edges, n)
        
#         if self.method == 'mmr':
#             return self._rerank_mmr(h, train_adj, test_nodes, k)
#         elif self.method == 'exposure':
#             return self._rerank_exposure(h, train_adj, test_nodes, k, n)
#         else:
#             raise ValueError(f"Unknown method: {self.method}")
    
#     def _build_adjacency(self, train_edges, n):
#         """Build adjacency list from training edges."""
#         train_adj = [set() for _ in range(n)]
#         train_edges_cpu = train_edges.cpu()
#         for u, v in train_edges_cpu.tolist():
#             if u < n and v < n:
#                 train_adj[u].add(v)
#                 train_adj[v].add(u)
#         return train_adj
    
#     def _rerank_mmr(self, h, train_adj, test_nodes, k):
#         """
#         Maximal Marginal Relevance re-ranking.
        
#         Balances relevance (model score) with diversity (exposure balance).
#         """
#         device = h.device
#         n = h.size(0)
#         exposure = torch.zeros(n, device='cpu')
#         recs_by_u = {}
        
#         # Process in chunks to avoid OOM
#         chunk_size = 1024
#         num_chunks = (len(test_nodes) + chunk_size - 1) // chunk_size
        
#         print(f"\n🔄 MMR Re-ranking: Processing {len(test_nodes)} nodes in {num_chunks} chunks...")
        
#         # Create progress bar for chunks
#         chunk_iterator = range(0, len(test_nodes), chunk_size)
#         pbar = tqdm(chunk_iterator, desc="MMR Re-ranking", unit="chunk", 
#                    total=num_chunks)
        
#         for chunk_start in pbar:
#             chunk_end = min(chunk_start + chunk_size, len(test_nodes))
#             chunk_test_nodes = test_nodes[chunk_start:chunk_end]
            
#             # Compute scores for this chunk
#             test_nodes_tensor = torch.tensor(chunk_test_nodes, dtype=torch.long, device=device)
#             h_test_chunk = h[test_nodes_tensor]
#             chunk_scores = torch.sigmoid(h_test_chunk @ h.t()).cpu()
#             del h_test_chunk  # Free GPU memory
            
#             # Process each user in the chunk
#             for i, u in enumerate(chunk_test_nodes):
#                 scores = chunk_scores[i].clone()
                
#                 # Mask invalid candidates (self-loops and training edges)
#                 scores[u] = float('-inf')
#                 for v in train_adj[u]:
#                     if v < n:
#                         scores[v] = float('-inf')
                
#                 # Greedy MMR selection
#                 selected = []
#                 for _ in range(k):
#                     if torch.all(torch.isinf(scores)):
#                         break
                    
#                     # MMR: balance relevance and diversity
#                     diversity_penalty = exposure / (exposure.max() + 1e-8)
#                     mmr_scores = scores - self.lambda_div * diversity_penalty
                    
#                     best_idx = mmr_scores.argmax().item()
#                     if torch.isinf(scores[best_idx]):
#                         break
                    
#                     selected.append(best_idx)
#                     exposure[best_idx] += 1
#                     scores[best_idx] = float('-inf')
                
#                 recs_by_u[u] = selected
            
#             # Update progress bar with statistics
#             pbar.set_postfix({
#                 'nodes': f'{chunk_end}/{len(test_nodes)}',
#                 'max_exposure': f'{exposure.max():.0f}'
#             })
        
#         pbar.close()
#         print(f"✓ MMR Re-ranking complete. Generated {len(recs_by_u)} recommendation lists")
        
#         return recs_by_u
    
#     def _rerank_exposure(self, h, train_adj, test_nodes, k, n):
#         """
#         Exposure-based re-ranking with hard caps.
        
#         Ensures no node appears in recommendations more than exposure_cap times.
#         """
#         device = h.device
        
#         if self.exposure_cap is None:
#             # Default: average recommendations per node + lambda_div buffer
#             avg_recs = (len(test_nodes) * k) / n
#             self.exposure_cap = int(avg_recs * (1 + self.lambda_div))
        
#         exposure_count = defaultdict(int)
#         recs_by_u = {}
        
#         # Process in chunks
#         chunk_size = 1024
#         num_chunks = (len(test_nodes) + chunk_size - 1) // chunk_size
        
#         print(f"\n🔄 Exposure Re-ranking: Processing {len(test_nodes)} nodes in {num_chunks} chunks...")
#         print(f"   Exposure cap: {self.exposure_cap} recommendations per node")
        
#         # Create progress bar for chunks
#         chunk_iterator = range(0, len(test_nodes), chunk_size)
#         pbar = tqdm(chunk_iterator, desc="Exposure Re-ranking", unit="chunk",
#                    total=num_chunks)
        
#         for chunk_start in pbar:
#             chunk_end = min(chunk_start + chunk_size, len(test_nodes))
#             chunk_test_nodes = test_nodes[chunk_start:chunk_end]
            
#             test_nodes_tensor = torch.tensor(chunk_test_nodes, dtype=torch.long, device=device)
#             h_test_chunk = h[test_nodes_tensor]
#             chunk_scores = torch.sigmoid(h_test_chunk @ h.t()).cpu()
#             del h_test_chunk
            
#             for i, u in enumerate(chunk_test_nodes):
#                 scores = chunk_scores[i].clone()
                
#                 # Mask invalid candidates
#                 scores[u] = float('-inf')
#                 for v in train_adj[u]:
#                     if v < n:
#                         scores[v] = float('-inf')
                
#                 # Select top-k respecting exposure caps
#                 selected = []
#                 topk_scores, topk_indices = torch.topk(scores, min(k * 3, n))
                
#                 for idx in topk_indices:
#                     v = idx.item()
#                     if torch.isinf(scores[v]):
#                         break
#                     if exposure_count[v] < self.exposure_cap:
#                         selected.append(v)
#                         exposure_count[v] += 1
#                         if len(selected) >= k:
#                             break
                
#                 recs_by_u[u] = selected
            
#             # Update progress bar with statistics
#             max_exposure = max(exposure_count.values()) if exposure_count else 0
#             pbar.set_postfix({
#                 'nodes': f'{chunk_end}/{len(test_nodes)}',
#                 'max_exposure': f'{max_exposure}/{self.exposure_cap}'
#             })
        
#         pbar.close()
#         print(f"✓ Exposure Re-ranking complete. Generated {len(recs_by_u)} recommendation lists")
        
#         return recs_by_u


from typing import Dict, List, Optional
import torch
import sys
import random
from collections import defaultdict
from tqdm import tqdm
import os

# Path Fix: Navigate to SNAP folder to find model.py
current_dir = os.path.dirname(os.path.abspath(__file__))
snap_path = os.path.abspath(os.path.join(current_dir, '../../SNAP'))
if snap_path not in sys.path:
    sys.path.append(snap_path)


class DiversityReRanker:
    """Re-rank recommendations for fairness while maintaining precision."""
    
    def __init__(self, method='mmr', lambda_div=0.3, exposure_cap=None, 
                 exposure_scale='linear', penalty_mode='global', shuffle_users=False):
        """
        Args:
            method: 'mmr' or 'exposure'
            lambda_div: Diversity strength (0 = no diversity, 1 = maximum diversity)
            exposure_cap: Maximum recommendations per node (for exposure method)
            exposure_scale: How to scale exposure for penalty calculation
                          - 'linear': penalty = exposure (default)
                          - 'log': penalty = log(1 + exposure) (smoother, less aggressive)
                          - 'sqrt': penalty = sqrt(exposure) (moderate smoothing)
            penalty_mode: MMR diversity penalty type
                         - 'global': penalize all exposure linearly (default)
                         - 'overexposed': only penalize nodes above average exposure
            shuffle_users: Whether to shuffle test nodes to remove early-user advantage
        """
        self.method = method
        self.lambda_div = lambda_div
        self.exposure_cap = exposure_cap
        self.exposure_scale = exposure_scale if method == 'mmr' else 'linear'
        self.penalty_mode = penalty_mode if method == 'mmr' else 'global'
        self.shuffle_users = shuffle_users
    
    def rerank_topk(self, h, train_edges, test_nodes, k, degrees, score_fn=None):
        """
        Re-rank recommendations for fairness.

        Args:
            h: Node embeddings [N, D]
            train_edges: Training edges to exclude
            test_nodes: Nodes to generate recommendations for
            k: Number of recommendations per node
            degrees: Node degrees (currently unused, kept for API consistency)
            score_fn: Optional scoring function (e.g., from make_predictor_score_fn()).
                      If None, uses dot product.

        Returns:
            recs_by_u: dict mapping node_id -> list of top-k recommended nodes
        """
        import numpy as np
        import scipy.sparse as sp
        device = h.device
        n = h.size(0)

        # Build sparse CSR mask for fast vectorized masking
        edges_cpu = train_edges.cpu().numpy()
        rows = np.concatenate([edges_cpu[0], edges_cpu[1]])
        cols = np.concatenate([edges_cpu[1], edges_cpu[0]])
        valid = (rows < n) & (cols < n)
        rows, cols = rows[valid], cols[valid]
        self._train_mask_csr = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(n, n)
        )

        if self.method == 'mmr':
            return self._rerank_mmr(h, None, test_nodes, k, score_fn=score_fn)
        elif self.method == 'exposure':
            return self._rerank_exposure(h, None, test_nodes, k, n, score_fn=score_fn)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _build_adjacency(self, train_edges, n):
        """Build adjacency list from training edges (vectorized)."""
        from core.data_loader import build_train_adjacency
        return build_train_adjacency(train_edges, n)
    
    def _rerank_mmr(self, h, train_adj, test_nodes, k, score_fn=None):
        """
        Maximal Marginal Relevance re-ranking.
        
        Balances relevance (model score) with diversity (exposure balance).
        """
        device = h.device
        n = h.size(0)
        exposure = torch.zeros(n, device='cpu')
        recs_by_u = {}
        
        # Shuffle test nodes if enabled (deterministic with set_seed)
        test_nodes_list = list(test_nodes)
        if self.shuffle_users:
            random.shuffle(test_nodes_list)
            print(f"🔀 Shuffled test nodes to remove early-user advantage")
        
        # Process in chunks to avoid OOM
        chunk_size = 1024
        num_chunks = (len(test_nodes_list) + chunk_size - 1) // chunk_size
        
        scoring_method = "trained predictor" if score_fn is not None else "dot product"
        print(f"\n📄 MMR Re-ranking: Processing {len(test_nodes_list)} nodes in {num_chunks} chunks...")
        print(f"   Exposure scaling: {self.exposure_scale}")
        print(f"   Penalty mode: {self.penalty_mode}")
        print(f"   Scoring: {scoring_method}")
        
        # Create progress bar for chunks
        chunk_iterator = range(0, len(test_nodes_list), chunk_size)
        pbar = tqdm(chunk_iterator, desc="MMR Re-ranking", unit="chunk", 
                   total=num_chunks)
        
        import numpy as np
        for chunk_start in pbar:
            chunk_end = min(chunk_start + chunk_size, len(test_nodes_list))
            chunk_test_nodes = test_nodes_list[chunk_start:chunk_end]

            # Compute scores for this chunk
            test_nodes_tensor = torch.tensor(chunk_test_nodes, dtype=torch.long, device=device)
            h_test_chunk = h[test_nodes_tensor]
            if score_fn is not None:
                chunk_scores = score_fn(h_test_chunk, h).cpu()
            else:
                chunk_scores = torch.sigmoid(h_test_chunk @ h.t()).cpu()
            del h_test_chunk  # Free GPU memory

            # Vectorized masking for entire chunk
            chunk_arr = np.array(chunk_test_nodes)
            chunk_scores[range(len(chunk_test_nodes)), chunk_arr] = float('-inf')
            mask_block = self._train_mask_csr[chunk_arr]
            mask_rows, mask_cols = mask_block.nonzero()
            chunk_scores[mask_rows, mask_cols] = float('-inf')

            # Process each user in the chunk
            for i, u in enumerate(chunk_test_nodes):
                scores = chunk_scores[i].clone()
                
                # Greedy MMR selection
                selected = []
                for _ in range(k):
                    if torch.all(torch.isinf(scores)):
                        break
                    
                    # Apply exposure scaling
                    if self.exposure_scale == 'log':
                        # Log scaling: penalty grows logarithmically
                        # More gentle - allows popular items to still be recommended
                        if exposure.max() > 0:
                            diversity_penalty = torch.log1p(exposure) / (torch.log1p(exposure.max()) + 1e-8)
                        else:
                            diversity_penalty = exposure
                    elif self.exposure_scale == 'sqrt':
                        # Square root scaling: moderate penalty growth
                        if exposure.max() > 0:
                            diversity_penalty = torch.sqrt(exposure) / (torch.sqrt(exposure.max()) + 1e-8)
                        else:
                            diversity_penalty = exposure
                    else:  # 'linear'
                        # Linear scaling: penalty proportional to exposure
                        if exposure.max() > 0:
                            diversity_penalty = exposure / (exposure.max() + 1e-8)
                        else:
                            diversity_penalty = exposure
                    
                    # Apply penalty mode
                    if self.penalty_mode == 'overexposed':
                        # Only penalize nodes above average exposure
                        avg_exp = exposure.mean()
                        over = (exposure - avg_exp).clamp_min(0)
                        if over.max() > 0:
                            diversity_penalty = over / (over.max() + 1e-8)
                        else:
                            diversity_penalty = over
                    # else: penalty_mode == 'global' (use diversity_penalty as-is)
                    
                    # MMR: balance relevance and diversity
                    mmr_scores = scores - self.lambda_div * diversity_penalty
                    
                    best_idx = mmr_scores.argmax().item()
                    if torch.isinf(scores[best_idx]):
                        break
                    
                    selected.append(best_idx)
                    exposure[best_idx] += 1
                    scores[best_idx] = float('-inf')
                
                recs_by_u[u] = selected
            
            # Update progress bar with statistics
            pbar.set_postfix({
                'nodes': f'{chunk_end}/{len(test_nodes_list)}',
                'max_exposure': f'{exposure.max():.0f}',
                'avg_exposure': f'{exposure.mean():.1f}'
            })
        
        pbar.close()
        print(f"✅ MMR Re-ranking complete. Generated {len(recs_by_u)} recommendation lists")
        print(f"   Final exposure stats: max={exposure.max():.0f}, "
              f"mean={exposure.mean():.1f}, std={exposure.std():.1f}")
        
        return recs_by_u
    
    def _rerank_exposure(self, h, train_adj, test_nodes, k, n, score_fn=None):
        """
        Exposure-based re-ranking with hard caps.
        
        Ensures no node appears in recommendations more than exposure_cap times.
        """
        device = h.device
        
        if self.exposure_cap is None:
            # Default: average recommendations per node + lambda_div buffer
            avg_recs = (len(test_nodes) * k) / n
            self.exposure_cap = int(avg_recs * (1 + self.lambda_div))
        
        exposure_count = defaultdict(int)
        recs_by_u = {}
        
        # Shuffle test nodes if enabled
        test_nodes_list = list(test_nodes)
        if self.shuffle_users:
            random.shuffle(test_nodes_list)
            print(f"🔀 Shuffled test nodes to remove early-user advantage")
        
        # Process in chunks
        chunk_size = 1024
        num_chunks = (len(test_nodes_list) + chunk_size - 1) // chunk_size
        
        scoring_method = "trained predictor" if score_fn is not None else "dot product"
        print(f"\n📄 Exposure Re-ranking: Processing {len(test_nodes_list)} nodes in {num_chunks} chunks...")
        print(f"   Exposure cap: {self.exposure_cap} recommendations per node")
        print(f"   Scoring: {scoring_method}")
        
        # Create progress bar for chunks
        chunk_iterator = range(0, len(test_nodes_list), chunk_size)
        pbar = tqdm(chunk_iterator, desc="Exposure Re-ranking", unit="chunk",
                   total=num_chunks)
        
        import numpy as np
        for chunk_start in pbar:
            chunk_end = min(chunk_start + chunk_size, len(test_nodes_list))
            chunk_test_nodes = test_nodes_list[chunk_start:chunk_end]

            test_nodes_tensor = torch.tensor(chunk_test_nodes, dtype=torch.long, device=device)
            h_test_chunk = h[test_nodes_tensor]
            if score_fn is not None:
                chunk_scores = score_fn(h_test_chunk, h).cpu()
            else:
                chunk_scores = torch.sigmoid(h_test_chunk @ h.t()).cpu()
            del h_test_chunk

            # Vectorized masking for entire chunk
            chunk_arr = np.array(chunk_test_nodes)
            chunk_scores[range(len(chunk_test_nodes)), chunk_arr] = float('-inf')
            mask_block = self._train_mask_csr[chunk_arr]
            mask_rows, mask_cols = mask_block.nonzero()
            chunk_scores[mask_rows, mask_cols] = float('-inf')

            for i, u in enumerate(chunk_test_nodes):
                scores = chunk_scores[i].clone()

                # Select top-k respecting exposure caps
                selected = []
                topk_scores, topk_indices = torch.topk(scores, min(k * 3, n))
                
                for idx in topk_indices:
                    v = idx.item()
                    if torch.isinf(scores[v]):
                        break
                    if exposure_count[v] < self.exposure_cap:
                        selected.append(v)
                        exposure_count[v] += 1
                        if len(selected) >= k:
                            break
                
                recs_by_u[u] = selected
            
            # Update progress bar with statistics
            max_exposure = max(exposure_count.values()) if exposure_count else 0
            pbar.set_postfix({
                'nodes': f'{chunk_end}/{len(test_nodes_list)}',
                'max_exposure': f'{max_exposure}/{self.exposure_cap}'
            })
        
        pbar.close()
        print(f"✅ Exposure Re-ranking complete. Generated {len(recs_by_u)} recommendation lists")
        
        return recs_by_u


class ProbabilisticMMRReRanker:
    """
    Probabilistic variant of MMR that samples from top candidates.
    Reduces determinism and can improve diversity.
    """
    
    def __init__(self, lambda_div=0.3, temperature=0.1, top_k_sample=5):
        """
        Args:
            lambda_div: Diversity penalty strength
            temperature: Sampling temperature (lower = more focused on top)
            top_k_sample: Sample from top K candidates at each step
        """
        self.lambda_div = lambda_div
        self.temperature = temperature
        self.top_k_sample = top_k_sample
    
    def rerank_topk(self, h, train_edges, test_nodes, k, degrees, score_fn=None):
        device = h.device
        n = h.size(0)
        train_adj = self._build_adjacency(train_edges, n)
        exposure = torch.zeros(n, device='cpu')
        recs_by_u = {}
        
        from tqdm import tqdm
        chunk_size = 1024
        scoring_method = "trained predictor" if score_fn is not None else "dot product"
        num_chunks = (len(test_nodes) + chunk_size - 1) // chunk_size

        for chunk_start in tqdm(range(0, len(test_nodes), chunk_size), total=num_chunks,
                                desc=f"Prob MMR ({scoring_method})", unit="chunk", file=sys.stdout):
            chunk_end = min(chunk_start + chunk_size, len(test_nodes))
            chunk_test_nodes = test_nodes[chunk_start:chunk_end]
            
            test_nodes_tensor = torch.tensor(chunk_test_nodes, dtype=torch.long, device=device)
            h_test_chunk = h[test_nodes_tensor]
            if score_fn is not None:
                chunk_scores = score_fn(h_test_chunk, h).cpu()
            else:
                chunk_scores = torch.sigmoid(h_test_chunk @ h.t()).cpu()
            del h_test_chunk
            
            for i, u in enumerate(chunk_test_nodes):
                scores = chunk_scores[i].clone()
                scores[u] = float('-inf')
                for v in train_adj[u]:
                    if v < n:
                        scores[v] = float('-inf')
                
                selected = []
                for _ in range(k):
                    if torch.all(torch.isinf(scores)):
                        break
                    
                    # Compute MMR scores
                    diversity_penalty = exposure / (exposure.max() + 1e-8)
                    mmr_scores = scores - self.lambda_div * diversity_penalty
                    
                    # Get top-K candidates
                    valid_scores = mmr_scores[torch.isfinite(mmr_scores)]
                    if len(valid_scores) == 0:
                        break
                    
                    top_k = min(self.top_k_sample, len(valid_scores))
                    topk_values, topk_indices = torch.topk(mmr_scores, top_k)
                    
                    # Sample probabilistically
                    probs = torch.softmax(topk_values / self.temperature, dim=0)
                    sampled_idx = torch.multinomial(probs, 1).item()
                    best_idx = topk_indices[sampled_idx].item()
                    
                    selected.append(best_idx)
                    exposure[best_idx] += 1
                    scores[best_idx] = float('-inf')
                
                recs_by_u[u] = selected
        
        return recs_by_u
    
    def _build_adjacency(self, train_edges, n):
        train_adj = [set() for _ in range(n)]
        for u, v in train_edges.cpu().tolist():
            if u < n and v < n:
                train_adj[u].add(v)
                train_adj[v].add(u)
        return train_adj


class EnsembleMMRReRanker:
    """
    Ensemble variant that runs MMR multiple times with different random seeds
    and aggregates results.
    """
    
    def __init__(self, lambda_div=0.3, temperature=0.1, n_runs=3, 
                 top_k_sample=5, base_seed=42):
        """
        Args:
            lambda_div: Diversity penalty strength
            temperature: Sampling temperature
            n_runs: Number of ensemble runs
            top_k_sample: Sample from top K at each step
            base_seed: Base random seed
        """
        self.lambda_div = lambda_div
        self.temperature = temperature
        self.n_runs = n_runs
        self.top_k_sample = top_k_sample
        self.base_seed = base_seed
    
    def rerank_topk(self, h, train_edges, test_nodes, k, degrees, score_fn=None):
        """Run multiple MMR passes and aggregate."""
        print(f"Running Ensemble MMR with {self.n_runs} runs...")
        
        all_recs = []
        for run_idx in range(self.n_runs):
            # Set seed for reproducibility
            seed = self.base_seed + run_idx
            random.seed(seed)
            torch.manual_seed(seed)
            
            reranker = ProbabilisticMMRReRanker(
                lambda_div=self.lambda_div,
                temperature=self.temperature,
                top_k_sample=self.top_k_sample
            )
            
            recs = reranker.rerank_topk(h, train_edges, test_nodes, k, degrees, score_fn=score_fn)
            all_recs.append(recs)
            print(f"  Run {run_idx + 1}/{self.n_runs} complete")
        
        # Aggregate by voting
        aggregated_recs = self._aggregate_recommendations(all_recs, test_nodes, k)
        return aggregated_recs
    
    def _aggregate_recommendations(self, all_recs, test_nodes, k):
        """Aggregate recommendations by vote counting."""
        aggregated = {}
        
        for u in test_nodes:
            # Count votes for each candidate
            vote_counts = defaultdict(int)
            for recs in all_recs:
                if u in recs:
                    for v in recs[u]:
                        vote_counts[v] += 1
            
            # Sort by votes (break ties by first appearance)
            candidates = sorted(vote_counts.items(), 
                              key=lambda x: (-x[1], x[0]))
            aggregated[u] = [v for v, _ in candidates[:k]]
        
        return aggregated