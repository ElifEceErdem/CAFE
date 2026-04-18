import torch


class SNAPEvaluator:
    """
    Evaluator for SNAP datasets.
    Implements Hits@K, MRR, Precision@K, and Fairness metrics.
    """
    
    def __init__(self):
        self.K = 100  # Default K for Hits@K
    
    def eval(self, input_dict):
        """
        Evaluate predictions
        
        Args:
            input_dict: Dictionary containing:
                - y_pred_pos: Predictions for positive edges
                - y_pred_neg: Predictions for negative edges
        
        Returns:
            Dictionary with evaluation results
        """
        y_pred_pos = input_dict['y_pred_pos']
        y_pred_neg = input_dict['y_pred_neg']
        
        # Ensure predictions are on CPU and numpy
        if torch.is_tensor(y_pred_pos):
            y_pred_pos = y_pred_pos.cpu().numpy()
        if torch.is_tensor(y_pred_neg):
            y_pred_neg = y_pred_neg.cpu().numpy()
        
        # Reshape if needed
        if y_pred_neg.ndim == 1:
            # Assume equal number of negatives per positive
            num_pos = len(y_pred_pos)
            num_neg_per_pos = len(y_pred_neg) // num_pos
            y_pred_neg = y_pred_neg.reshape(num_pos, num_neg_per_pos)
        
        return self._eval_hits(y_pred_pos, y_pred_neg)
    
    def _eval_hits(self, y_pred_pos, y_pred_neg):
        """
        Calculate Hits@K metric
        
        Args:
            y_pred_pos: Shape (num_pos,)
            y_pred_neg: Shape (num_pos, num_neg)
        
        Returns:
            Dictionary with hits@K score
        """
        results = {}
        
        # For each positive edge, count how many negatives have lower scores
        y_pred_pos = y_pred_pos.reshape(-1, 1)
        
        # Optimistic ranking: if pos score equals neg score, rank pos higher
        rankings = (y_pred_neg >= y_pred_pos).sum(axis=1) + 1
        
        # Calculate Hits@K
        hits_at_k = (rankings <= self.K).mean()
        
        results[f'hits@{self.K}'] = hits_at_k
        
        return results
    
    def eval_mrr(self, input_dict):
        """
        Calculate Mean Reciprocal Rank (MRR)
        
        Args:
            input_dict: Dictionary containing:
                - y_pred_pos: Predictions for positive edges
                - y_pred_neg: Predictions for negative edges
        
        Returns:
            Dictionary with MRR results
        """
        y_pred_pos = input_dict['y_pred_pos']
        y_pred_neg = input_dict['y_pred_neg']
        
        # Ensure predictions are on CPU
        if torch.is_tensor(y_pred_pos):
            y_pred_pos = y_pred_pos.cpu()
        if torch.is_tensor(y_pred_neg):
            y_pred_neg = y_pred_neg.cpu()
        
        # Ensure correct shape
        y_pred_pos = y_pred_pos.reshape(-1, 1)
        if y_pred_neg.ndim == 1:
            num_pos = y_pred_pos.size(0)
            num_neg_per_pos = y_pred_neg.size(0) // num_pos
            y_pred_neg = y_pred_neg.reshape(num_pos, num_neg_per_pos)
        
        # Calculate rankings
        rankings = (y_pred_neg >= y_pred_pos).sum(dim=1) + 1
        
        # Calculate MRR
        mrr_list = 1.0 / rankings.float()
        
        return {'mrr_list': mrr_list}
    
    def eval_precision_at_k(self, input_dict):
        """
        Calculate Precision@K using top-k recommendations per node.
        
        Args:
            input_dict: Dictionary containing:
                - h: Node embeddings (tensor of shape [num_nodes, hidden_dim])
                - train_edge_index: Training edges (tensor of shape [2, num_train_edges])
                - test_edge_set: Set of test edges as tuples (u, v)
                - k: Top-k for recommendations (default: 20)
                - chunk_size: Chunk size for processing (default: 1024)
        
        Returns:
            Dictionary with precision@k score
        """
        h = input_dict['h']
        train_edge_index = input_dict['train_edge_index']
        test_edge_set = input_dict['test_edge_set']
        k = input_dict.get('k', 20)
        chunk_size = input_dict.get('chunk_size', 1024)
        
        # Generate top-k recommendations
        recs_by_u = self.recommend_topk_per_node_with_chunks(
            h, train_edge_index, k, chunk_size
        )
        
        # Calculate precision@k
        precision = self.precision_at_k_micro(recs_by_u, test_edge_set, k)
        
        return {f'precision@{k}': precision}
    
    def eval_fairness(self, input_dict):
        """
        Calculate fairness metric (1 - Gini coefficient) for recommendation network.
        
        Args:
            input_dict: Dictionary containing:
                - h: Node embeddings (tensor of shape [num_nodes, hidden_dim])
                - train_edge_index: Training edges (tensor of shape [2, num_train_edges])
                - num_nodes: Total number of nodes
                - k: Top-k for recommendations (default: 20)
                - chunk_size: Chunk size for processing (default: 1024)
        
        Returns:
            Dictionary with fairness score
        """
        h = input_dict['h']
        train_edge_index = input_dict['train_edge_index']
        num_nodes = input_dict['num_nodes']
        k = input_dict.get('k', 20)
        chunk_size = input_dict.get('chunk_size', 1024)
        
        # Generate top-k recommendations
        recs_by_u = self.recommend_topk_per_node_with_chunks(
            h, train_edge_index, k, chunk_size
        )
        
        # Calculate fairness
        fairness = self.calculate_recommendation_network_gini(recs_by_u, num_nodes)
        
        return {'fairness': fairness}
    
    # --- Helper functions for P@k and Fairness ---
    
    @staticmethod
    def gini_coefficient_1d(x: torch.Tensor) -> float:
        """Calculate Gini coefficient for a 1D tensor."""
        if x.numel() == 0:
            return 0.0
        x = x.float().clamp_min(0)
        s = x.sum()
        if s.item() == 0:
            return 0.0
        x_sorted, _ = torch.sort(x.view(-1))
        n = x_sorted.numel()
        idx = torch.arange(1, n + 1, dtype=x_sorted.dtype, device=x_sorted.device)
        gini = 1.0 + (1.0 / n) - 2.0 * (torch.sum((n + 1 - idx) * x_sorted) / (n * s))
        return float(gini)
    
    @staticmethod
    def build_undirected_adj_list(edge_index: torch.Tensor, num_nodes: int):
        """Build adjacency list from edge index."""
        src = edge_index[0].detach().cpu().tolist()
        dst = edge_index[1].detach().cpu().tolist()
        neigh = [set() for _ in range(num_nodes)]
        for u, v in zip(src, dst):
            if u == v:
                continue
            neigh[u].add(v)
            neigh[v].add(u)
        return neigh
    
    @staticmethod
    def recommend_topk_per_node_with_chunks(z: torch.Tensor, train_edge_index: torch.Tensor, 
                                           k: int, chunk_size: int = 1024):
        """Generate top-k recommendations for each node, excluding training edges."""
        device = z.device
        n = z.size(0)
        neighbors = SNAPEvaluator.build_undirected_adj_list(train_edge_index, n)
        recs_by_u = {}
        
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            z_chunk = z[start:end, :]
            scores = (z_chunk @ z.t()).sigmoid().detach().cpu()
            
            for i, u in enumerate(range(start, end)):
                # Exclude self-loops
                scores[i, u] = float("-inf")
                # Exclude existing neighbors (training edges)
                for v in neighbors[u]:
                    if v < scores.shape[1]:
                        scores[i, v] = float("-inf")
                
                topk = min(k, scores.size(1))
                if topk <= 0:
                    recs_by_u[u] = []
                    continue
                
                vals, idx = torch.topk(scores[i], topk)
                valid = vals.isfinite()
                recs_by_u[u] = idx[valid].tolist()[:k]
        
        return recs_by_u
    
    @staticmethod
    def precision_at_k_micro(recs_by_u, pos_edges_set, k: int) -> float:
        """Calculate micro-averaged Precision@k."""
        total_rec, total_hit = 0, 0
        for u, vs in recs_by_u.items():
            if not vs:
                continue
            take = min(k, len(vs))
            total_rec += take
            for v in vs[:take]:
                if (u, v) in pos_edges_set:
                    total_hit += 1
        return 0.0 if total_rec == 0 else total_hit / total_rec
    
    @staticmethod
    def calculate_recommendation_network_gini(recs_by_u, num_nodes: int) -> float:
        """
        Calculate 1 - Gini coefficient for the degree distribution of the recommendation network
        (treating recommendations as edges).
        
        Returns:
            fairness_score: 1 - Gini (higher = more fair, ranges from 0 to 1)
        """
        # Count degree for each node in the recommendation network
        degree = torch.zeros(num_nodes, dtype=torch.int)
        
        for u, vs in recs_by_u.items():
            # Out-degree: user u recommends to |vs| nodes
            degree[u] += len(vs)
            # In-degree: each recommended node v receives a recommendation
            for v in vs:
                if v < num_nodes:
                    degree[v] += 1
        
        gini = SNAPEvaluator.gini_coefficient_1d(degree)
        fairness_score = 1.0 - gini
        
        return fairness_score