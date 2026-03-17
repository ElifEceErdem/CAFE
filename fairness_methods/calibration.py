import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_sparse import SparseTensor
import numpy as np
from metrics.coreness import compute_coreness

class CalibrationMLP(nn.Module):
    def __init__(self, input_dim=5, hidden_dims=[32, 16]):
        super(CalibrationMLP, self).__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class CalibrationReranker:
    def __init__(self, lambda_fair=0.1, lr=0.01, epochs=50, batch_size=1024, device='cuda'):
        self.lambda_fair = lambda_fair
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.model = CalibrationMLP().to(device)
        
    def _extract_features(self, scores_raw, u_core, v_core, u_deg, v_deg):
        """
        Construct feature matrix [N_samples, 5]
        """
        # Ensure all inputs are tensors on device and same shape
        # If scalar (broadcast), make sure to expand
        
        # Stack features
        # scores_raw: [B]
        # u_core: [B]
        # v_core: [B]
        # ...
        features = torch.stack([
            scores_raw,
            u_core.float(),
            v_core.float(),
            u_deg.float(),
            v_deg.float()
        ], dim=1)
        return features
        
    def train(self, h_emb, train_edges, val_edges, coreness, degrees):
        """
        Train the MLP on validation edges.
        """
        print(f"[Calibration] Training MLP for {self.epochs} epochs (lambda={self.lambda_fair})...")
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Prepare Validation Data
        # We need Pos and Neg samples from validation set
        pos_edge_index = val_edges['edge'].t() # [2, E]
        neg_edge_index = val_edges['edge_neg'].t() # [2, E]
        
        num_pos = pos_edge_index.size(1)
        
        # Precompute features that are static
        # Coreness and Degrees are full-node vectors
        
        self.model.train()
        
        for epoch in range(self.epochs):
            total_loss = 0
            
            # Shuffle
            perm = torch.randperm(num_pos)
            
            for i in range(0, num_pos, self.batch_size):
                batch_idx = perm[i:i+self.batch_size]
                
                # Positive samples
                u_pos = pos_edge_index[0, batch_idx].to(self.device)
                v_pos = pos_edge_index[1, batch_idx].to(self.device)
                
                # Negative samples (matched)
                u_neg = neg_edge_index[0, batch_idx].to(self.device)
                v_neg = neg_edge_index[1, batch_idx].to(self.device)
                
                # Get raw GNN scores (fixed)
                with torch.no_grad():
                    pos_scores_raw = torch.sigmoid((h_emb[u_pos] * h_emb[v_pos]).sum(dim=1))
                    neg_scores_raw = torch.sigmoid((h_emb[u_neg] * h_emb[v_neg]).sum(dim=1))
                
                # Node Features
                pos_core_u, pos_core_v = coreness[u_pos], coreness[v_pos]
                pos_deg_u, pos_deg_v = degrees[u_pos], degrees[v_pos]
                
                neg_core_u, neg_core_v = coreness[u_neg], coreness[v_neg]
                neg_deg_u, neg_deg_v = degrees[u_neg], degrees[v_neg]
                
                # MLP Input Features
                x_pos = self._extract_features(pos_scores_raw, pos_core_u, pos_core_v, pos_deg_u, pos_deg_v)
                x_neg = self._extract_features(neg_scores_raw, neg_core_u, neg_core_v, neg_deg_u, neg_deg_v)
                
                # MLP Prediction
                pos_out = self.model(x_pos).squeeze()
                neg_out = self.model(x_neg).squeeze()
                
                # 1. Ranking Loss (BPR)
                # Maximize pos > neg
                loss_rank = -torch.log(torch.sigmoid(pos_out - neg_out) + 1e-8).mean()
                
                # 2. Fairness Regularization (Coreness Correlation)
                # We want prediction NOT to correlate with v_core
                # covariance(out, v_core) should be 0?
                # Or simply: Penalize if high coreness gets higher score systematically?
                
                # Let's use Pearson Correlation penalty on POSITIVE samples only
                # (We want relevant items to have scores independent of coreness)
                # Or on ALL samples?
                
                # Implementation of differentiable correlation
                vx = pos_out - pos_out.mean()
                vy = pos_core_v.float() - pos_core_v.float().mean()
                correlation = (vx * vy).mean() / (torch.std(pos_out) * torch.std(pos_core_v.float()) + 1e-8)
                
                loss_fair = correlation ** 2 # Minimize magnitude of correlation
                
                loss = loss_rank + self.lambda_fair * loss_fair
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: Loss = {total_loss / (num_pos/self.batch_size):.4f}")
    
    def rerank_topk(self, h_emb, train_edges, test_nodes, k, coreness, degrees):
        """
        Generate calibrated recommendations.
        """
        self.model.eval()
        n = h_emb.size(0)
        
        # To avoid computing O(N^2) features, we only rerank TOP-candidate_k items
        # from the base model.
        candidate_k = min(1000, n) 
        
        recs_by_u = {}
        chunk_size = 512 # Smaller chunk due to MLP overhead
        
        print(f"[Calibration] Reranking {len(test_nodes)} users...")
        
        from core.data_loader import build_train_adjacency
        train_adj = build_train_adjacency(train_edges, n)
        
        for i in range(0, len(test_nodes), chunk_size):
            chunk = test_nodes[i:i+chunk_size]
            
            # 1. Get Base Scores (Top-Kandidates) using GNN
            with torch.no_grad():
                chunk_tensor = torch.tensor(chunk, dtype=torch.long, device=self.device)
                h_chunk = h_emb[chunk_tensor]
                
                # All scores (N) - careful with memory
                all_scores = torch.sigmoid(h_chunk @ h_emb.t()) # [chunk, N]
                
                # Mask training edges
                for idx, u in enumerate(chunk):
                    all_scores[idx, u] = -1e9
                    neighbors = train_adj[u]
                    # We need to broadcast or loop. Looping is safer for sparse check
                    # Actually for speed:
                    all_scores[idx, list(neighbors)] = -1e9
                
                # Select Top-Candidate_K
                top_vals, top_inds = torch.topk(all_scores, k=candidate_k, dim=1)
                
                # Now run MLP on these Top-Candidates
                # Flatten for batch processing
                # u_indices: [chunk * cand_k]
                u_indices = chunk_tensor.repeat_interleave(candidate_k)
                v_indices = top_inds.view(-1)
                raw_scores = top_vals.view(-1)
                
                u_core = coreness[u_indices]
                v_core = coreness[v_indices]
                u_deg = degrees[u_indices]
                v_deg = degrees[v_indices]
                
                x_feat = self._extract_features(raw_scores, u_core, v_core, u_deg, v_deg)
                
                calibrated_scores = self.model(x_feat).view(len(chunk), candidate_k)
                
                # Get final top-k
                final_topk_vals, final_topk_local_inds = torch.topk(calibrated_scores, k=k, dim=1)
                
                # Map back to global indices
                for idx, u in enumerate(chunk):
                    local_inds = final_topk_local_inds[idx]
                    global_inds = top_inds[idx, local_inds]
                    recs_by_u[u] = global_inds.tolist()
            
            if (i + chunk_size) % 5000 == 0:
                print(f"  Processed {i+chunk_size}...")
                
        return recs_by_u
