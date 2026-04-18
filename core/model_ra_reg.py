
import torch
import torch.nn.functional as F
from backbone.model import BaseModel
from torch.utils.data import DataLoader
from backbone.utils import get_pos_neg_edges

class RARegModel(BaseModel):
    def __init__(self, *args, ra_matrix=None, alpha_reg=0.1, **kwargs):
        """
        ra_matrix: Scipy sparse matrix [N, N] containing Normalized RA scores.
        alpha_reg: Weight for the regularization term.
        """
        super().__init__(*args, **kwargs)
        self.ra_matrix = ra_matrix
        self.alpha_reg = alpha_reg
        
    def train(self, data, split_edge, batch_size, neg_sampler_name, num_neg):
        self.encoder.train()
        self.predictor.train()

        if 'edge' in split_edge['train']:
            pos_train_edge = split_edge['train']['edge'].to(self.device)
        elif 'source_node' in split_edge['train']:
             source = split_edge['train']['source_node']
             target = split_edge['train']['target_node']
             pos_train_edge = torch.stack([source, target]).t().to(self.device)
        else:
             raise ValueError("Could not find training edges")

        if 'weight' in split_edge['train']:
            edge_weight_margin = split_edge['train']['weight'].to(self.device)
        else:
            edge_weight_margin = None

        total_loss = total_examples = 0

        _, neg_train_edge = get_pos_neg_edges('train', split_edge, edge_index=data.edge_index, num_nodes=data.num_nodes, neg_sampler_name=neg_sampler_name, num_neg=num_neg)
        
        if neg_train_edge is not None:
            neg_train_edge = neg_train_edge.to(self.device)
            
        for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
            self.optimizer.zero_grad()

            input_feat = self.create_input_feat(data)
            h = self.encoder(input_feat, data.adj_t, data.edge_weight)
            
            # Get Batch Edges
            pos_edge = pos_train_edge[perm].t() # [2, B]
            
            if neg_train_edge is not None:
                neg_edge = torch.reshape(neg_train_edge[perm], (-1, 2)).t() # [2, B*num_neg]
            else: # Local sampling fallback (if utilized)
                pass

            # Predictions
            pos_out = self.predictor(h[pos_edge[0]], h[pos_edge[1]])
            neg_out = self.predictor(h[neg_edge[0]], h[neg_edge[1]])

            weight_margin = edge_weight_margin[perm] if edge_weight_margin is not None else None

            # Standard Loss
            loss = self.calculate_loss(pos_out, neg_out, num_neg, margin=weight_margin)
            
            # === RA REGULARIZATION ===
            if self.alpha_reg > 0 and self.ra_matrix is not None:
                # 1. Get RA scores for Positive Edges
                # Need CPU indices for scipy lookup
                pos_u = pos_edge[0].cpu().numpy().copy()
                pos_v = pos_edge[1].cpu().numpy().copy()
                
                # Lookup RA
                # ra_matrix is [N, N]
                # indexing: ra_matrix[u, v]
                ra_pos = torch.from_numpy(self.ra_matrix[pos_u, pos_v].A1).float().to(self.device)
                
                # 2. Get RA scores for Negative Edges
                neg_u = neg_edge[0].cpu().numpy().copy()
                neg_v = neg_edge[1].cpu().numpy().copy()
                ra_neg = torch.from_numpy(self.ra_matrix[neg_u, neg_v].A1).float().to(self.device)
                
                # 3. Calculate MSE Loss
                # pos_out and neg_out are logits (unbounded)? Or sigmoid?
                # HLGNN predictor usually returns raw scores (dot product).
                # RA scores are also raw (but normalized).
                # We align them via MSE.
                
                reg_loss_pos = F.mse_loss(pos_out.squeeze(), ra_pos)
                reg_loss_neg = F.mse_loss(neg_out.squeeze(), ra_neg)
                
                reg_loss = (reg_loss_pos + reg_loss_neg) / 2.0
                
                loss = loss + self.alpha_reg * reg_loss

            loss.backward()

            if self.clip_norm >= 0:
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip_norm)
                torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), self.clip_norm)

            self.optimizer.step()

            num_examples = pos_out.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples

        return total_loss / total_examples
