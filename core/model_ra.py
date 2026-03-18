
import torch
from backbone.model import BaseModel
from core.ra_sampler import ra_guided_neg_sample
from torch.utils.data import DataLoader

class RAModel(BaseModel):
    def __init__(self, *args, ra_candidates=None, easy_ratio=0.5, **kwargs):
        """
        ra_candidates: Tensor [N, 2] of pre-computed hard negative candidates (RA > 0, no edge)
        easy_ratio: Ratio of easy (random) negatives to use
        """
        super().__init__(*args, **kwargs)
        self.ra_candidates = ra_candidates
        self.easy_ratio = easy_ratio
        
    def train(self, data, split_edge, batch_size, neg_sampler_name, num_neg):
        if neg_sampler_name == 'ra_guided':
            self.encoder.train()
            self.predictor.train()

            # 1. Get Positive Edges
            if 'edge' in split_edge['train']:
                pos_train_edge = split_edge['train']['edge']
            elif 'source_node' in split_edge['train']:
                 source = split_edge['train']['source_node']
                 target = split_edge['train']['target_node']
                 pos_train_edge = torch.stack([source, target]).t()
            else:
                 raise ValueError("Could not find training edges")
                 
            pos_train_edge = pos_train_edge.to(self.device)
            
            # 2. Sample Negatives (Mixed)
            neg_train_edge = ra_guided_neg_sample(
                pos_train_edge,
                data.edge_index,
                self.num_nodes,
                num_neg,
                self.ra_candidates,
                easy_ratio=self.easy_ratio
            )
            # neg_train_edge is [E * num_neg, 2]
            # We need to reshape to [E, num_neg, 2] to allow [perm] indexing 
            # only if we want to keep it aligned with pos edges (conceptually not needed for global, but needed for code compat).
            
            # However, since we shuffled in ra_sampler, alignment to specific 'u' is lost.
            # That's fine for "Global" RA sampling.
            # Just ensure we reshape so that 'perm' works.
            
            if neg_train_edge.size(0) != pos_train_edge.size(0) * num_neg:
                 # Should not happen if sampler is correct
                 pass
                 
            neg_train_edge = neg_train_edge.view(pos_train_edge.size(0), num_neg, 2)
            neg_train_edge = neg_train_edge.to(self.device)

            if 'weight' in split_edge['train']:
                edge_weight_margin = split_edge['train']['weight'].to(self.device)
            else:
                edge_weight_margin = None

            total_loss = total_examples = 0

            for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
                self.optimizer.zero_grad(set_to_none=True)

                input_feat = self.create_input_feat(data)
                h = self.encoder(input_feat, data.adj_t, data.edge_weight)
                
                pos_edge = pos_train_edge[perm].t()
                neg_edge = torch.reshape(neg_train_edge[perm], (-1, 2)).t()

                pos_out = self.predictor(h[pos_edge[0]], h[pos_edge[1]])
                neg_out = self.predictor(h[neg_edge[0]], h[neg_edge[1]])

                weight_margin = edge_weight_margin[perm] if edge_weight_margin is not None else None

                loss = self.calculate_loss(pos_out, neg_out, num_neg, margin=weight_margin)
                loss.backward()

                if self.clip_norm >= 0:
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip_norm)
                    torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), self.clip_norm)

                self.optimizer.step()

                num_examples = pos_out.size(0)
                total_loss += loss.item() * num_examples
                total_examples += num_examples

            return total_loss / total_examples
        else:
            return super().train(data, split_edge, batch_size, neg_sampler_name, num_neg)
