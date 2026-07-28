"""
Baseline Model Wrapper

Wraps baseline models to work with the existing evaluation framework.
Provides a unified interface compatible with BaseModel.
"""

import torch
import numpy as np
from .baseline_models import (
    HeuristicBaseline,
    RandomWalkEmbedding,
    PureGNN,
    RandomBaseline,
    TailSTEAKBaseline,
    LightGCNBaseline
)
from .fairness_aware_baseline_models import (
    FairWalkEmbedding,
    CrossWalkEmbedding,
    UGEBaseline,
    MORAL,
    FairLP_GNN,
    FairLinkModel,
    LinkBiasAmplification,
    CrossLinkBias,
    FairAdj
)
from torch.utils.data import DataLoader
from utils import get_pos_neg_edges
from loss import auc_loss
import torch.nn.functional as F
import sys
import os

class BaselineWrapper:
    """Wrapper to make baseline models compatible with evaluation framework"""
    
    def __init__(self, baseline_type, num_nodes, device='cuda:0', **kwargs):
        """
        Args:
            baseline_type: Type of baseline model
                - 'common_neighbors', 'adamic_adar', 'resource_allocation', 'preferential_attachment'
                - 'deepwalk', 'node2vec'
                - 'pure_gnn'
                - 'random'
                - 'moral'
                - 'tailsteak'
                - 'lightgcn'
                - 'ssnet'
            num_nodes: Number of nodes in graph
            device: Device to use
            **kwargs: Additional arguments for specific baselines
        """
        self.baseline_type = baseline_type
        self.num_nodes = num_nodes
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.kwargs = kwargs
        
        # Initialize baseline model
        if baseline_type in ['common_neighbors', 'adamic_adar', 'resource_allocation', 'preferential_attachment']:
            self.model = HeuristicBaseline(method=baseline_type)
            self.needs_training = False
            self.has_embeddings = False
        
        elif baseline_type in ['deepwalk', 'node2vec']:
            self.model = RandomWalkEmbedding(
                method=baseline_type,
                embedding_dim=kwargs.get('embedding_dim', 128),
                walk_length=kwargs.get('walk_length', 80),
                num_walks=kwargs.get('num_walks', 10),
                window_size=kwargs.get('window_size', 10),
                p=kwargs.get('p', 1.0),
                q=kwargs.get('q', 1.0)
            )
            self.needs_training = True
            self.has_embeddings = True
        
        elif baseline_type == 'pure_gnn':
            self.model = PureGNN(
                num_nodes=num_nodes,
                hidden_channels=kwargs.get('gnn_hidden_channels', 256),
                num_layers=kwargs.get('gnn_num_layers', 2),
                dropout=kwargs.get('dropout', 0.3),
                gnn_type=kwargs.get('gnn_type', 'gcn'),
                # Node feature parameters
                use_node_feats=kwargs.get('use_node_feats', False),
                num_node_feats=kwargs.get('num_node_feats', 0),
                train_node_emb=kwargs.get('train_node_emb', True),
                # Fairness parameters
                fairness_method=kwargs.get('fairness_method', 'none'),
                lambda_fair=kwargs.get('lambda_fair', 0.0),
                pa_aggregation=kwargs.get('pa_aggregation', 'product'),
                exposure_decay=kwargs.get('exposure_decay', 1.0)
            ).to(self.device)
            self.needs_training = True
            self.has_embeddings = True
            
            # Setup optimizer
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=kwargs.get('lr', 0.001)
            )
            
        elif baseline_type == 'lightgcn':
            self.model = LightGCNBaseline(
                num_nodes=num_nodes,
                embedding_dim=kwargs.get('embedding_dim', 64),
                num_layers=kwargs.get('gnn_num_layers', 2),
                alpha=kwargs.get('alpha', None)
            ).to(self.device)
            self.needs_training = True
            self.has_embeddings = True
            
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=kwargs.get('lr', 0.001)
            )
        
        elif baseline_type == 'random':
            self.model = RandomBaseline(seed=kwargs.get('seed', 42))
            self.needs_training = False
            self.has_embeddings = False
            
        elif baseline_type == 'ssnet':
            from .baseline_models import SSNet
            self.model = SSNet(
                num_nodes=num_nodes,
                hidden_channels=kwargs.get('gnn_hidden_channels', 64),
                num_layers=kwargs.get('gnn_num_layers', 1),
                use_node_feats=kwargs.get('use_node_feats', False),
                num_node_feats=kwargs.get('num_node_feats', 0),
                train_node_emb=kwargs.get('train_node_emb', True),
                fairness_method=kwargs.get('fairness_method', 'none'),
                lambda_fair=kwargs.get('lambda_fair', 0.0),
                pa_aggregation=kwargs.get('pa_aggregation', 'product'),
                exposure_decay=kwargs.get('exposure_decay', 1.0),
            ).to(self.device)
            self.needs_training = True
            self.has_embeddings = True

            # Setup optimizer (separate lr for embeddings vs MLP, matching original)
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=kwargs.get('lr', 0.001)
            )

        elif baseline_type == 'fairwalk':
            self.model = FairWalkEmbedding(
                embedding_dim=kwargs.get('embedding_dim', 128),
                walk_length=kwargs.get('walk_length', 80),
                num_walks=kwargs.get('num_walks', 10),
                window_size=kwargs.get('window_size', 10),
                p=kwargs.get('p', 1.0),
                q=kwargs.get('q', 1.0)
            )
            self.needs_training = True
            self.has_embeddings = True

        elif baseline_type == 'crosswalk':
            self.model = CrossWalkEmbedding(
                embedding_dim=kwargs.get('embedding_dim', 128),
                walk_length=kwargs.get('walk_length', 80),
                num_walks=kwargs.get('num_walks', 10),
                window_size=kwargs.get('window_size', 10),
                p_cross=kwargs.get('p_cross', 0.5)
            )
            self.needs_training = True
            self.has_embeddings = True

        elif baseline_type in ['uge-w', 'uge-r', 'uge-c']:
            self.model = UGEBaseline(
                num_nodes=num_nodes,
                hidden_channels=kwargs.get('gnn_hidden_channels', 128),
                out_channels=kwargs.get('embedding_dim', 64),
                num_layers=kwargs.get('gnn_num_layers', 2),
                dropout=kwargs.get('dropout', 0.3),
                gnn_type=kwargs.get('gnn_type', 'gcn'),
                debias_method=baseline_type,
                reg_weight=kwargs.get('reg_weight', 0.5)
            ).to(self.device)
            self.needs_training = True
            self.has_embeddings = True

            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=kwargs.get('lr', 0.001)
            )
        
        elif baseline_type == 'moral':
            self.needs_training = True
            self.has_embeddings = False # MORAL calculates scores on the fly or via embeddings, but Wrapper interface expects explicit get_embeddings
            # internally we will initialize MORAL in fit() because it needs full graph structure strongly typed
            self.model = None 

        elif baseline_type == 'fairadj':
            self.model = FairAdj(
                num_nodes=num_nodes,
                num_features=kwargs.get('num_node_feats', 0) if (kwargs.get('use_node_feats', True) and kwargs.get('num_node_feats', 0) > 0) else num_nodes,
                hidden_dim1=kwargs.get('gnn_hidden_channels', 32),
                hidden_dim2=kwargs.get('embedding_dim', 16),
                dropout=kwargs.get('dropout', 0.0),
                T1=kwargs.get('fairadj_T1', 50),
                T2=kwargs.get('fairadj_T2', 20),
                eta=kwargs.get('fairadj_eta', 0.2),
                outer_epochs=kwargs.get('fairadj_outer_epochs', 4),
                eq=kwargs.get('fairadj_eq', False),
                device=device
            )
            self.needs_training = True
            self.has_embeddings = True

        elif baseline_type == 'fairlp':
            self.model = FairLP_GNN(
                num_nodes=num_nodes,
                num_features=kwargs.get('num_node_feats', 0),
                hidden_dim=kwargs.get('gnn_hidden_channels', 64),
                out_dim=kwargs.get('embedding_dim', 64),
                num_layers=kwargs.get('gnn_num_layers', 2),
                dropout=kwargs.get('dropout', 0.5),
                density_mode=kwargs.get('density_mode', 'AvgD'),
                gnn_type=kwargs.get('gnn_type', 'gcn'),
                device=self.device
            ).to(self.device)
            self.needs_training = True
            self.has_embeddings = True
            self.lr = kwargs.get('lr', 0.01)

        elif baseline_type == 'fairlink':
            self.model = FairLinkModel(
                num_nodes=num_nodes,
                num_features=kwargs.get('num_node_feats', 0),
                hidden_dim=kwargs.get('gnn_hidden_channels', 64),
                out_dim=kwargs.get('embedding_dim', 64),
                num_layers=kwargs.get('gnn_num_layers', 2),
                dropout=kwargs.get('dropout', 0.5),
                alpha=kwargs.get('alpha', 1.0),
                beta=kwargs.get('beta', 1e-4),
                gamma=kwargs.get('gamma', 1.0),
                gnn_type=kwargs.get('gnn_type', 'gcn'),
                device=self.device
            ).to(self.device)
            self.needs_training = True
            self.has_embeddings = True
            self.lr = kwargs.get('lr', 0.01)

        elif baseline_type == 'lba':
            self.model = LinkBiasAmplification(
                num_nodes=num_nodes,
                num_features=kwargs.get('num_node_feats', 0) if (kwargs.get('use_node_feats', True) and kwargs.get('num_node_feats', 0) > 0) else num_nodes,
                hidden_dim=kwargs.get('gnn_hidden_channels', 128),
                out_dim=kwargs.get('embedding_dim', 64),
                num_layers=kwargs.get('gnn_num_layers', 4),
                conv_type=kwargs.get('conv_type', 'sym'),
                reg_weight=kwargs.get('reg_weight', 1.0),
                device=self.device
            ).to(self.device)
            self.needs_training = True
            self.has_embeddings = True
            self.lr = kwargs.get('lr', 0.01)

        elif baseline_type == 'cross_link_bias':
            self.model = CrossLinkBias(
                num_nodes=num_nodes,
                num_features=kwargs.get('num_node_feats', 0),
                hidden_dim=kwargs.get('embedding_dim', 64),
                out_dim=kwargs.get('embedding_dim', 64),
                num_layers=kwargs.get('gnn_num_layers', 2),
                aug_size=kwargs.get('aug_size', 1.25),
                ori_lr=kwargs.get('ori_lr', 0.001),
                aug_lr=kwargs.get('aug_lr', 0.001),
                alpha=kwargs.get('clb_alpha', 0.00005),
                beta=kwargs.get('clb_beta', 100),
                threshold=kwargs.get('clb_threshold', 50),
                batch_size=kwargs.get('batch_size', 4096),
                train_ratio=kwargs.get('train_ratio', 1.0),
                add_edge=kwargs.get('add_edge', False),
                device=self.device
            ).to(self.device)
            self.needs_training = True
            self.has_embeddings = True
            self.lr = kwargs.get('lr', 0.001)

        elif baseline_type == 'tailsteak':
            # TailSTEAK uses Node2Vec ID embeddings, not raw node features,
            # so always use embedding_dim for in_channels.
            ts_in_channels = kwargs.get('embedding_dim', 128)
            self.model = TailSTEAKBaseline(
                num_nodes=num_nodes,
                in_channels=ts_in_channels,
                hidden_channels=kwargs.get('gnn_hidden_channels', 32),
                num_layers=kwargs.get('gnn_num_layers', 1),
                heads=kwargs.get('heads', 4),
                ssl_temp=kwargs.get('ssl_temp', 0.5),
                ssl_reg=kwargs.get('ssl_reg', 1.0),
                deg_t_low=kwargs.get('deg_t_low', 15),
                U=kwargs.get('U', 500),
                base=kwargs.get('gnn_type', 'HGN'),
                lr=kwargs.get('lr', 0.01),
                weight_decay=kwargs.get('weight_decay', 0.0),
                ft_epoch=kwargs.get('ft_epoch', 2)
            ).to(self.device)
            self.needs_training = True
            self.has_embeddings = True

        elif baseline_type == 'ltlp':
            self.backbone_type = kwargs.get('ltlp_backbone', 'pure_gnn')
            if self.backbone_type == 'pure_gnn':
                self.model = PureGNN(
                    num_nodes=num_nodes,
                    hidden_channels=kwargs.get('gnn_hidden_channels', 256),
                    num_layers=kwargs.get('gnn_num_layers', 2),
                    dropout=kwargs.get('dropout', 0.3),
                    gnn_type=kwargs.get('gnn_type', 'gcn'),
                    use_node_feats=kwargs.get('use_node_feats', False),
                    num_node_feats=kwargs.get('num_node_feats', 0),
                    train_node_emb=kwargs.get('train_node_emb', True)
                ).to(self.device)
            elif self.backbone_type == 'lightgcn':
                self.model = LightGCNBaseline(
                    num_nodes=num_nodes,
                    embedding_dim=kwargs.get('embedding_dim', 64),
                    num_layers=kwargs.get('gnn_num_layers', 2)
                ).to(self.device)
            else:
                raise ValueError(f"Unknown LTLP backbone: {self.backbone_type}")

            self.needs_training = True
            self.has_embeddings = True
            
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=kwargs.get('lr', 0.001)
            )
            
            self.ltlp_epochs1 = kwargs.get('ltlp_epochs1', 100)
            self.ltlp_epochs2 = kwargs.get('ltlp_epochs2', 100)
            self.ltlp_K = kwargs.get('ltlp_K', 10)
            self.ltlp_phi = kwargs.get('ltlp_phi', 0.5)

        else:
            raise ValueError(f"Unknown baseline_type: {baseline_type}")
    
    def fit(self, data, split_edge, epochs=100, batch_size=16384, sens_attr=None, **kwargs):
        """Train the baseline model if needed"""
        train_edges = split_edge['train']['edge'].to(self.device)

        if self.baseline_type in ['common_neighbors', 'adamic_adar', 'resource_allocation', 'preferential_attachment']:
            print(f"\n[{self.baseline_type.upper()}] Building adjacency structure...")
            self.model.fit(train_edges, self.num_nodes)

        elif self.baseline_type in ['deepwalk', 'node2vec']:
            print(f"\n[{self.baseline_type.upper()}] Training embeddings...")
            self.model.fit(train_edges, self.num_nodes, device=self.device)

        elif self.baseline_type == 'pure_gnn':
            print(f"\n[PURE_GNN] Training GNN for {epochs} epochs...")
            self._train_gnn(data, split_edge, epochs, batch_size)
            
        elif self.baseline_type == 'ssnet':
            print(f"\n[SSNET] Training SSNet for {epochs} epochs...")
            self._train_ssnet(data, split_edge, epochs, batch_size)
            
        elif self.baseline_type == 'lightgcn':
            print(f"\n[LIGHTGCN] Training LightGCN for {epochs} epochs...")
            self._train_lightgcn(data, split_edge, epochs, batch_size)

        elif self.baseline_type == 'random':
            self.model.fit(train_edges, self.num_nodes)

        elif self.baseline_type == 'fairwalk':
            if sens_attr is None:
                raise ValueError("FairWalk requires sens_attr (sensitive attributes)")
            print(f"\n[FAIRWALK] Training fair walk embeddings...")
            self.model.fit(train_edges, self.num_nodes, sens_attr, device=self.device)

        elif self.baseline_type == 'crosswalk':
            if sens_attr is None:
                raise ValueError("CrossWalk requires sens_attr (sensitive attributes)")
            print(f"\n[CROSSWALK] Training cross-walk embeddings...")
            self.model.fit(train_edges, self.num_nodes, sens_attr, device=self.device)

        elif self.baseline_type in ['uge-w', 'uge-r', 'uge-c']:
            if sens_attr is None:
                raise ValueError(f"{self.baseline_type.upper()} requires sens_attr (sensitive attributes)")
            print(f"\n[{self.baseline_type.upper()}] Training UGE for {epochs} epochs...")
            self._train_uge(data, split_edge, sens_attr, epochs, batch_size)
            
        elif self.baseline_type == 'moral':
             from baselines.fairness_aware_baseline_models import train_moral
             
             # Create a dummy args object to pass to train_moral
             class Args:
                 pass
             args = Args()
             args.lr = self.kwargs.get('lr', 3e-4)
             args.weight_decay = self.kwargs.get('weight_decay', 0.0)
             args.emb_hidden_channels = self.kwargs.get('gnn_hidden_channels', 128)
             args.encoder = self.kwargs.get('encoder', 'gcn')
             args.predictor = self.kwargs.get('decoder', 'gae')
             args.batch_size = batch_size
             args.moral_epochs = epochs
             args.data_name = 'unknown' # or pass via kwargs
             args.data_path = '../../data/SNAP' # hardcode or pass via kwargs
             args.device = self.device
             
             # Handle sensitive attribute fallback if None
             # train_moral handles it internally via data_loader if sens not in data
             # but here we might have sens_attr passed explicitly
             if sens_attr is not None:
                 if not hasattr(data, 'sens'):
                     data.sens = torch.tensor(sens_attr, device=self.device) if not torch.is_tensor(sens_attr) else sens_attr.to(self.device)

             print(f"\n[MORAL] Initializing and training MORAL...")
             self.model, _ = train_moral(
                 base_model=self, # Pass self as mock base model
                 data=data,
                 split_edge=split_edge,
                 args=args
             )

        elif self.baseline_type == 'fairlp':
            if sens_attr is None:
                raise ValueError("FairLP requires sens_attr (sensitive attributes)")
            print(f"\n[FAIRLP] Training FairLP_GNN for {epochs} epochs...")
            features = getattr(data, 'x', None)
            if features is None:
                features = torch.eye(self.num_nodes, device=self.device)
            self.model.fit(train_edges, features, sens_attr, epochs=epochs, lr=self.lr)

        elif self.baseline_type == 'fairadj':
            if sens_attr is None:
                raise ValueError("FairAdj requires sens_attr (sensitive attributes)")
            print(f"\n[FAIRADJ] Training FairAdj for {epochs} epochs...")
            features = getattr(data, 'x', None)
            if features is None or not self.kwargs.get('use_node_feats', True):
                features = torch.eye(self.num_nodes, device=self.device)
            sens = sens_attr.to(self.device) if torch.is_tensor(sens_attr) else torch.tensor(sens_attr, device=self.device)
            fa_edges = train_edges.t() if train_edges.size(1) == 2 and train_edges.size(0) != 2 else train_edges
            self.model.fit(fa_edges, features, sens, epochs=epochs)

        elif self.baseline_type == 'fairlink':
            if sens_attr is None:
                raise ValueError("FairLink requires sens_attr (sensitive attributes)")
            print(f"\n[FAIRLINK] Training FairLinkModel for {epochs} epochs...")
            features = getattr(data, 'x', None)
            if features is None:
                features = torch.eye(self.num_nodes, device=self.device)
            self.model.fit(train_edges, features, sens_attr, epochs=epochs, lr=self.lr)

        elif self.baseline_type == 'lba':
            if sens_attr is None:
                raise ValueError("LBA requires sens_attr (sensitive attributes)")
            print(f"\n[LBA] Training LinkBiasAmplification for {epochs} epochs...")
            features = getattr(data, 'x', None)
            if features is None or not self.kwargs.get('use_node_feats', True):
                features = torch.eye(self.num_nodes, device=self.device)
            labels = getattr(data, 'y', None)
            
            if not torch.is_tensor(sens_attr):
                sens_attr = torch.tensor(sens_attr, device=self.device)
                
            lba_edges = train_edges.t() if train_edges.size(1) == 2 and train_edges.size(0) != 2 else train_edges
            self.model.fit(lba_edges, features, sens_attr, labels=labels, epochs=epochs, lr=self.lr)

        elif self.baseline_type == 'cross_link_bias':
            # CrossLinkBias does NOT need sens_attr -- it uses Louvain community detection internally
            print(f"\n[CROSS_LINK_BIAS] Training CrossLinkBias (Twins GNN) for {epochs} epochs...")
            edge_index = data.edge_index.to(self.device)
            # Pass validation edges for merge-phase inner validation (matches original)
            val_pos_edges = split_edge['valid']['edge'].to(self.device) if 'valid' in split_edge and 'edge' in split_edge['valid'] else None
            val_neg_edges = split_edge['valid']['edge_neg'].to(self.device) if 'valid' in split_edge and 'edge_neg' in split_edge['valid'] else None
            self.model.fit(edge_index, epochs=epochs, lr=self.lr,
                           val_pos_edges=val_pos_edges, val_neg_edges=val_neg_edges)

        elif self.baseline_type == 'tailsteak':
            print(f"\n[TAILSTEAK] Training dual-stage TailSTEAK for {epochs} epochs...")
            features = getattr(data, 'x', None)
            if features is None:
                # TailSTEAK ignores these input features entirely: forward()
                # overwrites x/ssl_x with its Node2Vec-initialised self.id_emb
                # ([num_nodes, in_channels]). The SocLoader only uses node_feat
                # for its num_nodes = node_feat.size(0) inference and to slice
                # node_feat[n_id] in the debias collate -- a slice that is built,
                # moved to device, then discarded. A dense eye(num_nodes) makes
                # that unused slice [batch, num_nodes]; on Brightkite (58k nodes)
                # it is ~80 GB and OOMs the CPU. A [num_nodes, 1] placeholder
                # keeps both uses correct at negligible cost and is numerically
                # identical (the slice is never read by the model).
                features = torch.zeros((self.num_nodes, 1), device=self.device)

            from baselines.tailsteak_dataloader import SocLoader
            graph_info = {
                'base_graph': [split_edge['train']['edge'].cpu().tolist()],
                'val_set': split_edge['valid']['edge'].cpu().tolist(),
                'test_set': split_edge['test']['edge'].cpu().tolist()
            }
            import copy
            # Store graph_info before SocLoader mutates it in-place
            self.model._graph_info = copy.deepcopy(graph_info)

            train_loader = SocLoader(features.cpu(), mode='train', debias_on=True,
                graph_path=graph_info,
                deg_t_low=self.model.deg_t_low, gamma=self.kwargs.get('gamma', 2),
                batch_size=batch_size, shuffle=True
            )
            self.model._node_features = features.cpu()
            self.model._gamma = self.kwargs.get('gamma', 2)
            val_pos_edges = split_edge['valid']['edge'].to(self.device) if 'valid' in split_edge and 'edge' in split_edge['valid'] else None
            val_neg_edges = split_edge['valid']['edge_neg'].to(self.device) if 'valid' in split_edge and 'edge_neg' in split_edge['valid'] else None
            train_edges = split_edge['train']['edge'].to(self.device)
            self.model.fit(train_loader, train_edges, epochs=epochs, device=self.device, val_pos_edges=val_pos_edges, val_neg_edges=val_neg_edges)

        elif self.baseline_type == 'ltlp':
            print(f"\n[LTLP] Training LTLP with {self.backbone_type} backbone...")
            self._train_ltlp(data, split_edge, batch_size)

    def _train_gnn(self, data, split_edge, epochs, batch_size):
        """Train the Pure GNN model with optional fairness loss"""
        self.model.train()
        
        # Get training edges
        pos_train_edge, neg_train_edge = get_pos_neg_edges(
            'train', split_edge,
            edge_index=data.edge_index,
            num_nodes=self.num_nodes,
            neg_sampler_name='global',
            num_neg=1
        )
        
        pos_train_edge = pos_train_edge.to(self.device)
        neg_train_edge = neg_train_edge.to(self.device)
        
        # Get degrees if using fairness
        if self.model.fairness_loss is not None:
            deg = torch.zeros(self.num_nodes, dtype=torch.float32, device=self.device)
            for u, v in pos_train_edge.tolist():
                if 0 <= u < self.num_nodes:
                    deg[u] += 1.0
                if 0 <= v < self.num_nodes:
                    deg[v] += 1.0
        else:
            deg = None
        
        best_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(1, epochs + 1):
            # Reset exposure if using exposure-based fairness
            if hasattr(self.model.fairness_loss, 'reset_exposure'):
                self.model.fairness_loss.reset_exposure(self.device)
            
            total_loss = 0
            total_base_loss = 0
            total_fair_penalty = 0
            total_examples = 0
            
            for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
                self.optimizer.zero_grad()

                # Forward pass
                # x = self.model.emb.weight # OLD
                x = self.model.create_input_feat(data)
                h = self.model(x, data.edge_index.to(self.device))

                # Get batch edges
                pos_edge = pos_train_edge[perm]
                neg_edge = neg_train_edge[perm].view(-1, 2)
                
                # Predictions
                pos_out = self.model.predict_batch(pos_edge, h)
                neg_out = self.model.predict_batch(neg_edge, h)
                
                # Compute loss
                if self.model.fairness_loss is not None:
                    # Fairness-aware loss
                    loss, base_loss, fair_penalty = self.model.fairness_loss(
                        pos_out, neg_out, num_neg=1,
                        pos_edge=pos_edge,
                        neg_edge=neg_edge.view(pos_edge.size(0), 1, 2),
                        degrees=deg
                    )
                    total_base_loss += base_loss * pos_edge.size(0)
                    total_fair_penalty += fair_penalty * pos_edge.size(0)
                else:
                    # Standard AUC loss
                    loss = auc_loss(pos_out, neg_out, num_neg=1)
                    total_base_loss += loss.item() * pos_edge.size(0)
                
                # Backward
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * pos_edge.size(0)
                total_examples += pos_edge.size(0)
            
            avg_loss = total_loss / total_examples
            avg_base = total_base_loss / total_examples
            avg_fair = total_fair_penalty / total_examples if self.model.fairness_loss else 0

            if epoch % 10 == 0:
                if self.model.fairness_loss is not None:
                    print(f"  Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f} "
                        f"(Base: {avg_base:.4f}, Fair: {avg_fair:.4f})")
                else:
                    print(f"  Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break
                    
    def _train_ssnet(self, data, split_edge, epochs, batch_size):
        """Train SSNet with BPR loss."""
        self.model.train()

        pos_train_edge, neg_train_edge = get_pos_neg_edges(
            'train', split_edge,
            edge_index=data.edge_index,
            num_nodes=self.num_nodes,
            neg_sampler_name='global',
            num_neg=1
        )
        pos_train_edge = pos_train_edge.to(self.device)
        neg_train_edge = neg_train_edge.to(self.device)

        # Fairness support
        if self.model.fairness_loss is not None:
            deg = torch.zeros(self.num_nodes, dtype=torch.float32, device=self.device)
            for u, v in pos_train_edge.tolist():
                if 0 <= u < self.num_nodes:
                    deg[u] += 1.0
                if 0 <= v < self.num_nodes:
                    deg[v] += 1.0
        else:
            deg = None

        best_loss = float('inf')
        patience_counter = 0
        patience = 20

        for epoch in range(1, epochs + 1):
            if hasattr(self.model.fairness_loss, 'reset_exposure'):
                self.model.fairness_loss.reset_exposure(self.device)

            total_loss = 0
            total_examples = 0

            for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
                self.optimizer.zero_grad()

                x = self.model.create_input_feat(data)
                h = self.model(x, data.edge_index.to(self.device))

                pos_edge = pos_train_edge[perm]
                neg_edge = neg_train_edge[perm].view(-1, 2)

                pos_out = self.model.predict_batch(pos_edge, h)
                neg_out = self.model.predict_batch(neg_edge, h)

                # BPR loss: mean(softplus(neg_score - pos_score))
                if self.model.fairness_loss is not None:
                    loss, _, _ = self.model.fairness_loss(
                        pos_out, neg_out, num_neg=1,
                        pos_edge=pos_edge,
                        neg_edge=neg_edge.view(pos_edge.size(0), 1, 2),
                        degrees=deg
                    )
                else:
                    loss = torch.mean(F.softplus(neg_out - pos_out))

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * pos_edge.size(0)
                total_examples += pos_edge.size(0)

            avg_loss = total_loss / total_examples

            if epoch % 10 == 0:
                print(f"  Epoch {epoch}/{epochs}, BPR Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

    def _train_lightgcn(self, data, split_edge, epochs, batch_size):
        """Train the LightGCN model"""        
        self.model.train()
        
        pos_train_edge, neg_train_edge = get_pos_neg_edges(
            'train', split_edge,
            edge_index=data.edge_index,
            num_nodes=self.num_nodes,
            neg_sampler_name='global',
            num_neg=1
        )
        
        pos_train_edge = pos_train_edge.to(self.device)
        neg_train_edge = neg_train_edge.to(self.device)
        
        best_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(1, epochs + 1):
            total_loss = 0
            total_examples = 0
            
            for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
                self.optimizer.zero_grad()
                
                # Get embeddings - using LightGCN model
                edge_index = data.edge_index.to(self.device)
                h = self.model(edge_index)
                
                pos_edge = pos_train_edge[perm]
                neg_edge = neg_train_edge[perm].view(-1, 2)
                
                pos_out = self.model.predict_batch(pos_edge, h)
                neg_out = self.model.predict_batch(neg_edge, h)
                
                # BPR loss
                loss = -torch.log(torch.sigmoid(pos_out - neg_out) + 1e-15).mean()
                
                # Optional: L2 regularization on embeddings
                l2_reg = h.norm(p=2)**2 / h.size(0)
                loss = loss + 1e-4 * l2_reg
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * pos_edge.size(0)
                total_examples += pos_edge.size(0)
                
            avg_loss = total_loss / total_examples
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
                
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break
                    
    def _train_ltlp(self, data, split_edge, batch_size):
        snap_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../SNAP'))
        if snap_path not in sys.path:
            sys.path.insert(0, snap_path)
            
        from utils import get_pos_neg_edges
        from loss import auc_loss
        from sklearn.metrics import roc_curve
        import torch.nn.functional as F

        epochs1 = self.ltlp_epochs1
        epochs2 = self.ltlp_epochs2
        
        self.model.train()
        
        pos_train_edge, neg_train_edge = get_pos_neg_edges(
            'train', split_edge,
            edge_index=data.edge_index,
            num_nodes=self.num_nodes,
            neg_sampler_name='global',
            num_neg=1
        )
        pos_train_edge = pos_train_edge.to(self.device)
        neg_train_edge = neg_train_edge.to(self.device)
        
        print(f"  [LTLP Phase 1] Pre-training for {epochs1} epochs...")
        
        adj = {i: set() for i in range(self.num_nodes)}
        for u, v in pos_train_edge.cpu().tolist():
            adj[u].add(v)
            adj[v].add(u)
            
        print("  [LTLP] Generating candidate set E_o...")
        E_o_set = set()
        for u, v in pos_train_edge.cpu().tolist():
            for i in adj[v]:
                if i != u and i not in adj[u]:
                    E_o_set.add((u, i))
            for j in adj[u]:
                if j != v and j not in adj[v]:
                    E_o_set.add((v, j))
                    
        E_o = torch.tensor(list(E_o_set), dtype=torch.long, device=self.device)
        print(f"  [LTLP] Candidate set E_o size: {E_o.size(0)}")
        
        candidate_scores_last_5 = []
        
        for epoch in range(1, epochs1 + 1):
            total_loss = 0
            total_examples = 0
            
            for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
                self.optimizer.zero_grad()
                
                if self.backbone_type == 'pure_gnn':
                    x = self.model.create_input_feat(data)
                    h = self.model(x, data.edge_index.to(self.device))
                else:
                    h = self.model(data.edge_index.to(self.device))
                    
                pos_edge = pos_train_edge[perm]
                neg_edge = neg_train_edge[perm].view(-1, 2)
                
                pos_out = self.model.predict_batch(pos_edge, h)
                neg_out = self.model.predict_batch(neg_edge, h)
                
                if self.backbone_type == 'pure_gnn':
                    loss = auc_loss(pos_out, neg_out, num_neg=1)
                else:
                    loss = -torch.log(torch.sigmoid(pos_out - neg_out) + 1e-15).mean()
                    
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * pos_edge.size(0)
                total_examples += pos_edge.size(0)
                
            if epoch > epochs1 - 5:
                self.model.eval()
                with torch.no_grad():
                    if self.backbone_type == 'pure_gnn':
                        x = self.model.create_input_feat(data)
                        h = self.model(x, data.edge_index.to(self.device))
                    else:
                        h = self.model(data.edge_index.to(self.device))
                        
                    scores = []
                    for i in range(0, E_o.size(0), batch_size):
                        batch_edges = E_o[i:i+batch_size]
                        s = self.model.predict_batch(batch_edges, h)
                        s = torch.sigmoid(s)
                        scores.append(s)
                    if scores:
                        candidate_scores_last_5.append(torch.cat(scores))
                self.model.train()
                
            if epoch % 10 == 0:
                print(f"    Epoch {epoch}/{epochs1}, Loss: {total_loss/total_examples:.4f}")
                
        print("  [LTLP] Calculating optimal threshold on validation set...")
        self.model.eval()
        with torch.no_grad():
            if self.backbone_type == 'pure_gnn':
                x = self.model.create_input_feat(data)
                h = self.model(x, data.edge_index.to(self.device))
            else:
                h = self.model(data.edge_index.to(self.device))
            
            val_pos = split_edge['valid']['edge'].to(self.device)
            val_neg = split_edge['valid']['edge_neg'].to(self.device)
            
            val_pos_scores = self.model.predict_batch(val_pos, h)
            val_neg_scores = self.model.predict_batch(val_neg, h)
            
            val_pos_scores = torch.sigmoid(val_pos_scores)
            val_neg_scores = torch.sigmoid(val_neg_scores)
                
            labels = torch.cat([torch.ones(val_pos.size(0)), torch.zeros(val_neg.size(0))]).cpu().numpy()
            preds = torch.cat([val_pos_scores, val_neg_scores]).cpu().numpy()
            
            # Avoid ValueError when only one class is present if sizes are 0
            if len(np.unique(labels)) > 1:
                fpr, tpr, thresholds = roc_curve(labels, preds)
                optimal_idx = np.argmax(tpr - fpr)
                tau = thresholds[optimal_idx]
            else:
                tau = 0.5
            
        print(f"  [LTLP] Optimal threshold tau: {tau:.4f}")
        
        if candidate_scores_last_5:
            candidate_scores_tensor = torch.stack(candidate_scores_last_5)
            mean_scores = candidate_scores_tensor.mean(dim=0)
            
            mask_score = mean_scores >= tau
            E_s = E_o[mask_score]
            E_s_scores = candidate_scores_tensor[:, mask_score]
            E_s_mean = mean_scores[mask_score]
            print(f"  [LTLP] Edges after score filtering E_s: {E_s.size(0)}")
            
            if E_s.size(0) > 0:
                variances = ((E_s_scores - E_s_mean.unsqueeze(0))**2).mean(dim=0)
                norm_variances = torch.sqrt(variances) / (E_s_mean + 1e-8)
                
                k_num = int(E_s.size(0) * (self.ltlp_K / 100.0))
                if k_num > 0:
                    _, indices = torch.topk(norm_variances, k_num, largest=False)
                    E_f = E_s[indices]
                    print(f"  [LTLP] Edges after variance filtering E_f: {E_f.size(0)}")
                    
                    E_f_symmetric = torch.cat([E_f, E_f.flip(-1)], dim=0)
                    aug_edge_index = torch.cat([data.edge_index.to(self.device), E_f_symmetric.t()], dim=1)
                else:
                    aug_edge_index = data.edge_index.to(self.device)
            else:
                aug_edge_index = data.edge_index.to(self.device)
        else:
            aug_edge_index = data.edge_index.to(self.device)
            
        self.aug_edge_index = aug_edge_index
        
        print(f"  [LTLP Phase 2] Continued training for {epochs2} epochs...")
        self.model.train()
        
        for epoch in range(1, epochs2 + 1):
            total_loss = 0
            total_examples = 0
            
            for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
                self.optimizer.zero_grad()
                
                if self.backbone_type == 'pure_gnn':
                    x = self.model.create_input_feat(data)
                    h = self.model(x, aug_edge_index)
                else:
                    h = self.model(aug_edge_index)
                    
                pos_edge = pos_train_edge[perm]
                neg_edge = neg_train_edge[perm].view(-1, 2)
                
                pos_out = self.model.predict_batch(pos_edge, h)
                neg_out = self.model.predict_batch(neg_edge, h)
                
                if self.backbone_type == 'pure_gnn':
                    l_ce = auc_loss(pos_out, neg_out, num_neg=1)
                else:
                    l_ce = -torch.log(torch.sigmoid(pos_out - neg_out) + 1e-15).mean()
                    
                z_pos = h[pos_edge[:, 0]] * h[pos_edge[:, 1]]
                z_neg = h[neg_edge[:, 0]] * h[neg_edge[:, 1]]
                
                c_pos = z_pos.mean(dim=0).detach()
                c_neg = z_neg.mean(dim=0).detach()
                
                l_re_pos = ((z_pos - c_pos)**2).sum(dim=1).mean()
                l_re_neg = ((z_neg - c_neg)**2).sum(dim=1).mean()
                l_re = l_re_pos + l_re_neg
                
                loss = self.ltlp_phi * l_ce + (1 - self.ltlp_phi) * l_re
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * pos_edge.size(0)
                total_examples += pos_edge.size(0)
                
            if epoch % 10 == 0:
                print(f"    Epoch {epoch}/{epochs2}, Loss: {total_loss/total_examples:.4f}")
                
    def _train_uge(self, data, split_edge, sens_attr, epochs, batch_size):
        """Train the UGE model with optional edge reweighting and regularization."""
        self.model.train()

        # Get training edges
        pos_train_edge, neg_train_edge = get_pos_neg_edges(
            'train', split_edge,
            edge_index=data.edge_index,
            num_nodes=self.num_nodes,
            neg_sampler_name='global',
            num_neg=1
        )

        pos_train_edge = pos_train_edge.to(self.device)
        neg_train_edge = neg_train_edge.to(self.device)

        # Compute edge weights for UGE-W and UGE-C
        edge_weight = None
        if self.model.debias_method in ['uge-w', 'uge-c']:
            edge_weight = self.model.compute_edge_weights(
                data.edge_index.to(self.device), sens_attr
            )
            print(f"  [UGE] Edge weights computed: mean={edge_weight.mean():.4f}, "
                  f"std={edge_weight.std():.4f}")

        # Move sens_attr to device for regularization
        if torch.is_tensor(sens_attr):
            sens_attr_device = sens_attr.to(self.device)
        else:
            sens_attr_device = torch.tensor(sens_attr, dtype=torch.float32, device=self.device)

        best_loss = float('inf')
        patience_counter = 0
        patience = 20

        for epoch in range(1, epochs + 1):
            total_loss = 0
            total_base_loss = 0
            total_reg_loss = 0
            total_examples = 0

            for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
                self.optimizer.zero_grad()

                # Forward pass
                x = self.model.emb.weight
                h = self.model(x, data.edge_index.to(self.device), edge_weight=edge_weight)

                # Get batch edges
                pos_edge = pos_train_edge[perm]
                neg_edge = neg_train_edge[perm].view(-1, 2)

                # Predictions (dot product)
                pos_out = (h[pos_edge[:, 0]] * h[pos_edge[:, 1]]).sum(dim=1)
                neg_out = (h[neg_edge[:, 0]] * h[neg_edge[:, 1]]).sum(dim=1)

                # BPR loss (from UGE paper)
                diff = pos_out - neg_out.view(pos_edge.size(0), -1)
                base_loss = -torch.log(torch.sigmoid(diff) + 1e-15).mean()

                loss = base_loss
                total_base_loss += base_loss.item() * pos_edge.size(0)

                # Regularization loss for UGE-R and UGE-C
                if self.model.debias_method in ['uge-r', 'uge-c']:
                    reg_loss = self.model.compute_regularization_loss(h, sens_attr_device)
                    loss = loss + self.model.reg_weight * reg_loss
                    total_reg_loss += reg_loss.item() * pos_edge.size(0)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * pos_edge.size(0)
                total_examples += pos_edge.size(0)

            avg_loss = total_loss / total_examples
            avg_base = total_base_loss / total_examples
            avg_reg = total_reg_loss / total_examples if self.model.debias_method in ['uge-r', 'uge-c'] else 0

            if epoch % 10 == 0:
                if self.model.debias_method in ['uge-r', 'uge-c']:
                    print(f"  Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f} "
                          f"(Base: {avg_base:.4f}, Reg: {avg_reg:.6f})")
                else:
                    print(f"  Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        # Store final embeddings
        self.model.eval()
        with torch.no_grad():
            x = self.model.emb.weight
            h = self.model(x, data.edge_index.to(self.device), edge_weight=edge_weight)
            self.model._embeddings = h

    def get_embeddings(self, data=None):
        """Get node embeddings"""
        if not self.has_embeddings:
            raise ValueError(f"{self.baseline_type} does not produce embeddings")

        if self.baseline_type in ['deepwalk', 'node2vec', 'fairwalk', 'crosswalk', 'fairlp', 'fairlink', 'lba', 'cross_link_bias', 'tailsteak', 'fairadj']:
            return self.model.get_embeddings()

        elif self.baseline_type in ['pure_gnn', 'ssnet', 'ltlp']:
            if self.baseline_type == 'ltlp' and self.backbone_type == 'lightgcn':
                self.model.eval()
                with torch.no_grad():
                    edge_index = getattr(self, 'aug_edge_index', data.edge_index).to(self.device)
                    h = self.model(edge_index)
                return h
            else:
                self.model.eval()
                with torch.no_grad():
                    # x = self.model.emb.weight # OLD
                    x = self.model.create_input_feat(data)
                    edge_index = getattr(self, 'aug_edge_index', data.edge_index).to(self.device) if self.baseline_type == 'ltlp' else data.edge_index.to(self.device)
                    h = self.model(x, edge_index)
                return h
            
        elif self.baseline_type == 'lightgcn':
            self.model.eval()
            with torch.no_grad():
                h = self.model(data.edge_index.to(self.device))
            return h

        elif self.baseline_type in ['uge-w', 'uge-r', 'uge-c']:
            return self.model.get_embeddings()
    
    @torch.no_grad()
    def batch_predict(self, h, edge_batch, batch_size):
        """Predict scores for edge batch"""
        all_scores = []

        for i in range(0, edge_batch.size(0), batch_size):
            batch = edge_batch[i:i+batch_size].to(self.device)

            if self.baseline_type in ['common_neighbors', 'adamic_adar', 'resource_allocation', 'preferential_attachment']:
                scores = self.model.predict_batch(batch)

            elif self.baseline_type == 'moral':
                scores = self.model.predict_scores(batch)

            elif self.baseline_type in ['deepwalk', 'node2vec', 'fairwalk', 'crosswalk',
                                         'uge-w', 'uge-r', 'uge-c', 'fairlp', 'fairlink', 'lba',
                                         'cross_link_bias', 'tailsteak', 'fairadj']:
                scores = self.model.predict_batch(batch)

            elif self.baseline_type in ['pure_gnn', 'lightgcn', 'ssnet', 'ltlp']:
                scores = self.model.predict_batch(batch, h)

            elif self.baseline_type == 'random':
                scores = self.model.predict_batch(batch)

            all_scores.append(scores.cpu())

        return torch.cat(all_scores, dim=0)
    
    def test(self, data, split_edge, batch_size, evaluator, eval_metric='all', fairness_mode='indegree',
             use_equality_gain=False, use_centrality_gain=False, centrality_types=None):
        """Evaluate baseline model with comprehensive metrics"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"\n[{self.baseline_type.upper()}] Evaluating...")

        # Get embeddings if applicable
        if self.has_embeddings:
            h = self.get_embeddings(data)
            # For compatibility, add mean node
            mean_h = torch.mean(h, dim=0, keepdim=True)
            h_with_mean = torch.cat([h, mean_h], dim=0)
        else:
            h_with_mean = None

        # Get test edges
        pos_test_edge = split_edge['test']['edge'].to(self.device)
        neg_test_edge = split_edge['test']['edge_neg'].to(self.device)

        # Compute predictions -- MORAL uses predict_scores() for per-edge scoring
        if self.baseline_type == 'moral':
            pos_test_pred = self.model.predict_scores(pos_test_edge).cpu()
            neg_test_pred = self.model.predict_scores(neg_test_edge.view(-1, 2)).cpu()
        else:
            pos_test_pred = self.batch_predict(h_with_mean, pos_test_edge, batch_size)
            neg_test_pred = self.batch_predict(h_with_mean, neg_test_edge.view(-1, 2), batch_size)

        # Compute Hits@K and MRR
        M = neg_test_pred.numel() // pos_test_pred.numel()
        neg = neg_test_pred.view(-1, M)
        ranks = 1 + (neg > pos_test_pred.view(-1, 1)).sum(dim=1)

        results = {'MRR': (0.0, (1.0 / ranks.float()).mean().item())}
        for K in [10, 20, 50, 100]:
            results[f'Hits@{K}'] = (0.0, (ranks <= K).float().mean().item())

        # Compute Precision@K and Fairness@K
        train_edges = split_edge['train']['edge'].to(self.device)
        test_nodes = set(pos_test_edge[:, 0].tolist()) | set(pos_test_edge[:, 1].tolist())
        test_nodes = sorted([u for u in test_nodes if u < self.num_nodes])

        # Generate recommendations
        if self.has_embeddings:
            from core.evaluation import build_recommendations, make_predictor_score_fn
            
            # PureGNN has a trained MLP predictor (concat-based), use it for scoring
            # Other embedding baselines (deepwalk, node2vec, etc.) use dot product (correct for them)
            if self.baseline_type == 'pure_gnn' or (self.baseline_type == 'ltlp' and self.backbone_type == 'pure_gnn'):
                score_fn = make_predictor_score_fn(
                    self.model.predictor, predictor_type='concat'
                )
            elif self.baseline_type == 'tailsteak':
                score_fn = make_predictor_score_fn(
                    self.model.model.pred_head, predictor_type='concat'
                )
            else:
                score_fn = None  # dot product for deepwalk, node2vec, fairwalk, etc.
            
            recs_by_u = build_recommendations(
                h, train_edges, test_nodes, k=100, device=self.device,
                score_fn=score_fn
            )
        elif self.baseline_type == 'moral':
            # MORAL predict_topk: (train_edges, test_nodes, k)
            recs_by_u = self.model.predict_topk(
                train_edges, test_nodes, k=100
            )
        else:
            recs_by_u = self.model.predict_topk(
                None, train_edges, test_nodes, k=100
            )

        # Compute Precision and Distribution Fairness
        from metrics.precision import compute_metrics_from_recs
        rec_results = compute_metrics_from_recs(
            recs_by_u, pos_test_edge, self.num_nodes, fairness_mode
        )
        results.update(rec_results)

        # Classification metrics (AUC-ROC, F1, Accuracy)
        from metrics.classification import compute_classification_metrics
        cls_metrics = compute_classification_metrics(pos_test_pred, neg_test_pred)
        results.update(cls_metrics)

        # Equality Gain
        if use_equality_gain:
            from metrics.equality_gain import compute_equality_gain_metrics
            eq_metrics = compute_equality_gain_metrics(
                recs_by_u, train_edges, pos_test_edge, self.num_nodes,
                k_values=[10, 20, 50, 100]
            )
            results.update(eq_metrics)

        # Centrality-based Equality Gain (pagerank, closeness, etc.)
        if use_centrality_gain:
            from metrics.centrality_equality_gain import compute_all_centrality_equality_gains
            cent_metrics = compute_all_centrality_equality_gains(
                recs_by_u, train_edges, pos_test_edge, self.num_nodes,
                k_values=[10, 20, 50, 100],
                centrality_types=centrality_types or ['pagerank', 'closeness'],
                use_sampling_closeness=True
            )
            results.update(cent_metrics)
            
        # Demographic Metrics
        # Check if sensitive attribute is available
        sens = getattr(data, 'sens', None)
        features = getattr(data, 'x', None)
        
        # Get embeddings for metrics if available
        # For MORAL, use final_embeddings
        metric_emb = locals().get('h', None)
        if self.baseline_type == 'moral' and hasattr(self.model, 'final_embeddings'):
             metric_emb = self.model.final_embeddings
             
        if sens is not None:
            from metrics.demographics import compute_demographic_metrics, compute_demographic_parity_ratio_change
            
            demo_metrics = compute_demographic_metrics(
                recs_by_u=recs_by_u,
                train_edges=train_edges,
                test_edges=pos_test_edge,
                num_nodes=self.num_nodes,
                sens=sens,
                h_emb=metric_emb, 
                features=features,
                adj=None, # Will be built from edges
                k=10
            )
            results.update(demo_metrics)
            
            # Delta DPR
            dpr_change = compute_demographic_parity_ratio_change(
                train_edges, recs_by_u, self.num_nodes, sens, k=10
            )
            results['Delta_DPR'] = dpr_change

        # NDKL Metrics
        from metrics.ndkl import compute_ndkl_metrics
        from torch_geometric.utils import degree
        
        # Compute degrees from train_edges for NDKL categories
        if torch.is_tensor(train_edges):
            # Assume undirected, so count both source and target
            # If train_edges is [E, 2] which is typical for split_edge['train']['edge']
            if train_edges.shape[1] == 2 and train_edges.shape[0] != 2:
                te_t = train_edges.t()
            else:
                te_t = train_edges
                
            row, col = te_t
            d = degree(row, self.num_nodes)
            
        else:
             # numpy fallback
            d = torch.zeros(self.num_nodes)
            unique, counts = np.unique(train_edges[0], return_counts=True)
            d[unique] = torch.from_numpy(counts).float()
            
        ndkl_res = compute_ndkl_metrics(
            recs_by_u=recs_by_u,
            degrees=d,
            train_edges=te_t,
            h_emb=metric_emb,
            k_values=[10, 20, 50, 100],
            ndkl_mode='per_user'
        )
        # Flatten NDKL results
        if 'ndkl_scores' in ndkl_res:
            for k, val in ndkl_res['ndkl_scores'].items():
                results[f'NDKL@{k}'] = val

        return results


def create_baseline_model(baseline_type, data_config, **kwargs):
    """
    Factory function to create baseline models.
    
    Args:
        baseline_type: Type of baseline
        data_config: DataConfig object with dataset parameters
        **kwargs: Additional model parameters
    
    Returns:
        BaselineWrapper instance
    """
    return BaselineWrapper(
        baseline_type=baseline_type,
        num_nodes=kwargs.pop('num_nodes', 1000),
        device=data_config.device,
        **kwargs
    )