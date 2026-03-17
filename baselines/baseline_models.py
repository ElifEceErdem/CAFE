"""
Baseline Models for Link Prediction

Implements various baseline methods:
- Heuristic: Common Neighbors, Adamic-Adar, Resource Allocation
- Embedding: DeepWalk, Node2Vec
- GNN: Pure GNN (without HLGNN's advanced features)
- Random: Random baseline
"""

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
import numpy as np
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn.models import LightGCN
from torch_geometric.utils import to_undirected
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from gensim.models import Word2Vec
from tqdm import tqdm


# ==================== HEURISTIC BASELINES ====================

class HeuristicBaseline:
    """Base class for heuristic-based link prediction methods"""
    
    def __init__(self, method='common_neighbors'):
        """
        Args:
            method: 'common_neighbors', 'adamic_adar', 'resource_allocation', 'preferential_attachment'
        """
        self.method = method
        self.graph = None
        self.adj_dict = None
    
    def fit(self, train_edges, num_nodes):
        """Build adjacency structure from training edges"""
        self.num_nodes = num_nodes
        
        # Build adjacency dictionary for fast neighbor lookup
        self.adj_dict = {i: set() for i in range(num_nodes)}
        
        for u, v in train_edges.cpu().tolist():
            if u < num_nodes and v < num_nodes:
                self.adj_dict[u].add(v)
                self.adj_dict[v].add(u)
        
        # Precompute node degrees for Adamic-Adar, Resource Allocation, and Preferential Attachment
        if self.method in ['adamic_adar', 'resource_allocation', 'preferential_attachment']:
            self.degrees = {i: len(neighbors) for i, neighbors in self.adj_dict.items()}
    
    def predict_batch(self, edge_batch):
        """Predict scores for a batch of edges"""
        scores = []
        
        for u, v in edge_batch.cpu().tolist():
            if u >= self.num_nodes or v >= self.num_nodes:
                scores.append(0.0)
                continue
            
            # Get common neighbors
            neighbors_u = self.adj_dict.get(u, set())
            neighbors_v = self.adj_dict.get(v, set())
            common = neighbors_u & neighbors_v
            
            if self.method == 'common_neighbors':
                score = float(len(common))
            
            elif self.method == 'adamic_adar':
                score = sum(1.0 / np.log(self.degrees[w] + 1e-8) 
                           for w in common if self.degrees.get(w, 0) > 1)
            
            elif self.method == 'resource_allocation':
                score = sum(1.0 / (self.degrees.get(w, 0) + 1e-8) for w in common)
            
            elif self.method == 'preferential_attachment':
                score = float(self.degrees.get(u, 0) * self.degrees.get(v, 0))
            
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            scores.append(score)
        
        return torch.tensor(scores, dtype=torch.float32)
    
    def predict_topk(self, h_emb, train_edges, test_nodes, k=100):
        """
        Generate top-k recommendations for each test node.
        
        Args:
            h_emb: Not used for heuristic methods (kept for interface compatibility)
            train_edges: Training edges
            test_nodes: Nodes to generate recommendations for
            k: Number of recommendations per node
        
        Returns:
            recs_by_u: dict mapping node_id -> list of top-k nodes
        """
        from core.data_loader import build_train_adjacency
        
        # Ensure model is fitted
        if self.adj_dict is None:
            self.fit(train_edges, h_emb.size(0))
        
        train_adj = build_train_adjacency(train_edges, self.num_nodes)
        recs_by_u = {}
        
        print(f"[{self.method.upper()}] Generating recommendations for {len(test_nodes)} nodes...")
        
        for idx, u in enumerate(test_nodes):
            # Compute scores for all candidate nodes
            candidates = []
            for v in range(self.num_nodes):
                # Skip self-loops and training edges
                if v == u or v in train_adj[u]:
                    continue
                
                # Compute heuristic score
                neighbors_u = self.adj_dict.get(u, set())
                neighbors_v = self.adj_dict.get(v, set())
                common = neighbors_u & neighbors_v
                
                if self.method == 'common_neighbors':
                    score = float(len(common))
                elif self.method == 'adamic_adar':
                    score = sum(1.0 / np.log(self.degrees.get(w, 1) + 1) 
                               for w in common if self.degrees.get(w, 0) > 1)
                elif self.method == 'resource_allocation':
                    score = sum(1.0 / (self.degrees.get(w, 0) + 1e-8) for w in common)
                elif self.method == 'preferential_attachment':
                    score = float(self.degrees.get(u, 0) * self.degrees.get(v, 0))
                
                if score > 0:  # Only keep candidates with non-zero score
                    candidates.append((v, score))
            
            # Sort by score and take top-k
            candidates.sort(key=lambda x: x[1], reverse=True)
            recs_by_u[u] = [v for v, _ in candidates[:k]]
            
            # If not enough candidates, fill with random nodes
            if len(recs_by_u[u]) < k:
                remaining = k - len(recs_by_u[u])
                used = set(recs_by_u[u]) | {u} | train_adj[u]
                available = [v for v in range(self.num_nodes) if v not in used]
                if available:
                    additional = np.random.choice(available, 
                                                 min(remaining, len(available)), 
                                                 replace=False).tolist()
                    recs_by_u[u].extend(additional)
            
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{len(test_nodes)} nodes...")
        
        return recs_by_u


# ==================== EMBEDDING BASELINES ====================

class RandomWalkEmbedding:
    """DeepWalk and Node2Vec embeddings"""
    
    def __init__(self, method='deepwalk', embedding_dim=128, walk_length=80, 
                 num_walks=10, window_size=10, p=1.0, q=1.0):
        """
        Args:
            method: 'deepwalk' or 'node2vec'
            embedding_dim: Dimension of node embeddings
            walk_length: Length of each random walk
            num_walks: Number of walks per node
            window_size: Context window size for Word2Vec
            p: Return parameter (node2vec only)
            q: In-out parameter (node2vec only)
        """
        self.method = method
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.p = p
        self.q = q
        self.embeddings = None
    
    def _random_walk(self, graph, start_node):
        """Perform a random walk starting from start_node"""
        walk = [start_node]
        
        for _ in range(self.walk_length - 1):
            current = walk[-1]
            neighbors = list(graph[current])
            
            if not neighbors:
                break
            
            if self.method == 'deepwalk':
                # Uniform sampling
                next_node = np.random.choice(neighbors)
            
            elif self.method == 'node2vec':
                # Biased sampling based on p and q
                if len(walk) == 1:
                    next_node = np.random.choice(neighbors)
                else:
                    prev_node = walk[-2]
                    weights = []
                    
                    for neighbor in neighbors:
                        if neighbor == prev_node:
                            # Return to previous node
                            weight = 1.0 / self.p
                        elif neighbor in graph[prev_node]:
                            # Stay in local neighborhood
                            weight = 1.0
                        else:
                            # Move outward
                            weight = 1.0 / self.q
                        weights.append(weight)
                    
                    weights = np.array(weights)
                    weights = weights / weights.sum()
                    next_node = np.random.choice(neighbors, p=weights)
            
            walk.append(next_node)
        
        return walk
    
    def fit(self, train_edges, num_nodes, device='cpu'):
        """Generate random walks and train Word2Vec"""
        print(f"[{self.method.upper()}] Generating random walks...")
        
        # Build graph
        graph = {i: set() for i in range(num_nodes)}
        for u, v in train_edges.cpu().tolist():
            if u < num_nodes and v < num_nodes:
                graph[u].add(v)
                graph[v].add(u)
        
        # Generate walks
        walks = []
        for node in range(num_nodes):
            for _ in range(self.num_walks):
                walk = self._random_walk(graph, node)
                walks.append([str(n) for n in walk])
        
        print(f"Generated {len(walks)} walks, training Word2Vec...")
        
        # Train Word2Vec
        model = Word2Vec(
            sentences=walks,
            vector_size=self.embedding_dim,
            window=self.window_size,
            min_count=0,
            sg=1,  # Skip-gram
            workers=4,
            epochs=5
        )
        
        # Extract embeddings
        embeddings = np.zeros((num_nodes, self.embedding_dim))
        for i in range(num_nodes):
            if str(i) in model.wv:
                embeddings[i] = model.wv[str(i)]
        
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32, device=device)
        print(f"Embeddings trained: shape {self.embeddings.shape}")
    
    def predict_batch(self, edge_batch):
        """Predict scores using dot product of embeddings"""
        if self.embeddings is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        u_emb = self.embeddings[edge_batch[:, 0]]
        v_emb = self.embeddings[edge_batch[:, 1]]
        # Remove Sigmoid: Return raw logits (dot product) to avoid saturation during re-ranking
        # Normalization (Min-Max) will be handled by the re-ranker
        scores = (u_emb * v_emb).sum(dim=1)
        return scores
    
    def get_embeddings(self):
        """Return learned embeddings"""
        return self.embeddings


# ==================== GNN BASELINE ====================

# In baseline_models.py, update PureGNN class

class PureGNN(torch.nn.Module):
    """Pure GNN without HLGNN's advanced features - NOW WITH FAIRNESS SUPPORT"""

    def __init__(self, num_nodes, hidden_channels=256, num_layers=2,
                 dropout=0.3, gnn_type='gcn',
                 # Node feature parameters
                 use_node_feats=False, num_node_feats=0, train_node_emb=True,
                 # Fairness parameters
                 fairness_method='none', lambda_fair=0.0,
                 pa_aggregation='product', exposure_decay=1.0):
        """
        Args:
            use_node_feats: Whether to use raw node features as input
            num_node_feats: Dimension of raw node features
            train_node_emb: Whether to also train learnable node embeddings
            fairness_method: 'none', 'inproc_pa', 'inproc_exposure', 'inproc_hybrid'
            lambda_fair: Fairness penalty strength
            pa_aggregation: PA score aggregation for inproc_pa
            exposure_decay: Decay factor for inproc_exposure
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        self.fairness_method = fairness_method
        self.use_node_feats = use_node_feats
        self.train_node_emb = train_node_emb

        # Input layer: node features, embeddings, or both
        if use_node_feats and num_node_feats > 0:
            input_dim = num_node_feats
            if train_node_emb:
                self.emb = torch.nn.Embedding(num_nodes, hidden_channels)
                input_dim += hidden_channels
            else:
                self.emb = None
            self.feat_proj = torch.nn.Linear(input_dim, hidden_channels)
        else:
            self.emb = torch.nn.Embedding(num_nodes, hidden_channels)
            self.feat_proj = None

        # GNN layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            if gnn_type == 'gcn':
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            elif gnn_type == 'sage':
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            else:
                raise ValueError(f"Unknown gnn_type: {gnn_type}")

        # MLP predictor
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, 1)
        )
        
        # === NEW: Fairness Loss ===
        if fairness_method != 'none':
            # Import fairness losses
            import sys
            import os
            fairness_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../fairness_methods'))
            if fairness_path not in sys.path:
                sys.path.append(fairness_path)
            
            from inprocessing import PAFairnessLoss, ExposureFairnessLoss, HybridFairnessLoss
            
            if fairness_method == 'inproc_pa':
                self.fairness_loss = PAFairnessLoss(
                    base_loss_name='AUC',
                    lambda_fair=lambda_fair,
                    pa_aggregation=pa_aggregation
                )
            elif fairness_method == 'inproc_exposure':
                self.fairness_loss = ExposureFairnessLoss(
                    base_loss_name='AUC',
                    lambda_fair=lambda_fair,
                    num_nodes=num_nodes,
                    exposure_decay=exposure_decay
                )
            elif fairness_method == 'inproc_hybrid':
                self.fairness_loss = HybridFairnessLoss(
                    base_loss_name='AUC',
                    lambda_fair=lambda_fair,
                    pa_aggregation=pa_aggregation,
                    num_nodes=num_nodes,
                    exposure_decay=exposure_decay
                )
        else:
            self.fairness_loss = None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.emb is not None:
            self.emb.reset_parameters()
        if self.feat_proj is not None:
            self.feat_proj.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for layer in self.predictor:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def create_input_feat(self, data):
        """Create input features, optionally combining node features with embeddings."""
        if self.use_node_feats and hasattr(data, 'x') and data.x is not None:
            input_feat = data.x
            if self.train_node_emb and self.emb is not None:
                input_feat = torch.cat([self.emb.weight, input_feat], dim=-1)
            input_feat = self.feat_proj(input_feat)
        else:
            input_feat = self.emb.weight
        return input_feat

    def forward(self, x, edge_index):
        """Forward pass through GNN"""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
    def predict_batch(self, edge_batch, h):
        """Predict scores for edge batch"""
        h_u = h[edge_batch[:, 0]]
        h_v = h[edge_batch[:, 1]]
        h_cat = torch.cat([h_u, h_v], dim=1)
        return self.predictor(h_cat).squeeze()

class LightGCNBaseline(torch.nn.Module):
    """LightGCN baseline using PyTorch Geometric"""

    def __init__(self, num_nodes, embedding_dim=64, num_layers=2, **kwargs):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        self.model = LightGCN(
            num_nodes=num_nodes,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            **kwargs
        )

    def forward(self, edge_index):
        return self.model.get_embedding(edge_index)

    def predict_batch(self, edge_batch, h):
        """Predict scores for edge batch via dot product"""
        h_u = h[edge_batch[:, 0]]
        h_v = h[edge_batch[:, 1]]
        return (h_u * h_v).sum(dim=-1)

class SSNet(torch.nn.Module):
    """
    Friend Recommendations with Self-Rescaling Graph Neural Networks (KDD '22)

    Uses a LightGCN backbone (no activation/dropout between layers, symmetric
    normalized adjacency) with a learned self-rescaling MLP applied to the
    final node embeddings. Scoring is via dot product.

    Matches the original implementation at /var/scratch/eerdem/SSNet/:
    - LightGCN: message passing with edge weight = 1/sqrt(d_u * d_v), no nonlinearities
    - Scaling MLP: Linear(64->32) -> Tanh -> Linear(32->1, no bias) -> Sigmoid
    - No tao division (use_tao defaults to 0/False in original config)
    - BPR loss: mean(softplus(neg_score - pos_score))
    """
    def __init__(self, num_nodes, hidden_channels=64, num_layers=1,
                 use_node_feats=False, num_node_feats=0, train_node_emb=True,
                 fairness_method='none', lambda_fair=0.0,
                 pa_aggregation='product', exposure_decay=1.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.use_node_feats = use_node_feats
        self.train_node_emb = train_node_emb
        self.fairness_method = fairness_method

        # Input layer: node features, embeddings, or both
        if use_node_feats and num_node_feats > 0:
            input_dim = num_node_feats
            if train_node_emb:
                self.emb = torch.nn.Embedding(num_nodes, hidden_channels)
                input_dim += hidden_channels
            else:
                self.emb = None
            self.feat_proj = torch.nn.Linear(input_dim, hidden_channels)
        else:
            self.emb = torch.nn.Embedding(num_nodes, hidden_channels)
            self.feat_proj = None

        # Self-rescaling MLP (2-layer, matches original default)
        self.mlp_scalar = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_channels // 2, 1, bias=False)
        )
        self.sigmoid = torch.nn.Sigmoid()

        # Fairness loss setup (reuses same pattern as PureGNN)
        if fairness_method != 'none':
            import sys, os
            fairness_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../fairness_methods'))
            if fairness_path not in sys.path:
                sys.path.append(fairness_path)
            from inprocessing import PAFairnessLoss, ExposureFairnessLoss, HybridFairnessLoss
            if fairness_method == 'inproc_pa':
                self.fairness_loss = PAFairnessLoss(
                    base_loss_name='AUC', lambda_fair=lambda_fair, pa_aggregation=pa_aggregation)
            elif fairness_method == 'inproc_exposure':
                self.fairness_loss = ExposureFairnessLoss(
                    base_loss_name='AUC', lambda_fair=lambda_fair,
                    num_nodes=num_nodes, exposure_decay=exposure_decay)
            elif fairness_method == 'inproc_hybrid':
                self.fairness_loss = HybridFairnessLoss(
                    base_loss_name='AUC', lambda_fair=lambda_fair,
                    pa_aggregation=pa_aggregation, num_nodes=num_nodes, exposure_decay=exposure_decay)
        else:
            self.fairness_loss = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.emb is not None:
            self.emb.reset_parameters()
        if self.feat_proj is not None:
            self.feat_proj.reset_parameters()
        for layer in self.mlp_scalar:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def create_input_feat(self, data):
        """Create input features, optionally combining node features with embeddings."""
        if self.use_node_feats and hasattr(data, 'x') and data.x is not None:
            input_feat = data.x
            if self.train_node_emb and self.emb is not None:
                input_feat = torch.cat([self.emb.weight, input_feat], dim=-1)
            input_feat = self.feat_proj(input_feat)
        else:
            input_feat = self.emb.weight
        return input_feat

    def _compute_lightgcn_adj(self, edge_index, num_nodes):
        """Compute symmetric normalized adjacency: weight = 1/sqrt(d_u * d_v).

        Matches original SSNet: gnn_module.add_edge_weights_to_graph()
        """
        row, col = edge_index[0], edge_index[1]
        deg = torch.zeros(num_nodes, device=edge_index.device)
        deg.scatter_add_(0, row, torch.ones(row.size(0), device=edge_index.device))
        edge_weight = 1.0 / (deg[row] * deg[col]).sqrt()
        # Handle division by zero for isolated nodes
        edge_weight = edge_weight.clamp(max=10.0)
        return edge_weight

    def forward(self, x, edge_index):
        """LightGCN forward: simple adjacency multiplication, no activation/dropout."""
        from torch_geometric.utils import to_torch_csr_tensor

        num_nodes = x.size(0)
        edge_weight = self._compute_lightgcn_adj(edge_index, num_nodes)

        # Build sparse adjacency for efficient matmul
        adj = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes)).coalesce()

        # LightGCN layers: just sparse matmul, no activation
        h = x
        for _ in range(self.num_layers):
            h = torch.sparse.mm(adj, h)

        # Self-rescaling
        w = self.mlp_scalar(h)
        h_rescaled = h * self.sigmoid(w)

        return h_rescaled

    def predict_batch(self, edge_batch, h):
        """Predict scores for edge batch via dot product"""
        h_u = h[edge_batch[:, 0]]
        h_v = h[edge_batch[:, 1]]
        return (h_u * h_v).sum(dim=-1)

# ==================== RANDOM BASELINE ====================

class RandomBaseline:
    """Random predictions baseline"""
    
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
    
    def fit(self, train_edges, num_nodes):
        """No training needed for random baseline"""
        self.num_nodes = num_nodes
    
    def predict_batch(self, edge_batch):
        """Return random scores"""
        return torch.rand(edge_batch.size(0))
    
    def predict_topk(self, h_emb, train_edges, test_nodes, k=100):
        """Generate random top-k recommendations"""
        from core.data_loader import build_train_adjacency
        
        train_adj = build_train_adjacency(train_edges, self.num_nodes)
        recs_by_u = {}
        
        print(f"[RANDOM] Generating random recommendations for {len(test_nodes)} nodes...")
        
        for u in test_nodes:
            # Get valid candidates (exclude self and training neighbors)
            invalid = {u} | train_adj[u]
            candidates = [v for v in range(self.num_nodes) if v not in invalid]
            
            # Random sample
            if len(candidates) >= k:
                recs_by_u[u] = np.random.choice(candidates, k, replace=False).tolist()
            else:
                recs_by_u[u] = candidates
        
        return recs_by_u

from typing import List, Optional
from typing import Union, Tuple, Callable
import torch

from torch import Tensor
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros, reset
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.nn.functional as F

class HGNConv(MessagePassing):

    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        edge_channel: int = 32,
        num_etype: int = 6,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        beta: float = 0.05,
        residual: bool = False,
        activation = None,
        # add_self_loops: bool = True,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        # self.add_self_loops = add_self_loops
        self.fill_value = fill_value

        self.edge_emb = nn.Parameter(Tensor(num_etype, edge_channel))
        self.beta = beta
        self.activation = activation
        self.residual = residual
        self.num_etype = num_etype

        # self.post_trans = nn.ModuleList()
        # for _ in range(num_etype):
        #     self.post_trans.append(Linear(out_channels, out_channels, bias=False, weight_initializer='glorot'))

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels,
                                  bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')
        self.lin_e = Linear(edge_channel, out_channels * heads, bias=False, weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_e = nn.Parameter(torch.Tensor(1, heads, out_channels))

        # if residual and self.in_channels != heads * out_channels:
        if residual:
            self.lin_res = Linear(self.in_channels[0], heads * out_channels, bias=False, weight_initializer='glorot')
        else:
            self.register_parameter('lin_res', None)

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()
        # self.message_args = inspect.getargspec(self.message)[0][1:]

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        self.lin_e.reset_parameters()
        if self.lin_res is not None:
            self.lin_res.reset_parameters()
        # for i in range(self.num_etype):
        #     self.post_trans[i].reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_e)
        glorot(self.edge_emb)
        zeros(self.bias)

    def pre_propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        """Compute and store attention coefficients for one edge type.

        This avoids calling internal PyG helpers (_check_input, _collect,
        inspector.distribute) whose signatures change across versions.
        We only need the attention computation from ``message`` when
        ``obtain_attn=True``.
        """
        alpha_src, alpha_dst = kwargs['alpha']
        edge_type = kwargs['edge_type']
        res_attn = kwargs.get('res_attn', None)

        # Lift node-level alpha to edge-level (same as PyG's message lifting)
        alpha_j = alpha_src[edge_index[0]]
        alpha_i = None if alpha_dst is None else alpha_dst[edge_index[1]]

        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        e_emb = self.edge_emb[edge_type]
        x_e = self.lin_e(e_emb).view(-1, self.heads, self.out_channels)
        alpha_e = (x_e * self.att_e).sum(-1)
        alpha = alpha + alpha_e
        if res_attn is not None:
            alpha = alpha * (1 - self.beta) + res_attn * self.beta

        alpha = F.leaky_relu(alpha, self.negative_slope)
        if self._alpha is None:
            self._alpha = [alpha]
        else:
            self._alpha.append(alpha)
    

    def forward(self, x: Union[Tensor, OptPairTensor], adjs, res_attn: OptTensor = None, size: Size = None):
        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            res_x = x
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else: 
            x_src, x_dst = x
            res_x = x_dst
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)
        x = (x_src, x_dst)

        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)
        num_nodes = x_src.size(0)

        total_edge_index = []
        for type_id, (edge_index, edge_weight) in enumerate(adjs):
            if edge_index == []:
                continue
            # if self.add_self_loops:
            #     if isinstance(edge_index, Tensor):
            #         if x_dst is not None:
            #             num_nodes = min(num_nodes, x_dst.size(0))
            #         num_nodes = min(size) if size is not None else num_nodes
            #         edge_index, edge_attr = remove_self_loops(
            #             edge_index, edge_attr)
            #         edge_index, edge_attr = add_self_loops(
            #             edge_index, edge_attr, fill_value=self.fill_value,
            #             num_nodes=num_nodes)
            self.pre_propagate(edge_index, x=x, alpha=alpha, res_attn=res_attn, edge_type=type_id, size=size, obtain_attn=True, bound=None, edge_weight=None)
            total_edge_index.append(edge_index[1])

        assert self._alpha is not None
        bound = [0]
        idx = 0
        # for ts in total_edge_index:
        for i, (edge_index, _) in enumerate(adjs):
            if edge_index == []:
                bound.append(bound[-1])
            else:
                bound.append(bound[-1] + total_edge_index[idx].size(0))
                idx += 1
        index = torch.cat(total_edge_index, dim=-1)
        alpha_backup = alpha
        alpha = torch.cat(self._alpha, dim=0)
        alpha = softmax(alpha, index, None, num_nodes)
        self._alpha = alpha
        out = None
        for type_id, (edge_index, edge_weight) in enumerate(adjs):
            if edge_index == []:
                continue
            
            cur_out = self.propagate(edge_index, x=x, alpha=alpha_backup, res_attn=res_attn, edge_type=type_id, size=size, obtain_attn=False, bound=bound, edge_weight=None)
            # if out is None:
            #     out = self.post_trans[type_id](cur_out)
            # else:
            #     out = out + self.post_trans[type_id](cur_out)
            if out is None:
                out = cur_out
            else:
                out = out + cur_out
        self._alpha = None
        
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        # if self.residual:
        if self.lin_res is not None:
            res_val = self.lin_res(res_x)
        # else:
        #     res_val = res_x
            out = res_val + out

        if self.bias is not None:
            out += self.bias
        if self.activation:
            out = self.activation(out)

        # if isinstance(return_attention_weights, bool):
        #     if isinstance(edge_index, Tensor):
        #         return out, (edge_index, alpha)
        #     elif isinstance(edge_index, SparseTensor):
        #         return out, edge_index.set_value(alpha, layout='coo')
        # else:
        return out, alpha


    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor, res_attn: OptTensor, edge_type: int, obtain_attn: bool, bound: Optional[List[int]], edge_weight: OptTensor,
                index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:

        if obtain_attn:        
            alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

            e_emb = self.edge_emb[edge_type]
            x_e = self.lin_e(e_emb).view(-1, self.heads, self.out_channels)
            alpha_e = (x_e * self.att_e).sum(-1)
            alpha = alpha + alpha_e
            if res_attn is not None:
                alpha = alpha * (1 - self.beta) + res_attn * self.beta

            # alpha = alpha * torch.log(edge_weight+1).unsqueeze(-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            if self._alpha is None:
                self._alpha = [alpha]
            else:
                self._alpha.append(alpha)
            return x_j
        else:
            lower, upper = bound[edge_type], bound[edge_type + 1]
            alpha = self._alpha[lower:upper]
            # alpha = alpha * torch.log(edge_weight+1).unsqueeze(-1)
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            # return x_j * self.edge_emb[edge_type].unsqueeze(0) * alpha.unsqueeze(-1)
            return x_j * alpha.unsqueeze(-1)


class LGCConv(MessagePassing):

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 normalize: bool = False,
                 bias: bool = True, **kwargs):  
        super(LGCConv, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.normalize = normalize
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.improved = False
        self.add_self_loops = True
        self.reset_parameters()
    
    def reset_parameters(self):
        pass

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        
        edge_index, edge_weight = gcn_norm(edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.conv import SAGEConv, GCNConv



class TailSTEAK(nn.Module):
    def __init__(self, in_channels, hidden_channels, id_emb, num_etype, edge_channels, num_layers,
                 heads=4, ssl_temp=0.5, deg_t_low=15, U=500, base='HGN'):
        super(TailSTEAK, self).__init__()
        self.base = base
        self.num_layers = num_layers
        self.in_channel = in_channels
        self.hidden_channel = hidden_channels
        self.num_etype = num_etype
        self.edge_channels = edge_channels
        self.heads = heads
        self.ssl_temp = ssl_temp
        self.U = U
        self.deg_t_low = int(deg_t_low)
        self.id_emb = nn.Parameter(id_emb)
        self.convs = nn.ModuleList()
        if base == 'HGN':
            self.pred_head = nn.Sequential(nn.Linear(hidden_channels*2*heads, hidden_channels), nn.LeakyReLU(negative_slope=0.2), nn.Dropout(p=0.3), nn.Linear(hidden_channels, 1))
            for i in range(num_layers):
                in_channels = in_channels if i == 0 else hidden_channels * heads
                self.convs.append(HGNConv((in_channels, in_channels), hidden_channels, heads=heads, edge_channel=edge_channels, num_etype=num_etype, residual=True))        
        elif base == 'LightGCN':
            self.pred_head = nn.Sequential(nn.Linear(self.in_channel * 2, self.in_channel), nn.LeakyReLU(negative_slope=0.2), nn.Dropout(p=0.3), nn.Linear(self.in_channel, 1))
            for i in range(num_layers):
                self.convs.append(LGCConv(self.in_channel))
        elif base == 'SAGE':
            self.pred_head = nn.Sequential(nn.Linear(hidden_channels*2, hidden_channels), nn.LeakyReLU(0.2),nn.Dropout(0.3), nn.Linear(hidden_channels, 1))
            for i in range(num_layers):
                in_channels = in_channels if i == 0 else hidden_channels
                self.convs.append(SAGEConv(in_channels, hidden_channels))
        elif base == 'GCN':
            self.pred_head = nn.Sequential(nn.Linear(hidden_channels*2, hidden_channels), nn.LeakyReLU(0.2), nn.Dropout(0.3), nn.Linear(hidden_channels,1))
            for i in range(num_layers):
                in_channel = self.in_channel if i == 0 else hidden_channels
                self.convs.append(GCNConv(in_channel, hidden_channels))
        self.reset_parameters()
    
    def reset_parameters(self, emb_init=None):
        if emb_init is not None:
            self.id_emb.data = emb_init
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
        
        self.pred_head.apply(weight_reset)
        for i in range(self.num_layers):
            self.convs[i].reset_parameters()
    
    def _run_encoder(self, x, adjs, id_emb=None, use_checkpoint=False):
        """Run GNN encoder layers. Returns (x, full_edge_index, out_dim).

        If use_checkpoint=True, uses gradient checkpointing to save memory
        (mathematically identical, just recomputes activations during backward).
        """
        if self.base == 'HGN':
            full_edge_index, _ = adjs[0]
            out_dim = self.heads * self.hidden_channel
            for i in range(self.num_layers):
                if use_checkpoint and x.requires_grad:
                    # Capture loop variable
                    conv = self.convs[i]
                    is_last = (i == self.num_layers - 1)
                    def _hgn_layer(x_in, _conv=conv, _adjs=adjs, _is_last=is_last):
                        out, _ = _conv(x_in, _adjs, res_attn=None)
                        if not _is_last:
                            out = out.relu()
                            out = F.dropout(out, p=0.3, training=self.training)
                        return out
                    x = ckpt.checkpoint(_hgn_layer, x, use_reentrant=False)
                else:
                    x, _ = self.convs[i](x, adjs, res_attn=None)
                    if i != self.num_layers - 1:
                        x = x.relu()
                        x = F.dropout(x, p=0.3, training=self.training)
        elif self.base == 'LightGCN':
            out_dim = self.in_channel
            full_edge_index, _ = adjs[0]
            final_x = id_emb if id_emb is not None else x
            for i in range(self.num_layers):
                if use_checkpoint and x.requires_grad:
                    conv = self.convs[i]
                    def _lgcn_layer(x_in, _conv=conv, _ei=full_edge_index):
                        return _conv(x_in, _ei)
                    x = ckpt.checkpoint(_lgcn_layer, x, use_reentrant=False)
                else:
                    x = self.convs[i](x, full_edge_index)
                final_x = final_x + x
            x = final_x / (self.num_layers + 1)
        elif self.base == 'SAGE' or self.base == 'GCN':
            out_dim = self.hidden_channel
            full_edge_index, _ = adjs[0]
            for i in range(self.num_layers):
                if use_checkpoint and x.requires_grad:
                    conv = self.convs[i]
                    is_last = (i == self.num_layers - 1)
                    def _sage_layer(x_in, _conv=conv, _ei=full_edge_index, _is_last=is_last):
                        out = _conv(x_in, _ei)
                        if not _is_last:
                            out = out.relu()
                            out = F.dropout(out, p=0.3, training=self.training)
                        return out
                    x = ckpt.checkpoint(_sage_layer, x, use_reentrant=False)
                else:
                    x = self.convs[i](x, full_edge_index)
                    if i != self.num_layers - 1:
                        x = x.relu()
                        x = F.dropout(x, p=0.3, training=self.training)
        return x, full_edge_index, out_dim

    def forward(self, x, adjs, n_id, neg_sample_num=19, ssl_x=None, ssl_adjs=None, ssl_n_id=None, diff_adjs=None,
                    out_deg=None, node_idx=None, ssl_node_idx=None, mode='train', device=None):
        id_emb =  self.id_emb

        if ssl_adjs is not None:
            ssl_adj = []
            for adj in ssl_adjs:
                ssl_adj.append([adj[0], adj[1]])
            ssl_adjs = ssl_adj
            ssl_x = self.id_emb.clone()

        x = id_emb

        x_backup = x
        use_ckpt = (mode == 'ft')  # Checkpoint in ft mode to save memory (3 encoder passes)
        x, full_edge_index, out_dim = self._run_encoder(x, adjs, id_emb=id_emb, use_checkpoint=use_ckpt)
        
        if mode == 'train':
            flag = False
            if isinstance(out_deg, tuple):
                flag = True
                out_deg, train_deg, in_deg = out_deg
                splits = train_deg.split(train_deg.size(0) // (2 + neg_sample_num), dim=0)
                train_deg = splits[0]
                splits = train_deg.split(in_deg.size(0) // (2 + neg_sample_num), dim=0)
                in_deg = splits[0]
            splits = out_deg.split(out_deg.size(0) // (2 + neg_sample_num), dim=0)
            out_deg = splits[0]
            splits = n_id.split(n_id.size(0) // (2 + neg_sample_num), dim=0)
            out_idx, pos_idx, neg_idx = splits[0], splits[1], torch.cat(list(splits[2:2+neg_sample_num]), dim=0)

            out, pos_out, neg_out = x[out_idx], x[pos_idx], x[neg_idx]

            # # Uncomment this block when train only on head nodes
            out = out[out_deg > self.deg_t_low]
            pos_out = pos_out[out_deg > self.deg_t_low]
            neg_out = neg_out.reshape(-1, out_dim * neg_sample_num)[out_deg > self.deg_t_low].reshape(-1, out_dim)
            tiled_out = out.repeat(1, neg_sample_num).reshape(-1, out_dim)

            pos_logit = torch.cat([out, pos_out], dim=-1)
            neg_logits = torch.cat([tiled_out, neg_out], dim=-1)
            pos_logit = self.pred_head(pos_logit)
            neg_logits = self.pred_head(neg_logits)

            # # Needed part
            out_tail, pos_tail, neg_tail = x[out_idx], x[pos_idx], x[neg_idx]
            mask_tail = (out_deg <= self.deg_t_low)
            out_tail = out_tail[mask_tail]
            pos_tail = pos_tail[mask_tail]

            pos_logit_tail = torch.cat([out_tail, pos_tail], dim=-1)

            ssl_eindex, _ = ssl_adjs[0]
            real_edge_index = ssl_node_idx[ssl_eindex]
            src_nid = torch.unique(out_idx, sorted=False)
            emask = ~(torch.isin(full_edge_index[0], src_nid) | torch.isin(full_edge_index[1], src_nid))
            ssl_eindex = full_edge_index[:, emask]
            if len(real_edge_index) != 0:
                inverted_eindex = torch.cat([real_edge_index[1], real_edge_index[0]]).view(2, -1)
                ssl_eindex = torch.cat([ssl_eindex, real_edge_index, inverted_eindex], dim=-1)
            ssl_adjs = [[ssl_eindex, None]]

            if self.base == 'HGN':
                for i in range(self.num_layers):
                    ssl_x, _ = self.convs[i](ssl_x, ssl_adjs, res_attn=None)
                    if i != self.num_layers - 1:
                        ssl_x = ssl_x.relu()
                        ssl_x = F.dropout(ssl_x, p=0.3, training=self.training)
            elif self.base == 'SAGE' or self.base == 'GCN':
                ssl_eindex, _ = ssl_adjs[0]
                for i in range(self.num_layers):
                    ssl_x = self.convs[i](ssl_x, ssl_eindex)
                    if i != self.num_layers - 1:
                        ssl_x = ssl_x.relu()
                        ssl_x = F.dropout(ssl_x, p=0.3, training=self.training)
            elif self.base == 'LightGCN':
                final_x = ssl_x
                for i in range(self.num_layers):
                    ssl_x = self.convs[i](ssl_x, ssl_eindex)
                    final_x = ssl_x + final_x
                ssl_x = final_x / (self.num_layers + 1)
            ssl_out = ssl_x[ssl_node_idx[ssl_n_id]]
            head_idx = out_idx[out_deg > self.deg_t_low]
            pos_head_idx = pos_idx[out_deg > self.deg_t_low]
            out_head = x[head_idx]

            pos_head = x[pos_head_idx]            
            tiled_ssl_out = ssl_out.repeat(1, neg_sample_num).reshape(-1, out_dim)
            ssl_pos_logit = torch.cat([ssl_out, pos_head], dim=-1)
            ssl_neg_logits = torch.cat([tiled_ssl_out, neg_out], dim=-1)
            ssl_pos_logit = self.pred_head(ssl_pos_logit)
            ssl_neg_logits = self.pred_head(ssl_neg_logits)
            pos_logit = torch.cat([pos_logit, ssl_pos_logit], dim=0)
            neg_logits = torch.cat([neg_logits, ssl_neg_logits], dim=0)

            ssl_out_norm, out_head = F.normalize(ssl_out, p=2, dim=-1), F.normalize(out_head, p=2, dim=-1)
            out_tail = F.normalize(out_tail, p=2, dim=-1)
            ssl_loss_1 = self.infonce(out_head, ssl_out_norm)
            ssl_loss_1 = ssl_loss_1.mean()
            ssl_loss_1 = (ssl_loss_1 + self.infonce(ssl_out_norm, out_head).mean()) / 2

            return pos_logit, neg_logits, ssl_loss_1
        else:
            splits = n_id.split(n_id.size(0) // (2 + neg_sample_num), dim=0)
            out_idx, pos_idx, neg_idx = splits[0], splits[1], torch.cat(list(splits[2:2+neg_sample_num]), dim=0)
            out, pos_out, neg_out = x[out_idx], x[pos_idx], x[neg_idx]

            if mode != 'ft' and mode != 'pseudo':
                splits = out_deg.split(out_deg.size(0) // (2 + neg_sample_num), dim=0)
                out_deg = splits[0]

                tiled_out = out.repeat(1, neg_sample_num).reshape(-1, out_dim)
                pos_logit = torch.cat([out, pos_out], dim=-1)
                neg_logits = torch.cat([tiled_out, neg_out], dim=-1)
                pos_logit = self.pred_head(pos_logit)
                neg_logits = self.pred_head(neg_logits)
                out_head, pos_out_head = F.normalize(out[out_deg > 3], p=2, dim=-1), F.normalize(pos_out[out_deg > 3], p=2, dim=-1)
                out_tail, pos_out_tail = F.normalize(out[out_deg <= 3], p=2, dim=-1), F.normalize(pos_out[out_deg <= 3], p=2, dim=-1)
                out, pos_out = F.normalize(out, p=2, dim=-1), F.normalize(pos_out, p=2, dim=-1)
                return pos_logit, neg_logits

            elif mode == 'pseudo':
                out_idx = n_id
                out = x[out_idx]
                neg_sample_num = self.U
                tiled_out = out.repeat(1, neg_sample_num).reshape(-1, neg_sample_num, out_dim)

                # Vectorized negative sampling (same result as per-node np.random.choice)
                n_nodes = x.size(0)
                batch_sz = out_idx.size(0)
                out_idx_cpu = out_idx.cpu()
                # Sample neg_sample_num candidates per node
                sampled_idx = torch.randint(0, n_nodes, (batch_sz, neg_sample_num), dtype=torch.int64)
                # Fix self-collisions: where sampled_idx[i,j] == out_idx[i], resample
                self_mask = (sampled_idx == out_idx_cpu.unsqueeze(1))
                while self_mask.any():
                    sampled_idx[self_mask] = torch.randint(0, n_nodes, (self_mask.sum(),), dtype=torch.int64)
                    self_mask = (sampled_idx == out_idx_cpu.unsqueeze(1))
                sampled_idx = sampled_idx.to(x.device)
                
                full_out = x[sampled_idx].reshape(-1, neg_sample_num, out_dim)
                logits = torch.cat([tiled_out, full_out], dim=-1)
                full_logits = self.pred_head(logits).squeeze()
                return sampled_idx.reshape(-1, neg_sample_num), full_logits

            elif mode == 'ft':
                splits = out_deg.split(out_deg.size(0) // (2 + neg_sample_num), dim=0)
                out_deg = splits[0]

                out_head = out[out_deg > self.deg_t_low]
                out_tail = out[out_deg <= self.deg_t_low]
                pos_out_head = pos_out[out_deg > self.deg_t_low]
                pos_out_tail = pos_out[out_deg <= self.deg_t_low]
                neg_out_head = neg_out.reshape(-1, out_dim * neg_sample_num)[out_deg > self.deg_t_low].reshape(-1, out_dim)
                neg_out_tail = neg_out.reshape(-1, out_dim * neg_sample_num)[out_deg <= self.deg_t_low].reshape(-1, out_dim)
                tiled_out = out.repeat(1, neg_sample_num).reshape(-1, out_dim)

                pos_logit = torch.cat([out, pos_out], dim=-1)
                neg_logits = torch.cat([tiled_out, neg_out], dim=-1)

                ssl_eindex, _ = ssl_adjs[0]
                real_edge_index = ssl_node_idx[ssl_eindex]
                src_nid = torch.unique(out_idx, sorted=False)
                emask = ~(torch.isin(full_edge_index[0], src_nid) | torch.isin(full_edge_index[1], src_nid))
                ssl_eindex = full_edge_index[:, emask]
                if len(real_edge_index) != 0:
                    inverted_eindex = torch.cat([real_edge_index[1], real_edge_index[0]]).view(2, -1)
                    ssl_eindex = torch.cat([ssl_eindex, real_edge_index, inverted_eindex], dim=-1)
                ssl_adjs = [[ssl_eindex, None]]

                ssl_x, _, _ = self._run_encoder(ssl_x, ssl_adjs, id_emb=ssl_x, use_checkpoint=True)
                ssl_out = ssl_x[ssl_node_idx[ssl_n_id]]

                tiled_ssl_out = ssl_out.repeat(1, neg_sample_num).reshape(-1, out_dim)
                ssl_pos_logit = torch.cat([ssl_out, pos_out_head], dim=-1)
                ssl_neg_logits = torch.cat([tiled_ssl_out, neg_out_head], dim=-1)
                pos_logit = torch.cat([pos_logit, ssl_pos_logit], dim=0)
                neg_logits = torch.cat([neg_logits, ssl_neg_logits], dim=0)
                pos_logit = self.pred_head(pos_logit)
                neg_logits = self.pred_head(neg_logits)

                ssl_out, out_head = F.normalize(ssl_out, p=2, dim=-1), F.normalize(out_head, p=2, dim=-1)
                out_tail = F.normalize(out_tail, p=2, dim=-1)

                ssl_loss_1 = self.infonce(out_head, ssl_out)# , out_tail)
                ssl_loss_1 = (ssl_loss_1 + self.infonce(ssl_out, out_head).mean()) / 2

                if diff_adjs is not None:
                    diff_x = self.id_emb
                    tail_idx = out_idx[out_deg <= self.deg_t_low]
                    diff_x, _, _ = self._run_encoder(diff_x, diff_adjs, id_emb=id_emb, use_checkpoint=True)
                    
                    diff_out = out[out_deg <= self.deg_t_low]
                    diff_out_dp = diff_x[tail_idx]
                    diff_out_tiled = diff_out.repeat(1, neg_sample_num).reshape(-1, out_dim)
                    pos_logit_tail = torch.cat([diff_out, pos_out_tail], dim=-1)
                    neg_logits_tail = torch.cat([diff_out_tiled, neg_out_tail], dim=-1)
                    pos_logit_tail = self.pred_head(pos_logit_tail)
                    neg_logits_tail = self.pred_head(neg_logits_tail)
                    pos_logit = torch.cat([pos_logit, pos_logit_tail], dim=0)
                    neg_logits = torch.cat([neg_logits, neg_logits_tail], dim=0)

                    diff_out, diff_out_dp = F.normalize(diff_out, p=2, dim=-1), F.normalize(diff_out_dp, p=2, dim=-1)
                    
                    diff_loss = self.infonce(diff_out, diff_out_dp)
                    diff_loss = (diff_loss + self.infonce(diff_out_dp, diff_out)) / 2
                    ssl_loss_1 = torch.cat([ssl_loss_1, diff_loss])
                ssl_loss_1 = ssl_loss_1.mean()

                return pos_logit, neg_logits, ssl_loss_1

    def infonce(self, out_1, out_2, out_tail=None):
        pos_score = (out_1 * out_2).sum(-1)
        total_score = torch.matmul(out_1, out_2.T)
        if out_tail is not None:
            total_score_2 = torch.matmul(out_1, out_tail.T)
            total_score = torch.cat([total_score, total_score_2], dim=-1)
        pos_score = torch.exp(pos_score / self.ssl_temp)
        total_score = (torch.exp(total_score / self.ssl_temp)).sum(-1)
        return -torch.log(pos_score / total_score)
    

    def embed(self, adjs, n_id, node_idx=None):
        total_adj = []
        for adj in adjs:
            total_adj.append([adj[0], adj[1]])
        adjs = total_adj
        
        if node_idx is None:
            x = self.id_emb
        else:
            x = self.id_emb[node_idx]
        if self.base == 'HGN':
            for i in range(self.num_layers):
                x, _ = self.convs[i](x, adjs, res_attn=None)
                if i != self.num_layers - 1:
                    x = x.relu()
                    x = F.dropout(x, p=0.3, training=self.training)
        elif self.base == 'LightGCN':
            full_edge_index, _ = adjs[0]
            final_x = x
            for i in range(self.num_layers):
                x = self.convs[i](x, full_edge_index)
                final_x = final_x + x
            x = final_x / (self.num_layers + 1)
        elif self.base in ['SAGE', 'GCN']:
            full_edge_index, _ = adjs[0]
            for i in range(self.num_layers):
                x = self.convs[i](x, full_edge_index)
                if i != self.num_layers - 1:
                    x = x.relu()
                    x = F.dropout(x, p=0.3, training=self.training)

        if isinstance(n_id, torch.Tensor):
            n_id = n_id.clone().detach().to(dtype=torch.int64).reshape(-1)
        else:
            n_id = torch.tensor(n_id, dtype=torch.int64).reshape(-1)
        return x[n_id]
import time
import torch
from torch.cuda import amp
import torch.nn.functional as F

class TailSTEAKBaseline(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels=32, num_etype=1, edge_channels=32, num_layers=1,
                 heads=4, ssl_temp=0.5, ssl_reg=1.0, deg_t_low=15, U=500, base='HGN', lr=0.01, weight_decay=0.0, ft_epoch=2):
        super(TailSTEAKBaseline, self).__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.ssl_reg = ssl_reg
        self.deg_t_low = deg_t_low
        self.lr = lr
        self.weight_decay = weight_decay
        self.ft_epoch = ft_epoch
        self._embeddings = None
        
        self.id_emb = torch.randn(num_nodes, in_channels)
        nn.init.xavier_uniform_(self.id_emb)
        
        self.model = TailSTEAK(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            id_emb=self.id_emb,
            num_etype=num_etype,
            edge_channels=edge_channels,
            num_layers=num_layers,
            heads=heads,
            ssl_temp=ssl_temp,
            deg_t_low=deg_t_low,
            U=U,
            base=base
        )

    def fit(self, train_loader, train_edges, epochs=30, device='cuda:0', val_pos_edges=None, val_neg_edges=None):
        import copy
        if getattr(self, '_id_emb_initialized', False) is False:
            print(f"[TAILSTEAK] Initializing ID embeddings with Node2Vec (dim={self.in_channels})...")
            n2v_model = RandomWalkEmbedding(
                method='node2vec',
                embedding_dim=self.in_channels,
                walk_length=15,
                num_walks=30,
                p=1.0,
                q=10.0
            )
            n2v_model.fit(train_edges, self.num_nodes, device=device)
            self.id_emb = n2v_model.embeddings.cpu().detach().clone()
            self._id_emb_initialized = True

        id_emb = self.id_emb.to(device)
        self.model.id_emb.data = id_emb
        self.model = self.model.to(device)
        self.model.train()
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        enable_amp = True if "cuda" in str(device) else False
        scaler = amp.GradScaler(enabled=enable_amp)
        
        neg_sample_num = 19
        full_edge_index = []
        for elist, wlist in zip(train_loader.edge_index, train_loader.edge_weight):
            full_edge_index.append([elist.to(device), wlist.to(device)])

        best_ndcg = -1.0
        best_state_dict = None

        from tqdm import tqdm
        for epoch in tqdm(range(1, epochs + 1), desc="Phase 1 Training"):
            step = 0
            for out in train_loader:
                batch_len, x, adjs, node_idx, n_id, ssl_x, ssl_adjs, ssl_node_idx, ssl_n_id, diff_x, diff_adjs, diff_node_idx, diff_n_id, out_deg, in_deg = out
                
                ssl_x = ssl_x.to(device)
                ssl_adjs = [adj.to(device) for adj in ssl_adjs]
                if diff_x is not None:
                    diff_x = diff_x.to(device)
                    diff_adjs = [adj.to(device) for adj in diff_adjs]
                    diff_node_idx = diff_node_idx.to(device)
                ssl_node_idx = ssl_node_idx.to(device)
                
                edge_index = full_edge_index
                in_deg, out_deg = in_deg.to(device), out_deg.to(device)
                n_id = n_id.to(device)

                optimizer.zero_grad()
                with amp.autocast(enabled=enable_amp):
                    pos_logit, neg_logits, ssl_loss_1 = self.model(
                        x, edge_index, n_id, ssl_x=ssl_x, ssl_adjs=ssl_adjs, ssl_n_id=ssl_n_id, 
                        neg_sample_num=neg_sample_num, out_deg=out_deg, node_idx=node_idx, 
                        ssl_node_idx=ssl_node_idx, device=device
                    )
                    
                    pos_loss = F.logsigmoid(pos_logit).mean()
                    neg_loss = F.logsigmoid(-neg_logits).reshape(neg_sample_num, -1).mean(dim=0).mean()
                    loss = -pos_loss - neg_loss + ssl_loss_1 * self.ssl_reg

                if np.isnan(loss.detach().cpu()):
                    print("NaN Loss Detected")
                    return
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                step += 1
            print(f"TailSTEAK Epoch {epoch}/{epochs} complete. Last loss {loss.item():.4f}")
            if val_pos_edges is not None and val_neg_edges is not None:
                self.model.eval()
                with torch.no_grad():
                    h = self.model.embed(full_edge_index, torch.arange(self.num_nodes, device=device), None)
                    pos_scores = (h[val_pos_edges[:, 0]] * h[val_pos_edges[:, 1]]).sum(dim=-1)
                    neg_scores = (h[val_neg_edges[:, 0]] * h[val_neg_edges[:, 1]]).sum(dim=-1)
                    from sklearn.metrics import roc_auc_score
                    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]).cpu().numpy()
                    preds = torch.cat([pos_scores, neg_scores]).cpu().numpy()
                    auc = roc_auc_score(labels, preds)
                    print(f"  --> Validation AUC: {auc:.4f}")
                    if auc > best_ndcg:
                        best_ndcg = auc
                        best_state_dict = copy.deepcopy(self.model.state_dict())
                self.model.train()
        
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            print(f"Loaded best Phase 1 model (AUC: {best_ndcg:.4f}) before Decoder Finetuning.")

        # Free GPU memory before finetuning phase
        torch.cuda.empty_cache()

        # Decoder fine-tune phase
        # Original TailSTEAK recreates the loader each epoch with pseudo edges (batch_size=256)
        if self.ft_epoch > 0:
            import copy
            from baselines.tailsteak_dataloader import SocLoader
            for epoch in tqdm(range(1, self.ft_epoch), desc="Decoder Finetuning"):
                pseudo_edge_index = self._pseudo_edge_predict(train_loader, device)

                ft_loader = SocLoader(
                    self._node_features, mode='train', debias_on=True,
                    graph_path=copy.deepcopy(self._graph_info),
                    deg_t_low=self.deg_t_low, gamma=self._gamma,
                    pseudo_edge_index=pseudo_edge_index,
                    batch_size=256, shuffle=True
                )

                best_ndcg, best_state_dict = self._decoder_finetune(
                    ft_loader, optimizer, scaler, enable_amp, full_edge_index, device,
                    val_pos_edges, val_neg_edges, best_ndcg, best_state_dict
                )
                print(f"TailSTEAK Decoder Finetuning Epoch {epoch}/{self.ft_epoch - 1} complete.")

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            print("Loaded absolutely best model from all phases.")

        # Cache embeddings
        self.model.eval()
        with torch.no_grad():
            self._embeddings = self.model.embed(full_edge_index, torch.arange(self.num_nodes, device=device), None)

    @torch.no_grad()
    def _pseudo_edge_predict(self, train_loader, device):
        self.model.eval()
        edge_index = []
        n_id = torch.arange(train_loader.homo_out_deg.size(0), dtype=torch.int64)
        n_id = n_id[train_loader.homo_out_deg <= self.deg_t_low].to(device)
        for elist, wlist in zip(train_loader.edge_index, train_loader.edge_weight):
                edge_index.append([elist.to(device), wlist.to(device)])
        x = self.model.id_emb.to(device)
        
        out_deg = train_loader.homo_out_deg.to(device)[n_id]
        pseudo_edge_index = []
        num_range = out_deg.size(0) // 2048 + 1
        for i in tqdm(range(num_range), desc="  Pseudo edges", leave=False):
            nid_batch = n_id[2048*i:min(out_deg.size(0), 2048*(i+1))]
            out_deg_batch = out_deg[2048*i:min(out_deg.size(0), 2048*(i+1))]
            sampled_idx, logits = self.model(x, edge_index, nid_batch, self.model.U, out_deg=out_deg_batch, mode='pseudo')
            logits = logits.reshape(nid_batch.size(0), self.model.U)   
            
            logits = logits.cpu()
            random_noise = torch.rand(logits.numpy().shape) * 1e-7
            logits = logits + random_noise

            sorted_idx = torch.argsort(logits, dim=-1, descending=True).numpy()
            sorted_idx = sorted_idx[:, :int(self.deg_t_low)]   
            
            nid_batch_cpu = nid_batch.cpu()
            sampled_idx_cpu = sampled_idx.cpu()
            sorted_idx_t = torch.from_numpy(sorted_idx)  # (batch, deg_t_low)
            # Gather true target node IDs: sampled_idx[m, sorted_idx[m, k]] for all m, k
            true_targets = sampled_idx_cpu.gather(1, sorted_idx_t)  # (batch, deg_t_low)
            # Expand source node IDs to match
            sources = nid_batch_cpu.unsqueeze(1).expand_as(true_targets)  # (batch, deg_t_low)
            # Mask out self-loops (where source == target)
            not_self = (sources != true_targets)
            edges = torch.stack([sources[not_self], true_targets[not_self]], dim=1)
            pseudo_edge_index.append(edges)
        pseudo_edge_index = torch.cat(pseudo_edge_index, dim=0).to(torch.int64) if pseudo_edge_index else torch.zeros((0, 2), dtype=torch.int64)
        return pseudo_edge_index

    def _decoder_finetune(self, train_loader, optimizer, scaler, enable_amp, edge_index, device, val_pos_edges, val_neg_edges, best_ndcg, best_state_dict):
        import copy
        self.model.train()
        neg_sample_num = 19
        for out in tqdm(train_loader, desc="  FT batches", leave=False):
            batch_len, x, adjs, node_idx, n_id, ssl_x, ssl_adjs, ssl_node_idx, ssl_n_id, diff_x, diff_adjs, diff_node_idx, diff_n_id, out_deg, in_deg = out
            ssl_x = ssl_x.to(device)
            ssl_adjs = [adj.to(device) for adj in ssl_adjs]
            if diff_x is not None:
                diff_x = diff_x.to(device)
                diff_adjs = [adj.to(device) for adj in diff_adjs]
                diff_node_idx = diff_node_idx.to(device)
            ssl_node_idx = ssl_node_idx.to(device)

            x = x.to(device)
            diff_eindex = []

            # Recreate edge_index for finetuning
            extra_eindex = train_loader.pseudo_edge_index.to(device)
            current_edge_index = []
            for i in range(len(edge_index)):
                current_edge_index.append([edge_index[i][0].to(device), edge_index[i][1].to(device)])
                eindex = current_edge_index[i][0]
                diff_eindex.append([torch.cat([eindex, extra_eindex], dim=-1), current_edge_index[i][1]])

            in_deg, out_deg = in_deg.to(device), out_deg.to(device)
            n_id = n_id.to(device)

            optimizer.zero_grad()
            with amp.autocast(enabled=enable_amp):
                pos_logit, neg_logits, ssl_loss_1 = self.model(
                    x, current_edge_index, n_id, ssl_x=ssl_x, ssl_adjs=ssl_adjs, ssl_n_id=ssl_n_id, diff_adjs=diff_eindex,
                    neg_sample_num=neg_sample_num, out_deg=out_deg, node_idx=node_idx, ssl_node_idx=ssl_node_idx, mode='ft', device=device
                )

                pos_loss = F.logsigmoid(pos_logit).mean()
                neg_loss = F.logsigmoid(-neg_logits).reshape(neg_sample_num, -1).mean(dim=0).mean()
                loss = -pos_loss - neg_loss + ssl_loss_1 * self.ssl_reg

            if np.isnan(loss.detach().cpu()):
                print("NaN Loss Detected")
                return best_ndcg, best_state_dict
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        if val_pos_edges is not None and val_neg_edges is not None:
            self.model.eval()
            with torch.no_grad():
                h = self.model.embed(edge_index, torch.arange(self.num_nodes, device=device), None)
                pos_scores = (h[val_pos_edges[:, 0]] * h[val_pos_edges[:, 1]]).sum(dim=-1)
                neg_scores = (h[val_neg_edges[:, 0]] * h[val_neg_edges[:, 1]]).sum(dim=-1)
                from sklearn.metrics import roc_auc_score
                labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]).cpu().numpy()
                preds = torch.cat([pos_scores, neg_scores]).cpu().numpy()
                auc = roc_auc_score(labels, preds)
                print(f"  --> Finetuning Validation AUC: {auc:.4f}")
                if auc > best_ndcg:
                    best_ndcg = auc
                    best_state_dict = copy.deepcopy(self.model.state_dict())
            self.model.train()

        return best_ndcg, best_state_dict

    @torch.no_grad()
    def predict_batch(self, edge_batch):
        if self._embeddings is None:
            raise ValueError("Model not fitted. Call fit() first.")
        u_emb = self._embeddings[edge_batch[:, 0]]
        v_emb = self._embeddings[edge_batch[:, 1]]
        
        # TailSTEAK uses pred_head on concatenated embeddings
        h_cat = torch.cat([u_emb, v_emb], dim=-1)
        return self.model.pred_head(h_cat).squeeze()

    def get_embeddings(self):
        return self._embeddings

    def predict_topk(self, h_emb, train_edges, test_nodes, k=100):
        from core.evaluation import build_recommendations
        return build_recommendations(
            h_emb, train_edges, test_nodes, k=k, device=h_emb.device, score_fn=None
        )
