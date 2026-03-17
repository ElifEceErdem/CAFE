"""
Fairness-Aware Baseline Models for Link Prediction

Implements fairness-aware embedding methods adapted from PyGDebias:
- FairWalk: Random walks with fair transition probabilities across sensitive groups
- CrossWalk: Random walks biased toward crossing group boundaries
- UGE: Unbiased Graph Embedding (UGE-W, UGE-R, UGE-C variants)

All methods produce node embeddings usable with dot-product scoring for link prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from gensim.models import Word2Vec
from torch_geometric.nn import GCNConv, SAGEConv
from dataclasses import dataclass
import networkx as nx
from tqdm import tqdm


# ==================== FAIRWALK ====================

class FairWalkEmbedding:
    """
    FairWalk: Random walks with fair transition probabilities.

    At each walk step, uniformly picks a sensitive group from groups represented
    among the current node's neighbors, then samples a neighbor within that group
    using Node2Vec-style p/q biased weights. This ensures each group gets equal
    visitation probability regardless of its neighborhood proportion.

    Reference: Rahman et al., "FairWalk: Towards Fair Graph Embedding" (IJCAI 2019)
    """

    def __init__(self, embedding_dim=128, walk_length=80, num_walks=10,
                 window_size=10, p=1.0, q=1.0):
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.p = p
        self.q = q
        self.embeddings = None

    def fit(self, train_edges, num_nodes, sens_attr, device='cpu'):
        """
        Train FairWalk embeddings.

        Args:
            train_edges: Training edges tensor [num_edges, 2]
            num_nodes: Total number of nodes
            sens_attr: Binary sensitive attribute tensor [num_nodes]
            device: Torch device
        """
        print(f"[FAIRWALK] Building graph and precomputing fair transition probabilities...")

        # Build adjacency dict
        adj = {i: set() for i in range(num_nodes)}
        for u, v in train_edges.cpu().tolist():
            if u < num_nodes and v < num_nodes:
                adj[u].add(v)
                adj[v].add(u)

        # Convert sens_attr to numpy for fast lookup
        if torch.is_tensor(sens_attr):
            groups = sens_attr.cpu().numpy().astype(int)
        else:
            groups = np.array(sens_attr, dtype=int)

        # Precompute fair transition probabilities
        # d_graph[node] = {
        #   'neighbors': {group_id: [neighbor_list]},
        #   'probabilities': {group_id: {prev_node: [prob_list]}},
        #   'first_travel': {group_id: [prob_list]}
        # }
        d_graph = defaultdict(dict)
        unique_groups = np.unique(groups)

        # Initialize structure
        for node in range(num_nodes):
            d_graph[node]['neighbors'] = {g: [] for g in unique_groups}
            d_graph[node]['probabilities'] = {g: {} for g in unique_groups}
            d_graph[node]['first_travel'] = {}

        print(f"[FAIRWALK] Precomputing probabilities for {num_nodes} nodes...")

        for source in range(num_nodes):
            neighbors_of_source = adj[source]

            for current_node in adj[source]:
                unnormalized_weights = []
                d_neighbors = []
                neighbor_groups = []

                for destination in adj[current_node]:
                    # Node2Vec p/q bias
                    if destination == source:
                        weight = 1.0 / self.p
                    elif destination in neighbors_of_source:
                        weight = 1.0
                    else:
                        weight = 1.0 / self.q

                    unnormalized_weights.append(weight)
                    d_neighbors.append(destination)
                    neighbor_groups.append(groups[destination])

                unnormalized_weights = np.array(unnormalized_weights)
                d_neighbors = np.array(d_neighbors)
                neighbor_groups = np.array(neighbor_groups)

                # Group by sensitive attribute and normalize within each group
                for group in unique_groups:
                    mask = neighbor_groups == group
                    if mask.sum() == 0:
                        continue
                    cur_weights = unnormalized_weights[mask]
                    cur_neighbors = d_neighbors[mask]

                    total = cur_weights.sum()
                    if total > 0:
                        d_graph[current_node]['probabilities'][group][source] = cur_weights / total
                    else:
                        d_graph[current_node]['probabilities'][group][source] = np.ones(len(cur_weights)) / len(cur_weights)
                    d_graph[current_node]['neighbors'][group] = list(cur_neighbors)

            # First travel probabilities (no previous node context)
            first_weights = []
            first_groups = []
            first_neighbors_list = []
            for neighbor in adj[source]:
                first_weights.append(1.0)  # uniform for first step
                first_groups.append(groups[neighbor])
                first_neighbors_list.append(neighbor)

            first_weights = np.array(first_weights)
            first_groups = np.array(first_groups)

            for group in unique_groups:
                mask = first_groups == group
                if mask.sum() == 0:
                    continue
                cur_weights = first_weights[mask]
                d_graph[source]['first_travel'][group] = cur_weights / cur_weights.sum()

            if (source + 1) % 5000 == 0:
                print(f"  Precomputed {source + 1}/{num_nodes} nodes...")

        # Generate fair walks
        print(f"[FAIRWALK] Generating {self.num_walks * num_nodes} fair walks...")
        walks = []

        for walk_iter in range(self.num_walks):
            nodes = list(range(num_nodes))
            np.random.shuffle(nodes)

            for source in nodes:
                walk = [source]

                while len(walk) < self.walk_length:
                    current = walk[-1]
                    group2neighbors = d_graph[current]['neighbors']

                    # Get groups that have at least one neighbor
                    available_groups = [g for g in group2neighbors if len(group2neighbors[g]) > 0]

                    if not available_groups:
                        break

                    # FairWalk core: uniformly pick a group
                    chosen_group = np.random.choice(available_groups)
                    walk_options = group2neighbors[chosen_group]

                    if len(walk) == 1:
                        # First step: use first_travel probabilities
                        if chosen_group in d_graph[current]['first_travel']:
                            probs = d_graph[current]['first_travel'][chosen_group]
                            next_node = np.random.choice(walk_options, p=probs)
                        else:
                            next_node = np.random.choice(walk_options)
                    else:
                        prev = walk[-2]
                        if chosen_group in d_graph[current]['probabilities'] and \
                           prev in d_graph[current]['probabilities'][chosen_group]:
                            probs = d_graph[current]['probabilities'][chosen_group][prev]
                            next_node = np.random.choice(walk_options, p=probs)
                        else:
                            next_node = np.random.choice(walk_options)

                    walk.append(next_node)

                walks.append([str(n) for n in walk])

            if (walk_iter + 1) % 2 == 0:
                print(f"  Walk iteration {walk_iter + 1}/{self.num_walks} done ({len(walks)} walks so far)")

        # Train Word2Vec
        print(f"[FAIRWALK] Training Word2Vec on {len(walks)} walks...")
        model = Word2Vec(
            sentences=walks,
            vector_size=self.embedding_dim,
            window=self.window_size,
            min_count=0,
            sg=1,
            workers=4,
            epochs=5
        )

        # Extract embeddings
        embeddings = np.zeros((num_nodes, self.embedding_dim))
        for i in range(num_nodes):
            if str(i) in model.wv:
                embeddings[i] = model.wv[str(i)]

        self.embeddings = torch.tensor(embeddings, dtype=torch.float32, device=device)
        print(f"[FAIRWALK] Embeddings trained: shape {self.embeddings.shape}")

    def predict_batch(self, edge_batch):
        """Predict scores using dot product of embeddings."""
        if self.embeddings is None:
            raise ValueError("Model not fitted. Call fit() first.")
        u_emb = self.embeddings[edge_batch[:, 0]]
        v_emb = self.embeddings[edge_batch[:, 1]]
        return (u_emb * v_emb).sum(dim=1)

    def get_embeddings(self):
        return self.embeddings


# ==================== CROSSWALK ====================

class CrossWalkEmbedding:
    """
    CrossWalk: Random walks biased toward crossing group boundaries.

    Nodes near the boundary between sensitive groups are upweighted during walks.
    At each step, with probability p_cross, the walker moves toward or across the
    group boundary, encouraging cross-group representation in embeddings.

    Reference: Khajehnejad et al., "CrossWalk: Fairness-Enhanced Node Representation Learning" (AAAI 2022)
    """

    def __init__(self, embedding_dim=128, walk_length=80, num_walks=10,
                 window_size=10, p_cross=0.5):
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.p_cross = p_cross
        self.embeddings = None

    def _compute_border_distance(self, adj, groups, num_nodes):
        """
        Compute border distance for each node via multi-source BFS.

        Border nodes are those with at least one neighbor of a different group.
        Distance = shortest path length to nearest border node.
        """
        from collections import deque

        border_distance = np.full(num_nodes, -1, dtype=int)
        queue = deque()

        # Find border nodes: nodes with at least one cross-group neighbor
        for node in range(num_nodes):
            node_group = groups[node]
            for neighbor in adj[node]:
                if groups[neighbor] != node_group:
                    border_distance[node] = 1
                    queue.append(node)
                    break

        # Multi-source BFS from all border nodes
        while queue:
            current = queue.popleft()
            for neighbor in adj[current]:
                if border_distance[neighbor] == -1:
                    border_distance[neighbor] = border_distance[current] + 1
                    queue.append(neighbor)

        # Isolated nodes (no neighbors) get distance 0
        border_distance[border_distance == -1] = 0

        return border_distance

    def fit(self, train_edges, num_nodes, sens_attr, device='cpu'):
        """
        Train CrossWalk embeddings.

        Args:
            train_edges: Training edges tensor [num_edges, 2]
            num_nodes: Total number of nodes
            sens_attr: Binary sensitive attribute tensor [num_nodes]
            device: Torch device
        """
        print(f"[CROSSWALK] Building graph and computing border distances...")

        # Build adjacency dict
        adj = {i: [] for i in range(num_nodes)}
        for u, v in train_edges.cpu().tolist():
            if u < num_nodes and v < num_nodes:
                adj[u].append(v)
                adj[v].append(u)

        # Deduplicate and sort
        for node in range(num_nodes):
            adj[node] = sorted(set(adj[node]))

        # Convert sens_attr
        if torch.is_tensor(sens_attr):
            groups = sens_attr.cpu().numpy().astype(int)
        else:
            groups = np.array(sens_attr, dtype=int)

        # Compute border distances
        border_distance = self._compute_border_distance(adj, groups, num_nodes)
        n_border = (border_distance == 1).sum()
        print(f"[CROSSWALK] Found {n_border} border nodes out of {num_nodes}")

        # Generate CrossWalk-biased random walks
        print(f"[CROSSWALK] Generating {self.num_walks * num_nodes} cross-walk biased walks (p_cross={self.p_cross})...")
        walks = []
        rng = np.random.RandomState(42)

        for walk_iter in range(self.num_walks):
            nodes = list(range(num_nodes))
            rng.shuffle(nodes)

            for source in nodes:
                walk = [source]

                while len(walk) < self.walk_length:
                    cur = walk[-1]
                    neighbors = adj[cur]

                    if not neighbors:
                        break

                    # Decide whether to use modified (cross-group biased) walk
                    if rng.random() < self.p_cross and border_distance[cur] > 0:
                        if border_distance[cur] == 1:
                            # At border: separate neighbors into same-group and cross-group
                            same_group = [u for u in neighbors if groups[u] == groups[cur]]
                            cross_group = [u for u in neighbors if groups[u] != groups[cur]]
                        else:
                            # Not at border: separate into "toward border" and "away from border"
                            cross_group = [u for u in neighbors if border_distance[u] < border_distance[cur]]
                            same_group = [u for u in neighbors if border_distance[u] >= border_distance[cur]]

                        if cross_group and same_group:
                            # With probability p_cross, move toward/across border
                            next_node = rng.choice(cross_group)
                        elif cross_group:
                            next_node = rng.choice(cross_group)
                        else:
                            next_node = rng.choice(neighbors)
                    else:
                        # Standard uniform random walk
                        next_node = rng.choice(neighbors)

                    walk.append(next_node)

                walks.append([str(n) for n in walk])

            if (walk_iter + 1) % 2 == 0:
                print(f"  Walk iteration {walk_iter + 1}/{self.num_walks} done ({len(walks)} walks so far)")

        # Train Word2Vec
        print(f"[CROSSWALK] Training Word2Vec on {len(walks)} walks...")
        model = Word2Vec(
            sentences=walks,
            vector_size=self.embedding_dim,
            window=self.window_size,
            min_count=0,
            sg=1,
            workers=4,
            epochs=5
        )

        # Extract embeddings
        embeddings = np.zeros((num_nodes, self.embedding_dim))
        for i in range(num_nodes):
            if str(i) in model.wv:
                embeddings[i] = model.wv[str(i)]

        self.embeddings = torch.tensor(embeddings, dtype=torch.float32, device=device)
        print(f"[CROSSWALK] Embeddings trained: shape {self.embeddings.shape}")

    def predict_batch(self, edge_batch):
        """Predict scores using dot product of embeddings."""
        if self.embeddings is None:
            raise ValueError("Model not fitted. Call fit() first.")
        u_emb = self.embeddings[edge_batch[:, 0]]
        v_emb = self.embeddings[edge_batch[:, 1]]
        return (u_emb * v_emb).sum(dim=1)

    def get_embeddings(self):
        return self.embeddings


# ==================== UGE (Unbiased Graph Embedding) ====================

class UGEBaseline(nn.Module):
    """
    UGE: Unbiased Graph Embedding using PyTorch Geometric.

    Three debiasing variants:
    - UGE-W: Reweight edges via inverse propensity weighting
    - UGE-R: Add regularization loss for attribute independence
    - UGE-C: Combined (UGE-W + UGE-R)

    Reference: Wang et al., "Unbiased Graph Embedding with Biased Graph Observations" (WWW 2022)
    """

    def __init__(self, num_nodes, hidden_channels=128, out_channels=64,
                 num_layers=2, dropout=0.3, gnn_type='gcn',
                 debias_method='uge-w', reg_weight=0.5):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        self.debias_method = debias_method
        self.reg_weight = reg_weight
        self._embeddings = None

        # Learnable node embeddings (since datasets may lack features)
        self.emb = nn.Embedding(num_nodes, hidden_channels)

        # GNN layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = hidden_channels
            out_ch = out_channels if i == num_layers - 1 else hidden_channels
            if gnn_type == 'gcn':
                self.convs.append(GCNConv(in_ch, out_ch))
            elif gnn_type == 'sage':
                self.convs.append(SAGEConv(in_ch, out_ch))
            else:
                raise ValueError(f"Unknown gnn_type: {gnn_type}")

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        """Forward pass through GNN layers."""
        for i, conv in enumerate(self.convs):
            if edge_weight is not None and self.gnn_type == 'gcn':
                x = conv(x, edge_index, edge_weight=edge_weight)
            else:
                x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def compute_edge_weights(self, edge_index, sens_attr):
        """
        Compute inverse propensity edge weights for UGE-W.

        Edges between same-group nodes get downweighted, cross-group edges
        get upweighted, to counteract homophily bias in the graph.

        Returns:
            edge_weight: Tensor [num_edges] with per-edge weights
        """
        sens = sens_attr.cpu().numpy() if torch.is_tensor(sens_attr) else np.array(sens_attr)

        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()

        src_groups = sens[src]
        dst_groups = sens[dst]

        # Count group sizes
        n0 = (sens == 0).sum()
        n1 = (sens == 1).sum()
        n = len(sens)

        # Count edge types
        same_00 = ((src_groups == 0) & (dst_groups == 0)).sum()
        same_11 = ((src_groups == 1) & (dst_groups == 1)).sum()
        cross = ((src_groups != dst_groups)).sum()
        total_edges = len(src)

        # Expected proportions under independence
        p00_expected = (n0 / n) * (n0 / n)
        p11_expected = (n1 / n) * (n1 / n)
        p_cross_expected = 2 * (n0 / n) * (n1 / n)

        # Actual proportions
        p00_actual = same_00 / max(total_edges, 1)
        p11_actual = same_11 / max(total_edges, 1)
        p_cross_actual = cross / max(total_edges, 1)

        # Inverse propensity weights
        weights = np.ones(total_edges, dtype=np.float32)

        mask_00 = (src_groups == 0) & (dst_groups == 0)
        mask_11 = (src_groups == 1) & (dst_groups == 1)
        mask_cross = src_groups != dst_groups

        if p00_actual > 0:
            weights[mask_00] = p00_expected / p00_actual
        if p11_actual > 0:
            weights[mask_11] = p11_expected / p11_actual
        if p_cross_actual > 0:
            weights[mask_cross] = p_cross_expected / p_cross_actual

        # Normalize so mean weight = 1
        weights = weights / weights.mean()

        return torch.tensor(weights, dtype=torch.float32, device=edge_index.device)

    def compute_regularization_loss(self, h, sens_attr):
        """
        Compute UGE-R regularization loss.

        Penalizes differences in average pairwise link scores between
        attribute-conditioned groups vs marginal (all-node) groups.
        For binary sensitive attributes, there are 4 (src, dst) group combos.
        """
        sens = sens_attr.long() if torch.is_tensor(sens_attr) else torch.tensor(sens_attr, dtype=torch.long)
        sens = sens.to(h.device)

        idx_0 = (sens == 0).nonzero(as_tuple=True)[0]
        idx_1 = (sens == 1).nonzero(as_tuple=True)[0]

        if len(idx_0) == 0 or len(idx_1) == 0:
            return torch.tensor(0.0, device=h.device)

        # Sample nodes for efficiency (max 200 per group)
        max_sample = 200
        if len(idx_0) > max_sample:
            perm = torch.randperm(len(idx_0), device=h.device)[:max_sample]
            idx_0_sample = idx_0[perm]
        else:
            idx_0_sample = idx_0

        if len(idx_1) > max_sample:
            perm = torch.randperm(len(idx_1), device=h.device)[:max_sample]
            idx_1_sample = idx_1[perm]
        else:
            idx_1_sample = idx_1

        # Marginal average score (sample from all nodes)
        all_idx = torch.cat([idx_0_sample, idx_1_sample])
        h_all = h[all_idx]
        marginal_score = (h_all @ h_all.t()).mean()

        # Group-conditioned scores for all 4 combos
        h0 = h[idx_0_sample]
        h1 = h[idx_1_sample]

        reg_loss = torch.tensor(0.0, device=h.device)

        # (0, 0)
        score_00 = (h0 @ h0.t()).mean()
        reg_loss = reg_loss + (score_00 - marginal_score) ** 2

        # (1, 1)
        score_11 = (h1 @ h1.t()).mean()
        reg_loss = reg_loss + (score_11 - marginal_score) ** 2

        # (0, 1)
        score_01 = (h0 @ h1.t()).mean()
        reg_loss = reg_loss + (score_01 - marginal_score) ** 2

        # (1, 0)
        score_10 = (h1 @ h0.t()).mean()
        reg_loss = reg_loss + (score_10 - marginal_score) ** 2

        return reg_loss / 4.0

    def predict_batch(self, edge_batch):
        """Predict scores using dot product of stored embeddings."""
        if self._embeddings is None:
            raise ValueError("Model not fitted. Call fit via BaselineWrapper first.")
        u_emb = self._embeddings[edge_batch[:, 0]]
        v_emb = self._embeddings[edge_batch[:, 1]]
        return (u_emb * v_emb).sum(dim=1)

    def get_embeddings(self):
        return self._embeddings


# ==================== MORAL ====================
# Faithful port of moral_original.py from /var/scratch/eerdem/MORAL/
# Key design:
#   - NUM_GROUPS = 3 (edge-level groups: sens[u]+sens[v] ∈ {0,1,2})
#   - One encoder + one predictor per edge-group (3 each)
#   - One Adam optimizer per group (3 total, independent)
#   - Per-group data loaders (edges partitioned by sens[u]+sens[v])
#   - Column-wise min-max feature normalization
#   - Best-checkpoint tracking (val loss)
#   - Dot-product decoder by default ("gae"); MLP decoder optional ("mlp")
#   - Hyperparameters match original: lr=3e-4, weight_decay=0.0,
#     num_hidden=128, dropout=0.2, batch_size=1024, epochs=300

_MORAL_NUM_GROUPS = 3  # sens[u]+sens[v] ∈ {0, 1, 2}


def _moral_normalize_features(features: torch.Tensor) -> torch.Tensor:
    """Column-wise min-max normalisation (matches moral_original.normalize_features)."""
    min_values = features.min(dim=0).values
    max_values = features.max(dim=0).values
    denom = torch.where(max_values > min_values,
                        max_values - min_values,
                        torch.ones_like(max_values))
    return (features - min_values) / denom


class TwoLayerGCN(nn.Module):
    """Two-layer GCN encoder (matches moral_original.TwoLayerGCN).
    dropout=0.2, uses nn.Dropout module (not functional dropout).
    """
    def __init__(self, in_channels: int, hidden_channels: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


class MoralTwoLayerGIN(nn.Module):
    """Two-layer GIN encoder (matches moral_original.TwoLayerGIN)."""
    def __init__(self, in_channels: int, hidden_channels: int, dropout: float = 0.2):
        super().__init__()
        mlp1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        mlp2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        from torch_geometric.nn import GINConv
        self.conv1 = GINConv(mlp1)
        self.conv2 = GINConv(mlp2)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


class MoralTwoLayerSAGE(nn.Module):
    """Two-layer SAGE encoder (matches moral_original.TwoLayerSAGE)."""
    def __init__(self, in_channels: int, hidden_channels: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


class MoralLinkPredictor(nn.Module):
    """Link predictor (matches moral_original.LinkPredictor).

    Two modes:
      - hidden_channels=None  → dot product (GAE / standard decoder)
      - hidden_channels=int   → MLP: Linear(2*in, hidden) → ReLU → Linear(hidden, 1)

    forward(embeddings, edge_index) where edge_index is [2, E].
    """
    def __init__(self, in_channels: int, hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            self.network = None
        else:
            self.network = nn.Sequential(
                nn.Linear(in_channels * 2, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1),
            )

    def forward(self, embeddings, edge_index):
        src, dst = edge_index
        if self.network is None:
            scores = (embeddings[src] * embeddings[dst]).sum(dim=-1, keepdim=True)
        else:
            features = torch.cat([embeddings[src], embeddings[dst]], dim=-1)
            scores = self.network(features)
        return scores.view(-1)


def _moral_build_encoder(name: str, in_channels: int, hidden_channels: int) -> nn.Module:
    name = name.lower()
    if name == "gcn":
        return TwoLayerGCN(in_channels, hidden_channels)
    if name == "gin":
        return MoralTwoLayerGIN(in_channels, hidden_channels)
    if name == "sage":
        return MoralTwoLayerSAGE(in_channels, hidden_channels)
    raise ValueError(f"[MORAL] Unsupported encoder '{name}'.")


def _moral_build_predictor(name: str, hidden_channels: int) -> MoralLinkPredictor:
    name = name.lower()
    if name in {"gae", "dot", "standard"}:
        return MoralLinkPredictor(hidden_channels, hidden_channels=None)
    if name in {"mlp", "mlp_decoder"}:
        return MoralLinkPredictor(hidden_channels, hidden_channels)
    raise ValueError(f"[MORAL] Unsupported decoder '{name}'.")


class _MoralEdgeLabelDataset(torch.utils.data.Dataset):
    """Edge dataset returning (edge_pair [2], label) tuples."""
    def __init__(self, edges: torch.Tensor, labels: torch.Tensor):
        if edges.size(0) != labels.size(0):
            raise ValueError("edges and labels must have the same number of rows")
        self.edges = edges.long()
        self.labels = labels.float()

    def __len__(self):
        return int(self.edges.size(0))

    def __getitem__(self, idx):
        return self.edges[idx], self.labels[idx]


class MORAL(nn.Module):
    """MORAL: Faithful port of moral_original.py.

    Groups edges by sens[u]+sens[v] ∈ {0,1,2} (NUM_GROUPS=3).
    Trains one (encoder, predictor, optimizer) triple per edge-group independently.
    Applies column-wise min-max feature normalization.
    Tracks best validation checkpoint and restores it after training.

    Reference: "Breaking the Dyadic Barrier: Rethinking Fairness in Link Prediction
    Beyond Demographic Parity" — the MORAL implementation used in the AAAI submission.

    Original: /var/scratch/eerdem/MORAL/moral_original.py
    """

    def __init__(
        self,
        adj,
        features,
        labels,
        idx_train,
        idx_val,
        idx_test,
        sens,
        sens_idx,
        edge_splits,
        dataset_name,
        num_hidden: int = 128,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        encoder: str = "gcn",
        decoder: str = "gae",
        batch_size: int = 1024,
        device='cpu',
    ):
        super().__init__()

        self.device = torch.device(device)
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_hidden = num_hidden
        self.criterion = nn.BCEWithLogitsLoss()

        # Feature normalization (matches original)
        self.features = _moral_normalize_features(features.float()).to(self.device)
        # adj must be coalesced before calling .indices() (matches original)
        self.edge_index = adj.coalesce().indices().to(self.device)

        self.labels = labels.to(self.device) if labels is not None else None
        self.sens = sens.to(self.device) if torch.is_tensor(sens) else torch.tensor(sens, device=self.device)
        self.sens_idx = sens_idx
        self.sens_cpu = self.sens.cpu()
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test

        in_channels = self.features.size(1)

        # 3 encoders + 3 predictors (one per edge-group), all on device
        self.encoders = nn.ModuleList([
            _moral_build_encoder(encoder, in_channels, num_hidden).to(self.device)
            for _ in range(_MORAL_NUM_GROUPS)
        ])
        self.predictors = nn.ModuleList([
            _moral_build_predictor(decoder, num_hidden).to(self.device)
            for _ in range(_MORAL_NUM_GROUPS)
        ])

        # One independent Adam optimizer per group (matches original)
        self.optimizers = [
            torch.optim.Adam(
                list(self.encoders[g].parameters()) + list(self.predictors[g].parameters()),
                lr=lr,
                weight_decay=weight_decay,
            )
            for g in range(_MORAL_NUM_GROUPS)
        ]

        # Build per-group loaders
        self.train_loaders = self._build_group_loaders(edge_splits.get("train"), shuffle=True)
        self.valid_loaders = self._build_group_loaders(edge_splits.get("valid"), shuffle=False)

        test_edges, test_labels = self._prepare_edges(edge_splits.get("test"))
        self.test_edges = test_edges  # on CPU (matching original)
        self.test_labels = test_labels

        self.best_state = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_edges(self, split):
        """Merge pos and neg edges with binary labels. Returns (edges [N,2], labels [N])."""
        if split is None:
            return torch.empty(0, 2, dtype=torch.long), torch.empty(0)
        pos_edges = split["edge"].long()
        neg_edges = split["edge_neg"].long()
        edges = torch.cat([pos_edges, neg_edges], dim=0)
        labels = torch.cat([
            torch.ones(pos_edges.size(0)),
            torch.zeros(neg_edges.size(0)),
        ], dim=0)
        return edges, labels

    def _build_group_loaders(self, split, shuffle: bool):
        """Build one DataLoader per edge-group (sens[u]+sens[v] ∈ {0,1,2}).
        Matches moral_original._build_group_loaders exactly.
        """
        from torch.utils.data import DataLoader
        if split is None:
            return [None] * _MORAL_NUM_GROUPS

        edges, labels = self._prepare_edges(split)
        # Group edges by sens[u] + sens[v] (the 3-group definition)
        sens_groups = self.sens_cpu[edges].sum(dim=1)  # [N] with values in {0,1,2}

        loaders = []
        for group in range(_MORAL_NUM_GROUPS):
            mask = sens_groups == group
            if mask.sum() == 0:
                loaders.append(None)
                continue
            dataset = _MoralEdgeLabelDataset(edges[mask], labels[mask])
            batch = len(dataset) if self.batch_size <= 0 else self.batch_size
            loader = DataLoader(dataset, batch_size=batch, shuffle=shuffle, drop_last=False)
            loaders.append(loader)
        return loaders

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train_epoch(self) -> float:
        """One training epoch — matches moral_original._train_epoch."""
        total_loss = 0.0
        total_batches = 0

        for group, loader in enumerate(self.train_loaders):
            if loader is None:
                continue

            encoder = self.encoders[group]
            predictor = self.predictors[group]
            optimizer = self.optimizers[group]

            encoder.train()
            predictor.train()

            for edges, labels in loader:
                optimizer.zero_grad()
                edges = edges.t().to(self.device)   # [2, B]
                labels = labels.to(self.device)

                embeddings = encoder(self.features, self.edge_index)
                logits = predictor(embeddings, edges)
                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                total_batches += 1

        return total_loss / max(total_batches, 1)

    @torch.no_grad()
    def _evaluate(self, loaders):
        """Evaluate on a set of group loaders — matches moral_original._evaluate."""
        total_loss = 0.0
        total_batches = 0

        for group, loader in enumerate(loaders):
            if loader is None:
                continue

            self.encoders[group].eval()
            self.predictors[group].eval()

            # Compute embeddings once per group (matches original)
            embeddings = self.encoders[group](self.features, self.edge_index)

            for edges, labels in loader:
                edges = edges.t().to(self.device)
                labels = labels.to(self.device)
                logits = self.predictors[group](embeddings, edges)
                loss = self.criterion(logits, labels)
                total_loss += float(loss.item())
                total_batches += 1

        if total_batches == 0:
            return None
        return total_loss / total_batches

    def fit(self, epochs: int = 300) -> None:
        """Train model, restore best validation checkpoint — matches moral_original.fit."""
        from copy import deepcopy

        best_val = float("inf")
        best_state = None

        print(f"[MORAL] Training for {epochs} epochs on {self.device}...")
        for epoch in tqdm(range(1, epochs + 1)):
            train_loss = self._train_epoch()
            val_loss = self._evaluate(self.valid_loaders)

            if val_loss is not None and val_loss < best_val:
                best_val = val_loss
                best_state = deepcopy(self.state_dict())

            if epoch % 10 == 0 or epoch == 1:
                if val_loss is None:
                    print(f"  [MORAL] Epoch {epoch}/{epochs} | train={train_loss:.4f}")
                else:
                    print(f"  [MORAL] Epoch {epoch}/{epochs} | train={train_loss:.4f}"
                          f" | val={val_loss:.4f} | best={best_val:.4f}")

        if best_state is not None:
            self.load_state_dict(best_state)
            self.best_state = best_state
        print("[MORAL] Training complete.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self) -> torch.Tensor:
        """Return logits for all test edges — matches moral_original.predict.

        Test edges are routed to the encoder/predictor of their edge-group
        (sens[u]+sens[v]).
        """
        if self.test_edges.numel() == 0:
            raise RuntimeError("[MORAL] Test split is empty.")

        outputs = torch.zeros(self.test_edges.size(0), device=self.device)
        sens_groups = self.sens_cpu[self.test_edges].sum(dim=1)

        for group in range(_MORAL_NUM_GROUPS):
            mask = sens_groups == group
            if mask.sum() == 0:
                continue

            edges = self.test_edges[mask].t().to(self.device)
            self.encoders[group].eval()
            self.predictors[group].eval()
            embeddings = self.encoders[group](self.features, self.edge_index)
            logits = self.predictors[group](embeddings, edges)
            outputs[mask] = logits

        return outputs

    def predict_batch(self, edge_batch):
        """Predict scores for a batch of edges [B, 2]. Uses per-group routing."""
        with torch.no_grad():
            self.eval()
            scores = torch.zeros(edge_batch.size(0), device=self.device)
            sens_groups = self.sens_cpu[edge_batch.cpu()].sum(dim=1)
            for group in range(_MORAL_NUM_GROUPS):
                mask = sens_groups == group
                if mask.sum() == 0:
                    continue
                edges = edge_batch[mask].t().to(self.device)
                embeddings = self.encoders[group](self.features, self.edge_index)
                scores[mask] = self.predictors[group](embeddings, edges)
        return scores

    @torch.no_grad()
    def predict_scores(self, edges):
        """Score an arbitrary set of edges [B, 2] using per-group routing.

        Unlike predict() which uses the stored test edges, this method
        accepts any edge list — enabling Hits@K / MRR computation on
        pos/neg test splits that the evaluation framework provides.
        """
        edges = edges.long()
        scores = torch.zeros(edges.size(0), device=self.device)
        sens_groups = self.sens_cpu[edges.cpu()].sum(dim=1)

        for group in range(_MORAL_NUM_GROUPS):
            mask = sens_groups == group
            if mask.sum() == 0:
                continue
            edge_t = edges[mask].t().to(self.device)
            self.encoders[group].eval()
            self.predictors[group].eval()
            embeddings = self.encoders[group](self.features, self.edge_index)
            scores[mask] = self.predictors[group](embeddings, edge_t)

        return scores

    @torch.no_grad()
    def predict_topk(self, train_edges, test_nodes, k=100, batch_size=4096):
        """Generate top-k link predictions per user, excluding training edges.

        Uses vectorised scoring: for each chunk of users, computes all
        (user, candidate) scores via the group-specific predictor, masks
        training edges with a sparse CSR matrix, then extracts top-k.

        Matches the scaffold's build_recommendations() pattern.

        Args:
            train_edges: [E, 2] tensor of training edges (to exclude)
            test_nodes: list of node IDs to generate recommendations for
            k: number of recommendations per node
            batch_size: number of candidate nodes scored per predictor call

        Returns:
            dict {user_id -> list[neighbour_id]} of length <= k
        """
        import scipy.sparse as sp
        from tqdm import tqdm

        num_nodes = self.features.size(0)
        sens_np = self.sens_cpu.numpy()

        # Pre-compute all group embeddings once
        group_embeddings = []
        for group in range(_MORAL_NUM_GROUPS):
            self.encoders[group].eval()
            self.predictors[group].eval()
            emb = self.encoders[group](self.features, self.edge_index)
            group_embeddings.append(emb)

        # Build sparse training-edge mask (CSR) for fast row-wise exclusion
        if train_edges is not None and train_edges.numel() > 0:
            edges_np = train_edges.cpu().numpy()
            rows = np.concatenate([edges_np[:, 0], edges_np[:, 1]])
            cols = np.concatenate([edges_np[:, 1], edges_np[:, 0]])
            valid = (rows < num_nodes) & (cols < num_nodes)
            rows, cols = rows[valid], cols[valid]
            train_mask_csr = sp.csr_matrix(
                (np.ones(len(rows), dtype=np.float32), (rows, cols)),
                shape=(num_nodes, num_nodes),
            )
        else:
            train_mask_csr = sp.csr_matrix((num_nodes, num_nodes))

        # Pre-compute which candidate nodes belong to each sens group
        # For user with sens=s_u, candidate v has group = s_u + sens[v]
        cand_sens = sens_np  # [N]

        recs_by_u = {}
        chunk_size = 128

        for i in tqdm(range(0, len(test_nodes), chunk_size),
                      desc="[MORAL] predict_topk", unit="chunk"):
            chunk = test_nodes[i:i + chunk_size]
            chunk_arr = np.array(chunk, dtype=np.int64)

            # Initialise score matrix for this chunk: [chunk_size, N]
            scores = torch.full(
                (len(chunk), num_nodes), float('-inf'), device=self.device
            )

            for group in range(_MORAL_NUM_GROUPS):
                emb = group_embeddings[group]
                predictor = self.predictors[group]

                for local_idx, u in enumerate(chunk):
                    u_int = int(u)
                    s_u = int(sens_np[u_int])

                    # Which candidates have sens[u]+sens[v] == group?
                    target_s_v = group - s_u
                    if target_s_v < 0 or target_s_v > 1:
                        continue  # impossible for binary sens

                    cand_idx = np.where(cand_sens == target_s_v)[0]
                    if len(cand_idx) == 0:
                        continue

                    cand_t = torch.from_numpy(cand_idx).long().to(self.device)

                    # Batch-score
                    for start in range(0, len(cand_t), batch_size):
                        end = min(start + batch_size, len(cand_t))
                        v_batch = cand_t[start:end]
                        u_batch = torch.full_like(v_batch, u_int)
                        edge_idx = torch.stack([u_batch, v_batch], dim=0)  # [2, B]
                        logits = predictor(emb, edge_idx)
                        scores[local_idx, v_batch.cpu().numpy()] = logits.cpu()

            # Mask self-loops
            scores[range(len(chunk)), chunk_arr] = float('-inf')

            # Mask training edges via sparse matrix
            mask_block = train_mask_csr[chunk_arr]  # sparse [chunk_size, N]
            mask_rows, mask_cols = mask_block.nonzero()
            scores[mask_rows, mask_cols] = float('-inf')

            # Top-k per user
            topk_scores, topk_indices = torch.topk(
                scores, min(k, num_nodes), dim=1
            )
            for local_idx, u in enumerate(chunk):
                valid_mask = torch.isfinite(topk_scores[local_idx])
                recs_by_u[int(u)] = topk_indices[local_idx][valid_mask][:k].tolist()

        return recs_by_u

    @staticmethod
    def fair_metric(pred, labels, sens):
        """Demographic parity and equality of opportunity — matches moral_original."""
        idx_s0 = sens == 0
        idx_s1 = sens == 1
        idx_s0_y1 = torch.logical_and(idx_s0, labels == 1)
        idx_s1_y1 = torch.logical_and(idx_s1, labels == 1)
        parity = torch.tensor(0.0)
        equality = torch.tensor(0.0)
        if idx_s0.sum() > 0 and idx_s1.sum() > 0:
            parity = (pred[idx_s0].mean() - pred[idx_s1].mean()).abs()
        if idx_s0_y1.sum() > 0 and idx_s1_y1.sum() > 0:
            equality = (pred[idx_s0_y1].mean() - pred[idx_s1_y1].mean()).abs()
        return float(parity.item()), float(equality.item())


def train_moral(base_model, data, split_edge, args, node_categories=None, target_dist=None):
    """Adapter to construct and train a MORAL model from the scaffold's data format.

    Hyperparameter defaults match moral_original:
      lr=3e-4, weight_decay=0.0, num_hidden=128, batch_size=1024, epochs=300,
      encoder="gcn", decoder="gae" (dot-product).
    """
    lr = getattr(args, 'lr', 3e-4)
    hidden_channels = getattr(args, 'emb_hidden_channels', 128)
    batch_size = getattr(args, 'batch_size', 1024)
    epochs = getattr(args, 'moral_epochs', 300)
    encoder_name = getattr(args, 'moral_encoder', 'gcn')
    decoder_name = getattr(args, 'moral_decoder', 'gae')
    weight_decay = getattr(args, 'weight_decay', 0.0)
    if hasattr(args, 'device'):
        if isinstance(args.device, int):
            device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        else:
            device = str(args.device)
    else:
        device = 'cpu'
    
    print(f"[MORAL DEBUG] Received args.device: {getattr(args, 'device', None)} (type: {type(getattr(args, 'device', None))}) => Resolving to: {device}, cuda_available={torch.cuda.is_available()}")

    num_nodes = data.num_nodes

    # Build sparse adjacency tensor
    idx = data.edge_index
    val = torch.ones(idx.size(1))
    adj = torch.sparse_coo_tensor(idx, val, (num_nodes, num_nodes)).coalesce()

    # Features
    if data.x is not None:
        features = data.x.float()
    else:
        features = torch.eye(num_nodes)

    # Sensitive attribute
    sens = None
    if hasattr(args, 'sens_attr') and args.sens_attr is not None:
        sens = args.sens_attr
    elif hasattr(data, 'sens'):
        sens = data.sens
    if sens is None:
        print("[MORAL] Warning: No sensitive attribute found, defaulting to zeros.")
        sens = torch.zeros(num_nodes, dtype=torch.long)

    # Ensure negative training edges exist
    if 'edge_neg' not in split_edge.get('train', {}):
        print("[MORAL] Generating negative training edges...")
        from torch_geometric.utils import negative_sampling, to_undirected
        pos_edge_index = split_edge['train']['edge'].t()
        neg_edge_index = negative_sampling(
            edge_index=to_undirected(pos_edge_index),
            num_nodes=num_nodes,
            num_neg_samples=pos_edge_index.size(1),
        )
        split_edge['train']['edge_neg'] = neg_edge_index.t()

    moral = MORAL(
        adj=adj,
        features=features,
        labels=torch.zeros(num_nodes),  # labels not used in link pred
        idx_train=None,
        idx_val=None,
        idx_test=None,
        sens=sens,
        sens_idx=None,
        edge_splits=split_edge,
        dataset_name=getattr(args, 'data_name', 'unknown'),
        num_hidden=hidden_channels,
        lr=lr,
        weight_decay=weight_decay,
        encoder=encoder_name,
        decoder=decoder_name,
        batch_size=batch_size,
        device=device,
    )

    moral.fit(epochs=epochs)
    return moral, {}

# ==================== Cross-Link-Bias (Twins GNN) ====================


class CLB_SAGE(nn.Module):
    """
    SAGE encoder matching the original Cross-Link-Bias implementation.
    Uses learned node embeddings (nn.Embedding), NOT node features.
    No dropout. All layers: emb_dim -> emb_dim.
    Matches original model.py SAGE class.
    """
    def __init__(self, num_nodes, emb_dim, num_layers):
        super().__init__()
        self.emb_table = nn.Embedding(num_nodes, emb_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(emb_dim, emb_dim))

    def forward(self, edge_index):
        x = self.emb_table.weight
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)
        return x


class CLBEdgeDataloader:
    """
    Mini-batch edge dataloader matching the original Cross-Link-Bias implementation.
    Each batch is an independent random sample of edges (sampling with replacement across batches).
    """
    def __init__(self, edge_set, confidence, batch_size, ratio):
        self.edge_set = edge_set
        self.batch_size = batch_size
        self.confidence = confidence
        self.num_edges = edge_set.size()[0]
        self.batch_per_epoch = max(int(self.num_edges * ratio / self.batch_size), 1)
        self.batch_remain = None

    def __len__(self):
        return self.batch_per_epoch

    def __iter__(self):
        self.batch_remain = self.batch_per_epoch
        return self

    def __next__(self):
        if self.batch_remain == 0:
            raise StopIteration
        self.batch_remain -= 1
        eid = torch.randperm(self.num_edges)[:self.batch_size]
        return (self.edge_set[eid], self.confidence[eid])


class Merger(nn.Module):
    """
    Embedding fusion module for Cross-Link-Bias.
    Matches original: single linear layer from in_dim -> out_dim (no hidden layers,
    no ReLU, no dropout). The multi-layer code is commented out in the original.
    Uses xavier_uniform weight init and zero bias init.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout):
        super(Merger, self).__init__()
        self.layers = nn.ModuleList()
        # Original: only a single linear layer (multi-layer code is commented out)
        self.layers.append(nn.Linear(in_dim, out_dim))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.layers:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.constant_(lin.bias, 0)

    def forward(self, emb_ori, emb_aug):
        x = torch.cat((emb_ori, emb_aug), dim=1)
        # With only 1 layer, this loop body never executes (matches original)
        for lin in self.layers[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
        x = self.layers[-1](x)
        return x


class CrossLinkBias(nn.Module):
    """
    Cross-Link-Bias (Twins GNN): Fairness-aware link prediction via
    twin GNN training with cross-group edge augmentation.

    Faithfully ported from the original implementation.

    Key design choices matching the original:
    - Uses nn.Embedding (learned embeddings), NOT node features
    - No dropout in SAGE encoder
    - model_aug is deepcopy of model_ori (shared initial weights)
    - Both twin GNNs share the same message-passing edge_index
    - Merger is a single linear layer (128 -> 64) with xavier init, dropout=0
    - 3-phase training per epoch: (1) ori alone, (2) aug alone, (3) joint merge
    - Sigmoid-based dynamic curriculum for merge phase LR and steps
    - Confidence-weighted BCE loss
    - Gradient clipping at max_norm=1.0
    - Negative sampling via uniform torch.randint (no filtering)
    - Random subsample 2*n_aug candidates before Jaccard scoring
    - Pseudo-edges added one-directionally to training edge set
    - Mini-batch training via CLBEdgeDataloader
    """
    def __init__(self, num_nodes, num_features, hidden_dim=64, out_dim=64, num_layers=2, dropout=0.5,
                 aug_size=1.25, ori_lr=0.001, aug_lr=0.001, alpha=0.00005, beta=100, threshold=50,
                 batch_size=4096, train_ratio=1.0, add_edge=True, device='cpu', gnn_type='sage'):
        super(CrossLinkBias, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.aug_size = aug_size
        self.ori_lr = ori_lr
        self.aug_lr = aug_lr
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.add_edge = add_edge
        self.device = device

        # Build SAGE encoder with nn.Embedding (no node features, no dropout)
        self.model_ori = CLB_SAGE(num_nodes, hidden_dim, num_layers).to(device)
        # Augmented model is a deepcopy (shared initial weights, matches original)
        import copy
        self.model_aug = copy.deepcopy(self.model_ori)

        # Merger: single linear layer (hidden_dim*2 -> out_dim), dropout=0
        # Matches original: merger(in_dim=128, hidden_dim=64, out_dim=64, num_layers=n_layer, dropout=0)
        self.model_cal = Merger(hidden_dim * 2, hidden_dim, out_dim, num_layers=num_layers, dropout=0)
        self.model_cal.reset_parameters()

        self.to(device)
        self.embeddings = None

    def _augment_graph(self, edge_index):
        """
        Generate augmented graph with cross-group pseudo-edges (Jaccard method).
        Matches original pseudo_generation() in augmentation.py.
        Uses Louvain community detection for group membership (NOT sensitive attributes).

        Returns:
            msg_edge_index: [2, E'] edge_index for GNN forward pass (both models)
            pseudo_edges: [2, P] selected pseudo-edges (one direction only)
        """
        import networkit as nk

        print(f"[CrossLinkBias] Generating augmented graph (aug_size={self.aug_size})...")

        edge_index_cpu = edge_index.cpu()
        num_nodes = self.num_nodes

        # Louvain community detection via networkit (PLM = Parallel Louvain Method)
        print("[CrossLinkBias] >> Louvain Clustering (networkit)...")
        nk_graph = nk.Graph(num_nodes, weighted=False, directed=False)
        src_np = edge_index_cpu[0].numpy()
        dst_np = edge_index_cpu[1].numpy()
        for i in range(src_np.shape[0]):
            u, v = int(src_np[i]), int(dst_np[i])
            if u != v and not nk_graph.hasEdge(u, v):
                nk_graph.addEdge(u, v)
        plm = nk.community.PLM(nk_graph)
        plm.run()
        partition = plm.getPartition()
        membership = np.array([partition.subsetOf(i) for i in range(num_nodes)])
        num_cluster = partition.numberOfSubsets()
        print(f"[CrossLinkBias] >> Louvain Clustering Finished, {num_cluster} communities detected.")

        # 1. 2-hop neighbors (using sparse multiplication)
        adj = torch.sparse_coo_tensor(edge_index_cpu, torch.ones(edge_index_cpu.shape[1]), (num_nodes, num_nodes))
        adj = adj.coalesce()
        adj_sq = torch.sparse.mm(adj, adj).coalesce()

        indices = adj_sq.indices()
        mask_no_loop = indices[0] != indices[1]
        indices = indices[:, mask_no_loop]

        # 2. Filter for cross-community
        src, dst = indices[0], indices[1]
        mask_cross = membership[src] != membership[dst]
        all_pseudo_edges = indices[:, mask_cross]

        if all_pseudo_edges.size(1) == 0:
            print("[CrossLinkBias] Warning: No 2-hop cross-group options found.")
            return edge_index.to(self.device), torch.zeros(2, 0, dtype=torch.long, device=self.device)

        print(f"[CrossLinkBias] Found {all_pseudo_edges.size(1)} candidate cross-group 2-hop edges.")

        # 3. Compute augment_size based on intra/inter community imbalance
        src_orig, dst_orig = edge_index_cpu[0], edge_index_cpu[1]
        intra_mask = membership[src_orig] == membership[dst_orig]
        intra_count = intra_mask.sum().item()
        inter_count = (~intra_mask).sum().item()
        n_aug = int((intra_count - inter_count) * self.aug_size)
        n_aug = max(n_aug, 0)

        if n_aug == 0:
            print("[CrossLinkBias] No augmentation needed (inter >= intra).")
            return edge_index.to(self.device), torch.zeros(2, 0, dtype=torch.long, device=self.device)

        # 4. Random subsample 2 * n_aug candidates (matches original torch.manual_seed(0))
        torch.manual_seed(0)
        n_candidates = all_pseudo_edges.size(1)
        n_sample = min(n_aug * 2, n_candidates)
        perm = torch.randperm(n_candidates)[:n_sample]
        sampled_candidates = all_pseudo_edges[:, perm]

        # 5. Jaccard similarity via vectorized sparse operations
        from scipy.sparse import csr_matrix
        print(f"[CrossLinkBias] Computing Jaccard on {sampled_candidates.size(1)} candidates (vectorized)...")

        # Build symmetric adjacency as scipy CSR for fast row access
        ones = np.ones(edge_index_cpu.shape[1], dtype=np.float32)
        adj_scipy = csr_matrix(
            (ones, (edge_index_cpu[0].numpy(), edge_index_cpu[1].numpy())),
            shape=(num_nodes, num_nodes)
        )
        # Degree vector for union computation: |N(u)| + |N(v)| - intersection
        degree = np.array(adj_scipy.sum(axis=1)).flatten()

        cand_src = sampled_candidates[0].numpy()
        cand_dst = sampled_candidates[1].numpy()

        # Intersection: number of common neighbors = (adj[u] * adj[v]).sum() per pair
        # Compute via batched sparse dot products
        src_rows = adj_scipy[cand_src]  # sparse matrix [n_candidates, num_nodes]
        dst_rows = adj_scipy[cand_dst]  # sparse matrix [n_candidates, num_nodes]
        intersection = np.array(src_rows.multiply(dst_rows).sum(axis=1)).flatten()

        union = degree[cand_src] + degree[cand_dst] - intersection
        # Jaccard = intersection / union (0 if union == 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            jaccard_scores = np.where(union > 0, intersection / union, 0.0)

        scores = torch.tensor(jaccard_scores, dtype=torch.float32)
        valid_edges = sampled_candidates  # [2, N], already in correct format

        # 6. Top-k by Jaccard score
        k = min(n_aug, len(scores))
        _, top_idx = scores.topk(k, dim=0)
        pseudo_edges = valid_edges[:, top_idx]

        print(f"[CrossLinkBias] Selected {pseudo_edges.size(1)} pseudo-edges.")

        # 7. Build message-passing edge_index (add_edge controls graph augmentation)
        if self.add_edge:
            # Add pseudo-edges bidirectionally to GNN message-passing graph (matches original)
            msg_edge_index = torch.cat([
                edge_index_cpu,
                pseudo_edges,
                torch.stack([pseudo_edges[1], pseudo_edges[0]])
            ], dim=1)
        else:
            msg_edge_index = edge_index_cpu

        return msg_edge_index.to(self.device), pseudo_edges.to(self.device)

    def _train_single_model(self, model, dataloader, lr, num_nodes):
        """
        Train a single GNN for one epoch.
        Matches Phase 1 and Phase 2 of the original e2e_train.
        Both models use self._msg_edge_index for message passing (matches original).
        """
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for data in dataloader:
            pos_edge = data[0].t().to(self.device)
            confidence = data[1].to(self.device)  # [B]

            embedding = model(self._msg_edge_index)

            # No .unsqueeze(-1) on scores: keeps [B] shape, avoiding [B,B] broadcast with confidence.
            # No / confidence.sum(): matches original's actual sum-reduction behavior
            # (the original's broadcast accidentally cancels the division).
            pos_out = torch.sigmoid(
                torch.sum(embedding[pos_edge[0]] * embedding[pos_edge[1]], dim=-1)
            )  # [B]
            pos_loss = (-torch.log(pos_out + 1e-15) * confidence).sum()

            neg_edge = torch.randint(0, num_nodes, pos_edge.size(), dtype=torch.long, device=self.device)
            neg_out = torch.sigmoid(
                torch.sum(embedding[neg_edge[0]] * embedding[neg_edge[1]], dim=-1)
            )  # [B]
            neg_loss = (-torch.log(1 - neg_out + 1e-15) * confidence).sum()

            loss = pos_loss + neg_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    def _merge_phase_eval(self, val_pos_edges, val_neg_edges):
        """
        Evaluate merged embeddings on validation set using Hits@50.
        Matches original e2e_test() in process.py (social dataset path).

        Args:
            val_pos_edges: [N, 2] positive validation edges
            val_neg_edges: [N, 2] negative validation edges

        Returns:
            hits_at_50: float
        """
        from torch.utils.data import DataLoader

        self.model_ori.eval()
        self.model_aug.eval()
        self.model_cal.eval()

        with torch.no_grad():
            emb_ori = self.model_ori(self._msg_edge_index)
            emb_aug = self.model_aug(self._msg_edge_index)
            embedding = self.model_cal(emb_ori, emb_aug).detach()

            # Positive predictions (batched, matches original)
            pos_preds = []
            for perm in DataLoader(range(val_pos_edges.size(0)), self.batch_size):
                edge = val_pos_edges[perm].t()
                pos_preds.append(
                    torch.sigmoid(torch.sum(embedding[edge[0]] * embedding[edge[1]], dim=-1).unsqueeze(-1)).squeeze().cpu()
                )
            pos_pred = torch.cat(pos_preds, dim=0)

            # Negative predictions (batched, matches original)
            neg_preds = []
            for perm in DataLoader(range(val_neg_edges.size(0)), self.batch_size):
                edge = val_neg_edges[perm].t()
                neg_preds.append(
                    torch.sigmoid(torch.sum(embedding[edge[0]] * embedding[edge[1]], dim=-1).unsqueeze(-1)).squeeze().cpu()
                )
            neg_pred = torch.cat(neg_preds, dim=0)

            # Hits@50 (matches original utils.py evaluate())
            K = 50
            if len(neg_pred) < K:
                return 1.0
            kth_score = torch.topk(neg_pred, K)[0][-1]
            hits_at_50 = float(torch.sum(pos_pred > kth_score).cpu()) / len(pos_pred)

        return hits_at_50

    def fit(self, edge_index, features=None, sens_attr=None, epochs=100, lr=0.01,
            ori_lr=None, aug_lr=None, val_pos_edges=None, val_neg_edges=None):
        """
        Train the Twins GNN model with the 3-phase training strategy.

        Note: features and sens_attr parameters are accepted for API compatibility
        but ignored. The original uses learned embeddings (nn.Embedding) and
        Louvain community detection (not sensitive attributes).

        Per epoch:
          Phase 1: Train model_ori alone on original edges (lr=ori_lr)
          Phase 2: Train model_aug alone on augmented edges (lr=aug_lr)
          Phase 3: Joint merge training with sigmoid-scheduled LR and STEP count
                   + inner validation to select best model_cal (matches original)

        Args:
            edge_index: [2, E] edge index tensor
            features: Ignored (API compatibility)
            sens_attr: Ignored (API compatibility)
            epochs: Number of training epochs
            lr: Ignored (uses ori_lr/aug_lr)
            ori_lr: Learning rate for model_ori
            aug_lr: Learning rate for model_aug
            val_pos_edges: [N, 2] positive validation edges for merge-phase inner validation
            val_neg_edges: [N, 2] negative validation edges for merge-phase inner validation
        """
        import math
        import copy

        self.embeddings = None
        ori_lr = ori_lr if ori_lr is not None else self.ori_lr
        aug_lr = aug_lr if aug_lr is not None else self.aug_lr

        has_val = val_pos_edges is not None and val_neg_edges is not None
        if has_val:
            val_pos_edges = val_pos_edges.to(self.device)
            val_neg_edges = val_neg_edges.to(self.device)

        # 1. Augment graph (uses Louvain community detection internally)
        msg_edge_index, pseudo_edges = self._augment_graph(edge_index)
        self._msg_edge_index = msg_edge_index  # shared by BOTH models for forward pass

        # Prepare training edge sets in [N, 2] format for dataloaders
        # Original training edges (for model_ori and merge phase)
        ori_train_edges = edge_index.t().to(self.device)   # [num_edges, 2]
        ori_confidence = torch.ones(ori_train_edges.shape[0], device=self.device)

        # Augmented training edges: original + pseudo-edges ONE direction (matches original)
        if pseudo_edges.size(1) > 0:
            aug_train_edges = torch.cat(
                [ori_train_edges, pseudo_edges.t().to(self.device)], dim=0
            )
        else:
            aug_train_edges = ori_train_edges
        aug_confidence = torch.ones(aug_train_edges.shape[0], device=self.device)

        # Early stopping patience (matches original e2e_main.py line 86)
        early_stop_patience = 5

        print(f"[CrossLinkBias] Training for {epochs} epochs...")
        print(f"[CrossLinkBias] ori_lr={ori_lr}, aug_lr={aug_lr}, alpha={self.alpha}, "
              f"beta={self.beta}, threshold={self.threshold}, batch_size={self.batch_size}")
        print(f"[CrossLinkBias] ori_train_edges={ori_train_edges.shape[0]}, "
              f"aug_train_edges={aug_train_edges.shape[0]}, "
              f"msg_edge_index={msg_edge_index.shape[1]}")
        if has_val:
            print(f"[CrossLinkBias] Merge-phase inner validation enabled "
                  f"(val_pos={val_pos_edges.shape[0]}, val_neg={val_neg_edges.shape[0]})")

        # Early stopping state (matches original e2e_main.py lines 78-88)
        best_epoch_hit = 0.0
        best_epoch = 0

        for epoch in tqdm(range(1, epochs + 1)):
            # Build dataloaders each epoch (matches original)
            dataloader_ori = CLBEdgeDataloader(ori_train_edges, ori_confidence,
                                              self.batch_size, self.train_ratio)
            dataloader_aug = CLBEdgeDataloader(aug_train_edges, aug_confidence,
                                              self.batch_size, self.train_ratio)

            # ---- Phase 1: Train model_ori alone on original edges ----
            self._train_single_model(self.model_ori, dataloader_ori, ori_lr, self.num_nodes)

            # ---- Phase 2: Train model_aug alone on augmented edges ----
            self._train_single_model(self.model_aug, dataloader_aug, aug_lr, self.num_nodes)

            # ---- Phase 3: Joint merge training with dynamic curriculum ----
            LR = self.alpha * 1.0 / (1.0 + math.exp(-epoch + self.threshold))
            STEP = int(self.beta * 1.0 / (1.0 + math.exp(-epoch + self.threshold))) + 1

            optimizer_cal = torch.optim.Adam(self.model_cal.parameters(), lr=LR)
            optimizer_aug_merge = torch.optim.Adam(self.model_aug.parameters(), lr=LR)
            optimizer_ori_merge = torch.optim.Adam(self.model_ori.parameters(), lr=LR)

            # Save model_cal state before merge phase (matches original torch.save before STEP loop)
            best_merge_hit = 0.0
            best_model_cal_state = copy.deepcopy(self.model_cal.state_dict())

            # Fix 3: Create dataloader once outside STEP loop (same data, avoids recreation overhead)
            dataloader_merge = CLBEdgeDataloader(ori_train_edges, ori_confidence,
                                                self.batch_size, self.train_ratio)

            for step in range(STEP):
                self.model_cal.train()

                for data in dataloader_merge:
                    pos_edge = data[0].t().to(self.device)
                    confidence = data[1].to(self.device)  # [B]

                    emb_ori = self.model_ori(self._msg_edge_index)
                    emb_aug = self.model_aug(self._msg_edge_index)
                    embedding = self.model_cal(emb_ori, emb_aug)

                    pos_out = torch.sigmoid(
                        torch.sum(embedding[pos_edge[0]] * embedding[pos_edge[1]], dim=-1)
                    )  # [B]
                    pos_loss = (-torch.log(pos_out + 1e-15) * confidence).sum()

                    neg_edge = torch.randint(0, self.num_nodes, pos_edge.size(),
                                             dtype=torch.long, device=self.device)
                    neg_out = torch.sigmoid(
                        torch.sum(embedding[neg_edge[0]] * embedding[neg_edge[1]], dim=-1)
                    )  # [B]
                    neg_loss = (-torch.log(1 - neg_out + 1e-15) * confidence).sum()

                    loss = pos_loss + neg_loss

                    optimizer_cal.zero_grad()
                    optimizer_aug_merge.zero_grad()
                    optimizer_ori_merge.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model_cal.parameters(), 1.0)
                    nn.utils.clip_grad_norm_(self.model_aug.parameters(), 1.0)
                    nn.utils.clip_grad_norm_(self.model_ori.parameters(), 1.0)
                    optimizer_cal.step()
                    optimizer_aug_merge.step()
                    optimizer_ori_merge.step()

            # Fix 1: Validate only once at the end of the merge phase (not every step)
            # Also used for early stopping (avoids redundant eval call)
            if has_val:
                epoch_hit = self._merge_phase_eval(val_pos_edges, val_neg_edges)
                if epoch_hit > best_merge_hit:
                    best_merge_hit = epoch_hit
                    best_model_cal_state = copy.deepcopy(self.model_cal.state_dict())

            # Restore best model_cal from this epoch's merge phase
            # (matches original: outer loop reloads best model_cal from disk)
            if has_val:
                self.model_cal.load_state_dict(best_model_cal_state)

            # Early stopping (matches original e2e_main.py lines 78-88)
            if has_val:
                if epoch_hit > best_epoch_hit:
                    best_epoch_hit = epoch_hit
                    best_epoch = epoch
                if epoch > best_epoch + early_stop_patience and epoch > self.threshold - 2:
                    print(f"[CrossLinkBias] Early stopping at epoch {epoch} "
                          f"(best_epoch={best_epoch}, best_hit@50={best_epoch_hit:.4f})")
                    break

        # Store final embedding (both models use same msg edge_index)
        with torch.no_grad():
            self.eval()
            emb_ori = self.model_ori(self._msg_edge_index)
            emb_aug = self.model_aug(self._msg_edge_index)
            self.embeddings = self.model_cal(emb_ori, emb_aug).detach()
            print(f"[CrossLinkBias] Training finished. Embeddings shape: {self.embeddings.shape}")

    def predict_batch(self, edge_batch):
        if self.embeddings is None:
             raise ValueError("Model not fitted.")
        u_emb = self.embeddings[edge_batch[:, 0]]
        v_emb = self.embeddings[edge_batch[:, 1]]
        return (u_emb * v_emb).sum(dim=1)

    def get_embeddings(self):
        return self.embeddings


class SimpleGNN(nn.Module):
    """
    Simple GNN backbone for CrossLinkBias. Takes node features as input
    (adapted from the original which uses learned embeddings via nn.Embedding,
    but here we use node features since our scaffold provides them).
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, gnn_type='gcn'):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout

        if gnn_type == 'gcn':
            Conv = GCNConv
        elif gnn_type == 'sage':
            Conv = SAGEConv
        else:
            raise ValueError("Unsupported GNN type")

        self.convs.append(Conv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(Conv(hidden_channels, hidden_channels))

        if num_layers > 1:
            self.convs.append(Conv(hidden_channels, out_channels))

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs[:-1]:
            if type(conv).__name__ == 'GCNConv' and edge_weight is not None:
                x = conv(x, edge_index, edge_weight=edge_weight)
            else:
                x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if type(self.convs[-1]).__name__ == 'GCNConv' and edge_weight is not None:
            x = self.convs[-1](x, edge_index, edge_weight=edge_weight)
        else:
            x = self.convs[-1](x, edge_index)
        return x


# ==================== Link Bias Amplification (LBA) ====================

def gcn_norm_lba(edge_index, edge_weight=None, num_nodes=None, flow="source_to_target", conv="sym", dtype=None):
    from torch_geometric.utils import add_remaining_self_loops
    from torch_geometric.utils.num_nodes import maybe_num_nodes
    from torch_scatter import scatter

    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, 1.0, num_nodes)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == 'source_to_target' else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
    
    if conv == "sym":
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif conv == "rw":
        deg_inv = deg.pow_(-1.0)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = edge_weight * deg_inv[col]

    return edge_index, edge_weight

class LBA_GCNConv(nn.Module):
    """
    Custom GCNConv for Link Bias Amplification supporting 'sym' and 'rw' normalization.
    Ported from gcn_conv.py in the repository.
    """
    def __init__(self, in_channels: int, out_channels: int, conv: str = 'sym', **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = conv
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x, edge_index):
        # Normalize
        edge_index, edge_weight = gcn_norm_lba(edge_index, num_nodes=x.size(0), conv=self.conv, dtype=x.dtype)
        
        # Linear transform
        x = self.lin(x)
        
        # Message Passing (simplified sparse matmul)
        # AxW
        # We can use spmm
        adj = torch.sparse_coo_tensor(edge_index, edge_weight, (x.size(0), x.size(0)))
        out = torch.sparse.mm(adj, x)
        return out

class LinkBiasAmplification(nn.Module):
    """
    Link Bias Amplification: Regularized GCN with Fairness Penalty.
    
    Fairness Metric: Statistical Parity of Link Scores within classes.
    "Networked Inequality: Preferential Attachment Bias in Graph Neural Network Link Prediction"
    """
    def __init__(self, num_nodes, num_features, hidden_dim=128, out_dim=64, num_layers=4,
                 dropout=0.0, conv_type='sym', reg_weight=1.0, device='cpu'):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type
        self.reg_weight = reg_weight
        self.device = device
        
        self.convs = nn.ModuleList()
        # Input
        self.convs.append(LBA_GCNConv(num_features, hidden_dim, conv=conv_type))
        # Hidden
        for _ in range(num_layers - 2):
            self.convs.append(LBA_GCNConv(hidden_dim, hidden_dim, conv=conv_type))
        # Output
        self.convs.append(LBA_GCNConv(hidden_dim, out_dim, conv=conv_type))
        
        self.to(device)
        self.embeddings = None

    def encode(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index).relu()
        x = self.convs[-1](x, edge_index)
        return x

    def decode(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def fit(self, edge_index, features, sens_attr, labels=None, epochs=100, lr=0.01):
        """
        Train LBA with fairness regularization.
        """
        self.embeddings = None
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        if labels is None:
            # If no labels, assume single class (global parity)
            labels = torch.zeros(self.num_nodes, dtype=torch.long, device=self.device)
        else:
            labels = labels.to(self.device).long()
            
        features = features.to(self.device)
        sens = sens_attr.to(self.device)
        
        print(f"[LBA] Training with conv={self.conv_type}, reg={self.reg_weight}...")
        
        from torch_geometric.utils import negative_sampling, coalesce

        criterion = torch.nn.BCEWithLogitsLoss()

        for epoch in tqdm(range(epochs)):
            self.train()
            optimizer.zero_grad()

            z = self.encode(features, edge_index)

            # Negative Sampling (match original: num_neg_samples = num positive edges, method='sparse')
            neg_edge_index = negative_sampling(
                edge_index=edge_index, num_nodes=self.num_nodes,
                num_neg_samples=edge_index.size(1), method='sparse')

            # Combine positive and negative edges
            edge_label_index = torch.cat([edge_index, neg_edge_index], dim=-1)
            edge_label = torch.cat([
                torch.ones(edge_index.size(1), device=self.device),
                torch.zeros(neg_edge_index.size(1), device=self.device)
            ], dim=0)

            # Coalesce to remove duplicates (matches original)
            edge_label_index, edge_label = coalesce(edge_label_index, edge_label)

            out = self.decode(z, edge_label_index).view(-1)
            loss = criterion(out, edge_label)

            # Fairness Regularization
            fairness_loss = torch.tensor(0.0, device=self.device)
            num_classes = 0

            all_edges = edge_label_index
            all_scores = out
            
            unique_classes = torch.unique(labels)
            
            for c in unique_classes:
                c_mask = (labels == c)
                c0_mask = c_mask & (sens == 0)
                c1_mask = c_mask & (sens == 1)
                
                # Edges (u, v) where v is in class C AND u is in (Class C & Sens 0/1)
                # Wait, repo logic:
                # c0a_mask = c0_mask[edge_index[0]] & c_mask[edge_index[1]]
                # This means u in (Group 0 AND Class C) and v in (Class C)
                # So it's enforcing parity for IN-CLASS LINKS score distribution?
                
                # Masks for source nodes of edges in the batch
                src_c0 = c0_mask[all_edges[0]]
                src_c1 = c1_mask[all_edges[0]]
                
                # Mask for target nodes of edges in the batch
                dst_c = c_mask[all_edges[1]]
                
                mask_0 = src_c0 & dst_c
                mask_1 = src_c1 & dst_c
                
                if mask_0.sum() == 0 or mask_1.sum() == 0:
                    continue
                
                avg_0 = torch.sigmoid(all_scores[mask_0]).mean()
                avg_1 = torch.sigmoid(all_scores[mask_1]).mean()
                
                fairness_loss += torch.abs(avg_0 - avg_1)
                num_classes += 1
            
            if num_classes > 0 and self.reg_weight > 0:
                loss += self.reg_weight * (fairness_loss / num_classes)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                pass # print(f"Epoch {epoch+1} Loss: {loss.item()}")
                
        # Store embeddings
        with torch.no_grad():
            self.eval()
            self.embeddings = self.encode(features, edge_index).detach()
            print(f"[LBA] Training finished. Embeddings shape: {self.embeddings.shape}")

    def predict_batch(self, edge_batch):
        if self.embeddings is None:
             raise ValueError("Model not fitted.")
        u_emb = self.embeddings[edge_batch[:, 0]]
        v_emb = self.embeddings[edge_batch[:, 1]]
        return (u_emb * v_emb).sum(dim=1)

    def get_embeddings(self):
        if self.embeddings is None:
            raise ValueError("Model not fitted.")
        return self.embeddings


# ==================== FairAdj ====================

class FairAdj_GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(FairAdj_GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        if adj.is_sparse:
             output = torch.sparse.mm(adj, support)
        else:
             output = torch.mm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class FairAdj_InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""
    def __init__(self, dropout, act=torch.sigmoid):
        super(FairAdj_InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


class FairAdj_GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(FairAdj_GCNModelVAE, self).__init__()
        self.gc1 = FairAdj_GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = FairAdj_GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = FairAdj_GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = FairAdj_InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(torch.clamp(logvar, min=-10, max=10))
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), z, mu, logvar


def fairadj_loss_function(z, pos_edge_index, mu, logvar, n_nodes):
    from torch_geometric.utils import negative_sampling
    num_pos = pos_edge_index.size(1)
    neg_edge_index = negative_sampling(pos_edge_index, num_nodes=n_nodes, num_neg_samples=num_pos)
    
    pos_scores = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)
    neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)
    
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
    
    cost = F.binary_cross_entropy_with_logits(scores, labels)

    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    logvar_clamped = torch.clamp(logvar, min=-10, max=10)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar_clamped - mu.pow(2) - logvar_clamped.exp().pow(2), 1))

    return cost + KLD


def fairadj_project_sparse(indices, values, num_rows, device):
    """
    Project each row's sparse values onto the probability simplex.

    indices: [2, nnz] - row and column indices of sparse tensor
    values: [nnz] - values at those positions
    num_rows: number of rows
    device: torch device

    Returns: projected values [nnz] with same ordering as input
    """
    row_indices = indices[0]
    projected = values.clone()

    for i in range(num_rows):
        mask = (row_indices == i)
        if mask.sum() == 0:
            continue

        row_vals = values[mask]

        # Simplex projection (same algorithm as original FairAdj project())
        des, sort_idx = torch.sort(row_vals, descending=True)
        cumsum = torch.cumsum(des, dim=0)
        k = torch.arange(1, des.shape[0] + 1, device=device).float()

        rho_cond = des + (1.0 / k) * (1.0 - cumsum)
        rho = (rho_cond > 0).sum()
        lambda_ = (1.0 / rho.float()) * (1.0 - cumsum[rho - 1])

        result = torch.clamp(row_vals + lambda_, min=0.0)
        projected[mask] = result

    return projected


class FairAdj(nn.Module):
    """
    FairAdj Baseline with Efficiency Optimizations
    """
    def __init__(self, num_nodes, num_features, hidden_dim1=32, hidden_dim2=16, dropout=0.,
                 T1=50, T2=20, eta=0.2, lr=0.01, outer_epochs=4, eq=False, device='cpu'):
        super(FairAdj, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.dropout = dropout
        self.T1 = T1
        self.T2 = T2
        self.eta = eta
        self.lr = lr
        self.outer_epochs = outer_epochs
        self.eq = eq
        self.device = device
        
        self.model = FairAdj_GCNModelVAE(num_features, hidden_dim1, hidden_dim2, dropout).to(device)
        self.embeddings = None
        
    def fit(self, edge_index, features, sens_attr, epochs=None):
        """
        Fit FairAdj. 
        Note: 'epochs' arg is ignored/overridden by T1/T2 to match repo behavior.
        """
        from torch_geometric.utils import to_dense_adj
        
        features = features.to(self.device)
        sens_attr = sens_attr.to(self.device)
        
        # 1. Construct Adj
        adj_sp = torch.sparse_coo_tensor(
            edge_index, 
            torch.ones(edge_index.shape[1], device=self.device), 
            (self.num_nodes, self.num_nodes)
        ).coalesce()
        
        # A + I with sparse operations
        idx = adj_sp.indices()
        val = adj_sp.values()
        loop_idx = torch.arange(self.num_nodes, device=self.device).repeat(2, 1)
        loop_val = torch.ones(self.num_nodes, device=self.device)
        
        adj_eye_idx = torch.cat([idx, loop_idx], dim=1)
        adj_eye_val = torch.cat([val, loop_val], dim=0)
        adj_eye_sp = torch.sparse_coo_tensor(
            adj_eye_idx, adj_eye_val, (self.num_nodes, self.num_nodes)
        ).coalesce()
        
        rowsum = torch.sparse.sum(adj_eye_sp, dim=1).to_dense()
        d_inv = torch.pow(rowsum, -1.0)
        d_inv[d_inv == float('inf')] = 0.
        
        norm_val = adj_eye_sp.values() * d_inv[adj_eye_sp.indices()[0]]
        adj_norm_sp = torch.sparse_coo_tensor(
            adj_eye_sp.indices(), norm_val, (self.num_nodes, self.num_nodes)
        ).coalesce()
        
        adj_norm_v = adj_norm_sp.clone()  # Always sparse!
        pos_edge_index = adj_eye_sp.indices().clone()
        
        # Free intermediates
        del adj_sp, idx, val, loop_idx, loop_val, adj_eye_idx, adj_eye_val, adj_eye_sp, rowsum, d_inv, norm_val

        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # ========== Precompute intra/inter masks for T2 ==========
        if self.eq:
            # --eq mode: use only existing edges (for Oklahoma/UNC)
            row, col = edge_index
            sens_row = sens_attr[row]
            sens_col = sens_attr[col]
            mask_intra_edge = (sens_row == sens_col)
            mask_inter_edge = (sens_row != sens_col)
            intra_idx = edge_index[:, mask_intra_edge]
            inter_idx = edge_index[:, mask_inter_edge]
        else:
            # Default mode: use ALL node pairs (matches original find_link)
            # Sample to avoid N^2 OOM
            num_samples = self.num_nodes * 10
            row = torch.randint(0, self.num_nodes, (num_samples,), device=self.device)
            col = torch.randint(0, self.num_nodes, (num_samples,), device=self.device)
            sens_row = sens_attr[row]
            sens_col = sens_attr[col]
            intra_idx = torch.stack([row[sens_row == sens_col], col[sens_row == sens_col]])
            inter_idx = torch.stack([row[sens_row != sens_col], col[sens_row != sens_col]])

        # ========== Outer loop: alternate T1 <-> T2 ==========
        self.model.train()
        for outer in range(self.outer_epochs):
            # ===== Phase T1: Train VGAE =====
            for epoch in tqdm(range(self.T1), desc=f"T1 outer={outer+1}"):
                optimizer.zero_grad()
                # Encode only — skip decoder to avoid N×N inner product matrix
                mu, logvar = self.model.encode(features, adj_norm_sp)
                z = self.model.reparameterize(mu, logvar)
                loss = fairadj_loss_function(z, pos_edge_index, mu, logvar, self.num_nodes)
                loss.backward()
                optimizer.step()

            # ===== Phase T2: Optimize Adj for Fairness =====
            adj_norm_v = adj_norm_sp.detach().clone().requires_grad_(True)

            for epoch in tqdm(range(self.T2), desc=f"T2 outer={outer+1}"):
                # GC1: relu(adj @ (dropout(x) @ w1))
                hidden1 = F.relu(torch.sparse.mm(adj_norm_v, torch.mm(
                    F.dropout(features, self.dropout, training=self.model.training),
                    self.model.gc1.weight)))

                # GC2 (mu): adj @ (hidden1 @ w2)
                mu = torch.sparse.mm(adj_norm_v, torch.mm(hidden1, self.model.gc2.weight))

                # GC3 (logvar): adj @ (hidden1 @ w3)
                logvar = torch.sparse.mm(adj_norm_v, torch.mm(hidden1, self.model.gc3.weight))

                # Reparameterize
                if self.model.training:
                    std = torch.exp(torch.clamp(logvar, min=-10, max=10))
                    eps = torch.randn_like(std)
                    z = eps.mul(std).add_(mu)
                else:
                    z = mu

                # Decode: raw logits, NO sigmoid (matches original dc with act=identity)
                z_do = F.dropout(z, self.dropout, training=self.model.training)

                # Fairness loss: MSE(intra_mean, inter_mean)
                intra_scores = (z_do[intra_idx[0]] * z_do[intra_idx[1]]).sum(dim=-1)
                inter_scores = (z_do[inter_idx[0]] * z_do[inter_idx[1]]).sum(dim=-1)

                if len(intra_scores) == 0 or len(inter_scores) == 0:
                    loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                else:
                    loss = F.mse_loss(intra_scores.mean(), inter_scores.mean())

                # Gradient update on adj_norm_v (sparse-only, no dense N×N)
                grad = torch.autograd.grad(loss, adj_norm_v)[0]
                with torch.no_grad():
                    adj_c = adj_norm_v.coalesce()
                    grad_c = grad.coalesce()
                    indices = adj_c.indices()
                    new_values = adj_c.values() - self.eta * grad_c.values()
                    new_values = fairadj_project_sparse(indices, new_values, self.num_nodes, self.device)
                    adj_norm_sp = torch.sparse_coo_tensor(indices, new_values, (self.num_nodes, self.num_nodes)).coalesce()
                    adj_norm_v = adj_norm_sp.clone().requires_grad_(True)
            
        # Evaluation / Final Embedding
        self.model.eval()
        
        # Use optimized adj for final embedding
        with torch.no_grad():
             mu, _ = self.model.encode(features, adj_norm_sp)
             self.embeddings = mu.detach()
        
        # print(f"[FairAdj] Training finished. Embeddings shape: {self.embeddings.shape}")

    def predict_batch(self, edge_batch):
        if self.embeddings is None:
             raise ValueError("Model not fitted.")
        u_emb = self.embeddings[edge_batch[:, 0]]
        v_emb = self.embeddings[edge_batch[:, 1]]
        return (u_emb * v_emb).sum(dim=1)

    def get_embeddings(self):
        return self.embeddings


# ==================== FairLP ====================

class FairLP_EdgeModifier:
    """
    FairLP: Pre-processing bias mitigation by density equating.
    """
    def __init__(self, num_nodes, density_mode='AvgD'):
        """
        density_mode: 'AvgD', 'MinD', 'MaxD'
        """
        self.num_nodes = num_nodes
        self.density_mode = density_mode

    def modify_edges(self, edge_index, sens_attr):
        """
        edge_index: [2, E]
        sens_attr: [N] binary (0 or 1)
        Returns: modified_edge_index: [2, E']
        """
        import networkx as nx
        device = edge_index.device
        edge_index_cpu = edge_index.cpu()
        sens = sens_attr.cpu().numpy() if torch.is_tensor(sens_attr) else np.array(sens_attr)
        
        # Binary sensitive groups
        group0_nodes = np.where(sens == 0)[0]
        group1_nodes = np.where(sens == 1)[0]
        
        k0 = len(group0_nodes)
        k1 = len(group1_nodes)
        
        # Intra-group (un-protected edge group) possible edges
        cmax_intra = k0 * (k0 - 1) // 2 + k1 * (k1 - 1) // 2
        # Inter-group (protected edge group) possible edges
        cmax_inter = k0 * k1
        
        # Treat network as undirected for simplicity of density matching
        G = nx.Graph()
        G.add_nodes_from(range(self.num_nodes))
        if edge_index_cpu.shape[1] == 2 and edge_index_cpu.shape[0] != 2:
            edges = edge_index_cpu.tolist()
        else:
            edges = edge_index_cpu.t().tolist()
        G.add_edges_from(edges)
        
        # Actual edges
        real_edges = list(G.edges())
        creal_intra = 0
        creal_inter = 0
        intra_edges = []
        inter_edges = []
        for u, v in real_edges:
            if sens[u] == sens[v]:
                creal_intra += 1
                intra_edges.append((u, v))
            else:
                creal_inter += 1
                inter_edges.append((u, v))
                
        density_intra = creal_intra / max(1, cmax_intra)
        density_inter = creal_inter / max(1, cmax_inter)
        
        if self.density_mode == 'AvgD':
            r = (density_intra + density_inter) / 2.0
        elif self.density_mode == 'MinD':
            r = min(density_intra, density_inter)
        elif self.density_mode == 'MaxD':
            r = max(density_intra, density_inter)
        else:
            raise ValueError(f"Unknown density_mode: {self.density_mode}")
            
        # Target edges
        cexp_intra = int(round(cmax_intra * r))
        cexp_inter = int(round(cmax_inter * r))
        
        k_intra = cexp_intra - creal_intra
        k_inter = cexp_inter - creal_inter
        
        print(f"[FairLP] density_mode={self.density_mode}. Target density={r:.4f}")
        print(f"[FairLP] Intra: real={creal_intra}, max={cmax_intra}, target={cexp_intra}, diff={k_intra}")
        print(f"[FairLP] Inter: real={creal_inter}, max={cmax_inter}, target={cexp_inter}, diff={k_inter}")
        
        # Process a group: returns updated G
        def adjust_edges(G_current, is_intra, k):
            if k == 0:
                print(f"[FairLP] Group is_intra={is_intra}: k=0, no changes.")
                return G_current
            
            if k < 0:
                # Remove |k| edges with LOWEST weight from the group
                num_to_remove = abs(k)
                
                # Get all edges in group
                candidate_edges = []
                for u, v in G_current.edges():
                    if (sens[u] == sens[v]) == is_intra:
                        candidate_edges.append((u, v))
                
                if num_to_remove > len(candidate_edges):
                    print(f"[FairLP] Warning: cannot remove {num_to_remove} edges, only {len(candidate_edges)} available.")
                    num_to_remove = len(candidate_edges)
                
                print(f"[FairLP] Removing {num_to_remove} edges from is_intra={is_intra}...")
                
                # Calculate weights once
                weights = []
                for u, v in candidate_edges:
                    w = len(list(nx.common_neighbors(G_current, u, v)))
                    weights.append((w, u, v))
                    
                weights.sort() # sort by weight ascending
                edges_to_remove = [(u, v) for w, u, v in weights[:num_to_remove]]
                G_current.remove_edges_from(edges_to_remove)
                
                return G_current
            
            else:
                # k > 0, Insert k edges with LOWEST weight into the group
                num_to_add = k
                print(f"[FairLP] Adding {num_to_add} edges to is_intra={is_intra}...")
                
                added = 0
                max_attempts = num_to_add * 200
                attempts = 0
                
                # For adding lowest weight edges, any random disconnected pair of the correct group
                # typically has 0 common neighbors in a sparse graph.
                new_edges = []
                while added < num_to_add and attempts < max_attempts:
                    attempts += 1
                    u = np.random.randint(0, self.num_nodes)
                    v = np.random.randint(0, self.num_nodes)
                    if u == v or G_current.has_edge(u, v):
                        continue
                    if (sens[u] == sens[v]) == is_intra:
                        w = len(list(nx.common_neighbors(G_current, u, v)))
                        if w == 0: # Lowest possible weight
                            new_edges.append((u, v))
                            G_current.add_edge(u, v)
                            added += 1
                
                if added < num_to_add:
                    print(f"[FairLP] Warning: Could only add {added} edges out of {num_to_add} after max attempts.")
                
                return G_current

        G = adjust_edges(G, True, k_intra)
        G = adjust_edges(G, False, k_inter)
        
        # Convert back to symmetric edge_index
        new_edge_list = list(G.edges())
        if len(new_edge_list) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)
            
        edge_tensor = torch.tensor(new_edge_list, dtype=torch.long).t()
        # Make symmetric
        modified_edge_index = torch.cat([edge_tensor, torch.stack([edge_tensor[1], edge_tensor[0]])], dim=1)
        
        return modified_edge_index.to(device)


class FairLP_GNN(nn.Module):
    """
    FairLP Baseline wrapper: Applies FairLP edge modification and then trains a SimpleGNN.
    """
    def __init__(self, num_nodes, num_features, hidden_dim=64, out_dim=64, num_layers=2, dropout=0.5,
                 density_mode='AvgD', gnn_type='gcn', device='cpu'):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.device = device
        self.edge_modifier = FairLP_EdgeModifier(num_nodes, density_mode=density_mode)
        self.gnn = SimpleGNN(num_features, hidden_dim, out_dim, num_layers, dropout, gnn_type=gnn_type).to(device)
        self.embeddings = None

    def fit(self, edge_index, features, sens_attr, epochs=100, lr=0.01):
        """
        Train the model.
        """
        self.embeddings = None
        
        # 1. Pre-processing
        modified_edge_index = self.edge_modifier.modify_edges(edge_index, sens_attr)
        
        # 2. Train GNN
        optimizer = torch.optim.Adam(self.gnn.parameters(), lr=lr)
        features = features.to(self.device)
        modified_edge_index = modified_edge_index.to(self.device)
        
        print(f"[FairLP_GNN] Training GNN on modified graph for {epochs} epochs...")
        from torch_geometric.utils import negative_sampling
        
        for epoch in tqdm(range(epochs)):
            self.gnn.train()
            optimizer.zero_grad()
            
            z = self.gnn(features, modified_edge_index)
            
            # Simple BCE Loss for Link Prediction
            pos_batch = modified_edge_index
            neg_batch = negative_sampling(modified_edge_index, num_nodes=self.num_nodes)
            
            # Protect against empty edge bounds
            if pos_batch.size(1) == 0:
                pos_score = torch.tensor([], device=self.device)
            else:
                pos_score = (z[pos_batch[0]] * z[pos_batch[1]]).sum(dim=1)
                
            if neg_batch.size(1) == 0:
                neg_score = torch.tensor([], device=self.device)
            else:
                neg_score = (z[neg_batch[0]] * z[neg_batch[1]]).sum(dim=1)
            
            # If no positive or no negative edges, something's very wrong, default to zero loss
            if len(pos_score) > 0 and len(neg_score) > 0:
                loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean() - \
                       torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()
                       
                loss.backward()
                optimizer.step()
                
        # Store embeddings based on the modified graph! 
        self.gnn.eval()
        with torch.no_grad():
            self.embeddings = self.gnn(features, modified_edge_index).detach()
            print(f"[FairLP_GNN] Training finished. Embeddings shape: {self.embeddings.shape}")

    def predict_batch(self, edge_batch):
        if self.embeddings is None:
             raise ValueError("Model not fitted.")
        u_emb = self.embeddings[edge_batch[:, 0]]
        v_emb = self.embeddings[edge_batch[:, 1]]
        return (u_emb * v_emb).sum(dim=1)

    def get_embeddings(self):
        return self.embeddings


# ==================== FairLink ====================

class FairLinkModel(nn.Module):
    """
    FairLink: Data-centric fairness enhancement via Gradient Matching.
    Learns a synthetic graph (X_f, A_f) where A_f is parameterized by an MLP \psi on X_f.
    """
    def __init__(self, num_nodes, num_features, hidden_dim=64, out_dim=64, num_layers=2, dropout=0.5,
                 alpha=1.0, beta=1e-4, gamma=1.0, gnn_type='gcn', device='cpu'):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.device = device
        
        feature_dim = num_features if num_features > 0 else num_nodes
        
        # X_f (Synthetic Node Features)
        self.x_f = nn.Parameter(torch.randn(num_nodes, feature_dim))
        nn.init.xavier_uniform_(self.x_f)
        
        # \psi (Adjacency Matrix Generator)
        self.psi = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Base predictor for Utility tracking (Gradient matching)
        self.base_gnn = SimpleGNN(feature_dim, hidden_dim, out_dim, num_layers, dropout, gnn_type=gnn_type).to(device)
        # Final predictor trained on the generated graph
        self.base_gnn_final = SimpleGNN(feature_dim, hidden_dim, out_dim, num_layers, dropout, gnn_type=gnn_type).to(device)
        self.embeddings = None

    def _get_af_scores(self, u_idx, v_idx):
        """Compute structural connection probability A_f(u, v) using \psi."""
        xu = self.x_f[u_idx]
        xv = self.x_f[v_idx]
        pairs = torch.cat([xu, xv], dim=-1)
        return self.psi(pairs).squeeze(-1)

    def fit(self, edge_index, features, sens_attr, epochs=100, lr=0.01):
        """Train X_f and \psi via alternating gradient matching, then train final downstream GNN."""
        print(f"[FairLink] Starting alternating optimization for {epochs} epochs...")
        
        opt_x = torch.optim.Adam([self.x_f], lr=lr, weight_decay=self.beta)
        opt_psi = torch.optim.Adam(self.psi.parameters(), lr=lr, weight_decay=self.beta)
        
        features = features.to(self.device).float()
        if features.dim() == 2 and features.size(1) == self.num_features:
            pass # Provided correctly
        elif self.num_features == 0:
            features = torch.eye(self.num_nodes, device=self.device)
            
        edge_index = edge_index.to(self.device)
        if edge_index.size(0) != 2 and edge_index.size(1) == 2:
            edge_index = edge_index.t()
            
        sens_attr = sens_attr.to(self.device)
        
        from torch_geometric.utils import negative_sampling
        import torch.nn.functional as F

        tau_x = 5
        tau_psi = 5
        
        for epoch in tqdm(range(epochs)):
            is_x_step = (epoch // tau_x) % 2 == 0
            
            optimizer = opt_x if is_x_step else opt_psi
            optimizer.zero_grad()
            
            # Sample edges for utility loss evaluation
            # To ensure memory efficiency, we use a batch instead of full graph
            perm = torch.randperm(edge_index.size(1))[:4096] # Batch size
            pos_batch = edge_index[:, perm]
            neg_batch = negative_sampling(edge_index, num_nodes=self.num_nodes, num_neg_samples=pos_batch.size(1))
            
            # Support structure: use true edges (can be sampled if too large, but PyG handles full easily)
            # For gradients, evaluating on full edge_index might be heavy. 
            # We use a subgraph or the full graph if small.
            support_edges = edge_index
            
            # 1. Gradient of G (Input Graph)
            # We must freeze base_gnn. We just compute gradients on it dynamically.
            theta = list(self.base_gnn.parameters())
            
            z_g = self.base_gnn(features, support_edges)
            pos_score_g = (z_g[pos_batch[0]] * z_g[pos_batch[1]]).sum(dim=1)
            neg_score_g = (z_g[neg_batch[0]] * z_g[neg_batch[1]]).sum(dim=1)
            loss_g = -torch.log(torch.sigmoid(pos_score_g) + 1e-15).mean() - torch.log(1 - torch.sigmoid(neg_score_g) + 1e-15).mean()
            
            grad_g = torch.autograd.grad(loss_g, theta, create_graph=False)
            
            # 2. Gradient of G_f (Synthetic Graph)
            # Compute dynamic weights A_f
            weights_f = self._get_af_scores(support_edges[0], support_edges[1])
            
            z_f = self.base_gnn(self.x_f, support_edges, edge_weight=weights_f)
            pos_score_f = (z_f[pos_batch[0]] * z_f[pos_batch[1]]).sum(dim=1)
            neg_score_f = (z_f[neg_batch[0]] * z_f[neg_batch[1]]).sum(dim=1)
            loss_f = -torch.log(torch.sigmoid(pos_score_f) + 1e-15).mean() - torch.log(1 - torch.sigmoid(neg_score_f) + 1e-15).mean()
            
            grad_f = torch.autograd.grad(loss_f, theta, create_graph=True)
            
            # 3. Utility Loss D(grad_g, grad_f)
            d_cos = 0.0
            d_euc = 0.0
            for g1, g2 in zip(grad_g, grad_f):
                g1_flat = g1.flatten()
                g2_flat = g2.flatten()
                # add eps to avoid divide by zero
                cos_sim = F.cosine_similarity(g1_flat, g2_flat, dim=0, eps=1e-8)
                d_cos += (1.0 - cos_sim)
                d_euc += F.pairwise_distance(g1_flat.unsqueeze(0), g2_flat.unsqueeze(0)).sum()
                
            loss_util = d_cos + self.gamma * d_euc
            
            # 4. Fairness Loss L_fair
            # Estimate over a random sample of pairs for efficiency
            idx_u = torch.randint(0, self.num_nodes, (1024,), device=self.device)
            idx_v = torch.randint(0, self.num_nodes, (1024,), device=self.device)
            
            su = sens_attr[idx_u]
            sv = sens_attr[idx_v]
            
            scores_fair = torch.sigmoid((z_f[idx_u] * z_f[idx_v]).sum(dim=1))
            mask_intra = (su == sv)
            mask_inter = (su != sv)
            
            p_intra = scores_fair[mask_intra].mean() if mask_intra.sum() > 0 else torch.tensor(0.0, device=self.device)
            p_inter = scores_fair[mask_inter].mean() if mask_inter.sum() > 0 else torch.tensor(0.0, device=self.device)
            
            loss_fair = torch.abs(p_intra - p_inter)
            
            # Total Loss
            total_loss = loss_util + self.alpha * loss_fair
            
            total_loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"  [FairLink Epoch {epoch}] L_util: {loss_util.item():.4f}, L_fair: {loss_fair.item():.4f}")

        # Train final downstream GNN using the learned X_f and \psi
        print(f"[FairLink] Pre-processing complete. Training final link predictor.")
        self.base_gnn_final.train()
        opt_final = torch.optim.Adam(self.base_gnn_final.parameters(), lr=lr)
        
        for epoch in tqdm(range(epochs)):
            opt_final.zero_grad()
            
            # Extract final graph
            with torch.no_grad():
                weights_f_final = self._get_af_scores(support_edges[0], support_edges[1]).detach()
                x_f_final = self.x_f.detach()
            
            z = self.base_gnn_final(x_f_final, support_edges, edge_weight=weights_f_final)
            
            perm = torch.randperm(edge_index.size(1))[:4096]
            pos_batch = edge_index[:, perm]
            neg_batch = negative_sampling(edge_index, num_nodes=self.num_nodes, num_neg_samples=pos_batch.size(1))
            
            pos_score = (z[pos_batch[0]] * z[pos_batch[1]]).sum(dim=1)
            neg_score = (z[neg_batch[0]] * z[neg_batch[1]]).sum(dim=1)
            loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean() - torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()
            
            loss.backward()
            opt_final.step()

        self.base_gnn_final.eval()
        with torch.no_grad():
            weights_f_final = self._get_af_scores(support_edges[0], support_edges[1]).detach()
            x_f_final = self.x_f.detach()
            self.embeddings = self.base_gnn_final(x_f_final, support_edges, edge_weight=weights_f_final).detach()
            print(f"[FairLink] Final embeddings shape: {self.embeddings.shape}")
            self.final_support_edges = support_edges
            self.final_edge_weights = weights_f_final
            self.final_x_f = x_f_final

    def predict_batch(self, edge_batch, h=None):
        if self.embeddings is None:
             raise ValueError("Model not fitted.")
        if h is None:
            h = self.embeddings
        u_emb = h[edge_batch[:, 0]]
        v_emb = h[edge_batch[:, 1]]
        return (u_emb * v_emb).sum(dim=1)

    def get_embeddings(self):
        return self.embeddings
