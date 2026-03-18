import os
import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split
from torch_geometric.utils import negative_sampling
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


class SNAPDataset(InMemoryDataset):
    """
    Dataset loader for SNAP datasets.
    Supports: ego-facebook, ego-twitter, soc-pokec, wiki-vote, lastfm_asia
    
    Expected directory structure:
    SNAP/
        ego-facebook/
            processed/
                data.pt
        ego-twitter/
            processed/
                data.pt
        soc-pokec/
            processed/
                data.pt
        wiki-vote/
            raw/
                Wiki-Vote.txt
    """
    
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        super(SNAPDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self._reindex_contiguous_in_place()

    @property
    def graph(self):
        """Get the single graph in this dataset."""
        return self.get(0)

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        if self.name == 'wiki-vote':
            return ['Wiki-Vote.txt']
        elif self.name == 'lastfm_asia':
            return ['lastfm_asia_edges.csv', 'lastfm_asia_features.json', 'lastfm_asia_target.csv']
        else:
            return ['data.pt']  # For preprocessed datasets

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # SNAP datasets should be manually downloaded
        pass
    def _reindex_contiguous_in_place(self):
        """
        Load processed/data.pt, reindex node ids to be contiguous (0..N-1),
        preserve node features x if present, make graph undirected,
        and overwrite processed/data.pt with the cleaned version.
        Safe to call repeatedly (idempotent).
        """
        processed_file = self.processed_paths[0]
        if not os.path.exists(processed_file):
            raise FileNotFoundError(f"Expected processed file at {processed_file}.")

        data_collated, slices = torch.load(processed_file)

        # Single-graph InMemoryDataset: data_collated is a Data object
        data = data_collated
        edge_index = data.edge_index

        # Build mapping old_id -> new_id (contiguous)
        flat = edge_index.view(-1)
        unique_nodes, inv = torch.unique(flat, sorted=True, return_inverse=True)
        edge_index_re = inv.view(2, -1)
        num_nodes_re = unique_nodes.numel()

        # Remap x if present (keeps only nodes that appear at least once)
        if getattr(data, "x", None) is not None:
            x_re = data.x[unique_nodes]
        else:
            x_re = None

        # Make undirected (idempotent if already undirected)
        edge_index_re = to_undirected(edge_index_re, num_nodes=num_nodes_re)

        # Rebuild a clean Data
        clean = Data(edge_index=edge_index_re, num_nodes=num_nodes_re)
        if x_re is not None:
            clean.x = x_re

        # Preserve and properly reindex node-level attributes
        for attr in ["y", "sens", "train_mask", "val_mask", "test_mask"]:
            val = getattr(data, attr, None)
            if val is not None and torch.is_tensor(val) and val.dim() >= 1:
                if val.shape[0] > num_nodes_re:
                    # Attribute was NOT reindexed in a previous run — reindex now
                    setattr(clean, attr, val[unique_nodes])
                elif val.shape[0] == num_nodes_re:
                    # Already correct size
                    setattr(clean, attr, val)

        # Collate and overwrite processed/data.pt
        data_out, slices_out = self.collate([clean])
        torch.save((data_out, slices_out), processed_file)

        # Update the in-memory copy too so the rest of your code sees the fix
        self.data, self.slices = data_out, slices_out

        # Quick log
        mx = int(edge_index_re.max().item()) if edge_index_re.numel() > 0 else -1
        print(f"[SNAPDataset] Reindexed {self.name}: num_nodes={num_nodes_re}, "
            f"max_id={mx}, num_edges={edge_index_re.size(1)}")
    
    def process(self):
        """Process the raw data files"""
        
        # Check if already processed
        if os.path.exists(self.processed_paths[0]):
            print(f"Loading existing processed data from {self.processed_paths[0]}")
            return

        
            
        if self.name == 'wiki-vote' or self.name == 'ego-facebook':
            # Process from raw txt file
            filename = 'Wiki-Vote.txt' if self.name == 'wiki-vote' else 'facebook_combined.txt'
            edge_list = []
            with open(os.path.join(self.raw_dir, filename), 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        src, dst = parts[:2]
                        edge_list.append([int(src), int(dst)])
            
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            
            # Remap node indices to be contiguous
            unique_nodes = torch.unique(edge_index)
            node_map = {old_id.item(): new_id for new_id, old_id in enumerate(unique_nodes)}
            edge_index = torch.tensor([
                [node_map[src.item()] for src in edge_index[0]],
                [node_map[dst.item()] for dst in edge_index[1]]
            ], dtype=torch.long)
            
            num_nodes = len(unique_nodes)
            
        elif self.name == 'lastfm_asia':
            import json
            import pandas as pd
            
            # 1. Load Edges
            edges_path = os.path.join(self.raw_dir, 'lastfm_asia_edges.csv')
            edges_df = pd.read_csv(edges_path)
            edge_index = torch.tensor([edges_df['node_1'].values, edges_df['node_2'].values], dtype=torch.long)
            
            # 2. Load Features
            features_path = os.path.join(self.raw_dir, 'lastfm_asia_features.json')
            with open(features_path, 'r') as f:
                features_dict = json.load(f)
            
            # Determine dimensions
            num_nodes = len(features_dict) # Based on observations, 7624
            # Find max feature index for dimension size (sparse -> dense multi-hot)
            max_feat_idx = 0
            for n_id, feats in features_dict.items():
                if feats:
                    max_feat_idx = max(max_feat_idx, max(feats))
            num_features = max_feat_idx + 1
            
            # Create multi-hot features
            x = torch.zeros(num_nodes, num_features, dtype=torch.float)
            for n_id, feats in features_dict.items():
                node_idx = int(n_id)
                if feats:
                    x[node_idx, feats] = 1.0
            
            # 3. Load Targets (if needed for node classification later, or just attributes)
            # lastfm_asia_target.csv has id,target
            target_path = os.path.join(self.raw_dir, 'lastfm_asia_target.csv')
            target_df = pd.read_csv(target_path)
            # Ensure sorting by ID just in case
            target_df = target_df.sort_values('id')
            y = torch.tensor(target_df['target'].values, dtype=torch.long)
            
            # Create Data object
            data = Data(edge_index=edge_index, x=x, y=y, num_nodes=num_nodes)
            
            # Make undirected
            data.edge_index = to_undirected(data.edge_index)
            
            # Save
            data_list = [data]
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            print(f"Processed data saved to {self.processed_paths[0]}")
            return

        else:
            # For ego-facebook, ego-twitter, soc-pokec
            # These should already be processed and have data.pt in processed folder
            processed_file = os.path.join(self.processed_dir, 'data.pt')
            if os.path.exists(processed_file):
                print(f"Data already processed at {processed_file}")
                return
            else:
                raise FileNotFoundError(
                    f"Processed data not found at {processed_file}. "
                    f"Please ensure data.pt exists in {self.processed_dir}"
                )
        
        # Make edges undirected
        edge_index = to_undirected(edge_index)
        
        # Create data object
        data = Data(edge_index=edge_index, num_nodes=num_nodes)
        
        # Save processed data
        data_list = [data]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Processed data saved to {self.processed_paths[0]}")

    def get_edge_split_old(self, val_ratio=0.05, test_ratio=0.1, seed=42):
        """
        Split edges into train/validation/test sets
        
        Args:
            val_ratio: Ratio of edges for validation
            test_ratio: Ratio of edges for testing
            seed: Random seed for reproducibility
        
        Returns:
            Dictionary with train/valid/test splits
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        data = self.data
        edge_index = data.edge_index
        
        # Remove duplicate edges and convert to undirected
        edge_index = to_undirected(edge_index)
        
        # Convert to numpy for splitting
        num_edges = edge_index.size(1) // 2  # Divide by 2 for undirected
        
        # Get unique edges (only one direction)
        edges = edge_index.t().numpy()
        mask = edges[:, 0] < edges[:, 1]
        edges = edges[mask]
        
        # Split edges
        num_val = int(num_edges * val_ratio)
        num_test = int(num_edges * test_ratio)
        num_train = num_edges - num_val - num_test
        
        # Random permutation
        perm = np.random.permutation(num_edges)
        train_edges = edges[perm[:num_train]]
        val_edges = edges[perm[num_train:num_train + num_val]]
        test_edges = edges[perm[num_train + num_val:]]
        
        # Generate negative edges for validation and test
        num_nodes = data.num_nodes
        
        def generate_neg_edges(pos_edges, num_neg_per_edge=1):
            neg_edges = []
            for i in range(len(pos_edges)):
                src = pos_edges[i, 0]
                for _ in range(num_neg_per_edge):
                    dst = np.random.randint(0, num_nodes)
                    # Simple negative sampling (may include false negatives)
                    neg_edges.append([src, dst])
            return np.array(neg_edges)
        
        val_neg_edges = generate_neg_edges(val_edges, num_neg_per_edge=1)
        test_neg_edges = generate_neg_edges(test_edges, num_neg_per_edge=1)

        train_edge_index = to_undirected(torch.tensor(train_edges, dtype=torch.long).t())
        train_data = Data(edge_index=train_edge_index, num_nodes=self.data.num_nodes)
    
        split_edge = {
            'train': {
                'edge': torch.tensor(train_edges, dtype=torch.long),
            },
            'valid': {
                'edge': torch.tensor(val_edges, dtype=torch.long),
                'edge_neg': torch.tensor(val_neg_edges, dtype=torch.long),
            },
            'test': {
                'edge': torch.tensor(test_edges, dtype=torch.long),
                'edge_neg': torch.tensor(test_neg_edges, dtype=torch.long),
            },
            'train_graph': train_data
        }
        
        return split_edge
    

    def get_edge_split(self, val_ratio=0.05, test_ratio=0.1, seed=42):
        """
        Split edges into train/validation/test sets and provides a training-only graph.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        data = self.data
        edge_index = data.edge_index
        
        # 1. Get unique undirected edges FIRST
        edges = to_undirected(edge_index).t().numpy()
        mask = edges[:, 0] < edges[:, 1]
        edges = edges[mask]
        
        # 2. Calculate num_edges from the TRUE number of unique edges
        num_edges = edges.shape[0]

        # --- CONNECTIVITY PRESERVATION ---
        # Compute a spanning tree/forest so these edges stay in the training set.
        # This guarantees the training graph preserves the original connectivity.
        row_idx = edges[:, 0]
        col_idx = edges[:, 1]
        weights = np.ones(len(edges))
        adj = csr_matrix(
            (np.concatenate([weights, weights]),
             (np.concatenate([row_idx, col_idx]),
              np.concatenate([col_idx, row_idx]))),
            shape=(data.num_nodes, data.num_nodes)
        )
        tree = minimum_spanning_tree(adj)
        tree_row, tree_col = tree.nonzero()
        tree_edge_set = set()
        for u, v in zip(tree_row, tree_col):
            tree_edge_set.add((min(u, v), max(u, v)))

        # Partition: tree edges (protected) vs non-tree edges (splittable)
        tree_mask = np.array([(r, c) in tree_edge_set for r, c in edges])
        protected_edges = edges[tree_mask]
        splittable_edges = edges[~tree_mask]
        num_splittable = len(splittable_edges)

        print(f"[Connectivity] Protected {len(protected_edges)} spanning-tree edges "
              f"({len(protected_edges)*100/num_edges:.1f}% of {num_edges} total edges)")

        # Compute desired val/test counts from total edges
        num_val = int(num_edges * val_ratio)
        num_test = int(num_edges * test_ratio)

        # Clamp if not enough splittable edges
        if num_val + num_test > num_splittable:
            ratio_sum = val_ratio + test_ratio
            num_val = int(num_splittable * (val_ratio / ratio_sum))
            num_test = num_splittable - num_val
            print(f"[Connectivity] Warning: reduced val/test sizes to {num_val}/{num_test} "
                  f"(not enough non-tree edges)")

        # Split only splittable edges
        perm = np.random.permutation(num_splittable)
        val_edges = splittable_edges[perm[:num_val]]
        test_edges = splittable_edges[perm[num_val:num_val + num_test]]
        remaining_train = splittable_edges[perm[num_val + num_test:]]

        # Combine: protected (tree) + remaining splittable = train
        train_edges = np.concatenate([protected_edges, remaining_train], axis=0)
        
        # --- FIX FOR DATA LEAKAGE ---
        # 3. Create a training graph Data object that ONLY contains training edges
        train_edge_index = torch.tensor(train_edges, dtype=torch.long).t()
        train_edge_index = to_undirected(train_edge_index)
        train_graph = Data(edge_index=train_edge_index, num_nodes=data.num_nodes)
        # Preserve node features if present in the original dataset
        if hasattr(data, 'x') and data.x is not None:
            train_graph.x = data.x

        # Generate negative edges for validation and test
        num_nodes = data.num_nodes
        
        def generate_neg_edges(pos_edges, num_samples):
            # Using PyG's negative_sampling is more robust
            # We sample more than needed and take the first `num_samples` unique pairs
            neg_edge_index = negative_sampling(
                edge_index=to_undirected(torch.tensor(pos_edges).t()), # Use all edges to avoid sampling existing links
                num_nodes=num_nodes,
                num_neg_samples=num_samples * 2
            )
            return neg_edge_index[:, :num_samples].t().numpy()

        def generate_eval_neg_edges(pos_edges, num_nodes, num_neg_per_pos):
            """Generates a large number of negative samples for each positive edge for evaluation."""
            num_pos = len(pos_edges)
            
            # Get the source nodes and repeat them for each negative sample
            pos_src = torch.tensor(pos_edges[:, 0], dtype=torch.long)
            neg_src = pos_src.repeat_interleave(num_neg_per_pos)
            
            # Generate random destination nodes
            neg_dst = torch.randint(0, num_nodes, (num_pos * num_neg_per_pos,), dtype=torch.long)
            
            # Reshape to (num_pos, num_neg_per_pos, 2)
            # First stack, then view. The final tensor should have pairs.
            neg_edges = torch.stack([neg_src, neg_dst], dim=-1)
            
            # Note: A more advanced sampler would check for collisions with true edges,
            # but for large, sparse graphs, this random approach is a standard baseline.
            return neg_edges.numpy()
    
        # val_neg_edges = generate_neg_edges(val_edges, len(val_edges))
        # test_neg_edges = generate_neg_edges(test_edges, len(test_edges))

        EVAL_NEG_SAMPLES = 1
        print(f"Generating {EVAL_NEG_SAMPLES} negative samples per positive edge for validation and testing.")
        val_neg_edges = generate_eval_neg_edges(val_edges, num_nodes, EVAL_NEG_SAMPLES)
        test_neg_edges = generate_eval_neg_edges(test_edges, num_nodes, EVAL_NEG_SAMPLES)
        
        split_edge = {
            'train': {
                'edge': torch.tensor(train_edges, dtype=torch.long),
            },
            'valid': {
                'edge': torch.tensor(val_edges, dtype=torch.long),
                'edge_neg': torch.tensor(val_neg_edges, dtype=torch.long),
            },
            'test': {
                'edge': torch.tensor(test_edges, dtype=torch.long),
                'edge_neg': torch.tensor(test_neg_edges, dtype=torch.long),
            },
            # 4. Add the training graph to the dictionary
            'train_graph': train_graph,
        }
        
        return split_edge