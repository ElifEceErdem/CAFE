# fairness_experiments_scaffold/core/data_loader.py
import os
import sys
import torch
import torch_geometric.transforms as T
from torch_sparse import SparseTensor

# Path Fix: Add the SNAP folder to sys.path to use your original snap_dataset.py
current_dir = os.path.dirname(os.path.abspath(__file__))
snap_path = os.path.abspath(os.path.join(current_dir, '../../SNAP'))
if snap_path not in sys.path:
    sys.path.append(snap_path)

from snap_dataset import SNAPDataset
from utils import gcn_normalization, adj_normalization

class DataConfig:
    """Configuration for dataset loading and preprocessing."""
    def __init__(self, data_name='ego-facebook', data_path='../../data/SNAP', 
                 val_ratio=0.05, test_ratio=0.1, encoder='HLGNN', device='cuda:0'):
        self.data_name = data_name
        self.data_path = data_path
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.encoder = encoder.upper()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

def load_dataset(args):
    """
    Load dataset using the original SNAPDataset.
    Returns: dataset, split_edge, data, degrees
    """
    dataset = SNAPDataset(root=args.data_path, name=args.data_name)
    
    # Check if a specific split file is provided
    if hasattr(args, 'split_file') and args.split_file and os.path.exists(args.split_file):
        print(f"Loading split from file: {args.split_file}")
        split_edge = torch.load(args.split_file)
    else:
        # Default behavior: Use seed 42 (unless data_split_seed is added later)
        # We can also support args.data_split_seed if needed, but file-based is priority now
        seed = getattr(args, 'data_split_seed', 42)
        split_edge = dataset.get_edge_split(val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=seed)
    
    data = split_edge['train_graph']
    # Use _data to bypass UserWarnings if the PyG version is newer
    num_nodes = dataset._data.num_nodes if hasattr(dataset, '_data') else dataset.data.num_nodes
    data.num_nodes = num_nodes
    
    # CRITICAL: Convert to SparseTensor AND create edge_index
    data = T.ToSparseTensor()(data)
    row, col, _ = data.adj_t.coo()
    data.edge_index = torch.stack([col, row], dim=0)  # ← MUST HAVE THIS!

    # Ensure node features are float if present
    if hasattr(data, 'x') and data.x is not None:
        data.x = data.x.to(torch.float)

    # Apply normalization based on encoder
    if args.encoder.upper() in ['GCN', 'SAGE']:
        data.adj_t = gcn_normalization(data.adj_t)
    elif args.encoder.upper() == 'WSAGE':
        data.adj_t = adj_normalization(data.adj_t)
    elif args.encoder.upper() == 'TRANSFORMER':
        # For transformer, recreate SparseTensor without edge weights
        row, col, edge_weight = data.adj_t.coo()
        data.adj_t = SparseTensor(row=row, col=col)
    
    # Move to device
    if str(args.device).lower() == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    data.edge_weight = None  # Ensure edge_weight is None after moving to device
    
    # Compute degrees for MMR/Exposure re-ranking
    deg = data.adj_t.sum(dim=1)
    if hasattr(deg, "to_dense"):
        deg = deg.to_dense()
    degrees = deg.cpu().float()
    
    return dataset, split_edge, data, degrees

def _get_facebook_style_node_mapping(data_name, root):
    """
    Derive the node mapping for oklahoma/unc datasets.

    These datasets have isolated nodes that get removed during
    _reindex_contiguous_in_place. We reconstruct the mapping by loading
    the original train/test pkl files and finding which nodes appear in edges.

    Returns:
        unique_nodes: sorted tensor of original node IDs that appear in edges
    """
    import pickle

    config = {
        'oklahoma': ('Oklahoma97', 'Oklahoma97'),
        'unc': ('UNC28', 'UNC28'),
    }
    folder, prefix = config[data_name]
    base = os.path.join(root, data_name, folder)

    train_items = pickle.load(open(os.path.join(base, f'{prefix}_train_items.pkl'), 'rb'))
    test_items = pickle.load(open(os.path.join(base, f'{prefix}_test_set.pkl'), 'rb'))

    edges = []
    for item in [train_items, test_items]:
        for key, value in item.items():
            for one in value:
                edges.append([key, one])

    edge_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
    unique_nodes = torch.unique(edge_tensor.reshape(-1), sorted=True)
    return unique_nodes


def load_sensitive_attr(data_name, num_nodes, root='./dataset'):
    """
    Load dataset-native sensitive attributes.

    Supported datasets: pokec_z, pokec_n, oklahoma, unc

    Args:
        data_name: Name of the dataset
        num_nodes: Number of nodes in the graph
        root: Root directory for dataset files

    Returns:
        torch.FloatTensor of shape [num_nodes] with binary values (0/1)
    """
    if data_name in ['oklahoma', 'unc']:
        return _load_facebook_style_sens(data_name, num_nodes, root)
    elif data_name in ['pokec_n', 'pokec_z']:
        return _load_pokec_sens(data_name, num_nodes, root)
    else:
        raise ValueError(
            f"No native sensitive attribute for dataset '{data_name}'. "
            f"Supported: oklahoma, unc, pokec_n, pokec_z"
        )


def _load_facebook_style_sens(data_name, num_nodes, root):
    """Load sensitive attributes for oklahoma/unc from pkl files with proper node alignment."""
    import pickle

    config = {
        'oklahoma': ('Oklahoma97', 'Oklahoma97'),
        'unc': ('UNC28', 'UNC28'),
    }
    folder, prefix = config[data_name]

    # Load sens pkl (try both possible locations)
    sens_path = os.path.join(root, data_name, folder, f'{prefix}_user_sen.pkl')
    if not os.path.exists(sens_path):
        sens_path = os.path.join(root, data_name, 'processed', f'{prefix}_user_sen.pkl')
    with open(sens_path, 'rb') as f:
        sens_dict = pickle.load(f)

    # Derive node mapping (which original node IDs were kept after reindexing)
    unique_nodes = _get_facebook_style_node_mapping(data_name, root)

    # Build properly aligned sens tensor
    sens = torch.FloatTensor([sens_dict[uid.item()] for uid in unique_nodes])
    sens = (sens > 0).float()

    if len(sens) >= num_nodes:
        return sens[:num_nodes]
    else:
        padded = torch.zeros(num_nodes, dtype=torch.float32)
        padded[:len(sens)] = sens
        return padded


def _load_pokec_sens(data_name, num_nodes, root):
    """Load sensitive attributes for pokec_n/pokec_z from CSV."""
    import pandas as pd

    csv_map = {
        'pokec_n': 'region_job_2.csv',
        'pokec_z': 'region_job.csv',
    }
    csv_file = csv_map[data_name]
    csv_path = os.path.join(root, data_name, csv_file)

    df = pd.read_csv(csv_path)
    sens = torch.FloatTensor(df['region'].values)
    sens = (sens > 0).float()

    if len(sens) >= num_nodes:
        return sens[:num_nodes]
    else:
        padded = torch.zeros(num_nodes, dtype=torch.float32)
        padded[:len(sens)] = sens
        return padded


def load_node_labels(data_name, num_nodes, root='./dataset'):
    """
    Load node class labels for the LBA fairness regularizer.

    For pokec_n/pokec_z: loads I_am_working_in_field from CSV (binarized).
    For oklahoma/unc: returns None (data.y already in data.pt, fixed by reindexing).

    Returns:
        torch.LongTensor of shape [num_nodes] or None
    """
    if data_name in ['pokec_n', 'pokec_z']:
        import pandas as pd
        csv_map = {
            'pokec_n': 'region_job_2.csv',
            'pokec_z': 'region_job.csv',
        }
        csv_file = csv_map[data_name]
        csv_path = os.path.join(root, data_name, csv_file)

        df = pd.read_csv(csv_path)
        labels = df['I_am_working_in_field'].values.copy()
        labels[labels > 1] = 1
        labels = torch.LongTensor(labels)
        return labels[:num_nodes]

    # oklahoma/unc: labels are in data.y (handled via _reindex_contiguous_in_place)
    return None


def build_train_adjacency(train_edges, num_nodes):
    """Build adjacency list for masking training edges (vectorized)."""
    import numpy as np
    edges_np = train_edges.cpu().numpy()
    # Filter valid edges
    mask = (edges_np[:, 0] < num_nodes) & (edges_np[:, 1] < num_nodes)
    edges_np = edges_np[mask]
    # Build symmetric edge arrays
    src = np.concatenate([edges_np[:, 0], edges_np[:, 1]])
    dst = np.concatenate([edges_np[:, 1], edges_np[:, 0]])
    # Sort by source for efficient grouping
    order = np.argsort(src)
    src = src[order]
    dst = dst[order]
    # Group by source node using numpy split
    split_pts = np.searchsorted(src, np.arange(num_nodes + 1))
    train_adj = [set(dst[split_pts[i]:split_pts[i+1]].tolist()) for i in range(num_nodes)]
    return train_adj


def augment_virtual_edges(data, coreness, similarity_threshold=0.20, max_candidates=1000):
    """
    Virtual Edge Augmentation (Graph Rewiring).
    adds synthetic "shortcut" edges between low-coreness nodes that share many neighbors.
    
    Args:
        data: PyG data object
        coreness: Coreness scores [num_nodes]
        similarity_threshold: Jaccard similarity threshold to add edge
        max_candidates: Max comparisons per node to avoid O(N^2)
        
    Returns:
        Modified data object
    """
    import torch
    from torch_sparse import SparseTensor
    
    print(f"[Virtual Edge Augmentation] Starting... (Threshold={similarity_threshold})")
    
    # 1. Identify Periphery Nodes (Low Coreness)
    # Define "Periphery" as bottom 20% or simply low raw value?
    # Let's use bottom 20% quantile.
    threshold_core = torch.quantile(coreness.float(), 0.2)
    periphery_mask = coreness <= threshold_core
    periphery_nodes = torch.nonzero(periphery_mask).squeeze()
    
    print(f"  Identified {len(periphery_nodes)} periphery nodes (Core <= {threshold_core.item()})")
    
    # 2. Compute Jaccard Similarity for Periphery Nodes
    # We only look for links BETWEEN periphery nodes or Periphery-Hub?
    # Implementation Plan says "between periphery nodes".
    
    # Convert edge_index to Adjacency Sets for fast lookup
    # (Assuming CPU for this part to save GPU memory)
    edge_index_cpu = data.edge_index.cpu()
    num_nodes = data.num_nodes
    adj_list = [set() for _ in range(num_nodes)]
    for i in range(edge_index_cpu.size(1)):
        u, v = edge_index_cpu[0, i].item(), edge_index_cpu[1, i].item()
        adj_list[u].add(v)
    
    new_edges = []
    
    # Naive iteration might be slow. Optimization:
    # Only check nodes that share at least 1 neighbor?
    # For now, let's just do random sampling of candidate pairs for speed.
    
    # Limit number of augmentations to avoid explosion
    max_new_edges = int(edge_index_cpu.size(1) * 0.05) # Max 5% increase
    
    # Efficient Jaccard for Sparse Graphs: Use Matrix Multiplication?
    # A @ A gives number of common neighbors.
    # We can use SparseTensor on GPU.
    
    # 2. Compute Jaccard Similarity via Sparse Matrix Multiplication
    # Jaccard(u, v) = |N(u) & N(v)| / |N(u) | N(v)|
    # Intersection = (A @ A)[u, v]
    # Union = deg[u] + deg[v] - Intersection
    
    try:
        from torch_sparse import SparseTensor
        
        # Ensure edge_index is on device for fast compute
        edge_index = data.edge_index
        N = data.num_nodes
        
        # Create SparseTensor (A) with explicit values=1.0 for counting
        row, col = edge_index
        # IMPORTANT: Must set values to 1.0 for matrix multiplication to count common neighbors
        values = torch.ones(row.size(0), dtype=torch.float, device=edge_index.device)
        A = SparseTensor(row=row, col=col, value=values, sparse_sizes=(N, N))
        
        # Compute A^2 (Intersection count = number of common neighbors)
        print("  Computing A^2 for common neighbors...")
        A2 = A @ A
        
        # Get candidate edges (u, v) where intersection > 0
        # coo() returns (row, col, value) - value is the common neighbor count
        row_A2, col_A2, intersection = A2.coo()
        
        # Safety check: if intersection is None, fall back to torch.ones
        if intersection is None:
            print("  Warning: A^2 has no values, using binary intersection...")
            intersection = torch.ones(row_A2.size(0), dtype=torch.float, device=edge_index.device)
        
        # Filter 1: Ignore existing edges
        # We can mask them out, or check later. Checking later is slower.
        # But A^2 typically is denser than A.
        # Let's filter by Periphery Node constraint first to reduce size.
        
        # We want at least one node to be Periphery? Or both?
        # Plan says "Add synthetic edges between periphery nodes". So BOTH.
        
        # Mask: u is periphery AND v is periphery
        mask_u = periphery_mask[row_A2]
        mask_v = periphery_mask[col_A2]
        mask_periphery = mask_u & mask_v
        
        # Filter candidates
        c_row = row_A2[mask_periphery]
        c_col = col_A2[mask_periphery]
        c_int = intersection[mask_periphery]
        
        print(f"  Found {c_row.size(0)} candidate pairs between periphery nodes.")
        
        # Calculate Degrees for Jaccard
        deg = A.sum(dim=1).to(torch.long) # [N]
        
        deg_u = deg[c_row]
        deg_v = deg[c_col]
        
        union = deg_u + deg_v - c_int
        jaccard = c_int.float() / (union.float() + 1e-8)
        
        # Filter 2: Jaccard Threshold
        mask_jaccard = jaccard >= similarity_threshold
        
        final_row = c_row[mask_jaccard]
        final_col = c_col[mask_jaccard]
        
        # Remove self-loops (u != v)
        mask_loop = final_row != final_col
        final_row = final_row[mask_loop]
        final_col = final_col[mask_loop]
        
        # Remove existing edges?
        # A @ A includes paths of length 2. If triangle exists, u-v exists.
        # We need to check if (u, v) is already in A.
        # For small graphs, we can use a set check or just let it be (multi-edge).
        # HLGNN handles multi-edges usually (sum). But let's be clean.
        
        # Fast check: (A._value(final_row, final_col) == 0) ?
        # SparseTensor doesn't support vector indexing easily for lookup.
        # Converting existing edges to set is O(E).
        
        existing_edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        
        # Collect strictly new edges
        new_edges_list = []
        count = 0
        
        # Iterate (on CPU for set check)
        final_row_cpu = final_row.cpu().tolist()
        final_col_cpu = final_col.cpu().tolist()
        
        # Limit max new edges
        max_new = int(edge_index.size(1) * 0.10) # Max 10%
        
        for u, v in zip(final_row_cpu, final_col_cpu):
            if count >= max_new: break
            if u > v: continue # Only keep one direction for sorting/checking (assuming undirected logic)
            
            # Check exist
            if (u, v) in existing_edges or (v, u) in existing_edges:
                continue
                
            new_edges_list.append([u, v])
            new_edges_list.append([v, u]) # Add both directons
            count += 2
            
        if len(new_edges_list) > 0:
            print(f"  Adding {len(new_edges_list)} new edges (valid rewiring).")
            new_edge_tensor = torch.tensor(new_edges_list, dtype=torch.long, device=edge_index.device).t()
            data.edge_index = torch.cat([data.edge_index, new_edge_tensor], dim=1)
            
            # Rebuild Data Adjacency
            # NOTE: We must normalize!
            row, col = data.edge_index
            data.adj_t = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
            # Normalization handled at end of function
        else:
            print("  No new edges met criteria.")

    except ImportError:
        print("  torch_sparse not available, skipping precise rewiring.")
        return data
    except Exception as e:
        print(f"  Rewiring failed: {e}")
        return data
    
    # Normalization (needed for GNN)
    if hasattr(data, 'adj_t'):
        # Re-apply normalization
        # We need to access args.encoder. But we don't have args here.
        # Let's assume GCN/SAGE default normalization.
        data.adj_t = gcn_normalization(data.adj_t)
        
    return data