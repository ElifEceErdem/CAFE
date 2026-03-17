"""
RA (Resource Allocation) Node Features Module.

Computes RA-derived node features for use as GNN input.
"""
import numpy as np
import scipy.sparse as sp


def compute_ra_scores(adj_matrix):
    """
    Compute Resource Allocation scores: RA(u,v) = sum(1/deg(z)) for z in shared_neighbors.
    
    Args:
        adj_matrix: scipy sparse adjacency matrix [N, N]
        
    Returns:
        ra_matrix: scipy sparse matrix of RA scores [N, N]
    """
    # Compute degrees
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    
    # Handle zero degrees to avoid div by zero
    degrees[degrees == 0] = 1.0
    
    # Create D^-1 matrix
    inv_deg = sp.diags(1.0 / degrees)
    
    # Compute A * D^-1 * A^T
    # This matrix multiplication yields the RA sums directly
    ra_matrix = (adj_matrix.dot(inv_deg)).dot(adj_matrix.T)
    
    return ra_matrix


def compute_ra_node_features(adj_matrix, num_nodes, normalize=True):
    """
    Compute ra_mean per node with optional normalization.
    
    Args:
        adj_matrix: scipy sparse adjacency matrix [N, N]
        num_nodes: Number of nodes in the graph
        normalize: Whether to apply log + z-score normalization
        
    Returns:
        ra_features: numpy array [N, 1] of ra_mean values
    """
    # Compute RA matrix
    ra_matrix = compute_ra_scores(adj_matrix)
    
    # Sum of RA scores per node
    ra_sum = np.array(ra_matrix.sum(axis=1)).flatten()
    
    # Node degrees
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    
    # ra_mean = ra_sum / degree
    ra_mean = np.divide(ra_sum, degrees, where=degrees > 0, out=np.zeros_like(ra_sum))
    
    if normalize:
        # Log transform (handles zeros with log1p)
        ra_mean = np.log1p(ra_mean)
        
        # Z-score normalization
        mean_val = ra_mean.mean()
        std_val = ra_mean.std()
        if std_val > 1e-8:
            ra_mean = (ra_mean - mean_val) / std_val
        else:
            ra_mean = ra_mean - mean_val
    
    return ra_mean.reshape(-1, 1).astype(np.float32)
