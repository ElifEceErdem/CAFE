"""
Per-Coreness Precision Breakdown

Computes Precision@K per node, groups nodes by coreness, and plots the
mean precision per coreness group. Supports two modes:

  --mode baseline   : Evaluates CN, RA, AA, LightGCN, HL-GNN (default)
  --mode hybrid     : Also evaluates the Hybrid method (GNN + RA + mobility boost)
                      Requires --hybrid_beta and --hybrid_lambda

Only nodes with ≥1 test edge are included in the evaluation.

Usage:
    # Baselines only
    python run_precision_by_coreness.py --data_name ego-facebook --runs 3

    # With hybrid (once you have best config)
    python run_precision_by_coreness.py --data_name ego-facebook --runs 3 \
        --mode hybrid --hybrid_beta 0.5 --hybrid_lambda 1.0
"""

import os
import sys
import argparse
import hashlib
import json
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
backbone_dir = os.path.abspath(os.path.join(current_dir, '../backbone'))
fairness_dir = os.path.abspath(os.path.join(current_dir, '../fairness_methods'))
snap_dir = os.path.abspath(os.path.join(current_dir, '../../SNAP'))

for path in [parent_dir, backbone_dir, fairness_dir, snap_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from core.utils import set_seed, str2bool
from core.data_loader import load_dataset, build_train_adjacency
from core.training import train_model
from core.evaluation import make_predictor_score_fn
from backbone.model import BaseModel
from baselines.baseline_wrapper import BaselineWrapper
from metrics.coreness import compute_coreness
from core.ra_features import compute_ra_scores
from metrics.centrality_equality_gain import (
    compute_degree_equality_gain_by_source_bins_for_models,
)
from metrics.graph_utils import canonicalize_undirected_edges


def argument():
    parser = argparse.ArgumentParser(
        description='Per-Coreness Precision Breakdown'
    )
    parser.add_argument('--data_name', type=str, default='ego-facebook')
    parser.add_argument('--data_path', type=str, default='../../data/SNAP')
    parser.add_argument('--val_ratio', type=float, default=0.05)
    parser.add_argument('--test_ratio', type=float, default=0.1)

    # Training
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16384)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--neg_sampler', type=str, default='global')
    parser.add_argument('--num_neg', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--grad_clip_norm', type=float, default=2.0)

    # HLGNN
    parser.add_argument('--alpha', type=float, default=0.5)

    # Analysis
    parser.add_argument('--K', type=int, default=10, help='K for Precision@K')
    parser.add_argument('--num_bins', type=int, default=10,
                        help='Number of coreness quantile bins')
    parser.add_argument('--num_degree_bins', type=int, default=4,
                        help='Number of degree groups for heatmap (default 4 = quartiles)')
    parser.add_argument('--num_coreness_bins', type=int, default=5,
                        help='Number of coreness bins within each degree group for heatmap')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--save_dir', type=str, default='plots/precision_by_coreness')

    # Mode
    parser.add_argument('--mode', type=str, default='baseline',
                        choices=['baseline', 'hybrid'],
                        help='baseline: CN+RA+AA+LightGCN+HLGNN; hybrid: also includes Hybrid method')
    parser.add_argument('--hybrid_beta', type=float, default=0.0,
                        help='RA score weight for hybrid method')
    parser.add_argument('--hybrid_lambda', type=float, default=0.0,
                        help='Mobility boost strength for hybrid method')

    # Other
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_metric', type=str, default='all')
    parser.add_argument('--fairness_mode', type=str, default='indegree')
    parser.add_argument('--use_node_feats', type=str2bool, default=False)
    parser.add_argument('--train_node_emb', type=str2bool, default=True)
    parser.add_argument('--encoder', type=str, default='HLGNN')
    parser.add_argument('--split_file', type=str, default='')
    parser.add_argument('--data_split_seed', type=int, default=42,
                        help='Fixed random-split seed (ignored by Brightkite temporal split)')

    return parser.parse_args()


# ========================================================================
# Model Training
# ========================================================================

def train_hlgnn(data, split_edge, num_nodes, device, args):
    """Train HL-GNN and return (model, embeddings)."""
    print("\n" + "=" * 60)
    print("TRAINING: HL-GNN")
    print("=" * 60)

    model = BaseModel(
        lr=args.lr,
        dropout=args.dropout,
        grad_clip_norm=args.grad_clip_norm,
        gnn_num_layers=15,
        mlp_num_layers=2,
        emb_hidden_channels=256,
        gnn_hidden_channels=256,
        mlp_hidden_channels=256,
        num_nodes=num_nodes,
        num_node_feats=0,
        gnn_encoder_name='HLGNN',
        predictor_name='MLP',
        loss_func='AUC',
        optimizer_name='Adam',
        device=device,
        use_node_feats=args.use_node_feats,
        train_node_emb=args.train_node_emb,
        pretrain_emb=None,
        alpha=args.alpha,
        init='KI'
    )
    model.param_init()

    model, best_epoch, _, _, _, _ = train_model(
        model, data, split_edge, args, patience=args.patience
    )

    model.encoder.eval()
    model.predictor.eval()
    with torch.no_grad():
        input_feat = model.create_input_feat(data)
        h = model.encoder(input_feat, data.adj_t, data.edge_weight)

    return model, h


def train_lightgcn(data, split_edge, num_nodes, device, args):
    """Train LightGCN and return embeddings on device."""
    print("\n" + "=" * 60)
    print("TRAINING: LightGCN")
    print("=" * 60)

    wrapper = BaselineWrapper(
        baseline_type='lightgcn',
        num_nodes=num_nodes,
        device=device,
        embedding_dim=64,
        gnn_num_layers=2,
        lr=args.lr,
    )

    wrapper.fit(data, split_edge, epochs=args.epochs, batch_size=args.batch_size)

    h = wrapper.get_embeddings(data)
    return h


# ========================================================================
# Heuristic Score Computation (Sparse)
# ========================================================================

def compute_aa_scores(adj_matrix):
    """
    Compute Adamic-Adar scores as a sparse matrix.
    AA(u,v) = sum(1/log(deg(z))) for z in common_neighbors(u,v).

    Nodes with degree <= 1 are excluded as common-neighbor contributors
    (log(1) = 0 would cause division by zero, matching standard AA definition).

    Args:
        adj_matrix: scipy sparse symmetric binary adjacency matrix [N, N]

    Returns:
        aa_matrix: scipy sparse CSR matrix of AA scores [N, N]
    """
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()

    # Compute 1/log(deg) for each node; zero out nodes with deg <= 1
    log_degrees = np.log(degrees.clip(min=2))  # clip to avoid log(0) and log(1)
    inv_log_deg = 1.0 / log_degrees
    inv_log_deg[degrees <= 1] = 0.0

    inv_log_diag = sp.diags(inv_log_deg)

    # AA = A * diag(1/log(deg)) * A^T
    aa_matrix = (adj_matrix.dot(inv_log_diag)).dot(adj_matrix.T)
    return aa_matrix.tocsr()


# ========================================================================
# Recommendation Generation
# ========================================================================

def generate_recs_from_sparse(score_sparse, adj_csr, eval_users, K, method_name):
    """
    Generate top-K recommendations from a precomputed sparse score matrix.
    Uses chunked processing and sparse CSR masking for efficiency.

    Works for CN (A*A), RA (A*D_inv*A^T), and AA (A*D_log_inv*A^T).

    Args:
        score_sparse: scipy sparse score matrix [N, N]
        adj_csr: scipy sparse CSR binary adjacency (for train-edge masking)
        eval_users: sorted list of node IDs to evaluate
        K: number of recommendations per node
        method_name: string for logging (e.g. 'CN', 'RA', 'AA')

    Returns:
        recs_by_u: dict {node_id: [top-K recommended node ids]}
    """
    recs_by_u = {}
    chunk_size = 512

    print(f"  Generating {method_name} recs for {len(eval_users)} users...")

    for i in range(0, len(eval_users), chunk_size):
        chunk_end = min(i + chunk_size, len(eval_users))
        chunk = eval_users[i:chunk_end]
        chunk_arr = np.array(chunk)
        B = len(chunk)

        # Extract scores for this chunk from sparse matrix -> dense [B, N]
        scores = torch.from_numpy(
            score_sparse[chunk_arr].toarray()
        ).float()

        # Mask self-loops
        self_cols = torch.tensor(chunk, dtype=torch.long)
        scores[torch.arange(B), self_cols] = float('-inf')

        # Mask train neighbors (sparse CSR)
        adj_chunk = adj_csr[chunk_arr]
        nb_r, nb_c = adj_chunk.nonzero()
        scores[nb_r, nb_c] = float('-inf')

        # Top-K
        topk_vals, topk_idx = torch.topk(scores, K, dim=1)
        for j in range(B):
            recs_by_u[chunk[j]] = topk_idx[j].tolist()

        if chunk_end % 5000 == 0 or chunk_end == len(eval_users):
            print(f"    Processed {chunk_end}/{len(eval_users)} nodes...")

    return recs_by_u


def build_recs_chunked(test_source_nodes, num_nodes, k, train_adj_csr, device,
                       h_emb=None, score_fn=None, score_matrix_sparse=None):
    """
    Efficient chunked top-K recommendation builder.

    Ported verbatim from HL-GNN/fairness_experiments_scaffold/experiments/
    run_coreness_hypothesis.py so the heatmap here matches that script's
    output (correct [:, 0] / [:, 1] train-edge masking via train_adj_csr).

    Scoring modes (exactly one must apply):
      - score_fn + h_emb: score_fn(h_batch, h_emb) -> [B, N]  (HL-GNN MLP predictor)
      - h_emb only (score_fn=None, score_matrix_sparse=None): dot product  (LightGCN)
      - score_matrix_sparse: precomputed sparse score matrix  (RA)
    """
    recs_by_u = {}
    chunk_size = 1024

    num_test = len(test_source_nodes)
    num_chunks = (num_test + chunk_size - 1) // chunk_size

    for chunk_i in range(num_chunks):
        chunk_start = chunk_i * chunk_size
        chunk_end = min(chunk_start + chunk_size, num_test)
        chunk_node_ids = test_source_nodes[chunk_start:chunk_end]
        B = len(chunk_node_ids)
        node_ids_np = np.array(chunk_node_ids)

        if chunk_i % 5 == 0 or chunk_i == num_chunks - 1:
            print(f"    Chunk {chunk_i+1}/{num_chunks} ({B} nodes)", flush=True)

        if score_matrix_sparse is not None:
            scores = torch.from_numpy(
                score_matrix_sparse[node_ids_np].toarray()
            ).float()

        elif score_fn is not None:
            batch_tensor = torch.tensor(chunk_node_ids, dtype=torch.long, device=device)
            with torch.no_grad():
                h_batch = h_emb[batch_tensor]
                scores = score_fn(h_batch, h_emb).cpu()

        else:
            batch_tensor = torch.tensor(chunk_node_ids, dtype=torch.long, device=device)
            with torch.no_grad():
                h_batch = h_emb[batch_tensor]
                scores = torch.mm(h_batch, h_emb.t()).cpu()

        # Mask self-loops
        self_cols = torch.tensor(chunk_node_ids, dtype=torch.long)
        scores[torch.arange(B), self_cols] = float('-inf')

        # Mask train neighbors (sparse CSR)
        adj_chunk = train_adj_csr[node_ids_np]
        nb_r, nb_c = adj_chunk.nonzero()
        scores[nb_r, nb_c] = float('-inf')

        # Top-K
        topk_vals, topk_idx = torch.topk(scores, k, dim=1)
        for i in range(B):
            recs_by_u[chunk_node_ids[i]] = topk_idx[i].tolist()

    return recs_by_u


def generate_recs_hlgnn(model, h, train_adj_csr, eval_users, K, device, num_nodes):
    """Generate top-K recommendations using HL-GNN's trained predictor."""
    score_fn = make_predictor_score_fn(model.predictor, predictor_type='two_arg')
    recs = build_recs_chunked(
        eval_users, num_nodes, k=K,
        train_adj_csr=train_adj_csr, device=device,
        h_emb=h, score_fn=score_fn,
    )
    return recs


def generate_recs_lightgcn(h_emb, train_adj_csr, eval_users, K, device, num_nodes):
    """Generate top-K recommendations using LightGCN embeddings (dot product)."""
    recs = build_recs_chunked(
        eval_users, num_nodes, k=K,
        train_adj_csr=train_adj_csr, device=device,
        h_emb=h_emb,
    )
    return recs


def generate_recs_hybrid(h_emb, train_edges, eval_users, K, device,
                          num_nodes, coreness, beta, lam):
    """
    Generate top-K recommendations using Hybrid scoring:
        final = (gnn_score + β * RA_score) * (1 + λ * mobility_boost)
    """
    print(f"  Computing RA scores (β={beta})...")
    # Build adjacency matrix
    train_np = train_edges.cpu().numpy()
    adj = sp.csr_matrix(
        (np.ones(len(train_np)), (train_np[:, 0], train_np[:, 1])),
        shape=(num_nodes, num_nodes)
    )
    adj = adj + adj.T
    adj[adj > 1] = 1

    ra_dense = compute_ra_scores(adj)

    # Compute mobility boost
    c_np = coreness.cpu().numpy()
    c_max = c_np.max()
    if c_max > 0:
        mobility_boost = 1.0 - (c_np / c_max)
    else:
        mobility_boost = np.zeros_like(c_np)

    mult_vec = 1.0 + lam * mobility_boost
    mult_vec = np.clip(mult_vec, 1.0, 5.0)
    mult_tensor = torch.from_numpy(mult_vec).float()

    # Build train adjacency for masking
    train_adj = build_train_adjacency(train_edges, num_nodes)

    recs_by_u = {}
    chunk_size = 1024

    print(f"  Generating hybrid recs for {len(eval_users)} users (β={beta}, λ={lam})...")

    for i in range(0, len(eval_users), chunk_size):
        chunk_end = min(i + chunk_size, len(eval_users))
        chunk = eval_users[i:chunk_end]
        chunk_tensor = torch.tensor(chunk, dtype=torch.long, device=device)

        with torch.no_grad():
            h_batch = h_emb[chunk_tensor]
            gnn_logits = h_batch @ h_emb.t()
            gnn_scores = torch.sigmoid(gnn_logits).cpu()

        # RA scores for this chunk
        ra_batch = torch.zeros(len(chunk), num_nodes, dtype=torch.float32)
        for idx, u in enumerate(chunk):
            ra_row = ra_dense[u]
            if hasattr(ra_row, 'toarray'):
                ra_batch[idx] = torch.from_numpy(ra_row.toarray().flatten()).float()
            else:
                ra_batch[idx] = torch.from_numpy(np.asarray(ra_row).flatten()).float()

        # Hybrid score
        hybrid_scores = gnn_scores + beta * ra_batch

        # Mobility boost
        final_scores = hybrid_scores * mult_tensor.unsqueeze(0)

        # Masking & TopK
        for idx, u in enumerate(chunk):
            u_scores = final_scores[idx].clone()
            u_scores[u] = float('-inf')  # mask self
            for v in train_adj[u]:
                if v < num_nodes:
                    u_scores[v] = float('-inf')

            topk_scores, topk_indices = torch.topk(u_scores, min(K, num_nodes))
            valid_mask = torch.isfinite(topk_scores)
            topk_indices = topk_indices[valid_mask][:K]
            recs_by_u[u] = topk_indices.tolist()

        if chunk_end % 5000 == 0 or chunk_end == len(eval_users):
            print(f"    Processed {chunk_end}/{len(eval_users)} nodes...")

    return recs_by_u


# ========================================================================
# Per-Node Precision Computation
# ========================================================================

def compute_per_node_precision(recs_by_u, test_edge_dict, K):
    """
    Compute Precision@K for each individual node.

    Args:
        recs_by_u: dict {node_id: [recommended_node_ids]}
        test_edge_dict: dict {source_node: set of true target nodes}
        K: number of recommendations to consider

    Returns:
        dict {node_id: precision@K}
    """
    precisions = {}
    for u, recs in recs_by_u.items():
        true_targets = test_edge_dict.get(u, set())
        if len(true_targets) == 0:
            continue  # skip nodes with no test edges
        top_k = recs[:K]
        hits = len(set(top_k) & true_targets)
        precisions[u] = hits / K
    return precisions


def build_test_edge_dict(pos_test_edge):
    """Build dict mapping source node -> set of true target nodes."""
    test_dict = {}
    edges = pos_test_edge.cpu().numpy()
    for u, v in edges:
        if u not in test_dict:
            test_dict[u] = set()
        test_dict[u].add(v)
        # Also add reverse since undirected
        if v not in test_dict:
            test_dict[v] = set()
        test_dict[v].add(u)
    return test_dict


def split_fingerprint(split_edge, num_nodes):
    """Hash the canonical train/test edge sets used by the diagnosis shard."""
    digest = hashlib.sha256()
    digest.update(f'num_nodes={int(num_nodes)}\n'.encode())
    for split_name in ('train', 'test'):
        edges = canonicalize_undirected_edges(
            split_edge[split_name]['edge'].cpu(), num_nodes
        )
        digest.update(f'{split_name}:{len(edges)}\n'.encode())
        digest.update(
            edges.to(dtype=torch.int64).contiguous().numpy().astype(
                '<i8', copy=False
            ).tobytes()
        )
    return digest.hexdigest()


# ========================================================================
# Coreness Binning and Plotting
# ========================================================================

MODEL_COLORS = {
    'CN': '#1abc9c',
    'RA': '#9b59b6',
    'AA': '#3498db',
    'LightGCN': '#e74c3c',
    'HL-GNN': '#e67e22',
    'Hybrid': '#2ecc71',
}

MODEL_MARKERS = {
    'CN': 'o',
    'RA': 's',
    'AA': '^',
    'LightGCN': 'D',
    'HL-GNN': 'p',
    'Hybrid': 'X',
}


def assign_coreness_bins(df, num_bins):
    """Assign quantile-based coreness bins."""
    try:
        df['coreness_bin'], bin_edges = pd.qcut(
            df['node_coreness'], q=num_bins, retbins=True,
            labels=False, duplicates='drop'
        )
    except ValueError:
        df['coreness_bin'], bin_edges = pd.cut(
            df['node_coreness'], bins=num_bins, retbins=True, labels=False
        )

    # Human-readable labels
    actual_bins = sorted(df['coreness_bin'].unique())
    bin_labels = {}
    for b in actual_bins:
        mask = df['coreness_bin'] == b
        lo = df.loc[mask, 'node_coreness'].min()
        hi = df.loc[mask, 'node_coreness'].max()
        bin_labels[b] = f"[{lo:.0f}-{hi:.0f}]"

    df['coreness_bin_label'] = df['coreness_bin'].map(bin_labels)
    return df, bin_labels


def assign_quantile_bins(series, num_bins, label_prefix=''):
    """
    Assign quantile-based bins to a pandas Series.
    Returns (bin_series, bin_label_map) where bin_label_map maps bin_index -> label string.
    Falls back to equal-width bins if quantile fails due to duplicates.
    """
    try:
        bin_series, bin_edges = pd.qcut(
            series, q=num_bins, retbins=True, labels=False, duplicates='drop'
        )
    except ValueError:
        bin_series, bin_edges = pd.cut(
            series, bins=num_bins, retbins=True, labels=False
        )

    actual_bins = sorted(bin_series.dropna().unique())
    bin_label_map = {}
    for b in actual_bins:
        mask = bin_series == b
        lo = series[mask].min()
        hi = series[mask].max()
        bin_label_map[b] = f"{label_prefix}[{lo:.0f}–{hi:.0f}]"

    return bin_series, bin_label_map


def plot_2d_heatmap(df, model_name, dataset_name, k,
                    num_degree_bins, num_coreness_bins, save_dir):
    """
    2D heatmap of mean Precision@K across (degree_bin, coreness_bin).

    If coreness matters independently, precision should increase left→right
    within every row of the heatmap.
    """
    df = df.copy()

    df['degree_bin'], deg_label_map = assign_quantile_bins(
        df['degree'], num_degree_bins, label_prefix='deg '
    )
    df['coreness_bin'], cor_label_map = assign_quantile_bins(
        df['coreness'], num_coreness_bins
    )

    degree_groups = sorted(df['degree_bin'].dropna().unique())
    coreness_groups = sorted(df['coreness_bin'].dropna().unique())

    # Build matrix: rows = degree (ascending), cols = coreness (ascending)
    matrix = np.full((len(degree_groups), len(coreness_groups)), np.nan)
    cell_counts = np.zeros_like(matrix, dtype=int)

    for i, dg in enumerate(degree_groups):
        for j, cg in enumerate(coreness_groups):
            cell = df[(df['degree_bin'] == dg) & (df['coreness_bin'] == cg)]
            if len(cell) >= 3:
                matrix[i, j] = cell['precision'].mean()
                cell_counts[i, j] = len(cell)

    fig, ax = plt.subplots(figsize=(max(6, len(coreness_groups) * 1.1),
                                     max(4, len(degree_groups) * 0.9)))
    ax.grid(False)
    ax.set_axisbelow(False)
    ax.tick_params(which='both', length=0)

    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn',
                   vmin=np.nanmin(matrix) if not np.all(np.isnan(matrix)) else 0,
                   vmax=np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 1)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f'Mean Precision@{k}', fontsize=11)

    # Annotate cells
    for i in range(len(degree_groups)):
        for j in range(len(coreness_groups)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f'{matrix[i, j]:.3f}\n(n={cell_counts[i,j]})',
                        ha='center', va='center', fontsize=7, color='black')
            else:
                ax.text(j, i, 'n/a', ha='center', va='center',
                        fontsize=7, color='gray')

    ax.set_xticks(range(len(coreness_groups)))
    ax.set_xticklabels(
        [cor_label_map.get(cg, str(cg)) for cg in coreness_groups],
        rotation=30, ha='right', fontsize=9
    )
    ax.set_yticks(range(len(degree_groups)))
    ax.set_yticklabels(
        [deg_label_map.get(dg, str(dg)) for dg in degree_groups],
        fontsize=9
    )

    ax.set_xlabel('Coreness (k-core number) →', fontsize=11)
    ax.set_ylabel('← Degree group', fontsize=11)
    ax.set_title(
        f'Precision@{k}: Degree × Coreness Heatmap\n'
        f'{model_name} — {dataset_name}\n'
        f'(If hypothesis holds: precision increases left→right within each row)',
        fontsize=11, fontweight='bold'
    )

    plt.tight_layout()
    fname = f'heatmap_{model_name.replace(" ", "_")}_{dataset_name}_k{k}.pdf'
    path = os.path.join(save_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def compute_eg_by_coreness_bin(all_recs, coreness_np, bin_assignments, bin_labels,
                               train_edges, pos_test_edge, num_nodes, K):
    """
    Compute degree-based Equality Gain per coreness bin per model.

    For each bin, we filter recs_by_u to only include nodes in that bin,
    then compute EG on the resulting predicted graph.

    Args:
        all_recs: dict {model_name: recs_by_u}
        coreness_np: array of coreness values per node
        bin_assignments: dict {node_id: bin_index}
        bin_labels: dict {bin_index: label_str}
        train_edges: training edges tensor (CPU)
        pos_test_edge: test edges tensor (CPU)
        num_nodes: total node count
        K: top-K for recommendations

    Returns:
        pd.DataFrame with columns [model, coreness_bin, coreness_bin_label, equality_gain]
    """
    rows = []
    bins = sorted(bin_labels.keys())
    results_by_model = compute_degree_equality_gain_by_source_bins_for_models(
        all_recs,
        bin_assignments,
        train_edges.cpu(),
        pos_test_edge.cpu(),
        num_nodes,
        K,
        bin_ids=bins,
    )

    for model_name, recs_by_u in all_recs.items():
        print(f"\n  {model_name}:")
        bin_results = results_by_model[model_name]
        for b in bins:
            eg, gini_control, gini_pred = bin_results[b]
            n_sources = sum(
                1 for u in recs_by_u if bin_assignments.get(int(u)) == b
            )

            rows.append({
                'model': model_name,
                'coreness_bin': b,
                'coreness_bin_label': bin_labels[b],
                'equality_gain': eg,
                'gini_control': gini_control,
                'gini_pred': gini_pred,
                'n_sources': n_sources,
            })
            print(
                f"    Bin {bin_labels[b]}: EG={eg:.2f}% "
                f"(Gini_ctrl={gini_control:.4f}, "
                f"Gini_pred={gini_pred:.4f}, n_sources={n_sources})"
            )

    return pd.DataFrame(rows)


def aggregate_precision_within_runs(prec_df):
    """Aggregate node precision to one value per run/model/coreness bin.

    Run-level replication is the experimental unit.  Computing whiskers from
    the raw node rows would treat thousands of users as independent reruns
    (pseudo-replication) and dramatically understate uncertainty.
    """
    required = {
        'model', 'coreness_bin', 'coreness_bin_label', 'precision',
    }
    missing = required - set(prec_df.columns)
    if missing:
        raise ValueError(
            f"Precision dataframe lacks columns: {sorted(missing)}"
        )

    working = prec_df.copy()
    if 'run' not in working.columns:
        working['run'] = 0

    return (
        working
        .groupby(
            ['run', 'model', 'coreness_bin', 'coreness_bin_label'],
            as_index=False,
        )
        .agg(
            precision=('precision', 'mean'),
            n_nodes=('node_id', 'nunique')
            if 'node_id' in working.columns
            else ('precision', 'count'),
        )
    )


def _ordered_models(values, requested_order=None):
    available = list(dict.fromkeys(values))
    if requested_order is None:
        return available
    ordered = [name for name in requested_order if name in available]
    ordered.extend(name for name in available if name not in ordered)
    return ordered


def plot_stacked_barchart(
    prec_df,
    eg_df,
    save_dir,
    dataset_name,
    K,
    model_order=None,
    display_name=None,
    filename_tag=None,
    figure_size=(6, 6),
    font_size=10,
    legend_inside=False,
):
    """
    Compact vertically-stacked grouped bar chart for paper figures.

    Two rows, one column, shared x-axis:
      Top:    Precision@K  (grouped bars + SEM error bars)
      Bottom: Equality Gain (%) (grouped bars + SEM error bars if `eg_sem`
              column is present in `eg_df`)

    X-tick labels only on the bottom panel.
    Output: stacked_barchart_{dataset_name}.pdf
    """
    os.makedirs(save_dir, exist_ok=True)

    models_prec = _ordered_models(prec_df['model'].unique(), model_order)
    models_eg = _ordered_models(eg_df['model'].unique(), model_order)
    bin_order = sorted(prec_df['coreness_bin'].unique())

    # Build x-tick labels from precision df
    label_map = (prec_df[['coreness_bin', 'coreness_bin_label']]
                 .drop_duplicates()
                 .set_index('coreness_bin'))
    xlabels = [label_map.loc[b, 'coreness_bin_label']
               if b in label_map.index else str(b)
               for b in bin_order]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figure_size, sharex=True,
        gridspec_kw={'hspace': 0.08}
    )

    x = np.arange(len(bin_order))

    # ------------------------------------------------------------------
    # Top panel: Precision@K grouped bars
    # ------------------------------------------------------------------
    num_models_prec = len(models_prec)
    width_prec = 0.7 / max(num_models_prec, 1)
    precision_by_run = aggregate_precision_within_runs(prec_df)

    for i, model_name in enumerate(models_prec):
        model_df = precision_by_run[
            precision_by_run['model'] == model_name
        ]
        grouped = model_df.groupby('coreness_bin')['precision'].agg(
            ['mean', 'std', 'count']
        )

        means = [grouped.loc[b, 'mean'] if b in grouped.index else 0
                 for b in bin_order]
        stds = [grouped.loc[b, 'std'] if b in grouped.index else 0
                for b in bin_order]
        counts = [grouped.loc[b, 'count'] if b in grouped.index else 0
                  for b in bin_order]
        sems = [s / np.sqrt(c) if c > 1 else 0
                for s, c in zip(stds, counts)]

        offset = (i - (num_models_prec - 1) / 2) * width_prec
        color = MODEL_COLORS.get(model_name, f'C{i}')
        ax_top.bar(x + offset, means, width_prec, yerr=sems,
                   label=model_name, color=color, alpha=0.85,
                   capsize=2, edgecolor='white', linewidth=0.5)

    tick_font_size = max(font_size - 2, 8)
    legend_font_size = max(font_size - 3, 8)

    ax_top.set_ylabel(f'Precision@{K}', fontsize=font_size)
    ax_top.tick_params(axis='y', labelsize=tick_font_size)
    ax_top.grid(axis='y', alpha=0.3, linewidth=0.5)

    # ------------------------------------------------------------------
    # Bottom panel: Equality Gain grouped bars
    # ------------------------------------------------------------------
    num_models_eg = len(models_eg)
    width_eg = 0.7 / max(num_models_eg, 1)
    eg_bins_available = set(eg_df['coreness_bin'].unique())
    has_eg_sem = 'eg_sem' in eg_df.columns

    for i, model_name in enumerate(models_eg):
        model_df = eg_df[eg_df['model'] == model_name]

        egs = []
        sems = []
        for b in bin_order:
            if b in eg_bins_available:
                match = model_df.loc[model_df['coreness_bin'] == b]
                if len(match) > 0:
                    egs.append(float(match['equality_gain'].values[0]))
                    if has_eg_sem:
                        sem_val = match['eg_sem'].values[0]
                        sems.append(float(sem_val) if not pd.isna(sem_val) else 0.0)
                    else:
                        sems.append(0.0)
                else:
                    egs.append(0.0)
                    sems.append(0.0)
            else:
                egs.append(0.0)
                sems.append(0.0)

        offset = (i - (num_models_eg - 1) / 2) * width_eg
        color = MODEL_COLORS.get(model_name, f'C{i}')
        ax_bot.bar(x + offset, egs, width_eg,
                   yerr=sems if has_eg_sem else None,
                   label=model_name,
                   color=color, alpha=0.85, capsize=2,
                   edgecolor='white', linewidth=0.5)

    ax_bot.axhline(y=0, color='gray', linestyle='--', alpha=0.5,
                   linewidth=0.8)
    ax_bot.set_ylabel('Equality Gain (%)', fontsize=font_size)
    ax_bot.set_xlabel('Node Coreness Group', fontsize=font_size)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(xlabels, rotation=45, ha='right',
                           fontsize=tick_font_size)
    ax_bot.tick_params(axis='y', labelsize=tick_font_size)
    ax_bot.grid(axis='y', alpha=0.3, linewidth=0.5)

    # ------------------------------------------------------------------
    # Shared legend at the top
    # ------------------------------------------------------------------
    handles, labels = ax_top.get_legend_handles_labels()
    bot_handles, bot_labels = ax_bot.get_legend_handles_labels()
    for h, l in zip(bot_handles, bot_labels):
        if l not in labels:
            handles.append(h)
            labels.append(l)

    if legend_inside:
        ax_top.legend(
            handles,
            labels,
            loc='upper left',
            ncol=min(3, len(labels)),
            fontsize=legend_font_size,
            frameon=True,
            framealpha=0.92,
            edgecolor='0.8',
            handletextpad=0.3,
            columnspacing=0.8,
        )
        title_y = 0.99
    else:
        fig.legend(handles, labels, loc='upper center',
                   ncol=len(labels), fontsize=legend_font_size, frameon=False,
                   bbox_to_anchor=(0.5, 1.02),
                   handletextpad=0.3, columnspacing=1.0)
        title_y = 1.06

    title = display_name if display_name is not None else dataset_name
    fig.suptitle(f'{title} (K={K})', fontsize=font_size + 1,
                 fontweight='bold', y=title_y)

    output_tag = filename_tag if filename_tag is not None else dataset_name
    path = os.path.join(save_dir,
                        f'stacked_barchart_{output_tag}.pdf')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")





# ========================================================================
# Main
# ========================================================================

def main():
    args = argument()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 80)
    print("PER-CORENESS PRECISION BREAKDOWN")
    print(f"Dataset: {args.data_name}")
    print(f"Mode: {args.mode}")
    print(f"Models: CN, RA, AA, LightGCN, HL-GNN")
    print(f"K: {args.K}")
    print(f"Runs: {args.runs}")
    print(f"Coreness bins: {args.num_bins}")
    if args.mode == 'hybrid':
        print(f"Hybrid β: {args.hybrid_beta}")
        print(f"Hybrid λ: {args.hybrid_lambda}")
    print(f"Device: {device}")
    print("=" * 80)

    all_dfs = []
    per_run_eg_inputs = []

    for run in range(args.runs):
        print(f"\n{'=' * 80}")
        print(f"RUN {run + 1}/{args.runs}")
        print(f"{'=' * 80}")

        # Load data
        dataset, split_edge, data, degrees = load_dataset(args)
        num_nodes = dataset._data.num_nodes if hasattr(dataset, '_data') else dataset.data.num_nodes

        pos_test_edge = split_edge['test']['edge'].to(device)
        train_edges = split_edge['train']['edge'].to(device)
        fingerprint = split_fingerprint(split_edge, num_nodes)
        training_seed = int(args.seed + run)

        # Compute coreness on training graph
        print("\nComputing coreness...")
        coreness = compute_coreness(train_edges, num_nodes).to(device)
        coreness_np = coreness.cpu().numpy()
        print(f"Coreness stats: min={coreness.min().item()}, "
              f"max={coreness.max().item()}, "
              f"mean={coreness.float().mean().item():.2f}")

        # Build test edge lookup (symmetric: dict[u] holds u's test neighbors
        # in either direction, used for hit-checking in compute_per_node_precision)
        test_edge_dict = build_test_edge_dict(pos_test_edge)

        # Evaluation users: every node with >=1 test edge (undirected).
        # test_edge_dict is built symmetrically, so each test edge (u,v)
        # contributes both u and v as eval users.
        eval_users = sorted(test_edge_dict.keys())
        print(f"Evaluation nodes (with >=1 test edge): {len(eval_users)}/{num_nodes}")

        # ---- Build symmetric binary adjacency (sparse) ----
        train_np = train_edges.cpu().numpy()
        adj = sp.csr_matrix(
            (np.ones(len(train_np)), (train_np[:, 0], train_np[:, 1])),
            shape=(num_nodes, num_nodes)
        )
        adj = adj + adj.T
        adj[adj > 1] = 1

        # ---- Precompute heuristic score matrices ----
        print("\nComputing heuristic score matrices...")

        # CN: A * A (counts common neighbors)
        cn_sparse = adj.dot(adj).tocsr()
        print(f"  CN (A²): {cn_sparse.nnz} nonzeros")

        # Exact local clustering on the simple undirected TRAINING graph.  The
        # numerator sum_v A_uv(A^2)_uv equals twice the triangles incident to u.
        if np.any(adj.diagonal() != 0):
            raise ValueError('Training adjacency contains self-loops')
        train_degree_np = np.asarray(adj.sum(axis=1)).reshape(-1)
        closed_wedges = np.asarray(
            cn_sparse.multiply(adj).sum(axis=1)
        ).reshape(-1)
        denominator = train_degree_np * (train_degree_np - 1.0)
        clustering_np = np.zeros(num_nodes, dtype=np.float64)
        valid = denominator > 0
        clustering_np[valid] = closed_wedges[valid] / denominator[valid]
        if not np.isfinite(clustering_np).all():
            raise ValueError('Non-finite local clustering coefficient')

        # RA: A * D_inv * A^T
        ra_sparse = compute_ra_scores(adj)
        print(f"  RA: {ra_sparse.nnz} nonzeros")

        # AA: A * D_log_inv * A^T
        aa_sparse = compute_aa_scores(adj)
        print(f"  AA: {aa_sparse.nnz} nonzeros")

        # ---- Evaluate: CN ----
        print("\nGenerating CN recommendations...")
        recs_cn = generate_recs_from_sparse(
            cn_sparse, adj, eval_users, args.K, 'CN'
        )
        prec_cn = compute_per_node_precision(recs_cn, test_edge_dict, args.K)

        # ---- Evaluate: RA ----
        print("\nGenerating RA recommendations...")
        recs_ra = generate_recs_from_sparse(
            ra_sparse, adj, eval_users, args.K, 'RA'
        )
        prec_ra = compute_per_node_precision(recs_ra, test_edge_dict, args.K)

        # ---- Evaluate: AA ----
        print("\nGenerating AA recommendations...")
        recs_aa = generate_recs_from_sparse(
            aa_sparse, adj, eval_users, args.K, 'AA'
        )
        prec_aa = compute_per_node_precision(recs_aa, test_edge_dict, args.K)

        # Free heuristic matrices
        del cn_sparse, ra_sparse, aa_sparse
        import gc; gc.collect()

        # ---- Train & Evaluate: LightGCN ----
        set_seed(training_seed)
        h_lgcn = train_lightgcn(data, split_edge, num_nodes, device, args)
        h_lgcn = h_lgcn.to(device)
        print("\nGenerating LightGCN recommendations...")
        recs_lgcn = generate_recs_lightgcn(
            h_lgcn, adj, eval_users, args.K, device, num_nodes
        )
        prec_lgcn = compute_per_node_precision(recs_lgcn, test_edge_dict, args.K)

        del h_lgcn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---- Train & Evaluate: HL-GNN ----
        set_seed(training_seed)
        model_hlgnn, h_hlgnn = train_hlgnn(data, split_edge, num_nodes, device, args)
        print("\nGenerating HL-GNN recommendations...")
        recs_hlgnn = generate_recs_hlgnn(
            model_hlgnn, h_hlgnn, adj, eval_users, args.K, device, num_nodes
        )
        prec_hlgnn = compute_per_node_precision(recs_hlgnn, test_edge_dict, args.K)

        # Build DataFrames for this run
        degrees_np = degrees.cpu().numpy()

        def build_df(precisions, model_name):
            rows = []
            for u, p in precisions.items():
                rows.append({
                    'node_id': u,
                    'model': model_name,
                    'precision': p,
                    'node_coreness': coreness_np[u],
                    'degree': float(degrees_np[u]),
                    'clustering_coefficient': float(clustering_np[u]),
                    'n_test_targets': int(len(test_edge_dict.get(u, set()))),
                    'run': training_seed,
                    'training_seed': training_seed,
                    'split_seed': int(args.data_split_seed),
                    'split_fingerprint': fingerprint,
                })
            return pd.DataFrame(rows)

        run_dfs = [
            build_df(prec_cn, 'CN'),
            build_df(prec_ra, 'RA'),
            build_df(prec_aa, 'AA'),
            build_df(prec_lgcn, 'LightGCN'),
            build_df(prec_hlgnn, 'HL-GNN'),
        ]

        # Store all recs for per-run EG computation
        all_recs_for_eg = {
            'CN': recs_cn,
            'RA': recs_ra,
            'AA': recs_aa,
            'LightGCN': recs_lgcn,
            'HL-GNN': recs_hlgnn,
        }

        # ---- Optional: Hybrid ----
        if args.mode == 'hybrid':
            print("\nGenerating Hybrid recommendations...")
            recs_hybrid = generate_recs_hybrid(
                h_hlgnn, train_edges, eval_users, args.K, device,
                num_nodes, coreness, args.hybrid_beta, args.hybrid_lambda
            )
            prec_hybrid = compute_per_node_precision(recs_hybrid, test_edge_dict, args.K)
            run_dfs.append(build_df(prec_hybrid, 'Hybrid'))
            all_recs_for_eg['Hybrid'] = recs_hybrid

        all_dfs.append(pd.concat(run_dfs, ignore_index=True))

        # Keep inputs for per-run EG computation (run after coreness bins are assigned)
        per_run_eg_inputs.append({
            'run': training_seed,
            'train_edges': train_edges.cpu(),
            'test_edges': pos_test_edge.cpu(),
            'num_nodes': num_nodes,
            'coreness_np': coreness_np.copy(),
            'recs': all_recs_for_eg,
        })

        # Free GPU memory
        del h_hlgnn, model_hlgnn
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Combine all runs
    full_df = pd.concat(all_dfs, ignore_index=True)

    # Assign coreness bins
    full_df, bin_labels = assign_coreness_bins(full_df, args.num_bins)

    # Save raw data
    csv_path = os.path.join(args.save_dir, f'precision_by_coreness_{args.data_name}.csv')
    full_df.to_csv(csv_path, index=False)
    print(f"\nRaw data saved to {csv_path}")

    # Compute Equality Gain by Coreness Bin — per run, then aggregate.
    print("\nComputing Equality Gain by Coreness Bin (per run)...")
    per_run_eg_dfs = []
    for rd in per_run_eg_inputs:
        print(
            f"\n--- EG training seed {rd['run']} "
            f"({len(per_run_eg_dfs) + 1}/{len(per_run_eg_inputs)}) ---"
        )
        # Bin assignments for this run: each node's bin comes from full_df's
        # global quantile binning, filtered to this run (a node's coreness
        # and thus its bin can differ between runs due to different splits).
        run_df = full_df[full_df['run'] == rd['run']]
        run_bin_assignments = {}
        for nid, b in zip(run_df['node_id'].astype(int),
                          run_df['coreness_bin'].astype(int)):
            if nid not in run_bin_assignments:
                run_bin_assignments[nid] = b

        eg_df_run = compute_eg_by_coreness_bin(
            rd['recs'], rd['coreness_np'], run_bin_assignments, bin_labels,
            rd['train_edges'], rd['test_edges'], rd['num_nodes'], args.K
        )
        eg_df_run['run'] = rd['run']
        per_run_eg_dfs.append(eg_df_run)

    eg_df_all_runs = pd.concat(per_run_eg_dfs, ignore_index=True)

    # Aggregate across runs: mean EG with SEM for error bars.
    eg_agg = (
        eg_df_all_runs
        .groupby(['model', 'coreness_bin', 'coreness_bin_label'], as_index=False)
        .agg(equality_gain=('equality_gain', 'mean'),
             eg_std=('equality_gain', 'std'),
             eg_count=('equality_gain', 'count'))
    )
    eg_agg['eg_sem'] = np.where(
        eg_agg['eg_count'] > 1,
        eg_agg['eg_std'].fillna(0.0) / np.sqrt(eg_agg['eg_count'].clip(lower=1)),
        0.0,
    )
    eg_df = eg_agg

    # Save per-run and aggregated EG data
    eg_runs_csv_path = os.path.join(
        args.save_dir, f'eg_by_coreness_{args.data_name}_per_run.csv'
    )
    eg_df_all_runs.to_csv(eg_runs_csv_path, index=False)
    print(f"\nPer-run EG data saved to {eg_runs_csv_path}")

    eg_csv_path = os.path.join(args.save_dir, f'eg_by_coreness_{args.data_name}.csv')
    eg_df.to_csv(eg_csv_path, index=False)
    print(f"Aggregated EG data saved to {eg_csv_path}")

    fingerprints = sorted(full_df['split_fingerprint'].astype(str).unique())
    if len(fingerprints) != 1:
        raise ValueError(f'Expected one fixed split fingerprint, got {fingerprints}')
    metadata = {
        'status': 'complete',
        'dataset': args.data_name,
        'methods': ['CN', 'RA', 'AA', 'LightGCN', 'HL-GNN'],
        'training_seeds': [int(args.seed + run) for run in range(args.runs)],
        'split_seed': int(args.data_split_seed),
        'split_fingerprint': fingerprints[0],
        'k': int(args.K),
        'num_bins_requested': int(args.num_bins),
        'evaluation_users': 'nodes incident to at least one test edge',
        'structural_features': 'training graph only',
        'clustering_definition': 'exact simple-undirected local clustering coefficient',
        'precision_rows': int(len(full_df)),
    }
    metadata_path = os.path.join(
        args.save_dir, f'metadata_precision_by_coreness_{args.data_name}.json'
    )
    temporary_metadata = f'{metadata_path}.tmp.{os.getpid()}'
    with open(temporary_metadata, 'w') as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    os.replace(temporary_metadata, metadata_path)

    # Plot stacked bar chart (compact bar chart version for paper)
    print("\nGenerating stacked bar chart...")
    plot_stacked_barchart(full_df, eg_df, args.save_dir, args.data_name, args.K)

    # Plot 2D heatmap per model (degree × coreness → mean precision).
    # Aggregate per-node Precision@K across runs (mean over runs in which the
    # node appears), then feed the aggregated per-node values to plot_2d_heatmap.
    print("\nGenerating heatmaps per model (aggregated across runs)...")
    for model_name in full_df['model'].unique():
        model_df = full_df[full_df['model'] == model_name].copy()
        model_df = model_df.rename(columns={'node_coreness': 'coreness'})

        # Average per-node precision across runs; keep degree & coreness
        # (deterministic per node given a fixed split) from the first row.
        agg_df = (
            model_df
            .groupby('node_id', as_index=False)
            .agg(precision=('precision', 'mean'),
                 degree=('degree', 'first'),
                 coreness=('coreness', 'first'))
        )

        plot_2d_heatmap(
            agg_df, model_name, args.data_name, args.K,
            args.num_degree_bins, args.num_coreness_bins, args.save_dir
        )

    # Print summary table
    print("\n" + "=" * 80)
    print(f"SUMMARY: Mean Precision@{args.K} by Coreness Group")
    print("=" * 80)

    for model_name in full_df['model'].unique():
        print(f"\n  {model_name}:")
        model_df = full_df[full_df['model'] == model_name]

        for b in sorted(full_df['coreness_bin'].unique()):
            bin_df = model_df[model_df['coreness_bin'] == b]
            if len(bin_df) > 0:
                mean_p = bin_df['precision'].mean()
                std_p = bin_df['precision'].std()
                n = bin_df['node_id'].nunique()
                bl = bin_labels.get(b, str(b))
                print(f"    Bin {bl}: P@{args.K}={mean_p:.4f} ± {std_p:.4f}  (n={n} nodes)")

    # Global precision (sanity check)
    print(f"\n  Global Precision@{args.K}:")
    for model_name in full_df['model'].unique():
        mean_p = full_df[full_df['model'] == model_name]['precision'].mean()
        print(f"    {model_name}: {mean_p:.4f}")

    print("\n Per-Coreness Precision Breakdown complete!")


if __name__ == "__main__":
    main()
