"""
Degree × Coreness Heatmap of Precision@K

For each model (RA, LightGCN, HL-GNN), produces ONE plot:
    heatmap_{model}_{dataset}_k{value}.pdf

Cells are the mean Precision@K of nodes in (degree_bin, coreness_bin).

Usage:
    python analysis/run_coreness_degree_heatmap.py --data_name ego-facebook --runs 1 --k 10
"""

import os
import sys
import argparse
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

for path in [parent_dir, backbone_dir, fairness_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from core.utils import set_seed, str2bool
from core.data_loader import load_dataset
from core.training import train_model
from core.evaluation import make_predictor_score_fn
from backbone.model import BaseModel
from baselines.baseline_wrapper import BaselineWrapper
from metrics.coreness import compute_coreness


# ============================================================
# Argument parsing
# ============================================================

def argument():
    parser = argparse.ArgumentParser(
        description='Degree × Coreness Precision@K Heatmap'
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
    parser.add_argument('--alpha', type=float, default=0.5)

    # Analysis
    parser.add_argument('--k', type=int, default=10,
                        help='Precision@K cutoff')
    parser.add_argument('--num_degree_bins', type=int, default=4,
                        help='Number of degree groups (default 4 = quartiles)')
    parser.add_argument('--num_coreness_bins', type=int, default=5,
                        help='Number of coreness bins within each degree group')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='plots/coreness_heatmap')

    # Misc
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_metric', type=str, default='all')
    parser.add_argument('--fairness_mode', type=str, default='indegree')
    parser.add_argument('--use_node_feats', type=str2bool, default=False)
    parser.add_argument('--train_node_emb', type=str2bool, default=True)
    parser.add_argument('--encoder', type=str, default='HLGNN')
    parser.add_argument('--split_file', type=str, default='')

    return parser.parse_args()


# ============================================================
# Model training
# ============================================================

def train_lightgcn(data, split_edge, num_nodes, device, args):
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


def train_hlgnn(data, split_edge, num_nodes, device, args):
    print("\n" + "=" * 60)
    print("TRAINING: HL-GNN (15-layer)")
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

    return h, model.predictor


# ============================================================
# RA sparse computation
# ============================================================

def compute_ra_scores_sparse(adj_matrix):
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1.0
    inv_deg = sp.diags(1.0 / degrees)

    ra_matrix = (adj_matrix.dot(inv_deg)).dot(adj_matrix.T)
    ra_matrix = ra_matrix.tocsr()

    ra_matrix.data = np.log1p(ra_matrix.data)
    max_ra = ra_matrix.data.max() if len(ra_matrix.data) > 0 else 1.0
    if max_ra > 0:
        ra_matrix.data /= max_ra

    nnz = ra_matrix.nnz
    nbytes = (ra_matrix.data.nbytes + ra_matrix.indices.nbytes + ra_matrix.indptr.nbytes) / 1e6
    print(f"  RA sparse: {nnz} nonzeros, {nbytes:.1f} MB")
    return ra_matrix


# ============================================================
# Recommendation building — chunked, with sparse masking
# ============================================================

def build_recs_chunked(test_source_nodes, num_nodes, k, train_adj_csr, device,
                       h_emb=None, score_fn=None, score_matrix_sparse=None):
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

        self_cols = torch.tensor(chunk_node_ids, dtype=torch.long)
        scores[torch.arange(B), self_cols] = float('-inf')

        adj_chunk = train_adj_csr[node_ids_np]
        nb_r, nb_c = adj_chunk.nonzero()
        scores[nb_r, nb_c] = float('-inf')

        topk_vals, topk_idx = torch.topk(scores, k, dim=1)
        for i in range(B):
            recs_by_u[chunk_node_ids[i]] = topk_idx[i].tolist()

    return recs_by_u


# ============================================================
# Per-node precision
# ============================================================

def compute_per_node_precision(recs_by_u, test_edges, k):
    test_edge_set = set()
    for u, v in test_edges.tolist():
        test_edge_set.add((u, v))
        test_edge_set.add((v, u))

    node_precision = {}
    for u, vs in recs_by_u.items():
        topk = vs[:k]
        if not topk:
            continue
        hits = sum(1 for v in topk if (u, v) in test_edge_set)
        node_precision[u] = hits / len(topk)

    return node_precision


# ============================================================
# Heatmap
# ============================================================

def build_node_dataframe(node_precision, degrees_np, coreness_np):
    rows = []
    for node_id, prec in node_precision.items():
        rows.append({
            'node_id': node_id,
            'degree': float(degrees_np[node_id]),
            'coreness': float(coreness_np[node_id]),
            'precision': prec,
        })
    return pd.DataFrame(rows)


def assign_quantile_bins(series, num_bins, label_prefix=''):
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
    df = df.copy()

    df['degree_bin'], deg_label_map = assign_quantile_bins(
        df['degree'], num_degree_bins, label_prefix='deg '
    )
    df['coreness_bin'], cor_label_map = assign_quantile_bins(
        df['coreness'], num_coreness_bins
    )

    degree_groups = sorted(df['degree_bin'].dropna().unique())
    coreness_groups = sorted(df['coreness_bin'].dropna().unique())

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

    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn',
                   vmin=np.nanmin(matrix) if not np.all(np.isnan(matrix)) else 0,
                   vmax=np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 1)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f'Mean Precision@{k}', fontsize=11)

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
        f'{model_name} — {dataset_name}',
        fontsize=11, fontweight='bold'
    )

    plt.tight_layout()
    fname = f'heatmap_{model_name.replace(" ", "_")}_{dataset_name}_k{k}.pdf'
    path = os.path.join(save_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def average_precision_across_runs(per_run_precisions):
    """
    Average each node's Precision@K across runs.

    Args:
        per_run_precisions: list of dicts, one per run: {node_id: precision_float}

    Returns:
        dict {node_id: mean_precision} over runs in which the node appears.
        (With a deterministic split, a node either appears in all runs or none.)
    """
    sums = {}
    counts = {}
    for run_prec in per_run_precisions:
        for nid, p in run_prec.items():
            sums[nid] = sums.get(nid, 0.0) + float(p)
            counts[nid] = counts.get(nid, 0) + 1
    return {nid: sums[nid] / counts[nid] for nid in sums}


def run_analysis(node_precision, degrees_np, coreness_np,
                 model_name, dataset_name, k,
                 num_degree_bins, num_coreness_bins, save_dir):
    df = build_node_dataframe(node_precision, degrees_np, coreness_np)
    print(f"\n  Nodes with precision computed: {len(df)}")
    print(f"  Degree  — min: {df['degree'].min():.0f},  max: {df['degree'].max():.0f},  "
          f"mean: {df['degree'].mean():.1f}")
    print(f"  Coreness — min: {df['coreness'].min():.0f}, max: {df['coreness'].max():.0f}, "
          f"mean: {df['coreness'].mean():.1f}")
    print(f"  Precision@{k} — mean: {df['precision'].mean():.4f}, "
          f"std: {df['precision'].std():.4f}")

    plot_2d_heatmap(
        df, model_name, dataset_name, k,
        num_degree_bins, num_coreness_bins, save_dir
    )


# ============================================================
# Main
# ============================================================

def main():
    args = argument()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.save_dir, exist_ok=True)

    model_names = ['RA', 'LightGCN', 'HL-GNN']

    print("=" * 80)
    print("DEGREE × CORENESS PRECISION@K HEATMAP")
    print(f"Dataset:         {args.data_name}")
    print(f"Models:          {', '.join(model_names)}")
    print(f"Precision@K:     K={args.k}")
    print(f"Degree bins:     {args.num_degree_bins}")
    print(f"Coreness bins:   {args.num_coreness_bins}")
    print(f"Runs:            {args.runs}")
    print(f"Device:          {device}")
    print(f"Save dir:        {args.save_dir}")
    print("=" * 80)

    # Accumulate per-run per-node precision for each model
    per_run_precisions = {name: [] for name in model_names}

    # Degrees and coreness come from the (deterministic) training split,
    # so they are identical every run. Capture them from run 0.
    degrees_np = None
    coreness_np = None

    for run in range(args.runs):
        print(f"\n{'=' * 80}")
        print(f"RUN {run + 1}/{args.runs}")
        print(f"{'=' * 80}")

        dataset, split_edge, data, degrees = load_dataset(args)
        num_nodes = dataset._data.num_nodes if hasattr(dataset, '_data') else dataset.data.num_nodes

        train_edges_cpu = split_edge['train']['edge']
        test_edges_cpu = split_edge['test']['edge']

        row = train_edges_cpu[:, 0].numpy()
        col = train_edges_cpu[:, 1].numpy()
        all_row = np.concatenate([row, col])
        all_col = np.concatenate([col, row])
        vals = np.ones_like(all_row)
        adj_sp = sp.csr_matrix((vals, (all_row, all_col)), shape=(num_nodes, num_nodes))

        train_adj_csr = adj_sp.copy()
        train_adj_csr.data = np.ones_like(train_adj_csr.data)

        test_source_nodes = sorted(set(test_edges_cpu[:, 0].tolist()))

        print("\nComputing coreness on training graph...")
        coreness = compute_coreness(train_edges_cpu, num_nodes)
        coreness_np = coreness.cpu().numpy()
        degrees_np = degrees.cpu().numpy()

        print(f"Coreness: min={coreness_np.min():.0f}, max={coreness_np.max():.0f}, "
              f"mean={coreness_np.mean():.2f}")

        # ---- RA ----
        print("\n--- Analyzing: RA ---")
        print("  Computing RA scores (sparse)...")
        ra_sparse = compute_ra_scores_sparse(adj_sp)

        print(f"  Building top-{args.k} recommendations for {len(test_source_nodes)} test nodes...")
        recs_ra = build_recs_chunked(
            test_source_nodes, num_nodes, k=args.k,
            train_adj_csr=train_adj_csr, device=device,
            score_matrix_sparse=ra_sparse
        )
        prec_ra = compute_per_node_precision(recs_ra, test_edges_cpu, args.k)
        per_run_precisions['RA'].append(prec_ra)

        del ra_sparse, recs_ra
        import gc; gc.collect()

        # ---- LightGCN ----
        print("\n--- Analyzing: LightGCN ---")
        h_lgcn = train_lightgcn(data, split_edge, num_nodes, device, args)
        h_lgcn = h_lgcn.to(device)

        print(f"  Building top-{args.k} recommendations for {len(test_source_nodes)} test nodes...")
        recs_lgcn = build_recs_chunked(
            test_source_nodes, num_nodes, k=args.k,
            train_adj_csr=train_adj_csr, device=device,
            h_emb=h_lgcn
        )
        prec_lgcn = compute_per_node_precision(recs_lgcn, test_edges_cpu, args.k)
        per_run_precisions['LightGCN'].append(prec_lgcn)

        del h_lgcn, recs_lgcn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---- HL-GNN ----
        print("\n--- Analyzing: HL-GNN ---")
        h_hlgnn, pred_hlgnn = train_hlgnn(data, split_edge, num_nodes, device, args)
        h_hlgnn = h_hlgnn.to(device)

        score_fn = make_predictor_score_fn(pred_hlgnn, predictor_type='two_arg')

        print(f"  Building top-{args.k} recommendations for {len(test_source_nodes)} test nodes...")
        recs_hlgnn = build_recs_chunked(
            test_source_nodes, num_nodes, k=args.k,
            train_adj_csr=train_adj_csr, device=device,
            h_emb=h_hlgnn, score_fn=score_fn
        )
        prec_hlgnn = compute_per_node_precision(recs_hlgnn, test_edges_cpu, args.k)
        per_run_precisions['HL-GNN'].append(prec_hlgnn)

        del h_hlgnn, pred_hlgnn, recs_hlgnn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ============================================================
    # Average per-node precision across runs, then plot one heatmap per model.
    # ============================================================
    print(f"\n{'=' * 80}")
    print(f"Aggregating per-node Precision@{args.k} across {args.runs} run(s) "
          f"and plotting heatmaps")
    print(f"{'=' * 80}")

    for model_name in model_names:
        runs_list = per_run_precisions[model_name]
        if not runs_list:
            print(f"\n[WARN] No runs collected for {model_name}; skipping.")
            continue

        avg_precision = average_precision_across_runs(runs_list)
        print(f"\n--- {model_name}: averaged over {len(runs_list)} run(s), "
              f"{len(avg_precision)} nodes ---")

        run_analysis(
            avg_precision, degrees_np, coreness_np,
            model_name=model_name,
            dataset_name=args.data_name,
            k=args.k,
            num_degree_bins=args.num_degree_bins,
            num_coreness_bins=args.num_coreness_bins,
            save_dir=args.save_dir
        )

    print(f"\nDone. Heatmaps saved to: {args.save_dir}/")


if __name__ == "__main__":
    main()
