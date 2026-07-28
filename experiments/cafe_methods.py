"""
Shared CAFE training / scoring helpers.

This module factors out the training + hybrid-scoring pipeline used by
experiments/run_cafe.py so that the new journal-revision experiments
(stratified evaluation, significance runs, runtime benchmark, closeness
mechanism analysis) all use EXACTLY the same code path as the main results.

RA computation and the mobility multiplier are imported from run_cafe.py
(single source of truth -- do not duplicate the formulas here).

All functions assume the same conventions as run_cafe.py:
  - undirected graphs, train edges as [E, 2] tensor
  - HLGNN encoder + MLP predictor ('two_arg' scoring)
  - sparse RA with log1p + max normalization
  - mobility boost multiplier clamped to [1, 5]
"""

import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.abspath(os.path.join(_current_dir, '..'))
for _p in [_parent_dir, _current_dir, os.path.join(_parent_dir, 'backbone')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_cafe import compute_ra_scores_sparse, compute_mobility_multipliers  # noqa: E402
from core import train_inprocessing_model, str2bool  # noqa: E402
from core.evaluation import make_predictor_score_fn  # noqa: E402
from fairness_methods.inprocessing import (  # noqa: E402
    FairnessAwareModel,
    apply_coreness_inverse_aggregation,
)


# ---------------------------------------------------------------------------
# Common CLI arguments (dataset / backbone / training), identical defaults
# to run_cafe.py so results are comparable across all revision experiments.
# ---------------------------------------------------------------------------

def add_common_args(parser):
    # Device & Dataset
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--data_name', type=str, default='ego-facebook')
    parser.add_argument('--data_path', type=str, default='../data/SNAP')
    parser.add_argument('--val_ratio', type=float, default=0.05)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--split_file', type=str, default=None)

    # Training
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16384)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--eval_skip_test', type=str2bool, default=False,
                        help='Skip test P@10 during training-time evals '
                             '(selection uses validation only; final metrics unaffected)')
    parser.add_argument('--eval_max_nodes', type=int, default=0,
                        help='Cap validation P@10 to a fixed random node subset during '
                             'training-time evals (0 = all nodes)')
    parser.add_argument('--skip_final_comprehensive_eval', type=str2bool,
                        default=False,
                        help='Skip the discarded post-training metric battery; '
                             'the calling runner performs its own evaluation')
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--neg_sampler', type=str, default='global')
    parser.add_argument('--num_neg', type=int, default=1)

    # Model architecture (HL-GNN backbone)
    parser.add_argument('--encoder', type=str, default='HLGNN')
    parser.add_argument('--predictor', type=str, default='MLP')
    parser.add_argument('--gnn_num_layers', type=int, default=15)
    parser.add_argument('--mlp_num_layers', type=int, default=2)
    parser.add_argument('--emb_hidden_channels', type=int, default=256)
    parser.add_argument('--gnn_hidden_channels', type=int, default=256)
    parser.add_argument('--mlp_hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--grad_clip_norm', type=float, default=2.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--init', type=str, default='KI')
    parser.add_argument('--train_node_emb', type=str2bool, default=True)
    parser.add_argument('--use_node_feats', type=str2bool, default=False)
    parser.add_argument('--pretrain_emb', type=str, default=None)
    parser.add_argument('--loss_func', type=str, default='AUC')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--eval_metric', type=str, default='all')
    parser.add_argument('--coreness_alpha', type=float, default=1.0)
    parser.add_argument('--penalty_reduction', type=str, default='mean',
                        choices=['mean', 'sum'],
                        help='Reduction used by the in-processing fairness penalty')

    # Evaluation
    parser.add_argument('--fairness_mode', type=str, default='indegree')
    parser.add_argument('--eval_chunk_size', type=int, default=1024)
    parser.add_argument('--eval_batch_cands', type=int, default=4096)
    parser.add_argument('--closeness_seed', type=int, default=42,
                        help='NetworKit RNG seed for approximate closeness')
    parser.add_argument('--closeness_sample_size', type=int, default=0,
                        help='Approximate-closeness pivots (0 = adaptive)')
    parser.add_argument('--closeness_epsilon', type=float, default=0.1)
    parser.add_argument('--source_manifest', type=str, default=None,
                        help='Optional egfix source-hash manifest for provenance')

    parser.add_argument('--seed', type=int, default=42)
    return parser


# ---------------------------------------------------------------------------
# Graph preprocessing
# ---------------------------------------------------------------------------

def build_train_adj_csr(train_edges, num_nodes):
    """Symmetric binary CSR adjacency from a [E, 2] train edge tensor."""
    row = train_edges[:, 0].cpu().numpy()
    col = train_edges[:, 1].cpu().numpy()
    all_row = np.concatenate([row, col])
    all_col = np.concatenate([col, row])
    vals = np.ones_like(all_row)
    adj_sp = sp.csr_matrix((vals, (all_row, all_col)), shape=(num_nodes, num_nodes))
    adj_sp.data = np.ones_like(adj_sp.data)  # binarize duplicates
    return adj_sp


def build_ra_sparse(adj_sp):
    """Normalized sparse RA score matrix (delegates to run_cafe.py)."""
    return compute_ra_scores_sparse(adj_sp)


def mobility_multiplier_vector(coreness_np, lam, coreness_alpha=1.0):
    """
    Per-node score multiplier (1 + lam * boost), clamped to [1, 5],
    identical to run_cafe.py's inner loop. Returns None when lam <= 0.
    """
    if lam is None or lam <= 0:
        return None
    mob_base = compute_mobility_multipliers(coreness_np, coreness_alpha)
    mult_vec = 1.0 + lam * torch.from_numpy(mob_base).float()
    return torch.clamp(mult_vec, 1.0, 5.0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_cafe_model(data, split_edge, args, degrees, coreness, device,
                     mode='none', pen_val=0.0):
    """
    Train an HL-GNN with optional coreness penalization, mirroring
    run_cafe.py's outer loop.

    Args:
        mode: 'none'    -> vanilla HL-GNN (no penalty)
              'inproc'  -> loss penalty with lambda_fair = pen_val
              'preproc' -> coreness-inverse adjacency modification, alpha = pen_val
        pen_val: penalization strength (ignored for mode='none')

    Returns:
        dict with keys:
          model      : trained FairnessAwareModel (best-epoch weights loaded)
          h_emb      : node embeddings [N, D] on device
          score_fn   : predictor scoring function (two_arg)
          best_epoch : best epoch found by early stopping
          train_time : wall-clock training seconds (excludes embedding extraction)
    """
    num_nodes = data.num_nodes

    original_adj_t = None
    if mode == 'preproc':
        original_adj_t = data.adj_t
        data.adj_t = apply_coreness_inverse_aggregation(
            data.adj_t, coreness, alpha=pen_val
        )
        effective_method = 'none'
        effective_lambda = 0.0
        print(f"Applied coreness-inverse pre-processing (alpha={pen_val})")
    elif mode == 'inproc':
        effective_method = 'coreness_inverse_inproc'
        effective_lambda = pen_val
    elif mode == 'none':
        effective_method = 'none'
        effective_lambda = 0.0
    else:
        raise ValueError(f"Unknown mode: {mode}")

    model = FairnessAwareModel(
        fairness_method=effective_method,
        lambda_fair=effective_lambda,
        num_nodes=num_nodes,
        lr=args.lr,
        dropout=args.dropout,
        grad_clip_norm=args.grad_clip_norm,
        gnn_num_layers=args.gnn_num_layers,
        mlp_num_layers=args.mlp_num_layers,
        emb_hidden_channels=args.emb_hidden_channels,
        gnn_hidden_channels=args.gnn_hidden_channels,
        mlp_hidden_channels=args.mlp_hidden_channels,
        num_node_feats=0,
        gnn_encoder_name=args.encoder,
        predictor_name=args.predictor,
        loss_func=args.loss_func,
        optimizer_name=args.optimizer,
        device=device,
        use_node_feats=args.use_node_feats,
        train_node_emb=args.train_node_emb,
        pretrain_emb=args.pretrain_emb,
        alpha=args.alpha,
        init=args.init,
        coreness_alpha=args.coreness_alpha,
        penalty_reduction=args.penalty_reduction,
    )
    model.param_init()

    t0 = time.perf_counter()
    model, best_epoch, _ = train_inprocessing_model(
        model, data, split_edge, args, degrees,
        patience=args.patience,
        coreness=coreness,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_time = time.perf_counter() - t0
    print(f"Training complete (mode={mode}, pen={pen_val}, "
          f"best epoch: {best_epoch}, {train_time:.1f}s)")

    # Extract embeddings using the adjacency the model was trained on
    # (for preproc this is the MODIFIED adjacency, consistent with run_cafe.py)
    model.encoder.eval()
    model.predictor.eval()
    with torch.no_grad():
        input_feat = model.create_input_feat(data)
        h_emb = model.encoder(input_feat, data.adj_t, data.edge_weight)

    # Restore adjacency AFTER embedding extraction
    if original_adj_t is not None:
        data.adj_t = original_adj_t

    score_fn = make_predictor_score_fn(model.predictor, predictor_type='two_arg')

    return {
        'model': model,
        'h_emb': h_emb,
        'score_fn': score_fn,
        'best_epoch': best_epoch,
        'train_time': train_time,
    }


# ---------------------------------------------------------------------------
# Hybrid scoring / recommendation generation
# ---------------------------------------------------------------------------

def build_cafe_recs(h_emb, score_fn, eval_nodes, num_nodes, device,
                    train_adj_csr, ra_sparse=None, beta=0.0, mult_vec=None,
                    k=100, chunk_size=1024, verbose=True):
    """
    Chunked top-k recommendation generation with the CAFE scoring rule:

        final(u, v) = (GNN(u, v) + beta * RA(u, v)) * mult_vec[v]

    Mirrors run_cafe.py's inner loop exactly (train-neighbor + self masking,
    sparse RA addition, clamped multiplier applied to columns/targets).

    Args:
        eval_nodes: sorted list of source node ids to evaluate
        ra_sparse: normalized sparse RA matrix (required if beta != 0)
        mult_vec: [N] float tensor from mobility_multiplier_vector(), or None
        k: number of recommendations per node

    Returns:
        recs_by_u: dict {node_id: [top-k recommended node ids]}
    """
    recs_by_u = {}
    num_eval = len(eval_nodes)
    num_chunks = (num_eval + chunk_size - 1) // chunk_size

    for chunk_i in range(num_chunks):
        chunk_start = chunk_i * chunk_size
        chunk_end = min(chunk_start + chunk_size, num_eval)
        chunk_node_ids = eval_nodes[chunk_start:chunk_end]
        B = len(chunk_node_ids)
        node_ids_np = np.array(chunk_node_ids)

        if verbose and (chunk_i % 5 == 0 or chunk_i == num_chunks - 1):
            print(f"    Chunk {chunk_i + 1}/{num_chunks} ({B} nodes)", flush=True)

        batch_tensor = torch.tensor(chunk_node_ids, dtype=torch.long, device=device)

        # 1. GNN scores via trained predictor
        with torch.no_grad():
            h_batch = h_emb[batch_tensor]
            scores = score_fn(h_batch, h_emb).cpu()  # [B, N]

        # 2. Sparse RA addition
        if beta != 0 and ra_sparse is not None:
            ra_chunk_coo = ra_sparse[node_ids_np].tocoo()
            if ra_chunk_coo.nnz > 0:
                ra_r = torch.from_numpy(ra_chunk_coo.row.astype(np.int64))
                ra_c = torch.from_numpy(ra_chunk_coo.col.astype(np.int64))
                ra_v = torch.from_numpy(ra_chunk_coo.data.astype(np.float32))
                scores[ra_r, ra_c] += beta * ra_v

        # 3. Mobility boost multiplier (targets)
        if mult_vec is not None:
            scores = scores * mult_vec.unsqueeze(0)

        # 4. Mask self-loops + train neighbors
        self_cols = torch.tensor(chunk_node_ids, dtype=torch.long)
        scores[torch.arange(B), self_cols] = float('-inf')
        adj_chunk = train_adj_csr[node_ids_np]
        nb_r, nb_c = adj_chunk.nonzero()
        scores[nb_r, nb_c] = float('-inf')

        # 5. Top-k
        topk_vals, topk_idx = torch.topk(scores, k, dim=1)
        for i in range(B):
            recs_by_u[chunk_node_ids[i]] = topk_idx[i].tolist()

    return recs_by_u


def build_heuristic_recs(score_sparse, train_adj_csr, eval_nodes, k,
                         chunk_size=1024, verbose=True):
    """
    Top-k recommendations from a precomputed sparse score matrix (e.g. pure RA),
    with the same masking conventions as build_cafe_recs.
    """
    recs_by_u = {}
    num_eval = len(eval_nodes)
    num_chunks = (num_eval + chunk_size - 1) // chunk_size

    for chunk_i in range(num_chunks):
        chunk_start = chunk_i * chunk_size
        chunk_end = min(chunk_start + chunk_size, num_eval)
        chunk_node_ids = eval_nodes[chunk_start:chunk_end]
        B = len(chunk_node_ids)
        node_ids_np = np.array(chunk_node_ids)

        if verbose and (chunk_i % 10 == 0 or chunk_i == num_chunks - 1):
            print(f"    Chunk {chunk_i + 1}/{num_chunks} ({B} nodes)", flush=True)

        scores = torch.from_numpy(score_sparse[node_ids_np].toarray()).float()

        self_cols = torch.tensor(chunk_node_ids, dtype=torch.long)
        scores[torch.arange(B), self_cols] = float('-inf')
        adj_chunk = train_adj_csr[node_ids_np]
        nb_r, nb_c = adj_chunk.nonzero()
        scores[nb_r, nb_c] = float('-inf')

        topk_vals, topk_idx = torch.topk(scores, k, dim=1)
        for i in range(B):
            recs_by_u[chunk_node_ids[i]] = topk_idx[i].tolist()

    return recs_by_u


def get_eval_nodes(pos_test_edge):
    """Sorted list of nodes with >= 1 test edge (either endpoint)."""
    return sorted(torch.unique(pos_test_edge.view(-1)).cpu().tolist())


def validate_recommendations(recs_by_u, eval_nodes, num_nodes, k,
                             train_adj_csr):
    """Fail fast unless a recommendation artifact obeys the top-K contract.

    The check is intentionally strict because these recommendation lists are
    the reusable boundary between model training and the corrected EG metric
    implementation.
    """
    expected_sources = {int(u) for u in eval_nodes}
    actual_sources = {int(u) for u in recs_by_u}
    if actual_sources != expected_sources:
        missing = sorted(expected_sources - actual_sources)[:5]
        extra = sorted(actual_sources - expected_sources)[:5]
        raise ValueError(
            f"Recommendation source mismatch: missing={missing}, extra={extra}"
        )

    for raw_u, raw_targets in recs_by_u.items():
        u = int(raw_u)
        targets = [int(v) for v in raw_targets]
        if len(targets) != k:
            raise ValueError(
                f"Source {u} has {len(targets)} recommendations; expected {k}"
            )
        if len(set(targets)) != k:
            raise ValueError(f"Source {u} has duplicate recommendations")
        if any(v < 0 or v >= num_nodes for v in targets):
            raise ValueError(f"Source {u} has an out-of-range target")
        if u in targets:
            raise ValueError(f"Source {u} recommends itself")
        if any(train_adj_csr[u, v] != 0 for v in targets):
            raise ValueError(
                f"Source {u} recommends an observed training neighbor"
            )

    return {
        'num_sources': len(expected_sources),
        'k': int(k),
        'passed': True,
    }
