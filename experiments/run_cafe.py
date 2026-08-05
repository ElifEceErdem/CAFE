"""
Hybrid Penalized GNN + RA Fairness Experiment (Configs C & D).

Trains a penalized GNN (pre-processing or in-processing), then combines
GNN embeddings with Resource Allocation (RA) scores at inference time.

Configs:
  C-pre: GNN(modified adj) + beta*RA           (no mobility boost)
  C-in:  GNN(loss penalty)  + beta*RA           (no mobility boost)
  D-pre: (GNN(modified adj) + beta*RA) * boost  (with mobility boost)
  D-in:  (GNN(loss penalty)  + beta*RA) * boost  (with mobility boost)

Usage:
    python run_hybrid_penalized.py \
        --penalization_mode preproc \
        --use_mobility_boost False \
        --preproc_alpha "0.05,0.1,0.3,0.5,1.0,2.0" \
        --beta_ra "0.0,0.5,1.0,2.0" \
        --runs 3
"""

import os
import sys
import argparse
import collections
import hashlib
import json
import torch
import numpy as np
import scipy.sparse as sp
from scipy import stats as scipy_stats

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
snap_dir = os.path.abspath(os.path.join(current_dir, '../backbone'))

for path in [parent_dir, snap_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from metrics.coreness import compute_coreness
from core.data_loader import load_sensitive_attr
from core.statistics import _serialize_stats

from core import (
    set_seed, str2bool, load_dataset,
    compute_statistics, create_experiment_info,
    train_inprocessing_model
)
from core.evaluation import compute_metrics_from_recs, compute_equality_gain_metrics, make_predictor_score_fn
from fairness_methods.inprocessing import (
    FairnessAwareModel,
    apply_coreness_inverse_aggregation
)
from metrics.centrality_equality_gain import compute_all_centrality_equality_gains
from metrics.demographics import compute_demographic_metrics, compute_demographic_parity_ratio_change
from metrics.graph_utils import canonicalize_undirected_edges

def argument():
    parser = argparse.ArgumentParser(
        description='Hybrid Penalized GNN + RA Fairness Experiment (Configs C & D)'
    )

    # Device & Dataset
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--data_name', type=str, default='ego-facebook')
    parser.add_argument('--data_path', type=str, default='../../data/SNAP')
    parser.add_argument('--val_ratio', type=float, default=0.05)
    parser.add_argument('--test_ratio', type=float, default=0.1)

    # Training
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16384)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--eval_skip_test', type=str2bool, default=False,
                        help='Skip test P@10 during training-time evals '
                             '(selection uses validation only; final metrics unaffected)')
    parser.add_argument('--eval_max_nodes', type=int, default=0,
                        help='Cap validation P@10 to a fixed random node subset during '
                             'training-time evals (0 = all nodes)')
    parser.add_argument('--skip_final_comprehensive_eval', type=str2bool,
                        default=False,
                        help='Skip the post-training test metric battery; required '
                             'for validation-only configuration selection')
    parser.add_argument('--neg_sampler', type=str, default='global')
    parser.add_argument('--num_neg', type=int, default=1)

    # Model Architecture
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

    # Penalization Config
    parser.add_argument('--penalization_mode', type=str, required=True,
                        choices=['preproc', 'inproc'],
                        help='preproc: modify adjacency; inproc: loss penalty')
    parser.add_argument('--use_mobility_boost', type=str2bool, default=False,
                        help='Apply mobility boost at inference (C=False, D=True)')

    # Penalization Parameters (training-time, outer loop)
    parser.add_argument('--preproc_alpha', type=str, default='0.5,1.0,2.0',
                        help='Pre-processing adjacency modification strengths (comma-separated)')
    parser.add_argument('--lambda_fair', type=str, default='0.01,0.05,0.1,0.3',
                        help='In-processing loss penalty weights (comma-separated)')

    # Hybrid Inference Parameters (inner loop)
    parser.add_argument('--beta_ra', type=str, default='0.5,1.0,2.0',
                        help='Weight for RA score in hybrid combination (comma-separated)')
    parser.add_argument('--lambda_mobility', type=str, default='0.3',
                        help='Mobility boost strength (comma-separated, D configs only)')
    parser.add_argument('--coreness_alpha', type=float, default=1.0,
                        help='Alpha exponent for coreness-based methods')
    parser.add_argument('--penalty_reduction', type=str, default='mean',
                        choices=['mean', 'sum'],
                        help="Batch reduction of the in-processing fairness penalty "
                             "relative to the base loss (the default AUC base loss sums). "
                             "'mean' (default) = historical behaviour, penalty is "
                             "~|E+_batch| times smaller than the base loss; "
                             "'sum' = both terms reduce identically, so lambda_fair "
                             "is a per-example weight. Ignored in preproc mode.")
    parser.add_argument('--reset_seed_per_model', type=str2bool, default=False,
                        help="Re-seed with (seed + run_index) immediately before "
                             "constructing every model, so each (run, penalty value) "
                             "starts from an identical RNG state. Required for paired "
                             "comparisons across separately launched arms. Default "
                             "False preserves the historical protocol, in which one "
                             "RNG stream evolves across penalty values and runs.")

    # Evaluation
    parser.add_argument('--fairness_mode', type=str, default='indegree')
    parser.add_argument('--use_equality_gain', type=str2bool, default=True)
    parser.add_argument('--use_centrality_gain', type=str2bool, default=True)
    # Parse to a LIST at argparse time, matching run_coreness_inverse.py and
    # run_mobility_boost.py. Leaving args.centrality_types as a raw string
    # breaks every downstream consumer that reads it off `args` rather than
    # from the local split below -- core.evaluation.compute_baseline_metrics
    # does exactly that, and iterating the string yields 'p', 'a', 'g', ...
    parser.add_argument('--centrality_types',
                        type=lambda s: (['pagerank', 'closeness'] if s is None
                                        else [x.strip() for x in s.split(',')]),
                        default='pagerank,closeness')
    parser.add_argument('--selection_mode', type=str2bool, default=False,
                        help='Strict validation-only configuration-selection mode')
    parser.add_argument('--evaluation_split', type=str, default='test',
                        choices=['valid', 'test'],
                        help='Held-out split used for sources, relevance, and EG control')
    parser.add_argument('--k_values', type=str, default='10,20,50,100',
                        help='Comma-separated recommendation cutoffs to evaluate')
    parser.add_argument('--use_demographics', type=str2bool, default=True,
                        help='Compute demographic metrics when attributes exist')
    parser.add_argument('--closeness_seed', type=int, default=42,
                        help='NetworKit RNG seed for approximate closeness')
    parser.add_argument('--closeness_sample_size', type=int, default=0,
                        help='Approximate-closeness pivots (0 = adaptive)')
    parser.add_argument('--closeness_epsilon', type=float, default=0.1)
    parser.add_argument('--source_manifest', type=str, default=None,
                        help='Optional egfix source-hash manifest for provenance')

    # Memory / evaluation chunking
    parser.add_argument('--eval_chunk_size', type=int, default=1024,
                        help='Chunk size for P@K evaluation (reduce to save memory)')
    parser.add_argument('--eval_batch_cands', type=int, default=4096,
                        help='Candidate batch size in predictor scoring (reduce to save memory)')

    # Results
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--results_file', type=str, default='results_hybrid_penalized.json')
    parser.add_argument('--save_results', type=str2bool, default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_file', type=str, default=None, help='Path to pre-split dataset file')

    return parser.parse_args()


def strict_serialize(data):
    if isinstance(data, dict):
        return {k: strict_serialize(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [strict_serialize(x) for x in data]
    elif hasattr(data, 'item'):
        return float(data.item())
    elif isinstance(data, (np.ndarray, np.generic)):
        return data.tolist() if isinstance(data, np.ndarray) else float(data)
    else:
        return data


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _split_fingerprint(split_edge, num_nodes, split_names):
    """Order/orientation-invariant fingerprint of selected edge splits."""
    digest = hashlib.sha256()
    digest.update(f"num_nodes={int(num_nodes)}\n".encode())
    for split_name in split_names:
        edges = canonicalize_undirected_edges(
            split_edge[split_name]['edge'].cpu(), num_nodes
        )
        digest.update(f"{split_name}:{len(edges)}\n".encode())
        digest.update(
            edges.to(dtype=torch.int64).contiguous().numpy().astype(
                '<i8', copy=False
            ).tobytes()
        )
    return digest.hexdigest()


def compute_ra_scores_sparse(adj_matrix):
    """
    Compute Resource Allocation scores as a normalized sparse CSR matrix.
    RA(u,v) = sum(1/deg(z)) for z in shared_neighbors.
    Returns sparse CSR with log1p + max normalization applied to nonzero entries.
    """
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1.0
    inv_deg = sp.diags(1.0 / degrees)

    # A * D^-1 * A^T -- result is sparse (only pairs with shared neighbors are nonzero)
    ra_matrix = (adj_matrix.dot(inv_deg)).dot(adj_matrix.T)
    ra_matrix = ra_matrix.tocsr()

    # Normalize in-place on sparse data: log1p + max scaling
    ra_matrix.data = np.log1p(ra_matrix.data)
    max_ra = ra_matrix.data.max() if len(ra_matrix.data) > 0 else 1.0
    if max_ra > 0:
        ra_matrix.data /= max_ra

    nnz = ra_matrix.nnz
    nbytes = (ra_matrix.data.nbytes + ra_matrix.indices.nbytes + ra_matrix.indptr.nbytes) / 1e6
    print(f"RA sparse: {nnz} nonzeros, {nbytes:.1f} MB")
    return ra_matrix


def compute_mobility_multipliers(coreness_np, coreness_alpha):
    """
    Compute per-node mobility boost values based on coreness percentiles.
    Returns numpy array of shape [num_nodes].
    """
    c_ranks = scipy_stats.rankdata(coreness_np, method='average')
    c_pctl = c_ranks / len(coreness_np)
    scaling = 100.0
    denom = np.log(c_pctl * scaling + np.e)
    mobility_boost = 1.0 / (denom ** coreness_alpha)
    return mobility_boost


def main():
    args = argument()

    k_values = [int(x.strip()) for x in args.k_values.split(',') if x.strip()]
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError(f"Invalid --k_values: {args.k_values!r}")
    if len(set(k_values)) != len(k_values):
        raise ValueError("--k_values must not contain duplicates")

    # Selection is a protocol boundary, not just a convenient output label.
    # Enforce every condition needed to keep the final test set out of both
    # early-stopping diagnostics and outer-grid objectives.
    if args.selection_mode:
        if args.evaluation_split != 'valid':
            raise ValueError(
                "--selection_mode True requires --evaluation_split valid"
            )
        if k_values != [10]:
            raise ValueError(
                "--selection_mode True requires --k_values 10"
            )
        if not args.reset_seed_per_model:
            raise ValueError(
                "--selection_mode True requires --reset_seed_per_model True"
            )
        if not args.use_equality_gain or not args.use_centrality_gain:
            raise ValueError(
                "Selection requires Degree, PageRank, and Closeness EG"
            )
        if args.use_demographics:
            raise ValueError(
                "--selection_mode True requires --use_demographics False"
            )
        args.eval_skip_test = True
        args.skip_final_comprehensive_eval = True

    set_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    # Parse sweep values
    betas = [float(x) for x in args.beta_ra.split(',')]
    centrality_types = list(args.centrality_types)
    if args.selection_mode and not {'pagerank', 'closeness'}.issubset(
        set(centrality_types)
    ):
        raise ValueError(
            "Selection requires --centrality_types pagerank,closeness"
        )
    closeness_sample_size = (
        None if args.closeness_sample_size == 0
        else args.closeness_sample_size
    )
    if closeness_sample_size is not None and closeness_sample_size <= 0:
        raise ValueError("--closeness_sample_size must be 0 or positive")

    source_manifest_sha256 = None
    if args.source_manifest is not None:
        if not os.path.isfile(args.source_manifest):
            raise FileNotFoundError(
                f"Source manifest not found: {args.source_manifest}"
            )
        source_manifest_sha256 = _sha256_file(args.source_manifest)

    if args.penalization_mode == 'preproc':
        pen_values = [float(x) for x in args.preproc_alpha.split(',')]
        pen_label = 'preproc_alpha'
    else:
        pen_values = [float(x) for x in args.lambda_fair.split(',')]
        pen_label = 'lambda_fair'

    if args.use_mobility_boost:
        lambdas_mobility = [float(x) for x in args.lambda_mobility.split(',')]
    else:
        lambdas_mobility = [0.0]

    # Config name for display
    config_letter = 'D' if args.use_mobility_boost else 'C'
    config_suffix = 'pre' if args.penalization_mode == 'preproc' else 'in'
    config_name = f"{config_letter}-{config_suffix}"

    # Set args.fairness_method for compatibility with train_inprocessing_model
    if args.penalization_mode == 'preproc':
        args.fairness_method = 'coreness_inverse_preproc'
    else:
        args.fairness_method = 'coreness_inverse_inproc'

    print("=" * 80)
    print(f"HYBRID PENALIZED EXPERIMENT: Config {config_name}")
    print(f"Penalization Mode: {args.penalization_mode}")
    print(f"Mobility Boost: {args.use_mobility_boost}")
    print(f"Penalization values ({pen_label}): {pen_values}")
    print(f"Beta RA values: {betas}")
    if args.use_mobility_boost:
        print(f"Lambda mobility values: {lambdas_mobility}")
    print(f"Runs: {args.runs}")
    print(f"Evaluation split: {args.evaluation_split}")
    print(f"K values: {k_values}")
    print(f"Selection mode: {args.selection_mode}")
    print(f"Penalty reduction: {args.penalty_reduction}")
    print(f"Approximate closeness seed: {args.closeness_seed}")
    print("=" * 80)

    results_by_config = collections.defaultdict(list)
    all_results = []
    split_fingerprints = []

    for run in range(args.runs):
        print(f"\n{'=' * 80}")
        print(f"RUN {run + 1}/{args.runs}")
        print(f"{'=' * 80}")

        # === Step 1: Load data ===
        dataset, split_edge, data, degrees = load_dataset(args)
        num_nodes = dataset._data.num_nodes if hasattr(dataset, '_data') else dataset.data.num_nodes
        train_edges = split_edge['train']['edge'].to(device)
        heldout_edges = split_edge[args.evaluation_split]['edge'].to(device)
        fingerprint_splits = (
            ['train', 'valid']
            if args.evaluation_split == 'valid'
            else ['train', 'test']
        )
        split_fingerprints.append(
            _split_fingerprint(split_edge, num_nodes, fingerprint_splits)
        )

        # === Step 1.5: Load sensitive attributes ===
        sens_attr = None
        if args.use_demographics:
            try:
                sens_attr = load_sensitive_attr(
                    args.data_name, num_nodes, root=args.data_path
                )
                g0 = int((sens_attr == 0).sum())
                g1 = int((sens_attr == 1).sum())
                print(f"Loaded sensitive attributes: group_0={g0}, group_1={g1}")
                data.sens = (
                    torch.tensor(sens_attr, device=device)
                    if not torch.is_tensor(sens_attr)
                    else sens_attr.to(device)
                )
            except ValueError as e:
                print(f"WARNING: {e}")
                print("Skipping demographic metrics because sensitive "
                      "attributes could not be loaded.")

        # === Step 2: Compute coreness ===
        print("\nComputing coreness...")
        coreness = compute_coreness(train_edges, num_nodes).to(device)
        data.coreness = coreness
        c_np = coreness.cpu().numpy()

        # === Step 3: Compute RA scores (sparse) ===
        print("Computing Resource Allocation (RA) scores...")
        row = train_edges[:, 0].cpu().numpy()
        col = train_edges[:, 1].cpu().numpy()
        all_row = np.concatenate([row, col])
        all_col = np.concatenate([col, row])
        vals = np.ones_like(all_row)
        adj_sp = sp.csr_matrix((vals, (all_row, all_col)), shape=(num_nodes, num_nodes))

        ra_sparse = compute_ra_scores_sparse(adj_sp)  # sparse CSR, already normalized

        # === Step 4: Build train adjacency for masking ===
        # Build CSR adjacency mask for vectorized train-edge masking
        train_adj_csr = adj_sp.copy()
        train_adj_csr.data = np.ones_like(train_adj_csr.data)  # binary mask

        # Keep CPU copies for metric functions
        train_edges_cpu = train_edges.cpu()
        heldout_edges_cpu = heldout_edges.cpu()
        centrality_control_cache = {}

        # === Step 5: Precompute mobility base (if needed) ===
        if args.use_mobility_boost:
            mob_base = compute_mobility_multipliers(c_np, args.coreness_alpha)
            mob_base_tensor = torch.from_numpy(mob_base).float()

        run_res = {}

        # === OUTER LOOP: Over penalization values (requires retraining) ===
        for pen_val in pen_values:
            print(f"\n{'=' * 60}")
            print(f"TRAINING: {pen_label}={pen_val}")
            print(f"{'=' * 60}")

            # --- Setup penalized GNN training ---
            original_adj_t = None
            if args.penalization_mode == 'preproc':
                original_adj_t = data.adj_t
                data.adj_t = apply_coreness_inverse_aggregation(
                    data.adj_t, coreness, alpha=pen_val
                )
                effective_method = 'none'
                effective_lambda = 0.0
                print(f"Applied coreness-inverse pre-processing (alpha={pen_val})")
            else:
                effective_method = 'coreness_inverse_inproc'
                effective_lambda = pen_val

            # OPT-IN: re-seed before every model so a given (run, penalty value)
            # starts from an identical RNG state regardless of what was trained
            # before it. Without it the stream carries over across penalty
            # values and runs, so any change that shifts the early-stopping
            # epoch desynchronises every subsequent configuration and silently
            # breaks paired comparisons across arms.
            # Default False: the historical protocol is unchanged unless asked.
            if args.reset_seed_per_model:
                model_seed = args.seed + run
                set_seed(model_seed)
                print(f"[seed] model seed = {model_seed} "
                      f"(args.seed={args.seed} + run={run})")

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
                penalty_reduction=args.penalty_reduction
            )
            model.param_init()

            # Temporarily set args.lambda_fair for train_inprocessing_model compatibility
            saved_lambda_fair = args.lambda_fair
            args.lambda_fair = pen_val

            model, best_epoch, _ = train_inprocessing_model(
                model, data, split_edge, args, degrees,
                patience=args.patience,
                coreness=coreness
            )

            args.lambda_fair = saved_lambda_fair
            print(f"Training complete (best epoch: {best_epoch})")

            # --- Extract embeddings (using current data.adj_t) ---
            # For preproc: this is the MODIFIED adjacency (consistent with training)
            model.encoder.eval()
            model.predictor.eval()
            with torch.no_grad():
                input_feat = model.create_input_feat(data)
                h_emb = model.encoder(input_feat, data.adj_t, data.edge_weight)

            # Create scoring function using trained predictor (MLP)
            score_fn = make_predictor_score_fn(model.predictor, predictor_type='two_arg')
            print("Using trained predictor (MLP) for GNN scoring")

            # Restore adjacency AFTER embedding extraction
            if original_adj_t is not None:
                data.adj_t = original_adj_t

            # Evaluate only nodes active in the requested held-out split.
            eval_nodes = torch.unique(heldout_edges.view(-1)).cpu().tolist()
            eval_nodes_sorted = sorted(eval_nodes)
            print(
                f"Evaluating {len(eval_nodes_sorted)} "
                f"{args.evaluation_split} nodes (out of {num_nodes} total)",
                flush=True,
            )

            # === INNER LOOP: Over (beta, lambda_mobility) pairs ===
            # The learned GNN score matrix is identical for all 30 inference
            # configurations attached to this trained penalty arm.  Score each
            # source chunk once, then apply RA/boost variants to clones of that
            # chunk.  This preserves exact rankings while avoiding 30 repeated
            # predictor passes (essential for Pokec-n and Brightkite).
            combo_specs = []
            for beta in betas:
                for lam in lambdas_mobility:
                    combo_key = f"{pen_label}_{pen_val}_beta_{beta}_lam_{lam}"

                    if args.use_mobility_boost and lam > 0:
                        mult_vec = 1.0 + lam * mob_base_tensor
                        mult_vec = torch.clamp(mult_vec, 1.0, 5.0)
                    else:
                        mult_vec = None
                    combo_specs.append({
                        'key': combo_key,
                        'beta': beta,
                        'lambda_mobility': lam,
                        'mult_vec': mult_vec,
                    })

            recs_by_combo = {spec['key']: {} for spec in combo_specs}
            chunk_size = 1024
            k = max(k_values)
            num_eval = len(eval_nodes_sorted)
            num_chunks = (num_eval + chunk_size - 1) // chunk_size
            print(
                f"Generating recommendations for {len(combo_specs)} "
                "inference configurations"
            )

            for chunk_i in range(num_chunks):
                chunk_start = chunk_i * chunk_size
                chunk_end = min(chunk_start + chunk_size, num_eval)
                chunk_node_ids = eval_nodes_sorted[chunk_start:chunk_end]
                B = len(chunk_node_ids)

                if chunk_i % 5 == 0 or chunk_i == num_chunks - 1:
                    print(
                        f"    Chunk {chunk_i + 1}/{num_chunks} ({B} nodes)",
                        flush=True,
                    )

                batch_tensor = torch.tensor(
                    chunk_node_ids, dtype=torch.long, device=device
                )
                node_ids_np = np.asarray(chunk_node_ids)
                with torch.no_grad():
                    h_batch = h_emb[batch_tensor]
                    gnn_scores = score_fn(h_batch, h_emb).cpu()

                ra_chunk_coo = ra_sparse[node_ids_np].tocoo()
                if ra_chunk_coo.nnz > 0:
                    ra_r = torch.from_numpy(
                        ra_chunk_coo.row.astype(np.int64)
                    )
                    ra_c = torch.from_numpy(
                        ra_chunk_coo.col.astype(np.int64)
                    )
                    ra_v = torch.from_numpy(
                        ra_chunk_coo.data.astype(np.float32)
                    )
                else:
                    ra_r = ra_c = ra_v = None

                self_cols = torch.tensor(
                    chunk_node_ids, dtype=torch.long
                )
                adj_chunk = train_adj_csr[node_ids_np]
                nb_r_np, nb_c_np = adj_chunk.nonzero()

                for spec in combo_specs:
                    final_scores = gnn_scores.clone()
                    if spec['beta'] != 0 and ra_r is not None:
                        final_scores[ra_r, ra_c] += spec['beta'] * ra_v
                    if spec['mult_vec'] is not None:
                        final_scores *= spec['mult_vec'].unsqueeze(0)

                    final_scores[
                        torch.arange(B), self_cols
                    ] = float('-inf')
                    final_scores[nb_r_np, nb_c_np] = float('-inf')
                    _, topk_idx = torch.topk(final_scores, k, dim=1)
                    combo_recs = recs_by_combo[spec['key']]
                    for i, source in enumerate(chunk_node_ids):
                        combo_recs[source] = topk_idx[i].tolist()

            for combo_index, spec in enumerate(combo_specs, start=1):
                combo_key = spec['key']
                recs_by_u = recs_by_combo[combo_key]
                print(
                    f"\n  Metrics {combo_index}/{len(combo_specs)}: "
                    f"{combo_key}"
                )
                res = compute_metrics_from_recs(
                    recs_by_u, heldout_edges_cpu, num_nodes,
                    args.fairness_mode, k_values=k_values
                )

                if args.use_equality_gain:
                    eq = compute_equality_gain_metrics(
                        recs_by_u, train_edges_cpu, heldout_edges_cpu,
                        num_nodes, k_values=k_values
                    )
                    res.update(eq)

                if args.use_centrality_gain:
                    cent = compute_all_centrality_equality_gains(
                        recs_by_u, train_edges_cpu, heldout_edges_cpu,
                        num_nodes, k_values=k_values,
                        centrality_types=centrality_types,
                        use_sampling_closeness=True,
                        closeness_random_seed=args.closeness_seed,
                        closeness_sample_size=closeness_sample_size,
                        closeness_epsilon=args.closeness_epsilon,
                        control_cache=centrality_control_cache,
                    )
                    res.update(cent)

                if sens_attr is not None:
                    features = getattr(data, 'x', None)
                    demo_metrics = compute_demographic_metrics(
                        recs_by_u=recs_by_u,
                        train_edges=train_edges_cpu,
                        test_edges=heldout_edges_cpu,
                        num_nodes=num_nodes,
                        sens=sens_attr,
                        h_emb=h_emb.cpu(),
                        features=(
                            features.cpu()
                            if features is not None else None
                        ),
                        adj=None,
                        k=10
                    )
                    res.update(demo_metrics)
                    res['Delta_DPR'] = (
                        compute_demographic_parity_ratio_change(
                            train_edges_cpu, recs_by_u, num_nodes,
                            sens_attr, k=10
                        )
                    )

                run_res[combo_key] = res
                results_by_config[combo_key].append(res)

        run_res['baseline'] = {}  # Placeholder for report compatibility
        all_results.append(run_res)

        # Persist completed runs so a wall-clock kill never loses everything.
        if args.save_results:
            with open(args.results_file + '.partial.json', 'w') as f:
                json.dump({'completed_runs': run + 1,
                           'individual_runs': strict_serialize(all_results)},
                          f, indent=2)
            print(f"Partial results ({run + 1} run(s)) saved to "
                  f"{args.results_file}.partial.json")

    # === Aggregate and Save ===
    if args.save_results:
        print("\nAggregating results across runs...")

        unique_split_fingerprints = sorted(set(split_fingerprints))
        if len(unique_split_fingerprints) != 1:
            raise RuntimeError(
                "The held-out graph changed across nominally fixed-split "
                f"runs: {unique_split_fingerprints}"
            )

        final_lambda_results = {}
        for key, run_list in results_by_config.items():
            final_lambda_results[key] = compute_statistics(run_list)

        experiment_info = create_experiment_info(args, f'Hybrid_Penalized_{config_name}')
        experiment_info['penalization_mode'] = args.penalization_mode
        experiment_info['use_mobility_boost'] = args.use_mobility_boost
        experiment_info['config_name'] = config_name
        experiment_info['selection_mode'] = bool(args.selection_mode)
        experiment_info['evaluation_split'] = args.evaluation_split
        # Compatibility alias for early egfix tooling.  Both fields describe
        # the same held-out metric boundary.
        experiment_info['metric_split'] = args.evaluation_split
        experiment_info['test_metrics_evaluated'] = (
            args.evaluation_split == 'test'
        )
        experiment_info['skip_final_comprehensive_eval'] = bool(
            args.skip_final_comprehensive_eval
        )
        experiment_info['reset_seed_per_model'] = bool(
            args.reset_seed_per_model
        )
        experiment_info['split_seed'] = int(
            getattr(args, 'data_split_seed', 42)
        )
        experiment_info['split_fingerprint'] = unique_split_fingerprints[0]
        experiment_info['k_values'] = k_values
        experiment_info['use_demographics'] = bool(args.use_demographics)
        experiment_info['centrality_types'] = centrality_types
        experiment_info['closeness_random_seed'] = int(args.closeness_seed)
        experiment_info['closeness_sample_size'] = (
            0 if closeness_sample_size is None else int(closeness_sample_size)
        )
        experiment_info['closeness_epsilon'] = float(args.closeness_epsilon)
        experiment_info['source_manifest_sha256'] = source_manifest_sha256
        experiment_info['model_seeds'] = [
            int(args.seed + run) for run in range(args.runs)
        ]
        experiment_info['grid'] = {
            pen_label: pen_values,
            'beta': betas,
            'lambda_mobility': lambdas_mobility,
        }

        split_sha256 = None
        if str(args.data_name).lower() == 'brightkite':
            split_path = args.split_file
            if not split_path:
                split_path = os.path.join(
                    args.data_path, 'brightkite', 'temporal_split.pt'
                )
            if str(split_path).lower() != 'random':
                if not os.path.isfile(split_path):
                    raise FileNotFoundError(
                        f"Brightkite split artifact missing: {split_path}"
                    )
                split_sha256 = _sha256_file(split_path)
        experiment_info['split_sha256'] = split_sha256

        final_output = {
            'experiment_info': experiment_info,
            'baseline': {'statistics': {}},
            'lambda_results': {}
        }

        for key, stats in final_lambda_results.items():
            final_output['lambda_results'][key] = {
                'statistics': _serialize_stats(stats)
            }

        final_output['individual_runs'] = strict_serialize(all_results)

        final_temp = f"{args.results_file}.tmp.{os.getpid()}"
        with open(final_temp, 'w') as f:
            json.dump(final_output, f, indent=2)
        os.replace(final_temp, args.results_file)

        partial_path = args.results_file + '.partial.json'
        if os.path.isfile(partial_path):
            os.unlink(partial_path)
        print(f"Results saved to {args.results_file}")
        print("EGFIX_RUN_COMPLETE")


if __name__ == "__main__":
    main()
