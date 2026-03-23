import os
import sys
import argparse
import torch
import numpy as np
import scipy.sparse as sp


# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
snap_dir = os.path.abspath(os.path.join(current_dir, '../backbone'))

for path in [parent_dir, snap_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from core.data_loader import build_train_adjacency
from metrics.coreness import compute_coreness

from core import (
    set_seed, str2bool, load_dataset, train_model,
    compute_statistics, compute_k_wise_statistics,
    print_enhanced_results_table, print_k_wise_summary,
    save_multi_lambda_results, create_experiment_info
)
from core.evaluation import compute_metrics_from_recs, compute_equality_gain_metrics, make_predictor_score_fn
from backbone.model import BaseModel
from metrics.centrality_equality_gain import compute_all_centrality_equality_gains
from metrics.demographics import compute_demographic_metrics, compute_demographic_parity_ratio_change

def argument():
    parser = argparse.ArgumentParser(description='Hybrid RA-GNN Fairness Experiment')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--data_name', type=str, default='ego-facebook')
    parser.add_argument('--data_path', type=str, default='../../data/SNAP')
    parser.add_argument('--val_ratio', type=float, default=0.05)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16384)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=50)
    
    # Model
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
    parser.add_argument('--neg_sampler', type=str, default='global')
    parser.add_argument('--num_neg', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--eval_metric', type=str, default='all')
    
    # Hybrid Parameters
    parser.add_argument('--beta_ra', type=str, default='0.5,1.0,2.0',
                       help='Weight for RA score (comma separated)')
    
    # Fairness Parameters
    parser.add_argument('--lambda_mobility', type=str, default='0.3', 
                       help='Fairness strength for Coreness Inverse')
    parser.add_argument('--coreness_alpha', type=float, default=1.0)
    parser.add_argument('--fairness_mode', type=str, default='indegree')
    parser.add_argument('--use_equality_gain', type=str2bool, default=True)
    parser.add_argument('--use_centrality_gain', type=str2bool, default=True)
    
    # Results
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--results_file', type=str, default='results_hybrid.json')
    parser.add_argument('--save_results', type=str2bool, default=True)
    parser.add_argument('--split_file', type=str, default=None, help='Path to pre-split dataset file')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()

def compute_ra_scores_sparse(adj_matrix):
    """
    Compute Resource Allocation scores as a normalized sparse CSR matrix.
    RA(u,v) = sum(1/deg(z)) for z in shared_neighbors.
    Returns sparse CSR with log1p + max normalization applied to nonzero entries.
    """
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1.0
    inv_deg = sp.diags(1.0 / degrees)

    # A * D^-1 * A^T — result is sparse (only pairs with shared neighbors are nonzero)
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

def main():
    args = argument()
    args.fairness_method = 'Hybrid RA+GNN'
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    
    betas = [float(x) for x in args.beta_ra.split(',')]
    lambdas = [float(x) for x in args.lambda_mobility.split(',')]
    
    print("="*80)
    print(f"HYBRID FAIRNESS EXPERIMENT: GNN + {args.beta_ra} * RA")
    print(f"Fairness Reranking: Coreness Inverse (Lambda={args.lambda_mobility})")
    print("="*80)
    
    # Initialize aggregation containers
    import collections
    results_by_config = collections.defaultdict(list)
    
    all_results = []
    
    for run in range(args.runs):
        print(f"\nRUN {run+1}/{args.runs}")
        
        # Load Data
        dataset, split_edge, data, degrees = load_dataset(args)
        num_nodes = dataset._data.num_nodes if hasattr(dataset, '_data') else dataset.data.num_nodes
        train_edges = split_edge['train']['edge'].to(device)
        pos_test_edge = split_edge['test']['edge'].to(device)
        
        # --- Load sensitive attributes ---
        from core.data_loader import load_sensitive_attr
        sens_attr = None
        try:
            sens_attr = load_sensitive_attr(args.data_name, num_nodes, root=args.data_path)
            g0 = int((sens_attr == 0).sum())
            g1 = int((sens_attr == 1).sum())
            print(f"Loaded sensitive attributes: group_0={g0}, group_1={g1}")
            data.sens = torch.tensor(sens_attr, device=device) if not torch.is_tensor(sens_attr) else sens_attr.to(device)
        except ValueError as e:
            print(f"WARNING: {e}")
            print("Skipping demographic metrics because sensitive attributes could not be loaded.")
        
        # --- Precompute RA Scores ---
        print("\nComputing Resource Allocation (RA) Scores...")
        # Build scipy sparse adj
        row = train_edges[:, 0].cpu().numpy()
        col = train_edges[:, 1].cpu().numpy()
        # Symmetrize
        all_row = np.concatenate([row, col])
        all_col = np.concatenate([col, row])
        vals = np.ones_like(all_row)
        adj_sp = sp.csr_matrix((vals, (all_row, all_col)), shape=(num_nodes, num_nodes))
        
        ra_sparse = compute_ra_scores_sparse(adj_sp)  # sparse CSR, already normalized
        
        # --- Train Baseline GNN ---
        print("\nTraining HL-GNN Baseline...")
        model = BaseModel(
            lr=args.lr, dropout=args.dropout, grad_clip_norm=args.grad_clip_norm,
            gnn_num_layers=args.gnn_num_layers, mlp_num_layers=args.mlp_num_layers,
            emb_hidden_channels=args.emb_hidden_channels, gnn_hidden_channels=args.gnn_hidden_channels,
            mlp_hidden_channels=args.mlp_hidden_channels, num_nodes=num_nodes, num_node_feats=0,
            gnn_encoder_name=args.encoder, predictor_name=args.predictor, loss_func=args.loss_func,
            optimizer_name=args.optimizer, device=device, use_node_feats=args.use_node_feats,
            train_node_emb=args.train_node_emb, pretrain_emb=args.pretrain_emb, alpha=args.alpha, init=args.init
        )
        model.param_init()
        model, _, _, _, _, _ = train_model(model, data, split_edge, args)
        print("Training complete.", flush=True)
        
        # Get Embeddings
        model.encoder.eval()
        model.predictor.eval()
        with torch.no_grad():
            input_feat = model.create_input_feat(data)
            h_emb = model.encoder(input_feat, data.adj_t, data.edge_weight)
        
        # Create scoring function using trained predictor (MLP)
        score_fn = make_predictor_score_fn(model.predictor, predictor_type='two_arg')
        print("Using trained predictor (MLP) for GNN scoring")
        
        # --- Prepare Evaluator ---
        # Build CSR adjacency mask for vectorized train-edge masking
        train_adj_csr = adj_sp.copy()
        train_adj_csr.data = np.ones_like(train_adj_csr.data)  # binary mask
        # Also keep list-of-sets for callers that need it
        train_adj = build_train_adjacency(train_edges, num_nodes)

        # Keep CPU copies for metric functions
        train_edges_cpu = train_edges.cpu()
        pos_test_edge_cpu = pos_test_edge.cpu()
        
        run_res = {}
        
        # --- Evaluate for each Beta/Lambda pair ---
        # Only evaluate test nodes: nodes appearing in test edges.
        # This is methodologically correct: the control graph (train+test) only adds
        # edges for test nodes, so the predicted graph should also only add recs for
        # test nodes to ensure an apples-to-apples Equality Gain comparison.
        test_nodes = torch.unique(pos_test_edge.view(-1)).cpu().tolist()
        test_nodes_sorted = sorted(test_nodes)
        print(f"Evaluating {len(test_nodes_sorted)} test nodes (out of {num_nodes} total)", flush=True)
        
        for beta in betas:
            for lam in lambdas:
                combo_key = f"beta_{beta}_lambda_{lam}"
                print(f"\nEvaluating Hybrid: Beta={beta}, Lambda={lam}", flush=True)
                
                # Hybrid Reranking Logic
                k = 100
                recs_by_u = {}
                chunk_size = 1024
                
                # Compute Coreness ONCE
                if not hasattr(data, 'coreness'):
                    data.coreness = compute_coreness(train_edges_cpu, num_nodes)
                c_np = data.coreness.cpu().numpy()
                
                # Precompute Mobility Multipliers (Vectorized)
                # Matches CorenessReRanker logic
                from scipy import stats
                c_ranks = stats.rankdata(c_np, method='average')
                c_pctl = c_ranks / len(c_np)
                scaling = 100.0
                denom = np.log(c_pctl * scaling + np.e)
                mobility_boost = 1.0 / (denom ** args.coreness_alpha) # [N]
                
                # Multiplier vector
                mult_vec = 1.0 + lam * mobility_boost
                mult_vec = np.clip(mult_vec, 1.0, 5.0)
                mult_tensor = torch.from_numpy(mult_vec).float()
                
                # Process test nodes in chunks
                num_test = len(test_nodes_sorted)
                num_chunks = (num_test + chunk_size - 1) // chunk_size
                for chunk_i in range(num_chunks):
                    chunk_start = chunk_i * chunk_size
                    chunk_end = min(chunk_start + chunk_size, num_test)
                    chunk_node_ids = test_nodes_sorted[chunk_start:chunk_end]
                    B = len(chunk_node_ids)
                    
                    if chunk_i % 5 == 0 or chunk_i == num_chunks - 1:
                        print(f"  Chunk {chunk_i+1}/{num_chunks} ({B} nodes)", flush=True)
                    
                    batch_tensor = torch.tensor(chunk_node_ids, dtype=torch.long, device=device)
                    node_ids_np = np.array(chunk_node_ids)

                    # 1. GNN Scores (using trained predictor)
                    with torch.no_grad():
                        h_batch = h_emb[batch_tensor]  # [B, D]
                        gnn_scores = score_fn(h_batch, h_emb).cpu()  # [B, N]

                    # 2. Hybrid Score = GNN + beta * RA (sparse RA addition)
                    #    Only add nonzero RA entries instead of densifying the
                    #    entire [B, N] RA matrix. Saves ~256MB per chunk.
                    hybrid_scores = gnn_scores  # safe to modify in-place (not reused)
                    if beta != 0:
                        ra_chunk_coo = ra_sparse[node_ids_np].tocoo()
                        if ra_chunk_coo.nnz > 0:
                            ra_r = torch.from_numpy(ra_chunk_coo.row.astype(np.int64))
                            ra_c = torch.from_numpy(ra_chunk_coo.col.astype(np.int64))
                            ra_v = torch.from_numpy(ra_chunk_coo.data.astype(np.float32))
                            hybrid_scores[ra_r, ra_c] += beta * ra_v

                    # 3. Fairness Boost
                    final_scores = hybrid_scores * mult_tensor.unsqueeze(0)

                    # 4. Masking: self-loops + train neighbors (sparse)
                    self_cols = torch.tensor(chunk_node_ids, dtype=torch.long)
                    final_scores[torch.arange(B), self_cols] = float('-inf')
                    #    Sparse neighbor mask — use nonzero indices directly
                    #    instead of creating dense [B, N] matrix. Saves ~256MB per chunk.
                    adj_chunk = train_adj_csr[node_ids_np]
                    nb_r, nb_c = adj_chunk.nonzero()
                    final_scores[nb_r, nb_c] = float('-inf')

                    # 6. Batch TopK
                    topk_vals, topk_idx = torch.topk(final_scores, k, dim=1)  # [B, k]
                    for i in range(B):
                        recs_by_u[chunk_node_ids[i]] = topk_idx[i].tolist()
                
                # Compute Metrics
                print("  Computing Metrics...")
                res = compute_metrics_from_recs(recs_by_u, pos_test_edge_cpu, num_nodes, args.fairness_mode)

                if args.use_equality_gain:
                    eq = compute_equality_gain_metrics(recs_by_u, train_edges_cpu, pos_test_edge_cpu, num_nodes, k_values=[10, 20, 50, 100])
                    res.update(eq)

                if args.use_centrality_gain:
                    cent_metrics = compute_all_centrality_equality_gains(
                        recs_by_u, train_edges_cpu, pos_test_edge_cpu, num_nodes,
                        k_values=[10, 20, 50, 100],
                        centrality_types=['degree', 'pagerank', 'closeness'],
                        use_sampling_closeness=True
                    )
                    res.update(cent_metrics)
                
                # Demographic metrics
                if sens_attr is not None:
                    features = getattr(data, 'x', None)
                    demo_metrics = compute_demographic_metrics(
                        recs_by_u=recs_by_u,
                        train_edges=train_edges_cpu,
                        test_edges=pos_test_edge_cpu,
                        num_nodes=num_nodes,
                        sens=sens_attr,
                        h_emb=h_emb.cpu(),
                        features=features.cpu() if features is not None else None,
                        adj=None,
                        k=10
                    )
                    res.update(demo_metrics)
                    
                    dpr_change = compute_demographic_parity_ratio_change(
                        train_edges_cpu, recs_by_u, num_nodes, sens_attr, k=10
                    )
                    res['Delta_DPR'] = dpr_change
                
                run_res[combo_key] = res
                results_by_config[combo_key].append(res)
        
        print("  Computing Pure GNN Baseline...")
        run_res['baseline'] = {} # Empty baseline to satisfy schema
        
        all_results.append(run_res)
        
    # Save
    if args.save_results:
        print("\nAggregating results across runs...")
        
        # 1. Compute Aggregated Stats per Config
        final_lambda_results = {}
        for key, run_list in results_by_config.items():
            final_lambda_results[key] = compute_statistics(run_list)
            
        experiment_info = create_experiment_info(args, args.fairness_method)
        
        from core.statistics import _serialize_stats
        
        final_output = {
            'experiment_info': experiment_info,
            'baseline': {
                'statistics': {} # Empty baseline
            },
            'lambda_results': {}
        }
        
        # Serialize each config
        for key, stats in final_lambda_results.items():
            final_output['lambda_results'][key] = {
                'statistics': _serialize_stats(stats)
            }
            
        # Helper to strict serialize
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

        final_output['individual_runs'] = strict_serialize(all_results)
        
        import json
        with open(args.results_file, 'w') as f:
            json.dump(final_output, f, indent=2)
        print(f"Aggregated results saved to {args.results_file}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
