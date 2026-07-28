"""
Weak-tie diversification baseline runner
(Sanz-Cruzado & Castells, RecSys 2018 — attribute-free structural
re-ranking on top of the HL-GNN backbone).

Trains the HL-GNN backbone once per run, then sweeps the diversification
trade-off lambda of the greedy weak-tie re-ranker
(fairness_methods/weak_tie_reranking.py) as a pure post-processing step
on the backbone's top-100 candidate pools. lambda = 0 reproduces the
backbone ranking; lambda = 1 ranks purely by structural diversity.

The diversification cutoff k (top-k forming the extended graph G') is 10
to match the paper's protocol and the study's headline K; metrics at
deeper K are computed from the same lists, whose tail keeps the
backbone order.

Usage:
    python run_weak_tie.py \
        --data_name unc \
        --weak_tie_variant cegc \
        --weak_tie_lambda 0.0,0.2,0.4,0.6,0.8,1.0 \
        --runs 10 \
        --use_equality_gain True --use_centrality_gain True
"""

import os
import sys
import argparse
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
snap_dir = os.path.abspath(os.path.join(current_dir, '../backbone'))
for path in [parent_dir, snap_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from core import (
    set_seed, str2bool, load_dataset, train_model,
    compute_statistics, compute_k_wise_statistics,
    print_enhanced_results_table, print_k_wise_summary,
    save_multi_lambda_results
)
from core.data_loader import load_sensitive_attr
from core.evaluation import (
    compute_baseline_metrics, compute_fairness_metrics_with_weak_tie
)
from backbone.model import BaseModel


def argument():
    parser = argparse.ArgumentParser(
        description='Weak-tie diversification re-ranking baseline'
    )

    # Device & Model (HL-GNN backbone, identical to the other runners)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--encoder', type=str, default='HLGNN')
    parser.add_argument('--predictor', type=str, default='MLP')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--loss_func', type=str, default='AUC')

    # Dataset
    parser.add_argument('--data_name', type=str, default='ego-facebook')
    parser.add_argument('--data_path', type=str, default='../data/SNAP')
    parser.add_argument('--val_ratio', type=float, default=0.05)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--split_file', type=str, default=None)

    # Training
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16384)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--neg_sampler', type=str, default='global')
    parser.add_argument('--num_neg', type=int, default=1)

    # Architecture (HL-GNN reference configuration)
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

    # Weak-tie re-ranking parameters
    parser.add_argument('--weak_tie_variant', type=str, default='cegc',
                        choices=['cegc', 'mc'],
                        help="Greedy diversity objective: 'cegc' (paper's "
                             "best) or 'mc' (binary weak-tie indicator)")
    parser.add_argument('--weak_tie_lambda', type=str,
                        default='0.0,0.2,0.4,0.6,0.8,1.0',
                        help='Diversification trade-off sweep (comma-separated)')
    parser.add_argument('--weak_tie_k', type=int, default=10,
                        help='Diversification cutoff (paper: 10)')
    parser.add_argument('--community_seed', type=int, default=42,
                        help='Seed for the Louvain (PLM) partition')

    # Evaluation
    parser.add_argument('--eval_metric', type=str, default='all')
    parser.add_argument('--fairness_mode', type=str, default='indegree',
                        choices=['indegree', 'outdegree', 'total'])
    parser.add_argument('--use_pa_aware', type=str2bool, default=False)
    parser.add_argument('--use_ndkl', type=str2bool, default=False)
    parser.add_argument('--ndkl_mode', type=str, default='per_user')
    parser.add_argument('--eval_user_mode', type=str, default='test_users')
    parser.add_argument('--use_equality_gain', type=str2bool, default=True)
    parser.add_argument('--use_stratified', type=str2bool, default=False)
    parser.add_argument('--stratification_mode', type=str, default='source')
    parser.add_argument('--use_degree_plots', type=str2bool, default=False)
    parser.add_argument('--plot_save_dir', type=str, default='plots')
    parser.add_argument('--use_centrality_gain', type=str2bool, default=True)

    def parse_centrality_types(s):
        return [x.strip() for x in s.split(',')]
    parser.add_argument('--centrality_types', type=parse_centrality_types,
                        default='degree,pagerank,closeness')

    # Results
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--save_results', type=str2bool, default=True)
    parser.add_argument('--results_file', type=str, default='results_weak_tie.json')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def main():
    args = argument()
    args.fairness_method = f'WeakTie-{args.weak_tie_variant.upper()}'
    set_seed(args.seed)

    lambda_values = [float(x.strip()) for x in args.weak_tie_lambda.split(',')]
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("WEAK-TIE DIVERSIFICATION BASELINE "
          f"(Sanz-Cruzado & Castells RecSys'18, variant={args.weak_tie_variant})")
    print(f"Dataset: {args.data_name} | lambda sweep: {lambda_values} | "
          f"k={args.weak_tie_k} | runs: {args.runs}")
    print("=" * 80)

    all_results = []

    for run in range(args.runs):
        print(f"\n{'=' * 80}\nRUN {run + 1}/{args.runs}\n{'=' * 80}")
        set_seed(args.seed + run)

        dataset, split_edge, data, degrees = load_dataset(args)
        num_nodes = dataset._data.num_nodes if hasattr(dataset, '_data') else dataset.data.num_nodes

        try:
            sens_attr = load_sensitive_attr(args.data_name, num_nodes, root=args.data_path)
            data.sens = (sens_attr if torch.is_tensor(sens_attr)
                         else torch.tensor(sens_attr)).to(device)
            print(f"Loaded sensitive attributes "
                  f"(g0={int((data.sens == 0).sum())}, g1={int((data.sens == 1).sum())})")
        except ValueError as e:
            print(f"NOTE: {e} -- demographic metrics skipped.")

        print(f"\n{'-' * 80}\nTRAINING HL-GNN BACKBONE\n{'-' * 80}")
        model = BaseModel(
            lr=args.lr, dropout=args.dropout,
            grad_clip_norm=args.grad_clip_norm,
            gnn_num_layers=args.gnn_num_layers,
            mlp_num_layers=args.mlp_num_layers,
            emb_hidden_channels=args.emb_hidden_channels,
            gnn_hidden_channels=args.gnn_hidden_channels,
            mlp_hidden_channels=args.mlp_hidden_channels,
            num_nodes=num_nodes, num_node_feats=0,
            gnn_encoder_name=args.encoder,
            predictor_name=args.predictor,
            loss_func=args.loss_func,
            optimizer_name=args.optimizer,
            device=device,
            use_node_feats=args.use_node_feats,
            train_node_emb=args.train_node_emb,
            pretrain_emb=args.pretrain_emb,
            alpha=args.alpha, init=args.init,
        )
        model.param_init()
        model, best_epoch, _, _, _, _ = train_model(model, data, split_edge, args)
        print(f"\nTraining complete. Best epoch: {best_epoch}")

        print(f"\n{'=' * 60}\nBASELINE (BACKBONE) EVALUATION\n{'=' * 60}")
        baseline_results = compute_baseline_metrics(model, data, split_edge, args)
        run_results = {
            'baseline': {k: v for k, v in baseline_results.items()
                         if not k.startswith('_')}
        }

        for val in lambda_values:
            print(f"\n{'=' * 60}\nWEAK-TIE RE-RANKING (lambda={val})\n{'=' * 60}")
            original = args.weak_tie_lambda
            args.weak_tie_lambda = val
            wt_results = compute_fairness_metrics_with_weak_tie(
                model, data, split_edge, args, baseline_results)
            args.weak_tie_lambda = original
            run_results[f'lambda_{val}'] = {k: v for k, v in wt_results.items()
                                            if not k.startswith('_')}

        all_results.append(run_results)

    print(f"\n\n{'=' * 80}\nAGGREGATED RESULTS ACROSS {args.runs} RUNS\n{'=' * 80}")
    baseline_stats = compute_statistics([r['baseline'] for r in all_results])
    lambda_stats = {val: compute_statistics([r[f'lambda_{val}'] for r in all_results])
                    for val in lambda_values}

    print_enhanced_results_table(
        baseline_stats, lambda_stats, lambda_values,
        use_pa_aware=args.use_pa_aware, use_ndkl=args.use_ndkl)

    k_stats = compute_k_wise_statistics(
        all_results, lambda_values,
        use_ndkl=args.use_ndkl, use_pa_aware=args.use_pa_aware)
    print_k_wise_summary(k_stats, lambda_values,
                         use_ndkl=args.use_ndkl, use_pa_aware=args.use_pa_aware)

    if args.save_results:
        save_multi_lambda_results(args, all_results, baseline_stats,
                                  lambda_stats, lambda_values, k_stats)
        print(f"\nResults saved to {args.results_file}")


if __name__ == "__main__":
    main()
