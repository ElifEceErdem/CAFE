"""
Main script for Coreness-based re-ranking fairness experiments.

Sweeps over lambda_mobility values to evaluate mobility boost / penalty re-ranking.

Usage:
    python run_mobility_boost.py \
        --data_name ego-facebook \
        --lambda_mobility 0.0,0.1,0.3 \
        --lambda_bridge 0.0 \
        --coreness_alpha 1.0 \
        --boosting_mode continuous \
        --reranking_approach boost \
        --runs 5 \
        --use_equality_gain True \
        --use_centrality_gain True
"""

import os
import sys
import argparse
import torch

# Path setup
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
from core.evaluation import compute_baseline_metrics, compute_fairness_metrics_with_coreness
from backbone.model import BaseModel


def create_experiment_summary(args, lambda_values):
    """Create a summary of the experiment configuration."""
    summary = []
    summary.append("="*80)
    summary.append("EXPERIMENT CONFIGURATION (CORENESS RE-RANKING)")
    summary.append("="*80)
    summary.append(f"Dataset: {args.data_name}")
    summary.append(f"Fairness Method: Coreness-based")
    summary.append(f"Lambda Mobility Values: {lambda_values}")
    summary.append(f"Lambda Bridge: {args.lambda_bridge}")
    summary.append(f"Coreness Alpha: {args.coreness_alpha}")
    summary.append(f"Boosting Mode: {args.boosting_mode}")
    summary.append(f"Reranking Approach: {args.reranking_approach}")
    summary.append(f"")
    summary.append(f"Training:")
    summary.append(f"   Epochs: {args.epochs}")
    summary.append(f"   Patience: {args.patience}")
    summary.append(f"   Batch Size: {args.batch_size}")
    summary.append(f"   Learning Rate: {args.lr}")
    summary.append(f"")
    summary.append(f"Model:")
    summary.append(f"   Encoder: {args.encoder}")
    summary.append(f"   Predictor: {args.predictor}")
    summary.append(f"")
    summary.append(f"Evaluation:")
    summary.append(f"   Runs: {args.runs}")
    summary.append(f"   Fairness Mode: {args.fairness_mode}")
    if args.use_equality_gain:
        summary.append(f"   Equality Gain metrics: ENABLED")
    if args.use_centrality_gain:
        summary.append(f"   Centrality Gain metrics: ENABLED")
    summary.append("="*80)
    return "\n".join(summary)


def argument():
    """Command-line arguments for Coreness re-ranking experiments."""
    parser = argparse.ArgumentParser(
        description='Coreness-based Re-ranking Fairness Experiments'
    )

    # Device & Model
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--encoder', type=str, default='HLGNN')
    parser.add_argument('--predictor', type=str, default='MLP')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--loss_func', type=str, default='AUC')

    # Dataset
    parser.add_argument('--data_name', type=str, default='ego-facebook')
    parser.add_argument('--data_path', type=str, default='../../data/SNAP')
    parser.add_argument('--val_ratio', type=float, default=0.05)
    parser.add_argument('--test_ratio', type=float, default=0.1)

    # Training Parameters
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16384)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--neg_sampler', type=str, default='global')
    parser.add_argument('--num_neg', type=int, default=1)

    # Model Architecture
    parser.add_argument('--gnn_num_layers', type=int, default=15)
    parser.add_argument('--mlp_num_layers', type=int, default=2)
    parser.add_argument('--emb_hidden_channels', type=int, default=256)
    parser.add_argument('--gnn_hidden_channels', type=int, default=256)
    parser.add_argument('--mlp_hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--grad_clip_norm', type=float, default=2.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--init', type=str, default='KI')

    # Node Features
    parser.add_argument('--train_node_emb', type=str2bool, default=True)
    parser.add_argument('--use_node_feats', type=str2bool, default=False)
    parser.add_argument('--pretrain_emb', type=str, default=None)

    # Coreness Re-ranking Parameters
    parser.add_argument('--lambda_mobility', type=str, default='0.3',
                       help='Mobility boost strength (comma-separated for sweep)')
    parser.add_argument('--lambda_bridge', type=float, default=0.0,
                       help='Bridge penalty strength (currently disabled, kept for compatibility)')
    parser.add_argument('--coreness_alpha', type=float, default=1.0,
                       help='Exponent for mobility boost')
    parser.add_argument('--boosting_mode', type=str, default='continuous',
                        help='Boosting mode: "continuous" (log-based) or "tiered" (discrete bands)')
    parser.add_argument('--reranking_approach', type=str, default='boost',
                        choices=['boost', 'penalty'],
                        help='Reranking approach: "boost" (promote periphery) or "penalty" (suppress core)')

    # Evaluation Parameters
    parser.add_argument('--eval_metric', type=str, default='all',
                       choices=['mrr', 'precision', 'fairness', 'all'])
    parser.add_argument('--fairness_mode', type=str, default='indegree',
                       choices=['indegree', 'outdegree', 'total'])

    # Advanced Fairness Metrics
    parser.add_argument('--use_pa_aware', type=str2bool, default=False)
    parser.add_argument('--use_ndkl', type=str2bool, default=False)
    parser.add_argument('--ndkl_mode', type=str, default='per_user',
                       choices=['global', 'per_user'])
    parser.add_argument('--eval_user_mode', type=str, default='test_users',
                       choices=['test_users', 'all_nodes'])
    parser.add_argument('--use_equality_gain', type=str2bool, default=True)
    parser.add_argument('--use_stratified', type=str2bool, default=True)
    parser.add_argument('--stratification_mode', type=str, default='source')
    parser.add_argument('--use_degree_plots', type=str2bool, default=True)
    parser.add_argument('--plot_save_dir', type=str, default='plots')
    parser.add_argument('--use_centrality_gain', type=str2bool, default=True)

    def parse_centrality_types(s):
        if s is None:
            return ['degree', 'katz', 'pagerank', 'eigenvector', 'closeness']
        return [x.strip() for x in s.split(',')]

    parser.add_argument('--centrality_types', type=parse_centrality_types,
                       default='degree,katz,pagerank,eigenvector,closeness')

    # Results
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--save_results', type=str2bool, default=True)
    parser.add_argument('--results_file', type=str, default='results_coreness.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_file', type=str, default=None, help='Path to pre-split dataset file')

    return parser.parse_args()


def main():
    args = argument()
    args.fairness_method = 'Coreness-based'
    set_seed(args.seed)

    # Parse lambda_mobility sweep values
    lambda_values = [float(x.strip()) for x in args.lambda_mobility.split(',')]

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    # Print experiment configuration
    print("\n" + create_experiment_summary(args, lambda_values))

    all_results = []
    degree_analysis_collector = {}

    for run in range(args.runs):
        print(f"\n{'='*80}")
        print(f"RUN {run + 1}/{args.runs}")
        print(f"{'='*80}")

        # Load dataset
        dataset, split_edge, data, degrees = load_dataset(args)
        num_nodes = dataset._data.num_nodes if hasattr(dataset, '_data') else dataset.data.num_nodes

        # Build and train baseline model
        print(f"\n{'-'*80}")
        print(f"TRAINING BASELINE MODEL")
        print(f"{'-'*80}")

        model = BaseModel(
            lr=args.lr,
            dropout=args.dropout,
            grad_clip_norm=args.grad_clip_norm,
            gnn_num_layers=args.gnn_num_layers,
            mlp_num_layers=args.mlp_num_layers,
            emb_hidden_channels=args.emb_hidden_channels,
            gnn_hidden_channels=args.gnn_hidden_channels,
            mlp_hidden_channels=args.mlp_hidden_channels,
            num_nodes=num_nodes,
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
            init=args.init
        )
        model.param_init()

        # Train model
        model, best_epoch, _, _, _, _ = train_model(model, data, split_edge, args)

        print(f"\n Training complete. Best epoch: {best_epoch}")

        # Evaluate baseline
        print(f"\n{'='*60}")
        print("BASELINE EVALUATION")
        print(f"{'='*60}")
        baseline_results = compute_baseline_metrics(model, data, split_edge, args)

        # Clean up internal keys before storing
        run_results = {
            'baseline': {k: v for k, v in baseline_results.items()
                        if not k.startswith('_')}
        }

        # Evaluate with different lambda_mobility values
        for val in lambda_values:
            print(f"\n{'='*60}")
            print(f"FAIRNESS EVALUATION (lambda_mobility={val})")
            print(f"{'='*60}")

            original_val = args.lambda_mobility
            args.lambda_mobility = val

            fairness_results = compute_fairness_metrics_with_coreness(
                model, data, split_edge, args, baseline_results
            )

            args.lambda_mobility = original_val

            # Collect degree analysis for aggregated plots
            if '_degree_analysis' in fairness_results:
                da = fairness_results['_degree_analysis']
                key = f'lambda_{val}'
                if key not in degree_analysis_collector:
                    degree_analysis_collector[key] = []
                degree_analysis_collector[key].append(da)

            # Use lambda_ prefix for compatibility with reporting tools
            run_results[f'lambda_{val}'] = {k: v for k, v in fairness_results.items()
                                          if not k.startswith('_')}

        all_results.append(run_results)

    # Aggregate and report results
    print(f"\n\n{'='*80}")
    print(f"AGGREGATED RESULTS ACROSS {args.runs} RUNS")
    print(f"{'='*80}")

    baseline_stats = compute_statistics([r['baseline'] for r in all_results])

    lambda_stats = {val: compute_statistics([r[f'lambda_{val}'] for r in all_results])
                   for val in lambda_values}

    print_enhanced_results_table(
        baseline_stats, lambda_stats, lambda_values,
        use_pa_aware=args.use_pa_aware,
        use_ndkl=args.use_ndkl
    )

    k_stats = compute_k_wise_statistics(
        all_results, lambda_values,
        use_ndkl=args.use_ndkl,
        use_pa_aware=args.use_pa_aware,
    )

    print_k_wise_summary(
        k_stats, lambda_values,
        use_ndkl=args.use_ndkl,
        use_pa_aware=args.use_pa_aware
    )

    if args.save_results:
        save_multi_lambda_results(args, all_results, baseline_stats,
                                 lambda_stats, lambda_values, k_stats)
        print(f"\n Results saved to {args.results_file}")

    # Generate Aggregated Plots
    if args.use_degree_plots and args.runs > 1:
        print(f"\nGenerating Aggregated Degree Analysis Plots (Mean +/- Std)...")
        from metrics.degree_analysis import plot_aggregated_degree_analysis

        save_dir = getattr(args, 'plot_save_dir', 'plots')

        for key, da_list in degree_analysis_collector.items():
            if not da_list:
                continue

            prec_results = []
            eq_results = []

            for da in da_list:
                prec_results.append((
                    da['precision_bin_centers'],
                    da['precision_mean'],
                    da['precision_std']
                ))
                eq_results.append((
                    da['equality_gain_bin_centers'],
                    da['equality_gains']
                ))

            # Clean key for filename
            clean_key = key.replace('.', 'p')  # 0.1 -> 0p1

            aggr_save_dir = os.path.join(save_dir, 'aggregated', clean_key)
            os.makedirs(aggr_save_dir, exist_ok=True)

            plot_aggregated_degree_analysis(
                prec_results, eq_results, k=10, save_dir=aggr_save_dir
            )
            print(f"  Saved aggregated plots for {key} to {aggr_save_dir}")


if __name__ == "__main__":
    main()
