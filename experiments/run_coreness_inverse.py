"""
Main script for in-processing and pre-processing fairness experiments.

Supports coreness-inverse and mobility-boost methods, both in-processing
(fairness penalty in loss) and pre-processing (adjacency modification).
Also supports PA, Exposure, and Hybrid in-processing baselines.

Usage:
    python run_coreness_inverse.py \
        --data_name ego-facebook \
        --fairness_method coreness_inverse_inproc \
        --lambda_fair 0.0,0.1,0.3 \
        --runs 5 \
        --use_equality_gain True \
        --use_centrality_gain True
"""

import os
import sys
import argparse
import torch
import traceback

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
snap_dir = os.path.abspath(os.path.join(current_dir, '../backbone'))

for path in [parent_dir, snap_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from core import (
    set_seed, str2bool, load_dataset,
    compute_statistics, compute_k_wise_statistics,
    print_enhanced_results_table, print_k_wise_summary,
    save_multi_lambda_results, train_inprocessing_model
)
from metrics.coreness import compute_coreness
from fairness_methods.inprocessing import FairnessAwareModel, apply_coreness_inverse_aggregation, apply_mobility_boost_aggregation
from metrics.degree_analysis import analyze_degree_effects_comparison


def create_experiment_summary(args, lambda_values):
    """Create a summary of the experiment configuration."""
    summary = []
    summary.append("="*80)
    summary.append("EXPERIMENT CONFIGURATION")
    summary.append("="*80)
    summary.append(f"Dataset: {args.data_name}")
    summary.append(f"Fairness Method: {args.fairness_method}")
    summary.append(f"Lambda Values: {lambda_values}")
    summary.append(f"Coreness Alpha: {args.coreness_alpha}")
    if args.fairness_method in ['inproc_pa', 'inproc_hybrid']:
        summary.append(f"PA Aggregation: {args.pa_agg}")
    if args.fairness_method in ['inproc_exposure', 'inproc_hybrid']:
        summary.append(f"Exposure Decay: {args.exposure_decay}")
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
    summary.append(f"   GNN Layers: {args.gnn_num_layers}")
    summary.append(f"   Hidden Channels: {args.gnn_hidden_channels}")
    summary.append(f"")
    summary.append(f"Evaluation:")
    summary.append(f"   Runs: {args.runs}")
    summary.append(f"   Validation Metric: {args.eval_metric}")
    summary.append(f"   Fairness Mode: {args.fairness_mode}")
    if args.use_pa_aware:
        summary.append(f"   PA-aware metrics: ENABLED")
    if args.use_ndkl:
        summary.append(f"   NDKL metrics: ENABLED (mode={args.ndkl_mode})")
        summary.append(f"   Eval user mode: {args.eval_user_mode}")
    if args.use_equality_gain:
        summary.append(f"   Equality Gain metrics: ENABLED")
    if args.use_centrality_gain:
        summary.append(f"   Centrality Gain metrics: ENABLED")
    summary.append("="*80)
    return "\n".join(summary)


def argument():
    """Command-line arguments for in-processing experiments."""
    parser = argparse.ArgumentParser(
        description='In-processing Fairness-Aware Link Prediction'
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
    parser.add_argument('--epochs', type=int, default=500,
                       help='Maximum number of training epochs')
    parser.add_argument('--eval_steps', type=int, default=5,
                       help='Evaluate every N epochs')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')
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

    # In-Processing Fairness Parameters
    parser.add_argument('--fairness_method', type=str, default='coreness_inverse_inproc',
                       choices=['coreness_inverse_preproc', 'mobility_boost_preproc',
                                'coreness_inverse_inproc', 'mobility_boost_inproc',
                                'inproc_pa', 'inproc_exposure', 'inproc_hybrid', 'none'],
                       help='Fairness method')
    parser.add_argument('--coreness_alpha', type=float, default=1.0,
                       help='Alpha exponent for coreness-based methods')
    parser.add_argument('--lambda_fair', type=str, default='0.1',
                       help='Fairness penalty strength (comma-separated for multiple values)')
    parser.add_argument('--penalty_reduction', type=str, default='mean',
                       choices=['mean', 'sum'],
                       help="Batch reduction of the in-processing fairness penalty "
                            "relative to the base loss (the default AUC base loss sums). "
                            "'mean' (default) = historical behaviour, penalty is "
                            "~|E+_batch| times smaller than the base loss; "
                            "'sum' = both terms reduce identically, so lambda_fair "
                            "is a per-example weight. Ignored for preproc methods.")
    parser.add_argument('--reset_seed_per_model', type=str2bool, default=False,
                       help="Re-seed with (seed + run_index) immediately before "
                            "constructing every model, so each (run, lambda) starts "
                            "from an identical RNG state. Required for paired "
                            "comparisons across separately launched arms. Default "
                            "False preserves the historical protocol, in which one "
                            "RNG stream evolves across lambdas and runs.")
    parser.add_argument('--pa_agg', type=str, default='product',
                       choices=['product', 'sum', 'max', 'log_product'],
                       help='PA score aggregation method')
    parser.add_argument('--exposure_decay', type=float, default=1.0,
                       help='Exposure decay factor for exposure-based methods')

    # Evaluation Parameters
    parser.add_argument('--eval_metric', type=str, default='all',
                       choices=['precision', 'fairness', 'all'],
                       help='Metric for model selection during training')
    parser.add_argument('--fairness_mode', type=str, default='indegree',
                       choices=['indegree', 'outdegree', 'total'])

    # Advanced Fairness Metrics
    parser.add_argument('--use_pa_aware', type=str2bool, default=False,
                       help='Enable PA-aware fairness metrics')
    parser.add_argument('--use_ndkl', type=str2bool, default=False,
                       help='Enable NDKL fairness metrics')
    parser.add_argument('--ndkl_mode', type=str, default='per_user',
                       choices=['global', 'per_user'],
                       help='NDKL computation mode')
    parser.add_argument('--eval_user_mode', type=str, default='test_users',
                       choices=['test_users', 'all_nodes'],
                       help='User set for evaluation')

    # Results
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of experimental runs')
    parser.add_argument('--save_results', type=str2bool, default=True)
    parser.add_argument('--results_file', type=str, default='results_inproc.json')
    parser.add_argument('--seed', type=int, default=42)

    # === Metric Flags ===
    parser.add_argument('--use_equality_gain', type=str2bool, default=True,
                       help='Enable Equality Gain metrics')

    parser.add_argument('--use_stratified', type=str2bool, default=True,
                       help='Enable stratified metrics (20/80 split) at K=10')

    parser.add_argument('--stratification_mode', type=str, default='source',
                       help='Stratification mode for metrics (source, target, mixed)')

    parser.add_argument('--use_degree_plots', type=str2bool, default=True,
                       help='Enable degree analysis plots')

    parser.add_argument('--plot_save_dir', type=str, default='plots',
                       help='Directory to save degree analysis plots')

    parser.add_argument('--use_centrality_gain', type=str2bool, default=True,
                       help='Enable Centrality Gain metrics')

    def parse_centrality_types(s):
        """Parse comma-separated centrality types."""
        if s is None:
            return ['degree', 'katz', 'pagerank', 'eigenvector', 'closeness']
        return [x.strip() for x in s.split(',')]

    parser.add_argument('--centrality_types', type=parse_centrality_types,
                   default='degree,katz,pagerank,eigenvector,closeness',
                   help='Comma-separated centrality types: degree,katz,pagerank,eigenvector,closeness')

    parser.add_argument('--split_file', type=str, default=None, help='Path to pre-split dataset file')
    parser.add_argument('--skip_baseline', type=str2bool, default=False,
                       help='Skip baseline (lambda=0) training and only run fairness lambda values')

    return parser.parse_args()


def main():
    args = argument()
    set_seed(args.seed)

    # Parse lambda values
    lambda_values = [float(x.strip()) for x in args.lambda_fair.split(',')]
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    # Print experiment configuration
    print("\n" + create_experiment_summary(args, lambda_values))

    all_results = []

    for run in range(args.runs):
        print(f"\n{'='*80}")
        print(f"RUN {run + 1}/{args.runs}")
        print(f"{'='*80}")

        # Load dataset
        dataset, split_edge, data, degrees = load_dataset(args)
        num_nodes = dataset._data.num_nodes if hasattr(dataset, '_data') else dataset.data.num_nodes

        coreness = None
        if args.fairness_method in ['coreness_inverse_preproc', 'mobility_boost_preproc', 'coreness_inverse_inproc', 'mobility_boost_inproc']:
             if hasattr(data, 'coreness'):
                 coreness = data.coreness
             else:
                 print("Computing coreness for fairness methods (using iterative H-index)...")
                 # compute_coreness expects [num_edges, 2]
                 coreness = compute_coreness(data.edge_index.t(), num_nodes)
                 coreness = coreness.to(device)
                 data.coreness = coreness # Cache it
        run_results = {}

        if args.skip_baseline:
            print(f"\n{'-'*80}")
            print(f"SKIPPING BASELINE (--skip_baseline=True)")
            print(f"{'-'*80}")
            run_results['baseline'] = None
        else:
            print(f"\n{'-'*80}")
            print(f"TRAINING BASELINE (lambda=0)")
            print(f"{'-'*80}")

            # OPT-IN (see run_cafe.py): re-seed before every model so each
            # (run, lambda) starts from an identical RNG state and paired
            # comparisons across arms remain valid. Default False keeps the
            # historical single evolving RNG stream.
            if args.reset_seed_per_model:
                model_seed = args.seed + run
                set_seed(model_seed)
                print(f"[seed] model seed = {model_seed} "
                      f"(args.seed={args.seed} + run={run})")

            baseline_model = FairnessAwareModel(
                fairness_method='none',
                lambda_fair=0.0,
                pa_aggregation=args.pa_agg,
                num_nodes=num_nodes,
                exposure_decay=args.exposure_decay,
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
                init=args.init
            )
            baseline_model.param_init()

            baseline_model, _, baseline_results = train_inprocessing_model(
                baseline_model, data, split_edge, args, degrees, patience=args.patience,
                coreness=coreness
            )

            print(f"\n Baseline training complete")
            run_results['baseline'] = baseline_results

        # Train with different lambda values
        for lv in lambda_values:
            if lv == 0.0:
                # Skip lambda=0 (already done as baseline)
                run_results['lambda_0.0'] = run_results['baseline']
                continue

            print(f"\n{'-'*80}")
            print(f"TRAINING WITH FAIRNESS (lambda={lv})")
            print(f"{'-'*80}")

            # Determine method and apply pre-processing if needed
            if args.fairness_method == 'coreness_inverse_preproc':
                print(f"Applying Coreness-Inverse Pre-Processing with alpha={lv}")
                original_adj_t = data.adj_t
                data.adj_t = apply_coreness_inverse_aggregation(data.adj_t, coreness, alpha=lv)
                current_fair = 0.0
                effective_method = 'none'
            elif args.fairness_method == 'mobility_boost_preproc':
                print(f"Applying Mobility-Boost Pre-Processing with alpha={lv}")
                original_adj_t = data.adj_t
                data.adj_t = apply_mobility_boost_aggregation(data.adj_t, coreness, alpha=lv)
                current_fair = 0.0
                effective_method = 'none'
            else:
                current_fair = lv
                effective_method = args.fairness_method
                original_adj_t = None

            if args.reset_seed_per_model:
                model_seed = args.seed + run
                set_seed(model_seed)
                print(f"[seed] model seed = {model_seed} "
                      f"(args.seed={args.seed} + run={run})")

            fairness_model = FairnessAwareModel(
                fairness_method=effective_method,
                lambda_fair=current_fair,
                pa_aggregation=args.pa_agg,
                num_nodes=num_nodes,
                exposure_decay=args.exposure_decay,
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
            fairness_model.param_init()

            # Update args temporarily for this lambda
            original_lambda = args.lambda_fair
            args.lambda_fair = lv

            # Train with fairness
            fairness_model, _, fairness_results = train_inprocessing_model(
                fairness_model, data, split_edge, args, degrees,
                patience=args.patience, baseline_results=run_results['baseline'],
                coreness=coreness
            )

            if args.fairness_method in ['coreness_inverse_preproc', 'mobility_boost_preproc']:
                # Restore original adj for pre-processing methods
                data.adj_t = original_adj_t

            args.lambda_fair = original_lambda

            print(f"\n Fairness training complete (lambda={lv})")

            run_results[f'lambda_{lv}'] = fairness_results

        all_results.append(run_results)

    # Aggregate and report results
    print(f"\n\n{'='*80}")
    print(f"AGGREGATED RESULTS ACROSS {args.runs} RUNS")
    print(f"{'='*80}")

    baseline_runs = [r['baseline'] for r in all_results if r['baseline'] is not None]
    baseline_stats = compute_statistics(baseline_runs) if baseline_runs else None
    lambda_stats = {
        lv: compute_statistics([r[f'lambda_{lv}'] for r in all_results])
        for lv in lambda_values
    }

    if baseline_stats is not None:
        print_enhanced_results_table(
            baseline_stats, lambda_stats, lambda_values,
            use_pa_aware=args.use_pa_aware,
            use_ndkl=args.use_ndkl,
            use_equality_gain=args.use_equality_gain,
            use_stratified=args.use_stratified,
            use_centrality_gain=args.use_centrality_gain
        )
    else:
        print("\n[Baseline skipped] Printing lambda-only results:")
        for lv in lambda_values:
            stats = lambda_stats[lv]
            for metric, vals in stats.items():
                if isinstance(vals, dict) and 'mean' in vals:
                    print(f"  λ={lv} | {metric}: {vals['mean']:.6f} ± {vals['std']:.6f}")

    # Filter out runs with None baselines for k-wise stats
    filtered_results = [
        {k: v for k, v in r.items() if v is not None}
        for r in all_results if r.get('baseline') is not None
    ]

    k_stats = compute_k_wise_statistics(
        filtered_results, lambda_values,
        use_ndkl=args.use_ndkl,
        use_pa_aware=args.use_pa_aware,
        use_equality_gain=args.use_equality_gain,
        use_stratified=args.use_stratified
    )
    print_k_wise_summary(
        k_stats, lambda_values,
        use_ndkl=args.use_ndkl,
        use_pa_aware=args.use_pa_aware,
        use_equality_gain=args.use_equality_gain,
        use_stratified=args.use_stratified
    )

    if args.save_results:
        save_multi_lambda_results(
            args, all_results, baseline_stats if baseline_stats is not None else {},
            lambda_stats, lambda_values, k_stats
        )
        print(f"\n Results saved to {args.results_file}")

    # Generate degree analysis comparison plots
    if args.use_degree_plots and len(all_results) > 0:
        print(f"\n\n{'='*80}")
        print("GENERATING DEGREE ANALYSIS COMPARISON PLOTS")
        print(f"{'='*80}")

        try:
            # Extract recommendations from results
            all_recs = {}
            for run_idx, run_results in enumerate(all_results):
                baseline_recs = run_results['baseline'].get('_recs_baseline', {})

                if baseline_recs:  # Only process if we have recommendations
                    if 'Baseline' not in all_recs:
                        all_recs['Baseline'] = baseline_recs

                    # Get recommendations for each lambda
                    for lv in lambda_values:
                        key = f'lambda_{lv}'
                        if key in run_results:
                            lambda_recs = run_results[key].get('_recs_baseline', {})
                            if lambda_recs and f'λ={lv}' not in all_recs:
                                all_recs[f'λ={lv}'] = lambda_recs

                    # Only process first run (since they all have same structure)
                    break

            if len(all_recs) > 1:
                # Get edges and degrees from dataset
                train_edges = split_edge['train']['edge'].to('cpu')
                test_edges = split_edge['test']['edge'].to('cpu')

                # Create save directory
                degree_plot_dir = args.plot_save_dir
                os.makedirs(degree_plot_dir, exist_ok=True)

                # Run comparison analysis
                results = analyze_degree_effects_comparison(
                    all_recs=all_recs,
                    train_edges=train_edges,
                    test_edges=test_edges,
                    degrees=degrees,
                    num_nodes=num_nodes,
                    k=10,
                    num_bins=10,
                    save_dir=degree_plot_dir
                )

                print(f"\n Degree analysis comparison plots saved to {degree_plot_dir}")
            else:
                print("\n Note: Recommendations not available for degree comparison plots")
        except Exception as e:
            print(f"\n Warning: Could not generate degree comparison plots: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
