"""
Statistics computation and result reporting utilities.

Enhanced to support PA-aware and NDKL metrics.
"""

import json
import numpy as np
from datetime import datetime

def compute_statistics(all_results):
    """Compute mean and std across multiple runs."""
    if not all_results:
        return {}

    metric_names = sorted({k for r in all_results for k in r.keys()})

    def _val(x):
        # supports (name, value) tuples and raw scalars
        # Also converts PyTorch tensors to Python floats
        if isinstance(x, (tuple, list)) and len(x) >= 2:
            val = x[1]
        else:
            val = x
        
        # Convert PyTorch tensor to Python scalar
        if hasattr(val, 'item'):  # PyTorch tensor
            return float(val.item())
        elif isinstance(val, (int, float)):
            return float(val)
        else:
            return float(val)

    stats = {}
    for metric in metric_names:
        # Skip internal keys (starting with _) - these are not metrics to aggregate
        if metric.startswith('_'):
            continue
            
        values = [_val(r[metric]) for r in all_results if metric in r]
        if values:
            stats[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "values": values,
            }
    return stats

def print_summary_table(baseline_stats, lambda_stats, lambda_values, 
                       metrics_to_show=None, use_pa_aware=False, use_ndkl=False,
                       use_equality_gain=False, use_stratified=False, use_centrality_gain=False):
    """
    Print enhanced results table with delta values.
    NOW SUPPORTS: PA-aware, NDKL, Equality Gain, and Stratified metrics.
    """
    if metrics_to_show is None:
        metrics_to_show = []
        for K in [10, 20, 50, 100]:
            metrics_to_show.extend([
                f'Precision@{K}',
                f'Distribution_Fairness@{K}',
            ])
            if use_equality_gain:
                metrics_to_show.append(f'Equality_Gain@{K}')
            if use_ndkl:
                metrics_to_show.append(f'NDKL@{K}')
            metrics_to_show.append(f'Hits@{K}')
        metrics_to_show.append('MRR')
        
        # Add stratified metrics (K=10 only)
        if use_stratified:
            metrics_to_show.extend([
                'Stratified_Precision_Low@10',
                'Stratified_Precision_High@10',
                'Stratified_EqualityGain_Low@10',
                'Stratified_EqualityGain_High@10'
            ])
        
        if use_pa_aware:
            metrics_to_show.extend([
                'PA_Bias_Score',
                'PA_Fairness_Score',
                'PA_Overlap_Mean'
            ])
    
    col_width = 22
    header = f"{'Metric':<35} {'Baseline':<{col_width}}"
    for lambda_val in lambda_values:
        header += f" λ={lambda_val} (Δ) {'':<{col_width-len(str(lambda_val))-6}}"
    print(header)
    print("-" * (35 + col_width * (1 + len(lambda_values))))
    
    for metric in metrics_to_show:
        if metric not in baseline_stats:
            continue
        
        # Special handling for PA metrics (no percentage scaling)
        if metric in ['PA_Bias_Score', 'PA_Fairness_Score', 'PA_Overlap_Mean']:
            b_mean = baseline_stats[metric]['mean']
            b_std = baseline_stats[metric]['std']
            line = f"{metric:<35} {b_mean:>7.3f}±{b_std:<5.3f}"
            
            for lambda_val in lambda_values:
                if metric in lambda_stats[lambda_val]:
                    f_mean = lambda_stats[lambda_val][metric]['mean']
                    f_std = lambda_stats[lambda_val][metric]['std']
                    delta = f_mean - b_mean
                    sign = '+' if delta >= 0 else ''
                    line += f" {f_mean:>7.3f}±{f_std:<5.3f} ({sign}{delta:>6.3f})"
            
            print(line)
            continue
        
        # Special handling for NDKL metrics (raw values, lower is better)
        if 'NDKL' in metric:
            b_mean = baseline_stats[metric]['mean']
            b_std = baseline_stats[metric]['std']
            line = f"{metric:<35} {b_mean:>8.4f}±{b_std:<6.4f}"
            
            for lambda_val in lambda_values:
                if metric in lambda_stats[lambda_val]:
                    f_mean = lambda_stats[lambda_val][metric]['mean']
                    f_std = lambda_stats[lambda_val][metric]['std']
                    delta = f_mean - b_mean
                    rel_impr = (delta / b_mean * 100) if b_mean != 0 else 0.0
                    sign = '+' if delta >= 0 else ''
                    sign_rel = '+' if rel_impr >= 0 else ''
                    line += f" {f_mean:>8.4f}±{f_std:<6.4f} ({sign}{delta:>6.4f}, {sign_rel}{rel_impr:>5.2f}%)"
            
            print(line)
            continue
        
        # Special handling for Equality Gain (both degree and centrality-based)
        if 'Equality_Gain' in metric or 'EG@' in metric or '_EG@' in metric:
            b_mean = baseline_stats[metric]['mean']
            b_std = baseline_stats[metric]['std']
            line = f"{metric:<35} {b_mean:>7.2f}±{b_std:<5.2f}%"
            
            for lambda_val in lambda_values:
                if metric in lambda_stats[lambda_val]:
                    f_mean = lambda_stats[lambda_val][metric]['mean']
                    f_std = lambda_stats[lambda_val][metric]['std']
                    delta = f_mean - b_mean
                    sign = '+' if delta >= 0 else ''
                    line += f" {f_mean:>7.2f}±{f_std:<5.2f}% ({sign}{delta:>6.2f} pp)"
            
            print(line)
            continue
        
        # Default: metrics in [0,1] shown as percentages
        b_mean = baseline_stats[metric]['mean'] * 100
        b_std = baseline_stats[metric]['std'] * 100
        line = f"{metric:<35} {b_mean:>6.2f}±{b_std:<4.2f}%"
        
        for lambda_val in lambda_values:
            if metric in lambda_stats[lambda_val]:
                f_mean = lambda_stats[lambda_val][metric]['mean'] * 100
                f_std = lambda_stats[lambda_val][metric]['std'] * 100
                delta_pp = f_mean - b_mean
                rel_impr = (delta_pp / b_mean * 100) if b_mean != 0 else 0.0
                
                sign_pp = '+' if delta_pp >= 0 else ''
                sign_rel = '+' if rel_impr >= 0 else ''
                
                line += (
                    f" {f_mean:>6.2f}±{f_std:<4.2f}% "
                    f"({sign_pp}{delta_pp:>5.2f} pp, {sign_rel}{rel_impr:>5.2f}%)"
                )
        
        print(line)
    
    # Print interpretation guide
    if use_ndkl or use_pa_aware or use_equality_gain:
        print("\nMetric Interpretation:")
        if use_ndkl:
            print("   NDKL@K: Lower is better (0 = perfectly fair ranking)")
            print("   Negative Δ in NDKL = improvement in fairness")
        if use_pa_aware:
            print("   PA_Bias_Score: Change in PA bias (negative = reduced bias)")
            print("   PA_Fairness_Score: Normalized fairness gain [-1, 1]")
            print("   PA_Overlap_Mean: Average overlap with PA recommendations")
        if use_equality_gain:
            print("   Equality_Gain@K: Percentage improvement over control (higher = more fair)")
            print("   Positive Δ = further improvement in equality")
        if use_centrality_gain:
            print("   *_EG@K (Centrality-based): Equality Gain for specific centrality measures")
            print("   Computed for: Degree, Katz, PageRank, Eigenvector, Closeness centrality")
            print("   Positive Δ = more equitable distribution across network positions")
        if use_stratified:
            print("   Stratified metrics show performance on low-degree (bottom 20%) and high-degree (top 20%) nodes")

def print_k_wise_summary(k_stats, lambda_values, k_values=[10, 20, 50, 100],
                        use_ndkl=False, use_pa_aware=False, use_equality_gain=False,
                        use_stratified=False):
    """Print K-wise summary table with optional PA, Equality Gain, and Stratified metrics."""
    print("\n" + "="*80)
    title = "K-WISE SUMMARY: Precision@K and Fairness@K Statistics"
    if use_equality_gain:
        title += " (with Equality Gain)"
    if use_ndkl:
        title += " (with NDKL)"
    if use_pa_aware:
        title += " (with PA metrics)"
    if use_stratified:
        title += " (with Stratified metrics)"
    print(title)
    print("="*80)
    
    for K in k_values:
        print(f"\n{'='*80}")
        print(f"K = {K}")
        print(f"{'='*80}")
        
        # Adjust header based on enabled metrics
        if use_ndkl and use_pa_aware:
            print(f"{'Method':<15} {'Precision@K':<18} {'Fairness@K':<18} "
                  f"{'NDKL@K':<18} {'PA_Bias':<18} "
                  f"{'ΔP':<12} {'ΔF':<12} {'ΔNDKL':<12} {'ΔPA_Bias':<12}")
            print("-" * 140)
        elif use_ndkl:
            print(f"{'Method':<15} {'Precision@K':<20} {'Fairness@K':<20} "
                  f"{'NDKL@K':<20} {'ΔP@K':<15} {'ΔF@K':<15} {'ΔNDKL@K':<15}")
            print("-" * 120)
        elif use_pa_aware:
            print(f"{'Method':<15} {'Precision@K':<20} {'Fairness@K':<20} "
                  f"{'PA_Bias':<20} {'ΔP@K':<15} {'ΔF@K':<15} {'ΔPA_Bias':<15}")
            print("-" * 120)
        else:
            print(f"{'Method':<20} {'Precision@K':<25} {'Fairness@K':<25} "
                  f"{'ΔP@K':<25} {'ΔF@K':<25}")
            print("-" * 120)
        
        # Baseline row
        b_p_mean = k_stats['baseline'][K]['precision']['mean'] * 100
        b_p_std = k_stats['baseline'][K]['precision']['std'] * 100
        b_f_mean = k_stats['baseline'][K]['fairness']['mean'] * 100
        b_f_std = k_stats['baseline'][K]['fairness']['std'] * 100
        
        baseline_row = f"{'Baseline':<15} {b_p_mean:>6.2f}±{b_p_std:<4.2f}%   {b_f_mean:>6.2f}±{b_f_std:<4.2f}%"
        
        if use_ndkl and 'ndkl' in k_stats['baseline'][K]:
            b_n_mean = k_stats['baseline'][K]['ndkl']['mean']
            b_n_std = k_stats['baseline'][K]['ndkl']['std']
            baseline_row += f"   {b_n_mean:>7.4f}±{b_n_std:<6.4f}"
        
        if use_pa_aware and 'pa_bias' in k_stats['baseline'][K]:
            b_pa_mean = k_stats['baseline'][K]['pa_bias']['mean']
            b_pa_std = k_stats['baseline'][K]['pa_bias']['std']
            baseline_row += f"   {b_pa_mean:>7.3f}±{b_pa_std:<5.3f}"
        
        baseline_row += "   --           --"
        if use_ndkl:
            baseline_row += "           --"
        if use_pa_aware:
            baseline_row += "           --"
        
        print(baseline_row)
        
        # Lambda values
        for lambda_val in lambda_values:
            lambda_key = f'lambda_{lambda_val}'
            l_p_mean = k_stats[lambda_key][K]['precision']['mean'] * 100
            l_p_std = k_stats[lambda_key][K]['precision']['std'] * 100
            l_f_mean = k_stats[lambda_key][K]['fairness']['mean'] * 100
            l_f_std = k_stats[lambda_key][K]['fairness']['std'] * 100
            
            delta_p = l_p_mean - b_p_mean
            delta_f = l_f_mean - b_f_mean
            
            rel_p = (delta_p / b_p_mean * 100) if b_p_mean != 0 else 0.0
            rel_f = (delta_f / b_f_mean * 100) if b_f_mean != 0 else 0.0
            
            p_sign_pp = '+' if delta_p >= 0 else ''
            f_sign_pp = '+' if delta_f >= 0 else ''
            
            lambda_row = (f"{'λ=' + str(lambda_val):<15} "
                         f"{l_p_mean:>6.2f}±{l_p_std:<4.2f}%   "
                         f"{l_f_mean:>6.2f}±{l_f_std:<4.2f}%")
            
            if use_ndkl and 'ndkl' in k_stats[lambda_key][K]:
                l_n_mean = k_stats[lambda_key][K]['ndkl']['mean']
                l_n_std = k_stats[lambda_key][K]['ndkl']['std']
                delta_n = l_n_mean - b_n_mean
                n_sign = '+' if delta_n >= 0 else ''
                lambda_row += f"   {l_n_mean:>7.4f}±{l_n_std:<6.4f}"
            
            if use_pa_aware and 'pa_bias' in k_stats[lambda_key][K]:
                l_pa_mean = k_stats[lambda_key][K]['pa_bias']['mean']
                l_pa_std = k_stats[lambda_key][K]['pa_bias']['std']
                delta_pa = l_pa_mean - b_pa_mean
                pa_sign = '+' if delta_pa >= 0 else ''
                lambda_row += f"   {l_pa_mean:>7.3f}±{l_pa_std:<5.3f}"
            
            lambda_row += f"   {p_sign_pp}{delta_p:>5.2f}pp    {f_sign_pp}{delta_f:>5.2f}pp"
            
            if use_ndkl:
                lambda_row += f"    {n_sign}{delta_n:>6.4f}"
            if use_pa_aware:
                lambda_row += f"    {pa_sign}{delta_pa:>6.3f}"
            
            print(lambda_row)


def compute_k_wise_statistics(all_results, lambda_values, k_values=[10, 20, 50, 100],
                              use_ndkl=False, use_pa_aware=False, use_equality_gain=False,
                              use_stratified=False):
    """Compute statistics grouped by K value, including PA, Equality Gain, and Stratified metrics if enabled."""
    k_stats = {'baseline': {}}
    
    for lambda_val in lambda_values:
        k_stats[f'lambda_{lambda_val}'] = {}
    
    for K in k_values:
        # Baseline stats
        precision_values = [run['baseline'][f'Precision@{K}'][1] for run in all_results]
        fairness_values = [run['baseline'][f'Distribution_Fairness@{K}'][1] for run in all_results]
        
        k_stats['baseline'][K] = {
            'precision': {
                'mean': np.mean(precision_values),
                'std': np.std(precision_values),
                'values': precision_values
            },
            'fairness': {
                'mean': np.mean(fairness_values),
                'std': np.std(fairness_values),
                'values': fairness_values
            }
        }
        
        if use_equality_gain:
            eg_values = [run['baseline'][f'Equality_Gain@{K}'][1] for run in all_results
                        if f'Equality_Gain@{K}' in run['baseline']]
            k_stats['baseline'][K]['equality_gain'] = {
                'mean': np.mean(eg_values) if eg_values else 0.0,
                'std': np.std(eg_values) if eg_values else 0.0,
                'values': eg_values
            }
        
        if use_stratified:
            # Stratified metrics
            stratified_keys = [
                (f'Stratified_Precision_Low@{K}', 'stratified_precision_low'),
                (f'Stratified_Precision_High@{K}', 'stratified_precision_high'),
                (f'Stratified_EqualityGain_Low@{K}', 'stratified_eq_gain_low'),
                (f'Stratified_EqualityGain_High@{K}', 'stratified_eq_gain_high')
            ]
            for metric_key, store_key in stratified_keys:
                values = [run['baseline'][metric_key][1] for run in all_results
                         if metric_key in run['baseline']]
                k_stats['baseline'][K][store_key] = {
                    'mean': np.mean(values) if values else 0.0,
                    'std': np.std(values) if values else 0.0,
                    'values': values
                }
        
        if use_ndkl:
            ndkl_values = [run['baseline'][f'NDKL@{K}'][1] for run in all_results
                          if f'NDKL@{K}' in run['baseline']]
            k_stats['baseline'][K]['ndkl'] = {
                'mean': np.mean(ndkl_values) if ndkl_values else 0.0,
                'std': np.std(ndkl_values) if ndkl_values else 0.0,
                'values': ndkl_values
            }
        
        if use_pa_aware:
            # PA metrics are NOT k-specific in results, but we can include them
            # Use PA_Bias_Score as a representative metric
            pa_values = [run['baseline']['PA_Bias_Score'][1] for run in all_results
                        if 'PA_Bias_Score' in run['baseline']]
            k_stats['baseline'][K]['pa_bias'] = {
                'mean': np.mean(pa_values) if pa_values else 0.0,
                'std': np.std(pa_values) if pa_values else 0.0,
                'values': pa_values
            }
        
        # Lambda stats
        for lambda_val in lambda_values:
            precision_values = [run[f'lambda_{lambda_val}'][f'Precision@{K}'][1] for run in all_results]
            fairness_values = [run[f'lambda_{lambda_val}'][f'Distribution_Fairness@{K}'][1] for run in all_results]
            
            k_stats[f'lambda_{lambda_val}'][K] = {
                'precision': {
                    'mean': np.mean(precision_values),
                    'std': np.std(precision_values),
                    'values': precision_values
                },
                'fairness': {
                    'mean': np.mean(fairness_values),
                    'std': np.std(fairness_values),
                    'values': fairness_values
                }
            }
            
            if use_equality_gain:
                eg_values = [run[f'lambda_{lambda_val}'][f'Equality_Gain@{K}'][1] for run in all_results
                            if f'Equality_Gain@{K}' in run[f'lambda_{lambda_val}']]
                k_stats[f'lambda_{lambda_val}'][K]['equality_gain'] = {
                    'mean': np.mean(eg_values) if eg_values else 0.0,
                    'std': np.std(eg_values) if eg_values else 0.0,
                    'values': eg_values
                }
            
            if use_stratified:
                # Stratified metrics
                stratified_keys = [
                    (f'Stratified_Precision_Low@{K}', 'stratified_precision_low'),
                    (f'Stratified_Precision_High@{K}', 'stratified_precision_high'),
                    (f'Stratified_EqualityGain_Low@{K}', 'stratified_eq_gain_low'),
                    (f'Stratified_EqualityGain_High@{K}', 'stratified_eq_gain_high')
                ]
                for metric_key, store_key in stratified_keys:
                    values = [run[f'lambda_{lambda_val}'][metric_key][1] for run in all_results
                             if metric_key in run[f'lambda_{lambda_val}']]
                    k_stats[f'lambda_{lambda_val}'][K][store_key] = {
                        'mean': np.mean(values) if values else 0.0,
                        'std': np.std(values) if values else 0.0,
                        'values': values
                    }
            
            if use_ndkl:
                ndkl_values = [run[f'lambda_{lambda_val}'][f'NDKL@{K}'][1] for run in all_results
                              if f'NDKL@{K}' in run[f'lambda_{lambda_val}']]
                k_stats[f'lambda_{lambda_val}'][K]['ndkl'] = {
                    'mean': np.mean(ndkl_values) if ndkl_values else 0.0,
                    'std': np.std(ndkl_values) if ndkl_values else 0.0,
                    'values': ndkl_values
                }
            
            if use_pa_aware:
                pa_values = [run[f'lambda_{lambda_val}']['PA_Bias_Score'][1] for run in all_results
                            if 'PA_Bias_Score' in run[f'lambda_{lambda_val}']]
                k_stats[f'lambda_{lambda_val}'][K]['pa_bias'] = {
                    'mean': np.mean(pa_values) if pa_values else 0.0,
                    'std': np.std(pa_values) if pa_values else 0.0,
                    'values': pa_values
                }
    
    return k_stats


def save_results(filepath, experiment_info, baseline_stats, lambda_stats, 
                lambda_values, k_stats=None, individual_runs=None):
    """Save detailed results to JSON file."""
    results_dict = {
        'experiment_info': experiment_info,
        'lambda_results': {
            f'lambda_{lv}': {'statistics': _serialize_stats(lambda_stats[lv])}
            for lv in lambda_values
        }
    }

    if baseline_stats is not None:
        results_dict['baseline'] = {
            'statistics': _serialize_stats(baseline_stats)
        }
    
    if k_stats:
        results_dict['k_wise_statistics'] = _serialize_k_stats(k_stats)
    
    if individual_runs:
        results_dict['individual_runs'] = _serialize_individual_runs(
            individual_runs, lambda_values
        )
    
    with open(filepath, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n Detailed results saved to {filepath}")


def _serialize_stats(stats):
    """Convert statistics dict to JSON-serializable format."""
    
    def _convert_value(val):
        """Convert various types to JSON-serializable format."""
        if isinstance(val, np.ndarray):
            return val.tolist()
        elif isinstance(val, (np.floating, np.integer)):
            return float(val)
        elif isinstance(val, list):
            # Recursively convert list elements
            return [_convert_value(x) for x in val]
        elif isinstance(val, tuple):
            return [_convert_value(x) for x in val]
        elif hasattr(val, 'item'):  # PyTorch tensor
            return float(val.item())
        elif isinstance(val, (int, float, str, bool, type(None))):
            return val
        else:
            return str(val)  # Fallback for unknown types
    
    return {
        k: {
            key: _convert_value(v[key])
            for key in ['mean', 'std', 'min', 'max', 'values']
            if key in v
        }
        for k, v in stats.items()
    }

def _serialize_k_stats(k_stats):
    """Convert K-wise statistics to JSON-serializable format."""
    
    def _convert_value(val):
        """Convert to JSON-serializable format."""
        if isinstance(val, (np.ndarray, np.generic)):
            return val.tolist() if isinstance(val, np.ndarray) else float(val)
        elif hasattr(val, 'item'):  # PyTorch tensor
            return float(val.item())
        elif isinstance(val, list):
            return [_convert_value(x) for x in val]
        elif isinstance(val, (int, float, str, bool, type(None))):
            return val
        else:
            return str(val)
    
    serialized = {}
    for method_key in k_stats:
        serialized[method_key] = {}
        for K in k_stats[method_key]:
            serialized[method_key][K] = {}
            for metric_type in k_stats[method_key][K]:
                serialized[method_key][K][metric_type] = {
                    'mean': _convert_value(k_stats[method_key][K][metric_type]['mean']),
                    'std': _convert_value(k_stats[method_key][K][metric_type]['std']),
                    'values': _convert_value(k_stats[method_key][K][metric_type]['values'])
                }
    return serialized

def _serialize_individual_runs(individual_runs, lambda_values):
    """Convert individual run results to JSON-serializable format."""
    
    def _make_serializable(val):
        """Convert any value to JSON-serializable format."""
        if isinstance(val, (np.ndarray, np.generic)):
            return val.tolist() if isinstance(val, np.ndarray) else float(val)
        elif isinstance(val, (tuple, list)):
            if len(val) >= 2:
                # Handle (name, value) tuples
                return _make_serializable(val[1])
            return [_make_serializable(x) for x in val]
        elif hasattr(val, 'item'):  # PyTorch tensor
            return float(val.item())
        elif isinstance(val, (int, float, str, bool, type(None))):
            return val
        elif isinstance(val, dict):
            return {k: _make_serializable(v) for k, v in val.items()}
        else:
            return str(val)  # Fallback
    
    serialized = []
    for run_idx, run_results in enumerate(individual_runs):
        run_dict = {
            'run': run_idx + 1,
        }
        if run_results.get('baseline') is not None:
            run_dict['baseline'] = {
                k: _make_serializable(v)
                for k, v in run_results['baseline'].items()
                if not k.startswith('_')  # Exclude internal keys
            }
        for lambda_val in lambda_values:
            run_dict[f'lambda_{lambda_val}'] = {
                k: _make_serializable(v)
                for k, v in run_results[f'lambda_{lambda_val}'].items()
                if not k.startswith('_')  # Exclude internal keys
            }
        serialized.append(run_dict)
    return serialized

# def create_experiment_info(args, method_name):
#     """Create experiment info dict from arguments."""
#     info = {
#         'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#         'dataset': args.data_name,
#         'encoder': args.encoder,
#         'predictor': args.predictor,
#         'fairness_method': method_name,
#         'fairness_mode': args.fairness_mode,
#         'num_runs': args.runs,
#         'epochs': args.epochs,
#         'batch_size': args.batch_size,
#         'lr': args.lr,
#     }
    
#     # Add PA-aware and NDKL info if enabled
#     if getattr(args, 'use_pa_aware', False):
#         info['use_pa_aware'] = True
    
#     if getattr(args, 'use_ndkl', False):
#         info['use_ndkl'] = True
#         info['ndkl_mode'] = getattr(args, 'ndkl_mode', 'per_user')
#         info['eval_user_mode'] = getattr(args, 'eval_user_mode', 'test_users')
    
#     return info


def create_experiment_info(args, method_name):
    """Create experiment info dict from arguments."""
    info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': args.data_name,
        'encoder': getattr(args, 'encoder', 'N/A'),  # <- Add default
        'predictor': getattr(args, 'predictor', 'N/A'),  # <- Add default
        'fairness_method': method_name,
        'fairness_mode': args.fairness_mode,
        'eg_graph_semantics': 'simple_undirected_set_union',
        'num_runs': args.runs,
        'epochs': getattr(args, 'epochs', 0),  # <- Add default
        'batch_size': getattr(args, 'batch_size', 0),  # <- Add default
        'lr': getattr(args, 'lr', 0.0),  # <- Add default
    }
    
    # Add baseline-specific info if applicable
    if hasattr(args, 'baseline'):
        info['baseline_model'] = args.baseline

    # Only runners that actually expose --penalty_reduction implement the
    # per-model reseeding rule below. Recording these unconditionally would
    # stamp a false seed_rule onto every other experiment's results.
    if hasattr(args, 'penalty_reduction'):
        info['penalty_reduction'] = args.penalty_reduction
        info['num_neg'] = getattr(args, 'num_neg', None)
        # seed_rule describes the OPT-IN per-model reseeding only; without the
        # flag the runner keeps one evolving RNG stream and no rule applies.
        if getattr(args, 'reset_seed_per_model', False):
            info['seed_base'] = getattr(args, 'seed', None)
            info['seed_rule'] = 'model_seed = seed_base + run_index'
    
    # Add PA-aware and NDKL info if enabled
    if getattr(args, 'use_pa_aware', False):
        info['use_pa_aware'] = True
    
    if getattr(args, 'use_ndkl', False):
        info['use_ndkl'] = True
        info['ndkl_mode'] = getattr(args, 'ndkl_mode', 'per_user')
        info['eval_user_mode'] = getattr(args, 'eval_user_mode', 'test_users')
    
    return info

# Aliases for compatibility
def print_enhanced_results_table(baseline_stats, lambda_stats, lambda_values,
                                use_pa_aware=False, use_ndkl=False,
                                use_equality_gain=False, use_stratified=False, use_centrality_gain=False):
    """Wrapper for print_summary_table."""
    return print_summary_table(baseline_stats, lambda_stats, lambda_values,
                              use_pa_aware=use_pa_aware, use_ndkl=use_ndkl,
                              use_equality_gain=use_equality_gain,
                              use_stratified=use_stratified, use_centrality_gain=use_centrality_gain)


def save_multi_lambda_results(args, all_results, baseline_stats, lambda_stats, 
                              lambda_values, k_stats):
    """Wrapper for save_results."""
    info = create_experiment_info(args, args.fairness_method)
    return save_results(args.results_file, info, baseline_stats, lambda_stats, 
                       lambda_values, k_stats, all_results)
