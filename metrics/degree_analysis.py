

"""
Degree-based Analysis and Plotting

Analyzes precision and equality gain as a function of node degree.
Now supports comparison plots across multiple methods.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from collections import defaultdict
from metrics.equality_gain import compute_equality_gain


def compute_equality_gain_vs_degree_by_source(
    recs_by_u: Dict[int, List[int]],
    train_edges: torch.Tensor,
    test_edges: torch.Tensor,
    degrees: torch.Tensor,
    num_nodes: int,
    k: int,
    num_bins: int = 10,
    fairness_mode: str = 'indegree'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute equality gain@K as a function of SOURCE node degree.
    
    Measures: "Do nodes at degree X recommend more equally than they naturally connect?"
    
    Args:
        recs_by_u: Recommendations
        train_edges: Training edges
        test_edges: Test edges
        degrees: Node degrees
        num_nodes: Total number of nodes
        k: Cutoff value
        num_bins: Number of degree bins
        fairness_mode: Degree counting mode
    
    Returns:
        (bin_centers, equality_gains)
    """
    degrees_np = degrees.cpu().numpy()
    deg_min, deg_max = degrees_np.min(), degrees_np.max()
    
    if deg_max == deg_min:
        # All nodes have same degree
        eq_gain, _, _ = compute_equality_gain(
            recs_by_u, train_edges, test_edges, num_nodes, k, fairness_mode
        )
        return np.array([deg_min]), np.array([eq_gain])
    
    bins = np.linspace(deg_min, deg_max, num_bins + 1)
    
    bin_centers = []
    bin_eq_gains = []
    
    for i in range(num_bins):
        # Get nodes in this bin
        bin_mask = (degrees_np >= bins[i]) & (degrees_np < bins[i + 1])
        bin_nodes = set(np.where(bin_mask)[0].tolist())
        
        if len(bin_nodes) == 0:
            continue
        
        # Filter by SOURCE only
        test_bin = test_edges[[u in bin_nodes for u, v in test_edges.tolist()]]
        recs_bin = {u: vs for u, vs in recs_by_u.items() if u in bin_nodes}
        
        if len(test_bin) == 0:
            continue
        
        # Compute equality gain for this bin
        eq_gain, _, _ = compute_equality_gain(
            recs_bin, train_edges, test_bin, num_nodes, k, fairness_mode
        )
        
        bin_centers.append((bins[i] + bins[i + 1]) / 2)
        bin_eq_gains.append(eq_gain)
    
    return np.array(bin_centers), np.array(bin_eq_gains)



def compute_equality_gain_vs_degree_by_target(
    recs_by_u: Dict[int, List[int]],
    train_edges: torch.Tensor,
    test_edges: torch.Tensor,
    degrees: torch.Tensor,
    num_nodes: int,
    k: int,
    num_bins: int = 10,
    fairness_mode: str = 'indegree'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute equality gain@K as a function of TARGET node degree.
    
    Measures: "Are nodes at degree X receiving more equal attention?"
    
    Args:
        recs_by_u: Recommendations
        train_edges: Training edges
        test_edges: Test edges
        degrees: Node degrees
        num_nodes: Total number of nodes
        k: Cutoff value
        num_bins: Number of degree bins
        fairness_mode: Degree counting mode (should be 'indegree' for target)
    
    Returns:
        (bin_centers, equality_gains)
    """
    from metrics.equality_gain import compute_gini_coefficient
    
    degrees_np = degrees.cpu().numpy()
    deg_min, deg_max = degrees_np.min(), degrees_np.max()
    
    if deg_max == deg_min:
        # All nodes have same degree - compute overall
        eq_gain, _, _ = compute_equality_gain(
            recs_by_u, train_edges, test_edges, num_nodes, k, fairness_mode
        )
        return np.array([deg_min]), np.array([eq_gain])
    
    bins = np.linspace(deg_min, deg_max, num_bins + 1)
    
    bin_centers = []
    bin_eq_gains = []
    
    for i in range(num_bins):
        # Get TARGET nodes in this bin
        bin_mask = (degrees_np >= bins[i]) & (degrees_np < bins[i + 1])
        bin_nodes = set(np.where(bin_mask)[0].tolist())
        
        if len(bin_nodes) == 0:
            continue
        
        # Control: count indegrees from test edges TO nodes in this bin
        deg_ctrl = np.zeros(num_nodes)
        for u, v in test_edges.tolist():
            if v in bin_nodes:  # v is TARGET
                deg_ctrl[v] += 1
        
        # Predicted: count indegrees from recommendations TO nodes in this bin
        deg_pred = np.zeros(num_nodes)
        for u, vs in recs_by_u.items():
            for v in vs[:k]:
                if v in bin_nodes:  # v is TARGET
                    deg_pred[v] += 1
        
        # Compute Gini only over nodes in this bin
        deg_ctrl_filtered = deg_ctrl[list(bin_nodes)]
        deg_pred_filtered = deg_pred[list(bin_nodes)]
        
        # Skip if no edges in control
        if deg_ctrl_filtered.sum() == 0:
            continue
        
        gini_ctrl = compute_gini_coefficient(deg_ctrl_filtered)
        gini_pred = compute_gini_coefficient(deg_pred_filtered)
        
        if gini_ctrl > 0:
            eq_gain = ((gini_ctrl - gini_pred) / gini_ctrl) * 100.0
        else:
            eq_gain = 0.0
        
        bin_centers.append((bins[i] + bins[i + 1]) / 2)
        bin_eq_gains.append(eq_gain)
    
    return np.array(bin_centers), np.array(bin_eq_gains)


def analyze_degree_effects(
    recs_by_u: Dict[int, List[int]],
    train_edges: torch.Tensor,
    test_edges: torch.Tensor,
    degrees: torch.Tensor,
    num_nodes: int,
    k: int = 10,
    num_bins: int = 10,
    fairness_mode: str = 'indegree',
    stratification_mode: str = 'source',  # NEW PARAMETER
    save_dir: str = None
) -> Dict[str, np.ndarray]:
    """
    Single-method degree-based analysis.
    
    Args:
        recs_by_u: Recommendations
        train_edges: Training edges
        test_edges: Test edges
        degrees: Node degrees
        num_nodes: Total number of nodes
        k: Cutoff value
        num_bins: Number of degree bins
        fairness_mode: Degree counting mode
        stratification_mode: 'source', 'target', or 'mixed' (default: 'source')
        save_dir: Directory to save plots (optional)
    
    Returns:
        dict with analysis results
    """
    print("\n" + "="*80)
    print(f"DEGREE-BASED ANALYSIS @ K={k} (Mode: {stratification_mode.upper()})")
    print("="*80)
    
    # Compute precision vs degree (always by SOURCE)
    print("\nComputing Precision vs Degree (by SOURCE)...")
    bin_centers_prec, mean_prec, std_prec = compute_precision_vs_degree(
        recs_by_u, test_edges, degrees, k, num_bins
    )
    
    # Compute equality gain vs degree based on mode
    print(f"Computing Equality Gain vs Degree (by {stratification_mode.upper()})...")
    if stratification_mode == 'source':
        bin_centers_eq, eq_gains = compute_equality_gain_vs_degree_by_source(
            recs_by_u, train_edges, test_edges, degrees, num_nodes,
            k, num_bins, fairness_mode
        )
    elif stratification_mode == 'target':
        bin_centers_eq, eq_gains = compute_equality_gain_vs_degree_by_target(
            recs_by_u, train_edges, test_edges, degrees, num_nodes,
            k, num_bins, fairness_mode
        )
    elif stratification_mode == 'mixed':
        bin_centers_eq, eq_gains = compute_equality_gain_vs_degree(
            recs_by_u, train_edges, test_edges, degrees, num_nodes,
            k, num_bins, fairness_mode
        )
    else:
        raise ValueError(f"Invalid stratification_mode: {stratification_mode}")
    
    # Print summary
    print("\n" + "-"*80)
    print("PRECISION vs DEGREE (by SOURCE):")
    print("-"*80)
    for deg, prec, std in zip(bin_centers_prec, mean_prec, std_prec):
        print(f"  Degree ~{deg:>6.1f}: Precision = {prec*100:>6.2f}% ± {std*100:>5.2f}%")
    
    print("\n" + "-"*80)
    print(f"EQUALITY GAIN vs DEGREE (by {stratification_mode.upper()}):")
    print("-"*80)
    for deg, eq in zip(bin_centers_eq, eq_gains):
        print(f"  Degree ~{deg:>6.1f}: Equality Gain = {eq:>7.2f}%")
    print("="*80)
    
    return {
        'precision_bin_centers': bin_centers_prec,
        'precision_mean': mean_prec,
        'precision_std': std_prec,
        'equality_gain_bin_centers': bin_centers_eq,
        'equality_gains': eq_gains,
        'stratification_mode': stratification_mode
    }


def analyze_degree_effects_comparison(
    all_recs: Dict[str, Dict[int, List[int]]],
    train_edges: torch.Tensor,
    test_edges: torch.Tensor,
    degrees: torch.Tensor,
    num_nodes: int,
    k: int = 10,
    num_bins: int = 10,
    fairness_mode: str = 'indegree',
    stratification_mode: str = 'source',  # NEW PARAMETER
    save_dir: str = None
) -> Dict[str, Dict]:
    """
    Complete degree-based analysis with comparison plots across methods.
    
    Args:
        all_recs: Dict[method_name] -> recommendations
        train_edges: Training edges
        test_edges: Test edges
        degrees: Node degrees
        num_nodes: Total number of nodes
        k: Cutoff value
        num_bins: Number of degree bins
        fairness_mode: Degree counting mode
        stratification_mode: 'source', 'target', or 'mixed' (default: 'source')
        save_dir: Directory to save plots
    
    Returns:
        dict with analysis results for all methods
    """
    print("\n" + "="*80)
    print(f"DEGREE-BASED COMPARISON ANALYSIS @ K={k} (Mode: {stratification_mode.upper()})")
    print("="*80)
    
    all_precision_results = {}
    all_eq_gain_results = {}
    
    # Compute metrics for each method
    for method_name, recs_by_u in all_recs.items():
        print(f"\nProcessing {method_name}...")
        
        # Compute precision vs degree (always by SOURCE)
        bin_centers_prec, mean_prec, std_prec = compute_precision_vs_degree(
            recs_by_u, test_edges, degrees, k, num_bins
        )
        all_precision_results[method_name] = (bin_centers_prec, mean_prec, std_prec)
        
        # Compute equality gain vs degree based on mode
        if stratification_mode == 'source':
            bin_centers_eq, eq_gains = compute_equality_gain_vs_degree_by_source(
                recs_by_u, train_edges, test_edges, degrees, num_nodes,
                k, num_bins, fairness_mode
            )
        elif stratification_mode == 'target':
            bin_centers_eq, eq_gains = compute_equality_gain_vs_degree_by_target(
                recs_by_u, train_edges, test_edges, degrees, num_nodes,
                k, num_bins, fairness_mode
            )
        elif stratification_mode == 'mixed':
            bin_centers_eq, eq_gains = compute_equality_gain_vs_degree(
                recs_by_u, train_edges, test_edges, degrees, num_nodes,
                k, num_bins, fairness_mode
            )
        else:
            raise ValueError(f"Invalid stratification_mode: {stratification_mode}")
            
        all_eq_gain_results[method_name] = (bin_centers_eq, eq_gains)
    
    # Print comparison summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY:")
    print("="*80)
    
    for method_name in all_recs.keys():
        bin_centers_prec, mean_prec, std_prec = all_precision_results[method_name]
        bin_centers_eq, eq_gains = all_eq_gain_results[method_name]
        
        print(f"\n{method_name}:")
        print(f"  Avg Precision@{k}: {np.mean(mean_prec)*100:.2f}% ± {np.mean(std_prec)*100:.2f}%")
        print(f"  Avg Equality Gain@{k}: {np.mean(eq_gains):.2f}%")
    
    print("="*80)
    
    # Generate comparison plots
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        prec_path = os.path.join(save_dir, f'precision_vs_degree_comparison_k{k}_{stratification_mode}.png')
        eq_path = os.path.join(save_dir, f'equality_gain_vs_degree_comparison_k{k}_{stratification_mode}.png')
    else:
        prec_path = None
        eq_path = None
    
    print("\nGenerating comparison plots...")
    plot_precision_comparison(all_precision_results, k, prec_path)
    plot_equality_gain_comparison(all_eq_gain_results, k, eq_path)
    
    return {
        'precision_results': all_precision_results,
        'eq_gain_results': all_eq_gain_results,
        'stratification_mode': stratification_mode
    }


def compute_precision_vs_degree(
    recs_by_u: Dict[int, List[int]],
    test_edges: torch.Tensor,
    degrees: torch.Tensor,
    k: int,
    num_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precision@K as a function of node degree.
    
    Args:
        recs_by_u: Recommendations
        test_edges: Test edges
        degrees: Node degrees
        k: Cutoff value
        num_bins: Number of degree bins
    
    Returns:
        (bin_centers, mean_precision, std_precision)
    """
    # Build test edge set
    test_edge_set = set()
    for u, v in test_edges.tolist():
        test_edge_set.add((u, v))
        test_edge_set.add((v, u))
    
    # Compute precision for each node
    node_precision = {}
    for u, vs in recs_by_u.items():
        topk = vs[:k]
        if not topk:
            continue
        hits = sum(1 for v in topk if (u, v) in test_edge_set)
        node_precision[u] = hits / len(topk)
    
    # Group by degree bins
    degrees_np = degrees.cpu().numpy()
    deg_min, deg_max = degrees_np.min(), degrees_np.max()
    
    if deg_max == deg_min:
        # All nodes have same degree
        return np.array([deg_min]), np.array([np.mean(list(node_precision.values()))]), np.array([0.0])
    
    bins = np.linspace(deg_min, deg_max, num_bins + 1)
    bin_precision = defaultdict(list)
    
    for node_id, prec in node_precision.items():
        deg = degrees_np[node_id]
        bin_idx = min(np.digitize(deg, bins) - 1, num_bins - 1)
        bin_precision[bin_idx].append(prec)
    
    # Compute statistics per bin
    bin_centers = []
    mean_prec = []
    std_prec = []
    
    for i in range(num_bins):
        if i in bin_precision and len(bin_precision[i]) > 0:
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            mean_prec.append(np.mean(bin_precision[i]))
            std_prec.append(np.std(bin_precision[i]))
    
    return np.array(bin_centers), np.array(mean_prec), np.array(std_prec)


def compute_equality_gain_vs_degree(
    recs_by_u: Dict[int, List[int]],
    train_edges: torch.Tensor,
    test_edges: torch.Tensor,
    degrees: torch.Tensor,
    num_nodes: int,
    k: int,
    num_bins: int = 10,
    fairness_mode: str = 'indegree'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute equality gain@K as a function of node degree.
    
    Strategy: For each degree bin, compute equality gain considering only
    edges and recommendations involving nodes in that bin.
    
    Args:
        recs_by_u: Recommendations
        train_edges: Training edges
        test_edges: Test edges
        degrees: Node degrees
        num_nodes: Total number of nodes
        k: Cutoff value
        num_bins: Number of degree bins
        fairness_mode: Degree counting mode
    
    Returns:
        (bin_centers, equality_gains)
    """
    degrees_np = degrees.cpu().numpy()
    deg_min, deg_max = degrees_np.min(), degrees_np.max()
    
    if deg_max == deg_min:
        # All nodes have same degree
        eq_gain, _, _ = compute_equality_gain(
            recs_by_u, train_edges, test_edges, num_nodes, k, fairness_mode
        )
        return np.array([deg_min]), np.array([eq_gain])
    
    bins = np.linspace(deg_min, deg_max, num_bins + 1)
    
    bin_centers = []
    bin_eq_gains = []
    
    for i in range(num_bins):
        # Get nodes in this bin
        bin_mask = (degrees_np >= bins[i]) & (degrees_np < bins[i + 1])
        bin_nodes = set(np.where(bin_mask)[0].tolist())
        
        if len(bin_nodes) == 0:
            continue
        
        # Filter edges and recommendations for this bin
        train_bin = train_edges[[u in bin_nodes or v in bin_nodes 
                                for u, v in train_edges.tolist()]]
        test_bin = test_edges[[u in bin_nodes or v in bin_nodes 
                              for u, v in test_edges.tolist()]]
        recs_bin = {u: vs for u, vs in recs_by_u.items() if u in bin_nodes}
        
        if len(train_bin) == 0 or len(test_bin) == 0:
            continue
        
        # Compute equality gain for this bin
        eq_gain, _, _ = compute_equality_gain(
            recs_bin, train_bin, test_bin, num_nodes, k, fairness_mode
        )
        
        bin_centers.append((bins[i] + bins[i + 1]) / 2)
        bin_eq_gains.append(eq_gain)
    
    return np.array(bin_centers), np.array(bin_eq_gains)


def plot_precision_comparison(
    all_results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    k: int,
    save_path: str = None
):
    """
    Plot Precision@K vs Degree for multiple methods.
    
    Args:
        all_results: Dict[method_name] -> (bin_centers, mean_prec, std_prec)
        k: Cutoff value
        save_path: Path to save figure
    """
    plt.figure(figsize=(12, 7))
    
    # Colors and markers for different methods
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    for idx, (method_name, (bin_centers, mean_prec, std_prec)) in enumerate(all_results.items()):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        plt.errorbar(bin_centers, mean_prec * 100, yerr=std_prec * 100,
                    fmt=f'{marker}-', capsize=5, linewidth=2, markersize=8,
                    label=method_name, color=color, alpha=0.8)
    
    plt.xlabel('Node Degree (Training Graph)', fontsize=13)
    plt.ylabel(f'Precision@{k} (%)', fontsize=13)
    plt.title(f'Precision@{k} vs Node Degree: Method Comparison', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_equality_gain_comparison(
    all_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    k: int,
    save_path: str = None
):
    """
    Plot Equality Gain@K vs Degree for multiple methods.
    
    Args:
        all_results: Dict[method_name] -> (bin_centers, eq_gains)
        k: Cutoff value
        save_path: Path to save figure
    """
    plt.figure(figsize=(12, 7))
    
    # Colors and markers for different methods
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    for idx, (method_name, (bin_centers, eq_gains)) in enumerate(all_results.items()):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        plt.plot(bin_centers, eq_gains, f'{marker}-', linewidth=2, markersize=8,
                label=method_name, color=color, alpha=0.8)
    
    plt.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5,
               label='No Change')
    
    plt.xlabel('Node Degree (Training Graph)', fontsize=13)
    plt.ylabel(f'Equality Gain@{k} (%)', fontsize=13)
    plt.title(f'Equality Gain@{k} vs Node Degree: Method Comparison', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_single_method_degree_analysis(
    bin_centers_prec, mean_prec, std_prec,
    bin_centers_eq, eq_gains,
    k,
    save_dir
):
    """Helper to plot single-method analysis."""
    import os
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Precision Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(bin_centers_prec, mean_prec * 100, yerr=std_prec * 100,
                fmt='o-', capsize=5, linewidth=2, markersize=8,
                label=f'Precision@{k}', color='#1f77b4')
    
    plt.xlabel('Node Degree (Training Graph)', fontsize=12)
    plt.ylabel(f'Precision@{k} (%)', fontsize=12)
    plt.title(f'Precision@{k} vs Node Degree', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'precision_vs_degree_k{k}.png'), dpi=300)
    plt.close()
    
    # 2. Equality Gain Plot
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers_eq, eq_gains, 's-', linewidth=2, markersize=8,
            label=f'Equality Gain@{k}', color='#ff7f0e')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='No Change')
    
    plt.xlabel('Node Degree (Training Graph)', fontsize=12)
    plt.ylabel(f'Equality Gain@{k} (%)', fontsize=12)
    plt.title(f'Equality Gain@{k} vs Node Degree', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'equality_gain_vs_degree_k{k}.png'), dpi=300)
    plt.close()


def plot_aggregated_degree_analysis(
    all_precision_results: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    all_eq_gain_results: List[Tuple[np.ndarray, np.ndarray]],
    k: int,
    save_dir: str
):
    """
    Plot Aggregated Degree Analysis (Mean +/- Std across runs).
    """
    import os
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    
    # --- Helper to aggregate ---
    def aggregate_metrics(results_list):
        # results_list is list of (bin_centers, means, [stds])
        # Assuming bin_centers are identical across runs for the same dataset/bins
        if not results_list:
            return None, None, None
            
        # Check consistency of bins
        ref_bins = results_list[0][0]
        
        # Stack values
        values_stack = []
        for res in results_list:
            # Resample if needed? For now assume exact bin match
            if len(res[0]) != len(ref_bins):
                # Simple fallback: skip incompatible runs or truncated
                continue
            values_stack.append(res[1])
            
        if not values_stack:
            return None, None, None
            
        values_matrix = np.vstack(values_stack)
        mean_values = np.mean(values_matrix, axis=0)
        std_values = np.std(values_matrix, axis=0)
        
        return ref_bins, mean_values, std_values

    # 1. Precision Aggregation
    bin_centers, mean_prec, std_prec = aggregate_metrics(all_precision_results)
    
    if bin_centers is not None:
        plt.figure(figsize=(10, 6))
        
        # Plot mean line
        plt.plot(bin_centers, mean_prec * 100, 'o-', linewidth=2, markersize=6, 
                 label=f'Mean Precision@{k}', color='#1f77b4')
        
        # Plot shaded error region (Mean +/- Std)
        plt.fill_between(bin_centers, 
                         (mean_prec - std_prec) * 100, 
                         (mean_prec + std_prec) * 100,
                         color='#1f77b4', alpha=0.2, label='Std Dev (Across Runs)')
        
        plt.xlabel('Node Degree (Training Graph)', fontsize=12)
        plt.ylabel(f'Precision@{k} (%)', fontsize=12)
        plt.title(f'Precision@{k} vs Node Degree (Aggregated over {len(all_precision_results)} runs)', 
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'precision_vs_degree_AGGR_k{k}.png'), dpi=300)
        plt.close()
        
    # 2. Equality Gain Aggregation
    bin_centers_eq, mean_eq, std_eq = aggregate_metrics(all_eq_gain_results)
    
    if bin_centers_eq is not None:
        plt.figure(figsize=(10, 6))
        
        # Plot mean line
        plt.plot(bin_centers_eq, mean_eq, 's-', linewidth=2, markersize=6, 
                 label=f'Mean Equality Gain@{k}', color='#ff7f0e')
        
        # Plot shaded error region
        plt.fill_between(bin_centers_eq, 
                         mean_eq - std_eq, 
                         mean_eq + std_eq,
                         color='#ff7f0e', alpha=0.2, label='Std Dev (Across Runs)')
        
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='No Change')
        
        plt.xlabel('Node Degree (Training Graph)', fontsize=12)
        plt.ylabel(f'Equality Gain@{k} (%)', fontsize=12)
        plt.title(f'Equality Gain@{k} vs Node Degree (Aggregated over {len(all_eq_gain_results)} runs)', 
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'equality_gain_vs_degree_AGGR_k{k}.png'), dpi=300)
        plt.close()


def analyze_degree_effects(
    recs_by_u: Dict[int, List[int]],
    train_edges: torch.Tensor,
    test_edges: torch.Tensor,
    degrees: torch.Tensor,
    num_nodes: int,
    k: int = 10,
    num_bins: int = 10,
    fairness_mode: str = 'indegree',
    save_dir: str = None
) -> Dict[str, np.ndarray]:
    """
    Single-method degree-based analysis with PLOTTING.
    """
    print("\n" + "="*80)
    print(f"DEGREE-BASED ANALYSIS @ K={k}")
    print("="*80)
    
    # Compute precision vs degree
    print("\nComputing Precision vs Degree...")
    bin_centers_prec, mean_prec, std_prec = compute_precision_vs_degree(
        recs_by_u, test_edges, degrees, k, num_bins
    )
    
    # Compute equality gain vs degree
    print("Computing Equality Gain vs Degree...")
    bin_centers_eq, eq_gains = compute_equality_gain_vs_degree(
        recs_by_u, train_edges, test_edges, degrees, num_nodes,
        k, num_bins, fairness_mode
    )
    
    # Print summary
    print("\n" + "-"*80)
    print("PRECISION vs DEGREE:")
    print("-"*80)
    for deg, prec, std in zip(bin_centers_prec, mean_prec, std_prec):
        print(f"  Degree ~{deg:>6.1f}: Precision = {prec*100:>6.2f}% +/- {std*100:>5.2f}%")
    
    print("\n" + "-"*80)
    print("EQUALITY GAIN vs DEGREE:")
    print("-"*80)
    for deg, eq in zip(bin_centers_eq, eq_gains):
        print(f"  Degree ~{deg:>6.1f}: Equality Gain = {eq:>7.2f}%")
    print("="*80)
    
    # SAVE PLOTS
    if save_dir:
        print(f"\nSaving plots to {save_dir}...")
        plot_single_method_degree_analysis(
            bin_centers_prec, mean_prec, std_prec,
            bin_centers_eq, eq_gains,
            k, save_dir
        )
    
    return {
        'precision_bin_centers': bin_centers_prec,
        'precision_mean': mean_prec,
        'precision_std': std_prec,
        'equality_gain_bin_centers': bin_centers_eq,
        'equality_gains': eq_gains
    }