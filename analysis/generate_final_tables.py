"""
Final Results Table Generator
=============================
Reads all experiment results from d_final_results_compare/,
selects best configurations per method per dataset using the
Pareto front + ASF logic from pareto_hypervolume_analysis.py,
and outputs two paper-style CSV tables:

  Table 1: Precision@10 + Equality Gain metrics
           (Equality_Gain@10, Pagerank_EG@10, Closeness_EG@10)

  Table 2: Precision@10 + Demographic metrics
           (Distribution_Fairness@10, NDKL@10)
"""

import json
import os
import glob
import pandas as pd
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import Hypervolume
from pymoo.decomposition.asf import ASF

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = "/home/eerdem/workspace/HL-GNN/fairness_experiments_scaffold/d_final_results_compare"
OUTPUT_DIR = BASE_DIR

# All metrics we want to extract
ALL_METRICS = [
    "Precision@10",
    "Equality_Gain@10",
    "Pagerank_EG@10",
    "Closeness_EG@10",
    "Distribution_Fairness@10",
    "NDKL@10",
    "Delta_SP@10",
    "Delta_EO@10",
    "IF",
    "GDIF",
    "Ranking_IF@10",
]

# Objectives for Pareto/ASF selection (all maximized)
PARETO_OBJECTIVES = [
    "Precision@10",
    "Equality_Gain@10",
    "Pagerank_EG@10",
    "Closeness_EG@10",
]

# ASF weights (same as in pareto_hypervolume_analysis.py)
WEIGHTS = np.array([0.6, 0.3, 0.1, 0.1])

# Display name mapping
METHOD_RENAME = {
    "common_neighbors": "CN",
    "preferential_attachment": "PA",
    "resource_allocation": "RA",
    "adamic_adar": "AA",
    "Hybrid Analysis": "CAFE-P",
    "Hybrid Penalized C (Inproc)": "CAFE-In-B",
    "Hybrid Penalized C (Preproc)": "CAFE-Pre-B",
    "Hybrid Penalized D (Inproc)": "CAFE-In",
    "Hybrid Penalized D (Preproc)": "CAFE-Pre",
    "Coreness Inverse Inproc": "CI-In",
    "Coreness Inverse Preproc": "CI-Pre",
    "Coreness Inverse Postproc": "CI-Post",
    "Coreness Mobility Boost Inproc": "MB-In",
    "Coreness Mobility Boost Preproc": "MB-Pre",
    "Coreness Mobility Boost Postproc": "MB-Post",
    "lightgcn": "LightGCN",
    "ssnet": "SSNet",
    "ltlp": "LTP",
    "tailsteak": "TailSteak",
}

# ---------------------------------------------------------------------------
# Data Extraction
# ---------------------------------------------------------------------------

def safe_get_stat(stats_dict, metric_name):
    """Extract mean and std from a statistics dict. Returns (mean, std) or (NaN, NaN)."""
    if stats_dict is None:
        return np.nan, np.nan
    try:
        entry = stats_dict[metric_name]
        mean_val = entry.get("mean", np.nan)
        std_val = entry.get("std", 0.0)
        if mean_val is None:
            mean_val = np.nan
        if std_val is None:
            std_val = 0.0
        return float(mean_val), float(std_val)
    except (KeyError, TypeError, ValueError):
        return np.nan, np.nan


def extract_baselines(base_dir):
    """Extract baseline results (no hyperparameters) from the baselines/ folder."""
    rows = []
    baselines_dir = os.path.join(base_dir, "baselines")
    if not os.path.exists(baselines_dir):
        return rows

    for fpath in sorted(glob.glob(os.path.join(baselines_dir, "*.json"))):
        with open(fpath) as f:
            data = json.load(f)

        dataset = data["experiment_info"]["dataset"]

        for method_name, method_results in data["results"].items():
            row = {
                "Method_Group": method_name,
                "Method_Detail": method_name,
                "Dataset": dataset,
                "Has_Config": False,
            }
            for m in ALL_METRICS:
                mean_val, std_val = safe_get_stat(method_results, m)
                row[f"{m}_Mean"] = mean_val
                row[f"{m}_Std"] = std_val
            rows.append(row)

    return rows


def extract_lambda_results(fpath, data, method_group):
    """Extract results from files with lambda_results structure."""
    rows = []
    dataset = data["experiment_info"]["dataset"]

    if "lambda_results" not in data:
        return rows

    for key, stats_container in data["lambda_results"].items():
        # Get statistics dict
        stats = stats_container.get("statistics", stats_container)
        if not isinstance(stats, dict):
            continue

        # Parse configuration parameters from the key
        config_str = key
        row = {
            "Method_Group": method_group,
            "Method_Detail": f"{method_group} ({key})",
            "Dataset": dataset,
            "Config_Key": key,
            "Has_Config": True,
        }

        for m in ALL_METRICS:
            mean_val, std_val = safe_get_stat(stats, m)
            row[f"{m}_Mean"] = mean_val
            row[f"{m}_Std"] = std_val
        rows.append(row)

    return rows


def extract_single_method_results(fpath, data, method_group):
    """Extract results from files with results/{method_name} structure (cross_link_bias, lba)."""
    rows = []
    dataset = data["experiment_info"]["dataset"]

    for method_name, method_results in data["results"].items():
        row = {
            "Method_Group": method_group,
            "Method_Detail": method_name,
            "Dataset": dataset,
            "Has_Config": False,
        }
        for m in ALL_METRICS:
            mean_val, std_val = safe_get_stat(method_results, m)
            row[f"{m}_Mean"] = mean_val
            row[f"{m}_Std"] = std_val
        rows.append(row)

    return rows


def extract_all_results(base_dir):
    """Walk through all subdirectories and extract results."""
    all_rows = []

    # 1. Baselines
    all_rows.extend(extract_baselines(base_dir))

    # 2. Methods with lambda_results
    lambda_method_dirs = {
        "hybrid_analysis": "Hybrid Analysis",
        "coreness_inverse_inproc": "Coreness Inverse Inproc",
        "coreness_inverse_postproc": "Coreness Inverse Postproc",
        "coreness_inverse_preproc": "Coreness Inverse Preproc",
        "coreness_mobility_boost_inproc": "Coreness Mobility Boost Inproc",
        "coreness_mobility_boost_postproc": "Coreness Mobility Boost Postproc",
        "coreness_mobility_boost_preproc": "Coreness Mobility Boost Preproc",
        "hybrid_penalized_C_in": "Hybrid Penalized C (Inproc)",
        "hybrid_penalized_C_pre": "Hybrid Penalized C (Preproc)",
        "hybrid_penalized_D_in": "Hybrid Penalized D (Inproc)",
        "hybrid_penalized_D_pre": "Hybrid Penalized D (Preproc)",
    }

    for dirname, method_group in lambda_method_dirs.items():
        dirpath = os.path.join(base_dir, dirname)
        if not os.path.exists(dirpath):
            continue
        for fpath in sorted(glob.glob(os.path.join(dirpath, "*.json"))):
            fname = os.path.basename(fpath)
            if "best_config" in fname:
                continue  # handled separately in aggregate_best_config_runs

            with open(fpath) as f:
                data = json.load(f)

            # Detect baseline-only files (e.g. *_baseline_*.json) that contain
            # only beta_0.0_lambda_0.0.  These represent the HLGNN baseline for
            # datasets that don't have a dedicated baselines/ entry.
            lr_keys = list(data.get("lambda_results", {}).keys())
            is_baseline_only = (
                "_baseline_" in fname
                and lr_keys == ["beta_0.0_lambda_0.0"]
            )

            if is_baseline_only:
                # Extract as HLGNN baseline row
                rows = extract_lambda_results(fpath, data, "HLGNN")
                for r in rows:
                    r["Has_Config"] = False  # single value, no selection needed
                all_rows.extend(rows)
            else:
                rows = extract_lambda_results(fpath, data, method_group)
                all_rows.extend(rows)

    # 2b. Additional baselines in common_baselines/ and preferential_attachment/
    for extra_baselines_dir in ["common_baselines", "preferential_attachment"]:
        extra_dir = os.path.join(base_dir, extra_baselines_dir)
        if not os.path.exists(extra_dir):
            continue
        for fpath in sorted(glob.glob(os.path.join(extra_dir, "*.json"))):
            with open(fpath) as f:
                data = json.load(f)
            dataset = data["experiment_info"]["dataset"]
            for method_name, method_results in data["results"].items():
                row = {
                    "Method_Group": method_name,
                    "Method_Detail": method_name,
                    "Dataset": dataset,
                    "Has_Config": False,
                }
                for m in ALL_METRICS:
                    mean_val, std_val = safe_get_stat(method_results, m)
                    row[f"{m}_Mean"] = mean_val
                    row[f"{m}_Std"] = std_val
                all_rows.append(row)

    # 3. Single-method results (cross_link_bias, lba, fairadj, moral)
    single_method_dirs = {
        "cross_link_bias": "Cross-Link Bias",
        "lba": "LBA",
        "fairadj": "FairAdj",
        "moral": "MORAL",
        "lightgcn": "LightGCN",
        "ssnet": "SSNet",
        "ltlp": "LTLP",
        "tailsteak": "TailSteak",
    }

    for dirname, method_group in single_method_dirs.items():
        dirpath = os.path.join(base_dir, dirname)
        if not os.path.exists(dirpath):
            continue
        for fpath in sorted(glob.glob(os.path.join(dirpath, "*.json"))):
            with open(fpath) as f:
                data = json.load(f)
            rows = extract_single_method_results(fpath, data, method_group)
            all_rows.extend(rows)

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Pareto / ASF Selection
# ---------------------------------------------------------------------------

def select_best_config_pareto_asf(df_group):
    """
    Given a DataFrame of configurations for a single (method_group, dataset),
    use Pareto front + ASF to select the best compromise.

    Returns a single-row DataFrame with the best configuration.
    """
    obj_cols_mean = [f"{o}_Mean" for o in PARETO_OBJECTIVES]

    # Drop rows with NaN in any objective
    df_valid = df_group.dropna(subset=obj_cols_mean)
    if df_valid.empty:
        return df_group.iloc[[0]]  # fallback to first row

    if len(df_valid) == 1:
        return df_valid

    F_max = df_valid[obj_cols_mean].values

    # Normalize to [0, 1]
    obj_min = F_max.min(axis=0)
    obj_max = F_max.max(axis=0)
    obj_range = obj_max - obj_min
    obj_range[obj_range == 0] = 1.0
    F_norm = (F_max - obj_min) / obj_range

    # Negate for pymoo (minimization)
    F_neg = -F_norm

    # Non-dominated sorting
    nds = NonDominatedSorting()
    try:
        fronts = nds.do(F_neg)
    except Exception:
        # Fallback: pick the row with highest weighted sum
        weighted = F_norm @ WEIGHTS
        best_idx = weighted.argmax()
        return df_valid.iloc[[best_idx]]

    pareto_indices = fronts[0]

    if len(pareto_indices) == 1:
        return df_valid.iloc[pareto_indices]

    pareto_F_neg = F_neg[pareto_indices]

    # ASF on Pareto front
    pf_ideal = pareto_F_neg.min(axis=0)
    pf_nadir = pareto_F_neg.max(axis=0)
    pf_range = pf_nadir - pf_ideal
    pf_range[pf_range == 0] = 1.0
    pf_normalized = (pareto_F_neg - pf_ideal) / pf_range

    decomp = ASF()
    asf_values = decomp.do(pf_normalized, 1.0 / WEIGHTS)
    best_idx_in_pareto = asf_values.argmin()
    best_idx_in_df = pareto_indices[best_idx_in_pareto]

    return df_valid.iloc[[best_idx_in_df]]


def select_best_configs(df):
    """
    For each (Method_Group, Dataset), select the best configuration
    using Pareto/ASF if there are multiple configs, or just return
    the single row if there's only one.
    """
    results = []

    for (method_group, dataset), group_df in df.groupby(["Method_Group", "Dataset"]):
        if len(group_df) == 1 or not group_df["Has_Config"].any():
            # No configuration search needed; just use the single row
            results.append(group_df.iloc[[0]])
        else:
            best = select_best_config_pareto_asf(group_df)
            results.append(best)

    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# Best-Config Aggregation (brightkite / pokec_n extra runs)
# ---------------------------------------------------------------------------

# Mapping from directory name to method group (same as lambda_method_dirs)
_BEST_CONFIG_DIRS = {
    "hybrid_analysis": "Hybrid Analysis",
    "hybrid_penalized_C_in": "Hybrid Penalized C (Inproc)",
    "hybrid_penalized_C_pre": "Hybrid Penalized C (Preproc)",
    "hybrid_penalized_D_in": "Hybrid Penalized D (Inproc)",
    "hybrid_penalized_D_pre": "Hybrid Penalized D (Preproc)",
}

# HLGNN baseline config key in hybrid_analysis
_HLGNN_BASELINE_KEY = "beta_0.0_lambda_0.0"


def _collect_values_for_config(dirpath, config_key, target_dataset):
    """
    Scan all non-best_config JSON files in *dirpath* for *config_key*
    that match *target_dataset*, and return the values arrays for every
    metric found.

    Returns dict: metric_name -> list of individual values
    """
    metric_values = {}
    for fpath in sorted(glob.glob(os.path.join(dirpath, "*.json"))):
        fname = os.path.basename(fpath)
        if "best_config" in fname:
            continue
        with open(fpath) as f:
            data = json.load(f)
        # Only consider files for the same dataset
        if data.get("experiment_info", {}).get("dataset") != target_dataset:
            continue
        lr = data.get("lambda_results", {})
        if config_key not in lr:
            continue
        stats = lr[config_key].get("statistics", lr[config_key])
        if not isinstance(stats, dict):
            continue
        for metric_name, metric_data in stats.items():
            if isinstance(metric_data, dict) and "values" in metric_data:
                metric_values.setdefault(metric_name, []).extend(metric_data["values"])
    return metric_values


def aggregate_best_config_runs(base_dir):
    """
    Discover best_config files, combine their values with the matching
    config from the original results files, and return aggregated rows.

    Returns list of dicts, each suitable for replacing/adding a row in df_best.
    """
    aggregated = []

    for dirname, method_group in _BEST_CONFIG_DIRS.items():
        dirpath = os.path.join(base_dir, dirname)
        if not os.path.exists(dirpath):
            continue

        # Group best_config files by (dataset, config_key)
        # so that split runs (run1, run2) are combined.
        groups = {}  # (dataset, config_key) -> list of values per metric
        for fpath in sorted(glob.glob(os.path.join(dirpath, "*best_config*.json"))):
            with open(fpath) as f:
                data = json.load(f)
            dataset = data["experiment_info"]["dataset"]
            lr = data.get("lambda_results", {})
            for config_key, stats_container in lr.items():
                stats = stats_container.get("statistics", stats_container)
                if not isinstance(stats, dict):
                    continue
                key = (dataset, config_key)
                if key not in groups:
                    groups[key] = {}
                for metric_name, metric_data in stats.items():
                    if isinstance(metric_data, dict) and "values" in metric_data:
                        groups[key].setdefault(metric_name, []).extend(
                            metric_data["values"]
                        )

        # For each (dataset, config_key), also collect values from original files
        for (dataset, config_key), bc_values in groups.items():
            orig_values = _collect_values_for_config(dirpath, config_key, dataset)

            # Merge: original values + best_config values
            all_metrics = set(list(orig_values.keys()) + list(bc_values.keys()))
            combined = {}
            for m in all_metrics:
                vals = orig_values.get(m, []) + bc_values.get(m, [])
                combined[m] = vals

            # Determine effective method group
            # hybrid_analysis with baseline config -> HLGNN
            if dirname == "hybrid_analysis" and config_key == _HLGNN_BASELINE_KEY:
                effective_method = "HLGNN"
            else:
                effective_method = method_group

            n_runs = len(combined.get("Precision@10", []))

            row = {
                "Method_Group": effective_method,
                "Method_Detail": f"{effective_method} ({config_key})",
                "Dataset": dataset,
                "Config_Key": config_key,
                "Has_Config": True,
                "_aggregated": True,
                "_n_runs": n_runs,
            }
            for m in ALL_METRICS:
                vals = combined.get(m, [])
                if vals:
                    row[f"{m}_Mean"] = float(np.mean(vals))
                    row[f"{m}_Std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                else:
                    row[f"{m}_Mean"] = np.nan
                    row[f"{m}_Std"] = np.nan

            aggregated.append(row)
            print(f"    Aggregated {effective_method} / {dataset}: "
                  f"config={config_key}, {n_runs} runs")

    return aggregated


# Single-method dirs + datasets that need multi-file aggregation
_SINGLE_METHOD_AGGREGATE = {
    # (dirname, method_group): [datasets to aggregate]
    ("tailsteak", "TailSteak"): ["brightkite"],
}


def _reconstruct_values(metric_data, n_runs):
    """
    Reconstruct individual run values from summary statistics.
    For 1 run: [mean]. For 2 runs: [min, max].
    """
    if n_runs == 1:
        return [metric_data["mean"]]
    elif n_runs == 2:
        return [metric_data["min"], metric_data["max"]]
    else:
        # Cannot reconstruct individual values for >2 runs without values array
        return None


def aggregate_single_method_runs(base_dir):
    """
    For single-method results dirs (e.g. tailsteak) where multiple JSON files
    exist for the same dataset, reconstruct individual run values from
    min/max/mean and aggregate across all files.

    Returns list of dicts suitable for overriding rows in df_best.
    """
    aggregated = []

    for (dirname, method_group), datasets in _SINGLE_METHOD_AGGREGATE.items():
        dirpath = os.path.join(base_dir, dirname)
        if not os.path.exists(dirpath):
            continue

        for target_dataset in datasets:
            # Collect values from all files for this dataset
            combined = {}  # metric_name -> list of values
            total_runs = 0

            for fpath in sorted(glob.glob(os.path.join(dirpath, "*.json"))):
                with open(fpath) as f:
                    data = json.load(f)
                if data.get("experiment_info", {}).get("dataset") != target_dataset:
                    continue

                n_runs = data.get("experiment_info", {}).get("runs", 1)
                method_key = list(data.get("results", {}).keys())[0]
                method_results = data["results"][method_key]

                for metric_name, metric_data in method_results.items():
                    if not isinstance(metric_data, dict) or "mean" not in metric_data:
                        continue
                    vals = _reconstruct_values(metric_data, n_runs)
                    if vals is not None:
                        combined.setdefault(metric_name, []).extend(vals)

                total_runs += n_runs

            if total_runs <= 1:
                continue  # no aggregation needed

            row = {
                "Method_Group": method_group,
                "Method_Detail": method_group,
                "Dataset": target_dataset,
                "Has_Config": False,
                "_aggregated": True,
                "_n_runs": total_runs,
            }
            for m in ALL_METRICS:
                vals = combined.get(m, [])
                if vals:
                    row[f"{m}_Mean"] = float(np.mean(vals))
                    row[f"{m}_Std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                else:
                    row[f"{m}_Mean"] = np.nan
                    row[f"{m}_Std"] = np.nan

            aggregated.append(row)
            print(f"    Aggregated {method_group} / {target_dataset}: "
                  f"{total_runs} runs from multiple files")

    return aggregated


def apply_best_config_overrides(df_best, base_dir):
    """
    Override rows in df_best with aggregated best-config results.
    Returns the updated DataFrame.
    """
    aggregated_rows = aggregate_best_config_runs(base_dir)
    aggregated_rows.extend(aggregate_single_method_runs(base_dir))
    if not aggregated_rows:
        return df_best

    df = df_best.copy()

    for agg_row in aggregated_rows:
        method = agg_row["Method_Group"]
        dataset = agg_row["Dataset"]

        # Find matching row in df_best
        mask = (df["Method_Group"] == method) & (df["Dataset"] == dataset)

        if mask.any():
            # Update existing row
            idx = df.index[mask][0]
            for col, val in agg_row.items():
                if col in df.columns or col.endswith("_Mean") or col.endswith("_Std"):
                    df.at[idx, col] = val
        else:
            # Add new row
            new_row = pd.DataFrame([agg_row])
            df = pd.concat([df, new_row], ignore_index=True)

    return df


# ---------------------------------------------------------------------------
# Table Formatting
# ---------------------------------------------------------------------------

def format_mean_std(mean_val, std_val, precision=4):
    """Format as 'mean ± std' string."""
    if pd.isna(mean_val):
        return "-"
    if pd.isna(std_val) or std_val == 0:
        return f"{mean_val:.{precision}f}"
    return f"{mean_val:.{precision}f} ± {std_val:.{precision}f}"


def create_paper_table(df_best, metrics_to_show, metric_labels, output_path):
    """
    Create a paper-style CSV table.

    Format:
      Method | Metric | Dataset1 | Dataset2 | ...
      MethodA | Precision@10 | 0.1234 ± 0.0012 | ...
      MethodA | Equality_Gain@10 | 5.58 ± 0.42 | ...
      MethodB | Precision@10 | ...
    """
    datasets = sorted(df_best["Dataset"].unique())
    method_groups = sorted(df_best["Method_Group"].unique())

    rows = []
    for method in method_groups:
        for metric, label in zip(metrics_to_show, metric_labels):
            row = {"Method": method, "Metric": label}
            for dataset in datasets:
                match = df_best[
                    (df_best["Method_Group"] == method) &
                    (df_best["Dataset"] == dataset)
                ]
                if match.empty:
                    row[dataset] = "-"
                else:
                    mean_col = f"{metric}_Mean"
                    std_col = f"{metric}_Std"
                    mean_val = match.iloc[0][mean_col]
                    std_val = match.iloc[0][std_col]
                    row[dataset] = format_mean_std(mean_val, std_val)
            rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_path, index=False)
    return result_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("FINAL RESULTS TABLE GENERATOR")
    print("=" * 80)

    # Step 1: Extract all results
    print("\n[1/5] Extracting results from all JSON files...")
    df = extract_all_results(BASE_DIR)
    print(f"  Extracted {len(df)} total rows")
    print(f"  Method groups: {sorted(df['Method_Group'].unique())}")
    print(f"  Datasets: {sorted(df['Dataset'].unique())}")

    # Step 2: Select best configurations via Pareto/ASF
    print("\n[2/5] Selecting best configurations via Pareto/ASF...")
    df_best = select_best_configs(df)
    print(f"  Selected {len(df_best)} best configurations")

    # Step 2b: Override with aggregated best-config runs (brightkite/pokec_n)
    print("\n[2b/5] Aggregating best-config runs for brightkite/pokec_n...")
    df_best = apply_best_config_overrides(df_best, BASE_DIR)

    # Apply display name mapping
    df_best["Method_Group"] = df_best["Method_Group"].replace(METHOD_RENAME)

    # Print summary of selected configs
    print("\n  Selected configurations:")
    for _, row in df_best.iterrows():
        config = row.get("Config_Key", "N/A")
        prec = row.get("Precision@10_Mean", np.nan)
        eg = row.get("Equality_Gain@10_Mean", np.nan)
        print(f"    {row['Method_Group']:45s} | {row['Dataset']:15s} | "
              f"Config: {str(config):50s} | P@10={prec:.4f} | EG@10={eg:.4f}")

    # Step 3: Create Table 1 - Precision + Equality Gain metrics
    print("\n[3/5] Creating Table 1: Precision + Equality Gain metrics...")
    table1_metrics = [
        "Precision@10",
        "Equality_Gain@10",
        "Pagerank_EG@10",
        "Closeness_EG@10",
    ]
    table1_labels = [
        "Precision@10",
        "Degree EG@10",
        "PageRank EG@10",
        "Closeness EG@10",
    ]
    table1_path = os.path.join(OUTPUT_DIR, "table1_precision_equality_gain.csv")
    table1 = create_paper_table(df_best, table1_metrics, table1_labels, table1_path)
    print(f"  Saved to: {table1_path}")

    # Step 4: Create Table 2 - Precision + Demographic metrics
    print("\n[4/5] Creating Table 2: Precision + Demographic metrics...")
    table2_metrics = [
        "Precision@10",
        "NDKL@10",
        "Delta_SP@10",
        "Delta_EO@10",
        "IF",
        "GDIF",
        "Ranking_IF@10",
    ]
    table2_labels = [
        "Precision@10",
        "NDKL@10",
        "Delta SP@10",
        "Delta EO@10",
        "IF",
        "GDIF",
        "Ranking IF@10",
    ]
    table2_exclude_methods = [
        "AA", "CN", "deepwalk", "node2vec",
        "PA", "pure_gnn", "random", "RA",
    ]
    df_best_table2 = df_best[~df_best["Method_Group"].isin(table2_exclude_methods)]
    table2_path = os.path.join(OUTPUT_DIR, "table2_precision_demographic.csv")
    table2 = create_paper_table(df_best_table2, table2_metrics, table2_labels, table2_path)
    print(f"  Saved to: {table2_path}")

    # Step 5: Create Best Configs table
    print("\n[5/5] Creating Best Configs table...")
    config_rows = []
    for _, row in df_best.iterrows():
        config_rows.append({
            "Method": row["Method_Group"],
            "Dataset": row["Dataset"],
            "Config_Key": row.get("Config_Key", "N/A"),
            "Method_Detail": row.get("Method_Detail", row["Method_Group"]),
        })
    config_df = pd.DataFrame(config_rows)
    config_path = os.path.join(OUTPUT_DIR, "best_configs.csv")
    config_df.to_csv(config_path, index=False)
    print(f"  Saved to: {config_path}")

    # Print preview
    print("\n" + "=" * 80)
    print("TABLE 1 PREVIEW: Precision + Equality Gain Metrics")
    print("=" * 80)
    print(table1.to_string(index=False))

    print("\n" + "=" * 80)
    print("TABLE 2 PREVIEW: Precision + Demographic Metrics")
    print("=" * 80)
    print(table2.to_string(index=False))

    print("\n" + "=" * 80)
    print("BEST CONFIGS PREVIEW")
    print("=" * 80)
    print(config_df.to_string(index=False))

    print("\nDone!")


if __name__ == "__main__":
    main()
