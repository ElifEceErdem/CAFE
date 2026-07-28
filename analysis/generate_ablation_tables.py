"""
Generate the CAFE ablation tables with the fusion-only control row added:

  Table 3 (main paper) : Oklahoma ablation
  Table 6 (supp.)      : full ablation across the remaining datasets

Row structure (matching the paper, plus the new row):

  Backbone         HL-GNN
  Single-comp.     CI-In, CI-Pre, MB-Post
  Partial-pipe.    HL-GNN+RA (CAFE^{-P-B})   <-- NEW: fusion-only control
                   CAFE-In^{-B}, CAFE-Pre^{-B}, CAFE^{-P}
  Full pipeline    CAFE-In*, CAFE-Pre*

Sources everything through generate_final_tables.py (same extraction, same
Pareto+ASF selection, same best-config run aggregation), so numbers are
produced by the identical procedure as the published tables. The HL-GNN
backbone row is taken from the beta_0.0_lambda_0.0 config of the
hybrid_analysis/cafe_p sweeps when no dedicated baseline file exists.

Outputs (into --out_dir):
  table3_ablation_oklahoma.csv / .tex
  table6_ablation_all.csv / .tex

Bold = best, underline = second-best per (dataset, metric) column across
the ablation rows. All EG metrics and precision are higher-is-better.

Usage:
    python generate_ablation_tables.py            # uses ../results
    python generate_ablation_tables.py --base_dir /path/to/full/results
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_final_tables as gft  # noqa: E402

# (csv_name, latex_label, category) in paper order
ROW_SPEC = [
    ("HLGNN",      r"HL-GNN",                          "Backbone"),
    ("CI-In",      r"CI-In",                           "Single-comp."),
    ("CI-Pre",     r"CI-Pre",                          "Single-comp."),
    ("MB-Post",    r"MB-Post",                         "Single-comp."),
    ("HLGNN+RA",   r"HL-GNN+RA (CAFE$^{-P{,}-B}$)",    "Partial-pipe."),
    ("CAFE-In-B",  r"CAFE-In$^{-B}$",                  "Partial-pipe."),
    ("CAFE-Pre-B", r"CAFE-Pre$^{-B}$",                 "Partial-pipe."),
    ("CAFE-P",     r"CAFE$^{-P}$",                     "Partial-pipe."),
    ("CAFE-In",    r"CAFE-In $\star$",                 "Full pipeline"),
    ("CAFE-Pre",   r"CAFE-Pre $\star$",                "Full pipeline"),
]

METRICS = [
    ("Precision@10",     "Prec@10"),
    ("Equality_Gain@10", "Deg EG"),
    ("Pagerank_EG@10",   "PR EG"),
    ("Closeness_EG@10",  "Cl. EG"),
]

DATASET_LABELS = {
    "ego-facebook": "Ego-FB",
    "oklahoma": "Oklahoma",
    "unc": "UNC",
    "pokec_n": "Pokec-n",
    "brightkite": "Brightkite",
}
SUPP_DATASET_ORDER = ["ego-facebook", "unc", "brightkite", "pokec_n"]


def extract_hlgnn_rows(base_dir):
    """
    HL-GNN backbone rows from the beta_0.0_lambda_0.0 config of
    hybrid_analysis / cafe_p sweeps (fallback when no baseline file exists).
    """
    rows = {}
    for dirname in ["hybrid_analysis", "cafe_p"]:
        for fpath in gft._json_files(os.path.join(base_dir, "*", dirname, "*.json")):
            with open(fpath) as f:
                data = json.load(f)
            dataset = data["experiment_info"]["dataset"]
            lr = data.get("lambda_results", {})
            if gft._HLGNN_BASELINE_KEY not in lr:
                continue
            stats = lr[gft._HLGNN_BASELINE_KEY].get("statistics", {})
            row = {"Method_Group": "HLGNN", "Dataset": dataset,
                   "Config_Key": gft._HLGNN_BASELINE_KEY, "Has_Config": False}
            for m in gft.ALL_METRICS:
                mean_val, std_val = gft.safe_get_stat(stats, m)
                row[f"{m}_Mean"] = mean_val
                row[f"{m}_Std"] = std_val
            rows[dataset] = row  # last file wins; identical config either way
    return list(rows.values())


def build_best_df(base_dir):
    """Run the exact published pipeline: extract -> ASF select -> aggregate."""
    df = gft.extract_all_results(base_dir)
    if df.empty:
        raise SystemExit(f"No results found under {base_dir}")
    df_best = gft.select_best_configs(df)
    df_best = gft.apply_best_config_overrides(df_best, base_dir)
    df_best["Method_Group"] = df_best["Method_Group"].replace(gft.METHOD_RENAME)

    # Add HL-GNN backbone rows where the pipeline didn't produce one
    have_hlgnn = set(df_best[df_best["Method_Group"] == "HLGNN"]["Dataset"])
    extra = [r for r in extract_hlgnn_rows(base_dir) if r["Dataset"] not in have_hlgnn]
    if extra:
        df_best = pd.concat([df_best, pd.DataFrame(extra)], ignore_index=True)
    return df_best


def fmt(mean, std, ndec, with_std):
    if pd.isna(mean):
        return "--"
    if with_std and not pd.isna(std):
        return f"{mean:.{ndec}f} $\\pm$ {std:.{ndec}f}"
    return f"{mean:.{ndec}f}"


def build_dataset_block(df_best, dataset, with_std=True):
    """
    Returns (list of row dicts, n_configs_found) for one dataset, with
    formatted cells, Delta-Prec, and bold/underline marks resolved.
    """
    sub = df_best[df_best["Dataset"] == dataset]
    by_method = {r["Method_Group"]: r for _, r in sub.iterrows()}

    hlgnn_prec = by_method.get("HLGNN", {}).get("Precision@10_Mean", np.nan)

    # Rank raw means per metric for bold/underline (higher is better)
    marks = {}
    for metric, _ in METRICS:
        vals = [(name, by_method[name].get(f"{metric}_Mean", np.nan))
                for name, _, _ in ROW_SPEC if name in by_method]
        vals = [(n, v) for n, v in vals if not pd.isna(v)]
        vals.sort(key=lambda t: -t[1])
        if len(vals) >= 1:
            marks[(vals[0][0], metric)] = "bold"
        if len(vals) >= 2:
            marks[(vals[1][0], metric)] = "underline"

    rows = []
    for name, latex_label, category in ROW_SPEC:
        if name not in by_method:
            continue
        r = by_method[name]
        prec_mean = r.get("Precision@10_Mean", np.nan)

        if name == "HLGNN" or pd.isna(hlgnn_prec) or pd.isna(prec_mean):
            dprec = "--"
        else:
            pct = (prec_mean - hlgnn_prec) / hlgnn_prec * 100.0
            dprec = f"{pct:+.0f}\\%"

        cells = {}
        for metric, label in METRICS:
            ndec = 4 if metric == "Precision@10" else 2
            cell = fmt(r.get(f"{metric}_Mean", np.nan),
                       r.get(f"{metric}_Std", np.nan), ndec, with_std)
            mark = marks.get((name, metric))
            if mark == "bold" and cell != "--":
                cell = f"\\textbf{{{cell}}}"
            elif mark == "underline" and cell != "--":
                cell = f"\\underline{{{cell}}}"
            cells[label] = cell

        rows.append({
            "category": category,
            "method": name,
            "latex_label": latex_label,
            "config": r.get("Config_Key", ""),
            "dprec": dprec,
            **cells,
            # raw values for the CSV
            **{f"{m}_Mean": r.get(f"{m}_Mean", np.nan) for m, _ in METRICS},
            **{f"{m}_Std": r.get(f"{m}_Std", np.nan) for m, _ in METRICS},
        })
    return rows


def latex_table(blocks, caption, label, multi_dataset):
    """blocks: list of (dataset, rows)."""
    ncols = 7  # [dataset] category/config, method, prec, dprec, 3 EGs
    lines = [
        f"% Auto-generated by analysis/generate_ablation_tables.py",
        f"\\begin{{table*}}[t]" if multi_dataset else "\\begin{table}[t]",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\centering",
        "\\small",
    ]
    header_cols = ("Dataset & " if multi_dataset else "Category & ") + \
        "Config & Prec@10 & $\\Delta$Prec & Deg EG & PR EG & Cl.\\ EG \\\\"
    lines += [
        "\\begin{tabular}{ll" + "r" * 5 + "}",
        "\\toprule",
        header_cols,
        "\\midrule",
    ]

    for bi, (dataset, rows) in enumerate(blocks):
        if bi > 0:
            lines.append("\\midrule")
        first_col_written = False
        prev_cat = None
        for r in rows:
            if multi_dataset:
                first = DATASET_LABELS.get(dataset, dataset) if not first_col_written else ""
                first_col_written = True
            else:
                first = r["category"] if r["category"] != prev_cat else ""
                if r["category"] != prev_cat and prev_cat is not None:
                    lines.append("\\addlinespace[2pt]")
                prev_cat = r["category"]
            lines.append(
                f"{first} & {r['latex_label']} & {r['Prec@10']} & {r['dprec']} & "
                f"{r['Deg EG']} & {r['PR EG']} & {r['Cl. EG']} \\\\"
            )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}" if multi_dataset else "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Ablation tables 3 & 6 with fusion row")
    parser.add_argument("--base_dir", type=str, default=gft.BASE_DIR,
                        help="results tree (default: ../results)")
    parser.add_argument("--out_dir", type=str,
                        default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--with_std", type=lambda x: x.lower() == "true", default=True,
                        help="include +/- std in LaTeX cells (journal style)")
    args = parser.parse_args()

    print(f"Building best-config table from {args.base_dir} ...")
    df_best = build_best_df(args.base_dir)

    wanted = {name for name, _, _ in ROW_SPEC}
    present = df_best[df_best["Method_Group"].isin(wanted)]
    print("\nAblation rows found per dataset:")
    for ds, grp in present.groupby("Dataset"):
        missing = wanted - set(grp["Method_Group"])
        print(f"  {ds:14s}: {len(grp):2d} rows"
              + (f"  MISSING: {sorted(missing)}" if missing else ""))
        for _, r in grp.iterrows():
            n_runs = r.get("N_Runs", "")
            print(f"      {r['Method_Group']:12s} config={str(r.get('Config_Key','')):45s}"
                  f" P@10={r.get('Precision@10_Mean', float('nan')):.4f}")

    # ---- Table 3: Oklahoma ----
    if "oklahoma" in set(present["Dataset"]):
        rows = build_dataset_block(df_best, "oklahoma", args.with_std)
        tex = latex_table(
            [("oklahoma", rows)],
            caption="CAFE ablation results on the Oklahoma dataset. "
                    "$\\Delta$Prec shows relative change vs.\\ HL-GNN. "
                    "HL-GNN+RA is the fusion-only control (RA fusion, no "
                    "coreness penalty, no mobility boost). "
                    "Bold: best; underline: second-best per column.",
            label="tab:ablation_oklahoma",
            multi_dataset=False,
        )
        path = os.path.join(args.out_dir, "table3_ablation_oklahoma.tex")
        with open(path, "w") as f:
            f.write(tex)
        pd.DataFrame(rows).to_csv(
            os.path.join(args.out_dir, "table3_ablation_oklahoma.csv"), index=False)
        print(f"\nTable 3 written: {path}")
    else:
        print("\nWARNING: no oklahoma results found; Table 3 skipped.")

    # ---- Table 6: remaining datasets ----
    blocks = []
    for ds in SUPP_DATASET_ORDER:
        if ds in set(present["Dataset"]):
            blocks.append((ds, build_dataset_block(df_best, ds, args.with_std)))
    if blocks:
        tex = latex_table(
            blocks,
            caption="Full ablation results across datasets. $\\star$ -- primary "
                    "full-pipeline models. $\\Delta$Prec shows relative change "
                    "vs.\\ HL-GNN. HL-GNN+RA is the fusion-only control. "
                    "Oklahoma ablation appears in the main text.",
            label="tab:ablation_all",
            multi_dataset=True,
        )
        path = os.path.join(args.out_dir, "table6_ablation_all.tex")
        with open(path, "w") as f:
            f.write(tex)
        all_rows = []
        for ds, rows in blocks:
            for r in rows:
                all_rows.append({"dataset": ds, **r})
        pd.DataFrame(all_rows).to_csv(
            os.path.join(args.out_dir, "table6_ablation_all.csv"), index=False)
        print(f"Table 6 written: {path}")
        missing_ds = [d for d in SUPP_DATASET_ORDER if d not in {b[0] for b in blocks}]
        if missing_ds:
            print(f"WARNING: Table 6 is missing datasets (no results in tree): {missing_ds}")
    else:
        print("WARNING: no supplementary datasets found; Table 6 skipped.")


if __name__ == "__main__":
    main()
