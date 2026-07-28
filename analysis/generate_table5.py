"""
Generate Table 5 (supplementary): demographic and structural fairness for
fairness-aware baselines and CAFE variants on the datasets with sensitive
attributes (Oklahoma, Pokec-n, UNC), optionally including the fusion-only
control (HL-GNN+RA).

Including the fusion row here is what turns "demographic fairness as a
byproduct" into an ablatable claim: if HL-GNN+RA alone already achieves the
demographic gains, they come from RA fusion; if the full CAFE variants
improve DeltaSP/DeltaEO beyond it, the coreness debiasing contributes.

Same sourcing as the published tables (generate_final_tables.py pipeline).

Columns: Prec@10 | Deg EG@10 | PR EG@10 | Cl. EG@10 | DeltaSP@10 | DeltaEO@10
Bold = best, underline = second-best per column within each dataset block.
Direction: Prec/EG higher-is-better; DeltaSP/DeltaEO lower-is-better.

Usage:
    python generate_table5.py                       # ../results, fusion row on
    python generate_table5.py --include_fusion false
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_final_tables as gft  # noqa: E402
from generate_ablation_tables import build_best_df  # noqa: E402

DATASETS = ["oklahoma", "pokec_n", "unc"]
DATASET_LABELS = {"oklahoma": "Oklahoma", "pokec_n": "Pokec-n", "unc": "UNC"}

# (csv_name, latex_label, group)
# FairAdj, LBA, and MORAL all consume the sensitive attribute during
# training (see baselines/fairness_aware_baseline_models.py). WeakTie rows
# are excluded: their metric path evaluates all graph nodes instead of the
# test-active users, so they are not comparable to the other rows.
BASELINE_ROWS = [
    ("FairAdj", "FairAdj", "Fairness (attr.-dep.)"),
    ("LBA", "LBA", "Fairness (attr.-dep.)"),
    ("MORAL", "MORAL", "Fairness (attr.-dep.)"),
]
FUSION_ROW = ("HLGNN+RA", r"HL-GNN+RA (CAFE$^{-P{,}-B}$)",
              "CAFE (attr.-free, ours)")
CAFE_ROWS = [
    ("CAFE-P", r"CAFE$^{-P}$", "CAFE (attr.-free, ours)"),
    ("CAFE-In-B", r"CAFE-In$^{-B}$", "CAFE (attr.-free, ours)"),
    ("CAFE-Pre-B", r"CAFE-Pre$^{-B}$", "CAFE (attr.-free, ours)"),
    ("CAFE-In", "CAFE-In", "CAFE (attr.-free, ours)"),
    ("CAFE-Pre", "CAFE-Pre", "CAFE (attr.-free, ours)"),
]

# (metric_key, label, higher_is_better, decimals, display_scale)
METRICS = [
    ("Precision@10", "Prec@10", True, 4, 1.0),
    ("Equality_Gain@10", "Deg EG@10", True, 2, 1.0),
    ("Pagerank_EG@10", "PR EG@10", True, 2, 1.0),
    ("Closeness_EG@10", "Cl.\\ EG@10", True, 2, 1.0),
    ("Delta_SP@10", "$\\Delta$SP@10 ($\\times 10^{-3}$)", False, 3, 1e3),
    ("Delta_EO@10", "$\\Delta$EO@10", False, 4, 1.0),
]


def build_block(df_best, dataset, row_spec, with_std=True):
    sub = df_best[df_best["Dataset"] == dataset]
    by_method = {r["Method_Group"]: r for _, r in sub.iterrows()}

    marks = {}
    for metric, _, higher, _, _ in METRICS:
        vals = [(name, by_method[name].get(f"{metric}_Mean", np.nan))
                for name, _, _ in row_spec if name in by_method]
        vals = [(n, v) for n, v in vals if not pd.isna(v)]
        vals.sort(key=lambda t: -t[1] if higher else t[1])
        if len(vals) >= 1:
            marks[(vals[0][0], metric)] = "bold"
        if len(vals) >= 2:
            marks[(vals[1][0], metric)] = "underline"

    rows = []
    for name, latex_label, group in row_spec:
        if name not in by_method:
            print(f"  NOTE: {dataset}: no results for {name}; row omitted.")
            continue
        r = by_method[name]
        cells = {}
        for metric, label, _, ndec, scale in METRICS:
            mean = r.get(f"{metric}_Mean", np.nan)
            std = r.get(f"{metric}_Std", np.nan)
            if pd.isna(mean):
                cells[label] = "--"
                continue
            cell = f"{mean * scale:.{ndec}f}"
            if with_std and not pd.isna(std):
                cell += f" $\\pm$ {std * scale:.{ndec}f}"
            mark = marks.get((name, metric))
            if mark == "bold":
                cell = f"\\textbf{{{cell}}}"
            elif mark == "underline":
                cell = f"\\underline{{{cell}}}"
            cells[label] = cell
        rows.append({"group": group, "method": name,
                     "latex_label": latex_label,
                     "config": r.get("Config_Key", ""), **cells,
                     **{f"{m}_Mean": r.get(f"{m}_Mean", np.nan) for m, _, _, _, _ in METRICS},
                     **{f"{m}_Std": r.get(f"{m}_Std", np.nan) for m, _, _, _, _ in METRICS}})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Table 5 with optional fusion row")
    parser.add_argument("--base_dir", type=str, default=gft.BASE_DIR)
    parser.add_argument("--out_dir", type=str,
                        default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--include_fusion", type=lambda x: x.lower() == "true",
                        default=True)
    parser.add_argument("--with_std", type=lambda x: x.lower() == "true",
                        default=True)
    args = parser.parse_args()

    df_best = build_best_df(args.base_dir)

    cafe_block = ([FUSION_ROW] if args.include_fusion else []) + CAFE_ROWS
    row_spec = BASELINE_ROWS + cafe_block

    header = " & ".join(label for _, label, _, _, _ in METRICS)
    lines = [
        "% Auto-generated by analysis/generate_table5.py",
        "\\begin{table*}[t]",
        "\\caption{Demographic and structural fairness on the three "
        "datasets with sensitive attributes (fixed split, $K{=}10$; mean "
        "$\\pm$ std over 10 repetitions). FairAdj, LBA, and MORAL observe "
        "the sensitive attribute during training; CAFE variants never do. "
        "Higher is better for Precision and EG; lower is better for "
        "$\\Delta$SP and $\\Delta$EO. Bold: best per column; "
        "\\underline{underline}: second-best. HL-GNN+RA is the fusion-only "
        "control (no coreness penalty, no boost).}",
        "\\label{tab:demographic_fairness}",
        "\\centering\\small",
    ]

    csv_rows = []
    for ds in DATASETS:
        if ds not in set(df_best["Dataset"]):
            print(f"WARNING: no results for {ds} in {args.base_dir}; block skipped.")
            continue
        print(f"\n[{ds}]")
        rows = build_block(df_best, ds, row_spec, args.with_std)
        if not rows:
            continue
        lines += [
            f"\\subcaption*{{({chr(97 + DATASETS.index(ds))}) {DATASET_LABELS[ds]}}}",
            "\\begin{tabular}{ll" + "r" * len(METRICS) + "}",
            "\\toprule",
            f"Model Type & Method & {header} \\\\",
            "\\midrule",
        ]
        prev_group = None
        for r in rows:
            grp = r["group"] if r["group"] != prev_group else ""
            if r["group"] != prev_group and prev_group is not None:
                lines.append("\\midrule")
            prev_group = r["group"]
            cells = " & ".join(r[label] for _, label, _, _, _ in METRICS)
            lines.append(f"{grp} & {r['latex_label']} & {cells} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\vspace{6pt}", ""]
        for r in rows:
            csv_rows.append({"dataset": ds, **r})

    lines.append("\\end{table*}")

    tex_path = os.path.join(args.out_dir, "table5_demographic_fairness.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    csv_path = os.path.join(args.out_dir, "table5_demographic_fairness.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"\nTable 5 written: {tex_path}")
    print(f"CSV written: {csv_path}")


if __name__ == "__main__":
    main()
