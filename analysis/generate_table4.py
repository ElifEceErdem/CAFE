"""
Generate Table 4 (supplementary): link prediction and structural fairness
results across methods and datasets -- the LaTeX rendering of
table1_precision_equality_gain.csv.

Layout follows the supplementary PDF: one subtable per dataset, rows grouped by
model type, columns Prec@10 | Deg EG@10 | PR EG@10 | Cl. EG@10, bold for best
and underline for second-best within each dataset block.  Because a LaTeX float
cannot break across pages, the five subtables are emitted as three continued
``table*`` floats (two, two, and one panel).

Sourced through build_best_df() -- the same
extract -> Pareto/ASF select -> best-config override pipeline that writes
table1_precision_equality_gain.csv -- so the numbers here are that CSV's, just
typeset (and rounded to the paper's per-metric decimals rather than the CSV's
uniform 4).

TWO DELIBERATE DEVIATIONS from generate_table5.py, both from the standing
instruction in the tab:main-results TODO (sections/07-results.tex):

  1. "missing cells must remain visibly marked rather than being filled from
     the conference results" -- a method absent for one dataset keeps its row
     and renders '--' there, instead of having the row silently dropped the way
     generate_table5.py drops it. Currently this shows up as LTLP and TailSTEAK
     on Brightkite, which have not finished. A method absent from EVERY dataset
     is dropped, with a warning, rather than emitting an all-'--' row.

  2. The supplementary caption's significance sentence ("CAFE shows
     statistically significant improvement over HL-GNN on all four objectives")
     is NOT reproduced. Under the reproduced paired protocol Degree EG
     *decreases* significantly on Ego-FB for both CAFE variants (§7.1), so that
     sentence no longer matches the numbers in the table. Re-add it only if a
     rerun supports it.

Rows excluded on purpose: WeakTie (evaluation-population bug, §7.1 -- excluded
until its lambda=0 identity control passes), the attribute-dependent fairness
baselines FairAdj/LBA/MORAL (they belong to Table 5), and every CAFE ablation
variant (Table 6 / Table 3).

Bold marks the best value per metric across ALL rows in the block, Random and
PA included, matching the published table -- Random wins several EG columns at
Precision@10 ~ 0.001, which is a property of the measure and is meant to be
visible rather than hidden.

Usage:
    python generate_table4.py --base_dir ../results_repro --out_dir <dir>
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_final_tables as gft  # noqa: E402
from generate_ablation_tables import build_best_df  # noqa: E402

DATASETS = ["brightkite", "ego-facebook", "oklahoma", "pokec_n", "unc"]
DATASET_LABELS = {
    "brightkite": "Brightkite",
    "ego-facebook": "Ego-FB",
    "oklahoma": "Oklahoma",
    "pokec_n": "Pokec-n",
    "unc": "UNC",
}

# (Method_Group as it appears in df_best, LaTeX label, model-type group)
ROW_SPEC = [
    ("random",          "Random",          "Heuristic"),
    ("PA",              "PA",              "Heuristic"),
    ("RA",              "RA",              "Heuristic"),
    ("AA",              "AA",              "Heuristic"),
    ("CN",              "CN",              "Heuristic"),
    ("LightGCN",        "LightGCN",        "GNN Backbone"),
    ("SSNet",           "SSNet",           "GNN Backbone"),
    ("HLGNN",           "HL-GNN",          "GNN Backbone"),
    ("Cross-Link Bias", "Cross-Link Bias", "Fairness (attr.-free)"),
    ("LTLP",            "LTLP",            "Deg./Bias Mit.\\ (attr.-free)"),
    ("TailSteak",       "TailSTEAK",       "Deg./Bias Mit.\\ (attr.-free)"),
    ("CAFE-In",         "CAFE-In",         "CAFE (ours)"),
    ("CAFE-Pre",        "CAFE-Pre",        "CAFE (ours)"),
]

# (metric_key, column label, decimals); all four are higher-is-better
METRICS = [
    ("Precision@10",     "Prec@10",       4),
    ("Equality_Gain@10", "Deg EG@10",     2),
    ("Pagerank_EG@10",   "PR EG@10",      2),
    ("Closeness_EG@10",  "Cl.\\ EG@10",   2),
]

MISSING = "--"
PANELS_PER_FLOAT = 2


def format_cell(mean, std, ndec, with_std):
    """
    'mean $\\pm$ std', or a bare mean for the deterministic heuristics.

    The std is suppressed when it is NaN or exactly 0, which is the rule
    generate_final_tables.format_mean_std uses, so a cell here matches the
    corresponding cell of table1_precision_equality_gain.csv.
    """
    if pd.isna(mean):
        return MISSING
    cell = f"{mean:.{ndec}f}"
    if with_std and not pd.isna(std) and std != 0:
        cell += f" $\\pm$ {std:.{ndec}f}"
    return cell


def build_block(df_best, dataset, row_spec, with_std=True):
    """Formatted rows for one dataset subtable, with bold/underline resolved."""
    sub = df_best[df_best["Dataset"] == dataset]
    by_method = {r["Method_Group"]: r for _, r in sub.iterrows()}

    # Rank per metric over the rows that actually have a value.
    marks = {}
    for metric, _, _ in METRICS:
        vals = [(name, by_method[name].get(f"{metric}_Mean", np.nan))
                for name, _, _ in row_spec if name in by_method]
        vals = [(n, v) for n, v in vals if not pd.isna(v)]
        vals.sort(key=lambda t: -t[1])
        if len(vals) >= 1:
            marks[(vals[0][0], metric)] = "bold"
        if len(vals) >= 2:
            marks[(vals[1][0], metric)] = "underline"

    rows = []
    for name, latex_label, group in row_spec:
        r = by_method.get(name)
        cells = {}
        for metric, label, ndec in METRICS:
            if r is None:
                cells[label] = MISSING
                continue
            cell = format_cell(r.get(f"{metric}_Mean", np.nan),
                               r.get(f"{metric}_Std", np.nan), ndec, with_std)
            if cell == MISSING:
                continue_mark = None
            else:
                continue_mark = marks.get((name, metric))
            if continue_mark == "bold":
                cell = f"\\textbf{{{cell}}}"
            elif continue_mark == "underline":
                cell = f"\\underline{{{cell}}}"
            cells[label] = cell

        if r is None:
            print(f"  NOTE: {dataset}: no results for {name}; "
                  f"row kept with '{MISSING}' cells.")

        rows.append({
            "group": group,
            "method": name,
            "latex_label": latex_label,
            "config": r.get("Config_Key", "") if r is not None else "",
            **cells,
            **{f"{m}_Mean": (r.get(f"{m}_Mean", np.nan) if r is not None else np.nan)
               for m, _, _ in METRICS},
            **{f"{m}_Std": (r.get(f"{m}_Std", np.nan) if r is not None else np.nan)
               for m, _, _ in METRICS},
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Table 4 (LaTeX) from the "
                                                 "table1 pipeline")
    parser.add_argument("--base_dir", type=str, default=gft.BASE_DIR)
    parser.add_argument("--out_dir", type=str,
                        default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--with_std", type=lambda x: x.lower() == "true",
                        default=True)
    args = parser.parse_args()

    df_best = build_best_df(args.base_dir)
    present = set(df_best["Method_Group"])

    # Drop methods with no results anywhere; keep partially-missing ones so the
    # gap stays visible (see module docstring, deviation 1).
    row_spec = [t for t in ROW_SPEC if t[0] in present]
    for name, _, _ in ROW_SPEC:
        if name not in present:
            print(f"WARNING: {name} has no results in {args.base_dir} for any "
                  f"dataset; row dropped entirely.")

    datasets = [d for d in DATASETS if d in set(df_best["Dataset"])]
    for d in DATASETS:
        if d not in datasets:
            print(f"WARNING: no results for {d}; subtable skipped.")

    header = " & ".join(label for _, label, _ in METRICS)
    lines = [
        "% Auto-generated by analysis/generate_table4.py -- do not edit by hand.",
        "% LaTeX rendering of table1_precision_equality_gain.csv.",
        "\\begin{table*}[t]",
        "\\caption{Link prediction and structural fairness results across "
        "methods and datasets (K=10). Results averaged over 10 seeds ($\\pm$ "
        "standard deviation); the deterministic heuristics are run once. "
        "Bold: best per metric; \\underline{underline}: second-best. "
        f"`{MISSING}' marks a configuration that has not finished and is left "
        "deliberately empty.}",
        "\\label{tab:main-results}",
        "\\centering\\small",
    ]

    csv_rows = []
    for i, ds in enumerate(datasets):
        print(f"\n[{ds}]")
        rows = build_block(df_best, ds, row_spec, args.with_std)
        if not rows:
            continue
        if i > 0 and i % PANELS_PER_FLOAT == 0:
            lines += [
                "\\end{table*}",
                "",
                "\\begin{table*}[t]",
                "\\ContinuedFloat",
                "\\caption[]{Link prediction and structural fairness results "
                "across methods and datasets (continued).}",
                "\\centering\\small",
            ]
        lines += [
            f"\\subcaption*{{({chr(97 + i)}) {DATASET_LABELS[ds]}}}",
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
            cells = " & ".join(r[label] for _, label, _ in METRICS)
            lines.append(f"{grp} & {r['latex_label']} & {cells} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\vspace{6pt}", ""]
        for r in rows:
            csv_rows.append({"dataset": ds, **r})

    lines.append("\\end{table*}")

    tex_path = os.path.join(args.out_dir, "table4_precision_equality_gain.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    csv_path = os.path.join(args.out_dir, "table4_precision_equality_gain.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"\nTable 4 written: {tex_path}")
    print(f"CSV written: {csv_path}")


if __name__ == "__main__":
    main()
