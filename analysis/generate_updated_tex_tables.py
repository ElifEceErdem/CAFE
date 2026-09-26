"""Generate paper-ready LaTeX tables with changed content marked in red.

The numeric source is the audited long-form comparison CSVs produced by
``generate_validation_comparisons.py``.  The output preserves the manuscript's
table grouping while making every changed value or ranking style visible with
``\\textcolor{red}{...}``.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COMPARISON_DIR = os.path.join(REPO, "comparison_tables")

METRICS = ["Precision@10", "Degree EG@10", "PageRank EG@10", "Closeness EG@10"]
DECIMALS = {
    "Precision@10": 4,
    "Degree EG@10": 2,
    "PageRank EG@10": 2,
    "Closeness EG@10": 2,
}

DATASET_LABELS = {
    "brightkite": "Brightkite",
    "ego-facebook": "Ego-FB",
    "oklahoma": "Oklahoma",
    "pokec_n": "Pokec-n",
    "unc": "UNC",
}

METHOD_LABELS = {
    "random": "Random",
    "HLGNN": "HL-GNN",
    "HLGNN+RA": "HL-GNN+RA",
    "TailSteak": "TailSTEAK",
    "CAFE-In-B": "CAFE-In$^{-B}$",
    "CAFE-Pre-B": "CAFE-Pre$^{-B}$",
    "CAFE-P": "CAFE$^{-P}$",
}

TABLE4_GROUPS = [
    ("Heuristic", ["random", "PA", "RA", "AA", "CN"]),
    ("GNN Backbone", ["LightGCN", "SSNet", "HLGNN"]),
    ("Fairness (attr.-free)", ["Cross-Link Bias"]),
    ("Fairness (attr.-dep.)", ["FairAdj", "LBA", "MORAL"]),
    ("Deg./Bias Mit.\\ (attr.-free)", ["LTLP", "TailSteak"]),
    ("CAFE (ours)", ["CAFE-In", "CAFE-Pre"]),
]

TABLE7_GROUPS = [
    ("Backbone", ["HLGNN"]),
    ("Single comp.", ["CI-In", "CI-Pre", "MB-Post", "HLGNN+RA"]),
    ("Two-stage", ["CAFE-In-B", "CAFE-Pre-B", "CAFE-P"]),
    ("Full pipeline", ["CAFE-In", "CAFE-Pre"]),
]


def read_rows(filename):
    with open(os.path.join(COMPARISON_DIR, filename), newline="") as handle:
        return list(csv.DictReader(handle))


def index_rows(rows):
    return {
        (row["dataset"], row["method"], row["metric"]): row
        for row in rows
    }


def red(content):
    return f"\\textcolor{{red}}{{{content}}}"


def styled(content, style):
    if style == "bold":
        return f"\\textbf{{{content}}}"
    if style == "underline":
        return f"\\underline{{{content}}}"
    return content


def rank_styles(rows, value_field):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["metric"])].append(row)
    result = {}
    for (dataset, metric), metric_rows in grouped.items():
        ordered = sorted(
            metric_rows,
            key=lambda row: float(row[value_field]),
            reverse=True,
        )
        if ordered:
            result[(dataset, ordered[0]["method"], metric)] = "bold"
        if len(ordered) > 1:
            result[(dataset, ordered[1]["method"], metric)] = "underline"
    return result


def value_text(row, metric, deterministic=False):
    decimals = DECIMALS[metric]
    mean = float(row["reproduced_mean"])
    if deterministic:
        return f"{mean:.{decimals}f}"
    std = float(row["reproduced_std"])
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def metric_cell(row, metric, current_style, paper_style, deterministic=False,
                force_red=False):
    content = styled(value_text(row, metric, deterministic), current_style)
    changed = row["recommended_action"] == "UPDATE"
    style_changed = current_style != paper_style
    return red(content) if changed or style_changed or force_red else content


def table4_tex(rows):
    by_key = index_rows(rows)
    current_styles = rank_styles(rows, "reproduced_mean")
    paper_styles = rank_styles(rows, "paper_mean")
    deterministic = {"PA", "RA", "AA", "CN"}

    def block(datasets, first=False):
        lines = ["\\begin{table*}[t]"]
        if first:
            lines.extend([
                "\\caption{Precision@10 and Equality Gain@10 across methods and datasets (fixed split; mean $\\pm$ SD over 10 seeds). Deterministic heuristics are run once. \\textbf{Bold} and \\underline{underline} denote best and second-best per metric.",
                red("Degree EG uses target-indegree event semantics, retaining duplicate and reciprocal directed recommendation events; PageRank and Closeness EG use simple-undirected distinct-pair graphs."),
                "}",
                "\\label{tab:main-results}",
            ])
        else:
            lines.extend([
                "\\ContinuedFloat",
                "\\caption[]{Link prediction and structural fairness results across methods and datasets (continued).}",
            ])
        lines.append("\\centering\\small")

        letters = {"brightkite": "a", "ego-facebook": "b", "oklahoma": "c", "pokec_n": "d", "unc": "e"}
        for dataset in datasets:
            lines.extend([
                f"\\subcaption*{{({letters[dataset]}) {DATASET_LABELS[dataset]}}}",
                "\\begin{tabular}{llrrrr}",
                "\\toprule",
                "Model Type & Method & Prec@10 & Deg EG@10 & PR EG@10 & Cl.\\ EG@10 \\\\",
                "\\midrule",
            ])
            available = {row["method"] for row in rows if row["dataset"] == dataset}
            first_group = True
            for category, methods in TABLE4_GROUPS:
                present = [method for method in methods if method in available]
                if not present:
                    continue
                if not first_group:
                    lines.append("\\midrule")
                first_group = False
                for offset, method in enumerate(present):
                    cells = []
                    for metric in METRICS:
                        row = by_key[(dataset, method, metric)]
                        key = (dataset, method, metric)
                        cells.append(metric_cell(
                            row,
                            metric,
                            current_styles.get(key),
                            paper_styles.get(key),
                            deterministic=method in deterministic,
                        ))
                    category_cell = category if offset == 0 else ""
                    label = METHOD_LABELS.get(method, method)
                    lines.append(
                        f"{category_cell} & {label} & " + " & ".join(cells) + " \\\\"
                    )
            lines.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\vspace{6pt}",
                "",
            ])
        lines.append("\\end{table*}")
        return lines

    output = ["% Requires \\usepackage{xcolor} in the manuscript preamble.", ""]
    output.extend(block(["brightkite", "ego-facebook"], first=True))
    output.extend(["", ""])
    output.extend(block(["oklahoma", "pokec_n"]))
    output.extend(["", ""])
    output.extend(block(["unc"]))
    return "\n".join(output) + "\n"


def table7_tex(rows):
    by_key = index_rows(rows)
    current_styles = rank_styles(rows, "reproduced_mean")
    paper_styles = rank_styles(rows, "paper_mean")

    # The supplied manuscript source had values for the late Pokec reruns but
    # retained ranking markup from an older draft.  Record that source markup
    # so style-only corrections are also colored red.
    paper_styles[("pokec_n", "CI-In", "PageRank EG@10")] = "bold"
    paper_styles.pop(("pokec_n", "CI-Pre", "PageRank EG@10"), None)
    paper_styles[("pokec_n", "CAFE-In", "PageRank EG@10")] = "underline"
    paper_styles[("pokec_n", "CI-In", "Closeness EG@10")] = "bold"
    paper_styles[("pokec_n", "CAFE-P", "Closeness EG@10")] = "underline"
    paper_styles.pop(("pokec_n", "CI-Pre", "Closeness EG@10"), None)

    def dprec(dataset, method, current):
        if method == "HLGNN":
            return "--"
        field = "reproduced_mean" if current else "paper_mean"
        base = float(by_key[(dataset, "HLGNN", "Precision@10")][field])
        value = float(by_key[(dataset, method, "Precision@10")][field])
        return f"{100.0 * (value / base - 1.0):+.0f}\\%"

    def block(datasets, first=False):
        lines = ["\\begin{table*}[t]"]
        if first:
            lines.extend([
                "\\caption{",
                "CAFE ablations at $K=10$ (fixed split; mean $\\pm$ SD over 10 training seeds.",
                "$\\Delta$Prec is relative to HL-GNN;",
                "$\\star$ marks the full pipelines.",
                "Rows are grouped by the number.",
                "\\textbf{Bold} and \\underline{underline} denote the best and second-best result.",
                "Each configuration is tuned on its validation grid (Table~\\ref{tab:grids}).",
                red("Degree EG uses target-indegree event semantics, retaining duplicate and reciprocal directed recommendation events; PageRank and Closeness EG use simple-undirected distinct-pair graphs."),
                "}",
                "\\label{tab:ablation}",
            ])
        else:
            lines.extend([
                "\\ContinuedFloat",
                "\\caption{Ablation results across five datasets (\\emph{continued}).}",
            ])
        lines.extend([
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{3.2pt}",
            "\\begin{tabular}{lllrrrrr}",
            "\\toprule",
            "Dataset & Category & Config & Prec@10 & $\\Delta$Prec & Deg EG & PR EG & Cl.\\ EG \\\\",
            "\\midrule",
        ])

        for dataset_index, dataset in enumerate(datasets):
            if dataset_index:
                lines.append("\\midrule")
            first_dataset_row = True
            for group_index, (category, methods) in enumerate(TABLE7_GROUPS):
                if group_index:
                    lines.append("\\addlinespace[2pt]")
                for offset, method in enumerate(methods):
                    cells = []
                    for metric in METRICS:
                        row = by_key[(dataset, method, metric)]
                        key = (dataset, method, metric)
                        cells.append(metric_cell(
                            row,
                            metric,
                            current_styles.get(key),
                            paper_styles.get(key),
                        ))
                    old_delta = dprec(dataset, method, current=False)
                    new_delta = dprec(dataset, method, current=True)
                    delta_cell = red(new_delta) if new_delta != old_delta else new_delta
                    dataset_cell = DATASET_LABELS[dataset] if first_dataset_row else ""
                    first_dataset_row = False
                    category_cell = category if offset == 0 else ""
                    label = METHOD_LABELS.get(method, method)
                    if method in {"CAFE-In", "CAFE-Pre"}:
                        label += " $\\star$"
                    lines.append(
                        f"{dataset_cell} & {category_cell} & {label} & {cells[0]} & "
                        f"{delta_cell} & {cells[1]} & {cells[2]} & {cells[3]} \\\\"
                    )
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
        return lines

    output = ["% Requires \\usepackage{xcolor} in the manuscript preamble.", ""]
    output.extend(block(["brightkite", "ego-facebook", "oklahoma"], first=True))
    output.extend(["", ""])
    output.extend(block(["pokec_n", "unc"]))
    return "\n".join(output) + "\n"


TABLE11_ORIGINAL = {
    "Brightkite": {
        "RA": [(-4.42, 0.00), (2.16, 0.00), (0.27, 0.00), (1.75, 0.00), (0.06, 0.00), (0.07, 0.00)],
        "HL-GNN": [(-8.10, 1.06), (2.30, 0.01), (-0.95, 0.60), (1.87, 0.05), (0.33, 0.51), (0.18, 0.39)],
        "HL-GNN+RA": [(-5.68, 0.62), (2.16, 0.01), (-0.53, 0.31), (1.94, 0.04), (0.30, 0.52), (0.05, 0.38)],
        "CAFE-In": [(-3.16, 0.82), (2.15, 0.02), (0.55, 0.41), (1.98, 0.05), (0.30, 0.60), (0.18, 0.21)],
        "CAFE-Pre": [(-5.22, 0.82), (2.21, 0.01), (-0.01, 0.50), (2.04, 0.04), (0.22, 0.32), (-0.00, 0.24)],
    },
    "Ego-FB": {
        "RA": [(9.10, 0.00), (3.81, 0.00), (4.25, 0.00), (1.78, 0.00), (22.13, 0.00), (-6.89, 0.00)],
        "HL-GNN": [(13.05, 0.27), (4.11, 0.06), (10.52, 0.29), (3.59, 0.09), (35.28, 1.60), (-8.74, 0.82)],
        "HL-GNN+RA": [(10.78, 0.12), (4.09, 0.03), (7.94, 0.14), (3.12, 0.06), (34.63, 1.63), (-7.44, 0.13)],
        "CAFE-In": [(11.76, 0.15), (4.34, 0.03), (11.65, 0.24), (4.88, 0.05), (22.61, 2.65), (-7.10, 0.09)],
        "CAFE-Pre": [(12.85, 0.15), (4.58, 0.07), (13.62, 0.29), (5.70, 0.18), (26.21, 2.98), (-7.63, 0.25)],
    },
    "Oklahoma": {
        "RA": [(5.25, 0.00), (0.96, 0.00), (8.61, 0.00), (2.49, 0.00), (12.02, 0.00), (-0.30, 0.00)],
        "HL-GNN": [(3.62, 0.38), (0.40, 0.12), (7.26, 0.28), (1.58, 0.08), (11.75, 0.12), (-0.13, 0.06)],
        "HL-GNN+RA": [(4.41, 0.09), (0.92, 0.02), (8.40, 0.07), (2.49, 0.01), (11.64, 0.06), (-0.22, 0.02)],
        "CAFE-In": [(5.62, 0.13), (1.11, 0.06), (9.85, 0.13), (2.97, 0.02), (12.54, 0.34), (-0.27, 0.01)],
        "CAFE-Pre": [(7.63, 0.33), (1.31, 0.08), (13.59, 0.34), (4.15, 0.17), (15.18, 0.34), (-0.32, 0.03)],
    },
    "Pokec-n": {
        "RA": [(4.64, 0.00), (-5.76, 0.00), (3.11, 0.00), (1.02, 0.00), (-1.27, 0.00), (-1.02, 0.00)],
        "HL-GNN": [(-8.47, 0.18), (-4.53, 0.05), (-2.70, 0.14), (0.65, 0.02), (0.74, 1.00), (1.92, 4.04)],
        "HL-GNN+RA": [(-0.07, 0.06), (-5.13, 0.04), (1.52, 0.05), (0.79, 0.02), (0.79, 1.85), (-0.30, 1.19)],
        "CAFE-In": [(3.90, 0.06), (-6.20, 0.04), (3.86, 0.06), (1.15, 0.01), (0.43, 1.01), (3.86, 5.74)],
        "CAFE-Pre": [(-2.45, 0.15), (-7.18, 0.05), (2.30, 0.12), (1.20, 0.02), (1.82, 2.84), (0.07, 0.89)],
    },
    "UNC": {
        "RA": [(0.32, 0.00), (-0.31, 0.00), (6.66, 0.00), (1.40, 0.00), (13.27, 0.00), (-1.30, 0.00)],
        "HL-GNN": [(-6.13, 0.35), (-1.20, 0.10), (2.44, 0.25), (0.45, 0.04), (11.70, 0.32), (-1.22, 0.03)],
        "HL-GNN+RA": [(-2.66, 0.11), (-0.75, 0.03), (5.23, 0.09), (1.16, 0.01), (11.93, 0.13), (-1.29, 0.02)],
        "CAFE-In": [(0.24, 0.18), (-0.43, 0.03), (7.62, 0.12), (1.66, 0.02), (13.35, 0.10), (-1.30, 0.01)],
        "CAFE-Pre": [(2.21, 0.49), (-0.27, 0.10), (11.00, 0.41), (2.40, 0.09), (15.07, 0.46), (-1.48, 0.04)],
    },
}

TABLE11_METRICS = [
    "Degree full", "Degree PW", "PageRank full", "PageRank PW",
    "Closeness full", "Closeness TP",
]


def table11_tex(rows):
    by_key = index_rows(rows)
    method_key = {"HL-GNN": "HL-GNN", "HL-GNN+RA": "HL-GNN+RA"}
    lines = [
        "% Requires \\usepackage{xcolor} in the manuscript preamble.",
        "% Only Oklahoma has corrected acceptance-model rerun evidence.",
        "\\begin{table*}[t]",
        "\\caption{",
        "EG under acceptance models. Degree and PageRank EG use full and precision-weighted acceptance; Closeness EG uses full and true-positive acceptance because precision weighting is undefined for closeness. Mean $\\pm$ SD over 10 seeds.",
        red("Degree uses target-indegree recommendation events under both full and PW acceptance. PageRank uses a simple-undirected distinct-pair graph under full acceptance and a weighted reciprocal-probability-union graph under PW acceptance."),
        "}",
        "\\label{tab:acceptance_models}",
        "\\centering\\small",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{Deg EG@10} & \\multicolumn{2}{c}{PR EG@10} & \\multicolumn{2}{c}{Cl.\\ EG@10} \\\\",
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}",
        "Dataset & Method & Full & PW & Full & PW & Full & TP \\\\",
        "\\midrule",
    ]
    for dataset_index, (dataset, methods) in enumerate(TABLE11_ORIGINAL.items()):
        if dataset_index:
            lines.append("\\midrule")
        for method_index, (method, values) in enumerate(methods.items()):
            cells = []
            for metric_index, (mean, std) in enumerate(values):
                content = f"{mean:.2f} $\\pm$ {std:.2f}"
                if dataset == "Oklahoma":
                    lookup_method = method_key.get(method, method)
                    row = by_key[("oklahoma", lookup_method, TABLE11_METRICS[metric_index])]
                    content = (
                        f"{float(row['reproduced_mean']):.2f} $\\pm$ "
                        f"{float(row['reproduced_std']):.2f}"
                    )
                    if row["recommended_action"] == "UPDATE":
                        content = red(content)
                cells.append(content)
            dataset_cell = dataset if method_index == 0 else ""
            lines.append(
                f"{dataset_cell} & {method} & " + " & ".join(cells) + " \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    return "\n".join(lines) + "\n"


def write(path, content):
    with open(os.path.join(COMPARISON_DIR, path), "w") as handle:
        handle.write(content)


def main():
    table4 = read_rows("table4_paper_vs_reproduced.csv")
    table7 = read_rows("table7_paper_vs_reproduced.csv")
    table11 = read_rows("table11_oklahoma_paper_vs_reproduced.csv")
    write("table4_updated_red.tex", table4_tex(table4))
    write("table7_updated_red.tex", table7_tex(table7))
    write("table11_acceptance_updated_red.tex", table11_tex(table11))
    print(f"Wrote updated TeX tables to {COMPARISON_DIR}")


if __name__ == "__main__":
    main()
