"""Figures 1, 3 and 4 -- the plot counterparts of the table suite.

Figure 1  coreness-stratified diagnosis on UNC (degree x coreness heatmap and
          per-coreness precision / EG bars, five methods).  Section 4.
Figure 3  Precision@10 vs Degree/PageRank/Closeness EG, five datasets x three
          EG dimensions, with per-panel Pareto frontiers.  Section 7.1.
Figure 4  source-coreness profiles on Oklahoma and Brightkite.  Section 7.2.

Figure 2 is a hand-drawn pipeline diagram and has no code path.

Usage mirrors the table modules::

    python -m experiments.tables.figures --stage plan
    python -m experiments.tables.figures --stage build
    python -m experiments.tables.figures --stage run --dataset unc --execute
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.tables._framework import (  # noqa: E402
    DATA_PATH, PLOTS_DIR, REPO, TABLES_DIR, Source, TableSpec,
    common_train_flags, main,
)

DIAG_DIR = os.path.join(PLOTS_DIR, "precision_by_coreness_aggregated")
STRAT_DIR = os.path.join(PLOTS_DIR, "stratified_cafe")


def runner(ds: str) -> list[str]:
    """Figure 1's diagnostic sweep (CN/RA/AA/LightGCN/HL-GNN), UNC in the paper.

    run_precision_by_coreness.py builds its own parser rather than using
    ``cafe_methods.add_common_args``, so it takes no ``--predictor``; the flag
    list here is deliberately narrower than ``common_train_flags``.
    """
    cmd = ["python", "-u", os.path.join(REPO, "analysis", "run_precision_by_coreness.py"),
           "--data_name", ds, "--data_path", DATA_PATH,
           "--encoder", "HLGNN",
           "--epochs", "500", "--eval_steps", "10", "--patience", "50",
           "--batch_size", "16384", "--lr", "0.001",
           "--runs", "3", "--K", "10", "--mode", "baseline",
           "--save_dir", DIAG_DIR]
    if ds == "brightkite":
        cmd += ["--split_file", os.path.join(DATA_PATH, "brightkite", "temporal_split.pt")]
    return cmd


SPEC = TableSpec(
    key="figs",
    number="1/3/4",
    title="Diagnosis, Pareto trade-off and source-coreness profiles",
    section="4, 7.1, 7.2",
    sources=[
        Source("Fig. 1 diagnostic sweep",
               "analysis/plots/precision_by_coreness_aggregated/precision_by_coreness_{ds}.csv",
               kind="csv", datasets=["unc"]),
        Source("Fig. 3 point set (Table 4 CSV)",
               "results_repro/tables/table1_precision_equality_gain.csv",
               kind="csv", datasets=["(one file)"]),
        Source("Fig. 4 stratified profiles",
               "analysis/plots/stratified_cafe/eg_by_coreness_cafe_{ds}.csv",
               kind="csv", datasets=["oklahoma", "brightkite"]),
    ],
    build=[
        ["python", "-u", "analysis/plot_precision_eg_tradeoff_table_match.py",
         "--csv", os.path.join(TABLES_DIR, "table1_precision_equality_gain.csv"),
         "--out_dir", TABLES_DIR],
    ],
    runner=runner,
    artifacts=[
        "results_repro/tables/full_comparison_matrix_table_match.pdf",
        "results_repro/tables/pokec_n_summary_table_match.pdf",
    ],
    notes="""
Figure 3 is a two-stage pipeline and stage 1's output is gitignored, so it
goes stale silently: generate_final_tables.py must write
table1_precision_equality_gain.csv BEFORE the plot script reads it.  Always
pass --csv/--out_dir pointing at results_repro/tables; the copy in analysis/
belongs to whichever tree was aggregated last.

The published figure carries CI-Post, MB-In and MB-Pre in its legend.  Those
variants appear in no table and are defined nowhere in the text -- drop them
from the plotted method set, or define them, before the next compile.
""",
)

if __name__ == "__main__":
    raise SystemExit(main(SPEC))
