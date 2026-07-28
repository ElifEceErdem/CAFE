"""Table 4 -- link prediction and structural fairness across methods and datasets.

Section 7.1 (RQ2).  Five per-dataset panels, Precision@10 plus Degree,
PageRank and Closeness EG@10, for every heuristic, GNN backbone, fairness and
tail-node baseline alongside the two CAFE variants.  Also the point set behind
Figure 3's Pareto panels.

This is the widest table in the paper and therefore the expensive one to
re-run: it needs every method, not just the CAFE core.  Under the corrected-EG
plan, all of these rows must be regenerated -- a table mixing pre- and
post-2026-07-23 Degree/PageRank EG is not comparable within a column.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.tables._framework import (  # noqa: E402
    ATTR_DATASETS, DATASETS, PUBLISHED, RESULTS, REPO, TABLES_DIR, Source, TableSpec,
    common_train_flags, eg_flags, large_dataset_flags, main,
)


def runner(ds: str) -> list[str]:
    """CAFE-In at the published configuration; the other rows have their own jobs."""
    cfg = PUBLISHED[ds]
    out = os.path.join(RESULTS, ds, "cafe_in")
    return (["python", "-u", "run_cafe.py"]
            + common_train_flags(ds)
            + ["--runs", "10",
               "--penalization_mode", "inproc", "--use_mobility_boost", "True",
               "--lambda_fair", f"{cfg['in_lambda']:g}",
               "--beta_ra", f"{cfg['in_beta']:g}",
               "--lambda_mobility", f"{cfg['in_lam']:g}",
               "--coreness_alpha", "1.0"]
            + eg_flags()
            + ["--results_file", os.path.join(out, f"results_cafe_in_{ds}.json")]
            + large_dataset_flags(ds))


SPEC = TableSpec(
    key="t04",
    number="4",
    title="Link prediction and structural fairness across methods and datasets",
    section="7.1 (RQ2)",
    sources=[
        Source("heuristics (random/PA/CN/AA/RA)", "results_repro/{ds}/baselines/*.json",
               min_runs=1, note="deterministic on the fixed split; one run each"),
        Source("LightGCN", "results_repro/{ds}/lightgcn/*.json"),
        Source("SSNet", "results_repro/{ds}/ssnet/*.json"),
        Source("HL-GNN backbone", "results_repro/{ds}/cafe_p/*_baseline_*.json"),
        Source("Cross-Link Bias", "results_repro/{ds}/cross_link_bias/*.json"),
        Source("LTLP", "results_repro/{ds}/ltlp/*.json"),
        Source("TailSTEAK", "results_repro/{ds}/tailsteak/*.json"),
        Source("FairAdj", "results_repro/{ds}/fairadj/*.json", datasets=ATTR_DATASETS),
        Source("LBA", "results_repro/{ds}/lba/*.json", datasets=ATTR_DATASETS),
        Source("MORAL", "results_repro/{ds}/moral/*.json", datasets=ATTR_DATASETS),
        Source("CAFE-In", "results_repro/{ds}/cafe_in/*.json"),
        Source("CAFE-Pre", "results_repro/{ds}/cafe_pre/*.json"),
    ],
    build=[
        ["python", "-u", "analysis/generate_final_tables.py",
         "--base_dir", RESULTS, "--out_dir", TABLES_DIR],
        ["python", "-u", "analysis/generate_table4.py",
         "--base_dir", RESULTS, "--out_dir", TABLES_DIR],
    ],
    runner=runner,
    artifacts=[
        "results_repro/tables/table1_precision_equality_gain.csv",
        "results_repro/tables/table4_precision_equality_gain.tex",
        "results_repro/tables/best_configs.csv",
    ],
    notes="""
Jobs: experiments/jobs/t04_*.job (one per method family).  The CAFE rows are
shared with Tables 7 and 8, so run t04_cafe.job once and all three benefit.

generate_final_tables.py re-runs the Pareto + ASF selection over whatever is
in --base_dir and writes best_configs.csv; diff that against
analysis/best_configs.csv.  Under the corrected-EG plan the selection must be
PINNED to the published configurations rather than re-selected -- re-selection
is a different experiment (see the todo3 campaign) and changes what CAFE-In
denotes.
""",
)

if __name__ == "__main__":
    raise SystemExit(main(SPEC))
