"""Table 16 -- paired per-seed contrasts for CAFE-In and CAFE-Pre.

Appendix B.  Three panels: against the HL-GNN backbone (a), the fusion-only
control (b) and RA (c).  Mean paired difference with a 95% t-based CI, Holm
corrected within each comparison's datasets x measures family, plus Wilcoxon
agreement.

Protocol note that matters more than it looks: this table uses the
SPLIT-RESAMPLED protocol -- each seed draws its own train/val/test partition
(Brightkite excepted, whose temporal cut is fixed) -- whereas Tables 4, 7 and
8 are fixed-split with training-seed repetitions.  Means therefore differ
between the two legitimately, sometimes substantially.  Never quote a paired
delta from here next to a fixed-split magnitude from Table 4 without saying
which protocol it came from.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.tables._framework import (  # noqa: E402
    DATA_PATH, PUBLISHED, RESULTS, TABLES_DIR, Source, TableSpec,
    common_train_flags, large_dataset_flags, main,
)


def runner(ds: str) -> list[str]:
    cfg = PUBLISHED[ds]
    out = os.path.join(RESULTS, ds, "significance")
    return (["python", "-u", "run_significance.py"]
            + common_train_flags(ds)
            + ["--num_seeds", "10", "--seed_start", "0",
               "--cafe_in_lambda", f"{cfg['in_lambda']:g}",
               "--cafe_in_beta", f"{cfg['in_beta']:g}",
               "--cafe_in_lam", f"{cfg['in_lam']:g}",
               "--cafe_pre_alpha", f"{cfg['pre_alpha']:g}",
               "--cafe_pre_beta", f"{cfg['pre_beta']:g}",
               "--cafe_pre_lam", f"{cfg['pre_lam']:g}",
               "--fusion_beta", f"{cfg['fusion_beta']:g}",
               # split resampling is the point of this experiment; Brightkite
               # keeps its fixed temporal cut.
               "--vary_split", "False" if ds == "brightkite" else "True",
               "--out_csv", os.path.join(out, f"per_seed_metrics_{ds}.csv")]
            + large_dataset_flags(ds))


SPEC = TableSpec(
    key="t16",
    number="16",
    title="Paired per-seed contrasts vs backbone / fusion / RA",
    section="Appendix B",
    sources=[
        Source("per-seed metrics", "results_repro/{ds}/significance/per_seed_metrics_{ds}.csv",
               kind="csv",
               note="long format: dataset, seed, split_seed, method, metric, value"),
    ],
    build=[
        ["python", "-u", "analysis/significance_tests.py",
         "--input_glob", os.path.join(RESULTS, "*", "significance", "per_seed_metrics_*.csv"),
         "--out_prefix", os.path.join(TABLES_DIR, "significance")],
    ],
    runner=runner,
    artifacts=[
        "results_repro/tables/significance_tests.csv",
        "results_repro/tables/significance_table.tex",
        "results_repro/tables/significance_table_compact.tex",
    ],
    notes="""
run_significance.py stamps eg_graph_semantics into its own metadata, but the
per-seed artifact is a CSV, so verify checks presence and seed count only.
Confirm the semantics from the job log or the run's metadata sidecar before
trusting the Degree/PageRank columns.

Family definition (Table 15): F1 = vs backbone, F2 = vs fusion, F3 = vs RA,
40 contrasts each, Holm within family.  Do not widen or narrow the families
after seeing the results.
""",
)

if __name__ == "__main__":
    raise SystemExit(main(SPEC))
