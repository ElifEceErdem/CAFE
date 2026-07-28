"""Table 17 -- DeltaSP@10 and DeltaEO@10 under both protocols.

Appendix B.  Absolute values under the fixed-split headline protocol (one
split, n=10 training repetitions, matching Table 8) and under the
split-resampled paired protocol (n=10 seeds, each drawing its own split), plus
the paired contrasts against the backbone and the fusion control, Holm
corrected within each (measure x reference) family.

The point of printing both protocols side by side is that the magnitudes
differ legitimately; this table is what stops a reader deriving a third number
from Table 8 and reporting an inconsistency.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.tables._framework import (  # noqa: E402
    ATTR_DATASETS, PLOTS_DIR, RESULTS, TABLES_DIR, Source, TableSpec, main,
)

GROUP_DIR = os.path.join(PLOTS_DIR, "coreness_by_group")

SPEC = TableSpec(
    key="t17",
    number="17",
    title="DeltaSP@10 and DeltaEO@10 under both protocols",
    section="Appendix B",
    sources=[
        Source("fixed-split HL-GNN baseline", "results_repro/{ds}/cafe_p/*_baseline_*.json",
               datasets=ATTR_DATASETS),
        Source("fixed-split CAFE rows (Table 8 CSV)",
               "results_repro/tables/table5_demographic_fairness.csv",
               kind="csv", datasets=["(one file)"]),
        Source("split-resampled per-seed metrics",
               "results_repro/{ds}/significance/per_seed_metrics_{ds}.csv",
               kind="csv", datasets=ATTR_DATASETS),
    ],
    build=[
        ["python", "-u", "analysis/generate_demographic_appendix_tables.py",
         "--results_dir", RESULTS, "--tables_dir", TABLES_DIR,
         "--analysis_dir", GROUP_DIR, "--out_dir", TABLES_DIR],
    ],
    artifacts=["results_repro/tables/demographic_protocol_table.tex"],
    notes="""
No runner: built from Table 8's fixed-split results and Table 16's
split-resampled per-seed CSVs.  Run those two first.

DeltaSP/DeltaEO are unaffected by the Equality-Gain semantics fix -- they are
computed from recommendation lists and group labels only.  This is the one
demographic table that does not need the corrected-EG rerun.
""",
)

if __name__ == "__main__":
    raise SystemExit(main(SPEC))
