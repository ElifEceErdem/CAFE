"""Table 8 -- demographic fairness on the three datasets with sensitive attributes.

Section 7.4 (RQ5).  Precision@10, DeltaSP@10 and DeltaEO@10 for the
attribute-dependent baselines (FairAdj, LBA, MORAL) against the backbone, the
fusion control and every CAFE configuration, on Oklahoma, Pokec-n and UNC.

No experiment of its own: it re-reads the Table 4 result tree and adds the
demographic columns, so run the Table 4 jobs first.  Including the fusion row
is what makes "demographic fairness as a byproduct" ablatable -- if HL-GNN+RA
alone already moves DeltaSP, the effect is RA fusion rather than coreness.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.tables._framework import (  # noqa: E402
    ATTR_DATASETS, RESULTS, TABLES_DIR, Source, TableSpec, main,
)

SPEC = TableSpec(
    key="t08",
    number="8",
    title="Demographic fairness on the datasets with sensitive attributes",
    section="7.4 (RQ5)",
    sources=[
        Source("HL-GNN backbone", "results_repro/{ds}/cafe_p/*_baseline_*.json",
               datasets=ATTR_DATASETS),
        Source("HL-GNN+RA", "results_repro/{ds}/fusion_only/*.json",
               datasets=ATTR_DATASETS),
        Source("FairAdj", "results_repro/{ds}/fairadj/*.json", datasets=ATTR_DATASETS),
        Source("LBA", "results_repro/{ds}/lba/*.json", datasets=ATTR_DATASETS),
        Source("MORAL", "results_repro/{ds}/moral/*.json", datasets=ATTR_DATASETS),
        Source("CAFE^-P", "results_repro/{ds}/cafe_p/*_cafe_p_*.json",
               datasets=ATTR_DATASETS),
        Source("CAFE-In^-B", "results_repro/{ds}/cafe_in_b/*.json", datasets=ATTR_DATASETS),
        Source("CAFE-Pre^-B", "results_repro/{ds}/cafe_pre_b/*.json", datasets=ATTR_DATASETS),
        Source("CAFE-In", "results_repro/{ds}/cafe_in/*.json", datasets=ATTR_DATASETS),
        Source("CAFE-Pre", "results_repro/{ds}/cafe_pre/*.json", datasets=ATTR_DATASETS),
    ],
    build=[
        ["python", "-u", "analysis/generate_table5.py",
         "--base_dir", RESULTS, "--out_dir", TABLES_DIR],
        ["python", "-u", "analysis/demographic_significance.py",
         "--tables_dir", TABLES_DIR],
    ],
    artifacts=[
        "results_repro/tables/table5_demographic_fairness.tex",
        "results_repro/tables/table5_demographic_fairness.csv",
        "results_repro/tables/table5_significance.csv",
    ],
    notes="""
DeltaSP/DeltaEO are computed from the same recommendation lists as
Precision@10 and are NOT affected by the Equality-Gain semantics fix; the
Prec/EG columns of this table are.  Verify still enforces the corrected
marker because the table prints EG columns alongside.

demographic_significance.py deliberately tests only Precision@10 and
Degree/PageRank EG: the paper makes no superiority claim on DeltaSP/DeltaEO.
""",
)

if __name__ == "__main__":
    raise SystemExit(main(SPEC))
