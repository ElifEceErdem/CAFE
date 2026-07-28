"""Table 9 -- structural mechanism behind the demographic byproduct.

Section 7.4 (RQ5).  Per attribute-carrying dataset: Cliff's delta for S0 vs S1
coreness on the training graph, which group is more core, which receives
higher baseline exposure, the predicted DeltaSP direction, and the measured
paired DeltaSP differences against the backbone and the fusion control.

Two independent inputs meet here: the structural half comes from
``coreness_by_group.py`` (CPU only, no training) and the DeltaSP half from the
split-resampled significance runs of Table 16.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.tables._framework import (  # noqa: E402
    ATTR_DATASETS, DATA_PATH, PLOTS_DIR, REPO, RESULTS, TABLES_DIR,
    Source, TableSpec, main,
)

GROUP_DIR = os.path.join(PLOTS_DIR, "coreness_by_group")
STRAT_DIR = os.path.join(PLOTS_DIR, "stratified_cafe")


def runner(ds: str) -> list[str]:
    """CPU-only structural analysis; --datasets takes the full list at once."""
    return ["python", "-u", os.path.join(REPO, "analysis", "coreness_by_group.py"),
            "--datasets", ds,
            "--data_path", DATA_PATH,
            "--stratified_csv_dir", STRAT_DIR,
            "--save_dir", GROUP_DIR]


SPEC = TableSpec(
    key="t09",
    number="9",
    title="Structural mechanisms driving the demographic byproduct",
    section="7.4 (RQ5)",
    sources=[
        Source("group-coreness association", "analysis/plots/coreness_by_group/coreness_group_association.csv",
               kind="csv", datasets=["(all datasets, one file)"]),
        Source("per-node exposure", "analysis/plots/coreness_by_group/exposure_per_node_{ds}.csv",
               kind="csv", datasets=ATTR_DATASETS),
        Source("per-seed metrics (DeltaSP)", "results_repro/{ds}/significance/per_seed_metrics_{ds}.csv",
               kind="csv", datasets=ATTR_DATASETS),
    ],
    build=[
        ["python", "-u", "analysis/generate_demographic_appendix_tables.py",
         "--results_dir", RESULTS, "--tables_dir", TABLES_DIR,
         "--analysis_dir", GROUP_DIR, "--out_dir", TABLES_DIR],
    ],
    runner=runner,
    artifacts=[
        "results_repro/tables/demographic_protocol_table.tex",
        "results_repro/tables/target_exposure_audit.csv",
        "results_repro/tables/demographic_significance_summary.tex",
    ],
    notes="""
The "more exposed" column and the predicted direction come from baseline
exposure under HL-GNN, not from any fitted quantity: Eq. 10 is monotone in
coreness percentile and Section 7.2 measures the resulting shift without ever
reading a demographic label.  Keep it that way -- the claim is that the rule
follows from the design, so nothing here may be tuned on the outcome.

Cliff's delta is computed on the loaded graph after isolated-node removal,
which is why the group sizes here differ slightly from the raw attribute files.
""",
)

if __name__ == "__main__":
    raise SystemExit(main(SPEC))
