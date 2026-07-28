"""Table 6 -- localization of CAFE-In beyond the fusion control.

Section 7.2 (RQ3).  Two panels:

(a) source side -- the CAFE-In-minus-fusion Precision@10 difference in the
    lowest coreness bin, and the lowest/highest-bin precision ratio under
    fusion and then under CAFE-In.  Read from the stratified campaign
    (Table 5's sources).
(b) target side -- relative change in mean per-node incoming exposure within
    each target-coreness tercile.  Read from the exposure campaign
    (Table 11's runner).

Panel (b) is the audit the draft marks as unavailable on Ego-FB and
Brightkite.  The cause is a gate in ``run_exposure_acceptance.py`` that
skipped the exposure block whenever a dataset had no sensitive attribute --
even though this panel pools the groups and needs no attribute at all.  That
gate has been removed, so a rerun on those two datasets completes the table.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.tables._framework import (  # noqa: E402
    DATASETS, PLOTS_DIR, TABLES_DIR, Source, TableSpec, main,
)

STRAT_DIR = os.path.join(PLOTS_DIR, "stratified_cafe")
EXPOSURE_DIR = os.path.join(PLOTS_DIR, "coreness_by_group")

SPEC = TableSpec(
    key="t06",
    number="6",
    title="Localization of CAFE-In beyond the fusion control",
    section="7.2 (RQ3)",
    sources=[
        Source("stratified per-node precision (panel a)",
               "analysis/plots/stratified_cafe/stratified_cafe_{ds}.csv", kind="csv"),
        Source("per-node target exposure (panel b)",
               "analysis/plots/coreness_by_group/exposure_per_node_{ds}.csv",
               kind="csv", datasets=DATASETS,
               note="written by run_exposure_acceptance.py; see Table 11's runner"),
    ],
    build=[
        ["python", "-u", "analysis/generate_localization_table.py",
         "--stratified_dir", STRAT_DIR,
         "--exposure_dir", EXPOSURE_DIR,
         "--out_dir", TABLES_DIR],
    ],
    artifacts=["results_repro/tables/table_localization.tex",
               "results_repro/tables/table_localization.csv"],
    notes="""
No runner of its own -- panel (a) comes from Table 5's stratified campaign and
panel (b) from Table 11's exposure campaign.  Run those two, then build here.

Repetition counts differ between the panels and across datasets in the
published data (source side n=3, Pokec-n n=1; target side n=10 on
Oklahoma/UNC, n=3 on Pokec-n).  The generator reports what it finds; put those
numbers in the caption rather than leaving "across reruns" unqualified.

An all-five version of panel (b) already exists from the corrected-EG
campaign at
results_egfix/20260724_degree_eg_sum_in_v1/todo3/finalized/
    target_exposure_cafe_in_vs_fusion_summary.csv
but it was produced at RE-SELECTED configurations, so its magnitudes are not
comparable with the published table.  Use it as a direction check, not a
drop-in.
""",
)

if __name__ == "__main__":
    raise SystemExit(main(SPEC))
