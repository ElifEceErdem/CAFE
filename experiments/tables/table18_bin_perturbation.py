"""Table 18 -- bin size does not drive the stratified Degree-EG gradient.

Appendix C.  Panel (a): per source-coreness bin, evaluated users |B_g| and
CAFE-In's stratified Degree EG@10.  Panel (b): Spearman correlation of EG with
coreness rank and with |B_g|, the partial correlation controlling for size, and
the two most size-matched bins per dataset.

This is the answer to the "your gradient is a perturbation-size gradient"
objection, so it must be built from the SAME stratified artifacts that back
Figure 4 and Table 5 -- never from a differently-seeded rerun, or the
size-matched pairs stop matching the plotted bins.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.tables._framework import (  # noqa: E402
    PLOTS_DIR, TABLES_DIR, Source, TableSpec, main,
)

STRAT_DIR = os.path.join(PLOTS_DIR, "stratified_cafe")
BIN_DIR = os.path.join(PLOTS_DIR, "bin_perturbation")

SPEC = TableSpec(
    key="t18",
    number="18",
    title="Bin size vs the stratified Degree-EG gradient",
    section="Appendix C",
    sources=[
        Source("stratified per-node precision", "analysis/plots/stratified_cafe/stratified_cafe_{ds}.csv",
               kind="csv"),
        Source("stratified per-bin EG", "analysis/plots/stratified_cafe/eg_by_coreness_cafe_{ds}.csv",
               kind="csv"),
    ],
    build=[
        ["python", "-u", "analysis/bin_perturbation_size.py",
         "--stratified_dir", STRAT_DIR, "--out_dir", BIN_DIR],
        ["python", "-u", "analysis/generate_bin_perturbation_tables.py",
         "--in_dir", BIN_DIR, "--out_dir", TABLES_DIR],
    ],
    artifacts=[
        "results_repro/tables/table_bin_perturbation.tex",
        "results_repro/tables/table_coreness_bins.tex",
    ],
    notes="""
Shares its runner with Table 5 -- run experiments/jobs/t05_stratified.job once.

bin_perturbation_size.py reports K|B_g| as an exact UPPER bound on the
distinct edges a bin adds; pass --npz_dir (with the campaign run under
--save_recommendations True) to recover the exact deduplicated count.

The published inputs predate the Equality-Gain rewrite.  The corrected
campaign reproduces the gradient's shape and sign pattern, so the argument
survives, but the tabulated levels do not -- rerun before quoting them.
""",
)

if __name__ == "__main__":
    raise SystemExit(main(SPEC))
