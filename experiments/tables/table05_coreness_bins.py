"""Table 5 -- realized coreness bins for the stratified profiles.

Section 7.2 (RQ3).  Per dataset: number of realized bins, coreness range,
evaluated users, min/max bin size, and the Spearman correlation between bin
size and coreness rank.  Coreness is integer-valued and heavily tied, so the
requested ten quantiles collapse to 6-10 bins.

Backed by the source-stratified campaign (``analysis/run_stratified_cafe.py``),
which is the same run that produces Figure 4, Table 6(a) and Table 18 -- run
it once per dataset and all four artifacts follow.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.tables._framework import (  # noqa: E402
    DATA_PATH, PLOTS_DIR, PUBLISHED, REPO, TABLES_DIR, Source, TableSpec,
    large_dataset_flags, main,
)

STRAT_DIR = os.path.join(PLOTS_DIR, "stratified_cafe")


def runner(ds: str) -> list[str]:
    cfg = PUBLISHED[ds]
    return (["python", "-u", os.path.join(REPO, "analysis", "run_stratified_cafe.py"),
             "--data_name", ds, "--data_path", DATA_PATH,
             "--cafe_in_lambda", f"{cfg['in_lambda']:g}",
             "--cafe_in_beta", f"{cfg['in_beta']:g}",
             "--cafe_in_lam", f"{cfg['in_lam']:g}",
             "--cafe_pre_alpha", f"{cfg['pre_alpha']:g}",
             "--cafe_pre_beta", f"{cfg['pre_beta']:g}",
             "--cafe_pre_lam", f"{cfg['pre_lam']:g}",
             "--runs", "3",
             "--include_fusion", "True",
             "--fusion_beta", f"{cfg['in_beta']:g}",
             "--save_dir", STRAT_DIR]
            + large_dataset_flags(ds))


SPEC = TableSpec(
    key="t05",
    number="5",
    title="Realized coreness bins for the stratified profiles",
    section="7.2 (RQ3)",
    sources=[
        Source("stratified per-node precision", "analysis/plots/stratified_cafe/stratified_cafe_{ds}.csv",
               kind="csv"),
        Source("stratified per-bin EG", "analysis/plots/stratified_cafe/eg_by_coreness_cafe_{ds}.csv",
               kind="csv"),
    ],
    build=[
        ["python", "-u", "analysis/bin_perturbation_size.py",
         "--stratified_dir", STRAT_DIR,
         "--out_dir", os.path.join(PLOTS_DIR, "bin_perturbation")],
        ["python", "-u", "analysis/generate_bin_perturbation_tables.py",
         "--in_dir", os.path.join(PLOTS_DIR, "bin_perturbation"),
         "--out_dir", TABLES_DIR],
    ],
    runner=runner,
    artifacts=["results_repro/tables/table_coreness_bins.tex"],
    notes="""
The published stratified artifacts date from 2026-07-16..19 and therefore
predate the Equality-Gain rewrite; their per-bin Degree EG is on the
superseded metric.  The gradient's DIRECTION is robust (the corrected todo3
campaign reproduces it), but the levels are not -- re-run before quoting any
per-bin number.

Run with --save_recommendations True if you also want Table 18's exact
deduplicated per-bin edge counts rather than the K|B_g| upper bound.
""",
)

if __name__ == "__main__":
    raise SystemExit(main(SPEC))
