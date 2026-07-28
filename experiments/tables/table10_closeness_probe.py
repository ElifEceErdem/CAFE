"""Table 10 -- stratum-level effect of the mobility boost (closeness probe).

Section 7.5, "The Closeness Exception".  Change in mean predicted closeness of
each target-coreness quintile when the boost is added to the fused scorer,
paired within rerun.  Panel (a) fixes lambda_mob = 0.5 on Ego-FB, Oklahoma and
UNC; panel (b) sweeps the dose on Ego-FB, the only probe with a dose curve.

The probe deliberately shares one trained backbone across arms so the ONLY
difference between +RA and +RA+boost is the multiplier -- that is what licenses
attributing the Ego-FB Closeness regression to the boost rather than to fusion.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.tables._framework import (  # noqa: E402
    DATA_PATH, PLOTS_DIR, PUBLISHED, REPO, TABLES_DIR, Source, TableSpec,
    common_train_flags, main,
)

PROBE_DATASETS = ["ego-facebook", "oklahoma", "unc"]
PROBE_DIR = os.path.join(PLOTS_DIR, "closeness_mechanism_paired")


def runner(ds: str) -> list[str]:
    cfg = PUBLISHED[ds]
    cmd = (["python", "-u", os.path.join(REPO, "analysis", "closeness_mechanism.py")]
           + common_train_flags(ds)
           + ["--runs", "10", "--K", "10",
              "--beta", f"{cfg['in_beta']:g}",
              "--lam", f"{cfg['in_lam']:g}",
              "--vary_split", "False",
              "--save_dir", PROBE_DIR])
    if ds == "ego-facebook":
        # panel (b): the dose sweep is Ego-FB only
        cmd += ["--lam_grid", "0.1,0.3,0.5,1.0"]
    return cmd


SPEC = TableSpec(
    key="t10",
    number="10",
    title="Stratum-level effect of the mobility boost (closeness probe)",
    section="7.5",
    sources=[
        Source("per-quintile closeness", "analysis/plots/closeness_mechanism_paired/closeness_by_bin_{ds}.csv",
               kind="csv", datasets=PROBE_DATASETS),
        Source("path statistics", "analysis/plots/closeness_mechanism_paired/path_stats_{ds}.csv",
               kind="csv", datasets=PROBE_DATASETS),
        Source("distance spectrum", "analysis/plots/closeness_mechanism_paired/distance_spectrum_{ds}.csv",
               kind="csv", datasets=PROBE_DATASETS),
    ],
    build=[
        ["python", "-u", "analysis/generate_closeness_stratum_table.py",
         "--plots_dir", PROBE_DIR, "--out_dir", TABLES_DIR],
    ],
    runner=runner,
    artifacts=[
        "results_repro/tables/closeness_strata_table.tex",
        "results_repro/tables/closeness_strata_dose_table.tex",
    ],
    notes="""
Closeness EG is NOT affected by the 2026-07-23 Equality-Gain rewrite (the
refactor changed degree counting and PageRank binarization), so this table's
published numbers remain valid.  It is listed here for completeness of the
table map and because a rerun is needed if you want the per-arm unique-edge
budget, which the earlier probe did not record.

Three probe networks only.  Pokec-n and Brightkite were never probed; the
paper scopes the claim accordingly, so do not silently generalize it.
""",
)

if __name__ == "__main__":
    raise SystemExit(main(SPEC))
