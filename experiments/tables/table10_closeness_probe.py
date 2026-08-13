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
    # The two probes do NOT share a resampling protocol, and the published
    # Table 10 depends on that: Ego-FB was run split-resampled (run r draws
    # split seed r), Oklahoma and UNC on the frozen split with ten training
    # seeds.  Running Ego-FB with --vary_split False silently regenerates a
    # different row (Closeness EG 34.63 -> 22.25 instead of 32.66 -> 17.32),
    # so this flag is per-dataset, not a constant.
    cmd = (["python", "-u", os.path.join(REPO, "analysis", "closeness_mechanism.py")]
           + common_train_flags(ds)
           + ["--runs", "10", "--K", "10",
              "--beta", f"{cfg['in_beta']:g}",
              "--lam", f"{cfg['in_lam']:g}",
              "--vary_split", "True" if ds == "ego-facebook" else "False",
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
        # Both generators take the plots ROOT and append the probe subdirectory
        # themselves; they read Ego-FB from closeness_mechanism_paired/ and
        # Oklahoma/UNC from closeness_mechanism/.
        ["python", "-u", "analysis/generate_closeness_stratum_table.py",
         "--plots_dir", PLOTS_DIR, "--out_dir", TABLES_DIR],
        # Section 7.5's per-arm distance spectrum and target degree; footnote 4
        # cites this script, so it ships with the table.
        ["python", "-u", "analysis/probe_diagnostics.py",
         "--plots_dir", PLOTS_DIR, "--out_dir", TABLES_DIR],
    ],
    runner=runner,
    artifacts=[
        "results_repro/tables/closeness_strata_table.tex",
        "results_repro/tables/closeness_strata_dose_table.tex",
        "results_repro/tables/probe_diagnostics.csv",
    ],
    notes="""
Closeness EG is NOT affected by the 2026-07-23 Equality-Gain rewrite (the
refactor changed degree counting and PageRank binarization), so this table's
published numbers remain valid.  It is listed here for completeness of the
table map and because a rerun is needed if you want the per-arm unique-edge
budget, which the earlier probe did not record.

The Ego-FB probe artifacts under closeness_mechanism_paired/ were overwritten
on 2026-07-29 by a --vary_split False rerun, which does not reproduce the
published row; they were restored from commit dcea835^ on 2026-08-07 and the
runner above now sets the flag per dataset.  If Ego-FB's Closeness EG comes
back as 34.63 -> 22.25 instead of 32.66 -> 17.32, the protocol flag is wrong
again.

Three probe networks only.  Pokec-n and Brightkite were never probed; the
paper scopes the claim accordingly, so do not silently generalize it.
""",
)

if __name__ == "__main__":
    raise SystemExit(main(SPEC))
