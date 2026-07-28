"""Table 11 -- Equality Gain under relaxed acceptance models (RQ6).

Section 7.6.  Degree and PageRank EG under full and precision-weighted (PW)
acceptance, and Closeness EG under full and true-positive (TP) acceptance, for
RA, HL-GNN, HL-GNN+RA, CAFE-In and CAFE-Pre.  Also emits the appendix table of
empirical rank precisions rho(r) that define the PW weighting -- the paper
quotes rho(1) in prose without a table to point at, so build both.

One runner produces this table AND Table 6(b): ``run_exposure_acceptance.py``
trains each method once per repetition and derives the acceptance-robust EG
and the per-node target exposure from the same recommendation lists.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.tables._framework import (  # noqa: E402
    DATA_PATH, PLOTS_DIR, PUBLISHED, REPO, TABLES_DIR, Source, TableSpec,
    common_train_flags, large_dataset_flags, main,
)

ACCEPT_GLOB = "results/*/acceptance_eg/acceptance_eg_*.csv"
EXPOSURE_DIR = os.path.join(PLOTS_DIR, "coreness_by_group")


def runner(ds: str) -> list[str]:
    cfg = PUBLISHED[ds]
    out = os.path.join(REPO, "results", ds, "acceptance_eg", f"acceptance_eg_{ds}.csv")
    return (["python", "-u", "run_exposure_acceptance.py"]
            + common_train_flags(ds)
            + ["--runs", "10", "--K", "10",
               "--cafe_in_lambda", f"{cfg['in_lambda']:g}",
               "--cafe_in_beta", f"{cfg['in_beta']:g}",
               "--cafe_in_lam", f"{cfg['in_lam']:g}",
               "--cafe_pre_alpha", f"{cfg['pre_alpha']:g}",
               "--cafe_pre_beta", f"{cfg['pre_beta']:g}",
               "--cafe_pre_lam", f"{cfg['pre_lam']:g}",
               "--fusion_beta", f"{cfg['fusion_beta']:g}",
               "--use_sampling_closeness", "True",
               "--acceptance_csv", out,
               "--exposure_dir", EXPOSURE_DIR]
            + large_dataset_flags(ds))


SPEC = TableSpec(
    key="t11",
    number="11",
    title="EG under acceptance models (full / PW / TP)",
    section="7.6 (RQ6)",
    sources=[
        Source("acceptance-robust EG", "results/{ds}/acceptance_eg/acceptance_eg_{ds}.csv",
               kind="csv",
               note="long format: run, method, metric, value, plus rho_rank_* rows"),
    ],
    build=[
        ["python", "-u", "analysis/generate_acceptance_table.py",
         "--input_glob", ACCEPT_GLOB,
         "--out", os.path.join(TABLES_DIR, "acceptance_table.tex"),
         "--rho_out", os.path.join(TABLES_DIR, "acceptance_weights_table.tex")],
    ],
    runner=runner,
    artifacts=[
        "results_repro/tables/acceptance_table.tex",
        "results_repro/tables/acceptance_weights_table.tex",
    ],
    notes="""
These CSVs live under results/ rather than results_repro/ -- that is the
established layout for this experiment, not an accident, and the generator's
--input_glob points there.

Repetition counts are UNEVEN in the published data: n=10 on Ego-FB, Oklahoma,
UNC and Brightkite, n=3 on Pokec-n.  The current caption does not say so.
Either bring Pokec-n to 10 or restore the disclosure; the generator prints
the counts it found, so check them against whatever the caption claims.

The fusion row uses CAFE-In's beta (matched-beta control), which on Pokec-n
differs from the independently tuned HL-GNN+RA row of Table 4.  That is
intentional; do not "fix" it by aligning the two.
""",
)

if __name__ == "__main__":
    raise SystemExit(main(SPEC))
