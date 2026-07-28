"""Table 7 -- ablation results across the five datasets.

Section 7.3 (RQ4).  Rows grouped by how many CAFE stages are applied to the
backbone: single component (CI-In, CI-Pre, MB-Post, HL-GNN+RA), two-stage
(CAFE-In/Pre without the boost, CAFE without the penalty), full pipeline.

Two provenance notes that this table's ``verify`` stage exists to catch:

* The Pokec-n CI-Pre and MB-Post runs finished after the published table was
  generated, which is why the draft omits them.  They are present now.
* ``results_repro/pokec_n/mb_post`` is the ONLY file in the tree carrying the
  corrected ``eg_graph_semantics`` marker.  In the published (pre-fix) table
  its Degree EG is therefore on a different scale from every neighbouring
  cell and would take the bold "best" mark.  Under the corrected-EG plan this
  inverts: every other row must be re-run to join it.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.tables._framework import (  # noqa: E402
    PUBLISHED_SINGLE, RESULTS, TABLES_DIR, Source, TableSpec,
    common_train_flags, eg_flags, large_dataset_flags, main,
)


def runner(ds: str) -> list[str]:
    """CI-In single-component ablation; CI-Pre/MB-Post differ only in the mode."""
    lam = PUBLISHED_SINGLE[ds]["ci_in"]
    out = os.path.join(RESULTS, ds, "ci_in")
    return (["python", "-u", "run_coreness_inverse.py"]
            + common_train_flags(ds)
            + ["--runs", "10", "--coreness_alpha", "1.0",
               "--fairness_method", "coreness_inverse_inproc",
               "--lambda_fair", f"{lam:g}",
               "--use_stratified", "False", "--use_pa_aware", "False",
               "--use_ndkl", "False"]
            + eg_flags()
            + ["--results_file", os.path.join(out, f"results_ci_in_{ds}.json")]
            + large_dataset_flags(ds))


SPEC = TableSpec(
    key="t07",
    number="7",
    title="Ablation results across five datasets",
    section="7.3 (RQ4)",
    sources=[
        Source("HL-GNN backbone", "results_repro/{ds}/cafe_p/*_baseline_*.json"),
        Source("CI-In", "results_repro/{ds}/ci_in/*.json"),
        Source("CI-Pre", "results_repro/{ds}/ci_pre/*.json"),
        Source("MB-Post", "results_repro/{ds}/mb_post/*.json"),
        Source("HL-GNN+RA (fusion control)", "results_repro/{ds}/fusion_only/*.json"),
        Source("CAFE-In^-B", "results_repro/{ds}/cafe_in_b/*.json"),
        Source("CAFE-Pre^-B", "results_repro/{ds}/cafe_pre_b/*.json"),
        Source("CAFE^-P", "results_repro/{ds}/cafe_p/*_cafe_p_*.json"),
        Source("CAFE-In", "results_repro/{ds}/cafe_in/*.json"),
        Source("CAFE-Pre", "results_repro/{ds}/cafe_pre/*.json"),
    ],
    build=[
        ["python", "-u", "analysis/generate_ablation_tables.py",
         "--base_dir", RESULTS, "--out_dir", TABLES_DIR],
        ["python", "-u", "analysis/generate_merged_ablation_table.py",
         "--tables_dir", TABLES_DIR,
         "--out", os.path.join(TABLES_DIR, "ablation_merged.tex")],
    ],
    runner=runner,
    artifacts=[
        "results_repro/tables/table6_ablation_all.tex",
        "results_repro/tables/table3_ablation_oklahoma.tex",
        "results_repro/tables/ablation_merged.tex",
    ],
    notes="""
generate_ablation_tables.py re-selects the best config per method group from
--base_dir rather than reading best_configs.csv, so newly finished runs are
picked up automatically.  It currently yields a complete 10-row block on all
five datasets.

Note that best_configs.csv has no MB-Post/pokec_n row; add it
(MB-Post,pokec_n,lambda_0.1,...) so the config file matches what the table
reports.
""",
)

if __name__ == "__main__":
    raise SystemExit(main(SPEC))
