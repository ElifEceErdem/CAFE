"""Table 12 -- per-dataset wall-clock runtime.

Appendix A.2.  One-time preprocessing (coreness, RA), training (vanilla,
CAFE-In, CAFE-Pre) and inference (vanilla, hybrid), mean +/- std over the
repetitions, in seconds.

Timings are hardware-bound, so this table must be regenerated on the machine
named in Appendix A.2 (single NVIDIA TITAN RTX) and never mixed across GPUs.
It is the one table in the paper the Equality-Gain fix cannot touch.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.tables._framework import (  # noqa: E402
    PUBLISHED, REPO, TABLES_DIR, Source, TableSpec,
    common_train_flags, large_dataset_flags, main,
)


def runner(ds: str) -> list[str]:
    cfg = PUBLISHED[ds]
    out = os.path.join(REPO, "results", ds, "runtime", f"runtime_{ds}.json")
    return (["python", "-u", "run_runtime_benchmark.py"]
            + common_train_flags(ds)
            + ["--runs", "10", "--rec_k", "10",
               "--cafe_in_lambda", f"{cfg['in_lambda']:g}",
               "--cafe_pre_alpha", f"{cfg['pre_alpha']:g}",
               "--cafe_beta", f"{cfg['in_beta']:g}",
               "--cafe_lam", f"{cfg['in_lam']:g}",
               "--out_json", out]
            + large_dataset_flags(ds))


SPEC = TableSpec(
    key="t12",
    number="12",
    title="Per-dataset wall-clock runtime",
    section="Appendix A.2",
    sources=[
        Source("runtime measurements", "results/{ds}/runtime/runtime_{ds}.json",
               kind="json", eg_sensitive=False, min_runs=1,
               note="timings only; carries no EG columns"),
    ],
    build=[
        ["python", "-u", "analysis/generate_runtime_table.py",
         "--results_dir", os.path.join(REPO, "results"),
         "--out", os.path.join(TABLES_DIR, "runtime_table.tex")],
    ],
    runner=runner,
    artifacts=["results_repro/tables/runtime_table.tex"],
    notes="""
Do not run this concurrently with other GPU jobs on the same device -- shared
occupancy inflates the training column and silently invalidates the 1.00x /
1.02-1.07x overhead claims in Section 5.5.
""",
)

if __name__ == "__main__":
    raise SystemExit(main(SPEC))
