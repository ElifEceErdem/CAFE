"""Table 2 -- dataset statistics.

Section 6.1.  Nodes, undirected edges, average degree, average clustering
coefficient and which sensitive attribute (if any) each network carries.

Computed directly from the loaded graphs, so it needs no GPU and no training
run -- but it must be built from the SAME loader the experiments use, because
the reported |V| is the graph after isolated-node removal (Ego-FB 4,039;
Oklahoma 3,076 from 3,111 raw; UNC 3,985 from 4,018 raw).  Quoting raw file
counts here would contradict the evaluation-population numbers in Appendix A.1.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.tables._framework import (  # noqa: E402
    DATA_PATH, REPO, TABLES_DIR, TableSpec, main,
)

SPEC = TableSpec(
    key="t02",
    number="2",
    title="Dataset statistics",
    section="6.1",
    sources=[],
    build=[
        ["python", "-u", "analysis/generate_dataset_table.py",
         "--data_path", DATA_PATH,
         "--out", os.path.join(TABLES_DIR, "dataset_table.tex")],
    ],
    artifacts=["results_repro/tables/dataset_table.tex",
               "results_repro/tables/dataset_table.csv"],
    notes="""
CPU only, a couple of minutes.  Clustering on the two large graphs is the
slow part; the generator uses the exact average clustering coefficient, which
is what the published table reports.
""",
)

if __name__ == "__main__":
    raise SystemExit(main(SPEC))
