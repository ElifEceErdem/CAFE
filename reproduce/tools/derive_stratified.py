"""Derive the small per-run inputs of Figures 1 and 4 from per-node results.

``analysis/run_precision_by_coreness.py`` (Figure 1) and
``analysis/run_stratified_cafe.py`` (Figure 4) save one row per evaluated user
and run (4-11 MB per dataset).  The figures only need

``precision_by_run_<ds>.csv``  Precision@10 per (run, model, coreness bin),
                               i.e. ``aggregate_precision_within_runs``;
``heatmap_nodes_<ds>.csv``     per-node HL-GNN precision averaged over runs,
                               with degree and coreness (Figure 1a).

Usage::

    python -m reproduce.tools.derive_stratified
"""
import sys
from pathlib import Path

import pandas as pd

from reproduce.paths import REPO

sys.path.insert(0, str(REPO / "analysis"))
from coreness_plots import aggregate_precision_within_runs  # noqa: E402

JOBS = [  # (per-node input, output directory, dataset, heatmap?)
    ("analysis/plots/precision_by_coreness_aggregated/precision_by_coreness_unc.csv",
     "analysis/plots/precision_by_coreness_aggregated", "unc", True),
    ("analysis/plots/stratified_figure4/stratified_cafe_oklahoma.csv",
     "analysis/plots/stratified_figure4", "oklahoma", False),
    ("analysis/plots/stratified_figure4/stratified_cafe_brightkite.csv",
     "analysis/plots/stratified_figure4", "brightkite", False),
]


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(REPO), help="tree holding the per-node files (default: repository)")
    root = Path(ap.parse_args(argv).root)
    for src, out, ds, heatmap in JOBS:
        path = root / src
        if not path.exists() and "stratified_figure4" in src:  # original campaign location
            path = REPO / src.replace("analysis/plots/stratified_figure4", "results/validation/20260813_184859/figure4")
        if not path.exists():
            print(f"skip {src} (not present)")
            continue
        df = pd.read_csv(path)
        agg = aggregate_precision_within_runs(df)
        (root / out).mkdir(parents=True, exist_ok=True)
        agg.to_csv(root / out / f"precision_by_run_{ds}.csv", index=False)
        msg = f"{ds}: {len(agg)} run/model/bin rows"
        if heatmap:
            nodes = (df[df["model"] == "HL-GNN"].groupby("node_id", as_index=False)
                     .agg(precision=("precision", "mean"), degree=("degree", "first"),
                          coreness=("node_coreness", "first")))
            nodes.to_csv(root / out / f"heatmap_nodes_{ds}.csv", index=False)
            msg += f", {len(nodes)} heatmap nodes"
        print(msg)


if __name__ == "__main__":
    main()
