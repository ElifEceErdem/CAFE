"""Figure 1 -- coreness-stratified diagnosis on UNC at K=10 (Section 4).

(a) HL-GNN Precision@10 by degree group x coreness group; (b) Precision@10
and Degree EG per source-coreness group for CN, RA, AA, LightGCN, HL-GNN.
Inputs from ``experiments/run_precision_by_coreness.py``,
reduced by ``python experiments/summarize_stratified.py``.
"""
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from tables._plotting import coreness_plots as cp
from tables._common import resolve

ID = "F01"
ROOT = "summaries/precision_by_coreness"
FILES = {"prec": f"{ROOT}/precision_by_run_unc.csv", "eg": f"{ROOT}/eg_by_coreness_unc.csv",
         "nodes": f"{ROOT}/heatmap_nodes_unc.csv"}


def input_files():
    return list(FILES.values())


def build(out):
    out.mkdir(parents=True, exist_ok=True)
    prec = pd.read_csv(resolve(FILES["prec"]))
    eg = pd.read_csv(resolve(FILES["eg"]))
    nodes = pd.read_csv(resolve(FILES["nodes"]))
    with tempfile.TemporaryDirectory() as tmp:
        cp.plot_stacked_barchart(prec, eg, tmp, "unc", 10, model_order=["CN", "RA", "AA", "LightGCN", "HL-GNN"])
        cp.plot_2d_heatmap(nodes, "HL-GNN", "unc", 10, 4, 5, tmp)
        shutil.copy2(Path(tmp) / "stacked_barchart_unc.pdf", out / f"{ID}b_profiles.pdf")
        shutil.copy2(Path(tmp) / "heatmap_HL-GNN_unc_k10.pdf", out / f"{ID}a_heatmap.pdf")
    return None
