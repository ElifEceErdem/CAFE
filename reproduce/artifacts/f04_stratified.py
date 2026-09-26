"""Figure 4 -- source-coreness profiles on Oklahoma and proxy-temporal Brightkite (Section 7.2).

Precision@10 (top) and stratified Degree EG@10 (bottom) for RA, HL-GNN, Fusion,
CAFE-In and CAFE-Pre.  Inputs: the stratified runs of 2026-08-13/14
(``analysis/run_stratified_cafe.py``, job ``t05_t18_stratified.job``), reduced by
``python -m reproduce.tools.derive_stratified``.
"""
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from reproduce.artifacts._plotting import coreness_plots as cp
from reproduce.paths import resolve

ID = "F04"
ROOT = "analysis/plots/stratified_figure4"  # stratified runs of 2026-08-13/14
DATASETS = {"oklahoma": "Oklahoma", "brightkite": "Brightkite"}
ORDER = ["RA", "HL-GNN", "Fusion", "CAFE-In", "CAFE-Pre"]


def input_files():
    return [f"{ROOT}/{kind}_{ds}.csv" for ds in DATASETS for kind in ("precision_by_run", "eg_by_coreness_cafe")]


def build(out):
    out.mkdir(parents=True, exist_ok=True)
    for ds, label in DATASETS.items():
        prec = pd.read_csv(resolve(f"{ROOT}/precision_by_run_{ds}.csv"))
        eg = pd.read_csv(resolve(f"{ROOT}/eg_by_coreness_cafe_{ds}.csv"))
        with tempfile.TemporaryDirectory() as tmp:
            cp.plot_stacked_barchart(prec, eg, tmp, ds, 10, model_order=ORDER, display_name=label,
                                     filename_tag=ds, figure_size=(6, 5.1), font_size=12, legend_inside=True)
            produced = next(Path(tmp).glob("*.pdf"))
            shutil.copy2(produced, out / f"{ID}_{ds}.pdf")
    return None
