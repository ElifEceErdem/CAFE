"""Inputs shared by Table 5 and Table 18 (stratified source-coreness profiles).

``summaries/bin_perturbation/`` is written by
``experiments/bin_perturbation_size.py`` from the stratified runs of
``experiments/run_stratified_cafe.py``.  The printed tables use the snapshot of
2026-07-27.
"""
from tables._common import resolve
from tables._common import read_csv

ROOT = "summaries/bin_perturbation"
FILES = {name: f"{ROOT}/{name}.csv" for name in
         ("bin_perturbation_summary", "bin_size_monotonicity", "bin_size_matched_pairs")}
FOCUS = "CAFE-In"


def input_files():
    return list(FILES.values())


def load(name):
    return read_csv(resolve(FILES[name]))


def label(text):
    return text.strip("[]")
