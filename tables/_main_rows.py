"""Per-method summaries shared by Table 4, Table 7 and Figure 3.

``results/index/main_results_rows.csv`` lists, for every (dataset, method)
point, the result files (or run directories), the configuration key or
baseline result key, and the number of runs the paper reports.
"""
import csv

from tables._common import METRICS, pooled, summarize
from tables._common import INDEX

ROWS_FILE = INDEX / "main_results_rows.csv"
DECIMALS = {"prec": 4, "deg_eg": 2, "pr_eg": 2, "cl_eg": 2}
DETERMINISTIC = {"PA", "RA", "AA", "CN"}


def load_rows():
    with open(ROWS_FILE) as fh:
        return {(r["dataset"], r["method"]): r for r in csv.DictReader(fh)}


def summaries(row):
    sources = row["sources"].split(";")
    out = {}
    for column, metric in METRICS.items():
        values = pooled(sources, config=row["config"] or None, key=row["results_key"] or None,
                        metric=metric, n=int(row["n"]))
        out[column] = summarize(values)
    return out


def input_files():
    files = []
    for row in load_rows().values():
        files.extend(row["sources"].split(";"))
    return files
