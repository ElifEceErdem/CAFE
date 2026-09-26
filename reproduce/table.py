"""Tidy table cells, CSV output and booktabs LaTeX rendering.

A built artifact is a list of *cells* with the same key as the expected
values extracted from the PDF (``artifact, panel, dataset, row, column``).
Numbers are stored unrounded; ``decimals`` is the printed precision.
"""
import csv
import math
from pathlib import Path

FIELDS = ["artifact", "panel", "dataset", "row", "column", "mean", "sd", "decimals", "extra"]
DATASET_LABEL = {"brightkite": "Brightkite", "ego-facebook": "Ego-FB", "oklahoma": "Oklahoma",
                 "pokec_n": "Pokec-n", "unc": "UNC"}
DATASET_ORDER = ["brightkite", "ego-facebook", "oklahoma", "pokec_n", "unc"]
METHOD_LABEL = {"CAFE-In-B": r"CAFE-In$^{-B}$", "CAFE-Pre-B": r"CAFE-Pre$^{-B}$", "CAFE-P": r"CAFE$^{-P}$"}


def cell(artifact, dataset, row, column, mean, sd=None, decimals=2, panel="", extra=""):
    return dict(artifact=artifact, panel=panel, dataset=dataset, row=row, column=column,
                mean=mean, sd=sd, decimals=decimals, extra=extra)


def summary_cells(artifact, dataset, row, summaries, decimals, panel="", deterministic=False):
    """Cells for one method row from ``{column: {"mean", "sd", "n"}}``."""
    out = []
    for column, s in summaries.items():
        sd = None if deterministic else s["sd"]
        out.append(cell(artifact, dataset, row, column, s["mean"], sd, decimals[column], panel))
    return out


def write_csv(cells, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for c in cells:
            w.writerow({k: ("" if c.get(k) is None else c.get(k)) for k in FIELDS})


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def fmt(mean, sd=None, decimals=2, sign=False):
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "--"
    body = f"{mean:+.{decimals}f}" if sign else f"{mean:.{decimals}f}"
    if sd is not None:
        body += f" $\\pm$ {sd:.{decimals}f}"
    return body


def emphasis(values, higher_is_better=True):
    """Map row index -> 'best' / 'second' by distinct unrounded means."""
    ranked = sorted({v for v in values if v is not None}, reverse=higher_is_better)
    out = {}
    for i, v in enumerate(values):
        if v is None:
            continue
        if ranked and v == ranked[0]:
            out[i] = "best"
        elif len(ranked) > 1 and v == ranked[1]:
            out[i] = "second"
    return out


def decorate(text, mark):
    if mark == "best":
        return rf"\textbf{{{text}}}"
    if mark == "second":
        return rf"\underline{{{text}}}"
    return text


def tabular(header, body_rows, align, caption=None, label=None, notes=None):
    """A standalone booktabs table body (``table`` environment optional)."""
    lines = []
    if caption:
        lines += [r"\begin{table}[t]", r"\centering\small", rf"\caption{{{caption}}}"]
        if label:
            lines.append(rf"\label{{{label}}}")
    lines += [rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for row in body_rows:
        lines.append(r"\midrule" if row == "MIDRULE" else " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        lines.append(rf"\par\footnotesize {notes}")
    if caption:
        lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def write_text(text, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
