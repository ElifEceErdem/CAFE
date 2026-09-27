"""Shared helpers of the table builders: locations, result readers, LaTeX.

Result files live under ``results/`` (override with ``CAFE_RESULTS`` to build
the tables from another tree with the same layout).  Output goes to ``build/``
(override with ``CAFE_BUILD``).

Two result layouts are read, both written by the runners in ``experiments/``:

* ``results[<method>][<metric>]["values"]`` -- ``run_baselines.py``;
* ``lambda_results[<config>]["statistics"][<metric>]["values"]`` -- the CAFE,
  ablation and fusion runners.  A source may also be a run directory, in which
  case its ``main_results.json`` is read.

Every summary is recomputed from per-run values.
"""
import csv
import json
import math
import os
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("CAFE_RESULTS", REPO / "results")).resolve()
BUILD = Path(os.environ.get("CAFE_BUILD", REPO / "build")).resolve()
INDEX = RESULTS / "index"


def resolve(rel):
    """Absolute path of a file under the results root."""
    path = RESULTS / str(rel)
    if not path.exists():
        raise FileNotFoundError(f"result file not found: {path}")
    return path


def exists(rel):
    return (RESULTS / str(rel)).exists()


METRICS = {"prec": "Precision@10", "deg_eg": "Equality_Gain@10",
           "pr_eg": "Pagerank_EG@10", "cl_eg": "Closeness_EG@10"}


def _json(rel):
    with open(resolve(rel)) as fh:
        return json.load(fh)


def run_file(rel):
    path = resolve(rel)
    return f"{rel}/main_results.json" if path.is_dir() else rel


def run_values(rel, *, config=None, key=None, metric):
    """Per-run values of ``metric`` from one result file or run directory."""
    data = _json(run_file(rel))
    if config:
        block = data["lambda_results"][config]
        stats = block.get("statistics", block)
    else:
        stats = data["results"][key]
    values = stats[metric]["values"]
    assert values and all(math.isfinite(v) for v in values), (rel, metric)
    return list(values)


def summarize(values):
    """Mean and population SD, the convention of every fixed-split table."""
    return {"mean": statistics.fmean(values), "sd": statistics.pstdev(values), "n": len(values)}


def pooled(sources, *, config=None, key=None, metric, n=None):
    """Concatenate per-run values across shards, keeping the first ``n``.

    Deterministic heuristics are stored once per shard; ``n=1`` keeps a single
    evaluation instead of repeated copies.
    """
    values = []
    for rel in sources:
        values.extend(run_values(rel, config=config, key=key, metric=metric))
    if n is not None:
        if len(values) < n:
            raise ValueError(f"{metric}: {len(values)} runs found, {n} required ({sources})")
        values = values[:n]
    return values


def jsonl(rel):
    with open(resolve(rel)) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_json(rel):
    return _json(rel)


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
