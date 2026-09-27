"""Table 15 -- paired contrasts of CAFE against HL-GNN, HL-GNN+RA and RA (Appendix B).

Split-resampled protocol: seed s draws split s (Ego-FB, Oklahoma, Pokec-n,
UNC); Brightkite keeps its temporal split.  For every (variant, dataset,
metric) the entry is the mean paired difference CAFE - reference with the
pointwise 95% t-interval half-width.  Stars are Holm-adjusted two-sided paired
t-tests within each predeclared 40-contrast family (one family per reference).

Observation files (``paired_observations.json``, one per dataset and seed):

* Ego-FB, Oklahoma, UNC -- ``20260905_163304_table15_aligned_prepared``;
* Brightkite, Pokec-n seeds 0-4 -- ``20260907_181811_closeness_exact_v1/table15``;
* Pokec-n seeds 5-9 -- ``20260910_061800_closeness_v1_recovery/table15``;

as listed in ``results/index/paired_observations.csv``.
"""
import csv
import math
import statistics

from scipy import stats as sps

from tables._common import METRICS, read_json, run_values
from tables._common import INDEX, resolve
from tables._common import DATASET_LABEL, DATASET_ORDER, cell, read_csv, tabular, write_text

ID = "T15"
VARIANTS = ["CAFE-In", "CAFE-Pre"]
REFERENCES = {"vs_hlgnn": ("HL-GNN", "(a) vs. HL-GNN backbone"),
              "vs_fusion": ("Fusion", "(b) vs. fusion-only control"),
              "vs_ra": ("RA", "(c) vs. RA")}
DECIMALS = {"prec": 4, "deg_eg": 2, "pr_eg": 2, "cl_eg": 2}


def index():
    with open(INDEX / "paired_observations.csv") as fh:
        return list(csv.DictReader(fh))


def input_files():
    return sorted({r["source"] for r in index()})


def observations():
    """(dataset, method, seed) -> metrics from the indexed observation files.

    ``paired_json``: one ``paired_observations.json`` per (dataset, seed), as
    written by the September validation campaigns.  ``per_seed_csv``: the
    long-format CSV of ``experiments/run_significance.py`` (all seeds of one
    dataset).  ``arm_json``: one runner result per (dataset, seed, method) with
    its configuration key (runs trained with ``experiments/run_with.py --split-seed``).
    """
    obs = {}
    for r in index():
        ds, seed = r["dataset"], int(r["seed"])
        if r["format"] == "paired_json":
            pack = read_json(r["source"])
            assert pack["pairing_verified"] and pack["task"]["dataset"] == ds and pack["task"]["seed"] == seed
            for m in pack["methods"]:
                obs[(ds, m["method"], seed)] = m["metrics"]
        elif r["format"] == "arm_json":
            obs[(ds, r["method"], seed)] = {
                metric: run_values(r["source"], config=r.get("config") or None, key=r.get("results_key") or None,
                                   metric=metric)[0]
                for metric in METRICS.values()}
        else:
            for row in read_csv(resolve(r["source"])):
                if int(row["seed"]) == seed and row["metric"] in METRICS.values():
                    obs.setdefault((ds, row["method"], seed), {})[row["metric"]] = float(row["value"])
    return obs


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def contrasts():
    obs = observations()
    rows = []
    for panel, (ref, _) in REFERENCES.items():
        family = []
        for variant in VARIANTS:
            for ds in DATASET_ORDER:
                for column, metric in METRICS.items():
                    diffs = [obs[(ds, variant, s)][metric] - obs[(ds, ref, s)][metric] for s in range(10)]
                    n, mean, sd = len(diffs), statistics.fmean(diffs), statistics.stdev(diffs)
                    half = sps.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
                    p = sps.ttest_1samp(diffs, 0.0).pvalue if sd > 0 else 0.0
                    family.append(dict(panel=panel, variant=variant, dataset=ds, column=column,
                                       mean=mean, half=half, p=p))
        for row, p_holm in zip(family, holm([r["p"] for r in family])):
            row["p_holm"], row["stars"] = p_holm, stars(p_holm)
        rows += family
    return rows


def build(out):
    rows = contrasts()
    cells = [cell(ID, r["dataset"], r["variant"], r["column"], r["mean"], r["half"], DECIMALS[r["column"]],
                  panel=r["panel"], extra=r["stars"]) for r in rows]
    lookup = {(r["panel"], r["variant"], r["dataset"], r["column"]): r for r in rows}
    tex = []
    for panel, (_, title) in REFERENCES.items():
        body = []
        for i, variant in enumerate(VARIANTS):
            if i:
                body.append("MIDRULE")
            for j, ds in enumerate(DATASET_ORDER):
                line = [variant if j == 0 else "", DATASET_LABEL[ds]]
                for column in METRICS:
                    r = lookup[(panel, variant, ds, column)]
                    d = DECIMALS[column]
                    sup = f"^{{{r['stars']}}}" if r["stars"] else ""
                    line.append(f"${r['mean']:+.{d}f} \\pm {r['half']:.{d}f}{sup}$")
                body.append(line)
        tex.append(rf"\subcaption*{{{title}}}")
        tex.append(tabular(["Variant", "Dataset", "Prec@10", "Deg EG@10", "PR EG@10", "Cl. EG@10"], body, "llrrrr"))
    write_text("% Table 15 -- generated by `python -m tables T15`\n" + "\n".join(tex), out / f"{ID}.tex")
    return cells
