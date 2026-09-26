"""Inputs shared by Tables 8, 9 and 16 (demographic fairness).

* Fixed split: ``results_repro/tables/table5_demographic_fairness.csv`` written by
  ``analysis/generate_table5.py`` (attribute-dependent baselines and CAFE
  variants) and the HL-GNN backbone runs in ``results_repro/demographic_backbone``.
* Split-resampled: ``results_repro/demographic_significance/per_seed_metrics_<ds>.csv``
  from ``experiments/run_significance.py`` (seed s trains on split s).
Job: ``reproduce/jobs/t08_demographic.job``.
"""
import statistics

from reproduce.loaders import read_json
from reproduce.paths import resolve
from reproduce.table import read_csv

DATASETS = ["oklahoma", "pokec_n", "unc"]
TABLE5 = "results_repro/tables/table5_demographic_fairness.csv"
BACKBONE = {ds: f"results_repro/demographic_backbone/results_cafe_p_baseline_{ds}.json"  # HL-GNN runs of 2026-07-19
            for ds in DATASETS}
PER_SEED = "results_repro/demographic_significance/per_seed_metrics_{ds}.csv"  # split-resampled runs of 2026-07-20
LABEL = {"HLGNN+RA": "HL-GNN+RA"}
MEASURES = {"prec": "Precision@10", "dsp": "Delta_SP@10", "deo": "Delta_EO@10"}


def input_files():
    return [TABLE5] + list(BACKBONE.values()) + [PER_SEED.format(ds=ds) for ds in DATASETS]


def fixed_split():
    """(dataset, method) -> {measure: (mean, sd)} on the fixed split."""
    out = {}
    for ds, rel in BACKBONE.items():
        stats = read_json(rel)["lambda_results"]["beta_0.0_lambda_0.0"]["statistics"]
        out[(ds, "HL-GNN")] = {k: (stats[m]["mean"], stats[m]["std"]) for k, m in MEASURES.items()}
    for r in read_csv(resolve(TABLE5)):
        method = LABEL.get(r["method"], r["method"])
        out[(r["dataset"], method)] = {k: (float(r[f"{m}_Mean"]), float(r[f"{m}_Std"])) for k, m in MEASURES.items()}
    return out


def resampled():
    """(dataset, method, metric) -> list of per-seed values (seed order)."""
    out = {}
    for ds in DATASETS:
        rows = read_csv(resolve(PER_SEED.format(ds=ds)))
        for r in sorted(rows, key=lambda r: int(r["seed"])):
            out.setdefault((ds, r["method"], r["metric"]), []).append(float(r["value"]))
    return out


def mean_sd(values):
    return statistics.fmean(values), statistics.stdev(values)
