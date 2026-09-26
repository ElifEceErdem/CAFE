"""Index the outputs of a reproduction job so the builders read them.

    python -m reproduce.tools.index_runs <OUT>

``<OUT>`` is a job output directory in the layout written by ``reproduce/jobs``::

    <OUT>/<ds>/<arm>/main_results.json          CAFE / ablation / backbone runners
    <OUT>/<ds>/<arm>/acceptance.jsonl           acceptance probe (Table 11)
    <OUT>/<ds>/baselines/<ds>/<group>/*.json    run_baselines.py (--results_dir)
    <OUT>/t15/<ds>/seed_<s>/<arm>/main_results.json   split-resampled arms (Table 15)
    <OUT>/t15/<ds>/seed_<s>/ra/<ds>/baselines/*.json

The frozen indexes in ``reproduce/sources`` are copied to ``<OUT>/sources`` and
every row the job re-ran is replaced by its new output (paths relative to
``<OUT>``).  Build with ``CAFE_EVIDENCE=<OUT> CAFE_SOURCES=<OUT>/sources``;
rows that were not re-run keep resolving to the committed evidence snapshot.
"""
import csv
import json
import shutil
import sys
from pathlib import Path

from reproduce.config import published
from reproduce.paths import PACKAGE

ARM_METHOD = {"hl-gnn": "HL-GNN", "fusion": "HL-GNN+RA", "cafe-in": "CAFE-In", "cafe-pre": "CAFE-Pre",
              "cafe-in-b": "CAFE-In-B", "cafe-pre-b": "CAFE-Pre-B", "cafe-p": "CAFE-P", "ci-in": "CI-In",
              "ci-pre": "CI-Pre", "mb-post": "MB-Post"}
BASELINE_KEY = {"random": "Random", "PA": "PA", "RA": "RA", "AA": "AA", "CN": "CN", "LightGCN": "LightGCN",
                "SSNet": "SSNet", "Cross-Link Bias": "Cross-LinkBias", "LTLP": "LTLP", "TailSteak": "TailSTEAK",
                "FairAdj": "FairAdj", "LBA": "LBA", "MORAL": "MORAL"}
DETERMINISTIC = {"PA", "RA", "AA", "CN"}
T15_METHOD = {"hl-gnn": "HL-GNN", "fusion": "Fusion", "cafe-in": "CAFE-In", "cafe-pre": "CAFE-Pre"}


def config_for(ds, method):
    c = published(ds)
    return {"HL-GNN": "beta_0.0_lambda_0.0",
            "HL-GNN+RA": f"lambda_fair_0.0_beta_{c['FUSION_BETA']}_lam_0.0",
            "CAFE-In": f"lambda_fair_{c['IN_LF']}_beta_{c['IN_BETA']}_lam_{c['IN_LAM']}",
            "CAFE-Pre": f"preproc_alpha_{c['PRE_A']}_beta_{c['PRE_BETA']}_lam_{c['PRE_LAM']}",
            "CAFE-In-B": f"lambda_fair_{c['INB_LF']}_beta_{c['INB_BETA']}_lam_0.0",
            "CAFE-Pre-B": f"preproc_alpha_{c['PREB_A']}_beta_{c['PREB_BETA']}_lam_0.0",
            "CAFE-P": f"beta_{c['P_BETA']}_lambda_{c['P_LAM']}",
            "CI-In": f"lambda_{c['CI_IN']}", "CI-Pre": f"lambda_{c['CI_PRE']}",
            "MB-Post": f"lambda_{c['MB_POST']}"}[method]


def read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def index_main(out, rows):
    replaced = 0
    by_key = {(r["dataset"], r["method"]): r for r in rows}
    for ds_dir in sorted(p for p in out.iterdir() if p.is_dir() and p.name not in ("sources", "t15", "build")):
        ds = ds_dir.name
        for arm, method in ARM_METHOD.items():
            f = ds_dir / arm / "main_results.json"
            if f.exists() and (ds, method) in by_key:
                by_key[(ds, method)].update(config=config_for(ds, method), results_key="", n="10",
                                            sources=str(f.relative_to(out)))
                replaced += 1
        for f in sorted((ds_dir / "baselines").glob(f"{ds}/*/results_*.json")):
            if f.name.endswith(".partial.json"):
                continue
            for key in json.loads(f.read_text()).get("results", {}):
                method = BASELINE_KEY.get(key)
                if method and (ds, method) in by_key:
                    by_key[(ds, method)].update(config="", results_key=key, sources=str(f.relative_to(out)),
                                                n="1" if method in DETERMINISTIC else "10")
                    replaced += 1
    return replaced


def index_acceptance(out, rows):
    replaced = 0
    for r in rows:
        arm = {"RA": "ra", "HL-GNN": "hl-gnn", "HL-GNN+RA": "fusion", "CAFE-In": "cafe-in",
               "CAFE-Pre": "cafe-pre"}[r["method"]]
        f = out / r["dataset"] / arm / "acceptance.jsonl"
        if f.exists():
            beta = published(r["dataset"])["FUSION_BETA_MATCHED"]
            r.update(format="jsonl", source=str(f.relative_to(out)),
                     config=f"lambda_fair_0.0_beta_{beta}_lam_0.0" if arm == "fusion" else "")
            replaced += 1
    return replaced


def index_t15(out, rows):
    root = out / "t15"
    if not root.exists():
        return rows, 0
    kept = {(r["dataset"], r["seed"]): r for r in rows}
    new_rows, replaced = [], 0
    for (ds, seed), frozen in kept.items():
        block = root / ds / f"seed_{seed}"
        arms = {m: block / a / "main_results.json" for a, m in T15_METHOD.items()}
        ra = sorted((block / "ra" / ds / "baselines").glob("results_baselines_*.json"))
        if all(p.exists() for p in arms.values()) and ra:
            for method, p in arms.items():
                cfg = "lambda_fair_0.0_beta_{}_lam_0.0".format(published(ds)["FUSION_BETA"]) if method == "Fusion" \
                    else config_for(ds, method)
                new_rows.append(dict(dataset=ds, seed=seed, format="arm_json", source=str(p.relative_to(out)),
                                     method=method, config=cfg, results_key=""))
            new_rows.append(dict(dataset=ds, seed=seed, format="arm_json", source=str(ra[0].relative_to(out)),
                                 method="RA", config="", results_key="RA"))
            replaced += 1
        else:
            new_rows.append(dict(frozen, method="", config="", results_key=""))
    return new_rows, replaced


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit(__doc__)
    out = Path(argv[0]).resolve()
    dst = out / "sources"
    dst.mkdir(parents=True, exist_ok=True)
    for f in (PACKAGE / "sources").glob("*.csv"):
        shutil.copy2(f, dst / f.name)
    main_rows = read(dst / "main_results_rows.csv")
    n_main = index_main(out, main_rows)
    write(dst / "main_results_rows.csv", main_rows)
    acc_rows = read(dst / "acceptance_rows.csv")
    n_acc = index_acceptance(out, acc_rows)
    write(dst / "acceptance_rows.csv", acc_rows)
    t15_rows, n_t15 = index_t15(out, read(dst / "paired_observations.csv"))
    write(dst / "paired_observations.csv", t15_rows)
    print(f"indexed {out}: {n_main} Table 4/7 rows, {n_acc} Table 11 rows, {n_t15} Table 15 blocks re-run")
    print(f"build with: CAFE_EVIDENCE={out} CAFE_SOURCES={dst} python -m reproduce check")


if __name__ == "__main__":
    main()
