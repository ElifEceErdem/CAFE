"""Did the reconstructed pre-2026-07-23 Equality Gain reproduce Table 4?

Compares the output of experiments/jobs/t25_legacy_eg_check.job against the
published values in results_repro/tables/table4_precision_equality_gain.csv.

Read Precision@10 first.  It is metric-independent, so if it does not
reproduce, the run differs from the Jul-19 vintage in more than the Equality
Gain definition and the Degree EG verdict means nothing.

Only metrics/equality_gain.py carries the legacy paste, so Degree EG@10 is the
reconstructed metric while PageRank/Closeness EG stay corrected.  PR EG is
therefore expected NOT to match its published value.

    python analysis/check_legacy_eg_reproduction.py
"""

import argparse
import csv
import glob
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PUBLISHED_CSV = os.path.join(
    REPO, "results_repro", "tables", "table4_precision_equality_gain.csv")

#: (csv metric prefix, json key, label, decimals, is the legacy paste's metric)
METRICS = [
    ("Precision@10",     "Precision@10",     "Prec@10",   4, False),
    ("Equality_Gain@10", "Equality_Gain@10", "Deg EG@10", 2, True),
    ("Pagerank_EG@10",   "Pagerank_EG@10",   "PR EG@10",  2, False),
    ("Closeness_EG@10",  "Closeness_EG@10",  "Cl. EG@10", 2, False),
]
ARMS = [("cafe_in", "CAFE-In"), ("cafe_pre", "CAFE-Pre")]


def published(csv_path):
    """{(dataset, method): {metric: (mean, std)}} from the published table."""
    out = {}
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            key = (row["dataset"], row["method"])
            out[key] = {m: (float(row[f"{m}_Mean"]), float(row[f"{m}_Std"]))
                        for m, _, _, _, _ in METRICS if row.get(f"{m}_Mean")}
    return out


def observed(run_dir):
    """{(dataset, method): {metric: (mean, std)}} from the test-run JSONs."""
    out = {}
    for ds in sorted(os.listdir(run_dir)):
        for arm, label in ARMS:
            paths = [p for p in sorted(glob.glob(
                os.path.join(run_dir, ds, arm, "*.json")))
                if not p.endswith(".partial.json")]
            if not paths:
                continue
            # A finished arm writes one JSON; take the newest if a rerun added more.
            data = json.load(open(paths[-1]))
            for cfg in (data.get("lambda_results") or {}).values():
                stats = cfg.get("statistics") if isinstance(cfg, dict) else None
                if not isinstance(stats, dict):
                    continue
                out[(ds, label)] = {
                    m: (stats[j]["mean"], stats[j].get("std"))
                    for m, j, _, _, _ in METRICS
                    if isinstance(stats.get(j), dict)
                }
                out[(ds, label)]["_semantics"] = (
                    data.get("experiment_info", {}).get("eg_graph_semantics"))
                out[(ds, label)]["_path"] = paths[-1]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir",
                    default=os.path.join(REPO, "results_legacy_eg_check"))
    ap.add_argument("--published_csv", default=PUBLISHED_CSV)
    args = ap.parse_args()

    if not os.path.isdir(args.run_dir):
        raise SystemExit(f"no results at {args.run_dir} -- has the job finished?")

    pub, obs = published(args.published_csv), observed(args.run_dir)
    if not obs:
        raise SystemExit(f"{args.run_dir} holds no finished result JSONs.")

    verdicts = []
    for (ds, label), got in sorted(obs.items()):
        want = pub.get((ds, label))
        print(f"\n{'='*74}\n{ds}  {label}")
        print(f"  semantics marker : {got.get('_semantics')}")
        print(f"  source           : {os.path.relpath(got['_path'], REPO)}")
        if want is None:
            print("  (no published row to compare against)")
            continue
        print(f"\n  {'metric':<12}{'published':>18}{'observed':>18}{'delta':>12}   verdict")
        print(f"  {'-'*72}")
        for m, _j, lbl, nd, is_legacy in METRICS:
            if m not in want or m not in got:
                continue
            (pm, ps), (om, os_) = want[m], got[m]
            d = om - pm
            # "Reproduced" = within one published standard deviation.
            ok = abs(d) <= max(ps, 10 ** -nd)
            if is_legacy:
                mark = "REPRODUCED" if ok else "*** MISMATCH ***"
                verdicts.append((ds, label, ok, d))
            elif m == "Precision@10":
                mark = "model matches" if ok else "*** MODEL DIFFERS ***"
                verdicts.append((ds, label + " [prec]", ok, d))
            else:
                mark = "(corrected metric, mismatch expected)"
            sfmt = f"{ps:.{nd}f}"
            print(f"  {lbl:<12}{pm:>10.{nd}f} ± {sfmt:<5}"
                  f"{om:>10.{nd}f} ± {(f'{os_:.{nd}f}' if os_ is not None else '?'):<5}"
                  f"{d:>+12.{nd}f}   {mark}")

    print(f"\n{'='*74}\nVERDICT")
    ok_all = all(v[2] for v in verdicts)
    for ds, label, ok, d in verdicts:
        print(f"  {'PASS' if ok else 'FAIL'}  {ds:<10}{label:<18}delta={d:+.4f}")
    print()
    if ok_all:
        print("  The reconstruction reproduces the published values. The Apr-21\n"
              "  metric is consistent with what ran on 2026-07-19.")
    else:
        print("  At least one target was missed. If a [prec] row failed, the model\n"
              "  itself differs from the Jul-19 vintage and the Degree EG result is\n"
              "  inconclusive rather than negative.")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
