"""
Welch tests for the Table 5 demographic comparison (RQ5).

Family, declared in advance: every CAFE-block row (fusion control plus the
five CAFE variants) against every attribute-dependent baseline (FairAdj,
LBA, MORAL), on Precision@10, Degree EG@10, and PageRank EG@10, on the three
datasets with sensitive attributes. Holm correction over the whole family.
DeltaSP and DeltaEO are deliberately not tested: the paper makes no
superiority claim on them.

Inputs are the per-method mean/std over the n=10 fixed-split repetitions
recorded in table5_demographic_fairness.csv (Welch's t needs only these
sufficient statistics). Stored stds may be population (ddof=0) depending on
the runner, so they are inflated by sqrt(n/(n-1)) to keep the test
conservative.

Usage:
    python demographic_significance.py [--tables_dir ../results_repro/tables]
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

N_RUNS = 10
METRICS = ["Precision@10", "Equality_Gain@10", "Pagerank_EG@10"]
BASELINES = ["FairAdj", "LBA", "MORAL"]


def welch(m1, s1, n1, m2, s2, n2):
    v1, v2 = s1**2 / n1, s2**2 / n2
    t = (m1 - m2) / np.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))
    p = 2 * stats.t.sf(abs(t), df)
    return t, df, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables_dir", type=str,
                        default=os.path.abspath(os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "../results_repro/tables")))
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.tables_dir,
                                  "table5_demographic_fairness.csv"))
    inflate = np.sqrt(N_RUNS / (N_RUNS - 1))

    rows = []
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds].set_index("method")
        cafe_rows = df[(df["dataset"] == ds)
                       & (df["group"].str.startswith("CAFE"))]["method"]
        for cafe in cafe_rows:
            for base in BASELINES:
                if base not in sub.index:
                    continue
                for m in METRICS:
                    m1 = sub.loc[cafe, f"{m}_Mean"]
                    s1 = sub.loc[cafe, f"{m}_Std"] * inflate
                    m2 = sub.loc[base, f"{m}_Mean"]
                    s2 = sub.loc[base, f"{m}_Std"] * inflate
                    t, dof, p = welch(m1, s1, N_RUNS, m2, s2, N_RUNS)
                    rows.append({"dataset": ds, "cafe": cafe,
                                 "baseline": base, "metric": m,
                                 "diff": m1 - m2, "t": t, "df": dof, "p": p})

    out = pd.DataFrame(rows)
    # Holm over the full declared family
    order = np.argsort(out["p"].values)
    k = len(out)
    adj = np.empty(k)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (k - rank) * out["p"].values[idx])
        adj[idx] = min(running_max, 1.0)
    out["p_holm"] = adj
    out["significant_0.001"] = out["p_holm"] < 0.001

    path = os.path.join(args.tables_dir, "table5_significance.csv")
    out.to_csv(path, index=False)
    print(f"{k} contrasts written: {path}")
    print(f"all CAFE-minus-baseline diffs positive: {(out['diff'] > 0).all()}")
    print(f"min |t| = {out['t'].abs().min():.2f}  "
          f"(dataset={out.loc[out['t'].abs().idxmin(), 'dataset']}, "
          f"{out.loc[out['t'].abs().idxmin(), 'cafe']} vs "
          f"{out.loc[out['t'].abs().idxmin(), 'baseline']} on "
          f"{out.loc[out['t'].abs().idxmin(), 'metric']})")
    print(f"max Holm-adjusted p = {out['p_holm'].max():.3g}")


if __name__ == "__main__":
    main()
