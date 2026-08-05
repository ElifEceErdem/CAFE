"""Side-by-side comparison of the two corrected-EG campaigns.

Option 1  corrected metric at the PUBLISHED configurations
          (results_repro/tables/table1_precision_equality_gain.csv)
Option 3  corrected metric at the ASF-RE-SELECTED configurations
          (results_asf/tables/table1_precision_equality_gain.csv)

Both trees share their baselines, backbone and fusion sweep, so the rows that
actually differ are the CAFE ones -- this script highlights exactly those and
reports, per dataset and EG dimension, which option puts CAFE in a better
position relative to the backbone.

The point of the comparison is NOT to pick whichever looks better. Option 3
buys its equality numbers with a different operating point (on four of five
networks the ASF selection sets lambda_fair = 0, switching the in-processing
penalty off entirely), so a gain there is not evidence that the penalty works.
Read the two together and choose on the argument you can defend, not the
larger number.

Usage:
    python compare_option1_option3.py \
        --option1 ../results_repro/tables/table1_precision_equality_gain.csv \
        --option3 ../results_asf/tables/table1_precision_equality_gain.csv \
        --out ../results_asf/tables/option1_vs_option3.csv
"""

import argparse
import os
import re

import pandas as pd

DATASETS = ["brightkite", "ego-facebook", "oklahoma", "pokec_n", "unc"]
METRICS = ["Precision@10", "Degree EG@10", "PageRank EG@10", "Closeness EG@10"]
CAFE_ROWS = ["CAFE-In", "CAFE-Pre", "CAFE-In-B", "CAFE-Pre-B", "CAFE-P"]
BACKBONE = "HLGNN"


def value(cell):
    """Mean from a 'x ± y' / 'x' cell; NaN for '-' and blanks."""
    if not isinstance(cell, str):
        return float("nan")
    cell = cell.strip().replace("−", "-")
    if cell in ("-", "", "n/a", "--"):
        return float("nan")
    m = re.match(r"^(-?[\d.]+)", cell)
    return float(m.group(1)) if m else float("nan")


def load(path, label):
    if not os.path.exists(path):
        raise SystemExit(f"{label} table not found: {path}")
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        for ds in DATASETS:
            if ds in df.columns:
                out[(r["Method"], r["Metric"], ds)] = value(r[ds])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--option1", default="../results_repro/tables/table1_precision_equality_gain.csv")
    ap.add_argument("--option3", default="../results_asf/tables/table1_precision_equality_gain.csv")
    ap.add_argument("--out", default="../results_asf/tables/option1_vs_option3.csv")
    args = ap.parse_args()

    o1 = load(args.option1, "Option 1")
    o3 = load(args.option3, "Option 3")

    rows = []
    for ds in DATASETS:
        for metric in METRICS:
            base1 = o1.get((BACKBONE, metric, ds), float("nan"))
            base3 = o3.get((BACKBONE, metric, ds), float("nan"))
            for meth in CAFE_ROWS:
                v1 = o1.get((meth, metric, ds), float("nan"))
                v3 = o3.get((meth, metric, ds), float("nan"))
                if pd.isna(v1) and pd.isna(v3):
                    continue
                rows.append(dict(
                    dataset=ds, method=meth, metric=metric,
                    option1=v1, option3=v3, delta=v3 - v1,
                    option1_vs_backbone=v1 - base1,
                    option3_vs_backbone=v3 - base3,
                ))
    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("no comparable rows -- has either campaign produced tables yet?")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)

    for metric in METRICS:
        sub = out[(out.metric == metric) & (out.method.isin(["CAFE-In", "CAFE-Pre"]))]
        if sub.empty:
            continue
        print(f"\n{metric}  -- CAFE minus backbone, per option")
        print(f"  {'dataset':<14}{'method':<10}{'opt1':>10}{'opt3':>10}"
              f"{'opt1-bb':>10}{'opt3-bb':>10}   better")
        for _, r in sub.iterrows():
            if pd.isna(r.option1) and pd.isna(r.option3):
                continue
            better = ("option3" if r.option3_vs_backbone > r.option1_vs_backbone
                      else "option1" if r.option3_vs_backbone < r.option1_vs_backbone else "=")
            if pd.isna(r.option1_vs_backbone) or pd.isna(r.option3_vs_backbone):
                better = "incomplete"
            print(f"  {r.dataset:<14}{r.method:<10}{r.option1:>10.3f}{r.option3:>10.3f}"
                  f"{r.option1_vs_backbone:>10.3f}{r.option3_vs_backbone:>10.3f}   {better}")

    print(f"\nwrote {args.out}")
    print("\nReminder: on Ego-FB, Oklahoma, UNC and Pokec-n the Option-3 selection sets\n"
          "lambda_fair = 0, so its CAFE-In carries NO in-processing penalty. Any\n"
          "Option-3 advantage there is evidence about fusion and the boost, not\n"
          "about the coreness loss penalty.")


if __name__ == "__main__":
    main()
