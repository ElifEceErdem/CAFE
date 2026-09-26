"""Derive the small exposure summaries used by Tables 6(b) and 9.

Input: per-node target exposure written by ``experiments/run_exposure_acceptance.py``
(``analysis/plots/coreness_by_group/exposure_per_node_<ds>.csv[.gz]``, tens of
MB each, not committed).  Output (committed in the evidence snapshot):

``tier_exposure.csv``   mean per-node exposure per (dataset, run, method,
                        target-coreness tercile).  Terciles follow the paper:
                        nodes with coreness <= q(1/3) are peripheral,
                        <= q(2/3) middle, the rest core -- boundary ties go to
                        the lower group, so coreness-1 nodes are always
                        peripheral.
``group_exposure.csv``  mean per-node exposure per (dataset, run, method,
                        demographic group).

Usage::

    python -m reproduce.tools.derive_exposure [--in_dir DIR] [--out_dir DIR]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from reproduce.paths import REPO

DATASETS = ["brightkite", "ego-facebook", "oklahoma", "pokec_n", "unc"]
DEFAULT_DIR = REPO / "analysis" / "plots" / "coreness_by_group"


def per_node_file(folder, ds):
    for suffix in (".csv", ".csv.gz"):
        path = Path(folder) / f"exposure_per_node_{ds}{suffix}"
        if path.exists():
            return path
    return None


def terciles(coreness):
    q1, q2 = coreness.quantile([1 / 3, 2 / 3])
    return pd.Series(np.where(coreness <= q1, "periphery", np.where(coreness <= q2, "middle", "core")),
                     index=coreness.index), (q1, q2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_dir", default=str(DEFAULT_DIR))
    ap.add_argument("--out_dir", default=str(DEFAULT_DIR))
    args = ap.parse_args(argv)
    tiers, groups = [], []
    for ds in DATASETS:
        path = per_node_file(args.in_dir, ds)
        if path is None:
            print(f"{ds:13s} no per-node exposure file in {args.in_dir}")
            continue
        df = pd.read_csv(path, usecols=["run", "method", "node_id", "group", "coreness", "exposure"])
        core = df.drop_duplicates("node_id").set_index("node_id")["coreness"]
        tier, (q1, q2) = terciles(core)
        df["tier"] = tier.reindex(df["node_id"]).to_numpy()
        t = df.groupby(["run", "method", "tier"])["exposure"].agg(["mean", "size"]).reset_index()
        t.insert(0, "dataset", ds)
        t["q1"], t["q2"] = q1, q2
        tiers.append(t.rename(columns={"mean": "exposure_mean", "size": "n_nodes"}))
        if (df["group"] >= 0).any():
            g = df[df["group"] >= 0].groupby(["run", "method", "group"])["exposure"].mean().reset_index()
            g.insert(0, "dataset", ds)
            groups.append(g.rename(columns={"exposure": "exposure_mean"}))
        print(f"{ds:13s} runs={df['run'].nunique()} terciles q=({q1:g},{q2:g})")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(tiers).to_csv(out / "tier_exposure.csv", index=False)
    pd.concat(groups).to_csv(out / "group_exposure.csv", index=False)
    print(f"wrote {out / 'tier_exposure.csv'} and {out / 'group_exposure.csv'}")


if __name__ == "__main__":
    main()
