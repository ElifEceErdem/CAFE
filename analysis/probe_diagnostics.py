"""Per-arm diagnostics for the shared-backbone closeness probe (Section 7.5).

Section 7.5 argues that the mobility boost's Closeness-EG effect is not
mediated by the distance profile of the recommended pairs.  That argument
quotes three quantities per probe arm -- the train-graph distance spectrum,
the mean target degree, and the graph-level Gini / Closeness EG -- none of
which appear in Table 10, which reports only the per-quintile closeness
effect.  This script renders them, so every number in the paragraph is
recoverable without hand-rolling pandas over the probe CSVs.

Nothing is retrained or recomputed here: the arms come from
``analysis/closeness_mechanism.py``, which writes ``distance_spectrum_*.csv``
and ``path_stats_*.csv`` alongside the ``closeness_by_bin_*.csv`` that backs
Table 10.

Source directories follow ``generate_closeness_stratum_table.py`` exactly --
Ego-FB from ``closeness_mechanism_paired/``, Oklahoma and UNC from
``closeness_mechanism/`` -- so the printed values are the same campaign that
backs Table 10.  Point ``--plots_dir`` elsewhere to diagnose another campaign,
but do not mix the two in one paragraph.

The two probes do not share a resampling protocol, and the printed
``protocol`` column says which one each dataset used.  Ego-FB was run with
``--vary_split true`` (run r draws split seed r, so the ten reruns are
split-resampled); Oklahoma and UNC were run on the frozen split with ten
training seeds.  Do not describe Table 10 as one protocol.

Usage:
    python analysis/probe_diagnostics.py
    python analysis/probe_diagnostics.py --plots_dir analysis/plots --out_dir /tmp
"""

import argparse
import os

import pandas as pd

# dataset -> (label, plots subdirectory); see module docstring
PROBES = [
    ("ego-facebook", "Ego-FB", "closeness_mechanism_paired"),
    ("oklahoma", "Oklahoma", "closeness_mechanism"),
    ("unc", "UNC", "closeness_mechanism"),
]

DISTANCE_COLUMNS = ["d=2", "d=3", "d=4", "d>=5", "unreachable"]
SCALAR_COLUMNS = ["closeness_eg", "gini_pred", "mean_target_degree",
                  "mean_path_length", "n_unique_new_edges"]


def load(plots_dir, dataset, subdir):
    spectrum = pd.read_csv(os.path.join(plots_dir, subdir,
                                        f"distance_spectrum_{dataset}.csv"))
    stats = pd.read_csv(os.path.join(plots_dir, subdir,
                                     f"path_stats_{dataset}.csv"))
    return spectrum, stats


def summarize(spectrum, stats, dataset, label):
    """Mean and sample SD over reruns, one row per probe arm."""
    protocol = ("split-resampled" if spectrum["split_seed"].nunique() > 1
                else "fixed-split")
    rows = []
    for arm in stats["method"].unique():
        arm_spectrum = spectrum.loc[spectrum["method"] == arm]
        arm_stats = stats.loc[stats["method"] == arm]
        row = {"dataset": dataset, "label": label, "arm": arm,
               "protocol": protocol,
               "n_runs": int(arm_stats["run"].nunique())}
        for column in DISTANCE_COLUMNS:
            row[f"{column} %"] = 100.0 * arm_spectrum[column].mean()
            row[f"{column} % sd"] = 100.0 * arm_spectrum[column].std(ddof=1)
        for column in SCALAR_COLUMNS:
            row[column] = arm_stats[column].mean()
            row[f"{column} sd"] = arm_stats[column].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots_dir", default="analysis/plots",
                        help="root holding the probe subdirectories")
    parser.add_argument("--out_dir", default=None,
                        help="if set, write probe_diagnostics.csv here")
    args = parser.parse_args()

    frames = []
    for dataset, label, subdir in PROBES:
        spectrum, stats = load(args.plots_dir, dataset, subdir)
        frames.append(summarize(spectrum, stats, dataset, label))
    summary = pd.concat(frames, ignore_index=True)

    display = ["arm", "n_runs", "d=2 %", "d=3 %", "d=4 %", "d>=5 %",
               "mean_target_degree", "gini_pred", "closeness_eg"]
    for label, block in summary.groupby("label", sort=False):
        print(f"\n=== {label} ({block['protocol'].iloc[0]}) ===")
        print(block[display].round(4).to_string(index=False))

    # The three numbers Section 7.5 quotes for the saturation argument.
    ego = summary.loc[summary["dataset"] == "ego-facebook"].set_index("arm")
    if {"+RA", "+RA+boost"} <= set(ego.index):
        control, boosted = ego.loc["+RA"], ego.loc["+RA+boost"]
        print("\n=== Section 7.5, Ego-FB saturation ===")
        print(f"distance-two share : {control['d=2 %']:.1f}% -> "
              f"{boosted['d=2 %']:.1f}%")
        print(f"mean target degree : {control['mean_target_degree']:.1f} -> "
              f"{boosted['mean_target_degree']:.1f}")
        print(f"Closeness EG       : {control['closeness_eg']:.2f} -> "
              f"{boosted['closeness_eg']:.2f} "
              f"({boosted['closeness_eg'] - control['closeness_eg']:+.1f} points)")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(args.out_dir, "probe_diagnostics.csv")
        summary.to_csv(path, index=False)
        print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
