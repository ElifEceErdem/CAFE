"""
Generate tab:closeness-strata for Appendix section sec:app-closeness.

The stratum-level account in Section 7.5 ("on Ego-FB the boost lowers mean
predicted closeness in every coreness quintile, most in the peripheral and
middle strata ... on Oklahoma and UNC the periphery instead gains slightly")
was previously uncited: the numbers lived only in the probe CSVs.  This
script renders them.

The reported quantity is the mobility boost's effect on the fused scorer,

    mean_closeness_pred(+RA+boost) - mean_closeness_pred(+RA),

per coreness quintile, paired within each rerun and then averaged.  The
+RA+boost arm is lambda_mob = 0.5 on all three networks, which is the
deployed CAFE-In operating point on Oklahoma and UNC and the only arm the
three probes share -- so the cross-network sign reversal is a like-for-like
contrast.  Panel (b) adds Ego-FB's remaining doses, which are only available
there.

Sources (nothing is retrained here):
    analysis/plots/closeness_mechanism/closeness_by_bin_{oklahoma,unc}.csv
    analysis/plots/closeness_mechanism_paired/closeness_by_bin_ego-facebook.csv
    ... and the sibling path_stats_{ds}.csv for Gini and Closeness EG.

Note on binning: the Oklahoma/UNC inputs are the regenerated float64,
closed-final-interval bins (see the "Stratified binning: bug scope"
paragraph of the appendix), NOT the closeness_mechanism_prebinfix/ copies.

Usage:
    python generate_closeness_stratum_table.py
"""

import argparse
import os

import numpy as np
import pandas as pd

# dataset -> (label, plots subdirectory, file stem)
PROBES = [
    ("ego-facebook", "Ego-FB", "closeness_mechanism_paired"),
    ("oklahoma", "Oklahoma", "closeness_mechanism"),
    ("unc", "UNC", "closeness_mechanism"),
]

CONTROL = "+RA"
DEPLOYED = "+RA+boost"          # lambda_mob = 0.5
EGO_DOSES = [("+boost(lam=0.1)", "0.1"), ("+boost(lam=0.3)", "0.3"),
             ("+RA+boost", "0.5"), ("+boost(lam=1)", "1.0")]
N_BINS = 5


def load(plots_dir, dataset, subdir):
    bins = pd.read_csv(os.path.join(plots_dir, subdir,
                                    f"closeness_by_bin_{dataset}.csv"))
    stats = pd.read_csv(os.path.join(plots_dir, subdir,
                                     f"path_stats_{dataset}.csv"))
    return bins, stats


def per_bin_effect(bins, arm, control=CONTROL):
    """Paired within-run (arm - control) mean closeness, per bin."""
    wide = bins.pivot_table(index=["run", "bin"], columns="method",
                            values="mean_closeness_pred")
    diff = (wide[arm] - wide[control]).unstack("bin")
    return diff.mean(), diff.std(ddof=1), len(diff)


def scalar_effect(stats, arm, column, control=CONTROL):
    wide = stats.pivot_table(index="run", columns="method", values=column)
    return wide[control].mean(), wide[arm].mean()


def fmt_effect(mean, sd):
    return f"${mean:+.4f}$" if np.isnan(sd) else f"${mean:+.4f}$"


def build(plots_dir):
    # ---------------- panel (a): deployed arm on all three networks --------
    lines_a = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
               r" & \multicolumn{5}{c}{$\Delta$ mean closeness by coreness quintile}"
               r" & \multicolumn{2}{c}{Graph level} \\",
               r"\cmidrule(lr){2-6}\cmidrule(lr){7-8}",
               r"Dataset & Q1 (periph.) & Q2 & Q3 & Q4 & Q5 (core)"
               r" & Gini & Cl.\ EG@10 \\", r"\midrule"]
    meta = {}
    csv_rows = []
    for dataset, label, subdir in PROBES:
        bins, stats = load(plots_dir, dataset, subdir)
        mean, sd, n = per_bin_effect(bins, DEPLOYED)
        g_ctrl, g_arm = scalar_effect(stats, DEPLOYED, "gini_pred")
        e_ctrl, e_arm = scalar_effect(stats, DEPLOYED, "closeness_eg")
        cells = [label] + [f"${mean[b]:+.4f}$" for b in range(N_BINS)]
        cells.append(f"${g_ctrl:.4f} \\to {g_arm:.4f}$")
        cells.append(f"${e_ctrl:.2f} \\to {e_arm:.2f}$")
        lines_a.append(" & ".join(cells) + r" \\")
        meta[dataset] = dict(
            n_runs=n,
            ranges=bins[["bin", "coreness_range", "n_nodes"]]
            .drop_duplicates("bin").sort_values("bin"))
        for b in range(N_BINS):
            csv_rows.append(dict(dataset=dataset, arm=DEPLOYED, lambda_mob=0.5,
                                 bin=b, delta_mean=mean[b], delta_sd=sd[b],
                                 n_runs=n, gini_control=g_ctrl, gini_arm=g_arm,
                                 eg_control=e_ctrl, eg_arm=e_arm))
    lines_a += [r"\bottomrule", r"\end{tabular}"]

    # ---------------- panel (b): Ego-FB dose curve -------------------------
    bins, stats = load(plots_dir, "ego-facebook", "closeness_mechanism_paired")
    lines_b = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
               r" & \multicolumn{5}{c}{$\Delta$ mean closeness by coreness quintile}"
               r" & \multicolumn{2}{c}{Graph level} \\",
               r"\cmidrule(lr){2-6}\cmidrule(lr){7-8}",
               r"$\lambda_{\mathrm{mob}}$ & Q1 (periph.) & Q2 & Q3 & Q4 & Q5 (core)"
               r" & Gini & Cl.\ EG@10 \\", r"\midrule"]
    for arm, dose in EGO_DOSES:
        mean, sd, n = per_bin_effect(bins, arm)
        g_ctrl, g_arm = scalar_effect(stats, arm, "gini_pred")
        e_ctrl, e_arm = scalar_effect(stats, arm, "closeness_eg")
        cells = [dose] + [f"${mean[b]:+.4f}$" for b in range(N_BINS)]
        cells.append(f"${g_arm:.4f}$")
        cells.append(f"${e_arm:.2f}$")
        lines_b.append(" & ".join(cells) + r" \\")
        for b in range(N_BINS):
            csv_rows.append(dict(dataset="ego-facebook", arm=arm,
                                 lambda_mob=float(dose), bin=b,
                                 delta_mean=mean[b], delta_sd=sd[b], n_runs=n,
                                 gini_control=g_ctrl, gini_arm=g_arm,
                                 eg_control=e_ctrl, eg_arm=e_arm))
    g_ctrl, _ = scalar_effect(stats, DEPLOYED, "gini_pred")
    e_ctrl, _ = scalar_effect(stats, DEPLOYED, "closeness_eg")
    lines_b.append(r"\midrule")
    lines_b.append(r"\emph{control} ($+$RA) & --- & --- & --- & --- & --- "
                   f"& ${g_ctrl:.4f}$ & ${e_ctrl:.2f}$" + r" \\")
    lines_b += [r"\bottomrule", r"\end{tabular}"]

    return "\n".join(lines_a), "\n".join(lines_b), meta, pd.DataFrame(csv_rows)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots_dir", default=os.path.join(here, "plots"))
    ap.add_argument("--out_dir",
                    default=os.path.join(root, "results_repro", "tables"))
    args = ap.parse_args()

    panel_a, panel_b, meta, csv = build(args.plots_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    for name, body in [("closeness_strata_table.tex", panel_a),
                       ("closeness_strata_dose_table.tex", panel_b)]:
        with open(os.path.join(args.out_dir, name), "w") as f:
            f.write(body + "\n")
        print("wrote", os.path.join(args.out_dir, name))
    csv.to_csv(os.path.join(args.out_dir, "closeness_strata.csv"), index=False)
    print("wrote", os.path.join(args.out_dir, "closeness_strata.csv"))

    print("\nquintile definitions (for the caption):")
    for dataset, info in meta.items():
        rng = ", ".join(f"{r.coreness_range} ({r.n_nodes})"
                        for r in info["ranges"].itertuples())
        print(f"  {dataset}: n_runs={info['n_runs']}; {rng}")


if __name__ == "__main__":
    main()
