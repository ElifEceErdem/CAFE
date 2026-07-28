"""
Generate the two RQ5 appendix tables referenced from Section 7.4:

  (1) tab:demographic-protocol -- DeltaSP@10 / DeltaEO@10 for the backbone,
      the fusion-only control, and the two CAFE variants under BOTH
      protocols: the fixed-split headline protocol (one split, n=10
      training repetitions) and the split-resampled paired protocol (n=10
      seeds, each drawing its own split).  The right-hand block gives the
      paired split-resampled contrasts against the backbone and against
      fusion, Holm-corrected within each (measure x reference) family.

  (2) tab:target-exposure-audit -- mean per-node incoming exposure by
      sensitive group and target-coreness tier, and its change relative to
      HL-GNN, on the three audited networks.  This is the group-split
      version of Table tab:rq3-localization panel (b), which pools groups.

Sources (all pre-existing; nothing is retrained here):
  fixed split       results_repro/{ds}/cafe_p/*_baseline_*.json   (HL-GNN)
                    results_repro/tables/table5_demographic_fairness.csv
  split-resampled   results_repro/{ds}/significance/per_seed_metrics_{ds}.csv
  exposure audit    analysis/plots/coreness_by_group/
                        exposure_by_group_and_tier_{ds}.csv

Usage:
    python generate_demographic_appendix_tables.py
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

DATASETS = ["oklahoma", "pokec_n", "unc"]
DATASET_LABELS = {"oklahoma": "Oklahoma", "pokec_n": "Pokec-n", "unc": "UNC"}
MEASURES = ["Delta_SP@10", "Delta_EO@10"]

# per-seed CSV method name -> paper label
METHODS = ["HL-GNN", "Fusion", "CAFE-In", "CAFE-Pre"]
METHOD_LABELS = {"HL-GNN": "HL-GNN", "Fusion": "HL-GNN+RA",
                 "CAFE-In": "CAFE-In", "CAFE-Pre": "CAFE-Pre"}
# table5 latex_label -> per-seed method name
T5_TO_METHOD = {
    "HL-GNN+RA (CAFE$^{-P{,}-B}$)": "Fusion",
    "CAFE-In": "CAFE-In",
    "CAFE-Pre": "CAFE-Pre",
}

TIERS = ["periphery", "mid", "core"]
TIER_LABELS = {"periphery": "Periphery", "mid": "Middle", "core": "Core"}


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    order = p.argsort()
    k = len(p)
    adj = np.maximum.accumulate([(k - i) * p[order][i] for i in range(k)])
    out = np.empty(k)
    out[order] = np.clip(adj, 0, 1)
    return out


# ----------------------------------------------------------------- loading
def load_fixed_split(results_dir, tables_dir):
    """(dataset, method, measure) -> (mean, std) on the fixed split."""
    out = {}
    for ds in DATASETS:
        pat = os.path.join(results_dir, ds, "cafe_p", "*_baseline_*.json")
        paths = [p for p in glob.glob(pat) if not p.endswith(".partial.json")]
        if not paths:
            raise FileNotFoundError(f"no HL-GNN baseline json for {ds}: {pat}")
        with open(paths[0]) as f:
            stat = json.load(f)["lambda_results"]["beta_0.0_lambda_0.0"]["statistics"]
        for m in MEASURES:
            out[(ds, "HL-GNN", m)] = (stat[m]["mean"], stat[m]["std"])

    t5 = pd.read_csv(os.path.join(tables_dir, "table5_demographic_fairness.csv"))
    for _, r in t5.iterrows():
        meth = T5_TO_METHOD.get(r["latex_label"])
        if meth is None:
            continue
        for m in MEASURES:
            out[(r["dataset"], meth, m)] = (r[f"{m}_Mean"], r[f"{m}_Std"])
    return out


def load_resampled(results_dir):
    """(dataset, measure) -> DataFrame [seed x method] under the paired protocol."""
    out = {}
    for ds in DATASETS:
        path = os.path.join(results_dir, ds, "significance",
                            f"per_seed_metrics_{ds}.csv")
        d = pd.read_csv(path)
        for m in MEASURES:
            out[(ds, m)] = d[d["metric"] == m].pivot(
                index="seed", columns="method", values="value")
    return out


# ------------------------------------------------------- (1) protocol table
def paired_contrasts(resampled):
    recs = []
    for m in MEASURES:
        for ref in ["HL-GNN", "Fusion"]:
            for ds in DATASETS:
                sub = resampled[(ds, m)]
                for variant in ["CAFE-In", "CAFE-Pre"]:
                    diff = sub[variant] - sub[ref]
                    n = len(diff)
                    t, p = stats.ttest_rel(sub[variant], sub[ref])
                    ci = stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n)
                    recs.append(dict(measure=m, reference=ref, dataset=ds,
                                     variant=variant, n=n, mean_diff=diff.mean(),
                                     ci95_halfwidth=ci, t=t, p_raw=p))
    df = pd.DataFrame(recs)
    df["p_holm"] = np.nan
    for _, g in df.groupby(["measure", "reference"]):
        df.loc[g.index, "p_holm"] = holm(g["p_raw"].values)
    return df


def fmt_pair(mean, std, scale, digits):
    return f"${mean / scale:.{digits}f} \\pm {std / scale:.{digits}f}$"


def fmt_diff(mean, ci, p, scale, digits):
    return (f"${mean / scale:+.{digits}f} \\pm {ci / scale:.{digits}f}$"
            + stars(p))


def build_protocol_table(fixed, resampled, contrasts):
    """LaTeX body for tab:demographic-protocol."""
    lines = []
    for m, scale, digits, unit in [
            ("Delta_SP@10", 1e-3, 3, r"$\Delta$SP@10 ($\times 10^{-3}$)"),
            ("Delta_EO@10", 1e-3, 2, r"$\Delta$EO@10 ($\times 10^{-3}$)")]:
        lines.append(r"\subcaption*{(%s) %s}" % ("a" if "SP" in m else "b", unit))
        lines.append(r"\begin{tabular}{llrrrr}")
        lines.append(r"\toprule")
        lines.append(r" & & \multicolumn{2}{c}{Absolute value} "
                     r"& \multicolumn{2}{c}{Split-resampled paired difference} \\")
        lines.append(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}")
        lines.append(r"Dataset & Method & Fixed split & Split-resampled "
                     r"& vs.\ HL-GNN & vs.\ HL-GNN+RA \\")
        lines.append(r"\midrule")
        for di, ds in enumerate(DATASETS):
            if di:
                lines.append(r"\addlinespace[2pt]")
            for mi, meth in enumerate(METHODS):
                fm, fs = fixed[(ds, meth, m)]
                sub = resampled[(ds, m)][meth]
                cells = [DATASET_LABELS[ds] if mi == 0 else "",
                         METHOD_LABELS[meth],
                         fmt_pair(fm, fs, scale, digits),
                         fmt_pair(sub.mean(), sub.std(ddof=1), scale, digits)]
                for ref in ["HL-GNN", "Fusion"]:
                    if meth in ("HL-GNN", "Fusion"):
                        cells.append("---")
                        continue
                    row = contrasts[(contrasts.measure == m)
                                    & (contrasts.reference == ref)
                                    & (contrasts.dataset == ds)
                                    & (contrasts.variant == meth)].iloc[0]
                    cells.append(fmt_diff(row.mean_diff, row.ci95_halfwidth,
                                          row.p_holm, scale, digits))
                lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        if "SP" in m:
            lines.append(r"\vspace{6pt}")
            lines.append("")
    return "\n".join(lines)


# ------------------------------------------------- (2) target-exposure table
def build_exposure_table(analysis_dir):
    frames = []
    for ds in DATASETS:
        path = os.path.join(analysis_dir, "plots", "coreness_by_group",
                            f"exposure_by_group_and_tier_{ds}.csv")
        frames.append(pd.read_csv(path))
    exp = pd.concat(frames, ignore_index=True)

    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r" & & \multicolumn{3}{c}{Group $S_0$} "
             r"& \multicolumn{3}{c}{Group $S_1$} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r"Dataset & Method & Periphery & Middle & Core "
             r"& Periphery & Middle & Core \\", r"\midrule"]
    csv_rows = []
    for di, ds in enumerate(DATASETS):
        if di:
            lines.append(r"\addlinespace[2pt]")
        sub = exp[exp["dataset"] == ds]
        for mi, meth in enumerate(["HL-GNN", "HL-GNN+RA", "CAFE-In", "CAFE-Pre"]):
            cells = [DATASET_LABELS[ds] if mi == 0 else "", meth]
            for grp in [0, 1]:
                for tier in TIERS:
                    r = sub[(sub.method == meth) & (sub.group == grp)
                            & (sub.tier == tier)].iloc[0]
                    base = exp[(exp.dataset == ds) & (exp.method == "HL-GNN")
                               & (exp.group == grp) & (exp.tier == tier)
                               ].iloc[0]["exposure_mean"]
                    if meth == "HL-GNN":
                        cells.append(f"${r.exposure_mean * 1e3:.3f}$")
                    else:
                        pct = 100.0 * (r.exposure_mean / base - 1.0)
                        cells.append(f"${pct:+.1f}\\%$")
                    csv_rows.append(dict(
                        dataset=ds, method=meth, group=grp, tier=tier,
                        exposure_mean=r.exposure_mean,
                        exposure_std=r.exposure_std,
                        pct_change_vs_hlgnn=100.0 * (r.exposure_mean / base - 1.0),
                        delta_sp_mean=r.delta_sp_mean))
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    # companion DeltaSP column (one value per dataset x method)
    sp_lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
                r"Dataset & HL-GNN & HL-GNN+RA & CAFE-In & CAFE-Pre \\",
                r"\midrule"]
    for ds in DATASETS:
        sub = exp[exp["dataset"] == ds]
        cells = [DATASET_LABELS[ds]]
        for meth in ["HL-GNN", "HL-GNN+RA", "CAFE-In", "CAFE-Pre"]:
            v = sub[sub.method == meth]["delta_sp_mean"].iloc[0]
            s = sub[sub.method == meth]["delta_sp_std"].iloc[0]
            cells.append(f"${v * 1e3:.4f} \\pm {s * 1e3:.4f}$")
        sp_lines.append(" & ".join(cells) + r" \\")
    sp_lines += [r"\bottomrule", r"\end{tabular}"]

    return "\n".join(lines), "\n".join(sp_lines), pd.DataFrame(csv_rows)


# --------------------------------------- (3) demographic significance table
SIG_METRIC_LABELS = {
    "Precision@10": "Prec@10",
    "Equality_Gain@10": "Deg EG@10",
    "Pagerank_EG@10": "PR EG@10",
}


def build_significance_table(tables_dir):
    """Compact summary of the 162 CAFE-vs-baseline Welch contrasts.

    Each cell condenses the six CAFE-block rows against one baseline on one
    measure into the smallest observed advantage and the largest Holm-adjusted
    p-value, which is what the "all contrasts significant" claim rests on.
    """
    d = pd.read_csv(os.path.join(tables_dir, "table5_significance.csv"))
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r" & & \multicolumn{2}{c}{Prec@10} & \multicolumn{2}{c}{Deg EG@10}"
             r" & \multicolumn{2}{c}{PR EG@10} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
             r"Dataset & Baseline & Min $\Delta$ & Max $p_{\mathrm{Holm}}$"
             r" & Min $\Delta$ & Max $p_{\mathrm{Holm}}$"
             r" & Min $\Delta$ & Max $p_{\mathrm{Holm}}$ \\", r"\midrule"]
    for di, ds in enumerate(DATASETS):
        if di:
            lines.append(r"\addlinespace[2pt]")
        for bi, base in enumerate(["FairAdj", "LBA", "MORAL"]):
            cells = [DATASET_LABELS[ds] if bi == 0 else "", base]
            for metric in ["Precision@10", "Equality_Gain@10", "Pagerank_EG@10"]:
                sub = d[(d.dataset == ds) & (d.baseline == base)
                        & (d.metric == metric)]
                lo = sub["diff"].min()
                pmax = sub["p_holm"].max()
                digits = 4 if metric == "Precision@10" else 2
                cells.append(f"${lo:+.{digits}f}$")
                cells.append(f"${pmax:.1e}$".replace("e-0", r"\mathrm{e}{-}")
                             .replace("e-", r"\mathrm{e}{-}")
                             .replace("$", "$", 1))
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    n_total = len(d)
    n_sig = int((d["p_holm"] < 0.05).sum())
    return "\n".join(lines), n_total, n_sig


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=os.path.join(root, "results_repro"))
    ap.add_argument("--tables_dir",
                    default=os.path.join(root, "results_repro", "tables"))
    ap.add_argument("--analysis_dir", default=here)
    ap.add_argument("--out_dir",
                    default=os.path.join(root, "results_repro", "tables"))
    args = ap.parse_args()

    fixed = load_fixed_split(args.results_dir, args.tables_dir)
    resampled = load_resampled(args.results_dir)
    contrasts = paired_contrasts(resampled)

    proto_tex = build_protocol_table(fixed, resampled, contrasts)
    exp_tex, sp_tex, exp_csv = build_exposure_table(args.analysis_dir)
    sig_tex, n_total, n_sig = build_significance_table(args.tables_dir)
    print(f"demographic significance: {n_sig}/{n_total} contrasts Holm-significant")

    os.makedirs(args.out_dir, exist_ok=True)
    paths = {
        "demographic_protocol_table.tex": proto_tex,
        "target_exposure_audit_table.tex": exp_tex,
        "target_exposure_audit_deltasp.tex": sp_tex,
        "demographic_significance_summary.tex": sig_tex,
    }
    for name, body in paths.items():
        with open(os.path.join(args.out_dir, name), "w") as f:
            f.write(body + "\n")
        print("wrote", os.path.join(args.out_dir, name))

    contrasts.to_csv(os.path.join(args.out_dir, "demographic_protocol_contrasts.csv"),
                     index=False)
    exp_csv.to_csv(os.path.join(args.out_dir, "target_exposure_audit.csv"),
                   index=False)
    print("wrote contrast and exposure CSVs")


if __name__ == "__main__":
    main()
