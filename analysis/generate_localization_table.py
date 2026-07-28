"""Generate Table 6 (Section 7.2, RQ3): localization of CAFE-In beyond fusion.

Panel (a), source side -- from ``analysis/plots/stratified_cafe/``:
    * ``DeltaPrecision@10``: CAFE-In minus the matched-beta fusion control,
      in the LOWEST source-coreness bin, paired within rerun.
    * ``P_low / P_high``: the lowest/highest-bin mean Precision@10 ratio,
      under fusion and then under CAFE-In.

Panel (b), target side -- from ``analysis/plots/coreness_by_group/``:
    relative change in mean per-node incoming exposure within each
    target-coreness tercile, 100 * (Exp_CAFE-In / Exp_Fusion - 1), with the
    absolute difference in parentheses.

Both panels are descriptive; no per-stratum hypothesis test is run, and the
caption must say so.  The generator prints the repetition count it found per
dataset and panel -- put those numbers in the caption instead of an
unqualified "across reruns", because they differ (source side n=3 with
Pokec-n at n=1; target side n=10 on Oklahoma/UNC and n=3 on Pokec-n in the
published data).

Panel (b) pools the demographic groups and therefore needs no sensitive
attribute.  If a dataset is missing here, it is because the exposure runner
skipped it, not because the quantity is undefined -- rerun
``experiments/run_exposure_acceptance.py`` for it.

Usage:
    python generate_localization_table.py \
        --stratified_dir plots/stratified_cafe \
        --exposure_dir plots/coreness_by_group \
        --out_dir ../results_repro/tables
"""

import argparse
import os

import numpy as np
import pandas as pd

DATASET_ORDER = ["brightkite", "ego-facebook", "oklahoma", "pokec_n", "unc"]
DATASET_LABELS = {
    "brightkite": "Brightkite", "ego-facebook": "Ego-FB", "oklahoma": "Oklahoma",
    "pokec_n": "Pokec-n", "unc": "UNC",
}
TIERS = ["periphery", "mid", "core"]
TIER_LABELS = {"periphery": "Peripheral tercile", "mid": "Middle tercile",
               "core": "Core tercile"}
FUSION = "Fusion"
CAFE = "CAFE-In"


# ------------------------------------------------------------------ panel (a)
def panel_a(stratified_dir):
    rows = []
    for ds in DATASET_ORDER:
        path = os.path.join(stratified_dir, f"stratified_cafe_{ds}.csv")
        if not os.path.exists(path):
            print(f"  (a) {ds:14s} MISSING {path}")
            continue
        df = pd.read_csv(path, usecols=["model", "precision", "run", "coreness_bin"])
        lo, hi = df.coreness_bin.min(), df.coreness_bin.max()

        # per-run bin means, so the CAFE-minus-fusion difference is paired
        per_run = (df.groupby(["model", "run", "coreness_bin"])["precision"]
                   .mean().reset_index())
        low = per_run[per_run.coreness_bin == lo].pivot(
            index="run", columns="model", values="precision")
        high = per_run[per_run.coreness_bin == hi].pivot(
            index="run", columns="model", values="precision")
        if FUSION not in low.columns or CAFE not in low.columns:
            print(f"  (a) {ds:14s} needs both {FUSION} and {CAFE}; found "
                  f"{sorted(low.columns)}")
            continue

        diff = (low[CAFE] - low[FUSION]).dropna()
        n = len(diff)
        ratio_fusion = low[FUSION].mean() / high[FUSION].mean()
        ratio_cafe = low[CAFE].mean() / high[CAFE].mean()
        rows.append(dict(dataset=ds, n_runs=n,
                         delta_precision_mean=diff.mean(),
                         delta_precision_std=diff.std(ddof=1) if n > 1 else np.nan,
                         ratio_fusion=ratio_fusion, ratio_cafe=ratio_cafe,
                         low_bin=int(lo), high_bin=int(hi)))
        print(f"  (a) {ds:14s} n={n}  dPrec={diff.mean():+.4f}  "
              f"ratio {ratio_fusion:.3f} -> {ratio_cafe:.3f}")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ panel (b)
def panel_b(exposure_dir):
    rows = []
    for ds in DATASET_ORDER:
        path = os.path.join(exposure_dir, f"exposure_per_node_{ds}.csv")
        if not os.path.exists(path):
            print(f"  (b) {ds:14s} MISSING {path}")
            continue
        df = pd.read_csv(path, usecols=["run", "method", "tier", "exposure"])
        methods = set(df.method.unique())
        fusion_key = "HL-GNN+RA" if "HL-GNN+RA" in methods else FUSION
        if fusion_key not in methods or CAFE not in methods:
            print(f"  (b) {ds:14s} needs {CAFE} and a fusion row; found "
                  f"{sorted(methods)}")
            continue

        # pool groups: mean per-node exposure within tier, per run
        per_run = (df.groupby(["run", "method", "tier"])["exposure"]
                   .mean().reset_index())
        n_runs = per_run.run.nunique()
        for tier in TIERS:
            sub = per_run[per_run.tier == tier].pivot(
                index="run", columns="method", values="exposure")
            if fusion_key not in sub.columns or CAFE not in sub.columns:
                continue
            paired = sub[[CAFE, fusion_key]].dropna()
            absolute = (paired[CAFE] - paired[fusion_key])
            relative = 100.0 * (paired[CAFE] / paired[fusion_key] - 1.0)
            rows.append(dict(dataset=ds, tier=tier, n_runs=len(paired),
                             rel_mean=relative.mean(),
                             rel_std=relative.std(ddof=1) if len(paired) > 1 else np.nan,
                             abs_mean=absolute.mean(),
                             positive_runs=int((absolute > 0).sum()),
                             negative_runs=int((absolute < 0).sum())))
        got = [r for r in rows if r["dataset"] == ds]
        if got:
            print(f"  (b) {ds:14s} n={n_runs}  " + "  ".join(
                f"{r['tier']}={r['rel_mean']:+.1f}%" for r in got))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- LaTeX
def latex(a, b):
    out = [
        r"\begin{table}[t]",
        r"\caption{Localization of CAFE-In beyond the fusion control "
        r"(HL-GNN+RA at CAFE-In's $\beta$). \textbf{(a)} $\Delta$Precision@10 "
        r"is the CAFE-In-minus-fusion difference in the lowest coreness bin; "
        r"$P_{\text{low}}/P_{\text{high}}$ is the ratio of mean Precision@10 "
        r"in the lowest coreness bin to that in the highest, given under "
        r"fusion and then under CAFE-In. \textbf{(b)} Relative change in mean "
        r"per-node target exposure within each coreness tercile, "
        r"$100(\overline{\mathrm{Exp}}_{\text{CAFE-In}}/"
        r"\overline{\mathrm{Exp}}_{\text{Fusion}}-1)$; parenthesized values "
        r"are absolute differences. All values are descriptive; no "
        r"per-stratum tests were run. FILL IN the repetition counts printed "
        r"by the generator.}",
        r"\label{tab:rq3-localization}",
        r"\centering",
        r"\small",
        r"\subfloat[Lowest source-coreness bin]{",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Dataset & $\Delta$Precision@10 & "
        r"$P_{\text{low}}/P_{\text{high}}$: Fusion $\rightarrow$ CAFE-In \\",
        r"\midrule",
    ]
    for _, r in a.iterrows():
        std = ("" if pd.isna(r.delta_precision_std)
               else f" $\\pm$ {r.delta_precision_std:.4f}")
        out.append(f"{DATASET_LABELS[r.dataset]} & "
                   f"{r.delta_precision_mean:+.4f}{std} & "
                   f"{r.ratio_fusion:.3f} $\\rightarrow$ {r.ratio_cafe:.3f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}}", r"", r"\vspace{1em}", r"",
            r"\subfloat[Target-coreness exposure]{",
            r"\begin{tabular}{lccc}", r"\toprule",
            r"Dataset & " + " & ".join(TIER_LABELS[t] for t in TIERS) + r" \\",
            r"\midrule"]
    for ds in DATASET_ORDER:
        sub = b[b.dataset == ds]
        if sub.empty:
            continue
        cells = []
        for tier in TIERS:
            row = sub[sub.tier == tier]
            if row.empty:
                cells.append("--")
                continue
            row = row.iloc[0]
            cells.append(f"{row.rel_mean:+.1f}\\% ({row.abs_mean:+.2e})")
        out.append(f"{DATASET_LABELS[ds]} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stratified_dir", default="plots/stratified_cafe")
    ap.add_argument("--exposure_dir", default="plots/coreness_by_group")
    ap.add_argument("--out_dir", default="../results_repro/tables")
    args = ap.parse_args()

    print("panel (a) -- source side")
    a = panel_a(args.stratified_dir)
    print("\npanel (b) -- target side")
    b = panel_b(args.exposure_dir)
    if a.empty and b.empty:
        raise SystemExit("no inputs found for either panel")

    os.makedirs(args.out_dir, exist_ok=True)
    tex = os.path.join(args.out_dir, "table_localization.tex")
    with open(tex, "w") as fh:
        fh.write(latex(a, b))
    a.to_csv(os.path.join(args.out_dir, "table_localization_source.csv"), index=False)
    b.to_csv(os.path.join(args.out_dir, "table_localization_target.csv"), index=False)
    print(f"\n  wrote {tex}")
    print(f"  wrote {os.path.join(args.out_dir, 'table_localization_source.csv')}")
    print(f"  wrote {os.path.join(args.out_dir, 'table_localization_target.csv')}")

    missing = [ds for ds in DATASET_ORDER if ds not in set(b.dataset)] if not b.empty else DATASET_ORDER
    if missing:
        print(f"\n  panel (b) missing: {', '.join(missing)} -- rerun "
              f"experiments/run_exposure_acceptance.py on those datasets.")


if __name__ == "__main__":
    main()
