"""
Fold the penalty-reduction diagnostic into the reproduction ablation tables.

Reads the JSONs written by jobs/penalty_reduction_ablation.job
(results_penalty_reduction/<ds>/{ci_in,cafe_in}/*_pen-{mean,sum}_*.json) and
puts them next to the corresponding rows of results_repro/tables, so the
'sum' reduction can be compared against its own 'mean' control and against the
published CI-In / CAFE-In rows.

WHAT COMPARES TO WHAT
  headline comparison  : sum arm vs mean arm, INSIDE results_penalty_reduction.
                         Both arms were submitted together with
                         --reset_seed_per_model True, so run i of one arm and
                         run i of the other share an initialization and the
                         reduction is the only difference.
  context only         : the results_repro CI-In / CAFE-In row. It was produced
                         under the historical RNG protocol and, for CAFE-In,
                         with (beta, lambda_mob) re-selected over a full grid,
                         while this diagnostic pins them. A gap against the
                         repro row therefore does NOT isolate the reduction --
                         only the sum-vs-mean gap does.

ROWS EMITTED PER (dataset, method, arm)
  lam=0     the built-in control: the loss is identical in both arms at
            lambda_fair = 0, so the sum-vs-mean gap there is pure training
            noise and calibrates how large a gap means nothing.
  ASF       the config the published selection procedure
            (generate_final_tables.select_best_config_pareto_asf) picks from
            that arm's own lambda grid.
  matched   the arm evaluated at the exact lambda_fair the results_repro row
            selected, when that lambda is in the grid. This is the only
            apples-to-apples row against the repro number.

PARTIAL RUNS
  Arms still on the queue are read from the runner's *.partial.json checkpoint
  and reconstructed from individual_runs (test value = element 1 of each metric
  pair). Those rows carry partial=True and n_runs < the full 10; they are a
  preview, not a result, and the paired arms are then unbalanced.

No file under results_repro/tables that repro_make_tables.job regenerates is
modified; every output here has a _with_pen / pen_reduction_ prefix of its own.
Bold/underline marks in the augmented tables are the ones the published
pipeline already resolved over the published rows -- the appended penalty rows
are deliberately left unmarked so they cannot silently change a paper table.

Usage:
    python add_penalty_reduction_to_tables.py
    python add_penalty_reduction_to_tables.py --datasets oklahoma unc
    python add_penalty_reduction_to_tables.py --no_partial
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_final_tables as gft  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# (directory under results_penalty_reduction/<ds>/, table row name)
PEN_METHOD_DIRS = {"ci_in": "CI-In", "cafe_in": "CAFE-In"}

METRICS = [
    ("Precision@10", "Prec@10", 4),
    ("Equality_Gain@10", "Deg EG", 2),
    ("Pagerank_EG@10", "PR EG", 2),
    ("Closeness_EG@10", "Cl. EG", 2),
]

# Closeness EG switches to NetworKit ApproxCloseness with unseeded pivots above
# EXACT_THRESHOLD = 5000 nodes (metrics/centrality_equality_gain.py), so it is
# not deterministic for these and must not drive a sum-vs-mean read.
APPROX_CLOSENESS_DATASETS = {"pokec_n", "brightkite"}

_RE_CAFE = re.compile(r"^lambda_fair_([0-9.eE+-]+)_beta_([0-9.eE+-]+)_lam_([0-9.eE+-]+)$")
_RE_CI = re.compile(r"^lambda_([0-9.eE+-]+)$")


def parse_lambda_fair(key):
    """lambda_fair out of either runner's config key, or NaN."""
    m = _RE_CAFE.match(key) or _RE_CI.match(key)
    return float(m.group(1)) if m else np.nan


# ---------------------------------------------------------------------------
# Loading the penalty-reduction tree
# ---------------------------------------------------------------------------

def _stats_from_individual_runs(data):
    """
    Rebuild {config_key: {metric: {mean, std, values}}} from a *.partial.json
    checkpoint, which carries individual_runs but no lambda_results yet.
    Each metric is stored as [<unused>, test_value]; index 1 is the test value
    that lambda_results['statistics'] aggregates.
    """
    per_config = {}
    for run in data.get("individual_runs", []):
        if not isinstance(run, dict):
            continue
        for cfg, metrics in run.items():
            if cfg == "baseline" or not isinstance(metrics, dict):
                continue
            for metric, pair in metrics.items():
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    per_config.setdefault(cfg, {}).setdefault(metric, []).append(pair[1])

    out = {}
    for cfg, metrics in per_config.items():
        out[cfg] = {
            m: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "values": v}
            for m, v in metrics.items()
        }
    return out


def load_pen_results(pen_dir, use_partial=True):
    """
    One row per (dataset, method, reduction, config) over the whole
    penalty-reduction tree. A finished JSON always wins over the .partial.json
    checkpoint the same run left behind.
    """
    rows = []
    for method_dir, method_name in PEN_METHOD_DIRS.items():
        for fpath in sorted(glob.glob(os.path.join(pen_dir, "*", method_dir, "*.json"))):
            partial = fpath.endswith(".partial.json")
            if partial:
                if not use_partial or os.path.exists(fpath[: -len(".partial.json")]):
                    continue  # superseded by the finished file, or not wanted

            with open(fpath) as f:
                data = json.load(f)

            dataset = os.path.basename(os.path.dirname(os.path.dirname(fpath)))
            info = data.get("experiment_info", {})
            reduction = info.get("penalty_reduction")
            if reduction is None:
                # partial checkpoints predate experiment_info; the arm is in
                # the filename the job template writes (..._pen-<red>_<ts>.json)
                m = re.search(r"_pen-(mean|sum)_", os.path.basename(fpath))
                reduction = m.group(1) if m else "unknown"

            stats_by_cfg = (
                _stats_from_individual_runs(data) if partial
                else {k: v.get("statistics", v) for k, v in data.get("lambda_results", {}).items()}
            )

            for cfg, stats in stats_by_cfg.items():
                if not isinstance(stats, dict):
                    continue
                vals = stats.get("Precision@10", {})
                row = {
                    "Dataset": dataset,
                    "Method_Group": method_name,
                    "Reduction": reduction,
                    "Config_Key": cfg,
                    "lambda_fair": parse_lambda_fair(cfg),
                    "n_runs": len(vals.get("values", [])) if isinstance(vals, dict) else 0,
                    "partial": partial,
                    "source": os.path.relpath(fpath, REPO),
                    "Has_Config": True,
                }
                for metric, _, _ in METRICS:
                    mean_val, std_val = gft.safe_get_stat(stats, metric)
                    row[f"{metric}_Mean"] = mean_val
                    row[f"{metric}_Std"] = std_val
                    entry = stats.get(metric)
                    row[f"{metric}_Values"] = (
                        list(entry.get("values", [])) if isinstance(entry, dict) else []
                    )
                rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reference rows from the published reproduction tables
# ---------------------------------------------------------------------------

def load_repro_rows(tables_dir):
    """
    The CI-In / CAFE-In rows already written by repro_make_tables.job, keyed by
    (dataset, method). Oklahoma lives in table3, everything else in table6.
    """
    frames = []
    t3 = os.path.join(tables_dir, "table3_ablation_oklahoma.csv")
    if os.path.exists(t3):
        df = pd.read_csv(t3)
        df["dataset"] = "oklahoma"
        frames.append(df)
    t6 = os.path.join(tables_dir, "table6_ablation_all.csv")
    if os.path.exists(t6):
        frames.append(pd.read_csv(t6))
    if not frames:
        return {}

    df = pd.concat(frames, ignore_index=True)
    return {
        (r["dataset"], r["method"]): r
        for _, r in df.iterrows()
        if r["method"] in PEN_METHOD_DIRS.values()
    }


# ---------------------------------------------------------------------------
# Row selection
# ---------------------------------------------------------------------------

def pick_rows(grid, repro_row):
    """
    (label, row) pairs for one (dataset, method, arm): the lambda = 0 control,
    the ASF pick over that arm's grid, and the config the repro row selected.
    """
    picked = []

    control = grid[grid["lambda_fair"] == 0.0]
    if not control.empty:
        picked.append(("lam=0", control.iloc[0]))

    asf = gft.select_best_config_pareto_asf(grid)
    if not asf.empty:
        picked.append(("ASF", asf.iloc[0]))

    if repro_row is not None:
        target = parse_lambda_fair(str(repro_row.get("config", "")))
        if not np.isnan(target):
            matched = grid[grid["lambda_fair"] == target]
            if not matched.empty:
                picked.append((f"matched(lam={target:g})", matched.iloc[0]))

    return picked


def fmt(mean, std, ndec):
    if pd.isna(mean):
        return "--"
    return f"{mean:.{ndec}f} $\\pm$ {std:.{ndec}f}" if not pd.isna(std) else f"{mean:.{ndec}f}"


def build_pen_table_rows(pen_df, repro_rows, datasets):
    """
    Table-shaped rows for the penalty arms, in the column schema of
    table3/table6 plus the bookkeeping columns the diagnostic needs.
    """
    out = []
    for dataset in datasets:
        for method in PEN_METHOD_DIRS.values():
            repro_row = repro_rows.get((dataset, method))
            for reduction in ("mean", "sum"):
                grid = pen_df[
                    (pen_df["Dataset"] == dataset)
                    & (pen_df["Method_Group"] == method)
                    & (pen_df["Reduction"] == reduction)
                ]
                if grid.empty:
                    continue

                for label, r in pick_rows(grid, repro_row):
                    cells = {
                        short: fmt(r[f"{metric}_Mean"], r[f"{metric}_Std"], ndec)
                        for metric, short, ndec in METRICS
                    }
                    out.append({
                        "dataset": dataset,
                        "category": "Penalty-reduction diag.",
                        "method": f"{method} [pen={reduction}, {label}]",
                        "latex_label": f"{method} (pen={reduction}, {label})",
                        "config": r["Config_Key"],
                        "dprec": "--",
                        **cells,
                        **{f"{metric}_Mean": r[f"{metric}_Mean"] for metric, _, _ in METRICS},
                        **{f"{metric}_Std": r[f"{metric}_Std"] for metric, _, _ in METRICS},
                        "pen_reduction": reduction,
                        "pen_row_kind": label,
                        "lambda_fair": r["lambda_fair"],
                        "n_runs": r["n_runs"],
                        "partial": r["partial"],
                        "source": r["source"],
                    })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# sum - mean deltas at matched configs
# ---------------------------------------------------------------------------

def build_delta_table(pen_df, datasets):
    """
    sum minus mean at every lambda both arms reached, per (dataset, method).

    PAIRED on run index: both arms ran with --reset_seed_per_model True, so run
    i of each arm was built from seed_base + i. The difference is therefore
    taken run by run over the first min(n_mean, n_sum) runs -- which matters
    while an arm is still on the queue, since a mean-of-means over unbalanced n
    would show a spurious gap even at lambda_fair = 0, where the two losses are
    identical by construction.

    Descriptive only: the reported spread is the SD of the paired differences,
    with no CI and no p-value. The Holm-corrected DiD inference against each
    arm's own lambda = 0 control is analysis/compare_penalty_reduction.py's job.
    """
    rows = []
    for dataset in datasets:
        approx_cl = dataset in APPROX_CLOSENESS_DATASETS
        for method in PEN_METHOD_DIRS.values():
            sub = pen_df[(pen_df["Dataset"] == dataset) & (pen_df["Method_Group"] == method)]
            mean_arm = sub[sub["Reduction"] == "mean"].set_index("lambda_fair")
            sum_arm = sub[sub["Reduction"] == "sum"].set_index("lambda_fair")
            for lam in sorted(set(mean_arm.index) & set(sum_arm.index)):
                m, s = mean_arm.loc[lam], sum_arm.loc[lam]
                n_paired = min(int(m["n_runs"]), int(s["n_runs"]))
                row = {
                    "dataset": dataset,
                    "method": method,
                    "lambda_fair": lam,
                    "n_runs_mean": m["n_runs"],
                    "n_runs_sum": s["n_runs"],
                    "n_paired": n_paired,
                    "partial": bool(m["partial"] or s["partial"]),
                    "balanced": m["n_runs"] == s["n_runs"],
                }
                for metric, short, _ in METRICS:
                    mv_all = np.asarray(m[f"{metric}_Values"], dtype=float)[:n_paired]
                    sv_all = np.asarray(s[f"{metric}_Values"], dtype=float)[:n_paired]
                    usable = approx_cl is False or metric != "Closeness_EG@10"
                    if n_paired == 0 or mv_all.size != n_paired or not usable:
                        # unseeded approximate closeness, or nothing to pair on
                        mv = sv = d = sd = np.nan
                    else:
                        mv, sv = mv_all.mean(), sv_all.mean()
                        diffs = sv_all - mv_all
                        d = diffs.mean()
                        sd = diffs.std(ddof=1) if n_paired > 1 else np.nan
                    row[f"{short} mean-arm"] = mv
                    row[f"{short} sum-arm"] = sv
                    row[f"{short} delta"] = d
                    row[f"{short} delta_sd"] = sd
                    row[f"{short} delta%"] = d / mv * 100.0 if mv else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Augmented ablation tables
# ---------------------------------------------------------------------------

def write_augmented_tables(pen_table, tables_dir, out_dir):
    """table3/table6 with the penalty rows appended under each dataset block."""
    written = []

    t3 = os.path.join(tables_dir, "table3_ablation_oklahoma.csv")
    if os.path.exists(t3):
        base = pd.read_csv(t3)
        extra = pen_table[pen_table["dataset"] == "oklahoma"].drop(columns=["dataset"])
        if not extra.empty:
            path = os.path.join(out_dir, "table3_ablation_oklahoma_with_pen.csv")
            pd.concat([base, extra], ignore_index=True).to_csv(path, index=False)
            written.append(path)

    t6 = os.path.join(tables_dir, "table6_ablation_all.csv")
    if os.path.exists(t6):
        base = pd.read_csv(t6)
        blocks = []
        for dataset in base["dataset"].unique():
            blocks.append(base[base["dataset"] == dataset])
            extra = pen_table[pen_table["dataset"] == dataset]
            if not extra.empty:
                blocks.append(extra)
        # datasets only present in the diagnostic (e.g. oklahoma) stay out of
        # table6, which by construction excludes the main-paper dataset
        path = os.path.join(out_dir, "table6_ablation_all_with_pen.csv")
        pd.concat(blocks, ignore_index=True).to_csv(path, index=False)
        written.append(path)

    return written


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def _plain(cell):
    """LaTeX table cell -> something readable in a terminal column."""
    return (str(cell).replace("$\\pm$", "+-")
                     .replace("\\textbf{", "").replace("\\underline{", "")
                     .replace("}", ""))


def print_report(pen_table, delta_table, repro_rows, datasets):
    for dataset in datasets:
        block = pen_table[pen_table["dataset"] == dataset]
        if block.empty:
            continue
        print(f"\n{'=' * 78}\n{dataset}\n{'=' * 78}")
        if dataset in APPROX_CLOSENESS_DATASETS:
            print("NOTE: Closeness EG uses unseeded ApproxCloseness here -- "
                  "arm differences on that column are not interpretable.")

        for method in PEN_METHOD_DIRS.values():
            rows = block[block["method"].str.startswith(method + " [")]
            if rows.empty:
                continue
            print(f"\n--- {method} ---")
            print(f"{'row':<40}{'config':<38}" + "".join(f"{s:>18}" for _, s, _ in METRICS))

            repro = repro_rows.get((dataset, method))
            if repro is not None:
                cells = "".join(f"{_plain(repro[s]):>18}" for _, s, _ in METRICS)
                print(f"{'results_repro (context)':<40}{str(repro['config']):<38}{cells}")

            for _, r in rows.iterrows():
                tag = r["method"].split("[", 1)[1].rstrip("]")
                flag = f"  [PARTIAL n={r['n_runs']}]" if r["partial"] else ""
                cells = "".join(f"{_plain(r[s]):>18}" for _, s, _ in METRICS)
                print(f"{tag + flag:<40}{r['config']:<38}{cells}")

            d = delta_table[(delta_table["dataset"] == dataset) & (delta_table["method"] == method)]
            if not d.empty:
                print("\n  sum - mean, paired by run index (mean +- SD of the "
                      "per-run differences; descriptive, no CI):")
                print(f"  {'lambda':<10}" + "".join(f"{s + ' d':>24}" for _, s, _ in METRICS)
                      + "   n_paired")
                for _, r in d.iterrows():
                    cells = ""
                    for _, s, _ in METRICS:
                        val, sd = r[s + " delta"], r[s + " delta_sd"]
                        cells += f"{'--':>24}" if pd.isna(val) else f"{val:>15.4f} +-{sd:7.4f}"
                    warn = "  <- preview, arm unfinished" if r["partial"] else ""
                    print(f"  {r['lambda_fair']:<10g}{cells}   {int(r['n_paired'])}{warn}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pen_dir", default=os.path.join(REPO, "results_penalty_reduction"))
    ap.add_argument("--tables_dir", default=os.path.join(REPO, "results_repro", "tables"))
    ap.add_argument("--out_dir", default=None, help="defaults to --tables_dir")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="restrict to these datasets (default: all found)")
    ap.add_argument("--no_partial", action="store_true",
                    help="ignore *.partial.json checkpoints of still-running arms")
    args = ap.parse_args()

    out_dir = args.out_dir or args.tables_dir
    os.makedirs(out_dir, exist_ok=True)

    pen_df = load_pen_results(args.pen_dir, use_partial=not args.no_partial)
    if pen_df.empty:
        raise SystemExit(f"No penalty-reduction results found under {args.pen_dir}")

    datasets = args.datasets or sorted(pen_df["Dataset"].unique())
    pen_df = pen_df[pen_df["Dataset"].isin(datasets)]

    repro_rows = load_repro_rows(args.tables_dir)
    pen_table = build_pen_table_rows(pen_df, repro_rows, datasets)
    delta_table = build_delta_table(pen_df, datasets)

    grid_cols = ["Dataset", "Method_Group", "Reduction", "lambda_fair", "Config_Key",
                 "n_runs", "partial", "source"] + \
                [f"{m}_{s}" for m, _, _ in METRICS for s in ("Mean", "Std")]
    grid_path = os.path.join(out_dir, "pen_reduction_lambda_grid.csv")
    pen_df[grid_cols].sort_values(
        ["Dataset", "Method_Group", "Reduction", "lambda_fair"]
    ).to_csv(grid_path, index=False)

    rows_path = os.path.join(out_dir, "pen_reduction_rows.csv")
    pen_table.to_csv(rows_path, index=False)

    delta_path = os.path.join(out_dir, "pen_reduction_sum_minus_mean.csv")
    delta_table.to_csv(delta_path, index=False)

    written = write_augmented_tables(pen_table, args.tables_dir, out_dir)

    print_report(pen_table, delta_table, repro_rows, datasets)

    print("\nWritten:")
    for p in [grid_path, rows_path, delta_path] + written:
        print("  " + os.path.relpath(p, REPO))

    incomplete = pen_df[pen_df["partial"]]
    if not incomplete.empty:
        print("\nSTILL RUNNING (rows above marked PARTIAL are previews, not results):")
        for src, g in incomplete.groupby("source"):
            print(f"  {src}  n_runs={int(g['n_runs'].max())}")


if __name__ == "__main__":
    main()
