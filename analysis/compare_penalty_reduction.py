"""
Compare the in-processing coreness penalty under both batch reductions.

Consumes the JSONs written by jobs/penalty_reduction_ablation.job and answers
one question: does the reduction change (and hence lambda_fair) actually move
the metrics?

Background (ToRS_paper/ANSWERS-sections-5-6.md, issue B2): the base loss in
backbone/loss.py reduces with .sum() over the batch while the fairness penalty
reduces with .mean(), so the penalty's contribution is ~1/|E+_batch| of the
base loss and is expected to be inert. The 'sum' arm makes both reductions
match.

PRIMARY ESTIMANDS (both paired by run index; the runners seed each model as
seed_base + run_index, so run i of the 'sum' arm and run i of the 'mean' arm
share an initialization):

  1. Direct paired difference at fixed lambda
         D[s, lam] = Y_sum[s, lam] - Y_mean[s, lam]

  2. Difference-in-differences against each arm's own lambda = 0 control
         DiD[s, lam] = (Y_sum[s, lam] - Y_sum[s, 0])
                     - (Y_mean[s, lam] - Y_mean[s, 0])
     This removes any arm-level offset and isolates the reduction change.

Both are reported as mean +/- 95% t-interval with the Hedges-corrected paired
standardized effect g_z. An optional TOST equivalence test can support a
positive claim of "no practically relevant difference"; a large p-value on its
own CANNOT.

MULTIPLICITY. The full grid yields up to 3 datasets x 2 methods x 7 lambdas x 4
metrics = 168 paired comparisons. Scanning all of them and reporting whichever
exclude zero would manufacture false positives. Inferential output is therefore
produced ONLY for a PRESPECIFIED primary family, passed via --primary_lambda,
and Holm-corrected within it. Without that flag every cell is printed as
exploratory and no p-value is shown.

EQUIVALENCE MARGINS are per metric and must be given explicitly: Precision@10
lives on [0,1] while the EG measures are in percentage-point-like units of
magnitude 5-15, so a single scalar margin is meaningless. Margins must be fixed
BEFORE looking at the results.

The across-lambda test within each arm is retained as a SECONDARY summary and
uses Friedman (repeated measures), because all lambdas share the same run seeds
and are therefore paired, not independent.

CAVEAT handled automatically: Closeness EG uses NetworKit's randomized
ApproxCloseness when num_nodes > 5000 (metrics/centrality_equality_gain.py),
with unseeded pivots. For such datasets Closeness_EG is flagged and excluded
from the headline table. Ego-FB (4039), Oklahoma (3076) and UNC (3985) are
below the threshold and use exact closeness, so they are unaffected.

Usage:
    python compare_penalty_reduction.py \
        --results_dir ../results_penalty_reduction \
        --out penalty_reduction_comparison.csv
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats as sps

METRICS = ['Precision@10', 'Equality_Gain@10', 'Pagerank_EG@10', 'Closeness_EG@10']

# Datasets whose Closeness EG is computed by randomized approximation
# (num_nodes > EXACT_THRESHOLD = 5000 in metrics/centrality_equality_gain.py).
APPROX_CLOSENESS_DATASETS = {'pokec_n', 'brightkite'}
CLOSENESS_METRIC = 'Closeness_EG@10'

# run_coreness_inverse.py writes 'lambda_<v>'
# run_cafe.py writes 'lambda_fair_<v>_beta_<b>_lam_<m>'
_RE_CAFE = re.compile(r'^lambda_fair_([0-9.eE+-]+)_beta_([0-9.eE+-]+)_lam_([0-9.eE+-]+)$')
_RE_CI = re.compile(r'^lambda_([0-9.eE+-]+)$')


def parse_config_key(key):
    """Return (lambda_fair, beta, lambda_mob) or None if unrecognized."""
    m = _RE_CAFE.match(key)
    if m:
        return float(m.group(1)), float(m.group(2)), float(m.group(3))
    m = _RE_CI.match(key)
    if m:
        return float(m.group(1)), np.nan, np.nan
    return None


def load_cells(results_dir):
    """
    Return {(dataset, method, reduction, beta, lam_mob, lambda_fair):
            {metric: [per-run values]}}.

    Observations for the same configuration coming from several timestamped
    files (e.g. a re-run of the job) are CONCATENATED into one cell rather than
    being treated as distinct groups.
    """
    pattern = os.path.join(results_dir, '*', '*', '*.json')
    files = sorted(p for p in glob.glob(pattern) if not p.endswith('.partial.json'))
    if not files:
        raise SystemExit(f"No result JSONs matched {pattern}")
    print(f"Reading {len(files)} file(s) from {results_dir}")

    cells = defaultdict(lambda: defaultdict(list))
    provenance = defaultdict(set)

    for path in files:
        with open(path) as f:
            data = json.load(f)

        info = data.get('experiment_info', {})
        dataset = info.get('dataset', path.split(os.sep)[-3])
        method = path.split(os.sep)[-2]

        reduction = info.get('penalty_reduction')
        if reduction is None:
            m = re.search(r'pen-(mean|sum)', os.path.basename(path))
            reduction = m.group(1) if m else 'mean'

        for key, entry in (data.get('lambda_results') or {}).items():
            parsed = parse_config_key(key)
            if parsed is None:
                continue
            lam_fair, beta, lam_mob = parsed
            cell = (dataset, method, reduction, beta, lam_mob, lam_fair)
            stats = (entry or {}).get('statistics', {})
            for metric in METRICS:
                vals = (stats.get(metric) or {}).get('values') or []
                cells[cell][metric].extend(float(v) for v in vals)
            provenance[cell].add(os.path.basename(path))

    multi = {k: v for k, v in provenance.items() if len(v) > 1}
    if multi:
        print(f"\nNOTE: {len(multi)} configuration(s) appear in more than one file; "
              f"their runs were merged. Run indices are then only comparable "
              f"across arms if the files were produced by matched job "
              f"submissions. Example: {sorted(next(iter(multi.values())))}")
    return cells


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values (NaNs pass through)."""
    p = np.asarray(pvals, dtype=float)
    idx = np.where(~np.isnan(p))[0]
    adj = np.full_like(p, np.nan)
    if idx.size == 0:
        return adj
    sub = p[idx]
    order = np.argsort(sub)
    m = len(sub)
    running = 0.0
    out = np.empty(m)
    for rank, j in enumerate(order):
        running = max(running, (m - rank) * sub[j])
        out[j] = min(running, 1.0)
    adj[idx] = out
    return adj


def paired_stats(diffs):
    """
    Mean paired difference, 95% t-interval, and the Hedges-corrected paired
    standardized effect g_z = J * mean(d) / sd(d), J = 1 - 3/(4(n-1)-1).
    (This is g_z, not Cohen's d_z: d_z is the uncorrected ratio.)
    """
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    if n < 2:
        return dict(n=n, mean=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                    p=np.nan, gz=np.nan)
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    tcrit = sps.t.ppf(0.975, df=n - 1)
    if sd == 0:
        return dict(n=n, mean=d.mean(), ci_lo=d.mean(), ci_hi=d.mean(),
                    p=1.0 if d.mean() == 0 else 0.0, gz=np.nan)
    _, p = sps.ttest_1samp(d, 0.0)
    J = 1.0 - 3.0 / (4.0 * (n - 1) - 1.0) if n > 2 else 1.0
    return dict(n=n, mean=d.mean(), ci_lo=d.mean() - tcrit * se,
                ci_hi=d.mean() + tcrit * se, p=p, gz=J * d.mean() / sd)


def tost(diffs, margin):
    """Two one-sided tests for equivalence within +/- margin. Returns p."""
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    if n < 2 or margin is None or not np.isfinite(margin) or margin <= 0:
        return np.nan
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0 if abs(d.mean()) < margin else 1.0
    se = sd / np.sqrt(n)
    df = n - 1
    p_lo = sps.t.sf((d.mean() + margin) / se, df)     # H0: mu <= -margin
    p_hi = sps.t.cdf((d.mean() - margin) / se, df)    # H0: mu >= +margin
    return max(p_lo, p_hi)


def build_paired(cells, margins=None, primary=None):
    """
    Primary table: paired sum-vs-mean differences and difference-in-differences.

    margins: {metric: equivalence margin} for TOST; metrics absent -> no TOST.
    primary: set of (dataset, method, lambda_fair) forming the prespecified
             family. Cells outside it are marked exploratory and receive no
             corrected p-value.
    """
    margins = margins or {}
    rows = []
    arms = defaultdict(dict)
    for (dataset, method, reduction, beta, lam_mob, lam_fair), metrics in cells.items():
        arms[(dataset, method, beta, lam_mob, lam_fair)][reduction] = metrics

    n_cells = n_both_arms = n_enough_runs = 0
    min_runs_seen = None
    for (dataset, method, beta, lam_mob, lam_fair), by_red in sorted(
            arms.items(), key=lambda kv: tuple(map(str, kv[0]))):
        n_cells += 1
        if 'mean' not in by_red or 'sum' not in by_red:
            continue
        n_both_arms += 1
        zero_key = (dataset, method, beta, lam_mob, 0.0)
        zero = arms.get(zero_key, {})

        for metric in METRICS:
            a = np.asarray(by_red['sum'].get(metric, []), dtype=float)
            b = np.asarray(by_red['mean'].get(metric, []), dtype=float)
            n = min(len(a), len(b))
            if n:
                min_runs_seen = n if min_runs_seen is None else min(min_runs_seen, n)
            if n < 2:
                continue
            n_enough_runs += 1
            direct = paired_stats(a[:n] - b[:n])

            did = dict(n=0, mean=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                       p=np.nan, gz=np.nan)
            did_vec = None
            if lam_fair != 0.0 and 'sum' in zero and 'mean' in zero:
                a0 = np.asarray(zero['sum'].get(metric, []), dtype=float)
                b0 = np.asarray(zero['mean'].get(metric, []), dtype=float)
                m = min(n, len(a0), len(b0))
                if m >= 2:
                    did_vec = (a[:m] - a0[:m]) - (b[:m] - b0[:m])
                    did = paired_stats(did_vec)

            is_primary = (primary is not None
                          and (dataset, method, lam_fair) in primary)
            rows.append({
                'dataset': dataset, 'method': method, 'metric': metric,
                'beta': beta, 'lambda_mob': lam_mob, 'lambda_fair': lam_fair,
                'family': 'primary' if is_primary else 'exploratory',
                'n_paired': direct['n'], 'n_did': did['n'],
                'mean_arm': float(np.mean(b[:n])), 'sum_arm': float(np.mean(a[:n])),
                'diff': direct['mean'], 'diff_ci_lo': direct['ci_lo'],
                'diff_ci_hi': direct['ci_hi'], 'diff_p_uncorr': direct['p'],
                'diff_gz': direct['gz'],
                'did': did['mean'], 'did_ci_lo': did['ci_lo'],
                'did_ci_hi': did['ci_hi'], 'did_p': did['p'], 'did_gz': did['gz'],
                'equiv_margin': margins.get(metric, np.nan),
                # TOST operates on the DiD vector -- the declared primary
                # estimand -- and only for prespecified primary cells.
                'tost_p': (tost(did_vec, margins.get(metric))
                           if (is_primary and did_vec is not None) else np.nan),
                'closeness_approx': (dataset in APPROX_CLOSENESS_DATASETS
                                     and metric == CLOSENESS_METRIC),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        # Say WHICH precondition failed - "no arms" and "not enough runs" are
        # very different problems and were previously reported identically.
        print("\nNo paired rows could be formed.")
        print(f"  configurations found                : {n_cells}")
        print(f"  ... with BOTH mean and sum arms     : {n_both_arms}")
        print(f"  ... with >= 2 runs (needed for a CI): {n_enough_runs}")
        if n_both_arms == 0:
            print("  -> the two arms never meet: run both --penalty_reduction "
                  "mean and sum for the same configurations.")
        elif min_runs_seen is not None and min_runs_seen < 2:
            print(f"  -> arms match, but only {min_runs_seen} run(s) per "
                  f"configuration. Paired statistics need >= 2; this is expected "
                  f"for a RUNS=1 smoke test. Use --raw to dump the matched values "
                  f"without statistics.")
        return df

    # ONE inferential family: the DiD test over prespecified primary cells.
    # The direct difference stays uncorrected and descriptive so that a second
    # inferential family is never created. TOST, when requested, is corrected
    # over the same family.
    df['did_p_holm'] = np.nan
    df['tost_p_holm'] = np.nan
    mask = (df['family'] == 'primary') & (~df['closeness_approx'])
    if mask.any():
        df.loc[mask, 'did_p_holm'] = holm(df.loc[mask, 'did_p'].values)
        df.loc[mask, 'tost_p_holm'] = holm(df.loc[mask, 'tost_p'].values)
    return df


def build_secondary(cells):
    """
    Secondary, descriptive: does the metric vary across lambda within one arm?

    All lambdas share the same run seeds (model_seed = seed_base + run_index),
    so observations are PAIRED across lambda and an independent-samples ANOVA
    would be the wrong model. Uses Friedman (repeated measures, k >= 3) or a
    Wilcoxon signed-rank test (k = 2), with Kendall's W as the effect size.
    Requires complete blocks: lambdas with unequal run counts are truncated to
    the common length.
    """
    rows = []
    groups = defaultdict(dict)
    for (dataset, method, reduction, beta, lam_mob, lam_fair), metrics in cells.items():
        groups[(dataset, method, reduction, beta, lam_mob)][lam_fair] = metrics

    for (dataset, method, reduction, beta, lam_mob), by_lam in groups.items():
        if len(by_lam) < 2:
            continue
        for metric in METRICS:
            lams = sorted(by_lam)
            samples = [np.asarray(by_lam[l].get(metric, []), dtype=float) for l in lams]
            samples = [s for s in samples if len(s) >= 2]
            if len(samples) < 2:
                continue
            n_blocks = min(len(s) for s in samples)      # complete blocks only
            if n_blocks < 2:
                continue
            blocks = [s[:n_blocks] for s in samples]
            k = len(blocks)

            p_val, stat_name, kendall_w = np.nan, '', np.nan
            try:
                if k >= 3:
                    chi2, p_val = sps.friedmanchisquare(*blocks)
                    stat_name = 'friedman'
                    kendall_w = chi2 / (n_blocks * (k - 1)) if n_blocks and k > 1 else np.nan
                else:
                    _, p_val = sps.wilcoxon(blocks[0], blocks[1])
                    stat_name = 'wilcoxon'
            except Exception:
                pass

            spread = float(np.max([b.mean() for b in blocks])
                           - np.min([b.mean() for b in blocks]))
            rows.append({
                'dataset': dataset, 'method': method, 'reduction': reduction,
                'metric': metric, 'n_lambda': k, 'n_blocks': n_blocks,
                'test': stat_name, 'p_paired': p_val, 'kendall_w': kendall_w,
                'spread_of_means': spread,
                'closeness_approx': (dataset in APPROX_CLOSENESS_DATASETS
                                     and metric == CLOSENESS_METRIC),
            })
    return pd.DataFrame(rows)


def parse_margins(spec):
    """'Precision@10=0.002,Equality_Gain@10=0.5' -> {metric: margin}."""
    if not spec:
        return {}
    try:
        float(spec)
    except ValueError:
        pass
    else:
        raise SystemExit(
            "--equiv_margin must be given PER METRIC, e.g.\n"
            "  --equiv_margin 'Precision@10=0.002,Equality_Gain@10=0.5,"
            "Pagerank_EG@10=0.5,Closeness_EG@10=0.5'\n"
            "A single scalar is not meaningful: Precision@10 lies on [0,1] while "
            "the EG measures are percentage-point-like with magnitudes of 5-15, "
            "so one margin cannot serve both. Fix margins BEFORE looking at the "
            "results, or omit the flag and make no equivalence claim.")
    out = {}
    for part in spec.split(','):
        if '=' not in part:
            raise SystemExit(f"Bad --equiv_margin entry {part!r}; expected metric=value")
        k, v = part.split('=', 1)
        k = k.strip()
        if k not in METRICS:
            raise SystemExit(f"Unknown metric {k!r} in --equiv_margin; known: {METRICS}")
        out[k] = float(v)
    return out


def parse_primary(spec):
    """'oklahoma:ci_in:1.0,oklahoma:cafe_in:0.3' -> {(ds, method, lambda)}."""
    if not spec:
        return None
    fam = set()
    for part in spec.split(','):
        bits = part.split(':')
        if len(bits) != 3:
            raise SystemExit(f"Bad --primary_lambda entry {part!r}; "
                             f"expected dataset:method:lambda")
        fam.add((bits[0].strip(), bits[1].strip(), float(bits[2])))
    return fam


def main():
    ap = argparse.ArgumentParser(description='Compare penalty reductions (issue B2)')
    ap.add_argument('--results_dir', type=str, default='../results_penalty_reduction')
    ap.add_argument('--out', type=str, default='penalty_reduction_comparison.csv')
    ap.add_argument('--equiv_margin', type=str, default=None,
                    help="PER-METRIC TOST margins, e.g. "
                         "'Precision@10=0.002,Equality_Gain@10=0.5'. Must be fixed "
                         "before inspecting results. Omit to make no equivalence claim.")
    ap.add_argument('--raw', action='store_true',
                    help='Dump matched per-run values with no statistics. Use for '
                         'RUNS=1 smoke tests, where paired CIs are undefined.')
    ap.add_argument('--primary_lambda', type=str, default=None,
                    help="Prespecified primary family as dataset:method:lambda "
                         "triples, e.g. 'oklahoma:ci_in:1.0,oklahoma:cafe_in:0.3'. "
                         "Holm correction is applied within this family only; "
                         "without it, every cell is reported as exploratory and no "
                         "p-value is shown.")
    args = ap.parse_args()

    margins = parse_margins(args.equiv_margin)
    primary = parse_primary(args.primary_lambda)

    cells = load_cells(args.results_dir)

    if args.raw:
        arms = defaultdict(dict)
        for (ds, meth, red, beta, lm, lf), mets in cells.items():
            arms[(ds, meth, beta, lm, lf)][red] = mets
        raw_rows = []
        for (ds, meth, beta, lm, lf), by_red in sorted(
                arms.items(), key=lambda kv: tuple(map(str, kv[0]))):
            if 'mean' not in by_red or 'sum' not in by_red:
                continue
            for metric in METRICS:
                a = by_red['sum'].get(metric, [])
                b = by_red['mean'].get(metric, [])
                for i in range(min(len(a), len(b))):
                    raw_rows.append({
                        'dataset': ds, 'method': meth, 'metric': metric,
                        'lambda_fair': lf, 'run': i,
                        'mean_arm': b[i], 'sum_arm': a[i],
                        'diff': a[i] - b[i]})
        raw = pd.DataFrame(raw_rows)
        if raw.empty:
            raise SystemExit("No matched (mean, sum) cells found.")
        raw_path = args.out.replace('.csv', '_raw.csv')
        raw.to_csv(raw_path, index=False)
        pd.set_option('display.width', 200)
        print(f"\nRaw matched values -> {raw_path}")
        print("\n" + "=" * 96)
        print("RAW MATCHED VALUES (no statistics)")
        print("=" * 96)
        print(raw.round(4).to_string(index=False))
        z = raw[raw['lambda_fair'] == 0.0]
        if not z.empty:
            print("\n" + "=" * 96)
            print("SANITY CHECK at lambda_fair = 0: both arms compute an IDENTICAL "
                  "loss from an identical seed,")
            print("so 'diff' here is pure run-to-run nondeterminism and calibrates "
                  "how large a difference means nothing.")
            print("=" * 96)
            print(z.groupby('metric')['diff'].agg(['min', 'max'])
                  .rename(columns={'min': 'diff_min', 'max': 'diff_max'})
                  .round(6).to_string())
        return

    paired = build_paired(cells, margins, primary)
    secondary = build_secondary(cells)

    if paired.empty:
        pass  # build_paired already printed exactly which precondition failed
    else:
        paired.to_csv(args.out, index=False)
        print(f"\nPrimary (paired) table -> {args.out}  ({len(paired)} rows)")
    sec_path = args.out.replace('.csv', '_secondary_paired.csv')
    secondary.to_csv(sec_path, index=False)
    print(f"Secondary (paired) table -> {sec_path}")

    pd.set_option('display.width', 250)
    if not paired.empty:
        headline = paired[~paired['closeness_approx']]
        excluded = paired[paired['closeness_approx']]

        base_cols = ['dataset', 'method', 'metric', 'lambda_fair', 'n_paired',
                     'n_did', 'mean_arm', 'sum_arm', 'diff', 'diff_gz',
                     'did', 'did_ci_lo', 'did_ci_hi', 'did_gz']

        prim = headline[headline['family'] == 'primary']
        expl = headline[headline['family'] == 'exploratory']

        if primary is None:
            print("\n" + "=" * 125)
            print("EXPLORATORY ONLY - no primary family was prespecified")
            print("=" * 125)
            print("Up to 168 paired comparisons live in this grid. Without "
                  "--primary_lambda, p-values are\nwithheld: picking the cells "
                  "that exclude zero after the fact is not a valid inference.\n"
                  "Re-run with --primary_lambda 'dataset:method:lambda,...' to get "
                  "Holm-corrected tests.")
            print(expl[base_cols].round(4).to_string(index=False))
        else:
            print("\n" + "=" * 125)
            print(f"PRIMARY: difference-in-differences, Holm-corrected within the "
                  f"prespecified family (m = {len(prim)} tests)")
            print("=" * 125)
            prim_cols = base_cols + ['did_p_holm']
            if margins:
                prim_cols += ['equiv_margin', 'tost_p_holm']
            print(prim[prim_cols].round(4).to_string(index=False))
            if not expl.empty:
                print("\n" + "-" * 125)
                print("EXPLORATORY (descriptive only - no p-values, not corrected, "
                      "do not use inferential language)")
                print("-" * 125)
                print(expl[base_cols].round(4).to_string(index=False))

        if not excluded.empty:
            print(f"\nEXCLUDED from the headline ({len(excluded)} rows): "
                  f"{CLOSENESS_METRIC} on {sorted(set(excluded['dataset']))} is "
                  f"computed with unseeded randomized ApproxCloseness "
                  f"(num_nodes > 5000), so differences there may be metric noise "
                  f"rather than a training effect. Reported in the CSV only.")

        print("\n" + "=" * 125)
        print("READING THIS TABLE")
        print("=" * 125)
        print("  DECISION uses did_p_holm on the primary family ONLY.")
        print("  did_p_holm < 0.05  -> the reduction change moved the metric.")
        print("  otherwise          -> INCONCLUSIVE, not evidence that the penalty is inert.")
        print("  The unadjusted did CI and the direct difference are DESCRIPTIVE: with many")
        print("  cells, 'the CI excludes zero' is not by itself a valid inference.")
        print("  diff_gz / did_gz are Hedges-corrected paired standardized effects (g_z).")
        if margins:
            print("  tost_p_holm < 0.05 -> DiD lies within the per-metric margin, i.e.")
            print("                        positive evidence of practical equivalence.")
            print(f"  margins used: {margins}")
            missing = [m for m in METRICS if m not in margins]
            if missing:
                print(f"  no margin given for {missing} -> no equivalence claim for those.")
        else:
            print("  No equivalence test was requested; no 'no effect' claim is supportable")
            print("  from this table. Omitting --equiv_margin is the defensible default.")
        n_small = int((paired['n_paired'] < 5).sum())
        if n_small:
            print(f"\n  WARNING: {n_small} cell(s) have fewer than 5 paired runs. "
                  f"Use RUNS=10 for a conclusive result.")

    if not secondary.empty:
        print("\n" + "=" * 125)
        print("SECONDARY (descriptive): variation across lambda within each arm, "
              "paired across runs (Friedman / Wilcoxon)")
        print("Uncorrected and not a decision rule - read the curves, not the p-values.")
        print("=" * 125)
        print(secondary[~secondary['closeness_approx']][
            ['dataset', 'method', 'reduction', 'metric', 'n_lambda', 'n_blocks',
             'test', 'p_paired', 'kendall_w', 'spread_of_means']]
            .round(4).to_string(index=False))


if __name__ == '__main__':
    main()
