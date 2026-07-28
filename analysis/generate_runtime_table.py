"""Generate Table 12 (Appendix A.2): per-dataset wall-clock runtime.

Renders the JSONs written by ``experiments/run_runtime_benchmark.py`` into the
paper's layout: one-time preprocessing (coreness, RA), training (vanilla,
CAFE-In, CAFE-Pre) and inference (vanilla, hybrid), all in seconds.

Training columns carry mean +/- std over the repetitions; the one-time and
inference columns are reported as means, matching the published table (their
run-to-run spread is below the printed precision).

The generator refuses to mix hardware: every input JSON records the GPU it ran
on, and a table spanning two devices would make the 1.00x / 1.02-1.07x
overhead claims of Section 5.5 meaningless.  Pass --allow_mixed_hardware only
if you intend to report that explicitly.

Usage:
    python generate_runtime_table.py --results_dir ../results \
        --out ../results_repro/tables/runtime_table.tex
"""

import argparse
import json
import os

DATASET_ORDER = ["ego-facebook", "oklahoma", "unc", "pokec_n", "brightkite"]
DATASET_LABELS = {
    "ego-facebook": "Ego-FB", "oklahoma": "Oklahoma", "unc": "UNC",
    "pokec_n": "Pokec-n", "brightkite": "Brightkite",
}


def load(results_dir):
    out = {}
    for ds in DATASET_ORDER:
        path = os.path.join(results_dir, ds, "runtime", f"runtime_{ds}.json")
        if not os.path.exists(path):
            print(f"  {ds:14s} MISSING {path}")
            continue
        with open(path) as fh:
            out[ds] = json.load(fh)
    return out


def fmt_mean(node, digits=2):
    if node is None:
        return "--"
    return f"{node['mean']:.{digits}f}"


def fmt_pm(node, digits=0):
    if node is None:
        return "--"
    return f"{node['mean']:.{digits}f} $\\pm$ {node['std']:.{digits}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="../results")
    ap.add_argument("--out", default="../results_repro/tables/runtime_table.tex")
    ap.add_argument("--allow_mixed_hardware", action="store_true")
    args = ap.parse_args()

    data = load(args.results_dir)
    if not data:
        raise SystemExit("no runtime JSONs found")

    gpus = {ds: d.get("environment", {}).get("gpu", "unknown") for ds, d in data.items()}
    if len(set(gpus.values())) > 1 and not args.allow_mixed_hardware:
        raise SystemExit(
            "runtime JSONs span multiple GPUs -- the table would not be "
            f"comparable across rows: {gpus}\n"
            "Re-run on one device, or pass --allow_mixed_hardware and say so "
            "in the caption.")

    runs = {ds: d.get("config", {}).get("runs") for ds, d in data.items()}
    print(f"  hardware: {sorted(set(gpus.values()))}")
    print(f"  repetitions per dataset: {runs}")

    lines = [
        r"\begin{table}[t]",
        r"\caption{Per-dataset wall-clock runtime (mean over "
        f"{sorted(set(str(r) for r in runs.values()))[0]} seeds $\\pm$ std). "
        r"All times in seconds.}",
        r"\label{tab:runtime}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r" & \multicolumn{2}{c}{One-time} & \multicolumn{3}{c}{Training}"
        r" & \multicolumn{2}{c}{Inference} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
        r"Dataset & Coreness & RA & Vanilla & CAFE-In & CAFE-Pre"
        r" & Vanilla & Hybrid \\",
        r"\midrule",
    ]
    csv_rows = ["dataset,coreness_s,ra_s,train_vanilla_s,train_cafe_in_s,"
                "train_cafe_pre_s,infer_vanilla_s,infer_hybrid_s,gpu,runs"]
    for ds in DATASET_ORDER:
        if ds not in data:
            continue
        t = data[ds]["timings"]
        lines.append(
            f"{DATASET_LABELS[ds]} & {fmt_mean(t.get('coreness_time'), 1)} & "
            f"{fmt_mean(t.get('ra_time'))} & {fmt_pm(t.get('train_vanilla'))} & "
            f"{fmt_pm(t.get('train_cafe_in'))} & {fmt_pm(t.get('train_cafe_pre'))} & "
            f"{fmt_mean(t.get('infer_vanilla'))} & {fmt_mean(t.get('infer_cafe'))} \\\\")
        csv_rows.append(",".join([
            ds,
            fmt_mean(t.get("coreness_time"), 1), fmt_mean(t.get("ra_time")),
            fmt_mean(t.get("train_vanilla"), 0), fmt_mean(t.get("train_cafe_in"), 0),
            fmt_mean(t.get("train_cafe_pre"), 0),
            fmt_mean(t.get("infer_vanilla")), fmt_mean(t.get("infer_cafe")),
            gpus[ds], str(runs[ds]),
        ]))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    csv_path = os.path.splitext(args.out)[0] + ".csv"
    with open(csv_path, "w") as fh:
        fh.write("\n".join(csv_rows) + "\n")
    print(f"  wrote {args.out}")
    print(f"  wrote {csv_path}")


if __name__ == "__main__":
    main()
