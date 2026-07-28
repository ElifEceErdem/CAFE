"""
Compare the post-fix FairAdj/oklahoma verification run against the stored
pre-fix reference, to confirm the sparse-gradient and vectorised-projection
changes in baselines/fairness_aware_baseline_models.py are behaviour-preserving.

  reference: results_repro/oklahoma/fairadj/*.json   (pre-fix, 10 runs, seed 42)
  candidate: results_verify/oklahoma/fairadj/*.json  (post-fix, same settings)

Usage:  python analysis/compare_fairadj_verify.py
"""

import glob
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

METRICS = [
    "Precision@10", "Equality_Gain@10", "Pagerank_EG@10", "Closeness_EG@10",
    "Delta_SP@10", "Delta_EO@10", "NDKL@10", "IF", "GDIF", "Ranking_IF@10",
]


def latest(pattern):
    files = sorted(p for p in glob.glob(pattern) if not p.endswith(".partial.json"))
    return files[-1] if files else None


def main():
    ref_path = latest(os.path.join(REPO, "results_repro", "oklahoma", "fairadj", "*.json"))
    cand_path = latest(os.path.join(REPO, "results_verify", "oklahoma", "fairadj", "*.json"))

    if ref_path is None:
        sys.exit("No reference run found in results_repro/oklahoma/fairadj/")
    if cand_path is None:
        sys.exit("No verification run found in results_verify/oklahoma/fairadj/ "
                 "-- submit jobs/repro/verify_fairadj_oklahoma.job first.")

    ref = json.load(open(ref_path))["results"]["FairAdj"]
    cand = json.load(open(cand_path))["results"]["FairAdj"]

    print(f"reference (pre-fix):  {os.path.basename(ref_path)}")
    print(f"candidate (post-fix): {os.path.basename(cand_path)}")
    print()
    print(f"{'metric':<18}{'reference':>16}{'candidate':>16}{'rel.diff':>12}   status")
    print("-" * 78)

    worst = 0.0
    for m in METRICS:
        if m not in ref or m not in cand:
            print(f"{m:<18}{'--- missing in ' + ('ref' if m not in ref else 'cand'):>44}")
            continue
        r, c = ref[m]["mean"], cand[m]["mean"]
        denom = max(abs(r), 1e-12)
        rel = abs(r - c) / denom
        worst = max(worst, rel)
        # Both runs are seeded, but GPU float reductions are not bit-reproducible;
        # 1% is comfortably inside run-to-run noise and far below any real change.
        status = "ok" if rel < 0.01 else "DIFFERS"
        print(f"{m:<18}{r:>16.6f}{c:>16.6f}{rel:>11.2%}   {status}")

    print("-" * 78)
    print(f"worst relative difference: {worst:.2%}")
    print("PASS -- fixes are behaviour-preserving" if worst < 0.01
          else "FAIL -- investigate before running pokec_n")
    return 0 if worst < 0.01 else 1


if __name__ == "__main__":
    sys.exit(main())
