"""Published configurations per dataset, as shell assignments for the jobs.

    eval "$(python -m reproduce.config pokec_n)"

The values are parsed from ``reproduce/sources/main_results_rows.csv``, i.e.
from the configuration keys recorded in the result files behind Tables 4 and
7 -- the single source of truth for what the paper reports (they agree with
``configs/best_configs.csv``, which additionally lists the selection grid).

Variables: IN_LF IN_BETA IN_LAM (CAFE-In), PRE_A PRE_BETA PRE_LAM (CAFE-Pre),
INB_LF INB_BETA (CAFE-In^-B), PREB_A PREB_BETA (CAFE-Pre^-B), P_BETA P_LAM
(CAFE^-P), FUSION_BETA (Table 7 / Table 15(b) fusion control),
FUSION_BETA_MATCHED (= IN_BETA; Table 6 / Table 11 matched control),
CI_IN CI_PRE MB_POST (single-component ablations).
"""
import re
import sys

from reproduce.artifacts._main_rows import load_rows

NUM = r"([0-9.]+)"


def parse(key, pattern):
    m = re.fullmatch(pattern, key)
    if not m:
        raise ValueError(f"unexpected configuration key {key!r} for pattern {pattern!r}")
    return m.groups()


def published(ds):
    rows = load_rows()
    cfg = {}
    cfg["IN_LF"], cfg["IN_BETA"], cfg["IN_LAM"] = parse(rows[(ds, "CAFE-In")]["config"],
                                                        rf"lambda_fair_{NUM}_beta_{NUM}_lam_{NUM}")
    cfg["PRE_A"], cfg["PRE_BETA"], cfg["PRE_LAM"] = parse(rows[(ds, "CAFE-Pre")]["config"],
                                                          rf"preproc_alpha_{NUM}_beta_{NUM}_lam_{NUM}")
    cfg["INB_LF"], cfg["INB_BETA"], _ = parse(rows[(ds, "CAFE-In-B")]["config"],
                                              rf"lambda_fair_{NUM}_beta_{NUM}_lam_{NUM}")
    cfg["PREB_A"], cfg["PREB_BETA"], _ = parse(rows[(ds, "CAFE-Pre-B")]["config"],
                                               rf"preproc_alpha_{NUM}_beta_{NUM}_lam_{NUM}")
    cfg["P_BETA"], cfg["P_LAM"] = parse(rows[(ds, "CAFE-P")]["config"], rf"beta_{NUM}_lambda_{NUM}")
    (cfg["FUSION_BETA"],) = parse(rows[(ds, "HL-GNN+RA")]["config"], rf"lambda_fair_0\.0_beta_{NUM}_lam_0\.0")
    cfg["FUSION_BETA_MATCHED"] = cfg["IN_BETA"]
    for var, method in (("CI_IN", "CI-In"), ("CI_PRE", "CI-Pre"), ("MB_POST", "MB-Post")):
        (cfg[var],) = parse(rows[(ds, method)]["config"], rf"lambda_{NUM}")
    return cfg


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit("usage: python -m reproduce.config <dataset>")
    for key, value in published(argv[0]).items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
