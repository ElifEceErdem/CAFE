"""Run a stage across several tables at once.

    python -m experiments.tables --stage verify
    python -m experiments.tables --stage verify --tables 4,7,8
    python -m experiments.tables --stage build  --tables all
    python -m experiments.tables --stage plan   --tables 6

``--stage verify`` exits non-zero if any table's sources are missing,
incomplete, or off the corrected Equality-Gain metric.  That exit code is what
``experiments/jobs/build_tables.job`` keys on: a table that fails here has
Degree/PageRank EG columns produced by the metric superseded on 2026-07-23 and
must not be pasted into the manuscript.
"""

import argparse
import importlib
import sys

from experiments.tables import MANUAL_TABLES, TABLE_MODULES
from experiments.tables._framework import build, check_published_configs, plan, verify


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["plan", "verify", "build", "configs"],
                   default="verify")
    p.add_argument("--tables", default="all",
                   help="comma-separated table numbers, or 'all'")
    args = p.parse_args()

    if args.stage == "configs":
        return 1 if check_published_configs() else 0

    if args.tables == "all":
        wanted = list(TABLE_MODULES)
    else:
        wanted = [t.strip() for t in args.tables.split(",")]

    unknown = [t for t in wanted if t not in TABLE_MODULES]
    if unknown:
        for t in unknown:
            reason = MANUAL_TABLES.get(t)
            print(f"Table {t}: " + (reason if reason else "unknown"))
        wanted = [t for t in wanted if t in TABLE_MODULES]

    failures = []
    for number in wanted:
        module = importlib.import_module(
            f"experiments.tables.{TABLE_MODULES[number]}")
        spec = module.SPEC
        print("\n" + "=" * 78)
        if args.stage == "plan":
            plan(spec)
        elif args.stage == "verify":
            if verify(spec):
                failures.append(number)
        else:
            if build(spec):
                failures.append(number)

    print("\n" + "=" * 78)
    if failures:
        print(f"{args.stage}: failed for table(s) {', '.join(failures)}")
        return 1
    print(f"{args.stage}: all requested tables OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
