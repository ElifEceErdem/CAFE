"""Command line: ``python -m tables [IDs] [--list] [--csv]`` (see tables/__init__.py)."""
import argparse
import importlib
import sys
import traceback

from tables import ARTIFACTS
from tables._common import BUILD, RESULTS, write_csv


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tables", description=__doc__)
    ap.add_argument("ids", nargs="*", help="artifacts to build, e.g. T04 F03 (default: all)")
    ap.add_argument("--list", action="store_true", help="list the artifacts and exit")
    ap.add_argument("--csv", action="store_true", help="also write every table cell to build/<ID>.csv")
    args = ap.parse_args(argv)

    if args.list:
        for a in ARTIFACTS:
            print(f"{a.id}  tables/{a.module}.py  {a.title}" + (f"  [{a.note}]" if a.note else ""))
        return 0

    wanted = {i.upper() for i in args.ids}
    unknown = wanted - {a.id for a in ARTIFACTS}
    if unknown:
        ap.error(f"unknown artifact(s): {', '.join(sorted(unknown))}; see --list")
    print(f"results: {RESULTS}\nbuild:   {BUILD}")
    failed = []
    for a in ARTIFACTS:
        if wanted and a.id not in wanted:
            continue
        try:
            cells = importlib.import_module(f"tables.{a.module}").build(BUILD)
            if cells and args.csv:
                write_csv(cells, BUILD / f"{a.id}.csv")
            print(f"  {a.id}  ok")
        except Exception as exc:  # report every artifact, then fail
            failed.append(a.id)
            print(f"  {a.id}  FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=2)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
