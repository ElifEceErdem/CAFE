"""Build and verify the paper's tables and figures.

    python -m reproduce list                  # artifacts, status, jobs
    python -m reproduce build  [T04 F03 ...]  # write build/<ID>.{csv,tex,pdf}
    python -m reproduce verify [T04 ...]      # compare with the printed values
    python -m reproduce check  [T04 ...]      # build + verify (non-zero exit on regression)

``CAFE_EVIDENCE`` points the builders at another result tree (default:
``evidence/``); ``CAFE_BUILD`` sets the output directory (default: ``build/``).
"""
import argparse
import importlib
import sys
import traceback

from reproduce import registry, verify as V
from reproduce.paths import BUILD, EVIDENCE
from reproduce.table import write_csv


def cmd_list(_args):
    print(f"{'ID':4} {'status':13} {'job':28} title")
    for a in registry.ARTIFACTS:
        print(f"{a.id:4} {a.status:13} {a.job or '-':28} {a.title}")


def build_one(a):
    module = importlib.import_module(f"reproduce.artifacts.{a.builder}")
    cells = module.build(BUILD, a) if a.builder == "static" else module.build(BUILD)
    if cells is not None:
        write_csv(cells, BUILD / f"{a.id}.csv")
    return cells


def cmd_build(args):
    failed = []
    for a in registry.select(args.ids):
        try:
            cells = build_one(a)
            print(f"{a.id}: built" + (f" ({len(cells)} cells)" if cells else ""))
        except Exception as exc:  # report every artifact, then fail
            failed.append(a.id)
            print(f"{a.id}: BUILD FAILED -- {type(exc).__name__}: {exc}")
            if args.traceback:
                traceback.print_exc()
    return 1 if failed else 0


def cmd_verify(args):
    status = 0
    print(f"evidence: {EVIDENCE}\nbuild:    {BUILD}\n")
    print(f"{'ID':4} {'ok':>5} {'known':>6} {'fail':>5} {'missing':>8}  verdict")
    for a in registry.select(args.ids):
        if not a.verified:
            continue
        path = BUILD / f"{a.id}.csv"
        if not path.exists():
            print(f"{a.id:4} {'':>5} {'':>6} {'':>5} {'':>8}  not built")
            status = status or (0 if a.status == "needs-rerun" else 1)
            continue
        results = V.compare(a.id, path, args.tolerance)
        n = V.summarize(results)
        bad = n["fail"] + (n["missing"] if a.status not in ("needs-rerun", "partial") else 0)
        verdict = "PASS" if not bad else "FAIL"
        if a.status in ("needs-rerun", "partial") and n["missing"]:
            verdict += f" ({a.status})"
        print(f"{a.id:4} {n['ok']:5d} {n['known']:6d} {n['fail']:5d} {n['missing']:8d}  {verdict}")
        if args.details or bad:
            for r in results:
                if r["state"] in ("fail",) or (args.details and r["state"] != "ok"):
                    print(f"      {r['state']:7} {'/'.join(r['key'][1:])} [{r['field']}] "
                          f"printed={r['printed']} built={r['built']}")
        status = status or (1 if bad else 0)
    return status


def cmd_check(args):
    return cmd_build(args) or cmd_verify(args)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m reproduce", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for name in ("build", "verify", "check"):
        p = sub.add_parser(name)
        p.add_argument("ids", nargs="*", default=["all"])
        p.add_argument("--tolerance", type=float, default=0.0,
                       help="allowed difference in printed units (default 0: exact)")
        p.add_argument("--details", action="store_true", help="list known deviations too")
        p.add_argument("--traceback", action="store_true")
    args = ap.parse_args(argv)
    return {"list": cmd_list, "build": cmd_build, "verify": cmd_verify, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
