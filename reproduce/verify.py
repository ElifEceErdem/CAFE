"""Compare built artifacts with the numbers printed in the paper.

A cell *matches* when the reproduced value, rounded to the printed precision,
equals the printed value (same for the SD).  Each printed cell ends up in one
of four states:

``ok``       reproduced exactly at printed precision;
``known``    differs, and the difference is recorded in
             ``reproduce/known_deviations.csv`` with its reason;
``fail``     differs and is not recorded -- a regression;
``missing``  printed but not produced by the builder.

``--tolerance`` loosens the comparison to a number of printed units, for
reruns that are expected to agree only statistically (GPU non-determinism,
new seeds).  The default (0) demands an exact reproduction of the frozen
evidence.
"""
import csv
import re
import math
from collections import Counter

from reproduce.paths import EXPECTED, PACKAGE
from reproduce.table import read_csv

KEY = ("artifact", "panel", "dataset", "row", "column")
STARS = re.compile(r"\**")


def _num(text):
    text = (text or "").strip()
    if text in ("", "--", "–"):
        return None
    return float(text.replace("+", ""))


def _round(x, decimals):
    # Round half away from zero, like the printed tables (Python rounds to even).
    q = 10 ** decimals
    return math.copysign(math.floor(abs(x) * q + 0.5) / q, x)


def _same(printed, built, decimals, tolerance):
    p, b = _num(printed), _num(built)
    if p is None and b is None:
        return True
    if p is None or b is None:
        return False
    if decimals in ("", None):  # scientific-notation cells: compare 3 significant digits
        return math.isclose(p, b, rel_tol=5e-3 + tolerance * 1e-2)
    text = printed.strip()
    d = len(text.split(".")[1]) if "." in text else 0  # precision as printed (SDs may differ)
    return abs(_round(b, d) - p) <= (tolerance + 1e-9) * 10 ** -d + 1e-12


def known_deviations():
    path = PACKAGE / "known_deviations.csv"
    with open(path) as fh:
        return {(r["artifact"], r["panel"], r["dataset"], r["row"], r["column"], r["field"]): r
                for r in csv.DictReader(fh)}


def compare(artifact, built_csv, tolerance=0.0):
    expected = read_csv(EXPECTED / f"{artifact}.csv")
    built = {tuple(r[k] for k in KEY): r for r in read_csv(built_csv)}
    known = known_deviations()
    results = []
    for e in expected:
        key = tuple(e[k] for k in KEY)
        b = built.get(key)
        if b is None:
            results.append(dict(key=key, field="cell", state="missing", printed=e["mean"], built=""))
            continue
        for field in ("mean", "sd"):
            if field == "sd" and not e["sd"]:
                continue
            if field == "mean" and not e["mean"] and e["extra"]:
                state = "ok" if b["extra"] == e["extra"] else "fail"
                if state == "fail" and key + ("extra",) in known:
                    state = "known"
                results.append(dict(key=key, field="extra", state=state, printed=e["extra"], built=b["extra"]))
                continue
            ok = _same(e[field], b[field], e["decimals"], tolerance)
            state = "ok" if ok else ("known" if key + (field,) in known else "fail")
            results.append(dict(key=key, field=field, state=state, printed=e[field], built=b[field]))
        printed_stars, built_stars = e["extra"], b.get("extra", "")
        if e["mean"] and STARS.fullmatch(printed_stars) and STARS.fullmatch(built_stars) \
                and (printed_stars or built_stars):
            # significance stars printed next to a number (absent stars are checked too)
            state = "ok" if printed_stars == built_stars else \
                ("known" if key + ("extra",) in known else "fail")
            results.append(dict(key=key, field="stars", state=state, printed=printed_stars, built=built_stars))
    return results


def summarize(results):
    return Counter(r["state"] for r in results)
