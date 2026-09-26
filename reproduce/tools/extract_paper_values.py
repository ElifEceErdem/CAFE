"""Extract every number printed in the paper's tables into reproduce/expected/.

The manuscript PDF is the source of truth for what a reproduction must match.
This tool reads its text layer (``pdftotext -layout``) and writes one tidy CSV
per table::

    artifact,panel,dataset,row,column,mean,sd,decimals,extra

``mean``/``sd`` are the printed numbers, ``decimals`` the printed precision
(used as the comparison tolerance) and ``extra`` any non-numeric annotation
such as significance stars.

Usage::

    python -m reproduce.tools.extract_paper_values [--pdf paper/CAFE_ToRS_EceErdem.pdf]

The output is committed; rerun only when the manuscript changes.
"""
import argparse
import csv
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reproduce" / "expected"
FIELDS = ["artifact", "panel", "dataset", "row", "column", "mean", "sd", "decimals", "extra"]

DATASETS = {"Brightkite": "brightkite", "Ego-FB": "ego-facebook", "Oklahoma": "oklahoma",
            "Pokec-n": "pokec_n", "UNC": "unc"}
# Longest labels first so that e.g. "HL-GNN+RA" wins over "HL-GNN".
METHODS = [
    ("Cross-LinkBias", "Cross-LinkBias"), ("Tail-STEAK", "TailSTEAK"), ("TailSTEAK", "TailSTEAK"),
    ("HL-GNN+RA", "HL-GNN+RA"), ("CAFE-Pre-B", "CAFE-Pre-B"), ("CAFE-In-B", "CAFE-In-B"),
    ("CAFE-Pre", "CAFE-Pre"), ("CAFE-In", "CAFE-In"), ("CAFE-P", "CAFE-P"), ("LightGCN", "LightGCN"),
    ("HL-GNN", "HL-GNN"), ("FairAdj", "FairAdj"), ("MB-Post", "MB-Post"), ("CI-Pre", "CI-Pre"),
    ("CI-In", "CI-In"), ("Random", "Random"), ("SSNet", "SSNet"), ("MORAL", "MORAL"),
    ("Fusion", "HL-GNN+RA"), ("LTLP", "LTLP"), ("LBA", "LBA"), ("PA", "PA"), ("RA", "RA"),
    ("AA", "AA"), ("CN", "CN"),
]
NUM = r"[+\-]?\d+\.\d+"
CELL = re.compile(rf"({NUM})(?:\s*±\s*({NUM}))?")


def load_text(pdf):
    raw = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], check=True,
                         capture_output=True, text=True).stdout
    text = unicodedata.normalize("NFKC", raw).replace("−", "-").replace("★", "").replace("∗", "*")
    # Normalise ablation labels: "CAFE-In -B" / "CAFE -P" (italic superscripts).
    text = re.sub(r"CAFE-(In|Pre)\s*-\s*B", r"CAFE-\1-B", text)
    text = re.sub(r"CAFE\s*-\s*P\b", "CAFE-P", text)
    text = re.sub(r"(\d)–(\d)", r"\1-\2", text)  # numeric ranges such as 1–96
    # Strip review-mode line numbers printed at column 0.
    return [re.sub(r"^\d{1,4}(?=\s|$)", "", line) for line in text.splitlines()]


def section(lines, start, end):
    """Lines from the caption 'Table <start>.' up to the next marker."""
    i = next(k for k, l in enumerate(lines) if re.search(rf"\bTable {start}\.", l))
    j = next((k for k in range(i + 1, len(lines)) if re.search(end, lines[k])), len(lines))
    return lines[i:j]


def decimals(token):
    return len(token.split(".")[1]) if "." in token else 0


def find_method(line):
    for label, canon in METHODS:
        m = re.search(rf"(?<![\w+\-]){re.escape(label)}(?![\w+\-])", line)
        if m:
            return canon, m.end()
    return None, None


def cells(text):
    return [(m.group(1), m.group(2)) for m in CELL.finditer(text)]


def rows_for(artifact, lines, columns, panel="", dataset_from_line=True, dataset=None, row_key=find_method,
             expect=None):
    out, ds = [], dataset
    for line in lines:
        if dataset_from_line:
            for label, canon in DATASETS.items():
                if re.match(rf"\s*{re.escape(label)}\s", line):
                    ds = canon
        name, pos = row_key(line)
        if not name:
            continue
        vals = cells(line[pos:])
        if len(vals) < len(columns):
            continue
        for col, (mean, sd) in zip(columns, vals[: len(columns)]):
            out.append(dict(artifact=artifact, panel=panel, dataset=ds, row=name, column=col,
                            mean=mean, sd=sd or "", decimals=decimals(mean), extra=""))
    if expect is not None and len(out) != expect:
        raise SystemExit(f"{artifact}{panel}: parsed {len(out)} cells, expected {expect}")
    return out


METRIC4 = ["prec", "deg_eg", "pr_eg", "cl_eg"]


def t02(lines):
    out = []
    for line in section(lines, 2, r"UNC \[44\]"):
        for label, canon in DATASETS.items():
            m = re.match(rf"\s*{re.escape(label)}\s+([\d,]+)\s+([\d,]+)\s+({NUM})\s+({NUM})\s+(\S+)", line)
            if m:
                for col, v in zip(["nodes", "edges", "avg_degree", "avg_clustering"], m.groups()[:4]):
                    out.append(dict(artifact="T02", panel="", dataset=canon, row=canon, column=col,
                                    mean=v.replace(",", ""), sd="", decimals=decimals(v), extra=""))
                out.append(dict(artifact="T02", panel="", dataset=canon, row=canon, column="attribute",
                                mean="", sd="", decimals="", extra=m.group(5).replace("—", "none")))
    assert len(out) == 25, len(out)
    return out


def t04(lines):
    block = section(lines, 4, r"Table 5\.")
    out, ds = [], None
    panels = {"(a) Brightkite": "brightkite", "(b) Ego-FB": "ego-facebook", "(c) Oklahoma": "oklahoma",
              "(d) Pokec-n": "pokec_n", "(e) UNC": "unc"}
    for line in block:
        for tag, canon in panels.items():
            if tag in line:
                ds = canon
        if ds is None or "7.1" in line[:12]:
            continue
        name, pos = find_method(line)
        vals = cells(line[pos:]) if name else []
        if name and len(vals) == 4 and not re.search(r"[a-z]{4,} [a-z]{4,}", line[pos:]):
            for col, (mean, sd) in zip(METRIC4, vals):
                out.append(dict(artifact="T04", panel="", dataset=ds, row=name, column=col, mean=mean,
                                sd=sd or "", decimals=decimals(mean), extra=""))
    # 13 + 13 + 16 + 16 + 16 rows x 4 metrics
    assert len(out) == 74 * 4, len(out)
    return out


def t05(lines):
    out = []
    for line in section(lines, 5, r"while LTLP"):
        for label, canon in DATASETS.items():
            m = re.match(rf"\s*{re.escape(label)}\s+(\d+)\s+(\d+)-(\d+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+({NUM})",
                         line)
            if m:
                vals = dict(bins=m.group(1), coreness_min=m.group(2), coreness_max=m.group(3),
                            eval_users=m.group(4), min_bin=m.group(5), max_bin=m.group(6), rho_s=m.group(7))
                for col, v in vals.items():
                    out.append(dict(artifact="T05", panel="", dataset=canon, row=canon, column=col,
                                    mean=v.replace(",", ""), sd="", decimals=decimals(v), extra=""))
    assert len(out) == 35, len(out)
    return out


def t06(lines):
    block = section(lines, 6, r"Table 7\.")
    out = []
    for line in block:
        for label, canon in DATASETS.items():
            m = re.match(rf"\s*{re.escape(label)}\s+({NUM})\s*±\s*({NUM})\s+({NUM})\s*→\s*({NUM})", line)
            if m:
                out.append(dict(artifact="T06", panel="a", dataset=canon, row=canon, column="dprec_lowest_bin",
                                mean=m.group(1), sd=m.group(2), decimals=4, extra=""))
                for col, v in [("ratio_fusion", m.group(3)), ("ratio_cafe_in", m.group(4))]:
                    out.append(dict(artifact="T06", panel="a", dataset=canon, row=canon, column=col, mean=v,
                                    sd="", decimals=decimals(v), extra=""))
            m = re.match(rf"\s*{re.escape(label)}\s+({NUM})%\s*\(\s*({NUM})\s*[x×]\s*10\s*-\s*(\d)\s*\)\s+"
                         rf"({NUM})%\s*\(\s*({NUM})\s*[x×]\s*10\s*-\s*(\d)\s*\)\s+({NUM})%\s*\(\s*({NUM})\s*[x×]\s*10\s*-\s*(\d)\s*\)",
                         line)
            if m:
                g = m.groups()
                for k, tier in enumerate(["periphery", "middle", "core"]):
                    rel, mant, exp = g[3 * k], g[3 * k + 1], g[3 * k + 2]
                    out.append(dict(artifact="T06", panel="b", dataset=canon, row=canon, column=f"rel_{tier}",
                                    mean=rel, sd="", decimals=1, extra=""))
                    out.append(dict(artifact="T06", panel="b", dataset=canon, row=canon, column=f"abs_{tier}",
                                    mean=f"{float(mant)}e-{exp}", sd="", decimals="", extra=f"{mant}x10^-{exp}"))
    assert len(out) == 15 + 30, len(out)
    return out


def t07(lines):
    block = section(lines, 7, r"Answer to RQ4")
    out, ds = [], None
    for line in block:
        for label, canon in DATASETS.items():
            if re.match(rf"\s*{re.escape(label)}\s", line):
                ds = canon
        name, pos = find_method(line)
        if not name or ds is None:
            continue
        rest = line[pos:]
        dprec = re.search(r"\s(–|[+\-]\d+%)\s", rest)
        vals = cells(rest)
        if len(vals) != 4 or not dprec:
            continue
        cols = ["prec", "deg_eg", "pr_eg", "cl_eg"]
        for col, (mean, sd) in zip(cols, vals):
            out.append(dict(artifact="T07", panel="", dataset=ds, row=name, column=col, mean=mean, sd=sd or "",
                            decimals=decimals(mean), extra=""))
        if dprec.group(1) != "–":
            out.append(dict(artifact="T07", panel="", dataset=ds, row=name, column="dprec_pct",
                            mean=dprec.group(1).rstrip("%"), sd="", decimals=0, extra=""))
    assert len(out) == 50 * 4 + 45, len(out)
    return out


def t08(lines):
    block = section(lines, 8, r"Table 9\.")
    out, ds = [], None
    panels = {"(a) Oklahoma": "oklahoma", "(b) Pokec-n": "pokec_n", "(c) UNC": "unc"}
    for line in block:
        for tag, canon in panels.items():
            if tag in line:
                ds = canon
        name, pos = find_method(line)
        vals = cells(line[pos:]) if name else []
        if ds and name and len(vals) == 3:
            for col, (mean, sd) in zip(["prec", "dsp_e3", "deo_e3"], vals):
                out.append(dict(artifact="T08", panel="", dataset=ds, row=name, column=col, mean=mean, sd=sd or "",
                                decimals=decimals(mean), extra=""))
    assert len(out) == 30 * 3, len(out)
    return out


def t09(lines):
    block = section(lines, 9, r"Answer to RQ5")
    out, ds = [], None
    for line in block:
        for label, canon in DATASETS.items():
            m = re.match(rf"\s*{re.escape(label)}\s+({NUM})\s+\((\w+)\)\s+(S\d)\s+(S\d)\s+(\S)", line)
            if m:
                ds = canon
                out.append(dict(artifact="T09", panel="", dataset=canon, row=canon, column="cliffs_delta",
                                mean=m.group(1), sd="", decimals=2, extra=m.group(2)))
                for col, v in [("more_core", m.group(3)), ("more_exposed", m.group(4)), ("predicted", m.group(5))]:
                    out.append(dict(artifact="T09", panel="", dataset=canon, row=canon, column=col, mean="",
                                    sd="", decimals="", extra=v))
        m = re.search(rf"(CAFE-In|CAFE-Pre)\s+({NUM})\s*±\s*({NUM})(\**)\s+({NUM})\s*±\s*({NUM})(\**)", line)
        if m:
            nxt = ds
            if nxt is None:  # the first variant line precedes the dataset line
                nxt = "pending"
            for col, a, b, s in [("vs_backbone", m.group(2), m.group(3), m.group(4)),
                                 ("vs_fusion", m.group(5), m.group(6), m.group(7))]:
                out.append(dict(artifact="T09", panel="", dataset=nxt, row=m.group(1), column=col, mean=a, sd=b,
                                decimals=3, extra=s))
    # Variant lines printed above their dataset label inherit the next dataset.
    order = ["oklahoma", "unc", "pokec_n"]
    variant_rows = [r for r in out if r["row"] in ("CAFE-In", "CAFE-Pre")]
    for k, r in enumerate(variant_rows):
        r["dataset"] = order[k // 4]
    assert len(out) == 3 * 4 + 12, len(out)
    return out


def t10(lines):
    block = section(lines, 10, r"case\. This suggests")
    out, panel = [], "a"
    for line in block:
        if "(b) Ego-FB dose curve" in line:
            panel = "b"
        if panel == "a":
            for label, canon in DATASETS.items():
                m = re.match(rf"\s*{re.escape(label)}\s+" + r"\s+".join([f"({NUM})"] * 5) +
                             rf"\s+({NUM})\s*→\s*({NUM})\s+({NUM})\s*→\s*({NUM})", line)
                if m:
                    g = m.groups()
                    cols = ["q1", "q2", "q3", "q4", "q5", "gini_control", "gini_boost", "cl_eg_control", "cl_eg_boost"]
                    for col, v in zip(cols, g):
                        out.append(dict(artifact="T10", panel="a", dataset=canon, row=canon, column=col, mean=v,
                                        sd="", decimals=decimals(v), extra=""))
        else:
            m = re.match(r"\s*(0\.1|0\.3|0\.5|1\.0)\s+" + r"\s+".join([f"({NUM})"] * 7), line)
            if m:
                g = m.groups()
                for col, v in zip(["q1", "q2", "q3", "q4", "q5", "gini", "cl_eg"], g[1:]):
                    out.append(dict(artifact="T10", panel="b", dataset="ego-facebook", row=f"lambda_{g[0]}",
                                    column=col, mean=v, sd="", decimals=decimals(v), extra=""))
            m = re.match(rf"\s*control \(\+RA\).*?({NUM})\s+({NUM})\s*$", line)
            if m:
                for col, v in zip(["gini", "cl_eg"], m.groups()):
                    out.append(dict(artifact="T10", panel="b", dataset="ego-facebook", row="control",
                                    column=col, mean=v, sd="", decimals=decimals(v), extra=""))
    assert len(out) == 27 + 30, len(out)
    return out


def t11(lines):
    block = section(lines, 11, r"this sensitivity")
    return rows_for("T11", block, ["deg_full", "deg_pw", "pr_full", "pr_pw", "cl_full", "cl_tp"], expect=25 * 6)


def t13(lines):
    out = []
    for line in section(lines, 13, r"Stage 1: Pareto"):
        for label, canon in DATASETS.items():
            m = re.match(rf"\s*{re.escape(label)}\s+({NUM})\s+({NUM})\s+(\d+)\s*±\s*(\d+)\s+(\d+)\s*±\s*(\d+)\s+"
                         rf"(\d+)\s*±\s*(\d+)\s+({NUM})\s+({NUM})", line)
            if m:
                g = m.groups()
                spec = [("coreness_s", g[0], ""), ("ra_s", g[1], ""), ("train_vanilla_s", g[2], g[3]),
                        ("train_cafe_in_s", g[4], g[5]), ("train_cafe_pre_s", g[6], g[7]),
                        ("infer_vanilla_s", g[8], ""), ("infer_hybrid_s", g[9], "")]
                for col, v, sd in spec:
                    out.append(dict(artifact="T13", panel="", dataset=canon, row=canon, column=col, mean=v, sd=sd,
                                    decimals=decimals(v), extra=""))
    assert len(out) == 35, len(out)
    return out


def t15(lines):
    block = section(lines, 15, r"Size-matched bins")
    out, panel, variant = [], None, None
    tags = {"(a) vs. HL-GNN": "vs_hlgnn", "(b) vs. fusion": "vs_fusion", "(c) vs. RA": "vs_ra"}
    for line in block:
        for tag, p in tags.items():
            if tag in line:
                panel = p
        m = re.match(r"\s*(CAFE-In|CAFE-Pre)\s", line)
        if m:
            variant = m.group(1)
        for label, canon in DATASETS.items():
            m = re.search(rf"(?:^|\s){re.escape(label)}\s+(.*)$", line)
            if m and panel and variant:
                vals = re.findall(rf"({NUM})\s*±\s*({NUM})(\**)", m.group(1))
                if len(vals) == 4:
                    for col, (a, b, s) in zip(METRIC4, vals):
                        out.append(dict(artifact="T15", panel=panel, dataset=canon, row=variant, column=col,
                                        mean=a, sd=b, decimals=decimals(a), extra=s))
    assert len(out) == 3 * 10 * 4, len(out)
    return out


def t16(lines):
    block = section(lines, 16, r"Reciprocal recommendations")
    out, panel, ds = [], None, None
    for line in block:
        if "(a) ΔSP" in line or "(a) Δ" in line:
            panel = "dsp_e3"
        if "(b) ΔEO" in line or "(b) Δ" in line:
            panel = "deo_e3"
        for label, canon in DATASETS.items():
            if re.match(rf"\s*{re.escape(label)}\s", line):
                ds = canon
        name, pos = find_method(line)
        if not name or not panel:
            continue
        rest = line[pos:]
        vals = re.findall(rf"({NUM})\s*±\s*({NUM})(\**)", rest)
        if len(vals) < 2:
            continue
        for col, (a, b, s) in zip(["fixed_split", "split_resampled", "diff_vs_hlgnn", "diff_vs_fusion"], vals):
            out.append(dict(artifact="T16", panel=panel, dataset=ds, row=name, column=col, mean=a, sd=b,
                            decimals=decimals(a), extra=s))
    assert len(out) == 2 * (6 * 2 + 6 * 4), len(out)
    return out


def t17(lines):
    out = []
    for line in section(lines, 17, r"spread aligns"):
        name, pos = find_method(line)
        m = re.search(rf"({NUM})\s+({NUM})\s+({NUM})\s*\[({NUM}),\s*({NUM})\]\s+({NUM})", line[pos:] if name else "")
        if name and m:
            g = m.groups()
            for col, v in zip(["lowest_p10", "highest_p10", "high_minus_low", "ci_low", "ci_high", "rho_s_deg_eg"], g):
                out.append(dict(artifact="T17", panel="", dataset="brightkite", row=name, column=col, mean=v, sd="",
                                decimals=decimals(v), extra=""))
    assert len(out) == 30, len(out)
    return out


def t18(lines):
    block = section(lines, 18, r"Table 19\.")
    out, cur = [], None
    for line in block:
        m = re.match(r"\s*(Ego-FB|Oklahoma|UNC|Pokec-n|Brightkite) coreness\s+(.*)$", line)
        if m:
            cur = DATASETS[m.group(1)]
            labels = re.findall(r"\[(\d+-\d+)\]", m.group(2))
            continue
        if cur and re.match(r"\s*\|?B\s*g\s*\|?", line) and "rho" not in line:
            sizes = re.findall(r"\d[\d,]*", line.split("|", 2)[-1] if "|" in line else line)
            sizes = [s.replace(",", "") for s in sizes][-len(labels):]
            for lab, v in zip(labels, sizes):
                out.append(dict(artifact="T18", panel="a", dataset=cur, row=lab, column="bin_size", mean=v, sd="",
                                decimals=0, extra=""))
        if cur and re.match(r"\s*Deg EG@10", line):
            vals = re.findall(NUM, line.replace("@10", ""))[-len(labels):]
            for lab, v in zip(labels, vals):
                out.append(dict(artifact="T18", panel="a", dataset=cur, row=lab, column="deg_eg", mean=v, sd="",
                                decimals=2, extra=""))
            if cur == "brightkite":
                cur = None
        for label, canon in DATASETS.items():
            m = re.match(rf"\s*{re.escape(label)}\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s+\[({NUM}),\s*({NUM})\]", line)
            if m:
                g = m.groups()
                for col, v in zip(["rho_coreness", "rho_degree", "rho_clustering", "rho_bin_size",
                                   "within_band_min", "within_band_max"], g):
                    out.append(dict(artifact="T18", panel="b", dataset=canon, row=canon, column=col, mean=v, sd="",
                                    decimals=2, extra=""))
            m = re.match(rf"\s*{re.escape(label)}\s+\[(\d+-\d+)\] vs\. \[(\d+-\d+)\]\s+([\d,]+) vs\. ([\d,]+)\s+"
                         rf"({NUM}) vs\. ({NUM})", line)
            if m:
                g = m.groups()
                for which, lab, size, eg in [("first", g[0], g[2], g[4]), ("second", g[1], g[3], g[5])]:
                    out.append(dict(artifact="T18", panel="c", dataset=canon, row=f"{which}:{lab}", column="bin_size",
                                    mean=size.replace(",", ""), sd="", decimals=0, extra=""))
                    out.append(dict(artifact="T18", panel="c", dataset=canon, row=f"{which}:{lab}", column="deg_eg",
                                    mean=eg, sd="", decimals=2, extra=""))
    n_bins = 10 + 9 + 9 + 10 + 6
    assert len(out) == 2 * n_bins + 30 + 20, len(out)
    return out


def t19(lines):
    block = section(lines, 19, r"Table 20\.|^\s*J\. ACM")
    out, pending = [], []
    methods = ["CN", "RA", "AA", "LightGCN", "HL-GNN"]
    ival = re.compile(rf"({NUM})\s*\[({NUM}),\s*({NUM})\]")
    lines_with = [l for l in block if len(ival.findall(l)) == 5]
    labels = []
    for l in block:
        for label, canon in DATASETS.items():
            if re.search(rf"(^|\s){re.escape(label)}(\s|$)", l) and len(ival.findall(l)) == 5:
                labels.append((l, canon))
    # Each dataset has two lines: controls=degree, then degree+clustering.
    order = ["ego-facebook", "oklahoma", "unc", "pokec_n", "brightkite"]
    assert len(lines_with) == 10, len(lines_with)
    for k, l in enumerate(lines_with):
        ds, model = order[k // 2], ("degree" if k % 2 == 0 else "degree_clustering")
        for meth, (est, lo, hi) in zip(methods, ival.findall(l)):
            out.append(dict(artifact="T19", panel=model, dataset=ds, row=meth, column="coef", mean=est,
                            sd="", decimals=2, extra=f"[{lo}, {hi}]"))
    return out


def t20(lines):
    block = section(lines, 20, r"J\. ACM, Vol\. 37, No\. 4, Article 111\. Publication date: August 2026\.\s*$")
    out, ds, meth = [], None, None
    for line in block:
        for label, canon in DATASETS.items():
            if re.match(rf"\s*{re.escape(label)}\s", line):
                ds = canon
        name, pos = find_method(line)
        if name:
            meth = name
            rest = line[pos:]
        else:
            rest = line
        m = re.match(rf"\s*(5|20)\s+({NUM})\s*±\s*({NUM})\s+({NUM})\s*±\s*({NUM})\s+({NUM})\s*±\s*({NUM})\s+"
                     rf"({NUM})\s*±\s*({NUM})", rest)
        if m and ds and meth:
            g = m.groups()
            for k, col in enumerate(METRIC4):
                out.append(dict(artifact="T20", panel=f"K{g[0]}", dataset=ds, row=meth, column=col,
                                mean=g[1 + 2 * k], sd=g[2 + 2 * k], decimals=decimals(g[1 + 2 * k]), extra=""))
    assert len(out) == 5 * 5 * 2 * 4, len(out)
    return out


EXTRACTORS = {"T02": t02, "T04": t04, "T05": t05, "T06": t06, "T07": t07, "T08": t08, "T09": t09,
              "T10": t10, "T11": t11, "T13": t13, "T15": t15, "T16": t16, "T17": t17, "T18": t18,
              "T19": t19, "T20": t20}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", default=str(REPO / "paper" / "CAFE_ToRS_EceErdem.pdf"))
    ap.add_argument("--only", nargs="*", default=list(EXTRACTORS))
    args = ap.parse_args(argv)
    lines = load_text(args.pdf)
    OUT.mkdir(parents=True, exist_ok=True)
    for artifact in args.only:
        rows = EXTRACTORS[artifact](lines)
        with open(OUT / f"{artifact}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"{artifact}: {len(rows)} printed values")


if __name__ == "__main__":
    sys.exit(main())
