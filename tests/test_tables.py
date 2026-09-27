"""Tests: the result snapshot is intact and every table and figure builds from it."""
import csv
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg", PYTHONPATH=str(REPO))


def test_result_checksums():
    result = subprocess.run(["sha256sum", "--check", "--quiet", "SHA256SUMS"], cwd=REPO / "results",
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_artifact_builds(tmp_path):
    result = subprocess.run([sys.executable, "-m", "tables", "--csv"], cwd=REPO,
                            env=dict(ENV, CAFE_BUILD=str(tmp_path)), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    built = {p.name for p in tmp_path.iterdir()}
    for name in ("T04.tex", "T07.tex", "T11.tex", "T15.tex", "F03.pdf", "F04_oklahoma.pdf", "F01a_heatmap.pdf"):
        assert name in built, name


def test_main_table_values(tmp_path):
    """Spot values of Table 4, recomputed from per-run results."""
    subprocess.run([sys.executable, "-m", "tables", "T04", "--csv"], cwd=REPO, check=True,
                   env=dict(ENV, CAFE_BUILD=str(tmp_path)), capture_output=True)
    cells = {(r["dataset"], r["row"], r["column"]): r for r in csv.DictReader(open(tmp_path / "T04.csv"))}
    cafe_in = cells[("oklahoma", "CAFE-In", "prec")]
    assert f"{float(cafe_in['mean']):.4f}" == "0.2003" and f"{float(cafe_in['sd']):.4f}" == "0.0005"
    assert f"{float(cells[('unc', 'CAFE-Pre', 'deg_eg')]['mean']):.2f}" == "2.21"
    assert f"{float(cells[('pokec_n', 'CAFE-In', 'cl_eg')]['mean']):.2f}" == "14.90"


def test_every_indexed_source_exists():
    for name, column in (("main_results_rows.csv", "sources"), ("acceptance_rows.csv", "source"),
                         ("paired_observations.csv", "source")):
        for row in csv.DictReader(open(REPO / "results" / "index" / name)):
            for rel in row[column].split(";"):
                assert (REPO / "results" / rel).exists(), rel
