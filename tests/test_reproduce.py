"""Tests of the paper-reproduction pipeline (``python -m reproduce``).

They run on CPU in seconds and need only the committed evidence snapshot.
"""
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg", PYTHONPATH=str(REPO))


def run(args, tmp, **extra):
    env = dict(ENV, CAFE_BUILD=str(tmp / "build"), **extra)
    return subprocess.run([sys.executable, "-m", *args], cwd=REPO, env=env, capture_output=True, text=True)


def test_evidence_manifest_matches():
    result = run(["reproduce.tools.freeze_evidence", "--check"], Path("/tmp"))
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_artifact_verifies_from_the_snapshot_alone(tmp_path):
    result = run(["reproduce", "check"], tmp_path, CAFE_EVIDENCE_ONLY="1")
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert "FAIL" not in result.stdout
    built = {p.name for p in (tmp_path / "build").iterdir()}
    for name in ("T04.tex", "T07.tex", "T11.tex", "T15.tex", "F03.pdf", "F04_oklahoma.pdf", "F02.pdf", "T01.tex"):
        assert name in built


def test_every_known_deviation_is_still_observed(tmp_path):
    """A recorded deviation that no longer occurs must be removed from the list."""
    result = run(["reproduce", "check", "--details"], tmp_path, CAFE_EVIDENCE_ONLY="1")
    assert result.returncode == 0
    listed = {line.split()[1] + "[" + line.split()[2].strip("[]") + "]"
              for line in result.stdout.splitlines() if line.strip().startswith("known ")}
    with open(REPO / "reproduce" / "known_deviations.csv") as fh:
        rows = list(csv.DictReader(fh))
    recorded = {f"{r['panel']}/{r['dataset']}/{r['row']}/{r['column']}[{r['field']}]" for r in rows}
    assert recorded == listed


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="pdftotext not installed")
def test_expected_values_match_the_manuscript(tmp_path):
    """reproduce/expected is exactly what the extractor reads from the PDF."""
    out = tmp_path / "expected"
    shutil.copytree(REPO / "reproduce" / "expected", out)
    env = dict(ENV)
    result = subprocess.run([sys.executable, "-c",
                             "import reproduce.tools.extract_paper_values as x, pathlib, sys;"
                             f"x.OUT = pathlib.Path({str(out)!r}); sys.exit(x.main([]))"],
                            cwd=REPO, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for committed in (REPO / "reproduce" / "expected").glob("*.csv"):
        assert committed.read_text() == (out / committed.name).read_text(), committed.name


def test_index_runs_replaces_rerun_rows(tmp_path):
    """A job output directory overrides exactly the rows it re-ran."""
    evidence = REPO / "evidence"
    out = tmp_path / "campaign"
    rows = list(csv.DictReader(open(REPO / "reproduce" / "sources" / "main_results_rows.csv")))
    source = next(r for r in rows if (r["dataset"], r["method"]) == ("oklahoma", "CAFE-In"))["sources"].split(";")[0]
    target = out / "oklahoma" / "cafe-in" / "main_results.json"
    target.parent.mkdir(parents=True)
    shutil.copy2(evidence / (source if source.endswith(".json") else source + "/main_results.json"), target)
    result = run(["reproduce.tools.index_runs", str(out)], tmp_path)
    assert result.returncode == 0, result.stderr
    indexed = list(csv.DictReader(open(out / "sources" / "main_results_rows.csv")))
    changed = [r for r in indexed if not r["sources"].startswith("results/")]
    assert [(r["dataset"], r["method"]) for r in changed] == [("oklahoma", "CAFE-In")]
    check = run(["reproduce", "check", "T04", "T07"], tmp_path, CAFE_EVIDENCE=str(out),
                CAFE_SOURCES=str(out / "sources"))
    assert check.returncode == 0, check.stdout[-3000:]


def test_published_configuration_matches_best_configs():
    from reproduce.config import published
    best = {(r["Method"], r["Dataset"]): r["Config_Key"]
            for r in csv.DictReader(open(REPO / "configs" / "best_configs.csv"))}
    for ds in ("brightkite", "ego-facebook", "oklahoma", "pokec_n", "unc"):
        c = published(ds)
        assert best[("CAFE-In", ds)] == f"lambda_fair_{c['IN_LF']}_beta_{c['IN_BETA']}_lam_{c['IN_LAM']}"
        assert best[("CAFE-Pre", ds)] == f"preproc_alpha_{c['PRE_A']}_beta_{c['PRE_BETA']}_lam_{c['PRE_LAM']}"
        assert best[("CI-In", ds)] == f"lambda_{c['CI_IN']}"


def test_jobs_are_valid_bash():
    for job in sorted((REPO / "reproduce" / "jobs").glob("*")):
        if job.suffix in (".job", ".sh"):
            assert subprocess.run(["bash", "-n", str(job)]).returncode == 0, job.name
