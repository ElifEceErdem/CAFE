"""Filesystem layout of the reproduction pipeline.

* ``EVIDENCE`` -- frozen result files the builders read (committed under
  ``evidence/``; override with ``CAFE_EVIDENCE`` to build from fresh runs).
* ``BUILD``    -- generated tables and figures (``build/``, not committed).
* ``SOURCES``  -- per-row source indexes (``reproduce/sources``; a rerun job
  writes its own under ``<output>/sources`` and sets ``CAFE_SOURCES``).

Source paths inside the pipeline are repository-relative strings such as
``results/validation/.../main_results.json``.  :func:`resolve` looks them up
under ``EVIDENCE`` first, then in the committed snapshot and finally in the
live repository tree, so the
same builder works on a fresh clone (evidence snapshot only) and on the
cluster checkout (full raw campaigns).  ``CAFE_EVIDENCE_ONLY=1`` disables the
fallback, which proves a build needs nothing outside the snapshot.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = Path(os.environ.get("CAFE_EVIDENCE", REPO / "evidence")).resolve()
BUILD = Path(os.environ.get("CAFE_BUILD", REPO / "build")).resolve()
PACKAGE = REPO / "reproduce"
EXPECTED = PACKAGE / "expected"
SOURCES = Path(os.environ.get("CAFE_SOURCES", PACKAGE / "sources")).resolve()
STATIC = PACKAGE / "static"


def resolve(rel):
    """Absolute path of a repository-relative source, evidence snapshot first."""
    rel = str(rel)
    snapshot = REPO / "evidence"
    roots = [EVIDENCE] if os.environ.get("CAFE_EVIDENCE_ONLY") == "1" else [EVIDENCE, snapshot, REPO]
    for root in roots:
        candidate = root / rel
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"source not found in {', '.join(map(str, roots))}: {rel}")


def exists(rel):
    try:
        resolve(rel)
        return True
    except FileNotFoundError:
        return False
