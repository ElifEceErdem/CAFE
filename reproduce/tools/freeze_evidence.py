"""Copy every result file the builders read into ``evidence/`` and checksum it.

Each builder module declares ``input_files()`` (repository-relative paths; run
directories are read through their ``main_results.json``).  This tool copies
those files, preserving their paths, and writes ``evidence/MANIFEST.sha256``.

    python -m reproduce.tools.freeze_evidence          # refresh the snapshot
    python -m reproduce.tools.freeze_evidence --check  # verify checksums only

Run it on the cluster checkout (where the raw campaigns live) after a new
campaign has been accepted; commit the result.
"""
import argparse
import hashlib
import importlib
import shutil
import sys

from reproduce import registry
from reproduce.loaders import run_file
from reproduce.paths import EVIDENCE, REPO

MANIFEST = EVIDENCE / "MANIFEST.sha256"
# Snapshot paths that differ from where the files were produced (the original
# directory names were misleading or campaign-dated).  snapshot prefix -> origin.
ALIASES = {
    "results_repro/demographic_significance/": "results_repro/_stale_significance_20260720/",
    "analysis/plots/stratified_figure4/": "results/validation/20260813_184859/figure4/",
    "results_repro/demographic_backbone/results_cafe_p_baseline_oklahoma.json":
        "results_repro/oklahoma/cafe_p/results_cafe_p_baseline_oklahoma_20260719_170453.json",
    "results_repro/demographic_backbone/results_cafe_p_baseline_pokec_n.json":
        "results_repro/pokec_n/cafe_p/results_cafe_p_baseline_pokec_n_20260719_180736.json",
    "results_repro/demographic_backbone/results_cafe_p_baseline_unc.json":
        "results_repro/unc/cafe_p/results_cafe_p_baseline_unc_20260719_172239.json",
}


def origin(rel):
    for snap, orig in ALIASES.items():
        if rel.startswith(snap):
            return orig + rel[len(snap):]
    return rel


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def declared_inputs():
    files = set()
    for a in registry.ARTIFACTS:
        if a.builder in ("", "static"):
            continue
        module = importlib.import_module(f"reproduce.artifacts.{a.builder}")
        for rel in getattr(module, "input_files", lambda: [])():
            path = REPO / origin(rel)
            files.add(run_file(rel) if path.is_dir() else rel)
    return sorted(files)


def check():
    bad = 0
    for line in MANIFEST.read_text().splitlines():
        digest, rel = line.split("  ", 1)
        path = EVIDENCE / rel
        if not path.exists() or sha256(path) != digest:
            print(f"MISMATCH {rel}")
            bad += 1
    print(f"{'OK' if not bad else 'FAILED'}: {MANIFEST}")
    return 1 if bad else 0


def freeze():
    files = declared_inputs()
    missing = [rel for rel in files if not (REPO / origin(rel)).exists()]
    if missing:
        raise SystemExit("inputs missing from the working tree:\n  " + "\n  ".join(missing))
    lines = []
    for rel in files:
        dst = EVIDENCE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if (REPO / origin(rel)).resolve() != dst.resolve():
            shutil.copy2(REPO / origin(rel), dst)
        lines.append(f"{sha256(dst)}  {rel}")
    MANIFEST.write_text("\n".join(lines) + "\n")
    size = sum((EVIDENCE / rel).stat().st_size for rel in files)
    print(f"froze {len(files)} files ({size / 1e6:.1f} MB) into {EVIDENCE}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    return check() if args.check else freeze()


if __name__ == "__main__":
    sys.exit(main())
