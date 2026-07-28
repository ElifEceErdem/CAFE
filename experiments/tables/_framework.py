"""Shared machinery for the table-indexed experiment suite.

Every table in the manuscript that is backed by an experiment has one module
in this package.  A module declares *what the table needs* (which runs, on
which datasets, at which published configuration) and *how it is built* (which
analysis generator turns results into LaTeX).  Nothing here re-implements any
science: the run commands shell out to the validated runners in
``experiments/`` and the build commands to the generators in ``analysis/``.

Four stages, uniform across tables::

    python -m experiments.tables.table04_main_results --stage plan
    python -m experiments.tables.table04_main_results --stage verify
    python -m experiments.tables.table04_main_results --stage build
    python -m experiments.tables.table04_main_results --stage run --dataset unc

``plan``    prints the runs the table needs and which are already present.
``verify``  hard-checks the results backing the table: presence, repetition
            count, selected configuration, and -- the reason this suite
            exists -- that every result JSON was produced under the corrected
            Equality-Gain semantics (``EG_SEMANTICS`` below).
``build``   invokes the generator(s) that emit the LaTeX/CSV artifacts.
``run``     prints (or, with ``--execute``, runs) the runner command for one
            dataset.  On the cluster prefer the matching job file in
            ``experiments/jobs/``.

The verify stage is the guard rail for the corrected-EG campaign.  Result
JSONs written before 2026-07-23 15:14 carry no ``eg_graph_semantics`` key
because ``core.statistics.create_experiment_info`` did not yet emit it; their
Degree- and PageRank-EG columns were computed by the superseded metric.  A
table whose sources fail ``verify`` must not be pasted into the manuscript.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Callable, Sequence

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(REPO, "results_repro")
TABLES_DIR = os.path.join(RESULTS, "tables")
PLOTS_DIR = os.path.join(REPO, "analysis", "plots")
DATA_PATH = os.path.join(REPO, "data", "SNAP")

#: Corrected post-recommendation graph semantics (metrics/equality_gain.py,
#: rewritten 2026-07-23 15:32).  Result JSONs lacking this marker predate the
#: rewrite and are not on the Section 3.3 metric.
EG_SEMANTICS = "simple_undirected_set_union"

DATASETS = ["ego-facebook", "oklahoma", "unc", "pokec_n", "brightkite"]
#: Datasets shipping a binary sensitive attribute (gender / region).
ATTR_DATASETS = ["oklahoma", "unc", "pokec_n"]
#: Datasets large enough to need the reduced training-time validation protocol.
LARGE_DATASETS = ["pokec_n", "brightkite"]

#: Published CAFE configurations (Pareto + ASF selection, ASF weights
#: 0.5/0.3/0.1/0.1 over Precision@10, Degree/PageRank/Closeness EG -- see
#: analysis/asf_weight_sensitivity.py).  Mirrors analysis/best_configs.csv and
#: the jobs/repro/*.job case statements; ``check_published_configs()`` asserts
#: the two stay in step.
PUBLISHED = {
    # dataset: (cafe_in lambda_fair, beta_ra, lambda_mob,
    #           cafe_pre alpha, beta_ra, lambda_mob, fusion beta)
    "ego-facebook": dict(in_lambda=0.3, in_beta=1.0, in_lam=0.5,
                         pre_alpha=0.05, pre_beta=0.5, pre_lam=0.3, fusion_beta=1.0),
    "oklahoma": dict(in_lambda=0.3, in_beta=5.0, in_lam=0.5,
                     pre_alpha=1.0, pre_beta=0.5, pre_lam=0.5, fusion_beta=5.0),
    "unc": dict(in_lambda=1.0, in_beta=5.0, in_lam=0.5,
                pre_alpha=1.0, pre_beta=1.0, pre_lam=0.5, fusion_beta=5.0),
    "pokec_n": dict(in_lambda=0.05, in_beta=1.0, in_lam=0.1,
                    pre_alpha=0.05, pre_beta=2.0, pre_lam=0.5, fusion_beta=5.0),
    "brightkite": dict(in_lambda=0.05, in_beta=5.0, in_lam=0.1,
                       pre_alpha=0.3, pre_beta=2.0, pre_lam=0.5, fusion_beta=5.0),
}

#: Single-component ablation lambdas (Table 7), from analysis/best_configs.csv.
PUBLISHED_SINGLE = {
    "ego-facebook": dict(ci_in=0.05, ci_pre=1.0, mb_post=0.1),
    "oklahoma": dict(ci_in=1.0, ci_pre=1.0, mb_post=0.3),
    "unc": dict(ci_in=0.05, ci_pre=0.5, mb_post=1.0),
    "pokec_n": dict(ci_in=0.1, ci_pre=0.3, mb_post=0.1),
    "brightkite": dict(ci_in=0.1, ci_pre=0.1, mb_post=0.1),
}

DEFAULT_RUNS = 10


# --------------------------------------------------------------------------
# specs
# --------------------------------------------------------------------------
@dataclass
class Source:
    """One family of result artifacts a table depends on."""

    name: str
    #: glob relative to REPO; ``{ds}`` is substituted per dataset.
    pattern: str
    datasets: Sequence[str] = field(default_factory=lambda: list(DATASETS))
    #: JSON sources are checked for repetition count and EG semantics.
    kind: str = "json"
    min_runs: int = DEFAULT_RUNS
    #: Set False for artifacts whose EG columns are not on the §3.3 metric by
    #: construction (e.g. per-node exposure counts, path statistics).
    eg_sensitive: bool = True
    note: str = ""


@dataclass
class TableSpec:
    key: str
    number: str
    title: str
    section: str
    sources: Sequence[Source] = field(default_factory=list)
    #: argv lists, executed from REPO.
    build: Sequence[Sequence[str]] = field(default_factory=list)
    #: dataset -> argv, the runner that produces the sources.
    runner: Callable[[str], Sequence[str]] | None = None
    artifacts: Sequence[str] = field(default_factory=list)
    notes: str = ""


# --------------------------------------------------------------------------
# helpers shared by the runner builders
# --------------------------------------------------------------------------
def large_dataset_flags(ds: str) -> list[str]:
    """Reduced training-time validation protocol for the two large graphs."""
    extra: list[str] = []
    if ds in LARGE_DATASETS:
        extra += ["--eval_skip_test", "True", "--eval_max_nodes", "5000",
                  "--eval_steps", "25", "--patience", "6"]
    if ds == "brightkite":
        extra += ["--split_file", os.path.join(DATA_PATH, "brightkite", "temporal_split.pt")]
    return extra


def common_train_flags(ds: str) -> list[str]:
    """Flags accepted by every runner via ``cafe_methods.add_common_args``.

    Deliberately excludes ``--runs``: run_cafe / run_coreness_inverse /
    run_mobility_boost / run_stratified_cafe / run_exposure_acceptance define
    it themselves, while run_significance takes ``--num_seeds`` instead.
    """
    return ["--data_name", ds, "--data_path", DATA_PATH,
            "--encoder", "HLGNN", "--predictor", "MLP",
            "--epochs", "500", "--eval_steps", "10", "--patience", "50",
            "--batch_size", "16384", "--lr", "0.001"]


def eg_flags() -> list[str]:
    return ["--use_equality_gain", "True", "--use_centrality_gain", "True",
            "--centrality_types", "pagerank,closeness"]


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------
def _json_files(pattern: str) -> list[str]:
    return [p for p in sorted(glob.glob(os.path.join(REPO, pattern)))
            if not p.endswith(".partial.json")]


def _inspect_json(path: str) -> tuple[str | None, int | None]:
    """Return (eg_graph_semantics, num_runs) from a result JSON."""
    try:
        with open(path) as fh:
            info = json.load(fh).get("experiment_info", {})
    except (OSError, ValueError):
        return None, None
    return info.get("eg_graph_semantics"), info.get("num_runs")


def verify(spec: TableSpec, strict: bool = True) -> int:
    """Check every source backing ``spec``.  Returns the number of problems."""
    problems = 0
    eg_checked = 0
    print(f"Table {spec.number} -- {spec.title}  ({spec.section})")
    for src in spec.sources:
        print(f"\n  source: {src.name}   [{src.kind}]")
        if src.note:
            print(f"    note: {src.note}")
        for ds in src.datasets:
            pattern = src.pattern.format(ds=ds)
            paths = (_json_files(pattern) if src.kind == "json"
                     else sorted(glob.glob(os.path.join(REPO, pattern))))
            if not paths:
                print(f"    {ds:14s} MISSING            {pattern}")
                problems += 1
                continue
            if src.kind != "json":
                print(f"    {ds:14s} ok ({len(paths)} file(s))")
                continue
            for path in paths:
                sem, runs = _inspect_json(path)
                flags = []
                if src.eg_sensitive:
                    eg_checked += 1
                if src.eg_sensitive and sem != EG_SEMANTICS:
                    flags.append(f"EG-SEMANTICS={sem or 'absent (pre-2026-07-23 metric)'}")
                if runs is not None and runs < src.min_runs:
                    flags.append(f"runs={runs} < {src.min_runs}")
                status = "ok" if not flags else "  ".join(flags)
                if flags:
                    problems += 1
                print(f"    {ds:14s} {status:60s} {os.path.relpath(path, REPO)}")
    print()
    if problems:
        print(f"  {problems} problem(s).  Do not paste this table until they are cleared.")
    elif eg_checked:
        print(f"  all sources present and complete; {eg_checked} JSON source(s) "
              f"confirmed on the corrected EG metric.")
    else:
        print("  all sources present. NOTE: none of them is a result JSON, so "
              "the EG semantics could NOT be checked here -- confirm from the "
              "upstream table or the job log.")
    return problems if strict else 0


def plan(spec: TableSpec) -> None:
    print(f"Table {spec.number} -- {spec.title}  ({spec.section})")
    if spec.notes:
        print(f"\n{spec.notes.strip()}\n")
    print("  artifacts:")
    for art in spec.artifacts:
        mark = "present" if os.path.exists(os.path.join(REPO, art)) else "absent "
        print(f"    [{mark}] {art}")
    print("\n  required runs:")
    for src in spec.sources:
        print(f"    {src.name:26s} {', '.join(src.datasets)}"
              f"   (n>={src.min_runs})" if src.kind == "json" else
              f"    {src.name:26s} {', '.join(src.datasets)}")
    if spec.runner is not None:
        print("\n  runner (per dataset), e.g.:")
        print("    " + " ".join(spec.runner(spec.sources[0].datasets[0])))
    print("\n  build:")
    for cmd in spec.build:
        print("    " + " ".join(cmd))


def build(spec: TableSpec) -> int:
    if not spec.build:
        print(f"Table {spec.number} has no build step (definitional or manual).")
        return 0
    for cmd in spec.build:
        print("+ " + " ".join(cmd))
        rc = subprocess.call(cmd, cwd=REPO)
        if rc != 0:
            print(f"build step failed with exit code {rc}")
            return rc
    return 0


def run(spec: TableSpec, dataset: str, execute: bool) -> int:
    if spec.runner is None:
        print(f"Table {spec.number} has no runner; it is built from other tables' results.")
        return 0
    cmd = list(spec.runner(dataset))
    print("+ " + " ".join(cmd))
    if not execute:
        print("\n(dry run -- pass --execute to launch here, or submit the matching "
              "job in experiments/jobs/)")
        return 0
    return subprocess.call(cmd, cwd=os.path.join(REPO, "experiments"))


def check_published_configs() -> int:
    """Assert PUBLISHED matches the authoritative best_configs.csv.

    ``results_repro/tables/best_configs.csv`` is authoritative: it is written
    by a selection over the complete post-bugfix tree.  ``analysis/best_configs.csv``
    is a by-product of whatever tree was aggregated last and goes stale
    silently -- running the generator against the partial ``results/`` tree
    overwrites it with a three-dataset table.  Check the former; warn if the
    latter disagrees.
    """
    import csv

    path = os.path.join(TABLES_DIR, "best_configs.csv")
    if not os.path.exists(path):
        print(f"{os.path.relpath(path, REPO)} not found; skipping cross-check.")
        return 0
    stale = os.path.join(REPO, "analysis", "best_configs.csv")
    if os.path.exists(stale):
        with open(stale) as fh:
            n = sum(1 for _ in fh) - 1
        if n < 100:
            print(f"  NOTE: analysis/best_configs.csv holds only {n} rows -- it "
                  f"was last written from a partial results tree. "
                  f"{os.path.relpath(path, REPO)} is authoritative.")
    # Config keys use Python's float repr (1.0, not 1), matching the runners.
    want = {}
    for ds, cfg in PUBLISHED.items():
        want[("CAFE-In", ds)] = (
            f"lambda_fair_{float(cfg['in_lambda'])}_beta_{float(cfg['in_beta'])}"
            f"_lam_{float(cfg['in_lam'])}")
        want[("CAFE-Pre", ds)] = (
            f"preproc_alpha_{float(cfg['pre_alpha'])}_beta_{float(cfg['pre_beta'])}"
            f"_lam_{float(cfg['pre_lam'])}")
    bad = 0
    with open(path) as fh:
        rows = {(r["Method"], r["Dataset"]): r["Config_Key"] for r in csv.DictReader(fh)}
    for key, expected in want.items():
        got = rows.get(key)
        if got is None:
            print(f"  {key}: absent from {os.path.relpath(path, REPO)}")
            bad += 1
        elif got != expected:
            print(f"  {key}: PUBLISHED={expected}  csv={got}")
            bad += 1
    print("  configs match best_configs.csv" if not bad else f"  {bad} mismatch(es)")
    return bad


# --------------------------------------------------------------------------
def main(spec: TableSpec, argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=f"Table {spec.number}: {spec.title}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=spec.notes)
    p.add_argument("--stage", choices=["plan", "verify", "build", "run"], default="plan")
    p.add_argument("--dataset", default=None, help="dataset for --stage run")
    p.add_argument("--execute", action="store_true",
                   help="actually launch the runner instead of printing it")
    p.add_argument("--lenient", action="store_true",
                   help="verify: report problems but exit 0")
    args = p.parse_args(argv)

    if args.stage == "plan":
        plan(spec)
        return 0
    if args.stage == "verify":
        return verify(spec, strict=not args.lenient)
    if args.stage == "build":
        return build(spec)
    if not args.dataset:
        p.error("--stage run requires --dataset")
    return run(spec, args.dataset, args.execute)


if __name__ == "__main__":  # pragma: no cover
    print(__doc__)
    sys.exit(0)
