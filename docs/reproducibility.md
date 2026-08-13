# Reproducibility protocol

## Frozen scientific behavior

This cleanup is intentionally non-scientific. It does not alter model,
training, evaluation, metric, or analysis Python files. The certified results
use the legacy Equality-Gain definition identified in their JSON metadata as:

```text
legacy_indegree_reconstructed
```

The frozen configuration source is `configs/best_configs.csv`, SHA-256
`44ea8142fa6e953dc2c79fbe3d5b6ae5e4eb77ad08a3a72636db31942e9c4ade`.

## Certified outputs

`results_legacy_eg_check` contains eight certified JSON files:

- CAFE-In and CAFE-Pre for Ego-Facebook;
- CAFE-In and CAFE-Pre for Oklahoma;
- CAFE-In and CAFE-Pre for UNC;
- CI-Pre and MB-Post diagnostics for Pokec-n.

Historical staging directories previously contained 150 absolute symlinks to
ignored local result trees. Those links could never resolve in a clone. Their
portable repository-relative targets and available file checksums are retained
in `docs/legacy-result-links.tsv`; the cached target files themselves are not
deleted by this cleanup.

Each file records ten runs. The CAFE result files record model seeds 42 through
51, 500 epochs, batch size 16,384, learning rate 0.001, and mean penalty
reduction.

Run `make verify-certified` before and after any repository maintenance. A
checksum failure means the release evidence has changed and the cleanup must
not be merged.

## Public reproducibility subset

Ego-Facebook, Oklahoma, and UNC are included. Pokec-n and Brightkite remain
paper datasets but are not distributed in the GitHub repository. The included
subset demonstrates both graph-only and sensitive-attribute workflows while
keeping the clone practical.

## Smoke protocol

`make smoke` runs the deterministic Resource Allocation baseline on Oklahoma:

- CPU only;
- one run;
- fixed 85/5/10 split produced by the existing loader;
- full non-neighbour ranking;
- K=10 headline metrics;
- legacy Degree EG plus the currently implemented PageRank and Closeness EG.

The smoke test verifies installation, data loading, candidate generation, and
metric serialization. It is not a replacement for the ten-seed GPU CAFE runs.

Observed reference values on 2026-08-13 were Precision@10 0.197413, Degree
EG@10 5.263598, PageRank EG@10 7.971036, and Closeness EG@10 12.041178. The
result JSON should report `legacy_indegree_reconstructed`.

## Environments

- `environment-smoke.yml` is the CPU environment for reviewer setup.
- `environment.yml` is the full historical GPU-capable environment.

The full environment is intentionally retained for provenance. The smoke
environment is narrower and does not claim to support every GPU experiment.
Its PyG extension packages use the official PyG CPU wheels so the environment
also works on GLIBC 2.28 systems.

## Safety and merging

Cleanup work is performed on `repo-cleanup`. `repo-cleanup-safety-base` points
to the untouched starting commit. `main` is not modified during cleanup.

Before proposing a merge:

1. run `make verify-certified`;
2. run `make verify-data`;
3. run `make smoke` in the smoke environment;
4. confirm no scientific Python, shell, or Slurm job file differs from the
   safety base;
5. review the commit diff;
6. merge only after explicit author approval.
