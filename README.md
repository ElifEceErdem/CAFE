# CAFE: Coreness-Aware Fair Link Recommendation

This repository accompanies the ToRS submission **Coreness-Aware Fair Link
Recommendation**. CAFE combines an HL-GNN link predictor with coreness-aware
pre-processing, in-processing, and ranking interventions.

## Release scope

This release deliberately preserves the legacy experimental implementation.
The scientific Python code and the reported values have not been rewritten as
part of the repository cleanup.

- The supported public-data subset is **Ego-Facebook, Oklahoma97, and UNC28**.
- The full paper also evaluates Pokec-n and Brightkite. Those datasets are too
  large to distribute through GitHub and are not required for the public smoke
  test.
- Equality Gain in the certified result files uses the legacy
  `legacy_indegree_reconstructed` definition.
- The only currently certified experiment outputs are the eight immutable JSON
  files under [`results_legacy_eg_check/`](results_legacy_eg_check/).
- The exact configuration source used for those runs is
  [`configs/best_configs.csv`](configs/best_configs.csv).
- The public manuscript is
  [`paper/CAFE_ToRS_EceErdem.pdf`](paper/CAFE_ToRS_EceErdem.pdf).

The repository contains additional historical campaigns and analysis outputs.
They are retained for provenance, but they are not designated as certified
release results.

## Quick start: Oklahoma smoke test

The smoke test uses the included Oklahoma graph, runs Resource Allocation on
CPU, evaluates the legacy metrics, and writes only to `/tmp`.

```bash
git clone <repository-url>
cd CAFE
conda env create -f environment-smoke.yml
conda run -n cafe-smoke make smoke
```

On the audited workstation the experiment itself completed in approximately
31 seconds, used about 627 MiB of memory, and required no GPU. Expected
Oklahoma RA values are approximately:

| Metric | Expected smoke value |
|---|---:|
| Precision@10 | 0.1974 |
| Degree EG@10 | 5.26 |
| PageRank EG@10 | 7.97 |
| Closeness EG@10 | 12.04 |

Small floating-point differences across platforms are acceptable. A successful
run creates a timestamped JSON below `/tmp/cafe-oklahoma-smoke/oklahoma/`.

If an existing compatible environment is already active, run:

```bash
make smoke
```

## Verify the certified release files

```bash
make verify-certified
```

This verifies the paper PDF, the exact `best_configs.csv`, and every certified
legacy result against checked-in SHA-256 hashes. It does not rerun the
ten-seed GPU experiments.

## Data

The repository includes the data needed for the three-dataset public subset:

```text
data/SNAP/
├── ego-facebook/
├── oklahoma/
└── unc/
```

Oklahoma97 and UNC28 can also be restored with
[`data/SNAP/download_oklahoma_unc.py`](data/SNAP/download_oklahoma_unc.py).
See [`data/README.md`](data/README.md) for layout, provenance, and the large
dataset policy.

## Repository layout

```text
CAFE/
├── backbone/                 # HL-GNN backbone
├── baselines/                # comparison methods
├── core/                     # data loading, training, and evaluation
├── fairness_methods/         # CAFE interventions
├── metrics/                  # recommendation and fairness metrics
├── experiments/              # experiment and artifact entry points
├── analysis/                 # table/figure analysis code
├── configs/                  # frozen release configurations and hashes
├── data/                     # included public reproducibility subset
├── results/                  # generated local results (normally ignored)
├── results_legacy_eg_check/  # certified immutable outputs
├── paper/                    # current public paper only
├── docs/                     # reproducibility and coverage documentation
├── scripts/                  # entry-point index; legacy paths are preserved
├── notebooks/                # exploratory-work policy
├── src/                      # migration note for the preserved legacy layout
├── environment-smoke.yml     # minimal CPU smoke environment
└── environment.yml           # full historical experiment environment
```

The source packages remain at their historical top-level paths so that the
certified experiment commands and imports are not changed during cleanup.

## Full experiments

The main runners are:

- [`experiments/run_baselines.py`](experiments/run_baselines.py): heuristic,
  GNN, tail-node, and attribute-aware baselines.
- [`experiments/run_cafe.py`](experiments/run_cafe.py): CAFE-In and CAFE-Pre.
- [`experiments/run_cafe_p.py`](experiments/run_cafe_p.py): post-processing and
  HL-GNN controls.
- [`experiments/run_coreness_inverse.py`](experiments/run_coreness_inverse.py)
  and [`experiments/run_mobility_boost.py`](experiments/run_mobility_boost.py):
  component ablations.
- [`experiments/run_exposure_acceptance.py`](experiments/run_exposure_acceptance.py):
  acceptance-model analysis.

Full CAFE runs use GPU resources and are not part of the under-ten-minute smoke
path. The table-oriented Slurm entry points are indexed in
[`experiments/jobs/README.md`](experiments/jobs/README.md).

## Paper artifacts and limitations

The mapping from paper artifacts to entry points, including currently
uncertified gaps, is documented in
[`docs/artifact-coverage.md`](docs/artifact-coverage.md). Reproduction protocol,
result status, and metric semantics are documented in
[`docs/reproducibility.md`](docs/reproducibility.md).

In particular, the cleanup does not claim that every historical result tree is
interchangeable. Tables 4, 7, 8, and 11 remain tied to the legacy implementation
and should not be regenerated with different metric semantics without an
explicit new experiment campaign.

## License and citation

The code is released under the [MIT License](LICENSE). Dataset redistribution
terms remain those of each original dataset provider.

Citation metadata is available in [`CITATION.cff`](CITATION.cff).
