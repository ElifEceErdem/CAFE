# CAFE: Coreness-Aware Fair Link Recommendation

Code, recorded results and reproduction pipeline for the ToRS paper
**Coreness-Aware Fair Link Recommendation**
([`paper/CAFE_ToRS_EceErdem.pdf`](paper/CAFE_ToRS_EceErdem.pdf)).
CAFE combines an HL-GNN link predictor with coreness-aware pre-processing,
in-processing and ranking interventions.

## Reproduce the paper's tables and figures

Every table and figure of the paper is rebuilt from recorded results and every
printed number is checked against the manuscript:

```bash
conda env create -f environment-smoke.yml        # CPU environment
conda activate cafe-smoke
make verify-paper                                # builds build/, compares 1,612 printed values
```

`make verify-paper` needs no GPU and no raw data: the builders read the
checksummed result snapshot in [`evidence/`](evidence/).  Output:

```text
ID      ok  known  fail  missing  verdict
T04    511      1     0        0  PASS
T07    438      7     0        0  PASS
...
```

*known* cells are documented differences between a printed value and its
recorded source (see [`reproduce/known_deviations.csv`](reproduce/known_deviations.csv));
*missing* cells belong to the two results that must be re-run (Table 6(a),
Table 17).  The artifact map, the verification rules and how to re-run each
experiment are in [`reproduce/README.md`](reproduce/README.md).

To re-run the experiment behind one table on a Slurm cluster and verify the
fresh numbers:

```bash
reproduce/jobs/submit.sh t04_main_results      # run array + build/verify stage
```

## Quick start: Oklahoma smoke test

Runs Resource Allocation on the included Oklahoma graph on CPU (about 30 s):

```bash
make smoke
```

| Metric | Expected value |
|---|---:|
| Precision@10 | 0.1974 |
| Degree EG@10 | 5.26 |
| PageRank EG@10 | 7.97 |
| Closeness EG@10 | 12.04 |

## Repository layout

```text
CAFE/
├── backbone/          HL-GNN backbone
├── baselines/         comparison methods
├── core/              data loading, training, evaluation
├── fairness_methods/  CAFE interventions (coreness penalty, re-weighting, mobility boost)
├── metrics/           precision, Equality Gain (degree, PageRank, closeness), demographic gaps
├── experiments/       experiment runners (run_*.py) and run_with.py (evaluation-protocol launcher)
├── analysis/          analysis runners behind Tables 2, 5, 9, 10, 17-19 and Figures 1, 4
├── reproduce/         table/figure builders, verification, source indexes, Slurm jobs
├── evidence/          checksummed snapshot of the result files behind the paper
├── configs/           published configurations and release checksums
├── data/              public datasets (Ego-Facebook, Oklahoma97, UNC28)
├── results_legacy_eg_check/  certified release outputs (hash-locked)
├── paper/             the manuscript
├── docs/              reproducibility protocol and audit
└── tests/             pipeline tests (make test)
```

## Data

Ego-Facebook, Oklahoma97 and UNC28 are included under `data/SNAP/`.  Pokec-n
and Brightkite are evaluated in the paper but too large to distribute; the
jobs expect them under `data/SNAP/pokec_n` and `data/SNAP/brightkite` (see
[`data/README.md`](data/README.md)).  Tables and figures still rebuild without
them, from `evidence/`.

## Experiments

The runners in `experiments/` train and evaluate one method family each:
`run_baselines.py` (heuristics, GNN, fairness and tail-node baselines),
`run_cafe.py` (CAFE-In, CAFE-Pre, fusion control, partial pipelines),
`run_cafe_p.py` (HL-GNN backbone, CAFE^-P), `run_coreness_inverse.py` and
`run_mobility_boost.py` (single-component ablations), `run_significance.py`
(split-resampled protocol), `run_exposure_acceptance.py` (exposure and
acceptance models), `run_runtime_benchmark.py`, `run_k_sensitivity.py`.
`experiments/run_with.py` adds the two evaluation options used for the paper
(exact Closeness on the large graphs, split-seed injection).  The jobs in
`reproduce/jobs/` contain the exact command lines.

Equality Gain follows the paper's released definition (result marker
`legacy_indegree_reconstructed`); see [`docs/reproducibility.md`](docs/reproducibility.md).

## Verification targets

| Target | Checks |
|---|---|
| `make verify-paper` | every table rebuilt from `evidence/` and compared with the manuscript |
| `make verify-evidence` | SHA-256 of every evidence file |
| `make verify-certified` | manuscript, configuration file and certified outputs |
| `make verify-data` | processed public datasets |
| `make test` | pipeline tests |
| `make check-release` | all of the above except the tests |

## License and citation

Code: [MIT License](LICENSE).  Dataset redistribution terms are those of the
original providers.  Citation metadata: [`CITATION.cff`](CITATION.cff).
