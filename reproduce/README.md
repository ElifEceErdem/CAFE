# Reproducing the paper's tables and figures

`python -m reproduce` rebuilds every table and figure of
`paper/CAFE_ToRS_EceErdem.pdf` from recorded result files and compares every
printed number with the manuscript.

```bash
python -m reproduce list                  # artifacts, status, jobs
python -m reproduce build  [T04 F03 ...]  # write build/<ID>.{csv,tex,pdf}
python -m reproduce verify [T04 ...]      # compare build/ with the manuscript
python -m reproduce check  [T04 ...]      # build + verify; non-zero exit on a regression
```

## How it works

```text
evidence/ (recorded results)  ──►  reproduce/artifacts/<id>.py  ──►  build/<ID>.csv/.tex/.pdf
reproduce/sources/ (row → files)                                          │
reproduce/expected/ (numbers printed in the PDF) ─────────────►  reproduce/verify.py
```

* **Builders** (`artifacts/`) recompute every summary from per-run values
  (means, population SDs, paired t-intervals, Holm corrections); nothing is
  copied from a cached table.  Each writes a tidy CSV (one row per printed
  cell) and a booktabs LaTeX table.
* **Source indexes** (`sources/`) name, for every row of Tables 4, 7, 11 and
  15, the result files, configuration key and number of runs behind it.
* **Expected values** (`expected/`) are the 1,612 numbers printed in the
  PDF, extracted by `tools/extract_paper_values.py` (`make test` checks that a
  fresh extraction is identical).
* **Verification** (`verify.py`) rounds each reproduced value to the printed
  precision and compares.  Every cell is `ok`, `known` (difference recorded in
  `known_deviations.csv` with its reason), `fail` or `missing`.
* **Evidence** (`../evidence/`) holds the 240 result files the builders read
  (5.5 MB, SHA-256 in `evidence/MANIFEST.sha256`).  `CAFE_EVIDENCE_ONLY=1`
  proves a build needs nothing else.

## Artifacts

| ID | Content | Builder | Status | Job |
|---|---|---|---|---|
| T01 | Positioning against related work | static | hand-authored | – |
| T02 | Dataset statistics | `t02_datasets` | reproducible (2 cells printed truncated) | `t02_datasets.job` |
| T03 | CAFE configurations | static | definitional | – |
| T04 | Precision and EG, all methods | `t04_main_results` | reproducible (1 known cell) | `t04_main_results.job` |
| T05 | Realized coreness bins | `t05_coreness_bins` | reproducible | `t05_t18_stratified.job` |
| T06 | CAFE-In vs fusion: localization | `t06_localization` | (b) percentages reproducible; (a) and absolute changes need the job | `t06_localization.job` |
| T07 | Ablations | `t07_ablation` | reproducible (Pokec-n fusion row, 1 Ego-FB cell known) | `t07_ablation.job` |
| T08 | Demographic fairness | `t08_demographic` | reproducible (10 known precision cells) | `t08_demographic.job` |
| T09 | Structural conditions, paired ΔSP | `t09_structure` | reproducible | `t08_demographic.job`, `t06_localization.job` |
| T10 | Mobility-boost closeness probe | `t10_closeness_probe` | reproducible | `t10_closeness_probe.job` |
| T11 | EG under acceptance models | `t11_acceptance` | reproducible (2 spliced rows, truncated TP cells known) | `t11_acceptance.job` |
| T12 | Hyperparameter grids | static | definitional | – |
| T13 | Runtime | `t13_runtime` | reproducible (Pokec-n row known) | `t13_runtime.job` |
| T14 | Comparison families | static | definitional | – |
| T15 | Paired contrasts | `t15_paired_contrasts` | reproducible (all 120 contrasts and stars) | `t15_paired_contrasts.job` |
| T16 | ΔSP/ΔEO under both protocols | `t16_demographic_protocol` | reproducible | `t08_demographic.job` |
| T17 | Brightkite random-split control | `t17_brightkite_random` | needs rerun (results not recorded) | `t17_brightkite_random.job` |
| T18 | Bin-size, degree, clustering controls | `t18_bin_controls` | reproducible | `t05_t18_stratified.job`, `t18_structural_controls.job` |
| T19 | Conditional logit coefficients | `t19_conditional_logit` | reproducible | `t19_conditional_logit.job` |
| T20 | K=5 / K=20 sensitivity | `t20_k_sensitivity` | reproducible | `t20_k_sensitivity.job` |
| F01 | UNC coreness diagnosis | `f01_diagnosis` | rebuilt from current runs (the printed render predates them) | `f01_diagnosis.job` |
| F02 | Pipeline diagram | static (cropped from the PDF) | hand-drawn | – |
| F03 | Precision vs EG Pareto panels | `f03_pareto` | reproducible (same points as T04/T07) | `t04_main_results.job` |
| F04 | Source-coreness profiles | `f04_stratified` | reproducible | `t05_t18_stratified.job` |

Figures are not compared numerically; F03 is drawn from exactly the values
verified in Tables 4 and 7.

### Known deviations (`known_deviations.csv`)

* **Row splices in the manuscript**: Ego-FB CAFE-In Degree EG is printed as
  11.64 in Tables 4, 7 and 11 (its run gives 11.76; 11.64 is from a later
  acceptance rerun); Table 7 Pokec-n HL-GNN+RA prints the precision of the
  β=5 run with the EG of the β=1 run; Table 11 Ego-FB CAFE-In and UNC CAFE-Pre
  print Table 4's Full values next to another run's PW/TP values; eight Table 8
  rows print the Table 4 precision mean next to the July run's SD, ΔSP and ΔEO.
* **Printing**: two Table 2 cells and several Table 11 TP cells are truncated
  rather than rounded.
* **Not on disk**: the Table 13 Pokec-n timings and the Table 6(b) absolute
  exposure changes (and its Pokec-n peripheral percentage).

## Re-running experiments

Each job re-runs one artifact's experiment into a fresh directory, indexes the
outputs (`tools/index_runs.py`), rebuilds the artifact from them
(`CAFE_EVIDENCE=<out> CAFE_SOURCES=<out>/sources`) and verifies it.  Rows not
re-run keep coming from the snapshot, so partial reruns (one dataset, one arm)
verify too.

```bash
cd /path/to/CAFE                                   # submit from the repository root
reproduce/jobs/submit.sh t07_ablation              # all datasets, then build + verify
sbatch --array=1 --export=ALL,ARMS=ci-in reproduce/jobs/t07_ablation.job   # one arm
DRY_RUN=1 SLURM_ARRAY_TASK_ID=3 bash reproduce/jobs/t04_main_results.job   # print commands
```

* Configurations come from `python -m reproduce.config <dataset>`, derived
  from the recorded result files (identical to `configs/best_configs.csv`).
* Pokec-n and Brightkite use exact generalized Closeness
  (`experiments/run_with.py --exact-closeness`); Table 15 trains seed s on
  split s (`--split-seed`).
* GPU training is not bit-reproducible across hardware.  Compare reruns with a
  tolerance in printed units, e.g. `TOLERANCE=5` accepts ±0.05 EG and
  ±0.0005 precision; runtimes (Table 13) default to ±15.
* Tables 17, 18(b), 19 and 20 are produced by self-contained campaign scripts
  (`analysis/finalize_brightkite_random_control.py`,
  `analysis/rq1_structural_controls.py`, `analysis/diagnosis_conditional_logit.py`,
  `experiments/run_k_sensitivity.py`) with their own lineage checks; the jobs
  call them with the paper's profile.

## Updating the snapshot

After a new campaign has been accepted, point the source indexes at it and
refresh the snapshot on the machine that holds the raw results:

```bash
python -m reproduce.tools.freeze_evidence        # copy declared inputs, rewrite the manifest
python -m reproduce check && make test
```

When the manuscript changes, re-extract its numbers with
`python -m reproduce.tools.extract_paper_values`.
