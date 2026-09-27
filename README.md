# CAFE: Coreness-Aware Fair Link Recommendation

Code and results for the paper *Coreness-Aware Fair Link Recommendation*.
CAFE combines an HL-GNN link predictor with coreness-aware re-weighting
(CAFE-Pre), a coreness penalty in the training loss (CAFE-In), fusion with the
Resource Allocation heuristic and a mobility boost for peripheral targets.

## Repository layout

```text
backbone/ baselines/ core/ fairness_methods/ metrics/   model, baselines and metrics
experiments/   scripts that train, evaluate and summarize (produce results)
tables/        one script per table and figure of the paper (read results)
results/       the results the paper's tables are built from
configs/       selected configurations
data/          public datasets (Ego-Facebook, Oklahoma97, UNC28)
tests/
```

## Installation

```bash
conda env create -f environment.yml          # full GPU environment (experiments)
conda env create -f environment-smoke.yml    # CPU environment (tables, tests, smoke test)
```

## Building the tables and figures

```bash
python -m tables              # every table and figure -> build/
python -m tables T04 F03      # selected ones
python -m tables --list       # what each script needs
```

No GPU or raw data is needed: every number is recomputed from the per-run
results in [`results/`](results/) (its README lists which files feed which
table).  Output: `build/T04.tex`, …, `build/F03.pdf`.  Tables 1, 3, 12 and 14
and Figure 2 are hand-authored.  Table 6(a) and Table 17 need experiment
outputs that are not part of `results/` (see below).

## Re-running the experiments

Training requires a GPU.  Pokec-n and Brightkite are not distributed (see
[`data/README.md`](data/README.md)); the other datasets are in `data/SNAP/`.
All runs use the configurations in [`configs/best_configs.csv`](configs/best_configs.csv),
training seeds 42-51 on a fixed 85/5/10 split, K=10, 500 epochs, batch size
16384 and learning rate 1e-3.  Every script documents its options with `--help`.

```bash
cd experiments
DATA=$PWD/../data/SNAP
COMMON="--data_path $DATA --encoder HLGNN --predictor MLP --epochs 500 --eval_steps 10 \
        --patience 50 --batch_size 16384 --lr 0.001 --runs 10 --coreness_alpha 1.0 \
        --use_equality_gain True --use_centrality_gain True"

# Baselines (Table 4): heuristics, GNN, fairness and tail-node methods
python run_baselines.py --data_name oklahoma --data_path $DATA --encoder HLGNN --runs 10 --seed 42 \
    --epochs 300 --batch_size 16384 --use_equality_gain True --use_centrality_gain True \
    --centrality_types pagerank,closeness --results_dir ../outputs \
    --methods random,pa,cn,aa,ra,lightgcn,ssnet,cross_link_bias,ltlp,tailsteak,fairadj,lba,moral

# HL-GNN backbone (no fusion, no boost)
python run_cafe_p.py --data_name oklahoma $COMMON --beta_ra 0.0 --lambda_mobility 0.0 \
    --results_file ../outputs/oklahoma/hl-gnn.json

# CAFE-In and CAFE-Pre (Oklahoma configurations from configs/best_configs.csv)
python run_cafe.py --data_name oklahoma $COMMON --centrality_types pagerank,closeness \
    --penalization_mode inproc --use_mobility_boost True \
    --lambda_fair 0.3 --beta_ra 5.0 --lambda_mobility 0.5 --results_file ../outputs/oklahoma/cafe-in.json
python run_cafe.py --data_name oklahoma $COMMON --centrality_types pagerank,closeness \
    --penalization_mode preproc --use_mobility_boost True \
    --preproc_alpha 1.0 --beta_ra 0.5 --lambda_mobility 0.5 --results_file ../outputs/oklahoma/cafe-pre.json
```

Protocol options:

* **Pokec-n and Brightkite.** For `run_cafe.py` add `--eval_skip_test True
  --eval_max_nodes 5000 --eval_steps 25 --patience 6`; for Brightkite pass its
  temporal split, `--split_file $DATA/brightkite/temporal_split.pt`.  Evaluate
  Closeness exactly by starting the runner through `run_with.py`:
  `python run_with.py --exact-closeness run_cafe.py --data_name pokec_n ...`.
* **Split-resampled protocol (Table 15).** Train seed *s* on split *s*:
  `python run_with.py --split-seed 3 run_cafe.py ... --runs 1 --seed 3`.
* **Acceptance models (Table 11).** Add `--acceptance_probe_jsonl <file>` to a runner.

| Artifact | Experiment scripts (`experiments/`) |
|---|---|
| Table 2 | `dataset_statistics.py` |
| Tables 4, 7, Figure 3 | `run_baselines.py`, `run_cafe_p.py`, `run_cafe.py`, `run_coreness_inverse.py`, `run_mobility_boost.py` |
| Tables 5, 18(a,c), Figure 4 | `run_stratified_cafe.py`, then `bin_perturbation_size.py` and `summarize_stratified.py` |
| Table 6 | `run_stratified_cafe.py --runs 10 --include_fusion True`; `run_exposure_acceptance.py`, then `summarize_exposure.py` |
| Tables 8, 9, 16 | `run_baselines.py` (FairAdj, LBA, MORAL), CAFE runners, `summarize_demographic.py`, `run_significance.py`, `coreness_by_group.py` |
| Table 10 | `closeness_mechanism.py`, then `summarize_closeness_probe.py` |
| Table 11 | CAFE runners and `run_baselines.py --methods ra` with `--acceptance_probe_jsonl` |
| Table 13 | `run_runtime_benchmark.py` |
| Table 15 | runners via `run_with.py --split-seed s`, seeds 0-9 |
| Table 17 | `run_stratified_cafe.py --split_file random --split_seed s --allow_brightkite_random_control True` (s = 42-51), then `finalize_brightkite_random_control.py` |
| Table 18(b) | `rq1_structural_controls.py` |
| Table 19 | `run_precision_by_coreness.py`, then `diagnosis_conditional_logit.py` |
| Table 20 | `run_k_sensitivity.py` |
| Figure 1 | `run_precision_by_coreness.py --data_name unc`, then `summarize_stratified.py` |

To build the tables from your own results, arrange them like `results/` and
run `CAFE_RESULTS=/path/to/results python -m tables`.

## Checks

```bash
make test          # result checksums, every table builds, spot values
make smoke         # Resource Allocation on Oklahoma, CPU, ~30 s
make verify-data   # included datasets
```

## License and citation

Code: [MIT License](LICENSE).  Datasets remain under the terms of their
original providers.  Citation metadata: [`CITATION.cff`](CITATION.cff).
