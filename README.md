# CAFE: Coreness-Aware Fairness Enhancement for Link Prediction

CAFE is a fairness framework for link prediction on social networks. It addresses **degree bias** — the tendency of GNN-based recommenders to over-recommend already well-connected (high-degree/core) nodes at the expense of peripheral nodes.

CAFE uses **HL-GNN** as its backbone encoder and combines it with coreness-based fairness interventions applied at different stages of the pipeline: pre-processing, in-processing, and post-processing.

---

## Repository Structure

```
CAFE/
├── backbone/                   # HL-GNN backbone (from SNAP)
├── core/                       # Pipeline: data loading, training, evaluation, RA models
├── fairness_methods/           # Fairness interventions (in-processing, re-ranking)
├── metrics/                    # Evaluation metrics (coreness, EG, fairness, precision, ...)
├── baselines/                  # Baseline and fairness-aware baseline implementations
├── experiments/                # Experiment runners
│   ├── tables/                 #   one module per manuscript table (plan/verify/build/run)
│   └── jobs/                   #   the SLURM job matching each table
├── analysis/                   # Table generators, post-hoc analyses, plots
├── data/SNAP/                  # Datasets
├── results_repro/tables/       # Generated LaTeX/CSV tables (tracked; the paper artifacts)
└── environment.yml             # Conda environment specification
```

Raw result trees (`results/`, `results_repro/<dataset>/`, `results_egfix/`) and
Slurm logs are reproducible from `experiments/jobs/` and are gitignored.

### File listing

- **[backbone/](backbone/)** — `model.py` (HLGNN encoder + MLP predictor), `layer.py`, `loss.py` (AUC/BCE), `utils.py`, `negative_sample.py`, `logger.py`, `snap_dataset.py`, `snap_evaluator.py`
- **[core/](core/)** — `data_loader.py`, `evaluation.py`, `training.py`, `training_inproc.py` (fairness-aware loop), `statistics.py`, `utils.py`, `model_ra.py`, `model_ra_reg.py`, `ra_features.py`, `ra_sampler.py`
- **[fairness_methods/](fairness_methods/)** — `inprocessing.py` (FairnessAwareModel, coreness-inverse loss, mobility-boost aggregation), `reranking_coreness.py` (CorenessReRanker, used by MB-Post), `moral_integration.py` (MORAL baseline integration)
- **[metrics/](metrics/)** — `coreness.py`, `equality_gain.py`, `centrality_equality_gain.py`, `demographics.py`, `degree_analysis.py`, `precision.py`, `fairness.py`, `pa_aware.py`, `ndkl.py`, `networked_inequality.py`, `classification.py`, `stratified.py`
- **[baselines/](baselines/)** — `baseline_models.py` (CN, RA, AA, PA, LightGCN, SSNet, TailSTEAK, ...), `baseline_wrapper.py`, `fairness_aware_baseline_models.py` (FairAdj, CrossWalk, LBA, MORAL, ...), `tailsteak_dataloader.py`
- **[experiments/](experiments/)** — `run_cafe.py` (CAFE-In / -Pre / -In-B / -Pre-B), `run_cafe_p.py` (CAFE-P, also HLGNN baseline), `run_coreness_inverse.py` (CI-In / CI-Pre, also MB-In / MB-Pre), `run_mobility_boost.py` (MB-Post, also CI-Post), `run_baselines.py` (Table 4/5 baselines: heuristics, LightGCN, SSNet, LTLP, TailSTEAK, Cross-Link Bias, FairAdj, LBA, MORAL), `run_significance.py` (per-seed paired runs)
- **[analysis/](analysis/)** — see [Analysis & Paper Figures](#analysis--paper-figures) below

---

## Method Variants

| Variant | Script | Stage | Mechanism |
|---|---|---|---|
| **CAFE-In** | `run_cafe.py` | In-processing | Loss penalty on core edges + RA hybrid (with mobility boost) |
| **CAFE-Pre** | `run_cafe.py` | Pre-processing | Adjacency modification + RA hybrid (with mobility boost) |
| **CAFE-In-B** | `run_cafe.py` | In-processing | Loss penalty + RA hybrid (no mobility boost) |
| **CAFE-Pre-B** | `run_cafe.py` | Pre-processing | Adjacency modification + RA hybrid (no mobility boost) |
| **CAFE-P** | `run_cafe_p.py` | Post-processing | RA hybrid + mobility-boost re-ranking |
| **CI-In / CI-Pre** | `run_coreness_inverse.py` | In / Pre | Coreness-inverse penalty only |
| **MB-Post** | `run_mobility_boost.py` | Post-processing | Mobility-boost re-ranking only |
| **HLGNN baseline** | `run_cafe_p.py` | — | `--beta_ra 0.0 --lambda_mobility 0.0` |

*Additional variants explored but not in the main paper — **CI-Post**, **MB-In**, **MB-Pre** — are reported in the supplementary ablation (Table 6).*

### Key concepts
- **Coreness**: H-index based structural measure (high = core, low = peripheral).
- **Resource Allocation (RA)**: Heuristic link score favouring low-degree shared neighbours.
- **Mobility Boost**: Inference-time score multiplier promoting peripheral nodes, derived from coreness percentile ranks.
- **Equality Gain (EG)**: Primary fairness metric; higher ⇒ more attention to under-connected nodes, negative ⇒ model amplifies degree bias.

---

## Installation

```bash
conda env create -f environment.yml
conda activate hl_gnn
```

Manual setup: Python 3.10, PyTorch 2.1 + CUDA 11.8, PyTorch Geometric + `pytorch-scatter`/`pytorch-sparse`/`pytorch-cluster`, plus `numpy scipy pandas scikit-learn matplotlib networkx tqdm gensim pymoo python-louvain scikit-network seaborn`.

---

## Data

```
data/SNAP/
├── brightkite/       (gitignored — large, obtain separately)
├── ego-facebook/
├── facebook/         (gitignored)
├── oklahoma/
├── pokec_n/          (gitignored)
├── pokec_z/          (gitignored)
└── unc/
```

Each dataset folder has `raw/` and `processed/`. Brightkite also ships a pre-computed temporal split (`temporal_split.pt`), passed via `--split_file`.

---

## Running Experiments

The example below runs **CAFE-In** on Brightkite. All other variants use the same skeleton with different flags — see the variant → flag reference that follows.

```bash
python experiments/run_cafe.py \
    --data_name brightkite \
    --data_path data/SNAP \
    --split_file data/SNAP/brightkite/temporal_split.pt \
    --encoder HLGNN --predictor MLP \
    --epochs 300 --patience 50 --batch_size 16384 --lr 0.001 --runs 10 \
    --penalization_mode inproc \
    --use_mobility_boost True \
    --lambda_fair 0.05 --beta_ra 5.0 --lambda_mobility 0.1 \
    --coreness_alpha 1.0 \
    --use_equality_gain True --use_centrality_gain True \
    --centrality_types pagerank,closeness \
    --results_file results/cafe_in_brightkite.json
```

### Variant → flag reference

| Variant | Script | Distinguishing flags |
|---|---|---|
| CAFE-In | `run_cafe.py` | `--penalization_mode inproc --use_mobility_boost True --lambda_fair … --beta_ra … --lambda_mobility …` |
| CAFE-Pre | `run_cafe.py` | `--penalization_mode preproc --use_mobility_boost True --preproc_alpha … --beta_ra … --lambda_mobility …` |
| CAFE-In-B | `run_cafe.py` | `--penalization_mode inproc --use_mobility_boost False --lambda_fair … --beta_ra …` |
| CAFE-Pre-B | `run_cafe.py` | `--penalization_mode preproc --use_mobility_boost False --preproc_alpha … --beta_ra …` |
| CAFE-P | `run_cafe_p.py` | `--beta_ra … --lambda_mobility …` |
| HLGNN baseline | `run_cafe_p.py` | `--beta_ra 0.0 --lambda_mobility 0.0` |
| CI-In | `run_coreness_inverse.py` | `--fairness_method coreness_inverse_inproc --lambda_fair …` |
| CI-Pre | `run_coreness_inverse.py` | `--fairness_method coreness_inverse_preproc --lambda_fair …` |
| MB-Post | `run_mobility_boost.py` | `--lambda_mobility … --boosting_mode continuous --reranking_approach boost` |

Best values for `lambda_fair`, `preproc_alpha`, `beta_ra`, and `lambda_mobility` per dataset (Pareto + ASF selection, ASF weights 0.5/0.3/0.1/0.1 over Precision@10 and the three EG dimensions) are in [`results_repro/tables/best_configs.csv`](results_repro/tables/best_configs.csv), and mirrored in [`experiments/tables/_framework.py`](experiments/tables/_framework.py) and [`experiments/jobs/_common.sh`](experiments/jobs/_common.sh) — change all three together. Empty cells indicate the parameter is not applicable to that method.

> `analysis/best_configs.csv` is **not** authoritative. `generate_final_tables.py` rewrites it from whichever results tree it was last pointed at, so it goes stale silently; it is gitignored for that reason.

---

## Reproducing the paper's tables

Experiments are organised **by manuscript table**. Each table has one module in
[`experiments/tables/`](experiments/tables/) declaring what it needs and how it
is built, and one SLURM job in [`experiments/jobs/`](experiments/jobs/).

```bash
# what does Table 6 need, and what is already there?
python -m experiments.tables.table06_localization --stage plan

# are the results backing every table complete and on the current EG metric?
python -m experiments.tables --stage verify --tables all

# rebuild the LaTeX/CSV artifacts from whatever results exist
python -m experiments.tables --stage build --tables 4,7,8

# submit everything, with aggregation chained afterwards
cd experiments/jobs && bash submit_all.sh
```

Array index → dataset: `0=ego-facebook 1=oklahoma 2=unc 3=pokec_n 4=brightkite`.
Submit selectively, e.g. `sbatch --array=0,4 t11_exposure.job`.

| Job | Produces |
|---|---|
| `t04_cafe.job` | Tables 4, 7, 8 — backbone, CAFE⁻ᴾ, fusion control, CAFE-In/Pre (± boost) |
| `t04_baselines.job` | Table 4 — heuristics, LightGCN, SSNet, Cross-Link Bias, LTLP, TailSTEAK, FairAdj/LBA/MORAL |
| `t07_ablation.job` | Table 7 — CI-In, CI-Pre, MB-Post |
| `t05_stratified.job` | Tables 5, 6(a), 18 and Figure 4 |
| `t11_exposure.job` | Tables 11 and 6(b) |
| `t16_significance.job` | Tables 16, 17 (split-resampled paired protocol) |
| `t10_closeness.job` | Table 10 (Ego-FB / Oklahoma / UNC only) |
| `t12_runtime.job` | Table 12 (`--exclusive`; shared GPUs invalidate the timings) |
| `cpu_analyses.job` | Tables 2, 9 (no GPU) |
| `build_tables.job` | verify + build everything |

Tables 1, 3, 13, 14 and 15 are definitional or analytical and have no code path.

### The Equality-Gain semantics contract

`metrics/equality_gain.py` and `metrics/graph_utils.py` were rewritten on
2026-07-23. The old Degree EG counted directed indegree with multiplicity and
PageRank used a non-binarised adjacency; the current metric is the simple
undirected post-recommendation graph of §3.3. Every result JSON written since
records this as

```json
"eg_graph_semantics": "simple_undirected_set_union"
```

`--stage verify` fails any table whose sources lack that marker. **A table
mixing marked and unmarked sources is not comparable within a Degree- or
PageRank-EG column** and must not be pasted into the manuscript. Precision@10,
Closeness EG, ΔSP and ΔEO are unaffected by the change.

---

## Analysis & Paper Figures

Contents of [`analysis/`](analysis/):

Prefer driving these through `experiments/tables/`, which passes the right
paths and checks the sources first. Called directly, they are:

| File | Feeds |
|---|---|
| `generate_final_tables.py` | Pareto + ASF selection → `table1_precision_equality_gain.csv`, `best_configs.csv` |
| `generate_table4.py` | Table 4 |
| `generate_ablation_tables.py`, `generate_merged_ablation_table.py` | Table 7 |
| `generate_table5.py`, `demographic_significance.py` | Table 8 |
| `generate_demographic_appendix_tables.py` | Tables 9, 17 |
| `generate_closeness_stratum_table.py` | Table 10 |
| `generate_acceptance_table.py` | Table 11 (+ the ρ(r) appendix table) |
| `bin_perturbation_size.py`, `generate_bin_perturbation_tables.py` | Tables 5, 18 |
| `generate_dataset_table.py` | Table 2 |
| `generate_runtime_table.py` | Table 12 |
| `generate_localization_table.py` | Table 6 |
| `plot_precision_eg_tradeoff_table_match.py` | Figure 3 |
| `run_stratified_cafe.py` | Tables 5, 6(a), 18, Figure 4 |
| `run_precision_by_coreness.py` | Figure 1 |
| `closeness_mechanism.py` | Table 10 |
| `coreness_by_group.py` | Table 9 |

Figure 3 is a two-stage pipeline: `generate_final_tables.py` must write
`table1_precision_equality_gain.csv` **before** the plot script reads it, and
both should point at `results_repro/tables/`.

---

## Metrics

Primary metrics reported in the paper:

- **Precision@10** — recommendation accuracy
- **Degree EG@10** — equality gain by node degree
- **PageRank EG@10** / **Closeness EG@10** — equality gain by centrality

---

## Table map

Numbering follows the ToRS manuscript. Modules live in `experiments/tables/`.

| Table | Content | Section | Module |
|---|---|---|---|
| 1 | Positioning against related methods | 2.5 | — hand-written |
| 2 | Dataset statistics | 6.1 | `table02_dataset_stats` |
| 3 | CAFE configurations | 6.4 | — definitional |
| 4 | Main results: Precision@10 + 3 EG dimensions | 7.1 | `table04_main_results` |
| 5 | Realized coreness bins | 7.2 | `table05_coreness_bins` |
| 6 | Localization of CAFE-In beyond fusion | 7.2 | `table06_localization` |
| 7 | Ablation across five datasets | 7.3 | `table07_ablation` |
| 8 | Demographic fairness | 7.4 | `table08_demographic` |
| 9 | Structural mechanism of the demographic byproduct | 7.4 | `table09_group_coreness` |
| 10 | Mobility boost, stratum-level closeness probe | 7.5 | `table10_closeness_probe` |
| 11 | EG under acceptance models | 7.6 | `table11_acceptance` |
| 12 | Per-dataset wall-clock runtime | A.2 | `table12_runtime` |
| 13 | Computational complexity | A.2 | — analytical |
| 14 | Hyperparameter search grids | A.3 | — definitional |
| 15 | Declared comparison families | B | — definitional |
| 16 | Paired per-seed contrasts | B | `table16_paired_contrasts` |
| 17 | ΔSP / ΔEO under both protocols | B | `table17_demographic_protocol` |
| 18 | Bin size vs stratified Degree EG | C | `table18_bin_perturbation` |

Figures 1, 3 and 4 are handled by `experiments/tables/figures.py`; Figure 2 is
a hand-drawn pipeline diagram.

### Two protocols, deliberately

Tables 4, 7, 8 and 12 are **fixed-split**: one train/val/test partition,
repetitions vary the training seed. Tables 16 and 17 are **split-resampled**:
each seed draws its own partition, so method differences are paired within
split (Brightkite excepted — its proxy-temporal cut is fixed). Means differ
between the two legitimately, sometimes substantially. Never quote a paired
delta from Table 16 next to a magnitude from Table 4 without saying which
protocol produced it.

---

<!-- ## Citation

```bibtex
@article{cafe2026,
  title     = {Coreness-Aware Fairness Enhancement for Link Prediction},
  author    = {},
  year      = {2026}
}
``` -->
