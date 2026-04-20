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
├── experiments/                # Main experiment runners (one per CAFE variant)
├── analysis/                   # Post-hoc analysis scripts, tables, plots
├── data/SNAP/                  # Datasets
├── results/                    # Experiment output JSONs
├── supplementary_materials/    # Paper appendix tables and figures
├── jobs/, logs/                # SLURM job scripts and training logs
└── environment.yml             # Conda environment specification
```

### File listing

- **[backbone/](backbone/)** — `model.py` (HLGNN encoder + MLP predictor), `layer.py`, `loss.py` (AUC/BCE), `utils.py`, `negative_sample.py`, `logger.py`, `snap_dataset.py`, `snap_evaluator.py`
- **[core/](core/)** — `data_loader.py`, `evaluation.py`, `training.py`, `training_inproc.py` (fairness-aware loop), `statistics.py`, `utils.py`, `model_ra.py`, `model_ra_reg.py`, `ra_features.py`, `ra_sampler.py`
- **[fairness_methods/](fairness_methods/)** — `inprocessing.py` (FairnessAwareModel, coreness-inverse loss, mobility-boost aggregation), `reranking_coreness.py` (CorenessReRanker, used by MB-Post), `moral_integration.py` (MORAL baseline integration)
- **[metrics/](metrics/)** — `coreness.py`, `equality_gain.py`, `centrality_equality_gain.py`, `demographics.py`, `degree_analysis.py`, `precision.py`, `fairness.py`, `pa_aware.py`, `ndkl.py`, `networked_inequality.py`, `classification.py`, `stratified.py`
- **[baselines/](baselines/)** — `baseline_models.py` (CN, RA, AA, PA, LightGCN, SSNet, TailSTEAK, ...), `baseline_wrapper.py`, `fairness_aware_baseline_models.py` (FairAdj, CrossWalk, LBA, MORAL, ...), `tailsteak_dataloader.py`
- **[experiments/](experiments/)** — `run_cafe.py` (CAFE-In / -Pre / -In-B / -Pre-B), `run_cafe_p.py` (CAFE-P, also HLGNN baseline), `run_coreness_inverse.py` (CI-In / CI-Pre), `run_mobility_boost.py` (MB-Post)
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

Best values for `lambda_fair`, `preproc_alpha`, `beta_ra`, and `lambda_mobility` per dataset (Pareto + ASF selection as reported in the paper) are in [`analysis/best_configs.csv`](analysis/best_configs.csv). Empty cells indicate the parameter is not applicable to that method.

---

## Analysis & Paper Figures

Contents of [`analysis/`](analysis/):

| File | Purpose |
|---|---|
| `generate_final_tables.py` | Pareto + ASF best-config selection; generates `table1_precision_equality_gain.csv` and `table2_precision_demographic.csv` |
| `plot_precision_eg_tradeoff_table_match.py` | Precision vs Equality-Gain scatter plots with table-matched highlights (star = best 4D trade-off; diamond = best-per-column) |
| `run_precision_by_coreness.py` | Per-coreness precision breakdown; produces heatmap and bar-chart figures |
| `best_configs.csv` | Best hyperparameter configs per method × dataset |
| `plots/` | Output PDFs (generated) |

Reproducing the paper's figures, from the repo root, after collecting per-run result JSONs under `results/<dataset>/<variant>/`:

```bash
python analysis/generate_final_tables.py
python analysis/plot_precision_eg_tradeoff_table_match.py
python analysis/run_precision_by_coreness.py --data_name ego-facebook --runs 10
```

---

## Metrics

Primary metrics reported in the paper:

- **Precision@10** — recommendation accuracy
- **Degree EG@10** — equality gain by node degree
- **PageRank EG@10** / **Closeness EG@10** — equality gain by centrality

---

## Supplementary Materials

The `supplementary_materials/` folder contains the paper's appendix artefacts:

- **Table 4** — Link prediction and structural fairness results across methods and datasets (§7.1, 7.2, 7.3)
- **Table 5** — Demographic and structural fairness results for fairness-aware baselines and CAFE variants (§7.2)
- **Table 6** — Full ablation results across all datasets, including CI-Post / MB-Pre / MB-In stage variants (§7.4)
- **Table 7** — Complexity comparison with CAFE and baseline models (§5)
- **Table 8** — Hyperparameter settings for attribute-dependent fairness baselines (FairAdj, LBA, MORAL)
- **Figure 5** — Precision@10 vs Equality Gain across all five datasets and three fairness dimensions (§7.1, 7.2)

---

<!-- ## Citation

```bibtex
@article{cafe2026,
  title     = {Coreness-Aware Fairness Enhancement for Link Prediction},
  author    = {},
  year      = {2026}
}
``` -->
