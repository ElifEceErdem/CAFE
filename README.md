# CAFE: Coreness-Aware Fairness Enhancement for Link Prediction

CAFE is a fairness framework for link prediction on social networks. It addresses **degree bias** -- the tendency of GNN-based recommenders to over-recommend already well-connected (high-degree/core) nodes at the expense of peripheral nodes.

CAFE uses **HL-GNN** as its backbone encoder and combines it with coreness-based fairness interventions applied at different stages of the pipeline: pre-processing, in-processing, and post-processing.

---

## Method Overview

CAFE introduces several variants that differ in *when* and *how* fairness is enforced:

| Variant | Script | Fairness Stage | Mechanism |
|---|---|---|---|
| **CAFE-In** | `run_cafe.py` | In-processing | Loss penalty on core edges + RA hybrid (with mobility boost) |
| **CAFE-Pre** | `run_cafe.py` | Pre-processing | Adjacency modification + RA hybrid (with mobility boost) |
| **CAFE-In-B** | `run_cafe.py` | In-processing | Loss penalty on core edges + RA hybrid (no mobility boost) |
| **CAFE-Pre-B** | `run_cafe.py` | Pre-processing | Adjacency modification + RA hybrid (no mobility boost) |
| **CAFE-P** | `run_cafe_p.py` | Post-processing | RA hybrid + coreness-based mobility boost re-ranking |
| **CI-In** | `run_coreness_inverse.py` | In-processing | Coreness-inverse loss penalty only |
| **CI-Pre** | `run_coreness_inverse.py` | Pre-processing | Coreness-inverse adjacency modification only |
| **MB-Post** | `run_mobility_boost.py` | Post-processing | Mobility boost re-ranking only |

The **HLGNN baseline** is run via `run_cafe_p.py` with `--beta_ra 0.0 --lambda_mobility 0.0`.

#### Optional Methods (not reported in paper)

The following variants were explored during development but did not show superior performance over the methods above and are therefore **not included in the paper results**:

| Variant | Script | Fairness Stage | Mechanism |
|---|---|---|---|
| **CI-Post** | `run_mobility_boost.py` | Post-processing | Coreness-inverse penalty re-ranking (`--boosting_mode continuous --reranking_approach penalty`) |
| **MB-In** | `run_coreness_inverse.py` | In-processing | Mobility boost in-processing loss (`--fairness_method mobility_boost_inproc`) |
| **MB-Pre** | `run_coreness_inverse.py` | Pre-processing | Mobility boost pre-processing adjacency modification (`--fairness_method mobility_boost_preproc`) |

### Key Concepts

- **Coreness**: H-index based structural measure; high-coreness nodes are densely connected "core" nodes, low-coreness nodes are "peripheral".
- **Resource Allocation (RA)**: Heuristic link score that gives weight to shared low-degree neighbors.
- **Mobility Boost**: Per-node score multiplier that promotes peripheral nodes at inference time, derived from coreness percentile ranks.
- **Equality Gain (EG)**: Primary fairness metric measuring how much the recommendation distribution shifts toward under-recommended nodes.

---

## Repository Structure

```
CAFE/
├── backbone/                   # HL-GNN backbone (from SNAP)
│   ├── model.py                # BaseModel class (HLGNN encoder + MLP predictor)
│   ├── layer.py                # GNN layer implementations
│   ├── loss.py                 # AUC and BCE loss functions
│   ├── utils.py                # Training utilities
│   ├── negative_sample.py      # Negative sampling strategies
│   └── logger.py               # Training logger
│
├── core/                       # Core pipeline utilities
│   ├── data_loader.py          # Dataset loading, splitting, sensitive attribute loading
│   ├── evaluation.py           # Metrics computation at inference
│   ├── training.py             # Standard GNN training loop
│   ├── training_inproc.py      # Fairness-aware in-processing training loop
│   ├── statistics.py           # Result aggregation and serialization
│   ├── utils.py                # Seed, str2bool helpers
│   ├── model_ra.py             # Resource Allocation model variants
│   ├── model_ra_reg.py         # RA model with regularization
│   ├── ra_features.py          # RA feature extraction
│   └── ra_sampler.py           # RA-based sampling
|
├── data/                       # Datasets
│   ├── SNAP                   
│       ├── ego-facebook 
│       ├── oklahoma
│       └── unc          
│
├── fairness_methods/           # Fairness intervention implementations
│   ├── inprocessing.py         # FairnessAwareModel, coreness-inverse loss, mobility-boost aggregation
│   ├── reranking_coreness.py   # CorenessReRanker (post-processing)
│   ├── reranking.py            # General diversity re-ranker
│   ├── reranking_pa.py         # PA-based dynamic re-ranker
│   ├── reranking_exposure.py   # Exposure-based re-ranker
│   ├── calibration.py          # Score calibration utilities
│   └── moral_integration.py    # MORAL fairness integration
│
├── metrics/                    # Evaluation metrics
│   ├── coreness.py             # H-index coreness computation
│   ├── equality_gain.py        # Degree-based Equality Gain
│   ├── centrality_equality_gain.py  # PageRank / Closeness EG
│   ├── demographics.py         # Demographic parity, Delta SP/EO
│   ├── degree_analysis.py      # Degree-stratified analysis and plots
│   ├── precision.py            # Precision@K
│   ├── fairness.py             # Distribution fairness
│   ├── pa_aware.py             # PA-aware fairness metrics
│   ├── ndkl.py                 # Normalized Distance KL divergence
│   ├── networked_inequality.py # Network inequality metrics (IF, GDIF)
│   ├── classification.py       # Classification-based metrics
│   └── stratified.py           # Stratified fairness metrics
│
├── baselines/                  # Comparison baseline implementations
│   ├── baseline_models.py      # CN, RA, AA, PA, LightGCN, SSNet, TailSTEAK, etc.
│   ├── baseline_wrapper.py     # Unified baseline runner
│   ├── fairness_aware_baseline_models.py  # FairAdj, CrossWalk, LBA, MORAL, etc.
│   └── tailsteak_dataloader.py # TailSTEAK data loader
│
├── experiments/                # Main experiment runner scripts
│   ├── run_cafe.py             # CAFE-In / CAFE-Pre / CAFE-In-B / CAFE-Pre-B
│   ├── run_cafe_p.py           # CAFE-P  (also HLGNN baseline at beta=0, lambda=0)
│   ├── run_coreness_inverse.py # CI-In / CI-Pre
│   └── run_mobility_boost.py   # MB-Post
│
└── analysis/
    └── generate_final_tables.py  # Pareto/ASF best-config selection + paper table generation
```

---

## Installation

### Option 1: From the provided environment file (recommended)

An `environment.yml` file is included in the repository, exported from the working `hl_gnn` environment.

```bash
# Create the environment from the file
conda env create -f environment.yml

# Activate the environment
conda activate hl_gnn
```

To export your current environment state (e.g. after adding new packages):

```bash
conda env export > environment.yml
```

### Option 2: Manual setup

```bash
conda create -n hl_gnn python=3.10
conda activate hl_gnn

# PyTorch + CUDA 11.8
conda install pytorch=2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# PyTorch Geometric + extensions
conda install pytorch_geometric pytorch-scatter pytorch-sparse pytorch-cluster -c pyg

# Other dependencies
conda install numpy scipy pandas scikit-learn matplotlib networkx tqdm gensim pymoo -c conda-forge
pip install python-louvain scikit-network seaborn
```

---

## Data

Datasets live under `data/SNAP/<dataset_name>/` inside the repo:

```
data/SNAP/
├── brightkite/          # gitignored (large)
├── ego-facebook/
├── facebook/            # gitignored (large)
├── oklahoma/
├── pokec_n/             # gitignored (large)
├── pokec_z/             # gitignored (large)
└── unc/
```

The folders `brightkite`, `facebook`, `pokec_n`, and `pokec_z` are excluded from version control via `.gitignore` due to their size and must be obtained separately.

Each dataset folder contains `raw/` and `processed/` subdirectories. Brightkite also includes a pre-computed temporal split (`temporal_split.pt`) used via `--split_file`.

When running experiments, pass:
```bash
--data_path data/SNAP                                        # from repo root
--split_file data/SNAP/brightkite/temporal_split.pt         # for Brightkite
```

---

## Running Experiments

All scripts are run from the `experiments/` directory.

### CAFE-In (in-processing + RA hybrid + mobility boost)
```bash
python run_cafe.py \
    --data_name brightkite \
    --data_path data/SNAP \
    --split_file data/SNAP/brightkite/temporal_split.pt \
    --encoder HLGNN --predictor MLP \
    --epochs 300 --patience 50 --batch_size 16384 --lr 0.001 --runs 3 \
    --penalization_mode inproc \
    --use_mobility_boost True \
    --lambda_fair "0.05" \
    --beta_ra "5.0" \
    --lambda_mobility "0.1" \
    --coreness_alpha 1.0 \
    --use_equality_gain True --use_centrality_gain True \
    --centrality_types pagerank,closeness \
    --results_file results/cafe_in_brightkite.json
```

### CAFE-Pre (pre-processing + RA hybrid + mobility boost)
```bash
python run_cafe.py \
    --data_name brightkite \
    --data_path data/SNAP \
    --split_file data/SNAP/brightkite/temporal_split.pt \
    --encoder HLGNN --predictor MLP \
    --epochs 300 --patience 50 --batch_size 16384 --lr 0.001 --runs 3 \
    --penalization_mode preproc \
    --use_mobility_boost True \
    --preproc_alpha "0.3" \
    --beta_ra "2.0" \
    --lambda_mobility "0.5" \
    --coreness_alpha 1.0 \
    --use_equality_gain True --use_centrality_gain True \
    --centrality_types pagerank,closeness \
    --results_file results/cafe_pre_brightkite.json
```

### CAFE-In-B (in-processing + RA hybrid, no mobility boost)
```bash
python run_cafe.py \
    --penalization_mode inproc --use_mobility_boost False \
    --lambda_fair "0.1" --beta_ra "5.0" \
    ...
```

### CAFE-Pre-B (pre-processing + RA hybrid, no mobility boost)
```bash
python run_cafe.py \
    --penalization_mode preproc --use_mobility_boost False \
    --preproc_alpha "0.3" --beta_ra "2.0" \
    ...
```

### CAFE-P (RA hybrid + mobility boost, standard GNN training)
```bash
python run_cafe_p.py \
    --data_name brightkite \
    --data_path data/SNAP \
    --split_file data/SNAP/brightkite/temporal_split.pt \
    --encoder HLGNN --predictor MLP \
    --epochs 300 --patience 50 --batch_size 16384 --lr 0.001 --runs 3 \
    --beta_ra "1.0" \
    --lambda_mobility "0.5" \
    --coreness_alpha 1.0 \
    --use_equality_gain True --use_centrality_gain True \
    --results_file results/cafe_p_brightkite.json
```

### HLGNN Baseline
```bash
python run_cafe_p.py \
    --beta_ra "0.0" --lambda_mobility "0.0" \
    ...
```

### CI-In / CI-Pre (Coreness-Inverse only)
```bash
# CI-In
python run_coreness_inverse.py \
    --data_name brightkite \
    --data_path data/SNAP \
    --encoder HLGNN --predictor MLP \
    --epochs 300 --patience 50 --batch_size 16384 --lr 0.001 --runs 1 \
    --fairness_method coreness_inverse_inproc \
    --lambda_fair 0.1 \
    --skip_baseline True \
    --use_equality_gain True --use_centrality_gain True \
    --centrality_types pagerank,closeness \
    --results_file results/ci_in_brightkite.json

# CI-Pre: replace --fairness_method coreness_inverse_preproc
```

### MB-Post (Mobility Boost post-processing only)
```bash
python run_mobility_boost.py \
    --data_name brightkite \
    --data_path data/SNAP \
    --encoder HLGNN --predictor MLP \
    --epochs 300 --patience 50 --batch_size 16384 --lr 0.001 --runs 1 \
    --coreness_alpha 1.0 \
    --lambda_mobility 0.1,0.3,0.5,0.8 \
    --lambda_bridge 0.0 \
    --boosting_mode continuous \
    --reranking_approach boost \
    --use_equality_gain True --use_centrality_gain True \
    --centrality_types pagerank,closeness \
    --results_file results/mb_post_brightkite.json
```

---

## Key Arguments (shared across scripts)

| Argument | Description | Default |
|---|---|---|
| `--data_name` | Dataset name | `ego-facebook` |
| `--data_path` | Path to dataset root | `data/SNAP` |
| `--split_file` | Pre-computed split file (for temporal datasets) | `None` |
| `--encoder` | GNN encoder (`HLGNN`) | `HLGNN` |
| `--predictor` | Link predictor (`MLP`) | `MLP` |
| `--epochs` | Max training epochs | `500` |
| `--patience` | Early stopping patience | `50` |
| `--runs` | Number of independent runs | `3` |
| `--coreness_alpha` | Exponent for coreness-based weighting | `1.0` |
| `--use_equality_gain` | Compute Degree EG metrics | `True` |
| `--use_centrality_gain` | Compute PageRank/Closeness EG metrics | `True` |
| `--centrality_types` | Centrality types for EG | `pagerank,closeness` |

---

## Best Hyperparameter Configurations

The best hyperparameter configurations selected via Pareto front + ASF for each method and dataset (as reported in the paper) are provided in [`analysis/best_configs.csv`](analysis/best_configs.csv).

Column definitions (corresponding to actual CLI argument names):

| Column | Argument | Used by | Description |
|---|---|---|---|
| `lambda_fair` | `--lambda_fair` | `run_cafe.py` (in-proc), `run_coreness_inverse.py` | Fairness penalty weight (in-processing) or adjacency modification strength (pre-processing) |
| `preproc_alpha` | `--preproc_alpha` | `run_cafe.py` (pre-proc) | Pre-processing adjacency modification strength |
| `beta_ra` | `--beta_ra` | `run_cafe.py`, `run_cafe_p.py` | Weight for RA score in hybrid combination |
| `lambda_mobility` | `--lambda_mobility` | `run_cafe.py`, `run_cafe_p.py`, `run_mobility_boost.py` | Mobility boost / penalty strength at inference |

Empty cells indicate that the parameter is not applicable to the method. Baseline methods (HLGNN, Cross-Link Bias, FairAdj) have no tuned fairness hyperparameters.

---

## Metrics

Primary metrics reported in the paper:

- **Precision@10**: Recommendation accuracy
- **Degree EG@10** (`Equality_Gain@10`): Equality gain based on node degree
- **PageRank EG@10**: Equality gain based on PageRank centrality
- **Closeness EG@10**: Equality gain based on closeness centrality

Higher EG values indicate the model gives relatively more attention to under-connected nodes. Negative EG means the model amplifies existing degree bias.

---

## Generating Paper Tables

After collecting result JSON files into subdirectories under `analysis/`, run:

```bash
cd analysis/
python generate_final_tables.py
```

This uses Pareto front + ASF to select the best hyperparameter configuration per method per dataset and outputs `table1_precision_equality_gain.csv` and `table2_precision_demographic.csv`.

---

## Citation

If you use CAFE in your work, please cite:

```bibtex
@article{cafe2025,
  title     = {CAFE: Coreness-Aware Fairness Enhancement for Link Prediction},
  author    = {},
  year      = {2025}
}
```