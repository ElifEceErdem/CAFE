# Paper artifact coverage

This matrix distinguishes code availability from currently certified results.

| Artifact | Existing entry point | Release status |
|---|---|---|
| Datasets | `core/data_loader.py`, `backbone/snap_dataset.py` | Ego-FB, Oklahoma, UNC included; Pokec-n/Brightkite external |
| Baselines | `experiments/run_baselines.py` | Oklahoma RA smoke verified; full result matrix not certified here |
| CAFE-In/Pre | `experiments/run_cafe.py` | Certified JSONs for Ego-FB, Oklahoma, UNC |
| Table 1 | hand-authored | Paper PDF only |
| Table 2 | `experiments.tables.table02_dataset_stats` | Generator present |
| Table 3 | `configs/best_configs.csv` | Configuration source preserved |
| Table 4 | `experiments.tables.table04_main_results` | Legacy implementation retained; only listed certified JSONs are release evidence |
| Table 5 | `experiments.tables.table05_coreness_bins` | Generator present; full five-dataset campaign not certified |
| Table 6 | `experiments.tables.table06_localization` | Generator present; full campaign not certified |
| Table 7 | `experiments.tables.table07_ablation` | Legacy implementation retained; certified subset only |
| Table 8 | `experiments.tables.table08_demographic` | Legacy implementation retained; certified subset only |
| Table 9 | `experiments.tables.table09_group_coreness` | Generator present; full campaign not certified |
| Table 10 | `experiments.tables.table10_closeness_probe` | Generator and inputs present |
| Table 11 | `experiments.tables.table11_acceptance` | Legacy implementation retained; full campaign not certified |
| Table 12 | `experiments.tables.table12_runtime` | Generator present; hardware-specific |
| Tables 13-14 | paper/configuration definitions | Paper PDF and frozen config source |
| Table 15 | `experiments.tables.table16_paired_contrasts` | Historical module numbering retained for code safety |
| Table 16 | `experiments.tables.table17_demographic_protocol` | Historical module numbering retained for code safety |
| Table 17 | `t20_brightkite_random_control.job` | External Brightkite data required |
| Table 18 | `table18_bin_perturbation.py`, structural-control analyses | Multiple historical entry points |
| Table 19 | `analysis/diagnosis_conditional_logit.py` | Analysis entry point present |
| Table 20 | `experiments/run_k_sensitivity.py` | Experiment entry point present |
| Figure 1 | `analysis/run_precision_by_coreness.py` | Entry point present |
| Figure 2 | hand-authored | Current rendering preserved in paper PDF |
| Figure 3 | `analysis/plot_precision_eg_tradeoff_table_match.py` | Entry point present; depends on Table 4 result matrix |
| Figure 4 | `analysis/run_stratified_cafe.py` | Entry point present; external Brightkite data required |

No row in this matrix upgrades a historical output to certified status. The
checksum-locked `results_legacy_eg_check` files are the sole certified outputs
for this release.
