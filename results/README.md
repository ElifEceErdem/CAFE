# Results

The result files the paper's tables and figures are built from
(`python -m tables`).  Every summary in the paper is recomputed from the
per-run values stored here.  `SHA256SUMS` lists every file.

```text
results/
├── index/       which result files feed each row of Tables 4, 7, 11 and 15
├── runs/        runner outputs, one folder per experiment campaign
└── summaries/   reduced outputs of the analysis scripts and the runtime benchmark
```

| Artifact | Files | Produced by |
|---|---|---|
| Table 2 | `summaries/dataset_table.csv` | `experiments/dataset_statistics.py` |
| Tables 4, 7, Figure 3 | `index/main_results_rows.csv` → `runs/2026-08-1*_fixed_split*`, `runs/2026-09-07_exact_closeness`, `runs/2026-09-10_exact_closeness_completion`, `runs/2026-09-13_ablations_exact_closeness` | `run_baselines.py`, `run_cafe.py`, `run_cafe_p.py`, `run_coreness_inverse.py`, `run_mobility_boost.py` |
| Tables 5, 18(a,c) | `summaries/bin_perturbation/` | `run_stratified_cafe.py` → `bin_perturbation_size.py` |
| Table 6(b), Table 9 | `summaries/coreness_by_group/` | `run_exposure_acceptance.py` → `summarize_exposure.py`; `coreness_by_group.py` |
| Table 8 | `summaries/demographic_fairness.csv`, `runs/2026-07-19_demographic_backbone` | `summarize_demographic.py`, `run_cafe_p.py` |
| Tables 9, 16 | `runs/2026-07-20_split_resampled_demographic` | `run_significance.py` |
| Table 10 | `summaries/closeness_strata.csv` | `closeness_mechanism.py` → `summarize_closeness_probe.py` |
| Table 11 | `index/acceptance_rows.csv` → `runs/2026-09-05_acceptance*`, `runs/2026-09-0*_exact_closeness*` | runners with `--acceptance_probe_jsonl` |
| Table 13 | `summaries/runtime/` | `run_runtime_benchmark.py` |
| Table 15 | `index/paired_observations.csv` → `runs/2026-09-05_split_resampled`, `runs/2026-09-0*_exact_closeness*` | runners via `run_with.py --split-seed` |
| Table 18(b) | `summaries/structural_controls/` | `rq1_structural_controls.py` |
| Table 19 | `summaries/diagnosis_conditional/` | `run_precision_by_coreness.py` → `diagnosis_conditional_logit.py` |
| Table 20 | `summaries/k_sensitivity/` | `run_k_sensitivity.py` |
| Figure 1 | `summaries/precision_by_coreness/` | `run_precision_by_coreness.py` → `summarize_stratified.py` |
| Figure 4 | `summaries/stratified_figure4/` | `run_stratified_cafe.py` → `summarize_stratified.py` |

Scripts are in `experiments/`.  Degree Equality Gain counts target-indegree
recommendation events (result marker `legacy_indegree_reconstructed`);
PageRank and Closeness EG use the simple undirected graph.  Closeness on
Pokec-n and Brightkite is the exact generalized closeness of
`metrics/closeness_exact_v1.py`.  The nine learned Pokec-n baselines of Table 4
have 8 runs (seeds 42-47, 50-51); every other learned method has 10.
