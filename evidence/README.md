# Evidence snapshot

The result files the paper's tables and figures are built from, copied from
the experiment campaigns with their original repository-relative paths.
`MANIFEST.sha256` records every file (`make verify-evidence`).  The snapshot is
written by `python -m reproduce.tools.freeze_evidence`; do not edit files here
by hand.

| Path | Campaign | Used by |
|---|---|---|
| `results/validation/20260813_184859/`, `20260814_082243_multi_tables/` | Aug 2026 fixed-split reruns, legacy Equality Gain (Ego-FB, Oklahoma, UNC) | T04, T07, F03 |
| `results/validation/20260907_181811_closeness_exact_v1/`, `20260910_061800_closeness_v1_recovery/` | Sep 2026 exact-Closeness campaign and its recovery (Brightkite, Pokec-n) | T04, T07, T11, T15, F03 |
| `results/validation/20260913_*_table7_*` | Sep 2026 ablation reruns (Brightkite, Pokec-n) | T07, F03 |
| `results/validation/20260905_*` | Table 11 acceptance replays (Oklahoma), aligned acceptance (Ego-FB, UNC), aligned paired campaign | T11, T15 |
| `results_repro/tables/` | July 2026 generated intermediates | T02, T08, T10 |
| `results_repro/demographic_backbone/`, `results_repro/demographic_significance/` | July 2026 HL-GNN backbone and split-resampled runs | T08, T09, T16 |
| `analysis/plots/bin_perturbation/` | stratified runs of 2026-07-27 | T05, T18 |
| `analysis/plots/stratified_figure4/` | stratified runs of 2026-08-13/14 (per-run summaries) | F04 |
| `analysis/plots/coreness_by_group/` | exposure runs (tier and group summaries), coreness-group association | T06, T09 |
| `analysis/plots/precision_by_coreness_aggregated/` | UNC diagnosis (per-run summaries) | F01 |
| `results_rq1_controls/*/finalized/` | structural controls and conditional logits | T18(b), T19 |
| `results_k_sensitivity/legacy_best_configs_nonedgewise_k5_k20_v1/finalized/` | K-sensitivity campaign | T20 |
| `results/<ds>/runtime/` | runtime benchmark | T13 |

Renamed on freezing (original location in brackets): `results_repro/demographic_significance/`
[`results_repro/_stale_significance_20260720/` -- the name was misleading, these
are the runs the paper reports], `results_repro/demographic_backbone/results_cafe_p_baseline_<ds>.json`
[`results_repro/<ds>/cafe_p/results_cafe_p_baseline_<ds>_20260719_*.json`],
`analysis/plots/stratified_figure4/` [`results/validation/20260813_184859/figure4/`].

Per-node files (tens of MB each) are reduced to the small per-run summaries
the figures and tables need by `reproduce/tools/derive_exposure.py` and
`reproduce/tools/derive_stratified.py`.  The raw campaign trees, per-node
files and the code state that produced them (git tag `paper-results-provenance`)
are kept outside this snapshot.
