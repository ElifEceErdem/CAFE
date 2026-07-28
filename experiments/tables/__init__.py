"""Table-indexed experiment suite.

One module per manuscript table that is backed by an experiment.  Run any of
them with ``--stage {plan,verify,build,run}``; see ``_framework`` for the
contract and ``experiments/jobs/`` for the cluster job matching each one.

Complete map of the draft's tables (``CAFE_ToRS_EceErdem.pdf``):

===== ==================================================== ==========================
Table Content                                              Module
===== ==================================================== ==========================
1     Positioning against related methods                  none -- hand-written
2     Dataset statistics                                   table02_dataset_stats
3     CAFE configurations                                  none -- definitional
4     Main results (Precision@10 + 3 EG dims)              table04_main_results
5     Realized coreness bins                               table05_coreness_bins
6     Localization of CAFE-In beyond fusion                table06_localization
7     Ablation across five datasets                        table07_ablation
8     Demographic fairness                                 table08_demographic
9     Structural mechanism of the demographic byproduct    table09_group_coreness
10    Mobility boost, stratum-level closeness probe        table10_closeness_probe
11    EG under acceptance models (RQ6)                     table11_acceptance
12    Per-dataset wall-clock runtime                       table12_runtime
13    Computational complexity                             none -- analytical
14    CAFE hyperparameter search grids                     none -- definitional
15    Declared comparison families                         none -- definitional
16    Paired per-seed contrasts (a/b/c)                    table16_paired_contrasts
17    DeltaSP / DeltaEO under both protocols               table17_demographic_protocol
18    Bin size vs stratified Degree EG                     table18_bin_perturbation
===== ==================================================== ==========================

Figures 1, 3 and 4 are produced by ``figures``.  Figure 2 is a hand-drawn
pipeline diagram.
"""

TABLE_MODULES = {
    "2": "table02_dataset_stats",
    "4": "table04_main_results",
    "5": "table05_coreness_bins",
    "6": "table06_localization",
    "7": "table07_ablation",
    "8": "table08_demographic",
    "9": "table09_group_coreness",
    "10": "table10_closeness_probe",
    "11": "table11_acceptance",
    "12": "table12_runtime",
    "16": "table16_paired_contrasts",
    "17": "table17_demographic_protocol",
    "18": "table18_bin_perturbation",
}

MANUAL_TABLES = {
    "1": "positioning against related methods (hand-written)",
    "3": "CAFE configurations (definitional)",
    "13": "computational complexity (analytical)",
    "14": "hyperparameter search grids (definitional)",
    "15": "declared comparison families (definitional)",
}
