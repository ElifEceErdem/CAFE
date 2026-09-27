"""Build the tables and figures of the CAFE paper from the result files in ``results/``.

    python -m tables              # every table and figure -> build/
    python -m tables T04 F03      # selected ones
    python -m tables --list       # what each one needs

One module per artifact (manuscript numbering).  Tables 1, 3, 12 and 14 and
Figure 2 are hand-authored and have no script.
"""
from collections import namedtuple

Artifact = namedtuple("Artifact", "id module title note")

ARTIFACTS = [
    Artifact("T02", "t02_datasets", "Dataset statistics", ""),
    Artifact("T04", "t04_main_results", "Precision@10 and Equality Gain@10, all methods", ""),
    Artifact("T05", "t05_coreness_bins", "Realized source-coreness bins", ""),
    Artifact("T06", "t06_localization", "CAFE-In versus the fusion control",
             "panel (a) needs experiments/run_stratified_cafe.py with 10 runs and the fusion arm"),
    Artifact("T07", "t07_ablation", "Ablations", ""),
    Artifact("T08", "t08_demographic", "Demographic fairness", ""),
    Artifact("T09", "t09_structure", "Structural conditions behind demographic transfer", ""),
    Artifact("T10", "t10_closeness_probe", "Mobility-boost closeness probe", ""),
    Artifact("T11", "t11_acceptance", "Equality Gain under acceptance models", ""),
    Artifact("T13", "t13_runtime", "Wall-clock runtime", "hardware-specific"),
    Artifact("T15", "t15_paired_contrasts", "Paired contrasts against HL-GNN, fusion and RA", ""),
    Artifact("T16", "t16_demographic_protocol", "Demographic gaps under both protocols", ""),
    Artifact("T17", "t17_brightkite_random", "Brightkite random-split control",
             "needs the random-split runs (not included in results/)"),
    Artifact("T18", "t18_bin_controls", "Bin size, degree and clustering controls", ""),
    Artifact("T19", "t19_conditional_logit", "Conditional logit coefficients", ""),
    Artifact("T20", "t20_k_sensitivity", "Cutoff sensitivity (K=5, K=20)", ""),
    Artifact("F01", "f01_diagnosis", "Coreness-stratified diagnosis on UNC", ""),
    Artifact("F03", "f03_pareto", "Precision versus Equality Gain, Pareto panels", ""),
    Artifact("F04", "f04_stratified", "Source-coreness profiles, Oklahoma and Brightkite", ""),
]
