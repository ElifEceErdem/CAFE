"""Every table and figure of the paper, in manuscript numbering.

``status`` records how far each artifact can be regenerated today:

* ``reproducible``  -- built from the evidence snapshot; verified against the PDF;
* ``partial``       -- some printed cells cannot be regenerated (see ``notes``);
* ``needs-rerun``   -- builder and job exist but the result files behind the
                       printed numbers are not available; run the job;
* ``manual``        -- hand-authored content, shipped as LaTeX in ``reproduce/static``.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Artifact:
    id: str
    title: str
    section: str
    status: str
    builder: str = ""          # module under reproduce.artifacts
    job: str = ""              # example Slurm job under reproduce/jobs
    notes: str = ""
    kind: str = "table"
    verified: bool = True      # compared cell-by-cell with reproduce/expected/<id>.csv


ARTIFACTS = [
    Artifact("T01", "Positioning of CAFE against related methods", "2.3", "manual",
             "static", notes="Hand-authored; see reproduce/static/T01.tex.", verified=False),
    Artifact("T02", "Dataset statistics", "6.1", "reproducible", "t02_datasets", "t02_datasets.job",
             "Two UNC cells are printed truncated (32.76, 0.16); the generator rounds (32.77, 0.17)."),
    Artifact("T03", "CAFE configurations", "6.4", "manual", "static", verified=False,
             notes="Definitional; the selected hyperparameters are configs/best_configs.csv."),
    Artifact("T04", "Precision@10 and Equality Gain@10, all methods", "7.1", "reproducible",
             "t04_main_results", "t04_main_results.job",
             "Pokec-n: nine learned baselines are 8-run means (seeds 42-47, 50-51)."),
    Artifact("T05", "Realized source-coreness bins", "7.2", "reproducible", "t05_coreness_bins",
             "t05_t18_stratified.job"),
    Artifact("T06", "CAFE-In versus the fusion control: localization", "7.2", "partial",
             "t06_localization", "t06_localization.job",
             "Panel (b) percentages rebuild from the exposure runs; panel (a) and the absolute "
             "exposure changes need the job."),
    Artifact("T07", "Ablations", "7.3", "reproducible", "t07_ablation", "t07_ablation.job"),
    Artifact("T08", "Demographic fairness", "7.4", "reproducible", "t08_demographic", "t08_demographic.job"),
    Artifact("T09", "Structural conditions behind demographic transfer", "7.4", "reproducible",
             "t09_structure", "t08_demographic.job"),
    Artifact("T10", "Mobility-boost closeness probe", "7.5", "reproducible", "t10_closeness_probe",
             "t10_closeness_probe.job"),
    Artifact("T11", "EG under acceptance models", "7.6", "reproducible", "t11_acceptance", "t11_acceptance.job"),
    Artifact("T12", "Hyperparameter search grids", "A.2", "manual", "static", verified=False),
    Artifact("T13", "Wall-clock runtime", "A.3", "reproducible", "t13_runtime", "t13_runtime.job",
             "Hardware-specific; reruns are compared with a tolerance."),
    Artifact("T14", "Declared comparison families", "A.5", "manual", "static", verified=False),
    Artifact("T15", "Paired contrasts against HL-GNN, fusion and RA", "B", "reproducible",
             "t15_paired_contrasts", "t15_paired_contrasts.job"),
    Artifact("T16", "Demographic gaps under both protocols", "B", "reproducible", "t16_demographic_protocol",
             "t08_demographic.job"),
    Artifact("T17", "Brightkite random-split control", "B.1", "needs-rerun", "t17_brightkite_random",
             "t17_brightkite_random.job", "Result files for the printed values are not in the evidence snapshot."),
    Artifact("T18", "Bin size, degree and clustering controls", "C", "reproducible", "t18_bin_controls",
             "t05_t18_stratified.job"),
    Artifact("T19", "Conditional logit coefficients", "D", "reproducible", "t19_conditional_logit",
             "t19_conditional_logit.job"),
    Artifact("T20", "Cutoff sensitivity (K=5, K=20)", "E", "reproducible", "t20_k_sensitivity",
             "t20_k_sensitivity.job"),
    Artifact("F01", "Coreness-stratified diagnosis on UNC", "4", "reproducible", "f01_diagnosis",
             "f01_diagnosis.job", kind="figure", verified=False),
    Artifact("F02", "The CAFE pipeline", "5", "manual", "static", kind="figure", verified=False,
             notes="Diagram; reproduce/static/F02.pdf is cropped from the manuscript."),
    Artifact("F03", "Precision versus EG Pareto panels", "7.1", "reproducible", "f03_pareto",
             "t04_main_results.job", kind="figure", verified=False),
    Artifact("F04", "Source-coreness profiles, Oklahoma and Brightkite", "7.2", "reproducible",
             "f04_stratified", "t05_t18_stratified.job", kind="figure", verified=False),
]

BY_ID = {a.id: a for a in ARTIFACTS}


def select(ids):
    if not ids or ids == ["all"]:
        return list(ARTIFACTS)
    out = []
    for token in ids:
        token = token.upper()
        if token not in BY_ID:
            raise SystemExit(f"unknown artifact {token!r}; see `python -m reproduce list`")
        out.append(BY_ID[token])
    return out
