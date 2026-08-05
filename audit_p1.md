# Phase 1 — inventory and result-provenance audit

## Scope and controlling evidence

Authoritative current draft: `ToRS_paper/CAFE_ToRS_EceErdem (2).pdf`, as corrected by the author. The originally supplied current-draft path, `ToRS_paper/CAFE_ToRS_EceErdem.pdf`, is absent. The originally supplied previous-submission path, `RecSys_2026_paper_486.pdf`, is also absent. A similarly named `RecSys_2026_paper_486 (3).pdf` exists, but I did not silently substitute it. The supplementary PDF is present.

Verdict for the current-draft identity: **MATCH**.

Verdict for claims requiring the exact previous-submission PDF: **CANNOT VERIFY FROM AVAILABLE FILES**.

The literal `experiments/` directory contains runners, table specifications, jobs, logs, and PNG diagnostics; it contains no result JSON or CSV. Its jobs declare their outputs in sibling trees—principally `results_repro/`, `results/`, and `analysis/plots/`—in `experiments/jobs/README.md:18-31`. `results_asf/` and `results_egfix/` contain alternative corrected-selection campaigns. Thus the literal claim that the new result cells are “in `experiments/`” is a **MISMATCH**; the inventory below follows the output paths declared by the jobs.

I count a JSON as corrected only if it contains
`experiment_info.eg_graph_semantics = simple_undirected_set_union`. The repository itself states that unmarked files predate the metric rewrite and must not be mixed with marked files (`experiments/tables/_framework.py:51-54`; `experiments/jobs/README.md:92-98`). A final-looking unmarked JSON is therefore not a corrected cell.

Overall Phase 1 verdict: **MISMATCH**. The paper-facing tables are not a coherent snapshot of the corrected results.

## 1. Cell inventory

### 1.1 Seed identity is not generally auditable

The requested unit is `(dataset × method × config × seed)`. For several runners, the files support only `(dataset × method × config × repetition index)`:

- The baseline runner explicitly uses `run_seed = seed + run` and reseeds at every repetition (`experiments/run_baselines.py:215-218`); PA, CN, AA, and RA stop after their first deterministic repetition (`experiments/run_baselines.py:262-267`). The aggregate JSONs retain values and repetition counts but not the actual seed labels. Verdict on the exact baseline seed IDs stored in the artifacts: **CANNOT VERIFY FROM AVAILABLE FILES**.
- `run_cafe.py` seeds once (`experiments/run_cafe.py:305-306`). It reseeds before a model only when `reset_seed_per_model=True` (`experiments/run_cafe.py:468-479`), while the fixed-config job does not pass that flag (`experiments/jobs/t04_cafe.job:118-122`). The corrected fixed-config JSONs explicitly record `reset_seed_per_model: false`, yet the writer unconditionally serializes `model_seeds = seed + run` (`experiments/run_cafe.py:730-755`). Those labels are not the RNG states actually used. Verdict on the serialized CAFE seed labels: **MISMATCH**.
- CAFE-P seeds once before its repetition loop (`experiments/run_cafe_p.py:117-124`); CI seeds once and only reseeds under the same opt-in behavior (`experiments/run_coreness_inverse.py:227-240,271-279`); MB-Post seeds once before its loop (`experiments/run_mobility_boost.py:174-190`). Their current jobs do not establish one explicit seed per stored repetition (`experiments/jobs/t07_ablation.job:56-68`). Verdict on exact CAFE-P/CI/MB-Post seed IDs: **CANNOT VERIFY FROM AVAILABLE FILES**.

Accordingly, “repetitions 1–10” below means ten stored results, not ten verified seed-labelled cells. Where a campaign metadata file really does record seeds, I list them explicitly.

This is not cosmetic. The draft says headline repetitions vary the training seed (Section 6.2, p. 13), but the main CAFE artifact labels do not prove that protocol.

### 1.2 Corrected fixed-split baseline cells in `results_repro/`

The paper requires the Table 4 method set shown in Section 6.3 and Table 4 (pp. 13, 15–17). The source globs are declared at `experiments/tables/table04_main_results.py:42-60`.

| Dataset | Corrected cells actually present | Repetitions present | Verdict against Table 4 caption |
|---|---|---|---|
| Ego-FB | Random; PA; CN; AA; RA; LightGCN; SSNet; Cross-Link Bias; LTLP; TailSTEAK; HL-GNN; CAFE-In; CAFE-Pre | Random and each learned method: ten stored repetitions; PA/CN/AA/RA: one | **MATCH** |
| Oklahoma | The Ego-FB set plus FairAdj, LBA, MORAL | Random and each learned method: ten stored repetitions; PA/CN/AA/RA: one | **MATCH** |
| UNC | The Oklahoma set | Random and each learned method: ten stored repetitions; PA/CN/AA/RA: one | **MATCH** |
| Brightkite | The Ego-FB set | Random and each learned method: ten stored repetitions; PA/CN/AA/RA: one | **MATCH** |
| Pokec-n | Random; PA; CN; AA; RA; LightGCN; SSNet; Cross-Link Bias; LTLP; TailSTEAK; FairAdj; LBA; MORAL. Corrected HL-GNN, CAFE-In, and CAFE-Pre are absent. | Random and every learned/attribute baseline present: four stored repetitions; PA/CN/AA/RA: one | **MISMATCH** |

The Pokec baseline artifact declares `experiment_info.num_runs: 10` while `runs_completed` is four for Random and each learned baseline. The verifier reads only declared `num_runs` (`experiments/tables/_framework.py:169-176,199-207`), so it can falsely accept a partial artifact. Verdict on the verifier's completeness check: **MISMATCH**.

### 1.3 Corrected CAFE and ablation cells at the draft configurations

Configuration notation:

- CAFE-P: `(beta, lambda_mobility)`;
- CAFE-In-B and CAFE-In: `(lambda_fair, beta, lambda_mobility)`;
- CAFE-Pre-B and CAFE-Pre: `(alpha, beta, lambda_mobility)`;
- CI-In, CI-Pre, and MB-Post: their single scalar strength.

The values below are quoted from `analysis/best_configs.csv`, mirrored by `experiments/jobs/_common.sh:32-58` and `experiments/tables/_framework.py:67-89`. Table 7 requires the methods listed in Table 3 (p. 14) over every dataset and ten repetitions (Table 7, pp. 22–23).

| Dataset | HL-GNN+RA | CAFE-P | In-B | Pre-B | CAFE-In | CAFE-Pre | CI-In | CI-Pre | MB-Post | Corrected artifact status |
|---|---|---|---|---|---|---|---|---|---|---|
| Ego-FB | beta `1.0` | `(0.1,0.3)` | `(0.3,1.0,0.0)` | `(1.0,0.1,0.0)` | `(0.3,1.0,0.5)` | `(0.05,0.5,0.3)` | `0.05` | `1.0` | `0.1` | All rows have ten stored repetitions: **MATCH** |
| Oklahoma | beta `5.0` | `(1.0,0.5)` | `(0.1,5.0,0.0)` | `(0.5,5.0,0.0)` | `(0.3,5.0,0.5)` | `(1.0,0.5,0.5)` | `1.0` | `1.0` | `0.3` | All required rows have ten stored repetitions: **MATCH**. The CAFE-Pre file also contains alpha `0.5` and `2.0`; alpha `1.0` is present. |
| UNC | beta `5.0` | `(2.0,0.5)` | `(0.5,5.0,0.0)` | `(0.3,5.0,0.0)` | `(1.0,5.0,0.5)` | `(1.0,1.0,0.5)` | `0.05` | `0.5` | `1.0` | All required rows have ten stored repetitions: **MATCH**. The CAFE-Pre file also contains alpha `0.5` and `2.0`; alpha `1.0` is present. |
| Brightkite | beta `5.0` | `(1.0,0.5)` | `(0.1,5.0,0.0)` | `(0.3,2.0,0.0)` | `(0.05,5.0,0.1)` | `(0.3,2.0,0.5)` | `0.1` | `0.1` | `0.1` | All rows have ten stored repetitions: **MATCH** |
| Pokec-n | beta `5.0` | `(1.0,0.5)` | `(0.1,0.5,0.0)` | `(0.3,2.0,0.0)` | `(0.05,1.0,0.1)` | `(0.05,2.0,0.5)` | `0.1` | `0.3` | `0.1` | MB-Post has ten corrected repetitions. Every other row in this table is absent on the corrected semantics: **MISMATCH** |

For Ego-Facebook, Oklahoma, UNC, and Brightkite, `fusion_only` contains beta values `{0.1,0.5,1.0,2.0,5.0}` with ten stored repetitions for each value. Pokec-n has no corrected fusion file. The Table 7 generator sources are enumerated in `experiments/tables/table07_ablation.py:45-60`.

### 1.4 Paper-required cells that are missing

| Paper requirement | Missing corrected cells | Verdict |
|---|---|---|
| Table 4, fixed split, ten repetitions for stochastic methods (pp. 15–17) | Pokec-n HL-GNN, CAFE-In, CAFE-Pre are absent. Pokec-n Random, LightGCN, SSNet, Cross-Link Bias, LTLP, TailSTEAK, FairAdj, LBA, and MORAL contain only four repetitions. | **MISMATCH** |
| Table 7, ten methods over every dataset (pp. 22–23) | On Pokec-n, corrected HL-GNN, CI-In, CI-Pre, HL-GNN+RA, CAFE-In-B, CAFE-Pre-B, CAFE-P, CAFE-In, and CAFE-Pre are absent. Only MB-Post is corrected and complete. | **MISMATCH** |
| Table 8, Oklahoma/UNC/Pokec-n, ten repetitions (p. 24) | On Pokec-n, corrected HL-GNN, HL-GNN+RA, CAFE-P, CAFE-In-B, CAFE-Pre-B, CAFE-In, and CAFE-Pre are absent. FairAdj/LBA/MORAL have four repetitions. | **MISMATCH** |
| Table 16, paired seeds `0` through `9` (p. 36) | At the read used for this audit: Ego-FB has seeds `{0,1}`; Oklahoma `{0,1,2}`; UNC `{0,1}`; Brightkite `{0}`; the Pokec-n consolidated CSV is absent. The files were actively changing. | **MISMATCH** |
| Table 17, split-resampled seeds `0` through `9` for the attribute datasets (p. 37) | Oklahoma has `{0,1,2}`, UNC `{0,1}`, and Pokec-n has no current consolidated CSV. | **MISMATCH** |

The Pokec logs show a first HL-GNN repetition, a first fusion-grid repetition, and a first CI-In repetition in progress. A growing log is not a result cell. Scheduler state itself is **CANNOT VERIFY FROM AVAILABLE FILES**.

The draft's Table 7 continuation says Pokec-n CI-Pre and MB-Post “have not finished and are omitted” (Table 7, p. 23). The corrected MB-Post artifact now exists, while the other corrected rows listed above do not. Verdict on that caption against disk: **MISMATCH**.

### 1.5 Other paper-facing campaign cells

| Campaign | Cells present | Paper destination | Verdict |
|---|---|---|---|
| Stratified, `analysis/plots/stratified_cafe/` | Every dataset; methods RA, HL-GNN, Fusion, CAFE-In, CAFE-Pre; explicit seeds `{42,43,44}`; split seed `42`; corrected-semantics metadata and recommendation NPZs | Tables 5, 6(a), 18; Figure 4 | Raw-cell coverage: **MATCH** |
| Target exposure, `analysis/plots/coreness_by_group/exposure_per_node_*.csv` | Every dataset; per-node exposure files exist | Tables 6(b), 9 | Files have no corrected-semantics or source-manifest field: **CANNOT VERIFY FROM AVAILABLE FILES** |
| Closeness probe, `analysis/plots/closeness_mechanism_paired/` | Ego-FB, Oklahoma, UNC; run indices `0` through `9`; no Pokec-n/Brightkite probe | Table 10 | The paper explicitly scopes the probe to these datasets (Table 10 and Section 7.5, p. 26): **MATCH** |
| Acceptance, `results/*/acceptance_eg/acceptance_eg_*.csv` | Every dataset; run indices `0` through `9`; RA, HL-GNN, HL-GNN+RA, CAFE-In, CAFE-Pre | Table 11 and Table 6(b) exposure output | CSV schema has no config or EG-semantics marker: **CANNOT VERIFY FROM AVAILABLE FILES** |
| Runtime, `results/*/runtime/runtime_*.json` | Every dataset; `config.runs: 10`; each timing array has ten values | Table 12 | Raw coverage: **MATCH**. Protocol/value consistency is a **MISMATCH**; see Sections 2 and 4. |
| Paired significance, `results_repro/*/significance/per_seed_metrics_*.csv` | Partial seed sets listed in Section 1.4 | Tables 9, 16, 17 | **MISMATCH** |

### 1.6 Corrected selection campaigns that do not fill the draft cells

`results_egfix/20260723_degree_eg_v1/` contains completed selection cells for CAFE-In and CAFE-Pre on every dataset at explicit seeds `{42,43,44}`. Its aggregate is present for Ego-FB seeds `{42,43,44,45,46,47,48,49,50,51}` and Oklahoma seeds `{42,43}`; the other aggregate dataset-seed cells are absent. Its `stratified/` directory is empty.

`results_egfix/20260724_degree_eg_sum_in_v1/todo3/` contains completed aggregate cells for every dataset, methods RA/HL-GNN/Fusion/CAFE-In/CAFE-Pre, explicit seeds `{42,43,44,45,46,47,48,49,50,51}`, and completed stratified cells at `{42,43,44}`. Its selected configurations are:

| Dataset | CAFE-In `(lambda_fair,beta,lambda_mobility)` | CAFE-Pre `(alpha,beta,lambda_mobility)` |
|---|---|---|
| Ego-FB | `(0.0,0.1,0.5)` | `(1.0,0.1,1.0)` |
| Oklahoma | `(0.0,0.5,1.0)` | `(1.0,0.5,1.0)` |
| UNC | `(0.0,5.0,1.0)` | `(1.0,2.0,1.0)` |
| Pokec-n | `(0.0,5.0,1.0)` | `(0.01,5.0,1.0)` |
| Brightkite | `(1.0,5.0,1.0)` | `(2.0,5.0,1.0)` |

These are not the fixed configurations printed by the current draft. Verdict on using them to fill missing Table 4/7/8 cells without changing the protocol and method definitions: **MISMATCH**.

`results_asf/` materializes the same alternative-selected fixed-split experiment. Ego-FB, Oklahoma, and UNC have final CAFE-In/In-B/Pre/Pre-B files with ten repetitions; Brightkite has final In/In-B/Pre and a Pre-B partial with `completed_runs: 9`; Pokec-n has final In, a Pre partial with `completed_runs: 3`, and no In-B or Pre-B. It has no generated `tables/` output. Verdict as a source for the current draft tables: **MISMATCH**.

### 1.7 Orphaned or non-paper artifact families

- Unmarked July JSONs in the same result directories reproduce many rendered table values, but are invalid sources for corrected Degree/PageRank EG. Verdict as corrected evidence: **MISMATCH**.
- `results_repro/_stale_significance_20260720/` is explicitly archived stale significance output and does not feed the corrected draft. Verdict as corrected evidence: **MISMATCH**.
- `results_repro/*/{weak_tie,coreness_inverse_postproc,coreness_mobility_boost_inproc,coreness_mobility_boost_preproc}/` has no table row in the draft. CI-Post, MB-In, and MB-Pre nevertheless appear in Figure 3's legend; the table specification itself flags them as undefined in the text (`experiments/tables/figures.py:78-86`). Verdict on the Figure 3 method set: **MISMATCH**.
- `results/*/sensitivity_*`, the penalty-reduction analysis files in `results_repro/tables/`, `experiments/plots/*.png`, and the `prebinfix`/pilot analysis directories have no identifiable rendered table or figure in the current PDF. Verdict on a current-paper mapping: **CANNOT VERIFY FROM AVAILABLE FILES**.

## 2. Artifact-to-table/figure provenance map

The repository's declared map is `experiments/tables/__init__.py:7-33`. The map below tests it against the corrected raw artifacts and the rendered draft.

| Draft item | Paper location | Identifiable source and generated artifact | Verdict and concrete note |
|---|---|---|---|
| Table 1 | Table 1, p. 5 | Hand-written related-work positioning; explicitly no result module (`experiments/tables/__init__.py:12`) | No empirical artifact is expected: **MATCH** |
| Table 2 | Section 6.1, Table 2, p. 13 | Raw graphs via `analysis/generate_dataset_table.py`; `results_repro/tables/dataset_table.{csv,tex}` (`experiments/tables/table02_dataset_stats.py:22-34`) | Rendered values match the generated artifact: **MATCH** |
| Table 3 | Section 6.4, Table 3, p. 14 | Definitional CAFE configurations; no result module (`experiments/tables/__init__.py:14`) | No empirical artifact is expected: **MATCH** |
| Table 4 | Section 7.1, Table 4, pp. 15–17 | Per-method JSON globs and `table1_precision_equality_gain.csv`, `table4_precision_equality_gain.tex`, `best_configs.csv` (`experiments/tables/table04_main_results.py:42-73`) | The generated table predates most corrected inputs and matches stale unmarked JSONs. Paper Oklahoma HL-GNN Degree EG is `3.62 ± 0.38`; corrected `results_cafe_p_baseline_oklahoma_20260728_151340.json` quotes mean `9.898910885601222`, std `0.31775091962925106`. Pokec corrected cells are also missing: **MISMATCH** |
| Table 5 | Section 7.2, Table 5, p. 19 | Corrected stratified CSVs; `results_repro/tables/table_coreness_bins.tex` (`experiments/tables/table05_coreness_bins.py:43-63`) | The table reports realized bin definitions/counts, which match the current source files: **MATCH** |
| Table 6 | Section 7.2, Table 6, p. 20 | Panel (a): stratified CSVs; panel (b): exposure-per-node CSVs. Expected `table_localization.{csv,tex}` (`experiments/tables/table06_localization.py:32-52`) | Both expected generated artifacts are absent. Candidate inputs exist, but the rendered values' exact source is unidentified: **CANNOT VERIFY FROM AVAILABLE FILES** |
| Table 7 | Section 7.3, Table 7, pp. 22–23 | HL-GNN/CI/MB/Fusion/CAFE JSONs; `table6_ablation_all.tex` and `ablation_merged.tex` (`experiments/tables/table07_ablation.py:45-74`) | Rendered TeX is stale; corrected Pokec has only MB-Post. Caption also misstates which Pokec runs are unfinished: **MISMATCH** |
| Table 8 | Section 7.4, Table 8, p. 24 | Attribute-dataset Table 4/ablation JSONs; `table5_demographic_fairness.{csv,tex}` (`experiments/tables/table08_demographic.py:22-52`) | Oklahoma/UNC corrected sources exist; most Pokec rows are absent or partial, and the rendered TeX predates corrected inputs: **MISMATCH** |
| Table 9 | Section 7.4, Table 9, p. 25 | `coreness_group_association.csv`, exposure CSVs, significance CSVs; component outputs listed at `experiments/tables/table09_group_coreness.py:36-59` | No single generated artifact reproduces the rendered combined table; current significance inputs are incomplete: **CANNOT VERIFY FROM AVAILABLE FILES** |
| Table 10 | Section 7.5, Table 10, p. 26 | `closeness_mechanism_paired/{closeness_by_bin,path_stats,distance_spectrum}_*.csv`; `closeness_strata*_table.tex` (`experiments/tables/table10_closeness_probe.py:42-63`) | Current sources reproduce the rendered table for the three scoped datasets: **MATCH** |
| Table 11 | Section 7.6, Table 11, p. 27 | `results/*/acceptance_eg/acceptance_eg_*.csv`; `acceptance_table.tex` and weights table (`experiments/tables/table11_acceptance.py:47-67`) | Current CSVs postdate the rendered TeX and lack config/semantics provenance. Worse, the job uses matched CAFE-In beta (`experiments/jobs/t11_exposure.job:20-21,39-47`), while the table-module runner passes standalone `fusion_beta` (`experiments/tables/table11_acceptance.py:28-44`). There is no single defensible Pokec fusion row: **MISMATCH** |
| Table 12 | Appendix A.2, Table 12, p. 33 | `results/*/runtime/runtime_*.json`; expected `runtime_table.tex` (`experiments/tables/table12_runtime.py:37-53`) | Expected TeX is absent; the current Pokec JSON disagrees with the paper and records a different cutoff. See Section 4.5: **MISMATCH** |
| Table 13 | Appendix A.2, Table 13, p. 33 | Analytical complexity; no result module (`experiments/tables/__init__.py:24`) | No empirical artifact is expected: **MATCH** |
| Table 14 | Appendix A.3, Table 14, p. 34 | Definitional search grids; no result module (`experiments/tables/__init__.py:25`) | No empirical result artifact is expected: **MATCH** |
| Table 15 | Appendix B, Table 15, p. 35 | Declared comparison families; no result module (`experiments/tables/__init__.py:26`) | No empirical result artifact is expected: **MATCH** |
| Table 16 | Appendix B, Table 16, p. 36 | Per-seed significance CSVs; `significance_tests.csv` and significance TeX files (`experiments/tables/table16_paired_contrasts.py:48-67`) | Rendered aggregate is stale; current corrected CSVs do not yet contain the required seed sets: **CANNOT VERIFY FROM AVAILABLE FILES** |
| Table 17 | Appendix B, Table 17, p. 37 | Fixed Table 8 sources plus significance CSVs; `demographic_protocol_table.tex` (`experiments/tables/table17_demographic_protocol.py:25-48`) | Fixed demographic metrics are EG-insensitive, but current paired inputs are incomplete and the rendered aggregate predates them: **CANNOT VERIFY FROM AVAILABLE FILES** |
| Table 18 | Appendix C, Table 18, p. 38 | Corrected stratified CSVs; `table_bin_perturbation.tex` (`experiments/tables/table18_bin_perturbation.py:26-46`) | Rendered TeX predates overwritten corrected inputs. Paper/old TeX Ego first-bin Degree EG is `+8.49`; the current CSV quotes `+7.925493151879809`: **MISMATCH** |
| Figure 1 | Section 4, Fig. 1, p. 8 | UNC diagnostic CSVs and current generated PDFs (`experiments/tables/figures.py:32-45,57-59`) | Embedded PDFs are older and differ from current generated plots; exact old upstream CSV is absent: **MISMATCH** |
| Figure 2 | Section 5, Fig. 2, p. 10 | Hand-drawn pipeline (`experiments/tables/figures.py:9`; `experiments/tables/__init__.py:32-33`) | No result artifact is expected: **MATCH** |
| Figure 3 | Section 7.1, Fig. 3, p. 18 | Nominal source `table1_precision_equality_gain.csv` (`experiments/tables/figures.py:60-62`) | The rendered legend contains methods absent from Table 4 and undefined in the text; the exact CSV used to render it is absent: **MISMATCH** |
| Figure 4 | Section 7.2, Fig. 4, pp. 18–19 | Oklahoma/Brightkite stratified CSVs (`experiments/tables/figures.py:63-65`) | Embedded PDFs are byte-identical to stale `_cafe.pdf` files, not the corrected plots produced from current CSVs. Current corrected magnitudes do not support the embedded figure: **MISMATCH** |

Tables with no identifiable exact artifact for their rendered empirical numbers are therefore Table 6, Table 9, the Pokec fusion row of Table 11, and the published Pokec row of Table 12. Tables 1, 3, 13, 14, and 15 have no empirical artifact by design; that is not itself a defect.

## 3. Equality-gain bug impact

### 3.1 Code paths directly affected

The corrected Degree EG path now:

1. canonicalizes `[E,2]`/`[2,E]` inputs;
2. orders undirected endpoints;
3. removes self-loops, duplicate rows, and reciprocal duplicates;
4. constructs the predicted graph as the set union `E_train ∪ Rec_K` and the control as `E_train ∪ E_test`; and
5. counts both endpoints of every simple edge.

These operations are implemented in `metrics/graph_utils.py:15-38,41-94,134-178` and consumed by `metrics/equality_gain.py:62-96,99-144`. This matches the draft's simple-undirected set-union definition (Section 3.3, Eqs. (1)–(3), pp. 6–7). Verdict on current Degree-EG code versus that graph definition: **MATCH**.

The same graph constructors now feed PageRank and the other structural-centrality EG diagnostics. The adjacency is explicitly binarized before PageRank (`metrics/centrality_equality_gain.py:89-125,218-264,518-592`). Source-bin Degree EG uses the corrected control graph and canonical novel edges (`metrics/centrality_equality_gain.py:595-708`). Acceptance full/TP Degree and PageRank EG use those same paths; PW uses the corresponding weighted expected graph (`metrics/acceptance_eg.py:272-311`).

Consequently the bug directly changes:

- Degree EG and PageRank EG in Tables 4, 7, 8, 11, 16 and the EG-bearing parts of Figures 1, 3 and 4;
- stratified/source-bin Degree EG in Tables 6(a), 18 and Figure 4;
- any hyperparameter selection, Pareto membership, method rank, or significance result computed from corrected Degree/PageRank EG;
- acceptance full/TP/PW Degree and PageRank EG.

Verdict on expecting old and new values along those paths to agree: **MISMATCH** is expected; agreement is not a validity test.

### 3.2 Results the isolated EG computation cannot affect

Recommendations are constructed before Precision and before the separate EG calls. `run_cafe.py` builds the recommendation lists and computes recommendation metrics at `experiments/run_cafe.py:620-646`, then calls Degree/PageRank/Closeness EG at `experiments/run_cafe.py:648-666`; baseline evaluation has the same ordering (`baselines/baseline_wrapper.py:1194-1238`). Demographic metrics consume the already-built recommendation lists (`experiments/run_cafe.py:668-684`). Runtime is explicitly timing-only (`experiments/tables/table12_runtime.py:1-9`).

Therefore an isolated post-ranking graph-semantics correction cannot change:

- recommendation lists;
- Precision@10, Distribution Fairness, DeltaSP, DeltaEO, AUC, Hits, or MRR;
- training losses, learned parameters, early stopping, or runtime;
- dataset statistics, coreness bins, RA scores, or raw exposure counts.

The table modules make the same dependency claim for demographic metrics (`experiments/tables/table08_demographic.py:53-60`; `experiments/tables/table17_demographic_protocol.py:46-52`). Verdict for those quantities under an isolated equality-only patch: **MATCH** is required.

Closeness deserves a separate statement. The old NetworKit graph builder already represented an unweighted undirected graph, so the simple-set rewrite does not mathematically change Closeness EG. The current code separately resets the approximate-closeness RNG for large graphs (`metrics/centrality_equality_gain.py:388-410`). Thus exact-mode closeness should match, while large-graph approximate values can differ for an independent RNG reason. This distinction is also stated in `experiments/tables/table10_closeness_probe.py:64-69`.

### 3.3 The rerun is causally confounded

Repository history shows that the equality rewrite was not isolated. The same broad change set also introduced/fixed:

- edge-layout canonicalization and the training-edge recommendation mask (`core/evaluation.py:34-50,408-468`);
- split handling, including the Brightkite temporal split (`core/data_loader.py:31-68`);
- score cloning per configuration, preventing in-place hybrid-score carry-over (`experiments/run_cafe.py:620-634`);
- fairness-penalty reduction/scaling choices (`fairness_methods/inprocessing.py:24-47,53-69,99-125`);
- baseline feature and method-specific handling (`baselines/baseline_wrapper.py:431-490`).

Those paths can change training, recommendations, Precision, and demographic metrics. Old JSONs omit split fingerprints and explicit per-run seeds, while current runners use inconsistent seed serialization as documented in Section 1.1. Therefore an observed old/new difference in a nominally EG-insensitive learned result cannot be attributed to the equality bug. Causal verdict: **CANNOT VERIFY FROM AVAILABLE FILES**.

## 4. Old-versus-new checks for EG-insensitive outputs

### 4.1 Clean deterministic control

The cleanest control is RA at the same fixed split: it is deterministic, has serialized std `0.0`, and does not depend on model training. Old and corrected baseline artifacts give:

| Dataset | Precision@10 old | Precision@10 corrected | Distribution Fairness old | Distribution Fairness corrected | Verdict |
|---|---:|---:|---:|---:|---|
| Ego-FB | `0.237315309651057` | `0.237315309651057` | `0.47921884059906006` | `0.47921884059906006` | **MATCH** |
| Oklahoma | `0.19741312741313014` | `0.19741312741313014` | `0.5520628690719604` | `0.5520628690719604` | **MATCH** |
| UNC | `0.11707612981498544` | `0.11707612981498544` | `0.530561625957489` | `0.530561625957489` | **MATCH** |
| Pokec-n | `0.050435171385985966` | `0.050435171385985966` | `0.4960787296295166` | `0.4960787296295166` | **MATCH** |
| Brightkite | `0.02217325227963496` | `0.02217325227963496` | `0.42379361391067505` | `0.42379361391067505` | **MATCH** |

Sources are the paired old/new files under `results_repro/<dataset>/baselines/`: the July 20 JSON and the marked July 28 JSON for Ego-FB, and the corresponding named files for the other datasets. Oklahoma DeltaSP is also identical at `0.0009738360069716885`, and DeltaEO at `0.02284705669372117`; UNC DeltaSP is identical at `0.00011365790678739308`, and DeltaEO at `0.02512581315407353`; Pokec-n DeltaSP is identical at `0.000019897156997683086`, and DeltaEO at `0.021400264504928185`. Each quoted RA std is `0.0`. Verdict: **MATCH**.

Ego-FB CAFE-In at `lambda_fair_0.3_beta_1.0_lam_0.5` is also exactly repeated: Precision `0.23716127004087156 ± 0.0007423911045962698` and Distribution Fairness `0.5113561570644378 ± 0.0019002701572683406` in both the unmarked July 19 and marked July 28 JSONs. Oklahoma CAFE-In at `lambda_fair_0.3_beta_5.0_lam_0.5` repeats Precision `0.2003822393822424 ± 0.0006772623400256648`, DeltaSP `0.0009233732484517278 ± 0.00001052591018467697`, and DeltaEO `0.01300516700359049 ± 0.002170903950348672`. Verdict for these exact pairs: **MATCH**.

### 4.2 Nominally unaffected values outside the old/new observed ranges

These are direct mismatches, not seed-noise-level movements: each old observed range and corrected observed range is disjoint.

| Cell and metric | Old mean ± std; observed range | Corrected mean ± std; observed range | Numeric verdict | Causal verdict |
|---|---|---|---|---|
| UNC SSNet, DeltaEO@10 | `0.0027867852260273244 ± 0.002131998656393016`; `[0.0002442822562641611, 0.008067742937145433]` | `0.018028147393757674 ± 0.004473059549184107`; `[0.010643810366840256, 0.0247833504694544]` | **MISMATCH** | **CANNOT VERIFY FROM AVAILABLE FILES** |
| UNC TailSTEAK, Precision@10 | `0.004410069760388223 ± 0.00030031892737607594`; `[0.003882317258113429, 0.0048225659690627745]` | `0.014267515923566948 ± 0.004590177870394686`; `[0.006035790112223235, 0.02202001819836218]` | **MISMATCH** | **CANNOT VERIFY FROM AVAILABLE FILES** |
| UNC CAFE-Pre, `preproc_alpha_1.0_beta_1.0_lam_0.5`, Precision@10 | `0.11224446466484857 ± 0.0008535842349300694`; `[0.11088868668486676, 0.11364877161055689]` | `0.10952380952381113 ± 0.0007653896412781811`; `[0.10834091598422939, 0.11085835608128759]` | **MISMATCH** | **CANNOT VERIFY FROM AVAILABLE FILES** |
| Pokec-n SSNet, AUCROC | `0.902314205263844 ± 0.00046098927946050777`; `[0.9016930441693608, 0.903100096842872]` | `0.9040389016834368 ± 0.00024240168720755106`; `[0.9037903686058532, 0.9044251618219485]` | **MISMATCH** | **CANNOT VERIFY FROM AVAILABLE FILES**; corrected file has four repetitions |
| Pokec-n FairAdj, Distribution Fairness@10 | `0.13767189383506775 ± 0.0037689955981053134`; `[0.13213276863098145, 0.14353793859481812]` | `0.1472005546092987 ± 0.004276493039993263`; `[0.14389127492904663, 0.15453588962554932]` | **MISMATCH** | **CANNOT VERIFY FROM AVAILABLE FILES**; corrected file has four repetitions |

Sources are the paired unmarked and marked JSONs in `results_repro/unc/{ssnet,tailsteak,cafe_pre}/` and `results_repro/pokec_n/{ssnet,fairadj}/`. The equality calculation cannot generate any of these changes. The broad code change, seed ambiguity, and partial Pokec repetitions prevent attribution to a single alternative cause.

For other stochastic pairs whose means changed but whose serialized ranges overlap, verdict on equality with matched randomness is **CANNOT VERIFY FROM AVAILABLE FILES**: the artifacts do not supply aligned per-seed identities and split fingerprints for the old runs.

### 4.3 Closeness mismatch caused by the separate approximation-RNG change

RA Closeness EG is identical in exact mode for the smaller graphs: Ego-FB old/corrected is `22.130047036750494`; Oklahoma `12.023803228657547`; UNC `13.270734525149816`; std is `0.0` in each artifact. Verdict: **MATCH**.

On the large graphs, RA changes despite being deterministic:

| Dataset | Old Closeness EG | Corrected Closeness EG | Std in both | Verdict |
|---|---:|---:|---:|---|
| Brightkite | `0.06012654399399944` | `0.30342180484798614` | `0.0` | **MISMATCH** |
| Pokec-n | `-1.2663594754554173` | `0.2920720523841337` | `0.0` | **MISMATCH** |

The Pokec value flips sign. This pattern is consistent with the independent approximate-closeness RNG reset at `metrics/centrality_equality_gain.py:388-410`, not the Degree/PageRank set-union correction. Causal verdict for the graph-semantics bug: **CANNOT VERIFY FROM AVAILABLE FILES** as the cause; the code identifies a separate cause.

### 4.4 Stale aggregate witness

The rendered Table 4 and `results_repro/tables/table4_precision_equality_gain.tex` quote Oklahoma HL-GNN Degree EG as `3.62 ± 0.38` (Table 4(c), p. 16). The corrected JSON quotes `9.898910885601222 ± 0.31775091962925106`. This is an affected metric, so the numeric change is expected; the defect is that the PDF still prints the superseded value. Verdict: **MISMATCH**.

### 4.5 Runtime: an equality-insensitive source mismatch

The paper quotes Pokec-n vanilla training time `8540 ± 16` seconds (Table 12, p. 33). `results/pokec_n/runtime/runtime_pokec_n.json` quotes mean `8797.479388300515`, std `350.4696178656877`, and `config.rec_k: 100`. The current declared runtime job requires `--rec_k 10` (`experiments/jobs/t12_runtime.job:47-51`), and the repository itself flags the stored `8797 ± 351` versus published `8540 ± 16` discrepancy as a contended-GPU run (`experiments/jobs/README.md:50-53`). Equality Gain cannot touch runtime (`experiments/tables/table12_runtime.py:7-9`). Verdict: **MISMATCH**.

## 5. What could not be checked, and why

1. Exact previous-submission paper comparisons: **CANNOT VERIFY FROM AVAILABLE FILES** because `RecSys_2026_paper_486.pdf` is absent. I did not assume `(3).pdf` is identical.
2. Exact seed-labelled fixed-config cells for CAFE, CAFE-P, CI, MB-Post, and most aggregate baselines: **CANNOT VERIFY FROM AVAILABLE FILES** because seed IDs are omitted or falsely serialized while a single RNG stream evolves.
3. Live job/scheduler state: **CANNOT VERIFY FROM AVAILABLE FILES**. Logs were changing, but a log line is not a completed result artifact.
4. Corrected semantics/config provenance for acceptance and exposure CSVs: **CANNOT VERIFY FROM AVAILABLE FILES** because the CSV schemas contain no semantics marker, config fields, source manifest, or sidecar tying them to the current code.
5. Corrected Tables 9, 16, and 17: **CANNOT VERIFY FROM AVAILABLE FILES** because the paired significance CSVs are incomplete and still changing.
6. Exact rendered source for Table 6: **CANNOT VERIFY FROM AVAILABLE FILES** because both expected `table_localization` artifacts are absent.
7. Exact rendered source for the combined Table 9: **CANNOT VERIFY FROM AVAILABLE FILES** because only component outputs exist and the necessary significance inputs are incomplete.
8. Published Pokec runtime raw data: **CANNOT VERIFY FROM AVAILABLE FILES** because the expected generated runtime TeX is absent and the current raw JSON disagrees with the paper.
9. Exact upstream CSVs for Figures 1, 3, 4 and the old Table 18 levels: **CANNOT VERIFY FROM AVAILABLE FILES** because the old CSVs were overwritten or are absent; the embedded/derived artifacts predate the current corrected sources.
10. Whether all stochastic EG-insensitive old/new quantities agree under identical randomness: **CANNOT VERIFY FROM AVAILABLE FILES** because old artifacts lack aligned per-seed values and split fingerprints, and the rerun bundled non-EG code changes.
11. Whether the alternative `results_asf/`/`todo3` configurations may replace the draft configurations without changing the experimental claim: **CANNOT VERIFY FROM AVAILABLE FILES** in Phase 1. They are demonstrably different cells; selection consistency belongs to Phase 4.

Phase 1 ends here. Phases 2–5 were not started.
