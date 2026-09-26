# Reproducibility audit — CAFE ToRS paper

Audit date: 2026-09-26. Branch `repo-cleanup` at `9d99be4` plus the uncommitted working tree.

**Source of truth.** `CAFE_ToRS_EceErdem (7).pdf` (repo root, untracked, compiled 2026-09-22, 42 pp., 20 tables, 4 figures). All table and figure numbers below are that PDF's numbers unless stated otherwise.

**Method.** Every table and figure caption and body was extracted from the PDF. Each was then traced to (i) the script that produced the printed values and (ii) the on-disk file whose numbers match the PDF. For Tables 2, 6 and 13, generators were re-run into a scratch directory. No repository file other than this one was written, no experiment was launched, and no metric code was touched.

**Status vocabulary.**

| Status | Meaning |
|---|---|
| REPRODUCIBLE | A generator exists, and an on-disk evidence file matches the PDF at printed precision. |
| STALE | A generator exists, but the on-disk evidence no longer matches the PDF. The inputs were overwritten or superseded, or the PDF carries hand edits. |
| ORPHANED | The PDF values match no generator/data combination found on disk. |
| MISSING | The evidence directory the generator expects does not exist. |
| HAND-AUTHORED | No computed values; the PDF is the only source. |

A `*` after REPRODUCIBLE means the table reproduces except for the named cells, which are listed in the notes and in §6.

---

## 1. Headline findings

1. **The code that produced the final numbers is not committed.** Every generator behind the Sep-2026 Tables 4/7/11/15 and Figure 3 is untracked (`??`) in git. So is the exact-Closeness metric itself, `metrics/closeness_exact_v1.py`, which produces every Brightkite and Pokec-n Closeness value in Tables 4/7/11/15 and Figure 3. The runners were modified in the working tree (`run_baselines.py`, `run_cafe.py`, `run_cafe_p.py`, `baselines/baseline_wrapper.py`). The diffs are additive acceptance-probe hooks only. A clean clone therefore cannot regenerate the final PDF. See §7.
2. **The tracked table package targets a superseded pipeline.** `experiments/tables/` reads `results_repro/` (July), and its `verify` stage requires `eg_graph_semantics == simple_undirected_set_union`. The PDF is built from the legacy EG metric (`legacy_indegree_reconstructed`) and from `results/validation/*` (Aug–Sep). Under the current metric decision, `python -m experiments.tables --stage verify` rejects every source the PDF actually uses.
3. **Four row-level splices were found in the PDF.** In each, cells in one printed row come from different runs or configurations:
   - **Ego-FB CAFE-In Degree EG = 11.64** (Tables 4, 7, 11). The coherent source, including the certified file, gives **11.76**. 11.64 is the Degree EG of the Sep-7 aligned acceptance run, whose own PageRank and Closeness (12.50, 24.49) were *not* used.
   - **Table 7, Pokec-n HL-GNN+RA.** Precision 0.0552 (+27%) is from β=5. Degree/PageRank/Closeness EG (−5.70 / −0.04 / 8.66) are from β=1, whose precision is 0.0516. No single configuration produces this row. The Sep-13 update guide explicitly said not to do this.
   - **Table 11, Ego-FB CAFE-In and UNC CAFE-Pre.** The Full columns come from Table 4; PW/TP come from a different run. The Sep-13 guide flagged both rows as unreconciled.
   - **Table 8, eight rows.** Precision means were overwritten with Table 4 values, while the SD, ΔSP and ΔEO remain from the July runs (details in §3).
4. **Disclosure gap in Table 4.** The caption says "mean ± SD over 10 seeds". The nine Pokec-n baselines (Random, LightGCN, SSNet, Cross-LinkBias, FairAdj, LBA, MORAL, LTLP, TailSTEAK) are **n=8** (seeds 42–47, 50–51). Figure 3 discloses this; Table 4 does not. The shard-48/49 rerun (job 107596_4) died after 3 of about 110 metric calls on 2026-09-13 12:23 and is not queued. The seed-49 recovery task (index 12 of array 107614) was never submitted.
5. **Evidence for Table 6(a) and Table 17 was not found anywhere on this machine** (search record in §9). Table 6(b) *was* recovered: its percentages come from on-disk files, using the tercile rule the paper states rather than the one the committed generator implements.
6. **Table 18(a)/(c) and Figure 4 come from two different runs of the stratified experiment.** Both reproduce exactly from files on disk (§9). They disagree numerically, e.g. Oklahoma CAFE-In lowest bin 7.21 (Table 18) vs 6.86 (Figure 4).
7. **Scientific flag, outside audit scope — author to verify.** `run_coreness_inverse.py` (the CI-In, CI-Pre and MB-Post rows of Table 7) trains an internal "baseline" whose precision equals the ablation's own on every dataset:

   | Dataset | CI-runner baseline Prec@10 | Table 7 HL-GNN Prec@10 |
   |---|---|---|
   | Brightkite | 0.0021 | 0.0186 |
   | Pokec-n | 0.0217 | 0.0434 |
   | Oklahoma | 0.1225 | 0.1466 |

   Table 7's single-component ΔPrec (−12% to −88%) is computed against the HL-GNN row. It may therefore reflect a weaker backbone inside that runner rather than the coreness component. §7.3 of the paper rests its "12–88% loss" sentence on these numbers.

---

## 2. The 24-row artifact table

Paths are repo-relative. "Aug-13/14 runs" means `results/validation/20260813_184859/table_runs/` (Oklahoma) and `results/validation/20260814_082243_multi_tables/table_runs/` (Ego-FB, UNC, Brightkite, Pokec-n). "Exact campaign" means `results/validation/20260907_181811_closeness_exact_v1/` plus `results/validation/20260910_061800_closeness_v1_recovery/`.

| PDF # | Content | Generator (what actually produced the PDF values) | Evidence (file whose numbers match) | Status | Notes |
|---|---|---|---|---|---|
| T1 | Positioning vs related methods | — | PDF only | HAND-AUTHORED | LaTeX source of the final paper is not in the repo (only `ToRS_paper/sections/A3-asf-selection.tex`). |
| T2 | Dataset statistics | `analysis/generate_dataset_table.py` | `results_repro/tables/dataset_table.tex` | REPRODUCIBLE | Two PDF cells differ: UNC avg. deg. 32.76 vs 32.77 (2·65,287/3,985 = 32.766, so the generator is right) and UNC clustering 0.16 vs 0.17. |
| T3 | CAFE configuration matrix | — | PDF only | HAND-AUTHORED | Definitional. Selected hyperparameters live in `configs/best_configs.csv` (hash-locked). |
| T4 | Main results, 5 panels | Ego-FB/Okla/UNC: `analysis/generate_validation_comparisons.py`. Brightkite/Pokec-n: `analysis/export_exact_closeness_rows.py`, then hand-adapted to `comparison_tables/table4_paste_ready_20260913.tex`. Both scripts untracked. | Ego-FB/Okla/UNC: Aug-13/14 runs (all `legacy_indegree_reconstructed`), cell-by-cell in `comparison_tables/table4_paper_vs_reproduced.csv`. Brightkite/Pokec-n: exact campaign, paste file above. | REPRODUCIBLE* | Exceptions: Ego-FB CAFE-In Deg EG 11.64 (spliced; source says 11.76); nine Pokec-n rows are n=8 while the caption says 10. The seed-45/47 reps of those nine rows come from the Sep-10 recovery, which dropped the heuristics and so ran a different RNG stream. |
| T5 | Realized coreness bins | `analysis/generate_bin_perturbation_tables.py` | `results_repro/tables/table_coreness_bins.tex` | REPRODUCIBLE | Bin sizes depend on the split, not the scorer, so the Jul-28/29 overwrite of stratified inputs does not affect them. |
| T6 | CAFE-In vs fusion: (a) lowest-bin precision, (b) target-tercile exposure | (b): *no committed script*. The committed `analysis/generate_localization_table.py` gives different numbers (below). (a): unknown. | (b): `analysis/plots/coreness_by_group/exposure_per_node_*.csv` (Pokec-n and Brightkite as `.csv.gz`). (a): none found. | (b) REPRODUCIBLE*; (a) ORPHANED | (b): 14 of 15 percentage cells reproduce exactly with the paper's rule ("boundary ties go to the lower tercile", paired per-run mean); recipe in §9. Exceptions: Pokec-n periphery +102.5 vs +99.2 (run 8 has no periphery value, n=9), and the absolute values in parentheses do not reproduce under any obvious definition. The committed generator uses the runner's precomputed `tier` column and cannot read `.gz`, hence its different output. (a): no stratified-precision source (5 tried, §9) matches. |
| T7 | Ablations, 5 datasets | Ego-FB/Okla/UNC: `generate_validation_comparisons.py` (`comparison_tables/table7_paper_vs_reproduced.*`). Brightkite + Pokec-n rows other than CI-In/CI-Pre: `analysis/export_table7_progress_rows.py` (`table7_updated_rows_20260915.tex`). Pokec-n CI-In/CI-Pre: hand-transcribed from rerun JSON. All untracked. | Aug-13/14 runs; `results/validation/20260913_135426_table7_exact_v1_missing/` and `20260913_135828_table7_defq_restart/` (all 12 `exact_process_complete.json` present); exact campaign for HL-GNN, CAFE-In and CAFE-Pre. | REPRODUCIBLE* | Exceptions: Pokec-n HL-GNN+RA β splice; Ego-FB CAFE-In Deg 11.64. Pokec-n CI-In (61.10) and CI-Pre (60.78) match `…_missing/table7/pokec_n/ci-{in,pre}/main_results.json` exactly, but no paste file records them (the Sep-15 paste still shows them as red "pending"). See also headline finding 7. |
| T8 | Demographic fairness (Prec, ΔSP, ΔEO) | `analysis/generate_table5.py` (historical name) | `results_repro/tables/table5_demographic_fairness.tex` (2026-07-28) for ΔSP/ΔEO | STALE | Hand-edited hybrid: every ΔSP/ΔEO cell matches the Jul-28 output, but eight rows had their precision mean replaced with the Table 4 value (§3). |
| T9 | Structural conditions + paired ΔSP | `analysis/coreness_by_group.py` (δ); `analysis/generate_demographic_appendix_tables.py` (paired ΔSP) | `analysis/plots/coreness_by_group/coreness_group_association.csv` (δ = +0.20 / +0.10 / −0.09); `results_repro/tables/demographic_protocol_table.tex` | REPRODUCIBLE | Paired columns are identical to Table 16(a). |
| T10 | Mobility-boost closeness probe | `analysis/generate_closeness_stratum_table.py` | `results_repro/tables/closeness_strata_table.tex`, `closeness_strata_dose_table.tex` | REPRODUCIBLE | Exact match, both panels. |
| T11 | EG under acceptance models | Brightkite/Pokec-n: `export_exact_closeness_rows.py` (`comparison_tables/table11_exact_20260913_rows.tex`). Ego-FB/Okla/UNC: `analysis/format_table11_against_table4.py` (`table11_aligned_20260907_red.tex`), with Full columns then swapped to Table 4 by hand. All untracked. | Exact campaign; `results/validation/20260905_163304_table11_table4_aligned{,_v2,_v3}/`; Oklahoma acceptance CSV from `20260814_082243_multi_tables/table11_oklahoma/` | REPRODUCIBLE* | Exceptions: Ego-FB CAFE-In and UNC CAFE-Pre rows are cross-run splices. The tracked module `table11_acceptance` still reads `results/*/acceptance_eg/` (pre-EG-fix, Pokec-n n=3), so it is stale. |
| T12 | Hyperparameter grids | — | PDF only | HAND-AUTHORED | |
| T13 | Wall-clock runtime | `analysis/generate_runtime_table.py` | `results/{ds}/runtime/runtime_{ds}.json` | STALE | Ego-FB, Okla, UNC and Brightkite reproduce exactly. Pokec-n on disk is 80.1 / 8797±350 / 8949±336 / 8802±336 / 60.31 / 63.52; the PDF prints 79.7 / 8540±16 / 8675±11 / 8532±9 / 57.15 / 60.23. The measurement behind the PDF row is not on disk. |
| T14 | Declared comparison families | — | PDF only | HAND-AUTHORED | |
| T15 | Paired contrasts (a/b/c) | `analysis/export_exact_closeness_rows.py` (untracked) | `comparison_tables/table15_exact_20260913_red.tex` and `table15_exact_20260913_contrasts.csv`. Observations: `results/validation/20260905_163304_table15_aligned_prepared/` (Ego-FB/Okla/UNC) plus the exact campaign (Brightkite/Pokec-n). | REPRODUCIBLE | All 120 cells and stars in panels (a)–(c) match. Prose conflict: §7.3 and footnote 4 say 15(b) is "at matched β", but the source caption records Pokec-n fusion at β=5 against CAFE-In β=1. The tracked module `table16_paired_contrasts` reads `results_repro/*/significance/`, which is superseded. |
| T16 | ΔSP/ΔEO under both protocols | `analysis/generate_demographic_appendix_tables.py` | `results_repro/tables/demographic_protocol_table.tex` (2026-07-24) | REPRODUCIBLE | Exact match, all 48 cells. |
| T17 | Brightkite random-split control | `experiments/jobs/t20_brightkite_random_control.job` → `analysis/finalize_brightkite_random_control.py` | **Absent.** Expected at `results_rq1_controls/brightkite_random_85_5_10_v1/finalized/brightkite_random_control_table.tex`. | MISSING | No logs in `experiments/jobs/logs/`. The one-split archive `results_random_split_archive/stratified/summary_precision_by_bin_brightkite.csv` is close but not equal (RA lowest/highest 0.0351/0.1969 vs PDF 0.0353/0.1955). |
| T18 | Bin size / degree / clustering controls | (a),(c): `analysis/generate_bin_perturbation_tables.py` **as of commit `e2f4e68`** (the HEAD version fails on these inputs with `KeyError: rho_eg_vs_coreness_partial_size`). (b): `analysis/rq1_structural_controls.py`. Panels merged by hand. | (a),(c): intermediates `analysis/plots/bin_perturbation/*.csv` (2026-07-27) → `results_repro/tables/table_bin_perturbation.tex`, reproduced byte-identical. (b): `results_rq1_controls/structural_controls_v1/finalized/table18_alternative_axes_extension.tex` | REPRODUCIBLE | The per-node stratified run behind the Jul-27 intermediates was overwritten on 2026-07-28/29, so only the intermediate step onward can be rerun. **Different run from Figure 4** (§9). |
| T19 | Conditional logit coefficients | `analysis/diagnosis_conditional_logit.py` | `results_rq1_controls/diagnosis_conditional_v1/finalized/diagnosis_conditional_hit_table.tex` | REPRODUCIBLE | Exact match. |
| T20 | K=5 / K=20 sensitivity | `experiments/run_k_sensitivity.py` | `results_k_sensitivity/legacy_best_configs_nonedgewise_k5_k20_v1/finalized/k_sensitivity_table.tex` | REPRODUCIBLE | Metric mismatch with the main tables: Brightkite/Pokec-n Closeness here is the *old approximate* metric (values 0.08–1.07, plus Pokec-n CAFE-Pre K=20 30.92 ± 34.36). Tables 4/7/11/15 use exact Closeness on those datasets. |
| F1 | UNC coreness diagnosis (heatmap + bars) | `analysis/run_precision_by_coreness.py` | Rendered files `ToRS_paper/figures/heatmap_HL-GNN_unc_k10.pdf` and `stacked_barchart_unc.pdf` (2026-07-16) | STALE | The rendered PDF matches the Jul-16 files (cell deg[17–26]×[1–12] = 0.046, n=48; EG axis to 10.0). Current data in `analysis/plots/precision_by_coreness_aggregated/` (Jul-28) gives 0.053 and a different axis. The Jul-16 render also predates the 2026-07-19 edge-shape fix; whether this script's path was affected has not been checked. |
| F2 | Pipeline diagram | — | PDF only | HAND-AUTHORED | |
| F3 | Precision vs EG Pareto grid | `analysis/plot_figure3_updated_20260918.py` (untracked) | `comparison_tables/figure3_updated_20260918/figure3_updated.pdf`, `figure3_points.csv`, `figure3_sources.json` (327 points; Degree = target-indegree) | REPRODUCIBLE | The figure plots *coherent* values that differ from the spliced table cells: Ego-FB CAFE-In Deg 11.76 (tables: 11.64); Pokec-n HL-GNN+RA β=5 (Table 7: β=1 EGs). `docs/artifact-coverage.md` names the wrong generator. The unused longer caption is in `figure3_caption.tex`. |
| F4 | Oklahoma/Brightkite source-coreness profiles | `analysis/run_stratified_cafe.py` | `results/validation/20260813_184859/figure4/stacked_barchart_{oklahoma,brightkite}.pdf` + `stratified_cafe_*.csv`, `eg_by_coreness_cafe_*.csv` (2026-08-13/14; `simple_undirected_set_union`, penalty `mean`, 3 seeds) | REPRODUCIBLE | Visually identical to the printed figure (titles, axes, bars). Same values as the current `analysis/plots/stratified_cafe/`. **Different run from Table 18(a)** (§9). |

The T6 recovery recipe and the sources tried are in §9.

---

## 3. Spot-check detail (Tables 4, 7, 8, 11, 15)

**Table 4.** Checked every row of the Brightkite and Pokec-n panels against `table4_paste_ready_20260913.tex`: identical. Checked Oklahoma LightGCN, FairAdj and Random; UNC HL-GNN, Cross-LinkBias, MORAL and LTLP; and Ego-FB PA against `table4_paper_vs_reproduced.csv`: identical, all with `eg_semantics = legacy_indegree_reconstructed`.

Six CAFE rows were checked against the certified JSONs in `results_legacy_eg_check/`. Five match all four cells: Oklahoma In/Pre, UNC In/Pre, Ego-FB Pre. **Ego-FB CAFE-In does not:** certified Degree EG is 11.7563 (→ 11.76), PDF 11.64.

The 11.64 traces to `comparison_tables/aligned_consistency_20260907.md`, line 34, which reports the Sep-7 aligned acceptance run: 11.64 / 12.50 / 24.49. Only its Degree cell was propagated into Tables 4, 7 and 11.

Pokec-n MORAL prints Degree EG −25.18 and Closeness EG −25.18. This is a genuine coincidence (−25.177 vs −25.180, different SDs), not a copy error.

**Table 7.** Every Brightkite row matches `table7_updated_rows_20260915.tex`. Pokec-n:

| Row | Source | Match |
|---|---|---|
| CI-In | `…/20260913_135426_table7_exact_v1_missing/table7/pokec_n/ci-in/main_results.json`: 0.02152 / −7.4727 / 17.6562 / 61.1050 | ✓ |
| CI-Pre | `…/ci-pre/main_results.json` λ=0.3: 0.0221 / −4.1072 / 19.9576 / 60.7768 | ✓ |
| HL-GNN+RA | `…/20260910_061800_closeness_v1_recovery/table11/pokec_n/fusion/main_results.json`: β=5 gives 0.05518 / 0.056 / 2.850 / 12.604; β=1 gives 0.05161 / −5.698 / −0.040 / 8.658 | ✗ PDF prints 0.0552 / −5.70 / −0.04 / 8.66: precision and ΔPrec from β=5, EGs from β=1 |
| Other 6 rows | Sep-15 paste | ✓ |

Ego-FB, Oklahoma and UNC match `table7_paper_vs_reproduced.csv`, except Ego-FB CAFE-In Degree EG (11.64 vs 11.76, as above).

**Table 8.** The ΔSP/ΔEO cells all match `results_repro/tables/table5_demographic_fairness.tex` (2026-07-28) and, for HL-GNN, `results_repro/{ds}/cafe_p/*_baseline_*_20260719_*.json`. Eight rows mix sources:

| Row | PDF | Jul-28 / Jul-19 source | Table 4 source |
|---|---|---|---|
| Okla FairAdj Prec | 0.0314 ± 0.0021 | 0.0296 ± 0.0021 | 0.0314 ± 0.0038 |
| Okla LBA Prec | 0.0074 ± 0.0005 | 0.0070 ± 0.0005 | 0.0074 ± 0.0004 |
| Okla CAFE-Pre Prec | 0.1874 ± 0.0010 | 0.1866 ± 0.0011 | 0.1874 ± 0.0010 |
| Okla HL-GNN Prec | 0.1466 ± 0.0018 | 0.1454 ± 0.0018 (Jul-19) | 0.1466 ± 0.0020 |
| UNC FairAdj Prec | 0.0180 ± 0.0012 | 0.0171 ± 0.0012 | 0.0180 ± 0.0020 |
| UNC MORAL Prec | 0.0314 ± 0.0039 | 0.0386 ± 0.0039 | 0.0314 ± 0.0048 |
| UNC CAFE-In Prec | 0.1202 ± 0.0003 | 0.1198 ± 0.0004 | 0.1202 ± 0.0003 |
| UNC HL-GNN Prec | 0.0803 ± 0.0012 | 0.0796 ± 0.0012 (Jul-19) | 0.0803 ± 0.0019 |

In each of these rows the ΔSP/ΔEO printed alongside were computed on the other run's recommendations. The UNC MORAL gap (0.0386 vs 0.0314) is 18%.

**Table 11.** Brightkite and Pokec-n match `table11_exact_20260913_rows.tex`. Ego-FB, Oklahoma and UNC match `table11_aligned_20260907_red.tex` in PW/TP everywhere, and in Full except:

- **Ego-FB CAFE-In Full:** PDF 11.64 / 12.65 / 22.61; aligned run 11.64 / 12.50 / 24.49.
- **UNC CAFE-Pre Full:** PDF 2.21 / 10.73 / 15.07 (= Table 4); aligned run 1.87 / 10.34 / 14.51.

Their PW/TP cells are from the aligned run. This is exactly what `comparison_tables/exact_update_guide_20260913.md` §"Separate consistency issues" warned against.

**Table 15.** All three panels were compared cell-by-cell, stars included, against `table15_exact_20260913_red.tex`: identical. The 3 Holm-star changes listed in the Sep-13 guide are present in the PDF.

---

## 4. Landmine verdicts

| Hypothesis | Verdict |
|---|---|
| **EG metric split at `dcea835`** | **Confirmed, contained in the main tables, leaking elsewhere.** PDF Degree EG is legacy (target-indegree events) throughout Tables 4/7/11/15/20 and Figure 3: every matched evidence file carries `legacy_indegree_reconstructed`, or declares target-indegree Degree semantics (the Sep-13 exports). But `results_repro/` CI/MB JSONs, `analysis/plots/stratified_cafe/*` (Figure 4 / Table 18 inputs, `metadata_*.json`), `results_egfix/` and the TODO3 exposure audit carry `simple_undirected_set_union`. Any regeneration of Table 18(a), Figure 4 or Table 6 from current inputs would mix metrics with the main tables. `analysis/generate_final_tables.py`'s `CAFE_REQUIRE_EG_SEMANTICS=1` still keeps *corrected* files only (backwards under the legacy decision), and `experiments/tables/_framework.py:54` hard-codes the corrected marker as the pass condition. |
| **Two recommendation builders (Table 4 vs acceptance table)** | **Numerically resolved in the PDF; still live in code.** Table 11's RA Full values now equal Table 4's on all five datasets: the Sep aligned campaign reports "exact" RA matches, and Table 11 RA rows were sourced to match. Both builders still exist: `baselines/baseline_models.py:162` (score>0 filter, then `np.random.choice` padding) and `experiments/cafe_methods.build_heuristic_recs` (`torch.topk` over the dense row). Any fresh acceptance run can reintroduce the disagreement. |
| **Module numbering drift** | **Confirmed.** Map in §5. Four numbering schemes coexist: the final PDF, `experiments/tables/__init__.py` (an older draft), `docs/artifact-coverage.md` (partly corrected, still wrong for T12–T14 and F3), and `results_repro/tables/*` file names (RecSys-era). |
| **Fixed-split vs split-resampled mixing** | **No mixing in printed numbers; two prose problems.** Tables 4/7/8/11/20 are fixed-split (training seeds 42–51); Tables 9/15/16 are split-resampled (seeds 0–9; Brightkite fixed temporal). Captions say so correctly. Problems: (i) §7.3 and footnote 4 describe Table 15(b) as "at matched β", but Pokec-n uses fusion β=5 against CAFE-In β=1. (ii) Footnote 4 says Tables 6, 10 and 15(b) share one control; Table 6's source cannot be established (ORPHANED), so that claim cannot be checked. |
| **Which of the 11 `results_*` dirs are live** | See §6. Live PDF evidence sits in only three places: `results/validation/` (Sept campaigns, gitignored), `results_repro/tables/` + `results_rq1_controls/` + `results_k_sensitivity/` (appendix tables), and `comparison_tables/` (the only record of how the Sept rows were assembled). `results_legacy_eg_check/` is certified, but 2 of its 8 files are superseded in the PDF and 1 cell disagrees. |
| **Leftover weak-tie code paths** | **Confirmed dead with respect to the paper.** The PDF mentions weak ties only in related work (§2.4). Code and result paths listed in §7. |

---

## 5. Numbering map

| Final PDF | Content | `experiments/tables` module (old draft #) | `docs/artifact-coverage.md` says | `results_repro/tables` / other artifact name |
|---|---|---|---|---|
| T1 | Positioning | — (manual "1") | T1 hand-authored ✓ | — |
| T2 | Dataset stats | `table02_dataset_stats` (2) ✓ | ✓ | `dataset_table.tex` |
| T3 | Config matrix | — (manual "3") | "`configs/best_configs.csv`" | — |
| T4 | Main results | `table04_main_results` (4) ✓ | ✓ | `table4_precision_equality_gain.tex`, `table1_precision_equality_gain.csv` (RecSys "Table 1"); final: `comparison_tables/table4_*` |
| T5 | Coreness bins | `table05_coreness_bins` (5) ✓ | ✓ | `table_coreness_bins.tex` |
| T6 | Localization | `table06_localization` (6) ✓ | ✓ | `table_localization.tex` (never written to `results_repro/tables`) |
| T7 | Ablations | `table07_ablation` (7) ✓ | ✓ | `table6_ablation_all.tex`, `table3_ablation_oklahoma.tex`, `ablation_merged.tex`; final: `comparison_tables/table7_*` |
| T8 | Demographic | `table08_demographic` (8) ✓ | ✓ | **`table5_demographic_fairness.tex`**, `table2_precision_demographic.csv`; generator `generate_table5.py` |
| T9 | Structural conditions | `table09_group_coreness` (9) ✓ | ✓ | `target_exposure_audit_*.tex`, `demographic_protocol_contrasts.csv` |
| T10 | Closeness probe | `table10_closeness_probe` (10) ✓ | ✓ | `closeness_strata_table.tex`, `closeness_strata_dose_table.tex` |
| T11 | Acceptance EG | `table11_acceptance` (11) ✓ | ✓ | `acceptance_table.tex` (stale); final: `comparison_tables/table11_*` |
| T12 | Hyperparameter grids | — (manual "**14**") | ✗ says runtime | — |
| T13 | Runtime | **`table12_runtime`** (12) | ✗ lumps 13–14 as "definitions" | `runtime_table.tex` (not present) |
| T14 | Comparison families | — (manual "**15**") | ✗ | — |
| T15 | Paired contrasts | **`table16_paired_contrasts`** (16) | ✓ (notes drift) | `significance_table.tex` (superseded); final: `comparison_tables/table15_exact_20260913_*` |
| T16 | ΔSP/ΔEO protocols | **`table17_demographic_protocol`** (17) | ✓ (notes drift) | `demographic_protocol_table.tex` |
| T17 | Brightkite random split | — (none) | ✓ job named | `t20_…job`, finalizer `brightkite_random_control_table.tex` |
| T18 | Bin-size/degree/clustering | `table18_bin_perturbation` (18) — covers (a)/(c) only | ✓ | `table_bin_perturbation.tex` + `table18_alternative_axes_extension.tex` |
| T19 | Conditional logit | — (none) | ✓ | `diagnosis_conditional_hit_table.tex`; job `rerun_t22_diagnosis_logits.sh` (another numbering: "t22") |
| T20 | K sensitivity | — (none) | ✓ | `k_sensitivity_table.tex` |
| — | (old draft T13 "computational complexity") | manual "13" | — | Not in final PDF |
| F1 | UNC diagnosis | `figures` | ✓ | `ToRS_paper/figures/heatmap_HL-GNN_unc_k10.pdf`, `stacked_barchart_unc.pdf` |
| F2 | Pipeline | — | ✓ | — |
| F3 | Pareto grid | `figures` (points to superseded script) | ✗ names `plot_precision_eg_tradeoff_table_match.py` | `comparison_tables/figure3_updated_20260918/`; older `full_comparison_matrix_table_match*.pdf` (RecSys "Figure 5") |
| F4 | Stratified profiles | `figures` | ✓ script, but the render is not reproducible | `stacked_barchart_{oklahoma,brightkite}_cafe.pdf` (title mismatch) |

Job-file prefixes (`t04`, `t07`, `t11`, `t15`, `t20`, `t22`) follow the *old* numbering. `t15_*` jobs feed PDF Table 15; `t20` feeds PDF Table 17; `t22` feeds PDF Table 19.

---

## 6. Results inventory

Recommendations only; nothing was moved or deleted. KEEP = PDF evidence or certified. ARCHIVE = superseded but provenance-relevant; move out of the working tree. DELETE = smoke/scratch, unreferenced by any PDF number.

| Directory | Size / files / tracked | Role | Rec. |
|---|---|---|---|
| `results/validation/20260813_184859` | — | T4/T7 Oklahoma rows (legacy); `figure4/` = Figure 4 source | **KEEP** |
| `results/validation/20260814_082243_multi_tables` | — | T4/T7 Ego-FB/UNC rows; T11 Oklahoma acceptance | **KEEP** |
| `results/validation/20260902_table11_corrected` | — | T11 acceptance CSVs (Brightkite/Ego-FB/Pokec-n/UNC) referenced by `table11_table15_sources.json` | **KEEP** |
| `results/validation/20260905_163304_table11_table4_aligned{,_v2,_v3}` | — | T11 Ego-FB/Okla/UNC PW/TP | **KEEP** until `aligned_sources_20260907.json` identifies which variant is authoritative, then ARCHIVE the other two |
| `results/validation/20260905_163304_table15_aligned_prepared` | — | T15 Ego-FB/Okla/UNC observations | **KEEP** |
| `results/validation/20260907_181811_closeness_exact_v1` | — | T4/T7/T11/T15 Brightkite + Pokec-n; F3 | **KEEP** |
| `results/validation/20260910_061800_closeness_v1_recovery` | — | Same (seeds 45/47; fusion sweep) | **KEEP** |
| `results/validation/20260913_135426_table7_exact_v1_missing`, `20260913_135828_table7_defq_restart` | — | T7 Brightkite/Pokec-n ablation reruns | **KEEP** |
| `results/validation/20260913_120827_closeness_v1_shard48` | 3 files | Dead rerun (3 of ~110 metric calls, no results) | ARCHIVE (records the failed attempt) |
| `results/validation/20260813_184859/table11`, `20260902_table15_rerun`, `20260905_*_{acceptance_pilot,acceptance_replay,cafe_pre_diagnostic×3,cafe_pre_ten_run_diagnostic,table15_smoke}` | — | Pilots, diagnostics, superseded T15 | ARCHIVE |
| `results/{ds}/runtime/` | — | T13 (4/5 rows) | **KEEP** |
| `results/{ds}/acceptance_eg`, `significance`, `fusion_only`, `sensitivity_*` | — | Pre-EG-fix RecSys/July campaigns; superseded by `validation/` | ARCHIVE |
| `results/oklahoma/*` (13 JSONs, **tracked**, Mar–Apr 2026) | — | Pre-bug-fix, pre-EG-fix | ARCHIVE (and untrack) |
| `results/eg_semantics_smoke/` | 1 file | Smoke | DELETE |
| `results_repro/tables/` | tracked subset | T2, T5, T8 (ΔSP/ΔEO), T9, T10, T16, T18(a/c) evidence | **KEEP** |
| `results_repro/{ds}/{fairadj,lba,moral,fusion_only,cafe_*,cafe_p}` | 22 MB total | Inputs behind T8/T16 | **KEEP** |
| `results_repro/{ds}/{lightgcn,ssnet,ltlp,tailsteak,cross_link_bias,baselines,ci_*,mb_post,coreness_*}` | — | Superseded for T4/T7 by `validation/`; mixed EG markers | ARCHIVE |
| `results_repro/{ds}/weak_tie`, `results_repro/_stale_significance_20260720`, `results_repro/tables/*_table_match.pdf`, `results_repro/tables/CAFE_ToRS_EceErdem.pdf` | — | Dead / stale copies | DELETE (weak_tie → ARCHIVE if the RecSys paper still cites it) |
| `results_asf/` | 576 KB, 18 files, tracked | ASF selection provenance (RecSys-era; legacy TSV shows it once symlinked into `results_repro`) | ARCHIVE |
| `results_egfix/` | 119 MB, 1,050 files, untracked | Corrected-EG campaign incl. TODO3; not PDF evidence under the legacy decision | ARCHIVE |
| `results_k_sensitivity/` | 25 MB, 460 tracked | T20 | **KEEP** |
| `results_legacy_eg_check/` | 8 files, tracked, hash-locked | Certified release outputs | **KEEP** — but see note below |
| `results_penalty_reduction/` | 3.9 MB | mean-vs-sum penalty study; no PDF table | ARCHIVE |
| `results_pen_smoke/` | 6 files | Smoke | DELETE |
| `results_random_split_archive/` | 7.8 MB | One-split Brightkite control; nearest substitute for missing T17 | ARCHIVE (do **not** delete while T17 is MISSING) |
| `results_rq1_controls/` | 215 MB, 83 tracked | T18(b), T19 | **KEEP** (`finalized/`) |
| `results_verify/` | 1 file | Scratch | DELETE |
| `comparison_tables/` | untracked | Only record of how T4/T7/T11/T15/F3 rows were assembled | **KEEP** (commit it). ARCHIVE the Sep-2/Sep-5 intermediates (`*_updated_red.tex`, `table11_oklahoma_*`, `archive/`) |
| `experiments/results-26-08-05/` | 63 files, gitignored | Stale Aug-5 staging copy + build logs | ARCHIVE |
| `analysis/table*.csv/.tex`, `analysis/best_configs.csv` | tracked | Stale copies of `results_repro/tables/` (RecSys numbering) | DELETE (after confirming no script reads them by default) |
| `analysis/plots/stratified_cafe/`, `precision_by_coreness_aggregated/` | — | Overwritten 07-28/29. Stratified data = Figure 4's run; the Figure 1 render predates it | KEEP |
| `analysis/plots/bin_perturbation/` | 3 CSVs, 2026-07-27 | Only surviving input for Table 18(a)/(c) | **KEEP** (do not regenerate in place) |
| `analysis/plots/coreness_by_group/exposure_per_node_*` (incl. `.gz`) | — | Table 6(b) source | **KEEP** |

Note on `results_legacy_eg_check/`: 6 of its 8 files match the PDF, except the Ego-FB CAFE-In Degree cell (PDF 11.64 vs certified 11.76). The other 2, Pokec-n CI-Pre and MB-Post, are **not** PDF evidence. They carry old approximate Closeness (~97.1), and CI-Pre Degree EG is −4.48 there vs −4.11 in the PDF. The certified hash also locks `paper/CAFE_ToRS_EceErdem.pdf` (Aug 13), which is not the final PDF.

---

## 7. Dead-code and hazard list

Candidates found by reference search, not a full call-graph analysis.

**Dead with respect to the final paper**
- Weak-tie: `experiments/run_weak_tie.py`, `fairness_methods/weak_tie_reranking.py`, `core/evaluation.py::compute_fairness_metrics_with_weak_tie` (l.474), and the weak-tie branches in `analysis/generate_final_tables.py`, `analysis/generate_table4.py`, `analysis/generate_table5.py`, `analysis/plot_precision_eg_tradeoff_table_match.py`, `experiments/tables/figures.py` (exclusion list).
- Corrected-EG tooling after the legacy decision: the commented-out `SIMPLE_UNDIRECTED` block in `metrics/equality_gain.py`; `core/statistics._eg_semantics_marker` (always returns legacy now); `analysis/check_eg_semantics.py`, `analysis/egfix_finalize_reruns.py`, `analysis/egfix_finalize_selection.py`, `analysis/egfix_preflight.py`, `experiments/egfix_protocol.py`, `jobs/egfix*`.
- Superseded figure/table generators: `analysis/plot_precision_eg_tradeoff_table_match.py` (old Fig 3/5), `analysis/generate_table4.py`, `analysis/generate_final_tables.py` (as T4 generator), `analysis/generate_ablation_tables.py`, `analysis/generate_merged_ablation_table.py`, `analysis/generate_acceptance_table.py`, `analysis/generate_table11_table15_updates.py` (superseded by `export_exact_closeness_rows.py`, yet `comparison_tables/README.md` still tells readers to use it).
- One-off diagnostics (untracked): `experiments/diagnose_cafe_pre*.py`, `experiments/pilot_table4_acceptance.py`, `experiments/smoke_table15_aligned.py`, `analysis/pilot_exact_closeness.py`, `analysis/diagnose_closeness_tp.py`, `analysis/update_table15_complete.py`, `analysis/generate_aligned_snapshot.py`.
- Config rows not in the paper: `configs/best_configs.csv` rows for CI-Post, MB-In, MB-Pre and dataset `facebook`. Do not edit; the file is hash-locked.
- Duplicate job trees: top-level `jobs/` (RecSys/egfix era) vs `experiments/jobs/` (48 files, many untracked).

**Live but misleading or hazardous**
- `experiments/tables/` package: its docstrings describe a corrected-EG rerun plan that was abandoned; its sources point at `results_repro/`; its `verify` gate (`_framework.py:54`, `:204`) fails on every legacy source.
- `analysis/generate_final_tables.py` `CAFE_REQUIRE_EG_SEMANTICS`: inverted relative to the legacy decision. There is no "legacy only" filter, so selection over `results_repro/` can mix metrics.
- Two heuristic recommendation builders (§4).
- `experiments/run_coreness_inverse.py` internal baseline (headline finding 7).

**Needed for reproduction but untracked** (must be committed before any claim that the repo reproduces the PDF):
`metrics/closeness_exact_v1.py`, `core/acceptance_probe.py`, `experiments/{reevaluate_closeness_v1,recover_closeness_v1,complete_closeness_v1,merge_baseline_shards,replay_table11_aligned,replay_table4_acceptance,prepare_table15_aligned}.py`, `analysis/{export_exact_closeness_rows,export_table7_progress_rows,generate_validation_comparisons,generate_updated_tex_tables,format_table11_against_table4,format_table11_tp_precision,plot_figure3_updated_20260918}.py`, `experiments/jobs/closeness_v1_*.job`, `t04_*`, `t07_*`, `t11_*`, `t15_*`, plus the uncommitted runner diffs.

---

## 8. Cannot-reproduce list

Each item below is a printed PDF number, or a whole artifact, that cannot currently be regenerated from repository code plus on-disk data.

1. **Table 6(a), entire panel.** No matching source; see §9.
2. **Table 17, entire table.** Not found on this machine; see §9.
3. **Ego-FB CAFE-In Degree EG = 11.64** in Tables 4, 7 and 11. The coherent source gives 11.76. *Acknowledged by author 2026-09-26.*
4. **Table 7 Pokec-n HL-GNN+RA row.** β=5 precision with β=1 EGs. *Acknowledged by author 2026-09-26.*
5. **Table 11 Ego-FB CAFE-In and UNC CAFE-Pre rows.** Full and PW/TP come from different runs. *Acknowledged by author 2026-09-26.*
6. **Table 8, eight rows** listed in §3. Printed ΔSP/ΔEO and precision describe different runs. *Status accepted by author 2026-09-26.*
7. **Table 13 Pokec-n row.** The on-disk measurement differs.
8. **Table 6(b) residue:** Pokec-n peripheral cell (+99.2 printed vs +102.5 recomputed) and the absolute values in parentheses.
9. **Figure 1.** Rendered from pre-Jul-28 data that no longer exists. *Status accepted by author 2026-09-26.*
10. **Table 18(a)/(c) vs Figure 4.** Each reproduces, but from different runs, so their shared cells disagree (§9). The per-node data behind Table 18 no longer exists; only its Jul-27 intermediates do.
11. **Table 4 Pokec-n baselines at n=10**, as the caption claims. Only 8 certified reps exist; seeds 48/49 are unrecoverable from disk, and the rerun died.
12. **Every Brightkite/Pokec-n Closeness value from a clean clone.** The metric module and all September generators are untracked, and both datasets are external.
13. **Table 2, UNC avg. degree and clustering.** The PDF (32.76 / 0.16) disagrees with the generator (32.77 / 0.17). Cosmetic.

Consistency issues that are *not* reproduction failures but will surface in review:
- Table 20 uses approximate Closeness on Brightkite/Pokec-n, unlike Tables 4/7/11/15.
- Figure 3 plots 11.76 and the β=5 fusion point where the tables print 11.64 and the β-spliced row.
- Table 4's caption omits the n=8 disclosure that Figure 3 makes.
- §7.3 and footnote 4 say "matched β" for Table 15(b); the Pokec-n source is β=5 vs β=1.
- `README.md` and `docs/artifact-coverage.md` describe `paper/CAFE_ToRS_EceErdem.pdf` (Aug 13) as the manuscript. The artifact-coverage matrix has wrong entries for T12, T13, T14 and F3.

---

## 9. Follow-up, 2026-09-26: author review and second search

**Author decisions recorded.**
- The three splices (Ego-FB CAFE-In 11.64; Table 7 Pokec-n HL-GNN+RA; Table 11 Ego-FB CAFE-In and UNC CAFE-Pre) are acknowledged. Work continues without changing them.
- The statuses for Tables 2, 4, 7, 8 and 11 and Figure 1 are accepted as reported.

**Where the second search looked, for Tables 6 and 17.**

| Place | Method | Outcome |
|---|---|---|
| Repo working tree (incl. gitignored and `.gz`) | Value scan of all csv/tsv/json/jsonl/tex/md/log/out/ipynb/py files; rounding-tolerant; signature = Table 17's CI triples (e.g. 0.1602/0.1585/0.1619) and Table 6's ratio pair 17.809/16.515 | No data file contains either signature |
| Git history, all branches (`main`, `repo-cleanup`, `repo-cleanup-safety-base`, both remotes) | `git log --all -S` on the values and on the expected file names | Only false positives. The Aug-5 audit (`audit_p1.md` in `dcea835`) already reported the Table 6 artifacts as absent |
| `/var/scratch/eerdem` (all project dirs; `random_splitted_datasets/` is empty, Feb) | Same value scan | No hit |
| `/home/eerdem/workspace/cafe_smoke_test/old_code` (Jul-17 copy) | Same value scan | No hit |
| Agent session logs (`~/.claude/projects`, `~/.codex/sessions`, `.gemini`, `.copilot`) | Value grep | Values appear only where an agent *read the PDF*. The Aug-13 Codex session (the earliest) also recorded Table 6 CSVs as missing and Table 17 as having no result |

**What that implies.** Both tables were already in the Aug-13 PDF. On Aug 5 the Table 17 job had never run and the draft had `[X]/[Y]` placeholders. The Table 17 job (`experiments/jobs/t20_brightkite_random_control.job`) is fail-closed by design and writes only to `${REPO}/results_rq1_controls/brightkite_random_85_5_10_v1/`, which never existed here. The computations therefore happened between Aug 5 and Aug 13 somewhere this audit cannot see: another machine, a compute node's local disk, an interactive session, or Overleaf history. Missing evidence on this machine is **not** evidence that the numbers are wrong.

**Table 6(b) recovery recipe** (run in scratch; nothing committed). Inputs are `analysis/plots/coreness_by_group/exposure_per_node_{ds}.csv[.gz]`; methods `CAFE-In` vs `HL-GNN+RA`.
1. Compute coreness quantiles q₁/₃ and q₂/₃ over unique nodes.
2. Assign periphery = coreness ≤ q₁/₃, middle = coreness ≤ q₂/₃, core = the rest (ties go to the lower group, as the paper states).
3. For each run and tier, take mean per-node exposure.
4. Report the mean over runs of 100·(CAFE-In/Fusion − 1).

| Dataset | Recomputed (P / M / C) | PDF |
|---|---|---|
| Brightkite | +12.3 / +14.3 / −5.7 | same |
| Ego-FB | +15.2 / −2.7 / −1.4 | same |
| Oklahoma | +18.9 / −1.7 / −2.5 | same |
| Pokec-n (n=9) | **+102.5** / +54.3 / −17.7 | **+99.2** / +54.3 / −17.7 |
| UNC | +25.9 / −0.5 / −4.7 | same |

Other sources tried for 6(b), none matching: the committed generator's `tier` column (Ego-FB +18.8%); `results/validation/20260902_table11_corrected/.../exposure/` (Ego-FB +15.8%); TODO3 exposure summary (Oklahoma +206%).

Sources tried for 6(a), none matching: `analysis/plots/stratified_cafe/` (Jul-28), `results/validation/20260813_184859/figure4/`, `analysis/plots/stratified_cafe_egfix_pilot/{105602,105617}/`, and `results_egfix/.../todo3/finalized/stratified_precision_by_run.csv`. All have n=1–3 runs, while the PDF reports "± SD over reruns". The 10-run exposure/acceptance runner saves no source-side precision.

**Table 18(a)/(c) vs Figure 4.** These are two runs of the same stratified experiment (CAFE-In per-bin stratified Degree EG):

| Dataset | Table 18(a), Jul-27 run: first … last bin | Figure 4 run (Aug-13; = current `analysis/plots/stratified_cafe`) | ρ_S on Figure 4 run | T18(c) pair difference, Figure 4 run |
|---|---|---|---|---|
| Ego-FB | +8.49 … −1.68 | +7.93 … −1.32 | −0.96 | +2.91 (PDF +3.07) |
| Oklahoma | +7.21 … −5.27 | +6.86 … −3.61 | −1.00 | +5.01 (PDF +5.42) |
| UNC | +10.00 … −4.40 | +9.68 … −3.13 | **−0.95** | +3.05 (PDF +4.31) |
| Pokec-n | +6.10 … −1.30 | +6.07 … −0.91 | −0.99 | +5.38 (PDF +5.47) |
| Brightkite | +4.32 … +0.55 | +5.17 … +0.76 | −1.00 | +0.19 (PDF +0.23) |

Every qualitative Table 18 / Appendix C claim holds on the Figure 4 run: stratified EG declines with coreness (ρ_S ≤ −0.95; not strictly monotone on Ego-FB, UNC or Pokec-n in either run), the more peripheral bin wins every size-matched pair, and Brightkite's five non-lowest bins decline monotonically. The printed magnitudes would change, and so would two numbers in the text: "ρ_S ≤ −0.96" becomes ≤ −0.95 (UNC), and the Brightkite lowest-bin "+4.32" becomes +5.17. Both runs carry `simple_undirected_set_union` Degree-EG semantics, whereas the main tables use legacy Degree EG. That is a separate, paper-wide point.

**Code paths for the two tables still without evidence** (checked 2026-09-26; nothing executed).
- **Table 17: complete pipeline, committed in `dcea835` (2026-08-05).**
  - Pipeline: `experiments/jobs/t20_brightkite_random_control.job` (array 0–9; split seed = training seed = 42+i) → `analysis/run_stratified_cafe.py --split_file random --allow_brightkite_random_control True …` (RA, HL-GNN, Fusion, CAFE-In, CAFE-Pre; per-user stratified precision and per-bin EG) → `analysis/finalize_brightkite_random_control.py finalize` (lowest/highest-bin precision, paired 95% t-CI, per-method Spearman ρ; writes `brightkite_random_control_table.tex`).
  - Every flag the job passes resolves in `run_stratified_cafe.py` / `experiments/cafe_methods.add_common_args`.
  - As written, the job exits before any GPU work. The Eq. (8) edgewise preflight returns `False` on this checkout, so it runs only with `BK_RANDOM_ALLOW_NONEDGEWISE=True`, and the finalizer then labels outputs non-publishable.
  - The configuration behind the printed numbers is not recorded. There are two candidates, and they differ for Brightkite:

    | Config set | Brightkite CAFE-In | Brightkite CAFE-Pre |
    |---|---|---|
    | `experiments/configs/legacy_best_configs_selected.csv` (= the paper's `configs/best_configs.csv`; needs `PENALTY_REDUCTION=mean`, `CONFIGURATION_ORIGIN=legacy_test_selected`) | λ=0.05, β=5, λ_mob=0.1 | α=0.3, β=2, λ_mob=0.5 |
    | Job default: TODO3 `selected_configs.csv` (sum, corrected-EG) | λ=1.0, β=5, λ_mob=1.0 | α=2.0, β=5, λ_mob=1.0 |

- **Table 6(a): generator present, input absent.** `analysis/generate_localization_table.py::panel_a` implements the caption: lowest source-coreness bin, CAFE-In minus fusion paired within run, and P_low/P_high under each. Its input `stratified_cafe_{ds}.csv` comes from `analysis/run_stratified_cafe.py`. No existing input reproduces the PDF (§9 list above). Every available input has at most 3 runs, and Pokec-n has none in `analysis/plots/stratified_cafe/`.

---

## 10. Resolution on branch `reproducibility` (2026-09-26)

The findings above led to the `reproduce/` pipeline.  Status changes relative
to §2:

| Artifact | Audit status | Now |
|---|---|---|
| T04, T07, T11, T15, F03 | generated by untracked one-off exporters | rebuilt by `reproduce/artifacts/`, verified cell-by-cell; splices recorded in `reproduce/known_deviations.csv` |
| T06 | ORPHANED | (b) percentages rebuilt with the paper's tercile rule; (a) and absolute values need `t06_localization.job` |
| T08 | STALE (hybrid) | rebuilt from its July sources; the eight hybrid precision cells are known deviations |
| T13 | STALE (Pokec-n row) | rebuilt; Pokec-n row is a known deviation |
| T17 | MISSING | builder and job exist; results must be re-run |
| T18 | STALE (inputs overwritten) | rebuilt from the surviving 2026-07-27 intermediates |
| F01 | STALE | rebuilt from the current UNC runs (the printed render predates them) |
| F04 | ORPHANED | source found (`results/validation/20260813_184859/figure4`); rebuilt |

Code that is required for reproduction is now tracked (exact Closeness,
acceptance probe, runner hooks, `experiments/run_with.py`); the September
campaign scripts are preserved under the git tag `paper-results-provenance`.
The raw campaign trees and superseded material were moved out of the working
tree to `/var/scratch/eerdem/CAFE_archive_2026-09-26/` (listed in its
`MOVED.txt`); the files the paper needs are in `evidence/`.
