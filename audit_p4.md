# Phase 4 — hyperparameter configurations

## Controlling sources and verdict

The authoritative paper is `ToRS_paper/CAFE_ToRS_EceErdem (2).pdf`. The old reported configurations are in `analysis/best_configs.csv`. The corrected validation-selection artifacts are `results_egfix/20260723_degree_eg_v1/selected_configs.csv` and `results_egfix/20260724_degree_eg_sum_in_v1/selected_cafe_in_configs.csv`; the composed rerun choice is `results_egfix/20260724_degree_eg_sum_in_v1/todo3/selected_configs.csv`.

Overall verdict on reporting corrected results at the `analysis/best_configs.csv` values while retaining the draft's present selection text: **MISMATCH**.

This is not a harmless “fixed hyperparameters versus retuning” choice. The draft represents the printed configurations as the output of a validation-only Pareto+ASF procedure under the reported Equality-Gain objectives. The old configurations are not the output of that procedure after the Equality-Gain correction, and the repository's own manuscript source says the published pipeline selected them on the test split.

There is an independent method-identity failure before configuration choice. With the configured MLP, positive predictions have shape `[B,1]` (`backbone/layer.py:147-168`; `backbone/model.py:699-706`), while the coreness weights have shape `[B]`; `pos_probs * core_weight` therefore broadcasts to `[B,B]` (`fairness_methods/inprocessing.py:660-681`). The implementation averages cross-edge products, not Eq. (8)'s matched-edge products. Both the old and corrected CAFE-In selections optimize this implemented loss, not the CAFE-In loss written in Section 5.2, Eq. (8), p. 11. Verdict: **MISMATCH**.

## 1. What the draft says, verbatim

Appendix A.3, “CAFE Hyperparameter Tuning & Selection,” p. 33:

> “Table 14 details the search grid (210 configurations per variant/dataset). Optimal configurations are selected along the Pareto frontier using the Achievement Scalarizing Function.”

Appendix A.3, Eq. (11), pp. 33–34:

> “Selection uses the four reported measures, all treated as maximized: \(f(c)=(\mathrm{Prec@10},\mathrm{EG}_{\deg}@10,\mathrm{EG}_{\mathrm{pr}}@10,\mathrm{EG}_{\mathrm{cl}}@10)(c)\), each averaged over the repetitions of configuration \(c\). Objective values are computed on the validation edges, never on the test edges used for the reported results, so no test information enters configuration selection.”

Appendix A.3, Stage 1, p. 34:

> “Within a (method, dataset) pair, each objective is min–max normalized across that pair’s candidate set, with a degenerate range replaced by 1. We retain the non-dominated front P: configuration c is discarded if some c′ is at least as good on all four objectives and strictly better on at least one.”

Appendix A.3, Eq. (12), p. 34:

> “\(c^\star=\arg\min_{c\in P}\max_{j\in\{1,\ldots,4\}} w_jd_j(c),\quad \mathbf{w}=(0.5,0.3,0.1,0.1).\)”

Appendix A.3, p. 34:

> “The weights are fixed once, before any results were inspected, and are identical for every method and dataset.”

Table 14, p. 34, prints these grids verbatim:

> “\(\lambda_{fair}\): {0, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0}”; “\(\alpha_{pre}\): {0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0}”; “\(\beta\): {0, 0.1, 0.5, 1.0, 2.0, 5.0}”; “\(\lambda_{mob}\): {0, 0.1, 0.3, 0.5, 1.0}.”

Section 7.3, p. 21, additionally says:

> “Every row of Table 7 is tuned separately.”

Section 5.4, p. 12, makes a configuration-dependent claim:

> “Since we search \(\lambda_{mob}\in[0,1]\), \(M_v<2\) throughout and neither clipping bound is ever active; at the largest selected value \(\lambda_{mob}=0.5\) we observe \(M_v\leq1.42\) across the five training graphs.”

Verdict on whether this text unambiguously defines a validation-only selection rule for CAFE-In and CAFE-Pre: **MATCH**.

Verdict on whether the text identifies how every single-component and partial-pipeline row in Table 7 is selected: **CANNOT VERIFY FROM AVAILABLE FILES**. The sentence says each row is tuned, but Appendix A.3 only defines the full CAFE variant grid and the corrected selection outputs contain CAFE-In and CAFE-Pre, not a corrected validation-selected row for every Table 7 ablation.

The PDF never enumerates the selected per-dataset configuration keys. Appendix A.3 and Table 14, pp. 33–34, give the grid and rule, while Section 5.4, p. 12, exposes only one claimed maximum selected mobility value. A reader cannot recover the reported operating points from the paper itself. Verdict on paper-level configuration reproducibility: **MISMATCH**.

### Terminology: `alpha_sf`

No parameter or field named `alpha_sf` occurs in the inspected current PDF, `analysis/best_configs.csv`, the corrected `selected_configs.csv` files, or the Python sources. The paper uses `alpha_pre` for the pre-processing exponent and “ASF” for the achievement scalarizing function; the corrected artifacts use `preproc_alpha` and `asf_score`. Verdict on what `alpha_sf` denotes: **CANNOT VERIFY FROM AVAILABLE FILES**. The audit therefore covers both plausible intended terms rather than silently equating them.

## 2. What the corrected selection code does

| Selection element | Code evidence | Verdict against Appendix A.3 | Note |
|---|---|---|---|
| Candidate grid | `experiments/egfix_protocol.py:91-99,260-269`; strict finalizer description at `analysis/egfix_finalize_selection.py:1-6` | **MATCH** | The strict campaign requires the printed penalty, beta, and mobility grids and checks `210` configurations per variant and dataset. |
| Objectives and weights | `experiments/egfix_protocol.py:78-84`; serialized rule at `analysis/egfix_finalize_selection.py:891-904` | **MATCH** | The objectives are `Precision@10`, `Equality_Gain@10`, `Pagerank_EG@10`, and `Closeness_EG@10`; weights are `(0.5, 0.3, 0.1, 0.1)`. |
| Validation-only boundary | `experiments/run_cafe.py:278-304,311-316`; protocol fields at `experiments/egfix_protocol.py:270-290` | **MATCH** | Strict selection requires `evaluation_split=valid`, `k_values=[10]`, no demographic computation, all EG dimensions, and per-model reseeding; it disables final test evaluation. |
| Repetitions | `experiments/egfix_protocol.py:276-290`; validation at `analysis/egfix_finalize_selection.py:1-6` | **MATCH** | The corrected campaign records `selection_runs=3` and requires every objective observation. |
| Pareto dominance | `analysis/egfix_finalize_selection.py:329-369` | **MATCH** | All objectives are maximized; a point is retained only if no other point weakly improves every objective and strictly improves one. |
| Full-set normalization | `analysis/egfix_finalize_selection.py:373-393` | **MATCH** | Each objective is min–max normalized; a constant range uses `1.0`. |
| Front renormalization and ASF | `analysis/egfix_finalize_selection.py:395-437` | **MATCH** | The front is renormalized and the maximum weighted loss is minimized. Exact ties use the lexicographically smallest `config_key`, a behavior the paper does not state. |
| Missing values | Strict finite-value checks culminate in `analysis/egfix_finalize_selection.py:350-359`; the legacy generator drops missing-objective rows at `analysis/generate_final_tables.py:381-389` | **CANNOT VERIFY FROM AVAILABLE FILES** | The current PDF does not state a missing-objective policy. The strict corrected finalizer rejects non-finite inputs; the legacy generator silently excludes them and falls back to the first row if none remain. |
| Error fallback | `analysis/generate_final_tables.py:403-411` | **MISMATCH** | The legacy paper-table generator silently substitutes a weighted-sum optimum if non-dominated sorting raises. Appendix A.3 explicitly distinguishes the Chebyshev ASF from a weighted sum and does not disclose this fallback. |
| Corrected-semantics filtering | `analysis/generate_final_tables.py:93-113` | **MISMATCH** | The legacy generator's corrected-EG filter is off by default. It can select across pre-fix and corrected EG values in a mixed results tree, contradicting a single well-defined objective. The strict finalizer avoids this. |
| Test/validation default | `experiments/run_cafe.py:159-163`; headline job call at `experiments/jobs/t04_cafe.job:118-122` | **MISMATCH** | `selection_mode` defaults to `False`, `evaluation_split` defaults to `test`, and the headline fixed-config job does not pass the strict selection flags. Those headline JSONs are test outputs, not validation-selection artifacts. |

The checked-in source of the appendix states the problem without ambiguity: `ToRS_paper/sections/A3-asf-selection.tex:5-14` says that the corrected `todo3` campaign selects on validation, but that the “PUBLISHED pipeline” scores candidates on the test split and that the headline JSONs contain no validation metrics. Its disclosure block says selecting one of `210` candidates on the reported test split gives CAFE a comparison advantage over untuned baselines (`ToRS_paper/sections/A3-asf-selection.tex:95-110`). This source statement directly contradicts the compiled PDF's assertion that “no test information enters configuration selection.” Paper-to-artifact verdict: **MISMATCH**.

The claim that the ASF weights were fixed before any result was inspected is historical intent, not encoded provenance. The constants exist in code (`experiments/egfix_protocol.py:78-84`; `analysis/generate_final_tables.py:42-51`), but no timestamped preregistration or immutable pre-result record establishes when they were chosen. Verdict: **CANNOT VERIFY FROM AVAILABLE FILES**.

## 3. What the written corrected procedure selects

### 3.1 Historical mean-reduction selection (not a faithful Eq. (8) implementation)

Equation (8), Section 5.2, p. 11, explicitly averages its added penalty over positive batch edges. The strict corrected-EG campaign freezes `PENALTY_REDUCTION = "mean"` (`experiments/egfix_protocol.py:70-84`). Because of the broadcasting defect, this matches only the declared reduction, not the edgewise formula. Its complete validation-only output for the implemented loss is `results_egfix/20260723_degree_eg_v1/selected_configs.csv`:

| Dataset | Corrected CAFE-In selection | Corrected CAFE-Pre selection | Old `analysis/best_configs.csv` CAFE-In | Old `analysis/best_configs.csv` CAFE-Pre | Verdict |
|---|---|---|---|---|---|
| ego-facebook | `lambda_fair_0.0_beta_0.5_lam_1.0` | `preproc_alpha_1.0_beta_0.1_lam_1.0` | `lambda_fair_0.3_beta_1.0_lam_0.5` | `preproc_alpha_0.05_beta_0.5_lam_0.3` | **MISMATCH** |
| oklahoma | `lambda_fair_0.0_beta_1.0_lam_1.0` | `preproc_alpha_1.0_beta_0.5_lam_1.0` | `lambda_fair_0.3_beta_5.0_lam_0.5` | `preproc_alpha_1.0_beta_0.5_lam_0.5` | **MISMATCH** |
| unc | `lambda_fair_0.0_beta_2.0_lam_1.0` | `preproc_alpha_1.0_beta_2.0_lam_1.0` | `lambda_fair_1.0_beta_5.0_lam_0.5` | `preproc_alpha_1.0_beta_1.0_lam_0.5` | **MISMATCH** |
| pokec_n | `lambda_fair_1.0_beta_0.1_lam_1.0` | `preproc_alpha_0.01_beta_5.0_lam_1.0` | `lambda_fair_0.05_beta_1.0_lam_0.1` | `preproc_alpha_0.05_beta_2.0_lam_0.5` | **MISMATCH** |
| brightkite | `lambda_fair_0.1_beta_5.0_lam_1.0` | `preproc_alpha_2.0_beta_5.0_lam_1.0` | `lambda_fair_0.05_beta_5.0_lam_0.1` | `preproc_alpha_0.3_beta_2.0_lam_0.5` | **MISMATCH** |

Every value in this table is copied from the named CSV row; none is reconstructed from a metric table. The old and corrected configuration keys differ for every displayed method–dataset pair. Reporting the old values as if the corrected validation-only rule selected them is therefore **MISMATCH**.

The corrected mean selection also contradicts Section 5.4's “largest selected value `lambda_mob=0.5`” claim: every displayed corrected CAFE-In and CAFE-Pre key has `lam_1.0`. Verdict: **MISMATCH**.

More seriously, the corrected rule selects `lambda_fair_0.0` for ego-facebook, oklahoma, and unc. At zero, the in-processing term vanishes in the implemented loss (`fairness_methods/inprocessing.py:87-125`). On those datasets, the selected “CAFE-In” is not evidence that the in-processing intervention contributes. Verdict on carrying over the draft's component-attribution interpretation to those selections: **MISMATCH**.

For pokec_n and brightkite, where the mean-selected penalty is nonzero, the broadcasting defect above means the selected model still does not implement Eq. (8). Verdict on treating any row in the corrected CAFE-In selection table as selection for the paper-defined in-processing method: **MISMATCH**.

### 3.2 Sum-reduced alternative

The repository also contains a second corrected selection because the AUC base loss is sum-reduced while the historical fairness penalty is mean-reduced. The code documents the scaling issue and the `sum` alternative at `fairness_methods/inprocessing.py:24-47,99-125`. This is not the same literal implementation as Eq. (8)'s displayed `1/|E_batch|` term unless the paper also defines and rescales `L_base` consistently.

`results_egfix/20260724_degree_eg_sum_in_v1/selected_cafe_in_configs.csv` selects:

| Dataset | Sum-reduced corrected CAFE-In selection |
|---|---|
| ego-facebook | `lambda_fair_0.0_beta_0.1_lam_0.5` |
| oklahoma | `lambda_fair_0.0_beta_0.5_lam_1.0` |
| unc | `lambda_fair_0.0_beta_5.0_lam_1.0` |
| pokec_n | `lambda_fair_0.0_beta_5.0_lam_1.0` |
| brightkite | `lambda_fair_1.0_beta_5.0_lam_1.0` |

The composed `todo3/selected_configs.csv` pairs those CAFE-In rows with the mean-campaign CAFE-Pre rows because reduction is not consulted by preprocessing (`jobs/egfix_sum_todo3/protocol.py:1-10,114-125`). The production job itself states that zero `lambda_fair` makes CAFE-In identical to fusion plus boost on ego-facebook, oklahoma, unc, and pokec_n and skips the redundant CAFE-P arm (`experiments/jobs/t04_cafe_asf.job:23-37,50-67,107-116`).

Verdict on treating this composed selection as the output of the method and loss currently written in the PDF: **MISMATCH**. It is defensible only after explicitly rewriting Eq. (8), the penalty-reduction protocol, the method interpretation, and every attribution claim affected by the zero penalty.

The sum reduction does not repair the broadcasting defect: it rescales the already cross-broadcast scalar at `fairness_methods/inprocessing.py:104-125`. In the composed campaign, four datasets disable the penalty and Brightkite applies the wrong cross-edge penalty. Hence no selected CAFE-In row in that campaign evaluates the nonzero edgewise penalty in Eq. (8). Verdict: **MISMATCH**.

### 3.3 Other Table 7 configurations

The corrected validation campaigns select CAFE-In and CAFE-Pre only. They do not produce corrected validation selections for CI-In, CI-Pre, MB-Post, CAFE-P, CAFE-In-B, or CAFE-Pre-B. The old values for those rows remain in `analysis/best_configs.csv`, and the corrected fixed-config artifacts merely evaluate those old cells where available. Verdict on “Every row of Table 7 is tuned separately” under corrected EG semantics: **CANNOT VERIFY FROM AVAILABLE FILES**.

## 4. Argument against reusing the old configurations

The strongest case against reuse is decisive:

1. The selection objective changed. Degree and PageRank EG are ASF objectives in Eq. (11), and those are the metrics whose graph construction was corrected (`metrics/equality_gain.py:62-144`; `metrics/centrality_equality_gain.py:89-125,218-264,518-592`). A configuration selected with the superseded objective is not the optimizer of the corrected objective.
2. The corrected validation campaign actually returns different keys for every displayed CAFE-In/CAFE-Pre dataset cell. This is direct artifact evidence, not a hypothetical sensitivity argument.
3. The old paper-table pipeline selected on the reported test split. The repository says so in `ToRS_paper/sections/A3-asf-selection.tex:5-14,95-110`; the runner defaults and headline job are consistent with that admission (`experiments/run_cafe.py:159-163`; `experiments/jobs/t04_cafe.job:118-122`). Retaining the old keys while claiming validation-only selection is false.
4. The legacy generator can mix corrected and uncorrected EG unless an environment switch is explicitly enabled, and it has an undisclosed weighted-sum fallback (`analysis/generate_final_tables.py:93-113,403-411`). `analysis/best_configs.csv` contains no manifest, selection split, EG-semantics marker, candidate set, ASF score, or selection-run lineage. Its provenance is inadequate for the protocol claimed.
5. Reuse changes the scientific estimand. “Performance of a configuration frozen before the metric repair” is a valid estimand if disclosed. “Performance of the corrected validation-selected CAFE method” is a different estimand. The current text claims the second while the proposed reuse supplies the first.
6. Reuse conceals a substantive story change. Corrected selection often chooses zero in-processing penalty. The old configuration preserves an apparently active intervention that the corrected rule rejects. That is exactly the sort of result-dependent choice a hostile reviewer will characterize as protecting the narrative.
7. Neither reuse nor corrected reselection can validate the paper-defined CAFE-In until the broadcasting defect is fixed. Old nonzero-penalty runs use the wrong cross-edge loss; most corrected sum-selected runs avoid that loss only by setting its coefficient to zero.

Verdict for silent reuse under the current prose: **MISMATCH**.

## 5. The limited argument for reuse

There is one defensible argument, but it requires abandoning the current selection claim. The old keys were fixed before inspecting the corrected EG outputs; keeping them can prevent a second round of tuning to the repaired metric and can measure how a previously frozen system behaves under corrected evaluation. That design may be useful as a robustness analysis.

It does not establish that the keys are Pareto+ASF choices under corrected EG, that selection was test-independent, or that they are the operating points the written procedure returns. Verdict on reuse as a disclosed, frozen legacy-configuration sensitivity analysis: **MATCH**. Verdict on reuse as the primary result while leaving Appendix A.3 unchanged: **MISMATCH**.

## 6. Defensible options and reviewer risk

### Option A — fix the paper-defined loss, then repeat literal corrected validation selection

First make the positive prediction and coreness-weight vectors conform edgewise, then rerun the validation grid under Eq. (8)'s mean penalty. The existing `results_egfix/20260723_degree_eg_v1/selected_configs.csv` cannot be reused as selection evidence for the repaired method because it was generated by the broadcasted loss. Freeze the new selected keys, run independent tests, and regenerate tables, figures, Pareto fronts, paired tests, ablations, and claims. State the deterministic tie break and missing-value policy. Do not claim an in-processing contribution wherever the repaired selection chooses `lambda_fair=0.0`.

Reviewer risk: lower protocol risk, but high narrative risk. The method attribution changes and the currently completed corrected aggregate evidence is incomplete. Evidence verdict that the existing draft already does this: **MISMATCH**.

### Option B — fix configurations a priori and rewrite the protocol honestly

Declare the `analysis/best_configs.csv` keys frozen legacy operating points chosen before the repair; remove every statement that corrected validation Pareto+ASF selected them. Disclose that their original selection used the same test split, superseded EG semantics, and the broadcasted in-processing loss. Present corrected-ASF selections as a sensitivity analysis of that historical implementation, not as post-hoc replacement winners or evidence for Eq. (8). Apply the same a-priori rule to every method.

Reviewer risk: high. This removes the false protocol claim but does not cure selection optimism or the asymmetric treatment of baselines. A reviewer can still reject the main comparison as test-tuned. Evidence verdict that this would be honest if fully disclosed: **MATCH**. Evidence verdict that the present draft contains that disclosure: **MISMATCH**.

### Option C — define the sum-reduced method as the actual method, then rerun coherently

Repair the edgewise broadcasting first. Then rewrite Eq. (8) and the training protocol so the scale of `L_base`, the fairness term, batch size, and `num_neg` agree with `penalty_reduction=sum` (`fairness_methods/inprocessing.py:24-47,99-125`). The existing sum-selected rows were selected before that repair and therefore cannot be reused as selections for the repaired algorithm. Reselect or predeclare all ablation controls consistently, state when zero penalty collapses CAFE-In to CAFE-P, and rerun them.

Reviewer risk: high method-drift risk. This is a changed algorithm, not merely corrected measurement, and most selected CAFE-In rows do not exercise the claimed in-processing stage. Evidence verdict that the current PDF describes this method: **MISMATCH**.

### Option D — redesign selection/evaluation separation

Use nested resampling or reserve an untouched final evaluation split: select configurations inside each training/validation fold, then evaluate once on data never used by ASF. Alternatively, report the complete validation Pareto set and predeclare one deployment rule independent of the test results. Apply equivalent tuning budgets or clearly separate fixed-paper baselines from tuned methods.

Reviewer risk: lowest inferential risk and highest computational cost. It is the cleanest answer to the existing test-selection defect. Evidence verdict that the current experiment tree contains such a completed design: **CANNOT VERIFY FROM AVAILABLE FILES**.

There is no defensible fifth option in which the old keys remain primary, Appendix A.3 remains unchanged, and the corrected results are presented as though selected by that appendix. That combination is **MISMATCH**.

## 7. What could not be checked, and why

1. What the author intended by `alpha_sf`: **CANNOT VERIFY FROM AVAILABLE FILES** because that identifier appears in none of the inspected paper, code, configuration, or corrected-selection artifacts.
2. Whether ASF weights were fixed before results: **CANNOT VERIFY FROM AVAILABLE FILES** because no preregistration or immutable pre-result timestamp establishes the chronology.
3. The exact generating invocation for `analysis/best_configs.csv`: **CANNOT VERIFY FROM AVAILABLE FILES** because the CSV has no source manifest, command record, EG-semantics field, selection split, or ASF diagnostics.
4. Corrected validation selection for every Table 7 ablation: **CANNOT VERIFY FROM AVAILABLE FILES** because the strict corrected selection artifacts cover CAFE-In and CAFE-Pre only.
5. Final test performance at every mean-campaign corrected selection: **CANNOT VERIFY FROM AVAILABLE FILES** because the corresponding aggregate rerun campaign is incomplete.
6. A complete primary result under the sum-selected `results_asf/` configuration set: **CANNOT VERIFY FROM AVAILABLE FILES** because Brightkite CAFE-Pre-B and Pokec-n CAFE-Pre are partial, Pokec-n In-B/Pre-B are absent, and no complete regenerated paper-table artifact exists.
7. Whether either corrected selection campaign was formally approved before reruns: **CANNOT VERIFY FROM AVAILABLE FILES** because the workflow requires human inspection, but an approval record is not present in the selected-config artifacts.
8. What the paper-defined edgewise CAFE-In would select after repairing the broadcasting defect: **CANNOT VERIFY FROM AVAILABLE FILES** because every available CAFE-In selection was generated by the current broadcasted implementation.

Phase 4 ends here.
