# Phase 5 — RQ adequacy and editorial recommendation

## Scope, snapshot, and decision rule

Authoritative paper: `ToRS_paper/CAFE_ToRS_EceErdem (2).pdf`. To keep Phases 3 and 5 comparable, I adopt Phase 3's frozen live-evidence boundary, `2026-08-04T10:10:30+02:00`. Files described as in flight below are assessed only as recorded at that cutoff in `audit_p3.md`; later appends are excluded.

I use **MATCH**, **MISMATCH**, and **CANNOT VERIFY FROM AVAILABLE FILES** only as evidence verdicts. The requested editorial recommendations—accept, minor, major, or reject—are labelled separately and are not evidence verdicts.

The completed integrity-attested corrected campaign under `results_egfix/20260724_degree_eg_sum_in_v1/todo3/` contains ten fixed-split aggregate seeds and three fixed-split stratified seeds for RA, HL-GNN, matched-beta Fusion, CAFE-In, and CAFE-Pre on every dataset; its protocol is frozen in `jobs/egfix_sum_todo3/protocol.py:33-61,76-99`, and completion/integrity is recorded in `finalized/todo3_integrity.json`. It does **not** contain the full baseline set, the full ablation set, demographic baselines, split-resampled inference, or a selected-configuration acceptance analysis.

There are two method-identity defects that affect RQ2–RQ4. First, the draft defines the in-processing penalty with an explicit batch mean (Section 5.2, Eq. (8), p. 11), while the completed corrected campaign fixes `penalty_reduction = "sum"` (`jobs/egfix_sum_todo3/protocol.py:45-48,93-99`), which multiplies the fairness penalty by the number of positive examples (`fairness_methods/inprocessing.py:99-125`). Second, the configured MLP emits positive predictions with shape `[B,1]` (`backbone/layer.py:147-168`; `backbone/model.py:699-706`), while the coreness weights have shape `[B]`; their product broadcasts to `[B,B]` and averages cross-edge rather than matched-edge products (`fairness_methods/inprocessing.py:660-681`). The sum reduction only rescales this wrong scalar. Verdict that the completed corrected CAFE-In campaign evaluates the method written in Eq. (8): **MISMATCH**.

This defect is exhaustive for the completed sum-selected CAFE-In evidence: Ego-FB, Oklahoma, UNC, and Pokec-n set `lambda_fair = 0.0`, so they do not exercise Eq. (8), while Brightkite sets `lambda_fair = 1.0` and exercises the broadcasted cross-edge penalty (`results_egfix/20260724_degree_eg_sum_in_v1/todo3/selected_configs.csv`). No dataset in that campaign evaluates the intended nonzero edgewise penalty. Verdict: **MISMATCH**.

The supplementary PDF does not cure any gap. Its Table 4 reports, for example, Brightkite RA Precision@10 as `0.0675` (Supplementary Table 4, p. 2), whereas the current draft reports `0.0222` (Table 4(a), p. 15) and the corrected aggregate artifact reports `0.02235562310030365` with std `0.0` (`finalized/aggregate_metrics_summary.csv`, Brightkite/RA row). It is an older RecSys result set, not a corrected ToRS result source. Verdict on using Supplementary Tables 4–6 (pp. 2–6) to fill current corrected cells: **MISMATCH**.

## Top findings

1. Only RQ3 has a complete, internally checked corrected dataset capable of answering the question descriptively. The current RQ3 answer nevertheless reports the wrong source-side conclusion and the wrong five-dataset mechanism attribution. Verdict: **MISMATCH**.
2. RQ2's full comparison, Pareto count, and Holm-corrected claims do not exist as one corrected campaign. The five-method corrected subset changes the RA comparison in multiple datasets. Verdict on the full RQ2 answer: **CANNOT VERIFY FROM AVAILABLE FILES**.
3. Corrected selection sets `lambda_fair` to `0.0` for CAFE-In on Ego-FB, Oklahoma, UNC, and Pokec-n, so the claimed intervention is absent there; Brightkite's nonzero case uses the broadcasted cross-edge penalty rather than Eq. (8). The paper's universal three-stage attribution in RQ4 is false for the selected system. Verdict: **MISMATCH**.
4. RQ5's fixed-split point estimates exist only unevenly at the corrected selected configurations, and the split-resampled demographic tests are both incomplete and still parameterized by the old configurations. Verdict: **CANNOT VERIFY FROM AVAILABLE FILES**.
5. RQ6's rendered Table 11 is stale, while its current raw CSV schema carries no configuration or graph-semantics provenance. Its two relaxed models are also endogenous descriptive transformations of the same test labels, not observed acceptance behavior. Verdict: **CANNOT VERIFY FROM AVAILABLE FILES**.

Overall evidence verdict for the draft as a report of the results now on disk: **MISMATCH**.

Overall editorial recommendation: **reject**. This is not a request for cosmetic revision. The central comparison uses stale affected metrics and configurations inconsistent with the stated selection rule; the implemented corrected CAFE-In is not the method defined in the paper; every RQ other than RQ3 lacks a coherent corrected evidence package; and the one corrected RQ that can be answered is answered incorrectly in the manuscript. A substantially rebuilt submission could be reviewable, but this draft is not.

There is also a submission-identity defect: the compiled “ToRS draft” names the RecSys '26 proceedings in its ACM reference and carries a J. ACM footer (PDF p. 1). Verdict that the supplied PDF is formatted and identified as a ToRS manuscript: **MISMATCH**.

## RQ1 — diagnosis

**Question.** “Does coreness capture structural bias in link recommendation beyond what degree explains, consistently across method families?” (Introduction, RQ1, p. 2).

### Does the evidence on disk answer it?

No. The evidence answers a narrower question. Figure 1(a) conditions jointly on degree and coreness only for HL-GNN on UNC; the cross-method panel is marginally stratified by coreness, not conditionally on degree (Section 4 and Figure 1, pp. 8–9). The draft then generalizes to every method on all five networks (Section 4, p. 9).

The corrected completed campaign provides partial confirmation of the source-coreness Degree-EG gradient for its narrower method set. On UNC, for example, RA moves from `9.12506141723754` with std `0.0` in bin `[1-9]` to `-3.113760943130688` with std `0.0` in bin `[22-24]`; HL-GNN moves from `8.697970436199531` with std `0.06162873385992925` to `-3.0384829771207955` with std `0.050063473825612756` (`finalized/stratified_degree_eg_summary.csv`, UNC rows). But the campaign's method set is RA, HL-GNN, Fusion, CAFE-In, and CAFE-Pre (`jobs/egfix_sum_todo3/protocol.py:37-44`), not the diagnostic set CN, AA, RA, LightGCN, and HL-GNN named in Section 4 (p. 8). No complete corrected Figure 1 source covers the stated diagnostic method families.

Evidence verdict on the all-dataset, cross-family, degree-controlled RQ1 claim: **CANNOT VERIFY FROM AVAILABLE FILES**.

### Strongest hostile-reviewer objection

The paper confuses a one-network binned illustration with evidence that coreness explains bias beyond degree generally. Degree and coreness are correlated; a heatmap for one model on UNC is not a conditional effect estimate, and the marginal all-network profiles do not resolve that confounding. A hostile reviewer can correctly demand a prespecified conditional model, matched-degree analysis, or repeated joint stratification across datasets and method families.

### Does the paper pre-empt it?

No. It acknowledges intermediate dips and small cells (Section 4, p. 8) and the Brightkite reversal (Section 4, p. 9), but it does not test the degree-controlled claim outside UNC or quantify uncertainty for a conditional coreness effect. Appendix C controls bin size for stratified Degree EG, not degree–coreness confounding in Precision@10 (Appendix C, pp. 35–38).

Editorial recommendation for RQ1: **major**. The diagnosis is plausible, but the evidence does not establish the question as phrased.

## RQ2 — effectiveness

**Question.** The draft asks how CAFE compares with heuristic, GNN, structural-debiasing, and tail-node baselines on Precision@10 and all three EG dimensions, including whether tail-node accuracy alone yields equality (Introduction, RQ2, p. 2). Its answer claims a CAFE variant is Pareto-optimal in `13` of `15` dataset-by-EG panels (Section 7.1 answer box, p. 17).

### Does the evidence on disk answer it?

No. The completed corrected aggregate covers only five methods, so it can answer CAFE versus RA, HL-GNN, and matched-beta Fusion descriptively, but not the full baseline question or the `13`-of-`15` frontier claim. Figure 3's exact source CSV is absent, its rendered method set is inconsistent with Table 4, and the full corrected Pokec-n comparison remains incomplete (Figure 3, p. 18; Table 4, pp. 15–17).

The corrected five-method subset also changes concrete claims in the prose. Its CAFE-In rows are results for the zero-penalty/broadcasted implementation described above, not the paper-defined Eq. (8) method:

- Brightkite RA Precision@10 is `0.02235562310030365` with std `0.0`, while corrected selected CAFE-In is `0.027443768996960105` with std `0.0003897615847342914` (`finalized/aggregate_metrics_summary.csv`). This is inconsistent with the draft's claim that CAFE-In is significantly worse than RA on Brightkite Precision@10 (Section 7.1, p. 14). Verdict: **MISMATCH**.
- Ego-FB RA Precision@10 is `0.23734674630619684` with std `0.0`, while corrected selected CAFE-In is `0.22293932725558357` with std `0.0012578198182463499` (same CSV). This is inconsistent with the draft's statement that CAFE-In matches RA precision there (Section 7.1, pp. 14–15). Verdict: **MISMATCH**.
- The corrected subset does retain a substantive equality result: on Ego-FB, CAFE-In Degree EG is `14.263469302228273` with std `0.09290142394666079` and PageRank EG is `16.01730419668338` with std `0.20577727054449413`, versus HL-GNN's `13.278781096847059` with std `0.18682550499762499` and `10.668614875735429` with std `0.3743272967001911` (same CSV). That is evidence for one comparison, not for the full RQ.

The inferential evidence is also not current. At the frozen read, the split-resampled files contained seeds `{0,1}` for Brightkite, `{0,1,2,3,4}` for Ego-FB, `{0,1,2,3,4,5,6}` for Oklahoma, `{0,1,2,3,4}` for UNC, and no Pokec-n file (`audit_p3.md`, “Controlling evidence and frozen read,” quoting `results_repro/*/significance/per_seed_metrics_*.csv`). More fundamentally, those CSVs serialize the old configurations—e.g. Oklahoma CAFE-In `(0.3,5.0,0.5)`—rather than the corrected selected `(0.0,0.5,1.0)` configuration (`results_egfix/20260724_degree_eg_sum_in_v1/todo3/selected_configs.csv`). Completing those old-config jobs would not validate the corrected selected method.

Evidence verdict on the full RQ2 answer and its Pareto/significance claims: **CANNOT VERIFY FROM AVAILABLE FILES**.

### Strongest hostile-reviewer objection

The comparison is selection-biased and method-asymmetric. The PDF says that `210` CAFE configurations per variant and dataset are selected on validation only (Appendix A.3, pp. 33–34), but the checked-in manuscript source states that the published pipeline scored them on the reported test split and that baselines were not given the same selection budget (`ToRS_paper/sections/A3-asf-selection.tex:5-14,95-110`). The headline runner defaults to `selection_mode=False` and `evaluation_split=test` (`experiments/run_cafe.py:159-163`). A reviewer need not accept a Pareto advantage produced by selecting CAFE from a large test-scored grid while holding baselines fixed.

### Does the paper pre-empt it?

No. It asserts the opposite: “no test information enters configuration selection” (Appendix A.3, pp. 33–34). The corrected validation campaign avoids that defect but selects different configurations and a sum-reduced CAFE-In, so it cannot retroactively validate Table 4 or Figure 3.

Editorial recommendation for RQ2: **reject**. This is the primary effectiveness claim, and neither the comparison set nor the selection/inference protocol is currently defensible.

## RQ3 — stratified effect

**Question.** “Does the intervention act where the diagnosis located the problem; that is, how are precision and equalization gains distributed across coreness strata?” (Introduction, RQ3, p. 2).

### Does the evidence on disk answer it?

Yes for the selected implementation as a descriptive fixed-split analysis, but not for the paper-defined active Eq. (8) intervention. The campaign contains three training seeds on every dataset for source precision, source-stratified Degree EG, and target exposure (`jobs/egfix_sum_todo3/protocol.py:56-61,132-147`; `finalized/todo3_integrity.json`). Availability verdict: **MATCH**. Method-identity verdict: **MISMATCH**.

The current answer box is nevertheless wrong. It says that, beyond Fusion, lowest-bin Precision@10 moves by at most `0.0022` and that CAFE leaves the source-side profile intact (Section 7.2 answer box, p. 21). The corrected Oklahoma lowest bin `[1-9]` is `0.049707602339181284` with std `0.0025383081186211337` for Fusion and `0.024561403508771933` with std `0.002740438482774265` for CAFE-In (`finalized/stratified_precision_summary.csv`). Brightkite's lowest bin `[1-3]` is `0.03769252310277233` with std `0.00027580111458404104` for Fusion and `0.048305796695603476` with std `0.0005827121827691128` for CAFE-In (same CSV). Verdict on the answer-box source-precision statement: **MISMATCH**.

The target-side result is strong and broader than the draft reports. On Oklahoma, CAFE-In versus Fusion has peripheral mean-exposure relative change `206.0523414239142` percent with std `10.815970844981042`, `positive_runs = 3`, while the core change is `-17.785102817788157` percent with std `0.21904802398774212`, `negative_runs = 3` (`finalized/target_exposure_cafe_in_vs_fusion_summary.csv`). The same file records periphery-up/core-down in all three runs on each of the five datasets. Verdict on the direction of target redistribution in the completed campaign: **MATCH**.

The mechanism attribution is not what the paper says. The selected CAFE-In configuration has `lambda_fair = 0.0` on Ego-FB, Oklahoma, UNC, and Pokec-n, and `1.0` on Brightkite (`todo3/selected_configs.csv`). With zero weight, the implemented total loss is the base loss (`fairness_methods/inprocessing.py:99-125`). Thus, beyond matched-beta Fusion, the active coreness intervention is only the mobility boost on four datasets. Brightkite has a nonzero coefficient, but its penalty is the broadcasted cross-edge quantity rather than Eq. (8). No dataset supplies a valid penalty-plus-boost contrast for the method written in the paper. Verdict on attributing the five-dataset shift jointly to “the coreness penalty and the boost” (Section 7.2, pp. 20–21): **MISMATCH**.

### Strongest hostile-reviewer objection

The paper labels boost-only contrasts as evidence for a joint penalty-plus-boost mechanism on four datasets; on the remaining dataset it uses a wrongly broadcasted penalty. This is not a subtle attribution problem. The corrected evidence localizes exposure movement, but it provides no valid instance in which the intended Eq. (8) penalty causes it.

### Does the paper pre-empt it?

No. Table 3 marks the loss penalty as present in every CAFE-In pipeline (p. 14), and the RQ3 answer attributes the target shift to “the coreness penalty and the boost” (p. 21). Neither location discloses that the selected penalty is inactive on four datasets. The paper also does not disclose that the localization evidence has only three fixed-split training seeds and no per-bin hypothesis tests.

Editorial recommendation for RQ3: **major**. No new model training is needed to answer this RQ descriptively, but Figure 4, Table 6, the source-side prose, the answer box, and the mechanism attribution must be replaced.

## RQ4 — component attribution

**Question.** “What is the contribution of each CAFE component to recommendation precision and Equality Gain?” (Introduction, RQ4, p. 2). The answer states that all stages are complementary, that RA fusion supplies precision and much of the Degree-EG gain, and that the coreness-aware stages add equality at no measurable precision cost on Oklahoma and UNC (Section 7.3 answer box, p. 21).

### Does the evidence on disk answer it?

No. The corrected validation campaigns select only CAFE-In and CAFE-Pre. They do not provide corrected selections for CI-In, CI-Pre, MB-Post, CAFE-P, CAFE-In-B, or CAFE-Pre-B, despite the claim that every Table 7 row is tuned separately (Section 7.3, p. 21). Corrected Pokec-n ablation cells are mostly absent, and Table 7 is stale (Table 7, pp. 22–23).

The completed five-method campaign can support a narrower Fusion-versus-full-pipeline contrast, but it contradicts the stated Oklahoma precision result. Oklahoma Fusion Precision@10 is `0.1821621621621649` with std `0.0012015382256869067`; CAFE-In is `0.1774208494208519` with std `0.0012342872430551267` (`finalized/aggregate_metrics_summary.csv`). The same rows report Fusion/CAFE-In Degree EG `9.24047457523029`/`12.689214532871071`, PageRank EG `6.878702613570782`/`11.73183910241005`, and Closeness EG `11.28107601773722`/`25.77481360054994`. The corrected system buys substantial equality on Oklahoma, but not at the paper's claimed negligible precision cost. Verdict on that claim: **MISMATCH**.

More decisively, `lambda_fair = 0.0` removes the in-processing component on four datasets, and the remaining nonzero case computes the wrong broadcasted penalty. Verdict on the universal “stages are complementary” conclusion for selected CAFE-In: **MISMATCH**. Evidence verdict on the contribution of *each* component under a coherent corrected selection protocol: **CANNOT VERIFY FROM AVAILABLE FILES**.

### Strongest hostile-reviewer objection

The claimed three-stage attribution is structurally unidentifiable from the reported selected system. A component whose selected coefficient is zero cannot receive causal credit; the only nonzero selected case implements a different loss; and independently tuning every ablation changes several factors at once rather than estimating a controlled remove-one-component effect.

### Does the paper pre-empt it?

No. It calls the matched-beta test a controlled contrast and every row separately tuned (Section 7.3, p. 21), but does not disclose the zero penalty or provide corrected selected configurations for the ablation rows. The supplementary full ablation table is an older result set (Supplementary Table 6, p. 6), so it does not repair the problem.

Editorial recommendation for RQ4: **reject**. Component attribution is a stated contribution, yet the selected method disables a claimed component on most datasets and the required corrected ablations do not exist.

## RQ5 — demographic byproduct

**Question.** “Can coreness-aware structural debiasing improve demographic fairness without sensitive attributes, and through what mechanism?” (Introduction, RQ5, p. 2). The answer claims competitive DeltaSP at much higher precision, a direction predictable from group–coreness coupling, and no DeltaEO benefit (Section 7.4 answer box, p. 24).

### Does the evidence on disk answer it?

No. Demographic metrics are not directly changed by the EG computation, but they are configuration- and recommendation-dependent. The corrected selected configurations differ from the Table 8 configurations, and several current point estimates therefore differ materially.

For Oklahoma CAFE-In, Table 8 reports Precision@10 `0.2004 ± 0.0007`, DeltaSP `0.923 ± 0.011` times `10^-3`, and DeltaEO `13.0 ± 2.2` times `10^-3` (Table 8(a), p. 24). The corrected selected-config artifact reports Precision@10 `0.1771042471042495` with std `0.001912113192467967`, DeltaSP `0.0006329989872368667` with std `0.00006129563330726671`, and DeltaEO `0.02429501704734031` with std `0.0037523359402928703` (`results_asf/oklahoma/cafe_in/results_cafe_in_oklahoma_20260729_102333.json`, config `lambda_fair_0.0_beta_0.5_lam_1.0`). Verdict that Table 8 reports the corrected selected method: **MISMATCH**.

Coverage is incomplete. Oklahoma and UNC have ten-run selected-config CAFE-In/CAFE-Pre artifacts, and Pokec-n CAFE-In has ten runs, but Pokec-n CAFE-Pre has only `completed_runs = 3` (`results_asf/pokec_n/cafe_pre/results_cafe_pre_pokec_n_20260729_132534.json.partial.json`). At the frozen read, the split-resampled significance CSV had Oklahoma seeds `{0,1,2,3,4,5,6}`, UNC seeds `{0,1,2,3,4}`, and no Pokec-n file (`audit_p3.md`, frozen-read ledger); moreover, those CSVs encode the old configurations and blank selected-config hashes. Table 17 claims ten resampled splits for all three datasets (Table 17, p. 37). Verdict on the current Table 9/17 significance and mechanism claims: **CANNOT VERIFY FROM AVAILABLE FILES**.

### Strongest hostile-reviewer objection

The “predictable mechanism” is post hoc and unvalidated. It is formulated and checked on the same three attribute datasets, two of which the paper itself calls negligibly coupled, while no corrected selected-config, split-resampled Pokec result exists. A three-network directional pattern is not evidence of prediction unless the rule is fixed and tested out of sample or under a completed resampling protocol.

### Does the paper pre-empt it?

Only superficially. It correctly states that attribute-blind structural debiasing cannot guarantee demographic parity and acknowledges weak coupling and DeltaEO failure (Section 7.4, pp. 23–24). It then asserts that the direction follows from the intervention rather than a fit (p. 23), without an independent test. It does not disclose that its rendered tables use obsolete configurations.

Editorial recommendation for RQ5: **major**. The qualitative caveat is appropriate, but the current empirical mechanism and significance package is incomplete and tied to the wrong operating points.

## RQ6 — measurement robustness

**Question.** “Are the structural-equality conclusions robust to alternative recommendation-acceptance assumptions?” (Introduction, RQ6, p. 2). The answer says PageRank ordering is preserved on all five networks, Degree equality only partially, and EG-TP is uninformative (Section 7.6 answer box, p. 27).

### Does the evidence on disk answer it?

No, not for the corrected selected system. The rendered Table 11 is stale. For deterministic Brightkite RA it reports full-acceptance Degree EG `-4.42 ± 0.00` (Table 11, p. 27), whereas the integrity-attested corrected aggregate reports `0.40138838901733037` with std `0.0` (`finalized/aggregate_metrics_summary.csv`). Verdict: **MISMATCH**.

Current acceptance CSVs exist for all datasets, but their schema is only `dataset,run,method,metric,value`; it records neither configuration nor `eg_graph_semantics`. The table runner takes every CAFE and Fusion setting from the old `PUBLISHED` map (`experiments/tables/_framework.py:67-79`; `experiments/tables/table11_acceptance.py:28-44`). Consequently the raw files cannot be tied to the corrected selected configurations or unambiguously aligned with the paper's fusion row. Provenance verdict: **CANNOT VERIFY FROM AVAILABLE FILES**.

The acceptance constructions are also not behavioral validation. The code computes rank precision and true-positive recommendations from the same test-pair set (`metrics/acceptance_eg.py:56-90,259-266`); PW assigns each recommendation event the aggregate precision of its rank (`metrics/acceptance_eg.py:65-79,131-177`), and TP retains only held-out true edges (`metrics/acceptance_eg.py:82-90`). Those are endogenous sensitivity transforms, not observations from an acceptance process.

Evidence verdict on the answer-box five-network ordering claim for the corrected selected system: **CANNOT VERIFY FROM AVAILABLE FILES**.

### Strongest hostile-reviewer objection

The paper calls this acceptance robustness without evaluating acceptance. PW is calibrated on the same held-out labels used for evaluation and has no user- or target-specific behavior; TP is an oracle filter. Neither model supports a claim about how deployed acceptance would change structural outcomes.

### Does the paper pre-empt it?

Substantially, but at the cost of conceding the RQ. Section 7.6 explicitly calls PW a descriptive reweighting, warns that levels cannot be compared, and says TP is uninformative (pp. 26–27). That is honest. It does not justify retaining the broader “measurement robustness” label, and it does not repair the stale/provenance-inadequate table.

Editorial recommendation for RQ6: **major**. Recast it as a limited sensitivity analysis, regenerate it at the canonical selected configurations with provenance, and remove behavioral-robustness language unless external acceptance evidence is added.

## Editorial synthesis

| RQ | Does current on-disk evidence answer the stated question? | Controlling evidence verdict | Editorial recommendation |
|---|---|---|---|
| RQ1 | No; only a narrower one-network degree-controlled diagnosis and partial corrected method coverage exist. | **CANNOT VERIFY FROM AVAILABLE FILES** | **major** |
| RQ2 | No; the corrected campaign covers only five methods, and the frontier/significance package is stale or incomplete. | **CANNOT VERIFY FROM AVAILABLE FILES** | **reject** |
| RQ3 | Yes descriptively for the implemented sum-reduced system, but not for the paper-defined active Eq. (8) intervention; the manuscript also gives the wrong source-side and mechanism conclusions. | **MISMATCH** | **major** |
| RQ4 | No; corrected component selections/ablations are missing and selected CAFE-In disables the penalty on four datasets. | **CANNOT VERIFY FROM AVAILABLE FILES**; current attribution is **MISMATCH** | **reject** |
| RQ5 | No; corrected selected-config demographic and paired evidence is incomplete. | **CANNOT VERIFY FROM AVAILABLE FILES** | **major** |
| RQ6 | No; the rendered table is stale and raw files lack the provenance needed to identify the tested system. | **CANNOT VERIFY FROM AVAILABLE FILES** | **major** |

The overall recommendation remains **reject**. RQ2 and RQ4 are central rather than optional appendices, and both currently fail at the level of experimental identity and causal attribution. “Some runs are still in flight” is not a defense for a paper whose existing tables make definitive all-dataset and Holm-corrected claims.

## What could not be checked, and why

1. The final seed sets or test results of the in-flight significance campaign: **CANNOT VERIFY FROM AVAILABLE FILES** because the files were actively changing; the exact snapshot sets are recorded above.
2. A corrected full RQ1 diagnostic across CN, AA, RA, LightGCN, and HL-GNN on every dataset: **CANNOT VERIFY FROM AVAILABLE FILES** because no complete provenance-attested artifact exists.
3. The `13`-of-`15` corrected Pareto count against the full Table 4 method set: **CANNOT VERIFY FROM AVAILABLE FILES** because no coherent corrected full table/figure source exists and Pokec-n baseline coverage is incomplete.
4. Holm-corrected claims at the corrected validation-selected configurations: **CANNOT VERIFY FROM AVAILABLE FILES** because current paired files are incomplete and encode the old configurations.
5. Corrected validation selection for every Table 7 ablation: **CANNOT VERIFY FROM AVAILABLE FILES** because the strict selection outputs cover only CAFE-In and CAFE-Pre.
6. A complete corrected selected-config demographic comparison including FairAdj, LBA, MORAL, and both CAFE variants on Pokec-n: **CANNOT VERIFY FROM AVAILABLE FILES** because CAFE-Pre and several baseline cells are partial and the resampled Pokec file is absent.
7. Exact configuration and EG-semantics provenance of the current acceptance CSVs: **CANNOT VERIFY FROM AVAILABLE FILES** because those fields are absent from the schema.
8. Robustness to real acceptance behavior: **CANNOT VERIFY FROM AVAILABLE FILES** because no logged, surveyed, or independently estimated acceptance data are present.
9. Whether “no measurable precision cost” is an equivalence claim: **CANNOT VERIFY FROM AVAILABLE FILES** because the paper reports null/non-significant difference tests, not a prespecified equivalence margin or TOST procedure.
10. Whether the ASF weights were fixed before results were inspected: **CANNOT VERIFY FROM AVAILABLE FILES** because no preregistration or immutable pre-result record establishes that chronology.
11. Results or selected configurations for the paper-defined edgewise Eq. (8) penalty: **CANNOT VERIFY FROM AVAILABLE FILES** because the configured MLP and one-dimensional coreness weights broadcast in every available nonzero-penalty CAFE-In run.

Phase 5 ends here.
