# Audit of draft 6 against the corrected TODO3 results

## Bottom line

The corrected results should be adopted. Leaving the draft unchanged is not a
scientifically defensible option because the old Degree Equality Gain (Degree
EG) values are known not to implement the simple, undirected,
deduplicated post-recommendation graph defined in the paper.

The new evidence makes **RQ3 substantially stronger and more complete**, but
the answer is less flattering and more precise:

- CAFE does **not** repair the source-user precision disparity. Relative to
  matched-\(\beta\) Fusion, the low/high source-precision ratio moves away
  from parity on all four random-split datasets. It moves toward parity only
  on Brightkite, where the precision gradient is reversed.
- Corrected source-stratified Degree EG is stronger: CAFE-In exceeds Fusion
  in every source-coreness bin in every rerun on all five datasets. The
  largest mean increment is in the lowest bin on four datasets and the
  second-lowest bin on Oklahoma.
- Direct target exposure is also stronger: CAFE-In moves recommendation
  event share from core to peripheral targets on all five datasets, with the
  periphery-up/core-down direction in every one of the three reruns.
- The mechanism must be narrowed. The selected CAFE-In configuration has
  \(\lambda_{\mathrm{fair}}=0\) on Ego-FB, Oklahoma, Pokec-n, and UNC.
  Beyond Fusion, the selected intervention on those four datasets is
  algorithmically the mobility boost, not a joint penalty-and-boost effect.
  Only Brightkite has an active in-processing penalty.

No further model experiment is needed to finish RQ3 as a descriptive,
fixed-split analysis. However, adopting the corrected configurations makes
several results outside RQ3 stale. The full draft is therefore **not yet
submission-ready**.

## Scope and evidence audited

I read the supplied 40-page PDF end to end:

- PDF: `ToRS_paper/CAFE_ToRS_EceErdem (6).pdf`
- PDF creation time: 25 July 2026, 17:42 CEST
- SHA-256:
  `08aa952e2a85df7bdd7a458d221ea7d18d11fa2ddeac35848de5ccdc5b711294`

I compared it with the completed corrected campaign:

- Campaign root:
  `results_egfix/20260724_degree_eg_sum_in_v1/todo3/`
- Completion marker:
  `results_egfix/20260724_degree_eg_sum_in_v1/todo3/_TODO3_COMPLETE`
- Integrity record:
  `results_egfix/20260724_degree_eg_sum_in_v1/todo3/finalized/todo3_integrity.json`
- Status: complete
- Aggregate jobs: 50 shards
- Source-stratified jobs: 15 shards
- Aggregate protocol: one fixed split, 10 training seeds per dataset
- Stratified and target-exposure protocol: one fixed split, 3 training seeds
  per dataset
- Graph semantics: `simple_undirected_set_union`
- Cross-path Degree-EG check: passed
- Manuscript files written by the campaign: none

The aggregate campaign covers only RA, HL-GNN, matched-\(\beta\) Fusion,
CAFE-In, and CAFE-Pre. It does not replace the complete baseline, ablation,
demographic, paired-significance, or acceptance experiments in the PDF.

### Important source synchronization problem

The PDF and the local TeX tree are not the same revision:

- The PDF has a complete abstract, whereas `ToRS_paper/main.tex` has an empty
  abstract.
- The PDF omits the Discussion, Limitations, and Conclusion sections even
  though corresponding local section files exist.
- The PDF's RQ3 prose is not identical to the current
  `ToRS_paper/sections/07-results.tex`.
- The PDF does not display the full Degree-EG hold paragraph in the local
  source, but its RQ3 answer still says that Degree EG is deferred.

Before editing the manuscript, establish which source is canonical. If the
PDF was built from an Overleaf or external tree, synchronize that tree into
the repository first. Otherwise, a local edit may not reproduce the document
being reviewed here.

## Does each research question become stronger or weaker?

| RQ | Effect of the corrected results | Judgment |
|---|---|---|
| RQ1 | Precision evidence is unaffected, but the five-method stratified Degree-EG diagnosis was not fully rerun. | Partly unresolved |
| RQ2 | CAFE's corrected Degree-EG point estimates are stronger, but precision against RA weakens on Ego-FB and Oklahoma; full baselines and paired tests are stale. | Mixed |
| RQ3 | The validity hold is resolved; source Degree EG and target exposure now cover all five datasets. The old source-precision conclusion reverses. | Substantially stronger |
| RQ4 | Evidence for Fusion plus mobility becomes clearer, but evidence for an active in-processing penalty and universal three-stage complementarity weakens. | Mixed to weaker |
| RQ5 | The all-five structural redistribution premise is stronger, but no new demographic outcome was computed. | Not reassessable yet |
| RQ6 | The acceptance analysis was not recomputed for the corrected graph semantics and selected configurations. | Currently unsupported |

The main distinction is between **better scientific evidence** and **a more
favorable result**. RQ3 is stronger scientifically precisely because it now
shows a consistent structural mechanism while rejecting the unsupported
claim that source-side accuracy becomes more equal.

---

# 1. Required revision of RQ3, “Where the Intervention Acts”

## 1.1 What remains valid

The following qualitative observations survive:

- The comparison set is RA, HL-GNN, Fusion, CAFE-In, and CAFE-Pre.
- On all four random-split datasets, every method has higher precision in the
  highest source-coreness bin than in the lowest bin.
- On temporal Brightkite, every method reverses that endpoint ordering.
- Brightkite's lowest source bin still contains 3,571 of 6,580 evaluated
  users, or 54.3%.
- The source/target distinction is essential: source strata identify whose
  lists are being evaluated; target strata identify who receives incoming
  exposure.

## 1.2 Source precision: the present conclusion must be reversed

The PDF currently says that CAFE-In narrows the source-side precision gap on
four datasets and widens it on UNC. Under the corrected selected
configurations, the opposite pattern holds.

| Dataset | Fusion low/high ratio | CAFE-In low/high ratio | Movement |
|---|---:|---:|---|
| Brightkite | 17.5125 | 15.9146 | Toward parity; its gradient is reversed |
| Ego-FB | 0.2033 | 0.1971 | Away from parity |
| Oklahoma | 0.1463 | 0.0724 | Away from parity, substantially |
| Pokec-n | 0.1648 | 0.1621 | Away from parity, slightly |
| UNC | 0.1912 | 0.1846 | Away from parity |

The corrected CAFE-In-minus-Fusion change in lowest-bin Precision@10 is:

- Brightkite: \(+0.01061\)
- Ego-FB: \(-0.00288\)
- Oklahoma: \(-0.02515\)
- Pokec-n: \(-0.00034\)
- UNC: \(-0.00158\)

Oklahoma is not a rounding-level exception. CAFE-In approximately halves
the low/high ratio there. This should be reported directly, not hidden behind
“approximately parallel profiles.”

The defensible conclusion is:

> CAFE does not flatten source-side accuracy stratification. Under the
> corrected selected configurations, it preserves or widens that disparity
> on the four random-split networks and moves toward parity only on
> Brightkite, whose gradient runs in the opposite direction.

This result does not invalidate CAFE's structural-equality contribution.
Instead, it separates recommendation quality for source users from exposure
redistribution toward target nodes.

Authoritative source:

`results_egfix/20260724_degree_eg_sum_in_v1/todo3/finalized/stratified_precision_summary.csv`

Per-run source:

`results_egfix/20260724_degree_eg_sum_in_v1/todo3/finalized/stratified_precision_by_run.csv`

## 1.3 Source-stratified Degree EG: the strongest new RQ3 result

The Degree-EG hold is resolved. For CAFE-In relative to matched-\(\beta\)
Fusion:

- All 132 bin-by-run differences are positive.
- The smallest bin-by-run difference is \(+0.0125\) Degree-EG points.
- The largest is \(+2.155\) points.
- The largest mean difference occurs in the lowest source-coreness bin on
  Brightkite, Ego-FB, Pokec-n, and UNC.
- On Oklahoma, the second-lowest bin has the largest mean difference
  (\(+1.957\)), narrowly above the lowest bin (\(+1.919\)).

The lowest-bin mean differences are:

- Brightkite: \(+0.817\)
- Ego-FB: \(+0.960\)
- Oklahoma: \(+1.919\)
- Pokec-n: \(+1.148\)
- UNC: \(+2.054\)

This supports “peripheral-weighted” or “largest at the lowest/next-lowest
coreness” wording. It does **not** support treating the bin values as an
additive decomposition of aggregate Degree EG. Each bin is a separate
counterfactual in which only recommendations issued by source users in that
bin materialize.

Use one caveat, stated once:

> Stratified EG values are non-additive source-bin counterfactuals; they
> localize the direction of the intervention but do not partition the
> aggregate EG value.

Authoritative sources:

- `results_egfix/20260724_degree_eg_sum_in_v1/todo3/finalized/stratified_degree_eg_summary.csv`
- `results_egfix/20260724_degree_eg_sum_in_v1/todo3/finalized/stratified_degree_eg_per_run.csv`

## 1.4 Target exposure: extend the claim to all five datasets

The direct audit now covers all five datasets with three training-seed
reruns each. CAFE-In shifts recommendation event share from core to
peripheral target tiers in every rerun.

| Dataset | Peripheral event-share change | Core event-share change | Relative mean exposure, periphery/core |
|---|---:|---:|---:|
| Brightkite | \(+5.35\) pp | \(-5.82\) pp | \(+22.43\%/-8.46\%\) |
| Ego-FB | \(+4.99\) pp | \(-4.15\) pp | \(+42.24\%/-8.44\%\) |
| Oklahoma | \(+15.60\) pp | \(-10.78\) pp | \(+206.05\%/-17.79\%\) |
| Pokec-n | \(+4.75\) pp | \(-4.84\) pp | \(+72.91\%/-6.85\%\) |
| UNC | \(+6.42\) pp | \(-5.23\) pp | \(+55.74\%/-8.71\%\) |

Use event-share changes in the main text. They are mass-conserving and easier
to interpret. Oklahoma's \(+206\%\) relative per-node increase is correct,
but the percentage is large because Fusion gives the peripheral tier a small
baseline exposure. It is better retained in a table or appendix than used as
the headline number.

Target exposure and Degree EG intentionally use different semantics:

- Target exposure counts directed recommendation events. Reciprocal events
  are distinct because both users receive exposure.
- Degree EG materializes a simple undirected graph and therefore
  canonicalizes and deduplicates reciprocal recommendations.

These are not inconsistent measurements; they answer different questions.

Authoritative sources:

- `results_egfix/20260724_degree_eg_sum_in_v1/todo3/finalized/target_exposure_cafe_in_vs_fusion_summary.csv`
- `results_egfix/20260724_degree_eg_sum_in_v1/todo3/finalized/target_exposure_cafe_in_vs_fusion_by_run.csv`
- General five-method profiles:
  `results_egfix/20260724_degree_eg_sum_in_v1/todo3/finalized/target_exposure_summary.csv`

## 1.5 Correct the mechanism attribution

The selected configurations are:

| Dataset | CAFE-In \((\lambda_{\mathrm{fair}},\beta,\lambda_{\mathrm{mob}})\) | CAFE-Pre \((\alpha,\beta,\lambda_{\mathrm{mob}})\) |
|---|---:|---:|
| Brightkite | \((1,5,1)\) | \((2,5,1)\) |
| Ego-FB | \((0,0.1,0.5)\) | \((1,0.1,1)\) |
| Oklahoma | \((0,0.5,1)\) | \((1,0.5,1)\) |
| Pokec-n | \((0,5,1)\) | \((0.01,5,1)\) |
| UNC | \((0,5,1)\) | \((1,2,1)\) |

Therefore:

- Do not write “the penalty and boost move exposure” as a five-dataset
  conclusion.
- On four datasets, the selected CAFE-In pipeline differs algorithmically
  from matched Fusion through the mobility boost; the in-processing penalty
  weight is zero.
- Brightkite is the only dataset where the selected CAFE-In contrast
  contains an active penalty and boost.
- Fusion and CAFE-In were separately trained, even when
  \(\lambda_{\mathrm{fair}}=0\). The comparison is matched in \(\beta\), but
  it is not a shared-checkpoint causal ablation.

Authoritative configuration and lineage files:

- `results_egfix/20260724_degree_eg_sum_in_v1/todo3/selected_configs.csv`
- `results_egfix/20260724_degree_eg_sum_in_v1/todo3/selected_configs_lineage.json`

## 1.6 Recommended compact structure for Section 7.2

The section can be shorter and clearer in this order:

1. **Scope in two sentences.** A recommendation has a source user whose list
   has measurable precision and a target node that receives exposure.
   Source-coreness and target-coreness analyses therefore answer different
   questions.
2. **Protocol in one compact paragraph.** Five methods, one fixed split,
   \(n=3\) training seeds on every dataset, no per-bin tests, tied coreness
   values, and non-additive source-bin EG.
3. **Source precision persists.** State the common endpoint ordering, the
   Brightkite reversal, and that the ratio moves away from parity on the
   four random-split networks.
4. **Source Degree EG is peripheral-weighted.** Report positivity in every
   bin and rerun, with the maximum in the lowest or next-lowest bin.
5. **Target exposure moves from core to periphery.** Report all-five
   event-share directions and the every-run replication.
6. **Answer box.** Synthesize instead of repeating the body.

The current answer box appears on PDF page 20 after the first portion of the
RQ4 ablation table. Reflow the float so that the RQ3 answer appears before
Table 6 and before RQ4 evidence.

### Suggested RQ3 answer

> **Answer to RQ3.** CAFE-In acts on structural exposure rather than by
> repairing source-side accuracy. Relative to matched-\(\beta\) Fusion, the
> low/high source-precision ratio moves away from parity on all four
> random-split datasets and toward parity only on Brightkite, whose gradient
> is reversed. In contrast, corrected Degree EG increases in every source
> bin and rerun, with the largest mean increment in the lowest or
> next-lowest bin. Directly, CAFE-In shifts recommendation-event share from
> core to peripheral targets on all five networks in every rerun. Under the
> selected configurations, the beyond-Fusion intervention is the mobility
> boost on four networks and penalty plus boost on Brightkite. These
> fixed-split, \(n=3\) localization results are descriptive; no per-bin
> hypothesis tests were run.

## 1.7 RQ3 table and figure replacements

I did not generate or overwrite any manuscript table or figure.

### Figure 4

The current Figure 4 uses pre-fix plots and must be replaced. Use these
manifest-attested corrected PDFs:

- Oklahoma:
  `results_egfix/20260724_degree_eg_sum_in_v1/todo3/finalized/figures/stacked_barchart_oklahoma.pdf`
- Brightkite:
  `results_egfix/20260724_degree_eg_sum_in_v1/todo3/finalized/figures/stacked_barchart_brightkite.pdf`

The three panels promised for the appendix also exist:

- `results_egfix/20260724_degree_eg_sum_in_v1/todo3/finalized/figures/stacked_barchart_ego-facebook.pdf`
- `results_egfix/20260724_degree_eg_sum_in_v1/todo3/finalized/figures/stacked_barchart_pokec_n.pdf`
- `results_egfix/20260724_degree_eg_sum_in_v1/todo3/finalized/figures/stacked_barchart_unc.pdf`

The PDF currently promises those appendix panels, but Appendix C.1 contains
only Table 13. Either insert the three corrected panels or remove the
promise.

Before publication, improve two visual details in the corrected plots:

- Label the lower axis explicitly as “Degree EG@10 (%)”, not only “Equality
  Gain (%)”.
- Give RA and CAFE-Pre more distinguishable colors; the two purple hues are
  too similar at journal scale.

The PNG files under `/tmp/cafe-todo4-figures.dGPz8w/` are only visual
previews. Do not use them as manuscript assets.

### Table 5

The current Table 5 is stale and must be rebuilt. There is no finalized TeX
table; use the authoritative CSVs:

- Source precision:
  `results_egfix/20260724_degree_eg_sum_in_v1/todo3/finalized/stratified_precision_summary.csv`
- Source Degree EG:
  `results_egfix/20260724_degree_eg_sum_in_v1/todo3/finalized/stratified_degree_eg_summary.csv`
- Target contrast:
  `results_egfix/20260724_degree_eg_sum_in_v1/todo3/finalized/target_exposure_cafe_in_vs_fusion_summary.csv`

A compact table can report, per dataset:

- lowest-bin \(\Delta\)Precision;
- Fusion \(\rightarrow\) CAFE-In low/high ratio;
- lowest-bin \(\Delta\)Degree EG, with a note that Oklahoma's maximum is in
  the second-lowest bin;
- peripheral and core target-event-share changes.

Prefer one compact five-row table over separate source and target panels if
page length is a concern.

### Appendix Table 13

The realized bin counts, coreness ranges, evaluated-user totals, and
smallest/largest bin sizes remain correct. Only its protocol caption is
stale: Pokec-n now has \(n=3\), as do all five datasets.

### Appendix Table 17

The old group-resolved target audit is not replaced by TODO3. The new target
audit is pooled and contains no group field. After updating Table 5, the
claim that pooling Table 17 “recovers Table 5(b)” becomes false. Retain
Table 17 only as a separate, explicitly old-configuration demographic audit
or rerun it for the selected configurations.

---

# 2. RQ-by-RQ impact outside Section 7.2

## 2.1 RQ1: diagnosis is only partly repaired

The source-side Precision@10 diagnosis does not use the invalid Degree-EG
path and remains usable. The corrected TODO3 profiles also support the
shared endpoint ordering for RA and HL-GNN.

However, Figure 1's Degree-EG panel compares CN, AA, RA, LightGCN, and
HL-GNN. TODO3 did not rerun CN, AA, or LightGCN in that analysis.
Consequently, the full claim that the Degree-EG gradient is consistent
across the five diagnostic methods is not yet repaired.

Required change:

- Correct the page-9 wording “recommendations targeting low-coreness users.”
  The bins are defined by **source users**, so write “recommendations issued
  by low-coreness source users” or “lists served to low-coreness users.”
- Rerun/recompute the complete Figure 1 Degree-EG panel before retaining the
  five-method equality claim.

No complete corrected Figure 1 is saved. RA and HL-GNN are available in the
TODO3 stratified artifacts; CN and AA can be regenerated without training,
while LightGCN requires a new learned-model run.

**Strength judgment:** the precision part remains strong; the Degree-EG
part of the current RQ1 answer is unresolved rather than strengthened.

## 2.2 RQ2: stronger Degree equality, mixed utility, incomplete inference

Within the five rerun methods, the corrected fixed-split results show:

- CAFE-In and CAFE-Pre exceed HL-GNN in Precision, Degree EG, and PageRank
  EG on all five datasets.
- Both exceed HL-GNN in Closeness EG on four of five datasets; Ego-FB remains
  the exception.
- CAFE-In exceeds RA on all 15 fixed-split EG comparisons.
- CAFE-In exceeds RA in precision on Brightkite, Pokec-n, and UNC, but loses
  to RA on Ego-FB and Oklahoma.

The RA precision comparison changes materially:

- Brightkite: CAFE-In is \(+22.8\%\) above RA, replacing the old reported
  loss.
- Ego-FB: CAFE-In is \(6.1\%\) below RA, replacing the old near-tie.
- Oklahoma: CAFE-In is \(10.2\%\) below RA, replacing the old win.
- Pokec-n: CAFE-In is \(8.5\%\) above RA.
- UNC: CAFE-In is \(2.2\%\) above RA.

The structural point estimates are stronger. Both CAFE variants now exceed
both RA and HL-GNN in corrected Degree EG on all five datasets. The old
Ego-FB Degree regression and the old Pokec-n-versus-RA Degree exception
disappear at the fixed-split point-estimate level.

But TODO3 does not support:

- the present 15-panel Pareto-membership counts;
- comparison against Random, PA, AA, CN, LightGCN, SSNet, Cross-Link Bias,
  LTLP, or TailSTEAK under the corrected semantics;
- the old split-resampled paired \(p\)-values;
- old statements such as “significantly worse in exactly two contrasts.”

Table 4 and Figure 3 cannot be partially patched and then described as a
coherent full comparison. The five corrected rows are available at:

`results_egfix/20260724_degree_eg_sum_in_v1/todo3/finalized/aggregate_metrics_summary.csv`

Per-run values are at:

`results_egfix/20260724_degree_eg_sum_in_v1/todo3/finalized/aggregate_metrics_long.csv`

There is no complete corrected Table 4 or Figure 3 artifact. Uncovered
learned baselines have no saved recommendation lists or checkpoints and
require new jobs. Deterministic heuristics can be regenerated without
training.

**Strength judgment:** descriptively stronger for structural equality,
weaker against RA on utility in two datasets, and inferentially incomplete.

## 2.3 RQ4: revise the component story

The corrected five-method results still support that Fusion establishes a
higher-precision operating point than HL-GNN on all five datasets. They also
show that CAFE-In exceeds Fusion in Degree EG in all ten fixed-split runs on
every dataset.

However, several current RQ4 statements are stale:

- Fusion now raises Degree EG above HL-GNN on Brightkite, Pokec-n, and UNC,
  not on four datasets.
- The old “Fusion supplies 40–68% of the Degree-EG gain on four networks”
  statement no longer holds. The corresponding corrected shares are about
  62% on Brightkite, 66% on Pokec-n, 23% on UNC, and negative on Ego-FB and
  Oklahoma because Fusion is below the backbone there.
- The old “no precision cost on Oklahoma and UNC” statement fails on
  Oklahoma. CAFE-In minus Fusion is \(-0.00474\), or \(-2.60\%\), and is
  negative in all ten fixed-split runs. UNC remains near zero
  (\(-0.00022\)) but has no new equivalence test.
- “The matched contrast isolates the joint penalty and boost” is false on
  four datasets because the selected penalty weight is zero.

Corrected CAFE-In-minus-Fusion means are:

| Dataset | \(\Delta\)Precision | \(\Delta\)Degree EG | \(\Delta\)PageRank EG | \(\Delta\)Closeness EG |
|---|---:|---:|---:|---:|
| Brightkite | \(+0.00628\) | \(+1.069\) | \(+0.961\) | \(+0.013\) |
| Ego-FB | \(-0.00325\) | \(+1.734\) | \(+6.315\) | \(-9.841\) |
| Oklahoma | \(-0.00474\) | \(+3.449\) | \(+4.853\) | \(+14.494\) |
| Pokec-n | \(-0.00053\) | \(+2.662\) | \(+3.610\) | \(+2.696\) |
| UNC | \(-0.00022\) | \(+3.598\) | \(+5.629\) | \(+4.404\) |

The defensible revised story is:

> Fusion establishes utility. Under the selected CAFE-In configurations,
> the mobility boost supplies the additional structural equalization on
> four datasets; an active in-processing penalty is selected only on
> Brightkite. CAFE-Pre remains an active preprocessing route on all five
> datasets.

This is cleaner attribution, but weaker evidence for a universally
complementary three-stage architecture.

The full ablation Table 6 cannot be regenerated from TODO3: single-component
and partial-pipeline recommendations/checkpoints were not saved. New
ablation jobs are required if the full table and its best-operating-point
claims are retained. The visible penalty-reduction TODO on PDF page 22 must
also be removed or replaced with the completed selection result.

## 2.4 RQ5: structural precursor strengthened; demographic answer stale

TODO3 does not compute \(\Delta SP\), \(\Delta EO\), group uplift, or
attribute-dependent baseline results. The new all-five target audit
strengthens the structural premise that CAFE moves exposure toward
peripheral targets, but it does not establish:

- group-specific redistribution under the selected configurations;
- which demographic group benefits;
- the direction of \(\Delta SP\) or \(\Delta EO\);
- significance against FairAdj, LBA, or MORAL.

Tables 7, 8, 16, and 17 use old selected configurations and/or old EG
semantics. The claim that all 162 structural/utility contrasts significantly
favor CAFE cannot be carried forward without recomputation.

The saved TODO3 recommendation NPZs are sufficient to recompute a new
five-method, \(n=3\), fixed-split demographic analysis without model
training. They are located under:

`results_egfix/20260724_degree_eg_sum_in_v1/todo3/stratified/<dataset>/seed_{42,43,44}/recommendations_*.npz`

Preserving the current \(n=10\) protocol and the attribute-dependent
baseline comparison requires new model jobs because no corresponding saved
rankings or checkpoints were found.

**Strength judgment:** the structural mechanism is stronger, but the
demographic RQ cannot yet be called stronger or weaker under the new
configurations.

## 2.5 RQ6: recompute before making a robustness claim

The current acceptance table is not compatible with the corrected primary
results. Its “Full” values do not reproduce the authoritative aggregate
values. For CAFE-In Degree EG:

| Dataset | PDF Table 9, Full | Corrected aggregate |
|---|---:|---:|
| Brightkite | \(-3.69\) | \(+2.80\) |
| Ego-FB | \(11.61\) | \(14.26\) |
| Oklahoma | \(5.66\) | \(12.69\) |
| Pokec-n | \(3.91\) | \(9.03\) |
| UNC | \(0.25\) | \(11.05\) |

There is also an internal inconsistency in the current PDF independent of
TODO3: deterministic RA differs between Table 4 and Table 9 even though
both tables claim a corresponding fixed-split full-acceptance setting. For
example, Brightkite Degree/PageRank EG is \(-4.42/0.27\) in Table 4 but
\(-8.69/-1.43\) in Table 9. Stochastic variation cannot explain a
deterministic RA discrepancy.

The qualitative argument that oracle true-positive acceptance confounds
accuracy and structural impact may remain conceptually useful. The empirical
PW/TP ordering claims, however, must be rerun for the canonical graph
construction and selected configurations.

The saved \(n=3\) recommendation NPZs permit an offline, no-training
acceptance recomputation for the five TODO3 methods. Preserving the current
\(n=10\) protocol on Brightkite, Ego-FB, Oklahoma, and UNC requires new
model jobs because aggregate jobs saved metrics but not ranked
recommendations.

**Strength judgment:** currently unsupported, not contradicted.

---

# 3. Table and figure propagation map

| PDF item | Required action | Existing corrected source |
|---|---|---|
| Figure 1, RQ1 | Rerun complete Degree-EG panel | Partial RA/HL evidence only in TODO3; no full replacement |
| Table 4, RQ2 | Rebuild all methods under one semantics/config protocol | Five-method subset in `aggregate_metrics_summary.csv`; no full replacement |
| Figure 3, RQ2 | Regenerate after complete Table 4 | No complete corrected figure |
| Figure 4, RQ3 | Replace | Five corrected PDFs in `todo3/finalized/figures/` |
| Table 5, RQ3 | Rebuild | Finalized precision, Degree-EG, and target-exposure CSVs |
| Table 6, RQ4 | Rerun ablations or narrow RQ4 | No full corrected replacement |
| Tables 7–8, RQ5 | Recompute selected configurations and baselines | \(n=3\), five-method offline recomputation possible; no final table saved |
| Table 9, RQ6 | Recompute Full/PW/TP | \(n=3\) offline recomputation possible; no corrected table saved |
| Table 13, appendix | Keep bin data; change protocol caption to \(n=3\) for all five | Verified against finalized stratified CSV |
| Figure 5, closeness | Stop calling its parameters “selected,” or rerun | Existing probe uses obsolete operating points |
| Table 14, paired tests | Rerun split-resampled families for new configurations | TODO3 fixed-split data cannot replace this protocol |
| Table 16, demographic protocols | Recompute selected configurations | No corrected \(n=10\) artifact |
| Table 17, group exposure | Separate from new Table 5 or rerun | No group fields in TODO3 target audit |
| Table 18, acceptance weights | Recompute with new scorer rankings | Possible at \(n=3\) from saved NPZs |

Figure 3 has an additional presentational inconsistency: its caption says it
contains methods from Table 4, but its legend includes ablation methods that
are not in Table 4. Correct this when regenerating it.

---

# 4. Minimum work paths

## Path A: finish RQ3 now

No additional model jobs are required.

1. Replace Figure 4 with the corrected Oklahoma and Brightkite PDFs.
2. Add the Ego-FB, Pokec-n, and UNC corrected panels to Appendix C.1.
3. Rebuild Table 5 from the finalized CSVs.
4. Replace the source-precision narrowing claim.
5. Add the corrected source-stratified Degree-EG result.
6. Extend the target audit to all five datasets.
7. Attribute the selected four-dataset contrast to mobility, not to an
   active in-processing penalty.
8. Change the RQ3 answer box and move it before the RQ4 float.
9. Update Appendix Table 13's protocol caption.

This gives a complete, honest RQ3 section. It is suitable for continuing the
paper section by section, but not for final submission while the other tables
remain stale.

## Path B: preserve the full journal-paper scope

New jobs are required for:

1. the uncovered learned methods in the full RQ1/RQ2 comparison;
2. split-resampled paired significance for the selected configurations;
3. the complete RQ4 ablation table;
4. the current \(n=10\) demographic comparison, including
   attribute-dependent baselines;
5. the current \(n=10\) acceptance analysis;
6. the selected-configuration closeness mechanism if it is still described
   as a deployed-operating-point probe.

Deterministic heuristic results can be regenerated without training.

## Path C: avoid most new jobs by narrowing the paper

It is technically possible to restrict RQ2–RQ6 to the five TODO3 methods and
use the saved \(n=3\) ranked recommendations for demographic and acceptance
analyses. This would materially reduce baseline breadth, inferential
strength, and protocol consistency. For a journal revision already claiming
extensive baselines and paired significance, I do not recommend this route
unless compute or schedule makes the full reruns impossible.

---

# 5. Method and protocol statements that must change

## 5.1 Penalty definition

Equation 8 in the PDF defines a mean penalty,
\( |E_{\mathrm{batch}}^+|^{-1}\sum\), but the corrected CAFE-In campaign
uses `penalty_reduction=sum`. The method definition and implementation
description must agree. Do not retain the mean equation while reporting the
sum-reduction selection.

The full corrected CAFE-In selection grid is saved at:

`results_egfix/20260724_degree_eg_sum_in_v1/cafe_in_selection_candidates.csv`

The CAFE-Pre grid is saved at:

`results_egfix/20260723_degree_eg_v1/selection_candidates.csv`

The appendix can regenerate parameter sensitivity and ASF summaries from
these files without new training.

## 5.2 Selection claims

The PDF says that Pokec-n and Brightkite were excluded from the full
selection and that the largest selected mobility value is 0.5. Both
statements are stale:

- The corrected artifacts contain 210 configurations per
  dataset/variant for all five datasets.
- \(\lambda_{\mathrm{mob}}=1\) is selected in nine of the ten
  dataset/variant configurations.

Update the selection and sensitivity prose from the saved grids. Do not
retain old “plateau” magnitudes without recomputing them.

## 5.3 CAFE-In identity

Table 3 depicts CAFE-In as containing an active in-processing penalty. The
architecture permits the penalty, but the selected weight is zero on four
datasets. Qualify the table or its caption so that an architectural checkmark
is not read as an empirically active component everywhere.

## 5.4 Closeness probe

Figure 5 and the closeness-mechanism prose call old parameter values the
“selected operating point.” The corrected selected values are:

- Ego-FB: \(\beta=0.1,\lambda_{\mathrm{mob}}=0.5\);
- Oklahoma: \(\beta=0.5,\lambda_{\mathrm{mob}}=1\);
- UNC: \(\beta=5,\lambda_{\mathrm{mob}}=1\).

The existing probes use different values on Ego-FB and Oklahoma/UNC. They
may remain as sensitivity or mechanism probes, but not as exact deployed
configuration replications. A new \(n=3\) selected-config analysis can be
computed offline from the saved recommendation NPZs; preserving the
current \(n=10\) dose-curve claims requires new jobs.

---

# 6. Other high-priority problems found in the PDF

These issues are independent of whether TODO3 is adopted.

## 6.1 Claim inconsistencies

- The abstract says higher coreness gives higher precision across five
  networks, but Brightkite reverses the gradient for every method.
- The abstract says “recommendations to nodes with higher coreness,” which
  sounds target-indexed; the precision analysis is indexed by source user.
- The abstract implies improvement on all three EG dimensions, but Ego-FB
  Closeness EG is the stated exception.
- The abstract says demographic fairness improves, while RQ5 reports that
  \(\Delta SP\) worsens on UNC and makes no general \(\Delta EO\) claim.
- The RQ1 wording “targeting low-coreness users” conflicts with the
  source-bin definition in Section 3.3.

These should be corrected when the abstract and introduction are revised.

## 6.2 Mathematical error

The page-6 statement that binary-relevance NDCG is a monotone transform of
Precision@\(K\) is false. Two lists can have the same number of relevant
items and hence the same Precision@\(K\), but different NDCG because the
relevant items occur at different ranks. Remove or replace this statement.

## 6.3 Dataset population inconsistency

Table 2 mixes raw and processed populations:

- Oklahoma reports 3,111 nodes, but its stated edge count and mean degree
  correspond to 3,076 nodes.
- UNC reports 4,018 nodes, but its stated mean degree corresponds to 3,985
  nodes.

Appendix A and TODO3 use 3,076 and 3,985. Label raw versus processed graphs
explicitly or use one consistent evaluation population.

## 6.4 Cross-references and missing sections

- The PDF contains 38 visible `??` markers.
- Page 3 promises Discussion, Limitations, and Conclusion sections, but they
  are absent from the supplied PDF.
- Page 7 calls the acceptance analysis RQ8; it is RQ6.
- Figure 4 promises appendix panels that are absent.

Resolve all references only after synchronizing the canonical source tree.

## 6.5 Production defects

- Front matter identifies RecSys '26 while the footers identify J. ACM.
- The percentile formula on page 11 runs beyond the margin.
- The CAFE row in Table 15 is clipped.
- The page-17 footer overlaps the RQ2 answer box.
- Visible text errors include “CAFE CAFE,” “inference..,” and “EG-TP..”.

These are straightforward production fixes, but together they make the PDF
look unfinished to a reviewer.

---

# 7. Artifact policy

Use only the integrity-attested outputs under:

`results_egfix/20260724_degree_eg_sum_in_v1/todo3/finalized/`

The principal files are:

- `aggregate_metrics_summary.csv`
- `aggregate_metrics_long.csv`
- `stratified_precision_summary.csv`
- `stratified_precision_by_run.csv`
- `stratified_degree_eg_summary.csv`
- `stratified_degree_eg_per_run.csv`
- `target_exposure_cafe_in_vs_fusion_summary.csv`
- `target_exposure_cafe_in_vs_fusion_by_run.csv`
- `target_exposure_summary.csv`
- `rerun_integrity.json`
- `target_exposure_integrity.json`
- `todo3_integrity.json`
- `figures/stacked_barchart_*.pdf`

Files under `/tmp` are scratch renders, preflight reviews, comparison CSVs,
or earlier pilots. In particular, do not cite or copy:

- `/tmp/cafe-todo4-figures.dGPz8w/*.png`;
- `/tmp/todo3_oklahoma.png`;
- `/tmp/penalty_reduction_review*`;
- `/tmp/todo3-review*`;
- `/tmp/cafe-egfix-*`;
- older `/tmp` RQ2 or stratified audit products.

They can be useful for visual inspection, but the finalized manifest does
not attest them as publication artifacts.

---

# Final recommendation

Update RQ3 now. It is complete, coherent, and materially stronger under the
corrected semantics. Present the result as a separation:

- source-user precision inequality persists and usually widens;
- source-stratified Degree equalization is strongest at low coreness;
- incoming target exposure shifts from core to periphery on every dataset.

At the paper level, do not submit a hybrid in which RQ3 uses corrected
configurations while RQ2, RQ4, RQ5, and RQ6 attach old tables and
significance statements to the same method names. Either run the propagation
experiments needed for the full journal scope or explicitly narrow those
claims and protocols. The corrected evidence supports a strong paper, but
the strongest defensible mechanism is now **Fusion plus peripheral-target
mobility**, with CAFE-Pre as the consistently active training-time
alternative—not a universally active CAFE-In penalty.
