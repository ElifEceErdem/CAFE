# Phase 3 — did the story change?

## Controlling evidence and frozen read

The authoritative manuscript for this phase is `ToRS_paper/CAFE_ToRS_EceErdem (2).pdf` (38 pages; SHA-256 `45ee85116e7599651acab89862cb4c62439a37dd3e7cd261503b9b300a82b4f8`). The exact supplied previous-submission path, `RecSys_2026_paper_486.pdf`, is absent. A similarly named `RecSys_2026_paper_486 (3).pdf` exists, but I did not silently substitute it. Consequently, “old” below means either a value printed in the current draft or a value in an unmarked pre-fix result artifact; it does **not** mean a verified value from the exact previous-submission PDF.

Verdict on comparisons requiring the exact previous-submission PDF: **CANNOT VERIFY FROM AVAILABLE FILES**.

I froze the live evidence boundary at `2026-08-04T10:10:30+02:00`. At that instant the corrected split-resampled CSVs contained:

| Dataset | File state at frozen read | Exact seeds present | SHA-256 at frozen read |
|---|---|---|---|
| Ego-FB | 125 data rows | `{0,1,2,3,4}` | `a80f20110ebf8d70124fa336898ad8970b7f2b39719f525ba64976a46af511c5` |
| Oklahoma | 336 data rows | `{0,1,2,3,4,5,6}` | `d36894697eafa33bf766f68c466c86eb07537391c5098205e0c95e1b25efef14` |
| UNC | 240 data rows | `{0,1,2,3,4}` | `4e7dbc07689db6292c55f257f49128bb56adcdd4f15678c9fdeee0270b5274d8` |
| Brightkite | 50 data rows | `{0,1}` | `27002581e9cd916391cafa9c3a3e8e2b72da5b915a5f3ee6dd1fd8234a91a777` |
| Pokec-n | consolidated CSV absent | none | not applicable |

Those files were actively advancing. All significance conclusions below use only this frozen coverage. The required ten-seed corrected Holm analysis does not exist on disk.

### Status and numeric rules

Every story status is exactly one of **STILL SUPPORTED**, **WEAKENED**, **NO LONGER SUPPORTED**, or **NOT YET TESTABLE**. A separate evidence verdict is exactly one of **MATCH**, **MISMATCH**, or **CANNOT VERIFY FROM AVAILABLE FILES**.

No artifact stores an old-result-minus-new-result field. Therefore every requested *old-to-new numeric delta* is **CANNOT VERIFY FROM AVAILABLE FILES**. I quote paired old and corrected means and their stored across-repetition standard deviations, but I do not subtract them. This is deliberate: manufacturing deltas would violate the evidence rule. Method-order and sign changes are reported directly from the quoted values.

Within-campaign corrected method contrasts are different: `ToRS_paper/investigation/should_update.md` stores CAFE-In-minus-Fusion contrasts already derived from the completed TODO3 CSVs. I use those quoted contrasts only after matching the underlying rows in `todo3/finalized/{aggregate_metrics_summary,stratified_precision_summary,stratified_degree_eg_summary,target_exposure_cafe_in_vs_fusion_summary}.csv`. I do not import that audit's paper locations or judgments, which concern an older draft. The `absolute_delta_mean` and `event_share_delta_mean` fields in the target-exposure CSV are also primary stored contrasts.

## Evidence ledger

### E1. Headline fixed-configuration results: old versus corrected

Each cell is `mean ± stored across-repetition SD`. Old values are from the unmarked July JSON; corrected values are from the JSON carrying `eg_graph_semantics = simple_undirected_set_union`. These are the draft's old configurations, not the corrected validation-selected configurations discussed in E4.

| Dataset / metric | Old HL-GNN | Old CAFE-In | Corrected HL-GNN | Corrected CAFE-In |
|---|---:|---:|---:|---:|
| Ego-FB Precision@10 | `0.193942156554546 ± 0.0018612013201338457` | `0.23716127004087156 ± 0.0007423911045962698` | `0.19477208425023923 ± 0.0031048229069213706` | `0.23716127004087156 ± 0.0007423911045962698` |
| Ego-FB Degree EG@10 | `13.048074601006665 ± 0.26942711049551865` | `11.756316204535503 ± 0.14770223063833893` | `13.243187832043793 ± 0.17772863268757444` | `12.695805944783979 ± 0.056521124514035403` |
| Ego-FB PageRank EG@10 | `10.523041987128007 ± 0.28679298468960823` | `11.65438011675256 ± 0.23510536784351269` | `10.595583717284262 ± 0.25141221642100603` | `12.652084088396713 ± 0.2124161791544401` |
| Oklahoma Precision@10 | `0.1454092664092687 ± 0.001785057794383322` | `0.2003822393822424 ± 0.0006772623400256648` | `0.14655984555984777 ± 0.001969070330394919` | `0.2003822393822424 ± 0.0006772623400256648` |
| Oklahoma Degree EG@10 | `3.6190112189763743 ± 0.38153636588877243` | `5.617993197003687 ± 0.12518281465977163` | `9.898910885601222 ± 0.31775091962925106` | `10.281336987915468 ± 0.04397189439228629` |
| Oklahoma PageRank EG@10 | `7.258424840176916 ± 0.2835028355953543` | `9.845780841413617 ± 0.1265985817481991` | `7.3155532824188025 ± 0.38848224262070663` | `9.075033755659065 ± 0.10150938550788986` |
| UNC Precision@10 | `0.07959963603275566 ± 0.0011625131212610832` | `0.1198028510767388 ± 0.0003540895000284915` | `0.08029117379435727 ± 0.0019396794485580136` | `0.12017288444040275 ± 0.00025645058931061834` |
| UNC Degree EG@10 | `-6.131182361722696 ± 0.3468750138863537` | `0.24487869585977484 ± 0.18045837178004545` | `5.996503793063118 ± 0.24807731649383885` | `8.932829555745425 ± 0.08765978619939399` |
| UNC PageRank EG@10 | `2.4377355185869694 ± 0.2506171349445179` | `7.624674794897153 ± 0.12008439504144658` | `3.360749810746711 ± 0.25561263907681975` | `7.366610654902206 ± 0.0895667355830938` |
| Brightkite Precision@10 | `0.019036474164133545 ± 0.0006678704529170754` | `0.021512158054710982 ± 0.0004391150601063535` | `0.019036474164133545 ± 0.0006678704529170754` | `0.021683890577507324 ± 0.0005105478281414532` |
| Brightkite Degree EG@10 | `-8.103306671900459 ± 1.0626407208463597` | `-3.160457108595104 ± 0.8182022451314399` | `0.31314481615522904 ± 0.4810331829427728` | `2.4357878793952503 ± 0.4149137797762859` |
| Brightkite PageRank EG@10 | `-0.95143423746416 ± 0.6043381824012578` | `0.5483236389425594 ± 0.40906723387141514` | `-0.6836785675144522 ± 0.6088302556535192` | `0.7540104795368956 ± 0.47872128702356687` |

Exact source artifacts are the old/corrected CAFE-In pairs `results_repro/ego-facebook/cafe_in/results_cafe_in_ego-facebook_{20260719_134119,20260728_193911}.json`, `results_repro/oklahoma/cafe_in/results_cafe_in_oklahoma_{20260719_134119,20260728_194544}.json`, `results_repro/unc/cafe_in/results_cafe_in_unc_{20260719_134119,20260728_200537}.json`, and `results_repro/brightkite/cafe_in/results_cafe_in_brightkite_{20260719_134119,20260729_100316}.json`; and the corresponding old/corrected HL-GNN/fusion/CAFE-Pre artifacts `results_repro/ego-facebook/cafe_p/results_cafe_p_{baseline_ego-facebook_20260719_165354,ego-facebook_20260719_165354,baseline_ego-facebook_20260728_151340,ego-facebook_20260728_151340}.json`, `results_repro/oklahoma/cafe_p/results_cafe_p_{baseline_oklahoma_20260719_170453,oklahoma_20260719_170453,baseline_oklahoma_20260728_151340,oklahoma_20260728_151340}.json`, `results_repro/unc/cafe_p/results_cafe_p_{baseline_unc_20260719_172239,unc_20260719_172239,baseline_unc_20260728_151335,unc_20260728_151335}.json`, and `results_repro/brightkite/cafe_p/results_cafe_p_{baseline_brightkite_20260719_181331,brightkite_20260719_181331,baseline_brightkite_20260728_211924,brightkite_20260728_211924}.json`. Braces here abbreviate literal filenames; they are not inferred files. Pokec-n has no corrected fixed-configuration HL-GNN or CAFE-In JSON, so the five-network fixed-config claim is incomplete.

The completed alternative-config aggregate, `results_egfix/20260724_degree_eg_sum_in_v1/todo3/finalized/aggregate_metrics_summary.csv`, does contain Pokec-n. At its different selected configuration it quotes HL-GNN `0.043104918032784406 ± 0.0006761271058608458` Precision, `1.2010818770824752 ± 0.2475027303101114` Degree EG, and `-1.7983015597510643 ± 0.3116970090824958` PageRank EG; CAFE-In is `0.05469955290610393 ± 0.0001263483078312976`, `9.029594603343492 ± 0.1208426132594186`, and `6.454169387416494 ± 0.1335328576111219`, respectively. This is directional evidence only: its CAFE-In configuration is `lambda_fair_0.0_beta_5.0_lam_1.0`, not the draft's Pokec-n configuration.

### E2. Mandatory sign and method-order changes

- UNC HL-GNN Degree EG flips from `-6.131182361722696 ± 0.3468750138863537` to `5.996503793063118 ± 0.24807731649383885`. Brightkite HL-GNN flips from `-8.103306671900459 ± 1.0626407208463597` to `0.31314481615522904 ± 0.4810331829427728`.
- Brightkite CAFE-In Degree EG flips from `-3.160457108595104 ± 0.8182022451314399` to `2.4357878793952503 ± 0.4149137797762859`. Brightkite CAFE-Pre flips from `-5.224240731494531 ± 0.8154691498702288` to `2.0711039575012835 ± 0.46390421450369995`.
- The Oklahoma Degree-EG order between the backbone and RA fusion reverses. Old HL-GNN is `3.6190112189763743 ± 0.38153636588877243` and old fusion at beta `5.0` is `4.413711891184425 ± 0.0923048575597987`; corrected HL-GNN is `9.898910885601222 ± 0.31775091962925106` and corrected fusion is `9.6967040321928 ± 0.04675729190628116`. Thus fusion no longer improves Degree EG there.
- The four corrected fixed-config fusion comparisons are: Ego-FB HL-GNN `13.243187832043793 ± 0.17772863268757444` versus fusion `12.020068512432786 ± 0.049682065144891534`; Oklahoma `9.898910885601222 ± 0.31775091962925106` versus `9.6967040321928 ± 0.04675729190628116`; UNC `5.996503793063118 ± 0.24807731649383885` versus `7.400263019040757 ± 0.08849930406733558`; Brightkite `0.31314481615522904 ± 0.4810331829427728` versus `1.4479184186378067 ± 0.21074338828717035`. Pokec-n fixed-config fusion is missing. The complete alternative-config aggregate instead quotes Pokec-n HL-GNN `1.2010818770824752 ± 0.2475027303101114` and fusion `6.368012582063278 ± 0.07880909825692795`.
- A separate approximate-closeness change produces another sign flip: deterministic Pokec-n RA Closeness EG is old `-1.2663594754554173 ± 0.0` and corrected `0.2920720523841337 ± 0.0`. `metrics/centrality_equality_gain.py:388-410` resets the approximation RNG, so this is not attributable solely to the Degree/PageRank graph-semantics fix.

These are material story changes even when a broad “CAFE beats the backbone” direction remains.

### E3. Corrected stratified evidence

The current published-config stratified artifacts carry corrected semantics, explicit training seeds `{42,43,44}`, and split seed `42`. For CAFE-In, the old Table 18 endpoint values and corrected endpoint means are:

| Dataset | Old peripheral → core Degree EG | Corrected peripheral → core Degree EG, mean ± SD |
|---|---|---|
| Ego-FB | `+8.49` → `-1.68` | `7.925493151879809 ± 0.004383503794322965` → `-1.3212786606559377 ± 0.0012498611896920783` |
| Oklahoma | `+7.21` → `-5.27` | `6.861527404031438 ± 0.020530333403496158` → `-3.6077288531133296 ± 0.011755022186055352` |
| UNC | `+10.00` → `-4.40` | `9.676043353997697 ± 0.032300415651513405` → `-3.125025110015347 ± 0.015835916415490297` |
| Pokec-n | `+6.10` → `-1.30` | `6.073582844469246 ± 0.018954493419715626` → `-0.9119555172003645 ± 0.0023830136774016506` |
| Brightkite | `+4.32` → `+0.55` | `5.1742327128922705 ± 0.39930006130951345` → `0.7626669873590833 ± 0.022981334446399673` |

Sources: old `results_repro/tables/table_bin_perturbation.tex`; corrected `analysis/plots/stratified_cafe/eg_by_coreness_cafe_<dataset>.csv`. The direction and Brightkite exception survive. The old-to-new deltas and corrected Spearman/bin-size-control statistics are not stored: **CANNOT VERIFY FROM AVAILABLE FILES**.

The corrected published-config precision endpoints also preserve the dataset-dependent ordering. CAFE-In is `0.1054320987654321 ± 0.0004276668660664029` at the Ego-FB periphery and `0.5106209150326797 ± 0.002418079834468763` at the core; Oklahoma `0.05953216374269007 ± 0.0010719475309838228` and `0.34915824915824917 ± 0.002367460718013057`; UNC `0.04783333333333334 ± 0.0006291528696058984` and `0.25230352303523035 ± 0.0022388498159699463`; Pokec-n `0.018016528925619835 ± 0.0002627931684344219` and `0.09807175596649281 ± 7.242888722863692e-05`; Brightkite reverses with `0.036329692896480914 ± 0.001408636761754575` and `0.0027593818984547464 ± 0.0002529015284191965`. Exact sources are `analysis/plots/stratified_cafe/summary_precision_by_bin_{ego-facebook,oklahoma,unc,pokec_n,brightkite}.csv` (braces abbreviate those five literal filenames).

At the draft's old configurations, the RQ3 threshold contrast was not regenerated as a stored delta. The current old-config lowest-bin means are, for example, Ego-FB fusion `0.10106995884773662 ± 0.0006213855502280415` and CAFE-In `0.1054320987654321 ± 0.0004276668660664029`; their paired difference is not stored.

At the corrected TODO3-selected configurations, however, `ToRS_paper/investigation/should_update.md` stores the CAFE-In-minus-Fusion lowest-bin Precision@10 changes as Brightkite `+0.01061`, Ego-FB `-0.00288`, Oklahoma `-0.02515`, Pokec-n `-0.00034`, and UNC `-0.00158`. These entries match the corresponding method means in `todo3/finalized/stratified_precision_summary.csv`. Therefore the answer box's “at most 0.0022” statement is a **MISMATCH**, not merely an unavailable calculation.

### E4. Corrected validation selection changes what “full CAFE-In” denotes

The draft's old fixed configurations use nonzero CAFE-In penalties: Ego-FB `0.3`, Oklahoma `0.3`, UNC `1.0`, Pokec-n `0.05`, Brightkite `0.05` (`analysis/best_configs.csv`). Corrected validation selection does not reproduce that method definition:

| Dataset | Corrected mean-reduction selected CAFE-In | Corrected sum-reduction selected CAFE-In |
|---|---|---|
| Ego-FB | `lambda_fair_0.0_beta_0.5_lam_1.0` | `lambda_fair_0.0_beta_0.1_lam_0.5` |
| Oklahoma | `lambda_fair_0.0_beta_1.0_lam_1.0` | `lambda_fair_0.0_beta_0.5_lam_1.0` |
| UNC | `lambda_fair_0.0_beta_2.0_lam_1.0` | `lambda_fair_0.0_beta_5.0_lam_1.0` |
| Pokec-n | `lambda_fair_1.0_beta_0.1_lam_1.0` | `lambda_fair_0.0_beta_5.0_lam_1.0` |
| Brightkite | `lambda_fair_0.1_beta_5.0_lam_1.0` | `lambda_fair_1.0_beta_5.0_lam_1.0` |

Sources: `results_egfix/20260723_degree_eg_v1/selected_configs.csv` and `results_egfix/20260724_degree_eg_sum_in_v1/selected_cafe_in_configs.csv`. The selection files store validation means and three selection runs, but no across-seed SD or old-to-new delta.

At `lambda_fair = 0`, the in-processing term contributes exactly zero because the implemented loss is `base_loss + lambda_fair * fairness_penalty` (`fairness_methods/inprocessing.py:99-127`). Thus, under mean selection, CAFE-In has no in-processing penalty on Ego-FB, Oklahoma, or UNC; under sum selection it has none on those three datasets and Pokec-n. Calling those selected models a three-stage “full pipeline” is a **MISMATCH**.

The draft further says “at the largest selected value lambda_mob = 0.5” (Section 5.4, p. 12). Every mean-reduction selected row above uses `lambda_mobility = 1.0`; four of five sum-reduction rows also use `1.0`. Verdict: **MISMATCH**.

### E5. Current fixed-config demographic evidence

These metrics are not directly changed by EG semantics, but the rerun changed some stochastic results and Pokec-n remains missing. Each cell is `mean ± SD`.

| Dataset / method | Corrected Precision@10 | Corrected DeltaSP@10 | Corrected DeltaEO@10 |
|---|---:|---:|---:|
| Oklahoma HL-GNN | `0.14655984555984777 ± 0.001969070330394919` | `0.0010644179138075195 ± 0.00004809651864202101` | `0.012951716802944815 ± 0.00717146803546067` |
| Oklahoma fusion | `0.19996911196911493 ± 0.00048525922296104604` | `0.001002105194082871 ± 0.000015856167978234714` | `0.015775819931810636 ± 0.003020710431877205` |
| Oklahoma CAFE-In | `0.2003822393822424 ± 0.0006772623400256648` | `0.0009233732484517278 ± 0.000010525910184676697` | `0.01300516700359049 ± 0.002170903950348672` |
| Oklahoma CAFE-Pre | `0.18722393822394093 ± 0.0008500261093854447` | `0.0009137326020479142 ± 0.00003971808949824886` | `0.020169455088555645 ± 0.004451859411075043` |
| UNC HL-GNN | `0.08029117379435727 ± 0.0019396794485580136` | `0.00008533895887817959 ± 0.000029369898559938403` | `0.013906478466733393 ± 0.008562733472779199` |
| UNC fusion | `0.11989080982711793 ± 0.00047335180284867` | `0.00012383649225421062 ± 0.00001583096365510271` | `0.02366684030057235 ± 0.0029251275357310372` |
| UNC CAFE-In | `0.12017288444040275 ± 0.00025645058931061834` | `0.00013688756196319793 ± 0.000007787157953091667` | `0.02121996584723671 ± 0.0027428215064591453` |
| UNC CAFE-Pre | `0.10952380952381113 ± 0.0007653896412781811` | `0.00008889834152608499 ± 0.000020815490881813228` | `0.014707038562700869 ± 0.004069765098319633` |

Old UNC CAFE-In was Precision `0.1198028510767388 ± 0.0003540895000284915`, DeltaSP `0.000123212039158087 ± 0.000010200016508559377`, and DeltaEO `0.022997421205336172 ± 0.0031538063581309497`; old UNC CAFE-Pre was `0.11224446466484857 ± 0.0008535842349300694`, `0.00006654292068485365 ± 0.00002076691510080597`, and `0.01898258193487909 ± 0.005632863374354296`. Old-to-new deltas are not stored. Pokec-n corrected HL-GNN/fusion/CAFE files are absent, and the current corrected Holm output is incomplete.

### E6. Current exposure direction, but at alternative configurations

`results_egfix/20260724_degree_eg_sum_in_v1/todo3/finalized/target_exposure_cafe_in_vs_fusion_summary.csv` stores paired exposure contrasts for seeds `{42,43,44}`. It records all three runs as positive in the peripheral tier and all three as negative in the core tier on every dataset. Selected stored contrasts include:

| Dataset | Peripheral `absolute_delta_mean ± absolute_delta_std` | Core `absolute_delta_mean ± absolute_delta_std` |
|---|---:|---:|
| Ego-FB | `0.00036889083131636094 ± 1.4156699605725788e-05` | `-0.0003233507864350135 ± 3.4828272758313466e-05` |
| Oklahoma | `0.0014785265970100093 ± 7.156505995877367e-05` | `-0.0010844992736340827 ± 2.1118645803886672e-05` |
| UNC | `0.0004287211568015214 ± 1.761768625377614e-05` | `-0.00040354992879173147 ± 1.8013404208776952e-05` |
| Pokec-n | `1.869136518039214e-05 ± 5.431334970979452e-07` | `-2.4375010032710468e-05 ± 5.776871333635618e-07` |
| Brightkite | `1.863059873150906e-05 ± 4.825780800382892e-06` | `-3.5975056376461435e-05 ± 1.1211234090511374e-05` |

The direction is robust, but the CAFE-In selection is the alternative E4 configuration. On four datasets its sum-selected `lambda_fair` is zero, so these files cannot attribute the shift to an in-processing penalty. They support the target-side mobility mechanism, not “penalty and boost together.”

The same primary CSV stores target event-share changes; the repository audit quotes them as Brightkite `+5.35` percentage points peripheral / `-5.82` core, Ego-FB `+4.99` / `-4.15`, Oklahoma `+15.60` / `-10.78`, Pokec-n `+4.75` / `-4.84`, and UNC `+6.42` / `-5.23`.

### E7. Stored corrected CAFE-In-minus-Fusion contrasts

The completed TODO3 audit stores the following contrasts, matched to the underlying ten-run aggregate summary:

| Dataset | Delta Precision@10 | Delta Degree EG | Delta PageRank EG | Delta Closeness EG |
|---|---:|---:|---:|---:|
| Brightkite | `+0.00628` | `+1.069` | `+0.961` | `+0.013` |
| Ego-FB | `-0.00325` | `+1.734` | `+6.315` | `-9.841` |
| Oklahoma | `-0.00474` | `+3.449` | `+4.853` | `+14.494` |
| Pokec-n | `-0.00053` | `+2.662` | `+3.610` | `+2.696` |
| UNC | `-0.00022` | `+3.598` | `+5.629` | `+4.404` |

It further records that the Oklahoma Precision contrast is `-2.60%` and is negative in all ten fixed-split runs; UNC remains near zero but has no new equivalence test. For source-stratified Degree EG, all `132` bin-by-run CAFE-In-minus-Fusion differences are positive, ranging from stored `+0.0125` to `+2.155`. Lowest-bin mean differences are Brightkite `+0.817`, Ego-FB `+0.960`, Oklahoma `+1.919`, Pokec-n `+1.148`, and UNC `+2.054`; Oklahoma's largest mean is instead the second-lowest bin at `+1.957`.

### E8. CAFE-In method identity is invalid independently of the EG repair

The configured MLP produces positive-edge outputs with shape `[B,1]` (`backbone/layer.py:147-168`; `backbone/model.py:699-706`). Eq. (8)'s coreness weight is one-dimensional, `[B]`, and the implementation multiplies the two without squeezing (`fairness_methods/inprocessing.py:660-681`). PyTorch therefore broadcasts the product to `[B,B]`: every score is paired with every edge weight before averaging, instead of using the matched-edge products in Section 5.2, Eq. (8), p. 11. Verdict that a nonzero-penalty CAFE-In result evaluates the method written in the draft: **MISMATCH**.

This is not a consequence of the Equality-Gain repair and therefore is not an old-to-new numerical shift. It is an independent validity failure affecting interpretation of both old and new CAFE-In runs. In the completed sum-selected campaign, Ego-FB, Oklahoma, UNC, and Pokec-n set `lambda_fair=0.0`, while Brightkite sets `lambda_fair=1.0` (`results_egfix/20260724_degree_eg_sum_in_v1/todo3/selected_configs.csv`). The first group does not exercise the penalty; the remaining dataset exercises the wrong broadcasted penalty. Thus none of those CAFE-In rows is evidence for the intended nonzero Eq. (8) intervention.

No across-seed SD is stored for the E7 derived contrasts themselves: **CANNOT VERIFY FROM AVAILABLE FILES**. The underlying per-method means and SDs are in `results_egfix/20260724_degree_eg_sum_in_v1/todo3/finalized/aggregate_metrics_summary.csv`; I did not synthesize a contrast SD.

## Numbered abstract claims

The abstract is p. 1 of the authoritative draft.

| ID | Distinct claim | Story status | Evidence verdict | Concrete assessment |
|---|---|---|---|---|
| A1 | Standard link recommenders favor well-connected/core users, degrading peripheral-user accuracy and amplifying structural inequality. | **STILL SUPPORTED** | **MATCH** | Corrected stratified precision rises toward the core on the four random-split datasets and corrected Degree EG falls toward the core on all five (E3). |
| A2 | Coreness organizes these biases along an axis degree alone does not capture. | **WEAKENED** | **CANNOT VERIFY FROM AVAILABLE FILES** | The current one-dimensional stratified results support coreness organization, but the fixed-degree-cell aggregates behind Fig. 1(a) were not regenerated as a stored summary. The paper's `0.046`, `0.094`, `0.218`, and `0.074–0.086` are old rendered values. |
| A3 | Recommendations to higher-coreness users are more accurate on four of five networks. | **STILL SUPPORTED** | **MATCH** | Corrected endpoint means in E3 retain the core-higher ordering on Ego-FB, Oklahoma, UNC, and Pokec-n and the reverse ordering on Brightkite, across RA, HL-GNN, fusion, CAFE-In, and CAFE-Pre. |
| A4 | Recommendations to higher-coreness users are consistently less equalizing on all five. | **STILL SUPPORTED** | **MATCH** | Every corrected CAFE-In endpoint in E3 declines from peripheral to core; the same endpoint order is present for all five methods in the corrected CSVs. |
| A5 | EG is instantiated with degree, PageRank, and closeness. | **STILL SUPPORTED** | **MATCH** | This is a measurement-definition claim; the three metrics are present in the corrected JSON schemas and in Section 3.3, pp. 6–7. |
| A6 | The accuracy gradient persists within fixed degree strata. | **NOT YET TESTABLE** | **CANNOT VERIFY FROM AVAILABLE FILES** | No corrected stored degree-by-coreness aggregate reproduces Fig. 1(a); computing it from per-node rows is outside the evidence rule. |
| A7 | The equality gradient tracks coreness rather than stratum size. | **NOT YET TESTABLE** | **CANNOT VERIFY FROM AVAILABLE FILES** | The old Table 18 correlations are based on stale EG. Corrected endpoints exist, but corrected Spearman and size-matched-gap statistics are not stored. |
| A8 | Both gradients hold across heuristic and GNN-based methods. | **STILL SUPPORTED** | **MATCH** | Corrected stratified files contain RA, HL-GNN, fusion, CAFE-In, and CAFE-Pre; E3's ordering is shared across these families. |
| A9 | CAFE combines GNN embeddings, RA, and pre-/in-/post-processing coreness interventions. | **WEAKENED** | **MISMATCH** | The intended architecture has those stages, but E8 shows that the implemented nonzero in-processing term is not Eq. (8), while E4 shows that corrected selection often sets its weight to zero. No corrected sum-selected dataset evaluates the intended active three-stage CAFE-In. |
| A10 | Standard recommenders concentrate predictive accuracy on core users. | **STILL SUPPORTED** | **MATCH** | Corrected precision endpoints in E3 support this on the four random-split networks, with the paper's stated Brightkite exception. |
| A11 | Standard recommenders concentrate exposure on core users and CAFE redistributes exposure toward the periphery. | **STILL SUPPORTED** | **MATCH** | E6 records peripheral-positive/core-negative exposure contrasts in every stored run on all five networks. This supports the mobility/target-side effect, not a nonzero CAFE-In penalty. |
| A12 | CAFE improves over its GNN backbone on Precision@10 and PageRank EG on all five networks. | **WEAKENED** | **MISMATCH** | The four complete fixed-config corrected datasets preserve the ordering (E1). Pokec-n fixed-config CAFE/HL-GNN is absent. Alternative-config Pokec-n also preserves it, but it changes CAFE-In to `lambda_fair=0.0, beta=5.0, lambda_mobility=1.0`; it cannot complete the stated fixed-config claim. |
| A13 | CAFE improves Degree EG over its backbone on four networks. | **WEAKENED** | **MISMATCH** | All four available corrected fixed-config datasets have at least one CAFE variant above the backbone, but Pokec-n is missing and the original four-network count has not been regenerated. Major sign changes in E2 alter the interpretation. |
| A14 | At least one CAFE variant lies on the Pareto frontier. | **NOT YET TESTABLE** | **CANNOT VERIFY FROM AVAILABLE FILES** | The only stored frontier is stale. No corrected paper-config Pareto artifact exists, and the Pokec-n cells are incomplete. |
| A15 | Without sensitive attributes, CAFE shifts demographic exposure parity in the direction predicted by group–coreness coupling; the effect is small and does not extend to equal opportunity. | **WEAKENED** | **CANNOT VERIFY FROM AVAILABLE FILES** | Oklahoma and UNC fixed-config means retain the predicted DeltaSP direction and no DeltaEO improvement over the backbone (E5), but Pokec-n fixed-config CAFE is missing and corrected Holm tests are incomplete. |
| A16 | Under partial acceptance, PageRank ordering is preserved on all five while degree equalization is acceptance-sensitive. | **NOT YET TESTABLE** | **CANNOT VERIFY FROM AVAILABLE FILES** | Current acceptance CSVs have no config/semantics sidecar and no current stored aggregate/SD; the rendered Table 11 predates them. |

## Numbered contribution claims

The contribution bullets are Section 1, p. 3.

| ID | Distinct claim | Story status | Evidence verdict | Concrete assessment |
|---|---|---|---|---|
| C1 | Coreness is a non-degree structural axis organizing the precision–fairness trade-off across five networks and method families. | **WEAKENED** | **CANNOT VERIFY FROM AVAILABLE FILES** | Five-network/method-family gradients survive (E3), but the corrected fixed-degree control is not stored (A2/A6). |
| C2 | CAFE is a framework combining a learning-stage intervention, RA fusion, and a post-processing boost. | **WEAKENED** | **MISMATCH** | The intended architecture exists, but the selected empirical system either nullifies the learning intervention or applies the broadcasted cross-edge penalty rather than Eq. (8) (E4, E8). |
| C3 | The study compares heuristic, GNN, demographic-fairness, structural-fairness, and tail-node baselines on five networks. | **NOT YET TESTABLE** | **MISMATCH** | Pokec-n corrected stochastic baselines have four repetitions, and corrected Pokec-n HL-GNN/CAFE rows are absent (`audit_p1.md`, Sections 1.2–1.4). |
| C4 | Evidence includes per-seed paired multi-split tests, Holm correction in predeclared families, paired differences, and confidence intervals. | **NOT YET TESTABLE** | **MISMATCH** | Frozen coverage is incomplete and Pokec-n absent. The stored Holm file is stale; no corrected Holm crossing can be checked. |
| C5 | A bin-size control establishes that the stratified equality gradient tracks coreness rather than bin size. | **NOT YET TESTABLE** | **CANNOT VERIFY FROM AVAILABLE FILES** | Corrected EG endpoints exist, but corrected correlations and size-matched gaps do not (E3). |
| C6 | Ablations separate the roles of RA fusion and coreness-aware intervention. | **NO LONGER SUPPORTED** | **MISMATCH** | Corrected selection sets CAFE-In `lambda_fair=0` on three datasets under mean reduction and four under sum reduction (E4). Where the coefficient is nonzero, the code applies the wrong broadcasted penalty (E8). The selected evidence never isolates the intended Eq. (8) intervention. |
| C7 | The coreness stages act on who receives exposure. | **WEAKENED** | **MISMATCH** | Target exposure shifts in the claimed direction (E6), but four sum-selected datasets isolate only the mobility boost and the remaining nonzero penalty is not Eq. (8) (E8). The plural-stage attribution is not tested. |
| C8 | Demographic improvements arise through group–coreness coupling. | **WEAKENED** | **CANNOT VERIFY FROM AVAILABLE FILES** | Oklahoma/UNC fixed means are directionally consistent; Pokec-n and corrected paired tests are unavailable (E5). |
| C9 | The analysis characterizes topology-dependent exceptions in equality. | **STILL SUPPORTED** | **MATCH** | Corrected patterns remain topology-dependent: Ego-FB loses closeness under full CAFE-In while Oklahoma/UNC improve; Brightkite retains its distinct precision profile. Exact old exception counts changed because Degree-EG signs changed (E2). |
| C10 | Structural-equality conclusions are re-evaluated under two relaxed acceptance models, with transfer/non-transfer reported. | **NOT YET TESTABLE** | **CANNOT VERIFY FROM AVAILABLE FILES** | Raw current CSVs exist, but the only aggregate table is stale and current provenance/SD are absent. |

## Numbered “Answer to RQ” box claims

There is no labelled “Answer to RQ1” box in the draft. The RQ1 conclusion is prose in Section 4, pp. 8–9; its claims are already covered by A2–A8 and C1.

### RQ2 box — Section 7.1, p. 17

| ID | Distinct claim | Story status | Evidence verdict | Concrete assessment |
|---|---|---|---|---|
| RQ2.1 | CAFE is the tested method that improves precision and structural equality jointly across networks and EG dimensions. | **WEAKENED** | **MISMATCH** | Four fixed-config corrected datasets preserve strong Precision/PageRank improvements (E1), and alternative-config aggregate results do so on all five; the required Pokec-n fixed-config rows are missing and method identity changes under corrected selection (E4). |
| RQ2.2 | CAFE-In or CAFE-Pre lies on the Pareto frontier in `13 of 15` dataset-by-EG panels. | **NOT YET TESTABLE** | **CANNOT VERIFY FROM AVAILABLE FILES** | No corrected paper-config frontier exists. The old frontier is stale, Pokec-n is incomplete, and the selected configurations changed. |
| RQ2.3 | Where CAFE is not best on EG, the winner is almost always a non-ranker. | **NOT YET TESTABLE** | **CANNOT VERIFY FROM AVAILABLE FILES** | This requires a regenerated corrected method ranking/frontier, which is absent. |
| RQ2.4 | RA is the baseline CAFE does not dominate outright. | **STILL SUPPORTED** | **MATCH** | Corrected Ego-FB RA has Precision `0.237315309651057 ± 0.0`, above CAFE-In `0.23716127004087156 ± 0.0007423911045962698`; Brightkite RA is `0.02217325227963496 ± 0.0`, above CAFE-In `0.021683890577507324 ± 0.0005105478281414532`. |
| RQ2.5 | Ego-FB Closeness is the other exception and is localized to the mobility boost. | **STILL SUPPORTED** | **MATCH** | Corrected Ego-FB HL-GNN Closeness is `33.429341691037365 ± 2.1811067089334406`, matched-beta fusion `33.35889755920291 ± 1.7271171540319732`, CAFE-In-B `32.61167836761675 ± 4.058959002658915`, and full CAFE-In `22.60566825319436 ± 2.6521766960266095`. |

### RQ3 box — Section 7.2, p. 21

| ID | Distinct claim | Story status | Evidence verdict | Concrete assessment |
|---|---|---|---|---|
| RQ3.1 | Beyond fusion, lowest-bin precision moves by at most `0.0022`. | **NO LONGER SUPPORTED** | **MISMATCH** | Corrected selected-config stored changes are Brightkite `+0.01061`, Ego-FB `-0.00288`, Oklahoma `-0.02515`, Pokec-n `-0.00034`, and UNC `-0.00158` (E3). The bound fails. No across-seed SD for these derived contrasts is stored: **CANNOT VERIFY FROM AVAILABLE FILES**. |
| RQ3.2 | The dataset-dependent core-more-accurate ordering never flips. | **STILL SUPPORTED** | **MATCH** | Corrected endpoints preserve core-higher precision on four datasets and Brightkite's periphery-higher exception (E3). |
| RQ3.3 | CAFE moves incoming exposure from core to peripheral nodes in every audited run. | **STILL SUPPORTED** | **MATCH** | The corrected alternative-config exposure summary records peripheral `positive_runs=3` and core `negative_runs=3` on every dataset, with stored event-share changes quoted in E6. |
| RQ3.4 | The penalty and boost together cause that target-side shift. | **NO LONGER SUPPORTED** | **MISMATCH** | Sum-selected CAFE-In has `lambda_fair=0` on Ego-FB, Oklahoma, UNC, and Pokec-n, so those runs have no in-processing penalty (`fairness_methods/inprocessing.py:125`). Brightkite's nonzero case uses the broadcasted cross-edge loss rather than Eq. (8) (E8). No dataset tests the claimed intended penalty-plus-boost combination. |
| RQ3.5 | The exposure shift is the mechanism behind aggregate Degree-EG gain. | **WEAKENED** | **CANNOT VERIFY FROM AVAILABLE FILES** | Exposure moves periphery-ward in every run (E6), aggregate Degree contrasts are positive on all five (E7), and all `132` source-bin-by-run Degree contrasts are positive. Those associations do not exactly decompose the non-additive Gini change, and the intended nonzero penalty is never tested (E8). |
| RQ3.6 | RA fusion supplies precision gains that CAFE retains on every network. | **STILL SUPPORTED** | **MATCH** | The complete alternative-config aggregate quotes fusion and CAFE precision above HL-GNN on all five. Fixed-config Pokec-n remains missing. |
| RQ3.7 | Coreness stages act on exposure rather than on who receives accurate recommendations. | **WEAKENED** | **MISMATCH** | Target exposure moves robustly (E6), but corrected lowest-bin precision changes are not negligible on Brightkite, Ego-FB, or Oklahoma, with Oklahoma `-0.02515` (E3). Moreover, most selected CAFE-In models omit the penalty, so the defensible attribution is to mobility on four networks. |

### RQ4 box — Section 7.3, p. 21

| ID | Distinct claim | Story status | Evidence verdict | Concrete assessment |
|---|---|---|---|---|
| RQ4.1 | The three stages are complementary rather than individually sufficient. | **NO LONGER SUPPORTED** | **MISMATCH** | Corrected validation selects `lambda_fair=0` for most CAFE-In datasets (E4), and the remaining nonzero cases apply a different loss from Eq. (8) (E8). The corrected evidence contains no valid active three-stage instance. |
| RQ4.2 | Single components buy equality only by collapsing accuracy, and Brightkite yields nothing usable. | **STILL SUPPORTED** | **MATCH** | Corrected Brightkite HL-GNN Precision is `0.019036474164133545 ± 0.0006678704529170754`; CI-In, CI-Pre, and MB-Post are `0.0021420965858349695 ± 0.00005117149943273507`, `0.0021596139314418953 ± 0.00006593177525455745`, and `0.0021481074397196992 ± 0.00008394069597203728`, despite very large EG values. |
| RQ4.3 | RA fusion supplies precision and a substantial Degree-EG share on four of five networks. | **NO LONGER SUPPORTED** | **MISMATCH** | Oklahoma's Degree-EG order flips from fusion-above-backbone to fusion-below-backbone (E2). Corrected fixed results show fusion below the backbone on Ego-FB and Oklahoma, above on UNC and Brightkite, with Pokec-n missing; alternative-config results put it above on Pokec-n. The old `four of five` statement does not survive. |
| RQ4.4 | Coreness-aware stages add further equality from fusion's higher operating point. | **WEAKENED** | **MISMATCH** | Corrected selected-config CAFE-In-minus-Fusion Degree/PageRank contrasts are positive on all five (E7). But `lambda_fair=0` on four means the gain is the mobility contrast there, while the remaining nonzero case uses the wrong broadcasted penalty (E8). The intended joint stages are not tested. |
| RQ4.5 | On Oklahoma and UNC the added equality has no measurable precision cost. | **NO LONGER SUPPORTED** | **MISMATCH** | The corrected selected-config Oklahoma Precision contrast is stored as `-0.00474` (`-2.60%`) and is negative in all ten fixed-split runs (E7). UNC is `-0.00022` but has no equivalence test. The conjunctive Oklahoma-and-UNC statement fails. |

### RQ5 box — Section 7.4, p. 24

| ID | Distinct claim | Story status | Evidence verdict | Concrete assessment |
|---|---|---|---|---|
| RQ5.1 | Attribute-blind structural debiasing cannot guarantee demographic parity but is “close to free.” | **WEAKENED** | **CANNOT VERIFY FROM AVAILABLE FILES** | Oklahoma/UNC retain high precision with small DeltaSP values (E5); Pokec-n corrected CAFE and current corrected significance are absent. |
| RQ5.2 | CAFE matches attribute-dependent methods on DeltaSP at `1.4–15×` their Precision@10. | **NOT YET TESTABLE** | **CANNOT VERIFY FROM AVAILABLE FILES** | The ratio is only in the old rendered story; no corrected artifact stores those ratios, and Pokec-n CAFE is missing. I did not calculate new ratios. |
| RQ5.3 | CAFE never observes a sensitive attribute. | **STILL SUPPORTED** | **MATCH** | This is a protocol claim for the CAFE rows; sensitive attributes are loaded only for post-hoc demographic evaluation after recommendations (`experiments/run_cafe.py:643-675`). |
| RQ5.4 | Demographic impact operates through group–coreness coupling: precision favors the more-core group while exposure shifts peripheral. | **WEAKENED** | **CANNOT VERIFY FROM AVAILABLE FILES** | Exposure direction survives (E6), but the current group-level precision-coupling aggregate and complete corrected paired analysis are absent. |
| RQ5.5 | The mechanism narrows gaps on Oklahoma/Pokec-n and widens them on UNC. | **WEAKENED** | **CANNOT VERIFY FROM AVAILABLE FILES** | Corrected fixed Oklahoma CAFE-In DeltaSP `0.0009233732484517278 ± 0.000010525910184676697` is below HL-GNN `0.0010644179138075195 ± 0.00004809651864202101`; UNC CAFE-In `0.00013688756196319793 ± 0.000007787157953091667` is above HL-GNN `0.00008533895887817959 ± 0.000029369898559938403`. Pokec-n is missing. |
| RQ5.6 | Demographic gains do not extend to equal opportunity; DeltaEO is no better than the backbone anywhere. | **WEAKENED** | **CANNOT VERIFY FROM AVAILABLE FILES** | Oklahoma/UNC corrected fixed means retain this direction (E5), but Pokec-n is missing and significance is unfinished. |
| RQ5.7 | DeltaEO degradation is attributable to RA fusion because no CAFE-versus-fusion contrast is significant. | **NOT YET TESTABLE** | **CANNOT VERIFY FROM AVAILABLE FILES** | The only Holm output is stale. The frozen corrected split-resampled files do not yet contain ten seeds, and Pokec-n is absent. |

### RQ6 box — Section 7.6, p. 27

| ID | Distinct claim | Story status | Evidence verdict | Concrete assessment |
|---|---|---|---|---|
| RQ6.1 | CAFE's PageRank ordering is preserved on all five under partial acceptance. | **NOT YET TESTABLE** | **CANNOT VERIFY FROM AVAILABLE FILES** | Current acceptance files contain run-level rows but no current stored aggregate/SD or provenance marker; the rendered Table 11 is stale. |
| RQ6.2 | The equalizing PageRank sign is not necessarily preserved. | **NOT YET TESTABLE** | **CANNOT VERIFY FROM AVAILABLE FILES** | Same missing current aggregate/provenance as RQ6.1. |
| RQ6.3 | Degree-based equalization is acceptance-sensitive and transfers only partially. | **NOT YET TESTABLE** | **CANNOT VERIFY FROM AVAILABLE FILES** | Same missing current aggregate/provenance as RQ6.1. |
| RQ6.4 | EG-TP filters along the measured correctness axis and is uninformative for method comparison. | **WEAKENED** | **CANNOT VERIFY FROM AVAILABLE FILES** | The construction argument remains valid, but the empirical rank/SD claims were not regenerated into a corrected stored summary. |

## Bottom-line story change

The diagnosis survives the equality fix: corrected stratified precision and Degree-EG gradients retain their directions across all five networks. The headline “CAFE improves precision and PageRank EG” direction also survives on every completed corrected fixed-config dataset and in the alternative-config five-network campaign.

The causal story does not survive intact. Corrected validation selection frequently deletes CAFE-In's learning-stage intervention by choosing `lambda_fair=0`; when the coefficient is nonzero, the implementation broadcasts Eq. (8) into a batch cross-product; the manuscript's statement that the largest selected mobility value is `0.5` is false because corrected selections mostly choose `1.0`; and RA fusion no longer improves Degree EG on Oklahoma, reducing the claimed four-of-five role. Those are **NO LONGER SUPPORTED**, not matters of wording.

All exact Pareto, Holm-corrected significance, five-network fixed-config, demographic-Pokec, and acceptance-robustness claims remain **NOT YET TESTABLE** at the frozen read.

## What could not be checked, and why

1. Exact previous-submission claims: **CANNOT VERIFY FROM AVAILABLE FILES** because `RecSys_2026_paper_486.pdf` is absent.
2. Old-to-new numeric deltas: **CANNOT VERIFY FROM AVAILABLE FILES** because no artifact stores them; only old/new means and SDs are available, and I did not calculate differences.
3. Corrected Pareto membership or the printed `13 of 15` count: **CANNOT VERIFY FROM AVAILABLE FILES** because the corrected paper-config frontier was never materialized and Pokec-n is incomplete.
4. Any corrected Holm significance crossing: **CANNOT VERIFY FROM AVAILABLE FILES** because the frozen multi-split CSVs were incomplete and the only Holm output predates them.
5. Complete corrected five-network fixed-config claims: **CANNOT VERIFY FROM AVAILABLE FILES** because corrected Pokec-n HL-GNN, fusion, CAFE-In, and CAFE-Pre are absent.
6. Corrected fixed-degree persistence and bin-size-control coefficients: **CANNOT VERIFY FROM AVAILABLE FILES** because the necessary corrected summaries/correlations are not stored.
7. Current acceptance-model means, SDs, and method ordering: **CANNOT VERIFY FROM AVAILABLE FILES** because current CSVs have no aggregate or provenance sidecar and the generated table is stale.
8. Exact seed-labelled fixed-config CAFE comparisons: **CANNOT VERIFY FROM AVAILABLE FILES** because `run_cafe.py` only reseeds under an opt-in flag (`experiments/run_cafe.py:468-479`) while serializing `seed + run` labels unconditionally (`experiments/run_cafe.py:753-755`); see `audit_p1.md` Section 1.1.
9. Whether live jobs completed after the frozen read: deliberately unchecked. Results appended after `2026-08-04T10:10:30+02:00` are outside this phase's evidence boundary.
10. Results for the paper-defined nonzero edgewise Eq. (8) penalty: **CANNOT VERIFY FROM AVAILABLE FILES** because every available nonzero-penalty CAFE-In run uses the broadcasted implementation described in E8.
