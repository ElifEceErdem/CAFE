# CAFE / ToRS — Full-Paper Consistency Audit
# Against CAFE_ToRS_EceErdem__5_.pdf plus the revised Section 7.2 draft.
#
# Every item was checked against the tables, not against memory of them.
# Where I recomputed a figure, the computation is shown so you can verify
# rather than trust.
#
# TIER 1 blocks submission. TIER 2 is claim-level. TIER 3 is mechanical.


================================================================
TIER 1 — MUST RESOLVE BEFORE ANYTHING ELSE
================================================================

T1.1  TABLE 8 DISAGREES WITH TABLE 4 ON A DETERMINISTIC METHOD.
      RA is run once. Its Full-acceptance EG values must be identical in
      both tables. They are not:

        RA, Degree EG @ Full     Table 4    Table 8
          Brightkite             -4.42      -8.69
          Ego-FB                  9.10       9.01
          Oklahoma                5.25       5.22
          Pokec-n                 4.64       4.16
          UNC                     0.32       0.23

        RA, PageRank EG @ Full   Table 4    Table 8
          Brightkite             +0.27      -1.43   <- sign flip
          Ego-FB                  4.25       4.11
          Oklahoma                8.61       8.69
          Pokec-n                 3.11       2.85
          UNC                     6.66       6.84

        RA, Closeness EG @ Full  Table 4    Table 8
          Brightkite              0.06       0.20
          Ego-FB                 22.13      14.78   <- -33%
          Oklahoma               12.02      11.98
          Pokec-n                -1.27      +1.52   <- sign flip
          UNC                    13.27      13.20

      Tables 4, 5 and 6 agree with each other exactly (Brightkite HL-GNN
      is identical in 4 and 5; Oklahoma CAFE-In identical in 4 and 6), so
      the divergence is specifically the acceptance-model path against the
      main path.

      CONSEQUENCE: the edge-counting hold is NOT Degree-EG-only. PageRank
      and Closeness diverge too. Section 7.5's closeness analysis and
      Section 7.6's "PageRank EG is robust" conclusion both sit on the
      affected side. Rescope the hold note accordingly.

      This is also visible to any reviewer who cross-checks two tables,
      so it cannot be left unaddressed even if the corrected values turn
      out close to the published ones.

T1.2  UNDOCUMENTED CLOSENESS APPROXIMATION ON LARGE GRAPHS.
      In Table 8, RA (deterministic) carries a nonzero standard deviation
      for Closeness EG on Brightkite (+/-0.51) and Pokec-n (+/-0.47), and
      exactly +/-0.00 on Ego-FB, Oklahoma and UNC. The two with variance
      are the two large networks (58k and 66k nodes).
      Corroborating: Table 4 Brightkite Random Cl EG = 18.34 +/- 25.49;
      SSNet Pokec-n Cl EG = 74.57 +/- 28.87. These are not seed noise.
      Section 3.3 states an exact Wasserman-Faust formula and mentions no
      approximation. If closeness is sampled above a size threshold, say
      so, give the sample size, and report it in Section 3.3 — not in a
      footnote.

T1.3  GROUP SIZES DO NOT SUM TO |V|, AND THE EXPLANATION WAS DELETED.
      A.1: Oklahoma 1,525/1,551 = 3,076; Table 2 says |V| = 3,111.
           UNC      2,307/1,678 = 3,985; Table 2 says |V| = 4,018.
           Pokec-n  47,338/19,231 = 66,569 = |V| exactly.
      A.1 asserts in the same paragraph that "the files carry no missing
      values", so the 35 and 33 missing nodes read as an error.
      The explanation existed in the old Appendix D.2 (nodes isolated in
      the raw data are dropped by the loader: 3,076 of 3,111 and 3,985 of
      4,018) and was cut in the restructure. Restore one sentence.

T1.4  SECTION 7.5 CITES TABLE 5 FOR THREE NUMBERS IT DOES NOT CONTAIN.
      (a) "an 11.0-point drop at lambda_mob = 0.5" — this is the Ego-FB
          shared-backbone probe (Fig. 5), not Table 5.
      (b) "the 11.1-point drop seen in the complete CAFE-In pipeline
          (Table 5)" — Table 5 gives 35.28 -> 22.61 = 12.67 against the
          backbone, and 34.63 -> 22.61 = 12.02 against fusion. The 11.06
          value is Table 13(b), paired protocol.
      (c) "adding it to the fused scorer raises Closeness EG by 1.0 and
          1.7 points (Table 5)" — Table 5 gives +0.90 (Oklahoma) and
          +1.42 (UNC). The 0.98/1.73 values are from the probe in C.2.
      Three protocols yield three values for the same quantity. Quote one
      and cite the table it came from, and say which protocol.

T1.5  SECTION 6.3 PROMISES BASELINE TUNING GRIDS THAT NO LONGER EXIST.
      "Implementation sources, task-specific adaptations, candidate
      construction, negative sampling, and tuning grids for every baseline
      are reported in Appendix ?? and Table ??."
      The baseline-hyperparameter table (old Table 10: FairAdj, LBA,
      MORAL settings) was removed in the restructure; A.4 is now repo URLs
      only. This is a reproducibility regression against the previous
      version. Either restore the table or delete the promise — do not
      leave a dangling reference to deleted content.


================================================================
TIER 2 — CLAIM-LEVEL INCONSISTENCIES
================================================================

T2.1  "ALL THREE PIPELINE STAGES" IS NOT TRUE OF ANY SINGLE VARIANT.
      Table 1 lists CAFE's stage as "pre+in+post"; Section 2.5 claims
      CAFE "intervenes on a structural axis (coreness) at all three
      pipeline stages". But Figure 2 shows Path A (pre) and Path B (in)
      as ALTERNATIVES, and Table 3 confirms it: CAFE-In has no graph
      re-weighting, CAFE-Pre has no loss penalty. No configuration uses
      both.
      This matters because it is your positioning claim against every
      method in Table 1. Reword to "pre OR in, plus fusion, plus post",
      or "at the learning stage (by either of two mechanisms) and at
      inference". As written a reviewer checking Table 3 will read it as
      an overstatement of the contribution.

T2.2  ABSTRACT PARETO CLAIM CONTRADICTS SECTION 7.1.
      Abstract: "Pareto improvements over its base GNN on both
      Precision@10 and nodes' equality according to three dimensions".
      Table 13(a): all four measures improve on Oklahoma and UNC only.
      Brightkite and Pokec-n show no significant Closeness change;
      Ego-FB loses 1.45 Degree-EG and 12.00 Closeness-EG points at
      p<0.001. Section 7.1 states this correctly.

T2.3  ABSTRACT DEMOGRAPHIC CLAIM CONTRADICTS SECTION 7.4.
      Abstract: "CAFE also improves fairness according to demographic
      properties of users". Section 7.4 and the RQ5 box: DeltaSP improves
      on Oklahoma and Pokec-n and REGRESSES on UNC; DeltaEO uninformative.

T2.4  "ALWAYS" / "CONSISTENTLY" IN SECTION 7.4 CONTRADICT APPENDIX C.3.
      Section 7.4: "the demographic group with higher initial coreness
      ALWAYS receives the larger precision uplift". RQ5 box repeats it
      with "consistently". Appendix C.3 documents the UNC peripheral-tier
      reversal (CAFE-In, S0 = .0143 against S1 = .0188). Insert
      "aggregated over coreness tiers".

T2.5  "WIDENS IT ON UNC" IS SIGNIFICANT FOR CAFE-IN ONLY.
      Section 7.4 states it unqualified. Table 15(a): UNC CAFE-In
      +0.033*, CAFE-Pre +0.016 n.s. Table 7 already records this
      correctly; only the prose overstates.

T2.6  "ON UNC, CAFE TRAILS RA" IS FALSE FOR CAFE-PRE.
      Section 7.6. Table 8 UNC PW Degree: RA -0.31, CAFE-In -0.43,
      CAFE-Pre -0.27. CAFE-Pre leads RA. Say "CAFE-In trails RA while
      CAFE-Pre is indistinguishable from it".

T2.7  "THE ONLY SOUND METHODOLOGIES" IS TOO STRONG.
      Section 7.6. Appendix C.4 now concedes EG-PW retains an accuracy
      coupling (higher-rho methods receive heavier edges). Soften, and
      point at C.4 for why the coupling is tolerable under PW and not
      under TP.

T2.8  BRIGHTKITE SPREAD: 0.16 OR 0.15?
      Section 7.6 says "a negligible 0.16-point spread". Table 8
      Brightkite PW Degree: max 2.30 (HL-GNN), min 2.15 (CAFE-In),
      spread 0.15. Probably unrounded values; confirm.

T2.9  CONTRIBUTION (3) PROMISES TWO THINGS THAT DO NOT APPEAR.
      "effect sizes" — Cohen's d_z is in the released CSVs only; Cliff's
      delta appears only for group-coreness coupling.
      "the weights of the configuration selection procedure" — there is
      no resampled-ASF-weights analysis anywhere in the paper.

T2.10 SECTION 6.2 PROMISES AN APPENDIX THAT DOES NOT EXIST.
      "aggregate multi-split results are reported in Appendix ??".
      Your multi-split protocol is a genuine methodological strength and
      is currently underused; this is worth delivering rather than cutting.

T2.11 SECTION 5.5 CLAIMS MODEL-AGNOSTICISM ON ONE BACKBONE.
      "the pre-processing, score fusion, and post-processing components
      are model-agnostic by construction: while we demonstrate CAFE with
      HL-GNN". Either soften to a design property rather than a
      demonstrated one, or list the missing evidence in limitations.

T2.12 SECTION 7.5 HAS NO RQ AND NO LABELLED ANSWER BOX.
      Sections 7.1-7.4 and 7.6 each end with a boxed "Answer to RQn".
      Section 7.5's box is unlabelled and the section is not tied to any
      research question. Six RQs, seven results subsections.

T2.13 SECTION 3.3 REFERS TO "RQ8". There is no RQ8; the paper has six.

T2.14 TAIL-GNN IS POSITIONED BUT NOT EVALUATED.
      Table 1 and Section 2.3 include Tail-GNN [23]; Section 6.3's
      tail-node baselines are LTLP and Tail-STEAK only. Add one clause
      saying why (link-prediction adaptation unavailable, or subsumed by
      LTLP/Tail-STEAK), or a reviewer will ask.

T2.15 THE PERIPHERAL-TO-CORE RATIO CLAIM CHANGED REFERENCE POINT.
      The PDF's Section 7.2 says the ratio "is effectively unchanged on
      Oklahoma and decreases on Ego-FB, Pokec-n, and UNC" — measured
      against HL-GNN. Your revised Section 7.2 measures against
      matched-beta fusion and finds movement TOWARD one on four of five.
      Both can be true. If both survive into the final draft without
      their reference points stated they read as a contradiction. Check
      Section 7.1 and the introduction for surviving copies of the old
      wording.


================================================================
TIER 3 — NAMING, NOTATION, REFERENCES
================================================================

T3.1  FOUR NAMES FOR REFERENCE [27].
        "Cross-links Matter"   (Table 1)
        "Cross-link Matters"   (Sections 2.3, 6.3)
        "Cross-Link Bias"      (Tables 4, 6)
        "CrossLinkBias"        (Table 14)
      Pick one and use it everywhere including figure legends.

T3.2  THREE SPELLINGS OF TAIL-STEAK.
        "Tail-STEAK"  (Sections 2.3, 6.3, Table 1)
        "TailSTEAK"   (Table 4)
        "TailSteak"   (Fig. 3 legend)

T3.3  FIGURE 3'S LEGEND CONTAINS ENTRIES THAT APPEAR NOWHERE ELSE.
      "WeakTie" and "WeakTie (no re-rank)" are plotted and labelled in
      Fig. 3 (and referenced in the Brightkite and Oklahoma Closeness
      panels) but appear in no table, no baseline list, and no prose.
      If this is Sanz-Cruzado & Castells' weak-tie recommender [36],
      it needs a row in Table 4 and a paragraph in Section 6.3. If it is
      a leftover from an earlier experiment, remove it from the figure.
      As it stands, the figure shows a method the paper never describes.

T3.4  FIGURE 3 AND THE TABLES USE DIFFERENT NOTATION FOR THE ABLATIONS.
      Figure legend: CAFE-P, CAFE-In-B, CAFE-Pre-B, HLGNN, HLGNN+RA,
        MB-In, MB-Pre, CI-Post, ego-facebook, pokec_n, unc, brightkite
      Tables: CAFE-P (as CAFE minus P), CAFE-In-B, CAFE-Pre-B, HL-GNN,
        HL-GNN+RA, Ego-FB, Pokec-n, UNC, Brightkite
      Also: MB-In, MB-Pre and CI-Post appear in the Fig. 3 legend but not
      in Table 3's configuration list, which defines MB-Post, CI-In and
      CI-Pre. Either they are additional configurations that Table 3 omits
      or the legend is stale. Check.

T3.5  SECTION HEADING 5 READS "Coreness-Aware Fair rEcommendation".
      The lowercase-E acronym device works in the abstract; in a section
      heading it reads as a typo. Consider spelling it normally in the
      heading and keeping the device only where the acronym is coined.

T3.6  BROKEN REFERENCES (??) — inventory by location.
      Section 1:   "Sections ??-??" (discussion/limitations/conclusion)
      Section 5:   "Eq. (??)" for the GNN link score (should be Eq. 6)
      Section 5.2: "RA (Section ??)" (should be 5.3)
      Section 5.5: "Table ??" for wall-clock (Table 9); "Appendix ??"
      Section 6.1: "Section ??" for split limitations
      Section 6.2: "Section ??" for significance tests; "Appendix ??"
      Section 6.3: "Appendix ?? and Table ??" (see T1.5)
      Section 6.5: "Appendix ??" for hardware
      Section 7.2: "Appendix Figure ??", "Appendix Table ??",
                   "Section ??" x3, "Table ??"
      Section 7.3: "Section ??" (hub suppression, = 5.3),
                   "Table ??b" x2 (= Table 13(b))
      Section 7.5: "Appendix Table ??a", "Section ??"
      Section 7.6: "Appendix ??"
      Table 11:    "Reported in" column, five entries
      Table 12:    "evaluated source users of Table ??"
      Table 16:    "recovers Table ??, panel (b)"
      Appendix C.2: "Section ??" opener, "Table ??" x2
      Appendix C.3: "Table ??" x3, "Section ??" x2

T3.7  UNDEFINED LABELS (referenced, never declared).
        tab:strata         <- cited by Section 7.5 AND Appendix C.2;
                              most urgent, it carries Section 7.5's
                              entire stratum-level argument
        tab:group-uplift   <- the only source for the UNC reversal
                              numbers quoted in C.3
        tab:coupling       <- now load-bearing for C.3's within-tier
                              argument (quintile shares)
        tab:evalpop        <- cited by Table 12's caption

T3.8  TABLE 16 CAPTION IS FACTUALLY WRONG AFTER THE 7.2 REVISION.
      "Pooling the two groups recovers Table ??, panel (b)."
      Table 16 is measured against HL-GNN; the revised Section 7.2 panel
      (b) is measured against matched-beta fusion. The underlying numbers
      ARE mutually consistent — pooling Table 16's Pokec-n CAFE-In rows
      against its fusion rows gives +99.0%/+99.7% versus your +99.2%;
      Oklahoma core gives -2.8% versus -2.7%; UNC periphery ~+29% versus
      +29.4% — so only the caption is wrong.

T3.9  BIN COUNTS STATED TWICE.
      The revised Section 7.2 gives 6/10/9/10/9; Appendix C.1 and Table 12
      give the same counts. Keep the counts in the main text and let C.1
      carry only the within-bin-comparison argument plus Table 12's
      ranges and extremes.

T3.10 SECTIONS 6.5 AND A.2 ARE NEAR-DUPLICATES.
      Both state the HL-GNN hyperparameters, PyTorch/PyG, K=10, and code
      availability. Keep the full statement in A.2 and reduce 6.5 to a
      pointer.

T3.11 TABLE 14'S CAFE ROW IS TRUNCATED MID-EXPRESSION in the rendered
      PDF ("RA matrix O(sum deg(v)^2) + coreness O("), and the source
      comment flagging the LTLP and TailSTEAK cells as unverified
      transcriptions is still present.

T3.12 PREAMBLE / FRONT MATTER.
      \acmJournal is still JACM (running foot reads "J. ACM, Vol. 37,
      No. 4"); the ACM reference-format line still says "In Proceedings
      of 20th ACM Conference on Recommender Systems (RecSys '26)".

T3.13 INTRO TYPO: "CAFE CAFE improves the balance".
      RQ2 phrasing: "does improving tail-node accuracy, already deliver
      structural equality?" — stray comma, misplaced "already".


================================================================
WHAT I CHECKED AND FOUND CORRECT
================================================================
Recording these so you know where NOT to spend time.

  * Section 7.1's "four of five" Degree EG, "all five" PageRank EG, and
    the Ego-FB-only-regression claim all match Table 13(a).
  * "31-69% lower precision" for Cross-Link Bias: recomputed as 69%
    (Brightkite), 31-36% (Oklahoma), 36-40% (UNC). Correct.
  * "LTLP reduces Precision@10 by 28-77%": recomputed 28.5%-76.7%.
    "TailSTEAK by 57-94%": recomputed 56.8%-94.5%. Both correct.
  * "74% lower precision" for LTLP on Brightkite: 0.0057/0.0215 = 73.5%.
  * "two of twenty paired measures": Table 13(c) gives exactly two
    significant negatives (Brightkite Prec, Pokec-n Degree EG).
  * Section 7.3's "11%-37%" and "50%-89%" single-stage precision losses.
  * "+14% to +51%" fusion precision gain over the backbone.
  * "10 of the 15 dataset-EG comparisons": counted 2+0+2+3+3 = 10.
  * "40%-68% of the backbone-to-CAFE-In Degree-EG gain": recomputed
    49.0% / 39.5% / 67.9% / 54.5% for Brightkite/Oklahoma/Pokec-n/UNC.
  * "improves 23 of the 30 EG values": counted row by row, exactly 23.
  * "+1.25/+1.32/+0.90 and +2.90/+2.39/+1.48" against Table 13(b).
  * "bounded within +/-0.0004 and +/-0.0003" paired precision CIs.
  * Section 7.6's rho values (4.2%, 42.4%, 1.2%-13.4%) against Table 17.
  * Bin counts 6/10/9/10/9 against Table 12.
  * Rerun counts (n=3 source, n=10/n=3 target) against Table 12 and
    Table 16 captions.
  * Tables 4, 5 and 6 agree with each other exactly where they overlap.
