# CAFE / ToRS — Writing TODO
# Abstract, Introduction, Discussion, Limitations, Future Work, Conclusion
#
# Every item is grounded in a specific section, table, or number in the
# current draft. Items marked [BLOCKER] must be fixed before submission;
# they are places where the paper currently contradicts itself or promises
# something it does not deliver.


================================================================
PART 1 — ABSTRACT
================================================================

[BLOCKER] A1. Fix the Pareto claim.
  Current: "CAFE achieves Pareto improvements over its base GNN on both
  Precision@10 and nodes' equality according to three dimensions of
  influence: degree, PageRank, and closeness."
  Table 13(a) shows this holds on Oklahoma and UNC only. Replace with the
  disaggregated version, which is still a strong result:
    - Precision@10 and PageRank EG improve on all five networks
    - Degree EG improves on four of five
    - Closeness EG improves on two, is unchanged on two, regresses on one
  Naming the Ego-FB regression in the abstract costs you two clauses and
  removes the single easiest reason to reject.

[BLOCKER] A2. Fix the demographic claim.
  Current: "CAFE also improves fairness according to demographic
  properties of users, without requiring any sensitive demographic
  attributes."
  §7.4 and the RQ5 box: DeltaSP improves on Oklahoma and Pokec-n,
  regresses on UNC; DeltaEO is uninformative. Rewrite to the mechanism
  claim, which is what you actually own:
    "...shifts demographic exposure parity without observing sensitive
    attributes, in a direction set by the coupling between group
    membership and coreness — improving parity on two of three networks
    and worsening it on the third."
  The mechanism is the contribution. The direction is not yours to
  promise.

[BLOCKER] A3. Separate the two halves of the diagnosis.
  Current: "recommendations to nodes with higher coreness have higher
  Precision@10 and lower fairness ... consistent across heuristic and
  GNN-based methods."
  Three problems.
    (i) "nodes" should be "source users" — §4 and §7.2 stratify by
        source-user coreness, not target.
    (ii) The abstract states the precision half and the equality half as
        one finding with one level of confidence. They do not have the
        same exposure (see L1). The precision gradient depends on how
        much evidence survives the split; the Degree-EG gradient does
        not, because EG is computed on the recommendation-induced graph
        and does not depend on whether the recommendations are correct.
    (iii) Brightkite reverses the precision gradient for every method
        (§7.2) while the Degree-EG gradient holds.
  Rewrite so the load-bearing half leads: recommendations issued to
  higher-coreness source users are systematically less equalizing
  (consistent on all five networks), and are typically more accurate
  (on four of five; the direction of the precision gradient depends on
  the evaluation regime). This is a stronger abstract, not a weaker one —
  the half that carries CAFE is the half that cannot be an artefact.

A4. Match the "key insight" sentence to the intro's.
  Abstract: "over-invest representational capacity in core-core pairs,
  negatively affecting both precision and fairness."
  Intro: "...creating a blind spot that harms both precision and fairness
  FOR PERIPHERAL USERS."
  The intro version is the supportable one. As written the abstract implies
  the over-investment hurts precision overall, which Table 5 contradicts —
  every single-stage ablation that removes it LOWERS precision (11%-89%).

A5. Add the RQ6 finding. One sentence.
  The abstract currently omits what may be the paper's most transferable
  result: degree-based equality conclusions are sensitive to the
  acceptance model, whereas PageRank-based ones are not. This qualifies
  prior simulation evidence [11] in a new setting and it is a finding
  other researchers will cite. Do not hide it.

A6. Tighten two phrasings.
  - "lower fairness (measured as variation in equality according to
    network centrality metrics)" — unclear. Say: measured as the change
    in centrality inequality that the recommendations induce.
  - "three dimensions of influence: degree, PageRank, and closeness" —
    §3.3 calls these visibility, influence, and accessibility. Only one
    is influence.

A7. Change the ACM reference-format line.
  It currently reads "In Proceedings of 20th ACM Conference on
  Recommender Systems (RecSys '26)". Also set \acmJournal to TORS
  (currently JACM in the preamble) so the running foot stops saying
  "J. ACM, Vol. 37, No. 4".


================================================================
PART 2 — INTRODUCTION
================================================================

[BLOCKER] I1. Contribution (3) promises something that does not exist.
  "...hyperparameter sensitivity analyses, including the weights of the
  configuration selection procedure."
  There is no resampled-ASF-weights analysis anywhere in the paper. A.3
  no longer promises it, so the intro is the only place it is claimed.
  Either run it (a re-scoring of the existing sweep — no retraining) or
  delete the clause.

[BLOCKER] I2. Contribution (3) also promises "effect sizes."
  Cohen's d_z is in the released CSVs, not the paper; Cliff's delta
  appears only for group-coreness coupling. Either add a d_z column to
  Table 13 or reword to "effect sizes released with the per-seed data."

I3. Fix the organization paragraph.
  "Sections ??-?? discuss implications, limitations, and conclusions"
  will become Sections 8-9 under the structure in Part 3. Also update
  the appendix description: it now says "protocol details, sensitivity
  analyses, cost measurements, and supplementary results", but sensitivity
  now lives inside A.3 and cost inside A.2.

I4. Preview the RQ3 asymmetry.
  §7.2 concludes that peripheral precision improves in absolute terms but
  the peripheral-to-core RATIO does not close. The intro promises
  "redistributing attention toward peripheral pairs" and a reader will
  reasonably expect the gap to narrow. One sentence in the "Our proposal"
  paragraph setting the expectation correctly prevents a reviewer
  discovering it as a shortfall in §7.2.

I5. Decide what to do about §7.5.
  You have six RQs and seven results subsections; §7.5 (The Closeness
  Exception) has no RQ. Two clean options:
    (a) frame it in the RQ list as an analysis of a residual cost
        identified under RQ2, and say so in §7.5's opening sentence; or
    (b) promote it to its own RQ and renumber (RQ6 closeness, RQ7
        robustness).
  Option (a) is less disruptive. Either way, say which, because right now
  the section looks orphaned.

I6. Hedge "Pareto improvements over strong baselines."
  RQ2's own answer says CAFE-In is significantly worse than RA on two of
  twenty paired measures. "Pareto improvements over strong baselines"
  in the intro is not what §7.1 concluded. Use §7.1's own framing:
  the only method improving both jointly across networks and EG
  dimensions rather than in isolated panels.

I7. Typos and grammar in the intro.
  - "CAFE CAFE improves" (duplicate word)
  - RQ2: "does improving tail-node accuracy, already deliver structural
    equality?" — stray comma, "already" misplaced
  - §3.3 refers to "Section 7.6, RQ8"; there is no RQ8


================================================================
PART 3 — RECOMMENDED STRUCTURE
================================================================

  8   Discussion
  8.1   What coreness adds, and what it does not
  8.2   Practical implications
  8.3   Limitations and threats to validity
  8.4   Future work
  9   Conclusion

  Rationale: three separate top-level sections fragment ~2 pages.
  Future work sits after limitations so each limitation names its own
  remedy. The conclusion carries no new content and no new citations.

  CHECK the ToRS author guidelines first: if a separately labelled
  limitations statement is required, promote 8.3 to §9 and renumber.


================================================================
PART 4 — DISCUSSION (§8.1-8.2) TODO
================================================================
Rule for this section: interpretation only. If a sentence could be moved
into §7 without loss, it belongs in §7.

§8.1 — What coreness adds, and what it does not

  D1. Why the axis is coreness and not degree. Recursive embeddedness
      means a node's own score depends on whether its neighbours are
      themselves embedded — which is exactly the property that determines
      how much common-neighbour and multi-hop evidence a scorer has to
      work with. Degree cannot express this. Point to Fig. 1(a) as the
      empirical form.

  D2. The trade-off is not inevitable, but it is not free either. §7.3
      shows fusion buys the operating point and the coreness stages spend
      part of it to buy equality. State this as the honest shape of the
      result rather than "CAFE wins on both."

  D3. Coreness is a diagnostic axis, not only an intervention axis. The
      per-bin profiles (§7.2) let a practitioner see WHERE a recommender
      concentrates advantage before deciding whether to intervene. This
      is arguably the more durable contribution and it is currently
      under-sold.

  D3b. [NEW — state this explicitly, it is currently nowhere in the paper]
      The two halves of the diagnosis have different evidential status,
      and the robust one is the one that matters. Precision@K depends on
      which edges were held out, so it inherits whatever bias the split
      procedure introduces. Equality Gain does not: it compares centrality
      distributions on the recommendation-induced graph and is defined
      whether or not the recommended edges are correct. The intervention
      claim rests on the leakage-immune measure — §7.2's own RQ3 answer
      already concedes the relative precision gap does not close and that
      "degree-based equality is where the intervention visibly acts."
      Say this once, plainly, in §8.1. It converts the paper's biggest
      methodological exposure into a statement about which of your two
      measures is load-bearing, and the answer favours you.

  D4. Attribute-free fairness transfers only through coupling. Group-
      coreness coupling is necessary for the transfer; the DIRECTION is
      set by topology. So the practical claim is diagnostic: measure the
      coupling on your graph and you can predict which group benefits,
      but you cannot promise parity. Note the effect sizes are small to
      negligible (delta = 0.20 / 0.10 / -0.09) — the mechanism is
      established on weak coupling, which is itself informative.

  D5. Structural and demographic fairness are not substitutes. CAFE
      dominates the attribute-dependent baselines on utility and
      structural equality (§7.4) while offering no parity guarantee.
      That is a statement about what the two families of method are for.

  D6. Measurement: the acceptance model matters more than prior work
      assumed. §3.3 declined to import Fabbri et al.'s [11] conclusion
      and verified it instead; the verification came out negative for
      degree-based measures. Say plainly that this qualifies [11] in a
      single-round structural setting, and that it is a finding about
      how the field should evaluate, not only about CAFE.

§8.2 — Practical implications

  D7. lambda_mob must be validated per network. §7.5 establishes the sign
      is topology-dependent. Give the operational rule: validate on a
      held-out split before deploying, and treat the published value as
      a starting point only.

  D8. What to measure before deploying. Group-coreness coupling (predicts
      which group gains), initial Closeness EG and clustering (predict
      whether the boost helps or hurts closeness — Ego-FB's clustering
      0.52 against 0.23/0.16, §7.5).

  D9. Cost is not a barrier. One-time preprocessing is under 1% of a
      training run (Table 9); fusion and boost are score-level and
      model-agnostic. Two sentences, no more — §5.5 and A.2 already
      cover it.

  D10. The intervention is attribute-free, which matters where sensitive
       attributes are legally restricted (§2.2 makes this point; close
       the loop here).


================================================================
PART 5 — LIMITATIONS (§8.3) TODO
================================================================
Ordered by how likely a reviewer is to raise it. The first three are the
ones that decide the review.

  L1. [MOST SERIOUS] Split regime. Four of five networks use random edge
      splits; only Brightkite has even a proxy-temporal ordering.

      What your multi-split protocol does and does not answer. The
      split-resampled runs (§6.2, Table 13) close the "you got a lucky
      split" objection properly — paired within split, all methods on the
      same partition, Holm-corrected. Cite this prominently; it is a
      real methodological strength and it is currently underused in the
      writing. But random-versus-temporal is a systematic question, not
      a sampling one. Removing 10% of edges uniformly from a fully formed
      graph leaves G_train contaminated by structure that the held-out
      edges helped create: edge (u,v) is gone, but u and v remain
      embedded in the neighbourhood that formed alongside it, and
      coreness computed on G_train inherits that. Ten draws from the same
      procedure narrow the interval around a possibly-biased estimate;
      they do not move it. Say this yourself rather than letting a
      reviewer say it.

      Why it plausibly runs along your axis. High-coreness nodes retain
      proportionally more residual evidence about their removed edges
      than peripheral nodes do — which is the gradient you are measuring.
      So "all methods saw the same split" does not settle it either: the
      concern is between strata, not between methods.

      The defence, and it is a good one. Split the diagnosis in two
      (D3b). The precision gradient is exposed to this; the Degree-EG
      gradient is not, because EG does not depend on recommendation
      correctness. The intervention claim rests on the immune half, and
      on Brightkite it is the immune half that survives — Degree EG
      remains largest in the lowest-coreness bin, as on the other four
      networks.

      What NOT to claim. Brightkite is not a controlled regime
      comparison. It differs from the other four in at least four ways
      simultaneously: proxy-temporal ordering, sparsest graph (avg.
      degree 7.35), 54.3% of evaluated users in one bin, and 6 realized
      bins against 9-10 elsewhere (Table 12). §7.2 already puts this
      correctly — "reflects the evaluation regime AS WELL AS the
      topology" — and the limitations paragraph must not upgrade it to
      evidence that the regime is the cause. It is a flag, not a test.

      Also state: the Brightkite ordering is a chronological proxy built
      from endpoint check-in dates, not an edge timestamp (§6.1).

      Engage the literature you already cite. Aiyappa et al. [2] is
      currently cited in §2.1 only for degree-based scoring on AUC
      benchmarks. It is directly about how link-prediction task
      construction manufactures structural bias. Citing it here turns
      "we did not consider this" into "we know this literature and here
      is where we sit in it." One sentence.

  L2. Acceptance is a counterfactual, not a behavioural model. Full
      acceptance is the standard convention but no user accepts every
      recommendation. EG-PW is calibrated and parameter-free but retains
      an accuracy coupling (App. C.4); EG-TP is unusable. §7.6 shows
      Degree-EG conclusions are sensitive to which one you pick. Own this
      as a limit on the degree-based claims specifically.

  L3. Single-round, static evaluation. EG scores one expanded graph.
      Real recommenders run in rounds with feedback, and inequality
      compounds across rounds [11]. No claim is made about long-run
      dynamics. Also: coreness is computed once on G_train and never
      recomputed (§5.2), which a deployed system would not do.

  L4. The Ego-FB Closeness regression is real and unexplained at the
      mechanism level. §7.5 traces it to the boost and characterises the
      strata gradient, but the topological account (aggregate of ego
      networks, hubs as global bridges) is a hypothesis. Say so.

  L5. Single backbone. §5.5 claims the pre-processing, fusion, and
      post-processing components are "model-agnostic by construction"
      but only HL-GNN is demonstrated. Either soften the claim or list
      the missing evidence here.

  L6. Fixed K = 10. No sensitivity to K is reported, and §7.6 gives
      reason to think K matters (rank precision decays 3-5x from rank 1
      to 10). Justified by platform practice (§3.3) but untested.

  L7. Binary, single-attribute demographic evaluation. Gender on two
      networks and region on one; two groups each; three of five networks
      have any attribute at all. No multi-group or intersectional
      analysis. In a fairness paper, treating gender as binary should be
      acknowledged explicitly as a property of the released data, not
      passed over.

  L8. Undirected graphs only. Several of the platform studies motivating
      the paper (Twitter "Who to Follow", [41]) are directed. The boost
      is applied to targets and is the sole asymmetry in the ranking
      (§5.4), so the extension is plausible but untested.

  L9. Gini as the inequality functional. It does not decompose into
      between-stratum terms (§7.5 flags this), and footnote 1 documents
      dropping zero-centrality nodes — defensible, but a choice.
      Alternative functionals (Atkinson, Theil) are untested.

  L10. Exposure is a proxy for outcome. EG and DeltaSP measure the
       distribution of recommended edges, not whether users act on them
       or benefit. No user study, no online evaluation.

  L11. Incomplete coverage in two analyses. The sensitivity sweep
       excludes Pokec-n and Brightkite for compute (A.3); the Pokec-n
       stratified profile rests on a single run (§7.2); Pokec-n CI-Pre
       and MB-Post are absent from Table 5. If these do not complete
       before submission, they must be stated here rather than left as
       table footnotes.

  L12. One baseline is non-deterministic. Cross-link Bias depends on
       Louvain, which may yield different structural groups across runs
       (§6.3, already noted — repeat once here).

  L13. DeltaEO is reported but uninformative. It does not survive a
       protocol change (App. C.3) and no error-parity claim is made.
       Better stated as a limitation than left implicit.


================================================================
PART 6 — FUTURE WORK (§8.4) TODO
================================================================
Pair each with its limitation. A reviewer reading L-then-F sees an agenda,
not a list of holes.

  F1. (L1) Temporal splits on all five networks. Does the coreness-
      precision gradient survive a chronological protocol? This is the
      highest-value follow-up in the list and you should say so. It needs
      edge timestamps that four of the five datasets do not ship, so it
      is genuinely future work rather than something you are declining
      to do — say that too.
      Two nearer-term checks are runnable NOW and belong in the paper
      rather than in future work; see X7 and X8.

  F2. (L2, L3) Multi-round simulation with a behavioural acceptance model
      calibrated on logged interaction data. Does the equality gain
      compound, persist, or wash out?

  F3. (L6) K-sensitivity. The concrete first step is already specified in
      App. C.4: recompute Degree EG at K = 1..10 and test whether CAFE's
      advantage over the backbone grows with K on Brightkite and Pokec-n.
      That also converts the rank-differential-placement hypothesis into
      a tested claim.

  F4. (L4, D7) Predicting the sign of the boost's closeness effect from
      topology. Clustering coefficient and initial Closeness EG are the
      obvious candidate predictors; five networks is too few to fit, so
      frame as a question for a larger network sample.

  F5. (L5) CAFE's score-level components on other backbones — LightGCN,
      SSNet, or a non-GNN scorer — to substantiate model-agnosticism.

  F6. (L7) Multi-group and intersectional demographic analysis; and
      whether group-coreness coupling predicts transfer with more than
      two groups.

  F7. (L8) Directed link recommendation, where the source/target
      asymmetry the boost already exploits becomes intrinsic.

  F8. (L3) Dynamic coreness, recomputed as the graph evolves, and whether
      the intervention remains stable under its own feedback.

  F9. Composing structural axes. §2.3 argues community-crossing [27] and
      coreness are related but distinct; whether they compose or conflict
      is directly testable with the code you already have.

  F10. (L9) Alternative inequality functionals, and whether the EG
       conclusions are functional-dependent in the way they turn out to
       be acceptance-model-dependent.


================================================================
PART 7 — CONCLUSION (§9) TODO
================================================================
Three paragraphs. No new content, no new citations, no future work, no
new numbers beyond the headline ones.

  C1. Para 1 — the diagnosis. Coreness organises the precision-fairness
      trade-off in link recommendation beyond what degree explains, and
      the trade-off is an artefact of how scorers interact with topology
      rather than a necessary property of the task.

  C2. Para 2 — the framework and the honest result. CAFE improves
      Precision@10 and PageRank EG on all five networks and Degree EG on
      four, with the Ego-FB closeness regression named rather than
      omitted. Demographic parity shifts as a byproduct through group-
      coreness coupling, in a direction set by topology.

  C3. Para 3 — what the reader should take away. Two sentences: coreness
      is a usable diagnostic axis for auditing any link recommender, and
      structural-equality conclusions depend on the acceptance model in a
      way the field has not been testing.

  C4. Do NOT end on a limitation and do NOT end on future work. Both
      belong in §8.


================================================================
PART 8 — PRE-SUBMISSION CHECKLIST (independent of the above)
================================================================

  X1. Undelivered promises:
      - intro contribution (3): resampled ASF weights — run or cut
      - §6.2: "aggregate multi-split results are reported in Appendix ??"
        — this appendix does not exist
      - §5.5 references "Table ??" for wall-clock (should be Table 9)
        and "Appendix ??"

  X2. Undefined labels still producing ?? in the compiled PDF:
        tab:strata (cited by §7.5 AND App. C.2 — most urgent)
        tab:coupling, tab:group-uplift, tab:evalpop
      Plus "Section ??" in App. C.2 and C.3 openers, and Table 11's
      "Reported in" column.

  X3. Numbers to confirm:
      - §7.6 "0.16-point spread" on Brightkite; Table 8 gives 2.30-2.15
        = 0.15
      - §7.6 "on UNC, CAFE trails RA": true for CAFE-In (-0.43 vs -0.31),
        false for CAFE-Pre (-0.27)
      - Table 9 caption says "mean over 10 seeds"; the earlier A.2 text
        said three repetitions

  X4. Claims to soften, carried from the appendix review:
      - §7.4 "always receives the larger precision uplift" and the RQ5
        box's "consistently" — App. C.3 documents the UNC peripheral-tier
        reversal. Add "aggregated over coreness tiers."
      - §7.4 "widens it on UNC" — significant for CAFE-In only (Table 7
        already records this correctly)
      - §7.6 "the only sound methodologies" — App. C.4 concedes EG-PW
        retains an accuracy coupling

  X5. Preamble: \acmJournal{JACM} -> TORS; author block still contains
      the ACM sample authors (Trovato, Tobin, ...) — harmless under
      \anonymous but fix before camera-ready.

  X6. Table 14 (complexity) still carries the source comment flagging the
      LTLP and TailSTEAK cells as unverified transcriptions, and the CAFE
      row is truncated mid-expression in the rendered PDF.

  X7. [RUN THIS] Removal-rate sensitivity. Re-split at 25% and 40% test
      on one or two networks and check whether the precision-coreness
      gradient attenuates. If it is a leakage artefact the gradient
      should flatten as removal grows; if it is structural it should
      persist. No new data and no new method — two extra training runs
      per network at the backbone plus CAFE-In. This is the closest
      thing to a controlled test of L1 available without timestamps, and
      one figure of it removes the objection's teeth entirely.
      Note the confound to acknowledge: larger removal also sparsifies
      G_train, which changes coreness values themselves. Report X8
      alongside so the two can be read together.

  X8. [RUN THIS — cheap] Coreness stability under removal. Spearman rho
      between C_train and C_obs (coreness on the training graph versus
      on E_train union E_test), per dataset. No training required. If it
      is high, say so: it establishes that the strata are not an artefact
      of which edges happened to be held out, which is a different and
      weaker-but-necessary condition than the gradient itself being
      unbiased. One row of numbers, probably one sentence in §6.2 or a
      line in Table 12.
