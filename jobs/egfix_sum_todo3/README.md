# RQ3 TODO3: corrected aggregate and five-dataset stratified reruns

This workflow is the isolated phase after the reviewed CAFE-In
`penalty_reduction=sum` selection. It does not alter either selection campaign
and never writes into `results/`, `analysis/plots/`, or `ToRS_paper/`.

It composes:

- CAFE-In: five rows from `20260724_degree_eg_sum_in_v1`;
- CAFE-Pre: five rows from `20260723_degree_eg_v1`.

The execution artifact records `penalty_reduction=sum` for both variants
because the existing runners require one reduction value per dataset.
The original selection reduction remains in a separate lineage column.
This normalization does not change CAFE-Pre: preprocessing does not instantiate
or evaluate the in-processing fairness loss.

## Submit

From the repository root:

```bash
bash jobs/egfix_sum_todo3/submit_reruns.sh \
  20260724_degree_eg_sum_in_v1 \
  20260723_degree_eg_v1 \
  --confirm-selected-configs
```

The guarded DAG is:

1. one CPU provenance/runtime/test preflight;
2. 50 aggregate tasks (five datasets x seeds 42--51);
3. 15 source-stratified tasks (five datasets x seeds 42--44);
4. one strict CPU finalizer after both arrays succeed.

All new files are placed under:

```text
results_egfix/20260724_degree_eg_sum_in_v1/todo3/
```

Before `submission.intent` is written, the launcher may resume only a
recognized, immutable partial preparation whose existing artifacts exactly
match recomputation. Once submission starts, it refuses to resume. If
submission is interrupted after `submission.intent`, inspect
`job_ids.todo3.tsv` and Slurm state; do not rerun the launcher and risk
duplicate GPU work.

## What the finalizer requires

- exact 50/15 success matrices and task checksums;
- the frozen `AUC`, one-negative, batch-16384, `sum` command tuple;
- exact dual-selection lineage and current attested conda environment;
- finite aggregate Precision, Degree EG, PageRank EG, and Closeness EG;
- corrected simple-undirected Degree-EG semantics;
- the deterministic RA aggregate/artifact cross-path check to pass;
- valid top-10 NPZs for all five methods and all 15 stratified tasks.

The same validated NPZs yield an all-five target-side audit without retraining.
It counts directed incoming recommendation events, includes zero-exposure
nodes, and uses right-closed training-coreness terciles:

```text
periphery: c <= q1
middle:    q1 < c <= q2
core:      c > q2
```

Right closure keeps Brightkite's large coreness-one tie in the peripheral
tier. Reciprocal recommendation events are retained separately; this exposure
measure must not be confused with the deduplicated simple-undirected graph used
by Degree EG.

Authoritative completion is:

```text
todo3/_TODO3_COMPLETE
todo3/finalized/todo3_integrity.json
```

Nothing is promoted to the manuscript automatically.
