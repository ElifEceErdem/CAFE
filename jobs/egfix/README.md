# Corrected Degree-EG rerun campaign

This directory provides a deliberately two-phase Slurm workflow. It keeps all
new artifacts under `results_egfix/<RUN_TAG>/` and never writes into the
historical `results/`, `results_repro/`, `analysis/plots/`, or paper trees.

The protocol isolates the Degree-EG graph-semantics correction. In particular,
it retains the historical in-processing penalty reduction (`mean`), uses one
canonical fixed split per dataset (split seed 42; the checked-in temporal split
for Brightkite), and fixes the approximate-closeness seed at 42.

This campaign clears the five-method fixed-split dependency for Section 7.2:
RA, HL-GNN, matched-beta Fusion, CAFE-In, and CAFE-Pre. It does not rebuild
the complete legacy-baseline table, the varying-split paired-significance
suite, or the ablation tables. Those are separate campaigns if the manuscript
later requires them.

## Workflow

### 1. Submit validation-only configuration selection

Choose a **new** run tag. The submission script refuses an existing campaign:

```bash
cd /home/eerdem/workspace/CAFE
bash jobs/egfix/submit_selection.sh 20260723_degree_eg_v1
```

This submits, using `afterok` dependencies only:

1. a CPU preflight;
2. a 70-task GPU array (5 datasets × 2 CAFE variants × 7 penalties);
3. a strict configuration-selection finalizer.

Each selection task uses three paired runs, evaluates only validation edges,
uses K=10 without demographic metrics, and evaluates the complete 6 × 5
beta/mobility inference grid for its one training-time penalty. The finalizer
must see all 420 configurations per dataset (210 per CAFE variant), finite
values for all four objectives, three observations per metric, and the
corrected graph-semantics metadata before it writes:

- `selected_configs.csv`
- `selection_candidates.csv`
- `selected_configs.sha256`

The source manifest is created before submission and checked by every job.
Changing any covered Python or orchestration source after submission causes
pending tasks to fail instead of silently mixing implementations.

### 2. Inspect and approve the frozen choices

Wait for `_SELECTION_COMPLETE`, then inspect at least:

```bash
RUN_TAG=20260723_degree_eg_v1
column -s, -t < "results_egfix/${RUN_TAG}/selected_configs.csv"
less "results_egfix/${RUN_TAG}/selection_candidates.csv"
cat "results_egfix/${RUN_TAG}/job_ids.selection.tsv"
```

Confirm that there are exactly ten selected rows (CAFE-In and CAFE-Pre for
each of five datasets), that every selected value belongs to its declared
grid, and that the Pareto/ASF diagnostics are sensible. Do not proceed merely
because the Slurm jobs exited successfully.

### 3. Explicitly submit aggregate and stratified reruns

The expensive reruns are never launched by phase one. After inspecting the
selection:

```bash
bash jobs/egfix/submit_reruns.sh \
  20260723_degree_eg_v1 \
  --confirm-selected-configs
```

This submits:

- 50 aggregate tasks: five datasets × training seeds 42–51;
- 15 stratified tasks: five datasets × training seeds 42–44;
- one CPU finalizer depending on both complete arrays via `afterok`.

The runners read the frozen `selected_configs.csv`; no selected configuration
is hardcoded in these job files. Aggregate tasks evaluate RA, HL-GNN, Fusion,
CAFE-In, and CAFE-Pre on the fixed test split. Stratified tasks save the raw
source-level results and validated top-10 recommendation artifacts so that
graph metrics can be recomputed later without retraining.

### 4. Validate before touching the manuscript

After `_RERUNS_COMPLETE` appears:

1. Inspect `finalized/aggregate_metrics_long.csv` and
   `finalized/aggregate_metrics_summary.csv`.
2. Confirm ten seeds for every dataset/method pair and finite headline
   Precision@10, Degree EG, PageRank EG, and Closeness EG.
3. Inspect the merged stratified CSVs and PDFs for all five datasets.
4. Confirm three run-level values per method/bin, expected model order
   (RA, HL-GNN, Fusion, CAFE-In, CAFE-Pre), and run-level—not node-level—
   uncertainty.
5. Confirm recommendation artifacts exist for every dataset/seed and passed
   the distinct, in-range, non-self, non-training-neighbor checks.
6. Only then update Section 7.2, its figure/table references, and the visible
   Degree-EG rerun hold.

Nothing in this workflow automatically promotes an artifact into the paper.

## Required runner interface

The jobs use these corrected-protocol interfaces; the preflight test suite
covers the validation-only and graph-semantics execution paths before the
selection array is released:

- `experiments/run_cafe.py`: `--evaluation_split valid`,
  `--selection_mode True`, `--skip_final_comprehensive_eval True`,
  `--k_values 10`, `--use_demographics False`, deterministic-closeness
  options, and `--source_manifest`.
- `experiments/run_significance.py`: `--selected_configs`,
  `--penalty_reduction mean`, deterministic-closeness options,
  `--skip_final_comprehensive_eval True`, and `--source_manifest`.
- `analysis/run_stratified_cafe.py`: `--selected_configs`,
  `--save_recommendations True`, the same pinned protocol metadata, and
  `--source_manifest`.
- Finalizers:
  `analysis/egfix_finalize_selection.py` and
  `analysis/egfix_finalize_reruns.py`, both accepting `--campaign_root` and
  `--source_manifest`; the rerun finalizer also accepts
  `--selected_configs`.

If any interface is absent, treat that as a preflight failure. Do not remove a
flag to make a job start: doing so would weaken the audit trail or reintroduce
test-set leakage.

## Recovery and monitoring

Job IDs and dependencies are recorded in `job_ids.selection.tsv` and
`job_ids.reruns.tsv`. Monitor with `squeue -u "$USER"` and inspect the campaign
`logs/` directory.

The phase-two submission script writes an intent marker before its first
`sbatch`. If submission is interrupted, it refuses to run again because doing
so could duplicate already-submitted GPU work. Inspect the recorded IDs and
Slurm state, then submit only the missing stages manually with the same
`CAFE_REPO` and `EGFIX_RUN_TAG` exports.
