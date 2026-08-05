# results-26-08-05 — corrected-EG table rebuild (partial, pokec_n pending)

Built 2026-08-05 from the post-EG-fix rerun tree. Every JSON-backed table was
aggregated with `CAFE_REQUIRE_EG_SEMANTICS=1`, so all pre-2026-07-23 result
files (no `eg_graph_semantics = simple_undirected_set_union` marker) were
excluded from extraction. Missing cells are experiments still running, not
omissions by the generators.

## Source data vintage (newest file per source)

| Source | Latest mtime | Status |
|---|---|---|
| results_repro/{ds}/baselines,lightgcn,ssnet,cross_link_bias,ltlp,tailsteak,fairadj,lba,moral | 2026-07-28/29 | 5/5 datasets post-fix |
| results_repro/{ds}/{cafe_p,cafe_in,cafe_pre,ci_in,ci_pre,mb_post,cafe_in_b,cafe_pre_b,fusion_only} | 2026-07-28..30 (pokec_n: fusion_only 2026-08-04, cafe_in_b partial in progress) | pokec_n mostly MISSING (see below) |
| results_repro/{ds}/significance/per_seed_metrics_*.csv | 2026-08-04/05 | 4/5 complete (10 seeds, `.complete.json` sidecar); pokec_n 6/10, still running |
| analysis/plots/stratified_cafe | 2026-07-29 | 5/5 (Tables 5, 6a, 18) |
| analysis/plots/coreness_by_group | 2026-07-29/30 | 5/5 (Tables 6b, 9) |
| analysis/plots/closeness_mechanism_paired | 2026-07-29 | 3/3 (Table 10) |
| results/{ds}/acceptance_eg | 2026-07-29/30 | 5/5 (Table 11) |
| results/{ds}/runtime | 2026-07-22..26 | 5/5 (Table 12, EG-insensitive) |
| dataset_table.* | copied from results_repro/tables (2026-07-29) | input data unchanged |

## Pinned configurations (stage_base/)

The rerun tree contains config sweeps for `fusion_only` (beta grid, all 5
datasets) and `cafe_pre` (alpha grid, oklahoma + unc). The stock generator
re-selects via Pareto/ASF and diverged from the published config for
CAFE-Pre/UNC (alpha 0.5 instead of the published 1.0). Per the table04 module
note ("selection must be PINNED to the published configurations"), the build
ran against `stage_base/`: symlinks to the rerun tree, except multi-config
JSONs replaced by filtered copies keeping only the published config
(`_framework.PUBLISHED`). Selected configs in `build_t04a.log` match
PUBLISHED/PUBLISHED_SINGLE for every available (method, dataset).

## pokec_n gaps (jobs 106561_3/106562_3/106563_3 still running)

- Tables 4/7: HL-GNN backbone, CAFE^-P, CAFE-In, CAFE-Pre, CAFE-In^-B,
  CAFE-Pre^-B, CI-In, CI-Pre render '--' / row omitted for pokec_n.
  Present for pokec_n: all attr.-dep. and tail/GNN baselines, HL-GNN+RA
  (fusion, 2026-08-04), MB-Post (carries the corrected marker from 2026-07-19).
- Table 8: pokec_n block has only FairAdj/LBA/MORAL/HL-GNN+RA (4 of 9 rows).
- Tables 16/9/17: pokec_n split-resampled columns are n=6 seeds (CSV live,
  still being written; symlinked, not copied).
- Table 17 fixed-split HL-GNN pokec_n cell uses the PRE-rerun baseline JSON
  (2026-07-19). DeltaSP/DeltaEO are unaffected by the EG fix, but it is the
  old training vintage — replace once t04_cafe pokec_n lands.
  (stage_appendix/ documents exactly which files fed this generator.)

## Generator issues hit during this build

- `analysis/generate_demographic_appendix_tables.py` patched (2026-08-05):
  missing fixed-split cells now render '--' instead of KeyError abort.
- Two path bugs in the table-spec build commands worked around by argument,
  not code: t10 needs `--plots_dir analysis/plots` (generator appends the
  subdir itself); t09/17 needs `--analysis_dir analysis` (same pattern).
- t16/t09/t17 verify cannot check EG semantics of the per-seed CSVs; the
  `.complete.json` sidecars confirm `simple_undirected_set_union` for the 4
  complete datasets. pokec_n has no sidecar yet (in progress).

## Headline changes vs the published tables

- Degree/PageRank EG levels moved materially under the corrected metric
  (e.g. oklahoma CAFE-In Degree EG 5.62 -> 10.28 vs the 2026-07-28 mixed
  table; Closeness EG unchanged, as expected).
- Table 8 Welch family: was 162/162 Holm-significant at 0.001 (old
  table5_significance.csv); now 104/117 at 0.05 (98/117 at 0.001), 13
  failures all on Oklahoma vs MORAL/FairAdj, incl. NEGATIVE Degree-EG diffs
  for HLGNN+RA, CAFE-In^-B, CAFE-In vs MORAL. The family is 117 (not 162)
  because pokec_n CAFE rows are missing.
- Table 16 pokec_n (n=6, preliminary): CAFE-In and CAFE-Pre LOSE Precision@10
  to the fusion control (Holm-sig.); CAFE-Pre also loses Degree EG to fusion.

## Build logs

`build_t*.log` capture every generator run; `verify_2026-08-05.log` is the
`python -m experiments.tables --stage verify --tables all` output from the
same morning.
