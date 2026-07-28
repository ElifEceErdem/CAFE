#!/bin/bash
# Submit the full table-indexed suite and chain the aggregation job after it.
#
#   cd experiments/jobs && bash submit_all.sh
#
# Selective submission is usually what you want, e.g.
#   sbatch t04_cafe.job                     # all five datasets
#   sbatch --array=1 t07_ablation.job       # oklahoma only
#   sbatch --array=0,4 t11_exposure.job     # the two datasets Table 6b lacks
#
# Array index -> dataset: 0=ego-facebook 1=oklahoma 2=unc 3=pokec_n 4=brightkite
# (t10_closeness.job is 0-2 only: ego-facebook, oklahoma, unc.)

set -e
cd "$(dirname "$0")"
mkdir -p logs

JOBS=(
  t04_cafe.job          # Tables 4, 7, 8 -- CAFE side
  t04_baselines.job     # Table 4 -- every baseline row
  t07_ablation.job      # Table 7 -- CI-In, CI-Pre, MB-Post
  t05_stratified.job    # Tables 5, 6a, 18, Figure 4
  t11_exposure.job      # Tables 11, 6b
  t16_significance.job  # Tables 16, 17
  t10_closeness.job     # Table 10
  t12_runtime.job       # Table 12
  cpu_analyses.job      # Tables 2, 9
  fig01_diagnosis.job   # Figure 1 (UNC)
)

DEPS=""
for j in "${JOBS[@]}"; do
  jid=$(sbatch --parsable "$j")
  echo "submitted $j -> $jid"
  DEPS="${DEPS}:${jid}"
done

# afterany, not afterok: one failed array task should not block aggregation,
# and the verify stage reports exactly which tables came up short.
tid=$(sbatch --parsable --dependency=afterany${DEPS} build_tables.job)
echo "submitted build_tables.job -> $tid"
echo ""
echo "monitor:  squeue -u \$USER"
echo "tables:   ${PWD}/../../results_repro/tables"
