#!/bin/bash
# Shared setup for the table-indexed job suite.  Source it, do not run it.
#
#   source "$(dirname "$0")/_common.sh"
#
# Provides: REPO, DATA, DS (from $SLURM_ARRAY_TASK_ID), TS, and the helper
# `extra_flags` holding the reduced validation protocol for the large graphs.

source ~/.bashrc
conda activate hl_gnn
# DAS-5: make libcuda.so.1 visible to torch
export LD_LIBRARY_PATH=/cm/local/apps/cuda-driver/libs/current/lib64:${LD_LIBRARY_PATH}
set -euo pipefail

REPO=/home/eerdem/workspace/CAFE
DATA=${REPO}/data/SNAP
TS=$(date +%Y%m%d_%H%M%S)

# The runners are invoked by bare filename from ${REPO}/experiments (or
# ${REPO}/analysis), so a wrong REPO surfaces as "can't open file" several
# minutes into the job rather than here. Check it up front.
for d in "${REPO}/experiments" "${REPO}/analysis" "${DATA}"; do
  if [ ! -d "$d" ]; then
    echo "FATAL: expected directory does not exist: $d" >&2
    exit 1
  fi
done

DATASETS=(ego-facebook oklahoma unc pokec_n brightkite)
DS=${DATASETS[${SLURM_ARRAY_TASK_ID:-0}]}

# Published configurations (analysis/best_configs.csv; mirrored in
# experiments/tables/_framework.py -- change both together).
case ${DS} in
#
# NOTE: CAFE-In^-B and CAFE-Pre^-B are separately selected configurations, NOT
# the full pipeline's config with the boost switched off. Deriving them from
# IN_*/PRE_* silently runs the wrong arm.
  ego-facebook) IN_LF=0.3;  IN_BETA=1.0; IN_LAM=0.5
                PRE_A=0.05; PRE_BETA=0.5; PRE_LAM=0.3; FUSION_BETA=1.0
                INB_LF=0.3; INB_BETA=1.0; PREB_A=1.0; PREB_BETA=0.1
                CI_IN=0.05; CI_PRE=1.0;  MB_POST=0.1 ;;
  oklahoma)     IN_LF=0.3;  IN_BETA=5.0; IN_LAM=0.5
                PRE_A=1.0;  PRE_BETA=0.5; PRE_LAM=0.5; FUSION_BETA=5.0
                INB_LF=0.1; INB_BETA=5.0; PREB_A=0.5; PREB_BETA=5.0
                CI_IN=1.0;  CI_PRE=1.0;  MB_POST=0.3 ;;
  unc)          IN_LF=1.0;  IN_BETA=5.0; IN_LAM=0.5
                PRE_A=1.0;  PRE_BETA=1.0; PRE_LAM=0.5; FUSION_BETA=5.0
                INB_LF=0.5; INB_BETA=5.0; PREB_A=0.3; PREB_BETA=5.0
                CI_IN=0.05; CI_PRE=0.5;  MB_POST=1.0 ;;
  pokec_n)      IN_LF=0.05; IN_BETA=1.0; IN_LAM=0.1
                PRE_A=0.05; PRE_BETA=2.0; PRE_LAM=0.5; FUSION_BETA=5.0
                INB_LF=0.1; INB_BETA=0.5; PREB_A=0.3; PREB_BETA=2.0
                CI_IN=0.1;  CI_PRE=0.3;  MB_POST=0.1 ;;
  brightkite)   IN_LF=0.05; IN_BETA=5.0; IN_LAM=0.1
                PRE_A=0.3;  PRE_BETA=2.0; PRE_LAM=0.5; FUSION_BETA=5.0
                INB_LF=0.1; INB_BETA=5.0; PREB_A=0.3; PREB_BETA=2.0
                CI_IN=0.1;  CI_PRE=0.1;  MB_POST=0.1 ;;
  *) echo "unknown dataset '${DS}'"; exit 1 ;;
esac

# Brightkite always uses its fixed proxy-temporal split rather than a random one.
SPLIT=""
if [ "${DS}" = "brightkite" ]; then
  SPLIT="--split_file ${DATA}/brightkite/temporal_split.pt"
fi

# Reduced training-time validation on the two large graphs (selection signal
# only; final reported metrics are unaffected). Honoured by run_cafe.py and
# run_significance.py; run_cafe_p.py, run_coreness_inverse.py,
# run_mobility_boost.py and run_baselines.py do NOT take these flags -- pass
# ${SPLIT} to those instead of ${EXTRA}.
EXTRA="${SPLIT}"
if [ "${DS}" = "pokec_n" ] || [ "${DS}" = "brightkite" ]; then
  EXTRA="--eval_skip_test True --eval_max_nodes 5000 --eval_steps 25 --patience 6 ${SPLIT}"
fi

COMMON="--data_name ${DS} --data_path ${DATA} \
  --encoder HLGNN --predictor MLP \
  --epochs 500 --eval_steps 10 --patience 50 \
  --batch_size 16384 --lr 0.001"

EG="--use_equality_gain True --use_centrality_gain True \
  --centrality_types pagerank,closeness"

echo "=============================================================="
echo "dataset : ${DS}"
echo "CAFE-In : lambda_fair=${IN_LF} beta=${IN_BETA} lambda_mob=${IN_LAM}"
echo "CAFE-Pre: alpha=${PRE_A} beta=${PRE_BETA} lambda_mob=${PRE_LAM}"
echo "fusion  : beta=${FUSION_BETA}"
echo "=============================================================="
