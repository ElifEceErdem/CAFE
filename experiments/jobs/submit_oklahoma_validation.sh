#!/bin/bash
# Submit the Oklahoma-only checks for Tables 4, 7, 8 and 11 plus Figures 3/4.
# Every generated artifact is isolated below results/validation so certified
# legacy outputs and the current paper artifacts cannot be overwritten.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

REPO=/home/eerdem/workspace/CAFE
CAMPAIGN_ID=${1:-$(date +%Y%m%d_%H%M%S)}
CAMPAIGN_ROOT=${REPO}/results/validation/${CAMPAIGN_ID}
TABLE_RESULTS=${CAMPAIGN_ROOT}/table_runs
T11_RESULTS=${CAMPAIGN_ROOT}/table11
T11_EXPOSURE=${CAMPAIGN_ROOT}/table11/exposure
FIG3_OUTPUT=${CAMPAIGN_ROOT}/figure3
FIG4_OUTPUT=${CAMPAIGN_ROOT}/figure4

mkdir -p "${TABLE_RESULTS}" "${T11_RESULTS}" "${T11_EXPOSURE}" \
    "${FIG3_OUTPUT}" "${FIG4_OUTPUT}"

echo "campaign: ${CAMPAIGN_ID}"
echo "outputs : ${CAMPAIGN_ROOT}"

jid_t04_cafe=$(sbatch --parsable --array=1 \
    --export=ALL,RESULTS_ROOT="${TABLE_RESULTS}" t04_cafe.job)
echo "submitted t04_cafe.job       -> ${jid_t04_cafe}"

jid_t04_base=$(sbatch --parsable --array=1 \
    --export=ALL,RESULTS_ROOT="${TABLE_RESULTS}" t04_baselines.job)
echo "submitted t04_baselines.job  -> ${jid_t04_base}"

jid_t07=$(sbatch --parsable --array=1 \
    --export=ALL,RESULTS_ROOT="${TABLE_RESULTS}" t07_ablation.job)
echo "submitted t07_ablation.job   -> ${jid_t07}"

jid_t11=$(sbatch --parsable --array=1 \
    --export=ALL,T11_RESULTS_ROOT="${T11_RESULTS}",T11_EXPOSURE_DIR="${T11_EXPOSURE}" \
    t11_exposure.job)
echo "submitted t11_exposure.job   -> ${jid_t11}"

jid_fig4=$(sbatch --parsable --array=1 \
    --export=ALL,STRATIFIED_OUTPUT_DIR="${FIG4_OUTPUT}" t05_stratified.job)
echo "submitted t05_stratified.job -> ${jid_fig4}"

jid_fig3=$(sbatch --parsable \
    --export=ALL,FIG3_OUTPUT_DIR="${FIG3_OUTPUT}" fig03_pareto.job)
echo "submitted fig03_pareto.job   -> ${jid_fig3}"

echo "monitor : squeue -j ${jid_t04_cafe},${jid_t04_base},${jid_t07},${jid_t11},${jid_fig4},${jid_fig3}"
