#!/bin/bash

# Phase 2 only.  This requires an explicit confirmation argument after the
# phase-one selection artifact has been inspected by a researcher.

set -euo pipefail
umask 027

usage() {
    cat <<'EOF'
Usage:
  bash jobs/egfix/submit_reruns.sh RUN_TAG --confirm-selected-configs

The command submits 50 aggregate tasks and 15 stratified tasks.  It is
intentionally separate from submit_selection.sh.
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
[[ -f "${REPO}/experiments/run_significance.py" &&
   -f "${REPO}/analysis/run_stratified_cafe.py" ]] || {
    echo "ERROR: ${REPO} is not a prepared CAFE repository." >&2
    exit 2
}

RUN_TAG="${1:-}"
CONFIRMATION="${2:-}"
if [[ -z "${RUN_TAG}" ||
      "${CONFIRMATION}" != "--confirm-selected-configs" ||
      $# != 2 ]]; then
    usage
    exit 2
fi
if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ||
      "${RUN_TAG}" == *".."* ||
      "$(basename -- "${RUN_TAG}")" != "${RUN_TAG}" ]]; then
    echo "ERROR: unsafe RUN_TAG: ${RUN_TAG}" >&2
    exit 2
fi
command -v sbatch >/dev/null || {
    echo "ERROR: sbatch is not available in this shell." >&2
    exit 2
}

CAMPAIGN="${REPO}/results_egfix/${RUN_TAG}"
SOURCE_MANIFEST="${CAMPAIGN}/source_manifest.json"
SELECTED_CONFIGS="${CAMPAIGN}/selected_configs.csv"
[[ -d "${CAMPAIGN}" &&
   -f "${CAMPAIGN}/_SELECTION_COMPLETE" &&
   -s "${SOURCE_MANIFEST}" &&
   -s "${SELECTED_CONFIGS}" &&
   -s "${CAMPAIGN}/selected_configs.sha256" ]] || {
    echo "ERROR: phase one is absent or incomplete for ${CAMPAIGN}." >&2
    exit 2
}

(
    cd "${REPO}"
    python3 -u analysis/egfix_preflight.py \
        --source_manifest "${SOURCE_MANIFEST}" \
        --verify-only
)
(cd "${CAMPAIGN}" && sha256sum --check --quiet selected_configs.sha256)

if [[ -e "${CAMPAIGN}/reruns_submission.intent" ||
      -e "${CAMPAIGN}/job_ids.reruns.tsv" ||
      -e "${CAMPAIGN}/aggregate" ||
      -e "${CAMPAIGN}/stratified" ]]; then
    echo "ERROR: phase two has already been attempted; refusing duplicate jobs." >&2
    echo "       Inspect ${CAMPAIGN}/job_ids.reruns.tsv and Slurm state." >&2
    exit 2
fi

mkdir "${CAMPAIGN}/aggregate"
mkdir "${CAMPAIGN}/stratified"
touch "${CAMPAIGN}/reruns_submission.intent"

JOB_IDS="${CAMPAIGN}/job_ids.reruns.tsv"
printf 'stage\tjob_id\tdependency\n' > "${JOB_IDS}"
EXPORTS="ALL,CAFE_REPO=${REPO},EGFIX_RUN_TAG=${RUN_TAG}"

AGGREGATE_RAW="$(
    sbatch --parsable \
        --job-name="egfix_agg_${RUN_TAG}" \
        --output="${CAMPAIGN}/logs/aggregate_%A_%a.out" \
        --export="${EXPORTS}" \
        "${SCRIPT_DIR}/aggregate.job"
)"
AGGREGATE_ID="${AGGREGATE_RAW%%;*}"
printf 'aggregate\t%s\t-\n' "${AGGREGATE_ID}" >> "${JOB_IDS}"

STRATIFIED_RAW="$(
    sbatch --parsable \
        --job-name="egfix_strat_${RUN_TAG}" \
        --output="${CAMPAIGN}/logs/stratified_%A_%a.out" \
        --export="${EXPORTS}" \
        "${SCRIPT_DIR}/stratified.job"
)"
STRATIFIED_ID="${STRATIFIED_RAW%%;*}"
printf 'stratified\t%s\t-\n' "${STRATIFIED_ID}" >> "${JOB_IDS}"

FINALIZE_RAW="$(
    sbatch --parsable \
        --job-name="egfix_done_${RUN_TAG}" \
        --output="${CAMPAIGN}/logs/finalize_reruns_%j.out" \
        --dependency="afterok:${AGGREGATE_ID}:${STRATIFIED_ID}" \
        --export="${EXPORTS}" \
        "${SCRIPT_DIR}/finalize_reruns.job"
)"
FINALIZE_ID="${FINALIZE_RAW%%;*}"
printf 'finalize_reruns\t%s\tafterok:%s:%s\n' \
    "${FINALIZE_ID}" "${AGGREGATE_ID}" "${STRATIFIED_ID}" >> "${JOB_IDS}"
touch "${CAMPAIGN}/_RERUNS_SUBMITTED"

echo "Phase 2 submitted for ${RUN_TAG}:"
echo "  aggregate array:    ${AGGREGATE_ID} (50 tasks, at most 10 concurrent)"
echo "  stratified array:   ${STRATIFIED_ID} (15 tasks, at most 5 concurrent)"
echo "  rerun finalizer:    ${FINALIZE_ID}"
echo "  campaign:           ${CAMPAIGN}"
