#!/bin/bash

# Resume the interrupted TODO1 campaign only.  This launcher validates and
# retains tasks 0..22, submits tasks 23..34, and finalizes five CAFE-In rows.
# It never submits aggregate or stratified phase-two work.

set -euo pipefail
umask 027

usage() {
    cat <<'EOF'
Usage:
  bash jobs/egfix_sum_recovery/submit_recovery.sh RUN_TAG

Expected interrupted campaign:
  20260724_degree_eg_sum_in_v1
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
RUN_TAG="${1:-}"
if [[ -z "${RUN_TAG}" || $# -ne 1 ]]; then
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
RECOVERY_MANIFEST="${CAMPAIGN}/recovery_manifest.json"
[[ -d "${CAMPAIGN}" &&
   -s "${SOURCE_MANIFEST}" &&
   -s "${CAMPAIGN}/hl_gnn.conda-explicit.txt" ]] || {
    echo "ERROR: interrupted campaign inputs are missing: ${CAMPAIGN}" >&2
    exit 2
}
[[ ! -e "${CAMPAIGN}/_CAFE_IN_SUM_SELECTION_COMPLETE" ]] || {
    echo "ERROR: TODO1 is already complete." >&2
    exit 2
}
for path in \
    "${RECOVERY_MANIFEST}" \
    "${CAMPAIGN}/job_ids.recovery.tsv" \
    "${CAMPAIGN}/_CAFE_IN_SUM_RECOVERY_SUBMISSION_INTENT"; do
    [[ ! -e "${path}" ]] || {
        echo "ERROR: recovery was already prepared or submitted: ${path}" >&2
        exit 2
    }
done

RESULT_COUNT="$(
    find "${CAMPAIGN}/selection" -type f -name results.json -print |
        wc -l | tr -d ' '
)"
SUCCESS_COUNT="$(
    find "${CAMPAIGN}/selection" -type f -name _SUCCESS -print |
        wc -l | tr -d ' '
)"
PARTIAL_COUNT="$(
    find "${CAMPAIGN}/selection" -type f -name '*.partial.json' -print |
        wc -l | tr -d ' '
)"
[[ "${RESULT_COUNT}" == "23" &&
   "${SUCCESS_COUNT}" == "16" &&
   "${PARTIAL_COUNT}" == "0" ]] || {
    echo "ERROR: interrupted task state differs from the reviewed recovery:" >&2
    echo "       results=${RESULT_COUNT}, successes=${SUCCESS_COUNT}, partials=${PARTIAL_COUNT}" >&2
    exit 2
}

cd "${REPO}"
python3 -u -m jobs.egfix_sum_recovery.provenance \
    --campaign-root "${CAMPAIGN}" \
    --check-only
python3 -u -m jobs.egfix_sum_recovery.provenance \
    --campaign-root "${CAMPAIGN}" \
    --manifest-path "${RECOVERY_MANIFEST}" \
    --create

{
    printf 'run_tag=%s\n' "${RUN_TAG}"
    printf 'created_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'original_source_manifest=%s\n' "${SOURCE_MANIFEST}"
    printf 'recovery_manifest=%s\n' "${RECOVERY_MANIFEST}"
    printf 'validated_existing_task_ids=0-22\n'
    printf 'adopt_task_ids=13,17-22\n'
    printf 'rerun_task_ids=23-34\n'
    printf 'phase_two_ready=false\n'
} > "${CAMPAIGN}/recovery.env"
touch "${CAMPAIGN}/_CAFE_IN_SUM_RECOVERY_SUBMISSION_INTENT"

JOB_IDS="${CAMPAIGN}/job_ids.recovery.tsv"
printf 'stage\tjob_id\tdependency\n' > "${JOB_IDS}"
EXPORTS="ALL,CAFE_REPO=${REPO},EGFIX_SUM_RUN_TAG=${RUN_TAG}"

PREFLIGHT_RAW="$(
    sbatch --parsable \
        --job-name="egfix_rec_pre_${RUN_TAG}" \
        --output="${CAMPAIGN}/logs/recovery_preflight_%j.out" \
        --export="${EXPORTS}" \
        "${SCRIPT_DIR}/preflight.job"
)"
PREFLIGHT_ID="${PREFLIGHT_RAW%%;*}"
printf 'recovery_preflight\t%s\t-\n' "${PREFLIGHT_ID}" >> "${JOB_IDS}"

SELECTION_RAW="$(
    sbatch --parsable \
        --job-name="egfix_rec_sel_${RUN_TAG}" \
        --output="${CAMPAIGN}/logs/recovery_selection_%A_%a.out" \
        --dependency="afterok:${PREFLIGHT_ID}" \
        --array="23-34%10" \
        --export="${EXPORTS}" \
        "${SCRIPT_DIR}/selection.job"
)"
SELECTION_ID="${SELECTION_RAW%%;*}"
printf 'recovery_selection\t%s\tafterok:%s\n' \
    "${SELECTION_ID}" "${PREFLIGHT_ID}" >> "${JOB_IDS}"

FINALIZE_RAW="$(
    sbatch --parsable \
        --job-name="egfix_rec_pick_${RUN_TAG}" \
        --output="${CAMPAIGN}/logs/recovery_finalize_%j.out" \
        --dependency="afterok:${SELECTION_ID}" \
        --export="${EXPORTS}" \
        "${SCRIPT_DIR}/finalize_selection.job"
)"
FINALIZE_ID="${FINALIZE_RAW%%;*}"
printf 'recovery_finalize\t%s\tafterok:%s\n' \
    "${FINALIZE_ID}" "${SELECTION_ID}" >> "${JOB_IDS}"

echo "TODO1 recovery submitted for ${RUN_TAG}:"
echo "  audit preflight: ${PREFLIGHT_ID}"
echo "  tasks 23-34:     ${SELECTION_ID}"
echo "  finalizer:       ${FINALIZE_ID}"
echo
echo "No TODO2 approval or TODO3 reruns were submitted."
