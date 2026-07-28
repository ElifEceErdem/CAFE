#!/bin/bash

# Submit only TODO3: one CPU preflight, two independent GPU arrays, and one
# strict CPU finalizer.  The two upstream selections are never modified.

set -euo pipefail
umask 027

usage() {
    cat <<'EOF'
Usage:
  bash jobs/egfix_sum_todo3/submit_reruns.sh \
    SUM_CAFE_IN_RUN_TAG CAFE_PRE_RUN_TAG --confirm-selected-configs

This creates SUM_CAFE_IN_RUN_TAG/todo3 and submits:
  preflight -> {50 aggregate tasks, 15 stratified tasks} -> finalizer
EOF
}

validate_run_tag() {
    local value="$1"
    [[ "${value}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] &&
        [[ "${value}" != *".."* ]] &&
        [[ "$(basename -- "${value}")" == "${value}" ]]
}

validate_pre_submission_campaign() {
    local root="$1" entry name nested
    [[ -d "${root}" && ! -L "${root}" ]] || return 1
    [[ ! -e "${root}/submission.intent" &&
       ! -e "${root}/job_ids.todo3.tsv" &&
       ! -e "${root}/_TODO3_SUBMITTED" ]] || return 1
    nested="$(find "${root}" -type l -print -quit)"
    [[ -z "${nested}" ]] || return 1
    while IFS= read -r entry; do
        name="$(basename -- "${entry}")"
        case "${name}" in
            source_manifest.json|selected_configs.csv|\
            selected_configs.sha256|selected_configs_lineage.json|\
            campaign.env)
                [[ -f "${entry}" ]] || return 1
                ;;
            logs|aggregate|stratified)
                [[ -d "${entry}" ]] || return 1
                nested="$(find "${entry}" -mindepth 1 -print -quit)"
                [[ -z "${nested}" ]] || return 1
                ;;
            *)
                return 1
                ;;
        esac
    done < <(find "${root}" -mindepth 1 -maxdepth 1 -print)
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
[[ -f "${REPO}/experiments/run_significance.py" &&
   -f "${REPO}/analysis/run_stratified_cafe.py" &&
   -f "${SCRIPT_DIR}/provenance.py" ]] || {
    echo "ERROR: ${REPO} is not a prepared CAFE repository." >&2
    exit 2
}

SUM_RUN_TAG="${1:-}"
PRE_RUN_TAG="${2:-}"
CONFIRMATION="${3:-}"
if [[ $# != 3 ||
      "${CONFIRMATION}" != "--confirm-selected-configs" ]]; then
    usage
    exit 2
fi
if ! validate_run_tag "${SUM_RUN_TAG}" ||
   ! validate_run_tag "${PRE_RUN_TAG}"; then
    echo "ERROR: unsafe upstream run tag." >&2
    usage
    exit 2
fi
if [[ "${SUM_RUN_TAG}" == "${PRE_RUN_TAG}" ]]; then
    echo "ERROR: CAFE-In and CAFE-Pre selections must be distinct." >&2
    exit 2
fi
command -v sbatch >/dev/null || {
    echo "ERROR: sbatch is not available in this shell." >&2
    exit 2
}

SUM_CAMPAIGN="${REPO}/results_egfix/${SUM_RUN_TAG}"
PRE_CAMPAIGN="${REPO}/results_egfix/${PRE_RUN_TAG}"
CAMPAIGN="${SUM_CAMPAIGN}/todo3"
[[ -f "${SUM_CAMPAIGN}/_CAFE_IN_SUM_SELECTION_COMPLETE" &&
   -f "${SUM_CAMPAIGN}/cafe_in_sum_recovery_lineage.json" &&
   -f "${PRE_CAMPAIGN}/_SELECTION_COMPLETE" &&
   -s "${PRE_CAMPAIGN}/selected_configs.csv" ]] || {
    echo "ERROR: one or both frozen selection campaigns are incomplete." >&2
    exit 2
}
if [[ -e "${CAMPAIGN}" ]]; then
    if ! validate_pre_submission_campaign "${CAMPAIGN}"; then
        echo "ERROR: TODO3 already exists and is not a safely resumable" \
             " pre-submission preparation: ${CAMPAIGN}" >&2
        exit 2
    fi
    echo "Resuming verified TODO3 preparation before submission intent."
else
    mkdir "${CAMPAIGN}"
fi

mkdir -p "${CAMPAIGN}/logs"
SOURCE_MANIFEST="${CAMPAIGN}/source_manifest.json"

cd "${REPO}"
python3 -u -m jobs.egfix_sum_todo3.provenance \
    --source-manifest "${SOURCE_MANIFEST}" \
    --sum-campaign "${SUM_CAMPAIGN}" \
    --pre-campaign "${PRE_CAMPAIGN}" \
    --repo-root "${REPO}" \
    --create
python3 -u -m jobs.egfix_sum_todo3.compose_selected_configs \
    --output-dir "${CAMPAIGN}" \
    --source-manifest "${SOURCE_MANIFEST}" \
    --sum-campaign "${SUM_CAMPAIGN}" \
    --pre-campaign "${PRE_CAMPAIGN}" \
    --repo-root "${REPO}" \
    --create

{
    printf 'profile=rq3_degree_eg_sum_todo3_v1\n'
    printf 'sum_run_tag=%s\n' "${SUM_RUN_TAG}"
    printf 'pre_run_tag=%s\n' "${PRE_RUN_TAG}"
    printf 'todo3_root=%s\n' "${CAMPAIGN}"
    printf 'penalty_reduction=sum\n'
    printf 'loss_func=AUC\n'
    printf 'num_neg=1\n'
    printf 'batch_size=16384\n'
    printf 'aggregate_tasks=50\n'
    printf 'stratified_tasks=15\n'
    printf 'manuscript_writes=false\n'
} > "${CAMPAIGN}/campaign.env"

mkdir -p "${CAMPAIGN}/aggregate"
mkdir -p "${CAMPAIGN}/stratified"
touch "${CAMPAIGN}/submission.intent"
JOB_IDS="${CAMPAIGN}/job_ids.todo3.tsv"
printf 'stage\tjob_id\tdependency\n' > "${JOB_IDS}"
EXPORTS="ALL,CAFE_REPO=${REPO},EGFIX_SUM_RUN_TAG=${SUM_RUN_TAG},EGFIX_PRE_RUN_TAG=${PRE_RUN_TAG}"

PREFLIGHT_RAW="$(
    sbatch --parsable \
        --job-name="egfix_t3_pre_${SUM_RUN_TAG}" \
        --output="${CAMPAIGN}/logs/preflight_%j.out" \
        --export="${EXPORTS}" \
        "${SCRIPT_DIR}/preflight.job"
)"
PREFLIGHT_ID="${PREFLIGHT_RAW%%;*}"
printf 'preflight\t%s\t-\n' "${PREFLIGHT_ID}" >> "${JOB_IDS}"

AGGREGATE_RAW="$(
    sbatch --parsable \
        --job-name="egfix_t3_agg_${SUM_RUN_TAG}" \
        --output="${CAMPAIGN}/logs/aggregate_%A_%a.out" \
        --dependency="afterok:${PREFLIGHT_ID}" \
        --export="${EXPORTS}" \
        "${SCRIPT_DIR}/aggregate.job"
)"
AGGREGATE_ID="${AGGREGATE_RAW%%;*}"
printf 'aggregate\t%s\tafterok:%s\n' \
    "${AGGREGATE_ID}" "${PREFLIGHT_ID}" >> "${JOB_IDS}"

STRATIFIED_RAW="$(
    sbatch --parsable \
        --job-name="egfix_t3_strat_${SUM_RUN_TAG}" \
        --output="${CAMPAIGN}/logs/stratified_%A_%a.out" \
        --dependency="afterok:${PREFLIGHT_ID}" \
        --export="${EXPORTS}" \
        "${SCRIPT_DIR}/stratified.job"
)"
STRATIFIED_ID="${STRATIFIED_RAW%%;*}"
printf 'stratified\t%s\tafterok:%s\n' \
    "${STRATIFIED_ID}" "${PREFLIGHT_ID}" >> "${JOB_IDS}"

FINALIZE_RAW="$(
    sbatch --parsable \
        --job-name="egfix_t3_done_${SUM_RUN_TAG}" \
        --output="${CAMPAIGN}/logs/finalize_%j.out" \
        --dependency="afterok:${AGGREGATE_ID}:${STRATIFIED_ID}" \
        --export="${EXPORTS}" \
        "${SCRIPT_DIR}/finalize_reruns.job"
)"
FINALIZE_ID="${FINALIZE_RAW%%;*}"
printf 'finalize\t%s\tafterok:%s:%s\n' \
    "${FINALIZE_ID}" "${AGGREGATE_ID}" "${STRATIFIED_ID}" >> "${JOB_IDS}"
touch "${CAMPAIGN}/_TODO3_SUBMITTED"

echo "TODO3 submitted:"
echo "  preflight:       ${PREFLIGHT_ID}"
echo "  aggregate array: ${AGGREGATE_ID} (50 tasks; max 10)"
echo "  stratified array:${STRATIFIED_ID} (15 tasks; max 5)"
echo "  finalizer:       ${FINALIZE_ID}"
echo "  campaign:        ${CAMPAIGN}"
