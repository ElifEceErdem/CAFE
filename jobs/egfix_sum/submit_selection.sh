#!/bin/bash

# Submit only TODO1: the isolated CAFE-In/sum validation selection.
# No aggregate or stratified test reruns are launched by this script.

set -euo pipefail
umask 027

usage() {
    cat <<'EOF'
Usage:
  bash jobs/egfix_sum/submit_selection.sh RUN_TAG

RUN_TAG must be new and filesystem-safe, for example:
  20260724_degree_eg_sum_in_v1
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
[[ -f "${REPO}/experiments/run_cafe.py" &&
   -f "${REPO}/jobs/egfix_sum/provenance.py" ]] || {
    echo "ERROR: ${REPO} is not a prepared CAFE repository." >&2
    exit 2
}

RUN_TAG="${1:-}"
if [[ -z "${RUN_TAG}" ]]; then
    usage
    exit 2
fi
if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ||
      "${RUN_TAG}" == *".."* ||
      "$(basename -- "${RUN_TAG}")" != "${RUN_TAG}" ]]; then
    echo "ERROR: unsafe RUN_TAG: ${RUN_TAG}" >&2
    usage
    exit 2
fi
if (( $# != 1 )); then
    usage
    exit 2
fi
command -v sbatch >/dev/null || {
    echo "ERROR: sbatch is not available in this shell." >&2
    exit 2
}

RESULTS_BASE="${REPO}/results_egfix"
CAMPAIGN="${RESULTS_BASE}/${RUN_TAG}"
mkdir -p "${RESULTS_BASE}"
if [[ -e "${CAMPAIGN}" ]]; then
    echo "ERROR: campaign already exists; refusing to mix results:" >&2
    echo "       ${CAMPAIGN}" >&2
    exit 2
fi
mkdir "${CAMPAIGN}"
mkdir -p "${CAMPAIGN}/logs" "${CAMPAIGN}/selection"

SOURCE_MANIFEST="${CAMPAIGN}/source_manifest.json"
(
    cd "${REPO}"
    python3 -u -m jobs.egfix_sum.provenance \
        --source_manifest "${SOURCE_MANIFEST}"
)
[[ -s "${SOURCE_MANIFEST}" ]] || {
    echo "ERROR: failed to create the immutable source manifest." >&2
    exit 1
}

GIT_COMMIT="$(git -C "${REPO}" rev-parse HEAD 2>/dev/null || printf 'unavailable')"
GIT_DIRTY_COUNT="$(
    git -C "${REPO}" status --porcelain --untracked-files=all 2>/dev/null |
        wc -l | tr -d ' '
)"
{
    printf 'run_tag=%s\n' "${RUN_TAG}"
    printf 'created_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'repository=%s\n' "${REPO}"
    printf 'git_commit=%s\n' "${GIT_COMMIT}"
    printf 'git_dirty_entries=%s\n' "${GIT_DIRTY_COUNT}"
    printf 'profile=cafe_in_sum_v1\n'
    printf 'todo_stage=TODO1_validation_selection\n'
    printf 'selection_variant=cafe_in\n'
    printf 'selection_tasks=35\n'
    printf 'selection_split=valid\n'
    printf 'selection_runs=3\n'
    printf 'model_seeds=42,43,44\n'
    printf 'graph_semantics=simple_undirected_set_union\n'
    printf 'penalty_reduction=sum\n'
    printf 'loss_func=AUC\n'
    printf 'num_neg=1\n'
    printf 'batch_size=16384\n'
    printf 'closeness_seed=42\n'
    printf 'closeness_sample_size=0\n'
    printf 'closeness_epsilon=0.1\n'
    printf 'asf_weights=0.5,0.3,0.1,0.1\n'
    printf 'selection_beta_grid=0.0,0.1,0.5,1.0,2.0,5.0\n'
    printf 'selection_mobility_grid=0.0,0.1,0.3,0.5,1.0\n'
    printf 'selection_in_grid=0.0,0.01,0.05,0.1,0.3,0.5,1.0\n'
    printf 'phase_two_ready=false\n'
    printf 'runtime_environment_snapshot=hl_gnn.conda-explicit.txt\n'
} > "${CAMPAIGN}/campaign.env"

JOB_IDS="${CAMPAIGN}/job_ids.selection.tsv"
printf 'stage\tjob_id\tdependency\n' > "${JOB_IDS}"
EXPORTS="ALL,CAFE_REPO=${REPO},EGFIX_SUM_RUN_TAG=${RUN_TAG}"

PREFLIGHT_RAW="$(
    sbatch --parsable \
        --job-name="egfix_sum_pre_${RUN_TAG}" \
        --output="${CAMPAIGN}/logs/preflight_%j.out" \
        --export="${EXPORTS}" \
        "${SCRIPT_DIR}/preflight.job"
)"
PREFLIGHT_ID="${PREFLIGHT_RAW%%;*}"
printf 'preflight\t%s\t-\n' "${PREFLIGHT_ID}" >> "${JOB_IDS}"

SELECTION_RAW="$(
    sbatch --parsable \
        --job-name="egfix_sum_sel_${RUN_TAG}" \
        --output="${CAMPAIGN}/logs/selection_%A_%a.out" \
        --dependency="afterok:${PREFLIGHT_ID}" \
        --array="0-34%10" \
        --export="${EXPORTS}" \
        "${SCRIPT_DIR}/selection.job"
)"
SELECTION_ID="${SELECTION_RAW%%;*}"
printf 'selection\t%s\tafterok:%s\n' \
    "${SELECTION_ID}" "${PREFLIGHT_ID}" >> "${JOB_IDS}"

FINALIZE_RAW="$(
    sbatch --parsable \
        --job-name="egfix_sum_pick_${RUN_TAG}" \
        --output="${CAMPAIGN}/logs/finalize_selection_%j.out" \
        --dependency="afterok:${SELECTION_ID}" \
        --export="${EXPORTS}" \
        "${SCRIPT_DIR}/finalize_selection.job"
)"
FINALIZE_ID="${FINALIZE_RAW%%;*}"
printf 'finalize_selection\t%s\tafterok:%s\n' \
    "${FINALIZE_ID}" "${SELECTION_ID}" >> "${JOB_IDS}"

echo "CAFE-In/sum TODO1 submitted for ${RUN_TAG}:"
echo "  preflight:           ${PREFLIGHT_ID}"
echo "  selection array:     ${SELECTION_ID} (35 tasks; max 10 concurrent)"
echo "  selection finalizer: ${FINALIZE_ID}"
echo "  campaign:            ${CAMPAIGN}"
echo
echo "No TODO2, aggregate, or stratified reruns were submitted."
