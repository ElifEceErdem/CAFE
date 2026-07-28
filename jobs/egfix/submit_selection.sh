#!/bin/bash

# Phase 1 only: create a new immutable campaign, run the CPU preflight, launch
# the 70-task validation-only selection array, and finalize the ASF choices.
# This script deliberately does not submit aggregate or stratified reruns.

set -euo pipefail
umask 027

usage() {
    cat <<'EOF'
Usage:
  bash jobs/egfix/submit_selection.sh RUN_TAG

RUN_TAG must be a new filesystem-safe label, for example:
  20260723_degree_eg_v1
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
[[ -f "${REPO}/experiments/run_cafe.py" &&
   -f "${REPO}/analysis/egfix_preflight.py" ]] || {
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
    echo "ERROR: campaign already exists; refusing to mix or overwrite results:" >&2
    echo "       ${CAMPAIGN}" >&2
    exit 2
fi
mkdir "${CAMPAIGN}"
mkdir -p \
    "${CAMPAIGN}/logs" \
    "${CAMPAIGN}/selection"

SOURCE_MANIFEST="${CAMPAIGN}/source_manifest.json"
(
    cd "${REPO}"
    python3 -u analysis/egfix_preflight.py \
        --source_manifest "${SOURCE_MANIFEST}"
)
[[ -s "${SOURCE_MANIFEST}" ]] || {
    echo "ERROR: failed to create the immutable JSON source manifest." >&2
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
    printf 'graph_semantics=simple_undirected_set_union\n'
    printf 'selection_split=valid\n'
    printf 'headline_split=fixed_42_or_brightkite_temporal\n'
    printf 'selection_runs=3\n'
    printf 'aggregate_training_seeds=42-51\n'
    printf 'stratified_training_seeds=42-44\n'
    printf 'penalty_reduction=mean\n'
    printf 'closeness_seed=42\n'
    printf 'closeness_sample_size=0\n'
    printf 'closeness_epsilon=0.1\n'
    printf 'asf_weights=0.5,0.3,0.1,0.1\n'
    printf 'selection_beta_grid=0.0,0.1,0.5,1.0,2.0,5.0\n'
    printf 'selection_mobility_grid=0.0,0.1,0.3,0.5,1.0\n'
    printf 'selection_in_grid=0.0,0.01,0.05,0.1,0.3,0.5,1.0\n'
    printf 'selection_pre_grid=0.01,0.05,0.1,0.3,0.5,1.0,2.0\n'
} > "${CAMPAIGN}/campaign.env"

JOB_IDS="${CAMPAIGN}/job_ids.selection.tsv"
printf 'stage\tjob_id\tdependency\n' > "${JOB_IDS}"
EXPORTS="ALL,CAFE_REPO=${REPO},EGFIX_RUN_TAG=${RUN_TAG}"

PREFLIGHT_RAW="$(
    sbatch --parsable \
        --job-name="egfix_pre_${RUN_TAG}" \
        --output="${CAMPAIGN}/logs/preflight_%j.out" \
        --export="${EXPORTS}" \
        "${SCRIPT_DIR}/preflight.job"
)"
PREFLIGHT_ID="${PREFLIGHT_RAW%%;*}"
printf 'preflight\t%s\t-\n' "${PREFLIGHT_ID}" >> "${JOB_IDS}"

SELECTION_RAW="$(
    sbatch --parsable \
        --job-name="egfix_sel_${RUN_TAG}" \
        --output="${CAMPAIGN}/logs/selection_%A_%a.out" \
        --dependency="afterok:${PREFLIGHT_ID}" \
        --export="${EXPORTS}" \
        "${SCRIPT_DIR}/selection.job"
)"
SELECTION_ID="${SELECTION_RAW%%;*}"
printf 'selection\t%s\tafterok:%s\n' \
    "${SELECTION_ID}" "${PREFLIGHT_ID}" >> "${JOB_IDS}"

FINALIZE_RAW="$(
    sbatch --parsable \
        --job-name="egfix_pick_${RUN_TAG}" \
        --output="${CAMPAIGN}/logs/finalize_selection_%j.out" \
        --dependency="afterok:${SELECTION_ID}" \
        --export="${EXPORTS}" \
        "${SCRIPT_DIR}/finalize_selection.job"
)"
FINALIZE_ID="${FINALIZE_RAW%%;*}"
printf 'finalize_selection\t%s\tafterok:%s\n' \
    "${FINALIZE_ID}" "${SELECTION_ID}" >> "${JOB_IDS}"

echo "Phase 1 submitted for ${RUN_TAG}:"
echo "  preflight:          ${PREFLIGHT_ID}"
echo "  selection array:    ${SELECTION_ID} (70 tasks, at most 10 concurrent)"
echo "  selection finalizer:${FINALIZE_ID}"
echo "  campaign:           ${CAMPAIGN}"
echo
echo "No aggregate or stratified reruns were submitted."
echo "After the finalizer succeeds, inspect selected_configs.csv and then run:"
echo "  bash jobs/egfix/submit_reruns.sh ${RUN_TAG} --confirm-selected-configs"
