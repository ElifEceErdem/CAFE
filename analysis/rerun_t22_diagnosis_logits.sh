#!/bin/bash
# Rerun ONLY the conditional-logit step of t22 (diagnosis_conditional_logit.py)
# for datasets whose GPU ranking artifacts already exist but whose logit step
# failed under the old summed-score convergence gate.  No model is retrained.
#
# Safe to rerun: completed shards (_SUCCESS) are skipped, and datasets whose
# rankings are not on disk yet (e.g. pokec_n while t22 task 3 is still
# training) are skipped with a message.  Once all five shards are complete
# the strict finalizer writes the campaign table.
set -eo pipefail
if [[ -f "${HOME}/.bashrc" ]]; then
    source "${HOME}/.bashrc"
fi
conda activate hl_gnn
set -u

REPO=/home/eerdem/workspace/CAFE
CAMPAIGN=${REPO}/results_rq1_controls/diagnosis_conditional_v1
cd "${REPO}"

for DATASET in ego-facebook oklahoma unc pokec_n brightkite; do
    SHARD=${CAMPAIGN}/shards/${DATASET}
    RAW_CSV=${SHARD}/precision_by_coreness_${DATASET}.csv
    METADATA=${SHARD}/metadata_precision_by_coreness_${DATASET}.json
    if [[ -f "${SHARD}/_SUCCESS" ]]; then
        echo "SKIP ${DATASET}: shard already complete"
        continue
    fi
    if [[ ! -f "${RAW_CSV}" || ! -f "${METADATA}" ]]; then
        echo "SKIP ${DATASET}: ranking artifacts not ready yet (${RAW_CSV})"
        continue
    fi
    echo "RUN ${DATASET}: conditional logit"
    python -u analysis/diagnosis_conditional_logit.py run \
        --dataset "${DATASET}" \
        --input-csv "${RAW_CSV}" \
        --input-metadata "${METADATA}" \
        --output-dir "${SHARD}"
done

python -u analysis/diagnosis_conditional_logit.py finalize \
    --campaign-dir "${CAMPAIGN}" --if-complete

echo "Done. Rerun this script after remaining t22 GPU tasks finish."
