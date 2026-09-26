#!/bin/bash
# Submit a reproduction job and its build/verify stage as one dependency chain.
#
#   reproduce/jobs/submit.sh t04_main_results [extra sbatch options]
#
# The run stage is submitted as the job's Slurm array; the build stage (CPU,
# STAGE=build) starts after every array task succeeded.  Both share one OUT,
# which is printed.  Run from the repository root.
set -euo pipefail
job=${1:?usage: submit.sh <job-name> [sbatch options]}; shift
file=reproduce/jobs/${job}.job
[[ -f "${file}" ]] || { echo "no such job: ${file}" >&2; exit 2; }
export OUT=${OUT:-${PWD}/results/reproduce/${job}_$(date +%Y%m%d_%H%M%S)}
mkdir -p logs "${OUT}"
if grep -q '^#SBATCH --array' "${file}"; then
    run_id=$(sbatch --parsable --export=ALL,OUT="${OUT}" "$@" "${file}")
    build_id=$(sbatch --parsable --dependency=afterok:${run_id%%;*} --gres=none --array=0 \
        --export=ALL,OUT="${OUT}",STAGE=build "${file}")
    echo "run: ${run_id}  build: ${build_id}  out: ${OUT}"
else
    echo "single-stage job: $(sbatch --parsable --export=ALL,OUT="${OUT}" "$@" "${file}")  out: ${OUT}"
fi
