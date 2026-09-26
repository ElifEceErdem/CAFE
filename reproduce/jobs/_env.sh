#!/bin/bash
# Shared setup for the reproduction jobs.  Sourced, never executed.
#
# Submit every job from the repository root (Slurm runs a spool copy of the
# script, so its own location cannot be used to find the repository):
#
#     cd /path/to/CAFE && sbatch reproduce/jobs/t04_main_results.job
#
# Environment overrides (all optional):
#   CAFE_REPO   repository root              (default: $SLURM_SUBMIT_DIR, else $PWD)
#   CAFE_DATA   dataset root                 (default: $CAFE_REPO/data/SNAP)
#   CONDA_ENV   conda environment            (default: hl_gnn; see environment.yml)
#   OUT         output root of this campaign (default: results/reproduce/<job>)
#   TOLERANCE   verification tolerance, in printed units (default per job)
#   DRY_RUN=1   print the commands instead of running them
#
# Provides: REPO DATA OUT DS SPLIT EXTRA EXACT COMMON EG and the helpers
#   run_py      run a python command (honours DRY_RUN)
#   runner      run an experiments/ runner through experiments/run_with.py
#   build_and_verify ID...   index OUT, build the artifacts, compare with the paper

set -euo pipefail

REPO=${CAFE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
if [[ ! -f "${REPO}/reproduce/__main__.py" ]]; then
    echo "ERROR: ${REPO} is not the CAFE repository; submit from the repository root or set CAFE_REPO." >&2
    exit 2
fi
cd "${REPO}"

if [[ -z "${CAFE_NO_CONDA:-}" ]] && command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-hl_gnn}"
fi
# Cluster-specific CUDA driver location (DAS-5); harmless elsewhere.
if [[ -d /cm/local/apps/cuda-driver/libs/current/lib64 ]]; then
    export LD_LIBRARY_PATH=/cm/local/apps/cuda-driver/libs/current/lib64:${LD_LIBRARY_PATH:-}
fi
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg PYTHONPATH=${REPO}${PYTHONPATH:+:${PYTHONPATH}}
export MPLCONFIGDIR=${TMPDIR:-/tmp}/cafe-mpl-${SLURM_JOB_ID:-$$}
mkdir -p "${MPLCONFIGDIR}"

DATA=${CAFE_DATA:-${REPO}/data/SNAP}
JOB_NAME=${JOB_NAME:?set JOB_NAME before sourcing _env.sh}
OUT=${OUT:-${REPO}/results/reproduce/${JOB_NAME}}
mkdir -p "${OUT}"
STAGE=${STAGE:-run}

DATASETS=(ego-facebook oklahoma unc pokec_n brightkite)
DS=${DS:-${DATASETS[${SLURM_ARRAY_TASK_ID:-0}]}}
eval "$(python -m reproduce.config "${DS}")"   # IN_LF IN_BETA ... (published configuration)

# Brightkite always uses its proxy-temporal split.
SPLIT=""
[[ "${DS}" == "brightkite" ]] && SPLIT="--split_file ${DATA}/brightkite/temporal_split.pt"
# Reduced training-time validation on the two large graphs (run_cafe.py only).
EXTRA="${SPLIT}"
if [[ "${DS}" == "pokec_n" || "${DS}" == "brightkite" ]]; then
    EXTRA="--eval_skip_test True --eval_max_nodes 5000 --eval_steps 25 --patience 6 ${SPLIT}"
fi
# Exact generalized Closeness for the two large graphs (Tables 4, 7, 11, 15).
EXACT=""
[[ "${DS}" == "pokec_n" || "${DS}" == "brightkite" ]] && EXACT="--exact-closeness"

COMMON="--data_name ${DS} --data_path ${DATA} --encoder HLGNN --predictor MLP \
  --epochs 500 --eval_steps 10 --patience 50 --batch_size 16384 --lr 0.001"
EG="--use_equality_gain True --use_centrality_gain True --centrality_types pagerank,closeness"

run_py () {
    echo "+ python $*"
    [[ "${DRY_RUN:-0}" == "1" ]] || python -u "$@"
}

# runner <record-dir> <runner.py> [args...]  -- standard runner, paper protocol options
runner () {
    local record=$1; shift
    mkdir -p "${record}"
    run_py experiments/run_with.py ${EXACT} --record-dir "${record}" ${RUN_WITH_OPTS:-} "$@"
}

# build_and_verify ID... -- index the campaign outputs, build, compare with the paper
build_and_verify () {
    run_py -m reproduce.tools.index_runs "${OUT}"
    CAFE_EVIDENCE="${OUT}" CAFE_SOURCES="${OUT}/sources" CAFE_BUILD="${OUT}/build" \
        run_py -m reproduce check --tolerance "${TOLERANCE:-0}" "$@"
}

echo "== ${JOB_NAME} stage=${STAGE} dataset=${DS} out=${OUT}"
