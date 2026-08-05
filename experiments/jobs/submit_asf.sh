#!/bin/bash
# Submit the Option-3 (ASF-re-selected) campaign and chain its table build.
#
#   cd experiments/jobs && bash submit_asf.sh
#
# Option 3 only needs the CAFE arms: baselines, backbone and the fusion sweep
# are shared with Option 1 by symlink (see setup_asf_tree.sh). If the Option-1
# jobs are still running, submit this anyway -- build_tables_asf.job refreshes
# the symlinks before it builds, so whatever has landed by then is picked up.

set -e
cd "$(dirname "$0")"
mkdir -p logs

bash setup_asf_tree.sh

jid=$(sbatch --parsable t04_cafe_asf.job)
echo "submitted t04_cafe_asf.job -> $jid"
bid=$(sbatch --parsable --dependency=afterany:${jid} build_tables_asf.job)
echo "submitted build_tables_asf.job -> $bid (afterany:${jid})"
echo ""
echo "monitor: squeue -u \$USER"
echo "tables:  ${PWD}/../../results_asf/tables"
