#!/bin/bash
# Build the Option-3 (ASF-re-selected) results tree.
#
# Only the CAFE arms differ between Option 1 and Option 3 -- the baselines, the
# HL-GNN backbone and the fusion sweep do not depend on which CAFE
# configuration was selected, so they are SHARED by symlink rather than
# retrained. That is what makes Option 3 cheap: 4 arms per dataset instead of a
# full campaign.
#
#   results_asf/<ds>/
#     baselines/ lightgcn/ ssnet/ cross_link_bias/ ltlp/ tailsteak/
#     fairadj/ lba/ moral/ fusion_only/      -> symlinks into results_repro
#     cafe_p/results_cafe_p_baseline_*.json  -> symlinked backbone file only
#     cafe_in/ cafe_pre/ cafe_in_b/ cafe_pre_b/  -> real dirs, written by
#                                                   t04_cafe_asf.job
#
# fusion_only is shared because the Option-1 job sweeps beta over
# {0.1,0.5,1.0,2.0,5.0} in one training pass and stores every beta; all the ASF
# betas (0.1/0.5/5.0) are already in that sweep.
#
# Safe to re-run: existing symlinks are refreshed, real directories untouched.
#
# Usage:  bash experiments/jobs/setup_asf_tree.sh

set -euo pipefail
REPO=/home/eerdem/workspace/CAFE
SRC=${REPO}/results_repro
DST=${REPO}/results_asf

SHARED=(baselines lightgcn ssnet cross_link_bias ltlp tailsteak fusion_only)
# FairAdj / LBA / MORAL need a sensitive attribute and exist only on these:
ATTR_ONLY=(fairadj lba moral)
OWN=(cafe_in cafe_pre cafe_in_b cafe_pre_b cafe_p)

missing=0
for ds in ego-facebook oklahoma unc pokec_n brightkite; do
  mkdir -p "${DST}/${ds}"
  echo "=== ${ds}"
  expected=("${SHARED[@]}")
  case ${ds} in
    oklahoma|unc|pokec_n) expected+=("${ATTR_ONLY[@]}") ;;
  esac
  for d in "${expected[@]}"; do
    if [ -d "${SRC}/${ds}/${d}" ]; then
      ln -sfn "${SRC}/${ds}/${d}" "${DST}/${ds}/${d}"
      printf "    shared  %-18s -> results_repro\n" "${d}"
    else
      printf "    ABSENT  %-18s (Option-1 run not finished yet)\n" "${d}"
      missing=$((missing+1))
    fi
  done
  # backbone only, not the whole cafe_p dir: Option 3 writes its own CAFE^-P there
  mkdir -p "${DST}/${ds}/cafe_p"
  shopt -s nullglob
  found=0
  for f in "${SRC}/${ds}/cafe_p"/results_cafe_p_baseline_"${ds}"_2026072[89]*.json; do
    ln -sfn "$f" "${DST}/${ds}/cafe_p/$(basename "$f")"
    printf "    shared  %-18s -> %s\n" "backbone" "$(basename "$f")"
    found=1
  done
  shopt -u nullglob
  if [ $found -eq 0 ]; then
    printf "    ABSENT  %-18s (corrected-metric backbone not produced yet)\n" "backbone"
    missing=$((missing+1))
  fi
  for d in "${OWN[@]}"; do mkdir -p "${DST}/${ds}/${d}"; done
done

mkdir -p "${DST}/tables"
echo
if [ ${missing} -gt 0 ]; then
  echo "${missing} shared artifact(s) still missing -- re-run this script once the"
  echo "Option-1 jobs finish, then rebuild the Option-3 tables."
else
  echo "All shared artifacts linked. results_asf/ is ready."
fi
