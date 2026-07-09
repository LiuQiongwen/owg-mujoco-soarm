#!/bin/bash
# Extends run_consensus_n10_matched.sh's consensus(n=10) design to
# TomatoSoupCan (same grid/checkpoint as ikmargin_TomatoSoupCan.jsonl from
# run_ikmargin_n10_tomatosoupcan.sh), so the two strategies are directly
# comparable at matched ensemble size for this 4th object. See that script's
# header for the ensemble-size-confound rationale; identical here, just one
# object instead of the original three.
#
# Motivation (2026-07-09): closes the "only one of the two named
# unreliable-across-seeds objects (Pear, TomatoSoupCan) has ikmargin-vs-
# consensus mitigation data" gap. Does not touch or re-run the existing
# Pear/MustardBottle/CrackerBox files.
set -u
cd /lena/projects/OWG-main/.claude/worktrees/fix-eval-seeding

CUPTI_SO=/home/lina/miniforge3/envs/tango/lib/python3.10/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12
NCCL_SO=/home/lina/miniforge3/envs/tango/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2
export LD_PRELOAD="$CUPTI_SO:$NCCL_SO"

OBJ="TomatoSoupCan"
ORIENT_SEEDS=(5 6 7 8 9)
BASES_N10=(1 11 21 31 41 51 61 71 81 91)
CONSENSUS_N=10
CKPT="grasp_6dof/models/cfm_allobj_ot.pt"

RESULTS_DIR="/lena/projects/OWG-main/paperA_data/phase1_matched_n10"
mkdir -p "$RESULTS_DIR"
raw="$RESULTS_DIR/consensus_n10_${OBJ}.jsonl"
> "$raw"

for oseed in "${ORIENT_SEEDS[@]}"; do
  obj_success=0
  for base in "${BASES_N10[@]}"; do
    OUTPUT=$(timeout 90 conda run -n tango python demo.py \
      --stage 4 --prompt "$OBJ" --seed "$oseed" --gen-seed "$base" --consensus-n "$CONSENSUS_N" \
      --once --verbose 0 --gate-delta 0.0 --mc-gate-delta 0.0 --cfm-ckpt "$CKPT" --no-semantic 2>&1)
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
      SUCCESS="timeout"
    elif echo "$OUTPUT" | grep -q "Done pick"; then
      SUCCESS="true"
      obj_success=$((obj_success+1))
    else
      SUCCESS="false"
    fi
    echo "{\"strategy\":\"consensus\",\"object\":\"$OBJ\",\"orient_seed\":$oseed,\"ensemble_base\":$base,\"success\":\"$SUCCESS\"}" >> "$raw"
  done
  echo "[progress-consensus-n10-$OBJ] orient_seed=$oseed: $obj_success/${#BASES_N10[@]}"
done
echo "[progress-consensus-n10-$OBJ] === DONE ==="
