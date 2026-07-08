#!/bin/bash
# Ensemble-size-matched consensus re-run: --consensus-n 10 (same pool size as
# ikmargin_*.jsonl's --ikmargin-n 10), same object/orient_seed/base grid as
# phase1_v2/ikmargin_*.jsonl, so the two strategies are directly comparable
# without the ensemble-size confound documented in the original
# ikmargin_vs_consensus.csv.
#
# Environment note: the `tango` conda env's own torch(2.10.0)/cupti(12.8.90)
# are correct and matched (unchanged since 2026-06-29, same as what produced
# the original 2026-07-03 data). A stray user-level
# nvidia-cuda-cupti-cu12==12.1.105 (installed 2026-07-07 by an unrelated task)
# shadows tango's own cupti via Python's default user-site precedence, and a
# matching stray libnccl also shadows tango's own nccl. Fixed here via
# LD_PRELOAD of tango's own correct .so files -- this leaves user-site enabled
# (so mujoco/open3d/etc., which were never installed into tango itself and
# were always resolved via user-site, keep working exactly as they did on
# 2026-07-03), and does not modify, remove, or touch the stray package at all.
set -u
cd /lena/projects/OWG-main/.claude/worktrees/fix-eval-seeding

CUPTI_SO=/home/lina/miniforge3/envs/tango/lib/python3.10/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12
NCCL_SO=/home/lina/miniforge3/envs/tango/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2
export LD_PRELOAD="$CUPTI_SO:$NCCL_SO"

OBJECTS=("Pear" "MustardBottle" "CrackerBox")
ORIENT_SEEDS=(5 6 7 8 9)
BASES_N10=(1 11 21 31 41 51 61 71 81 91)
CONSENSUS_N=10
CKPT="grasp_6dof/models/cfm_allobj_ot.pt"

RESULTS_DIR="/lena/projects/OWG-main/paperA_data/phase1_matched_n10"
mkdir -p "$RESULTS_DIR"

for obj in "${OBJECTS[@]}"; do
  raw="$RESULTS_DIR/consensus_n10_${obj}.jsonl"
  > "$raw"
  for oseed in "${ORIENT_SEEDS[@]}"; do
    obj_success=0
    for base in "${BASES_N10[@]}"; do
      OUTPUT=$(timeout 90 conda run -n tango python demo.py \
        --stage 4 --prompt "$obj" --seed "$oseed" --gen-seed "$base" --consensus-n "$CONSENSUS_N" \
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
      echo "{\"strategy\":\"consensus\",\"object\":\"$obj\",\"orient_seed\":$oseed,\"ensemble_base\":$base,\"success\":\"$SUCCESS\"}" >> "$raw"
    done
    echo "[progress-consensus-n10-$obj] orient_seed=$oseed: $obj_success/${#BASES_N10[@]}"
  done
  echo "[progress-consensus-n10-$obj] === DONE ==="
done

echo "[phase1-matched-n10] ALL COMPLETE"
