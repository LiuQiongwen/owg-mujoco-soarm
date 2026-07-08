#!/bin/bash
set -u
cd /lena/projects/OWG-main/.claude/worktrees/fix-eval-seeding

OBJECTS=("Banana" "TomatoSoupCan" "Pear" "MustardBottle" "Scissors" "CrackerBox" "PowerDrill")
ORIENT_SEEDS=(5 6 7 8 9)
GEN_SEEDS=(1 2 3 4 5 6 7 8 9 10)
CKPT="grasp_6dof/models/cfm_allobj_ot.pt"

RESULTS_DIR="/home/lina/.claude/jobs/b899ad73/tmp/exp1_variance"
mkdir -p "$RESULTS_DIR"
RAW="$RESULTS_DIR/raw_results.jsonl"
> "$RAW"

echo "[exp1] starting: ${#OBJECTS[@]} objects x ${#ORIENT_SEEDS[@]} orientations x ${#GEN_SEEDS[@]} gen-seeds = $(( ${#OBJECTS[@]} * ${#ORIENT_SEEDS[@]} * ${#GEN_SEEDS[@]} )) trials"

for obj in "${OBJECTS[@]}"; do
  for oseed in "${ORIENT_SEEDS[@]}"; do
    obj_orient_success=0
    for gseed in "${GEN_SEEDS[@]}"; do
      OUTPUT=$(timeout 90 conda run -n tango python demo.py \
        --stage 4 --prompt "$obj" --seed "$oseed" --gen-seed "$gseed" --once --verbose 0 \
        --gate-delta 0.0 --mc-gate-delta 0.0 --cfm-ckpt "$CKPT" --no-semantic 2>&1)
      EXIT_CODE=$?
      if [ $EXIT_CODE -eq 124 ]; then
        SUCCESS="timeout"
      elif echo "$OUTPUT" | grep -q "Done pick"; then
        SUCCESS="true"
        obj_orient_success=$((obj_orient_success+1))
      else
        SUCCESS="false"
      fi
      echo "{\"object\":\"$obj\",\"orient_seed\":$oseed,\"gen_seed\":$gseed,\"success\":\"$SUCCESS\"}" >> "$RAW"
    done
    echo "[progress] $obj orient_seed=$oseed: $obj_orient_success/${#GEN_SEEDS[@]} across gen-seeds"
  done
  echo "[progress] === $obj done ==="
done

echo "[exp1] ALL COMPLETE"
