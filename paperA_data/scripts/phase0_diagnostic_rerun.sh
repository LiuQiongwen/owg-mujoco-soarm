#!/bin/bash
set -u
cd /lena/projects/OWG-main/.claude/worktrees/fix-eval-seeding

rm -f logs/ui_grasp_exec.jsonl
OBJECTS=("Pear" "MustardBottle" "CrackerBox")
ORIENT_SEEDS=(5 6 7 8 9)
GEN_SEEDS=(1 2 3 4 5 6 7 8 9 10)
CKPT="grasp_6dof/models/cfm_allobj_ot.pt"

RESULTS="/home/lina/.claude/jobs/b899ad73/tmp/phase0_diag/trials.jsonl"
mkdir -p "$(dirname "$RESULTS")"
> "$RESULTS"

for obj in "${OBJECTS[@]}"; do
  for oseed in "${ORIENT_SEEDS[@]}"; do
    for gseed in "${GEN_SEEDS[@]}"; do
      OUTPUT=$(timeout 90 conda run -n tango python demo.py \
        --stage 4 --prompt "$obj" --seed "$oseed" --gen-seed "$gseed" --once --verbose 0 \
        --gate-delta 0.0 --mc-gate-delta 0.0 --cfm-ckpt "$CKPT" --no-semantic 2>&1)
      EXIT_CODE=$?
      if [ $EXIT_CODE -eq 124 ]; then
        SUCCESS="timeout"
      elif echo "$OUTPUT" | grep -q "Done pick"; then
        SUCCESS="true"
      else
        SUCCESS="false"
      fi
      echo "{\"object\":\"$obj\",\"orient_seed\":$oseed,\"gen_seed\":$gseed,\"success\":\"$SUCCESS\"}" >> "$RESULTS"
    done
    echo "[progress] $obj orient_seed=$oseed done"
  done
done

cp logs/ui_grasp_exec.jsonl "/home/lina/.claude/jobs/b899ad73/tmp/phase0_diag/ui_grasp_exec_snapshot.jsonl"
echo "[phase0-diag] COMPLETE"
