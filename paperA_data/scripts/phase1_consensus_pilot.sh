#!/bin/bash
set -u
cd /lena/projects/OWG-main/.claude/worktrees/fix-eval-seeding

OBJECTS=("Pear" "CrackerBox")
ORIENT_SEEDS=(5 6 7 8 9)
ENSEMBLE_BASES=(1 6 11 16 21)   # 5 non-overlapping ensembles of N=5 each
CONSENSUS_N=5
CKPT="grasp_6dof/models/cfm_allobj_ot.pt"

RESULTS="/home/lina/.claude/jobs/b899ad73/tmp/phase1_pilot/consensus_trials.jsonl"
mkdir -p "$(dirname "$RESULTS")"
> "$RESULTS"

for obj in "${OBJECTS[@]}"; do
  for oseed in "${ORIENT_SEEDS[@]}"; do
    obj_success=0
    for base in "${ENSEMBLE_BASES[@]}"; do
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
      echo "{\"object\":\"$obj\",\"orient_seed\":$oseed,\"ensemble_base\":$base,\"success\":\"$SUCCESS\"}" >> "$RESULTS"
    done
    echo "[progress] $obj orient_seed=$oseed: $obj_success/${#ENSEMBLE_BASES[@]} consensus trials"
  done
done

echo "[phase1-pilot] COMPLETE"
