#!/bin/bash
set -u
cd /lena/projects/OWG-main/.claude/worktrees/fix-eval-seeding

OBJECTS=("Banana" "TomatoSoupCan" "Pear" "MustardBottle" "Scissors" "CrackerBox" "PowerDrill")
ORIENT_SEEDS=(5 6 7 8 9)
GEN_SEEDS=(1 2 3 4 5 6 7 8 9 10)

RESULTS_DIR="/home/lina/.claude/jobs/b899ad73/tmp/exp1_variance"
mkdir -p "$RESULTS_DIR"

run_method () {
  local method="$1" ckpt="$2" extra_env="$3"
  local raw="$RESULTS_DIR/raw_results_${method}.jsonl"
  > "$raw"
  echo "[exp1-$method] starting: ${#OBJECTS[@]} x ${#ORIENT_SEEDS[@]} x ${#GEN_SEEDS[@]} trials"
  for obj in "${OBJECTS[@]}"; do
    for oseed in "${ORIENT_SEEDS[@]}"; do
      obj_orient_success=0
      for gseed in "${GEN_SEEDS[@]}"; do
        if [ -n "$extra_env" ]; then
          OUTPUT=$(timeout 90 conda run -n tango env $extra_env python demo.py \
            --stage 4 --prompt "$obj" --seed "$oseed" --gen-seed "$gseed" --once --verbose 0 \
            --gate-delta 0.0 --mc-gate-delta 0.0 --cfm-ckpt "$ckpt" --no-semantic 2>&1)
        else
          OUTPUT=$(timeout 90 conda run -n tango python demo.py \
            --stage 4 --prompt "$obj" --seed "$oseed" --gen-seed "$gseed" --once --verbose 0 \
            --gate-delta 0.0 --mc-gate-delta 0.0 --cfm-ckpt "$ckpt" --no-semantic 2>&1)
        fi
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 124 ]; then
          SUCCESS="timeout"
        elif echo "$OUTPUT" | grep -q "Done pick"; then
          SUCCESS="true"
          obj_orient_success=$((obj_orient_success+1))
        else
          SUCCESS="false"
        fi
        echo "{\"object\":\"$obj\",\"orient_seed\":$oseed,\"gen_seed\":$gseed,\"success\":\"$SUCCESS\"}" >> "$raw"
      done
      echo "[progress-$method] $obj orient_seed=$oseed: $obj_orient_success/${#GEN_SEEDS[@]}"
    done
    echo "[progress-$method] === $obj done ==="
  done
  echo "[exp1-$method] COMPLETE"
}

run_method "CFM-noOT" "grasp_6dof/models/cfm_allobj.pt" ""
run_method "DDPM" "grasp_6dof/models/ddpm_allobj.pt" "DDIM_STEPS=50"

echo "[exp1] ALL METHODS COMPLETE"
