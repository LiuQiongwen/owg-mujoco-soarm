#!/bin/bash
set -u
cd /lena/projects/OWG-main/.claude/worktrees/fix-eval-seeding

ORIENT_SEEDS=(5 6 7 8 9)
# spacing must match ensemble_n so each of the 10 per-cell trials draws from
# a non-overlapping seed pool: consensus-n=5 -> spacing 5; ikmargin-n=10 -> spacing 10
BASES_N5=(1 6 11 16 21 26 31 36 41 46)
BASES_N10=(1 11 21 31 41 51 61 71 81 91)
CKPT="grasp_6dof/models/cfm_allobj_ot.pt"

RESULTS_DIR="/home/lina/.claude/jobs/b899ad73/tmp/phase1_v2"
mkdir -p "$RESULTS_DIR"

run_batch () {
  local strategy="$1" flag="$2" ensemble_n="$3" obj="$4"
  local -n bases_ref="$5"
  local raw="$RESULTS_DIR/${strategy}_${obj}.jsonl"
  > "$raw"
  for oseed in "${ORIENT_SEEDS[@]}"; do
    obj_success=0
    for base in "${bases_ref[@]}"; do
      OUTPUT=$(timeout 90 conda run -n tango python demo.py \
        --stage 4 --prompt "$obj" --seed "$oseed" --gen-seed "$base" $flag "$ensemble_n" \
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
      echo "{\"strategy\":\"$strategy\",\"object\":\"$obj\",\"orient_seed\":$oseed,\"ensemble_base\":$base,\"success\":\"$SUCCESS\"}" >> "$raw"
    done
    echo "[progress-$strategy-$obj] orient_seed=$oseed: $obj_success/${#bases_ref[@]}"
  done
  echo "[progress-$strategy-$obj] === DONE ==="
}

# 1. Consensus for MustardBottle only (Pear/CrackerBox already have n=10 consensus data)
run_batch "consensus" "--consensus-n" 5 "MustardBottle" BASES_N5

# 2. IK-margin for all 3 objects
run_batch "ikmargin" "--ikmargin-n" 10 "Pear" BASES_N10
run_batch "ikmargin" "--ikmargin-n" 10 "MustardBottle" BASES_N10
run_batch "ikmargin" "--ikmargin-n" 10 "CrackerBox" BASES_N10

echo "[phase1-v2] ALL COMPLETE"
