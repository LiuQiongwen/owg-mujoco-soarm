#!/bin/bash
# Extends the phase0 diagnostic pipeline (originally Pear/MustardBottle/CrackerBox,
# see phase0_diagnostic_rerun.sh) to 3 more objects: TomatoSoupCan, PowerDrill,
# Scissors -- chosen because they're the only remaining exp1_variance objects
# with a real success/fail split under OT-CFM (Banana is 100% success, useless
# for a success-vs-fail contact-feature comparison).
#
# Same design as the original: single-draw (no --consensus-n/--ikmargin-n),
# OT-CFM checkpoint, 5 orient_seeds x 10 gen_seeds = 50 trials/object.
#
# Environment: same LD_PRELOAD fix as run_consensus_n10_matched.sh (tango's
# own torch/cupti/nccl are correct; a stray user-level cupti/nccl shadow them
# via Python's default user-site precedence -- preload tango's own .so files
# to bypass, without touching torch, tango's cupti/nccl, or the stray package).
set -u
cd /lena/projects/OWG-main/.claude/worktrees/fix-eval-seeding

CUPTI_SO=/home/lina/miniforge3/envs/tango/lib/python3.10/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12
NCCL_SO=/home/lina/miniforge3/envs/tango/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2
export LD_PRELOAD="$CUPTI_SO:$NCCL_SO"

OBJECTS=("TomatoSoupCan" "PowerDrill" "Scissors")
ORIENT_SEEDS=(5 6 7 8 9)
GEN_SEEDS=(1 2 3 4 5 6 7 8 9 10)
CKPT="grasp_6dof/models/cfm_allobj_ot.pt"

RESULTS_DIR="/lena/projects/OWG-main/paperA_data/phase0_diag_extended"
mkdir -p "$RESULTS_DIR"
RAW="$RESULTS_DIR/trials_new3.jsonl"
> "$RAW"
rm -f logs/ui_grasp_exec.jsonl

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
      echo "{\"object\":\"$obj\",\"orient_seed\":$oseed,\"gen_seed\":$gseed,\"success\":\"$SUCCESS\"}" >> "$RAW"
    done
    echo "[progress-phase0-ext] $obj orient_seed=$oseed done"
  done
done

cp logs/ui_grasp_exec.jsonl "$RESULTS_DIR/ui_grasp_exec_snapshot_new3.jsonl"
echo "[phase0-ext] COMPLETE"
