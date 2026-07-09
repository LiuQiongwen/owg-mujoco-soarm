#!/bin/bash
# Extends the ikmargin(n=10) side of phase1_v2/ikmargin_{Pear,MustardBottle,
# CrackerBox}.jsonl to TomatoSoupCan, matching the exact design (5 orient_seeds
# x 10 ensemble_bases, --ikmargin-n 10, same OT-CFM checkpoint) so it is
# directly comparable via run_ikmargin_vs_consensus_matched.py against
# consensus_n10_TomatoSoupCan.jsonl (run_consensus_n10_tomatosoupcan.sh).
#
# Motivation (2026-07-09): the paper's "Reliability across generation seeds"
# paragraph names Pear AND TomatoSoupCan as objects where OT-CFM is less
# reliable than CFM-noOT/DDPM across seeds. The existing ikmargin-vs-consensus
# data only covers Pear/MustardBottle/CrackerBox -- this closes the gap for
# the second named object.
#
# Same LD_PRELOAD workaround as run_consensus_n10_matched.sh: a stray
# user-site nvidia-cuda-cupti-cu12==12.1.105/libnccl shadow tango's own
# matched versions via Python user-site precedence.
set -u
cd /lena/projects/OWG-main/.claude/worktrees/fix-eval-seeding

CUPTI_SO=/home/lina/miniforge3/envs/tango/lib/python3.10/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12
NCCL_SO=/home/lina/miniforge3/envs/tango/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2
export LD_PRELOAD="$CUPTI_SO:$NCCL_SO"

OBJ="TomatoSoupCan"
ORIENT_SEEDS=(5 6 7 8 9)
BASES_N10=(1 11 21 31 41 51 61 71 81 91)
IKMARGIN_N=10
CKPT="grasp_6dof/models/cfm_allobj_ot.pt"

RESULTS_DIR="/lena/projects/OWG-main/paperA_data/phase1_v2"
mkdir -p "$RESULTS_DIR"
raw="$RESULTS_DIR/ikmargin_${OBJ}.jsonl"
> "$raw"

for oseed in "${ORIENT_SEEDS[@]}"; do
  obj_success=0
  for base in "${BASES_N10[@]}"; do
    OUTPUT=$(timeout 90 conda run -n tango python demo.py \
      --stage 4 --prompt "$OBJ" --seed "$oseed" --gen-seed "$base" --ikmargin-n "$IKMARGIN_N" \
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
    echo "{\"strategy\":\"ikmargin\",\"object\":\"$OBJ\",\"orient_seed\":$oseed,\"ensemble_base\":$base,\"success\":\"$SUCCESS\"}" >> "$raw"
  done
  echo "[progress-ikmargin-$OBJ] orient_seed=$oseed: $obj_success/${#BASES_N10[@]}"
done
echo "[progress-ikmargin-$OBJ] === DONE ==="
