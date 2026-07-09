#!/bin/bash
# Completes the full 7-object (6 non-Scissors) n=50 EBM v2 evaluation, reusing
# the already-collected clean n=50 Baseline data from the Table I/II rebuild
# for comparison. Scissors excluded: its OT-CFM-style name-matching bug
# (never matches "cylinder") applies identically to the EBM's stats lookup,
# so it would silently fall back to random-CoM sampling here too -- same
# reason it's already excluded from the OT-CFM/DDPM comparisons.
#
# Seed ranges per object depend on what was already collected in the
# earlier n=25 pilot (Pear/TomatoSoupCan/CrackerBox, seeds 1-25):
#   Banana/MustardBottle/PowerDrill: full seeds 1-50 (new)
#   Pear/TomatoSoupCan/CrackerBox:   seeds 26-50 only (new half)
set -u
cd /lena/projects/OWG-main

CUPTI_SO=/home/lina/miniforge3/envs/tango/lib/python3.10/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12
NCCL_SO=/home/lina/miniforge3/envs/tango/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2
export LD_PRELOAD="$CUPTI_SO:$NCCL_SO"

CKPT="grasp_6dof/models/ebm_allobj_v2.pt"
OUT_DIR="/lena/projects/OWG-main/paperA_data/phase0_diag_extended"
mkdir -p "$OUT_DIR"

run_seeds () {
  local obj="$1"; shift
  local -n seeds_ref="$1"
  local raw="$OUT_DIR/ebm_v2_full_${obj}_${2:-seeds}.jsonl"
  raw="$OUT_DIR/ebm_v2_full_${obj}.jsonl"
  > "$raw"
  succ=0
  for seed in "${seeds_ref[@]}"; do
    OUTPUT=$(timeout 90 conda run -n tango python demo.py \
      --stage 4 --prompt "$obj" --seed "$seed" --once --verbose 0 \
      --gate-delta 0.0 --mc-gate-delta 0.0 --ebm-ckpt "$CKPT" --no-semantic 2>&1)
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
      SUCCESS="timeout"
    elif echo "$OUTPUT" | grep -q "Done pick"; then
      SUCCESS="true"
      succ=$((succ+1))
    else
      SUCCESS="false"
    fi
    echo "{\"condition\":\"ebm_v2\",\"object\":\"$obj\",\"seed\":$seed,\"success\":\"$SUCCESS\"}" >> "$raw"
  done
  echo "[ebm-v2-full-$obj] === DONE: $succ/${#seeds_ref[@]} ==="
}

SEEDS_1_50=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50)
SEEDS_26_50=(26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50)

for obj in Banana MustardBottle PowerDrill; do
  run_seeds "$obj" SEEDS_1_50
done
for obj in Pear TomatoSoupCan CrackerBox; do
  run_seeds "$obj" SEEDS_26_50
done
