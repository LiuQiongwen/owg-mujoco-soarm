#!/usr/bin/env bash
set -e
objs=(cube cylinder capsule sphere_small)
scales=(0.06 0.08 0.10)
seeds=(1 2 3 4 5)
for O in "${objs[@]}"; do
  for S in "${seeds[@]}"; do
    for H in "${scales[@]}"; do
      python grasp_6dof/validate_grasps_panda.py \
        --obj grasp_6dof/assets/${O}.urdf --cube-scale $H \
        --ee-index 11 --vis 0 --fast --fast-scale 0.85 \
        --descent-step 0.002 --descend-clear 0.025 \
        --vel-close 0.40 --pos-close 1100 --squeeze 0.50 \
        --topk 12 --seed $S \
        --grasps grasp_6dof/dataset/sanity_${O}.json \
        --out runs/baseline/${O}_H${H}_S${S}.json \
        --summary-csv runs/baseline/summary.csv
    done
  done
done

