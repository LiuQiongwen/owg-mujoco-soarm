#!/usr/bin/env bash
set -euo pipefail

SUMMARY="grasp_6dof/out/summary.csv"
mkdir -p "$(dirname "$SUMMARY")"

# 仅一次性设置 EGL（headless 渲染）
export PYBULLET_EGL=1

run_gen() {
  local OBJ="$1" SCALE="$2" YAW="$3" VOX="$4" ZM="$5" SEED="$6" TAG="$7"
  local OUT="grasp_6dof/dataset/gen_${TAG}.json"

  for TRY in 1 2 3; do
    echo "--- GEN [$TAG] try $TRY ---"
    if PYBULLET_EGL=1 python grasp_6dof/generate_grasps_open3d.py \
        --obj "$OBJ" --cube-scale "$SCALE" \
        --n-cand 1000 --yaw-samples "$YAW" --voxel "$VOX" \
        --topk 50 --topk-bullet 150 \
        --renderer opengl --vis 0 --img 256 256 \
        --seed "$SEED" --z-margin "$ZM" \
        --out "$OUT"; then
      echo "OK [$TAG]"
      return 0
    else
      echo "[WARN] $TAG failed. retry $TRY/3 ..."
      sleep 1
    fi
  done
  echo "[ERR] $TAG failed after 3 tries"
  return 1
}

run_val() {
  local OBJ="$1" SCALE="$2" SEED="$3" TAG="$4"
  local IN="grasp_6dof/dataset/gen_${TAG}.json"
  local OUT="grasp_6dof/dataset/${TAG}_validated.json"

  python grasp_6dof/validate_grasps_panda.py \
    --obj "$OBJ" --cube-scale "$SCALE" \
    --vis 0 --fast --fast-scale 0.85 \
    --descent-step 0.002 --descend-clear 0.020 \
    --vel-close 0.30 --pos-close 950 --squeeze 0.40 \
    --ee-index 11 --topk 12 --seed "$SEED" \
    --grasps "$IN" \
    --out "$OUT" \
    --summary-csv "$SUMMARY"
}

echo "== Grid start @ $(date '+%F %T') =="
echo "Summary CSV: $SUMMARY"

# 物体清单：优先使用 pybullet_data 自带
for OBJ in cube.urdf sphere_small.urdf cylinder.urdf duck_vhacd.urdf; do
  for SCALE in 0.08; do
    for YAW in 8 12 16; do
      for VOX in 0.004 0.005 0.006; do
        for ZM in 0.003 0.004; do
          for SEED in 1 2 3; do
            TAG="$(basename "$OBJ" .urdf)_y${YAW}_v${VOX}_zm${ZM}_s${SEED}"
            run_gen "$OBJ" "$SCALE" "$YAW" "$VOX" "$ZM" "$SEED" "$TAG" && \
            run_val "$OBJ" "$SCALE" "$SEED" "$TAG" || true
          done
        done
      done
    done
  done
done

echo "== Grid done @ $(date '+%F %T') =="
python scripts/summarize.py "$SUMMARY" || true

