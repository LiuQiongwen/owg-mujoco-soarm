#!/usr/bin/env bash
set -euo pipefail

# ==== 可调参数（按需改） ====
OBJ="cube.urdf"
CUBE_SCALE=0.08

# 采样与筛选
N_CAND=1000
YAW_SAMPLES_LIST=(8 12)
VOXEL_LIST=(0.004 0.005)
SEEDS=(1 2 3)

TOPK=50
TOPK_BULLET=150
IMG_W=256
IMG_H=256

# 验证参数
TOPK_VALIDATE=12
FAST=1           # 1=开 --fast；0=关
FAST_SCALE=0.85
DESCENT_STEP=0.002
DESCEND_CLEAR=0.020
VEL_CLOSE=0.30
POS_CLOSE=950
SQUEEZE=0.40

# 输出
OUT_DIR="grasp_6dof"
DATA_DIR="$OUT_DIR/dataset"
OUT_DIR_IMG="$OUT_DIR/out"
SUMMARY_CSV="$OUT_DIR/out/summary.csv"

# ==== 函数 ====
run_generate() {
  local seed="$1" yaw="$2" voxel="$3"
  local tag="s${seed}_yaw${yaw}_vox${voxel}"
  local out_json="${DATA_DIR}/gen_${tag}.json"

  # 生成（无头 + EGL + OpenGL 渲染器）
  export PYBULLET_EGL=1
  python grasp_6dof/generate_grasps_open3d.py \
    --obj "$OBJ" --cube-scale "$CUBE_SCALE" \
    --n-cand "$N_CAND" --yaw-samples "$yaw" --voxel "$voxel" \
    --topk "$TOPK" --topk-bullet "$TOPK_BULLET" \
    --renderer opengl --vis 0 --img "$IMG_W" "$IMG_H" \
    --seed "$seed" \
    --out "$out_json"

  echo "$out_json"
}

run_vis_headless() {
  local grasps_json="$1"
  local png_out="$2"

  # 可视化（无头 Tiny 渲染，保存 PNG）
  export PYBULLET_EGL=0
  python grasp_6dof/vis_grasps.py \
    --grasps "$grasps_json" \
    --cube-scale "$CUBE_SCALE" \
    --headless \
    --out-png "$png_out" \
    --show-n 20
}

run_validate() {
  local seed="$1" grasps_json="$2" out_json="$3"

  # 验证（GUI/无头均可；这里无头运行更稳）
  local fast_flag=()
  if [[ "$FAST" -eq 1 ]]; then
    fast_flag=(--fast --fast-scale "$FAST_SCALE")
  fi

  export PYBULLET_EGL=1
  python grasp_6dof/validate_grasps_panda.py \
    --obj "$OBJ" \
    --cube-scale "$CUBE_SCALE" \
    --grasps "$grasps_json" \
    --out "$out_json" \
    --vis 0 \
    --topk "$TOPK_VALIDATE" \
    "${fast_flag[@]}" \
    --descent-step "$DESCENT_STEP" \
    --descend-clear "$DESCEND_CLEAR" \
    --vel-close "$VEL_CLOSE" \
    --pos-close "$POS_CLOSE" \
    --squeeze "$SQUEEZE" \
    --seed "$seed" \
    --summary-csv "$SUMMARY_CSV"
}

# ==== 主流程 ====
main() {
  mkdir -p "$DATA_DIR" "$OUT_DIR_IMG"

  echo "== Pipeline start =="
  echo "Summary CSV: $SUMMARY_CSV"
  # 如果 summary.csv 不存在，让验证脚本去创建/追加表头
  # 我们这里不清空，方便累计实验

  for seed in "${SEEDS[@]}"; do
    for yaw in "${YAW_SAMPLES_LIST[@]}"; do
        for voxel in "${VOXEL_LIST[@]}"; do
            tag="s${seed}_yaw${yaw}_vox${voxel}"
            echo "--- Running tag: ${tag} ---"

            gen_log="${OUT_DIR_IMG}/gen_${tag}.log"
            gen_json="${DATA_DIR}/gen_${tag}.json"

            # 生成阶段：把 stdout/err 都写到日志，别回传到变量
            export PYBULLET_EGL=1
            python grasp_6dof/generate_grasps_open3d.py \
                --obj "$OBJ" --cube-scale "$CUBE_SCALE" \
                --n-cand "$N_CAND" --yaw-samples "$yaw" --voxel "$voxel" \
                --topk "$TOPK" --topk-bullet "$TOPK_BULLET" \
                --renderer opengl --vis 0 --img "$IMG_W" "$IMG_H" \
                --seed "$seed" --out "$gen_json" \
                >"$gen_log" 2>&1

            # 无头可视化
            run_vis_headless "$gen_json" "${OUT_DIR_IMG}/vis_${tag}.png"

            # 验证
            val_json="${DATA_DIR}/validated_${tag}.json"
            run_validate "$seed" "$gen_json" "$val_json"

            echo "Done ${tag}"
          done
      done
  done


  echo "== Pipeline done =="
  echo "Last few lines of summary:"
  tail -n 10 "$SUMMARY_CSV" || true
}

main "$@"

