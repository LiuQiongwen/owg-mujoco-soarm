#!/usr/bin/env bash
# run_grid.sh — 批量生成候选 → 验证 → 写入 summary.csv
# 用法：
#   chmod +x scripts/run_grid.sh
#   ./scripts/run_grid.sh
#
# 可选环境变量：
#   SEEDS="1 2 3"      YAWS="8 12 16"     VOXELS="0.004 0.005 0.006"
#   ZMARGINS="0.003 0.004"  TOPK=12  CUBE_SCALE=0.08  FORCE=0/1

set -u  # 不自动忽略未定义变量

# ---- 参数网格（可用环境变量覆盖）----
SEEDS=${SEEDS:-"1 2 3"}
YAWS=${YAWS:-"8 12 16"}
VOXELS=${VOXELS:-"0.004 0.005 0.006"}
ZMARGINS=${ZMARGINS:-"0.003 0.004"}

TOPK=${TOPK:-12}
CUBE_SCALE=${CUBE_SCALE:-0.08}
FORCE=${FORCE:-0}

GEN=grasp_6dof/generate_grasps_open3d.py
VAL=grasp_6dof/validate_grasps_panda.py

OUTDIR=grasp_6dof/dataset
SUMCSV=grasp_6dof/out/summary.csv
LOGDIR=grasp_6dof/out/logs
mkdir -p "$OUTDIR" "$LOGDIR" "$(dirname "$SUMCSV")"

echo "== Pipeline start =="
echo "Summary CSV: $SUMCSV"

# 小工具：安全执行并将stderr写日志，不中断整个循环
run() {
  local tag="$1"; shift
  local log="$LOGDIR/${tag}_$(date +%Y%m%d-%H%M%S).log"
  echo "--- RUN [$tag] ---"
  "$@" >"$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "!! FAIL [$tag], see $log"
  else
    echo "OK [$tag]"
  fi
  return $rc
}

export PYBULLET_EGL=1  # 生成阶段走 EGL，无 GUI
IMG_W=256; IMG_H=256

for yaw in $YAWS; do
  for vox in $VOXELS; do
    for zm in $ZMARGINS; do
      for s in $SEEDS; do
        tag="y${yaw}_v${vox}_zm${zm}_s${s}"
        GEN_JSON="$OUTDIR/owg_${tag}.json"
        VAL_JSON="$OUTDIR/owg_${tag}_validated.json"

        # 1) 生成 grasps（如存在且 FORCE=0 则跳过）
        if [ $FORCE -eq 1 ] || [ ! -s "$GEN_JSON" ]; then
          run "gen_$tag" \
            python "$GEN" \
              --obj cube.urdf --cube-scale "$CUBE_SCALE" \
              --n-cand 1000 --yaw-samples "$yaw" --voxel "$vox" \
              --topk 50 --topk-bullet 150 \
              --renderer opengl --vis 0 --img $IMG_W $IMG_H \
              --seed "$s" --z-margin "$zm" \
              --out "$GEN_JSON"
        else
          echo "SKIP gen ($GEN_JSON exists)"
        fi

        # 2) 验证（headless + fast）
        run "val_$tag" \
          python "$VAL" \
            --obj cube.urdf --cube-scale "$CUBE_SCALE" --vis 0 \
            --fast --fast-scale 0.85 \
            --descent-step 0.002 --descend-clear 0.020 \
            --vel-close 0.30 --pos-close 950 --squeeze 0.40 \
            --topk "$TOPK" --seed "$s" \
            --grasps "$GEN_JSON" \
            --out "$VAL_JSON" \
            --summary-csv "$SUMCSV"

      done
    done
  done
done

echo "== Pipeline done =="

