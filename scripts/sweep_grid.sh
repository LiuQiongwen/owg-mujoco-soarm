#!/usr/bin/env bash
set -euo pipefail

# --------- 可配参数 ----------
YAW_LIST=("8" "12" "16")
VOX_LIST=("0.004" "0.005" "0.006")
SEED_LIST=("1" "2" "3")
# 可选的 z-margin（仅当 generate_grasps_open3d.py 支持 --z-margin 时使用）
Z_MARGIN_LIST=("0.002" "0.003" "0.004")
USE_Z_MARGIN=1         # 1=尝试传 --z-margin，若报错自动降级
OBJ="cube.urdf"
CUBE_SCALE="0.08"
IMG_W=256
IMG_H=256
TOPK=50
TOPK_BULLET=150
TOPK_VALIDATE=12
SUMMARY_CSV="grasp_6dof/out/summary.csv"

# --------- 运行环境 ----------
mkdir -p grasp_6dof/dataset grasp_6dof/out scripts
# 服务器/本机无 GUI 执行时建议开启 EGL；本地可关闭
export PYBULLET_EGL=1

echo "== Pipeline start =="
echo "Summary CSV: $SUMMARY_CSV"

for yaw in "${YAW_LIST[@]}"; do
  for vox in "${VOX_LIST[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
      # 如果你想扫 z-margin，循环三层外再包一层；否则注释掉这个 for 和末尾的 done
      for zmar in "${Z_MARGIN_LIST[@]}"; do
        TAG="y${yaw}_v${vox}_s${seed}"
        [[ $USE_Z_MARGIN -eq 1 ]] && TAG="y${yaw}_v${vox}_m${zmar}_s${seed}"
        echo "--- Running tag: ${TAG} ---"

        GEN_LOG="grasp_6dof/out/gen_${TAG}.log"
        # 构造输出 JSON 路径
        OUT_JSON="grasp_6dof/dataset/owg_${TAG}.json"

        # 生成抓取（尝试带 z-margin；失败则不带）
        set +e
        if [[ $USE_Z_MARGIN -eq 1 ]]; then
          python grasp_6dof/generate_grasps_open3d.py \
            --obj "${OBJ}" --cube-scale "${CUBE_SCALE}" \
            --n-cand 1000 --yaw-samples "${yaw}" --voxel "${vox}" \
            --topk "${TOPK}" --topk-bullet "${TOPK_BULLET}" \
            --renderer opengl --vis 0 --img "${IMG_W}" "${IMG_H}" \
            --seed "${seed}" --out "${OUT_JSON}" \
            --z-margin "${zmar}" 2>&1 | tee "${GEN_LOG}"
          GEN_RC=$?
          if [[ $GEN_RC -ne 0 ]]; then
            echo "[WARN] --z-margin not supported or failed. Retrying without it..."
            python grasp_6dof/generate_grasps_open3d.py \
              --obj "${OBJ}" --cube-scale "${CUBE_SCALE}" \
              --n-cand 1000 --yaw-samples "${yaw}" --voxel "${vox}" \
              --topk "${TOPK}" --topk-bullet "${TOPK_BULLET}" \
              --renderer opengl --vis 0 --img "${IMG_W}" "${IMG_H}" \
              --seed "${seed}" --out "${OUT_JSON}" 2>&1 | tee "${GEN_LOG}"
          fi
        else
          python grasp_6dof/generate_grasps_open3d.py \
            --obj "${OBJ}" --cube-scale "${CUBE_SCALE}" \
            --n-cand 1000 --yaw-samples "${yaw}" --voxel "${vox}" \
            --topk "${TOPK}" --topk-bullet "${TOPK_BULLET}" \
            --renderer opengl --vis 0 --img "${IMG_W}" "${IMG_H}" \
            --seed "${seed}" --out "${OUT_JSON}" 2>&1 | tee "${GEN_LOG}"
        fi
        set -e

        # 从日志稳健提取实际写入的 JSON（以防将来文件名发生改变）
        OUT_PATH=$(grep -o 'grasp_6dof/dataset/[^ ]*\.json' "${GEN_LOG}" | tail -n 1)
        if [[ -z "${OUT_PATH:-}" ]]; then
          # 回退到我们预期的路径
          OUT_PATH="${OUT_JSON}"
        fi

        # 验证
        VAL_JSON="${OUT_PATH/.json/_validated.json}"
        python grasp_6dof/validate_grasps_panda.py \
          --obj "${OBJ}" --cube-scale "${CUBE_SCALE}" --vis 0 \
          --fast --fast-scale 0.85 \
          --descent-step 0.002 --descend-clear 0.020 \
          --vel-close 0.30 --pos-close 950 --squeeze 0.40 \
          --topk "${TOPK_VALIDATE}" --seed "${seed}" \
          --grasps "${OUT_PATH}" \
          --out "${VAL_JSON}" \
          --summary-csv "${SUMMARY_CSV}"
      done
    done
  done
done

echo "== Pipeline done =="

