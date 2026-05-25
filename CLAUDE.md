# OWG-main — Claude Code 项目上下文

## 项目简介
Open-World Grasping (OWG) 机器人抓取项目。
主要机器人：**SO-ARM101**（MuJoCo 仿真）+ VLM Grounding + LGGSN pairwise 评分模型。

## 核心架构
- Stage 3：VLM Grounding → 随机抓取规划（baseline）
- Stage 4：VLM Grounding → LGGSN pairwise BPR 评分 → 最优抓取（当前重点）
- **6-DoF pipeline**：Open3D mesh → `grasp_6dof/grasp_generator_6dof.py` → `grasp_6dof/generate_grasps_open3d.py` → SO-ARM101 MuJoCo 验证（`benchmark/replay_soarm_grasp.py`）

## 机器人
- **主要（当前）**：SO-ARM101，MuJoCo 仿真 — `owg_robot/`，`benchmark/`
- **历史 baseline**：Panda + PyBullet — 代码已移至 `legacy/pybullet_panda/`

## 当前研究状态
- BPR pairwise + margin_0.00 为当前最优 LGGSN 设置
- 6-DoF grasp pipeline 活跃开发中（feature/vis-aabb 分支）
- SO-ARM101 MuJoCo：`physics_weld_after_bilateral` 执行模式
- Banana 验证：75% success（PCA lateral prior 有效）

## 常用命令
### Stage 4 评测
```bash
conda run -n owg-mujoco python demo.py \
  --stage 4 --prompt Banana --seed 1 --once --verbose 1 2>&1 | \
  grep -E 'LGGSN grasp scores|Final action|Done pick'
```

### 批量评测
```bash
bash scripts/quick_eval.sh
```

### 训练
```bash
conda run -n owg-mujoco python train_lggsn.py
```

### 6-DoF 抓取生成
```bash
conda run -n owg-mujoco python grasp_6dof/grasp_generator_6dof.py \
  --obj <mesh.ply> --out grasp_6dof/dataset/<name>.json \
  --world-pos 0.38,0.0,0.027
```

### SO-ARM101 抓取验证
```bash
conda run -n owg-mujoco python benchmark/replay_soarm_grasp.py \
  --grasps grasp_6dof/dataset/<name>.json
```

## Metric
success rate = "Done pick" 次数 / 总尝试次数

## Conda 环境
主环境：**owg-mujoco**，所有命令前缀：`conda run -n owg-mujoco`
