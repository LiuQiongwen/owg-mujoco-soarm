# OWG-main — Claude Code 项目上下文

## 项目简介
Open-World Grasping (OWG) 机器人抓取项目。
基于 PyBullet 仿真环境 + VLM Grounding + LGGSN pairwise 评分模型。

## 核心架构
- Stage 3：VLM Grounding → 随机抓取规划（baseline）
- Stage 4：VLM Grounding → LGGSN pairwise BPR 评分 → 最优抓取（当前重点）

## 当前研究状态
- BPR pairwise + margin_0.00 为当前最优设置
- 主要挑战：PyBullet/VLM non-determinism 导致评测噪声高
- 目标：提高 grasp success rate

## 常用命令
### Stage 4 评测
```bash
conda run -n owg2 python demo.py \
  --stage 4 --prompt Banana --seed 1 --once --verbose 1 2>&1 | \
  grep -E 'LGGSN grasp scores|Final action|Done pick'
```

### 批量评测
```bash
bash scripts/quick_eval.sh
```

### 训练
```bash
conda run -n owg2 python train_lggsn.py
```

## Metric
success rate = "Done pick" 次数 / 总尝试次数

## Conda 环境
主环境：owg2，所有命令前缀：conda run -n owg2
