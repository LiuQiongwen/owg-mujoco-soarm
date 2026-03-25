# /autoresearch — OWG Grasp Success Rate 自动优化循环

## GOAL
提高 OWG Stage 4 grasp success rate。
当前最优：BPR pairwise + margin_0.00。

## METRIC
```bash
bash scripts/quick_eval.sh 2>&1 | grep "RESULT:"
```

## SCOPE
只允许修改：train_lggsn.py、lggsn_model.py、config/ 下配置文件
不允许修改：demo.py、owg/、owg_robot/、grasp_6dof/

## LOOP
每次迭代：
1. git log --oneline -10 | grep experiment  # 看历史
2. 提出一个假设（每次只改一个变量）
3. 训练：conda run -n owg2 python train_lggsn.py
4. 评测：bash scripts/quick_eval.sh
5. 记录：git commit -m "experiment: <描述> | success=X/20"
6. 决策：比上次好 → 保留，否则 git revert HEAD

## 停止条件
连续 3 次无提升，或跑满 20 次迭代后汇报结果。
