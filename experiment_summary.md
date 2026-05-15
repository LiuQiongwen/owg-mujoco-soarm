# 实验结果总结

## Baseline
- Stage 3 random: 16/20 (80%)
- Stage 4 BPR margin_0.00: 16/20 (80%)

## 关键发现
- BPR pairwise > BCE supervision
- margin_0.00 > margin_0.05
- Stage 4 目前和 Stage 3 持平，LGGSN 尚未带来提升

## 数据集
- YCB 物品：Banana, TomatoSoupCan, Pear, MustardBottle
- 训练数据：4190 samples，正负比 58:42
- 评测：20次/配置（4 objects × 5 seeds）

## 待补充
[把你 autoresearch 跑出来的最好结果填在这里]
