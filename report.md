# 实验报告：闯关类手游用户流失预测

## 一、任务
根据 2.1–2.4 的交互日志，预测次周（2.7–2.13）是否登录（1=流失，0=留存）。评价指标：AUC。

## 二、数据
- `train.csv` / `dev.csv` / `test.csv`（均为 **制表符分隔**）
- `level_seq.csv`：逐关卡序列日志，含 `user_id, level_id, f_success, f_duration, f_reststep, f_help, time`
- `level_meta.csv`：关卡层面统计，含 `level_id, f_avg_passrate, f_avg_duration, f_avg_win_duration, f_avg_retrytimes`
- （可选）Groundtruth 用于 Test 复核

## 三、方法概述
**思路**：将原始序列按 `user_id` 聚合为用户级特征表 → 传统模型建模（可解释、鲁棒、效率高）→ 多模型集成提升上限。

### 3.1 特征工程（用户级聚合）
- **体量/多样性**：总游玩次数 `plays_total`、不同关卡数 `levels_unique`、`level_max`
- **结果表现**：成功次数/成功率 `succ_cnt/succ_rate`
- **过程强度**：时长 `dur_sum/dur_mean`、剩余步数均值 `reststep_mean`、求助比例 `help_rate`
- **关卡元信息**（与 `level_meta` 交叉）：用户经历关卡的平均通关率/平均时长/平均胜利时长/平均重试次数
- **衍生比率**：`succ_per_play`、`dur_per_day` 等
> 数据清洗：`time` 解析为 datetime，仅在聚合时使用，最终输入模型的均为数值列，避免 datetime 与 float 混型导致的 dtype 错误。

### 3.2 模型与训练
- **基学习器**：
  - HistGradientBoosting（树模型，能捕捉非线性、交互）
  - Logistic Regression（配合缺失值填充与标准化）
- **堆叠策略**：5 折 Stratified KFold 生成 OOF（`hgb_oof`、`lr_oof`）→ 作为元特征喂给 **元学习器 LR**
- **评估**：AUC；绘制 ROC/PR 以观察阈值变化下的性能

## 四、实验结果
- 训练样本数：**8158**
- 开发集样本数：**2658**
- 测试集样本数：**2773**
- 最终特征维度：**55**
- **Dev AUC：0.799207**
- 生成提交文件：`submission.csv`（由 Notebook 自动导出）

> 说明：AUC 在 0.79–0.80 区间的轻微波动属于正常（随机种子、环境、并行线程等会有细微影响）。

## 五、误差分析（要点）
- 早期短时段数据对“长期留存”存在天然不确定性，模型对短频快用户更敏感；
- 关卡特征与用户行为的交互尚未充分展开（例如分段“卡关”行为、最近行为衰减）；
- 部分用户数据较稀疏，聚合特征不稳定，可通过目标编码/贝叶斯平滑缓解。

## 六、业务落地建议
- 以概率分层：Top 10/20/30% 高风险用户进行不同力度干预（奖励、短信、Push、客服回访）；
- A/B 验证：不同干预策略的 **增量留存率** 与 **单位成本**；
- 在线监控：AUC 与 PSI（特征分布漂移）预警，版本更新或节假日前后重点关注。

## 七、改进方向（可冲击 +0.5~1 个百分点 AUC）
1. **序列特征**：最近 N 局加权、最长失败连串、卡关时长/解锁速度、日/时段节律；
2. **更多基模型**：LightGBM / CatBoost 纳入堆叠；
3. **阈值与校准**：Platt / Isotonic 校准后按业务目标优化阈值；
4. **更细致的冷启动处理**：安装来源、设备、渠道特征（若可用）；
5. **K 折数上调 + 多次重启**：降低方差，提交集成平均。

## 八、可复现实验文件
- Notebook：`final_project_notebook.ipynb`（包含完整训练/评估/提交导出）
- 提交：`submission.csv`
- 报告：`report.md`

