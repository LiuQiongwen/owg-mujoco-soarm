# Piper 抓取调查：现状总结

内部总结文档，面向自己/合作者，聚焦结论和下一步决策，不追求论文级别的完整性和措辞。详细的、按时间顺序
的诊断过程见 `cr_cfm/IMPROVEMENT_PLAN.md`（工作日志，200+ 条）和 `README.md`（同步的对外可读版本）。

## 一句话结论

**CR-CFM（49K参数flow-matching模型 + RHC）不起作用；真正有效的是一个简单、训练-free、机制明确的抓取
朝向修复（wrist-friendly grasp orientation），它的适用范围本身可以用一个廉价的 IK 诊断工具提前预测。**

## 已确认结论

### 1. CR-CFM 被决定性证伪

最初的假设是"学习式流匹配修正模型 + 退避视界控制（RHC）能提升抓取成功率"。三条独立证据都不支持这个假设：

- **关键消融**（隔离 wrist-fix 和 CR-CFM 的混淆）：n=32 时完全打平（26/32 vs 26/32，0 个不一致 pair）。
  怀疑是样本量不足掩盖了真实效果，于是扩大到 n=152（匹配 Stage 12 的统计功效）——**结果还是不显著**
  （baseline 115/152=75.7% vs CR-CFM 110/152=72.4%，McNemar p=0.33），且不一致的部分还偏向 baseline
  （11 vs 6）。5倍样本量不仅没有揭示隐藏优势，方向还反过来了。
- **扰动测试**：给 descend 阶段注入一次性关节扰动，模拟外部干扰——CR-CFM 依然没有表现出优势。
- **架构级解释**：`target_qpos` 在每次 descend 阶段开始前只求解一次（`solve_multi_seed`），RHC 循环
  只重新读取机械臂自身的关节状态，从不重新感知物体位置或重新求解目标。也就是说 CR-CFM 的"闭环"只是
  绕着自己转，从未真正响应环境——这从机制上解释了为什么它测不出优势，不只是"这次实验没测到"。

### 2. 唯一验证有效的贡献：wrist-friendly grasp orientation

`piper_pick_and_place.py` 里的 `pick_wrist_friendly_orientation`：对同一个抓取目标求解两个候选朝向
（原始朝向 + 180°翻转），保留哪个让 joint6（腕部旋转关节，真实硬件限位 ±3.14 rad）离限位更远。

- **机制诊断**：joint6 顶到限位与抓取失败强相关，Fisher's exact p=1.8e-5，odds ratio 90。
- **统计确认**：Cracker 上 n=152，McNemar p=0.027，显著改善。
- **训练-free、模型无关**：不依赖任何学习模型，纯几何/IK 判断，因此和 CR-CFM 是否存在无关——这也是
  为什么关键消融能干净地把两者的贡献分开。

### 3. 修复的物体依赖性，且这种依赖性可以提前预测

不是所有物体都受益：

| 物体 | 朴素姿态下 joint6 顶限位的比例 | wrist-fix 是否有效 |
|---|---|---|
| Cracker | ~30-34%（IK-proxy 6/20，历史基准 11/32） | **有效**（p=1.8e-5 / p=0.027） |
| Pear | 0%（执行验证 0/16） | **无效**（执行验证：两条件完全打平，6/8 vs 6/8） |
| Mustard | 5%（仅 IK-proxy 预测，未做完整执行验证） | **预测无效**（按预注册决策规则跳过昂贵的执行验证） |

关键方法论贡献：构建并**严格验证**了一个廉价的 IK-only 预测工具——先发现代理工具本身有 bug（错误地
测量了"IK 求解出的目标关节角"而不是"进入 descend 阶段时机械臂的当前/种子关节角"），修复后精确复现了
Cracker 已知的 11/32 基准率（trial-for-trial 完全一致）。这个工具让"物体是否会受益于 wrist-fix"变成
一个几乎零成本、可提前回答的问题，而不需要每个新物体都跑一遍完整的物理执行实验。

### 4. Real-Hardware Readiness 5-gate 全部走完（仿真侧）

- Gate 1（真机基线对比）：通过，p=0.0156。
- Gate 2（多物体 pilot）：最初失败（Pear 上 CR-CFM+wristfix 明显更差），但后续诊断完全解释了原因
  （数据量不足 → 扩大到143条轨迹后达到平台期 → 最终发现问题根本不在 CR-CFM，是整个方法论的问题）。
- Gate 3（安全裁剪层）：通过，发现并修复了一个真实的安全覆盖漏洞（`lower_into_tray` 阶段未被保护），
  `clip_action_to_real_limits` 验证后 275→0 违规。
- Gate 4（失效模式分类）：通过，已知失效模式均为良性/可恢复。
- Gate 5（分阶段真机部署程序）：已文档化，但**从未在真实硬件上执行**（本会话全程仅有仿真访问权限）。

## Track A / Track B 进展（2026-07-22）

**Track A（论文/报告叙事）**：完成。本文档 + 内部总结 artifact（n=152 消融对比、跨物体 joint6 分布
gauge、Real-Hardware Readiness 5-gate 表）。

**Track B（真机部署，软件侧准备）**：`piper_sdk` 已安装（v0.6.1，网络可用），并对照官方
`piper_sdk/demo/V2/*.py` 逐条核实了 `piper_real_backend.py` 的每个方法体——两个具体阻塞项已解决：

1. **API 层全部核实并修正**：主类应为 `C_PiperInterface_V2`（不是原来假设的 `C_PiperInterface`，
   两者是完全独立的类）；`JointCtrl`/`GetArmJointMsgs` 单位是 0.001°整数（换算系数
   `RAD_TO_MILLIDEG=57295.7795`，来自官方 demo `piper_ctrl_joint.py` 自己的 `factor` 定义）；
   `GripperCtrl`/`GetArmGripperMsgs` 单位是 0.001mm 整数；`EnablePiper()` 是官方文档化的轮询式使能
   方法（内部调用 `EnableArm(7)`，需要 `while not piper.EnablePiper(): sleep(...)` 轮询，已加超时保护）；
   `MotionCtrl_2(0x01,0x01,speed,0x00)` 必须在 `JointCtrl` 前调用以切换到关节控制模式。所有 "VERIFY:"
   占位注释已替换为经核实的确定值。
2. **夹爪单位换算 blocker 已解决**：直接用真实仿真运行核查 `env.sim.data.ctrl[-1]`（`PiperTrajectoryRecorder.
   snap()` 记录的量），发现它其实已经是"单指绝对位置（米，-0.05 到 -0.004）"，不是文档原先声称的
   "原始 ±1 动作信号"——这个说法本身是过时的（对应更早一次已修复的"双重缩放"bug，修复后从未同步更新
   到这份文档）。用两个已有的标定锚点（XML 的 `ctrlrange` 边界 + `piper_real_backend.py` 已经过实测的
   `GRIP_OPEN_M=0.12`）建立线性换算 `finger_qpos_to_span_m()`，`PiperTrajectoryReplayer.replay()`
   现在能真正驱动夹爪回放，不再抛 `NotImplementedError`。

**⚠️ 新发现的关键差异，尚未解决**：`piper_sdk` 自己的 `JointCtrl` 文档把 joint6（腕部旋转）的**默认**
限位列为 **±2.09439 rad（±120°）**，比整个 CR-CFM/wrist-fix 调查假设的 ±3.14 rad（±180°，来自
`robot_arm.xml` 的 MuJoCo 关节 range）窄了整整 60°。这是一个可通过 `MotorAngleLimitMaxSpdSet`
（CAN 0x474，写入 flash）重新配置的软件限位，官方 demo 示例把它放宽到过 ±170°（仍不到 ±180°）。
**在真机上测试 wrist-fix 之前，必须先查询机械臂实际配置的 joint6 限位，不能直接沿用仿真里的 3.14
这个数字**——如果真实限位确实更窄，`pick_wrist_friendly_orientation(joint6_limit=3.14)` 这个阈值本身
需要针对真实配置重新计算，否则可能悄悄削弱甚至失效整个修复的实际效果。

**仍未解决/依赖硬件**：CAN 适配器激活、真实电机通讯延迟/噪声特性、上述 joint6 限位差异的最终核实——
这些只能在物理接入机械臂后验证。

## 下一步建议（详见 `/home/lina/.claude/plans/floating-crunching-yeti.md`）

- Track A 已完成。
- Track B：软件侧的三个阻塞项中两个已解决（API 核实、夹爪换算），新发现一个需要真机才能最终核实的
  限位差异问题。下一步是物理接入机械臂后：(1) 查询实际 joint6 限位配置，(2) 按 Gate 5 已经写好的
  分阶段程序推进——回放的应该是 wrist-fix 轨迹，不是 CR-CFM（仿真结果已经决定性地排除了这条路）。
