# 阶段 1 coverage wall 恢复计划

日期：2026-07-11

状态：实现与预验证已完成，计划继续冻结；等待项目负责人运行 120 局候选矩阵。若需改变目标或门槛，
必须先在本文件追加“计划偏差”，不得在实现中静默放宽。

## 1. 背景与已确认事实

阶段 1 的功能交付已经存在，但 2026-06-13 全量矩阵未达到总体设计 §4 的阶段 1 门槛：

- residual 相对 v31 的全矩阵 recall 为 0.646 vs 0.625，配对置换 `p=0.11989`，未显著提升；
- `not_reached` 为 177 vs 186，仅下降 9 个，且 open world 无改善；
- precision 95% 下界为 0.960，duplicate rate 为 0.014 vs 0.018，这两项通过；
- FUSE 与 clearance 计算低于 100 ms，算力项通过。

本轮重新读取运行日志后确认了一个独立于权重选择的执行契约错误：

- `select_residual_target()` 默认 `min_d=2.5 m`；
- COARSE waypoint 的到达半径 `frontier_arrival_r=2.5 m`；
- `_residual_waypoint()` 没有传入已有的 `survey_waypoint_min_d=5.0 m` 和
  `survey_waypoint_max_d=14.0 m`；
- 因而新目标可以在生成当刻即满足“已到达”，状态机下一 tick 又选相邻网格；
- 只统计每局主 `launch.log`，旧 residual 全矩阵累计出现 5176 次
  `COARSE_WP#... arrival`，其中 20 个 static5 run 为 723 次且只有 2 次 timeout；
  open/static5/seed101 一次运行中出现百次级连续换点。若连同 `ros_logs` 内副本会重复计数，
  因而不得用 1446 作为 static5 主日志计数。

在修复这个确定性缺陷之前，不能用旧矩阵评价 residual 覆盖策略本身。

## 2. 本轮目标与非目标

### 目标

1. 建立“规划最小距离必须大于控制器到达半径”的显式契约和回归测试。
2. residual COARSE 目标使用配置中的 survey 距离范围，并对近目标做运行时防御。
3. 保持真实残差峰优先、可见性三态、直线可达性和已确认源去重语义不变。
4. 新增阶段 1 专用门槛工具，逐项执行总体设计 §4 的统计判定。
5. 完成纯测试、构建和短冒烟；全量矩阵只准备命令，由项目负责人运行。

### 非目标

- 本轮不实现阶段 2 Kalman 跟踪或重访队列；
- 不通过调低确认阈值、增大 FOV 或移动真值源制造 recall；
- 不修改 v31/full 锚点行为；
- 不改写既有 phase0/phase1 矩阵产物；
- 不以单 seed 冒烟代替全量门槛；
- 不在阶段 1 门槛通过前更新总体 spec 为“已完成”。

## 3. 设计决策

### 3.1 有效最小目标距离

新增纯函数计算 operational minimum：

```text
effective_min_d = max(configured_min_d,
                      arrival_radius + max(configured_margin, 2 * resolution))
```

理由：目标栅格量化、里程计抖动和 `<= arrival_radius` 判定都要求严格正裕量。默认配置下仍由
`survey_waypoint_min_d=5.0 m` 主导，不引入新的行为常数。

### 3.2 residual waypoint 接线

`ControllerNode._residual_waypoint()` 必须显式传入：

- `min_d=effective_min_d`；
- `max_d=survey_waypoint_max_d`；
- `safe_dist=_safe_dist()`；
- 原有 occupancy 直线可达函数。

选出的目标若因任何原因仍落在 operational minimum 内，记录
`[RESIDUAL_TARGET_REJECT/too_close]` 并返回 `None`，由已有 source-seek selector 回退。

### 3.3 阶段 1 门槛判定

新增 `scripts/phase1_gate.py`，输入 phase0 baseline root 与候选 root，输出 JSON + Markdown，且失败时返回非零。

门槛必须按以下固定口径判断：

1. **完整性**：两边 `(case, seed)` 完全配对；候选每局没有 `missing_counts`；每局有
   `attribution.json`、`source_summary.json`、`clearance.csv` 和 `launch.log`。
2. **4/5 源 recall**：根据每局 `truth_count >= 4` 选择用例，不靠名称猜测；候选均值高于基线，配对置换 `p < 0.05`。
3. **precision**：矩阵事件汇总，run 级 cluster bootstrap 95% 下界 `>= 0.90`。
4. **duplicate**：候选矩阵事件汇总 duplicate rate 不高于 v31。
5. **not_reached**：按 run 配对的 `failure_counts.not_reached` 均值下降，配对置换 `p < 0.05`。
6. **算力**：解析候选 `launch.log` 中全部 `FUSE` 与 `CLEARANCE eval_ms`；证据不得缺失，p99 均 `< 100 ms`。
7. **行为健康守卫**：报告 waypoint arrival/timeout 数和每分钟 arrival rate；如果任一 run
   超过 12 arrivals/min，标为 operational FAIL。该项是本轮新增的回归守卫，不替代 spec 指标。

旧 `matrix_compare.py` 保留为通用比较器；阶段通过与否只由 `phase1_gate.py` 判定。

## 4. TDD 实现步骤

### Task 1：锁定目标距离契约

文件：

- `src/thermal_robot/tests/test_phase1_observation.py`
- `src/thermal_robot/thermal_motion_controller/thermal_motion_controller/planning.py`

步骤：

1. 先增加失败测试：configured min 小于到达半径时，有效最小距离仍严格大于到达半径；
2. 增加栅格分辨率裕量测试；
3. 实现纯函数；
4. 运行 `test_phase1_observation.py -k WaypointDistance`。

### Task 2：修复 controller 接线

文件：

- `src/thermal_robot/thermal_motion_controller/thermal_motion_controller/controller_node.py`
- `src/thermal_robot/thermal_bringup/config/params.yaml`（只在需要新增可调裕量时修改）
- `src/thermal_robot/tests/test_phase1_observation.py`

步骤：

1. 增加接线测试，证明 residual 使用 survey min/max、arrival radius 和 map resolution；
2. `_residual_waypoint()` 计算 operational minimum 并传入 planner；
3. 增加近目标运行时拒绝与 fallback；
4. 保证 `full/frontier/levy` 分支无变化。

### Task 3：实现自动门槛工具

文件：

- `src/thermal_robot/scripts/phase1_gate.py`
- `src/thermal_robot/tests/test_phase1_observation.py`

测试夹具必须覆盖：

- 4/5 源筛选不包含 static2/static3；
- recall 显著提升时 PASS，无提升时 FAIL；
- `not_reached` 没有显著下降时 FAIL；
- precision 下界不足时 FAIL；
- duplicate 高于 baseline 时 FAIL；
- 缺 artifact 或 runtime 样本时 FAIL；
- 快速 waypoint churn 时 FAIL；
- 报告同时给出原始值、p 值、阈值与判定，不能只给总 PASS/FAIL。

### Task 4：回归验证

按顺序执行：

```bash
python3 -m pytest src/thermal_robot/tests/test_phase1_observation.py -q
python3 -m pytest src/thermal_robot/tests/ -q
```

随后构建接口与运行包。构建不得覆盖或清理用户现有工作树文件。

### Task 5：短冒烟与行为诊断

先运行不超过 2 个 seed 的定向冒烟：

- open/static5/seed101，验证不再 arrival churn；
- walls/static5/seed101，验证障碍可见性与 fallback 仍工作。

冒烟验收：

- 无仿真启动缺失；
- residual/clearance/FUSE 日志存在；
- waypoint 初始距离严格大于 arrival radius；
- arrival rate `<= 12/min`；
- 机器人路径不是原地旋转或亚米级滞留；
- 冒烟 recall 仅记录，不作为阶段通过依据。

若冒烟仍显示 coverage wall，则依据轨迹和目标日志建立下一轮最小诊断，不直接调权重。

## 5. 项目负责人运行的全量矩阵

只有 Task 1–5 全部通过后才交付全量命令。候选必须使用新 out-root，禁止覆盖旧数据：

```bash
python3 -u src/thermal_robot/scripts/run_multiscenario_matrix.py \
  --preset phase0 \
  --seeds 101,102,103,104,105 \
  --strategy residual \
  --duration 120 --warmup 36 \
  --domain-start 71 \
  --out-root bags/matrix/phase1_residual_recovery_<date>
```

完成后执行：

```bash
python3 src/thermal_robot/scripts/phase1_gate.py \
  bags/matrix/phase0_full_20260611 \
  bags/matrix/phase1_residual_recovery_<date> \
  --out bags/matrix/phase1_residual_recovery_<date>/phase1_gate_report.md \
  --json-out bags/matrix/phase1_residual_recovery_<date>/phase1_gate_report.json
```

## 6. 阶段收尾规则

- 门槛全部 PASS：独立只读验收后，写 devlog、更新总体 spec 阶段 1 状态，再制定阶段 2 plan；
- 任一算法门槛 FAIL：保留产物，按失败归因进入新诊断循环，不启动阶段 2；
- 仅 artifact/runtime 缺失：先修评测工具并补跑受影响 run，不用无条件重跑全部；
- waypoint churn 仍 FAIL：视为实现缺陷，不能用 recall 偶然提升覆盖；
- 禁止因为接近目标数值而放宽 `p<0.05`、precision 下界或完整性要求。

## 7. 计划偏差记录

### 2026-07-11 独立审计后的范围修正

独立只读审计确认 waypoint 距离冲突是首因，同时发现四个会在首因修复后继续阻断门槛的问题。
这些问题有代码与旧日志证据，故在全量矩阵前扩大本轮修复范围，不改变既定统计门槛：

1. **冷场噪声被残差相对归一化放大**：所有正残差的 p95 归一化没有绝对幅值门槛，
   亚摄氏度噪声也能形成满分 `residual_mass`。增加绝对 residual floor，并测试冷场优先 unseen、
   真实大面积热残差仍优先。
2. **不可达目标仍被执行**：top-K 都不可达时当前代码返回最高分不可达目标；修为返回
   `None`，让控制器走已有安全 fallback。不可达目标不得进入 direct `/cmd_vel` fallback。
3. **单格分数不能表示相机覆盖收益**：将 residual/unseen/blocked/age 从单格值改为候选目标
   周围相机 footprint 的聚合收益；测试“稍远的大块未见区域”胜过“近处孤立像素”。
4. **Nav2 ACTIVE 期间可连续发送不同目标**：为目标替换建立显式取消/节流契约，任何状态下
   goal send 间隔不得小于配置限流，避免旧 goal `status=6` 风暴。

新增实现顺序：距离契约 → residual 绝对门槛与 footprint → 严格可达性 → Nav2 生命周期 →
门槛工具 → 冒烟。完成这些项目之前不交付全量矩阵命令作为“可运行候选”。

统计工具同时补充：阶段 1 的 recall、precision、duplicate、not_reached 主表均报告
`truth_count >= 4` cohort；全矩阵结果作为辅助表，避免 static2 的满分掩盖多源表现。

### 2026-07-11 冒烟后的 residual 最小距离修正

第一次修复后，`min_d=5.0 m` 与“目标点必须处于 SLAM 已知自由区”组合，在两个 open/static5
冒烟中使 residual planner 全部返回 `None`；系统实际走的是 legacy coverage fallback。为保持
严格大于 2.5 m arrival radius 的不变量，同时允许从当前已建图区域选择能覆盖未知 footprint 的
观测位姿，增加独立参数 `residual_waypoint_min_d`。第二次 30 秒冒烟表明 3.5 m 环带仍没有
已知自由候选，因此最终取由不变量直接推出的 3.0 m（2.5 m arrival + 0.5 m 栅格/制动裕量）。
operational minimum 仍取
`max(该参数, arrival + 栅格/制动裕量)`，不能配置回出生即到达状态。survey 最大距离仍为 14 m。

冒烟还发现 `/map` 启动期先发布 `0x0`，随后发布尺寸非零但全为 `-1` 的全未知栅格。
两者都不含任何可达性证据，不能覆盖最近一个有效 occupancy，也不能据此把所有 residual 候选
判为不可达。mapper/controller 因此只缓存“尺寸一致且至少含一个已知 cell”的地图。

### 2026-07-11 最终安全与证据来源修正

全量矩阵交付前的第二次只读审计发现两项高风险边界，因此继续扩大实现范围，但不改变算法门槛：

1. **Nav2 可规划不等于直接运动安全**：`line_reachable_plannable()` 允许路径中间经过未知区，
   原 COARSE/FRONTIER fallback 却会在 Nav2 非 ACTIVE 时直接发布线速度。新增 `/scan` 局部守卫，
   将直控明确分成“整段已知自由”与“仅由实时激光逐周期守卫”两类；两类都要求扫描新鲜且前方
   距离大于停止阈值。扫描缺失、过期或被挡时线速度归零，允许原地对准。DEPARTURE、
   COARSE_SURVEY 与 FRONTIER_NAV 共用同一守卫。该修正保留 open world 在 SLAM 全未知时的探索能力，
   同时不再把未知空间解释成整段无碰路径。
2. **断点身份不足以证明实验同质**：原 `--resume` 只检查 case/seed/strategy 和文件存在。
   checkpoint 现绑定 world/scenario/occupancy 内容哈希、全部运行参数及源码+install 运行 bundle 哈希；
   JSON/CSV、collector return code、topic counts 和 residual clearance 都需完整可解析。runner 启动前拒绝
   source/install 不一致，运行中检测 bundle 变化；不同指纹的旧 checkpoint 绝不覆盖，要求新 out-root。
   只有无 checkpoint 或同指纹但不完整的单局才会移入 `.incomplete/` 后重跑，旧证据保留且不会残留
   CSV 污染新归因。

### 2026-07-11 提交前独立代码审查修正

提交前独立审查进一步给出 1 个 Critical 与 3 个 Important，均已在全量命令交付前纳入：

1. **小矩阵可能误过 gate（Critical）**：gate 不再只比较“两边 key 相同”，而是硬验证 phase0
   固定 protocol：4 worlds × 6 scenarios × 5 seeds = 120 个唯一 `(case, seed)`，同时检查
   `matrix_summary.n_runs`、raw run 数、重复 key、preset/strategy/duration/warmup/jitter/seed 集合。
   相同的截断矩阵、重复 key 和 1-run smoke 都必须在 pairing 项 FAIL。
2. **gate 必须独立验证 provenance**：候选逐局要求 collector return code 0、`missing_counts=[]`、
   当前 runtime bundle、可重算的 run fingerprint、可复用 checkpoint，且 checkpoint 必须与
   `matrix_summary` 一致。JSON 损坏转为 artifact FAIL，不得以异常退出掩盖结论。
3. **checkpoint 必须保留原始证据**：resume 除汇总 JSON 外，逐一核对 trajectory、thermal、field、
   thermal-map、gradient、truth、cmd_vel、scan 和 clearance CSV 的存在与实际数据行数，且数据行数
   必须与 metadata/checkpoint counts 相等；source summary 与 attribution 也必须和 checkpoint 一致。
4. **严格射线不得跨过 0.05 m 栅格**：plannable/known-free reachability 改为 conservative
   supercover grid traversal，包含角点相邻格；visibility 的向量化射线步长上限改为 occupancy
   resolution 的一半。单格 occupied/unknown gap 和斜穿角点均有回归测试。

### 2026-07-11 门槛封口复审修正

对 gate 的第二、三轮独立只读复审又发现三类“评测工具自身不能证明结论”的边界，全部按
fail-closed 原则修正，不改变任何阶段统计阈值：

1. **case 名不能掩盖自定义实验路径**：baseline 与 candidate 的每个 run 都必须把 resolved
   world/scenario 路径逐项对应到 `phase0_cases()` 的规范映射；即使重新计算 fingerprint，使用更容易的
   自定义 world/scenario 也会使 pairing FAIL。
2. **语法正确不等于证据有效**：metadata、source summary、attribution、run result 以及
   matrix summary 的 gate-authoritative 字段均做 schema/type 检查。坏类型只记录 artifact、health 或
   protocol issue；无效 run 在进入 recall、not_reached、truth 与 pooled-event 统计前被隔离，不能抛异常，
   也不能以默认数值参与门槛。
3. **失败报告本身也必须可交付**：高源 cohort 为空、矩阵截断或 summary 损坏时，CLI 仍写出 Markdown
   与 JSON FAIL 报告并返回 2；缺失的 pooled statistics 显示 `n/a`，不再因渲染 `KeyError` 中断。
