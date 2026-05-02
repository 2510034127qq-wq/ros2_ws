# 2026-05-02 18:52 CST 多源集合扇区扫掠、tracker 去重记忆与 5 源波动分析

## 背景

上一轮变量源数量矩阵显示：2 源 open 场景已经稳定到 2/2，但 4 源和 5 源仍主要停留在 2 个确认源。参考《动态多热源热导航算法改进分析报告》中“未知源数、重复发现、信息增益与覆盖预算应优先于继续追当前热点”的判断，本轮继续沿轻量工程闭环迭代，不引入 GNN/RL/PF 重模型。

实际数据暴露出两个问题：

- 4 源 mixed 场景中，source-set expansion 能找到第三个源，但旧 confirmed track 进入 stale 后可能被重新 birth 成另一个 id，造成重复确认。
- 5 源 zigzag 场景中，机器人确认前两个源后容易沿 source-set outward 或同一个北/东扇区继续扩张，南侧/西侧源长期没有进入有效视场。

## 改动

- source tracker 增加 `ever_confirmed` 记忆：
  - confirmed 源进入 stale 后，短期内仍作为重复源排斥记忆；
  - 同一源重新被观测时恢复原 track id；
  - 新增 `duplicate_memory_s=60.0`，避免永久排斥，过期后允许旧源附近重新出生候选，保留动态出生/消失的通用性。
- source-set expansion 重排候选方向：
  - 多源后优先 source-pair lateral 与 source-gap；
  - outward、map sector、coverage phase 只作为 fallback；
  - 加入 `_source_set_sweep_idx`，避免每次从同一候选开始。
- coverage ring 增加 `angle_span_rad`：
  - source-set lateral/gap 不再只是对 `preferred_yaw` 软加分，而是在扇区窗口内选点；
  - 修复日志写着 `source_lateral=-122deg`，实际目标却跑到东侧的偏差。
- `source_set_lateral_max_d` 调整为 `12.0m`，在 4 源 mixed 场景中保留第三源发现能力，同时限制横向验证点过度贴近远端地图边界。

## 验证

基础检查：

- `python3 -m py_compile src/thermal_robot/thermal_motion_controller/thermal_motion_controller/controller_node.py src/thermal_robot/thermal_motion_controller/thermal_motion_controller/planning.py src/thermal_robot/thermal_motion_controller/thermal_motion_controller/source_tracking.py src/thermal_robot/thermal_motion_controller/thermal_motion_controller/source_tracker_node.py src/thermal_robot/scripts/run_multiscenario_matrix.py`：通过
- `python3 -m pytest src/thermal_robot/tests/test_thermal_system.py -q`：30 passed
- `colcon build --packages-select thermal_motion_controller thermal_bringup`：通过

source-set 扇区窗口后的 5 源单场复测：

- artifact: `/tmp/thermal_world_scenario_matrix_zigzag5_sector_bound`
- 命令口径：`--case zigzag_dynamic_5src ... --duration 120 --warmup 36 --domain-start 205 --min-recall 0.4`
- `zigzag_dynamic_5src`: truth=5, matched=3, recall=0.600, precision=1.000, duplicate=0

最终变量源数量矩阵：

- artifact: `/tmp/thermal_world_scenario_matrix_variable120_sector_bound`
- 命令口径：`--preset variable --duration 120 --warmup 36 --domain-start 211 --min-recall 0.4`
- `open_static_2src`: truth=2, matched=2, recall=1.000, precision=1.000, duplicate=0
- `mixed_dynamic_4src`: truth=4, matched=3, recall=0.750, precision=1.000, duplicate=0
- `zigzag_dynamic_5src`: truth=5, matched=2, recall=0.400, precision=1.000, duplicate=0

## 中间失败与处理

- 将去重记忆做成永久 `ever_confirmed` 会提升短期去重，但会把旧源附近变成永久禁区，不适合动态出生/消失源。本轮改为 `duplicate_memory_s` 时间窗口，并补了“记忆过期后允许新候选出生”的纯测试。
- 一次尝试把 source-set lateral/gap 目标锚到当前机器人位置，希望减少不可达跨轴目标；5 源复测仍为 2/5，未保留该改动。
- 最终 5 源在 120s 下存在明显运行波动：单场可达 3/5，但完整 variable 矩阵中仍出现 2/5。这不是 launch/collector 问题，话题计数稳定，主要是后半段覆盖预算和重访策略不足。

## 结论

本轮相对上一轮变量源数量结果有实际提升：4 源 mixed 从 2/4 提升到 3/4，重复确认保持 0；5 源 zigzag 出现 3/5 的单场结果，但完整矩阵仍可能回落到 2/5。因此当前算法比只适配 3 源 Config-B 更通用，但还不能宣称已经充分泛化。

下一轮不要继续堆某个 world 的方向常数，应做更结构化的机制：

- 增加全局覆盖预算，确保确认 2 个源后仍强制覆盖源集合相反半平面；
- 给 stale/弱候选源建立重访队列，避免只追已确认源外侧；
- 让 benchmark 记录每个真值源的最小视距和可见时间窗，用数据区分“没走到”和“走到了但 tracker 未确认”。
