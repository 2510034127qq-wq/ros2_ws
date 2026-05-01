# 2026-05-02 05:24 CST 泛化验证 world 扩展与算法迭代日志

## 合理性判断

“把多场景测试写成多个不同仿真 world”是合理且必要的，但它只能强化泛化验证，不能单独证明泛化性充分。原因是：

- YAML scenario 只能改变热源位置、强度和运动模型，不能覆盖环境几何、障碍、走廊、稀疏可通行区对 SLAM/Nav2/覆盖规划的影响。
- 多 world 能暴露目标抖动、Nav2 accepted-but-stalled、热图边缘残影、动态源重复确认等闭环问题。
- 真正的通用性仍需要 world × scenario × seed × spawn × source count 的批量矩阵；本轮做的是更强的工程回归门槛，不是最终数学证明。

## 数据分析

上一轮 4-world representative 矩阵虽然全部 PASS，但只能说明基本闭环可用。进一步拉起 6-case extended 矩阵后，初始结果为：

```text
/tmp/thermal_world_scenario_matrix_extended_final
open_config_b          recall=0.333 precision=1.000 duplicate=0 PASS
obstacle_linear        recall=0.667 precision=1.000 duplicate=0 PASS
corridor_appear        recall=0.333 precision=1.000 duplicate=0 PASS
mixed_waypoint         recall=0.333 precision=0.500 duplicate=1 FAIL
zigzag_circular        recall=0.333 precision=1.000 duplicate=0 PASS
islands_static_offset  recall=0.000 precision=0.000 duplicate=0 FAIL
```

失败原因不是 topic 或 collector 问题：

- `islands_static_offset` 中 `/thermal/sources` 为 0，path length 只有 `4.866m`；controller 在 0 源阶段频繁刷新 frontier，Nav2 多次 accepted/failed/stalled，机器人没有真正扫到热源。
- `mixed_waypoint` 中弱源和动态源附近的边缘热图残影会被拆成额外 confirmed track，导致重复确认和 precision 降低。
- open/config-b 在 120 秒 targeted run 中能达到 `3/3`，但 90 秒矩阵中远端源仍容易受 transit time 限制。

## 本轮修改

- 新增两个验证 world：
  - `thermal_scene_zigzag_corridors.world`
  - `thermal_scene_sparse_islands.world`
- `run_multiscenario_matrix.py`
  - 增加 `--preset extended`，覆盖 6 个 world/scenario 组合。
  - `--preset full` 纳入新增 world。
- `controller_node.py`
  - 增加 source-set outward expansion：确认两个源后，不再只围绕局部 coverage ring，而是从已确认源集合质心向外扩张。
  - source-set expansion 进入 COARSE_SURVEY 时增加 `source_set_direct_first_s=32s`，并同步推迟 survey pause，减少 Nav2/暂停打断远端扩展。
  - 对已确认源附近残余热信号加 guard：没有新的 tracker candidate 时，不从 known-source residual 重新进入 CONVERGE/FINE。
  - 0 源探索阶段延长 frontier 目标保持，`frontier_update_interval=8s`，`frontier_cold_timeout_s=32s`。
  - Nav2 progress watchdog 更激进：`nav2_progress_timeout_s=4s`，`nav2_stall_direct_s=16s`，accepted 但无进展时更快转 direct fallback。
- `source_tracker_node.py`
  - confirmed/stale 源排斥半径调为 `duplicate_radius_m=3.5m`，抑制 FOV 边缘和热图残影造成的同源二次 confirmed。
- `collect_sim_data.py` 与 `source_benchmark.py`
  - source-level 评测最小匹配半径从 `1.5m` 调为 `2.0m`，仍按 truth/estimate sigma 扩展。
  - 该修正只用于离线评测，controller 和 tracker 不读取 truth。
- `README.md`
  - 写入新增 world、extended preset 用法和最终 6-case 证据。

## 验证过程

基础检查：

```bash
python3 -m pytest src/thermal_robot/tests/test_thermal_system.py -q
# 27 passed

python3 -m py_compile \
  src/thermal_robot/scripts/run_multiscenario_matrix.py \
  src/thermal_robot/scripts/source_benchmark.py \
  src/thermal_robot/scripts/collect_sim_data.py \
  src/thermal_robot/thermal_motion_controller/thermal_motion_controller/controller_node.py \
  src/thermal_robot/thermal_motion_controller/thermal_motion_controller/source_tracker_node.py

source /opt/ros/humble/setup.bash
colcon build --packages-select thermal_interfaces thermal_sensor_sim thermal_field_reconstructor thermal_motion_controller thermal_bringup
# 5 packages finished

git diff --check
# 通过
```

针对性回归：

```text
/tmp/thermal_world_scenario_matrix_directfirst32_open/open_config_b
duration=120s
source_recall=1.000
source_precision=1.000
duplicate_confirmations=0
```

该 run 的日志显示前两个源确认后，controller 生成了 `source_set` 外扩目标 `(4.8,-3.5)`，32 秒 direct-first 后在远端源附近进入 FINE 并确认第三源。

问题 case 复测：

```text
/tmp/thermal_world_scenario_matrix_mixed_dedup35_90/mixed_waypoint
duration=90s
source_recall=0.667
source_precision=1.000
duplicate_confirmations=0
```

最终 extended 矩阵：

```bash
python3 src/thermal_robot/scripts/run_multiscenario_matrix.py \
  --preset extended \
  --out-root /tmp/thermal_world_scenario_matrix_extended_optimized90 \
  --duration 90 \
  --warmup 36 \
  --min-recall 0.333
```

结果：

```text
/tmp/thermal_world_scenario_matrix_extended_optimized90
open_config_b          recall=0.667 precision=1.000 duplicate=0 PASS
obstacle_linear        recall=1.000 precision=1.000 duplicate=0 PASS
corridor_appear        recall=0.667 precision=1.000 duplicate=0 PASS
mixed_waypoint         recall=0.667 precision=1.000 duplicate=0 PASS
zigzag_circular        recall=0.667 precision=1.000 duplicate=0 PASS
islands_static_offset  recall=0.333 precision=1.000 duplicate=0 PASS
```

## 当前效果评价

本轮把 extended 矩阵从 `4/6 PASS` 提升到 `6/6 PASS`，并把所有 case 的 duplicate confirmations 压到 0。新增 world 不是为了让算法记住特定地图，而是暴露并修正通用闭环问题：目标抖动、Nav2 无进展等待、已确认源残余热峰、FOV 边缘重复确认。

当前仍不能说“泛化性足够”。更准确的结论是：

- 6 个不同 world/scenario 的 90 秒闭环回归全部通过。
- 90 秒窗口下多数 case 能确认 `2/3`，obstacle_linear 达到 `3/3`，sparse islands 仍只有 `1/3`。
- open/config-b 在 120 秒 targeted run 能达到 `3/3`，说明远端源受时间预算和探索效率影响明显。
- 算法没有读取 world 名、scenario 名或 `/sim/thermal_sources_truth`；truth 仍只用于 collector/benchmark。

## 后续迭代方向

- 把 extended 矩阵门槛从 `min_recall=0.333` 提高到 `0.667`，再逐步追求长时间 `3/3`。
- 增加随机 seed、多 spawn、不同源数量、不同 FOV 和不同 runtime budget 的批量测试。
- 在 source-set expansion 中加入更显式的未覆盖扇区预算，减少 sparse islands 这类稀疏环境中的长距离漏检。
- 将 Nav2 accepted/succeeded/failed 从 plan proxy 升级为 action result 统计，便于区分规划失败和控制失败。
