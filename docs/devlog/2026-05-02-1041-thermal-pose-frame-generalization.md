# 2026-05-02 10:41 CST 热导航坐标一致性与后确认探索优化日志

## 目标

继续优化动态多热源热导航算法，要求不是只让某一个仿真 world 变好，而是在多 world、多 scenario 的闭环矩阵里提高通用性。本轮重点处理上一轮遗留的 `islands_static_offset` 短板，并把验证门槛从 `min_recall=0.333` 提高到 `0.667`。

## 数据分析

上一轮 extended 90 秒矩阵为 6/6 PASS，但门槛较低：

```text
/tmp/thermal_world_scenario_matrix_extended_optimized90
open_config_b          recall=0.667 precision=1.000 duplicate=0
obstacle_linear        recall=1.000 precision=1.000 duplicate=0
corridor_appear        recall=0.667 precision=1.000 duplicate=0
mixed_waypoint         recall=0.667 precision=1.000 duplicate=0
zigzag_circular        recall=0.667 precision=1.000 duplicate=0
islands_static_offset  recall=0.333 precision=1.000 duplicate=0
```

针对 `islands_static_offset` 的 90 秒与 120 秒 run 暴露了两个问题：

- 90 秒 run 中只确认 `S3_southwest`，首源确认后又经历 `RELOCATE -> DEPARTURE -> COARSE`，真正进入远场覆盖时剩余时间不足。
- 120 秒旧算法虽然能继续发现第二个热峰，但该热峰定位在 `(-1.45, 0.96)` 附近，和真值 `S1_north(-3, 6)` / `S2_east(4.5, 1.5)` 都不匹配，导致 `precision=0.5`、`duplicate=1`。

进一步对比 `/tmp/thermal_world_scenario_matrix_islands_depwatch120` 中的 `slam_trajectory.csv` 与 `trajectory.csv`：

- 同一时刻 SLAM/map 位姿约为 `(2.64, 1.89)`，controller 按 `map + spawn` 解释为 world `(-3.36, 1.89)`。
- odom/Gazebo 物理位姿约为 world `(-5.19, 4.32)`，此时机器人实际靠近 `S1_north` 的热场。
- thermal image 由 Gazebo/odom 物理位姿生成，但 `thermal_mapper_node` 和 controller 采用 `map + spawn` 投影和规划，SLAM 漂移会把真实热源投影到错误 world 坐标，形成假源。

结论：问题不是继续给单个 frontier 目标调参，而是热图投影、source tracker、controller 使用的 thermal world frame 必须和热相机生成帧一致。

## 修改内容

- `thermal_mapper_node`
  - 新增 `pose_source` 参数，默认和配置均为 `odom`。
  - 在 `pose_source=odom` 时使用 `spawn + /odom` 投影 `/thermal/filtered` 到 `/thermal/map`。
  - 保留 TF 逻辑作为可选模式。

- `controller_node`
  - 新增 `thermal_pose_source` 参数，默认和配置均为 `odom`。
  - controller 的热导航 world pose 使用 `spawn + /odom`，与 `/thermal/map` 和 `/thermal/sources` 保持同一物理坐标系。
  - Nav2 goal 不再只用固定 `world - spawn` 转换；当 TF 可用且热导航使用 odom frame 时，使用当前 `map->base_link` 位姿和 yaw delta 把 odom-world 目标转换到 map frame。
  - 确认源后新增 `post_confirm_immediate_departure=true`，跳过固定 2m `RELOCATE`，直接计算信息增益/覆盖环 departure 目标，减少无信息移动时间。
  - 给 `DEPARTURE` 增加 progress watchdog：若目标距离长时间无改善，则转入 `COARSE_SURVEY`，避免卡在不可达 departure waypoint。
  - `departure_arrived/stalled/timeout` 后进入 COARSE 时增加 `post_confirm_direct_first_s=24s`，减少 Nav2/暂停打断刚确认源后的远场覆盖。

- `params.yaml`
  - 写入 `thermal_mapper_node.pose_source: "odom"`。
  - 写入 `controller_node.thermal_pose_source: "odom"`。
  - 写入后确认探索和 departure progress 参数。

- `README.md`
  - 写入 odom-aligned thermal frame 的运行说明。
  - 更新 latest extended evidence。

- `test_thermal_system.py`
  - 增加配置/代码字符串回归，确保 odom thermal frame、实时 Nav2 map 转换、immediate departure 和 departure progress watchdog 不被误删。

## 验证过程

基础检查：

```bash
python3 -m py_compile \
  src/thermal_robot/thermal_motion_controller/thermal_motion_controller/controller_node.py \
  src/thermal_robot/thermal_field_reconstructor/thermal_field_reconstructor/thermal_mapper_node.py

python3 -m pytest src/thermal_robot/tests/test_thermal_system.py -q
# 27 passed

source /opt/ros/humble/setup.bash
colcon build --packages-select thermal_interfaces thermal_sensor_sim thermal_field_reconstructor thermal_motion_controller thermal_bringup
# 5 packages finished

git diff --check
# 通过
```

问题 case targeted run：

```bash
python3 src/thermal_robot/scripts/run_multiscenario_matrix.py \
  --case islands_static_offset:src/thermal_robot/thermal_bringup/worlds/thermal_scene_sparse_islands.world:src/thermal_robot/thermal_bringup/config/scenarios/static_offset_sources.yaml \
  --out-root /tmp/thermal_world_scenario_matrix_islands_odom_immediate90 \
  --duration 90 \
  --warmup 36 \
  --min-recall 0.0
```

结果：

```text
/tmp/thermal_world_scenario_matrix_islands_odom_immediate90/islands_static_offset
source_recall=0.667
source_precision=1.000
duplicate_confirmations=0
matched: S3_southwest, S2_east
```

关键日志显示：

- `thermal_mapper_node ... pose_source=odom`
- `controller_node ... pose_source=odom`
- `AT_PEAK->DEPARTURE` 在首源确认后直接进入 departure，不再先固定 relocate。
- 第二源估计为 `(3.75, 1.10)`，与 `S2_east(4.5, 1.5)` 误差 `0.647m`，假源消失。

最终 extended 矩阵：

```bash
python3 src/thermal_robot/scripts/run_multiscenario_matrix.py \
  --preset extended \
  --out-root /tmp/thermal_world_scenario_matrix_extended_odom_immediate90 \
  --duration 90 \
  --warmup 36 \
  --min-recall 0.667
```

结果：

```text
/tmp/thermal_world_scenario_matrix_extended_odom_immediate90
open_config_b          recall=1.000 precision=1.000 duplicate=0 PASS
obstacle_linear        recall=0.667 precision=1.000 duplicate=0 PASS
corridor_appear        recall=0.667 precision=1.000 duplicate=0 PASS
mixed_waypoint         recall=0.667 precision=1.000 duplicate=0 PASS
zigzag_circular        recall=1.000 precision=1.000 duplicate=0 PASS
islands_static_offset  recall=0.667 precision=1.000 duplicate=0 PASS
```

所有 case 的 collector returncode 为 0，`/thermal/map`、`/thermal/sources`、truth、gradient、cmd_vel、scan、Nav2 plan artifacts 均有输出。

## 当前效果评价

本轮把 extended 矩阵门槛从 `min_recall=0.333` 提高到 `0.667` 后仍然 6/6 PASS，并且所有 case 的 precision 都为 1.0、重复确认都为 0。相比上一轮：

- `open_config_b`: `0.667 -> 1.000`
- `zigzag_circular`: `0.667 -> 1.000`
- `islands_static_offset`: `0.333 -> 0.667`
- 其他复杂/动态 case 保持 `0.667` 且没有重复确认回退。

这说明本轮优化不是针对单一 sparse islands world 的硬编码，而是修正了热图/控制器坐标帧不一致这一通用闭环问题。controller 仍不订阅 `/sim/thermal_sources_truth`，truth 只用于 collector/benchmark。

## 仍然存在的限制

- 90 秒窗口下还有 4 个 case 是 `2/3`，尚未稳定达到 `3/3`。
- 当前 extended 矩阵是固定 seed、固定 spawn、固定 3 源；还不能宣称充分泛化。
- 后续需要 `full` world × scenario 矩阵、随机 seed、多 spawn、不同源数量、不同 FOV 和更长 runtime budget。
- Nav2 统计仍主要依赖 plan proxy，后续应补 action accepted/succeeded/failed 的显式记录。
