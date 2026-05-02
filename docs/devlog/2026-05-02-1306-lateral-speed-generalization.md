# 2026-05-02 13:06 CST 动态多热源泛化迭代：两源横向扩展与闭环节拍

## 背景

上一轮已经把 mapper/controller 的热图坐标对齐到 odom/world 语义，并在 6 个 world/scenario 组合上达到闭环可运行。本轮继续针对多场景数据做算法迭代，重点不是为单个 world 调参，而是提升动态多热源场景中“确认两个源以后继续扩展搜索”的泛化能力。

## 保留改动

- 在 controller 的 source-set expansion 中增加源对横向扩展候选：
  - 从已确认源中选择距离最远的源对；
  - 沿源对轴线的左右法向生成候选 yaw；
  - 与原有 outward、source-gap、map-sector、coverage-phase 候选一起交给 world thermal map 的 coverage ring 评分；
  - 使用较短的 `source_set_lateral_max_d`，避免横向验证点过远。
- 小幅提高闭环探索节拍：
  - `max_linear_vel: 0.25 -> 0.28`
  - `frontier_nav_lin_vel: 0.22 -> 0.25`
  - `departure_speed: 0.22 -> 0.25`
- 保持采样确认时长和 tracker 验证等待不变，避免过早确认导致定位误差变大。

## 撤回的试验

- 相反半平面加权：曾让 obstacle 单场景达到 1.0 recall，但在 islands 场景降到 0.333，判定为不泛化，撤回。
- 单源横向扫掠/单源 coarse 环绕：能在部分 open/mixed targeted run 中改善，但 extended run 中会把 robot 带到冷区，导致 mixed 或 islands 掉到 0.333，撤回。
- 缩短 `sample_hold_s` 和 `tracker_verify_min_elapsed_s`：mixed 场景出现过早确认，source estimate 与真值 W1 超出匹配半径，导致 recall=0、duplicate=1，撤回。

## 验证

基础检查：

- `python3 -m py_compile src/thermal_robot/thermal_motion_controller/thermal_motion_controller/controller_node.py`
- `python3 -m pytest src/thermal_robot/tests/test_thermal_system.py -q`
- `colcon build --packages-select thermal_motion_controller thermal_bringup`
- `git diff --check -- src/thermal_robot/thermal_motion_controller/thermal_motion_controller/controller_node.py src/thermal_robot/thermal_bringup/config/params.yaml src/thermal_robot/tests/test_thermal_system.py`

有效运行测试：

- targeted mixed 回归：
  - artifact: `/tmp/thermal_world_scenario_matrix_lateral_fast_mixed90`
  - `mixed_waypoint`: recall=0.667, precision=1.0, duplicate=0
- extended 6 场景回归：
  - artifact: `/tmp/thermal_world_scenario_matrix_lateral_speed_extended90`
  - `open_config_b`: recall=0.667, precision=1.0, duplicate=0
  - `obstacle_linear`: recall=0.667, precision=1.0, duplicate=0
  - `corridor_appear`: recall=0.667, precision=1.0, duplicate=0
  - `mixed_waypoint`: recall=0.667, precision=1.0, duplicate=0
  - `zigzag_circular`: recall=0.667, precision=1.0, duplicate=0
  - `islands_static_offset`: recall=0.667, precision=1.0, duplicate=0

## 结论

本轮最终保留的是较稳的泛化改动：两源后基于源集合几何做横向扩展，并通过轻微提速扩大同等时间内的覆盖范围。未保留单源强方向策略，因为它在多 world 下表现不稳定。当前 90 秒 extended 门槛为 6/6 通过，但各场景仍主要停留在 2/3 recall，下一轮应继续围绕第三源发现能力做更系统的候选生成和弱源路径覆盖，而不是继续加单场景方向偏置。
