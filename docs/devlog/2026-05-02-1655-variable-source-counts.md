# 2026-05-02 16:55 CST 非 3 源动态多热源验证与开放源数改造

## 背景

上一轮 extended 6 场景虽然覆盖了多个 Gazebo world 和动态类型，但每个 thermal scenario 都仍是 3 个热源。这会让验证结论偏窄：算法可能只是适配了 3 源发现节奏，而不是真正支持动态多热源导航。本轮目标是把 2、4、5 源纳入全流程测试，并去掉 runtime 中的固定 3 源截断。

## 改动

- `sensor_node` 在显式传入 `scenario_file` 时加载 YAML 中的全部热源；`num_sources` 只保留给空 `scenario_file` 的 Config-B fallback 使用。
- `scenario.py` 支持 `num_sources <= 0` 表示不截断源集合，同时保留正数截断能力用于单测。
- controller 默认 `num_sources: -1`，不再找到 3 个源就进入 DONE。
- 将开放源数模式下的 `no_new_source_timeout` 放宽到 180 秒，避免 4/5 源场景在发现第一个源后过早停止。
- 新增非 3 源场景：
  - `static_two_sources.yaml`
  - `dynamic_four_sources.yaml`
  - `dynamic_five_sources.yaml`
- `run_multiscenario_matrix.py` 新增 `variable` preset：
  - open world + 2 static sources
  - mixed rooms world + 4 dynamic/static sources
  - zigzag corridors world + 5 dynamic/static sources
- source-level summary 和 matrix summary 增加 `truth_count`、`matched_count`、`confirmed_count`，便于直接检查源数覆盖。

## 验证

基础检查：

- `python3 -m py_compile ...`：通过
- `python3 -m pytest src/thermal_robot/tests/test_thermal_system.py -q`：27 passed
- `colcon build --packages-select thermal_sensor_sim thermal_motion_controller thermal_bringup`：通过
- `git diff --check`：通过

非 3 源闭环矩阵：

- artifact: `/tmp/thermal_world_scenario_matrix_variable120_openended`
- 命令口径：`--preset variable --duration 120 --warmup 36 --min-recall 0.4`
- `open_static_2src`: truth=2, matched=1, recall=0.500, precision=1.000, duplicate=0
- `mixed_dynamic_4src`: truth=4, matched=2, recall=0.500, precision=1.000, duplicate=0
- `zigzag_dynamic_5src`: truth=5, matched=2, recall=0.400, precision=1.000, duplicate=0

3 源基线回归：

- artifact: `/tmp/thermal_world_scenario_matrix_extended_openended90`
- 6 个 extended case 中 5 个通过；`islands_static_offset` 单次 run 因首源 67 秒才确认，90 秒窗口内只到 1/3。
- 对 `islands_static_offset` 单独复跑 90 秒通过：
  - artifact: `/tmp/thermal_world_scenario_matrix_islands_openended90`
  - truth=3, matched=2, recall=0.667, precision=1.000, duplicate=0

## 结论

当前系统已经不再被 3 源配置截断：scenario 文件可以承载 2、4、5 个热源，controller 也不会在第 3 个源处自动停止。非 3 源矩阵完成了全流程闭环，但召回仍偏低：2 源只稳定到 1/2，4 源到 2/4，5 源到 2/5。下一步优化不应再停留在“能不能跑非 3 源”，而应提升后续源发现率，特别是远端源、间歇源和弱源的持续覆盖策略。
