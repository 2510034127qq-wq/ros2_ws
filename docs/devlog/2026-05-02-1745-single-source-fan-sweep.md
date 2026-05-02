# 2026-05-02 17:45 CST 单源后扇形扫掠与多源矩阵 runner 稳定性修复

## 背景

上一轮已经去掉固定 3 源截断，并加入 2、4、5 源矩阵，但实测暴露出一个更具体的问题：首个源确认后，coarse planner 容易沿同一半平面继续外扩。典型表现是 2 源 open 场景只确认北侧首源，随后持续向西南或单一外向方向探索，错过东侧第二源。这不是某个 world 的特例，而是单源后缺少围绕已确认源的扇区覆盖节奏。

## 改动

- controller 增加单源后 source-anchored fan sweep：
  - 以第一个已确认源为锚点；
  - 按斜向、反斜向、外向等扇区轮换选点；
  - 避免每次都被同一个最高分方向吸走。
- 多源后 coarse waypoint 继续使用 source-set expansion，而不是只在首次 departure 时使用，提升源集合外侧覆盖连续性。
- `source_set_lateral_max_d` 从 `10.0m` 放宽到 `14.0m`，避免横向跨轴验证点因为从当前机器人位置看太远而被过滤。
- COARSE_SURVEY 增加源集合半径回收保护，防止无热信号时长期漂到已发现源集合外的单一远端半平面。
- `run_multiscenario_matrix.py` 增加运行隔离：
  - 每个 case 使用独立 `GAZEBO_MASTER_URI`；
  - 每个 case 使用临时 `HOME` 和 `GAZEBO_LOG_PATH`，不写用户主目录；
  - 显式使用本地 `GAZEBO_MODEL_PATH`，禁用远程模型数据库；
  - 对 `ROS_DOMAIN_ID > 232` 做保护，避免 FastDDS 端口计算失败。

## 验证

基础检查：

- `python3 -m py_compile src/thermal_robot/scripts/run_multiscenario_matrix.py src/thermal_robot/thermal_motion_controller/thermal_motion_controller/controller_node.py`：通过
- `python3 -m pytest src/thermal_robot/tests/test_thermal_system.py -q`：27 passed
- `colcon build --packages-select thermal_motion_controller thermal_bringup`：通过

非 3 源闭环矩阵：

- artifact: `/tmp/thermal_world_scenario_matrix_variable120_fan_ports`
- 命令口径：`--preset variable --duration 120 --warmup 36 --domain-start 151 --min-recall 0.4`
- `open_static_2src`: truth=2, matched=2, recall=1.000, precision=1.000, duplicate=0
- `mixed_dynamic_4src`: truth=4, matched=2, recall=0.500, precision=1.000, duplicate=0
- `zigzag_dynamic_5src`: truth=5, matched=2, recall=0.400, precision=1.000, duplicate=0

3 源回归：

- artifact: `/tmp/thermal_world_scenario_matrix_configb_fan_ports`
- 命令口径：`--case open_config_b ... --duration 90 --warmup 36 --domain-start 181 --min-recall 0.667`
- `open_config_b`: truth=3, matched=2, recall=0.667, precision=1.000, duplicate=0

## 中间失败与处理

- 最初的“粗搜索半径回收 + 多源 coarse expansion”让 4 源场景退化到 1/4，原因是仍没有解决首源后的方向偏置；该结果未作为最终算法证据。
- 后续测试一度出现 `path_length=0`、无 `/odom`、无 `/scan`。排查后不是 controller 算法失败，而是 Gazebo 11345 端口冲突、临时 HOME 下缺少本地模型路径、以及过高 `ROS_DOMAIN_ID` 造成的启动失败。runner 已补齐隔离和保护。

## 结论

本轮优化把 2 源 open 场景从上一轮的 1/2 提升到 2/2，同时保持 4 源、5 源变量矩阵通过，并保留 3 源 Config-B 回归通过。当前算法更接近通用动态多热源导航：它不依赖真值、不固定源数，也不只在 3 源配置上闭环。

仍然存在的主要瓶颈是后续源召回率：4 源和 5 源仍只稳定确认 2 个源。下一轮应继续提升远端/弱源/间歇源发现率，例如引入更明确的全局覆盖预算、候选源重访队列、以及按未覆盖热图区域的长期探索记忆，而不是继续增加特定场景规则。
