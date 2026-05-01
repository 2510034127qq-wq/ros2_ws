# 2026-05-02 03:26 CST 多 world 动态多热源闭环测试与泛化修正

## 背景

本轮目标不是继续只在单一 `thermal_scene_nav.world` 中切换热源 YAML，而是把测试扩展成“仿真环境 world × 热源动态 scenario”的组合。这样可以同时覆盖环境几何泛化和热源动态泛化，避免算法只对某个固定仿真场景有效。

## 主要改动

- `sim_nav_slam_launch.py` 增加 `world_file` launch 参数，默认仍为原来的 `thermal_scene_nav.world`。
- 新增三个不同环境几何的 Gazebo world：
  - `thermal_scene_obstacle_field.world`
  - `thermal_scene_corridor_rooms.world`
  - `thermal_scene_mixed_rooms.world`
- 新增 `scripts/run_multiscenario_matrix.py`，按 case 自动执行：
  - 启动 Gazebo / SLAM / Nav2 / thermal pipeline
  - 启动 collector
  - 采集 `/thermal/map`、`/thermal/sources`、truth、`/cmd_vel`、`/plan`
  - 清理进程组
  - 输出 `matrix_summary.json`
- `SourceTrackerCore` 增加动态源修正：
  - 同一张热图中同一 track 只计一次观测，避免相邻局部峰瞬间刷 confirmed。
  - 增加 `update_alpha_min`，长期观测的动态源不会因融合权重趋近 0 而冻结在旧位置。
  - 使用 `ThermalMap.last_seen_age_s` 过滤过期热图峰，避免移动源离开后旧热图 ghost 持续刷新 confirmed track。
  - 修正无观测衰减为增量时间衰减，避免概率被累计时间反复相乘。
- source-level summary 改为“任意 confirmed 观测是否覆盖对应时刻 truth”，更适合动态源评测；重复确认仍按唯一 estimate id 统计。

## 测试结果

### 纯测试与构建

- `python3 -m pytest src/thermal_robot/tests/test_thermal_system.py -q`
  - `23 passed in 0.10s`
- `colcon build --packages-select thermal_interfaces thermal_sensor_sim thermal_field_reconstructor thermal_motion_controller thermal_bringup`
  - 5 个包构建通过
- `git diff --check`
  - 通过
- 运行后确认无残留 `gzserver/gzclient`

### 多 world × 多 scenario 闭环矩阵

命令：

```bash
python3 src/thermal_robot/scripts/run_multiscenario_matrix.py \
  --preset representative \
  --out-root /tmp/thermal_world_scenario_matrix_20260502_v5 \
  --duration 90 \
  --warmup 36 \
  --min-recall 0.333
```

结果：`4/4 PASS`，全部 case `source_precision=1.0`，`duplicate_confirmations=0`。

| case | world | scenario | recall | precision | duplicates | path |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| open_config_b | `thermal_scene_nav.world` | `config_b_sources.yaml` | 0.667 | 1.0 | 0 | 14.322m |
| obstacle_linear | `thermal_scene_obstacle_field.world` | `dynamic_linear_sources.yaml` | 0.667 | 1.0 | 0 | 13.972m |
| corridor_appear | `thermal_scene_corridor_rooms.world` | `dynamic_appear_disappear_sources.yaml` | 0.667 | 1.0 | 0 | 11.069m |
| mixed_waypoint | `thermal_scene_mixed_rooms.world` | `dynamic_waypoint_random_sources.yaml` | 0.333 | 1.0 | 0 | 6.342m |

最终矩阵目录：

- `/tmp/thermal_world_scenario_matrix_20260502_v5/matrix_summary.json`

## 关键发现

- 只切换 `scenario_file` 不足以证明泛化；world 几何变化会暴露 Nav2、SLAM、覆盖规划和动态源跟踪的耦合问题。
- 动态源不能用静态源的“长期均值融合”方式处理，否则 track 会冻结在早期位置。
- world thermal map 对动态源有 ghost 风险，source tracker 必须使用观测年龄过滤。
- 动态源评测不能只看最终位置；更合理的是判断每个 truth source 是否曾被 confirmed estimate 在对应时刻覆盖。

## 剩余问题

- 代表矩阵已经全 PASS，但 `mixed_waypoint` 仍只有 `1/3` recall。它说明随机/航点动态源加复杂环境时，当前探索覆盖仍不足。
- 当前矩阵门槛是 `min_recall=0.333`，用于保证完整闭环不退化；下一轮应把代表矩阵目标提高到稳定 `>= 0.667`，再逐步追求 `3/3`。
- mixed world 中仍能看到 Nav2 TF 过旧日志，direct `/cmd_vel` fallback 保证了闭环不中断，但 Nav2 时间/TF 稳定性仍值得单独处理。
