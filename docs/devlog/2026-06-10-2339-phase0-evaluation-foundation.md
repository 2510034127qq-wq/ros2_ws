# 2026-06-10 23:39 阶段0评测地基: 多seed统计、失败归因、基线挂架

## 背景

依据 `docs/superpowers/specs/2026-06-10-thermal-multisource-program-design.md` 阶段0，本轮交付可复现、多 seed、可归因的评测基础，供后续遮挡感知观测、残差探索、重访调度和慢层信念系统使用。

## 改动

- 新增 `scripts/matrix_stats.py`：多 run 聚合、样本标准差、Mann-Whitney U、Markdown 报告渲染。
- 新增 `scripts/world_occupancy.py`：从 Gazebo SDF world 解析 box collision，生成真值占据栅格并提供 line-of-sight 检查。
- 生成 6 个世界的 `thermal_bringup/worlds/occupancy/*.npz` 真值占据栅格。
- 新增 `thermal_bringup/config/scenarios/static_five_sources.yaml`。
- 在 `thermal_sensor_sim.scenario` 增加 `apply_run_seed()`，支持运行级 seed 扰动 source seed、位置和幅值。
- 在 `sensor_node` 接入 `scenario_seed` 与 `scenario_jitter_std_m`，并让 sensor noise 在 run_seed > 0 时跟随运行 seed。
- 在 `controller_node` 接入 `random_seed` 与 `strategy=full|frontier|levy`，只替换 COARSE 探索目标选择，FINE 梯度/确认逻辑保持一致。
- 在 `sim_nav_slam_launch.py` 暴露并透传 `run_seed`、`strategy`、`scenario_jitter_std_m`。
- 新增 `scripts/attribution.py`：复用 collector 产物和真值占据栅格，输出每源归因和 `failure_counts`。
- 重写 `scripts/run_multiscenario_matrix.py`：新增 `phase0` preset、`--seeds`、`--worlds/--cases`、`--strategy`、`--jitter`、自动归因、`matrix_summary.json`、`matrix_report.md` 和 `--health-only`。
- 新增 `tests/test_phase0_evaluation.py`，覆盖阶段0纯模块和 runner 配置；保留旧 runner preset 符号兼容。

## 验证

- `python3 -m py_compile ...`: 通过。
- `python3 -m pytest src/thermal_robot/tests/ -q`: `64 passed`。
- `colcon build --packages-select thermal_sensor_sim thermal_motion_controller thermal_bringup`: 通过。
- `ros2 launch thermal_bringup sim_nav_slam_launch.py --show-args`: 通过，列出 `run_seed`、`strategy`、`scenario_jitter_std_m`。
- v31 anchor: `git tag -f v31-anchor && git tag -l v31-anchor` 输出 `v31-anchor`。

## 冒烟结果

### full 策略单 run

命令:

```bash
python3 src/thermal_robot/scripts/run_multiscenario_matrix.py \
  --preset phase0 --cases open__static2 --seeds 101 \
  --duration 45 --warmup 36 --domain-start 201 \
  --out-root /tmp/phase0_smoke
```

结果:

- 退出码 0。
- `PASS open__static2 seed=101`
- `source_recall=0.5`, `source_precision=1.0`, `duplicate_confirmations=0`
- 产物存在：`/tmp/phase0_smoke/open__static2/seed101/attribution.json`、`source_summary.json`、`metadata.json`、`/tmp/phase0_smoke/matrix_summary.json`、`matrix_report.md`
- `launch.log` 命中 `run_seed=101`
- attribution failure counts: `not_reached=1`, `occluded=0`, `timing_missed=0`, `not_confirmed=0`

### levy 策略单 run

命令:

```bash
python3 src/thermal_robot/scripts/run_multiscenario_matrix.py \
  --preset phase0 --cases open__static2 --seeds 101 \
  --strategy levy --duration 45 --warmup 36 --domain-start 203 \
  --out-root /tmp/phase0_smoke_levy
```

结果:

- 退出码 0。
- `PASS open__static2 seed=101`
- `source_recall=1.0`, `source_precision=1.0`, `duplicate_confirmations=0`
- `launch.log` 命中 `strategy=levy random_seed=101`
- attribution failure counts: `not_reached=0`, `occluded=0`, `timing_missed=0`, `not_confirmed=0`

## 障碍世界 Nav2 健康

命令:

```bash
python3 src/thermal_robot/scripts/run_multiscenario_matrix.py \
  --preset phase0 --cases boxes__static3,walls__static3,mixed__static3 --seeds 101 \
  --health-only --duration 40 --warmup 40 --domain-start 205 \
  --out-root /tmp/phase0_nav2_health
```

结果: `2 / 3` passed。

- `boxes__static3`: PASS. Lifecycle active; TF true; cmd_vel true count 87; plan true count 1.
- `walls__static3`: PASS. Lifecycle active; TF true; cmd_vel true count 347; plan true count 1.
- `mixed__static3`: FAIL. Lifecycle active for `/bt_navigator`, `/controller_server`, `/planner_server`; TF true; cmd_vel true count 408; `/plan` false, `plan_count=0`.

结论: 这是 spec 风险 1 的实测证据。mixed obstacle world 中 Nav2 节点能进入 active 且控制输出存在，但 health 窗口内没有观测到 planner path。阶段 1 的可达性检查需要保留直线无碰降级和直接 `/cmd_vel` fallback，不能假设所有障碍世界都有稳定 Nav2 plan 事件。

## 中间失败与处理

- `ros2 launch --show-args` 首次失败，因为 ROS 默认写 `/home/hanchen/.ros/log`，当前沙箱只读。设置 `ROS_LOG_DIR=/tmp/ros2_ws_launch_logs` 后通过。
- 旧测试 `T-PY23` 要求 runner 文本保留 `ROS_DOMAIN_ID must stay <= 232` 保护；已补回 domain_start 上限检查，并用可用 domain span 串行复用 domain id。
- `kill_gz.sh` 杀掉残留 `gzserver` 后返回 137；单独 `pgrep -a gzserver || echo "gazebo clean"` 确认 `gazebo clean`。

## 结论与下一步

阶段0门槛核对:

- [x] 扩展矩阵多 seed 指标可复现：runner 和统计报告已交付，单 seed 冒烟验证链路，全量 5 seed 矩阵留待门槛评审运行。
- [x] 基线挂架可运行：`full` 与 `levy` 冒烟通过，`frontier` 使用同一 `strategy` 参数链路。
- [x] 每局自动生成归因报告：冒烟 run 已生成 `attribution.json` 并汇总进矩阵报告。
- [x] 障碍世界 Nav2 风险已实测：mixed world health-only 失败已记录。

下一步: 运行完整 phase0 矩阵，例如 `--preset phase0 --seeds 101,102,103,104,105`，并用 `matrix_summary.json` 与 `matrix_report.md` 提交阶段0门槛评审。
