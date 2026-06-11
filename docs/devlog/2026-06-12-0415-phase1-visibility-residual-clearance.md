# Phase 1 visibility/residual/clearance delivery

## 背景

按 `docs/superpowers/specs/2026-06-10-thermal-multisource-program-design.md` 的阶段 1 和
`docs/superpowers/plans/2026-06-12-phase1-visibility-residual-clearance.md` 实现遮挡感知观测、
残差探索、清场概率 v1、配对统计工具和热相机选型文档。新决策行为挂在
`strategy=residual` 下，`full/frontier/levy` 旧策略路径保持可运行。

## 改动

- 新增共享栅格几何常数 `grid_geometry.py`，`world_occupancy.py` 和 `WorldThermalGrid` 共用中心/边长。
- 新增观测契约 `observation.py`，包含 timestamp/frame_id/sensor pose/measurement_type 和 top-down projector。
- 新增 `visibility.py` 射线可见性，mapper 订阅 SLAM `/map` 后只融合可见格。
- `WorldThermalGrid` 增加三态 view_state、blocked_count、8 扇区 view_sectors；`ThermalMap.msg` 发布新字段。
- 新增 `residual.py`，用已确认源的高斯正向预测生成残差场。
- `planning.select_residual_target` 新增残差质量 + 未见/被挡覆盖 + 可达性检查目标选择。
- 新增 `clearance.py`，实现 Poisson 空间先验 x 扇区漏检似然的清场概率 v1 与 ECE 工具。
- 控制器接入 `strategy=residual`，发布 `/thermal/clearance`，collector 记录 `clearance.csv`。
- `matrix_stats.py` 增加 paired permutation 和 run-level cluster bootstrap；新增 `matrix_compare.py`。
- 新增 `docs/hardware/2026-06-thermal-camera-selection.md`。

## 验证

纯测试：

```bash
python3 -m pytest src/thermal_robot/tests/ -q
```

结果：`83 passed in 0.33s`。旧 68 个测试未回退，新增阶段 1 测试覆盖 15 个用例。

构建：

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select thermal_interfaces
source install/setup.bash
colcon build --packages-select g1_description thermal_sensor_sim signal_preprocessor \
    thermal_field_reconstructor thermal_gradient_processor thermal_motion_controller thermal_bringup
```

结果：`thermal_interfaces` 和 7 个依赖包全部构建通过。

接口检查：

```bash
ros2 interface show thermal_interfaces/msg/ThermalMap | tail -4
```

关键输出：

```text
uint32[] visit_count
float32[] last_seen_age_s
uint8[] view_state
uint8[] view_sectors
```

统计 CLI：

```bash
python3 src/thermal_robot/scripts/matrix_compare.py --help
```

结果：正常输出用法。

### 冒烟 1：open/static2 residual

命令：

```bash
python3 src/thermal_robot/scripts/run_multiscenario_matrix.py \
  --preset phase0 --cases open__static2 --seeds 101 \
  --strategy residual --duration 60 --warmup 36 --domain-start 214 \
  --out-root /tmp/phase1_smoke_open
```

结果：`PASS open__static2 seed=101: recall=1.0 precision=1.0 dup=0`。

关键 log：

```text
[CLEARANCE] p_no_undetected=0.0120 eps=0.05 eval_ms=0.8
[CLEARANCE] p_no_undetected=0.0132 eps=0.05 eval_ms=0.8
[CLEARANCE] p_no_undetected=0.0140 eps=0.05 eval_ms=1.0
[FUSE] n=100 0.7ms occ_map=yes cells_clear=402 cells_blocked_only=0
[FUSE] n=200 0.7ms occ_map=yes cells_clear=475 cells_blocked_only=0
[FUSE] n=300 0.7ms occ_map=yes cells_clear=606 cells_blocked_only=0
```

`/tmp/phase1_smoke_open/open__static2/seed101/clearance.csv` 为 7 行，非空。

### 冒烟 2：walls/static3 residual

命令：

```bash
python3 src/thermal_robot/scripts/run_multiscenario_matrix.py \
  --preset phase0 --cases walls__static3 --seeds 101 \
  --strategy residual --duration 90 --warmup 36 --domain-start 216 \
  --out-root /tmp/phase1_smoke_walls
```

结果：`PASS walls__static3 seed=101: recall=0.333 precision=1.0 dup=0`。

关键 log：

```text
[FUSE] n=600 1.0ms occ_map=yes cells_clear=1006 cells_blocked_only=206
[FUSE] n=700 0.9ms occ_map=yes cells_clear=1085 cells_blocked_only=271
[FUSE] n=800 1.1ms occ_map=yes cells_clear=1131 cells_blocked_only=335
[FUSE] n=900 0.8ms occ_map=yes cells_clear=1218 cells_blocked_only=496
[FUSE] n=1000 1.2ms occ_map=yes cells_clear=1257 cells_blocked_only=501
[CLEARANCE] p_no_undetected=0.0120 eps=0.05 eval_ms=0.7
[CLEARANCE] p_no_undetected=0.0131 eps=0.05 eval_ms=0.9
[CLEARANCE] p_no_undetected=0.0136 eps=0.05 eval_ms=1.0
```

`cells_blocked_only > 0`，说明 SLAM occupancy 可见性掩码真实生效。`clearance.csv` 为 10 行，非空。

### 冒烟 3：open/static2 full 锚点

命令：

```bash
python3 src/thermal_robot/scripts/run_multiscenario_matrix.py \
  --preset phase0 --cases open__static2 --seeds 101 \
  --strategy full --duration 45 --warmup 36 --domain-start 218 \
  --out-root /tmp/phase1_smoke_anchor
```

结果：`PASS open__static2 seed=101: recall=1.0 precision=1.0 dup=0`。

关键 log：

```text
[FUSE] n=100 0.7ms occ_map=yes cells_clear=398 cells_blocked_only=0
[FUSE] n=200 0.7ms occ_map=yes cells_clear=473 cells_blocked_only=0
```

清理：

```bash
bash src/thermal_robot/kill_gz.sh
pgrep -a gzserver || echo "gazebo clean"
```

结果：`gazebo clean`。

## 中间失败与处理

- 初次阶段 1 测试中，`line_reachable` 测试复用了含原点占据格的地图，导致 1m 线段误判不可达；修正测试为独立墙体地图。
- 初次 residual target 测试中，残差团太稀疏，通用 `_norm_clip` 的 95 分位为 0，导致残差项被归零；在 `select_residual_target` 内改用正残差子集的 95 分位做稀疏归一化。

## 结论

阶段 1 计划的代码交付物、统计工具、硬件选型文档和三类冒烟均完成。当前冒烟证明：

- 可见性融合在障碍世界产生 blocked-only 状态；
- residual 策略可启动、发布 clearance、生成 collector artifact；
- FUSE 和 clearance eval 均远低于 100 ms 预算；
- full 锚点策略仍可跑完。

门槛评审仍需按计划跑全量矩阵，对比 `bags/matrix/phase0_full_20260611` 和 residual 全矩阵结果。
