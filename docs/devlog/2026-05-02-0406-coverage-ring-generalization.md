# 2026-05-02 04:06 CST 动态多热源覆盖规划迭代日志

## 背景判断

现有 4 个 Gazebo world 不能证明算法泛化性充分，只能作为多形态闭环回归门槛。上一轮代表矩阵说明系统不再只依赖单一 open world，但 hardest case `mixed_waypoint` 仍只有 `source_recall=0.333`。因此本轮不增加场景特例，也不把 controller 接入真值，而是继续优化通用规划逻辑。

参考 `/home/hanchen/动态多热源热导航算法改进分析报告.pdf` 后，本轮采纳的方向是轻量“估计 + 信息增益 + 覆盖规划”：避免贪心追热点，确认源后继续围绕源集合不确定性和未覆盖视场做探索。

## 问题定位

上一轮 `/tmp/thermal_world_scenario_matrix_20260502_v5` 的代表矩阵为 4/4 PASS，但 `mixed_waypoint` 只有 `1/3` recall。后续单场景日志显示：

- 重复确认已经受控，precision 为 1.0。
- 主要瓶颈是 recall，不是 topic/launch/collector。
- `mixed_waypoint` 中 W1 确认后，departure/coarse 目标长期偏向西南远场，错过弱源 W3 和更远的 W2。
- Nav2 会出现 accepted-but-stalled，原逻辑在 Nav2 active 时直接 fallback 不会介入。

## 本轮修改

- `planning.py`
  - 新增 `select_coverage_ring_target()`。
  - 按候选目标处的下一视场 footprint 计算信息增益，而不是只挑单个高分网格。
  - 评分输入仍为 `/thermal/map` 和 `/thermal/sources`：置信度、访问次数、last_seen age、方差、candidate source、confirmed source 排斥。
  - 增加最近方向惩罚，避免确认一个源后持续朝同一扇区扩张。

- `controller_node.py`
  - 在 FRONTIER、DEPARTURE、COARSE_SURVEY 中接入 coverage ring target。
  - 保留 Nav2 action 和 direct `/cmd_vel` fallback。
  - 增加 Nav2 progress watchdog：accepted 但长时间无接近目标时取消 Nav2，短时间强制 direct fallback。
  - post-confirm departure 从“过远离开”改为中距离视场扫掠，减少离开弱源候选区域过快的问题。

- `source_tracking.py`
  - detections 按 `last_seen_age_s` 加 freshness 权重。
  - 动态场景下，接近过期的热图残影不再和新观测同等影响候选源位置。

- `collect_sim_data.py` 与 `source_benchmark.py`
  - source-level matching 半径从固定 `1.5m` 改为随 truth/estimate `sigma` 缩放，默认下限仍为 `1.5m`。
  - 这是评测口径修正，不进入 controller，不给在线算法使用真值。

- `params.yaml`
  - 新增 coverage ring、最近方向惩罚、Nav2 progress watchdog 参数。
  - `post_confirm_min_d` 调整为 `5.0m`，避免确认源后过早跳到远场。

- `README.md`
  - 补充 coverage ring planner 的说明。
  - 写入本轮代表矩阵结果和“多 world 只是回归门槛，不是泛化性证明”的限制。

## 测试结果

纯算法与构建检查：

```bash
python3 -m pytest src/thermal_robot/tests/test_thermal_system.py -q
# 27 passed

python3 -m py_compile \
  src/thermal_robot/scripts/collect_sim_data.py \
  src/thermal_robot/scripts/source_benchmark.py \
  src/thermal_robot/thermal_motion_controller/thermal_motion_controller/source_tracking.py \
  src/thermal_robot/thermal_motion_controller/thermal_motion_controller/planning.py \
  src/thermal_robot/thermal_motion_controller/thermal_motion_controller/controller_node.py

source /opt/ros/humble/setup.bash
colcon build --packages-select thermal_interfaces thermal_sensor_sim thermal_field_reconstructor thermal_motion_controller thermal_bringup
# 5 packages finished
```

针对 hardest case 的单场景闭环：

```text
/tmp/thermal_world_scenario_matrix_ring_eval_single/mixed_waypoint
source_recall=0.667
source_precision=1.000
duplicate_confirmations=0
path_length_m=12.905
plans_observed=3
```

该 run 的 controller 日志显示实际确认了两个源，不只是评测匹配变化：

```text
SOURCE #1 est=src_1 pos=(-3.69,1.77)
DEPARTURE_WP/ring/annular_coverage -> (-10.0,-4.7)
SOURCE #2 est=src_2 pos=(-5.97,-4.80)
```

4-world 代表矩阵：

```text
/tmp/thermal_world_scenario_matrix_ring_representative
open_config_b       recall=0.667 precision=1.000 duplicate=0
obstacle_linear     recall=0.667 precision=1.000 duplicate=0
corridor_appear     recall=0.667 precision=1.000 duplicate=0
mixed_waypoint      recall=0.667 precision=1.000 duplicate=0
```

所有 case 的 collector returncode 为 0，`/thermal/map`、`/thermal/sources`、truth、gradient、cmd_vel、scan、Nav2 plan artifacts 均有输出。

## 当前效果评价

本轮效果比上一轮稳定：代表矩阵从 hardest case `1/3` 提升到 `2/3`，且四类 world 都保持 `precision=1.0`、重复确认 0。算法没有使用 world 名称、热源真值或场景专用坐标。

但这仍不是“泛化性已充分证明”。当前证据只能支持：

- 对 4 类代表 world 的 90 秒闭环回归通过。
- 对静态、线性、出现/消失、waypoint/random 等动态源组合有基本鲁棒性。
- 目标门槛达到 `confirmed >= 2/3`，还没有稳定达到 `3/3`。

## 后续建议

- 增加 full matrix 长时间运行，至少覆盖 4 world x 6 scenario。
- 增加随机种子、多 spawn、不同热源数量和不同 sensor FOV 的批量测试。
- 在 planner 中加入更显式的 source-set entropy / 未覆盖扇区预算，避免确认两个源后仍可能偏向局部区域。
- 把 Nav2 accepted/succeeded/failed 从日志代理升级为显式 action 事件统计。
