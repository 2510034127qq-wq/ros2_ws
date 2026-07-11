# Phase 1 coverage wall recovery

## 背景

阶段 1 首轮全量矩阵没有通过总体设计 §4 门槛。原 `matrix_compare.py` 汇总全部 120 局后得到
recall 0.625 → 0.646、`p=0.11989`，但门槛要求的是 4/5 源场景。新增专用 gate 后重算：

- 4/5 源 60 个配对 run：recall 0.5225 → 0.5408，`p=0.38363`；
- 4/5 源 `not_reached/run`：2.2833 → 2.2000，`p=0.50567`；
- high-source precision 95% 下界 0.933；
- duplicate rate 0.0286 → 0.0214；
- 旧 residual 全矩阵主 `launch.log` 有 5176 次 COARSE arrival，v31 为 165 次。

因此阶段 1 仍为 FAIL。运行日志进一步证明首要执行缺陷：residual planner 默认最小目标距离
2.5 m，与控制器 arrival radius 2.5 m 相等，导致目标生成后下一 tick 立即判定到达。

详细冻结计划见 `docs/superpowers/plans/2026-07-11-phase1-coverage-wall-recovery.md`。

## 改动

### residual 覆盖规划

- 新增 operational waypoint minimum：严格大于 arrival radius，并包含栅格/制动裕量；
- residual 目标距离范围改为 3.0–14.0 m；
- 低于 1.5°C 的正残差不再通过相对归一化放大为满分；
- residual/unseen/blocked/age 改为目标周围 2.0 m footprint 聚合收益；
- top-K 候选增加空间去重，避免全部候选聚集在同一不可达小区域；
- candidate mask 在评分前过滤无效 endpoint；全部候选不可达时返回 `None`，不再返回
  `reachable=False` 的最高分目标；
- 增加 `[RESIDUAL_PLAN]` 规划耗时、候选数量和距离边界日志。

### occupancy 与执行安全

- 新增“已知自由 endpoint + 路径不穿过已知障碍”的 Nav2-preferred 可达性检查；
- 另保留“完整路径必须已知自由”的严格检查，供直接运动安全路径使用；
- 两类严格 reachability 使用 conservative supercover 栅格遍历，不能再用 0.2 m 采样跨过
  0.05 m SLAM 单格障碍/未知缝隙；向量化 visibility 射线步长不大于 occupancy resolution 的一半；
- mapper/controller 不再缓存 `/map` 启动期的 `0x0` 或全未知栅格；只有尺寸一致且至少含一个
  已知 cell 的 occupancy 才能替换最近有效地图；
- Nav2 goal 发送限流覆盖所有状态；ACTIVE 目标替换先显式 cancel；generation token 阻止旧 goal
  callback 覆盖新 goal 状态。
- `/scan` 直控守卫覆盖 DEPARTURE、COARSE_SURVEY、FRONTIER_NAV：完整已知自由路径记录为
  `known_free`，未知中间区只允许 `scan_guarded` 的逐周期局部运动；扫描缺失、超过 0.6 s 或前方
  ±60° 扇区内小于等于 0.65 m 时进入 `stop`，线速度归零；
- `thermal_motion_controller/package.xml` 显式声明 `sensor_msgs`，守卫参数全部位于 `params.yaml`。

### 评测与恢复

- 新增 `scripts/phase1_gate.py`：自动判定配对完整性、候选 artifact、4/5 源 recall、precision、
  duplicate、4/5 源 `not_reached`、FUSE/clearance/residual planner p99 和 waypoint churn；
- gate 对旧矩阵按正确 cohort 得出 FAIL，并检测到最大 91 arrivals/min；
- matrix runner 新增原子 `run_result.json` checkpoint 与 `--resume`。中断后用相同 out-root
  重跑即可跳过已完成 run。
- resume checkpoint 绑定调用参数、world/scenario/occupancy 内容与 source/install runtime bundle 指纹；
  collector 非零、missing counts、JSON 不可解析、空 clearance 或计数不一致均不可复用；
- runner 启动前逐文件检查 source/install 同步，运行中再次哈希以禁止代码/配置变更；不同指纹
  checkpoint 直接拒绝，同指纹的残缺单局先保留到 `.incomplete/` 再重跑，避免旧 CSV 混入新归因；
- `/scan` 和 residual `/thermal/clearance` 已加入每局必需 topic counts。
- gate 的 pairing 现硬验证固定 24 case × 5 seed = 120 个唯一 run、固定 protocol 与无重复 key；
  两边相同的小子集也绝不可能 PASS；
- gate 逐局重算 fingerprint，并复用 runner 的严格 checkpoint 验证；metadata、run_result、8 类原始
  CSV、clearance、source summary、attribution、launch log 均须存在、可解析且彼此一致。
- baseline/candidate 每个 run 的 resolved world/scenario 必须与 `phase0_cases()` 规范映射一致，case 名
  不能掩盖自定义实验路径；
- gate 对 matrix summary 与四类 JSON 证据做 schema/type 校验，无效 run 在统计前隔离。截断矩阵、坏指标
  类型或空高源 cohort 都会生成完整 Markdown/JSON FAIL 报告并返回 2，不会异常退出或用默认值误算。

## 验证

纯测试：

```bash
python3 -m pytest src/thermal_robot/tests/ -q
```

结果：`136 passed`（最终新鲜运行见本页末尾“最终验证”）。

依赖：

```text
rosdep check: All system dependencies have been satisfied
```

构建：

```text
thermal_interfaces: 1 package finished
g1_description + 6 runtime/bringup packages: 7 packages finished
```

### open/static5 定向冒烟

最终接线的 seed102 短冒烟中：

- 空 `0x0` 和全未知 occupancy 被跳过；
- residual planner 实际生成 target；
- `min_d=3.00`，严格大于 2.5 m arrival radius；
- planner `eval_ms` 约 31–36 ms；
- 未出现 COARSE arrival churn 或 `status=6`；
- Nav2 goal 发送间隔不小于 3 秒。

短 run 的 recall 不作门槛判断。

### walls/static5 定向冒烟

30 秒采集、seed101：

- occupancy `233x136`、free=797、occupied=13，随后随 SLAM 扩展；
- `cells_blocked_only` 从 8 增长到 126，遮挡融合生效；
- residual planner 从初期无候选过渡到有效 target；
- planner `eval_ms` 为 5.5–7.3 ms；
- 路径长度 5.449 m；
- 仅 1 次 waypoint arrival，即 2 arrivals/min，低于 12/min 行为守卫；
- 未出现 `status=6`，无数据缺失。

短 run 的 recall=0 仅因采集窗口短，不作门槛判断。

### 最终 direct-guard + provenance + supercover 接线冒烟

独立审查修正并重建后，使用全新 `/tmp` 根目录运行：

- open/static5/seed102，warmup 36 s + 采集 15 s：collector return code 0、`missing_counts=[]`、
  trajectory 286 条、scan 29 条、clearance 2 条，路径 2.389 m；FUSE max 0.8 ms，residual planner
  25.6–25.9 ms；guard 含初始 `stop` 和 4 次 `scan_guarded`，0 arrival、0 `status=6`；
- walls/static5/seed103，warmup 36 s + 采集 30 s：collector return code 0、`missing_counts=[]`、
  trajectory 586 条、scan 58 条、clearance 3 条，路径 3.114 m；有效 occupancy 为 233×136、
  residual planner 从无候选过渡到 target，3 次评估为 0.6/6.1/5.8 ms；FUSE max 1.5 ms，
  guard 记录 3 次 `stop` 和 5 次 `scan_guarded`，0 arrival、0 `status=6`；
- 两局均生成严格 checkpoint 所需的全部原始 CSV 和 JSON，随后原样加
  `--resume` 均直接复用 checkpoint，未启动 Gazebo；
- 把 1-run open smoke 送入最终 gate：artifact 与 candidate health PASS，但 pairing 明确报告
  `expected_pairs=120`、缺 119 局并整体 FAIL；
- 停止后无遗留 gzserver/gzclient/ROS launch 进程。

这两个 15 s run 只证明当前接线、安全守卫、断点与产物闭环，不用于 recall 门槛判断。

## 全量矩阵运行手册（项目负责人执行）

### 1. 预检查

在 workspace 根目录执行：

```bash
source /opt/ros/humble/setup.bash
rosdep check --from-paths src/thermal_robot --ignore-src
colcon build --packages-select thermal_interfaces
source install/setup.bash
colcon build --packages-select \
  g1_description thermal_sensor_sim signal_preprocessor \
  thermal_field_reconstructor thermal_gradient_processor \
  thermal_motion_controller thermal_bringup
source install/setup.bash
python3 -m pytest src/thermal_robot/tests/ -q
pgrep -a gzserver || true
pgrep -a gzclient || true
df -h .
```

应看到 rosdep 无缺项、两次构建成功、136 tests 通过、没有遗留 Gazebo 进程，并有足够空间。
runner 还会自动拒绝 source 与 install 内容不一致的工作区，因此代码变化后不能跳过构建。

### 2. 启动全量候选矩阵

选择一个不会覆盖旧数据的新目录：

```bash
set -o pipefail
python3 -u src/thermal_robot/scripts/run_multiscenario_matrix.py \
  --preset phase0 \
  --seeds 101,102,103,104,105 \
  --strategy residual \
  --jitter 0 --min-recall 0 \
  --duration 120 --warmup 36 \
  --domain-start 71 \
  --resume \
  --out-root bags/matrix/phase1_residual_recovery_20260711 \
  2>&1 | tee bags/matrix/phase1_residual_recovery_20260711.log
```

runner 每完成一局写入该局的 `run_result.json`。若进程中断，确认没有遗留 Gazebo 后，原样重跑
同一命令；`--resume` 只跳过指纹完全匹配且产物可解析的 run。不要修改 strategy、seed、duration、
warmup、jitter、min-recall、world/scenario、代码、配置或 out-root。若确实修改了代码/配置，先构建并
使用新的 out-root；runner 会拒绝把不同实验混在原目录。

中断发生在 `run_result.json` 写入前时，该残缺单局会在重跑时原样移动到 out-root 下的
`.incomplete/`，再从空的 `seed` 目录重跑；不会删除旧数据，也不会让旧 CSV 参与新归因。

runner 可能因个别 duplicate 算法 FAIL 最终返回 1；这不等于矩阵损坏，先检查完整性。

### 3. 完整性检查

```bash
jq '{n_runs,n_passed,all_passed}' \
  bags/matrix/phase1_residual_recovery_20260711/matrix_summary.json

jq '[.runs[] | select((.missing_counts | length) > 0)] | length' \
  bags/matrix/phase1_residual_recovery_20260711/matrix_summary.json

find bags/matrix/phase1_residual_recovery_20260711 \
  -mindepth 3 -maxdepth 3 -name run_result.json | wc -l
```

期望 `n_runs=120`、missing-count runs 为 0、`run_result.json` 为 120。最终 gate 还会独立检查
120 个唯一 key、固定 protocol、每局原始 CSV 行数、checkpoint 与 runtime fingerprint；`n_passed` 不作为阶段门槛
的替代，因为 runner 仍保留逐局 duplicate=0 的旧 PASS 规则。

### 4. 自动门槛判定

```bash
python3 src/thermal_robot/scripts/phase1_gate.py \
  bags/matrix/phase0_full_20260611 \
  bags/matrix/phase1_residual_recovery_20260711 \
  --label-a v31 \
  --label-b residual_recovery \
  --out bags/matrix/phase1_residual_recovery_20260711/phase1_gate_report.md \
  --json-out bags/matrix/phase1_residual_recovery_20260711/phase1_gate_report.json
```

gate 返回 0 表示全部门槛 PASS；返回 2 表示至少一项 FAIL。两种情况都保留报告，不得手工改阈值。

### 5. 回传

运行完成后提供以下两个文件即可进入独立验收：

- `bags/matrix/phase1_residual_recovery_20260711/phase1_gate_report.md`
- `bags/matrix/phase1_residual_recovery_20260711/phase1_gate_report.json`

若完整性检查失败，同时提供 `matrix_summary.json`，先补跑缺失 run，不直接重跑全部。

## 当前结论

代码、测试、构建、open/walls 冒烟、自动 gate 和可恢复 runner 均已准备好。阶段 1 本身尚不能标记
通过；必须等待项目负责人运行新的 120 局全量候选矩阵，并由独立验收依据 gate report 判定。

## 最终验证

最终文档收口前重新执行：

```bash
git diff --check
python3 -m py_compile \
  src/thermal_robot/scripts/run_multiscenario_matrix.py \
  src/thermal_robot/scripts/phase1_gate.py \
  src/thermal_robot/thermal_motion_controller/thermal_motion_controller/navigation_policy.py \
  src/thermal_robot/thermal_motion_controller/thermal_motion_controller/controller_node.py
python3 -m pytest src/thermal_robot/tests/ -q
```

验收数字以最后一次命令的实际输出为准，不以前一次运行或短烟测外推。

最后一次新鲜纯测试结果：`136 passed`。最后三轮独立只读代码审查提出的 canonical case、JSON/schema
fail-closed、统计隔离与空 cohort CLI 落盘问题均已逐项修复并回归。
