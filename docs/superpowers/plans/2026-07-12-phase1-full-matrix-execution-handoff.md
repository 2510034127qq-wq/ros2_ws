# Phase 1 全量矩阵执行交接计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to execute this file task-by-task. Do not redesign the experiment, edit runtime code, or dispatch implementation agents.

**Goal:** 在固定代码、固定参数和固定输出目录下完成 Phase 1 residual 候选的 120 局矩阵；即使 Codex 会话或单局运行中断，也只重跑未完成/不完整的局，最终生成可交回主会话验收的 gate 报告。

**Architecture:** runner 为每个完整 run 原子写入 `run_result.json`，`--resume` 只复用指纹一致且原始证据完整的 checkpoint。中断局不会被拼接续写，而是在恢复时完整移入 `.incomplete/` 后单局重跑；最终 `matrix_summary.json` 只在 120 局全部遍历完后生成。Phase 1 是否通过只由 `phase1_gate.py` 判定，不由 runner 的逐局 `PASS/FAIL` 或人为判断代替。

**Tech Stack:** ROS 2 Humble、Gazebo 11、Python 3、pytest、colcon、`run_multiscenario_matrix.py`、`phase1_gate.py`。

---

## 执行状态清单

- [ ] 每次进入会话均完成第 3、4 节重入检查
- [ ] 首次运行前验证 source/install、测试、进程和磁盘
- [ ] 用第 6 节唯一命令首次运行或恢复到 120 个 checkpoint
- [ ] 第 8 节完整性检查全部满足
- [ ] 第 9 节 gate 生成 Markdown 和 JSON
- [ ] 第 11 节 `FULL_MATRIX_RETURN.md` 和结果路径已回传

## 1. 代理职责与严格边界

你是“全量矩阵运行代理”，不是实现代理。你的工作只有：

1. 检查当前是否已有仍在运行或已部分完成的同一矩阵；
2. 首次运行或用完全相同的命令恢复运行；
3. 监控并保留全部证据；
4. 完成后执行完整性检查和 Phase 1 gate；
5. 把结果文件路径和客观判定返回给用户。

不得做以下事情：

- 不修改 `src/thermal_robot/` 下任何源码、配置、world、scenario、URDF 或脚本；
- 不修改统计阈值、seed、strategy、duration、warmup、jitter、min-recall、domain-start 或 case 集；
- 不删除候选目录、`.incomplete/`、checkpoint、旧矩阵或日志；
- 不使用 `git reset`、`git checkout --`、`git clean`、`rm -rf`；
- 不清理用户已有 dirty worktree，也不把无关文件加入提交；
- 不用小矩阵、短冒烟或部分结果代替 120 局；
- 不因为 runner 返回 1 就立即重跑全部；先按本文件检查完整性；
- 不更新总体 spec 的阶段状态，不启动 Phase 2；这些属于主会话的独立验收工作。

当前冻结实现的代码基点为：

```text
603f3b9e9f3d8ee047aa2f46a27d11f8a7e414be
fix: recover phase1 residual coverage
```

允许 HEAD 比该提交多出纯文档提交，但 `src/thermal_robot/` 相对该提交必须没有变化。用户工作树中已有无关修改/未跟踪文件是预期状态，必须保留。

## 2. 本次实验不可改变的身份

| 项目 | 固定值 |
|---|---|
| workspace | `/home/hanchen/ros2_ws` |
| baseline | `bags/matrix/phase0_full_20260611` |
| candidate | `bags/matrix/phase1_residual_recovery_20260711` |
| console log | `bags/matrix/phase1_residual_recovery_20260711.console.log` |
| preset | `phase0` |
| worlds/cases | canonical phase0：4 worlds × 6 scenarios |
| seeds | `101,102,103,104,105` |
| strategy | `residual` |
| duration | `120 s` |
| warmup | `36 s` |
| jitter | `0` |
| min recall（runner 旧逐局规则） | `0` |
| domain start | `71` |
| expected runs | `120` |

无论首次运行还是第几次恢复，都必须使用同一个 candidate 路径和第 6 节的原样命令。

## 3. 每次进入/恢复会话时先执行的决策树

每次收到“开始”“继续”“恢复”或会话压缩后，都从本节开始，不要凭聊天记忆决定。

### 3.1 进入 workspace 并重新读取本文件

```bash
cd /home/hanchen/ros2_ws
sed -n '1,460p' docs/superpowers/plans/2026-07-12-phase1-full-matrix-execution-handoff.md
```

### 3.2 检查是否已有 matrix runner 在运行

```bash
pgrep -af '[r]un_multiscenario_matrix.py' || true
pgrep -af '[r]os2 launch thermal_bringup sim_nav_slam_launch.py' || true
pgrep -a -x gzserver || true
```

判定：

- **runner 仍在运行**：绝对不要再启动第二个 runner。用第 7 节的命令监控进度和 console log；等待现有进程结束。
- **runner 不在，但 launch/gzserver 仍在**：先确认它们确实属于本矩阵的遗留进程。向对应 launch 进程组发送 `SIGINT`，等待最多 10 秒后复查；不要对不明进程执行广泛 `pkill`。遗留进程未退出前不要恢复矩阵。
- **都不在运行**：继续统计 checkpoint。

### 3.3 以顶层有效 checkpoint 数判断进度

```bash
find bags/matrix/phase1_residual_recovery_20260711 \
  -mindepth 3 -maxdepth 3 -name run_result.json \
  -not -path '*/.incomplete/*' 2>/dev/null | wc -l
```

解释：

- 输出 `0` 且 candidate 不存在：走第 4、5、6 节首次运行；
- 输出 `1`–`119`：跳过重建和重新设计，完成第 4 节只读不变量检查后，直接用第 6 节原命令 `--resume`；
- 输出 `120` 但没有 `matrix_summary.json`：上次可能在最后一局后、汇总前中断，仍用第 6 节原命令；120 个 checkpoint 会快速复用，然后生成汇总；
- 输出 `120` 且已有 `matrix_summary.json`：不要再跑仿真，直接进入第 8、9 节；
- checkpoint 数大于 120：停止并报告，不要删除任何证据。

`.incomplete/` 下的旧尝试不计入完成数，也绝不能手工搬回正式 run 目录。

## 4. 每次运行前的只读不变量检查

### 4.1 验证代码基点和 runtime 源码未改变

```bash
git merge-base --is-ancestor 603f3b9e9f3d8ee047aa2f46a27d11f8a7e414be HEAD
git diff --quiet 603f3b9e9f3d8ee047aa2f46a27d11f8a7e414be -- src/thermal_robot
git status --short
```

预期：前两条命令返回 0。`git status --short` 可以显示用户原有的无关 dirty 文件；不要清理。

若 `src/thermal_robot` 已变化：停止，不要恢复或新跑矩阵，把差异路径返回给用户。旧 checkpoint 与新 runtime 不能混用。

### 4.2 验证 baseline 是固定 120 局

```bash
jq '{n_runs,config}' bags/matrix/phase0_full_20260611/matrix_summary.json
test "$(jq -r '.n_runs' bags/matrix/phase0_full_20260611/matrix_summary.json)" = "120"
```

预期：`n_runs=120`，strategy 为 `full`，duration 为 `120.0`，warmup 为 `36.0`，seeds 为 101–105。

### 4.3 验证 source 与 ROS install 内容同步

该检查只读，不会改环境：

```bash
python3 - <<'PY'
import importlib.util
import sys
from pathlib import Path

path = Path('src/thermal_robot/scripts/run_multiscenario_matrix.py').resolve()
spec = importlib.util.spec_from_file_location('phase1_matrix_preflight', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
mismatches = module.runtime_sync_mismatches(module.runtime_sync_pairs())
print(f'source_install_mismatches={len(mismatches)}')
for source, installed in mismatches[:20]:
    print(f'{source} != {installed}')
raise SystemExit(1 if mismatches else 0)
PY
```

预期：`source_install_mismatches=0`。

- candidate 已有任何 checkpoint 时，若此项非 0，立即停止并返回差异；不要构建后继续混跑。
- candidate 尚不存在且此项非 0，才执行第 5 节构建，然后重新运行本检查。

## 5. 仅首次运行前执行的验证

如果 candidate 已有任何正式 `run_result.json`，不要重复本节构建；直接使用第 4 节的同步检查。

### 5.1 依赖与构建（仅在 source/install 不同步时需要构建）

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
```

预期：rosdep 报 `All system dependencies have been satisfied`，接口包和 7 个 runtime/bringup 包构建成功。构建后必须重新执行第 4.3 节，确认 mismatch 为 0。

### 5.2 新鲜纯测试

```bash
python3 -m pytest src/thermal_robot/tests -q
```

当前冻结代码预期：`136 passed`，退出码 0。若失败，停止并返回完整失败输出，不要带病运行 120 局。

### 5.3 进程与磁盘

```bash
pgrep -af '[r]un_multiscenario_matrix.py' || true
pgrep -af '[r]os2 launch thermal_bringup sim_nav_slam_launch.py' || true
pgrep -a -x gzserver || true
df -h .
test "$(df -Pk . | awk 'NR==2 {print $4}')" -ge 5242880
```

预期：没有同类 runner/launch/Gazebo 进程，workspace 可用空间至少 5 GiB。

## 6. 唯一允许的首次/恢复命令

不要后台化，不要使用 `timeout`，不要拆成多个并行矩阵。使用带 PTY 的长运行工具启动，并保存该工具返回的 session id 以便轮询。

先为追加日志写入本次启动时间：

```bash
date --iso-8601=seconds | tee -a bags/matrix/phase1_residual_recovery_20260711.console.log
```

然后原样运行：

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
  2>&1 | tee -a bags/matrix/phase1_residual_recovery_20260711.console.log
```

正常总耗时约 5–6 小时。每局约包含 36 秒 warmup、120 秒采集和清理时间。不要因一两分钟没有新 console 行就判断卡死。

runner 最终退出码解释：

- `0`：120 局遍历结束，且 runner 的旧逐局规则全部 PASS；
- `1`：120 局可能已经完整，但至少一局不满足旧逐局规则（常见于 duplicate）；这不等于矩阵损坏，继续第 8 节；
- Python traceback、指纹冲突、source/install mismatch：属于执行失败，按第 10 节停止条件处理。

## 7. 运行中监控与用户中断处理

### 7.1 正常监控

长工具调用每 30–60 秒轮询一次，不要调用超过 60 秒的阻塞等待。每隔约 10 局向用户报告一次：已完成 checkpoint 数、当前 case/seed、是否出现执行错误。

可在不干扰 runner 的情况下执行：

```bash
find bags/matrix/phase1_residual_recovery_20260711 \
  -mindepth 3 -maxdepth 3 -name run_result.json \
  -not -path '*/.incomplete/*' 2>/dev/null | wc -l
tail -n 40 bags/matrix/phase1_residual_recovery_20260711.console.log
```

`matrix_summary.json` 只在所有 case/seed 遍历结束后生成；中途不存在不是错误。

### 7.2 用户中断 Codex 会话时

不要承诺“后台一定继续”。下次同一个会话恢复后，必须重新从第 3 节判断真实进程状态：

- runner 还活着：只监控，不启动第二个；
- runner 已退出：确认无遗留 launch/Gazebo，然后用第 6 节原命令；
- 当前局没有完整 checkpoint：runner 会把该局目录移到
  `bags/matrix/phase1_residual_recovery_20260711/.incomplete/<case>/`，再从空目录重跑；
- 已有完整 checkpoint：runner 会逐局校验 JSON、原始 CSV 行数、collector 状态、参数与 runtime 指纹，合格才显示 `[matrix] resume ...`。

恢复时不得手工编辑 `run_result.json`、`metadata.json`、CSV 或 `matrix_summary.json`。

### 7.3 指纹冲突时

若出现：

```text
incompatible checkpoint ... use a new --out-root
```

立即停止。不要删除旧 checkpoint，也不要自行改用新 out-root，因为那意味着代码、配置或调用身份已经变化，需要主会话决定是否废弃已完成局。

## 8. runner 结束后的完整性检查与必要补跑

### 8.1 检查正式 checkpoint 和汇总

```bash
find bags/matrix/phase1_residual_recovery_20260711 \
  -mindepth 3 -maxdepth 3 -name run_result.json \
  -not -path '*/.incomplete/*' | wc -l

jq '{n_runs,n_passed,all_passed,config}' \
  bags/matrix/phase1_residual_recovery_20260711/matrix_summary.json

jq '[.runs[] | select((.missing_counts | length) > 0)] | length' \
  bags/matrix/phase1_residual_recovery_20260711/matrix_summary.json

jq '[.runs[] | [.name,.seed]] | unique | length' \
  bags/matrix/phase1_residual_recovery_20260711/matrix_summary.json
```

完整矩阵要求：

- 正式 `run_result.json` 数量为 120；
- `matrix_summary.n_runs` 为 120；
- 唯一 `(name, seed)` 数为 120；
- `missing_counts` 非空的 run 数为 0；
- config 精确对应第 2 节固定身份；
- `n_passed`/`all_passed` 不作为 Phase 1 最终门槛替代物。

### 8.2 完整性不足时只恢复缺失/损坏局

若 checkpoint 少于 120、汇总不存在或 missing-count runs 非 0：

1. 不删除任何目录；
2. 确认没有 runner/launch/Gazebo 遗留进程；
3. 重新执行第 4 节不变量检查；
4. 原样执行第 6 节；
5. runner 会复用合格局，只归档并重跑不完整局；
6. 再执行本节检查。

若同一 run 连续 3 次仍产生 collector 非零、缺 topic 或损坏 artifact，停止并返回该 case/seed 的
`launch.log`、`collector.log`、`metadata.json` 和最近的 traceback，不要无限重试。

## 9. Phase 1 自动 gate

仅在第 8 节完整性满足后运行：

```bash
python3 src/thermal_robot/scripts/phase1_gate.py \
  bags/matrix/phase0_full_20260611 \
  bags/matrix/phase1_residual_recovery_20260711 \
  --label-a v31 \
  --label-b residual_recovery \
  --out bags/matrix/phase1_residual_recovery_20260711/phase1_gate_report.md \
  --json-out bags/matrix/phase1_residual_recovery_20260711/phase1_gate_report.json
```

gate 返回值：

- `0`：所有 Phase 1 门槛 PASS；
- `2`：至少一项门槛 FAIL，但 Markdown/JSON 报告仍是有效结果，必须保留并回传；
- traceback/未生成报告：工具执行错误，回传错误与 `matrix_summary.json`，不得把它写成算法 FAIL。

禁止手工修改 gate JSON、Markdown、阈值或输入矩阵来改变判定。

## 10. 必须停止并回报、不能自行绕过的情况

遇到以下任一情况停止：

1. `src/thermal_robot` 相对 `603f3b9` 有变化；
2. 已有 checkpoint 后 source/install mismatch 非 0；
3. runtime bundle 或 run fingerprint 冲突；
4. baseline 不存在或不是固定 120 局；
5. 可用磁盘低于 5 GiB；
6. checkpoint 数大于 120 或 case/seed 身份重复；
7. 同一 run 连续 3 次采集失败；
8. 无法确认遗留 ROS/Gazebo 进程是否属于本矩阵；
9. gate 自身 traceback，无法写出两份报告。

回报必须包含：执行到哪一步、完整命令、退出码、相关 case/seed、错误原文、checkpoint 数和保留的证据路径。不要提出降低矩阵规模或阈值作为解决办法。

## 11. 完成后生成并返回的结果

### 11.1 必须存在的机器生成文件

```text
bags/matrix/phase1_residual_recovery_20260711/matrix_summary.json
bags/matrix/phase1_residual_recovery_20260711/matrix_report.md
bags/matrix/phase1_residual_recovery_20260711/phase1_gate_report.md
bags/matrix/phase1_residual_recovery_20260711/phase1_gate_report.json
bags/matrix/phase1_residual_recovery_20260711.console.log
```

另外保留 120 个正式 run 目录及所有 `.incomplete/` 历史；不要压缩或复制数百 MB 原始数据，主会话与本会话共享 workspace，可直接读取。

### 11.2 创建人工交接摘要

用 `apply_patch` 创建：

```text
bags/matrix/phase1_residual_recovery_20260711/FULL_MATRIX_RETURN.md
```

只能填写实际读取到的值，至少包含：

- 最终 `git rev-parse HEAD` 和冻结实现基点；
- candidate/baseline 路径；
- 是否发生过中断、恢复次数、`.incomplete/` 尝试数；
- runner 最终退出码；
- `n_runs`、`n_passed`、`all_passed`、missing-count run 数；
- gate 退出码和总 PASS/FAIL；
- gate 每个 check 的 PASS/FAIL；
- 任何异常 case/seed 及证据路径；
- 返回给主会话的五个文件路径。

不要在摘要里自行宣布总体项目完成、修改总体 spec 或建议进入 Phase 2；只陈述 Phase 1 gate 的机器判定。

### 11.3 最终回复用户

回复中给出：

1. 矩阵是否完整（120/120、missing=0）；
2. runner 与 gate 的退出码；
3. gate 总 PASS/FAIL；
4. `FULL_MATRIX_RETURN.md`、`phase1_gate_report.md`、`phase1_gate_report.json` 和
   `matrix_summary.json` 的绝对可点击路径；
5. 若 FAIL，列出失败 check 名称，但不在该会话擅自修代码。

## 12. 完成定义

运行代理的任务只有在以下条件同时满足时完成：

- 120 个 canonical `(case, seed)` 均有正式、可复用 checkpoint；
- `matrix_summary.json` 的 120 局身份和固定 config 完整；
- missing-count runs 为 0；
- gate 成功生成 Markdown 和 JSON（gate 结果允许 PASS 或 FAIL）；
- `FULL_MATRIX_RETURN.md` 已按实际数据创建；
- 所有原始证据和中断历史均被保留；
- 用户获得四个核心结果文件的绝对路径。

算法是否通过 Phase 1 是 gate/主会话的结论；运行代理不得把“执行完整”与“指标通过”混为一谈。
