# Phase 1 覆盖墙诊断实验：duration 扫描

## 背景

阶段 1 门槛评审显示 recall 瓶颈来自覆盖，而非观测质量。本实验固定 `strategy=full`，只在源全程活跃且 recall 受限的 static5 场景上改变 `duration`，记录 120s、240s、360s 下的 recall 与漏源距离趋势。

120s 数据复用只读基线目录 `bags/matrix/phase0_full_20260611/`；本次仅新增：

- `bags/matrix/phase1_diag_dur240/`
- `bags/matrix/phase1_diag_dur240.log`
- `bags/matrix/phase1_diag_dur360/`
- `bags/matrix/phase1_diag_dur360.log`

## 改动

未修改任何源码、参数或测试。本次只执行矩阵扫描并新增本 devlog。

执行口径：

```bash
python3 -u src/thermal_robot/scripts/run_multiscenario_matrix.py \
  --preset phase0 \
  --cases open__static5,boxes__static5,walls__static5,mixed__static5 \
  --seeds 101,102,103 \
  --strategy full \
  --duration 240 --warmup 36 \
  --domain-start 150 \
  --out-root bags/matrix/phase1_diag_dur240 \
  2>&1 | tee bags/matrix/phase1_diag_dur240.log
```

```bash
python3 -u src/thermal_robot/scripts/run_multiscenario_matrix.py \
  --preset phase0 \
  --cases open__static5,boxes__static5,walls__static5,mixed__static5 \
  --seeds 101,102,103 \
  --strategy full \
  --duration 360 --warmup 36 \
  --domain-start 150 \
  --out-root bags/matrix/phase1_diag_dur360 \
  2>&1 | tee bags/matrix/phase1_diag_dur360.log
```

总耗时：约 2h16m26s。

- 240s：2026-06-13 12:04:41 +0800 至 13:00:52 +0800，约 56m10s。
- 360s：2026-06-13 13:00:59 +0800 至 14:21:07 +0800，约 1h20m08s。

## 验证

执行前检查：

- `git log --oneline -1`：`4985bea docs: log phase1 residual full matrix and paired comparison`
- `python3 -m pytest src/thermal_robot/tests/ -q`：`86 passed in 0.42s`
- `ros2 interface show thermal_interfaces/msg/ThermalMap | tail -2`：

```text
uint8[] view_state
uint8[] view_sectors
```

- `pgrep -a gzserver || echo clean`：`clean`
- `df -h . | tail -1`：可用空间 70G

执行后清理：

- `bash src/thermal_robot/kill_gz.sh`：未发现 `gzserver` / `gzclient` 残留。
- `pgrep -a gzserver || echo "gazebo clean"`：`gazebo clean`
- `ls -la bags/matrix/phase0_full_20260611 | head -3` 显示基线目录自身 mtime 仍为 `Jun 12 01:40`。

完整性：

- `bags/matrix/phase1_diag_dur240/matrix_summary.json` 存在。
- `bags/matrix/phase1_diag_dur240/matrix_report.md` 存在。
- `bags/matrix/phase1_diag_dur360/matrix_summary.json` 存在。
- `bags/matrix/phase1_diag_dur360/matrix_report.md` 存在。

n_passed：

- 240s：`n_runs=12`，`n_passed=10`，`all_passed=false`
- 360s：`n_runs=12`，`n_passed=12`，`all_passed=true`

中途失败：

- 240s `boxes__static5/seed101`：算法性 FAIL，`source_recall=0.4`，`source_precision=0.667`，`duplicate_confirmations=1`，`missing_counts=[]`
- 240s `mixed__static5/seed101`：算法性 FAIL，`source_recall=0.4`，`source_precision=0.667`，`duplicate_confirmations=1`，`missing_counts=[]`
- 360s：无 FAIL
- 两个 duration 均无 `missing_counts` 仿真启动类故障。

判读表：

```text
=== recall vs duration (mean over seeds 101-103) ===
case                  120s    240s    360s
open__static5        0.400   0.467   0.400
boxes__static5       0.400   0.467   0.333
walls__static5       0.400   0.400   0.467
mixed__static5       0.400   0.467   0.400
```

```text
=== 漏源 min_center_dist 中位数 vs duration ===
  120s: n_missed=36 median_min_dist=5.70m
  240s: n_missed=33 median_min_dist=6.05m
  360s: n_missed=36 median_min_dist=6.18m
```

## 结论

24 个新增诊断 run 已完成，两个 out-root 与判读表已生成。未对 H1/H2 做结论性判定，数据留给门槛评审使用。
