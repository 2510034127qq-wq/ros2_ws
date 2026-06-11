# Phase 1 residual full matrix interrupted

## Background

- Goal: run the phase 1 `strategy=residual` full matrix against the existing v31 baseline at `bags/matrix/phase0_full_20260611/`.
- Code state checked before execution: `git log --oneline -1` reported `6122185 test: restore monte-carlo calibration, speed budget, npz consistency guards`.
- Test check before execution: `python3 -m pytest src/thermal_robot/tests/ -q` reported `86 passed in 0.47s`.
- Baseline presence check passed: `bags/matrix/phase0_full_20260611/matrix_summary.json` exists.
- Installed interface check passed: `ros2 interface show thermal_interfaces/msg/ThermalMap | tail -2` reported `view_state` and `view_sectors`.
- Gazebo pre-check found no `gzserver` or `gzclient`.
- Disk pre-check reported about 72G available on `/home/hanchen/ros2_ws`.

## Command

The matrix was started as a foreground long session to avoid the previous `nohup` parent-session issue, with the same matrix parameters:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 src/thermal_robot/scripts/run_multiscenario_matrix.py \
  --preset phase0 \
  --seeds 101,102,103,104,105 \
  --strategy residual \
  --duration 120 --warmup 36 \
  --domain-start 71 \
  --out-root bags/matrix/phase1_residual_20260612
```

- Out root: `bags/matrix/phase1_residual_20260612`
- Companion log: `bags/matrix/phase1_residual_20260612.log`
- Start: `2026-06-12T04:37:00+08:00`
- Stop: `2026-06-12T07:16:38+08:00`
- Elapsed wall time: about 2h 39m 38s
- Exit: `143` after manual termination.

## Interruption

The run was not completed. It was manually interrupted on user request.

- Completed runs with `metadata.json`: 59/120
- Created but incomplete run directory: `bags/matrix/phase1_residual_20260612/boxes__birthdeath/seed105`
- Completed through: `boxes__birthdeath/seed104`
- No official full-matrix `matrix_summary.json` or `matrix_report.md` was produced because the batch was stopped before the script's final aggregation step.
- `matrix_compare.py` was not run, so no official paired comparison report was produced.

Completed runs by case:

| case | completed seeds |
| --- | ---: |
| open__static2 | 5 |
| open__static3 | 5 |
| open__static5 | 5 |
| open__dyn4 | 5 |
| open__dyn5 | 5 |
| open__birthdeath | 5 |
| boxes__static2 | 5 |
| boxes__static3 | 5 |
| boxes__static5 | 5 |
| boxes__dyn4 | 5 |
| boxes__dyn5 | 5 |
| boxes__birthdeath | 4 |

Partial completed-run status using the matrix script's PASS criteria (`missing_counts` empty, non-empty source summary, recall >= 0.0, duplicate confirmations == 0):

- Partial completed runs: 59
- Partial passed: 53
- Partial failed: 6
- Suspected simulation-not-started runs: 0

Partial algorithmic FAIL runs among completed runs:

| run | recall | precision | duplicate_confirmations | missing_counts |
| --- | ---: | ---: | ---: | --- |
| open__dyn4/seed101 | 0.75 | 0.75 | 1 | [] |
| open__dyn4/seed102 | 0.75 | 0.75 | 1 | [] |
| boxes__static3/seed101 | 0.667 | 0.667 | 1 | [] |
| boxes__static3/seed103 | 0.667 | 0.667 | 1 | [] |
| boxes__dyn4/seed102 | 0.75 | 0.75 | 1 | [] |
| boxes__dyn4/seed103 | 0.75 | 0.75 | 1 | [] |

## Clearance Data

- Completed runs with `clearance_final` in `metadata.json`: 59/59 completed
- Runs with `clearance.csv`: 59/59 completed
- `clearance_final` min: 0.011994434520602226
- `clearance_final` median: 0.02085428684949875
- `clearance_final` max: 0.028995690867304802

## Runtime Evidence

Sample from `open__static2/seed101/launch.log`:

```text
[thermal_mapper_node-16] [INFO] [1781210359.204397046] [thermal_mapper_node]: [FUSE] n=1200 0.7ms occ_map=yes cells_clear=1719 cells_blocked_only=0
[controller_node-19] [INFO] [1781210360.938782444] [controller_node]: [CLEARANCE] p_no_undetected=0.0203 eps=0.05 eval_ms=0.9
[thermal_mapper_node-16] [INFO] [1781210369.204067176] [thermal_mapper_node]: [FUSE] n=1300 0.7ms occ_map=yes cells_clear=1813 cells_blocked_only=0
[controller_node-19] [INFO] [1781210370.940020839] [controller_node]: [CLEARANCE] p_no_undetected=0.0203 eps=0.05 eval_ms=1.4
```

Sample from `boxes__static2/seed101/launch.log`:

```text
[thermal_mapper_node-16] [INFO] [1781215167.918998939] [thermal_mapper_node]: [FUSE] n=1200 0.9ms occ_map=yes cells_clear=1763 cells_blocked_only=689
[controller_node-19] [INFO] [1781215169.750092479] [controller_node]: [CLEARANCE] p_no_undetected=0.0234 eps=0.05 eval_ms=0.9
[thermal_mapper_node-16] [INFO] [1781215177.918931609] [thermal_mapper_node]: [FUSE] n=1300 0.8ms occ_map=yes cells_clear=1796 cells_blocked_only=690
[controller_node-19] [INFO] [1781215179.849848946] [controller_node]: [CLEARANCE] p_no_undetected=0.0234 eps=0.05 eval_ms=0.9
```

## Cleanup And Baseline Check

- Cleanup command run: `bash src/thermal_robot/kill_gz.sh`
- Cleanup result: no `gzserver` or `gzclient` found after cleanup.
- Partial directory was retained: `bags/matrix/phase1_residual_20260612`
- Baseline directory was not intentionally written or deleted. Post-stop listing began with:

```text
总计 304
drwxrwxr-x 26 hanchen hanchen   4096 Jun 12 01:40 .
drwxrwxr-x  4 hanchen hanchen   4096 Jun 12 04:37 ..
```

## Conclusion

This matrix run is incomplete. It stopped at 59/120 completed runs because of manual interruption. No official paired comparison was generated.
