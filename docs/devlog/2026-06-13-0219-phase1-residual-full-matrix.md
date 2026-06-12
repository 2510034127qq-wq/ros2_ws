# Phase 1 residual full matrix

## Background

- Purpose: run the phase 1 `strategy=residual` full matrix and compare it with the existing v31 baseline.
- Baseline: `bags/matrix/phase0_full_20260611/`
- Residual output root: `bags/matrix/phase1_residual_20260612_restart1/`
- The earlier interrupted partial directory `bags/matrix/phase1_residual_20260612/` was retained and not reused.

## Execution

Pre-run checks:

- `git log --oneline -2` showed the active HEAD `70c2c29 docs: log interrupted phase1 residual matrix`, with parent `6122185 test: restore monte-carlo calibration, speed budget, npz consistency guards`.
- `python3 -m pytest src/thermal_robot/tests/ -q`: `86 passed in 0.51s`.
- `bags/matrix/phase0_full_20260611/matrix_summary.json` existed.
- `ros2 interface show thermal_interfaces/msg/ThermalMap | tail -2` showed `view_state` and `view_sectors`.
- No `gzserver` or `gzclient` was present before launch.
- Disk space before launch was about 71G available.

Command:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 -u src/thermal_robot/scripts/run_multiscenario_matrix.py \
  --preset phase0 \
  --seeds 101,102,103,104,105 \
  --strategy residual \
  --duration 120 --warmup 36 \
  --domain-start 71 \
  --out-root bags/matrix/phase1_residual_20260612_restart1
```

- Start: `2026-06-12T20:57:46+08:00`
- End: `2026-06-13T02:18:13+08:00`
- Total elapsed wall time: about 5h 20m 27s
- Matrix script exit: `1`, because `all_passed` was false after algorithmic duplicate FAILs.
- Final matrix artifacts:
  - `bags/matrix/phase1_residual_20260612_restart1/matrix_summary.json`
  - `bags/matrix/phase1_residual_20260612_restart1/matrix_report.md`
  - `bags/matrix/phase1_residual_20260612_restart1/compare_report.md`

## Validation

- `n_runs`: 120/120
- `n_passed`: 114/120
- Suspected simulation-not-started runs: 0
- Runs with `missing_counts`: 0
- Runs with `clearance.csv`: 120/120

Algorithmic FAIL runs:

| run | recall | precision | duplicate_confirmations | missing_counts |
|---|---:|---:|---:|---|
| open__dyn4/seed102 | 0.75 | 0.75 | 1 | [] |
| open__dyn4/seed103 | 0.75 | 0.75 | 1 | [] |
| boxes__dyn4/seed103 | 0.75 | 0.75 | 1 | [] |
| walls__dyn4/seed105 | 0.75 | 0.75 | 1 | [] |
| mixed__dyn4/seed101 | 0.75 | 0.75 | 1 | [] |
| mixed__dyn4/seed103 | 0.75 | 0.75 | 1 | [] |

## Compare Report

Source: `bags/matrix/phase1_residual_20260612_restart1/compare_report.md`

### Pooled paired tests (sign-flip permutation)

| metric | n | mean A | mean B | p |
|---|---|---|---|---|
| source_recall | 120 | 0.625 | 0.646 | 0.11989 |
| time_to_first_source | 119 | 6.328 | 9.364 | 0.09720 |
| path_length_m | 120 | 21.726 | 19.337 | 0.00005 |

### Event-pooled precision / duplicate rate (run-level cluster bootstrap 95% CI)

| side | precision [lo, hi] | duplicate rate [lo, hi] |
|---|---|---|
| v31 | 0.969 [0.949, 0.988] | 0.018 [0.007, 0.031] |
| residual | 0.978 [0.960, 0.993] | 0.014 [0.004, 0.025] |

### Per-case recall (paired permutation)

| case | n | recall A | recall B | p |
|---|---|---|---|---|
| boxes__birthdeath | 5 | 0.667 | 0.667 | 1.00000 |
| boxes__dyn4 | 5 | 0.700 | 0.750 | 1.00000 |
| boxes__dyn5 | 5 | 0.400 | 0.480 | 0.50000 |
| boxes__static2 | 5 | 1.000 | 1.000 | 1.00000 |
| boxes__static3 | 5 | 0.600 | 0.600 | 1.00000 |
| boxes__static5 | 5 | 0.400 | 0.480 | 0.50000 |
| mixed__birthdeath | 5 | 0.667 | 0.667 | 1.00000 |
| mixed__dyn4 | 5 | 0.750 | 0.750 | 1.00000 |
| mixed__dyn5 | 5 | 0.480 | 0.520 | 1.00000 |
| mixed__static2 | 5 | 1.000 | 1.000 | 1.00000 |
| mixed__static3 | 5 | 0.400 | 0.600 | 0.37500 |
| mixed__static5 | 5 | 0.440 | 0.360 | 0.62500 |
| open__birthdeath | 5 | 0.667 | 0.667 | 1.00000 |
| open__dyn4 | 5 | 0.600 | 0.650 | 1.00000 |
| open__dyn5 | 5 | 0.440 | 0.480 | 1.00000 |
| open__static2 | 5 | 1.000 | 1.000 | 1.00000 |
| open__static3 | 5 | 0.533 | 0.533 | 1.00000 |
| open__static5 | 5 | 0.480 | 0.400 | 0.62500 |
| walls__birthdeath | 5 | 0.667 | 0.667 | 1.00000 |
| walls__dyn4 | 5 | 0.700 | 0.700 | 1.00000 |
| walls__dyn5 | 5 | 0.440 | 0.600 | 0.12500 |
| walls__static2 | 5 | 1.000 | 1.000 | 1.00000 |
| walls__static3 | 5 | 0.533 | 0.600 | 1.00000 |
| walls__static5 | 5 | 0.440 | 0.320 | 0.50000 |

## Failure Attribution

Source: `bags/matrix/phase1_residual_20260612_restart1/matrix_report.md`

### Residual

| case | not_reached | occluded | timing_missed | not_confirmed |
|---|---|---|---|---|
| boxes__birthdeath | 5 | 0 | 0 | 0 |
| boxes__dyn4 | 5 | 0 | 0 | 0 |
| boxes__dyn5 | 13 | 0 | 0 | 0 |
| boxes__static2 | 0 | 0 | 0 | 0 |
| boxes__static3 | 6 | 0 | 0 | 0 |
| boxes__static5 | 13 | 0 | 0 | 0 |
| mixed__birthdeath | 5 | 0 | 0 | 0 |
| mixed__dyn4 | 5 | 0 | 0 | 0 |
| mixed__dyn5 | 12 | 0 | 0 | 0 |
| mixed__static2 | 0 | 0 | 0 | 0 |
| mixed__static3 | 6 | 0 | 0 | 0 |
| mixed__static5 | 16 | 0 | 0 | 0 |
| open__birthdeath | 5 | 0 | 0 | 0 |
| open__dyn4 | 7 | 0 | 0 | 0 |
| open__dyn5 | 13 | 0 | 0 | 0 |
| open__static2 | 0 | 0 | 0 | 0 |
| open__static3 | 7 | 0 | 0 | 0 |
| open__static5 | 15 | 0 | 0 | 0 |
| walls__birthdeath | 5 | 0 | 0 | 0 |
| walls__dyn4 | 6 | 0 | 0 | 0 |
| walls__dyn5 | 10 | 0 | 0 | 0 |
| walls__static2 | 0 | 0 | 0 | 0 |
| walls__static3 | 6 | 0 | 0 | 0 |
| walls__static5 | 17 | 0 | 0 | 0 |

### Baseline v31

Source: `bags/matrix/phase0_full_20260611/matrix_report.md`

| case | not_reached | occluded | timing_missed | not_confirmed |
|---|---|---|---|---|
| boxes__birthdeath | 5 | 0 | 0 | 0 |
| boxes__dyn4 | 6 | 0 | 0 | 0 |
| boxes__dyn5 | 15 | 0 | 0 | 0 |
| boxes__static2 | 0 | 0 | 0 | 0 |
| boxes__static3 | 6 | 0 | 0 | 0 |
| boxes__static5 | 15 | 0 | 0 | 0 |
| mixed__birthdeath | 5 | 0 | 0 | 0 |
| mixed__dyn4 | 5 | 0 | 0 | 0 |
| mixed__dyn5 | 13 | 0 | 0 | 0 |
| mixed__static2 | 0 | 0 | 0 | 0 |
| mixed__static3 | 9 | 0 | 0 | 0 |
| mixed__static5 | 14 | 0 | 0 | 0 |
| open__birthdeath | 5 | 0 | 0 | 0 |
| open__dyn4 | 8 | 0 | 0 | 0 |
| open__dyn5 | 14 | 0 | 0 | 0 |
| open__static2 | 0 | 0 | 0 | 0 |
| open__static3 | 7 | 0 | 0 | 0 |
| open__static5 | 13 | 0 | 0 | 0 |
| walls__birthdeath | 5 | 0 | 0 | 0 |
| walls__dyn4 | 6 | 0 | 0 | 0 |
| walls__dyn5 | 14 | 0 | 0 | 0 |
| walls__static2 | 0 | 0 | 0 | 0 |
| walls__static3 | 7 | 0 | 0 | 0 |
| walls__static5 | 14 | 0 | 0 | 0 |

World-level `not_reached` totals changed from baseline to residual as follows: open 47 -> 47, boxes 47 -> 42, walls 46 -> 44, mixed 46 -> 44; total 186 -> 177.

## Clearance Data

- `metadata.json` entries with `clearance_final`: 120/120
- `clearance.csv` timelines: 120/120
- `clearance_final` min: 0.011994434520602226
- `clearance_final` median: 0.021056320518255234
- `clearance_final` max: 0.029829727485775948

## Runtime Evidence

Sample from `open__static2/seed101/launch.log`:

```text
[thermal_mapper_node-16] [INFO] [1781269205.551462204] [thermal_mapper_node]: [FUSE] n=1200 0.6ms occ_map=yes cells_clear=1716 cells_blocked_only=0
[controller_node-19] [INFO] [1781269207.681035402] [controller_node]: [CLEARANCE] p_no_undetected=0.0204 eps=0.05 eval_ms=0.9
[thermal_mapper_node-16] [INFO] [1781269215.552043180] [thermal_mapper_node]: [FUSE] n=1300 0.9ms occ_map=yes cells_clear=1744 cells_blocked_only=0
[controller_node-19] [INFO] [1781269217.683195250] [controller_node]: [CLEARANCE] p_no_undetected=0.0204 eps=0.05 eval_ms=1.0
```

Sample from `boxes__static2/seed101/launch.log`:

```text
[thermal_mapper_node-16] [INFO] [1781274026.688985745] [thermal_mapper_node]: [FUSE] n=1200 0.8ms occ_map=yes cells_clear=1674 cells_blocked_only=549
[controller_node-19] [INFO] [1781274029.711688982] [controller_node]: [CLEARANCE] p_no_undetected=0.0205 eps=0.05 eval_ms=0.7
[thermal_mapper_node-16] [INFO] [1781274036.688956935] [thermal_mapper_node]: [FUSE] n=1300 0.8ms occ_map=yes cells_clear=1759 cells_blocked_only=549
[controller_node-19] [INFO] [1781274039.721278080] [controller_node]: [CLEARANCE] p_no_undetected=0.0205 eps=0.05 eval_ms=1.0
```

Sample from `walls__static2/seed101/launch.log`:

```text
[thermal_mapper_node-16] [INFO] [1781278828.053374813] [thermal_mapper_node]: [FUSE] n=1200 0.9ms occ_map=yes cells_clear=1616 cells_blocked_only=476
[controller_node-19] [INFO] [1781278830.486919434] [controller_node]: [CLEARANCE] p_no_undetected=0.0219 eps=0.05 eval_ms=0.9
[thermal_mapper_node-16] [INFO] [1781278838.052945251] [thermal_mapper_node]: [FUSE] n=1300 0.6ms occ_map=yes cells_clear=1682 cells_blocked_only=497
[controller_node-19] [INFO] [1781278840.583594238] [controller_node]: [CLEARANCE] p_no_undetected=0.0219 eps=0.05 eval_ms=1.0
```

## Cleanup

- Cleanup command: `bash src/thermal_robot/kill_gz.sh`
- Cleanup result: no `gzserver` or `gzclient` found.
- Baseline directory post-run listing began with:

```text
总计 304
drwxrwxr-x 26 hanchen hanchen   4096 Jun 12 01:40 .
drwxrwxr-x  4 hanchen hanchen   4096 Jun 12 04:37 ..
```

## Conclusion

The restarted residual full matrix completed all 120 runs. The run produced `matrix_summary.json`, `matrix_report.md`, and `compare_report.md` under `bags/matrix/phase1_residual_20260612_restart1/`.
