# Phase 0 Full Matrix Gate-Review Data

## Execution Context

- Workspace: `/home/hanchen/ros2_ws`
- HEAD: `a936256 fix: single-source failure classes and scenario-driven attribution FOV`
- Precheck tests: `python3 -m pytest src/thermal_robot/tests/ -q` -> `68 passed in 0.24s`
- `install/setup.bash`: present
- Initial Gazebo process check: no `gzserver` / `gzclient` output
- `/tmp` free space before run: 72G available

## Frontier Smoke

Command:

```bash
python3 src/thermal_robot/scripts/run_multiscenario_matrix.py \
  --preset phase0 --cases open__static2 --seeds 101 \
  --strategy frontier --duration 45 --warmup 36 --domain-start 198 \
  --out-root /tmp/phase0_smoke_frontier
```

Result:

- Exit code: 0
- Run result: `PASS open__static2 seed=101: recall=1.0 precision=1.0 dup=0`
- `grep "strategy=frontier" /tmp/phase0_smoke_frontier/open__static2/seed101/launch.log`: matched `[controller_node]: strategy=frontier random_seed=101`
- `/tmp/phase0_smoke_frontier/open__static2/seed101/attribution.json`: present
- Frontier smoke summary: `n_runs=1`, `n_passed=1`, `missing_counts=[]`

## Full Matrix Command

Requested background command shape:

```bash
mkdir -p bags/matrix
nohup python3 src/thermal_robot/scripts/run_multiscenario_matrix.py \
  --preset phase0 \
  --seeds 101,102,103,104,105 \
  --strategy full \
  --duration 120 --warmup 36 \
  --domain-start 71 \
  --out-root bags/matrix/phase0_full_20260611 \
  > bags/matrix/phase0_full_20260611.log 2>&1 &
echo $! > /tmp/phase0_matrix.pid
```

Execution note:

- The Codex sandbox `nohup` attempt produced an empty log and no run directory because the sandbox process tree was killed with the parent.
- The actual data run used the same matrix arguments and the same out-root/log path in a foreground long session:

```bash
python3 src/thermal_robot/scripts/run_multiscenario_matrix.py \
  --preset phase0 \
  --seeds 101,102,103,104,105 \
  --strategy full \
  --duration 120 --warmup 36 \
  --domain-start 71 \
  --out-root bags/matrix/phase0_full_20260611 \
  > bags/matrix/phase0_full_20260611.log 2>&1
```

Artifacts:

- Out-root: `bags/matrix/phase0_full_20260611`
- Log: `bags/matrix/phase0_full_20260611.log`
- Summary: `bags/matrix/phase0_full_20260611/matrix_summary.json`
- Report: `bags/matrix/phase0_full_20260611/matrix_report.md`

## Completion Summary

- Matrix process exit code: 1
- `n_runs`: 120
- `n_passed`: 112 / 120
- `all_passed`: false
- Suspected simulation-not-started runs (`missing_counts` non-empty): 0, `[]`
- Algorithm/metric FAIL runs: 8
  - `open__dyn4/seed102`
  - `boxes__dyn4/seed102`
  - `boxes__dyn4/seed103`
  - `boxes__dyn4/seed104`
  - `boxes__dyn4/seed105`
  - `walls__dyn4/seed102`
  - `walls__dyn4/seed103`
  - `mixed__dyn4/seed103`
- Sum of per-run `elapsed_wall_s`: 19245.704s = 5.346h
- Estimated command wall elapsed from first run timing to summary mtime: 19274.101s = 5.354h
- First estimated start: 2026-06-11 20:19:01 +0800
- Summary mtime: 2026-06-12 01:40:15 +0800
- Post-run cleanup: `bash src/thermal_robot/kill_gz.sh` killed one leftover `gzserver`; follow-up checks reported `gazebo clean` and `gzclient clean`.

## Per-Case Multi-Seed Statistics

| case | runs | passed | recall | precision | duplicates | t_first(s) |
|---|---|---|---|---|---|---|
| boxes__birthdeath | 5 | 5 | 0.667+/-0.000 (n=5) | 1.000+/-0.000 (n=5) | 0.000+/-0.000 (n=5) | 0.732+/-0.458 (n=5) |
| boxes__dyn4 | 5 | 1 | 0.700+/-0.112 (n=5) | 0.800+/-0.112 (n=5) | 0.800+/-0.447 (n=5) | 8.534+/-7.319 (n=5) |
| boxes__dyn5 | 5 | 5 | 0.400+/-0.000 (n=5) | 1.000+/-0.000 (n=5) | 0.000+/-0.000 (n=5) | 0.432+/-0.379 (n=5) |
| boxes__static2 | 5 | 5 | 1.000+/-0.000 (n=5) | 1.000+/-0.000 (n=5) | 0.000+/-0.000 (n=5) | 0.374+/-0.415 (n=5) |
| boxes__static3 | 5 | 5 | 0.600+/-0.149 (n=5) | 1.000+/-0.000 (n=5) | 0.000+/-0.000 (n=5) | 4.067+/-5.865 (n=5) |
| boxes__static5 | 5 | 5 | 0.400+/-0.000 (n=5) | 1.000+/-0.000 (n=5) | 0.000+/-0.000 (n=5) | 19.271+/-19.162 (n=5) |
| mixed__birthdeath | 5 | 5 | 0.667+/-0.000 (n=5) | 1.000+/-0.000 (n=5) | 0.000+/-0.000 (n=5) | 0.382+/-0.416 (n=5) |
| mixed__dyn4 | 5 | 4 | 0.750+/-0.000 (n=5) | 0.950+/-0.112 (n=5) | 0.200+/-0.447 (n=5) | 3.239+/-5.913 (n=5) |
| mixed__dyn5 | 5 | 5 | 0.480+/-0.179 (n=5) | 1.000+/-0.000 (n=5) | 0.000+/-0.000 (n=5) | 0.290+/-0.260 (n=5) |
| mixed__static2 | 5 | 5 | 1.000+/-0.000 (n=5) | 1.000+/-0.000 (n=5) | 0.000+/-0.000 (n=5) | 0.289+/-0.261 (n=5) |
| mixed__static3 | 5 | 5 | 0.400+/-0.149 (n=5) | 1.000+/-0.000 (n=5) | 0.000+/-0.000 (n=5) | 16.992+/-21.973 (n=5) |
| mixed__static5 | 5 | 5 | 0.440+/-0.167 (n=5) | 1.000+/-0.000 (n=5) | 0.000+/-0.000 (n=5) | 12.157+/-7.938 (n=5) |
| open__birthdeath | 5 | 5 | 0.667+/-0.000 (n=5) | 1.000+/-0.000 (n=5) | 0.000+/-0.000 (n=5) | 0.565+/-0.491 (n=5) |
| open__dyn4 | 5 | 4 | 0.600+/-0.224 (n=5) | 0.950+/-0.112 (n=5) | 0.200+/-0.447 (n=5) | 3.252+/-5.659 (n=5) |
| open__dyn5 | 5 | 5 | 0.440+/-0.089 (n=5) | 1.000+/-0.000 (n=5) | 0.000+/-0.000 (n=5) | 0.504+/-0.363 (n=5) |
| open__static2 | 5 | 5 | 1.000+/-0.000 (n=5) | 1.000+/-0.000 (n=5) | 0.000+/-0.000 (n=5) | 0.341+/-0.257 (n=5) |
| open__static3 | 5 | 5 | 0.533+/-0.380 (n=5) | 0.800+/-0.447 (n=5) | 0.000+/-0.000 (n=5) | 17.101+/-22.202 (n=4) |
| open__static5 | 5 | 5 | 0.480+/-0.179 (n=5) | 1.000+/-0.000 (n=5) | 0.000+/-0.000 (n=5) | 19.883+/-17.001 (n=5) |
| walls__birthdeath | 5 | 5 | 0.667+/-0.000 (n=5) | 1.000+/-0.000 (n=5) | 0.000+/-0.000 (n=5) | 0.517+/-0.262 (n=5) |
| walls__dyn4 | 5 | 3 | 0.700+/-0.112 (n=5) | 0.900+/-0.137 (n=5) | 0.400+/-0.548 (n=5) | 5.826+/-7.577 (n=5) |
| walls__dyn5 | 5 | 5 | 0.440+/-0.089 (n=5) | 1.000+/-0.000 (n=5) | 0.000+/-0.000 (n=5) | 0.792+/-0.304 (n=5) |
| walls__static2 | 5 | 5 | 1.000+/-0.000 (n=5) | 1.000+/-0.000 (n=5) | 0.000+/-0.000 (n=5) | 0.519+/-0.493 (n=5) |
| walls__static3 | 5 | 5 | 0.533+/-0.183 (n=5) | 1.000+/-0.000 (n=5) | 0.000+/-0.000 (n=5) | 14.967+/-23.841 (n=5) |
| walls__static5 | 5 | 5 | 0.440+/-0.167 (n=5) | 1.000+/-0.000 (n=5) | 0.000+/-0.000 (n=5) | 22.990+/-20.511 (n=5) |

## Missed-Source Failure Attribution

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

## World-Class Observations

World-class aggregate recall and pass counts:

| world | runs | passed | mean recall | not_reached | occluded | timing_missed | not_confirmed |
|---|---:|---:|---:|---:|---:|---:|---:|
| open | 30 | 29 | 0.620 | 47 | 0 | 0 | 0 |
| boxes | 30 | 26 | 0.628 | 47 | 0 | 0 | 0 |
| walls | 30 | 28 | 0.630 | 46 | 0 | 0 | 0 |
| mixed | 30 | 29 | 0.623 | 46 | 0 | 0 | 0 |

Observed patterns:

- Open vs obstacle worlds: aggregate recall is close across all four world classes (`open=0.620`, `boxes=0.628`, `walls=0.630`, `mixed=0.623`). In this run, obstacle worlds did not show a large aggregate recall drop relative to open.
- Pass count differs by world: `open=29/30`, `boxes=26/30`, `walls=28/30`, `mixed=29/30`. The additional FAILs are concentrated in dyn4 cases with duplicate confirmations or precision below the pass threshold.
- Case/source-count pattern is more visible than world class: static2 cases are `1.000` recall in all world classes; static5 and dyn5 cases are mostly around `0.400-0.480` recall; birthdeath cases are `0.667` recall in all world classes.
- Attribution four-class pattern: all missed-source attribution counts are `not_reached`; `occluded`, `timing_missed`, and `not_confirmed` are all zero across the matrix.
- Total missed-source attribution counts by world are nearly equal: `open=47`, `boxes=47`, `walls=46`, `mixed=46`.
