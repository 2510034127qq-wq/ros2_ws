# 2026-05-02 01:50 CST Dynamic Thermal Navigation Runtime Devlog

## Scope

- Added world-frame thermal map interfaces and source estimate interfaces.
- Added scenario-file driven thermal sources with Config-B YAML fallback and a truth topic for collection/benchmark use.
- Added `thermal_mapper_node` to fuse camera-frame thermal observations into a world grid.
- Added `source_tracker_node` for candidate, confirmed, stale, and suppressed source estimates.
- Rewired the motion controller to use `/thermal/map` and `/thermal/sources` for information-gain exploration while keeping the direct `/cmd_vel` fallback.
- Extended collection, benchmark, documentation, launch, config, and pure tests around source-level discovery.

## Verification

Commands run from `/home/hanchen/ros2_ws`:

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash && colcon build --packages-select thermal_interfaces thermal_sensor_sim thermal_field_reconstructor thermal_motion_controller thermal_bringup
source /opt/ros/humble/setup.bash && source install/setup.bash && python3 -m pytest src/thermal_robot/tests/test_thermal_system.py -q
git diff --check
```

Results:

- Build passed for 5 selected packages.
- Pytest passed: `13 passed in 0.06s`.
- `git diff --check` passed.

## Runtime Test

Headless launch and collector were run in the same shell with `ROS_LOG_DIR=/tmp/ros2_ws_ros_logs`:

```bash
ros2 launch thermal_bringup sim_nav_slam_launch.py use_rviz:=false use_gzclient:=false
python3 src/thermal_robot/scripts/collect_sim_data.py --duration 60 --out-dir /tmp/thermal_runtime_20260502_0144
```

Observed output:

- `/thermal/map`: 3.89 Hz, 233 collected map rows.
- `/thermal/sources`: 3.89 Hz, 710 collected source estimate rows.
- `/sim/thermal_sources_truth`: 10.0 Hz, 1800 collected truth rows.
- `/cmd_vel`: 10.0 Hz, 600 collected command rows.
- SLAM was available after 1.93 s with 582 collected SLAM poses.
- Collector wrote `metadata.json`, `source_summary.json`, CSV data, and 11 snapshot pairs.

Source-level summary:

- `source_recall`: 0.333.
- `source_precision`: 1.0.
- `time_to_first_source`: 1.694 s.
- Confirmed/matched source: `SA_left` via `src_1`, localization error 0.876 m.
- `duplicate_confirmations`: 0.
- `path_length_m`: 1.918.

## Residual Issues

- The 60 s runtime did not reach the target `confirmed >= 2/3`; it confirmed 1 of 3 Config-B sources.
- Nav2 still produced no `/plan` samples in this run and showed stale-transform failures under the current default `use_sim_time=false`; the controller's direct velocity fallback kept `/cmd_vel` active.
- The next algorithm pass should focus on faster source verification/exploration after the first confirmed source, not on launch plumbing.
