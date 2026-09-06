# thermal_robot

ROS 2 Humble workspace for stationary, continuously emitting thermal-source inspection. The robot detects hotspots, locates them on a map, merges repeated observations and explores around obstacles. Moving-source prediction, emission schedules and automatic all-clear decisions have been removed from the active code.

Current scope: [static inspection simplification](../../docs/superpowers/specs/2026-09-06-static-thermal-inspection-design.md). The pre-simplification implementation and research tools remain available in Git at `581c802`; historical measurements do not validate the current version.

## Runtime pipeline

- `thermal_sensor_sim`: A-level ideal field or B-level perspective surface/depth observations in Gazebo.
- `signal_preprocessor`: temporal/spatial thermal filtering.
- `thermal_field_reconstructor`: field reconstruction, calibrated perspective projection, depth registration, radiometric input and world thermal map.
- `thermal_gradient_processor`: thermal gradients.
- `thermal_motion_controller`: static source association, position filtering, optional slow posterior and navigation.
- `thermal_interfaces`: thermal maps, source estimates and belief messages.
- `thermal_bringup`: active launch files, SLAM/Nav2 configuration, scenarios and worlds.
- `g1_description`: differential-drive simulation model and retained G1 assets.

SLAM/Nav2 provide geometry, localization and route execution. Thermal coverage is tracked separately: a geometrically mapped region can still contain unseen thermal surfaces. Truth topics are used only by evaluation tools.

## Build and test

```bash
cd /home/hanchen/ros2_ws
source /opt/ros/humble/setup.bash
rosdep check --from-paths src/thermal_robot --ignore-src
colcon build --packages-select thermal_interfaces
source install/setup.bash
colcon build --packages-select g1_description thermal_sensor_sim signal_preprocessor thermal_field_reconstructor thermal_gradient_processor thermal_motion_controller thermal_bringup
source install/setup.bash
python3 -m pytest src/thermal_robot/tests -q
```

After upgrading from the older implementation, use a clean build/install for the affected packages. Incremental Python/CMake installs can retain removed modules or scenario files. SourceEstimate no longer contains velocity or reacquisition fields; rebuild dependent packages and update external consumers. Prefer replaying original sensor messages over older derived-message bags.

## Simulation

```bash
ros2 launch thermal_bringup sim_nav_slam_launch.py use_rviz:=true use_gzclient:=true

# Headless B-level surface camera, static sources, slow posterior enabled
ros2 launch thermal_bringup sim_nav_slam_launch.py \
  use_rviz:=false use_gzclient:=false sensor_model:=b strategy:=dual belief_mode:=online
```

Defaults: `sensor_model:=a`, `strategy:=dual`, `belief_mode:=online`, simulation time enabled. `fast` and `gp_ucb` remain comparison strategies; `full/frontier/levy/residual` retain the older static policies. `belief_mode:=shadow` evaluates the slow worker without feedback; `off` disables estimation.

Runtime parameters live in `thermal_bringup/config/params.yaml` and `multisource.yaml`. There is no extra task overlay. See [runtime and UGV instructions](../../docs/software/multisource_runtime.md) for parameter details.

Default Config-B contains three fixed sources. `scenario_file` selects a different static layout; `world_file` independently selects geometry. Active scenarios include `static_two_sources.yaml`, `static_five_sources.yaml`, `static_offset_sources.yaml` and `software_contract_multisource.yaml`. Dynamic schedules or trajectories in YAML raise an error.

```bash
ros2 launch thermal_bringup sim_nav_slam_launch.py \
  use_rviz:=false use_gzclient:=false sensor_model:=b \
  world_file:=/home/hanchen/ros2_ws/src/thermal_robot/thermal_bringup/worlds/thermal_scene_obstacle_field.world \
  scenario_file:=/home/hanchen/ros2_ws/src/thermal_robot/thermal_bringup/config/scenarios/static_five_sources.yaml
```

Worlds include open, obstacle-field, corridor-room, mixed-room, zigzag and sparse-island layouts. Inspection runs until stopped by the operator or an external budget. No source-count or no-new-source timeout claims the search is complete.

## Validation and data

```bash
python3 src/thermal_robot/scripts/run_software_validation.py \
  --out bags/software_validation/my_static_b --sensor-model b --strategy dual \
  --scenario static_two_sources.yaml --duration 90 --domain 151

python3 src/thermal_robot/scripts/evaluate_detection_tracks.py bags/software_validation/my_static_b
python3 src/thermal_robot/scripts/plot_software_validation.py \
  bags/software_validation/my_static_b --out /tmp/static_b.png
```

Use a fresh output directory each run. The runner checks source/install consistency and serializes Gazebo sessions. It records runtime health separately from physical-source matching, localization and duplicate labels. A passing runtime probe alone does not establish complete detection or real-device performance.

`run_multiscenario_matrix.py` keeps static 2/3/5-source presets, six world layouts, source matching and resumable evidence checks. Its historical `phase0` preset name now selects a 4-world × 3-static-layout grid. The original dynamic grid and `phase1_gate.py` research acceptance contract have been retired. Use `--help` for current runner options.

For an already running main launch:

```bash
python3 src/thermal_robot/tests/test_thermal_system.py --ros
python3 src/thermal_robot/scripts/collect_sim_data.py --help
python3 src/thermal_robot/scripts/plot_all_figures.py --help
python3 src/thermal_robot/scripts/plot_slam_nav2.py --help
```

Check `/sim/thermal_raw`, `/thermal/filtered`, `/thermal/field`, `/thermal/gradient`, `/odom`, `/scan` and `/cmd_vel` with `ros2 topic hz`. `/thermal/clearance` is no longer published or required.

## UGV and radiometric camera

```bash
ros2 launch thermal_bringup ugv_thermal_launch.py
python3 src/thermal_robot/scripts/validate_hardware_contract.py \
  --out bags/software_validation/my_hardware_contract
```

The UGV overlay defaults to `enable_motion:=false`. The existing robot stack owns odometry, LaserScan, map/TF and the Nav2 action. Thermal localization requires real radiometric input, CameraInfo, calibrated optical TF and registered depth. Lepton/PureThermal does not itself supply depth. Pseudocolor RGB is not a temperature image.

The contract check uses synthetic ROS inputs and verifies temperature units, depth registration, mapping, source confirmation and disabled motion. Physical camera calibration, Pi 5 load and real vehicle behavior remain unverified.

## Development

Tests are under `tests/`; evaluation scripts are under `scripts/`. Keep tunable values in YAML and package dependencies synchronized with imports. The active simulation entry remains `sim_nav_slam_launch.py`; do not restore legacy launch files. Generated build/install/log/bag/cache outputs are excluded from Git.
