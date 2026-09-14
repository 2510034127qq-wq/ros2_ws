# CLAUDE.md

## Repository scope

ROS 2 Humble / Gazebo Classic workspace. Active code is under `src/thermal_robot/`; follow [AGENTS.md](AGENTS.md). Current scope is stationary, continuously emitting thermal-source inspection. Moving-source tracking, emission schedules, predictive revisits and automatic clearance decisions are removed.

Use [README](src/thermal_robot/README.md) for commands and [handover](docs/handover/00-总览与导读.md) for implementation details. Historical logs and design/Plan files describe their own versions, not the current runtime.

## Build and test

```bash
cd /home/hanchen/ros2_ws
source /opt/ros/humble/setup.bash
rosdep check --from-paths src/thermal_robot --ignore-src
colcon build --packages-select thermal_interfaces
source install/setup.bash
colcon build --packages-select g1_description thermal_sensor_sim signal_preprocessor thermal_field_reconstructor thermal_gradient_processor thermal_motion_controller thermal_bringup
source install/setup.bash
python3 -m pytest src/thermal_robot/tests/ -q
```

Rebuild interfaces before consumers after message changes. Removed Python modules/configuration may survive incremental installs; check source/install consistency. Keep generated build/install/log/bag/cache data out of Git.

## Launch and inspect

```bash
ros2 launch thermal_bringup sim_nav_slam_launch.py use_rviz:=false use_gzclient:=false
ros2 launch thermal_bringup sim_nav_slam_launch.py --show-args
```

Run one simulation at a time. Defaults: A-level 64×48 at 10 Hz, `dual`, `online`, `use_sim_time:=true`. B-level (`sensor_model:=b`) is 160×120 at 8.6 Hz with depth; B preprocessing defaults to passthrough. Config-B is a three-source layout, distinct from the B observation model.

After launch, inspect `/sim/thermal_raw`, `/thermal/filtered`, `/thermal/field`, `/thermal/gradient`, `/thermal/map`, `/thermal/sources`, `/thermal/belief`, `/odom`, `/scan` and `/cmd_vel`. Rates depend on observation mode and controller/Nav2 activity; not every topic runs at 10 Hz.

```bash
python3 src/thermal_robot/tests/test_thermal_system.py --ros
python3 src/thermal_robot/scripts/nav2_health_check.py --timeout 20
ros2 service call /thermal/get_field_info thermal_interfaces/srv/GetFieldInfo "{include_full_data: true}"
```

## Architecture and constraints

The thermal image feeds both field/gradient reconstruction and the world thermal mapper. The tracker consumes the thermal map and owns source IDs; the optional slow posterior consumes those registered sources. `dual` admits only healthy, fresh online posterior feedback. `fast` and `gp_ucb` reject slow feedback.

B mode and A with `fast/dual/gp_ucb` use world-coordinate approach/exploration in `_surface_timer`; only A with legacy strategies uses the detailed gradient FSM. Direct Twist commands and Nav2 action execution are alternative command paths. Nav2 completion is not proof of physical source approach.

Simulation odom is absolute Gazebo world pose. Main launch sets mapper/controller spawn offsets to zero; do not add the spawn translation twice. ROS simulation time is enabled, but some strategy durations use monotonic wall time.

Parameters are layered: `params.yaml`, `multisource.yaml` or `software_params`, then launch overrides. UGV uses an additional `ugv_thermal.yaml` or `hardware_params` overlay. The hardware launch defaults to `enable_motion:=false`, `use_sim_time:=false` and needs external odom/scan/map/TF/Nav2, radiometric images, calibrated CameraInfo and registered depth. Hardware and Pi 5 performance remain unverified.

Use [evaluation instructions](docs/handover/04-评测体系与实验.md) for run evidence. Source-count messages, passing probes and unit tests do not establish complete physical-source detection or real-device readiness. Do not feed simulation source truth into controller, mapper, tracker or belief inference.
