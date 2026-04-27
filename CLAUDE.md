# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

**Interfaces must be built first** (other packages depend on them):
```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select thermal_interfaces
source install/setup.bash
colcon build --packages-select \
    g1_description thermal_sensor_sim signal_preprocessor \
    thermal_field_reconstructor thermal_gradient_processor \
    thermal_motion_controller thermal_bringup
source install/setup.bash
```

Build a single package:
```bash
colcon build --packages-select <package_name>
source install/setup.bash
```

## Testing

Run unit tests (pure Python, no ROS required):
```bash
python3 -m pytest src/thermal_robot/tests/test_thermal_system.py -v
```

Run algorithm benchmarks:
```bash
python3 src/thermal_robot/tests/test_thermal_system.py --bench
```

Verify topic rates after launching:
```bash
ros2 topic hz /sim/thermal_raw      # ~10 Hz
ros2 topic hz /thermal/filtered     # ~10 Hz
ros2 topic hz /thermal/field        # ~10 Hz
ros2 topic hz /thermal/gradient     # ~10 Hz
ros2 topic hz /cmd_vel              # ~10 Hz
```

## Launch

Full simulation pipeline:
```bash
ros2 launch thermal_bringup sim_nav_slam_launch.py
```

Single node test:
```bash
ros2 run thermal_sensor_sim sensor_node
```

Query thermal field service:
```bash
ros2 service call /thermal/get_field_info \
  thermal_interfaces/srv/GetFieldInfo "{include_full_data: true}"
```

Analyze collected simulation data:
```bash
python3 src/thermal_robot/scripts/plot_all_figures.py bags/collected/<timestamp>
python3 src/thermal_robot/scripts/plot_slam_nav2.py bags/collected/<timestamp>
```

Kill leftover Gazebo processes:
```bash
bash src/thermal_robot/kill_gz.sh
```

## Architecture

This is a ROS 2 Humble workspace implementing autonomous thermal source-seeking navigation for a Unitree G1 robot.

### Processing Pipeline

```
sensor_node  →  /sim/thermal_raw  (Image 32FC1 64×48 @ 10Hz)
                        ↓
preprocessor_node  →  /thermal/filtered  (Image 32FC1 @ 10Hz)
                        ↓
reconstructor_node  →  /thermal/field  (ThermalField @ 10Hz)
                        ↓
gradient_node  →  /thermal/gradient  (GradientArray @ 10Hz)
                        ↓
controller_node  →  /cmd_vel  (Twist @ 10Hz)  →  [Nav2 SLAM]
```

All packages live under `src/thermal_robot/`.

### Package Roles

- **thermal_interfaces** — Custom msg/srv definitions (`ThermalPoint`, `ThermalField`, `Gradient`, `GradientArray`, `GetFieldInfo`). Must be built before all other packages.
- **g1_description** — URDF and meshes for Unitree G1 29-DOF + Inspire hands. Contains `g1_nav.urdf` (simplified diff-drive) and `g1_thermal.urdf.xacro` (parametric with sensor).
- **thermal_sensor_sim** — Simulates a 64×48 thermal camera with 3 configurable heat sources publishing `sensor_msgs/Image` (32FC1 encoding).
- **signal_preprocessor** — Applies temporal Kalman/MA/EMA filtering and spatial Gaussian blur to raw thermal images.
- **thermal_field_reconstructor** — Converts filtered images to `ThermalField` messages via linear interpolation; detects hotspots using NMS at `mean + 4°C`.
- **thermal_gradient_processor** — Computes Sobel or Central Difference gradients at stride-4 subsampling; outputs `GradientArray` with peak detection.
- **thermal_motion_controller** — Multi-state FSM (ASCENT → CONVERGE → SAMPLE → AT_PEAK → RELOCATE → ESCAPE → FRONTIER_NAV → COARSE_SURVEY → DEPARTURE) implementing gradient ascent + Lévy flight exploration with Nav2 integration.
- **thermal_bringup** — Main SLAM/Nav2 launch file, RViz config, and central `params.yaml` with tunable parameters for the full pipeline.

### Key Configuration

All node parameters are in `src/thermal_robot/thermal_bringup/config/params.yaml`. Changes here affect behavior of all pipeline nodes simultaneously without code changes.

Nav2 and SLAM Toolbox are configured in `nav2_params.yaml` and `slam_params.yaml` respectively. The controller has a direct `/cmd_vel` fallback when Nav2 is unavailable.

### Custom Interfaces

When adding or modifying `.msg`/`.srv` files in `thermal_interfaces`, rebuild that package first before dependent packages.
