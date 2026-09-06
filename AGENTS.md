# Repository Guidelines

## Project Structure & Module Organization

This is a ROS 2 Humble workspace. Active code lives under `src/thermal_robot/`.
The main packages are:

- `thermal_bringup`: active launch, Nav2/SLAM configs, RViz config, and Gazebo world.
- `thermal_sensor_sim`, `signal_preprocessor`, `thermal_field_reconstructor`, `thermal_gradient_processor`, `thermal_motion_controller`: runtime thermal pipeline and controller nodes.
- `thermal_interfaces`: custom messages and services.
- `g1_description`: active `g1_nav.urdf` plus retained G1 mesh/URDF assets.

Tests are in `src/thermal_robot/tests/`. Data tools are in `src/thermal_robot/scripts/`. The current main launch is `sim_nav_slam_launch.py`; do not restore removed legacy launch or plotting files.

## Build, Test, and Development Commands

From the workspace root:

```bash
cd /home/hanchen/ros2_ws
source /opt/ros/humble/setup.bash
rosdep check --from-paths src/thermal_robot --ignore-src
colcon build --packages-select thermal_interfaces
source install/setup.bash
colcon build --packages-select g1_description thermal_sensor_sim signal_preprocessor thermal_field_reconstructor thermal_gradient_processor thermal_motion_controller thermal_bringup
source install/setup.bash
```

Run the main simulation:

```bash
ros2 launch thermal_bringup sim_nav_slam_launch.py use_rviz:=true use_gzclient:=true
```

Headless mode:

```bash
ros2 launch thermal_bringup sim_nav_slam_launch.py use_rviz:=false use_gzclient:=false
```

Collect and plot data with `scripts/collect_sim_data.py`, `scripts/plot_all_figures.py`, and `scripts/plot_slam_nav2.py`.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation and `snake_case` names. ROS node files use the `*_node.py` pattern. Keep package metadata in each `package.xml` synchronized with actual imports and launch-time dependencies. Put tunable runtime values in YAML files under `thermal_bringup/config/`, not hardcoded in nodes unless there is a clear reason.

## Testing Guidelines

Use `pytest` for pure algorithm tests:

```bash
python3 -m pytest src/thermal_robot/tests/test_thermal_system.py -q
```

ROS integration checks require the main launch to be running first:

```bash
python3 src/thermal_robot/tests/test_thermal_system.py --ros
```

After launch, verify `/sim/thermal_raw`, `/thermal/filtered`, `/thermal/field`, `/thermal/gradient`, `/odom`, `/scan`, and `/cmd_vel` with `ros2 topic hz`.

## Commit & Pull Request Guidelines

The current history uses concise Conventional Commit style, for example `chore: organize thermal robot workspace`. Prefer `type: imperative summary` such as `fix: update Nav2 parameters` or `docs: expand simulation guide`.

Pull requests should include the reason for the change, affected packages, commands run, and any launch/runtime impact. Include screenshots or generated plots when RViz, Gazebo, or analysis figures change.

## Generated Files & Local State

Do not commit `build/`, `install/`, `log/`, `bags/`, Python caches, or local tool directories. These are ignored by `.gitignore`.
