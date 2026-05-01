# thermal_robot

ROS 2 Humble workspace package set for simulated multi-source thermal navigation. The current implementation uses a simplified differential-drive Unitree G1 navigation model, a synthetic thermal camera pipeline, SLAM Toolbox, Nav2, and a thermal-driven controller.

## Current Architecture

```text
Gazebo g1_nav model
  -> /odom, /scan
  -> slam_toolbox
  -> /map + map->odom TF
  -> Nav2 NavigateToPose

/sim/thermal_raw
  -> /thermal/filtered
  -> /thermal/field + /thermal/get_field_info
  -> /thermal/map
  -> /thermal/sources
  -> /thermal/gradient
  -> controller_node
       FINE states: direct /cmd_vel
       COARSE states: Nav2 preferred, direct /cmd_vel fallback
```

The main launch file is:

```bash
ros2 launch thermal_bringup sim_nav_slam_launch.py
```

## Packages

| Package | Type | Role |
|---|---|---|
| `thermal_interfaces` | CMake | Custom messages and service: `ThermalField`, `ThermalMap`, `SourceEstimateArray`, `GradientArray`, `GetFieldInfo` |
| `g1_description` | CMake | `g1_nav.urdf` simplified diff-drive model, full G1 assets, meshes |
| `thermal_sensor_sim` | Python | Synthetic 64x48 thermal camera and RGB colorizer |
| `signal_preprocessor` | Python | Temporal Kalman/MA/EMA filtering plus spatial smoothing |
| `thermal_field_reconstructor` | Python | Converts filtered images to `ThermalField`; fuses `/thermal/map`; provides `/thermal/get_field_info` |
| `thermal_gradient_processor` | Python | Sobel or central-difference gradient extraction |
| `thermal_motion_controller` | Python | Source tracker plus v31 multi-state thermal navigation controller |
| `thermal_bringup` | CMake | Launch files, parameters, Nav2/SLAM config, RViz config, Gazebo world |

## Main Topics and Services

| Name | Type | Producer | Typical Rate |
|---|---|---|---|
| `/sim/thermal_raw` | `sensor_msgs/msg/Image` `32FC1` | `sensor_node` | 10 Hz |
| `/sim/thermal_colorized` | `sensor_msgs/msg/Image` `rgb8` | `colorizer_node` | 10 Hz |
| `/thermal/filtered` | `sensor_msgs/msg/Image` `32FC1` | `preprocessor_node` | 10 Hz |
| `/thermal/field` | `thermal_interfaces/msg/ThermalField` | `reconstructor_node` | 10 Hz |
| `/thermal/map` | `thermal_interfaces/msg/ThermalMap` | `thermal_mapper_node` | 5 Hz |
| `/thermal/sources` | `thermal_interfaces/msg/SourceEstimateArray` | `source_tracker_node` | 5 Hz |
| `/sim/thermal_sources_truth` | `thermal_interfaces/msg/SourceEstimateArray` | `sensor_node` | 10 Hz |
| `/thermal/gradient` | `thermal_interfaces/msg/GradientArray` | `gradient_node` | 10 Hz |
| `/thermal/get_field_info` | `thermal_interfaces/srv/GetFieldInfo` | `reconstructor_node` | service |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | `controller_node` or Nav2 | about 10 Hz |
| `/odom` | `nav_msgs/msg/Odometry` | Gazebo diff-drive plugin | about 20 Hz |
| `/scan` | `sensor_msgs/msg/LaserScan` | Gazebo ray sensor | 10 Hz |
| `/map` | `nav_msgs/msg/OccupancyGrid` | `slam_toolbox` | configured |
| `/plan` | `nav_msgs/msg/Path` | Nav2 planner | on goal/planning events |

## Simulation Scenario

Current Config-B thermal source layout is available both as the empty-`scenario_file`
fallback in `sensor_node.py` and as `thermal_bringup/config/config_b_sources.yaml`:

| Source | World Position | Peak Approx. | Purpose |
|---|---:|---:|---|
| `SA_left` | `(-1.0, 3.5)` | 57 C | Strong source, convergence check |
| `SB_far` | `(6.0, -3.0)` | 44 C | Far source, exploration check |
| `SC_weak` | `(-5.0, -5.5)` | 38 C | Weak source, sensitivity check |

Robot spawn is `(-6.0, 0.0)`. The thermal camera simulation uses a 4.0 m by 3.0 m field of view and publishes 64x48 float images.

Custom dynamic scenarios can be passed at launch:

```bash
ros2 launch thermal_bringup sim_nav_slam_launch.py \
  scenario_file:=/home/hanchen/ros2_ws/src/thermal_robot/thermal_bringup/config/config_b_sources.yaml
```

The scenario schema supports static, linear, circular, waypoint-loop,
appear/disappear, and random-walk source motion plus optional strength drift.
The online controller does not subscribe to `/sim/thermal_sources_truth`; that
topic is for collector and benchmark evaluation only.

The Gazebo world, scenario file, and `sensor_node.py` fallback must stay consistent:

```text
thermal_bringup/worlds/thermal_scene_nav.world
thermal_sensor_sim/thermal_sensor_sim/sensor_node.py
```

## Build

Interfaces must be built before packages that import them.

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash

colcon build --packages-select thermal_interfaces
source install/setup.bash

colcon build --packages-select \
  g1_description \
  thermal_sensor_sim \
  signal_preprocessor \
  thermal_field_reconstructor \
  thermal_gradient_processor \
  thermal_motion_controller \
  thermal_bringup

source install/setup.bash
```

After rebuilding, the startup logs should show current versions such as:

```text
sensor_node v13
preprocessor_node v2
thermal_mapper_node
source_tracker_node
gradient_node v2
controller_node v31
```

## Operation Checklist

Use this sequence for a normal simulation run. The full command handbook is in
`/home/hanchen/ros2_ws/PROJECT_ANALYSIS_REPORT.md`.

Every terminal:

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
```

Before launching:

```bash
rosdep check --from-paths src/thermal_robot --ignore-src
python3 -m pytest src/thermal_robot/tests/test_thermal_system.py -q
ros2 launch thermal_bringup sim_nav_slam_launch.py --show-args
```

Full GUI run, with Gazebo client and RViz:

```bash
ros2 launch thermal_bringup sim_nav_slam_launch.py use_rviz:=true use_gzclient:=true
```

RViz starts about 20 seconds after launch. If `use_rviz:=false` is used, no RViz
window will appear.

Headless run:

```bash
ros2 launch thermal_bringup sim_nav_slam_launch.py use_rviz:=false use_gzclient:=false
```

Startup timing:

```text
t=0s   gzserver + robot_state_publisher
t=5s   gzclient, when use_gzclient:=true
t=6s   robot spawn at x=-6, y=0
t=9s   slam_toolbox
t=12s  Nav2
t=18s  thermal pipeline + controller_node
t=20s  colorizer_node + RViz, when use_rviz:=true
```

After launch, check core runtime state:

```bash
ros2 node list
ros2 action info /navigate_to_pose
ros2 lifecycle get /planner_server
ros2 lifecycle get /controller_server
ros2 lifecycle get /bt_navigator
```

Check core topic rates:

```bash
ros2 topic hz /sim/thermal_raw
ros2 topic hz /sim/thermal_colorized
ros2 topic hz /thermal/filtered
ros2 topic hz /thermal/field
ros2 topic hz /thermal/map
ros2 topic hz /thermal/sources
ros2 topic hz /sim/thermal_sources_truth
ros2 topic hz /thermal/gradient
ros2 topic hz /odom
ros2 topic hz /scan
ros2 topic hz /cmd_vel
```

Expected steady-state rates are roughly:

```text
/sim/thermal_raw        10 Hz
/sim/thermal_colorized  10 Hz
/thermal/filtered       10 Hz
/thermal/field          10 Hz
/thermal/map            5 Hz
/thermal/sources        5 Hz
/thermal/gradient       10 Hz
/odom                   about 20 Hz
/scan                   10 Hz
```

Query thermal field summary:

```bash
ros2 service call /thermal/get_field_info \
  thermal_interfaces/srv/GetFieldInfo "{include_full_data: false}"
```

Collect structured data after the launch has been running for about 20 seconds:

```bash
python3 src/thermal_robot/scripts/collect_sim_data.py --duration 120
```

Stop the collector with Ctrl+C. Data is written to:

```text
bags/collected/<YYYYMMDD_HHMMSS>/
```

Generate analysis figures:

```bash
python3 src/thermal_robot/scripts/plot_all_figures.py bags/collected/<timestamp>
python3 src/thermal_robot/scripts/plot_slam_nav2.py bags/collected/<timestamp>
python3 src/thermal_robot/scripts/source_benchmark.py bags/collected/<timestamp>
```

Stop or clean Gazebo:

```bash
bash src/thermal_robot/kill_gz.sh
```

`plot_all_figures.py`, `plot_slam_nav2.py`, and `collect_sim_data.py` match the current Config-B scenario.

## Controller Notes

`thermal_motion_controller` currently starts in `FRONTIER_NAV`. It subscribes
to `/thermal/map` and `/thermal/sources` for global exploration, uses thermal
signal strength to switch into fine local behavior, and uses Nav2 for coarse
navigation when available.

Important states:

```text
FRONTIER_NAV     large-scale frontier target selection
COARSE_SURVEY    systematic cold-region survey, Nav2 preferred
SURVEY_PAUSE     stop and sense for thermal signal
ASCENT           direct gradient following
CONVERGE         slow fine approach to a source
SAMPLE           stop and confirm a source
AT_PEAK          hold after confirmation
RELOCATE         leave confirmed source exclusion zone
DEPARTURE        direct move away from known-source centroid
ESCAPE           stuck or exclusion-zone recovery
DONE             all expected sources found
```

Source confirmation is source-estimate based: `SAMPLE` records the tracker
estimate position, not the robot pose. The legacy pixel-gradient chain remains
the fine-approach signal, while global target choice now uses
information-gain, source-probability, coverage, travel-cost, duplicate, and
risk terms from the world map.

## Important Caveats

- Treat `src/` as the source of truth. Rebuild before running if `install/` may be stale.
- `g1_nav.urdf` is the active simulation model for current launch files. The full G1 URDF assets remain in `g1_description` but are not the main runtime model.
- `use_sim_time` currently defaults to `false` in launch/config. Change it only after validating Gazebo `/clock`, TF, SLAM, and Nav2 timing together.
- The current source launch file is `sim_nav_slam_launch.py`. Older non-SLAM launch and plotting scripts have been removed.

## References

- Macenski et al. (2022) DOI:10.1126/scirobotics.abm6074
- Wiedemann et al. (2021) DOI:10.1016/j.robot.2020.103687
- Nakagawa et al. (2020) DOI:10.1109/JSEN.2020.2984234
- Reggente & Lilienthal (2009) DOI:10.1109/ICSENS.2009.5398427
- Sousa et al. (2006) DOI:10.1109/ROBOT.2006.1642286
