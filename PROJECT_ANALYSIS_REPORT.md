# thermal_robot 项目交接分析报告

生成时间：2026-04-27  
工作区：`/home/hanchen/ros2_ws`  
结论版本：基于整理后的当前实际源码

## 1. 总体结论

这个工作区现在已经整理成一条明确主线：

```text
Gazebo g1_nav 简化差速模型
  -> /odom + /scan
  -> slam_toolbox
  -> /map + map->odom TF
  -> Nav2 NavigateToPose

/sim/thermal_raw
  -> /thermal/filtered
  -> /thermal/field + /thermal/get_field_info
  -> /thermal/gradient
  -> controller_node
       FINE: 直接 /cmd_vel
       COARSE: Nav2 优先，直接 /cmd_vel 兜底
```

当前唯一主线启动文件是：

```text
src/thermal_robot/thermal_bringup/launch/sim_nav_slam_launch.py
```

旧的非 SLAM launch、旧 RViz、旧测试 runner、旧绘图脚本、旧 Dockerfile、Python/pytest 缓存都已删除。`build/`、`install/`、`log/`、`bags/`、本地工具配置已经通过 `.gitignore` 忽略。

当前项目可以构建，依赖声明可由 `rosdep` 解析，纯算法测试通过。主线 launch 能独立拉起 Gazebo、SLAM Toolbox、Nav2、thermal pipeline 和 controller；但从最近实际运行日志看，Nav2 规划闭环仍有 costmap/map 边界问题，需要后续继续修。

## 2. 当前仓库状态

根目录关键文件：

```text
.gitignore
CLAUDE.md
PROJECT_ANALYSIS_REPORT.md
src/thermal_robot/
```

本地存在但已忽略的生成/数据目录：

```text
build/
install/
log/
bags/
.codex
.claude/
```

当前 `git status --short --ignored` 的重要含义：

```text
?? .gitignore
?? CLAUDE.md
?? PROJECT_ANALYSIS_REPORT.md
?? src/
!! build/
!! install/
!! log/
!! bags/
```

说明源码和文档还未提交；构建产物、日志、实验数据已经不再污染版本控制。

工作区体积约 `506M`。其中 `bags/` 是实验数据，`g1_description/meshes` 是完整 G1 模型资产的大头。当前主线运行不依赖 mesh，但如果未来恢复完整 G1 可视化/模型路径，mesh 仍有保留价值。

## 3. 已删除的旧文件

以下文件已经从源码删除，删除后不影响当前主线：

```text
src/thermal_robot/thermal_bringup/launch/sim_nav_launch.py
src/thermal_robot/thermal_bringup/rviz/thermal_nav.rviz
src/thermal_robot/scripts/run_tests.sh
src/thermal_robot/tests/run_full_test.sh
src/thermal_robot/scripts/plot_metrics.py
src/thermal_robot/scripts/plot_from_log.py
src/thermal_robot/Dockerfile
src/thermal_robot/tests/__pycache__/
src/thermal_robot/.pytest_cache/
```

对应的旧安装副本也已从 `install/thermal_bringup/share/thermal_bringup` 清理。当前源码和安装空间里只保留：

```text
sim_nav_slam_launch.py
thermal_nav_slam.rviz
```

## 4. 包结构

当前 `colcon list` 识别 8 个包：

```text
thermal_interfaces             ament_cmake
g1_description                 ament_cmake
thermal_sensor_sim             ament_python
signal_preprocessor            ament_python
thermal_field_reconstructor    ament_python
thermal_gradient_processor     ament_python
thermal_motion_controller      ament_python
thermal_bringup                ament_cmake
```

包职责：

```text
thermal_interfaces
  自定义 msg/srv：ThermalPoint, ThermalField, Gradient, GradientArray, GetFieldInfo

g1_description
  当前主线使用 g1_nav.urdf；完整 G1 URDF/xacro/meshes 作为可选资产保留

thermal_sensor_sim
  sensor_node 发布 /sim/thermal_raw
  colorizer_node 发布 /sim/thermal_colorized

signal_preprocessor
  preprocessor_node 将 /sim/thermal_raw 滤波为 /thermal/filtered

thermal_field_reconstructor
  reconstructor_node 将 filtered 图像转为 ThermalField，并提供 /thermal/get_field_info

thermal_gradient_processor
  gradient_node 从 ThermalField 计算 Sobel/central diff 梯度，发布 /thermal/gradient

thermal_motion_controller
  controller_node v31，多状态热导航控制器，集成 Nav2 action 和 /cmd_vel 兜底

thermal_bringup
  主线 launch、参数、Nav2/SLAM 配置、RViz、Gazebo world
```

## 5. 主线启动

主线启动命令：

```bash
cd /home/hanchen/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch thermal_bringup sim_nav_slam_launch.py
```

无 RViz：

```bash
ros2 launch thermal_bringup sim_nav_slam_launch.py use_rviz:=false
```

无 RViz 且不启动 Gazebo GUI：

```bash
ros2 launch thermal_bringup sim_nav_slam_launch.py use_rviz:=false use_gzclient:=false
```

当前 launch 参数：

```text
use_rviz       default true
use_gzclient   default true
use_sim_time   default false
```

启动顺序：

```text
t=0s   gzserver + robot_state_publisher
t=5s   gzclient（use_gzclient:=true）
t=6s   spawn_entity，机器人出生在 x=-6.0, y=0.0
t=9s   async_slam_toolbox_node
t=12s  Nav2 controller/smoother/planner/behavior/bt/waypoint/velocity_smoother/lifecycle
t=18s  thermal pipeline：sensor/preprocessor/reconstructor/gradient/controller
t=20s  colorizer + RViz（use_rviz:=true）
```

## 6. 运行话题和接口

当前主线关键话题：

```text
/sim/thermal_raw          sensor_msgs/Image 32FC1       sensor_node           10 Hz
/sim/thermal_colorized    sensor_msgs/Image rgb8         colorizer_node        10 Hz
/thermal/filtered         sensor_msgs/Image 32FC1        preprocessor_node     10 Hz
/thermal/field            thermal_interfaces/ThermalField reconstructor_node   10 Hz
/thermal/gradient         thermal_interfaces/GradientArray gradient_node       10 Hz
/thermal/get_field_info   thermal_interfaces/GetFieldInfo reconstructor_node   service
/cmd_vel                  geometry_msgs/Twist            controller/Nav2       about 10 Hz
/odom                     nav_msgs/Odometry              Gazebo diff_drive     about 20 Hz
/scan                     sensor_msgs/LaserScan          Gazebo ray sensor     10 Hz
/map                      nav_msgs/OccupancyGrid         slam_toolbox
/plan                     nav_msgs/Path                  Nav2 planner
/navigate_to_pose         nav2_msgs/action/NavigateToPose Nav2 action
```

已经确认主线 RViz 配置使用实际存在的话题：

```text
/sim/thermal_colorized
/thermal/filtered
/map
/plan
/odom
/scan
```

不存在发布者的旧话题 `/thermal/colorized`、`/thermal/markers`、`/thermal/robot_path`、`/thermal/gradient_markers` 已不再被源码引用。

## 7. 热场场景

当前 Config-B 热源配置在 `sensor_node.py` 中硬编码，并与 README/数据脚本保持一致：

```text
SA_left  world=(-1.0,  3.5)  amplitude=35.0  sigma=1.1  peak≈57C
SB_far   world=( 6.0, -3.0)  amplitude=22.0  sigma=0.9  peak≈44C
SC_weak  world=(-5.0, -5.5)  amplitude=16.0  sigma=0.8  peak≈38C
```

机器人出生点：

```text
spawn_x = -6.0
spawn_y =  0.0
```

热相机仿真：

```text
resolution: 64 x 48
FOV: 4.0m x 3.0m
ambient: 22.0C
noise_std: 0.5C
publish_rate: 10Hz
```

## 8. 控制器状态机

当前控制器是 `controller_node v31`。核心修复点：

```text
FIX-1: SLAM map 坐标加 spawn 偏移，得到 world 坐标
FIX-2: COARSE_SURVEY 在 Nav2 不可用/失败时直接 /cmd_vel 兜底
FIX-3: Nav2 goal 使用 map frame，world -> map 时减 spawn 偏移
FIX-4: FRONTIER_NAV 保证持续输出运动指令
```

状态集合：

```text
FINE states:
  ASCENT
  CONVERGE
  SAMPLE
  AT_PEAK
  RELOCATE
  ESCAPE

COARSE states:
  FRONTIER_NAV
  COARSE_SURVEY
  DEPARTURE
  SURVEY_PAUSE

Terminal:
  DONE
```

设计原则：

```text
局部热信号强时：直接梯度跟随和确认
大范围探索时：Nav2 NavigateToPose 优先
Nav2 不可用或失败时：直接 /cmd_vel 继续移动
热源坐标不作为先验输入，导航由传感器信号和 belief/frontier 逻辑驱动
```

## 9. 参数配置

主参数文件：

```text
src/thermal_robot/thermal_bringup/config/params.yaml
```

重要参数：

```text
sensor_node.publish_rate = 10.0
sensor_node.image_width = 64
sensor_node.image_height = 48
sensor_node.num_sources = 3

preprocessor_node.filter_method = kalman
preprocessor_node.kalman_q = 0.08
preprocessor_node.kalman_r = 4.0
preprocessor_node.spatial_smooth = true
preprocessor_node.spatial_sigma = 0.8

gradient_node.method = sobel
gradient_node.subsample_stride = 4

controller_node.publish_rate = 10.0
controller_node.max_linear_vel = 0.25
controller_node.max_angular_vel = 0.5
controller_node.num_sources = 3
controller_node.no_new_source_timeout = 60.0
controller_node.post_confirm_rounds = 10
controller_node.post_confirm_min_d = 7.0
controller_node.levy_post_confirm_step = 10.0
```

Nav2 配置：

```text
src/thermal_robot/thermal_bringup/config/nav2_params.yaml
```

关键设计：

```text
controller: RegulatedPurePursuitController
planner: NavfnPlanner with A*
xy_goal_tolerance: 1.5m
allow_unknown: true
local_costmap: odom frame, rolling window 6m x 6m
global_costmap: map frame, static + obstacle + inflation
```

SLAM 配置：

```text
src/thermal_robot/thermal_bringup/config/slam_params.yaml
```

关键设计：

```text
mode: mapping
scan_topic: /scan
odom_frame: odom
map_frame: map
base_frame: base_link
resolution: 0.05
max_laser_range: 12.0
transform_publish_period: 0.02
map_update_interval: 5.0
```

## 10. 构建与依赖

构建顺序：

```bash
cd /home/hanchen/ros2_ws
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

依赖整理结果：

```text
rosdep check --from-paths src/thermal_robot --ignore-src
=> All system dependencies have been satisfied
```

已补齐或修正的典型依赖：

```text
nav_msgs
python3-numpy
python3-matplotlib
python3-pytest
gazebo_ros
gazebo_plugins
Nav2 具体节点/插件包
rviz_default_plugins
nav2_rviz_plugins
```

已移除不符合当前写法或旧路径的依赖：

```text
ament_python buildtool_depend
ros2_control / ros2_controllers / gazebo_ros2_control
joint_state_publisher / joint_state_publisher_gui
```

说明：ROS Humble 官方 Python 包通常只在 `<export>` 中声明 `<build_type>ament_python</build_type>`，不写 `ament_python` 作为 buildtool rosdep key。本项目已按此方式整理。

## 11. 测试状态

当前测试文件：

```text
src/thermal_robot/tests/test_thermal_system.py
```

纯算法测试覆盖：

```text
T-PY1  高斯热场范围
T-PY2  Sobel 梯度方向
T-PY3  Kalman 降噪
T-PY4  梯度上升收敛
T-PY5  热场连续性
T-PY6  多热源热点检测
T-PY7  热源中心梯度小
T-PY8  Sobel/central diff 一致性
```

最近验证：

```text
python3 -m pytest src/thermal_robot/tests/test_thermal_system.py -q
=> 8 passed
```

ROS 集成测试入口仍在该文件中，通过 `--ros` 触发，需要先启动主线 launch。它已不再订阅旧 `/thermal/robot_path`。

## 12. 数据采集和绘图

当前保留的数据工具：

```text
src/thermal_robot/scripts/collect_sim_data.py
src/thermal_robot/scripts/plot_all_figures.py
src/thermal_robot/scripts/plot_slam_nav2.py
src/thermal_robot/scripts/send_cmd_vel.sh
```

推荐流程：

```bash
# Terminal 1
cd /home/hanchen/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch thermal_bringup sim_nav_slam_launch.py use_rviz:=false use_gzclient:=false

# Terminal 2，等待主线启动约 20s 后
cd /home/hanchen/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 src/thermal_robot/scripts/collect_sim_data.py

# 采集结束后
python3 src/thermal_robot/scripts/plot_all_figures.py bags/collected/<timestamp>
python3 src/thermal_robot/scripts/plot_slam_nav2.py bags/collected/<timestamp>
```

`bags/` 已被 `.gitignore` 忽略。建议不要把实验原始数据直接提交到源码仓库；需要保留基准数据时，应单独归档或明确挑选小型样例。

## 13. 当前验证记录

本轮整理后已经执行：

```text
colcon list
  8 packages recognized

rosdep check --from-paths src/thermal_robot --ignore-src
  All system dependencies have been satisfied

colcon build
  thermal_interfaces 通过
  其余 7 个包通过

python3 -m pytest src/thermal_robot/tests/test_thermal_system.py -q
  8 passed

ros2 launch thermal_bringup sim_nav_slam_launch.py --show-args
  use_rviz, use_gzclient, use_sim_time 三个参数可见
```

之前短时运行主线时已确认：

```text
Gazebo 启动
机器人 spawn 成功
/odom 发布
slam_toolbox 启动并注册 /scan
Nav2 lifecycle 进入 active
thermal pipeline 五个核心节点启动
controller_node v31 连接 NavigateToPose action
/map /scan /odom /cmd_vel /thermal/* 等主线话题存在
```

同时观察到 Nav2 规划问题，见下一节。

## 14. 仍需关注的问题

### 14.1 Nav2 costmap/map 边界问题

短时运行时出现过：

```text
Robot is out of bounds of the costmap
Received map message is malformed
Cannot create a plan: the robot's start position is off the global costmap
Planning algorithm GridBased failed
```

这说明主线能启动完整进程链，但 Nav2 全局规划闭环还不健康。当前 controller 有直接 `/cmd_vel` 兜底，所以机器人不会完全停死；但如果要证明 SLAM + Nav2 闭环可靠，需要优先修这个问题。

建议排查：

```text
slam_toolbox 初始 map 范围和 origin
global_costmap 的 static_layer 与未知空间处理
机器人 spawn 世界坐标、map 坐标、controller world/map 转换
Nav2 goal 是否落在当前可规划地图范围内
map_update_interval 导致的地图扩张滞后
```

### 14.2 Nav2 BT XML 仍是绝对路径

`nav2_params.yaml` 中仍有：

```text
/home/hanchen/ros2_ws/install/thermal_bringup/share/thermal_bringup/config/navigate_to_pose_simple.xml
```

这在当前机器可用，但换用户、换目录或容器后会失效。建议后续把 BT XML 路径从 launch 文件中动态注入，或在启动时生成替换参数。

### 14.3 use_sim_time 默认 false

当前 Gazebo/SLAM/Nav2 统一使用 wall time：

```text
use_sim_time: false
```

这不是 Gazebo 常见默认，但当前主线按这个配置能启动。不要单独改一个节点的时间源；若切到 simulated time，必须一起验证 `/clock`、TF、slam_toolbox、Nav2 lifecycle、Gazebo 插件时间戳。

### 14.4 reconstructor shutdown

`reconstructor_node.py` 的退出路径仍直接调用 `rclpy.shutdown()`，和其他节点的 `try_shutdown()` 风格不完全一致。之前见过 Ctrl-C 后二次 shutdown 异常。该问题不影响算法，但会污染日志，建议后续统一处理。

### 14.5 完整 G1 资产是否保留

当前主线使用：

```text
src/thermal_robot/g1_description/urdf/g1_nav.urdf
```

不依赖：

```text
src/thermal_robot/g1_description/urdf/g1_thermal.urdf.xacro
src/thermal_robot/g1_description/urdf/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf
src/thermal_robot/g1_description/meshes/
```

这些是完整 G1 资产，体积大，但可能对未来恢复完整模型有价值。当前没有删除，建议等项目目标明确后再决定是否归档。

## 15. 后续开发建议

优先级从高到低：

```text
1. 修 Nav2 costmap/map 边界和规划失败问题
2. 把 nav2_params.yaml 的 BT XML 绝对路径改为 launch 动态路径
3. 运行 10-15 分钟主线实验，采集 collect_sim_data.py 数据
4. 用 plot_all_figures.py / plot_slam_nav2.py 生成结果并记录指标
5. 针对 SB_far 和 SC_weak 的发现率优化 COARSE_SURVEY/frontier 策略
6. 统一各节点 shutdown 行为
7. 决定完整 G1 资产是否保留在源码仓库
```

当前不建议再从旧非 SLAM 路径恢复工作。所有实验、修复和报告都应围绕：

```text
sim_nav_slam_launch.py
g1_nav.urdf
Config-B thermal scenario
controller_node v31
SLAM Toolbox + Nav2 + direct /cmd_vel fallback
```

## 16. 仿真全流程命令汇总

这一节按一次完整实验的实际操作顺序整理。后续接手时，优先按这里执行；不要再找旧的 `sim_nav_launch.py`、`thermal_nav.rviz`、`plot_metrics.py` 或 `run_tests.sh`。

### 16.1 所有终端的通用环境

每开一个新终端都先进入工作区并加载环境：

```bash
cd /home/hanchen/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
```

如果 `install/setup.bash` 不存在，说明还没有构建，先执行 16.2。

### 16.2 首次接手或代码变化后的构建

先检查系统依赖：

```bash
cd /home/hanchen/ros2_ws
source /opt/ros/humble/setup.bash
rosdep check --from-paths src/thermal_robot --ignore-src
```

如果依赖缺失，用 rosdep 安装缺失包：

```bash
rosdep install --from-paths src/thermal_robot --ignore-src -r -y
```

构建顺序建议先构建接口包，再构建其余包：

```bash
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

确认 ROS 2 能找到 8 个包：

```bash
colcon list
```

当前应包含：

```text
g1_description
signal_preprocessor
thermal_bringup
thermal_field_reconstructor
thermal_gradient_processor
thermal_interfaces
thermal_motion_controller
thermal_sensor_sim
```

### 16.3 启动仿真前检查

查看主线 launch 参数：

```bash
ros2 launch thermal_bringup sim_nav_slam_launch.py --show-args
```

当前有效参数：

```text
use_rviz       default true
use_gzclient   default true
use_sim_time   default false
```

运行纯算法测试，确认 Python 侧基础逻辑没有坏：

```bash
python3 -m pytest src/thermal_robot/tests/test_thermal_system.py -q
```

如果 ROS 日志目录权限异常，可以给本次启动指定临时日志目录：

```bash
mkdir -p /tmp/ros2_ws_logs
ROS_LOG_DIR=/tmp/ros2_ws_logs ros2 launch thermal_bringup sim_nav_slam_launch.py --show-args
```

### 16.4 启动完整 GUI 仿真

Terminal 1：

```bash
cd /home/hanchen/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch thermal_bringup sim_nav_slam_launch.py use_rviz:=true use_gzclient:=true
```

等效简写：

```bash
ros2 launch thermal_bringup sim_nav_slam_launch.py
```

预期启动节奏：

```text
t=0s   gzserver + robot_state_publisher
t=5s   Gazebo 图形客户端 gzclient
t=6s   spawn g1_nav_robot，初始位姿 x=-6, y=0
t=9s   slam_toolbox
t=12s  Nav2 planner/controller/bt/lifecycle
t=18s  thermal pipeline + controller_node
t=20s  colorizer_node + RViz
```

所以 RViz 不是立刻出现，要等约 20 秒；如果启动时写了 `use_rviz:=false`，就不会有 RViz 界面。

### 16.5 不同启动模式

只开 RViz，不开 Gazebo 图形客户端：

```bash
ros2 launch thermal_bringup sim_nav_slam_launch.py use_rviz:=true use_gzclient:=false
```

完全无界面启动，适合远程、测试或采集：

```bash
ros2 launch thermal_bringup sim_nav_slam_launch.py use_rviz:=false use_gzclient:=false
```

不要随意改 `use_sim_time`。当前主线默认是 `false`，报告中的验证也是基于这个默认值；改成 `true` 前需要同时验证 Gazebo `/clock`、TF、SLAM、Nav2 lifecycle 和 controller 的时间行为。

### 16.6 启动后基础检查

Terminal 2：

```bash
cd /home/hanchen/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
```

查看节点、话题、服务、action 是否存在：

```bash
ros2 node list
ros2 topic list
ros2 service list
ros2 action list
```

关键节点应包含：

```text
/robot_state_publisher
/slam_toolbox
/planner_server
/controller_server
/bt_navigator
/lifecycle_manager_nav
/sensor_node
/preprocessor_node
/reconstructor_node
/gradient_node
/controller_node
/colorizer_node
/rviz2                  # use_rviz:=true 时才有
```

关键 action：

```bash
ros2 action info /navigate_to_pose
```

Nav2 lifecycle 状态检查：

```bash
ros2 lifecycle get /planner_server
ros2 lifecycle get /controller_server
ros2 lifecycle get /bt_navigator
```

正常应为 `active`。如果不是 active，先看 Terminal 1 的 Nav2 lifecycle 日志。

### 16.7 话题频率检查

每条 `ros2 topic hz` 建议观察 5-10 秒后 Ctrl+C：

```bash
ros2 topic hz /sim/thermal_raw
ros2 topic hz /sim/thermal_colorized
ros2 topic hz /thermal/filtered
ros2 topic hz /thermal/field
ros2 topic hz /thermal/gradient
ros2 topic hz /odom
ros2 topic hz /scan
ros2 topic hz /cmd_vel
```

当前期望稳态频率：

```text
/sim/thermal_raw        10 Hz
/sim/thermal_colorized  10 Hz
/thermal/filtered       10 Hz
/thermal/field          10 Hz
/thermal/gradient       10 Hz
/odom                   about 20 Hz
/scan                   10 Hz
/cmd_vel                about 10 Hz when controller/Nav2 is commanding
```

查看一次热梯度消息：

```bash
ros2 topic echo /thermal/gradient --once
```

查询当前热场统计：

```bash
ros2 service call /thermal/get_field_info \
  thermal_interfaces/srv/GetFieldInfo "{include_full_data: false}"
```

### 16.8 TF、SLAM、地图检查

检查 odom 到 base_link：

```bash
ros2 run tf2_ros tf2_echo odom base_link
```

检查 SLAM 发布的 map 到 base_link：

```bash
ros2 run tf2_ros tf2_echo map base_link
```

检查地图是否发布：

```bash
ros2 topic echo /map --once
```

检查激光雷达：

```bash
ros2 topic echo /scan --once
```

如果 `/scan` 有数据但 `/map` 或 `map->base_link` 长时间没有，重点看 `slam_toolbox` 日志和 TF。

### 16.9 RViz 界面检查

RViz 使用：

```text
src/thermal_robot/thermal_bringup/rviz/thermal_nav_slam.rviz
```

RViz 中重点看：

```text
TF tree
LaserScan /scan
Map /map
RobotModel
Path /plan
Thermal raw/colorized image: /sim/thermal_colorized
Thermal field/gradient related displays
```

如果没有 RViz 窗口：

```bash
echo $DISPLAY
ros2 node list
```

判断顺序：

```text
1. 确认启动命令不是 use_rviz:=false
2. 等待至少 20 秒
3. 确认 /rviz2 是否出现在 ros2 node list
4. 确认当前环境有图形显示能力，尤其是 SSH/容器环境的 DISPLAY
5. 需要只有 RViz 时，用 use_rviz:=true use_gzclient:=false
```

### 16.10 数据采集

Terminal 3，等主线启动约 20-30 秒后开始采集：

```bash
cd /home/hanchen/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 src/thermal_robot/scripts/collect_sim_data.py
```

采集时保持 Terminal 1 的仿真继续运行。需要结束采集时，在 Terminal 3 按 Ctrl+C；脚本会保存 CSV、metadata 和 snapshots。

输出位置：

```text
bags/collected/<YYYYMMDD_HHMMSS>/
```

查看已有采集目录：

```bash
ls -td bags/collected/*
```

### 16.11 生成实验图表

指定采集目录生成完整热导航图：

```bash
python3 src/thermal_robot/scripts/plot_all_figures.py bags/collected/<YYYYMMDD_HHMMSS>
```

指定采集目录生成 SLAM/Nav2 图：

```bash
python3 src/thermal_robot/scripts/plot_slam_nav2.py bags/collected/<YYYYMMDD_HHMMSS>
```

也可以自动使用最新采集目录：

```bash
python3 src/thermal_robot/scripts/plot_all_figures.py
python3 src/thermal_robot/scripts/plot_slam_nav2.py
```

主要输出：

```text
bags/collected/<timestamp>/figures/
```

### 16.12 测试和基准

纯算法测试：

```bash
python3 -m pytest src/thermal_robot/tests/test_thermal_system.py -q
```

详细输出：

```bash
python3 -m pytest src/thermal_robot/tests/test_thermal_system.py -v
```

算法基准：

```bash
python3 src/thermal_robot/tests/test_thermal_system.py --bench
```

ROS 集成测试需要先启动主线 launch：

```bash
python3 src/thermal_robot/tests/test_thermal_system.py --ros
```

### 16.13 手动运动诊断

只用于诊断 `/cmd_vel -> /odom/Gazebo` 链路，不作为正常实验控制方式。因为主线 controller/Nav2 也会发布 `/cmd_vel`，手动测试可能和自动控制抢速度指令。

```bash
bash src/thermal_robot/scripts/send_cmd_vel.sh
bash src/thermal_robot/scripts/send_cmd_vel.sh 0.2 0.3
```

也可直接发一次 Twist：

```bash
ros2 topic pub --times 10 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.1, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

### 16.14 停止仿真和清理残留进程

正常停止：

```text
在 Terminal 1 按 Ctrl+C
```

如果 Gazebo 进程残留：

```bash
bash src/thermal_robot/kill_gz.sh
```

查看是否仍有 Gazebo 进程：

```bash
pgrep -a gzserver
pgrep -a gzclient
```

没有输出表示 Gazebo 已清理干净。

### 16.15 一次完整实验的最短命令顺序

Terminal 1，启动仿真：

```bash
cd /home/hanchen/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch thermal_bringup sim_nav_slam_launch.py use_rviz:=true use_gzclient:=true
```

Terminal 2，启动后检查：

```bash
cd /home/hanchen/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 node list
ros2 action info /navigate_to_pose
ros2 topic hz /thermal/gradient
ros2 service call /thermal/get_field_info \
  thermal_interfaces/srv/GetFieldInfo "{include_full_data: false}"
```

Terminal 3，采集和绘图：

```bash
cd /home/hanchen/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 src/thermal_robot/scripts/collect_sim_data.py
```

采集结束 Ctrl+C 后：

```bash
python3 src/thermal_robot/scripts/plot_all_figures.py
python3 src/thermal_robot/scripts/plot_slam_nav2.py
```

结束仿真：

```bash
bash src/thermal_robot/kill_gz.sh
```

## 17. 文件索引

核心启动与配置：

```text
src/thermal_robot/thermal_bringup/launch/sim_nav_slam_launch.py
src/thermal_robot/thermal_bringup/config/params.yaml
src/thermal_robot/thermal_bringup/config/nav2_params.yaml
src/thermal_robot/thermal_bringup/config/slam_params.yaml
src/thermal_robot/thermal_bringup/config/navigate_to_pose_simple.xml
src/thermal_robot/thermal_bringup/rviz/thermal_nav_slam.rviz
src/thermal_robot/thermal_bringup/worlds/thermal_scene_nav.world
```

机器人模型：

```text
src/thermal_robot/g1_description/urdf/g1_nav.urdf
src/thermal_robot/g1_description/urdf/g1_thermal.urdf.xacro
src/thermal_robot/g1_description/urdf/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf
src/thermal_robot/g1_description/meshes/
```

核心节点：

```text
src/thermal_robot/thermal_sensor_sim/thermal_sensor_sim/sensor_node.py
src/thermal_robot/thermal_sensor_sim/thermal_sensor_sim/colorizer_node.py
src/thermal_robot/signal_preprocessor/signal_preprocessor/preprocessor_node.py
src/thermal_robot/thermal_field_reconstructor/thermal_field_reconstructor/reconstructor_node.py
src/thermal_robot/thermal_gradient_processor/thermal_gradient_processor/gradient_node.py
src/thermal_robot/thermal_motion_controller/thermal_motion_controller/controller_node.py
```

接口：

```text
src/thermal_robot/thermal_interfaces/msg/ThermalPoint.msg
src/thermal_robot/thermal_interfaces/msg/ThermalField.msg
src/thermal_robot/thermal_interfaces/msg/Gradient.msg
src/thermal_robot/thermal_interfaces/msg/GradientArray.msg
src/thermal_robot/thermal_interfaces/srv/GetFieldInfo.srv
```

数据和测试：

```text
src/thermal_robot/scripts/collect_sim_data.py
src/thermal_robot/scripts/plot_all_figures.py
src/thermal_robot/scripts/plot_slam_nav2.py
src/thermal_robot/scripts/send_cmd_vel.sh
src/thermal_robot/tests/test_thermal_system.py
```

工程整理：

```text
.gitignore
CLAUDE.md
PROJECT_ANALYSIS_REPORT.md
```
