#!/usr/bin/env python3
"""
sim_nav_slam_launch.py — G1 热导航仿真启动文件（SLAM + Nav2 版）
================================================================
架构（v30，相比 v1 新增 SLAM + Nav2）：
  Gazebo → /scan(LaserScan) + /odom → slam_toolbox → /map + TF(map→odom)
  /map + /scan + /odom → Nav2（规划+控制）→ /cmd_vel
  thermal_pipeline → controller_node → NavigateToPose Action → Nav2
                                      ↘ 直接 /cmd_vel（FINE 模式）

启动顺序：
  t=0s:   gzserver（载入 world_file，默认 thermal_scene_nav.world）
  t=0s:   robot_state_publisher（g1_nav.urdf，含激光雷达 TF）
  t=5s:   gzclient（use_gzclient:=true 时）
  t=6s:   spawn_entity（x=-6, y=0）
  t=9s:   slam_toolbox（在线异步建图，等待机器人 spawn 后再启动）
  t=12s:  Nav2 全套节点（等待 SLAM 稳定后启动）
  t=18s:  thermal pipeline（等待 Nav2 完全就绪）
  t=20s:  colorizer + RViz

话题接口（新增）：
  /scan              sensor_msgs/LaserScan  10Hz  → slam_toolbox
  /map               nav_msgs/OccupancyGrid       ← slam_toolbox
  /navigate_to_pose  Action                       ← controller_node → Nav2

参考：
  Macenski 2020 DOI:10.1109/IROS45743.2020.9341207 (Nav2)
  Macenski 2021 arXiv:2010.10195 (slam_toolbox)
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup_dir = get_package_share_directory('thermal_bringup')
    g1_dir      = get_package_share_directory('g1_description')

    # ── 文件路径 ───────────────────────────────────────────────────────────
    default_world_file = os.path.join(bringup_dir, 'worlds',  'thermal_scene_nav.world')
    rviz_config   = os.path.join(bringup_dir, 'rviz',    'thermal_nav_slam.rviz')
    params_file   = os.path.join(bringup_dir, 'config',  'params.yaml')
    nav2_params   = os.path.join(bringup_dir, 'config',  'nav2_params.yaml')
    slam_params   = os.path.join(bringup_dir, 'config',  'slam_params.yaml')
    urdf_path     = os.path.join(g1_dir,      'urdf',    'g1_nav.urdf')  # v2（含激光雷达）

    # ── URDF 读取 ──────────────────────────────────────────────────────────
    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    # ── Launch 参数 ────────────────────────────────────────────────────────
    use_rviz     = LaunchConfiguration('use_rviz',     default='true')
    use_gzclient = LaunchConfiguration('use_gzclient', default='true')
    use_sim_t    = LaunchConfiguration('use_sim_time', default='false')
    scenario_file = LaunchConfiguration('scenario_file', default='')
    world_file = LaunchConfiguration('world_file', default=default_world_file)

    gz_env = dict(os.environ)
    gz_env['QT_QPA_PLATFORM']   = 'xcb'
    gz_env['GAZEBO_MODEL_PATH'] = ':'.join([
        g1_dir,
        gz_env.get('GAZEBO_MODEL_PATH', '')
    ])

    # ── Nav2 节点列表（由 lifecycle_manager_nav 管理）────────────────────
    # 注意：nav2_amcl 不需要（slam_toolbox 已提供定位）
    nav2_node_names = [
        'controller_server',
        'smoother_server',
        'planner_server',
        'behavior_server',
        'bt_navigator',
        'waypoint_follower',
        'velocity_smoother',
    ]

    return LaunchDescription([
        DeclareLaunchArgument('use_rviz',     default_value='true'),
        DeclareLaunchArgument('use_gzclient', default_value='true'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('scenario_file', default_value=''),
        DeclareLaunchArgument('world_file', default_value=default_world_file),

        # ══════════════════════════════════════════════════════════════════
        # t=0s: Gazebo + Robot State Publisher
        # ══════════════════════════════════════════════════════════════════
        ExecuteProcess(
            cmd=['gzserver', '--verbose', world_file,
                 '-s', 'libgazebo_ros_init.so',
                 '-s', 'libgazebo_ros_factory.so'],
            additional_env=gz_env,
            output='screen',
        ),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': use_sim_t,
            }],
            output='screen',
        ),

        # ══════════════════════════════════════════════════════════════════
        # t=5s: gzclient
        # ══════════════════════════════════════════════════════════════════
        TimerAction(period=5.0, actions=[
            ExecuteProcess(
                cmd=['gzclient'],
                additional_env=gz_env,
                output='screen',
                condition=IfCondition(use_gzclient),
            )
        ]),

        # ══════════════════════════════════════════════════════════════════
        # t=6s: spawn robot
        # ══════════════════════════════════════════════════════════════════
        TimerAction(period=6.0, actions=[
            ExecuteProcess(
                cmd=['ros2', 'run', 'gazebo_ros', 'spawn_entity.py',
                     '-entity', 'g1_nav_robot',
                     '-topic',  'robot_description',
                     '-x', '-6.0',
                     '-y',  '0.0',
                     '-z',  '0.141'],
                output='screen',
            )
        ]),

        # ══════════════════════════════════════════════════════════════════
        # t=9s: slam_toolbox（在线异步建图）
        #   发布: /map, TF(map→odom)
        #   订阅: /scan, /odom, TF(odom→base_link)
        # ══════════════════════════════════════════════════════════════════
        TimerAction(period=9.0, actions=[
            Node(
                package='slam_toolbox',
                executable='async_slam_toolbox_node',
                name='slam_toolbox',
                parameters=[
                    slam_params,
                    {'use_sim_time': use_sim_t}
                ],
                output='screen',
            ),
        ]),

        # ══════════════════════════════════════════════════════════════════
        # t=12s: Nav2 全套节点
        # ══════════════════════════════════════════════════════════════════
        TimerAction(period=12.0, actions=[
            # controller_server：局部路径跟踪，输出 /cmd_vel
            Node(
                package='nav2_controller',
                executable='controller_server',
                name='controller_server',
                parameters=[nav2_params, {'use_sim_time': use_sim_t}],
                output='screen',
            ),
            # smoother_server：路径平滑
            Node(
                package='nav2_smoother',
                executable='smoother_server',
                name='smoother_server',
                parameters=[nav2_params, {'use_sim_time': use_sim_t}],
                output='screen',
            ),
            # planner_server：全局路径规划（NavFn A*）
            Node(
                package='nav2_planner',
                executable='planner_server',
                name='planner_server',
                parameters=[nav2_params, {'use_sim_time': use_sim_t}],
                output='screen',
            ),
            # behavior_server：恢复行为（spin, backup, wait）
            Node(
                package='nav2_behaviors',
                executable='behavior_server',
                name='behavior_server',
                parameters=[nav2_params, {'use_sim_time': use_sim_t}],
                output='screen',
            ),
            # bt_navigator：行为树导航，处理 NavigateToPose Action
            Node(
                package='nav2_bt_navigator',
                executable='bt_navigator',
                name='bt_navigator',
                parameters=[nav2_params, {'use_sim_time': use_sim_t}],
                output='screen',
            ),
            # waypoint_follower：路径点跟随（可选）
            Node(
                package='nav2_waypoint_follower',
                executable='waypoint_follower',
                name='waypoint_follower',
                parameters=[nav2_params, {'use_sim_time': use_sim_t}],
                output='screen',
            ),
            # velocity_smoother：速度平滑（消除急动）
            Node(
                package='nav2_velocity_smoother',
                executable='velocity_smoother',
                name='velocity_smoother',
                parameters=[nav2_params, {'use_sim_time': use_sim_t}],
                output='screen',
            ),
            # lifecycle_manager：管理 Nav2 所有节点的生命周期
            Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='lifecycle_manager_nav',
                parameters=[{
                    'use_sim_time': use_sim_t,
                    'autostart': True,
                    'node_names': nav2_node_names,
                }],
                output='screen',
            ),
        ]),

        # ══════════════════════════════════════════════════════════════════
        # t=18s: thermal pipeline（等待 Nav2 完全就绪后启动）
        # ══════════════════════════════════════════════════════════════════
        TimerAction(period=18.0, actions=[
            Node(
                package='thermal_sensor_sim',
                executable='sensor_node',
                name='sensor_node',
                parameters=[params_file, {
                    'use_sim_time': use_sim_t,
                    'scenario_file': scenario_file,
                }],
                output='both',
            ),
            Node(
                package='signal_preprocessor',
                executable='preprocessor_node',
                name='preprocessor_node',
                parameters=[params_file, {'use_sim_time': use_sim_t}],
                output='both',
            ),
            Node(
                package='thermal_field_reconstructor',
                executable='reconstructor_node',
                name='reconstructor_node',
                parameters=[params_file, {'use_sim_time': use_sim_t}],
                output='both',
            ),
            Node(
                package='thermal_field_reconstructor',
                executable='thermal_mapper_node',
                name='thermal_mapper_node',
                parameters=[params_file, {'use_sim_time': use_sim_t}],
                output='both',
            ),
            Node(
                package='thermal_gradient_processor',
                executable='gradient_node',
                name='gradient_node',
                parameters=[params_file, {'use_sim_time': use_sim_t}],
                output='both',
            ),
            Node(
                package='thermal_motion_controller',
                executable='source_tracker_node',
                name='source_tracker_node',
                parameters=[params_file, {'use_sim_time': use_sim_t}],
                output='both',
            ),
            # controller_node v30: 热导航决策层
            # - FINE 模式（ASCENT/CONVERGE/SAMPLE）: 直接发布 /cmd_vel
            # - COARSE 模式（DEPARTURE/COARSE_SURVEY/FRONTIER）: NavigateToPose Action
            Node(
                package='thermal_motion_controller',
                executable='controller_node',
                name='controller_node',
                parameters=[params_file, {'use_sim_time': use_sim_t}],
                output='both',
            ),
        ]),

        # ══════════════════════════════════════════════════════════════════
        # t=20s: colorizer + RViz
        # ══════════════════════════════════════════════════════════════════
        TimerAction(period=20.0, actions=[
            Node(
                package='thermal_sensor_sim',
                executable='colorizer_node',
                name='colorizer_node',
                parameters=[{'t_min': 22.0, 't_max': 65.0}],
                output='both',
            ),
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['-d', rviz_config],
                output='screen',
                condition=IfCondition(use_rviz),
            ),
        ]),
    ])
