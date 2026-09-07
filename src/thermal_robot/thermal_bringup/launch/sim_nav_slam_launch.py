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
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction, OpaqueFunction, SetLaunchConfiguration, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnShutdown
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def prepare_thermal_world(context):
    import tempfile
    import yaml
    from thermal_sensor_sim.scenario import load_scenario_file, default_config_b_scenario, apply_run_seed
    from thermal_sensor_sim.surface_scene import write_surface_world
    original=LaunchConfiguration('world_file').perform(context)
    if LaunchConfiguration('sensor_model').perform(context)!='b':
        return [SetLaunchConfiguration('thermal_world_file',original)]
    scenario_path=LaunchConfiguration('scenario_file').perform(context)
    scenario=load_scenario_file(scenario_path,num_sources=0) if scenario_path else default_config_b_scenario(3)
    apply_run_seed(scenario,int(LaunchConfiguration('run_seed').perform(context)),
                   jitter_std_m=float(LaunchConfiguration('scenario_jitter_std_m').perform(context)))
    with open(LaunchConfiguration('software_params').perform(context)) as stream:
        config=yaml.safe_load(stream).get('sensor_node',{}).get('ros__parameters',{})
    fd,path=tempfile.mkstemp(prefix='thermal_surface_',suffix='.world');os.close(fd)
    write_surface_world(original,scenario,path,config.get('surface_height_m',.8),
                        config.get('surface_diameter_m',.3),config.get('surface_shape','box'))
    def cleanup(context):
        if os.path.exists(path):os.unlink(path)
        return []
    return [SetLaunchConfiguration('thermal_world_file',path),
            RegisterEventHandler(OnShutdown(on_shutdown=[OpaqueFunction(function=cleanup)]))]


def generate_launch_description():
    bringup_dir = get_package_share_directory('thermal_bringup')
    g1_dir      = get_package_share_directory('g1_description')

    # ── 文件路径 ───────────────────────────────────────────────────────────
    default_world_file = os.path.join(bringup_dir, 'worlds',  'thermal_scene_nav.world')
    rviz_config   = os.path.join(bringup_dir, 'rviz',    'thermal_nav_slam.rviz')
    params_file   = os.path.join(bringup_dir, 'config',  'params.yaml')
    nav2_params   = os.path.join(bringup_dir, 'config',  'nav2_params.yaml')
    nav2_bt_xml   = os.path.join(bringup_dir, 'config',  'navigate_to_pose_simple.xml')
    slam_params   = os.path.join(bringup_dir, 'config',  'slam_params.yaml')
    urdf_path     = os.path.join(g1_dir,      'urdf',    'g1_nav.urdf')  # v2（含激光雷达）

    # ── URDF 读取 ──────────────────────────────────────────────────────────
    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    # ── Launch 参数 ────────────────────────────────────────────────────────
    use_rviz     = LaunchConfiguration('use_rviz',     default='true')
    use_gzclient = LaunchConfiguration('use_gzclient', default='true')
    use_sim_t    = LaunchConfiguration('use_sim_time', default='true')
    scenario_file = LaunchConfiguration('scenario_file', default='')
    world_file = LaunchConfiguration('world_file', default=default_world_file)
    run_seed = LaunchConfiguration('run_seed', default='0')
    strategy = LaunchConfiguration('strategy', default='dual')
    sensor_model=LaunchConfiguration('sensor_model',default='a')
    belief_mode=LaunchConfiguration('belief_mode',default='online')
    software_params=LaunchConfiguration('software_params',default=os.path.join(bringup_dir,'config','multisource.yaml'))
    estimator_model=ParameterValue(PythonExpression(["'kalman' if '",strategy,"' in ('fast','dual','gp_ucb') else 'legacy'"]),value_type=str)
    fusion_memory=ParameterValue(PythonExpression(["3.0 if '",strategy,"' in ('fast','dual','gp_ucb') else 1.e9"]),value_type=float)
    shared={'use_sim_time':use_sim_t,'sensor_model':sensor_model}
    scenario_jitter = LaunchConfiguration('scenario_jitter_std_m', default='0.0')

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
        'planner_server',
        'behavior_server',
        'bt_navigator',
    ]

    return LaunchDescription([
        DeclareLaunchArgument('use_rviz',     default_value='true'),
        DeclareLaunchArgument('use_gzclient', default_value='true'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('scenario_file', default_value=''),
        DeclareLaunchArgument('world_file', default_value=default_world_file),
        DeclareLaunchArgument('run_seed', default_value='0'),
        DeclareLaunchArgument('strategy', default_value='dual'),
        DeclareLaunchArgument('sensor_model',default_value='a',choices=['a','b']),
        DeclareLaunchArgument('belief_mode',default_value='online',choices=['off','shadow','online']),
        DeclareLaunchArgument('software_params',default_value=os.path.join(bringup_dir,'config','multisource.yaml')),
        DeclareLaunchArgument('scenario_jitter_std_m', default_value='0.0'),

        OpaqueFunction(function=prepare_thermal_world),
        # ══════════════════════════════════════════════════════════════════
        # t=0s: Gazebo + Robot State Publisher
        # ══════════════════════════════════════════════════════════════════
        ExecuteProcess(
            cmd=['gzserver', '--verbose', LaunchConfiguration('thermal_world_file'),
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
                parameters=[nav2_params, {
                    'use_sim_time': use_sim_t,
                    'default_nav_to_pose_bt_xml': nav2_bt_xml,
                    'default_nav_through_poses_bt_xml': nav2_bt_xml,
                }],
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
                parameters=[params_file,software_params,shared, {
                    'use_sim_time': use_sim_t,
                    'scenario_file': scenario_file,
                    'world_file':world_file,
                    'image_width':ParameterValue(PythonExpression(["160 if '",sensor_model,"'=='b' else 64"]),value_type=int),
                    'image_height':ParameterValue(PythonExpression(["120 if '",sensor_model,"'=='b' else 48"]),value_type=int),
                    'publish_rate':ParameterValue(PythonExpression(["8.6 if '",sensor_model,"'=='b' else 10.0"]),value_type=float),
                    'scenario_seed': ParameterValue(run_seed, value_type=int),
                    'scenario_jitter_std_m': ParameterValue(scenario_jitter, value_type=float),
                }],
                output='both',
            ),
            Node(
                package='signal_preprocessor',
                executable='preprocessor_node',
                name='preprocessor_node',
                parameters=[params_file,software_params,shared,{
                    'filter_method':ParameterValue(PythonExpression(["'passthrough' if '",sensor_model,"'=='b' else 'kalman'"]),value_type=str),
                    'spatial_smooth':ParameterValue(PythonExpression(["'",sensor_model,"'!='b'"]),value_type=bool)}],
                output='both',
            ),
            Node(
                package='thermal_field_reconstructor',
                executable='reconstructor_node',
                name='reconstructor_node',
                parameters=[params_file,software_params,shared],
                output='both',
            ),
            Node(
                package='thermal_field_reconstructor',
                executable='thermal_mapper_node',
                name='thermal_mapper_node',
                parameters=[params_file,software_params,shared,{'fusion_memory_s':fusion_memory}],
                output='both',
            ),
            Node(
                package='thermal_gradient_processor',
                executable='gradient_node',
                name='gradient_node',
                parameters=[params_file,software_params,shared],
                output='both',
            ),
            Node(
                package='thermal_motion_controller',
                executable='source_tracker_node',
                name='source_tracker_node',
                parameters=[params_file,software_params,shared,{'estimator_model':estimator_model,'strategy':strategy,
                    'gate_m':ParameterValue(PythonExpression(["3.0 if '",strategy,"' in ('fast','dual','gp_ucb') else 1.25"]),value_type=float),
                    'max_detection_age_s':ParameterValue(PythonExpression(["1.5 if '",strategy,"' in ('fast','dual','gp_ucb') else 8.0"]),value_type=float),
                    'merge_radius_m':ParameterValue(PythonExpression(["0.5 if '",strategy,"' in ('fast','dual','gp_ucb') else 1.0"]),value_type=float)}],
                output='both',
            ),
            Node(package='thermal_motion_controller',executable='belief_node',name='belief_node',
                 parameters=[software_params,{'use_sim_time':use_sim_t,
                     'mode':ParameterValue(belief_mode,value_type=str)}],output='both'),
            # controller_node v30: 热导航决策层
            # - FINE 模式（ASCENT/CONVERGE/SAMPLE）: 直接发布 /cmd_vel
            # - COARSE 模式（DEPARTURE/COARSE_SURVEY/FRONTIER）: NavigateToPose Action
            Node(
                package='thermal_motion_controller',
                executable='controller_node',
                name='controller_node',
                parameters=[params_file,software_params,shared, {
                    'use_sim_time': use_sim_t,
                    'random_seed': ParameterValue(run_seed, value_type=int),
                    'strategy': strategy,
                }],
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
                condition=IfCondition(use_rviz),
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
