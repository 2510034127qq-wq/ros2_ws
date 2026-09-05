"""Thermal autonomy overlay for a running UGV ROS 2/Nav2 stack or bag replay.

The UGV driver owns /odom, /scan, map->odom->base_link and NavigateToPose.
Calibrated camera optical TF and CameraInfo must be provided by the camera stack.
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share=get_package_share_directory('thermal_bringup')
    base=os.path.join(share,'config','params.yaml')
    software=os.path.join(share,'config','multisource.yaml')
    hardware=LaunchConfiguration('hardware_params',default=os.path.join(share,'config','ugv_thermal.yaml'))
    clock=LaunchConfiguration('use_sim_time',default='false')
    mode=LaunchConfiguration('belief_mode',default='online')
    strategy=LaunchConfiguration('strategy',default='dual')
    common=[base,software,hardware,{'use_sim_time':clock}]
    nodes=[
        Node(package='thermal_field_reconstructor',executable='radiometric_input_node',
             name='radiometric_input_node',parameters=common,output='screen',
             condition=IfCondition(LaunchConfiguration('start_input',default='true'))),
        Node(package='thermal_field_reconstructor',executable='depth_registration_node',
             name='depth_registration_node',parameters=common,output='screen',
             condition=IfCondition(LaunchConfiguration('register_depth',default='true'))),
        Node(package='signal_preprocessor',executable='preprocessor_node',name='preprocessor_node',
             parameters=common,remappings=[('/sim/thermal_raw','/thermal/raw')],output='screen'),
        Node(package='thermal_field_reconstructor',executable='reconstructor_node',name='reconstructor_node',
             parameters=common,output='screen'),
        Node(package='thermal_field_reconstructor',executable='thermal_mapper_node',name='thermal_mapper_node',
             parameters=common,output='screen'),
        Node(package='thermal_gradient_processor',executable='gradient_node',name='gradient_node',
             parameters=common,output='screen'),
        Node(package='thermal_motion_controller',executable='source_tracker_node',name='source_tracker_node',
             parameters=common+[{'strategy':strategy}],output='screen'),
        Node(package='thermal_motion_controller',executable='belief_node',name='belief_node',
             parameters=common+[{'mode':ParameterValue(mode,value_type=str)}],output='screen'),
        Node(package='thermal_motion_controller',executable='controller_node',name='controller_node',
             parameters=common+[{'strategy':strategy}],output='screen',
             condition=IfCondition(LaunchConfiguration('enable_motion',default='false'))),
    ]
    return LaunchDescription([
        DeclareLaunchArgument('hardware_params',default_value=os.path.join(share,'config','ugv_thermal.yaml')),
        DeclareLaunchArgument('use_sim_time',default_value='false'),
        DeclareLaunchArgument('strategy',default_value='dual'),
        DeclareLaunchArgument('belief_mode',default_value='online'),
        DeclareLaunchArgument('start_input',default_value='true'),
        DeclareLaunchArgument('register_depth',default_value='true'),
        DeclareLaunchArgument('enable_motion',default_value='false'),
        *nodes])
