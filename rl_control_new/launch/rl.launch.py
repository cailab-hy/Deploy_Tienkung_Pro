#!/usr/bin/env python3
"""
ROS2 Launch file for rl_control_new package
Converted from ROS1 nodelet-based launch to ROS2 Node-based launch
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Generate launch description for rl_control_new"""
    
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true'
    )
    
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='',
        description='Path to configuration file'
    )
    
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Log level for nodes'
    )

    # Option 1: Run as regular ROS2 node (commented out - use composable node instead)
    """
    rl_control_node = Node(
        package='rl_control_new',
        executable='rl_control_new_plugin',
        name='rl_control_new',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'config_file': LaunchConfiguration('config_file'),
        }],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
    )
    """
    
    # Joy node for gamepad input
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        parameters=[{
            'device': '/dev/input/js0',
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
        output='screen',
    )

    # Option 2: Run as composable node (similar to nodelet)
    rl_control_container = ComposableNodeContainer(
        name='rl_control_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='rl_control_new',
                plugin='rl_control_new::RLControlNewPlugin',
                name='rl_control_new',
                parameters=[{
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'config_file': LaunchConfiguration('config_file'),
                }],
            )
        ],
        output='screen',
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
    )

    # Log information
    log_info = LogInfo(
        msg=[
            'Starting TienKung RL Control System (ROS2)\n',
            'Configuration: ', LaunchConfiguration('config_file'), '\n',
            'Use sim time: ', LaunchConfiguration('use_sim_time'), '\n',
            'Log level: ', LaunchConfiguration('log_level')
        ]
    )

    return LaunchDescription([
        # Launch arguments
        use_sim_time_arg,
        config_file_arg,
        log_level_arg,
        
        # Log information
        log_info,
        
        # Joy node for gamepad input
        joy_node,
        
        # Nodes - using composable node container
        rl_control_container,
        
        # Uncomment this line and comment rl_control_container if using regular nodes
        # rl_control_node,
    ])
