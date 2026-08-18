#!/usr/bin/env python3
"""
ROS2 Launch file for hdmi_inference package (Jetson Orin)
Runs the HDMI policy inference node at 50Hz.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Log level for nodes'
    )

    hdmi_node = Node(
        package='hdmi_inference',
        executable='hdmi_inference_node',
        name='hdmi_inference',
        output='screen',
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
    )

    log_info = LogInfo(
        msg=['Starting HDMI Inference Node on Jetson Orin (50Hz policy inference)']
    )

    return LaunchDescription([
        log_level_arg,
        log_info,
        hdmi_node,
    ])
