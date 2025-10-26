from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
import os

def generate_launch_description():

    # Path to your URDF file
    urdf_file = os.path.join(
        os.path.dirname(__file__), '..', 'urdf', 'robotic_arm.urdf'
    )

    return LaunchDescription([
        # Start Gazebo with ROS plugin
        ExecuteProcess(
            cmd=['gazebo', '--verbose', '--pause', '-s', 'libgazebo_ros_factory.so'],
            output='screen'),

        # Spawn the robot into Gazebo
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-entity', 'robotic_arm', '-file', urdf_file],
            output='screen'
        ),
    ])
