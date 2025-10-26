from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('robotic_arm')

    urdf_file = os.path.join(pkg_share, 'urdf', 'robotic_arm.urdf')
    controllers_file = os.path.join(pkg_share, 'config', 'ros2_controllers.yaml')

    return LaunchDescription([

        # Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            arguments=[urdf_file],
        ),

        # Gazebo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('gazebo_ros'),
                             'launch', 'gazebo.launch.py')
            ),
            
        ),

        # Spawn entity in Gazebo
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-entity', 'robotic_arm', '-file', urdf_file],
            output='screen',
        ),

        # Delay spawners so Gazebo plugin finishes initializing
        TimerAction(
            period=5.0,  # delay 5s
            actions=[
                Node(
                    package='controller_manager',
                    executable='spawner',
                    arguments=['joint_state_broadcaster',
                               '--controller-manager', '/controller_manager'],
                    output='screen',
                )
            ]
        ),

        TimerAction(
            period=7.0,  # delay a bit longer
            actions=[
                Node(
                    package='controller_manager',
                    executable='spawner',
                    arguments=['joint_trajectory_controller',
                               '--controller-manager', '/controller_manager'],
                    output='screen',
                )
            ]
        ),
    ])
