from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    tier = context.launch_configurations.get('tier', '1')
    executable_name = f'tier{tier}_node' if tier in ('1', '2', '3') else 'solution_node'
    return [
        Node(
            package='postman_solution',
            executable=executable_name,
            name='solution_node',
            output='screen',
            parameters=[{'use_sim_time': True}],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'tier',
            default_value='1',
            choices=['1', '2', '3'],
            description='Select solution node to execute: 1 (tier1_node), 2 (tier2_node), 3 (tier3_node)'
        ),
        OpaqueFunction(function=launch_setup),
    ])
