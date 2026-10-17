import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    pkg = get_package_share_directory('postman_pkg')
    pkg_share_parent = os.path.dirname(pkg)

    tier_str = context.launch_configurations.get('tier', '1')
    seed_str = context.launch_configurations.get('seed', '42')
    seed = int(seed_str) if seed_str.isdigit() else 42

    tier = int(tier_str) if tier_str in ('1', '2', '3') else 1

    urdf = os.path.join(pkg, 'urdf', 'robot.urdf')
    bridge_cfg = os.path.join(pkg, 'config', 'bridge.yaml')
    rviz_cfg = os.path.join(pkg, 'config', 'postman.rviz')

    # Tier 1 & 2: Clean arena (no obstacles)
    # Tier 3: Cluttered arena (obstacles with magenta tops)
    world_name = 'arena_obstacles.sdf' if tier == 3 else 'arena.sdf'
    world = os.path.join(pkg, 'worlds', world_name)

    with open(urdf) as f:
        desc = f.read()

    set_gz_resource = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=pkg_share_parent + ':' + os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    )

    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py'
        )),
        launch_arguments={
            'gz_args': '-r ' + world,
            'on_exit_shutdown': 'true',
        }.items()
    )

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': desc, 'use_sim_time': True}]
    )

    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-name', 'postman_bot', '-topic', 'robot_description',
                   '-x', '0', '-y', '0', '-z', '0.05']
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{'config_file': bridge_cfg, 'use_sim_time': True}]
    )

    # Waypoint node handles tier rules:
    # Tier 1: publishes coordinates, spawns 2 dynamic at 5s, updates /waypoints
    # Tier 2: coordinates withheld, spawns 2 dynamic visually at 7s
    # Tier 3: coordinates withheld, obstacles present, spawns 2 dynamic visually at 7s
    wp = Node(
        package='postman_pkg',
        executable='waypoint_node',
        parameters=[{
            'tier': tier,
            'seed': seed,
            'delay_tier1': 5.0,
            'delay_tier2_3': 7.0,
            'use_sim_time': True
        }]
    )

    # Injects Gaussian velocity noise (sigma=0.002 rad/s) on /joint_states during motion
    enc_noise = Node(
        package='postman_pkg',
        executable='encoder_noise_node',
        parameters=[{'use_sim_time': True, 'velocity_noise_stddev': 0.002}]
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_cfg],
        parameters=[{'use_sim_time': True}]
    )

    return [set_gz_resource, gz, rsp, spawn, bridge, enc_noise, wp, rviz]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'tier',
            default_value='1',
            choices=['1', '2', '3'],
            description='Competition tier: 1 (Dead reckoning, 5s dynamic), 2 (Vision, 7s dynamic), 3 (Vision + obstacles, 7s dynamic)'
        ),
        DeclareLaunchArgument(
            'seed',
            default_value='42',
            description='RNG seed for reproducible procedural waypoint placement (e.g. seed:=42)'
        ),
        OpaqueFunction(function=launch_setup),
    ])
