#!/usr/bin/env python3

import os
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch_ros.descriptions import ParameterFile


def _load_node_params(config_path, node_name):
    try:
        with open(config_path, 'r', encoding='utf-8') as cfg:
            content = yaml.safe_load(cfg) or {}
        return content.get(node_name, {}).get('ros__parameters', {})
    except Exception:
        return {}


def _launch_setup(context, *args, **kwargs):
    pkg_dir = get_package_share_directory('sins')
    env = LaunchConfiguration('env')

    # cfg/<env>/calibration_icp.yaml (see cfg/README-equivalent notes in
    # ../README.md) — indoor read here just to pull out frame names below;
    # the actual node parameters (env-selected) come from the ParameterFile
    # passed to calibration_node further down.
    config_file = os.path.join(pkg_dir, 'cfg', 'indoor', 'calibration_icp.yaml')
    rviz_config_file = os.path.join(pkg_dir, 'rviz', 'calibration_icp_debug.rviz')

    # Read the actual TF frame names calibration_icp_node will use, instead
    # of hardcoding them here, so the static initial-guess TF below always
    # matches what the node looks up (see calibration_icp_node.cpp
    # fetchInitialGuessFromTF: lidar_frame -> camera_frame_initial_guess).
    calibration_params = _load_node_params(config_file, 'calibration_icp_node')
    lidar_frame = str(calibration_params.get('lidar_frame', 'velodyne'))
    yaml_camera_frame_initial_guess = str(calibration_params.get('camera_frame_initial_guess', 'camera_initial_guess'))

    depth_intrinsics_str = LaunchConfiguration('depth_intrinsics').perform(context)
    depth_intrinsics = [float(v) for v in depth_intrinsics_str.split(',')]
    if len(depth_intrinsics) != 4:
        raise ValueError(
            f"depth_intrinsics must be 4 comma-separated numbers [fx,fy,cx,cy], got: {depth_intrinsics_str!r}")

    # robot_tf_camera_frame lets a bag/robot that already publishes a real,
    # complete TF tree (robot_state_publisher from /robot_description) supply
    # the initial guess directly, instead of the hand-measured static TF
    # below. Empty (default) preserves the old behavior. Non-empty: skip the
    # static publisher and point camera_frame_initial_guess straight at that
    # frame (e.g. front_camera_color_optical_frame) — tf2 composes it from
    # the robot's own chain (velodyne -> base_link -> ... -> that frame), no
    # extra publisher needed.
    #
    # This matters more than it sounds: for bags/*_validation_lab, the
    # hand-measured guess below is off by ~42cm from the robot's real,
    # URDF-derived extrinsics — and icp_max_correspondence_distance (0.2m
    # indoor / 0.35m outdoor) is well under that, so ICP can't even consider
    # the correct correspondences and converges to a worse-than-initial
    # result (fitness ~0.5, and the "refined" TF ends up excluding *more*
    # lidar points from the camera FOV than the bad initial guess did).
    # Starting from the real TF instead needs at most a few cm of ICP
    # correction, which the default correspondence distance handles fine.
    robot_tf_camera_frame = LaunchConfiguration('robot_tf_camera_frame').perform(context)
    camera_frame_initial_guess = robot_tf_camera_frame if robot_tf_camera_frame else yaml_camera_frame_initial_guess

    calibration_node = Node(
        package='sins',
        executable='calibration_icp_node',
        name='calibration_icp_node',
        output='screen',
        parameters=[
            ParameterFile(
                PathJoinSubstitution([pkg_dir, 'cfg', env, 'calibration_icp.yaml']),
                allow_substs=True),
            {
                'use_sim_time': True,
                'depth_topic': LaunchConfiguration('depth_topic'),
                'depth_transport': LaunchConfiguration('depth_transport'),
                'depth_intrinsics': depth_intrinsics,
                'camera_frame_initial_guess': camera_frame_initial_guess,
            },
        ],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz_calibration_icp',
        output='screen',
        arguments=['-d', rviz_config_file],
    )

    actions = [calibration_node, rviz_node]

    if not robot_tf_camera_frame:
        # Initial-guess TF velodyne -> camera_initial_guess. Values below are
        # a hand-measured guess from a previous session/rig; adjust them (or
        # pass robot_tf_camera_frame instead, see above) as needed.
        actions.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='calib_icp_tf_velodyne_to_camera_link_initial_guess',
            parameters=[{'use_sim_time': True}],
            arguments=[
                '--x', '0.218304037',
                '--y', '0.107423631',
                '--z', '-0.020496928',
                '--roll', '-1.576042033',
                '--pitch', '-0.037013778',
                '--yaw', '-1.578200049',
                '--frame-id', lidar_frame,
                '--child-frame-id', yaml_camera_frame_initial_guess,
            ],
        ))

    return actions


def generate_launch_description():
    # cfg/<env>/calibration_icp.yaml. indoor mainly differs from outdoor by
    # is_indoor (no calibration timeout fallback, waits for a full ICP
    # calibration) and by tighter ICP thresholds.
    env_arg = DeclareLaunchArgument(
        'env', default_value='indoor',
        description="Which config to use: 'indoor' or 'outdoor'")

    # Overridable per bag: different recordings use different depth topics/
    # transports/intrinsics. Defaults below target bags/20260910_124939_
    # validation_lab, the bag currently in use for this calibration work: it
    # only records .../aligned_depth_to_color/image_raw/zstd (no
    # .../depth/image_rect_raw via compressedDepth at all), and aligned-to-
    # color depth must be back-projected with the COLOR camera's intrinsics
    # (from that bag's .../aligned_depth_to_color/camera_info), not the raw
    # depth sensor's. Pass different values on the command line for another
    # bag/live camera.
    depth_topic_arg = DeclareLaunchArgument(
        'depth_topic', default_value='/front_camera/camera/aligned_depth_to_color/image_raw',
        description='Depth image base topic (image_transport will append the transport-specific suffix)')
    depth_transport_arg = DeclareLaunchArgument(
        'depth_transport', default_value='zstd',
        description="image_transport plugin depth_topic is published with: 'compressedDepth', 'zstd', 'raw', ...")
    depth_intrinsics_arg = DeclareLaunchArgument(
        'depth_intrinsics', default_value='644.1800537109375,643.2573852539062,647.415283203125,361.88623046875',
        description='Comma-separated fx,fy,cx,cy matching depth_topic (must be the COLOR camera intrinsics when '
                     'depth_topic is an aligned_depth_to_color stream)')
    robot_tf_camera_frame_arg = DeclareLaunchArgument(
        'robot_tf_camera_frame', default_value='front_camera_color_optical_frame',
        description='If set (e.g. front_camera_color_optical_frame), use the TF already published by the bag/'
                     "robot's own robot_state_publisher as the initial guess instead of the hand-measured static "
                     'TF (skips that static_transform_publisher entirely). Pass an empty string to fall back to '
                     'the hand-measured static TF instead.')

    return LaunchDescription([
        env_arg,
        depth_topic_arg,
        depth_transport_arg,
        depth_intrinsics_arg,
        robot_tf_camera_frame_arg,
        OpaqueFunction(function=_launch_setup),
    ])
