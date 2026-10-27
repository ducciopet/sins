#!/usr/bin/env python3

"""Standalone launch file for JUST yolo_seg_track_node.py — same role as
run_calibration_icp.launch.py (its own file, own rviz, run in isolation),
not a phase entry in run_detector.launch.py's single growing pipeline. Use
this to look at seg+track detections on their own (against a live camera or
a looping bag) without bringing up preprocessing_node/static_structures_node/
calibration_icp_node at all. run_detector.launch.py still launches this same
node as part of the full pipeline (see its own detection_nodes) — that one
also shows the annotated feed, via the same two Image displays folded into
rviz/preprocessing_debug.rviz (disabled by default there; enabled by default
in this file's own rviz/yolo_debug.rviz, since watching detections is this
config's whole purpose).

One node instance PER camera in camera_names (same reasoning as
run_detector.launch.py's own detection_nodes loop: a single instance
handling every camera shares one process/GIL/timer budget across them,
which measurably dropped frames — see yolo_seg_track_detector.py's file
header). Empty camera_names (only reachable by explicitly passing an empty
string) is the one case that launches a single un-parallelized instance,
falling through to detection.yaml's own default camera list internally."""

import os
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile
from ament_index_python.packages import get_package_share_directory


def _load_node_params(config_path, node_name):
    try:
        with open(config_path, 'r', encoding='utf-8') as cfg:
            content = yaml.safe_load(cfg) or {}
        return content.get(node_name, {}).get('ros__parameters', {})
    except Exception:
        return {}


def _launch_setup(context, *args, **kwargs):
    pkg_dir = get_package_share_directory('sins')
    env = LaunchConfiguration('env').perform(context)
    detection_config_path = os.path.join(pkg_dir, 'cfg', env, 'detection.yaml')

    # camera_names as a comma-separated launch arg (not read from
    # preprocessing.yaml like run_detector.launch.py does): this file runs
    # yolo_seg_track_node.py on its own, with no preprocessing_node in the
    # picture to be the single source of truth for — so cfg/<env>/
    # detection.yaml's own default (['front_camera', 'back_camera']) is the
    # only source, overridable here for a quick single-camera run.
    camera_names_str = LaunchConfiguration('camera_names').perform(context)
    camera_names = [n.strip() for n in camera_names_str.split(',') if n.strip()]

    use_sim_time = LaunchConfiguration('use_sim_time')

    # One process per camera — see file header. camera_names empty (only
    # via an explicit empty string) is the sole fallback to a single,
    # un-parallelized instance using detection.yaml's own default list.
    #
    # A plain dict via _load_node_params, NOT ParameterFile(detection_config,
    # ...), in the per-camera branch: same pitfall/fix as
    # run_detector.launch.py's own detection_nodes loop — ParameterFile only
    # applies a file's parameters to a node named EXACTLY its top-level key
    # ('yolo_seg_track_detector:'), and each per-camera instance below is
    # named 'yolo_seg_track_detector_<name>' so they don't collide in the
    # ROS graph. The single-instance fallback keeps ParameterFile since its
    # node name matches the yaml key exactly (no allow_substs lost there).
    if camera_names:
        detection_params = _load_node_params(detection_config_path, 'yolo_seg_track_detector')
        yolo_nodes = [
            Node(
                package='sins',
                executable='yolo_seg_track_node.py',
                name=f'yolo_seg_track_detector_{name}',
                output='screen',
                parameters=[detection_params, {'use_sim_time': use_sim_time, 'camera_names': [name]}],
            )
            for name in camera_names
        ]
    else:
        yolo_nodes = [Node(
            package='sins',
            executable='yolo_seg_track_node.py',
            name='yolo_seg_track_detector',
            output='screen',
            parameters=[ParameterFile(detection_config_path, allow_substs=True), {'use_sim_time': use_sim_time}],
        )]

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz_yolo',
        output='screen',
        arguments=['-d', os.path.join(pkg_dir, 'rviz', 'yolo_debug.rviz')],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return yolo_nodes + [rviz_node]


def generate_launch_description():
    env_arg = DeclareLaunchArgument(
        'env', default_value='indoor',
        description="Which cfg/<env>/detection.yaml profile to use: 'outdoor' or 'indoor'")
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use /clock — set true when replaying bags with --clock')
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Whether to launch RViz (rviz/yolo_debug.rviz — annotated front/back camera feeds)')
    camera_names_arg = DeclareLaunchArgument(
        'camera_names', default_value='front_camera,back_camera',
        description="Comma-separated camera list to run seg+track on, e.g. 'front_camera' for just one. "
                     "Empty string falls through to detection.yaml's own default.")

    return LaunchDescription([
        env_arg,
        use_sim_time_arg,
        rviz_arg,
        camera_names_arg,
        OpaqueFunction(function=_launch_setup),
    ])
