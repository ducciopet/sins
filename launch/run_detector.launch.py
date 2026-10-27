#!/usr/bin/env python3

"""
Single, growing launch file for the whole sins pipeline —
mirrors jo_detector/launch/run_detector.launch.py's role (the "launch
everything" file, as opposed to run_calibration_icp.launch.py and
run_yolo.launch.py, which each stay their own standalone file for
tuning/testing in isolation, with their own rviz).

Do NOT add a new run_<phase>.launch.py per phase — extend this one instead,
the same way jo_detector's run_detector.launch.py grew a node at a
time (detector_node, then static_structures_node, then yolo_node, ...). Today
it launches one calibration_icp_node PER CAMERA (see calibration_nodes
below) + preprocessing_node + static_structures_node (standalone, asynchronous
wall/floor detector — see src/static_structures_node.cpp's own header) + one
yolo_seg_track_detector PER CAMERA too (see detection_nodes below and
scripts/yolo_seg_track_detector.py's own header — one OS process per
camera, for real parallelism, same reasoning as calibration_nodes) — for
whichever detection model(s) enable_yolo11/enable_yolo26 turn on (see
those args and cfg/<env>/detection_yolo26.yaml: same node/script, just a
different cfg file/node-name/topic prefix per model, so trying a new
checkpoint is a new cfg/<env>/detection_<model>.yaml + one more entry in
detection_models below, not a new node script) + one
yolo_semantic_seg_detector PER CAMERA (see semantic_seg_nodes below and
scripts/yolo_semantic_seg_detector.py's own header — a separate model/task,
semantic not instance segmentation, gated by its own enable_semantic_segmentation
arg); when the
tracking+classification phase exists, add its Node(...) entry and
config lookup here too, following the calibration/preprocessing entries
below as the template (ParameterFile from
cfg/<env>/<node>.yaml, use_sim_time, output='screen').
"""

import os
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
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


def _parse_camera_intrinsics_map(spec):
    """'name:fx,fy,cx,cy;name2:fx,fy,cx,cy' -> {name: [fx,fy,cx,cy]}."""
    result = {}
    for entry in spec.split(';'):
        entry = entry.strip()
        if not entry:
            continue
        name, _, values = entry.partition(':')
        result[name.strip()] = [float(v) for v in values.split(',')]
    return result


def _launch_setup(context, *args, **kwargs):
    pkg_dir = get_package_share_directory('sins')
    env = LaunchConfiguration('env')
    env_str = env.perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time')

    calibration_config = os.path.join(pkg_dir, 'cfg', env_str, 'calibration_icp.yaml')
    preprocessing_config = os.path.join(pkg_dir, 'cfg', env_str, 'preprocessing.yaml')
    static_structures_config = os.path.join(pkg_dir, 'cfg', env_str, 'static_structures.yaml')

    # Read frame names straight from the calibration config instead of
    # hardcoding them, so the static initial-guess TF below always matches
    # what calibration_icp_node actually looks up (same reasoning as
    # run_calibration_icp.launch.py).
    calibration_params = _load_node_params(calibration_config, 'calibration_icp_node')
    lidar_frame = str(calibration_params.get('lidar_frame', 'velodyne'))

    # camera_names drives BOTH preprocessing_node's per-camera CameraStreams
    # and, here, one calibration_icp_node instance per camera — same list,
    # read from the same config, so the two can never disagree about which
    # cameras exist. Falls back to a single front_camera if the key is
    # missing, matching preprocessing_node.cpp's own declare_parameter
    # default.
    preprocessing_params = _load_node_params(preprocessing_config, 'preprocessing_node')
    camera_names = list(preprocessing_params.get('camera_names', ['front_camera']))
    # Optional launch-time override (e.g. camera_names:=front_camera for a bag
    # recorded with a single camera), forwarded below to every node that takes
    # the list, so no node falls back to its own YAML default.
    camera_names_override = LaunchConfiguration('camera_names').perform(context).strip()
    if camera_names_override:
        camera_names = [n.strip() for n in camera_names_override.split(',') if n.strip()]

    camera_frame_initial_guess_template = LaunchConfiguration('camera_frame_initial_guess_template').perform(context)
    camera_depth_intrinsics = _parse_camera_intrinsics_map(
        LaunchConfiguration('camera_depth_intrinsics').perform(context))
    depth_transport = LaunchConfiguration('depth_transport')

    # Loaded here (ahead of preprocessing_node's own Node(...) below) so its
    # segmentation_mask_topic_template can be forwarded into preprocessing_node's
    # per-camera <name>.ground_mask_topic (see that param's own comment in
    # preprocessing_node.cpp) — single source of truth for the topic name,
    # same reasoning as calibration_params/detection_params elsewhere in this
    # file. Reused again below for semantic_seg_nodes itself.
    enable_semantic_segmentation = LaunchConfiguration('enable_semantic_segmentation').perform(context) == 'true'
    # semantic_ground_points (preprocessing_node applying the segmentation mask to the
    # depth — per pixel on the aligned image, or by projecting each native-cloud point
    # onto the mask) — see that arg's own description.
    enable_semantic_ground_points = LaunchConfiguration('enable_semantic_ground_points').perform(context) == 'true'
    semantic_seg_params = {}
    if enable_semantic_segmentation:
        semantic_seg_params = _load_node_params(
            os.path.join(pkg_dir, 'cfg', env_str, 'semantic_segmentation.yaml'), 'yolo_semantic_seg_detector')

    # Loaded here (ahead of preprocessing_node's own Node(...) below), same reasoning as semantic_seg_params
    # above — single source of truth for detection.yaml's own topic names, needed to wire preprocessing_node's
    # "mask" yolo_leaf_method (see preprocessing_node.cpp's own yolo_leaf_method_ comment and this file's own
    # yolo_leaf_method_arg description).
    yolo_leaf_method = LaunchConfiguration('yolo_leaf_method').perform(context)
    detection_yaml_params = {}
    if yolo_leaf_method == 'mask':
        detection_yaml_params = _load_node_params(
            os.path.join(pkg_dir, 'cfg', env_str, 'detection.yaml'), 'yolo_seg_track_detector')

    # robot_tf_camera_frame: escape hatch, same idea as
    # run_calibration_icp.launch.py's own arg of the same name — use the
    # robot/bag's own TF chain (camera_frame_initial_guess_template, one
    # frame per camera) as calibration_icp_node's initial guess instead of
    # a hand-measured static TF. Only a SINGLE hand-measured fallback exists
    # (for the first camera, front_camera, against velodyne — see the
    # static_transform_publisher below) since that's the only one this
    # package has ever had measured; a second/third camera has no such
    # fallback and needs the robot TF chain (the default) to work.
    robot_tf_camera_frame = LaunchConfiguration('robot_tf_camera_frame').perform(context)
    use_robot_tf_guess = bool(robot_tf_camera_frame)

    # enable_calibration — see that arg's own DeclareLaunchArgument
    # description below. false (the current default): skip launching
    # calibration_icp_node entirely (calibration_nodes stays empty, and the
    # initial-guess static TF below — pointless with nothing to consume it —
    # is skipped too) and tell preprocessing_node/static_structures_node to
    # trust each camera's raw URDF/bag TF chain permanently instead of ever
    # looking for a <name>_refined calibration TF, via calibration_enabled
    # in each node's own parameters dict below.
    enable_calibration = LaunchConfiguration('enable_calibration').perform(context) == 'true'

    # One calibration_icp_node instance per camera. Each gets: its own
    # node name, its own depth_topic/depth_intrinsics (the aligned-depth
    # stream and the COLOR camera's own intrinsics — see
    # run_calibration_icp.launch.py's env_arg comment for the full
    # reasoning, identical here), its own camera_frame_initial_guess
    # (<name>_color_optical_frame, from the robot's own TF chain) and its
    # own refined_camera_frame (<name>_refined — this is what
    # preprocessing_node.cpp's CameraStream::refined_camera_frame looks up
    # per camera, see ensureCameraCalibrationTf()), remapped debug
    # topics so the two instances' /calibration/velodyne_points and
    # /calibration/depth_cloud_in_velodyne (otherwise hardcoded, identical,
    # names in calibration_icp_node.cpp) don't collide, and its own
    # depth_cloud_topic/depth_to_color_extrinsics_topic — forwarded from
    # preprocessing.yaml's own per-camera aligned_depth_cloud_topic choice
    # (see depth_cloud_topic's own comment below) so this node uses the
    # SAME native-cloud-or-deprojection choice preprocessing_node makes for
    # that camera, without a second setting to keep in sync by hand.
    calibration_nodes = []
    for name in (camera_names if enable_calibration else []):
        # A camera_names entry with no matching camera_depth_intrinsics map
        # key is expected, not an error: adding a camera is meant to be a
        # one-line camera_names edit in preprocessing.yaml (see that file
        # and setupCamera() in both preprocessing_node.cpp and
        # static_structures_node.cpp, which already build every per-camera
        # default from the name itself), not something that also requires
        # touching this launch file. depth_intrinsics is only
        # calibration_icp_node's bootstrap value anyway (see its own
        # declare_parameter comment) — real, live intrinsics come from
        # camera_info_topic_ moments later regardless — so an unlisted
        # camera just omits the launch-time override here and falls back
        # to calibration_icp_node.cpp's own generic default, with a
        # one-time warning so it's still visible that the map is
        # incomplete for anyone who wants a tighter bootstrap guess.
        camera_intrinsics_params = {}
        if name in camera_depth_intrinsics:
            fx, fy, cx, cy = camera_depth_intrinsics[name]
            camera_intrinsics_params['depth_intrinsics'] = [fx, fy, cx, cy]
        else:
            print(
                f"[run_detector.launch.py] No entry for camera '{name}' in camera_depth_intrinsics "
                f"(got: {sorted(camera_depth_intrinsics.keys())}) — using calibration_icp_node's own "
                "default bootstrap intrinsics for it. Add 'name:fx,fy,cx,cy' to that launch arg for a "
                "tighter initial guess; not required for the node to work.")
        camera_frame_initial_guess = (
            camera_frame_initial_guess_template.format(camera=name) if use_robot_tf_guess
            else f'{name}_initial_guess'
        )
        # Native-cloud choice forwarded straight from preprocessing.yaml's
        # own per-camera <name>.aligned_depth_cloud_topic (already loaded
        # above as preprocessing_params, for camera_names) — NOT a separate
        # setting to keep in sync by hand. Whatever preprocessing_node uses
        # for a given camera, this calibration_icp_node instance uses too:
        # single source of truth for "does this camera have a native depth
        # cloud", same reasoning as camera_names itself just above. Empty
        # (missing key, or preprocessing.yaml leaves it unset/commented, as
        # cfg/outdoor/preprocessing.yaml currently does) falls through to
        # calibration_icp_node.cpp's own default: deprojection, unchanged.
        depth_cloud_topic = str((preprocessing_params.get(name, {}) or {}).get('aligned_depth_cloud_topic', ''))
        calibration_nodes.append(Node(
            package='sins',
            executable='calibration_icp_node',
            name=f'calibration_icp_node_{name}',
            output='screen',
            parameters=[
                # A plain dict, NOT ParameterFile(calibration_config, ...):
                # ParameterFile-with-a-file whose YAML has an explicit
                # top-level node-name key ('calibration_icp_node:') only
                # applies its parameters to a node named EXACTLY that —
                # since each instance here is named
                # 'calibration_icp_node_<name>' (needs to differ per camera
                # so the two don't collide in the ROS graph), that key
                # would never match and every yaml value (icp_fitness_threshold,
                # is_indoor, icp_voxel_size, ...) would silently fall back
                # to calibration_icp_node.cpp's own declare_parameter
                # defaults instead — found by fitness thresholds showing up
                # as the code's 0.5 default instead of this cfg's 0.3.
                # calibration_params (already parsed above, for lidar_frame)
                # is a plain {key: value} dict, which — unlike ParameterFile
                # — applies to whatever node it's handed to regardless of
                # name, so reusing it here sidesteps the mismatch entirely.
                calibration_params,
                {
                    'use_sim_time': use_sim_time,
                    'depth_topic': f'/{name}/camera/aligned_depth_to_color/image_raw',
                    'depth_transport': depth_transport,
                    **camera_intrinsics_params,
                    # depth_intrinsics (when present above) is only
                    # calibration_icp_node's bootstrap value —
                    # camera_info_topic makes it read the SAME color
                    # camera's live intrinsics instead (K is in pixel units
                    # against that camera's resolution, matching depth_topic
                    # being an aligned-to-color stream), the same fix as
                    # camera_depth_intrinsics is a manual, easy-to-drift
                    # substitute for. See calibration_icp_node.cpp's own
                    # onCameraInfo() comment.
                    'camera_info_topic': f'/{name}/camera/color/camera_info',
                    'urdf_camera_frame': f'{name}_color_optical_frame',
                    'camera_frame_initial_guess': camera_frame_initial_guess,
                    'refined_camera_frame': f'{name}_refined',
                    'depth_cloud_topic': depth_cloud_topic,
                    'depth_to_color_extrinsics_topic': f'/{name}/camera/extrinsics/depth_to_color',
                },
            ],
            remappings=[
                ('/calibration/velodyne_points', f'/calibration/{name}/velodyne_points'),
                ('/calibration/depth_cloud_in_velodyne', f'/calibration/{name}/depth_cloud_in_velodyne'),
            ],
        ))

    # raw_depth_transport/aligned_depth_transport override the cfg's
    # 'compressedDepth' default (matches the live camera / jo_sim) — pass
    # 'zstd' for bags/*_validation_lab, which only records aligned depth as
    # zstd (see preprocessing.yaml's own comment on these two keys). Named
    # distinctly from each calibration_node's own depth_topic/depth_transport
    # above (same names as run_calibration_icp.launch.py, on purpose): the
    # two nodes' "depth" defaults point at genuinely different streams here
    # — calibration_icp_node's default is unaligned depth same as this raw
    # one, but for this bag it needs to move to the ALIGNED stream, while
    # this raw_depth_transport controls preprocessing_node's still-unaligned,
    # acquired-only path — sharing one arg name would silently repoint both.
    # <name>.ground_mask_topic — forwarded from semantic_seg_params'
    # segmentation_mask_topic_template (loaded above) so preprocessing_node
    # can extract semantic_ground_points for each camera. Empty (feature off
    # for that camera, see preprocessing_node.cpp's own default) whenever
    # enable_semantic_segmentation is false or the template key is missing —
    # same "single source of truth, one place can turn this off" pattern as
    # static_structures_camera_params below.
    preprocessing_camera_params = {}
    # <name>.yolo_detections_topic: the YOLO 2D detections (with track ids) preprocessing_node turns into 3D semantic leaves,
    # taken from semantic_segmentation.yaml's detections_topic_template when that node runs with detection enabled.
    if enable_semantic_segmentation and bool(semantic_seg_params.get('enable_detection', False)):
        det_template = str(semantic_seg_params.get('detections_topic_template', ''))
        if det_template:
            for name in camera_names:
                preprocessing_camera_params[f'{name}.yolo_detections_topic'] = '/' + det_template.format(camera=name).lstrip('/')
    if (enable_semantic_segmentation and enable_semantic_ground_points
            and bool(semantic_seg_params.get('enable_semantic', True))):
        mask_template = str(semantic_seg_params.get('segmentation_mask_topic_template', ''))
        if mask_template:
            for name in camera_names:
                preprocessing_camera_params[f'{name}.ground_mask_topic'] = mask_template.format(camera=name)
    # <name>.yolo_mask_topic / REPOINTED <name>.yolo_detections_topic: yolo_leaf_method=="mask" needs
    # yolo_seg_track_node's own per-pixel instance-mask + detections topics instead of the box detector's (see
    # preprocessing_node.cpp's own yolo_leaf_method_ comment) — applied AFTER the box-method wiring above so it
    # correctly overrides yolo_detections_topic when active; "box" (default) leaves that wiring untouched.
    if yolo_leaf_method == 'mask' and detection_yaml_params:
        mask_template2 = str(detection_yaml_params.get('detected_masks_topic_template', ''))
        det_template2 = str(detection_yaml_params.get('detected_detections_topic_template', ''))
        for name in camera_names:
            if mask_template2:
                preprocessing_camera_params[f'{name}.yolo_mask_topic'] = '/' + mask_template2.format(camera=name).lstrip('/')
            if det_template2:
                preprocessing_camera_params[f'{name}.yolo_detections_topic'] = '/' + det_template2.format(camera=name).lstrip('/')

    preprocessing_node = Node(
        package='sins',
        executable='preprocessing_node',
        name='preprocessing_node',
        output='screen',
        parameters=[
            ParameterFile(preprocessing_config, allow_substs=True),
            {
                'use_sim_time': use_sim_time,
                'depth_transport': LaunchConfiguration('raw_depth_transport'),
                'aligned_depth_transport': LaunchConfiguration('aligned_depth_transport'),
                'calibration_enabled': enable_calibration,
                'yolo_leaf_method': yolo_leaf_method,
                'camera_names': camera_names,
                **preprocessing_camera_params,
            },
        ],
    )

    # Standalone, asynchronous wall/floor detector — see
    # src/static_structures_node.cpp's own header for the full architecture
    # (mirrors jo_detector's own wall_detector_node: own executable,
    # own node, no coupling to preprocessing_node's callbacks/executor).
    # camera_names is overridden from preprocessing_config's own list
    # (already loaded above, for the calibration loop) rather than left to
    # static_structures.yaml's own default, so the two can never disagree about
    # which cameras exist — same reasoning as the calibration_nodes loop.
    # Per-camera is_native_depth_frame/depth_to_color_extrinsics_topic —
    # same forwarding as the calibration_nodes loop's own depth_cloud_topic
    # above (single source of truth: preprocessing.yaml's own <name>.
    # aligned_depth_cloud_topic), so static_structures_node's floor-plane fit for
    # a camera applies the same depth<->color correction preprocessing_node
    # itself applies for that camera, instead of assuming every camera's
    # voxelized_depth_topic is always the deprojected (color-optical-frame)
    # convention. See CameraSource::is_native_depth_frame's own comment.
    static_structures_camera_params = {}
    for name in camera_names:
        cam_cfg = preprocessing_params.get(name, {}) or {}
        static_structures_camera_params[f'{name}.is_native_depth_frame'] = bool(cam_cfg.get('aligned_depth_cloud_topic', ''))
        static_structures_camera_params[f'{name}.depth_to_color_extrinsics_topic'] = f'/{name}/camera/extrinsics/depth_to_color'

    static_structures_node = Node(
        package='sins',
        executable='static_structures_node',
        name='static_structures_node',
        output='screen',
        parameters=[
            ParameterFile(static_structures_config, allow_substs=True),
            {
                'use_sim_time': use_sim_time,
                'camera_names': camera_names,
                'calibration_enabled': enable_calibration,
                **static_structures_camera_params,
            },
        ],
    )

    # DBSCAN + 3D OBB detection on the LiDAR and every camera (see
    # src/dbscan_detector_node.cpp's header). Ticks on the processed LiDAR
    # cloud, so it only needs preprocessing_node running.
    dbscan_detector_nodes = []
    if LaunchConfiguration('enable_dbscan_detector').perform(context) == 'true':
        dbscan_detector_nodes.append(Node(
            package='sins',
            executable='dbscan_detector_node',
            name='dbscan_detector_node',
            output='screen',
            parameters=[
                ParameterFile(os.path.join(pkg_dir, 'cfg', env_str, 'dbscan_detector.yaml'), allow_substs=True),
                {
                    'use_sim_time': use_sim_time,
                    'camera_names': camera_names,
                    # fused clusters use the LiDAR's own range-banded voxel values (single source of truth)
                    **{f'fusion_{k[6:]}' if k.startswith('lidar_voxel_') else k: preprocessing_params[k]
                       for k in ('lidar_voxel_near', 'lidar_voxel_far', 'lidar_voxel_split_range') if k in preprocessing_params},
                },
            ],
        ))

    # Kalman-filter tracker on dbscan_detector_node's own fused objects (see src/tracker_node.cpp's header) —
    # persistent ids, velocity, dynamic/static classification. Needs dbscan_detector_node's fused/detections3d,
    # so it's a no-op (nothing to subscribe to) when that node is off.
    tracker_nodes = []
    if LaunchConfiguration('enable_tracker').perform(context) == 'true':
        tracker_nodes.append(Node(
            package='sins',
            executable='tracker_node',
            name='tracker_node',
            output='screen',
            parameters=[
                ParameterFile(os.path.join(pkg_dir, 'cfg', env_str, 'tracker.yaml'), allow_substs=True),
                {'use_sim_time': use_sim_time, 'camera_names': camera_names,
                 'track_debug_lifecycle': LaunchConfiguration('tracker_debug_lifecycle').perform(context) == 'true'},
            ],
        ))

    # Phase 2 prototype: instance segmentation + tracking (see
    # scripts/yolo_seg_track_detector.py's file header). ONE NODE INSTANCE
    # PER CAMERA — same pattern as calibration_nodes above, and for the same
    # reason: real parallelism. A single instance handling every camera
    # (the original design) shares one Python process/GIL and one
    # timer_period_sec budget across all of them, processed strictly in
    # turn — with 2 cameras at ~15.5ms of actual work each, that ate ~31ms
    # of the 33ms tick and measurably dropped ~20% of incoming frames
    # (measured: 30Hz camera in, ~24Hz detections out). Splitting into one
    # OS process per camera removes the GIL/shared-timer contention
    # entirely: each camera gets its own full 33ms budget and its own
    # CUDA context, so both can approach the input rate independently.
    # yolo_seg_track_detector.py itself needed NO changes for this — its
    # camera_names loop already works fine with a single-entry list, exactly
    # like every per-camera instance below gets via the 'camera_names':
    # [name] override.
    #
    # detection_params: a plain dict, NOT ParameterFile(detection_config,
    # ...) — same pitfall/fix as calibration_nodes' own calibration_params
    # above. detection.yaml's top-level key is 'yolo_seg_track_detector:'
    # (ParameterFile only applies a file's parameters to a node named
    # EXACTLY that), but each instance below is named
    # 'yolo_seg_track_detector_<name>' so they don't collide in the ROS
    # graph — with ParameterFile every value (weights_path, tracker,
    # score_threshold, ...) would silently fall back to
    # yolo_seg_track_detector.py's own declare_parameter defaults instead
    # (found the same way calibration_nodes' own comment describes: as a
    # RuntimeError from missing required params, not a silent wrong value,
    # since this node treats them as required — see its own __init__).
    # Two independent model toggles (enable_yolo11/enable_yolo26 below) —
    # same node/script for both, just a different cfg file, node-name
    # prefix and topic prefix each (see cfg/<env>/detection_yolo26.yaml's
    # own comment on why the topics don't collide even if both are ever on
    # at once). Building the (config_path, node_prefix) list this way
    # instead of writing the loop twice keeps the per-camera Node(...)
    # construction itself in one place.
    # yolo_leaf_method=="mask" is useless without yolo_seg_track_node actually running (preprocessing_node
    # would just silently get no mask leaves, no error) — coupled here rather than leaving it a footgun the
    # user has to remember to also pass enable_yolo11:=true for.
    enable_yolo11 = LaunchConfiguration('enable_yolo11').perform(context) == 'true' or yolo_leaf_method == 'mask'
    enable_yolo26 = LaunchConfiguration('enable_yolo26').perform(context) == 'true'
    detection_models = []
    if enable_yolo11:
        detection_models.append(('detection.yaml', 'yolo_seg_track_detector'))
    if enable_yolo26:
        detection_models.append(('detection_yolo26.yaml', 'yolo26_seg_track_detector'))

    detection_nodes = []
    for config_filename, node_prefix in detection_models:
        # detection_params: a plain dict, NOT ParameterFile(detection_config,
        # ...) — same pitfall/fix as calibration_nodes' own calibration_params
        # above. Every one of these cfg files' top-level key is
        # 'yolo_seg_track_detector:' regardless of node_prefix (ParameterFile
        # only applies a file's parameters to a node named EXACTLY that), but
        # each instance below is named '<node_prefix>_<name>' so they don't
        # collide in the ROS graph — with ParameterFile every value
        # (weights_path, tracker, score_threshold, ...) would silently fall
        # back to yolo_seg_track_detector.py's own declare_parameter defaults
        # instead (found the same way calibration_nodes' own comment
        # describes: as a RuntimeError from missing required params, not a
        # silent wrong value, since this node treats them as required — see
        # its own __init__).
        detection_params = _load_node_params(
            os.path.join(pkg_dir, 'cfg', env_str, config_filename), 'yolo_seg_track_detector')
        for name in camera_names:
            detection_nodes.append(Node(
                package='sins',
                executable='yolo_seg_track_node.py',
                name=f'{node_prefix}_{name}',
                output='screen',
                parameters=[
                    detection_params,
                    {
                        'use_sim_time': use_sim_time,
                        'camera_names': [name],
                    },
                ],
            ))

    # Generic configurable-class semantic segmentation — separate model/node/
    # task from detection_nodes above (semantic, not instance segmentation;
    # see scripts/yolo_semantic_seg_detector.py's own header). One process
    # per camera, same GIL/CUDA-context reasoning as detection_nodes.
    # enable_semantic_segmentation/semantic_seg_params already loaded above
    # (ahead of preprocessing_node's own Node(...), which needs
    # semantic_seg_params too — see that block's own comment).
    semantic_seg_nodes = []
    if enable_semantic_segmentation:
        for name in camera_names:
            semantic_seg_nodes.append(Node(
                package='sins',
                executable='yolo_semantic_seg_node.py',
                name=f'yolo_semantic_seg_detector_{name}',
                output='screen',
                parameters=[
                    semantic_seg_params,
                    {
                        'use_sim_time': use_sim_time,
                        'camera_names': [name],
                    },
                ],
            ))

    # Detection phase's first model: 3D detection on raw /velodyne_points via
    # PointPillars — see scripts/pointpillars_detector.py's own header for
    # the full story (patched/vendored OpenPCDet, official nuScenes
    # checkpoint, known accuracy caveats, and why raw LiDAR rather than a
    # fused/preprocessed cloud — preprocessing_node.cpp used to also publish
    # one for this specifically, removed once that comparison was
    # conclusive). ONE instance regardless of camera_names (it consumes a
    # single whole-scene cloud, not a per-camera stream — see that file's
    # own header on why this differs from the per-camera pattern
    # detection_nodes/semantic_seg_nodes use).
    enable_pointpillars = LaunchConfiguration('enable_pointpillars').perform(context) == 'true'
    pointpillars_nodes = []
    if enable_pointpillars:
        pointpillars_params = _load_node_params(
            os.path.join(pkg_dir, 'cfg', env_str, 'pointpillars.yaml'), 'pointpillars_detector')
        pointpillars_nodes.append(Node(
            package='sins',
            executable='pointpillars_node.py',
            name='pointpillars_detector',
            output='screen',
            parameters=[
                pointpillars_params,
                {
                    'use_sim_time': use_sim_time,
                    'global_frame': preprocessing_params.get('global_frame', 'odom'),
                    'base_frame': preprocessing_params.get('base_frame', 'base_link'),
                    # Single source of truth: same odometry stream
                    # preprocessing_node itself uses for T_global_base_latest_
                    # (see scripts/pointpillars_detector.py's own header on
                    # why an Odometry subscription, not TF).
                    'odom_topic': preprocessing_params.get('odom_topic', '/odometry/filtered'),
                },
            ],
        ))

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz_tracks',
        output='screen',
        arguments=['-d', os.path.join(pkg_dir, 'rviz', 'tracks_only.rviz')],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    actions = (
        list(calibration_nodes) + [preprocessing_node, static_structures_node] + dbscan_detector_nodes
        + tracker_nodes + detection_nodes + semantic_seg_nodes + pointpillars_nodes + [rviz_node]
    )

    if enable_calibration and not use_robot_tf_guess and camera_names:
        # Initial-guess TF velodyne -> <first_camera>_initial_guess. Values
        # below are the most recently tuned hand-measured guess (matches the
        # active one in jo_detector/launch/run_detector.launch.py, not
        # the older one in that package's run_calibration_icp.launch.py) —
        # measured against front_camera only, so this fallback only ever
        # applies to camera_names[0]. Any additional camera has no
        # hand-measured guess and needs use_robot_tf_guess (the default).
        first_camera = camera_names[0]
        actions.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name=f'calib_icp_tf_velodyne_to_{first_camera}_initial_guess',
            parameters=[{'use_sim_time': use_sim_time}],
            arguments=[
                '--x', '0.169686', '--y', '0.137954', '--z', '-0.410085',
                '--roll', '-1.476806', '--pitch', '0.015591', '--yaw', '-1.584388',
                '--frame-id', lidar_frame,
                '--child-frame-id', f'{first_camera}_initial_guess',
            ],
        ))

    return actions


def generate_launch_description():
    # Defaults below all target bags/20260910_124939_validation_lab — the
    # only thing this package has actually been run against so far (v2 isn't
    # deployed on the real robot yet, see ../README.md). Override for a
    # different bag/live camera the same way run_calibration_icp.launch.py
    # documents for its own, identically-named args.
    env_arg = DeclareLaunchArgument(
        'env', default_value='indoor',
        description="Which cfg/<env>/ profile to use for every node in this launch: 'outdoor' or 'indoor'")
    tracker_debug_lifecycle_arg = DeclareLaunchArgument(
        'tracker_debug_lifecycle', default_value='false',
        description="tracker_node diagnostics: log every track birth/deletion and why nearby tracks didn't match")
    camera_names_arg = DeclareLaunchArgument(
        'camera_names', default_value='',
        description="Comma-separated camera list overriding cfg/<env>/preprocessing.yaml's camera_names "
                    "(empty = use the YAML list)")
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use /clock — set true when replaying bags with --clock')
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Whether to launch RViz (rviz/tracks_only.rviz — tracked boxes colored by dynamic/'
                     'potentially_dynamic/static status, fused clusters, processed LiDAR/depth clouds, and the '
                     'YOLO track camera views; see the other rviz/*.rviz configs for the box/preprocessing/'
                     'calibration/wall debug views this one does not show)')
    enable_calibration_arg = DeclareLaunchArgument(
        'enable_calibration', default_value='true',
        description="Whether to run lidar<->camera ICP calibration at all. true (current default): one "
                     'calibration_icp_node instance per camera, <name>_refined preferred once it converges '
                     '(see camera_frame_initial_guess_template/robot_tf_camera_frame below for its own '
                     'initial-guess options). false: skip launching calibration_icp_node entirely (and the '
                     "hand-measured initial-guess static TF, which nothing would consume) — preprocessing_node "
                     "and static_structures_node instead trust each camera's own raw URDF/bag TF chain permanently "
                     '(see calibration_enabled in both node\'s own source) — for when that chain is already '
                     "close enough that ICP isn't worth running.")
    robot_tf_camera_frame_arg = DeclareLaunchArgument(
        'robot_tf_camera_frame', default_value='true',
        description="If non-empty (default: 'true'), use each camera's own "
                     "<name>_color_optical_frame TF (already published by the robot's own "
                     'robot_state_publisher) as that camera\'s calibration_icp_node initial guess, '
                     'per camera_frame_initial_guess_template — skips the hand-measured static TF below '
                     'entirely. Pass an empty string to fall back to the hand-measured static TF instead '
                     '(only covers the FIRST camera in camera_names; any other camera has no such fallback).')
    camera_frame_initial_guess_template_arg = DeclareLaunchArgument(
        'camera_frame_initial_guess_template', default_value='{camera}_color_optical_frame',
        description="Per-camera initial-guess TF frame name, '{camera}' replaced with each camera_names "
                     'entry (e.g. front_camera -> front_camera_color_optical_frame). Only used when '
                     'robot_tf_camera_frame is non-empty.')
    # preprocessing_node's two independent depth streams (see that file's
    # header comment). aligned_depth_transport is the one that actually
    # matters day to day — raw_depth_transport is acquired-only for now
    # (future UV-map stage) and this bag has no unaligned depth topic at
    # all, so its default is moot either way.
    raw_depth_transport_arg = DeclareLaunchArgument(
        'raw_depth_transport', default_value='compressedDepth')
    aligned_depth_transport_arg = DeclareLaunchArgument(
        'aligned_depth_transport', default_value='zstd')

    # Every calibration_icp_node instance's depth stream (see
    # calibration_nodes' own comment above for why these can't just share
    # preprocessing's args). depth_topic is always
    # /<name>/camera/aligned_depth_to_color/image_raw (this bag has no
    # unaligned depth at all — same convention preprocessing_node.cpp's own
    # aligned_depth_topic default uses), depth_transport is shared across
    # cameras (both this robot's cameras record aligned depth as zstd — see
    # preprocessing.yaml's own comment on this), and depth_intrinsics is
    # necessarily per-camera (the COLOR camera's own intrinsics — see
    # run_calibration_icp.launch.py's env_arg comment for the full
    # reasoning, identical here), passed as a 'name:fx,fy,cx,cy;...' map
    # since intrinsics can't be derived from a camera's name the way topics
    # can.
    depth_transport_arg = DeclareLaunchArgument(
        'depth_transport', default_value='zstd',
        description="image_transport plugin every calibration_icp_node's depth_topic is published with")
    camera_depth_intrinsics_arg = DeclareLaunchArgument(
        'camera_depth_intrinsics',
        default_value='front_camera:644.1800537109375,643.2573852539062,647.415283203125,361.88623046875;'
                       'back_camera:642.116,641.205,648.238,373.166',
        description="Per-camera 'name:fx,fy,cx,cy;name2:fx,fy,cx,cy' COLOR-camera intrinsics map — optional "
                     "bootstrap-only guess (see calibration_nodes' own comment: real intrinsics come from "
                     "each camera's camera_info at runtime regardless). A camera_names entry (preprocessing.yaml) "
                     'with no matching key here just uses the node default instead of failing.')

    # Which detection model(s) to launch — independent toggles, same
    # true/false-string pattern as enable_calibration above. Both read the
    # SAME node/script (yolo_seg_track_node.py / yolo_seg_track_detector.py,
    # entirely weights_path/topic-driven, no model-specific code) under two
    # different cfg files and two different node-name/topic prefixes (see
    # cfg/<env>/detection_yolo26.yaml's own comment), so both CAN run at
    # once with no collision — both default OFF for now (GPU budget: this
    # package's per-camera-per-model process convention means every enabled
    # model is another concurrent GPU consumer sharing one 8GB laptop GPU
    # with everything else in run_detector.launch.py; enable_semantic_segmentation
    # below is the one kept on by default) — flip either back to true to
    # compare against semantic segmentation (they publish under different
    # topic prefixes, no collision).
    enable_yolo11_arg = DeclareLaunchArgument(
        'enable_yolo11', default_value='false',
        description='Whether to launch the YOLO11n-seg detection_nodes (cfg/<env>/detection.yaml). Off by '
                     'default — see the comment above this arg for why. Forced on regardless when '
                     'yolo_leaf_method=="mask" (that method needs this node\'s own instance masks).')
    yolo_leaf_method_arg = DeclareLaunchArgument(
        'yolo_leaf_method', default_value='box',
        description='How preprocessing_node turns a YOLO 2D detection into 3D "semantic leaf" points (see '
                     'preprocessing_node.cpp\'s own yolo_leaf_method_ comment). "box" (default): crop the '
                     'detection\'s 2D box out of the depth cloud, then isolate its own depth cluster within '
                     'that crop (histogram mode + valley-bounded window) — works with the existing COCO '
                     'detection leg (enable_semantic_segmentation\'s own enable_detection), full class '
                     'coverage, no extra GPU process. "mask": use yolo_seg_track_node\'s (enable_yolo11, '
                     'forced on) own per-pixel INSTANCE-segmentation masks instead — pixel-exact object '
                     'membership in image space (no 2D box/inflate heuristics needed), but STILL goes through '
                     'the same depth mode+valley window as "box" (a silhouette edge can have a background '
                     'point reproject into the mask\'s own footprint from depth<->color parallax alone), and '
                     'costs its own GPU inference pass and is limited to whatever cfg/<env>/detection.yaml\'s '
                     'target_classes lists (just "person" by default) rather than the full COCO set.')
    enable_yolo26_arg = DeclareLaunchArgument(
        'enable_yolo26', default_value='false',
        description='Whether to launch the YOLO26s-seg instance-segmentation detection_nodes '
                     '(cfg/<env>/detection_yolo26.yaml, weights confirmed working). Off by default — see the '
                     'comment above this arg for why.')
    enable_semantic_segmentation_arg = DeclareLaunchArgument(
        'enable_semantic_segmentation', default_value='true',
        description='Whether to launch yolo_semantic_seg_detector.py, one instance per camera — it also hosts '
                     'the YOLO 2D detection+tracking thread, so it is what runs the tracker even when '
                     'cfg/<env>/semantic_segmentation.yaml has enable_semantic: false (semantic model off) '
                     '(cfg/<env>/semantic_segmentation.yaml) — a separate model/task from enable_yolo11/'
                     'enable_yolo26 (semantic segmentation, not instance segmentation+tracking; see '
                     'scripts/yolo_semantic_seg_detector.py\'s own header). Which classes it segments is '
                     'entirely decided by that YAML\'s segmentation_classes list, not by this node. The one '
                     'model kept ON by default (see enable_yolo26/enable_yolo11\'s own comment) — 2 GPU '
                     'processes total (front+back) instead of 4.')
    enable_dbscan_detector_arg = DeclareLaunchArgument(
        'enable_dbscan_detector', default_value='true',
        description='Whether to launch dbscan_detector_node (cfg/<env>/dbscan_detector.yaml): DBSCAN + 3D OBB '
                     'detection on the processed LiDAR cloud and on every camera\'s processed depth cloud.')
    enable_tracker_arg = DeclareLaunchArgument(
        'enable_tracker', default_value='true',
        description='Whether to launch tracker_node (cfg/<env>/tracker.yaml): Kalman-filter tracking of '
                     'dbscan_detector_node\'s own fused objects into persistent ids with velocity and a '
                     'dynamic/static classification — see src/tracker_node.cpp\'s own header. No-op without '
                     'enable_dbscan_detector.')
    enable_pointpillars_arg = DeclareLaunchArgument(
        'enable_pointpillars', default_value='false',
        description='Whether to launch pointpillars_detector.py (cfg/<env>/pointpillars.yaml) — 3D detection '
                     'on raw /velodyne_points, see scripts/pointpillars_detector.py\'s own header. '
                     'Off by default: same GPU-budget reasoning as enable_yolo11/enable_yolo26 above (this is '
                     'yet another concurrent GPU consumer on the same 8GB laptop GPU) — test it on its own '
                     'first before enabling alongside enable_semantic_segmentation.')
    enable_semantic_ground_points_arg = DeclareLaunchArgument(
        'enable_semantic_ground_points', default_value='false',
        description='Whether preprocessing_node also builds semantic_ground_points: the depth points that '
                     'enable_semantic_segmentation\'s segmentation_mask marks as ground. On by default — '
                     'tracks_only.rviz shows it for both cameras as a direct visual comparison against the '
                     'RANSAC floor slab and gseg3d_ground. Works on both depth paths: per pixel on the aligned '
                     'depth image, or — for a camera using the native point cloud (aligned_depth_cloud_topic '
                     'set) — by projecting each point onto the mask through the depth->color extrinsic and the '
                     'color intrinsics. Only takes effect together with enable_semantic_segmentation AND that '
                     'node\'s own enable_semantic (cfg/<env>/semantic_segmentation.yaml, now true — it was the '
                     'real gate keeping this mask from ever being computed, see that file\'s own comment). This '
                     'arg alone is free (plain CPU deprojection), but enable_semantic itself is NOT: it forces a '
                     'full-resolution decode instead of the half-size one detection-only uses, plus a second '
                     'inference pass at inference_size (1024) on yolo_semantic_seg_detector\'s own already-busy '
                     'GPU process, on top of the person detection+tracking it also runs.')

    return LaunchDescription([
        env_arg,
        camera_names_arg,
        tracker_debug_lifecycle_arg,
        use_sim_time_arg,
        rviz_arg,
        enable_calibration_arg,
        enable_yolo11_arg,
        yolo_leaf_method_arg,
        enable_yolo26_arg,
        enable_semantic_segmentation_arg,
        enable_semantic_ground_points_arg,
        enable_dbscan_detector_arg,
        enable_tracker_arg,
        enable_pointpillars_arg,
        robot_tf_camera_frame_arg,
        camera_frame_initial_guess_template_arg,
        raw_depth_transport_arg,
        aligned_depth_transport_arg,
        depth_transport_arg,
        camera_depth_intrinsics_arg,
        OpaqueFunction(function=_launch_setup),
    ])
