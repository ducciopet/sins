cmake_minimum_required(VERSION 3.8)
project(sins)

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "Suppress third-party CMake developer warnings")

# --------------------------------------------------
# Compile options
# --------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-O3 -Wall)

# Set RPATH so binary can find libraries
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# --------------------------------------------------
# Find dependencies
# --------------------------------------------------
# Phase 1 (preprocessing + calibration) dependencies. Add more
# find_package()/ament_target_dependencies entries as later phases are built
# (vision_msgs/jo_msgs for detection and tracking messages,
# rosidl_default_generators if a srv/msg is added back, ...) instead of
# front-loading everything from jo_detector/CMakeLists.txt now.
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(image_transport REQUIRED)
unset(FLANN_ROOT)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
# RealSense's own depth<->color extrinsics message (see preprocessing_node.cpp
# file header for the "not verified against a real build yet" caveat on this
# one).
find_package(realsense2_camera_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
# jo_msgs/ObstacleArray: the pre-existing, stable interface glim_ros2's own BBOX dynamic-rejection
# (dynamic_rejection_type=="BBOX") already subscribes to — see tracker_node.cpp's own pub_obstacles_ comment.
find_package(jo_msgs REQUIRED)

# gtsam / gtsam_points: used by calibration_icp_node for GICP-based
# depth<->lidar registration (same registration stack already required by
# the glim submodule, see glim/CMakeLists.txt, and by jo_detector's own
# calibration_icp_node — this is a straight port of that node).
find_package(GTSAM 4.2 REQUIRED)
find_package(gtsam_points 1.2.0 REQUIRED)

# --------------------------------------------------
# Include directories
# --------------------------------------------------
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

# --------------------------------------------------
# Executable
# --------------------------------------------------
# Phase 1: preprocessing. src/preprocessing_node.cpp is currently a stub
# (subscribes + syncs depth/lidar, does nothing with them yet) — fill it in
# here rather than adding a second executable, until it's ready to split.
add_executable(preprocessing_node src/preprocessing_node.cpp)

ament_target_dependencies(preprocessing_node
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  visualization_msgs
  cv_bridge
  message_filters
  image_transport
  pcl_conversions
  tf2
  tf2_ros
  tf2_geometry_msgs
  realsense2_camera_msgs
  vision_msgs
  ament_index_cpp
)

target_link_libraries(preprocessing_node
  ${PCL_LIBRARIES}
)

# static_structures_node: standalone, asynchronous wall+floor detector — see
# src/static_structures_node.cpp's own header for the full architecture
# (subscribes to preprocessing_node's already-voxelized LiDAR/depth
# clouds instead of raw sensor topics, so it needs neither image_transport
# nor cv_bridge, unlike preprocessing_node). realsense2_camera_msgs IS
# needed though, same as preprocessing_node/calibration_icp_node: a camera
# whose voxelized_depth_topic is a native cloud needs the same depth<->color
# extrinsic correction those two apply — see CameraSource::is_native_depth_frame.
add_executable(static_structures_node src/static_structures_node.cpp)

ament_target_dependencies(static_structures_node
  rclcpp
  geometry_msgs
  sensor_msgs
  nav_msgs
  visualization_msgs
  pcl_conversions
  tf2
  tf2_ros
  realsense2_camera_msgs
)

target_link_libraries(static_structures_node
  ${PCL_LIBRARIES}
)

# calibration_icp_node: straight port of jo_detector's node (see
# ../jo_detector/src/calibration_icp_node.cpp) — same gtsam_points/GICP
# registration, same manual lidar/depth sync (message_filters::Synchronizer
# doesn't work here, see the comment on lidar_cache_ in the file itself).
# Only change: parameter names no longer carry the stale
# "jo_detector." prefix (depth_intrinsics, not
# jo_detector.depth_intrinsics). realsense2_camera_msgs added for its
# own native-cloud path's depth<->color extrinsic (see the file's own
# comments on depth_cloud_topic_/T_depth_color_) — same message
# preprocessing_node already depends on, for the identical purpose.
add_executable(calibration_icp_node src/calibration_icp_node.cpp)

ament_target_dependencies(calibration_icp_node
  rclcpp
  geometry_msgs
  sensor_msgs
  cv_bridge
  message_filters
  image_transport
  pcl_conversions
  tf2
  tf2_ros
  realsense2_camera_msgs
)

target_link_libraries(calibration_icp_node
  ${PCL_LIBRARIES}
  gtsam
  gtsam_points::gtsam_points
)

# dbscan_detector_node: DBSCAN + 3D OBB on the processed LiDAR cloud and on
# every camera's processed depth cloud, one tick per LiDAR scan — see
# src/dbscan_detector_node.cpp's header.
add_executable(dbscan_detector_node src/dbscan_detector_node.cpp)
ament_target_dependencies(dbscan_detector_node
  rclcpp
  nav_msgs
  sensor_msgs
  visualization_msgs
  vision_msgs
  ament_index_cpp
)
find_package(Eigen3 REQUIRED)
target_link_libraries(dbscan_detector_node Eigen3::Eigen)

# tracker_node: Kalman-filters dbscan_detector_node's fused objects into persistent tracks — see
# src/tracker_node.cpp's own header. sensor_msgs/tf2_ros: SEMANTIC DECAY's own per-camera FOV test (CameraInfo
# subscriptions + a TF lookup against each camera's "<camera>_refined" frame, same frame calibration_icp_node
# already publishes for preprocessing_node/static_structures_node's own alignment).
add_executable(tracker_node src/tracker_node.cpp)
ament_target_dependencies(tracker_node
  rclcpp
  vision_msgs
  visualization_msgs
  sensor_msgs
  tf2_ros
  jo_msgs
)
target_link_libraries(tracker_node Eigen3::Eigen)

# --------------------------------------------------
# Install
# --------------------------------------------------
install(TARGETS
  preprocessing_node
  static_structures_node
  dbscan_detector_node
  tracker_node
  calibration_icp_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY cfg/
  DESTINATION share/${PROJECT_NAME}/cfg
)

install(DIRECTORY rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
)

# Phase 2 prototype (segmentation + tracking — see
# scripts/yolo_seg_track_detector.py's file header), generic
# configurable-class semantic segmentation (yolo_semantic_seg_node.py/
# scripts/yolo_semantic_seg_detector.py, a separate node+model — see that
# file's own header), plus the detection phase's first model
# (pointpillars_node.py/scripts/pointpillars_detector.py — 3D detection on
# preprocessing_node's fused_cloud via a patched, vendored OpenPCDet, see
# that script's own header): the only Python in this package, so it can't go
# through install(TARGETS ...) like the C++ nodes above. Each NODE (src/) is
# a thin entrypoint that lives alongside those nodes' own .cpp sources and
# installs to the SAME lib/${PROJECT_NAME} destination as their executables
# (PROGRAMS, so ros2 run can find it). The actual implementation is a SCRIPT
# each imports at runtime (via get_package_share_directory, not
# sys.path-of-this-file) — installed separately under
# share/${PROJECT_NAME}/scripts/, same split jo_detector/CMakeLists.txt
# uses for its own yolo_detector/ scripts.
install(PROGRAMS
  src/yolo_seg_track_node.py
  src/yolo_semantic_seg_node.py
  src/pointpillars_node.py
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY scripts/
  DESTINATION share/${PROJECT_NAME}/scripts
  USE_SOURCE_PERMISSIONS
)

# Model weights for the Python detection node above — own top-level install
# destination (share/${PROJECT_NAME}/weights), resolved at runtime via
# get_package_share_directory (see yolo_seg_track_detector.py), same as
# cfg/ just above.
install(DIRECTORY weights/
  DESTINATION share/${PROJECT_NAME}/weights
)

# --------------------------------------------------
# Export
# --------------------------------------------------
ament_export_include_directories(include)
ament_package()
