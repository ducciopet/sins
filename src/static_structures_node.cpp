/*
    FILE: static_structures_node.cpp
    ---------------------------------
    Standalone, asynchronous wall+floor detector — architecturally a
    faithful port of jo_detector/scripts/wall_detector/wallDetector.{h,cpp}
    (own executable, own node, spun independently via rclcpp::spin, no
    coupling to preprocessing_node's own callbacks/executor): same iterative
    RANSAC, same PCA refitPlane, same WallBBoxRegistry persistent tracking
    (include/sins/wall_registry.hpp), same "detect in the
    sensor's own frame, transform only the small set of FOUND boxes/planes
    afterward" design as the original's pointCloudCallback.

    Two real differences from the original, both explicit per what this
    file was written for:

    1. Point ACQUISITION, not detection logic. The original subscribes
       directly to raw /velodyne_points (and a raw depth image, only for a
       flat ground-height estimate) and voxel-filters them itself every
       scan. This node instead subscribes to preprocessing_node's own
       already-voxel-downsampled — but deliberately NOT range/ground-
       cropped — LiDAR and per-camera depth clouds (see that file's own
       header comment on the "wall thickness" pass, and its voxelized_sensor
       publishers): the expensive voxel-grid pass happens exactly once, in
       preprocessing_node, and both this node and the general-purpose
       processed-cloud output reuse it, instead of each node repeating its
       own separate acquisition+voxelization of the same raw sensor data.
       Range cropping still happens here (own params, own moment — on the
       already-downsampled centroids, so it's cheap regardless) — same
       two-stage "voxelize once, crop separately per consumer" split, just
       with the voxelize half moved upstream.

    2. Floor is a real, potentially-tilted plane, fit by the SAME RANSAC
       engine that finds walls (near-horizontal classification — see
       detectFromSensorPoints/isFloorCandidate), not the original's separate
       depth-image bottom-rows RANSAC on z=c (flat only). It's transformed
       to global_frame_ and EMA-tracked (floor_ema_alpha) exactly the way a
       wall's box is — "puoi tranquillamente trasformare i vertici delle
       box e anche per i piani del pavimento utilizzare una tecnica simile"
       — normal + d instead of 8 corners, but the same "detect in sensor
       frame, transform the small result, not the point cloud" idea.

       Detection is now explicitly split by source, per request: LiDAR
       (onLidarVoxelized) only ever looks for WALLS, every camera in
       camera_names (onCameraVoxelized) only ever looks for FLOOR — see
       detectFromSensorPoints's detect_walls/detect_floor flags. Reasoning:
       the LiDAR's 360°, consistent-height scan is the better source for
       vertical wall planes across the whole range, while a downward-angled
       depth camera sees the floor immediately ahead at far higher density
       than the LiDAR's own sparse near-field rings — which is what
       actually matters for catching a slope early. This replaces an
       earlier version of this file where every source contributed BOTH
       categories independently ("more viewpoints = more evidence" for
       either); that's still true in principle (a camera COULD see a wall
       the LiDAR's own crop missed) but is deliberately not done here now.

       Floor is now ONE plane PER CAMERA, not one shared estimate merged
       across all of them: each camera's own CameraSource EMA-tracks only
       ITS OWN successive detections (mergeDetection's floor_source
       parameter), never blended with another camera's. preprocessing_node
       treats the space below the UNION of every camera's plane as "the
       ground" (see that file's floorHeightAboveGround()) — two cameras
       looking at the floor from different angles/positions can disagree
       slightly (mounting offset, local surface variation), and unioning
       their independent estimates is more robust than forcing one blended
       compromise plane that might fit neither view well.

    Everything else — RANSAC/refit/classification math, WallBBoxRegistry's
    match/merge/expiry semantics, the wall_markers MarkerArray as the
    actual data channel (not just visualization) other nodes parse back
    into planes, the "only call registry update() when this callback found
    at least one wall" quirk — is intentionally identical to the original.

    Published: /sins/static_structures/wall_markers
    (visualization_msgs/MarkerArray) — ns="wall_bbox" (CUBE per tracked
    wall, position=center/orientation=rotation/scale=size, exactly the
    original's wire format) plus one ns="floor" CUBE slab PER CAMERA
    (id = that camera's index in cameras_, orientation encodes that
    camera's own plane normal via its local +Z axis, position is a point ON
    the plane) — see publishMarkers(). preprocessing_node subscribes to
    this same topic and parses both back into plane equations for its own
    point-stripping crop — see that file's onWallMarkers().

    Also published, raw RANSAC inlier points behind the boxes/plane above
    (DetectionResult::wall_points_global/floor_points_global — debug-only,
    nothing subscribes to these for logic): /sins/
    static_structures/wall_points (LiDAR, all wall planes this scan combined —
    walls only ever come from LiDAR, see the split above) and
    /sins/<camera>/floor_points (one PER CAMERA, that
    camera's own RANSAC floor detection this callback — NOT the
    EMA-tracked floor_plane_global, its raw per-frame input instead).

    Also, per camera, /sins/<camera>/depth_calibrated: that
    camera's own INPUT points (same `pts` GSeg3D/RANSAC both consume, no
    filtering beyond finite-check), placed via T_global_cam — the SAME
    transform GSeg3D/wall_points/floor_points use, correct for that
    camera's own native-vs-deprojected convention (see onCameraVoxelized's
    own is_native_depth_frame handling) and for whether calibration_enabled_
    is on or off. Exists because rviz's own "Depth Original" display (the
    RAW /<camera>/depth/cloud topic from preprocessing_node, unmodified)
    resolves its frame_id through that camera's OWN TF chain instead — a
    DIFFERENT path, un-refined even when calibration_enabled_ is on (see
    that param's own comment) — so the two are only pixel-aligned by
    accident. This publishes the identical points through the correct path
    instead, so comparing them against gseg3d_ground/wall_points is
    actually meaningful regardless of calibration_enabled_.

    GSeg3D ground estimate (runGseg3d(), gseg3d_enabled) — an independent,
    SIDE-BY-SIDE ground/non-ground point-cloud split, published on its own
    debug topics, NOT wired into floor_plane_global/mergeDetection's
    EMA-tracked plane used by preprocessing_node's crop. Deliberately kept
    separate for now: this is meant to be compared against the existing
    RANSAC floor slab live (same rviz, same bag) before deciding whether it
    should replace or feed that plane at all. Runs on BOTH sources now —
    per camera (.../<camera>/gseg3d_ground, .../gseg3d_non_ground — same as
    that camera's own RANSAC floor detection) AND on LiDAR
    (.../lidar/gseg3d_ground, .../gseg3d_non_ground — LiDAR itself only
    ever does WALLS via RANSAC, see the split above; GSeg3D isn't
    source-restricted the way that split is, and LiDAR's 360° coverage
    reaches ground the cameras' limited FOV never sees at all, e.g. behind
    the robot). One shared runGseg3d() for both — see its own comment for
    why publishers are passed in rather than read off a CameraSource.
    Vendored header-only from include/sins/gseg3d/
    (github.com/dfki-ric/ground_segmentation, BSD-3-Clause — see that
    directory's own LICENSE file), a grid-based per-cell PCA/RANSAC/slope
    classifier — see that header's own comments for the algorithm. Runs on
    the SAME already-voxelized, finite-filtered points as the RANSAC path
    (that source's own `pts`), rotated into a gravity-aligned-but-
    sensor-centered frame first (T_global_sensor.linear() only, no
    translation) since the library bins points into its grid directly
    along their raw X/Y/Z — unlike detectFromSensorPoints's RANSAC (which
    only ever rotates a single reference "up" vector, see up_sensor, never
    the points themselves), GSeg3D's cellSizeZ must actually correspond to
    vertical for the grid to mean anything. Output points get the
    translation added back before
    publishing in global_frame_, so they render directly alongside every
    other global_frame_ display in rviz without needing a TF lookup.
*/
#include <array>
#include <numeric>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
// Depth<->color factory extrinsic — needed only for cameras whose
// voxelized_depth_topic is a native (depth-optical-frame) cloud, not the
// deprojected (color-optical-frame) one. See CameraSource::is_native_depth_frame
// and onDepthToColorExtrinsics()'s own comment; same message and math as
// preprocessing_node.cpp's/calibration_icp_node.cpp's own use of it.
#include <realsense2_camera_msgs/msg/extrinsics.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/exceptions.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "sins/wall_registry.hpp"
#include "sins/gseg3d/ground_detection.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <memory>

using sins::WallBBox;
using sins::WallBBoxRegistry;

class StaticStructuresNode : public rclcpp::Node {
public:
    StaticStructuresNode() : Node("static_structures_node"), rng_(std::random_device{}()) {
        lidar_frame_ = this->declare_parameter("lidar_frame", std::string("velodyne"));
        base_frame_ = this->declare_parameter("base_frame", std::string("base_link"));
        global_frame_ = this->declare_parameter("global_frame", std::string("odom"));
        odom_topic_ = this->declare_parameter("odom_topic", std::string("/odometry/filtered"));
        // false = skip lidar_frame_ -> <name>_refined entirely and place
        // every camera's cloud via the robot's own raw URDF/bag TF chain
        // instead (base_frame_ -> whatever frame it actually arrives in —
        // see ensureBaseCameraTf()) — for when that raw chain is already
        // close enough that ICP calibration isn't worth running at all.
        // Forwarded from run_detector.launch.py's own enable_calibration
        // arg, same single source of truth preprocessing_node.cpp's own
        // calibration_enabled_ uses (and whether calibration_icp_node even
        // gets launched in the first place).
        calibration_enabled_ = this->declare_parameter("calibration_enabled", true);
        lidar_voxelized_topic_ = this->declare_parameter(
            "lidar_voxelized_topic", std::string("/sins/lidar/voxelized_sensor"));

        // Own range crop, applied to preprocessing_node's already-voxelized
        // (not range-cropped) LiDAR centroids — see the file header. Cheap
        // regardless of value: it runs on a few hundred/thousand points,
        // not the ~29k-point raw scan.
        lidar_range_x_ = this->declare_parameter("lidar_range_x", 15.0);
        lidar_range_y_ = this->declare_parameter("lidar_range_y", 15.0);

        max_planes_ = this->declare_parameter("max_planes", 6);
        ransac_max_iterations_ = this->declare_parameter("ransac_max_iterations", 80);
        ransac_inlier_threshold_ = this->declare_parameter("ransac_inlier_threshold", 0.07);
        ransac_min_inliers_ = this->declare_parameter("ransac_min_inliers", 25);
        ransac_confidence_ = this->declare_parameter("ransac_confidence", 0.99);
        wall_vertical_angle_deg_ = this->declare_parameter("wall_vertical_angle_deg", 5.0);
        wall_bbox_max_aspect_ratio_ = this->declare_parameter("wall_bbox_max_aspect_ratio", 5.0);
        // Rendered/published wall thickness (buildBoxFromPlane's size.x(), the box's along-normal extent) is
        // its own knob, decoupled from ransac_inlier_threshold_ (which stays the RANSAC point-to-plane fit
        // tolerance): thickening the visual box shouldn't loosen what counts as a plane inlier.
        wall_min_thickness_ = this->declare_parameter("wall_min_thickness", 0.20);
        ransac_local_radius_ = this->declare_parameter("ransac_local_radius", 0.0);
        marker_lifetime_sec_ = this->declare_parameter("marker_lifetime_sec", 0.5);
        lidar_wall_mode_ = this->declare_parameter("lidar_wall_mode", std::string("ransac3d"));
        wall2d_iterations_ = this->declare_parameter("wall2d_iterations", 150);
        wall2d_local_radius_ = this->declare_parameter("wall2d_local_radius", 1.5);
        wall2d_min_vertical_extent_ = this->declare_parameter("wall2d_min_vertical_extent", 1.0);
        wall2d_max_gap_ = this->declare_parameter("wall2d_max_gap", 1.5);
        wall2d_min_length_ = this->declare_parameter("wall2d_min_length", 0.0);
        wall2d_bin_ = this->declare_parameter("wall2d_bin", 0.5);
        wall2d_min_bin_fraction_ = this->declare_parameter("wall2d_min_bin_fraction", 0.0);
        wall2d_min_bin_points_ = this->declare_parameter("wall2d_min_bin_points", 2);
        wall2d_suppress_dist_ = this->declare_parameter("wall2d_suppress_dist", 0.0);
        wall2d_axis_align_tol_deg_ = this->declare_parameter("wall2d_axis_align_tol_deg", 0.0);
        wall2d_split_at_intersections_ = this->declare_parameter("wall2d_split_at_intersections", false);
        wall_split_at_junctions_ = this->declare_parameter("wall_split_at_junctions", false);
        wall2d_perp_tol_deg_ = this->declare_parameter("wall2d_perp_tol_deg", 20.0);
        wall2d_split_margin_ = this->declare_parameter("wall2d_split_margin", 0.5);
        wall2d_split_half_gap_ = this->declare_parameter("wall2d_split_half_gap", 0.3);
        wall2d_junction_reach_ = this->declare_parameter("wall2d_junction_reach", 1.0);
        nested_merge_iov_thresh_ = this->declare_parameter("nested_merge_iov_thresh", 0.5);

        // Floor plane fit — see isFloorCandidate/detectFromSensorPoints.
        // ground_height_ only bootstraps each camera's OWN floor_plane_global
        // (flat, until that camera's first real fit arrives — see
        // setupCamera(), which reads this member) and positions its floor
        // marker before that — same bootstrap idea preprocessing_node's own
        // ground_height param uses.
        floor_max_tilt_deg_ = this->declare_parameter("floor_max_tilt_deg", 25.0);
        floor_ema_alpha_ = this->declare_parameter("floor_ema_alpha", 0.1);
        ground_height_ = this->declare_parameter("ground_height", -0.3);

        // GSeg3D ground estimate — see the file header's own section and
        // runGseg3d(). Independent of the RANSAC floor above: no EMA, no
        // registry, just a per-frame ground/non-ground split published for
        // side-by-side comparison. slopeThresholdDegrees intentionally
        // REUSES floor_max_tilt_deg_ (no separate gseg3d_max_tilt_deg param)
        // so the two methods are judged against the same traversability
        // angle, not two independently-tuned thresholds that could make
        // one look better than the other for the wrong reason.
        // Max rate at which each camera's cloud is processed (floor RANSAC).
        // 0 = every message. The floor changes slowly, so 30 Hz is wasted
        // work; floor_ema_alpha should be scaled by the same factor to keep
        // the same smoothing time constant.
        camera_process_hz_ = this->declare_parameter("camera_process_hz", 0.0);
        gseg3d_enabled_ = this->declare_parameter("gseg3d_enabled", true);
        gseg3d_config_.cellSizeX = this->declare_parameter("gseg3d_cell_size_xy", 1.0);
        gseg3d_config_.cellSizeY = gseg3d_config_.cellSizeX;
        gseg3d_config_.cellSizeZ = this->declare_parameter("gseg3d_cell_size_z", 10.0);
        gseg3d_config_.slopeThresholdDegrees = floor_max_tilt_deg_;
        gseg3d_config_.groundInlierThreshold = this->declare_parameter("gseg3d_ground_inlier_threshold", 0.1);
        gseg3d_config_.centroidSearchRadius = this->declare_parameter("gseg3d_centroid_search_radius", 5.0);
        gseg3d_config_.maxGroundHeightDeviation = this->declare_parameter("gseg3d_max_ground_height_deviation", 0.3);
        gseg3d_config_.processing_phase = static_cast<uint16_t>(this->declare_parameter("gseg3d_processing_phase", 1));

        WallBBoxRegistry::Config reg_cfg;
        reg_cfg.overlap_threshold = this->declare_parameter("wall_overlap_threshold", 0.3);
        reg_cfg.merge_weight = this->declare_parameter("wall_merge_weight", 0.3);
        reg_cfg.enable_expiry = this->declare_parameter("wall_enable_expiry", true);
        reg_cfg.max_missed_frames = this->declare_parameter("wall_max_missed_frames", 30);
        reg_cfg.min_normal_dot = this->declare_parameter("wall_min_normal_dot", 0.9);
        reg_cfg.max_center_distance = this->declare_parameter("wall_max_center_distance", 5.0);
        reg_cfg.plane_match_dist = this->declare_parameter("wall_plane_match_dist", 0.0);
        reg_cfg.plane_match_gap = this->declare_parameter("wall_plane_match_gap", 1.0);
        wall_registry_ = std::make_shared<WallBBoxRegistry>(reg_cfg);

        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this, false);

        pub_wall_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/sins/static_structures/wall_markers", rclcpp::QoS(10));
        // Raw RANSAC inlier points behind wall_markers' ns="wall_bbox" boxes
        // above — see DetectionResult::wall_points_global's own comment.
        // Single/shared: only LiDAR ever contributes walls (see
        // onLidarVoxelized), so there's no per-camera split to make here,
        // unlike floor_points below.
        pub_wall_points_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/sins/static_structures/wall_points", rclcpp::SensorDataQoS());
        // GSeg3D on LiDAR — see runGseg3d()'s own comment. Single/shared,
        // same reasoning as pub_wall_points_ above: one LiDAR, not one per
        // camera.
        pub_gseg3d_lidar_ground_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/sins/lidar/gseg3d_ground", rclcpp::SensorDataQoS());
        pub_gseg3d_lidar_non_ground_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/sins/lidar/gseg3d_non_ground", rclcpp::SensorDataQoS());

        sub_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
            odom_topic_, rclcpp::SensorDataQoS(),
            [this](const nav_msgs::msg::Odometry::ConstSharedPtr& msg) { onOdom(msg); });
        sub_lidar_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            lidar_voxelized_topic_, rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) { onLidarVoxelized(msg); });

        // ---- Cameras: an arbitrary list, matching preprocessing_node's own
        // camera_names (pass the SAME list at launch — see
        // run_detector.launch.py — so the two can't disagree about which
        // cameras exist). Each name gets its own voxelized-depth
        // subscription (default topic built from the name, matching
        // preprocessing_node.cpp's own "<name>/depth/voxelized_sensor"
        // publisher) and its own calibration TF (lidar_frame_ -> <name>_refined,
        // same convention as preprocessing_node's CameraStream). ----
        camera_names_ = this->declare_parameter(
            "camera_names", std::vector<std::string>{"front_camera"});
        for (const auto& name : camera_names_) {
            setupCamera(name);
        }

        RCLCPP_INFO(this->get_logger(), "static_structures_node ready");
        RCLCPP_INFO(this->get_logger(), "  lidar voxelized: %s (own range crop %.1fx%.1f)",
                    lidar_voxelized_topic_.c_str(), lidar_range_x_, lidar_range_y_);
        for (const auto& cam : cameras_) {
            RCLCPP_INFO(this->get_logger(), "  [%s] voxelized depth: %s (calib TF %s -> %s)",
                        cam->name.c_str(), cam->voxelized_depth_topic.c_str(),
                        lidar_frame_.c_str(), cam->refined_camera_frame.c_str());
        }
        RCLCPP_INFO(this->get_logger(), "  ransac: max_planes=%d iters=%d inlier_thresh=%.3f min_inliers=%d",
                    max_planes_, ransac_max_iterations_, ransac_inlier_threshold_, ransac_min_inliers_);
        RCLCPP_INFO(this->get_logger(), "  floor (per camera, one plane each): max_tilt=%.1fdeg ema_alpha=%.2f bootstrap_height=%.2f",
                    floor_max_tilt_deg_, floor_ema_alpha_, ground_height_);
    }

private:
    // A plane n·p + d = 0. Always GLOBAL frame once stored (each camera's
    // own floor_plane_global, wall boxes in the registry) — detection itself
    // runs in each source's
    // own sensor frame (see detectFromSensorPoints's "up" parameter), then
    // gets transformed once via transformPlane()/WallBBox::transform(), same
    // split the original's pointCloudCallback makes (see its own comment on
    // why: transforming ~15 numbers per found box/plane is vastly cheaper
    // every scan than transforming the whole point cloud).
    struct PlaneModel {
        Eigen::Vector3d normal{Eigen::Vector3d::UnitZ()};
        double d{0.0};
        double distance(const Eigen::Vector3d& p) const { return std::abs(normal.dot(p) + d); }
    };

    struct CameraSource {
        std::string name;
        std::string voxelized_depth_topic;
        std::string refined_camera_frame;
        bool has_lidar_camera_tf{false};
        Eigen::Matrix4d T_lidar_camera{Eigen::Matrix4d::Identity()};
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub;

        // True when voxelized_depth_topic carries a NATIVE point cloud
        // (preprocessing_node's onNativeDepthCloud path — arrives in this
        // camera's own depth_optical_frame) rather than the deprojected one
        // (arrives in color_optical_frame, the convention refined_camera_frame
        // — i.e. T_lidar_camera above — was actually calibrated against).
        // Forwarded from run_detector.launch.py, mirroring whatever
        // preprocessing.yaml's own <name>.aligned_depth_cloud_topic says for
        // this camera — see that param's own comment. When true, the SAME
        // depth<->color correction preprocessing_node.cpp applies internally
        // (finishDepthCloud's is_native_depth_frame) is needed here too,
        // before this camera's points can be treated as sensor-frame input
        // in refined_camera_frame's own convention — see onCameraVoxelized().
        bool is_native_depth_frame{false};
        std::string depth_to_color_extrinsics_topic;
        bool has_depth_color_matrix{false};
        // p_coloropt = T_color_depth * p_depthopt — the RealSense
        // depth_to_color message's own R,t (NOT inverted), same convention as
        // preprocessing_node.cpp's CameraStream::T_color_depth.
        Eigen::Matrix4d T_color_depth{Eigen::Matrix4d::Identity()};
        rclcpp::Subscription<realsense2_camera_msgs::msg::Extrinsics>::SharedPtr sub_depth_to_color_extrinsics;

        // calibration_enabled_ == false only: cached base_frame_ ->
        // (whichever frame this camera's cloud actually arrives in) TF —
        // see ensureBaseCameraTf()'s own comment. Same "rigid, cache once"
        // pattern as ensureStaticTf()/ensureBaseDepthTf() (preprocessing_node.cpp's
        // own equivalent for the same purpose).
        bool has_base_camera_tf{false};
        std::string cached_camera_frame;
        Eigen::Matrix4d T_base_camera{Eigen::Matrix4d::Identity()};

        // This camera's OWN floor plane — per the explicit request, floor
        // detection is per-camera now, not one shared/merged estimate: each
        // camera EMA-tracks only its own successive detections (see
        // mergeDetection), never blended with another camera's. Bootstrapped
        // flat at ground_height in the ctor, same "keep the last estimate,
        // never revert to the bootstrap" idea the old single-plane version
        // used, just duplicated per camera instead of shared.
        bool has_floor_estimate{false};
        PlaneModel floor_plane_global;
        double last_processed_stamp{-1e9};

        // GSeg3D debug output — see runGseg3d(). Independent of the
        // floor_plane_global EMA state above; published every callback
        // (when gseg3d_enabled_), not tracked/merged across frames.
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_gseg3d_ground;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_gseg3d_non_ground;

        // This camera's own input points (cam.voxelized_depth_topic — same
        // `pts` GSeg3D/RANSAC both consume), pre-transformed into
        // global_frame_ via the SAME T_global_cam those two use, and
        // published there directly (no TF lookup needed to view it) — see
        // onCameraVoxelized()'s own comment on why this exists: rviz's
        // "Depth Original" display resolves its frame_id through the
        // camera's OWN (uncalibrated when calibration_enabled_, still
        // un-refined regardless) URDF TF chain, a DIFFERENT path than the
        // calibrated one GSeg3D/RANSAC use — so the two are only ever
        // pixel-aligned by accident. This publishes the same input points
        // through the correct/calibrated path instead, so it's directly,
        // meaningfully comparable to gseg3d_ground/wall_points regardless
        // of whether calibration_enabled_ is on or off.
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_depth_calibrated;

        // Raw RANSAC floor-plane inlier points (this camera's OWN detection
        // each callback — NOT the EMA-tracked floor_plane_global above) —
        // see DetectionResult::floor_points_global's own comment. Per
        // camera, unlike pub_wall_points_ above: floor is detected per
        // camera too (floor_source=&cam in mergeDetection), so each
        // camera's own points get their own topic, same convention as
        // pub_gseg3d_ground/non_ground.
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_floor_points;
    };

    struct DetectionResult {
        std::vector<WallBBox> wall_boxes_global;
        bool floor_found{false};
        PlaneModel floor_plane_global;

        // Raw RANSAC inlier points behind the boxes/plane above, already in
        // GLOBAL frame — for the wall_points/floor_points debug clouds (see
        // onLidarVoxelized/onCameraVoxelized). wall_points_global accumulates
        // across EVERY wall plane found this call (up to max_planes_, see
        // detectFromSensorPoints's loop); floor_points_global holds just the
        // one floor plane's inliers (detect_floor only ever accepts the
        // FIRST candidate, !result.floor_found gate).
        std::vector<Eigen::Vector3d> wall_points_global;
        std::vector<Eigen::Vector3d> floor_points_global;
    };

    // ---- Setup ----
    void setupCamera(const std::string& name) {
        auto cam = std::make_shared<CameraSource>();
        cam->name = name;
        auto p = [&](const std::string& key, auto def) {
            return this->declare_parameter(name + "." + key, def);
        };
        cam->voxelized_depth_topic = p(
            "voxelized_depth_topic", std::string("/sins/") + name + "/depth/voxelized_sensor");
        cam->refined_camera_frame = p("refined_camera_frame", name + "_refined");
        cam->floor_plane_global.normal = Eigen::Vector3d::UnitZ();
        cam->floor_plane_global.d = -ground_height_;
        cam->sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            cam->voxelized_depth_topic, rclcpp::SensorDataQoS(),
            [this, cam](const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) { onCameraVoxelized(*cam, msg); });
        cam->pub_gseg3d_ground = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            std::string("/sins/") + name + "/gseg3d_ground", rclcpp::SensorDataQoS());
        cam->pub_gseg3d_non_ground = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            std::string("/sins/") + name + "/gseg3d_non_ground", rclcpp::SensorDataQoS());
        cam->pub_floor_points = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            std::string("/sins/") + name + "/floor_points", rclcpp::SensorDataQoS());
        cam->pub_depth_calibrated = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            std::string("/sins/") + name + "/depth_calibrated", rclcpp::SensorDataQoS());

        cam->is_native_depth_frame = p("is_native_depth_frame", false);
        if (cam->is_native_depth_frame) {
            cam->depth_to_color_extrinsics_topic = p(
                "depth_to_color_extrinsics_topic", std::string("/") + name + "/camera/extrinsics/depth_to_color");
            // transient_local — see calibration_icp_node.cpp's/
            // preprocessing_node.cpp's own subscription to this exact topic
            // for the full reasoning (published exactly once, latched).
            cam->sub_depth_to_color_extrinsics = this->create_subscription<realsense2_camera_msgs::msg::Extrinsics>(
                cam->depth_to_color_extrinsics_topic, rclcpp::QoS(1).transient_local(),
                [this, cam](const realsense2_camera_msgs::msg::Extrinsics::ConstSharedPtr& msg) { onDepthToColorExtrinsics(*cam, msg); });
        }

        cameras_.push_back(cam);
    }

    // Native path only: caches the RealSense factory depth<->color
    // extrinsic. Mirrors preprocessing_node.cpp's/calibration_icp_node.cpp's
    // own onDepthToColorExtrinsics() — same message, same math.
    void onDepthToColorExtrinsics(CameraSource& cam, const realsense2_camera_msgs::msg::Extrinsics::ConstSharedPtr& msg) {
        if (cam.has_depth_color_matrix) {
            return;  // static, only needs to be cached once
        }
        Eigen::Matrix3d R;
        R << msg->rotation[0], msg->rotation[3], msg->rotation[6],
             msg->rotation[1], msg->rotation[4], msg->rotation[7],
             msg->rotation[2], msg->rotation[5], msg->rotation[8];
        Eigen::Quaterniond q(R);
        q.normalize();
        cam.T_color_depth.setIdentity();
        cam.T_color_depth.block<3, 3>(0, 0) = q.toRotationMatrix();
        cam.T_color_depth(0, 3) = msg->translation[0];
        cam.T_color_depth(1, 3) = msg->translation[1];
        cam.T_color_depth(2, 3) = msg->translation[2];
        cam.has_depth_color_matrix = true;
        RCLCPP_INFO(this->get_logger(), "[%s] Cached depth<->color extrinsics from %s",
                    cam.name.c_str(), cam.depth_to_color_extrinsics_topic.c_str());
    }

    static Eigen::Isometry3d toIsometry(const Eigen::Matrix4d& M) {
        Eigen::Isometry3d T;
        T.matrix() = M;
        return T;
    }

    static Eigen::Matrix4d odomToGlobalBase(const nav_msgs::msg::Odometry::ConstSharedPtr& odom) {
        Eigen::Quaterniond q(
            odom->pose.pose.orientation.w, odom->pose.pose.orientation.x,
            odom->pose.pose.orientation.y, odom->pose.pose.orientation.z);
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
        T(0, 3) = odom->pose.pose.position.x;
        T(1, 3) = odom->pose.pose.position.y;
        T(2, 3) = odom->pose.pose.position.z;
        return T;
    }

    // Cached rigid base_frame_ -> lidar_frame_ TF — same "looked up once,
    // zero-timeout, assumed rigid forever after" pattern as
    // preprocessing_node's ensureStaticTf() (own tf_buffer_ here, no
    // dedicated thread, so a nonzero timeout would just spam warnings).
    bool ensureStaticTf() {
        if (has_static_tf_) return true;
        try {
            auto tf_stamped = tf_buffer_->lookupTransform(base_frame_, lidar_frame_, tf2::TimePointZero);
            Eigen::Quaterniond q(
                tf_stamped.transform.rotation.w, tf_stamped.transform.rotation.x,
                tf_stamped.transform.rotation.y, tf_stamped.transform.rotation.z);
            T_base_lidar_.setIdentity();
            T_base_lidar_.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
            T_base_lidar_(0, 3) = tf_stamped.transform.translation.x;
            T_base_lidar_(1, 3) = tf_stamped.transform.translation.y;
            T_base_lidar_(2, 3) = tf_stamped.transform.translation.z;
            has_static_tf_ = true;
            RCLCPP_INFO(this->get_logger(), "Cached static TF %s -> %s", base_frame_.c_str(), lidar_frame_.c_str());
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Waiting for static TF %s -> %s: %s", base_frame_.c_str(), lidar_frame_.c_str(), ex.what());
        }
        return has_static_tf_;
    }

    // Per-camera lidar_frame_ -> <name>_refined calibration TF, same
    // pattern/caveats as preprocessing_node's own ensureCameraCalibrationTf
    // (own copy here — this node has its own tf_buffer_, independent of
    // preprocessing_node's).
    bool ensureCameraCalibrationTf(CameraSource& cam) {
        if (cam.has_lidar_camera_tf) return true;
        try {
            auto tf_stamped = tf_buffer_->lookupTransform(lidar_frame_, cam.refined_camera_frame, tf2::TimePointZero);
            Eigen::Quaterniond q(
                tf_stamped.transform.rotation.w, tf_stamped.transform.rotation.x,
                tf_stamped.transform.rotation.y, tf_stamped.transform.rotation.z);
            cam.T_lidar_camera.setIdentity();
            cam.T_lidar_camera.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
            cam.T_lidar_camera(0, 3) = tf_stamped.transform.translation.x;
            cam.T_lidar_camera(1, 3) = tf_stamped.transform.translation.y;
            cam.T_lidar_camera(2, 3) = tf_stamped.transform.translation.z;
            cam.has_lidar_camera_tf = true;
            RCLCPP_INFO(this->get_logger(), "Cached calibration TF %s -> %s for camera '%s'",
                        lidar_frame_.c_str(), cam.refined_camera_frame.c_str(), cam.name.c_str());
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Waiting for calibration TF %s -> %s (camera '%s'): %s",
                lidar_frame_.c_str(), cam.refined_camera_frame.c_str(), cam.name.c_str(), ex.what());
        }
        return cam.has_lidar_camera_tf;
    }

    // calibration_enabled_ == false only: base_frame_ -> frame_id, straight
    // off the robot's own URDF/bag TF chain — same purpose and pattern as
    // preprocessing_node.cpp's own ensureBaseDepthTf() (own tf_buffer_
    // here, independent copy, same reasoning as ensureCameraCalibrationTf
    // above being its own copy too). frame_id is whatever
    // msg->header.frame_id actually names for THIS message (native
    // depth-optical or deprojected color-optical) — a direct lookup to that
    // exact frame needs no depth<->color correction, unlike the
    // calibration_enabled_ path above.
    bool ensureBaseCameraTf(CameraSource& cam, const std::string& frame_id) {
        if (cam.has_base_camera_tf && cam.cached_camera_frame == frame_id) return true;
        try {
            auto tf_stamped = tf_buffer_->lookupTransform(base_frame_, frame_id, tf2::TimePointZero);
            Eigen::Quaterniond q(
                tf_stamped.transform.rotation.w, tf_stamped.transform.rotation.x,
                tf_stamped.transform.rotation.y, tf_stamped.transform.rotation.z);
            cam.T_base_camera.setIdentity();
            cam.T_base_camera.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
            cam.T_base_camera(0, 3) = tf_stamped.transform.translation.x;
            cam.T_base_camera(1, 3) = tf_stamped.transform.translation.y;
            cam.T_base_camera(2, 3) = tf_stamped.transform.translation.z;
            cam.has_base_camera_tf = true;
            cam.cached_camera_frame = frame_id;
            RCLCPP_INFO(this->get_logger(), "[%s] Cached static TF %s -> %s (calibration disabled)",
                        cam.name.c_str(), base_frame_.c_str(), frame_id.c_str());
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "[%s] Waiting for static TF %s -> %s: %s", cam.name.c_str(), base_frame_.c_str(), frame_id.c_str(), ex.what());
        }
        return cam.has_base_camera_tf;
    }

    void onOdom(const nav_msgs::msg::Odometry::ConstSharedPtr& msg) {
        T_global_base_latest_ = odomToGlobalBase(msg);
        has_odom_ = true;
    }

    // =====================================================================
    // RANSAC / classification — same algorithm as
    // jo_detector/wall_detector's WallDetector (fitPlane/findInliers/
    // refitPlane/ransacOnce/isWallPlane/buildWallBBox), including the
    // confidence-based early stop ransacOnce uses.
    // =====================================================================

    static PlaneModel fitPlane3(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1, const Eigen::Vector3d& p2) {
        const Eigen::Vector3d n = (p1 - p0).cross(p2 - p0);
        const double norm = n.norm();
        PlaneModel plane;
        if (norm < 1e-9) {
            plane.normal = Eigen::Vector3d::UnitZ();
            plane.d = -plane.normal.dot(p0);
            return plane;
        }
        plane.normal = n / norm;
        plane.d = -plane.normal.dot(p0);
        return plane;
    }

    static std::vector<int> findPlaneInliers(
            const std::vector<Eigen::Vector3d>& pts, const PlaneModel& plane, double threshold) {
        std::vector<int> inliers;
        inliers.reserve(pts.size() / 4);
        for (int i = 0; i < static_cast<int>(pts.size()); ++i) {
            if (plane.distance(pts[i]) < threshold) inliers.push_back(i);
        }
        return inliers;
    }

    // PCA refit over all inliers — see wall_registry.hpp's merge() comment
    // for why this matters beyond accuracy: it's what keeps a tracked
    // wall's per-scan orientation jitter (and therefore, combined with the
    // registry's EMA-thickness merge, its long-run thickness) bounded.
    static PlaneModel refitPlane(const std::vector<Eigen::Vector3d>& pts, const std::vector<int>& inlier_idx) {
        Eigen::Vector3d mean = Eigen::Vector3d::Zero();
        for (int idx : inlier_idx) mean += pts[idx];
        mean /= static_cast<double>(inlier_idx.size());

        Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
        for (int idx : inlier_idx) {
            const Eigen::Vector3d d = pts[idx] - mean;
            cov += d * d.transpose();
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
        const Eigen::Vector3d normal = solver.eigenvectors().col(0).normalized();

        PlaneModel plane;
        plane.normal = normal;
        plane.d = -normal.dot(mean);
        return plane;
    }

    bool ransacOnePlane(std::vector<Eigen::Vector3d>& pts, PlaneModel& best_plane,
                         std::vector<Eigen::Vector3d>& out_inliers) {
        const int n = static_cast<int>(pts.size());
        if (n < 3) return false;

        std::uniform_int_distribution<int> dist(0, n - 1);
        int best_count = 0;
        std::vector<int> best_inlier_idx;

        std::vector<int> local;
        for (int iter = 0; iter < ransac_max_iterations_; ++iter) {
            int i0 = dist(rng_), i1, i2;
            if (ransac_local_radius_ > 0.0) {
                // Localized sampling: the other two points come from the neighbourhood of the first one. With
                // uniform triples a plane holding ~10% of the points is hit with probability ~0.1^3 per iteration,
                // so 80 iterations almost never find a partition; a triple drawn inside a small ball lies on the
                // same surface most of the time.
                local.clear();
                const double r2 = ransac_local_radius_ * ransac_local_radius_;
                for (int j = 0; j < n; ++j)
                    if (j != i0 && (pts[j] - pts[i0]).squaredNorm() <= r2) local.push_back(j);
                if (local.size() < 2) continue;
                std::uniform_int_distribution<int> ld(0, static_cast<int>(local.size()) - 1);
                i1 = local[ld(rng_)];
                do { i2 = local[ld(rng_)]; } while (i2 == i1);
            } else {
                do { i1 = dist(rng_); } while (i1 == i0);
                do { i2 = dist(rng_); } while (i2 == i0 || i2 == i1);
            }

            const PlaneModel candidate = fitPlane3(pts[i0], pts[i1], pts[i2]);
            if (candidate.normal.norm() < 0.5) continue;

            auto inliers = findPlaneInliers(pts, candidate, ransac_inlier_threshold_);
            if (static_cast<int>(inliers.size()) > best_count) {
                best_count = static_cast<int>(inliers.size());
                best_inlier_idx = std::move(inliers);
                best_plane = candidate;

                const double w = static_cast<double>(best_count) / n;
                const double w3 = w * w * w;
                if (w3 > 1.0 - 1e-9) break;
                const double n_iter = std::log(1.0 - ransac_confidence_) / std::log(1.0 - w3 + 1e-10);
                if (iter + 1 >= static_cast<int>(std::ceil(n_iter))) break;
            }
        }

        if (best_count < ransac_min_inliers_) return false;

        best_plane = refitPlane(pts, best_inlier_idx);

        out_inliers.clear();
        out_inliers.reserve(best_count);
        for (int idx : best_inlier_idx) out_inliers.push_back(pts[idx]);

        std::vector<bool> is_inlier(n, false);
        for (int idx : best_inlier_idx) is_inlier[idx] = true;
        std::vector<Eigen::Vector3d> rest;
        rest.reserve(n - best_count);
        for (int i = 0; i < n; ++i) {
            if (!is_inlier[i]) rest.push_back(pts[i]);
        }
        pts = std::move(rest);
        return true;
    }

    // up: world +Z expressed in whatever frame plane.normal is in (this
    // source's own sensor frame — see detectFromSensorPoints) — same
    // generalization jo_detector/wall_detector's own isWallPlane makes
    // for detecting off the lidar's own (possibly non-level) frame.
    bool isWallPlane(const PlaneModel& plane, const Eigen::Vector3d& up) const {
        const double sin_thr = std::sin(wall_vertical_angle_deg_ * M_PI / 180.0);
        return std::abs(plane.normal.dot(up)) < sin_thr;
    }

    bool isFloorCandidate(const PlaneModel& plane, const Eigen::Vector3d& up) const {
        const double cos_thr = std::cos(floor_max_tilt_deg_ * M_PI / 180.0);
        return std::abs(plane.normal.dot(up)) > cos_thr;
    }

    // Density of the plane's support over its box face (length x height): a real wall face is densely
    // sampled, while a sparse scatter of points along a long line fits a big plane but is not a wall.
    bool isWallBBoxCompact(const WallBBox& bbox) const {
        const Eigen::Vector3d& s = bbox.get_size();
        if (s.minCoeff() < 1e-3) return false;
        Eigen::Vector3d dims = s;
        std::sort(dims.data(), dims.data() + 3);
        return (dims[2] / dims[1]) < wall_bbox_max_aspect_ratio_;
    }

    WallBBox buildBoxFromPlane(const std::vector<Eigen::Vector3d>& inlier_pts, const PlaneModel& plane) const {
        const Eigen::Vector3d n = plane.normal.normalized();
        Eigen::Vector3d ref = Eigen::Vector3d::UnitX();
        if (std::abs(n.dot(ref)) > 0.9) ref = Eigen::Vector3d::UnitY();
        const Eigen::Vector3d right = (ref - n.dot(ref) * n).normalized();
        const Eigen::Vector3d up = n.cross(right).normalized();

        Eigen::Matrix3d R;
        R.col(0) = n; R.col(1) = right; R.col(2) = up;

        const Eigen::Vector3d origin = inlier_pts[0];
        Eigen::Vector3d local_min(1e9, 1e9, 1e9), local_max(-1e9, -1e9, -1e9);
        for (const auto& p : inlier_pts) {
            const Eigen::Vector3d local = R.transpose() * (p - origin);
            local_min = local_min.cwiseMin(local);
            local_max = local_max.cwiseMax(local);
        }

        Eigen::Vector3d size = local_max - local_min;
        size.x() = std::max(size.x(), wall_min_thickness_);

        const Eigen::Vector3d local_center = 0.5 * (local_max + local_min);
        WallBBox box;
        box.center = R * local_center + origin;
        box.size = size;
        box.rotation = R;
        return box;
    }

    // Transform a plane (normal·p + d = 0, expressed in some LOCAL frame)
    // into the frame T maps local -> global: new_normal = R*normal,
    // new_d = d - new_normal·t (R = T.linear(), t = T.translation()) — the
    // "trasforma il piano" equivalent of WallBBox::transform() for a plane
    // equation instead of 8 corners.
    // Shared by wall_points/floor_points/gseg3d's own toGlobalMsg — points
    // already in GLOBAL frame (Eigen::Vector3d) straight to PointCloud2.
    void publishGlobalPoints(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pub,
                              const std::vector<Eigen::Vector3d>& points,
                              const builtin_interfaces::msg::Time& stamp) const {
        pcl::PointCloud<pcl::PointXYZ> cloud;
        cloud.points.reserve(points.size());
        for (const auto& p : points) {
            cloud.points.emplace_back(static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z()));
        }
        cloud.width = static_cast<uint32_t>(cloud.points.size());
        cloud.height = 1;
        cloud.is_dense = true;
        sensor_msgs::msg::PointCloud2 msg;
        pcl::toROSMsg(cloud, msg);
        msg.header.frame_id = global_frame_;
        msg.header.stamp = stamp;
        pub->publish(msg);
    }

    static PlaneModel transformPlane(const PlaneModel& p, const Eigen::Isometry3d& T) {
        PlaneModel r;
        r.normal = T.linear() * p.normal;
        r.d = p.d - r.normal.dot(T.translation());
        return r;
    }

    // Runs the full plane-extraction pass on ONE source's sensor-frame
    // points (LiDAR or one camera's depth), classifies each extracted
    // plane, and returns wall boxes / the floor candidate ALREADY
    // transformed into global_frame_ — mirrors the original's
    // pointCloudCallback steps 5-6c, generalized to an arbitrary source
    // (any frame, via T_global_sensor) instead of assuming lidar_frame_.
    //
    // detect_walls/detect_floor: which categories THIS source is allowed to
    // contribute — per the user's explicit split, LiDAR only ever passes
    // detect_walls=true (walls only) and every camera only ever passes
    // detect_floor=true (floor only), see onLidarVoxelized/onCameraVoxelized.
    // A plane matching a DISABLED category is simply discarded (not
    // reclassified as the other) — RANSAC still keeps extracting up to
    // max_planes_ planes from what's left, since ransacOnePlane() already
    // strips each extracted plane's inliers from pts regardless of whether
    // this function does anything with the result.
    // Wall detection in the TOP VIEW (lidar_wall_mode "ransac2d"): a wall is a vertical plane, i.e. a LINE once the
    // points are projected on the horizontal plane (perpendicular to `up`). All the LiDAR rings of a wall collapse
    // onto that line and vote together, so a far or thin wall is found with a 2-point RANSAC instead of a 3-point
    // plane RANSAC that needs a lucky triple. What makes a line a WALL (and not a row of low furniture) is its
    // vertical extent: the inliers must span at least wall2d_min_vertical_extent_ of height. Collinear inliers are
    // split into separate walls wherever they have a gap longer than wall2d_max_gap_ (e.g. a doorway). Each wall is
    // returned as an exactly vertical plane through its points, so the box is built by the same code as the 3D path.
    // Wall normals must be (nearly) parallel or perpendicular to the robot: |yaw| of the normal in the base frame within
    // wall2d_axis_align_tol_deg_ of a multiple of 90 deg. 0 = off.
    bool isAxisAligned(const Eigen::Vector3d& normal_sensor) const {
        if (wall2d_axis_align_tol_deg_ <= 0.0) return true;
        const Eigen::Vector3d n = T_base_lidar_.block<3, 3>(0, 0) * normal_sensor;
        double yaw = std::fmod(std::abs(std::atan2(n.y(), n.x())) * 180.0 / M_PI, 90.0);
        return std::min(yaw, 90.0 - yaw) <= wall2d_axis_align_tol_deg_;
    }

    DetectionResult detectWalls2D(std::vector<Eigen::Vector3d> pts, const Eigen::Isometry3d& T_global_sensor) {
        DetectionResult result;
        const Eigen::Vector3d up = (T_global_sensor.linear().transpose() * Eigen::Vector3d::UnitZ()).normalized();
        Eigen::Vector3d e1 = Eigen::Vector3d::UnitX() - Eigen::Vector3d::UnitX().dot(up) * up;
        e1.normalize();
        const Eigen::Vector3d e2 = up.cross(e1);

        std::vector<Eigen::Vector2d> uv;  // top-view coordinates, index-aligned with pts
        std::vector<double> hz;           // height along up
        uv.reserve(pts.size());
        hz.reserve(pts.size());
        for (const auto& q : pts) {
            uv.emplace_back(q.dot(e1), q.dot(e2));
            hz.push_back(q.dot(up));
        }
        std::vector<char> alive(pts.size(), 1);
        size_t remaining = pts.size();
        const double thr = ransac_inlier_threshold_;
        int accepted = 0;
        struct Wall2D {
            std::vector<Eigen::Vector3d> pts;
            std::vector<double> t;  // coordinate of each point along dir
            Eigen::Vector2d mean, dir, nrm;
            double t_lo, t_hi;
        };
        std::vector<Wall2D> found;

        for (int round = 0; round < 3 * max_planes_ && accepted < max_planes_; ++round) {
            if (static_cast<int>(remaining) < ransac_min_inliers_) break;
            std::vector<int> idx;
            idx.reserve(remaining);
            for (size_t i = 0; i < pts.size(); ++i)
                if (alive[i]) idx.push_back(static_cast<int>(i));
            std::uniform_int_distribution<int> pick(0, static_cast<int>(idx.size()) - 1);

            int best_count = 0;
            Eigen::Vector2d best_p0, best_nrm;
            std::vector<int> local;
            for (int it = 0; it < wall2d_iterations_; ++it) {
                const Eigen::Vector2d p0 = uv[idx[pick(rng_)]];
                local.clear();
                for (int j : idx) {
                    const double d = (uv[j] - p0).norm();
                    if (d >= 0.3 && d <= wall2d_local_radius_) local.push_back(j);
                }
                if (local.empty()) continue;
                std::uniform_int_distribution<int> ld(0, static_cast<int>(local.size()) - 1);
                const Eigen::Vector2d dir = (uv[local[ld(rng_)]] - p0).normalized();
                const Eigen::Vector2d nrm(-dir.y(), dir.x());
                if (!isAxisAligned(e1 * nrm.x() + e2 * nrm.y())) continue;  // walls only parallel/perpendicular to the robot
                int count = 0;
                double hmin = std::numeric_limits<double>::max(), hmax = std::numeric_limits<double>::lowest();
                for (int j : idx) {
                    if (std::abs(nrm.dot(uv[j] - p0)) < thr) {
                        ++count;
                        hmin = std::min(hmin, hz[j]);
                        hmax = std::max(hmax, hz[j]);
                    }
                }
                if (count > best_count && hmax - hmin >= wall2d_min_vertical_extent_) {
                    best_count = count;
                    best_p0 = p0;
                    best_nrm = nrm;
                }
            }
            if (best_count < ransac_min_inliers_) break;

            // Refit the line on its inliers (2D PCA) and collect them again.
            std::vector<int> in;
            Eigen::Vector2d mean = Eigen::Vector2d::Zero();
            for (int j : idx)
                if (std::abs(best_nrm.dot(uv[j] - best_p0)) < thr) { in.push_back(j); mean += uv[j]; }
            mean /= static_cast<double>(in.size());
            Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
            for (int j : in) cov += (uv[j] - mean) * (uv[j] - mean).transpose();
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(cov);
            const Eigen::Vector2d dir = es.eigenvectors().col(1);
            const Eigen::Vector2d nrm(-dir.y(), dir.x());
            in.clear();
            for (int j : idx)
                if (std::abs(nrm.dot(uv[j] - mean)) < 1.5 * thr) in.push_back(j);
            for (int j : in) { alive[j] = 0; --remaining; }

            // Split the collinear inliers wherever there is a gap (doorway, separate walls on one line).
            std::sort(in.begin(), in.end(), [&](int a, int b) { return dir.dot(uv[a]) < dir.dot(uv[b]); });
            std::vector<std::vector<int>> segments(1);
            for (size_t k = 0; k < in.size(); ++k) {
                if (k > 0 && dir.dot(uv[in[k]]) - dir.dot(uv[in[k - 1]]) > wall2d_max_gap_) segments.emplace_back();
                segments.back().push_back(in[k]);
            }
            // Density trimming: a wall is densely sampled all along its length. Along each segment the inliers are
            // counted in bins of wall2d_bin_ metres; bins holding fewer than wall2d_min_bin_fraction_ of the segment's
            // fullest bin (and at least wall2d_min_bin_points_) are "thin" and cut the segment there, so a sparse tail
            // of scattered points (which chains together under the gap test) does not stretch the wall.
            if (wall2d_min_bin_fraction_ > 0.0) {
                std::vector<std::vector<int>> trimmed;
                for (const auto& seg : segments) {
                    if (seg.size() < 2) { trimmed.push_back(seg); continue; }
                    const double t0 = dir.dot(uv[seg.front()]);
                    const int nb = std::max(1, static_cast<int>(std::ceil((dir.dot(uv[seg.back()]) - t0) / wall2d_bin_)) + 1);
                    std::vector<int> cnt(static_cast<size_t>(nb), 0);
                    for (int j : seg) ++cnt[std::min(nb - 1, static_cast<int>((dir.dot(uv[j]) - t0) / wall2d_bin_))];
                    const int mx = *std::max_element(cnt.begin(), cnt.end());
                    const double need = std::max(static_cast<double>(wall2d_min_bin_points_), wall2d_min_bin_fraction_ * mx);
                    std::vector<int> cur;
                    int last_bin = -1;
                    for (int j : seg) {
                        const int b = std::min(nb - 1, static_cast<int>((dir.dot(uv[j]) - t0) / wall2d_bin_));
                        const bool good = cnt[static_cast<size_t>(b)] >= need;
                        if (!good || (last_bin >= 0 && b - last_bin > 1 && !cur.empty())) {
                            if (!cur.empty()) trimmed.push_back(std::move(cur));
                            cur.clear();
                        }
                        if (good) cur.push_back(j);
                        last_bin = b;
                    }
                    if (!cur.empty()) trimmed.push_back(std::move(cur));
                }
                segments = std::move(trimmed);
            }
            // Swallow the points next to a line segment (its own thickness / a near-parallel duplicate) so they cannot
            // seed another wall: leftover points within wall2d_suppress_dist_ of its line, over its span. Done for
            // rejected segments too (too short / too low), otherwise their neighbourhood becomes phantom walls.
            auto suppress = [&](double lo, double hi) {
                for (int j : idx) {
                    if (!alive[j]) continue;
                    const double t = dir.dot(uv[j]);
                    if (t >= lo - 0.5 && t <= hi + 0.5 && std::abs(nrm.dot(uv[j] - mean)) < wall2d_suppress_dist_) {
                        alive[j] = 0;
                        --remaining;
                    }
                }
            };
            for (const auto& seg : segments) {
                if (seg.empty()) continue;
                const double t_lo = dir.dot(uv[seg.front()]), t_hi = dir.dot(uv[seg.back()]);
                if (static_cast<int>(seg.size()) < ransac_min_inliers_ || t_hi - t_lo < wall2d_min_length_) {
                    suppress(t_lo, t_hi);
                    continue;
                }
                double hmin = std::numeric_limits<double>::max(), hmax = std::numeric_limits<double>::lowest();
                std::vector<Eigen::Vector3d> inl;
                inl.reserve(seg.size());
                Eigen::Vector3d c = Eigen::Vector3d::Zero();
                for (int j : seg) {
                    inl.push_back(pts[j]);
                    c += pts[j];
                    hmin = std::min(hmin, hz[j]);
                    hmax = std::max(hmax, hz[j]);
                }
                if (hmax - hmin < wall2d_min_vertical_extent_) {
                    suppress(t_lo, t_hi);
                    continue;
                }
                Wall2D w;
                w.pts = std::move(inl);
                w.t.reserve(seg.size());
                for (int j : seg) w.t.push_back(dir.dot(uv[j]));
                w.mean = mean;
                w.dir = dir;
                w.nrm = nrm;
                w.t_lo = t_lo;
                w.t_hi = t_hi;
                found.push_back(std::move(w));
                ++accepted;
                suppress(t_lo, t_hi);
            }
        }

        // Interrupt walls at their intersections: where a wall meets a (nearly) perpendicular one, a wall that runs
        // THROUGH the junction is cut there, so each piece ends at the other wall instead of crossing it. A cut is
        // made in wall i at the crossing point of wall j when the point lies inside i's span (not at its ends,
        // wall2d_split_margin_) and j actually reaches it (within wall2d_junction_reach_ of j's span).
        std::vector<std::vector<double>> cuts(found.size());
        if (wall2d_split_at_intersections_) {
            const double max_cos = std::sin(wall2d_perp_tol_deg_ * M_PI / 180.0);
            for (size_t i = 0; i < found.size(); ++i) {
                for (size_t j = 0; j < found.size(); ++j) {
                    if (i == j) continue;
                    const Wall2D &a = found[i], &b = found[j];
                    if (std::abs(a.dir.dot(b.dir)) > max_cos) continue;  // not perpendicular enough
                    Eigen::Matrix2d M;
                    M.col(0) = a.dir;
                    M.col(1) = -b.dir;
                    if (std::abs(M.determinant()) < 1e-6) continue;
                    const Eigen::Vector2d ab = M.inverse() * (b.mean - a.mean);
                    const Eigen::Vector2d X = a.mean + ab.x() * a.dir;
                    const double ta = a.dir.dot(X), tb = b.dir.dot(X);
                    if (ta > a.t_lo + wall2d_split_margin_ && ta < a.t_hi - wall2d_split_margin_ &&
                        tb > b.t_lo - wall2d_junction_reach_ && tb < b.t_hi + wall2d_junction_reach_)
                        cuts[i].push_back(ta);
                }
            }
        }
        for (size_t i = 0; i < found.size(); ++i) {
            std::sort(cuts[i].begin(), cuts[i].end());
            // Piece k spans (cut[k-1] + half_gap, cut[k] - half_gap): a small gap is left at each junction so the
            // registry (which merges coplanar faces closer than wall_plane_match_gap) does not glue the pieces back.
            std::vector<double> lo_edge{-std::numeric_limits<double>::max()}, hi_edge;
            for (double c : cuts[i]) { hi_edge.push_back(c - wall2d_split_half_gap_); lo_edge.push_back(c + wall2d_split_half_gap_); }
            hi_edge.push_back(std::numeric_limits<double>::max());
            for (size_t k = 0; k < lo_edge.size(); ++k) {
                std::vector<Eigen::Vector3d> part;
                Eigen::Vector3d c = Eigen::Vector3d::Zero();
                double lo = std::numeric_limits<double>::max(), hi = std::numeric_limits<double>::lowest();
                for (size_t q = 0; q < found[i].pts.size(); ++q) {
                    const double t = found[i].t[q];
                    if (t < lo_edge[k] || t > hi_edge[k]) continue;
                    part.push_back(found[i].pts[q]);
                    c += found[i].pts[q];
                    lo = std::min(lo, t);
                    hi = std::max(hi, t);
                }
                if (static_cast<int>(part.size()) < ransac_min_inliers_ || hi - lo < wall2d_min_length_) continue;
                c /= static_cast<double>(part.size());
                PlaneModel plane;
                plane.normal = (e1 * found[i].nrm.x() + e2 * found[i].nrm.y()).normalized();
                plane.d = -plane.normal.dot(c);
                WallBBox box = buildBoxFromPlane(part, plane);
                if (!isWallBBoxCompact(box)) continue;
                box.transform(T_global_sensor);
                result.wall_boxes_global.push_back(box);
                for (const auto& q : part) result.wall_points_global.push_back(T_global_sensor * q);
            }
        }
        return result;
    }

    DetectionResult detectFromSensorPoints(std::vector<Eigen::Vector3d> pts, const Eigen::Isometry3d& T_global_sensor,
                                            bool detect_walls, bool detect_floor) {
        const Eigen::Vector3d up_sensor = T_global_sensor.linear().transpose() * Eigen::Vector3d::UnitZ();
        DetectionResult result;

        for (int p = 0; p < max_planes_; ++p) {
            if (static_cast<int>(pts.size()) < ransac_min_inliers_) break;
            PlaneModel plane;
            std::vector<Eigen::Vector3d> inliers;
            if (!ransacOnePlane(pts, plane, inliers)) break;

            if (detect_walls && isWallPlane(plane, up_sensor)) {
                WallBBox box = buildBoxFromPlane(inliers, plane);
                if (!isWallBBoxCompact(box)) continue;
                box.transform(T_global_sensor);
                result.wall_boxes_global.push_back(box);
                // Accumulate this plane's own inliers too (global frame) —
                // see DetectionResult::wall_points_global's own comment.
                result.wall_points_global.reserve(result.wall_points_global.size() + inliers.size());
                for (const auto& p : inliers) result.wall_points_global.push_back(T_global_sensor * p);
            } else if (detect_floor && !result.floor_found && isFloorCandidate(plane, up_sensor)) {
                // Orient normal toward "up" before transforming, same
                // convention preprocessing_node's own floor fit used —
                // signedDistance() downstream (there, not here — this node
                // only ever publishes a plane, never filters with one)
                // relies on "positive = above the floor".
                if (plane.normal.dot(up_sensor) < 0.0) {
                    plane.normal = -plane.normal;
                    plane.d = -plane.d;
                }
                result.floor_plane_global = transformPlane(plane, T_global_sensor);
                result.floor_found = true;
                result.floor_points_global.reserve(inliers.size());
                for (const auto& p : inliers) result.floor_points_global.push_back(T_global_sensor * p);
            }
        }
        return result;
    }

    // Folds one source's DetectionResult into the shared registry (walls,
    // always) and — when floor_source is non-null — that ONE camera's own
    // floor state (never any other camera's, never LiDAR's, since LiDAR has
    // none — see the ctor comment on that). Note the SAME quirk the
    // original has: bbox_registry_->update() (and therefore missed_frames/
    // expiry bookkeeping) is only invoked when this callback found at least
    // one wall — a totally-empty detection doesn't advance expiry either.
    // Kept faithfully rather than "fixed", per this being a logic-preserving
    // port.
    void mergeDetection(const DetectionResult& det, CameraSource* floor_source) {
        if (!det.wall_boxes_global.empty()) {
            const auto merged = wall_registry_->mergeNested(det.wall_boxes_global, nested_merge_iov_thresh_);
            wall_registry_->update(merged, Eigen::Isometry3d::Identity());
        }
        if (det.floor_found && floor_source != nullptr) {
            PlaneModel& floor = floor_source->floor_plane_global;
            if (!floor_source->has_floor_estimate) {
                floor = det.floor_plane_global;
                floor_source->has_floor_estimate = true;
            } else {
                Eigen::Vector3d blended =
                    (1.0 - floor_ema_alpha_) * floor.normal + floor_ema_alpha_ * det.floor_plane_global.normal;
                if (blended.norm() > 1e-6) blended.normalize();
                floor.normal = blended;
                floor.d = (1.0 - floor_ema_alpha_) * floor.d + floor_ema_alpha_ * det.floor_plane_global.d;
            }
        }
        publishMarkers();
    }

    void onLidarVoxelized(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) {
        if (!has_odom_ || !ensureStaticTf()) return;

        pcl::PointCloud<pcl::PointXYZ> cloud;
        pcl::fromROSMsg(*msg, cloud);

        std::vector<Eigen::Vector3d> pts;
        pts.reserve(cloud.points.size());
        for (const auto& pt : cloud.points) {
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
            if (std::abs(pt.x) > lidar_range_x_ || std::abs(pt.y) > lidar_range_y_) continue;
            pts.emplace_back(pt.x, pt.y, pt.z);
        }
        if (pts.size() < 3) return;

        const Eigen::Isometry3d T_global_lidar = toIsometry(T_global_base_latest_ * T_base_lidar_);
        // LiDAR: walls only — see the user's explicit split (cameras handle
        // ground/floor instead, onCameraVoxelized below). The LiDAR's own
        // 360°, consistent-height scan is the better source for vertical
        // wall planes across the whole range; it never contributes to any
        // floor state (floor_source=nullptr — there's no LiDAR-owned floor
        // plane to update).
        // GSeg3D on LiDAR too, per explicit request — same function as the
        // camera path (see runGseg3d's own comment): LiDAR's 360° coverage
        // reaches ground the cameras' limited FOV never sees at all (e.g.
        // behind the robot, with only front_camera/back_camera
        // configured). Needs pts BEFORE detectFromSensorPoints below moves
        // them out from under it (std::move) — same ordering as
        // onCameraVoxelized's own call.
        runGseg3d(pts, T_global_lidar, msg->header.stamp, pub_gseg3d_lidar_ground_, pub_gseg3d_lidar_non_ground_);

        DetectionResult det = (lidar_wall_mode_ == "ransac2d")
            ? detectWalls2D(std::move(pts), T_global_lidar)
            : detectFromSensorPoints(std::move(pts), T_global_lidar, /*detect_walls=*/true, /*detect_floor=*/false);
        publishGlobalPoints(pub_wall_points_, det.wall_points_global, msg->header.stamp);
        mergeDetection(det, /*floor_source=*/nullptr);
    }

    void onCameraVoxelized(CameraSource& cam, const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) {
        if (!has_odom_ || !ensureStaticTf()) return;
        if (camera_process_hz_ > 0.0) {
            const double t = rclcpp::Time(msg->header.stamp).seconds();
            if (t >= cam.last_processed_stamp && t - cam.last_processed_stamp < 0.9 / camera_process_hz_) return;
            cam.last_processed_stamp = t;
        }

        // calibration_enabled_ picks which transform places this camera's
        // points in global_frame_ — see calibration_enabled_'s own comment
        // for why this exists at all (ICP calibration made optional).
        Eigen::Isometry3d T_global_cam;
        if (calibration_enabled_) {
            if (!ensureCameraCalibrationTf(cam)) return;
            // Same reasoning as calibration_icp_node.cpp's own native-cloud
            // gate: starting to fit/publish a floor plane before this
            // camera's depth<->color correction has arrived would place it
            // off by that factory extrinsic (a few cm) — wait rather than
            // publish a knowingly-wrong plane. transient_local (see
            // setupCamera()) makes this resolve near-instantly in practice,
            // not a real stall.
            if (cam.is_native_depth_frame && !cam.has_depth_color_matrix) return;

            // cam.T_lidar_camera (-> refined_camera_frame) was calibrated
            // against the DEPROJECTED (color-optical-frame) convention —
            // see CameraSource::is_native_depth_frame's own comment. When
            // this camera's voxelized_depth_topic is the native cloud
            // instead (still in depth-optical-frame), compose the same
            // correction preprocessing_node.cpp's finishDepthCloud applies,
            // so `pts` below (untouched, still raw sensor-frame) get
            // interpreted through the matching "lidar <- depth-optical"
            // transform instead of the wrong "lidar <- color-optical" one —
            // otherwise every point (and the floor plane fit from them)
            // would sit off by that factory extrinsic's translation,
            // visually detached from anything actually placed via the
            // correct convention (e.g. the deprojected path, or any other
            // camera).
            Eigen::Matrix4d T_lidar_frame = cam.T_lidar_camera;
            if (cam.is_native_depth_frame) {
                T_lidar_frame = Eigen::Matrix4d(T_lidar_frame * cam.T_color_depth);
            }
            T_global_cam = toIsometry(T_global_base_latest_ * T_base_lidar_ * T_lidar_frame);
        } else {
            // Calibration disabled: trust the robot's own raw TF chain
            // directly (base_frame_ -> whichever frame this cloud is
            // actually in) instead of routing through lidar_frame_/
            // T_lidar_camera at all — see ensureBaseCameraTf()'s own
            // comment. No depth<->color correction needed either way: this
            // looks up whatever frame msg->header.frame_id actually names,
            // native or deprojected, directly — that ambiguity only ever
            // existed because T_lidar_camera was calibrated against ONE
            // specific convention (color-optical).
            if (!ensureBaseCameraTf(cam, msg->header.frame_id)) return;
            T_global_cam = toIsometry(T_global_base_latest_ * cam.T_base_camera);
        }

        pcl::PointCloud<pcl::PointXYZ> cloud;
        pcl::fromROSMsg(*msg, cloud);

        // No extra range crop here — the depth cloud is already bounded by
        // preprocessing_node's own depth_min_value/depth_max_value upstream.
        std::vector<Eigen::Vector3d> pts;
        pts.reserve(cloud.points.size());
        for (const auto& pt : cloud.points) {
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
            pts.emplace_back(pt.x, pt.y, pt.z);
        }
        if (pts.size() < 3) return;

        // Same input points, correct/calibrated path — see
        // CameraSource::pub_depth_calibrated's own comment. Full
        // T_global_cam (rotation AND translation, unlike GSeg3D's
        // rotation-only pre-step), points already end up in global_frame_
        // directly.
        {
            std::vector<Eigen::Vector3d> global_pts;
            global_pts.reserve(pts.size());
            for (const auto& p : pts) global_pts.push_back(T_global_cam * p);
            publishGlobalPoints(cam.pub_depth_calibrated, global_pts, msg->header.stamp);
        }

        // Cameras: floor/ground only — see the user's explicit split (LiDAR
        // handles walls instead, onLidarVoxelized above). A downward-angled
        // depth camera sees the floor immediately in front of the robot at
        // much higher density than the LiDAR's own sparse near-field rings,
        // which is what actually matters for catching a slope early; it no
        // longer contributes wall boxes to wall_registry_ at all.
        //
        // floor_source=&cam: per explicit request, floor is now ONE plane
        // PER CAMERA, not one shared/merged estimate across all of them —
        // this camera's detection only ever updates its OWN
        // cam.floor_plane_global (see mergeDetection/CameraSource), never
        // blended with any other camera's. The union of the space below
        // every camera's own plane is what preprocessing_node treats as
        // "the ground" for its crop — see that file's onWallMarkers()/
        // floorHeightAboveGround().
        // GSeg3D needs `pts` too (below) — detectFromSensorPoints takes it
        // by value (std::move), so run GSeg3D first; it only reads pts, it
        // doesn't consume/move them.
        runGseg3d(pts, T_global_cam, msg->header.stamp, cam.pub_gseg3d_ground, cam.pub_gseg3d_non_ground);

        DetectionResult det = detectFromSensorPoints(std::move(pts), T_global_cam, /*detect_walls=*/false, /*detect_floor=*/true);
        publishGlobalPoints(cam.pub_floor_points, det.floor_points_global, msg->header.stamp);
        mergeDetection(det, /*floor_source=*/&cam);
    }

    // GSeg3D ground estimate — see file header's own section. Independent
    // debug output, not wired into floor_plane_global/mergeDetection.
    // Generic over source — CAMERA (onCameraVoxelized) or LIDAR
    // (onLidarVoxelized both call this now; LiDAR's own 360° coverage
    // reaches ground the cameras' limited FOV never sees at all, e.g.
    // behind the robot with only front_camera/back_camera configured).
    // Publishers passed in rather than read off a CameraSource: the ONLY
    // per-source-type difference here is which pair of publishers gets the
    // result — everything else (rotation-align, distToGround seed, the
    // grid itself) is identical regardless of what sensor pts came from.
    void runGseg3d(const std::vector<Eigen::Vector3d>& pts, const Eigen::Isometry3d& T_global_sensor,
                    const builtin_interfaces::msg::Time& stamp,
                    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pub_ground,
                    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pub_non_ground) {
        if (!gseg3d_enabled_ || pts.size() < 3) return;

        // Rotate (NOT translate) into a gravity-aligned-but-sensor-centered
        // frame: GSeg3D bins points into its grid along their raw X/Y/Z, so
        // cellSizeZ only means "vertical" if the input's own Z axis already
        // is — unlike detectFromSensorPoints's RANSAC, which never rotates
        // the points themselves (only a reference "up" vector, see
        // up_sensor), GSeg3D needs this done up front. See file header.
        const Eigen::Matrix3d R = T_global_sensor.linear();
        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());
        aligned->points.reserve(pts.size());
        for (const auto& p : pts) {
            const Eigen::Vector3d rp = R * p;
            aligned->points.emplace_back(
                static_cast<float>(rp.x()), static_cast<float>(rp.y()), static_cast<float>(rp.z()));
        }
        aligned->width = static_cast<uint32_t>(aligned->points.size());
        aligned->height = 1;
        aligned->is_dense = true;

        // Ground's height relative to the SENSOR's own origin, in these
        // gravity-aligned axes — just a seed for the region-growing
        // expansion below (see ground_detection.hpp's getGroundCells()),
        // doesn't need to be exact. ground_height_ is the same bootstrap
        // global-frame floor Z every camera's own RANSAC path also starts
        // from (see CameraSource ctor) — NOT any source's current
        // floor_plane_global estimate, deliberately: this stays fully
        // independent of the RANSAC path, see file header.
        gseg3d_config_.distToGround = ground_height_ - T_global_sensor.translation().z();

        ground_segmentation::PointCloudGrid<pcl::PointXYZ> grid(gseg3d_config_);
        grid.setInputCloud(aligned, Eigen::Quaterniond::Identity());
        auto [ground, non_ground] = grid.segmentPoints();

        // Add the translation back (rotation-only above was only so the
        // grid's own axes lined up with gravity) to place these in
        // global_frame_ properly, then publish there directly — same frame
        // every other display in preprocessing_debug.rviz already uses, no
        // extra TF needed to view them.
        auto toGlobalMsg = [&](const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
            pcl::PointCloud<pcl::PointXYZ> global_cloud;
            global_cloud.points.reserve(cloud->points.size());
            for (const auto& p : cloud->points) {
                const Eigen::Vector3d gp =
                    Eigen::Vector3d(p.x, p.y, p.z) + T_global_sensor.translation();
                global_cloud.points.emplace_back(
                    static_cast<float>(gp.x()), static_cast<float>(gp.y()), static_cast<float>(gp.z()));
            }
            global_cloud.width = static_cast<uint32_t>(global_cloud.points.size());
            global_cloud.height = 1;
            global_cloud.is_dense = true;
            sensor_msgs::msg::PointCloud2 msg;
            pcl::toROSMsg(global_cloud, msg);
            msg.header.frame_id = global_frame_;
            msg.header.stamp = stamp;
            return msg;
        };

        pub_ground->publish(toGlobalMsg(ground));
        pub_non_ground->publish(toGlobalMsg(non_ground));
    }

    // ns="wall_bbox": one CUBE per tracked wall — center/rotation/size,
    // exactly the original's wire format (dynamicDetector's wallMarkersCB
    // parses these same three fields back into an OBB; preprocessing_node's
    // onWallMarkers() does the same, see that file). ns="floor": one CUBE
    // slab PER CAMERA (id = that camera's index in cameras_), each encoding
    // that camera's OWN cam.floor_plane_global — orientation's local +Z
    // axis IS the plane normal (Eigen::Quaterniond::FromTwoVectors(UnitZ,
    // normal)), position is any point ON the plane (solved at the robot's
    // current (x,y) from the plane equation) — losslessly recoverable by a
    // subscriber, no separate message type needed. preprocessing_node's
    // onWallMarkers() collects ALL of these (not just one) into its own
    // floor_planes_global_ — see that file.
    // Junction splitting of the TRACKED walls (done on what is published, so it is stable: the registry keeps both
    // walls even when a scan misses the crossing one). Where two (nearly) perpendicular walls cross, a wall that runs
    // through the junction is cut there — pieces end wall2d_split_half_gap_ before the crossing wall's plane — so each
    // wall stops at the other instead of passing through it. Ends closer than wall2d_split_margin_ are not cut.
    std::vector<WallBBox> splitAtJunctions(const std::vector<WallBBox>& in) const {
        if (!wall_split_at_junctions_) return in;
        const double max_cos = std::sin(wall2d_perp_tol_deg_ * M_PI / 180.0);
        auto hAxis = [](const WallBBox& b) {  // in-plane axis that is horizontal (col 1 or 2), -1 if none
            for (int a = 1; a <= 2; ++a)
                if (std::abs(b.rotation.col(a).z()) < 0.5) return a;
            return -1;
        };
        std::vector<WallBBox> out;
        for (size_t i = 0; i < in.size(); ++i) {
            const WallBBox& a = in[i];
            const int ha = hAxis(a);
            if (ha < 0) { out.push_back(a); continue; }
            const Eigen::Vector2d da = a.rotation.col(ha).head<2>().normalized(), ca = a.center.head<2>();
            const double he = a.size[ha] * 0.5;
            std::vector<double> cuts;
            for (size_t j = 0; j < in.size(); ++j) {
                if (i == j) continue;
                const WallBBox& b = in[j];
                const int hb = hAxis(b);
                if (hb < 0) continue;
                const Eigen::Vector2d db = b.rotation.col(hb).head<2>().normalized(), cb = b.center.head<2>();
                if (std::abs(da.dot(db)) > max_cos) continue;
                Eigen::Matrix2d M;
                M.col(0) = da;
                M.col(1) = -db;
                if (std::abs(M.determinant()) < 1e-6) continue;
                const Eigen::Vector2d ab = M.inverse() * (cb - ca);
                const double ta = ab.x(), tb = ab.y();  // position of the crossing along each wall, from its centre
                const double heb = b.size[hb] * 0.5;
                if (std::abs(ta) < he - wall2d_split_margin_ && std::abs(tb) < heb + wall2d_junction_reach_) cuts.push_back(ta);
            }
            if (cuts.empty()) { out.push_back(a); continue; }
            std::sort(cuts.begin(), cuts.end());
            std::vector<double> lo{-he}, hi;
            for (double c : cuts) { hi.push_back(c - wall2d_split_half_gap_); lo.push_back(c + wall2d_split_half_gap_); }
            hi.push_back(he);
            for (size_t k = 0; k < lo.size(); ++k) {
                if (hi[k] - lo[k] < wall2d_min_length_) continue;
                WallBBox piece = a;
                piece.size[ha] = hi[k] - lo[k];
                piece.center = a.center + a.rotation.col(ha) * (0.5 * (lo[k] + hi[k]));
                out.push_back(piece);
            }
        }
        return out;
    }

    void publishMarkers() {
        visualization_msgs::msg::MarkerArray markers;
        const auto stamp = this->now();

        visualization_msgs::msg::Marker del;
        del.header.frame_id = global_frame_;
        del.header.stamp = stamp;
        del.action = visualization_msgs::msg::Marker::DELETEALL;
        markers.markers.push_back(del);

        const std::vector<WallBBox> tracked = splitAtJunctions(wall_registry_->bboxes());
        for (size_t i = 0; i < tracked.size(); ++i) {
            const auto& box = tracked[i];
            visualization_msgs::msg::Marker m;
            m.header.frame_id = global_frame_;
            m.header.stamp = stamp;
            m.ns = "wall_bbox";
            m.id = static_cast<int>(i);
            m.type = visualization_msgs::msg::Marker::CUBE;
            m.action = visualization_msgs::msg::Marker::ADD;
            m.lifetime = rclcpp::Duration::from_seconds(marker_lifetime_sec_);
            m.pose.position.x = box.center.x();
            m.pose.position.y = box.center.y();
            m.pose.position.z = box.center.z();
            const Eigen::Quaterniond q(box.rotation);
            m.pose.orientation.x = q.x(); m.pose.orientation.y = q.y();
            m.pose.orientation.z = q.z(); m.pose.orientation.w = q.w();
            m.scale.x = box.size.x(); m.scale.y = box.size.y(); m.scale.z = box.size.z();
            m.color.r = 0.9f; m.color.g = 0.2f; m.color.b = 0.2f; m.color.a = 0.5f;
            markers.markers.push_back(m);
        }

        // Small fixed hue cycle so N cameras' floor slabs stay visually
        // distinguishable in rviz even when they overlap (not just a single
        // color as before, now that there's genuinely more than one plane).
        static constexpr float kFloorColors[][3] = {
            {0.20f, 0.60f, 0.90f},  // blue
            {0.20f, 0.85f, 0.45f},  // green
            {0.90f, 0.60f, 0.15f},  // orange
            {0.75f, 0.30f, 0.85f},  // purple
        };
        const Eigen::Vector3d robot_xy = T_global_base_latest_.block<3, 1>(0, 3);
        for (size_t i = 0; i < cameras_.size(); ++i) {
            const auto& cam = *cameras_[i];
            if (!cam.has_floor_estimate) continue;  // nothing to show yet for this camera

            visualization_msgs::msg::Marker floor;
            floor.header.frame_id = global_frame_;
            floor.header.stamp = stamp;
            floor.ns = "floor";
            floor.id = static_cast<int>(i);
            floor.type = visualization_msgs::msg::Marker::CUBE;
            floor.action = visualization_msgs::msg::Marker::ADD;
            floor.lifetime = rclcpp::Duration::from_seconds(marker_lifetime_sec_);

            const auto& n = cam.floor_plane_global.normal;
            const double floor_z = std::abs(n.z()) > 1e-6
                ? -(cam.floor_plane_global.d + n.x() * robot_xy.x() + n.y() * robot_xy.y()) / n.z()
                : robot_xy.z();
            floor.pose.position.x = robot_xy.x();
            floor.pose.position.y = robot_xy.y();
            floor.pose.position.z = floor_z;
            const Eigen::Quaterniond floor_q = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), n);
            floor.pose.orientation.x = floor_q.x(); floor.pose.orientation.y = floor_q.y();
            floor.pose.orientation.z = floor_q.z(); floor.pose.orientation.w = floor_q.w();
            floor.scale.x = 2.0 * lidar_range_x_;
            floor.scale.y = 2.0 * lidar_range_y_;
            // 0.02m used to be enough from the default top-down view (a filled rectangle reads fine edge-on
            // or not), but from a LATERAL view a 2cm slab at this alpha all but disappears — thickened so the
            // floor stays legible from the side too, still thin enough not to look like a real physical slab.
            floor.scale.z = 0.08;
            const auto& color = kFloorColors[i % (sizeof(kFloorColors) / sizeof(kFloorColors[0]))];
            floor.color.r = color[0]; floor.color.g = color[1]; floor.color.b = color[2]; floor.color.a = 0.55f;
            markers.markers.push_back(floor);
        }

        pub_wall_markers_->publish(markers);
    }

    // ---- Params ----
    std::string lidar_frame_, base_frame_, global_frame_, odom_topic_, lidar_voxelized_topic_;
    bool calibration_enabled_{true};
    double lidar_range_x_{15.0}, lidar_range_y_{15.0};
    int max_planes_{6};
    int ransac_max_iterations_{80};
    double ransac_inlier_threshold_{0.07};
    double wall_min_thickness_{0.20};
    int ransac_min_inliers_{25};
    double ransac_confidence_{0.99};
    double wall_vertical_angle_deg_{5.0};
    double wall_bbox_max_aspect_ratio_{5.0};
    double ransac_local_radius_{0.0};
    double marker_lifetime_sec_{0.5};
    std::string lidar_wall_mode_{"ransac3d"};
    int wall2d_iterations_{150};
    double wall2d_local_radius_{1.5}, wall2d_min_vertical_extent_{1.0}, wall2d_max_gap_{1.5}, wall2d_min_length_{0.0}, wall2d_suppress_dist_{0.0}, wall2d_bin_{0.5}, wall2d_min_bin_fraction_{0.0};
    int wall2d_min_bin_points_{2};
    double wall2d_axis_align_tol_deg_{0.0};
    bool wall2d_split_at_intersections_{false};
    bool wall_split_at_junctions_{false};
    double wall2d_perp_tol_deg_{20.0}, wall2d_split_margin_{0.5}, wall2d_split_half_gap_{0.3}, wall2d_junction_reach_{1.0};
    double nested_merge_iov_thresh_{0.5};
    double floor_max_tilt_deg_{25.0};
    double floor_ema_alpha_{0.1};
    double ground_height_{-0.3};

    // GSeg3D ground estimate — see file header + runGseg3d(). One shared
    // config (constructor fills it from gseg3d_* params); a fresh
    // ground_segmentation::PointCloudGrid is constructed from it PER CALL
    // in runGseg3d() rather than kept as per-camera state, since it fully
    // clears/rebuilds its grid on every setInputCloud() anyway — cheaper to
    // reason about than adding yet another piece of per-camera mutable
    // state to CameraSource.
    bool gseg3d_enabled_{true};
    double camera_process_hz_{0.0};
    ground_segmentation::GridConfig gseg3d_config_;

    std::vector<std::string> camera_names_;
    std::vector<std::shared_ptr<CameraSource>> cameras_;

    // ---- ROS I/O ----
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_lidar_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_wall_markers_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_wall_points_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_gseg3d_lidar_ground_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_gseg3d_lidar_non_ground_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // ---- State ----
    bool has_static_tf_{false};
    Eigen::Matrix4d T_base_lidar_{Eigen::Matrix4d::Identity()};
    bool has_odom_{false};
    Eigen::Matrix4d T_global_base_latest_{Eigen::Matrix4d::Identity()};

    // Floor state lives per-camera now (CameraSource::floor_plane_global/
    // has_floor_estimate — see that struct's own comment), NOT here — LiDAR
    // never contributes floor at all (detect_floor=false, see
    // onLidarVoxelized), so there's no "shared" floor state to keep.
    WallBBoxRegistry::Ptr wall_registry_;

    std::mt19937 rng_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<StaticStructuresNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
