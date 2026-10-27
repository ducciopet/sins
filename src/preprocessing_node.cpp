/*
    FILE: preprocessing_node.cpp
    ---------------------------------
    Phase 1 of the jo_detector rewrite (see ../README.md).

    Sequential (no threading — see the discussion in the chat this came out
    of: at VLP-16 scan sizes the single-threaded cost is small enough that
    thread spawn/join overhead isn't obviously worth it; revisit with
    ros2 topic bw/hz + std::chrono numbers before parallelizing).

    Wall/floor DETECTION does not happen in this file anymore — it moved to
    its own standalone, asynchronous node (static_structures_node.cpp), exactly
    mirroring jo_detector's own architecture (dynamicDetector doesn't
    run wall RANSAC either — it consumes static_structures_node's output over
    ROS topics). This file's LiDAR pipeline is now:
      1. sensor-frame validity + range crop + distance-weighted subsample
         -> local_pts (the general-purpose cloud's own thinning, unrelated
         to detection).
      2. Separately, straight off the raw scan (no crop of any kind, not
         even step 1's range box): voxel-downsample and publish on
         pub_lidar_voxelized_ (lidar_frame_, i.e. sensor frame) — this is
         the ONLY point static_structures_node's own LiDAR subscription reads
         from; it does its own range crop on these already-downsampled
         centroids (cheap) rather than each node repeating the full
         acquire+voxelize step independently. See that file's own header
         comment for the full reasoning and why this specific split
         ("voxelize once here, crop separately per consumer") is what
         actually fixes the wall-thickness bug this design replaced.
      3. local_pts -> global frame + floor/roof crop, using
         heightAboveGround() (bounded to [wall_removal_margin_,
         ground_roof_offset_]) — the MINIMUM signed distance across every
         camera's own, possibly-tilted ax+by+cz+d=0 plane, not a single
         shared plane and not a flat Z-band: per explicit request, floor
         detection is one plane PER CAMERA (see static_structures_node.cpp's own
         header), and the ground is the UNION of the space below each one.
         None of these planes are fit here either — they arrive
         asynchronously via onWallMarkers(), parsed back out of
         static_structures_node's own MarkerArray publish (one "floor" marker
         per camera, see that method). Until the first message arrives,
         this stays at its single flat ground_height_ bootstrap.
      4. voxel downsample (3)'s output — the published general-purpose
         cloud's basis.
      5. strip any point within wall_removal_margin_ of a wall_planes_global_
         entry (also parsed from onWallMarkers(), the TRACKED set
         static_structures_node's own WallBBoxRegistry maintains — not
         necessarily this exact scan's fresh detections, since that
         update is async and on its own cadence), then publish. Combined
         with step 3's slope-aware crop, this is "remove points inside the
         floor and walls" for LiDAR.

    The global-frame transform itself is also cheap by construction: the
    rigid base_frame->lidar_frame offset never changes, so it's looked up
    over TF once (ensureStaticTf()) and cached; every callback then only
    composes it with the pose already carried in the synced odometry
    message (T_global_base * T_base_lidar) — one 4x4 matrix multiply per
    callback, not a TF buffer lookup/interpolation per callback. This
    mirrors dynamicDetector::ensureStaticTfs()/getLidarPose() in
    jo_detector, which does the same thing for the same reason.

    Depth side: an arbitrary number of cameras (camera_names param — this
    robot has two, front_camera and back_camera, but nothing here assumes a
    fixed count), each an independent CameraStream with its own three
    streams, each on its own camera_info (own intrinsics, never mixed up
    within a camera, and never shared across cameras):
      - aligned_depth_topic (+ aligned_depth_camera_info_topic, which per
        jo_navigation/config/visual_odom.yaml is the COLOR camera's own
        camera_info — aligned depth lives in the color image's pixel grid,
        so it shares the color intrinsics, not the depth sensor's own) is
        deprojected straight into a point cloud: the RealSense driver has
        already done the depth<->color registration for us, so this is the
        "solve the projection problem by not doing it ourselves" path.
        See processAlignedDepth().

        OR, if <name>.aligned_depth_cloud_topic is set (empty/unset by
        default — see setupCamera()), that whole deprojection is skipped
        entirely: some drivers (e.g. RealSense's own "pointcloud" filter,
        <name>/camera/depth/color/points in bags/*_validation_lab) already
        publish a deprojected PointCloud2 themselves, and there's no reason
        to decode an image and reconstruct one ourselves when the input is
        already a point cloud — see onNativeDepthCloud(). The one real
        difference: that native cloud arrives in THIS camera's own
        depth_optical_frame (its own intrinsics), not color_optical_frame
        like the aligned-depth path above — finishDepthCloud() accounts for
        that when composing the calibrated global-frame transform (see its
        own comment on is_native_depth_frame and cam.T_color_depth).

        Either way, the result is published as-is (pub_depth_cloud,
        "original") and, separately, voxel-downsampled with floor/wall
        points stripped (pub_depth_processed, mirroring the LiDAR path) —
        see finishDepthCloud(), shared by both paths. pub_depth_cloud and
        pub_depth_voxelized stay in whichever frame the source actually
        reported (color-optical for the deprojected path, depth-optical for
        the native one) — the floor/wall crop only needs a global-frame
        *copy* of each point to decide what to keep, so there's no reason to
        transform the whole cloud just for those two. pub_depth_processed,
        the one meant to feed the (not yet built) detection phase, is
        different: it's published in global_frame_ (every point actually
        transformed, not just a copy used for the crop test), matching
        pub_lidar_processed_'s own convention — a detector consuming both
        this and the LiDAR cloud needs them in the same frame without
        re-deriving each camera's calibration itself. That global-frame
        transform prefers going through THIS camera's own calibration_icp_node output
        (lidar_frame_ -> <name>_refined, one instance per camera_names_
        entry — see run_detector.launch.py and ensureCameraCalibrationTf()):
        T_global_base_latest_ * T_base_lidar_ * cam.T_lidar_camera. That's
        the actual lidar<->depth alignment this pipeline calibrates, so
        it's what should place each camera's depth points in the same
        global frame the LiDAR cloud and wall/floor planes already live in
        — using the robot's own raw URDF/bag TF chain instead (the
        original design) skips that calibration entirely, which is wrong
        whenever the two disagree. cam.T_base_depth (cached once per
        camera, rigid mount — ensureBaseDepthTf()) is kept only as the
        fallback used before that camera's calibration TF is available.
        T_global_base_latest_ itself is the LiDAR path's most recently
        synced odom pose (see that member's comment for why not a TF
        lookup to global_frame_, which was tried first and needs
        something, e.g. ekf_node, reliably broadcasting
        global_frame_->base_frame_ over TF specifically — empirically NOT
        reliably available even when /odometry/filtered itself is
        publishing fine).
      - depth_topic (the sensor's own, unaligned depth) is only subscribed
        and decoded, not deprojected here — it's for the future UV-map
        detection stage (phase 2), which works on the depth image directly.
      - <name>.ground_mask_topic (empty/unset by default, same convention as
        aligned_depth_cloud_topic): when set, subscribes to a mono8 0/255
        image (yolo_semantic_seg_detector.py's segmentation_mask, run over
        this SAME camera's color image) and publishes semantic_ground_points:
        the depth points that mask marks 255, transformed into global_frame_
        via the exact same computeGlobalDepthTransform() this camera's
        depth_processed output already uses. Two ways to get the labels,
        one per depth path:
          * aligned-depth IMAGE (processAlignedDepth): aligned depth lives
            in the color pixel grid, so the mask is indexed pixel by pixel.
          * native point cloud (onNativeDepthCloud): unorganized, no pixel
            grid — so each point is PROJECTED onto the color image instead:
            depth frame -> color frame through T_color_depth (the color
            camera sits ~5.9 cm off the depth one, so ignoring it would
            shift every pixel), then the color camera's pinhole intrinsics
            (camera_info), then the mask is read at that pixel.
        Best-effort pairing, not hard-synced: a depth frame is masked
        against whatever segmentation_mask arrived most recently (the mask
        arrives on its own async Python-node cadence and can drop frames
        under load — see that node's own header). Fine for a
        debug/visualization output.
      - depth_to_color_extrinsics_topic is the RealSense driver's own
        factory-calibrated depth<->color extrinsic (distinct from our own
        lidar<->camera ICP calibration). Rather than keep it as a private
        cached matrix only this node knows about, it's rebroadcast once as
        a static TF (depth_optical_frame -> color_optical_frame) so any
        node can relate the two frames the normal way, the same as the
        driver's own TF usually already does — this is the explicit,
        defensive version of that in case it isn't published for a given
        setup. See the file-level NOTE below on the one part of this that's
        unverified against the real message definition.

    Each camera's aligned-depth cloud, before its own floor/wall crop, is
    ALSO voxel-downsampled and republished as-is on pub_depth_voxelized
    (see processAlignedDepth) — the depth-side equivalent of
    pub_lidar_voxelized_ above, zero extra compute since that voxel grid
    is already computed for this camera's own crop anyway. This is what
    lets static_structures_node also detect off every camera's depth cloud
    (more viewpoints = more evidence — a rear camera seeing a wall the
    LiDAR's own crop/subsample missed), not just LiDAR, unlike
    jo_detector/wall_detector_node's own precedent (depth contributed
    only a simple bottom-rows ground estimate there, never full wall
    RANSAC).
*/
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/header.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <fstream>
#include <map>
#include <utility>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2/exceptions.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

// NOTE (build risk, flagged explicitly): this is the one part of this file
// not cross-checked against real code in this repo (unlike everything
// else here, which mirrors an existing, working pattern). The exact
// package/message name for the RealSense depth<->color extrinsics topic,
// and whether its rotation matrix is row-major or column-major (the
// librealsense SDK's own rs2_extrinsics struct is column-major, which is
// what onDepthToColorExtrinsics() below assumes) needs verifying against
// what's actually installed. If the build fails on this include or a
// camera's rebroadcast TF looks wrong (mirrored/transposed), this is the
// first place to check.
#include <realsense2_camera_msgs/msg/extrinsics.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <deque>
#include <vector>
#include <memory>

class PreprocessingNode : public rclcpp::Node {
public:
    PreprocessingNode() : Node("preprocessing_node"), rng_(std::random_device{}()) {
        lidar_topic_ = this->declare_parameter("lidar_topic", std::string("/velodyne_points"));
        odom_topic_ = this->declare_parameter("odom_topic", std::string("/odometry/filtered"));

        lidar_frame_ = this->declare_parameter("lidar_frame", std::string("velodyne"));
        base_frame_ = this->declare_parameter("base_frame", std::string("base_link"));
        global_frame_ = this->declare_parameter("global_frame", std::string("odom"));
        // No calibration gate on the LiDAR path anymore — matches
        // jo_detector/dynamicDetector, which never gated its own
        // lidarOdomCB on calibration either (only static_structures_node's own
        // per-camera detection needs each camera's calibration TF — see
        // ensureCameraCalibrationTf below, still used for the depth crop's
        // own global-frame placement, a separate concern).

        // false = every camera's depth crop skips ensureCameraCalibrationTf
        // entirely and always uses ensureBaseDepthTf's raw URDF/bag TF
        // chain instead — for when that chain already places the depth
        // cloud close enough to the LiDAR one that running ICP calibration
        // isn't worth it. See finishDepthCloud()'s own use of this.
        // Forwarded from run_detector.launch.py's enable_calibration arg —
        // same single source of truth static_structures_node.cpp's own
        // calibration_enabled_ uses, and whether calibration_icp_node even
        // gets launched in the first place.
        calibration_enabled_ = this->declare_parameter("calibration_enabled", true);

        depth_scale_ = this->declare_parameter("depth_scale_factor", 1000.0);
        depth_min_ = this->declare_parameter("depth_min_value", 0.5);
        depth_max_ = this->declare_parameter("depth_max_value", 5.0);
        depth_skip_ = this->declare_parameter("depth_skip_pixel", 2);

        lidar_range_x_ = this->declare_parameter("lidar_range_x", 15.0);
        lidar_range_y_ = this->declare_parameter("lidar_range_y", 15.0);
        // LiDAR general-purpose cloud thinning: deterministic voxel-centroid, in two range bands instead
        // of a random subsample. The LiDAR is far denser near the robot than far away, so the near band
        // gets the coarser leaf (density evened out, noise averaged) and the far band a finer one (its
        // points are already sparser than the leaf, so they pass through almost untouched).
        // Wall-detection input (lidar/voxelized_sensor, read by static_structures_node): the raw scan with
        // (1) floor/ceiling points removed — a band of heights above the local ground, with wall_input_height_margin_
        //     cut at both ends (0 = no height crop) — so the ground rings stop eating RANSAC's plane budget, and
        // (2) a RANGE-DEPENDENT voxel: point density falls with the square of the distance, so with one uniform leaf
        //     a far wall holds ~1% of the points and is outvoted by near ones. wall_input_voxel_sizes[i] applies up to
        //     wall_input_band_edges[i] (one more size than edges): coarse near, fine far, evening out the density
        //     per unit of wall surface. Empty sizes = the plain voxel_resolution voxel.
        wall_input_height_margin_ = this->declare_parameter("wall_input_height_margin", 0.0);
        wall_input_voxel_sizes_ = this->declare_parameter("wall_input_voxel_sizes", std::vector<double>{});
        wall_input_band_edges_ = this->declare_parameter("wall_input_band_edges", std::vector<double>{});
        lidar_voxel_near_ = this->declare_parameter("lidar_voxel_near", 0.08);
        lidar_voxel_far_ = this->declare_parameter("lidar_voxel_far", 0.04);
        lidar_voxel_split_range_ = this->declare_parameter("lidar_voxel_split_range", 6.0);

        ground_height_ = this->declare_parameter("ground_height", -0.3);
        ground_roof_offset_ = this->declare_parameter("ground_roof_offset", 5.0);

        voxel_resolution_ = this->declare_parameter("voxel_resolution", 0.15);

        // pub_lidar_processed_'s OWN voxel step (Pass 4), decoupled from
        // voxel_resolution_ above (which stays static_structures_node's shared
        // stage — Pass 2/pub_lidar_voxelized_ — untouched; calibration_icp_node
        // has its own separate pipeline entirely and never reads from here).
        // Mirrors jo_detector's dynamicDetector::lidarPoseCB/
        // lidarOdomCB: voxel-downsample only kicks in if the point count still
        // exceeds lidar_processed_downsample_threshold_ after Pass 1-3, starting
        // at lidar_processed_voxel_size_ and growing the leaf by 10% at a time
        // (re-filtering the previous iteration's output, not the original cloud)
        // until back under threshold. dynamicDetector's own loop actually
        // multiplies the leaf by 1.1 BEFORE the first filter() call (so its real
        // first attempt is 0.11 m, not the 0.1 m it's configured with) — that
        // looks like an off-by-one rather than intent, so it's not reproduced
        // here: the first attempt below is at the configured size itself.
        lidar_processed_voxel_size_ = this->declare_parameter("lidar_processed_voxel_size", 0.1);
        lidar_processed_downsample_threshold_ = this->declare_parameter("downsample_threshold", 7000);

        // pub_depth_processed_'s OWN density filter (finishDepthCloud, used
        // for depth_final only), decoupled from voxel_resolution_ above the
        // same way — pub_depth_voxelized (static_structures_node's shared depth
        // input) keeps using voxel_resolution_ untouched. Inspired by
        // dynamicDetector::voxelFilter's density gate (a voxel with fewer
        // than depth_processed_voxel_occupied_thresh_ points is dropped
        // entirely, unlike plain pcl::VoxelGrid which emits a centroid for
        // ANY occupied voxel, even a single-point one) but not a literal
        // port of it: jo_detector's own version keeps one raw,
        // scan-order-arbitrary point per voxel once the count hits the
        // threshold; this one accumulates every point a qualifying voxel
        // gets and emits their centroid instead — see occupancyVoxelFilter()'s
        // own comment. Declared as a double, not an int, because the yaml
        // value is written as N.0 — jo_detector's own equivalent
        // (voxel_occupied_thresh) hit a rclcpp parameter-type mismatch when
        // that mismatch runs the other way (yaml float vs. declared int).
        // Runs on finishDepthCloud's `cloud` argument directly (full
        // resolution, sensor frame) — by the time that function runs, both
        // processAlignedDepth (2D image -> deprojected points) and
        // onNativeDepthCloud (already a PointCloud2) have already converged
        // to the same pcl::PointCloud<PointXYZ> representation, so one
        // filter (occupancyVoxelFilter()) covers both sources.
        // The occupancy gate is range-dependent: the native cloud is decimated by the driver, so its
        // spacing grows with distance (~1 cm at 1 m, ~6 cm at 4 m) and one fixed voxel is either too
        // coarse near or too empty far. depth_processed_voxel_size/_occupied_thresh below are the NEAR
        // band (camera-frame range < depth_processed_split_range); *_far are the band beyond it — a
        // larger voxel so distant voxels can still gather enough points, and usually a lower threshold.
        depth_processed_voxel_size_far_ = this->declare_parameter("depth_processed_voxel_size_far", 0.1);
        depth_processed_voxel_occupied_thresh_far_ = this->declare_parameter("depth_processed_voxel_occupied_thresh_far", 2.0);
        depth_processed_split_range_ = this->declare_parameter("depth_processed_split_range", 2.0);
        // true: native depth clouds are processed only for the frame closest in time to each LiDAR scan
        // (scan_trigger), not for every camera frame — the detection tick is the scan, so the other
        // ~2 of every 3 frames would be processed and never used.
        depth_process_on_scan_trigger_ = this->declare_parameter("depth_process_on_scan_trigger", false);
        // "box" (default): crop each detection's (inflated) 2D box out of the depth cloud, then isolate the
        // detected object's own depth cluster within that crop (mode + valley-bounded window — see
        // depthModeValleyWindow()). Works with ANY plain object-detection model (the existing COCO detection
        // leg already running). "mask": skip the box/inflate heuristics and use a YOLO INSTANCE-segmentation
        // model's own per-pixel masks for pixel-EXACT object membership in IMAGE space instead — still goes
        // through the SAME depth mode/valley window as "box" though (empirically confirmed necessary: a
        // silhouette edge can have a background point reproject into the mask's own 2D footprint purely from
        // depth<->color sensor parallax, despite genuinely being behind the object — pixel-exact membership in
        // image space does not imply depth consistency; see depthModeValleyWindow()'s own comment), at the
        // cost of needing yolo_seg_track_node running (its own GPU budget) and narrower class coverage
        // (whatever target_classes that node's own config lists, e.g. just "person" by default) instead of the
        // full COCO set the box method gets from the existing detection leg. Chosen per the run_detector
        // launch's own yolo_leaf_method arg, which — when "mask" — points yolo_mask_topic at that node's
        // detected_masks_topic_template and REPOINTS yolo_detections_topic at its own detections (same message
        // shape either detector publishes, see yolo_mask_topic's own comment) instead of the box detector's.
        yolo_leaf_method_ = this->declare_parameter("yolo_leaf_method", std::string("box"));
        yolo_leaf_box_inflate_ = this->declare_parameter("yolo_leaf_box_inflate", 0.10);
        yolo_leaf_min_points_ = this->declare_parameter("yolo_leaf_min_points", 10);
        yolo_leaf_depth_tolerance_ = this->declare_parameter("yolo_leaf_depth_tolerance", 0.7);
        yolo_leaf_max_age_ = this->declare_parameter("yolo_leaf_max_age", 0.3);
        // Depth-histogram bin width for the MODE-based foreground extraction (see publishSemanticLeaves's own
        // comment on why mode, not nearest-depth) — fine enough to separate the detected object from a nearer
        // occluder a typical room's worth of distance away, coarse enough to be robust to per-point depth noise.
        yolo_leaf_depth_bin_ = this->declare_parameter("yolo_leaf_depth_bin", 0.3);
        // Valley-stop fraction for the depth window around that mode (see publishSemanticLeaves's own comment):
        // expansion away from the peak bin stops at the first bin whose point count drops below this fraction
        // of the peak's own count — the "gap" that marks where the detected object's own surface ends and
        // whatever's adjacent to it in depth begins.
        yolo_leaf_valley_fraction_ = this->declare_parameter("yolo_leaf_valley_fraction", 0.15);
        // Same class list the YOLO detection nodes use (cfg/coco.names by default) — loaded here only to turn a
        // detection's class STRING into a small numeric index for the semantic-leaves cloud (see
        // publishSemanticLeaves's own comment on why a number, not the string, travels on the wire).
        {
            const std::string class_names_path = ament_index_cpp::get_package_share_directory("sins") +
                "/" + this->declare_parameter("yolo_class_names_path", std::string("cfg/coco.names"));
            std::ifstream f(class_names_path);
            std::string line;
            while (std::getline(f, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.empty()) continue;
                coco_class_to_idx_[line] = static_cast<int>(coco_class_names_.size());
                coco_class_names_.push_back(line);
            }
            RCLCPP_INFO(this->get_logger(), "Loaded %zu YOLO class names from %s for semantic leaves",
                        coco_class_names_.size(), class_names_path.c_str());
        }
        depth_trigger_max_wait_sec_ = this->declare_parameter("depth_trigger_max_wait_sec", 0.15);
        depth_processed_voxel_size_ = this->declare_parameter("depth_processed_voxel_size", 0.1);
        depth_processed_voxel_occupied_thresh_ = this->declare_parameter("depth_processed_voxel_occupied_thresh", 5.0);

        // [m] a point within this distance of a wall plane (received via
        // onWallMarkers(), see the file header) is dropped from the
        // published (LiDAR and depth) clouds.
        wall_removal_margin_ = this->declare_parameter("wall_removal_margin", 0.10);
        // [m] how far past a wall's own in-plane extent (its tracked box) points are still stripped — see isNearAnyWall()
        wall_removal_extent_margin_ = this->declare_parameter("wall_removal_extent_margin", 0.20);

        // Bootstrap floor_planes_global_ with one flat plane at
        // ground_height_ — real, possibly-tilted, per-camera fits arrive
        // asynchronously via onWallMarkers() once static_structures_node starts
        // publishing (see the file header); this is only what's used before
        // the first one lands.
        {
            PlaneModel bootstrap;
            bootstrap.normal = Eigen::Vector3d::UnitZ();
            bootstrap.d = -ground_height_;
            floor_planes_global_.push_back(bootstrap);
        }

        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this, false);
        static_tf_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

        rclcpp::QoS qos(10);
        pub_lidar_processed_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/sins/lidar/processed", qos);
        // Tiny per-scan trigger (header only, stamp = the scan's own stamp),
        // published the moment a scan is accepted in onLidarOdom, before any
        // heavy processing — lets other nodes (YOLO detection) run ONE
        // inference per scan on the camera frame closest to it instead of
        // free-running on every frame.
        pub_scan_trigger_ = this->create_publisher<std_msgs::msg::Header>(
            "/sins/lidar/scan_trigger", qos);
        // Voxel-downsampled, NOT range/ground-cropped — the ONLY thing
        // static_structures_node's own LiDAR subscription reads (see the file
        // header and that node's own header comment). Sensor frame
        // (lidar_frame_), published straight from onLidarOdom's Pass 2.
        pub_lidar_voxelized_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/sins/lidar/voxelized_sensor", qos);
        // static_structures_node's own output — walls (ns="wall_bbox") and each
        // camera's own fitted floor plane (ns="floor", one per camera),
        // parsed back into wall_planes_global_/floor_planes_global_ by
        // onWallMarkers(). Also this node's only source of wall/floor
        // visualization now (no
        // separate publishDetectedPlanes() here anymore — rviz subscribes
        // to this same topic directly).
        sub_wall_markers_ = this->create_subscription<visualization_msgs::msg::MarkerArray>(
            "/sins/static_structures/wall_markers", qos,
            [this](const visualization_msgs::msg::MarkerArray::ConstSharedPtr& msg) { onWallMarkers(msg); });

        sub_lidar_.subscribe(this, lidar_topic_);
        sub_odom_.subscribe(this, odom_topic_);
        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(20), sub_lidar_, sub_odom_);
        sync_->registerCallback(&PreprocessingNode::onLidarOdom, this);

        // ---- Cameras: an arbitrary list, not hardcoded to one. Each name
        // in camera_names gets its own parameter sub-tree (<name>.depth_topic,
        // <name>.aligned_depth_topic, ...) — same key names as the old
        // single-camera flat parameters, just prefixed per camera now.
        // Defaults below are front_camera's actual values (this robot's
        // primary camera; back_camera has no sensible universal default, so
        // its cfg entry must set every key explicitly — see
        // cfg/indoor/preprocessing.yaml). ----
        camera_names_ = this->declare_parameter(
            "camera_names", std::vector<std::string>{"front_camera"});
        for (const auto& name : camera_names_) {
            setupCamera(name);
        }

        RCLCPP_INFO(this->get_logger(), "preprocessing_node ready");
        RCLCPP_INFO(this->get_logger(), "  lidar topic: %s", lidar_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "  odom topic:  %s", odom_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "  cameras: %zu", cameras_.size());
        for (const auto& cam : cameras_) {
            RCLCPP_INFO(this->get_logger(), "    [%s] depth (raw, acquired only): %s [%s] + %s",
                        cam->name.c_str(), cam->depth_topic.c_str(), cam->depth_transport.c_str(),
                        cam->depth_camera_info_topic.c_str());
            if (cam->aligned_depth_cloud_topic.empty()) {
                RCLCPP_INFO(this->get_logger(), "    [%s] depth (aligned, deprojected from image): %s [%s] + %s",
                            cam->name.c_str(), cam->aligned_depth_topic.c_str(), cam->aligned_depth_transport.c_str(),
                            cam->aligned_depth_camera_info_topic.c_str());
            } else {
                RCLCPP_INFO(this->get_logger(), "    [%s] depth (aligned, native point cloud, no deprojection): %s",
                            cam->name.c_str(), cam->aligned_depth_cloud_topic.c_str());
            }
            RCLCPP_INFO(this->get_logger(), "    [%s] depth<->color extrinsics: %s -> static TF %s -> %s",
                        cam->name.c_str(), cam->depth_to_color_extrinsics_topic.c_str(),
                        cam->depth_optical_frame.c_str(), cam->color_optical_frame.c_str());
        }
        RCLCPP_INFO(this->get_logger(), "  frames: lidar=%s base=%s global=%s",
                    lidar_frame_.c_str(), base_frame_.c_str(), global_frame_.c_str());
        RCLCPP_INFO(this->get_logger(), "  ground(bootstrap)=%.2f roof_offset=%.2f voxel=%.3f removal_margin=%.3f",
                    ground_height_, ground_roof_offset_, voxel_resolution_, wall_removal_margin_);
        RCLCPP_INFO(this->get_logger(), "  wall/floor planes sourced from static_structures_node via /sins/static_structures/wall_markers");
    }

private:
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::PointCloud2,
        nav_msgs::msg::Odometry>;

    // =====================================================================
    // CameraStream — everything one camera needs: its own params, decoded
    // intrinsics, cached TF, subscribers and publishers. cameras_ holds one
    // per entry in camera_names_ — nothing else in this file assumes a
    // fixed count or specific names.
    // =====================================================================
    struct CameraStream {
        std::string name;

        // Scan-triggered depth processing (depth_process_on_scan_trigger_): recent native cloud
        // messages + the LiDAR scan stamps still waiting for a camera frame at/after them.
        std::deque<sensor_msgs::msg::PointCloud2::ConstSharedPtr> native_ring;
        std::deque<double> native_targets;
        double last_processed_native_stamp{-1e9};

        // ---- Params ----
        std::string depth_topic, depth_camera_info_topic, depth_transport;
        std::string aligned_depth_topic, aligned_depth_camera_info_topic, aligned_depth_transport;
        // Empty (default) = semantic ground-point extraction off for this
        // camera — see the file header. Only meaningful alongside the
        // aligned-depth-IMAGE path (aligned_depth_cloud_topic empty).
        std::string ground_mask_topic;
        std::string depth_to_color_extrinsics_topic;
        std::string depth_optical_frame, color_optical_frame;
        // If set (non-empty), this camera's already-deprojected PointCloud2
        // is read directly from here (e.g. a RealSense driver's own
        // "<name>/camera/depth/color/points") instead of decoding+deprojecting
        // aligned_depth_topic ourselves — see setupCamera() and
        // onNativeDepthCloud(). aligned_depth_topic/aligned_depth_camera_info_topic
        // and their subscriptions are unused in that case.
        std::string aligned_depth_cloud_topic;

        // ---- Decoded intrinsics — each stream's own, never shared/reused
        // across streams or cameras. ----
        double fx_depth{0.0}, fy_depth{0.0}, cx_depth{0.0}, cy_depth{0.0};
        double fx_aligned{0.0}, fy_aligned{0.0}, cx_aligned{0.0}, cy_aligned{0.0};
        int aligned_width{0}, aligned_height{0};   // color camera_info resolution K refers to
        bool has_depth_camera_info{false};
        bool has_aligned_depth_camera_info{false};
        bool has_extrinsics{false};
        bool logged_first_raw_depth{false};

        // Latest received segmentation_mask (mono8, 0/255), used best-effort
        // against whichever aligned-depth frame arrives next — see
        // ground_mask_topic's own comment and onGroundMask().
        cv::Mat latest_ground_mask;
        bool has_ground_mask{false};

        // ---- Cached static transform (base_frame -> this camera's aligned
        // depth cloud frame), same idea as the LiDAR path's T_base_lidar_ —
        // lazily resolved once the frame_id is known (first depth message),
        // then assumed rigid and never re-queried. ----
        bool has_base_depth_tf{false};
        std::string cached_depth_frame;
        Eigen::Matrix4d T_base_depth{Eigen::Matrix4d::Identity()};

        // ---- Per-camera calibration TF (lidar_frame_ -> this camera's own
        // refined_camera_frame, e.g. "front_camera_refined"), published by a
        // dedicated calibration_icp_node instance for THIS camera — see
        // ensureCameraCalibrationTf(). This is the actual lidar<->depth
        // alignment used to bring this camera's depth cloud into
        // global_frame_ (T_global_base_latest_ * T_base_lidar_ *
        // T_lidar_camera), replacing T_base_depth (the robot's own raw
        // URDF/bag TF chain) for that purpose — T_base_depth stays only as
        // the fallback used before calibration is ready. ----
        std::string refined_camera_frame;
        bool has_lidar_camera_tf{false};
        Eigen::Matrix4d T_lidar_camera{Eigen::Matrix4d::Identity()};

        // ---- Depth->color factory extrinsic, cached as a matrix (not
        // just broadcast as a TF — see onDepthToColorExtrinsics):
        // p_coloropt = T_color_depth * p_depthopt. This is EXACTLY the
        // realsense2_camera_msgs/Extrinsics "depth_to_color" message's own
        // R,t (rs2_extrinsics: rotate+translate a point FROM the depth
        // frame INTO the color frame) — NOT its inverse. (An earlier version
        // of this file, and calibration_icp_node/static_structures_node, cached
        // it under the name T_depth_color and applied .inverse(): that
        // inverted the ~5.9 cm baseline's sign — verified against the
        // aligned-depth image's own deprojection, where applying the
        // message directly gives ~0.5 cm agreement and the inverse ~4.9 cm,
        // worse than no correction at all.) Needed ONLY when
        // aligned_depth_cloud_topic is set, to bring a native cloud (which
        // arrives in depth_optical_frame) into the SAME frame convention
        // T_lidar_camera was actually calibrated against (aligned-depth's
        // own color-intrinsics deprojection, i.e. color_optical_frame) —
        // see finishDepthCloud(). Unused, harmlessly, in the default
        // (aligned-depth-image) mode.
        bool has_depth_color_matrix{false};
        Eigen::Matrix4d T_color_depth{Eigen::Matrix4d::Identity()};

        // ---- ROS I/O ----
        image_transport::Subscriber sub_depth;
        rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_depth_camera_info;
        image_transport::Subscriber sub_aligned_depth;
        rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_aligned_depth_camera_info;
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_aligned_depth_cloud;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_ground_mask;
        rclcpp::Subscription<realsense2_camera_msgs::msg::Extrinsics>::SharedPtr sub_depth_to_color_extrinsics;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_depth_cloud;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_depth_processed;
        // Voxel-downsampled, NOT floor/wall-cropped — the depth-side
        // equivalent of pub_lidar_voxelized_ (see the file header), read by
        // static_structures_node's own per-camera subscription. Published from
        // the SAME voxel grid processAlignedDepth already computes for its
        // own crop, so this costs nothing extra.
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_depth_voxelized;
        // Debug-only: the SAME cloud as pub_depth_voxelized, but pre-
        // transformed into global_frame_ instead of left in this camera's
        // own sensor frame_id — see finishDepthCloud()'s own comment on why.
        // Nothing downstream reads this; it exists purely so rviz (or
        // anything else with no independent TF opinion) can render it next
        // to static_structures_node's floor-plane markers (also global_frame_)
        // and have the comparison mean something.
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_depth_voxelized_global;
        // Only created when ground_mask_topic is set — see that member's
        // own comment and publishGroundPoints().
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_ground_points;
        // YOLO 2D detections -> 3D "semantic leaves" (see publishSemanticLeaves): the depth points that fall in each
        // detection's (inflated) box, foreground only, in global_frame with the detection's track id per point.
        std::string yolo_detections_topic;
        rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr sub_yolo;
        std::deque<vision_msgs::msg::Detection2DArray::ConstSharedPtr> yolo_ring;  // last few detections, newest last
        // Instance-MASK leaf extraction (yolo_leaf_method=="mask", an alternative to the box-crop method above —
        // see publishSemanticLeaves's own comment): a 32SC1 label image (0=background, pixel value = the SAME
        // id as the matching Detection2DArray entry's own det.id, string-parsed) from a YOLO INSTANCE-
        // segmentation node (yolo_seg_track_node.py's own detected_masks_topic_template) — pixel-EXACT object
        // membership, no box/depth-window heuristics needed at all, since instance segmentation already
        // separates objects (including occlusion) in 2D. yolo_detections_topic above is reused as-is for the
        // matching Detection2DArray (same message shape either detector publishes) — only the launch file's
        // topic wiring differs between the two methods, not this node's own subscription code.
        std::string yolo_mask_topic;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_yolo_mask;
        std::deque<sensor_msgs::msg::Image::ConstSharedPtr> yolo_mask_ring;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_semantic_leaves;
    };

    std::vector<std::string> camera_names_;
    std::vector<std::shared_ptr<CameraStream>> cameras_;

    // ---- ROS I/O (LiDAR) ----
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> sub_lidar_;
    message_filters::Subscriber<nav_msgs::msg::Odometry> sub_odom_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_lidar_processed_;
    rclcpp::Publisher<std_msgs::msg::Header>::SharedPtr pub_scan_trigger_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_lidar_voxelized_;
    rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr sub_wall_markers_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;

    // ---- Parameters ----
    std::string lidar_topic_, odom_topic_;
    std::string lidar_frame_, base_frame_, global_frame_;

    double depth_scale_{1000.0}, depth_min_{0.5}, depth_max_{5.0};
    int depth_skip_{2};
    double lidar_range_x_{15.0}, lidar_range_y_{15.0};
    double wall_input_height_margin_{0.0};
    std::vector<double> wall_input_voxel_sizes_, wall_input_band_edges_;
    double lidar_voxel_near_{0.08}, lidar_voxel_far_{0.04}, lidar_voxel_split_range_{6.0};
    double ground_height_{-0.3}, ground_roof_offset_{5.0};
    double voxel_resolution_{0.15};
    double lidar_processed_voxel_size_{0.1};
    int lidar_processed_downsample_threshold_{7000};
    double depth_processed_voxel_size_{0.1};
    double depth_processed_voxel_size_far_{0.1}, depth_processed_voxel_occupied_thresh_far_{2.0}, depth_processed_split_range_{2.0};
    bool depth_process_on_scan_trigger_{false};
    std::vector<std::string> coco_class_names_;
    std::map<std::string, int> coco_class_to_idx_;
    std::string yolo_leaf_method_{"box"};
    double yolo_leaf_box_inflate_{0.10}, yolo_leaf_depth_tolerance_{0.7}, yolo_leaf_max_age_{0.3}, yolo_leaf_depth_bin_{0.3};
    double yolo_leaf_valley_fraction_{0.15};
    int yolo_leaf_min_points_{10};
    double depth_trigger_max_wait_sec_{0.15};
    double depth_processed_voxel_occupied_thresh_{5.0};
    double wall_removal_margin_{0.10};
    double wall_removal_extent_margin_{0.20};

    // A plane n·p + d = 0. Always global frame here (both wall_planes_global_
    // and floor_planes_global_ below) — this file never fits one itself
    // anymore (see the file header), only receives already-global-frame
    // planes from static_structures_node via onWallMarkers() and uses them to
    // decide what to crop.
    struct PlaneModel {
        Eigen::Vector3d normal{Eigen::Vector3d::UnitZ()};
        double d{0.0};
        double distance(const Eigen::Vector3d& p) const { return std::abs(normal.dot(p) + d); }
        // Positive on the side normal points toward. For a floor plane
        // (normal oriented upward), positive = above the floor surface,
        // which is what the slope-aware crop needs (plain distance() can't
        // tell above from below).
        double signedDistance(const Eigen::Vector3d& p) const { return normal.dot(p) + d; }
    };

    // wall_planes_global_/floor_planes_global_: both populated by
    // onWallMarkers(), parsed out of static_structures_node's own MarkerArray
    // publish (ns="wall_bbox" -> one plane per tracked wall,
    // rotation.col(0)=normal, using the box's own center as a point on the
    // plane; ns="floor" -> one entry in floor_planes_global_ PER CAMERA, see
    // that method). Neither is fit in this file. floor_planes_global_
    // starts as one flat bootstrap plane at ground_height_ in the ctor and
    // is only ever overwritten wholesale by an incoming set of "floor"
    // markers — never reverts to the bootstrap once real data has been
    // seen, same "keep the last estimate" idea onWallMarkers' own comment
    // explains. The crop (floorHeightAboveGround(), below) treats the
    // union of the space below every one of these planes as "the ground" —
    // per explicit request, not a single shared/merged plane.
    // A tracked wall: its plane PLUS its extent (the wall_bbox CUBE: local x = normal/thickness, y/z = in-plane).
    // Removal is bounded to the wall's own box — see isNearAnyWall().
    struct WallModel {
        PlaneModel plane;
        Eigen::Vector3d center{Eigen::Vector3d::Zero()};
        Eigen::Matrix3d R{Eigen::Matrix3d::Identity()};
        Eigen::Vector3d half_size{Eigen::Vector3d::Zero()};
    };
    std::vector<WallModel> wall_planes_global_;
    std::vector<PlaneModel> floor_planes_global_;

    // ---- Cached static transform (base_frame -> lidar_frame), see the
    // file-level comment: looked up once over TF, not per callback. ----
    bool has_static_tf_{false};
    Eigen::Matrix4d T_base_lidar_{Eigen::Matrix4d::Identity()};

    // Latest synced odom pose (T_global_base), refreshed every onLidarOdom
    // call. processAlignedDepth reuses this for every camera — instead of
    // its own TF lookup to global_frame_ — to place depth points in the
    // global frame for the floor/wall crop: depth arrives faster than LiDAR
    // and unsynced, so this is at most one LiDAR period stale, and
    // critically doesn't add a dependency on anything (e.g. ekf_node)
    // broadcasting global_frame_->base_frame_ over TF — which, unlike the
    // robot's own rigid base_frame_->camera TF, may not be reliably
    // available (this was tried first and empirically was NOT).
    bool has_odom_{false};
    Eigen::Matrix4d T_global_base_latest_{Eigen::Matrix4d::Identity()};
    bool calibration_enabled_{true};

    std::mt19937 rng_;

    // ---- Per-camera setup: declares that camera's parameters (prefixed
    // <name>.), wires its subscriptions/publishers, all bound to its own
    // CameraStream instance via a capturing lambda (not std::bind — a
    // member-only callback has nowhere to carry "which camera" otherwise).
    void setupCamera(const std::string& name) {
        auto cam = std::make_shared<CameraStream>();
        cam->name = name;

        auto p = [&](const std::string& key, auto def) {
            return this->declare_parameter(name + "." + key, def);
        };

        cam->depth_topic = p("depth_topic", std::string("/") + name + "/camera/depth/image_rect_raw");
        cam->depth_camera_info_topic = p("depth_camera_info_topic", std::string("/") + name + "/camera/depth/camera_info");
        cam->depth_transport = p("depth_transport", std::string("compressedDepth"));

        cam->aligned_depth_topic = p("aligned_depth_topic", std::string("/") + name + "/camera/aligned_depth_to_color/image_raw");
        cam->aligned_depth_camera_info_topic = p("aligned_depth_camera_info_topic", std::string("/") + name + "/camera/color/camera_info");
        cam->aligned_depth_transport = p("aligned_depth_transport", std::string("compressedDepth"));
        // Empty (default) = decode+deproject aligned_depth_topic ourselves,
        // as below. Non-empty = skip that entirely and just read an
        // already-deprojected PointCloud2 straight from this topic instead
        // (e.g. a RealSense driver's own "<name>/camera/depth/color/points",
        // when the input is already a point cloud there's no reason to
        // reconstruct one from its source image) — see onNativeDepthCloud().
        cam->aligned_depth_cloud_topic = p("aligned_depth_cloud_topic", std::string(""));
        cam->ground_mask_topic = p("ground_mask_topic", std::string(""));
        cam->yolo_detections_topic = p("yolo_detections_topic", std::string(""));
        cam->yolo_mask_topic = p("yolo_mask_topic", std::string(""));

        cam->depth_to_color_extrinsics_topic = p("depth_to_color_extrinsics_topic", std::string("/") + name + "/camera/extrinsics/depth_to_color");
        cam->depth_optical_frame = p("depth_optical_frame", name + "_depth_optical_frame");
        cam->color_optical_frame = p("color_optical_frame", name + "_color_optical_frame");
        // Default follows the same "<name>_refined" convention
        // run_detector.launch.py uses for each camera's calibration_icp_node
        // instance's own refined_camera_frame arg (see that launch file).
        cam->refined_camera_frame = p("refined_camera_frame", name + "_refined");

        rclcpp::QoS qos(10);
        cam->pub_depth_cloud = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/sins/" + name + "/depth/cloud", qos);
        cam->pub_depth_processed = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/sins/" + name + "/depth/processed", qos);
        cam->pub_depth_voxelized = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/sins/" + name + "/depth/voxelized_sensor", qos);
        cam->pub_depth_voxelized_global = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/sins/" + name + "/depth/voxelized_global", qos);

        cam->sub_depth = image_transport::create_subscription(
            this, cam->depth_topic,
            [this, cam](const sensor_msgs::msg::Image::ConstSharedPtr& msg) { onRawDepth(*cam, msg); },
            cam->depth_transport, rmw_qos_profile_sensor_data);
        cam->sub_depth_camera_info = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            cam->depth_camera_info_topic, rclcpp::SensorDataQoS(),
            [this, cam](const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg) { onDepthCameraInfo(*cam, msg); });

        if (!cam->aligned_depth_cloud_topic.empty()) {
            // Native point cloud already deprojected upstream — no image to
            // decode or deproject on our side.
            cam->sub_aligned_depth_cloud = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                cam->aligned_depth_cloud_topic, rclcpp::SensorDataQoS(),
                [this, cam](const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) { onNativeDepthArrival(*cam, msg); });
            // camera_info is still needed ONLY to project native points onto
            // a 2D segmentation mask (semantic ground, see onNativeDepthCloud)
            // — the color camera's own intrinsics, same topic the image path
            // reads for its deprojection.
            if (!cam->ground_mask_topic.empty() || !cam->yolo_detections_topic.empty()) {
                cam->sub_aligned_depth_camera_info = this->create_subscription<sensor_msgs::msg::CameraInfo>(
                    cam->aligned_depth_camera_info_topic, rclcpp::SensorDataQoS(),
                    [this, cam](const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg) { onAlignedDepthCameraInfo(*cam, msg); });
            }
        } else {
            cam->sub_aligned_depth = image_transport::create_subscription(
                this, cam->aligned_depth_topic,
                [this, cam](const sensor_msgs::msg::Image::ConstSharedPtr& msg) { processAlignedDepth(*cam, msg); },
                cam->aligned_depth_transport, rmw_qos_profile_sensor_data);
            cam->sub_aligned_depth_camera_info = this->create_subscription<sensor_msgs::msg::CameraInfo>(
                cam->aligned_depth_camera_info_topic, rclcpp::SensorDataQoS(),
                [this, cam](const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg) { onAlignedDepthCameraInfo(*cam, msg); });
        }
        if (!cam->yolo_detections_topic.empty()) {
            cam->sub_yolo = this->create_subscription<vision_msgs::msg::Detection2DArray>(
                cam->yolo_detections_topic, rclcpp::QoS(10),
                [cam](const vision_msgs::msg::Detection2DArray::ConstSharedPtr& msg) {
                    cam->yolo_ring.push_back(msg);
                    while (cam->yolo_ring.size() > 3) cam->yolo_ring.pop_front();
                });
            cam->pub_semantic_leaves = this->create_publisher<sensor_msgs::msg::PointCloud2>(
                "/sins/" + name + "/semantic_leaves", qos);
        }
        if (!cam->yolo_mask_topic.empty()) {
            cam->sub_yolo_mask = this->create_subscription<sensor_msgs::msg::Image>(
                cam->yolo_mask_topic, rclcpp::SensorDataQoS(),
                [cam](const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
                    cam->yolo_mask_ring.push_back(msg);
                    while (cam->yolo_mask_ring.size() > 3) cam->yolo_mask_ring.pop_front();
                });
        }
        // Semantic ground works on BOTH depth paths (per-pixel masking of the
        // aligned image, or per-point projection of a native cloud onto the
        // mask) — see ground_mask_topic's own comment.
        if (!cam->ground_mask_topic.empty()) {
            cam->sub_ground_mask = this->create_subscription<sensor_msgs::msg::Image>(
                cam->ground_mask_topic, rclcpp::SensorDataQoS(),
                [this, cam](const sensor_msgs::msg::Image::ConstSharedPtr& msg) { onGroundMask(*cam, msg); });
            cam->pub_ground_points = this->create_publisher<sensor_msgs::msg::PointCloud2>(
                "/sins/" + name + "/semantic_ground_points", qos);
        }

        // transient_local — matches what the RealSense driver (and this
        // bag, which recorded the SAME offered QoS) actually publishes this
        // topic with: exactly once, a latched static fact, not a stream
        // (see calibration_icp_node.cpp's own onDepthToColorExtrinsics
        // subscription for the full reasoning — same fix, same topic kind).
        // A plain volatile QoS(1) here meant a late-joining subscriber (any
        // restart of this node after the bag's own single send for the
        // current loop) could wait up to a full bag loop before this
        // arrived, during which finishDepthCloud() fell back to the
        // degraded ensureBaseDepthTf() path instead.
        cam->sub_depth_to_color_extrinsics = this->create_subscription<realsense2_camera_msgs::msg::Extrinsics>(
            cam->depth_to_color_extrinsics_topic, rclcpp::QoS(1).transient_local(),
            [this, cam](const realsense2_camera_msgs::msg::Extrinsics::ConstSharedPtr& msg) { onDepthToColorExtrinsics(*cam, msg); });

        cameras_.push_back(cam);
    }

    bool ensureStaticTf() {
        if (has_static_tf_) {
            return true;
        }
        try {
            // Zero timeout: tf_listener_ has no dedicated thread (see the
            // ctor), so any nonzero-timeout lookupTransform call can't
            // actually be serviced and just prints tf2_buffer's "Do not
            // call lookupTransform with a timeout..." warning instead of
            // waiting. This is called from every onLidarOdom callback
            // until it succeeds (has_static_tf_ latches true after), so at
            // ~10Hz a nonzero timeout here would spam that warning for
            // however long the static TF takes to appear.
            auto tf_stamped = tf_buffer_->lookupTransform(
                base_frame_, lidar_frame_, tf2::TimePointZero);
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
            RCLCPP_WARN(this->get_logger(), "Waiting for static TF %s -> %s: %s",
                        base_frame_.c_str(), lidar_frame_.c_str(), ex.what());
        }
        return has_static_tf_;
    }

    // Same idea as ensureStaticTf(), for one camera's aligned-depth cloud
    // frame instead of lidar_frame_. That frame isn't known until the
    // first depth message from THAT camera arrives (it's whatever the
    // driver reports), so this is called from processAlignedDepth instead
    // of setupCamera(); once resolved for a given frame name it's assumed
    // rigid and never re-queried, same assumption ensureStaticTf() makes.
    bool ensureBaseDepthTf(CameraStream& cam, const std::string& depth_frame) {
        if (cam.has_base_depth_tf && cam.cached_depth_frame == depth_frame) {
            return true;
        }
        try {
            auto tf_stamped = tf_buffer_->lookupTransform(
                base_frame_, depth_frame, tf2::TimePointZero);
            Eigen::Quaterniond q(
                tf_stamped.transform.rotation.w, tf_stamped.transform.rotation.x,
                tf_stamped.transform.rotation.y, tf_stamped.transform.rotation.z);
            cam.T_base_depth.setIdentity();
            cam.T_base_depth.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
            cam.T_base_depth(0, 3) = tf_stamped.transform.translation.x;
            cam.T_base_depth(1, 3) = tf_stamped.transform.translation.y;
            cam.T_base_depth(2, 3) = tf_stamped.transform.translation.z;
            cam.has_base_depth_tf = true;
            cam.cached_depth_frame = depth_frame;
            RCLCPP_INFO(this->get_logger(), "Cached static TF %s -> %s", base_frame_.c_str(), depth_frame.c_str());
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Waiting for static TF %s -> %s: %s", base_frame_.c_str(), depth_frame.c_str(), ex.what());
        }
        return cam.has_base_depth_tf;
    }

    // Per-camera equivalent of ensureBaseDepthTf, but sourced from
    // calibration instead of the robot's own URDF/bag TF chain: looks up
    // lidar_frame_ -> cam.refined_camera_frame (e.g. "front_camera_refined"),
    // published by the calibration_icp_node instance dedicated to THIS
    // camera (see run_detector.launch.py, one instance per camera_names_
    // entry). This is what actually aligns each camera's depth cloud with
    // the LiDAR cloud for the global-frame crop/registry-strip in
    // processAlignedDepth — T_base_depth (the raw TF chain) is only the
    // fallback used before this is ready. Cached once resolved, same
    // "rigid enough, don't re-query every callback" tradeoff
    // ensureBaseDepthTf makes; a running calibration_icp_node in periodic
    // recalibration mode would need this invalidated to pick up a refined
    // estimate, which isn't done here (out of scope for now — the initial
    // ICP convergence is what this pipeline depends on being correct).
    bool ensureCameraCalibrationTf(CameraStream& cam) {
        if (cam.has_lidar_camera_tf) {
            return true;
        }
        try {
            auto tf_stamped = tf_buffer_->lookupTransform(
                lidar_frame_, cam.refined_camera_frame, tf2::TimePointZero);
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
                "Waiting for calibration TF %s -> %s (camera '%s'): %s — falling back to raw TF chain",
                lidar_frame_.c_str(), cam.refined_camera_frame.c_str(), cam.name.c_str(), ex.what());
        }
        return cam.has_lidar_camera_tf;
    }

    // =====================================================================
    // Wall/floor points are stripped using wall_planes_global_/
    // floor_planes_global_, both populated externally by onWallMarkers()
    // below — see the file header for the full architecture.
    // =====================================================================

    // Within wall_removal_margin_ of a wall's plane AND within its in-plane extent (+ wall_removal_extent_margin_).
    // The plane alone is unbounded: every point within the margin of a wall's infinite EXTENSION was dropped too —
    // observed on a lab bag: a person standing still 5 m from a 3.6 m wall, on the line of its extension, lost
    // ~60% of their LiDAR points the moment that wall was registered, and with them their detection.
    bool isNearAnyWall(const Eigen::Vector3d& p_global) const {
        for (const auto& w : wall_planes_global_) {
            if (w.plane.distance(p_global) >= wall_removal_margin_) continue;
            const Eigen::Vector3d local = w.R.transpose() * (p_global - w.center);
            if (std::abs(local.y()) <= w.half_size.y() + wall_removal_extent_margin_ &&
                std::abs(local.z()) <= w.half_size.z() + wall_removal_extent_margin_)
                return true;
        }
        return false;
    }

    // "Height above ground" at a point, generalized to N floor planes (one
    // per camera — see the file header and onWallMarkers()): the MINIMUM
    // signedDistance across every plane. Per explicit request, the ground
    // is the UNION of the space below each camera's own plane, not a
    // single shared/merged one — taking the min is exactly that union: if
    // ANY plane's signedDistance is small/negative (the point is at/below
    // that camera's floor), the min captures it and the existing
    // "h < margin" crop test drops the point; a point only survives the
    // "too high above the ground" (roof) test when it's far above EVERY
    // plane, i.e. the min itself exceeds the roof offset. With exactly one
    // plane (the bootstrap, or only one camera ever finding a floor) this
    // reduces to the original single-plane behavior exactly.
    double heightAboveGround(const Eigen::Vector3d& p_global) const {
        double h_min = std::numeric_limits<double>::infinity();
        for (const auto& plane : floor_planes_global_) {
            h_min = std::min(h_min, plane.signedDistance(p_global));
        }
        return h_min;
    }

    // Parses static_structures_node's own MarkerArray publish back into
    // wall_planes_global_/floor_planes_global_ — the async replacement for
    // this file's own former inline RANSAC. Mirrors
    // jo_detector/dynamicDetector's wallMarkersCB exactly (same
    // ns="wall_bbox" CUBE -> OBB parse), generalized to ALSO recover EVERY
    // camera's own floor plane, one per ns="floor" marker (static_structures_node
    // encodes each losslessly: orientation's local +Z axis is the plane
    // normal — see that file's own publishMarkers() — position is any
    // point on the plane, so d = -normal·position).
    void onWallMarkers(const visualization_msgs::msg::MarkerArray::ConstSharedPtr& msg) {
        std::vector<WallModel> new_walls;
        new_walls.reserve(msg->markers.size());
        std::vector<PlaneModel> new_floors;

        for (const auto& m : msg->markers) {
            if (m.action == visualization_msgs::msg::Marker::DELETEALL) continue;
            const Eigen::Quaterniond q(
                m.pose.orientation.w, m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z);
            const Eigen::Matrix3d R = q.normalized().toRotationMatrix();
            const Eigen::Vector3d center(m.pose.position.x, m.pose.position.y, m.pose.position.z);

            if (m.ns == "wall_bbox") {
                WallModel wall;
                wall.plane.normal = R.col(0);  // rotation.col(0) = normal, by convention (see static_structures_node's WallBBox)
                wall.plane.d = -wall.plane.normal.dot(center);
                wall.center = center;
                wall.R = R;
                wall.half_size = 0.5 * Eigen::Vector3d(m.scale.x, m.scale.y, m.scale.z);
                new_walls.push_back(wall);
            } else if (m.ns == "floor") {
                // One marker per camera now (id = that camera's index in
                // static_structures_node's own cameras_) — collect all of them,
                // don't just keep the last one seen.
                PlaneModel plane;
                plane.normal = R.col(2);  // local +Z axis = the plane's normal (see static_structures_node)
                plane.d = -plane.normal.dot(center);
                new_floors.push_back(plane);
            }
        }

        wall_planes_global_ = std::move(new_walls);
        // Only overwrite on an actual non-empty floor set — never revert to
        // the ground_height_ bootstrap just because a given publish
        // happened not to include any (shouldn't normally happen once at
        // least one camera has an estimate, but defensive all the same).
        if (!new_floors.empty()) {
            floor_planes_global_ = std::move(new_floors);
        }
    }

    void onDepthCameraInfo(CameraStream& cam, const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg) {
        cam.fx_depth = msg->k[0]; cam.fy_depth = msg->k[4]; cam.cx_depth = msg->k[2]; cam.cy_depth = msg->k[5];
        if (!cam.has_depth_camera_info) {
            cam.has_depth_camera_info = true;
            RCLCPP_INFO(this->get_logger(), "[%s] Depth (raw) intrinsics: fx=%.3f fy=%.3f cx=%.3f cy=%.3f",
                        cam.name.c_str(), cam.fx_depth, cam.fy_depth, cam.cx_depth, cam.cy_depth);
        }
    }

    void onAlignedDepthCameraInfo(CameraStream& cam, const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg) {
        cam.fx_aligned = msg->k[0]; cam.fy_aligned = msg->k[4]; cam.cx_aligned = msg->k[2]; cam.cy_aligned = msg->k[5];
        cam.aligned_width = static_cast<int>(msg->width); cam.aligned_height = static_cast<int>(msg->height);
        if (!cam.has_aligned_depth_camera_info) {
            cam.has_aligned_depth_camera_info = true;
            RCLCPP_INFO(this->get_logger(), "[%s] Depth (aligned) intrinsics (= color camera's): fx=%.3f fy=%.3f cx=%.3f cy=%.3f",
                        cam.name.c_str(), cam.fx_aligned, cam.fy_aligned, cam.cx_aligned, cam.cy_aligned);
        }
    }

    // See the file-level NOTE on this function: rotation is assumed
    // column-major (librealsense's rs2_extrinsics convention); if a
    // camera's rebroadcast TF looks mirrored/wrong, transpose it here.
    void onDepthToColorExtrinsics(CameraStream& cam, const realsense2_camera_msgs::msg::Extrinsics::ConstSharedPtr& msg) {
        if (cam.has_extrinsics) {
            return;  // static, only needs to be broadcast once
        }
        Eigen::Matrix3d R;
        R << msg->rotation[0], msg->rotation[3], msg->rotation[6],
             msg->rotation[1], msg->rotation[4], msg->rotation[7],
             msg->rotation[2], msg->rotation[5], msg->rotation[8];
        Eigen::Quaterniond q(R);
        q.normalize();

        // The message's own R,t map a DEPTH-frame point into the COLOR
        // frame: p_color = R p_depth + t (see CameraStream::T_color_depth's
        // own comment).
        cam.T_color_depth.setIdentity();
        cam.T_color_depth.block<3, 3>(0, 0) = q.toRotationMatrix();
        cam.T_color_depth(0, 3) = msg->translation[0];
        cam.T_color_depth(1, 3) = msg->translation[1];
        cam.T_color_depth(2, 3) = msg->translation[2];
        cam.has_depth_color_matrix = true;

        // A TF message (parent=depth_optical, child=color_optical) carries
        // the child's pose in the parent, i.e. p_parent = T * p_child =
        // p_depth = T * p_color: the INVERSE of the matrix above. (Publishing
        // the message's numbers as-is, as an earlier version did, put the
        // ~5.9 cm baseline on the wrong side; this now agrees with the
        // RealSense driver's own depth->color TF that the robot/bag already
        // broadcasts, instead of fighting it.)
        const Eigen::Matrix4d T_depth_color_tf = cam.T_color_depth.inverse();
        const Eigen::Quaterniond q_tf(Eigen::Matrix3d(T_depth_color_tf.block<3, 3>(0, 0)));
        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp = this->now();
        tf_msg.header.frame_id = cam.depth_optical_frame;
        tf_msg.child_frame_id = cam.color_optical_frame;
        tf_msg.transform.translation.x = T_depth_color_tf(0, 3);
        tf_msg.transform.translation.y = T_depth_color_tf(1, 3);
        tf_msg.transform.translation.z = T_depth_color_tf(2, 3);
        tf_msg.transform.rotation.w = q_tf.w();
        tf_msg.transform.rotation.x = q_tf.x();
        tf_msg.transform.rotation.y = q_tf.y();
        tf_msg.transform.rotation.z = q_tf.z();
        static_tf_broadcaster_->sendTransform(tf_msg);

        cam.has_extrinsics = true;
        RCLCPP_INFO(this->get_logger(), "[%s] Published static TF %s -> %s from depth<->color extrinsics",
                    cam.name.c_str(), cam.depth_optical_frame.c_str(), cam.color_optical_frame.c_str());
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

    // Acquired only — see file header. Nothing consumes this yet; the
    // future UV-map detection stage (phase 2) will subscribe to each
    // camera's depth_topic directly itself rather than go through this
    // node, this is just here so the wiring/config exists and the image is
    // known to decode correctly end to end.
    void onRawDepth(CameraStream& cam, const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg) {
        if (!cam.logged_first_raw_depth) {
            cam.logged_first_raw_depth = true;
            RCLCPP_INFO(this->get_logger(), "[%s] First raw depth frame received (%ux%u, %s) — acquired, not processed here",
                        cam.name.c_str(), depth_msg->width, depth_msg->height, depth_msg->encoding.c_str());
        }
    }

    // mono8, 0/255 — see ground_mask_topic's own comment. Stored as-is;
    // matched against whichever aligned-depth frame processAlignedDepth
    // next receives, not this exact one (best-effort, see file header).
    void onGroundMask(CameraStream& cam, const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        try {
            cam.latest_ground_mask = cv_bridge::toCvCopy(msg, "mono8")->image;
            cam.has_ground_mask = true;
        } catch (const std::exception& ex) {
            RCLCPP_ERROR(this->get_logger(), "[%s] Ground mask cv_bridge conversion failed: %s", cam.name.c_str(), ex.what());
        }
    }

    // Global-frame transform for THIS camera's aligned-depth-frame points —
    // factored out of finishDepthCloud (still its only other caller) so
    // publishGroundPoints() can place semantic-ground points in the exact
    // same global frame as depth_final without re-deriving this camera's
    // calibration a second, possibly-drifting way. See finishDepthCloud's
    // own (unchanged) comment for the full reasoning behind this specific
    // composition — not repeated here.
    bool computeGlobalDepthTransform(CameraStream& cam, const std::string& frame_id,
                                      bool is_native_depth_frame, Eigen::Matrix4d& T_global_depth) {
        const bool have_depth_color_tf = !is_native_depth_frame || cam.has_depth_color_matrix;
        const bool have_calibration_tf = calibration_enabled_ && have_depth_color_tf &&
            ensureStaticTf() && ensureCameraCalibrationTf(cam);
        // Calibration is enabled but not applied this frame: the fallback
        // below silently uses the raw URDF/bag TF chain, and points from
        // this camera then do NOT line up with the (calibrated) LiDAR
        // cloud — normally only for the first moments after startup, but
        // it must never be invisible. The missing-calibration-TF case is
        // already warned inside ensureCameraCalibrationTf(); the missing
        // depth<->color extrinsic (native-cloud path only) had no log.
        if (calibration_enabled_ && !have_depth_color_tf) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "[%s] calibration enabled but NOT applied: native depth cloud needs the depth<->color extrinsic "
                "(%s), not received yet — using the raw TF chain, depth will not be aligned with the LiDAR until it arrives",
                cam.name.c_str(), cam.depth_to_color_extrinsics_topic.c_str());
        }
        const bool have_global_tf = has_odom_ &&
            (have_calibration_tf || ensureBaseDepthTf(cam, frame_id));
        if (!have_global_tf) {
            return false;
        }
        Eigen::Matrix4d T_lidar_frame = cam.T_lidar_camera;
        if (is_native_depth_frame && have_calibration_tf) {
            // p_lidar = T_lidar_camera * p_color = T_lidar_camera * T_color_depth * p_depth
            T_lidar_frame = Eigen::Matrix4d(T_lidar_frame * cam.T_color_depth);
        }
        T_global_depth = have_calibration_tf
            ? Eigen::Matrix4d(T_global_base_latest_ * T_base_lidar_ * T_lidar_frame)
            : Eigen::Matrix4d(T_global_base_latest_ * cam.T_base_depth);
        return true;
    }

    // floor_cloud: points processAlignedDepth already deprojected AND found
    // masked 255 in cam.latest_ground_mask — this just transforms them into
    // global_frame_ (same convention as depth_processed, see the file
    // header) and publishes, no further crop (the mask already decided
    // "this is ground", unlike depth_processed's own geometric height/wall
    // test). No-op if this camera has no ground_mask_topic configured
    // (pub_ground_points null) or nothing survived the mask this frame.
    void publishGroundPoints(CameraStream& cam, pcl::PointCloud<pcl::PointXYZ>& floor_cloud,
                              const builtin_interfaces::msg::Time& stamp, const std::string& frame_id,
                              bool is_native_depth_frame) {
        if (!cam.pub_ground_points || floor_cloud.points.empty()) {
            return;
        }
        Eigen::Matrix4d T_global_depth;
        if (!computeGlobalDepthTransform(cam, frame_id, is_native_depth_frame, T_global_depth)) {
            return;
        }
        pcl::PointCloud<pcl::PointXYZ> global_cloud;
        global_cloud.points.reserve(floor_cloud.points.size());
        for (const auto& pt : floor_cloud.points) {
            const Eigen::Vector4d p_global4 = T_global_depth * Eigen::Vector4d(pt.x, pt.y, pt.z, 1.0);
            global_cloud.points.emplace_back(
                static_cast<float>(p_global4(0)), static_cast<float>(p_global4(1)), static_cast<float>(p_global4(2)));
        }
        global_cloud.width = static_cast<uint32_t>(global_cloud.points.size());
        global_cloud.height = 1;
        sensor_msgs::msg::PointCloud2 out;
        pcl::toROSMsg(global_cloud, out);
        out.header.stamp = stamp;
        out.header.frame_id = global_frame_;
        cam.pub_ground_points->publish(out);
    }

    void processAlignedDepth(CameraStream& cam, const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg) {
        if (!cam.has_aligned_depth_camera_info) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                "[%s] No camera_info yet on %s, skipping aligned depth frame",
                cam.name.c_str(), cam.aligned_depth_camera_info_topic.c_str());
            return;
        }

        cv::Mat depth_image;
        try {
            depth_image = cv_bridge::toCvShare(depth_msg)->image;
        } catch (const std::exception& ex) {
            RCLCPP_ERROR(this->get_logger(), "[%s] Aligned depth cv_bridge conversion failed: %s", cam.name.c_str(), ex.what());
            return;
        }

        const bool is_u16 = depth_image.type() == CV_16UC1;
        const bool is_f32 = depth_image.type() == CV_32FC1;
        if (!is_u16 && !is_f32) {
            RCLCPP_ERROR(this->get_logger(), "[%s] Unsupported aligned depth encoding (need 16UC1 or 32FC1)", cam.name.c_str());
            return;
        }

        pcl::PointCloud<pcl::PointXYZ> cloud;
        cloud.points.reserve((depth_image.rows / depth_skip_) * (depth_image.cols / depth_skip_));

        // Only meaningful when both true: a mask exists AND is the exact
        // same resolution as this depth frame (aligned depth lives in the
        // color pixel grid — see file header — and so does
        // segmentation_mask, since it's run on that same camera's color
        // image; a size mismatch means something upstream reconfigured
        // without the other, so skip rather than index out of bounds/wrong).
        const bool have_mask = cam.pub_ground_points && cam.has_ground_mask &&
            cam.latest_ground_mask.rows == depth_image.rows && cam.latest_ground_mask.cols == depth_image.cols;
        if (cam.pub_ground_points && cam.has_ground_mask && !have_mask) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "[%s] ground mask size (%dx%d) != aligned depth size (%dx%d), skipping semantic ground points this frame",
                cam.name.c_str(), cam.latest_ground_mask.cols, cam.latest_ground_mask.rows,
                depth_image.cols, depth_image.rows);
        }
        pcl::PointCloud<pcl::PointXYZ> floor_cloud;

        for (int y = 0; y < depth_image.rows; y += depth_skip_) {
            for (int x = 0; x < depth_image.cols; x += depth_skip_) {
                double z = 0.0;
                if (is_u16) {
                    const uint16_t d = depth_image.at<uint16_t>(y, x);
                    if (d == 0) continue;
                    z = static_cast<double>(d) / depth_scale_;
                } else {
                    const float d = depth_image.at<float>(y, x);
                    if (!std::isfinite(d) || d <= 0.0f) continue;
                    z = static_cast<double>(d);
                }
                if (z < depth_min_ || z > depth_max_) continue;

                // Aligned depth's own (= this camera's color) intrinsics —
                // never the raw depth sensor's fx_depth/cx_depth/etc, and
                // never another camera's.
                pcl::PointXYZ pt;
                pt.x = static_cast<float>((static_cast<double>(x) - cam.cx_aligned) * z / cam.fx_aligned);
                pt.y = static_cast<float>((static_cast<double>(y) - cam.cy_aligned) * z / cam.fy_aligned);
                pt.z = static_cast<float>(z);
                cloud.points.push_back(pt);

                if (have_mask && cam.latest_ground_mask.at<uint8_t>(y, x) >= 128) {
                    floor_cloud.points.push_back(pt);
                }
            }
        }

        // Aligned depth lives in the color camera's pixel grid, so its
        // points are expressed in the color optical frame, not the depth
        // sensor's own — use the driver-reported frame_id when present
        // (it should already say so), falling back to this camera's
        // configured color_optical_frame otherwise.
        const std::string frame_id = !depth_msg->header.frame_id.empty() ? depth_msg->header.frame_id : cam.color_optical_frame;
        finishDepthCloud(cam, cloud, depth_msg->header.stamp, frame_id, /*is_native_depth_frame=*/false);
        publishGroundPoints(cam, floor_cloud, depth_msg->header.stamp, frame_id, /*is_native_depth_frame=*/false);
    }

    // Alternative to processAlignedDepth() above, when
    // aligned_depth_cloud_topic is set (see setupCamera): the upstream
    // driver has already deprojected depth into a PointCloud2 (e.g. a
    // RealSense driver's own "<name>/camera/depth/color/points"), so there's
    // no image to decode or intrinsics to apply — pcl::fromROSMsg extracts
    // just the x/y/z fields it needs, silently ignoring any extra ones
    // (e.g. an rgb field) the source cloud may carry. Only real added work
    // is the depth_min_/depth_max_ range filter processAlignedDepth's own
    // per-pixel loop would otherwise have applied — everything else is
    // shared via finishDepthCloud.
    static double stampSec(const builtin_interfaces::msg::Time& t) { return t.sec + t.nanosec * 1e-9; }

    void onNativeDepthArrival(CameraStream& cam, const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) {
        if (!depth_process_on_scan_trigger_) {
            onNativeDepthCloud(cam, msg);
            return;
        }
        if (!cam.native_ring.empty() && stampSec(msg->header.stamp) < stampSec(cam.native_ring.back()->header.stamp) - 1.0) {
            cam.native_ring.clear();  // clock jumped back (bag loop)
            cam.native_targets.clear();
            cam.last_processed_native_stamp = -1e9;
        }
        cam.native_ring.push_back(msg);
        while (cam.native_ring.size() > 6) cam.native_ring.pop_front();
        resolveNativeTargets(cam);
    }

    // A LiDAR scan arrived: every native-cloud camera will process the frame closest to its stamp,
    // as soon as a frame at/after that stamp exists (camera clouds arrive ~30 ms after their stamp).
    void queueDepthTargets(const builtin_interfaces::msg::Time& scan_stamp) {
        if (!depth_process_on_scan_trigger_) return;
        const double t = stampSec(scan_stamp);
        for (auto& cam : cameras_) {
            if (cam->aligned_depth_cloud_topic.empty()) continue;
            if (!cam->native_targets.empty() && t < cam->native_targets.back() - 1.0) cam->native_targets.clear();
            cam->native_targets.push_back(t);
            while (cam->native_targets.size() > 4) cam->native_targets.pop_front();
            resolveNativeTargets(*cam);
        }
    }

    void resolveNativeTargets(CameraStream& cam) {
        while (!cam.native_targets.empty() && !cam.native_ring.empty()) {
            const double target = cam.native_targets.front();
            const double newest = stampSec(cam.native_ring.back()->header.stamp);
            if (newest >= target) {
                sensor_msgs::msg::PointCloud2::ConstSharedPtr best = cam.native_ring.front();
                double best_dt = std::abs(stampSec(best->header.stamp) - target);
                for (const auto& m : cam.native_ring) {
                    const double dt = std::abs(stampSec(m->header.stamp) - target);
                    if (dt < best_dt) { best_dt = dt; best = m; }
                }
                cam.native_targets.pop_front();
                const double bs = stampSec(best->header.stamp);
                if (bs != cam.last_processed_native_stamp) {
                    cam.last_processed_native_stamp = bs;
                    onNativeDepthCloud(cam, best);
                }
            } else if (newest < target - depth_trigger_max_wait_sec_) {
                cam.native_targets.pop_front();  // no frame is coming for this scan
            } else {
                break;
            }
        }
    }

    // YOLO 2D detections -> 3D leaves. Every native depth point is projected onto the colour image (depth -> colour
    // extrinsic, colour intrinsics); the points inside a detection's box, inflated by yolo_leaf_box_inflate (the YOLO frame
    // is older than the depth frame by a few tens of ms and the object moved), form its candidate cluster; only the
    // DOMINANT DEPTH CLUSTER is kept — the box's own depth histogram (binned by yolo_leaf_depth_bin), starting at
    // its mode (peak bin) and expanding outward until the point density drops below yolo_leaf_valley_fraction of
    // the peak's own count (a VALLEY, not a fixed distance — see below for why), capped by
    // yolo_leaf_depth_tolerance as an absolute safety bound — so a wall or boxes sharing the same rectangle are
    // left out. This used to anchor on the single NEAREST depth (as jo_detector does) — correct for the
    // common case (the detected object is the closest thing in its own box) but systematically wrong for a
    // PARTIALLY-OCCLUDED detection (visible through/around something nearer within the same inflated box, e.g. a
    // person behind boxes): the nearest-depth anchor would lock onto the occluder's own surface every time,
    // silently losing the actual detection's semantic leaf even though YOLO classified it correctly. The mode
    // (and the valley-bounded window around it) is a much better proxy for "the object this box was actually
    // drawn around", since a correctly classified detection's own pixels dominate its box even when some of the
    // occluder's nearer pixels are unavoidably included too.
    //
    // Output: global_frame points with the detection's track id AND class (as class_idx, an index into coco_class_names_ so it fits a PointCloud2 field —
    // see coco_class_to_idx_'s own comment), published on every depth frame (empty if there is none, so consumers
    // can tell). The track id is a LOCAL, SAME-MESSAGE key only — grouping points into one leaf per detection — it
    // is not meant to, and does not need to, survive past dbscan_detector_node's onLeaves(): identity across ticks
    // and across cameras is the 3D tracker's own job (position/motion in global_frame, camera-agnostic by
    // construction), not something built out of a 2D per-camera ByteTrack id that changes the moment an object
    // crosses from one camera's field of view into the other's.
    // Depth-histogram MODE, not nearest-depth (see publishSemanticLeaves's own comment on why): the DOMINANT
    // depth cluster (most points, binned by yolo_leaf_depth_bin_) within a set of candidate points that are
    // already known to belong to one detection/instance in IMAGE SPACE — a box-crop's foreground split, or a
    // mask label's own reprojected points. Shared by both yolo_leaf_method branches because both need it: a
    // box crop can contain an occluder's own nearer pixels alongside the detected object's; a per-pixel mask,
    // even though it is already pixel-exact in image space, has no such guarantee in DEPTH — at a silhouette
    // edge, a background point (behind the object) can reproject into the mask's own 2D footprint purely from
    // depth<->color sensor parallax (different viewpoints, same scene), despite never actually being part of
    // that instance. Valley-bounded expansion from the mode bin, not a fixed +/- window: it follows the real
    // SHAPE of the depth distribution, stopping at the first near-empty bin (density below
    // yolo_leaf_valley_fraction_ of the peak) — i.e. the actual gap between the object and whatever's adjacent
    // to it in depth, however close that gap is, while keeping the object's own full extent if it's genuinely
    // deep (points stay dense all the way through, no valley to stop at). yolo_leaf_depth_tolerance_ remains an
    // absolute safety cap on top, in case a noisy/sparse histogram never finds a clean valley.
    std::pair<float, float> depthModeValleyWindow(const std::vector<float>& depths) const {
        const float dmin = *std::min_element(depths.begin(), depths.end());
        const float dmax = *std::max_element(depths.begin(), depths.end());
        const int nBins = std::max(1, static_cast<int>(std::ceil((dmax - dmin) / static_cast<float>(yolo_leaf_depth_bin_))) + 1);
        std::vector<int> hist(static_cast<size_t>(nBins), 0);
        for (float d : depths) {
            int bin = static_cast<int>((d - dmin) / static_cast<float>(yolo_leaf_depth_bin_));
            bin = std::min(std::max(bin, 0), nBins - 1);
            ++hist[static_cast<size_t>(bin)];
        }
        int bestBin = 0;
        for (int i = 1; i < nBins; ++i) if (hist[static_cast<size_t>(i)] > hist[static_cast<size_t>(bestBin)]) bestBin = i;
        const int valleyFloor = std::max(1, static_cast<int>(std::ceil(hist[static_cast<size_t>(bestBin)] * yolo_leaf_valley_fraction_)));
        int lo = bestBin, hi = bestBin;
        while (lo > 0 && hist[static_cast<size_t>(lo - 1)] >= valleyFloor) --lo;
        while (hi < nBins - 1 && hist[static_cast<size_t>(hi + 1)] >= valleyFloor) ++hi;
        const float modeDepth = dmin + (static_cast<float>(bestBin) + 0.5f) * static_cast<float>(yolo_leaf_depth_bin_);
        const float depthLo = std::max(dmin + static_cast<float>(lo) * static_cast<float>(yolo_leaf_depth_bin_),
                                       modeDepth - static_cast<float>(yolo_leaf_depth_tolerance_));
        const float depthHi = std::min(dmin + static_cast<float>(hi + 1) * static_cast<float>(yolo_leaf_depth_bin_),
                                       modeDepth + static_cast<float>(yolo_leaf_depth_tolerance_));
        return {depthLo, depthHi};
    }

    void publishSemanticLeaves(CameraStream& cam, const pcl::PointCloud<pcl::PointXYZ>& cloud,
                               const builtin_interfaces::msg::Time& stamp, const std::string& frame_id) {
        if (!cam.pub_semantic_leaves) return;
        Eigen::Matrix4d T_global_depth;
        if (!computeGlobalDepthTransform(cam, frame_id, /*is_native_depth_frame=*/true, T_global_depth)) return;

        // Leaf.class_idx: index into class_names_ (loaded from yolo_class_names_path, the same cfg/coco.names the
        // YOLO detection nodes use) for the ORIGINAL detection's own class string — -1 when that string isn't in
        // the list. Encoded as a number (not the string itself) because PointCloud2 fields are numeric; the class
        // is a per-DETECTION property (one leaf = one detection), so it never needs to survive past this camera's
        // own detection message — no track id crosses this boundary (see the file header's own reasoning on why
        // semantic identity lives on the 3D track, not on any 2D id).
        struct Leaf { double id; int class_idx; std::vector<Eigen::Vector3f> pts; };
        std::vector<Leaf> leaves;
        // the detection whose frame stamp is closest to this depth frame's stamp (the newest one may still be in flight)
        vision_msgs::msg::Detection2DArray::ConstSharedPtr det;
        for (const auto& d : cam.yolo_ring)
            if (!det || std::abs(rclcpp::Time(d->header.stamp).seconds() - rclcpp::Time(stamp).seconds()) <
                            std::abs(rclcpp::Time(det->header.stamp).seconds() - rclcpp::Time(stamp).seconds()))
                det = d;
        const bool ready = det && !det->detections.empty() && cam.has_depth_color_matrix && cam.has_aligned_depth_camera_info &&
                           std::abs(rclcpp::Time(det->header.stamp).seconds() - rclcpp::Time(stamp).seconds()) < yolo_leaf_max_age_;

        // "mask" method (see yolo_leaf_method_'s own comment): pixel-EXACT object membership from a YOLO
        // instance-segmentation node's own per-pixel label image — no 2D box/inflate heuristics needed to
        // find an instance's own pixels. Still needs the SAME depth-window filtering as the box method below
        // (depthModeValleyWindow(), shared) though: pixel membership is exact in IMAGE space, but at a
        // silhouette edge a background point can reproject into the mask's own footprint purely from
        // depth<->color parallax, despite genuinely being behind the object — see depthModeValleyWindow's own
        // comment. Empirically confirmed: without this filter, mask-derived leaves for a person were
        // routinely 1-3m deep (reprojected background bleeding in at the edges), worse than the box method's
        // own pre-valley-fix leaves.
        if (yolo_leaf_method_ == "mask") {
            sensor_msgs::msg::Image::ConstSharedPtr mask_img;
            for (const auto& m : cam.yolo_mask_ring)
                if (!mask_img || std::abs(rclcpp::Time(m->header.stamp).seconds() - rclcpp::Time(stamp).seconds()) <
                                    std::abs(rclcpp::Time(mask_img->header.stamp).seconds() - rclcpp::Time(stamp).seconds()))
                    mask_img = m;
            const bool mask_ready = ready && mask_img && mask_img->encoding == "32SC1" &&
                                    std::abs(rclcpp::Time(mask_img->header.stamp).seconds() - rclcpp::Time(stamp).seconds()) < yolo_leaf_max_age_;
            if (mask_ready) {
                // label (== the SAME id a Detection2DArray entry's own det.id carries, see
                // yolo_seg_track_detector.py's own mask_label comment) -> (id, class_idx), built once per tick.
                std::map<int32_t, std::pair<double, int>> byLabel;
                for (const auto& d : det->detections) {
                    int32_t label = 0;
                    try { label = static_cast<int32_t>(std::stol(d.id)); } catch (...) { continue; }
                    int class_idx = -1;
                    if (!d.results.empty()) {
                        const auto it = coco_class_to_idx_.find(d.results[0].hypothesis.class_id);
                        if (it != coco_class_to_idx_.end()) class_idx = it->second;
                    }
                    byLabel[label] = {static_cast<double>(label), class_idx};
                }
                // (camera-frame depth, point in depth-sensor frame) per label — depth kept per-point so the
                // same mode+valley window used by the box method can reject background bleed-through here too.
                std::map<int32_t, std::vector<std::pair<float, Eigen::Vector3f>>> ptsByLabel;
                const auto* data = reinterpret_cast<const int32_t*>(mask_img->data.data());
                const int w = static_cast<int>(mask_img->width), h = static_cast<int>(mask_img->height);
                const size_t stepElems = mask_img->step / sizeof(int32_t);
                for (const auto& pt : cloud.points) {
                    const Eigen::Vector4d pc = cam.T_color_depth * Eigen::Vector4d(pt.x, pt.y, pt.z, 1.0);
                    if (pc.z() <= 0.05) continue;
                    const int u = static_cast<int>(cam.fx_aligned * pc.x() / pc.z() + cam.cx_aligned);
                    const int v = static_cast<int>(cam.fy_aligned * pc.y() / pc.z() + cam.cy_aligned);
                    if (u < 0 || u >= w || v < 0 || v >= h) continue;
                    const int32_t label = data[static_cast<size_t>(v) * stepElems + static_cast<size_t>(u)];
                    if (label == 0 || byLabel.find(label) == byLabel.end()) continue;  // background, or a stale/stray label not in this tick's own detections
                    ptsByLabel[label].emplace_back(static_cast<float>(pc.z()), Eigen::Vector3f(pt.x, pt.y, pt.z));
                }
                for (auto& [label, pv] : ptsByLabel) {
                    if (static_cast<int>(pv.size()) < yolo_leaf_min_points_) continue;
                    std::vector<float> depths;
                    depths.reserve(pv.size());
                    for (const auto& dp : pv) depths.push_back(dp.first);
                    const auto [depthLo, depthHi] = depthModeValleyWindow(depths);
                    const auto& info = byLabel.at(label);
                    Leaf leaf{info.first, info.second, {}};
                    for (const auto& dp : pv)
                        if (dp.first >= depthLo && dp.first <= depthHi) {
                            const Eigen::Vector4d pg = T_global_depth * Eigen::Vector4d(dp.second.x(), dp.second.y(), dp.second.z(), 1.0);
                            leaf.pts.emplace_back(static_cast<float>(pg.x()), static_cast<float>(pg.y()), static_cast<float>(pg.z()));
                        }
                    if (static_cast<int>(leaf.pts.size()) >= yolo_leaf_min_points_) leaves.push_back(std::move(leaf));
                }
            }
        } else if (ready) {
            struct Box { double x0, x1, y0, y1, area; double id; int class_idx; };
            std::vector<Box> boxes;
            for (size_t i = 0; i < det->detections.size(); ++i) {
                const auto& d = det->detections[i];
                const double hw = d.bbox.size_x * 0.5 * (1.0 + 2.0 * yolo_leaf_box_inflate_), hh = d.bbox.size_y * 0.5 * (1.0 + 2.0 * yolo_leaf_box_inflate_);
                double id = static_cast<double>(i);
                try { id = std::stod(d.id); } catch (...) {}
                int class_idx = -1;
                if (!d.results.empty()) {
                    const auto it = coco_class_to_idx_.find(d.results[0].hypothesis.class_id);
                    if (it != coco_class_to_idx_.end()) class_idx = it->second;
                }
                boxes.push_back({d.bbox.center.position.x - hw, d.bbox.center.position.x + hw,
                                 d.bbox.center.position.y - hh, d.bbox.center.position.y + hh, 4.0 * hw * hh, id, class_idx});
            }
            std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) { return a.area < b.area; });  // smallest first
            std::vector<std::vector<std::pair<float, Eigen::Vector3f>>> inside(boxes.size());  // (depth, point in depth frame)
            for (const auto& pt : cloud.points) {
                const Eigen::Vector4d pc = cam.T_color_depth * Eigen::Vector4d(pt.x, pt.y, pt.z, 1.0);
                if (pc.z() <= 0.05) continue;
                const double u = cam.fx_aligned * pc.x() / pc.z() + cam.cx_aligned, v = cam.fy_aligned * pc.y() / pc.z() + cam.cy_aligned;
                for (size_t b = 0; b < boxes.size(); ++b)
                    if (u >= boxes[b].x0 && u <= boxes[b].x1 && v >= boxes[b].y0 && v <= boxes[b].y1) {
                        inside[b].push_back({static_cast<float>(pc.z()), Eigen::Vector3f(pt.x, pt.y, pt.z)});
                        break;  // a point belongs to the smallest box that contains it
                    }
            }
            for (size_t b = 0; b < boxes.size(); ++b) {
                auto& v = inside[b];
                if (static_cast<int>(v.size()) < yolo_leaf_min_points_) continue;
                std::vector<float> depths;
                depths.reserve(v.size());
                for (const auto& q : v) depths.push_back(q.first);
                // Partially-occluded detection (visible through/around something nearer within the same
                // inflated box, e.g. a person behind boxes) inevitably has SOME of the occluder's own nearer
                // pixels inside its box too — depthModeValleyWindow() isolates the dominant depth cluster
                // (see its own comment) instead of anchoring on the single nearest depth, which systematically
                // picked the occluder's surface instead of the detected object's own.
                const auto [depthLo, depthHi] = depthModeValleyWindow(depths);
                Leaf leaf{boxes[b].id, boxes[b].class_idx, {}};
                for (const auto& q : v)
                    if (q.first >= depthLo && q.first <= depthHi) {
                        const Eigen::Vector4d pg = T_global_depth * Eigen::Vector4d(q.second.x(), q.second.y(), q.second.z(), 1.0);
                        leaf.pts.emplace_back(static_cast<float>(pg.x()), static_cast<float>(pg.y()), static_cast<float>(pg.z()));
                    }
                if (static_cast<int>(leaf.pts.size()) >= yolo_leaf_min_points_) leaves.push_back(std::move(leaf));
            }
        }

        sensor_msgs::msg::PointCloud2 out;
        out.header.stamp = stamp;
        out.header.frame_id = global_frame_;
        sensor_msgs::PointCloud2Modifier mod(out);
        mod.setPointCloud2Fields(5, "x", 1, sensor_msgs::msg::PointField::FLOAT32, "y", 1, sensor_msgs::msg::PointField::FLOAT32,
                                 "z", 1, sensor_msgs::msg::PointField::FLOAT32, "track_id", 1, sensor_msgs::msg::PointField::FLOAT32,
                                 "class_idx", 1, sensor_msgs::msg::PointField::FLOAT32);
        size_t n = 0;
        for (const auto& l : leaves) n += l.pts.size();
        mod.resize(n);
        sensor_msgs::PointCloud2Iterator<float> ox(out, "x"), oy(out, "y"), oz(out, "z"), oid(out, "track_id"), ocls(out, "class_idx");
        for (const auto& l : leaves)
            for (const auto& q : l.pts) {
                *ox = q.x(); *oy = q.y(); *oz = q.z(); *oid = static_cast<float>(l.id); *ocls = static_cast<float>(l.class_idx);
                ++ox; ++oy; ++oz; ++oid; ++ocls;
            }
        cam.pub_semantic_leaves->publish(out);
    }

    void onNativeDepthCloud(CameraStream& cam, const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) {
        pcl::PointCloud<pcl::PointXYZ> raw_cloud;
        pcl::fromROSMsg(*msg, raw_cloud);

        pcl::PointCloud<pcl::PointXYZ> cloud;
        cloud.points.reserve(raw_cloud.points.size());
        pcl::PointCloud<pcl::PointXYZ> floor_cloud;
        const bool can_project = cam.pub_ground_points && cam.has_ground_mask && cam.has_depth_color_matrix &&
            cam.has_aligned_depth_camera_info && cam.aligned_width > 0 && cam.aligned_height > 0 &&
            !cam.latest_ground_mask.empty();
        // The mask is the COLOR image's resolution; camera_info's K is in
        // that same resolution's pixels — scale only if they ever differ.
        const double mask_sx = can_project ? static_cast<double>(cam.latest_ground_mask.cols) / cam.aligned_width : 1.0;
        const double mask_sy = can_project ? static_cast<double>(cam.latest_ground_mask.rows) / cam.aligned_height : 1.0;
        for (const auto& pt : raw_cloud.points) {
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
            // z is this point's depth along the optical axis, same quantity
            // processAlignedDepth's own per-pixel z value is — same bounds
            // apply for the same reason (too close/far to trust).
            if (pt.z < depth_min_ || pt.z > depth_max_) continue;
            cloud.points.push_back(pt);

            // Semantic ground: a native cloud is unorganized (no pixel grid
            // to index a 2D mask with), so instead PROJECT each point onto
            // the color image: depth-frame point -> color frame through the
            // factory extrinsic (T_color_depth: the color camera sits ~5.9 cm
            // from the depth one, ignoring it would shift every projected
            // pixel), then the color camera's own pinhole intrinsics, then
            // read the mask there. Same "latest mask, best-effort pairing"
            // semantics as the aligned-depth-image path.
            if (can_project) {
                const Eigen::Vector4d pc = cam.T_color_depth * Eigen::Vector4d(pt.x, pt.y, pt.z, 1.0);
                if (pc.z() > 0.05) {
                    const int u = static_cast<int>(std::lround((cam.fx_aligned * pc.x() / pc.z() + cam.cx_aligned) * mask_sx));
                    const int v = static_cast<int>(std::lround((cam.fy_aligned * pc.y() / pc.z() + cam.cy_aligned) * mask_sy));
                    if (u >= 0 && u < cam.latest_ground_mask.cols && v >= 0 && v < cam.latest_ground_mask.rows &&
                        cam.latest_ground_mask.at<uint8_t>(v, u) >= 128) {
                        floor_cloud.points.push_back(pt);
                    }
                }
            }
        }

        const std::string frame_id = !msg->header.frame_id.empty() ? msg->header.frame_id : cam.depth_optical_frame;
        publishSemanticLeaves(cam, cloud, msg->header.stamp, frame_id);
        finishDepthCloud(cam, cloud, msg->header.stamp, frame_id, /*is_native_depth_frame=*/true);
        publishGroundPoints(cam, floor_cloud, msg->header.stamp, frame_id, /*is_native_depth_frame=*/true);
    }

    // Shared tail end of both depth sources above: publish the raw
    // deprojected/native cloud as-is, voxel-downsample + republish
    // uncropped (static_structures_node's own input, see the file header),
    // then crop by floor/wall and publish the processed cloud.
    //
    // is_native_depth_frame: whether `cloud`'s points are expressed in this
    // camera's own depth_optical_frame (onNativeDepthCloud) rather than its
    // color_optical_frame (processAlignedDepth's aligned-depth deprojection,
    // and what cam.T_lidar_camera was actually calibrated against — see
    // ensureCameraCalibrationTf). When true, the calibrated global-frame
    // transform gets one extra step, composing with
    // cam.T_color_depth (the RealSense factory depth->color extrinsic,
    // cached in onDepthToColorExtrinsics) to bring a depth-frame point into
    // the SAME frame convention the calibration was computed against,
    // before applying it — otherwise every point would be off by that
    // factory extrinsic's translation (a few cm, not negligible for a
    // wall/floor margin test). The raw-TF-chain fallback (cam.T_base_depth)
    // needs no such correction: ensureBaseDepthTf() looks up base_frame_ ->
    // frame_id directly, whichever frame_id this call was actually given.
    // See depth_processed_voxel_size_'s own declare_parameter comment for
    // why this exists and what it does differently from pcl::VoxelGrid. Key
    // collisions across voxel cells are astronomically unlikely at LiDAR/
    // depth-camera scales (points stay within a few hundred meters at most),
    // so a plain hashed-index map is used instead of jo_detector's own
    // fixed-size array + localSensorRange_ bound (voxelFilter/posToAddress)
    // — simpler, and doesn't need a separate range parameter.
    //
    // Deliberately NOT a literal port of jo_detector's voxelFilter(),
    // which keeps a single RAW point per voxel — specifically whichever one
    // happens to arrive Nth in scan order once a voxel's running count hits
    // the threshold exactly. That point is arbitrary (an accident of pixel
    // scan order, not a geometric property) and still carries that one
    // sample's own sensor noise. This version keeps the actually useful part
    // — voxels with fewer than the threshold's worth of points are pure
    // noise/too-sparse-to-trust and are dropped entirely — but accumulates
    // every point in a voxel that DOES clear the threshold and emits their
    // centroid instead of one raw sample, denoising the emitted point and
    // removing the scan-order dependency. Two passes over the accumulator
    // instead of one over the points, but still linear.
    pcl::PointCloud<pcl::PointXYZ>::Ptr occupancyVoxelFilter(const pcl::PointCloud<pcl::PointXYZ>& in) const {
        auto out = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        if (depth_processed_voxel_size_ <= 0.0 || in.points.empty()) {
            *out = in;
            return out;
        }

        // Two range bands (range = distance from the camera, the cloud is in its own optical frame),
        // each with its own voxel size and minimum point count. Exact voxel key (no hash collisions):
        // 21 bits per axis, offset so negative cells are fine.
        struct VoxelAccum {
            Eigen::Vector3d sum{Eigen::Vector3d::Zero()};
            int count{0};
        };
        auto runBand = [&](bool far_band) {
            const double res = far_band ? depth_processed_voxel_size_far_ : depth_processed_voxel_size_;
            const int thresh = static_cast<int>(far_band ? depth_processed_voxel_occupied_thresh_far_
                                                         : depth_processed_voxel_occupied_thresh_);
            if (res <= 0.0) return;
            std::unordered_map<std::uint64_t, VoxelAccum> occupancy;
            occupancy.reserve(in.points.size());
            constexpr std::int64_t kOffset = 1 << 20;
            for (const auto& pt : in.points) {
                const double r = std::sqrt(static_cast<double>(pt.x) * pt.x + static_cast<double>(pt.y) * pt.y +
                                           static_cast<double>(pt.z) * pt.z);
                if ((r >= depth_processed_split_range_) != far_band) continue;
                const std::int64_t ix = static_cast<std::int64_t>(std::floor(pt.x / res)) + kOffset;
                const std::int64_t iy = static_cast<std::int64_t>(std::floor(pt.y / res)) + kOffset;
                const std::int64_t iz = static_cast<std::int64_t>(std::floor(pt.z / res)) + kOffset;
                const std::uint64_t key = (static_cast<std::uint64_t>(ix & 0x1FFFFF) << 42) |
                                          (static_cast<std::uint64_t>(iy & 0x1FFFFF) << 21) |
                                          static_cast<std::uint64_t>(iz & 0x1FFFFF);
                VoxelAccum& acc = occupancy[key];
                acc.sum += Eigen::Vector3d(pt.x, pt.y, pt.z);
                ++acc.count;
            }
            for (const auto& kv : occupancy) {
                if (kv.second.count < thresh) continue;
                const Eigen::Vector3d centroid = kv.second.sum / static_cast<double>(kv.second.count);
                out->points.emplace_back(
                    static_cast<float>(centroid.x()), static_cast<float>(centroid.y()), static_cast<float>(centroid.z()));
            }
        };
        out->points.reserve(in.points.size() / 4);
        runBand(false);
        runBand(true);
        out->width = static_cast<uint32_t>(out->points.size());
        out->height = 1;
        return out;
    }

    void finishDepthCloud(CameraStream& cam, pcl::PointCloud<pcl::PointXYZ>& cloud,
                           const builtin_interfaces::msg::Time& stamp, const std::string& frame_id,
                           bool is_native_depth_frame) {
        cloud.width = static_cast<uint32_t>(cloud.points.size());
        cloud.height = 1;

        sensor_msgs::msg::PointCloud2 out;
        pcl::toROSMsg(cloud, out);
        out.header.stamp = stamp;
        out.header.frame_id = frame_id;
        cam.pub_depth_cloud->publish(out);

        // ---- Downsampled + floor/wall-removed depth cloud ----
        // Voxel-downsample first (in the depth cloud's own frame — cheap,
        // frame-independent), then decide what to drop using a *global-frame
        // copy* of each surviving point; the published cloud itself stays in
        // frame_id throughout — only the removal decision needs the
        // global-frame position.
        pcl::PointCloud<pcl::PointXYZ>::Ptr depth_downsampled(new pcl::PointCloud<pcl::PointXYZ>());
        if (voxel_resolution_ > 0.0 && !cloud.points.empty()) {
            pcl::VoxelGrid<pcl::PointXYZ> voxel;
            voxel.setInputCloud(cloud.makeShared());
            voxel.setLeafSize(
                static_cast<float>(voxel_resolution_), static_cast<float>(voxel_resolution_), static_cast<float>(voxel_resolution_));
            voxel.filter(*depth_downsampled);
        } else {
            *depth_downsampled = cloud;
        }

        // Republish this SAME voxel grid, as-is (no crop) — the depth-side
        // equivalent of pub_lidar_voxelized_, and static_structures_node's own
        // per-camera input (see the file header and CameraStream's own
        // comment on pub_depth_voxelized). Zero extra compute: it's the
        // exact cloud just built above for this camera's own crop.
        {
            sensor_msgs::msg::PointCloud2 voxelized_out;
            pcl::toROSMsg(*depth_downsampled, voxelized_out);
            voxelized_out.header.stamp = stamp;
            voxelized_out.header.frame_id = frame_id;
            cam.pub_depth_voxelized->publish(voxelized_out);
        }

        // See computeGlobalDepthTransform()'s own comment for the full
        // reasoning behind this composition (calibrated T_lidar_camera
        // preferred, cam.T_base_depth as the fallback) — factored out from
        // here so publishGroundPoints() can reuse the exact same logic.
        Eigen::Matrix4d T_global_depth;
        const bool have_global_tf = computeGlobalDepthTransform(cam, frame_id, is_native_depth_frame, T_global_depth);

        // Debug-only global-frame republish (pub_depth_voxelized_global —
        // see that member's own comment on why). Same T_global_depth as the
        // crop decision below, just applied to every surviving voxel
        // instead of only used for a threshold test — no separate transform
        // to keep in sync.
        if (have_global_tf) {
            pcl::PointCloud<pcl::PointXYZ> voxelized_global;
            voxelized_global.points.reserve(depth_downsampled->points.size());
            for (const auto& pt : depth_downsampled->points) {
                const Eigen::Vector4d p_global4 = T_global_depth * Eigen::Vector4d(pt.x, pt.y, pt.z, 1.0);
                voxelized_global.points.emplace_back(
                    static_cast<float>(p_global4(0)), static_cast<float>(p_global4(1)), static_cast<float>(p_global4(2)));
            }
            voxelized_global.width = static_cast<uint32_t>(voxelized_global.points.size());
            voxelized_global.height = 1;
            sensor_msgs::msg::PointCloud2 voxelized_global_out;
            pcl::toROSMsg(voxelized_global, voxelized_global_out);
            voxelized_global_out.header.stamp = stamp;
            voxelized_global_out.header.frame_id = global_frame_;
            cam.pub_depth_voxelized_global->publish(voxelized_global_out);
        }

        // Density-gate filter for the processed/detection cloud specifically
        // — separate from depth_downsampled above (voxel_resolution_, stays
        // feeding pub_depth_voxelized only). Runs on the full-resolution
        // `cloud`, not on depth_downsampled — jo_detector's own
        // voxelFilter() likewise runs on the raw deprojected points, not a
        // pre-downsampled cloud.
        pcl::PointCloud<pcl::PointXYZ>::Ptr depth_detection = occupancyVoxelFilter(cloud);

        // Published in global_frame_ (odom), not frame_id — matching
        // pub_lidar_processed_'s own convention (onLidarOdom's Pass 5,
        // always global) instead of every other depth publisher above
        // (pub_depth_cloud/pub_depth_voxelized, deliberately left in
        // sensor frame — see their own comments). This is the one depth
        // output meant to feed the (not-yet-built) detection phase
        // directly, the same way pub_lidar_processed_ already does for
        // LiDAR — a detector consuming both needs them in the SAME frame
        // without independently re-deriving this camera's calibrated
        // transform itself. No have_global_tf fallback: without a global
        // transform there's no correct way to label these points
        // global_frame_ at all, so this cycle's cloud is just skipped
        // (same choice pub_depth_voxelized_global already makes) rather
        // than publish sensor-frame points mislabeled as global.
        if (have_global_tf) {
            pcl::PointCloud<pcl::PointXYZ> depth_final;
            depth_final.points.reserve(depth_detection->points.size());
            for (const auto& pt : depth_detection->points) {
                const Eigen::Vector4d p_global4 = T_global_depth * Eigen::Vector4d(pt.x, pt.y, pt.z, 1.0);
                const Eigen::Vector3d p_global = p_global4.head<3>();
                const double h = heightAboveGround(p_global);
                // Lower bound is wall_removal_margin_, not 0.0 — see the
                // matching comment on the LiDAR crop in onLidarOdom for why
                // a plain "h < 0" test leaves roughly half of the actual
                // floor-surface points in the cloud instead of dropping them.
                if (h < wall_removal_margin_ || h > ground_roof_offset_) continue;
                if (isNearAnyWall(p_global)) continue;
                depth_final.points.emplace_back(
                    static_cast<float>(p_global4(0)), static_cast<float>(p_global4(1)), static_cast<float>(p_global4(2)));
            }
            depth_final.width = static_cast<uint32_t>(depth_final.points.size());
            depth_final.height = 1;
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 10000,
                "[%s] depth point counts: input=%zu | plain voxel %.2fm (voxelized_sensor)=%zu | "
                "occupancy gate (voxel %.2fm, >=%d pts)=%zu | after floor/roof/wall crop (processed)=%zu",
                cam.name.c_str(), cloud.points.size(), voxel_resolution_, depth_downsampled->points.size(),
                depth_processed_voxel_size_, static_cast<int>(depth_processed_voxel_occupied_thresh_),
                depth_detection->points.size(), depth_final.points.size());

            sensor_msgs::msg::PointCloud2 processed_out;
            pcl::toROSMsg(depth_final, processed_out);
            processed_out.header.stamp = stamp;
            processed_out.header.frame_id = global_frame_;
            cam.pub_depth_processed->publish(processed_out);
        }
    }

    void onLidarOdom(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg,
                      const nav_msgs::msg::Odometry::ConstSharedPtr& odom_msg) {
        pub_scan_trigger_->publish(lidar_msg->header);
        queueDepthTargets(lidar_msg->header.stamp);
        if (!ensureStaticTf()) {
            return;
        }
        if (!odom_msg->child_frame_id.empty() && odom_msg->child_frame_id != base_frame_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "odom child_frame_id '%s' != configured base_frame '%s' — check odom_topic/base_frame",
                odom_msg->child_frame_id.c_str(), base_frame_.c_str());
        }
        if (!odom_msg->header.frame_id.empty() && odom_msg->header.frame_id != global_frame_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "odom header.frame_id '%s' != configured global_frame '%s' — check odom_topic/global_frame",
                odom_msg->header.frame_id.c_str(), global_frame_.c_str());
        }

        pcl::PointCloud<pcl::PointXYZ> raw_cloud;
        pcl::fromROSMsg(*lidar_msg, raw_cloud);

        // Pass 1 (sensor frame, cheap): validity + local range crop, then a deterministic
        // voxel-centroid in two range bands (lidar_voxel_near_ up to lidar_voxel_split_range_,
        // lidar_voxel_far_ beyond) -> local_pts, the general-purpose cloud's own thinning
        // (unrelated to detection, which moved out to static_structures_node — see the file header).
        // Each surviving point is the centroid (mean) of the raw points in its voxel; no minimum
        // point count is required.
        pcl::PointCloud<pcl::PointXYZ>::Ptr band_near(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr band_far(new pcl::PointCloud<pcl::PointXYZ>());
        for (const auto& pt : raw_cloud.points) {
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
            if (std::abs(pt.x) > lidar_range_x_ || std::abs(pt.y) > lidar_range_y_) continue;
            const double r = std::sqrt(static_cast<double>(pt.x) * pt.x + static_cast<double>(pt.y) * pt.y +
                                       static_cast<double>(pt.z) * pt.z);
            (r < lidar_voxel_split_range_ ? band_near : band_far)->points.push_back(pt);
        }
        std::vector<Eigen::Vector4d> local_pts;
        local_pts.reserve(band_near->points.size() + band_far->points.size());
        for (auto& band_leaf : {std::make_pair(band_near, lidar_voxel_near_), std::make_pair(band_far, lidar_voxel_far_)}) {
            auto band = band_leaf.first;
            if (band->points.empty()) continue;
            band->width = static_cast<uint32_t>(band->points.size());
            band->height = 1;
            pcl::PointCloud<pcl::PointXYZ> out;
            if (band_leaf.second > 0.0) {
                pcl::VoxelGrid<pcl::PointXYZ> voxel;
                voxel.setInputCloud(band);
                voxel.setLeafSize(static_cast<float>(band_leaf.second), static_cast<float>(band_leaf.second),
                                  static_cast<float>(band_leaf.second));
                voxel.filter(out);
            } else {
                out = *band;
            }
            for (const auto& pt : out.points) local_pts.emplace_back(pt.x, pt.y, pt.z, 1.0);
        }

        // Pass 2: voxel-downsample the RAW scan directly — no crop of any
        // kind, sensor frame — and publish it as-is. This is the ONLY
        // point static_structures_node's own LiDAR subscription reads from
        // (see the file header and that node's own header comment on why
        // sharing this one voxel-grid pass, instead of it re-acquiring and
        // re-voxelizing raw /velodyne_points itself, is the actual fix for
        // the wall-thickness bug this architecture replaced).
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr raw_voxelized(new pcl::PointCloud<pcl::PointXYZ>());
            const bool banded = !wall_input_voxel_sizes_.empty() &&
                                wall_input_band_edges_.size() + 1 == wall_input_voxel_sizes_.size();
            if ((banded || wall_input_height_margin_ > 0.0) && !raw_cloud.points.empty()) {
                const Eigen::Matrix4d T_gl = odomToGlobalBase(odom_msg) * T_base_lidar_;
                std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> bands;
                for (size_t b = 0; b < (banded ? wall_input_voxel_sizes_.size() : 1); ++b)
                    bands.emplace_back(new pcl::PointCloud<pcl::PointXYZ>());
                for (const auto& pt : raw_cloud.points) {
                    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
                    if (wall_input_height_margin_ > 0.0) {
                        const Eigen::Vector4d pg = T_gl * Eigen::Vector4d(pt.x, pt.y, pt.z, 1.0);
                        const double h = heightAboveGround(pg.head<3>());
                        if (h < wall_input_height_margin_ || h > ground_roof_offset_ - wall_input_height_margin_) continue;
                    }
                    size_t b = 0;
                    if (banded) {
                        const double r = std::sqrt(static_cast<double>(pt.x) * pt.x + static_cast<double>(pt.y) * pt.y +
                                                   static_cast<double>(pt.z) * pt.z);
                        while (b < wall_input_band_edges_.size() && r >= wall_input_band_edges_[b]) ++b;
                    }
                    bands[b]->points.push_back(pt);
                }
                for (size_t b = 0; b < bands.size(); ++b) {
                    if (bands[b]->points.empty()) continue;
                    bands[b]->width = static_cast<uint32_t>(bands[b]->points.size());
                    bands[b]->height = 1;
                    const double leaf = banded ? wall_input_voxel_sizes_[b] : voxel_resolution_;
                    if (leaf > 0.0) {
                        pcl::PointCloud<pcl::PointXYZ> out;
                        pcl::VoxelGrid<pcl::PointXYZ> voxel;
                        voxel.setInputCloud(bands[b]);
                        voxel.setLeafSize(static_cast<float>(leaf), static_cast<float>(leaf), static_cast<float>(leaf));
                        voxel.filter(out);
                        raw_voxelized->points.insert(raw_voxelized->points.end(), out.points.begin(), out.points.end());
                    } else {
                        raw_voxelized->points.insert(raw_voxelized->points.end(), bands[b]->points.begin(), bands[b]->points.end());
                    }
                }
                raw_voxelized->width = static_cast<uint32_t>(raw_voxelized->points.size());
                raw_voxelized->height = 1;
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 10000,
                    "wall input: raw=%zu -> voxelized_sensor=%zu (height margin %.2f, %zu range bands)",
                    raw_cloud.points.size(), raw_voxelized->points.size(), wall_input_height_margin_, bands.size());
            } else if (voxel_resolution_ > 0.0 && !raw_cloud.points.empty()) {
                pcl::VoxelGrid<pcl::PointXYZ> voxel;
                voxel.setInputCloud(raw_cloud.makeShared());
                voxel.setLeafSize(
                    static_cast<float>(voxel_resolution_), static_cast<float>(voxel_resolution_), static_cast<float>(voxel_resolution_));
                voxel.filter(*raw_voxelized);
            } else {
                *raw_voxelized = raw_cloud;
            }
            sensor_msgs::msg::PointCloud2 voxelized_out;
            pcl::toROSMsg(*raw_voxelized, voxelized_out);
            voxelized_out.header.stamp = lidar_msg->header.stamp;
            voxelized_out.header.frame_id = lidar_msg->header.frame_id.empty() ? lidar_frame_ : lidar_msg->header.frame_id;
            pub_lidar_voxelized_->publish(voxelized_out);
        }

        // The transform itself is one matrix multiply per point (T_global_base
        // is already in the synced odom message; T_base_lidar_ is cached),
        // not a TF lookup per point or even per callback.
        const Eigen::Matrix4d T_global_base = odomToGlobalBase(odom_msg);
        const Eigen::Matrix4d T_global_lidar = T_global_base * T_base_lidar_;

        // Cache for processAlignedDepth to reuse (see the member comment) —
        // depth arrives faster than LiDAR and unsynced, so it can't get its
        // own fresh T_global_base this way; this is at most one LiDAR
        // period stale for it, which is fine for a floor/wall crop.
        T_global_base_latest_ = T_global_base;
        has_odom_ = true;

        // Pass 3 (global frame, one combined loop): transform local_pts +
        // slope-aware floor/roof crop together, using heightAboveGround()
        // (bounded to [wall_removal_margin_, ground_roof_offset_]) — the
        // MINIMUM signed distance across every camera's own, possibly-tilted
        // plane received from static_structures_node (onWallMarkers()), not fit
        // here; per explicit request, the ground is the UNION of the space
        // below each camera's plane, not one shared/merged plane — see that
        // method's own comment. The lower bound is wall_removal_margin_, NOT
        // 0.0: a real floor-surface point's signedDistance sits close to
        // zero on EITHER side (RANSAC fits the plane roughly through its
        // inliers, not strictly above them), so a plain "h < 0" test only
        // drops the roughly-half of floor points that end up marginally
        // below the fit and keeps the rest — this margin instead drops
        // anything close to the floor plane at all, symmetric to
        // isNearAnyWall's own margin-gated wall strip below.
        pcl::PointCloud<pcl::PointXYZ> global_cloud;
        global_cloud.points.reserve(local_pts.size());
        for (const auto& p : local_pts) {
            const Eigen::Vector4d p_global4 = T_global_lidar * p;
            const Eigen::Vector3d p_global = p_global4.head<3>();
            const double h = heightAboveGround(p_global);
            if (h < wall_removal_margin_ || h > ground_roof_offset_) continue;
            global_cloud.points.emplace_back(
                static_cast<float>(p_global.x()), static_cast<float>(p_global.y()), static_cast<float>(p_global.z()));
        }
        global_cloud.width = static_cast<uint32_t>(global_cloud.points.size());
        global_cloud.height = 1;

        // Pass 4: voxel downsample the general-purpose cloud — adaptive leaf
        // size, matching jo_detector's dynamicDetector::lidarPoseCB/
        // lidarOdomCB (see lidar_processed_voxel_size_'s own declare_parameter
        // comment): only runs at all if still above the point budget, and
        // escalates the leaf size until back under it.
        pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled = global_cloud.makeShared();
        if (lidar_processed_voxel_size_ > 0.0) {
            float leaf = static_cast<float>(lidar_processed_voxel_size_);
            pcl::VoxelGrid<pcl::PointXYZ> voxel;
            while (static_cast<int>(downsampled->size()) > lidar_processed_downsample_threshold_) {
                voxel.setInputCloud(downsampled);
                voxel.setLeafSize(leaf, leaf, leaf);
                pcl::PointCloud<pcl::PointXYZ>::Ptr next(new pcl::PointCloud<pcl::PointXYZ>());
                voxel.filter(*next);
                downsampled = next;
                leaf *= 1.1f;
            }
        }

        // Pass 5: strip points near a wall_planes_global_ entry (also
        // received from static_structures_node), then publish.
        pcl::PointCloud<pcl::PointXYZ> final_cloud;
        final_cloud.points.reserve(downsampled->points.size());
        for (const auto& pt : downsampled->points) {
            if (isNearAnyWall(Eigen::Vector3d(pt.x, pt.y, pt.z))) continue;
            final_cloud.points.push_back(pt);
        }
        final_cloud.width = static_cast<uint32_t>(final_cloud.points.size());
        final_cloud.height = 1;

        sensor_msgs::msg::PointCloud2 out;
        pcl::toROSMsg(final_cloud, out);
        out.header.stamp = lidar_msg->header.stamp;
        out.header.frame_id = global_frame_;
        pub_lidar_processed_->publish(out);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PreprocessingNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
