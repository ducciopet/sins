/*
    FILE: dbscan_detector_node.cpp
    ---------------------------------
    DBSCAN obstacle detection on the LiDAR cloud and on every depth camera's
    cloud, producing 3D oriented bounding boxes (OBB), and FUSING the sources:
    each source's boxes are still published separately (to check them one by
    one), and the fused set — one box per physical object, LiDAR + cameras —
    is published on .../fused/*. No tracking yet.

    Fusion (fuseAndPublish): a pending "tick" holds the LiDAR result of one scan
    and gathers each camera's result (their stamps agree within
    fusion_stamp_tolerance; a camera cloud arrives ~45 ms after the LiDAR one,
    so the tick waits for all cameras or fusion_timeout_sec). Then:
      1. One-to-one MUTUAL best match between sources (like jo_detector's
         BboxesMerger, but on rotated footprints): two boxes of different
         sources are the same object only if each is the other's best
         3D-IoU partner in that source and IoU >= fusion_iou_thresh (0.6, as
         jo_detector's lidar_visual_filtering_BBox_IOU_threshold; peers within
         one source use fusion_iou_same 0.4 = samegroupIOU_threshold).
         A big box can never swallow two neighbours through a chain.
      2. NESTING as a parent/child hierarchy. A group nested in a bigger one
         (volume IOV >= fusion_iov_cross across sensors, fusion_iov_same within one) is
         its child only with evidence of being the same object: it brings a
         source the parent lacks, or its points are attached to the parent's
         (closest pair <= fusion_nest_attach_dist — a fragment of it). A small
         object in the empty interior of a bigger box (notch of an L-shaped
         cluster) is NOT a child and stays separate. Each child takes its best
         parent; parents are resolved bottom-up: one child is absorbed; two or
         more mutually disjoint children mean the parent was an over-merged
         cluster and it is replaced by its leaves (fusion_nested_multi_child).
    Whenever groups are merged: union of the points, range-banded voxel
    centroid (fusion_voxel_*, the LiDAR's own values, so the density is not
    changed by the union) and an OBB refit on that union (not a union of boxes,
    which would inflate an oriented box). Boxes seen by one source only pass
    through untouched. No tracking yet.
*/
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <functional>
#include <deque>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "sins/box_fusion.hpp"
#include "sins/cluster_obb.hpp"

using sins::Obb;

namespace {

struct SourceResult {
    std::vector<Obb> boxes;
    std::vector<std::vector<int>> box_points;   // indices into pts of each kept box's cluster
    std::vector<std::string> box_class;         // YOLO class name per box, only set for "<camera>_yolo" leaf sources (else empty)
    std::vector<int> box_track_id;               // YOLO 2D (ByteTrack) track id per box, only set for "<camera>_yolo" leaf sources (else -1) — see onLeaves()'s own comment on why this used to be discarded
    std::vector<Eigen::Vector3f> pts;           // all input points
    std::vector<int> point_cluster;             // cluster index per point, -1 = none/dropped
    size_t num_points = 0;
    size_t num_clusters = 0;
    size_t flat_boxes = 0;    // boxes lower than 0.2 m but longer than 1 m (a wall split into one box per ring)
    size_t large_boxes = 0;   // boxes with a footprint side > 1.5 m (fusion indicator)
    size_t splits = 0;        // clusters split into a tall column + the low rest
    double ms = 0.0;
    bool valid = false;
};

// Range-adaptive DBSCAN radius: eps grows linearly from `near` (at the sensor) to `far` (at range_far
// and beyond), following the LiDAR's spacing that grows with distance. near == far = constant eps.
struct EpsRange {
    double near = 0.2, far = 0.2, range_far = 10.0;
    // Vertical radius: at least the horizontal eps, and at least margin * (vertical spacing of the
    // sensor's rings at that range) when vert_angle_deg > 0 (0 = isotropic).
    double vert_angle_deg = 0.0, vert_margin = 1.4;
    float at(double r) const {
        const double t = range_far > 0.0 ? std::min(1.0, std::max(0.0, r / range_far)) : 1.0;
        return static_cast<float>(near + (far - near) * t);
    }
    float atZ(double r) const {
        const float xy = at(r);
        if (vert_angle_deg <= 0.0) return xy;
        return std::max(xy, static_cast<float>(vert_margin * r * std::tan(vert_angle_deg * M_PI / 180.0)));
    }
};

// Geometric split of an over-merged cluster ("a person walking past some boxes"): a tall column sticking out of a low plateau.
struct SplitParams {
    bool enabled = false;
    double min_footprint = 0.8;   // [m^2] only clusters whose box footprint is at least this big are examined
    double split_height = 0.7;    // [m] height above the local ground separating the low plateau from the tall part
    double tall_height = 1.0;     // [m] a part above split_height is a tall object only if it reaches this height
    double column_margin = 0.15;  // [m] low points closer than this (in the horizontal plane) to a tall part belong to its column
};

struct FloorPlane {
    Eigen::Vector3d point;   // a point on the plane (global frame)
    Eigen::Vector3d normal;  // unit, pointing up
};

struct Stats {
    double ms = 0, points = 0, clusters = 0, boxes = 0, large = 0, flat = 0, splits = 0;
    int n = 0;
};

}  // namespace

class DbscanDetectorNode : public rclcpp::Node {
public:
    DbscanDetectorNode() : Node("dbscan_detector_node") {
        global_frame_ = declare_parameter<std::string>("global_frame", "odom");
        camera_names_ = declare_parameter<std::vector<std::string>>("camera_names", {"front_camera", "back_camera"});
        lidar_topic_ = declare_parameter<std::string>("lidar_topic", "/sins/lidar/processed");
        camera_topic_template_ = declare_parameter<std::string>(
            "camera_topic_template", "/sins/{camera}/depth/processed");

        const std::string obb_mode = declare_parameter<std::string>("obb_mode", "pca3d");
        yaw_only_ = (obb_mode == "yaw");
        if (obb_mode != "yaw" && obb_mode != "pca3d")
            RCLCPP_WARN(get_logger(), "obb_mode '%s' unknown, using pca3d", obb_mode.c_str());

        lidar_eps_.near = declare_parameter<double>("lidar_eps_near", 0.12);
        lidar_eps_.far = declare_parameter<double>("lidar_eps_far", 0.3);
        lidar_eps_.range_far = declare_parameter<double>("lidar_eps_range_far", 15.0);
        lidar_eps_.vert_angle_deg = declare_parameter<double>("lidar_eps_vertical_angle_deg", 0.0);
        lidar_eps_.vert_margin = declare_parameter<double>("lidar_eps_vertical_margin", 1.4);
        lidar_min_points_ = declare_parameter<int>("lidar_min_points", 8);
        camera_eps_.near = declare_parameter<double>("camera_eps_near", 0.1);
        camera_eps_.far = declare_parameter<double>("camera_eps_far", 0.1);
        camera_eps_.range_far = declare_parameter<double>("camera_eps_range_far", 5.0);
        odom_topic_ = declare_parameter<std::string>("odom_topic", "/odometry/filtered");
        camera_min_points_ = declare_parameter<int>("camera_min_points", 8);

        const auto max_ext = declare_parameter<std::vector<double>>("max_box_extent", {3.0, 3.0, 2.5});
        if (max_ext.size() == 3) max_extent_ = Eigen::Vector3d(max_ext[0], max_ext[1], max_ext[2]);
        max_volume_ = declare_parameter<double>("max_box_volume", 12.0);
        min_volume_ = declare_parameter<double>("min_box_volume", 1e-3);
        stats_period_ = declare_parameter<double>("stats_log_period_sec", 5.0);

        pub_markers_ = create_publisher<visualization_msgs::msg::MarkerArray>(
            "/sins/dbscan_detector/boxes_markers", rclcpp::QoS(10));
        lidar_pub_ = create_publisher<vision_msgs::msg::Detection3DArray>("/sins/dbscan_detector/lidar/detections3d", rclcpp::QoS(10));
        lidar_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/sins/dbscan_detector/lidar/clusters", rclcpp::SensorDataQoS());

        marker_lifetime_sec_ = declare_parameter<double>("marker_lifetime_sec", 0.0);
        // Same class list preprocessing_node loads (cfg/coco.names by default) to turn the semantic leaves'
        // numeric class_idx back into a name — see onLeaves() and this node's own header.
        {
            const std::string class_names_path = ament_index_cpp::get_package_share_directory("sins") +
                "/" + declare_parameter<std::string>("yolo_class_names_path", "cfg/coco.names");
            std::ifstream f(class_names_path);
            std::string line;
            while (std::getline(f, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (!line.empty()) class_names_.push_back(line);
            }
            RCLCPP_INFO(get_logger(), "Loaded %zu YOLO class names from %s", class_names_.size(), class_names_path.c_str());
        }
        split_.enabled = declare_parameter<bool>("split_enabled", false);
        split_.min_footprint = declare_parameter<double>("split_min_footprint", 0.8);
        split_.split_height = declare_parameter<double>("split_height", 0.7);
        split_.tall_height = declare_parameter<double>("split_tall_height", 1.0);
        split_.column_margin = declare_parameter<double>("split_column_margin", 0.15);
        yolo_refine_enabled_ = declare_parameter<bool>("yolo_refine_enabled", true);
        // 0.2 was picked assuming a reasonably tight 3D match between a leaf and its true fused object; measured
        // against this bag, a CORRECT match for a non-rectangular object (a person, via a plain 2D box with no
        // instance segmentation — the box's rectangle always includes some same-depth background around the
        // actual silhouette) tops out around 0.15-0.3 IoU even with the valley-bounded depth extraction above,
        // not the near-1.0 a tight match would otherwise suggest. The REAL correctness guard here is the
        // distance-sanity check below (candidate center must be close to the fused object's own), not this
        // threshold — so it's lowered to match what a genuinely correct match actually achieves instead of
        // rejecting it outright.
        yolo_refine_min_iou_ = declare_parameter<double>("yolo_refine_min_iou", 0.12);
        yolo_refine_margin_ = declare_parameter<double>("yolo_refine_margin", 0.10);
        // Misattribution guard: when a person stands close to another object, its fused box can end up a
        // near-tie with the person's own fused box for IoU against the YOLO leaf (imperfect box fit, partial
        // occlusion, ...) — refining the wrong one is exactly how a NEARBY GENERIC OBJECT inherits the semantic
        // class, at the detection/fusion level rather than by any track-association mistake downstream in
        // tracker_node. Two independent checks below, both "skip this tick" (never refine the wrong object,
        // even at the cost of an occasional missed refinement):
        //  - min IoU margin over the runner-up candidate (a real near-tie is ambiguous, not a clear winner)
        //  - the leaf's own box and the chosen fused box must be near-coincident (they are built from
        //    overlapping views of the SAME physical object) — a "best" that is really just an adjacent object
        //    is rejected even when it is the ONLY candidate above yolo_refine_min_iou_.
        // Lowered alongside yolo_refine_min_iou above, proportionately: an 0.08 ABSOLUTE margin was a large
        // fraction of a ~0.2+ typical IoU, but is disproportionately strict now that genuinely correct matches
        // often only reach ~0.15-0.3 — a real competitor's IoU and the true match's can both be modest in that
        // range without the pairing actually being ambiguous.
        yolo_refine_min_iou_margin_ = declare_parameter<double>("yolo_refine_min_iou_margin", 0.05);
        // Retention-fraction guard (v1 has no cut at all, so no direct equivalent — but it carries the same
        // spirit as dynamicDetector.cpp's own yoloPointFractionThresh_/sparseLargeBox checks: don't trust a
        // match built from only a SLIVER of the evidence). The cut below keeps only the fused object's own
        // points that fall inside the leaf's rotated box; camera_min_points_ is an ABSOLUTE floor on that
        // "inside" count, with no check on how much of the PRE-CUT object it actually represents. A fused
        // object with plenty of points pre-cut can still pass that absolute floor while keeping only a thin,
        // barely-representative corner of itself — observed on a live bag (negative-X "person" misclassification
        // report): a healthy ~200-point fused object intermittently produced sub-20-point "inside" sets a few
        // ticks in a row, each one individually clearing camera_min_points_, that the tracker then received as
        // that tick's own detection size — this is what let it "follow" a nearby, mostly-unrelated sliver of
        // points instead of its own well-formed object. Requiring a minimum RETENTION fraction in addition to
        // the absolute floor rejects a marginal/coincidental overlap outright rather than accepting it as a
        // valid, if small, match.
        yolo_refine_min_retention_ = declare_parameter<double>("yolo_refine_min_retention", 0.35);
        yolo_keep_rest_ = declare_parameter<bool>("yolo_refine_keep_rest", true);
        leaves_enabled_ = declare_parameter<bool>("leaves_enabled", true);
        leaves_topic_template_ = declare_parameter<std::string>("leaves_topic_template", "/sins/{camera}/semantic_leaves");
        const auto fc = declare_parameter<std::vector<double>>("fused_color", {1.0, 1.0, 0.0});
        if (fc.size() == 3) fused_color_ = {static_cast<float>(fc[0]), static_cast<float>(fc[1]), static_cast<float>(fc[2])};
        fusion_enabled_ = declare_parameter<bool>("fusion_enabled", true);
        fusion_footprint_metric_ = declare_parameter<std::string>("fusion_metric", "footprint") == "footprint";
        fusion_z_overlap_ = declare_parameter<double>("fusion_z_overlap", 0.3);
        fusion_nest_thresh_ = declare_parameter<double>("fusion_nest_thresh", 0.8);
        fusion_nest_z_ = declare_parameter<double>("fusion_nest_z", 0.6);
        fusion_iou_thresh_ = declare_parameter<double>("fusion_iou_thresh", 0.2);
        fusion_iou_same_ = declare_parameter<double>("fusion_iou_same", 0.0);
        fusion_iov_cross_ = declare_parameter<double>("fusion_iov_cross", 0.7);
        fusion_iov_same_ = declare_parameter<double>("fusion_iov_same", 0.5);
        fusion_margin_ = declare_parameter<double>("fusion_margin", 0.05);
        fusion_stamp_tolerance_ = declare_parameter<double>("fusion_stamp_tolerance", 0.08);
        fusion_timeout_sec_ = declare_parameter<double>("fusion_timeout_sec", 0.12);
        fusion_voxel_near_ = declare_parameter<double>("fusion_voxel_near", 0.05);
        fusion_voxel_far_ = declare_parameter<double>("fusion_voxel_far", 0.03);
        fusion_voxel_split_ = declare_parameter<double>("fusion_voxel_split_range", 6.0);
        fusion_nest_attach_dist_ = declare_parameter<double>("fusion_nest_attach_dist", 0.4);
        fusion_child_overlap_ = declare_parameter<double>("fusion_child_overlap", 0.1);
        fusion_nested_leaves_ = declare_parameter<std::string>("fusion_nested_multi_child", "leaves") == "leaves";
        fused_pub_ = create_publisher<vision_msgs::msg::Detection3DArray>("/sins/dbscan_detector/fused/detections3d", rclcpp::QoS(10));
        fused_markers_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/sins/dbscan_detector/fused/boxes_markers", rclcpp::QoS(10));
        fused_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/sins/dbscan_detector/fused/clusters", rclcpp::SensorDataQoS());
        fusion_timer_ = create_wall_timer(std::chrono::milliseconds(20), [this] { tryFuse(); });

        int color_idx = 0;
        for (const auto& name : camera_names_) {
            ++color_idx;
            Camera cam;
            cam.name = name;
            cam.eps_scale = declare_parameter<double>(name + ".eps_scale", 1.0);
            cam.pub = create_publisher<vision_msgs::msg::Detection3DArray>("/sins/dbscan_detector/" + name + "/detections3d", rclcpp::QoS(10));
            cam.cloud_pub = create_publisher<sensor_msgs::msg::PointCloud2>("/sins/dbscan_detector/" + name + "/clusters", rclcpp::SensorDataQoS());
            std::string topic = camera_topic_template_;
            const auto pos = topic.find("{camera}");
            if (pos != std::string::npos) topic.replace(pos, 8, name);
            cameras_.emplace(name, std::move(cam));
            auto* cam_ptr = &cameras_.at(name);
            cam_ptr->sub = create_subscription<sensor_msgs::msg::PointCloud2>(
                topic, rclcpp::SensorDataQoS(),
                [this, cam_ptr, color_idx](const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) { onCameraCloud(*cam_ptr, color_idx, msg); });
            if (leaves_enabled_) {
                std::string ltopic = leaves_topic_template_;
                const auto lp = ltopic.find("{camera}");
                if (lp != std::string::npos) ltopic.replace(lp, 8, name);
                cam_ptr->leaves_sub = create_subscription<sensor_msgs::msg::PointCloud2>(
                    ltopic, rclcpp::QoS(10),
                    [this, cam_ptr, color_idx](const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) { onLeaves(*cam_ptr, color_idx + 3, msg); });
            }
        }
        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            odom_topic_, rclcpp::QoS(10), [this](const nav_msgs::msg::Odometry::ConstSharedPtr msg) {
                origin_ = Eigen::Vector3d(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
            });
        wall_markers_sub_ = create_subscription<visualization_msgs::msg::MarkerArray>(
            "/sins/static_structures/wall_markers", rclcpp::QoS(10),
            [this](const visualization_msgs::msg::MarkerArray::ConstSharedPtr msg) { onWallMarkers(msg); });
        lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            lidar_topic_, rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) { onLidar(msg); });

        last_log_ = now();
        RCLCPP_INFO(get_logger(), "dbscan_detector_node ready: lidar eps %.2f->%.2f (over %.0f m) minPts=%d | camera eps %.2f->%.2f (over %.0f m) minPts=%d | obb=%s",
                    lidar_eps_.near, lidar_eps_.far, lidar_eps_.range_far, lidar_min_points_,
                    camera_eps_.near, camera_eps_.far, camera_eps_.range_far, camera_min_points_, yaw_only_ ? "yaw" : "pca3d");
    }

private:
    struct Camera {
        std::string name;
        double eps_scale = 1.0;  // per-camera multiplier on camera_eps_near/far
        rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr pub;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub;
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub;
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr leaves_sub;
        double leaves_seen = -1e9;  // wall-clock time of the last leaves message (a source only counts while it is alive)
    };

    // static_structures_node encodes each camera's floor plane as an ns="floor" marker: position = a
    // point on the plane, local +Z axis of the orientation = its normal.
    void onWallMarkers(const visualization_msgs::msg::MarkerArray::ConstSharedPtr& msg) {
        std::vector<FloorPlane> planes;
        for (const auto& m : msg->markers) {
            if (m.ns != "floor" || m.action != visualization_msgs::msg::Marker::ADD) continue;
            const Eigen::Quaterniond q(m.pose.orientation.w, m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z);
            Eigen::Vector3d n = q.normalized().toRotationMatrix().col(2);
            if (n.z() < 0.0) n = -n;
            planes.push_back({Eigen::Vector3d(m.pose.position.x, m.pose.position.y, m.pose.position.z), n});
        }
        if (!planes.empty()) floor_planes_ = std::move(planes);
    }

    static double stampSec(const builtin_interfaces::msg::Time& t) { return t.sec + t.nanosec * 1e-9; }

    // Local "up" for a cluster: the normal of the floor plane (from static_structures_node) whose
    // reference point is nearest to the cluster, world +Z when no plane is known. On a slope the
    // yaw box then stands on the slope instead of being cut by the world horizontal.
    static Eigen::Vector3d localUp(const std::vector<FloorPlane>& planes, const Eigen::Vector3d& c) {
        const FloorPlane* best = nullptr;
        double best_d = std::numeric_limits<double>::max();
        for (const auto& pl : planes) {
            const double d = (pl.point - c).squaredNorm();
            if (d < best_d) { best_d = d; best = &pl; }
        }
        return best ? best->normal : Eigen::Vector3d::UnitZ();
    }

    // Split a cluster that is a tall part sticking out of a low plateau (a person next to boxes, a pole on a pallet). Heights
    // are measured above the nearest floor plane along its normal. The points above split_height are clustered on their own
    // (DBSCAN, same radii); each component that reaches tall_height is a tall object and takes every point of the cluster
    // within column_margin of it in the horizontal plane (its column, including the low points at its feet); what is left is
    // the low rest, clustered again. If there is no tall component, or nothing (or too little) is left, the cluster is
    // returned unchanged.
    static std::vector<std::vector<int>> splitTallColumns(const std::vector<Eigen::Vector3f>& pts, const std::vector<int>& idx,
                                                          const std::vector<float>& eps_i, const std::vector<float>& epsz_i, int min_pts,
                                                          const std::vector<FloorPlane>& planes, const SplitParams& sp) {
        std::vector<std::vector<int>> whole{idx};
        if (static_cast<int>(idx.size()) < 3 * min_pts) return whole;
        Eigen::Vector3d c = Eigen::Vector3d::Zero();
        for (int i : idx) c += pts[i].cast<double>();
        c /= static_cast<double>(idx.size());
        const FloorPlane* plane = nullptr;
        double bd = std::numeric_limits<double>::max();
        for (const auto& pl : planes) {
            const double d = (pl.point - c).squaredNorm();
            if (d < bd) { bd = d; plane = &pl; }
        }
        if (!plane) return whole;
        double xmin = 1e9, xmax = -1e9, ymin = 1e9, ymax = -1e9, hmax = -1e9;
        std::vector<double> h(idx.size());
        for (size_t k = 0; k < idx.size(); ++k) {
            const Eigen::Vector3d p = pts[idx[k]].cast<double>();
            h[k] = (p - plane->point).dot(plane->normal);
            hmax = std::max(hmax, h[k]);
            xmin = std::min(xmin, p.x()); xmax = std::max(xmax, p.x());
            ymin = std::min(ymin, p.y()); ymax = std::max(ymax, p.y());
        }
        if (hmax < sp.tall_height || (xmax - xmin) * (ymax - ymin) < sp.min_footprint) return whole;

        std::vector<int> high;  // indices (into idx) above split_height
        for (size_t k = 0; k < idx.size(); ++k) if (h[k] >= sp.split_height) high.push_back(static_cast<int>(k));
        if (static_cast<int>(high.size()) < min_pts) return whole;
        std::vector<Eigen::Vector3f> hp;
        std::vector<float> he, hez;
        for (int k : high) { hp.push_back(pts[idx[k]]); he.push_back(eps_i[idx[k]]); hez.push_back(epsz_i[idx[k]]); }
        const auto comps = sins::dbscanGrid(hp, he, hez, min_pts);

        std::vector<std::vector<int>> tall;  // points of each tall component (indices into idx)
        std::vector<std::vector<int>> tall_high;
        for (const auto& comp : comps) {
            double top = -1e9;
            for (int q : comp) top = std::max(top, h[static_cast<size_t>(high[static_cast<size_t>(q)])]);
            if (top < sp.tall_height) continue;
            tall_high.emplace_back();
            for (int q : comp) tall_high.back().push_back(high[static_cast<size_t>(q)]);
        }
        if (tall_high.empty()) return whole;
        tall.assign(tall_high.size(), {});
        std::vector<char> taken(idx.size(), 0);
        const double m2 = sp.column_margin * sp.column_margin;
        for (size_t t = 0; t < tall_high.size(); ++t)
            for (size_t k = 0; k < idx.size(); ++k) {
                if (taken[k]) continue;
                const Eigen::Vector3f& p = pts[idx[k]];
                for (int q : tall_high[t]) {
                    const Eigen::Vector3f& r = pts[idx[static_cast<size_t>(q)]];
                    if ((p.x() - r.x()) * (p.x() - r.x()) + (p.y() - r.y()) * (p.y() - r.y()) <= m2) {
                        tall[t].push_back(static_cast<int>(k));
                        taken[k] = 1;
                        break;
                    }
                }
            }
        std::vector<int> rest;
        for (size_t k = 0; k < idx.size(); ++k) if (!taken[k]) rest.push_back(static_cast<int>(k));
        if (static_cast<int>(rest.size()) < min_pts) return whole;  // nothing worth keeping apart from the tall part(s)

        std::vector<std::vector<int>> out;
        for (auto& t : tall) {
            if (static_cast<int>(t.size()) < min_pts) return whole;
            std::vector<int> g;
            for (int k : t) g.push_back(idx[static_cast<size_t>(k)]);
            out.push_back(std::move(g));
        }
        std::vector<Eigen::Vector3f> rp;
        std::vector<float> re, rez;
        for (int k : rest) { rp.push_back(pts[idx[static_cast<size_t>(k)]]); re.push_back(eps_i[idx[static_cast<size_t>(k)]]); rez.push_back(epsz_i[idx[static_cast<size_t>(k)]]); }
        const auto rest_clusters = sins::dbscanGrid(rp, re, rez, min_pts);
        if (rest_clusters.empty()) return whole;
        for (const auto& rc : rest_clusters) {
            std::vector<int> g;
            for (int q : rc) g.push_back(idx[static_cast<size_t>(rest[static_cast<size_t>(q)])]);
            out.push_back(std::move(g));
        }
        return out;
    }

    static SourceResult cluster(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud, const EpsRange& eps, int min_pts,
                                const Eigen::Vector3d& origin,
                                bool yaw_only, const Eigen::Vector3d& max_extent, double max_volume, double min_volume,
                                const std::vector<FloorPlane>& planes, const SplitParams& split) {
        SourceResult res;
        if (!cloud) return res;
        const auto t0 = std::chrono::steady_clock::now();
        res.valid = true;
        res.pts.reserve(static_cast<size_t>(cloud->width) * cloud->height);
        sensor_msgs::PointCloud2ConstIterator<float> ix(*cloud, "x"), iy(*cloud, "y"), iz(*cloud, "z");
        for (; ix != ix.end(); ++ix, ++iy, ++iz)
            if (std::isfinite(*ix) && std::isfinite(*iy) && std::isfinite(*iz)) res.pts.emplace_back(*ix, *iy, *iz);
        res.num_points = res.pts.size();
        res.point_cluster.assign(res.pts.size(), -1);

        std::vector<float> eps_i(res.pts.size()), epsz_i(res.pts.size());
        for (size_t i = 0; i < res.pts.size(); ++i) {
            const double r = (res.pts[i].cast<double>() - origin).norm();
            eps_i[i] = eps.at(r);
            epsz_i[i] = eps.atZ(r);
        }
        auto clusters = sins::dbscanGrid(res.pts, eps_i, epsz_i, min_pts);
        if (split.enabled && yaw_only && !planes.empty()) {
            std::vector<std::vector<int>> refined;
            for (auto& idx : clusters) {
                auto parts = splitTallColumns(res.pts, idx, eps_i, epsz_i, min_pts, planes, split);
                if (parts.size() > 1) ++res.splits;
                for (auto& pt : parts) refined.push_back(std::move(pt));
            }
            clusters = std::move(refined);
        }
        res.num_clusters = clusters.size();
        for (const auto& idx : clusters) {
            Eigen::Vector3d up = Eigen::Vector3d::UnitZ();
            if (yaw_only && !planes.empty()) {
                Eigen::Vector3d c = Eigen::Vector3d::Zero();
                for (int i : idx) c += res.pts[i].cast<double>();
                up = localUp(planes, c / static_cast<double>(idx.size()));
            }
            Obb box = sins::computeObb(res.pts, idx, yaw_only, up);
            if (box.volume() < min_volume || box.volume() > max_volume) continue;
            if (yaw_only) {
                // Box z axis = local up: size.z is the height above the local ground plane, x/y the footprint.
                const double h0 = std::max(box.size.x(), box.size.y()), h1 = std::min(box.size.x(), box.size.y());
                const double l0 = std::max(max_extent.x(), max_extent.y()), l1 = std::min(max_extent.x(), max_extent.y());
                if (h0 > l0 || h1 > l1 || box.size.z() > max_extent.z()) continue;
            } else {
                // pca3d orders axes by variance, not gravity: compare sorted extents and check the world z range.
                std::array<double, 3> sz{box.size.x(), box.size.y(), box.size.z()};
                std::array<double, 3> lim{max_extent.x(), max_extent.y(), max_extent.z()};
                std::sort(sz.begin(), sz.end(), std::greater<double>());
                std::sort(lim.begin(), lim.end(), std::greater<double>());
                if (sz[0] > lim[0] || sz[1] > lim[1] || sz[2] > lim[2]) continue;
                float zmin = std::numeric_limits<float>::max(), zmax = std::numeric_limits<float>::lowest();
                for (int i : idx) { zmin = std::min(zmin, res.pts[i].z()); zmax = std::max(zmax, res.pts[i].z()); }
                if (zmax - zmin > max_extent.z()) continue;
            }
            const int id = static_cast<int>(res.boxes.size());
            res.boxes.push_back(box);
            res.box_points.push_back(idx);
            if (std::max(box.size.x(), box.size.y()) > 1.5) ++res.large_boxes;
            if (yaw_only ? (box.size.z() < 0.2 && std::max(box.size.x(), box.size.y()) > 1.0) : false) ++res.flat_boxes;
            for (int i : idx) res.point_cluster[i] = id;
        }
        res.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return res;
    }

    vision_msgs::msg::Detection3DArray toDetections(const std::string& source, const std::vector<Obb>& boxes,
                                                    const std_msgs::msg::Header& header) const {
        vision_msgs::msg::Detection3DArray arr;
        arr.header = header;
        arr.header.frame_id = global_frame_;
        for (size_t i = 0; i < boxes.size(); ++i) {
            vision_msgs::msg::Detection3D d;
            d.header = arr.header;
            d.id = source + "_" + std::to_string(i);
            d.bbox.center.position.x = boxes[i].center.x();
            d.bbox.center.position.y = boxes[i].center.y();
            d.bbox.center.position.z = boxes[i].center.z();
            d.bbox.center.orientation.w = boxes[i].rotation.w();
            d.bbox.center.orientation.x = boxes[i].rotation.x();
            d.bbox.center.orientation.y = boxes[i].rotation.y();
            d.bbox.center.orientation.z = boxes[i].rotation.z();
            d.bbox.size.x = boxes[i].size.x();
            d.bbox.size.y = boxes[i].size.y();
            d.bbox.size.z = boxes[i].size.z();
            arr.detections.push_back(d);
        }
        return arr;
    }

    void appendMarkers(visualization_msgs::msg::MarkerArray& out, const std::string& source, int color_idx,
                       const std::vector<Obb>& boxes, const std_msgs::msg::Header& header) const {
        static const float palette[6][3] = {{0.f, 1.f, 0.f}, {0.f, 0.8f, 1.f}, {1.f, 0.6f, 0.f}, {1.f, 0.f, 1.f}, {0.6f, 0.3f, 1.f}, {1.f, 1.f, 1.f}};
        const float* c = palette[color_idx % 6];
        if (boxes.empty()) return;
        // One LINE_LIST marker per source holding all its boxes' 12 edges each: only the
        // borders are drawn (no filled cube), points are in the global frame.
        static const int edges[12][2] = {{0, 1}, {1, 3}, {3, 2}, {2, 0}, {4, 5}, {5, 7},
                                         {7, 6}, {6, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
        visualization_msgs::msg::Marker m;
        m.header = header;
        m.header.frame_id = global_frame_;
        m.ns = source;
        m.id = 0;
        m.type = visualization_msgs::msg::Marker::LINE_LIST;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.pose.orientation.w = 1.0;
        m.scale.x = 0.03;
        m.color.r = c[0]; m.color.g = c[1]; m.color.b = c[2]; m.color.a = 1.0f;
        m.lifetime = rclcpp::Duration::from_seconds(marker_lifetime_sec_);  // 0 = stay until replaced/deleted (survive a paused bag)
        m.points.reserve(boxes.size() * 24);
        for (const auto& box : boxes) {
            const Eigen::Matrix3d R = box.rotation.toRotationMatrix();
            Eigen::Vector3d corner[8];
            for (int k = 0; k < 8; ++k) {
                const Eigen::Vector3d local(((k & 1) ? 0.5 : -0.5) * box.size.x(),
                                            ((k & 2) ? 0.5 : -0.5) * box.size.y(),
                                            ((k & 4) ? 0.5 : -0.5) * box.size.z());
                corner[k] = box.center + R * local;
            }
            for (const auto& e : edges)
                for (int v : {e[0], e[1]}) {
                    geometry_msgs::msg::Point pt;
                    pt.x = corner[v].x(); pt.y = corner[v].y(); pt.z = corner[v].z();
                    m.points.push_back(pt);
                }
        }
        out.markers.push_back(m);
    }

    void publishClusterCloud(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pub,
                             const SourceResult& res, const std_msgs::msg::Header& header, int color_idx) const {
        if (pub->get_subscription_count() == 0) return;
        static const uint8_t palette[3][3] = {{0, 255, 0}, {0, 200, 255}, {255, 150, 0}};
        sensor_msgs::msg::PointCloud2 cloud;
        cloud.header = header;
        cloud.header.frame_id = global_frame_;
        sensor_msgs::PointCloud2Modifier mod(cloud);
        mod.setPointCloud2FieldsByString(2, "xyz", "rgb");
        size_t kept = 0;
        for (int c : res.point_cluster) kept += (c >= 0);
        mod.resize(kept);
        sensor_msgs::PointCloud2Iterator<float> ox(cloud, "x"), oy(cloud, "y"), oz(cloud, "z");
        sensor_msgs::PointCloud2Iterator<uint8_t> r(cloud, "r"), g(cloud, "g"), b(cloud, "b");
        for (size_t i = 0; i < res.pts.size(); ++i) {
            if (res.point_cluster[i] < 0) continue;
            *ox = res.pts[i].x(); *oy = res.pts[i].y(); *oz = res.pts[i].z();
            *r = palette[color_idx % 3][0]; *g = palette[color_idx % 3][1]; *b = palette[color_idx % 3][2];
            ++ox; ++oy; ++oz; ++r; ++g; ++b;
        }
        pub->publish(cloud);
    }

    // One marker per source (ns = source, id 0) published by its own callback; an empty result deletes it
    // right away, and lifetime covers a source that stops publishing.
    void publishMarkers(const std::string& source, int color_idx, const std::vector<Obb>& boxes,
                        const std_msgs::msg::Header& header) {
        visualization_msgs::msg::MarkerArray markers;
        if (boxes.empty()) {
            visualization_msgs::msg::Marker del;
            del.header = header;
            del.header.frame_id = global_frame_;
            del.ns = source;
            del.id = 0;
            del.action = visualization_msgs::msg::Marker::DELETE;
            markers.markers.push_back(del);
        } else {
            appendMarkers(markers, source, color_idx, boxes, header);
        }
        pub_markers_->publish(markers);
    }

    void onCameraCloud(Camera& cam, int color_idx, const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud) {
        EpsRange eps = camera_eps_;
        eps.near *= cam.eps_scale;
        eps.far *= cam.eps_scale;
        auto res = std::make_shared<SourceResult>(cluster(cloud, eps, camera_min_points_, origin_, yaw_only_, max_extent_,
                                                          max_volume_, min_volume_, floor_planes_, split_));
        cam.pub->publish(toDetections(cam.name, res->boxes, cloud->header));
        publishMarkers(cam.name, color_idx, res->boxes, cloud->header);
        publishClusterCloud(cam.cloud_pub, *res, cloud->header, color_idx);
        record(cam.name, *res);
        if (fusion_enabled_) {
            const double stamp = stampSec(cloud->header.stamp);
            last_cam_[cam.name] = {stamp, res};
            if (pending_.active && !pending_.cams.count(cam.name) && std::abs(stamp - pending_.stamp) <= fusion_stamp_tolerance_)
                pending_.cams[cam.name] = res;
            tryFuse();
        }
        logStats();
    }

    // YOLO semantic leaves (preprocessing_node): depth points inside each detection's box, one cluster per track id. They are not
    // clustered again — each id is a cluster — and are used only as REGION PRIORS to refine the fused objects (see fuseAndPublish).
    void onLeaves(Camera& cam, int color_idx, const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud) {
        cam.leaves_seen = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        auto res = std::make_shared<SourceResult>();
        res->valid = true;
        // by_id groups points by the ORIGINAL detection's track id. This file only ever needed it to be
        // consistent WITHIN one PointCloud2 (grouping points into one leaf per id) — but the value itself,
        // inherited from preprocessing_node's own Leaf.id (which is the YOLO 2D detection's own persistent
        // ByteTrack id, "d.id", parsed as a number — see publishSemanticLeaves's own box/mask branches), is
        // genuinely stable ACROSS ticks for as long as YOLO's own 2D tracker keeps that identity. Carried into
        // box_track_id below (previously discarded here) so tracker_node can use 2D-track continuity as a
        // cross-check — see decaySemanticEvidence()'s own comment on why. class_idx travels alongside as the
        // same detection's class (an index into class_names_, -1 = unknown) — one value per id, read off its
        // first point since every point of one leaf shares its source detection's class by construction.
        std::map<int, std::vector<int>> by_id;
        std::map<int, int> class_by_id;
        sensor_msgs::PointCloud2ConstIterator<float> ix(*cloud, "x"), iy(*cloud, "y"), iz(*cloud, "z"), iid(*cloud, "track_id"),
            icls(*cloud, "class_idx");
        for (; ix != ix.end(); ++ix, ++iy, ++iz, ++iid, ++icls) {
            const int id = static_cast<int>(*iid);
            by_id[id].push_back(static_cast<int>(res->pts.size()));
            class_by_id.emplace(id, static_cast<int>(*icls));
            res->pts.emplace_back(*ix, *iy, *iz);
        }
        res->num_points = res->pts.size();
        res->point_cluster.assign(res->pts.size(), -1);
        for (auto& [id, idx] : by_id) {
            if (static_cast<int>(idx.size()) < camera_min_points_) continue;
            Eigen::Vector3d c = Eigen::Vector3d::Zero();
            for (int i : idx) c += res->pts[i].cast<double>();
            c /= static_cast<double>(idx.size());
            const Eigen::Vector3d up = yaw_only_ ? localUp(floor_planes_, c) : Eigen::Vector3d::UnitZ();
            Obb box = sins::computeObb(res->pts, idx, yaw_only_, up);
            if (!boxOk(box, res->pts)) continue;
            for (int i : idx) res->point_cluster[i] = static_cast<int>(res->boxes.size());
            res->boxes.push_back(box);
            res->box_points.push_back(idx);
            const int cidx = class_by_id.at(id);
            res->box_class.push_back(cidx >= 0 && cidx < static_cast<int>(class_names_.size()) ? class_names_[static_cast<size_t>(cidx)] : "");
            res->box_track_id.push_back(id);
        }
        res->num_clusters = res->boxes.size();
        const std::string name = cam.name + "_yolo";
        publishMarkers(name, color_idx, res->boxes, cloud->header);
        record(name, *res);
        if (fusion_enabled_) {
            const double stamp = stampSec(cloud->header.stamp);
            last_cam_[name] = {stamp, res};
            if (pending_.active && !pending_.cams.count(name) && std::abs(stamp - pending_.stamp) <= fusion_stamp_tolerance_)
                pending_.cams[name] = res;
            tryFuse();
        }
    }

    void onLidar(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar) {
        auto res = std::make_shared<SourceResult>(cluster(lidar, lidar_eps_, lidar_min_points_, origin_, yaw_only_, max_extent_,
                                                          max_volume_, min_volume_, floor_planes_, split_));
        lidar_pub_->publish(toDetections("lidar", res->boxes, lidar->header));
        publishMarkers("lidar", 0, res->boxes, lidar->header);
        publishClusterCloud(lidar_cloud_pub_, *res, lidar->header, 0);
        record("lidar", *res);
        if (fusion_enabled_) {
            if (pending_.active) fuseAndPublish();  // the previous scan never got all its cameras: publish what it has
            pending_ = Pending();
            pending_.active = true;
            pending_.stamp = stampSec(lidar->header.stamp);
            pending_.header = lidar->header;
            pending_.lidar = res;
            pending_.arrived = std::chrono::steady_clock::now();
            for (const auto& [name, sc] : last_cam_)
                if (std::abs(sc.first - pending_.stamp) <= fusion_stamp_tolerance_) pending_.cams[name] = sc.second;
            tryFuse();
        }
        logStats();
    }

    // ---- fusion -------------------------------------------------------------------------------------------

    struct Pending {
        bool active = false;
        double stamp = 0.0;
        std_msgs::msg::Header header;
        std::shared_ptr<SourceResult> lidar;
        std::map<std::string, std::shared_ptr<SourceResult>> cams;
        std::chrono::steady_clock::time_point arrived;
    };

    void tryFuse() {
        if (!pending_.active) return;
        // expected: every camera, plus the YOLO leaves of each camera whose leaves are alive (seen in the last second)
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        size_t expected = cameras_.size();
        for (const auto& kv : cameras_) if (wall - kv.second.leaves_seen < 1.0) ++expected;
        const bool complete = pending_.cams.size() >= expected;
        const double waited = std::chrono::duration<double>(std::chrono::steady_clock::now() - pending_.arrived).count();
        if (complete || waited > fusion_timeout_sec_) {
            fuseAndPublish();
            pending_.active = false;
        }
    }

    bool boxOk(const Obb& box, const std::vector<Eigen::Vector3f>& pts) const {
        if (box.volume() < min_volume_ || box.volume() > max_volume_) return false;
        if (yaw_only_) {
            const double h0 = std::max(box.size.x(), box.size.y()), h1 = std::min(box.size.x(), box.size.y());
            const double l0 = std::max(max_extent_.x(), max_extent_.y()), l1 = std::min(max_extent_.x(), max_extent_.y());
            return !(h0 > l0 || h1 > l1 || box.size.z() > max_extent_.z());
        }
        std::array<double, 3> sz{box.size.x(), box.size.y(), box.size.z()};
        std::array<double, 3> lim{max_extent_.x(), max_extent_.y(), max_extent_.z()};
        std::sort(sz.begin(), sz.end(), std::greater<double>());
        std::sort(lim.begin(), lim.end(), std::greater<double>());
        if (sz[0] > lim[0] || sz[1] > lim[1] || sz[2] > lim[2]) return false;
        float zmin = std::numeric_limits<float>::max(), zmax = std::numeric_limits<float>::lowest();
        for (const auto& p : pts) { zmin = std::min(zmin, p.z()); zmax = std::max(zmax, p.z()); }
        return zmax - zmin <= max_extent_.z();
    }

    struct Fused {
        Obb box;
        std::vector<Eigen::Vector3f> pts;
        unsigned mask = 0;  // bit 0 = LiDAR, bit k = camera k (in camera_names_ order)
        bool yolo_refined = false;  // true once a YOLO leaf's region refined this object (yolo_class may still be
                                    // empty — "unknown class name", e.g. not in class_names_ — that is different
                                    // from "never refined at all", which is what gates re-use below)
        std::string yolo_class;
        int yolo_track_id = -1;    // the refining leaf's own YOLO 2D (ByteTrack) track id, -1 = never refined —
                                    // see onLeaves()'s own comment on why this now survives past this file
    };

    std::string maskName(unsigned mask) const {
        std::string s;
        for (size_t k = 0; k <= 2 * camera_names_.size(); ++k) {
            if (!(mask & (1u << k))) continue;
            if (!s.empty()) s += "+";
            if (k == 0) s += "L";
            else if (k <= camera_names_.size()) s += std::string(1, static_cast<char>(std::toupper(camera_names_[k - 1][0])));
            else s += "Y" + std::string(1, static_cast<char>(std::toupper(camera_names_[k - camera_names_.size() - 1][0])));
        }
        return s;
    }

    void fuseAndPublish() {
        struct Item { unsigned src; const SourceResult* res; size_t idx; };
        std::vector<Item> items;
        if (pending_.lidar) for (size_t i = 0; i < pending_.lidar->boxes.size(); ++i) items.push_back({0, pending_.lidar.get(), i});
        for (size_t k = 0; k < camera_names_.size(); ++k) {
            const auto it = pending_.cams.find(camera_names_[k]);
            if (it != pending_.cams.end())
                for (size_t i = 0; i < it->second->boxes.size(); ++i) items.push_back({static_cast<unsigned>(k + 1), it->second.get(), i});
        }
        namespace obf = sins;

        // One object candidate: its points, OBB and which sources contributed.
        struct Group {
            Obb box;
            std::vector<Eigen::Vector3f> pts;
            unsigned mask = 0;
            double area = 0.0;
        };
        auto itemGroup = [&](const Item& it) {
            Group g;
            g.box = it.res->boxes[it.idx];
            for (int q : it.res->box_points[it.idx]) g.pts.push_back(it.res->pts[q]);
            g.mask = 1u << it.src;
            return g;
        };
        size_t pts_before = 0, pts_after = 0, merges_done = 0, split_residuals = 0, no_evidence = 0, merge_failed = 0, merged_groups = 0, nested_absorbed = 0, nested_split = 0, matched_pairs = 0;

        // Union of groups: union of points -> range-banded voxel centroid (same density as the sensors) -> OBB refit.
        // Returns false (and leaves `out` untouched) when the merged box is not a plausible object.
        auto mergeGroups = [&](const std::vector<const Group*>& gs, Group& out) {
            std::vector<Eigen::Vector3f> uni;
            unsigned mask = 0;
            for (const Group* g : gs) { uni.insert(uni.end(), g->pts.begin(), g->pts.end()); mask |= g->mask; }
            const size_t before = uni.size();
            uni = obf::voxelCentroidBands(uni, origin_, fusion_voxel_near_, fusion_voxel_far_, fusion_voxel_split_);
            std::vector<int> idx(uni.size());
            std::iota(idx.begin(), idx.end(), 0);
            Eigen::Vector3d c = Eigen::Vector3d::Zero();
            for (const auto& q : uni) c += q.cast<double>();
            c /= static_cast<double>(std::max<size_t>(1, uni.size()));
            const Eigen::Vector3d up = yaw_only_ ? localUp(floor_planes_, c) : Eigen::Vector3d::UnitZ();
            Obb box = obf::computeObb(uni, idx, yaw_only_, up);
            if (!boxOk(box, uni)) return false;
            pts_before += before;
            pts_after += uni.size();
            ++merges_done;
            out.box = box;
            out.pts = std::move(uni);
            out.mask = mask;
            return true;
        };

        // ---- Phase 1: one-to-one MUTUAL best match between sources (as jo_detector's BboxesMerger does) ----
        // Two boxes of different sources are the same object only if each is the other's best footprint-IoU match
        // (within that other source) and the IoU and the height overlap are high enough. Unlike linking everything
        // that overlaps, one big box can never swallow two neighbouring objects of the other source through a chain.
        const size_t n = items.size();
        const unsigned nsrc = static_cast<unsigned>(camera_names_.size() + 1);
        std::vector<std::vector<double>> iou(n, std::vector<double>(n, 0.0));
        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n; ++j) {
                const Obb &a = items[i].res->boxes[items[i].idx], &b = items[j].res->boxes[items[j].idx];
                if (fusion_footprint_metric_) {
                    // footprint IoU (inflated by fusion_margin) gated by the height overlap
                    if (obf::zOverlapFraction(a, b) < fusion_z_overlap_) continue;
                    iou[i][j] = iou[j][i] = obf::footprintIoU(a, b, fusion_margin_);
                } else {
                    iou[i][j] = iou[j][i] = obf::obbOverlap3D(a, b, fusion_margin_).iou;
                }
            }
        // best partner of item i inside source `src` (for src == the item's own source: another box of that source)
        auto bestIn = [&](size_t i, unsigned src) {
            int best = -1;
            double bv = 0.0;
            for (size_t j = 0; j < n; ++j)
                if (j != i && items[j].src == src && iou[i][j] > bv) { bv = iou[i][j]; best = static_cast<int>(j); }
            return best;
        };
        std::vector<size_t> parent(n);
        std::iota(parent.begin(), parent.end(), 0);
        std::function<size_t(size_t)> find = [&](size_t x) { while (parent[x] != x) x = parent[x] = parent[parent[x]]; return x; };
        for (size_t i = 0; i < n; ++i)
            for (unsigned s = 0; s < nsrc; ++s) {
                if (s == items[i].src && fusion_iou_same_ <= 0.0) continue;  // no peer matching inside one source
                const int j = bestIn(i, s);
                if (j < 0) continue;
                const double thr = (s == items[i].src) ? fusion_iou_same_ : fusion_iou_thresh_;  // peers within one source use the lower threshold
                if (iou[i][j] < thr) continue;
                if (bestIn(static_cast<size_t>(j), items[i].src) != static_cast<int>(i)) continue;  // not mutual
                parent[find(i)] = find(static_cast<size_t>(j));
                ++matched_pairs;
            }
        std::map<size_t, std::vector<size_t>> roots;
        for (size_t i = 0; i < n; ++i) roots[find(i)].push_back(i);
        std::vector<Group> groups;
        for (const auto& [root, members] : roots) {
            (void)root;
            std::vector<Group> mem;
            for (size_t m : members) mem.push_back(itemGroup(items[m]));
            if (mem.size() == 1) { groups.push_back(std::move(mem[0])); continue; }
            std::vector<const Group*> ptrs;
            for (const auto& g : mem) ptrs.push_back(&g);
            Group merged;
            if (mergeGroups(ptrs, merged)) { ++merged_groups; groups.push_back(std::move(merged)); }
            else for (auto& g : mem) groups.push_back(std::move(g));  // implausible union: keep them apart
        }
        for (auto& g : groups) g.area = obf::polygonArea(obf::obbFootprint(g.box, 0.0));

        // ---- Phase 2: NESTING as a parent/child hierarchy ----
        // A group is nested in a bigger one when (almost) all of its footprint lies inside it and their heights overlap.
        // It is a CHILD of that parent only with evidence that they are the same object: it brings a source the parent
        // lacks (two sensors see the same thing), or — same sources — its points are attached to the parent's points
        // (closest pair <= fusion_nest_attach_dist: a fragment of it). A small object that just sits in the empty interior
        // of a bigger box (the notch of an L-shaped cluster) has no such evidence and stays a separate object.
        // Each child picks its best parent (highest containment); parents are resolved bottom-up:
        //   - 1 child                      -> absorbed (union of points, voxel, OBB refit)
        //   - >=2 children, mutually disjoint, and fusion_nested_multi_child == "leaves"
        //                                  -> the parent was an over-merged cluster: replaced by its leaves
        //   - otherwise                    -> all absorbed.
        const size_t g = groups.size();
        std::vector<int> best_parent(g, -1);
        std::vector<std::vector<size_t>> children(g);
        for (size_t c = 0; c < g; ++c) {
            double best_cont = 0.0;
            for (size_t pp = 0; pp < g; ++pp) {
                if (pp == c) continue;
                if (!(groups[pp].area > groups[c].area * 1.02 || (groups[pp].area >= groups[c].area * 0.98 && pp < c))) continue;
                const bool new_source = (groups[c].mask & ~groups[pp].mask) != 0;
                double iov;
                if (fusion_footprint_metric_) {
                    // footprint containment >= fusion_nest_thresh AND height containment >= fusion_nest_z
                    const auto nr = obf::obbNested(groups[c].box, groups[pp].box, fusion_nest_thresh_, fusion_nest_z_);
                    if (!nr.nested) continue;
                    iov = nr.containment;
                } else {
                    // v1's thresholds: IOV (intersection / volume of the smaller box) between sensors / within one
                    iov = obf::obbOverlap3D(groups[c].box, groups[pp].box, 0.0).iov;
                    if (iov < (new_source ? fusion_iov_cross_ : fusion_iov_same_)) continue;
                }
                if (iov <= best_cont) continue;
                const bool attached = obf::minPointDistance(groups[c].pts, groups[pp].pts, fusion_nest_attach_dist_) <= fusion_nest_attach_dist_;
                if (!new_source && !attached) { ++no_evidence; continue; }
                best_cont = iov;
                best_parent[c] = static_cast<int>(pp);
            }
            if (best_parent[c] >= 0) children[static_cast<size_t>(best_parent[c])].push_back(c);
        }
        std::function<std::vector<Group>(size_t)> resolve = [&](size_t idx) -> std::vector<Group> {
            if (children[idx].empty()) return {groups[idx]};
            std::vector<std::vector<Group>> kids;
            for (size_t c : children[idx]) kids.push_back(resolve(c));
            std::vector<Group> flat;
            for (auto& k : kids) for (auto& x : k) flat.push_back(std::move(x));
            bool disjoint = flat.size() >= 2;
            for (size_t i = 0; i < flat.size() && disjoint; ++i)
                for (size_t j = i + 1; j < flat.size() && disjoint; ++j)
                    if (obf::footprintIoU(flat[i].box, flat[j].box, 0.0) >= fusion_child_overlap_) disjoint = false;
            if (disjoint && fusion_nested_leaves_) {
                // The parent was an over-merged cluster: it is replaced by its leaves and its points are PARTITIONED among
                // them (each point goes to the nearest leaf box), so no point is lost and the refit leaves do not overlap
                // (a residual box around the leaves would have re-created the overlap).
                ++nested_split;
                std::vector<std::vector<Eigen::Vector3f>> assigned(flat.size());
                for (const auto& q : groups[idx].pts) {
                    size_t best = 0;
                    double bd = std::numeric_limits<double>::max();
                    for (size_t k = 0; k < flat.size(); ++k) {
                        const Eigen::Vector3d d = flat[k].box.rotation.toRotationMatrix().transpose() * (q.cast<double>() - flat[k].box.center);
                        const Eigen::Vector3d out = (d.cwiseAbs() - flat[k].box.size * 0.5).cwiseMax(0.0);
                        if (out.norm() < bd) { bd = out.norm(); best = k; }
                    }
                    assigned[best].push_back(q);
                }
                for (size_t k = 0; k < flat.size(); ++k) {
                    if (assigned[k].empty()) continue;
                    Group extra;
                    extra.pts = std::move(assigned[k]);
                    extra.mask = groups[idx].mask;
                    Group refit;
                    if (mergeGroups({&flat[k], &extra}, refit)) { refit.area = flat[k].area; flat[k] = std::move(refit); ++split_residuals; }
                }
                return flat;
            }
            std::vector<const Group*> ptrs{&groups[idx]};
            for (const auto& x : flat) ptrs.push_back(&x);
            Group merged;
            if (mergeGroups(ptrs, merged)) { ++nested_absorbed; return {merged}; }
            ++merge_failed;
            std::vector<Group> all{groups[idx]};
            for (auto& x : flat) all.push_back(std::move(x));
            return all;
        };
        std::vector<Fused> fused;
        for (size_t i = 0; i < g; ++i) {
            if (best_parent[i] >= 0) continue;  // resolved through its parent
            for (auto& r : resolve(i)) fused.push_back({r.box, std::move(r.pts), r.mask});
        }
        const size_t nested_links = nested_absorbed + nested_split;
        (void)matched_pairs;
        // Check: output boxes that are STILL nested in another output box (no evidence that they belong together)
        size_t nested_left = 0;
        for (size_t i = 0; i < fused.size(); ++i)
            for (size_t j = i + 1; j < fused.size(); ++j)
            {
                const bool cross = ((fused[i].mask & ~fused[j].mask) | (fused[j].mask & ~fused[i].mask)) != 0;
                const auto ov = obf::obbOverlap3D(fused[i].box, fused[j].box, 0.0);
                if (ov.iov >= (cross ? fusion_iov_cross_ : fusion_iov_same_)) {
                    ++nested_left;
                }
            }

        // ---- YOLO REFINEMENT (never enlarges a box) ----
        // Each YOLO leaf (the depth points of a detection, foreground only) is a REGION PRIOR: the fused object that overlaps it most
        // is cut to the points that fall inside the leaf's oriented box (plus yolo_refine_margin). Those points become the
        // person's box — rebuilt from the object's OWN points, so it can only shrink or split — and the remaining points are
        // re-clustered into separate objects (the boxes / clutter the person was merged with). The leaf's own points are never
        // added, which is what made the boxes 2x too big when they were fused as another source.
        size_t refined = 0;
        if (yolo_refine_enabled_) {
            // Flat list of every (camera, leaf) pair this tick, to be processed in GLOBAL best-match-first
            // order rather than fixed camera-list-then-leaf-index order (not a v1 concern — v1 never claims a
            // fused object at all, so it has no equivalent ordering problem). The loop below claims a fused
            // object (fused[fi].yolo_refined = true) the moment one leaf accepts it, so whichever leaf is
            // processed FIRST gets first pick — with the old fixed order, a mediocre match (just above
            // yolo_refine_min_iou_) for leaf A processed early could claim an object that leaf B, processed
            // later, would have matched far more convincingly (much higher IoU) — leaving B to settle for its
            // own second-best candidate, or nothing. Ranking every leaf by its own best achievable IoU BEFORE
            // any claiming starts, and processing strongest-match-first, means the leaf with the clearest claim
            // on an object always gets it, independent of which camera or array position it happened to be in.
            struct LeafRef { size_t k, li; double rank_iou; };
            std::vector<LeafRef> leaf_refs;
            for (size_t k = 0; k < camera_names_.size(); ++k) {
                const auto ly = pending_.cams.find(camera_names_[k] + "_yolo");
                if (ly == pending_.cams.end()) continue;
                const SourceResult& leaves = *ly->second;
                for (size_t li = 0; li < leaves.boxes.size(); ++li) {
                    double rank_iou = -1.0;
                    for (const auto& f : fused) rank_iou = std::max(rank_iou, obf::obbOverlap3D(leaves.boxes[li], f.box, 0.0).iou);
                    leaf_refs.push_back({k, li, rank_iou});
                }
            }
            std::stable_sort(leaf_refs.begin(), leaf_refs.end(),
                [](const LeafRef& a, const LeafRef& b) { return a.rank_iou > b.rank_iou; });

            for (const auto& ref : leaf_refs) {
                const size_t k = ref.k, li = ref.li;
                {
                    const auto ly = pending_.cams.find(camera_names_[k] + "_yolo");
                    const SourceResult& leaves = *ly->second;
                    const Obb& lb = leaves.boxes[li];
                    int best = -1;
                    double best_iou = -1.0, second_iou = -1.0;
                    int best_any = -1;       // DIAGNOSTIC ONLY: best candidate ignoring yolo_refine_min_iou
                    double best_any_iou = -1.0;
                    for (size_t fi = 0; fi < fused.size(); ++fi) {
                        if (fused[fi].yolo_refined) continue;  // already used by another detection
                        const double v = obf::obbOverlap3D(lb, fused[fi].box, 0.0).iou;
                        if (v > best_any_iou) { best_any_iou = v; best_any = static_cast<int>(fi); }
                        if (v < yolo_refine_min_iou_) continue;  // not a real candidate at all
                        if (v > best_iou) { second_iou = best_iou; best_iou = v; best = static_cast<int>(fi); }
                        else if (v > second_iou) { second_iou = v; }
                    }
                    if (best < 0) {
                        if (best_any >= 0) {
                            const auto& bf = fused[static_cast<size_t>(best_any)];
                            const double d = (lb.center - bf.box.center).norm();
                            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                "yolo_refine: no candidate at all (%s leaf %zu class=%s, leaf_pos=[%.2f,%.2f,%.2f] leaf_size=[%.2f,%.2f,%.2f]) — "
                                "closest fused obj: iou=%.3f dist=%.2fm pos=[%.2f,%.2f,%.2f] size=[%.2f,%.2f,%.2f] mask=%s",
                                camera_names_[k].c_str(), li, leaves.box_class[li].c_str(),
                                lb.center.x(), lb.center.y(), lb.center.z(), lb.size.x(), lb.size.y(), lb.size.z(),
                                best_any_iou, d, bf.box.center.x(), bf.box.center.y(), bf.box.center.z(),
                                bf.box.size.x(), bf.box.size.y(), bf.box.size.z(), maskName(bf.mask).c_str());
                        } else {
                            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                "yolo_refine: no candidate at all (%s leaf %zu class=%s, leaf_pos=[%.2f,%.2f,%.2f] leaf_size=[%.2f,%.2f,%.2f]) — "
                                "ZERO unclaimed fused objects exist at all this tick (all already yolo_refined, or fused[] is empty)",
                                camera_names_[k].c_str(), li, leaves.box_class[li].c_str(), lb.center.x(), lb.center.y(), lb.center.z(),
                                lb.size.x(), lb.size.y(), lb.size.z());
                        }
                        continue;
                    }
                    // ambiguous: a near-tied runner-up means this could easily be the wrong (merely adjacent)
                    // object — skip rather than risk tagging a generic neighbor with the person's class.
                    if (second_iou >= 0.0 && (best_iou - second_iou) < yolo_refine_min_iou_margin_) {
                        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                            "yolo_refine: skipped ambiguous match (%s leaf %zu class=%s, best_iou=%.2f second_iou=%.2f) — "
                            "would have risked tagging the wrong nearby object", camera_names_[k].c_str(), li, leaves.box_class[li].c_str(), best_iou, second_iou);
                        continue;
                    }
                    // sanity: the leaf and its true match are built from overlapping views of the SAME object,
                    // so their centers should be near-coincident — reject a "best" that is really just adjacent.
                    const double center_dist = (lb.center - fused[static_cast<size_t>(best)].box.center).norm();
                    const double allowed = 0.5 * lb.size.norm() + yolo_refine_margin_;
                    if (center_dist > allowed) {
                        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                            "yolo_refine: skipped distant match (%s leaf %zu class=%s, center_dist=%.2fm > allowed=%.2fm, iou=%.2f, leaf_size=[%.2f,%.2f,%.2f], fused_size=[%.2f,%.2f,%.2f]) — "
                            "closest-IoU candidate was not actually co-located with the leaf", camera_names_[k].c_str(), li, leaves.box_class[li].c_str(), center_dist, allowed, best_iou,
                            lb.size.x(), lb.size.y(), lb.size.z(),
                            fused[static_cast<size_t>(best)].box.size.x(), fused[static_cast<size_t>(best)].box.size.y(), fused[static_cast<size_t>(best)].box.size.z());
                        continue;
                    }
                    Fused& o = fused[static_cast<size_t>(best)];
                    // Successful-match diagnostic (the failure-path warnings above only cover rejections — this
                    // is the one that actually matters for tracing a WRONG-but-accepted match, e.g. the leaf
                    // genuinely landing near a different, already-separate fused object rather than the real
                    // one — no merge required at all, see tracker_node.cpp's own "class_source" diagnostic for
                    // the downstream half of this trace).
                    // NOT throttled on purpose: a 1s per-call-site throttle was hiding events from a SECOND
                    // camera/leaf firing in the same window as a first one — rare enough (bounded by camera
                    // count x leaves/tick) that every one is worth seeing while tracing this.
                    RCLCPP_WARN(this->get_logger(),
                        "yolo_refine: ACCEPTED (%s leaf %zu class=%s, leaf_pos=[%.2f,%.2f,%.2f] leaf_size=[%.2f,%.2f,%.2f]) "
                        "-> fused obj BEFORE cut: pos=[%.2f,%.2f,%.2f] size=[%.2f,%.2f,%.2f] mask=%s, iou=%.2f dist=%.2fm",
                        camera_names_[k].c_str(), li, leaves.box_class[li].c_str(),
                        lb.center.x(), lb.center.y(), lb.center.z(), lb.size.x(), lb.size.y(), lb.size.z(),
                        o.box.center.x(), o.box.center.y(), o.box.center.z(), o.box.size.x(), o.box.size.y(), o.box.size.z(),
                        maskName(o.mask).c_str(), best_iou, center_dist);
                    const Eigen::Matrix3d Rl = lb.rotation.toRotationMatrix().transpose();
                    std::vector<Eigen::Vector3f> inside, rest;
                    for (const auto& q : o.pts) {
                        const Eigen::Vector3d d = Rl * (q.cast<double>() - lb.center);
                        (std::abs(d.x()) <= lb.size.x() * 0.5 + yolo_refine_margin_ && std::abs(d.y()) <= lb.size.y() * 0.5 + yolo_refine_margin_ &&
                         std::abs(d.z()) <= lb.size.z() * 0.5 + yolo_refine_margin_ ? inside : rest).push_back(q);
                    }
                    // Absolute floor (camera_min_points_) AND relative retention floor (see
                    // yolo_refine_min_retention_'s own comment) — a cut that keeps few points outright, or
                    // keeps plenty in absolute terms but only a thin sliver of what the fused object actually
                    // was pre-cut, is rejected the same way: skip this tick, never publish a marginal match.
                    const double retention = o.pts.empty() ? 0.0 :
                        static_cast<double>(inside.size()) / static_cast<double>(o.pts.size());
                    if (static_cast<int>(inside.size()) < camera_min_points_ || retention < yolo_refine_min_retention_) {
                        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                            "yolo_refine: skipped low-retention cut (%s leaf %zu class=%s, inside=%zu rest=%zu "
                            "of %zu pre-cut, retention=%.0f%% < %.0f%%)", camera_names_[k].c_str(), li,
                            leaves.box_class[li].c_str(), inside.size(), rest.size(), o.pts.size(),
                            retention * 100.0, yolo_refine_min_retention_ * 100.0);
                        continue;
                    }
                    std::vector<int> iidx(inside.size());
                    std::iota(iidx.begin(), iidx.end(), 0);
                    Eigen::Vector3d ic = Eigen::Vector3d::Zero();
                    for (const auto& q : inside) ic += q.cast<double>();
                    ic /= static_cast<double>(inside.size());
                    Obb pbox = obf::computeObb(inside, iidx, yaw_only_, yaw_only_ ? localUp(floor_planes_, ic) : Eigen::Vector3d::UnitZ());
                    if (!boxOk(pbox, inside)) continue;
                    const unsigned omask = o.mask;
                    // Re-clustered with the LiDAR's own radii/min-points — correct for any object that went
                    // through mergeGroups (Phase 1 mutual-match, split-residual, or Phase 2 nested-absorb, all
                    // three call the SAME lambda): its points are already range-banded voxel-centroided to
                    // LiDAR density at merge time (see mergeGroups's own comment and the file header), so
                    // lidar_eps_ matches what's actually in `rest` regardless of which sources contributed.
                    // The one case that's NOT true: omask has exactly one bit set and it ISN'T LiDAR's (bit 0)
                    // — a camera-only object that was never merged with anything "passes through untouched"
                    // (same file-header comment) at the depth camera's own, denser native resolution, so
                    // lidar_eps_/lidar_min_points_ would be too tight for it (and camera_eps_/camera_min_points_
                    // already exist for exactly this purpose, used elsewhere for camera-only clustering).
                    const bool cameraOnlySingleSource = omask != 0 && (omask & (omask - 1)) == 0 && (omask & 1u) == 0;
                    const EpsRange& rest_eps = cameraOnlySingleSource ? camera_eps_ : lidar_eps_;
                    const int rest_min_points = cameraOnlySingleSource ? camera_min_points_ : lidar_min_points_;
                    std::vector<Fused> remainder;
                    if (yolo_keep_rest_ && static_cast<int>(rest.size()) >= rest_min_points) {
                        std::vector<float> re(rest.size()), rez(rest.size());
                        for (size_t q = 0; q < rest.size(); ++q) {
                            const double r = (rest[q].cast<double>() - origin_).norm();
                            re[q] = rest_eps.at(r);
                            rez[q] = rest_eps.atZ(r);
                        }
                        for (const auto& comp : obf::dbscanGrid(rest, re, rez, rest_min_points)) {
                            std::vector<Eigen::Vector3f> cp;
                            for (int q : comp) cp.push_back(rest[static_cast<size_t>(q)]);
                            std::vector<int> cidx(cp.size());
                            std::iota(cidx.begin(), cidx.end(), 0);
                            Eigen::Vector3d cc = Eigen::Vector3d::Zero();
                            for (const auto& q : cp) cc += q.cast<double>();
                            cc /= static_cast<double>(cp.size());
                            Obb cb = obf::computeObb(cp, cidx, yaw_only_, yaw_only_ ? localUp(floor_planes_, cc) : Eigen::Vector3d::UnitZ());
                            if (boxOk(cb, cp)) remainder.push_back({cb, std::move(cp), omask, false, ""});
                        }
                    }
                    o.box = pbox;
                    o.pts = std::move(inside);
                    o.yolo_refined = true;
                    o.yolo_class = leaves.box_class[li];
                    o.yolo_track_id = leaves.box_track_id[li];
                    for (auto& rr : remainder) fused.push_back(std::move(rr));
                    ++refined;
                }
            }
        }

        // ---- publish ----
        vision_msgs::msg::Detection3DArray arr;
        arr.header = pending_.header;
        arr.header.frame_id = global_frame_;
        std::map<std::string, std::vector<Obb>> by_name;
        for (size_t i = 0; i < fused.size(); ++i) {
            const Obb& b = fused[i].box;
            vision_msgs::msg::Detection3D d;
            d.header = arr.header;
            d.id = "fused_" + std::to_string(i);
            d.bbox.center.position.x = b.center.x(); d.bbox.center.position.y = b.center.y(); d.bbox.center.position.z = b.center.z();
            d.bbox.center.orientation.w = b.rotation.w(); d.bbox.center.orientation.x = b.rotation.x();
            d.bbox.center.orientation.y = b.rotation.y(); d.bbox.center.orientation.z = b.rotation.z();
            d.bbox.size.x = b.size.x(); d.bbox.size.y = b.size.y(); d.bbox.size.z = b.size.z();
            vision_msgs::msg::ObjectHypothesisWithPose h;
            h.hypothesis.class_id = maskName(fused[i].mask);  // which sources contributed, e.g. "L+B"
            h.hypothesis.score = static_cast<double>(fused[i].pts.size());
            d.results.push_back(h);
            if (fused[i].yolo_refined && !fused[i].yolo_class.empty()) {
                vision_msgs::msg::ObjectHypothesisWithPose y;
                y.hypothesis.class_id = fused[i].yolo_class;  // this tick's YOLO class for this object (see tracker_node for the per-track vote across ticks)
                // Repurposed from a constant 1.0 (unread anywhere) to the refining leaf's own YOLO 2D track id —
                // see tracker_node.cpp's detectionYoloTrackId()/decaySemanticEvidence() for the consumer.
                y.hypothesis.score = static_cast<double>(fused[i].yolo_track_id);
                d.results.push_back(y);
            }
            arr.detections.push_back(d);
            by_name[maskName(fused[i].mask)].push_back(b);
        }
        fused_pub_->publish(arr);

        // All fused boxes share ONE colour (fused_color); which sources built each is in the detection's class_id.
        visualization_msgs::msg::MarkerArray markers;
        if (fused.empty()) {
            visualization_msgs::msg::Marker del;
            del.header = arr.header;
            del.ns = "fused";
            del.id = 0;
            del.action = visualization_msgs::msg::Marker::DELETE;
            markers.markers.push_back(del);
        } else {
            std::vector<Obb> boxes;
            for (const auto& f : fused) boxes.push_back(f.box);
            visualization_msgs::msg::MarkerArray tmp;
            appendMarkers(tmp, "fused", 0, boxes, arr.header);
            for (auto& m : tmp.markers) {
                m.scale.x = 0.05;
                m.color.r = fused_color_[0]; m.color.g = fused_color_[1]; m.color.b = fused_color_[2];
                markers.markers.push_back(m);
            }
        }
        fused_markers_pub_->publish(markers);

        if (fused_cloud_pub_->get_subscription_count() > 0) {
            sensor_msgs::msg::PointCloud2 cloud;
            cloud.header = arr.header;
            sensor_msgs::PointCloud2Modifier mod(cloud);
            // det_idx: which fused[i]/detections[i] (SAME index, no filtering between the two loops above) each
            // point belongs to — tracker_node's own point-cloud motion voting (PATH 3b) reads this to recover,
            // per tick, which points are this detection's own, the same "stash an index as a float field"
            // convention preprocessing_node's own class_idx/track_id fields already use.
            mod.setPointCloud2Fields(7, "x", 1, sensor_msgs::msg::PointField::FLOAT32, "y", 1, sensor_msgs::msg::PointField::FLOAT32,
                                     "z", 1, sensor_msgs::msg::PointField::FLOAT32, "r", 1, sensor_msgs::msg::PointField::UINT8,
                                     "g", 1, sensor_msgs::msg::PointField::UINT8, "b", 1, sensor_msgs::msg::PointField::UINT8,
                                     "det_idx", 1, sensor_msgs::msg::PointField::FLOAT32);
            size_t n = 0;
            for (const auto& f : fused) n += f.pts.size();
            mod.resize(n);
            sensor_msgs::PointCloud2Iterator<float> ox(cloud, "x"), oy(cloud, "y"), oz(cloud, "z"), oidx(cloud, "det_idx");
            sensor_msgs::PointCloud2Iterator<uint8_t> r(cloud, "r"), g(cloud, "g"), b(cloud, "b");
            for (size_t i = 0; i < fused.size(); ++i) {
                const uint8_t cr = static_cast<uint8_t>((i * 97) % 200 + 55), cg = static_cast<uint8_t>((i * 57) % 200 + 55),
                              cb = static_cast<uint8_t>((i * 137) % 200 + 55);
                const float fi = static_cast<float>(i);
                for (const auto& q : fused[i].pts) {
                    *ox = q.x(); *oy = q.y(); *oz = q.z(); *r = cr; *g = cg; *b = cb; *oidx = fi;
                    ++ox; ++oy; ++oz; ++r; ++g; ++b; ++oidx;
                }
            }
            fused_cloud_pub_->publish(cloud);
        }

        fusion_stats_.refined += refined;
        fusion_stats_.nested += nested_absorbed;
        fusion_stats_.nested_left += nested_left;
        for (const auto& it : items) if (it.src == 0) ++fusion_stats_.lidar_in;
        for (const auto& fo : fused) if (fo.mask & 1u) ++fusion_stats_.lidar_out;
        fusion_stats_.nested_split += nested_split;
        (void)nested_links;
        fusion_stats_.ticks++;
        fusion_stats_.sources += items.size();
        fusion_stats_.fused += fused.size();
        fusion_stats_.merged += merged_groups;
        fusion_stats_.merges += merges_done;
        fusion_stats_.residuals += split_residuals;
        fusion_stats_.no_evidence += no_evidence;
        fusion_stats_.merge_failed += merge_failed;
        fusion_stats_.pts_before += pts_before;
        fusion_stats_.pts_after += pts_after;
        fusion_stats_.cams += pending_.cams.size();
        fusion_stats_.latency_ms += 1000.0 * std::chrono::duration<double>(std::chrono::steady_clock::now() - pending_.arrived).count();
    }

    void appendFusedMarkers(visualization_msgs::msg::MarkerArray& out, const std::string& name, int color_idx,
                            const std::vector<Obb>& boxes, const std_msgs::msg::Header& header) const {
        // LiDAR green, camera 1 cyan, camera 2 orange, boxes merged from several sources yellow
        static const float palette[4][3] = {{0.f, 1.f, 0.f}, {0.f, 0.8f, 1.f}, {1.f, 0.6f, 0.f}, {1.f, 1.f, 0.f}};
        const float* c = palette[std::min(color_idx, 3)];
        // reuse the per-source marker builder with a dedicated namespace and a thicker line
        visualization_msgs::msg::MarkerArray tmp;
        appendMarkers(tmp, name, 0, boxes, header);
        for (auto& m : tmp.markers) {
            m.scale.x = 0.05;
            m.color.r = c[0]; m.color.g = c[1]; m.color.b = c[2];
            out.markers.push_back(m);
        }
    }

    void record(const std::string& source, const SourceResult& r) {
        if (!r.valid) return;
        Stats& s = stats_[source];
        s.ms += r.ms; s.points += r.num_points; s.clusters += r.num_clusters; s.boxes += r.boxes.size(); s.large += r.large_boxes; s.flat += r.flat_boxes; s.splits += r.splits; ++s.n;
    }

    void logStats() {
        if ((now() - last_log_).seconds() < stats_period_) return;
        last_log_ = now();
        std::string line = "dbscan | floor planes=" + std::to_string(floor_planes_.size());
        for (const auto& pl : floor_planes_)
            line += " (tilt " + fmt(std::acos(std::min(1.0, pl.normal.z())) * 180.0 / M_PI) + "deg)";
        line += " |";
        for (auto& [name, s] : stats_) {
            if (s.n == 0) continue;
            line += " " + name + ": pts=" + fmt(s.points / s.n) + " clusters=" + fmt(s.clusters / s.n) +
                    " boxes=" + fmt(s.boxes / s.n) + " (>1.5m: " + fmt(s.large / s.n) + ", flat: " + fmt(s.flat / s.n) + ", tall/low splits: " + fmt(s.splits / s.n) + ") " + fmt(s.ms / s.n) + "ms |";
        }
        if (fusion_enabled_ && fusion_stats_.ticks > 0) {
            const auto& f = fusion_stats_;
            line += " FUSION: " + fmt(1.0 * f.sources / f.ticks) + " boxes -> " + fmt(1.0 * f.fused / f.ticks) + " objects (" +
                    fmt(1.0 * f.merged / f.ticks) + " merged, " + fmt(1.0 * f.nested / f.ticks) + " nested parents absorbed a child, " + fmt(1.0 * f.nested_split / f.ticks) + " over-merged parents split into leaves (" + fmt(1.0 * f.residuals / f.ticks) + " leaves refit with the parent's points); YOLO-refined objects: " + fmt(1.0 * f.refined / f.ticks) + "; nested-but-not-merged: " + fmt(1.0 * f.no_evidence / f.ticks) + " without evidence, " + fmt(1.0 * f.merge_failed / f.ticks) + " merge rejected by the size filter; " + fmt(1.0 * f.nested_left / f.ticks) + " nested pairs left in the output), merged pts " + fmt(f.merges ? 1.0 * f.pts_before / f.merges : 0.0) +
                    " -> " + fmt(f.merges ? 1.0 * f.pts_after / f.merges : 0.0) + " | LiDAR boxes in " + fmt(1.0 * f.lidar_in / f.ticks) + ", objects containing LiDAR out " + fmt(1.0 * f.lidar_out / f.ticks) + " per group, cameras/tick " +
                    fmt(1.0 * f.cams / f.ticks) + ", wait " + fmt(f.latency_ms / f.ticks) + "ms |";
        }
        fusion_stats_ = FusionStats();
        RCLCPP_INFO(get_logger(), "%s", line.c_str());
        stats_.clear();
    }

    static std::string fmt(double v) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.1f", v);
        return buf;
    }

    std::string global_frame_, lidar_topic_, camera_topic_template_;
    std::vector<std::string> camera_names_;
    EpsRange lidar_eps_, camera_eps_;
    std::string odom_topic_;
    Eigen::Vector3d origin_{Eigen::Vector3d::Zero()};
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    int lidar_min_points_{8}, camera_min_points_{8};
    bool yaw_only_{false};
    Eigen::Vector3d max_extent_{3.0, 3.0, 2.5};
    double max_volume_{12.0}, min_volume_{1e-3}, stats_period_{5.0};

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
    rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr wall_markers_sub_;
    std::vector<FloorPlane> floor_planes_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
    rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr lidar_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_cloud_pub_;
    std::map<std::string, Camera> cameras_;

    // fusion state
    double marker_lifetime_sec_{0.0};
    SplitParams split_;
    bool yolo_refine_enabled_{true}, yolo_keep_rest_{true};
    double yolo_refine_min_iou_{0.12}, yolo_refine_margin_{0.10}, yolo_refine_min_iou_margin_{0.05};
    double yolo_refine_min_retention_{0.35};
    bool leaves_enabled_{true};
    std::string leaves_topic_template_;
    std::vector<std::string> class_names_;
    std::array<float, 3> fused_color_{1.f, 1.f, 0.f};
    bool fusion_enabled_{true};
    bool fusion_footprint_metric_{true};
    double fusion_z_overlap_{0.3}, fusion_nest_thresh_{0.8}, fusion_nest_z_{0.6};
    bool fusion_nested_leaves_{true};
    double fusion_iou_thresh_{0.6}, fusion_iou_same_{0.4}, fusion_iov_cross_{0.7}, fusion_iov_same_{0.5}, fusion_child_overlap_{0.1}, fusion_margin_{0.0}, fusion_stamp_tolerance_{0.08},
        fusion_timeout_sec_{0.12}, fusion_nest_attach_dist_{0.4}, fusion_voxel_near_{0.05}, fusion_voxel_far_{0.03}, fusion_voxel_split_{6.0};
    Pending pending_;
    std::map<std::string, std::pair<double, std::shared_ptr<SourceResult>>> last_cam_;
    rclcpp::TimerBase::SharedPtr fusion_timer_;
    rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr fused_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr fused_markers_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr fused_cloud_pub_;
    std::set<std::string> fused_last_names_;
    struct FusionStats {
        size_t ticks = 0, sources = 0, fused = 0, merged = 0, pts_before = 0, pts_after = 0, cams = 0, nested = 0, nested_split = 0, nested_left = 0, merges = 0, refined = 0, residuals = 0, no_evidence = 0, merge_failed = 0, lidar_in = 0, lidar_out = 0;
        double latency_ms = 0.0;
    } fusion_stats_;

    rclcpp::Time last_log_;
    std::map<std::string, Stats> stats_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DbscanDetectorNode>());
    rclcpp::shutdown();
    return 0;
}
