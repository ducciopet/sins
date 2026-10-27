/*
    FILE: wall_registry.hpp
    ---------------------------------
    Persistent wall tracking, ported from jo_detector/scripts/
    wall_detector/wallDetector.{h,cpp}'s WallBBox/WallBBoxRegistry —
    same algorithm (OBB-vs-OBB IoU/IoV via separating-axis theorem,
    EMA-merge on match, missed-frame-count expiry), header-only here
    since sins doesn't split executables into
    library-and-node yet (see ../README.md).

    Why this exists: a per-scan wall-plane RANSAC (see preprocessing_node.cpp)
    produces a fresh, independent set of boxes every scan — nothing connects
    "the wall I found this frame" to "the wall I found last frame", so a
    single sparse/occluded scan makes a real wall vanish from the output
    for that frame, and two RANSAC runs of the same physical wall rarely
    agree on exact extents. WallBBoxRegistry::update() fixes both: it
    matches each new detection against the existing registry (by IoU,
    gated on rough normal alignment and center distance) and merges into
    the match instead of replacing it, so a wall's box stabilizes over
    time instead of jittering; unmatched existing entries are NOT deleted
    immediately — missed_frames increments, and only once it exceeds
    max_missed_frames does the entry get dropped, so one bad/occluded scan
    never erases an otherwise-persistent wall.
*/
#ifndef SINS_WALL_REGISTRY_HPP
#define SINS_WALL_REGISTRY_HPP

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

namespace sins {

// =====================================================================
// WallBBox — oriented bounding box for a wall (or floor slab) segment.
// rotation.col(0) is the box's "normal" axis (thickness direction) by
// convention throughout this file, matching buildBoxFromPlane() in
// preprocessing_node.cpp.
// =====================================================================
struct WallBBox {
    Eigen::Vector3d size     = Eigen::Vector3d::Zero();
    Eigen::Vector3d center   = Eigen::Vector3d::Zero();
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();

    const Eigen::Vector3d& get_size()     const { return size; }
    const Eigen::Vector3d& get_center()   const { return center; }
    const Eigen::Matrix3d& get_rotation() const { return rotation; }

    void transform(const Eigen::Isometry3d& T) {
        center   = T * center;
        rotation = T.rotation() * rotation;
    }
};

// =====================================================================
// WallBBoxRegistry — persistent wall tracking across frames.
// =====================================================================
class WallBBoxRegistry {
public:
    using Ptr = std::shared_ptr<WallBBoxRegistry>;

    struct Config {
        double overlap_threshold    = 0.3;
        double merge_weight         = 0.3;
        bool   enable_expiry        = true;
        int    max_missed_frames    = 30;
        double min_normal_dot       = 0.9;
        double max_center_distance  = 5.0;
        // Plane matching (replaces the IoU test when plane_match_dist > 0): a wall is a THIN box (~0.14 m), so a few
        // cm of jitter between two scans drops the IoU under overlap_threshold and every scan spawned a NEW entry that
        // then lived max_missed_frames — the registry filled with duplicates of the same wall. Instead: same normal
        // (min_normal_dot), centre within plane_match_dist of the tracked plane, and the two faces overlapping or at
        // most plane_match_gap apart in-plane.
        double plane_match_dist     = 0.0;
        double plane_match_gap      = 1.0;
    };

    explicit WallBBoxRegistry(const Config& config) : config_(config) {}

    // new_bboxes: this scan's fresh detections (any frame — same frame as
    // the registry's own storage, since delta_pose transforms the EXISTING
    // entries to match; pass Identity() when detection already runs in the
    // registry's frame, e.g. preprocessing_node's global-frame RANSAC).
    // Each incoming box is matched against the closest existing entry
    // (by IoU, gated on rough normal alignment + center distance) and
    // merged in if found; otherwise added as a new entry. Existing entries
    // that matched nothing this frame are NOT removed — missed_frames
    // increments instead, and only entries whose missed_frames exceeds
    // max_missed_frames get dropped (enable_expiry gates this entirely).
    void update(const std::vector<WallBBox>& new_bboxes,
                const Eigen::Isometry3d& delta_pose) {
        transform_existing_bboxes(delta_pose);
        std::vector<bool> matched(registry_.size(), false);

        for (const auto& incoming : new_bboxes) {
            if (incoming.get_size().isZero()) continue;

            int    best_idx = -1;
            double best_iou = config_.overlap_threshold;

            for (int i = 0; i < static_cast<int>(registry_.size()); ++i) {
                const Eigen::Vector3d n_existing = registry_[i].get_rotation().col(0);
                const Eigen::Vector3d n_incoming = incoming.get_rotation().col(0);
                if (std::abs(n_existing.dot(n_incoming)) < config_.min_normal_dot)
                    continue;

                if (config_.plane_match_dist <= 0.0 &&
                    (registry_[i].get_center() - incoming.get_center()).norm() > config_.max_center_distance)
                    continue;

                if (config_.plane_match_dist > 0.0) {
                    if (!samePlane(registry_[i], incoming)) continue;
                    const double score = 1.0 - std::abs(n_existing.dot(incoming.get_center() - registry_[i].get_center()));
                    if (best_idx < 0 || score > best_iou) {
                        best_iou = score;
                        best_idx = i;
                    }
                    continue;
                }
                const double overlap = iou(registry_[i], incoming);
                if (overlap > best_iou) {
                    best_iou = overlap;
                    best_idx = i;
                }
            }

            if (best_idx >= 0) {
                registry_[best_idx]      = merge(registry_[best_idx], incoming, config_.merge_weight);
                matched[best_idx]        = true;
                missed_frames_[best_idx] = 0;
            } else {
                registry_.push_back(incoming);
                missed_frames_.push_back(0);
                matched.push_back(true);  // new entry: counts as matched on first frame
            }
        }

        if (config_.plane_match_dist > 0.0) dedupeRegistry(matched);

        if (config_.enable_expiry) {
            for (int i = 0; i < static_cast<int>(registry_.size()); ++i)
                if (!matched[i]) ++missed_frames_[i];

            for (int i = static_cast<int>(registry_.size()) - 1; i >= 0; --i) {
                if (missed_frames_[i] > config_.max_missed_frames) {
                    registry_.erase(registry_.begin() + i);
                    missed_frames_.erase(missed_frames_.begin() + i);
                }
            }
        }
    }

    void remove_expired(const std::vector<bool>& empty_bboxes) {
        for (int i = static_cast<int>(registry_.size()) - 1; i >= 0; --i) {
            if (i < static_cast<int>(empty_bboxes.size()) && empty_bboxes[i]) {
                registry_.erase(registry_.begin() + i);
                missed_frames_.erase(missed_frames_.begin() + i);
            }
        }
    }

    const std::vector<WallBBox>& bboxes() const { return registry_; }

    // Merge nested/overlapping OBBs within a single detection frame before
    // passing them to update(). Uses IOV = vol_intersection / vol_smaller_OBB
    // so that containment (small box inside large box) is detected even when
    // IoU is low. Iterates until convergence.
    std::vector<WallBBox> mergeNested(const std::vector<WallBBox>& input, double iov_thresh) const {
        std::vector<WallBBox> current = input;

        bool changed = true;
        while (changed) {
            changed = false;
            std::vector<bool> removed(current.size(), false);

            for (int i = 0; i < static_cast<int>(current.size()); ++i) {
                if (removed[i]) continue;
                for (int j = i + 1; j < static_cast<int>(current.size()); ++j) {
                    if (removed[j]) continue;

                    const Eigen::Vector3d ni = current[i].get_rotation().col(0);
                    const Eigen::Vector3d nj = current[j].get_rotation().col(0);
                    if (std::abs(ni.dot(nj)) < config_.min_normal_dot) continue;

                    if (iov(current[i], current[j]) >= iov_thresh) {
                        current[i] = merge(current[i], current[j], config_.merge_weight);
                        removed[j] = true;
                        changed    = true;
                    }
                }
            }

            std::vector<WallBBox> next;
            next.reserve(current.size());
            for (int i = 0; i < static_cast<int>(current.size()); ++i)
                if (!removed[i]) next.push_back(current[i]);
            current = std::move(next);
        }

        return current;
    }

private:
    struct OBB {
        Eigen::Vector3d center;
        Eigen::Matrix3d axes;
        Eigen::Vector3d half_extents;
    };

    Config config_;
    std::vector<WallBBox> registry_;
    std::vector<int> missed_frames_;

    OBB to_obb(const WallBBox& bbox) const {
        OBB obb;
        obb.center       = bbox.get_center();
        obb.axes         = bbox.get_rotation();
        obb.half_extents = bbox.get_size() * 0.5;
        return obb;
    }

    double project_obb(const OBB& obb, const Eigen::Vector3d& axis) const {
        return obb.half_extents[0] * std::abs(obb.axes.col(0).dot(axis))
             + obb.half_extents[1] * std::abs(obb.axes.col(1).dot(axis))
             + obb.half_extents[2] * std::abs(obb.axes.col(2).dot(axis));
    }

    double intersection_volume(const OBB& a, const OBB& b) const {
        const Eigen::Vector3d t = b.center - a.center;

        double overlap[15];
        Eigen::Vector3d axes[15];

        int k = 0;
        for (int i = 0; i < 3; ++i) axes[k++] = a.axes.col(i);
        for (int i = 0; i < 3; ++i) axes[k++] = b.axes.col(i);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                axes[k] = a.axes.col(i).cross(b.axes.col(j));
                if (axes[k].norm() > 1e-6) axes[k].normalize();
                ++k;
            }

        for (int i = 0; i < 15; ++i) {
            if (axes[i].norm() < 1e-6) { overlap[i] = 1e9; continue; }
            const double center_proj = std::abs(t.dot(axes[i]));
            const double ra = project_obb(a, axes[i]);
            const double rb = project_obb(b, axes[i]);
            overlap[i] = std::max(0.0, ra + rb - center_proj);
            if (overlap[i] <= 0.0) return 0.0;
        }

        return std::min(overlap[0], 2.0 * a.half_extents[0])
             * std::min(overlap[1], 2.0 * a.half_extents[1])
             * std::min(overlap[2], 2.0 * a.half_extents[2]);
    }

    double iou(const WallBBox& a, const WallBBox& b) const {
        const OBB oa = to_obb(a);
        const OBB ob = to_obb(b);

        const double vol_a = 8.0 * oa.half_extents.prod();
        const double vol_b = 8.0 * ob.half_extents.prod();
        if (vol_a < 1e-9 || vol_b < 1e-9) return 0.0;

        const double vol_inter = intersection_volume(oa, ob);
        return vol_inter / (vol_a + vol_b - vol_inter + 1e-9);
    }

    double iov(const WallBBox& a, const WallBBox& b) const {
        const OBB oa = to_obb(a);
        const OBB ob = to_obb(b);

        const double vol_a = 8.0 * oa.half_extents.prod();
        const double vol_b = 8.0 * ob.half_extents.prod();
        if (vol_a < 1e-9 || vol_b < 1e-9) return 0.0;

        const double vol_inter = intersection_volume(oa, ob);
        return vol_inter / (std::min(vol_a, vol_b) + 1e-9);
    }

    // Local X (existing.rotation.col(0), the box's "normal"/thickness axis
    // by convention — see WallBBox's comment) is handled DIFFERENTLY from
    // local Y/Z (the in-plane, along-the-wall extents):
    //   - Y/Z: union of projected corners, as before — legitimate growth.
    //     A robot moving along a corridor sees more of a real wall's
    //     length/height over time; the box SHOULD keep extending to cover
    //     that, and a plane fit's Y/Z extent (bounded by where the inliers
    //     actually are) doesn't drift from orientation noise the way X does.
    //   - X: EMA-blended (weight) instead of unioned. A real wall has a
    //     fixed physical thickness; any apparent difference scan to scan is
    //     RANSAC/sensor noise in the fitted plane's orientation, not new
    //     information about a thicker wall. Unioning that noise, forever,
    //     with a rotation that's frozen at whichever detection happened to
    //     be "existing" first (this function never updates `rotation`) is
    //     exactly what made tracked walls grow thicker without bound every
    //     scan — the bug reported against sins's wall
    //     detector (too thick/unrealistic vs. jo_detector's own
    //     static_structures_node, which sins is meant to match).
    //     EMA-blending the thickness and its local-X center instead lets
    //     the box settle at a stable value close to the true physical
    //     thickness (biased slightly wide by wall_ransac_inlier_threshold_'s
    //     own minimum-thickness floor — see buildBoxFromPlane in
    //     preprocessing_node.cpp — but bounded, not runaway). This is also
    //     the first real use of `weight`/merge_weight, previously an unused
    //     parameter in both jo_detector's original and this port.
    //
    //     Getting the incoming thickness right matters: the FIRST version
    //     of this fix read it off the X-span of incoming's corners after
    //     projecting them through R_in_local (existing's frame) — same
    //     machinery as the Y/Z union below — and that still grew without
    //     bound. Why: match() only requires min_normal_dot (~25°) between
    //     the two boxes' normals, so R_in_local can carry a real few-degree
    //     rotation; projecting incoming's LONG Y/Z half-extents (a wall can
    //     be 10+ m long) through even a small angle throws meters onto the
    //     X axis (13m * sin(15°) ≈ 3.4m) — that's orientation-mismatch-
    //     amplified-by-length, not thickness noise, and it grows right
    //     along with the box's own Y extent as more wall comes into view.
    //     Using incoming's OWN local thickness (size.x(), in its own
    //     frame, never rotated) and its center's offset along existing's
    //     normal (c_in_local.x() — one point, not an extent, so no
    //     amplification) avoids the whole problem.
    WallBBox merge(const WallBBox& existing, const WallBBox& incoming, double weight) const {
        const Eigen::Matrix3d& R     = existing.get_rotation();
        const Eigen::Vector3d& c_ex  = existing.get_center();
        const Eigen::Vector3d  he_ex = existing.get_size() * 0.5;

        const Eigen::Vector3d c_in_local = R.transpose() * (incoming.get_center() - c_ex);
        const Eigen::Vector3d he_in      = incoming.get_size() * 0.5;
        const Eigen::Matrix3d R_in_local = R.transpose() * incoming.get_rotation();

        Eigen::Vector3d local_min = -he_ex;
        Eigen::Vector3d local_max =  he_ex;

        for (int sx : {-1, 1})
        for (int sy : {-1, 1})
        for (int sz : {-1, 1}) {
            const Eigen::Vector3d corner_local =
                R_in_local * Eigen::Vector3d(sx * he_in.x(), sy * he_in.y(), sz * he_in.z());
            const Eigen::Vector3d v = c_in_local + corner_local;
            local_min = local_min.cwiseMin(v);
            local_max = local_max.cwiseMax(v);
        }
        // local_min.x()/local_max.x() from the loop above are discarded —
        // see the comment above for why — and replaced below.

        const double incoming_thickness = incoming.get_size().x();
        const double incoming_center_x  = c_in_local.x();
        const double existing_thickness = he_ex.x() * 2.0;
        const double w = std::clamp(weight, 0.0, 1.0);
        const double new_thickness = (1.0 - w) * existing_thickness + w * incoming_thickness;
        const double new_center_x  = (1.0 - w) * 0.0              + w * incoming_center_x;  // existing's local-X center is 0 by definition
        local_min.x() = new_center_x - 0.5 * new_thickness;
        local_max.x() = new_center_x + 0.5 * new_thickness;

        const Eigen::Vector3d new_size   = local_max - local_min;
        const Eigen::Vector3d new_center = c_ex + R * (0.5 * (local_max + local_min));

        WallBBox result;
        result.size     = new_size;
        result.center   = new_center;
        result.rotation = R;
        return result;
    }

    // Plane test between two boxes (see Config::plane_match_dist).
    bool samePlane(const WallBBox& ex, const WallBBox& in) const {
        const Eigen::Vector3d n_ex = ex.get_rotation().col(0);
        if (std::abs(n_ex.dot(in.get_rotation().col(0))) < config_.min_normal_dot) return false;
        const Eigen::Matrix3d& Re = ex.get_rotation();
        const Eigen::Vector3d dc = in.get_center() - ex.get_center();
        if (std::abs(n_ex.dot(dc)) > config_.plane_match_dist) return false;
        const Eigen::Vector3d he = ex.get_size() * 0.5, hi = in.get_size() * 0.5;
        const Eigen::Matrix3d& Ri = in.get_rotation();
        for (int a = 1; a <= 2; ++a) {
            const Eigen::Vector3d axis = Re.col(a);
            const double proj_in = std::abs(Ri.col(0).dot(axis)) * hi.x() + std::abs(Ri.col(1).dot(axis)) * hi.y() +
                                   std::abs(Ri.col(2).dot(axis)) * hi.z();
            if (std::abs(dc.dot(axis)) - he[a] - proj_in > config_.plane_match_gap) return false;
        }
        return true;
    }

    // Tracked entries that became the same wall are merged, keeping the more recently seen one's miss counter.
    void dedupeRegistry(std::vector<bool>& matched) {
        bool changed = true;
        while (changed) {
            changed = false;
            for (size_t i = 0; i < registry_.size() && !changed; ++i)
                for (size_t j = i + 1; j < registry_.size() && !changed; ++j)
                    if (samePlane(registry_[i], registry_[j])) {
                        registry_[i] = merge(registry_[i], registry_[j], config_.merge_weight);
                        missed_frames_[i] = std::min(missed_frames_[i], missed_frames_[j]);
                        matched[i] = matched[i] || matched[j];
                        registry_.erase(registry_.begin() + static_cast<long>(j));
                        missed_frames_.erase(missed_frames_.begin() + static_cast<long>(j));
                        matched.erase(matched.begin() + static_cast<long>(j));
                        changed = true;
                    }
        }
    }

    void transform_existing_bboxes(const Eigen::Isometry3d& delta_pose) {
        for (auto& bbox : registry_) bbox.transform(delta_pose);
    }
};

}  // namespace sins

#endif  // SINS_WALL_REGISTRY_HPP
