#pragma once
// Geometry helpers for fusing oriented boxes from different sensors: rotated-footprint overlap, height overlap,
// and a range-banded voxel-centroid filter used to bring a merged point set back to the sensors' own density.

#include "sins/cluster_obb.hpp"

#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace sins {

// Footprint of a box on the horizontal plane (its first two axes), counter-clockwise, inflated by `margin` per side.
inline std::vector<Eigen::Vector2d> obbFootprint(const Obb& b, double margin) {
    const Eigen::Matrix3d R = b.rotation.toRotationMatrix();
    const Eigen::Vector2d c = b.center.head<2>(), ax = R.col(0).head<2>(), ay = R.col(1).head<2>();
    const double hx = b.size.x() * 0.5 + margin, hy = b.size.y() * 0.5 + margin;
    std::vector<Eigen::Vector2d> p{c + ax * hx + ay * hy, c - ax * hx + ay * hy, c - ax * hx - ay * hy, c + ax * hx - ay * hy};
    double a2 = 0.0;
    for (size_t i = 0; i < 4; ++i) {
        const Eigen::Vector2d &u = p[i], &v = p[(i + 1) % 4];
        a2 += u.x() * v.y() - v.x() * u.y();
    }
    if (a2 < 0.0) std::reverse(p.begin(), p.end());
    return p;
}

inline double polygonArea(const std::vector<Eigen::Vector2d>& p) {
    double a2 = 0.0;
    for (size_t i = 0; i < p.size(); ++i) {
        const Eigen::Vector2d &u = p[i], &v = p[(i + 1) % p.size()];
        a2 += u.x() * v.y() - v.x() * u.y();
    }
    return 0.5 * std::abs(a2);
}

// Sutherland-Hodgman clipping of `subject` by the convex, counter-clockwise polygon `clip`.
inline std::vector<Eigen::Vector2d> clipConvex(std::vector<Eigen::Vector2d> subject, const std::vector<Eigen::Vector2d>& clip) {
    for (size_t e = 0; e < clip.size() && !subject.empty(); ++e) {
        const Eigen::Vector2d &a = clip[e], &b = clip[(e + 1) % clip.size()];
        auto side = [&](const Eigen::Vector2d& p) { return (b.x() - a.x()) * (p.y() - a.y()) - (b.y() - a.y()) * (p.x() - a.x()); };
        std::vector<Eigen::Vector2d> out;
        for (size_t i = 0; i < subject.size(); ++i) {
            const Eigen::Vector2d &cur = subject[i], &prev = subject[(i + subject.size() - 1) % subject.size()];
            const double sc = side(cur), sp = side(prev);
            auto crossing = [&]() {
                const double t = sp / (sp - sc);
                return Eigen::Vector2d(prev + t * (cur - prev));
            };
            if (sc >= 0.0) {
                if (sp < 0.0) out.push_back(crossing());
                out.push_back(cur);
            } else if (sp >= 0.0) {
                out.push_back(crossing());
            }
        }
        subject = std::move(out);
    }
    return subject;
}

inline void obbZInterval(const Obb& b, double& lo, double& hi) {
    const Eigen::Matrix3d R = b.rotation.toRotationMatrix();
    const double half = 0.5 * (std::abs(R(2, 0)) * b.size.x() + std::abs(R(2, 1)) * b.size.y() + std::abs(R(2, 2)) * b.size.z());
    lo = b.center.z() - half;
    hi = b.center.z() + half;
}

// True when two boxes overlap enough to be the same object: footprint intersection over the SMALLER footprint
// (both inflated by `margin`) >= iov_thresh and the height overlap >= z_thresh of the shorter box.
inline bool obbsAssociated(const Obb& a, const Obb& b, double margin, double iov_thresh, double z_thresh) {
    double alo, ahi, blo, bhi;
    obbZInterval(a, alo, ahi);
    obbZInterval(b, blo, bhi);
    const double zov = std::min(ahi, bhi) - std::max(alo, blo);
    const double hmin = std::min(ahi - alo, bhi - blo);
    if (hmin <= 1e-6 || zov / hmin < z_thresh) return false;
    const auto fa = obbFootprint(a, margin), fb = obbFootprint(b, margin);
    const double inter = polygonArea(clipConvex(fa, fb));
    const double smaller = std::min(polygonArea(fa), polygonArea(fb));
    return smaller > 1e-9 && inter / smaller >= iov_thresh;
}

// Footprint IoU (intersection over union) of two boxes, each inflated by `margin`.
inline double footprintIoU(const Obb& a, const Obb& b, double margin) {
    const auto fa = obbFootprint(a, margin), fb = obbFootprint(b, margin);
    const double inter = polygonArea(clipConvex(fa, fb));
    const double uni = polygonArea(fa) + polygonArea(fb) - inter;
    return uni > 1e-9 ? inter / uni : 0.0;
}

// 3D IoU of two boxes approximated as (footprint intersection x height overlap) — the footprints are the rotated
// rectangles, heights are the boxes' z ranges — like jo_detector's calBoxIOU but on oriented boxes.
// `intersection_volume` / `smaller_volume` are also returned so the IOV (intersection / volume of the smaller box) can be derived.
struct Overlap3D {
    double iou = 0.0;
    double iov = 0.0;
};

inline Overlap3D obbOverlap3D(const Obb& a, const Obb& b, double margin = 0.0) {
    Overlap3D o;
    double alo, ahi, blo, bhi;
    obbZInterval(a, alo, ahi);
    obbZInterval(b, blo, bhi);
    const double zov = std::min(ahi, bhi) - std::max(alo, blo);
    if (zov <= 0.0) return o;
    const auto fa = obbFootprint(a, margin), fb = obbFootprint(b, margin);
    const double inter = polygonArea(clipConvex(fa, fb)) * zov;
    const double va = polygonArea(fa) * (ahi - alo), vb = polygonArea(fb) * (bhi - blo);
    const double uni = va + vb - inter;
    if (uni > 1e-9) o.iou = inter / uni;
    const double smaller = std::min(va, vb);
    if (smaller > 1e-9) o.iov = inter / smaller;
    return o;
}

// Fraction of the shorter box's height range that overlaps the other's.
inline double zOverlapFraction(const Obb& a, const Obb& b) {
    double alo, ahi, blo, bhi;
    obbZInterval(a, alo, ahi);
    obbZInterval(b, blo, bhi);
    const double hmin = std::min(ahi - alo, bhi - blo);
    return hmin > 1e-6 ? (std::min(ahi, bhi) - std::max(alo, blo)) / hmin : 0.0;
}

// Nesting: is one box's footprint (almost) inside the other's, with a similar height range? `first_is_small` says which
// one is the contained box. containment = intersection / area of the SMALLER footprint (no inflation).
struct NestResult {
    bool nested = false;
    bool first_is_small = false;
    double containment = 0.0;  // footprint intersection / smaller footprint
};

inline NestResult obbNested(const Obb& a, const Obb& b, double containment_thresh, double z_thresh) {
    NestResult r;
    const auto fa = obbFootprint(a, 0.0), fb = obbFootprint(b, 0.0);
    const double area_a = polygonArea(fa), area_b = polygonArea(fb);
    r.first_is_small = area_a <= area_b;
    const double smaller = std::min(area_a, area_b);
    if (smaller < 1e-9) return r;
    r.containment = polygonArea(clipConvex(fa, fb)) / smaller;
    if (r.containment < containment_thresh) return r;
    double alo, ahi, blo, bhi;
    obbZInterval(a, alo, ahi);
    obbZInterval(b, blo, bhi);
    const double hs = r.first_is_small ? (ahi - alo) : (bhi - blo);
    const double zov = std::min(ahi, bhi) - std::max(alo, blo);
    r.nested = hs > 1e-6 && zov / hs >= z_thresh;
    return r;
}

// Smallest distance between two point sets, early exit once it is below `stop_below` (brute force: clusters are small).
inline double minPointDistance(const std::vector<Eigen::Vector3f>& a, const std::vector<Eigen::Vector3f>& b, double stop_below) {
    double best2 = std::numeric_limits<double>::max();
    const double stop2 = stop_below * stop_below;
    for (const auto& p : a)
        for (const auto& q : b) {
            const double d2 = (p - q).cast<double>().squaredNorm();
            if (d2 < best2) {
                best2 = d2;
                if (best2 <= stop2) return std::sqrt(best2);
            }
        }
    return std::sqrt(best2);
}

// Voxel-centroid filter in two range bands (distance from `origin`): leaf `near_leaf` closer than `split`,
// `far_leaf` beyond — the same scheme preprocessing_node applies to the LiDAR. World-aligned exact voxel keys.
inline std::vector<Eigen::Vector3f> voxelCentroidBands(const std::vector<Eigen::Vector3f>& pts, const Eigen::Vector3d& origin,
                                                       double near_leaf, double far_leaf, double split) {
    if (near_leaf <= 0.0 && far_leaf <= 0.0) return pts;
    struct Acc { Eigen::Vector3d sum = Eigen::Vector3d::Zero(); int n = 0; };
    std::unordered_map<std::uint64_t, Acc> vox;
    vox.reserve(pts.size());
    constexpr std::int64_t kOff = 1 << 20;
    for (const auto& p : pts) {
        const bool far_band = (p.cast<double>() - origin).norm() >= split;
        const double leaf = far_band ? far_leaf : near_leaf;
        if (leaf <= 0.0) continue;
        const std::int64_t ix = static_cast<std::int64_t>(std::floor(p.x() / leaf)) + kOff;
        const std::int64_t iy = static_cast<std::int64_t>(std::floor(p.y() / leaf)) + kOff;
        const std::int64_t iz = static_cast<std::int64_t>(std::floor(p.z() / leaf)) + kOff;
        const std::uint64_t key = (static_cast<std::uint64_t>(far_band ? 1 : 0) << 63) |
                                  (static_cast<std::uint64_t>(ix & 0x1FFFFF) << 42) |
                                  (static_cast<std::uint64_t>(iy & 0x1FFFFF) << 21) | static_cast<std::uint64_t>(iz & 0x1FFFFF);
        Acc& a = vox[key];
        a.sum += p.cast<double>();
        ++a.n;
    }
    std::vector<Eigen::Vector3f> out;
    out.reserve(vox.size());
    for (const auto& kv : vox) out.emplace_back((kv.second.sum / kv.second.n).cast<float>());
    return out;
}

}  // namespace sins
