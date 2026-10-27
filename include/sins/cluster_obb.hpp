#pragma once
// Grid-accelerated DBSCAN + 3D oriented bounding boxes (OBB), header-only.
//
// DBSCAN: same core/border/noise semantics as jo_detector's dbscan.cpp
// (min_pts counts the point itself), but neighbours come from a hash grid with
// cell size = the largest eps (27 cells per query) instead of a brute-force
// O(n^2) scan. eps can vary per point (range-adaptive: a rotating LiDAR's
// spacing grows with distance, so a fixed eps either fuses near objects or
// fragments far ones): two points are neighbours when their distance is within
// the MEAN of their two eps values (symmetric, as DBSCAN requires).
// The vertical radius can differ from the horizontal one (eps_z per point):
// a spinning multi-ring LiDAR (VLP-16: 2 deg between rings = 0.31 m at 9 m,
// but 0.03 m between points along a ring) samples a wall as horizontal rows
// far apart vertically; an isotropic eps either splits the wall into one flat
// box per ring or, if large enough to join the rings, fuses everything. The
// neighbourhood is then an ellipsoid: (dx^2+dy^2)/eps_xy^2 + dz^2/eps_z^2 <= 1.
//
// OBB: "pca3d" = full 3D orientation from the covariance eigenvectors (axes
// sorted by decreasing variance, right-handed); "yaw" = box whose z axis is
// the given `up` vector (world +Z on flat ground, the local ground normal on
// a slope) and whose x/y axes are rotated about it by the yaw that PCA of the
// points projected on the plane perpendicular to `up` gives. So a box on a
// ramp stands on the ramp instead of being cut by the world horizontal.
// Extents are the min/max of the points projected on the box axes, so the
// box is tight, not variance-based.

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace sins {

struct Obb {
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();  // box axes -> world
    Eigen::Vector3d size = Eigen::Vector3d::Zero();                // full extents along the box axes
    int num_points = 0;
    double volume() const { return size.x() * size.y() * size.z(); }
};

inline std::vector<std::vector<int>> dbscanGrid(const std::vector<Eigen::Vector3f>& pts,
                                                const std::vector<float>& eps_i, const std::vector<float>& epsz_i,
                                                int min_pts) {
    const int n = static_cast<int>(pts.size());
    std::vector<std::vector<int>> clusters;
    if (n == 0 || static_cast<int>(eps_i.size()) != n || static_cast<int>(epsz_i.size()) != n) return clusters;
    const float eps_max = *std::max_element(eps_i.begin(), eps_i.end());
    const float epsz_max = *std::max_element(epsz_i.begin(), epsz_i.end());
    if (eps_max <= 0.f || epsz_max <= 0.f) return clusters;

    const float inv = 1.0f / eps_max, invz = 1.0f / epsz_max;
    auto cellKey = [](int ix, int iy, int iz) -> int64_t {
        return (static_cast<int64_t>(ix) * 73856093LL) ^ (static_cast<int64_t>(iy) * 19349663LL) ^
               (static_cast<int64_t>(iz) * 83492791LL);
    };
    std::unordered_map<int64_t, std::vector<int>> grid;
    grid.reserve(static_cast<size_t>(n));
    std::vector<Eigen::Vector3i> cell(n);
    for (int i = 0; i < n; ++i) {
        cell[i] = Eigen::Vector3i(static_cast<int>(std::floor(pts[i].x() * inv)),
                                  static_cast<int>(std::floor(pts[i].y() * inv)),
                                  static_cast<int>(std::floor(pts[i].z() * invz)));
        grid[cellKey(cell[i].x(), cell[i].y(), cell[i].z())].push_back(i);
    }
    auto neighbours = [&](int i, std::vector<int>& out) {
        out.clear();
        const Eigen::Vector3i& c = cell[i];
        for (int dx = -1; dx <= 1; ++dx)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dz = -1; dz <= 1; ++dz) {
                    auto it = grid.find(cellKey(c.x() + dx, c.y() + dy, c.z() + dz));
                    if (it == grid.end()) continue;
                    for (int j : it->second) {
                        const float e = 0.5f * (eps_i[i] + eps_i[j]);
                        const float ez = 0.5f * (epsz_i[i] + epsz_i[j]);
                        const Eigen::Vector3f d = pts[j] - pts[i];
                        if ((d.x() * d.x() + d.y() * d.y()) / (e * e) + (d.z() * d.z()) / (ez * ez) <= 1.0f) out.push_back(j);
                    }
                }
    };

    constexpr int kUnvisited = -1, kNoise = -2;
    std::vector<int> label(n, kUnvisited);
    std::vector<int> nb, nb2, queue;
    for (int i = 0; i < n; ++i) {
        if (label[i] != kUnvisited) continue;
        neighbours(i, nb);
        if (static_cast<int>(nb.size()) < min_pts) { label[i] = kNoise; continue; }
        const int id = static_cast<int>(clusters.size());
        clusters.emplace_back();
        label[i] = id;
        clusters[id].push_back(i);
        queue.assign(nb.begin(), nb.end());
        for (size_t q = 0; q < queue.size(); ++q) {
            const int j = queue[q];
            if (label[j] == kNoise) { label[j] = id; clusters[id].push_back(j); continue; }
            if (label[j] != kUnvisited) continue;
            label[j] = id;
            clusters[id].push_back(j);
            neighbours(j, nb2);
            if (static_cast<int>(nb2.size()) >= min_pts)
                queue.insert(queue.end(), nb2.begin(), nb2.end());
        }
    }
    return clusters;
}

// Isotropic per-point eps (eps_z = eps_xy).
inline std::vector<std::vector<int>> dbscanGrid(const std::vector<Eigen::Vector3f>& pts,
                                                const std::vector<float>& eps_i, int min_pts) {
    return dbscanGrid(pts, eps_i, eps_i, min_pts);
}

// Constant-eps convenience overload.
inline std::vector<std::vector<int>> dbscanGrid(const std::vector<Eigen::Vector3f>& pts, float eps, int min_pts) {
    return dbscanGrid(pts, std::vector<float>(pts.size(), eps), min_pts);
}

inline Obb computeObb(const std::vector<Eigen::Vector3f>& pts, const std::vector<int>& idx, bool yaw_only,
                      const Eigen::Vector3d& up_in = Eigen::Vector3d::UnitZ()) {
    Obb box;
    box.num_points = static_cast<int>(idx.size());
    if (idx.empty()) return box;

    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (int i : idx) mean += pts[i].cast<double>();
    mean /= static_cast<double>(idx.size());
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (int i : idx) {
        const Eigen::Vector3d d = pts[i].cast<double>() - mean;
        cov += d * d.transpose();
    }
    cov /= static_cast<double>(idx.size());

    Eigen::Matrix3d R;
    if (yaw_only) {
        const Eigen::Vector3d up = up_in.normalized();
        Eigen::Vector3d e1 = Eigen::Vector3d::UnitX() - Eigen::Vector3d::UnitX().dot(up) * up;
        if (e1.norm() < 1e-3) e1 = Eigen::Vector3d::UnitY() - Eigen::Vector3d::UnitY().dot(up) * up;
        e1.normalize();
        const Eigen::Vector3d e2 = up.cross(e1);
        double c11 = 0.0, c12 = 0.0, c22 = 0.0;
        for (int i : idx) {
            const Eigen::Vector3d d = pts[i].cast<double>() - mean;
            const double a = d.dot(e1), b = d.dot(e2);
            c11 += a * a; c12 += a * b; c22 += b * b;
        }
        const double yaw = 0.5 * std::atan2(2.0 * c12, c11 - c22);
        const Eigen::Vector3d x_axis = std::cos(yaw) * e1 + std::sin(yaw) * e2;
        R.col(0) = x_axis;
        R.col(1) = up.cross(x_axis);
        R.col(2) = up;
    } else {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
        Eigen::Vector3d a0 = es.eigenvectors().col(2);  // largest variance
        Eigen::Vector3d a1 = es.eigenvectors().col(1);
        R.col(0) = a0;
        R.col(1) = a1;
        R.col(2) = a0.cross(a1);
    }

    Eigen::Vector3d lo = Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
    Eigen::Vector3d hi = Eigen::Vector3d::Constant(std::numeric_limits<double>::lowest());
    for (int i : idx) {
        const Eigen::Vector3d q = R.transpose() * (pts[i].cast<double>() - mean);
        lo = lo.cwiseMin(q);
        hi = hi.cwiseMax(q);
    }
    box.size = (hi - lo).cwiseMax(0.02);
    box.center = mean + R * (0.5 * (lo + hi));
    box.rotation = Eigen::Quaterniond(R);
    box.rotation.normalize();
    return box;
}

}  // namespace sins
