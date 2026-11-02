// ground_grid.hpp
// ─────────────────────────────────────────────────────────────────────────────
// Terrain-following ground model for preprocessing_node (ground_model: grid), the alternative to the per-camera floor
// planes. A height map grown outward from the robot:
//
//   cells        square cells of `cell` metres, fixed in the global (odom) frame; each scan, every LiDAR return and
//                every recent camera depth point falls into one.
//   surface      a cell's candidate ground height: its returns with the local slope removed (heights referred to the
//                cell centre along the gradient of the cell it is reached from), the lowest one not more than `tol`
//                below the prediction, then the median of those within `band` above it (the lowest return alone lies
//                under the surface by the sensor noise; deeper ones are reflections/multipath).
//   seeds        the cells under the robot (within robot_radius): the robot stands on the ground, so its base plane
//                (height and slope, from the odometry's attitude) IS the terrain there. What the sensors see under it
//                is used if it agrees within 0.3 m; otherwise the base plane fills the LiDAR's blind cone.
//   growth       cells are visited nearest-first; each is reached from its already-set neighbour closest to the robot.
//                Prediction = neighbour height + neighbour gradient x step. The surface is ground if it lies within
//                `tol` of the prediction: the cell takes it and the gradient is updated (capped at max_grad). Higher
//                is an object (a leg, a table top, a boulder): the cell keeps the predicted height and half the
//                gradient, so a smooth slope is followed however steep (up to max_grad) while a step never is.
//   memory       accepted ground heights are remembered `memory_s` seconds, so ground seen earlier stays known under
//                a person or a robot passing over it.
//   query        height_above(p) = p.z - (h + g . (p.xy - centre)): each cell is a small tilted plane. NaN outside
//                the grid (the caller falls back to its planes).
//
// Prototype and evaluation: eval/diagnostics/ground_grid_proto.py (CODa terrain labels), ground_grid_bag.py (videos).
// Thread-safe (one mutex): preprocessing_node feeds camera points and LiDAR scans from different callbacks.

#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sins {

class GroundGrid {
public:
    struct Params {
        double cell = 0.25;          // [m]
        double tol = 0.15;           // [m] accepted |surface - predicted|
        double max_grad_deg = 40.0;  // [deg] cap on the terrain slope the grid follows
        double radius = 15.0;        // [m] half-size of the grid around the robot
        double memory_s = 30.0;      // [s]
        double robot_radius = 0.6;   // [m] cells seeded from the robot's base plane
        double band = 0.10;          // [m] surface = median of the returns within this band above the lowest valid one
        double grad_alpha = 0.5;     // gradient update gain along the growth direction
        double camera_max_age = 0.3; // [s] camera points older than this are not used for a scan
    };

    explicit GroundGrid(const Params& p) : p_(p) {
        R_ = static_cast<int>(std::ceil(p_.radius / p_.cell));
        W_ = 2 * R_ + 1;
        // visiting order: window offsets sorted by distance from the centre
        for (int i = -R_; i <= R_; ++i)
            for (int j = -R_; j <= R_; ++j) order_.emplace_back(i, j);
        std::stable_sort(order_.begin(), order_.end(), [](const auto& a, const auto& b) {
            return a.first * a.first + a.second * a.second < b.first * b.first + b.second * b.second;
        });
        gmax_ = std::tan(p_.max_grad_deg * M_PI / 180.0);
        h_.assign(static_cast<size_t>(W_) * W_, std::numeric_limits<double>::quiet_NaN());
        g_.assign(static_cast<size_t>(W_) * W_, Eigen::Vector2d::Zero());
    }

    // Latest depth points of one camera, already in the global frame.
    void setCameraPoints(const std::string& cam, std::vector<Eigen::Vector3d> pts, double t) {
        std::lock_guard<std::mutex> lk(m_);
        cams_[cam] = std::make_pair(t, std::move(pts));
    }

    // One LiDAR scan (global frame) + the robot's pose (global <- base) at its time.
    void update(const std::vector<Eigen::Vector3d>& lidar, const Eigen::Matrix4d& T_global_base, double t) {
        std::lock_guard<std::mutex> lk(m_);
        const double c = p_.cell;
        const Eigen::Vector3d rp = T_global_base.block<3, 1>(0, 3);
        ci_ = static_cast<int64_t>(std::floor(rp.x() / c));
        cj_ = static_cast<int64_t>(std::floor(rp.y() / c));
        // points per window cell: offset from the cell centre (x, y) and height
        std::vector<std::vector<Eigen::Vector3d>> cells(static_cast<size_t>(W_) * W_);
        auto add = [&](const Eigen::Vector3d& q) {
            const int64_t i = static_cast<int64_t>(std::floor(q.x() / c)), j = static_cast<int64_t>(std::floor(q.y() / c));
            const int64_t di = i - ci_, dj = j - cj_;
            if (std::abs(di) > R_ || std::abs(dj) > R_) return;
            cells[idx(static_cast<int>(di), static_cast<int>(dj))].emplace_back(
                q.x() - (static_cast<double>(i) + 0.5) * c, q.y() - (static_cast<double>(j) + 0.5) * c, q.z());
        };
        for (const auto& q : lidar) add(q);
        for (const auto& kv : cams_)
            if (t - kv.second.first <= p_.camera_max_age && kv.second.first - t <= p_.camera_max_age)
                for (const auto& q : kv.second.second) add(q);
        // memory: drop the old and the far
        for (auto it = mem_.begin(); it != mem_.end();) {
            const int64_t i = it->first.first, j = it->first.second;
            if (t - it->second.second > p_.memory_s || std::abs(i - ci_) > 2 * R_ || std::abs(j - cj_) > 2 * R_)
                it = mem_.erase(it);
            else
                ++it;
        }
        std::fill(h_.begin(), h_.end(), std::numeric_limits<double>::quiet_NaN());
        std::fill(g_.begin(), g_.end(), Eigen::Vector2d::Zero());
        // robot base plane: z = rp.z + g0 . (xy - rp.xy), g0 from the base_link z axis
        const Eigen::Vector3d nz = T_global_base.block<3, 1>(0, 2);
        Eigen::Vector2d g0 = Eigen::Vector2d::Zero();
        if (nz.z() > 0.5) g0 = Eigen::Vector2d(-nz.x() / nz.z(), -nz.y() / nz.z());
        const int nr = static_cast<int>(std::ceil(p_.robot_radius / c));
        for (const auto& o : order_) {
            const int di = o.first, dj = o.second;
            const size_t k = idx(di, dj);
            const Eigen::Vector2d centre((static_cast<double>(ci_ + di) + 0.5) * c, (static_cast<double>(cj_ + dj) + 0.5) * c);
            if (std::abs(di) <= nr && std::abs(dj) <= nr) {
                const double zb = rp.z() + g0.dot(centre - rp.head<2>());
                double z;
                const bool ok = surface(cells[k], zb - 0.3, g0, z) && z <= zb + 0.3;
                h_[k] = ok ? z : zb;
                g_[k] = g0;
                continue;
            }
            // parent: the already-set neighbour closest to the robot
            int pi = 0, pj = 0;
            int best = std::numeric_limits<int>::max();
            for (int a = -1; a <= 1; ++a)
                for (int b = -1; b <= 1; ++b) {
                    if (!a && !b) continue;
                    const int ni = di + a, nj = dj + b;
                    if (std::abs(ni) > R_ || std::abs(nj) > R_) continue;
                    if (std::isnan(h_[idx(ni, nj)])) continue;
                    const int d2 = ni * ni + nj * nj;
                    if (d2 < best) { best = d2; pi = ni; pj = nj; }
                }
            if (best == std::numeric_limits<int>::max()) continue;
            const size_t kp = idx(pi, pj);
            const Eigen::Vector2d step(static_cast<double>(di - pi) * c, static_cast<double>(dj - pj) * c);
            const double pred = h_[kp] + g_[kp].dot(step);
            const double lo = pred - p_.tol, hi = pred + p_.tol;
            double z;
            bool ok = surface(cells[k], lo, g_[kp], z) && z <= hi;
            if (!ok) {
                const auto m = mem_.find({ci_ + di, cj_ + dj});
                if (m != mem_.end() && m->second.first >= lo && m->second.first <= hi) {
                    z = m->second.first;
                    ok = true;
                }
            }
            if (ok) {
                h_[k] = z;
                const double dl = step.norm();
                const Eigen::Vector2d u = step / dl;
                Eigen::Vector2d gn = g_[kp] + p_.grad_alpha * ((z - h_[kp]) / dl - g_[kp].dot(u)) * u;
                const double nrm = gn.norm();
                if (nrm > gmax_) gn *= gmax_ / nrm;
                g_[k] = gn;
                if (!cells[k].empty()) mem_[{ci_ + di, cj_ + dj}] = std::make_pair(z, t);
            } else {
                h_[k] = pred;                  // under an object: keep the trend, let it fade
                g_[k] = 0.5 * g_[kp];
            }
        }
        valid_ = true;
    }

    // Height of p above the ground model; NaN outside the grid or before the first update.
    double heightAbove(const Eigen::Vector3d& p) const {
        std::lock_guard<std::mutex> lk(m_);
        if (!valid_) return std::numeric_limits<double>::quiet_NaN();
        const double c = p_.cell;
        const int64_t i = static_cast<int64_t>(std::floor(p.x() / c)), j = static_cast<int64_t>(std::floor(p.y() / c));
        const int64_t di = i - ci_, dj = j - cj_;
        if (std::abs(di) > R_ || std::abs(dj) > R_) return std::numeric_limits<double>::quiet_NaN();
        const size_t k = idx(static_cast<int>(di), static_cast<int>(dj));
        if (std::isnan(h_[k])) return std::numeric_limits<double>::quiet_NaN();
        const Eigen::Vector2d off(p.x() - (static_cast<double>(i) + 0.5) * c, p.y() - (static_cast<double>(j) + 0.5) * c);
        return p.z() - (h_[k] + g_[k].dot(off));
    }

    // Cell centres with their ground height (debug output).
    std::vector<Eigen::Vector3d> cellsForDisplay() const {
        std::lock_guard<std::mutex> lk(m_);
        std::vector<Eigen::Vector3d> out;
        if (!valid_) return out;
        out.reserve(h_.size());
        for (int di = -R_; di <= R_; ++di)
            for (int dj = -R_; dj <= R_; ++dj) {
                const double z = h_[idx(di, dj)];
                if (!std::isnan(z))
                    out.emplace_back((static_cast<double>(ci_ + di) + 0.5) * p_.cell,
                                     (static_cast<double>(cj_ + dj) + 0.5) * p_.cell, z);
            }
        return out;
    }

    const Params& params() const { return p_; }

private:
    size_t idx(int di, int dj) const { return static_cast<size_t>(di + R_) * W_ + static_cast<size_t>(dj + R_); }

    // A cell's surface along gradient g: heights referred to the centre (z - g . offset), the lowest one >= lo, then
    // the median of those within band above it.
    bool surface(const std::vector<Eigen::Vector3d>& pts, double lo, const Eigen::Vector2d& g, double& z) const {
        if (pts.empty()) return false;
        std::vector<double> v;
        v.reserve(pts.size());
        for (const auto& q : pts) {
            const double zc = q.z() - g.x() * q.x() - g.y() * q.y();
            if (zc >= lo) v.push_back(zc);
        }
        if (v.empty()) return false;
        std::sort(v.begin(), v.end());
        const double top = v.front() + p_.band;
        const size_t n = static_cast<size_t>(std::upper_bound(v.begin(), v.end(), top) - v.begin());
        z = (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
        return true;
    }

    Params p_;
    int R_{0}, W_{0};
    double gmax_{0.0};
    std::vector<std::pair<int, int>> order_;
    std::vector<double> h_;
    std::vector<Eigen::Vector2d> g_;
    int64_t ci_{0}, cj_{0};
    bool valid_{false};
    std::map<std::pair<int64_t, int64_t>, std::pair<double, double>> mem_;
    std::unordered_map<std::string, std::pair<double, std::vector<Eigen::Vector3d>>> cams_;
    mutable std::mutex m_;
};

}  // namespace sins
