/*
    FILE: tracker_node.cpp
    ---------------------------------
    Tracks dbscan_detector_node's fused objects (.../fused/detections3d) across scans — a reimplementation of
    jo_detector's own dynamicDetector tracking core (kalmanFilterMatrixAccV2 / boxAssociation /
    computeAssociationScore / assignMatchesGlobally), carried over as closely as that core generalizes to v2's
    already-fused, oriented boxes:

      - KALMAN FILTER: same constant-ACCELERATION model and position-only observation as v1's
        kalmanFilterMatrixAccV2 (state per axis = position, velocity, acceleration; only position is observed,
        velocity/acceleration are estimated internally) — EXTENDED to z (v1 only ever tracked x,y), per explicit
        request: state = [x,y,z, vx,vy,vz, ax,ay,az], 9-dim, observation = [x,y,z]. dt is the REAL elapsed time
        between two fused messages' own stamps (v1 used a fixed nominal time_step instead — this is the one
        deliberate improvement over v1, not a v1 quirk worth copying). Q/R are constant, not dt-scaled, matching
        kalmanFilterMatrixAccV2 literally.

      - ASSOCIATION: the same multi-term weighted score v1's computeAssociationScore builds — distance to the
        KALMAN-PREDICTED position, distance to the track's PREVIOUS raw observation, 2D IoU against each of those
        two references, and a relative-size-difference term — each with its own weight, gated by a hard reject on
        max match distance/speed/size (track_max_match_dist/speed/size_diff, same names/values as v1's
        max_match_range/max_match_speed/max_relative_size_diff_match). IoU2D uses each OBB's AXIS-ALIGNED
        world-frame footprint (the bounding rectangle of its rotated corners) — the natural generalization of
        v1's own x_width/y_width formula, which was already axis-aligned since v1's boxes carried no orientation.
        Global assignment is a max-weight bipartite matching (Hungarian, hungarianMinCost) over the scored
        candidates; v1's own augmenting-path matching (Kuhn's algorithm, best-score-first) stays available as
        track_assignment_method: greedy_kuhn — not just a greedy nearest-neighbour pick either way.

      - LIFECYCLE: hits accumulate on every match (never reset by a miss, matching v1's own hitStreak_ handling);
        a track is CONFIRMED once hits >= track_min_hits AND (found during the broader v1 review that produced
        the next two bullets, a deliberate BEHAVIOR CHANGE from this file's own earlier bare-hit-count gate —
        v1: shouldConfirmTrack) it is EITHER classified (this tick, ever, or via the LiDAR-only pixel fallback —
        confirms immediately, no motion needed, same as before) OR shows actual observed motion above
        track_stationary_speed_thresh (a non-classified track that's simply been reliably re-detected while
        dead still does not confirm) — and STAYS confirmed until deleted once reached (v1's own "no active
        de-confirmation" comment) — dropped after track_max_missed consecutive missed ticks (track_max_missed_classified,
        longer, for a track carrying a whitelisted class — v1: maxMissedFramesYolo_ vs maxMissedFrames_, see the
        constructor's own comment). A track propagated only by KALMAN PREDICTION this tick (no match) is never
        published, matching v1's own "internal propagation, never in output" — true whether it's coasting on the
        plain or the classified bar. Size is the LATEST matched detection's own size, not smoothed or max-held
        (v1 does the same: currDetectedBBox.x_width/y_width/z_width copied straight through each match).

      - DUPLICATE-TRACK SUPPRESSION (v1's isCloseToExistingTrack/isDetectionDuplicateOfPrediction, found during
        the broader v1 review): before spawning a brand-new track for an UNMATCHED detection, check it isn't
        actually a second, fragmented candidate for an object that already has a track — the main association
        is a GLOBAL one-to-one assignment, so when two detections this tick are both plausible for the same
        real object, only one wins it, and without this check the leftover would spawn a spurious duplicate.

      - TWO SPEED BARS, not one: v1 itself never compares a single speed against a single threshold — dynaVelThresh_
        (0.35, the DYNAMIC label) and stationarySpeedThresh_ (0.10, "is this moving AT ALL") are two separate v1
        params. An earlier version of this file collapsed them into one (0.2) and, combined with process noise
        picked without checking v1's own real defaults, let a lot of ordinary per-tick OBB-refit jitter cross
        straight into "dynamic" — see track_dynamic_speed_thresh_/stationary_speed_thresh_ and the Kalman noise
        constructor comment below for the actual fix. A track between the two bars still publishes ("static"),
        it just isn't labelled dynamic and isn't aged toward death either (see next point).

      - DYNAMIC LABEL DEBOUNCE + CLASSIFICATION ROBUSTNESS (v1's classificationCB, PATHs 1-3, ALL FOUR pieces
        ported per explicit request — see onFused()'s matched-branch comment and publish()'s own): speed must
        stay >= track_dynamic_speed_thresh for track_dynamic_confirm_ticks CONSECUTIVE ticks (dynamic_ticks)
        before the label flips — v1's PATH 3c anti-flicker consistency check. The motion-evidence test feeding
        dynamic_ticks itself mirrors v1's own PATH 1/3 split: a CLASSIFIED track (this tick or ever) needs only
        the plain speed check (PATH 1 — YOLO-certified tracks classify on KF speed alone); an UNCLASSIFIED track
        additionally needs corroboration from EITHER of:
          - KF-VELOCITY CONFIDENCE (v1 PATH 3a, dynaKfVelStdRatio_): a track's recent speed samples having a
            mean above the bar AND a small std relative to that mean — a genuinely moving object's single noisy
            dip doesn't reset the streak.
          - POINT-CLOUD MOTION VOTING (v1 PATH 3b, pointCloudVoteRatio()): per-point nearest-neighbor voting
            between this track's own points now and from the last matched tick, corroborating the KF estimate
            with the raw geometry directly. Needed plumbing dbscan_detector_node's fused/clusters cloud through
            to this node (a det_idx field identifying which points belong to which detection — see that node's
            own comment on it) and subscribing to it here — the one piece of this whole port that IS a real
            architecture change, not just a classification tweak, done because it was explicitly asked for.
        STICKY DYNAMIC / HISTORICAL CONTINUITY (v1 PATH 2, forceDynaFrames_/forceDynaCheckRange_) is the
        equivalent of PATH 2's bypass: once enough of a track's recent history was dynamic, the label
        re-confirms on continued motion evidence without re-running the full fresh debounce after one bad tick.
        SIZE-SANITY GATE (v1's constrainSize_/targetObjectSize_, defaulted OFF in v1, ON here — this session's
        whole arc has been about false "dynamic" positives): an implausibly-sized moving blob never gets the
        DYNAMIC label. Scoped to UNCLASSIFIED tracks only (a v2 addition beyond v1's own design, enabled by
        having semantic classification v1 never did): a track YOLO has already identified doesn't need a
        generic size heuristic and shouldn't be second-guessed by one.

      - DISOCCLUSION-ARTIFACT SUPPRESSION (isDisocclusionArtifact() — NOT from v1, the user's own design from a
        separate discussion, applying ERASOR/DUFOMap's free/occupied/unknown distinction as one targeted
        geometric check instead of a full occupancy grid): when a track moves away from a spot, whatever was
        BEHIND it along the LiDAR's own line of sight becomes newly visible and can read as an apparent jump/
        velocity for something that is actually static — the classic "person walks past a wall, the wall patch
        behind them reappears and looks like it moved" artifact. A candidate is flagged when it lies close to
        the same LiDAR ray through some OTHER track's own recent position AND is noticeably farther from the
        LiDAR than that position was. When flagged, this tick simply does not count toward dynamic_ticks (the
        match/track itself is unaffected — the object is almost certainly real and static, just newly visible).
        Exempted (on top of classified_now) whenever the track's own dynamic_hist shows it was moving before
        this tick — a crowded-scene refinement: the SAME geometric signature (same ray, farther range, occluder
        moving) is produced when a real dynamic/potentially-dynamic object D/PD is revealed again after a
        different dynamic object D walked in front of it, not just by a revealed static wall, and classified_now
        alone doesn't catch a D/PD track whose class was cleared by the FOV-based decay trigger during a long
        enough occlusion.

      - OCCLUSION-AWARE MISSED-TICKS BUDGET (isOcclusionExplained() — the mirror check, not from v1 either): a
        track that misses this tick because some OTHER, currently-matched track is physically standing between
        it and the LiDAR along roughly the same ray (same crowded D-in-front-of-D/PD scenario as above, from the
        occluded side instead of the revealed side) does not burn its missed_in_a_row budget for that tick —
        without this, an occlusion that happens to outlast max_missed_classified_ kills the occluded track
        outright and forces a brand-new id once the occluder moves on, even though the miss had a known, benign
        physical cause the tracker could see for itself. Scoped to tracks "worth protecting" (classified now, or
        with any dynamic_hist entry) so a run-of-the-mill static/noise track sitting near another track's ray
        still ages out normally.

      - STATIC TRACKS MUST DIE: this detector's whole purpose is finding dynamic/potentially-dynamic objects, so
        a track that settles for good must stop being published and eventually be deleted, not linger forever.
        The trigger condition is v1's own ("suppress stationary non-YOLO tracks", kalmanFilterAndUpdateHist):
        a CONFIRMED, matched-this-tick track is only published if its horizontal speed exceeds
        track_stationary_speed_thresh (the LOWER bar) OR it has semantic evidence of a potentially-dynamic class
        (best_class non-empty — this v2's equivalent of v1's is_yolo_candidate exemption), unless
        track_publish_stationary overrides the whole rule (v1's trackSteadyObjects_). What v1 itself does NOT do:
        v1 keeps a suppressed track's Kalman state alive forever, unbounded. Here, consecutive suppressed ticks
        are counted (stationary_ticks, reset the instant it moves or gets a class vote) and the track is fully
        deleted past track_stationary_max_ticks — the one deliberate addition beyond v1 in this file, made
        explicitly because never freeing a settled object's track/id contradicts the point of a DYNAMIC-object
        detector.

      - HORIZONTAL-ONLY SPEED: both bars above and the reported speed use sqrt(vx^2+vy^2) only — see
        planarSpeed()'s own comment on why vz is excluded (sensor shadow-cone noise, not real vertical motion).
        vz is still estimated and still published raw in results[0]'s pose field, just never used for
        classification.

      - ID-SWAP GUARD (brought back from v1's match_yolo_class_consistency_weight/yolo_track_assoc_bonus, which
        an earlier version of this file dropped along with the rest of v1's FOV-specific machinery — this one
        piece generalizes fine and the user asked for it explicitly): in the association loop, a candidate
        pairing a detection classified as X against a track already carrying a DIFFERENT class Y is a HARD
        reject, not a soft penalty — see the association loop's own comment. A person track can not be stolen by
        a nearby differently-classified object (or a generic unclassified one, when a differently-classified
        detection is actually the better match) while both are visible to a camera; agreement instead adds
        track_class_match_bonus so the correct pairing wins the global assignment over an unclassified competitor
        at similar distance.

      - SEMANTIC DECAY (v1's own two-part mechanism, dynamicDetector.cpp's "YOLO decay (FOV-based)" / "YOLO decay
        (base size mismatch)", brought back closely per explicit request — see decaySemanticEvidence()): the ID-
        swap guard above only stops a BAD match from happening; this stops a track that already HAS a class from
        going stale and keeping it forever once v1-style voting alone no longer agrees it should. Three
        triggers (v1 has the first two; the third has no v1 equivalent at all and has been removed and
        re-added once already — see decaySemanticEvidence()'s own comment for the full back-and-forth):
          1. INSIDE any camera's FOV (own per-camera pinhole projection, replicated from v1's isInCameraFOV, OR'd
             across front_camera/back_camera since v1 only ever had one camera): if a classified track goes
             track_max_non_yolo_in_fov_frames consecutive MATCHED ticks with NO fresh YOLO evidence while
             plainly visible to a camera, its class is cleared (not just unpublished — the vote history resets
             too, so a later re-classification starts clean). This is exactly what the user asked for: a track
             cannot keep being "semantic" on old evidence alone while in FOV and YOLO is silently not confirming
             it tick after tick.
          2. Against the largest size ever seen WITH real YOLO evidence (a high-water mark, never shrunk):
             flags GROWTH past it for track_max_yolo_base_mismatch_frames consecutive ticks as an ID swap onto a
             bigger object (wall, vehicle, noisy merged cluster) — v1 only ever ran this outside FOV (see this
             trigger's own comment in decaySemanticEvidence() for why it now runs inside FOV too).
          3. The SAME high-water mark, but for SHRINKAGE, GATED on the YOLO 2D (ByteTrack) track id instead of
             running unconditionally: a box smaller than its own baseline is only trusted as a real ID swap when
             YOLO's OWN 2D tracker also disagrees (this tick's track id differs from — or is absent vs. — the
             one the baseline was built on); when the id matches, the shrink is treated as v1 would treat any
             shrink — a normal physical artifact (occlusion, LiDAR shadow cone, viewing angle), not an ID swap.
        Whitelist-gated like everything else that treats class evidence as "potentially-dynamic" (see
        isDynamicClass()/semantic_dynamic_classes_'s own comment) only in the sense that a track only HAS a
        class to decay in the first place if isDynamicClass() let it become best_class's winner via voteClass()
        being called at all — decaySemanticEvidence() itself runs for ANY non-empty best_class, whitelisted or
        not, same as v1 would if it tracked a specific class string (it doesn't). No v1 equivalent exists for
        WHEN this runs on a missed tick either way — it only runs on a MATCHED tick, same as v1 (verified
        against dynamicDetector.cpp's own kalmanFilterAndUpdateHist: the YOLO-decay block lives entirely inside
        its "1) Track matchate" loop, not its separate "2) Track non matchate" propagation-only one).

      - VELOCITY-DIRECTION GATE (v1's computeVelocityDirectionError/passesVelocityDirectionGate, ported per
        explicit request — see velocityDirectionError() and the constructor's own comment on the three
        collapsed tiers): a candidate whose IMPLIED motion points in a very different direction (or magnitude)
        than the KF's own PREDICTED velocity is suspicious — orthogonal to and independent of the position/
        size/IoU/class terms, both a hard gate (tiered: strict for an unconfirmed track, looser once confirmed,
        further relaxed when semantically classified or outside every camera's FOV) and a soft score penalty.

      - LIDAR-ONLY-OUTSIDE-FOV RELAXATION (v1's isLidarOnlyOutsideFovAssociation + its outside-FOV-specific
        gates/bonuses, ported per explicit request — see the association loop's own comment on the three
        outside_fov_* params): pure LiDAR tracking (both predicted and current position outside every camera's
        FOV, neither side classified) has a noisier DBSCAN centroid with no camera to corroborate it, so v1
        trades a WIDER position/size match tolerance for a NARROWER speed tolerance there (a large implied
        speed is more likely a bad match than genuine fast motion in this regime) plus a stability bonus
        against losing a legitimate track's identity to ordinary centroid jitter.

      - ID-SWAP GUARD (brought back from v1's match_yolo_class_consistency_weight/yolo_track_assoc_bonus, which
        an earlier version of this file dropped along with the rest of v1's FOV-specific machinery — this one
        piece generalizes fine and the user asked for it explicitly): in the association loop, a candidate
        pairing a detection classified as X against a track already carrying a DIFFERENT class Y is a HARD
        reject, not a soft penalty — see the association loop's own comment. A person track can not be stolen by
        a nearby differently-classified object (or a generic unclassified one, when a differently-classified
        detection is actually the better match) while both are visible to a camera; agreement instead adds
        track_class_match_bonus so the correct pairing wins the global assignment over an unclassified competitor
        at similar distance. FOV-gated (v1's own matchYoloClassConsistencyWeight_ behavior): the continuity
        penalty for an unclassified-vs-classified pairing only applies when the track's PREDICTED position is
        inside a camera's FOV — outside it, absence of fresh evidence isn't suspicious, since YOLO could not
        have confirmed anything there anyway.

      - NATURAL-MOTION GATE (v1's isNaturalMotion, ported per explicit request — see isNaturalMotion() and the
        association loop's own comment): a track not yet confirmed AND not semantically classified either side
        has no identity evidence of its own yet, so its very first few matches are held to a stricter bar — the
        implied move must clear a minimum (inside FOV only) and never exceed an outright-implausible maximum,
        and the KF's own prediction innovation must stay within an adaptive bound. An already-confirmed or
        classified track skips this entirely — it has stronger evidence this gate can't improve on.

      - CONFIRMED-TRACK STABILITY BONUS (v1's confirmedTrackAssocBonus_, ported per explicit request): a flat
        score bonus for any already-confirmed track, independent of class or FOV, on top of (not instead of)
        the LiDAR-only-outside-FOV bonus above — a newer or more tentative competitor needs a real edge, not
        just a marginal one, to win a confirmed track's rightful match.

      Orientation (yaw) is carried through unfiltered from the latest matched detection (not Kalman-tracked) —
      v1 had no orientation at all (axis-aligned boxes); the user cares about position/velocity/z, not yaw
      precision, so a plain pass-through is enough.

    SEMANTIC CLASS (voteClass): dbscan_detector_node tags a fused object with a YOLO class name (results[1]) only
    on ticks where a YOLO leaf actually refined it (see that node's own header) — the class is otherwise absent,
    not "unknown". Each such tick casts one vote for that class on whichever TRACK the object matched to this
    tick; the published class is the argmax of the track's own vote histogram, so a single misclassified frame
    doesn't flip the label, and it is what STATIC TRACKS MUST DIE above uses as its "potentially dynamic" evidence
    (a person standing still is still a person). No 2D-track-id matching happens anywhere: a camera handoff
    (front_camera's YOLO id disappearing, back_camera's appearing) is invisible here, because the vote is cast
    against the already-persistent 3D track (matched by position/motion, camera-agnostic), never against any 2D
    id — SEMANTIC DECAY above is what keeps this vote history honest over time instead of accumulating forever.

    LIDAR-ONLY SEMANTIC FALLBACK (pixelCandidates + onFused()'s own resolution block, no v1 equivalent — v1 only
    ever had the one depth-coupled camera, so "LiDAR-only beyond the camera's range" wasn't a reachable state for
    it the way it is here): the depth-leaf mechanism above requires a DEPTH point at the detection's pixels to
    deproject in the first place — a person beyond the depth camera's own range (typically 4.5-5m) produces zero
    leaf points no matter how confidently YOLO sees them on the plain COLOR image, so results[0]'s mask is "L"
    (LiDAR contributed, no camera did) and results[1] never gets set by dbscan_detector_node. For exactly that
    case (mask == "L" AND no results[1] this tick), each such detection's own position is projected straight into
    each camera's pixel space — unlike insideAnyCameraFov's FOV test, no depth_min/depth_max range gate here,
    since a color-image detection isn't limited by the depth sensor's range either — and checked against each
    camera's RAW 2D YOLO boxes (vision_msgs/Detection2DArray, latest message per camera, discarded past
    yolo2d_max_age) directly.
      Monocular projection is depth-blind, though: two genuinely different LiDAR-only objects at different real
    depths along roughly the same camera ray can BOTH land inside the SAME YOLO box — a first version of this
    fallback matched per-track independently and let exactly that happen (a far-away box getting tagged through
    whichever box a near one legitimately matched). A YOLO box can only ever be ONE physical object, so this is
    now resolved GLOBALLY, once per tick, across every fallback-eligible detection at once (matched-track or
    about-to-become-a-new-track, before that distinction is even made): each (camera, YOLO box) pair is awarded
    to its single closest-to-box-center candidate only, everyone else gets nothing from that box this tick.
      A box that already has SOME camera depth coverage never uses this path (deliberately): it already has the
    stronger, depth-corroborated mechanism above, and widening this fallback to it would trade that corroboration
    for a weaker, projection-only match — exactly what this whole session's ID-swap hardening has been guarding
    against.

    Subscribes:  /sins/dbscan_detector/fused/detections3d   vision_msgs/Detection3DArray
    Publishes:   /sins/tracker/tracks                       vision_msgs/Detection3DArray
                 (id = "track_<n>", results[0] = {class_id: "dynamic"|"static", score: horizontal speed [m/s]},
                 results[0].pose.pose.position = (vx, vy, vz) [m/s], vz unused for classification — same "stash
                 extra numbers in the unused pose field" convention jo_detector's own YOLO detection callback
                 uses for pixel coords; results[1] = {class_id: <YOLO class name>, score: vote fraction}, present
                 only once a YOLO leaf has ever refined this track. Only CONFIRMED, matched-this-tick, non-static
                 (or class-exempt) tracks are published — see STATIC TRACKS MUST DIE above.
                 /sins/tracker/tracks_markers                visualization_msgs/MarkerArray
                 (box edges + a velocity arrow + an id/[DYN|POT|STA] tag/class/speed label — dynamic=blue,
                 potentially_dynamic=green, static=neutral grey, v1 parity, see appendTrackMarkers()'s own
                 comment)
                 /perception/tracked_dynamic_obstacles                 jo_msgs/ObstacleArray
                 (pub_obstacles_ — the pre-existing, stable interface glim_ros2's own BBOX dynamic-rejection
                 mode (dynamic_rejection_type=="BBOX" in config_ros.json) already subscribes to; v1's own
                 jo_detector published this same topic/type, GLIM-side needs no change at all to pick v2
                 up instead. Only dynamic/potentially_dynamic tracks are published here — v1's own
                 publishDynamicObstacleArray() skips static tracks outright, matched here. See publish()'s own
                 comment on the field mapping.)
*/
#include <rclcpp/rclcpp.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <jo_msgs/msg/obstacle_array.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/exceptions.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

// Constant-acceleration Kalman filter, state = [x,y,z, vx,vy,vz, ax,ay,az]; observation = [x,y,z]. Mirrors
// jo_detector's kalmanFilterMatrixAccV2 (there: x,y only) — same A/H/Q/R structure, extended to z.
class TrackKF {
public:
    void init(const Eigen::Vector3d& pos, double p_init) {
        x_ = Eigen::VectorXd::Zero(9);
        x_.head<3>() = pos;
        P_ = Eigen::MatrixXd::Identity(9, 9) * p_init;
    }

    void predict(double dt, double q_pos, double q_vel, double q_acc) {
        Eigen::MatrixXd A = Eigen::MatrixXd::Identity(9, 9);
        const double dt2 = 0.5 * dt * dt;
        for (int i = 0; i < 3; ++i) {
            A(i, i + 3) = dt;
            A(i, i + 6) = dt2;
            A(i + 3, i + 6) = dt;
        }
        x_ = A * x_;
        P_ = A * P_ * A.transpose() + Q(q_pos, q_vel, q_acc);
    }

    void update(const Eigen::Vector3d& z, double r_pos) {
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 9);
        H.block<3, 3>(0, 0) = Eigen::MatrixXd::Identity(3, 3);
        const Eigen::MatrixXd R = Eigen::MatrixXd::Identity(3, 3) * r_pos;
        const Eigen::Vector3d y = z - H * x_;
        const Eigen::MatrixXd S = H * P_ * H.transpose() + R;
        const Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();
        x_ += K * y;
        P_ = (Eigen::MatrixXd::Identity(9, 9) - K * H) * P_;
    }

    static Eigen::MatrixXd Q(double q_pos, double q_vel, double q_acc) {
        Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(9, 9);
        Q.diagonal().segment<3>(0).setConstant(q_pos);
        Q.diagonal().segment<3>(3).setConstant(q_vel);
        Q.diagonal().segment<3>(6).setConstant(q_acc);
        return Q;
    }

    Eigen::Vector3d pos() const { return x_.head<3>(); }
    Eigen::Vector3d vel() const { return x_.segment<3>(3); }
    Eigen::Vector3d acc() const { return x_.segment<3>(6); }

    Eigen::VectorXd x_;
    Eigen::MatrixXd P_;
};

// Axis-aligned world-frame footprint of an OBB (bounding rectangle of its 4 rotated corners) — the natural
// generalization of jo_detector's own x_width/y_width, which were already axis-aligned.
struct Footprint {
    double xmin, xmax, ymin, ymax, zmin, zmax;
};

Footprint footprintOf(const geometry_msgs::msg::Point& c, const geometry_msgs::msg::Quaternion& q,
                      const vision_msgs::msg::BoundingBox3D::_size_type& s) {
    const double w = q.w, x = q.x, y = q.y, z = q.z;
    Eigen::Matrix3d R;
    R << 1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y);
    double xmin = 1e18, xmax = -1e18, ymin = 1e18, ymax = -1e18, zmin = 1e18, zmax = -1e18;
    for (int cx : {-1, 1})
        for (int cy : {-1, 1})
            for (int cz : {-1, 1}) {
                const Eigen::Vector3d local(cx * 0.5 * s.x, cy * 0.5 * s.y, cz * 0.5 * s.z);
                const Eigen::Vector3d p = Eigen::Vector3d(c.x, c.y, c.z) + R * local;
                xmin = std::min(xmin, p.x()); xmax = std::max(xmax, p.x());
                ymin = std::min(ymin, p.y()); ymax = std::max(ymax, p.y());
                zmin = std::min(zmin, p.z()); zmax = std::max(zmax, p.z());
            }
    return {xmin, xmax, ymin, ymax, zmin, zmax};
}

double iou2D(const Footprint& a, const Footprint& b) {
    const double ix = std::max(0.0, std::min(a.xmax, b.xmax) - std::max(a.xmin, b.xmin));
    const double iy = std::max(0.0, std::min(a.ymax, b.ymax) - std::max(a.ymin, b.ymin));
    const double inter = ix * iy;
    const double areaA = std::max(0.0, a.xmax - a.xmin) * std::max(0.0, a.ymax - a.ymin);
    const double areaB = std::max(0.0, b.xmax - b.xmin) * std::max(0.0, b.ymax - b.ymin);
    const double uni = areaA + areaB - inter;
    return uni > 1e-9 ? inter / uni : 0.0;
}

// floor: a purely relative diff (|dx|/max(a,b)) punishes small objects far more than large ones for the
// SAME absolute jitter (3cm on a 15cm box is 20%, on a 1.5m box is 2%) — and a small DBSCAN cluster's
// extent is inherently noisier tick-to-tick (a couple of boundary points appearing/disappearing swings it
// a lot in percentage terms) since Track::size is copied raw from each match with no smoothing. Clamping
// the denominator to at least `floor` makes the gate compare against a realistic minimum object scale
// instead of the box's own (possibly tiny, possibly noisy) size — this is what actually protects small
// objects from spurious match rejection -> fallback to a nearby wrong candidate, not the DBSCAN radius.
// HORIZONTAL-ONLY, like planarSpeed()'s own vz exclusion (see its own comment): Z is dominated by sensor
// shadow-cone artifacts, not just in the noise-floor sense that justifies dropping vz from speed, but
// systematically near the robot — the LiDAR's own near-field blind cone cuts into a close object's own LOWER
// extent (e.g. a person's legs), shrinking its box's Z size specifically as it gets close, regardless of
// whether the object itself changed at all. Averaging that into a 3-axis size-difference score (the original
// (dx+dy+dz)/3) could inflate the match-rejection gate right as an object nears the robot — exactly the worst
// moment to lose a track. X/Y footprint is comparatively robust to this (the shadow cone acts along Z, not
// radially), so it alone carries this comparison now, same as every other association-time X/Y-vs-Z split in
// this file.
double relSizeDiff(const vision_msgs::msg::BoundingBox3D::_size_type& a, const vision_msgs::msg::BoundingBox3D::_size_type& b, double floor) {
    const double dx = std::abs(a.x - b.x) / std::max({a.x, b.x, floor});
    const double dy = std::abs(a.y - b.y) / std::max({a.y, b.y, floor});
    return (dx + dy) / 2.0;
}

struct Track {
    int id = 0;
    TrackKF kf;
    vision_msgs::msg::BoundingBox3D::_size_type size;   // latest matched detection's own size (no smoothing)
    geometry_msgs::msg::Quaternion orientation;          // latest matched detection's own orientation (unfiltered)
    geometry_msgs::msg::Point last_obs;                  // last matched detection's RAW position (v1's "previous observation" reference)
    double last_obs_stamp = 0.0;                         // [s] stamp of last_obs — see velocityDirectionError()
    int hits = 0;               // total matches since creation (never reset by a miss)
    int missed_in_a_row = 0;
    bool confirmed = false;

    // Consecutive ticks this (confirmed, matched) track was slow AND had no semantic evidence of being a
    // potentially-dynamic class — see the file header's own section on why static tracks must actually die.
    // Reset to 0 the moment it moves fast enough, or gets a class vote. v1 itself never resets this to 0 in the
    // sense of ever deleting the track — this counter and the death it triggers are the one deliberate addition
    // beyond v1 (see the header).
    int stationary_ticks = 0;
    int dynamic_ticks = 0;   // consecutive ticks with (horizontal) speed >= track_dynamic_speed_thresh — see dynamic_confirm_ticks_

    // Semantic class: a majority vote across every tick that carried a YOLO class for this track (see
    // dbscan_detector_node's own "yolo_refined"/"yolo_class" fields on its fused objects), NOT tied to any
    // particular camera or 2D track id — whichever camera currently sees the object just adds a vote, so a
    // handoff from one camera's FOV to the other's is a non-event here: the 3D track's own identity (from
    // position/motion association) is what persists, the class label just keeps getting re-confirmed onto it.
    std::map<std::string, int> class_votes;
    std::string best_class;
    // First-classification debounce (no v1 equivalent — see voteClass()'s own comment): candidate class/streak
    // for a track that has NEVER been classified yet, kept separate from class_votes/best_class so a single
    // contaminated tick can't set best_class outright.
    std::string pending_class;
    int pending_class_ticks = 0;
    int pending_strong_ticks = 0;  // consecutive pending votes from depth-leaf-direct evidence — see voteClass()
    // Class handover (see handOverClass()): this track was born from a detection that a nearby CLASSIFIED track
    // passed over for another one — if that track turns out to have swapped identity, its class comes here.
    // Smoothed TOP height (box max z, global frame) of the matched detections — see the association's height term.
    double top_z = 0.0;
    int top_samples = 0;
    int heir_of = -1;
    std::string heir_class;
    double heir_stamp = 0.0;
    geometry_msgs::msg::Point pending_class_start_pos;  // raw position when the current streak began — see voteClass()

    // FOV-aware semantic decay state (v1: nonYoloInFovStreak_ / yoloXWidth_ / yoloYWidth_ / yoloBaseMismatchStreak_
    // — see decaySemanticEvidence()'s own comment and the file header's SEMANTIC DECAY section).
    int non_yolo_in_fov_streak = 0;
    // MOTION-EARNED PD: published-dynamic ticks of a track that never had a class, and whether that history has
    // earned it the potentially_dynamic label once it stops (see publish()). Revoked like a class (decay below).
    int motion_dynamic_ticks = 0;
    bool motion_pd = false;
    int motion_pd_fov_streak = 0;
    double yolo_x_width = 0.0, yolo_y_width = 0.0;  // high-water mark: largest size seen WITH fresh YOLO evidence
    int yolo_base_mismatch_streak = 0;
    int last_yolo_track_id = -1;  // the YOLO 2D track id active when yolo_x_width/yolo_y_width were last updated — see decaySemanticEvidence()'s own comment

    // Historical continuity ("sticky dynamic") + KF-velocity confidence — v1's PATH 2
    // (forceDynaFrames_/forceDynaCheckRange_) and PATH 3a (dynaKfVelStdRatio_) of its classification
    // machinery, see the constructor's own comment. dynamic_hist holds this track's last N post-debounce
    // dynamic/static decisions (pushed once per matched tick); speed_hist holds its last N raw horizontal-
    // speed samples, for the mean/std confidence check.
    std::deque<bool> dynamic_hist;
    std::deque<double> speed_hist;

    // Point-cloud motion voting (v1 PATH 3b) + disocclusion-artifact detection (not from v1, see
    // isDisocclusionArtifact()'s own comment) — see pointCloudVoteRatio()'s own comment for prev_points, and
    // the constructor's own comment for pos_hist (a short window of this track's own recent raw observations).
    std::vector<Eigen::Vector3f> prev_points;
    std::deque<Eigen::Vector3d> pos_hist;
};

// Kuhn's algorithm augmenting-path search — mirrors jo_detector's own tryAugmentDetectionMatch/
// assignMatchesGlobally: candidates arrive best-score-first, each detection tries its candidate tracks in that
// order and steals one from a worse-matched detection if needed.
bool tryAugment(size_t det, const std::vector<std::vector<size_t>>& candidates_by_det, std::vector<int>& track_to_det,
                std::vector<int>& det_to_track, std::vector<int>& visited, int token) {
    for (size_t tr : candidates_by_det[det]) {
        if (visited[tr] == token) continue;
        visited[tr] = token;
        if (track_to_det[tr] == -1 || tryAugment(static_cast<size_t>(track_to_det[tr]), candidates_by_det, track_to_det, det_to_track, visited, token)) {
            track_to_det[tr] = static_cast<int>(det);
            det_to_track[det] = static_cast<int>(tr);
            return true;
        }
    }
    return false;
}

// Weighted bipartite matching (Kuhn-Munkres/Hungarian, O(n^3)) — the default assignment; alternative to tryAugment's
// max-CARDINALITY heuristic above (see track_assignment_method and its call site's own comment). Minimizes
// total cost over a SQUARE matrix; the caller pads a rectangular (nt x nd) problem to n = max(nt, nd) and
// marks non-candidate pairs with a sentinel the real -score costs never reach, so the solver only uses one
// when truly forced to (never, once padding rows/columns give it a free zero-cost way to leave a real
// track/detection unmatched instead). Standard textbook formulation (see e.g. cp-algorithms.com/graph/hungarian-algorithm.html) — not
// specific to this codebase, reproduced here instead of adding a dependency for one O(n^3) routine over
// a handful of tracks/detections.
std::vector<int> hungarianMinCost(const std::vector<std::vector<double>>& cost) {
    const int n = static_cast<int>(cost.size());
    const double INF = std::numeric_limits<double>::max() / 4;
    std::vector<double> u(n + 1, 0.0), v(n + 1, 0.0);
    std::vector<int> p(n + 1, 0), way(n + 1, 0);
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(n + 1, INF);
        std::vector<bool> used(n + 1, false);
        do {
            used[j0] = true;
            const int i0 = p[j0];
            int j1 = -1;
            double delta = INF;
            for (int j = 1; j <= n; ++j) {
                if (used[j]) continue;
                const double cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else { minv[j] -= delta; }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    std::vector<int> row_to_col(n, -1);
    for (int j = 1; j <= n; ++j) {
        if (p[j] > 0) row_to_col[p[j] - 1] = j - 1;
    }
    return row_to_col;
}

}  // namespace

// Per-camera pinhole model + live pose, for the FOV test SEMANTIC DECAY needs (v1's own isInCameraFOV, here OR'd
// across however many cameras are configured instead of v1's single camera). Intrinsics come from that camera's
// own color CameraInfo (YOLO runs on the color image, same frame convention as dbscan_detector_node's leaves);
// pose is a plain TF lookup against the SAME "<camera>_refined" frame the rest of this package already aligns
// into (calibration_icp_node), refreshed once per tick (refreshCameraTf()), not per track.
struct CamFov {
    std::string name, frame_id;
    double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;
    int width = 0, height = 0;
    bool has_info = false;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub;
    // per-tick cache
    bool tf_ok = false;
    Eigen::Isometry3d T_cam_global = Eigen::Isometry3d::Identity();  // maps a GLOBAL-frame point into this camera's frame

    // Raw 2D YOLO detections (pixel boxes, no depth involved) — see yoloClassAtPixel()'s own comment for why
    // this exists alongside the depth-leaf mechanism. Just the latest message, like preprocessing_node's own
    // yolo_ring but simpler: this is a fallback path only, not the primary (depth-corroborated) one.
    rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr sub_2d;
    vision_msgs::msg::Detection2DArray::ConstSharedPtr latest_2d;
};

class TrackerNode : public rclcpp::Node {
public:
    TrackerNode() : Node("tracker_node"), tf_buffer_(get_clock()), tf_listener_(tf_buffer_) {
        global_frame_ = declare_parameter<std::string>("global_frame", "odom");

        max_match_dist_ = declare_parameter<double>("track_max_match_dist", 2.0);       // v1: max_match_range
        max_match_speed_ = declare_parameter<double>("track_max_match_speed", 8.0);      // v1: max_match_speed
        max_size_diff_ = declare_parameter<double>("track_max_size_diff", 0.8);          // v1: max_relative_size_diff_match
        // Absolute floor for relSizeDiff's denominator (see its own comment) — v1 has no equivalent (its
        // real-world objects were rarely this small), a deliberate v2 addition against small-object ID swaps.
        min_size_diff_floor_ = declare_parameter<double>("track_min_size_diff_floor", 0.15);
        w_pos_ = declare_parameter<double>("track_match_pos_weight", 1.0);               // v1: match_pos_score_weight
        w_size_ = declare_parameter<double>("track_match_size_weight", 0.35);            // v1: match_size_score_weight
        w_iou_ = declare_parameter<double>("track_match_iou_weight", 0.15);              // v1: match_iou2d_score_weight
        w_prev_pos_ = declare_parameter<double>("track_match_prev_pos_weight", 0.25);    // v1: match_prev_obs_pos_score_weight
        w_prev_iou_ = declare_parameter<double>("track_match_prev_iou_weight", 0.10);    // v1: match_prev_obs_iou2d_score_weight
        min_match_score_ = declare_parameter<double>("track_min_match_score", -2.5);     // v1: min_match_score

        min_hits_ = declare_parameter<int>("track_min_hits", 4);                         // v1: min_confirm_hits
        max_missed_ = declare_parameter<int>("track_max_missed", 5);                     // v1: max_missed_frames
        // Classified tracks coast longer (v1: maxMissedFramesYolo_ vs maxMissedFrames_, a 3x ratio there —
        // kept here applied to THIS file's own max_missed_ above, whatever it's set to): a track with a
        // WHITELISTED class (isDynamicClass(), same whitelist as everywhere else that treats class evidence as
        // "potentially-dynamic" — v1's own is_yolo_candidate is itself whitelist-filtered at ingestion, so this
        // is exact parity, not a new design choice) represents a known dynamic-class object that may
        // temporarily leave the FOV or be occluded, and deserves more patience than generic unclassified
        // clutter before its track id/history is actually dropped. Still purely internal coasting either way —
        // see the association loop's own kf.predict() and publish()'s own "predicted-only this tick (missed)
        // never published" gate: a coasting track is never visible in RViz regardless of this value.
        max_missed_classified_ = declare_parameter<int>("track_max_missed_classified", max_missed_ * 3);

        // TWO separate speed bars, exactly like v1's own two separate params (dynaVelThresh_ vs
        // stationarySpeedThresh_ — these were NOT the same threshold in v1, collapsing them into one was an
        // earlier mistake here that let a lot of ordinary position jitter cross straight into "dynamic"):
        //   - track_dynamic_speed_thresh (v1: dynamic_velocity_threshold, default 0.35): the DYNAMIC label bar.
        //   - track_stationary_speed_thresh (v1: stationarySpeedThresh_, default 0.10): the much lower bar below
        //     which a track counts as "not moving at all" for suppression/aging-to-death. A track between the two
        //     (0.10-0.35 m/s here) is published as "static" (not suppressed, some real jitter/creep) but not yet
        //     labelled "dynamic" — v1's own middle ground, lost when this was a single threshold.
        dynamic_speed_thresh_ = declare_parameter<double>("track_dynamic_speed_thresh", 0.35);
        stationary_speed_thresh_ = declare_parameter<double>("track_stationary_speed_thresh", 0.10);
        // Debounce before the DYNAMIC label is granted: speed must stay >= track_dynamic_speed_thresh for this
        // many CONSECUTIVE ticks (reset to 0 the instant it drops below, UNLESS kf_vel_confident below also
        // holds — see the matched-branch comment). This is v1's PATH 3c anti-flicker consistency check; the
        // three params right after it port v1's remaining classification machinery (PATH 2 historical
        // continuity, PATH 3a KF-velocity confidence) — point-cloud motion voting (PATH 3b) is the one piece
        // NOT ported: it needs raw per-object points over time, which this node doesn't receive (only already-
        // fused boxes) — plumbing that through would be a real architecture change, not a classification tweak.
        dynamic_confirm_ticks_ = declare_parameter<int>("track_dynamic_confirm_ticks", 3);
        // Same debounce for a track with NO semantic evidence of a dynamic class that is inside a camera's FOV
        // (see publish()): unrecognized "motion" there must persist longer (8 ticks = 0.8 s at 10 Hz) before it
        // is believed. Set equal to track_dynamic_confirm_ticks to disable.
        dynamic_confirm_ticks_unclassified_ = declare_parameter<int>("track_dynamic_confirm_ticks_unclassified", 8);
        // ...and outside every camera FOV, where nothing can be classified and a long wait only delays real movers.
        dynamic_confirm_ticks_unclassified_out_fov_ = declare_parameter<int>("track_dynamic_confirm_ticks_unclassified_out_fov", 5);
        // MOTION-EARNED PD: an unclassified track that has been published dynamic for this many ticks is
        // potentially_dynamic when it stops, instead of static — the camera never saw it (e.g. a person walking
        // behind a robot with no rear camera), but it has shown it moves. Kept while the track lives; revoked,
        // like a class, after track_max_non_yolo_in_fov_frames ticks in the reliable FOV with no YOLO class.
        // 0 disables.
        motion_pd_min_dynamic_ticks_ = declare_parameter<int>("track_motion_pd_min_dynamic_ticks", 20);

        // Historical continuity / "sticky dynamic" (v1: forceDynaFrames_/forceDynaCheckRange_, PATH 2) — once a
        // track has been dynamic for a good fraction of its recent history, re-confirm it immediately on
        // continued motion evidence instead of re-running the full debounce from scratch after one noisy dip.
        // Protects a genuinely moving object from flickering back to "static" from a single bad tick.
        dynamic_sticky_frames_ = declare_parameter<int>("track_dynamic_sticky_frames", 20);  // v1: forceDynaFrames_
        dynamic_sticky_window_ = declare_parameter<int>("track_dynamic_sticky_window", 30);  // v1: forceDynaCheckRange_

        // KF-velocity confidence (v1: PATH 3a, dynaKfVelStdRatio_) — an ADDITIONAL way to count as "moving
        // evidence" each tick, alongside the plain instantaneous speed check: the track's own recent speed
        // samples have a MEAN above the dynamic bar AND a standard deviation small relative to that mean (the
        // KF has been CONSISTENTLY estimating real motion, not one noisy spike) — catches a genuinely moving
        // object whose single-tick speed estimate briefly dips from measurement noise.
        dynamic_kf_confidence_window_ = declare_parameter<int>("track_dynamic_kf_confidence_window", 5);
        dynamic_kf_std_ratio_ = declare_parameter<double>("track_dynamic_kf_std_ratio", 0.50);  // v1: dynaKfVelStdRatio_

        // Size-sanity gate on the DYNAMIC label (v1: constrainSize_/targetObjectSize_ — a general post-filter
        // against an implausibly-sized moving blob, e.g. a noisy wide LiDAR cluster or a swinging door's sweep,
        // crossing the speed bar and getting the label anyway). v1 defaults this OFF and supports a LIST of
        // target sizes (it has no semantic classification of its own to lean on instead); here it defaults ON
        // — this session's whole arc has been about false "dynamic" positives, and this is a cheap, well-
        // targeted guard — but is deliberately scoped to UNCLASSIFIED tracks only (best_class empty): a track
        // YOLO has already identified has stronger, specific identity evidence that a single generic person-
        // sized box can't improve on (and would actively hurt for anything else YOLO might ever classify).
        dynamic_constrain_size_ = declare_parameter<bool>("track_dynamic_constrain_size", true);
        dynamic_target_size_ = Eigen::Vector3d(declare_parameter<double>("track_dynamic_target_size_x", 0.5),
                                               declare_parameter<double>("track_dynamic_target_size_y", 0.5),
                                               declare_parameter<double>("track_dynamic_target_size_z", 1.7));
        dynamic_size_tolerance_ = Eigen::Vector3d(declare_parameter<double>("track_dynamic_size_tolerance_x", 0.8),
                                                  declare_parameter<double>("track_dynamic_size_tolerance_y", 0.8),
                                                  declare_parameter<double>("track_dynamic_size_tolerance_z", 1.0));

        // Point-cloud motion voting (v1 PATH 3b, ported per explicit request — see pointCloudVoteRatio()'s own
        // comment): per-point nearest-neighbor voting between this tick's and the LAST matched tick's own
        // points, corroborating (not replacing) the plain speed check for a track with no classification
        // evidence of its own — see the matched-branch comment on how this, kf_vel_confident, and the plain
        // speed check combine. Needs dbscan_detector_node's fused/clusters cloud (det_idx field).
        fused_clusters_topic_ = declare_parameter<std::string>("fused_clusters_topic", "/sins/dbscan_detector/fused/clusters");
        point_vote_thresh_ = declare_parameter<double>("track_point_vote_thresh", 0.7);         // v1: dynaVoteThresh_
        point_vote_max_points_ = declare_parameter<int>("track_point_vote_max_points", 500);    // cap — brute-force nearest-neighbor is O(n*m); v1 never capped this (its own clusters were smaller), this file's can be larger (LiDAR walls, etc.)
        point_vote_max_age_ = declare_parameter<double>("point_vote_max_age", 0.5);  // [s] ignore a stale clusters cloud (same convention as yolo2d_max_age)

        // Disocclusion-artifact detection — NOT from v1 (the user's own design, from a separate discussion, see
        // isDisocclusionArtifact()'s own comment): when a track moves away from a spot, whatever was BEHIND it
        // (occluded until then) becomes newly visible and can read as an apparent jump/velocity for something
        // that is actually static and was simply never visible before. lidar_frame is the sensor origin used
        // for the ray-alignment test (LiDAR: 360 degrees, covers the general case; a camera-only occlusion
        // release is not covered by this first pass).
        lidar_frame_ = declare_parameter<std::string>("lidar_frame", "velodyne");
        disocclusion_pos_hist_window_ = declare_parameter<int>("track_disocclusion_pos_hist_window", 10);
        disocclusion_min_cos_angle_ = declare_parameter<double>("track_disocclusion_min_cos_angle", 0.97);   // ~14 degrees
        disocclusion_min_range_margin_ = declare_parameter<double>("track_disocclusion_min_range_margin", 0.3);  // [m]

        // Dynamic-class whitelist (v1's own yoloDynamicClasses_, see dynamicDetector.cpp's own ingestion filter
        // — a detection whose class isn't in this list is never even considered a "YOLO candidate" there). v1
        // default reused as-is. v2's best_class/class_votes (SEMANTIC CLASS, see the file header) vote on and
        // display ANY class YOLO reports — richer than v1's plain boolean is_yolo_candidate — but everywhere
        // that class evidence is used as "potentially-dynamic" EXEMPTION (static-track-death below, the
        // potentially_dynamic label itself, the dynamic_constrain_size_ gate, and the classified-track fast
        // paths for confirmation/dynamic-label-debounce) now goes through isDynamicClass() first, so a
        // correctly-recognized but definitely-static object (e.g. "chair", "tv") behaves exactly as it would
        // if YOLO had never classified it at all — same as v1. The association loop's own ID-swap guard
        // (class_match_bonus_/class_continuity_weight_ below) deliberately stays class-agnostic: avoiding an
        // identity swap between two DIFFERENT recognized objects is useful regardless of whether either one's
        // class is in this whitelist, and v1 has no equivalent concept to match against there anyway (it never
        // carried a specific class string per track to begin with).
        {
            const auto v = declare_parameter<std::vector<std::string>>("track_semantic_dynamic_classes",
                {"person", "car", "bus", "truck", "motorbike", "bicycle", "dog", "cat", "horse", "cow", "sheep"});
            semantic_dynamic_classes_ = std::set<std::string>(v.begin(), v.end());
        }

        // Static-track death (v1's own "suppress stationary non-YOLO tracks" trigger, see the file header): a
        // CONFIRMED track with no semantic evidence of being a potentially-dynamic-WHITELISTED class (see
        // isDynamicClass() above) and below track_stationary_speed_thresh is not published — v1's exact rule,
        // "kept alive internally so it re-enters the output immediately if it starts moving again".
        // track_publish_stationary escapes the whole rule (publish static tracks too, v1's trackSteadyObjects_).
        // track_stationary_max_ticks is the one thing v1 does NOT do: after this many CONSECUTIVE suppressed
        // ticks the track is actually deleted (not just hidden) — this detector exists to find
        // dynamic/potentially-dynamic objects, so an object that settles for good should stop costing a track
        // slot and an id, not linger forever the way v1's own tracks do.
        track_publish_stationary_ = declare_parameter<bool>("track_publish_stationary", false);
        track_stationary_max_ticks_ = declare_parameter<int>("track_stationary_max_ticks", 50);

        // First-classification debounce (no v1 equivalent — see voteClass()'s own comment): how many CONSECUTIVE
        // ticks the SAME class must be voted before a track that has never been classified gets best_class set
        // for the first time at all. 1 = no debounce (old behavior, immediate). Guards against a single
        // contaminated tick (e.g. dbscan_detector_node's own nested-cluster merge briefly fusing a person with
        // nearby static clutter) painting a long-lived static track with a passing person's class.
        first_class_confirm_ticks_ = declare_parameter<int>("track_first_class_confirm_ticks", 2);
        // Established-track motion corroboration (no v1 equivalent — see voteClass()'s own comment): a track
        // with at least this many matched hits (and still unclassified) uses its own debounce window
        // (track_first_class_established_confirm_ticks) and additionally requires NET displacement across
        // that whole window to clear track_first_class_min_jump (reusing min_natural_motion_dist_'s own "is
        // this real motion" bar for consistency) before accepting the class — this displacement check, not a
        // longer tick count, is what actually distinguishes real directed walking from centroid jitter now
        // that the upstream root cause (a bad leaf occasionally bleeding into nearby static clutter — see
        // preprocessing_node's own yolo_leaf_valley_fraction/yolo_leaf_depth_tolerance) is fixed at the source.
        first_class_established_hits_ = declare_parameter<int>("track_first_class_established_hits", 10);
        first_class_established_confirm_ticks_ = declare_parameter<int>("track_first_class_established_confirm_ticks", 2);
        first_class_min_jump_ = declare_parameter<double>("track_first_class_min_jump", 0.08);
        // ...unless the class comes from depth-corroborated (depth-leaf-direct) evidence on this many CONSECUTIVE
        // votes (5 = 0.5 s at 10 Hz): then a standing person can be classified without moving. See voteClass().
        first_class_static_confirm_ticks_ = declare_parameter<int>("track_first_class_static_confirm_ticks", 5);
        // In-FOV semantic decay only counts a tick when the track's projected centre is at least this fraction
        // of the image size away from the border (see decaySemanticEvidence()).
        decay_fov_border_frac_ = declare_parameter<double>("track_decay_fov_border_frac", 0.10);
        // ...within that border band the same absence-of-evidence streak clears the class after this many ticks
        // instead (1.5 s: enough for a person to cross the band, not for clutter sitting in it).
        max_non_yolo_fov_border_frames_ = declare_parameter<int>("track_max_non_yolo_fov_border_frames", 15);
        // Class handover (see handOverClass()): a new track born within this distance [m] of a classified track's
        // prediction, from a detection that track passed over, inherits its class if within this window [s] the
        // classified track gives the class up or dies.
        class_handover_dist_ = declare_parameter<double>("track_class_handover_dist", 0.5);
        class_handover_window_ = declare_parameter<double>("track_class_handover_window", 3.0);
        // Height term of the association (see the candidate loop): penalty per metre of difference between the
        // candidate's box top and the track's smoothed top, hard reject beyond track_max_top_diff, active once the
        // track has track_height_min_samples matched ticks; the smoothing is an EMA with this alpha.
        w_height_ = declare_parameter<double>("track_match_height_weight", 0.5);
        max_top_diff_ = declare_parameter<double>("track_max_top_diff", 1.0);
        height_min_samples_ = declare_parameter<int>("track_height_min_samples", 5);
        height_ema_alpha_ = declare_parameter<double>("track_height_ema_alpha", 0.3);
        // Output box completion for "person" tracks (see completePersonBox()).
        person_box_completion_ = declare_parameter<bool>("track_person_box_completion", true);
        floor_frame_ = declare_parameter<std::string>("floor_frame", "base_footprint");
        person_min_height_ = declare_parameter<double>("track_person_min_height", 1.7);   // [m]
        person_ray_offset_ = declare_parameter<double>("track_person_ray_offset", 0.07);  // [m]

        // Class-conflict guard against ID swaps (v1's own match_yolo_class_consistency_weight /
        // yolo_track_assoc_bonus, brought back specifically for this — see the association loop): a candidate
        // pairing a detection classified as X against a track already carrying a DIFFERENT class is a HARD reject
        // (not just a penalty), and a pairing where they AGREE gets a bonus so it wins the global assignment over
        // an unclassified competitor at similar distance.
        class_match_bonus_ = declare_parameter<double>("track_class_match_bonus", 0.25);  // v1: yolo_track_assoc_bonus
        // Extra guard for the far more common case: this tick's TRUE match for a classified track is itself
        // unclassified (YOLO doesn't refine every tick). See the association loop's own comment. FOV-gated per
        // v1's own matchYoloClassConsistencyWeight_ (which only penalizes when the predicted box is actually
        // inside the camera FOV — outside it, YOLO cannot confirm anything anyway, so absence of evidence isn't
        // suspicious there and this term is simply not applied, see the association loop).
        class_continuity_weight_ = declare_parameter<double>("track_class_continuity_weight", 0.5);
        // Camera-only fragment guard (see the association loop): a candidate with NO LiDAR point (provenance mask
        // without "L") is dropped for a track that also has a LiDAR-backed candidate within this distance [m] of
        // it this tick — the two are one object split by the fusion, and the camera-only half is YOLO-mask depth
        // that landed on the background. 0 disables.
        camera_fragment_dist_ = declare_parameter<double>("track_camera_fragment_dist", 0.6);
        // Class ownership guard (see the association loop): a detection carrying a class is kept from any
        // unclassified track that scores it lower than the best track already classified with that class.
        class_ownership_ = declare_parameter<bool>("track_class_ownership", true);

        // Velocity-direction gate + soft penalty (v1: computeVelocityDirectionError/passesVelocityDirectionGate),
        // ported per explicit request — a candidate whose IMPLIED motion (from the track's last raw observation
        // to this candidate) points in a very different direction (or magnitude) than the KF's own PREDICTED
        // velocity is suspicious, independent of and complementary to the position/size/IoU/class terms above.
        // v1 has FIVE threshold tiers (Confirm/Tracked x normal/"Dynamic" + a separate outside-FOV-Confirm one);
        // collapsed here to THREE for a sane param surface: a strict tier for a not-yet-confirmed track, a
        // looser one once confirmed (v1's own "Tracked" is looser than "Confirm" — an established track earns
        // more benefit of the doubt), and a single further-relaxed tier applied whenever EITHER side carries
        // semantic evidence (v1's "Dynamic" variants: a YOLO-confirmed object, e.g. a person, can legitimately
        // change direction more abruptly than a generic physics prior expects) OR the predicted position is
        // outside every camera's FOV (v1's own outside-FOV allowance — LiDAR-only tracking there is noisier,
        // and a genuine sharp turn is harder to tell from that noise).
        match_veldir_weight_ = declare_parameter<double>("track_match_veldir_weight", 0.12);  // v1: matchVelocityDirectionScoreWeight_
        max_veldir_error_unconfirmed_ = declare_parameter<double>("track_max_veldir_error_unconfirmed", 1.20);  // v1: maxVelocityDirectionErrorConfirm_
        max_veldir_error_confirmed_ = declare_parameter<double>("track_max_veldir_error_confirmed", 2.50);      // v1: maxVelocityDirectionErrorTracked_
        max_veldir_error_relaxed_ = declare_parameter<double>("track_max_veldir_error_relaxed", 3.00);          // v1: the "Dynamic"/outside-FOV tiers, collapsed
        // [m] no hard velocity-direction rejection for a candidate this close to the KF prediction (see the
        // association loop): ~half a person's width, the centroid shift between a partial and a full box of
        // the same object.
        veldir_gate_min_innovation_ = declare_parameter<double>("track_veldir_gate_min_innovation", 0.30);

        // LiDAR-only-outside-FOV association relaxation (v1: isLidarOnlyOutsideFovAssociation + the
        // outside-FOV-specific gates/bonuses) — ported per explicit request. When BOTH the predicted and the
        // current position are outside every camera's FOV AND neither the track nor this tick's detection
        // carries any semantic evidence (a classified pairing already has its own, separate FOV-aware handling
        // above), this is pure LiDAR-only tracking: DBSCAN's cluster centroid is noisier there (no camera
        // corroboration), so v1 trades a WIDER position/size tolerance for a NARROWER speed tolerance (a large
        // IMPLIED speed is more likely a bad match than genuine fast motion in this noisier regime) plus a
        // stability bonus so a legitimate track doesn't lose its identity to ordinary centroid jitter. v1's own
        // numbers are relative to ITS OWN (much smaller, fixed-dt) base gates; the values below instead keep
        // v1's RATIOS (~2.4x wider distance, ~0.5x narrower speed, ~1.4x wider size) applied to this file's own
        // (real-dt, already-recalibrated-against-the-bag) base gates, documented per-field below.
        outside_fov_max_match_dist_ = declare_parameter<double>("track_outside_fov_max_match_dist", 3.5);   // v1 ratio: ~2.4x track_max_match_dist
        outside_fov_max_match_speed_ = declare_parameter<double>("track_outside_fov_max_match_speed", 4.0); // v1 ratio: ~0.5x track_max_match_speed (TIGHTER, not looser)
        outside_fov_max_size_diff_ = declare_parameter<double>("track_outside_fov_max_size_diff", 1.2);     // v1 ratio: ~1.4x track_max_size_diff
        outside_fov_confirmed_bonus_ = declare_parameter<double>("track_outside_fov_confirmed_bonus", 0.75); // v1: outsideFovConfirmedAssocBonus_ (same value — pure score-space bonus, scale-independent)
        outside_fov_tentative_bonus_ = declare_parameter<double>("track_outside_fov_tentative_bonus", 0.35); // v1: outsideFovTentativeAssocBonus_

        // Natural-motion gate (v1: isNaturalMotion, ported per explicit request) — guards a track's very FIRST
        // few matches, before it has any identity evidence of its own: not yet confirmed AND not semantically
        // classified. Three checks against a candidate: (1) inside a camera's FOV, the implied move must clear
        // a minimum (near-zero motion proves nothing about identity yet, unless track_publish_stationary says
        // keep it anyway) — NOT required outside FOV, where even real motion can be tiny between ticks;
        // (2) the move must never exceed an outright-implausible maximum, any FOV; (3) the KF's own innovation
        // (distance from the PURE prediction) must stay within an adaptive bound, tighter inside FOV than out.
        // v1's own values are physical per-tick distances at a comparable (~10Hz) tick rate and this file's KF
        // now matches v1's real process-noise numbers (see below), so these are v1's raw values, not a ratio.
        min_natural_motion_dist_ = declare_parameter<double>("track_min_natural_motion_dist", 0.08);  // v1: minNaturalMotionDist_
        max_natural_motion_dist_ = declare_parameter<double>("track_max_natural_motion_dist", 1.50);  // v1: maxNaturalMotionDist_
        max_natural_innovation_ = declare_parameter<double>("track_max_natural_innovation", 0.60);               // v1: maxNaturalInnovation_
        max_natural_innovation_outside_fov_ = declare_parameter<double>("track_max_natural_innovation_outside_fov", 1.00);  // v1: maxNaturalInnovationOutsideFov_

        // Generic confirmed-track stability bonus (v1: confirmedTrackAssocBonus_, ported per explicit request) —
        // independent of class/FOV, simply rewards being matched to an already-established track so a newer or
        // more tentative competitor doesn't win purely on a marginal score edge.
        confirmed_track_bonus_ = declare_parameter<double>("track_confirmed_bonus", 0.20);  // v1: confirmedTrackAssocBonus_

        // Duplicate-track suppression (v1: isCloseToExistingTrack/isDetectionDuplicateOfPrediction — found
        // during the broader v1 review, not originally in the earlier port list, but directly relevant: the
        // main association is a GLOBAL one-to-one assignment, so if TWO detections this tick are both
        // plausible for the SAME real object (e.g. a fragmented/split cluster), only one wins it — the other
        // reaches the "new track" loop unmatched. Without this check it would spawn a spurious SECOND track
        // for an object that already has one. v1's own shipped default for the size-similarity leg
        // (duplicateSizeRelThresh_=0) makes that leg permanently false (relSizeDiff can never be < 0), which
        // looks like an oversight rather than intent — track_duplicate_size_thresh below uses a working value
        // instead (this file's own min_size_diff_floor_-consistent scale) so the check actually functions.
        duplicate_track_dist_thresh_ = declare_parameter<double>("track_duplicate_dist_thresh", 0.5);   // v1: duplicateTrackDistThresh_
        duplicate_track_iou_thresh_ = declare_parameter<double>("track_duplicate_iou_thresh", 0.3);     // v1: duplicateTrackIou2DThresh_
        duplicate_track_size_thresh_ = declare_parameter<double>("track_duplicate_size_thresh", 0.5);   // v1 ships 0 here (effectively disabled) — see comment above

        // Motion-gated confirmation (v1: shouldConfirmTrack — found during the broader v1 review): v2's own
        // confirmation gate was bare hit-count (track_min_hits matches, any speed, even dead stationary from
        // birth). v1 is stricter for a NON-classified track: it must show ACTUAL motion (observed speed above
        // track_stationary_speed_thresh) AND pass the velocity-direction gate, on top of the hit count — a
        // track that has simply been reliably re-detected while never moving does not confirm. A classified
        // track (either side, this tick or ever) still confirms immediately on its first hit, same as before —
        // YOLO evidence is already strong identity, these extra checks don't add anything there. This is a
        // deliberate BEHAVIOR CHANGE from this file's own earlier bare-hit-count gate, ported because it
        // directly continues this session's "false dynamic/spurious track" hardening.
        min_confirm_obs_speed_ = declare_parameter<double>("track_min_confirm_obs_speed", -1.0);  // < 0 = reuse track_stationary_speed_thresh (v1: stationarySpeedThresh_)

        // v1's own real defaults (eP_v2_/eQPos_v2_/eQVel_v2_/eQAcc_v2_/eRPos_v2_ — found in dynamicDetector.cpp's
        // parameter loading, NOT in its sample yaml, which doesn't override them): this file's own earlier
        // defaults (0.5/1.0 process noise) were picked without checking those and left the filter far more
        // reactive to ordinary per-tick centroid jitter (OBB refit noise) than v1's — a second, compounding
        // source of the same false-"dynamic" problem the split thresholds above address.
        q_pos_ = declare_parameter<double>("track_kf_pos_process_var", 0.05);    // v1: eQPos_v2_
        q_vel_ = declare_parameter<double>("track_kf_vel_process_var", 0.15);    // v1: eQVel_v2_
        q_acc_ = declare_parameter<double>("track_kf_acc_process_var", 0.10);    // v1: eQAcc_v2_
        r_pos_ = declare_parameter<double>("track_kf_pos_meas_var", 0.02);       // v1: eRPos_v2_
        p_init_ = declare_parameter<double>("track_kf_pos_init_var", 0.25);      // v1: eP_v2_

        marker_lifetime_sec_ = declare_parameter<double>("marker_lifetime_sec", 0.0);

        // SEMANTIC DECAY (see the file header's own section) — v1's own isInCameraFOV (PIXEL frustum only, no
        // depth-range gate — see insideAnyCameraFov's own comment on why) + the two decay streak limits, carried
        // over closely. Camera list/topic/frame naming mirrors dbscan_detector_node's own camera_names +
        // {camera} template convention.
        camera_names_ = declare_parameter<std::vector<std::string>>("camera_names", {"front_camera", "back_camera"});
        camera_info_topic_template_ = declare_parameter<std::string>("camera_info_topic_template", "/{camera}/camera/color/camera_info");
        camera_frame_template_ = declare_parameter<std::string>("camera_frame_template", "{camera}_refined");
        max_non_yolo_in_fov_frames_ = declare_parameter<int>("track_max_non_yolo_in_fov_frames", 5);     // v1: maxNonYoloInFovFrames_
        max_yolo_base_mismatch_frames_ = declare_parameter<int>("track_max_yolo_base_mismatch_frames", 5); // v1: maxYoloBaseMismatchFrames_
        yolo_base_mismatch_thresh_ = declare_parameter<double>("track_yolo_base_mismatch_thresh", 0.50);  // v1: yoloBaseMismatchThresh_
        // No v1 equivalent — re-added after being removed and then found to be worth keeping after all, now
        // gated on the YOLO 2D track id instead of running unconditionally; see decaySemanticEvidence()'s own
        // comment for the full back-and-forth.
        yolo_base_shrink_thresh_ = declare_parameter<double>("track_yolo_base_shrink_thresh", 0.60);

        // Raw-2D-projection semantic fallback (see yoloClassAtPixel()'s own comment): a LiDAR-only box (no camera
        // depth points at all, e.g. a person beyond the depth camera's own range) can never get a depth leaf —
        // dbscan_detector_node's refinement has nothing to deproject. YOLO on the COLOR image has no such range
        // limit, so this projects the TRACK's own 3D position straight into the image and checks it against the
        // RAW 2D YOLO boxes directly, bypassing the depth-leaf step entirely for this one case.
        yolo2d_topic_template_ = declare_parameter<std::string>("yolo2d_topic_template", "/yolo_semantic_seg/{camera}/detections");
        yolo2d_max_age_ = declare_parameter<double>("yolo2d_max_age", 0.5);  // [s] ignore a camera's latest 2D detections once this stale
        yolo2d_min_iou_ = declare_parameter<double>("yolo2d_min_iou", 0.2);  // min IoU between the box's PROJECTED footprint and a YOLO box — see pixelCandidates()'s own comment

        cams_.reserve(camera_names_.size());  // each camera's subscription callback captures &cams_.back() below —
                                               // without reserving upfront, a later push_back can reallocate the
                                               // vector and dangle every pointer captured by an earlier camera's
                                               // callback (exactly what silently broke front_camera's FOV intrinsics)
        for (const auto& name : camera_names_) {
            CamFov cam;
            cam.name = name;
            cam.frame_id = camera_frame_template_;
            if (const auto p = cam.frame_id.find("{camera}"); p != std::string::npos) cam.frame_id.replace(p, 8, name);
            std::string topic = camera_info_topic_template_;
            if (const auto p = topic.find("{camera}"); p != std::string::npos) topic.replace(p, 8, name);
            cams_.push_back(std::move(cam));
            CamFov* cam_ptr = &cams_.back();
            cam_ptr->sub = create_subscription<sensor_msgs::msg::CameraInfo>(
                topic, rclcpp::SensorDataQoS(),
                [this, cam_ptr](const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg) {
                    if (cam_ptr->has_info) return;  // intrinsics don't change at runtime — latch the first one
                    cam_ptr->fx = msg->k[0]; cam_ptr->fy = msg->k[4]; cam_ptr->cx = msg->k[2]; cam_ptr->cy = msg->k[5];
                    cam_ptr->width = static_cast<int>(msg->width); cam_ptr->height = static_cast<int>(msg->height);
                    cam_ptr->has_info = true;
                    RCLCPP_INFO(get_logger(), "[%s] FOV intrinsics: fx=%.2f fy=%.2f cx=%.2f cy=%.2f %dx%d (frame=%s)",
                                cam_ptr->name.c_str(), cam_ptr->fx, cam_ptr->fy, cam_ptr->cx, cam_ptr->cy,
                                cam_ptr->width, cam_ptr->height, cam_ptr->frame_id.c_str());
                });
            std::string topic2d = yolo2d_topic_template_;
            if (const auto p = topic2d.find("{camera}"); p != std::string::npos) topic2d.replace(p, 8, name);
            cam_ptr->sub_2d = create_subscription<vision_msgs::msg::Detection2DArray>(
                topic2d, rclcpp::QoS(10),
                [cam_ptr](const vision_msgs::msg::Detection2DArray::ConstSharedPtr& msg) { cam_ptr->latest_2d = msg; });
        }

        pub_tracks_ = create_publisher<vision_msgs::msg::Detection3DArray>("/sins/tracker/tracks", rclcpp::QoS(10));
        pub_markers_ = create_publisher<visualization_msgs::msg::MarkerArray>("/sins/tracker/tracks_markers", rclcpp::QoS(10));
        // jo_msgs/ObstacleArray — see the file header's own comment: the pre-existing stable interface
        // glim_ros2's BBOX dynamic-rejection mode already subscribes to (its own bbox_topic config, default
        // matches this default exactly) to strip dynamic/potentially-dynamic points out of its own map.
        obstacles_topic_ = declare_parameter<std::string>("track_obstacles_topic", "/perception/tracked_dynamic_obstacles");
        // Diagnostics only: log every track birth (with nearby tracks and why each did not take the detection)
        // and every deletion (with the reason). Off by default.
        debug_lifecycle_ = declare_parameter<bool>("track_debug_lifecycle", false);
        // "hungarian" (default): true max-weight bipartite matching (hungarianMinCost) over the SAME
        // per-pair scores/gates above. On the mocap benchmark it matches greedy_kuhn's MOTA/IDF1 within
        // replay noise and has 2 fewer ID switches in every comparison (eval/STATUS.md, 2026-10-14).
        // "greedy_kuhn": v1-compatible max-CARDINALITY augmenting-path heuristic (tryAugment), kept for
        // the ablation. See this file's header and hungarianMinCost's own comment.
        assignment_method_ = declare_parameter<std::string>("track_assignment_method", "hungarian");
        pub_obstacles_ = create_publisher<jo_msgs::msg::ObstacleArray>(obstacles_topic_, rclcpp::QoS(10));
        // Point-cloud motion voting's own input (see the constructor's own comment above) — just the latest
        // message cached, matched against each tick's own detections by stamp (see parsePointsByDetection()).
        // dbscan_detector_node only actually publishes this cloud when it has a subscriber, so subscribing here
        // is what turns that publishing on.
        sub_clusters_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            fused_clusters_topic_, rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) { latest_clusters_ = msg; });
        sub_ = create_subscription<vision_msgs::msg::Detection3DArray>(
            "/sins/dbscan_detector/fused/detections3d", rclcpp::QoS(10),
            [this](const vision_msgs::msg::Detection3DArray::ConstSharedPtr& msg) { onFused(msg); });

        RCLCPP_INFO(get_logger(), "tracker_node ready: max_match_dist=%.2f max_match_speed=%.2f min_hits=%d max_missed=%d max_missed_classified=%d",
                    max_match_dist_, max_match_speed_, min_hits_, max_missed_, max_missed_classified_);
    }

private:
    static double stampSec(const builtin_interfaces::msg::Time& t) { return t.sec + t.nanosec * 1e-9; }

    // Horizontal-only speed: z-velocity is left OUT of the dynamic/static decision and the reported speed
    // (though it is still estimated in the Kalman state and published raw in vz) — the z estimate is much
    // noisier than x/y, driven by how much of an object's vertical extent each sensor's shadow cone happens to
    // see this tick, not by real vertical motion.
    static double planarSpeed(const Eigen::Vector3d& v) { return std::hypot(v.x(), v.y()); }

    // dbscan_detector_node's fused detections carry results[0] = provenance mask (always) and, only when a YOLO
    // leaf refined that object THIS tick, results[1] = {class_id: <name>, ...} (see that node's own header) —
    // empty when there is none this tick (most ticks, for most objects: YOLO only refines a fraction each tick).
    static std::string detectionClass(const vision_msgs::msg::Detection3D& d) {
        return d.results.size() < 2 ? std::string() : d.results[1].hypothesis.class_id;
    }

    // results[1].hypothesis.score, when detectionClass(d) is non-empty, is the refining YOLO leaf's own 2D
    // (ByteTrack) track id — see dbscan_detector_node.cpp's publish loop, which repurposed this field from an
    // unread constant 1.0. -1 means "no id" (never refined this tick, or the original YOLO id string failed to
    // parse as a number upstream in preprocessing_node — see that file's own box/mask branches).
    static int detectionYoloTrackId(const vision_msgs::msg::Detection3D& d) {
        return d.results.size() < 2 ? -1 : static_cast<int>(d.results[1].hypothesis.score);
    }

    // results[0] is ALWAYS the provenance mask (e.g. "L", "L+F", "L+B" — which sources contributed points, see
    // dbscan_detector_node's maskName()); "L" alone means LiDAR-only — no camera depth point at all went into
    // this box, which is exactly when yoloClassAtPixel()'s fallback applies (see its own comment).
    static std::string detectionMask(const vision_msgs::msg::Detection3D& d) {
        return d.results.empty() ? std::string() : d.results[0].hypothesis.class_id;
    }

    // Refreshes each configured camera's live pose (global_frame_ -> camera) ONCE per tick, not per track —
    // see CamFov's own comment. Failure (TF not yet available) just marks that camera unusable this tick; FOV
    // checks fall back to whichever camera(s) ARE available, matching v1's own graceful degradation.
    void refreshCameraTf() {
        for (auto& cam : cams_) {
            cam.tf_ok = false;
            if (!cam.has_info) continue;
            try {
                const auto tf = tf_buffer_.lookupTransform(cam.frame_id, global_frame_, tf2::TimePointZero);
                const auto& t = tf.transform;
                Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
                T.linear() = Eigen::Quaterniond(t.rotation.w, t.rotation.x, t.rotation.y, t.rotation.z).toRotationMatrix();
                T.translation() = Eigen::Vector3d(t.translation.x, t.translation.y, t.translation.z);
                cam.T_cam_global = T;
                cam.tf_ok = true;
            } catch (const tf2::TransformException&) {
                // not yet available (e.g. calibration_icp_node hasn't published "<camera>_refined" yet) — skip
            }
        }
    }

    // LiDAR origin in global_frame_ — the sensor origin isDisocclusionArtifact() rays from. Refreshed once per
    // tick, same pattern as refreshCameraTf().
    void refreshLidarTf() {
        lidar_tf_ok_ = false;
        try {
            const auto tf = tf_buffer_.lookupTransform(global_frame_, lidar_frame_, tf2::TimePointZero);
            lidar_origin_ = Eigen::Vector3d(tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z);
            lidar_tf_ok_ = true;
        } catch (const tf2::TransformException&) {
            // not yet available — skip this tick
        }
        floor_ok_ = false;
        try {
            floor_z_ = tf_buffer_.lookupTransform(global_frame_, floor_frame_, tf2::TimePointZero).transform.translation.z;
            floor_ok_ = true;
        } catch (const tf2::TransformException&) {
        }
    }

    // v1's own isInCameraFOV, OR'd across every camera with a usable pose this tick (v1 only ever had one) — PIXEL
    // frustum only, deliberately NOT gated by track_depth_min_/track_depth_max_ (the DEPTH CAMERA's own effective
    // range). That gate made sense when it did, back when the only way to reconfirm a class was the depth-leaf
    // mechanism, which genuinely cannot see past it — but now pixelCandidates()'s raw-2D fallback CAN reconfirm a
    // class all the way out to wherever YOLO can still see the object on the plain color image, which has no
    // such limit. Gating decay's own "can this still be reconfirmed" test by the depth camera's range was
    // exactly why a wrongly-tagged, beyond-depth-range track never decayed: this function said "outside FOV" for
    // it, so the inside-FOV absence-of-evidence trigger below never even looked at it.
    // border_frac > 0: the centre must also stay that fraction of the image width/height away from every border
    // — see decaySemanticEvidence(): near the border YOLO only sees part of the object and cannot be expected to
    // keep re-confirming it.
    bool insideAnyCameraFov(const Eigen::Vector3d& pos_global, double border_frac = 0.0) const {
        for (const auto& cam : cams_) {
            if (!cam.has_info || !cam.tf_ok) continue;
            const Eigen::Vector3d p = cam.T_cam_global * pos_global;  // p.z() = depth along the optical axis
            if (p.z() <= 1e-3) continue;  // behind the camera
            const double u = cam.fx * p.x() / p.z() + cam.cx;
            const double v = cam.fy * p.y() / p.z() + cam.cy;
            const double mu = border_frac * cam.width, mv = border_frac * cam.height;
            if (u >= mu && u < cam.width - mu && v >= mv && v < cam.height - mv) return true;
        }
        return false;
    }

    // v1's computeVelocityDirectionError, horizontal-only (vx,vy) — consistent with this file's own "vz
    // excluded" convention (see planarSpeed's own comment). Combines a magnitude-difference term with a
    // (1 - cosine-similarity) direction term between a candidate's OBSERVED velocity (implied by the track's
    // last RAW observation -> this candidate) and the KF's own PREDICTED velocity (already computed this tick,
    // before any update — same "fresh predict()-only" comparison v1 makes). Near-stationary on both sides is
    // never an error (0.0) — direction is undefined for a near-zero vector, and small jitter around standing
    // still shouldn't look like a direction mismatch; one side stationary and the other not still contributes
    // the magnitude-difference term, just no direction term.
    // The implied ("observed") velocity is the displacement from last_obs over the time ELAPSED SINCE last_obs,
    // not over this tick's dt: after k missed ticks the displacement spans k+1 ticks, and dividing it by one dt
    // inflated the implied speed k+1-fold, so a briefly-occluded track rejected its own object on reappearance.
    double velocityDirectionError(const Track& tr, const Eigen::Vector3d& obs_pos, double now, double dt) const {
        const double elapsed = tr.last_obs_stamp > 0.0 ? now - tr.last_obs_stamp : dt;
        const double idt = 1.0 / std::max(elapsed > 0.0 ? elapsed : dt, 1e-3);
        const double obs_vx = (obs_pos.x() - tr.last_obs.x) * idt, obs_vy = (obs_pos.y() - tr.last_obs.y) * idt;
        const double pred_vx = tr.kf.vel().x(), pred_vy = tr.kf.vel().y();
        const double obs_speed = std::hypot(obs_vx, obs_vy), pred_speed = std::hypot(pred_vx, pred_vy);
        if (obs_speed < stationary_speed_thresh_ && pred_speed < stationary_speed_thresh_) return 0.0;
        const double vel_diff = std::hypot(obs_vx - pred_vx, obs_vy - pred_vy);
        double dir_err = 0.0;
        if (obs_speed >= stationary_speed_thresh_ && pred_speed >= stationary_speed_thresh_) {
            const double dot = obs_vx * pred_vx + obs_vy * pred_vy;
            const double cos_sim = std::max(-1.0, std::min(1.0, dot / std::max(obs_speed * pred_speed, 1e-6)));
            dir_err = 1.0 - cos_sim;
        }
        return vel_diff + dir_err;
    }

    // v1's isNaturalMotion — see the constructor's own comment. Only meant to be called where v1 calls it: the
    // track not yet confirmed AND not semantically exempt (an already-confirmed or classified track has
    // stronger identity evidence and skips this gate entirely — see the association loop).
    bool isNaturalMotion(const Track& tr, const Eigen::Vector3d& obs_pos, const Eigen::Vector3d& pred_pos,
                         bool inside_fov_curr) const {
        const double obsDist = std::hypot(obs_pos.x() - tr.last_obs.x, obs_pos.y() - tr.last_obs.y);
        if (inside_fov_curr && !track_publish_stationary_ && obsDist < min_natural_motion_dist_) return false;
        if (obsDist > max_natural_motion_dist_) return false;
        const double innovation = std::hypot(obs_pos.x() - pred_pos.x(), obs_pos.y() - pred_pos.y());
        const double maxInnovation = inside_fov_curr ? max_natural_innovation_ : max_natural_innovation_outside_fov_;
        return innovation <= maxInnovation;
    }

    // v1's isCloseToExistingTrack/isDetectionDuplicateOfPrediction — see the constructor's own comment on why
    // this matters. Checked before spawning a brand-new track for an unmatched detection: is it actually a
    // near-duplicate of some OTHER, already-existing track's own prediction this same tick?
    bool isDuplicateOfExistingTrack(const vision_msgs::msg::Detection3D& d, const std::vector<int>& track_ids,
                                    const std::vector<Eigen::Vector3d>& predicted_pos) const {
        const auto& c = d.bbox.center.position;
        for (size_t ti = 0; ti < track_ids.size(); ++ti) {
            const Track& tr = tracks_.at(track_ids[ti]);
            const double centerDist = std::hypot(predicted_pos[ti].x() - c.x, predicted_pos[ti].y() - c.y);
            if (centerDist >= duplicate_track_dist_thresh_) continue;
            const Footprint predFp{predicted_pos[ti].x() - tr.size.x * 0.5, predicted_pos[ti].x() + tr.size.x * 0.5,
                                   predicted_pos[ti].y() - tr.size.y * 0.5, predicted_pos[ti].y() + tr.size.y * 0.5, 0, 0};
            const Footprint curFp = footprintOf(c, d.bbox.center.orientation, d.bbox.size);
            const double iou = iou2D(predFp, curFp);
            const double sizeDiff = relSizeDiff(tr.size, d.bbox.size, min_size_diff_floor_);
            if (iou > duplicate_track_iou_thresh_ && sizeDiff < duplicate_track_size_thresh_) return true;
        }
        return false;
    }

    // Parses the cached fused/clusters cloud's det_idx field (dbscan_detector_node's own addition, see that
    // file's own comment on it) into one point list per THIS tick's detection index — empty if the cloud is
    // missing, stale (older than point_vote_max_age_ relative to this tick's own stamp), or its own stamp
    // doesn't match this tick's detections (dbscan_detector_node publishes both from the SAME processing tick,
    // so a mismatch means the cloud lagged behind or a subscriber only just appeared).
    std::map<size_t, std::vector<Eigen::Vector3f>> parsePointsByDetection(double now_sec) const {
        std::map<size_t, std::vector<Eigen::Vector3f>> out;
        if (!latest_clusters_) return out;
        if (std::abs(now_sec - stampSec(latest_clusters_->header.stamp)) > point_vote_max_age_) return out;
        sensor_msgs::PointCloud2ConstIterator<float> ix(*latest_clusters_, "x"), iy(*latest_clusters_, "y"),
            iz(*latest_clusters_, "z"), iidx(*latest_clusters_, "det_idx");
        for (; ix != ix.end(); ++ix, ++iy, ++iz, ++iidx) {
            out[static_cast<size_t>(*iidx + 0.5f)].emplace_back(*ix, *iy, *iz);
        }
        return out;
    }

    // v1 PATH 3b, ported per explicit request: per-point nearest-neighbor motion voting between this track's
    // points NOW and its points from the LAST matched tick (brute-force nearest-neighbor, same as v1 — clusters
    // are small enough, tens to a few hundred points, that this is cheap; v1 does the same with no KD-tree,
    // see track_point_vote_max_points for the safety cap this file adds on top for larger LiDAR-only clusters).
    // For each current point, find its nearest neighbor in the previous cloud, derive a per-point velocity, and
    // check its alignment (cosine similarity) with the track's own KF velocity (v1 uses a box-centroid finite-
    // difference instead; the KF estimate is this file's own established "box velocity", used everywhere else
    // here too): a point moving OPPOSITE the track is excluded (background/noise bleeding into the cluster),
    // one moving WITH it and fast enough votes "dynamic". Returns the vote ratio, or a negative value when
    // there isn't enough data to have an opinion (no previous points, too many points to bother, or the track
    // isn't moving at all) — the caller treats a negative return as "abstain", not "not dynamic".
    double pointCloudVoteRatio(const std::vector<Eigen::Vector3f>& curr, const std::vector<Eigen::Vector3f>& prev,
                               const Eigen::Vector3d& box_vel, double dt) const {
        if (curr.empty() || prev.empty() || dt <= 1e-3) return -1.0;
        if (static_cast<int>(curr.size()) > point_vote_max_points_ || static_cast<int>(prev.size()) > point_vote_max_points_) return -1.0;
        const double boxSpeed = std::hypot(box_vel.x(), box_vel.y());
        if (boxSpeed < 1e-6) return -1.0;
        int numPoints = static_cast<int>(curr.size());
        int votes = 0;
        for (const auto& cp : curr) {
            double minDist = std::numeric_limits<double>::max();
            Eigen::Vector3f nearest = Eigen::Vector3f::Zero();
            for (const auto& pp : prev) {
                const double dist = (cp - pp).norm();
                if (dist < minDist) { minDist = dist; nearest = cp - pp; }
            }
            const double vx = nearest.x() / dt, vy = nearest.y() / dt;
            const double vNorm = std::hypot(vx, vy);
            if (vNorm < 1e-6) continue;
            const double velSim = (vx * box_vel.x() + vy * box_vel.y()) / (vNorm * boxSpeed);
            if (velSim < 0.0) { --numPoints; continue; }
            if (vNorm > dynamic_speed_thresh_) ++votes;
        }
        return numPoints > 0 ? static_cast<double>(votes) / static_cast<double>(numPoints) : 0.0;
    }

    // Disocclusion-artifact detection — NOT from v1 (the user's own design, from a separate discussion, applying
    // the same free/occupied/unknown distinction ERASOR/DUFOMap use for map cleaning, as a single targeted
    // geometric check hung on this file's existing gate architecture instead of a full occupancy grid): when a
    // track moves away from a spot, whatever was BEHIND it along the LiDAR's own line of sight (occluded until
    // then) becomes newly visible — a wall patch, furniture — and can get matched to a track, reading as an
    // apparent jump/velocity for something that is actually static and was simply never visible before.
    // Candidate position C is a probable disocclusion artifact if, for SOME other ACTIVELY MOVING track (see the
    // dynamic_ticks check below — a static track's history is never a valid reference) with a recent position P
    // in its own history window, C lies close to the same LiDAR line of sight through P (small angular
    // deviation) AND is noticeably FARTHER from the LiDAR than P was — exactly the signature of "the wall
    // behind where that other track recently was is now visible". Scoped to the LiDAR (360 degrees, covers the
    // general indoor/outdoor case); a camera-only occlusion release is not covered by this first pass.
    bool isDisocclusionArtifact(const Eigen::Vector3d& cand_pos, int exclude_track_id,
                                const std::vector<int>& track_ids) const {
        if (!lidar_tf_ok_) return false;
        const Eigen::Vector3d toCand = cand_pos - lidar_origin_;
        const double candRange = toCand.norm();
        if (candRange < 1e-3) return false;
        for (int other_id : track_ids) {
            if (other_id == exclude_track_id) continue;
            const auto it = tracks_.find(other_id);
            if (it == tracks_.end()) continue;
            // Only a track that has ITSELF actually been moving recently (dynamic_ticks > 0) can have revealed
            // anything by moving away — a static track's own historical positions are not a disocclusion
            // reference at all (nothing left there, so nothing was ever hidden behind it). Added after an
            // empirical failure: without this, a room with many tracks (including ordinary static clutter —
            // furniture, DBSCAN noise fragments) gave a genuinely walking, correctly-classified person's own
            // CURRENT position hundreds of chances per run to coincidentally fall within the angular/range
            // window of SOME static track's stored history, permanently wiping its dynamic_ticks streak and
            // keeping it stuck at "potentially_dynamic" (green) despite plainly moving — confirmed on a live
            // bag: 425 such resets in one 99s run, almost entirely against tracks that had never moved at all.
            if (it->second.dynamic_ticks <= 0) continue;
            for (const auto& P : it->second.pos_hist) {
                const Eigen::Vector3d toP = P - lidar_origin_;
                const double pRange = toP.norm();
                if (pRange < 1e-3) continue;
                const double cosAngle = toCand.dot(toP) / (candRange * pRange);
                if (cosAngle < disocclusion_min_cos_angle_) continue;
                if (candRange <= pRange + disocclusion_min_range_margin_) continue;
                return true;
            }
        }
        return false;
    }

    // Mirror of isDisocclusionArtifact() above, checked the other way round: a track that MISSED this tick (no
    // detection matched it) might simply be occluded RIGHT NOW by some OTHER, currently-matched track lying
    // closer to the LiDAR along roughly the same ray — e.g. a crowded scene where a dynamic object D walks
    // between the LiDAR and a dynamic/potentially-dynamic object D/PD it's about to pass in front of. Without
    // this, every miss counts equally toward missed_in_a_row/max_missed_classified_ regardless of WHY the
    // object wasn't seen — a plain sensor occlusion (explainable: something the tracker itself is already
    // tracking is physically in the way) is treated identically to an unexplained disappearance, so an
    // occlusion that happens to outlast the flat missed-ticks budget kills the occluded track and forces a
    // brand-new id on reappearance (a real-world crowded-scene failure, not a false-positive concern like
    // isDisocclusionArtifact's own — hence no dynamic_ticks>0-style gate on the OCCLUDER here: literally any
    // currently-matched detection standing in the way is a valid physical occluder, moving or not).
    bool isOcclusionExplained(const Eigen::Vector3d& predicted_pos, int exclude_track_id,
                              const std::vector<int>& track_ids, const std::vector<int>& track_to_det,
                              const vision_msgs::msg::Detection3DArray::ConstSharedPtr& msg) const {
        if (!lidar_tf_ok_) return false;
        const Eigen::Vector3d toPred = predicted_pos - lidar_origin_;
        const double predRange = toPred.norm();
        if (predRange < 1e-3) return false;
        for (size_t ti2 = 0; ti2 < track_ids.size(); ++ti2) {
            if (track_ids[ti2] == exclude_track_id) continue;
            const int di2 = track_to_det[ti2];
            if (di2 < 0) continue;  // only a track with a REAL detection this tick can physically occlude anything
            const auto& occ = msg->detections[static_cast<size_t>(di2)];
            const Eigen::Vector3d toOcc(occ.bbox.center.position.x - lidar_origin_.x(),
                                        occ.bbox.center.position.y - lidar_origin_.y(),
                                        occ.bbox.center.position.z - lidar_origin_.z());
            const double occRange = toOcc.norm();
            if (occRange < 1e-3) continue;
            const double cosAngle = toPred.dot(toOcc) / (predRange * occRange);
            if (cosAngle < disocclusion_min_cos_angle_) continue;
            if (occRange >= predRange - disocclusion_min_range_margin_) continue;  // occluder must be CLOSER than the predicted position
            return true;
        }
        return false;
    }

    // One candidate match of a 3D box against a camera's raw 2D YOLO boxes — see pixelCandidates()'s own comment
    // for why this returns EVERY match instead of picking one itself.
    struct PixelCandidate { int cam_idx; size_t det2d_idx; double iou; std::string cls; };

    // Semantic fallback for a box built from LiDAR points ONLY, with no camera depth contribution at all (e.g. a
    // person beyond the depth camera's own range — depth_max is typically 4.5-5m, the LiDAR sees much farther).
    // dbscan_detector_node's own YOLO refinement works by deprojecting a 2D detection into DEPTH points and
    // matching that leaf against a fused box — with no depth at that pixel, there is no leaf, so that mechanism
    // structurally cannot reach this object no matter how confidently YOLO sees it on the plain COLOR image.
    // This instead projects a 3D box straight into each camera's pixel space — no depth/range limit here, unlike
    // insideAnyCameraFov's old depth_min/depth_max gate, since a color-image detection has no such limit either —
    // and checks it against the RAW 2D YOLO boxes for that camera directly.
    //
    // A simple "does the projected CENTER point fall inside the YOLO box" test (this function's first version)
    // is too permissive: it accepts ANY box whose center happens to land inside, regardless of whether that box
    // is remotely the right SIZE/SHAPE for its distance — a wall segment, a piece of furniture, or anything else
    // near the same ray can satisfy it just as easily as the real object, which is exactly why unrelated/far-away
    // boxes kept getting tagged "person". Real depth is known here (it's a LiDAR-built box), so rather than
    // collapsing the box into one orientation-agnostic size number (this function's SECOND version — still wrong:
    // these are OBBs, not axis-aligned, so a box seen edge-on projects narrow and the same box seen face-on
    // projects wide; a single size estimate gets both cases wrong the same way footprintOf() would if it ignored
    // orientation), all 8 of the OBB's own corners (center + orientation + size, the same rotation convention
    // footprintOf() uses for the WORLD-frame footprint) are projected through the camera's pinhole model
    // individually, and their axis-aligned bounding rectangle IN PIXEL SPACE is what gets compared — this
    // correctly captures foreshortening/viewing-angle the way a flat size-to-pixels formula cannot. That
    // rectangle's 2D IoU against the YOLO box is required to clear yolo2d_min_iou_ — an object of the wrong real
    // size/shape for its distance and viewing angle, or merely a center-aligned coincidence, now fails this on
    // footprint SHAPE, not just position.
    //
    // Monocular projection is still depth-blind for objects of similar apparent footprint, though: TWO different
    // LiDAR-only boxes of comparable projected size at different depths along roughly the same ray can still
    // both pass the IoU bar. A YOLO box can only ever be ONE physical object, so this returns every passing
    // candidate — rather than picking one itself — and lets the CALLER resolve this globally across every
    // fallback-eligible detection this tick, awarding each (camera, 2D box) pair to its single HIGHEST-IoU
    // candidate only (see onFused()'s own resolution block).
    std::vector<PixelCandidate> pixelCandidates(const geometry_msgs::msg::Point& center,
                                                const geometry_msgs::msg::Quaternion& q,
                                                const vision_msgs::msg::BoundingBox3D::_size_type& size,
                                                double now_sec) const {
        std::vector<PixelCandidate> out;
        if (size.x <= 1e-3 || size.y <= 1e-3 || size.z <= 1e-3) return out;
        Eigen::Matrix3d R;
        R << 1 - 2 * (q.y * q.y + q.z * q.z), 2 * (q.x * q.y - q.z * q.w), 2 * (q.x * q.z + q.y * q.w),
            2 * (q.x * q.y + q.z * q.w), 1 - 2 * (q.x * q.x + q.z * q.z), 2 * (q.y * q.z - q.x * q.w),
            2 * (q.x * q.z - q.y * q.w), 2 * (q.y * q.z + q.x * q.w), 1 - 2 * (q.x * q.x + q.y * q.y);
        const Eigen::Vector3d c(center.x, center.y, center.z);
        Eigen::Vector3d corners[8];
        int ni = 0;
        for (int sx : {-1, 1})
            for (int sy : {-1, 1})
                for (int sz : {-1, 1})
                    corners[ni++] = c + R * Eigen::Vector3d(sx * 0.5 * size.x, sy * 0.5 * size.y, sz * 0.5 * size.z);

        for (size_t ci = 0; ci < cams_.size(); ++ci) {
            const auto& cam = cams_[ci];
            if (!cam.has_info || !cam.tf_ok || !cam.latest_2d) continue;
            if (std::abs(now_sec - stampSec(cam.latest_2d->header.stamp)) > yolo2d_max_age_) continue;  // stale

            double ax0 = 1e18, ax1 = -1e18, ay0 = 1e18, ay1 = -1e18;
            bool behind = false;
            for (const auto& corner : corners) {
                const Eigen::Vector3d p = cam.T_cam_global * corner;
                if (p.z() <= 1e-3) { behind = true; break; }  // box straddles/behind the camera plane — skip
                ax0 = std::min(ax0, cam.fx * p.x() / p.z() + cam.cx);
                ax1 = std::max(ax1, cam.fx * p.x() / p.z() + cam.cx);
                ay0 = std::min(ay0, cam.fy * p.y() / p.z() + cam.cy);
                ay1 = std::max(ay1, cam.fy * p.y() / p.z() + cam.cy);
            }
            if (behind || ax1 < 0.0 || ax0 >= cam.width || ay1 < 0.0 || ay0 >= cam.height) continue;
            for (size_t di2 = 0; di2 < cam.latest_2d->detections.size(); ++di2) {
                const auto& d2 = cam.latest_2d->detections[di2];
                if (d2.results.empty()) continue;
                const double bx0 = d2.bbox.center.position.x - d2.bbox.size_x * 0.5;
                const double bx1 = d2.bbox.center.position.x + d2.bbox.size_x * 0.5;
                const double by0 = d2.bbox.center.position.y - d2.bbox.size_y * 0.5;
                const double by1 = d2.bbox.center.position.y + d2.bbox.size_y * 0.5;
                const double ix = std::max(0.0, std::min(ax1, bx1) - std::max(ax0, bx0));
                const double iy = std::max(0.0, std::min(ay1, by1) - std::max(ay0, by0));
                const double inter = ix * iy;
                const double uni = std::max(0.0, ax1 - ax0) * std::max(0.0, ay1 - ay0) +
                                   std::max(0.0, bx1 - bx0) * std::max(0.0, by1 - by0) - inter;
                const double iou = uni > 1e-6 ? inter / uni : 0.0;
                if (iou >= yolo2d_min_iou_)
                    out.push_back({static_cast<int>(ci), di2, iou, d2.results[0].hypothesis.class_id});
            }
        }
        return out;
    }

    // SEMANTIC DECAY — see the file header's own section for the full rationale. Only called for a track that
    // was MATCHED this tick (a missed tick leaves semantic state untouched, like everything else in this file).
    // CLASS HANDOVER (no v1 equivalent). When a classified track passes over a detection right next to its own
    // prediction for a different one, and that detection then starts a new track, one of the two is the real
    // object. If within track_class_handover_window the classified track gives its class up (ID-swap evidence or
    // decay) or dies, the evidence was that it swapped identity: the class belongs to the newborn track, which
    // inherits it instead of the semantic information being lost. Observed on a lab bag: a person leaving the
    // camera FOV sideways lost their (fused) camera points, their centroid jumped, and the "person" track slid
    // onto static clutter at the image border while the person continued — unclassified — behind the robot.
    void handOverClass(int from_id, const std::string& cls) {
        if (!isDynamicClass(cls)) return;
        Track* heir = nullptr;
        for (auto& [id, t] : tracks_) {
            if (id == from_id || t.heir_of != from_id || !t.best_class.empty() || t.hits < min_hits_) continue;
            if (last_stamp_ - t.heir_stamp > class_handover_window_) continue;
            if (!heir || t.heir_stamp > heir->heir_stamp) heir = &t;
        }
        if (!heir) return;
        heir->best_class = cls;
        heir->class_votes[cls] = 1;
        heir->confirmed = true;  // a classified track confirms on its first classified hit anyway
        heir->heir_of = -1;
        RCLCPP_INFO(get_logger(), "track #%d: inherited class '%s' from track #%d (identity swap: #%d gave the class up)",
                    heir->id, cls.c_str(), from_id, from_id);
    }

    // A track that may move even when still: semantic evidence of a dynamic class, or motion-earned PD.
    bool potentiallyDynamic(const Track& tr) const { return isDynamicClass(tr.best_class) || tr.motion_pd; }

    void decaySemanticEvidence(int id, Track& tr, const std::string& det_class_this_tick,
                               const vision_msgs::msg::BoundingBox3D::_size_type& det_size, int det_track_id,
                               bool camera_covered) {
        if (tr.best_class.empty()) {
            tr.non_yolo_in_fov_streak = 0;
            tr.yolo_base_mismatch_streak = 0;
            tr.last_yolo_track_id = -1;
            // motion-earned PD: in the reliable FOV YOLO is trusted, so a "mover" it does not recognize loses it
            if (tr.motion_pd && det_class_this_tick.empty() &&
                insideAnyCameraFov(tr.kf.pos(), decay_fov_border_frac_)) {
                if (++tr.motion_pd_fov_streak >= max_non_yolo_in_fov_frames_) {
                    tr.motion_pd = false;
                    tr.motion_pd_fov_streak = 0;
                    if (debug_lifecycle_)
                        RCLCPP_INFO(get_logger(), "[lifecycle] track #%d: motion-earned PD revoked (in FOV, no YOLO)", id);
                }
            } else {
                tr.motion_pd_fov_streak = 0;
            }
            return;
        }
        const bool inside_fov = insideAnyCameraFov(tr.kf.pos());
        // Absence-of-evidence only counts away from the image border: a person crossing it is only partly
        // visible and routinely not detected for a few ticks, and with a short max_non_yolo_in_fov_frames_ that
        // used to strip the class right as they left the FOV. (A border margin on the CENTRE, not the object's
        // full extent: requiring the whole box inside exempted wrongly-classified static objects that merely sit
        // near the image edge from ever decaying.)
        const bool in_reliable_fov = insideAnyCameraFov(tr.kf.pos(), decay_fov_border_frac_);
        const std::string cleared_class = tr.best_class;  // for logging, in case this tick clears it

        // (1) inside-FOV: sustained absence of fresh 2D confirmation clears the class outright.
        if (debug_lifecycle_)
            RCLCPP_INFO(get_logger(), "[lifecycle] decay #%d cls='%s' pos=(%.2f,%.2f) in_fov=%d reliable=%d det_cls='%s' streak=%d",
                        id, tr.best_class.c_str(), tr.kf.pos().x(), tr.kf.pos().y(), inside_fov, in_reliable_fov,
                        det_class_this_tick.c_str(), tr.non_yolo_in_fov_streak);
        // Near the border the streak still runs, against a longer limit: a person crossing the border band is
        // through it in well under a second, while an object that SITS there unconfirmed (e.g. a track that slid
        // onto static clutter at the image edge as the person left) must not keep its class forever.
        const int decay_limit = in_reliable_fov ? max_non_yolo_in_fov_frames_ : max_non_yolo_fov_border_frames_;
        if (inside_fov && det_class_this_tick.empty()) {
            if (++tr.non_yolo_in_fov_streak >= decay_limit) {
                tr.best_class.clear();
                tr.class_votes.clear();
                tr.pending_class.clear();
                tr.pending_class_ticks = 0;
                tr.non_yolo_in_fov_streak = 0;
                tr.yolo_base_mismatch_streak = 0;
                tr.yolo_x_width = tr.yolo_y_width = 0.0;
                tr.last_yolo_track_id = -1;
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                    "track #%d: cleared class '%s' — in camera FOV with no fresh YOLO evidence for %d ticks, "
                    "pos=[%.2f,%.2f,%.2f] size=[%.2f,%.2f,%.2f]",
                    id, cleared_class.c_str(), decay_limit,
                    tr.last_obs.x, tr.last_obs.y, tr.last_obs.z, tr.size.x, tr.size.y, tr.size.z);
                handOverClass(id, cleared_class);
                return;  // class just cleared — nothing left to check below this tick
            }
        } else {
            tr.non_yolo_in_fov_streak = 0;
        }

        // high-water mark: largest size ever seen on a tick WITH real YOLO evidence, never shrunk — plus the
        // YOLO 2D (ByteTrack) track id active at that moment (prev_track_id below is the value from BEFORE this
        // tick's own update, i.e. the id the current baseline was actually built on).
        const int prev_track_id = tr.last_yolo_track_id;
        if (!det_class_this_tick.empty()) {
            tr.yolo_x_width = std::max(tr.yolo_x_width, det_size.x);
            tr.yolo_y_width = std::max(tr.yolo_y_width, det_size.y);
            tr.last_yolo_track_id = det_track_id;
        }

        // (2) GROWTH (v1 parity: dynamicDetector.cpp's own "YOLO decay (base size mismatch)" flags growth, never
        // shrinkage — v1's own comment there: "Shrinkage is ignored — it is a physical artifact, not an ID
        // swap") against the high-water mark — v1 only ever ran this outside FOV (trigger (1) above was assumed
        // sufficient inside it), but that assumption broke in practice: a track can keep MATCHING every tick via
        // LiDAR-only/depth-only association (no fresh 2D YOLO re-confirmation needed for that) while its own box
        // silently grows onto nearby static clutter, INSIDE camera FOV, for as long as trigger (1)'s own
        // max_non_yolo_in_fov_frames_ window allows before it fires — confirmed on a live bag: a person's track,
        // correctly classified "person" at spawn (size ~0.8x0.5x1.6), grew to ~1.6x0.5x1.9 (furniture-sized)
        // over the following ~9 ticks before trigger (1) finally cleared it. So this now runs regardless of
        // inside_fov.
        //
        // PLUS a SHRINK trigger, GATED on the YOLO 2D track id (track_yolo_base_shrink_thresh): a round trip on
        // this one. It originally ran unconditionally and was removed after a live-bag diagnostic (negative-X
        // "person" misclassification report) traced it clearing "person" off a track that was, in fact, still
        // the SAME real person the whole time — the box legitimately shrinks for a handful of ticks (LiDAR
        // shadow cone up close, a nearby fused object intermittently winning the Kuhn assignment, viewing-angle
        // changes...). But removing it outright brought back a DIFFERENT failure the unconditional version used
        // to catch: a track's identity genuinely drifting onto a smaller, different object. The fix is neither
        // "always trust shrink" nor "never trust it" — it's "trust it only when YOLO's OWN 2D tracker also says
        // this is a different box": det_track_id (this tick's own YOLO 2D track id, from
        // dbscan_detector_node's yolo_refine) vs. prev_track_id (the id the current high-water mark was built
        // on, captured above BEFORE this tick's update). Same real object, momentarily smaller for a sensor
        // reason -> det_track_id stays the SAME as prev_track_id (YOLO's own tracker never lost it) -> shrink is
        // ignored, exactly like the fix that removed the false positives. Genuinely a different, smaller object
        // -> either a brand-new det_track_id (YOLO itself re-acquired/switched identity) or no fresh YOLO
        // evidence at all this tick (det_track_id < 0, e.g. LiDAR-only) -> shrink is NOT ignored, restoring the
        // original protection. Only ever evaluated with BOTH ids known (>= 0) — a LiDAR-only tick has nothing to
        // compare and falls through to "not ignored" by the id_changed test below, same as before this fix.
        if (tr.yolo_x_width > 0.0 && tr.yolo_y_width > 0.0) {
            const double growth_x = (det_size.x - tr.yolo_x_width) / std::max(tr.yolo_x_width, 0.01);
            const double growth_y = (det_size.y - tr.yolo_y_width) / std::max(tr.yolo_y_width, 0.01);
            const bool grew = growth_x > yolo_base_mismatch_thresh_ || growth_y > yolo_base_mismatch_thresh_;
            const bool id_known_same = det_track_id >= 0 && prev_track_id >= 0 && det_track_id == prev_track_id;
            // Shrink only counts inside the FOV, where YOLO can tell a different object from the same one seen
            // smaller. Outside it nothing can adjudicate, and a close person's LiDAR box shrinks routinely
            // (shadow cone, seen from behind) — that cleared "person" off people standing behind the robot.
            // v1 ignores shrinkage altogether for the same reason ("a physical artifact, not an ID swap").
            // ...and only for a box that has camera points this tick: the baseline is built on camera-covered
            // (L+camera) boxes, and the LiDAR-only box of the same person is far narrower. On the benchmark bags
            // the depth stream drops frames for up to 0.6 s; each gap made the person's box "LiDAR-only", the
            // shrink fired and stripped "person" off a person standing in front of the camera.
            const bool shrank = inside_fov && camera_covered && !id_known_same &&
                                 (growth_x < -yolo_base_shrink_thresh_ || growth_y < -yolo_base_shrink_thresh_);
            if (grew || shrank) {
                if (++tr.yolo_base_mismatch_streak >= max_yolo_base_mismatch_frames_) {
                    tr.best_class.clear();
                    tr.class_votes.clear();
                    tr.pending_class.clear();
                    tr.pending_class_ticks = 0;
                    tr.yolo_base_mismatch_streak = 0;
                    tr.yolo_x_width = tr.yolo_y_width = 0.0;
                    tr.last_yolo_track_id = -1;
                    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                        "track #%d: cleared class '%s' — box %s %.0f%%/%.0f%% past its YOLO baseline for %d "
                        "ticks (%s FOV, yolo_id %d->%d)",
                        id, cleared_class.c_str(), grew ? "grew" : "shrank", growth_x * 100.0, growth_y * 100.0,
                        max_yolo_base_mismatch_frames_, inside_fov ? "inside" : "outside",
                        prev_track_id, det_track_id);
                    handOverClass(id, cleared_class);
                }
            } else {
                tr.yolo_base_mismatch_streak = 0;
            }
        } else {
            tr.yolo_base_mismatch_streak = 0;
        }
    }

    // A vote is cast whenever a detection carries a class, regardless of which camera produced it — see
    // Track::class_votes. Takes the class directly (not a Detection3D) since the caller may have substituted
    // yoloClassAtPixel()'s fallback for the primary depth-leaf class — see the call sites' own comments.
    //
    // FIRST-CLASSIFICATION DEBOUNCE (no v1 equivalent): a track that already HAS a class keeps the original
    // immediate argmax behavior below — by then its identity is well-established and SEMANTIC DECAY already
    // guards against staleness. A track that has NEVER been classified requires the SAME class on
    // track_first_class_confirm_ticks CONSECUTIVE ticks before best_class is set for the first time at all.
    // Added after an observed failure: dbscan_detector_node's own nested-cluster merge/absorb can, for exactly
    // one tick, fuse a person's cluster with a nearby STATIC object's (e.g. a person brushing past furniture) —
    // YOLO refinement correctly re-splits them back into two boxes most of the time, but when the leftover
    // (static-object-only) points fall below its own minimum-point floor that tick, no residual box is
    // produced at all, and the single merged "person"-refined box can end up matched to the FURNITURE's own
    // long-confirmed, previously-unclassified track if its predicted position happens to be the closer one —
    // one single-tick voteClass() call was enough to paint it "person" (green) outright, with nothing but the
    // (much slower, up-to-track_max_non_yolo_in_fov_frames-tick) SEMANTIC DECAY to eventually clear it again.
    // Requiring the SAME class to repeat is a real bar: a one-off contamination tick essentially never repeats
    // identically on the SAME track the very next tick (the merge/split is itself usually different then) — BUT
    // empirically (live-bag diagnostic logging, see the first-classification RCLCPP_WARN at the call site) this
    // alone was NOT enough: a long-lived, previously-unclassified track (hundreds of matched ticks — almost
    // certainly a static object, never a brand-new detection) was observed repeatedly winning "person" over
    // several consecutive ticks. A first attempt gated each INDIVIDUAL tick's raw jump against a minimum — this
    // was STILL not enough: it can't tell real directed walking apart from ordinary centroid JITTER, since a
    // static cluster's noisy centroid can easily clear a modest per-tick threshold on 2-3 ticks in a row without
    // the track ever actually going anywhere net (confirmed on the live bag: one track accumulated three
    // individually-qualifying ~0.1m ticks while its NET displacement across the whole window was ~0.28m over 5
    // SECONDS — essentially stationary; a genuinely walking person covers that in under a second). Fixed by
    // checking NET displacement over the whole debounce window instead of any single tick: pending_class_ticks
    // counts how long the SAME class has been pending, pending_class_start_pos is the raw position when that
    // streak began. For an ESTABLISHED track (hits >= first_class_established_hits_ — a brand-new track is
    // never old enough to trigger any of this, so a genuinely freshly-visible real object is unaffected): the
    // debounce window is first_class_established_confirm_ticks (its own tunable, independent of the plain
    // first_class_confirm_ticks_ above even though both currently default to the same value), and when that
    // many ticks are reached the NET displacement from pending_class_start_pos to right now must clear
    // first_class_min_jump — if it hasn't, the window simply restarts from here (not rejected outright: if this
    // IS a real, just-slower-than-usual person, sustained real motion will eventually clear the bar over a
    // later window; a genuinely static object sitting in a contamination hot-spot never will, by construction).
    // strong: the class came from dbscan_detector_node's own depth-leaf refinement (YOLO mask corroborated by
    // the depth points inside it), not from the monocular LiDAR-only pixel fallback.
    void voteClass(Track& tr, const std::string& cls, const geometry_msgs::msg::Point& cur_pos, bool strong) const {
        if (cls.empty()) return;
        if (tr.best_class.empty()) {
            if (cls != tr.pending_class) {
                tr.pending_class = cls;
                tr.pending_class_ticks = 0;
                tr.pending_strong_ticks = 0;
                tr.pending_class_start_pos = cur_pos;
            }
            ++tr.pending_class_ticks;
            tr.pending_strong_ticks = strong ? tr.pending_strong_ticks + 1 : 0;
            const bool established = tr.hits >= first_class_established_hits_;
            const int need_ticks = established ? first_class_established_confirm_ticks_ : first_class_confirm_ticks_;
            if (tr.pending_class_ticks < need_ticks) return;
            if (established) {
                const double net = std::hypot(cur_pos.x - tr.pending_class_start_pos.x, cur_pos.y - tr.pending_class_start_pos.y);
                // An established, never-classified track must have MOVED to take its first class (contamination
                // guard, see above) — except when the evidence is strong and keeps repeating: a person who was
                // already standing there before YOLO first saw them (robot start-up, or reappearing after an
                // occlusion long enough to lose the class) would otherwise never become potentially_dynamic
                // until they walked. A static object can still be painted by a one-off depth-leaf mistake, but
                // not for track_first_class_static_confirm_ticks consecutive ticks.
                const bool repeated_strong = tr.pending_strong_ticks >= first_class_static_confirm_ticks_;
                if (net < first_class_min_jump_ && !repeated_strong) {
                    if (!strong) {  // weak evidence: restart the displacement window, as before
                        tr.pending_class_ticks = 0;
                        tr.pending_class_start_pos = cur_pos;
                    }
                    return;
                }
            }
            tr.pending_class.clear();
            tr.pending_class_ticks = 0;
            tr.pending_strong_ticks = 0;
        }
        const int votes = ++tr.class_votes[cls];
        if (tr.best_class.empty() || votes > tr.class_votes[tr.best_class]) tr.best_class = cls;
    }

    // Whitelist membership check (see semantic_dynamic_classes_'s own constructor comment, v1 parity): ANY
    // class still gets voted/displayed via voteClass() above, but only a class in this list counts as
    // "potentially-dynamic" semantic EVIDENCE — i.e. is allowed to exempt a track from static-death, grant the
    // potentially_dynamic label, or fast-path confirmation/dynamic-label-debounce the way v1's is_yolo_candidate
    // (itself already whitelist-filtered at ingestion) does.
    bool isDynamicClass(const std::string& cls) const {
        return !cls.empty() && semantic_dynamic_classes_.count(cls) > 0;
    }

    void onFused(const vision_msgs::msg::Detection3DArray::ConstSharedPtr& msg) {
        const double stamp = stampSec(msg->header.stamp);
        double dt = last_stamp_ > 0.0 ? stamp - last_stamp_ : 0.1;
        if (dt <= 0.0 || dt > 1.0) dt = 0.1;  // clock jump (bag loop) or first message: fall back to the nominal LiDAR period
        last_stamp_ = stamp;
        refreshCameraTf();  // once per tick, not per track — see its own comment
        refreshLidarTf();
        const std::map<size_t, std::vector<Eigen::Vector3f>> points_by_det = parsePointsByDetection(stamp);

        // Predicted position per track (this tick's Kalman prediction) + each track's own previous raw
        // observation — the same two reference points v1's computeAssociationScore compares a detection against.
        std::vector<int> track_ids;
        std::vector<Eigen::Vector3d> predicted_pos;
        for (auto& [id, tr] : tracks_) {
            tr.kf.predict(dt, q_pos_, q_vel_, q_acc_);
            track_ids.push_back(id);
            predicted_pos.push_back(tr.kf.pos());
        }

        const size_t nt = track_ids.size(), nd = msg->detections.size();
        struct Cand { size_t det, track; double score; };
        // track_debug_lifecycle: why each (track, detection) pair was NOT a candidate this tick
        std::map<std::pair<size_t, size_t>, const char*> reject_reason;
        auto noteReject = [&](size_t ti, size_t di, const char* why) {
            if (debug_lifecycle_) reject_reason[{ti, di}] = why;
        };
        std::vector<Cand> candidates;
        for (size_t ti = 0; ti < nt; ++ti) {
            Track& tr = tracks_.at(track_ids[ti]);
            const Footprint predFp{predicted_pos[ti].x() - tr.size.x * 0.5, predicted_pos[ti].x() + tr.size.x * 0.5,
                                   predicted_pos[ti].y() - tr.size.y * 0.5, predicted_pos[ti].y() + tr.size.y * 0.5, 0, 0};
            const Footprint prevFp{tr.last_obs.x - tr.size.x * 0.5, tr.last_obs.x + tr.size.x * 0.5,
                                   tr.last_obs.y - tr.size.y * 0.5, tr.last_obs.y + tr.size.y * 0.5, 0, 0};
            // Hoisted out of the per-detection loop below (doesn't depend on di) — used both to FOV-gate the
            // class-mismatch continuity penalty and as half of the LiDAR-only-outside-FOV relaxation test.
            const bool inside_fov_pred = insideAnyCameraFov(predicted_pos[ti]);
            for (size_t di = 0; di < nd; ++di) {
                const auto& d = msg->detections[di];
                const auto& c = d.bbox.center.position;
                const std::string det_class = detectionClass(d);

                // LiDAR-only-outside-FOV relaxation (v1: isLidarOnlyOutsideFovAssociation) — see this
                // constructor's own comment on the three outside_fov_* params. BOTH the predicted and the
                // current position must be outside every camera's FOV, and NEITHER side may carry semantic
                // evidence (a classified pairing already gets its own, separate FOV-aware handling below) —
                // this is the "pure LiDAR, nothing to corroborate against" case the relaxation targets.
                const bool inside_fov_curr = insideAnyCameraFov(Eigen::Vector3d(c.x, c.y, c.z));
                const bool lidar_only_outside_fov = !inside_fov_pred && !inside_fov_curr &&
                                                    det_class.empty() && tr.best_class.empty();
                const double gate_dist = lidar_only_outside_fov ? outside_fov_max_match_dist_ : max_match_dist_;
                const double gate_speed = lidar_only_outside_fov ? outside_fov_max_match_speed_ : max_match_speed_;
                const double gate_size = lidar_only_outside_fov ? outside_fov_max_size_diff_ : max_size_diff_;

                const double predDist = std::hypot(c.x - predicted_pos[ti].x(), c.y - predicted_pos[ti].y());
                if (predDist >= gate_dist) { noteReject(ti, di, "dist"); continue; }
                const double reqSpeed = predDist / std::max(dt, 1e-3);
                if (reqSpeed >= gate_speed) { noteReject(ti, di, "speed"); continue; }
                const double sizeDiff = relSizeDiff(tr.size, d.bbox.size, min_size_diff_floor_);
                if (sizeDiff >= gate_size) { noteReject(ti, di, "size"); continue; }
                // HEIGHT: compare the candidate's TOP (box max z) with the track's smoothed top. Footprint size
                // alone can't tell a person from a table/cart of similar footprint, which is how a track slid onto
                // nearby clutter when its object turned or lost its camera points. The TOP, not the height: the
                // bottom is cut by floor removal and by the LiDAR cone up close (a person's box starts ~0.7 m
                // above the floor at 1 m), while the top still separates a person (>= ~1.2 m even up close) from
                // furniture. Soft score term, plus a hard reject only for a gross mismatch, and only once the
                // track has a settled top (track_height_min_samples matched ticks).
                const double det_top = c.z + 0.5 * d.bbox.size.z;
                const bool height_known = tr.top_samples >= height_min_samples_;
                const double topDiff = height_known ? std::abs(det_top - tr.top_z) : 0.0;
                if (topDiff > max_top_diff_) { noteReject(ti, di, "height"); continue; }
                // Classified-track baseline size gate (no v1 equivalent): the check above compares the
                // candidate against tr.size, the LATEST matched size — which can drift gradually tick-by-tick,
                // never individually exceeding gate_size, while compounding into a large cumulative change.
                // Confirmed on a live bag: a track classified "person" at a plausible size kept matching a
                // nearby STATIC object every following tick (no fresh 2D YOLO re-confirmation on any of them —
                // det_class.empty() below), each tick's own relative change small enough to pass this gate, yet
                // the box grew ~2x over 9 ticks before decaySemanticEvidence's own (now-faster) growth check
                // finally cleared it — by then the wrong object had already wrongly worn the "person" label for
                // ~1s. tr.yolo_x_width/tr.yolo_y_width (the track's own high-water mark — see
                // decaySemanticEvidence's own comment) is a STABLE reference that does not drift tick-to-tick,
                // so gating the CANDIDATE against that instead — only when nothing this tick freshly
                // re-confirms it is really still the classified object (det_class.empty()) — stops the drift at
                // the very first offending tick, not the ninth. The track simply coasts (missed) rather than
                // matching a since-the-person-left stationary stand-in; max_missed_classified_'s own extra
                // patience still gives the real object plenty of room to be legitimately reacquired later.
                // GROWTH only (v1 parity — see decaySemanticEvidence()'s own comment on why the symmetric
                // shrink check that used to live here and there was removed): rejecting a candidate for being
                // too SMALL blocked re-matching the SAME real person during ordinary occlusion/shadow-cone
                // shrinkage, which is exactly the "fast, unexplained decay" failure this file was chasing.
                if (isDynamicClass(tr.best_class) && det_class.empty() && tr.yolo_x_width > 0.0 && tr.yolo_y_width > 0.0) {
                    const double bx = (d.bbox.size.x - tr.yolo_x_width) / std::max(tr.yolo_x_width, 0.01);
                    const double by = (d.bbox.size.y - tr.yolo_y_width) / std::max(tr.yolo_y_width, 0.01);
                    if (bx > yolo_base_mismatch_thresh_ || by > yolo_base_mismatch_thresh_) {
                        noteReject(ti, di, "yolo_base_growth");
                        continue;
                    }
                }

                const Footprint curFp = footprintOf(c, d.bbox.center.orientation, d.bbox.size);
                const double predIou = iou2D(predFp, curFp);
                const double prevDist = std::hypot(c.x - tr.last_obs.x, c.y - tr.last_obs.y);
                const double prevIou = iou2D(prevFp, curFp);

                // Velocity-direction gate + soft penalty (v1: computeVelocityDirectionError /
                // passesVelocityDirectionGate — see this constructor's own comment on the three tiers).
                const double velErr = velocityDirectionError(tr, Eigen::Vector3d(c.x, c.y, c.z), stamp, dt);
                const bool yolo_exempt = !det_class.empty() || !tr.best_class.empty();
                double veldir_gate = tr.confirmed ? max_veldir_error_confirmed_ : max_veldir_error_unconfirmed_;
                if (!inside_fov_pred || yolo_exempt) veldir_gate = std::max(veldir_gate, max_veldir_error_relaxed_);
                // Hard gate only for a candidate that is NOT already where the KF predicts the object to be: within
                // track_veldir_gate_min_innovation of the prediction the implied velocity is dominated by centroid
                // jitter (the same standing person alternates between a partial LiDAR-only box and the full fused
                // box, whose centres differ by ~0.2-0.3 m = 2-3 m/s over one LiDAR tick), not by real motion, and
                // rejecting it there spawned a duplicate track next to the real one. The soft velErr penalty in
                // the score below still applies.
                if (velErr > veldir_gate && predDist > veldir_gate_min_innovation_) {
                    noteReject(ti, di, "veldir");
                    continue;
                }

                // Natural-motion gate (v1: isNaturalMotion — see its own comment) — only for a track with no
                // identity evidence of its own yet: not confirmed, not classified either side.
                if (!tr.confirmed && !yolo_exempt &&
                    !isNaturalMotion(tr, Eigen::Vector3d(c.x, c.y, c.z), predicted_pos[ti], inside_fov_curr)) {
                    noteReject(ti, di, "natural_motion");
                    continue;
                }

                // ID-swap guard (v1's own match_yolo_class_consistency_weight/yolo_track_assoc_bonus), two parts:
                //  1) When BOTH sides carry semantic evidence this tick and it DISAGREES, this is almost certainly
                //     two different physical objects near each other — hard reject, not a soft penalty. Agreement
                //     gets a bonus so the correct pairing wins the global assignment over a rival at similar
                //     distance.
                //  2) The far more common case: exactly ONE side of the pairing carries semantic evidence this
                //     tick. Two different directions, same underlying risk:
                //       a) this tick's TRUE match for an ALREADY-classified track is itself unclassified (YOLO
                //          only refines a fraction of ticks) — a nearby generic object's Kalman PREDICTION can
                //          drift toward the classified track under noise/acceleration even when its own last
                //          RAW observation did not.
                //       b) the mirror, easy to miss: a freshly classified DETECTION (person just became visible
                //          to YOLO) gets absorbed into a nearby, long-lived UNCLASSIFIED track (e.g. a chair's
                //          track) instead of spawning/rejoining the person's own track, simply because that
                //          track's prediction happened to be closest — this is how a generic object "inherits"
                //          a semantic label that was never really its own.
                //     Case (1) is neutral on both — neither pairing has a conflicting class to reject. So here an
                //     extra continuity weight is applied against the TRACK's own last_obs whenever exactly one
                //     side has class evidence, on top of the (smaller) w_prev_pos_/w_prev_iou_ every pair already
                //     gets. Soft/graded, not a hard cutoff, so genuinely fast motion still degrades gracefully —
                //     it just forces a cross-class pairing to be considerably closer to where THIS track really
                //     was than to the raw prediction to win. FOV-gated (v1's own matchYoloClassConsistencyWeight_
                //     behavior): only applied when the TRACK's predicted position is inside a camera's FOV —
                //     outside it, YOLO cannot confirm anything anyway, so absence of fresh evidence there isn't
                //     suspicious and this penalty simply does not apply.
                double class_score = 0.0;
                if (!det_class.empty() && !tr.best_class.empty()) {
                    if (det_class != tr.best_class) { noteReject(ti, di, "class"); continue; }  // hard reject: conflicting semantic evidence
                    class_score = class_match_bonus_;
                } else if (det_class.empty() != tr.best_class.empty() && inside_fov_pred) {
                    class_score = -class_continuity_weight_ * (prevDist / gate_dist);
                }

                double score = -w_pos_ * (predDist / gate_dist) - w_size_ * sizeDiff + w_iou_ * predIou -
                               w_prev_pos_ * (prevDist / gate_dist) + w_prev_iou_ * prevIou + class_score -
                               match_veldir_weight_ * velErr - w_height_ * topDiff;
                // Generic stability bonus for any already-confirmed track (v1: confirmedTrackAssocBonus_) —
                // independent of class/FOV, on top of (not instead of) the outside-FOV-specific bonus below.
                if (tr.confirmed) score += confirmed_track_bonus_;
                // Stability bonus for the same LiDAR-only-outside-FOV case the relaxed gates above target —
                // v1's own outsideFovConfirmedAssocBonus_/outsideFovTentativeAssocBonus_: without it, a
                // confirmed track's ordinary centroid jitter out there can lose to a nearby false competitor
                // at a similar (now much wider) gate.
                if (lidar_only_outside_fov) score += tr.confirmed ? outside_fov_confirmed_bonus_ : outside_fov_tentative_bonus_;
                if (debug_lifecycle_ && isDynamicClass(tr.best_class))
                    RCLCPP_INFO(get_logger(), "[lifecycle] cand #%d<-det(%.2f,%.2f) size %.2fx%.2f cls='%s' score=%.3f | "
                                "pred=%.2f prev=%.2f size=%.2f iou=%.2f prevIou=%.2f class=%.3f velErr=%.2f topDiff=%.2f conf=%d",
                                tr.id, c.x, c.y, d.bbox.size.x, d.bbox.size.y, det_class.c_str(), score, predDist, prevDist,
                                sizeDiff, predIou, prevIou, class_score, velErr, topDiff, tr.confirmed);
                if (score < min_match_score_) { noteReject(ti, di, "score"); continue; }
                candidates.push_back({di, ti, score});
            }
        }
        // Camera-only fragment guard (track_camera_fragment_dist). Seen on a person leaving the camera FOV
        // sideways: the fusion split her into a LiDAR-only box on her and a camera-only box 0.4 m away on the
        // background, which carried this tick's YOLO class. The class bonus (and the class-continuity penalty on
        // the unclassified LiDAR box) made the track take the fragment, slide onto the furniture behind it and
        // lose the person, who was reborn as a new track. A person in LiDAR range is always hit by the LiDAR, so
        // when the same track also has a LiDAR-backed candidate right next to the camera-only one, the
        // camera-only one is not the object.
        if (camera_fragment_dist_ > 0.0) {
            auto hasLidar = [&](size_t di) { const std::string m = detectionMask(msg->detections[di]); return !m.empty() && m[0] == 'L'; };
            std::vector<char> drop(candidates.size(), 0);
            for (size_t a = 0; a < candidates.size(); ++a) {
                if (hasLidar(candidates[a].det)) continue;
                const auto& ca = msg->detections[candidates[a].det].bbox.center.position;
                for (const auto& b : candidates) {
                    if (b.track != candidates[a].track || !hasLidar(b.det)) continue;
                    const auto& cb = msg->detections[b.det].bbox.center.position;
                    if (std::hypot(ca.x - cb.x, ca.y - cb.y) < camera_fragment_dist_) { drop[a] = 1; break; }
                }
                if (drop[a]) noteReject(candidates[a].track, candidates[a].det, "camera_fragment");
            }
            size_t k = 0;
            for (size_t a = 0; a < candidates.size(); ++a) if (!drop[a]) candidates[k++] = candidates[a];
            candidates.resize(k);
        }
        // Class ownership guard (track_class_ownership). The assignment maximises the number of pairs first,
        // so a classified track can be pushed onto a poor leftover when an unclassified rival has only one
        // candidate: the detection carrying the class. Seen with a person walking past a static object: the
        // object's track, dragged towards her by merged person+object boxes, took her "person" box (score
        // -0.47) while her own track, which scored that box +0.35, got a fragment (-0.72) and died. A detection
        // carrying class X is therefore kept from every unclassified track that scores it lower than the best
        // X-classified track does. A new person's own unclassified track keeps her box: it scores it higher.
        if (class_ownership_) {
            std::map<size_t, double> best_owner;  // det -> best score among tracks of the det's class
            for (const auto& c : candidates) {
                const std::string dc = detectionClass(msg->detections[c.det]);
                if (dc.empty() || tracks_.at(track_ids[c.track]).best_class != dc) continue;
                auto it = best_owner.find(c.det);
                if (it == best_owner.end() || c.score > it->second) best_owner[c.det] = c.score;
            }
            size_t k = 0;
            for (const auto& c : candidates) {
                const auto it = best_owner.find(c.det);
                if (it != best_owner.end() && tracks_.at(track_ids[c.track]).best_class.empty() && c.score < it->second) {
                    noteReject(c.track, c.det, "class_owner");
                    continue;
                }
                candidates[k++] = c;
            }
            candidates.resize(k);
        }
        std::sort(candidates.begin(), candidates.end(), [](const Cand& a, const Cand& b) { return a.score > b.score; });

        // Global assignment (v1's own assignMatchesGlobally/tryAugmentDetectionMatch) — see
        // track_assignment_method's own comment in the constructor for the two methods below.
        std::vector<int> track_to_det(nt, -1), det_to_track(nd, -1);
        if (assignment_method_ == "hungarian") {
            // Default: TRUE max-weight bipartite matching over the same candidates (hungarianMinCost,
            // see its own comment). Build an (n x n) cost matrix, n = max(nt, nd): a real candidate pair
            // gets -score (Hungarian minimizes); every OTHER real-track/real-detection pair gets FORBIDDEN
            // (it never passed the gates above, so it must never be used); padding rows/columns (the
            // rectangular slack when nt != nd) get cost 0, so the solver can always leave a real
            // track/detection unmatched for free instead of ever being forced onto a FORBIDDEN pair.
            const size_t n = std::max(nt, nd);
            const double FORBIDDEN = 1e6;
            std::vector<std::vector<double>> cost(n, std::vector<double>(n, 0.0));
            for (size_t ti = 0; ti < nt; ++ti)
                for (size_t di = 0; di < nd; ++di) cost[ti][di] = FORBIDDEN;
            for (const auto& c : candidates) cost[c.track][c.det] = -c.score;
            const std::vector<int> row_to_col = hungarianMinCost(cost);
            for (size_t ti = 0; ti < nt; ++ti) {
                const int di = row_to_col[ti];
                if (di < 0 || static_cast<size_t>(di) >= nd) continue;  // unmatched, or matched to padding
                if (cost[ti][static_cast<size_t>(di)] >= FORBIDDEN) continue;  // forced onto a non-candidate pair
                track_to_det[ti] = di;
                det_to_track[static_cast<size_t>(di)] = static_cast<int>(ti);
            }
        } else {
            // Ablation (greedy_kuhn): Kuhn's augmenting-path matching over the best-first candidate list.
            std::vector<std::vector<size_t>> candidates_by_det(nd);
            for (const auto& c : candidates) candidates_by_det[c.det].push_back(c.track);
            std::vector<int> visited(nt, -1);
            // Detections are given their turn to claim/steal a track in order of their OWN best candidate's
            // score, strongest first — not raw array index order (no v1 equivalent to compare against, a
            // general weighted-bipartite-matching concern independent of it). tryAugment's "steal" semantics
            // make this an augmenting-path heuristic for maximum-CARDINALITY matching, not a true
            // maximum-weight one (see track_assignment_method's "hungarian" option above for that) — the
            // actual assignment it converges to depends on which detection gets to claim first. With raw
            // index order, a detection with only a mediocre, borderline-score match could claim a track
            // before a DIFFERENT detection with a far stronger, more confident match for that SAME track
            // ever gets a turn — the strong match then has to settle for stealing (if tryAugment's recursion
            // finds a free alternative for the mediocre one) or its own second-best. Processing
            // strongest-match-first means the most confident pairing anywhere is locked in before any
            // weaker, more ambiguous one gets to contest it.
            std::vector<double> det_best_score(nd, -std::numeric_limits<double>::infinity());
            for (const auto& c : candidates) det_best_score[c.det] = std::max(det_best_score[c.det], c.score);
            std::vector<size_t> det_order(nd);
            std::iota(det_order.begin(), det_order.end(), 0);
            std::stable_sort(det_order.begin(), det_order.end(),
                [&](size_t a, size_t b) { return det_best_score[a] > det_best_score[b]; });
            int token = 0;
            for (size_t di : det_order) {
                ++token;
                tryAugment(di, candidates_by_det, track_to_det, det_to_track, visited, token);
            }
        }

        // LiDAR-only semantic fallback — GLOBAL resolution (see pixelCandidates()'s own comment on why: a YOLO
        // box can only ever be ONE physical object, so every fallback-eligible detection THIS tick — whether it
        // will end up matched to an existing track or start a new one — competes for the same pool of raw 2D
        // YOLO boxes, and each box is awarded to its single HIGHEST-IoU candidate only. Keyed by detection index
        // into msg->detections (not by track), since matched/new-track status isn't decided yet here.
        std::map<size_t, std::string> fallback_class;
        {
            struct Claim { size_t di; double iou; std::string cls; };
            std::map<std::pair<int, size_t>, Claim> best_per_box;  // (cam_idx, 2D det idx) -> current best claim

            // A 2D YOLO box already correctly claimed THIS TICK by a DEPTH-classified detection (dbscan_detector_node's
            // own box/mask refinement, detectionClass(d) non-empty) is not up for grabs here — a YOLO box can only
            // ever be ONE physical object, and the depth-based match is the stronger one (real depth corroboration,
            // vs. this fallback's pure monocular projection). Without this exclusion, a STATIC object reachable
            // only via this fallback (e.g. just beyond the depth camera's own coverage/FOV, but still within raw
            // 2D YOLO's wider monocular view) could steal that SAME already-correctly-claimed box for itself,
            // voting its class onto its own unrelated track — confirmed on a live bag: a cabinet just outside the
            // depth camera's coverage kept getting tagged "person" via this path even though the real person
            // right in front of it, within the camera's depth coverage, was ALREADY correctly matched through the
            // normal depth-leaf route. Found by the SAME projection this fallback itself uses, so a depth-classified
            // detection's own box is excluded by whichever 2D YOLO box IT projects into best, not by position alone.
            std::set<std::pair<int, size_t>> claimed_by_depth;
            for (size_t di = 0; di < nd; ++di) {
                const auto& d = msg->detections[di];
                if (detectionClass(d).empty()) continue;
                double best_iou = -1.0;
                std::pair<int, size_t> best_key{-1, 0};
                for (const auto& c : pixelCandidates(d.bbox.center.position, d.bbox.center.orientation, d.bbox.size, stamp))
                    if (c.iou > best_iou) { best_iou = c.iou; best_key = {c.cam_idx, c.det2d_idx}; }
                if (best_iou >= 0.0) claimed_by_depth.insert(best_key);
            }
            // Best IoU each 2D box reaches against any camera-covered (NOT fallback-eligible) detection this tick,
            // classified or not. A YOLO box is one physical object: if a camera-covered detection's projection
            // explains it better than a LiDAR-only one does, the box is that object's, not the LiDAR-only
            // candidate's. Without this, a person whose depth refinement happened not to classify them this tick
            // left their own box "unclaimed", and a nearby low LiDAR-only object (out of depth coverage) whose
            // projection merely overlapped the person's large close-range box took the "person" class.
            std::map<std::pair<int, size_t>, double> covered_best_iou;
            for (size_t di = 0; di < nd; ++di) {
                const auto& d = msg->detections[di];
                if (detectionClass(d).empty() && detectionMask(d) == "L") continue;  // fallback-eligible itself
                for (const auto& c : pixelCandidates(d.bbox.center.position, d.bbox.center.orientation, d.bbox.size, stamp)) {
                    auto& b = covered_best_iou[{c.cam_idx, c.det2d_idx}];
                    b = std::max(b, c.iou);
                }
            }

            for (size_t di = 0; di < nd; ++di) {
                const auto& d = msg->detections[di];
                if (!detectionClass(d).empty() || detectionMask(d) != "L") continue;  // not fallback-eligible
                // Same reliable image region the semantic decay uses (track_decay_fov_border_frac): at the border
                // YOLO sees only part of things, so a monocular match there can neither confirm nor (see
                // decaySemanticEvidence) be un-confirmed — a class given there would never decay.
                if (!insideAnyCameraFov(Eigen::Vector3d(d.bbox.center.position.x, d.bbox.center.position.y,
                                                        d.bbox.center.position.z), decay_fov_border_frac_))
                    continue;
                for (const auto& c : pixelCandidates(d.bbox.center.position, d.bbox.center.orientation, d.bbox.size, stamp)) {
                    const auto key = std::make_pair(c.cam_idx, c.det2d_idx);
                    if (claimed_by_depth.count(key)) {
                        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                            "LiDAR-only fallback: blocked a steal attempt on %s det2d %zu (iou=%.2f, class='%s') — "
                            "that box is already claimed by a depth-classified detection this tick",
                            camera_names_[static_cast<size_t>(c.cam_idx)].c_str(), c.det2d_idx, c.iou, c.cls.c_str());
                        continue;
                    }
                    if (const auto cv = covered_best_iou.find(key); cv != covered_best_iou.end() && cv->second >= c.iou)
                        continue;  // a camera-covered detection explains this box at least as well
                    const auto it = best_per_box.find(key);
                    if (it == best_per_box.end() || c.iou > it->second.iou)
                        best_per_box[key] = {di, c.iou, c.cls};
                }
            }
            for (const auto& [key, claim] : best_per_box) {
                fallback_class[claim.di] = claim.cls;
                if (debug_lifecycle_) {
                    const auto& c = msg->detections[claim.di].bbox.center.position;
                    const auto cv = covered_best_iou.find(key);
                    RCLCPP_INFO(get_logger(), "[lifecycle] fallback '%s' -> det at (%.2f,%.2f,%.2f) size %.2fx%.2fx%.2f iou=%.2f "
                                "covered_best=%.2f", claim.cls.c_str(), c.x, c.y, c.z,
                                msg->detections[claim.di].bbox.size.x, msg->detections[claim.di].bbox.size.y,
                                msg->detections[claim.di].bbox.size.z, claim.iou,
                                cv != covered_best_iou.end() ? cv->second : -1.0);
                }
            }
        }

        std::vector<char> det_used(nd, 0);
        std::set<int> matched_ids;  // tracks matched THIS tick — v1's own output gate: a track propagated only by
                                    // prediction (no match this tick) is never published, matched or not
        for (size_t ti = 0; ti < nt; ++ti) {
            Track& tr = tracks_.at(track_ids[ti]);
            const int di = track_to_det[ti];
            if (di < 0) {
                // Occlusion-aware miss: a track worth protecting (classified, or with ANY dynamic tick in its
                // own recent history — see Track::dynamic_hist) that is plausibly occluded right now by some
                // other, currently-matched track doesn't burn through its missed-ticks budget for a miss that
                // has a known physical explanation — see isOcclusionExplained()'s own comment. Scoped to
                // "worth protecting" tracks only: a random static/noise track sitting near another track's
                // ray should still age out normally, not get indefinite patience.
                const bool worth_protecting = isDynamicClass(tr.best_class) ||
                    std::any_of(tr.dynamic_hist.begin(), tr.dynamic_hist.end(), [](bool b) { return b; });
                const bool occluded = worth_protecting &&
                    isOcclusionExplained(predicted_pos[ti], tr.id, track_ids, track_to_det, msg);
                if (occluded) {
                    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                        "track #%d ('%s'): miss not counted against its budget — occluded by a closer, "
                        "currently-matched track along the same LiDAR ray", tr.id, tr.best_class.c_str());
                } else {
                    ++tr.missed_in_a_row;
                }
                continue;
            }
            det_used[static_cast<size_t>(di)] = 1;
            tr.missed_in_a_row = 0;
            ++tr.hits;
            const auto& d = msg->detections[static_cast<size_t>(di)];
            // Motion-gated confirmation (v1: shouldConfirmTrack — see the constructor's own comment on this
            // being a deliberate behavior change). A classified track (this tick's detection, the track's own
            // persistent best_class, or the LiDAR-only pixel fallback) confirms immediately once it clears the
            // hit count, same as before. A non-classified track additionally needs ACTUAL motion: observed
            // speed against the track's own PREVIOUS raw observation (still unmodified here, overwritten
            // below) above the stationary bar — a track that's simply been reliably re-detected while dead
            // still does not confirm. (v1 also re-checks the velocity-direction gate here; redundant in this
            // file's architecture, since that gate already ran as a hard candidate-generation reject above —
            // a matched detection has necessarily already passed it.) Sticky: once confirmed, stays confirmed
            // until dropped.
            if (!tr.confirmed && tr.hits >= min_hits_) {
                const auto fb_it = fallback_class.find(static_cast<size_t>(di));
                const bool classified_now = isDynamicClass(tr.best_class) || isDynamicClass(detectionClass(d)) ||
                                            (fb_it != fallback_class.end() && isDynamicClass(fb_it->second));
                if (classified_now) {
                    tr.confirmed = true;
                } else {
                    const double obsSpeed = std::hypot(d.bbox.center.position.x - tr.last_obs.x,
                                                       d.bbox.center.position.y - tr.last_obs.y) / std::max(dt, 1e-3);
                    const double minSpeed = min_confirm_obs_speed_ >= 0.0 ? min_confirm_obs_speed_ : stationary_speed_thresh_;
                    if (track_publish_stationary_ || obsSpeed >= minSpeed) tr.confirmed = true;
                }
            }
            tr.kf.update(Eigen::Vector3d(d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z), r_pos_);
            tr.size = d.bbox.size;
            {
                const double top = d.bbox.center.position.z + 0.5 * d.bbox.size.z;
                tr.top_z = tr.top_samples == 0 ? top : (1.0 - height_ema_alpha_) * tr.top_z + height_ema_alpha_ * top;
                ++tr.top_samples;
            }
            tr.orientation = d.bbox.center.orientation;
            // Captured BEFORE the overwrite below, for the first-classification diagnostic right after voteClass().
            const geometry_msgs::msg::Point prev_obs = tr.last_obs;
            const bool was_unclassified = tr.best_class.empty();
            tr.last_obs = d.bbox.center.position;
            tr.last_obs_stamp = stamp;
            // LiDAR-only semantic fallback — already resolved GLOBALLY above (one YOLO box -> one detection, not
            // one per track), just look this detection's own result up.
            const bool had_depth_class = !detectionClass(d).empty();
            std::string eff_class = detectionClass(d);
            if (eff_class.empty()) {
                if (const auto it = fallback_class.find(static_cast<size_t>(di)); it != fallback_class.end()) {
                    eff_class = it->second;
                    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                        "track #%d: LiDAR-only fallback matched '%s' via raw 2D YOLO projection (no depth coverage)",
                        track_ids[ti], eff_class.c_str());
                }
            }
            // Did THIS class come from dbscan_detector_node's own direct depth-leaf yolo_refine (had_depth_class,
            // i.e. detectionClass(d) was already non-empty on arrival) or from tracker_node's own LiDAR-only
            // pixel-projection fallback (!had_depth_class but eff_class non-empty)? See the first-classification
            // log's own use of this — the two paths fail differently: the depth-leaf path can pick the WRONG
            // already-separate fused object outright (its own best-IoU-candidate logic, no merge needed at all),
            // while the fallback is monocular/depth-blind (pixelCandidates()'s own comment) and can award the
            // same 2D YOLO box to a competing LiDAR-only detection when the real object isn't itself going
            // through the fallback (e.g. it has its own depth classification already, in which case the
            // already-claimed-box exclusion applies — or, if the real object is ALSO fallback-only, the two
            // compete purely on IoU with no depth to disambiguate them, and the wrong one can still win).
            const char* class_source = had_depth_class ? "depth-leaf-direct" : (!eff_class.empty() ? "lidar-fallback" : "none");
            // Only the depth-leaf-direct path carries a real YOLO 2D track id through from
            // dbscan_detector_node's yolo_refine (see detectionYoloTrackId()'s own comment) — the LiDAR-only
            // fallback path (pixelCandidates()) never built one, so it's treated as "no id" here, same as a
            // plain LiDAR-only match with no YOLO evidence at all.
            const int det_track_id = had_depth_class ? detectionYoloTrackId(d) : -1;
            // Raw-observation displacement since last tick — only used for the first-classification diagnostic
            // below now (voteClass() itself tracks NET displacement across its own debounce window internally,
            // from the raw position passed in here).
            const double jump = std::hypot(tr.last_obs.x - prev_obs.x, tr.last_obs.y - prev_obs.y);
            voteClass(tr, eff_class, tr.last_obs, had_depth_class);
            // First-classification diagnostic (see voteClass()'s own comment on the debounce this feeds into):
            // logs the exact moment a track accepts its very first class, with enough context to tell an ID
            // SWAP (the track's own position just jumped onto a different physical object — "jump" large,
            // relative to a track that had been barely moving) apart from plain CLASS CONTAMINATION bleeding in
            // from upstream (same position as always — "jump" small — a long-sitting static object just got
            // mislabeled by dbscan_detector_node's own fusion/refinement, the track's own identity/position was
            // never actually hijacked). INFO + throttled, not WARN: a track getting classified is routine
            // operation, not a warning condition — this is a debugging aid (kept at INFO rather than DEBUG so
            // it's still visible without rebuilding with debug logging enabled), throttled the same way every
            // other frequent diagnostic in this file is, so continuous operation with heavy track churn can't
            // flood the log.
            if (was_unclassified && !tr.best_class.empty()) {
                RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                    "track #%d: first classification '%s' via %s (mask=%s) (hits=%d, stationary_ticks=%d, "
                    "confirmed=%d) — position jump from previous observation = %.2fm, absolute pos=[%.2f,%.2f,%.2f], "
                    "size=[%.2f,%.2f,%.2f] (large+established => likely ID swap; small+established => likely "
                    "class contamination from a nearby merge; an anomalously LARGE size vs. the track's own "
                    "usual one is a merged/not-yet-split fused box, not the track's real object; compare "
                    "absolute pos ACROSS repeated events on the SAME track id to tell real walking from "
                    "jitter-around-a-fixed-spot)",
                    track_ids[ti], tr.best_class.c_str(), class_source, detectionMask(d).c_str(), tr.hits,
                    tr.stationary_ticks, tr.confirmed ? 1 : 0, jump,
                    tr.last_obs.x, tr.last_obs.y, tr.last_obs.z, tr.size.x, tr.size.y, tr.size.z);
            }
            decaySemanticEvidence(track_ids[ti], tr, eff_class, d.bbox.size, det_track_id, detectionMask(d) != "L");
            matched_ids.insert(track_ids[ti]);

            // Static-track aging (see this class's own constructor comment): counts consecutive ticks this track
            // was below the (lower) STATIONARY bar AND had no semantic evidence of a potentially-dynamic class;
            // reset the moment either stops being true, so only SUSTAINED stillness accumulates toward death, not
            // one slow tick after a burst. Uses stationary_speed_thresh_ (0.10), not the dynamic label's own,
            // higher bar (0.35) — a track creeping at 0.2 m/s is neither "dynamic" nor "truly stopped".
            const double speed = planarSpeed(tr.kf.vel());
            const bool exempt = potentiallyDynamic(tr) || track_publish_stationary_;
            tr.stationary_ticks = (!exempt && speed < stationary_speed_thresh_) ? tr.stationary_ticks + 1 : 0;

            // KF-velocity confidence (v1 PATH 3a, see the constructor's own comment): rolling mean/std of this
            // track's own recent horizontal speed samples — mean above the dynamic bar AND stable relative to
            // that mean counts as corroborating evidence of motion (see the PATH1/PATH3 split below).
            tr.speed_hist.push_back(speed);
            while (static_cast<int>(tr.speed_hist.size()) > dynamic_kf_confidence_window_) tr.speed_hist.pop_front();
            double speed_mean = 0.0;
            for (double v : tr.speed_hist) speed_mean += v;
            speed_mean /= static_cast<double>(std::max<size_t>(tr.speed_hist.size(), 1));
            double speed_var = 0.0;
            for (double v : tr.speed_hist) speed_var += (v - speed_mean) * (v - speed_mean);
            speed_var /= static_cast<double>(std::max<size_t>(tr.speed_hist.size(), 1));
            const bool kf_vel_confident = speed_mean >= dynamic_speed_thresh_ &&
                                          std::sqrt(speed_var) <= dynamic_kf_std_ratio_ * speed_mean;

            // Point-cloud motion voting (v1 PATH 3b, see pointCloudVoteRatio()'s own comment) — corroborating
            // evidence from this track's own points, independent of the KF. prev_points is updated for next
            // tick's use regardless of whether voting happened this tick (empty if the clusters cloud was
            // unavailable/stale, which simply makes next tick's vote abstain too).
            const auto pit = points_by_det.find(static_cast<size_t>(di));
            const double voteRatio = pit != points_by_det.end()
                ? pointCloudVoteRatio(pit->second, tr.prev_points, tr.kf.vel(), dt) : -1.0;
            const bool point_vote_evidence = voteRatio >= point_vote_thresh_;  // a negative (abstain) ratio never clears this
            tr.prev_points = pit != points_by_det.end() ? pit->second : std::vector<Eigen::Vector3f>{};

            // Disocclusion-artifact check (not from v1, see isDisocclusionArtifact()'s own comment) — if this
            // tick's position reads as "the wall behind where some OTHER track recently was, now visible",
            // don't let it count as motion evidence even if the raw numbers above say otherwise. Logged below,
            // once it's known whether this actually blocks anything (see blocked_by_disocclusion's own comment).
            const bool disocclusion = isDisocclusionArtifact(
                Eigen::Vector3d(d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z),
                tr.id, track_ids);

            // Dynamic-label debounce (v1 PATHs 1/3, see the constructor's own comment): a CLASSIFIED track
            // (this tick or ever) needs only the plain speed check, like v1's PATH 1 (YOLO-certified tracks
            // classify on KF speed alone). An UNCLASSIFIED track needs the same speed check PLUS corroboration
            // from EITHER KF-confidence or point-cloud voting, like v1's PATH 3 (speed alone isn't enough
            // without a non-YOLO track's own history to lean on) — the sticky-dynamic shortcut below is PATH
            // 2's equivalent bypass for an already-established track.
            const bool speed_ok = speed >= dynamic_speed_thresh_;
            const bool classified_now = isDynamicClass(tr.best_class) || isDynamicClass(eff_class);
            const bool motion_evidence = classified_now ? speed_ok : (speed_ok && (kf_vel_confident || point_vote_evidence));
            // Disocclusion only gates the UNCLASSIFIED (PATH3) case — see this check's own first attempt at
            // scoping (only an actively-moving OTHER track counts at all) and the comment above on why PATH1/
            // PATH3 exist as separate bars in the first place: a CLASSIFIED track already has strong, direct
            // YOLO identity evidence that this geometric heuristic cannot improve on and should not override —
            // the whole point of the disocclusion check is to second-guess AMBIGUOUS motion on a featureless,
            // unclassified LiDAR blob, not a track YOLO has already confirmed as "person". Confirmed empirically
            // necessary even after the first scoping fix: a genuinely walking, correctly "person"-classified
            // track could still get its dynamic_ticks streak wiped, tick after tick, by coincidental angular/
            // range alignment against some OTHER simultaneously-dynamic track's own recent trail in a room with
            // several moving/noisy tracks at once — permanently stuck at "potentially_dynamic" (green) despite
            // plainly moving (205 resets observed on classified tracks in one 99s run even after the first fix).
            // Crowded-scene refinement: classified_now alone misses a track that WAS dynamic/potentially-
            // dynamic before this tick but lost its class during a long occlusion (decaySemanticEvidence's own
            // FOV-based trigger — no fresh YOLO confirmation for track_max_non_yolo_in_fov_frames ticks while
            // occluded) and hasn't been re-classified yet at the exact tick it reappears. Exactly the
            // D-passes-in-front-of-D/PD case: the reappearing D/PD object is textbook-indistinguishable from a
            // revealed wall by the geometric test alone (same ray, farther range, occluder was moving) unless
            // this track's OWN recent history says it was genuinely moving before — tr.dynamic_hist (pushed
            // AFTER this decision, so it still reads as "before this tick" here) is exactly that record.
            const bool had_dynamic_history = std::any_of(tr.dynamic_hist.begin(), tr.dynamic_hist.end(),
                                                          [](bool b) { return b; });
            const bool blocked_by_disocclusion = disocclusion && !classified_now && !had_dynamic_history;
            if (blocked_by_disocclusion)
                RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                    "track #%d: suppressed as a probable disocclusion artifact (background revealed behind another track)",
                    tr.id);
            tr.dynamic_ticks = (motion_evidence && !blocked_by_disocclusion) ? tr.dynamic_ticks + 1 : 0;

            // Position history for isDisocclusionArtifact()'s own use by OTHER tracks next tick.
            tr.pos_hist.push_back(Eigen::Vector3d(d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z));
            while (static_cast<int>(tr.pos_hist.size()) > disocclusion_pos_hist_window_) tr.pos_hist.pop_front();

            // Historical continuity / "sticky dynamic" (v1 PATH 2, see the constructor's own comment) — tracked
            // AFTER this tick's debounce decision, so next tick's sticky check already sees today's result.
            tr.dynamic_hist.push_back(tr.dynamic_ticks >= dynamic_confirm_ticks_);
            while (static_cast<int>(tr.dynamic_hist.size()) > dynamic_sticky_window_) tr.dynamic_hist.pop_front();
        }

        for (size_t di = 0; di < nd; ++di) {
            if (det_used[di]) continue;
            const auto& d = msg->detections[di];
            // Duplicate-track suppression (v1: isCloseToExistingTrack — see the constructor's own comment):
            // this "unmatched" detection may really be a second, fragmented candidate for an object that
            // already has a track (only one such candidate can win the global one-to-one assignment above) —
            // don't spawn a redundant track for it.
            if (isDuplicateOfExistingTrack(d, track_ids, predicted_pos)) continue;
            Track tr;
            tr.id = next_id_++;
            {   // potential heir of a nearby classified track that passed this detection over (see handOverClass())
                const auto& c = d.bbox.center.position;
                double best = class_handover_dist_;
                for (size_t ti = 0; ti < nt; ++ti) {
                    const Track& o = tracks_.at(track_ids[ti]);
                    if (!isDynamicClass(o.best_class) || track_to_det[ti] < 0) continue;
                    const double dist = std::hypot(predicted_pos[ti].x() - c.x, predicted_pos[ti].y() - c.y);
                    if (dist < best) { best = dist; tr.heir_of = o.id; tr.heir_class = o.best_class; tr.heir_stamp = stamp; }
                }
            }
            if (debug_lifecycle_) {
                const auto& c = d.bbox.center.position;
                std::string near;
                for (size_t ti = 0; ti < nt; ++ti) {
                    const double dist = std::hypot(predicted_pos[ti].x() - c.x, predicted_pos[ti].y() - c.y);
                    if (dist > 1.5) continue;
                    const Track& o = tracks_.at(track_ids[ti]);
                    const auto r = reject_reason.find({ti, di});
                    const bool matched_other = track_to_det[ti] >= 0;
                    near += " | #" + std::to_string(o.id) + " d=" + std::to_string(dist).substr(0, 4) +
                            " cls='" + o.best_class + "' size=" + std::to_string(o.size.x).substr(0, 4) + "x" +
                            std::to_string(o.size.y).substr(0, 4) + " base=" + std::to_string(o.yolo_x_width).substr(0, 4) +
                            "x" + std::to_string(o.yolo_y_width).substr(0, 4) + " why=" +
                            (r != reject_reason.end() ? r->second : (matched_other ? "matched_other" : "candidate_unassigned"));
                }
                RCLCPP_INFO(get_logger(), "[lifecycle] spawn #%d det_class='%s' size=%.2fx%.2f at (%.2f,%.2f)%s", tr.id,
                            detectionClass(d).c_str(), d.bbox.size.x, d.bbox.size.y, c.x, c.y, near.c_str());
            }
            tr.kf.init(Eigen::Vector3d(d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z), p_init_);
            tr.size = d.bbox.size;
            {
                const double top = d.bbox.center.position.z + 0.5 * d.bbox.size.z;
                tr.top_z = tr.top_samples == 0 ? top : (1.0 - height_ema_alpha_) * tr.top_z + height_ema_alpha_ * top;
                ++tr.top_samples;
            }
            tr.orientation = d.bbox.center.orientation;
            tr.last_obs = d.bbox.center.position;
            tr.last_obs_stamp = stamp;
            tr.hits = 1;
            // Same LiDAR-only fallback as the matched branch above (already resolved GLOBALLY) — a brand-new
            // track beyond the depth camera's range should be able to pick up its YOLO class immediately too,
            // not just on a later matched tick.
            std::string eff_class = detectionClass(d);
            if (eff_class.empty()) {
                if (const auto it = fallback_class.find(di); it != fallback_class.end()) eff_class = it->second;
            }
            // hits == 1 is always well below first_class_established_hits_, so voteClass()'s motion gate never
            // engages for a just-spawned track regardless.
            voteClass(tr, eff_class, tr.last_obs, !detectionClass(d).empty());
            tracks_.emplace(tr.id, std::move(tr));
        }

        for (auto it = tracks_.begin(); it != tracks_.end();) {
            // Classified (whitelisted) tracks get more coasting patience — see max_missed_classified_'s own
            // comment (v1 parity: maxMissedFramesYolo_ vs maxMissedFrames_).
            const int effective_max_missed = isDynamicClass(it->second.best_class) ? max_missed_classified_ : max_missed_;
            const bool dead = it->second.missed_in_a_row > effective_max_missed ||
                              it->second.stationary_ticks > track_stationary_max_ticks_;
            if (dead && debug_lifecycle_)
                RCLCPP_INFO(get_logger(), "[lifecycle] delete #%d cls='%s' hits=%d missed=%d/%d stationary=%d/%d", it->first,
                            it->second.best_class.c_str(), it->second.hits, it->second.missed_in_a_row, effective_max_missed,
                            it->second.stationary_ticks, track_stationary_max_ticks_);
            if (dead && isDynamicClass(it->second.best_class)) {
                const int dead_id = it->first;
                const std::string dead_class = it->second.best_class;
                it = tracks_.erase(it);
                handOverClass(dead_id, dead_class);
                continue;
            }
            it = dead ? tracks_.erase(it) : std::next(it);
        }

        publish(msg->header, matched_ids);
    }

    // OUTPUT BOX COMPLETION for person tracks (published box only — association keeps using the observed box).
    // The LiDAR/depth points only cover what the sensors see: up close the VLP-16's +-15 deg cone cuts a person's
    // legs AND head (box bottom ~0.7 m and top ~1.3 m above the floor at 1 m, behind the robot), and only the
    // surface facing the sensor is seen, so the box centre sits ~7-12 cm short of the person's real centre along
    // the sensor ray (measured against motion capture on the lab bags). For a "person" track the published box is
    // therefore stood on the floor (floor_frame's height, the robot's own footprint), given at least
    // track_person_min_height, and its centre pushed track_person_ray_offset further along the LiDAR ray.
    void completePersonBox(const Track& tr, Eigen::Vector3d& center, vision_msgs::msg::BoundingBox3D::_size_type& size) const {
        if (!person_box_completion_ || tr.best_class != "person") return;
        if (lidar_tf_ok_ && person_ray_offset_ > 0.0) {
            Eigen::Vector2d ray(center.x() - lidar_origin_.x(), center.y() - lidar_origin_.y());
            if (ray.norm() > 1e-3) {
                ray.normalize();
                center.x() += person_ray_offset_ * ray.x();
                center.y() += person_ray_offset_ * ray.y();
            }
        }
        if (floor_ok_) {
            const double top = std::max(center.z() + 0.5 * size.z, floor_z_ + person_min_height_);
            center.z() = 0.5 * (floor_z_ + top);
            size.z = top - floor_z_;
        }
    }

    void publish(const std_msgs::msg::Header& header, const std::set<int>& matched_ids) {
        vision_msgs::msg::Detection3DArray arr;
        arr.header = header;
        arr.header.frame_id = global_frame_;
        visualization_msgs::msg::MarkerArray markers;
        jo_msgs::msg::ObstacleArray obstacles;
        obstacles.header = arr.header;
        std::set<int> now_ids;

        for (auto& [id, tr] : tracks_) {
            if (!tr.confirmed) continue;               // internal propagation only until confirmed — never published, matching v1
            if (!matched_ids.count(id)) continue;       // predicted-only this tick (missed) — never published either way, matching v1
            const Eigen::Vector3d v = tr.kf.vel();
            const double speed = planarSpeed(v);        // vz excluded — see planarSpeed's own comment
            // A track with no semantic evidence of a dynamic class, INSIDE a camera's FOV, must keep up its motion
            // evidence for longer before it is labelled dynamic: there the camera would already have recognized
            // a real person, so short "moving" bursts on unrecognized objects (a fragment next to a person,
            // background revealed behind one, a static object's box jittering) are by far the most common false
            // dynamic. Outside every FOV nothing can classify anything, so the longer wait would only delay real
            // movers (e.g. a person walking behind the robot): a shorter, separate debounce applies there.
            const int confirm_ticks = isDynamicClass(tr.best_class) ? dynamic_confirm_ticks_
                                    : insideAnyCameraFov(tr.kf.pos()) ? dynamic_confirm_ticks_unclassified_
                                    : dynamic_confirm_ticks_unclassified_out_fov_;
            const bool debounced_dynamic = tr.dynamic_ticks >= confirm_ticks;
            // Sticky-dynamic shortcut (v1 PATH 2, see the constructor's own comment): once enough of the
            // track's recent history was dynamic, don't require the FULL fresh debounce streak again after one
            // noisy dip — a single tick's instantaneous speed still has to clear the bar, though.
            int sticky_count = 0;
            for (bool b : tr.dynamic_hist) if (b) ++sticky_count;
            const bool sticky_dynamic = static_cast<int>(tr.dynamic_hist.size()) >= dynamic_sticky_window_ &&
                                        sticky_count >= dynamic_sticky_frames_ && speed >= dynamic_speed_thresh_;
            bool dynamic = debounced_dynamic || sticky_dynamic;
            // Size-sanity gate (v1: constrainSize_ — see the constructor's own comment on why this is scoped to
            // UNCLASSIFIED tracks only): an implausibly-sized moving blob never gets the DYNAMIC label, though
            // it still publishes as "static" rather than being suppressed outright by this check alone.
            if (dynamic && dynamic_constrain_size_ && !isDynamicClass(tr.best_class)) {
                const bool sizeOk = std::abs(tr.size.x - dynamic_target_size_.x()) < dynamic_size_tolerance_.x() &&
                                    std::abs(tr.size.y - dynamic_target_size_.y()) < dynamic_size_tolerance_.y() &&
                                    std::abs(tr.size.z - dynamic_target_size_.z()) < dynamic_size_tolerance_.z();
                if (!sizeOk) dynamic = false;
            }
            // motion-earned PD (see track_motion_pd_min_dynamic_ticks): counts only the unclassified history
            if (!tr.best_class.empty()) {
                tr.motion_dynamic_ticks = 0;
                tr.motion_pd = false;
            } else if (dynamic && motion_pd_min_dynamic_ticks_ > 0 &&
                       ++tr.motion_dynamic_ticks >= motion_pd_min_dynamic_ticks_) {
                tr.motion_pd = true;
            }
            const bool exempt = potentiallyDynamic(tr) || track_publish_stationary_;
            // Suppress only truly-stationary (below the LOWER bar), unclassified tracks — a track above that bar
            // but below the dynamic-label bar (e.g. 0.2 m/s) still publishes, it just isn't "dynamic".
            if (!exempt && speed < stationary_speed_thresh_) continue;
            now_ids.insert(id);

            // Three-way status (v1 parity, framework_architecture.md's own "Marker color convention"/
            // STATUS_DYNAMIC|POTENTIALLY_DYNAMIC|STATIC): "dynamic" means moving now; "potentially_dynamic"
            // means YOLO recognized a WHITELISTED dynamic-class object (isDynamicClass(tr.best_class), see
            // semantic_dynamic_classes_'s own comment) that just isn't moving fast enough right now (a standing
            // person, a parked car) — this is the SAME semantic-evidence exemption STATIC TRACKS MUST DIE above
            // already uses to keep such a track alive/published, now finally surfaced as its own label instead
            // of collapsing into "static"; "static" is everything else, INCLUDING a track classified as a
            // non-whitelisted class (e.g. "chair") — that class still shows in the text label, it just doesn't
            // grant the potentially_dynamic exemption, same as v1.
            const bool potentially_dynamic = !dynamic && potentiallyDynamic(tr);
            const char* status = dynamic ? "dynamic" : (potentially_dynamic ? "potentially_dynamic" : "static");

            vision_msgs::msg::Detection3D d;
            d.header = arr.header;
            d.id = "track_" + std::to_string(id);
            Eigen::Vector3d box_center = tr.kf.pos();
            auto box_size = tr.size;
            completePersonBox(tr, box_center, box_size);
            d.bbox.center.position.x = box_center.x();
            d.bbox.center.position.y = box_center.y();
            d.bbox.center.position.z = box_center.z();
            d.bbox.center.orientation = tr.orientation;
            d.bbox.size = box_size;
            vision_msgs::msg::ObjectHypothesisWithPose h;
            h.hypothesis.class_id = status;
            h.hypothesis.score = speed;
            h.pose.pose.position.x = v.x();
            h.pose.pose.position.y = v.y();
            h.pose.pose.position.z = v.z();
            d.results.push_back(h);
            if (!tr.best_class.empty()) {
                vision_msgs::msg::ObjectHypothesisWithPose sem;
                sem.hypothesis.class_id = tr.best_class;
                sem.hypothesis.score = static_cast<double>(tr.class_votes.at(tr.best_class)) / std::max(tr.hits, 1);  // fraction of matched ticks that voted for this class
                d.results.push_back(sem);
            }
            arr.detections.push_back(d);

            // jo_msgs/ObstacleArray — see the file header's own comment. Only dynamic/potentially_dynamic
            // tracks are pushed, matching v1's publishDynamicObstacleArray() skipping static tracks outright;
            // a "static" track here (including one carrying a non-whitelisted class) is simply never added, so
            // glim_ros2's own bbox_callback (is_rejection_obstacle()) never even has to filter it back out.
            if (dynamic || potentially_dynamic) {
                jo_msgs::msg::Obstacle obs;
                obs.header = arr.header;
                obs.track_id = static_cast<uint32_t>(id);
                obs.track_age = static_cast<uint32_t>(tr.hits);
                obs.status = dynamic ? jo_msgs::msg::Obstacle::STATUS_DYNAMIC : jo_msgs::msg::Obstacle::STATUS_POTENTIALLY_DYNAMIC;
                obs.pose.position = d.bbox.center.position;
                obs.pose.orientation = tr.orientation;
                obs.size.x = box_size.x; obs.size.y = box_size.y; obs.size.z = box_size.z;
                obs.twist.linear.x = v.x(); obs.twist.linear.y = v.y(); obs.twist.linear.z = v.z();
                const Eigen::Vector3d a = tr.kf.acc();
                obs.accel.linear.x = a.x(); obs.accel.linear.y = a.y(); obs.accel.linear.z = a.z();
                obstacles.obstacles.push_back(obs);
            }

            appendTrackMarkers(markers, id, tr, box_center, box_size, dynamic, potentially_dynamic, speed, v, header);
        }
        for (int old : last_published_ids_)
            if (!now_ids.count(old)) {
                for (const char* ns : {"tracks", "tracks_vel", "tracks_label"}) {
                    visualization_msgs::msg::Marker del;
                    del.header = header;
                    del.header.frame_id = global_frame_;
                    del.ns = ns;
                    del.id = old;
                    del.action = visualization_msgs::msg::Marker::DELETE;
                    markers.markers.push_back(del);
                }
            }
        last_published_ids_ = now_ids;

        pub_tracks_->publish(arr);
        pub_markers_->publish(markers);
        pub_obstacles_->publish(obstacles);
    }

    void appendTrackMarkers(visualization_msgs::msg::MarkerArray& out, int id, const Track& tr,
                            const Eigen::Vector3d& c, const vision_msgs::msg::BoundingBox3D::_size_type& bsize, bool dynamic,
                            bool potentially_dynamic, double speed, const Eigen::Vector3d& vel,
                            const std_msgs::msg::Header& header) const {
        // Box/label color encodes the three-way dynamic/potentially_dynamic/static status (v1 parity — see
        // framework_architecture.md's own "Marker color convention"): dynamic -> blue, potentially_dynamic
        // (YOLO-recognized dynamic-class object currently stationary) -> green, static -> neutral grey.
        float r, g, b_col;
        if (dynamic)                   { r = 0.1f; g = 0.3f; b_col = 1.0f; }   // blue
        else if (potentially_dynamic)  { r = 0.15f; g = 0.85f; b_col = 0.15f; } // green
        else                            { r = 0.8f; g = 0.8f; b_col = 0.8f; }   // neutral grey
        const Eigen::Quaterniond q(tr.orientation.w, tr.orientation.x, tr.orientation.y, tr.orientation.z);
        const Eigen::Matrix3d R = q.normalized().toRotationMatrix();

        static const int edges[12][2] = {{0, 1}, {1, 3}, {3, 2}, {2, 0}, {4, 5}, {5, 7},
                                         {7, 6}, {6, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
        visualization_msgs::msg::Marker box;
        box.header = header;
        box.header.frame_id = global_frame_;
        box.ns = "tracks";
        box.id = id;
        box.type = visualization_msgs::msg::Marker::LINE_LIST;
        box.action = visualization_msgs::msg::Marker::ADD;
        box.pose.orientation.w = 1.0;
        box.scale.x = 0.05;
        box.color.r = r; box.color.g = g; box.color.b = b_col; box.color.a = 1.f;
        box.lifetime = rclcpp::Duration::from_seconds(marker_lifetime_sec_);
        for (const auto& e : edges)
            for (int v : {e[0], e[1]}) {
                const Eigen::Vector3d local(((v & 1) ? 0.5 : -0.5) * bsize.x, ((v & 2) ? 0.5 : -0.5) * bsize.y,
                                            ((v & 4) ? 0.5 : -0.5) * bsize.z);
                const Eigen::Vector3d p = c + R * local;
                geometry_msgs::msg::Point pt;
                pt.x = p.x(); pt.y = p.y(); pt.z = p.z();
                box.points.push_back(pt);
            }
        out.markers.push_back(box);

        if (dynamic) {
            visualization_msgs::msg::Marker arrow;
            arrow.header = header;
            arrow.header.frame_id = global_frame_;
            arrow.ns = "tracks_vel";
            arrow.id = id;
            arrow.type = visualization_msgs::msg::Marker::ARROW;
            arrow.action = visualization_msgs::msg::Marker::ADD;
            arrow.scale.x = 0.05; arrow.scale.y = 0.1; arrow.scale.z = 0.0;
            arrow.color.r = 1.f; arrow.color.g = 0.f; arrow.color.b = 0.f; arrow.color.a = 0.9f;
            arrow.lifetime = rclcpp::Duration::from_seconds(marker_lifetime_sec_);
            geometry_msgs::msg::Point p0, p1;
            p0.x = c.x(); p0.y = c.y(); p0.z = c.z() + bsize.z * 0.5 + 0.1;
            p1.x = c.x() + vel.x(); p1.y = c.y() + vel.y(); p1.z = p0.z + vel.z();
            arrow.points.push_back(p0);
            arrow.points.push_back(p1);
            out.markers.push_back(arrow);
        }

        visualization_msgs::msg::Marker label;
        label.header = header;
        label.header.frame_id = global_frame_;
        label.ns = "tracks_label";
        label.id = id;
        label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        label.action = visualization_msgs::msg::Marker::ADD;
        label.pose.position.x = c.x(); label.pose.position.y = c.y(); label.pose.position.z = c.z() + bsize.z * 0.5 + 0.25;
        label.scale.z = 0.25;
        label.color.r = r; label.color.g = g; label.color.b = b_col; label.color.a = 1.f;
        label.lifetime = rclcpp::Duration::from_seconds(marker_lifetime_sec_);
        const char* tag = dynamic ? "DYN" : (potentially_dynamic ? "POT" : "STA");
        char buf[96];
        if (tr.best_class.empty()) std::snprintf(buf, sizeof(buf), "#%d [%s] %.1fm/s", id, tag, speed);
        else std::snprintf(buf, sizeof(buf), "#%d [%s] %s %.1fm/s", id, tag, tr.best_class.c_str(), speed);
        label.text = buf;
        out.markers.push_back(label);
    }

    std::string global_frame_;
    bool debug_lifecycle_{false};
    std::string assignment_method_{"hungarian"};
    double veldir_gate_min_innovation_{0.30};
    double max_match_dist_{2.0}, max_match_speed_{8.0}, max_size_diff_{0.8}, min_size_diff_floor_{0.15};
    double w_pos_{1.0}, w_size_{0.35}, w_iou_{0.15}, w_prev_pos_{0.25}, w_prev_iou_{0.10}, min_match_score_{-2.5};
    int min_hits_{4}, max_missed_{5};
    int max_missed_classified_{15};
    double dynamic_speed_thresh_{0.35};
    double stationary_speed_thresh_{0.10};
    int dynamic_confirm_ticks_{3};
    int dynamic_confirm_ticks_unclassified_{8};
    int dynamic_confirm_ticks_unclassified_out_fov_{5};
    int motion_pd_min_dynamic_ticks_{20};
    int first_class_static_confirm_ticks_{5};
    double decay_fov_border_frac_{0.10};
    int max_non_yolo_fov_border_frames_{15};
    double class_handover_dist_{0.5}, class_handover_window_{3.0};
    double w_height_{0.5}, max_top_diff_{1.0}, height_ema_alpha_{0.3};
    int height_min_samples_{5};
    bool person_box_completion_{true};
    std::string floor_frame_{"base_footprint"};
    double person_min_height_{1.7}, person_ray_offset_{0.07};
    double floor_z_{0.0};
    bool floor_ok_{false};
    int first_class_confirm_ticks_{2};
    int first_class_established_hits_{10};
    int first_class_established_confirm_ticks_{2};
    double first_class_min_jump_{0.08};
    double class_match_bonus_{0.25};
    double class_continuity_weight_{0.5};
    double camera_fragment_dist_{0.6};
    bool class_ownership_{true};
    double match_veldir_weight_{0.12};
    double max_veldir_error_unconfirmed_{1.20}, max_veldir_error_confirmed_{2.50}, max_veldir_error_relaxed_{3.00};
    double outside_fov_max_match_dist_{3.5}, outside_fov_max_match_speed_{4.0}, outside_fov_max_size_diff_{1.2};
    double outside_fov_confirmed_bonus_{0.75}, outside_fov_tentative_bonus_{0.35};
    double min_natural_motion_dist_{0.08}, max_natural_motion_dist_{1.50};
    double max_natural_innovation_{0.60}, max_natural_innovation_outside_fov_{1.00};
    double confirmed_track_bonus_{0.20};
    double duplicate_track_dist_thresh_{0.5}, duplicate_track_iou_thresh_{0.3}, duplicate_track_size_thresh_{0.5};
    double min_confirm_obs_speed_{-1.0};
    int dynamic_sticky_frames_{20}, dynamic_sticky_window_{30};
    int dynamic_kf_confidence_window_{5};
    double dynamic_kf_std_ratio_{0.50};
    bool dynamic_constrain_size_{true};
    Eigen::Vector3d dynamic_target_size_{0.5, 0.5, 1.7}, dynamic_size_tolerance_{0.8, 0.8, 1.0};
    std::string fused_clusters_topic_;
    double point_vote_thresh_{0.7};
    int point_vote_max_points_{500};
    double point_vote_max_age_{0.5};
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_clusters_;
    sensor_msgs::msg::PointCloud2::ConstSharedPtr latest_clusters_;
    std::string lidar_frame_;
    int disocclusion_pos_hist_window_{10};
    double disocclusion_min_cos_angle_{0.97}, disocclusion_min_range_margin_{0.3};
    bool lidar_tf_ok_{false};
    Eigen::Vector3d lidar_origin_{0.0, 0.0, 0.0};
    bool track_publish_stationary_{false};
    int track_stationary_max_ticks_{50};
    double q_pos_{0.05}, q_vel_{0.5}, q_acc_{1.0}, r_pos_{0.04}, p_init_{1.0};
    double marker_lifetime_sec_{0.0};

    std::vector<std::string> camera_names_;
    std::string camera_info_topic_template_, camera_frame_template_, yolo2d_topic_template_;
    double yolo2d_max_age_{0.5}, yolo2d_min_iou_{0.2};
    int max_non_yolo_in_fov_frames_{5};
    int max_yolo_base_mismatch_frames_{5};
    double yolo_base_mismatch_thresh_{0.50};
    double yolo_base_shrink_thresh_{0.60};
    std::set<std::string> semantic_dynamic_classes_;
    std::vector<CamFov> cams_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_;
    rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr pub_tracks_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
    std::string obstacles_topic_;
    rclcpp::Publisher<jo_msgs::msg::ObstacleArray>::SharedPtr pub_obstacles_;

    std::map<int, Track> tracks_;
    std::set<int> last_published_ids_;
    int next_id_{1};
    double last_stamp_{-1.0};
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TrackerNode>());
    rclcpp::shutdown();
    return 0;
}
