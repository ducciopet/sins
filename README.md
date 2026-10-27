# SINS: Still Is Not Static

Dynamic and potentially-dynamic obstacle detection and tracking from LiDAR and RGB-D, with a three-state label per
track: DYNAMIC (moving), POTENTIALLY_DYNAMIC (standing still but able to move, e.g. a person) and STATIC. ROS 2
package `sins` (named `jo_detector_v2` until 2026-10-26; topics were `/jo_detector_v2/...`, now `/sins/...`).

Ground-up rewrite of [`jo_detector`](../jo_detector), built one
stage at a time. `jo_detector` is untouched and stays the deployed
package (referenced in `docker/Dockerfile`, `run_detector.launch.py`, etc.)
until a given stage here is proven out and ready to take over.

## Phase plan

1. **Preprocessing** (current) — depth/lidar ingestion and cleanup.
   `src/preprocessing_node.cpp`: LiDAR is time-synced with odometry (via
   `message_filters`); depth is handled as two independent, unsynced streams
   (see Status below) — none of it goes through the raw (uncompressed)
   topics. Reference for what the current pipeline does at the LiDAR stage:
   `dynamicDetector.cpp` `lidarPoseCB`/`lidarOdomCB` (~line 1600 on) — range
   crop, distance-weighted subsample, ground/roof height crop, wall-bbox
   exclusion, adaptive voxel downsample.
2. **Detection** — replace YOLOv11 (bounding boxes) with a YOLO segmentation
   model. Needs deciding early because it changes preprocessing too: a
   segmentation mask lets you extract the point cloud per-mask-pixel instead
   of per-bounding-box, which changes what phase 1 needs to hand off (depth
   aligned to a mask, not just a raw depth crop).
3. **Sloped terrain** (done, folded into phase 1) — `static_structures_node`
   fits a real plane (`ax + by + cz + d = 0`, EMA-tracked, one PER CAMERA
   — `CameraSource::floor_plane_global`) instead of jo_detector's
   original flat `z = c` ground-height estimate, and both the LiDAR and
   depth crops in `preprocessing_node` filter against the union of the
   space below every camera's plane. See Status below and
   `docs/framework_architecture.md` §5.
4. **Tracking + classification** — last, since it's the most sensitive to
   unstable input from the stages above.

## Layout

Config is split by environment, one subfolder each, not by filename suffix:

```
cfg/
  indoor/
    calibration_icp.yaml
    preprocessing.yaml
    static_structures.yaml
  outdoor/
    calibration_icp.yaml
    preprocessing.yaml
    static_structures.yaml
```

Applied consistently to every node in this package (not just calibration) so
there's a single convention, not a per-node one. A launch file selects a
whole environment with one `env:=indoor|outdoor` argument
(`PathJoinSubstitution([pkg_dir, 'cfg', env, '<node>.yaml'])`); node config
filenames are just the node's role, no platform/robot qualifier baked in —
add a further subfolder level if a second platform ever needs its own
profile.

**Launch files: one per standalone workflow, not one per phase.**
`run_calibration_icp.launch.py` stays its own file — it's a genuinely
different workflow (tuning, with rviz, run in isolation). Everything else
lives in `run_detector.launch.py`, the single "launch the pipeline" file
that grows a node at a time as phases are built (mirrors how
`jo_detector/launch/run_detector.launch.py` grew: detector_node, then
static_structures_node, then yolo_node, ...). When the detection phase exists,
its `Node(...)` and config lookup get added to that same file, not to a new
`run_detection.launch.py`.

## Status

See [`docs/framework_architecture.md`](docs/framework_architecture.md) for
the full technical reference (every node's I/O, the LiDAR/depth pipelines
step by step, the wall registry). This section stays a narrative changelog.

**Calibration**: `src/calibration_icp_node.cpp` is a straight port of
`jo_detector`'s node of the same name — same gtsam_points/GICP
registration, same manual lidar/depth sync (message_filters' own
`Synchronizer<ApproximateTime>` doesn't work here: at least the `zstd`
image_transport decoder returns a zeroed `header.stamp`, so scans were
cached and matched by arrival time instead — see the comment on
`lidar_cache_` in the file). The only change from the original: parameter
names dropped the stale `jo_detector.` prefix (`depth_intrinsics`, not
`jo_detector.depth_intrinsics`) for consistency with the flat naming
`preprocessing_node.cpp` already uses. `run_calibration_icp.launch.py` is
also a straight port, including `robot_tf_camera_frame` (use the bag/robot's
own TF chain as the initial guess instead of a hand-measured static TF —
the hand-measured one was found to be ~42cm off against
`bags/*_validation_lab`, well past `icp_max_correspondence_distance`, which
made ICP converge to something worse than the initial guess instead of
refining it).

Found while porting: the default `depth_transport` for calibration is
`zstd` (matches that bag), but `zstd_image_transport` wasn't in
`docker/Dockerfile` at all — added it there and to this package's
`package.xml`.

Also found while chasing `ros2 bag play --loop` support: one-shot
calibration used to publish `camera_refined` exactly once on `/tf_static`.
That's permanently lost from a listener's TF buffer after a backward
clock jump (which `--loop` causes on every restart, since `tf2_ros::Buffer`
clears itself on one), since nothing ever re-sends it. Fixed by always
running the (already-existing, previously periodic-mode-only) 500ms `/tf`
heartbeat — same unchanged result, just continuously re-asserted instead of
sent once. A second, separate `--loop` failure mode is still open — see
`docs/framework_architecture.md`'s "Known limitations" section.

**Preprocessing**: `src/preprocessing_node.cpp` — LiDAR (general-purpose
cloud: range crop + Gaussian subsample + global transform + slope-aware
floor/roof crop + voxel downsample + wall-plane strip) and depth (one
independent pipeline per camera — see below). It does **not** run any
wall/floor RANSAC itself anymore — see the next entry. Full technical
reference: `docs/framework_architecture.md`.

**Wall/floor detection is now a separate, asynchronous node** —
`src/static_structures_node.cpp` — architecturally a faithful port of
`jo_detector/scripts/wall_detector/wallDetector.{h,cpp}` (own
executable, own `rclcpp::spin()`, no coupling to `preprocessing_node`'s own
callbacks, exactly like the original's `dynamicDetector`/`wall_detector_node`
split, which `sins` didn't have until this pass — wall/floor
detection used to run inline inside `preprocessing_node`'s own LiDAR
callback). Two deliberate differences from the original: (1) it reads
`preprocessing_node`'s already-voxel-downsampled — but NOT range/ground-
cropped — LiDAR and per-camera depth clouds
(`/sins/lidar/voxelized_sensor`,
`/sins/<camera>/depth/voxelized_sensor`) instead of
re-subscribing to raw sensor topics and re-voxelizing itself, so the
expensive voxel-grid pass happens exactly once, shared; its own range crop
still happens on the cheap, already-downsampled centroids. (2) floor is a
real, potentially-tilted plane fit by the SAME RANSAC engine that finds
walls (near-horizontal classification, EMA-tracked), not the original's
separate flat depth-image bottom-rows estimate.

**Detection is split by source, on purpose**: LiDAR only ever detects
walls, every camera in `camera_names` only ever detects floor/ground
(`detectFromSensorPoints`'s `detect_walls`/`detect_floor` flags) — the
LiDAR's 360°, consistent-height scan is the better source for vertical
walls across the whole range, while a downward-angled depth camera sees
the floor immediately ahead at far higher density than the LiDAR's own
sparse near-field rings, which is what actually matters for catching a
slope early. Publishes `/sins/static_structures/wall_markers`
(`MarkerArray`, `ns="wall_bbox"` + `ns="floor"`), which `preprocessing_node`
parses back into filtering planes (`onWallMarkers()`) and rviz subscribes
to directly. See `docs/framework_architecture.md` §5.

**Wall thickness fix**: tracked walls used to grow thicker without bound
over time — traced to two compounding causes, both fixed in
`static_structures_node`. (1) Wall RANSAC's input cloud was accidentally the
same Gaussian-randomly-subsampled cloud used for the general-purpose
output (from when detection still lived in `preprocessing_node`), so each
scan's random subset gave the fitted plane's normal a different jitter;
fixed by giving RANSAC a clean, deterministic-ish input (the voxelized
clouds described above), plus a PCA `refitPlane` step (ported from
`jo_detector/wall_detector`, was missing here) that averages the
normal over all inliers instead of trusting the raw 3-point RANSAC
candidate. (2) `WallBBoxRegistry::merge()`'s box-union never shrinks and
freezes `rotation` at the first detection, so even the residual per-scan
orientation noise from (1) still unioned onto the wall's thickness axis
forever — and a few degrees of allowed mismatch, projected through a long
wall's length, threw real meters onto that axis. Fixed by EMA-blending
thickness (`wall_merge_weight`, finally used for something) instead of
unioning it, while still unioning length/height (legitimate growth as more
of a wall comes into view). See `docs/framework_architecture.md` §5.

**Slope-aware floor, one plane per camera**: ground/roof filtering no
longer imposes a flat, perfectly horizontal `ground_height`/`roof_height`
Z-band, and no longer forces a single shared floor plane either. Each
camera in `static_structures_node` fits its OWN real plane (`ax+by+cz+d=0`)
from the same RANSAC pass wall detection uses (classifying the largest
near-horizontal extracted plane, within `floor_max_tilt_deg` of level, as
that camera's floor candidate) and EMA-blends it into that camera's own
persistent `cam.floor_plane_global` (`floor_ema_alpha`) — never blended
with another camera's — then publishes it as its own `ns="floor"` marker
(id = that camera's index, distinct colors in rviz). `preprocessing_node`
receives ALL of them via `onWallMarkers()` into `floor_planes_global_`, and
both the LiDAR crop and every camera's depth crop filter against
`heightAboveGround(p)` — the MINIMUM signed distance across every one of
those planes — instead of one plane's distance or a constant world-Z
threshold. Taking the minimum is exactly the union of "below any camera's
plane": a point close to or below EITHER plane gets dropped as ground, and
a point survives the roof cutoff only once it's far above EVERY plane. So
a sloped section of floor tilts the crop instead of clipping it flat, two
cameras seeing the floor slightly differently (mounting offset, local
surface variation) both get to contribute rather than being forced into
one compromise plane, and both floor markers visibly tilt to match.

Both crops bound `heightAboveGround(p)` to `[wall_removal_margin,
ground_roof_offset]`, not `[0, ground_roof_offset]` — a real floor-surface
point's signed distance to its own fitted plane sits close to zero on
*either* side (RANSAC fits roughly through its inliers, not strictly above
them), so a plain "below the plane" test only ever dropped about half the
actual floor points and left the rest in both the LiDAR and depth
processed clouds. The margin bound drops anything close to a floor plane
at all, symmetric to the wall strip's own margin test. See
`docs/framework_architecture.md` §4/§5/§6.

**Per-camera calibration (`<name>_refined`)**: each camera in `camera_names`
now gets its own `calibration_icp_node` instance (`run_detector.launch.py`
loops over the same list `preprocessing_node`/`static_structures_node` use),
publishing its own `velodyne -> <name>_refined` TF — e.g.
`front_camera_refined`, `back_camera_refined` — instead of one shared
`camera_refined`. Each camera's depth-cloud crop (`preprocessing_node`) and
each camera's own wall/floor detection (`static_structures_node`) now go
through *its own* calibration (`T_global_base_latest_ × T_base_lidar_ ×
cam.T_lidar_camera`) to reach `global_frame_` — `preprocessing_node` falls
back to the robot's raw URDF/bag TF chain until that camera's calibration
TF is available; `static_structures_node` simply waits (no raw-chain fallback,
since a camera's detections would otherwise land in the wrong place
entirely). Previously the raw TF chain was the *only* path for depth, so
ICP's result never actually reached the depth pipeline at all. See
`docs/framework_architecture.md` §2, §5 and §6.

**Multi-camera**: depth isn't hardcoded to one camera. `camera_names` (a
list param, default `["front_camera"]`, this robot's cfg sets
`["front_camera", "back_camera"]`) drives construction of one `CameraStream`
per name in `preprocessing_node`'s constructor — each with its own topics,
intrinsics, cached TF and publishers
(`/sins/<name>/depth/{cloud,processed,voxelized_sensor}`) —
and one `CameraSource` per name in `static_structures_node`'s own constructor,
reading that last topic. Adding a camera is a cfg change, not a code
change, as long as it follows the same `/<name>/camera/...` topic-naming
convention this robot's cameras already do (front_camera's and
back_camera's topics matched exactly, so neither needed per-topic
overrides — only the shared per-bag `aligned_depth_transport: zstd`
override applies to both, plus giving `static_structures_node`'s own
`camera_names` the same list — `run_detector.launch.py` does this
automatically).

**Wall registry**: persistent tracking across scans, ported from
`jo_detector/wall_detector_node`'s `WallBBoxRegistry` — see
`include/sins/wall_registry.hpp` and
`docs/framework_architecture.md` §5. Lives in `static_structures_node` now (not
`preprocessing_node`, see above). A fresh RANSAC detection merges into a
matching existing entry (EMA-extend its length/height, EMA-blend its
thickness — see the wall-thickness-fix entry above) rather than replacing
it; an entry unmatched this scan is not deleted immediately — its
missed-frame count increments, and only once that exceeds
`wall_max_missed_frames` (default 30 scans, ~3s at 10Hz) does it actually
get dropped. `preprocessing_node` strips points against the registry's
*tracked* planes (received via `onWallMarkers()`), not just this scan's
fresh ones.

**Depth, two independent streams per camera** (see the file header comment
for the full reasoning): `aligned_depth_topic` (RealSense's own
depth-registered-to-color stream) is what actually gets deprojected into
the published depth cloud, using its own camera_info (which — per
`jo_navigation/config/visual_odom.yaml` — is the COLOR camera's
`camera_info`, not a depth one, since aligned depth lives in the color
image's pixel grid). `depth_topic` (the sensor's own, unaligned depth) is
only subscribed/decoded, not deprojected — it's there for the future
UV-map detection stage, which needs the raw depth image itself, not a
cloud. Both go exclusively through image_transport (never the raw topic),
transport plugin configurable per stream (`depth_transport`/
`aligned_depth_transport`, default `compressedDepth`) since recorded bags
(`bags/*_validation_lab`) only have aligned depth as `zstd`, not
`compressedDepth` — same situation `visodom.launch.py` already handles.
Each stream keeps its own intrinsics, never cross-used within a camera or
shared across cameras (see Multi-camera above).

**Native point cloud input, per camera, optional**: `<name>.aligned_depth_cloud_topic`
(unset by default) skips decoding+deprojecting `aligned_depth_topic`
entirely and reads an already-deprojected `PointCloud2` straight from that
topic instead (`onNativeDepthCloud()`) — e.g. RealSense's own `pointcloud`
filter output, `<name>/camera/depth/color/points`, present in
`bags/*_validation_lab`. Point acquisition (image vs. point cloud) is now
handled per its actual input type instead of always deprojecting: if a
depth **image** comes in, it gets deprojected (`processAlignedDepth`); if a
**point cloud** is already available, using it directly makes more sense
than reconstructing one from a source image. The only real wrinkle: a
native cloud arrives in `<name>_depth_optical_frame` (confirmed by actually
inspecting both topics in the bag), not `_color_optical_frame` like the
deprojected path — and `cam.T_lidar_camera` was calibrated against the
*color*-frame convention (`calibration_icp_node` uses the same deprojection
as `processAlignedDepth` internally). `finishDepthCloud()` — the shared
tail of both paths now — accounts for this with one extra step,
`T_lidar_camera × T_depth_color⁻¹` (`T_depth_color` = the RealSense factory
extrinsic, already parsed from `depth_to_color_extrinsics_topic` below,
cached as a matrix alongside the TF broadcast), before applying the
calibrated transform to a native-cloud point.

RealSense's own factory depth<->color extrinsic, per camera
(`depth_to_color_extrinsics_topic`, message type
`realsense2_camera_msgs/msg/Extrinsics` — **unverified against a real build,
see the file header NOTE**, this is the single highest-risk part of this
change) is subscribed and rebroadcast once as a static TF
(`depth_optical_frame -> color_optical_frame`), so the rest of the system
gets it through normal TF instead of a RealSense-specific message.

Not yet done / open: a per-camera calibration TF is cached once and never
invalidated, in both `preprocessing_node` and `static_structures_node` (a
periodic-recalibration `calibration_icp_node` wouldn't have its refined
result picked up by either — see `docs/framework_architecture.md` §6/§5),
no thread parallelism (deliberately sequential for now — see file
header), and a second `--loop` bag-playback failure mode
(message_filters::Synchronizer getting stuck after a backward clock jump —
see `docs/framework_architecture.md`'s "Known limitations").
