#!/usr/bin/env python3

"""YOLO instance-segmentation + tracking node — sins's Phase 2
prototype (see ../README.md's phase plan: "replace YOLOv11 (bounding boxes)
with a YOLO segmentation model"). This node gets segmentation + per-instance
tracking working and publishing on EVERY camera in camera_names; it does NOT
yet feed preprocessing_node's point-cloud extraction (that hand-off — depth
aligned to a mask instead of a raw depth crop — is the next step once this
is validated against a real/bagged camera stream).

Model: Ultralytics YOLO11n-seg (COCO-pretrained) — the smallest seg
checkpoint, chosen because the GPU (RTX 4060 Laptop, 8GB) is shared with
glim's SLAM and preprocessing_node at the same time; step up to yolo11s-seg
(same weights_path convention) if recall on small/distant people is
insufficient.

Tracker: Ultralytics' own built-in tracker (ByteTrack by default, BoT-SORT
selectable via the `tracker` param) via model.track(..., persist=True) —
no extra dependency (both configs ship inside the ultralytics package
already installed in docker/Dockerfile) and no re-id model to run, which
BoT-SORT-with-appearance would add on top of an already GPU-constrained
budget. ByteTrack is IoU/motion-based only.

Multi-camera: this class supports N cameras in ONE process (one INDEPENDENT
`YOLO(weights_path)` instance per camera — not one shared model instance
called for both streams: Ultralytics' persist=True tracker state lives on
the model/predictor object, so feeding two unrelated camera streams through.
the same instance would let ByteTrack match a box from front_camera against
a track born on back_camera). That capability is still here and still used
by a standalone/single-camera run, but BOTH launch files
(run_detector.launch.py's detection_nodes, run_yolo.launch.py's yolo_nodes)
now launch ONE PROCESS PER CAMERA instead — each with camera_names
restricted to a single entry — following calibration_icp_node's pattern,
not static_structures_node's. Originally rejected here (see git history) on the
assumption GPU inference has nothing to gain from process-level isolation
the way GICP does; measurement said otherwise — two cameras sharing one
process/GIL/timer budget measurably dropped ~20% of incoming frames (30Hz
in, ~24Hz out), and splitting into separate processes removes that
contention entirely (own CUDA context, own dedicated worker thread with
nothing else competing for the GIL). The per-camera capability in this
file didn't need to change at all for that: the launch files just stopped
handing it more than one camera_names entry at a time. (Separately, the
timer itself turned out to have its own, independent loss source — see
the "Processing is WORKER-THREAD-DRIVEN" section below.)

Unlike jo_detector/scripts/yolo_detector/yolov11_detector.py, inference
does NOT hand-roll PIL/torch resizing or NMS — model.track() accepts a raw
BGR numpy frame directly and does its own preprocessing, tracking and
per-class filtering (via `classes=`) internally. That hand-rolled path in
the v1 node predates Ultralytics handling this robustly; no need to repeat
it here.

Processing is WORKER-THREAD-DRIVEN, not timer-polled — a change from this
file's original design, which cached the latest decoded frame in the
subscription callback and processed "whatever's cached now" off a periodic
timer_period_sec timer. That measurably lost frames (~18%, both cameras,
steady-state, even in their own dedicated per-camera process with GPU time
to spare) whenever a camera's own publish jitter (measured 23-43ms, not a
clean 33ms) put two frames inside one timer period: the second silently
overwrote the first before the timer got to it. Timer-vs-publisher are two
independent clocks — no period alignment between them fixes that.

The fix is NOT "process every message serially" either (tried, rejected):
under a burst or a slow frame, a plain per-message queue backs up and
LATENCY grows unbounded — every subsequent frame gets processed later and
later, which is worse for a real-time perception feed than dropping a
frame outright. What this file does instead, per camera: the subscription
callback (on the main executor thread) only decodes the JPEG and stashes
the result as ctx.pending_img/pending_header (overwriting any not-yet-
picked-up previous one) before signaling ctx.new_frame_event — cheap, never
blocks. A dedicated background thread per camera (ctx.thread, started in
__init__) loops: wait for that event, atomically take whatever is
currently pending (clearing it), and run inference+publish on it. If a
second frame arrives while the worker is still busy on the first, it just
overwrites pending_img — dropped, same "freshest wins" semantics as the
original timer design — but the worker itself is never gated by a fixed
clock: it picks up the next pending frame the INSTANT it finishes the
current one, so at ~15ms of work against a ~33ms average gap it actually
keeps up with nearly every frame in practice, unlike the old fixed-33ms
timer. Bounded latency (nothing ever queues up) AND no periodic-phase loss
— the two properties the timer design and a naive per-message queue each
got only one of.

Layout: the NODE is yolo_seg_track_node.py, in src/ alongside every other
node's source (preprocessing_node.cpp, static_structures_node.cpp,
calibration_icp_node.cpp) and installed to lib/sins/ like
their executables — that's the thin entrypoint ros2 run actually invokes.
This file is the SCRIPT it wraps: the detection/tracking implementation,
under scripts/ (share/sins/scripts/ once installed, found
at runtime via get_package_share_directory — see yolo_seg_track_node.py).
weights_path/class_names_path below are resolved the same way, against
share/sins/ (weights/ and cfg/ are their own top-level
install destinations, not installed next to scripts/).
"""

import os
import threading
import time

import cv2
import numpy as np
import rclpy
import torch
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float64
from ultralytics import YOLO
from vision_msgs.msg import BoundingBox2D, Detection2D, Detection2DArray, ObjectHypothesisWithPose

device = "cuda" if torch.cuda.is_available() else "cpu"


def _to_image_msg(cv_img, encoding, header):
    """Build a sensor_msgs/Image directly instead of going through
    cv_bridge.cv2_to_imgmsg(). NOT a style choice: this container's numpy
    was upgraded to 2.x for ultralytics (Dockerfile pins "numpy>=2.0"), but
    the apt-installed cv_bridge's compiled boost extension was built
    against numpy 1.x — encoding_to_cvtype2()'s dtype lookup then returns a
    cv_type not in cv_bridge's own cvtype_to_name table, and cv2_to_imgmsg
    raises `KeyError: 16` for EVERY image, regardless of content (confirmed
    against jo_detector's own yolov11_detector.py publish path too, so
    this isn't new to this node). A plain sensor_msgs/Image has no
    OpenCV-side type checking to trip over: encoding just needs to match
    cv_img's actual dtype/channel count, which the caller controls."""
    msg = Image()
    msg.header = header
    msg.height, msg.width = cv_img.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    channels = 1 if cv_img.ndim == 2 else cv_img.shape[2]
    msg.step = msg.width * channels * cv_img.dtype.itemsize
    msg.data = np.ascontiguousarray(cv_img).tobytes()
    return msg


class _CameraContext:
    """Per-camera state: own subscription, own model/tracker instance, own
    publisher set, own worker thread + pending-frame slot. See file header
    for why the model instance isn't shared across cameras, and for the
    lock/event/pending_img protocol between the subscription callback (main
    executor thread) and this camera's own worker thread."""

    __slots__ = (
        'name', 'model', 'first_logged',
        'img_pub', 'img_compressed_pub', 'mask_pub', 'det_pub', 'time_pub',
        'lock', 'new_frame_event', 'pending_img', 'pending_header', 'thread',
    )

    def __init__(self, name):
        self.name = name
        self.model = None
        self.first_logged = False
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.pending_img = None
        self.pending_header = None
        self.thread = None


class YoloSegTrackDetector(Node):
    def __init__(self):
        super().__init__('yolo_seg_track_detector')
        self.get_logger().info("[joDetectorV2]: yolo seg+track detector init...")

        # camera_names: same convention as static_structures_node/preprocessing_node
        # — run_detector.launch.py overrides this from preprocessing.yaml's
        # own camera_names so every node in the pipeline agrees on which
        # cameras exist; this default is only the standalone-run fallback.
        self.camera_names = list(self.declare_parameter('camera_names', ['front_camera', 'back_camera']).value)
        self.image_topic_template = str(self.declare_parameter('image_topic_template', '/{camera}/camera/color/image_raw').value)
        self.detected_image_topic_template = str(self.declare_parameter('detected_image_topic_template', '').value)
        self.detected_masks_topic_template = str(self.declare_parameter('detected_masks_topic_template', '').value)
        self.detected_detections_topic_template = str(self.declare_parameter('detected_detections_topic_template', '').value)
        self.inference_time_topic_template = str(self.declare_parameter('inference_time_topic_template', '').value)
        self.weights_path = str(self.declare_parameter('weights_path', '').value)
        self.class_names_path = str(self.declare_parameter('class_names_path', '').value)
        self.target_classes = list(self.declare_parameter('target_classes', ['person']).value)
        self.score_threshold = float(self.declare_parameter('score_threshold', 0.5).value)
        self.inference_size = int(self.declare_parameter('inference_size', 640).value)
        self.queue_size = int(self.declare_parameter('queue_size', 10).value)
        self.jpeg_quality = int(self.declare_parameter('jpeg_quality', 80).value)
        # 'bytetrack.yaml' (default, motion/IoU only) or 'botsort.yaml'
        # (adds camera-motion compensation + optional re-id, heavier) — both
        # ship inside the installed ultralytics package, resolved by name.
        # Only used when enable_tracking (below) is true.
        self.tracker = str(self.declare_parameter('tracker', 'bytetrack.yaml').value)
        # true (default, existing behavior unchanged): model.track(),
        # persistent ids via self.tracker. false: plain model.predict(), no
        # tracker at all — no persist state, no tracker dependency, nothing
        # tracking-specific that can fail. Added for cfg/<env>/
        # detection_yolo26.yaml specifically (see that file's own comment):
        # get segmentation working on a new checkpoint first, without also
        # having to debug the tracker on top of it. mask_label/Detection2D
        # ids become a per-frame index, not a persistent track id, when this
        # is false — see infer_and_annotate's own comment on that.
        self.enable_tracking = bool(self.declare_parameter('enable_tracking', True).value)

        missing = [
            name for name, val in [
                ('camera_names', self.camera_names),
                ('detected_image_topic_template', self.detected_image_topic_template),
                ('detected_masks_topic_template', self.detected_masks_topic_template),
                ('detected_detections_topic_template', self.detected_detections_topic_template),
                ('inference_time_topic_template', self.inference_time_topic_template),
                ('weights_path', self.weights_path),
                ('class_names_path', self.class_names_path),
                ('target_classes', self.target_classes),
            ] if not val
        ]
        if missing:
            self.get_logger().error('Missing required parameters in YAML: ' + ', '.join(missing))
            raise RuntimeError('YOLO seg+track parameter initialization failed')

        # weights_path/class_names_path are share-relative (e.g.
        # 'weights/yolo11n-seg.pt', 'cfg/coco.names') — resolved against the
        # PACKAGE SHARE dir, not this file's own install dir (lib/), since
        # weights/ and cfg/ are their own install(DIRECTORY ...)
        # destinations under share/sins/. See file header.
        pkg_share = get_package_share_directory('sins')
        self.weights_path = os.path.join(pkg_share, self.weights_path)
        self.class_names_path = os.path.join(pkg_share, self.class_names_path)
        self.use_half = (device == "cuda")

        with open(self.class_names_path, 'r') as f:
            self.label_names = [line.strip() for line in f.readlines()]
        # Ultralytics filters by class INDEX, not name — resolve
        # target_classes (names, matches jo_detector's own param
        # convention) against class_names_path once at startup instead of
        # filtering every detection post-hoc.
        self.target_class_ids = [
            i for i, name in enumerate(self.label_names) if name in self.target_classes
        ]
        unknown = set(self.target_classes) - set(self.label_names)
        if unknown:
            self.get_logger().warning(f"target_classes not found in {self.class_names_path}: {sorted(unknown)}")
        if not self.target_class_ids:
            self.get_logger().error('None of target_classes matched class_names_path — no detections will ever pass.')
            raise RuntimeError('YOLO seg+track target_classes resolved to an empty set')

        self.cameras = {}
        for name in self.camera_names:
            ctx = _CameraContext(name)

            ctx.model = YOLO(self.weights_path)
            if ctx.model.task != 'segment':
                self.get_logger().warning(
                    f"weights_path '{self.weights_path}' is a '{ctx.model.task}' model, not 'segment' — "
                    "masks will be empty. Point weights_path at a *-seg.pt checkpoint."
                )

            image_topic = self.image_topic_template.format(camera=name)
            # Same image_transport-mangled compressed-topic subscription as
            # yolov11_detector.py — see that file's comment for why (no
            # rclpy-side transport-hint helper).
            self.create_subscription(
                CompressedImage, image_topic + '/compressed',
                self._make_image_callback(ctx), self.queue_size)

            detected_image_topic = self.detected_image_topic_template.format(camera=name)
            ctx.img_pub = self.create_publisher(Image, detected_image_topic, self.queue_size)
            # Compressed (JPEG) counterpart of the annotated debug image,
            # published alongside the raw one at the image_transport
            # convention (<topic>/compressed) — cheaper to view over the
            # network (rviz/foxglove) than the raw stream, matching how
            # every camera driver in this stack already offers both. Only
            # the annotated color image gets this: detected_masks_topic
            # below is an int32 label map (track ids), which JPEG/PNG can't
            # represent losslessly, so it stays raw-only.
            ctx.img_compressed_pub = self.create_publisher(
                CompressedImage, detected_image_topic + '/compressed', self.queue_size)
            ctx.mask_pub = self.create_publisher(
                Image, self.detected_masks_topic_template.format(camera=name), self.queue_size)
            ctx.det_pub = self.create_publisher(
                Detection2DArray, self.detected_detections_topic_template.format(camera=name), self.queue_size)
            ctx.time_pub = self.create_publisher(
                Float64, self.inference_time_topic_template.format(camera=name), 1)

            self.cameras[name] = ctx

        self.get_logger().info(
            f"YOLO seg+track configured: device={device}, half={self.use_half}, "
            f"enable_tracking={self.enable_tracking}, tracker={self.tracker if self.enable_tracking else 'n/a'}, "
            f"cameras={self.camera_names}, weights={self.weights_path}, "
            f"target_classes={self.target_classes} (ids={self.target_class_ids}), "
            f"score_threshold={self.score_threshold:.2f}"
        )

        # One worker thread per camera — started here, after that camera's
        # publishers exist. See file header for the lock/event/pending_img
        # protocol between this thread and _make_image_callback below.
        # daemon=True: never blocks interpreter exit on its own, though
        # destroy_node() below still stops/joins it properly on a normal
        # ROS shutdown.
        self._stop_event = threading.Event()
        for ctx in self.cameras.values():
            ctx.thread = threading.Thread(
                target=self._worker_loop, args=(ctx,), name=f'yolo_worker_{ctx.name}', daemon=True)
            ctx.thread.start()

    def destroy_node(self):
        # Wake every worker (they're blocked on new_frame_event.wait())
        # so each notices _stop_event and exits, instead of leaking
        # threads past node shutdown.
        self._stop_event.set()
        for ctx in self.cameras.values():
            ctx.new_frame_event.set()
            if ctx.thread is not None:
                ctx.thread.join(timeout=2.0)
        super().destroy_node()

    def _make_image_callback(self, ctx):
        def _cb(msg):
            try:
                compressed = np.frombuffer(msg.data, dtype=np.uint8)
                decoded = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
                if decoded is None:
                    raise ValueError(f"cv2.imdecode failed (format='{msg.format}', {len(msg.data)} bytes)")
                img = np.ascontiguousarray(decoded, dtype=np.uint8)

                # Hand off to ctx's worker thread: stash + signal, don't
                # process here. This callback runs on the main executor
                # thread — keeping it to "decode and stash" only (a couple
                # ms) means it's always ready for the NEXT message the
                # instant it arrives, even while the worker is still busy
                # on a previous frame. Overwrites any not-yet-picked-up
                # pending_img — see file header for why that's correct
                # ("freshest wins", not a queue).
                with ctx.lock:
                    ctx.pending_img = img
                    ctx.pending_header = msg.header
                    ctx.new_frame_event.set()

                if not ctx.first_logged:
                    h, w = img.shape[:2]
                    self.get_logger().info(f"[{ctx.name}] First image received ({w}x{h}, format={msg.format})")
                    ctx.first_logged = True

            except Exception as e:
                self.get_logger().error(f"[{ctx.name}] Image callback error: {type(e).__name__}: {e}")
        return _cb

    def _worker_loop(self, ctx):
        """Runs on ctx's own dedicated thread for the node's whole
        lifetime. Blocks until a frame is pending, takes exactly that one
        (clearing pending_img so a frame is never processed twice), then
        runs inference+publish on it — all off the main executor thread, so
        the subscription callback above is never blocked by GPU work."""
        while not self._stop_event.is_set():
            if not ctx.new_frame_event.wait(timeout=1.0):
                continue  # plain timeout check for _stop_event; loop again
            with ctx.lock:
                img = ctx.pending_img
                header = ctx.pending_header
                ctx.pending_img = None
                ctx.new_frame_event.clear()
            if img is None or self._stop_event.is_set():
                continue
            self._process_frame(ctx, img, header)

    def _process_frame(self, ctx, img_bgr, header):
        start_time = time.time()
        try:
            annotated, detections, mask_label = self.infer_and_annotate(ctx, img_bgr, header)

            ctx.img_pub.publish(_to_image_msg(annotated, "bgr8", header))

            ok, jpeg_buf = cv2.imencode(
                '.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if ok:
                compressed_msg = CompressedImage()
                compressed_msg.header = header
                compressed_msg.format = 'jpeg'
                compressed_msg.data = jpeg_buf.tobytes()
                ctx.img_compressed_pub.publish(compressed_msg)
            else:
                self.get_logger().warning(f"[{ctx.name}] JPEG encode of annotated image failed")

            ctx.mask_pub.publish(_to_image_msg(mask_label, "32SC1", header))

            det_array = Detection2DArray()
            det_array.header = header
            det_array.detections = detections
            ctx.det_pub.publish(det_array)
        except Exception:
            import traceback
            self.get_logger().error(f"[{ctx.name}] Error in inference:\n{traceback.format_exc()}")
        elapsed = time.time() - start_time
        time_msg = Float64()
        time_msg.data = float(elapsed)
        ctx.time_pub.publish(time_msg)

    def infer_and_annotate(self, ctx, img_bgr, header):
        """Run seg(+track, if enable_tracking) on one BGR frame from ctx's
        camera (ctx.model is that camera's OWN model/tracker instance — see
        file header). Returns (annotated_bgr, Detection2D list, mask_label
        int32 HxW image where each pixel holds the id of the instance
        covering it, 0 = background — a persistent track id when
        enable_tracking, otherwise just this frame's own detection index,
        see the id-handling comment below)."""
        common_kwargs = dict(
            classes=self.target_class_ids,
            conf=self.score_threshold,
            imgsz=self.inference_size,
            device=device,
            # quantize=16 is the current spelling of what used to be
            # half=True (fp16 inference) — 'half' still works but logs a
            # deprecation warning every single call.
            quantize=16 if self.use_half else None,
            verbose=False,
        )
        if self.enable_tracking:
            results = ctx.model.track(img_bgr, persist=True, tracker=self.tracker, **common_kwargs)
        else:
            # Plain per-frame inference, no tracker at all — no persist
            # state, no tracker config/dependency, nothing that can fail on
            # the tracking side specifically. See enable_tracking's own
            # declare_parameter comment for why this exists as a separate
            # path instead of always tracking.
            results = ctx.model.predict(img_bgr, **common_kwargs)
        result = results[0]

        # result.plot() draws boxes + segmentation masks (+ track ids, only
        # when boxes.id is set) in one call either way — no need to
        # hand-roll PIL/cv2 drawing like yolov11_detector.py's postprocess()
        # does for plain boxes.
        annotated = result.plot()

        h, w = img_bgr.shape[:2]
        mask_label = np.zeros((h, w), dtype=np.int32)

        detections = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return annotated, detections, mask_label
        if self.enable_tracking and boxes.id is None:
            # Tracking mode specifically: no CONFIRMED tracks yet this frame
            # (e.g. the first few frames before ByteTrack assigns an id) —
            # publish an empty frame rather than detections with no track
            # id, since track id is what downstream consumers key on. Only
            # applies when tracking is on; plain predict() never sets
            # boxes.id at all, which is the normal case below, not this one.
            return annotated, detections, mask_label

        # ids: a real, PERSISTENT track id across frames when enable_tracking
        # (boxes.id, assigned by self.tracker); otherwise NEGATIVE
        # (-1, -2, ...), not a plain 0..N-1 index — still unique WITHIN this
        # one frame (preprocessing_node/dbscan_detector_node's onLeaves()
        # still groups a mask's per-pixel labels by it), but a downstream
        # consumer that treats this id as a cross-frame persistent identity
        # (tracker_node's decaySemanticEvidence(), see its own comment) sees
        # a negative value and correctly refuses to trust it — see
        # yolo_semantic_seg_detector.py's own identical comment, this is the
        # same fix mirrored for the mask/instance-seg path.
        ids = (
            boxes.id.cpu().numpy().astype(np.int64) if self.enable_tracking
            else -(np.arange(len(boxes), dtype=np.int64) + 1)
        )
        cls_ids = boxes.cls.cpu().numpy().astype(np.int64)
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        # mask.xy: per-instance polygon in ORIGINAL image pixel coords
        # (Ultralytics rescales internally), same order as boxes — filling
        # polygons directly sidesteps having to know/resize masks.data's
        # own (possibly downscaled) resolution.
        polygons = result.masks.xy if result.masks is not None else [None] * len(ids)

        for i in range(len(ids)):
            obj_id = int(ids[i])
            polygon = polygons[i] if i < len(polygons) else None
            # mask_label's 0 means background. A real track id (enable_tracking=true) is assigned by
            # Ultralytics' own tracker starting at 1, so it's used as-is. A plain per-frame detection index
            # (enable_tracking=false) DOES start at 0 — offset by 1 so it never collides with background.
            # det.id below must carry this SAME value, not the raw obj_id, since preprocessing_node's mask
            # branch keys its byLabel map off det.id to look up each mask pixel's own label value.
            label_value = obj_id if self.enable_tracking else obj_id + 1
            if polygon is not None and len(polygon) >= 3:
                cv2.fillPoly(mask_label, [polygon.astype(np.int32)], color=label_value)

            det = Detection2D()
            det.header = header
            # Must match the mask's own label_value exactly (consumers key byLabel off this id) —
            # label_value, not the raw obj_id, since enable_tracking=false shifts it by +1 to avoid colliding
            # with background (0). See mask_label's own comment above on this offset.
            det.id = str(label_value)
            x1, y1, x2, y2 = xyxy[i]
            bbox = BoundingBox2D()
            bbox.center.position.x = float((x1 + x2) / 2.0)
            bbox.center.position.y = float((y1 + y2) / 2.0)
            bbox.size_x = float(x2 - x1)
            bbox.size_y = float(y2 - y1)
            det.bbox = bbox
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = self.label_names[cls_ids[i]]
            hyp.hypothesis.score = float(confs[i])
            det.results.append(hyp)
            detections.append(det)

        return annotated, detections, mask_label
