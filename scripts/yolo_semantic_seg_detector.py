#!/usr/bin/env python3

"""YOLO26 semantic segmentation node — generic, configurable-class semantic
segmentation, a separate capability from yolo_seg_track_detector.py's
instance segmentation + tracking (different Ultralytics task entirely:
'semantic', not 'segment' — see below).

Model: Ultralytics YOLO26s-sem, Cityscapes-pretrained (19 classes: road,
sidewalk, building, wall, fence, pole, traffic light, traffic sign,
vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle,
bicycle — see cfg/cityscapes.names, in that exact index order).
segmentation_classes (YAML, no default baked in behavior beyond the sample
config) picks which of those classes get folded into the binary
segmentation_mask/overlay published below — this node itself has no notion
of "floor"/"ground"/any other fixed purpose; a caller picks the classes that
matter for its own use case (e.g. road+sidewalk+terrain for a floor mask,
person+rider for a pedestrian mask, car+truck+bus for a vehicle mask, ...).
The full 19-class semantic_mask is always published regardless, so a
consumer that wants a different class combination than segmentation_classes
can just derive it from that without needing a second node instance.

API shape is fundamentally different from instance segmentation, not just a
different weights_path: a semantic model's result has no .boxes/.masks/
.probs at all (confirmed empirically — all None/empty for this task type).
The whole-frame classification lives in result.semantic_mask.data: a single
HxW uint8 tensor (already resized to the original image's own resolution,
no manual resize needed regardless of inference_size), one Cityscapes class
index per pixel. No tracker, no per-instance anything, no confidence score
per pixel either (Ultralytics doesn't expose one for this task) — so unlike
yolo_seg_track_detector.py there's no score_threshold/tracker/enable_tracking
here, nothing to toggle.

Multi-camera / threading: same worker-thread-per-camera design as
yolo_seg_track_detector.py, for the same reason (a fixed-period timer
measurably drops frames to phase misalignment against the camera's own
publish jitter — see that file's own header for the full measurement) — see
that file for the detailed protocol comment, not repeated here. Also the
same one-OS-process-per-camera launch pattern (run_detector.launch.py's own
semantic_seg_nodes, same as its detection_nodes) for the same GIL/CUDA-context
contention reason.

Since the worker thread only ever keeps the FRESHEST pending frame (see
_make_image_callback), a frame that arrives while the previous one is still
being inferred on is dropped, by design — this is the same
latency-over-completeness tradeoff yolo_seg_track_detector.py makes, not a
bug. frames_received/frames_dropped are tracked per camera and logged every
stats_log_period_sec seconds alongside a rolling average inference time, so
this tradeoff is visible instead of silent — see _maybe_log_stats.

semantic_color (the full 19-class colorized overlay) runs on its OWN third
thread per camera, separate from the inference worker above — it's a debug
visualization, not something any real consumer of segmentation_mask/
semantic_mask depends on, so its LUT-indexing + blend + JPEG-encode cost
must never slow down the inference path those consumers DO depend on. The
inference worker hands off (img_bgr, class_map, header) to this thread via
the same freshest-job/lock/event protocol as the image callback -> inference
worker handoff (see _color_worker_loop) instead of computing it inline, so a
slow color-overlay frame only ever drops more color frames, never delays the
next inference. Its own stats (jobs received/dropped, avg time) are logged
separately from the inference worker's — see _maybe_log_color_stats.

Layout: same NODE (src/) + SCRIPT (scripts/) split as
yolo_seg_track_node.py/yolo_seg_track_detector.py — see that file's own
header for why (installed executable vs. share-dir-resolved implementation).
"""

import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import rclpy
import torch
import torch.nn.functional as F
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float64, Header
from vision_msgs.msg import BoundingBox2D, Detection2D, Detection2DArray, ObjectHypothesisWithPose
from ultralytics import YOLO

device = "cuda" if torch.cuda.is_available() else "cpu"

# Official cityscapesScripts label colors (RGB, see cityscapesscripts/helpers/
# labels.py), in the same 0-18 index order as cfg/cityscapes.names — one
# fixed color per class, used for the full multi-class overlay published on
# semantic_color_topic_template (as opposed to detected_image, which only
# tints the classes chosen by segmentation_classes). Converted to BGR below
# since that's cv2's/this node's own convention (see _to_image_msg).
_CITYSCAPES_COLORS_RGB = [
    (128, 64, 128),   # 0 road
    (244, 35, 232),   # 1 sidewalk
    (70, 70, 70),     # 2 building
    (102, 102, 156),  # 3 wall
    (190, 153, 153),  # 4 fence
    (153, 153, 153),  # 5 pole
    (250, 170, 30),   # 6 traffic light
    (220, 220, 0),    # 7 traffic sign
    (107, 142, 35),   # 8 vegetation
    (152, 251, 152),  # 9 terrain
    (70, 130, 180),   # 10 sky
    (220, 20, 60),    # 11 person
    (255, 0, 0),      # 12 rider
    (0, 0, 142),      # 13 car
    (0, 0, 70),       # 14 truck
    (0, 60, 100),     # 15 bus
    (0, 80, 100),     # 16 train
    (0, 0, 230),      # 17 motorcycle
    (119, 11, 32),    # 18 bicycle
]


def _to_image_msg(cv_img, encoding, header):
    """Same reasoning as yolo_seg_track_detector.py's own _to_image_msg:
    cv_bridge's compiled extension is numpy-1.x-only in this container,
    cv2_to_imgmsg raises on every call regardless of content — build the
    message directly instead."""
    msg = Image()
    msg.header = header
    msg.height, msg.width = cv_img.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    channels = 1 if cv_img.ndim == 2 else cv_img.shape[2]
    msg.step = msg.width * channels * cv_img.dtype.itemsize
    msg.data = np.ascontiguousarray(cv_img).tobytes()
    return msg


def _avg_ms(dq):
    return 1000.0 * sum(dq) / len(dq) if dq else 0.0


def _p95_ms(dq):
    return 1000.0 * float(np.percentile(list(dq), 95)) if dq else 0.0


def _avg_speed(dq, key):
    return sum(d.get(key, 0.0) for d in dq) / len(dq) if dq else 0.0


class _CameraContext:
    """Per-camera state — see yolo_seg_track_detector.py's own
    _CameraContext for the full lock/event/pending_img protocol this
    mirrors exactly (own subscription, own model instance, own publishers,
    own worker thread + pending-frame slot). inference_times/frames_received/
    frames_dropped/last_stats_log are the stats-reporting addition — see
    _maybe_log_stats."""

    __slots__ = (
        'name', 'model', 'first_logged',
        'img_pub', 'img_compressed_pub', 'semantic_mask_pub', 'segmentation_mask_pub',
        'semantic_color_pub', 'semantic_color_compressed_pub', 'time_pub',
        'lock', 'new_frame_event', 'pending_img', 'pending_header', 'thread',
        'inference_times', 'frames_received', 'frames_dropped', 'last_stats_log',
        # semantic_color debug thread — own lock/event/pending-job slot and
        # own stats, entirely separate from the inference worker's above
        # (see file header on why this must never share the inference
        # path's timing/drop accounting).
        'color_lock', 'color_new_job_event', 'pending_color_img', 'pending_color_class_map',
        'pending_color_header', 'color_thread',
        'color_times', 'color_jobs_received', 'color_jobs_dropped', 'color_last_stats_log',
        'pending_color_seg_mask',
        'predict_times', 'post_times', 'publish_times', 'speed_times',
        # detection+tracking thread (enable_detection) — own model/tracker
        # instance, own freshest-frame slot and own stats, same protocol as
        # the semantic worker above.
        'det_model', 'det_pub', 'det_img_pub', 'det_img_compressed_pub', 'det_time_pub',
        'det_lock', 'det_event', 'det_pending_img', 'det_pending_header', 'det_thread',
        'det_times', 'det_received', 'det_dropped', 'det_last_stats_log', 'det_started',
        'det_seen', 'det_pending_recv', 'det_pending_scale', 'det_pending_msg', 'det_ring', 'det_targets', 'trig_offsets', 'trig_missed', 'decode_times', 'det_wait_times', 'det_age_times',
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
        self.inference_times = deque(maxlen=200)
        self.frames_received = 0
        self.frames_dropped = 0
        self.last_stats_log = time.time()

        self.color_lock = threading.Lock()
        self.color_new_job_event = threading.Event()
        self.pending_color_img = None
        self.pending_color_class_map = None
        self.pending_color_header = None
        self.color_thread = None
        self.color_times = deque(maxlen=200)
        self.color_jobs_received = 0
        self.color_jobs_dropped = 0
        self.color_last_stats_log = time.time()
        self.pending_color_seg_mask = None
        self.predict_times = deque(maxlen=200)
        self.post_times = deque(maxlen=200)
        self.publish_times = deque(maxlen=200)
        self.speed_times = deque(maxlen=200)

        self.det_model = None
        self.det_lock = threading.Lock()
        self.det_event = threading.Event()
        self.det_pending_img = None
        self.det_pending_header = None
        self.det_thread = None
        self.det_times = deque(maxlen=200)
        self.det_received = 0
        self.det_dropped = 0
        self.det_last_stats_log = time.time()
        self.det_started = time.time()
        self.det_seen = 0
        self.det_pending_scale = 1
        self.det_pending_msg = None
        self.det_ring = deque(maxlen=6)
        self.det_targets = deque(maxlen=8)
        self.trig_offsets = deque(maxlen=200)
        self.trig_missed = 0
        self.det_pending_recv = 0.0
        self.decode_times = deque(maxlen=200)
        self.det_wait_times = deque(maxlen=200)
        self.det_age_times = deque(maxlen=200)


class YoloSemanticSegDetector(Node):
    def __init__(self):
        super().__init__('yolo_semantic_seg_detector')
        self.get_logger().info("[joDetectorV2]: yolo semantic seg detector init...")

        # camera_names: same convention as yolo_seg_track_detector.py/
        # static_structures_node/preprocessing_node — run_detector.launch.py
        # overrides this from preprocessing.yaml's own list; this default is
        # only the standalone-run fallback.
        self.camera_names = list(self.declare_parameter('camera_names', ['front_camera', 'back_camera']).value)
        self.image_topic_template = str(self.declare_parameter('image_topic_template', '/{camera}/camera/color/image_raw').value)
        self.detected_image_topic_template = str(self.declare_parameter('detected_image_topic_template', '').value)
        self.semantic_mask_topic_template = str(self.declare_parameter('semantic_mask_topic_template', '').value)
        self.segmentation_mask_topic_template = str(self.declare_parameter('segmentation_mask_topic_template', '').value)
        self.semantic_color_topic_template = str(self.declare_parameter('semantic_color_topic_template', '').value)
        self.inference_time_topic_template = str(self.declare_parameter('inference_time_topic_template', '').value)
        self.weights_path = str(self.declare_parameter('weights_path', '').value)
        self.class_names_path = str(self.declare_parameter('class_names_path', '').value)
        # Which of class_names_path's classes get folded into
        # segmentation_mask/the overlay — purely a YAML choice, this node has
        # no built-in notion of what those classes are for. See file header.
        # Declared by TYPE, not by an empty [] default: rclpy re-infers a
        # declare_parameter's type from its default VALUE regardless of any
        # ParameterDescriptor.type passed alongside it (see node.py's
        # declare_parameters — descriptor.type gets overwritten by
        # Parameter.Type.from_parameter_value(value) whenever a plain value
        # is given), and an empty [] infers as BYTE_ARRAY, not STRING_ARRAY.
        # Passing Parameter.Type.STRING_ARRAY directly as the second
        # argument takes the OTHER branch of that function, which sets the
        # type without inferring from any value — the .value below then
        # comes from this camera's YAML override, which is always required
        # (see the missing-params check just below).
        self.segmentation_classes = list(self.declare_parameter(
            'segmentation_classes', Parameter.Type.STRING_ARRAY).value)
        self.inference_size = int(self.declare_parameter('inference_size', 1024).value)
        self.queue_size = int(self.declare_parameter('queue_size', 10).value)
        self.jpeg_quality = int(self.declare_parameter('jpeg_quality', 80).value)
        # BGR, matches cv2's own channel order (see infer_and_annotate) —
        # not RGB.
        self.overlay_color_bgr = list(self.declare_parameter('overlay_color_bgr', [0, 255, 0]).value)
        self.overlay_alpha = float(self.declare_parameter('overlay_alpha', 0.5).value)
        # Blend strength for semantic_color_topic_template's full 19-class
        # overlay (independent from overlay_alpha, which only applies to
        # segmentation_classes' subset on detected_image).
        self.semantic_color_alpha = float(self.declare_parameter('semantic_color_alpha', 0.6).value)
        # How often (seconds) each camera's worker logs its rolling inference
        # speed + frame-drop stats — see _maybe_log_stats.
        self.stats_log_period_sec = float(self.declare_parameter('stats_log_period_sec', 5.0).value)

        # false: don't load/run the semantic model at all (no semantic worker,
        # no semantic/overlay topics populated) — leaves only the optional
        # detection+tracking thread below. Class/LUT setup stays (cheap).
        sys.setswitchinterval(0.001)  # ROS callbacks wait less for the GIL held by inference threads
        self.enable_semantic = bool(self.declare_parameter('enable_semantic', True).value)
        # true: letterbox/BGR->RGB/normalize on the GPU and hand Ultralytics
        # a ready tensor, instead of its CPU preprocessing (measured ~9 ms of
        # CPU per frame under full-stack load — the machine is CPU-starved,
        # not GPU-bound). Same aspect ratio => no padding, same result.
        self.gpu_preprocess = bool(self.declare_parameter('gpu_preprocess', True).value) and device == 'cuda'

        # Optional 2D detection + tracking, run by a SECOND model (a
        # Cityscapes semantic checkpoint has no .boxes at all, see file
        # header) on its own thread per camera, fed from the same decoded
        # frame. Fully independent of the semantic worker's timing: it has
        # its own freshest-frame slot, so a slow detection frame never
        # delays or drops a semantic one (only GPU sharing couples them).
        self.enable_detection = bool(self.declare_parameter('enable_detection', False).value)
        self.detection_weights_path = str(self.declare_parameter('detection_weights_path', 'weights/yolo26s.pt').value)
        self.detection_class_names_path = str(self.declare_parameter('detection_class_names_path', 'cfg/coco.names').value)
        self.detection_target_classes = list(self.declare_parameter('detection_target_classes', ['person']).value)
        # Process one frame out of every N received (1 = every frame). Cheaper
        # than a rate cap that sleeps until the period is over: the selected
        # frame is used the moment it arrives, so no extra wait is added on
        # top of transport+inference (a sleep-based 15 Hz cap added ~17 ms of
        # latency on average). Skipped frames are intentional, not 'dropped'.
        # The fusion clock is the LiDAR (10 Hz); stride 2 of a 30 Hz camera
        # = 15 Hz of detections.
        self.detection_frame_stride = max(1, int(self.declare_parameter('detection_frame_stride', 2).value))
        # LiDAR-triggered mode (detection_trigger_topic set): no free-running
        # inference at all. Every trigger message (a tiny std_msgs/Header
        # published by preprocessing_node at each LiDAR scan arrival, stamp =
        # scan stamp) requests ONE detection per camera on the camera frame
        # whose stamp is closest to that scan (+ detection_trigger_stamp_offset_sec).
        # Camera messages are only buffered compressed; only the SELECTED
        # frame is decoded and inferred, so no frame is decoded/inferred just
        # to be discarded. Empty = free-running with detection_frame_stride.
        self.detection_trigger_topic = str(self.declare_parameter('detection_trigger_topic', '').value)
        self.detection_trigger_stamp_offset_sec = float(self.declare_parameter('detection_trigger_stamp_offset_sec', 0.0).value)
        # 'latest': infer on the newest frame already received when the trigger arrives (no wait); 'nearest_wait': wait for a
        # frame at/after the scan stamp and use the closest one (aligned in time, ~35 ms later).
        self.detection_trigger_policy = str(self.declare_parameter('detection_trigger_policy', 'nearest_wait').value)
        self.detection_trigger_max_wait_sec = float(self.declare_parameter('detection_trigger_max_wait_sec', 0.15).value)
        # JPEG decode at 1/N size (1, 2, 4, 8; libjpeg DCT-domain scaling, much
        # cheaper than full decode + resize). The detector runs at
        # detection_inference_size (640) anyway, so 2 gives it 640x360 for
        # free. Ignored (full size) while enable_semantic needs the full frame.
        # Published boxes are scaled back to full-resolution pixel coordinates.
        self.detection_decode_reduced = int(self.declare_parameter('detection_decode_reduced', 2).value)
        self.detection_score_threshold = float(self.declare_parameter('detection_score_threshold', 0.5).value)
        self.detection_inference_size = int(self.declare_parameter('detection_inference_size', 640).value)
        self.detection_enable_tracking = bool(self.declare_parameter('detection_enable_tracking', True).value)
        self.detection_tracker = str(self.declare_parameter('detection_tracker', 'bytetrack.yaml').value)
        self.detections_topic_template = str(self.declare_parameter('detections_topic_template', 'yolo_semantic_seg/{camera}/detections').value)
        self.detection_image_topic_template = str(self.declare_parameter('detection_image_topic_template', 'yolo_semantic_seg/{camera}/detection_image').value)
        self.detection_time_topic_template = str(self.declare_parameter('detection_time_topic_template', 'yolo_semantic_seg/{camera}/detection_time').value)

        missing = [
            name for name, val in [
                ('camera_names', self.camera_names),
                ('detected_image_topic_template', self.detected_image_topic_template),
                ('semantic_mask_topic_template', self.semantic_mask_topic_template),
                ('segmentation_mask_topic_template', self.segmentation_mask_topic_template),
                ('semantic_color_topic_template', self.semantic_color_topic_template),
                ('inference_time_topic_template', self.inference_time_topic_template),
                ('weights_path', self.weights_path),
                ('class_names_path', self.class_names_path),
                ('segmentation_classes', self.segmentation_classes),
            ] if not val
        ]
        if missing:
            self.get_logger().error('Missing required parameters in YAML: ' + ', '.join(missing))
            raise RuntimeError('YOLO semantic seg parameter initialization failed')

        # share-relative (e.g. 'weights/yolo26s-sem.pt', 'cfg/cityscapes.names')
        # — see yolo_seg_track_detector.py's own comment on why this is
        # resolved against the package SHARE dir, not this file's own
        # install dir.
        pkg_share = get_package_share_directory('sins')
        self.weights_path = os.path.join(pkg_share, self.weights_path)
        self.class_names_path = os.path.join(pkg_share, self.class_names_path)
        self.use_half = (device == "cuda")

        with open(self.class_names_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.segmentation_class_ids = [
            i for i, name in enumerate(self.class_names) if name in self.segmentation_classes
        ]
        unknown = set(self.segmentation_classes) - set(self.class_names)
        if unknown:
            self.get_logger().warning(f"segmentation_classes not found in {self.class_names_path}: {sorted(unknown)}")
        if not self.segmentation_class_ids:
            self.get_logger().error('None of segmentation_classes matched class_names_path — segmentation_mask will always be empty.')
            raise RuntimeError('YOLO semantic seg segmentation_classes resolved to an empty set')

        # Color LUT for semantic_color_topic_template, indexed by class id —
        # built from _CITYSCAPES_COLORS_RGB (converted to BGR), one entry per
        # class_names_path line. class_names_path isn't guaranteed to be
        # exactly the 19-class Cityscapes list forever (a future checkpoint
        # could use a different taxonomy/count), so this pads with a neutral
        # gray for any class beyond the known palette rather than crashing.
        palette_bgr = [tuple(reversed(c)) for c in _CITYSCAPES_COLORS_RGB]
        if len(self.class_names) > len(palette_bgr):
            self.get_logger().warning(
                f"class_names_path has {len(self.class_names)} classes but only "
                f"{len(palette_bgr)} known colors — classes beyond that will render gray."
            )
        self.color_lut = np.array(
            [palette_bgr[i] if i < len(palette_bgr) else (128, 128, 128)
             for i in range(len(self.class_names))],
            dtype=np.uint8,
        )
        self.get_logger().info(
            "semantic_color legend: " + ", ".join(
                f"{name}={tuple(int(v) for v in self.color_lut[i])}"
                for i, name in enumerate(self.class_names)
            )
        )

        # segmentation_classes -> 0/255 LUT, applied on the model's own
        # device to the class map before it is copied to the CPU (one cheap
        # gather instead of a CPU np.isin over ~0.9M pixels per frame).
        seg_lut = np.zeros(max(256, len(self.class_names)), dtype=np.uint8)
        seg_lut[self.segmentation_class_ids] = 255
        self.seg_lut_t = torch.from_numpy(seg_lut).to(device)
        self._geo_cache = {}

        if self.enable_detection:
            self.detection_weights_path = os.path.join(pkg_share, self.detection_weights_path)
            det_names_path = os.path.join(pkg_share, self.detection_class_names_path)
            with open(det_names_path, 'r') as f:
                self.det_label_names = [line.strip() for line in f.readlines()]
            self.det_target_class_ids = [
                i for i, n in enumerate(self.det_label_names) if n in self.detection_target_classes]
            if not self.det_target_class_ids:
                raise RuntimeError('detection_target_classes resolved to an empty set')

        self.cameras = {}
        for name in self.camera_names:
            ctx = _CameraContext(name)

            if self.enable_semantic:
                ctx.model = YOLO(self.weights_path)
                if ctx.model.task != 'semantic':
                    self.get_logger().warning(
                        f"weights_path '{self.weights_path}' is a '{ctx.model.task}' model, not 'semantic' — "
                        "this node expects result.semantic_mask, which only a semantic-segmentation checkpoint "
                        "(e.g. a YOLO26*-sem one) provides. Point weights_path at one of those."
                    )

            image_topic = self.image_topic_template.format(camera=name)
            # Same image_transport-mangled compressed-topic subscription as
            # yolov11_detector.py/yolo_seg_track_detector.py — see either
            # file's comment for why (no rclpy-side transport-hint helper).
            self.create_subscription(
                CompressedImage, image_topic + '/compressed',
                self._make_image_callback(ctx), self.queue_size)

            if self.enable_detection and self.detection_trigger_topic:
                self.create_subscription(
                    Header, self.detection_trigger_topic, self._make_trigger_callback(ctx), 10)

            detected_image_topic = self.detected_image_topic_template.format(camera=name)
            ctx.img_pub = self.create_publisher(Image, detected_image_topic, self.queue_size)
            ctx.img_compressed_pub = self.create_publisher(
                CompressedImage, detected_image_topic + '/compressed', self.queue_size)
            # mono8, not 32SC1 like yolo_seg_track_detector.py's instance
            # mask: only 19 classes, easily fits a byte, and mono8 is what
            # rviz's own Image display expects for a directly-viewable
            # grayscale label map (raw only — JPEG would corrupt exact class
            # index values via lossy compression, same reasoning as that
            # file's own instance_masks topic).
            ctx.semantic_mask_pub = self.create_publisher(
                Image, self.semantic_mask_topic_template.format(camera=name), self.queue_size)
            # mono8, 0 or 255 — the binary "is this pixel one of
            # segmentation_classes" answer, derived from semantic_mask
            # (isin(segmentation_class_ids)) so it's redundant with it, but
            # publishing it separately means a downstream consumer doesn't
            # need to know about Cityscapes indices at all, just "0/255".
            ctx.segmentation_mask_pub = self.create_publisher(
                Image, self.segmentation_mask_topic_template.format(camera=name), self.queue_size)
            # Full 19-class colorized overlay (color_lut above) — bgr8, +
            # '/compressed' JPEG counterpart same as detected_image, since
            # this one's for eyeballing, not exact per-pixel class recovery
            # (semantic_mask is the source of truth for that).
            semantic_color_topic = self.semantic_color_topic_template.format(camera=name)
            ctx.semantic_color_pub = self.create_publisher(Image, semantic_color_topic, self.queue_size)
            ctx.semantic_color_compressed_pub = self.create_publisher(
                CompressedImage, semantic_color_topic + '/compressed', self.queue_size)
            ctx.time_pub = self.create_publisher(
                Float64, self.inference_time_topic_template.format(camera=name), 1)

            if self.enable_detection:
                # Own model instance per camera: the tracker's state lives
                # on it, and sharing one across cameras would let ByteTrack
                # match a box from one camera against the other's frames.
                ctx.det_model = YOLO(self.detection_weights_path)
                if ctx.det_model.task != 'detect':
                    self.get_logger().warning(
                        f"detection_weights_path is a '{ctx.det_model.task}' model, expected 'detect'.")
                ctx.det_pub = self.create_publisher(
                    Detection2DArray, self.detections_topic_template.format(camera=name), self.queue_size)
                det_img_topic = self.detection_image_topic_template.format(camera=name)
                ctx.det_img_pub = self.create_publisher(Image, det_img_topic, self.queue_size)
                ctx.det_img_compressed_pub = self.create_publisher(
                    CompressedImage, det_img_topic + '/compressed', self.queue_size)
                ctx.det_time_pub = self.create_publisher(
                    Float64, self.detection_time_topic_template.format(camera=name), 1)
                # Warm-up: the first inference pays CUDA/cuDNN initialization (~2 s measured on the
                # benchmark bags), which used to delay every class by that much at start-up. predict(),
                # not track(), so ByteTrack's state stays clean.
                t_warm = time.time()
                ctx.det_model.predict(
                    np.zeros((720, 1280, 3), dtype=np.uint8), classes=self.det_target_class_ids,
                    conf=self.detection_score_threshold, imgsz=self.detection_inference_size, device=device,
                    quantize=16 if self.use_half else None, verbose=False)
                self.get_logger().info(f"[{name}] detection model warm-up: {(time.time() - t_warm) * 1000:.0f} ms")

            self.cameras[name] = ctx

        self.get_logger().info(
            f"YOLO semantic seg configured: device={device}, half={self.use_half}, "
            f"cameras={self.camera_names}, weights={self.weights_path}, "
            f"semantic={self.enable_semantic}, detection={self.enable_detection}, segmentation_classes={self.segmentation_classes} (ids={self.segmentation_class_ids}), "
            f"inference_size={self.inference_size}"
        )

        # One worker thread per camera — see yolo_seg_track_detector.py's
        # own file header for the full "why not a timer / why not a plain
        # queue" reasoning this mirrors exactly. Plus one more thread per
        # camera for the semantic_color debug overlay (see file header) —
        # deliberately a THIRD thread, not folded into the inference worker.
        self._stop_event = threading.Event()
        for ctx in self.cameras.values():
            if self.enable_semantic:
                ctx.thread = threading.Thread(
                    target=self._worker_loop, args=(ctx,), name=f'semantic_seg_worker_{ctx.name}', daemon=True)
                ctx.thread.start()
                ctx.color_thread = threading.Thread(
                    target=self._color_worker_loop, args=(ctx,), name=f'semantic_color_worker_{ctx.name}', daemon=True)
                ctx.color_thread.start()
            if self.enable_detection:
                ctx.det_thread = threading.Thread(
                    target=self._det_worker_loop, args=(ctx,), name=f'detection_worker_{ctx.name}', daemon=True)
                ctx.det_thread.start()

    def destroy_node(self):
        self._stop_event.set()
        for ctx in self.cameras.values():
            ctx.new_frame_event.set()
            ctx.color_new_job_event.set()
            ctx.det_event.set()
            if ctx.det_thread is not None:
                ctx.det_thread.join(timeout=2.0)
            if ctx.thread is not None:
                ctx.thread.join(timeout=2.0)
            if ctx.color_thread is not None:
                ctx.color_thread.join(timeout=2.0)
        super().destroy_node()

    def _det_scale(self):
        return 1 if self.enable_semantic else max(1, self.detection_decode_reduced)

    def _decode(self, msg, scale):
        flag = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2,
                4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}.get(scale, cv2.IMREAD_COLOR)
        decoded = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), flag)
        if decoded is None:
            raise ValueError(f"cv2.imdecode failed (format='{msg.format}', {len(msg.data)} bytes)")
        return np.ascontiguousarray(decoded, dtype=np.uint8)

    def _submit_detection(self, ctx, msg, img, scale):
        """Hand one frame to the detection thread (freshest-frame slot)."""
        # img None = still compressed: the detection thread decodes it itself
        # (cv2 releases the GIL there; decoding in the ROS callback thread
        # instead measured 5 ms, mostly waiting for the GIL held by the
        # inference thread, vs ~1.3 ms of real decode work).
        with ctx.det_lock:
            if ctx.det_received == 0:
                ctx.det_started = time.time()
            ctx.det_received += 1
            if ctx.det_pending_img is not None or ctx.det_pending_msg is not None:
                ctx.det_dropped += 1
            ctx.det_pending_img = img
            ctx.det_pending_msg = msg if img is None else None
            ctx.det_pending_header = msg.header
            ctx.det_pending_recv = time.time()
            ctx.det_pending_scale = scale
            ctx.det_event.set()

    @staticmethod
    def _stamp_sec(header):
        return header.stamp.sec + header.stamp.nanosec * 1e-9

    def _det_input(self, ctx, msg, full_img):
        """Camera frame -> detection thread. Free-running: every Nth frame.
        Triggered: buffer compressed frames; when a pending trigger target has
        a frame at/after it, pick the closest frame and decode only that one."""
        scale = 1 if full_img is not None else self._det_scale()
        if not self.detection_trigger_topic:
            ctx.det_seen += 1
            if ctx.det_seen % self.detection_frame_stride == 0:
                self._submit_detection(ctx, msg, full_img, scale)
            return
        stamp = self._stamp_sec(msg.header)
        ctx.det_ring.append((stamp, msg, full_img))
        self._resolve_targets(ctx, scale)

    def _resolve_targets(self, ctx, scale):
        while ctx.det_targets:
            target = ctx.det_targets[0]
            newest = ctx.det_ring[-1][0] if ctx.det_ring else None
            if self.detection_trigger_policy == 'latest':
                # Do not wait for a frame at/after the scan stamp: camera frames arrive ~35 ms after their stamp, so
                # waiting for one costs that plus half a frame period. Use the newest frame ALREADY here (about 35-70 ms
                # older than the scan): the detection is ready by the time the scan's own clusters are, and the fusion
                # inflates the 2D box to cover the object's motion over that offset.
                ctx.det_targets.popleft()
                if newest is None or target - newest > self.detection_trigger_max_wait_sec:
                    ctx.trig_missed += 1
                    continue
                ctx.trig_offsets.append(newest - target)
                last = ctx.det_ring[-1]
                self._submit_detection(ctx, last[1], last[2], 1 if last[2] is not None else scale)
                continue
            if newest is not None and newest >= target:
                best = min(ctx.det_ring, key=lambda r: abs(r[0] - target))
                ctx.det_targets.popleft()
                ctx.trig_offsets.append(best[0] - target)
                self._submit_detection(ctx, best[1], best[2], 1 if best[2] is not None else scale)
            elif newest is not None and (newest - target) < -self.detection_trigger_max_wait_sec:
                ctx.det_targets.popleft()
                ctx.trig_missed += 1
            else:
                break

    def _make_trigger_callback(self, ctx):
        def _cb(msg):
            try:
                target = self._stamp_sec(msg) + self.detection_trigger_stamp_offset_sec
                if ctx.det_ring and target < ctx.det_ring[0][0] - 1.0:
                    ctx.det_targets.clear()  # clock jumped back (bag loop)
                    ctx.det_ring.clear()
                ctx.det_targets.append(target)
                self._resolve_targets(ctx, self._det_scale())
            except Exception as e:
                self.get_logger().error(f"[{ctx.name}] Trigger callback error: {type(e).__name__}: {e}")
        return _cb

    def _make_image_callback(self, ctx):
        def _cb(msg):
            try:
                img = None
                if self.enable_semantic:
                    t_cb0 = time.time()
                    img = self._decode(msg, 1)
                    ctx.decode_times.append(time.time() - t_cb0)
                    with ctx.lock:
                        ctx.frames_received += 1
                        # A pending frame still sitting here means the worker
                        # hasn't picked up the PREVIOUS one yet — it's about to
                        # be overwritten and lost, by design (freshest-frame
                        # policy, see file header) rather than queued.
                        if ctx.pending_img is not None:
                            ctx.frames_dropped += 1
                        ctx.pending_img = img
                        ctx.pending_header = msg.header
                        ctx.new_frame_event.set()

                if self.enable_detection:
                    self._det_input(ctx, msg, img)

                if not ctx.first_logged:
                    self.get_logger().info(f"[{ctx.name}] First image received (format={msg.format})")
                    ctx.first_logged = True

            except Exception as e:
                self.get_logger().error(f"[{ctx.name}] Image callback error: {type(e).__name__}: {e}")
        return _cb

    def _worker_loop(self, ctx):
        while not self._stop_event.is_set():
            if not ctx.new_frame_event.wait(timeout=1.0):
                continue
            with ctx.lock:
                img = ctx.pending_img
                header = ctx.pending_header
                ctx.pending_img = None
                ctx.new_frame_event.clear()
            if img is None or self._stop_event.is_set():
                continue
            self._process_frame(ctx, img, header)

    def _color_worker_loop(self, ctx):
        while not self._stop_event.is_set():
            if not ctx.color_new_job_event.wait(timeout=1.0):
                continue
            with ctx.color_lock:
                job = (ctx.pending_color_img, ctx.pending_color_class_map,
                       ctx.pending_color_seg_mask, ctx.pending_color_header)
                ctx.pending_color_img = None
                ctx.pending_color_class_map = None
                ctx.pending_color_seg_mask = None
                ctx.color_new_job_event.clear()
            if job[0] is None or job[1] is None or self._stop_event.is_set():
                continue
            self._process_debug_overlays(ctx, *job)

    @staticmethod
    def _overlays_wanted(ctx):
        return any(p.get_subscription_count() > 0 for p in (
            ctx.img_pub, ctx.img_compressed_pub, ctx.semantic_color_pub, ctx.semantic_color_compressed_pub))

    def _process_debug_overlays(self, ctx, img_bgr, class_map, seg_mask, header):
        """Everything that is only for eyeballing — the segmentation_classes
        tint (detected_image + JPEG) and the full 19-class semantic_color
        (+ JPEG). Runs off the inference path (see file header)."""
        start_time = time.time()
        try:
            n = lambda pub: pub.get_subscription_count()
            if n(ctx.img_pub) or n(ctx.img_compressed_pub):
                selected = seg_mask.astype(bool)
                annotated = img_bgr.copy()
                color = np.array(self.overlay_color_bgr, dtype=np.float32)
                annotated[selected] = (annotated[selected].astype(np.float32) * (1.0 - self.overlay_alpha)
                                       + color * self.overlay_alpha).astype(np.uint8)
                if n(ctx.img_pub):
                    ctx.img_pub.publish(_to_image_msg(annotated, "bgr8", header))
                if n(ctx.img_compressed_pub):
                    self._publish_jpeg(ctx.img_compressed_pub, annotated, header)

            if n(ctx.semantic_color_pub) or n(ctx.semantic_color_compressed_pub):
                class_colors = self.color_lut[class_map]  # HxWx3 BGR, fancy-indexed
                semantic_color = cv2.addWeighted(
                    img_bgr, 1.0 - self.semantic_color_alpha, class_colors, self.semantic_color_alpha, 0.0)
                if n(ctx.semantic_color_pub):
                    ctx.semantic_color_pub.publish(_to_image_msg(semantic_color, "bgr8", header))
                if n(ctx.semantic_color_compressed_pub):
                    self._publish_jpeg(ctx.semantic_color_compressed_pub, semantic_color, header)
        except Exception:
            import traceback
            self.get_logger().error(f"[{ctx.name}] Error in debug overlays:\n{traceback.format_exc()}")
        elapsed = time.time() - start_time
        ctx.color_times.append(elapsed)
        self._maybe_log_color_stats(ctx)

    def _maybe_log_color_stats(self, ctx):
        now = time.time()
        if now - ctx.color_last_stats_log < self.stats_log_period_sec:
            return
        with ctx.color_lock:
            received, dropped = ctx.color_jobs_received, ctx.color_jobs_dropped
        times = list(ctx.color_times)
        ctx.color_last_stats_log = now
        if not times:
            return
        avg_ms = 1000.0 * sum(times) / len(times)
        max_ms = 1000.0 * max(times)
        drop_rate = (100.0 * dropped / received) if received else 0.0
        self.get_logger().info(
            f"[{ctx.name}] debug overlays (own thread): avg={avg_ms:.1f}ms max={max_ms:.1f}ms | "
            f"jobs received={received} dropped={dropped} ({drop_rate:.1f}%)"
        )

    def _maybe_log_stats(self, ctx):
        now = time.time()
        if now - ctx.last_stats_log < self.stats_log_period_sec:
            return
        with ctx.lock:
            received, dropped = ctx.frames_received, ctx.frames_dropped
        times = list(ctx.inference_times)
        ctx.last_stats_log = now
        if not times:
            return
        avg_ms = 1000.0 * sum(times) / len(times)
        max_ms = 1000.0 * max(times)
        hz = 1.0 / (sum(times) / len(times)) if avg_ms > 0 else 0.0
        drop_rate = (100.0 * dropped / received) if received else 0.0
        self.get_logger().info(
            f"[{ctx.name}] inference: avg={avg_ms:.1f}ms max={max_ms:.1f}ms (~{hz:.1f}Hz) | "
            f"predict={_avg_ms(ctx.predict_times):.1f} post={_avg_ms(ctx.post_times):.1f} "
            f"publish={_avg_ms(ctx.publish_times):.1f}ms | "
            f"predict split (ultralytics) pre={_avg_speed(ctx.speed_times, 'preprocess'):.1f} "
            f"net={_avg_speed(ctx.speed_times, 'inference'):.1f} "
            f"postproc={_avg_speed(ctx.speed_times, 'postprocess'):.1f}ms | "
            f"frames received={received} dropped={dropped} ({drop_rate:.1f}%)"
        )

    def _publish_jpeg(self, pub, img_bgr, header):
        ok, jpeg_buf = cv2.imencode(
            '.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            self.get_logger().warning("JPEG encode failed")
            return
        msg = CompressedImage()
        msg.header = header
        msg.format = 'jpeg'
        msg.data = jpeg_buf.tobytes()
        pub.publish(msg)

    def _process_frame(self, ctx, img_bgr, header):
        start_time = time.time()
        try:
            t0 = time.time()
            class_map, segmentation_mask, t_predict = self.infer(ctx, img_bgr)
            t1 = time.time()

            ctx.semantic_mask_pub.publish(_to_image_msg(class_map, "mono8", header))
            ctx.segmentation_mask_pub.publish(_to_image_msg(segmentation_mask, "mono8", header))
            t2 = time.time()
            ctx.predict_times.append(t_predict - t0)
            ctx.post_times.append(t1 - t_predict)
            ctx.publish_times.append(t2 - t1)

            # Debug overlays are handed to their own thread (freshest-job
            # protocol, same as the image callback -> this worker) — and only
            # when something subscribes to them: rendering + publishing them
            # cost roughly a full core per camera.
            if self._overlays_wanted(ctx):
                with ctx.color_lock:
                    ctx.color_jobs_received += 1
                    if ctx.pending_color_class_map is not None:
                        ctx.color_jobs_dropped += 1
                    ctx.pending_color_img = img_bgr
                    ctx.pending_color_class_map = class_map
                    ctx.pending_color_seg_mask = segmentation_mask
                    ctx.pending_color_header = header
                    ctx.color_new_job_event.set()
        except Exception:
            import traceback
            self.get_logger().error(f"[{ctx.name}] Error in inference:\n{traceback.format_exc()}")
        elapsed = time.time() - start_time
        time_msg = Float64()
        time_msg.data = float(elapsed)
        ctx.time_pub.publish(time_msg)
        ctx.inference_times.append(elapsed)
        self._maybe_log_stats(ctx)

    def infer(self, ctx, img_bgr):
        """Run semantic segmentation on one BGR frame. Returns (class_map
        uint8 HxW Cityscapes index per pixel, segmentation_mask uint8 HxW
        0/255 for segmentation_classes, timestamp taken right after
        model.predict returned). The 0/255 mask is built on the model's
        device with a LUT before the CPU copy."""
        h, w = img_bgr.shape[:2]
        if self.gpu_preprocess:
            geo = self._geometry(h, w)
            t = torch.from_numpy(img_bgr).to(device, non_blocking=True)
            t = t.permute(2, 0, 1)[[2, 1, 0]][None].float().div_(255.0)
            t = F.interpolate(t, size=geo['net_hw'], mode='bilinear', align_corners=False)
            source = t
        else:
            geo = None
            source = img_bgr
        results = ctx.model.predict(
            source,
            imgsz=self.inference_size,
            device=device,
            # Same quantize=16 (fp16) convention as yolo_seg_track_detector.py
            # — see that file's own comment on why not 'half=True'.
            quantize=16 if self.use_half else None,
            verbose=False,
        )
        t_predict = time.time()
        ctx.speed_times.append(results[0].speed)
        # CPU path: semantic_mask.data comes back already resized to
        # img_bgr's resolution. GPU path: it is at the network input size,
        # so it is expanded here (nearest, row/column gather).
        sem = results[0].semantic_mask.data
        if geo is not None:
            sem = sem[geo['rows']][:, geo['cols']]
        mask_t = self.seg_lut_t[sem.long()]
        class_map = sem.to(torch.uint8).cpu().numpy()
        segmentation_mask = mask_t.cpu().numpy()
        return class_map, segmentation_mask, t_predict

    def _geometry(self, h, w):
        key = (h, w)
        geo = self._geo_cache.get(key)
        if geo is None:
            scale = self.inference_size / max(h, w)
            nh = int(np.ceil(h * scale / 32.0) * 32)
            nw = int(np.ceil(w * scale / 32.0) * 32)
            rows = (torch.arange(h, device=device) * nh // h).long()
            cols = (torch.arange(w, device=device) * nw // w).long()
            geo = {'net_hw': (nh, nw), 'rows': rows, 'cols': cols}
            self._geo_cache[key] = geo
        return geo

    # ---- detection + tracking thread ---------------------------------

    def _det_worker_loop(self, ctx):
        while not self._stop_event.is_set():
            if not ctx.det_event.wait(timeout=1.0):
                continue
            with ctx.det_lock:
                img = ctx.det_pending_img
                pmsg = ctx.det_pending_msg
                header = ctx.det_pending_header
                recv = ctx.det_pending_recv
                scale = ctx.det_pending_scale
                ctx.det_pending_img = None
                ctx.det_pending_msg = None
                ctx.det_event.clear()
            if (img is None and pmsg is None) or self._stop_event.is_set():
                continue
            ctx.det_wait_times.append(time.time() - recv)
            if img is None:
                t_dec = time.time()
                try:
                    img = self._decode(pmsg, scale)
                except Exception as e:
                    self.get_logger().error(f"[{ctx.name}] detection decode error: {e}")
                    continue
                ctx.decode_times.append(time.time() - t_dec)
            self._process_detection(ctx, img, header, scale)

    def _process_detection(self, ctx, img_bgr, header, scale=1):
        start_time = time.time()
        try:
            kwargs = dict(
                classes=self.det_target_class_ids,
                conf=self.detection_score_threshold,
                imgsz=self.detection_inference_size,
                device=device,
                quantize=16 if self.use_half else None,
                verbose=False,
            )
            if self.detection_enable_tracking:
                results = ctx.det_model.track(img_bgr, persist=True, tracker=self.detection_tracker, **kwargs)
            else:
                results = ctx.det_model.predict(img_bgr, **kwargs)
            result = results[0]

            det_array = Detection2DArray()
            det_array.header = header
            boxes = result.boxes
            # Tracking mode: a box with no confirmed track id yet (first
            # frames before ByteTrack assigns one) is skipped — the id is
            # what consumers key on. Without tracking there is no stable
            # identity at all, so the id published here is NEGATIVE
            # (-1, -2, ...) rather than this frame's own plain index
            # (0, 1, ...): still unique WITHIN this one frame (downstream
            # per-message grouping, e.g. preprocessing_node/
            # dbscan_detector_node's onLeaves(), still needs that), but a
            # consumer that treats this as a cross-frame PERSISTENT
            # identity (tracker_node's decaySemanticEvidence(), gating a
            # shrink on whether the YOLO 2D track id matches) sees a
            # negative value and correctly refuses to trust it as a known
            # id — a positive-looking per-frame index would otherwise be
            # silently indistinguishable from a real ByteTrack id to code
            # that never learns detection_enable_tracking's own value.
            if boxes is not None and len(boxes) > 0 and not (self.detection_enable_tracking and boxes.id is None):
                ids = (boxes.id.cpu().numpy().astype(np.int64) if self.detection_enable_tracking
                       else -(np.arange(len(boxes), dtype=np.int64) + 1))
                cls_ids = boxes.cls.cpu().numpy().astype(np.int64)
                confs = boxes.conf.cpu().numpy()
                xyxy = boxes.xyxy.cpu().numpy() * float(scale)
                for i in range(len(ids)):
                    det = Detection2D()
                    det.header = header
                    det.id = str(int(ids[i]))
                    x1, y1, x2, y2 = xyxy[i]
                    bbox = BoundingBox2D()
                    bbox.center.position.x = float((x1 + x2) / 2.0)
                    bbox.center.position.y = float((y1 + y2) / 2.0)
                    bbox.size_x = float(x2 - x1)
                    bbox.size_y = float(y2 - y1)
                    det.bbox = bbox
                    hyp = ObjectHypothesisWithPose()
                    hyp.hypothesis.class_id = self.det_label_names[cls_ids[i]]
                    hyp.hypothesis.score = float(confs[i])
                    det.results.append(hyp)
                    det_array.detections.append(det)
            ctx.det_pub.publish(det_array)
            stamp_s = header.stamp.sec + header.stamp.nanosec * 1e-9
            now_s = self.get_clock().now().nanoseconds * 1e-9
            if stamp_s > 0.0:
                ctx.det_age_times.append(now_s - stamp_s)

            # Debug image (boxes + track ids), published from this same
            # thread only when someone is listening.
            if (ctx.det_img_pub.get_subscription_count() > 0
                    or ctx.det_img_compressed_pub.get_subscription_count() > 0):
                annotated = result.plot()
                if ctx.det_img_pub.get_subscription_count() > 0:
                    ctx.det_img_pub.publish(_to_image_msg(annotated, "bgr8", header))
                if ctx.det_img_compressed_pub.get_subscription_count() > 0:
                    self._publish_jpeg(ctx.det_img_compressed_pub, annotated, header)
        except Exception:
            import traceback
            self.get_logger().error(f"[{ctx.name}] Error in detection:\n{traceback.format_exc()}")
        elapsed = time.time() - start_time
        time_msg = Float64()
        time_msg.data = float(elapsed)
        ctx.det_time_pub.publish(time_msg)
        ctx.det_times.append(elapsed)
        self._maybe_log_det_stats(ctx)

    def _maybe_log_det_stats(self, ctx):
        now = time.time()
        if now - ctx.det_last_stats_log < self.stats_log_period_sec:
            return
        with ctx.det_lock:
            received, dropped = ctx.det_received, ctx.det_dropped
        ctx.det_last_stats_log = now
        if not ctx.det_times:
            return
        drop_rate = (100.0 * dropped / received) if received else 0.0
        self.get_logger().info(
            f"[{ctx.name}] detection+track: infer avg={_avg_ms(ctx.det_times):.1f}ms "
            f"max={1000.0 * max(ctx.det_times):.1f}ms | decode={_avg_ms(ctx.decode_times):.1f}ms "
            f"queue-wait={_avg_ms(ctx.det_wait_times):.1f}ms | stamp->published: "
            f"avg={_avg_ms(ctx.det_age_times):.1f}ms p95={_p95_ms(ctx.det_age_times):.1f}ms | "
            f"rate={(received - dropped) / max(now - ctx.det_started, 1e-6):.1f}Hz "
            + (f"(trigger: frame-target offset avg={1000.0 * (sum(ctx.trig_offsets) / len(ctx.trig_offsets)) if ctx.trig_offsets else 0.0:.1f}ms "
               f"|max|={1000.0 * max((abs(o) for o in ctx.trig_offsets), default=0.0):.1f}ms, missed={ctx.trig_missed}) "
               if self.detection_trigger_topic else f"(stride {self.detection_frame_stride}) ")
            + f"received={received} dropped={dropped} ({drop_rate:.1f}%)"
        )
