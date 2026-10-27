#!/usr/bin/env python3
"""
extract_obstacles.py
---------------------
One-shot ROS2 subscriber that records every message on a jo_msgs/ObstacleArray
topic into a flat JSON time series, for offline evaluation against a manually
annotated ground truth (see eval_tracking.py and GT_SCHEMA.md in this folder).

Works unmodified against EITHER jo_detector (v1) or sins's
tracker_node: both publish to the same topic name and message type
(/perception/tracked_dynamic_obstacles, jo_msgs/ObstacleArray) — this
script does not care which pipeline produced the data, which is the whole
point: run it once per pipeline, against the same bag, and diff the two
resulting JSON files with the same eval_tracking.py call.

Usage (inside the container, with the pipeline under test + a bag already
running — rviz not required):

    python3 extract_obstacles.py --out v2_run1.json --duration 90
    python3 extract_obstacles.py --out v1_run1.json --duration 90 \\
        --topic /perception/tracked_dynamic_obstacles

Output JSON shape — a list of frames, each:
    {"t": <float, header stamp in seconds>,
     "obstacles": [
         {"track_id": <int>, "status": "dynamic"|"potentially_dynamic"|"static"|"tracked",
          "x": <float>, "y": <float>, "z": <float>}
         , ...
     ]}

One JSON frame per received ObstacleArray message — NOT resampled/binned, so
two runs recorded on the same bag with --clock will have near-identical (but
not bit-identical) timestamps; eval_tracking.py resamples both series onto a
common frame grid before matching (see its own --frame-period).
"""
import argparse
import json
import time

import rclpy
from rclpy.node import Node
from jo_msgs.msg import ObstacleArray, Obstacle

STATUS_NAMES = {
    Obstacle.STATUS_STATIC: "static",
    Obstacle.STATUS_DYNAMIC: "dynamic",
    Obstacle.STATUS_POTENTIALLY_DYNAMIC: "potentially_dynamic",
    Obstacle.STATUS_TRACKED: "tracked",
}


class Extractor(Node):
    def __init__(self, topic):
        super().__init__('obstacle_extractor')
        self.frames = []
        self.create_subscription(ObstacleArray, topic, self.cb, 10)
        self.get_logger().info(f"subscribed to {topic}")

    def cb(self, msg: ObstacleArray):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        obstacles = []
        for o in msg.obstacles:
            obstacles.append({
                "track_id": int(o.track_id),
                "status": STATUS_NAMES.get(o.status, f"unknown_{o.status}"),
                "x": o.pose.position.x,
                "y": o.pose.position.y,
                "z": o.pose.position.z,
            })
        self.frames.append({"t": t, "obstacles": obstacles})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topic', default='/perception/tracked_dynamic_obstacles')
    ap.add_argument('--out', required=True)
    ap.add_argument('--duration', type=float, default=90.0,
                     help="wall-clock seconds to record (default covers the validation_lab bag's own ~99s length)")
    args = ap.parse_args()

    rclpy.init()
    node = Extractor(args.topic)
    t0 = time.time()
    while rclpy.ok() and time.time() - t0 < args.duration:
        rclpy.spin_once(node, timeout_sec=0.2)
    with open(args.out, 'w') as f:
        json.dump(node.frames, f)
    node.get_logger().info(f"wrote {len(node.frames)} frames to {args.out}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
