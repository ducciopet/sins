#!/usr/bin/env python3

"""Thin ROS2 entrypoint for yolo_semantic_seg_detector.py — same NODE/SCRIPT
split as yolo_seg_track_node.py/yolo_seg_track_detector.py, see either
file's own header for why."""

import os
import sys
import traceback

import rclpy
from rclpy.logging import get_logger
from ament_index_python.packages import get_package_share_directory


def main():
    _pkg_share = get_package_share_directory("sins")
    _scripts_dir = os.path.join(_pkg_share, "scripts")
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)

    try:
        from yolo_semantic_seg_detector import YoloSemanticSegDetector
    except Exception:
        get_logger("yolo_semantic_seg_node").fatal(
            f"Import failed — check torch/ultralytics installation:\n{traceback.format_exc()}"
        )
        raise SystemExit(1)

    rclpy.init()
    try:
        node = YoloSemanticSegDetector()
    except Exception:
        get_logger("yolo_semantic_seg_node").fatal(
            f"Node initialization failed:\n{traceback.format_exc()}"
        )
        rclpy.shutdown()
        raise SystemExit(1)

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
