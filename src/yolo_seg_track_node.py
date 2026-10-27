#!/usr/bin/env python3

"""Thin ROS2 entrypoint — this file is the "node" (lives in src/ alongside
every other node's source, installs to lib/sins/ like their
executables); yolo_seg_track_detector.py itself is the "script" holding the
actual detection/tracking implementation and lives under scripts/ (share/
sins/scripts/ once installed), so it's resolved via
get_package_share_directory rather than sys.path-of-this-file like before.
Explicit fatal log instead of a bare traceback if torch/ultralytics aren't
importable."""

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
        from yolo_seg_track_detector import YoloSegTrackDetector
    except Exception:
        get_logger("yolo_seg_track_node").fatal(
            f"Import failed — check torch/ultralytics installation:\n{traceback.format_exc()}"
        )
        raise SystemExit(1)

    rclpy.init()
    try:
        node = YoloSegTrackDetector()
    except Exception:
        get_logger("yolo_seg_track_node").fatal(
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
