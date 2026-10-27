# Dynamic/potentially-dynamic object evaluation

Evaluation harness for comparing jo_detector (v1) and
sins's own dynamic-object classification against hand-labeled
ground truth and, eventually, against external baselines (AB3DMOT, a
SemanticKITTI-MOS method) on the same bags. Built for the paper evaluating
the tracker's dynamic/potentially_dynamic classification — not the ground
segmentation work in `docs/`.

Both pipelines publish to the exact same topic and message type
(`/perception/tracked_dynamic_obstacles`, `jo_msgs/ObstacleArray`), so
every tool here works unmodified against either one — run one pipeline at a
time against the same bag, extract, repeat for the other, diff the results.

## Files

- `extract_obstacles.py` — one-shot ROS2 subscriber, dumps a bag replay's
  `ObstacleArray` stream to JSON. Run once per pipeline under test.
- `GT_SCHEMA.md` — the hand-annotation format (waypoint-based: a few
  timestamped position+status points per real object, interpolated at eval
  time) and how to annotate a bag using the report's own camera-track videos
  as reference.
- `eval_tracking.py` — compares an extracted JSON against a ground-truth
  JSON. Reports CLEAR MOT metrics (MOTA/MOTP/IDF1/ID-switches, via
  `py-motmetrics` — same library/protocol as MOTChallenge/KITTI-tracking, so
  numbers are comparable to published 3D-MOT results), classification
  precision/recall/F1 on `dynamic`/`potentially_dynamic` restricted to
  correctly-matched pairs, and debounce latency (real elapsed time from a
  true motion onset to the first correct predicted label — the number that
  actually matters for a real-time system, not just a configured tick count).
- `synthetic_gt.json` / `synthetic_pred.json` — a tiny built-in regression
  test (one object, one state transition, an injected ~1.2s debounce lag).
  Re-run after touching `eval_tracking.py`:
  `python3 eval_tracking.py --pred synthetic_pred.json --gt synthetic_gt.json`
  — MOTA/IDF1 should both read 100% and debounce latency should read ~1.5s
  (the 1.2s injected lag plus up to one 0.5s grid step).

## Running a comparison

```bash
# 1. Launch the pipeline under test (v1 OR v2) + localization, per the usual
#    protocol, then start the bag.
# 2. In another terminal, inside the container:
python3 extract_obstacles.py --out v2_run1.json --duration 90
# 3. Repeat steps 1-2 with the other pipeline (same bag).
python3 extract_obstacles.py --out v1_run1.json --duration 90

# 4. Once lab_session1_gt.json exists (see GT_SCHEMA.md):
python3 eval_tracking.py --pred v2_run1.json --gt lab_session1_gt.json --out v2_run1_metrics.json
python3 eval_tracking.py --pred v1_run1.json --gt lab_session1_gt.json --out v1_run1_metrics.json
```

`py-motmetrics` is not in `package.xml`/the container image yet — install
with `pip install --user motmetrics` until it's added to the Dockerfile.

## Not built yet

- The ablation variants that isolate v2's 3D-state and real-dt changes from
  its disocclusion-suppression addition (both currently require a code
  change to `tracker_node.cpp`, not just a config flag — flagging this
  rather than touching the production tracker without discussing it first).
- The actual ground-truth annotation for any bag — this needs a human
  watching the two camera videos and filling in `GT_SCHEMA.md`'s format.
- A head-to-head external baseline (AB3DMOT or a SemanticKITTI-MOS method)
  wired to consume the same bag's detections and emit a comparable JSON.
