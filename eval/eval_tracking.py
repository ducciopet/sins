#!/usr/bin/env python3
"""
eval_tracking.py
------------------
Compares a prediction JSON (produced by extract_obstacles.py, from EITHER
jo_detector v1 or sins — same schema, see that file's
own header) against a hand-annotated ground truth JSON (waypoint-based, see
GT_SCHEMA.md), and reports:

  - Standard CLEAR MOT identity/position metrics (MOTA, MOTP, IDF1, id
    switches, ...) via py-motmetrics — the same library/protocol used by
    MOTChallenge/KITTI-tracking/nuScenes-tracking, so these numbers are
    directly comparable to published 3D-MOT results (AB3DMOT and similar).
  - Classification quality on dynamic/potentially_dynamic, computed ONLY
    over matched (ground-truth, predicted) pairs (precision/recall/F1 per
    status, confusion matrix) — this is the metric the paper is actually
    about; MOTA/IDF1 are supporting evidence that the identity/position
    side of the pipeline isn't silently inflating or deflating it.
  - Debounce latency: for every GT status transition into "dynamic", the
    real elapsed time until the matched predicted obstacle's own status
    first agrees — directly measures the debounce window's real-world cost,
    not just its configured tick count.

No ROS dependency: pure Python + numpy/pandas/motmetrics, so it runs equally
well outside the container once the two input JSONs exist.

Usage:
    python3 eval_tracking.py --pred v2_run1.json --gt lab_session1_gt.json
    python3 eval_tracking.py --pred v1_run1.json --gt lab_session1_gt.json \\
        --out v1_run1_metrics.json
"""
import argparse
import bisect
import json

import numpy as np
import motmetrics as mm


def interpolate_gt(objects, grid_t):
    """objects: GT_SCHEMA.md's 'objects' list. Returns {gt_id: [(t, x, y, status), ...]}
    sampled at every timestamp in grid_t (status is a step function, position
    is linearly interpolated; a GT object absent from the grid point's time
    range - i.e. before its first or after its last waypoint - is omitted)."""
    out = {}
    for obj in objects:
        wps = sorted(obj['waypoints'], key=lambda w: w['t'])
        ts = [w['t'] for w in wps]
        series = []
        for t in grid_t:
            if t < ts[0] or t > ts[-1]:
                continue
            i = bisect.bisect_right(ts, t) - 1
            i = max(0, min(i, len(wps) - 2)) if len(wps) > 1 else 0
            w0, w1 = wps[i], wps[min(i + 1, len(wps) - 1)]
            if w1['t'] > w0['t']:
                frac = (t - w0['t']) / (w1['t'] - w0['t'])
            else:
                frac = 0.0
            x = w0['x'] + frac * (w1['x'] - w0['x'])
            y = w0['y'] + frac * (w1['y'] - w0['y'])
            # status: step function using the LAST waypoint at or before t
            status_i = bisect.bisect_right(ts, t) - 1
            status = wps[max(0, status_i)]['status']
            series.append((t, x, y, status))
        out[obj['gt_id']] = series
    return out


def nearest_pred_frame(pred_frames, t, max_dt):
    """pred_frames: list of {'t':..., 'obstacles':[...]}, sorted by t. Returns
    the closest frame's obstacle list, or None if nothing within max_dt."""
    ts = [f['t'] for f in pred_frames]
    i = bisect.bisect_left(ts, t)
    candidates = [j for j in (i - 1, i) if 0 <= j < len(ts)]
    if not candidates:
        return None
    best = min(candidates, key=lambda j: abs(ts[j] - t))
    if abs(ts[best] - t) > max_dt:
        return None
    return pred_frames[best]['obstacles']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True, help="output of extract_obstacles.py")
    ap.add_argument('--gt', required=True, help="hand-annotated ground truth, see GT_SCHEMA.md")
    ap.add_argument('--frame-period', type=float, default=0.5,
                     help="evaluation grid spacing in seconds")
    ap.add_argument('--max-dist', type=float, default=1.0,
                     help="[m] matching cutoff for GT<->predicted association")
    ap.add_argument('--max-frame-gap', type=float, default=0.3,
                     help="[s] max gap to the nearest actual predicted frame before treating a grid point as 'no prediction'")
    ap.add_argument('--out', default=None, help="optional JSON file to also write the summary to")
    args = ap.parse_args()

    pred_frames = json.load(open(args.pred))
    pred_frames.sort(key=lambda f: f['t'])
    gt = json.load(open(args.gt))

    t0 = max(pred_frames[0]['t'], min(w['t'] for o in gt['objects'] for w in o['waypoints']))
    t1 = min(pred_frames[-1]['t'], max(w['t'] for o in gt['objects'] for w in o['waypoints']))
    grid_t = list(np.arange(t0, t1, args.frame_period))
    gt_series = interpolate_gt(gt['objects'], grid_t)

    acc = mm.MOTAccumulator(auto_id=True)
    status_pairs = []  # (gt_status, pred_status) for every matched pair, every frame
    transition_pending = {}  # gt_id -> t of the GT transition INTO "dynamic" awaiting confirmation
    debounce_latencies = []

    # motmetrics wants numeric GT/hypothesis ids; keep a stable string->int map
    # for GT ids (hypothesis ids are already the tracker's own numeric track_id).
    gt_id_to_num = {gid: i for i, gid in enumerate(gt_series.keys())}

    for t in grid_t:
        gt_ids, gt_pts, gt_statuses, gt_id_strs = [], [], [], []
        for gt_id, series in gt_series.items():
            match = [s for s in series if abs(s[0] - t) < 1e-6]
            if not match:
                continue
            _, x, y, status = match[0]
            gt_ids.append(gt_id_to_num[gt_id])
            gt_id_strs.append(gt_id)
            gt_pts.append((x, y))
            gt_statuses.append(status)

        obstacles = nearest_pred_frame(pred_frames, t, args.max_frame_gap) or []
        pred_ids = [int(o['track_id']) for o in obstacles]
        pred_pts = [(o['x'], o['y']) for o in obstacles]
        pred_statuses = [o['status'] for o in obstacles]

        if gt_pts and pred_pts:
            dists = mm.distances.norm2squared_matrix(np.array(gt_pts), np.array(pred_pts), max_d2=args.max_dist ** 2)
            dists = np.sqrt(dists)
        else:
            dists = np.empty((len(gt_pts), len(pred_pts)))
        acc.update(gt_ids, pred_ids, dists)

        # status agreement on matched pairs only, via the same greedy row-wise
        # argmin motmetrics itself uses internally for MOTP - good enough for
        # a secondary classification metric, doesn't need to be the official
        # CLEAR MOT assignment.
        if gt_pts and pred_pts:
            for gi, g_status in enumerate(gt_statuses):
                row = dists[gi]
                if row.size == 0 or np.all(np.isnan(row)):
                    continue
                pj = int(np.nanargmin(row))
                if np.isnan(row[pj]):
                    continue
                status_pairs.append((g_status, pred_statuses[pj]))

                gt_id_str = gt_id_strs[gi]
                if pred_statuses[pj] == 'dynamic' and gt_id_str in transition_pending:
                    debounce_latencies.append(t - transition_pending.pop(gt_id_str))

        # detect GT transitions into "dynamic" for the debounce-latency metric
        for gi, gt_id_str in enumerate(gt_id_strs):
            if gt_statuses[gi] == 'dynamic' and gt_id_str not in transition_pending:
                prev = [s for s in gt_series[gt_id_str] if s[0] < t]
                if not prev or prev[-1][3] != 'dynamic':
                    transition_pending[gt_id_str] = t

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['mota', 'motp', 'idf1', 'num_switches',
                                        'num_false_positives', 'num_misses',
                                        'num_objects', 'num_predictions'], name='run')
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))

    print("\n--- classification quality (matched pairs only) ---")
    classes = ['static', 'potentially_dynamic', 'dynamic']
    confusion = {g: {p: 0 for p in classes + ['tracked', 'other']} for g in classes}
    for g, p in status_pairs:
        row = confusion.setdefault(g, {c: 0 for c in classes + ['tracked', 'other']})
        key = p if p in row else 'other'
        row[key] += 1

    report = {}
    for cls in ['dynamic', 'potentially_dynamic']:
        tp = confusion.get(cls, {}).get(cls, 0)
        fn = sum(v for k, v in confusion.get(cls, {}).items()) - tp
        fp = sum(confusion.get(g, {}).get(cls, 0) for g in classes if g != cls)
        precision = tp / (tp + fp) if (tp + fp) else float('nan')
        recall = tp / (tp + fn) if (tp + fn) else float('nan')
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float('nan')
        report[cls] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': tp + fn}
        print(f"{cls:22s} precision={precision:.3f} recall={recall:.3f} f1={f1:.3f} (n={tp + fn})")

    if debounce_latencies:
        print(f"\ndebounce latency (GT->dynamic until predicted agrees): "
              f"mean={np.mean(debounce_latencies):.2f}s median={np.median(debounce_latencies):.2f}s "
              f"n={len(debounce_latencies)}")
        report['debounce_latency_sec'] = {
            'mean': float(np.mean(debounce_latencies)),
            'median': float(np.median(debounce_latencies)),
            'n': len(debounce_latencies),
        }
    else:
        print("\ndebounce latency: no GT->dynamic transitions found (or none ever matched)")

    if args.out:
        full = {'mot_summary': summary.to_dict(), 'classification': report}
        with open(args.out, 'w') as f:
            json.dump(full, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
