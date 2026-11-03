# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Held-out choice of the APPROACH fit-residual cut (zone_alert.MAX_RESIDUAL) on MOT17-02.

Why held out: a first look at the evaluation's own warnings (MOT17-04 / MOT17-09) suggested a cut near 0.035 body
heights, so a cut read off those warnings would be fitted to the test data. MOT17-02 is the third static-camera MOT17
sequence; its evaluation half (frames 302-600) is used neither to train the detectors (training halves only) nor by
compare.py / compare_v2.py, and no v2 setting was chosen on it.

Rule (fixed before this script was first run):
    data     the stored benchmark ByteTrack outputs of both models on the MOT17-02 evaluation half at their cross-fitted
             thresholds (results/mot17_trackers, the same tracker code and settings as the app), and the MOT17-02 GT;
    frames   every track observation where the app's velocity fit exists (fit_s 0.8 s, >= 13 observations) and the
             speed is >= MIN_SPEED, i.e. the observations on which an APPROACH could be raised;
    clean    the track is matched (per-frame Hungarian, IoU >= 0.5, as compare.py) to one and the same GT pedestrian
             on every observed frame of its fit window: a correctly tracked person;
    cut      the 99th percentile of the RMS fit residual / box height over the clean observations, both models pooled,
             rounded up to 0.005. A correctly tracked mover then fails the check on about 1 % of its frames, and
             APPROACH needs k = 5 passing frames in a row anyway.
The script also reports how many of the other observations (track on no person, or on several people within the
window: box jumps, identity switches) lie above the cut, and the GT walking speeds of the scene (scene dependence of
MIN_SPEED).

usage (repository root): python -B apps/zone_alert/calibrate_v2.py   -> apps/zone_alert/calibration_v2.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zone_alert as za  # noqa: E402
import compare  # noqa: E402
import compare_v2  # noqa: E402  (scene_speeds)
from efpmot import paths  # noqa: E402
from efpmot.data import mot17 as md  # noqa: E402

SEQ = "MOT17-02-FRCNN"
PERCENTILE = 99.0


def stored_tracks(model: str, thr: float) -> dict[int, list]:
    """Benchmark tracker output of the evaluation half -> {frame: [[x1, y1, x2, y2, id, conf], ...]}."""
    run_dir = compare.MODELS[model][1]
    path = os.path.join(paths.require("outputs.results"), "mot17_trackers", f"{run_dir}__bytetrack_t{thr:g}", "data",
                        SEQ + ".txt")
    offset = md.half_offset(SEQ)
    tracks = {f: [] for f in md.eval_frames(SEQ)}
    with open(path) as fh:
        for line in fh:
            p = line.split(",")
            x, y, w, h, conf = (float(v) for v in p[2:7])
            tracks[int(p[0]) + offset].append([x, y, x + w, y + h, float(p[1]), conf])
    return tracks


def residuals(tracks: dict[int, list], gt: compare.GroundTruth, fps: float) -> tuple[list[float], list[float]]:
    """Residual / box height of every observation with a fit and speed >= MIN_SPEED -> (clean, other)."""
    matches, _, _ = compare.track_gt_match(tracks, gt, compare.MATCH_IOU)
    monitor = za.ZoneMonitor([], fps)              # only its motion fit (same window and minimum as the app)
    clean, other = [], []
    for f in sorted(tracks):
        feet, heights = za.observations(np.asarray(tracks[f], np.float32).reshape(-1, 6))
        for tid, xy in feet.items():
            motion = monitor._motion(tid, f, xy, heights[tid])
            if motion is None or motion.speed is None or motion.speed < za.MIN_SPEED:
                continue
            window = [o[0] for o in monitor.history[tid] if o[0] > f - monitor.fit_frames]
            gids = {matches.get((g, tid)) for g in window}
            (clean if len(gids) == 1 and None not in gids else other).append(motion.residual_bh)
    return clean, other


def main() -> None:
    fps = float(md.seqinfo(SEQ)["fps"])
    gt = compare.read_gt(SEQ)
    data = {}
    for model, (_, _, variant) in compare.MODELS.items():
        thr = compare.thresholds(variant)[SEQ]
        data[model] = (thr, *residuals(stored_tracks(model, thr), gt, fps))
    pooled_clean = [v for _, clean, _ in data.values() for v in clean]
    p = float(np.percentile(pooled_clean, PERCENTILE))
    cut = math.ceil(p / 0.005 - 1e-9) * 0.005
    per_model = {model: dict(thr=thr, clean_n=len(clean), clean_p50=round(float(np.median(clean)), 4),
                             clean_above_cut=sum(v > cut for v in clean), other_n=len(other),
                             other_p50=round(float(np.median(other)), 4) if other else None,
                             other_above_cut=sum(v > cut for v in other))
                 for model, (thr, clean, other) in data.items()}
    out = dict(created=time.strftime("%Y-%m-%d %H:%M"), sequence=SEQ, frames=[md.eval_frames(SEQ)[0],
               md.eval_frames(SEQ)[-1]], rule=__doc__.split("Rule (fixed before this script was first run):")[1]
               .split("The script also")[0].strip(), percentile=PERCENTILE, pooled_clean_n=len(pooled_clean),
               pooled_clean_percentile=round(p, 4), max_residual=round(cut, 3), per_model=per_model,
               gt_speeds=compare_v2.scene_speeds(gt, md.eval_frames(SEQ), fps))
    with open(os.path.join(HERE, "calibration_v2.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rule"}, indent=1))


if __name__ == "__main__":
    main()
