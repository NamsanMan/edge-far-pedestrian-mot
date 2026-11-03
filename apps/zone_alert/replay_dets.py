# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Replay a detection cache through the app's tracker and zone logic (no detector, no GPU, no images).

The detection cache of the benchmark (scripts/export_dets.py, <cache>/<tag>/mot17/<seq>.npz) holds exactly what the app's
PedestrianDetector returns on the same frames (same letterbox, FP16 export, conf 0.01, class-wise NMS 0.65). Its pedestrian
boxes go through PedestrianTracker (ByteTrack, the benchmark settings) and ZoneMonitor with the evaluation zone file, as in
compare.py. With --expect, the events are compared field by field with a stored run (compare.py writes cache/<seq>__<model>.json).

usage (repository root):
    python -B apps/zone_alert/replay_dets.py --cache <cache>/mot_v8n_416s3x_s0/mot17/MOT17-09-FRCNN.npz \
        --zones apps/zone_alert/zones/MOT17-09.json --thr 0.8 [--expect apps/zone_alert/cache/MOT17-09__s3x.json] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zone_alert as za  # noqa: E402  (also puts the repository root on the path)
from efpmot.data import mot17 as md  # noqa: E402


def replay(cache: str, zones_file: str, thr: float, fps: float = 30.0) -> dict:
    z = np.load(cache, allow_pickle=True)
    frames = [md.frame_no(str(f)) for f in z["frames"]]
    tracker = za.PedestrianTracker(thr, fps)
    monitor = za.ZoneMonitor(za.load_zones(zones_file), fps, frame0=frames[0])
    blank = np.zeros((1, 1, 3), np.uint8)   # ByteTrack ignores the image
    tracks, events = {}, []
    for f, d in zip(frames, z["dets"]):
        d = np.asarray(d, np.float32).reshape(-1, 6)
        ped = d[d[:, 5] == za.PED, :5]
        t = tracker.update(ped, blank)
        tracks[f] = t.tolist()
        events += [asdict(e) for e in monitor.update(f, *za.observations(t))]
    events = [dict(e, foot=None if e["foot"] is None else list(e["foot"])) for e in events]
    return dict(thr=thr, frames=len(frames), tracks=tracks, events=events)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="<cache>/<tag>/mot17/<seq>.npz")
    ap.add_argument("--zones", required=True, help="zone file (zones/<seq>.json)")
    ap.add_argument("--thr", type=float, required=True, help="ByteTrack track threshold")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--expect", default=None, help="stored run (compare.py cache json) to compare the events with")
    ap.add_argument("--out", default=None, help="write the replayed events as JSON")
    a = ap.parse_args()
    r = replay(a.cache, a.zones, a.thr, a.fps)
    counts = {}
    for e in r["events"]:
        counts[e["type"]] = counts.get(e["type"], 0) + 1
    print(f"replayed {r['frames']} frames: {len(r['events'])} events {counts}")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(r["events"], fh, indent=1)
    if a.expect:
        with open(a.expect) as fh:
            ref = json.load(fh)
        same_tracks = all(np.allclose(np.asarray(r["tracks"][int(f)], np.float32).reshape(-1, 6),
                                      np.asarray(v, np.float32).reshape(-1, 6), atol=1e-3)
                          if len(r["tracks"][int(f)]) == len(v) else False for f, v in ref["tracks"].items())
        same = r["events"] == ref["events"]
        print(f"tracks equal on all {len(ref['tracks'])} frames: {same_tracks}; events equal ({len(ref['events'])} stored): {same}")
        print("EQUAL" if same and same_tracks else "DIFFERENT")
        sys.exit(0 if same and same_tracks else 1)


if __name__ == "__main__":
    main()
