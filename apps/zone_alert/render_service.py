# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Single-panel service demo: the zone alert app as it would run on the YOLOv8n + S3x tracker (ByteTrack, seed 0,
cross-fitted threshold). This is a product view, so no ground truth is drawn.

Window rule (fixed before anything was rendered; uses ground truth only, never what the model shows):
    candidates = every WIN_S = 8 s window (stride 1 s) inside the MOT17-04 evaluation half (frames 527..1050);
    score      = number of scored GT zone episodes, any zone, that start after the first evaluation frame and are
                 confirmed inside the window (people entering a zone during the clip; results.json, compare.py);
    choose     = highest score; tie -> earliest start.
The v2 demo (--v2) uses the same rule on the same results.json, so it shows the same window as v1.

Rendering: every 3rd source frame (30 -> 10 fps). Zones semi-transparent, ignore masks (if the zone file has any)
hatched grey, tracks green / red inside a zone / faint grey 'masked'; top banner with the newest alert; below the video
(never over it, so no zone or track is covered) a strip with camera, stream clock, model and an FPS badge with the
pipeline FPS MEASURED on this PC by bench.py (timing.json, MOT17-04 S3x row); then the rolling list of the last alerts
with stream timestamps, and the MOT17 attribution. The zone logic is replayed from the stream start
(first evaluation frame) on the cached S3x tracks, so the clip starts with the state a running app would have; with
a zone file without "ignore" the replayed events are asserted equal to compare.py's.
--v2 (zones/MOT17-04_v2.json): additionally the alarm-direction arrows of the zones (half-transparent, unlabelled;
explained in the info strip) and the tripwire; a track with a raised APPROACH warning is drawn yellow with its predicted
foot path (arrow, at most the next 1.5 s) and 'APPROACH n s' while the approach condition holds and up to 0.3 s after
(n = predicted seconds to the entry, counted down), the banner turns amber for it unless an alarm is also recent (red
wins); a stay that started with a non-alarm ENTRY is drawn blue and only listed in the log. The FPS badge is the v2 row
of timing_v2.json (bench.py --v2, the new logic included). The replay is asserted to give compare.py's events once
APPROACH / LINE_CROSS are dropped and ENTRY is read as INTRUSION. The log lists each APPROACH of the clip with its
outcome in compare_v2.py's evaluation (results_v2.json, v2 zone files as configured).
MP4 at 1280 px; GIF at the largest width in GIF_WIDTHS that stays <= 8 MiB (8,388,608 bytes).

Inputs: cache/MOT17-04__s3x.json and results.json (compare.py), timing.json (bench.py) or timing_v2.json (--v2),
results_v2.json (--v2, compare_v2.py).
Outputs: demo/zone_alert_service.gif / .mp4, demo/service_log.json (v2: *_v2.gif / .mp4 / service_log_v2.json)
usage (repository root): python -B apps/zone_alert/render_service.py [--v2] [--zones FILE]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zone_alert as za  # noqa: E402
import compare  # noqa: E402
import render_demo  # noqa: E402  (load_json, write_gif)
from efpmot.data import mot17 as md  # noqa: E402

SEQ = "MOT17-04-FRCNN"
MODEL = "s3x"
WIN_S = 8.0
STEP = 3                         # 30 -> 10 fps
OUT_W = 1280
GIF_WIDTHS = (1280, 1120, 960)
GIF_MAX = 8 * 1024 * 1024
OUT = os.path.join(HERE, "demo")
LOG_TYPES = ("INTRUSION", "LOITERING", "APPROACH", "ENTRY", "LINE_CROSS")


def choose_window(results: dict) -> dict:
    """Apply the window rule of the module docstring to the GT episodes in results.json."""
    fps = int(md.seqinfo(SEQ)["fps"])
    frames = md.eval_frames(SEQ)
    length = int(WIN_S * fps)
    candidates = []
    for start in range(frames[0], frames[-1] - length + 2, fps):
        n = sum(ep["scored"] and not ep["initial"] and start <= ep["confirm"] < start + length
                for ep in results["episodes"][SEQ])
        candidates.append(dict(start=start, end=start + length - 1, gt_entries=int(n)))
    best = max(c["gt_entries"] for c in candidates)
    chosen = next(c for c in candidates if c["gt_entries"] == best)     # earliest of the best
    return dict(chosen, seq=SEQ, n_candidates=len(candidates), scores=[c["gt_entries"] for c in candidates])


def fps_badge_text(timing: dict, zones: str | None = None) -> str:
    """'58 FPS (55-60)  PC GPU, detect+track+alert' from the MOT17-04 S3x row of timing.json (or the row with
    zones == `zones` of timing_v2.json)."""
    row = next(r for r in timing["rows"] if r["seq"] == SEQ and r["model"] == MODEL and r.get("zones") == zones)
    p = row["pipeline_fps"]
    return f"{p['median']:.0f} FPS ({p['min']:.0f}-{p['max']:.0f})  PC GPU, detect+track+alert"


def info_strip(width: int, left: list[str], right: list[str], scale: float) -> np.ndarray:
    """Strip below the video (nothing is drawn over the camera image, so no zone or track is hidden): camera / clock /
    model lines on the left, the FPS badge lines right-aligned on the right (first right line highlighted)."""
    lh = int(round(34 * scale))
    strip = np.full((lh * max(len(left), len(right)) + 8, width, 3), (48, 48, 48), np.uint8)
    for i, s in enumerate(left):
        za.text(strip, s, (8, lh * (i + 1) - 2), scale, za.WHITE)
    for i, s in enumerate(right):
        tw = cv2.getTextSize(s, za.FONT, scale, 1)[0][0]
        za.text(strip, s, (width - tw - 10, lh * (i + 1) - 2), scale, (120, 230, 120) if i == 0 else za.GREY)
    return strip


def frame_view(img: np.ndarray, zones: list[za.Zone], monitor: za.ZoneMonitor, tracks: np.ndarray, log: list,
               frame: int, frame0: int, fps: float, badge: str, thr: float, width: int, v2: bool = False) -> np.ndarray:
    """One service frame at `width` px: banner, video with zones / masks / (v2: directions, tripwires) / tracks (no
    overlay on the image), info strip (camera, clock, model, FPS badge), event list, attribution."""
    u = width / 1280                                     # text scale relative to the 1280 px layout
    s = width / img.shape[1]
    vis = cv2.resize(img, (width, int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    za.draw_zones(vis, zones, s)
    za.draw_ignore(vis, monitor.ignore, s, label_scale=0.5 * u)
    if v2:
        za.draw_directions(vis, zones, s, label=False)
        za.draw_lines(vis, monitor.lines, s, scale=0.5 * u)
    za.draw_tracks(vis, tracks, monitor, s, id_scale=0.42 * u)
    t = (frame - frame0) / fps
    left = [f"CAM {SEQ[:8]} (recorded, replayed)   stream time {int(t // 60):02d}:{t % 60:04.1f}",
            f"YOLOv8n + S3x  |  ByteTrack thr {thr:g}"]
    if v2:
        left.append("zone arrows: alarm entry direction (within 60 deg)   magenta line: tripwire")
    info = info_strip(width, left,
                      [badge, "median of 3 runs, bench.py (RTX 5060 Ti)" + (" --v2, new logic incl." if v2 else "")],
                      0.55 * u)
    recent = [e for e in log if frame - e.frame < za.BANNER_S * fps and e.type != "ENTRY"]
    counts = {k: sum(e.type == k for e in log) for k in LOG_TYPES}
    if v2:
        header = (f"log since stream start: {counts['INTRUSION']} INTRUSION, {counts['LOITERING']} LOITERING, "
                  f"{counts['LINE_CROSS']} LINE_CROSS, {counts['APPROACH']} APPROACH, {counts['ENTRY']} ENTRY (no alarm)")
        legend = "yellow = APPROACH (arrow: predicted path)  red = alarm  blue = ENTRY"
    else:
        header = (f"alert log since stream start: {counts['INTRUSION']} INTRUSION, {counts['LOITERING']} LOITERING"
                  "   (newest first, time = stream time)")
        legend = "tracks: green = person, red = inside a zone"
    return za.pad_to(np.vstack([za.status_bar(width, recent, zones, scale=0.68 * u), vis, info,
                                za.event_strip(width, log, n=4, scale=0.56 * u, header=header),
                                za.attribution_bar(width, 0.5 * u, extra=legend)]))


def zones_label(path: str) -> str:
    """Zone file path for the log: relative to this folder when possible, absolute otherwise (another drive)."""
    try:
        return os.path.relpath(path, HERE).replace(os.sep, "/")
    except ValueError:
        return os.path.abspath(path)


class Frozen:
    """Monitor view frozen at one frame (draw_tracks / draw_ignore / draw_lines read only these)."""

    def __init__(self, monitor: za.ZoneMonitor, tids: list[int]):
        self.ignore, self.lines = monitor.ignore, monitor.lines
        self._m = {tid: monitor.is_masked(tid) for tid in tids}
        self._i = {tid: monitor.inside_zones(tid) for tid in tids}
        self._a = {tid: monitor.alarm_zones(tid) for tid in tids}
        self._w = {tid: monitor.approach_view(tid) for tid in tids}

    def is_masked(self, tid: int) -> bool:
        return self._m.get(tid, False)

    def inside_zones(self, tid: int) -> list[str]:
        return self._i.get(tid, [])

    def alarm_zones(self, tid: int) -> list[str]:
        return self._a.get(tid, [])

    def approach_view(self, tid: int) -> tuple | None:
        return self._w.get(tid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", action="store_true", help="v2 demo: APPROACH, direction filter, tripwire")
    ap.add_argument("--zones", default=None, help="zone file (may carry an 'ignore' list); default zones/MOT17-04"
                                                  "[_v2].json")
    args = ap.parse_args()
    suffix = "_v2" if args.v2 else ""
    zpath = args.zones or os.path.join(HERE, "zones", SEQ[:8] + suffix + ".json")
    results = render_demo.load_json(os.path.join(HERE, "results.json"))
    timing = render_demo.load_json(os.path.join(HERE, "timing_v2.json" if args.v2 else "timing.json"))
    run = render_demo.load_json(os.path.join(HERE, "cache", f"{SEQ[:8]}__{MODEL}.json"))
    window = choose_window(results)
    w0, w1 = window["start"], window["end"]
    fps = float(md.seqinfo(SEQ)["fps"])
    frames = md.eval_frames(SEQ)
    zones, ignore, lines = za.load_zones(zpath), za.load_ignore(zpath), za.load_lines(zpath)
    monitor = za.ZoneMonitor(zones, fps, frame0=frames[0], ignore=ignore, lines=lines)
    badge = fps_badge_text(timing, "v2" if args.v2 else None)
    shown = set(range(w0, w1 + 1, STEP))
    all_events, log, kept = [], [], []          # kept: per shown frame (frame, tracks, log, frozen monitor view)
    for f in frames:
        tracks = np.asarray(run["tracks"][str(f)], np.float32).reshape(-1, 6)
        events = monitor.update(f, *za.observations(tracks))
        all_events += events
        log += [e for e in events if e.type != "EXIT"]
        if f in shown:
            kept.append((f, tracks, list(log), Frozen(monitor, tracks[:, 4].astype(int).tolist())))
    if not ignore:
        replayed = [("INTRUSION" if e.type == "ENTRY" else e.type, e.zone, e.track_id, e.frame) for e in all_events
                    if e.type not in ("APPROACH", "LINE_CROSS")]
        cached = [(e["type"], e["zone"], e["track_id"], e["frame"]) for e in run["events"]]
        assert replayed == cached, "replay differs from compare.py"

    def render(width: int) -> list[np.ndarray]:
        return [frame_view(cv2.imread(md.frame_path(SEQ, f)), zones, view, tr, lg, f, frames[0], fps, badge,
                           run["thr"], width, args.v2) for f, tr, lg, view in kept]

    os.makedirs(OUT, exist_ok=True)
    mp4 = os.path.join(OUT, f"zone_alert_service{suffix}.mp4")
    gif = os.path.join(OUT, f"zone_alert_service{suffix}.gif")
    out = render(OUT_W)
    writer, codec = za.open_mp4(mp4, fps / STEP, (out[0].shape[1], out[0].shape[0]))
    for image in out:
        writer.write(image)
    writer.release()
    gif_w, size = None, None
    for width in GIF_WIDTHS:
        size = render_demo.write_gif(gif, out if width == OUT_W else render(width), fps / STEP)
        if size <= GIF_MAX:
            gif_w = width
            break
    assert gif_w is not None, f"GIF {size / 2**20:.2f} MiB > 8 MiB at every width"
    in_clip = [e for e in log if w0 <= e.frame <= w1]
    outcomes = {}
    if args.v2:                                  # outcome of each APPROACH in compare_v2.py's evaluation
        res_v2 = render_demo.load_json(os.path.join(HERE, "results_v2.json"))
        outcomes = {(w["zone"], w["track_id"], w["frame"]): w["outcome"] for w in res_v2["warnings"][SEQ][MODEL]["v2"]}
    approaches = [dict(zone=e.zone, track_id=e.track_id, frame=e.frame, t=round((e.frame - frames[0]) / fps, 2),
                       eta_s=e.eta_s, outcome=outcomes.get((e.zone, e.track_id, e.frame)))
                  for e in log if e.type == "APPROACH" and w0 - 3 * fps <= e.frame <= w1]
    types = LOG_TYPES if args.v2 else ("INTRUSION", "LOITERING")
    log_out = dict(window=window, window_s=[round((w0 - frames[0]) / fps, 2), round((w1 + 1 - frames[0]) / fps, 2)],
                   zones_file=zones_label(zpath), ignore_regions=len(ignore), fps_badge=badge,
                   out_frames=len(out), out_fps=fps / STEP, mp4_wh=[out[0].shape[1], out[0].shape[0]], mp4_codec=codec,
                   mp4_bytes=os.path.getsize(mp4), gif_width=gif_w, gif_bytes=size,
                   alerts_in_clip={k: sum(e.type == k for e in in_clip) for k in types},
                   alerts_in_clip_list=[[e.type, e.zone, e.track_id, round((e.frame - frames[0]) / fps, 2)]
                                        for e in in_clip],
                   approach_outcomes=approaches)
    with open(os.path.join(OUT, f"service_log{suffix}.json"), "w", encoding="utf-8") as fh:
        json.dump(log_out, fh, indent=1)
    print(json.dumps({k: v for k, v in log_out.items() if k != "alerts_in_clip_list"}, indent=1))


if __name__ == "__main__":
    main()
