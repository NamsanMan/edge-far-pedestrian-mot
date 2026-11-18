# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""The demo video of Thermal Watch (README.md in this folder): the app screens only.
  live    70m-90_1   0-20 s     source video of HIT-UAV recording r011: a group on foot seen from 70 m, camera straight down
          130m-60_3  24-34.5 s  source video of r260: an empty field from 130 m, then three people about 160 m away
  review  70m-90_1   the whole 33 s flight analysed first; a scripted pointer clicks two alert cards and one bar mark (the same
                     ReviewScreen as `thermal_watch.py --review`, with simulated input)
Model: the S3x release file of HIT-UAV fold A (r011 and r260 are in its test split). H.264, short fades through black between parts.
  default                      1280 x 720, 7.5 fps (the camera rate), about 9 MB
  --fps 30 --size 1920x1080    the YouTube version: each camera frame is drawn 4 times so pointer, playhead, clock and alert pulse
                               move at 30 fps; the thermal image and its boxes still change at 7.5 fps (no invented frames)
usage: python apps/thermal_watch/demo.py --videos DIR [--fps 7.5|30] [--size WxH] [--out PATH]
  DIR = the video_sample folder of the HIT-UAV repository (default: paths key data.hituav_videos)
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import thermal_watch as TW  # noqa: E402
from efpmot import paths  # noqa: E402

SCENES = [("70m-90_1", 0.0, 20.0), ("130m-60_3", 24.0, 10.5)]
REVIEW = "70m-90_1"
SRC_FPS, FADE, THUMB = 7.5, 4, ("70m-90_1", 120)  # thumbnail: scene, camera frame


def review_plan(scr, sub: int) -> list[dict]:
    """scripted pointer on the review screen: hold on the full alert list, then card -> card -> bar mark, each followed by about
    2.7 s of playback; returns the render arguments of every output frame (sub output frames per camera frame)"""
    rec, plan = scr.rec, []
    st = dict(i=0, sel=None, cur=(430.0, 430.0))

    def emit(n, playing=False, click=False, move_to=None):
        x0, y0 = st["cur"]
        N = n * sub
        for q in range(N):
            if move_to is not None:
                u = (q + 1) / N
                u = u * u * (3 - 2 * u)
                st["cur"] = (x0 + (move_to[0] - x0) * u, y0 + (move_to[1] - y0) * u)
            plan.append(dict(i=st["i"], playing=playing, offset=scr.offset_for(scr.current(st["i"], st["sel"])), sel=st["sel"],
                             cursor=(int(st["cur"][0]), int(st["cur"][1])), click_age=q / sub if click else None,
                             sub=(q % sub) / sub if playing else 0.0))
            if playing and q % sub == sub - 1 and st["i"] < scr.n - 1:
                st["i"] += 1

    def card_point(k):
        for kk, (x1, y1, x2, y2) in scr.card_rects(scr.offset_for(scr.current(st["i"], st["sel"]))):
            if kk == k:
                return (x1 + 150.0, (y1 + y2) / 2.0)
        return None

    ne = len(rec.events)
    emit(12)
    for k, on_bar in ((min(4, ne - 1), False), (min(6, ne - 1), False), (min(12, ne - 1), True)):
        if k < 0:
            break
        tgt = (scr.x_of(rec.events[k]["frame"]), float(scr.bar()[2])) if on_bar else card_point(k)
        if tgt is None:
            continue
        emit(7, move_to=tgt)
        st["i"], st["sel"] = rec.events[k]["frame"], k  # click -> jump
        emit(3, playing=True, click=True)
        emit(17, playing=True)
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=None, help="HIT-UAV video_sample folder (default: paths key data.hituav_videos)")
    ap.add_argument("--fps", type=float, default=SRC_FPS)
    ap.add_argument("--size", default=f"{TW.W_OUT}x{TW.H_OUT}")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    vdir = a.videos or paths.get("data.hituav_videos")
    if not vdir or not os.path.isdir(vdir):
        raise SystemExit("give --videos: the video_sample folder of the HIT-UAV repository (docs/DATA.md)")
    sub = max(1, int(round(a.fps / SRC_FPS)))
    size = tuple(int(v) for v in a.size.lower().split("x"))
    od = paths.out("results", "thermal_watch")
    a.out = a.out or os.path.join(od, "thermal_watch_demo.mp4" if (sub, size) == (1, (TW.W_OUT, TW.H_OUT))
                                  else f"thermal_watch_demo_{size[0]}x{size[1]}_{SRC_FPS * sub:g}fps.mp4")
    video = lambda stem: os.path.join(vdir, stem + ".mov")  # noqa: E731
    det, thr = TW.resolve_model(video(SCENES[0][0]), None, None, None, None)
    vw = TW.open_writer(a.out, SRC_FPS * sub, size)
    fade, n, thumb = FADE * sub, [0], {}

    def segment(total, grab=None):
        """streaming writer for one part of known length: fades its ends, resizes, keeps one raw frame for the thumbnail"""
        j = [0]

        def put(f):
            if grab is not None and j[0] == grab:
                thumb["img"] = f.copy()
            k = min(1.0, (j[0] + 1) / fade, (total - j[0]) / fade)
            f = f if k >= 1.0 else (f.astype(np.float32) * k).astype(np.uint8)
            if size != (TW.W_OUT, TW.H_OUT):
                f = cv2.resize(f, size, interpolation=cv2.INTER_LANCZOS4 if size[0] > TW.W_OUT else cv2.INTER_AREA)
            vw.write(f)
            j[0] += 1
            n[0] += 1
        return put

    for stem, start, dur in SCENES:
        cap = cv2.VideoCapture(video(stem))
        fps, n_all = cap.get(cv2.CAP_PROP_FPS) or SRC_FPS, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        f0 = int(round(start * fps))
        f1 = n_all if not dur else min(n_all, f0 + int(round(dur * fps)))
        grab = (THUMB[1] - f0) * sub if stem == THUMB[0] and f0 <= THUMB[1] < f1 else None
        TW.process(video(stem), det, thr, start, dur, sink=segment((f1 - f0) * sub, grab), write=False, subframes=sub)
    scr = TW.ReviewScreen(TW.Recording(video(REVIEW), det, thr))
    plan = review_plan(scr, sub)
    put = segment(len(plan))
    for kw in plan:
        put(scr.render(**kw))
    vw.release()
    if "img" in thumb:
        cv2.imwrite(os.path.join(od, "thermal_watch_demo.jpg"), thumb["img"], [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("DEMO_DONE", a.out, f"{n[0]} frames = {n[0] / (SRC_FPS * sub):.1f} s", f"{os.path.getsize(a.out) / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
