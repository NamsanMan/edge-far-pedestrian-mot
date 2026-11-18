# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Rebuild sample_videos.json: which HIT-UAV recording each video_sample/*.mov is the source of, and which video frames are
its labelled images.

1. Every 3rd video frame and every dataset image -> gray 80 x 64 (INTER_AREA), zero-mean, unit-norm; the recording of the best
   image (normalised correlation >= 0.95) is counted; the recording with the most such frames is the video's recording.
2. For each labelled image of that recording, its video frame = the frame of highest correlation over all frames (gray 160 x 128).
   On the five sample videos every image matches one frame at >= 0.9996, on a grid of every 7th frame.
Recordings, conditions and folds: the HIT-UAV split of efpmot (efpmot/data/hituav.py); images: paths key data.hituav.
usage: python -B apps/thermal_watch/match_videos.py --videos <video_sample> [--out FILE]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from efpmot.data import hituav as H  # noqa: E402

ABOUT = ("HIT-UAV video_sample/*.mov (640 x 512, 7.5 fps; github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset) and the "
         "recording each one is the source of: every labelled image of that recording equals one video frame (normalised "
         "correlation >= 0.9996, gray 160 x 128; apps/thermal_watch/match_videos.py). recording / test_fold refer to "
         "efpmot's HIT-UAV recording-disjoint split; labelled_frames maps video frame index -> image file name.")


def feat(gray: np.ndarray, size) -> np.ndarray:
    v = cv2.resize(gray, size, interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


def read_gray(video: str) -> list[np.ndarray]:
    cap, out = cv2.VideoCapture(video), []
    while True:
        ok, f = cap.read()
        if not ok:
            return out
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="the video_sample folder of the HIT-UAV repository")
    ap.add_argument("--out", default=os.path.join(HERE, "sample_videos.json"))
    a = ap.parse_args()
    recs = {r["id"]: r for r in H.recordings()}
    test_fold = {rid: f for f in ("recA", "recB") for rid in H.fold_recordings(f)["test"]}
    alt_test = set(H.fold_recordings("altLH")["test"])
    name2rec = {n: rid for rid, r in recs.items() for n in r["images"]}
    loc = H.official_location()
    path = lambda n: os.path.join(H.img_dir(loc[n]), n)  # noqa: E731
    names = sorted(name2rec)
    D = np.stack([feat(cv2.imread(path(n), cv2.IMREAD_GRAYSCALE), (80, 64)) for n in names])
    out = {"about": ABOUT, "videos": {}}
    for v in sorted(glob.glob(os.path.join(a.videos, "*.mov"))):
        stem, frames = os.path.splitext(os.path.basename(v))[0], read_gray(v)
        S = np.stack([feat(f, (80, 64)) for f in frames[::3]]) @ D.T
        votes = Counter(name2rec[names[b]] for b, s in zip(S.argmax(1), S.max(1)) if s >= 0.95)
        if not votes:
            print(f"{stem}: no recording found")
            continue
        rid = votes.most_common(1)[0][0]
        r = recs[rid]
        F = np.stack([feat(f, (160, 128)) for f in frames])
        rn = sorted(r["images"])
        C = np.stack([feat(cv2.imread(path(n), cv2.IMREAD_GRAYSCALE), (160, 128)) for n in rn]) @ F.T
        fi, sc = C.argmax(1), C.max(1)
        out["videos"][stem] = dict(recording=rid, code=r["code"], altitude=r["altitude"], angle=r["angle"], daynight=r["daynight"],
                                   test_fold=test_fold[rid], altLH_test=rid in alt_test, persons_labelled=r["n_person"],
                                   labelled_frames={str(int(fi[k])): rn[k] for k in np.argsort(fi)})
        print(f"{stem}: {rid} ({r['code']}, fold {test_fold[rid]}), {len(rn)} images, min corr {sc.min():.4f}")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("written", a.out)


if __name__ == "__main__":
    main()
