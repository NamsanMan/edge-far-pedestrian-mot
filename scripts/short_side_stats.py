# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Ground-truth geometry behind the short-side rule (docs/METHOD.md, "how many objects fall below one stride").

For every annotated box: aspect ratio w / h, whether the width is the short side, and whether the short side falls below
one P3 stride (8 px) once the frame is letterboxed to a long side of 416 and of 640 px (scale = L / max(W, H) of the
native frame). Ground truth only; no model, no GPU.

    python -B scripts/short_side_stats.py [--data visdrone mot17] [--json OUT]

- VisDrone2019-MOT train + val (`data.visdrone_mot`): boxes with the consider flag 1, w, h > 0, categories pedestrian (1),
  people (2) and car (4); the native frame size is read from the first image of each sequence.
- MOT17 train, FRCNN copies (`data.mot17_train`): class 1 (pedestrian), consider flag 1, visibility >= 0.25, w, h > 0;
  frame size from seqinfo.ini. All frames of each sequence (both halves).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from efpmot.data import mot17 as M  # noqa: E402
from efpmot.data import visdrone as V  # noqa: E402

STRIDE = 8.0
LONG = (416, 640)
VIS_CLASSES = {1: "pedestrian", 2: "people", 4: "car"}


def visdrone_boxes() -> dict[str, list[np.ndarray]]:
    """class -> list of [N, 3] arrays (w, h, native long side) over VisDrone2019-MOT train + val"""
    import cv2
    out: dict[str, list[np.ndarray]] = {c: [] for c in VIS_CLASSES.values()}
    for split in ("train", "val"):
        for seq in V.list_seqs(split):
            img = cv2.imread(V.frame_paths(split, seq)[0])
            if img is None:
                raise FileNotFoundError(f"no frames for VisDrone {split}/{seq} under {V.img_root(split)}")
            longside = float(max(img.shape[:2]))
            a = np.loadtxt(os.path.join(V.root(), V.SPLIT_DIR[split], "annotations", seq + ".txt"), delimiter=",",
                           ndmin=2, usecols=range(8))
            w, h, flag, cat = a[:, 4], a[:, 5], a[:, 6].astype(int), a[:, 7].astype(int)
            for k, name in VIS_CLASSES.items():
                m = (cat == k) & (flag == 1) & (w > 0) & (h > 0)
                out[name].append(np.stack([w[m], h[m], np.full(int(m.sum()), longside)], 1))
    return out


def mot17_boxes(min_vis: float = 0.25) -> dict[str, list[np.ndarray]]:
    out = {"pedestrian": []}
    for seq in M.SEQS:
        H, W = M.seq_hw(seq)
        a = np.loadtxt(os.path.join(M.data_root(), seq, "gt", "gt.txt"), delimiter=",", ndmin=2)
        w, h, flag, cl, vis = a[:, 4], a[:, 5], a[:, 6].astype(int), a[:, 7].astype(int), a[:, 8]
        m = (cl == M.PED_CLASS) & (flag == 1) & (vis >= min_vis) & (w > 0) & (h > 0)
        out["pedestrian"].append(np.stack([w[m], h[m], np.full(int(m.sum()), float(max(H, W)))], 1))
    return out


def summarise(b: np.ndarray) -> dict:
    w, h, L = b[:, 0], b[:, 1], b[:, 2]
    short = np.minimum(w, h)
    row = dict(boxes=int(len(b)), median_w_over_h=float(np.median(w / h)), width_is_short=float(np.mean(w < h)))
    for side in LONG:
        small = short * side / L < STRIDE
        row[f"short_below_{int(STRIDE)}px_at_{side}"] = float(small.mean())
        if small.any():
            row[f"of_those_width_short_at_{side}"] = float(np.mean(w[small] < h[small]))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", default=["visdrone", "mot17"], choices=["visdrone", "mot17"])
    ap.add_argument("--json", default=None, help="also write the rows to this file")
    a = ap.parse_args()
    rows = []
    for data in a.data:
        boxes = visdrone_boxes() if data == "visdrone" else mot17_boxes()
        for cls, parts in boxes.items():
            rows.append(dict(data=data, cls=cls, **summarise(np.concatenate(parts))))
    print("| data | class | boxes | median w / h | width is the short side | short side < 8 px at 416 / 640 |"
          " of those, width is short at 640 |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['data']} | {r['cls']} | {r['boxes']:,} | {r['median_w_over_h']:.2f} | {100 * r['width_is_short']:.0f} % | "
              f"{100 * r['short_below_8px_at_416']:.0f} % / {100 * r['short_below_8px_at_640']:.0f} % | "
              f"{100 * r.get('of_those_width_short_at_640', float('nan')):.0f} % |")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
            fh.write("\n")


if __name__ == "__main__":
    main()
