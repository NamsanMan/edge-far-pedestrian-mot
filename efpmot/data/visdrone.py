# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""VisDrone2019-MOT (drone views) for training and the tracking-by-detection evaluation.

Data: the official archives, <visdrone_mot>/VisDrone2019-MOT-{train,val,test-dev}/{sequences/<seq>/0000001.jpg,
annotations/<seq>.txt} (optional faster image copy: data.visdrone_images with the same split folders; a missing split falls
back to the main root). Annotation line: frame, id, left, top, w, h, score, category, truncation, occlusion
(score 1 = consider, 0 = ignore).

Decisions (fixed before any VisDrone training; docs/PREREGISTRATION.md)
1. Class mapping to the 8-class heads (BDD_CLASSES order: pedestrian, rider, car, bus, truck, train, motorcycle, bicycle):
     1 pedestrian -> pedestrian          2 people -> rider               3 bicycle -> bicycle
     4 car -> car                        5 van -> train (head 5)         6 truck -> truck                9 bus -> bus
     10 motor -> motorcycle              7 tricycle, 8 awning-tricycle -> motorcycle
     0 ignored region, 11 others -> no class: ignored regions (see 2)
   - people -> rider, NOT pedestrian. VisDrone 'people' = humans that are not standing / walking (sitting, riding bicycles or
     motors, lying), annotated as a separate class. The evaluated 'pedestrian' class is VisDrone category 1 ONLY, for GT and
     predictions alike: this is the class measured in the pre-registration (pedestrian n = 266,709 vs people n = 112,304,
     different aspect statistics), and the official VisDrone-MOT evaluation classes (pedestrian, car, van, bus, truck) also
     leave 'people' out. 'rider' (BDD: a person on a bike) is the closest of the 8 heads; keeping these persons as a foreground
     class (instead of background, or merged into pedestrian) keeps the pedestrian class definition identical to the
     pre-registered one.
   - car = VisDrone category 4 ONLY; van (5) is its own class in the otherwise unused BDD head 'train' (index 5; no VisDrone
     counterpart). The evaluated 'car' is therefore exactly the pre-registered car row (n = 537,122 = train 505,301 + val
     31,821, category 4 alone) and the official VisDrone-MOT car class (car and van scored separately). A car prediction on a
     van counts as a car FP, as in the official evaluation. (An earlier draft merged van into car, which would have measured
     car U van, a set that is neither the pre-registered car nor the official one; it was replaced before any training.)
   - tricycle / awning-tricycle -> motorcycle: there is no three-wheeler head; they are mostly motorised three-wheelers. They
     stay foreground (not background, not masked) in a class that is never evaluated, so they never count as car TP or FP.
2. Ignored regions: every box with category 0 or 11 or score 0 (in VisDrone2019-MOT exactly the category 0 and 11 boxes:
   train 62,369 + 1,474, val 3,994 + 1, test-dev 23,472 + 2,429; every other box has score 1).
   - GT objects whose box centre lies inside an ignored region of the same frame are removed (training targets and
     evaluation GT).
   - Training: ignored-region pixels AND the pixels of the GT boxes removed by the centre rule are greyed out (114, the
     letterbox fill) in BOTH frames of a pair before augmentation, except pixels inside a kept GT box of that frame, so
     unlabelled objects inside ignored regions, and the parts of removed objects that stick out of the region (train: 12,623
     removed boxes, 14.7 % of their area outside every region on average), are never taught as background.
   - Evaluation: tracker outputs whose box centre lies inside an ignored region of that frame are removed before TrackEval
     (filter_ignored(), applied after tracking, as the official toolkit drops results in ignored regions at evaluation time;
     tracking itself sees all detections). The GT is written already filtered (build_index()).
3. Training frames: all 56 VisDrone2019-MOT-train sequences, every annotated frame (TRAIN_STRIDE = 1: 24,198 frames = 1512
   iterations / epoch at batch 16, the same size as the DSEC runs (24,168 / 1510) and below the BDD 35k (2187)). Consecutive
   frames are highly redundant, but stride 2 would halve the data; 6 epochs as for every dataset.
   Frame pairs: prev = the previous ANNOTATED frame of the same sequence (labels keyed by position k in the sorted list of
   annotated frames, so the max(k - 1, 0) lookup of FramePairs is exactly that frame); k = 0 pairs with itself.
   MAX_OBJS = 256 targets per frame (VisDrone has up to 180 non-ignored objects per frame; 120 would truncate).
   As for every dataset, boxes that are <= 1 px wide or high on the letterboxed canvas are dropped.
4. Splits: VisDrone2019-MOT-val (7 sequences) is unused (no model selection); VisDrone2019-MOT-test-dev (17 sequences) is the
   evaluation split (detection caches <cache>/<tag>/visdrone/<seq>.npz, all frames; GT <derived>/visdrone_mot/gt_bdd/
   test-dev/<seq>.json in BDD / Scalabel format for TrackEval's BDD100K evaluator; run-tag prefix 'vis_').
"""
from __future__ import annotations

import glob
import json
import os

import cv2
import numpy as np

from .. import paths
from .common import BDD_CLASSES
from .pairs import FramePairs

SPLIT_DIR = {"train": "VisDrone2019-MOT-train", "val": "VisDrone2019-MOT-val", "test": "VisDrone2019-MOT-test-dev"}
VIS_CATEGORIES = ["ignored region", "pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus",
                  "motor", "others"]
VIS2BDD = {1: "pedestrian", 2: "rider", 3: "bicycle", 4: "car", 5: "train", 6: "truck", 7: "motorcycle", 8: "motorcycle", 9: "bus",
           10: "motorcycle"}
IGNORE_CATS = (0, 11)
TRAIN_STRIDE = 1
MAX_OBJS = 256
NSEQ_TEST = 17


def _sp(split: str) -> str:
    return "test" if split in ("test", "test-dev", "testdev") else split


def root() -> str:
    return paths.require("data.visdrone_mot")


def derived_dir() -> str:
    return os.path.join(paths.require("outputs.derived"), "visdrone_mot")


def split_name(split: str) -> str:
    """'test' -> 'test-dev' (folder names of gt_bdd)"""
    return SPLIT_DIR[_sp(split)].replace("VisDrone2019-MOT-", "")


def img_root(split: str) -> str:
    d = SPLIT_DIR[_sp(split)]
    fast = paths.get("data.visdrone_images")
    return paths.first_dir(os.path.join(fast, d, "sequences") if fast else None, os.path.join(root(), d, "sequences"))


def list_seqs(split: str) -> list[str]:
    return sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(root(), SPLIT_DIR[_sp(split)], "annotations"))
                  if f.endswith(".txt"))


def frame_paths(split: str, seq: str) -> list[str]:
    return sorted(glob.glob(os.path.join(img_root(split), seq, "*.jpg")))


def frame_no(path) -> int:
    return int(os.path.splitext(os.path.basename(path))[0])


def centre_in(boxes, regions) -> np.ndarray:
    """boxes [N, 4], regions [M, 4] xyxy -> bool [N]: box centre inside (borders included) any region"""
    boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
    regions = np.asarray(regions, np.float64).reshape(-1, 4)
    if not len(boxes) or not len(regions):
        return np.zeros(len(boxes), bool)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    return ((cx[:, None] >= regions[None, :, 0]) & (cx[:, None] <= regions[None, :, 2]) &
            (cy[:, None] >= regions[None, :, 1]) & (cy[:, None] <= regions[None, :, 3])).any(1)


def read_annotations(split: str, seq: str, return_dropped: bool = False):
    """reads annotations/<seq>.txt -> (frames, objs, ign) [, dropped]
    frames: sorted frame numbers with at least one annotation line (any category);
    objs: {frame: float32 [N, 6] (cls_bdd, x1, y1, x2, y2, track_id)}: boxes of mapped classes with w, h > 0 whose centre is NOT
          in an ignored region of the frame;
    ign: {frame: float32 [M, 4]} ignored regions (category 0 / 11 or score 0);
    dropped (return_dropped=True only): {frame: float32 [K, 4]} xyxy of the mapped boxes removed by the centre rule."""
    o, g, fr_all = {}, {}, set()
    with open(os.path.join(root(), SPLIT_DIR[_sp(split)], "annotations", seq + ".txt")) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 8:
                continue
            fr, tid = int(p[0]), int(p[1])
            l, t, w, h = (float(v) for v in p[2:6])
            sc, cat = int(p[6]), int(p[7])
            fr_all.add(fr)
            if cat in IGNORE_CATS or sc == 0:
                g.setdefault(fr, []).append((l, t, l + w, t + h))
                continue
            if cat not in VIS2BDD or w <= 0 or h <= 0:
                continue
            o.setdefault(fr, []).append((BDD_CLASSES.index(VIS2BDD[cat]), l, t, l + w, t + h, tid))
    ign = {k: np.asarray(v, np.float32).reshape(-1, 4) for k, v in g.items()}
    objs, dropped = {}, {}
    for k, v in o.items():
        a = np.asarray(v, np.float32).reshape(-1, 6)
        if k in ign:
            inside = centre_in(a[:, 1:5], ign[k])
            if inside.any():
                dropped[k] = a[inside, 1:5].copy()
            a = a[~inside]
        objs[k] = a
    if return_dropped:
        return sorted(fr_all), objs, ign, dropped
    return sorted(fr_all), objs, ign


def filter_ignored(rows, regions):
    """tracker output rows [M, >= 4] (x1, y1, x2, y2, ...) -> the rows whose box centre is not inside an ignored region"""
    if regions is None or not len(rows) or not len(regions):
        return rows
    return rows[~centre_in(np.asarray(rows)[:, :4], regions)]


def gt_frames_bdd(split: str, seq: str) -> list[dict]:
    """BDD / Scalabel GT for TrackEval: one entry per image (all frames of the sequence), ignored-centre objects removed"""
    paths_ = frame_paths(split, seq)
    _, objs, _ = read_annotations(split, seq)
    out = []
    for i, p in enumerate(paths_):
        a = objs.get(frame_no(p), np.zeros((0, 6), np.float32))
        labels = [{"id": int(r[5]), "category": BDD_CLASSES[int(r[0])], "attributes": {"Occluded": False, "Truncated": False, "Crowd": False},
                   "box2d": {"x1": float(r[1]), "y1": float(r[2]), "x2": float(r[3]), "y2": float(r[4])}} for r in a]
        out.append({"video_name": seq, "name": os.path.basename(p), "index": i, "labels": labels})
    return out


def gt_dir(split: str = "test") -> str:
    """<derived>/visdrone_mot/gt_bdd/<split name>"""
    return os.path.join(derived_dir(), "gt_bdd", split_name(split))


def index_file() -> str:
    return os.path.join(derived_dir(), "frame_index.json")


def seq_hw(seq: str) -> tuple[int, int]:
    """native (H, W) of a sequence from the frame index (build_index()); falls back to reading its first frame"""
    try:
        with open(index_file()) as f:
            return tuple(json.load(f)[seq]["hw"])
    except (OSError, KeyError, ValueError):
        for sp in ("test", "train", "val"):
            if seq in list_seqs(sp):
                return cv2.imread(frame_paths(sp, seq)[0]).shape[:2]
        raise


def eval_seqs() -> list[str]:
    return list_seqs("test")


def eval_frame_paths(seq: str) -> list[str]:
    return frame_paths("test", seq)


class VisDronePairs(FramePairs):
    """frame pairs (cur, prev = previous annotated frame) in the BDD class space; labels / ignored regions keyed by
    k = position in the sequence's sorted annotated frames; ignored regions greyed out (decision 2)"""

    def __init__(self, seqs, size_hw=(384, 640), train=True, max_objs=MAX_OBJS, stride=TRAIN_STRIDE, split="train", mask_ignored=True):
        self.items, self.labels, self.ignores, self.frames = [], {}, {}, {}
        for s in seqs:
            fpaths = {frame_no(p): p for p in frame_paths(split, s)}
            frames, objs, ign, drop = read_annotations(split, s, return_dropped=True)
            frames = [f for f in frames if f in fpaths]
            self.frames[s] = frames
            self.labels[s] = {k: objs.get(f, np.zeros((0, 6), np.float32)) for k, f in enumerate(frames)}
            # regions to grey out: ignored regions + the boxes the centre rule removed
            self.ignores[s] = {k: (np.concatenate([ign[f], drop[f]], 0) if f in drop else ign[f]) for k, f in enumerate(frames) if f in ign}
            for k in range(0, len(frames), stride):
                self.items.append((s, k, fpaths[frames[k]], fpaths[frames[max(k - 1, 0)]]))
        self.size, self.train, self.max_objs = size_hw, train, max_objs
        self.mask_ignored = mask_ignored

    def _load(self, seq, key, path):
        img = cv2.imread(path)
        reg = self.ignores[seq].get(key) if self.mask_ignored else None
        if reg is not None and len(reg):
            img = mask_regions(img, reg, self.labels[seq].get(key, np.zeros((0, 6), np.float32))[:, 1:5])
        return img


def mask_regions(img: np.ndarray, regions, keep_boxes, fill: int = 114) -> np.ndarray:
    """grey out regions [M, 4] (native px) except the pixels inside keep_boxes [N, 4]; returns the (modified) image"""
    H, W = img.shape[:2]
    m = np.zeros((H, W), bool)
    for x1, y1, x2, y2 in np.asarray(regions, np.float64).reshape(-1, 4):
        m[max(int(np.floor(y1)), 0):max(int(np.ceil(y2)), 0), max(int(np.floor(x1)), 0):max(int(np.ceil(x2)), 0)] = True
    for x1, y1, x2, y2 in np.asarray(keep_boxes, np.float64).reshape(-1, 4):
        m[max(int(np.floor(y1)), 0):max(int(np.ceil(y2)), 0), max(int(np.floor(x1)), 0):max(int(np.ceil(x2)), 0)] = False
    img[m] = fill
    return img


def build_index() -> str:
    """frame index (all splits) + BDD-format GT (val, test-dev) -> <derived>/visdrone_mot"""
    index = {}
    for sp in ("train", "val", "test"):
        for seq in list_seqs(sp):
            fps = frame_paths(sp, seq)
            frames, objs, ign = read_annotations(sp, seq)
            index[seq] = dict(split=split_name(sp), n_images=len(fps), n_annotated=len(frames), hw=list(cv2.imread(fps[0]).shape[:2]),
                              n_objs=int(sum(len(v) for v in objs.values())), n_ignore_regions=int(sum(len(v) for v in ign.values())))
            if sp != "train":
                d = gt_dir(sp)
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, seq + ".json"), "w") as f:
                    json.dump(gt_frames_bdd(sp, seq), f)
        print(sp, len(list_seqs(sp)), "seqs", flush=True)
    os.makedirs(derived_dir(), exist_ok=True)
    with open(index_file(), "w") as f:
        json.dump(index, f, indent=0)
    print("VisDrone index:", len(index), "sequences ->", derived_dir(), flush=True)
    return derived_dir()
