# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""DSEC-Det: RGB frames remapped to the 640x480 event-camera view, with the DSEC-Detection track labels, in the BDD class space.

Layout (configs/paths.yaml data.dsec_det; docs/DATA.md):
  <dsec_det>/images/{train,test}/<seq>/images/left/distorted/*.png       (optional faster copy: data.dsec_images)
  <dsec_det>/labels/{train,test}/<seq>/object_detections/left/tracks.npy
  <dsec_det>/meta/{train,test}/<seq>/images/timestamps.txt               (maps label timestamps to image indices)
Splits: the official DSEC-Det train / val / test split (splits/dsec_det_split.json; val sequences live in the train folders).
build_index() writes <derived>/dsec_det/frame_index.json and <derived>/dsec_det/gt_bdd/<seq>.json (BDD / Scalabel format for
TrackEval's BDD100K evaluator). Only labelled frames (timestamps present in tracks.npy) are training targets and enter the GT;
unlabelled frames are long contiguous runs without annotation and are excluded from evaluation. The previous frame (i - 1) of a
training pair may be unlabelled (it is an input only).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from .. import paths
from .common import BDD_CLASSES
from .pairs import FramePairs

DSEC_CLASSES = ("pedestrian", "rider", "car", "bus", "truck", "bicycle", "motorcycle", "train")  # tracks.npy class_id order
SPLIT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits", "dsec_det_split.json")
NATIVE_HW = (480, 640)
_index: dict | None = None


def root() -> str:
    return paths.require("data.dsec_det")


def image_root() -> str:
    return paths.first_dir(paths.get("data.dsec_images"), os.path.join(root(), "images"))


def derived_dir() -> str:
    return os.path.join(paths.require("outputs.derived"), "dsec_det")


def gt_dir() -> str:
    return os.path.join(derived_dir(), "gt_bdd")


def frame_index() -> dict:
    """{seq: {split, folder, group, n_images, labelled: [image indices]}} from <derived>/dsec_det/frame_index.json (cached)"""
    global _index
    if _index is None:
        with open(os.path.join(derived_dir(), "frame_index.json")) as f:
            _index = json.load(f)
    return _index


def split_seqs(split: str) -> list[str]:
    """split: train | val | test (sorted sequence names)"""
    with open(SPLIT_JSON) as f:
        d = json.load(f)
    return sorted(d["summary"][split]["seqs"])


def frame_paths(seq: str) -> list[str]:
    folder = frame_index()[seq]["folder"]
    return sorted(glob.glob(os.path.join(image_root(), folder, seq, "images", "left", "distorted", "*.png")))


def read_labels(seq: str) -> dict:
    """-> {image_index: [[cls_bdd, x1, y1, x2, y2, track_id], ...]} (labelled frames only)"""
    out = {}
    with open(os.path.join(gt_dir(), seq + ".json")) as f:
        frames = json.load(f)
    for fr in frames:
        out[fr["image_index"]] = [[BDD_CLASSES.index(lb["category"]), lb["box2d"]["x1"], lb["box2d"]["y1"], lb["box2d"]["x2"],
                                   lb["box2d"]["y2"], lb["id"]] for lb in fr["labels"]]
    return out


def labelled_names(seq: str) -> set[str]:
    return {f"{i:06d}.png" for i in frame_index()[seq]["labelled"]}


def eval_seqs() -> list[str]:
    """the detection-export set: val + test (19 sequences); the TrackEval subset is test (13)"""
    return split_seqs("val") + split_seqs("test")


def eval_frame_paths(seq: str) -> list[str]:
    return frame_paths(seq)


def native_hw(seq: str | None = None) -> tuple[int, int]:
    return NATIVE_HW


class DsecPairs(FramePairs):
    """labelled frames of the given sequences (every stride-th labelled frame) paired with the previous image"""

    def __init__(self, seqs, size_hw=(480, 640), train=True, max_objs=120, stride=1):
        self.items = []
        self.labels = {}
        for s in seqs:
            frames = frame_paths(s)
            self.labels[s] = read_labels(s)
            lab = sorted(self.labels[s])
            for i in lab[::stride]:
                self.items.append((s, i, frames[i], frames[max(i - 1, 0)]))
        self.size, self.train, self.max_objs = size_hw, train, max_objs


def build_index(out_root: str | None = None) -> str:
    """frame index + BDD-format GT from tracks.npy and the image timestamps -> <out_root or <derived>/dsec_det>"""
    out = out_root or derived_dir()
    with open(SPLIT_JSON) as f:
        d = json.load(f)
    seqs = d["sequences"]
    split_of = {s: k for k in ("train", "val", "test") for s in d["summary"][k]["seqs"]}
    os.makedirs(os.path.join(out, "gt_bdd"), exist_ok=True)
    index = {}
    tot_fr = tot_box = 0
    for seq, st in sorted(seqs.items()):
        sp = split_of[seq]
        folder = "train" if sp in ("train", "val") else "test"
        T = np.genfromtxt(os.path.join(root(), "meta", folder, seq, "images", "timestamps.txt"), dtype="int64")
        tr = np.load(os.path.join(root(), "labels", folder, seq, "object_detections", "left", "tracks.npy"))
        n_img = len(glob.glob(os.path.join(image_root(), folder, seq, "images", "left", "distorted", "*.png")))
        assert n_img == len(T), (seq, n_img, len(T))
        lt = np.unique(tr["t"])
        idx = np.searchsorted(T, lt)
        assert (T[idx] == lt).all(), seq
        by_t = {int(t): [] for t in lt}
        for r in tr:
            by_t[int(r["t"])].append(r)
        frames = []
        for fi, (t, i) in enumerate(zip(lt, idx)):
            labels = [{"id": int(r["track_id"]), "category": DSEC_CLASSES[int(r["class_id"])],
                       "attributes": {"Occluded": False, "Truncated": False, "Crowd": False},
                       "box2d": {"x1": float(r["x"]), "y1": float(r["y"]), "x2": float(r["x"] + r["w"]), "y2": float(r["y"] + r["h"])}}
                      for r in by_t[int(t)]]
            frames.append({"name": f"{int(i):06d}.png", "video_name": seq, "index": fi, "image_index": int(i), "t": int(t),
                           "labels": labels})
            tot_box += len(labels)
        tot_fr += len(frames)
        with open(os.path.join(out, "gt_bdd", seq + ".json"), "w") as f:
            json.dump(frames, f)
        index[seq] = dict(split=sp, folder=folder, group=st.get("group", "day"), n_images=int(len(T)), labelled=[int(x) for x in idx])
    with open(os.path.join(out, "frame_index.json"), "w") as f:
        json.dump(index, f)
    global _index
    _index = None
    print("DSEC index:", len(index), "sequences,", tot_fr, "labelled frames,", tot_box, "boxes ->", out, flush=True)
    return out
