# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""MOT17 (street cameras, pedestrians), half split, for training and the tracking-by-detection evaluation.

Data: the MOT17-train FRCNN copy of each sequence (the DPM / SDP folders hold the same images and GT):
<mot17_train>/MOT17-XX-FRCNN/{img1/000001.jpg, gt/gt.txt, seqinfo.ini} (optional faster image copy: data.mot17_images).
gt.txt line: frame, id, bb_left, bb_top, w, h, consider, class, visibility (1-based frames).

Decisions (fixed before any MOT17 training; docs/PREREGISTRATION.md)
1. Half split = the CenterTrack / ByteTrack / FairMOT 'half' convention: with n = seqLength and 0-based image index i, the
   training half is i in [0, n // 2] (frames 1 .. n//2 + 1) and the evaluation half is i in [n // 2 + 1, n - 1] (frames
   n//2 + 2 .. n). No frame is in both halves; the middle frame goes to training. 7 sequences: 2,664 training frames, 2,652
   evaluation frames. TrackEval has no built-in half convention, so the evaluation GT is written as its own MOTChallenge folder
   (write_half_gt): <gt_root>/<seq>/gt/gt.txt = every raw gt line of the evaluation half (all classes, so the official
   distractor preprocessing still works) with the frame renumbered frame - (n//2 + 1) (first evaluation frame = 1, as in
   CenterTrack's gt_val_half.txt), other columns verbatim; <gt_root>/<seq>/seqinfo.ini = the original with
   seqLength = n - (n//2 + 1); seqmaps <gt_root>/seqmaps/MOT17-halfval-{all,static,moving}.txt. Tracker outputs use the same
   renumbered frames.
2. Class mapping to the 8-class heads: MOT17 class 1 (pedestrian; in MOT17-train every class-1 line has consider = 1 and every
   other line consider = 0) -> head 'pedestrian'. All class-1 boxes are training targets whatever their visibility (the
   official evaluation scores all of them). Boxes are clipped to the frame for training (people leaving the image); the
   evaluation GT is not clipped (official). Coordinates are used as given (bb_left, bb_top are 1-based in the files; the model
   learns and outputs the same convention).
3. Distractors = classes 2 person_on_vehicle, 7 static_person, 8 distractor, 12 reflection (the TrackEval MOTChallenge
   preprocessing for MOT17). Training: their pixels are greyed out (114, the letterbox fill) in both frames of a pair before
   augmentation, except pixels inside a kept pedestrian box of that frame, so they are taught neither as pedestrian nor as
   background (visdrone.mask_regions). Other classes (vehicles 3-6, occluders 9-11, crowd 13) are not people and stay
   unmasked background, as the official evaluation counts a prediction on them as a false positive. Evaluation: the official
   TrackEval MOTChallenge preprocessing (DO_PREPROC, BENCHMARK MOT17): tracker boxes matched (IoU >= 0.5, Hungarian) to a
   distractor GT box are removed, GT = class 1 with consider != 0.
4. Training frames: the 7 training halves, every frame (2,664 frames = 166 iterations / epoch at batch 16); prev = the previous
   frame of the same half (k = 0 pairs with itself). 55 epochs (iterations matched to the DSEC / VisDrone runs). At most 52
   pedestrians per frame, so the 120-row target padding never truncates.
5. Evaluation: the 7 evaluation halves, every frame; detection caches <cache>/<tag>/mot17/<seq>.npz (frames = evaluation-half
   image names); pedestrian only; run-tag prefix 'mot_'. The static-camera sequences 02, 04, 09 (STATIC_SEQS) are reported
   separately (descriptive).
"""
from __future__ import annotations

import configparser
import os

import numpy as np

from .. import paths
from .common import BDD_CLASSES
from .pairs import FramePairs
from .visdrone import mask_regions

SEQS = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN", "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]
STATIC_SEQS = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-09-FRCNN"]
MOVING_SEQS = [s for s in SEQS if s not in STATIC_SEQS]
PED_CLASS = 1
DISTRACTOR_CLASSES = (2, 7, 8, 12)
MAX_OBJS = 120
SEQMAP = {"all": SEQS, "static": STATIC_SEQS, "moving": MOVING_SEQS}
_INFO: dict = {}


def data_root() -> str:
    """<mot17_train>: the MOT17 train folder (GT, seqinfo.ini, images)"""
    return paths.require("data.mot17_train")


def img_root() -> str:
    return paths.first_dir(paths.get("data.mot17_images"), data_root())


def gt_root() -> str:
    """<derived>/mot17/gt_halfval: the evaluation-half GT in TrackEval's MOTChallenge layout (write_half_gt)"""
    return os.path.join(paths.require("outputs.derived"), "mot17", "gt_halfval")


def seqinfo(seq: str) -> dict:
    """-> dict(n, H, W, fps) from the sequence's seqinfo.ini"""
    if seq not in _INFO:
        c = configparser.ConfigParser()
        c.read(os.path.join(data_root(), seq, "seqinfo.ini"))
        q = c["Sequence"]
        _INFO[seq] = dict(n=int(q["seqLength"]), H=int(q["imHeight"]), W=int(q["imWidth"]), fps=int(q["frameRate"]))
    return _INFO[seq]


def seq_hw(seq: str) -> tuple[int, int]:
    i = seqinfo(seq)
    return (i["H"], i["W"])


def half_offset(seq: str) -> int:
    """frames 1..off are the training half, off+1..n the evaluation half; evaluation frame f is written as f - off"""
    return seqinfo(seq)["n"] // 2 + 1


def train_frames(seq: str) -> list[int]:
    return list(range(1, half_offset(seq) + 1))


def eval_frames(seq: str) -> list[int]:
    return list(range(half_offset(seq) + 1, seqinfo(seq)["n"] + 1))


def frame_path(seq: str, fr: int) -> str:
    return os.path.join(img_root(), seq, "img1", f"{fr:06d}.jpg")


def eval_frame_paths(seq: str) -> list[str]:
    return [frame_path(seq, f) for f in eval_frames(seq)]


def eval_seqs() -> list[str]:
    return list(SEQS)


def frame_no(path) -> int:
    return int(os.path.splitext(os.path.basename(str(path)))[0])


def read_gt(seq: str, clip: bool = True):
    """reads gt/gt.txt -> (peds, distr)
    peds: {frame: float32 [N, 6] (cls_bdd = pedestrian, x1, y1, x2, y2, track_id)}: class 1 with consider = 1 and w, h > 0,
          clipped to the frame when clip (boxes with no area left inside the frame are dropped);
    distr: {frame: float32 [M, 4]} xyxy (unclipped) of the distractor classes 2, 7, 8, 12."""
    H, W = seq_hw(seq)
    p_, d_ = {}, {}
    ped = BDD_CLASSES.index("pedestrian")
    with open(os.path.join(data_root(), seq, "gt", "gt.txt")) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 8:
                continue
            fr, tid = int(p[0]), int(p[1])
            l, t, w, h = (float(v) for v in p[2:6])
            flag, cl = int(float(p[6])), int(float(p[7]))
            if cl in DISTRACTOR_CLASSES:
                d_.setdefault(fr, []).append((l, t, l + w, t + h))
                continue
            if cl != PED_CLASS or flag != 1 or w <= 0 or h <= 0:
                continue
            x1, y1, x2, y2 = l, t, l + w, t + h
            if clip:
                x1, x2 = min(max(x1, 0.0), W), min(max(x2, 0.0), W)
                y1, y2 = min(max(y1, 0.0), H), min(max(y2, 0.0), H)
                if x2 <= x1 or y2 <= y1:
                    continue
            p_.setdefault(fr, []).append((ped, x1, y1, x2, y2, tid))
    peds = {k: np.asarray(v, np.float32).reshape(-1, 6) for k, v in p_.items()}
    distr = {k: np.asarray(v, np.float32).reshape(-1, 4) for k, v in d_.items()}
    return peds, distr


def write_half_gt(root: str | None = None, seqs=None) -> str:
    """MOTChallenge GT folder of the evaluation halves (decision 1) -> the folder (default gt_root())"""
    out_root = root or gt_root()
    seqs = seqs or SEQS
    for s in seqs:
        off = half_offset(s)
        n = seqinfo(s)["n"]
        d = os.path.join(out_root, s, "gt")
        os.makedirs(d, exist_ok=True)
        out = []
        with open(os.path.join(data_root(), s, "gt", "gt.txt")) as f:
            for line in f:
                p = line.strip().split(",")
                if len(p) < 8 or int(p[0]) <= off:
                    continue
                out.append(",".join([str(int(p[0]) - off)] + p[1:]))
        with open(os.path.join(d, "gt.txt"), "w") as f:
            f.write("\n".join(out) + "\n")
        with open(os.path.join(data_root(), s, "seqinfo.ini")) as f:
            ini = f.read()
        lines = [(f"seqLength={n - off}" if ln.strip().lower().startswith("seqlength") else ln) for ln in ini.splitlines()]
        with open(os.path.join(out_root, s, "seqinfo.ini"), "w") as f:
            f.write("\n".join(lines) + "\n")
    os.makedirs(os.path.join(out_root, "seqmaps"), exist_ok=True)
    for k, v in SEQMAP.items():
        with open(os.path.join(out_root, "seqmaps", f"MOT17-halfval-{k}.txt"), "w") as f:
            f.write("name\n" + "".join(q + "\n" for q in v if q in seqs))
    return out_root


def seqmap_file(subset: str = "all", root: str | None = None) -> str:
    return os.path.join(root or gt_root(), "seqmaps", f"MOT17-halfval-{subset}.txt")


class Mot17Pairs(FramePairs):
    """frame pairs (cur, prev = previous frame of the training half) in the BDD class space; labels / distractor boxes keyed
    by k = position in the training half (frame = k + 1); distractor pixels greyed out (decision 3)"""

    def __init__(self, seqs=None, size_hw=(256, 416), train=True, max_objs=MAX_OBJS, stride=1, mask_distractors=True):
        seqs = SEQS if seqs is None else seqs
        self.items, self.labels, self.ignores, self.frames = [], {}, {}, {}
        for s in seqs:
            frames = train_frames(s)
            peds, distr = read_gt(s)
            self.frames[s] = frames
            self.labels[s] = {k: peds.get(f, np.zeros((0, 6), np.float32)) for k, f in enumerate(frames)}
            self.ignores[s] = {k: distr[f] for k, f in enumerate(frames) if f in distr}
            for k in range(0, len(frames), stride):
                self.items.append((s, k, frame_path(s, frames[k]), frame_path(s, frames[max(k - 1, 0)])))
        self.size, self.train, self.max_objs = size_hw, train, max_objs
        self.mask_distractors = mask_distractors

    def _load(self, seq, key, path):
        import cv2
        img = cv2.imread(path)
        reg = self.ignores[seq].get(key) if self.mask_distractors else None
        if reg is not None and len(reg):
            img = mask_regions(img, reg, self.labels[seq].get(key, np.zeros((0, 6), np.float32))[:, 1:5])
        return img


def build_index() -> str:
    """write_half_gt() into <derived>/mot17/gt_halfval and print the split per sequence"""
    out = write_half_gt()
    for s in SEQS:
        n = seqinfo(s)["n"]
        off = half_offset(s)
        print(s, f"n {n} train frames 1..{off} ({off}) eval frames {off + 1}..{n} ({n - off})", flush=True)
    print("MOT17 half split:", sum(half_offset(s) for s in SEQS), "train frames,",
          sum(seqinfo(s)["n"] - half_offset(s) for s in SEQS), "eval frames ->", out, flush=True)
    return out
