# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Frame-pair training set shared by every dataset: (current frame, previous frame) with the same augmentation for both.

Each item is a pair because the training data is organised by tracking sequences; the detector loss uses the current frame
(the previous frame and its targets are returned as well, so the same loader serves temporal objectives). Single-image
datasets (BDD100K detection) pair an image with itself.
"""
from __future__ import annotations

import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .common import letterbox


class FramePairs(Dataset):
    """Subclasses fill, in __init__:
    items  [(seq, key, cur_path, prev_path)]; key indexes labels[seq] (the previous frame uses key max(key - 1, 0));
    labels {seq: {key: rows (cls, x1, y1, x2, y2, track_id)}} in native image pixels;
    size (H, W) canvas, train (augmentation on / off), max_objs (target rows per frame).
    __getitem__ -> (cur, prev, tgt, ptgt): float32 BGR [3, H, W] 0..255 twice, then [max_objs, 6] (cls, cx, cy, w, h, id;
    padding rows: zeros with id -1) twice.

    Augmentation (train=True), drawn in this order for every item: a horizontal flip of both frames with probability 0.5
    (random.random() < 0.5), a scale factor sc = random.uniform(0.8, 1.2) applied to both frames when |sc - 1| > 0.02, one
    brightness gain np.random.uniform(0.7, 1.3) for both frames; then the letterbox. Boxes are scaled to the canvas, clipped,
    and kept when wider and higher than 1 px.

    load_prev = False (the training harness sets it): the previous frame is neither read nor augmented, and the current
    frame and its targets are returned in its place. The random draws above do not depend on the previous frame, so the
    current frame, its targets and the random state after each item are the same either way.
    """
    load_prev = True
    items: list
    labels: dict
    size: tuple
    train: bool
    max_objs: int

    def __len__(self):
        return len(self.items)

    def _pack(self, boxes, cls, ids) -> np.ndarray:
        """-> [max_objs, 6]: cls, cx, cy, w, h, track_id (-1 pad)"""
        tgt = np.zeros((self.max_objs, 6), np.float32)
        tgt[:, 5] = -1
        n = min(len(boxes), self.max_objs)
        if n:
            tgt[:n, 0] = cls[:n]
            tgt[:n, 1] = (boxes[:n, 0] + boxes[:n, 2]) / 2
            tgt[:n, 2] = (boxes[:n, 1] + boxes[:n, 3]) / 2
            tgt[:n, 3] = boxes[:n, 2] - boxes[:n, 0]
            tgt[:n, 4] = boxes[:n, 3] - boxes[:n, 1]
            tgt[:n, 5] = ids[:n]
        return tgt

    def _load(self, seq, key, path) -> np.ndarray:
        """image read hook: cv2.imread; VisDrone / MOT17 override it to grey out ignored / distractor regions.
        key = the item's key (current frame) or max(key - 1, 0) (previous frame)."""
        return cv2.imread(path)

    def __getitem__(self, idx):
        s, i, cur_p, prev_p = self.items[idx]
        two = self.load_prev
        cur = self._load(s, i, cur_p)
        prev = self._load(s, max(i - 1, 0), prev_p) if two else cur
        lab_c = self.labels[s].get(i, [])
        lab_p = self.labels[s].get(max(i - 1, 0), []) if two else []
        boxes = np.array([b[1:5] for b in lab_c], np.float32).reshape(-1, 4)
        cls = np.array([b[0] for b in lab_c], np.float32)
        ids = np.array([b[5] for b in lab_c], np.float32)
        pboxes = np.array([b[1:5] for b in lab_p], np.float32).reshape(-1, 4)
        pcls = np.array([b[0] for b in lab_p], np.float32)
        pids = np.array([b[5] for b in lab_p], np.float32)
        if self.train:  # the same flip, scale and brightness for both frames
            if random.random() < 0.5:
                cur = cur[:, ::-1]
                prev = prev[:, ::-1] if two else cur
                w = cur.shape[1]
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                pboxes[:, [0, 2]] = w - pboxes[:, [2, 0]]
            sc = random.uniform(0.8, 1.2)
            if abs(sc - 1) > 0.02:
                cur = cv2.resize(cur, None, fx=sc, fy=sc)
                prev = cv2.resize(prev, None, fx=sc, fy=sc) if two else cur
                boxes *= sc
                pboxes *= sc
            gain = np.random.uniform(0.7, 1.3)
            cur = np.clip(cur.astype(np.float32) * gain, 0, 255).astype(np.uint8)
            prev = np.clip(prev.astype(np.float32) * gain, 0, 255).astype(np.uint8) if two else cur
        cur, r = letterbox(np.ascontiguousarray(cur), self.size)
        prev = letterbox(np.ascontiguousarray(prev), self.size)[0] if two else cur
        H, W = self.size

        def clipkeep(b, c, t):
            b = b * r
            b[:, [0, 2]] = b[:, [0, 2]].clip(0, W)
            b[:, [1, 3]] = b[:, [1, 3]].clip(0, H)
            k = ((b[:, 2] - b[:, 0]) > 1) & ((b[:, 3] - b[:, 1]) > 1)
            return b[k], c[k], t[k]
        boxes, cls, ids = clipkeep(boxes, cls, ids)
        pboxes, pcls, pids = clipkeep(pboxes, pcls, pids)
        tgt = self._pack(boxes, cls, ids)
        ptgt = self._pack(pboxes, pcls, pids)

        def to_t(im):
            return torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))).float()
        if not two:
            c, t = to_t(cur), torch.from_numpy(tgt)
            return c, c, t, t
        return to_t(cur), to_t(prev), torch.from_numpy(tgt), torch.from_numpy(ptgt)
