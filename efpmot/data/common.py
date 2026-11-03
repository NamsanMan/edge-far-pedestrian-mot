# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Shared constants and image helpers: the 8-class BDD100K order of every detector head, letterbox, box IoU."""
from __future__ import annotations

import cv2
import numpy as np

# class order of every detector head (all four datasets): pedestrian = 0, car = 2
BDD_CLASSES = ["pedestrian", "rider", "car", "bus", "truck", "train", "motorcycle", "bicycle"]
PED = BDD_CLASSES.index("pedestrian")
CAR = BDD_CLASSES.index("car")


def letterbox(img: np.ndarray, size_hw, fill: int = 114) -> tuple[np.ndarray, float]:
    """resize keeping the aspect ratio into an (H, W) canvas: image at the top-left, padding (fill) at the bottom / right.
    -> (canvas uint8 [H, W, 3], scale r); canvas px = native px * r"""
    h, w = img.shape[:2]
    H, W = size_hw
    r = min(H / h, W / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    out = np.full((H, W, 3), fill, np.uint8)
    out[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a [N, 4], b [M, 4] xyxy -> IoU [N, M]"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)
