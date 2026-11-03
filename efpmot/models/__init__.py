# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""efpmot.models: S3x / S3y anisotropic-stride P3 for YOLOv8n and YOLO11n, per-axis loss, adapters, build().

    from efpmot.models import build
    ad = build("yolov8n_s3x", 8, "coco")                                  # start of a training run
    ad = build("yolov8n_s3x", 8, "release_assets/yolov8n_s3x_dsec_416x320_s0.pt")  # a trained checkpoint
    boxes, scores = ad.infer(x)   # x: BGR 0..255 float [B, 3, H, W] letterboxed canvas -> xyxy canvas px, sigmoid scores

Importing this package registers torch.nn.AUpsample (needed before any S3 model dict is parsed).
"""
from __future__ import annotations

import copy
import pickle
import warnings

import torch

from ..release import release_meta

from . import aniso, aniso_y11  # noqa: F401  (registers torch.nn.AUpsample)
from .adapters import Adapter, V8Adapter, V8AnisoAdapter, Y11Adapter, iou_xyxy  # noqa: F401

FAMILIES = ("yolov8n", "yolov8n_s3x", "yolov8n_s3y", "yolo11n", "yolo11n_s3x", "yolo11n_s3y")
DET_TAG = {"yolov8n": "v8n", "yolov8n_s3x": "v8n", "yolov8n_s3y": "v8n", "yolo11n": "y11n", "yolo11n_s3x": "y11n", "yolo11n_s3y": "y11n"}
_AXIS = {"yolov8n": None, "yolov8n_s3x": "x", "yolov8n_s3y": "y", "yolo11n": None, "yolo11n_s3x": "x", "yolo11n_s3y": "y"}


def make_adapter(family: str, nc: int = 8, coco: bool = True) -> Adapter:
    """family -> adapter; coco: start from the Ultralytics COCO checkpoint (else random init)"""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; choose from {FAMILIES}")
    w = "coco" if coco else None
    axis = _AXIS[family]
    if family.startswith("yolo11n"):
        return Y11Adapter(nc, axis, w)
    return V8Adapter(nc, w) if axis is None else V8AnisoAdapter(nc, axis, w)


def load_checkpoint(adapter: Adapter, path: str, strict: bool = True) -> dict:
    """load {"model": state_dict, ...} (training checkpoint) or a bare state dict (release file) into adapter.model;
    returns the other keys (metadata). Every file is read with weights_only=True first. Only a training checkpoint that
    is not a release file and fails with pickle.UnpicklingError (its training args hold non-tensor objects) falls back to
    the full unpickler, with a warning: only load such files when you trust them."""
    ck = load_trusted(path)
    if isinstance(ck, dict) and "model" in ck and isinstance(ck["model"], dict):
        sd, meta = ck["model"], {k: v for k, v in ck.items() if k != "model"}
    else:
        sd, meta = ck, {}
    adapter.model.load_state_dict(sd, strict=strict)
    return meta


def load_trusted(path: str):
    """torch.load on CPU with weights_only=True; for a file that is not a release file (efpmot.release) and only fails on
    pickle.UnpicklingError, retry with weights_only=False after a warning. A release file never takes the unsafe path."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        if release_meta(path) is not None:
            raise
        warnings.warn(f"{path} does not load with weights_only=True; reading it with the full unpickler "
                      "(only do this for checkpoints you trust)", stacklevel=2)
        return torch.load(path, map_location="cpu", weights_only=False)


def build(family: str, nc: int = 8, weights: str | None = None) -> Adapter:
    """weights: None = random init; "coco" = start from the Ultralytics COCO checkpoint (paths weights.yolov8n / yolo11n; by
    name + shape, the S3 level-0 box branch sliced to its leading channels, every class branch fresh); any other str = a
    trained checkpoint (training last.pth or release file), loaded with strict=True."""
    if weights == "coco":
        return make_adapter(family, nc, coco=True)
    ad = make_adapter(family, nc, coco=False)
    if weights is not None:
        load_checkpoint(ad, weights, strict=True)
    return ad


def flops(family: str, hw: tuple[int, int] = (320, 416), nc: int = 8) -> float:
    """GFLOPs = 2 x thop MACs of the eval graph on a 1x3xHxW input. At (320, 416), nc 8: yolov8n 2.665, yolov8n_s3x 2.955
    (+10.9 %), yolo11n 2.101, yolo11n_s3x 2.333 (+11.0 %)."""
    import thop
    m = make_adapter(family, nc, coco=False).model
    fl, _ = thop.profile(copy.deepcopy(m).eval(), inputs=[torch.zeros(1, 3, *hw)], verbose=False)
    return 2 * fl / 1e9
