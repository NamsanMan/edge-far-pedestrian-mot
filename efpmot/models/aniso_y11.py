# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
# Portions adapted from Ultralytics YOLO 8.3.170 (AGPL-3.0), https://github.com/ultralytics/ultralytics
"""YOLO11n port of the anisotropic-stride P3 level (efpmot.models.aniso).

- s3_dict(nc, axis): yolo11n.yaml (scale n) with the four S3 edits at the YOLO11 layer indices (Ultralytics 8.3.170
  cfg/models/11/yolo11.yaml): layer 3 Conv P2 -> P3 stride s23, layer 5 Conv P3 -> P4 stride s34, layer 14 nn.Upsample
  P4 -> P3 -> nn.AUpsample scale s34, layer 17 Conv P3 -> P4 (bottom-up) stride s34, with (s23, s34) = aniso.AXES[axis].
  At 416x320: S3x levels 40x104 (P3, stride 8 (y) x 4 (x)), 20x26, 10x13; S3y 80x52 (stride 4 (y) x 8 (x)), 20x26, 10x13.
- AnisoDetect11: AnisoDetect with the YOLO11 head layout (legacy = False): box branch Conv 3x3, Conv 3x3, Conv2d 1x1 (as v8);
  class branch [DWConv 3x3 + Conv 1x1], [DWConv 3x3 + Conv 1x1], Conv2d 1x1. A per-level width c thins BOTH branches of that
  level (None = plain YOLO11 widths: box 64, class max(ch, min(nc, 100))). Per-axis stride_xy / anchors / decode / bias_init
  are inherited; the loss is aniso.AnisoV8Loss unchanged (YOLO11 shares the Detect output layout).
- HEAD_C, the P3 head width of S3x / S3y, follows a rule fixed before measuring: among HEAD_CANDIDATES, the width whose S3x
  FLOPs overhead over plain yolo11n (2 x thop MACs, eval graph, 1x3x320x416, nc 8) is closest to the v8n S3x overhead
  (+10.9 %: 2.955 vs 2.665 GF); ties go to the wider head. Plain yolo11n 2.101 GF; S3x with c = 16 / 24 / 32 / 48 / 64:
  +7.4 / +11.0 / +15.2 / +24.9 / +36.7 % -> HEAD_C = 24 (S3x 2.333 GF; S3y costs the same).
- COCO: yolo11n.pt by name + shape. Backbone / neck tensors all match (the strided convs keep their 3x3 shapes); the class
  branch of every level stays fresh except its first DWConv (shape [ch, 1, 3, 3], independent of nc), in the plain and the S3
  model alike. The level-0 box branch of the S3 model (model.23.cv2.0.*) takes the leading channels of the COCO tensors as in
  aniso.load_coco: Conv(64 -> c) keeps filters 0..c-1 (+ their BN), Conv(c -> c) the leading c x c block, the DFL
  Conv2d(c -> 64) all 64 bins over inputs 0..c-1 (11 sliced tensors for c < 64). The slice is restricted to the Detect prefix.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv, Detect, DWConv

from .aniso import AXES, AnisoDetect, install_det

HEAD_CANDIDATES = (16, 24, 32, 48, 64)
V8_S3X_OVERHEAD = 2.955 / 2.665 - 1  # v8n S3x over plain v8n (thop, 416x320, nc 8)
HEAD_C = 24
DET_I = 23  # index of Detect in yolo11n (Detect(P3 16, P4 19, P5 22))


def _base(nc: int) -> dict:
    from ultralytics.nn.tasks import yaml_model_load
    d = copy.deepcopy(yaml_model_load("yolo11n.yaml"))
    d["scale"] = "n"
    d["nc"] = nc
    return d


def plain_dict(nc: int = 8) -> dict:
    return _base(nc)


def s3_dict(nc: int = 8, axis: str = "x") -> dict:
    d = _base(nc)
    bb, hd = d["backbone"], d["head"]
    s23, s34 = AXES[axis]
    assert bb[3][2] == "Conv" and bb[5][2] == "Conv" and hd[3][2] == "nn.Upsample" and hd[6][2] == "Conv", "yolo11.yaml layout changed"
    bb[3] = [-1, 1, "Conv", [256, 3, s23]]                   # 3: P2 -> P3 (stride 8 (y) x 4 (x) for 'x')
    bb[5] = [-1, 1, "Conv", [512, 3, s34]]                   # 5: P3 -> P4 (16, 16)
    hd[3] = [-1, 1, "nn.AUpsample", [None, s34, "nearest"]]  # 14: P4 -> P3
    hd[6] = [-1, 1, "Conv", [256, 3, s34]]                   # 17: P3 -> P4 (bottom-up)
    return d


def pick_head_c(table: dict, target: float = V8_S3X_OVERHEAD) -> int:
    """the head-width rule: table {c: S3x FLOPs overhead over plain yolo11n (fraction)} -> c closest to target; ties -> wider"""
    return min(table, key=lambda c: (round(abs(table[c] - target), 9), -c))


class AnisoDetect11(AnisoDetect):
    """AnisoDetect with the YOLO11 head layout (legacy = False) and per-level head widths (None = plain YOLO11 widths)"""
    legacy = False  # YOLO11 class branch (DWConv + Conv 1x1, twice), whatever parse_model last set on Detect

    def __init__(self, nc=80, ch=(), widths=None):
        Detect.__init__(self, nc, ch)  # plain YOLO11 heads, then the thinned levels replaced
        self.widths = list(widths) if widths is not None else [None] * self.nl
        for L, (x, c) in enumerate(zip(ch, self.widths)):
            if c:
                self.cv2[L] = nn.Sequential(Conv(x, c, 3), Conv(c, c, 3), nn.Conv2d(c, 4 * self.reg_max, 1))
                self.cv3[L] = nn.Sequential(nn.Sequential(DWConv(x, x, 3), Conv(x, c, 1)),
                                            nn.Sequential(DWConv(c, c, 3), Conv(c, c, 1)), nn.Conv2d(c, self.nc, 1))
        self.register_buffer("stride_xy", torch.ones(self.nl, 2), persistent=False)  # (sx, sy) per level; set by install_det()


def build(d: dict, nc: int, widths, probe: int = 256):
    """DetectionModel from model dict d with its Detect replaced by AnisoDetect11(widths)"""
    from ultralytics.nn.tasks import DetectionModel
    m = DetectionModel(d, nc=nc, verbose=False)
    assert m.model[-1].i == DET_I, m.model[-1].i
    det = AnisoDetect11(nc, tuple(s[0].conv.in_channels for s in m.model[-1].cv2), widths)
    return install_det(m, det, "efpmot.models.aniso_y11.AnisoDetect11", probe)


def _coco_sd(weights: str) -> dict:
    return torch.load(weights, map_location="cpu", weights_only=False)["model"].float().state_dict()


def load_coco(m, weights: str | None, tag: str = "", sd: dict | None = None):
    """COCO checkpoint by name + shape; tensors of a thinned level-0 box branch (model.23.cv2.0.*) take the leading channels.
    sd: a yolo11n state dict instead of the checkpoint file. -> (m, loaded keys, sliced keys)"""
    sd = _coco_sd(weights) if sd is None else sd
    msd = m.state_dict()
    keep, sl = {}, []
    pre = f"model.{DET_I}.cv2.0."
    for k, v in msd.items():
        if k not in sd:
            continue
        if sd[k].shape == v.shape:
            keep[k] = sd[k]
        elif k.startswith(pre) and sd[k].ndim == v.ndim and all(a_ >= b_ for a_, b_ in zip(sd[k].shape, v.shape)):
            keep[k] = sd[k][tuple(slice(0, n_) for n_ in v.shape)].clone()
            sl.append(k)
    m.load_state_dict(keep, strict=False)
    print(f"{tag}: loaded {len(keep)}/{len(msd)} tensors ({len(sl)} level-0 box-branch tensors sliced)", flush=True)
    return m, sorted(keep), sl


def y11_model(nc: int, weights: str | None = None, sd: dict | None = None):
    """plain YOLO11n (Ultralytics Detect); COCO weights by name + shape"""
    from ultralytics.nn.tasks import DetectionModel
    m = DetectionModel(plain_dict(nc), nc=nc, verbose=False)
    if weights or sd is not None:
        sd = _coco_sd(weights) if sd is None else sd
        msd = m.state_dict()
        keep = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
        m.load_state_dict(keep, strict=False)
        print(f"yolo11n: loaded {len(keep)}/{len(msd)} tensors", flush=True)
    return m


def s3_model(nc: int, axis: str, weights: str | None = None, head_c: int = HEAD_C, sd: dict | None = None):
    """YOLO11n S3x ('x') / S3y ('y') with a head_c-wide P3 head; COCO weights when a checkpoint path (or sd) is given"""
    m = build(s3_dict(nc, axis), nc, (head_c, None, None))
    if weights or sd is not None:
        load_coco(m, weights, f"yolo11n-s3{axis}", sd)
    return m
