# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
# Portions adapted from Ultralytics YOLO 8.3.170 (AGPL-3.0), https://github.com/ultralytics/ultralytics
"""Detector adapters: one interface over plain / S3x / S3y YOLOv8n and YOLO11n for training, decoding and export.

Every adapter holds an Ultralytics DetectionModel (`model`, with model.args = the default Ultralytics config, which supplies
the loss gains box 7.5 / cls 0.5 / dfl 1.5), the AdamW learning rate of the Ultralytics 'auto' optimiser rule
lr = round(0.002 * 5 / (4 + nc), 6) and weight decay 5e-4. Inputs are BGR float tensors 0..255 [B, 3, H, W] on the letterboxed
canvas; norm() turns them into RGB / 255. Targets are [B, max_objs, 6] (cls, cx, cy, w, h, track id) in canvas pixels, with
padding rows of zero width / height.
"""
from __future__ import annotations

import os

import torch

from .. import paths


def iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a [N, 4], b [M, 4] xyxy -> IoU [N, M]"""
    x1 = torch.maximum(a[:, None, 0], b[None, :, 0])
    y1 = torch.maximum(a[:, None, 1], b[None, :, 1])
    x2 = torch.minimum(a[:, None, 2], b[None, :, 2])
    y2 = torch.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-6)


def coco_weights(key: str) -> str:
    """path of a COCO checkpoint from configs/paths.yaml (weights.yolov8n | weights.yolo11n); it must exist"""
    w = paths.require(f"weights.{key}")
    if not os.path.isfile(w):
        raise FileNotFoundError(f"{w} missing: training starts from the Ultralytics COCO checkpoint {key}.pt (docs/DATA.md)")
    return w


class Adapter:
    """shared interface; subclasses set model, ncls and aniso (True: AnisoDetect head + AnisoV8Loss)"""
    aniso = False

    def _finish(self, ncls: int, loss_cls):
        from ultralytics.cfg import get_cfg
        from ultralytics.utils import DEFAULT_CFG
        self.ncls = ncls
        self.model.args = get_cfg(DEFAULT_CFG)
        self._loss_cls = loss_cls
        self.crit = None
        self.lr, self.wd = round(0.002 * 5 / (4 + ncls), 6), 5e-4  # Ultralytics 'auto' optimiser rule (AdamW)

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, [2, 1, 0]] / 255.0

    def train_forward(self, x: torch.Tensor, tgt: torch.Tensor):
        """-> (loss scalar, loss items {box, cls, dfl}, raw per-level head outputs)"""
        if self.crit is None:
            self.crit = self._loss_cls(self.model)
        if self.aniso:
            self.crit.img_hw = tuple(x.shape[2:])  # image size from the batch tensor
        feats = self.model(self.norm(x))
        B, _, H, W = x.shape
        m = (tgt[..., 3] > 0) & (tgt[..., 4] > 0)
        bi = torch.nonzero(m)[:, 0].float()
        t = tgt[m]
        batch = {"batch_idx": bi, "cls": t[:, 0:1], "bboxes": torch.stack([t[:, 1] / W, t[:, 2] / H, t[:, 3] / W, t[:, 4] / H], 1)}
        loss, items = self.crit(feats, batch)
        return loss.sum() / B, {"box": float(items[0]), "cls": float(items[1]), "dfl": float(items[2])}, feats

    def decode_feats(self, feats):
        """raw head outputs -> (boxes xyxy [B, A, 4] canvas px, scores [B, A, C] sigmoid)"""
        y = self.model.model[-1]._inference(feats).permute(0, 2, 1)  # [B, A, 4 + C], xywh canvas px
        xy, wh = y[..., :2], y[..., 2:4]
        return torch.cat([xy - wh / 2, xy + wh / 2], -1), y[..., 4:]

    def infer(self, x: torch.Tensor):
        """eval-mode forward -> (boxes xyxy [B, A, 4] canvas px, scores [B, A, C] sigmoid)"""
        self.model.eval()
        y = self.model(self.norm(x))
        y = (y[0] if isinstance(y, (list, tuple)) else y).permute(0, 2, 1)
        xy, wh = y[..., :2], y[..., 2:4]
        return torch.cat([xy - wh / 2, xy + wh / 2], -1), y[..., 4:]


class V8Adapter(Adapter):
    """plain Ultralytics YOLOv8n (anchor-free, no objectness, TAL assigner, DFL); COCO weights by name + shape (the class
    branch of every level stays fresh: its width depends on nc)"""

    def __init__(self, ncls: int, weights: str | None = "coco"):
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils.loss import v8DetectionLoss
        self.model = DetectionModel("yolov8n.yaml", nc=ncls, verbose=False)
        if weights == "coco":
            ck = torch.load(coco_weights("yolov8n"), map_location="cpu", weights_only=False)
            sd = ck["model"].float().state_dict()
            msd = self.model.state_dict()
            keep = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
            self.model.load_state_dict(keep, strict=False)
            print(f"yolov8n: loaded {len(keep)}/{len(msd)} tensors", flush=True)
        self._finish(ncls, v8DetectionLoss)


class V8AnisoAdapter(Adapter):
    """YOLOv8n with the anisotropic-stride P3 level (aniso.s3_model): axis 'x' = S3x, 'y' = S3y"""
    aniso = True

    def __init__(self, ncls: int, axis: str, weights: str | None = "coco"):
        from . import aniso
        self.axis = axis
        self.model = aniso.s3_model(ncls, axis, coco_weights("yolov8n") if weights == "coco" else None)
        self._finish(ncls, aniso.AnisoV8Loss)


class Y11Adapter(Adapter):
    """Ultralytics YOLO11n: axis None = plain yolo11n (Detect, v8DetectionLoss); 'x' / 'y' = S3x / S3y (P3 head
    aniso_y11.HEAD_C channels, AnisoV8Loss). Same lr / wd rule, norm, decode and export as V8Adapter."""

    def __init__(self, ncls: int, axis: str | None = None, weights: str | None = "coco"):
        from ultralytics.utils.loss import v8DetectionLoss
        from . import aniso, aniso_y11
        w = coco_weights("yolo11n") if weights == "coco" else None
        self.axis, self.aniso = axis, axis is not None
        self.model = aniso_y11.y11_model(ncls, w) if axis is None else aniso_y11.s3_model(ncls, axis, w)
        self._finish(ncls, v8DetectionLoss if axis is None else aniso.AnisoV8Loss)
