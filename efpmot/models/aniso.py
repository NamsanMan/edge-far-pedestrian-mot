# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
# Portions adapted from Ultralytics YOLO 8.3.170 (AGPL-3.0), https://github.com/ultralytics/ultralytics
"""Anisotropic-stride P3 level (S3) for YOLOv8n: the P3 level runs at a finer stride along ONE axis, at full P3 depth.

- s3_dict("x") = S3x: layer 3 Conv stride (2, 1) -> P3 grid at stride 8 (y) x 4 (x); the P3 -> P4 convs (layer 5 and head
  layer 16) use stride (2, 4) and the P4 -> P3 upsample (layer 13) scale (2, 4), so P4 / P5 keep their isotropic 16 / 32 grid.
  s3_dict("y") = S3y, the transposed control (stride 4 (y) x 8 (x)).
  The P3 head is thinned to 32 channels (box and class branches), so S3x costs 2.955 GFLOPs at 416x320 (plain v8n 2.665 GF
  with nc = 8, +10.9 %).
- AnisoDetect: per-level strides stride_xy [nl, 2] = (sx, sy), measured from a probe forward (sx = W_in / W_level,
  sy = H_in / H_level); anchors per axis; decode dbox = dist2bbox(dfl(box), anchors) * (sx, sy, sx, sy); bias_init with the
  isotropic-equivalent stride sqrt(sx * sy); a per-level head width.
- AnisoV8Loss: the Ultralytics 8.3.170 v8DetectionLoss call with stride tensors [A, 2] (anchor points) / [A, 4] (boxes), image
  size taken from the input batch, CIoU on pixel boxes (CIoU is not invariant to a per-axis scaling of the box coordinates),
  DFL targets in per-axis grid units.
- COCO yolov8n weights load by name + shape (the strided convs keep their 3x3 kernel shapes); the thinned level-0 box branch
  takes the leading 32 channels of the COCO tensors. The class branch stays freshly initialised at every level, as in the plain
  v8n load (the COCO class width 80 differs from 64 for nc = 8).

Importing this module registers torch.nn.AUpsample (model-YAML name 'nn.AUpsample'); nothing else is patched.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv, Detect
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist


class AUpsample(nn.Upsample):
    """nn.Upsample with a per-axis (y, x) scale tuple; exports to ONNX as Resize with scales (1, 1, sy, sx)"""

    def __init__(self, size=None, scale_factor=None, mode="nearest"):
        super().__init__(size, tuple(float(v) for v in scale_factor) if isinstance(scale_factor, (list, tuple)) else scale_factor,
                         mode)


nn.AUpsample = AUpsample  # ultralytics parse_model resolves 'nn.X' via getattr(torch.nn, X)

# axis -> (P2 -> P3 conv stride, P3 -> P4 conv stride = P4 -> P3 upsample scale), each in (y, x) order
AXES = {"x": ([2, 1], [2, 4]), "y": ([1, 2], [4, 2])}


def _base(nc: int) -> dict:
    from ultralytics.nn.tasks import yaml_model_load
    d = copy.deepcopy(yaml_model_load("yolov8n.yaml"))
    d["scale"] = "n"
    d["nc"] = nc
    return d


def s3_dict(nc: int = 8, axis: str = "x") -> dict:
    """yolov8n model dict with the four S3 edits (axis 'x' = S3x, 'y' = S3y)"""
    d = _base(nc)
    bb, hd = d["backbone"], d["head"]
    s23, s34 = AXES[axis]
    bb[3] = [-1, 1, "Conv", [256, 3, s23]]                   # 3: P2 -> P3 (stride 8 (y) x 4 (x) for 'x')
    bb[5] = [-1, 1, "Conv", [512, 3, s34]]                   # 5: P3 -> P4 (back to 16 x 16)
    hd[3] = [-1, 1, "nn.AUpsample", [None, s34, "nearest"]]  # 13: P4 -> P3 (top-down)
    hd[6] = [-1, 1, "Conv", [256, 3, s34]]                   # 16: P3 -> P4 (bottom-up)
    return d


# ------------------------------------------------------------------ head
def make_anchors_xy(feats, stride_xy, offset: float = 0.5):
    """make_anchors with per-axis strides -> anchor points [A, 2] (x, y) in each level's grid units, strides [A, 2] (sx, sy)"""
    pts, sts = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i, f in enumerate(feats):
        h, w = f.shape[2:]
        sx = torch.arange(w, device=device, dtype=dtype) + offset
        sy = torch.arange(h, device=device, dtype=dtype) + offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        pts.append(torch.stack((sx, sy), -1).view(-1, 2))
        sts.append(stride_xy[i].to(device=device, dtype=dtype).view(1, 2).expand(h * w, 2))
    return torch.cat(pts), torch.cat(sts)


class AnisoDetect(Detect):
    """Detect with per-level (sx, sy) strides and per-level head widths (None = plain v8n width)"""
    legacy = True  # v8 head layout (Conv-Conv-Conv2d class branch), whatever parse_model last set on Detect

    def __init__(self, nc=80, ch=(), widths=None):
        super().__init__(nc, ch)
        self.widths = list(widths) if widths is not None else [None] * self.nl
        for L, (x, c) in enumerate(zip(ch, self.widths)):
            if c:
                self.cv2[L] = nn.Sequential(Conv(x, c, 3), Conv(c, c, 3), nn.Conv2d(c, 4 * self.reg_max, 1))
                self.cv3[L] = nn.Sequential(Conv(x, c, 3), Conv(c, c, 3), nn.Conv2d(c, self.nc, 1))
        self.register_buffer("stride_xy", torch.ones(self.nl, 2), persistent=False)  # (sx, sy) per level; set by install_det()

    def _inference(self, x):
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            a, s = make_anchors_xy(x, self.stride_xy, 0.5)
            self.anchors, self.strides = a.transpose(0, 1), s.repeat(1, 2).transpose(0, 1)  # [2, A], [4, A] = (sx, sy, sx, sy)
            self.shape = shape
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides  # per-axis grid units -> px (xywh)
        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        for a, b, s in zip(self.cv2, self.cv3, self.stride_xy.prod(1).sqrt().tolist()):
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)  # cls: the same prior per image area as Detect


def install_det(m, det, tname: str, probe: int = 256):
    """replace m's Detect by det (same inputs); per-axis strides from a probe x probe forward; bias_init"""
    from ultralytics.utils.torch_utils import initialize_weights
    old = m.model[-1]
    det.f, det.i, det.type, det.inplace = old.f, old.i, tname, old.inplace
    initialize_weights(det)  # BN eps / momentum as in the rest of the model
    det.np = sum(p.numel() for p in det.parameters())
    m.model[-1] = det
    m.model.eval()
    det.training = True  # Detect returns the per-level maps
    with torch.no_grad():
        fs = m.forward(torch.zeros(1, 3, probe, probe))
    m.model.train()
    det.stride_xy.copy_(torch.tensor([[probe / f.shape[-1], probe / f.shape[-2]] for f in fs]))
    det.stride = det.stride_xy.prod(1).sqrt()  # isotropic-equivalent stride (bias_init-like uses only); use stride_xy elsewhere
    m.stride = det.stride
    det.bias_init()
    return m


def build(d: dict, nc: int, widths, probe: int = 256):
    """DetectionModel from model dict d with its Detect replaced by AnisoDetect(widths)"""
    from ultralytics.nn.tasks import DetectionModel
    m = DetectionModel(d, nc=nc, verbose=False)
    det = AnisoDetect(nc, tuple(s[0].conv.in_channels for s in m.model[-1].cv2), widths)
    return install_det(m, det, "efpmot.models.aniso.AnisoDetect", probe)


def load_coco(m, weights: str, tag: str = ""):
    """COCO checkpoint by name + shape; tensors of the thinned level-0 box branch (cv2.0) take the leading channels"""
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    sd = ck["model"].float().state_dict()
    msd = m.state_dict()
    keep, n_sl = {}, 0
    for k, v in msd.items():
        if k not in sd:
            continue
        if sd[k].shape == v.shape:
            keep[k] = sd[k]
        elif ".cv2.0." in k and sd[k].ndim == v.ndim and all(a_ >= b_ for a_, b_ in zip(sd[k].shape, v.shape)):
            keep[k] = sd[k][tuple(slice(0, n_) for n_ in v.shape)].clone()
            n_sl += 1
    m.load_state_dict(keep, strict=False)
    print(f"{tag}: loaded {len(keep)}/{len(msd)} tensors ({n_sl} level-0 box-branch tensors sliced)", flush=True)
    return m


def s3_model(nc: int, axis: str, weights: str | None = None, head_c: int = 32):
    """YOLOv8n S3x ('x') / S3y ('y') with a head_c-wide P3 head; COCO weights loaded when a checkpoint path is given"""
    m = build(s3_dict(nc, axis), nc, (head_c, None, None))
    return load_coco(m, weights, f"yolov8n-s3{axis}") if weights else m


# ------------------------------------------------------------------ loss
class AnisoV8Loss(v8DetectionLoss):
    """v8DetectionLoss (Ultralytics 8.3.170) for AnisoDetect. img_hw = (H, W) of the input batch, set by the adapter before
    each call (feats[0] size x stride[0] is wrong per axis)."""

    def __init__(self, model, tal_topk: int = 10):
        super().__init__(model, tal_topk)
        self.stride_xy = model.model[-1].stride_xy
        self.img_hw = None

    def assign(self, feats, batch) -> dict:
        """-> dict: pred_distri, pred_scores [B, A, C] logits, pred_bboxes (xyxy, per-axis grid units), anchor_points [A, 2],
        st4 [A, 4], target_bboxes (px), target_scores, fg_mask, target_gt_idx, gt_bboxes (px), gt_labels"""
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        sxy = self.stride_xy.to(self.device)
        hw = self.img_hw if self.img_hw is not None else (feats[0].shape[2] * float(sxy[0, 1]), feats[0].shape[3] * float(sxy[0, 0]))
        imgsz = torch.tensor([float(hw[0]), float(hw[1])], device=self.device, dtype=dtype)  # image size (h, w)
        anchor_points, st2 = make_anchors_xy(feats, sxy, 0.5)
        st4 = st2.repeat(1, 2)
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy px
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, per-axis grid units
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * st4).type(gt_bboxes.dtype), anchor_points * st2, gt_labels,
            gt_bboxes, mask_gt)
        return dict(pred_distri=pred_distri, pred_scores=pred_scores, pred_bboxes=pred_bboxes, anchor_points=anchor_points,
                    st4=st4, target_bboxes=target_bboxes, target_scores=target_scores, fg_mask=fg_mask,
                    target_gt_idx=target_gt_idx, gt_bboxes=gt_bboxes, gt_labels=gt_labels)

    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        r = self.assign(feats, batch)
        pred_scores, fg_mask, target_scores = r["pred_scores"], r["fg_mask"], r["target_scores"]
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        if fg_mask.sum():
            st4, tb = r["st4"], r["target_bboxes"]
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
            iou = bbox_iou((r["pred_bboxes"].float() * st4.float())[fg_mask], tb[fg_mask], xywh=False, CIoU=True)  # pixel boxes
            loss[0] = ((1.0 - iou) * weight).sum() / target_scores_sum
            target_ltrb = bbox2dist(r["anchor_points"], tb / st4, self.reg_max - 1)  # DFL in per-axis grid units
            ld = self.bbox_loss.dfl_loss(r["pred_distri"][fg_mask].view(-1, self.reg_max), target_ltrb[fg_mask]) * weight
            loss[2] = ld.sum() / target_scores_sum
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return loss * batch_size, loss.detach()
