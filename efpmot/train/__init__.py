# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""efpmot.train: the training loop train(cfg) and the detection-cache export export_dets().

Recipe of every published run: COCO-pretrained start, batch 16 frame pairs (the detector loss uses the current frame), AdamW
(weight decay 5e-4 on weights with ndim > 1, 0 on biases / BN; betas 0.9 / 0.999; lr = round(0.002 * 5 / (4 + nc), 6) =
0.000833 for nc = 8), linear warm-up over min(500, iterations per epoch) from 0.1 x lr, then cosine to 0.05 x lr, FP16 autocast
with a GradScaler, gradient-norm clip 10. Epochs: 6 (DSEC, VisDrone, BDD100K), 55 (MOT17, iterations matched to the others).
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time

import cv2
import numpy as np
import torch

from .. import paths
from ..data import get as get_data, letterbox, train_dataset
from ..models import build

TRAIN_DEFAULTS = {"dsec": dict(hw=(320, 416), epochs=6), "mot17": dict(hw=(256, 416), epochs=55),
                  "visdrone": dict(hw=(384, 640), epochs=6), "bdd": dict(hw=(256, 416), epochs=6)}
CFG_DEFAULTS = dict(bs=16, workers=4, seed=0, dsec_stride=2, vis_stride=None, max_iters=0, export=True, device="cuda")


def _cfg(cfg) -> dict:
    c = dict(vars(cfg)) if isinstance(cfg, argparse.Namespace) else dict(cfg)
    for k in ("family", "data", "name"):
        if not c.get(k):
            raise ValueError(f"train(): '{k}' is required")
    for k, v in CFG_DEFAULTS.items():
        c.setdefault(k, v)
    d = TRAIN_DEFAULTS[c["data"]]
    c["hw"] = list(c.get("hw") or d["hw"])
    c["epochs"] = int(c.get("epochs") or d["epochs"])
    return c


def train(cfg) -> str:
    """cfg (dict or argparse.Namespace): family, data, name, hw (H, W), epochs, bs=16, workers=4, seed=0, dsec_stride=2,
    vis_stride=None (-> 1), max_iters=0 (smoke test: stop after N iterations, no checkpoint / export), export=True,
    device="cuda" (the published recipe: FP16 autocast + GradScaler; "cpu" runs FP32 without scaling, for smoke tests only).
    Writes <runs>/<name>/{last.pth (after every epoch), log.txt} and, when export, the detection cache of the evaluation
    frames. Returns the run directory."""
    a = _cfg(cfg)
    torch.manual_seed(a["seed"])
    np.random.seed(a["seed"])
    random.seed(a["seed"])
    run_dir = paths.out("runs", a["name"])
    dev = a["device"]
    cuda = dev.startswith("cuda")
    ad = build(a["family"], 8, "coco")
    ad.model.to(dev)
    size_hw = tuple(a["hw"])
    log = open(os.path.join(run_dir, "log.txt"), "a")

    def P(*s):
        msg = " ".join(str(x) for x in s)
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    data = a["data"]
    if data == "dsec":
        ds = train_dataset("dsec", size_hw, a["dsec_stride"])
    elif data == "visdrone":
        ds = train_dataset("visdrone", size_hw, a["vis_stride"])
        P(f"visdrone train: {len(ds.labels)} seqs, stride {a['vis_stride'] or get_data('visdrone').TRAIN_STRIDE}, max_objs {ds.max_objs}")
    elif data == "mot17":
        ds = train_dataset("mot17", size_hw)
        P(f"mot17 train: {len(ds.labels)} seqs, training halves {sum(len(v) for v in ds.frames.values())} frames, max_objs {ds.max_objs}")
    else:
        ds = train_dataset("bdd", size_hw)
    ds.load_prev = False   # the detector loss uses the current frame only (same frames, targets and random draws)
    dl = torch.utils.data.DataLoader(ds, batch_size=a["bs"], shuffle=True, num_workers=a["workers"], drop_last=True, pin_memory=cuda,
                                     persistent_workers=a["workers"] > 0)
    P(f"== {a['name']} family={a['family']} hw={size_hw} train_frames={len(ds)} iters/epoch={len(dl)} lr={ad.lr} wd={ad.wd}")
    if a["family"].startswith("yolo11n"):
        d_ = ad.model.model[-1]
        P(f"yolo11n axis={getattr(ad, 'axis', None)} strides_xy={d_.stride_xy.tolist() if hasattr(d_, 'stride_xy') else d_.stride.tolist()} "
          f"p3_head_w={(d_.widths[0] if hasattr(d_, 'widths') else None) or d_.cv2[0][0].conv.out_channels}")
    decay, no_decay = [], []
    for _, p in ad.model.named_parameters():
        (decay if p.ndim > 1 else no_decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": ad.wd}, {"params": no_decay, "weight_decay": 0.0}], lr=ad.lr, betas=(0.9, 0.999))
    total = a["epochs"] * len(dl)
    warm = min(500, len(dl))

    def lr_at(it):
        if it < warm:
            return ad.lr * (0.1 + 0.9 * it / warm)
        return 0.05 * ad.lr + 0.95 * ad.lr * 0.5 * (1 + math.cos(math.pi * (it - warm) / max(total - warm, 1)))
    scaler = torch.amp.GradScaler("cuda") if cuda else torch.amp.GradScaler("cpu", enabled=False)
    it = 0
    t0 = time.time()
    for ep in range(a["epochs"]):
        ad.model.train()
        agg: dict = {}
        for batch in dl:
            cur, tgt = batch[0], batch[2]  # (cur, prev, tgt, ptgt): the detector loss uses the current frame
            for g in opt.param_groups:
                g["lr"] = lr_at(it)
            cur, tgt = cur.to(dev, non_blocking=True), tgt.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                loss, items, _ = ad.train_forward(cur, tgt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(ad.model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            for k, v in items.items():
                agg[k] = agg.get(k, 0) + v
            agg["total"] = agg.get("total", 0) + float(loss.detach())
            it += 1
            if a["max_iters"] and it >= a["max_iters"]:
                P(f"SMOKE_DONE it {it} {time.time() - t0:.0f}s " + " ".join(f"{k}={v / it:.3f}" for k, v in agg.items()))
                log.close()
                return run_dir
            if it % 100 == 0:
                P(f"ep {ep} it {it}/{total} loss {float(loss.detach()):.3f} lr {lr_at(it):.6f} {time.time() - t0:.0f}s")
        P(f"== epoch {ep} done: " + " ".join(f"{k}={v / len(dl):.3f}" for k, v in agg.items()) + f" time {time.time() - t0:.0f}s")
        torch.save({"model": ad.model.state_dict(), "epoch": ep, "args": dict(a)}, os.path.join(run_dir, "last.pth"))
    if a["export"]:
        export_dets(ad, a["name"], data, size_hw, device=dev)
    P("TRAIN_DONE", a["name"])
    log.close()
    return run_dir


@torch.no_grad()
def export_dets(adapter, name: str, data: str, size_hw, conf: float = 0.01, nms: float = 0.65, max_det: int = 300,
                device: str = "cuda", seqs: list[str] | None = None) -> str:
    """detection cache of the evaluation frames -> <cache>/<name>/<data>/<seq>.npz (frames = image file names, dets[i] =
    float32 [N, 6] x1, y1, x2, y2 (native px, clipped to the image), score, class index). Letterbox to size_hw, batch 1, FP16
    autocast, one class per anchor (max score), conf >= conf, class-wise NMS, at most max_det boxes. seqs: export only these
    evaluation sequences (default: all of them). Returns the folder."""
    import torchvision
    mod = get_data(data)
    out_dir = paths.out("cache", name, data)
    cuda = device.startswith("cuda")
    adapter.model.to(device)
    n, t_model = 0, 0.0
    all_seqs = mod.eval_seqs()
    if seqs:
        unknown = [q for q in seqs if q not in all_seqs]
        if unknown:
            raise ValueError(f"not evaluation sequences of {data}: {unknown}")
    for s in all_seqs:
        if seqs and s not in seqs:
            continue
        names, dets = [], []
        for fp in mod.eval_frame_paths(s):
            img = cv2.imread(fp)
            h, w = img.shape[:2]
            lb, r = letterbox(img, size_hw)
            x = torch.from_numpy(np.ascontiguousarray(lb.transpose(2, 0, 1))).float().unsqueeze(0).to(device)
            if cuda:
                torch.cuda.synchronize()
            t1 = time.time()
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                boxes, scores = adapter.infer(x)
            boxes, scores = boxes[0].float(), scores[0].float()
            conf_v, cls_i = scores.max(1)
            k = conf_v >= conf
            bx, cv, ci = boxes[k], conf_v[k], cls_i[k]
            if len(bx):
                keep = torchvision.ops.batched_nms(bx, cv, ci, nms)[:max_det]
                bx, cv, ci = bx[keep], cv[keep], ci[keep]
            if cuda:
                torch.cuda.synchronize()
            t_model += time.time() - t1
            n += 1
            if len(bx) == 0:
                d = np.zeros((0, 6), np.float32)
            else:
                b = (bx / r).cpu().numpy()
                b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
                b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
                d = np.concatenate([b, cv.cpu().numpy()[:, None], ci.cpu().numpy()[:, None].astype(np.float32)], 1).astype(np.float32)
            names.append(os.path.basename(fp))
            dets.append(d)
        np.savez_compressed(os.path.join(out_dir, s + ".npz"), frames=np.array(names), dets=np.array(dets, dtype=object))
    print(f"export {name}: {n} frames, model {n / max(t_model, 1e-6):.1f} fps ({'fp16' if cuda else 'fp32 cpu'}, bs1, incl. NMS)", flush=True)
    return out_dir
