# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""BDD100K: detection images for training, MOT val sequences for the tracking evaluation.

Training: a fixed 35k-image subset of the official detection train split (splits/bdd_det_train_35k.txt), COCO-format labels
with the 8 classes in BDD_CLASSES order. prepare_det() converts the legacy 100k labels + images into <bdd_det>/{train2017,
val2017, annotations/det_{train,val}.json}; build_subset() writes <derived>/bdd100k_det/annotations/det_train_35k.json. The det
val split is not used: its images come from the same videos as the MOT val sequences that are tracked.
Evaluation: BDD100K MOT val (200 sequences), images <bdd_mot_val_images>/<seq>/*.jpg and the TrackEval BDD100K GT
<bdd_mot_val_gt>/<seq>.json.
"""
from __future__ import annotations

import glob
import json
import os
import time
import zipfile

from .. import paths
from .common import BDD_CLASSES
from .pairs import FramePairs

NATIVE_HW = (720, 1280)
SUBSET_LIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits", "bdd_det_train_35k.txt")
LEGACY = {"person": "pedestrian", "bike": "bicycle", "motor": "motorcycle"}  # legacy 100k label names -> BDD MOT names


# ------------------------------------------------------------------ MOT val (evaluation)
def gt_dir() -> str:
    return paths.require("data.bdd_mot_val_gt")


def list_val_seqs() -> list[str]:
    return sorted(os.path.splitext(f)[0] for f in os.listdir(gt_dir()) if f.endswith(".json"))


def val_frame_paths(seq: str) -> list[str]:
    return sorted(glob.glob(os.path.join(paths.require("data.bdd_mot_val_images"), seq, "*.jpg")))


def eval_seqs() -> list[str]:
    return list_val_seqs()


def eval_frame_paths(seq: str) -> list[str]:
    return val_frame_paths(seq)


def native_hw(seq: str | None = None) -> tuple[int, int]:
    return NATIVE_HW


# ------------------------------------------------------------------ detection training set
def subset_file() -> str:
    return os.path.join(paths.require("outputs.derived"), "bdd100k_det", "annotations", "det_train_35k.json")


def build_subset() -> str:
    """<bdd_det>/annotations/det_train.json restricted to the 35k image names of splits/bdd_det_train_35k.txt: images in the
    order of the list, annotations in their det_train.json order -> subset_file()"""
    with open(SUBSET_LIST) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    with open(os.path.join(paths.require("data.bdd_det"), "annotations", "det_train.json")) as f:
        d = json.load(f)
    by_name = {im["file_name"]: im for im in d["images"]}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ValueError(f"{len(missing)} subset images are not in det_train.json, e.g. {missing[:3]}")
    images = [by_name[n] for n in names]
    keep = {im["id"] for im in images}
    out = dict(images=images, annotations=[a for a in d["annotations"] if a["image_id"] in keep], categories=d["categories"])
    dst = subset_file()
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f)
    print(f"BDD det subset: {len(images)} images, {len(out['annotations'])} boxes -> {dst}", flush=True)
    return dst


def _load_coco(ann: str):
    with open(ann) as f:
        d = json.load(f)
    boxes = {}
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        if w > 1 and h > 1:
            boxes.setdefault(a["image_id"], []).append((a["category_id"] - 1, x, y, x + w, y + h))
    return {im["id"]: im["file_name"] for im in d["images"]}, boxes


class BddDet(FramePairs):
    """single images as (cur, prev = cur) pairs, so the FramePairs augmentation and letterbox apply unchanged; images in
    image-id order, boxes wider and higher than 1 px"""

    def __init__(self, size_hw=(256, 416), train=True, max_objs=120, ann: str | None = None):
        names, boxes = _load_coco(ann or subset_file())
        img_dir = os.path.join(paths.require("data.bdd_det"), "train2017")
        self.items, lab = [], {}
        for k, (i, fn) in enumerate(sorted(names.items())):
            p = os.path.join(img_dir, fn)
            self.items.append(("bdd", k, p, p))
            lab[k] = [(c, x1, y1, x2, y2, j) for j, (c, x1, y1, x2, y2) in enumerate(boxes.get(i, []))]
        self.labels = {"bdd": lab}
        self.size, self.train, self.max_objs = size_hw, train, max_objs


def prepare_det(images_zip: str, labels_zip: str, out: str | None = None, max_train: int = 0, skip_extract: bool = False) -> str:
    """legacy BDD100K 100k labels (bdd100k_labels.zip) + 100k images zip -> <out>/{train2017, val2017}/*.jpg and
    <out>/annotations/det_{train,val}.json (COCO, 8 classes, boxes wider and higher than 1 px); out default data.bdd_det"""
    out = out or paths.require("data.bdd_det")
    t0 = time.time()
    zl = zipfile.ZipFile(labels_zip)
    label_files: dict = {}
    for n in zl.namelist():
        if n.endswith(".json") and "/train/" in n:
            label_files.setdefault("train", []).append(n)
        elif n.endswith(".json") and "/val/" in n:
            label_files.setdefault("val", []).append(n)
    print({k: len(v) for k, v in label_files.items()}, flush=True)
    zi = zipfile.ZipFile(images_zip)
    img_entries = {os.path.basename(n): n for n in zi.namelist() if n.endswith(".jpg")}
    print("image entries", len(img_entries), flush=True)
    for split, coco_name in (("val", "val2017"), ("train", "train2017")):
        files = sorted(label_files[split])
        if split == "train" and max_train:
            files = files[:max_train]
        out_img = os.path.join(out, coco_name)
        os.makedirs(out_img, exist_ok=True)
        images, anns = [], []
        aid, miss = 1, 0
        for i, n in enumerate(files):
            d = json.loads(zl.read(n))
            name = d["name"] + ".jpg"
            if name not in img_entries:
                miss += 1
                continue
            dst = os.path.join(out_img, name)
            if not skip_extract and not os.path.exists(dst):
                with zi.open(img_entries[name]) as src, open(dst, "wb") as f:
                    f.write(src.read())
            iid = len(images) + 1
            at = d.get("attributes", {})
            images.append(dict(id=iid, file_name=name, width=1280, height=720, timeofday=at.get("timeofday"), weather=at.get("weather"),
                               scene=at.get("scene")))
            for o in d["frames"][0].get("objects", []):
                c = LEGACY.get(o["category"], o["category"])
                if c not in BDD_CLASSES or "box2d" not in o:
                    continue
                b = o["box2d"]
                w = b["x2"] - b["x1"]
                h = b["y2"] - b["y1"]
                if w <= 1 or h <= 1:
                    continue
                anns.append(dict(id=aid, image_id=iid, category_id=BDD_CLASSES.index(c) + 1, bbox=[b["x1"], b["y1"], w, h], area=w * h, iscrowd=0))
                aid += 1
            if i % 5000 == 0:
                print(split, i, len(files), f"{time.time() - t0:.0f}s", flush=True)
        os.makedirs(os.path.join(out, "annotations"), exist_ok=True)
        with open(os.path.join(out, "annotations", f"det_{split}.json"), "w") as f:
            json.dump(dict(images=images, annotations=anns, categories=[dict(id=i + 1, name=c) for i, c in enumerate(BDD_CLASSES)]), f)
        print(f"{split}: images {len(images)} anns {len(anns)} missing_img {miss}", flush=True)
    print("BDD det prepared", f"{time.time() - t0:.0f}s ->", out, flush=True)
    return out
