# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""HIT-UAV (Suo et al., Scientific Data 2023): aerial thermal-infrared drone images (640x512, single-channel JPEG) with persons and
vehicles, COCO-like labels; licence CC BY 4.0 (GitHub) / CC0 1.0 (LICENSE of the release zip). Detection only (single images).

Data (configs/paths.yaml data.hituav = the extracted HIT-UAV folder of the release zip HIT-UAV.zip, v1.2.1):
  <hituav>/normal_json/{train,val,test}/<name>.jpg and <hituav>/normal_json/annotations/{train,val,test}.json
  (the normal_xml / rotate_* copies are not used). The JSON is COCO-like, BUT the annotation list key is 'annotation' (singular)
  and the images carry 'filename'; bbox = [x, y, w, h] in integer pixels; categories 0 Person, 1 Car, 2 Bicycle, 3 OtherVehicle,
  4 DontCare (checked when read).
File name {daynight}_{altitude m}_{camera angle deg}_{weather}_{serial:05d}.jpg: daynight 0 = day, 1 = night (read off the images:
  flag-0 images show sun-cast thermal shadows), altitude 60, 70, ..., 130, angle 30, ..., 90 (90 = looking straight down), weather
  always 0; the serial is a global counter (0..10,170, unique). Code = the name without the serial, e.g. 1_60_30_0.
Facts (all images): train / val / test = 2,029 / 290 / 579 images, the JSON image lists equal the folders, every box integer and
  inside its frame; boxes Person 8,533 / 1,168 / 2,611, Car 5,253 / 719 / 1,339, Bicycle 3,630 / 554 / 796, OtherVehicle
  102 / 12 / 34, DontCare 110 / 7 / 31; at most 42 boxes per image. The official split interleaves frames of the same recordings
  (for 96 % of the test images a train image with the same code lies within 10 serials), so its test measures seen recordings.
Serial order is not time order. Each recording is one block of serials: serial = block base + rank of the extraction index in a
  STRING sort of 0..M-1 (0, 1, 10, 11, ..., 19, 2, 20, ...), M = number of frames extracted from the source video (7.5 fps, every
  7th frame, SECONDS_PER_INDEX = 0.93 s apart); only some extracted frames are in the release. Checked on the 5 recordings that
  have a source video (video_sample/*.mov of the release). time_indices() / recording_time_indices() map serials to time indices.

Decisions (fixed before any HIT-UAV training; the folds of decision 7 were added after the official-split runs had started)
1. Splits: 'official' = the normal_json train / val / test folders; 'recA' / 'recB' / 'altLH' = the recording-disjoint folds of
   decision 7 (images read from their official folders). Training = the split's train part; val serves one choice only, the
   operating confidence threshold (decision 5); test gives the reported numbers. No early stopping (the model after the schedule).
2. Classes into the 8-class heads (BDD_CLASSES order): Person -> pedestrian, Car -> car, Bicycle -> bicycle, OtherVehicle -> truck;
   the other 4 heads get no targets and are never evaluated. DontCare is not a training target: its boxes are dropped and its pixels
   are NOT greyed out (unlike the VisDrone ignored regions), so it is background for the loss. In evaluation the DontCare boxes are
   ignore regions (decision 5). Training boxes are clipped to the frame (a no-op on this data); the evaluation GT = the raw boxes.
3. Training items = single images: every image of the train part, the ones without a box included (they teach background); each
   image is its own one-image sequence (key 0), returned as (image, image, targets, targets); FramePairs augmentation (flip, scale
   0.8-1.2, gain 0.7-1.3), letterbox and packing unchanged. Epochs from the 9,060-iteration budget of DSEC (6 x 1,510):
   round(9060 / (n_train // 16)) = official 72 (2,029 images, 126 iterations / epoch), recA 116 (1,263 / 78), recB 118 (1,237 / 77),
   altLH 112 (1,305 / 81); warm-up = the harness rule min(500, iterations / epoch).
4. Export: every val and test image of the split, <cache>/<tag>/hituav/{val,test}.npz (the part plays the role of a sequence;
   frames = image names in sorted order, dets = float32 [N, 6] x1, y1, x2, y2, score, class in native pixels), conf EXPORT_CONF =
   0.001 (so that AP reaches low scores), class-wise NMS 0.65, max 300 per image, one class per anchor, FP16, letterbox.
5. Evaluation (efpmot/eval/detection.py): pedestrian (primary) and car on test and val. Per image and class, greedy matching in
   descending score order: a detection takes the unmatched same-class GT box with the highest IoU >= t (TP); otherwise it is
   ignored if its IoU with some DontCare box is >= t (a DontCare box absorbs any number of detections); otherwise FP. AP = COCO
   101-point interpolated precision over every exported detection; AP50 at t = 0.5, AP50:95 = mean over t = 0.50:0.05:0.95.
   Operating point: the score threshold maximising the class's F1 at IoU 0.5 on VAL; at it, test precision / recall / F1 and test
   recall by altitude, camera angle, day / night and native GT short side (<6, 6-8, 8-10, 10-12, 12-16, >=16 px).
6. Run tags hit_<det>_<canvas width>[<module>][<fold>]_s<seed>: hit_v8n_416_s0 = yolov8n on the 320x416 canvas (letterbox scale
   0.625: 400x320 + 16 px right padding), hit_v8n_416s3xrecA_s0 = yolov8n_s3x on fold recA, hit_v8n_448x352altLH_s1 = the uniform
   352x448 canvas on altLH. A fold run carries its fold marker, an official-split run none (check_tag()).
7. Recording-disjoint folds (the shipped splits/hituav_split.json, written by build_split_file(): scripts/prepare_data.py hituav).
   Recording = a maximal run, in serial order, of images with the same code whose consecutive serials differ by at most REC_GAP =
   30; flight = a maximal run of consecutive recordings with the same daynight / altitude / weather. From the file names alone:
   264 recordings in 38 flights, the same as grouping by (code, date_captured) (every such group is one contiguous serial run; the
   largest step inside a recording is 29, so REC_GAP splits nothing in this release). 37 flights sweep all 7 angles; flight f00
   lacks 40 deg (its 1_60_40_0 block, serials 150-280, is not in the release).
   recA / recB: the recordings of every (altitude, day/night) cell are split into two halves by exhaustive search (the cell's first
   recording in half 1) minimising |dP| / P + 0.25 |dI| / I + 0.1 |dC| / C + 0.1 sum_angle |dI_angle| / I (P, I, C = person boxes,
   images, car boxes of the cell from the official annotations, DontCare not counted; d = half 1 - half 2); the halves are oriented
   one cell at a time, largest cell first, keeping the image totals of the folds as equal as possible: recA test = one half of every
   cell, recB test = the other (every recording is test in exactly one fold). Val of each fold = at most one training recording per
   cell, chosen by coordinate descent so that val holds ~10 % of the fold's training recordings, images, person and car boxes, day /
   night and per-altitude person boxes; train = the rest. Every test recording shares its flight with training recordings of its
   fold: the folds test unseen recordings, not unseen sites.
   altLH (altitude shift): test = every recording at 100-130 m; val = per altitude 60 / 70 / 80 / 90 m the recordings in the order
   of zlib.crc32('altLH-' + recording id) until they hold >= 10 % of that altitude's images; train = the other 60-90 m recordings.
   No flight has recordings on both sides, so the test flights are unseen.
     split   train images  val  test   test person boxes
     recA          1,263   187  1,448   6,301
     recB          1,237   211  1,450   6,011
     altLH         1,305   226  1,367   3,305
"""
from __future__ import annotations

import json
import os
import re
import types
import zlib

import numpy as np

from .. import paths
from .common import BDD_CLASSES
from .pairs import FramePairs

PARTS = ("train", "val", "test")
EXPORT_PARTS = ("val", "test")                 # decision 4
FOLDS = ("recA", "recB", "altLH")              # decision 7
SPLITS = ("official",) + FOLDS
SPLIT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits", "hituav_split.json")
CATEGORIES = ("Person", "Car", "Bicycle", "OtherVehicle", "DontCare")   # category_id order of the annotation files
CLASS2BDD = {0: "pedestrian", 1: "car", 2: "bicycle", 3: "truck"}      # category_id -> BDD head (decision 2)
DONTCARE = 4
EVAL_CLASSES = ("pedestrian", "car")
NATIVE_HW = (512, 640)
MAX_OBJS = 120                                 # at most 42 boxes per image
ITER_BUDGET = 9060                             # DSEC 6 x 1510 (decision 3)
EXPORT_CONF = 0.001                            # decision 4
REC_GAP = 30                                   # decision 7
SECONDS_PER_INDEX = 7 / 7.5                    # one time index = every 7th frame of the 7.5 fps source video
W_IMG, W_CAR, W_ANG, VAL_SHARE = 0.25, 0.1, 0.1, 0.10   # decision 7: recA / recB cost weights, val share
ALTITUDES, ANGLES = tuple(range(60, 131, 10)), tuple(range(30, 91, 10))
NAME_RE = re.compile(r"^([01])_(\d+)_(\d+)_(\d+)_(\d+)\.jpg$")
FOLD_RE = re.compile(r"(?:rec[AB]|altLH)(?=_|$)")   # fold marker of a run tag (decision 6)
_ANN: dict = {}
_LOC: dict = {}
_SPLIT: dict = {}


# ------------------------------------------------------------------ names, tags
def parse_name(name: str) -> dict:
    """'1_130_90_0_04077.jpg' -> dict(daynight=1, night=True, altitude=130, angle=90, weather=0, serial=4077, code='1_130_90_0')"""
    m = NAME_RE.match(os.path.basename(str(name)))
    if m is None:
        raise ValueError(f"not a HIT-UAV file name: {name}")
    dn, alt, ang, wea, ser = (int(v) for v in m.groups())
    return dict(daynight=dn, night=bool(dn), altitude=alt, angle=ang, weather=wea, serial=ser, code=f"{dn}_{alt}_{ang}_{wea}")


def image_name(code: str, serial: int) -> str:
    return f"{code}_{int(serial):05d}.jpg"


def split_of_tag(tag: str) -> str:
    """'hit_v8n_416s3xrecA_s0' -> 'recA'; a tag without a fold marker -> 'official'"""
    f = set(FOLD_RE.findall(str(tag)))
    if len(f) > 1:
        raise ValueError(f"run tag {tag!r} carries several fold markers {sorted(f)}")
    return f.pop() if f else "official"


def check_split(split: str) -> str:
    if split not in SPLITS:
        raise ValueError(f"HIT-UAV split {split!r}: choose from {SPLITS}")
    return split


def check_tag(tag: str, split: str) -> None:
    """decision 6: a fold run's tag carries its fold marker and an official-split tag none, so a split can never be trained or
    exported into another split's tag; raises ValueError"""
    s = split_of_tag(tag)
    if s != check_split(split):
        hint = "official-split tags carry no fold marker" if split == "official" else f"fold runs are named like hit_v8n_416s3x{split}_s0"
        raise ValueError(f"run tag {tag!r} {'carries ' + repr(s) if s != 'official' else 'has no fold marker'} but split {split!r} "
                         f"is selected ({hint})")


# ------------------------------------------------------------------ files of the official release
def root() -> str:
    return paths.require("data.hituav")


def img_dir(part: str) -> str:
    """folder of an OFFICIAL part"""
    return os.path.join(root(), "normal_json", part)


def ann_path(part: str) -> str:
    return os.path.join(root(), "normal_json", "annotations", part + ".json")


def official_images(part: str) -> list[str]:
    """sorted image names (the *.jpg of the official part's folder)"""
    return sorted(f for f in os.listdir(img_dir(part)) if f.lower().endswith(".jpg"))


def official_location() -> dict:
    """{image name: official part} of every image of the release (cached)"""
    if not _LOC:
        _LOC.update({f: p for p in PARTS for f in official_images(p)})
    return _LOC


def _read_official(part: str) -> dict:
    """annotations/<part>.json -> {name: dict(hw=(H, W), boxes=float64 [N, 6] (category, x1, y1, x2, y2, annotation id))} for every
    image of the official part (no box -> [0, 6]), raw boxes in file order, DontCare included. Checks the category list and that the
    JSON image list equals the folder. Cached."""
    if part not in _ANN:
        with open(ann_path(part), encoding="utf-8") as f:
            d = json.load(f)
        cats = {int(c["id"]): c["name"] for c in d["categories"]}
        if cats != dict(enumerate(CATEGORIES)):
            raise ValueError(f"{ann_path(part)}: categories {cats} != {dict(enumerate(CATEGORIES))}")
        imgs = {im["id"]: im for im in d["images"]}
        names = sorted(im["filename"] for im in imgs.values())
        if not len(names) == len(set(names)) == len(d["images"]):
            raise ValueError(f"{ann_path(part)}: duplicate file names / image ids")
        if names != official_images(part):
            raise ValueError(f"{ann_path(part)}: the image list differs from the folder {img_dir(part)}")
        rows = {im["filename"]: [] for im in imgs.values()}
        for a in d["annotation"]:   # 'annotation' (singular) in HIT-UAV
            x, y, w, h = (float(v) for v in a["bbox"])
            rows[imgs[a["image_id"]]["filename"]].append((a["category_id"], x, y, x + w, y + h, a["id"]))
        hw = {im["filename"]: (int(im["height"]), int(im["width"])) for im in imgs.values()}
        _ANN[part] = {n: dict(hw=hw[n], boxes=np.asarray(rows[n], np.float64).reshape(-1, 6)) for n in names}
    return _ANN[part]


# ------------------------------------------------------------------ splits
def load_split() -> dict:
    """the shipped splits/hituav_split.json (cached)"""
    if not _SPLIT:
        with open(SPLIT_JSON, encoding="utf-8") as f:
            _SPLIT.update(json.load(f))
    return _SPLIT


def recordings() -> list[dict]:
    """recordings of the shipped split, serial order: dict(id, code, daynight, altitude, angle, weather, flight, n_person, n_car,
    serials, images)"""
    out = []
    for rid, r in load_split()["recordings"].items():
        p = parse_name(image_name(r["code"], r["serials"][0]))
        out.append(dict(id=rid, code=r["code"], daynight=p["daynight"], altitude=p["altitude"], angle=p["angle"], weather=p["weather"],
                        flight=r["flight"], n_person=r["n_person"], n_car=r["n_car"], serials=list(r["serials"]),
                        images=[image_name(r["code"], s) for s in r["serials"]]))
    return out


def recording_of(name: str) -> str:
    """image name -> recording id of the shipped split"""
    key = os.path.basename(name)
    for r in recordings():
        if key in r["images"]:
            return r["id"]
    raise KeyError(f"{name} is in no recording of {SPLIT_JSON}")


def fold_recordings(fold: str) -> dict:
    """{'train' | 'val' | 'test': recording ids} of a fold (checked: disjoint, covering every recording)"""
    d = load_split()
    if fold not in FOLDS:
        raise ValueError(f"not a fold: {fold!r} (choose from {FOLDS})")
    parts = {p: list(d["folds"][fold][p]) for p in PARTS}
    ids = [i for p in PARTS for i in parts[p]]
    if len(ids) != len(set(ids)) or set(ids) != set(d["recordings"]):
        raise ValueError(f"{SPLIT_JSON}: fold {fold} parts overlap or do not cover the recordings")
    return parts


def fold_parts(fold: str) -> dict:
    """{'train' | 'val' | 'test': sorted image names} of a fold"""
    imgs = {r["id"]: r["images"] for r in recordings()}
    return {p: sorted(n for i in ids for n in imgs[i]) for p, ids in fold_recordings(fold).items()}


def list_images(part: str, split: str = "official") -> list[str]:
    """sorted image names of a part ('train' | 'val' | 'test') of the official split or of a fold"""
    if check_split(split) == "official":
        return official_images(part)
    return fold_parts(split)[part]


def image_paths(part: str, split: str = "official") -> list[str]:
    """every image of the part, sorted by name (the export order); a fold's images are read from their official folders"""
    loc = official_location()
    names = list_images(part, split)
    miss = [n for n in names if n not in loc]
    if miss:
        raise FileNotFoundError(f"{len(miss)} images of {split} {part} are not in {root()} (e.g. {miss[:3]})")
    return [os.path.join(img_dir(loc[n]), n) for n in names]


def read_annotations(part: str, split: str = "official") -> dict:
    """{name: dict(hw, boxes [N, 6] (category, x1, y1, x2, y2, annotation id))} of every image of the part (official files pooled for
    a fold)"""
    if check_split(split) == "official":
        return _read_official(part)
    pool = {}
    for p in PARTS:
        pool.update(_read_official(p))
    return {n: pool[n] for n in list_images(part, split)}


def read_labels(part: str, split: str = "official", clip: bool = True) -> dict:
    """training labels {name: float32 [N, 6] (cls_bdd, x1, y1, x2, y2, annotation id)}: the 4 mapped categories (DontCare dropped),
    clipped to the frame when clip (boxes with no area left are dropped)"""
    out = {}
    for n, a in read_annotations(part, split).items():
        H, W = a["hw"]
        r = []
        for c, x1, y1, x2, y2, aid in a["boxes"]:
            if int(c) not in CLASS2BDD:
                continue
            if clip:
                x1, x2 = min(max(x1, 0.0), W), min(max(x2, 0.0), W)
                y1, y2 = min(max(y1, 0.0), H), min(max(y2, 0.0), H)
            if x2 <= x1 or y2 <= y1:
                continue
            r.append((BDD_CLASSES.index(CLASS2BDD[int(c)]), x1, y1, x2, y2, aid))
        out[n] = np.asarray(r, np.float32).reshape(-1, 6)
    return out


def eval_gt(part: str, split: str = "official") -> dict:
    """evaluation GT {name: dict(gt=float64 [N, 5] (cls_bdd, x1, y1, x2, y2) raw boxes of the 4 mapped categories in file order,
    dontcare=float64 [M, 4] raw DontCare boxes, meta=parse_name(name), hw=(H, W))} for every image of the part"""
    out = {}
    for n, a in read_annotations(part, split).items():
        b = a["boxes"]
        cat = b[:, 0].astype(int)
        keep = np.isin(cat, list(CLASS2BDD))
        cls = np.array([BDD_CLASSES.index(CLASS2BDD[c]) for c in cat[keep]], np.float64)
        out[n] = dict(gt=np.concatenate([cls[:, None], b[keep, 1:5]], 1).reshape(-1, 5), dontcare=b[cat == DONTCARE, 1:5].reshape(-1, 4),
                      meta=parse_name(n), hw=a["hw"])
    return out


def eval_seqs() -> list[str]:
    """the exported parts (decision 4); a part plays the role of a sequence"""
    return list(EXPORT_PARTS)


def eval_frame_paths(part: str, split: str = "official") -> list[str]:
    return image_paths(part, split)


def native_hw(seq: str | None = None) -> tuple[int, int]:
    return NATIVE_HW


def view(split: str):
    """the export view of a split for efpmot.train.export_dets: eval_seqs() = val, test; eval_frame_paths(part) = that part of the
    split; EXPORT_CONF"""
    split = check_split(split)
    return types.SimpleNamespace(split=split, eval_seqs=eval_seqs, eval_frame_paths=lambda part: image_paths(part, split),
                                 native_hw=native_hw, EXPORT_CONF=EXPORT_CONF)


def train_epochs(n_train: int, bs: int = 16) -> int:
    """decision 3: round(ITER_BUDGET / (n_train // bs))"""
    return round(ITER_BUDGET / max(n_train // bs, 1))


class HituavImages(FramePairs):
    """single images in the BDD class space (decision 3): every image is its own one-image sequence (key 0), returned as (image,
    image, targets, targets); FramePairs augmentation, letterbox and packing unchanged. Images are read from their official folders."""
    load_prev = False

    def __init__(self, names=None, size_hw=(320, 416), train=True, max_objs=MAX_OBJS, part="train", split="official"):
        names = list_images(part, split) if names is None else list(names)
        lab = read_labels(part, split, clip=True)
        loc = official_location()
        self.items, self.labels, self.names = [], {}, names
        for n in names:
            p = os.path.join(img_dir(loc[n]), n)
            self.labels[n] = {0: lab[n]}
            self.items.append((n, 0, p, p))
        ann = read_annotations(part, split)
        self.n_boxes = int(sum(len(lab[n]) for n in names))
        self.n_dontcare = int(sum((ann[n]["boxes"][:, 0] == DONTCARE).sum() for n in names))
        self.size, self.train, self.max_objs, self.split = size_hw, train, max_objs, split


# ------------------------------------------------------------------ serial -> time
def rank_order(m: int) -> np.ndarray:
    """time index of each serial rank in a block of m extracted frames: 0..m-1 sorted as strings (0, 1, 10, ..., 19, 2, 20, ...)"""
    return np.array(sorted(range(int(m)), key=str), np.int64)


def time_indices(serials, base: int, m: int) -> np.ndarray:
    """file-name serials of one recording -> time indices (time = index x SECONDS_PER_INDEX s after the block's first extracted
    frame); base = the serial of rank 0, m = the block length (number of extracted frames)"""
    r = np.asarray(serials, np.int64).reshape(-1) - int(base)
    if len(r) and (r.min() < 0 or r.max() >= m):
        raise ValueError(f"serials outside the block [{base}, {int(base) + int(m) - 1}]")
    return rank_order(m)[r]


def recording_time_indices(rec: str, m: int | None = None) -> dict:
    """{image name: time index} of one recording of the shipped split. base = its first serial (assumes the frame with extraction
    index 0 is in the release, true for the 5 recordings with a source video). m default = the next recording's first serial - base:
    an ASSUMPTION that fails when a whole block is missing between the two, e.g. r000 (true m = 150, default 281, because the
    1_60_40_0 block, serials 150-280, is not in the release); the last recording defaults to its last serial - base + 1."""
    rs = recordings()
    k = next((i for i, r in enumerate(rs) if r["id"] == rec), None)
    if k is None:
        raise KeyError(f"no recording {rec!r} in {SPLIT_JSON}")
    r = rs[k]
    base = r["serials"][0]
    if m is None:
        m = rs[k + 1]["serials"][0] - base if k + 1 < len(rs) else r["serials"][-1] - base + 1
    return dict(zip(r["images"], time_indices(r["serials"], base, m).tolist()))


# ------------------------------------------------------------------ building the split (decision 7)
def recordings_from_names(names, gap: int = REC_GAP) -> list[dict]:
    """image names (all parts pooled) -> recordings, serial order: dict(id 'r000', ..., code, daynight, altitude, angle, weather,
    flight 'f00', ..., serials, images); see decision 7"""
    rows = sorted((parse_name(n)["serial"], os.path.basename(n)) for n in names)
    ser = [s for s, _ in rows]
    if len(set(ser)) != len(ser):
        raise ValueError("HIT-UAV serials are not unique")
    runs, cur = [], [rows[0]]
    for a, b in zip(rows, rows[1:]):
        if parse_name(b[1])["code"] != parse_name(a[1])["code"] or b[0] - a[0] > gap:
            runs.append(cur)
            cur = []
        cur.append(b)
    runs.append(cur)
    out, prev, nf = [], None, -1
    for i, run in enumerate(runs):
        p = parse_name(run[0][1])
        key = (p["daynight"], p["altitude"], p["weather"])
        if key != prev:
            nf, prev = nf + 1, key
        out.append(dict(id=f"r{i:03d}", code=p["code"], daynight=p["daynight"], altitude=p["altitude"], angle=p["angle"],
                        weather=p["weather"], flight=f"f{nf:02d}", serials=[s for s, _ in run], images=[n for _, n in run]))
    return out


def best_halves(v: np.ndarray, w_ang: float = W_ANG) -> np.ndarray:
    """v int [n, 3 + 7] (persons, images, cars, images at angle 30..90) of one cell's recordings (serial order) -> bool [n], True =
    half 1 (v[0] always in half 1): the first mask of minimum cost |dP| / P + W_IMG |dI| / I + W_CAR |dC| / C + w_ang
    sum_angle |dI_angle| / I by exhaustive search"""
    n = len(v)
    tot = v.sum(0).astype(np.float64)
    den = np.maximum(tot, 1.0)
    best, best_mask = np.inf, 0
    for start in range(0, 2 ** (n - 1), 1 << 16):
        m = np.arange(start, min(start + (1 << 16), 2 ** (n - 1)), dtype=np.int64)
        bits = ((m[:, None] >> np.arange(n - 1)) & 1).astype(np.float64)
        s1 = v[0] + bits @ v[1:].astype(np.float64)
        d = np.abs(2 * s1 - tot)
        cost = d[:, 0] / den[0] + W_IMG * d[:, 1] / den[1] + W_CAR * d[:, 2] / den[2] + w_ang * d[:, 3:].sum(1) / den[1]
        i = int(np.argmin(cost))
        if cost[i] < best:
            best, best_mask = float(cost[i]), int(m[i])
    return np.array([True] + [bool((best_mask >> j) & 1) for j in range(n - 1)])


def val_cost(sel: list, pool: list, T: dict) -> float:
    """the val objective of the recordings sel against the fold's training recordings pool: the shares of recordings, images, person
    boxes, car boxes (x 0.5), day / night person boxes (mean, x 0.5) and person boxes per altitude (mean) should all be VAL_SHARE"""
    def s(rs, k):
        return sum(r[k] for r in rs)
    c = abs(len(sel) / len(pool) - VAL_SHARE) + abs(s(sel, "n_img") / T["n_img"] - VAL_SHARE)
    c += abs(s(sel, "n_person") / T["n_person"] - VAL_SHARE)
    c += 0.5 * abs(s(sel, "n_car") / T["n_car"] - VAL_SHARE)
    c += 0.5 * np.mean([abs(sum(r["n_person"] for r in sel if r["daynight"] == dn) / max(T["p_dn"][dn], 1) - VAL_SHARE) for dn in (0, 1)])
    c += np.mean([abs(sum(r["n_person"] for r in sel if r["altitude"] == al) / max(T["p_alt"][al], 1) - VAL_SHARE) for al in ALTITUDES])
    return c


def choose_val(pool: list) -> list:
    """at most one recording per (altitude, day/night) cell; coordinate descent over the sorted cells (start: none; each cell takes
    the option, none or one of its recordings in serial order, first of equal costs, minimising val_cost given the others) until
    stable"""
    T = dict(n_person=sum(r["n_person"] for r in pool), n_img=sum(r["n_img"] for r in pool), n_car=sum(r["n_car"] for r in pool),
             p_dn={dn: sum(r["n_person"] for r in pool if r["daynight"] == dn) for dn in (0, 1)},
             p_alt={al: sum(r["n_person"] for r in pool if r["altitude"] == al) for al in ALTITUDES})
    cells: dict = {}
    for r in pool:
        cells.setdefault((r["altitude"], r["daynight"]), []).append(r)
    pick = {c: None for c in sorted(cells)}
    for _ in range(50):
        changed = False
        for c in sorted(cells):
            opts = [None] + cells[c]
            costs = [val_cost([x for k, x in pick.items() if k != c and x is not None] + ([o] if o is not None else []), pool, T) for o in opts]
            o = opts[int(np.argmin(costs))]
            if o is not pick[c]:
                pick[c], changed = o, True
        if not changed:
            break
    return [r for r in pick.values() if r is not None]


def build_folds(recs: list, w_ang: float = W_ANG) -> dict:
    """recordings (serial order, with n_img, n_person, n_car) -> {fold: {part: recording ids in serial order}} (decision 7)"""
    cells: dict = {}
    for r in recs:
        cells.setdefault((r["altitude"], r["daynight"]), []).append(r)
    halves = {}
    for c, rs in cells.items():
        v = np.array([[r["n_person"], r["n_img"], r["n_car"]] + [r["n_img"] * (r["angle"] == g) for g in ANGLES] for r in rs], np.int64)
        h = best_halves(v, w_ang)
        halves[c] = ([r for r, k in zip(rs, h) if k], [r for r, k in zip(rs, h) if not k])
    test_of, ia, ib = {}, 0, 0
    for c in sorted(cells, key=lambda c: (-sum(r["n_img"] for r in cells[c]), c)):
        h1, h2 = halves[c]
        i1, i2 = sum(r["n_img"] for r in h1), sum(r["n_img"] for r in h2)
        ha, hb = (h2, h1) if abs(ia + i2 - (ib + i1)) < abs(ia + i1 - (ib + i2)) else (h1, h2)
        test_of.update({r["id"]: "recA" for r in ha})
        test_of.update({r["id"]: "recB" for r in hb})
        ia += sum(r["n_img"] for r in ha)
        ib += sum(r["n_img"] for r in hb)
    folds = {}
    for f, other in (("recA", "recB"), ("recB", "recA")):
        val = {r["id"] for r in choose_val([r for r in recs if test_of[r["id"]] == other])}
        folds[f] = dict(train=[r["id"] for r in recs if test_of[r["id"]] == other and r["id"] not in val],
                        val=[r["id"] for r in recs if r["id"] in val], test=[r["id"] for r in recs if test_of[r["id"]] == f])
    val = []
    for al in (60, 70, 80, 90):
        rs = sorted((r for r in recs if r["altitude"] == al), key=lambda r: (zlib.crc32(("altLH-" + r["id"]).encode()), r["id"]))
        tot, acc = sum(r["n_img"] for r in rs), 0
        for r in rs:
            if 10 * acc >= tot:
                break
            val.append(r["id"])
            acc += r["n_img"]
    folds["altLH"] = dict(train=[r["id"] for r in recs if r["altitude"] <= 90 and r["id"] not in val],
                          val=[r["id"] for r in recs if r["id"] in val], test=[r["id"] for r in recs if r["altitude"] >= 100])
    return folds


def build_split(names, n_person: dict, n_car: dict) -> dict:
    """the split file content (decision 7) from every image name of the release and its Person / Car box counts"""
    recs = recordings_from_names(names)
    for r in recs:
        r.update(n_img=len(r["images"]), n_person=int(sum(n_person.get(n, 0) for n in r["images"])),
                 n_car=int(sum(n_car.get(n, 0) for n in r["images"])))
    folds = build_folds(recs)
    return dict(about="HIT-UAV recording-disjoint folds (efpmot/data/hituav.py decision 7); image name = <code>_<serial:05d>.jpg",
                builder="python -B scripts/prepare_data.py hituav", rec_gap=REC_GAP,
                cost=dict(w_img=W_IMG, w_car=W_CAR, w_ang=W_ANG, val_share=VAL_SHARE), n_images=sum(r["n_img"] for r in recs),
                n_recordings=len(recs), n_flights=len({r["flight"] for r in recs}),
                recordings={r["id"]: dict(code=r["code"], flight=r["flight"], n_person=r["n_person"], n_car=r["n_car"], serials=r["serials"])
                            for r in recs},
                folds=folds)


def split_text(d: dict) -> str:
    """deterministic JSON text of a split: one line per header key, recording and fold part"""
    head = [f" {json.dumps(k)}: {json.dumps(v)}" for k, v in d.items() if k not in ("recordings", "folds")]
    recs = [f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in d["recordings"].items()]
    folds = []
    for f, parts in d["folds"].items():
        folds.append(f"  {json.dumps(f)}: {{\n" + ",\n".join(f"   {json.dumps(p)}: {json.dumps(ids)}" for p, ids in parts.items()) + "\n  }")
    return "{\n" + ",\n".join(head) + ',\n "recordings": {\n' + ",\n".join(recs) + '\n },\n "folds": {\n' + ",\n".join(folds) + "\n }\n}\n"


def split_from_dataset() -> dict:
    """build_split() from the official folders and annotation files"""
    n_person, n_car = {}, {}
    for p in PARTS:
        for n, a in _read_official(p).items():
            c = a["boxes"][:, 0].astype(int)
            n_person[n], n_car[n] = int((c == 0).sum()), int((c == 1).sum())
    return build_split(list(official_location()), n_person, n_car)


def split_stats(d: dict) -> list[str]:
    """printable per-fold statistics of a split dict"""
    recs = d["recordings"]
    out = [f"recordings {d['n_recordings']} (REC_GAP {d['rec_gap']}), flights {d['n_flights']}, images {d['n_images']}"]
    for f, parts in d["folds"].items():
        cols = []
        for p in PARTS:
            ids = parts[p]
            n_img, n_person = sum(len(recs[i]["serials"]) for i in ids), sum(recs[i]["n_person"] for i in ids)
            cols.append(f"{p} {n_img} img / {len(ids)} rec / {n_person} person")
        n_tr = sum(len(recs[i]["serials"]) for i in parts["train"])
        out.append(f"{f:6s} " + ", ".join(cols) + f"; epochs round({ITER_BUDGET} / {n_tr // 16}) = {train_epochs(n_tr)}")
    return out


def build_split_file(force: bool = False) -> str:
    """writes splits/hituav_split.json from the dataset when it is missing; an existing file is kept when identical, and never replaced
    by a different split without force (the different build goes to <derived>/hituav/hituav_split.json). Returns the path written
    or checked."""
    d = split_from_dataset()
    txt = split_text(d)
    print("\n".join(split_stats(d)), flush=True)
    if os.path.isfile(SPLIT_JSON):
        with open(SPLIT_JSON, encoding="utf-8") as f:
            same = f.read() == txt
        if same:
            print("UNCHANGED", SPLIT_JSON, "(the dataset gives the shipped split)", flush=True)
            return SPLIT_JSON
        if not force:
            dst = os.path.join(paths.out("derived", "hituav"), "hituav_split.json")
            with open(dst, "w", encoding="utf-8", newline="\n") as f:
                f.write(txt)
            raise SystemExit(f"the dataset gives a split that differs from the shipped {SPLIT_JSON}; written to {dst} "
                             "(replace the shipped file with --force only on purpose: runs depend on it)")
    with open(SPLIT_JSON, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    _SPLIT.clear()
    print("WROTE", SPLIT_JSON, flush=True)
    return SPLIT_JSON


def compare_split(other_json: str) -> list[str]:
    """differences between the shipped split and another split file in the research layout (recordings {id: images, flight, ...},
    folds {fold: {part: image names, recordings: {part: ids}}}); [] = the same recordings, flights and folds"""
    with open(other_json, encoding="utf-8") as f:
        o = json.load(f)
    diff = []
    mine = {r["id"]: r for r in recordings()}
    if list(mine) != list(o["recordings"]):
        diff.append(f"recording ids differ: {len(mine)} vs {len(o['recordings'])}")
    for i, r in mine.items():
        q = o["recordings"].get(i)
        if q is None or q["images"] != r["images"] or q["flight"] != r["flight"]:
            diff.append(f"recording {i} differs")
    for fold in FOLDS:
        if fold not in o["folds"]:
            diff.append(f"fold {fold} missing in {other_json}")
            continue
        parts, ids = fold_parts(fold), fold_recordings(fold)
        for p in PARTS:
            if sorted(o["folds"][fold][p]) != parts[p]:
                diff.append(f"{fold} {p}: image lists differ")
            if sorted(o["folds"][fold]["recordings"][p]) != sorted(ids[p]):
                diff.append(f"{fold} {p}: recording ids differ")
    return diff
