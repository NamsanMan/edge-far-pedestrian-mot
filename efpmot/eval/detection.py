# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Detection AP and operating point of exported detection caches (numpy only): HIT-UAV, rules in efpmot/data/hituav.py
(decisions 4 and 5).

Per image and class, greedy matching in descending score order at each IoU threshold t: a detection takes the unmatched same-class
GT box with the highest IoU >= t (TP; equal IoUs -> the later GT, as pycocotools); otherwise it is ignored when its IoU with some
DontCare box is >= t (a DontCare box absorbs any number of detections); otherwise it is a FP. AP = COCO 101-point interpolated
precision over every exported detection (pycocotools accumulate(); ignored detections are zero-increment positions); AP50 at
t = 0.5, AP50:95 = the mean over t = 0.50, 0.55, ..., 0.95. Operating point: the score threshold maximising the class's F1 at
IoU 0.5 on val (candidates = every distinct detection score, kept = score >= thr; equal F1 -> the higher threshold); at it, test
(and val) TP / FP / FN, precision, recall, F1, and test recall by altitude, camera angle, day / night and native GT short side.
Greedy score-ordered matching makes thresholding after matching the same as matching only the detections above the threshold, so
a GT counts as recalled at thr iff its IoU-0.5 match has score >= thr.
Checks of these rules (on the original evaluator): GT boxes as detections give AP = 100 and recall 100 % in every bin, also with
every DontCare box added as an extra detection; with the DontCare rule off, AP at all 10 IoU thresholds equals pycocotools 2.0.11
COCOeval (maxDets 300) exactly on a real export.

    from efpmot.eval.detection import evaluate_hituav
    r = evaluate_hituav("hit_v8n_416s3xrecA_s0")   # <cache>/<tag>/hituav/{val,test}.npz; split = the tag's fold marker
"""
from __future__ import annotations

import os

import numpy as np

from .. import paths
from ..data import BDD_CLASSES
from ..data import hituav as H

IOU_THRS = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1)   # = pycocotools Params.iouThrs
REC_THRS = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1)   # = pycocotools Params.recThrs (101 points)
SIDE_BINS = (("<6", 0, 6), ("6-8", 6, 8), ("8-10", 8, 10), ("10-12", 10, 12), ("12-16", 12, 16), (">=16", 16, np.inf))
UNITS = "AP / P / R / F1 / recall in %, thr = raw detection score (kept: score >= thr)"
EVALUATOR = "efpmot.eval.detection (rules: efpmot/data/hituav.py decision 5)"


def iou(a, b) -> np.ndarray:
    """a [N, 4], b [M, 4] xyxy -> IoU [N, M] in float64 (the pycocotools box formula; no overlap -> 0)"""
    a = np.asarray(a, np.float64).reshape(-1, 4)
    b = np.asarray(b, np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    w = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    h = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    inter = np.where((w > 0) & (h > 0), w * h, 0.0)
    u = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None] + ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :] - inter
    return np.where(inter > 0, inter / np.where(u > 0, u, 1.0), 0.0)


def match_image(det, gt, ignore, thrs=IOU_THRS):
    """one image, one class: det [D, 4] in descending score order, gt [G, 4], ignore [M, 4] (DontCare boxes), xyxy ->
    status int8 [T, D] (1 TP, 0 FP, -1 ignored) and gt_det int64 [T, G] (index of the matching detection, -1 = missed) per IoU
    threshold"""
    D, G = len(det), len(gt)
    status = np.zeros((len(thrs), D), np.int8)
    gt_det = np.full((len(thrs), G), -1, np.int64)
    if D == 0:
        return status, gt_det
    ig = iou(det, gt)
    dmax = iou(det, ignore).max(1) if len(ignore) else np.zeros(D)
    for ti, t in enumerate(thrs):
        taken = np.zeros(G, bool)
        for d in np.nonzero((ig >= t).any(1))[0]:
            row = np.where(taken, -1.0, ig[d])
            j = G - 1 - int(np.argmax(row[::-1]))   # highest IoU among the unmatched GT; equal IoUs -> the later GT (pycocotools)
            if row[j] >= t:
                taken[j] = True
                gt_det[ti, j] = d
                status[ti, d] = 1
        status[ti, (status[ti] == 0) & (dmax >= t)] = -1
    return status, gt_det


def coco_ap(scores, status, npos: int) -> float | None:
    """COCO 101-point interpolated AP (%) of one class at one IoU threshold (pycocotools accumulate()): every detection (score [D],
    status [D] 1 TP / 0 FP / -1 ignored), npos = number of GT boxes; None without GT"""
    if npos == 0:
        return None
    if len(scores) == 0:
        return 0.0
    st = status[np.argsort(-scores, kind="mergesort")]
    tp = np.cumsum(st == 1).astype(np.float64)
    fp = np.cumsum(st == 0).astype(np.float64)
    rc = tp / npos
    pr = tp / (fp + tp + np.spacing(1))
    pr = np.maximum.accumulate(pr[::-1])[::-1]
    i = np.searchsorted(rc, REC_THRS, side="left")
    q = np.where(i < len(pr), pr[np.minimum(i, len(pr) - 1)], 0.0)
    return 100.0 * float(q.mean())


def _pct(x):
    return None if x is None else 100.0 * float(x)


def load_cache(tag: str, part: str, cache_root: str | None = None) -> dict:
    """<cache_root>/<tag>/hituav/<part>.npz (default outputs.cache) -> {image name: float32 [N, 6] x1, y1, x2, y2, score, cls}"""
    f = os.path.join(cache_root or paths.require("outputs.cache"), tag, "hituav", part + ".npz")
    with np.load(f, allow_pickle=True) as z:
        fr, dets = [str(x) for x in z["frames"]], z["dets"]
        if len(fr) != len(dets):
            raise ValueError(f"{f}: {len(fr)} frames but {len(dets)} detection arrays")
        return {n: np.asarray(dets[i], np.float32).reshape(-1, 6) for i, n in enumerate(fr)}


def eval_part(dets: dict, part: str, split: str, classes, dontcare: bool = True) -> dict:
    """dets {image name: [N, 6]} -> {class: dict(n_img, n_gt, scores [D], status [T, D], gt_score [G], gt_side [G], gt_meta [G])}
    over every image of the part (each must be in dets); dontcare=False switches the ignore rule off (cross-checks only)"""
    G = H.eval_gt(part, split)
    names = sorted(G)
    miss = [n for n in names if n not in dets]
    if miss:
        raise ValueError(f"{split} {part}: {len(miss)} of {len(names)} images missing in the cache (e.g. {miss[:3]}): export every image")
    out = {}
    for c in classes:
        ci = BDD_CLASSES.index(c)
        S, ST, GS, GSIDE, GM = [], [], [], [], []
        for n in names:
            d = dets[n]
            d = d[d[:, 5].astype(np.int64) == ci]
            d = d[np.argsort(-d[:, 4], kind="mergesort")]
            g = G[n]["gt"]
            gb = g[g[:, 0] == ci, 1:5]
            st, gd = match_image(d[:, :4], gb, G[n]["dontcare"] if dontcare else np.zeros((0, 4)))
            sc = d[:, 4].astype(np.float64)
            S.append(sc)
            ST.append(st)
            GS.append(np.where(gd[0] >= 0, sc[np.maximum(gd[0], 0)] if len(sc) else 0.0, -np.inf))   # IoU 0.5 = IOU_THRS[0]
            GSIDE.append(np.minimum(gb[:, 2] - gb[:, 0], gb[:, 3] - gb[:, 1]))
            GM += [G[n]["meta"]] * len(gb)
        out[c] = dict(n_img=len(names), n_gt=int(sum(len(x) for x in GS)), scores=np.concatenate(S), status=np.concatenate(ST, 1),
                      gt_score=np.concatenate(GS), gt_side=np.concatenate(GSIDE), gt_meta=GM)
    return out


def ap_block(e: dict) -> dict:
    aps = [coco_ap(e["scores"], e["status"][ti], e["n_gt"]) for ti in range(len(IOU_THRS))]
    s05 = e["status"][0]
    return dict(n_images=e["n_img"], n_gt=e["n_gt"], n_det=int(len(e["scores"])), n_tp50=int((s05 == 1).sum()), n_fp50=int((s05 == 0).sum()),
                n_ignored50=int((s05 == -1).sum()), AP50=aps[0], AP50_95=None if aps[0] is None else float(np.mean(aps)),
                AP_per_iou={f"{t:.2f}": a for t, a in zip(IOU_THRS, aps)})


def best_f1(e: dict) -> float | None:
    """the score threshold maximising F1 at IoU 0.5 over 'score >= s' (s = every distinct score of a non-ignored detection); ties ->
    the higher threshold; None without GT or detections"""
    k = e["status"][0] >= 0
    s = e["scores"][k]
    t = e["status"][0][k] == 1
    if e["n_gt"] == 0 or len(s) == 0:
        return None
    o = np.argsort(-s, kind="mergesort")
    s, t = s[o], t[o]
    tp = np.cumsum(t)
    n = np.arange(1, len(s) + 1)
    last = np.r_[s[1:] != s[:-1], True]   # the end of each run of equal scores
    s, tp, n = s[last], tp[last], n[last]
    P = tp / n
    R = tp / e["n_gt"]
    F1 = np.where(tp > 0, 2 * P * R / np.maximum(P + R, 1e-12), 0.0)
    return float(s[int(np.argmax(F1))])


def at_thr(e: dict, thr: float) -> dict:
    s05 = e["status"][0]
    keep = e["scores"] >= thr
    TP = int(((s05 == 1) & keep).sum())
    FP = int(((s05 == 0) & keep).sum())
    if TP != int((e["gt_score"] >= thr).sum()):
        raise RuntimeError("matching / recall bookkeeping mismatch")
    P = TP / (TP + FP) if TP + FP else None
    R = TP / e["n_gt"] if e["n_gt"] else None
    F1 = 2 * P * R / (P + R) if P and R else 0.0
    return dict(TP=TP, FP=FP, FN=e["n_gt"] - TP, P=_pct(P), R=_pct(R), F1=_pct(F1))


def recall_by(e: dict, thr: float) -> dict:
    """recall at thr by altitude, camera angle, day / night and native GT short side"""
    hit = e["gt_score"] >= thr
    m, side = e["gt_meta"], e["gt_side"]

    def tab(keys, order):
        keys = np.asarray(keys, object)
        extra = set(keys.tolist()) - set(order)
        if extra:
            raise ValueError(f"unexpected bin keys {extra}")
        return {str(k): dict(n=int((keys == k).sum()), recall=_pct(hit[keys == k].mean()) if (keys == k).any() else None) for k in order}
    sb = [next(lab for lab, lo, hi in SIDE_BINS if lo <= v < hi) for v in side]
    return dict(altitude=tab([x["altitude"] for x in m], list(H.ALTITUDES)), angle=tab([x["angle"] for x in m], list(H.ANGLES)),
                daynight=tab(["night" if x["night"] else "day" for x in m], ["day", "night"]),
                short_side=tab(sb, [lab for lab, _, _ in SIDE_BINS]))


def evaluate_hituav(tag: str, split: str | None = None, classes=H.EVAL_CLASSES, cache_root: str | None = None,
                    dontcare: bool = True) -> dict:
    """one run -> result dict (json-ready): test and val AP blocks per class and the operating point of each class. split default =
    the tag's fold marker (official without one); a given split must match it. Raises if a cache part is missing or incomplete."""
    split = split or H.split_of_tag(tag)
    H.check_tag(tag, split)
    E = {p: eval_part(load_cache(tag, p, cache_root), p, split, classes, dontcare) for p in H.EXPORT_PARTS}
    r = dict(tag=tag, split=split, evaluator=EVALUATOR, dontcare_ignore=dontcare, classes=list(classes), units=UNITS,
             iou_thrs=[float(t) for t in IOU_THRS], ap={p: {c: ap_block(E[p][c]) for c in classes} for p in ("test", "val")}, op={})
    for c in classes:
        thr = best_f1(E["val"][c])
        r["op"][c] = dict(thr=thr, rule="max F1 at IoU 0.5 on val") if thr is None else dict(
            thr=thr, rule="max F1 at IoU 0.5 on val", val=at_thr(E["val"][c], thr), test=at_thr(E["test"][c], thr),
            test_recall=recall_by(E["test"][c], thr))
    return r
