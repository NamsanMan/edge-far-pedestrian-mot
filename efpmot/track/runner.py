# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Tracking of cached detections with BoxMOT ByteTrack / OC-SORT.

Settings (every dataset, every arm): one tracker instance per class, frame_rate 30, lost tracks kept for 15 frames
(ByteTrack track_buffer, OC-SORT max_age), match_thresh 0.8, min_conf 0.1, min_hits 1, IoU threshold 0.3, OC-SORT with
BYTE association. The detection
threshold ``thr`` is the tracker's track_thresh / det_thresh and is swept over GRID; detections scoring below
0.2 x thr are dropped before the tracker (lower-scoring boxes still reach BYTE's second association stage), and boxes
not wider and higher than 0.5 px are dropped. Track ids: BoxMOT id + 1 + class index x 100000.

Detection cache (efpmot.train.export_dets): <outputs.cache>/<tag>/<dataset>/<seq>.npz with ``frames`` (file names) and
``dets`` (float32 [N, 6] per frame: x1, y1, x2, y2 in native image pixels, score, class index in BDD order).

Writers (<outputs.results>/<dataset>_trackers/<name>/data/):
- dsec / bdd / visdrone: BDD100K / Scalabel JSON, one entry per frame. DSEC keeps only the labelled frames, re-indexed
  in GT order; VisDrone drops outputs whose box centre lies in an ignored region of that frame (after tracking).
- mot17: MOTChallenge text, pedestrian only, frames renumbered as in the half-split GT (frame - half_offset).
"""
from __future__ import annotations

import glob
import inspect
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .. import paths

GRID = {"bytetrack": ["0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"], "ocsort": ["0.5", "0.6", "0.7", "0.8", "0.9"]}
NSEQ_TRACKED = {"dsec": 19, "mot17": 7, "visdrone": 17, "bdd": 200}     # files per tracker folder when complete
DATASETS = tuple(NSEQ_TRACKED)


def make_tracker(kind: str, frame_rate: int = 30, track_buffer: int = 15, track_thresh: float = 0.5,
                 match_thresh: float = 0.8, min_conf: float = 0.1):
    """BoxMOT tracker with the candidate settings its constructor (or BaseTracker, through **kwargs) accepts.

    With boxmot 25.0 this gives ByteTrack(min_conf, track_thresh, match_thresh, track_buffer, frame_rate, min_hits=1,
    iou_threshold=0.3) and OcSort(min_conf, use_byte=True, min_hits=1, iou_threshold=0.3, det_thresh=track_thresh,
    max_age=track_buffer).

    BoxMOT's ByteTrack logs "max_age=30, det_thresh=0.3" when it is created. Those are BaseTracker defaults, printed
    before ByteTrack applies its own settings: it keeps lost tracks for track_buffer frames (15 here, scaled by
    frame_rate / 30) and sets det_thresh = track_thresh, so the log line does not contradict the settings above."""
    import boxmot
    from boxmot.trackers.base import BaseTracker
    cls = {"bytetrack": boxmot.ByteTrack, "ocsort": boxmot.OcSort}[kind]
    sig = inspect.signature(cls.__init__).parameters
    base_sig = inspect.signature(BaseTracker.__init__).parameters if "kwargs" in sig else {}
    cand = dict(frame_rate=frame_rate, track_buffer=track_buffer, track_thresh=track_thresh,
                match_thresh=match_thresh, min_conf=min_conf, det_thresh=track_thresh, max_age=track_buffer,
                min_hits=1, iou_threshold=0.3, use_byte=True, per_class=False,
                track_high_thresh=track_thresh, track_low_thresh=0.1, new_track_thresh=track_thresh + 0.1,
                use_embeddings=False, use_cmc=False)
    kw = {k: v for k, v in cand.items() if k in sig}
    if "kwargs" in sig:  # BaseTracker params; skip the ones the subclass aliases (det_thresh -> track_thresh, max_age -> track_buffer)
        for k in ("min_hits", "iou_threshold", "det_thresh", "max_age"):
            if k in base_sig and k not in sig and not (k == "det_thresh" and "track_thresh" in sig) \
                    and not (k == "max_age" and "track_buffer" in sig):
                kw[k] = cand[k]
    return cls(**kw)


def track_sequence(dets_per_frame: list[np.ndarray], img_hw: tuple[int, int], classes: list[str], kind: str,
                   det_thresh: float, tr_kw: dict, keep_classes: list[str] | None = None) -> list[np.ndarray]:
    """dets_per_frame: float32 [N, 6] (x1, y1, x2, y2, score, class index) per frame -> per frame float32 [M, 7]
    (x1, y1, x2, y2, track id, score, class index); one tracker per class (only keep_classes when given)"""
    cls_ids = [c for c in range(len(classes)) if keep_classes is None or classes[c] in keep_classes]
    trackers = {c: make_tracker(kind, **tr_kw) for c in cls_ids}
    dummy = np.zeros((img_hw[0], img_hw[1], 3), np.uint8)   # motion-only trackers: the image is never read
    out = []
    for d in dets_per_frame:
        rows = []
        for c in cls_ids:
            dc = d[d[:, 5] == c] if len(d) else d
            dc = dc[dc[:, 4] >= det_thresh * 0.2] if len(dc) else dc  # keep low-score dets for BYTE's second stage
            inp = dc[:, :6].astype(np.float32) if len(dc) else np.zeros((0, 6), np.float32)
            r = trackers[c].update(inp, dummy)
            if r is not None and len(r):
                r = np.asarray(r, dtype=np.float32)
                # boxmot output: x1, y1, x2, y2, id, conf, cls, ind
                ids = r[:, 4:5] + 1 + c * 100000  # unique across per-class trackers, > 0
                rows.append(np.concatenate([r[:, :4], ids, r[:, 5:6], np.full((len(r), 1), c, np.float32)], 1))
        out.append(np.concatenate(rows, 0) if rows else np.zeros((0, 7), np.float32))
    return out


def tracker_name(tag: str, tracker: str, thr: float) -> str:
    """'<tag>__<tracker>_t<thr>' with thr printed as Python prints the float (0.2, 0.5, ...)"""
    return f"{tag}__{tracker}_t{float(thr)}"


def _clean(dets: list[np.ndarray], ncls: int) -> list[np.ndarray]:
    """drop boxes of unknown classes and boxes not wider / higher than 0.5 px"""
    out = []
    for d in dets:
        if len(d) == 0:
            out.append(np.zeros((0, 6), np.float32)); continue
        keep = np.array([0 <= int(c) < ncls for c in d[:, 5]]) & ((d[:, 2] - d[:, 0]) > 0.5) & ((d[:, 3] - d[:, 1]) > 0.5)
        d = d[keep].copy()
        d[:, 5] = [int(c) for c in d[:, 5]]
        out.append(d.astype(np.float32))
    return out


def _write_json(path: str, seq: str, pairs, classes: list[str]) -> None:
    out = []
    for i, (fn, r) in enumerate(pairs):
        labels = [{"id": int(x[4]), "category": classes[int(x[6])], "score": float(x[5]),
                   "box2d": {"x1": float(x[0]), "y1": float(x[1]), "x2": float(x[2]), "y2": float(x[3])}} for x in r]
        out.append({"video_name": seq, "name": fn, "index": i, "labels": labels})
    with open(path, "w") as fo:
        json.dump(out, fo)


def run(dataset: str, tag: str, tracker: str = "bytetrack", thr: float = 0.5, match_thresh: float = 0.8,
        track_buffer: int = 15, name: str | None = None, cache_root: str | None = None, verbose: bool = True) -> str:
    """track every <cache>/<tag>/<dataset>/*.npz and write the TrackEval input files; returns the tracker folder
    <outputs.results>/<dataset>_trackers/<name> (name default tracker_name(tag, tracker, thr))"""
    from ..data import BDD_CLASSES
    from ..data import get as get_data
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}")
    D = get_data(dataset)
    classes = list(BDD_CLASSES)
    keep_classes = ["pedestrian"] if dataset == "mot17" else None   # the MOTChallenge evaluator scores pedestrian only
    name = name or tracker_name(tag, tracker, thr)
    cache_dir = os.path.join(cache_root or paths.require("outputs.cache"), tag, dataset)
    trk_dir = os.path.join(paths.require("outputs.results"), f"{dataset}_trackers", name)
    out_dir = os.path.join(trk_dir, "data")
    os.makedirs(out_dir, exist_ok=True)
    tr_kw = dict(track_thresh=thr, match_thresh=match_thresh, track_buffer=track_buffer)
    files = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"no detection cache in {cache_dir}")
    t0 = time.time()
    for fi, f in enumerate(files):
        seq = os.path.splitext(os.path.basename(f))[0]
        z = np.load(f, allow_pickle=True); frames = list(z["frames"]); dets = _clean(list(z["dets"]), len(classes))
        hw = D.seq_hw(seq) if dataset in ("visdrone", "mot17") else D.native_hw()
        res = track_sequence(dets, hw, classes, tracker, thr, tr_kw, keep_classes=keep_classes)
        if dataset == "mot17":  # every evaluation-half frame, renumbered as in the half-split GT
            off = D.half_offset(seq); fno = [D.frame_no(str(fn)) for fn in frames]
            assert fno == D.eval_frames(seq), (seq, len(fno), fno[:2], off)
            with open(os.path.join(out_dir, seq + ".txt"), "w") as fo:
                for fn_, r in zip(fno, res):
                    for x in r:
                        assert classes[int(x[6])] == "pedestrian"
                        fo.write(f"{fn_ - off},{int(x[4])},{x[0]:.2f},{x[1]:.2f},{x[2] - x[0]:.2f},{x[3] - x[1]:.2f},{x[5]:.4f},-1,-1,-1\n")
        else:
            if dataset == "dsec":  # keep only the labelled (GT) frames, re-indexed in GT order
                keep = D.labelled_names(seq)
                pairs = [(fn, r) for fn, r in zip(frames, res) if fn in keep]
                assert len(pairs) == len(keep), (seq, len(pairs), len(keep))
            elif dataset == "visdrone":  # every frame is a GT frame; outputs centred in an ignored region are removed
                _, _, ign = D.read_annotations("test", seq)
                assert len(frames) == len(D.frame_paths("test", seq)), (seq, len(frames))
                pairs = [(fn, D.filter_ignored(r, ign.get(D.frame_no(str(fn))))) for fn, r in zip(frames, res)]
            else:
                pairs = list(zip(frames, res))
            _write_json(os.path.join(out_dir, seq + ".json"), seq, pairs, classes)
        if verbose and fi % 50 == 0:
            print(f"[{name}] {fi + 1}/{len(files)} {time.time() - t0:.0f}s", flush=True)
    if verbose:
        print("DONE", name, flush=True)
    return trk_dir


def _done(dataset: str, name: str) -> bool:
    d = os.path.join(paths.require("outputs.results"), f"{dataset}_trackers", name, "data")
    return os.path.isdir(d) and len(os.listdir(d)) >= NSEQ_TRACKED[dataset]


def _run_job(args: tuple) -> str:
    dataset, tag, trk, th, cache_root, cfg = args
    if cfg:
        os.environ[paths.ENV_VAR] = cfg
    run(dataset, tag, trk, float(th), cache_root=cache_root, verbose=False)
    return tracker_name(tag, trk, float(th))


def sweep(dataset: str, tag: str, trackers=("bytetrack", "ocsort"), jobs: int = 4, cache_root: str | None = None) -> list[str]:
    """run() over the threshold grid of each tracker in a process pool (no console windows), skipping tracker folders
    that already hold every sequence; returns the tracker names of the whole grid"""
    names = [(trk, th) for trk in trackers for th in GRID[trk]]
    cfg = paths._CFG_FILE or paths.config_file()   # the configuration this process uses, also in the workers
    todo = [(dataset, tag, trk, th, cache_root, cfg) for trk, th in names
            if not _done(dataset, tracker_name(tag, trk, float(th)))]
    if todo:
        if jobs <= 1:
            for j in todo:
                print("tracked", _run_job(j), flush=True)
        else:
            with ProcessPoolExecutor(min(jobs, len(todo))) as ex:
                for n in ex.map(_run_job, todo):
                    print("tracked", n, flush=True)
    print(f"TRACK_DONE {tag}: {len(todo)} of {len(names)} tracker outputs written", flush=True)
    return [tracker_name(tag, trk, float(th)) for trk, th in names]
