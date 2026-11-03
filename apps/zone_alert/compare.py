# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Offline evaluation of the zone alerts against MOT17 ground truth: plain YOLOv8n vs YOLOv8n + S3x (ByteTrack, seed 0).

Data: the evaluation halves (frames n//2+2 .. n; the detectors were trained on the first halves) of the static-camera
sequences MOT17-04 and MOT17-09, zones from zones/<seq>.json (fixed from scene geometry before any model output was
seen).

Ground-truth (GT) episodes
    The SAME ZoneMonitor as the app runs on the GT foot points of every annotated frame of every MOT17 pedestrian
    (class 1, consider 1), whatever its visibility; GT identity is exact, so a GT stay is never closed for missing
    frames. An episode starts at the first of the K inside frames, is confirmed at the K-th (the earliest a perfect
    tracker could alert) and ends at the GT exit or the person's last annotated frame. GT LOITERING = the monitor's
    LOITERING event of that episode. An episode is SCORED when the person is inside the zone with visibility >= 0.5
    on at least K of its frames; other episodes (people hidden for the whole stay) are neither counted as missed nor
    as alerted.

Alert attribution (per zone, alerts in time order; window W = the K frames up to and including the alert frame f)
    * An INTRUSION of track T at frame f belongs to GT episode E of person g in the same zone when T is matched to g
      (per-frame Hungarian, IoU >= 0.5, all GT pedestrians whatever their visibility) on at least one frame of W and
      E.start - ATTR_S * fps <= f <= E.end. Several candidates -> the one with the closest confirmation frame.
      Scored E: first alert -> 'hit' (delay = f - E.confirm), later alerts -> 'duplicate' (e.g. a new track id after an
      identity switch). Unscored E -> 'unscored'.
    * Not attributed: T matched on W to a GT pedestrian whose GT foot point is in the zone, or lying on a MOT17
      'person on vehicle' (2) / 'static person' (7) box -> 'unscored' (a real person in the zone without an evaluable
      episode); matched to a GT pedestrian outside the zone -> 'false_person_outside'; on a 'distractor' (8) or
      'reflection' (12) box -> 'false_distractor'; on nothing -> 'false_no_person'. False alerts = the three 'false'
      kinds.
    * LOITERING: same attribution; hit when the episode has a GT LOITERING (delay to it), 'early' when it has none
      (counted as false loitering), else as above.

Thresholds: each model's cross-fitted per-sequence ByteTrack threshold (efpmot.eval.protocol, method()['thr']['cross'],
seed 0: leave-one-sequence-out over the 7 MOT17 evaluation halves; never chosen on the scored sequence; read from
results/stats). The tracks are checked against the benchmark tracker outputs <results>/mot17_trackers (scripts/track.py).
Weights: <runs>/<run>/last.pth when it exists, else the release file of the run (efpmot.release, release_assets/).

usage (repository root): python -B apps/zone_alert/compare.py [--reuse]
writes apps/zone_alert/results.json, results.md (report.py), cache/<seq>__<model>.json (per-frame tracks and events,
used by render_demo.py). Timing is measured separately by bench.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zone_alert as za  # noqa: E402  (also puts the repository root on the path)
from efpmot import paths  # noqa: E402
from efpmot.data import mot17 as md  # noqa: E402
import report  # noqa: E402

SEQS = ["MOT17-04-FRCNN", "MOT17-09-FRCNN"]
MODELS = {  # name -> (family, run folder, protocol variant)
    "plain": ("yolov8n", "mot_v8n_416_s0", "416"),
    "s3x": ("yolov8n_s3x", "mot_v8n_416s3x_s0", "416s3x"),
}
ZONE_KINDS = ("far", "near")
VIS_MIN = 0.5           # an episode is scored when the person is >= 50 % visible inside the zone on >= K frames
MATCH_IOU = 0.5         # track <-> GT person association (primary)
SENS_IOU = 0.3          # post-hoc sensitivity: loose boxes on small, crowded people
ATTR_S = 1.0            # an alert up to 1 s before the GT episode starts may still belong to it
PEOPLE_CLASSES = (2, 7)       # MOT17 person on vehicle, static person: real people, alerts on them are 'unscored'
NON_PEOPLE_CLASSES = (8, 12)  # MOT17 distractor, reflection: alerts on them are false
FALSE_KINDS = ("false_person_outside", "false_distractor", "false_no_person")
CACHE = os.path.join(HERE, "cache")
FPS = 30.0              # both sequences


# ---------------------------------------------------------------------------------------------------------- inputs
@dataclass
class GroundTruth:
    peds: dict            # frame -> list of (gid, box xyxy, visibility), class 1 / consider 1, unclipped
    people_other: dict    # frame -> [N, 4] boxes of classes 2 and 7
    non_people: dict      # frame -> [N, 4] boxes of classes 8 and 12


def thresholds(variant: str) -> dict[str, float]:
    """Cross-fitted per-sequence ByteTrack threshold of seed 0 (leave-one-sequence-out, efpmot.eval.protocol)."""
    from efpmot.eval import protocol
    method = protocol.method("mot17", "v8n", variant, "bytetrack", "pedestrian")
    per_fold = method["thr"]["cross"][method["seeds"].index(0)]
    seqs, folds = protocol.REFSEQ["mot17"], protocol.folds_for("mot17")
    fold_ids = list(np.unique(folds))
    return {seq: float(per_fold[fold_ids.index(folds[i])]) for i, seq in enumerate(seqs)}


def read_gt(seq: str) -> GroundTruth:
    """Read gt.txt of a MOT17 training sequence (from data.mot17_train)."""
    peds, people_other, non_people = {}, {}, {}
    with open(os.path.join(paths.require("data.mot17_train"), seq, "gt", "gt.txt")) as fh:
        for line in fh:
            p = line.strip().split(",")
            frame, gid = int(p[0]), int(p[1])
            left, top, w, h = (float(v) for v in p[2:6])
            consider, cls, vis = int(float(p[6])), int(float(p[7])), float(p[8])
            box = np.array([left, top, left + w, top + h], np.float32)
            if cls == md.PED_CLASS and consider == 1:
                peds.setdefault(frame, []).append((gid, box, vis))
            elif cls in PEOPLE_CLASSES:
                people_other.setdefault(frame, []).append(box)
            elif cls in NON_PEOPLE_CLASSES:
                non_people.setdefault(frame, []).append(box)
    return GroundTruth(peds, {f: np.stack(v) for f, v in people_other.items()},
                       {f: np.stack(v) for f, v in non_people.items()})


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xyxy boxes: a [N, 4], b [M, 4] -> [N, M]."""
    a = a.reshape(-1, 4).astype(np.float64)
    b = b.reshape(-1, 4).astype(np.float64)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


# ---------------------------------------------------------------------------------------------------------- model runs
def weights_for(run_dir: str) -> str:
    """Checkpoint of a run: <runs>/<run>/last.pth when it exists, else its release file (efpmot.release)."""
    from efpmot.release import checkpoint_for
    p = checkpoint_for(run_dir)
    if p is None:
        raise FileNotFoundError(f"no checkpoint for {run_dir}: neither <runs>/{run_dir}/last.pth nor its release file in "
                                "release_assets/ (see release_assets/MODEL_CARD.md)")
    return p


def run_model(seq: str, name: str, thr: float, zones: list[za.Zone], fps: float) -> dict:
    """Detect + track + zone logic over the evaluation half -> per-frame tracks and events."""
    family, run_dir, _ = MODELS[name]
    detector = za.PedestrianDetector(family, weights_for(run_dir))
    tracker = za.PedestrianTracker(thr, fps)
    frames = md.eval_frames(seq)
    monitor = za.ZoneMonitor(zones, fps, frame0=frames[0])
    frame_iter = ((f, cv2.imread(md.frame_path(seq, f))) for f in frames)
    tracks, events = {}, []
    for step in za.pipeline(frame_iter, detector, tracker, monitor):
        tracks[step.frame] = step.tracks.tolist()
        events += [asdict(e) for e in step.events]
    return dict(seq=seq, model=name, thr=thr, tracks=tracks, events=events)


def _same_rows(ours: list[tuple], ref: list[tuple], tol_xy: float = 0.006, tol_conf: float = 1e-4) -> bool:
    """Two sorted lists of (id, x, y, w, h, conf) rows describe the same tracks: equal ids, coordinates within the
    rounding of the stored file (2 decimals) and confidences within its 4 decimals."""
    if len(ours) != len(ref):
        return False
    for a, b in zip(ours, ref):
        if a[0] != b[0]:
            return False
        if max(abs(x - y) for x, y in zip(a[1:5], b[1:5])) > tol_xy or abs(a[5] - b[5]) > tol_conf:
            return False
    return True


def benchmark_consistency(seq: str, model: str, thr: float, tracks: dict) -> dict:
    """Compare the app's tracks with the stored benchmark tracker output on every evaluation frame, all columns."""
    run_dir = MODELS[model][1]
    results = paths.require("outputs.results")
    path = os.path.join(results, "mot17_trackers", f"{run_dir}__bytetrack_t{thr:g}", "data", seq + ".txt")
    offset = md.half_offset(seq)
    ref = {}
    with open(path) as fh:
        for line in fh:
            p = line.split(",")
            ref.setdefault(int(p[0]) + offset, []).append((int(p[1]), *(float(v) for v in p[2:7])))
    frames = md.eval_frames(seq)
    same, max_xy, max_conf = 0, 0.0, 0.0
    for f in frames:
        ours = sorted((int(r[4]), r[0], r[1], r[2] - r[0], r[3] - r[1], r[5]) for r in tracks[f])
        theirs = sorted(ref.get(f, []))
        same += _same_rows(ours, theirs)
        if len(ours) == len(theirs):
            for a, b in zip(ours, theirs):
                max_xy = max(max_xy, *(abs(x - y) for x, y in zip(a[1:5], b[1:5])))
                max_conf = max(max_conf, abs(a[5] - b[5]))
    return dict(exact_frames=f"{same}/{len(frames)}", max_abs_diff_xywh=round(max_xy, 5),
                max_abs_diff_conf=round(max_conf, 6),
                benchmark_file="<results>/" + os.path.relpath(path, results).replace(os.sep, "/"))


# ---------------------------------------------------------------------------------------------------------- GT episodes
def gt_episodes(seq: str, zones: list[za.Zone], fps: float, gt: GroundTruth) -> list[dict]:
    """GT zone stays of every pedestrian (all annotated frames), with the 'scored' flag (see module docstring)."""
    frames = md.eval_frames(seq)
    monitor = za.ZoneMonitor(zones, fps, max_missing=None, frame0=frames[0])
    first_seen, last_seen = {}, {}
    for f in frames:
        for gid, _, _ in gt.peds.get(f, []):
            first_seen.setdefault(gid, f)
            last_seen[gid] = f
    episodes, open_episodes = [], {}
    for f in frames:
        feet = {gid: za.foot_point(box) for gid, box, _ in gt.peds.get(f, [])}
        for event in monitor.update(f, feet):
            key = (event.track_id, event.zone)
            if event.type == "INTRUSION":
                episode = dict(gid=event.track_id, zone=event.zone, start=event.entered, confirm=event.frame,
                               end=last_seen[event.track_id], loiter=None, initial=event.entered == frames[0],
                               appeared_inside=event.entered == first_seen[event.track_id])
                open_episodes[key] = episode
                episodes.append(episode)
            elif event.type == "LOITERING":
                open_episodes[key]["loiter"] = event.frame
            elif event.type == "EXIT":
                open_episodes.pop(key)["end"] = event.frame
    zone_by_name = {z.name: z for z in zones}
    for episode in episodes:
        zone = zone_by_name[episode["zone"]]
        visible_inside = 0
        for f in range(episode["start"], episode["end"] + 1):
            for gid, box, vis in gt.peds.get(f, []):
                if gid == episode["gid"] and vis >= VIS_MIN - 1e-9 and zone.contains(za.foot_point(box)):
                    visible_inside += 1
        episode["visible_inside_frames"] = visible_inside
        episode["scored"] = visible_inside >= za.K_FRAMES
    return episodes


# ---------------------------------------------------------------------------------------------------------- attribution
def track_gt_match(tracks: dict, gt: GroundTruth, iou_thr: float) -> tuple[dict, set, set]:
    """Per-frame Hungarian matching (IoU >= iou_thr) of tracks to GT pedestrians -> {(frame, tid): gid}.
    Unmatched tracks lying on a class 2/7 box or on a class 8/12 box (IoU >= iou_thr) -> two sets of (frame, tid)."""
    matches, on_people, on_non_people = {}, set(), set()
    for f, rows in tracks.items():
        if not rows:
            continue
        boxes = np.asarray(rows, np.float32)
        ids = boxes[:, 4].astype(int)
        matched = set()
        persons = gt.peds.get(f, [])
        if persons:
            overlap = iou(boxes[:, :4], np.stack([box for _, box, _ in persons]))
            for r, c in zip(*linear_sum_assignment(-overlap)):
                if overlap[r, c] >= iou_thr:
                    matches[(f, int(ids[r]))] = persons[c][0]
                    matched.add(r)
        for others, target in ((gt.people_other.get(f), on_people), (gt.non_people.get(f), on_non_people)):
            if others is None:
                continue
            overlap = iou(boxes[:, :4], others)
            for r in range(len(boxes)):
                if r not in matched and overlap[r].max() >= iou_thr:
                    target.add((f, int(ids[r])))
    return matches, on_people, on_non_people


def attribute(run: dict, episodes: list[dict], zones: list[za.Zone], gt: GroundTruth, fps: float,
              iou_thr: float = MATCH_IOU) -> dict:
    """Classify every INTRUSION / LOITERING alert of a model run against the GT episodes (see module docstring).
    Returns {"alerts": [...], "intrusion": {episode: delay or None}, "loitering": {episode: delay or None}}."""
    tracks = {int(f): rows for f, rows in run["tracks"].items()}
    matches, on_people, on_non_people = track_gt_match(tracks, gt, iou_thr)
    zone_by_name = {z.name: z for z in zones}
    gt_feet = {f: {gid: za.foot_point(box) for gid, box, _ in rows} for f, rows in gt.peds.items()}
    credited = {"INTRUSION": set(), "LOITERING": set()}
    alerts = []
    for event in sorted(run["events"], key=lambda e: e["frame"]):
        if event["type"] == "EXIT":
            continue
        frame, tid, zone_name = event["frame"], event["track_id"], event["zone"]
        window = range(frame - za.K_FRAMES + 1, frame + 1)
        gids = [matches.get((f, tid)) for f in window]
        candidates = [i for i, ep in enumerate(episodes)
                      if ep["zone"] == zone_name and ep["gid"] in gids
                      and ep["start"] - ATTR_S * fps <= frame <= ep["end"]]
        episode_idx, delay = None, None
        if candidates:
            episode_idx = min(candidates, key=lambda i: abs(episodes[i]["confirm"] - frame))
            episode = episodes[episode_idx]
            ref = episode["confirm"] if event["type"] == "INTRUSION" else episode["loiter"]
            if not episode["scored"]:
                outcome = "unscored"
            elif ref is None:
                outcome = "early"            # LOITERING on an episode whose GT stay never reached dwell_s
            elif episode_idx in credited[event["type"]]:
                outcome = "duplicate"
            else:
                outcome, delay = "hit", frame - ref
                credited[event["type"]].add(episode_idx)
        else:
            zone = zone_by_name[zone_name]
            in_zone = any(g is not None and g in gt_feet.get(f, {}) and zone.contains(gt_feet[f][g])
                          for f, g in zip(window, gids))
            if in_zone or any((f, tid) in on_people for f in window):
                outcome = "unscored"
            elif any(g is not None for g in gids):
                outcome = "false_person_outside"
            elif any((f, tid) in on_non_people for f in window):
                outcome = "false_distractor"
            else:
                outcome = "false_no_person"
        alerts.append(dict(type=event["type"], zone=zone_name, track_id=tid, frame=frame, outcome=outcome,
                           episode=episode_idx, delay=delay))
    result = {"alerts": alerts, "intrusion": {}, "loitering": {}}
    hit_delay = {(a["type"], a["episode"]): a["delay"] for a in alerts if a["outcome"] == "hit"}
    for i, episode in enumerate(episodes):
        if not episode["scored"]:
            continue
        result["intrusion"][i] = hit_delay.get(("INTRUSION", i))
        if episode["loiter"] is not None:
            result["loitering"][i] = hit_delay.get(("LOITERING", i))
    return result


# ---------------------------------------------------------------------------------------------------------- summaries
def _seconds(values: list[float], fps: float, stat: Callable) -> float | None:
    """stat(values) in frames -> seconds (None for an empty list)."""
    return round(float(stat(values)) / fps, 3) if values else None


def _p90(values: list[float]) -> float:
    return float(np.percentile(values, 90))


def episode_summary(episodes: list[dict], att: dict, fps: float, keep: Callable[[dict], bool]) -> dict:
    """Recall and delay over the scored episodes selected by keep(episode)."""
    idx = [i for i, ep in enumerate(episodes) if ep["scored"] and keep(ep)]
    delays = [att["intrusion"][i] for i in idx if att["intrusion"][i] is not None]
    loiter_idx = [i for i in idx if i in att["loitering"]]
    loiter_delays = [att["loitering"][i] for i in loiter_idx if att["loitering"][i] is not None]
    return dict(
        gt_episodes=len(idx), alerted=len(delays), recall=round(len(delays) / len(idx), 3) if idx else None,
        unscored_episodes=sum(not ep["scored"] and keep(ep) for ep in episodes),
        median_delay_frames=float(np.median(delays)) if delays else None,
        median_delay_s=_seconds(delays, fps, np.median), mean_delay_s=_seconds(delays, fps, np.mean),
        p90_delay_s=_seconds(delays, fps, _p90), negative_delays=sum(d < 0 for d in delays),
        gt_loitering=len(loiter_idx), loitering_hit=len(loiter_delays),
        loitering_median_delay_s=_seconds(loiter_delays, fps, np.median))


def alert_summary(alerts: list[dict], keep: Callable[[dict], bool]) -> dict:
    """Counts of INTRUSION / LOITERING outcomes over the alerts selected by keep(alert)."""
    chosen = [a for a in alerts if keep(a)]

    def count(kind: str, outcome: str) -> int:
        return sum(a["type"] == kind and a["outcome"] == outcome for a in chosen)

    out = dict(intrusion_alerts=sum(a["type"] == "INTRUSION" for a in chosen),
               false_alerts=sum(count("INTRUSION", k) for k in FALSE_KINDS),
               duplicates=count("INTRUSION", "duplicate"), unscored=count("INTRUSION", "unscored"),
               loitering_false=sum(count("LOITERING", k) for k in ("early",) + FALSE_KINDS),
               loitering_duplicates=count("LOITERING", "duplicate"))
    out.update({k: count("INTRUSION", k) for k in FALSE_KINDS})
    return out


def person_size(episodes: list[dict], gt: GroundTruth, keep: Callable[[dict], bool]) -> dict:
    """Median GT box height (source px and 416x256 canvas px) at the confirmation frames of the scored episodes."""
    heights = []
    for ep in episodes:
        if ep["scored"] and keep(ep):
            box = next(box for gid, box, _ in gt.peds[ep["confirm"]] if gid == ep["gid"])
            heights.append(float(box[3] - box[1]))
    if not heights:
        return dict(median_height_px=None, median_height_canvas_px=None)
    scale = min(256 / 1080, 416 / 1920)
    median = float(np.median(heights))
    return dict(median_height_px=round(median, 1), median_height_canvas_px=round(median * scale, 1))


def of_zone(name: str) -> Callable[[dict], bool]:
    """Select episodes / alerts of one zone."""
    def keep(item: dict) -> bool:
        return item["zone"] == name
    return keep


def of_kind(kind: str, entries_only: bool = False) -> Callable[[dict], bool]:
    """Select episodes / alerts of one zone kind; entries_only keeps episodes that start after the first frame."""
    def keep(item: dict) -> bool:
        return item["zone_kind"] == kind and not (entries_only and item.get("initial", False))
    return keep


def summary(episodes: list[dict], att: dict, fps: float, keep: Callable[[dict], bool]) -> dict:
    """Episode and alert summaries of one selection, merged."""
    return {**episode_summary(episodes, att, fps, keep), **alert_summary(att["alerts"], keep)}


# ---------------------------------------------------------------------------------------------------------- main
def load_or_run(seq: str, model: str, thr: float, zones: list[za.Zone], fps: float, reuse: bool) -> dict:
    """Model run of one sequence, from cache/ when --reuse is given and the file exists."""
    path = os.path.join(CACHE, f"{seq[:8]}__{model}.json")
    if reuse and os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    run = run_model(seq, model, thr, zones, fps)
    with open(path, "w") as fh:
        json.dump(run, fh)
    return run


def pool(pooled: dict, episodes: list[dict], att: dict, offset: int) -> None:
    """Append one sequence's attribution to the pooled one, re-indexing its episodes by `offset`."""
    pooled["intrusion"].update({offset + i: d for i, d in att["intrusion"].items()})
    pooled["loitering"].update({offset + i: d for i, d in att["loitering"].items()})
    for alert in att["alerts"]:
        episode = None if alert["episode"] is None else offset + alert["episode"]
        pooled["alerts"].append(dict(alert, episode=episode))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true",
                    help="reuse cache/<seq>__<model>.json instead of running the models")
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    thr = {m: thresholds(v[2]) for m, v in MODELS.items()}
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    results = dict(meta=dict(created=time.strftime("%Y-%m-%d %H:%M"), k=za.K_FRAMES, vis_min=VIS_MIN,
                             attr_s=ATTR_S, match_iou=MATCH_IOU, sens_iou=SENS_IOU, gpu=gpu),
                   seqs=SEQS, models=list(MODELS), thresholds={m: {s: thr[m][s] for s in SEQS} for m in MODELS},
                   zones={}, per_zone={}, sizes={}, consistency={}, episodes={}, alerts={})
    all_episodes = []
    pooled_att = {t: {m: {"intrusion": {}, "loitering": {}, "alerts": []} for m in MODELS}
                  for t in (MATCH_IOU, SENS_IOU)}
    for seq in SEQS:
        zones = za.load_zones(os.path.join(HERE, "zones", seq[:8] + ".json"))
        fps = float(md.seqinfo(seq)["fps"])
        kind = {z.name: z.kind for z in zones}
        results["zones"][seq] = [dict(name=z.name, kind=z.kind, dwell_s=z.dwell_s) for z in zones]
        gt = read_gt(seq)
        episodes = gt_episodes(seq, zones, fps, gt)
        for ep in episodes:
            ep["seq"], ep["zone_kind"] = seq, kind[ep["zone"]]
        results["episodes"][seq] = episodes
        results["sizes"][seq] = {z.name: person_size(episodes, gt, of_zone(z.name)) for z in zones}
        results["per_zone"][seq] = {z.name: {} for z in zones}
        results["consistency"][seq], results["alerts"][seq] = {}, {}
        for model in MODELS:
            run = load_or_run(seq, model, thr[model][seq], zones, fps, args.reuse)
            tracks = {int(f): rows for f, rows in run["tracks"].items()}
            check = benchmark_consistency(seq, model, thr[model][seq], tracks)
            results["consistency"][seq][model] = check
            for iou_thr in (MATCH_IOU, SENS_IOU):
                att = attribute(run, episodes, zones, gt, fps, iou_thr)
                for alert in att["alerts"]:
                    alert["seq"], alert["zone_kind"] = seq, kind[alert["zone"]]
                if iou_thr == MATCH_IOU:
                    results["alerts"][seq][model] = att["alerts"]
                    for z in zones:
                        results["per_zone"][seq][z.name][model] = summary(episodes, att, fps, of_zone(z.name))
                pool(pooled_att[iou_thr][model], episodes, att, len(all_episodes))
            print(f"{seq} {model}: thr {thr[model][seq]:g}, benchmark tracks reproduced {check['exact_frames']} "
                  f"(max |diff| xywh {check['max_abs_diff_xywh']}, conf {check['max_abs_diff_conf']})", flush=True)
        all_episodes += episodes
    for key, iou_thr, entries_only in (("pooled", MATCH_IOU, False), ("pooled_entries_only", MATCH_IOU, True),
                                       ("pooled_iou03", SENS_IOU, False)):
        results[key] = {k: {m: summary(all_episodes, pooled_att[iou_thr][m], FPS, of_kind(k, entries_only))
                            for m in MODELS} for k in ZONE_KINDS}
    with open(os.path.join(HERE, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    report.write_results_md(results, os.path.join(HERE, "results.md"))
    for k in ZONE_KINDS:
        for m in MODELS:
            s = results["pooled"][k][m]
            print(f"{k:4s} {m:5s}: alerted {s['alerted']}/{s['gt_episodes']}  median delay "
                  f"{report.fmt(s['median_delay_s'])} s  p90 {report.fmt(s['p90_delay_s'])} s  false "
                  f"{s['false_alerts']}  dup {s['duplicates']}  loiter {s['loitering_hit']}/{s['gt_loitering']} "
                  f"false {s['loitering_false']}")


if __name__ == "__main__":
    main()
