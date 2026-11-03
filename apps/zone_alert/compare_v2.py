# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Offline evaluation of the motion features (APPROACH warnings, direction-filtered INTRUSION, tripwires) against MOT17
ground truth, for plain YOLOv8n and YOLOv8n + S3x (ByteTrack, seed 0), on the cached tracks of compare.py.

Data and episodes are exactly those of compare.py (evaluation halves of MOT17-04 / MOT17-09, GT pedestrians on every
annotated frame, 'scored' = >= 0.5 visible inside the zone on >= K frames). The zone files are zones/<seq>_v2.json:
the same polygons with the motion options, chosen from scene geometry before any output of the new logic was seen.
compare.py, results.md / .json and the cache are not changed; the model events are replayed from cache/.

Ground truth
    * Walk-in episode: a scored GT episode that starts after the first evaluation frame of a person seen outside the
      zone before (not 'initial', not 'appeared_inside').
    * GT entry direction / decision: the SAME ZoneMonitor (v2 zone file) run on the GT foot points and GT box heights
      (never closing a silent person) labels every GT entry INTRUSION or ENTRY with the same rule as the app.
    * GT crossing: the same tripwire logic on the GT foot points. Scored when the person is >= 0.5 visible on >= K
      of the frames from K before the first frame on the new side to K after the confirmation.

Association: as compare.py, a model event at frame f belongs to GT person g when its track is matched to g (per-frame
Hungarian, IoU >= 0.5) on at least one of the K frames up to f.

APPROACH (per warning, in time order; H = the zone's horizon_s)
    hit           the matched person starts a walk-in episode of that zone in (f, f + (H + 1 s)]; lead = (episode start
                  - f) / fps, i.e. how long before the GT foot point entered; a later warning for the same episode is a
                  'duplicate'; an entry in that window that is not a target walk-in (unscored, or the wrong direction
                  in the gated run) is 'other_entry'
    late          the matched person is already inside the zone (GT) at f
    censored      no entry, but the H + 1 s window runs past the last evaluation frame: cannot be judged (added after
                  a first run showed that most of the GT oracle's 'false' warnings fell in the last seconds of a clip)
    false         'false_no_entry': the matched person does not enter that zone within H + 1 s;
                  'false_no_person': the track is not matched to any GT pedestrian on those K frames
    Two runs: 'ungated' (direction filters removed, targets = all walk-ins) isolates the APPROACH rule; 'v2' (the zone
    files as configured, APPROACH only in the alarm direction, targets = walk-ins whose GT decision is INTRUSION).
    The same scoring on the GT trajectories themselves ('GT oracle') shows what the rule does with perfect tracks.
    'useful' hits have a lead >= 0.5 s (a warning 0.03 s before the entry counts as a hit but helps nobody).
    Rule variants (ungated, post hoc, on this evaluation data; reported only, never used to set a default): the
    fit-residual check off, and lower min_speed values. The residual cut itself was set on MOT17-02 (calibrate_v2.py).

Direction filter: per walk-in episode of a zone with a filter, the model's decision is the type (INTRUSION / ENTRY) of
the first entry event attributed to it (attribution as compare.py); agreement = equal to the GT decision, over the
walk-ins that have a model decision.

Tripwire: a model LINE_CROSS is a hit when its matched person has a scored GT crossing of that line within 1 s (first
one per crossing; later ones are duplicates); recall over the scored GT crossings.

usage (repository root, after compare.py): python -B apps/zone_alert/compare_v2.py   -> results_v2.json, results_v2.md
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, replace
from typing import Callable

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zone_alert as za  # noqa: E402
import compare  # noqa: E402
from efpmot.data import mot17 as md  # noqa: E402
from report import fmt, table  # noqa: E402

K = za.K_FRAMES
KINDS = ("far", "near")
SOURCES = ("gt", "plain", "s3x")
LABEL = {"plain": "plain", "s3x": "S3x", "gt": "GT oracle"}
BASE_FIELDS = ("type", "zone", "track_id", "frame", "t", "entered", "foot")   # the v1 event record
USEFUL_S = 0.5          # a hit is 'useful' when the warning comes at least this long before the GT entry
VARIANTS = {            # post-hoc variants of the ungated APPROACH rule: ApproachRule fields that differ
    "as configured": {}, "residual check off": dict(max_residual=None),
    "min_speed 0.20": dict(min_speed=0.20), "min_speed 0.15": dict(min_speed=0.15),
    "min_speed 0.10": dict(min_speed=0.10)}
FAR_BAND_04 = (70, 330)   # foot rows of the MOT17-04 far zone, for the scene speeds


# ---------------------------------------------------------------------------------------------------------- replays
def replay(frames: list[int], obs: Callable[[int], tuple[dict, dict]], zones: list[za.Zone],
           lines: list[za.Tripwire], fps: float, max_missing: int | None) -> list[za.Event]:
    """Run a ZoneMonitor over per-frame observations ({id: foot}, {id: box height}) -> all events."""
    monitor = za.ZoneMonitor(zones, fps, frame0=frames[0], max_missing=max_missing, lines=lines)
    events = []
    for f in frames:
        events += monitor.update(f, *obs(f))
    return events


def gt_obs(gt: compare.GroundTruth) -> Callable[[int], tuple[dict, dict]]:
    """GT pedestrians of a frame as monitor observations (GT id = track id)."""
    def obs(f: int) -> tuple[dict, dict]:
        rows = gt.peds.get(f, [])
        return ({gid: za.foot_point(box) for gid, box, _ in rows},
                {gid: float(box[3] - box[1]) for gid, box, _ in rows})
    return obs


def track_obs(run: dict) -> Callable[[int], tuple[dict, dict]]:
    """Cached model tracks of a frame as monitor observations."""
    def obs(f: int) -> tuple[dict, dict]:
        return za.observations(np.asarray(run["tracks"][str(f)], np.float32).reshape(-1, 6))
    return obs


def model_who(matches: dict) -> Callable[[za.Event], set]:
    """GT persons matched to the event's track on one of the K frames up to the event."""
    def who(e: za.Event) -> set:
        return {matches[(f, e.track_id)] for f in range(e.frame - K + 1, e.frame + 1) if (f, e.track_id) in matches}
    return who


def gt_who(e: za.Event) -> set:
    return {e.track_id}


# ---------------------------------------------------------------------------------------------------------- scoring
def walk_ins(episodes: list[dict]) -> set[int]:
    """Scored episodes that start after the first frame, of people seen outside the zone before."""
    return {i for i, ep in enumerate(episodes) if ep["scored"] and not ep["initial"] and not ep["appeared_inside"]}


def score_approach(events: list[za.Event], episodes: list[dict], who: Callable, fps: float,
                   horizon: dict[str, float], targets: set[int], last_frame: int) -> dict:
    """Classify every APPROACH warning (module docstring) -> {"warnings": [...], "lead": {episode: seconds}}."""
    warnings, lead = [], {}
    for e in sorted((e for e in events if e.type == "APPROACH"), key=lambda e: e.frame):
        gids = who(e)
        window = (horizon[e.zone] + 1.0) * fps
        entering = [i for i, ep in enumerate(episodes) if ep["zone"] == e.zone and ep["gid"] in gids
                    and e.frame < ep["start"] <= e.frame + window]
        if not gids:
            outcome = "false_no_person"
        elif entering:
            i = min(entering, key=lambda j: episodes[j]["start"])
            if i not in targets:
                outcome = "other_entry"
            elif i in lead:
                outcome = "duplicate"
            else:
                outcome, lead[i] = "hit", round((episodes[i]["start"] - e.frame) / fps, 3)
        elif any(ep["zone"] == e.zone and ep["gid"] in gids and ep["start"] <= e.frame <= ep["end"] for ep in episodes):
            outcome = "late"
        elif e.frame + window > last_frame:
            outcome = "censored"
        else:
            outcome = "false_no_entry"
        warnings.append(dict(zone=e.zone, track_id=e.track_id, frame=e.frame, eta_s=e.eta_s, outcome=outcome))
    return dict(warnings=warnings, lead=lead)


def best_iou(warning: dict, run: dict, gt: compare.GroundTruth) -> tuple[float, float]:
    """Highest IoU of the warning's track box on the K frames up to the warning with any GT pedestrian box, and with
    any MOT17 'person on vehicle' / 'static person' box (classes 2 / 7, e.g. a seated person)."""
    best = [0.0, 0.0]
    for f in range(warning["frame"] - K + 1, warning["frame"] + 1):
        rows = [r for r in run["tracks"].get(str(f), []) if int(r[4]) == warning["track_id"]]
        if not rows:
            continue
        box = np.asarray(rows[0][:4])
        if gt.peds.get(f):
            best[0] = max(best[0], float(compare.iou(box, np.stack([b for _, b, _ in gt.peds[f]])).max()))
        if f in gt.people_other:
            best[1] = max(best[1], float(compare.iou(box, gt.people_other[f]).max()))
    return round(best[0], 3), round(best[1], 3)


def scene_speeds(gt: compare.GroundTruth, frames: list[int], fps: float, rows: tuple | None = None) -> dict:
    """GT walking speeds of a scene: foot displacement over 1 s (start points every 0.5 s) / mean box height, for
    movers >= 0.1 BH/s with the foot in `rows` (image rows; None = anywhere); 'sideways' = |dx| > 2 |dy| (not
    foreshortened), 'along' = |dy| > 2 |dx| (along the viewing direction). Also a line fit of box height against foot
    row (visibility > 0.8, every 10th frame): how much the person size changes with depth."""
    step = int(round(fps))
    boxes = {}
    for f in frames:
        for gid, box, _ in gt.peds.get(f, []):
            boxes.setdefault(gid, {})[f] = box
    speeds = {"all": [], "sideways": [], "along": []}
    for per in boxes.values():
        for f in range(frames[0], frames[-1] - step + 1, step // 2):
            if f not in per or f + step not in per:
                continue
            a, b = per[f], per[f + step]
            if rows is not None and not rows[0] <= (a[3] + b[3]) / 2 <= rows[1]:
                continue
            h = ((a[3] - a[1]) + (b[3] - b[1])) / 2
            dx, dy = ((b[0] + b[2]) - (a[0] + a[2])) / 2 / h, (b[3] - a[3]) / h
            speed = float(np.hypot(dx, dy))
            if speed < 0.1:
                continue
            speeds["all"].append(speed)
            if abs(dx) > 2 * abs(dy):
                speeds["sideways"].append(speed)
            elif abs(dy) > 2 * abs(dx):
                speeds["along"].append(speed)
    out = {k: dict(n=len(v), median=round(float(np.median(v)), 3), p25=round(float(np.percentile(v, 25)), 3),
                   p75=round(float(np.percentile(v, 75)), 3)) if v else dict(n=0) for k, v in speeds.items()}
    ys, hs = zip(*[(box[3], box[3] - box[1]) for f in frames[::10] for _, box, vis in gt.peds.get(f, []) if vis > 0.8])
    slope, intercept = np.polyfit(ys, hs, 1)
    return dict(out, rows=rows, height_vs_row=dict(slope=round(float(slope), 4), intercept=round(float(intercept), 1)))


def gt_decisions(events: list[za.Event]) -> dict[tuple, dict]:
    """(gid, zone, start) -> GT entry label and velocity (body heights / s) from the GT replay."""
    return {(e.track_id, e.zone, e.entered): dict(type=e.type, velocity=getattr(e, "velocity", None))
            for e in events if e.type in ("INTRUSION", "ENTRY")}


def model_decisions(events: list[za.Event], episodes: list[dict], who: Callable, fps: float) -> dict[int, tuple]:
    """Episode index -> (type, direction unknown?) of the first INTRUSION / ENTRY attributed to it (attribution as
    compare.py)."""
    decided = {}
    for e in sorted((e for e in events if e.type in ("INTRUSION", "ENTRY")), key=lambda e: e.frame):
        gids = who(e)
        cands = [i for i, ep in enumerate(episodes) if ep["zone"] == e.zone and ep["gid"] in gids
                 and ep["start"] - compare.ATTR_S * fps <= e.frame <= ep["end"]]
        if cands:
            decided.setdefault(min(cands, key=lambda i: abs(episodes[i]["confirm"] - e.frame)),
                               (e.type, heading(getattr(e, "velocity", None)) == "unknown"))
    return decided


def heading(velocity: tuple | None) -> str:
    """Image-plane heading of an entry velocity: up / down / left / right (nearest axis), or unknown."""
    if velocity is None or np.hypot(*velocity) < za.MIN_SPEED:
        return "unknown"
    vx, vy = velocity
    if abs(vx) > abs(vy):
        return "right" if vx > 0 else "left"
    return "down" if vy > 0 else "up"


def gt_crossings(events: list[za.Event], gt: compare.GroundTruth) -> list[dict]:
    """GT LINE_CROSS events with the 'scored' flag (visibility rule of the module docstring)."""
    out = []
    for e in events:
        if e.type != "LINE_CROSS":
            continue
        visible = sum(1 for f in range(e.entered - K, e.frame + K + 1) for gid, _, vis in gt.peds.get(f, [])
                      if gid == e.track_id and vis >= compare.VIS_MIN - 1e-9)
        out.append(dict(line=e.zone, gid=e.track_id, frame=e.frame, scored=visible >= K))
    return out


def score_lines(events: list[za.Event], crossings: list[dict], who: Callable, fps: float) -> dict:
    """Classify model LINE_CROSS events against the GT crossings -> counts and delays."""
    credited, outcomes, delays = set(), [], []
    for e in sorted((e for e in events if e.type == "LINE_CROSS"), key=lambda e: e.frame):
        gids = who(e)
        cands = [i for i, c in enumerate(crossings) if c["line"] == e.zone and c["gid"] in gids
                 and abs(c["frame"] - e.frame) <= fps]
        if not gids:
            outcomes.append("false_no_person")
        elif not cands:
            outcomes.append("false_no_cross")
        else:
            i = min(cands, key=lambda j: abs(crossings[j]["frame"] - e.frame))
            if not crossings[i]["scored"]:
                outcomes.append("unscored")
            elif i in credited:
                outcomes.append("duplicate")
            else:
                credited.add(i)
                outcomes.append("hit")
                delays.append((e.frame - crossings[i]["frame"]) / fps)
    gt_n = sum(c["scored"] for c in crossings)
    return dict(gt_crossings=gt_n, gt_unscored=len(crossings) - gt_n, model_crossings=len(outcomes),
                hit=len(credited), recall=round(len(credited) / gt_n, 3) if gt_n else None,
                median_delay_s=round(float(np.median(delays)), 3) if delays else None,
                **{k: outcomes.count(k) for k in ("false_no_person", "false_no_cross", "duplicate", "unscored")})


# ---------------------------------------------------------------------------------------------------------- summaries
OUTCOMES = ("hit", "duplicate", "other_entry", "late", "censored", "false_no_entry", "false_no_person")


def approach_summary(rows: list[dict]) -> dict:
    """Pool per-zone approach records: {"targets", "leads", "outcomes"} -> hit rate, lead statistics, counts."""
    targets = sum(r["targets"] for r in rows)
    leads = [v for r in rows for v in r["leads"]]
    counts = {k: sum(r["outcomes"].get(k, 0) for r in rows) for k in OUTCOMES}
    return dict(targets=targets, warned=len(leads), hit_rate=round(len(leads) / targets, 3) if targets else None,
                useful=sum(v >= USEFUL_S for v in leads),
                median_lead_s=round(float(np.median(leads)), 2) if leads else None,
                min_lead_s=round(min(leads), 2) if leads else None, max_lead_s=round(max(leads), 2) if leads else None,
                warnings=sum(counts.values()), false=counts["false_no_entry"] + counts["false_no_person"], **counts)


def approach_record(score: dict, targets: set[int], episodes: list[dict], zone: str) -> dict:
    """One zone's approach record (targets of that zone, their leads, the zone's warning outcomes)."""
    mine = {i for i in targets if episodes[i]["zone"] == zone}
    outcomes = [w["outcome"] for w in score["warnings"] if w["zone"] == zone]
    return dict(targets=len(mine), leads=[score["lead"][i] for i in sorted(mine) if i in score["lead"]],
                outcomes={k: outcomes.count(k) for k in OUTCOMES})


def direction_record(episodes: list[dict], targets: set[int], zone: str, gt_dec: dict, model_dec: dict) -> dict:
    """Confusion counts GT decision x model decision over the zone's walk-ins, plus how many of the GT / model
    alarms are fail-safe ones (direction unknown)."""
    pairs, gt_unknown, model_unknown, entry_alarm_unknown = [], 0, 0, 0
    for i in sorted(targets):
        ep = episodes[i]
        if ep["zone"] == zone:
            gt = gt_dec[(ep["gid"], zone, ep["start"])]
            model = model_dec.get(i)
            pairs.append((gt["type"], model[0] if model else None))
            gt_unknown += heading(gt["velocity"]) == "unknown"
            model_unknown += bool(model and model[1])
            entry_alarm_unknown += bool(gt["type"] == "ENTRY" and model and model[0] == "INTRUSION" and model[1])
    out = {f"{g}->{m or 'none'}": sum(p == (g, m) for p in pairs) for g in ("INTRUSION", "ENTRY")
           for m in ("INTRUSION", "ENTRY", None)}
    return dict(out, gt_unknown=gt_unknown, model_unknown=model_unknown, entry_alarm_unknown=entry_alarm_unknown)


def direction_summary(rows: list[dict]) -> dict:
    """Pool confusion records -> counts and agreement."""
    c = {k: sum(r[k] for r in rows) for k in rows[0]}
    decided = sum(v for k, v in c.items() if "->" in k and not k.endswith("none"))
    agree = c["INTRUSION->INTRUSION"] + c["ENTRY->ENTRY"]
    return dict(c, walk_ins=sum(v for k, v in c.items() if "->" in k), decided=decided, agree=agree,
                agreement=round(agree / decided, 3) if decided else None)


# ---------------------------------------------------------------------------------------------------------- main
def main() -> None:
    thr = {m: compare.thresholds(v[2]) for m, v in compare.MODELS.items()}
    res = dict(meta=dict(created=time.strftime("%Y-%m-%d %H:%M"), k=K, match_iou=compare.MATCH_IOU,
                         fit_s=za.FIT_S, rearm_s=za.REARM_S, rearm_bh=za.REARM_BH, min_speed_bh=za.MIN_SPEED,
                         max_residual_bh=za.MAX_RESIDUAL, useful_lead_s=USEFUL_S, attr_s=compare.ATTR_S),
               zones={}, lines={}, per_zone={}, per_seq={}, tripwire={}, heading={}, alarm_volume={}, replay_check={},
               no_person_iou=[], warnings={}, scene_speed={})
    pooled = {cfg: {k: {m: [] for m in SOURCES} for k in KINDS} for cfg in ("ungated", "v2", *VARIANTS)}
    pooled_dir = {k: {m: [] for m in SOURCES} for k in KINDS}
    for seq in compare.SEQS:
        zpath = os.path.join(HERE, "zones", seq[:8] + "_v2.json")
        zones, lines = za.load_zones(zpath), za.load_lines(zpath)
        zones_v1 = za.load_zones(os.path.join(HERE, "zones", seq[:8] + ".json"))
        assert all(np.array_equal(a.polygon, b.polygon) and a.name == b.name for a, b in zip(zones, zones_v1))
        ungated = [replace(z, direction=None) for z in zones]
        with open(zpath, encoding="utf-8") as fh:
            notes = {z["name"]: z.get("direction_note", "") for z in json.load(fh)["zones"]}
        res["zones"][seq] = [dict(name=z.name, kind=z.kind, horizon_s=z.approach.horizon_s,
                                  min_speed=z.approach.min_speed, alarm_vector=z.direction.vector,
                                  min_cos=z.direction.min_cos, note=notes[z.name]) for z in zones]
        res["lines"][seq] = [dict(name=ln.name, a=ln.a, b=ln.b, alarm_normal=ln.normal, note=ln.note) for ln in lines]
        fps = float(md.seqinfo(seq)["fps"])
        frames = md.eval_frames(seq)
        gt = compare.read_gt(seq)
        res["scene_speed"][seq] = dict(scene=scene_speeds(gt, frames, fps))
        if seq.startswith("MOT17-04"):
            res["scene_speed"][seq]["far_band"] = scene_speeds(gt, frames, fps, FAR_BAND_04)
        episodes = compare.gt_episodes(seq, zones_v1, fps, gt)
        targets_all = walk_ins(episodes)
        horizon = {z.name: z.approach.horizon_s for z in zones}
        kind = {z.name: z.kind for z in zones}
        gt_v2 = replay(frames, gt_obs(gt), zones, lines, fps, None)
        gt_ungated = replay(frames, gt_obs(gt), ungated, [], fps, None)
        gt_dec = gt_decisions(gt_v2)
        targets_alarm = {i for i in targets_all
                         if gt_dec[(episodes[i]["gid"], episodes[i]["zone"], episodes[i]["start"])]["type"] == "INTRUSION"}
        crossings = gt_crossings(gt_v2, gt)
        res["heading"][seq] = {z.name: {} for z in zones}
        res.setdefault("entry_speed", {})[seq] = {z.name: [] for z in zones}
        for i in sorted(targets_all):
            ep = episodes[i]
            h = heading(gt_dec[(ep["gid"], ep["zone"], ep["start"])]["velocity"])
            res["heading"][seq][ep["zone"]][h] = res["heading"][seq][ep["zone"]].get(h, 0) + 1
            v = gt_dec[(ep["gid"], ep["zone"], ep["start"])]["velocity"]
            if v is not None:
                res["entry_speed"][seq][ep["zone"]].append(round(float(np.hypot(*v)), 3))
        res["per_zone"][seq] = {z.name: {} for z in zones}
        res["tripwire"][seq] = {}
        res["alarm_volume"][seq] = {}
        sources, runs, obs_of = {"gt": (gt_v2, gt_ungated, gt_who)}, {}, {"gt": (gt_obs(gt), None)}
        for model in compare.MODELS:
            with open(os.path.join(compare.CACHE, f"{seq[:8]}__{model}.json")) as fh:
                run = json.load(fh)
            runs[model] = run
            obs_of[model] = (track_obs(run), za.MAX_MISSING)
            tracks = {int(f): rows for f, rows in run["tracks"].items()}
            matches, _, _ = compare.track_gt_match(tracks, gt, compare.MATCH_IOU)
            ev_v2 = replay(frames, track_obs(run), zones, lines, fps, za.MAX_MISSING)
            ev_ungated = replay(frames, track_obs(run), ungated, [], fps, za.MAX_MISSING)
            base = [{k: v for k, v in asdict(e).items() if k in BASE_FIELDS} for e in ev_v2
                    if e.type not in ("APPROACH", "LINE_CROSS")]
            v1_like = [dict(e, type="INTRUSION" if e["type"] == "ENTRY" else e["type"],
                            foot=None if e["foot"] is None else list(e["foot"])) for e in base]
            res["replay_check"].setdefault(seq, {})[model] = v1_like == run["events"]
            sources[model] = (ev_v2, ev_ungated, model_who(matches))
            res["tripwire"][seq][model] = {ln.name: score_lines(ev_v2, [c for c in crossings if c["line"] == ln.name],
                                                                model_who(matches), fps) for ln in lines}
            res["alarm_volume"][seq][model] = {
                z.name: dict(v1_intrusion=sum(e["type"] == "INTRUSION" and e["zone"] == z.name for e in run["events"]),
                             v2_intrusion=sum(e.type == "INTRUSION" and e.zone == z.name for e in ev_v2),
                             v2_entry=sum(e.type == "ENTRY" and e.zone == z.name for e in ev_v2),
                             v2_approach=sum(e.type == "APPROACH" and e.zone == z.name for e in ev_v2))
                for z in zones}
        res["tripwire"][seq]["gt"] = {ln.name: dict(gt_crossings=sum(c["scored"] for c in crossings
                                                                      if c["line"] == ln.name),
                                                    gt_unscored=sum(not c["scored"] for c in crossings
                                                                    if c["line"] == ln.name)) for ln in lines}
        res["warnings"][seq] = {}
        res["per_seq"][seq] = {}
        for src, (ev_v2, ev_ungated, who) in sources.items():
            s_ungated = score_approach(ev_ungated, episodes, who, fps, horizon, targets_all, frames[-1])
            s_v2 = score_approach(ev_v2, episodes, who, fps, horizon, targets_alarm, frames[-1])
            res["warnings"][seq][src] = dict(ungated=s_ungated["warnings"], v2=s_v2["warnings"])
            if src != "gt":
                res["no_person_iou"].extend(best_iou(w, runs[src], gt) for w in s_ungated["warnings"]
                                            if w["outcome"] == "false_no_person")
            dec = model_decisions(ev_v2, episodes, who, fps)
            variant_scores = {"as configured": s_ungated}
            for name, change in VARIANTS.items():
                if change:
                    zs = [replace(z, approach=replace(z.approach, **change)) for z in ungated]
                    variant_scores[name] = score_approach(replay(frames, obs_of[src][0], zs, [], fps, obs_of[src][1]),
                                                          episodes, who, fps, horizon, targets_all, frames[-1])
            per_kind = {k: [] for k in KINDS}
            for z in zones:
                rec = dict(ungated=approach_record(s_ungated, targets_all, episodes, z.name),
                           v2=approach_record(s_v2, targets_alarm, episodes, z.name),
                           direction=direction_record(episodes, targets_all, z.name, gt_dec, dec))
                res["per_zone"][seq][z.name][src] = dict(
                    approach_ungated=approach_summary([rec["ungated"]]), approach_v2=approach_summary([rec["v2"]]),
                    direction=direction_summary([rec["direction"]]))
                pooled["ungated"][kind[z.name]][src].append(rec["ungated"])
                pooled["v2"][kind[z.name]][src].append(rec["v2"])
                pooled_dir[kind[z.name]][src].append(rec["direction"])
                per_kind[kind[z.name]].append(rec["ungated"])
                for name, score in variant_scores.items():
                    pooled[name][kind[z.name]][src].append(approach_record(score, targets_all, episodes, z.name))
            res["per_seq"][seq][src] = {k: approach_summary(v) for k, v in per_kind.items()}
        print(f"{seq}: {len(targets_all)} walk-ins ({len(targets_alarm)} labelled INTRUSION on the GT trajectory), "
              f"{sum(c['scored'] for c in crossings)} scored GT crossings; replay check {res['replay_check'][seq]}",
              flush=True)
    res["pooled"] = {cfg: {k: {m: approach_summary(v) for m, v in per.items()} for k, per in pooled[cfg].items()}
                     for cfg in pooled}
    res["pooled_direction"] = {k: {m: direction_summary(v) for m, v in per.items()} for k, per in pooled_dir.items()}
    with open(os.path.join(HERE, "results_v2.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    write_md(res, os.path.join(HERE, "results_v2.md"))
    print(open(os.path.join(HERE, "results_v2.md"), encoding="utf-8").read())


# ---------------------------------------------------------------------------------------------------------- report
def _approach_table(res: dict, cfg: str) -> list[str]:
    header = ["zone", "source", "target walk-ins", "warned before entry", f"of these lead >= {USEFUL_S:g} s",
              "median lead [s]", "lead range [s]", "warnings", "false (no entry / no person)", "late", "duplicate",
              "other entry", "censored (clip end)"]
    rows = []
    for kind in KINDS:
        for src in SOURCES:
            s = res["pooled"][cfg][kind][src]
            rows.append([kind, LABEL[src], s["targets"], s["warned"], s["useful"], fmt(s["median_lead_s"]),
                         f"{fmt(s['min_lead_s'])} - {fmt(s['max_lead_s'])}", s["warnings"],
                         f"{s['false']} ({s['false_no_entry']} / {s['false_no_person']})", s["late"],
                         s["duplicate"], s["other_entry"], s["censored"]])
    return table(header, rows)


def _per_seq_table(res: dict) -> list[str]:
    """Ungated APPROACH per sequence and zone kind."""
    header = ["sequence", "zone", "source", "walk-ins", "warned before entry", f"lead >= {USEFUL_S:g} s",
              "median lead [s]", "false (no entry / no person)", "censored"]
    rows = []
    for seq, per in res["per_seq"].items():
        for kind in KINDS:
            for src in SOURCES:
                s = per[src][kind]
                rows.append([seq[:8], kind, LABEL[src], s["targets"], s["warned"], s["useful"],
                             fmt(s["median_lead_s"]), f"{s['false']} ({s['false_no_entry']} / {s['false_no_person']})",
                             s["censored"]])
    return table(header, rows)


def _variant_table(res: dict) -> list[str]:
    """Ungated APPROACH under the post-hoc rule variants: warned / useful / false per kind and source."""
    header = ["rule (ungated, both sequences)"] + [f"{kind}: {LABEL[src]}" for kind in KINDS for src in SOURCES]
    rows = []
    for name in VARIANTS:
        cells = []
        for kind in KINDS:
            for src in SOURCES:
                s = res["pooled"][name][kind][src]
                cells.append(f"{s['warned']} / {s['useful']} / {s['false']}")
        rows.append([name] + cells)
    return table(header, rows)


def _direction_table(res: dict) -> list[str]:
    header = ["zone", "source", "walk-ins", "with a model decision", "agreement", "GT alarm -> alarm",
              "GT alarm -> ENTRY", "GT ENTRY -> alarm", "GT ENTRY -> ENTRY", "not alerted (GT alarm / GT ENTRY)",
              "direction unknown -> fail-safe alarm (GT / model)"]
    rows = []
    for kind in KINDS:
        for src in ("plain", "s3x"):
            s = res["pooled_direction"][kind][src]
            rows.append([kind, LABEL[src], s["walk_ins"], s["decided"], f"{s['agree']} / {s['decided']} "
                         f"({fmt(s['agreement'], 2)})", s["INTRUSION->INTRUSION"], s["INTRUSION->ENTRY"],
                         f"{s['ENTRY->INTRUSION']} ({s['entry_alarm_unknown']} fail-safe)", s["ENTRY->ENTRY"], f"{s['INTRUSION->none']} / {s['ENTRY->none']}",
                         f"{s['gt_unknown']} / {s['model_unknown']}"])
    return table(header, rows)


def _heading_table(res: dict) -> list[str]:
    header = ["sequence", "zone (kind)", "alarm direction", "walk-ins", "GT heading at entry (up / down / left / right "
              "/ unknown)", "median GT entry speed [BH/s]", "GT decision INTRUSION / ENTRY"]
    rows = []
    for seq, zones in res["zones"].items():
        for z in zones:
            h = res["heading"][seq][z["name"]]
            d = res["per_zone"][seq][z["name"]]["gt"]["direction"]
            vec = f"({z['alarm_vector'][0]:.2f}, {z['alarm_vector'][1]:.2f}), cos >= {z['min_cos']:g}"
            rows.append([seq[:8], f"{z['name']} ({z['kind']})", vec, sum(h.values()),
                         " / ".join(str(h.get(k, 0)) for k in ("up", "down", "left", "right", "unknown")),
                         fmt(float(np.median(res["entry_speed"][seq][z["name"]])) if res["entry_speed"][seq][z["name"]]
                             else None),
                         f"{d['INTRUSION->INTRUSION']} / {d['ENTRY->ENTRY']}"])
    return table(header, rows)


def _per_zone_table(res: dict) -> list[str]:
    header = ["sequence", "zone", "source", "APPROACH (filter off): warned / walk-ins, median lead [s], false",
              "APPROACH (v2): warned / alarm walk-ins, false", "direction agreement", "INTRUSION alerts v1 -> v2 "
              "(+ ENTRY)"]
    rows = []
    for seq, zones in res["per_zone"].items():
        for zone, per in zones.items():
            for src in ("gt", "plain", "s3x"):
                a, b, d = per[src]["approach_ungated"], per[src]["approach_v2"], per[src]["direction"]
                vol = res["alarm_volume"][seq].get(src, {}).get(zone)
                volume = f"{vol['v1_intrusion']} -> {vol['v2_intrusion']} (+ {vol['v2_entry']})" if vol else "-"
                agreement = "-" if src == "gt" else f"{d['agree']} / {d['decided']}"
                rows.append([seq[:8], zone, LABEL[src], f"{a['warned']} / {a['targets']}, {fmt(a['median_lead_s'])}, "
                             f"{a['false']}", f"{b['warned']} / {b['targets']}, {b['false']}", agreement, volume])
    return table(header, rows)


def _tripwire_table(res: dict) -> list[str]:
    header = ["sequence", "line", "model", "scored GT crossings (+ unscored)", "detected", "recall", "median delay [s]",
              "false (no crossing / no person)", "duplicates", "on unscored crossings"]
    rows = []
    for seq, per in res["tripwire"].items():
        for model in compare.MODELS:
            for line, s in per[model].items():
                rows.append([seq[:8], line, LABEL[model], f"{s['gt_crossings']} (+ {s['gt_unscored']})", s["hit"],
                             fmt(s["recall"], 2), fmt(s["median_delay_s"]),
                             f"{s['false_no_cross'] + s['false_no_person']} ({s['false_no_cross']} / "
                             f"{s['false_no_person']})", s["duplicate"], s["unscored"]])
    return table(header, rows)


def _data_notes(res: dict) -> list[str]:
    """Notes whose numbers come from the results (headline, speeds, false warnings, direction filter)."""
    walk = sum(res["pooled"]["ungated"][k]["gt"]["targets"] for k in KINDS)
    alarm = sum(res["pooled"]["v2"][k]["gt"]["targets"] for k in KINDS)
    unknown = sum(res["pooled_direction"][k]["gt"]["gt_unknown"] for k in KINDS)
    far04 = res["heading"]["MOT17-04-FRCNN"]["far_walkway"]
    speed04 = res["entry_speed"]["MOT17-04-FRCNN"]["far_walkway"]
    far09 = res["heading"]["MOT17-09-FRCNN"]["far_storefront"]
    o04, o09 = (res["per_seq"][s]["gt"]["far"] for s in ("MOT17-04-FRCNN", "MOT17-09-FRCNN"))
    oracle, plain, s3x = (res["pooled"]["ungated"]["far"][m] for m in SOURCES)
    low = {m: res["pooled"]["min_speed 0.10"]["far"][m] for m in SOURCES}
    off = {m: res["pooled"]["residual check off"]["far"][m] for m in SOURCES}
    band = res["scene_speed"]["MOT17-04-FRCNN"]["far_band"]
    hv = band["height_vs_row"]
    s09 = res["scene_speed"]["MOT17-09-FRCNN"]["scene"]["all"]
    ious = res["no_person_iou"]
    loose = sum(0.2 <= p < 0.5 for p, _ in ious)
    taken = sum(p >= 0.5 for p, _ in ious)
    seated = sum(o >= 0.3 for _, o in ious)
    dis = [res["pooled_direction"][k][m] for k in KINDS for m in ("plain", "s3x")]
    dis_n, dis_safe = sum(d["ENTRY->INTRUSION"] for d in dis), sum(d["entry_alarm_unknown"] for d in dis)

    def q(d: dict) -> str:
        return f"{d['median']:.2f} BH/s (IQR {d['p25']:.2f}-{d['p75']:.2f}, n {d['n']})"

    return [
        f"- APPROACH is limited by its speed threshold on clean paths and by the tracks on the models. With GT paths "
        f"the rule warns {o04['warned']} of {o04['targets']} walk-ins in the MOT17-04 far zone (the small-person zone) "
        f"and {o09['warned']} of {o09['targets']} in the MOT17-09 one; pooled {oracle['warned']} of "
        f"{oracle['targets']}, of which {oracle['useful']} at least {USEFUL_S:g} s ahead. The models warn "
        f"{plain['warned']} (plain) and {s3x['warned']} (S3x) far-zone walk-ins ({plain['useful']} and {s3x['useful']} "
        f"at least {USEFUL_S:g} s ahead). Lowering min_speed to 0.10 BH/s (post hoc, on this data) lifts the oracle to "
        f"{low['gt']['warned']} but the models only to {low['plain']['warned']} / {low['s3x']['warned']}, with more "
        f"false warnings (oracle {low['gt']['false']}, models {low['plain']['false']} / {low['s3x']['false']}); any "
        "new threshold would have to be chosen on held-out data.",
        f"- Walking speed is scene-dependent. In the MOT17-04 far band (foot rows {band['rows'][0]}-{band['rows'][1]}) "
        f"GT movers measure {q(band['sideways'])} when moving sideways, which foreshortening does not affect, and "
        f"{q(band['along'])} along the viewing direction. Box height changes little with depth (fit: "
        f"{hv['slope'] * 200 + hv['intercept']:.0f} px at row 200, {hv['slope'] * 700 + hv['intercept']:.0f} px at row "
        "700): the camera looks steeply down, so foreshortening is mild and these people simply stroll. The 0.25 "
        "threshold is about the median moving person here, not a standing threshold (MOT17-09: movers "
        f"{q(s09)}; brisk walking across the image would be about 0.8 BH/s). The median GT entry speed in that zone "
        f"is {np.median(speed04):.2f} BH/s, so {far04.get('unknown', 0)} of {sum(far04.values())} walk-ins have no "
        "direction at the entry and fall back to INTRUSION.",
        f"- Few true approaches in an alarm direction: {walk} scored walk-ins in the 4 zones; the rule on the GT "
        f"trajectories labels {alarm} of them INTRUSION, {unknown} of those only by the fail-safe (direction unknown): "
        f"{alarm - unknown} walk-ins entered in a configured alarm direction at a measurable speed. In the MOT17-09 "
        "far zone every walk-in with a measured direction moves right, along the pavement "
        f"({far09.get('right', 0)} right, {far09.get('unknown', 0)} unknown). So the direction-gated APPROACH table has "
        "very few targets; the ungated table is the test of the APPROACH rule.",
        f"- False warnings on no person: {len(ious)} of the models' warnings (ungated, both models, all zones) are on "
        f"tracks matched to no GT pedestrian at IoU >= 0.5. {loose} overlap a pedestrian loosely (IoU 0.2-0.5), "
        f"{taken} a pedestrian already matched to another track (IoU >= 0.5), and {seated} a MOT17 'static person' "
        "box (IoU >= 0.3): S3x track 94 in MOT17-04 switches between a person sitting by a planter and the person "
        "standing behind, so its foot point jumps. The fit-residual check (cut "
        f"{res['meta']['max_residual_bh']:g} body heights, set on MOT17-02) removes "
        f"{off['plain']['false'] - plain['false']} (plain) and {off['s3x']['false'] - s3x['false']} (S3x) far-zone "
        "false warnings and no hit; it does not stop track 94, whose switched box stays for most of the fit window.",
        f"- The direction filter never turned a GT alarm into an ENTRY ('GT alarm -> ENTRY' is 0 for both models). Its "
        f"disagreements are model alarms where the GT trajectory gives ENTRY: {dis_n} (both models), {dis_safe} of "
        "them fail-safe alarms of tracks too young or too slow for a direction at the entry.",
    ]


def write_md(res: dict, path: str) -> None:
    """results_v2.md from the results dict."""
    meta = res["meta"]
    check = all(v for per in res["replay_check"].values() for v in per.values())
    lines = [
        "# Zone alerts v2 on MOT17: APPROACH warnings, direction filter, tripwires", "",
        f"Generated by `apps/zone_alert/compare_v2.py` on {meta['created']} from the cached tracks of `compare.py` "
        "(plain YOLOv8n and YOLOv8n + S3x, ByteTrack, seed 0, cross-fitted threshold 0.8) and the zone files "
        "`zones/MOT17-04_v2.json`, `zones/MOT17-09_v2.json` (same polygons as v1; directions and tripwires chosen from "
        "scene geometry before any output of the new logic was seen). Definitions: docstring of `compare_v2.py`. "
        f"Velocity: least squares over {meta['fit_s']:g} s; speed in body heights / s (min {meta['min_speed_bh']:g}); "
        f"fit residual <= {meta['max_residual_bh']:g} body heights (set on MOT17-02, `calibration_v2.json`); re-armed "
        f"after {meta['rearm_s']:g} s and {meta['rearm_bh']:g} body heights away from the zone; K = {meta['k']}; "
        f"association IoU >= {meta['match_iou']}.", "",
        f"Replay check: removing APPROACH / LINE_CROSS and renaming ENTRY to INTRUSION gives the stored v1 events of all "
        f"4 runs: **{'yes' if check else 'NO'}**.", "",
        "## APPROACH warnings, direction filter off (all walk-ins)", "",
        "Target = a scored walk-in episode (person seen outside, then entering). Hit = a warning for that person "
        "before the GT foot point entered and at most horizon + 1 s before it; lead = GT entry - warning. False = the "
        "warned person does not enter within horizon + 1 s, or the track is on no GT person. 'GT oracle' = the same "
        "rule on the GT trajectories.", "",
        *_approach_table(res, "ungated"), "",
        "Per sequence (direction filter off):", "",
        *_per_seq_table(res), "",
        "Post-hoc rule variants (direction filter off; warned / lead >= 0.5 s / false; chosen on this evaluation "
        "data, so they show sensitivity only and set nothing):", "",
        *_variant_table(res), "",
        "## APPROACH warnings as configured in v2 (only in the alarm direction)", "",
        "Target = walk-ins whose GT entry is labelled INTRUSION by the same rule on the GT trajectory.", "",
        *_approach_table(res, "v2"), "",
        "## Direction filter: model decision vs the decision on the GT trajectory", "",
        *_direction_table(res), "",
        "GT entry headings of the walk-ins (image plane, nearest axis of the fitted velocity at the entry):", "",
        *_heading_table(res), "",
        "## Tripwires", "",
        *_tripwire_table(res), "",
        "## Per sequence and zone", "",
        *_per_zone_table(res), "",
        "## Notes (read before quoting the numbers)", "",
        "- Two sequences (MOT17-04 and MOT17-09 evaluation halves, 17.5 s and 8.7 s), one seed (0), one threshold per "
        "model; no confidence intervals. The walk-in counts per zone are small, so single-digit differences are not "
        "evidence.",
        *_data_notes(res),
        "- Speed and direction are measured in the image plane, normalised by the box height (body heights / s); there "
        "is no ground calibration. Perspective bends straight ground paths and compresses far motion, so the "
        "predicted entry time is approximate, more so in the far zones.",
        "- The GT decisions use the same rule on annotated trajectories, which include interpolated foot points of "
        "hidden people; 'agreement' measures whether the model's tracks reproduce the rule's decision, not whether "
        "the rule matches human intent.",
        "- Directions and tripwires were fixed from the first evaluation frame only, before any output of the new "
        "logic was seen, and were not changed afterwards. Rule changes after the first evaluation (from a review): "
        "fit-residual check (cut from held-out MOT17-02), re-arming only after moving away from the zone, fitted foot "
        "point must be outside the zone, tripwire starting side confirmed over K frames and K frames beyond a "
        "segment end forget the side. That end rule cost one true crossing per model on MOT17-09 (feet passing 3-8 px "
        "beyond the line's top end), so the ends were extended by 0.25 box heights; this margin was chosen after "
        "seeing that miss, and it also changes which GT crossings count (MOT17-09: 4 scored instead of 3).",
        "",
        "Licence: MOT17 frames and annotations CC BY-NC-SA 3.0 (motchallenge.net); results shared under the same "
        "licence.",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
