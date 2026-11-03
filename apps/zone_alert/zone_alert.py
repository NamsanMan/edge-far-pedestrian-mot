# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Zone intrusion and loitering alerts on top of an edge pedestrian tracker.

Pipeline per frame
    detect : YOLOv8n (plain or with the S3x anisotropic P3 level) on a 416x256 letterbox canvas, FP16 autocast,
             conf >= 0.01, class-wise NMS 0.65 (exactly the export of the tracking benchmark), pedestrians kept.
    track  : ByteTrack (BoxMOT) with the benchmark settings: track_thresh = --thr, match 0.8, buffer 15 frames,
             detections below 0.2 x thr dropped before the BYTE second stage.
    alerts : each track's foot point (bottom-centre of its box) is tested against named polygons. Zone membership has
             a hysteresis of K consecutive observed frames: a track ENTERS after K frames inside and EXITS after
             K frames outside. INTRUSION is emitted on entry, LOITERING once per stay when the dwell (counted from the
             first of the K inside frames) reaches the zone's dwell_s. A track that the tracker stops reporting keeps
             its state; after MAX_MISSING frames without a report (= the tracker's own buffer) the stay is closed.
             Optional ignore masks (top-level "ignore" list of the zone file, set at installation): a foot point
             inside a mask counts as outside every zone, so a track standing in one raises no zone events; a track
             in a mask for K frames is drawn faint grey as 'masked'.
    motion : optional, per zone / per file (all off by default; a zone file without these keys gives exactly the events
             above). The foot-point velocity is a least-squares line over the track's last FIT_S seconds, divided by
             the median box height: speed in body heights per second (BH/s), a calibration-free proxy that makes far
             and near people comparable. Brisk walking across the image is about 0.8 BH/s (1.4 m/s, 1.7 m tall);
             the strolling people of MOT17-04 move about 0.25 BH/s, so MIN_SPEED is scene-dependent (README).
             APPROACH (zone key "approach"): the straight extrapolated foot path enters the polygon within horizon_s,
             the speed is >= min_speed, the line fits the recent foot points (RMS residual <= max_residual body
             heights; a box that jumps or switches between people does not) and all of it holds for k observed
             frames in a row -> one warning per track and zone; re-armed after REARM_S of not heading in AND after
             moving REARM_BH body heights farther from the zone than the closest point since the warning (so standing
             or milling at the edge is not re-warned), or after the stay's EXIT.
             Direction filter (zone key "alarm_direction" or "alarm_edge", with "min_cos"): an entry raises INTRUSION
             only when cos(velocity at the entry, alarm direction) >= min_cos; otherwise the stay is logged as a
             non-alarm ENTRY (it can still raise LOITERING). Unknown direction (track too short, standing) -> INTRUSION
             (fail-safe). With a direction filter, APPROACH is only raised for motion in the alarm direction.
             Tripwires (top-level "lines"): LINE_CROSS when the foot point changes side of the segment in the alarm
             direction, with the same K-frame hysteresis; the starting side also needs K frames, and K frames beyond
             the segment's ends (plus END_MARGIN_BH body heights) forget the side (walking around an end is no
             crossing); the other way raises nothing.

Usage (MOT17-04, evaluation half = frames 527..1050), from the repository root (<mot17_train> = data.mot17_train of
configs/paths.yaml):
    python -B apps/zone_alert/zone_alert.py --frames <mot17_train>/MOT17-04-FRCNN/img1 \
        --start 527 --stop 1050 --family yolov8n_s3x --weights release_assets/yolov8n_s3x_mot17_416x256_s0.pt \
        --zones apps/zone_alert/zones/MOT17-04.json --thr 0.8 --out apps/zone_alert/out/MOT17-04_s3x.mp4

Outputs: an annotated MP4 (zones, tracks, alert banner, event list) and <out>.events.jsonl (one JSON event per line).
Code: AGPL-3.0 (it links Ultralytics). MOT17 media: CC BY-NC-SA 3.0, attribution drawn on every frame.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Iterator, Sequence

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root
sys.path.insert(0, ROOT)                     # efpmot: model builder, letterbox, tracker factory, MOT17 helpers

K_FRAMES = 5          # hysteresis: 5 consecutive observed frames (167 ms at 30 fps) to enter / to exit a zone
MAX_MISSING = 15      # frames a silent track keeps its zone state (ByteTrack track_buffer of the benchmark)
BANNER_S = 1.5        # an alert stays in the top banner for this many seconds
PED = 0               # 'pedestrian' in efpmot.data.BDD_CLASSES, the 8-class head order
ATTRIBUTION = "MOT17 (MOTChallenge, motchallenge.net) - CC BY-NC-SA 3.0 - overlays added"
FIT_S = 0.8           # velocity: least-squares line over the foot points of the last 0.8 s (24 frames at 30 fps)
REARM_S = 1.0         # APPROACH re-arms after 1 s of observed frames without the approach condition
MIN_SPEED = 0.25      # body heights / s: slower motion has no defined direction (box jitter); scene-dependent
MAX_RESIDUAL = 0.075  # RMS line-fit residual / box height above which APPROACH is not raised (box jumps, id switches):
                      # 99th percentile of correctly tracked movers on held-out data (MOT17-02), calibrate_v2.py
REARM_BH = 0.5        # APPROACH re-arms only after the foot moved this many body heights away from the zone
END_MARGIN_BH = 0.25  # tripwire ends are extended by a quarter of the person's box height (foot jitter at an end)
HOLD_S = 0.3          # drawing: an APPROACH warning is drawn while its condition holds and up to 0.3 s after


# --------------------------------------------------------------------------------------------------------- zones
@dataclass
class ApproachRule:
    """Zone key "approach": true (defaults) or {"horizon_s", "min_speed", "k"}."""
    horizon_s: float = 3.0    # warn when the extrapolated foot path enters the zone within this many seconds
    min_speed: float = MIN_SPEED   # body heights per second
    k: int = K_FRAMES         # consecutive observed frames the condition must hold
    max_residual: float | None = MAX_RESIDUAL   # RMS line-fit residual in body heights (None: no check)


@dataclass
class DirectionRule:
    """Zone key "alarm_direction": [dx, dy] (image axes, y down) or "alarm_edge": i (cross polygon edge i -> i+1
    inward), with "min_cos" (default 0.5 = within 60 degrees)."""
    vector: tuple[float, float]   # unit vector, image coordinates
    min_cos: float = 0.5

    def matches(self, motion: Motion | None) -> bool | None:
        """True / False for motion in / not in the alarm direction; None when the direction is unknown (no motion
        estimate or slower than MIN_SPEED)."""
        if motion is None or motion.speed is None or motion.speed < MIN_SPEED:
            return None
        vx, vy = motion.velocity
        norm = (vx * vx + vy * vy) ** 0.5
        return (vx * self.vector[0] + vy * self.vector[1]) / norm >= self.min_cos


@dataclass
class Zone:
    name: str
    polygon: np.ndarray       # [P, 2] float32, source-image pixels
    dwell_s: float            # continuous stay that raises LOITERING
    kind: str = ""            # free tag, e.g. 'far' / 'near'
    note: str = ""
    approach: ApproachRule | None = None      # None: no APPROACH warnings
    direction: DirectionRule | None = None    # None: every entry is an INTRUSION

    def contains(self, xy: tuple[float, float]) -> bool:
        """True when the point is inside the polygon or on its edge."""
        return cv2.pointPolygonTest(self.polygon, (float(xy[0]), float(xy[1])), False) >= 0


def inward_normal(polygon: np.ndarray, i: int) -> tuple[float, float]:
    """Unit normal of edge i (vertex i -> i+1) pointing into the polygon (any simple polygon, either orientation)."""
    pts = polygon.astype(np.float64)
    area2 = float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]))
    dx, dy = pts[(i + 1) % len(pts)] - pts[i]
    nx, ny = (-dy, dx) if area2 > 0 else (dy, -dx)         # interior lies left of every edge when area2 > 0
    norm = float(np.hypot(nx, ny))
    return float(nx / norm), float(ny / norm)


def _unit(v: Sequence[float]) -> tuple[float, float]:
    norm = float(np.hypot(v[0], v[1]))
    if norm == 0:
        raise ValueError("direction vector must not be zero")
    return float(v[0]) / norm, float(v[1]) / norm


def _zone_from_cfg(z: dict) -> Zone:
    polygon = np.asarray(z["polygon"], np.float32).reshape(-1, 2)
    approach = z.get("approach")
    if approach is True:
        approach = ApproachRule()
    elif isinstance(approach, dict):
        approach = ApproachRule(**approach)
    else:
        approach = None
    direction = None
    if "alarm_direction" in z:
        direction = DirectionRule(_unit(z["alarm_direction"]), float(z.get("min_cos", 0.5)))
    elif "alarm_edge" in z:
        direction = DirectionRule(inward_normal(polygon, int(z["alarm_edge"])), float(z.get("min_cos", 0.5)))
    return Zone(name=z["name"], polygon=polygon, dwell_s=float(z["dwell_s"]), kind=z.get("kind", ""),
                note=z.get("note", ""), approach=approach, direction=direction)


def load_zones(path: str) -> list[Zone]:
    """Read a zone file: {"zones": [{"name", "polygon": [[x, y], ...], "dwell_s", "kind"?, "note"?,
    "approach"?, "alarm_direction"? | "alarm_edge"?, "min_cos"?}, ...]} (see ApproachRule, DirectionRule)."""
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    return [_zone_from_cfg(z) for z in cfg["zones"]]


@dataclass
class Tripwire:
    """Virtual line segment a -> b; crossing it toward the alarm side raises LINE_CROSS."""
    name: str
    a: tuple[float, float]
    b: tuple[float, float]
    normal: tuple[float, float]   # unit normal pointing to the alarm side (crossing direction)
    note: str = ""

    def side(self, xy: tuple[float, float], margin: float = 0.0) -> int | None:
        """+1 alarm side, -1 other side, 0 on the line; None beyond the ends of the segment extended by `margin` px."""
        ex, ey = self.b[0] - self.a[0], self.b[1] - self.a[1]
        wx, wy = xy[0] - self.a[0], xy[1] - self.a[1]
        length2 = ex * ex + ey * ey
        u, slack = (wx * ex + wy * ey) / length2, margin / math.sqrt(length2)
        if not -slack <= u <= 1.0 + slack:
            return None
        d = wx * self.normal[0] + wy * self.normal[1]
        return 1 if d > 0 else -1 if d < 0 else 0


def load_lines(path: str) -> list[Tripwire]:
    """Read the optional top-level "lines" list: [{"name", "points": [[x1, y1], [x2, y2]], "alarm_direction": [dx, dy],
    "note"}, ...]. The alarm direction must not be parallel to the line; every line needs a non-empty note."""
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    lines = []
    for i, ln in enumerate(cfg.get("lines", [])):
        name = ln.get("name", f"line_{i}")
        if not str(ln.get("note", "")).strip():
            raise ValueError(f"{path}: line '{name}' has no 'note'")
        (x1, y1), (x2, y2) = ln["points"]
        nx, ny = _unit((-(y2 - y1), x2 - x1))
        dx, dy = _unit(ln["alarm_direction"])
        dot = nx * dx + ny * dy
        if abs(dot) < 0.1:
            raise ValueError(f"{path}: line '{name}': alarm_direction is (nearly) parallel to the line")
        if dot < 0:
            nx, ny = -nx, -ny
        lines.append(Tripwire(name, (float(x1), float(y1)), (float(x2), float(y2)), (nx, ny), ln["note"]))
    return lines


@dataclass
class IgnoreRegion:
    """Installation-time ignore mask: a track whose foot point stays in it raises no zone events."""
    name: str
    polygon: np.ndarray       # [P, 2] float32, source-image pixels
    note: str = ""            # why the mask exists (who added it, when, after which false alarm)

    def contains(self, xy: tuple[float, float]) -> bool:
        """True when the point is inside the polygon or on its edge."""
        return cv2.pointPolygonTest(self.polygon, (float(xy[0]), float(xy[1])), False) >= 0


def load_ignore(path: str) -> list[IgnoreRegion]:
    """Read the optional top-level "ignore" list of a zone file: [{"polygon": [[x, y], ...], "note", "name"?}, ...].
    A file without "ignore" gives [] (the monitor then behaves exactly as without masks). Every mask must carry a
    non-empty "note" (why it exists); a mask without one is refused with ValueError."""
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    regions = []
    for i, m in enumerate(cfg.get("ignore", [])):
        name = m.get("name", f"ignore_{i}")
        if not str(m.get("note", "")).strip():
            raise ValueError(f"{path}: ignore mask '{name}' has no 'note' (say who added it, when and why)")
        regions.append(IgnoreRegion(name=name, polygon=np.asarray(m["polygon"], np.float32).reshape(-1, 2),
                                    note=m["note"]))
    return regions


def foot_point(box: np.ndarray) -> tuple[float, float]:
    """Bottom-centre of an (x1, y1, x2, y2, ...) box: where the person stands."""
    return float((box[0] + box[2]) / 2), float(box[3])


# --------------------------------------------------------------------------------------------------------- motion
@dataclass
class Motion:
    foot: tuple[float, float]       # fitted foot point at the current frame (px)
    velocity: tuple[float, float]   # px / s, image axes
    height: float | None            # median box height over the fit window (px)
    residual: float = 0.0           # RMS distance of the foot points from the fitted line (px)

    @property
    def speed(self) -> float | None:
        """Body heights per second (None without box heights)."""
        if not self.height:
            return None
        return math.hypot(*self.velocity) / self.height

    @property
    def residual_bh(self) -> float | None:
        """RMS fit residual in body heights (None without box heights)."""
        if not self.height:
            return None
        return self.residual / self.height

    def velocity_bh(self) -> tuple[float, float] | None:
        """Velocity in body heights per second (image x, y), rounded for the event log."""
        if not self.height:
            return None
        return round(self.velocity[0] / self.height, 3), round(self.velocity[1] / self.height, 3)


def fit_motion(obs: Sequence[tuple[int, float, float, float | None]], frame: int, fps: float,
               min_obs: int) -> Motion | None:
    """Least-squares line through (frame, x, y) observations -> foot point at `frame`, velocity and RMS residual;
    None with fewer than `min_obs` observations. A line over ~0.8 s averages per-frame box jitter: with jitter sigma,
    the slope error is sigma / sqrt(sum (t - mean t)^2), e.g. 3 px jitter over 24 frames at 30 fps -> 2.65 px/s.
    The residual (from the same sums) is about sigma for jitter, but large when the box jumps or switches people."""
    n = len(obs)
    if n < min_obs:
        return None
    st = stt = sx = sy = stx = sty = sxx = syy = 0.0     # one pass over the window (runs per track and frame)
    for f, x, y, _ in obs:
        t = (f - frame) / fps
        st, stt, sx, sy, stx, sty = st + t, stt + t * t, sx + x, sy + y, stx + t * x, sty + t * y
        sxx, syy = sxx + x * x, syy + y * y
    tm, xm, ym = st / n, sx / n, sy / n
    stt_c = stt - st * tm                                # centred sums
    if stt_c <= 1e-12:
        return None
    stx_c, sty_c = stx - tm * sx, sty - tm * sy
    vx, vy = stx_c / stt_c, sty_c / stt_c
    rss = (sxx - sx * xm - vx * stx_c) + (syy - sy * ym - vy * sty_c)   # sum of squared residuals, x and y
    heights = sorted(o[3] for o in obs if o[3])
    m = len(heights)
    height = (heights[m // 2] if m % 2 else (heights[m // 2 - 1] + heights[m // 2]) / 2) if m else None
    return Motion((xm - vx * tm, ym - vy * tm), (vx, vy), height, math.sqrt(max(rss, 0.0) / n))


def time_to_enter(foot: tuple[float, float], velocity: tuple[float, float], polygon: np.ndarray,
                  horizon_s: float) -> float | None:
    """Seconds tau in (0, horizon_s] at which the straight path foot + velocity * tau first touches a polygon edge;
    None when it does not within the horizon."""
    px, py = foot
    vx, vy = velocity
    pts = polygon.tolist()
    best = None
    for i, (ax, ay) in enumerate(pts):
        bx, by = pts[(i + 1) % len(pts)]
        ex, ey = bx - ax, by - ay
        den = vx * ey - vy * ex                       # cross(v, e); 0 = path parallel to the edge
        if abs(den) < 1e-12:
            continue
        wx, wy = ax - px, ay - py
        tau = (wx * ey - wy * ex) / den               # foot + v tau = a + e u  (Cramer's rule)
        u = (wx * vy - wy * vx) / den
        if 0.0 < tau <= horizon_s and 0.0 <= u <= 1.0 and (best is None or tau < best):
            best = tau
    return best


# --------------------------------------------------------------------------------------------------------- alerts
ALARM_TYPES = ("INTRUSION", "LOITERING", "LINE_CROSS")   # red banner; APPROACH = warning, ENTRY / EXIT = log only


@dataclass
class Event:
    type: str                 # INTRUSION | LOITERING | EXIT | ENTRY | APPROACH | LINE_CROSS
    zone: str                 # zone name (tripwire name for LINE_CROSS)
    track_id: int
    frame: int                # frame number at which the event is raised
    t: float                  # seconds from the first processed frame
    entered: int              # first frame of the run that led to the event (inside run / approach run / new side)
    foot: tuple[float, float] | None = None   # None when a lost track is closed


@dataclass
class MotionEvent(Event):
    """Event of the motion features (ENTRY / INTRUSION of a zone with a direction filter, APPROACH, LINE_CROSS)."""
    velocity: tuple[float, float] | None = None   # body heights / s, image x, y (None: too few observations)
    eta_s: float | None = None                     # APPROACH: predicted seconds until the foot enters the zone


@dataclass
class _Stay:
    inside: bool = False
    run: int = 0              # consecutive observations contradicting `inside`
    first_in: int = -1        # first frame of the current inside run
    loitered: bool = False
    alarm: bool = True        # False for a non-alarm ENTRY (direction filter)


@dataclass
class _Approach:
    armed: bool = True
    run: int = 0              # consecutive observations meeting the approach condition
    away: int = 0             # consecutive observations not meeting it while disarmed
    raised: int = -1          # frame of the last warning
    closest: float = 0.0      # smallest foot distance to the zone (px) since the last warning
    seen: int = -1            # last frame the condition held
    view: tuple | None = None  # (eta_s, fitted foot, predicted entry point) at frame `seen`


@dataclass
class _Side:
    side: int = 0             # confirmed side of a tripwire (0 = none yet, or forgotten beyond the segment's ends)
    cand: int = 0             # side of the current run of observations that differ from `side`
    run: int = 0
    first: int = -1           # first frame of that run
    beyond: int = 0           # consecutive observations beyond the segment's ends


@dataclass
class _Mask:
    masked: bool = False
    run: int = 0              # consecutive observations contradicting `masked`


@dataclass
class ZoneMonitor:
    """Per (track, zone) state machine with a K-frame hysteresis; update() returns the events of one frame.

    Dwell is wall-clock time since the first of the K inside frames, so LOITERING can fire on a frame where the foot
    is already outside again (during the K-frame exit delay) or on the first report after a gap of missing frames.

    Ignore masks (optional): a foot point inside any ignore polygon counts as OUTSIDE every zone, frame by frame, before
    the zone hysteresis. So a track standing on a mask never completes K inside frames (no INTRUSION / LOITERING), an
    open stay that settles on a mask is closed by the normal K-frame EXIT, and a box jittering across the mask edge
    cannot build up an entry either (its in-zone frames are never K in a row). A track leaving the mask into a zone
    needs K inside frames, as any other track. The per-track mask state (K frames in a mask -> 'masked', K frames out ->
    not) only drives the faint 'masked' drawing. With no ignore polygons the monitor is exactly the unmasked one.

    Motion features (module docstring) run only when a zone has "approach" / a direction filter or `lines` is not
    empty; they never change when a stay starts or ends, only how an entry is labelled (INTRUSION / ENTRY) and which
    extra events (APPROACH, LINE_CROSS) are added. Tripwires ignore the masks.
    """
    zones: list[Zone]
    fps: float
    k: int = K_FRAMES
    max_missing: int | None = MAX_MISSING     # None: never close a silent track (used for ground truth)
    frame0: int = 0
    state: dict = field(default_factory=dict)      # (track_id, zone name) -> _Stay
    last_seen: dict = field(default_factory=dict)  # track_id -> frame
    ignore: list[IgnoreRegion] = field(default_factory=list)
    mask_state: dict = field(default_factory=dict)  # track_id -> _Mask (only filled when `ignore` is not empty)
    lines: list[Tripwire] = field(default_factory=list)
    fit_s: float = FIT_S
    rearm_s: float = REARM_S
    history: dict = field(default_factory=dict)     # track_id -> deque of (frame, x, y, box height)
    approach_state: dict = field(default_factory=dict)  # (track_id, zone name) -> _Approach
    line_state: dict = field(default_factory=dict)  # (track_id, line name) -> _Side

    def __post_init__(self) -> None:
        self.needs_heights = any(z.approach or z.direction for z in self.zones)
        self.motion_on = bool(self.lines) or self.needs_heights
        self.fit_frames = max(3, int(round(self.fit_s * self.fps)))
        self.last_frame = -1

    def _motion(self, tid: int, frame: int, xy: tuple[float, float], height: float | None) -> Motion | None:
        """Remember this observation and fit the track's motion over the last fit_s seconds."""
        hist = self.history.setdefault(tid, deque(maxlen=self.fit_frames))
        hist.append((frame, float(xy[0]), float(xy[1]), height))
        recent = [o for o in hist if o[0] > frame - self.fit_frames]
        return fit_motion(recent, frame, self.fps, min_obs=max(3, self.fit_frames // 2 + 1))

    def _entry(self, zone: Zone, tid: int, frame: int, t: float, stay: _Stay, foot: tuple,
               motion: Motion | None) -> Event:
        """INTRUSION, or a non-alarm ENTRY when the zone's direction filter rejects the motion at entry."""
        if zone.direction is None:
            return Event("INTRUSION", zone.name, tid, frame, t, stay.first_in, foot)
        stay.alarm = zone.direction.matches(motion) is not False      # unknown direction -> alarm (fail-safe)
        return MotionEvent("INTRUSION" if stay.alarm else "ENTRY", zone.name, tid, frame, t, stay.first_in, foot,
                           velocity=motion.velocity_bh() if motion else None)

    def _approach(self, zone: Zone, tid: int, frame: int, t: float, xy: tuple, foot: tuple, stay: _Stay,
                  in_mask: bool, motion: Motion | None) -> Event | None:
        """Advance the APPROACH state of (track, zone) by one observation; return the warning when it fires."""
        rule = zone.approach
        st = self.approach_state.setdefault((tid, zone.name), _Approach())
        eta = None
        if (not stay.inside and not in_mask and motion is not None and motion.speed is not None
                and motion.speed >= rule.min_speed
                and (rule.max_residual is None or motion.residual_bh <= rule.max_residual)
                and not zone.contains(xy) and not zone.contains(motion.foot)
                and (zone.direction is None or zone.direction.matches(motion))):
            eta = time_to_enter(motion.foot, motion.velocity, zone.polygon, rule.horizon_s)
        dist = 0.0
        if not st.armed:                     # foot distance outside the zone (px), only needed to re-arm
            dist = -cv2.pointPolygonTest(zone.polygon, (float(xy[0]), float(xy[1])), True)
            st.closest = min(st.closest, dist)
        if eta is None:
            st.run = 0
            if not st.armed and not stay.inside:
                st.away += 1
                if (st.away >= self.rearm_s * self.fps and motion is not None and motion.height
                        and dist - st.closest >= REARM_BH * motion.height):
                    st.armed, st.away = True, 0
            return None
        target = (motion.foot[0] + motion.velocity[0] * eta, motion.foot[1] + motion.velocity[1] * eta)
        st.run, st.away, st.seen, st.view = st.run + 1, 0, frame, (eta, motion.foot, target)
        if st.armed and st.run >= rule.k:
            st.armed, st.raised = False, frame
            st.closest = -cv2.pointPolygonTest(zone.polygon, (float(xy[0]), float(xy[1])), True)
            return MotionEvent("APPROACH", zone.name, tid, frame, t, frame - st.run + 1, foot,
                               velocity=motion.velocity_bh(), eta_s=round(eta, 2))
        return None

    def _cross(self, line: Tripwire, tid: int, frame: int, t: float, xy: tuple, foot: tuple,
               motion: Motion | None, height: float | None) -> Event | None:
        """Advance the tripwire side state of (track, line); return LINE_CROSS on a confirmed alarm-direction cross.
        A side, also the first one, is confirmed after K observations on it in a row; K observations in a row beyond
        the segment's ends (extended by END_MARGIN_BH box heights, so a foot jittering just past an end still counts)
        forget the side, so walking around an end of the segment is not a crossing."""
        side = line.side(xy, END_MARGIN_BH * height if height else 0.0)
        st = self.line_state.setdefault((tid, line.name), _Side())
        if side is None:                    # beyond the segment's ends
            st.beyond += 1
            if st.beyond >= self.k:
                st.side, st.cand, st.run = 0, 0, 0
            return None
        st.beyond = 0
        if side == 0:                       # exactly on the line: no observation
            return None
        if side == st.side:
            st.cand, st.run = 0, 0
            return None
        if side != st.cand:
            st.cand, st.run, st.first = side, 0, frame
        st.run += 1
        if st.run < self.k:
            return None
        old, st.side, st.cand, st.run = st.side, side, 0, 0
        if old != 0 and side > 0:
            return MotionEvent("LINE_CROSS", line.name, tid, frame, t, st.first, foot,
                               velocity=motion.velocity_bh() if motion else None)
        return None

    def _update_mask(self, tid: int, now_in: bool) -> bool:
        """Advance the track's mask display state by one observation (foot in any mask or not); return it."""
        mask = self.mask_state.setdefault(tid, _Mask())
        if now_in != mask.masked:
            mask.run += 1
            if mask.run >= self.k:
                mask.masked, mask.run = now_in, 0
        else:
            mask.run = 0
        return mask.masked

    def is_masked(self, tid: int) -> bool:
        """True while the track is masked by an ignore polygon (after hysteresis)."""
        mask = self.mask_state.get(tid)
        return mask is not None and mask.masked

    def update(self, frame: int, feet: dict[int, tuple[float, float]],
               heights: dict[int, float] | None = None) -> list[Event]:
        """Feed the foot points (and box heights, used by the motion features) of the tracks reported at `frame`;
        return the events raised at this frame."""
        if heights is None and self.needs_heights:
            raise ValueError("zones with 'approach' or a direction filter need the box heights: update(..., heights)")
        events: list[Event] = []
        self.last_frame = frame
        t = round((frame - self.frame0) / self.fps, 3)
        for tid, xy in feet.items():
            self.last_seen[tid] = frame
            foot = (round(xy[0], 1), round(xy[1], 1))
            motion = self._motion(tid, frame, xy, (heights or {}).get(tid)) if self.motion_on else None
            in_mask = False
            if self.ignore:                      # a foot point in a mask counts as outside every zone
                in_mask = any(region.contains(xy) for region in self.ignore)
                self._update_mask(tid, in_mask)
            for zone in self.zones:
                stay = self.state.setdefault((tid, zone.name), _Stay())
                now_in = zone.contains(xy) and not in_mask
                if now_in != stay.inside:
                    stay.run += 1
                    if now_in and stay.run == 1:
                        stay.first_in = frame
                    if stay.run >= self.k:
                        stay.inside, stay.run = now_in, 0
                        if now_in:
                            events.append(self._entry(zone, tid, frame, t, stay, foot, motion))
                        else:
                            events.append(Event("EXIT", zone.name, tid, frame, t, stay.first_in, foot))
                            stay.loitered, stay.alarm = False, True
                            if zone.approach:
                                self.approach_state.pop((tid, zone.name), None)    # re-armed after the exit
                else:
                    stay.run = 0
                dwell_frames = frame - stay.first_in
                if stay.inside and not stay.loitered and dwell_frames >= zone.dwell_s * self.fps:
                    stay.loitered = True
                    events.append(Event("LOITERING", zone.name, tid, frame, t, stay.first_in, foot))
                if zone.approach:
                    warning = self._approach(zone, tid, frame, t, xy, foot, stay, in_mask, motion)
                    if warning is not None:
                        events.append(warning)
            for line in self.lines:
                crossing = self._cross(line, tid, frame, t, xy, foot, motion, (heights or {}).get(tid))
                if crossing is not None:
                    events.append(crossing)
        if self.max_missing is not None:
            lost = [tid for tid, seen in self.last_seen.items() if frame - seen > self.max_missing]
            for tid in lost:
                for zone in self.zones:
                    stay = self.state.pop((tid, zone.name), None)
                    if stay is not None and stay.inside:
                        events.append(Event("EXIT", zone.name, tid, frame, t, stay.first_in))
                    self.approach_state.pop((tid, zone.name), None)
                for line in self.lines:
                    self.line_state.pop((tid, line.name), None)
                del self.last_seen[tid]
                self.mask_state.pop(tid, None)
                self.history.pop(tid, None)
        return events

    def inside_zones(self, tid: int) -> list[str]:
        """Names of the zones the track is currently inside (after hysteresis)."""
        names = []
        for zone in self.zones:
            stay = self.state.get((tid, zone.name))
            if stay is not None and stay.inside:
                names.append(zone.name)
        return names

    def alarm_zones(self, tid: int) -> list[str]:
        """Zones the track is inside with an alarm (INTRUSION) stay; non-alarm ENTRY stays are left out."""
        return [name for name in self.inside_zones(tid) if self.state[(tid, name)].alarm]

    def approach_view(self, tid: int) -> tuple | None:
        """(zone, eta_s, fitted foot, predicted entry point) of the track's newest raised APPROACH warning whose
        condition held at the last update or at most HOLD_S before it (eta counted down since then), else None."""
        best, newest = None, -1
        for zone in self.zones:
            st = self.approach_state.get((tid, zone.name))
            if (st is None or st.armed or st.view is None or self.last_frame - st.seen > HOLD_S * self.fps
                    or st.raised <= newest):
                continue
            eta, foot, target = st.view
            best, newest = (zone.name, max(0.0, eta - (self.last_frame - st.seen) / self.fps), foot, target), st.raised
        return best


# ------------------------------------------------------------------------------------------------ detect / track
class PedestrianDetector:
    """Trained 8-class checkpoint -> pedestrian boxes [N, 5] (x1, y1, x2, y2, score) in source pixels."""

    def __init__(self, family: str, weights: str, hw: tuple[int, int] = (256, 416), device: str | None = None):
        from efpmot.data import letterbox           # the benchmark's letterbox
        from efpmot.models import build             # plain / S3x YOLOv8n graph + trained checkpoint (strict load)
        self.letterbox, self.hw = letterbox, tuple(hw)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.adapter = build(family, 8, weights)
        self.adapter.model.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, img: np.ndarray, conf: float = 0.01, nms: float = 0.65, max_det: int = 300) -> np.ndarray:
        import torchvision
        h, w = img.shape[:2]
        canvas, ratio = self.letterbox(img, self.hw)
        x = torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1))).float().unsqueeze(0).to(self.device)
        with torch.autocast(self.device, dtype=torch.float16, enabled=self.device == "cuda"):
            boxes, scores = self.adapter.infer(x)
        boxes, scores = boxes[0].float(), scores[0].float()
        score, cls = scores.max(1)
        keep = score >= conf
        boxes, score, cls = boxes[keep], score[keep], cls[keep]
        if len(boxes):
            keep = torchvision.ops.batched_nms(boxes, score, cls, nms)[:max_det]
            boxes, score, cls = boxes[keep], score[keep], cls[keep]
        is_ped = cls == PED
        out = (boxes[is_ped] / ratio).cpu().numpy()
        out[:, [0, 2]] = out[:, [0, 2]].clip(0, w)
        out[:, [1, 3]] = out[:, [1, 3]].clip(0, h)
        return np.concatenate([out, score[is_ped].cpu().numpy()[:, None]], 1).astype(np.float32)


class PedestrianTracker:
    """ByteTrack with the benchmark settings (efpmot.track, MOT17) -> tracks [M, 6] (x1, y1, x2, y2, id, score)."""

    def __init__(self, thr: float, fps: float = 30.0):
        import boxmot.utils  # noqa: F401  (sets up the 'boxmot' logger on import, so quiet it afterwards)
        from efpmot.track import make_tracker       # the benchmark's tracker factory
        # BoxMOT logs its base-class defaults (det_thresh=0.3, max_age=30) before ByteTrack overrides them;
        # the values in use are track_thresh=thr and track_buffer=15.
        logging.getLogger("boxmot").setLevel(logging.WARNING)
        self.thr = thr
        self.tracker = make_tracker("bytetrack", frame_rate=int(round(fps)), track_buffer=15, track_thresh=thr,
                                    match_thresh=0.8)

    def update(self, dets: np.ndarray, img: np.ndarray) -> np.ndarray:
        """dets [N, 5] (x1, y1, x2, y2, score) -> active tracks [M, 6]."""
        widths, heights = dets[:, 2] - dets[:, 0], dets[:, 3] - dets[:, 1]
        ok = (widths > 0.5) & (heights > 0.5) & (dets[:, 4] >= self.thr * 0.2)
        kept = dets[ok]
        if len(kept):
            inp = np.concatenate([kept, np.full((len(kept), 1), PED, np.float32)], 1)
        else:
            inp = np.zeros((0, 6), np.float32)
        res = self.tracker.update(inp, img)
        if res is None or not len(res):
            return np.zeros((0, 6), np.float32)
        res = np.asarray(res, np.float32)             # boxmot: x1, y1, x2, y2, id, conf, cls, det index
        return np.concatenate([res[:, :4], res[:, 4:5] + 1, res[:, 5:6]], 1)   # ids + 1 as in efpmot.track


# --------------------------------------------------------------------------------------------------------- frames
def iter_frames(frames_dir: str | None = None, video: str | None = None, start: int = 1,
                stop: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame number, BGR image). frames_dir: sorted *.jpg / *.png numbered 1..N; start / stop inclusive."""
    if frames_dir:
        paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")) + glob.glob(os.path.join(frames_dir, "*.png")))
        for number, path in enumerate(paths, 1):
            if number >= start and (stop is None or number <= stop):
                yield number, cv2.imread(path)
        return
    cap = cv2.VideoCapture(video)
    number = 0
    while True:
        ok, img = cap.read()
        number += 1
        if not ok or (stop is not None and number > stop):
            break
        if number >= start:
            yield number, img
    cap.release()


# --------------------------------------------------------------------------------------------------------- drawing
ZONE_COLORS = [(40, 40, 230), (0, 170, 255), (220, 120, 0), (180, 0, 180)]   # BGR, zone i
GREEN, RED, WHITE, GREY, DARK = (60, 200, 60), (40, 40, 235), (255, 255, 255), (170, 170, 170), (32, 32, 32)
YELLOW, NEUTRAL, LINE_COLOR = (0, 230, 255), (215, 165, 70), (230, 70, 230)    # APPROACH, ENTRY, tripwire (BGR)
EVENT_COLORS = {"INTRUSION": (110, 110, 255), "LOITERING": (0, 200, 255), "APPROACH": (90, 240, 240),
                "ENTRY": (230, 195, 140), "LINE_CROSS": (245, 130, 245)}
ARROW_S = 1.5         # the APPROACH arrow shows at most the next 1.5 s of the predicted foot path
FONT = cv2.FONT_HERSHEY_SIMPLEX


def zone_color(i: int) -> tuple[int, int, int]:
    """BGR colour of the i-th zone."""
    return ZONE_COLORS[i % len(ZONE_COLORS)]


def text(img: np.ndarray, s: str, org: tuple[int, int], scale: float, color=WHITE, bg=None, th: int = 1) -> int:
    """cv2 text with an optional filled background; returns the x after the text."""
    (tw, h), base = cv2.getTextSize(s, FONT, scale, th)
    if bg is not None:
        cv2.rectangle(img, (org[0] - 2, org[1] - h - 3), (org[0] + tw + 2, org[1] + base), bg, -1)
    cv2.putText(img, s, org, FONT, scale, color, th, cv2.LINE_AA)
    return org[0] + tw


def draw_zones(img: np.ndarray, zones: list[Zone], s: float = 1.0, alpha: float = 0.25) -> None:
    """Semi-transparent zone polygons (s: display scale of source pixels)."""
    over = img.copy()
    polys = [np.round(z.polygon * s).astype(np.int32) for z in zones]
    for i, poly in enumerate(polys):
        cv2.fillPoly(over, [poly], zone_color(i))
    cv2.addWeighted(over, alpha, img, 1 - alpha, 0, dst=img)
    for i, poly in enumerate(polys):
        cv2.polylines(img, [poly], True, zone_color(i), 2, cv2.LINE_AA)


def draw_ignore(img: np.ndarray, regions: list[IgnoreRegion], s: float = 1.0, label_scale: float = 0.45) -> None:
    """Ignore masks: grey hatching inside the polygon, grey outline and an 'ignore' label."""
    for region in regions:
        poly = np.round(region.polygon * s).astype(np.int32)
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        hatch = np.zeros_like(mask)
        h, w = mask.shape
        for c in range(-h, w, 10):                 # 45-degree lines every 10 px
            cv2.line(hatch, (c, h), (c + h, 0), 255, 2)
        img[(mask > 0) & (hatch > 0)] = (150, 150, 150)
        cv2.polylines(img, [poly], True, (150, 150, 150), 2, cv2.LINE_AA)
        x, y = poly.min(0)
        text(img, f"ignore: {region.name}", (int(x), max(12, int(y) - 4)), label_scale, WHITE, (110, 110, 110))


def draw_directions(img: np.ndarray, zones: list[Zone], s: float = 1.0, scale: float = 0.45,
                    label: bool = True) -> None:
    """Alarm direction of each zone with a direction filter: an arrow across the polygon edge it faces, labelled with
    the accepted angle. label=False (live views): a shorter, half-transparent arrow without text, so people under it
    stay visible."""
    over = img.copy()
    for i, zone in enumerate(zones):
        if zone.direction is None:
            continue
        dx, dy = zone.direction.vector
        edge = max(range(len(zone.polygon)), key=lambda j: np.dot(inward_normal(zone.polygon, j), (dx, dy)))
        mx, my = (zone.polygon[edge] + zone.polygon[(edge + 1) % len(zone.polygon)]) / 2 * s
        length = 60 if label else 40                   # display px
        tail = (int(mx - dx * length), int(my - dy * length))
        head = (int(mx + dx * length * 0.6), int(my + dy * length * 0.6))
        if not label:
            cv2.arrowedLine(over, tail, head, zone_color(i), 3, cv2.LINE_AA, tipLength=0.35)
            continue
        cv2.arrowedLine(img, tail, head, WHITE, 6, cv2.LINE_AA, tipLength=0.35)
        cv2.arrowedLine(img, tail, head, zone_color(i), 3, cv2.LINE_AA, tipLength=0.35)
        name = f"alarm entry (within {math.degrees(math.acos(zone.direction.min_cos)):.0f} deg)"
        tw = cv2.getTextSize(name, FONT, scale, 1)[0][0]
        text(img, name, (int(np.clip(tail[0] + 6, 4, img.shape[1] - tw - 4)), tail[1] + 4), scale, WHITE,
             zone_color(i))
    if not label:
        cv2.addWeighted(over, 0.6, img, 0.4, 0, dst=img)


def draw_lines(img: np.ndarray, lines: list[Tripwire], s: float = 1.0, scale: float = 0.45) -> None:
    """Tripwires: the segment and an arrow across its middle in the alarm direction."""
    for line in lines:
        a, b = (int(line.a[0] * s), int(line.a[1] * s)), (int(line.b[0] * s), int(line.b[1] * s))
        cv2.line(img, a, b, WHITE, 5, cv2.LINE_AA)
        cv2.line(img, a, b, LINE_COLOR, 3, cv2.LINE_AA)
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        nx, ny = line.normal
        length = 40                                    # display px
        tail, head = (int(mx - nx * length), int(my - ny * length)), (int(mx + nx * length), int(my + ny * length))
        cv2.arrowedLine(img, tail, head, WHITE, 6, cv2.LINE_AA, tipLength=0.3)
        cv2.arrowedLine(img, tail, head, LINE_COLOR, 3, cv2.LINE_AA, tipLength=0.3)
        text(img, f"tripwire {line.name}", (a[0] + 4, a[1] - 8), scale, WHITE, LINE_COLOR)


def draw_tracks(img: np.ndarray, tracks: np.ndarray, monitor: ZoneMonitor, s: float = 1.0,
                id_scale: float = 0.4) -> None:
    """Boxes and foot points: red while the track is inside a zone with an alarm, blue inside after a non-alarm
    ENTRY, yellow with the predicted-path arrow and 'APPROACH n s' during an active APPROACH warning, faint grey with
    'masked' while it is masked, else green."""
    for x1, y1, x2, y2, tid, _ in tracks:
        if monitor.is_masked(int(tid)):
            p1, p2 = (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s))
            over = img.copy()
            cv2.rectangle(over, p1, p2, GREY, 1, cv2.LINE_AA)
            cv2.addWeighted(over, 0.5, img, 0.5, 0, dst=img)
            text(img, f"{int(tid)} masked", (p1[0] + 1, max(10, p1[1] - 2)), id_scale * 0.9, (210, 210, 210))
            continue
        hot = bool(monitor.alarm_zones(int(tid)))
        entry = not hot and bool(monitor.inside_zones(int(tid)))
        warn = None if hot or entry else monitor.approach_view(int(tid))
        color = RED if hot else NEUTRAL if entry else YELLOW if warn else GREEN
        p1, p2 = (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s))
        foot = (int((x1 + x2) / 2 * s), int(y2 * s))
        cv2.rectangle(img, p1, p2, color, 2 if hot or warn else 1, cv2.LINE_AA)
        cv2.circle(img, foot, 3, color, -1, cv2.LINE_AA)
        label = str(int(tid))
        if warn:
            _, eta, start, target = warn
            frac = min(1.0, ARROW_S / eta)
            head = (int((start[0] + (target[0] - start[0]) * frac) * s), int((start[1] + (target[1] - start[1]) * frac) * s))
            cv2.arrowedLine(img, foot, head, YELLOW, 2, cv2.LINE_AA, tipLength=0.3)
            label = f"{int(tid)} APPROACH {eta:.1f} s"
        text(img, label, (p1[0] + 1, max(10, p1[1] - 2)), id_scale, (0, 0, 0) if warn else WHITE, color)


def status_bar(width: int, recent: list[Event], zones: list[Zone], scale: float = 0.55) -> np.ndarray:
    """Top bar: red banner with the newest alarm while any alarm is recent, amber with the newest APPROACH warning when
    only warnings are, else 'monitoring' and the zone legend. `recent` holds alarms and warnings only (no ENTRY /
    EXIT); '(+n)' counts the other recent ones."""
    h = int(round(54 * scale))
    alarms = [e for e in recent if e.type != "APPROACH"]
    warning = bool(recent) and not alarms
    bar = np.full((h, width, 3), (0, 120, 200) if warning else (0, 0, 190) if recent else DARK, np.uint8)
    y = int(h * 0.68)
    if recent:
        newest = alarms[-1] if alarms else recent[-1]
        more = f"  (+{len(recent) - 1})" if len(recent) > 1 else ""
        if warning:
            msg = (f"WARNING  APPROACH  {newest.zone}  track {newest.track_id}  entry in ~{newest.eta_s:.1f} s  "
                   f"at {newest.t:.1f} s{more}")
        else:
            msg = f"ALERT  {newest.type}  {newest.zone}  track {newest.track_id}  at {newest.t:.1f} s{more}"
        text(bar, msg, (8, y), scale, WHITE, th=2)
        return bar
    x = text(bar, "monitoring - no active alert", (8, y), scale, GREY) + 24
    unit = scale / 0.55
    for i, zone in enumerate(zones):
        cv2.rectangle(bar, (x, y - int(12 * unit)), (x + int(12 * unit), y), zone_color(i), -1)
        x = text(bar, f"{zone.name} (loiter {zone.dwell_s:g} s)", (x + int(18 * unit), y), scale * 0.85, WHITE) + 16
    return bar


def event_strip(width: int, log: list[Event], n: int = 3, scale: float = 0.5, header: str | None = None) -> np.ndarray:
    """Bottom strip: optional header line (e.g. a counter) and the last n alerts, newest first."""
    line_h = int(round(40 * scale))
    rows = ([header] if header else []) + list(reversed(log[-n:]))
    strip = np.full((line_h * (n + bool(header)) + 6, width, 3), DARK, np.uint8)
    for i, row in enumerate(rows):
        y = line_h * (i + 1) - 4
        if isinstance(row, str):
            text(strip, row, (8, y), scale, WHITE, th=2 if scale >= 0.5 else 1)
        else:
            msg = f"{row.t:5.1f} s  {row.type:<9}  {row.zone}  track {row.track_id}"
            if row.type == "APPROACH":
                msg += f"  (entry predicted in {row.eta_s:.1f} s)"
            elif row.type == "ENTRY":
                msg += "  (not the alarm direction)"
            text(strip, msg, (8, y), scale, EVENT_COLORS.get(row.type, WHITE))
    return strip


def zone_layout(img: np.ndarray, zones: list[Zone], width: int = 1280,
                ignore: list[IgnoreRegion] | None = None, lines: list[Tripwire] | None = None) -> np.ndarray:
    """Still image of the zone layout (name, kind and loitering limit at each polygon's centroid; ignore masks; alarm
    directions and tripwires with arrows)."""
    s = width / img.shape[1]
    vis = cv2.resize(img, (width, int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    draw_zones(vis, zones, s, alpha=0.35)
    draw_ignore(vis, ignore or [], s)
    draw_directions(vis, zones, s)
    draw_lines(vis, lines or [], s)
    for i, zone in enumerate(zones):
        cx, cy = (zone.polygon.mean(0) * s).astype(int)
        extra = f", approach {zone.approach.horizon_s:g} s" if zone.approach else ""
        if zone.kind:
            label = f"{zone.name} ({zone.kind}, loiter {zone.dwell_s:g} s{extra})"
        else:
            label = f"{zone.name} (loiter {zone.dwell_s:g} s{extra})"
        tw = cv2.getTextSize(label, FONT, 0.6, 2)[0][0]
        x = int(np.clip(cx - tw // 2, 4, width - tw - 4))
        text(vis, label, (x, int(cy)), 0.6, WHITE, zone_color(i), 2)
    return np.vstack([vis, attribution_bar(width)])


def attribution_bar(width: int, scale: float = 0.45, extra: str = "") -> np.ndarray:
    """Black bar with the MOT17 licence attribution (drawn on every rendered frame)."""
    bar = np.full((int(round(46 * scale)), width, 3), 0, np.uint8)
    line = (extra + "   " if extra else "") + ATTRIBUTION
    text(bar, line, (8, int(bar.shape[0] * 0.72)), scale, (200, 200, 200))
    return bar


# --------------------------------------------------------------------------------------------------------- video
def pad_to(img: np.ndarray, m: int = 8) -> np.ndarray:
    """Pad the bottom / right with black to multiples of m (the H.264 encoder needs even sizes)."""
    h, w = img.shape[:2]
    return cv2.copyMakeBorder(img, 0, -h % m, 0, -w % m, cv2.BORDER_CONSTANT, value=0)


def open_mp4(path: str, fps: float, wh: tuple[int, int]) -> tuple[cv2.VideoWriter, str]:
    """H.264 through Windows Media Foundation when available (the pip OpenCV FFMPEG build ships no H.264 encoder),
    else avc1 through the default backend, else mp4v."""
    for api, fourcc in ((cv2.CAP_MSMF, "avc1"), (cv2.CAP_MSMF, "H264"), (None, "avc1")):
        code = cv2.VideoWriter_fourcc(*fourcc)
        try:
            if api is None:
                writer = cv2.VideoWriter(path, code, fps, wh)
            else:
                writer = cv2.VideoWriter(path, api, code, fps, wh)
            if writer.isOpened():
                return writer, f"h264 ({fourcc})"
            writer.release()
        except cv2.error:
            pass
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, wh), "mp4v"


# --------------------------------------------------------------------------------------------------------- app
@dataclass
class Step:
    frame: int
    img: np.ndarray
    tracks: np.ndarray                    # [M, 6] x1, y1, x2, y2, id, score
    events: list[Event]
    ms: dict[str, float]                  # read / detect / track / alert wall time of this frame


def observations(tracks: np.ndarray) -> tuple[dict[int, tuple[float, float]], dict[int, float]]:
    """Tracks [M, >=5] (x1, y1, x2, y2, id, ...) -> ({id: foot point}, {id: box height}) for ZoneMonitor.update."""
    return ({int(r[4]): foot_point(r) for r in tracks}, {int(r[4]): float(r[3] - r[1]) for r in tracks})


def pipeline(frames: Iterator[tuple[int, np.ndarray]], detector: PedestrianDetector, tracker: PedestrianTracker,
             monitor: ZoneMonitor) -> Iterator[Step]:
    """Detect -> track -> zone logic, one frame at a time, with per-stage wall times (GPU synchronised)."""
    use_cuda = detector.device == "cuda"
    while True:
        t0 = time.perf_counter()
        item = next(frames, None)
        if item is None:
            return
        number, img = item
        t1 = time.perf_counter()
        dets = detector(img)
        if use_cuda:
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        tracks = tracker.update(dets, img)
        t3 = time.perf_counter()
        events = monitor.update(number, *observations(tracks))
        t4 = time.perf_counter()
        ms = {"read": 1e3 * (t1 - t0), "detect": 1e3 * (t2 - t1), "track": 1e3 * (t3 - t2), "alert": 1e3 * (t4 - t3)}
        yield Step(number, img, tracks, events, ms)


def timing_report(ms_rows: list[dict[str, float]], warmup: int, device: str = "cuda") -> dict:
    """Mean ms per stage over the frames after `warmup`; pipeline FPS = 1000 / (detect + track + alert)."""
    rows = ms_rows[warmup:]
    ms = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    pipe = ms["detect"] + ms["track"] + ms["alert"]
    gpu = torch.cuda.get_device_name(0) if device == "cuda" and torch.cuda.is_available() else "cpu"
    return dict(timed_frames=len(rows), ms_per_frame=ms, pipeline_fps=1000 / pipe,
                end_to_end_fps=1000 / sum(ms.values()), gpu=gpu)


def render_frame(step: Step, zones: list[Zone], monitor: ZoneMonitor, log: list[Event], fps: float,
                 out_width: int) -> np.ndarray:
    """Annotated output frame: alert banner, zones, ignore masks and tracks, last alerts, attribution."""
    s = out_width / step.img.shape[1]
    vis = cv2.resize(step.img, (out_width, int(round(step.img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    draw_zones(vis, zones, s)
    draw_ignore(vis, monitor.ignore, s)
    draw_directions(vis, zones, s, label=False)
    draw_lines(vis, monitor.lines, s)
    draw_tracks(vis, step.tracks, monitor, s)
    recent = [e for e in log if step.frame - e.frame < BANNER_S * fps and e.type != "ENTRY"]
    width = vis.shape[1]
    return pad_to(np.vstack([status_bar(width, recent, zones), vis, event_strip(width, log), attribution_bar(width)]))


def run(args: argparse.Namespace) -> dict:
    """Run the app on a frames folder or a video; write the annotated MP4 and the events file; return timing."""
    if not args.weights:
        raise SystemExit("--weights is required")
    zones = load_zones(args.zones)
    fps = args.fps
    if args.video:
        fps = cv2.VideoCapture(args.video).get(cv2.CAP_PROP_FPS) or args.fps
    detector = PedestrianDetector(args.family, args.weights, tuple(args.hw), args.device)
    tracker = PedestrianTracker(args.thr, fps)
    monitor = ZoneMonitor(zones, fps, k=args.k, frame0=args.start, ignore=load_ignore(args.zones),
                          lines=load_lines(args.zones))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    events_path = os.path.splitext(args.out)[0] + ".events.jsonl"
    writer, codec, log, ms_rows = None, "", [], []
    frames = iter_frames(args.frames, args.video, args.start, args.stop)
    with open(events_path, "w", encoding="utf-8") as events_file:
        for step in pipeline(frames, detector, tracker, monitor):
            t0 = time.perf_counter()
            for event in step.events:
                events_file.write(json.dumps(asdict(event)) + "\n")
            log += [e for e in step.events if e.type != "EXIT"]
            vis = render_frame(step, zones, monitor, log, fps, args.out_width)
            if writer is None:
                writer, codec = open_mp4(args.out, fps, (vis.shape[1], vis.shape[0]))
            writer.write(vis)
            ms_rows.append({**step.ms, "draw+write": 1e3 * (time.perf_counter() - t0)})
    if writer is not None:
        writer.release()
    counts = {t: sum(e.type == t for e in log) for t in ("INTRUSION", "LOITERING")}
    counts.update({t: n for t in ("APPROACH", "ENTRY", "LINE_CROSS") if (n := sum(e.type == t for e in log))})
    report = dict(frames=len(ms_rows), **timing_report(ms_rows, args.warmup, detector.device), video=args.out, codec=codec,
                  events_file=events_path, events=counts)
    stages = "  ".join(f"{k} {v:.2f}" for k, v in report["ms_per_frame"].items())
    print(f"{report['frames']} frames on {report['gpu']} (first {args.warmup} excluded from timing)")
    print(f"per stage [ms/frame]: {stages}")
    print(f"pipeline (detect + track + alert) {report['pipeline_fps']:.1f} FPS | end-to-end incl. JPEG decode, "
          f"drawing and encoding {report['end_to_end_fps']:.1f} FPS")
    print(f"alerts: {counts}  ->  {args.out} [{codec}], {events_path}")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames", help="folder of numbered frames (*.jpg / *.png)")
    src.add_argument("--video", help="video file")
    ap.add_argument("--start", type=int, default=1, help="first frame number (1-based, inclusive)")
    ap.add_argument("--stop", type=int, default=None, help="last frame number (inclusive)")
    ap.add_argument("--fps", type=float, default=30.0, help="frame rate of a frames folder")
    ap.add_argument("--family", default="yolov8n_s3x", choices=["yolov8n", "yolov8n_s3x"])
    ap.add_argument("--weights", help="trained checkpoint: release file or <runs>/<name>/last.pth")
    ap.add_argument("--hw", type=int, nargs=2, default=[256, 416], help="detector canvas H W")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"],
                    help="detector device (default: cuda when available; cpu runs FP32 and is slow)")
    ap.add_argument("--zones", required=True)
    ap.add_argument("--thr", type=float, default=0.8, help="ByteTrack track threshold")
    ap.add_argument("--k", type=int, default=K_FRAMES, help="zone hysteresis in frames")
    ap.add_argument("--out", default="annotated.mp4")
    ap.add_argument("--out_width", type=int, default=960)
    ap.add_argument("--warmup", type=int, default=10, help="frames excluded from the timing")
    ap.add_argument("--report", default=None, help="optional json with the timing report")
    ap.add_argument("--layout", default=None, help="only write the zone layout on the first frame to this PNG and exit")
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.layout:
        _, first = next(iter_frames(args.frames, args.video, args.start, args.stop))
        cv2.imwrite(args.layout, zone_layout(first, load_zones(args.zones), ignore=load_ignore(args.zones),
                                             lines=load_lines(args.zones)))
        return
    report = run(args)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
