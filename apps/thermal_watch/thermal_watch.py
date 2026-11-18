# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Thermal Watch: person search in night thermal drone video, on the S3x detector (README.md in this folder).

Live screen, per frame (causal: only past frames are used, so a camera can replace the file):
  frame (640 x 512 LWIR, white-hot) -> person detector (release weights, letterbox to the model canvas, FP16 on CUDA, conf 0.05,
  class-wise NMS 0.65) -> drone-motion compensation (global shift between frames by phase correlation, accumulated; detections are
  tracked in these ground-stabilised coordinates) -> ByteTrack on persons (track_thresh = the model's validation max-F1 person
  threshold, match 0.8, buffer 15) -> confirmation: one alert when a track was seen in >= 3 of its last 5 frames -> re-acquisition:
  a track that starts within max(20 px, 2.5 x box size) of a confirmed person lost for <= 6 s keeps that person's number (no new
  alert) -> flat-ground slant range from altitude, camera angle (90 = straight down) and the box centre (f = 1470 px: 25 mm lens,
  17 um pixels) -> screen 1280 x 720.
Review screen (--review): the recording is analysed once from start to end with the same pipeline, so every alert is already on
the playback bar and in the list; click a card or a bar mark to jump there.

usage:
  python apps/thermal_watch/thermal_watch.py VIDEO [--show | --review] [--weights FILE] [--start S] [--dur D] [--out DIR]
  --weights defaults to the S3x release file of the HIT-UAV fold whose test split holds the video's recording (sample_videos.json);
  for another checkpoint also give --family, --hw H W and --thr.
Outputs (<outputs.results>/thermal_watch/<video>__<weights>/): console.mp4, events.jsonl, chips/*.png, event_index.png, summary.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # run from a checkout without installing

from efpmot import paths  # noqa: E402
from efpmot.data import BDD_CLASSES, letterbox  # noqa: E402
from efpmot.release import RELEASES, release_meta  # noqa: E402
from efpmot.track.runner import make_tracker  # noqa: E402

PED = BDD_CLASSES.index("pedestrian")
F_PX = 1470.0
W_OUT, H_OUT = 1280, 720
SAMPLE_VIDEOS = os.path.join(HERE, "sample_videos.json")


# ---------------------------------------------------------------------------------------------------------------- inputs
def video_meta(path: str) -> dict:
    """altitude / camera angle / day-night of the HIT-UAV recording a sample video is the source of (sample_videos.json), else
    parsed from a name like '120m-30_3' (120 m, 30 degrees)"""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"(\d+)m-(\d+)_", stem)
    meta = dict(stem=stem, alt=float(m.group(1)) if m else None, angle=float(m.group(2)) if m else None, daynight=None,
                rec=None, test_fold=None)
    if os.path.isfile(SAMPLE_VIDEOS):
        v = json.load(open(SAMPLE_VIDEOS, encoding="utf-8"))["videos"].get(stem)
        if v:
            meta.update(rec=v["recording"], alt=float(v["altitude"]), angle=float(v["angle"]), daynight=v["daynight"], test_fold=v["test_fold"])
    return meta


def release_for(fold: str | None, family: str = "yolov8n_s3x") -> str | None:
    """release file of `family` trained on the HIT-UAV fold `fold` (seed 0), e.g. for a video whose recording is in that fold's
    test split; None if there is none"""
    for fn, m in RELEASES.items():
        if fold and m.get("family") == family and str(m.get("dataset", "")).startswith("hituav") and str(m["dataset"]).endswith(fold):
            return os.path.join(paths.repo_root(), "release_assets", fn)
    return None


def threshold_for(weights: str) -> float | None:
    """the validation max-F1 person threshold of a release file (efpmot.release metadata or results/hituav/<run>.json)"""
    m = release_meta(weights) or {}
    if m.get("thr") is not None:
        return float(m["thr"])
    f = os.path.join(paths.repo_root(), "results", "hituav", f"{m.get('run', '')}.json")
    if m.get("run") and os.path.isfile(f):
        d = json.load(open(f, encoding="utf-8"))
        op = d.get("op", {}).get("pedestrian", {})
        if "thr" in op:
            return float(op["thr"])
    return None


class Detector:
    """person detector on a trained checkpoint: letterbox to the canvas, FP16 on CUDA, max class score >= conf, class-wise NMS
    0.65, at most 300 boxes; boxes in native pixels (the settings of the HIT-UAV evaluation export)"""

    def __init__(self, weights: str, family: str | None = None, hw=None, device: str | None = None):
        import torch
        from efpmot.models import build
        m = release_meta(weights) or {}
        self.family = family or m.get("family")
        self.hw = tuple(int(v) for v in (hw or m.get("hw") or (320, 416)))
        if self.family is None:
            raise ValueError(f"{weights} is not a release file: give --family (and --hw, --thr)")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ad = build(self.family, 8, weights)
        self.ad.model.to(self.device).eval()
        self.name = os.path.splitext(os.path.basename(weights))[0]

    def __call__(self, img: np.ndarray, conf: float = 0.05) -> np.ndarray:
        import torch
        import torchvision
        can, r = letterbox(img, self.hw)
        x = torch.from_numpy(np.ascontiguousarray(can.transpose(2, 0, 1))).float().unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.device.startswith("cuda")):
            boxes, scores = self.ad.infer(x)
        boxes, scores = boxes[0].float(), scores[0].float()
        cv, ci = scores.max(1)
        k = cv >= conf
        bx, cv, ci = boxes[k], cv[k], ci[k]
        if not len(bx):
            return np.zeros((0, 6), np.float32)
        keep = torchvision.ops.batched_nms(bx, cv, ci, 0.65)[:300]
        return np.concatenate([(bx[keep] / r).cpu().numpy(), cv[keep].cpu().numpy()[:, None], ci[keep].cpu().numpy()[:, None]], 1).astype(np.float32)


# --------------------------------------------------------------------------------------------------------------- pipeline
def slant_range(box, meta: dict, H: int = 512, W: int = 640) -> float | None:
    """flat-ground distance from the camera to the box centre [m]: depression = camera angle + atan((yc - H/2) / f),
    lateral = atan((xc - W/2) / f), range = altitude / sin(depression) / cos(lateral)"""
    if meta.get("alt") is None or meta.get("angle") is None:
        return None
    xc, yc = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    dep = math.radians(meta["angle"]) + math.atan((yc - H / 2) / F_PX)
    lat = math.atan((xc - W / 2) / F_PX)
    return meta["alt"] / math.sin(min(max(dep, 0.05), math.pi - 0.05)) / math.cos(lat)


def make_crop(frame: np.ndarray, box, out: int = 120) -> np.ndarray:
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    s = int(min(max(24.0, 3.0 * max(x2 - x1, y2 - y1)), H, W))
    a = int(min(max(round((x1 + x2) / 2 - s / 2), 0), W - s))
    b = int(min(max(round((y1 + y2) / 2 - s / 2), 0), H - s))
    return cv2.resize(frame[b:b + s, a:a + s], (out, out), interpolation=cv2.INTER_CUBIC)


class ThermalWatch:
    """streaming detector + tracker + confirmation; step(frame) -> the person tracks of this frame (dicts). detector(frame) ->
    float32 [N, 6] (x1, y1, x2, y2, score, class index); thr = the tracker's track_thresh."""

    def __init__(self, detector, thr: float, meta: dict, fps: float, k: int = 3, n: int = 5, cmc: bool = True, relink_s: float = 6.0):
        self.detector, self.thr, self.meta, self.fps, self.k, self.n = detector, float(thr), meta, fps, k, n
        self.tracker = make_tracker("bytetrack", track_thresh=self.thr, match_thresh=0.8, track_buffer=15)
        self.cmc, self.relink = cmc, int(round(relink_s * fps))
        self.C, self.prev, self.win, self.shifts = np.zeros(2), None, None, []
        self.seen, self.disp, self.confirm_frame, self.last, self.events, self.t = {}, {}, {}, {}, [], -1
        self.where = {}  # raw id -> (stabilised centre, size, last frame, stabilised box) of confirmed tracks
        self.relinked, self._now, self.off = 0, set(), np.zeros(4, np.float32)

    def _shift(self, frame: np.ndarray) -> np.ndarray:
        g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (frame.shape[1] // 2, frame.shape[0] // 2), interpolation=cv2.INTER_AREA).astype(np.float32)
        if self.win is None:
            self.win = cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F)
        s = np.zeros(2)
        if self.prev is not None:
            (dx, dy), resp = cv2.phaseCorrelate(self.prev, g, self.win)
            if resp > 0.05 and abs(dx) < g.shape[1] / 4 and abs(dy) < g.shape[0] / 4:
                s = 2.0 * np.array([dx, dy])
        self.prev = g
        self.shifts.append(s.tolist())
        return s

    def step(self, frame: np.ndarray, dets: np.ndarray | None = None) -> list[dict]:
        """dets: precomputed detections of this frame (tests); default: self.detector(frame)"""
        self.t += 1
        t = self.t
        if self.cmc:
            self.C = self.C + self._shift(frame)
        off = np.array([self.C[0], self.C[1], self.C[0], self.C[1]], np.float32)
        d = self.detector(frame) if dets is None else dets
        dp = d[(d[:, 5] == PED) & (d[:, 4] >= self.thr * 0.2)] if len(d) else np.zeros((0, 6), np.float32)
        inp = dp[:, :6].astype(np.float32).copy() if len(dp) else np.zeros((0, 6), np.float32)
        if len(inp):
            inp[:, :4] -= off
        r = self.tracker.update(inp, frame)
        rows = np.asarray(r, dtype=np.float32) if r is not None and len(r) else []
        now = {int(row[4]) for row in rows}
        self._now, self.off = now, off
        tracks = []
        for row in rows:
            tid = int(row[4])
            bw = row[:4].astype(np.float64)
            box = tuple(float(v) for v in (bw + off))
            sc = float(row[5])
            h = self.seen.setdefault(tid, [])
            h.append(t)
            rng = slant_range(box, self.meta)
            cw, sz = np.array([(bw[0] + bw[2]) / 2, (bw[1] + bw[3]) / 2]), float(max(bw[2] - bw[0], bw[3] - bw[1]))
            if tid not in self.disp and (len(h) == 1 or h[-2] < t - 1) and self.relink:  # (re)appearing: a person we just lost?
                best, bd, shown = None, None, {self.disp.get(x) for x in now if x != tid}
                for raw, (c0, s0, t0, _b) in self.where.items():
                    if raw in now or t - t0 > self.relink or self.disp.get(raw) in shown:
                        continue
                    dist = float(np.linalg.norm(cw - c0))
                    if dist <= max(20.0, 2.5 * max(sz, s0)) and (bd is None or dist < bd):
                        best, bd = raw, dist
                if best is not None:  # hand the number over; the old raw track is retired (if it revives it must re-qualify)
                    self.disp[tid] = self.disp.pop(best)
                    self.confirm_frame[tid] = self.confirm_frame.pop(best)
                    self.where.pop(best)
                    self.last[tid] = self.last.pop(best)
                    self.relinked += 1
            if tid not in self.disp and sum(1 for s in h if s > t - self.n) >= self.k:
                self.disp[tid] = 1 + max(self.disp.values(), default=0)
                self.confirm_frame[tid] = t
                self.events.append(dict(id=self.disp[tid], frame=t, t=round(t / self.fps, 2), box=[round(v, 1) for v in box],
                                        score=round(sc, 3), range_m=None if rng is None else int(round(rng)), chip=make_crop(frame, box)))
            if tid in self.disp:
                self.last[tid] = dict(crop=make_crop(frame, box), t=t, range=rng)
                self.where[tid] = (cw, sz, t, bw.copy())
            tracks.append(dict(raw=tid, id=self.disp.get(tid), box=box, score=sc, range=rng))
        return tracks

    def held(self, max_age: int = 3) -> list[dict]:
        """confirmed persons missed for 1..max_age frames, drawn at their last stabilised box moved by the drone motion since"""
        out = []
        for raw, (_c, _s, t0, bw) in self.where.items():
            if raw in self._now or raw not in self.disp or not 0 < self.t - t0 <= max_age:
                continue
            out.append(dict(raw=raw, id=self.disp[raw], box=tuple(float(v) for v in (bw + self.off)), score=0.0,
                            range=self.last.get(raw, {}).get("range")))
        return out


# ----------------------------------------------------------------------------------------------------------------- screens
_FONT_DIRS = [r"C:\Windows\Fonts", "/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype", "/usr/share/fonts"]


def _font(names, size: int, inst: str | None = None):
    """first available font of `names` (Windows fonts, else DejaVu on Linux); inst = a named instance of a variable font"""
    for n in list(names) + ["DejaVuSans.ttf"]:
        for d in _FONT_DIRS:
            p = os.path.join(d, n)
            if os.path.isfile(p):
                f = ImageFont.truetype(p, size)
                if inst:
                    for name in (inst, inst.encode()):
                        try:
                            f.set_variation_by_name(name)
                            break
                        except Exception:
                            pass
                return f
    return ImageFont.load_default()


def fmt_t(s: float) -> str:
    return f"{int(s // 60):02d}:{s % 60:04.1f}"


class Console:
    """The live screen: video first (640 x 512 -> 900 x 720) with an on-screen display (telemetry, REC clock, centre reticle),
    corner brackets on confirmed persons (amber; red with a widening pulse for the first second after the alert) and an alert list
    of snapshot cards (snapshot at alert time, number, distance, time, in view or not). The drawing pieces are shared with
    ReviewScreen."""
    VW, VH = 900, 720
    BG, CARD, CARD_NEW, CARD_CUR, EDGE = (13, 15, 18), (23, 26, 30), (54, 24, 21), (44, 38, 24), (38, 42, 47)
    TXT, TXT2, TXT3 = (236, 239, 241), (150, 158, 165), (96, 103, 110)
    AMBER, RED, WHITE = (246, 168, 38), (255, 76, 60), (248, 249, 250)

    def __init__(self, meta: dict, n_frames: int, fps: float):
        self.meta, self.n_frames, self.fps = meta, n_frames, fps
        B = ["bahnschrift.ttf"]
        self.f_mark, self.f_osd, self.f_tag = _font(B, 21, "Bold SemiCondensed"), _font(B, 16, "SemiBold"), _font(B, 15, "SemiBold")
        self.f_count, self.f_label, self.f_id = _font(B, 66, "Bold"), _font(B, 16, "Regular"), _font(B, 26, "Bold")
        self.f_dist, self.f_head, self.f_status = _font(B, 20, "SemiBold"), _font(B, 19, "SemiBold"), _font(B, 14, "Regular")
        self.f_time = _font(["consola.ttf", "DejaVuSansMono.ttf"], 15)

    def _bracket(self, d, X1, Y1, X2, Y2, col, wd: int = 2):
        L = max(6.0, 0.3 * min(X2 - X1, Y2 - Y1))
        for px, py, dx, dy in ((X1, Y1, 1, 1), (X2, Y1, -1, 1), (X1, Y2, 1, -1), (X2, Y2, -1, -1)):
            d.line([(px, py), (px + dx * L, py)], fill=col, width=wd)
            d.line([(px, py), (px, py + dy * L)], fill=col, width=wd)

    def _base(self, frame: np.ndarray):
        """opaque layer: the video and the rail with its wordmark"""
        base = Image.new("RGB", (W_OUT, H_OUT), self.BG)
        base.paste(Image.fromarray(cv2.cvtColor(cv2.resize(frame, (self.VW, self.VH), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)), (0, 0))
        db = ImageDraw.Draw(base)
        db.line([(self.VW, 0), (self.VW, H_OUT)], fill=self.EDGE)
        x = self.VW + 28
        for ch in "THERMAL WATCH":
            db.text((x, 26), ch, font=self.f_mark, fill=self.TXT)
            x += db.textlength(ch, font=self.f_mark) + 2.4
        return base, db

    def _overlay(self):
        """translucent layer on the video: top shade, centre reticle, telemetry"""
        VW, VH = self.VW, self.VH
        ov = Image.new("RGBA", (W_OUT, H_OUT), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        for i in range(64):
            d.line([(0, i), (VW - 1, i)], fill=(0, 0, 0, int(120 * (1 - i / 64) ** 1.6)))
        cx, cy = VW // 2, VH // 2
        for a, b in (((cx - 20, cy), (cx - 8, cy)), ((cx + 8, cy), (cx + 20, cy)), ((cx, cy - 20), (cx, cy - 8)), ((cx, cy + 8), (cx, cy + 20))):
            d.line([a, b], fill=(255, 255, 255, 110), width=1)
        m, tele = self.meta, []
        if m.get("alt") is not None:
            tele.append(f"ALT {m['alt']:.0f} m")
        if m.get("angle") is not None:
            tele.append(f"CAM {m['angle']:.0f}°")
        if m.get("daynight") is not None:
            tele.append("NIGHT" if m["daynight"] == 1 else "DAY")
        tele.append("WHITE HOT")
        x = 24
        for s_ in tele:
            d.text((x, 20), s_, font=self.f_osd, fill=self.WHITE + (235,))
            x += d.textlength(s_, font=self.f_osd) + 26
        return ov, d

    def _marks(self, d, frame_w: int, shown: list[dict], age_of, bottom: int = 6):
        """corner brackets (red + widening pulse while 0 <= age < 1 s, else amber) and number / distance tags placed clear of
        every other tag and bracket (number only, or no tag, when there is no room; the distance is also on the alert card)"""
        fps, VW, VH = self.fps, self.VW, self.VH
        s = VW / frame_w
        marks = []
        for tr in sorted(shown, key=lambda z: z["id"]):
            x1, y1, x2, y2 = (v * s for v in tr["box"])
            X1, Y1, X2, Y2 = x1 - 5, y1 - 5, x2 + 5, y2 + 5
            if X2 - X1 < 24:
                c = (X1 + X2) / 2
                X1, X2 = c - 12, c + 12
            if Y2 - Y1 < 24:
                c = (Y1 + Y2) / 2
                Y1, Y2 = c - 12, c + 12
            age = age_of(tr)
            new = 0 <= age < fps
            col = self.RED if new else self.AMBER
            self._bracket(d, X1, Y1, X2, Y2, col + (255,))
            if new:
                g = 6 + 4 * age
                d.rectangle([X1 - g, Y1 - g, X2 + g, Y2 + g], outline=col + (int(230 * (1 - age / fps)),), width=1)
            marks.append((tr, (X1, Y1, X2, Y2), col))

        def hit(a, b):
            return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
        placed = []
        for k, (tr, (X1, Y1, X2, Y2), col) in enumerate(marks):
            others = [r for j, (_, r, _) in enumerate(marks) if j != k]
            idt = f"#{tr['id']}"
            rgt = f"  {tr['range']:.0f} m" if tr["range"] else ""
            done = False
            for rg in (rgt, ""):
                wi, wr = d.textlength(idt, font=self.f_tag), d.textlength(rg, font=self.f_tag)
                tw = wi + wr + 12
                for lx, ly in ((X2 + 6, Y1 - 1), (X1 - tw - 6, Y1 - 1), (X1, Y1 - 27), (X1, Y2 + 5), (X2 + 6, Y2 - 21), (X1 - tw - 6, Y2 - 21)):
                    lx = min(max(lx, 4), VW - tw - 4)
                    rb = (lx, ly, lx + tw, ly + 22)
                    if ly > 48 and ly + 22 < VH - bottom and not any(hit(rb, q) for q in placed + others):
                        placed.append(rb)
                        d.rounded_rectangle(list(rb), radius=3, fill=(8, 10, 12, 175))
                        d.text((lx + 6, ly + 2), idt, font=self.f_tag, fill=col + (255,))
                        if rg:
                            d.text((lx + 6 + wi, ly + 2), rg, font=self.f_tag, fill=self.WHITE + (235,))
                        done = True
                        break
                if done:
                    break

    def _card(self, base, db, y: int, ev: dict, look: str, status=None):
        """one alert card; look: new | live | dim | current | plain"""
        X0 = self.VW
        db.rounded_rectangle([X0 + 20, y, W_OUT - 20, y + 112], radius=4, fill={"new": self.CARD_NEW, "current": self.CARD_CUR}.get(look, self.CARD))
        chip = cv2.cvtColor(cv2.resize(ev["chip"], (96, 96), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
        base.paste(Image.fromarray((chip * 0.55).astype(np.uint8) if look == "dim" else chip), (X0 + 28, y + 8))
        db.rectangle([X0 + 27, y + 7, X0 + 124, y + 104], outline={"new": self.RED, "live": self.AMBER, "current": self.AMBER}.get(look, self.EDGE), width=1)
        tx = X0 + 142
        db.text((tx, y + 9), f"#{ev['id']}", font=self.f_id, fill=self.TXT)
        if ev["range_m"] is not None:
            db.text((tx, y + 46), f"{ev['range_m']} m away", font=self.f_dist, fill=self.TXT)
        db.text((tx, y + 80), fmt_t(ev["t"]), font=self.f_time, fill=self.TXT2)
        if status:
            st, sc = status
            sx_ = W_OUT - 34 - db.textlength(st, font=self.f_status)
            db.ellipse([sx_ - 13, y + 18, sx_ - 5, y + 26], fill=sc)
            db.text((sx_, y + 13), st, font=self.f_status, fill=sc)

    @staticmethod
    def _finish(base, ov) -> np.ndarray:
        return cv2.cvtColor(np.asarray(Image.alpha_composite(base.convert("RGBA"), ov).convert("RGB")), cv2.COLOR_RGB2BGR)

    def render(self, frame: np.ndarray, tracks: list[dict], w: ThermalWatch, sub: float = 0.0) -> np.ndarray:
        """sub in [0, 1): time inside the camera frame, so clock, blink and pulse can move at a higher output rate while the
        thermal image and its boxes stay at the camera's rate"""
        t, fps, X0 = w.t + sub, self.fps, self.VW
        shown = [z for z in tracks if z["id"] is not None] + w.held()
        vis = {z["raw"] for z in shown}
        base, db = self._base(frame)
        n = len(shown)
        db.text((X0 + 26, 70), str(n), font=self.f_count, fill=self.AMBER if n else self.TXT3)
        nx = X0 + 30 + db.textlength(str(n), font=self.f_count) + 14
        db.text((nx, 92), "person in view" if n == 1 else "persons in view", font=self.f_label, fill=self.TXT)
        rngs = [z["range"] for z in shown if z["range"]]
        if rngs:
            db.text((nx, 115), f"nearest {min(rngs):.0f} m away", font=self.f_label, fill=self.TXT2)
        db.text((X0 + 28, 176), "Alerts", font=self.f_head, fill=self.TXT)
        if w.events:
            c = str(len(w.events))
            db.text((W_OUT - 28 - db.textlength(c, font=self.f_label), 178), c, font=self.f_label, fill=self.TXT2)
        else:
            db.text((X0 + 28, 214), "No alerts yet", font=self.f_label, fill=self.TXT3)
        by_id = {v: k for k, v in w.disp.items()}
        y = 210
        for ev in w.events[::-1][:4]:
            live, new = by_id.get(ev["id"]) in vis, t - ev["frame"] < fps
            self._card(base, db, y, ev, "new" if new else ("live" if live else "dim"),
                       ("new", self.RED) if new else (("in view", self.AMBER) if live else ("out of view", self.TXT3)))
            y += 122
        ov, d = self._overlay()
        rec = f"REC  {fmt_t(t / fps)}"
        rw = d.textlength(rec, font=self.f_osd)
        if (t // max(1, int(fps / 2))) % 2 == 0:
            d.ellipse([self.VW - 24 - rw - 18, 25, self.VW - 24 - rw - 8, 35], fill=self.RED + (255,))
        d.text((self.VW - 24 - rw, 20), rec, font=self.f_osd, fill=self.WHITE + (235,))
        self._marks(d, frame.shape[1], shown, lambda tr: t - w.confirm_frame[tr["raw"]])
        return self._finish(base, ov)


class Recording:
    """a flight analysed once, start to end, for review: JPEG frames, the persons shown in each frame (confirmed + coasting) and
    the alerts; the same causal pipeline as the live screen, so the review shows exactly the live alerts"""

    def __init__(self, video: str, detector, thr: float, start: float = 0.0, dur: float = 0.0, progress: bool = False):
        meta = video_meta(video)
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 7.5
        n_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        f0 = int(round(start * fps))
        f1 = n_all if not dur else min(n_all, f0 + int(round(dur * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        w = ThermalWatch(detector, thr, meta, fps)
        self.jpgs, self.shown = [], []
        for k in range(f1 - f0):
            ok, frame = cap.read()
            if not ok:
                break
            tracks = w.step(frame)
            self.shown.append([dict(id=z["id"], raw=z["raw"], box=z["box"], range=z["range"]) for z in tracks if z["id"] is not None] + w.held())
            self.jpgs.append(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
            if progress and k % 75 == 0:
                print(f"analysing {meta['stem']}: {100 * k / max(1, f1 - f0):.0f} %", flush=True)
        self.events, self.meta, self.fps, self.name = w.events, meta, fps, meta["stem"]

    def frame(self, i: int) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(self.jpgs[i], np.uint8), cv2.IMREAD_COLOR)


class ReviewScreen(Console):
    """post-flight review: every alert already marked on the playback bar and listed in time order; the card of the alert the
    playhead is in is highlighted ('now'); the top right shows play / pause and the position"""
    N_CARDS, LIST_Y = 5, 112

    def __init__(self, rec: Recording):
        super().__init__(rec.meta, len(rec.jpgs), rec.fps)
        self.rec, self.n = rec, len(rec.jpgs)
        self.ev_frame = {e["id"]: e["frame"] for e in rec.events}

    def bar(self):
        return 24, self.VW - 24, self.VH - 24  # x0, x1, y of the playback bar

    def x_of(self, i: float) -> float:
        a0, a1, _ = self.bar()
        return a0 + (a1 - a0) * i / max(1, self.n - 1)

    def frame_at_x(self, x: float) -> int:
        a0, a1, _ = self.bar()
        return int(round(min(max((x - a0) / (a1 - a0), 0.0), 1.0) * (self.n - 1)))

    def current(self, i: int, sel: int | None = None) -> int | None:
        """the alert the playhead is in: the selected one until the next later alert, else the last alert at or before i"""
        evs = self.rec.events
        if sel is not None and evs[sel]["frame"] <= i:
            later = [e["frame"] for e in evs[sel + 1:] if e["frame"] > evs[sel]["frame"]]
            if not later or i < later[0]:
                return sel
        k = None
        for j, e in enumerate(evs):
            if e["frame"] <= i:
                k = j
        return k

    def offset_for(self, cur: int | None) -> int:
        return int(min(max((cur or 0) - 2, 0), max(0, len(self.rec.events) - self.N_CARDS)))

    def card_rects(self, offset: int):
        return [(k, (self.VW + 20, self.LIST_Y + j * 122, W_OUT - 20, self.LIST_Y + j * 122 + 112))
                for j, k in enumerate(range(offset, min(len(self.rec.events), offset + self.N_CARDS)))]

    def render(self, i: int, playing: bool = False, offset: int = 0, sel: int | None = None, cursor=None, click_age=None, sub: float = 0.0) -> np.ndarray:
        rec, fps, X0, VW, VH = self.rec, self.fps, self.VW, self.VW, self.VH
        frame = rec.frame(i)
        dur = (self.n - 1) / fps
        cur = self.current(i, sel)
        tt = min(i + sub, self.n - 1)
        base, db = self._base(frame)
        db.text((X0 + 28, 60), f"Flight review  ·  {rec.name}", font=self.f_label, fill=self.TXT2)
        db.text((X0 + 28, 82), f"{len(rec.events)} alerts in {fmt_t(dur)}", font=self.f_label, fill=self.TXT2)
        for k, (_x1, y1, _x2, _y2) in self.card_rects(offset):
            self._card(base, db, y1, rec.events[k], "current" if k == cur else "plain", ("now", self.AMBER) if k == cur else None)
        if not rec.events:
            db.text((X0 + 28, self.LIST_Y + 4), "No alerts in this flight", font=self.f_label, fill=self.TXT3)
        ov, d = self._overlay()
        for r in range(60):
            d.line([(0, VH - 1 - r), (VW - 1, VH - 1 - r)], fill=(0, 0, 0, int(130 * (1 - r / 60) ** 1.6)))
        pos = f"{fmt_t(tt / fps)} / {fmt_t(dur)}"
        px = VW - 24 - d.textlength(pos, font=self.f_osd)
        ix = px - 22
        d.text((px, 20), pos, font=self.f_osd, fill=self.WHITE + (235,))
        if playing:
            d.polygon([(ix, 24), (ix, 37), (ix + 11, 30.5)], fill=self.WHITE + (235,))
        else:
            d.rectangle([ix, 24, ix + 3, 37], fill=self.WHITE + (235,))
            d.rectangle([ix + 7, 24, ix + 10, 37], fill=self.WHITE + (235,))
        a0, a1, yb = self.bar()
        xp = self.x_of(tt)
        d.line([(a0, yb), (a1, yb)], fill=(255, 255, 255, 80), width=3)
        d.line([(a0, yb), (xp, yb)], fill=(255, 255, 255, 220), width=3)
        for k, e in enumerate(rec.events):
            xe, h = self.x_of(e["frame"]), (9 if k == cur else 7)
            d.rounded_rectangle([xe - 2, yb - h, xe + 2, yb + h], radius=1, fill=(self.WHITE if k == cur else self.AMBER) + (255,))
        d.ellipse([xp - 6, yb - 6, xp + 6, yb + 6], fill=self.WHITE + (255,))
        self._marks(d, frame.shape[1], rec.shown[i], lambda tr: tt - self.ev_frame.get(tr["id"], -10 ** 6), bottom=44)
        if cursor is not None:  # pointer of the scripted demo (demo.py); the interactive window shows the system pointer
            cx, cy = cursor
            if click_age is not None and click_age < 4:
                r = 8 + 5 * click_age
                d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 255, int(220 * (1 - click_age / 4))), width=2)
            pts = [(cx, cy), (cx, cy + 19), (cx + 5, cy + 14), (cx + 9, cy + 22), (cx + 12, cy + 21), (cx + 8, cy + 13), (cx + 14, cy + 13)]
            d.polygon(pts, fill=(255, 255, 255, 255), outline=(20, 20, 20, 255))
        return self._finish(base, ov)


def event_index(events: list[dict], title: str, path: str, cols: int = 5, cell: int = 150) -> None:
    """every alert as a time-stamped snapshot (find the moments without replaying the recording)"""
    f_title, f_label, f_mono = _font(["bahnschrift.ttf"], 21, "Bold"), _font(["bahnschrift.ttf"], 13), _font(["consola.ttf", "DejaVuSansMono.ttf"], 15)
    rows = max(1, math.ceil(len(events) / cols))
    im = Image.new("RGB", (24 + cols * (cell + 16), 70 + rows * (cell + 30) + 10), Console.BG)
    d = ImageDraw.Draw(im)
    d.text((24, 16), "EVENT INDEX", font=f_title, fill=Console.TXT)
    d.text((24, 44), title, font=f_label, fill=Console.TXT2)
    for i, ev in enumerate(events):
        x, y = 24 + (i % cols) * (cell + 16), 70 + (i // cols) * (cell + 30)
        im.paste(Image.fromarray(cv2.cvtColor(cv2.resize(ev["chip"], (cell, cell), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)), (x, y))
        d.rectangle([x - 1, y - 1, x + cell, y + cell], outline=Console.AMBER, width=1)
        d.text((x, y + cell + 4), f"{fmt_t(ev['t'])}   #{ev['id']}" + (f"   {ev['range_m']} m" if ev["range_m"] is not None else ""), font=f_mono, fill=Console.TXT)
    if not events:
        d.text((24, 80), "no alerts", font=f_label, fill=Console.TXT2)
    im.save(path)


def open_writer(path: str, fps: float, size) -> cv2.VideoWriter:
    """H.264 mp4: on Windows straight to Media Foundation (the FFmpeg backend of the opencv-python wheel first tries an OpenH264
    DLL that does not load); elsewhere the default backend, else mp4v"""
    if os.name == "nt":
        w = cv2.VideoWriter(path, cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*"avc1"), fps, size)
        if w.isOpened():
            return w
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"avc1"), fps, size)
    return w if w.isOpened() else cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)


# ------------------------------------------------------------------------------------------------------------------ runs
def resolve_model(video: str, weights: str | None, family: str | None, hw, thr: float | None):
    """-> (Detector, threshold); weights default: the S3x release file of the fold whose test split holds the video's recording"""
    if weights is None:
        fold = video_meta(video).get("test_fold")
        weights = release_for(fold)
        if weights is None:
            raise SystemExit("no default weights for this video: give --weights (and --family, --hw, --thr for a non-release checkpoint)")
    if not os.path.isfile(weights):
        raise SystemExit(f"{weights} not found: download the release files into release_assets/ (README.md, 'Pretrained weights')")
    thr = thr if thr is not None else threshold_for(weights)
    if thr is None:
        raise SystemExit(f"no person threshold known for {weights}: give --thr")
    return Detector(weights, family, hw), thr


def process(video: str, detector, thr: float, start: float = 0.0, dur: float = 0.0, out: str | None = None, sink=None,
            show: bool = False, write: bool = True, subframes: int = 1) -> tuple[dict, list[dict]]:
    """the live screen over a video (or a slice): writes console.mp4, events.jsonl, chips/, event_index.png and summary.json;
    sink(frame) receives every screen frame; subframes > 1 draws each camera frame that many times with clock and pulse advanced"""
    meta = video_meta(video)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 7.5
    n_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f0 = int(round(start * fps))
    f1 = n_all if not dur else min(n_all, f0 + int(round(dur * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    w = ThermalWatch(detector, thr, meta, fps)
    con = Console(meta, f1 - f0, fps)
    od = out or paths.out("results", "thermal_watch", *(["live"] if show else []),
                          f"{meta['stem']}__{getattr(detector, 'name', 'model')}" + (f"__{start:g}s" if start or dur else ""))
    if write:
        os.makedirs(os.path.join(od, "chips"), exist_ok=True)
    vw = open_writer(os.path.join(od, "console.mp4"), fps * subframes, (W_OUT, H_OUT)) if write else None
    n_conf = []
    for _ in range(f1 - f0):
        ok, frame = cap.read()
        if not ok:
            break
        tracks = w.step(frame)
        fr = None
        for k in range(subframes):
            fk = con.render(frame, tracks, w, sub=k / subframes)
            fr = fk if k == 0 else fr
            if vw is not None:
                vw.write(fk)
            if sink is not None:
                sink(fk)
        n_conf.append(sum(z["id"] is not None for z in tracks))
        if show:  # space = pause / resume, q or Esc = quit
            cv2.imshow("Thermal Watch", fr)
            key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
            while key == ord(" "):
                key = cv2.waitKey(0) & 0xFF
                key = -1 if key == ord(" ") else (key if key in (27, ord("q")) else ord(" "))
            if key in (27, ord("q")):
                break
    if vw is not None:
        vw.release()
    if show:
        cv2.destroyAllWindows()
    summ = dict(video=os.path.basename(video), weights=getattr(detector, "name", None), thr=thr, fps=fps, frames=len(n_conf), start_frame=f0,
                recording=meta.get("rec"), alerts=len(w.events), relinked=w.relinked,
                mean_confirmed=float(np.mean(n_conf)) if n_conf else 0.0, out=od if write else None)
    if write:
        for ev in w.events:
            cv2.imwrite(os.path.join(od, "chips", f"{ev['frame']:05d}_id{ev['id']}.png"), ev["chip"])
        with open(os.path.join(od, "events.jsonl"), "w", encoding="utf-8") as fo:
            for ev in w.events:
                fo.write(json.dumps({k: v for k, v in ev.items() if k != "chip"}) + "\n")
        event_index(w.events, f"{meta['stem']}  ·  {len(w.events)} alerts in {fmt_t((f1 - f0) / fps)}", os.path.join(od, "event_index.png"))
        json.dump(summ, open(os.path.join(od, "summary.json"), "w"), indent=1)
    return summ, w.events


def review(video: str, detector, thr: float, start: float = 0.0, dur: float = 0.0) -> None:
    """interactive review window. Mouse: click an alert card or a mark on the bar to jump there (elsewhere on the bar = seek),
    wheel over the list to scroll. Keys: space play / pause, left / right (or a / d) 1 s, n / p next / previous alert, q quit."""
    rec = Recording(video, detector, thr, start, dur, progress=True)
    scr = ReviewScreen(rec)
    n, fps = scr.n, rec.fps
    st = dict(i=0, playing=False, offset=0, sel=None, follow=True)
    win = "Thermal Watch - review"

    def jump(k):
        st.update(i=rec.events[k]["frame"], sel=k, follow=True, playing=True)

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            for k, (x1, y1, x2, y2) in scr.card_rects(st["offset"]):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    jump(k)
                    return
            a0, a1, yb = scr.bar()
            if a0 - 8 <= x <= a1 + 8 and abs(y - yb) <= 14:
                near = [k for k, e in enumerate(rec.events) if abs(scr.x_of(e["frame"]) - x) <= 6]
                if near:
                    jump(near[0])
                else:
                    st.update(i=scr.frame_at_x(x), sel=None, follow=True)
        elif ev == cv2.EVENT_MOUSEWHEEL and x >= scr.VW:
            st["offset"] = int(min(max(st["offset"] + (-1 if cv2.getMouseWheelDelta(flags) > 0 else 1), 0), max(0, len(rec.events) - scr.N_CARDS)))
            st["follow"] = False

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    print("review: click a card or a bar mark to jump | space play/pause | arrows 1 s | n/p next/previous alert | q quit", flush=True)
    while True:
        if st["follow"]:
            st["offset"] = scr.offset_for(scr.current(st["i"], st["sel"]))
        cv2.imshow(win, scr.render(st["i"], st["playing"], st["offset"], st["sel"]))
        key = cv2.waitKeyEx(max(1, int(1000 / fps)) if st["playing"] else 30)
        if key in (27, ord("q")) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord(" "):
            st["playing"] = not st["playing"]
        elif key in (2424832, ord("a")):
            st.update(i=max(0, st["i"] - int(round(fps))), sel=None, follow=True)
        elif key in (2555904, ord("d")):
            st.update(i=min(n - 1, st["i"] + int(round(fps))), sel=None, follow=True)
        elif key == ord("n"):
            nxt = [k for k, e in enumerate(rec.events) if e["frame"] > st["i"]]
            if nxt:
                jump(nxt[0])
        elif key == ord("p"):
            prv = [k for k, e in enumerate(rec.events) if e["frame"] < st["i"]]
            if prv:
                jump(prv[-1])
        if st["playing"]:
            if st["i"] >= n - 1:
                st["playing"] = False
            else:
                st["i"] += 1
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Thermal Watch: person search in thermal drone video")
    ap.add_argument("video")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--show", action="store_true", help="live window (space pause, q quit); outputs go to .../thermal_watch/live/")
    g.add_argument("--review", action="store_true", help="post-flight review window (every alert on the playback bar and in the list)")
    ap.add_argument("--weights", default=None, help="release file or training checkpoint (default: see the module docstring)")
    ap.add_argument("--family", default=None, help="model family of a non-release checkpoint, e.g. yolov8n_s3x")
    ap.add_argument("--hw", type=int, nargs=2, default=None, metavar=("H", "W"), help="canvas of a non-release checkpoint")
    ap.add_argument("--thr", type=float, default=None, help="person threshold (default: the release file's validation max-F1 threshold)")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    det, thr = resolve_model(a.video, a.weights, a.family, a.hw, a.thr)
    print(f"model: {det.name} ({det.family}, canvas {det.hw[1]} x {det.hw[0]}), person threshold {thr:.3f}", flush=True)
    if a.review:
        review(a.video, det, thr, a.start, a.dur)
        return
    summ, _ = process(a.video, det, thr, a.start, a.dur, out=a.out, show=a.show)
    print("THERMAL_WATCH_DONE", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
