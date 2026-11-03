# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Side-by-side demo of the zone alerts: left plain YOLOv8n, right YOLOv8n + S3x (ByteTrack, seed 0, cross-fitted
thresholds).

Window rule (fixed before anything was rendered; uses ground truth only, never which model wins):
    candidates = every WIN_S = 7 s window (stride 1 s) inside the evaluation halves of MOT17-04 and MOT17-09;
    score      = number of scored GT far-zone episodes confirmed inside the window that start after the first
                 evaluation frame (walk-ins, or people whose annotation starts inside the zone; not people already
                 standing in the zone when the clip starts);
    choose     = highest score; tie -> earliest (MOT17-04 before MOT17-09, then the earlier start).
The rule was re-applied unchanged after the GT episode definition in compare.py was corrected.

Rendering: every 3rd source frame (30 -> 10 fps), panels 640 px wide (1/3 of 1920), output width 1280.
Per panel: zones, tracks (red = inside a zone), alert banner with the clip time, the last alerts, and the counter
'intrusions alerted k / n (GT)' over the scored GT episodes confirmed in the window so far (k = those whose attributed
INTRUSION alert, as in compare.py, was raised by now). White ring = a person inside a zone (GT) whose entry this model
has not alerted yet.

Inputs: cache/<seq>__<model>.json and results.json written by compare.py (run it first).
Outputs: demo/zone_alert_compare.gif (<= 8 MB) and demo/zone_alert_compare.mp4, demo/render_log.json
"""
from __future__ import annotations

import json
import os
import sys
from typing import Iterator

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zone_alert as za  # noqa: E402
import compare  # noqa: E402
from efpmot.data import mot17 as md  # noqa: E402

WIN_S = 7.0
STEP = 3                   # source frames per output frame (30 -> 10 fps)
PANEL_W = 640
GIF_MAX = 8 * 1024 * 1024
OUT = os.path.join(HERE, "demo")
TITLES = {"plain": "YOLOv8n (plain)", "s3x": "YOLOv8n + S3x (anisotropic P3)"}


def load_json(path: str):
    """Read a JSON file."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------------------------------------- window
def counts_as_entry(ep: dict, start: int, length: int) -> bool:
    """A scored far-zone episode that starts after the first frame and is confirmed inside [start, start + length)."""
    return (ep["scored"] and ep["zone_kind"] == "far" and not ep["initial"]
            and start <= ep["confirm"] < start + length)


def choose_window(results: dict) -> dict:
    """Apply the window rule of the module docstring to the GT episodes in results.json."""
    candidates = []
    for seq in compare.SEQS:
        fps = md.seqinfo(seq)["fps"]
        frames = md.eval_frames(seq)
        length = int(WIN_S * fps)
        for start in range(frames[0], frames[-1] - length + 2, fps):
            n = sum(counts_as_entry(ep, start, length) for ep in results["episodes"][seq])
            candidates.append(dict(seq=seq, start=start, end=start + length - 1, far_entries=n))
    best = max(c["far_entries"] for c in candidates)
    chosen = next(c for c in candidates if c["far_entries"] == best)    # earliest of the best
    return dict(chosen, n_candidates=len(candidates),
                score_distribution=np.bincount([c["far_entries"] for c in candidates]).tolist())


# ---------------------------------------------------------------------------------------------------------- panels
def replay(run: dict, zones: list[za.Zone], frames: list[int],
           fps: float) -> Iterator[tuple[int, np.ndarray, za.ZoneMonitor]]:
    """Re-run the zone monitor on the cached tracks (identical events, asserted) and yield (frame, tracks, monitor)."""
    monitor = za.ZoneMonitor(zones, fps, frame0=frames[0])
    events = []
    for f in frames:
        tracks = np.asarray(run["tracks"][str(f)], np.float32).reshape(-1, 6)
        events += monitor.update(f, {int(r[4]): za.foot_point(r) for r in tracks})
        yield f, tracks, monitor
    replayed = [(e.type, e.zone, e.track_id, e.frame) for e in events]
    cached = [(e["type"], e["zone"], e["track_id"], e["frame"]) for e in run["events"]]
    assert replayed == cached, "replay differs from compare.py"


def alerted_by(hits: dict, episode: int, frame: int) -> bool:
    """True when the episode's attributed INTRUSION was raised at or before `frame`."""
    return hits.get(episode) is not None and hits[episode] <= frame


def panel(img: np.ndarray, zones: list[za.Zone], tracks: np.ndarray, monitor: za.ZoneMonitor, events: list[dict],
          episodes: list[dict], hits: dict, gt_feet: dict, frame: int, w0: int, fps: float, title: str) -> np.ndarray:
    """One model's panel at `frame`: title, alert banner, zones and tracks with GT rings, counter and last alerts.
    hits: episode index -> frame of its hit INTRUSION; w0: first frame of the window (clip time 0)."""
    s = PANEL_W / img.shape[1]
    vis = cv2.resize(img, (PANEL_W, int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    za.draw_zones(vis, zones, s)
    za.draw_tracks(vis, tracks, monitor, s, id_scale=0.33)
    for i, ep in enumerate(episodes):   # GT people inside a zone whose entry this model has not alerted (yet)
        active = ep["scored"] and ep["confirm"] <= frame <= ep["end"]
        if active and not alerted_by(hits, i, frame) and ep["gid"] in gt_feet.get(frame, {}):
            x, y = gt_feet[frame][ep["gid"]]
            cv2.circle(vis, (int(x * s), int(y * s)), 7, za.WHITE, 2, cv2.LINE_AA)
    clip = [za.Event(e["type"], e["zone"], e["track_id"], e["frame"], (e["frame"] - w0) / fps, e["entered"])
            for e in events if e["type"] != "EXIT" and w0 <= e["frame"] <= frame]
    recent = [e for e in clip if frame - e.frame < za.BANNER_S * fps]
    in_window = [i for i, ep in enumerate(episodes) if ep["scored"] and w0 <= ep["confirm"] <= frame]
    alerted = {k: sum(alerted_by(hits, i, frame) for i in in_window if episodes[i]["zone_kind"] == k)
               for k in ("far", "near")}
    total = {k: sum(episodes[i]["zone_kind"] == k for i in in_window) for k in ("far", "near")}
    header = (f"intrusions alerted {alerted['far'] + alerted['near']} / {total['far'] + total['near']} (GT)   "
              f"far {alerted['far']}/{total['far']}  near {alerted['near']}/{total['near']}")
    title_bar = np.full((28, PANEL_W, 3), (60, 45, 30), np.uint8)
    za.text(title_bar, title, (8, 20), 0.55, za.WHITE, th=2)
    return np.vstack([title_bar, za.status_bar(PANEL_W, recent, zones, scale=0.45), vis,
                      za.event_strip(PANEL_W, clip, n=3, scale=0.42, header=header)])


def compose(panels: list[np.ndarray], seq: str, clip_s: float) -> np.ndarray:
    """Header, the two panels side by side, legend and attribution."""
    body = np.hstack(panels)
    head = np.full((30, body.shape[1], 3), 20, np.uint8)
    za.text(head, f"{seq[:8]} evaluation half  |  clip {clip_s:4.1f} s of {WIN_S:g} s  |  window rule: most GT "
                  "far-zone entries (fixed before rendering)", (8, 21), 0.5, za.WHITE)
    legend = np.full((24, body.shape[1], 3), 20, np.uint8)
    za.text(legend, "red box = track inside a zone (alert raised)   white ring = person in a zone (GT) not alerted "
                    "yet   ByteTrack, seed 0, same zones and logic for both", (8, 17), 0.45, (200, 200, 200))
    return za.pad_to(np.vstack([head, body, legend, za.attribution_bar(body.shape[1], 0.45)]))


# ---------------------------------------------------------------------------------------------------------- GIF
def write_gif(path: str, frames: list[np.ndarray], fps: float, delta: int = 10) -> int:
    """One global 255-colour palette and temporal delta coding (unchanged pixels -> transparent index 255)."""
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    pick = np.linspace(0, len(rgb) - 1, 10).astype(int)
    palette_img = Image.fromarray(np.vstack([rgb[i] for i in pick])).quantize(255, method=Image.Quantize.MEDIANCUT)
    palette = np.asarray(palette_img.getpalette()[:765], np.int16).reshape(-1, 3)
    palette = np.vstack([palette, np.zeros((256 - len(palette), 3), np.int16)])
    flat = palette.astype(np.uint8).ravel().tolist()
    shown, out = None, []
    for frame in rgb:
        idx = np.asarray(Image.fromarray(frame).quantize(palette=palette_img, dither=Image.Dither.NONE), np.uint8)
        if shown is None:
            shown = coded = idx
        else:
            changed = np.abs(palette[idx] - palette[shown]).max(-1) > delta
            coded = np.where(changed, idx, 255).astype(np.uint8)
            shown = np.where(changed, idx, shown)
        image = Image.fromarray(coded, "P")
        image.putpalette(flat)
        out.append(image)
    out[0].save(path, save_all=True, append_images=out[1:], duration=int(round(1000 / fps)), loop=0, disposal=1,
                transparency=255, optimize=False)
    return os.path.getsize(path)


# ---------------------------------------------------------------------------------------------------------- main
def main() -> None:
    results = load_json(os.path.join(HERE, "results.json"))
    window = choose_window(results)
    seq, w0, w1 = window["seq"], window["start"], window["end"]
    fps = float(md.seqinfo(seq)["fps"])
    frames = md.eval_frames(seq)
    zones = za.load_zones(os.path.join(HERE, "zones", seq[:8] + ".json"))
    episodes = results["episodes"][seq]
    gt = compare.read_gt(seq)
    gt_feet = {f: {gid: za.foot_point(box) for gid, box, _ in rows} for f, rows in gt.peds.items()}
    runs = {m: load_json(os.path.join(HERE, "cache", f"{seq[:8]}__{m}.json")) for m in compare.MODELS}
    hits = {m: {} for m in compare.MODELS}       # episode index -> frame of its hit INTRUSION
    for m in compare.MODELS:
        for alert in results["alerts"][seq][m]:
            if alert["type"] == "INTRUSION" and alert["outcome"] == "hit":
                hits[m][alert["episode"]] = alert["frame"]
    shown = set(range(w0, w1 + 1, STEP))
    streams = {m: replay(runs[m], zones, frames, fps) for m in compare.MODELS}
    out = []
    for f in frames:
        states = {m: next(streams[m]) for m in compare.MODELS}
        if f not in shown:
            continue
        img = cv2.imread(md.frame_path(seq, f))
        panels = [panel(img, zones, states[m][1], states[m][2], runs[m]["events"], episodes, hits[m], gt_feet, f, w0,
                        fps, f"{TITLES[m]}   thr {runs[m]['thr']:g}") for m in compare.MODELS]
        out.append(compose(panels, seq, (f - w0) / fps))
    for m in compare.MODELS:   # finish the replays so their assertion runs
        for _ in streams[m]:
            pass
    os.makedirs(OUT, exist_ok=True)
    mp4 = os.path.join(OUT, "zone_alert_compare.mp4")
    gif = os.path.join(OUT, "zone_alert_compare.gif")
    writer, codec = za.open_mp4(mp4, fps / STEP, (out[0].shape[1], out[0].shape[0]))
    for image in out:
        writer.write(image)
    writer.release()
    size = write_gif(gif, out, fps / STEP)
    assert size <= GIF_MAX, f"GIF {size / 2**20:.1f} MB > 8 MB"
    in_window = [i for i, ep in enumerate(episodes) if ep["scored"] and w0 <= ep["confirm"] <= w1]
    final = {m: {k: [sum(alerted_by(hits[m], i, w1) for i in in_window if episodes[i]["zone_kind"] == k),
                     sum(episodes[i]["zone_kind"] == k for i in in_window)] for k in ("far", "near")}
             for m in compare.MODELS}
    log = dict(window=window, window_s=[round((w0 - frames[0]) / fps, 2), round((w1 + 1 - frames[0]) / fps, 2)],
               image_frames=[w0, w1], out_frames=len(out), out_fps=fps / STEP,
               size_wh=[out[0].shape[1], out[0].shape[0]], gif_bytes=size, mp4_bytes=os.path.getsize(mp4),
               mp4_codec=codec, alerted_in_window=final)
    with open(os.path.join(OUT, "render_log.json"), "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=1)
    print(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
