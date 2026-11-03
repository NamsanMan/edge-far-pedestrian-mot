# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Timing of the app: REPEATS full runs of zone_alert.py per sequence and model (repetitions interleaved), reported as
the median with the min-max range. Each run includes JPEG decode, drawing and H.264 encoding; the pipeline FPS counts
only detect + track + alert logic. The first 10 frames of every run are excluded.

usage (repository root): python -B apps/zone_alert/bench.py
writes apps/zone_alert/timing.json and timing.md; the annotated videos and events go to out/ (identical in every run)

    python -B apps/zone_alert/bench.py --v2
runs every (sequence, model) with the v1 zone files and with zones/<seq>_v2.json (APPROACH, direction filter,
tripwire), all repetitions interleaved in one session, and writes timing_v2.json / timing_v2.md (timing.json and
timing.md are left as they are); videos and events go to out/v2bench/.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zone_alert as za  # noqa: E402
import compare  # noqa: E402
from efpmot.data import mot17 as md  # noqa: E402

REPEATS = 3
STAGES = ("read", "detect", "track", "alert", "draw+write")


def app_args(seq: str, model: str, thr: float, zones: str = "", out_dir: str = "out") -> list[str]:
    """Command-line arguments of the README run for one sequence and model (zones: '' = v1 file, '_v2')."""
    family, run_dir, _ = compare.MODELS[model]
    frames = md.eval_frames(seq)
    return ["--frames", os.path.join(md.img_root(), seq, "img1"), "--start", str(frames[0]), "--stop", str(frames[-1]),
            "--family", family, "--weights", compare.weights_for(run_dir),
            "--zones", os.path.join(HERE, "zones", seq[:8] + zones + ".json"), "--thr", f"{thr:g}",
            "--out", os.path.join(HERE, out_dir, f"{seq[:8]}_{model}{zones}.mp4")]


def spread(values: list[float], digits: int = 1) -> dict:
    """Median, min, max and the single runs."""
    return dict(median=round(statistics.median(values), digits), min=round(min(values), digits),
                max=round(max(values), digits), runs=[round(v, digits) for v in values])


def write_md(timing: dict, path: str) -> None:
    """timing.md: one row per sequence and model (and zone file in --v2 mode), medians with the min-max range."""
    v2 = "zones" in timing["rows"][0]
    lines = ["# Zone alert app: timing" + (" (v1 vs v2 zone files)" if v2 else ""), "",
             f"`bench.py{' --v2' if v2 else ''}`, {timing['repeats']} runs per row (interleaved), GPU {timing['gpu']}. "
             "Median (min-max) over the runs; first 10 frames of each run excluded. Pipeline = detect + track + alert "
             "logic; end-to-end adds JPEG decode, drawing and H.264 encoding."
             + (" v2 = APPROACH on every zone, direction filters and one tripwire (zones/<seq>_v2.json)." if v2 else ""),
             "",
             "| sequence | model |" + (" zones |" if v2 else "") + " detect [ms] | track [ms] | alert logic [ms] | "
             "pipeline FPS | end-to-end FPS |",
             "|---|---|" + ("---|" if v2 else "") + "---|---|---|---|---|"]
    for row in timing["rows"]:
        ms = row["ms_median"]
        pipe, e2e = row["pipeline_fps"], row["end_to_end_fps"]
        zones = f" {row['zones']} |" if v2 else ""
        lines.append(f"| {row['seq'][:8]} | {row['model']} |{zones} {ms['detect']:.1f} | {ms['track']:.1f} | "
                     f"{ms['alert']:.2f} | {pipe['median']:.0f} ({pipe['min']:.0f}-{pipe['max']:.0f}) | "
                     f"{e2e['median']:.0f} ({e2e['min']:.0f}-{e2e['max']:.0f}) |")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", action="store_true", help="v1 vs v2 zone files, interleaved -> timing_v2.json / .md")
    v2 = ap.parse_args().v2
    thr = {m: compare.thresholds(v[2]) for m, v in compare.MODELS.items()}
    variants = ("", "_v2") if v2 else ("",)
    out_dir = os.path.join("out", "v2bench") if v2 else "out"
    configs = [(seq, model, zones) for seq in compare.SEQS for model in compare.MODELS for zones in variants]
    reports = {c: [] for c in configs}
    for rep in range(REPEATS):
        for seq, model, zones in configs:
            print(f"--- run {rep + 1}/{REPEATS}: {seq} {model} {zones or 'v1'}", flush=True)
            args = za.parse_args(app_args(seq, model, thr[model][seq], zones, out_dir))
            reports[(seq, model, zones)].append(za.run(args))
    rows = []
    for (seq, model, zones), runs in reports.items():
        ms = {k: round(statistics.median(r["ms_per_frame"][k] for r in runs), 3) for k in STAGES}
        rows.append(dict(seq=seq, model=model, thr=thr[model][seq], ms_median=ms,
                         pipeline_fps=spread([r["pipeline_fps"] for r in runs]),
                         end_to_end_fps=spread([r["end_to_end_fps"] for r in runs]),
                         events=runs[-1]["events"], codec=runs[-1]["codec"]))
        if v2:
            rows[-1]["zones"] = zones.strip("_") or "v1"
    timing = dict(repeats=REPEATS, gpu=reports[configs[0]][0]["gpu"], rows=rows)
    name = "timing_v2" if v2 else "timing"
    with open(os.path.join(HERE, name + ".json"), "w", encoding="utf-8") as fh:
        json.dump(timing, fh, indent=1)
    write_md(timing, os.path.join(HERE, name + ".md"))
    with open(os.path.join(HERE, name + ".md"), encoding="utf-8") as fh:
        print(fh.read())


if __name__ == "__main__":
    main()
