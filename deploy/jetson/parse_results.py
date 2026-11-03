# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""bench.log (bench.sh) -> latency table: results/latency_orin_nano.csv and results/latency_orin_nano.md.

This is the only source of the Jetson numbers in the repository: results/RESULTS.md (efpmot.eval.table) and the docs
read or quote results/latency_orin_nano.csv. Convention: canvas sizes are W x H; the reported statistic is the median of
trtexec's GPU compute time (the CSV also keeps min / max / mean).

Each RESULT line gives the trtexec GPU compute time statistics of one (model, precision) run. A (model, precision) that
appears again later in the log is a repeated pass (the drift check of bench.sh -r) and gets pass = 2, 3, ...
Models listed in MODELS get their description; OTHER holds the designs measured in the same session whose code is not in
this repository (shown in a separate table); unknown names are reported as-is.

usage (repository root):
    python -B deploy/jetson/parse_results.py [deploy/jetson/results/bench_2026-10-27.log] [--out deploy/jetson/results] [--trt 10.3]
"""
from __future__ import annotations

import argparse
import csv
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
LINE = re.compile(r"RESULT (\S+)\.onnx (\w+) GPU Compute Time: min = ([\d.]+) ms, max = ([\d.]+) ms, "
                  r"mean = ([\d.]+) ms, median = ([\d.]+) ms")
POWER = re.compile(r"power mode: (.*)$")
# graph name -> (model, trained on, canvas W x H, input)
MODELS = {
    "v8n_416_uniform_canvas": ("YOLOv8n", "DSEC", "416 x 320", "canvas"),
    "v8n_416_s3x_canvas": ("YOLOv8n + S3x", "DSEC", "416 x 320", "canvas"),
    "v8n_416_s3y_canvas": ("YOLOv8n + S3y", "DSEC", "416 x 320", "canvas"),
    "v8n_448x352_uniform_canvas": ("YOLOv8n", "DSEC", "448 x 352", "canvas"),
    "v8n_640_uniform_canvas": ("YOLOv8n", "DSEC", "640 x 480", "canvas"),
    "v8n_640_uniform_native": ("YOLOv8n", "DSEC", "640 x 480", "native 640 x 480"),
    "bdd_416_uniform_canvas": ("YOLOv8n", "BDD100K", "416 x 256", "canvas"),
    "bdd_416_s3x_canvas": ("YOLOv8n + S3x", "BDD100K", "416 x 256", "canvas"),
    "bdd_416_uniform_native": ("YOLOv8n", "BDD100K", "416 x 256", "native 1280 x 720"),
    "bdd_640_uniform_canvas": ("YOLOv8n", "BDD100K", "640 x 384", "canvas"),
    "bdd_640_uniform_native": ("YOLOv8n", "BDD100K", "640 x 384", "native 1280 x 720"),
}
OTHER = {
    "v8n_416_p2_canvas": ("YOLOv8n + P2 level (stride 4 head)", "416 x 320"),
    "v8n_416_mxxr_canvas": ("YOLOv8n + shallow x-fine P3 branch (MatrixNets-style, FLOPs matched to S3x)", "416 x 320"),
    "v8n_416_casp_canvas": ("YOLOv8n + two half-width P3 branches (x-fine and y-fine)", "416 x 320"),
}
FIELDS = ["graph", "model", "trained_on", "canvas_wxh", "input", "precision", "pass", "min_ms", "max_ms", "mean_ms",
          "median_ms"]


def parse(log: str) -> tuple[list[dict], str]:
    """RESULT rows of a bench.log and the power-mode line."""
    rows, seen, power = [], {}, ""
    with open(log, encoding="utf-8") as fh:
        for line in fh:
            m = POWER.search(line)
            if m and not power:
                power = m.group(1).strip()
            m = LINE.search(line)
            if not m:
                continue
            graph, prec = m.group(1), m.group(2)
            seen[(graph, prec)] = seen.get((graph, prec), 0) + 1
            if graph in MODELS:
                model, data, canvas, inp = MODELS[graph]
            else:                              # other designs (DSEC-trained) or an unknown graph
                model, canvas = OTHER.get(graph, (graph, "?"))
                data, inp = "DSEC", "canvas"
            rows.append(dict(graph=graph, model=model, trained_on=data, canvas_wxh=canvas, input=inp, precision=prec,
                             **{"pass": seen[(graph, prec)]},
                             **{k: float(v) for k, v in zip(("min_ms", "max_ms", "mean_ms", "median_ms"),
                                                            m.groups()[2:])}))
    return rows, power


def markdown(rows: list[dict], power: str, log: str, trt: str) -> str:
    """The latency tables (main models, drift check, other designs) with the measurement caveats."""
    def ms(r: dict) -> str:
        return f"{r['median_ms']:.2f}"

    main = [r for r in rows if r["graph"] in MODELS]
    other = [r for r in rows if r["graph"] in OTHER]
    first = {(r["graph"], r["precision"]): r for r in main if r["pass"] == 1}
    out = ["# TensorRT latency on a Jetson Orin Nano", "",
           f"Device: Jetson Orin Nano, power mode `{power or 'not recorded'}`, TensorRT {trt} (`trtexec --warmUp=500 "
           f"--duration=10 --useSpinWait`). Source: `{os.path.basename(log)}`, parsed by `parse_results.py`.", "",
           "**What is timed:** the GPU compute time of the detector engine alone (trtexec), batch 1, raw head output. "
           "Pre-processing on the CPU, NMS, tracking and the zone logic are not included. `native` graphs take the "
           "camera frame and do the letterbox resize inside the engine.", "",
           "**INT8:** the engines were built with `--int8 --fp16` and no calibration set, so the INT8 rows are "
           "latency only. No accuracy is claimed for them.", "",
           "| model | trained on | canvas (W x H) | input | precision | median [ms] |",
           "|---|---|---|---|---|---|"]
    for r in main:
        if r["pass"] == 1:
            out.append(f"| {r['model']} | {r['trained_on']} | {r['canvas_wxh']} | {r['input']} | "
                       f"{r['precision'].upper()} | {ms(r)} |")
    rep = [r for r in main if r["pass"] > 1]
    if rep:
        out += ["", "Drift check: FP16 graphs timed again at the end of the session, interleaved.", "",
                "| model | canvas | input | first pass median [ms] | repeat median [ms] |", "|---|---|---|---|---|"]
        for r in rep:
            f = first[(r["graph"], r["precision"])]
            out.append(f"| {r['model']} | {r['canvas_wxh']} | {r['input']} | {f['median_ms']:.2f} | "
                       f"{r['median_ms']:.2f} |")
    plain, s3x = first.get(("v8n_416_uniform_canvas", "fp16")), first.get(("v8n_416_s3x_canvas", "fp16"))
    if plain and s3x:
        p8, s8 = first.get(("v8n_416_uniform_canvas", "int8")), first.get(("v8n_416_s3x_canvas", "int8"))
        line = (f"Headline (416 x 320 canvas, FP16, median): YOLOv8n {plain['median_ms']:.2f} ms, YOLOv8n + S3x "
                f"{s3x['median_ms']:.2f} ms (+{100 * (s3x['median_ms'] / plain['median_ms'] - 1):.0f} %, close to its "
                "+11 % FLOPs)")
        if p8 and s8:
            line += f"; INT8 {p8['median_ms']:.2f} vs {s8['median_ms']:.2f} ms"
        out += ["", line + "."]
    if other:
        out += ["", "Other P3 designs timed in the same session (their code is not part of this repository; see "
                "`docs/EXPERIMENTS_LOG.md`):", "", "| design | canvas (W x H) | precision | median [ms] |",
                "|---|---|---|---|"]
        for r in other:
            out.append(f"| {r['model']} | {r['canvas_wxh']} | {r['precision'].upper()} | {ms(r)} |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log", nargs="?", default=os.path.join(HERE, "results", "bench_2026-10-27.log"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--trt", default="10.3", help="TensorRT version printed in the trtexec logs")
    a = ap.parse_args()
    rows, power = parse(a.log)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "latency_orin_nano.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, "latency_orin_nano.md"), "w", encoding="utf-8") as fh:
        fh.write(markdown(rows, power, a.log, a.trt))
    print(f"{len(rows)} rows -> {a.out}/latency_orin_nano.csv, .md")


if __name__ == "__main__":
    main()
