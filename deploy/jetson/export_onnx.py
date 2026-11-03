# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Export trained detectors to ONNX for the TensorRT latency benchmark (run on the PC, CPU only).

Two graph variants per model, both ending in the raw head output before NMS (as the Ultralytics ONNX export):
  <name>_canvas.onnx  input `images` = the letterboxed detector canvas, BGR 0..255, float [1, 3, H, W]
  <name>_native.onnx  input `images` = the native camera frame, BGR 0..255, [1, 3, H0, W0]; the letterbox (bilinear
                      resize to fit, pad bottom / right with 114) runs inside the graph, so TensorRT does it on the GPU
Output `output0` = [1, 4 + 8, A]: box centre x, y, width, height in canvas pixels, then the 8 class scores (sigmoid), for
every anchor A of the three levels. Latency depends only on the graph, not on the trained weights.

Anchor grid. By default it is traced as a constant (the Detect head caches it after one forward pass), as in every
graph of the measurement session except the two plain DSEC canvas graphs, which computed the grid inside the graph
(PUBLIC column `graph_anchors`). Both forms give identical outputs; `--all` reproduces each benchmarked graph node for
node.

Every exported file is run once with onnxruntime (CPU) and compared with the PyTorch module on the same input; the script
prints max|ort-torch|.

usage (repository root):
    python -B deploy/jetson/export_onnx.py --weights release_assets/yolov8n_s3x_dsec_416x320_s0.pt --family yolov8n_s3x \
        --hw 320 416 --out deploy/jetson/export/v8n_416_s3x_canvas.onnx
    python -B deploy/jetson/export_onnx.py ... --native 480 640 --out .../v8n_416_s3x_native.onnx
    python -B deploy/jetson/export_onnx.py --all [--out_dir deploy/jetson/export]   (the models of results/)
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from efpmot import paths  # noqa: E402
from efpmot.models import build  # noqa: E402

DSEC_HW, BDD_HW = (480, 640), (720, 1280)
# name, run tag of the checkpoint, family, canvas (H, W), native frame (H, W) or None (canvas graph only),
# graph_anchors (canvas graph computes the anchor grid in-graph). The graphs of deploy/jetson/results/latency_orin_nano.md.
PUBLIC = [
    ("v8n_416_uniform", "dsec_v8n_416_s0", "yolov8n", (320, 416), None, True),
    ("v8n_416_s3x", "dsec_v8n_416s3x_s0", "yolov8n_s3x", (320, 416), None, False),
    ("v8n_416_s3y", "dsec_v8n_416s3y_s0", "yolov8n_s3y", (320, 416), None, False),
    ("v8n_448x352_uniform", "dsec_v8n_448x352_s0", "yolov8n", (352, 448), None, False),
    ("v8n_640_uniform", "dsec_v8n_640_s0", "yolov8n", (480, 640), DSEC_HW, True),
    ("bdd_416_uniform", "bdd_v8n_416_s0", "yolov8n", (256, 416), BDD_HW, False),
    ("bdd_416_s3x", "bdd_v8n_416s3x_s0", "yolov8n_s3x", (256, 416), None, False),
    ("bdd_640_uniform", "bdd_v8n_640_s0", "yolov8n", (384, 640), BDD_HW, False),
]


class Canvas(torch.nn.Module):
    """Canvas BGR 0..255 -> raw head output [1, 4 + C, A] (inference mode of the Ultralytics Detect head)."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.m = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.m(x[:, [2, 1, 0]] / 255.0)
        return y[0] if isinstance(y, (list, tuple)) else y


class Native(torch.nn.Module):
    """Native H0 x W0 BGR frame -> letterbox inside the graph (resize to fit, pad bottom / right with 114) -> Canvas."""

    def __init__(self, model: torch.nn.Module, hw: tuple[int, int], native: tuple[int, int]):
        super().__init__()
        self.det = Canvas(model)
        (h, w), (h0, w0) = hw, native
        r = min(h / h0, w / w0)
        self.resize = (int(round(h0 * r)), int(round(w0 * r)))
        self.hw = (h, w)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(img.float(), size=self.resize, mode="bilinear", align_corners=False)
        x = F.pad(x, (0, self.hw[1] - self.resize[1], 0, self.hw[0] - self.resize[0]), value=114.0)
        return self.det(x)


def export_model(model: torch.nn.Module, hw: tuple[int, int], native: tuple[int, int] | None, out: str,
                 graph_anchors: bool = False) -> float:
    """Export one graph (canvas, or native when `native` is given) to `out` (opset 17); return max|ort-torch|.
    graph_anchors: trace the anchor-grid computation into the graph instead of a cached constant (same outputs)."""
    import onnxruntime as ort
    mod = (Native(model, hw, native) if native else Canvas(model)).eval()
    shape = (1, 3, *(native or hw))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    head = model.model[-1]
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")                  # TorchScript-exporter deprecation and tracer notices
        if graph_anchors:
            head.shape = None                            # forces make_anchors inside the traced forward
        else:
            mod(torch.zeros(shape))                      # caches the anchor grid -> traced as a constant
        torch.onnx.export(mod, torch.zeros(shape), out, opset_version=17, input_names=["images"],
                          output_names=["output0"], dynamo=False)
        x = torch.rand(shape, generator=torch.Generator().manual_seed(0)) * 255.0   # check on a non-trivial input
        ref = mod(x).numpy()
    got = ort.InferenceSession(out, providers=["CPUExecutionProvider"]).run(None, {"images": x.numpy()})[0]
    err = float(np.abs(got - ref).max())
    print(f"exported {os.path.basename(out)} {shape} -> {tuple(got.shape)}  max|ort-torch| {err:.2e}", flush=True)
    return err


def load_for_export(family: str, weights: str) -> torch.nn.Module:
    """Detector in export mode: FP32, eval, Detect head returning the concatenated raw output."""
    model = build(family, 8, weights).model.float().eval()
    head = model.model[-1]
    head.export, head.format = True, "onnx"
    return model


def find_weights(run: str) -> str | None:
    """<outputs.runs>/<run>/last.pth, else the release file of the run (efpmot.release), else None."""
    from efpmot.release import checkpoint_for
    return checkpoint_for(run)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--all", action="store_true", help="export the PUBLIC list (skips runs whose checkpoint is missing)")
    ap.add_argument("--out_dir", default=os.path.join("deploy", "jetson", "export"))
    ap.add_argument("--weights", help="checkpoint (training last.pth or release file)")
    ap.add_argument("--family", choices=["yolov8n", "yolov8n_s3x", "yolov8n_s3y", "yolo11n", "yolo11n_s3x", "yolo11n_s3y"])
    ap.add_argument("--hw", type=int, nargs=2, metavar=("H", "W"), help="detector canvas")
    ap.add_argument("--native", type=int, nargs=2, metavar=("H0", "W0"), help="native frame: export the letterbox in-graph")
    ap.add_argument("--out", help="output .onnx")
    a = ap.parse_args()
    if a.all:
        worst = 0.0
        for name, run, family, hw, native, graph_anchors in PUBLIC:
            weights = find_weights(run)
            if weights is None:
                print(f"skip {name}: no checkpoint for {run} in outputs.runs or release_assets/", flush=True)
                continue
            model = load_for_export(family, weights)
            worst = max(worst, export_model(model, hw, None, os.path.join(a.out_dir, f"{name}_canvas.onnx"),
                                            graph_anchors))
            if native:
                worst = max(worst, export_model(model, hw, native, os.path.join(a.out_dir, f"{name}_native.onnx")))
        print(f"EXPORT_DONE {a.out_dir}  worst max|ort-torch| {worst:.2e}")
        return
    if not (a.weights and a.family and a.hw and a.out):
        ap.error("--weights, --family, --hw and --out are required without --all")
    export_model(load_for_export(a.family, a.weights), tuple(a.hw), tuple(a.native) if a.native else None, a.out)


if __name__ == "__main__":
    main()
