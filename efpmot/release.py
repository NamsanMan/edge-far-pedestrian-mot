# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Released detector weights (GitHub Releases, downloaded into <repo>/release_assets/).

Each release file is a bare fp32 state dict (torch.save of model.state_dict(); loads with torch.load(weights_only=True)).
The file name carries the family, training data, canvas (W x H) and seed; RELEASES maps it to the run tag used by the
detection caches, tracker outputs and results/stats, and to the settings needed to rebuild the model:

    from efpmot.models import build
    ad = build("yolov8n_s3x", 8, "release_assets/yolov8n_s3x_dsec_416x320_s0.pt")

HIT-UAV files (dataset 'hituav-<fold>': trained on the train part of that recording-disjoint fold, efpmot/data/hituav.py) also
carry thr = the person score threshold with the best F1 on the fold's val part (the operating point of
results/hituav/<run>.json, op.pedestrian.thr; results/HITUAV.md).
"""
from __future__ import annotations

import os

from . import paths

RELEASE_DIR = "release_assets"

RELEASES = {
    "yolov8n_plain_dsec_416x320_s0.pt": dict(run="dsec_v8n_416_s0", family="yolov8n", dataset="dsec", hw=(320, 416),
                                             seed=0, epochs=6),
    "yolov8n_s3x_dsec_416x320_s0.pt": dict(run="dsec_v8n_416s3x_s0", family="yolov8n_s3x", dataset="dsec", hw=(320, 416),
                                           seed=0, epochs=6),
    "yolov8n_plain_mot17_416x256_s0.pt": dict(run="mot_v8n_416_s0", family="yolov8n", dataset="mot17", hw=(256, 416),
                                              seed=0, epochs=55),
    "yolov8n_s3x_mot17_416x256_s0.pt": dict(run="mot_v8n_416s3x_s0", family="yolov8n_s3x", dataset="mot17", hw=(256, 416),
                                            seed=0, epochs=55),
    "yolov8n_plain_hituav-recA_416x320_s0.pt": dict(run="hit_v8n_416recA_s0", family="yolov8n", dataset="hituav-recA",
                                                    hw=(320, 416), seed=0, epochs=116, thr=0.43017578125),
    "yolov8n_s3x_hituav-recA_416x320_s0.pt": dict(run="hit_v8n_416s3xrecA_s0", family="yolov8n_s3x", dataset="hituav-recA",
                                                  hw=(320, 416), seed=0, epochs=116, thr=0.154052734375),
    "yolov8n_plain_hituav-recB_416x320_s0.pt": dict(run="hit_v8n_416recB_s0", family="yolov8n", dataset="hituav-recB",
                                                    hw=(320, 416), seed=0, epochs=118, thr=0.056640625),
    "yolov8n_s3x_hituav-recB_416x320_s0.pt": dict(run="hit_v8n_416s3xrecB_s0", family="yolov8n_s3x", dataset="hituav-recB",
                                                  hw=(320, 416), seed=0, epochs=118, thr=0.3056640625),
}
RUN_TO_FILE = {v["run"]: k for k, v in RELEASES.items()}


def dataset_of_run(run: str) -> str:
    """dataset name of a run tag in release file names: the dataset key (efpmot.dataset_of_tag), or for HIT-UAV runs 'hituav'
    (official split) / 'hituav-<fold>' (the tag's fold marker, e.g. hit_v8n_416s3xrecA_s0 -> 'hituav-recA')"""
    if run.startswith("hit_"):
        from .data.hituav import split_of_tag
        s = split_of_tag(run)
        return "hituav" if s == "official" else f"hituav-{s}"
    from . import dataset_of_tag
    return dataset_of_tag(run)


def release_meta(path: str) -> dict | None:
    """settings of a release file (by its file name), or None for any other checkpoint"""
    m = RELEASES.get(os.path.basename(path))
    return dict(m, file=os.path.basename(path)) if m else None


def release_path(run: str) -> str | None:
    """<repo>/release_assets/<file> of a released run tag when that file exists, else None"""
    fn = RUN_TO_FILE.get(run)
    if fn is None:
        return None
    p = os.path.join(paths.repo_root(), RELEASE_DIR, fn)
    return p if os.path.isfile(p) else None


def checkpoint_for(run: str) -> str | None:
    """<outputs.runs>/<run>/last.pth when it exists, else the release file of the run, else None"""
    runs = paths.get("outputs.runs")
    if runs:
        p = os.path.join(runs, run, "last.pth")
        if os.path.isfile(p):
            return p
    return release_path(run)
