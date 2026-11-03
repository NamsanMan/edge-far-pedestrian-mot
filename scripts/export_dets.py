# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Export the detection cache of a trained detector on the evaluation frames of a dataset.

    python -B scripts/export_dets.py --data dsec --name dsec_v8n_416s3x_s0 [--family yolov8n_s3x] [--hw 320 416]
           [--weights PATH] [--device cuda] [--seqs SEQ ...]

--weights defaults to <runs>/<name>/last.pth, else the release file of that run tag (release_assets/, efpmot.release).
--family / --hw default to the training args of the checkpoint or, for a release file, to its entry in efpmot.release. --seqs restricts the export to some evaluation sequences
(quick checks). Writes <cache>/<name>/<data>/<seq>.npz.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from efpmot import DATASETS, paths  # noqa: E402
from efpmot.models import FAMILIES, build, load_trusted  # noqa: E402
from efpmot.release import checkpoint_for, release_meta  # noqa: E402
from efpmot.train import export_dets  # noqa: E402


def checkpoint_meta(path: str) -> dict:
    """family / hw of a release file (efpmot.release) or recorded in a training checkpoint (args)"""
    rel = release_meta(path)
    if rel:
        return {"family": rel["family"], "hw": list(rel["hw"])}
    ck = load_trusted(path)
    src = ck.get("args") if isinstance(ck, dict) else None
    return {k: src[k] for k in ("family", "hw") if k in src} if isinstance(src, dict) else {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, choices=DATASETS)
    ap.add_argument("--name", required=True, help="cache tag (run tag), e.g. dsec_v8n_416s3x_s0")
    ap.add_argument("--family", default=None, choices=FAMILIES)
    ap.add_argument("--hw", type=int, nargs=2, default=None, metavar=("H", "W"))
    ap.add_argument("--weights", default=None, help="checkpoint (default <runs>/<name>/last.pth)")
    ap.add_argument("--device", default="cuda", help="cuda (FP16, as the published caches) or cpu (FP32)")
    ap.add_argument("--seqs", nargs="*", default=None, help="only these evaluation sequences (default: all)")
    a = ap.parse_args()
    w = a.weights or checkpoint_for(a.name) or os.path.join(paths.require("outputs.runs"), a.name, "last.pth")
    meta = checkpoint_meta(w)
    family = a.family or meta.get("family")
    hw = a.hw or meta.get("hw")
    if family is None or hw is None:
        ap.error(f"{w} records no family / hw: pass --family and --hw")
    print("config:", dict(data=a.data, name=a.name, family=family, hw=list(hw), weights=w, device=a.device,
                          seqs=a.seqs or "all"), flush=True)
    ad = build(family, 8, w)
    print("cache:", export_dets(ad, a.name, a.data, tuple(hw), device=a.device, seqs=a.seqs), flush=True)


if __name__ == "__main__":
    main()
