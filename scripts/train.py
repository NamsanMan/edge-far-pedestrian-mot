# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Train one detector arm and export its detection cache.

    python -B scripts/train.py --family yolov8n_s3x --data dsec --name dsec_v8n_416s3x_s0 [--hw 320 416] [--epochs 6]
           [--bs 16] [--workers 4] [--seed 0] [--dsec_stride 2] [--vis_stride 1] [--max_iters N] [--no_export]
           [--device cuda]

--hw / --epochs default to efpmot.train.TRAIN_DEFAULTS[--data]. Writes <runs>/<name>/{last.pth, log.txt} and
<cache>/<name>/<data>/<seq>.npz (configs/paths.yaml: outputs.runs, outputs.cache). Run-tag convention:
<prefix>_<det>_<variant>_s<seed>, e.g. dsec_v8n_416s3x_s0, mot_v8n_480x256_s1 (docs/REPRODUCE.md).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from efpmot import DATASETS  # noqa: E402
from efpmot.models import FAMILIES  # noqa: E402
from efpmot.train import TRAIN_DEFAULTS, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=FAMILIES)
    ap.add_argument("--data", required=True, choices=DATASETS)
    ap.add_argument("--name", required=True, help="run tag, e.g. dsec_v8n_416s3x_s0")
    ap.add_argument("--hw", type=int, nargs=2, default=None, metavar=("H", "W"), help="canvas size (default per dataset)")
    ap.add_argument("--epochs", type=int, default=None, help="default per dataset (6; MOT17 55)")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dsec_stride", type=int, default=2, help="DSEC: every k-th labelled frame")
    ap.add_argument("--vis_stride", type=int, default=None, help="VisDrone: frame stride over the annotated frames (default 1)")
    ap.add_argument("--max_iters", type=int, default=0, help="smoke test: stop after N iterations (no checkpoint, no export)")
    ap.add_argument("--no_export", dest="export", action="store_false", help="skip the detection-cache export at the end")
    ap.add_argument("--device", default="cuda", help="cuda (published recipe, FP16) or cpu (FP32, smoke tests only)")
    a = ap.parse_args()
    a.hw = list(a.hw or TRAIN_DEFAULTS[a.data]["hw"])
    a.epochs = a.epochs or TRAIN_DEFAULTS[a.data]["epochs"]
    print("config:", vars(a), flush=True)
    print("run dir:", train(a), flush=True)


if __name__ == "__main__":
    main()
