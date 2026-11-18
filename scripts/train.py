# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Train one detector arm and export its detection cache.

    python -B scripts/train.py --family yolov8n_s3x --data dsec --name dsec_v8n_416s3x_s0 [--hw 320 416] [--epochs 6]
           [--bs 16] [--workers 4] [--seed 0] [--dsec_stride 2] [--vis_stride 1] [--max_iters N] [--no_export]
           [--device cuda]
    python -B scripts/train.py --family yolov8n_s3x --data hituav --split recA --name hit_v8n_416s3xrecA_s0

--hw / --epochs default to efpmot.train.TRAIN_DEFAULTS[--data] (HIT-UAV: epochs of the --split). Writes
<runs>/<name>/{last.pth, log.txt} and <cache>/<name>/<data>/<seq>.npz (configs/paths.yaml: outputs.runs, outputs.cache;
HIT-UAV: <cache>/<name>/hituav/{val,test}.npz, evaluated by scripts/evaluate_hituav.py). Run-tag convention:
<prefix>_<det>_<variant>_s<seed>, e.g. dsec_v8n_416s3x_s0, mot_v8n_480x256_s1 (docs/REPRODUCE.md); HIT-UAV
hit_<det>_<variant>[<fold>]_s<seed>, e.g. hit_v8n_416s3xrecA_s0 (a fold run must carry its fold marker).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from efpmot.models import FAMILIES  # noqa: E402
from efpmot.train import TRAIN_DEFAULTS, default_epochs, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=FAMILIES)
    ap.add_argument("--data", required=True, choices=tuple(TRAIN_DEFAULTS))
    ap.add_argument("--name", required=True, help="run tag, e.g. dsec_v8n_416s3x_s0")
    ap.add_argument("--split", default=None, choices=("official", "recA", "recB", "altLH"),
                    help="HIT-UAV only: official split (default) or a recording-disjoint fold (efpmot/data/hituav.py)")
    ap.add_argument("--hw", type=int, nargs=2, default=None, metavar=("H", "W"), help="canvas size (default per dataset)")
    ap.add_argument("--epochs", type=int, default=None, help="default per dataset (6; MOT17 55; HIT-UAV 72 / 116 / 118 / 112)")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dsec_stride", type=int, default=2, help="DSEC: every k-th labelled frame")
    ap.add_argument("--vis_stride", type=int, default=None, help="VisDrone: frame stride over the annotated frames (default 1)")
    ap.add_argument("--max_iters", type=int, default=0, help="smoke test: stop after N iterations (no checkpoint, no export)")
    ap.add_argument("--no_export", dest="export", action="store_false", help="skip the detection-cache export at the end")
    ap.add_argument("--device", default="cuda", help="cuda (published recipe, FP16) or cpu (FP32, smoke tests only)")
    a = ap.parse_args()
    if a.split and a.data != "hituav":
        ap.error("--split is a HIT-UAV option")
    if a.data == "hituav":
        from efpmot.data.hituav import check_tag
        a.split = a.split or "official"
        try:
            check_tag(a.name, a.split)
        except ValueError as ex:
            ap.error(str(ex))
    a.hw = list(a.hw or TRAIN_DEFAULTS[a.data]["hw"])
    a.epochs = a.epochs or default_epochs(a.data, a.split)
    print("config:", vars(a), flush=True)
    print("run dir:", train(a), flush=True)


if __name__ == "__main__":
    main()
