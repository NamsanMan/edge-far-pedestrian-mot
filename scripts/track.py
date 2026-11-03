# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Track the cached detections of one run with ByteTrack / OC-SORT over the threshold grid.

    python -B scripts/track.py --dataset dsec --tag dsec_v8n_416s3x_s0                    # both trackers, whole grid
    python -B scripts/track.py --dataset dsec --tag dsec_v8n_416s3x_s0 --tracker bytetrack --thr 0.5
    [--jobs 4] [--cache_root DIR]
Reads <outputs.cache>/<tag>/<dataset>/*.npz, writes <outputs.results>/<dataset>_trackers/<tag>__<tracker>_t<thr>/data/.
Tracker folders that already hold every sequence are skipped by the grid sweep (not by an explicit --thr run).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from efpmot import DATASETS  # noqa: E402
from efpmot.track import GRID, run, sweep  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tracker", default="all", choices=["all", *GRID])
    ap.add_argument("--thr", type=float, nargs="*", default=None, help="explicit thresholds (default: the tracker grid)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--cache_root", default=None)
    a = ap.parse_args()
    trackers = list(GRID) if a.tracker == "all" else [a.tracker]
    print(f"track {a.dataset} {a.tag}: trackers {trackers} thresholds {a.thr or 'grid'}")
    if a.thr:
        for trk in trackers:
            for th in a.thr:
                run(a.dataset, a.tag, trk, th, cache_root=a.cache_root)
    else:
        sweep(a.dataset, a.tag, trackers, jobs=a.jobs, cache_root=a.cache_root)


if __name__ == "__main__":
    main()
