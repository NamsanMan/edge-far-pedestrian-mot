# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""TrackEval <cls>_detailed.csv files -> per-run stats files results/stats/<dataset>/<tag>__<tracker>__<cls>.npz.

    python -B scripts/extract_stats.py --contrasts results/contrasts.yaml       # every run the published contrasts use
    python -B scripts/extract_stats.py --dataset dsec --tags dsec_v8n_416s3x_s0 --classes car pedestrian
    [--eval_root DIR]   folder holding <dataset>_eval/<subset>/ (default outputs.results)
    [--out DIR]         stats root (default results/stats)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml  # noqa: E402

from efpmot.eval import protocol as P  # noqa: E402
from efpmot.eval import stats  # noqa: E402


def jobs_from_contrasts(path: str, eval_root: str) -> list[tuple[str, str, str, str]]:
    """(dataset, tag, tracker, class) of every seed of every arm in the contrasts file, plus the reference arms of the
    transferred-margin status (VisDrone / MOT17 ByteTrack)"""
    want = set()
    for c in yaml.safe_load(open(path, encoding="utf-8")):
        for det, var in ((c["det"], c["a"]), (c.get("det_b", c["det"]), c["b"])):  # det_b: arm B of another detector
            for cls in c["classes"]:
                for trk in P.TRACKERS:
                    want.add((c["dataset"], det, str(var), trk, cls))
    for data, (arms, cls) in P.TRANSFERRED_MARGIN.items():
        for det, var in arms:
            want.add((data, det, var, "bytetrack", cls))
    out = []
    for data, det, var, trk, cls in sorted(want):
        ev = os.path.join(eval_root, *P.EVAL_DIR[data])
        for s in P.SEEDS:
            tag = P.tag_of(data, det, var, s)
            if any(os.path.isdir(os.path.join(ev, f"{tag}__{t}_t0.5")) for t in P.TRACKERS):
                out.append((data, tag, trk, cls))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contrasts", default=None)
    ap.add_argument("--dataset", choices=sorted(P.EVAL_DIR))
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--classes", nargs="*", default=["car", "pedestrian"])
    ap.add_argument("--trackers", nargs="*", default=list(P.TRACKERS))
    ap.add_argument("--eval_root", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from efpmot import paths
    root = os.path.abspath(a.eval_root) if a.eval_root else paths.require("outputs.results")
    if a.contrasts:
        jobs = jobs_from_contrasts(a.contrasts, root)
    elif a.dataset and a.tags:
        jobs = [(a.dataset, t, trk, cls) for t in a.tags for trk in a.trackers for cls in a.classes]
    else:
        ap.error("give --contrasts, or --dataset with --tags")
    print(f"eval root {root}: {len(jobs)} stats files")
    for data, tag, trk, cls in jobs:
        f = stats.extract(data, tag, trk, cls, eval_root=os.path.join(root, *P.EVAL_DIR[data]), out_root=a.out)
        print(("wrote " + os.path.relpath(f)) if f else f"no readable TrackEval output: {data} {tag} {trk} {cls}")


if __name__ == "__main__":
    main()
