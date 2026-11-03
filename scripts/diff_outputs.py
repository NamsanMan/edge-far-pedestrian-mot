# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Compare pipeline outputs with a reference copy (reproduction checks, docs/PORT_VERIFICATION.md).

    python -B scripts/diff_outputs.py cache    NEW_DIR REF_DIR       # <cache>/<tag>/<dataset>/ folders of .npz files
    python -B scripts/diff_outputs.py trackers NEW_DIR REF_DIR       # <results>/<dataset>_trackers/<name>/ folders
    python -B scripts/diff_outputs.py eval     NEW_DIR REF_DIR [--cls pedestrian car]   # TrackEval <tracker>/ folders

Only the sequences present in NEW_DIR are compared, so a partial re-run can be checked against a full reference.
- cache: frame names must match. Reported: bit-identical frames; frames whose box count differs; on the frames with equal
  counts, the maximum |box| (native px) and |score| differences after pairing boxes by class and position (Hungarian
  assignment on the L1 box distance); and the share of confident reference boxes (score >= 0.3) that the new cache
  reproduces (same class, IoU >= 0.9, score >= 0.15).
- trackers: byte equality of every data/<seq> file.
- eval: every number of the per-sequence rows of <cls>_detailed.csv (the COMBINED row is skipped, as it depends on the
  set of sequences).
Exit code 0 when everything is equal (cache: when every count matches), 1 otherwise.
"""
import argparse
import csv
import filecmp
import glob
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1, y1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def _pairs(da: np.ndarray, db: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """same-class pairs of a Hungarian assignment on the L1 box distance"""
    if not len(da) or not len(db):
        return np.zeros(0, int), np.zeros(0, int)
    cost = np.abs(da[:, None, :4] - db[None, :, :4]).sum(-1) + 1e6 * (da[:, None, 5] != db[None, :, 5])
    i, j = linear_sum_assignment(cost)
    k = cost[i, j] < 1e6
    return i[k], j[k]


def diff_cache(new: str, ref: str, conf: float = 0.3) -> bool:
    """Equal-count frames: max |box| / |score| of the paired boxes (numerical noise). Every frame: the reference boxes with
    score >= conf that have a same-class new box at IoU >= 0.9 with score >= conf / 2 (agreement of confident boxes)."""
    files = sorted(glob.glob(os.path.join(new, "*.npz")))
    if not files:
        sys.exit(f"no .npz files in {new}")
    ok = True
    keys = ("frames", "boxes", "exact", "count_diff", "conf_ref", "conf_hit")
    tot = dict.fromkeys(keys, 0) | dict(box=0.0, score=0.0)
    for f in files:
        seq = os.path.basename(f)[:-4]
        a, b = np.load(f, allow_pickle=True), np.load(os.path.join(ref, seq + ".npz"), allow_pickle=True)
        if list(a["frames"]) != list(b["frames"]):
            print(f"{seq}: frame lists differ ({len(a['frames'])} vs {len(b['frames'])})")
            ok = False
            continue
        s = dict.fromkeys(keys, 0) | dict(box=0.0, score=0.0)
        for da, db in zip(a["dets"], b["dets"]):
            s["frames"] += 1
            s["boxes"] += len(db)
            if da.shape == db.shape and np.array_equal(da, db):
                s["exact"] += 1
            if len(da) != len(db):
                s["count_diff"] += 1
            elif len(da):
                i, j = _pairs(da, db)
                if len(i) == len(da):
                    s["box"] = max(s["box"], float(np.abs(da[i, :4] - db[j, :4]).max()))
                    s["score"] = max(s["score"], float(np.abs(da[i, 4] - db[j, 4]).max()))
            rc = db[db[:, 4] >= conf] if len(db) else db
            s["conf_ref"] += len(rc)
            if len(rc):
                nc = da[da[:, 4] >= conf / 2] if len(da) else da
                if len(nc):
                    m = (_iou(rc, nc) >= 0.9) & (rc[:, None, 5] == nc[None, :, 5])
                    s["conf_hit"] += int(m.any(1).sum())
        print(f"{seq}: {s['frames']} frames, {s['boxes']} reference boxes, bit-identical frames {s['exact']}, frames with a "
              f"different box count {s['count_diff']}, equal-count frames max |box| {s['box']:.2e} px max |score| "
              f"{s['score']:.2e}, confident boxes matched {s['conf_hit']}/{s['conf_ref']}")
        for k in keys:
            tot[k] += s[k]
        tot["box"], tot["score"] = max(tot["box"], s["box"]), max(tot["score"], s["score"])
        ok &= s["count_diff"] == 0
    pct = 100.0 * tot["conf_hit"] / max(tot["conf_ref"], 1)
    print(f"TOTAL: {tot['frames']} frames, {tot['boxes']} reference boxes, bit-identical frames {tot['exact']}, frames with a "
          f"different box count {tot['count_diff']}, equal-count frames max |box| {tot['box']:.2e} px max |score| "
          f"{tot['score']:.2e}, confident (score >= {conf}) reference boxes matched at IoU >= 0.9: {tot['conf_hit']}/"
          f"{tot['conf_ref']} ({pct:.2f} %)")
    return ok


def diff_trackers(new: str, ref: str) -> bool:
    files = sorted(glob.glob(os.path.join(new, "data", "*")))
    if not files:
        sys.exit(f"no files in {new}/data")
    same = [f for f in files if filecmp.cmp(f, os.path.join(ref, "data", os.path.basename(f)), shallow=False)]
    for f in files:
        if f not in same:
            print("DIFFERENT:", os.path.basename(f))
    print(f"{os.path.basename(os.path.normpath(new))}: {len(same)}/{len(files)} files byte-identical")
    return len(same) == len(files)


def _rows(path: str) -> dict:
    with open(path, newline="") as f:
        r = list(csv.reader(f))
    return {row[0]: dict(zip(r[0][1:], row[1:])) for row in r[1:] if row and row[0] != "COMBINED"}


def diff_eval(new: str, ref: str, classes: list[str]) -> bool:
    ok = True
    for cls in classes:
        a, b = _rows(os.path.join(new, f"{cls}_detailed.csv")), _rows(os.path.join(ref, f"{cls}_detailed.csv"))
        for seq, ra in a.items():
            rb = b[seq]
            bad = [k for k in ra if not np.isclose(float(ra[k] or "nan"), float(rb[k] or "nan"), rtol=0, atol=0, equal_nan=True)]
            hota = float(ra["HOTA___AUC"]) * 100
            print(f"{os.path.basename(os.path.normpath(new))} {cls:10s} {seq:18s} HOTA {hota:6.2f}  "
                  f"{'all ' + str(len(ra)) + ' columns equal' if not bad else 'DIFFERENT: ' + ', '.join(bad[:5])}")
            ok &= not bad
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["cache", "trackers", "eval"])
    ap.add_argument("new")
    ap.add_argument("ref")
    ap.add_argument("--cls", nargs="+", default=["pedestrian", "car"])
    a = ap.parse_args()
    if a.kind == "cache":
        ok = diff_cache(a.new, a.ref)
    elif a.kind == "trackers":
        ok = diff_trackers(a.new, a.ref)
    else:
        ok = diff_eval(a.new, a.ref, a.cls)
    print("EQUAL" if ok else "DIFFERENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
