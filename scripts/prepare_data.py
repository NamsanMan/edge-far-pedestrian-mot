# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Convert dataset labels into the derived files the code reads (outputs.derived; dataset folders are only read).

    python -B scripts/prepare_data.py dsec        # <derived>/dsec_det/{frame_index.json, gt_bdd/<seq>.json}
    python -B scripts/prepare_data.py mot17       # <derived>/mot17/gt_halfval/ (evaluation-half GT + seqmaps)
    python -B scripts/prepare_data.py visdrone    # <derived>/visdrone_mot/{frame_index.json, gt_bdd/{val,test-dev}/}
    python -B scripts/prepare_data.py bdd [--images_zip Z --labels_zip Z]
        # optional first step: legacy 100k zips -> data.bdd_det (train2017/, val2017/, annotations/det_{train,val}.json);
        # then <derived>/bdd100k_det/annotations/det_train_35k.json
    python -B scripts/prepare_data.py hituav [--force] [--compare SPLIT_JSON]
        # rebuilds the HIT-UAV recording-disjoint split from the dataset and checks it against the shipped
        # efpmot/data/splits/hituav_split.json (a different build is never written over it without --force);
        # --compare: also list the differences to another split file in the original research layout
    python -B scripts/prepare_data.py all         # dsec, mot17, visdrone, bdd
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from efpmot.data import bdd, dsec, hituav, mot17, visdrone  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["dsec", "mot17", "visdrone", "bdd", "hituav", "all"])
    ap.add_argument("--images_zip", default=None, help="bdd: BDD100K 100k images zip (runs prepare_det first)")
    ap.add_argument("--labels_zip", default=None, help="bdd: legacy BDD100K labels zip (bdd100k_labels.zip)")
    ap.add_argument("--max_train", type=int, default=0, help="bdd prepare_det: limit the train label files (0 = all)")
    ap.add_argument("--force", action="store_true", help="hituav: replace a shipped split that differs from the rebuild")
    ap.add_argument("--compare", default=None, help="hituav: a split file in the research layout to compare with")
    a = ap.parse_args()
    if bool(a.images_zip) != bool(a.labels_zip):
        ap.error("bdd: give both --images_zip and --labels_zip, or neither")
    todo = ["dsec", "mot17", "visdrone", "bdd"] if a.dataset == "all" else [a.dataset]
    for d in todo:
        print(f"--- {d}", flush=True)
        if d == "dsec":
            dsec.build_index()
        elif d == "mot17":
            mot17.build_index()
        elif d == "visdrone":
            visdrone.build_index()
        elif d == "hituav":
            hituav.build_split_file(force=a.force)
            if a.compare:
                diff = hituav.compare_split(a.compare)
                print(f"compare with {a.compare}: " + ("identical recordings, flights and folds" if not diff else
                                                       f"{len(diff)} differences:\n  " + "\n  ".join(diff)), flush=True)
                if diff:
                    sys.exit(1)
        else:
            if a.images_zip:
                bdd.prepare_det(a.images_zip, a.labels_zip, max_train=a.max_train)
            bdd.build_subset()


if __name__ == "__main__":
    main()
