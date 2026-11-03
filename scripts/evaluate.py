# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""TrackEval HOTA / CLEAR / Identity of tracker outputs.

    python -B scripts/evaluate.py --dataset dsec --tag dsec_v8n_416s3x_s0                 # every grid output of the run
    python -B scripts/evaluate.py --dataset mot17 --trackers NAME [NAME ...] [--subset all|static|moving]
    [--classes car pedestrian] [--all_classes] [--seqs SEQ ...]
Writes <outputs.results>/<dataset>_eval/<subset>/<tracker>/<cls>_{detailed.csv,summary.txt} and summary_*.json.
--seqs evaluates only some sequences (subset folder '<subset>_sel_<hash>'); their per-sequence rows equal the full run's.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from efpmot import DATASETS  # noqa: E402
from efpmot.eval import evaluate, evaluate_tag  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tag", help="evaluate '<tag>__<tracker>_t<thr>' for both trackers over the whole grid")
    g.add_argument("--trackers", nargs="+", help="tracker folder names under <outputs.results>/<dataset>_trackers")
    ap.add_argument("--subset", default=None, help="dsec test|val|train|all, mot17 all|static|moving, visdrone all|val")
    ap.add_argument("--classes", nargs="*", default=None, help="classes to evaluate (default: all 8 BDD classes)")
    ap.add_argument("--all_classes", action="store_true", help="print every class (default: mean, car, pedestrian)")
    ap.add_argument("--seqs", nargs="*", default=None, help="evaluate only these sequences of the subset")
    a = ap.parse_args()
    if a.tag:
        evaluate_tag(a.dataset, a.tag, classes=a.classes, quiet=not a.all_classes, subset=a.subset, seqs=a.seqs)
    else:
        evaluate(a.dataset, a.trackers, subset=a.subset, classes=a.classes, quiet=not a.all_classes, seqs=a.seqs)


if __name__ == "__main__":
    main()
