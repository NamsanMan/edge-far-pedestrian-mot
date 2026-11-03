# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""TrackEval (HOTA, CLEAR, Identity) on the tracker outputs of efpmot.track.

- DSEC-Det, BDD100K MOT val and VisDrone2019-MOT test-dev: TrackEval's BDD100K evaluator on BDD / Scalabel-format GT.
  DSEC uses a GT folder holding only the sequences of the evaluated split (<outputs.results>/gt_subsets/dsec_<split>).
- MOT17 evaluation halves: TrackEval's MOTChallenge evaluator with the official MOT17 preprocessing (tracker boxes
  matched to distractor GT are removed), pedestrian only, on the half-split GT of efpmot.data.mot17.

Output: <outputs.results>/<dataset>_eval/<subset>/<tracker>/<cls>_{detailed.csv,summary.txt} and a summary JSON.
TrackEval is imported from paths 'third_party.trackeval' (commit 12c8791b303e0a0b50f753af204249e622d0281a).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import sys

import numpy as np

from .. import paths
from ..track import GRID

SUBSET = {"dsec": "test", "mot17": "all", "visdrone": "all", "bdd": "all"}


# numpy >= 1.24 removed aliases TrackEval still uses. Patched at import time so that TrackEval's worker processes
# (multiprocessing re-imports the calling script, which imports this module) see them too.
for _n, _t in (("bool", bool), ("int", int), ("float", float), ("object", object)):
    if _n not in vars(np):
        setattr(np, _n, _t)


def _trackeval():
    """import TrackEval from the configured clone (paths 'third_party.trackeval')"""
    root = paths.require("third_party.trackeval")
    if root not in sys.path:
        sys.path.insert(0, root)
    import trackeval
    return trackeval


def dsec_gt_subset(subset: str = "test") -> str:
    """GT folder holding exactly the DSEC-Det sequences of a split (train | val | test | all); copied from the BDD-format GT"""
    from ..data import dsec
    src = dsec.gt_dir()
    sel = [os.path.splitext(f)[0] for f in os.listdir(src) if f.endswith(".json")] if subset == "all" else dsec.split_seqs(subset)
    d = paths.out("results", "gt_subsets", "dsec_" + subset)
    for f in glob.glob(os.path.join(d, "*.json")):  # keep the folder exactly equal to the subset
        if os.path.splitext(os.path.basename(f))[0] not in sel:
            os.remove(f)
    for s in sel:
        if not os.path.exists(os.path.join(d, s + ".json")):
            shutil.copy(os.path.join(src, s + ".json"), d)
    return d


def _seq_selection(dataset: str, subset: str, seqs: list[str], gt: str | None) -> tuple[str, str]:
    """GT folder (BDD format) or seqmap file (MOT17) restricted to the given sequences, for quick checks on a few
    sequences; returns (path, output subset name). Per-sequence numbers equal those of the full evaluation."""
    seqs = sorted(set(seqs))
    key = "sel_" + hashlib.md5(" ".join(seqs).encode()).hexdigest()[:8]
    if dataset == "mot17":
        from ..data import mot17
        with open(mot17.seqmap_file(subset)) as f:
            known = [ln.strip() for ln in f.read().splitlines()[1:] if ln.strip()]
        missing = [q for q in seqs if q not in known]
        if missing:
            raise ValueError(f"mot17 subset {subset!r} has no sequences {missing}")
        d = paths.out("results", "gt_subsets", "mot17_seqmaps")
        fn = os.path.join(d, f"MOT17-halfval-{key}.txt")
        with open(fn, "w") as f:
            f.write("name\n" + "".join(q + "\n" for q in known if q in seqs))
        return fn, f"{subset}_{key}"
    missing = [q for q in seqs if not os.path.exists(os.path.join(gt, q + ".json"))]
    if missing:
        raise ValueError(f"{dataset}: no GT for {missing} in {gt}")
    d = paths.out("results", "gt_subsets", f"{dataset}_{subset}_{key}")
    for f in glob.glob(os.path.join(d, "*.json")):
        if os.path.splitext(os.path.basename(f))[0] not in seqs:
            os.remove(f)
    for q in seqs:
        if not os.path.exists(os.path.join(d, q + ".json")):
            shutil.copy(os.path.join(gt, q + ".json"), d)
    return d, f"{subset}_{key}"


def _value(d: dict, k: str) -> float:
    v = d.get(k, np.nan)
    return float(np.mean(v)) if isinstance(v, np.ndarray) else float(v)


def evaluate(dataset: str, trackers: list[str], subset: str | None = None, classes: list[str] | None = None,
             quiet: bool = True, seqs: list[str] | None = None) -> list[dict]:
    """TrackEval on <outputs.results>/<dataset>_trackers/<tracker> for each tracker name; returns the summary rows
    {tracker, cls, HOTA, DetA, AssA, MOTA, IDF1, IDSW, FP, FN, Frag, GT_IDs} (percentages x100; the class average is
    named 'mean'). subset: dsec test | val | train | all (default test); visdrone all (= test-dev) | val; mot17 all |
    static | moving; bdd all. quiet: print only the mean / car / pedestrian rows. seqs: evaluate only these sequences of the
    subset (output subset folder '<subset>_sel_<hash>'; per-sequence rows equal those of the full evaluation)."""
    trackeval = _trackeval()
    subset = subset or SUBSET[dataset]
    results = paths.require("outputs.results")
    out_subset = subset
    ev_cfg = trackeval.Evaluator.get_default_eval_config()
    ev_cfg.update(dict(USE_PARALLEL=True, NUM_PARALLEL_CORES=6, PRINT_RESULTS=False, PRINT_ONLY_COMBINED=True,
                       PRINT_CONFIG=False, TIME_PROGRESS=False, DISPLAY_LESS_PROGRESS=True, OUTPUT_SUMMARY=True,
                       OUTPUT_DETAILED=True, PLOT_CURVES=False, LOG_ON_ERROR=None))
    trk_root = os.path.join(results, f"{dataset}_trackers")
    if dataset in ("dsec", "bdd", "visdrone"):
        from ..data import BDD_CLASSES
        if dataset == "dsec":
            gt = dsec_gt_subset(subset)
        elif dataset == "bdd":
            if subset != "all":
                raise ValueError("bdd: only subset 'all'")
            from ..data import bdd
            gt = bdd.gt_dir()
        else:
            from ..data import visdrone
            gt = visdrone.gt_dir("val" if subset == "val" else "test")
        if seqs:
            gt, out_subset = _seq_selection(dataset, subset, seqs, gt)
        out_root = os.path.join(results, f"{dataset}_eval", out_subset)
        ds_cfg = trackeval.datasets.BDD100K.get_default_dataset_config()
        ds_cfg.update(dict(GT_FOLDER=gt, TRACKERS_FOLDER=trk_root, OUTPUT_FOLDER=out_root, TRACKERS_TO_EVAL=trackers,
                           PRINT_CONFIG=False, CLASSES_TO_EVAL=classes or BDD_CLASSES))
        ds = trackeval.datasets.BDD100K(ds_cfg)
    elif dataset == "mot17":
        from ..data import mot17
        seqmap = mot17.seqmap_file(subset)
        if seqs:
            seqmap, out_subset = _seq_selection(dataset, subset, seqs, None)
        out_root = os.path.join(results, f"{dataset}_eval", out_subset)
        ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        ds_cfg.update(dict(GT_FOLDER=mot17.gt_root(), TRACKERS_FOLDER=trk_root, OUTPUT_FOLDER=out_root,
                           TRACKERS_TO_EVAL=trackers, PRINT_CONFIG=False, CLASSES_TO_EVAL=["pedestrian"], BENCHMARK="MOT17",
                           SPLIT_TO_EVAL="train", DO_PREPROC=True, SKIP_SPLIT_FOL=True, SEQMAP_FILE=seqmap))
        ds = trackeval.datasets.MotChallenge2DBox(ds_cfg)
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    res, _ = trackeval.Evaluator(ev_cfg).evaluate([ds], metrics)
    rows = []
    for trk, r in res[list(res.keys())[0]].items():
        for cls, m in r["COMBINED_SEQ"].items():
            if cls == "cls_comb_det_av":
                continue
            cls = "mean" if cls == "cls_comb_cls_av" else cls
            h, c, i = m.get("HOTA", {}), m.get("CLEAR", {}), m.get("Identity", {})
            rows.append(dict(tracker=trk, cls=cls, HOTA=_value(h, "HOTA") * 100, DetA=_value(h, "DetA") * 100,
                             AssA=_value(h, "AssA") * 100, MOTA=_value(c, "MOTA") * 100, IDF1=_value(i, "IDF1") * 100,
                             IDSW=_value(c, "IDSW"), FP=_value(c, "CLR_FP"), FN=_value(c, "CLR_FN"), Frag=_value(c, "Frag"),
                             GT_IDs=_value(c, "GT_IDs")))
    os.makedirs(out_root, exist_ok=True)
    name = "summary_" + "_".join(trackers)[:max(16, min(120, 230 - len(out_root)))]   # stay below the Windows path limit
    with open(os.path.join(out_root, name + ".json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(f"== {dataset} subset={out_subset}" + (f" sequences {sorted(set(seqs))}" if seqs else ""))
    print(f"{'tracker':50s} {'cls':11s} {'HOTA':>6s} {'DetA':>6s} {'AssA':>6s} {'MOTA':>6s} {'IDF1':>6s} {'IDSW':>6s} {'FP':>7s} {'FN':>7s}")
    for r in rows:
        if quiet and r["cls"] not in ("mean", "car", "pedestrian"):
            continue
        print(f"{r['tracker'][:50]:50s} {r['cls']:11s} {r['HOTA']:6.1f} {r['DetA']:6.1f} {r['AssA']:6.1f} {r['MOTA']:6.1f} "
              f"{r['IDF1']:6.1f} {r['IDSW']:6.0f} {r['FP']:7.0f} {r['FN']:7.0f}")
    return rows


def evaluate_tag(dataset: str, tag: str, trackers=("bytetrack", "ocsort"), classes: list[str] | None = None,
                 quiet: bool = True, subset: str | None = None, seqs: list[str] | None = None) -> list[dict]:
    """evaluate every grid tracker output of one run (names '<tag>__<tracker>_t<thr>') in one TrackEval call"""
    names = [f"{tag}__{t}_t{float(th)}" for t in trackers for th in GRID[t]]
    return evaluate(dataset, names, subset=subset, classes=classes, quiet=quiet, seqs=seqs)
