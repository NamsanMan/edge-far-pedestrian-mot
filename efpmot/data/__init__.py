# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""efpmot.data: dataset readers and frame-pair training sets (DSEC-Det, MOT17, VisDrone2019-MOT, BDD100K), letterbox.

Every module exposes eval_seqs() and eval_frame_paths(seq) (the frames whose detections are exported) and native_hw / seq_hw.
Converted labels (frame indices, BDD-format GT, the MOT17 half-split GT, the BDD 35k subset) are written under
outputs.derived by build_index() / build_subset() (scripts/prepare_data.py); dataset folders are only read.
"""
from __future__ import annotations

import importlib

from .common import BDD_CLASSES, CAR, PED, iou_matrix, letterbox  # noqa: F401
from .pairs import FramePairs  # noqa: F401

_MODULES = {"dsec": "dsec", "mot17": "mot17", "visdrone": "visdrone", "bdd": "bdd"}


def get(dataset: str):
    """dataset key -> module efpmot.data.<dsec | mot17 | visdrone | bdd>"""
    if dataset not in _MODULES:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {sorted(_MODULES)}")
    return importlib.import_module(f"{__name__}.{_MODULES[dataset]}")


def train_dataset(dataset: str, size_hw, stride: int | None = None) -> FramePairs:
    """the training set of each dataset: dsec = split 'train', every stride-th labelled frame (default 2); visdrone = all
    train sequences, stride default 1; mot17 = the 7 training halves; bdd = the 35k detection subset (stride unused)"""
    m = get(dataset)
    size_hw = tuple(size_hw)
    if dataset == "dsec":
        return m.DsecPairs(m.split_seqs("train"), size_hw, train=True, stride=2 if stride is None else stride)
    if dataset == "visdrone":
        return m.VisDronePairs(m.list_seqs("train"), size_hw, train=True, stride=stride or m.TRAIN_STRIDE)
    if dataset == "mot17":
        return m.Mot17Pairs(m.SEQS, size_hw, train=True)
    return m.BddDet(size_hw, train=True)
