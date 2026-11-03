# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Per-run stats files: the HOTA sufficient statistics of one run, tracker and class over the threshold grid.

``results/stats/<dataset>/<tag>__<tracker>__<cls>.npz`` holds

- ``seqs``  str [S], sorted sequence names;
- ``ths``   str [T], the thresholds of the tracker grid whose TrackEval output was readable;
- ``X``     float64 [T, S, 4, 19] = HOTA_TP, HOTA_FN, HOTA_FP, AssA * HOTA_TP at alpha = 0.05 ... 0.95;
- ``comb``  float64 [T], the COMBINED HOTA x100 of each TrackEval file.

The file is lossless for the protocol: ``protocol.load_run(..., source="stats")`` returns the same arrays as reading the
TrackEval ``<cls>_detailed.csv`` files, so the published tables can be regenerated without any dataset.
"""
from __future__ import annotations

import os

import numpy as np

from . import protocol as P


def extract(data: str, tag: str, trk: str, cls: str, eval_root: str | None = None, out_root: str | None = None) -> str | None:
    """TrackEval CSVs of every grid threshold -> stats file; returns its path, or None when no threshold is readable.

    eval_root: folder holding the '<tag>__<trk>_t<th>' TrackEval output folders (default <outputs.results>/<dataset>_eval/
    <subset>); out_root: stats root (default <repo>/results/stats)."""
    root = eval_root or P.eval_root(data)
    ths, Xs, combs, ref = [], [], [], None
    for th in P.GRID[trk]:
        r = P.read_detailed(os.path.join(root, f"{tag}__{trk}_t{th}", f"{cls}_detailed.csv"))
        if r is None:
            continue
        seqs, X, comb = r
        if ref is None:
            ref = sorted(seqs)
        if sorted(seqs) != ref:
            raise ValueError(f"{tag} {trk} t{th} {cls}: sequence set differs from the other thresholds")
        order = {q: i for i, q in enumerate(seqs)}
        ths.append(th); Xs.append(X[[order[q] for q in ref]]); combs.append(comb)
    if not ths:
        return None
    out = P.stats_file(data, tag, trk, cls, out_root)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, seqs=np.array(ref), ths=np.array(ths), X=np.stack(Xs).astype(np.float64),
                        comb=np.array(combs, np.float64))
    return out


def read(data: str, tag: str, trk: str, cls: str, root: str | None = None) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """-> (seqs, ths, X [T, S, 4, 19], comb [T])"""
    with np.load(P.stats_file(data, tag, trk, cls, root), allow_pickle=False) as z:
        return [str(s) for s in z["seqs"]], [str(t) for t in z["ths"]], z["X"], z["comb"]
