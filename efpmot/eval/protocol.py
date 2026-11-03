# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Comparison protocol for tracking HOTA of two detector arms (CPU only; reads TrackEval output or stats files).

An arm is a detector variant trained with several seeds; its run tags are ``<prefix>_<det>_<variant>_s<seed>``
(``tag_of``). For one class and one tracker the protocol

(a) picks the tracker detection threshold without scoring a test sequence on itself. Rule ``cross`` (primary):
    leave-one-sequence-out cross-fitting on DSEC test (13 sequences), MOT17 evaluation halves (7) and VisDrone test-dev
    (17); 10-fold video cross-fitting on BDD100K val (200 videos, fixed fold assignment: sorted names, numpy
    ``default_rng(0)`` permutation, position mod 10). Each run and class gets its own threshold per held-out fold,
    chosen by the pooled HOTA of the other folds (ties: closest to 0.5 as computed in floating point, then the higher
    value; so 0.6 beats 0.4, 0.7 beats 0.3 and 0.2 beats 0.8). ``best`` (one threshold maximising pooled HOTA over all
    test sequences) and ``fixed`` (0.5) are sensitivity rules;
(b) recombines TrackEval HOTA exactly from ``<class>_detailed.csv``: per alpha, HOTA_TP / FN / FP are summed over
    sequences and AssA is weighted by HOTA_TP; DetA = TP / max(1, TP + FN + FP), AssA = sum(AssA * TP) / max(1, TP),
    HOTA = mean over the 19 alphas of sqrt(DetA * AssA) (TrackEval ``HOTA.combine_sequences``). Every file is checked
    against its COMBINED row;
(c) paired hierarchical bootstrap, B = 10000: sequences (BDD: videos) are resampled with replacement with the same draw
    for both arms, then seeds are resampled with replacement within each arm; statistic = seed-mean pooled HOTA of A
    minus that of B; 95 % and 90 % percentile intervals;
(d) paired approximate randomisation: per sequence the seed-averaged sufficient statistics of A and B are swapped with
    probability 1/2. When 2^K <= B (K swap units: DSEC 13, MOT17 7) all 2^K patterns are enumerated and
    p = #(|T*| >= |T|) / 2^K exactly; otherwise (VisDrone 17, BDD 200) B random patterns are drawn and
    p = (1 + #(|T*| >= |T|)) / (B + 1); two-sided;
(e) TOST equivalence against a frozen margin (``MARGINS``); TOST p = max of the two one-sided bootstrap p-values
    (equivalently: the 90 % interval lies inside +-margin); minimum detectable effect at 80 % power, two-sided
    alpha 0.05: (z_0.975 + z_0.80) x bootstrap SE;
(f) labels (``label``): win / loss if the Holm-adjusted randomisation p < 0.05 (sign of d), else equivalent if the
    Holm-adjusted TOST p < 0.05, else inconclusive; a win / loss that also passes TOST gets the note ' <m'
    (significant but smaller than the margin).

Margins. DSEC 1.1955 and BDD 0.0421 HOTA are 2 x the median within-arm seed SD (ddof 1) of pooled ByteTrack car HOTA
under the cross-fitted rule, computed once from the seed spread of earlier 416 / 640 canvas arms and frozen before the
comparisons reported here were read; they do not depend on any difference between arms.
VisDrone and MOT17 use the DSEC value, frozen before any training on them (same leave-one-sequence-out design and about
the same number of iterations); their own seed-SD rule would rest on 2 arms x 2 seeds and is reported for reference only
(``margin_status``).

Runs whose threshold sweep is incomplete (ByteTrack 0.2..0.9 = 8 values, OC-SORT 0.5..0.9 = 5 values, each with the
full sequence set and a matching COMBINED row) are excluded and listed in ``exclA`` / ``exclB``.

Recording clusters. The 13 DSEC test sequences come from 8 drives (``zurich_city_14_a`` / ``_b`` / ``_c`` are one drive)
and the 17 VisDrone test-dev clips from 14 source videos. The primary rule treats each sequence as a unit; the
sensitivity checks ``sensitivity_video`` (VisDrone, inside ``one``) and ``cluster_sensitivity`` (DSEC drives, reported
by the table) cross-fit the thresholds leave-one-cluster-out and resample / swap whole clusters (``clusters_for``).

Data sources (``source``): ``"trackeval"`` reads ``<outputs.results>/<dataset>_eval/<subset>/<tag>__<trk>_t<th>/
<cls>_detailed.csv``; ``"stats"`` reads the lossless per-run stats files ``results/stats/<dataset>/<tag>__<trk>__<cls>.npz``
shipped with the repository (``efpmot.eval.stats``); ``"auto"`` uses the stats file when it exists.
"""
from __future__ import annotations

import csv
import os
import re

import numpy as np
from scipy.stats import norm

from .. import TAG_PREFIX, paths

EVAL_DIR = {"dsec": ("dsec_eval", "test"), "mot17": ("mot17_eval", "all"), "visdrone": ("visdrone_eval", "all"),
            "bdd": ("bdd_eval", "all")}
NSEQ = {"dsec": 13, "mot17": 7, "visdrone": 17, "bdd": 200}
GRID = {"bytetrack": ["0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"], "ocsort": ["0.5", "0.6", "0.7", "0.8", "0.9"]}
TRACKERS = ("bytetrack", "ocsort")
AL = [str(a) for a in range(5, 100, 5)]
SEEDS = range(10)
BDD_FOLDS, FOLD_SEED = 10, 0
RULES = ("cross", "best", "fixed")
Z80 = float(norm.ppf(0.975) + norm.ppf(0.80))
MARGINS = {"dsec": 1.1955, "bdd": 0.0421, "visdrone": 1.1955, "mot17": 1.1955}
# datasets whose margin is the transferred DSEC value; the arms / class of their reference seed-SD rule
TRANSFERRED_MARGIN = {"visdrone": ([("v8n", "640"), ("v8n", "704x384")], "car"),
                      "mot17": ([("v8n", "416"), ("v8n", "480x256")], "pedestrian")}
# MOT17 static-camera sequences (descriptive split of `one`)
STATIC = {"mot17": ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-09-FRCNN"]}

# leave-one-recording-cluster-out threshold rule of the sensitivity checks (clusters_for)
CLUSTER_RULE = {"visdrone": "cross_vid", "dsec": "cross_drive"}

REFSEQ: dict[str, list[str]] = {}       # dataset -> sorted sequence names (reference order of every X array)
_METHODS: dict[tuple, dict] = {}


def stats_root() -> str:
    """folder of the shipped stats files: <repo>/results/stats"""
    return os.path.join(paths.repo_root(), "results", "stats")


def stats_file(data: str, tag: str, trk: str, cls: str, root: str | None = None) -> str:
    return os.path.join(root or stats_root(), data, f"{tag}__{trk}__{cls}.npz")


def eval_root(data: str) -> str:
    """<outputs.results>/<dataset>_eval/<subset> of the protocol"""
    return os.path.join(paths.require("outputs.results"), *EVAL_DIR[data])


# ---- reading ------------------------------------------------------------------------------------------------------------
def read_detailed(path: str) -> tuple[list[str], np.ndarray, float] | None:
    """TrackEval <cls>_detailed.csv -> (sequence names, X [S, 4, 19] = TP, FN, FP, AssA * TP per alpha,
    COMBINED HOTA x100), or None if the file is missing, unreadable or partial"""
    if not os.path.isfile(path):
        return None
    seqs, X, comb = [], [], None
    try:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r["seq"] == "COMBINED":
                    comb = 100 * float(r["HOTA___AUC"])
                    continue
                tp = np.array([float(r[f"HOTA_TP___{a}"]) for a in AL]); fn = np.array([float(r[f"HOTA_FN___{a}"]) for a in AL])
                fp = np.array([float(r[f"HOTA_FP___{a}"]) for a in AL]); ass = np.array([float(r[f"AssA___{a}"]) for a in AL])
                X.append(np.stack([tp, fn, fp, ass * tp])); seqs.append(r["seq"])
    except (KeyError, ValueError, TypeError, csv.Error):
        return None
    if comb is None or not seqs:
        return None
    return seqs, np.stack(X), comb


def hota(X: np.ndarray) -> np.ndarray:
    """X [..., 4, 19] summed sufficient statistics -> HOTA x100 (TrackEval combine rule)"""
    tp, fn, fp, at = X[..., 0, :], X[..., 1, :], X[..., 2, :], X[..., 3, :]
    det = tp / np.maximum(1.0, tp + fn + fp); ass = at / np.maximum(1.0, tp)
    return 100 * np.sqrt(det * ass).mean(-1)


def tag_of(data: str, det: str, var: str, seed: int) -> str:
    """run tag of an arm and seed; a variant 'base_suffix' (e.g. '416_p2') gives '<prefix>_<det>_<base>_s<seed>_<suffix>'"""
    base, _, suf = var.partition("_")
    return f"{TAG_PREFIX.get(data, data)}_{det}_{base}_s{seed}" + (f"_{suf}" if suf else "")


def _source_of(data: str, tag: str, trk: str, cls: str, source: str) -> str:
    if source == "auto":
        return "stats" if os.path.isfile(stats_file(data, tag, trk, cls)) else "trackeval"
    if source not in ("stats", "trackeval"):
        raise ValueError(f"source must be auto | stats | trackeval, not {source!r}")
    return source


def _run_files(data: str, tag: str, trk: str, cls: str, source: str):
    """yields (threshold, parsed file or None, missing: bool) over GRID[trk]"""
    if source == "stats":
        from .stats import read as read_stats
        f = stats_file(data, tag, trk, cls)
        got = {}
        if os.path.isfile(f):
            seqs, ths, X, comb = read_stats(data, tag, trk, cls)
            got = {t: (list(seqs), X[i], float(comb[i])) for i, t in enumerate(ths)}
        for th in GRID[trk]:
            yield th, got.get(th), th not in got
        return
    for th in GRID[trk]:
        p = os.path.join(eval_root(data), f"{tag}__{trk}_t{th}", f"{cls}_detailed.csv")
        if not os.path.isfile(p):
            yield th, None, True
        else:
            yield th, read_detailed(p), False


def load_run(data: str, tag: str, trk: str, cls: str, source: str = "auto") -> tuple[dict[str, np.ndarray], str]:
    """-> (dict threshold -> X [S, 4, 19] in reference sequence order, problem string or '')"""
    src = _source_of(data, tag, trk, cls, source)
    d, miss = {}, []
    for th, r, missing in _run_files(data, tag, trk, cls, src):
        if missing:
            miss.append(th); continue
        if r is None:
            miss.append(th + "(unreadable)"); continue
        seqs, X, comb = r
        err = abs(float(hota(X.sum(0))) - comb)
        if err > 1e-6:
            miss.append(th + f"(recomb err {err:.2g})"); continue
        if data not in REFSEQ:
            REFSEQ[data] = sorted(seqs)
        if len(seqs) != NSEQ[data] or sorted(seqs) != REFSEQ[data]:
            miss.append(th + f"({len(seqs)} seqs)"); continue
        order = {q: i for i, q in enumerate(seqs)}
        d[th] = X[[order[q] for q in REFSEQ[data]]]
    return d, ("missing/bad thresholds " + ",".join(miss) if miss else "")


def _run_exists(data: str, tag: str, source: str) -> bool:
    """a run counts (and is excluded if incomplete) when any of its tracker outputs exists"""
    if source in ("stats", "auto"):
        d = os.path.join(stats_root(), data)
        if os.path.isdir(d) and any(f.startswith(tag + "__") and f.endswith(".npz") for f in os.listdir(d)):
            return True
        if source == "stats":
            return False
    return any(os.path.isdir(os.path.join(eval_root(data), f"{tag}__{t}_t0.5")) for t in TRACKERS)


# ---- thresholds -----------------------------------------------------------------------------------------------------------
def folds_for(data: str) -> np.ndarray:
    """fold id per reference sequence: BDD 10 fixed video folds, else one fold per sequence"""
    S = NSEQ[data]
    if data == "bdd":
        perm = np.random.default_rng(FOLD_SEED).permutation(S)
        f = np.empty(S, int); f[perm] = np.arange(S) % BDD_FOLDS
        return f
    return np.arange(S)


def clusters_for(data: str) -> np.ndarray:
    """recording clusters per reference sequence: DSEC drives (x_00_a and x_00_b are one recording), VisDrone source
    videos uav<id> (17 test-dev clips from 14 videos); identity elsewhere"""
    seqs = REFSEQ[data]
    if data == "visdrone":
        rec = [q.split("_")[0] for q in seqs]
    elif data == "dsec":
        rec = [re.sub(r"_[a-z]$", "", q) for q in seqs]
    else:
        return np.arange(len(seqs))
    u = sorted(set(rec))
    return np.array([u.index(r) for r in rec])


def pick(scores: dict[str, float]) -> str:
    """threshold with the highest score; ties: closest to 0.5, then the higher value"""
    m = max(scores.values())
    c = [t for t, v in scores.items() if v >= m - 1e-9]
    return min(c, key=lambda t: (abs(float(t) - 0.5), -float(t)))


def apply_rule(d: dict[str, np.ndarray], rule: str, folds: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """-> (X [S, 4, 19] with each sequence taken at its selected threshold, list of selected thresholds)"""
    if rule == "fixed":
        return d["0.5"], ["0.5"]
    tot = {t: d[t].sum(0) for t in d}
    if rule == "best":
        t = pick({t: float(hota(tot[t])) for t in d})
        return d[t], [t]
    out = np.empty_like(d["0.5"]); chosen = []
    for k in np.unique(folds):
        m = folds == k
        t = pick({t: float(hota(tot[t] - d[t][m].sum(0))) for t in d})
        out[m] = d[t][m]; chosen.append(t)
    return out, chosen


def method(data: str, det: str, var: str, trk: str, cls: str, source: str = "auto") -> dict:
    """one arm over its seeds -> dict(seeds, X={rule: [X per seed]}, H={rule: [pooled HOTA per seed]},
    thr={rule: [chosen thresholds per seed]}, excluded=[(tag, problem)]); VisDrone adds the rule 'cross_vid'
    (thresholds cross-fitted leave-one-source-video-out), DSEC the rule 'cross_drive' (leave-one-drive-out)"""
    key = (data, det, var, trk, cls, source)
    if key in _METHODS:
        return _METHODS[key]
    folds = None
    M = dict(seeds=[], X={r: [] for r in RULES}, H={r: [] for r in RULES}, thr={r: [] for r in RULES}, excluded=[])
    for s in SEEDS:
        tag = tag_of(data, det, var, s)
        if not _run_exists(data, tag, source):
            continue
        d, prob = load_run(data, tag, trk, cls, source)
        if prob:
            M["excluded"].append((tag, prob))
            continue
        if folds is None:
            folds = folds_for(data)
        M["seeds"].append(s)
        for r in RULES:
            X, ch = apply_rule(d, r, folds)
            M["X"][r].append(X); M["H"][r].append(float(hota(X.sum(0)))); M["thr"][r].append(ch)
        if data in CLUSTER_RULE:
            X, ch = apply_rule(d, "cross", clusters_for(data))
            for k_, v_ in (("X", X), ("H", float(hota(X.sum(0)))), ("thr", ch)):
                M[k_].setdefault(CLUSTER_RULE[data], []).append(v_)
    _METHODS[key] = M
    return M


# ---- statistics -----------------------------------------------------------------------------------------------------------
def compare(XA, XB, B: int, seed: int, delta: float, clusters=None, paired: bool = False) -> dict:
    """paired hierarchical bootstrap + approximate randomisation + TOST of arm A vs arm B.

    XA, XB: lists (one per seed) of X [S, 4, 19] at the selected thresholds. clusters: resampling / swap unit per
    sequence (None = each sequence). paired=True: XA[i] and XB[i] come from the same trained model, so the seed level is
    resampled jointly; otherwise seeds are resampled independently within each arm."""
    A = np.stack(XA); Bm = np.stack(XB)            # [n, S, 4, 19]
    nA, S = A.shape[:2]; nB = Bm.shape[0]
    cl = np.arange(S) if clusters is None else np.asarray(clusters)
    K = int(cl.max()) + 1
    rng = np.random.default_rng(seed)
    HA, HB = hota(A.sum(1)), hota(Bm.sum(1))
    d = float(HA.mean() - HB.mean())
    W = rng.multinomial(K, np.full(K, 1.0 / K), size=B).astype(float)[:, cl]    # [B, S]

    def boot(M):
        return np.stack([hota((W @ M[i].reshape(S, -1)).reshape(B, 4, 19)) for i in range(M.shape[0])], 1)
    BA, BB = boot(A), boot(Bm)
    ia = rng.integers(0, nA, (B, nA)); ib = ia if paired else rng.integers(0, nB, (B, nB))
    db = np.take_along_axis(BA, ia, 1).mean(1) - np.take_along_axis(BB, ib, 1).mean(1)
    dseq = BA.mean(1) - BB.mean(1)
    Ab, Bb = A.mean(0), Bm.mean(0)                   # seed-averaged sufficient statistics [S, 4, 19]
    T = float(hota(Ab.sum(0)) - hota(Bb.sum(0)))
    exact = 2 ** K <= B          # enumerate all 2^K swap patterns when that is no more work than B random draws
    if exact:
        Zk = ((np.arange(2 ** K)[:, None] >> np.arange(K)[None, :]) & 1).astype(float)
    else:
        Zk = rng.integers(0, 2, (B, K)).astype(float)
    Z = Zk[:, cl]
    D = (Bb - Ab).reshape(S, -1)
    sA = Ab.sum(0).reshape(1, -1) + Z @ D; sB = Bb.sum(0).reshape(1, -1) - Z @ D
    Tn = hota(sA.reshape(len(Z), 4, 19)) - hota(sB.reshape(len(Z), 4, 19))
    hit = int(np.sum(np.abs(Tn) >= abs(T) - 1e-9))
    p_ar = float(hit / len(Z)) if exact else float((1 + hit) / (B + 1))
    se = float(db.std(ddof=1))
    p_boot = float(min(1.0, 2 * min((1 + np.sum(db <= 0)) / (B + 1), (1 + np.sum(db >= 0)) / (B + 1))))
    p_lo = float((1 + np.sum(db <= -delta)) / (B + 1)); p_hi = float((1 + np.sum(db >= delta)) / (B + 1))
    hsA, hsB = hota(A).mean(0), hota(Bm).mean(0)     # per-sequence HOTA, seed mean
    return dict(nA=nA, nB=nB, A=float(HA.mean()), B=float(HB.mean()), sdA=float(HA.std(ddof=1)) if nA > 1 else None,
                sdB=float(HB.std(ddof=1)) if nB > 1 else None, d=d, T_ar=T,
                ci95=[float(v) for v in np.percentile(db, [2.5, 97.5])], ci90=[float(v) for v in np.percentile(db, [5, 95])],
                ci95_seqonly=[float(v) for v in np.percentile(dseq, [2.5, 97.5])], se=se, p_ar=p_ar, p_boot=p_boot,
                p_dec=max(p_ar, p_boot), paired=paired, ar_exact=exact,
                p_tost=max(p_lo, p_hi), delta=delta, mde80=Z80 * se, seq_wins=int(np.sum(hsA > hsB + 1e-9)),
                seq_losses=int(np.sum(hsA < hsB - 1e-9)), S=S)


def holm(ps) -> np.ndarray:
    """Holm step-down adjusted p-values (input order kept)"""
    ps = np.asarray(ps, float); m = len(ps); adj = np.empty(m); run = 0.0
    for i, j in enumerate(np.argsort(ps, kind="stable")):
        run = max(run, min(1.0, (m - i) * ps[j])); adj[j] = run
    return adj


def label(r: dict, alpha: float = 0.05, key: str = "p_ar_adj") -> str:
    """rule (f): win / loss iff the Holm-adjusted randomisation p < alpha; else equivalent iff the Holm-adjusted TOST
    p < alpha; else inconclusive"""
    if r[key] < alpha:
        lab = "win" if r["d"] > 0 else "loss"
        return lab + (" <m" if r["p_tost_adj"] < alpha else "")
    return "equivalent" if r["p_tost_adj"] < alpha else "inconclusive"


# ---- margins, descriptive splits, one contrast ----------------------------------------------------------------------------
def margin_status(data: str, source: str = "auto") -> str | None:
    """VisDrone / MOT17: 'frozen (seed-SD rule now X, reference only)' with X = 2 x the median within-arm seed SD of
    pooled ByteTrack HOTA (cross-fitted rule) over the reference arms with >= 2 seeds; None for DSEC / BDD"""
    if data not in TRANSFERRED_MARGIN:
        return None
    arms, cls = TRANSFERRED_MARGIN[data]
    sds = []
    for det, v in arms:
        H = method(data, det, v, "bytetrack", cls, source)["H"]["cross"]
        if len(H) >= 2:
            sds.append(float(np.std(H, ddof=1)))
    return "frozen" + ("" if not sds else f" (seed-SD rule now {2 * float(np.median(sds)):.4f}, reference only)")


def static_split(data: str, MA: dict, MB: dict, rule: str = "cross") -> dict:
    """MOT17, descriptive only: seed-mean pooled HOTA of A and B and d = A - B over the static-camera sequences and over
    the other sequences, each sequence taken at its threshold of `rule`"""
    seqs = REFSEQ[data]; st = np.array([q in STATIC[data] for q in seqs])
    out = {}
    for nm, m in (("static", st), ("moving", ~st)):
        hA = [float(hota(X[m].sum(0))) for X in MA["X"][rule]]; hB = [float(hota(X[m].sum(0))) for X in MB["X"][rule]]
        out[nm] = dict(seqs=[q for q, k in zip(seqs, m) if k], A=float(np.mean(hA)), B=float(np.mean(hB)),
                       d=float(np.mean(hA) - np.mean(hB)), A_seeds=hA, B_seeds=hB)
    return out


def one(data: str, det: str, a: str, b: str, cls: str = "car", trk: str = "bytetrack", rule: str = "cross",
        B: int = 10000, margin: float | None = None, source: str = "auto") -> dict:
    """one contrast A vs B -> dict(seedsA, seedsB, exclA, exclB, [margin_status], [noninferior_90, static_split (MOT17) |
    sensitivity_video (VisDrone)], then the keys of compare()). Bootstrap / randomisation seed 0; margin = MARGINS[data]
    unless given."""
    MA, MB = method(data, det, a, trk, cls, source), method(data, det, b, trk, cls, source)
    extra = {}
    if margin is not None:
        delta = margin; extra = dict(margin_status="--margin (user given)")
    else:
        delta = MARGINS[data]
        st = margin_status(data, source)
        if st is not None:
            extra = dict(margin_status=st)
    if not MA["seeds"] or not MB["seeds"]:
        raise ValueError(f"no complete runs: {tag_of(data, det, a, 0)[:-1]}* seeds {MA['seeds']} (excluded {MA['excluded']}), "
                         f"{tag_of(data, det, b, 0)[:-1]}* seeds {MB['seeds']} (excluded {MB['excluded']})")
    r = compare(MA["X"][rule], MB["X"][rule], B, 0, delta)
    if data in STATIC:
        extra["noninferior_90"] = bool(r["ci90"][0] > -delta)
        extra["static_split"] = static_split(data, MA, MB, rule)
    elif data == "visdrone":
        extra["noninferior_90"] = bool(r["ci90"][0] > -delta)
        sr = "cross_vid" if rule == "cross" else rule
        cl = clusters_for(data)
        rv = compare(MA["X"][sr], MB["X"][sr], B, 0, delta, clusters=cl)
        extra["sensitivity_video"] = dict(rule=sr + " + video-cluster bootstrap / randomisation", clusters=int(cl.max()) + 1,
                                          **{k: rv[k] for k in ("A", "B", "d", "ci95", "ci90", "p_ar", "ar_exact", "p_tost", "se")},
                                          noninferior_90=bool(rv["ci90"][0] > -delta))
    return dict(seedsA=MA["seeds"], seedsB=MB["seeds"], exclA=MA["excluded"], exclB=MB["excluded"], **extra, **r)


def cluster_sensitivity(data: str, det: str, a: str, b: str, cls: str = "car", trk: str = "bytetrack", B: int = 10000,
                        margin: float | None = None, source: str = "auto") -> dict:
    """sensitivity check of a contrast at the recording-cluster level (DSEC: 8 drives; VisDrone: 14 source videos):
    thresholds cross-fitted leave-one-cluster-out, bootstrap and randomisation over whole clusters (seed 0). Reported,
    never deciding. With K clusters the exact randomisation p is at least 2 / 2^K (DSEC: 2/256 = 0.0078)."""
    MA, MB = method(data, det, a, trk, cls, source), method(data, det, b, trk, cls, source)
    rule = CLUSTER_RULE[data]
    cl = clusters_for(data)
    delta = MARGINS[data] if margin is None else margin
    rv = compare(MA["X"][rule], MB["X"][rule], B, 0, delta, clusters=cl)
    return dict(rule=rule + " + cluster bootstrap / randomisation", clusters=int(cl.max()) + 1,
                **{k: rv[k] for k in ("A", "B", "d", "ci95", "ci90", "p_ar", "ar_exact", "p_tost", "se")},
                noninferior_90=bool(rv["ci90"][0] > -delta))


def reset() -> None:
    """forget the cached arms and reference sequence orders (e.g. after new stats files or TrackEval output)"""
    REFSEQ.clear(); _METHODS.clear()
