# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""efpmot.eval.protocol: HOTA recombination, threshold rules, stats files and one() against stored reference outputs
(tests/data/one_golden.json). Needs only the shipped results/stats. Run: python -B tests/test_protocol.py"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from efpmot.eval import protocol as P  # noqa: E402
from efpmot.eval import stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = json.load(open(os.path.join(HERE, "data", "one_golden.json")))


def close(tc, got, ref, path=""):
    """recursive equality with a 1e-9 tolerance on floats (bit-identical on the reference machine; BLAS may differ)"""
    if isinstance(ref, dict):
        tc.assertEqual(set(got), set(ref), path)
        for k in ref:
            close(tc, got[k], ref[k], f"{path}.{k}")
    elif isinstance(ref, list):
        tc.assertEqual(len(got), len(ref), path)
        for i, (g, r) in enumerate(zip(got, ref)):
            close(tc, g, r, f"{path}[{i}]")
    elif isinstance(ref, float) and not isinstance(ref, bool):
        tc.assertAlmostEqual(got, ref, delta=1e-9, msg=path)
    else:
        tc.assertEqual(got, ref, path)


class TestHota(unittest.TestCase):
    def test_hand_computation(self):
        X = np.zeros((4, 19)); X[0] = 80; X[1] = 20; X[2] = 0; X[3] = 0.5 * 80   # DetA 0.8, AssA 0.5 at every alpha
        self.assertAlmostEqual(float(P.hota(X)), 100 * np.sqrt(0.8 * 0.5), places=12)
        X[:, 10:] = 0                                                             # empty alphas count as 0
        self.assertAlmostEqual(float(P.hota(X)), 100 * np.sqrt(0.4) * 10 / 19, places=12)

    def test_pooling_is_sum_of_statistics(self):
        rng = np.random.default_rng(0)
        X = np.stack([np.stack([t, f, p, a * t]) for t, f, p, a in zip(rng.integers(1, 9, (3, 19)).astype(float),
                     rng.integers(0, 9, (3, 19)).astype(float), rng.integers(0, 9, (3, 19)).astype(float), rng.uniform(0, 1, (3, 19)))])
        tp, fn, fp, at = X.sum(0)
        ref = 100 * np.mean(np.sqrt(tp / np.maximum(1, tp + fn + fp) * at / np.maximum(1, tp)))
        self.assertAlmostEqual(float(P.hota(X.sum(0))), ref, places=12)


class TestRules(unittest.TestCase):
    def test_pick_ties(self):
        self.assertEqual(P.pick({"0.4": 1.0, "0.6": 1.0, "0.7": 0.5}), "0.6")   # closest to 0.5, then the higher value
        self.assertEqual(P.pick({"0.2": 2.0, "0.5": 1.0}), "0.2")

    def test_cross_fitting_never_uses_the_scored_sequence(self):
        S = 4; d = {}
        for i, t in enumerate(("0.4", "0.5", "0.6")):
            X = np.zeros((S, 4, 19)); X[:, 0] = 10; X[:, 1] = 10; X[:, 3] = 10
            X[i % S, 0] = 100            # threshold i is excellent on sequence i only
            X[:, 3] = X[:, 0]            # AssA = 1
            d[t] = X
        X, chosen = P.apply_rule(d, "cross", np.arange(S))
        self.assertEqual(chosen[0], "0.5")   # held out: 0.4 (excellent on sequence 0 only) is not chosen for it
        self.assertEqual(chosen[1], "0.6")   # 0.4 / 0.6 tie on the other sequences: closest to 0.5, then the higher
        self.assertEqual(chosen[2], "0.5")
        self.assertEqual(P.apply_rule(d, "fixed", np.arange(S))[1], ["0.5"])

    def test_bdd_folds(self):
        f = P.folds_for("bdd")
        self.assertEqual(sorted(np.bincount(f)), [20] * 10)
        np.testing.assert_array_equal(P.folds_for("dsec"), np.arange(13))

    def test_tag_of(self):
        self.assertEqual(P.tag_of("dsec", "v8n", "416s3x", 1), "dsec_v8n_416s3x_s1")
        self.assertEqual(P.tag_of("mot17", "v8n", "480x256", 0), "mot_v8n_480x256_s0")
        self.assertEqual(P.tag_of("dsec", "v8n", "416_p2", 0), "dsec_v8n_416_s0_p2")


class TestStats(unittest.TestCase):
    def test_shipped_stats_recombine(self):
        root = P.stats_root(); n = 0
        for data in os.listdir(root):
            for f in os.listdir(os.path.join(root, data)):
                tag, trk, cls = f[:-4].split("__")
                seqs, ths, X, comb = stats.read(data, tag, trk, cls)
                self.assertEqual(seqs, sorted(seqs)); self.assertEqual(len(seqs), P.NSEQ[data])
                self.assertEqual(ths, P.GRID[trk])
                for i in range(len(ths)):
                    self.assertLess(abs(float(P.hota(X[i].sum(0))) - comb[i]), 1e-6)
                n += 1
        self.assertGreater(n, 0)

    def test_extract_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            ev = os.path.join(d, "eval"); seqs = ["b", "a"]
            rng = np.random.default_rng(1)
            for th in P.GRID["ocsort"]:
                os.makedirs(os.path.join(ev, f"t_s0__ocsort_t{th}"))
                rows, Xs = [], []
                for q in seqs:
                    tp = rng.integers(1, 50, 19).astype(float); fn = rng.integers(0, 50, 19).astype(float)
                    fp = rng.integers(0, 50, 19).astype(float); ass = rng.uniform(0, 1, 19).round(6)
                    Xs.append(np.stack([tp, fn, fp, ass * tp])); rows.append((q, tp, fn, fp, ass))
                comb = float(P.hota(np.stack(Xs).sum(0)))
                cols = ["seq"] + [f"{k}___{a}" for k in ("HOTA_TP", "HOTA_FN", "HOTA_FP", "AssA") for a in P.AL] + ["HOTA___AUC"]
                with open(os.path.join(ev, f"t_s0__ocsort_t{th}", "car_detailed.csv"), "w") as fh:
                    fh.write(",".join(cols) + "\n")
                    for q, tp, fn, fp, ass in rows:
                        fh.write(",".join([q] + [repr(float(v)) for v in (*tp, *fn, *fp, *ass)] + ["0"]) + "\n")
                    fh.write(",".join(["COMBINED"] + ["0"] * 76 + [repr(comb / 100)]) + "\n")
            f = stats.extract("dsec", "t_s0", "ocsort", "car", eval_root=ev, out_root=os.path.join(d, "stats"))
            s, ths, X, cb = stats.read("dsec", "t_s0", "ocsort", "car", root=os.path.join(d, "stats"))
            self.assertTrue(os.path.isfile(f)); self.assertEqual(s, ["a", "b"]); self.assertEqual(ths, P.GRID["ocsort"])
            for i, th in enumerate(ths):
                q, Xr, c = P.read_detailed(os.path.join(ev, f"t_s0__ocsort_t{th}", "car_detailed.csv"))
                np.testing.assert_array_equal(X[i], Xr[[q.index("a"), q.index("b")]])
                self.assertEqual(cb[i], c)


class TestOne(unittest.TestCase):
    """one() from the shipped stats equals the stored outputs of the reference implementation (full B = 10000)"""

    def setUp(self):
        P.reset()

    def check(self, key):
        data, det, a, b, cls, trk = key.split("__")
        got = json.loads(json.dumps(P.one(data, det, a, b, cls=cls, trk=trk, source="stats")))
        ref = GOLD[key]
        self.assertEqual(list(got), list(ref))
        close(self, got, ref, key)

    def test_dsec(self):
        self.check("dsec__v8n__416s3x__416__pedestrian__bytetrack")

    def test_mot17_static_split(self):
        self.check("mot17__v8n__416s3x__416__pedestrian__ocsort")

    def test_visdrone_video_sensitivity(self):
        self.check("visdrone__v8n__640s3x__640__pedestrian__bytetrack")

    def test_bdd_ten_folds(self):
        self.check("bdd__v8n__416s3x__416__car__ocsort")


if __name__ == "__main__":
    unittest.main()
