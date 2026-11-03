# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""efpmot.eval.protocol.compare / holm / hota on synthetic data against golden values produced by the reference
implementation of the protocol (tests/data/compare_golden.json). Run: python -B tests/test_compare.py"""
import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from efpmot.eval import protocol as P  # noqa: E402

GOLD = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "compare_golden.json")))


def synth(rng, n, S):
    """n seeds x S sequences of sufficient statistics [S, 4, 19] (TP, FN, FP, AssA * TP)"""
    tp = rng.integers(50, 400, (n, S, 19)).astype(float); fn = rng.integers(10, 300, (n, S, 19)).astype(float)
    fp = rng.integers(5, 200, (n, S, 19)).astype(float); ass = rng.uniform(0.2, 0.9, (n, S, 19))
    return [np.stack([tp[i], fn[i], fp[i], ass[i] * tp[i]], 1) for i in range(n)]


class TestCompare(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(20261102)   # same draw order as the golden file
        cls.XA, cls.XB = synth(rng, 2, 6), synth(rng, 3, 6)
        cls.YA, cls.YB = synth(rng, 2, 12), synth(rng, 2, 12)

    def assertSame(self, got, ref):
        self.assertEqual(set(got), set(ref))
        for k, v in ref.items():
            g = got[k]
            if isinstance(v, list):
                np.testing.assert_allclose(g, v, rtol=0, atol=1e-9, err_msg=k)
            elif isinstance(v, float):
                self.assertAlmostEqual(g, v, delta=1e-9, msg=k)
            else:
                self.assertEqual(g, v, k)

    def test_exact_randomisation(self):   # 2^6 swap patterns <= B: exact enumeration
        r = P.compare(self.XA, self.XB, 2000, 0, 1.0)
        self.assertTrue(r["ar_exact"])
        self.assertSame(r, GOLD["exact"])

    def test_clusters(self):
        self.assertSame(P.compare(self.XA, self.XB, 2000, 7, 0.5, clusters=[0, 0, 1, 2, 2, 3]), GOLD["clusters"])

    def test_random_randomisation(self):  # 2^12 > B: random swap patterns
        r = P.compare(self.YA, self.YB, 2000, 3, 1.1955)
        self.assertFalse(r["ar_exact"])
        self.assertSame(r, GOLD["random"])

    def test_paired(self):
        self.assertSame(P.compare(self.YA, self.YB, 1000, 0, 1.0, paired=True), GOLD["paired"])

    def test_deterministic(self):
        self.assertEqual(P.compare(self.XA, self.XB, 500, 1, 1.0), P.compare(self.XA, self.XB, 500, 1, 1.0))

    def test_holm(self):
        np.testing.assert_allclose(P.holm([0.01, 0.04, 0.03, 0.5]), GOLD["holm"], atol=1e-15)

    def test_hota(self):
        self.assertAlmostEqual(float(P.hota(self.XA[0].sum(0))), GOLD["hota"], delta=1e-9)


if __name__ == "__main__":
    unittest.main()
