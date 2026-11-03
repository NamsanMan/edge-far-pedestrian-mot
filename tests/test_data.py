# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""efpmot.data: letterbox, target packing and augmentation of FramePairs on synthetic frames, VisDrone ignore rules, MOT17 half
split; dataset-dependent cases call skipTest when the paths (or the derived files) are not configured.
Run: python -B tests/test_data.py"""
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from efpmot import paths  # noqa: E402
from efpmot.data import BDD_CLASSES, CAR, PED, FramePairs, get, iou_matrix, letterbox  # noqa: E402
from efpmot.data.visdrone import centre_in, filter_ignored, mask_regions  # noqa: E402


class Synthetic(FramePairs):
    def __init__(self, d, train):
        img = np.full((480, 640, 3), 100, np.uint8)
        p = os.path.join(d, "f.png")
        cv2.imwrite(p, img)
        self.items = [("s", 0, p, p), ("s", 1, p, p)]
        self.labels = {"s": {0: [(PED, 10.0, 20.0, 30.0, 80.0, 7)], 1: [(CAR, 600.0, 400.0, 700.0, 500.0, 3), (PED, 1.0, 1.0, 1.5, 9.0, 4)]}}
        self.size, self.train, self.max_objs = (320, 416), train, 120


def _usable(*keys):
    return all(paths.get(k) and os.path.exists(paths.get(k)) for k in keys)


class TestData(unittest.TestCase):
    def test_classes(self):
        self.assertEqual(BDD_CLASSES[PED], "pedestrian")
        self.assertEqual(BDD_CLASSES[CAR], "car")
        self.assertEqual(len(BDD_CLASSES), 8)

    def test_letterbox(self):
        img = np.zeros((480, 640, 3), np.uint8)
        out, r = letterbox(img, (320, 416))
        self.assertEqual(out.shape, (320, 416, 3))
        self.assertAlmostEqual(r, 0.65)
        self.assertTrue((out[312:] == 114).all())  # 480 * 0.65 = 312 image rows, padding below
        self.assertTrue((out[:312] == 0).all())

    def test_iou(self):
        a = np.array([[0, 0, 10, 10]], float)
        b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
        self.assertTrue(np.allclose(iou_matrix(a, b), [[1, 1 / 3, 0]], atol=1e-6))
        self.assertEqual(iou_matrix(a[:0], b).shape, (0, 3))

    def test_pairs_eval(self):
        with tempfile.TemporaryDirectory() as d:
            ds = Synthetic(d, train=False)
            cur, prev, tgt, ptgt = ds[0]
            self.assertEqual(tuple(cur.shape), (3, 320, 416))
            self.assertEqual(tuple(tgt.shape), (120, 6))
            r = 0.65
            self.assertTrue(np.allclose(tgt[0].numpy(), [PED, 20 * r, 50 * r, 20 * r, 60 * r, 7], atol=1e-4))
            self.assertEqual(float(tgt[1, 5]), -1.0)
            cur, prev, tgt, ptgt = ds[1]
            # the car (390, 260, 455, 325 on the canvas) is clipped to the 416 x 320 canvas; the 0.325 px wide pedestrian is dropped
            self.assertTrue(np.allclose(tgt[0, :5].numpy(), [CAR, (390 + 416) / 2, (260 + 320) / 2, 26, 60], atol=1e-3))
            self.assertEqual(int((tgt[:, 3] > 0).sum()), 1)
            self.assertTrue(np.allclose(ptgt[0].numpy(), [PED, 20 * r, 50 * r, 20 * r, 60 * r, 7], atol=1e-4))  # prev = key 0

    def test_pairs_without_prev(self):
        """load_prev = False gives the same current frame, targets and random state as the full pair"""
        with tempfile.TemporaryDirectory() as d:
            ds = Synthetic(d, train=True)
            for i in (0, 1):
                random.seed(5); np.random.seed(5)
                ds.load_prev = True
                full = ds[i]
                st = (random.random(), float(np.random.uniform()))
                random.seed(5); np.random.seed(5)
                ds.load_prev = False
                cur = ds[i]
                self.assertEqual(st, (random.random(), float(np.random.uniform())))
                self.assertTrue(torch.equal(full[0], cur[0]) and torch.equal(full[2], cur[2]))
                self.assertTrue(torch.equal(cur[0], cur[1]) and torch.equal(cur[2], cur[3]))

    def test_pairs_train_rng(self):
        """the augmentation draws random.random, random.uniform and np.random.uniform once each per item, in this order"""
        with tempfile.TemporaryDirectory() as d:
            ds = Synthetic(d, train=True)
            random.seed(3)
            np.random.seed(3)
            a = ds[0]
            random.seed(3)
            np.random.seed(3)
            flip, sc = random.random() < 0.5, random.uniform(0.8, 1.2)
            gain = np.random.uniform(0.7, 1.3)
            nxt = (random.random(), np.random.uniform())
            random.seed(3)
            np.random.seed(3)
            ds[0]
            self.assertEqual((random.random(), np.random.uniform()), nxt)
            img = np.full((480, 640, 3), 100, np.uint8)
            if abs(sc - 1) > 0.02:
                img = cv2.resize(img, None, fx=sc, fy=sc)
            img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
            ref, r = letterbox(img, (320, 416))
            self.assertTrue(np.array_equal(a[0].numpy().transpose(1, 2, 0), ref.astype(np.float32)))
            k = sc if abs(sc - 1) > 0.02 else 1.0
            x1, x2 = ((640 - 30) * k, (640 - 10) * k) if flip else (10 * k, 30 * k)
            self.assertAlmostEqual(float(a[2][0, 1]), (x1 + x2) / 2 * r, places=3)

    def test_visdrone_rules(self):
        boxes = np.array([[0, 0, 10, 10], [20, 20, 40, 40]], float)
        reg = np.array([[0, 0, 6, 6]], float)
        self.assertEqual(centre_in(boxes, reg).tolist(), [True, False])
        rows = np.array([[0, 0, 10, 10, 1, 0.9], [20, 20, 40, 40, 2, 0.8]])
        self.assertEqual(filter_ignored(rows, reg).shape[0], 1)
        img = np.zeros((10, 10, 3), np.uint8)
        out = mask_regions(img, [[0, 0, 6, 6]], [[2, 2, 4, 4]])
        self.assertEqual(int(out[0, 0, 0]), 114)
        self.assertEqual(int(out[3, 3, 0]), 0)  # inside a kept box
        self.assertEqual(int(out[8, 8, 0]), 0)

    def test_mot17_half_split(self):
        if not _usable("data.mot17_train"):
            self.skipTest("data.mot17_train not configured")
        m = get("mot17")
        self.assertEqual(sum(m.half_offset(s) for s in m.SEQS), 2664)
        self.assertEqual(sum(len(m.eval_frames(s)) for s in m.SEQS), 2652)
        s = m.SEQS[0]
        self.assertEqual(m.train_frames(s)[-1] + 1, m.eval_frames(s)[0])
        ds = m.Mot17Pairs(m.SEQS, (256, 416), train=False)
        self.assertEqual(len(ds), 2664)
        cur, prev, tgt, _ = ds[10]
        self.assertTrue((tgt[:, 0][tgt[:, 3] > 0] == PED).all())

    def test_dsec_index(self):
        if not _usable("data.dsec_det") or not os.path.isfile(os.path.join(paths.require("outputs.derived"), "dsec_det", "frame_index.json")):
            self.skipTest("DSEC-Det or its derived frame index (scripts/prepare_data.py dsec) not available")
        m = get("dsec")
        self.assertEqual((len(m.split_seqs("train")), len(m.split_seqs("val")), len(m.split_seqs("test"))), (41, 6, 13))
        ds = m.DsecPairs(m.split_seqs("train"), (320, 416), train=False, stride=2)
        self.assertEqual(len(ds), 24168)
        self.assertEqual(len(m.eval_seqs()), 19)

    def test_visdrone_bdd_counts(self):
        if _usable("data.visdrone_mot"):
            v = get("visdrone")
            self.assertEqual(len(v.list_seqs("train")), 56)
            self.assertEqual(len(v.eval_seqs()), v.NSEQ_TEST)
        if _usable("data.bdd_mot_val_gt"):
            self.assertEqual(len(get("bdd").list_val_seqs()), 200)
        with open(get("bdd").SUBSET_LIST) as f:
            self.assertEqual(sum(1 for ln in f if ln.strip()), 35000)


if __name__ == "__main__":
    unittest.main()
