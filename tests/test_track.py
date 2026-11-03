# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""efpmot.track: tracker factory settings and the writer formats on a synthetic 3-frame detection cache (no dataset needed;
the dataset helpers are patched). Run: python -B tests/test_track.py"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from efpmot import paths  # noqa: E402
from efpmot import track  # noqa: E402
from efpmot.data import BDD_CLASSES  # noqa: E402


class TestFactory(unittest.TestCase):
    def test_bytetrack(self):
        t = track.make_tracker("bytetrack", track_thresh=0.3)
        self.assertEqual(type(t).__name__, "ByteTrack")
        self.assertAlmostEqual(t.track_thresh, 0.3); self.assertAlmostEqual(t.match_thresh, 0.8)
        self.assertEqual(t.min_hits, 1); self.assertAlmostEqual(t.iou_threshold, 0.3)

    def test_ocsort(self):
        t = track.make_tracker("ocsort", track_thresh=0.7)
        self.assertEqual(type(t).__name__, "OcSort")
        self.assertAlmostEqual(t.det_thresh, 0.7); self.assertEqual(t.max_age, 15)
        self.assertEqual(t.min_hits, 1); self.assertAlmostEqual(t.iou_threshold, 0.3); self.assertTrue(t.use_byte)

    def test_names_and_grid(self):
        self.assertEqual(track.tracker_name("dsec_v8n_416s3x_s0", "bytetrack", 0.5), "dsec_v8n_416s3x_s0__bytetrack_t0.5")
        self.assertEqual(track.tracker_name("t", "ocsort", 0.9), "t__ocsort_t0.9")
        self.assertEqual(len(track.GRID["bytetrack"]), 8); self.assertEqual(len(track.GRID["ocsort"]), 5)


def synthetic_dets():
    """3 frames: a pedestrian (score 0.9) moving right, a car (0.8), a low-score car (0.05) and a 0.3 px wide box"""
    out = []
    for i in range(3):
        out.append(np.array([[100 + 2 * i, 50, 110 + 2 * i, 80, 0.9, 0],
                             [200, 100, 260, 140, 0.8, 2],
                             [300, 100, 330, 120, 0.05, 2],
                             [400, 100, 400.3, 120, 0.9, 0]], np.float32))
    return out


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


class TestWriters(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.pop(paths.ENV_VAR, None)
        self.tmp = tempfile.TemporaryDirectory(); d = self.tmp.name.replace("\\", "/")
        cfg = os.path.join(self.tmp.name, "p.yaml")
        with open(cfg, "w", encoding="utf-8") as fh:
            fh.write(f"outputs:\n  cache: {d}/cache\n  results: {d}/results\n")
        os.environ[paths.ENV_VAR] = cfg
        paths.load(reload=True)

    def tearDown(self):
        os.environ.pop(paths.ENV_VAR, None)
        if self._env is not None:
            os.environ[paths.ENV_VAR] = self._env
        paths._CFG = paths._CFG_FILE = None
        self.tmp.cleanup()

    def cache(self, tag, data, seq, frames):
        d = paths.out("cache", tag, data)
        np.savez_compressed(os.path.join(d, seq + ".npz"), frames=np.array(frames),
                            dets=np.array(synthetic_dets(), dtype=object), allow_pickle=True)

    def test_bdd_json(self):
        self.cache("bdd_t_s0", "bdd", "seqA", ["f-1.jpg", "f-2.jpg", "f-3.jpg"])
        out = track.run("bdd", "bdd_t_s0", "bytetrack", 0.5, verbose=False)
        self.assertTrue(out.endswith("bdd_t_s0__bytetrack_t0.5"))
        js = read_json(os.path.join(out, "data", "seqA.json"))
        self.assertEqual([f["name"] for f in js], ["f-1.jpg", "f-2.jpg", "f-3.jpg"]); self.assertEqual([f["index"] for f in js], [0, 1, 2])
        labels = [lb for f in js for lb in f["labels"]]
        self.assertTrue(labels)
        self.assertEqual(set(js[0]["labels"][0]), {"id", "category", "score", "box2d"})
        for lb in labels:
            self.assertIn(lb["category"], ("pedestrian", "car"))
            self.assertEqual(lb["id"] // 100000, BDD_CLASSES.index(lb["category"]))   # id = boxmot id + 1 + class * 100000
            self.assertGreater(lb["box2d"]["x2"] - lb["box2d"]["x1"], 0.5)           # the 0.3 px box never reaches a tracker
        self.assertNotIn(0.05, [round(lb["score"], 2) for lb in labels])           # below 0.2 x thr: dropped

    def test_dsec_keeps_labelled_frames(self):
        self.cache("dsec_t_s0", "dsec", "zurich_city_00_a", ["000000.png", "000001.png", "000002.png"])
        with mock.patch("efpmot.data.dsec.labelled_names", return_value={"000000.png", "000002.png"}):
            out = track.run("dsec", "dsec_t_s0", "ocsort", 0.5, verbose=False)
        js = read_json(os.path.join(out, "data", "zurich_city_00_a.json"))
        self.assertEqual([(f["name"], f["index"]) for f in js], [("000000.png", 0), ("000002.png", 1)])

    def test_visdrone_drops_ignored_centres(self):
        self.cache("vis_t_s0", "visdrone", "uav1", ["0000001.jpg", "0000002.jpg", "0000003.jpg"])
        ign = {2: np.array([[150, 50, 300, 200]], np.float32)}   # covers the car centre in frame 2
        with mock.patch("efpmot.data.visdrone.read_annotations", return_value=([1, 2, 3], {}, ign)), \
                mock.patch("efpmot.data.visdrone.frame_paths", return_value=["a", "b", "c"]), \
                mock.patch("efpmot.data.visdrone.seq_hw", return_value=(540, 960)):
            out = track.run("visdrone", "vis_t_s0", "bytetrack", 0.5, verbose=False)
        js = read_json(os.path.join(out, "data", "uav1.json"))
        cats = [[lb["category"] for lb in f["labels"]] for f in js]
        self.assertNotIn("car", cats[1]); self.assertIn("car", cats[2])

    def test_mot17_txt(self):
        self.cache("mot_t_s0", "mot17", "MOT17-02-FRCNN", ["000302.jpg", "000303.jpg", "000304.jpg"])
        with mock.patch("efpmot.data.mot17.seq_hw", return_value=(1080, 1920)), \
                mock.patch("efpmot.data.mot17.half_offset", return_value=301), \
                mock.patch("efpmot.data.mot17.eval_frames", return_value=[302, 303, 304]):
            out = track.run("mot17", "mot_t_s0", "bytetrack", 0.5, verbose=False)
        with open(os.path.join(out, "data", "MOT17-02-FRCNN.txt")) as fh:
            lines = fh.read().splitlines()
        self.assertTrue(lines)
        for ln in lines:
            p = ln.split(",")
            self.assertEqual(len(p), 10); self.assertIn(int(p[0]), (1, 2, 3)); self.assertEqual(p[7:], ["-1", "-1", "-1"])
            self.assertLess(int(p[1]), 100000)          # pedestrian only (class 0)
            self.assertEqual(len(p[2].split(".")[1]), 2); self.assertEqual(len(p[6].split(".")[1]), 4)


if __name__ == "__main__":
    unittest.main()
