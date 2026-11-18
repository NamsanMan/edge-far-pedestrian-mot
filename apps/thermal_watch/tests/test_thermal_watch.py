# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Dataset-free tests of apps/thermal_watch (CPU): distance geometry, one alert per confirmed person, re-acquisition keeps the
number, both screens render, the sample-video table is consistent.
python -B apps/thermal_watch/tests/test_thermal_watch.py"""
import json
import os
import sys
import unittest

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
import thermal_watch as TW  # noqa: E402

META = dict(stem="synthetic", alt=70.0, angle=90.0, daynight=1, rec=None, test_fold=None)


def det(x, y, w=10.0, h=10.0, score=0.9):
    return np.array([[x, y, x + w, y + h, score, TW.PED]], np.float32)


NONE = np.zeros((0, 6), np.float32)


class Geometry(unittest.TestCase):
    def test_slant_range(self):
        c = (315.0, 251.0, 325.0, 261.0)  # box centred on the image centre (320, 256)
        self.assertAlmostEqual(TW.slant_range(c, dict(alt=70.0, angle=90.0)), 70.0, places=6)
        self.assertAlmostEqual(TW.slant_range(c, dict(alt=130.0, angle=60.0)), 130.0 / np.sin(np.radians(60.0)), places=6)
        top = TW.slant_range((315.0, 18.0, 325.0, 28.0), dict(alt=130.0, angle=60.0))  # higher in the image = farther
        self.assertTrue(160.0 < top < 175.0, top)
        self.assertIsNone(TW.slant_range(c, dict(alt=None, angle=None)))


class Alerts(unittest.TestCase):
    def run_seq(self, seq):
        w = TW.ThermalWatch(lambda f: NONE, thr=0.3, meta=META, fps=7.5, cmc=False)
        frame = np.zeros((512, 640, 3), np.uint8)
        for d in seq:
            w.step(frame, d)
        return w

    def test_one_alert_after_confirmation(self):
        w = self.run_seq([det(100, 100)] * 2)
        self.assertEqual(len(w.events), 0)              # seen in 2 of the last 5 frames: still a candidate
        w = self.run_seq([det(100 + i, 100) for i in range(12)])
        self.assertEqual(len(w.events), 1)             # one alert, however long the person stays
        self.assertEqual(w.events[0]["id"], 1)
        self.assertEqual(w.events[0]["range_m"], round(TW.slant_range(w.events[0]["box"], META)))  # 71 m: off the image centre

    def test_reacquired_person_keeps_number(self):
        seq = [det(100, 100)] * 10 + [NONE] * 20 + [det(103, 101)] * 6   # lost longer than the tracker buffer, back in place
        w = self.run_seq(seq)
        self.assertEqual(len(w.events), 1)
        self.assertGreaterEqual(w.relinked, 1)
        self.assertEqual(sorted(set(w.disp.values())), [1])

    def test_far_new_person_alerts(self):
        seq = [det(100, 100)] * 10 + [NONE] * 20 + [det(400, 300)] * 6   # reappears far away: a new person
        w = self.run_seq(seq)
        self.assertEqual([e["id"] for e in w.events], [1, 2])


class Screens(unittest.TestCase):
    def test_live_and_review_render(self):
        w = TW.ThermalWatch(lambda f: NONE, thr=0.3, meta=META, fps=7.5, cmc=False)
        frame = np.full((512, 640, 3), 90, np.uint8)
        con = TW.Console(META, 12, 7.5)
        for i in range(12):
            img = con.render(frame, w.step(frame, det(100 + i, 100)), w, sub=0.5)
        self.assertEqual(img.shape, (720, 1280, 3))
        rec = TW.Recording.__new__(TW.Recording)  # a recording without a video file
        rec.jpgs = [TW.cv2.imencode(".jpg", frame)[1].tobytes()] * 3
        rec.shown = [[], [dict(id=1, raw=1, box=(100.0, 100.0, 110.0, 110.0), range=70.0)], []]
        rec.events, rec.meta, rec.fps, rec.name = w.events, META, 7.5, "synthetic"
        scr = TW.ReviewScreen(rec)
        img = scr.render(1, playing=True, offset=0, sel=0, cursor=(500, 400), click_age=1.0, sub=0.25)
        self.assertEqual(img.shape, (720, 1280, 3))
        self.assertEqual(scr.current(2, None), 0)


class SampleVideos(unittest.TestCase):
    def test_table(self):
        v = json.load(open(TW.SAMPLE_VIDEOS, encoding="utf-8"))["videos"]
        self.assertEqual(len(v), 5)
        for stem, m in v.items():
            self.assertIn(m["test_fold"], ("recA", "recB"))
            alt, ang = stem.split("m-")[0], stem.split("-")[1].split("_")[0]
            self.assertEqual((int(alt), int(ang)), (m["altitude"], m["angle"]))
            self.assertTrue(all(int(f) % 7 == 0 for f in m["labelled_frames"]))  # images = every 7th frame of the 7.5 fps video


if __name__ == "__main__":
    unittest.main()
