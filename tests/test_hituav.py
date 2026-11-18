# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""HIT-UAV without the dataset (CPU): recording / flight / fold rules on synthetic file names, the shipped split rebuilt from its own
recordings, the serial -> time rank mapping, AP and operating point on toy detections, and the shipped results (results/hituav,
results/HITUAV.md, the release thresholds). Run: python -B tests/test_hituav.py"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from efpmot.data import BDD_CLASSES, PED  # noqa: E402
from efpmot.data import hituav as H  # noqa: E402
from efpmot.eval import detection as D  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results", "hituav")
CAR = BDD_CLASSES.index("car")


class TestNames(unittest.TestCase):
    def test_parse_and_tags(self):
        p = H.parse_name("1_130_90_0_04077.jpg")
        self.assertEqual((p["daynight"], p["night"], p["altitude"], p["angle"], p["weather"], p["serial"], p["code"]),
                         (1, True, 130, 90, 0, 4077, "1_130_90_0"))
        self.assertEqual(H.image_name(p["code"], p["serial"]), "1_130_90_0_04077.jpg")
        with self.assertRaises(ValueError):
            H.parse_name("frame_0001.jpg")
        for tag, split in (("hit_v8n_416_s0", "official"), ("hit_v8n_416s3xrecA_s0", "recA"), ("hit_v8n_448x352recB_s1", "recB"),
                           ("hit_v8n_416altLH_s2", "altLH"), ("hit_v8n_416s3xrecA_smoke", "recA")):
            self.assertEqual(H.split_of_tag(tag), split)
            H.check_tag(tag, split)
        with self.assertRaises(ValueError):
            H.split_of_tag("hit_v8n_recA_416recB_s0")
        for tag, split in (("hit_v8n_416_s0", "recA"), ("hit_v8n_416recA_s0", "official"), ("hit_v8n_416recA_s0", "recB"),
                           ("hit_v8n_416_s0", "foldX")):
            with self.assertRaises(ValueError):
                H.check_tag(tag, split)

    def test_recordings_and_flights(self):
        names = ["1_60_30_0_00000.jpg", "1_60_30_0_00031.jpg", "1_60_30_0_00001.jpg",   # steps 1, 30: one recording
                 "1_60_30_0_00062.jpg",                                                 # step 31: a new recording, same flight
                 "1_60_40_0_00063.jpg",                                                 # angle change: new recording, same flight
                 "1_70_30_0_00070.jpg",                                                 # altitude change: new flight
                 "0_70_30_0_00080.jpg",                                                 # day / night change: new flight
                 os.path.join("any", "dir", "1_60_30_0_00090.jpg")]                     # same code as r000, later flight
        R = H.recordings_from_names(names)
        self.assertEqual([r["id"] for r in R], [f"r{i:03d}" for i in range(6)])
        self.assertEqual([r["serials"] for r in R], [[0, 1, 31], [62], [63], [70], [80], [90]])
        self.assertEqual([r["flight"] for r in R], ["f00", "f00", "f00", "f01", "f02", "f03"])
        self.assertEqual(R[0]["images"], ["1_60_30_0_00000.jpg", "1_60_30_0_00001.jpg", "1_60_30_0_00031.jpg"])
        self.assertEqual([r["code"] for r in R][:3], ["1_60_30_0", "1_60_30_0", "1_60_40_0"])
        with self.assertRaises(ValueError):   # serials are a global counter
            H.recordings_from_names(["1_60_30_0_00005.jpg", "1_60_40_0_00005.jpg"])


def synthetic_recordings(seed=0):
    """2 altitudes below and 2 above 90 m, day and night, 5-7 recordings per cell with random counts"""
    rng = np.random.default_rng(seed)
    recs, serial = [], 0
    for alt in (60, 80, 100, 120):
        for dn in (0, 1):
            for ang in H.ANGLES[: int(rng.integers(5, 8))]:
                n = int(rng.integers(3, 15))
                recs.append(dict(id=f"r{len(recs):03d}", code=f"{dn}_{alt}_{ang}_0", daynight=dn, altitude=alt, angle=ang, weather=0,
                                 n_img=n, n_person=int(rng.integers(0, 60)), n_car=int(rng.integers(0, 30)),
                                 serials=list(range(serial, serial + n))))
                serial += n + 40
    return recs


class TestFolds(unittest.TestCase):
    def test_best_halves_first_minimum(self):
        rng = np.random.default_rng(1)
        v = np.concatenate([rng.integers(0, 50, (7, 3)), rng.integers(0, 4, (7, 7))], 1)
        h = H.best_halves(v)
        tot = v.sum(0).astype(float)
        den = np.maximum(tot, 1.0)

        def cost(mask):
            s1 = v[mask].sum(0)
            d = np.abs(2 * s1 - tot)
            return d[0] / den[0] + H.W_IMG * d[1] / den[1] + H.W_CAR * d[2] / den[2] + H.W_ANG * d[3:].sum() / den[1]
        masks = [np.array([True] + [bool((m >> j) & 1) for j in range(6)]) for m in range(2 ** 6)]
        costs = [cost(m) for m in masks]
        self.assertTrue(h[0])
        self.assertTrue(np.array_equal(h, masks[int(np.argmin(costs))]))   # the first mask of minimum cost

    def test_fold_rules(self):
        recs = synthetic_recordings()
        F = H.build_folds(recs)
        ids = [r["id"] for r in recs]
        by = {r["id"]: r for r in recs}
        for f in H.FOLDS:
            parts = [F[f][p] for p in H.PARTS]
            self.assertEqual(sorted(sum(parts, [])), ids)                      # a partition of the recordings
            for p in parts:
                self.assertEqual(p, [i for i in ids if i in p])                # serial order
        self.assertEqual(sorted(F["recA"]["test"] + F["recB"]["test"]), ids)    # every recording is test exactly once
        for f, other in (("recA", "recB"), ("recB", "recA")):
            self.assertTrue(set(F[f]["val"]) <= set(F[other]["test"]))
            cells = [(by[i]["altitude"], by[i]["daynight"]) for i in F[f]["val"]]
            self.assertEqual(len(cells), len(set(cells)))                     # at most one val recording per cell
        alt = F["altLH"]
        self.assertTrue(all(by[i]["altitude"] >= 100 for i in alt["test"]) and all(by[i]["altitude"] <= 90 for i in alt["train"] + alt["val"]))
        self.assertEqual(len(alt["test"]), sum(1 for r in recs if r["altitude"] >= 100))
        import zlib
        for a in (60, 80):   # val = the crc32-ordered prefix of each altitude that first reaches 10 % of its images
            order = sorted((r for r in recs if r["altitude"] == a), key=lambda r: zlib.crc32(("altLH-" + r["id"]).encode()))
            tot = sum(r["n_img"] for r in order)
            val = [r["id"] for r in order if r["id"] in alt["val"]]
            k = len(val)
            self.assertEqual(val, [r["id"] for r in order[:k]])
            self.assertGreaterEqual(10 * sum(r["n_img"] for r in order[:k]), tot)
            self.assertLess(10 * sum(r["n_img"] for r in order[:k - 1]), tot)
        self.assertEqual(H.build_folds(recs), F)                               # deterministic

    def test_shipped_split_rebuilds(self):
        """the shipped file = build_split() of its own recordings (per-recording counts), byte for byte"""
        with open(H.SPLIT_JSON, encoding="utf-8") as f:
            txt = f.read()
        d = json.loads(txt)
        self.assertEqual((d["n_images"], d["n_recordings"], d["n_flights"], d["rec_gap"]), (2898, 264, 38, 30))
        names, n_person, n_car = [], {}, {}
        for r in d["recordings"].values():
            imgs = [H.image_name(r["code"], s) for s in r["serials"]]
            names += imgs
            n_person[imgs[0]], n_car[imgs[0]] = r["n_person"], r["n_car"]
        self.assertEqual(H.split_text(H.build_split(names[::-1], n_person, n_car)), txt)
        sizes = {f: {p: len(v) for p, v in H.fold_parts(f).items()} for f in H.FOLDS}
        self.assertEqual(sizes, {"recA": {"train": 1263, "val": 187, "test": 1448}, "recB": {"train": 1237, "val": 211, "test": 1450},
                                 "altLH": {"train": 1305, "val": 226, "test": 1367}})
        from efpmot.train import TRAIN_DEFAULTS, default_epochs
        for f in H.FOLDS:
            self.assertEqual(default_epochs("hituav", f), H.train_epochs(sizes[f]["train"]))
        self.assertEqual(default_epochs("hituav"), H.train_epochs(2029))
        self.assertEqual(TRAIN_DEFAULTS["hituav"]["split_epochs"], {"official": 72, "recA": 116, "recB": 118, "altLH": 112})
        self.assertEqual(H.recording_of("1_60_30_0_00062.jpg"), "r000")

    def test_split_option(self):
        from efpmot.data import train_dataset
        with self.assertRaises(ValueError):
            train_dataset("dsec", (320, 416), split="recA")
        from efpmot.train import _cfg
        self.assertEqual(_cfg({"family": "yolov8n_s3x", "data": "hituav", "name": "hit_v8n_416s3xrecB_s0", "split": "recB"})["epochs"], 118)
        self.assertEqual(_cfg({"family": "yolov8n", "data": "hituav", "name": "hit_v8n_416_s0"})["split"], "official")
        with self.assertRaises(ValueError):
            _cfg({"family": "yolov8n", "data": "hituav", "name": "hit_v8n_416_s0", "split": "altLH"})
        with self.assertRaises(ValueError):
            _cfg({"family": "yolov8n", "data": "dsec", "name": "x", "split": "recA"})


class TestTime(unittest.TestCase):
    def test_rank_order(self):
        self.assertEqual(H.rank_order(12).tolist(), [0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(H.rank_order(3).tolist(), [0, 1, 2])
        for m in (1, 9, 10, 31, 150, 281):
            t = H.time_indices(np.arange(500, 500 + m), 500, m)
            self.assertEqual(sorted(t.tolist()), list(range(m)))                  # a permutation of the time indices
            rank = {s: k for k, s in enumerate(sorted(str(i) for i in range(m)))}
            self.assertEqual([500 + rank[str(i)] for i in t.tolist()], list(range(500, 500 + m)))
        with self.assertRaises(ValueError):
            H.time_indices([10, 41], 10, 31)

    def test_source_video_facts(self):
        """images that equal frames 7 * i of the release's sample videos (60m-30_1 -> r000, 120m-30_3 -> r236)"""
        self.assertEqual(H.time_indices([0, 1, 62, 73, 84, 139, 2, 13], 0, 150).tolist(), [0, 1, 2, 3, 4, 9, 10, 11])
        t = H.recording_time_indices("r236")   # default m = 9295 - 9264 = 31
        self.assertEqual({int(k[-9:-4]): v for k, v in t.items()}, {9264: 0, 9289: 4, 9266: 10, 9270: 14, 9274: 18, 9278: 21, 9285: 28})
        self.assertEqual(H.recording_time_indices("r000", m=150)["1_60_30_0_00062.jpg"], 2)
        self.assertNotEqual(H.recording_time_indices("r000")["1_60_30_0_00062.jpg"], 2)   # default m = 281: a whole block is missing
        self.assertAlmostEqual(H.SECONDS_PER_INDEX, 0.9333333, places=6)


class TestAP(unittest.TestCase):
    def test_match_rules(self):
        gt = np.array([[0, 0, 10, 10]], float)
        dc = np.array([[20, 20, 30, 30]], float)
        det = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [20, 20, 30, 30], [50, 50, 60, 60]], float)   # descending score
        st, gd = D.match_image(det, gt, dc)
        self.assertEqual(st[0].tolist(), [1, 0, -1, 0])   # TP, duplicate = FP, inside DontCare = ignored, background = FP
        self.assertEqual(gd[:, 0].tolist(), [0] * len(D.IOU_THRS))
        st, gd = D.match_image(det[:1], np.array([[0, 0, 10, 10], [0, 0, 10, 10]], float), np.zeros((0, 4)))
        self.assertEqual(gd[0].tolist(), [-1, 0])         # equal IoU -> the later GT (pycocotools)
        st, _ = D.match_image(np.array([[0, 0, 10, 12.6]]), gt, np.zeros((0, 4)))   # IoU 0.794: TP up to t = 0.75
        self.assertEqual(st[:, 0].tolist(), [1, 1, 1, 1, 1, 1, 0, 0, 0, 0])

    def test_coco_ap(self):
        st = np.array([1, 0, 1], np.int8)
        self.assertAlmostEqual(D.coco_ap(np.array([0.9, 0.8, 0.7]), st, 2), 100 * (51 + 50 * 2 / 3) / 101, places=9)
        self.assertAlmostEqual(D.coco_ap(np.array([0.7, 0.8, 0.9]), st[::-1].copy(), 2), 100 * (51 + 50 * 2 / 3) / 101, places=9)
        self.assertEqual(D.coco_ap(np.array([0.9, 0.5]), np.array([1, 1], np.int8), 2), 100.0)
        self.assertAlmostEqual(D.coco_ap(np.array([0.9, 0.5]), np.array([1, -1], np.int8), 2), 100 * 51 / 101, places=9)   # ignored = no step
        self.assertIsNone(D.coco_ap(np.zeros(0), np.zeros(0, np.int8), 0))
        self.assertEqual(D.coco_ap(np.zeros(0), np.zeros(0, np.int8), 3), 0.0)

    def test_operating_point(self):
        e = dict(scores=np.array([0.9, 0.8, 0.7, 0.6]), status=np.array([[1, 0, 1, -1]], np.int8), n_gt=2,
                 gt_score=np.array([0.9, 0.7]))
        self.assertEqual(D.best_f1(e), 0.7)   # F1 0.667 / 0.5 / 0.8 at 0.9 / 0.8 / 0.7; the ignored detection is no candidate
        o = D.at_thr(e, 0.7)
        self.assertEqual((o["TP"], o["FP"], o["FN"]), (2, 1, 0))
        for k, v in (("P", 100 * 2 / 3), ("R", 100.0), ("F1", 80.0)):
            self.assertAlmostEqual(o[k], v, places=9)
        e2 = dict(e, scores=np.array([0.9, 0.9, 0.5, 0.4]), status=np.array([[1, 1, 0, 0]], np.int8), gt_score=np.array([0.9, 0.9]))
        self.assertEqual(D.best_f1(e2), 0.9)

    def test_evaluate_on_synthetic_gt(self):
        """GT boxes (+ one detection per DontCare box) as detections: AP 100 and recall 100 % in every bin; a missed box lowers it"""
        names = ["1_60_30_0_00000.jpg", "0_130_90_0_00100.jpg"]
        G = {names[0]: dict(gt=np.array([[PED, 10, 10, 15, 22], [CAR, 100, 100, 140, 120]], float),
                            dontcare=np.array([[300, 300, 320, 320]], float), meta=H.parse_name(names[0]), hw=(512, 640)),
             names[1]: dict(gt=np.array([[PED, 50, 50, 58, 70], [PED, 200, 200, 220, 240]], float), dontcare=np.zeros((0, 4)),
                            meta=H.parse_name(names[1]), hw=(512, 640))}
        dets = {n: np.array([[*g[1:5], 0.9 - 0.1 * k, g[0]] for k, g in enumerate(G[n]["gt"])], np.float32) for n in names}
        dets[names[0]] = np.concatenate([dets[names[0]], [[300, 300, 320, 320, 0.95, PED]]]).astype(np.float32)
        orig = D.H.eval_gt
        D.H.eval_gt = lambda part, split: G
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.makedirs(os.path.join(tmp, "hit_v8n_416recA_s0", "hituav"))
                for p in H.EXPORT_PARTS:
                    np.savez_compressed(os.path.join(tmp, "hit_v8n_416recA_s0", "hituav", p + ".npz"), frames=np.array(names),
                                        dets=np.array([dets[n] for n in names], dtype=object))
                r = D.evaluate_hituav("hit_v8n_416recA_s0", cache_root=tmp)
                self.assertEqual(r["split"], "recA")
                for p in ("test", "val"):
                    for c in ("pedestrian", "car"):   # precision tp / (tp + fp + eps), as pycocotools: 100 up to rounding
                        self.assertAlmostEqual(r["ap"][p][c]["AP50"], 100.0, places=9)
                        self.assertAlmostEqual(r["ap"][p][c]["AP50_95"], 100.0, places=9)
                self.assertEqual(r["ap"]["test"]["pedestrian"]["n_ignored50"], 1)
                rec = r["op"]["pedestrian"]["test_recall"]
                self.assertEqual({k: v["recall"] for k, v in rec["daynight"].items()}, {"day": 100.0, "night": 100.0})
                self.assertEqual({k: v["n"] for k, v in rec["short_side"].items()},   # short sides 5, 8, 20 px
                                 {"<6": 1, "6-8": 0, "8-10": 1, "10-12": 0, "12-16": 0, ">=16": 1})
                self.assertEqual(rec["altitude"]["130"]["n"], 2)
                with self.assertRaises(ValueError):
                    D.evaluate_hituav("hit_v8n_416recA_s0", split="recB", cache_root=tmp)
                dets[names[1]] = dets[names[1]][:1]          # one person missed
                np.savez_compressed(os.path.join(tmp, "hit_v8n_416recA_s0", "hituav", "test.npz"), frames=np.array(names),
                                    dets=np.array([dets[n] for n in names], dtype=object))
                r = D.evaluate_hituav("hit_v8n_416recA_s0", cache_root=tmp)
                self.assertAlmostEqual(r["op"]["pedestrian"]["test"]["R"], 100 * 2 / 3)
        finally:
            D.H.eval_gt = orig


def _have_dataset():
    from efpmot import paths
    p = paths.get("data.hituav")
    return bool(p) and os.path.isdir(os.path.join(p, "normal_json"))


@unittest.skipUnless(_have_dataset(), "data.hituav not configured")
class TestDataset(unittest.TestCase):
    def test_split_from_dataset(self):
        with open(H.SPLIT_JSON, encoding="utf-8") as f:
            self.assertEqual(H.split_text(H.split_from_dataset()), f.read())

    def test_counts(self):
        n = {p: len(H.list_images(p)) for p in H.PARTS}
        self.assertEqual(n, {"train": 2029, "val": 290, "test": 579})
        cats = {p: np.bincount(np.concatenate([a["boxes"][:, 0] for a in H.read_annotations(p).values()]).astype(int), minlength=5).tolist()
                for p in H.PARTS}
        self.assertEqual(cats, {"train": [8533, 5253, 3630, 102, 110], "val": [1168, 719, 554, 12, 7], "test": [2611, 1339, 796, 34, 31]})
        g = H.eval_gt("test", "recA")
        self.assertEqual(len(g), 1448)
        self.assertEqual(sum(int((v["gt"][:, 0] == PED).sum()) for v in g.values()), 6301)
        ds = H.HituavImages(size_hw=(320, 416), train=False, split="recB")
        self.assertEqual((len(ds), ds.n_boxes, ds.n_dontcare), (1237, 11332, 119))
        cur, prev, tgt, ptgt = ds[0]
        self.assertEqual(tuple(cur.shape), (3, 320, 416))
        self.assertTrue(cur.data_ptr() == prev.data_ptr() and tgt.data_ptr() == ptgt.data_ptr())   # one image, returned twice


class TestShippedResults(unittest.TestCase):
    def test_results_and_release_thresholds(self):
        from efpmot.eval import hituav_table as T
        from efpmot.release import RELEASES
        files = sorted(f for f in os.listdir(RESULTS) if f.endswith(".json"))
        self.assertGreater(len(files), 0)
        res = {}
        for f in files:
            with open(os.path.join(RESULTS, f), encoding="utf-8") as fh:
                r = json.load(fh)
            res[r["tag"]] = r
            self.assertEqual(f, r["tag"] + ".json")
            self.assertEqual(r["split"], H.split_of_tag(r["tag"]))
            self.assertEqual(r["arm"], T.arm_of(r["tag"], r["family"], r["hw"]))
            self.assertTrue(r["tag"].endswith(f"_s{r['seed']}"))
            self.assertEqual(set(r["ap"]), {"test", "val"})
        for fn, m in RELEASES.items():
            if m["dataset"].startswith("hituav"):
                r = res[m["run"]]
                self.assertEqual(m["thr"], r["op"]["pedestrian"]["thr"])
                self.assertEqual((m["family"], list(m["hw"]), m["seed"], m["epochs"]), (r["family"], r["hw"], r["seed"], r["epochs"]))
                self.assertEqual(m["dataset"], f"hituav-{r['split']}")
        with open(os.path.join(REPO, "results", "HITUAV.md"), encoding="utf-8") as f:
            self.assertEqual(T.markdown(T.load_results(RESULTS)), f.read())   # the page is regenerated from these files


if __name__ == "__main__":
    unittest.main()
