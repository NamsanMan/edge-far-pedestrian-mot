# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""The DSEC detector comparison from the shipped results/stats: results/DETECTORS.md, results/summary_detectors.csv and
results/summary_detectors/dsec.json regenerate identically (scripts/compare.py --detectors), cross_one() equals
protocol.one / cluster_sensitivity for two arms of one detector, the cost table matches efpmot.models.flops, and
deploy/jetson/parse_results.py regenerates the shipped latency files. Run: python -B tests/test_detectors.py"""
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
import warnings
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from efpmot.eval import protocol as P  # noqa: E402
from efpmot.eval import table as T  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
try:
    import thop  # noqa: F401
    HAVE_THOP = True
except ImportError:
    HAVE_THOP = False


def text(path: str) -> str:
    """file content with CRLF -> LF (a Windows checkout may convert line endings)"""
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read().replace("\r\n", "\n")


def close(tc, got, ref, path=""):
    """recursive equality with a 1e-9 tolerance on floats (bit-identical on the reference machine; BLAS may differ)"""
    if isinstance(ref, dict):
        tc.assertEqual(list(got), list(ref), path)
        for k in ref:
            close(tc, got[k], ref[k], f"{path}.{k}")
    elif isinstance(ref, list):
        tc.assertEqual(len(got), len(ref), path)
        for i, (g, r) in enumerate(zip(got, ref)):
            close(tc, g, r, f"{path}[{i}]")
    elif isinstance(ref, float):
        tc.assertAlmostEqual(got, ref, delta=1e-9, msg=path)
    else:
        tc.assertEqual(got, ref, path)


class TestDetectorTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        P.reset()
        cls.rows = T.run_table(T.load_contrasts(os.path.join(RES, "contrasts_detectors.yaml")), source="stats")
        cls.tmp = tempfile.TemporaryDirectory()
        T.write_detectors_md(cls.rows, os.path.join(cls.tmp.name, "DETECTORS.md"), 10000)
        T.write_detectors_csv(cls.rows, os.path.join(cls.tmp.name, "summary_detectors.csv"))
        T.write_json(cls.rows, os.path.join(cls.tmp.name, "summary_detectors"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_markdown_and_csv(self):
        for f in ("DETECTORS.md", "summary_detectors.csv"):
            self.assertEqual(text(os.path.join(self.tmp.name, f)), text(os.path.join(RES, f)), f)

    def test_json(self):
        got = json.load(open(os.path.join(self.tmp.name, "summary_detectors", "dsec.json")))
        close(self, got, json.load(open(os.path.join(RES, "summary_detectors", "dsec.json"))))

    def test_holm_families(self):   # own families: 11 contrasts per class x tracker, none shared with contrasts.yaml
        fam = {}
        for r in self.rows:
            fam.setdefault((r["cls"], r["trk"]), []).append(r)
        self.assertEqual(sorted(fam), [(c, t) for c in ("car", "pedestrian") for t in P.TRACKERS])
        self.assertTrue(all(len(v) == 11 and all(r["holm_m"] == 11 for r in v) for v in fam.values()))

    def test_pedestrian_labels(self):
        lab = {(r["det_b"], r["b"], r["trk"]): r["label"] for r in self.rows if r["cls"] == "pedestrian"}
        for trk in P.TRACKERS:
            self.assertEqual(lab[("rtd", "416", trk)], "loss")
            for k in (("v8s", "416"), ("tiny", "416"), ("v8n", "416_p2"), ("v8n", "640")):
                self.assertEqual(lab[(*k, trk)], "inconclusive", k)
            for k in (("v8n", "416"), ("y11n", "416"), ("y26n", "416"), ("nano", "416"), ("v8n", "448x352"), ("nd", "416")):
                want = "inconclusive" if trk == "bytetrack" and k in (("v8n", "448x352"), ("nd", "416")) else "win"
                self.assertEqual(lab[(*k, trk)], want, k)

    def test_cross_one_equals_one(self):   # the cross-detector path is protocol.one with the arms of two detectors
        for cls in ("car", "pedestrian"):
            r = T.cross_one("dsec", "v8n", "416s3x", "v8n", "448x352", cls=cls, trk="ocsort", B=2000, source="stats")
            self.assertEqual(r, P.one("dsec", "v8n", "416s3x", "448x352", cls=cls, trk="ocsort", B=2000, source="stats"))
            r = T.cross_one("dsec", "v8n", "416s3x", "v8n", "416", cls=cls, B=2000, source="stats", drive=True)
            self.assertEqual(r, P.cluster_sensitivity("dsec", "v8n", "416s3x", "416", cls=cls, B=2000, source="stats"))
        with self.assertRaises(ValueError):
            T.cross_one("mot17", "v8n", "416s3x", "y11n", "416")


class TestCost(unittest.TestCase):
    @unittest.skipUnless(HAVE_THOP, "needs the thop module (pip install -e .[dev])")
    def test_repository_models(self):   # T.DET_COST of the repository arms = efpmot.models (GFLOPs, params)
        warnings.filterwarnings("ignore")
        from efpmot.models import flops, make_adapter
        fam = {("v8n", "416s3x"): "yolov8n_s3x", ("v8n", "416"): "yolov8n", ("v8n", "448x352"): "yolov8n",
               ("v8n", "640"): "yolov8n", ("y11n", "416"): "yolo11n"}
        self.assertEqual(sorted(fam), sorted(k for k in T.DET_COST if k[0] in T.REPO_DETS and "_" not in k[1]))
        for (det, var), f in fam.items():
            w, h = (int(v) for v in T.arm_name("dsec", var).split()[1].split("x"))
            pa, gf = T.DET_COST[(det, var)]
            self.assertAlmostEqual(flops(f, (h, w)), gf, delta=0.0001, msg=f"{det} {var}")
            m = make_adapter(f, 8, coco=False).model
            self.assertAlmostEqual(sum(p.numel() for p in m.parameters()) / 1e6, pa, delta=0.005, msg=f"{det} {var}")


class TestLatencyFiles(unittest.TestCase):
    def test_parse_results_regenerates(self):   # deploy/jetson/parse_results.py on the stored logs = the shipped files
        spec = importlib.util.spec_from_file_location("parse_results", os.path.join(ROOT, "deploy", "jetson", "parse_results.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        res = os.path.join(ROOT, "deploy", "jetson", "results")
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(sys, "argv", ["parse_results.py", "--out", d]), contextlib.redirect_stdout(io.StringIO()):
                mod.main()
            for f in ("latency_orin_nano.csv", "latency_orin_nano_detectors.csv", "latency_orin_nano.md"):
                self.assertEqual(text(os.path.join(d, f)), text(os.path.join(res, f)), f)


if __name__ == "__main__":
    unittest.main()
