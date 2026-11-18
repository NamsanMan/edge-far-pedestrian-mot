# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Release-file table (efpmot.release): names, run tags and settings agree with each other and with the packager.

usage: python -B tests/test_release.py
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from efpmot.models import FAMILIES  # noqa: E402
from efpmot.release import RELEASES, RUN_TO_FILE, dataset_of_run, release_meta  # noqa: E402
from efpmot.train import TRAIN_DEFAULTS, default_epochs  # noqa: E402


class _Args:   # a non-tensor object, as in the "args" of a training checkpoint
    family = "yolov8n"


class TestRelease(unittest.TestCase):
    def test_unsafe_load_only_for_non_release_files(self):
        import pickle
        import tempfile
        import warnings

        import torch

        from efpmot.models import load_trusted
        obj = {"model": {"w": torch.zeros(2)}, "args": _Args()}
        with tempfile.TemporaryDirectory() as d:
            rel = os.path.join(d, next(iter(RELEASES)))
            torch.save(obj, rel)
            with self.assertRaises(pickle.UnpicklingError):
                load_trusted(rel)                       # a release file never takes the full unpickler
            own = os.path.join(d, "last.pth")
            torch.save(obj, own)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                ck = load_trusted(own)                  # an own training checkpoint: allowed, with a warning
            self.assertTrue(any("full unpickler" in str(x.message) for x in w))
            self.assertEqual(ck["args"].family, "yolov8n")
            torch.save({"w": torch.ones(1)}, own)       # a bare state dict loads safely, without a warning
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                self.assertTrue(torch.equal(load_trusted(own)["w"], torch.ones(1)))
            self.assertFalse(any("full unpickler" in str(x.message) for x in w))

    def test_table_is_consistent(self):
        self.assertEqual(len(RUN_TO_FILE), len(RELEASES))
        for fn, m in RELEASES.items():
            self.assertIn(m["family"], FAMILIES)
            self.assertEqual(dataset_of_run(m["run"]), m["dataset"])
            data, _, split = m["dataset"].partition("-")   # HIT-UAV: 'hituav-<fold>'
            self.assertEqual(tuple(m["hw"]), tuple(TRAIN_DEFAULTS[data]["hw"]))
            self.assertEqual(m["epochs"], default_epochs(data, split or None))
            self.assertTrue(m["run"].endswith(f"_s{m['seed']}"))
            self.assertEqual(release_meta(os.path.join("any", "dir", fn))["run"], m["run"])
            if data == "hituav":   # the app's person threshold (val max F1, results/hituav/<run>.json)
                self.assertTrue(0.0 < m["thr"] < 1.0)
        self.assertIsNone(release_meta("last.pth"))

    def test_packager_names(self):
        from package_weights import release_name
        for fn, m in RELEASES.items():
            self.assertEqual(release_name(m["run"], m["family"], m["hw"], m["seed"]), fn)
        # a run outside the table gets the same naming scheme
        self.assertEqual(release_name("dsec_v8n_416s3y_s1", "yolov8n_s3y", [320, 416], 1), "yolov8n_s3y_dsec_416x320_s1.pt")
        self.assertEqual(release_name("dsec_y11n_416_s0", "yolo11n", [320, 416], 0), "yolo11n_plain_dsec_416x320_s0.pt")
        self.assertEqual(release_name("hit_v8n_416s3xaltLH_s1", "yolov8n_s3x", [320, 416], 1), "yolov8n_s3x_hituav-altLH_416x320_s1.pt")
        self.assertEqual(release_name("hit_v8n_416_s2", "yolov8n", [320, 416], 2), "yolov8n_plain_hituav_416x320_s2.pt")


if __name__ == "__main__":
    unittest.main()
