# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""efpmot.paths: lookup order, resolution against the repository root, null keys. Run: python -B tests/test_paths.py"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from efpmot import paths  # noqa: E402


class TestPaths(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.pop(paths.ENV_VAR, None)

    def tearDown(self):
        os.environ.pop(paths.ENV_VAR, None)
        if self._env is not None:
            os.environ[paths.ENV_VAR] = self._env
        paths._CFG = paths._CFG_FILE = None  # next load() follows the normal lookup order again

    def test_env_override_and_resolution(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "p.yaml")
            with open(f, "w", encoding="utf-8") as fh:
                fh.write("data:\n  dsec_det: some/rel\n  dsec_images: null\noutputs:\n  runs: " + d.replace("\\", "/") + "/runs\n")
            os.environ[paths.ENV_VAR] = f
            paths.load(reload=True)
            self.assertEqual(paths.config_file(), os.path.abspath(f))
            self.assertEqual(paths.get("data.dsec_det"), os.path.normpath(os.path.join(paths.repo_root(), "some/rel")))
            self.assertIsNone(paths.get("data.dsec_images"))
            self.assertIsNone(paths.get("data.missing"))
            with self.assertRaises(KeyError):
                paths.require("data.dsec_images")
            r = paths.out("runs", "x")
            self.assertTrue(os.path.isdir(r) and r.startswith(os.path.normpath(d)))

    def test_example_has_all_keys(self):
        cfg = paths.load(os.path.join(paths.repo_root(), "configs", "paths.example.yaml"), reload=True)
        for k in ("data", "weights", "outputs", "third_party"):
            self.assertIn(k, cfg)
        for k in ("derived", "runs", "cache", "results"):
            self.assertIn(k, cfg["outputs"])


if __name__ == "__main__":
    unittest.main()
