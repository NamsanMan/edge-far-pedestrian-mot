# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""efpmot.train on CPU: config defaults, the detection-cache file format of export_dets() on synthetic frames, and a 2-iteration
smoke training on MOT17 when the dataset is configured (written to a temporary outputs folder).
Run: python -B tests/test_train.py"""
import os
import sys
import tempfile
import types
import unittest
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import efpmot.train as T  # noqa: E402
from efpmot import paths  # noqa: E402
from efpmot.models import build  # noqa: E402


class TempOutputs:
    """EFPMOT_PATHS -> a copy of the active configuration with every outputs.* key in a temporary folder"""

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        cfg = paths.load()

        def absolute(section):  # resolved paths of the active file (its relative paths refer to the repository root)
            return {k: (paths.get(f"{section}.{k}") or "").replace("\\", "/") or None for k in (cfg.get(section) or {})}
        new = {s: absolute(s) for s in ("data", "weights", "third_party")}
        new["outputs"] = {k: os.path.join(self.tmp.name, k).replace("\\", "/") for k in ("runs", "cache", "results")}
        new["outputs"]["derived"] = absolute("outputs").get("derived") or os.path.join(self.tmp.name, "derived").replace("\\", "/")
        f = os.path.join(self.tmp.name, "paths.yaml")
        with open(f, "w", encoding="utf-8") as fh:
            yaml.safe_dump(new, fh)
        self.old = os.environ.get(paths.ENV_VAR)
        os.environ[paths.ENV_VAR] = f
        paths.load(reload=True)
        return self.tmp.name

    def __exit__(self, *exc):
        if self.old is None:
            os.environ.pop(paths.ENV_VAR, None)
        else:
            os.environ[paths.ENV_VAR] = self.old
        paths._CFG = paths._CFG_FILE = None
        self.tmp.cleanup()


class TestTrain(unittest.TestCase):
    def test_cfg_defaults(self):
        c = T._cfg({"family": "yolov8n_s3x", "data": "mot17", "name": "x"})
        self.assertEqual((c["hw"], c["epochs"], c["bs"], c["workers"], c["seed"]), ([256, 416], 55, 16, 4, 0))
        c = T._cfg({"family": "yolov8n", "data": "dsec", "name": "x", "hw": (352, 448)})
        self.assertEqual((c["hw"], c["epochs"], c["dsec_stride"]), ([352, 448], 6, 2))
        with self.assertRaises(ValueError):
            T._cfg({"data": "dsec", "name": "x"})

    def test_export_format(self):
        with TempOutputs() as tmp:
            img_dir = os.path.join(tmp, "imgs")
            os.makedirs(img_dir)
            fps = []
            for i in range(3):
                p = os.path.join(img_dir, f"{i:06d}.png")
                cv2.imwrite(p, (np.random.RandomState(i).rand(240, 320, 3) * 255).astype(np.uint8))
                fps.append(p)
            fake = types.SimpleNamespace(eval_seqs=lambda: ["seqA"], eval_frame_paths=lambda s: fps)
            orig = T.get_data
            T.get_data = lambda d: fake
            try:
                torch.manual_seed(0)
                ad = build("yolov8n_s3x", 8)
                out = T.export_dets(ad, "unit", "dsec", (256, 416), device="cpu")
            finally:
                T.get_data = orig
            with np.load(os.path.join(out, "seqA.npz"), allow_pickle=True) as z:
                files, frames, dets = sorted(z.files), z["frames"].tolist(), list(z["dets"])
            self.assertEqual(files, ["dets", "frames"])  # numpy >= 2 takes allow_pickle as an option, not as an array
            self.assertEqual(frames, ["000000.png", "000001.png", "000002.png"])
            self.assertEqual(len(dets), 3)
            for d in dets:
                d = np.asarray(d, np.float32).reshape(-1, 6)
                self.assertLessEqual(len(d), 300)
                if len(d):
                    self.assertTrue((d[:, 4] >= 0.01).all())
                    self.assertTrue((d[:, [0, 2]] >= 0).all() and (d[:, [0, 2]] <= 320).all())
                    self.assertTrue((d[:, [1, 3]] >= 0).all() and (d[:, [1, 3]] <= 240).all())
                    self.assertTrue(set(d[:, 5].astype(int).tolist()) <= set(range(8)))

    def test_smoke_train_mot17(self):
        if not (paths.get("data.mot17_train") and os.path.isdir(paths.get("data.mot17_train"))):
            self.skipTest("data.mot17_train not configured")
        w = paths.get("weights.yolov8n")
        if not (w and os.path.isfile(w)):
            self.skipTest("weights.yolov8n not configured")
        with TempOutputs():
            run = T.train(dict(family="yolov8n_s3x", data="mot17", name="smoke", bs=2, workers=0, max_iters=2, device="cpu"))
            with open(os.path.join(run, "log.txt")) as f:
                log = f.read()
            self.assertIn("== smoke family=yolov8n_s3x hw=(256, 416) train_frames=2664", log)
            self.assertIn("SMOKE_DONE it 2", log)
            self.assertFalse(os.path.exists(os.path.join(run, "last.pth")))


if __name__ == "__main__":
    unittest.main()
