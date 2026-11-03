# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""efpmot.models on CPU: S3x / S3y grid shapes and per-axis strides, decode round trip, a toy loss + backward, FLOPs, ONNX-free
eval decode == decode_feats, checkpoint round trip; COCO loading when the weights are configured.
Run: python -B tests/test_models.py"""
import os
import sys
import tempfile
import unittest
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore")
import torch  # noqa: E402

from efpmot import paths  # noqa: E402
from efpmot.models import FAMILIES, build, flops, load_checkpoint  # noqa: E402
from efpmot.models.aniso import make_anchors_xy  # noqa: E402

try:
    import thop  # noqa: F401
    HAVE_THOP = True
except ImportError:
    HAVE_THOP = False

H, W = 320, 416
# expected (H_level, W_level) of the three levels at 320x416 and the per-level (sx, sy)
GRIDS = {
    "yolov8n": ([(40, 52), (20, 26), (10, 13)], [[8, 8], [16, 16], [32, 32]]),
    "yolov8n_s3x": ([(40, 104), (20, 26), (10, 13)], [[4, 8], [16, 16], [32, 32]]),
    "yolov8n_s3y": ([(80, 52), (20, 26), (10, 13)], [[8, 4], [16, 16], [32, 32]]),
    "yolo11n": ([(40, 52), (20, 26), (10, 13)], [[8, 8], [16, 16], [32, 32]]),
    "yolo11n_s3x": ([(40, 104), (20, 26), (10, 13)], [[4, 8], [16, 16], [32, 32]]),
    "yolo11n_s3y": ([(80, 52), (20, 26), (10, 13)], [[8, 4], [16, 16], [32, 32]]),
}
HEAD_W = {"yolov8n_s3x": 32, "yolov8n_s3y": 32, "yolo11n_s3x": 24, "yolo11n_s3y": 24}


def toy_batch(seed=0, B=2):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(B, 3, H, W, generator=g) * 255
    tgt = torch.zeros(B, 120, 6)
    tgt[:, :, 5] = -1
    for b in range(B):
        for j in range(4):
            w, h = 6 + 30 * torch.rand(1, generator=g).item(), 12 + 60 * torch.rand(1, generator=g).item()
            tgt[b, j] = torch.tensor([j % 3, 50 + 300 * torch.rand(1, generator=g).item(), 50 + 200 * torch.rand(1, generator=g).item(), w, h, j])
    return x, tgt


class TestModels(unittest.TestCase):
    def test_grids_and_strides(self):
        for fam in FAMILIES:
            ad = build(fam, 8)
            ad.model.train()
            feats = ad.model(ad.norm(torch.zeros(1, 3, H, W)))
            shapes = [tuple(f.shape[2:]) for f in feats]
            self.assertEqual(shapes, GRIDS[fam][0], fam)
            det = ad.model.model[-1]
            sxy = det.stride_xy.tolist() if hasattr(det, "stride_xy") else [[s, s] for s in det.stride.tolist()]
            self.assertEqual(sxy, GRIDS[fam][1], fam)
            if fam in HEAD_W:
                self.assertEqual(det.cv2[0][0].conv.out_channels, HEAD_W[fam], fam)
                self.assertEqual(det.cv2[1][0].conv.out_channels, 64, fam)

    def test_s3x_layer_edits(self):
        for fam, idx in (("yolov8n_s3x", (3, 5, 13, 16)), ("yolo11n_s3x", (3, 5, 14, 17))):
            m = build(fam, 8).model.model
            self.assertEqual(tuple(m[idx[0]].conv.stride), (2, 1), fam)
            self.assertEqual(tuple(m[idx[1]].conv.stride), (2, 4), fam)
            self.assertEqual(tuple(m[idx[2]].scale_factor), (2.0, 4.0), fam)
            self.assertEqual(tuple(m[idx[3]].conv.stride), (2, 4), fam)

    def test_anchors_xy(self):
        feats = [torch.zeros(1, 1, 2, 3), torch.zeros(1, 1, 1, 1)]
        pts, st = make_anchors_xy(feats, torch.tensor([[4.0, 8.0], [16.0, 16.0]]))
        self.assertEqual(pts.shape, (7, 2))
        self.assertEqual(pts[:3].tolist(), [[0.5, 0.5], [1.5, 0.5], [2.5, 0.5]])
        self.assertEqual(st[0].tolist(), [4.0, 8.0])
        self.assertEqual(st[-1].tolist(), [16.0, 16.0])

    def test_decode_axes(self):
        """a zero DFL distribution decodes to a box centred on its anchor, scaled per axis (x by sx, y by sy)"""
        ad = build("yolov8n_s3x", 8)
        det = ad.model.model[-1]
        feats = [torch.zeros(1, det.no, *s) for s in GRIDS["yolov8n_s3x"][0]]
        boxes, _ = ad.decode_feats(feats)
        b0 = boxes[0, 0]  # first P3 anchor (0.5, 0.5) grid units, uniform DFL -> 7.5 units per side
        self.assertAlmostEqual(float((b0[0] + b0[2]) / 2), 0.5 * 4, places=4)
        self.assertAlmostEqual(float((b0[1] + b0[3]) / 2), 0.5 * 8, places=4)
        self.assertAlmostEqual(float(b0[2] - b0[0]), 15 * 4, places=3)
        self.assertAlmostEqual(float(b0[3] - b0[1]), 15 * 8, places=3)

    def test_loss_backward_and_eval_decode(self):
        x, tgt = toy_batch()
        for fam in FAMILIES:
            torch.manual_seed(0)
            ad = build(fam, 8)
            ad.model.train()
            loss, items, feats = ad.train_forward(x, tgt)
            self.assertTrue(torch.isfinite(loss).item(), fam)
            self.assertEqual(set(items), {"box", "cls", "dfl"})
            loss.backward()
            g = [p.grad for p in ad.model.parameters() if p.grad is not None]
            self.assertTrue(g and all(torch.isfinite(t).all() for t in g), fam)
            self.assertGreater(sum(float(t.abs().sum()) for t in g), 0.0, fam)
            with torch.no_grad():
                b2, s2 = ad.infer(x)  # eval mode (BN running statistics)
                b1, s1 = ad.decode_feats(ad.model(ad.norm(x))[1])  # the eval forward's raw per-level maps
            self.assertTrue(torch.allclose(b1, b2, atol=1e-3) and torch.allclose(s1, s2, atol=1e-5), fam)

    @unittest.skipUnless(HAVE_THOP, "needs the thop module (pip install -e .[dev])")
    def test_flops(self):
        ref = {"yolov8n": 2.665, "yolov8n_s3x": 2.955, "yolo11n": 2.101, "yolo11n_s3x": 2.333}
        for fam, gf in ref.items():
            self.assertAlmostEqual(flops(fam), gf, delta=0.002, msg=fam)

    @unittest.skipUnless(HAVE_THOP, "needs the thop module (pip install -e .[dev])")
    def test_flops_of_the_results_tables(self):   # efpmot.eval.table.GFLOPS (headline tables of results/RESULTS.md)
        from efpmot.eval.table import GFLOPS
        fam = {("v8n", "plain"): "yolov8n", ("v8n", "uniform"): "yolov8n", ("v8n", "S3x"): "yolov8n_s3x",
               ("y11n", "plain"): "yolo11n", ("y11n", "uniform"): "yolo11n", ("y11n", "S3x"): "yolo11n_s3x"}
        for (det, arm), gf in GFLOPS.items():
            kind, wh = arm.split()
            w, h = (int(v) for v in wh.split("x"))
            self.assertAlmostEqual(flops(fam[(det, kind)], (h, w)), gf, delta=0.0006, msg=f"{det} {arm}")

    def test_checkpoint_roundtrip(self):
        ad = build("yolov8n_s3x", 8)
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "x.pth")
            torch.save({"model": ad.model.state_dict(), "family": "yolov8n_s3x", "hw": [320, 416]}, f)
            ad2 = build("yolov8n_s3x", 8, f)
            meta = load_checkpoint(ad2, f)
            self.assertEqual(meta["family"], "yolov8n_s3x")
            sd1, sd2 = ad.model.state_dict(), ad2.model.state_dict()
            self.assertTrue(all(torch.equal(sd1[k], sd2[k]) for k in sd1))
            with self.assertRaises(RuntimeError):
                build("yolov8n", 8, f)  # S3x weights do not load strictly into the plain model

    def test_coco_load(self):
        for fam, key in (("yolov8n_s3x", "weights.yolov8n"), ("yolo11n_s3x", "weights.yolo11n")):
            w = paths.get(key)
            if not w or not os.path.isfile(w):
                self.skipTest(f"{key} not configured")
            ad = build(fam, 8, "coco")
            sd = torch.load(w, map_location="cpu", weights_only=False)["model"].float().state_dict()
            det = ad.model.model[-1]
            k = f"model.{det.i}.cv2.0.0.conv.weight"
            self.assertTrue(torch.equal(ad.model.state_dict()[k], sd[k][:HEAD_W[fam]]), fam)  # level-0 box branch sliced
            self.assertTrue(torch.equal(ad.model.state_dict()["model.0.conv.weight"], sd["model.0.conv.weight"]), fam)


if __name__ == "__main__":
    unittest.main()
