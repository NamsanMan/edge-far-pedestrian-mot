# Experiments log: what else was tried

S3x is what remained after a longer search. This page records the alternatives, the numbers they reached and why
they were dropped, so that nobody has to repeat them blindly. None of this code is in the repository; the numbers are
given as text.

**How to read the numbers.**
- Unless stated otherwise: DSEC test (13 sequences), HOTA in points, cross-fitted tracker thresholds.
- Each entry is written as "d = A - B, ByteTrack / OC-SORT", with the 95 % bootstrap interval and the **unadjusted**
  randomisation p where it matters ([`PROTOCOL.md`](PROTOCOL.md)).
- "1 seed" marks pilots with a single training seed; their differences are within about +-1 point of seed noise. The
  plain 416 x 320 baseline has 3 seeds.
- These are exploratory results. Only the contrasts in [`results/RESULTS.md`](../results/RESULTS.md) were planned and
  Holm-adjusted.

## Reference points

| arm (YOLOv8n, DSEC) | car HOTA (BT / OC) | pedestrian HOTA (BT / OC) | GFLOPs | Orin Nano FP16 |
|---|---|---|---|---|
| plain 416 x 320 | 52.97 / 55.91 | 27.92 / 28.93 | 2.665 | 2.94 ms |
| + S3x 416 x 320 | 55.81 / 58.09 | 33.80 / 34.42 | 2.955 | 3.26 ms |
| plain 640 x 480 | 59.82 / 62.47 | 39.05 / 39.52 | 6.15 | 4.72 ms |

S3x - plain 640 is -4.00 / -4.38 for cars and -5.25 / -5.10 for pedestrians. S3x closes part of the gap to a canvas with 2.3x
the pixels (307,200 vs 133,120), not all of it.

## 1. Other ways to refine P3

All on the 416 x 320 canvas, 2 seeds each.

- **Isotropic P2 level** (an extra stride-4 head, about 4.9 GFLOPs, 4.51 ms on the Orin Nano).
  - S3x - P2: car +1.42 / +1.56, pedestrian +1.63 / +3.03, none significant.
  - S3x matches or beats the stride-4 level at about 60 % of its FLOPs and 72 % of its latency.
  - A finer grid in both axes is not needed; the pedestrian's short axis is enough.
- **Two half-width P3 branches** (one x-fine at 8 y x 4 x, one y-fine at 4 y x 8 x, 2.751 GFLOPs, +3.2 %).
  - Against plain: pedestrian +4.54 [+3.18, +8.57] p 0.024 / +3.94 p 0.041, but cars +1.79 (p 0.14) / -0.71 (p 0.40).
  - S3x - two-branch: car +1.05 / +2.90 (p 0.011), pedestrian +1.35 / +1.55 (n.s.).
  - Dropped for two reasons. The car gain required by the criterion fixed before the runs was missing. And the engine
    took 4.14 ms on the Orin Nano (+41 %), although the FLOPs rose by only 3 %: the split branches do not fuse well in
    TensorRT.
- **Shallow MatrixNets-style x-fine branch** (an `AUpsample` (1, 2) plus one 3 x 3 conv off the plain P3, with its own
  head; FLOPs matched to S3x; 3.55 ms on the Orin Nano).
  - S3x - shallow branch: pedestrian +4.10 [+1.46, +7.89] p 0.007 / +4.14 [+1.60, +8.88] p 0.003; car +2.24 (p 0.13) /
    +3.25 (p 0.020).
  - At equal FLOPs the full-depth fine grid wins clearly. The finer stride has to run through the P3 stage, not be
    bolted on after it.
  - A variant matched to the two-branch FLOPs (6 channels) did not train.
- **Longer training of S3x** (9 instead of 6 epochs, 1 seed): car 56.0 / 57.9, pedestrian 34.8. This is the same as
  6 epochs within seed noise.
- **Assignment fix** (give every GT box that has no anchor centre inside it its nearest anchor as a candidate, 1 seed).
  This attacks the grid-gap problem of [`METHOD.md`](METHOD.md) in the assigner instead of the architecture.
  - Car +1.46 (p 0.021) / -1.57, pedestrian +0.06 / +0.73: no consistent effect.
  - A positive anchor whose receptive field is centred off the object does not learn to detect it. A finer grid does.

## 2. Training-side and tracker-side alternatives

Training-side alternatives (distillation and assignment-target variants, no inference cost) and tracker-side changes
(score filters, track-birth rules) gave no consistent gain over the same detectors, so the work moved to the
detector's output grid.

## 3. Other detector families

The S3 edit was also ported to three other detectors. DSEC unless marked, 2 seeds (S3y 1 seed). The code of these ports
is not included (YOLOX and NanoDet are Apache-2.0 projects; YOLO26 needs Ultralytics 8.4).

| detector | FLOPs of the S3x edit | S3x - plain 416, pedestrian | S3x - plain 416, car | S3x - compute-matched uniform |
|---|---|---|---|---|
| YOLOX-Nano | +10.5 % (0.917 GF, P3 head 24 ch) | +4.64 [+3.50, +6.29] p 0.007 / +3.99 [+2.99, +8.38] p 0.010 | +0.92 (p 0.045) / +1.59 (p 0.11) | vs 512 (1.5x pixels): pedestrian -0.13 / -0.58, car -1.43 / -2.50 |
| NanoDet-Plus | +29.9 % (1.564 vs 1.204 GF) | +4.04 [+2.79, +9.71] p 0.033 / +1.74 (n.s.) | +1.08 / +2.41 (p 0.001) | vs 448 x 384: pedestrian +3.38 [+2.73, +7.30] p 0.009 / +1.85 (n.s.) |
| YOLO26n | +24.9 % (1.744 -> 2.178 GF) | +3.48 / +3.51 (both p <= 0.01) | +2.34 / +2.39 | vs 480 x 352: pedestrian +1.14 (p 0.057) / +0.53, car +0.92 / -1.11 |
| YOLO26n, BDD100K | +24.9 % | +1.82 [+1.15, +2.48] / +2.11 | +0.49 / +0.99 | vs 512 x 256: pedestrian +0.39 / +0.31 (n.s.), car **-0.53 [-0.83, -0.22] / -0.35** |

- The pedestrian gain over the plain model of the same canvas reappears in every family, on DSEC and on BDD100K.
- **NanoDet.** The ShuffleNet stage 2 and the GhostPAN P3 path make a finer P3 expensive. Thinning the head alone cannot
  bring the overhead below about +26 %.
- **YOLO26n.** Its default head is already thin, and its P3 neck uses C3k2 blocks, so +11 % was out of reach. Ultralytics
  8.4's assigner also enlarges GT boxes smaller than 16 px before assignment.
  - At its +25 % cost, S3x did not beat a uniform canvas of the same cost: it lost on BDD100K cars.
  - The efficiency claim therefore holds for YOLOv8n and YOLO11n, not for YOLO26n.
- **S3y on NanoDet** (1 seed): pedestrian -2.41 / -2.59, car +1.22 / +1.86. The same axis pattern as YOLOv8n.

## 4. Not measured

- INT8 accuracy of S3x. The INT8 Jetson engines were built without calibration, so they give latency only.
- The end-to-end camera -> TensorRT -> tracker -> alert latency on the Jetson.
- More than 2 seeds per S3x arm. The baselines have 3 seeds on DSEC and BDD100K.
