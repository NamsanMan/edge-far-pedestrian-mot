# Experiments log: what else was tried

S3x changes four strides inside YOLOv8n so that its P3 grid, the finest grid on which it predicts objects, is twice as
fine horizontally (4 px instead of 8 px apart). It is what remained after a longer search. This page records the
alternatives, the numbers they reached and why they were dropped, so that nobody has to repeat them blindly. None of
this code is in the repository; the numbers are given as text.

**How to read the numbers.**
- Unless stated otherwise, results are on the DSEC test set (13 sequences), in HOTA points (HOTA is the standard
  multi-object tracking score, 0-100), with tracker thresholds cross-fitted, that is, chosen on sequences other than
  the one being scored.
- Each entry gives a difference d = A - B (positive: A tracks better), first with ByteTrack, then with OC-SORT
  ("ByteTrack / OC-SORT"). Where it matters, the 95 % bootstrap interval and the **unadjusted** randomisation p are
  added; unadjusted means without the Holm correction for multiple comparisons ([`PROTOCOL.md`](PROTOCOL.md) explains
  both tests). n.s. = not significant.
- "1 seed" marks pilots with a single training seed. Seed noise alone can move such a result by about +-1 point. The
  plain 416 x 320 baseline has 3 seeds.
- These are exploratory results. Only the comparisons in [`results/RESULTS.md`](../results/RESULTS.md) were planned and
  Holm-adjusted. The comparison with other detectors in [`results/DETECTORS.md`](../results/DETECTORS.md) was added
  afterwards and is Holm-adjusted within its own families (groups of comparisons that are corrected together).

## Reference points

| model (YOLOv8n, DSEC) | car HOTA (ByteTrack / OC-SORT) | pedestrian HOTA (ByteTrack / OC-SORT) | GFLOPs | Jetson Orin Nano FP16 latency |
|---|---|---|---|---|
| plain 416 x 320 | 52.97 / 55.91 | 27.92 / 28.93 | 2.665 | 2.94 ms |
| + S3x 416 x 320 | 55.81 / 58.09 | 33.80 / 34.42 | 2.955 | 3.26 ms |
| plain 640 x 480 | 59.82 / 62.47 | 39.05 / 39.52 | 6.15 | 4.72 ms |

S3x - plain 640 x 480 is -4.00 / -4.38 for cars and -5.25 / -5.10 for pedestrians. S3x closes part of the gap to a
canvas with 2.3x the pixels (307,200 vs 133,120), not all of it.

## 1. Other ways to refine P3

All alternatives below use the 416 x 320 canvas and 2 seeds each, unless marked "1 seed".

- **Isotropic P2 level** (an extra head at stride 4 in both axes, about 4.9 GFLOPs, 4.51 ms on the Orin Nano).
  - S3x - P2: car +1.42 / +1.56, pedestrian +1.63 / +3.03, none significant.
  - S3x is not significantly different from the stride-4 level (the point estimates favour S3x) at about 60 % of its
    FLOPs and 72 % of its latency.
  - A grid that is finer in both axes is not needed; refining the pedestrian's short axis (horizontal) is enough.
- **Two half-width P3 branches** (one finer in x, with stride 8 in y and 4 in x, and one finer in y, with stride 4 in y
  and 8 in x; 2.751 GFLOPs, +3.2 % over plain).
  - Against plain: pedestrian +4.54 [+3.18, +8.57] p 0.024 / +3.94 p 0.041, but cars +1.79 (p 0.14) / -0.71 (p 0.40).
  - S3x - two-branch: car +1.05 / +2.90 (p 0.011), pedestrian +1.35 / +1.55 (n.s.).
  - Dropped for two reasons. First, the criterion fixed before the runs required a car gain, and that gain was missing.
    Second, the TensorRT engine took 4.14 ms on the Orin Nano (+41 % over plain) although the FLOPs rose by only 3 %,
    because the split branches do not fuse well in TensorRT.
- **Shallow MatrixNets-style x-fine branch** (an extra branch off the plain P3: an `AUpsample` (1, 2) that doubles the
  horizontal resolution, one 3 x 3 conv and its own head; FLOPs matched to S3x; 3.55 ms on the Orin Nano).
  - S3x - shallow branch: pedestrian +4.10 [+1.46, +7.89] p 0.007 / +4.14 [+1.60, +8.88] p 0.003; car +2.24 (p 0.13) /
    +3.25 (p 0.020).
  - At equal FLOPs, S3x wins clearly. The finer stride has to run through the whole P3 stage, as in S3x, not be bolted
    on after it.
  - A variant with 6 channels, matched to the FLOPs of the two-branch design, failed to train.
- **Longer training of S3x** (9 instead of 6 epochs, 1 seed): car 56.0 / 57.9, pedestrian 34.8. These values are
  within seed noise of the 6-epoch results.
- **Assignment fix** (every ground-truth box that contains no anchor centre gets its nearest anchor as an assignment
  candidate; 1 seed). It tackles in the assigner, instead of the architecture, the grid-gap problem of
  [`METHOD.md`](METHOD.md): a box narrower than the 8 px anchor spacing can fall between two anchor columns and get no
  positive anchor.
  - Car +1.46 (p 0.021) / -1.57, pedestrian +0.06 / +0.73: no consistent effect.
  - An anchor made positive although its receptive field is centred off the object does not learn to detect it. With
    a finer grid, an anchor centre can fall inside the narrow box, and such an anchor does learn it.

## 2. Training-side and tracker-side alternatives

Training-side alternatives, which add no inference cost (knowledge distillation, variants of the assignment targets),
and tracker-side changes (score filters, rules for starting new tracks) gave no consistent gain over the same detectors
without them, so the work moved to the detector's output grid.

## 3. Other detector families

The S3 edit (S3x, and S3y where marked) was also ported to three other detectors. Results are on DSEC unless marked,
with 2 seeds (1 seed for S3y). The code of these ports is not included: the YOLOX and NanoDet ports live in separate
codebases (the YOLOX and NanoDet repositories), and the YOLO26 port needs Ultralytics 8.4, while this repository pins
Ultralytics 8.3.170. Their training scripts were not ported to or verified in this repository.

In the table, GF = GFLOPs and ch = channels. The last column compares S3x with the plain model of the
same family on a larger input (the canvas is named in each cell). That input has about the same FLOPs as S3x
(compute-matched uniform), except in the YOLOX-Nano row, where it has 1.5x the pixels.

| detector | FLOPs of the S3x edit | S3x - plain 416, pedestrian | S3x - plain 416, car | S3x - compute-matched uniform |
|---|---|---|---|---|
| YOLOX-Nano | +10.5 % (0.917 GF, P3 head 24 ch) | +4.64 [+3.50, +6.29] p 0.007 / +3.99 [+2.99, +8.38] p 0.010 | +0.92 (p 0.045) / +1.59 (p 0.11) | vs 512 (1.5x pixels): pedestrian -0.13 / -0.58, car -1.43 / -2.50 |
| NanoDet-Plus | +29.9 % (1.564 vs 1.204 GF) | +4.04 [+2.79, +9.71] p 0.033 / +1.74 (n.s.) | +1.08 / +2.41 (p 0.001) | vs 448 x 384: pedestrian +3.38 [+2.73, +7.30] p 0.009 / +1.85 (n.s.) |
| YOLO26n | +24.9 % (1.744 -> 2.178 GF) | +3.48 / +3.51 (both p <= 0.01) | +2.34 / +2.39 | vs 480 x 352: pedestrian +1.14 (p 0.057) / +0.53, car +0.92 / -1.11 |
| YOLO26n, BDD100K | +24.9 % | +1.82 [+1.15, +2.48] / +2.11 | +0.49 / +0.99 | vs 512 x 256: pedestrian +0.39 / +0.31 (n.s.), car **-0.53 [-0.83, -0.22] / -0.35** |

- The pedestrian gain over the plain model of the same canvas reappears in every family, on DSEC and on BDD100K.
- **NanoDet.** The ShuffleNet stage 2 and the GhostPAN P3 path make a finer P3 expensive. Thinning the head alone cannot
  bring the FLOPs overhead below about +26 %.
- **YOLO26n.** Its default head is already thin, and its P3 neck uses C3k2 blocks, so the +11 % FLOPs overhead of
  YOLOv8n was out of reach. Ultralytics 8.4's assigner also enlarges ground-truth boxes smaller than 16 px before
  assignment.
  - At its +25 % cost, S3x did not beat the plain YOLO26n on a larger input of the same FLOPs, and it lost to it on
    BDD100K cars.
  - Against the plain model on a larger input of the same FLOPs, S3x therefore wins or shows no significant
    difference for YOLOv8n and YOLO11n (YOLO11n was tested on BDD100K only), but loses for YOLO26n (BDD100K cars).
- **S3y on NanoDet** (1 seed): pedestrian -2.41 / -2.59, car +1.22 / +1.86. This is the same axis pattern as YOLOv8n.

## 4. Not measured

- INT8 accuracy of S3x. The INT8 Jetson engines were built without calibration, so they give latency only.
- The end-to-end latency of the whole pipeline (camera, pre-processing, engine, NMS, tracker) on the Jetson.
- More than 2 seeds for any S3x model. The baselines have 3 seeds on DSEC and BDD100K.
