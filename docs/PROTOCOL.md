# Evaluation protocol

Every comparison in this repository answers one question: does arm A track better than arm B under the same tracker?
An arm is a detector variant (for example YOLOv8n + S3x on a 416 x 320 canvas) trained with several seeds. The protocol
is implemented in `efpmot/eval/protocol.py` and runs on CPU from the shipped `results/stats/` files; no dataset is
needed. Commands: [`REPRODUCE.md`](REPRODUCE.md).

## 1. From detections to HOTA

1. Each trained model exports its detections once: post-NMS, conf >= 0.01, class-wise NMS IoU 0.65, at most 300 boxes,
   FP16, batch 1.
2. The cached detections go through BoxMOT 25.0 trackers with fixed settings, the same for every detector:
   - ByteTrack: track buffer 15 frames, match threshold 0.8 (BoxMOT prints "max_age=30, det_thresh=0.3" when it
     creates a ByteTrack: base-class defaults, printed before ByteTrack applies the buffer and threshold above);
   - OC-SORT: max age 15, min hits 1, IoU threshold 0.3, BYTE association on;
   - one tracker per class;
   - detections scoring below 0.2 x the threshold are dropped before tracking.
3. Only the detection threshold varies: 8 values for ByteTrack (0.2 ... 0.9) and 5 for OC-SORT (0.5 ... 0.9). Every
   value is run on every sequence.
4. TrackEval (commit `12c8791`) scores every (run, tracker, threshold) per sequence:
   - DSEC test, BDD100K MOT val and VisDrone2019-MOT test-dev with its BDD100K evaluator;
   - MOT17 with its MOTChallenge evaluator, using the official distractor preprocessing.

## 2. The threshold is never chosen on the sequence it scores

A threshold chosen to maximise HOTA on the test sequences would inflate every arm by a different amount. The primary
rule, `cross`, cross-fits it:

- **DSEC (13 test sequences from 8 drives), MOT17 (7 evaluation halves), VisDrone (17 test-dev clips from 14 source
  videos): leave-one-sequence-out.**
  For each held-out sequence, every run gets the threshold with the best pooled HOTA on the other sequences. The
  held-out sequence is then scored at that threshold.
- **BDD100K (200 videos): 10 folds.** The fold assignment is fixed: sorted names, a NumPy `default_rng(0)` permutation,
  position mod 10.
- **Ties** go to the threshold closest to 0.5, then to the higher one (0.6 over 0.4). The distance to 0.5 is computed in
  floating point, so 0.7 beats 0.3 and 0.2 beats 0.8 (`pick()` in `efpmot/eval/protocol.py`).
- **Sensitivity rules**, reported but never deciding: `best` (one threshold maximising pooled HOTA over all test
  sequences) and `fixed` (0.5).

## 3. HOTA is recombined exactly

TrackEval's per-sequence `detailed.csv` holds, for each of the 19 localisation thresholds alpha = 0.05 ... 0.95, the
counts HOTA_TP, HOTA_FN and HOTA_FP and the association score AssA. The protocol recombines them as TrackEval does:

- per alpha, sum the counts over the sequences (each sequence at its own selected threshold);
- DetA = TP / (TP + FN + FP);
- AssA = sum(AssA x TP) / sum(TP);
- HOTA = mean over the alphas of sqrt(DetA x AssA).

Every file is checked against its COMBINED row (error <= 1e-6). `results/stats/` stores exactly these four statistics
per sequence, threshold and alpha, so the comparisons can be recomputed without TrackEval output.

## 4. Tests

For arms A and B, with d = seed-mean pooled HOTA(A) - seed-mean pooled HOTA(B):

- **Paired hierarchical bootstrap (B = 10,000, seed 0).**
  1. Resample the sequences (BDD100K: videos) with replacement, with the same draw for both arms.
  2. Within each arm, resample its seeds with replacement.

  This gives 95 % and 90 % percentile intervals for d. Both sources of variation that matter here, which scenes were
  recorded and which seed was trained, are in the interval.
- **Paired randomisation test (two-sided).** Per sequence, the seed-averaged statistics of A and B are swapped with
  probability 1/2.
  - When there are few sequences, all 2^K swap patterns are enumerated: exact for DSEC (K = 13) and MOT17 (K = 7; the
    smallest attainable p is 2/128 = 0.0156).
  - Otherwise 10,000 random patterns are drawn (VisDrone K = 17, BDD100K K = 200).
- **TOST equivalence.** Two one-sided bootstrap tests against a margin delta; equivalently, the 90 % interval lies inside
  +-delta.
- **Margins, frozen before the comparisons they judge.**
  - DSEC: 1.1955 HOTA. BDD100K: 0.0421 HOTA.
  - Each is 2 x the median within-arm seed SD of pooled ByteTrack car HOTA under the cross-fitted rule, computed once over
    the plain and reference arms.
  - VisDrone and MOT17 take the DSEC value, frozen before any training on them. Their own seed-SD estimate would rest on
    2 arms x 2 seeds.
- **Multiplicity.** For the result tables, randomisation and TOST p-values are Holm-adjusted within each family
  (dataset, class, tracker) over the contrasts listed in `results/contrasts.yaml`.
- **Labels.**
  - `win` / `loss`: the Holm-adjusted randomisation p < 0.05 (the sign of d decides which).
  - `equivalent`: otherwise, the Holm-adjusted TOST p < 0.05.
  - `inconclusive`: neither.
  - A win or loss that also passes TOST is marked `<m`: significant, but smaller than the margin.

**Recording clusters (sensitivity).** Sequences cut from the same recording are not independent: the 13 DSEC test
sequences come from 8 drives (for example `zurich_city_14_a`, `_b` and `_c`), and the 17 VisDrone clips from 14 source
videos. A sequence of the same drive can also help choose the held-out sequence's threshold. Both tables therefore carry
a sensitivity check that cross-fits the thresholds leave-one-cluster-out and resamples and swaps whole drives / videos
(`cluster_sensitivity` and the VisDrone `sensitivity_video` in `efpmot/eval/protocol.py`). On DSEC the headline
pedestrian results keep their sign and size at drive level (S3x - plain: +5.88, p 0.016 ByteTrack; +5.35, p 0.016
OC-SORT; S3x - uniform 448 x 352: +2.24, p 0.023; +2.68, p 0.008). With 8 drives the smallest attainable p is
2/256 = 0.0078, so with 7 contrasts per Holm family none of them could pass Holm at drive level; these p-values are
unadjusted.

**What the tests do not cover.** The unit of resampling is the sequence. Two seeds per arm (three for some baselines) is
few, and the seed level of the bootstrap is coarse with so few seeds. The detectors were trained for short schedules
(about 9,000 iterations) on a single GPU.

## Pre-registered tests: outcomes

The VisDrone2019-MOT and MOT17 tests were pre-registered before any training on those datasets
([`PREREGISTRATION.md`](PREREGISTRATION.md)). They have their own reading rules:

- P1 and P2 hold only if d > 0 with unadjusted p_AR < 0.05 for ByteTrack and for OC-SORT;
- the efficiency and car predictions are read by non-inferiority at the 90 % interval (lower bound > -1.1955).

Below: 2 seeds per arm, cross-fitted thresholds, d in HOTA points, 95 % bootstrap interval, p_AR. For the
non-inferiority rows, the 90 % interval is given instead.

### VisDrone2019-MOT test-dev (17 clips, canvas 640 x 384)

| prediction | contrast | ByteTrack | OC-SORT | reading |
|---|---|---|---|---|
| P1 pedestrian: S3x > plain | `640s3x` - `640` | +2.58 [+0.24, +5.38], p 0.004 | +3.05 [+0.48, +5.86], p 0.005 | **holds** |
| P2 pedestrian: S3x > S3y | `640s3x` - `640s3y` | +2.70 [+0.78, +4.94], p 0.011 | +2.40 [+0.62, +4.74], p 0.014 | **holds** |
| P3 car: S3y >= S3x | `640s3y` - `640s3x` | +0.85, 90 % [+0.12, +1.61] | +0.49, 90 % [-0.61, +1.43] | **holds** (non-inferior) |
| P4 pedestrian: S3x >= uniform 704 x 384 | `640s3x` - `704x384` | +0.07, 90 % [-1.36, +1.22], p 0.76 | +0.22, 90 % [-1.44, +1.59], p 0.58 | **inconclusive** |

- The video-cluster sensitivity check leaves P1 and P2 significant. It cross-fits leave-one-video-out and resamples the
  14 source videos instead of the 17 clips. The p values are 0.015 / 0.018 for P1 and 0.039 / 0.038 for P2 (ByteTrack /
  OC-SORT).
- Descriptive, cars, S3x - plain: +0.23 (ByteTrack, p 0.69) and +0.98 (OC-SORT, p 0.11). S3x neither helped nor hurt
  cars measurably. S3y was the better arm for cars with ByteTrack (+0.85, p 0.009); with OC-SORT the difference is
  +0.49 and not significant (p 0.25).
- Reading of the pre-registration: P1 and P2 hold, so the short-side rule transfers from driving cameras to drone views.
- P4: S3x and the 704 x 384 uniform canvas are tied on the point estimate, but the interval is too wide to show
  non-inferiority.

### MOT17 train, evaluation halves (7 sequences, canvas 416 x 256)

| prediction | contrast | ByteTrack | OC-SORT | reading |
|---|---|---|---|---|
| P1 pedestrian: S3x > plain | `416s3x` - `416` | +2.70 [+0.43, +4.01], p 0.062 | +2.12 [-0.34, +3.73], p 0.016 | **not shown** (not refuted) |
| P2 pedestrian: S3x > S3y | `416s3x` - `416s3y` | +2.38 [+1.06, +5.54], p 0.016 | +2.11 [+0.78, +3.65], p 0.031 | **holds** |
| P4 pedestrian: S3x >= uniform 480 x 256 | `416s3x` - `480x256` | -0.51, 90 % [-1.60, +0.29], p 0.14 | +0.77, 90 % [-0.62, +1.77], p 0.38 | **inconclusive** (ByteTrack is primary; OC-SORT alone would be non-inferior) |

- P1 misses the strict both-trackers rule only on ByteTrack, where the exact randomisation p is 8/128 (6 of 7
  sequences favour S3x). Under the pre-registered wording it is "not shown". It is not "refuted": the 95 % interval does
  not exclude +0.5.
- Static / moving split (descriptive), S3x - plain:
  - static cameras (02, 04, 09): +3.08 (ByteTrack) and +2.52 (OC-SORT);
  - moving cameras (05, 10, 11, 13): +1.99 and +1.16.
- The MOT17 test has little power: 7 sequences, 2 seeds, and pedestrians mostly larger than one P3 stride (18 % below
  8 px at a 416 long side, against 76 % on VisDrone). The direction of every pedestrian contrast matches DSEC, BDD100K
  and VisDrone.

All contrasts with Holm-adjusted p-values and labels, for DSEC and BDD100K too, are in
[`results/RESULTS.md`](../results/RESULTS.md).
