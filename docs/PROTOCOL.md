# Evaluation protocol

Every comparison in this repository answers one question: does arm A track better than arm B under the same tracker?
An arm is a detector variant (for example YOLOv8n + S3x on a 416 x 320 canvas) trained with several seeds; each of its
trained models (one per seed) is called a run. The protocol is implemented in `efpmot/eval/protocol.py` and runs on CPU
from the shipped `results/stats/` files; no dataset is needed. Commands: [`REPRODUCE.md`](REPRODUCE.md).

## 1. From detections to HOTA

1. Each trained model exports its detections once; every tracker and threshold setting below works from this cache.
   The export keeps the boxes with confidence >= 0.01 that survive class-wise non-maximum suppression
   (NMS, IoU 0.65), at most 300 per frame, and runs in FP16 with batch 1.
2. The cached detections go through BoxMOT 25.0 trackers with fixed settings, the same for every detector:
   - ByteTrack: track buffer 15 frames, match threshold 0.8. When BoxMOT creates a ByteTrack tracker it prints
     "max_age=30, det_thresh=0.3". These base-class defaults are printed before ByteTrack applies its own track buffer
     and detection threshold, so they are not the values in use;
   - OC-SORT: max age 15, min hits 1, IoU threshold 0.3, BYTE association on (low-score detections get a second
     matching stage, as in ByteTrack);
   - one tracker per class;
   - detections scoring below 0.2 x the detection threshold (step 3) are dropped before tracking.
3. Only the detection threshold varies: 8 values for ByteTrack (0.2 to 0.9 in steps of 0.1) and 5 for OC-SORT
   (0.5 to 0.9). Every value is run on every sequence; section 2 explains how one is chosen for each sequence.
4. TrackEval (commit `12c8791`) scores every combination of run, tracker and threshold, separately for each sequence:
   - DSEC test, BDD100K MOT val and VisDrone2019-MOT test-dev with TrackEval's BDD100K evaluator;
   - MOT17 with TrackEval's MOTChallenge evaluator, using the official preprocessing that removes distractor classes.

## 2. The threshold is never chosen on the sequence it scores

A threshold chosen to maximise HOTA on the test sequences would inflate the HOTA of every arm, each by a different
amount. The primary rule, `cross`, therefore cross-fits the threshold: it is chosen on sequences other than the one
being scored.

- **DSEC (13 test sequences from 8 drives), MOT17 (7 evaluation halves: the second half of each train sequence),
  VisDrone (17 test-dev clips from 14 source videos): leave-one-sequence-out.**
  For each held-out sequence, every run gets the threshold with the best pooled HOTA on the other sequences, that is,
  the HOTA of those sequences combined (section 3). The held-out sequence is then scored at that threshold.
- **BDD100K (200 videos): 10 folds.** For each run, each fold of 20 videos is scored at the threshold with the best
  pooled HOTA on the other nine folds. The folds are fixed: the video names are sorted and shuffled by a NumPy
  `default_rng(0)` permutation, and the video at position i of the shuffled order goes to fold i mod 10.
- **Ties** (thresholds with the same pooled HOTA) go to the threshold closest to 0.5, then to the higher one
  (0.6 over 0.4). The distance to 0.5 is computed in floating point, so two thresholds that are equally far from 0.5
  in exact arithmetic can come out slightly different: 0.7 beats 0.3, and 0.2 beats 0.8 (`pick()` in
  `efpmot/eval/protocol.py`).
- **Sensitivity rules.** Two other rules are reported to check that the conclusions do not depend on the choice of
  rule, but they never decide a result: `best` (the one threshold that maximises pooled HOTA over all test sequences)
  and `fixed` (0.5).

## 3. HOTA is recombined exactly

TrackEval's per-sequence `detailed.csv` holds, for each of the 19 localisation thresholds alpha = 0.05 ... 0.95 (a
match needs an IoU of at least alpha), the counts HOTA_TP, HOTA_FN and HOTA_FP and the association score AssA. Because
each sequence may be scored at a different tracker threshold (section 2), the protocol recombines these numbers itself,
exactly as TrackEval does:

- per alpha, sum the counts over the sequences (each sequence at its own selected threshold);
- DetA = TP / (TP + FN + FP);
- AssA = sum(AssA x TP) / sum(TP);
- HOTA = mean over the alphas of sqrt(DetA x AssA).

Each file covers one run, tracker, threshold and class. As a check, recombining its per-sequence rows must reproduce
the HOTA of its COMBINED row (error <= 1e-6). `results/stats/` stores exactly these four statistics (HOTA_TP,
HOTA_FN, HOTA_FP and AssA) per sequence, threshold and alpha, so the comparisons can be recomputed without TrackEval
output.

## 4. Tests

For arms A and B, let d = HOTA(A) - HOTA(B), where each HOTA is pooled over the test sequences and then averaged over
the arm's seeds.

- **Paired hierarchical bootstrap (10,000 resamples, random seed 0).** The bootstrap estimates the uncertainty of d by
  recomputing it on resampled data, at two levels:
  1. Resample the sequences (for BDD100K, the videos) with replacement, with the same draw for both arms (paired).
  2. Within each arm, resample its seeds with replacement.

  The 95 % interval of d spans the 2.5th to 97.5th percentiles of the resampled values, and the 90 % interval the 5th
  to 95th. Both sources of variation that matter here, which scenes were recorded and which seeds the models were
  trained with, are therefore in the interval.
- **Paired randomisation test (two-sided).** If A and B were equally good, it would not matter which of the two is
  called A on any sequence. The test therefore swaps the seed-averaged statistics (section 3) of A and B on each
  sequence with probability 1/2 and recomputes the difference. The p-value measures how often a swapped difference is
  at least as large, in absolute value, as the observed one.
  - With K sequences there are 2^K swap patterns. When 2^K is at most 10,000, all of them are enumerated and p is
    exact: DSEC (K = 13) and MOT17 (K = 7, where the smallest attainable p is 2/128 = 0.0156).
  - Otherwise 10,000 random patterns are drawn: VisDrone (K = 17) and BDD100K (K = 200).
- **TOST equivalence.** TOST (two one-sided tests) asks whether A and B are equivalent, that is, whether d is smaller
  in size than a margin delta. It runs two one-sided bootstrap tests, against -delta and against +delta; passing both
  is the same as the 90 % interval of d lying inside +-delta.
- **Margins (delta), frozen before the comparisons they judge.**
  - DSEC: 1.1955 HOTA. BDD100K: 0.0421 HOTA.
  - Each is 2 x the median seed-to-seed standard deviation (SD) within an arm: the SD of pooled ByteTrack car HOTA
    across an arm's seeds, under the cross-fitted rule, with the median taken once over the earlier plain arms on the 416-wide canvas and the 640-wide
    reference canvas.
  - VisDrone and MOT17 take the DSEC value, frozen before any training on them. An SD estimate from their own runs
    would rest on only 2 arms x 2 seeds.
- **Multiple comparisons.** For the result tables, the randomisation and TOST p-values are Holm-adjusted: Holm's
  correction adjusts a family of p-values together so that the chance of even one false positive in the family stays
  at or below 5 %. A family is the set of contrasts (planned A - B comparisons) in `results/contrasts.yaml` that share
  a dataset, class and tracker. The comparison with other detectors (listed in `results/contrasts_detectors.yaml`,
  reported in `results/DETECTORS.md`) forms its own families, one per class and tracker with 11 contrasts each. A
  contrast reported in both files therefore has the same d and unadjusted p but its own Holm-adjusted p in each.
- **Labels in the result tables.**
  - `win` / `loss`: the Holm-adjusted randomisation p < 0.05 (the sign of d decides which).
  - `equivalent`: otherwise, the Holm-adjusted TOST p < 0.05.
  - `inconclusive`: neither.
  - A win or loss that also passes TOST is marked `<m`: significant, but smaller than the margin.

**Recording clusters (sensitivity).** Sequences cut from the same recording are not independent: the 13 DSEC test
sequences come from 8 drives (for example `zurich_city_14_a`, `_b` and `_c`), and the 17 VisDrone clips from 14 source
videos. Under leave-one-sequence-out, the other sequences of the same drive also help choose the held-out sequence's
threshold. The DSEC and VisDrone results therefore include a sensitivity check that treats each drive or source video
as one cluster: it cross-fits the thresholds leave-one-cluster-out and resamples and swaps whole clusters
(`cluster_sensitivity` and the VisDrone `sensitivity_video` in `efpmot/eval/protocol.py`). On DSEC the main pedestrian
results keep their sign and size at drive level. S3x - plain is +5.88 (p 0.016) with ByteTrack and +5.35 (p 0.016)
with OC-SORT. S3x - uniform 448 x 352 is +2.24 (p 0.023) with ByteTrack and +2.68 (p 0.008) with OC-SORT, where
uniform 448 x 352 is the plain model on a larger input with about the same FLOPs as S3x (a "compute-matched uniform"
canvas). With 8 drives the smallest attainable p is 2/256 = 0.0078, but Holm with 7 contrasts per family requires the
smallest p to be below 0.05 / 7 = 0.0071. No drive-level contrast could therefore pass Holm, and the p-values above
are unadjusted.

**What the tests do not cover.** The unit of resampling is the sequence. Two seeds per arm (three for some baselines) is
few, and it makes the seed level of the bootstrap coarse: two seeds resampled with replacement give only three
different seed means. The detectors were trained for short schedules (about 9,000 iterations) on a single GPU.

## Pre-registered tests: outcomes

The VisDrone2019-MOT and MOT17 tests were pre-registered, that is, their predictions were written down before any
training on those datasets ([`PREREGISTRATION.md`](PREREGISTRATION.md)). They have their own reading rules:

- the pedestrian predictions P1 and P2 hold only if d > 0 with unadjusted p_AR < 0.05 for ByteTrack and for OC-SORT,
  where p_AR is the p-value of the paired randomisation test;
- the car prediction (P3) and the efficiency prediction (P4) are read by non-inferiority: they hold if the lower bound
  of the 90 % interval of d is above -1.1955, that is, if any loss is shown to be smaller than the DSEC margin.

In the tables below, each cell gives d in HOTA points with its 95 % bootstrap interval and p_AR (2 seeds per arm,
cross-fitted thresholds). For the non-inferiority rows, the 90 % interval is given instead.

### VisDrone2019-MOT test-dev (17 clips, canvas 640 x 384)

| prediction | contrast | ByteTrack | OC-SORT | outcome |
|---|---|---|---|---|
| P1 pedestrian: S3x > plain | `640s3x` - `640` | +2.58 [+0.24, +5.38], p 0.004 | +3.05 [+0.48, +5.86], p 0.005 | **holds** |
| P2 pedestrian: S3x > S3y | `640s3x` - `640s3y` | +2.70 [+0.78, +4.94], p 0.011 | +2.40 [+0.62, +4.74], p 0.014 | **holds** |
| P3 car: S3y >= S3x | `640s3y` - `640s3x` | +0.85, 90 % [+0.12, +1.61] | +0.49, 90 % [-0.61, +1.43] | **holds** (non-inferior) |
| P4 pedestrian: S3x >= uniform 704 x 384 | `640s3x` - `704x384` | +0.07, 90 % [-1.36, +1.22], p 0.76 | +0.22, 90 % [-1.44, +1.59], p 0.58 | **inconclusive** |

- The video-cluster sensitivity check leaves P1 and P2 significant. It cross-fits the thresholds leave-one-video-out
  and resamples the 14 source videos instead of the 17 clips. The p-values are 0.015 / 0.018 for P1 and 0.039 / 0.038
  for P2 (ByteTrack / OC-SORT).
- Cars, S3x - plain (descriptive, not a pre-registered prediction): +0.23 (ByteTrack, p 0.69) and +0.98 (OC-SORT,
  p 0.11). S3x neither helped nor hurt cars measurably. For cars, S3y was better than S3x with ByteTrack (+0.85,
  p 0.009); with OC-SORT the difference is +0.49 and not significant (p 0.25).
- Conclusion under the pre-registration: P1 and P2 hold, so the short-side rule (a small object is missed when its
  short side falls below about one stride of the P3 grid) transfers from driving cameras to drone views.
- P4: the point estimates of S3x - uniform 704 x 384 are close to zero, +0.07 (ByteTrack) and +0.22 (OC-SORT), but the
  90 % intervals are too wide to show non-inferiority.

### MOT17 train, evaluation halves (7 sequences, canvas 416 x 256)

MOT17 has no evaluated car class, so there is no P3 prediction.

| prediction | contrast | ByteTrack | OC-SORT | outcome |
|---|---|---|---|---|
| P1 pedestrian: S3x > plain | `416s3x` - `416` | +2.70 [+0.43, +4.01], p 0.062 | +2.12 [-0.34, +3.73], p 0.016 | **not shown** (not refuted) |
| P2 pedestrian: S3x > S3y | `416s3x` - `416s3y` | +2.38 [+1.06, +5.54], p 0.016 | +2.11 [+0.78, +3.65], p 0.031 | **holds** |
| P4 pedestrian: S3x >= uniform 480 x 256 | `416s3x` - `480x256` | -0.51, 90 % [-1.60, +0.29], p 0.14 | +0.77, 90 % [-0.62, +1.77], p 0.38 | **inconclusive** (ByteTrack is primary; OC-SORT alone would be non-inferior) |

- P1 is significant with OC-SORT but not with ByteTrack, whose exact randomisation p is 8/128 (6 of 7 sequences favour
  S3x), so it misses the rule that requires both trackers. Under the pre-registered wording it is therefore
  "not shown". It is not "refuted" either: that would need the 95 % interval to exclude +0.5 (upper bound < +0.5), and
  this interval does not.
- S3x - plain by camera type (descriptive):
  - static cameras (sequences 02, 04, 09): +3.08 (ByteTrack) and +2.52 (OC-SORT);
  - moving cameras (sequences 05, 10, 11, 13): +1.99 and +1.16.
- The MOT17 test has little power, that is, a real effect can easily fail to reach significance. It has 7 sequences
  and 2 seeds, and most of its pedestrians are at least one P3 grid stride (8 px) wide: 18 % have a shorter side under
  8 px when frames are scaled to 416 px on the long side, against 76 % on VisDrone. Apart from the same-FLOPs
  comparison with ByteTrack (-0.51), every pedestrian contrast points the same way as on DSEC, BDD100K and VisDrone.

All contrasts with Holm-adjusted p-values and labels, for DSEC and BDD100K too, are in
[`results/RESULTS.md`](../results/RESULTS.md).
