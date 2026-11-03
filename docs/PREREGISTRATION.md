# Pre-registrations: VisDrone2019-MOT and MOT17

S3x was designed and first tested on DSEC and BDD100K, both driving data. Before training anything on two new domains, a
drone dataset (VisDrone2019-MOT) and static / moving street cameras (MOT17), the predictions and reading rules were
written down. They are reproduced here with their dates. Only these edits were made:

- internal file names were replaced by the names of this repository's modules;
- lines that scheduled the training jobs were removed;
- scratch and backup paths were removed.

The outcomes are in [`PROTOCOL.md`](PROTOCOL.md#pre-registered-tests-outcomes) and
[`results/RESULTS.md`](../results/RESULTS.md).

Notation:
- `416s3x` = YOLOv8n + S3x on a 416-wide canvas;
- `704x384` = plain YOLOv8n on a 704 x 384 canvas;
- p_AR = paired randomisation p-value (two-sided, unadjusted);
- `noninferior_90` = the lower bound of the 90 % bootstrap interval of d = A - B is above -margin.

---

## VisDrone2019-MOT

### Pre-registration (2026-10-30, before any VisDrone training)

**Ground-truth measurement.**
- VisDrone train+val, consider flag 1. MOT17-train FRCNN copies, class 1, visibility >= 0.25.
- Frames letterboxed to a long side of 416 / 640 px.

| data | class | n | median w/h | width is the short side | short side < 8 px @416 / @640 | of those, width short @640 |
|---|---|---|---|---|---|---|
| VisDrone | pedestrian | 266,709 | 0.47 | 98 % | 76 % / 40 % | 97 % |
| VisDrone | people | 112,304 | 0.61 | 93 % | 71 % / 33 % | 97 % |
| VisDrone | car | 537,122 | 1.26 | 26 % | 37 % / 12 % | 14 % (height short 86 %) |
| MOT17 | pedestrian | 85,828 | 0.35 | 100 % | 18 % / 8 % | 100 % |

**Rule under test.** A small object is missed when its SHORT side falls below about one P3 stride, so extra resolution
helps along the short side. VisDrone drone views are oblique, so pedestrians stay tall and narrow (width short) and cars
stay wide (height short), as in the driving data.

**Setup (fixed now).**
- YOLOv8n, letterbox canvas 640 x 384. This is the VisDrone main resolution: at 416, 76 % of pedestrians are below 8 px.
- Train on VisDrone2019-MOT-train; evaluate on VisDrone2019-MOT-test-dev (17 sequences).
- ByteTrack and OC-SORT with cross-fitted (leave-one-sequence-out) tracker thresholds, 2 seeds, the same harness as
  DSEC / BDD.
- Arms: plain 640 x 384, S3x 640 x 384, S3y 640 x 384, and a compute-matched uniform 704 x 384 (+10 % pixels; S3x is
  about +11 % FLOPs).

**Predictions (decided before training).**
- P1 pedestrian HOTA: S3x - plain > 0 (significant, both trackers).
- P2 pedestrian HOTA: S3x - S3y > 0 (axis specificity).
- P3 car HOTA: (S3y - plain) >= (S3x - plain) (cars are height-limited).
- P4 efficiency: pedestrian HOTA S3x - uniform 704 x 384 >= 0.

**Reading.** If P1 and P2 hold, the rule transfers to drone views and the project may be named for edge cameras. If P1
fails, the project keeps the driving-camera scope.

### Addendum (2026-10-30, before any VisDrone training or result; the pre-registration itself is unchanged)

Fixed after the code review of the VisDrone integration, with no VisDrone run, detection cache or result in existence.

1. **Car class.** The evaluated car is VisDrone category 4 only, exactly the pre-registered car row (n = 537,122 = train
   505,301 + val 31,821).
   - Van (category 5; 53,782 train+val boxes, 11,223 in test-dev) is trained and scored as its own class in the BDD head
     `train`, which has no VisDrone counterpart. A car prediction on a van is a car false positive, as in the official
     VisDrone-MOT evaluation (car and van are separate classes there).
   - The first implementation had merged van into car; it was changed before any training.
   - Pedestrian = category 1 only and people = head `rider`, unchanged (the pre-registration rows reproduce exactly).
2. **Training mask.** Besides the ignored regions (categories 0 and 11), the GT boxes removed by the
   centre-in-ignored-region rule are greyed out as well (except pixels of kept GT boxes). So no part of a removed object
   is trained as background. Same for every arm.
3. **Equivalence margin.** Frozen at 1.1955 HOTA, the DSEC value: same leave-one-sequence-out design, 1512 vs 1510
   iterations per epoch.
   - The seed-SD rule of the other datasets would rest on 2 arms x 2 seeds here. Its 5-95 % range would be 0.40-3.17 x
     the true seed SD, so it is printed for reference only.
4. **Reading rules.** Cross-fitted thresholds, `efpmot.eval.protocol.one("visdrone", "v8n", A, B, ...)`.
   - **P1** (`640s3x` vs `640`, pedestrian) and **P2** (`640s3x` vs `640s3y`, pedestrian) hold iff d > 0 and
     p_AR < 0.05 for ByteTrack AND for OC-SORT. This is "significant, both trackers"; requiring both needs no
     multiplicity correction.
   - **P3** (`640s3y` vs `640s3x`, car; the plain arm cancels) and **P4** (`640s3x` vs `704x384`, pedestrian):
     - holds iff non-inferior, i.e. the 90 % bootstrap interval lower bound > -1.1955 (`noninferior_90`);
     - fails iff d < 0 with p_AR < 0.05;
     - otherwise inconclusive.

     ByteTrack is primary; OC-SORT is reported beside it.
   - **Sensitivity**, reported but not deciding: `sensitivity_video` (leave-one-video-out thresholds and a bootstrap /
     randomisation over the 14 source videos of the 17 test-dev clips), and the best-on-test and fixed-0.5 rules.
5. **Known ceiling.** Perfect detections through the unchanged trackers (t = 0.5, all 17 test-dev sequences, car =
   category 4 GT of this addendum) do not reach HOTA 100. Two things cap the measurable HOTA for every arm alike:
   - association on small, fast boxes (consecutive-frame IoU below the gate);
   - whole-view camera jumps in 6 of the 17 clips.

---

## MOT17 (street cameras)

### Pre-registration (2026-10-30, before any MOT17 training)

**GT measurement.**
- MOT17-train FRCNN copies, class 1, visibility >= 0.25, n = 85,828.
- Pedestrian median w/h 0.35; the width is the short side in 100 % of the boxes.
- Short side < 8 px: 18 % at a 416 long side and 8 % at 640 (VisDrone pedestrians: 76 % / 40 %).
- Expected effect: the same sign as DSEC / BDD, smaller size.

**Setup (fixed now).**
- MOT17 train sequences only (the test GT is not public), one image copy per sequence (FRCNN).
- The standard half split: the first half of the frames of each of the 7 sequences for training, the second half for
  evaluation.
- YOLOv8n fine-tuned with the same harness, letterbox canvas 416 x 256 (the BDD edge budget; frames are 16:9 except
  MOT17-05, which is 4:3).
- Evaluation: pedestrian HOTA with TrackEval's MOTChallenge evaluator (distractor classes removed as in the official
  preprocessing), ByteTrack and OC-SORT, cross-fitted leave-one-sequence-out thresholds over the 7 evaluation halves,
  2 seeds.
- The static-camera sequences (02, 04, 09) are also reported separately (descriptive only: 3 sequences).
- Arms: plain 416 x 256, S3x 416 x 256, S3y 416 x 256, and a compute-matched uniform 480 x 256 (+15 % pixels; S3x is
  about +11 % FLOPs).

**Predictions (decided before training).**
- P1 pedestrian HOTA: S3x - plain > 0.
- P2 pedestrian HOTA: S3x - S3y > 0.
- P4 efficiency: S3x - uniform 480 x 256 >= 0.

With 7 sequences and 2 seeds the test has little power. A null P1 is "not shown", not "refuted", unless the CI
excludes +0.5.

### Addendum (2026-10-30, before any MOT17 training or result; the pre-registration itself is unchanged)

Fixed with the implementation of the MOT17 data path (`efpmot.data.mot17`), with no MOT17 run, cache or result in
existence.

1. **Half split.** The CenterTrack / ByteTrack / FairMOT "half" convention (`convert_mot_to_coco.py`). With
   n = seqLength and 0-based image index i:
   - training = i in [0, n//2] (frames 1 .. n//2+1);
   - evaluation = i in [n//2+1, n-1] (frames n//2+2 .. n).

   No frame is in both halves (tested); the 7 FRCNN sequences give 2,664 training and 2,652 evaluation frames.

   TrackEval has no built-in half convention, so the evaluation GT is written as its own MOTChallenge folder
   (`<derived>/mot17/gt_halfval/<seq>/{gt/gt.txt, seqinfo.ini}`):
   - every raw gt line of the evaluation half, all classes, so the official distractor preprocessing works;
   - the frame renumbered frame - (n//2+1), so the first evaluation frame is 1, as CenterTrack's `gt_val_half.txt`;
   - all other columns verbatim; seqLength = n - (n//2+1);
   - seqmaps `MOT17-halfval-{all,static,moving}.txt`.

   Tracker outputs use the same frame numbers. The raw gt.txt / seqinfo.ini are byte-identical to TrackEval's
   MOT17-train copies.
2. **Classes.** MOT17 class 1 (pedestrian) -> the 8-class head `pedestrian`, all visibilities. In MOT17-train,
   consider = 1 for every class-1 line and 0 for every other line. The official evaluation scores all class-1 boxes, and
   ByteTrack's MOT17 conversion also keeps them.
   - Training boxes are clipped to the frame (16,368 of 112,297 extend outside; 2 have no area inside and are dropped).
     Evaluation GT is not clipped.
   - The other 7 heads are trained as background everywhere (they are never evaluated or tracked).
3. **Distractors.** Classes 2 person_on_vehicle, 7 static_person, 8 distractor and 12 reflection (19,750 boxes).
   - In training they are greyed out (114, the letterbox fill) in both frames of a pair before augmentation, except
     pixels inside a kept pedestrian box of that frame. This is the VisDrone ignored-region rule: they are taught neither
     as pedestrian nor as background.
   - Vehicles (3-6), occluders (9-11) and crowd (13) stay unmasked background: the official evaluation counts a
     prediction on them as a false positive.
   - Evaluation: TrackEval MOTChallenge2DBox, BENCHMARK MOT17, DO_PREPROC. Tracker boxes matched at IoU >= 0.5 to a
     distractor GT box are removed; GT = class 1 with consider != 0.
   - Tested: GT plus exact copies of all 9,762 evaluation-half distractor boxes as tracker output gives HOTA 100, FP 0.
4. **Epochs: 55.** The 2,664 training frames give 166 iterations per epoch at batch 16 (drop_last). 55 epochs =
   9,130 iterations, the closest whole number of epochs to the DSEC (6 x 1510 = 9,060) and VisDrone (6 x 1512 = 9,072)
   schedules (54 epochs = 8,964). Same for every arm. Consequences, accepted in advance:
   - each training frame is seen about 55 times (6 for DSEC / VisDrone);
   - the harness's warm-up rule min(500, iterations per epoch) gives 166 warm-up iterations instead of 500 (the harness
     is not changed);
   - the cosine schedule runs over 9,130 iterations;
   - a checkpoint is written every epoch.
5. **Canvases.** Letterbox 416 x 256 (H 256, W 416) for plain / S3x / S3y, 480 x 256 for the uniform arm.
   - 1920 x 1080 frames: content 416 x 234 vs 455 x 256 (+19.7 % content pixels, +15.4 % canvas pixels).
   - MOT17-05 (640 x 480) is height-limited: its content is 341 x 256 on BOTH canvases. So the uniform arm adds
     resolution only to the six 16:9 sequences, while S3x / S3y act on all seven.
6. **Tracking.** Per-class trackers as everywhere, pedestrian only, with unchanged tracker parameters: frame_rate 30,
   track_buffer 15, ByteTrack thresholds 0.2-0.9, OC-SORT 0.5-0.9. MOT17-05 (14 fps) and MOT17-13 (25 fps) are not given
   their own frame rate.
7. **Equivalence margin.** Frozen at 1.1955 HOTA, the DSEC / VisDrone value: same leave-one-sequence-out design, 9,130 vs
   9,060 / 9,072 iterations.
   - The seed-SD rule (2 x median within-method seed SD of pooled ByteTrack pedestrian HOTA, cross-fitted rule, arms
     uniform 416 x 256 and 480 x 256) would rest on 2 arms x 2 seeds and is printed for reference only.
8. **Reading rules.** Cross-fitted thresholds, leave-one-sequence-out over the 7 evaluation halves:
   `efpmot.eval.protocol.one("mot17", "v8n", A, B, cls="pedestrian", trk=...)`. p_AR is exact over the 2^7 = 128 swap
   patterns, so the smallest attainable p is 2/128 = 0.0156. The rules are identical to the VisDrone addendum:
   - **P1** (`416s3x` vs `416`) and **P2** (`416s3x` vs `416s3y`) hold iff d > 0 and p_AR < 0.05 for ByteTrack AND for
     OC-SORT (requiring both needs no multiplicity correction).
     - As in the pre-registration, a null P1 is "not shown", not "refuted", unless the CI excludes +0.5.
     - The CI is fixed here as the 95 % bootstrap interval `ci95` of the same comparison (upper bound < +0.5).
   - **P4** (`416s3x` vs `480x256`):
     - holds iff non-inferior, i.e. the 90 % bootstrap interval lower bound > -1.1955 (`noninferior_90`);
     - fails iff d < 0 with p_AR < 0.05;
     - otherwise inconclusive.

     ByteTrack is primary; OC-SORT is reported beside it.
   - There is no P3: MOT17 has no evaluated car class.
   - **Static-camera split** (`static_split`): seed-mean pooled HOTA of A and B, and d, over 02 / 04 / 09 and over
     05 / 10 / 11 / 13, each sequence at its cross-fitted threshold. Descriptive only (3 and 4 sequences): no test, no
     reading rule.
   - Best-on-test and fixed-0.5 rules are reported as sensitivity, not deciding.
9. **Known ceiling.** GT pedestrian boxes of the evaluation halves were run as detections through the unchanged trackers
   (clipped to the frame like the export, score 1, t = 0.5):

   | tracker | HOTA | static | moving | 02 | 04 | 05 | 09 | 10 | 11 | 13 |
   |---|---|---|---|---|---|---|---|---|---|---|
   | ByteTrack | 86.85 | 87.02 | 86.38 | 95.7 | 83.7 | 76.2 | 86.4 | 92.3 | 90.8 | 78.5 |
   | OC-SORT | 90.04 | 89.21 | 91.95 | 98.2 | 85.7 | 83.5 | 89.6 | 97.1 | 94.0 | 88.0 |

   The cap comes from association (occlusions longer than the 15-frame buffer, the 14 fps MOT17-05 and the moving
   MOT17-13) and from the clipped boxes, the same for every arm.
10. **Ceiling note** (added 2026-10-30, before any MOT17 training; predictions unchanged).
    - Besides association limits, the cap also comes from evaluation-half pedestrian boxes that lie mostly outside the
      frame: 3,042 of 53,890 have less than half their area inside (2,491 in MOT17-04, 10.3 % of its boxes, mostly
      people leaving through the top edge).
    - The export clips to the frame, so a detection of the visible part cannot reach IoU 0.5 with them.
    - Clipped GT boxes with their true ids as tracker output score HOTA 91.48 (MOT17-04 85.66, MOT17-05 89.76, others
      94.8-99.1).
    - The same cap applies to every arm.
