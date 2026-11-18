# Port verification: this code against the original research outputs

The code of this repository is a port of the original research code. This page records how it was checked against the
outputs of that original code: the detection caches, tracker outputs, TrackEval files, comparison-protocol results,
HIT-UAV evaluations and app runs behind `results/` and `apps/thermal_watch`. It gives the exact commands and the
measured differences. [`REPRODUCE.md`](REPRODUCE.md) explains how to run the pipeline yourself, and the
[README](../README.md) defines S3x, the canvas and HOTA.

Placeholders:
- `<ref>`: the folder that holds the original outputs:
  - `<ref>/runs/<tag>/last.pth`
  - `<ref>/cache/<tag>/<dataset>/<seq>.npz`
  - `<ref>/results/<dataset>_trackers/...`
  - `<ref>/results/<dataset>_eval/...`
  - `<ref>/analysis/thermal_watch/_demo_scenes/<video>/events.jsonl` (the original Thermal Watch runs)
- `<tag>`: a run tag, which names one trained model (a run) by dataset prefix, detector, variant and seed. For
  example, `dsec_v8n_416s3x_s0` is YOLOv8n + S3x trained on DSEC at the 416 x 320 canvas with seed 0, and
  `dsec_y11n_416s3x_s0` is the same with YOLO11n; HIT-UAV tags add the fold, as in `hit_v8n_416s3xrecA_s0`. The scheme
  is in [`REPRODUCE.md`](REPRODUCE.md). `<dataset>`, `<seq>` and `<video>` stand for a dataset, a sequence and a sample
  video.
- `ref_results.yaml`: a copy of `configs/paths.yaml` with `outputs.results: <ref>/results`. With the environment
  variable `EFPMOT_PATHS` pointing to it, `scripts/compare.py` in (d) reads the original TrackEval outputs. The original
  outputs are only read.
- `<video_sample>`: the folder of the five HIT-UAV sample videos (`data.hituav_videos` in `configs/paths.yaml`).

Environment:
- Windows 11.
- Python 3.10.
- torch 2.8.0+cu129, ultralytics 8.3.170, boxmot 25.0.0, numpy 2.2.6, scipy 1.15.3, opencv-python 4.11.0.86.
- TrackEval at commit 12c8791b.

Every command runs from the repository root. Outputs go to `outputs/`, which is git-ignored, with three exceptions:
`package_weights.py` in (a) writes to `release_assets/` (including `SHA256SUMS.txt`), the commands in (d) write
`a.json` and `b.json` to the repository root, and (g) rewrites `results/summary_detectors*` and `results/DETECTORS.md`.

## Summary

Each row is one check, described in the section with the same letter below. An arm is a detector variant, such as
YOLOv8n + S3x on the 416 x 320 canvas, trained with several seeds; a contrast is one comparison of two arms by the
statistical protocol of [`PROTOCOL.md`](PROTOCOL.md).

| Step | What was compared | Scope | Result |
|---|---|---|---|
| (a) model | state dict, loss, gradients, `decode_feats`, `infer`, new code vs original code, same device | all 6 families | bit-identical |
| (a) weights | the 4 release files vs the original `last.pth` | all 355 tensors each | bit-identical |
| (a) export, same device | `scripts/export_dets.py` vs the original export, both on CPU | MOT17 S3x (2,652 frames); DSEC v8n / y11n S3x (25,617 frames) | MOT17 bit-identical. DSEC within 3e-4 px (CPU thread count), equal box counts in 25,616 of 25,617 frames |
| (a) export vs the published caches | new CPU FP32 export vs the original FP16 GPU caches | 2 DSEC test sequences x 2 models, 1 MOT17 half, 1 DSEC sequence (YOLO11n) | 99.60-99.83 % of the confident boxes are reproduced (IoU >= 0.9). A GPU FP16 re-export is pending (see the last section) |
| (b) tracking | `scripts/track.py` on the original caches vs the original tracker outputs | 52 tracker configurations (both trackers, full threshold grid), 78 files | 78 / 78 byte-identical |
| (c) TrackEval | `scripts/evaluate.py --seqs` vs the per-sequence rows of the original `*_detailed.csv` | 52 configurations, 143 class x sequence rows, 275 columns each | all equal |
| (d) protocol | `scripts/compare.py --one` (from the original TrackEval CSVs and from `results/stats`) vs the original `protocol.py --one` | 8 contrasts | identical JSON (all keys, same key order) |
| (e) Thermal Watch app | `apps/thermal_watch/tests` | 6 tests | 6 / 6 pass |
| (e) Thermal Watch runs | `apps/thermal_watch/thermal_watch.py` vs the original app, same checkpoint (`hit_v8n_416s3xrecA_s0`) | 2 sample-video slices (70m-90_1 0-20 s, 130m-60_3 24-34.5 s) | every alert equal in number, frame and distance (13 and 5 alerts); alert boxes identical |
| (f) HIT-UAV split | `scripts/prepare_data.py hituav --compare` vs the original split file | 264 recordings, 38 flights, folds recA / recB / altLH | identical |
| (f) HIT-UAV evaluator | `scripts/evaluate_hituav.py` on the original detection caches vs the original evaluation | 51 runs | identical in every field (`results/hituav/`, `results/HITUAV.md` come from this) |
| (f) HIT-UAV export, same GPU | the repository export vs the original export, both on one RTX 5060 Ti | 5 runs, 7,419 frames | bit-identical |
| (f) HIT-UAV training data | items, labels, augmented samples, RNG state and batches of the training loader | official / recA / recB / altLH | bit-identical; 20 deterministic iterations of S3x on recA give the same parameter hash and losses |
| (f) HIT-UAV re-export on another GPU | RTX 5060 Ti re-export + evaluation vs the original Colab L4 caches | 45 runs (P2 excluded) | test person AP50 within 0.55 per run, 0.15 per arm mean (FP16 differences only) |
| (f) HIT-UAV release files | the 4 release files vs the original `last.pth` | 355 tensors each | bit-identical |
| (g) detector comparison | `scripts/compare.py --detectors` from `results/stats` vs the original analysis | 11 arms x 2 classes x 2 trackers, 308 numbers | identical (largest difference 0); `results/RESULTS.md` unchanged |

**Changes after the check.** Two code paths changed after these checks, without changing any output:
- The training loader no longer reads and augments the previous frame of each frame pair, which the detector loss never
  used (`FramePairs.load_prev = False` in the training loop). The current frame, its targets and the random draws are
  unchanged; `test_pairs_without_prev` in `tests/test_data.py` checks this.
- `scripts/export_dets.py` loads checkpoints with `weights_only=True` first and no longer passes `allow_pickle` to
  `np.savez_compressed`: under NumPy >= 2.2, which the repository requires, pickling is allowed by default. A CPU
  re-export of `mot_v8n_416s3x_s0` on MOT17-09-FRCNN is bit-identical to the export of the check below (262 / 262
  frames).

## (a) Models, weights and detection export

**Models.** For all six families (`yolov8n`, `yolov8n_s3x`, `yolov8n_s3y`, `yolo11n`, `yolo11n_s3x`, `yolo11n_s3y`), the
new code and the original code were run in separate processes with `torch.manual_seed(0)` on CPU. These are
bit-identical:
- the COCO-initialised state dict;
- the loss and the loss items;
- every gradient;
- `decode_feats` and `infer`.

Every original checkpoint on the release list loads with `strict=True` into `efpmot.models.build(family, 8)`, where 8 is
the number of classes. So do two checkpoints that are not released, `vis_v8n_640s3x_s0` and `bdd_y11n_416s3x_s0`.

**Release weights.** `python -B scripts/package_weights.py --runs_dir <ref>/runs` strips the original training
checkpoints into release files. Each of the four DSEC and MOT17 files equals the fp32 state dict of the original
`last.pth`: all 355 tensors are equal under `torch.equal`.

**Export: new code vs original code on the same device.**
- Both exports ran on CPU.
- MOT17 `mot_v8n_416s3x_s0`: every one of the 2,652 frames is bit-identical.
- DSEC `dsec_v8n_416s3x_s0` and `dsec_y11n_416s3x_s0`: the largest box differences are 2e-4 px and 3e-4 px, with equal
  box counts in 25,616 of 25,617 frames. The difference comes from the CPU thread count (4 vs 6 threads): with either
  code, the same frame differs by 1.5e-4 px between the two thread counts.

**Export vs the published FP16 GPU caches.** The published caches were exported in FP16 on a GPU. The commands below
export in FP32 on the CPU, and the table gives the numbers of that CPU run; with `--device cuda` instead of
`--device cpu` they follow the FP16 recipe of the published caches.

```bash
python -B scripts/export_dets.py --data dsec --name dsec_v8n_416s3x_s0 --weights <ref>/runs/dsec_v8n_416s3x_s0/last.pth \
    --device cpu --seqs thun_01_b zurich_city_14_b
python -B scripts/export_dets.py --data dsec --name dsec_v8n_416_s0 --weights <ref>/runs/dsec_v8n_416_s0/last.pth \
    --device cpu --seqs thun_01_b zurich_city_14_b
python -B scripts/export_dets.py --data mot17 --name mot_v8n_416s3x_s0 --weights <ref>/runs/mot_v8n_416s3x_s0/last.pth \
    --device cpu --seqs MOT17-09-FRCNN
python -B scripts/export_dets.py --data dsec --name dsec_y11n_416s3x_s0 --weights <ref>/runs/dsec_y11n_416s3x_s0/last.pth \
    --device cpu --seqs zurich_city_13_b
python -B scripts/diff_outputs.py cache outputs/cache/dsec_v8n_416s3x_s0/dsec <ref>/cache/dsec_v8n_416s3x_s0/dsec
# likewise for the other three
```

| Run | Sequences | Frames | Reference boxes (original cache) | Frames with a different box count | Confident boxes reproduced (score >= 0.3, same class, IoU >= 0.9) |
|---|---|---|---|---|---|
| `dsec_v8n_416s3x_s0` | thun_01_b, zurich_city_14_b | 1,656 | 21,453 | 270 | 5,489 / 5,503 (99.75 %) |
| `dsec_v8n_416_s0` | thun_01_b, zurich_city_14_b | 1,656 | 18,970 | 205 | 5,177 / 5,186 (99.83 %) |
| `mot_v8n_416s3x_s0` | MOT17-09-FRCNN | 262 | 2,674 | 16 | 1,926 / 1,931 (99.74 %) |
| `dsec_y11n_416s3x_s0` | zurich_city_13_b | 315 | 7,291 | 98 | 1,744 / 1,751 (99.60 %) |

- FP32 on the CPU and FP16 autocast on the GPU do not give identical boxes.
- The different box counts come from low-score boxes near the 0.01 confidence floor of the export, which appear in one
  cache but not in the other.
- Because of this, steps (b)-(d) use the original caches as input. The identity of the new and the original outputs
  follows from the same-device export check in (a) above together with (b)-(d).

## (b) Tracking

The original caches of the sequences above were copied to `outputs/cache_orig/<tag>/<dataset>/`. Each run was then
tracked over the whole grid of detection thresholds: ByteTrack at 8 thresholds and OC-SORT at 5. One folder is written
per tracker and threshold, named `<tag>__<tracker>_t<threshold>` (for example `dsec_v8n_416s3x_s0__bytetrack_t0.5`).

```bash
python -B scripts/track.py --dataset dsec  --tag dsec_v8n_416s3x_s0  --cache_root outputs/cache_orig
python -B scripts/track.py --dataset dsec  --tag dsec_v8n_416_s0     --cache_root outputs/cache_orig
python -B scripts/track.py --dataset dsec  --tag dsec_y11n_416s3x_s0 --cache_root outputs/cache_orig
python -B scripts/track.py --dataset mot17 --tag mot_v8n_416s3x_s0   --cache_root outputs/cache_orig
python -B scripts/diff_outputs.py trackers outputs/results/dsec_trackers/dsec_v8n_416s3x_s0__bytetrack_t0.5 \
    <ref>/results/dsec_trackers/dsec_v8n_416s3x_s0__bytetrack_t0.5        # for each of the 52 folders
```

Result: all 78 sequence files in the 52 tracker folders are byte-identical to the original tracker outputs.

The tracker-output writers produce the same files as the original code:
- DSEC: BDD-format JSON, written only for the labelled frames.
- MOT17: MOTChallenge text files, with the frames renumbered from 1 at the start of the evaluation half.

## (c) TrackEval

`--seqs` evaluates only the given sequences and writes the results to a separate subset folder, here `test_sel_<hash>`,
where `<hash>` is a short hash of the sequence names. The per-sequence rows of `<cls>_detailed.csv` (one file per
class) do not depend on which other sequences are evaluated, so they can be compared one to one with the full original
evaluation. The COMBINED row, which aggregates over the evaluated sequences, is not compared.

```bash
python -B scripts/evaluate.py --dataset dsec  --tag dsec_v8n_416s3x_s0  --seqs thun_01_b zurich_city_14_b
python -B scripts/evaluate.py --dataset dsec  --tag dsec_v8n_416_s0     --seqs thun_01_b zurich_city_14_b
python -B scripts/evaluate.py --dataset dsec  --tag dsec_y11n_416s3x_s0 --seqs zurich_city_13_b
python -B scripts/evaluate.py --dataset mot17 --tag mot_v8n_416s3x_s0   --seqs MOT17-09-FRCNN
python -B scripts/diff_outputs.py eval outputs/results/dsec_eval/test_sel_<hash>/dsec_v8n_416s3x_s0__bytetrack_t0.5 \
    <ref>/results/dsec_eval/test/dsec_v8n_416s3x_s0__bytetrack_t0.5 --cls pedestrian car    # each folder
```

Result: in 52 of 52 tracker folders, every per-sequence row is equal in all 275 columns. Summed over the folders, these
are 143 rows, one per class and sequence in each folder: pedestrian and car for DSEC, pedestrian for MOT17.

Examples at tracker threshold 0.5. The HOTA column is the mean over the 19 localisation (IoU) thresholds:

| Run, tracker | Class | Sequence | HOTA (new = original) |
|---|---|---|---|
| `dsec_v8n_416s3x_s0`, ByteTrack 0.5 | pedestrian | thun_01_b | 64.47 |
| `dsec_v8n_416s3x_s0`, ByteTrack 0.5 | pedestrian | zurich_city_14_b | 28.29 |
| `dsec_v8n_416_s0`, ByteTrack 0.5 | pedestrian | thun_01_b | 26.61 |
| `dsec_v8n_416_s0`, ByteTrack 0.5 | pedestrian | zurich_city_14_b | 26.31 |
| `dsec_v8n_416s3x_s0`, OC-SORT 0.5 | car | thun_01_b | 61.57 |
| `dsec_y11n_416s3x_s0`, ByteTrack 0.5 | pedestrian | zurich_city_13_b | 21.56 |
| `mot_v8n_416s3x_s0`, ByteTrack 0.5 | pedestrian | MOT17-09-FRCNN | 44.86 |
| `mot_v8n_416s3x_s0`, OC-SORT 0.5 | pedestrian | MOT17-09-FRCNN | 42.01 |

## (d) Comparison protocol

Each contrast was run three ways:
- the original `protocol.py --one`;
- `scripts/compare.py --one` reading the original TrackEval CSVs (`EFPMOT_PATHS=ref_results.yaml`, `--source trackeval`);
- `scripts/compare.py --one` reading the shipped `results/stats` (`--source stats`).

```bash
EFPMOT_PATHS=ref_results.yaml python -B scripts/compare.py --one dsec v8n 416s3x 416 --cls pedestrian --trk bytetrack \
    --source trackeval --json a.json
python -B scripts/compare.py --one dsec v8n 416s3x 416 --cls pedestrian --trk bytetrack --source stats --json b.json
```

All three JSON outputs are identical: every key, every value and the same key order. That holds for each of these
contrasts:

| Contrast | Class | Tracker | Seeds A / B | d [95 % CI] | p_AR | p_TOST |
|---|---|---|---|---|---|---|
| DSEC v8n S3x 416 vs plain 416 | pedestrian | ByteTrack | 0,1 / 0,1,2 | +5.88 [+4.95, +9.91] | 0.0039 | 0.9999 |
| DSEC v8n S3x 416 vs plain 416 | pedestrian | OC-SORT | 0,1 / 0,1,2 | +5.49 [+4.37, +11.43] | 0.0039 | 0.9996 |
| DSEC v8n S3x 416 vs plain 416 | car | ByteTrack | 0,1 / 0,1,2 | +2.84 [+0.57, +4.67] | 0.0010 | 0.9157 |
| DSEC v8n S3x 416 vs plain 416 | car | OC-SORT | 0,1 / 0,1,2 | +2.18 [+1.15, +3.43] | 0.0076 | 0.9708 |
| MOT17 v8n S3x 416 vs plain 416 | pedestrian | ByteTrack | 0,1 / 0,1 | +2.70 [+0.43, +4.01] | 0.0625 | 0.9035 |
| MOT17 v8n S3x 416 vs plain 416 | pedestrian | OC-SORT | 0,1 / 0,1 | +2.12 [-0.34, +3.73] | 0.0156 | 0.8411 |
| VisDrone v8n S3x 640 vs plain 640 | pedestrian | ByteTrack | 0,1 / 0,1 | +2.58 [+0.24, +5.38] | 0.0038 | 0.8568 |
| VisDrone v8n S3x 640 vs plain 640 | pedestrian | OC-SORT | 0,1 / 0,1 | +3.05 [+0.48, +5.86] | 0.0049 | 0.9120 |

d is the HOTA of arm A minus that of arm B, with its 95 % bootstrap confidence interval. p_AR is the p-value of the
paired randomisation test and p_TOST that of the TOST equivalence test, both before the Holm correction for multiple
comparisons.

When the repository was built, every row of the results table at that time (66 rows, one per contrast, tracker and
class) was checked in the same way against the original `protocol.py --one`. The table in
[`results/RESULTS.md`](../results/RESULTS.md) now has 64 rows. One VisDrone car contrast had appeared twice, as S3x vs
S3y and as its reverse, S3y vs S3x (the pre-registered prediction P3, [`PREREGISTRATION.md`](PREREGISTRATION.md)); it is
now listed once. This changes only the Holm adjustment of the VisDrone car rows.

## (e) Thermal Watch app

```bash
python -B apps/thermal_watch/tests/test_thermal_watch.py
python -B apps/thermal_watch/thermal_watch.py <video_sample>/70m-90_1.mov --weights <ref>/runs/hit_v8n_416s3xrecA_s0/last.pth \
    --family yolov8n_s3x --hw 320 416 --thr 0.154052734375 --dur 20 --out outputs/check/70m-90_1
# likewise 130m-60_3.mov with --start 24 --dur 10.5; compare events.jsonl with <ref>/analysis/thermal_watch/_demo_scenes/<video>/
```

- `last.pth` is a training checkpoint, not a release file, so the command passes `--family`, `--hw` and `--thr`. The
  value of `--thr`, 0.154052734375, is the person threshold of this run: its best-F1 threshold on the val part of fold
  recA, also stored in `efpmot/release.py`.
- The tests pass: 6 / 6. They check the distance geometry, the single alert per confirmed person, a re-acquired person
  keeping its number, a new far person raising its own alert, the rendering of both screens and the sample-video table.
- On both video slices (0-20 s of 70m-90_1 and 24-34.5 s of 130m-60_3, run with CUDA in FP16), every alert of the
  repository app equals the corresponding alert of the original run in person number, frame and distance (13 and 5
  alerts), and the alert boxes are identical (largest difference 0 px).
- `match_videos.py` rebuilds `apps/thermal_watch/sample_videos.json`, the table that links each sample video to its
  HIT-UAV recording, identically from the dataset and the five sample videos.

## (f) HIT-UAV data, training and evaluation

```bash
python -B scripts/prepare_data.py hituav --compare <ref split file splits_recording.json>
# the shipped results: the repository evaluator on the original caches (Colab L4, FP16)
python -B scripts/evaluate_hituav.py "hit_v8n_*" --runs_dir <ref>/runs --cache_root <ref>/cache --out outputs/results/hituav_check
python -B scripts/evaluate_hituav.py --compare <ref>/results/hituav_eval --out outputs/results/hituav_check
# the full path on this machine: re-export on the RTX 5060 Ti, evaluate, compare
python -B scripts/evaluate_hituav.py "hit_v8n_*" --runs_dir <ref>/runs --export --out outputs/results/hituav_rtx5060ti
python -B scripts/evaluate_hituav.py --compare <ref>/results/hituav_eval --out outputs/results/hituav_rtx5060ti
```

- The split rebuilt from the dataset equals the original one: all 264 recordings, 38 flights and the folds recA, recB
  and altLH. In the recording-disjoint folds recA and recB, every recording is in the test part of exactly one fold;
  altLH trains at 60-90 m altitude and tests at 100-130 m. Recordings and flights come from the file names; to balance
  recA against recB, the split also reads the person and car box counts of each recording.
- The evaluator reproduces every field of the 51 original evaluations from the original caches.
- On the same GPU the export is bit-identical (5 runs, 7,419 frames). Re-exported on the RTX 5060 Ti instead of the
  original Colab L4, the test person AP50 of a run differs by at most 0.55, its AP50:95 by at most 0.07 and its
  operating threshold (the person score threshold with the best F1 on the val part) by at most 0.114. The arm means
  differ by at most 0.15 AP50. Two cells of `results/HITUAV.md` would change in the first decimal: plain YOLOv8n at
  448 x 352 (the larger input of about the same FLOPs as S3x) on the recA + recB folds, 60.2 -> 60.1, and S3x on altLH,
  34.1 -> 34.2. The S3x gains stay almost the same (original -> re-export): +7.10 -> +7.08 on the folds and
  +5.63 -> +5.65 on altLH.
- The training loader gives bit-identical items, labels, augmented samples, RNG states and batches on all four splits,
  and 20 deterministic training iterations of S3x on recA end at the same parameter hash and losses. Full trainings were
  not repeated.
- The P2 arm (YOLOv8n with an extra stride-4 P2 detection level, whose code is not included) is scored by the
  repository evaluator from its original caches, so it is not part of the re-export.

## (g) Detector comparison

```bash
python -B scripts/compare.py --detectors results/contrasts_detectors.yaml
```

- The 68 statistics files added for this comparison (YOLO26n, YOLOX-Nano / -Tiny, NanoDet-Plus-m, YOLOv8s, RT-DETR-L)
  were extracted with `scripts/extract_stats.py` from the original TrackEval outputs. Re-extracting 5 run tags that were
  already in `results/stats` gives identical arrays.
- All 308 numbers of the comparison equal the original analysis exactly: HOTA of A and B, d, the 95 % interval, p and
  the Holm-adjusted p, for 11 arms x 2 classes x 2 trackers. With these files added, rerunning
  `scripts/compare.py --table results/contrasts.yaml` leaves `results/RESULTS.md` and `results/summary.csv`
  byte-identical.

## Not yet run: GPU FP16 re-export

The FP16 GPU re-export has not been run yet. These commands finish step (a) on a CUDA GPU:

```bash
python -B scripts/export_dets.py --data dsec --name dsec_v8n_416s3x_s0 --weights <ref>/runs/dsec_v8n_416s3x_s0/last.pth \
    --seqs thun_01_b zurich_city_14_b
python -B scripts/diff_outputs.py cache outputs/cache/dsec_v8n_416s3x_s0/dsec <ref>/cache/dsec_v8n_416s3x_s0/dsec
# likewise for dsec_v8n_416_s0, mot_v8n_416s3x_s0 (--seqs MOT17-09-FRCNN) and dsec_y11n_416s3x_s0 (--seqs zurich_city_13_b)
```

Expected result: equality up to FP16 noise on the same GPU model.
