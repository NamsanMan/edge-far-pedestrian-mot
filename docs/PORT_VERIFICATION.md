# Port verification: this code against the original research outputs

The code of this repository is a port of the original research code. This page records how it was checked against the
outputs of that original code (the detection caches, tracker outputs, TrackEval files, comparison-protocol results,
HIT-UAV evaluations and app runs behind `results/` and `apps/thermal_watch`) and the measured differences. The checks
need those original outputs, which are not published, so this page is a record rather than a recipe;
[`REPRODUCE.md`](REPRODUCE.md) explains how to run the pipeline yourself.

Environment: Windows 11, Python 3.10, torch 2.8.0+cu129, ultralytics 8.3.170, boxmot 25.0.0, numpy 2.2.6,
scipy 1.15.3, opencv-python 4.11.0.86, TrackEval at commit 12c8791b.

## Summary

An arm is a detector variant, such as YOLOv8n + S3x on the 416 x 320 canvas, trained with several seeds; a contrast is
one comparison of two arms by the statistical protocol of [`PROTOCOL.md`](PROTOCOL.md). A run tag such as
`dsec_v8n_416s3x_s0` names one trained model (dataset, detector, variant, seed; see [`REPRODUCE.md`](REPRODUCE.md)).

| Step | What was compared | Scope | Result |
|---|---|---|---|
| (a) model | state dict, loss, gradients, `decode_feats`, `infer`, new code vs original code, same device | all 6 families | bit-identical |
| (a) weights | the 4 release files vs the original `last.pth` | all 355 tensors each | bit-identical |
| (a) export, same device | `scripts/export_dets.py` vs the original export, both on CPU | MOT17 S3x (2,652 frames); DSEC v8n / y11n S3x (25,617 frames) | MOT17 bit-identical. DSEC within 3e-4 px (CPU thread count), equal box counts in 25,616 of 25,617 frames |
| (a) export vs the published caches | new CPU FP32 export vs the original FP16 GPU caches | 2 DSEC test sequences x 2 models, 1 MOT17 half, 1 DSEC sequence (YOLO11n) | 99.60-99.83 % of the confident boxes are reproduced (IoU >= 0.9) |
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

## Notes

- **CPU export vs the published caches.** The published DSEC and MOT17 caches were exported in FP16 on a GPU, and FP32
  on a CPU does not give identical boxes; the differences are mostly low-score boxes near the 0.01 confidence floor.
  Steps (b)-(d) therefore start from the original caches; together with the same-device export check in (a), they
  show that the new code gives the same outputs. A GPU FP16 re-export of the DSEC and MOT17 runs was not done.
- **Protocol table.** When the repository was built, every row of the results table at that time (66 rows) was checked
  against the original `protocol.py --one`. [`results/RESULTS.md`](../results/RESULTS.md) now has 64 rows: one VisDrone
  car contrast had appeared twice (S3x vs S3y and its reverse, prediction P3 of
  [`PREREGISTRATION.md`](PREREGISTRATION.md)) and is now listed once, which changes only the Holm adjustment of the
  VisDrone car rows.
- **HIT-UAV on another GPU.** Re-exported on an RTX 5060 Ti instead of the original Colab L4, two cells of
  [`results/HITUAV.md`](../results/HITUAV.md) would change in the first decimal (plain YOLOv8n at 448 x 352 on the
  folds, 60.2 -> 60.1; S3x on altLH, 34.1 -> 34.2). The S3x gains stay almost the same: +7.10 -> +7.08 on the folds
  and +5.63 -> +5.65 on altLH. Full trainings were not repeated. The P2 arm (code not included) is scored from its
  original caches only.
- **Detector comparison.** The 68 statistics files of the comparison (YOLO26n, YOLOX-Nano / -Tiny, NanoDet-Plus-m,
  YOLOv8s, RT-DETR-L) were extracted with `scripts/extract_stats.py` from the original TrackEval outputs; re-extracting
  5 run tags already in `results/stats` gives identical arrays.

**Changes after the check.** Two code paths changed after these checks, without changing any output:
- The training loader no longer reads and augments the previous frame of each frame pair, which the detector loss never
  used (`FramePairs.load_prev = False` in the training loop). The current frame, its targets and the random draws are
  unchanged; `test_pairs_without_prev` in `tests/test_data.py` checks this.
- `scripts/export_dets.py` loads checkpoints with `weights_only=True` first and no longer passes `allow_pickle` to
  `np.savez_compressed`: under NumPy >= 2.2, which the repository requires, pickling is allowed by default. A CPU
  re-export of `mot_v8n_416s3x_s0` on MOT17-09-FRCNN is bit-identical to the export of the check (262 / 262 frames).
