# Port verification: this code against the original research outputs

This page records how the code of this repository was checked against the outputs of the original research code. Those
outputs are the detection caches, tracker outputs, TrackEval files, protocol results and zone-alert runs behind
`results/`. It gives the exact commands and the measured differences. [`REPRODUCE.md`](REPRODUCE.md) explains how to
run the pipeline yourself.

Placeholders:
- `<ref>`: the folder that holds the original outputs:
  - `<ref>/runs/<tag>/last.pth`
  - `<ref>/cache/<tag>/<dataset>/<seq>.npz`
  - `<ref>/results/<dataset>_trackers/...`
  - `<ref>/results/<dataset>_eval/...`
  - `<ref>/apps/zone_alert/cache/<seq>__<model>.json`
- `ref_results.yaml`: a copy of `configs/paths.yaml` with `outputs.results: <ref>/results`. The original outputs are only
  read.

Environment:
- Windows 11.
- Python 3.10.
- torch 2.8.0+cu129, ultralytics 8.3.170, boxmot 25.0.0, numpy 2.2.6, scipy 1.15.3, opencv-python 4.11.0.86.
- TrackEval at commit 12c8791b.

Every command runs from the repository root. All outputs go to `outputs/`, which is git-ignored.

## Summary

| Step | What was compared | Scope | Result |
|---|---|---|---|
| (a) model | state dict, loss, gradients, `decode_feats`, `infer`, new code vs original code, same device | all 6 families | bit-identical |
| (a) weights | the 4 release files vs the original `last.pth` | all 355 tensors each | bit-identical |
| (a) export, same device | `scripts/export_dets.py` vs the original export, both on CPU | MOT17 S3x (2,652 frames); DSEC v8n / y11n S3x (25,617 frames) | MOT17 bit-identical. DSEC within 3e-4 px (CPU thread count), equal box counts in 25,616 of 25,617 frames |
| (a) export vs the published caches | new CPU FP32 export vs the original FP16 GPU caches | 2 DSEC test sequences x 2 models, 1 MOT17 half, 1 DSEC sequence (YOLO11n) | 99.60-99.83 % of the confident boxes are reproduced (IoU >= 0.9). A GPU FP16 re-export is pending (see the last section) |
| (b) tracking | `scripts/track.py` on the original caches vs the original tracker outputs | 52 tracker configurations (both trackers, full threshold grid), 78 files | 78 / 78 byte-identical |
| (c) TrackEval | `scripts/evaluate.py --seqs` vs the per-sequence rows of the original `*_detailed.csv` | 52 configurations, 143 class x sequence rows, 275 columns each | all equal |
| (d) protocol | `scripts/compare.py --one` (from the original TrackEval CSVs and from `results/stats`) vs the original `protocol.py --one` | 8 contrasts | identical JSON (all keys, same key order) |
| (e) zone-alert app | `apps/zone_alert/tests` | 46 tests | 46 / 46 pass |
| (e) zone-alert replay | `apps/zone_alert/replay_dets.py` on the original detection caches vs the stored app runs | MOT17-04 / -09 x plain / S3x | tracks and every event equal (47, 188, 49 and 211 events) |

**Changes after the check.** Two code paths changed after these checks, without changing any output:
- The training loader no longer reads and augments the previous frame of each pair, which the detector loss never used
  (`FramePairs.load_prev = False` in the training loop). The current frame, its targets and the random draws are the
  same (`tests/test_data.py`, `test_pairs_without_prev`).
- `scripts/export_dets.py` loads checkpoints with `weights_only=True` first and no longer passes `allow_pickle` to
  `np.savez_compressed` (the default under the required NumPy >= 2.2). A CPU re-export of `mot_v8n_416s3x_s0` on
  MOT17-09-FRCNN is bit-identical to the export of the check below (262 / 262 frames).

## (a) Models, weights and detection export

**Models.** For all six families (`yolov8n`, `yolov8n_s3x`, `yolov8n_s3y`, `yolo11n`, `yolo11n_s3x`, `yolo11n_s3y`), the
new code and the original code were run in separate processes with `torch.manual_seed(0)` on CPU. These are
bit-identical:
- the COCO-initialised state dict;
- the loss and the loss items;
- every gradient;
- `decode_feats` and `infer`.

Every original checkpoint of the release list loads with `strict=True` into `efpmot.models.build(family, 8)`. So do
`vis_v8n_640s3x_s0` and `bdd_y11n_416s3x_s0`.

**Release weights.** Run `python -B scripts/package_weights.py --runs_dir <ref>/runs`. Each of the four files equals the
fp32 state dict of the original `last.pth` (355 tensors, `torch.equal`).

**Export: new code vs original code on the same device.**
- Both exports ran on CPU.
- MOT17 `mot_v8n_416s3x_s0`: every one of the 2,652 frames is bit-identical.
- DSEC `dsec_v8n_416s3x_s0` and `dsec_y11n_416s3x_s0`: the largest box differences are 2e-4 px and 3e-4 px, with equal
  box counts in 25,616 of 25,617 frames. The difference comes from the CPU thread count (4 vs 6 threads): the same frame
  differs by 1.5e-4 px between thread counts with either code.

**Export vs the published FP16 GPU caches.** These are the commands. Run them with `--device cuda` for the published FP16
recipe; the numbers below are the CPU run.

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

| Run | Sequences | Frames | Reference boxes | Frames with a different box count | Confident boxes reproduced (score >= 0.3, same class, IoU >= 0.9) |
|---|---|---|---|---|---|
| `dsec_v8n_416s3x_s0` | thun_01_b, zurich_city_14_b | 1,656 | 21,453 | 270 | 5,489 / 5,503 (99.75 %) |
| `dsec_v8n_416_s0` | thun_01_b, zurich_city_14_b | 1,656 | 18,970 | 205 | 5,177 / 5,186 (99.83 %) |
| `mot_v8n_416s3x_s0` | MOT17-09-FRCNN | 262 | 2,674 | 16 | 1,926 / 1,931 (99.74 %) |
| `dsec_y11n_416s3x_s0` | zurich_city_13_b | 315 | 7,291 | 98 | 1,744 / 1,751 (99.60 %) |

- FP32 on CPU against FP16 autocast on the GPU does not give identical boxes.
- The differing counts come from low-score boxes near the 0.01 confidence floor, which appear or disappear.
- Because of this, steps (b)-(e) use the original caches as input. Their identity follows from (a) "same device"
  above together with (b)-(e).

## (b) Tracking

The original caches of the sequences above were copied to `outputs/cache_orig/<tag>/<dataset>/`. Each run was then
tracked over the whole grid: ByteTrack at 8 thresholds and OC-SORT at 5.

```bash
python -B scripts/track.py --dataset dsec  --tag dsec_v8n_416s3x_s0  --cache_root outputs/cache_orig
python -B scripts/track.py --dataset dsec  --tag dsec_v8n_416_s0     --cache_root outputs/cache_orig
python -B scripts/track.py --dataset dsec  --tag dsec_y11n_416s3x_s0 --cache_root outputs/cache_orig
python -B scripts/track.py --dataset mot17 --tag mot_v8n_416s3x_s0   --cache_root outputs/cache_orig
python -B scripts/diff_outputs.py trackers outputs/results/dsec_trackers/dsec_v8n_416s3x_s0__bytetrack_t0.5 \
    <ref>/results/dsec_trackers/dsec_v8n_416s3x_s0__bytetrack_t0.5        # for each of the 52 folders
```

Result: 52 tracker folders, 78 sequence files, all byte-identical to the original tracker outputs.

The writers produce the same files:
- DSEC: BDD-format JSON, labelled frames only.
- MOT17: MOTChallenge text, renumbered to the evaluation half.

## (c) TrackEval

`--seqs` evaluates only the given sequences. The per-sequence rows of `<cls>_detailed.csv` do not depend on which other
sequences are evaluated, so they can be compared one to one with the full original evaluation. The COMBINED row is not
compared.

```bash
python -B scripts/evaluate.py --dataset dsec  --tag dsec_v8n_416s3x_s0  --seqs thun_01_b zurich_city_14_b
python -B scripts/evaluate.py --dataset dsec  --tag dsec_v8n_416_s0     --seqs thun_01_b zurich_city_14_b
python -B scripts/evaluate.py --dataset dsec  --tag dsec_y11n_416s3x_s0 --seqs zurich_city_13_b
python -B scripts/evaluate.py --dataset mot17 --tag mot_v8n_416s3x_s0   --seqs MOT17-09-FRCNN
python -B scripts/diff_outputs.py eval outputs/results/dsec_eval/test_sel_<hash>/dsec_v8n_416s3x_s0__bytetrack_t0.5 \
    <ref>/results/dsec_eval/test/dsec_v8n_416s3x_s0__bytetrack_t0.5 --cls pedestrian car    # each folder
```

Result: in 52 of 52 tracker folders, every per-sequence row is equal in all 275 columns. That is 143 class x sequence rows:
pedestrian and car for DSEC, pedestrian for MOT17.

Examples at threshold 0.5. HOTA is the mean over the 19 localisation thresholds:

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

When the repository was built, each of the 66 contrast x tracker x class rows of the table at that time was checked the
same way against the original `protocol.py --one`. The table now has 64 rows: a VisDrone car contrast that appeared
twice (S3x vs S3y and its reverse, the pre-registered P3) is listed once, which changes only the Holm adjustment of the
VisDrone car rows.

## (e) Zone-alert app

```bash
python -B apps/zone_alert/tests/test_zone_monitor.py
python -B apps/zone_alert/replay_dets.py --cache <ref>/cache/mot_v8n_416s3x_s0/mot17/MOT17-09-FRCNN.npz \
    --zones apps/zone_alert/zones/MOT17-09.json --thr 0.8 --expect <ref>/apps/zone_alert/cache/MOT17-09__s3x.json
# likewise MOT17-04 and the plain model (mot_v8n_416_s0, <seq>__plain.json)
```

- The tests pass: 46 / 46. This includes the replay of the 4 stored runs through the alert logic, with and without an
  ignore mask.
- `replay_dets.py` runs the full app chain from the original FP16 detection caches: pedestrian boxes, then the app's
  ByteTrack, then `ZoneMonitor`. It runs on CPU and needs no images.
- On all 4 runs, the tracks equal the stored app runs on every frame (262 or 524 frames). Every event equals the stored
  one in every field:

| Sequence | Model | Events (stored = replayed) |
|---|---|---|
| MOT17-09 | S3x | 47 (27 INTRUSION, 20 EXIT) |
| MOT17-04 | S3x | 188 (99 INTRUSION, 80 EXIT, 9 LOITERING) |
| MOT17-09 | plain | 49 (26 INTRUSION, 20 EXIT, 3 LOITERING) |
| MOT17-04 | plain | 211 (111 INTRUSION, 91 EXIT, 9 LOITERING) |

## Not yet run: GPU FP16 re-export

The FP16 GPU re-export has not been run yet. These commands finish step (a) on a CUDA GPU:

```bash
python -B scripts/export_dets.py --data dsec --name dsec_v8n_416s3x_s0 --weights <ref>/runs/dsec_v8n_416s3x_s0/last.pth \
    --seqs thun_01_b zurich_city_14_b
python -B scripts/diff_outputs.py cache outputs/cache/dsec_v8n_416s3x_s0/dsec <ref>/cache/dsec_v8n_416s3x_s0/dsec
# likewise for dsec_v8n_416_s0, mot_v8n_416s3x_s0 (--seqs MOT17-09-FRCNN) and dsec_y11n_416s3x_s0 (--seqs zurich_city_13_b)
```

Expected result: equality up to FP16 noise on the same GPU model.
