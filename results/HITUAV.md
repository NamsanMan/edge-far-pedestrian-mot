# HIT-UAV results

Person detection on HIT-UAV: aerial thermal-infrared drone images, 640x512 (Suo et al., Scientific Data 2023). pedestrian = the HIT-UAV class Person; car = Car. Detection only: the release holds single images, so there is no tracking evaluation. Every number on this page is regenerated from the per-run files in `results/hituav/` (no dataset needed):

```
python -B scripts/evaluate_hituav.py --table
```

Rules (`efpmot/data/hituav.py`, decisions 1-7; evaluator `efpmot/eval/detection.py`):
- Training: COCO-pretrained YOLOv8n arms, the efpmot recipe (batch 16, AdamW, FP16) and the ~9,060-iteration budget of the other datasets: official 72 epochs, recA 116, recB 118, altLH 112. The model after the full schedule is evaluated.
- Detections: FP16 export of every val and test image, conf 0.001, class-wise NMS 0.65, at most 300 per image.
- AP50, AP50:95: COCO 101-point interpolated AP over all detections; DontCare boxes are ignore regions.
- Operating point: the score threshold with the best person F1 at IoU 0.5 on the split's val part. F1 / P / R and recall are measured on test at that threshold. Recall < 8 px: GT persons whose native short side is below 8 px.
- Values: mean over seeds. recA + recB: the mean of the two fold means, so every recording is tested exactly once.
- GFLOPs: `efpmot.models.flops` at the canvas, 8 classes.

## Recording-disjoint folds (recA + recB): unseen recordings

Recordings (one camera angle of one flight) are split into two folds that balance person boxes per altitude and day / night cell; each fold's model is tested on the recordings of the other half. Every test recording shares its flight with training recordings: the folds test unseen recordings, not unseen sites.

| arm | GFLOPs | seeds recA / recB | person AP50 | vs plain 416x320 | person AP50:95 | person F1 / P / R | person recall < 8 px | car AP50 | person AP50 per seed: recA; recB |
|---|---|---|---|---|---|---|---|---|---|
| plain 416x320 | 2.665 | 3 / 3 | **55.8** |  | 19.6 | 63.3 / 70.1 / 58.7 | 23.6 | 78.1 | 52.9 53.8 51.8; 57.5 59.5 59.6 |
| S3x 416x320 | 2.955 | 3 / 3 | **62.9** | +7.1 | 23.7 | 68.8 / 74.2 / 64.5 | 35.8 | 77.0 | 59.3 61.9 60.6; 65.7 65.8 64.4 |
| S3y 416x320 | 2.955 | 2 / 2 | **59.9** | +4.1 | 22.1 | 64.6 / 69.7 / 62.1 | 27.6 | 77.3 | 56.4 56.9; 63.3 63.0 |
| P2 416x320 (code not included) | - | 2 / 2 | **63.1** | +7.2 | 23.6 | 68.0 / 74.1 / 63.4 | 33.0 | 76.6 | 59.7 61.3; 65.9 65.3 |
| uniform 448x352 | 3.157 | 2 / 2 | **60.2** | +4.4 | 21.9 | 66.7 / 74.2 / 60.9 | 23.4 | 77.9 | 57.6 58.4; 62.3 62.6 |
| plain 640x512 | 6.559 | 2 / 2 | **68.0** | +12.2 | 27.6 | 71.6 / 76.9 / 67.9 | 39.3 | 81.2 | 64.4 67.9; 70.5 69.4 |

## Altitude shift (altLH): train at 60-90 m, test at 100-130 m

Training and val use the recordings flown at 60-90 m, test every recording at 100-130 m. No flight has recordings on both sides, so the test flights are unseen and the persons are smaller.

| arm | GFLOPs | seeds | person AP50 | vs plain 416x320 | person AP50:95 | person F1 / P / R | person recall < 8 px | car AP50 | person AP50 per seed |
|---|---|---|---|---|---|---|---|---|---|
| plain 416x320 | 2.665 | 3 | **28.5** |  | 9.1 | 40.2 / 34.4 / 48.7 | 13.5 | 56.5 | 27.2 28.7 29.7 |
| S3x 416x320 | 2.955 | 3 | **34.1** | +5.6 | 11.4 | 46.1 / 41.1 / 52.8 | 16.3 | 55.8 | 35.1 33.5 33.9 |
| uniform 448x352 | 3.157 | 2 | **28.8** | +0.3 | 9.2 | 43.4 / 39.3 / 48.4 | 12.3 | 57.6 | 27.5 30.1 |

## Official split (inflated by recording overlap)

The official train / val / test split interleaves frames of the same recordings: for 96 % of the test images a training image of the same recording lies within 10 serial numbers. These numbers measure recordings the model has seen; they are listed for comparison with work that reports the official split, not as a generalisation estimate.

| arm | GFLOPs | seeds | person AP50 | vs plain 416x320 | person AP50:95 | person F1 / P / R | person recall < 8 px | car AP50 | person AP50 per seed |
|---|---|---|---|---|---|---|---|---|---|
| plain 416x320 | 2.665 | 3 | **76.0** |  | 32.3 | 77.2 / 79.4 / 75.1 | 44.8 | 92.5 | 75.6 76.4 75.9 |
| S3x 416x320 | 2.955 | 3 | **82.8** | +6.8 | 38.3 | 83.4 / 86.4 / 80.5 | 61.9 | 92.9 | 82.5 82.7 83.4 |
| P2 416x320 (code not included) | - | 2 | **83.9** | +7.9 | 37.9 | 84.3 / 87.2 / 81.5 | 62.2 | 92.1 | 83.9 83.8 |
| uniform 448x352 | 3.157 | 3 | **78.6** | +2.6 | 34.0 | 79.5 / 82.2 / 77.0 | 44.7 | 92.6 | 78.8 77.8 79.3 |
| plain 640x512 | 6.559 | 2 | **84.5** | +8.5 | 40.2 | 85.3 / 87.5 / 83.3 | 61.8 | 93.6 | 85.0 84.1 |
| S3x 640x512 | 7.273 | 2 | **85.8** | +9.8 | 42.6 | 85.7 / 89.5 / 82.2 | 60.0 | 93.2 | 85.1 86.5 |

## Released weights

Each released model is evaluated on its fold's test part, the recordings it never saw. For a video of a HIT-UAV recording, use the model of the fold that did not train on that recording (`efpmot.data.hituav.fold_recordings` lists the recordings of each fold). The person threshold is the operating point above; it is also stored in `efpmot.release.RELEASES` (`thr`) and in `results/hituav/<run>.json` (`op.pedestrian.thr`).

| file | run | trained on | person threshold (val max F1) | test person AP50 | test person F1 / P / R |
|---|---|---|---|---|---|
| `yolov8n_plain_hituav-recA_416x320_s0.pt` | `hit_v8n_416recA_s0` | recA train (1,263 images), 116 epochs | 0.430 | 52.9 | 60.3 / 77.6 / 49.3 |
| `yolov8n_s3x_hituav-recA_416x320_s0.pt` | `hit_v8n_416s3xrecA_s0` | recA train (1,263 images), 116 epochs | 0.154 | 59.3 | 67.2 / 73.4 / 62.0 |
| `yolov8n_plain_hituav-recB_416x320_s0.pt` | `hit_v8n_416recB_s0` | recB train (1,237 images), 118 epochs | 0.057 | 57.5 | 63.8 / 61.4 / 66.3 |
| `yolov8n_s3x_hituav-recB_416x320_s0.pt` | `hit_v8n_416s3xrecB_s0` | recB train (1,237 images), 118 epochs | 0.306 | 65.7 | 70.0 / 72.4 / 67.7 |

## Notes

- P2 416x320 (code not included): YOLOv8n with an extra stride-4 P2 level from the original research code. Its model is not part of this repository; its rows are this repository's evaluation of the detection caches of those runs, under the same rules.
- Detection caches: written at the end of training by the original research code (Colab L4 GPU, torch 2.8.0+cu129, FP16).
- Runs: 51 files in `results/hituav/`.
