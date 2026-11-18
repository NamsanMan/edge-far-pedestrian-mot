# Detector comparison on DSEC

YOLOv8n + S3x 416x320 against other lightweight and real-time detectors on DSEC-Det test (13 sequences from 8 drives), with the protocol of [`RESULTS.md`](RESULTS.md): cross-fitted thresholds, paired hierarchical bootstrap (B = 10000), paired randomisation test, Holm adjustment, TOST against the frozen margin. d = HOTA(YOLOv8n + S3x) - HOTA(detector), in HOTA points. Every number on this page is regenerated from the per-run statistics in `results/stats/` and the latency tables in `deploy/jetson/results/` (no dataset needed):

```
python -B scripts/compare.py --detectors results/contrasts_detectors.yaml
```

Outputs: `results/summary_detectors/<dataset>.json` (every field of each contrast), `results/summary_detectors.csv` (one row per contrast, class and tracker) and this page. The contrasts of `results/contrasts_detectors.yaml` form their own Holm families (class x tracker, 11 contrasts each). Contrasts that `RESULTS.md` also reports have the same d (sign flipped where A and B are swapped) and unadjusted p there, but a different Holm adjustment.

## Pedestrians: cost, latency and tracking accuracy

Sorted by GFLOPs. HOTA: seed-mean pooled HOTA of the arm. d = HOTA(YOLOv8n + S3x) - HOTA(detector) with its 95 % interval; p = Holm-adjusted randomisation p. Bold: significant (p < 0.05); n.s.: not significant. code: `repo` = a model family of this repository; `results only` = its code is not part of this repository and its rows are regenerated from the shipped stats (like the P2 row of `RESULTS.md`).

| detector | canvas | code | params [M] | GFLOPs | Jetson FP16 [ms] | seeds | HOTA ByteTrack / OC-SORT | ByteTrack d [95 % CI] | p | OC-SORT d [95 % CI] | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLOX-Nano | 416x320 | results only | 0.90 | 0.83 | 3.12 | 0,1,2 | 24.47 / 24.22 | **+9.34 [+8.36, +13.83]** | 0.040 | **+10.20 [+8.76, +17.64]** | 0.035 |
| NanoDet-Plus-m | 416x320 | results only | 1.17 | 1.20 | 4.69 | 0,1,2 | 23.89 / 21.47 | +9.91 [+8.30, +16.21] n.s. | 0.107 | **+12.94 [+10.61, +20.24]** | 0.030 |
| YOLO26n | 416x320 | results only | 2.38 | 1.74 | 3.64 | 0,1 | 28.82 / 30.16 | **+4.99 [+1.88, +8.80]** | 0.040 | **+4.25 [+0.82, +9.43]** | 0.035 |
| YOLO11n | 416x320 | repo | 2.59 | 2.10 | 3.49 | 0,1 | 28.29 / 28.84 | **+5.52 [+3.93, +10.16]** | 0.040 | **+5.58 [+3.61, +10.63]** | 0.035 |
| YOLOv8n | 416x320 | repo | 3.01 | 2.66 | 2.94 | 0,1,2 | 27.92 / 28.93 | **+5.88 [+4.95, +9.91]** | 0.040 | **+5.49 [+4.37, +11.43]** | 0.035 |
| **YOLOv8n + S3x** | 416x320 | repo | 2.92 | 2.95 | 3.27 | 0,1 | 33.80 / 34.42 | - |  | - |  |
| YOLOv8n | 448x352 | repo | 3.01 | 3.16 | 3.13 (a) | 0,1 | 31.56 / 31.61 | +2.24 [+1.19, +8.09] n.s. | 0.060 | **+2.81 [+1.53, +9.38]** | 0.035 |
| YOLOv8n + P2 | 416x320 | results only | 3.18 | 4.93 | 4.51 (a) | 0,1 | 32.17 / 31.39 | +1.63 [-1.08, +5.65] n.s. | 0.756 | +3.03 [-0.71, +7.78] n.s. | 0.286 |
| YOLOX-Tiny | 416x320 | results only | 5.06 | 4.96 | 3.90 | 0,1,2 | 33.61 / 33.58 | +0.19 [-2.23, +2.97] n.s. | 1.000 | +0.84 [-2.98, +4.32] n.s. | 0.453 |
| YOLOv8n | 640x480 | repo | 3.01 | 6.15 | 4.72 (a) | 0,1,2 | 39.05 / 39.52 | -5.25 [-8.65, -4.22] n.s. | 0.107 | -5.10 [-10.77, -3.62] n.s. | 0.093 |
| YOLOv8s | 416x320 | results only | 11.14 | 9.31 | 4.79 | 0,1,2 | 34.15 / 34.03 | -0.35 [-1.32, +5.33] n.s. | 1.000 | +0.39 [-0.87, +6.64] n.s. | 0.453 |
| RT-DETR-L | 416x320 | results only | 32.82 | 39.03 | 26.70 | 0,1,2 | 39.98 / 40.27 | **-6.17 [-10.19, -5.32]** | 0.041 | **-5.86 [-11.61, -4.54]** | 0.030 |

(a) Timed in the first Jetson session (2026-10-27, `deploy/jetson/results/latency_orin_nano.csv`); the other latencies come from the detector session (2026-11-13, `deploy/jetson/results/latency_orin_nano_detectors.csv`).

- At most 10 % more FLOPs than YOLOv8n + S3x (YOLOX-Nano, NanoDet-Plus-m, YOLO26n, YOLO11n, YOLOv8n, YOLOv8n 448x352): d from +2.24 to +12.94 over both trackers; not significant after Holm: NanoDet-Plus-m ByteTrack (+9.91, p 0.107), YOLOv8n 448x352 ByteTrack (+2.24, p 0.060).
- More expensive detectors (FLOPs and Jetson latency relative to YOLOv8n + S3x; d ByteTrack / OC-SORT, bold: significant after Holm): YOLOv8n + P2 1.7x / 1.4x: +1.63 / +3.03; YOLOX-Tiny 1.7x / 1.2x: +0.19 / +0.84; YOLOv8n 640x480 2.1x / 1.4x: -5.25 / -5.10; YOLOv8s 3.2x / 1.5x: -0.35 / +0.39; RT-DETR-L 13.2x / 8.2x: **-6.17** / **-5.86**.

## All contrasts

Equivalence margin 1.1955 HOTA. Holm families: class x tracker over the 11 contrasts of this page.

| A | B | class | tracker | seeds A / B | HOTA A | HOTA B | d | 95 % CI | p_AR | Holm p_AR | TOST p (Holm) | label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv8n + S3x 416x320 | YOLOX-Nano 416x320 | car | ByteTrack | 0,1 / 0,1,2 | 55.81 | 49.63 | **+6.18** | [+4.53, +7.48] | <0.001 | 0.003 | 1.000 | win |
| YOLOv8n + S3x 416x320 | YOLOX-Nano 416x320 | car | OC-SORT | 0,1 / 0,1,2 | 58.09 | 49.29 | **+8.80** | [+6.83, +9.97] | <0.001 | 0.003 | 1.000 | win |
| YOLOv8n + S3x 416x320 | YOLOX-Nano 416x320 | pedestrian | ByteTrack | 0,1 / 0,1,2 | 33.80 | 24.47 | **+9.34** | [+8.36, +13.83] | 0.004 | 0.040 | 1.000 | win |
| YOLOv8n + S3x 416x320 | YOLOX-Nano 416x320 | pedestrian | OC-SORT | 0,1 / 0,1,2 | 34.42 | 24.22 | **+10.20** | [+8.76, +17.64] | 0.004 | 0.035 | 1.000 | win |
| YOLOv8n + S3x 416x320 | NanoDet-Plus-m 416x320 | car | ByteTrack | 0,1 / 0,1,2 | 55.81 | 47.94 | **+7.87** | [+5.99, +9.78] | <0.001 | 0.003 | 1.000 | win |
| YOLOv8n + S3x 416x320 | NanoDet-Plus-m 416x320 | car | OC-SORT | 0,1 / 0,1,2 | 58.09 | 45.69 | **+12.40** | [+9.32, +14.62] | <0.001 | 0.003 | 1.000 | win |
| YOLOv8n + S3x 416x320 | NanoDet-Plus-m 416x320 | pedestrian | ByteTrack | 0,1 / 0,1,2 | 33.80 | 23.89 | **+9.91** | [+8.30, +16.21] | 0.021 | 0.107 | 1.000 | inconclusive |
| YOLOv8n + S3x 416x320 | NanoDet-Plus-m 416x320 | pedestrian | OC-SORT | 0,1 / 0,1,2 | 34.42 | 21.47 | **+12.94** | [+10.61, +20.24] | 0.003 | 0.030 | 1.000 | win |
| YOLOv8n + S3x 416x320 | YOLO26n 416x320 | car | ByteTrack | 0,1 / 0,1 | 55.81 | 53.76 | **+2.05** | [+0.08, +3.60] | 0.015 | 0.092 | 1.000 | inconclusive |
| YOLOv8n + S3x 416x320 | YOLO26n 416x320 | car | OC-SORT | 0,1 / 0,1 | 58.09 | 54.76 | **+3.32** | [+1.21, +5.19] | 0.020 | 0.119 | 1.000 | inconclusive |
| YOLOv8n + S3x 416x320 | YOLO26n 416x320 | pedestrian | ByteTrack | 0,1 / 0,1 | 33.80 | 28.82 | **+4.99** | [+1.88, +8.80] | 0.004 | 0.040 | 1.000 | win |
| YOLOv8n + S3x 416x320 | YOLO26n 416x320 | pedestrian | OC-SORT | 0,1 / 0,1 | 34.42 | 30.16 | **+4.25** | [+0.82, +9.43] | 0.004 | 0.035 | 1.000 | win |
| YOLOv8n + S3x 416x320 | YOLO11n 416x320 | car | ByteTrack | 0,1 / 0,1 | 55.81 | 53.80 | **+2.01** | [-0.02, +3.56] | 0.068 | 0.342 | 1.000 | inconclusive |
| YOLOv8n + S3x 416x320 | YOLO11n 416x320 | car | OC-SORT | 0,1 / 0,1 | 58.09 | 56.02 | **+2.07** | [+0.47, +3.03] | 0.047 | 0.233 | 1.000 | inconclusive |
| YOLOv8n + S3x 416x320 | YOLO11n 416x320 | pedestrian | ByteTrack | 0,1 / 0,1 | 33.80 | 28.29 | **+5.52** | [+3.93, +10.16] | 0.004 | 0.040 | 1.000 | win |
| YOLOv8n + S3x 416x320 | YOLO11n 416x320 | pedestrian | OC-SORT | 0,1 / 0,1 | 34.42 | 28.84 | **+5.58** | [+3.61, +10.63] | 0.004 | 0.035 | 1.000 | win |
| YOLOv8n + S3x 416x320 | YOLOv8n 416x320 | car | ByteTrack | 0,1 / 0,1,2 | 55.81 | 52.97 | **+2.84** | [+0.57, +4.67] | <0.001 | 0.008 | 1.000 | win |
| YOLOv8n + S3x 416x320 | YOLOv8n 416x320 | car | OC-SORT | 0,1 / 0,1,2 | 58.09 | 55.91 | **+2.18** | [+1.15, +3.43] | 0.008 | 0.061 | 1.000 | inconclusive |
| YOLOv8n + S3x 416x320 | YOLOv8n 416x320 | pedestrian | ByteTrack | 0,1 / 0,1,2 | 33.80 | 27.92 | **+5.88** | [+4.95, +9.91] | 0.004 | 0.040 | 1.000 | win |
| YOLOv8n + S3x 416x320 | YOLOv8n 416x320 | pedestrian | OC-SORT | 0,1 / 0,1,2 | 34.42 | 28.93 | **+5.49** | [+4.37, +11.43] | 0.004 | 0.035 | 1.000 | win |
| YOLOv8n + S3x 416x320 | YOLOv8n 448x352 | car | ByteTrack | 0,1 / 0,1 | 55.81 | 55.29 | **+0.52** | [-1.38, +1.97] | 0.419 | 1.000 | 1.000 | inconclusive |
| YOLOv8n + S3x 416x320 | YOLOv8n 448x352 | car | OC-SORT | 0,1 / 0,1 | 58.09 | 56.32 | **+1.77** | [-0.03, +2.89] | 0.104 | 0.261 | 1.000 | inconclusive |
| YOLOv8n + S3x 416x320 | YOLOv8n 448x352 | pedestrian | ByteTrack | 0,1 / 0,1 | 33.80 | 31.56 | **+2.24** | [+1.19, +8.09] | 0.010 | 0.060 | 1.000 | inconclusive |
| YOLOv8n + S3x 416x320 | YOLOv8n 448x352 | pedestrian | OC-SORT | 0,1 / 0,1 | 34.42 | 31.61 | **+2.81** | [+1.53, +9.38] | 0.005 | 0.035 | 1.000 | win |
| YOLOv8n + S3x 416x320 | YOLOv8n + P2 416x320 | car | ByteTrack | 0,1 / 0,1 | 55.81 | 54.39 | **+1.42** | [-0.85, +3.66] | 0.394 | 1.000 | 1.000 | inconclusive |
| YOLOv8n + S3x 416x320 | YOLOv8n + P2 416x320 | car | OC-SORT | 0,1 / 0,1 | 58.09 | 56.53 | **+1.56** | [-0.28, +3.99] | 0.087 | 0.261 | 1.000 | inconclusive |
| YOLOv8n + S3x 416x320 | YOLOv8n + P2 416x320 | pedestrian | ByteTrack | 0,1 / 0,1 | 33.80 | 32.17 | **+1.63** | [-1.08, +5.65] | 0.252 | 0.756 | 1.000 | inconclusive |
| YOLOv8n + S3x 416x320 | YOLOv8n + P2 416x320 | pedestrian | OC-SORT | 0,1 / 0,1 | 34.42 | 31.39 | **+3.03** | [-0.71, +7.78] | 0.095 | 0.286 | 1.000 | inconclusive |
| YOLOv8n + S3x 416x320 | YOLOX-Tiny 416x320 | car | ByteTrack | 0,1 / 0,1,2 | 55.81 | 54.71 | **+1.11** | [-1.42, +3.17] | 0.694 | 1.000 | 1.000 | inconclusive |
| YOLOv8n + S3x 416x320 | YOLOX-Tiny 416x320 | car | OC-SORT | 0,1 / 0,1,2 | 58.09 | 55.96 | **+2.13** | [-0.27, +3.74] | 0.065 | 0.261 | 1.000 | inconclusive |
| YOLOv8n + S3x 416x320 | YOLOX-Tiny 416x320 | pedestrian | ByteTrack | 0,1 / 0,1,2 | 33.80 | 33.61 | **+0.19** | [-2.23, +2.97] | 0.606 | 1.000 | 1.000 | inconclusive |
| YOLOv8n + S3x 416x320 | YOLOX-Tiny 416x320 | pedestrian | OC-SORT | 0,1 / 0,1,2 | 34.42 | 33.58 | **+0.84** | [-2.98, +4.32] | 0.389 | 0.453 | 1.000 | inconclusive |
| YOLOv8n + S3x 416x320 | YOLOv8n 640x480 | car | ByteTrack | 0,1 / 0,1,2 | 55.81 | 59.82 | **-4.00** | [-7.82, -1.42] | <0.001 | 0.003 | 1.000 | loss |
| YOLOv8n + S3x 416x320 | YOLOv8n 640x480 | car | OC-SORT | 0,1 / 0,1,2 | 58.09 | 62.47 | **-4.38** | [-7.00, -2.71] | <0.001 | 0.003 | 1.000 | loss |
| YOLOv8n + S3x 416x320 | YOLOv8n 640x480 | pedestrian | ByteTrack | 0,1 / 0,1,2 | 33.80 | 39.05 | **-5.25** | [-8.65, -4.22] | 0.022 | 0.107 | 1.000 | inconclusive |
| YOLOv8n + S3x 416x320 | YOLOv8n 640x480 | pedestrian | OC-SORT | 0,1 / 0,1,2 | 34.42 | 39.52 | **-5.10** | [-10.77, -3.62] | 0.023 | 0.093 | 1.000 | inconclusive |
| YOLOv8n + S3x 416x320 | YOLOv8s 416x320 | car | ByteTrack | 0,1 / 0,1,2 | 55.81 | 56.20 | **-0.39** | [-4.33, +2.20] | 0.600 | 1.000 | 1.000 | inconclusive |
| YOLOv8n + S3x 416x320 | YOLOv8s 416x320 | car | OC-SORT | 0,1 / 0,1,2 | 58.09 | 58.97 | **-0.88** | [-3.07, +0.45] | 0.184 | 0.261 | 1.000 | inconclusive |
| YOLOv8n + S3x 416x320 | YOLOv8s 416x320 | pedestrian | ByteTrack | 0,1 / 0,1,2 | 33.80 | 34.15 | **-0.35** | [-1.32, +5.33] | 0.845 | 1.000 | 1.000 | inconclusive |
| YOLOv8n + S3x 416x320 | YOLOv8s 416x320 | pedestrian | OC-SORT | 0,1 / 0,1,2 | 34.42 | 34.03 | **+0.39** | [-0.87, +6.64] | 0.227 | 0.453 | 1.000 | inconclusive |
| YOLOv8n + S3x 416x320 | RT-DETR-L 416x320 | car | ByteTrack | 0,1 / 0,1,2 | 55.81 | 59.18 | **-3.37** | [-6.76, -0.88] | <0.001 | 0.008 | 1.000 | loss |
| YOLOv8n + S3x 416x320 | RT-DETR-L 416x320 | car | OC-SORT | 0,1 / 0,1,2 | 58.09 | 60.52 | **-2.43** | [-5.14, -0.15] | 0.008 | 0.061 | 1.000 | inconclusive |
| YOLOv8n + S3x 416x320 | RT-DETR-L 416x320 | pedestrian | ByteTrack | 0,1 / 0,1,2 | 33.80 | 39.98 | **-6.17** | [-10.19, -5.32] | 0.006 | 0.041 | 1.000 | loss |
| YOLOv8n + S3x 416x320 | RT-DETR-L 416x320 | pedestrian | OC-SORT | 0,1 / 0,1,2 | 34.42 | 40.27 | **-5.86** | [-11.61, -4.54] | 0.003 | 0.030 | 1.000 | loss |

Sensitivity (reported, not deciding): thresholds cross-fitted leave-one-drive-out, bootstrap and exact randomisation over the 8 drives of the 13 test sequences. With 8 drives the smallest attainable p_AR is 2/256 = 0.0078, so with 11 contrasts per Holm family no drive-level difference can reach a Holm-adjusted p below 0.05; the unadjusted p_AR is shown.

| B | class | tracker | d | 95 % CI | p_AR |
|---|---|---|---|---|---|
| YOLOX-Nano 416x320 | car | ByteTrack | +6.24 | [+4.61, +7.52] | 0.008 |
| YOLOX-Nano 416x320 | car | OC-SORT | +8.94 | [+7.10, +10.22] | 0.008 |
| YOLOX-Nano 416x320 | pedestrian | ByteTrack | +9.34 | [+8.34, +15.19] | 0.016 |
| YOLOX-Nano 416x320 | pedestrian | OC-SORT | +10.07 | [+8.57, +17.62] | 0.016 |
| NanoDet-Plus-m 416x320 | car | ByteTrack | +7.93 | [+6.06, +9.71] | 0.008 |
| NanoDet-Plus-m 416x320 | car | OC-SORT | +12.39 | [+9.41, +14.40] | 0.008 |
| NanoDet-Plus-m 416x320 | pedestrian | ByteTrack | +10.07 | [+7.17, +17.18] | 0.094 |
| NanoDet-Plus-m 416x320 | pedestrian | OC-SORT | +12.81 | [+9.16, +19.89] | 0.016 |
| YOLO26n 416x320 | car | ByteTrack | +2.11 | [+0.30, +3.62] | 0.031 |
| YOLO26n 416x320 | car | OC-SORT | +3.25 | [+1.18, +5.35] | 0.039 |
| YOLO26n 416x320 | pedestrian | ByteTrack | +4.93 | [+1.67, +14.02] | 0.016 |
| YOLO26n 416x320 | pedestrian | OC-SORT | +4.12 | [-0.70, +13.92] | 0.016 |
| YOLO11n 416x320 | car | ByteTrack | +2.07 | [+0.12, +3.56] | 0.078 |
| YOLO11n 416x320 | car | OC-SORT | +2.07 | [+0.54, +2.98] | 0.055 |
| YOLO11n 416x320 | pedestrian | ByteTrack | +5.52 | [+4.02, +14.16] | 0.016 |
| YOLO11n 416x320 | pedestrian | OC-SORT | +5.45 | [+2.58, +11.81] | 0.016 |
| YOLOv8n 416x320 | car | ByteTrack | +2.90 | [+0.80, +4.73] | 0.008 |
| YOLOv8n 416x320 | car | OC-SORT | +2.18 | [+1.24, +3.42] | 0.023 |
| YOLOv8n 416x320 | pedestrian | ByteTrack | +5.88 | [+4.98, +12.73] | 0.016 |
| YOLOv8n 416x320 | pedestrian | OC-SORT | +5.35 | [+3.73, +12.03] | 0.016 |
| YOLOv8n 448x352 | car | ByteTrack | +0.58 | [-1.19, +1.95] | 0.352 |
| YOLOv8n 448x352 | car | OC-SORT | +1.79 | [+0.19, +2.96] | 0.078 |
| YOLOv8n 448x352 | pedestrian | ByteTrack | +2.24 | [+1.25, +13.18] | 0.023 |
| YOLOv8n 448x352 | pedestrian | OC-SORT | +2.68 | [+1.21, +10.96] | 0.008 |
| YOLOv8n + P2 416x320 | car | ByteTrack | +1.48 | [-0.64, +3.85] | 0.312 |
| YOLOv8n + P2 416x320 | car | OC-SORT | +1.56 | [-0.28, +3.97] | 0.086 |
| YOLOv8n + P2 416x320 | pedestrian | ByteTrack | +1.63 | [-0.83, +10.30] | 0.258 |
| YOLOv8n + P2 416x320 | pedestrian | OC-SORT | +2.89 | [-2.00, +10.19] | 0.109 |
| YOLOX-Tiny 416x320 | car | ByteTrack | +1.20 | [-1.16, +3.21] | 0.617 |
| YOLOX-Tiny 416x320 | car | OC-SORT | +2.13 | [-0.21, +3.79] | 0.039 |
| YOLOX-Tiny 416x320 | pedestrian | ByteTrack | +0.19 | [-1.98, +6.80] | 0.703 |
| YOLOX-Tiny 416x320 | pedestrian | OC-SORT | +0.73 | [-3.93, +7.25] | 0.516 |
| YOLOv8n 640x480 | car | ByteTrack | -3.94 | [-7.68, -1.43] | 0.008 |
| YOLOv8n 640x480 | car | OC-SORT | -4.38 | [-7.05, -2.69] | 0.008 |
| YOLOv8n 640x480 | pedestrian | ByteTrack | -5.25 | [-8.52, -2.10] | 0.094 |
| YOLOv8n 640x480 | pedestrian | OC-SORT | -5.28 | [-12.01, -3.09] | 0.094 |
| YOLOv8s 416x320 | car | ByteTrack | -0.32 | [-4.26, +2.22] | 0.594 |
| YOLOv8s 416x320 | car | OC-SORT | -0.88 | [-3.01, +0.43] | 0.188 |
| YOLOv8s 416x320 | pedestrian | ByteTrack | -0.35 | [-1.18, +9.53] | 0.992 |
| YOLOv8s 416x320 | pedestrian | OC-SORT | +0.25 | [-1.54, +10.53] | 0.320 |
| RT-DETR-L 416x320 | car | ByteTrack | -3.31 | [-6.80, -0.89] | 0.008 |
| RT-DETR-L 416x320 | car | OC-SORT | -2.43 | [-5.30, -0.15] | 0.016 |
| RT-DETR-L 416x320 | pedestrian | ByteTrack | -6.13 | [-10.30, -5.21] | 0.016 |
| RT-DETR-L 416x320 | pedestrian | OC-SORT | -5.99 | [-12.62, -4.44] | 0.016 |

## Notes

- **Training.** Every arm: the DSEC-Det training split (24,168 frames), 9,060 iterations (6 epochs), from the family's public COCO checkpoint with the family's default learning rate, on the 416 x 320 letterbox canvas (the 448 x 352 and 640 x 480 arms on their own canvas); tracked with the fixed ByteTrack / OC-SORT settings over the threshold grid and scored with TrackEval like every arm of `RESULTS.md`. Seeds 1 and 2 of YOLOv8s and RT-DETR-L were trained on a cloud GPU with the same code and settings.
- **Results only.** The model code of YOLOX-Nano, NanoDet-Plus-m, YOLO26n, YOLOv8n + P2, YOLOX-Tiny, YOLOv8s and RT-DETR-L is not part of this repository: their rows are regenerated from `results/stats/`, but these detectors cannot be retrained with this code.
- **Cost.** Parameters and GFLOPs (2 x thop MACs) of the inference graph for one 3 x 320 x 416 input (the 448 x 352 and 640 x 480 arms: their canvas), 8 classes. Repository models: `efpmot.models.flops`. The other families were measured the same way on their own implementations: NanoDet-Plus-m without its training-only auxiliary head, YOLO26n as the deployed one-to-one (NMS-free) graph.
- **Latency.** Jetson Orin Nano, 15 W power mode, TensorRT 10.3 FP16: GPU compute time of the engine alone (trtexec median over 10 s; no pre-processing, NMS or tracking), canvas input and the family's raw head output (YOLO26n: one-to-one head with every anchor decoded, no top-k; RT-DETR-L: its 300 decoder queries). Full tables: `deploy/jetson/results/latency_orin_nano.md`. The detector session timed YOLOv8n and YOLOv8n + S3x again at its end (2.93 and 3.29 ms); the first session measured the same graphs at 2.94 and 3.26 ms.
