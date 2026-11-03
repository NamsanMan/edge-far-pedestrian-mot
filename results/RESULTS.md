# Results

Tracking HOTA of each detector arm on held-out evaluation data, compared with the cross-fitted paired protocol (`efpmot/eval/protocol.py`). d = HOTA(A) - HOTA(B), in HOTA points. Every number on this page is regenerated from the per-run statistics in `results/stats/` (no dataset needed):

```
python -B scripts/compare.py --table results/contrasts.yaml
```

Outputs: `results/summary/<dataset>.json` (every field of each contrast), `results/summary.csv` (one row per contrast, class and tracker) and this page.

## S3x against the plain detector

Same detector, same canvas, same training schedule; only the P3 grid differs (S3x: twice as many columns, about +11 % FLOPs). d = HOTA(S3x) - HOTA(plain) with its 95 % interval; p = Holm-adjusted randomisation p. Bold: a win (p < 0.05); n.s.: not significant.

| data | detector | canvas | class | ByteTrack d [95 % CI] | p | OC-SORT d [95 % CI] | p |
|---|---|---|---|---|---|---|---|
| DSEC | YOLOv8n | 416x320 | car | **+2.84 [+0.57, +4.67]** | 0.005 | **+2.18 [+1.15, +3.43]** | 0.045 |
| DSEC | YOLOv8n | 416x320 | pedestrian | **+5.88 [+4.95, +9.91]** | 0.024 | **+5.49 [+4.37, +11.43]** | 0.023 |
| DSEC | YOLO11n | 416x320 | car | **+2.13 [+1.11, +3.23]** | 0.006 | +0.19 [-2.37, +2.17] n.s. | 0.716 |
| DSEC | YOLO11n | 416x320 | pedestrian | **+6.24 [+4.99, +8.08]** | 0.024 | **+4.84 [+3.22, +7.75]** | 0.009 |
| MOT17 | YOLOv8n | 416x256 | pedestrian | +2.70 [+0.43, +4.01] n.s. | 0.125 | **+2.12 [-0.34, +3.73]** | 0.047 |
| VisDrone | YOLOv8n | 640x384 | pedestrian | **+2.58 [+0.24, +5.38]** | 0.011 | **+3.05 [+0.48, +5.86]** | 0.015 |
| VisDrone | YOLOv8n | 640x384 | car | +0.23 [-0.81, +1.37] n.s. | 0.693 | +0.98 [-0.22, +2.13] n.s. | 0.228 |
| BDD100K | YOLOv8n | 416x256 | car | **+1.03 [+0.70, +1.35]** | <0.001 | **+0.98 [+0.70, +1.28]** | <0.001 |
| BDD100K | YOLOv8n | 416x256 | pedestrian | **+2.52 [+1.83, +3.33]** | <0.001 | **+2.56 [+1.90, +3.40]** | <0.001 |
| BDD100K | YOLO11n | 416x256 | car | **+0.89 [+0.54, +1.25]** | <0.001 | **+0.92 [+0.54, +1.26]** | <0.001 |
| BDD100K | YOLO11n | 416x256 | pedestrian | **+1.65 [+0.83, +2.50]** | <0.001 | **+1.89 [+1.14, +2.69]** | <0.001 |

## S3x against a uniform canvas of the same cost

S3x adds about 11 % FLOPs. The fair question is whether spending them on a slightly larger uniform canvas does as well. Each compute-matched canvas is a uniform letterbox canvas whose FLOPs are within a few per cent of those of S3x: 0 to 7 % more, VisDrone 0.8 % less ("matched" means matched on FLOPs, not on latency). Same detector, schedule and number of seeds.

| data | detector | S3x canvas | uniform canvas | GFLOPs S3x / uniform | class | ByteTrack d [95 % CI] | p | OC-SORT d [95 % CI] | p |
|---|---|---|---|---|---|---|---|---|---|
| DSEC | YOLOv8n | 416x320 | 448x352 | 2.955 / 3.157 | car | +0.52 [-1.38, +1.97] n.s. | 1.000 | +1.77 [-0.03, +2.89] n.s. | 0.436 |
| DSEC | YOLOv8n | 416x320 | 448x352 | 2.955 / 3.157 | pedestrian | **+2.24 [+1.19, +8.09]** | 0.040 | **+2.81 [+1.53, +9.38]** | 0.023 |
| MOT17 | YOLOv8n | 416x256 | 480x256 | 2.364 / 2.460 | pedestrian | -0.51 [-1.84, +0.43] n.s. | 0.141 | +0.77 [-0.83, +2.08] n.s. | 0.375 |
| VisDrone | YOLOv8n | 640x384 | 704x384 | 5.455 / 5.411 | pedestrian | +0.07 [-1.63, +1.39] n.s. | 0.764 | +0.22 [-1.76, +1.71] n.s. | 0.584 |
| BDD100K | YOLOv8n | 416x256 | 480x256 | 2.364 / 2.460 | car | +0.21 [-0.13, +0.55] n.s. | 0.174 | -0.05 [-0.31, +0.23] n.s. | 0.683 |
| BDD100K | YOLOv8n | 416x256 | 480x256 | 2.364 / 2.460 | pedestrian | **+1.54 [+0.79, +2.38]** | <0.001 | **+1.37 [+0.66, +2.18]** | <0.001 |
| BDD100K | YOLO11n | 416x256 | 480x256 | 1.865 / 1.939 | car | -0.14 [-0.39, +0.14] n.s. | 0.174 | -0.17 [-0.56, +0.18] n.s. | 0.437 |
| BDD100K | YOLO11n | 416x256 | 480x256 | 1.865 / 1.939 | pedestrian | **+0.64 [-0.06, +1.33]** | 0.008 | +0.57 [-0.31, +1.42] n.s. | 0.126 |

- Pedestrians: S3x beats the same-cost canvas on DSEC and BDD100K (YOLO11n on BDD100K: ByteTrack only); MOT17 and VisDrone are ties. Cars tie everywhere.
- Latency is not FLOPs: on the Jetson Orin Nano the 448x352 uniform engine takes 3.13 ms and the S3x engine 3.26 ms (TensorRT FP16, table at the end of this page).
- A 640-wide uniform canvas (about 2.1x the FLOPs of S3x) still beats S3x (plain 640 minus S3x, ByteTrack / OC-SORT): DSEC car +4.00 / +4.38; DSEC pedestrian +5.25 / +5.10; BDD100K car +3.18 / +3.63; BDD100K pedestrian +3.06 / +3.38.

## Pre-registered predictions (MOT17, VisDrone)

Written before any training on these datasets (`docs/PREREGISTRATION.md`). Reading rules: P1 / P2 hold iff d > 0 and the unadjusted randomisation p < 0.05 for ByteTrack and OC-SORT; P3 / P4 hold iff non-inferior (ByteTrack 90 % bootstrap interval lower bound above -1.1955), fail iff d < 0 with p < 0.05, else inconclusive. For MOT17 a P1 that does not hold is "not shown", not "refuted", unless the 95 % interval excludes +0.5.

| data | id | contrast | class | ByteTrack d [90 % CI], p_AR | OC-SORT d [90 % CI], p_AR | verdict |
|---|---|---|---|---|---|---|
| mot17 | P1 | S3x 416x256 vs plain 416x256 | pedestrian | +2.70 [+0.78, +3.83], 0.062 | +2.12 [+0.12, +3.37], 0.016 | **not shown** |
| mot17 | P2 | S3x 416x256 vs S3y 416x256 | pedestrian | +2.38 [+1.16, +4.84], 0.016 | +2.11 [+1.10, +3.34], 0.031 | **holds** |
| mot17 | P4 | S3x 416x256 vs uniform 480x256 | pedestrian | -0.51 [-1.60, +0.29], 0.141 | +0.77 [-0.62, +1.77], 0.375 | **inconclusive** |
| visdrone | P1 | S3x 640x384 vs plain 640x384 | pedestrian | +2.58 [+0.55, +4.92], 0.004 | +3.05 [+0.90, +5.42], 0.005 | **holds** |
| visdrone | P2 | S3x 640x384 vs S3y 640x384 | pedestrian | +2.70 [+1.05, +4.55], 0.011 | +2.40 [+0.88, +4.30], 0.014 | **holds** |
| visdrone | P3 | S3y 640x384 vs S3x 640x384 | car | +0.85 [+0.12, +1.61], 0.009 | +0.49 [-0.61, +1.43], 0.253 | **holds (non-inferior)** |
| visdrone | P4 | S3x 640x384 vs uniform 704x384 | pedestrian | +0.07 [-1.36, +1.22], 0.764 | +0.22 [-1.44, +1.59], 0.584 | **inconclusive** |

With 7 MOT17 sequences the exact randomisation test has 2^7 = 128 swap patterns, so its smallest attainable p is 2/128 = 0.0156.

## DSEC-Det test (13 sequences from 8 drives)

Equivalence margin 1.1955 HOTA. Holm families: class x tracker over the contrasts of this dataset.

| detector | A | B | class | tracker | seeds A / B | HOTA A | HOTA B | d | 95 % CI | p_AR | Holm p_AR | TOST p (Holm) | label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv8n | S3x 416x320 | plain 416x320 | car | ByteTrack | 0,1 / 0,1,2 | 55.81 | 52.97 | **+2.84** | [+0.57, +4.67] | <0.001 | 0.005 | 1.000 | win |
| YOLOv8n | S3x 416x320 | plain 416x320 | car | OC-SORT | 0,1 / 0,1,2 | 58.09 | 55.91 | **+2.18** | [+1.15, +3.43] | 0.008 | 0.045 | 1.000 | win |
| YOLOv8n | S3x 416x320 | plain 416x320 | pedestrian | ByteTrack | 0,1 / 0,1,2 | 33.80 | 27.92 | **+5.88** | [+4.95, +9.91] | 0.004 | 0.024 | 1.000 | win |
| YOLOv8n | S3x 416x320 | plain 416x320 | pedestrian | OC-SORT | 0,1 / 0,1,2 | 34.42 | 28.93 | **+5.49** | [+4.37, +11.43] | 0.004 | 0.023 | 1.000 | win |
| YOLOv8n | S3x 416x320 | S3y 416x320 | car | ByteTrack | 0,1 / 0,1 | 55.81 | 55.13 | **+0.68** | [-2.03, +2.69] | 0.758 | 1.000 | 1.000 | inconclusive |
| YOLOv8n | S3x 416x320 | S3y 416x320 | car | OC-SORT | 0,1 / 0,1 | 58.09 | 56.67 | **+1.42** | [-0.86, +2.79] | 0.188 | 0.436 | 1.000 | inconclusive |
| YOLOv8n | S3x 416x320 | S3y 416x320 | pedestrian | ByteTrack | 0,1 / 0,1 | 33.80 | 28.90 | **+4.90** | [+3.19, +8.65] | 0.004 | 0.024 | 1.000 | win |
| YOLOv8n | S3x 416x320 | S3y 416x320 | pedestrian | OC-SORT | 0,1 / 0,1 | 34.42 | 29.34 | **+5.08** | [+2.49, +9.42] | 0.004 | 0.023 | 1.000 | win |
| YOLOv8n | S3x 416x320 | uniform 448x352 | car | ByteTrack | 0,1 / 0,1 | 55.81 | 55.29 | **+0.52** | [-1.38, +1.97] | 0.419 | 1.000 | 1.000 | inconclusive |
| YOLOv8n | S3x 416x320 | uniform 448x352 | car | OC-SORT | 0,1 / 0,1 | 58.09 | 56.32 | **+1.77** | [-0.03, +2.89] | 0.104 | 0.436 | 1.000 | inconclusive |
| YOLOv8n | S3x 416x320 | uniform 448x352 | pedestrian | ByteTrack | 0,1 / 0,1 | 33.80 | 31.56 | **+2.24** | [+1.19, +8.09] | 0.010 | 0.040 | 1.000 | win |
| YOLOv8n | S3x 416x320 | uniform 448x352 | pedestrian | OC-SORT | 0,1 / 0,1 | 34.42 | 31.61 | **+2.81** | [+1.53, +9.38] | 0.005 | 0.023 | 1.000 | win |
| YOLOv8n | S3y 416x320 | plain 416x320 | car | ByteTrack | 0,1 / 0,1,2 | 55.13 | 52.97 | **+2.15** | [+0.75, +3.66] | <0.001 | 0.002 | 1.000 | win |
| YOLOv8n | S3y 416x320 | plain 416x320 | car | OC-SORT | 0,1 / 0,1,2 | 56.67 | 55.91 | **+0.76** | [-0.44, +3.17] | 0.105 | 0.436 | 1.000 | inconclusive |
| YOLOv8n | S3y 416x320 | plain 416x320 | pedestrian | ByteTrack | 0,1 / 0,1,2 | 28.90 | 27.92 | **+0.98** | [+0.03, +3.13] | 0.040 | 0.080 | 1.000 | inconclusive |
| YOLOv8n | S3y 416x320 | plain 416x320 | pedestrian | OC-SORT | 0,1 / 0,1,2 | 29.34 | 28.93 | **+0.41** | [-0.17, +3.74] | 0.174 | 0.190 | 1.000 | inconclusive |
| YOLOv8n | plain 640x480 | S3x 416x320 | car | ByteTrack | 0,1,2 / 0,1 | 59.82 | 55.81 | **+4.00** | [+1.40, +7.67] | <0.001 | 0.002 | 1.000 | win |
| YOLOv8n | plain 640x480 | S3x 416x320 | car | OC-SORT | 0,1,2 / 0,1 | 62.47 | 58.09 | **+4.38** | [+2.66, +6.97] | <0.001 | 0.002 | 1.000 | win |
| YOLOv8n | plain 640x480 | S3x 416x320 | pedestrian | ByteTrack | 0,1,2 / 0,1 | 39.05 | 33.80 | **+5.25** | [+4.21, +8.60] | 0.022 | 0.066 | 1.000 | inconclusive |
| YOLOv8n | plain 640x480 | S3x 416x320 | pedestrian | OC-SORT | 0,1,2 / 0,1 | 39.52 | 34.42 | **+5.10** | [+3.63, +10.63] | 0.023 | 0.070 | 1.000 | inconclusive |
| YOLOv8n | S3x 416x320 | P2 416x320 | car | ByteTrack | 0,1 / 0,1 | 55.81 | 54.39 | **+1.42** | [-0.85, +3.66] | 0.394 | 1.000 | 1.000 | inconclusive |
| YOLOv8n | S3x 416x320 | P2 416x320 | car | OC-SORT | 0,1 / 0,1 | 58.09 | 56.53 | **+1.56** | [-0.28, +3.99] | 0.087 | 0.436 | 1.000 | inconclusive |
| YOLOv8n | S3x 416x320 | P2 416x320 | pedestrian | ByteTrack | 0,1 / 0,1 | 33.80 | 32.17 | **+1.63** | [-1.08, +5.65] | 0.252 | 0.252 | 1.000 | inconclusive |
| YOLOv8n | S3x 416x320 | P2 416x320 | pedestrian | OC-SORT | 0,1 / 0,1 | 34.42 | 31.39 | **+3.03** | [-0.71, +7.78] | 0.095 | 0.190 | 1.000 | inconclusive |
| YOLO11n | S3x 416x320 | plain 416x320 | car | ByteTrack | 0,1 / 0,1 | 55.93 | 53.80 | **+2.13** | [+1.11, +3.23] | 0.001 | 0.006 | 1.000 | win |
| YOLO11n | S3x 416x320 | plain 416x320 | car | OC-SORT | 0,1 / 0,1 | 56.21 | 56.02 | **+0.19** | [-2.37, +2.17] | 0.716 | 0.716 | 1.000 | inconclusive |
| YOLO11n | S3x 416x320 | plain 416x320 | pedestrian | ByteTrack | 0,1 / 0,1 | 34.52 | 28.29 | **+6.24** | [+4.99, +8.08] | 0.003 | 0.024 | 1.000 | win |
| YOLO11n | S3x 416x320 | plain 416x320 | pedestrian | OC-SORT | 0,1 / 0,1 | 33.68 | 28.84 | **+4.84** | [+3.22, +7.75] | 0.001 | 0.009 | 1.000 | win |

Sensitivity (reported, not deciding): the 13 test sequences come from 8 drives (for example zurich_city_14_a, _b and _c are one drive). Thresholds cross-fitted leave-one-drive-out, bootstrap and exact randomisation over the 8 drives. With 8 drives the smallest attainable p_AR is 2/256 = 0.0078, so with 7 contrasts per Holm family no drive-level difference can reach a Holm-adjusted p below 0.05; the unadjusted p_AR is shown.

| detector | A | B | class | tracker | d | 95 % CI | p_AR |
|---|---|---|---|---|---|---|---|
| YOLOv8n | S3x 416x320 | plain 416x320 | car | ByteTrack | +2.90 | [+0.80, +4.73] | 0.008 |
| YOLOv8n | S3x 416x320 | plain 416x320 | car | OC-SORT | +2.18 | [+1.24, +3.42] | 0.023 |
| YOLOv8n | S3x 416x320 | plain 416x320 | pedestrian | ByteTrack | +5.88 | [+4.98, +12.73] | 0.016 |
| YOLOv8n | S3x 416x320 | plain 416x320 | pedestrian | OC-SORT | +5.35 | [+3.73, +12.03] | 0.016 |
| YOLOv8n | S3x 416x320 | S3y 416x320 | car | ByteTrack | +0.74 | [-1.81, +2.79] | 0.734 |
| YOLOv8n | S3x 416x320 | S3y 416x320 | car | OC-SORT | +1.42 | [-0.72, +2.91] | 0.172 |
| YOLOv8n | S3x 416x320 | S3y 416x320 | pedestrian | ByteTrack | +4.90 | [+3.28, +12.22] | 0.016 |
| YOLOv8n | S3x 416x320 | S3y 416x320 | pedestrian | OC-SORT | +4.94 | [+1.17, +10.90] | 0.016 |
| YOLOv8n | S3x 416x320 | uniform 448x352 | car | ByteTrack | +0.58 | [-1.19, +1.95] | 0.352 |
| YOLOv8n | S3x 416x320 | uniform 448x352 | car | OC-SORT | +1.79 | [+0.19, +2.96] | 0.078 |
| YOLOv8n | S3x 416x320 | uniform 448x352 | pedestrian | ByteTrack | +2.24 | [+1.25, +13.18] | 0.023 |
| YOLOv8n | S3x 416x320 | uniform 448x352 | pedestrian | OC-SORT | +2.68 | [+1.21, +10.96] | 0.008 |
| YOLOv8n | S3y 416x320 | plain 416x320 | car | ByteTrack | +2.15 | [+0.77, +3.60] | 0.008 |
| YOLOv8n | S3y 416x320 | plain 416x320 | car | OC-SORT | +0.76 | [-0.54, +3.09] | 0.180 |
| YOLOv8n | S3y 416x320 | plain 416x320 | pedestrian | ByteTrack | +0.98 | [-0.74, +3.11] | 0.156 |
| YOLOv8n | S3y 416x320 | plain 416x320 | pedestrian | OC-SORT | +0.41 | [-1.23, +3.52] | 0.688 |
| YOLOv8n | plain 640x480 | S3x 416x320 | car | ByteTrack | +3.94 | [+1.41, +7.64] | 0.008 |
| YOLOv8n | plain 640x480 | S3x 416x320 | car | OC-SORT | +4.38 | [+2.69, +7.04] | 0.008 |
| YOLOv8n | plain 640x480 | S3x 416x320 | pedestrian | ByteTrack | +5.25 | [+2.09, +8.45] | 0.094 |
| YOLOv8n | plain 640x480 | S3x 416x320 | pedestrian | OC-SORT | +5.28 | [+3.09, +12.08] | 0.094 |
| YOLOv8n | S3x 416x320 | P2 416x320 | car | ByteTrack | +1.48 | [-0.64, +3.85] | 0.312 |
| YOLOv8n | S3x 416x320 | P2 416x320 | car | OC-SORT | +1.56 | [-0.28, +3.97] | 0.086 |
| YOLOv8n | S3x 416x320 | P2 416x320 | pedestrian | ByteTrack | +1.63 | [-0.83, +10.30] | 0.258 |
| YOLOv8n | S3x 416x320 | P2 416x320 | pedestrian | OC-SORT | +2.89 | [-2.00, +10.19] | 0.109 |
| YOLO11n | S3x 416x320 | plain 416x320 | car | ByteTrack | +2.13 | [+1.19, +3.23] | 0.008 |
| YOLO11n | S3x 416x320 | plain 416x320 | car | OC-SORT | +0.23 | [-2.37, +2.08] | 0.672 |
| YOLO11n | S3x 416x320 | plain 416x320 | pedestrian | ByteTrack | +6.24 | [+5.20, +8.63] | 0.016 |
| YOLO11n | S3x 416x320 | plain 416x320 | pedestrian | OC-SORT | +4.84 | [+2.51, +7.53] | 0.008 |

`P2 416x320` is YOLOv8n with an extra stride-4 (P2) detection level, the common small-object baseline, trained with the same schedule; it is reported for reference and its model code is not part of this repository.

## MOT17-train evaluation halves (7 sequences)

Equivalence margin 1.1955 HOTA. Holm families: class x tracker over the contrasts of this dataset.

| detector | A | B | class | tracker | seeds A / B | HOTA A | HOTA B | d | 95 % CI | p_AR | Holm p_AR | TOST p (Holm) | label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv8n | S3x 416x256 | plain 416x256 | pedestrian | ByteTrack | 0,1 / 0,1 | 43.44 | 40.74 | **+2.70** | [+0.43, +4.01] | 0.062 | 0.125 | 1.000 | inconclusive |
| YOLOv8n | S3x 416x256 | plain 416x256 | pedestrian | OC-SORT | 0,1 / 0,1 | 44.20 | 42.08 | **+2.12** | [-0.34, +3.73] | 0.016 | 0.047 | 1.000 | win |
| YOLOv8n | S3x 416x256 | S3y 416x256 | pedestrian | ByteTrack | 0,1 / 0,1 | 43.44 | 41.07 | **+2.38** | [+1.06, +5.54] | 0.016 | 0.047 | 1.000 | win |
| YOLOv8n | S3x 416x256 | S3y 416x256 | pedestrian | OC-SORT | 0,1 / 0,1 | 44.20 | 42.09 | **+2.11** | [+0.78, +3.65] | 0.031 | 0.062 | 1.000 | inconclusive |
| YOLOv8n | S3x 416x256 | uniform 480x256 | pedestrian | ByteTrack | 0,1 / 0,1 | 43.44 | 43.95 | **-0.51** | [-1.84, +0.43] | 0.141 | 0.141 | 0.417 | inconclusive |
| YOLOv8n | S3x 416x256 | uniform 480x256 | pedestrian | OC-SORT | 0,1 / 0,1 | 44.20 | 43.43 | **+0.77** | [-0.83, +2.08] | 0.375 | 0.375 | 0.701 | inconclusive |

Static-camera sequences (MOT17-02, -04, -09) against the moving-camera ones (-05, -10, -11, -13), each sequence at its cross-fitted threshold; descriptive only (3 and 4 sequences, no test).

| A | B | tracker | d static | d moving |
|---|---|---|---|---|
| S3x 416x256 | plain 416x256 | ByteTrack | +3.08 | +1.99 |
| S3x 416x256 | plain 416x256 | OC-SORT | +2.52 | +1.16 |
| S3x 416x256 | S3y 416x256 | ByteTrack | +2.22 | +3.14 |
| S3x 416x256 | S3y 416x256 | OC-SORT | +2.56 | +0.89 |
| S3x 416x256 | uniform 480x256 | ByteTrack | -0.18 | -1.30 |
| S3x 416x256 | uniform 480x256 | OC-SORT | +1.14 | -0.25 |

## VisDrone2019-MOT test-dev (17 sequences)

Equivalence margin 1.1955 HOTA. Holm families: class x tracker over the contrasts of this dataset.

| detector | A | B | class | tracker | seeds A / B | HOTA A | HOTA B | d | 95 % CI | p_AR | Holm p_AR | TOST p (Holm) | label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv8n | S3x 640x384 | plain 640x384 | pedestrian | ByteTrack | 0,1 / 0,1 | 22.00 | 19.41 | **+2.58** | [+0.24, +5.38] | 0.004 | 0.011 | 1.000 | win |
| YOLOv8n | S3x 640x384 | plain 640x384 | pedestrian | OC-SORT | 0,1 / 0,1 | 22.96 | 19.91 | **+3.05** | [+0.48, +5.86] | 0.005 | 0.015 | 1.000 | win |
| YOLOv8n | S3x 640x384 | S3y 640x384 | pedestrian | ByteTrack | 0,1 / 0,1 | 22.00 | 19.30 | **+2.70** | [+0.78, +4.94] | 0.011 | 0.022 | 1.000 | win |
| YOLOv8n | S3x 640x384 | S3y 640x384 | pedestrian | OC-SORT | 0,1 / 0,1 | 22.96 | 20.56 | **+2.40** | [+0.62, +4.74] | 0.014 | 0.029 | 1.000 | win |
| YOLOv8n | S3y 640x384 | S3x 640x384 | car | ByteTrack | 0,1 / 0,1 | 44.90 | 44.05 | **+0.85** | [-0.05, +1.77] | 0.009 | 0.019 | 0.209 | win |
| YOLOv8n | S3y 640x384 | S3x 640x384 | car | OC-SORT | 0,1 / 0,1 | 46.55 | 46.06 | **+0.49** | [-0.91, +1.59] | 0.253 | 0.253 | 0.231 | inconclusive |
| YOLOv8n | S3x 640x384 | uniform 704x384 | pedestrian | ByteTrack | 0,1 / 0,1 | 22.00 | 21.93 | **+0.07** | [-1.63, +1.39] | 0.764 | 0.764 | 0.201 | inconclusive |
| YOLOv8n | S3x 640x384 | uniform 704x384 | pedestrian | OC-SORT | 0,1 / 0,1 | 22.96 | 22.74 | **+0.22** | [-1.76, +1.71] | 0.584 | 0.584 | 0.456 | inconclusive |
| YOLOv8n | S3x 640x384 | plain 640x384 | car | ByteTrack | 0,1 / 0,1 | 44.05 | 43.82 | **+0.23** | [-0.81, +1.37] | 0.693 | 0.693 | 0.095 | inconclusive |
| YOLOv8n | S3x 640x384 | plain 640x384 | car | OC-SORT | 0,1 / 0,1 | 46.06 | 45.08 | **+0.98** | [-0.22, +2.13] | 0.114 | 0.228 | 0.329 | inconclusive |

Sensitivity (reported, not deciding): thresholds cross-fitted leave-one-source-video-out and bootstrap / randomisation over the 14 source videos of the 17 test-dev clips.

| A | B | class | tracker | d | 95 % CI | p_AR |
|---|---|---|---|---|---|---|
| S3x 640x384 | plain 640x384 | pedestrian | ByteTrack | +2.58 | [+0.11, +5.57] | 0.015 |
| S3x 640x384 | plain 640x384 | pedestrian | OC-SORT | +2.92 | [+0.21, +5.95] | 0.018 |
| S3x 640x384 | S3y 640x384 | pedestrian | ByteTrack | +2.76 | [+0.43, +5.01] | 0.039 |
| S3x 640x384 | S3y 640x384 | pedestrian | OC-SORT | +2.33 | [+0.30, +4.70] | 0.038 |
| S3y 640x384 | S3x 640x384 | car | ByteTrack | +0.85 | [-0.03, +1.83] | 0.019 |
| S3y 640x384 | S3x 640x384 | car | OC-SORT | +0.44 | [-0.95, +1.64] | 0.289 |
| S3x 640x384 | uniform 704x384 | pedestrian | ByteTrack | +0.07 | [-1.54, +1.60] | 0.794 |
| S3x 640x384 | uniform 704x384 | pedestrian | OC-SORT | +0.10 | [-1.99, +1.78] | 0.791 |
| S3x 640x384 | plain 640x384 | car | ByteTrack | +0.22 | [-0.91, +1.31] | 0.731 |
| S3x 640x384 | plain 640x384 | car | OC-SORT | +1.06 | [-0.19, +2.14] | 0.114 |

## BDD100K MOT val (200 videos)

Equivalence margin 0.0421 HOTA. Holm families: class x tracker over the contrasts of this dataset.

| detector | A | B | class | tracker | seeds A / B | HOTA A | HOTA B | d | 95 % CI | p_AR | Holm p_AR | TOST p (Holm) | label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv8n | S3x 416x256 | plain 416x256 | car | ByteTrack | 0,1 / 0,1,2 | 41.51 | 40.48 | **+1.03** | [+0.70, +1.35] | <0.001 | <0.001 | 1.000 | win |
| YOLOv8n | S3x 416x256 | plain 416x256 | car | OC-SORT | 0,1 / 0,1,2 | 45.47 | 44.49 | **+0.98** | [+0.70, +1.28] | <0.001 | <0.001 | 1.000 | win |
| YOLOv8n | S3x 416x256 | plain 416x256 | pedestrian | ByteTrack | 0,1 / 0,1,2 | 15.98 | 13.45 | **+2.52** | [+1.83, +3.33] | <0.001 | <0.001 | 1.000 | win |
| YOLOv8n | S3x 416x256 | plain 416x256 | pedestrian | OC-SORT | 0,1 / 0,1,2 | 18.90 | 16.34 | **+2.56** | [+1.90, +3.40] | <0.001 | <0.001 | 1.000 | win |
| YOLOv8n | S3x 416x256 | uniform 480x256 | car | ByteTrack | 0,1 / 0,1 | 41.51 | 41.30 | **+0.21** | [-0.13, +0.55] | 0.087 | 0.174 | 1.000 | inconclusive |
| YOLOv8n | S3x 416x256 | uniform 480x256 | car | OC-SORT | 0,1 / 0,1 | 45.47 | 45.51 | **-0.05** | [-0.31, +0.23] | 0.683 | 0.683 | 1.000 | inconclusive |
| YOLOv8n | S3x 416x256 | uniform 480x256 | pedestrian | ByteTrack | 0,1 / 0,1 | 15.98 | 14.44 | **+1.54** | [+0.79, +2.38] | <0.001 | <0.001 | 1.000 | win |
| YOLOv8n | S3x 416x256 | uniform 480x256 | pedestrian | OC-SORT | 0,1 / 0,1 | 18.90 | 17.53 | **+1.37** | [+0.66, +2.18] | <0.001 | <0.001 | 1.000 | win |
| YOLOv8n | plain 640x384 | S3x 416x256 | car | ByteTrack | 0,1,2 / 0,1 | 44.69 | 41.51 | **+3.18** | [+2.77, +3.57] | <0.001 | <0.001 | 1.000 | win |
| YOLOv8n | plain 640x384 | S3x 416x256 | car | OC-SORT | 0,1,2 / 0,1 | 49.09 | 45.47 | **+3.63** | [+3.26, +3.98] | <0.001 | <0.001 | 1.000 | win |
| YOLOv8n | plain 640x384 | S3x 416x256 | pedestrian | ByteTrack | 0,1,2 / 0,1 | 19.04 | 15.98 | **+3.06** | [+2.36, +3.75] | <0.001 | <0.001 | 1.000 | win |
| YOLOv8n | plain 640x384 | S3x 416x256 | pedestrian | OC-SORT | 0,1,2 / 0,1 | 22.28 | 18.90 | **+3.38** | [+2.65, +4.10] | <0.001 | <0.001 | 1.000 | win |
| YOLO11n | S3x 416x256 | plain 416x256 | car | ByteTrack | 0,1 / 0,1 | 41.00 | 40.11 | **+0.89** | [+0.54, +1.25] | <0.001 | <0.001 | 1.000 | win |
| YOLO11n | S3x 416x256 | plain 416x256 | car | OC-SORT | 0,1 / 0,1 | 44.85 | 43.93 | **+0.92** | [+0.54, +1.26] | <0.001 | <0.001 | 1.000 | win |
| YOLO11n | S3x 416x256 | plain 416x256 | pedestrian | ByteTrack | 0,1 / 0,1 | 15.29 | 13.65 | **+1.65** | [+0.83, +2.50] | <0.001 | <0.001 | 1.000 | win |
| YOLO11n | S3x 416x256 | plain 416x256 | pedestrian | OC-SORT | 0,1 / 0,1 | 18.04 | 16.15 | **+1.89** | [+1.14, +2.69] | <0.001 | <0.001 | 1.000 | win |
| YOLO11n | S3x 416x256 | uniform 480x256 | car | ByteTrack | 0,1 / 0,1 | 41.00 | 41.14 | **-0.14** | [-0.39, +0.14] | 0.157 | 0.174 | 1.000 | inconclusive |
| YOLO11n | S3x 416x256 | uniform 480x256 | car | OC-SORT | 0,1 / 0,1 | 44.85 | 45.02 | **-0.17** | [-0.56, +0.18] | 0.219 | 0.437 | 1.000 | inconclusive |
| YOLO11n | S3x 416x256 | uniform 480x256 | pedestrian | ByteTrack | 0,1 / 0,1 | 15.29 | 14.66 | **+0.64** | [-0.06, +1.33] | 0.008 | 0.008 | 1.000 | win |
| YOLO11n | S3x 416x256 | uniform 480x256 | pedestrian | OC-SORT | 0,1 / 0,1 | 18.04 | 17.47 | **+0.57** | [-0.31, +1.42] | 0.126 | 0.126 | 1.000 | inconclusive |

## Detector latency on a Jetson Orin Nano

GPU compute time of the TensorRT engine alone (trtexec, 15 W power mode, batch 1, canvas input; no pre-processing, NMS or tracking), median over a 10 s run after 500 ms warm-up. The INT8 engines were built without a calibration set, so they give latency only and no accuracy claim. Rows marked pass 2 repeat the measurement at the end of the session (drift check). Source, with the native-frame graphs: `deploy/jetson/results/latency_orin_nano.csv` (written by `deploy/jetson/parse_results.py`).

| trained on | arm | canvas (W x H) | precision | median ms | pass |
|---|---|---|---|---|---|
| dsec | plain 416x320 | 416 x 320 | FP16 | 2.939 | 1 |
| dsec | S3x 416x320 | 416 x 320 | FP16 | 3.263 | 1 |
| dsec | S3y 416x320 | 416 x 320 | FP16 | 3.274 | 1 |
| dsec | P2 416x320 (reference, model not in this repository) | 416 x 320 | FP16 | 4.507 | 1 |
| dsec | uniform 448x352 | 448 x 352 | FP16 | 3.127 | 1 |
| dsec | plain 640x480 | 640 x 480 | FP16 | 4.720 | 1 |
| bdd | plain 416x256 | 416 x 256 | FP16 | 2.522 | 1 |
| bdd | S3x 416x256 | 416 x 256 | FP16 | 2.743 | 1 |
| bdd | plain 640x384 | 640 x 384 | FP16 | 3.805 | 1 |
| dsec | plain 416x320 | 416 x 320 | INT8 | 2.340 | 1 |
| dsec | S3x 416x320 | 416 x 320 | INT8 | 2.712 | 1 |
| bdd | plain 640x384 | 640 x 384 | INT8 | 3.007 | 1 |
| dsec | plain 416x320 | 416 x 320 | FP16 | 2.887 | 2 |
| dsec | S3x 416x320 | 416 x 320 | FP16 | 3.270 | 2 |
| dsec | plain 640x480 | 640 x 480 | FP16 | 4.723 | 2 |

## Protocol

- **Thresholds are never chosen on the sequence they score.** Each tracker is run over a grid of detection thresholds (ByteTrack 0.2-0.9, OC-SORT 0.5-0.9). For every held-out sequence the threshold is the one with the best pooled HOTA on the other sequences (leave-one-sequence-out; BDD100K: 10 fixed folds of 20 videos), separately for each run, class and tracker.
- **HOTA is recombined exactly** from TrackEval's per-sequence sufficient statistics (TP, FN, FP and AssA x TP at the 19 localisation thresholds), so a pooled score over any set of sequences equals what TrackEval would report for that set.
- **Paired hierarchical bootstrap** (B = 10000): sequences are resampled with the same draw for both arms, then seeds within each arm; the 95 % and 90 % percentile intervals of d are reported.
- **Paired randomisation test**: per sequence, the seed-averaged statistics of A and B are swapped with probability 1/2 (exact enumeration when 2^K <= B: DSEC 2^13, MOT17 2^7; else B random patterns); two-sided p_AR.
- **Equivalence (TOST)** against a margin frozen before the comparisons were read: 2 x the median within-arm seed SD of pooled ByteTrack car HOTA over earlier 416 / 640 canvas arms (DSEC 1.1955, BDD100K 0.0421); it measures seed noise only, not any difference between arms. MOT17 and VisDrone use the DSEC value, frozen before any training on them, because a seed-SD estimate from 2 arms x 2 seeds would be too noisy.
- **Labels**: win / loss iff the Holm-adjusted p_AR < 0.05; otherwise equivalent iff the Holm-adjusted TOST p < 0.05; otherwise inconclusive. ' <m' marks a significant difference that is smaller than the margin.
- **Seeds**: every arm has 2 or 3 independently trained seeds (listed per row); all of them are used.

VisDrone2019-MOT and BDD100K are research-only datasets: this repository ships numbers for them, no media and no weights.
