# TensorRT latency on a Jetson Orin Nano

Device: Jetson Orin Nano, power mode `NV Power Mode: 15W`, TensorRT 10.3 (`trtexec --warmUp=500 --duration=10 --useSpinWait`). Source: `bench_2026-10-27.log`, parsed by `parse_results.py`.

**What is timed:** the GPU compute time of the detector engine alone (trtexec), batch 1, raw head output. Pre-processing on the CPU, NMS and tracking are not included. `native` graphs take the camera frame and do the letterbox resize inside the engine.

**INT8:** the engines were built with `--int8 --fp16` and no calibration set, so the INT8 rows are latency only. No accuracy is claimed for them.

| model | trained on | canvas (W x H) | input | precision | median [ms] |
|---|---|---|---|---|---|
| YOLOv8n | DSEC | 416 x 320 | canvas | FP16 | 2.94 |
| YOLOv8n + S3x | DSEC | 416 x 320 | canvas | FP16 | 3.26 |
| YOLOv8n + S3y | DSEC | 416 x 320 | canvas | FP16 | 3.27 |
| YOLOv8n | DSEC | 448 x 352 | canvas | FP16 | 3.13 |
| YOLOv8n | DSEC | 640 x 480 | canvas | FP16 | 4.72 |
| YOLOv8n | DSEC | 640 x 480 | native 640 x 480 | FP16 | 4.73 |
| YOLOv8n | BDD100K | 416 x 256 | canvas | FP16 | 2.52 |
| YOLOv8n + S3x | BDD100K | 416 x 256 | canvas | FP16 | 2.74 |
| YOLOv8n | BDD100K | 640 x 384 | canvas | FP16 | 3.81 |
| YOLOv8n | BDD100K | 416 x 256 | native 1280 x 720 | FP16 | 2.74 |
| YOLOv8n | BDD100K | 640 x 384 | native 1280 x 720 | FP16 | 4.18 |
| YOLOv8n | DSEC | 416 x 320 | canvas | INT8 | 2.34 |
| YOLOv8n + S3x | DSEC | 416 x 320 | canvas | INT8 | 2.71 |
| YOLOv8n | BDD100K | 640 x 384 | canvas | INT8 | 3.01 |

Drift check: FP16 graphs timed again at the end of the session, interleaved.

| model | canvas | input | first pass median [ms] | repeat median [ms] |
|---|---|---|---|---|
| YOLOv8n | 416 x 320 | canvas | 2.94 | 2.89 |
| YOLOv8n + S3x | 416 x 320 | canvas | 3.26 | 3.27 |
| YOLOv8n | 640 x 480 | canvas | 4.72 | 4.72 |
| YOLOv8n | 640 x 480 | native 640 x 480 | 4.73 | 4.78 |

Headline (416 x 320 canvas, FP16, median): YOLOv8n 2.94 ms, YOLOv8n + S3x 3.26 ms (+11 %, close to its +11 % FLOPs); INT8 2.34 vs 2.71 ms.

Other P3 designs timed in the same session (their code is not part of this repository; see `docs/EXPERIMENTS_LOG.md`):

| design | canvas (W x H) | precision | median [ms] |
|---|---|---|---|
| YOLOv8n + P2 level (stride 4 head) | 416 x 320 | FP16 | 4.51 |
| YOLOv8n + shallow x-fine P3 branch (MatrixNets-style, FLOPs matched to S3x) | 416 x 320 | FP16 | 3.55 |
| YOLOv8n + two half-width P3 branches (x-fine and y-fine) | 416 x 320 | FP16 | 4.14 |

**Detector comparison, second session** (`bench_2026-11-13.log`, power mode `NV Power Mode: 15W`, same trtexec settings): DSEC-trained detectors on the 416 x 320 canvas; every graph takes the canvas input of `export_onnx.py` and ends in the family's raw head output before NMS. YOLOv8n, YOLOv8n + S3x and YOLO11n are model families of this repository; the other detectors come from their own implementations, whose code is not part of this repository. Their tracking accuracy: `results/DETECTORS.md`.

| model | canvas (W x H) | precision | median [ms] |
|---|---|---|---|
| YOLOv8n | 416 x 320 | FP16 | 2.94 |
| YOLOv8n + S3x | 416 x 320 | FP16 | 3.27 |
| YOLOv8s | 416 x 320 | FP16 | 4.79 |
| YOLO11n | 416 x 320 | FP16 | 3.49 |
| YOLO26n (one-to-one head, every anchor decoded, no top-k) | 416 x 320 | FP16 | 3.64 |
| YOLOX-Nano | 416 x 320 | FP16 | 3.12 |
| YOLOX-Tiny | 416 x 320 | FP16 | 3.90 |
| NanoDet-Plus-m (without the auxiliary training head) | 416 x 320 | FP16 | 4.69 |
| RT-DETR-L (300 decoder queries, NMS-free) | 416 x 320 | FP16 | 26.70 |

Drift check: timed again at the end of this session, interleaved: YOLOv8n 2.94 -> 2.93 ms, YOLOv8n + S3x 3.27 -> 3.29 ms. The first session timed the same ONNX graphs at 2.94 ms (YOLOv8n) and 3.26 ms (YOLOv8n + S3x).
