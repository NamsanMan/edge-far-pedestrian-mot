# Model card: edge-far-pedestrian-mot detectors (efpmot 0.2.0)

Eight YOLOv8n detectors trained to find far, small people at the low input resolutions used on edge devices. They come
in pairs of one unmodified ("plain") YOLOv8n and one **YOLOv8n + S3x**, the main model of this project. There is one
pair for each of DSEC (driving) and MOT17 (street scenes), evaluated on tracking, and one for each of the two folds of
HIT-UAV, the thermal drone dataset of the Thermal Watch app (`apps/thermal_watch`). The HIT-UAV folds are
recording-disjoint: no recording appears in both the training and the test part of a fold. S3x changes four strides
inside YOLOv8n and adds no new operator. Its P3 grid, the finest grid of prediction points, keeps stride 8 vertically
but uses stride 4 horizontally: the grid becomes anisotropic, with twice as many columns, at about +11 % FLOPs.
Author: In Ho Jeung.

## Files

| File | Run tag (results/) | Family | Trained on | Canvas H x W | Epochs | Params | GFLOPs (canvas) |
|---|---|---|---|---|---|---|---|
| `yolov8n_plain_dsec_416x320_s0.pt` | `dsec_v8n_416_s0` | `yolov8n` | DSEC-Det train | 320 x 416 | 6 | 3.01 M | 2.67 |
| `yolov8n_s3x_dsec_416x320_s0.pt` | `dsec_v8n_416s3x_s0` | `yolov8n_s3x` | DSEC-Det train | 320 x 416 | 6 | 2.92 M | 2.96 |
| `yolov8n_plain_mot17_416x256_s0.pt` | `mot_v8n_416_s0` | `yolov8n` | MOT17 train, first halves | 256 x 416 | 55 | 3.01 M | 2.13 |
| `yolov8n_s3x_mot17_416x256_s0.pt` | `mot_v8n_416s3x_s0` | `yolov8n_s3x` | MOT17 train, first halves | 256 x 416 | 55 | 2.92 M | 2.36 |
| `yolov8n_plain_hituav-recA_416x320_s0.pt` | `hit_v8n_416recA_s0` | `yolov8n` | HIT-UAV fold recA, train part (1,263 images) | 320 x 416 | 116 | 3.01 M | 2.67 |
| `yolov8n_s3x_hituav-recA_416x320_s0.pt` | `hit_v8n_416s3xrecA_s0` | `yolov8n_s3x` | HIT-UAV fold recA, train part (1,263 images) | 320 x 416 | 116 | 2.92 M | 2.96 |
| `yolov8n_plain_hituav-recB_416x320_s0.pt` | `hit_v8n_416recB_s0` | `yolov8n` | HIT-UAV fold recB, train part (1,237 images) | 320 x 416 | 118 | 3.01 M | 2.67 |
| `yolov8n_s3x_hituav-recB_416x320_s0.pt` | `hit_v8n_416s3xrecB_s0` | `yolov8n_s3x` | HIT-UAV fold recB, train part (1,237 images) | 320 x 416 | 118 | 2.92 M | 2.96 |

- Run tag is the name under which the run's statistics are stored in `results/stats` (HIT-UAV: `results/hituav`).
  Family is the model name that `efpmot.models.build` takes.
- The canvas is the detector input size; every frame is letterboxed to it (see Input below). The table gives it as
  H x W, the order of the input tensor, while the file names give W x H.
- Each file is a bare fp32 PyTorch state dict of about 12 MB (`torch.save(model.state_dict())`). It loads with
  `torch.load(..., weights_only=True)`.
- Checksums are in `SHA256SUMS.txt`; check them with `sha256sum -c SHA256SUMS.txt`.
- The weights are bit-identical to the trained checkpoints behind the numbers in `results/`. Each file is the seed-0 run
  of its model; the published comparisons average 2 or 3 seeds (independent training runs) of each model.
- GFLOPs are 2 x the multiply-accumulate count (MACs) that thop reports for the inference graph on one canvas.
- S3x has fewer parameters than plain YOLOv8n because its P3 head is thinned to 32 channels (from 64). Its extra FLOPs
  come from the P3 feature map, which has twice as many columns.

## Use

`build` takes the family name, the number of classes (8) and the weights file:

```python
from efpmot.models import build
ad = build("yolov8n_s3x", 8, "release_assets/yolov8n_s3x_dsec_416x320_s0.pt")   # strict state-dict load
boxes, scores = ad.infer(x)   # x: float [B, 3, 320, 416], BGR 0..255, letterboxed -> xyxy canvas px, [B, A, 8] sigmoid
```

- **Input.** The frame is resized with its aspect ratio kept, placed at the top-left of the canvas and padded with grey
  (114) at the bottom and right (`efpmot.data.letterbox`). A DSEC frame (640 x 480) becomes 416 x 312 + padding; a MOT17
  frame (1920 x 1080) becomes 416 x 234 + padding.
- **Output.** For each anchor (one of the A prediction points of the three detection levels), a box as x1, y1, x2, y2
  in canvas pixels and 8 sigmoid class scores. Divide the box by the letterbox scale to get pixels of the original
  frame. The classes follow the BDD100K order: pedestrian, rider, car, bus, truck, train, motorcycle, bicycle.
  Pedestrian is 0 and car is 2.
- **Post-processing used for every number below.** Each anchor keeps only its highest class score. Boxes with
  conf >= 0.01 go through class-wise NMS at IoU 0.65, and at most 300 boxes are kept. Inference runs in FP16.
  `scripts/export_dets.py` does all of this.
- **Tracking.** ByteTrack or OC-SORT (BoxMOT 25.0), as in `efpmot.track`.
- **Only pedestrian is trained in the MOT17 files.** Their other seven class outputs are untrained and must be ignored.
- **HIT-UAV files** are trained on four outputs: pedestrian (HIT-UAV's Person class), car, bicycle and truck (HIT-UAV's
  OtherVehicle class). The other four outputs are untrained. Their evaluation export (`scripts/evaluate_hituav.py`) uses
  conf >= 0.001 instead of 0.01. The Thermal Watch app keeps person detections above 0.05 and uses the file's person
  threshold (`efpmot.release.RELEASES[...]["thr"]`, listed in the HIT-UAV table below) as the ByteTrack track
  threshold.
- **ONNX / TensorRT.** `deploy/jetson/export_onnx.py` exports the raw head output, before NMS (opset 17).

## Evaluation

The table gives the tracking HOTA of each file on held-out data, from `results/stats` (computed with `efpmot.eval.protocol.method`).
HOTA is the standard multi-object tracking score (0-100). The tracker threshold is the detection score threshold given
to the tracker. "Cross-fitted" means that each held-out sequence is scored at the threshold chosen on the other
sequences (leave-one-sequence-out). "thr 0.5" is a fixed threshold of 0.5 for every sequence.

| File | Test data | Class | ByteTrack cross-fitted | ByteTrack thr 0.5 | OC-SORT cross-fitted | OC-SORT thr 0.5 |
|---|---|---|---|---|---|---|
| `yolov8n_plain_dsec_416x320_s0.pt` | DSEC-Det test (13 seq.) | pedestrian | 27.69 | 26.67 | 28.63 | 28.63 |
| `yolov8n_s3x_dsec_416x320_s0.pt` | DSEC-Det test (13 seq.) | pedestrian | 33.43 | 33.16 | 33.59 | 33.59 |
| `yolov8n_plain_dsec_416x320_s0.pt` | DSEC-Det test (13 seq.) | car | 52.43 | 52.79 | 56.14 | 55.90 |
| `yolov8n_s3x_dsec_416x320_s0.pt` | DSEC-Det test (13 seq.) | car | 54.96 | 54.21 | 57.98 | 55.54 |
| `yolov8n_plain_mot17_416x256_s0.pt` | MOT17 train, second halves (7 seq.) | pedestrian | 41.02 | 40.84 | 41.31 | 40.86 |
| `yolov8n_s3x_mot17_416x256_s0.pt` | MOT17 train, second halves (7 seq.) | pedestrian | 43.58 | 42.51 | 44.55 | 41.08 |

The comparisons below use all seeds of each model and come from `results/RESULTS.md`. Each value is the difference
S3x - plain in HOTA points. The brackets give its 95 % confidence interval from the paired hierarchical bootstrap, which
resamples the test sequences (the same draw for both models) and the seeds of each model:

- DSEC pedestrian: ByteTrack +5.88 [95 % CI +4.95, +9.91], OC-SORT +5.49 [+4.37, +11.43].
- DSEC car: ByteTrack +2.84 [+0.57, +4.67], OC-SORT +2.18 [+1.15, +3.43].
- MOT17 pedestrian: ByteTrack +2.70 [+0.43, +4.01], OC-SORT +2.12 [-0.34, +3.73]. The prediction P1 (S3x beats plain),
  written down before any training on MOT17 (`docs/PREREGISTRATION.md`), holds only if the paired randomisation test
  gives p < 0.05 with both trackers. The ByteTrack p is 0.0625, so P1 is "not shown".
- S3x against the plain model on a larger input of about the same FLOPs (those models are not released). Values are
  ByteTrack / OC-SORT, and n.s. means not significant. DSEC at 448 x 352: pedestrian +2.24 / +2.81, car +0.52 / +1.77
  (n.s.). MOT17 at 480 x 256: pedestrian -0.51 / +0.77 (n.s.).
- These intervals resample the 13 DSEC test sequences, which come from only 8 drives, and sequences of one drive are
  not independent. When the test is repeated with whole drives as the unit (`results/RESULTS.md`), the DSEC pedestrian
  results keep their sign and size (ByteTrack: S3x - plain +5.88, p 0.016).

Cross-fitting selects a threshold for each held-out sequence. The thresholds selected most often for these files are:
- DSEC, ByteTrack: 0.4 for pedestrian. For car, plain takes 0.2; S3x takes 0.6 on most sequences and 0.4 on the rest.
- DSEC, OC-SORT: 0.5 for pedestrian, 0.6 for car.
- MOT17, ByteTrack: 0.8 (plain takes 0.9 on one sequence).
- MOT17, OC-SORT: 0.9 (plain also takes 0.8 on one sequence).

Latency on a Jetson Orin Nano (TensorRT FP16, GPU compute time of the detector engine only, 320 x 416 canvas):
2.94 ms for plain and 3.26 ms for S3x (`deploy/jetson/results/latency_orin_nano.md`). These timings were measured on the
DSEC-trained graphs. Latency depends only on the graph, not on the weights, so they also apply to the HIT-UAV files,
which have the same architecture and canvas. The MOT17 files have the same architecture on a smaller 256 x 416 canvas,
which the full latency table times with BDD100K-trained graphs.

**HIT-UAV files** (person detection, `results/HITUAV.md`): each file is evaluated on the test part of its own fold,
which holds only recordings the file never saw in training. AP50 is the average precision at IoU 0.5. F1, P (precision)
and R (recall) are measured at the person threshold, the score threshold with the best person F1 on the fold's val part.

| File | Person AP50 (test) | Person F1 / P / R at the threshold | Person threshold |
|---|---|---|---|
| `yolov8n_plain_hituav-recA_416x320_s0.pt` | 52.9 | 60.3 / 77.6 / 49.3 | 0.430 |
| `yolov8n_s3x_hituav-recA_416x320_s0.pt` | 59.3 | 67.2 / 73.4 / 62.0 | 0.154 |
| `yolov8n_plain_hituav-recB_416x320_s0.pt` | 57.5 | 63.8 / 61.4 / 66.3 | 0.057 |
| `yolov8n_s3x_hituav-recB_416x320_s0.pt` | 65.7 | 70.0 / 72.4 / 67.7 | 0.306 |

Over all seeds (3 per fold) and both folds, person AP50 is 55.8 for plain and 62.9 for S3x. The thresholds spread
widely because the val parts are small: one fold's val part is a single recording at 60 m altitude. Compare models by
AP, not by F1, P or R at these thresholds.

## Intended use

- **Intended:** research and education on small-object detection and tracking, reproduction of the results in this
  repository, a starting point for fine-tuning on your own data, and latency studies on embedded GPUs.
- **Out of scope:** safety-critical decisions (driving, collision avoidance), identification or profiling of individuals,
  and any deployment without validation on the target camera and scene. The detectors were trained on three small datasets:
  - DSEC: driving in and around Zurich and Interlaken, a single camera rig, including night sequences.
  - MOT17: 7 pedestrian scenes; only the first half of each sequence was used for training.
  - HIT-UAV: one thermal camera (DJI Zenmuse XT2, 640 x 512, white-hot) on a drone flying at 60-130 m, by day and by
    night; most person boxes are in night images. The 38 flights cover schools, parking lots, roads and playgrounds.

  Performance outside these domains is unknown.
- **Scores are not calibrated probabilities.** Choose the tracker threshold on validation data of your own; the
  cross-fitted thresholds above are one example.

## Training

- Every run starts from the Ultralytics COCO checkpoint `yolov8n.pt`. The P3 box branch of S3x, thinned to 32 channels,
  takes the leading 32 channels of the COCO weights. Every class branch is newly initialised, because its width depends
  on the number of classes.
- Batch 16, AdamW with lr 0.000833 and weight decay 5e-4 (on weights only, not on biases or batch-norm parameters), and
  a linear warm-up followed by cosine decay to 0.05 x lr.
- FP16 autocast and gradient-norm clipping at 10.
- Augmentation: horizontal flip, scale 0.8-1.2, brightness 0.7-1.3, then letterbox.
- DSEC: every second labelled frame of the official train split (41 sequences), 6 epochs.
- MOT17: the first half of each of the 7 MOT17-train sequences, 55 epochs. The FRCNN copy of each sequence is used; the
  DPM and SDP copies hold the same images and labels. The second halves were never seen in training. Pixels of the
  MOT17 distractor classes (person on vehicle, static person, distractor, reflection) are greyed out in the training
  images, so these objects are learned neither as pedestrian nor as background.
- HIT-UAV: the train part of one recording-disjoint fold (`efpmot/data/splits/hituav_split.json`), trained for the same
  ~9,060 iterations as the other datasets (116 epochs for recA, 118 for recB). Boxes of HIT-UAV's DontCare class are
  left out of training. The test recordings of the fold were never seen in training.
- Full recipe: `efpmot/train/__init__.py`. Command: `scripts/train.py`.

## Licence and data terms

- **Code:** AGPL-3.0 (this repository).
- **Weights:** every model starts from the Ultralytics COCO weights (AGPL-3.0) and is trained on the data below. The
  files therefore carry the upstream terms of both: the AGPL-3.0 of Ultralytics and the terms of the training data. This
  card adds no terms of its own.
- **DSEC files:** trained on DSEC frames (CC BY-SA 4.0, https://dsec.ifi.uzh.ch) with the DSEC-Detection labels
  (https://github.com/uzh-rpg/dsec-det). **The DSEC-Detection labels state no licence.** Training on them and
  publishing the resulting weights is not explicitly covered by any licence; check the current terms of DSEC-Detection
  (or ask its authors) before you redistribute or build on these files. Attribute DSEC (Gehrig et al., RA-L 2021) and
  DSEC-Detection (Gehrig and Scaramuzza, Nature 2024) when you use or share them.
- **MOT17 files:** trained on MOT17 data, licensed CC BY-NC-SA 3.0 (https://motchallenge.net).
  - **For non-commercial use only**, as required by the MOT17 licence (CC BY-NC-SA 3.0). Do not use them in a commercial
    product or service.
  - Share derived models under the same terms, with attribution to MOT17 (Milan et al., arXiv:1603.00831).
- **HIT-UAV files:** trained on HIT-UAV (Suo et al., Scientific Data 2023,
  https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset), CC BY 4.0 on GitHub (the LICENSE file of the v1.2.1
  release archive states CC0 1.0). Attribute HIT-UAV when you use or share them. Detecting people from the air is
  regulated in many places: check the privacy and surveillance law that applies before any real use.
- No dataset images or labels are included in the release.
- No weights trained on VisDrone or BDD100K are released: VisDrone states no licence, and the BDD100K licence allows
  only educational, research and not-for-profit use.

## Citation

```bibtex
@software{jeung2026efpmot,
  author = {Jeung, In Ho},
  title  = {edge-far-pedestrian-mot: anisotropic P3 grids for tracking far, small pedestrians on edge devices},
  year   = {2026},
  version = {0.2.0}
}
```
