# Model card: edge-far-pedestrian-mot detectors (efpmot 0.1.0)

Four YOLOv8n detectors trained for tracking far, small pedestrians at low input resolution on edge devices. For each of two
datasets there is one plain YOLOv8n and one **YOLOv8n + S3x**, the anisotropic P3 grid of this project. S3x keeps stride 8
vertically and uses stride 4 horizontally, so the grid has twice as many columns, at about +11 % FLOPs. Author: In Ho Jeung.

## Files

| File | Run tag (results/) | Family | Trained on | Canvas H x W | Epochs | Params | GFLOPs (canvas) |
|---|---|---|---|---|---|---|---|
| `yolov8n_plain_dsec_416x320_s0.pt` | `dsec_v8n_416_s0` | `yolov8n` | DSEC-Det train | 320 x 416 | 6 | 3.01 M | 2.67 |
| `yolov8n_s3x_dsec_416x320_s0.pt` | `dsec_v8n_416s3x_s0` | `yolov8n_s3x` | DSEC-Det train | 320 x 416 | 6 | 2.92 M | 2.96 |
| `yolov8n_plain_mot17_416x256_s0.pt` | `mot_v8n_416_s0` | `yolov8n` | MOT17 train, first halves | 256 x 416 | 55 | 3.01 M | 2.13 |
| `yolov8n_s3x_mot17_416x256_s0.pt` | `mot_v8n_416s3x_s0` | `yolov8n_s3x` | MOT17 train, first halves | 256 x 416 | 55 | 2.92 M | 2.36 |

- Each file is a bare fp32 PyTorch state dict of about 12 MB (`torch.save(model.state_dict())`). It loads with
  `torch.load(..., weights_only=True)`.
- Checksums: `SHA256SUMS.txt` (`sha256sum -c SHA256SUMS.txt`).
- The weights are bit-identical to the trained checkpoints behind the numbers in `results/`. They are seed 0 of each arm;
  the published comparisons average 2 or 3 seeds per arm.
- GFLOPs are 2 x thop MACs of the inference graph on one canvas.
- S3x has fewer parameters than plain YOLOv8n because its P3 head is thinned to 32 channels. Its cost is in the
  higher-resolution P3 feature map.

## Use

```python
from efpmot.models import build
ad = build("yolov8n_s3x", 8, "release_assets/yolov8n_s3x_dsec_416x320_s0.pt")   # strict state-dict load
boxes, scores = ad.infer(x)   # x: float [B, 3, 320, 416], BGR 0..255, letterboxed -> xyxy canvas px, [B, A, 8] sigmoid
```

- **Input.** The frame is resized with its aspect ratio kept, placed at the top-left of the canvas and padded with grey
  (114) at the bottom and right (`efpmot.data.letterbox`). A DSEC frame (640 x 480) becomes 416 x 312 + padding; a MOT17
  frame (1920 x 1080) becomes 416 x 234 + padding.
- **Output.** Boxes in canvas pixels (divide by the letterbox scale for native pixels) and 8 class scores. The classes
  follow the BDD100K order: pedestrian, rider, car, bus, truck, train, motorcycle, bicycle. Pedestrian is 0 and car is 2.
- **Post-processing used for every number below.** The highest class score per anchor, conf >= 0.01, class-wise NMS at
  IoU 0.65, at most 300 boxes, FP16 inference. This is `scripts/export_dets.py`.
- **Tracking.** ByteTrack or OC-SORT (BoxMOT 25.0), as in `efpmot.track`.
- **Only pedestrian is trained in the MOT17 files.** Their other seven class outputs are untrained and must be ignored.
- **ONNX / TensorRT.** `deploy/jetson/export_onnx.py` exports the raw head (opset 17).

## Evaluation

The numbers are tracking HOTA on held-out data, from `results/stats` (`python -B scripts/compare.py`). "Cross-fitted"
means that each held-out sequence is scored at the tracker threshold chosen on the other sequences (leave-one-sequence-out).
"thr 0.5" is a fixed threshold of 0.5 for every sequence.

| File | Test data | Class | ByteTrack cross-fitted | ByteTrack thr 0.5 | OC-SORT cross-fitted | OC-SORT thr 0.5 |
|---|---|---|---|---|---|---|
| `yolov8n_plain_dsec_416x320_s0.pt` | DSEC-Det test (13 seq.) | pedestrian | 27.69 | 26.67 | 28.63 | 28.63 |
| `yolov8n_s3x_dsec_416x320_s0.pt` | DSEC-Det test (13 seq.) | pedestrian | 33.43 | 33.16 | 33.59 | 33.59 |
| `yolov8n_plain_dsec_416x320_s0.pt` | DSEC-Det test (13 seq.) | car | 52.43 | 52.79 | 56.14 | 55.90 |
| `yolov8n_s3x_dsec_416x320_s0.pt` | DSEC-Det test (13 seq.) | car | 54.96 | 54.21 | 57.98 | 55.54 |
| `yolov8n_plain_mot17_416x256_s0.pt` | MOT17 train, second halves (7 seq.) | pedestrian | 41.02 | 40.84 | 41.31 | 40.86 |
| `yolov8n_s3x_mot17_416x256_s0.pt` | MOT17 train, second halves (7 seq.) | pedestrian | 43.58 | 42.51 | 44.55 | 41.08 |

Arm-level comparisons over all seeds, with the paired hierarchical bootstrap. From `results/RESULTS.md`; d = S3x - plain,
in HOTA points:

- DSEC pedestrian: ByteTrack +5.88 [95 % CI +4.95, +9.91], OC-SORT +5.49 [+4.37, +11.43].
- DSEC car: ByteTrack +2.84 [+0.57, +4.67], OC-SORT +2.18 [+1.15, +3.43].
- MOT17 pedestrian: ByteTrack +2.70 [+0.43, +4.01], OC-SORT +2.12 [-0.34, +3.73]. The pre-registered prediction P1 is
  "not shown", because the ByteTrack randomisation p is 0.0625 and the rule needs p < 0.05 for both trackers.
- S3x against a uniform canvas of about the same FLOPs (those models are not released): DSEC 448 x 352,
  pedestrian +2.24 / +2.81, car +0.52 / +1.77 (n.s.); MOT17 480 x 256, pedestrian -0.51 / +0.77 (n.s., a tie).
- These intervals resample the 13 DSEC test sequences, which come from 8 drives; the drive-level check in
  `results/RESULTS.md` keeps the DSEC pedestrian results (S3x - plain +5.88, p 0.016).

Thresholds most often selected by cross-fitting for these files:
- DSEC, ByteTrack: 0.4 for pedestrian. For car, plain takes 0.2 and S3x takes 0.6 / 0.4.
- DSEC, OC-SORT: 0.5 for pedestrian, 0.6 for car.
- MOT17, ByteTrack: 0.8 (plain takes 0.9 on one sequence).
- MOT17, OC-SORT: 0.9 (plain also takes 0.8 on one sequence).

Latency on a Jetson Orin Nano (TensorRT FP16, GPU compute time of the detector engine only, 320 x 416 canvas):
2.94 ms for plain and 3.26 ms for S3x (`deploy/jetson/results/latency_orin_nano.md`). These timings are for the DSEC-trained
graphs. The MOT17 graphs have the same architecture on a smaller 256 x 416 canvas.

## Intended use

- **Intended:** research and education on small-object detection and tracking. Reproduction of the results in this
  repository. A starting point for fine-tuning on your own data. Latency studies on embedded GPUs.
- **Out of scope:** safety-critical decisions (driving, collision avoidance), identification or profiling of individuals,
  and any deployment without validation on the target camera and scene. The detectors were trained on two small datasets:
  - DSEC: driving in and around Zurich and Interlaken, a single camera rig, including night sequences.
  - MOT17: 7 pedestrian scenes, half of each sequence.

  Performance outside these domains is unknown.
- **Scores are not calibrated probabilities.** Choose the tracker threshold on validation data of your own; the
  cross-fitted thresholds above are one example.

## Training

- Every run starts from the Ultralytics COCO checkpoint `yolov8n.pt`. The S3 P3 box branch is sliced from the COCO weights;
  every class branch starts fresh.
- Batch 16. AdamW with lr 0.000833 and weight decay 5e-4 (on weights only). Linear warm-up, then cosine decay to 0.05 x lr.
- FP16 autocast and gradient clipping at 10.
- Augmentation: horizontal flip, scale 0.8-1.2, brightness 0.7-1.3, then letterbox.
- DSEC: the official train split (41 sequences), every second labelled frame, 6 epochs.
- MOT17: the first half of each of the 7 MOT17-train sequences (FRCNN copies), 55 epochs. The second halves were never
  seen in training. Distractor classes are masked in the training images.
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
  - **For non-commercial research use only**, as required by the MOT17 licence. Do not use them in a commercial
    product or service.
  - Share derived models under the same terms, with attribution to MOT17 (Milan et al., arXiv:1603.00831).
- No dataset images or labels are included in the release.
- No weights trained on VisDrone or BDD100K are released, because those datasets are for research use only.

## Citation

```bibtex
@software{jeung2026efpmot,
  author = {Jeung, In Ho},
  title  = {edge-far-pedestrian-mot: anisotropic P3 grids for tracking far, small pedestrians on edge devices},
  year   = {2026},
  version = {0.1.0}
}
```
