# Edge tracking of far, small pedestrians

Multi-object tracking of far, small pedestrians with the small input canvases that edge detectors use. A far
pedestrian is missed when its **short side**, its width, falls below about one P3 stride (8 px). **S3x** gives the P3
stage of YOLOv8n / YOLO11n a horizontal grid twice as fine (stride 8 vertically, 4 horizontally) for about +11 % FLOPs.
It is evaluated on four datasets with fixed trackers (ByteTrack, OC-SORT), cross-fitted tracker thresholds and paired
statistical tests, timed on a Jetson Orin Nano, and used in a zone-intrusion alert prototype.

![DSEC thun_01_b: two far pedestrians, plain YOLOv8n vs YOLOv8n + S3x](demo/dsec_clear_1.gif)

*Selected best case; the selection rule is in [`demo/README.md`](demo/README.md). DSEC thun_01_b, same ByteTrack
threshold for both models: plain YOLOv8n tracks the two far pedestrians (about 8 x 20 px on the 416 x 320 input) in
14 % of their frames, YOLOv8n + S3x in 100 %. Averages are in the results table below.*
*Frames: DSEC (https://dsec.ifi.uzh.ch), CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/); cropped,
enlarged and annotated with model tracks; the clip is shared under CC BY-SA 4.0.*

## Overview

- **Finding.** On small canvases, detection of a far object follows its short side, not its area: pedestrians are
  width-limited, cars height-limited. At a 416 px long side, 76 % of the VisDrone pedestrians have a short side below
  8 px.
- **Module.** S3x refines the P3 grid along the pedestrian's short axis by re-arranging four strides in the existing
  YOLOv8n graph. No new operator, P4 / P5 unchanged, +11 % FLOPs, ONNX / TensorRT friendly. S3y, the transposed
  module, is the control.
- **Result.** Pedestrian HOTA is higher than with the plain model of the same canvas on every dataset (+1.65 to +6.24;
  significant everywhere except MOT17 with ByteTrack). S3x also beats a uniform canvas of the same cost on DSEC and
  BDD100K; on MOT17 and VisDrone the two are tied.
- **Edge cost.** On a Jetson Orin Nano (TensorRT FP16, detector engine) S3x takes 3.26 ms instead of 2.94 ms.
- **Application.** A zone intrusion / loitering alert prototype on top of the tracker
  ([`apps/zone_alert`](apps/zone_alert/README.md)).

## Key finding: the short side decides

YOLOv8's task-aligned assigner only considers anchors whose centre lies inside a ground-truth box. A box narrower than
the 8 px anchor spacing can fall between two anchor columns, get no positive anchor at P3, and be trained as background.

![P3 anchor centres on a 4.3 px wide pedestrian](demo/grid_explainer.png)

*MOT17-02, frame 361: a pedestrian 4.3 x 12.6 px at the 416 x 256 input has no plain P3 anchor centre inside its box
and two with S3x. Frame and annotation: MOT17 (https://motchallenge.net), CC BY-NC-SA 3.0
(https://creativecommons.org/licenses/by-nc-sa/3.0/); cropped, enlarged and annotated; non-commercial use only.*

- In a logistic regression of "detected" on log canvas width and height (DSEC val, uniform YOLOv8n 416 x 320), the
  pedestrian width coefficient is +2.57 (SE 0.48) and the height coefficient +0.15 (SE 0.54); for cars it is the
  reverse (+0.03 and +2.87). This analysis code is not part of the repository.
- Ground-truth geometry (`python -B scripts/short_side_stats.py`): at a 416 px long side, 76 % of VisDrone pedestrians,
  37 % of VisDrone cars and 18 % of MOT17 pedestrians have a short side below 8 px; for 98 % of the VisDrone and 100 %
  of the MOT17 pedestrians the short side is the width.

Details: [`docs/METHOD.md`](docs/METHOD.md).

## Method: S3x, a horizontally finer P3 grid

| layer (YOLOv8n, strides as (y, x)) | plain | S3x |
|---|---|---|
| 3: Conv P2 -> P3 | (2, 2) | **(2, 1)** |
| 5: Conv P3 -> P4 | (2, 2) | **(2, 4)** |
| 13: Upsample P4 -> P3 | 2 | **(2, 4)** |
| 16: Conv P3 -> P4 (bottom-up) | (2, 2) | **(2, 4)** |
| P3 head width | 64 ch | **32 ch** |
| P3 grid at 416 x 320 | 40 x 52 | 40 x 104 |

- Anchors, box decoding and the loss are per-axis (`AnisoDetect`, `AnisoV8Loss`); everything else is the Ultralytics
  8.3.170 pipeline. All backbone and neck weights load from the COCO checkpoint.
- YOLO11n port: the same edits at layers 3, 5, 14 and 17, P3 head 24 channels (+11.0 % FLOPs).
- Cost at 416 x 320: 2.665 -> 2.955 GFLOPs (+10.9 %).

Code: [`efpmot/models`](efpmot/models). Design notes and prior work: [`docs/METHOD.md`](docs/METHOD.md).

## Results

### Tracking accuracy (cross-fitted thresholds, paired tests)

Pedestrian HOTA difference d, ByteTrack / OC-SORT, all seeds (2-3 per arm). n.s. = not significant (Holm-adjusted
randomisation p >= 0.05); every other entry is a significant win.

| dataset (test set) | detector, canvas | S3x - plain, same canvas | S3x - uniform canvas of the same cost |
|---|---|---|---|
| DSEC-Det test (13 sequences from 8 drives) | YOLOv8n, 416 x 320 | +5.88 / +5.49 | +2.24 / +2.81 (448 x 352) |
| BDD100K MOT val (200 videos) | YOLOv8n, 416 x 256 | +2.52 / +2.56 | +1.54 / +1.37 (480 x 256) |
| VisDrone2019-MOT test-dev (17 clips) | YOLOv8n, 640 x 384 | +2.58 / +3.05 | +0.07 / +0.22, n.s. (704 x 384) |
| MOT17 train, evaluation halves (7 sequences) | YOLOv8n, 416 x 256 | +2.70, n.s. (p 0.125) / +2.12 | -0.51 / +0.77, n.s. (480 x 256) |

- **Same cost** means matched on FLOPs: the uniform canvas costs 0 to 7 % more than S3x (VisDrone 0.8 % less).
- **YOLO11n** + S3x vs plain: DSEC +6.24 / +4.84, BDD100K +1.65 / +1.89.
- **Cars** gain against the plain model on DSEC and BDD100K (not significant for YOLO11n with OC-SORT on DSEC) and not
  on VisDrone; they tie with the same-cost canvas everywhere.
- **A larger canvas still wins.** A 640-wide canvas (about 2.1x the FLOPs of S3x) beats S3x: DSEC pedestrian
  +5.25 / +5.10, BDD100K pedestrian +3.06 / +3.38.
- **Pre-registered tests** (VisDrone, MOT17; written before any training on them): VisDrone P1 (S3x > plain) and P2
  (S3x > S3y) hold. On MOT17, P2 holds and P1 is "not shown" (ByteTrack p 0.062 under the both-trackers rule). The
  same-cost predictions (P4) are inconclusive on both.
- **Drive-level check.** The DSEC test sequences come from 8 drives. Resampling drives instead of sequences keeps the
  pedestrian results (S3x - plain +5.88, p 0.016).

All 64 contrast rows with intervals, p-values and labels: [`results/RESULTS.md`](results/RESULTS.md). Protocol:
[`docs/PROTOCOL.md`](docs/PROTOCOL.md).

### Latency on a Jetson Orin Nano

TensorRT 10.3, FP16, batch 1, 15 W mode; GPU compute time of the detector engine only (median of trtexec; no
pre-processing, NMS or tracking).

| model | canvas (W x H) | ms |
|---|---|---|
| YOLOv8n | 416 x 320 | 2.94 |
| YOLOv8n + S3x | 416 x 320 | 3.26 |
| YOLOv8n, uniform | 448 x 352 | 3.13 |
| YOLOv8n + P2 level (reference, code not included) | 416 x 320 | 4.51 |
| YOLOv8n | 640 x 480 | 4.72 |
| INT8 plain / S3x (no calibration, latency only) | 416 x 320 | 2.34 / 2.71 |

The same-cost 448 x 352 engine is slightly faster than S3x (3.13 vs 3.26 ms): FLOPs are not latency. The full
camera-to-alert pipeline has not been measured on the Jetson. Export and benchmark scripts:
[`deploy/jetson`](deploy/jetson/README.md).

## Demo

| clip | what |
|---|---|
| [`demo/dsec_clear_1.gif`](demo/dsec_clear_1.gif) | DSEC, two far pedestrians, crop, 3 s (above) |
| [`demo/dsec_compare_2.gif`](demo/dsec_compare_2.gif) | DSEC, same scene, full frame + 2x zoom, 7 s |
| [`demo/mot17_clear_1.gif`](demo/mot17_clear_1.gif) | MOT17-10, one far pedestrian, moving camera, 3 s |
| [`demo/grid_explainer.png`](demo/grid_explainer.png) | P3 anchor centres on a narrow pedestrian (above) |

Every clip is a selected best case, chosen by a fixed rule from all windows of the evaluation sequences; the rule, the
counts behind each caption and how rare such cases are: [`demo/README.md`](demo/README.md). No ground-truth box is
drawn on the DSEC clips.

## Application: zone intrusion and loitering alerts

[`apps/zone_alert`](apps/zone_alert/README.md) raises INTRUSION and LOITERING alerts for named polygons on the
S3x + ByteTrack tracker; v2 adds an approach warning, a direction filter and tripwires. On the MOT17-04 / -09
evaluation halves (one seed, no confidence intervals) S3x and plain alert 40 vs 39 of 48 far-zone entries, with 28 vs 42
false alerts; plain catches more loitering (6 vs 4 of 18). The v2 approach warning is not yet useful (4 of 24 far-zone
walk-ins warned, against 10 false warnings). It is a research prototype; see the responsible-use note there.

## Installation

```bash
# 1. PyTorch with CUDA from https://pytorch.org (pip from PyPI gives a CPU-only build on Windows)
# 2. this package (Python 3.10-3.13)
pip install -e .            # add .[dev] for the FLOPs test, .[deploy] for the ONNX export
# 3. tests (CPU, no dataset needed; data-dependent tests are skipped)
python -B -m unittest discover -s tests
python -B apps/zone_alert/tests/test_zone_monitor.py
```

Tested with Python 3.10, torch 2.8.0 + CUDA 12.9, ultralytics 8.3.170 and boxmot 25.0.0 (both pinned).

## Data

No dataset is included. [`docs/DATA.md`](docs/DATA.md) explains where to get DSEC-Detection, MOT17, VisDrone2019-MOT
and BDD100K, their terms, the expected layout and the conversion (`scripts/prepare_data.py`). Paths are set in
`configs/paths.yaml`, copied from `configs/paths.example.yaml` (or a file named by `EFPMOT_PATHS`).

## Reproducing the results

Every table of `results/RESULTS.md` is regenerated on CPU in under a minute from the shipped per-sequence statistics,
without any dataset:

```bash
python -B scripts/compare.py --table results/contrasts.yaml
```

- One contrast as JSON: `python -B scripts/compare.py --one dsec v8n 416s3x 416 --cls pedestrian --trk bytetrack`.
- Evaluating the release weights and the full pipeline (train, export detections, track, TrackEval, compare):
  [`docs/REPRODUCE.md`](docs/REPRODUCE.md).
- How this code was checked against the original research outputs: [`docs/PORT_VERIFICATION.md`](docs/PORT_VERIFICATION.md).
- Alternatives that were tried and dropped: [`docs/EXPERIMENTS_LOG.md`](docs/EXPERIMENTS_LOG.md).

## Pretrained weights

Four YOLOv8n checkpoints (seed 0) are published on the GitHub Releases page of this repository, with
[`release_assets/MODEL_CARD.md`](release_assets/MODEL_CARD.md) and `SHA256SUMS.txt`:

| file | trained on | canvas (W x H) |
|---|---|---|
| `yolov8n_plain_dsec_416x320_s0.pt` / `yolov8n_s3x_dsec_416x320_s0.pt` | DSEC-Det train | 416 x 320 |
| `yolov8n_plain_mot17_416x256_s0.pt` / `yolov8n_s3x_mot17_416x256_s0.pt` | MOT17 train, first halves | 416 x 256 |

Put them in `release_assets/` (`sha256sum -c SHA256SUMS.txt`) and load them with
`efpmot.models.build("yolov8n_s3x", 8, "release_assets/yolov8n_s3x_dsec_416x320_s0.pt")`. The weights carry the terms
of their training data: the MOT17 files are for non-commercial research use only, and the DSEC-Detection labels state
no licence (see the model card). No weights trained on VisDrone or BDD100K are released.

## Repository layout

```
efpmot/            Python package: models (S3x / S3y), data readers, training, tracking, evaluation protocol
scripts/           command-line entry points: prepare_data, train, export_dets, track, evaluate, compare, ...
configs/           paths.example.yaml (copy to paths.yaml)
results/           contrasts.yaml, per-run HOTA statistics (stats/), generated tables (RESULTS.md, summary*)
docs/              METHOD, PROTOCOL, PREREGISTRATION, DATA, REPRODUCE, PORT_VERIFICATION, EXPERIMENTS_LOG
demo/              demo clips and the grid figure
deploy/jetson/     ONNX export, TensorRT benchmark script and the measured latency table
apps/zone_alert/   zone intrusion / loitering alert prototype, its evaluation and demo
release_assets/    model card and checksums of the release weights (the weights themselves are not committed)
tests/             unit tests (CPU)
```

## Licence and data terms

| what | licence |
|---|---|
| Code and documentation (`efpmot/`, `scripts/`, `apps/zone_alert/*.py`, `deploy/`, `tests/`, `docs/`) | AGPL-3.0 ([`LICENSE`](LICENSE)), because Ultralytics YOLO and BoxMOT are AGPL-3.0 |
| `demo/dsec_clear_1.*`, `demo/dsec_compare_2.*` | CC BY-SA 4.0 (derived from DSEC frames) |
| `demo/mot17_clear_1.*`, `demo/grid_explainer.*` | CC BY-NC-SA 3.0 (derived from MOT17), non-commercial |
| `apps/zone_alert/demo/`, `apps/zone_alert/zones/*.png` | CC BY-NC-SA 3.0 (derived from MOT17), non-commercial |
| `apps/zone_alert/results*.json`, `apps/zone_alert/demo/*_log*.json` | CC BY-NC-SA 3.0 (hold episodes derived from MOT17 ground truth) |
| `results/` | aggregate evaluation statistics only; no images or labels of any dataset |
| Release weights | the terms in [`release_assets/MODEL_CARD.md`](release_assets/MODEL_CARD.md) |

No dataset images or labels are redistributed except the derived media listed above. VisDrone and BDD100K are
research-only datasets: this repository holds numbers for them and nothing else. Third-party software and method
notices: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Citation

See [`CITATION.cff`](CITATION.cff), or:

```bibtex
@software{jeung2026efpmot,
  author  = {Jeung, In Ho},
  title   = {edge-far-pedestrian-mot: anisotropic P3 grids for tracking far, small pedestrians on edge devices},
  year    = {2026},
  version = {0.1.0}
}
```

## Author

In Ho Jeung
