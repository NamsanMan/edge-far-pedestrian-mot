# Third-party notices

This repository is released under the GNU Affero General Public License v3.0 (`LICENSE`). It builds on the software and
data listed below. Their licences apply to them; nothing here changes those terms.

## Software

| Component | Use in this repository | Licence |
|---|---|---|
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 8.3.170 | Python dependency. YOLOv8n / YOLO11n model definitions, `Detect` head, task-aligned assigner and `v8DetectionLoss` are imported; `efpmot/models` adapts the 8.3.170 detection loss to per-axis strides. Training starts from the Ultralytics COCO checkpoints `yolov8n.pt` / `yolo11n.pt`. | AGPL-3.0 |
| [BoxMOT](https://github.com/mikel-brostrom/boxmot) 25.0.0 | Python dependency. ByteTrack and OC-SORT implementations. | AGPL-3.0 |
| [TrackEval](https://github.com/JonathonLuiten/TrackEval) (commit `12c8791`) | Not included. Cloned by the user into `third_party/TrackEval` (see `docs/DATA.md`); HOTA / CLEAR / Identity evaluation. | MIT (text below) |
| PyTorch, torchvision, NumPy, SciPy, OpenCV, PyYAML, Pillow, ONNX, ONNX Runtime | Python dependencies. | BSD-style / Apache-2.0 / MIT, see each project |

Because Ultralytics YOLO and BoxMOT are AGPL-3.0, this repository is AGPL-3.0 as a whole. Network use of a modified
version obliges you to offer its source code (AGPL-3.0 section 13).

### TrackEval licence

```
MIT License

Copyright (c) 2020 Jonathon Luiten

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Datasets

No dataset, label file or converted label file is included in this repository. `docs/DATA.md` explains where to get
each dataset and how to convert it.

| Dataset | What this repository contains from it | Licence / terms |
|---|---|---|
| [DSEC](https://dsec.ifi.uzh.ch) / DSEC-Detection | Demo clips made from DSEC frames, with tracker boxes drawn on them; released model weights trained on it. The sequence split file `efpmot/data/splits/dsec_det_split.json`: the official train / val / test split (sequence names) and a day / night tag per sequence; nothing derived from the labels. | DSEC frames: CC BY-SA 4.0, so the demo clips are shared under CC BY-SA 4.0. The DSEC-Detection labels state no licence: they are never redistributed and no ground-truth box is drawn on any DSEC clip. |
| [MOT17](https://motchallenge.net) | Demo clips and the grid figure made from MOT17 frames and annotations (attribution drawn on the frames); released model weights trained on the MOT17-train first halves. | CC BY-NC-SA 3.0: the clips and the MOT17-trained weights are for non-commercial use, shared alike. |
| [HIT-UAV](https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset) | The Thermal Watch demo (video on YouTube, `apps/thermal_watch/media/thermal_watch_demo.jpg`) made from the HIT-UAV sample videos; released model weights trained on HIT-UAV; the recording split file `efpmot/data/splits/hituav_split.json` (recordings as serial-number blocks of each condition code, their flights and folds, derived from the file names and box counts); `apps/thermal_watch/sample_videos.json` (which recording each sample video is the source of, and which video frames are labelled images). No images and no labels. | CC BY 4.0 (the GitHub repository; the LICENSE file of the v1.2.1 release archive is CC0 1.0): the demo media and the weights are shared with attribution. |
| [VisDrone2019-MOT](https://github.com/VisDrone/VisDrone-Dataset) | Aggregate numbers only (evaluation statistics in `results/`). No images, no labels, no weights. | Research use only. |
| [BDD100K](https://bdd-data.berkeley.edu) | Aggregate numbers only (evaluation statistics in `results/`) and a list of BDD100K image file names that defines the 35k training subset. No images, no labels, no weights. | Research use only (BDD100K licence). |
| COCO (via the Ultralytics checkpoints) | Only as the pre-training of the starting checkpoints. | See the Ultralytics and COCO terms. |

## Methods referenced

No code of the following projects is included; they are named in the documentation only.

- Trackers and metrics: ByteTrack (Zhang et al., ECCV 2022), OC-SORT (Cao et al., CVPR 2023), HOTA (Luiten et al.,
  IJCV 2021).
- Detectors: YOLOv8 and YOLO11 (Ultralytics, AGPL-3.0) are used as dependencies. A port of the S3 edit (the stride
  change behind S3x and S3y) to YOLO26 (Ultralytics 8.4, AGPL-3.0) was tried; its code is not included. The S3 edit was
  also ported to YOLOX (Ge et al., 2021; Megvii, Apache-2.0) and NanoDet-Plus (RangiLyu, Apache-2.0) in experiments
  whose code is not included (`docs/EXPERIMENTS_LOG.md`). YOLOX-Nano / -Tiny, NanoDet-Plus-m, YOLO26n, YOLOv8s
  (Ultralytics) and RT-DETR-L (Zhao et al., CVPR 2024; trained with the Ultralytics implementation) are compared in
  `results/DETECTORS.md`. Only their evaluation statistics are included, not their training code.
- Prior work on non-square strides and axis-specific grids, cited in `docs/METHOD.md`: MatrixNets (Rashwan et al.,
  2019), LaserNet (Meyer et al., CVPR 2019) and the (2, 1) strides of text recognisers (for example CRNN, Shi et al.,
  TPAMI 2017).
- Evaluation split: the MOT17 half split (first half of each training sequence for training, second half for
  evaluation) follows CenterTrack (Zhou et al., ECCV 2020).
- Dataset papers: DSEC (Gehrig et al., RA-L 2021), DSEC-Detection (Gehrig and Scaramuzza, Nature 2024),
  MOT16/MOT17 (Milan et al., arXiv:1603.00831), VisDrone (Zhu et al., TPAMI 2021), BDD100K (Yu et al., CVPR 2020),
  HIT-UAV (Suo et al., Scientific Data 2023).
