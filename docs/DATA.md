# Datasets, weights and third-party code

No dataset, label file, converted label file or detection cache is part of this repository. This page says where to
get each one, what licence it has, where `configs/paths.yaml` expects it and how to convert it.

1. Copy `configs/paths.example.yaml` to `configs/paths.yaml` and set the paths. Relative paths are resolved against the
   repository root; the environment variable `EFPMOT_PATHS` can point to another file. Below, `<key>` stands for the
   folder of that key.
2. Dataset folders are only read. Everything the code writes goes under `outputs.*`. Converted labels go to
   `outputs.derived` (`<derived>`).
3. Run the conversion of each dataset you use: `python -B scripts/prepare_data.py {dsec,mot17,visdrone,bdd,all}`.

| dataset | used for | licence / terms | redistributed here |
|---|---|---|---|
| DSEC + DSEC-Detection | training and evaluation (driving, 640 x 480) | DSEC: CC BY-SA 4.0. DSEC-Detection labels: no licence stated | demo clips without GT boxes; trained weights |
| MOT17 | training (first halves) and evaluation (second halves) | CC BY-NC-SA 3.0 | demo clips and app media; trained weights (non-commercial) |
| VisDrone2019-MOT | training and evaluation (drone views) | research use only | numbers only |
| BDD100K | training (detection images) and evaluation (MOT val) | BDD100K licence, research use only | numbers only, plus a list of image file names |

## DSEC-Detection (`data.dsec_det`)

- **Source.** The DSEC dataset (https://dsec.ifi.uzh.ch) and the DSEC-Detection labels
  (https://github.com/uzh-rpg/dsec-det): `train_object_detections.zip` and `test_object_detections.zip`.
- **Images.** The left RGB images remapped to the 640 x 480 event-camera view (`images/left/distorted`), as distributed
  for DSEC-Detection, and `images/timestamps.txt` of each sequence, from the DSEC image archives. The timestamps map the
  label timestamps to image indices.
- **Layout** under `<dsec_det>`:

  ```
  images/{train,test}/<seq>/images/left/distorted/000000.png ...
  labels/{train,test}/<seq>/object_detections/left/tracks.npy
  meta/{train,test}/<seq>/images/timestamps.txt
  ```

  The validation sequences sit in the `train` folders. `data.dsec_images` may point to a faster copy of `images/` with
  the same layout (for example on an SSD); it is used when the folder exists.
- **Split.** The official train / val / test split, shipped as `efpmot/data/splits/dsec_det_split.json`. It holds
  the sequence names of each split and a day / night tag per sequence, nothing derived from the labels. Training: the
  train sequences, every 2nd labelled frame. Evaluation: the 13 test sequences, which come from 8 drives. Detections are
  exported for val + test (19 sequences).
- **Conversion.** `python -B scripts/prepare_data.py dsec` writes `<derived>/dsec_det/frame_index.json` (image count and
  labelled image indices per sequence) and `<derived>/dsec_det/gt_bdd/<seq>.json`. The latter is the BDD / Scalabel
  format that the TrackEval BDD100K evaluator reads.
  - Only labelled frames enter the GT. The unlabelled frames are long contiguous runs, so they are also left out of the
    evaluation; tracker outputs are restricted to the same frames.
  - Classes are mapped to the 8-class order used everywhere (`pedestrian, rider, car, bus, truck, train, motorcycle,
    bicycle`).
- **Licence.** The DSEC dataset page states CC BY-SA 4.0. The DSEC-Detection label page and repository state no licence
  for the labels, so this repository never redistributes them and never draws a DSEC ground-truth box.

## MOT17 (`data.mot17_train`)

- **Source.** MOTChallenge MOT17, train split (https://motchallenge.net). The test GT is not public, so only the train
  sequences are used.
- **Layout.** One image copy per sequence, the FRCNN folders (the DPM / SDP folders hold the same images and GT):

  ```
  <mot17_train>/MOT17-{02,04,05,09,10,11,13}-FRCNN/{img1/000001.jpg ..., gt/gt.txt, seqinfo.ini}
  ```

  `data.mot17_images` may point to a faster copy (only `img1/` is read from it).
- **Half split.** The standard CenterTrack / ByteTrack / FairMOT "half" convention. With n = seqLength, frames
  1 .. n//2 + 1 are for training and frames n//2 + 2 .. n for evaluation. No frame is in both halves; the 7 sequences
  give 2,664 training and 2,652 evaluation frames.
- **Conversion.** `python -B scripts/prepare_data.py mot17` writes the evaluation GT as its own MOTChallenge folder,
  because TrackEval has no built-in half convention: `<derived>/mot17/gt_halfval/<seq>/gt/gt.txt`, `seqinfo.ini` and
  `seqmaps/MOT17-halfval-{all,static,moving}.txt`.
  - `gt.txt` holds every raw line of the evaluation half (all classes, so the official distractor preprocessing still
    works), with frames renumbered from 1.
  - Tracker outputs use the same numbering.
- **Classes.**
  - MOT17 class 1 (pedestrian) is the `pedestrian` head, all visibilities.
  - The distractor classes 2, 7, 8 and 12 are greyed out in training and removed by the official preprocessing in
    evaluation.
  - Details: [`PREREGISTRATION.md`](PREREGISTRATION.md).
- **Licence.** CC BY-NC-SA 3.0. The MOT17-trained weights and all MOT17 media in this repository are for non-commercial
  use, with attribution, shared alike.

## VisDrone2019-MOT (`data.visdrone_mot`)

- **Source.** The official VisDrone2019-MOT train / val / test-dev archives
  (https://github.com/VisDrone/VisDrone-Dataset).
- **Layout.**

  ```
  <visdrone_mot>/VisDrone2019-MOT-{train,val,test-dev}/{sequences/<seq>/0000001.jpg ..., annotations/<seq>.txt}
  ```

  `data.visdrone_images` may point to a faster copy of the `sequences` folders (a missing split falls back to the main
  root).
- **Use.** Training on all 56 train sequences, every annotated frame. Evaluation on the 17 test-dev sequences. The val
  split is not used.
- **Conversion.** `python -B scripts/prepare_data.py visdrone` writes `<derived>/visdrone_mot/frame_index.json` and
  `<derived>/visdrone_mot/gt_bdd/{test-dev,val}/<seq>.json` (BDD format).
- **Classes and ignored regions** (decided before any VisDrone training, see [`PREREGISTRATION.md`](PREREGISTRATION.md)):
  - `pedestrian` = VisDrone category 1 only; `people` (2) is trained as `rider`.
  - `car` = category 4 only; `van` (5) is its own class in the otherwise unused `train` head.
  - Tricycles go to `motorcycle`.
  - Categories 0 and 11 are ignored regions. GT objects whose box centre lies in one are removed. Those pixels are greyed
    out in training, and tracker outputs centred in them are dropped before evaluation.
- **Terms.** Research use only. This repository contains aggregate numbers only: no images, labels or weights.

## BDD100K (`data.bdd_mot_val_images`, `data.bdd_mot_val_gt`, `data.bdd_det`)

- **Source.** BDD100K (https://bdd-data.berkeley.edu).
- **Evaluation** on the MOT val split, 200 sequences:
  - images under `<bdd_mot_val_images>/<seq>/*.jpg`;
  - the BDD100K-format GT that ships with TrackEval's example data (`data.zip` of the TrackEval repository,
    `data/gt/bdd100k/bdd100k_val/<seq>.json`) under `<bdd_mot_val_gt>`.
- **Training** on detection images with the legacy 100k labels (the 100k images zip and `bdd100k_labels.zip`):

  ```bash
  python -B scripts/prepare_data.py bdd --images_zip <images zip> --labels_zip <labels zip>
  ```

  - This extracts `train2017/` and `val2017/` into `<bdd_det>` and writes COCO annotations
    `annotations/det_train.json` / `det_val.json` with the 8 classes.
  - It then builds the 35k training subset `<derived>/bdd100k_det/annotations/det_train_35k.json` from
    `efpmot/data/splits/bdd_det_train_35k.txt`, the list of the 35,000 image file names used by every BDD run.
  - Without the zip options, only the subset is rebuilt from an existing `<bdd_det>`.
  - The detection val split is not used: its images come from the same videos as the MOT val sequences.
- **Terms.** Research use only (BDD100K licence). This repository contains aggregate numbers and the subset file-name
  list only: no images, labels or weights.

## Pretrained weights (`weights.yolov8n`, `weights.yolo11n`)

Every training run starts from the Ultralytics COCO checkpoints `yolov8n.pt` and `yolo11n.pt` (AGPL-3.0, from the
Ultralytics releases, https://github.com/ultralytics/assets/releases). Put them in `weights/` or set the two keys.

## TrackEval (`third_party.trackeval`)

HOTA, CLEAR and Identity metrics. MIT licence (text in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)). Use the
commit the numbers were computed with:

```bash
git clone https://github.com/JonathonLuiten/TrackEval third_party/TrackEval
git -C third_party/TrackEval checkout 12c8791b303e0a0b50f753af204249e622d0281a
```

`efpmot.eval` imports it from that folder. It does not need to be installed, and the NumPy aliases that this TrackEval
version still uses (`np.bool`, `np.int`, `np.float`, `np.object`) are patched at import time.

## What this repository does not redistribute

- Images, videos or labels of any dataset, and labels converted from them (frame indices, BDD-format GT, half-split
  GT, the 35k COCO subset).
- Detection caches and tracker outputs computed on the datasets. `results/stats/` holds only per-sequence HOTA
  sufficient statistics (true positive, false negative and false positive counts and association sums), which are
  enough to recompute the reported comparisons.
- DSEC ground-truth boxes, in any media.
- VisDrone and BDD100K media or weights.
