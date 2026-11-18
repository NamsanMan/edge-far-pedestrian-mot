# Datasets, weights and third-party code

No dataset, label file, converted label file or detection cache is part of this repository. For each dataset, this page
says where to get it, what licence it has, where `configs/paths.yaml` expects it and how to convert it.

1. Copy `configs/paths.example.yaml` to `configs/paths.yaml` and set the paths. Relative paths are resolved against the
   repository root. To use another configuration file, set the environment variable `EFPMOT_PATHS` to its path. Below,
   a key name in angle brackets stands for the folder set by that key: `<dsec_det>` is the folder of `data.dsec_det`.
2. Dataset folders are only read. Everything the code writes goes under `outputs.*`. Converted labels go to
   `outputs.derived` (`<derived>`).
3. Run the conversion of each dataset you use: `python -B scripts/prepare_data.py {dsec,mot17,visdrone,bdd,all}`. Give
   one dataset name per call, or `all` for these four. HIT-UAV needs no conversion.

| dataset | used for | licence / terms | redistributed here |
|---|---|---|---|
| DSEC + DSEC-Detection | training and evaluation (driving, 640 x 480) | DSEC: CC BY-SA 4.0. DSEC-Detection labels: no licence stated | demo clips without GT boxes; trained weights |
| MOT17 | training (first halves) and evaluation (second halves) | CC BY-NC-SA 3.0 | demo clips and the grid figure; trained weights (non-commercial) |
| VisDrone2019-MOT | training and evaluation (drone views) | research use only | numbers only |
| BDD100K | training (detection images) and evaluation (MOT val) | BDD100K licence, research use only | numbers only, plus a list of image file names |
| HIT-UAV | person detection on thermal drone images (the Thermal Watch app) | CC BY 4.0 (GitHub; the release archive's LICENSE: CC0 1.0) | app demo media; trained weights; the recording split |

## DSEC-Detection (`data.dsec_det`)

- **Source.** The DSEC dataset (https://dsec.ifi.uzh.ch) and the DSEC-Detection labels
  (https://github.com/uzh-rpg/dsec-det): `train_object_detections.zip` and `test_object_detections.zip`.
- **Images.** Each sequence needs the left RGB images remapped to the 640 x 480 event-camera view
  (`images/left/distorted`), as distributed for DSEC-Detection. It also needs its `images/timestamps.txt` from the DSEC
  image archives, which maps the label timestamps to image indices.
- **Layout** under `<dsec_det>`:

  ```
  images/{train,test}/<seq>/images/left/distorted/000000.png ...
  labels/{train,test}/<seq>/object_detections/left/tracks.npy
  meta/{train,test}/<seq>/images/timestamps.txt
  ```

  The validation sequences sit in the `train` folders. `data.dsec_images` may point to a faster copy of `images/` with
  the same layout (for example on an SSD); it is used when the folder exists.
- **Split.** The official train / val / test split is shipped as `efpmot/data/splits/dsec_det_split.json`. It holds
  the sequence names of each split and a day / night tag per sequence; nothing in it is derived from the labels.
  Training uses every 2nd labelled frame of the train sequences. Evaluation uses the 13 test sequences, which come from
  8 drives. Detections are exported for val + test (19 sequences), that is, computed once and cached as tracker input.
- **Conversion.** `python -B scripts/prepare_data.py dsec` writes `<derived>/dsec_det/frame_index.json` (image count and
  labelled image indices per sequence) and `<derived>/dsec_det/gt_bdd/<seq>.json`. The latter is the ground truth (GT)
  of each sequence in the BDD / Scalabel format, which the TrackEval BDD100K evaluator reads.
  - Only labelled frames enter the GT. The unlabelled frames come in long contiguous runs without annotation, so they
    are left out of the evaluation too: tracker outputs are restricted to the labelled frames.
  - Classes are mapped to the 8-class order of every detector head in this repository (`pedestrian, rider, car, bus,
    truck, train, motorcycle, bicycle`).
- **Licence.** The DSEC dataset page states CC BY-SA 4.0. The DSEC-Detection label page and repository state no licence
  for the labels, so this repository never redistributes them and never draws a DSEC ground-truth box.

## MOT17 (`data.mot17_train`)

- **Source.** MOTChallenge MOT17, train split (https://motchallenge.net). The test GT is not public, so only the train
  sequences are used.
- **Layout.** One copy of each sequence is read, the FRCNN folder; the DPM and SDP folders hold the same images and GT:

  ```
  <mot17_train>/MOT17-{02,04,05,09,10,11,13}-FRCNN/{img1/000001.jpg ..., gt/gt.txt, seqinfo.ini}
  ```

  `data.mot17_images` may point to a faster copy (only `img1/` is read from it).
- **Half split.** The split follows the standard "half" convention of CenterTrack, ByteTrack and FairMOT. With
  n = seqLength (the frame count in `seqinfo.ini`), frames 1 .. n//2 + 1 are for training and frames n//2 + 2 .. n for
  evaluation. No frame is in both halves; the 7 sequences give 2,664 training and 2,652 evaluation frames.
- **Conversion.** TrackEval has no built-in half convention, so `python -B scripts/prepare_data.py mot17` writes the GT
  of the evaluation halves as a MOTChallenge folder of its own: `<derived>/mot17/gt_halfval/<seq>/gt/gt.txt`,
  `seqinfo.ini` and `seqmaps/MOT17-halfval-{all,static,moving}.txt`.
  - `gt.txt` holds every raw line of the evaluation half, with frames renumbered from 1. All classes are kept so that
    the official distractor preprocessing (see Classes) still works.
  - Tracker outputs use the same numbering.
  - The seqmaps are the sequence lists that TrackEval reads: all sequences, the static-camera ones and the
    moving-camera ones.
- **Classes.**
  - MOT17 class 1 (pedestrian) maps to the `pedestrian` head. Every class-1 box is used, whatever its visibility.
  - The distractor classes 2, 7, 8 and 12 (person on vehicle, static person, distractor, reflection) are greyed out in
    training: their pixels are filled with grey, so the model learns them neither as pedestrians nor as background. In
    evaluation, the official TrackEval preprocessing removes them together with the tracker boxes that match them.
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

  `data.visdrone_images` may point to a faster copy of the `sequences` folders. A split that is missing from the copy
  is read from `<visdrone_mot>`.
- **Use.** Training uses every annotated frame of all 56 train sequences, and evaluation the 17 test-dev sequences. The
  val split is not used.
- **Conversion.** `python -B scripts/prepare_data.py visdrone` writes `<derived>/visdrone_mot/frame_index.json` and
  `<derived>/visdrone_mot/gt_bdd/{test-dev,val}/<seq>.json` (BDD format).
- **Classes and ignored regions** (decided before any VisDrone training, see [`PREREGISTRATION.md`](PREREGISTRATION.md)):
  - `pedestrian` = VisDrone category 1 only; `people` (2) is trained as `rider`.
  - `car` = category 4 only; `van` (5) is trained as a class of its own in the `train` head, which is otherwise unused.
  - Tricycles go to `motorcycle`.
  - Categories 0 and 11 are ignored regions. GT objects whose box centre lies in one are removed. The pixels of these
    regions and of the removed objects are greyed out in training, and tracker outputs centred in an ignored region are
    dropped before evaluation.
- **Terms.** Research use only. This repository contains aggregate numbers only: no images, labels or weights.

## BDD100K (`data.bdd_mot_val_images`, `data.bdd_mot_val_gt`, `data.bdd_det`)

- **Source.** BDD100K (https://bdd-data.berkeley.edu).
- **Evaluation** on the MOT val split, 200 sequences:
  - images under `<bdd_mot_val_images>/<seq>/*.jpg`;
  - the BDD100K-format GT under `<bdd_mot_val_gt>`. It ships with TrackEval's example data, as
    `data/gt/bdd100k/bdd100k_val/<seq>.json` inside `data.zip` of the TrackEval repository.
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

## HIT-UAV (`data.hituav`, `data.hituav_videos`)

- **Source.** HIT-UAV (Suo et al., Scientific Data 2023; https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset):
  the release archive `HIT-UAV.zip` of v1.2.1 (775 MiB). It holds 2,898 thermal images (640 x 512) taken with a DJI
  Zenmuse XT2 on a drone at 60-130 m altitude, at camera angles of 30-90 degrees (90 = looking straight down), by day
  and by night. The labels are boxes only, without track IDs.
- **Layout** under `<hituav>`, the extracted `HIT-UAV` folder (its `normal_xml` and `rotate_*` folders are not used):

  ```
  normal_json/{train,val,test}/<day/night>_<altitude>_<angle>_<weather>_<serial>.jpg
  normal_json/annotations/{train,val,test}.json      (COCO-like; the box list is under "annotation")
  ```

  In the image names, `<day/night>` is 0 (day) or 1 (night), `<altitude>` is in m, `<angle>` in degrees, and
  `<weather>` is always 0. `<serial>` is a running number that is unique across the dataset. The name without it (for
  example `1_60_30_0`) is called the condition code below.
- **Sample videos** (`data.hituav_videos`, optional): the five `video_sample/*.mov` files of the GitHub repository
  (640 x 512, 7.5 fps). Only `apps/thermal_watch` uses them. Each one is the source video of one recording of the
  dataset: every labelled image of that recording is a frame of the video.
  [`apps/thermal_watch/sample_videos.json`](../apps/thermal_watch/sample_videos.json) lists the recording and the
  labelled frames of each video.
- **Classes.** Person -> `pedestrian`, Car -> `car`, Bicycle -> `bicycle`, OtherVehicle -> `truck`. DontCare boxes are
  left out of training; unlike the VisDrone ignored regions, their pixels are not greyed out, so the loss treats them as
  background. In evaluation they are ignore regions: a detection that is not a true positive but matches a
  DontCare box is ignored, not counted as a false positive.
- **Split.** The official train / val / test split interleaves images of the same recordings: for 96 % of the test
  images, a training image of the same recording is at most 10 serial numbers away. The official test set therefore
  measures recordings seen in training, which inflates person AP50 by about 20 points. This repository uses
  recording-disjoint splits instead, in which no recording has images in both the training and the test part. They
  are shipped as `efpmot/data/splits/hituav_split.json`:
  - A recording is a run of images with one condition code whose consecutive serial numbers differ by at most 30. The
    dataset has 264 recordings in 38 flights (runs of consecutive recordings with the same day / night flag, altitude
    and weather).
  - Folds `recA` / `recB`: every recording is in the test part of exactly one of the two folds, and the person boxes
    of each altitude and day / night group are balanced between the two test parts. Every test recording shares its
    flight with training recordings of its fold, so these folds test unseen recordings, not unseen places.
  - Fold `altLH` (unseen altitude): train and val at 60-90 m, test at 100-130 m. No flight has recordings on both
    sides.

  `python -B scripts/prepare_data.py hituav` rebuilds the split from the file names and the box counts of the
  annotation files and checks it against the shipped file. A rebuild that differs replaces the shipped file only with
  `--force`.
- **Serial numbers are not time order.** Within a recording, consecutive serials follow the frame indices sorted as text
  (0, 1, 10, ..., 19, 2, 20, ...), not as numbers. `efpmot.data.hituav.recording_time_indices` maps serials back to
  time indices. For this it needs the block length, that is, the number of frames extracted from the source video. By
  default it takes the block length from the first serial of the next recording, which is wrong when a whole block of
  images is missing from the release: for recording r000, pass `m=150`. In the five recordings that have a sample
  video, the images are every 7th frame of the 7.5 fps video (0.93 s apart).
- **Licence.** CC BY 4.0 on GitHub; the LICENSE file of the v1.2.1 archive states CC0 1.0. This repository holds no
  HIT-UAV images or labels: only the split file (recordings as serial-number blocks of each condition code, their flights, box counts and
  folds), the demo media of the app and the
  trained weights, all with attribution.

## Pretrained weights (`weights.yolov8n`, `weights.yolo11n`)

Every training run starts from the Ultralytics COCO checkpoint of its model, `yolov8n.pt` or `yolo11n.pt` (AGPL-3.0,
from the Ultralytics releases, https://github.com/ultralytics/assets/releases). Put them in `weights/` or set
`weights.yolov8n` and `weights.yolo11n` to their paths.

## TrackEval (`third_party.trackeval`)

TrackEval computes the HOTA, CLEAR and Identity metrics. It is under the MIT licence (text in
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)). Use the commit that the reported numbers were computed with:

```bash
git clone https://github.com/JonathonLuiten/TrackEval third_party/TrackEval
git -C third_party/TrackEval checkout 12c8791b303e0a0b50f753af204249e622d0281a
```

`efpmot.eval` imports it from that folder, so it does not need to be installed. This TrackEval version still uses the
NumPy aliases `np.bool`, `np.int`, `np.float` and `np.object`, which NumPy 1.24 removed; `efpmot.eval` adds them back
at import time.

## What this repository does not redistribute

- Images, videos or labels of any dataset, and labels converted from them (frame indices, BDD-format GT, half-split
  GT, the 35k COCO subset).
- Detection caches and tracker outputs computed on the datasets. `results/stats/` holds only the per-sequence
  sufficient statistics of HOTA (true positive, false negative and false positive counts and association sums), which
  are enough to recompute the reported comparisons.
- DSEC ground-truth boxes, in any media.
- VisDrone and BDD100K media or weights.
- HIT-UAV images, videos or labels (the app demo shows frames of the sample videos; see `demo/README.md`).
