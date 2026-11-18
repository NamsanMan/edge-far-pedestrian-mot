# Reproduce

S3x is this repository's edit of YOLOv8n and YOLO11n: four stride changes that make the finest detection grid (P3)
twice as fine horizontally. S3y refines the vertical axis instead and serves as the control. The canvas is the detector
input size, for example 416 x 320. The [README](../README.md) explains these terms in more detail.

There are three levels, from the cheapest to the full pipeline:

1. **Tables from the shipped statistics**: takes minutes on a CPU and needs no dataset.
2. **Evaluate the release weights**: needs one GPU and the DSEC, MOT17 or HIT-UAV data.
3. **Full pipeline per arm**: train, export the detections, track, evaluate and compare. An arm is one detector
   variant, for example YOLOv8n + S3x on the 416 x 320 canvas, trained with several seeds.

All commands run from the repository root.

**Install.**
1. Python 3.10-3.13.
2. Install PyTorch with CUDA from https://pytorch.org first (for example the CUDA 12.x wheel for your driver). On
   Windows, `pip` from PyPI installs a CPU-only build, which is enough for level 1 and the tests but far too slow for training.
3. `pip install -e .` (add `.[dev]` for the FLOPs test, `.[deploy]` for the ONNX export).

Tested with Python 3.10, torch 2.8.0 + CUDA 12.9, ultralytics 8.3.170, boxmot 25.0.0, numpy 2.2.6. Levels 2 and 3 need
the dataset and output folders set in `configs/paths.yaml` ([`DATA.md`](DATA.md)). Below, `<runs>`, `<cache>`,
`<results>` and `<derived>` stand for its output folders `outputs.runs`, `outputs.cache`, `outputs.results` and
`outputs.derived`. The scripts also run without `pip install -e .`, as `python -B scripts/...`, if the dependencies are
installed.

**Tests** run on a CPU and need no dataset; tests that need data or optional packages are skipped:

```bash
python -B -m unittest discover -s tests
python -B apps/thermal_watch/tests/test_thermal_watch.py
```

[`PORT_VERIFICATION.md`](PORT_VERIFICATION.md) records how this code was checked against the original research outputs,
with the measured differences.

## 1. Tables from `results/stats` (no dataset)

The published contrasts are the comparisons of two arms listed in `results/contrasts.yaml` and
`results/contrasts_detectors.yaml`. For every run behind them, `results/stats/<dataset>/<tag>__<tracker>__<class>.npz`
holds the HOTA sufficient statistics of each sequence at each tracker threshold: the counts and association sums from
which HOTA, the standard multi-object tracking score, is recomputed exactly. `<tag>` is the run tag, the name of one
trained model: `dsec_v8n_416s3x_s0`, for example, is YOLOv8n + S3x trained on DSEC at the 416 x 320 canvas with seed 0
(scheme in section 3). The comparison protocol ([`PROTOCOL.md`](PROTOCOL.md)) reads these files directly.

```bash
# one contrast (prints the JSON of the comparison: HOTA A / B, d, CIs, p_AR, TOST, ...)
python -B scripts/compare.py --one dsec v8n 416s3x 416 --cls pedestrian --trk bytetrack
python -B scripts/compare.py --one mot17 v8n 416s3x 416 --cls pedestrian --trk ocsort
python -B scripts/compare.py --one visdrone v8n 640s3x 704x384 --cls pedestrian --trk bytetrack

# every published contrast -> results/summary/*.json and results/RESULTS.md
python -B scripts/compare.py --table results/contrasts.yaml

# the comparison with other detectors on DSEC -> results/summary_detectors/, results/DETECTORS.md
python -B scripts/compare.py --detectors results/contrasts_detectors.yaml
```

`--one DATA DET A B` compares arm A with arm B for one detector on one dataset: `--one dsec v8n 416s3x 416` compares the
runs `dsec_v8n_416s3x_s*` (S3x) with `dsec_v8n_416_s*` (plain); `--cls` and `--trk` pick the class and the tracker.
In its JSON, d is the HOTA of A minus that of B, the CIs are bootstrap confidence intervals of d, p_AR is the p-value of
the paired randomisation test and TOST is the equivalence test ([`PROTOCOL.md`](PROTOCOL.md)).

The bootstrap and the randomisation test use seed 0 and B = 10,000 random draws, so the output is deterministic.

## 2. Evaluate the release weights

The release files on the GitHub Releases page hold plain YOLOv8n and YOLOv8n + S3x, trained with seed 0 on DSEC, on
MOT17 and on each of the two HIT-UAV folds, recA and recB. HIT-UAV is a thermal drone dataset, and its two folds are
recording-disjoint: no recording appears in both the training and the test part of a fold. Each file is a bare fp32
state dict, that is, a `torch.save` of `model.state_dict()` and nothing else. Put them in `release_assets/` and check
them with `sha256sum -c SHA256SUMS.txt`. The model card
[`release_assets/MODEL_CARD.md`](../release_assets/MODEL_CARD.md) is published with them. `efpmot/release.py` maps
each file to its run tag, model family (`--family`) and canvas:

| File | Run tag | Family | Canvas H x W |
|---|---|---|---|
| `yolov8n_plain_dsec_416x320_s0.pt` | `dsec_v8n_416_s0` | `yolov8n` | 320 x 416 |
| `yolov8n_s3x_dsec_416x320_s0.pt` | `dsec_v8n_416s3x_s0` | `yolov8n_s3x` | 320 x 416 |
| `yolov8n_plain_mot17_416x256_s0.pt` | `mot_v8n_416_s0` | `yolov8n` | 256 x 416 |
| `yolov8n_s3x_mot17_416x256_s0.pt` | `mot_v8n_416s3x_s0` | `yolov8n_s3x` | 256 x 416 |
| `yolov8n_plain_hituav-recA_416x320_s0.pt` | `hit_v8n_416recA_s0` | `yolov8n` | 320 x 416 |
| `yolov8n_s3x_hituav-recA_416x320_s0.pt` | `hit_v8n_416s3xrecA_s0` | `yolov8n_s3x` | 320 x 416 |
| `yolov8n_plain_hituav-recB_416x320_s0.pt` | `hit_v8n_416recB_s0` | `yolov8n` | 320 x 416 |
| `yolov8n_s3x_hituav-recB_416x320_s0.pt` | `hit_v8n_416s3xrecB_s0` | `yolov8n_s3x` | 320 x 416 |

```bash
# detections of the evaluation frames -> <cache>/dsec_v8n_416s3x_s0/dsec/*.npz
python -B scripts/export_dets.py --data dsec --name dsec_v8n_416s3x_s0 --weights release_assets/yolov8n_s3x_dsec_416x320_s0.pt
# tracker sweep (ByteTrack 8 thresholds + OC-SORT 5) -> <results>/dsec_trackers/
python -B scripts/track.py --dataset dsec --tag dsec_v8n_416s3x_s0 --tracker all
# TrackEval -> <results>/dsec_eval/test/
python -B scripts/evaluate.py --dataset dsec --tag dsec_v8n_416s3x_s0
# statistics for the protocol -> results/stats/dsec/ (overwrites the shipped file of this run with your own)
python -B scripts/extract_stats.py --eval_root <results> --dataset dsec --tags dsec_v8n_416s3x_s0
```

- `export_dets.py` takes the family and canvas of a release file from `efpmot/release.py`, by file name. Without
  `--weights`, it loads your own training checkpoint `<runs>/<name>/last.pth` (`<name>` is the `--name` run tag) if that
  file exists, and otherwise the release file of that run tag.
- For MOT17, use `--data mot17` (for `export_dets.py`) or `--dataset mot17` (for the other scripts) and the `mot_*`
  tags. First run `python -B scripts/prepare_data.py mot17` once: it writes the ground truth of the evaluation halves
  (each MOT17 training sequence is split into a first half for training and a second half for evaluation).
- The release checkpoints are bit-identical to the trained ones. Detections re-exported on a different GPU can still
  differ from the published ones by FP16 rounding, which can change individual tracker decisions.
- The HIT-UAV release files are scored on person detection only, because HIT-UAV has no track IDs:

  ```bash
  # export val + test, evaluate -> outputs/results/hituav/hit_v8n_416s3xrecA_s0.json (without --out: results/hituav/, the shipped file)
  python -B scripts/evaluate_hituav.py hit_v8n_416s3xrecA_s0 --weights release_assets/yolov8n_s3x_hituav-recA_416x320_s0.pt --out outputs/results/hituav
  # differences to the shipped results
  python -B scripts/evaluate_hituav.py --compare results/hituav --out outputs/results/hituav
  ```

## 3. Full pipeline per arm

The table below lists the arms; each arm is trained once per seed. Run tags, as used in `results/`, have the form
`<prefix>_<det>_<variant>_s<seed>`: dataset prefix, detector, variant (the arm column of the table) and seed.

```bash
python -B scripts/prepare_data.py dsec                    # converted labels -> <derived> (once per dataset you use)
python -B scripts/train.py --family yolov8n_s3x --data dsec --name dsec_v8n_416s3x_s0 --seed 0
#   -> <runs>/dsec_v8n_416s3x_s0/{last.pth, log.txt}; exports the detection cache at the end (unless --no_export)
python -B scripts/track.py --dataset dsec --tag dsec_v8n_416s3x_s0 --tracker all
python -B scripts/evaluate.py --dataset dsec --tag dsec_v8n_416s3x_s0
python -B scripts/extract_stats.py --eval_root <results> --dataset dsec --tags dsec_v8n_416s3x_s0
python -B scripts/compare.py --one dsec v8n 416s3x 416 --cls pedestrian --trk bytetrack --source stats
```

`--hw` and `--epochs` default to the dataset's main canvas and schedule. Give `--hw H W` for the other canvases.

| dataset | arm (variant in the run tag) | `--family` | `--hw` (H W) | epochs | seeds |
|---|---|---|---|---|---|
| dsec | `416` | yolov8n | 320 416 | 6 | 0, 1, 2 |
| dsec | `416s3x` / `416s3y` | yolov8n_s3x / yolov8n_s3y | 320 416 | 6 | 0, 1 |
| dsec | `448x352` (compute-matched uniform) | yolov8n | 352 448 | 6 | 0, 1 |
| dsec | `640` (reference) | yolov8n | 480 640 | 6 | 0, 1, 2 |
| dsec | y11n `416` / `416s3x` | yolo11n / yolo11n_s3x | 320 416 | 6 | 0, 1 |
| mot17 | `416`, `416s3x`, `416s3y` | yolov8n, yolov8n_s3x, yolov8n_s3y | 256 416 | 55 | 0, 1 |
| mot17 | `480x256` (compute-matched) | yolov8n | 256 480 | 55 | 0, 1 |
| visdrone | `640`, `640s3x`, `640s3y` | yolov8n, yolov8n_s3x, yolov8n_s3y | 384 640 | 6 | 0, 1 |
| visdrone | `704x384` (compute-matched) | yolov8n | 384 704 | 6 | 0, 1 |
| bdd | v8n `416`, `416s3x` | yolov8n, yolov8n_s3x | 256 416 | 6 | 0, 1, 2 / 0, 1 |
| bdd | v8n `480x256` | yolov8n | 256 480 | 6 | 0, 1 |
| bdd | v8n `640` (reference) | yolov8n | 384 640 | 6 | 0, 1, 2 |
| bdd | y11n `416`, `416s3x`, `480x256` | yolo11n, yolo11n_s3x, yolo11n | 256 416 / 256 416 / 256 480 | 6 | 0, 1 |

- The tag prefixes are `dsec_`, `mot_`, `vis_` (VisDrone) and `bdd_` (BDD100K); the detector is `v8n` (YOLOv8n) or
  `y11n` (YOLO11n). For example, `mot_v8n_480x256_s1` is plain YOLOv8n on MOT17 at 480 x 256 with seed 1, and
  `bdd_y11n_416s3x_s0` is YOLO11n + S3x on BDD100K at 416 x 256 with seed 0.
- The variants `448x352`, `480x256` and `704x384` (compute-matched uniform) put the plain model on a larger input with
  about the same FLOPs as S3x. The variant `640` on DSEC and BDD100K (reference) puts it on a 640-wide canvas, with
  about 2.1x the FLOPs of S3x.
- Every arm is trained with batch size 16, 4 data-loader workers, learning rate 0.000833 and weight decay 5e-4. DSEC
  uses every 2nd labelled frame for training; the other datasets use every frame.
- To smoke-test the training loop without writing a checkpoint or an export, run
  `python -B scripts/train.py --family yolov8n_s3x --data mot17 --name smoke --max_iters 30`.

## HIT-UAV, the Thermal Watch app and Jetson latency

HIT-UAV is used for person detection on thermal drone images ([`DATA.md`](DATA.md)). The commands below train and
evaluate one run, YOLOv8n + S3x on fold recA with seed 0:

```bash
python -B scripts/prepare_data.py hituav      # optional: rebuilds the shipped efpmot/data/splits/hituav_split.json and checks it
python -B scripts/train.py --family yolov8n_s3x --data hituav --split recA --name hit_v8n_416s3xrecA_s0 --seed 0
python -B scripts/evaluate_hituav.py hit_v8n_416s3xrecA_s0 --out outputs/results/hituav
python -B scripts/evaluate_hituav.py --table  # results/HITUAV.md from the shipped results/hituav/*.json
```

- Run tags are `hit_v8n_<variant><fold>_s<seed>`. Variants: `416` (yolov8n), `416s3x` (yolov8n_s3x), `416s3y`
  (yolov8n_s3y), `448x352` (yolov8n, `--hw 352 448`), `640` (yolov8n, `--hw 512 640`), `640s3x` (yolov8n_s3x,
  `--hw 512 640`, official split only). The folds are `recA` and `recB` (recording-disjoint, see section 2) and
  `altLH`, which trains at 60-90 m altitude and tests at 100-130 m. A tag without a fold marker uses the official split
  and is trained without `--split`. A tag whose fold marker does not match `--split` is rejected.
- The default number of epochs gives every split the same budget of ~9,060 iterations: official 72, recA 116, recB 118,
  altLH 112. [`results/HITUAV.md`](../results/HITUAV.md) lists the seeds of each arm.
- Thermal Watch is the person-search app; [`apps/thermal_watch/README.md`](../apps/thermal_watch/README.md) describes
  its live and review screens, the demo video and the tests. It needs the HIT-UAV release files, or training checkpoints
  given with `--family`, `--hw` and `--thr` (the person score threshold). The demo also needs the five HIT-UAV sample
  videos, whose folder is set by `data.hituav_videos` in `configs/paths.yaml`.
- Jetson Orin Nano latency: [`deploy/jetson/README.md`](../deploy/jetson/README.md). Export the detectors to ONNX on the
  PC, run `bench.sh` on the Jetson to build and time the TensorRT engines, then run `parse_results.py` back on the PC.

## Determinism

- Training seeds torch, NumPy and Python's `random` with `--seed`. cuDNN and the data-loader workers are not made
  deterministic, so a retrained model differs from the released one in its weights. The published numbers come from the
  runs behind the release files and `results/stats`.
- Tracking, TrackEval and the protocol are deterministic given the detection caches.
