# Reproduce

Three levels, from cheapest to full:

1. **Tables from the shipped statistics**: CPU, minutes, no dataset.
2. **Evaluate the release weights**: one GPU, the DSEC or MOT17 data.
3. **Full pipeline per arm**: train, export detections, track, evaluate, compare.

All commands run from the repository root.

**Install.**
1. Python 3.10-3.13.
2. Install PyTorch with CUDA from https://pytorch.org first (for example the CUDA 12.x wheel for your driver). On
   Windows, `pip` from PyPI installs a CPU-only build, which is enough for level 1 and the tests but far too slow for training.
3. `pip install -e .` (add `.[dev]` for the FLOPs test, `.[deploy]` for the ONNX export).

Tested with Python 3.10, torch 2.8.0 + CUDA 12.9, ultralytics 8.3.170, boxmot 25.0.0, numpy 2.2.6. Configure the paths
for levels 2 and 3 ([`DATA.md`](DATA.md)). The scripts also run without the install, as `python -B scripts/...`.

**Tests** (CPU, no dataset; tests that need data or optional packages are skipped):

```bash
python -B -m unittest discover -s tests
python -B apps/zone_alert/tests/test_zone_monitor.py
```

How this code was checked against the original research outputs (exact commands and measured differences):
[`PORT_VERIFICATION.md`](PORT_VERIFICATION.md).

## 1. Tables from `results/stats` (no dataset)

`results/stats/<dataset>/<tag>__<tracker>__<class>.npz` holds the per-sequence HOTA sufficient statistics of every
threshold of every run used in the published contrasts. The comparison protocol reads them directly.

```bash
# one contrast (prints the JSON of the comparison: HOTA A / B, d, CIs, p_AR, TOST, ...)
python -B scripts/compare.py --one dsec v8n 416s3x 416 --cls pedestrian --trk bytetrack
python -B scripts/compare.py --one mot17 v8n 416s3x 416 --cls pedestrian --trk ocsort
python -B scripts/compare.py --one visdrone v8n 640s3x 704x384 --cls pedestrian --trk bytetrack

# every published contrast -> results/summary/*.json and results/RESULTS.md
python -B scripts/compare.py --table results/contrasts.yaml
```

The bootstrap and the randomisation use seed 0 and B = 10,000, so the output is deterministic.

## 2. Evaluate the release weights

The release files (GitHub Releases; DSEC and MOT17, seed 0, plain YOLOv8n and YOLOv8n + S3x) are bare fp32 state dicts.
Put them in `release_assets/` and check them with `sha256sum -c SHA256SUMS.txt`. [`release_assets/MODEL_CARD.md`](../release_assets/MODEL_CARD.md)
is published with them; `efpmot/release.py` maps each file to its run tag:

| File | Run tag | Family | Canvas H x W |
|---|---|---|---|
| `yolov8n_plain_dsec_416x320_s0.pt` | `dsec_v8n_416_s0` | `yolov8n` | 320 x 416 |
| `yolov8n_s3x_dsec_416x320_s0.pt` | `dsec_v8n_416s3x_s0` | `yolov8n_s3x` | 320 x 416 |
| `yolov8n_plain_mot17_416x256_s0.pt` | `mot_v8n_416_s0` | `yolov8n` | 256 x 416 |
| `yolov8n_s3x_mot17_416x256_s0.pt` | `mot_v8n_416s3x_s0` | `yolov8n_s3x` | 256 x 416 |

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

- The family and canvas of a release file come from `efpmot/release.py` (by file name); without `--weights`,
  `export_dets.py` takes `<runs>/<name>/last.pth` and else the release file of that run tag.
- For MOT17, use `--data mot17` / `--dataset mot17` and the `mot_*` tags. Run `python -B scripts/prepare_data.py mot17`
  once first, for the half-split GT.
- The release checkpoints are bit-identical to the trained ones. Re-exported detections can differ from the published
  ones by FP16 rounding on a different GPU, which can move single tracker decisions.

## 3. Full pipeline per arm

One arm = one row below, trained once per seed. Run tags are `<prefix>_<det>_<variant>_s<seed>`, as used in
`results/`.

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

| dataset | arm (tag variant) | `--family` | `--hw` (H W) | epochs | seeds |
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

- Tag prefixes are `dsec_`, `mot_`, `vis_` and `bdd_`. The detector is `v8n` or `y11n`. Examples: `mot_v8n_480x256_s1`,
  `bdd_y11n_416s3x_s0`.
- All arms: batch 16, 4 data workers, lr 0.000833, weight decay 5e-4. DSEC uses every 2nd labelled frame for training;
  the other datasets use every frame.
- A smoke test of the training loop, which writes no checkpoint and no export:
  `python -B scripts/train.py --family yolov8n_s3x --data mot17 --name smoke --max_iters 30`.

## Jetson latency and the zone-alert app

- Jetson Orin Nano latency: [`deploy/jetson/README.md`](../deploy/jetson/README.md). Export on the PC, `bench.sh` on
  the device, `parse_results.py` back on the PC.
- Zone intrusion / loitering app and its MOT17 evaluation:
  [`apps/zone_alert/README.md`](../apps/zone_alert/README.md). It needs the two seed-0 MOT17 runs (`mot_v8n_416_s0`,
  `mot_v8n_416s3x_s0`) tracked by step 2 or 3.

## Determinism

- Training seeds torch, NumPy and Python's `random` with `--seed`. cuDNN and the data-loader workers are not made
  deterministic, so a retrained model differs from the released one in its weights. The published numbers come from the
  runs behind the release files and `results/stats`.
- Tracking, TrackEval and the protocol are deterministic given the detection caches.
