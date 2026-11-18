# Jetson Orin Nano latency

Export the detectors to ONNX on the PC, build and time TensorRT engines on a Jetson, and turn the log into a table.
The measured table is [`results/latency_orin_nano.md`](results/latency_orin_nano.md).

## Measured (Jetson Orin Nano, 15 W power mode, TensorRT 10.3)

The canvas is the detector input size; every frame is downscaled and letterboxed to it. YOLOv8n is the unmodified
("plain") model. S3x changes four strides inside YOLOv8n so that its finest prediction grid (P3) has points every 4 px
horizontally instead of every 8 px, for about +11 % FLOPs. S3y, which refines the vertical axis, serves as a control.

Each value is the GPU compute time of one detector engine, timed with `trtexec` at batch 1 up to the raw head output,
before NMS. It is the median of a 10 s run, as everywhere in this repository; the CSV also keeps the min, max and mean.
FPS = 1000 / median, the ceiling set by the engine alone, not the frame rate of a whole pipeline:

| model | canvas (W x H) | FP16 median [ms] | FP16 FPS | INT8 median [ms] | INT8 FPS |
|---|---|---|---|---|---|
| YOLOv8n | 416 x 320 | 2.94 (repeat 2.89) | 340 | 2.34 | 427 |
| YOLOv8n + S3x | 416 x 320 | 3.26 (repeat 3.27) | 307 | 2.71 | 369 |
| YOLOv8n + S3y | 416 x 320 | 3.27 | 306 | - | - |
| YOLOv8n, larger canvas (FLOPs ≈ S3x) | 448 x 352 | 3.13 | 319 | - | - |
| YOLOv8n, larger canvas (reference) | 640 x 480 | 4.72 | 212 | - | - |

- S3x costs +11 % latency on the 416 x 320 canvas (2.94 -> 3.26 ms FP16, 340 -> 307 FPS), the same as its +11 %
  FLOPs. The repeated pass at the end of the session, a check for run-to-run drift (`bench.sh -r` below), gives
  2.89 vs 3.27 ms (346 vs 306 FPS).
- On the 640 x 480 reference canvas, the full DSEC frame without downscaling, plain YOLOv8n costs +61 % over the
  416 x 320 canvas (4.72 ms, 212 FPS).
- Plain YOLOv8n on the 448 x 352 canvas, the larger input of about the same FLOPs as S3x, is slightly faster on the
  device than S3x (3.13 vs 3.26 ms, 319 vs 307 FPS) although it has 7 % more FLOPs: FLOPs do not translate one to one
  into latency.
- For comparison, three other ways to get a finer grid than the plain P3 were timed in the same session on the same
  canvas. Their code is not in this repository (see [`docs/EXPERIMENTS_LOG.md`](../../docs/EXPERIMENTS_LOG.md)).
  Here x-fine means finer horizontally and y-fine finer vertically.
  - an isotropic P2 level (an extra stride-4 head): 4.51 ms (222 FPS);
  - a shallow MatrixNets-style x-fine branch with the same FLOPs as S3x: 3.55 ms (282 FPS);
  - two half-width P3 branches, one x-fine and one y-fine: 4.14 ms (242 FPS), although they add only +3 % FLOPs.
- The full table also has BDD100K-sized canvases (416 x 256, 640 x 384) and `native` graphs, which take the camera
  frame and do the letterbox resize inside the engine.

**What these numbers are not.**
- They time the engine alone. CPU pre-processing, NMS and tracking are not included, and the
  whole pipeline has not been timed on the Jetson.
- The INT8 engines were built with `--int8 --fp16` and no calibration set. They give the latency of an INT8 engine
  only; no INT8 accuracy is claimed.

## Reproduce

1. **Export on the PC.** This runs on the CPU and needs `onnx` and `onnxruntime` (`pip install -e .[deploy]`):

   ```bash
   # one graph
   python -B deploy/jetson/export_onnx.py --weights release_assets/yolov8n_s3x_dsec_416x320_s0.pt --family yolov8n_s3x \
       --hw 320 416 --out deploy/jetson/export/v8n_416_s3x_canvas.onnx
   # add --native 480 640 to put the letterbox of a 640 x 480 frame inside the graph
   # all graphs of the table (checkpoints from outputs.runs or release_assets/; missing ones are skipped)
   python -B deploy/jetson/export_onnx.py --all
   ```

   - `--family` is the model family, as in `efpmot.models.build`; `--hw` gives the canvas as height and width (`320 416`
     for the 416 x 320 canvas of the table).
   - The graph's input `images` is a float tensor `[1, 3, H, W]`, BGR, 0..255. Its output `output0` is `[1, 12, A]`:
     for each of the A anchors (the prediction points of the three detection levels), the box centre x, y, width and
     height in canvas pixels and the 8 class scores. Opset 17, no NMS.
   - As a check, the script runs each exported file once with onnxruntime (CPU) on a random input and prints the
     largest absolute difference from the PyTorch output, `max|ort-torch|` (about 2e-4 on box coordinates in pixels).
   - With `--all`, each graph is identical node for node to the one that was benchmarked, down to whether the anchor
     grid is a constant or computed in the graph. The outputs are the same either way.
   - Latency depends only on the graph, not on the trained weights. The DSEC release checkpoints cover the plain and
     S3x rows at 416 x 320. The other rows, S3y included, need checkpoints trained with `scripts/train.py`; any checkpoint of the
     same family and canvas gives the same engine timing.

2. **Run on the Jetson.**
   - Copy `bench.sh` and the `.onnx` files to a folder on the device, then run there:

     ```bash
     bash bench.sh -p fp16 -r v8n_416_uniform_canvas.onnx,v8n_416_s3x_canvas.onnx *.onnx
     bash bench.sh -p int8 v8n_416_uniform_canvas.onnx v8n_416_s3x_canvas.onnx
     ```

   - The first command times every graph in FP16, then times plain YOLOv8n and S3x once more as a drift check
     (`uniform` in a file name marks plain YOLOv8n). The second command times the same two graphs in INT8.

   - Copy the `*.log` files back to `deploy/jetson/results/raw/` (git-ignored) on the PC.

   - `bench.sh` options:
     - `-p fp16,int8`: the precisions to time (default both).
     - `-r A,B`: models timed once more in FP16 at the end of the session, interleaved, to check for run-to-run drift.
     - `-w`: wait until the GPU load (tegrastats `GR3D_FREQ`) is below 20 % before each build.
     - `-t PATH`: path of the trtexec binary (default `/usr/src/tensorrt/bin/trtexec`).
   - The script does not change the power mode, the clocks or any package. Record the power mode you use: the table
     above was measured in the 15 W mode, and `bench.sh` writes the current mode to `bench.log`.
   - Each engine is built, timed for 10 s after a warm-up (`--warmUp=500 --duration=10 --useSpinWait`) and deleted; the
     trtexec log is kept. A build takes several minutes on an Orin Nano.

3. **Parse on the PC.**

   ```bash
   python -B deploy/jetson/parse_results.py deploy/jetson/results/raw/bench.log
   ```

   This writes `results/latency_orin_nano.csv` and `.md`, the only source of the latency numbers in the repository;
   `scripts/compare.py --table` reads the CSV for `results/RESULTS.md`. Without an argument, the script parses the two
   logs stored in `results/`. `results/bench_2026-10-27.log` holds the RESULT lines (one per timed engine) of the
   session behind the table above, minus the rows of input-warping experiments that are not part of this repository.
   `results/bench_2026-11-13.log` is the session that timed the other detectors in
   [`results/DETECTORS.md`](../../results/DETECTORS.md); it is parsed into `results/latency_orin_nano_detectors.csv`,
   which `scripts/compare.py --detectors` reads.

## Files

| file | what |
|---|---|
| `export_onnx.py` | ONNX export (canvas or native graph) and the onnxruntime check |
| `bench.sh` | trtexec benchmark, runs on the Jetson |
| `parse_results.py` | `bench.log` -> `results/latency_orin_nano.{csv,md}` (+ `latency_orin_nano_detectors.csv`) |
| `results/bench_2026-10-27.log` | RESULT lines of the measurement session (YOLOv8n, S3x, S3y, canvases) |
| `results/bench_2026-11-13.log` | RESULT lines of the second session (other detectors, YOLOv8n and S3x again) |
