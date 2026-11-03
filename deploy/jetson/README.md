# Jetson Orin Nano latency

Export the detectors to ONNX on the PC, build and time TensorRT engines on a Jetson, and turn the log into a table.
The measured table is [`results/latency_orin_nano.md`](results/latency_orin_nano.md).

## Measured (Jetson Orin Nano, 15 W power mode, TensorRT 10.3)

GPU compute time of the detector engine (`trtexec`, batch 1, raw head output before NMS; median of the 10 s run,
as everywhere in this repository; the CSV also keeps min, max and mean):

| model | canvas (W x H) | FP16 median [ms] | INT8 median [ms] |
|---|---|---|---|
| YOLOv8n | 416 x 320 | 2.94 (repeat 2.89) | 2.34 |
| YOLOv8n + S3x | 416 x 320 | 3.26 (repeat 3.27) | 2.71 |
| YOLOv8n + S3y | 416 x 320 | 3.27 | - |
| YOLOv8n, compute-matched uniform | 448 x 352 | 3.13 | - |
| YOLOv8n, uniform reference | 640 x 480 | 4.72 | - |

- S3x costs +11 % latency on the 416 x 320 canvas (2.94 -> 3.26 ms FP16), the same as its +11 % FLOPs. The repeated
  pass at the end of the session gives 2.89 vs 3.27 ms.
- The 640 x 480 reference costs +61 % (4.72 ms).
- The compute-matched 448 x 352 uniform canvas has 7 % more FLOPs than S3x but is slightly faster on the device
  (3.13 vs 3.26 ms): FLOPs do not translate one to one into latency.
- For comparison, three other ways to refine P3 were timed in the same session on the same canvas. Their code is not
  in this repository (see [`docs/EXPERIMENTS_LOG.md`](../../docs/EXPERIMENTS_LOG.md)):
  - an isotropic P2 level (stride-4 head): 4.51 ms;
  - a shallow MatrixNets-style x-fine branch with the same FLOPs as S3x: 3.55 ms;
  - two half-width P3 branches (x-fine and y-fine): 4.14 ms, although they add only +3 % FLOPs.
- BDD100K-sized canvases (416 x 256, 640 x 384) and graphs with the letterbox inside the engine (`native`) are in the
  full table.

**What these numbers are not.**
- They time the engine alone. CPU pre-processing, NMS, tracking and the zone logic are not included. The end-to-end
  camera -> TensorRT -> tracker -> alert pipeline has not been measured on the Jetson.
- The INT8 engines were built with `--int8 --fp16` and no calibration set. They give the latency of an INT8 engine
  only; no INT8 accuracy is claimed.

## Reproduce

1. **Export on the PC** (CPU only; needs `onnx` and `onnxruntime`, `pip install -e .[deploy]`):

   ```bash
   # one graph
   python -B deploy/jetson/export_onnx.py --weights release_assets/yolov8n_s3x_dsec_416x320_s0.pt --family yolov8n_s3x \
       --hw 320 416 --out deploy/jetson/export/v8n_416_s3x_canvas.onnx
   # add --native 480 640 to put the letterbox of a 640 x 480 frame inside the graph
   # all graphs of the table (checkpoints from outputs.runs or release_assets/; missing ones are skipped)
   python -B deploy/jetson/export_onnx.py --all
   ```

   - Input `images`: BGR 0..255, float, `[1, 3, H, W]`. Output `output0`: `[1, 12, A]` = box centre x, y, width,
     height in canvas pixels and the 8 class scores, for all A anchors. Opset 17, no NMS.
   - The script runs each file once with onnxruntime (CPU) on a random input and prints `max|ort-torch|` (about 2e-4 on
     box coordinates in pixels).
   - With `--all`, each graph is identical node for node to the one benchmarked (including whether the anchor grid is
     a constant or computed in the graph; the outputs are the same either way).
   - Latency depends only on the graph, not on the trained weights. The DSEC release checkpoints cover the 416 x 320
     rows (plain, S3x, S3y). The other rows need checkpoints trained with `scripts/train.py`; any checkpoint of the
     same family and canvas gives the same engine timing.

2. **Run on the Jetson.**
   - Copy `bench.sh` and the `.onnx` files to a folder on the device, then run there:

     ```bash
     bash bench.sh -p fp16 -r v8n_416_uniform_canvas.onnx,v8n_416_s3x_canvas.onnx *.onnx
     bash bench.sh -p int8 v8n_416_uniform_canvas.onnx v8n_416_s3x_canvas.onnx
     ```

   - Copy the `*.log` files back to `deploy/jetson/results/raw/` (git-ignored) on the PC.

   - `bench.sh` options:
     - `-p fp16,int8`: the precisions to time (default both).
     - `-r A,B`: models timed once more in FP16 at the end, interleaved, as a drift check.
     - `-w`: wait until the GPU load (tegrastats `GR3D_FREQ`) is below 20 % before each build.
     - `-t PATH`: the trtexec binary.
   - The script does not change the power mode, the clocks or any package. Record the power mode you use; the table
     above is the 15 W mode, and `bench.sh` writes the current mode to `bench.log`.
   - Each engine is built, timed (`--warmUp=500 --duration=10 --useSpinWait`) and deleted; the trtexec log is kept. A
     build takes several minutes on an Orin Nano.

3. **Parse on the PC.**

   ```bash
   python -B deploy/jetson/parse_results.py deploy/jetson/results/raw/bench.log
   ```

   This writes `results/latency_orin_nano.csv` and `.md`, the only source of the latency numbers in the repository
   (`scripts/compare.py --table` reads the CSV for `results/RESULTS.md`). Without an argument it parses the stored log
   `results/bench_2026-10-27.log`: the RESULT lines of the measurement session behind the table above, without the
   rows of input-warping experiments that are not part of this repository.

## Files

| file | what |
|---|---|
| `export_onnx.py` | ONNX export (canvas or native graph) and the onnxruntime check |
| `bench.sh` | trtexec benchmark, runs on the Jetson |
| `parse_results.py` | `bench.log` -> `results/latency_orin_nano.{csv,md}` |
| `results/bench_2026-10-27.log` | RESULT lines of the measurement session |
