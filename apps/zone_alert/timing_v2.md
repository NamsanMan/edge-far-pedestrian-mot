# Zone alert app: timing (v1 vs v2 zone files)

`bench.py --v2`, 3 runs per row (interleaved), GPU NVIDIA GeForce RTX 5060 Ti. Median (min-max) over the runs; first 10 frames of each run excluded. Pipeline = detect + track + alert logic; end-to-end adds JPEG decode, drawing and H.264 encoding. v2 = APPROACH on every zone, direction filters and one tripwire (zones/<seq>_v2.json).

| sequence | model | zones | detect [ms] | track [ms] | alert logic [ms] | pipeline FPS | end-to-end FPS |
|---|---|---|---|---|---|---|---|
| MOT17-04 | plain | v1 | 11.0 | 5.8 | 0.23 | 59 (55-62) | 38 (35-40) |
| MOT17-04 | plain | v2 | 10.6 | 5.7 | 1.15 | 57 (53-58) | 37 (35-37) |
| MOT17-04 | s3x | v1 | 10.5 | 5.6 | 0.22 | 61 (58-61) | 39 (38-40) |
| MOT17-04 | s3x | v2 | 10.6 | 5.7 | 1.16 | 58 (57-58) | 37 (37-38) |
| MOT17-09 | plain | v1 | 10.3 | 1.8 | 0.07 | 82 (69-83) | 49 (43-49) |
| MOT17-09 | plain | v2 | 10.6 | 1.9 | 0.27 | 78 (78-82) | 47 (47-48) |
| MOT17-09 | s3x | v1 | 10.5 | 1.8 | 0.06 | 81 (81-82) | 49 (46-49) |
| MOT17-09 | s3x | v2 | 10.8 | 1.9 | 0.27 | 77 (74-82) | 45 (45-48) |
