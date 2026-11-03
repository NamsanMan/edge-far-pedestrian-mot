# Zone alert app: timing

`bench.py`, 3 runs per row (interleaved), GPU NVIDIA GeForce RTX 5060 Ti. Median (min-max) over the runs; first 10 frames of each run excluded. Pipeline = detect + track + alert logic; end-to-end adds JPEG decode, drawing and H.264 encoding.

| sequence | model | detect [ms] | track [ms] | alert logic [ms] | pipeline FPS | end-to-end FPS |
|---|---|---|---|---|---|---|
| MOT17-04 | plain | 11.5 | 6.0 | 0.21 | 56 (56-60) | 36 (36-37) |
| MOT17-04 | s3x | 11.3 | 5.8 | 0.20 | 58 (55-60) | 36 (36-38) |
| MOT17-09 | plain | 11.1 | 2.0 | 0.06 | 76 (71-78) | 45 (44-46) |
| MOT17-09 | s3x | 11.3 | 1.9 | 0.06 | 75 (68-78) | 44 (42-45) |
