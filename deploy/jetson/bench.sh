#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
#
# TensorRT latency of ONNX detectors on a Jetson (run ON the device, in the folder that holds the .onnx files).
#
# For every model and precision: build an engine with trtexec, time it (--warmUp=500 --duration=10 --useSpinWait),
# keep the trtexec log as <model>_<precision>.log, append one line to bench.log and delete the engine. The device
# configuration (power mode, clocks, packages) is never changed; the current power mode is written to bench.log.
#
#   RESULT <model>.onnx <precision> GPU Compute Time: min = .. ms, max = .. ms, mean = .. ms, median = .. ms
#
# usage: bench.sh [-p fp16,int8] [-r MODEL[,MODEL...]] [-w] [-t TRTEXEC] MODEL.onnx [MODEL.onnx ...]
#   -p  precisions to time (default fp16,int8). int8 = --int8 --fp16 without a calibration set: latency only.
#   -r  models (file names) timed once more in FP16 at the end, interleaved, as a run-to-run drift check
#   -w  before each build, wait until the GPU load reported by tegrastats (GR3D_FREQ) is below 20 %
#       (checked once a minute, at most 60 minutes)
#   -t  trtexec binary (default /usr/src/tensorrt/bin/trtexec)
# Parse the log on the PC with deploy/jetson/parse_results.py.
set -u
TRT=/usr/src/tensorrt/bin/trtexec
PRECISIONS=fp16,int8
REPEAT=""
WAIT=0
while getopts "p:r:wt:" opt; do
  case $opt in
    p) PRECISIONS=$OPTARG ;;
    r) REPEAT=$OPTARG ;;
    w) WAIT=1 ;;
    t) TRT=$OPTARG ;;
    *) sed -n '5,20p' "$0"; exit 1 ;;
  esac
done
shift $((OPTIND - 1))
[ $# -ge 1 ] || { sed -n '5,20p' "$0"; exit 1; }
[ -x "$TRT" ] || { echo "trtexec not found: $TRT (use -t)"; exit 1; }

log() { echo "$(date '+%F %T') $*" | tee -a bench.log; }

gpu_busy() {  # 1 when tegrastats reports GR3D_FREQ above 20 %, else 0
  timeout 3 tegrastats --interval 1000 | head -2 | tail -1 | grep -o 'GR3D_FREQ [0-9]*' | awk '{print ($2 > 20)}'
}

wait_idle() {
  [ "$WAIT" = 1 ] || return 0
  for _ in $(seq 1 60); do
    [ "$(gpu_busy)" = "0" ] && return 0
    log "GPU load above 20 %, waiting"
    sleep 60
  done
  log "GPU load still above 20 % after 60 min, stop"
  exit 2
}

run() {  # $1 model.onnx  $2 fp16|int8
  local m=$1 p=$2 e=${1%.onnx}_$2.engine flags="--fp16"
  [ -f "$m" ] || { log "RESULT $m $p MISSING"; return; }
  [ "$p" = "int8" ] && flags="--int8 --fp16"
  wait_idle
  "$TRT" --onnx="$m" --saveEngine="$e" $flags --warmUp=500 --duration=10 --useSpinWait > "${m%.onnx}_$p.log" 2>&1
  local r
  r=$(grep -o "GPU Compute Time: min = [0-9.]* ms, max = [0-9.]* ms, mean = [0-9.]* ms, median = [0-9.]* ms" "${m%.onnx}_$p.log")
  log "RESULT $m $p ${r:-FAILED}"
  rm -f "$e"
}

log "start; power mode: $(nvpmodel -q 2>/dev/null | head -1)"
for p in ${PRECISIONS//,/ }; do
  for m in "$@"; do run "$m" "$p"; done
done
for m in ${REPEAT//,/ }; do run "$m" fp16; done
log "BENCH_DONE"
