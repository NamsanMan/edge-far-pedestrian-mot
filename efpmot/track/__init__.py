# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""efpmot.track: ByteTrack / OC-SORT (BoxMOT) on cached detections and TrackEval-format writers.

    from efpmot import track
    track.run("dsec", "dsec_v8n_416s3x_s0", "bytetrack", 0.5)    # -> <results>/dsec_trackers/dsec_v8n_416s3x_s0__bytetrack_t0.5
    track.sweep("dsec", "dsec_v8n_416s3x_s0")                     # the whole threshold grid of both trackers
"""
from .runner import GRID, NSEQ_TRACKED, make_tracker, run, sweep, track_sequence, tracker_name

__all__ = ["GRID", "NSEQ_TRACKED", "make_tracker", "run", "sweep", "track_sequence", "tracker_name"]
