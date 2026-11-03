# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""efpmot.eval: TrackEval runner, per-run stats files and the cross-fitted paired comparison protocol.

    from efpmot import eval as E
    E.evaluate_tag("dsec", "dsec_v8n_416s3x_s0")                              # TrackEval on the whole threshold grid
    E.one("dsec", "v8n", "416s3x", "416", cls="pedestrian", trk="bytetrack")  # one contrast (protocol.one)
"""
from .protocol import (EVAL_DIR, GRID, MARGINS, NSEQ, apply_rule, clusters_for, compare, folds_for, holm, hota, label,
                       load_run, method, one, read_detailed, static_split, tag_of)
from .trackeval_runner import SUBSET, evaluate, evaluate_tag

__all__ = ["EVAL_DIR", "GRID", "MARGINS", "NSEQ", "SUBSET", "apply_rule", "clusters_for", "compare", "evaluate",
           "evaluate_tag", "folds_for", "holm", "hota", "label", "load_run", "method", "one", "read_detailed",
           "static_split", "tag_of"]
