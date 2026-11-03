# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""efpmot: edge multi-object tracking of far, small pedestrians.

Subpackages: models (S3x / S3y YOLOv8n and YOLO11n), data (DSEC-Det, MOT17, VisDrone2019-MOT, BDD100K), train (training
loop + detection export), track (ByteTrack / OC-SORT on cached detections), eval (TrackEval + the cross-fitted paired
comparison protocol). Paths: efpmot.paths (configs/paths.yaml).
"""
__version__ = "0.1.0"

# dataset key -> run-tag prefix (run tags: <prefix>_<det>_<variant>_s<seed>, e.g. dsec_v8n_416s3x_s0, mot_v8n_416s3x_s0)
TAG_PREFIX = {"dsec": "dsec", "mot17": "mot", "visdrone": "vis", "bdd": "bdd"}
DATASETS = tuple(TAG_PREFIX)


def dataset_of_tag(tag: str) -> str:
    """run tag -> dataset key (inverse of TAG_PREFIX)"""
    for data, pre in TAG_PREFIX.items():
        if tag.startswith(pre + "_"):
            return data
    raise ValueError(f"run tag {tag!r} has no known dataset prefix {sorted(TAG_PREFIX.values())}")
