# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""results/HITUAV.md from the per-run files results/hituav/<tag>.json (efpmot.eval.detection, scripts/evaluate_hituav.py): seed
means per arm on the recording-disjoint folds (recA + recB = the mean of the two fold means), the altitude shift (altLH) and the
official split, and the released HIT-UAV weights with their operating thresholds. Deterministic: the same files give the same page.
"""
from __future__ import annotations

import glob
import json
import os
import re

from ..data import hituav as H
from ..release import RELEASES

# GFLOPs (2 x thop MACs, 8 classes) at the canvas, from efpmot.models.flops; the P2 model is not in this repository
GFLOPS = {"plain 416x320": 2.665, "S3x 416x320": 2.955, "S3y 416x320": 2.955, "uniform 448x352": 3.157, "plain 640x512": 6.559,
          "S3x 640x512": 7.273}
MODULE = {"yolov8n": "plain", "yolov8n_s3x": "S3x", "yolov8n_s3y": "S3y", "yolov8n_p2": "P2"}
MOD_RANK = {"plain": 0, "uniform": 0, "S3x": 1, "S3y": 2, "P2": 3}
BASE_ARM = "plain 416x320"
N_TRAIN_OFFICIAL = 2029   # images of the official train folder (the folds' counts come from the shipped split)


def arm_of(tag: str, family: str, hw) -> str:
    """'plain 416x320', 'S3x 416x320', 'uniform 448x352' (a plain run whose tag names the canvas WxH), 'P2 416x320', ..."""
    var = re.sub(r"_s\d+$", "", tag.split("_", 2)[-1])
    var = re.sub(r"(rec[AB]|altLH)$", "", var)
    canvas = f"{int(hw[1])}x{int(hw[0])}"
    mod = MODULE.get(family, family)
    if mod == "plain" and re.fullmatch(r"\d+x\d+", var):
        mod = "uniform"
    return f"{mod} {canvas}"


def _arm_key(arm: str):
    mod, canvas = arm.split(" ")
    w, h = (int(v) for v in canvas.split("x"))
    return (w * h, MOD_RANK.get(mod, 9), arm)


def values(r: dict) -> dict:
    """the page's numbers of one run: test person AP50, AP50:95, threshold, F1 / P / R at it, recall < 8 px, test car AP50"""
    ap, op = r["ap"]["test"], r["op"]["pedestrian"]
    ss = op["test_recall"]["short_side"]
    n8 = ss["<6"]["n"] + ss["6-8"]["n"]
    hits = sum(round(ss[k]["recall"] * ss[k]["n"] / 100) for k in ("<6", "6-8") if ss[k]["n"])
    return dict(ap50=ap["pedestrian"]["AP50"], ap5095=ap["pedestrian"]["AP50_95"], thr=op["thr"], f1=op["test"]["F1"], p=op["test"]["P"],
                r=op["test"]["R"], r8=100.0 * hits / n8 if n8 else None, car50=ap["car"]["AP50"])


def load_results(results_dir: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(os.path.join(results_dir, "hit_*.json"))):
        with open(f, encoding="utf-8") as fh:
            out.append(json.load(fh))
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _f(v, d=1, sign=False):
    return "-" if v is None else (f"{v:+.{d}f}" if sign else f"{v:.{d}f}")


def _group(results, split):
    """{arm: [results of that arm on split, by seed]}"""
    g: dict = {}
    for r in results:
        if r["split"] == split:
            g.setdefault(r["arm"], []).append(r)
    return {a: sorted(rs, key=lambda r: r["seed"]) for a, rs in sorted(g.items(), key=lambda kv: _arm_key(kv[0]))}


def _arm_cell(arm, rs):
    return f"{arm} (code not included)" if any(not r.get("code_included", True) for r in rs) else arm


def _gflops(arm):
    return f"{GFLOPS[arm]:.3f}" if arm in GFLOPS else "-"


def _fold_table(results) -> list[str]:
    A, B = _group(results, "recA"), _group(results, "recB")
    arms = sorted(set(A) & set(B), key=_arm_key)
    rows, means = [], {}
    for arm in arms:
        va, vb = [values(r) for r in A[arm]], [values(r) for r in B[arm]]
        means[arm] = {k: _mean([_mean([v[k] for v in va]), _mean([v[k] for v in vb])]) for k in va[0]}
    base = means.get(BASE_ARM, {}).get("ap50")
    for arm in arms:
        m = means[arm]
        seeds = "; ".join(" ".join(_f(values(r)["ap50"]) for r in rs) for rs in (A[arm], B[arm]))
        rows.append(f"| {_arm_cell(arm, A[arm] + B[arm])} | {_gflops(arm)} | {len(A[arm])} / {len(B[arm])} | **{_f(m['ap50'])}** | "
                    f"{'' if arm == BASE_ARM or base is None else _f(m['ap50'] - base, sign=True)} | {_f(m['ap5095'])} | "
                    f"{_f(m['f1'])} / {_f(m['p'])} / {_f(m['r'])} | {_f(m['r8'])} | {_f(m['car50'])} | {seeds} |")
    head = ["| arm | GFLOPs | seeds recA / recB | person AP50 | vs plain 416x320 | person AP50:95 | person F1 / P / R | "
            "person recall < 8 px | car AP50 | person AP50 per seed: recA; recB |", "|---|---|---|---|---|---|---|---|---|---|"]
    return head + rows


def _split_table(results, split) -> list[str]:
    g = _group(results, split)
    means = {arm: {k: _mean([values(r)[k] for r in rs]) for k in values(rs[0])} for arm, rs in g.items()}
    base = means.get(BASE_ARM, {}).get("ap50")
    rows = []
    for arm, rs in g.items():
        m = means[arm]
        rows.append(f"| {_arm_cell(arm, rs)} | {_gflops(arm)} | {len(rs)} | **{_f(m['ap50'])}** | "
                    f"{'' if arm == BASE_ARM or base is None else _f(m['ap50'] - base, sign=True)} | {_f(m['ap5095'])} | "
                    f"{_f(m['f1'])} / {_f(m['p'])} / {_f(m['r'])} | {_f(m['r8'])} | {_f(m['car50'])} | "
                    f"{' '.join(_f(values(r)['ap50']) for r in rs)} |")
    head = ["| arm | GFLOPs | seeds | person AP50 | vs plain 416x320 | person AP50:95 | person F1 / P / R | person recall < 8 px | "
            "car AP50 | person AP50 per seed |", "|---|---|---|---|---|---|---|---|---|---|"]
    return head + rows


def _release_table(results) -> list[str]:
    by_tag = {r["tag"]: r for r in results}
    rows = []
    for fn, m in RELEASES.items():
        if not str(m["dataset"]).startswith("hituav"):
            continue
        r = by_tag.get(m["run"])
        v = values(r) if r else {}
        split = r["split"] if r else m["dataset"].partition("-")[2] or "official"
        n_train = N_TRAIN_OFFICIAL if split == "official" else len(H.fold_parts(split)["train"])
        rows.append(f"| `{fn}` | `{m['run']}` | {split} train ({n_train:,} images), {m['epochs']} epochs | "
                    f"{_f(m.get('thr'), 3)} | {_f(v.get('ap50'))} | {_f(v.get('f1'))} / {_f(v.get('p'))} / {_f(v.get('r'))} |")
    if not rows:
        return []
    return ["| file | run | trained on | person threshold (val max F1) | test person AP50 | test person F1 / P / R |",
            "|---|---|---|---|---|---|"] + rows


def markdown(results: list[dict]) -> str:
    notes = sorted({r.get("detections") for r in results if r.get("detections")})
    L = ["# HIT-UAV results", "",
         "Person detection on HIT-UAV: aerial thermal-infrared drone images, 640x512 (Suo et al., Scientific Data 2023). "
         "pedestrian = the HIT-UAV class Person; car = Car. Detection only: the release holds single images, so there is no "
         "tracking evaluation. Every number on this page is regenerated from the per-run files in `results/hituav/` "
         "(no dataset needed):", "", "```", "python -B scripts/evaluate_hituav.py --table", "```", "",
         "Rules (`efpmot/data/hituav.py`, decisions 1-7; evaluator `efpmot/eval/detection.py`):",
         "- Training: COCO-pretrained YOLOv8n arms, the efpmot recipe (batch 16, AdamW, FP16) and the ~9,060-iteration budget of "
         "the other datasets: official 72 epochs, recA 116, recB 118, altLH 112. The model after the full schedule is evaluated.",
         "- Detections: FP16 export of every val and test image, conf 0.001, class-wise NMS 0.65, at most 300 per image.",
         "- AP50, AP50:95: COCO 101-point interpolated AP over all detections; DontCare boxes are ignore regions.",
         "- Operating point: the score threshold with the best person F1 at IoU 0.5 on the split's val part. F1 / P / R and recall "
         "are measured on test at that threshold. Recall < 8 px: GT persons whose native short side is below 8 px.",
         "- Values: mean over seeds. recA + recB: the mean of the two fold means, so every recording is tested exactly once.",
         "- GFLOPs: `efpmot.models.flops` at the canvas, 8 classes.", ""]
    L += ["## Recording-disjoint folds (recA + recB): unseen recordings", "",
          "Recordings (one camera angle of one flight) are split into two folds that balance person boxes per altitude and "
          "day / night cell; each fold's model is tested on the recordings of the other half. Every test recording shares its "
          "flight with training recordings: the folds test unseen recordings, not unseen sites.", ""]
    L += _fold_table(results) + [""]
    L += ["## Altitude shift (altLH): train at 60-90 m, test at 100-130 m", "",
          "Training and val use the recordings flown at 60-90 m, test every recording at 100-130 m. No flight has recordings on "
          "both sides, so the test flights are unseen and the persons are smaller.", ""]
    L += _split_table(results, "altLH") + [""]
    L += ["## Official split (inflated by recording overlap)", "",
          "The official train / val / test split interleaves frames of the same recordings: for 96 % of the test images a "
          "training image of the same recording lies within 10 serial numbers. These numbers measure recordings the model has "
          "seen; they are listed for comparison with work that reports the official split, not as a generalisation estimate.", ""]
    L += _split_table(results, "official") + [""]
    rel = _release_table(results)
    if rel:
        L += ["## Released weights", "",
              "Each released model is evaluated on its fold's test part, the recordings it never saw. For a video of a HIT-UAV "
              "recording, use the model of the fold that did not train on that recording (`efpmot.data.hituav.fold_recordings` "
              "lists the recordings of each fold). "
              "The person threshold is the operating point above; it is also stored in `efpmot.release.RELEASES` (`thr`) and in "
              "`results/hituav/<run>.json` (`op.pedestrian.thr`).", ""]
        L += rel + [""]
    L += ["## Notes", ""]
    if any(not r.get("code_included", True) for r in results):
        L.append("- P2 416x320 (code not included): YOLOv8n with an extra stride-4 P2 level from the original research code. Its "
                 "model is not part of this repository; its rows are this repository's evaluation of the detection caches of those "
                 "runs, under the same rules.")
    for n in notes:
        L.append(f"- Detection caches: {n}.")
    L.append(f"- Runs: {len(results)} files in `results/hituav/`.")
    return "\n".join(L) + "\n"


def write(results_dir: str, md_path: str) -> str:
    txt = markdown(load_results(results_dir))
    with open(md_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    return md_path
