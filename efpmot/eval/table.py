# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Published comparison tables: every contrast of results/contrasts.yaml through protocol.one(), Holm adjustment per
(dataset, class, tracker), labels, pre-registered prediction verdicts, and the writers of results/summary/*.json,
results/summary.csv and results/RESULTS.md."""
from __future__ import annotations

import csv
import json
import os

import yaml

from .. import paths
from . import protocol as P

DET_NAME = {"v8n": "YOLOv8n", "y11n": "YOLO11n"}
DATA_NAME = {"dsec": "DSEC-Det test (13 sequences from 8 drives)", "mot17": "MOT17-train evaluation halves (7 sequences)",
             "visdrone": "VisDrone2019-MOT test-dev (17 sequences)", "bdd": "BDD100K MOT val (200 videos)"}
CANVAS_H = {"dsec": {"416": 320, "640": 480}, "mot17": {"416": 256}, "visdrone": {"640": 384}, "bdd": {"416": 256, "640": 384}}
TRK_NAME = {"bytetrack": "ByteTrack", "ocsort": "OC-SORT"}
DATA_ORDER = ("dsec", "mot17", "visdrone", "bdd")
DATA_SHORT = {"dsec": "DSEC", "mot17": "MOT17", "visdrone": "VisDrone", "bdd": "BDD100K"}
# GFLOPs (2 x thop MACs, 8 classes) of the arms in the headline tables, from efpmot.models.flops (tests/test_models.py)
GFLOPS = {("v8n", "plain 416x320"): 2.665, ("v8n", "S3x 416x320"): 2.955, ("v8n", "uniform 448x352"): 3.157,
          ("v8n", "plain 640x480"): 6.149, ("v8n", "plain 416x256"): 2.132, ("v8n", "S3x 416x256"): 2.364,
          ("v8n", "uniform 480x256"): 2.460, ("v8n", "plain 640x384"): 4.920, ("v8n", "S3x 640x384"): 5.455,
          ("v8n", "uniform 704x384"): 5.411, ("y11n", "plain 416x320"): 2.101, ("y11n", "S3x 416x320"): 2.333,
          ("y11n", "plain 416x256"): 1.680, ("y11n", "S3x 416x256"): 1.865, ("y11n", "uniform 480x256"): 1.939}
# Jetson latency (deploy/jetson/results/latency_orin_nano.csv, written by deploy/jetson/parse_results.py):
# canvas-input graphs of the arms in this table -> (trained on, arm)
LATENCY_CSV = os.path.join("deploy", "jetson", "results", "latency_orin_nano.csv")
LAT_ARMS = {"v8n_416_uniform_canvas": ("dsec", "plain 416x320"), "v8n_416_s3x_canvas": ("dsec", "S3x 416x320"),
            "v8n_416_s3y_canvas": ("dsec", "S3y 416x320"),
            "v8n_416_p2_canvas": ("dsec", "P2 416x320 (reference, model not in this repository)"),
            "v8n_448x352_uniform_canvas": ("dsec", "uniform 448x352"), "v8n_640_uniform_canvas": ("dsec", "plain 640x480"),
            "bdd_416_uniform_canvas": ("bdd", "plain 416x256"), "bdd_416_s3x_canvas": ("bdd", "S3x 416x256"),
            "bdd_640_uniform_canvas": ("bdd", "plain 640x384")}


def arm_name(data: str, var: str) -> str:
    """run-tag variant -> readable arm, e.g. dsec '416s3x' -> 'S3x 416x320', '448x352' -> 'uniform 448x352'"""
    base, _, suf = var.partition("_")
    if "x" in base and base.split("x")[0].isdigit() and base.split("x")[1].isdigit():
        return f"uniform {base}"
    mod = ""
    for m in ("s3x", "s3y"):
        if base.endswith(m):
            base, mod = base[: -len(m)], m
    canvas = f"{base}x{CANVAS_H[data][base]}"
    if suf == "p2":
        return f"P2 {canvas}"
    return f"{'S3x' if mod == 's3x' else 'S3y' if mod == 's3y' else 'plain'} {canvas}"


def load_contrasts(path: str) -> list[dict]:
    rows = yaml.safe_load(open(path, encoding="utf-8"))
    for c in rows:
        c["a"], c["b"] = str(c["a"]), str(c["b"])
    return rows


def run_table(contrasts: list[dict], B: int = 10000, source: str = "auto") -> list[dict]:
    """one() for every contrast x class x tracker (rule cross), then Holm over each (dataset, class, tracker) family"""
    rows = []
    for c in contrasts:
        for cls in c["classes"]:
            for trk in P.TRACKERS:
                r = P.one(c["dataset"], c["det"], c["a"], c["b"], cls=cls, trk=trk, rule="cross", B=B, source=source)
                rows.append(dict(dataset=c["dataset"], det=c["det"], a=c["a"], b=c["b"], cls=cls, trk=trk, rule="cross",
                                 group=c["group"], prediction=c.get("prediction"), A_name=arm_name(c["dataset"], c["a"]),
                                 B_name=arm_name(c["dataset"], c["b"]), **r))
                if c["dataset"] == "dsec":   # drive-level sensitivity (13 test sequences from 8 drives)
                    rows[-1]["sensitivity_drive"] = P.cluster_sensitivity(c["dataset"], c["det"], c["a"], c["b"], cls=cls,
                                                                          trk=trk, B=B, source=source)
    fam: dict[tuple, list[dict]] = {}
    for r in rows:
        fam.setdefault((r["dataset"], r["cls"], r["trk"]), []).append(r)
    for rs in fam.values():
        pa, pt = P.holm([r["p_ar"] for r in rs]), P.holm([r["p_tost"] for r in rs])
        for r, x, y in zip(rs, pa, pt):
            r["p_ar_adj"], r["p_tost_adj"], r["holm_m"] = float(x), float(y), len(rs)
            r["label"] = P.label(r)
    return rows


def verdicts(rows: list[dict]) -> list[dict]:
    """pre-registered reading rules (cross-fitted thresholds, unadjusted p_AR):
    P1 / P2 hold iff d > 0 and p_AR < 0.05 for ByteTrack AND OC-SORT; a MOT17 P1 that does not hold is 'not shown'
    unless the 95 % CI upper bound is below +0.5 for both trackers ('refuted');
    P3 / P4 hold iff non-inferior (90 % CI lower bound > -margin), fail iff d < 0 with p_AR < 0.05, else inconclusive;
    ByteTrack is primary, OC-SORT is reported beside it."""
    out = []
    keys = sorted({(r["dataset"], r["prediction"], r["cls"]) for r in rows if r.get("prediction")},
                  key=lambda k: (DATA_ORDER.index(k[0]), k[1]))
    for data, pid, cls in keys:
        rs = {r["trk"]: r for r in rows if r["dataset"] == data and r["prediction"] == pid and r["cls"] == cls}
        bt, oc = rs["bytetrack"], rs["ocsort"]
        if pid in ("P1", "P2"):
            if all(r["d"] > 0 and r["p_ar"] < 0.05 for r in (bt, oc)):
                v = "holds"
            elif data == "mot17" and pid == "P1" and all(r["ci95"][1] < 0.5 for r in (bt, oc)):
                v = "refuted"
            else:
                v = "not shown"
            rule = "d > 0 and p_AR < 0.05, both trackers"
        else:
            if bt["noninferior_90"]:
                v = "holds (non-inferior)"
            elif bt["d"] < 0 and bt["p_ar"] < 0.05:
                v = "fails"
            else:
                v = "inconclusive"
            rule = f"ByteTrack 90 % CI lower bound > -{bt['delta']}"
        out.append(dict(dataset=data, prediction=pid, cls=cls, A_name=bt["A_name"], B_name=bt["B_name"], rule=rule, verdict=v,
                        bytetrack=_brief(bt), ocsort=_brief(oc)))
    return out


def _brief(r: dict) -> dict:
    return dict(d=r["d"], ci95=r["ci95"], ci90=r["ci90"], p_ar=r["p_ar"], noninferior_90=r.get("noninferior_90"))


# ---- writers --------------------------------------------------------------------------------------------------------------
CSV_COLS = ["dataset", "detector", "A", "B", "A_name", "B_name", "class", "tracker", "group", "prediction", "seeds",
            "hota_A", "hota_B", "diff", "ci95_lo", "ci95_hi", "ci90_lo", "ci90_hi", "p_ar", "p_ar_holm", "p_tost",
            "p_tost_holm", "margin", "label", "wins_losses", "n_seq"]


def _csv_row(r: dict) -> dict:
    return {"dataset": r["dataset"], "detector": DET_NAME[r["det"]], "A": r["a"], "B": r["b"], "A_name": r["A_name"],
            "B_name": r["B_name"], "class": r["cls"], "tracker": r["trk"], "group": r["group"], "prediction": r["prediction"] or "",
            "seeds": f"{','.join(map(str, r['seedsA']))}/{','.join(map(str, r['seedsB']))}",
            "hota_A": f"{r['A']:.2f}", "hota_B": f"{r['B']:.2f}", "diff": f"{r['d']:+.2f}",
            "ci95_lo": f"{r['ci95'][0]:+.2f}", "ci95_hi": f"{r['ci95'][1]:+.2f}", "ci90_lo": f"{r['ci90'][0]:+.2f}",
            "ci90_hi": f"{r['ci90'][1]:+.2f}", "p_ar": f"{r['p_ar']:.4f}", "p_ar_holm": f"{r['p_ar_adj']:.4f}",
            "p_tost": f"{r['p_tost']:.4f}", "p_tost_holm": f"{r['p_tost_adj']:.4f}", "margin": r["delta"], "label": r["label"],
            "wins_losses": f"{r['seq_wins']}/{r['seq_losses']}", "n_seq": r["S"]}


def write_json(rows: list[dict], out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for data in DATA_ORDER:
        rs = [r for r in rows if r["dataset"] == data]
        if not rs:
            continue
        f = os.path.join(out_dir, f"{data}.json")
        with open(f, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(rs, fh, indent=1)
            fh.write("\n")
        files.append(f)
    return files


def write_csv(rows: list[dict], path: str) -> str:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, CSV_COLS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(_csv_row(r))
    return path


def _p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _table(rows: list[dict]) -> list[str]:
    L = ["| detector | A | B | class | tracker | seeds A / B | HOTA A | HOTA B | d | 95 % CI | p_AR | Holm p_AR | TOST p (Holm) | label |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {DET_NAME[r['det']]} | {r['A_name']} | {r['B_name']} | {r['cls']} | {TRK_NAME[r['trk']]} | "
                 f"{','.join(map(str, r['seedsA']))} / {','.join(map(str, r['seedsB']))} | {r['A']:.2f} | {r['B']:.2f} | "
                 f"**{r['d']:+.2f}** | [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] | {_p(r['p_ar'])} | {_p(r['p_ar_adj'])} | "
                 f"{_p(r['p_tost_adj'])} | {r['label']} |")
    return L


def _latency() -> dict:
    """(trained on, arm, precision, pass) -> (canvas W x H, median ms as text) of the canvas graphs of LAT_ARMS, in file
    order; {} when the latency file is absent"""
    f = os.path.join(paths.repo_root(), LATENCY_CSV)
    if not os.path.isfile(f):
        return {}
    out = {}
    with open(f, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["graph"] in LAT_ARMS and r["input"] == "canvas":
                data, arm = LAT_ARMS[r["graph"]]
                out[(data, arm, r["precision"].upper(), r["pass"])] = (r["canvas_wxh"], f"{float(r['median_ms']):.3f}")
    return out


def write_md(rows: list[dict], path: str, B: int) -> str:
    V = verdicts(rows)
    by = {d: [r for r in rows if r["dataset"] == d] for d in DATA_ORDER}
    L = ["# Results", "",
         "Tracking HOTA of each detector arm on held-out evaluation data, compared with the cross-fitted paired protocol "
         "(`efpmot/eval/protocol.py`). d = HOTA(A) - HOTA(B), in HOTA points. Every number on this page is regenerated "
         "from the per-run statistics in `results/stats/` (no dataset needed):", "",
         "```", "python -B scripts/compare.py --table results/contrasts.yaml", "```", "",
         "Outputs: `results/summary/<dataset>.json` (every field of each contrast), `results/summary.csv` (one row per "
         "contrast, class and tracker) and this page.", ""]
    # headline
    def cell(r):
        txt = f"{r['d']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]"
        return (f"**{txt}**" if r["label"].startswith(("win", "loss")) else txt + " n.s.") + f" | {_p(r['p_ar_adj'])}"

    def pairs(group):
        for data in DATA_ORDER:
            for r in by[data]:
                if r["group"] != group or r["trk"] != "bytetrack":
                    continue
                o = next(x for x in by[data] if x["group"] == group and x["det"] == r["det"] and x["cls"] == r["cls"]
                         and x["b"] == r["b"] and x["trk"] == "ocsort")
                yield data, r, o

    lat = _latency()
    L += ["## S3x against the plain detector", "",
          "Same detector, same canvas, same training schedule; only the P3 grid differs (S3x: twice as many columns, "
          "about +11 % FLOPs). d = HOTA(S3x) - HOTA(plain) with its 95 % interval; p = Holm-adjusted randomisation p. "
          "Bold: a win (p < 0.05); n.s.: not significant.", "",
          "| data | detector | canvas | class | ByteTrack d [95 % CI] | p | OC-SORT d [95 % CI] | p |",
          "|---|---|---|---|---|---|---|---|"]
    for data, r, o in pairs("s3x_vs_plain"):
        L.append(f"| {DATA_SHORT[data]} | {DET_NAME[r['det']]} | {r['A_name'].split()[1]} | {r['cls']} | {cell(r)} | {cell(o)} |")
    L += ["", "## S3x against a uniform canvas of the same cost", "",
          "S3x adds about 11 % FLOPs. The fair question is whether spending them on a slightly larger uniform canvas "
          "does as well. Each compute-matched canvas is a uniform letterbox canvas whose FLOPs are within a few per cent "
          "of those of S3x: 0 to 7 % more, VisDrone 0.8 % less (\"matched\" means matched on FLOPs, not on latency). Same "
          "detector, schedule and number of seeds.", "",
          "| data | detector | S3x canvas | uniform canvas | GFLOPs S3x / uniform | class | ByteTrack d [95 % CI] | p | "
          "OC-SORT d [95 % CI] | p |", "|---|---|---|---|---|---|---|---|---|---|"]
    for data, r, o in pairs("s3x_vs_compute_matched"):
        ga, gb = GFLOPS.get((r["det"], r["A_name"])), GFLOPS.get((r["det"], r["B_name"]))
        fl = f"{ga:.3f} / {gb:.3f}" if ga and gb else "-"
        L.append(f"| {DATA_SHORT[data]} | {DET_NAME[r['det']]} | {r['A_name'].split()[1]} | {r['B_name'].split()[1]} | {fl} | "
                 f"{r['cls']} | {cell(r)} | {cell(o)} |")
    L += ["", "- Pedestrians: S3x beats the same-cost canvas on DSEC and BDD100K (YOLO11n on BDD100K: ByteTrack only); "
          "MOT17 and VisDrone are ties. Cars tie everywhere."]
    s3x, uni = lat.get(("dsec", "S3x 416x320", "FP16", "1")), lat.get(("dsec", "uniform 448x352", "FP16", "1"))
    if s3x and uni:
        L += [f"- Latency is not FLOPs: on the Jetson Orin Nano the 448x352 uniform engine takes {float(uni[1]):.2f} ms and "
              f"the S3x engine {float(s3x[1]):.2f} ms (TensorRT FP16, table at the end of this page)."]
    ref = list(pairs("reference_640_vs_s3x"))
    if ref:
        txt = "; ".join(f"{DATA_SHORT[d]} {r['cls']} {r['d']:+.2f} / {o['d']:+.2f}" for d, r, o in ref)
        L += [f"- A 640-wide uniform canvas (about 2.1x the FLOPs of S3x) still beats S3x (plain 640 minus S3x, ByteTrack / "
              f"OC-SORT): {txt}."]
    L += [""]
    # predictions
    L += ["## Pre-registered predictions (MOT17, VisDrone)", "",
          "Written before any training on these datasets (`docs/PREREGISTRATION.md`). Reading rules: P1 / P2 hold iff "
          "d > 0 and the unadjusted randomisation p < 0.05 for ByteTrack and OC-SORT; P3 / P4 hold iff non-inferior "
          "(ByteTrack 90 % bootstrap interval lower bound above -1.1955), fail iff d < 0 with p < 0.05, else inconclusive. "
          "For MOT17 a P1 that does not hold is \"not shown\", not \"refuted\", unless the 95 % interval excludes +0.5.", "",
          "| data | id | contrast | class | ByteTrack d [90 % CI], p_AR | OC-SORT d [90 % CI], p_AR | verdict |",
          "|---|---|---|---|---|---|---|"]
    for v in V:
        b, o = v["bytetrack"], v["ocsort"]
        L.append(f"| {v['dataset']} | {v['prediction']} | {v['A_name']} vs {v['B_name']} | {v['cls']} | "
                 f"{b['d']:+.2f} [{b['ci90'][0]:+.2f}, {b['ci90'][1]:+.2f}], {_p(b['p_ar'])} | "
                 f"{o['d']:+.2f} [{o['ci90'][0]:+.2f}, {o['ci90'][1]:+.2f}], {_p(o['p_ar'])} | **{v['verdict']}** |")
    L += ["", "With 7 MOT17 sequences the exact randomisation test has 2^7 = 128 swap patterns, so its smallest attainable "
          "p is 2/128 = 0.0156.", ""]
    # per dataset
    for data in DATA_ORDER:
        rs = by[data]
        if not rs:
            continue
        L += [f"## {DATA_NAME[data]}", "", f"Equivalence margin {P.MARGINS[data]} HOTA. Holm families: class x tracker over "
              f"the contrasts of this dataset.", ""]
        L += _table(rs) + [""]
        if data == "dsec":
            L += ["Sensitivity (reported, not deciding): the 13 test sequences come from 8 drives (for example "
                  "zurich_city_14_a, _b and _c are one drive). Thresholds cross-fitted leave-one-drive-out, bootstrap and "
                  "exact randomisation over the 8 drives. With 8 drives the smallest attainable p_AR is 2/256 = 0.0078, so "
                  "with 7 contrasts per Holm family no drive-level difference can reach a Holm-adjusted p below 0.05; "
                  "the unadjusted p_AR is shown.", "",
                  "| detector | A | B | class | tracker | d | 95 % CI | p_AR |", "|---|---|---|---|---|---|---|---|"]
            for r in rs:
                s_ = r["sensitivity_drive"]
                L.append(f"| {DET_NAME[r['det']]} | {r['A_name']} | {r['B_name']} | {r['cls']} | {TRK_NAME[r['trk']]} | "
                         f"{s_['d']:+.2f} | [{s_['ci95'][0]:+.2f}, {s_['ci95'][1]:+.2f}] | {_p(s_['p_ar'])} |")
            L += [""]
            L += ["`P2 416x320` is YOLOv8n with an extra stride-4 (P2) detection level, the common small-object baseline, "
                  "trained with the same schedule; it is reported for reference and its model code is not part of this "
                  "repository.", ""]
        if data == "mot17":
            L += ["Static-camera sequences (MOT17-02, -04, -09) against the moving-camera ones (-05, -10, -11, -13), each "
                  "sequence at its cross-fitted threshold; descriptive only (3 and 4 sequences, no test).", "",
                  "| A | B | tracker | d static | d moving |", "|---|---|---|---|---|"]
            for r in rs:
                s = r["static_split"]
                L.append(f"| {r['A_name']} | {r['B_name']} | {TRK_NAME[r['trk']]} | {s['static']['d']:+.2f} | {s['moving']['d']:+.2f} |")
            L += [""]
        if data == "visdrone":
            L += ["Sensitivity (reported, not deciding): thresholds cross-fitted leave-one-source-video-out and bootstrap / "
                  "randomisation over the 14 source videos of the 17 test-dev clips.", "",
                  "| A | B | class | tracker | d | 95 % CI | p_AR |", "|---|---|---|---|---|---|---|"]
            for r in rs:
                s = r["sensitivity_video"]
                L.append(f"| {r['A_name']} | {r['B_name']} | {r['cls']} | {TRK_NAME[r['trk']]} | {s['d']:+.2f} | "
                         f"[{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}] | {_p(s['p_ar'])} |")
            L += [""]
    if lat:
        L += ["## Detector latency on a Jetson Orin Nano", "",
              "GPU compute time of the TensorRT engine alone (trtexec, 15 W power mode, batch 1, canvas input; no "
              "pre-processing, NMS or tracking), median over a 10 s run after 500 ms warm-up. The INT8 engines were built "
              "without a calibration set, so they give latency only and no accuracy claim. Rows marked pass 2 repeat the "
              "measurement at the end of the session (drift check). Source, with the native-frame graphs: "
              "`deploy/jetson/results/latency_orin_nano.csv` (written by `deploy/jetson/parse_results.py`).", "",
              "| trained on | arm | canvas (W x H) | precision | median ms | pass |", "|---|---|---|---|---|---|"]
        for (data, arm, prec, pas), (canvas, med) in lat.items():
            L.append(f"| {data} | {arm} | {canvas} | {prec} | {med} | {pas} |")
        L += [""]
    L += ["## Protocol", "",
          "- **Thresholds are never chosen on the sequence they score.** Each tracker is run over a grid of detection "
          "thresholds (ByteTrack 0.2-0.9, OC-SORT 0.5-0.9). For every held-out sequence the threshold is the one with the "
          "best pooled HOTA on the other sequences (leave-one-sequence-out; BDD100K: 10 fixed folds of 20 videos), "
          "separately for each run, class and tracker.",
          "- **HOTA is recombined exactly** from TrackEval's per-sequence sufficient statistics (TP, FN, FP and "
          "AssA x TP at the 19 localisation thresholds), so a pooled score over any set of sequences equals what "
          "TrackEval would report for that set.",
          f"- **Paired hierarchical bootstrap** (B = {B}): sequences are resampled with the same draw for both arms, then "
          "seeds within each arm; the 95 % and 90 % percentile intervals of d are reported.",
          "- **Paired randomisation test**: per sequence, the seed-averaged statistics of A and B are swapped with "
          "probability 1/2 (exact enumeration when 2^K <= B: DSEC 2^13, MOT17 2^7; else B random patterns); two-sided p_AR.",
          "- **Equivalence (TOST)** against a margin frozen before the comparisons were read: 2 x the median within-arm seed "
          "SD of pooled ByteTrack car HOTA over earlier 416 / 640 canvas arms (DSEC 1.1955, BDD100K 0.0421); it measures "
          "seed noise only, not any difference between arms. MOT17 and VisDrone use the DSEC value, frozen "
          "before any training on them, because a seed-SD estimate from 2 arms x 2 seeds would be too noisy.",
          "- **Labels**: win / loss iff the Holm-adjusted p_AR < 0.05; otherwise equivalent iff the Holm-adjusted TOST "
          "p < 0.05; otherwise inconclusive. ' <m' marks a significant difference that is smaller than the margin.",
          "- **Seeds**: every arm has 2 or 3 independently trained seeds (listed per row); all of them are used.", "",
          "VisDrone2019-MOT and BDD100K are research-only datasets: this repository ships numbers for them, no media and no "
          "weights.", ""]
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L))
    return path
