# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Published comparison tables: every contrast of results/contrasts.yaml through protocol.one(), Holm adjustment per
(dataset, class, tracker), labels, pre-registered prediction verdicts, and the writers of results/summary/*.json,
results/summary.csv and results/RESULTS.md. Contrasts between detectors (results/contrasts_detectors.yaml, key det_b)
go through cross_one() and have their own writers (results/summary_detectors*, results/DETECTORS.md)."""
from __future__ import annotations

import csv
import json
import os

import yaml

from .. import paths
from . import protocol as P

DET_NAME = {"v8n": "YOLOv8n", "y11n": "YOLO11n", "y26n": "YOLO26n", "nano": "YOLOX-Nano", "tiny": "YOLOX-Tiny",
            "nd": "NanoDet-Plus-m", "v8s": "YOLOv8s", "rtd": "RT-DETR-L"}
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
# Detector comparison (results/contrasts_detectors.yaml -> results/DETECTORS.md). (detector, variant) -> (params [M],
# GFLOPs): 2 x thop MACs of the inference graph on 1 x 3 x H x W at the arm's canvas, nc 8. Repository models from
# efpmot.models.flops at its precision (tests/test_detectors.py); the other families measured the same way on their own
# implementations (not part of this repository): NanoDet-Plus-m without its training-only auxiliary head, YOLO26n as the
# deployed one-to-one graph.
DET_COST = {("v8n", "416s3x"): (2.92, 2.9547), ("v8n", "416"): (3.01, 2.6647), ("v8n", "448x352"): (3.01, 3.1567),
            ("v8n", "640"): (3.01, 6.1494), ("y11n", "416"): (2.59, 2.1013), ("v8n", "416_p2"): (3.18, 4.93),
            ("y26n", "416"): (2.38, 1.74), ("nano", "416"): (0.90, 0.83), ("tiny", "416"): (5.06, 4.96),
            ("nd", "416"): (1.17, 1.20), ("v8s", "416"): (11.14, 9.31), ("rtd", "416"): (32.82, 39.03)}
REPO_DETS = ("v8n", "y11n")      # detectors of efpmot.models; a variant with a suffix (416_p2) is not a repository model
# Jetson latency of the detector table: the detector session (DET_LATENCY_CSV) where it timed the arm, else the first
# session (LATENCY_CSV); canvas graph -> (detector, variant)
DET_LATENCY_CSV = os.path.join("deploy", "jetson", "results", "latency_orin_nano_detectors.csv")
DET_LAT_ARMS = {"v8n_416_s3x_canvas": ("v8n", "416s3x"), "v8n_416_uniform_canvas": ("v8n", "416"),
                "v8n_448x352_uniform_canvas": ("v8n", "448x352"), "v8n_640_uniform_canvas": ("v8n", "640"),
                "v8n_416_p2_canvas": ("v8n", "416_p2"), "y11n_416_canvas": ("y11n", "416"),
                "y26n_416_canvas": ("y26n", "416"), "yoloxn_416_canvas": ("nano", "416"),
                "yoloxt_416_canvas": ("tiny", "416"), "nanodet_416_canvas": ("nd", "416"),
                "v8s_416_canvas": ("v8s", "416"), "rtdetrl_416_canvas": ("rtd", "416")}


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


def det_arm(data: str, det: str, var: str) -> tuple[str, str]:
    """(detector, variant) -> (model, canvas), e.g. ('v8n', '416s3x') -> ('YOLOv8n + S3x', '416x320'), ('v8n', '448x352')
    -> ('YOLOv8n', '448x352'), ('nano', '416') -> ('YOLOX-Nano', '416x320')"""
    kind, canvas = arm_name(data, var).split()
    return DET_NAME[det] + (f" + {kind}" if kind in ("S3x", "S3y", "P2") else ""), canvas


def load_contrasts(path: str) -> list[dict]:
    rows = yaml.safe_load(open(path, encoding="utf-8"))
    for c in rows:
        c["a"], c["b"] = str(c["a"]), str(c["b"])
    return rows


def cross_one(data: str, det: str, a: str, det_b: str, b: str, cls: str = "car", trk: str = "bytetrack", B: int = 10000,
              source: str = "auto", drive: bool = False) -> dict:
    """protocol.one (rule cross) or, with drive=True, protocol.cluster_sensitivity for arm A of detector det against arm B
    of another detector det_b: the same calls with the arms of two detectors (bootstrap / randomisation seed 0, frozen
    margin of the dataset). DSEC and BDD100K only (one() adds nothing else there)."""
    if data not in ("dsec", "bdd"):
        raise ValueError(f"contrasts between detectors: DSEC / BDD100K only, not {data}")
    MA, MB = P.method(data, det, a, trk, cls, source), P.method(data, det_b, b, trk, cls, source)
    if not MA["seeds"] or not MB["seeds"]:
        raise ValueError(f"no complete runs: {P.tag_of(data, det, a, 0)} seeds {MA['seeds']} (excluded {MA['excluded']}), "
                         f"{P.tag_of(data, det_b, b, 0)} seeds {MB['seeds']} (excluded {MB['excluded']})")
    delta = P.MARGINS[data]
    if not drive:
        return dict(seedsA=MA["seeds"], seedsB=MB["seeds"], exclA=MA["excluded"], exclB=MB["excluded"],
                    **P.compare(MA["X"]["cross"], MB["X"]["cross"], B, 0, delta))
    rule, cl = P.CLUSTER_RULE[data], P.clusters_for(data)
    rv = P.compare(MA["X"][rule], MB["X"][rule], B, 0, delta, clusters=cl)
    return dict(rule=rule + " + cluster bootstrap / randomisation", clusters=int(cl.max()) + 1,
                **{k: rv[k] for k in ("A", "B", "d", "ci95", "ci90", "p_ar", "ar_exact", "p_tost", "se")},
                noninferior_90=bool(rv["ci90"][0] > -delta))


def run_table(contrasts: list[dict], B: int = 10000, source: str = "auto") -> list[dict]:
    """one() for every contrast x class x tracker (rule cross), then Holm over each (dataset, class, tracker) family.
    A contrast whose arm B is another detector (key det_b) goes through cross_one()."""
    rows = []
    for c in contrasts:
        db = c.get("det_b", c["det"])
        for cls in c["classes"]:
            for trk in P.TRACKERS:
                if db == c["det"]:
                    r = P.one(c["dataset"], c["det"], c["a"], c["b"], cls=cls, trk=trk, rule="cross", B=B, source=source)
                else:
                    r = cross_one(c["dataset"], c["det"], c["a"], db, c["b"], cls=cls, trk=trk, B=B, source=source)
                rows.append(dict(dataset=c["dataset"], det=c["det"], **({"det_b": db} if "det_b" in c else {}), a=c["a"],
                                 b=c["b"], cls=cls, trk=trk, rule="cross", group=c["group"], prediction=c.get("prediction"),
                                 A_name=arm_name(c["dataset"], c["a"]), B_name=arm_name(c["dataset"], c["b"]), **r))
                if c["dataset"] == "dsec":   # drive-level sensitivity (13 test sequences from 8 drives)
                    rows[-1]["sensitivity_drive"] = (
                        P.cluster_sensitivity(c["dataset"], c["det"], c["a"], c["b"], cls=cls, trk=trk, B=B, source=source)
                        if db == c["det"] else
                        cross_one(c["dataset"], c["det"], c["a"], db, c["b"], cls=cls, trk=trk, B=B, source=source, drive=True))
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
          "of those of S3x: 4 to 7 % more, VisDrone 0.8 % less (\"matched\" means matched on FLOPs, not on latency). Same "
          "detector, schedule and number of seeds.", "",
          "| data | detector | S3x canvas | uniform canvas | GFLOPs S3x / uniform | class | ByteTrack d [95 % CI] | p | "
          "OC-SORT d [95 % CI] | p |", "|---|---|---|---|---|---|---|---|---|---|"]
    for data, r, o in pairs("s3x_vs_compute_matched"):
        ga, gb = GFLOPS.get((r["det"], r["A_name"])), GFLOPS.get((r["det"], r["B_name"]))
        fl = f"{ga:.3f} / {gb:.3f}" if ga and gb else "-"
        L.append(f"| {DATA_SHORT[data]} | {DET_NAME[r['det']]} | {r['A_name'].split()[1]} | {r['B_name'].split()[1]} | {fl} | "
                 f"{r['cls']} | {cell(r)} | {cell(o)} |")
    L += ["", "- Pedestrians: S3x beats the same-cost canvas on DSEC and BDD100K (YOLO11n on BDD100K: ByteTrack only); "
          "on MOT17 and VisDrone the difference is not significant. No car difference is significant."]
    s3x, uni = lat.get(("dsec", "S3x 416x320", "FP16", "1")), lat.get(("dsec", "uniform 448x352", "FP16", "1"))
    if s3x and uni:
        L += [f"- Latency is not FLOPs: on the Jetson Orin Nano the 448x352 uniform engine takes {float(uni[1]):.2f} ms and "
              f"the S3x engine {float(s3x[1]):.2f} ms (TensorRT FP16, table at the end of this page)."]
    ref = list(pairs("reference_640_vs_s3x"))
    if ref:
        def val(x):
            return f"{x['d']:+.2f}" + ("" if x["label"].startswith(("win", "loss")) else " n.s.")

        txt = "; ".join(f"{DATA_SHORT[d]} {r['cls']} {val(r)} / {val(o)}" for d, r, o in ref)
        L += [f"- A 640-wide uniform canvas (about 2.1x the FLOPs of S3x) is ahead of S3x (plain 640 minus S3x, ByteTrack / "
              f"OC-SORT; n.s. = not significant): {txt}."]
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
          "VisDrone2019-MOT (no licence stated) and BDD100K (licence: educational, research and not-for-profit use): this "
          "repository ships numbers for them, no media and no weights.", ""]
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L))
    return path


# ---- detector comparison writers (results/summary_detectors*, results/DETECTORS.md) ----------------------------------------
def write_detectors_csv(rows: list[dict], path: str) -> str:
    """the summary.csv columns with the detector of each arm (detector_A, detector_B) in place of detector"""
    cols = ["dataset", "detector_A", "detector_B"] + CSV_COLS[2:]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            row = _csv_row(r)
            del row["detector"]
            w.writerow(dict(row, detector_A=DET_NAME[r["det"]], detector_B=DET_NAME[r.get("det_b", r["det"])]))
    return path


def _cell(r: dict) -> str:
    """'d [95 % CI] | Holm p' with the interval bold for a win / loss and 'n.s.' otherwise (the headline cells of write_md)"""
    txt = f"{r['d']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]"
    return (f"**{txt}**" if r["label"].startswith(("win", "loss")) else txt + " n.s.") + f" | {_p(r['p_ar_adj'])}"


def _det_latency() -> dict:
    """(latency csv, detector, variant, pass) -> FP16 median ms of the canvas graphs of DET_LAT_ARMS in both Jetson
    sessions; a missing file gives no entries"""
    out = {}
    for f in (DET_LATENCY_CSV, LATENCY_CSV):
        p = os.path.join(paths.repo_root(), f)
        if not os.path.isfile(p):
            continue
        with open(p, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                if r["graph"] in DET_LAT_ARMS and r["input"] == "canvas" and r["precision"] == "fp16":
                    out[(f, *DET_LAT_ARMS[r["graph"]], int(r["pass"]))] = float(r["median_ms"])
    return out


def write_detectors_md(rows: list[dict], path: str, B: int) -> str:
    """results/DETECTORS.md (one dataset, one arm A): every arm with its cost, Jetson latency and pedestrian HOTA against
    arm A, every contrast, the drive-level sensitivity check (DSEC) and the notes"""
    data, ref = rows[0]["dataset"], (rows[0]["det"], rows[0]["a"])
    if any((r["dataset"], r["det"], r["a"]) != (data, *ref) for r in rows):
        raise ValueError("write_detectors_md: every contrast needs the same dataset and arm A")
    lat, m = _det_latency(), max(r["holm_m"] for r in rows)

    def key(r):
        return r.get("det_b", r["det"]), r["b"]

    def full(k):
        return " ".join(det_arm(data, *k))

    def short(k):
        nm, canvas = det_arm(data, *k)
        return nm if canvas == det_arm(data, *ref)[1] else f"{nm} {canvas}"

    def in_repo(k):
        return k[0] in REPO_DETS and "_" not in k[1]

    def ms(k):   # first pass: the detector session, else the first session (flag True)
        for f in (DET_LATENCY_CSV, LATENCY_CSV):
            if (f, *k, 1) in lat:
                return lat[(f, *k, 1)], f == LATENCY_CSV
        return None, False

    def sig(r):
        return f"**{r['d']:+.2f}**" if r["label"].startswith(("win", "loss")) else f"{r['d']:+.2f}"

    ped = {(key(r), r["trk"]): r for r in rows if r["cls"] == "pedestrian"}
    arms = sorted({k for k, _ in ped} | {ref}, key=lambda k: DET_COST.get(k, (0, float("inf")))[1])
    nm_ref = det_arm(data, *ref)[0]
    L = [f"# Detector comparison on {DATA_SHORT[data]}", "",
         f"{full(ref)} against other lightweight and real-time detectors on {DATA_NAME[data]}, with the protocol of "
         f"[`RESULTS.md`](RESULTS.md): cross-fitted thresholds, paired hierarchical bootstrap (B = {B}), paired "
         "randomisation test, Holm adjustment, TOST against the frozen margin. d = "
         f"HOTA({nm_ref}) - HOTA(detector), in HOTA points. Every number on this page is regenerated from the per-run "
         "statistics in `results/stats/` and the latency tables in `deploy/jetson/results/` (no dataset needed):", "",
         "```", "python -B scripts/compare.py --detectors results/contrasts_detectors.yaml", "```", "",
         "Outputs: `results/summary_detectors/<dataset>.json` (every field of each contrast), "
         "`results/summary_detectors.csv` (one row per contrast, class and tracker) and this page. The contrasts of "
         f"`results/contrasts_detectors.yaml` form their own Holm families (class x tracker, {m} contrasts each). "
         "Contrasts that `RESULTS.md` also reports have the same d (sign flipped where A and B are swapped) and "
         "unadjusted p there, but a different Holm adjustment.", ""]
    L += ["## Pedestrians: cost, latency and tracking accuracy", "",
          f"Sorted by GFLOPs. HOTA: seed-mean pooled HOTA of the arm. d = HOTA({nm_ref}) - HOTA(detector) with its 95 % "
          "interval; p = Holm-adjusted randomisation p. Bold: significant (p < 0.05); n.s.: not significant. code: `repo` = "
          "a model family of this repository; `results only` = its code is not part of this repository and its rows are "
          "regenerated from the shipped stats (like the P2 row of `RESULTS.md`).", "",
          "| detector | canvas | code | params [M] | GFLOPs | Jetson FP16 [ms] | seeds | HOTA ByteTrack / OC-SORT | "
          "ByteTrack d [95 % CI] | p | OC-SORT d [95 % CI] | p |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    a_bt, a_oc = (next(r for r in ped.values() if r["trk"] == t) for t in P.TRACKERS)
    for k in arms:
        nm, canvas = det_arm(data, *k)
        pa, gf = DET_COST.get(k, (None, None))
        t, first = ms(k)
        cols = [f"**{nm}**" if k == ref else nm, canvas, "repo" if in_repo(k) else "results only",
                "-" if pa is None else f"{pa:.2f}", "-" if gf is None else f"{gf:.2f}",
                "-" if t is None else f"{t:.2f}" + (" (a)" if first else "")]
        if k == ref:
            cols += [",".join(map(str, a_bt["seedsA"])), f"{a_bt['A']:.2f} / {a_oc['A']:.2f}", "-", "", "-", ""]
        else:
            bt, oc = ped[(k, "bytetrack")], ped[(k, "ocsort")]
            sd = ",".join(map(str, bt["seedsB"]))
            sd += "" if oc["seedsB"] == bt["seedsB"] else " / " + ",".join(map(str, oc["seedsB"]))
            cols += [sd, f"{bt['B']:.2f} / {oc['B']:.2f}", _cell(bt), _cell(oc)]
        L.append("| " + " | ".join(cols) + " |")
    L += ["", "(a) Timed in the first Jetson session (2026-10-27, `deploy/jetson/results/latency_orin_nano.csv`); the other "
          "latencies come from the detector session (2026-11-13, `deploy/jetson/results/latency_orin_nano_detectors.csv`).",
          ""]
    gf_ref, t_ref = DET_COST[ref][1], ms(ref)[0]
    same = [k for k in arms if k != ref and k in DET_COST and DET_COST[k][1] <= 1.1 * gf_ref]
    if same:
        rs = [ped[(k, t)] for k in same for t in P.TRACKERS]
        ns = [f"{short(key(r))} {TRK_NAME[r['trk']]} ({r['d']:+.2f}, p {_p(r['p_ar_adj'])})" for r in rs
              if not r["label"].startswith(("win", "loss"))]
        L.append(f"- At most 10 % more FLOPs than {nm_ref} ({', '.join(short(k) for k in same)}): d from "
                 f"{min(r['d'] for r in rs):+.2f} to {max(r['d'] for r in rs):+.2f} over both trackers"
                 + (f"; not significant after Holm: {', '.join(ns)}." if ns else ", all significant after Holm."))
    dear = [k for k in arms if k != ref and k in DET_COST and k not in same and ms(k)[0]]
    if dear and t_ref:
        L.append(f"- More expensive detectors (FLOPs and Jetson latency relative to {nm_ref}; d ByteTrack / OC-SORT, bold: "
                 "significant after Holm): " + "; ".join(
                     f"{short(k)} {DET_COST[k][1] / gf_ref:.1f}x / {ms(k)[0] / t_ref:.1f}x: {sig(ped[(k, 'bytetrack')])} / "
                     f"{sig(ped[(k, 'ocsort')])}" for k in dear) + ".")
    L += ["", "## All contrasts", "",
          f"Equivalence margin {P.MARGINS[data]} HOTA. Holm families: class x tracker over the {m} contrasts of this page.",
          "", "| A | B | class | tracker | seeds A / B | HOTA A | HOTA B | d | 95 % CI | p_AR | Holm p_AR | TOST p (Holm) | "
          "label |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {full(ref)} | {full(key(r))} | {r['cls']} | {TRK_NAME[r['trk']]} | "
                 f"{','.join(map(str, r['seedsA']))} / {','.join(map(str, r['seedsB']))} | {r['A']:.2f} | {r['B']:.2f} | "
                 f"**{r['d']:+.2f}** | [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] | {_p(r['p_ar'])} | {_p(r['p_ar_adj'])} | "
                 f"{_p(r['p_tost_adj'])} | {r['label']} |")
    if all("sensitivity_drive" in r for r in rows):
        L += ["", "Sensitivity (reported, not deciding): thresholds cross-fitted leave-one-drive-out, bootstrap and exact "
              "randomisation over the 8 drives of the 13 test sequences. With 8 drives the smallest attainable p_AR is "
              f"2/256 = 0.0078, so with {m} contrasts per Holm family no drive-level difference can reach a Holm-adjusted p "
              "below 0.05; the unadjusted p_AR is shown.", "",
              "| B | class | tracker | d | 95 % CI | p_AR |", "|---|---|---|---|---|---|"]
        for r in rows:
            s_ = r["sensitivity_drive"]
            L.append(f"| {full(key(r))} | {r['cls']} | {TRK_NAME[r['trk']]} | {s_['d']:+.2f} | "
                     f"[{s_['ci95'][0]:+.2f}, {s_['ci95'][1]:+.2f}] | {_p(s_['p_ar'])} |")
    rep = [lat.get((DET_LATENCY_CSV, *k, 2)) for k in (("v8n", "416"), ref)]
    old = [lat.get((LATENCY_CSV, *k, 1)) for k in (("v8n", "416"), ref)]
    only = [short(k) for k in arms if not in_repo(k)]
    only = ", ".join(only[:-1]) + " and " + only[-1] if len(only) > 1 else "".join(only)
    L += ["", "## Notes", "",
          "- **Training.** Every arm: the DSEC-Det training split (24,168 frames), 9,060 iterations (6 epochs), from the "
          "family's public COCO checkpoint with the family's default learning rate, on the 416 x 320 letterbox canvas "
          "(the 448 x 352 and 640 x 480 arms on their own canvas); tracked with the fixed ByteTrack / OC-SORT settings over "
          "the threshold grid and scored with TrackEval like every arm of `RESULTS.md`. Seeds 1 and 2 of YOLOv8s and "
          "RT-DETR-L were trained on a cloud GPU with the same code and settings.",
          *([f"- **Results only.** The model code of {only} is not part of this repository: their rows are regenerated "
             "from `results/stats/`, but these detectors cannot be retrained with this code."] if only else []),
          "- **Cost.** Parameters and GFLOPs (2 x thop MACs) of the inference graph for one 3 x 320 x 416 input (the "
          "448 x 352 and 640 x 480 arms: their canvas), 8 classes. Repository models: `efpmot.models.flops`. The other "
          "families were measured the same way on their own implementations: NanoDet-Plus-m without its training-only "
          "auxiliary head, YOLO26n as the deployed one-to-one (NMS-free) graph.",
          "- **Latency.** Jetson Orin Nano, 15 W power mode, TensorRT 10.3 FP16: GPU compute time of the engine alone "
          "(trtexec median over 10 s; no pre-processing, NMS or tracking), canvas input and the family's raw head output "
          "(YOLO26n: one-to-one head with every anchor decoded, no top-k; RT-DETR-L: its 300 decoder queries). Full "
          "tables: `deploy/jetson/results/latency_orin_nano.md`."
          + (f" The detector session timed YOLOv8n and {nm_ref} again at its end ({rep[0]:.2f} and {rep[1]:.2f} ms); the "
             f"first session measured the same graphs at {old[0]:.2f} and {old[1]:.2f} ms." if all(rep + old) else ""), ""]
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L))
    return path
