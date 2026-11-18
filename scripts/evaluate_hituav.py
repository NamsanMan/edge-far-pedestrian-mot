# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""HIT-UAV detection evaluation: export the val + test detections of trained checkpoints, evaluate them (efpmot/eval/detection.py,
rules in efpmot/data/hituav.py) and write one small JSON per run; regenerate results/HITUAV.md.

    python -B scripts/evaluate_hituav.py TAG [TAG ...] [--runs_dir DIR] [--weights FILE] [--export] [--device cuda]
           [--cache_root DIR] [--out results/hituav] [--detections TEXT]
    python -B scripts/evaluate_hituav.py --table [--out results/hituav] [--md results/HITUAV.md]
    python -B scripts/evaluate_hituav.py --compare REF_DIR [--out results/hituav]

TAG = run tag hit_<det>_<variant>[<fold>]_s<seed>, e.g. hit_v8n_416s3xrecA_s0; the split is the tag's fold marker (none =
official). Glob patterns ("hit_v8n_416*rec*") are matched against the run folders of --runs_dir. Detections are read from
<outputs.cache>/<tag>/hituav/{val,test}.npz; when they are missing, or with --export, they are first exported from --weights or
<runs_dir>/<tag>/last.pth (runs_dir default outputs.runs; else the release file of the run) with the export rule of the published
runs: FP16 on CUDA, conf 0.001, class-wise NMS 0.65, at most 300 per image. --cache_root DIR reads existing caches from
DIR/<tag>/hituav/ instead and never exports. The family, canvas, seed and epochs written with the results come from the checkpoint
(training args, or efpmot.release for a release file); a run whose family is not in efpmot.models (e.g. the P2 arm of the
original research code, evaluated from its existing caches) is written with code_included = false. --detections stores a note on
where the caches come from.
--compare: for every <out>/<tag>.json that has a REF_DIR/<tag>.json, the absolute differences of AP50 and AP50:95 (test and val),
of the operating thresholds and of test P / R / F1 at them; per-run rows and the maxima.
"""
import argparse
import fnmatch
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from efpmot import paths  # noqa: E402
from efpmot.data import hituav as H  # noqa: E402
from efpmot.eval import detection as D  # noqa: E402
from efpmot.eval import hituav_table as T  # noqa: E402
from efpmot.release import release_meta, release_path  # noqa: E402

OUT = os.path.join("results", "hituav")
MD = os.path.join("results", "HITUAV.md")
META_KEYS = ("tag", "split", "arm", "family", "hw", "seed", "epochs", "code_included", "detections")


def checkpoint_args(path: str) -> dict:
    """family / hw / seed / epochs of a release file (efpmot.release) or from the training args of a checkpoint"""
    rel = release_meta(path)
    if rel:
        return {k: rel[k] for k in ("family", "hw", "seed", "epochs")}
    from efpmot.models import load_trusted
    ck = load_trusted(path)
    src = ck.get("args") if isinstance(ck, dict) else None
    return {k: src[k] for k in ("family", "hw", "seed", "epochs") if k in src} if isinstance(src, dict) else {}


def weights_for(tag: str, runs_dir: str | None, weights: str | None) -> str | None:
    if weights:
        return weights
    if runs_dir:
        p = os.path.join(runs_dir, tag, "last.pth")
        if os.path.isfile(p):
            return p
    return release_path(tag)


def expand(tags, runs_dir: str | None) -> list[str]:
    out = []
    for t in tags:
        if any(ch in t for ch in "*?["):
            hit = sorted(d for d in os.listdir(runs_dir) if fnmatch.fnmatch(d, t)) if runs_dir and os.path.isdir(runs_dir) else []
            if not hit:
                print(f"(no run folder in {runs_dir} matches {t})", flush=True)
            out += hit
        else:
            out.append(t)
    return list(dict.fromkeys(out))


def with_meta(r: dict, meta: dict) -> dict:
    """result dict with the run fields first (META_KEYS order)"""
    head = {k: meta[k] for k in META_KEYS if k in meta and meta[k] is not None}
    return {**head, **{k: v for k, v in r.items() if k not in head}}


def write_json(r: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, r["tag"] + ".json")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(r, f, indent=1)
        f.write("\n")
    return p


def _fmt(v, d=2):
    return "-" if v is None else f"{v:.{d}f}"


def run_fields(tag: str, args: dict, code_included: bool, detections: str | None) -> dict:
    split = H.split_of_tag(tag)
    hw = [int(v) for v in args["hw"]]
    return dict(tag=tag, split=split, arm=T.arm_of(tag, args["family"], hw), family=args["family"], hw=hw, seed=int(args["seed"]),
                epochs=int(args["epochs"]), code_included=code_included, detections=detections)


def evaluate_tags(a, tags: list[str]) -> int:
    import torch

    from efpmot.models import FAMILIES, build
    from efpmot.train import export_dets
    runs_dir = os.path.abspath(a.runs_dir) if a.runs_dir else paths.get("outputs.runs")
    cache_dir = os.path.abspath(a.cache_root) if a.cache_root else paths.require("outputs.cache")
    out_dir, n_ok = paths.resolve(a.out), 0
    for tag in tags:
        split = H.split_of_tag(tag)
        w = weights_for(tag, runs_dir, a.weights)
        args = checkpoint_args(w) if w else {}
        if not all(k in args for k in ("family", "hw", "seed", "epochs")):
            print(f"SKIP {tag}: no checkpoint with training args ({w})", flush=True)
            continue
        have = all(os.path.isfile(os.path.join(cache_dir, tag, "hituav", p + ".npz")) for p in H.EXPORT_PARTS)
        note = a.detections
        if a.cache_root:
            if not have:
                print(f"SKIP {tag}: no cache in {cache_dir} (--cache_root never exports)", flush=True)
                continue
        elif a.export or not have:
            if args["family"] not in FAMILIES:
                print(f"SKIP {tag}: family {args['family']} is not in this repository (evaluate its existing caches with "
                      "--cache_root)", flush=True)
                continue
            ad = build(args["family"], 8, w)
            export_dets(ad, tag, "hituav", tuple(int(v) for v in args["hw"]), device=a.device, split=split)
            del ad
            if a.device.startswith("cuda"):
                note = note or f"FP16 export by scripts/evaluate_hituav.py on {torch.cuda.get_device_name()}"
                torch.cuda.empty_cache()
            else:
                note = note or "FP32 CPU export by scripts/evaluate_hituav.py"
        r = D.evaluate_hituav(tag, split, cache_root=cache_dir)
        r = with_meta(r, run_fields(tag, args, args["family"] in FAMILIES, note))
        p = write_json(r, out_dir)
        o = r["op"]["pedestrian"]
        print(f"{tag} [{split}, {r['arm']}]: test person AP50 {_fmt(r['ap']['test']['pedestrian']['AP50'])} AP50:95 "
              f"{_fmt(r['ap']['test']['pedestrian']['AP50_95'])}; val-F1 thr {_fmt(o['thr'], 4)} -> test F1 "
              f"{_fmt(o.get('test', {}).get('F1'))} -> {p}", flush=True)
        n_ok += 1
    print(f"HITUAV_EVAL_DONE {n_ok} of {len(tags)} runs -> {out_dir}", flush=True)
    return n_ok


def compare(out_dir: str, ref_dir: str) -> None:
    fields = []
    for sp in ("test", "val"):
        for c in H.EVAL_CLASSES:
            fields += [(f"{sp} {c} AP50", ("ap", sp, c, "AP50")), (f"{sp} {c} AP50:95", ("ap", sp, c, "AP50_95"))]
    for c in H.EVAL_CLASSES:
        fields += [(f"{c} thr", ("op", c, "thr"))] + [(f"test {c} {k}", ("op", c, "test", k)) for k in ("P", "R", "F1")]

    def get(d, ks):
        for k in ks:
            d = d.get(k) if isinstance(d, dict) else None
        return d
    rows, worst = [], {lab: (0.0, None) for lab, _ in fields}
    for f in sorted(os.listdir(out_dir)):
        ref = os.path.join(ref_dir, f)
        if not (f.startswith("hit_") and f.endswith(".json") and os.path.isfile(ref)):
            continue
        with open(os.path.join(out_dir, f), encoding="utf-8") as fh:
            r = json.load(fh)
        with open(ref, encoding="utf-8") as fh:
            q = json.load(fh)
        diffs = {}
        for lab, ks in fields:
            x, y = get(r, ks), get(q, ks)
            diffs[lab] = 0.0 if x is None and y is None else (float("inf") if x is None or y is None else abs(x - y))
            if diffs[lab] > worst[lab][0]:
                worst[lab] = (diffs[lab], r["tag"])
        rows.append((r["tag"], diffs))
    if not rows:
        raise SystemExit(f"no result in {out_dir} has a reference in {ref_dir}")
    print(f"{'run':28s} {'pedAP50':>9s} {'pedAP5095':>9s} {'pedThr':>9s} {'carAP50':>9s} {'carThr':>9s} {'max(any)':>9s}")
    for tag, d in rows:
        print(f"{tag:28s} " + " ".join(f"{d[k]:9.2g}" for k in ("test pedestrian AP50", "test pedestrian AP50:95", "pedestrian thr",
                                                                 "test car AP50", "car thr")) + f" {max(d.values()):9.2g}")
    print(f"\n{len(rows)} runs compared with {ref_dir}; {sum(1 for _, d in rows if max(d.values()) == 0)} identical in every field")
    print("max |difference| per field (run):")
    for lab, _ in fields:
        print(f"  {lab:24s} {worst[lab][0]:.6g}" + (f"  ({worst[lab][1]})" if worst[lab][1] else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tags", nargs="*", help="run tags or glob patterns over --runs_dir")
    ap.add_argument("--runs_dir", default=None, help="folder of <tag>/last.pth (default outputs.runs)")
    ap.add_argument("--weights", default=None, help="checkpoint or release file (one tag only)")
    ap.add_argument("--export", action="store_true", help="export even when the cache exists")
    ap.add_argument("--device", default="cuda", help="export device: cuda (FP16, the published rule) or cpu (FP32)")
    ap.add_argument("--cache_root", default=None, help="read existing caches <DIR>/<tag>/hituav/ (no export)")
    ap.add_argument("--out", default=OUT, help="result folder (relative = repository root; default results/hituav)")
    ap.add_argument("--detections", default=None, help="note on where the detection caches come from (stored in the JSON)")
    ap.add_argument("--table", action="store_true", help="regenerate --md from the JSON files of --out")
    ap.add_argument("--md", default=MD, help="results page (default results/HITUAV.md)")
    ap.add_argument("--compare", default=None, metavar="REF_DIR", help="differences of <out>/<tag>.json to REF_DIR/<tag>.json")
    a = ap.parse_args()
    if a.weights and len(a.tags) != 1:
        ap.error("--weights needs exactly one tag")
    if a.cache_root and a.export:
        ap.error("--cache_root reads existing caches; it cannot be combined with --export")
    runs_dir = os.path.abspath(a.runs_dir) if a.runs_dir else paths.get("outputs.runs")
    tags = expand(a.tags, runs_dir)
    for t in tags:
        if not t.startswith("hit_"):
            ap.error(f"{t}: HIT-UAV run tags start with 'hit_'")
        H.split_of_tag(t)
    if tags:
        evaluate_tags(a, tags)
    if a.compare:
        compare(paths.resolve(a.out), a.compare)
    if a.table:
        print("wrote", T.write(paths.resolve(a.out), paths.resolve(a.md)), flush=True)
    if not (tags or a.compare or a.table):
        ap.error("give run tags, --table or --compare")


if __name__ == "__main__":
    main()
