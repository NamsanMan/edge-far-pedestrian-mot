# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Strip training checkpoints into release files (bare fp32 state dicts) and write SHA256SUMS.txt.

    python -B scripts/package_weights.py [--out release_assets] [--runs_dir DIR] [--runs TAG ...]

Release file <out>/<family>_<dataset>_<W>x<H>_s<seed>.pt (plain families get '_plain', e.g.
yolov8n_plain_dsec_416x320_s0.pt) = torch.save(state_dict) with every floating tensor in fp32; it loads with
torch.load(weights_only=True) and efpmot.models.build(family, 8, path). Default runs: the four published release files
(efpmot.release.RELEASES); --runs_dir defaults to outputs.runs. Each file is checked to load strictly into the efpmot model.
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from efpmot import dataset_of_tag, paths  # noqa: E402
from efpmot.models import build  # noqa: E402
from efpmot.release import RELEASES, RUN_TO_FILE  # noqa: E402


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def release_name(tag: str, family: str, hw, seed: int) -> str:
    if tag in RUN_TO_FILE:
        return RUN_TO_FILE[tag]
    fam = family if "_s3" in family else family + "_plain"
    return f"{fam}_{dataset_of_tag(tag)}_{int(hw[1])}x{int(hw[0])}_s{int(seed)}.pt"


def strip(src: str, tag: str, out_dir: str) -> tuple[str, dict]:
    """training checkpoint -> release file (bare fp32 state dict, checked to load strictly into the efpmot model)"""
    ck = torch.load(src, map_location="cpu", weights_only=False)
    a = ck["args"]
    sd = {k: (v.float() if torch.is_floating_point(v) else v).contiguous() for k, v in ck["model"].items()}
    build(a["family"], 8).model.load_state_dict(sd, strict=True)
    fn = release_name(tag, a["family"], a["hw"], a["seed"])
    if fn in RELEASES:
        m = RELEASES[fn]
        assert (m["family"], tuple(m["hw"]), m["seed"]) == (a["family"], tuple(int(v) for v in a["hw"]), int(a["seed"])), (fn, a)
    dst = os.path.join(out_dir, fn)
    torch.save(sd, dst)
    torch.load(dst, map_location="cpu", weights_only=True)
    return fn, a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="release_assets", help="output folder (relative = repository root)")
    ap.add_argument("--runs_dir", default=None, help="folder holding <tag>/last.pth (default outputs.runs)")
    ap.add_argument("--runs", nargs="+", default=[m["run"] for m in RELEASES.values()])
    a = ap.parse_args()
    out = paths.resolve(a.out)
    runs_dir = os.path.abspath(a.runs_dir) if a.runs_dir else paths.require("outputs.runs")
    os.makedirs(out, exist_ok=True)
    for tag in a.runs:
        fn, args = strip(os.path.join(runs_dir, tag, "last.pth"), tag, out)
        print(f"{tag}: {args['family']} hw {list(args['hw'])} epochs {args['epochs']} -> {fn}", flush=True)
    files = sorted(f for f in os.listdir(out) if f.endswith(".pt"))
    with open(os.path.join(out, "SHA256SUMS.txt"), "w", newline="\n") as f:
        for fn in files:
            f.write(f"{sha256(os.path.join(out, fn))}  {fn}\n")
    print(f"{len(files)} files, SHA256SUMS.txt -> {out}", flush=True)


if __name__ == "__main__":
    main()
