# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Cross-fitted paired comparison of two detector arms, or the whole published table.

    python -B scripts/compare.py --one dsec v8n 416s3x 416 --cls pedestrian --trk bytetrack
           [--rule cross|best|fixed] [--B 10000] [--margin M] [--source auto|stats|trackeval] [--json OUT]
    python -B scripts/compare.py --table results/contrasts.yaml [--B 10000] [--source auto|stats|trackeval]
    python -B scripts/compare.py --detectors results/contrasts_detectors.yaml [--B 10000] [--source ...]
--one prints the contrast as JSON. --table writes results/summary/<dataset>.json, results/summary.csv and
results/RESULTS.md next to the contrasts file. --detectors (contrasts whose arm B is another detector, key det_b; their
own Holm families) writes summary_detectors/<dataset>.json, summary_detectors.csv and DETECTORS.md next to its contrasts
file. source: 'stats' = results/stats (shipped), 'trackeval' = <outputs.results>/<dataset>_eval (your own runs),
'auto' = stats when present.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from efpmot.eval import protocol as P  # noqa: E402
from efpmot.eval import table as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--one", nargs=4, metavar=("DATA", "DET", "A", "B"))
    g.add_argument("--table", metavar="CONTRASTS_YAML")
    g.add_argument("--detectors", metavar="CONTRASTS_YAML")
    ap.add_argument("--cls", default="car")
    ap.add_argument("--trk", default="bytetrack", choices=P.TRACKERS)
    ap.add_argument("--rule", default="cross", choices=P.RULES)
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--margin", type=float, default=None, help="equivalence margin (default: the frozen margin of the dataset)")
    ap.add_argument("--source", default="auto", choices=["auto", "stats", "trackeval"])
    ap.add_argument("--json", default=None, help="--one: also write the JSON to this file")
    a = ap.parse_args()
    if a.one:
        data, det, va, vb = a.one
        r = P.one(data, det, va, vb, cls=a.cls, trk=a.trk, rule=a.rule, B=a.B, margin=a.margin, source=a.source)
        s = json.dumps(r, indent=1)
        print(s)
        if a.json:
            with open(a.json, "w", encoding="utf-8") as f:
                f.write(s + "\n")
        return
    if a.detectors:
        out = os.path.dirname(os.path.abspath(a.detectors))
        rows = T.run_table(T.load_contrasts(a.detectors), B=a.B, source=a.source)
        files = T.write_json(rows, os.path.join(out, "summary_detectors"))
        files.append(T.write_detectors_csv(rows, os.path.join(out, "summary_detectors.csv")))
        files.append(T.write_detectors_md(rows, os.path.join(out, "DETECTORS.md"), a.B))
        print(f"{len(rows)} rows ->", ", ".join(_rel(f) for f in files))
        return
    out = os.path.dirname(os.path.abspath(a.table))
    rows = T.run_table(T.load_contrasts(a.table), B=a.B, source=a.source)
    files = T.write_json(rows, os.path.join(out, "summary"))
    files.append(T.write_csv(rows, os.path.join(out, "summary.csv")))
    files.append(T.write_md(rows, os.path.join(out, "RESULTS.md"), a.B))
    for v in T.verdicts(rows):
        print(f"{v['dataset']:9s} {v['prediction']} {v['cls']:10s} {v['A_name']} vs {v['B_name']}: {v['verdict']}")
    print(f"{len(rows)} rows ->", ", ".join(_rel(f) for f in files))


def _rel(f: str) -> str:
    """path relative to the working directory when possible (os.path.relpath fails across Windows drives)"""
    try:
        return os.path.relpath(f)
    except ValueError:
        return f


if __name__ == "__main__":
    main()
