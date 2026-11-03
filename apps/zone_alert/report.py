# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Markdown report of the zone-alert evaluation (results.json written by compare.py -> results.md)."""
from __future__ import annotations

ZONE_KINDS = ("far", "near")
MODEL_LABEL = {"plain": "plain", "s3x": "S3x"}


def fmt(value, digits: int = 2) -> str:
    """'-' for None, floats with `digits` decimals, everything else as str."""
    if value is None:
        return "-"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def table(header: list[str], rows: list[list]) -> list[str]:
    """Markdown table lines."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def _pooled_table(results: dict) -> list[str]:
    header = ["zone", "model", "GT episodes", "alerted", "recall", "median delay [s]", "p90 delay [s]",
              "false alerts", "duplicates", "unscored alerts", "loitering hit / GT", "false loitering"]
    rows = []
    for kind in ZONE_KINDS:
        for model in results["models"]:
            s = results["pooled"][kind][model]
            rows.append([kind, MODEL_LABEL[model], s["gt_episodes"], s["alerted"], fmt(s["recall"], 3),
                         fmt(s["median_delay_s"]), fmt(s["p90_delay_s"]), s["false_alerts"], s["duplicates"],
                         s["unscored"], f"{s['loitering_hit']} / {s['gt_loitering']}", s["loitering_false"]])
    return table(header, rows)


def _entries_table(results: dict) -> list[str]:
    header = ["zone", "model", "episodes starting during the clip", "alerted", "recall", "median delay [s]"]
    rows = []
    for kind in ZONE_KINDS:
        for model in results["models"]:
            s = results["pooled_entries_only"][kind][model]
            rows.append([kind, MODEL_LABEL[model], s["gt_episodes"], s["alerted"], fmt(s["recall"], 3),
                         fmt(s["median_delay_s"])])
    return table(header, rows)


def _sensitivity_table(results: dict) -> list[str]:
    header = ["zone", "model", "episodes", "alerted", "recall", "median delay [s]", "false alerts", "duplicates"]
    rows = []
    for kind in ZONE_KINDS:
        for model in results["models"]:
            s = results["pooled_iou03"][kind][model]
            rows.append([kind, MODEL_LABEL[model], s["gt_episodes"], s["alerted"], fmt(s["recall"], 3),
                         fmt(s["median_delay_s"]), s["false_alerts"], s["duplicates"]])
    return table(header, rows)


def _per_zone_table(results: dict) -> list[str]:
    header = ["sequence", "zone (kind)", "GT height, median [px source / canvas]", "model", "thr", "episodes",
              "unscored episodes", "alerted", "median delay [s]", "false (outside / distractor / none)", "dup.",
              "unscored alerts", "loitering hit / GT", "false loitering"]
    rows = []
    for seq in results["seqs"]:
        for zone in results["zones"][seq]:
            size = results["sizes"][seq][zone["name"]]
            height = f"{fmt(size['median_height_px'], 0)} / {fmt(size['median_height_canvas_px'], 0)}"
            for model in results["models"]:
                s = results["per_zone"][seq][zone["name"]][model]
                false = (f"{s['false_alerts']} ({s['false_person_outside']} / {s['false_distractor']} / "
                         f"{s['false_no_person']})")
                rows.append([seq[:8], f"{zone['name']} ({zone['kind']})", height, MODEL_LABEL[model],
                             f"{results['thresholds'][model][seq]:g}", s["gt_episodes"], s["unscored_episodes"],
                             s["alerted"], fmt(s["median_delay_s"]), false, s["duplicates"], s["unscored"],
                             f"{s['loitering_hit']} / {s['gt_loitering']}", s["loitering_false"]])
    return table(header, rows)


def _consistency_table(results: dict) -> list[str]:
    header = ["sequence", "model", "frames with identical tracks", "max abs. difference x, y, w, h [px]",
              "max abs. difference conf", "benchmark file"]
    rows = []
    for seq in results["seqs"]:
        for model in results["models"]:
            c = results["consistency"][seq][model]
            rows.append([seq[:8], MODEL_LABEL[model], c["exact_frames"], c["max_abs_diff_xywh"],
                         c["max_abs_diff_conf"], f"`{c['benchmark_file']}`"])
    return table(header, rows)


def _entry_kinds(results: dict, kind: str) -> tuple[int, int]:
    """(walk-ins, appeared inside) among the scored episodes of `kind` that start after the first frame."""
    walk, appeared = 0, 0
    for seq in results["seqs"]:
        for ep in results["episodes"][seq]:
            if ep["scored"] and ep["zone_kind"] == kind and not ep["initial"]:
                appeared += ep["appeared_inside"]
                walk += not ep["appeared_inside"]
    return walk, appeared


def _findings(results: dict) -> list[str]:
    far_plain, far_s3x = results["pooled"]["far"]["plain"], results["pooled"]["far"]["s3x"]
    sens_plain, sens_s3x = results["pooled_iou03"]["far"]["plain"], results["pooled_iou03"]["far"]["s3x"]
    near_plain, near_s3x = results["pooled"]["near"]["plain"], results["pooled"]["near"]["s3x"]
    per_seq = []
    for seq in results["seqs"]:
        for zone in results["zones"][seq]:
            if zone["kind"] == "far":
                z = results["per_zone"][seq][zone["name"]]
                per_seq.append(f"{seq[:8]} {z['plain']['alerted']} vs {z['s3x']['alerted']} of "
                               f"{z['plain']['gt_episodes']}")
    return [
        f"- Far zones, strict attribution: plain alerted {far_plain['alerted']} and S3x {far_s3x['alerted']} of "
        f"{far_plain['gt_episodes']} GT intrusion episodes (per sequence, plain vs S3x: {'; '.join(per_seq)}). "
        "This is not a clear gain for S3x: the difference is a few episodes or none.",
        f"- Far zones, alert quality (plain vs S3x): false alerts {far_plain['false_alerts']} vs "
        f"{far_s3x['false_alerts']}, duplicates {far_plain['duplicates']} vs {far_s3x['duplicates']}, false loitering "
        f"{far_plain['loitering_false']} vs {far_s3x['loitering_false']}. With the post-hoc IoU 0.3 association: "
        f"alerted {sens_plain['alerted']} vs {sens_s3x['alerted']}, false {sens_plain['false_alerts']} vs "
        f"{sens_s3x['false_alerts']}.",
        f"- Delay in the far zones: median {fmt(far_plain['median_delay_s'])} s (plain) and "
        f"{fmt(far_s3x['median_delay_s'])} s (S3x), p90 {fmt(far_plain['p90_delay_s'])} s and "
        f"{fmt(far_s3x['p90_delay_s'])} s.",
        f"- Near zones (control): plain {near_plain['alerted']} and S3x {near_s3x['alerted']} of "
        f"{near_plain['gt_episodes']} episodes, false alerts {near_plain['false_alerts']} and "
        f"{near_s3x['false_alerts']}.",
    ]


def write_results_md(results: dict, path: str) -> None:
    """Write results.md from the results dict of compare.py."""
    meta = results["meta"]
    seqs = ", ".join(s[:8] for s in results["seqs"])
    walk, appeared = _entry_kinds(results, "far")
    lines = [
        "# Zone alerts on MOT17: plain YOLOv8n vs YOLOv8n + S3x", "",
        f"Generated by `apps/zone_alert/compare.py` on {meta['created']}. Evaluation halves of {seqs} (frames "
        "n//2+2..n; the detectors were trained on the first halves). ByteTrack, seed 0, cross-fitted threshold per "
        f"model and sequence. Hysteresis K = {meta['k']} frames.", "",
        "Definitions (details in the docstring of `compare.py`):", "",
        "- **GT episode:** a GT pedestrian's foot point is inside the zone for K frames, using every annotated frame "
        "whatever the visibility. Its start is the first of those frames and its confirmation the K-th. The episode is "
        f"**scored** when the person is inside and at least {meta['vis_min']:g} visible on at least K of its frames; "
        "episodes of people hidden for the whole stay are not scored.",
        f"- **Hit:** an INTRUSION whose track is matched (IoU >= {meta['match_iou']}) to the episode's person on one "
        "of the K frames up to the alert. **Delay** = alert frame - GT confirmation frame (0 = as early as a perfect "
        "tracker; negative = the model's foot point confirmed the entry before the GT foot point did).",
        "- **False alert:** the alerting track is matched, on those K frames, to a GT person outside the zone, to a "
        "MOT17 'distractor' or 'reflection' box, or to nothing. Alerts on a real person without a scored episode "
        "(including MOT17 'static person' and 'person on vehicle' boxes) are **unscored**, neither hit nor false.",
        "- **Duplicate:** a further INTRUSION for an already-alerted episode (usually a new track id after an "
        "identity switch).", "",
        "## Far vs near zone (both sequences pooled)", "",
        *_pooled_table(results), "",
        "The table above includes episodes that already exist at the first evaluation frame (people standing in a zone "
        "when the clip starts). The next table keeps only the episodes that start after the first frame. These are "
        f"not all walk-ins: of the {walk + appeared} far-zone ones, {walk} are people seen outside the zone before "
        f"and {appeared} are people whose annotation starts inside the zone.", "",
        *_entries_table(results), "",
        f"Sensitivity (post hoc: added after the primary table had been computed): the same alerts attributed with "
        f"IoU >= {meta['sens_iou']} instead of {meta['match_iou']}. Many strict far-zone false alerts are loose boxes "
        "(IoU 0.2-0.5) on small people in a crowd; with the looser association such an alert counts as a hit.", "",
        *_sensitivity_table(results), "",
        "## Per sequence and zone", "",
        *_per_zone_table(results), "",
        "## Same tracks as the benchmark", "",
        "The app's tracks compared with the stored benchmark tracker output on every evaluation frame, all columns. "
        "The stored file keeps 2 decimals for x, y, w, h and 4 for the confidence.", "",
        *_consistency_table(results), "",
        "Timing is measured by `bench.py` (median of 3 runs, see `timing.md`).", "",
        "## What the numbers say", "",
        *_findings(results), "",
        "## Notes (read before quoting the numbers)", "",
        "- Two sequences only (MOT17-04 and MOT17-09 evaluation halves: 17.5 s and 8.7 s), one training seed (0), one "
        "threshold per model and sequence from leave-one-sequence-out cross-fitting (never tuned on the scored "
        "sequence). No confidence intervals: the episode counts are small, so differences of a few episodes are not "
        "evidence on their own.",
        "- The far zones are where S3x (finer horizontal stride on P3 for small, narrow objects) is expected to "
        "matter; the near zones are the control. In MOT17-09 the far-zone people are still large (see the height "
        "column), so that zone is a weak test.",
        "- The zones were drawn from scene geometry before any model output was seen and were not changed afterwards. "
        "One consequence: the near zone of MOT17-04 ends 5 px above the image edge, so people whose feet are cut off "
        "by the bottom edge are outside it (the same for GT and models).",
        "- Delays include time a person spends hidden behind others after entering: the GT stay starts when the "
        "annotated foot point enters, whether or not the person can be seen.",
        "- Rule history: the first version of this report credited a hit when the track had matched the "
        "person at any frame of the episode (even after the alert), counted alerts on 'static person' boxes as false, "
        "and built GT episodes only from frames with visibility >= 0.5. A review found that these rules credited some "
        "hits to the wrong person and produced negative delays from the visibility filter, so all three were changed "
        "(this version). The first version reported far-zone alerted 38 vs 39 of 47, false alerts 24 vs 21 and "
        "duplicates 47 vs 37 (plain vs S3x).",
        "- Offline: frames are read from disk at full speed and timed on a desktop GPU, not the Jetson; the "
        "end-to-end edge pipeline (camera capture -> TensorRT -> tracker -> alert) has not been measured yet.", "",
        "Licence: MOT17 frames and annotations CC BY-NC-SA 3.0 (motchallenge.net); results shared under the same "
        "licence.",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
