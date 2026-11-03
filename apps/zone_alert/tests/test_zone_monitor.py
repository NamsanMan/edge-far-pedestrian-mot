# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Synthetic tests of the alert logic (no GPU, no data): foot point, polygon edge, hysteresis, loitering, exit,
missing-track closing, overlapping zones, time stamps and ignore masks (incl. boxes jittering across a mask edge);
APPROACH warnings (lead time, parallel / away / standing, box jitter, box jumps, re-arming, no re-warning while milling
at the edge, drawing hold and countdown, missing heights), the direction filter (allow, block, fail-safe, alarm edge),
tripwires (both directions, jitter, segment ends, starting side, walking around an end) and the banner priority;
plus replays of the cached model runs
(cache/, skipped when absent) proving that zone files without "ignore", with a mask no foot point reaches, and without
the motion keys give exactly the stored events, and that the v2 zone files only add events or relabel entries.

usage (repository root): python -B apps/zone_alert/tests/test_zone_monitor.py      (or: pytest apps/zone_alert/tests)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import zone_alert as za  # noqa: E402

INSIDE, OUTSIDE = (50.0, 50.0), (50.0, 150.0)
SQUARE = za.Zone("z", np.array([[0, 0], [100, 0], [100, 100], [0, 100]], np.float32), dwell_s=1.0)


def simulate(feet: list, fps: float = 10.0, max_missing: int | None = 15) -> list[tuple[str, int, int]]:
    """One track (id 7) with a foot point per frame (None = not reported) -> [(type, frame, entered), ...]."""
    monitor = za.ZoneMonitor([SQUARE], fps, k=5, max_missing=max_missing, frame0=0)
    events = []
    for frame, xy in enumerate(feet):
        reported = {} if xy is None else {7: xy}
        events += [(e.type, e.frame, e.entered) for e in monitor.update(frame, reported)]
    return events


def test_foot_point_is_bottom_centre():
    assert za.foot_point(np.array([10, 20, 30, 80], np.float32)) == (20.0, 80.0)


def test_polygon_edge_counts_as_inside():
    assert SQUARE.contains((100.0, 100.0))
    assert not SQUARE.contains((100.5, 50.0))


def test_entry_after_k_frames_then_loitering():
    # INTRUSION at the 5th inside frame; LOITERING when frame - first inside frame >= dwell_s * fps = 10
    assert simulate([OUTSIDE] * 5 + [INSIDE] * 20) == [("INTRUSION", 9, 5), ("LOITERING", 15, 5)]


def test_jitter_shorter_than_k_never_enters():
    assert simulate([OUTSIDE] * 3 + [INSIDE] * 4 + [OUTSIDE] + [INSIDE] * 4 + [OUTSIDE] * 5) == []


def test_exit_needs_k_outside_frames_and_resets_loitering():
    feet = [INSIDE] * 6 + [OUTSIDE] * 4 + [INSIDE] * 3 + [OUTSIDE] * 5 + [INSIDE] * 16
    assert simulate(feet) == [("INTRUSION", 4, 0), ("LOITERING", 10, 0), ("EXIT", 17, 0),
                              ("INTRUSION", 22, 18), ("LOITERING", 28, 18)]


def test_silent_track_kept_for_max_missing_frames():
    # dwell is wall-clock time, so LOITERING fires on the first report after the gap
    assert simulate([INSIDE] * 5 + [None] * 15 + [INSIDE] * 2) == [("INTRUSION", 4, 0), ("LOITERING", 20, 0)]


def test_silent_track_closed_after_max_missing_frames():
    assert simulate([INSIDE] * 5 + [None] * 16 + [INSIDE] * 5) == [("INTRUSION", 4, 0), ("EXIT", 20, 0),
                                                                   ("INTRUSION", 25, 21)]


def test_missing_frames_do_not_break_the_inside_run():
    assert simulate([INSIDE] * 3 + [None] * 4 + [INSIDE] * 2) == [("INTRUSION", 8, 0)]


def test_ground_truth_mode_never_closes_a_stay():
    feet = [INSIDE] * 5 + [None] * 40 + [OUTSIDE] * 5
    assert simulate(feet, max_missing=None) == [("INTRUSION", 4, 0), ("LOITERING", 45, 0), ("EXIT", 49, 0)]


def test_overlapping_zones_are_independent():
    big = za.Zone("z2", np.array([[40, 40], [200, 40], [200, 200], [40, 200]], np.float32), dwell_s=100.0)
    monitor = za.ZoneMonitor([SQUARE, big], 10.0, frame0=0)
    events = []
    for frame in range(6):
        events += [(e.type, e.zone, e.frame) for e in monitor.update(frame, {1: INSIDE})]
    assert events == [("INTRUSION", "z", 4), ("INTRUSION", "z2", 4)]


def test_event_time_is_measured_from_the_first_frame():
    monitor = za.ZoneMonitor([SQUARE], 30.0, frame0=527)
    first = [e for frame in range(527, 540) for e in monitor.update(frame, {3: INSIDE})][0]
    assert (first.frame, first.t, first.entered) == (531, round(4 / 30, 3), 527)


# ------------------------------------------------------------------------------------------------- ignore masks
MASK = za.IgnoreRegion("post", np.array([[40, 40], [60, 40], [60, 60], [40, 60]], np.float32), note="test")
IN_MASK, IN_ZONE = (50.0, 50.0), (20.0, 20.0)      # both inside SQUARE; only IN_MASK inside MASK


def simulate_masked(feet: list, ignore: list | None = None, fps: float = 10.0) -> tuple[list, list]:
    """Like simulate() with ignore masks -> ([(type, frame, entered), ...], [masked flag per frame])."""
    monitor = za.ZoneMonitor([SQUARE], fps, k=5, max_missing=15, frame0=0, ignore=[MASK] if ignore is None else ignore)
    events, flags = [], []
    for frame, xy in enumerate(feet):
        reported = {} if xy is None else {7: xy}
        events += [(e.type, e.frame, e.entered) for e in monitor.update(frame, reported)]
        flags.append(monitor.is_masked(7))
    return events, flags


def test_masked_track_raises_no_intrusion_or_loitering():
    events, flags = simulate_masked([IN_MASK] * 60)          # 6 s inside the zone, all of it on the mask
    assert events == [] and flags[3] is False and all(flags[4:])


def test_track_leaving_the_mask_into_a_zone_alerts_after_hysteresis():
    # on the mask for frames 0..9 (counts as outside the zone); in the zone from frame 10 -> INTRUSION at the K-th
    # inside frame (14), entered 10; LOITERING 10 frames after entry. The 'masked' drawing flag clears at frame 14 too.
    events, flags = simulate_masked([IN_MASK] * 10 + [IN_ZONE] * 20)
    assert events == [("INTRUSION", 14, 10), ("LOITERING", 20, 10)]
    assert flags[13] is True and flags[14] is False


OFF_MASK = (62.0, 50.0)                              # 2 px right of the mask edge, still inside SQUARE


def test_box_jittering_across_the_mask_edge_never_alerts():
    # off the mask 1 frame in 4: without the mask this track alerts; with it, no K consecutive in-zone frames
    feet = ([IN_MASK] * 3 + [OFF_MASK]) * 15
    assert simulate(feet) == [("INTRUSION", 4, 0), ("LOITERING", 10, 0)]
    assert simulate_masked(feet)[0] == []


def test_new_track_with_one_jitter_frame_on_the_mask_never_alerts():
    # a fresh id (e.g. after an identity switch) with one off-mask frame among its first 5
    feet = [IN_MASK] * 2 + [OFF_MASK] + [IN_MASK] * 40
    events, flags = simulate_masked(feet)
    assert events == [] and flags[-1] is True


def test_track_stepping_onto_the_mask_briefly_keeps_its_stay():
    events, _ = simulate_masked([IN_ZONE] * 6 + [IN_MASK] * 4 + [IN_ZONE] * 6)
    assert events == [("INTRUSION", 4, 0), ("LOITERING", 10, 0)]


def test_alerted_track_that_settles_on_the_mask_is_closed():
    events, _ = simulate_masked([IN_ZONE] * 6 + [IN_MASK] * 20)
    assert events == [("INTRUSION", 4, 0), ("EXIT", 10, 0)]


def test_no_ignore_list_is_identical_to_the_plain_monitor():
    feet = [OUTSIDE] * 3 + [IN_MASK] * 12 + [None] * 3 + [IN_ZONE] * 8 + [OUTSIDE] * 6 + [IN_MASK] * 20
    assert simulate_masked(feet, ignore=[])[0] == simulate(feet)


def test_zone_file_without_ignore_gives_no_masks():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert za.load_ignore(os.path.join(here, "zones", "MOT17-04.json")) == []


def test_mask_without_note_is_refused():
    import tempfile
    cfg = {"zones": [], "ignore": [{"name": "x", "polygon": [[0, 0], [1, 0], [1, 1]]}]}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "z.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh)
        try:
            za.load_ignore(path)
        except ValueError:
            return
    raise AssertionError("a mask without 'note' was accepted")


def _replay_cached(extra_ignore: list) -> None:
    """Replay the 4 cached model runs (compare.py) through the current monitor with the evaluation zone files plus
    `extra_ignore`: every event, all fields, must equal the stored one. Skipped (passes) when cache/ is absent."""
    from dataclasses import asdict
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for seq in ("MOT17-04", "MOT17-09"):
        zpath = os.path.join(here, "zones", seq + ".json")
        for model in ("plain", "s3x"):
            path = os.path.join(here, "cache", f"{seq}__{model}.json")
            if not os.path.exists(path):
                continue
            with open(path) as fh:
                run = json.load(fh)
            frames = sorted(int(f) for f in run["tracks"])
            monitor = za.ZoneMonitor(za.load_zones(zpath), 30.0, frame0=frames[0],
                                     ignore=za.load_ignore(zpath) + extra_ignore)
            events = []
            for f in frames:
                tracks = np.asarray(run["tracks"][str(f)], np.float32).reshape(-1, 6)
                events += [asdict(e) for e in monitor.update(f, {int(r[4]): za.foot_point(r) for r in tracks})]
            norm = [dict(e, foot=None if e["foot"] is None else list(e["foot"])) for e in events]
            assert norm == run["events"], f"{seq} {model}"


def test_cached_runs_unchanged_without_ignore():
    _replay_cached([])


def test_cached_runs_unchanged_with_a_mask_no_foot_reaches():
    # a non-empty ignore list exercises the masked code path on real tracks; a mask above the image touches no foot
    far = za.IgnoreRegion("off_image", np.array([[0, -60], [200, -60], [200, -10], [0, -10]], np.float32), note="test")
    _replay_cached([far])


# ------------------------------------------------------------------------------------------- APPROACH warnings
H = 100.0                                            # box height [px]: 10 px / frame at 10 fps = 1 body height / s
SQ_APPROACH = za.Zone("z", SQUARE.polygon, dwell_s=100.0, approach=za.ApproachRule())   # horizon 3 s, k 5
UP = za.DirectionRule((0.0, -1.0), 0.5)              # alarm when moving up the image (within 60 degrees)


def simulate_motion(feet: list, zones: list | None = None, lines: list | None = None, heights: list | None = None,
                    fps: float = 10.0) -> tuple[list, za.ZoneMonitor]:
    """One track (id 7) with the motion features -> ([(type, zone, frame, entered, eta_s), ...], monitor)."""
    monitor = za.ZoneMonitor([SQ_APPROACH] if zones is None else zones, fps, k=5, max_missing=15, frame0=0,
                             lines=lines or [])
    events = []
    for frame, xy in enumerate(feet):
        if xy is None:
            continue
        h = H if heights is None else heights[frame]
        events += [(e.type, e.zone, e.frame, e.entered, getattr(e, "eta_s", None))
                   for e in monitor.update(frame, {7: xy}, {7: h})]
    return events, monitor


def straight(x0: float, y0: float, dx: float, dy: float, n: int) -> list[tuple[float, float]]:
    """n foot points moving (dx, dy) px per frame from (x0, y0)."""
    return [(x0 + dx * f, y0 + dy * f) for f in range(n)]


def test_straight_walk_raises_approach_before_intrusion_with_the_expected_lead():
    # 1 body height / s straight up toward the square (bottom edge y = 100). The extrapolated path reaches the edge
    # within 3 s from frame 21 (y = 395, 2.95 s); k = 5 -> APPROACH at frame 25 with eta 2.55 s. The foot enters at
    # frame 51 -> INTRUSION at 55: the warning comes 30 frames = 3.0 s before the alarm, once (no repeat while
    # the person keeps approaching).
    events, _ = simulate_motion(straight(50, 605, 0, -10, 60))
    assert events == [("APPROACH", "z", 25, 21, 2.55), ("INTRUSION", "z", 55, 51, None)]


def test_parallel_away_and_standing_raise_no_approach():
    assert simulate_motion(straight(-300, 150, 10, 0, 60))[0] == []      # passes below the zone, parallel to its edge
    assert simulate_motion(straight(50, 150, 0, 10, 60))[0] == []        # walks away from it
    assert simulate_motion([(50.0, 150.0)] * 60)[0] == []                # stands 50 px (half a body) from it


def test_box_jitter_does_not_fake_or_hide_an_approach():
    rng = np.random.default_rng(0)
    n = 100
    jitter = rng.normal(0, 3, (n, 3))                  # 3 px on x, y and the box height (3 % of the height)
    standing = [(50 + jx, 115 + jy) for jx, jy, _ in jitter]          # 15 px below the zone edge
    heights = [H + jh for _, _, jh in jitter]
    assert simulate_motion(standing, heights=heights)[0] == []
    walking = [(x + jx, y + jy) for (x, y), (jx, jy, _) in zip(straight(50, 605, 0, -10, 60), jitter)]
    events, _ = simulate_motion(walking, heights=heights)
    assert [e[0] for e in events] == ["APPROACH", "INTRUSION"]
    assert 23 <= events[0][2] <= 27 and 2.7 <= (events[1][2] - events[0][2]) / 10 <= 3.3


def test_approach_rearms_after_turning_away():
    # approach (warning at 25), turn and walk away for 2 s (re-armed after 1 s without the condition), approach again
    feet = straight(50, 605, 0, -10, 30) + straight(50, 325, 0, 10, 20) + straight(50, 505, 0, -10, 30)
    events, _ = simulate_motion(feet)
    assert [e[0] for e in events] == ["APPROACH", "APPROACH"] and events[0][2] == 25 and events[1][2] > 55


def test_approach_rearms_after_exit():
    # walk up through the zone and out of its top, then come back down toward it
    feet = straight(50, 605, 0, -10, 81) + straight(50, -185, 0, 10, 30)
    events, _ = simulate_motion(feet)
    assert [e[0] for e in events][:4] == ["APPROACH", "INTRUSION", "EXIT", "APPROACH"]


def test_box_jump_toward_the_zone_raises_no_approach():
    # standing 60 px off the corner; the box jumps 50 px toward the zone for 3 frames (outside it) and back. The
    # straight line through the window then points into the zone; the fit residual (~0.2 body heights) blocks it.
    feet = [(160.0, 160.0)] * 20 + [(110.0, 110.0)] * 3 + [(160.0, 160.0)] * 20
    assert simulate_motion(feet)[0] == []
    unchecked = za.Zone("z", SQUARE.polygon, dwell_s=100.0, approach=za.ApproachRule(max_residual=None))
    assert [e[0] for e in simulate_motion(feet, zones=[unchecked])[0]] == ["APPROACH"]


def test_no_new_warning_while_milling_at_the_edge():
    # approach, then move up and down 40 px (0.4 body heights) in front of the zone for 8 s: one warning only
    feet = straight(50, 605, 0, -10, 40)
    for _ in range(4):
        feet += straight(50, 211, 0, -4, 10) + straight(50, 175, 0, 4, 10)
    assert [e[:3] for e in simulate_motion(feet)[0]] == [("APPROACH", "z", 25)]


def test_active_approach_is_visible_to_the_drawing():
    _, monitor = simulate_motion(straight(50, 605, 0, -10, 30))
    zone, eta, start, target = monitor.approach_view(7)
    assert zone == "z" and abs(eta - 2.15) < 1e-6 and abs(target[1] - 100) < 1e-6


def test_drawn_warning_counts_down_and_disappears_after_the_hold():
    # walk up, then stop: the condition last holds at frame 31; the drawn eta counts down for HOLD_S (3 frames at
    # 10 fps), then the warning is no longer drawn (it stays in the event log)
    monitor = za.ZoneMonitor([SQ_APPROACH], 10.0, k=5, max_missing=15, frame0=0)
    views = []
    for frame, xy in enumerate(straight(50, 605, 0, -10, 30) + [(50.0, 315.0)] * 10):
        monitor.update(frame, {7: xy}, {7: H})
        views.append(monitor.approach_view(7))
    seen = monitor.approach_state[(7, "z")].seen
    eta = monitor.approach_state[(7, "z")].view[0]
    assert all(abs(views[seen + d][1] - (eta - d / 10)) < 1e-9 for d in range(4))
    assert views[seen + 4] is None and views[-1] is None


def test_missing_heights_are_refused_when_motion_needs_them():
    monitor = za.ZoneMonitor([SQ_APPROACH], 10.0)
    try:
        monitor.update(0, {7: OUTSIDE})
    except ValueError:
        return
    raise AssertionError("update() without heights silently switched APPROACH off")


def test_banner_shows_an_alarm_over_a_newer_warning():
    intrusion = za.Event("INTRUSION", "z", 1, 10, 1.0, 6, (0.0, 0.0))
    approach = za.MotionEvent("APPROACH", "z", 2, 12, 1.2, 8, (0.0, 0.0), velocity=(0.0, -0.5), eta_s=2.0)
    red, amber = (0, 0, 190), (0, 120, 200)
    assert tuple(za.status_bar(400, [intrusion, approach], [])[0, 0]) == red
    assert tuple(za.status_bar(400, [approach], [])[0, 0]) == amber


# ------------------------------------------------------------------------------------------- direction filter
def directed(direction: za.DirectionRule, approach: bool = False) -> za.Zone:
    return za.Zone("z", SQUARE.polygon, dwell_s=3.0, direction=direction,
                   approach=za.ApproachRule() if approach else None)


def test_direction_filter_allows_the_alarm_direction():
    events, monitor = simulate_motion(straight(50, 200, 0, -10, 20), zones=[directed(UP)])
    assert [e[0] for e in events] == ["INTRUSION"] and monitor.alarm_zones(7) == ["z"]


def test_direction_filter_blocks_other_directions_but_logs_entry_and_loitering():
    events, monitor = simulate_motion(straight(50, -100, 0, 10, 15) + [(50.0, 50.0)] * 40, zones=[directed(UP)])
    assert [e[0] for e in events] == ["ENTRY", "LOITERING"]                  # moving down: not the alarm direction
    assert monitor.inside_zones(7) == ["z"] and monitor.alarm_zones(7) == []
    sideways, _ = simulate_motion(straight(-100, 50, 10, 0, 20), zones=[directed(UP)])
    assert [e[0] for e in sideways] == ["ENTRY"]                              # cos = 0 < 0.5
    diagonal, _ = simulate_motion(straight(-50, 150, 7.07, -7.07, 20), zones=[directed(UP)])
    assert [e[0] for e in diagonal] == ["INTRUSION"]                          # 45 degrees: cos 0.71 >= 0.5


def test_unknown_direction_fails_safe_to_intrusion():
    events, _ = simulate_motion([(50.0, 50.0)] * 10, zones=[directed(UP)])   # appears inside and stands still
    assert [e[0] for e in events] == ["INTRUSION"]


def test_approach_only_in_the_alarm_direction_when_filtered():
    up, _ = simulate_motion(straight(50, 605, 0, -10, 40), zones=[directed(UP, approach=True)])
    down, _ = simulate_motion(straight(50, -505, 0, 10, 40), zones=[directed(UP, approach=True)])
    assert [e[0] for e in up] == ["APPROACH"] and down == []


def _write_cfg(cfg: dict, fn):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "z.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh)
        return fn(path)


def test_alarm_edge_is_the_inward_normal_of_that_edge():
    square = [[0, 0], [100, 0], [100, 100], [0, 100]]
    cfg = {"zones": [{"name": "a", "polygon": square, "dwell_s": 1, "alarm_edge": 2},            # bottom edge
                     {"name": "b", "polygon": square[::-1], "dwell_s": 1, "alarm_edge": 0}]}     # same edge, reversed
    zones = _write_cfg(cfg, za.load_zones)
    for zone in zones:
        assert np.allclose(zone.direction.vector, (0, -1)) and zone.direction.min_cos == 0.5


# ------------------------------------------------------------------------------------------- tripwires
def tripwire(direction: list) -> za.Tripwire:
    cfg = {"zones": [], "lines": [{"name": "w", "points": [[0, 200], [100, 200]], "alarm_direction": direction,
                                   "note": "test"}]}
    return _write_cfg(cfg, za.load_lines)[0]


def test_tripwire_alarm_direction_only():
    up = tripwire([0, -1])
    events, _ = simulate_motion(straight(50, 300, 0, -10, 25), zones=[], lines=[up])
    assert events == [("LINE_CROSS", "w", 15, 11, None)]      # first frame above the line: 11; K = 5 -> 15
    events, _ = simulate_motion(straight(50, 100, 0, 10, 25), zones=[], lines=[up])
    assert events == []                                        # crossing the other way raises nothing
    events, _ = simulate_motion(straight(50, 100, 0, 10, 25), zones=[], lines=[tripwire([0, 1])])
    assert [e[0] for e in events] == ["LINE_CROSS"]            # ... unless that is the alarm direction


def test_tripwire_ignores_jitter_and_paths_beyond_its_ends():
    up = tripwire([0, -1])
    jitter = ([(50.0, 203.0)] * 3 + [(50.0, 197.0)] * 3) * 10        # 3 frames per side < K
    assert simulate_motion(jitter, zones=[], lines=[up])[0] == []
    assert simulate_motion(straight(300, 300, 0, -10, 25), zones=[], lines=[up])[0] == []


def test_tripwire_starting_side_needs_k_frames():
    # a track first seen 1 frame below the line, then above: no crossing (the old rule took the first frame's side)
    up = tripwire([0, -1])
    assert simulate_motion([(50.0, 203.0)] + [(50.0, 197.0)] * 10, zones=[], lines=[up])[0] == []
    assert [e[0] for e in simulate_motion([(50.0, 203.0)] * 5 + [(50.0, 197.0)] * 10, zones=[],
                                          lines=[up])[0]] == ["LINE_CROSS"]


def test_tripwire_crossing_starts_at_the_last_run_on_the_new_side():
    # below, 2 frames above (jitter), back below, then 6 frames above: 'entered' is the first of those 6 (frame 14)
    up = tripwire([0, -1])
    feet = [(50.0, 210.0)] * 10 + [(50.0, 190.0)] * 2 + [(50.0, 210.0)] * 2 + [(50.0, 190.0)] * 6
    assert simulate_motion(feet, zones=[], lines=[up])[0] == [("LINE_CROSS", "w", 18, 14, None)]


def test_tripwire_walk_around_an_end_is_no_crossing():
    # below the line, out past its right end (x > 100) for >= K frames, up, and back left above the line
    up = tripwire([0, -1])
    feet = straight(50, 300, 10, 0, 10) + straight(150, 300, 0, -20, 10) + straight(150, 100, -10, 0, 12)
    assert simulate_motion(feet, zones=[], lines=[up])[0] == []


def test_line_parallel_to_its_alarm_direction_is_refused():
    try:
        tripwire([1, 0])
    except ValueError:
        return
    raise AssertionError("a tripwire with an alarm direction along the line was accepted")


# ------------------------------------------------------------------------------------------- old files unchanged
def test_zone_files_without_motion_keys_switch_motion_off():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for seq in ("MOT17-04", "MOT17-09"):
        path = os.path.join(here, "zones", seq + ".json")
        zones = za.load_zones(path)
        assert all(z.approach is None and z.direction is None for z in zones) and za.load_lines(path) == []
        assert not za.ZoneMonitor(zones, 30.0).motion_on


def test_v2_zone_files_only_add_events_or_relabel_entries():
    """Replay the 4 cached runs with zones/<seq>_v2.json (approach + direction + tripwire on): dropping APPROACH /
    LINE_CROSS, renaming ENTRY to INTRUSION and keeping the base fields gives exactly the stored events, so the motion
    features never move a zone entry, exit or loitering event."""
    from dataclasses import asdict
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = ("type", "zone", "track_id", "frame", "t", "entered", "foot")
    kinds = set()
    for seq in ("MOT17-04", "MOT17-09"):
        zpath = os.path.join(here, "zones", seq + "_v2.json")
        for model in ("plain", "s3x"):
            path = os.path.join(here, "cache", f"{seq}__{model}.json")
            if not os.path.exists(path):
                continue
            with open(path) as fh:
                run = json.load(fh)
            frames = sorted(int(f) for f in run["tracks"])
            monitor = za.ZoneMonitor(za.load_zones(zpath), 30.0, frame0=frames[0], lines=za.load_lines(zpath))
            assert monitor.motion_on
            events = []
            for f in frames:
                tracks = np.asarray(run["tracks"][str(f)], np.float32).reshape(-1, 6)
                for e in monitor.update(f, *za.observations(tracks)):
                    kinds.add(e.type)
                    if e.type in ("APPROACH", "LINE_CROSS"):
                        continue
                    d = {k: v for k, v in asdict(e).items() if k in base}
                    d["type"] = "INTRUSION" if d["type"] == "ENTRY" else d["type"]
                    events.append(dict(d, foot=None if d["foot"] is None else list(d["foot"])))
            assert events == run["events"], f"{seq} {model}"
    if kinds:                                          # the new code paths really ran on the real tracks
        assert {"APPROACH", "ENTRY", "LINE_CROSS"} <= kinds, kinds


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS", name)
        except AssertionError as err:
            failed += 1
            print("FAIL", name, err)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
