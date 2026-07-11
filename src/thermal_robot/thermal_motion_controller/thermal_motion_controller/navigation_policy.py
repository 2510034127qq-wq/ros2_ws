"""Pure policy for rate-limited Nav2 goal lifecycle decisions."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple


GOAL_KEEP = "keep"
GOAL_WAIT = "wait"
GOAL_SEND = "send"
GOAL_REPLACE = "replace"

DIRECT_STOP = "stop"
DIRECT_KNOWN_FREE = "known_free"
DIRECT_SCAN_GUARDED = "scan_guarded"


def forward_clearance(
    ranges: Sequence[float],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    half_angle_rad: float,
) -> Optional[float]:
    """Return the closest valid return in a forward-facing scan cone.

    Positive infinity is the normal LaserScan representation for no return, so
    it contributes ``range_max``.  Invalid samples and rays outside the cone do
    not contribute evidence that direct motion is clear.
    """
    best: Optional[float] = None
    cone = max(0.0, float(half_angle_rad))
    lower = max(0.0, float(range_min))
    upper = float(range_max)
    if not math.isfinite(upper) or upper <= lower:
        return None
    for index, raw_range in enumerate(ranges):
        angle = float(angle_min) + index * float(angle_increment)
        wrapped = math.atan2(math.sin(angle), math.cos(angle))
        if abs(wrapped) > cone:
            continue
        value = float(raw_range)
        if math.isinf(value) and value > 0.0:
            value = upper
        if not math.isfinite(value) or value < lower or value > upper:
            continue
        best = value if best is None else min(best, value)
    return best


def decide_direct_motion(
    path_known_free: bool,
    scan_clearance_m: Optional[float],
    scan_age_s: float,
    stop_distance_m: float,
    scan_stale_s: float,
) -> str:
    """Classify a direct command using global-path and local-scan evidence.

    Every direct command requires a fresh, locally clear scan.  A fully
    known-free occupancy segment is distinguished from incremental motion into
    unknown space; the latter is allowed only one control tick at a time under
    the same live scan guard.
    """
    if scan_clearance_m is None:
        return DIRECT_STOP
    if not math.isfinite(float(scan_age_s)) or float(scan_age_s) < 0.0:
        return DIRECT_STOP
    if float(scan_age_s) > max(0.0, float(scan_stale_s)):
        return DIRECT_STOP
    if float(scan_clearance_m) <= max(0.0, float(stop_distance_m)):
        return DIRECT_STOP
    return DIRECT_KNOWN_FREE if path_known_free else DIRECT_SCAN_GUARDED


def decide_goal_action(
    state: str,
    current_goal: Optional[Tuple[float, float]],
    requested_goal: Tuple[float, float],
    now: float,
    last_send_t: float,
    min_interval_s: float,
    same_goal_tolerance_m: float = 0.5,
) -> str:
    """Choose whether to keep, wait, send, or replace a Nav2 goal."""
    state = str(state).lower()
    if state == "active" and current_goal is not None:
        if math.hypot(
                requested_goal[0] - current_goal[0],
                requested_goal[1] - current_goal[1]) < same_goal_tolerance_m:
            return GOAL_KEEP
    if state == "sending":
        return GOAL_WAIT
    if float(now) - float(last_send_t) < max(0.0, float(min_interval_s)):
        return GOAL_WAIT
    if state == "active":
        return GOAL_REPLACE
    if state in ("idle", "done"):
        return GOAL_SEND
    return GOAL_WAIT
