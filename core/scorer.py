"""Per-seat sun exposure scoring across route segments."""
import math
from typing import Literal

from core.solar import get_sun_position

# Angle each window faces, relative to vehicle forward (clockwise from nose).
#   0°  = windshield  (front)
#   90° = right door windows
#  180° = rear window (back)
#  270° = left door windows
_SEAT_FACING: dict[str, float] = {
    "FL": 270.0,  # front-left  → left window
    "FR": 90.0,   # front-right → right window
    "RL": 270.0,  # rear-left   → left window
    "RR": 90.0,   # rear-right  → right window
}


def _angular_diff(a: float, b: float) -> float:
    """Shortest arc (0–180°) between two bearings."""
    return abs((a - b + 180) % 360 - 180)


def score_seats(
    segments: list[dict],
    drive_side: Literal["LHD", "RHD"] = "LHD",
) -> dict:
    """
    Accumulate sun-exposure scores for all four seats over a list of segments.

    Each segment dict must contain:
        lat             – float, decimal degrees
        lng             – float, decimal degrees
        timestamp_utc   – float, Unix timestamp (UTC)
        heading_degrees – float, vehicle heading (0 = north, clockwise)
        uv_index        – float, UV index (0–11+)
        cloud_cover_pct – float, 0–100

    Args:
        segments:   Ordered list of route segment dicts (see above).
        drive_side: "LHD" (driver front-left, default) or "RHD" (driver front-right).

    Returns:
        {
            "FL": float,        # accumulated exposure score
            "FR": float,
            "RL": float,
            "RR": float,
            "best_seat":  str,  # seat with lowest total exposure
            "worst_seat": str,  # seat with highest total exposure
            "driver_seat": str, # FL for LHD, FR for RHD
        }
    """
    scores: dict[str, float] = {seat: 0.0 for seat in _SEAT_FACING}

    for seg in segments:
        solar = get_sun_position(seg["lat"], seg["lng"], seg["timestamp_utc"])

        # Skip segments where the sun is below the horizon.
        if solar["elevation"] <= 0:
            continue

        relative_angle = (solar["azimuth"] - seg["heading_degrees"]) % 360

        # Effective weight: UV intensity reduced by cloud opacity.
        weight = seg["uv_index"] * (1.0 - seg["cloud_cover_pct"] / 100.0)
        if weight <= 0:
            continue

        for seat, facing in _SEAT_FACING.items():
            diff = _angular_diff(relative_angle, facing)
            # cos-based factor: 1.0 when sun is square-on to the window, 0 at 90°.
            exposure = max(0.0, math.cos(math.radians(diff)))
            scores[seat] += exposure * weight

    driver_seat = "FL" if drive_side == "LHD" else "FR"
    best_seat = min(scores, key=scores.get)
    worst_seat = max(scores, key=scores.get)

    return {
        "FL": round(scores["FL"], 4),
        "FR": round(scores["FR"], 4),
        "RL": round(scores["RL"], 4),
        "RR": round(scores["RR"], 4),
        "best_seat": best_seat,
        "worst_seat": worst_seat,
        "driver_seat": driver_seat,
    }
