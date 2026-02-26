"""Seat scoring: determine which side of the vehicle gets more sun."""
from typing import Literal


def score_seat(
    solar_azimuth: float,
    vehicle_heading: float,
) -> dict:
    """
    Given solar azimuth and vehicle heading (degrees, 0=North, clockwise),
    determine which side of the vehicle is sunnier.

    Returns a score dict with the sunnier side and an intensity value (0-1).
    """
    # Relative angle of sun from vehicle's perspective
    relative_angle = (solar_azimuth - vehicle_heading) % 360

    # 0-180 means sun is on the right side, 180-360 on the left
    if 0 <= relative_angle < 180:
        sunny_side: Literal["right", "left"] = "right"
        intensity = 1.0 - abs(relative_angle - 90) / 90  # peaks at 90 deg
    else:
        sunny_side = "left"
        intensity = 1.0 - abs(relative_angle - 270) / 90
        intensity = max(0.0, min(intensity, 1.0))

    return {
        "sunny_side": sunny_side,
        "shaded_side": "left" if sunny_side == "right" else "right",
        "intensity": round(intensity, 3),
    }
