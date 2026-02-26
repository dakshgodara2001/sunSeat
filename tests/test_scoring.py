"""Tests for seat scoring logic."""
from core.scoring import score_seat


def test_sun_directly_to_right():
    # Vehicle heading north (0), sun at east (90) → sun on right
    result = score_seat(solar_azimuth=90, vehicle_heading=0)
    assert result["sunny_side"] == "right"
    assert result["shaded_side"] == "left"
    assert result["intensity"] == 1.0


def test_sun_directly_to_left():
    # Vehicle heading north (0), sun at west (270) → sun on left
    result = score_seat(solar_azimuth=270, vehicle_heading=0)
    assert result["sunny_side"] == "left"
    assert result["shaded_side"] == "right"
    assert result["intensity"] == 1.0


def test_sun_ahead_low_intensity():
    # Vehicle heading north (0), sun directly ahead (0/360) → low intensity
    result = score_seat(solar_azimuth=0, vehicle_heading=0)
    assert result["intensity"] == 0.0
