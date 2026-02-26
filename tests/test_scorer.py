"""Unit tests for core.scorer.score_seats.

All tests mock get_sun_position so they exercise only the scoring logic,
not the pvlib integration (that is covered by test_solar.py).
"""
import math
from unittest.mock import patch

import pytest

from core.scorer import score_seats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg(
    heading: float = 0.0,
    uv_index: float = 5.0,
    cloud_cover_pct: float = 0.0,
) -> dict:
    """Return a minimal segment dict; lat/lng/timestamp are arbitrary (mocked)."""
    return {
        "lat": 0.0,
        "lng": 0.0,
        "timestamp_utc": 1_700_000_000.0,
        "heading_degrees": heading,
        "uv_index": uv_index,
        "cloud_cover_pct": cloud_cover_pct,
    }


def _mock_sun(azimuth: float, elevation: float = 45.0):
    """Return a get_sun_position patch that always yields a fixed solar position."""
    return lambda lat, lng, ts: {"azimuth": azimuth, "elevation": elevation}


# ---------------------------------------------------------------------------
# Geometry tests
# ---------------------------------------------------------------------------

def test_sun_directly_to_right_scores_right_seats():
    """Heading north, sun due east (90°) → right windows fully exposed."""
    with patch("core.scorer.get_sun_position", side_effect=_mock_sun(azimuth=90)):
        result = score_seats([_seg(heading=0)])

    assert result["FR"] == pytest.approx(result["RR"])  # symmetric
    assert result["FL"] == pytest.approx(0.0, abs=1e-9)
    assert result["RL"] == pytest.approx(0.0, abs=1e-9)
    assert result["FR"] > 0
    assert result["worst_seat"] in ("FR", "RR")
    assert result["best_seat"] in ("FL", "RL")


def test_sun_directly_to_left_scores_left_seats():
    """Heading north, sun due west (270°) → left windows fully exposed."""
    with patch("core.scorer.get_sun_position", side_effect=_mock_sun(azimuth=270)):
        result = score_seats([_seg(heading=0)])

    assert result["FL"] == pytest.approx(result["RL"])
    assert result["FR"] == pytest.approx(0.0, abs=1e-9)
    assert result["RR"] == pytest.approx(0.0, abs=1e-9)
    assert result["FL"] > 0
    assert result["worst_seat"] in ("FL", "RL")
    assert result["best_seat"] in ("FR", "RR")


def test_sun_directly_ahead_gives_zero_side_exposure():
    """Sun straight ahead (relative 0°) is perpendicular to all side windows → 0."""
    with patch("core.scorer.get_sun_position", side_effect=_mock_sun(azimuth=0)):
        result = score_seats([_seg(heading=0)])

    for seat in ("FL", "FR", "RL", "RR"):
        assert result[seat] == pytest.approx(0.0, abs=1e-9)


def test_heading_rotates_relative_angle():
    """
    Sun is north (0°), vehicle heading east (90°).
    Relative angle = (0 - 90) % 360 = 270° → sun on LEFT side.
    """
    with patch("core.scorer.get_sun_position", side_effect=_mock_sun(azimuth=0)):
        result = score_seats([_seg(heading=90)])

    assert result["FL"] > 0
    assert result["RL"] > 0
    assert result["FR"] == pytest.approx(0.0, abs=1e-9)
    assert result["RR"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Weighting tests
# ---------------------------------------------------------------------------

def test_full_cloud_cover_zeroes_all_scores():
    """100 % cloud cover → effective weight 0 → all seats score 0."""
    with patch("core.scorer.get_sun_position", side_effect=_mock_sun(azimuth=90)):
        result = score_seats([_seg(heading=0, cloud_cover_pct=100)])

    for seat in ("FL", "FR", "RL", "RR"):
        assert result[seat] == pytest.approx(0.0, abs=1e-9)


def test_uv_index_scales_scores():
    """Doubling the UV index should double every seat's score."""
    with patch("core.scorer.get_sun_position", side_effect=_mock_sun(azimuth=90)):
        low = score_seats([_seg(heading=0, uv_index=3)])
    with patch("core.scorer.get_sun_position", side_effect=_mock_sun(azimuth=90)):
        high = score_seats([_seg(heading=0, uv_index=6)])

    for seat in ("FL", "FR", "RL", "RR"):
        assert high[seat] == pytest.approx(2 * low[seat], rel=1e-6)


def test_cloud_cover_reduces_scores_proportionally():
    """50 % cloud cover halves the score compared to clear sky."""
    with patch("core.scorer.get_sun_position", side_effect=_mock_sun(azimuth=90)):
        clear = score_seats([_seg(heading=0, cloud_cover_pct=0)])
    with patch("core.scorer.get_sun_position", side_effect=_mock_sun(azimuth=90)):
        cloudy = score_seats([_seg(heading=0, cloud_cover_pct=50)])

    for seat in ("FL", "FR", "RL", "RR"):
        assert cloudy[seat] == pytest.approx(0.5 * clear[seat], rel=1e-6)


# ---------------------------------------------------------------------------
# Elevation / horizon tests
# ---------------------------------------------------------------------------

def test_sun_below_horizon_skipped():
    """Sun with elevation ≤ 0 (night / below horizon) contributes nothing."""
    with patch("core.scorer.get_sun_position", side_effect=_mock_sun(azimuth=90, elevation=0)):
        result = score_seats([_seg(heading=0)])

    for seat in ("FL", "FR", "RL", "RR"):
        assert result[seat] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Accumulation tests
# ---------------------------------------------------------------------------

def test_scores_accumulate_across_segments():
    """N identical segments should produce N× the single-segment score."""
    seg = _seg(heading=0)
    with patch("core.scorer.get_sun_position", side_effect=_mock_sun(azimuth=90)):
        single = score_seats([seg])
    with patch("core.scorer.get_sun_position", side_effect=_mock_sun(azimuth=90)):
        triple = score_seats([seg, seg, seg])

    for seat in ("FL", "FR", "RL", "RR"):
        assert triple[seat] == pytest.approx(3 * single[seat], rel=1e-6)


def test_mixed_segments_accumulate_correctly():
    """Sun from right for 1 seg, then from left for 1 seg → symmetric total scores."""
    seg_right = _seg(heading=0)  # sun azimuth=90 → right exposed
    seg_left = _seg(heading=0)   # sun azimuth=270 → left exposed

    def _alternating(lat, lng, ts):
        # Called twice; return right then left
        if not hasattr(_alternating, "call_count"):
            _alternating.call_count = 0
        _alternating.call_count += 1
        if _alternating.call_count % 2 == 1:
            return {"azimuth": 90.0, "elevation": 45.0}
        return {"azimuth": 270.0, "elevation": 45.0}

    with patch("core.scorer.get_sun_position", side_effect=_alternating):
        result = score_seats([seg_right, seg_left])

    assert result["FR"] == pytest.approx(result["FL"], rel=1e-6)
    assert result["RR"] == pytest.approx(result["RL"], rel=1e-6)


# ---------------------------------------------------------------------------
# Drive-side tests
# ---------------------------------------------------------------------------

def test_lhd_driver_seat_is_fl():
    with patch("core.scorer.get_sun_position", side_effect=_mock_sun(azimuth=90)):
        result = score_seats([_seg()], drive_side="LHD")
    assert result["driver_seat"] == "FL"


def test_rhd_driver_seat_is_fr():
    with patch("core.scorer.get_sun_position", side_effect=_mock_sun(azimuth=90)):
        result = score_seats([_seg()], drive_side="RHD")
    assert result["driver_seat"] == "FR"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_segments_returns_zero_scores():
    result = score_seats([])
    assert result["FL"] == 0.0
    assert result["FR"] == 0.0
    assert result["RL"] == 0.0
    assert result["RR"] == 0.0


def test_return_keys_always_present():
    with patch("core.scorer.get_sun_position", side_effect=_mock_sun(azimuth=90)):
        result = score_seats([_seg()])
    for key in ("FL", "FR", "RL", "RR", "best_seat", "worst_seat", "driver_seat"):
        assert key in result


# ---------------------------------------------------------------------------
# Short trip / single-segment tests
# ---------------------------------------------------------------------------

def test_single_segment_short_trip_returns_valid_result():
    """A single segment (very short trip) should still produce a full result."""
    with patch("core.scorer.get_sun_position", side_effect=_mock_sun(azimuth=90)):
        result = score_seats([_seg(heading=0, uv_index=3.0)])

    assert result["FR"] > 0
    assert result["best_seat"] in ("FL", "RL")
    assert result["worst_seat"] in ("FR", "RR")
    for key in ("FL", "FR", "RL", "RR", "best_seat", "worst_seat", "driver_seat"):
        assert key in result


# ---------------------------------------------------------------------------
# Sun overhead / no significant difference
# ---------------------------------------------------------------------------

def test_sun_directly_overhead_all_seats_near_zero():
    """Sun at azimuth 180° heading north → relative 180° → perpendicular to
    all side windows → all scores ≈ 0."""
    with patch("core.scorer.get_sun_position", side_effect=_mock_sun(azimuth=180, elevation=85)):
        result = score_seats([_seg(heading=0)])

    for seat in ("FL", "FR", "RL", "RR"):
        assert result[seat] == pytest.approx(0.0, abs=1e-9)


def test_sun_overhead_north_south_route_symmetric():
    """Heading north then south with sun at 180° → all seats get equal (zero) exposure."""
    seg_north = _seg(heading=0)
    seg_south = _seg(heading=180)

    def _alternating(lat, lng, ts):
        # Sun stays due south at high elevation throughout
        return {"azimuth": 180.0, "elevation": 80.0}

    with patch("core.scorer.get_sun_position", side_effect=_alternating):
        result = score_seats([seg_north, seg_south])

    # All seats should be equal (or both zero)
    assert result["FL"] == pytest.approx(result["FR"], abs=1e-6)
    assert result["RL"] == pytest.approx(result["RR"], abs=1e-6)
