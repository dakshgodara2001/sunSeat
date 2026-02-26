"""Unit tests for the weather-enrichment fallback in api.routes."""
from datetime import datetime, timezone

import pytest

from api.routes import _estimate_uv_from_time


def _ts(hour: int) -> float:
    """Unix timestamp for 2024-06-21 at the given UTC hour."""
    return datetime(2024, 6, 21, hour, 0, 0, tzinfo=timezone.utc).timestamp()


def test_noon_returns_peak_uv():
    """Solar noon (12 UTC) should give the highest UV value."""
    uv = _estimate_uv_from_time(_ts(12))
    assert uv == pytest.approx(6.0, abs=0.01)


def test_night_returns_zero():
    """Hours outside 6–18 should give UV 0."""
    for hour in (0, 3, 5, 19, 23):
        assert _estimate_uv_from_time(_ts(hour)) == 0.0


def test_morning_and_evening_symmetric():
    """UV at 9 AM and 3 PM (symmetric around noon) should be equal."""
    uv_9 = _estimate_uv_from_time(_ts(9))
    uv_15 = _estimate_uv_from_time(_ts(15))
    assert uv_9 == pytest.approx(uv_15, abs=0.01)
    assert uv_9 > 0


def test_returns_non_negative():
    """UV should never be negative for any hour."""
    for hour in range(24):
        assert _estimate_uv_from_time(_ts(hour)) >= 0.0
