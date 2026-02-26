"""Tests for solar calculation logic."""
from datetime import datetime, timezone
from core.solar import get_solar_position, get_sun_position


def test_solar_position_returns_expected_keys():
    dt = datetime(2024, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
    result = get_solar_position(37.7749, -122.4194, dt)
    assert "azimuth" in result
    assert "elevation" in result


def test_solar_elevation_midday_summer():
    # Midday in summer at equator should have high elevation
    dt = datetime(2024, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
    result = get_solar_position(0.0, 0.0, dt)
    assert result["elevation"] > 60


def test_get_sun_position_returns_expected_keys():
    ts = datetime(2024, 3, 21, 6, 51, 0, tzinfo=timezone.utc).timestamp()
    result = get_sun_position(28.6, 77.2, ts)
    assert "azimuth" in result
    assert "elevation" in result


def test_get_sun_position_delhi_equinox_solar_noon():
    # Delhi is at 77.2°E → solar noon in UTC ≈ 12:00 - 77.2/15h ≈ 06:51 UTC.
    # On the vernal equinox the sun transits due south, so azimuth ≈ 180°.
    # (12:00 UTC is ~5.5 h past solar noon for Delhi; azimuth there is ~263°.)
    ts = datetime(2024, 3, 21, 6, 51, 0, tzinfo=timezone.utc).timestamp()
    result = get_sun_position(28.6, 77.2, ts)
    assert abs(result["azimuth"] - 180) < 10, (
        f"Expected azimuth ~180°, got {result['azimuth']}°"
    )
    assert result["elevation"] > 50  # sun is high at solar noon on equinox
