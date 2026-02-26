"""Integration tests for the full recommend pipeline.

Each test mocks the Google Maps Directions API (httpx) and the solar position
API (get_sun_position), then drives the POST /recommend endpoint end-to-end
and asserts best_seat and confidence.

Includes edge-case scenarios: short trips, sun overhead, missing weather data,
and invalid origin/destination.
"""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Simple 3-point eastbound polyline: (40.0, -74.0) → (40.0, -73.99) → (40.0, -73.98)
# Heading ≈ 90° (due east) for both sub-segments.
_EASTBOUND_POLYLINE = "_oydFvkrbM?oR?oR"

# Simple 3-point westbound polyline: reverse of the above
_WESTBOUND_POLYLINE = "_oydFbiobM?nR?nR"


def _make_directions_response(polyline: str, step_duration: int = 1800) -> dict:
    """Fabricate a minimal Google Directions API JSON response."""
    return {
        "status": "OK",
        "routes": [
            {
                "legs": [
                    {
                        "steps": [
                            {
                                "polyline": {"points": polyline},
                                "duration": {"value": step_duration},
                                "start_location": {"lat": 40.0, "lng": -74.0},
                                "end_location": {"lat": 40.0, "lng": -73.98},
                            }
                        ]
                    }
                ]
            }
        ],
    }


def _patch_httpx(directions_json: dict):
    """Context manager that makes httpx.AsyncClient.get return a mock response."""
    mock_response = MagicMock()
    mock_response.json.return_value = directions_json
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    return patch("core.routing.httpx.AsyncClient", return_value=mock_client)


def _fixed_sun(azimuth: float, elevation: float):
    """Return a callable that always yields the same solar position."""
    return lambda lat, lng, ts: {"azimuth": azimuth, "elevation": elevation}


# ---------------------------------------------------------------------------
# Scenario 1 – Morning eastbound trip (sun ahead / to the right)
#
# Vehicle heading ≈ 90° (east).  Morning sun is in the east, azimuth ≈ 100°.
# Relative angle = (100 - 90) % 360 = 10°  → sun nearly ahead.
# At 10° relative the right-window facing (90°) gets:
#   angular_diff(10, 90) = 80° → cos(80°) ≈ 0.17
# Left-window facing (270°):
#   angular_diff(10, 270) = 100° → cos(100°) < 0 → clamped to 0.
# So right seats (FR, RR) accumulate exposure, left seats stay near zero.
# best_seat should be FL or RL (left side, shaded).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_morning_eastbound_sun_ahead(monkeypatch):
    """Morning eastbound: sun roughly ahead-right → left seats are shadiest."""
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "fake-key")

    directions = _make_directions_response(_EASTBOUND_POLYLINE)
    sun_mock = _fixed_sun(azimuth=100.0, elevation=30.0)

    with (
        _patch_httpx(directions),
        patch("core.scorer.get_sun_position", side_effect=sun_mock),
        patch("api.routes.get_sun_position", side_effect=sun_mock),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/recommend",
                json={
                    "origin": "New York, NY",
                    "destination": "Newark, NJ",
                    "departure_time": "2024-06-21T08:00:00Z",
                    "drive_side": "LHD",
                },
            )

    assert resp.status_code == 200
    data = resp.json()

    # Left seats should be best (lowest exposure)
    assert data["best_seat"] in ("FL", "RL")
    # Right seats should be worst (highest exposure)
    assert data["worst_seat"] in ("FR", "RR")


# ---------------------------------------------------------------------------
# Scenario 2 – Evening westbound trip (sun directly hitting driver side)
#
# Vehicle heading ≈ 270° (west).  Evening sun is in the west, azimuth ≈ 260°.
# Relative angle = (260 - 270) % 360 = 350°  → sun nearly ahead.
# angular_diff(350, 270) = 80° → cos(80°) ≈ 0.17  (left windows get some)
# angular_diff(350, 90)  = 100° → cos(100°) < 0 → clamped to 0.
#
# With a more side-hitting azimuth of 220°:
# Relative angle = (220 - 270) % 360 = 310°
# angular_diff(310, 270) = 40° → cos(40°) ≈ 0.77  (left windows strong)
# angular_diff(310, 90)  = 140° → cos(140°) < 0 → 0  (right windows zero)
# Left seats (FL, RL) get hammered.  best_seat = FR or RR.
# For LHD, driver is FL = sun-blasted side.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evening_westbound_sun_hits_driver_side(monkeypatch):
    """Evening westbound: sun from the south-west hits left (driver) side."""
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "fake-key")

    directions = _make_directions_response(_WESTBOUND_POLYLINE)
    # Sun from ~220° azimuth with moderate elevation
    sun_mock = _fixed_sun(azimuth=220.0, elevation=25.0)

    with (
        _patch_httpx(directions),
        patch("core.scorer.get_sun_position", side_effect=sun_mock),
        patch("api.routes.get_sun_position", side_effect=sun_mock),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/recommend",
                json={
                    "origin": "Newark, NJ",
                    "destination": "New York, NY",
                    "departure_time": "2024-06-21T18:00:00Z",
                    "drive_side": "LHD",
                },
            )

    assert resp.status_code == 200
    data = resp.json()

    # Left side gets the sun → right seats are shadiest
    assert data["best_seat"] in ("FR", "RR")
    # Left side is worst
    assert data["worst_seat"] in ("FL", "RL")


# ---------------------------------------------------------------------------
# Scenario 3 – Overcast conditions (confidence should be "low")
#
# Sun below the horizon (elevation ≤ 0) simulates heavy overcast / night.
# All UV values will be 0, all scores will be 0. Confidence → "low".
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_overcast_conditions_low_confidence(monkeypatch):
    """Overcast / no sun: all scores zero, confidence must be 'low'."""
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "fake-key")

    directions = _make_directions_response(_EASTBOUND_POLYLINE)
    # Elevation ≤ 0 means sun is below horizon → UV = 0 → no exposure
    sun_mock = _fixed_sun(azimuth=100.0, elevation=-5.0)

    with (
        _patch_httpx(directions),
        patch("core.scorer.get_sun_position", side_effect=sun_mock),
        patch("api.routes.get_sun_position", side_effect=sun_mock),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/recommend",
                json={
                    "origin": "New York, NY",
                    "destination": "Newark, NJ",
                    "departure_time": "2024-06-21T08:00:00Z",
                    "drive_side": "LHD",
                },
            )

    assert resp.status_code == 200
    data = resp.json()

    assert data["confidence"] == "low"
    # All scores should be zero (or near-zero) under overcast
    for seat in ("FL", "FR", "RL", "RR"):
        assert data["scores"][seat] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Edge case 1 – Very short trip (<5 min)
#
# A single-step route with a 2-minute duration should still return a valid
# 200 response with scores and a best_seat.
# ---------------------------------------------------------------------------

# Minimal 2-point polyline: (40.0, -74.0) → (40.0, -73.999)
# Heading ≈ 90° (east), very short distance.
_SHORT_POLYLINE = "_oydFvkrbM?E"


@pytest.mark.asyncio
async def test_short_trip_still_returns_result(monkeypatch):
    """A trip under 5 minutes must still produce a valid recommendation."""
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "fake-key")

    directions = _make_directions_response(_SHORT_POLYLINE, step_duration=120)
    sun_mock = _fixed_sun(azimuth=90.0, elevation=45.0)

    with (
        _patch_httpx(directions),
        patch("core.scorer.get_sun_position", side_effect=sun_mock),
        patch("api.routes.get_sun_position", side_effect=sun_mock),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/recommend",
                json={
                    "origin": "A St, NY",
                    "destination": "B St, NY",
                    "departure_time": "2024-06-21T10:00:00Z",
                    "drive_side": "LHD",
                },
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["best_seat"] in ("FL", "FR", "RL", "RR")
    assert data["worst_seat"] in ("FL", "FR", "RL", "RR")
    assert data["confidence"] in ("high", "moderate", "low")
    # Sun from due east → right seats exposed, left seats shaded
    assert data["best_seat"] in ("FL", "RL")


# ---------------------------------------------------------------------------
# Edge case 2 – North-south route at noon (sun roughly overhead)
#
# Vehicle heading north (0°), sun at azimuth 180° and high elevation.
# Relative angle = 180° → perpendicular to both left (270°) and right (90°)
# windows → cos(90°) = 0 for all seats → all scores ≈ 0.
# Summary should say "no significant difference".
# ---------------------------------------------------------------------------

# Northbound polyline: (40.0, -74.0) → (40.01, -74.0) → (40.02, -74.0)
_NORTHBOUND_POLYLINE = "_oydFvkrbMoR?oR?"


@pytest.mark.asyncio
async def test_noon_north_south_no_significant_difference(monkeypatch):
    """Sun directly ahead/behind on a N-S route → no seat difference."""
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "fake-key")

    directions = _make_directions_response(_NORTHBOUND_POLYLINE)
    # Sun due south at 80° elevation (roughly overhead at noon)
    sun_mock = _fixed_sun(azimuth=180.0, elevation=80.0)

    with (
        _patch_httpx(directions),
        patch("core.scorer.get_sun_position", side_effect=sun_mock),
        patch("api.routes.get_sun_position", side_effect=sun_mock),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/recommend",
                json={
                    "origin": "Philadelphia, PA",
                    "destination": "Trenton, NJ",
                    "departure_time": "2024-06-21T12:00:00Z",
                    "drive_side": "LHD",
                },
            )

    assert resp.status_code == 200
    data = resp.json()
    assert "no significant difference" in data["summary"].lower()
    # All scores should be near-zero (sun ahead/behind, not hitting side windows)
    scores = data["scores"]
    spread = max(scores.values()) - min(scores.values())
    assert spread < 0.1


# ---------------------------------------------------------------------------
# Edge case 3 – Missing weather data (solar API fails for some segments)
#
# Simulate get_sun_position raising an exception on every call.
# The pipeline should fall back to the time-of-day UV estimate and still
# return a result rather than crashing with a 500.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_weather_falls_back_to_time_estimate(monkeypatch):
    """When solar API fails, the pipeline uses time-of-day UV fallback."""
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "fake-key")

    directions = _make_directions_response(_EASTBOUND_POLYLINE)

    def _solar_api_fails(lat, lng, ts):
        raise ConnectionError("weather service unavailable")

    # The scorer's get_sun_position still works (separate mock);
    # only the weather-enrichment call in routes.py fails.
    scorer_sun = _fixed_sun(azimuth=90.0, elevation=45.0)

    with (
        _patch_httpx(directions),
        patch("core.scorer.get_sun_position", side_effect=scorer_sun),
        patch("api.routes.get_sun_position", side_effect=_solar_api_fails),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/recommend",
                json={
                    "origin": "New York, NY",
                    "destination": "Newark, NJ",
                    "departure_time": "2024-06-21T12:00:00Z",
                    "drive_side": "LHD",
                },
            )

    assert resp.status_code == 200
    data = resp.json()
    # The fallback UV at noon (hour 12) should be ~6.0 → scores are non-zero
    assert data["best_seat"] in ("FL", "FR", "RL", "RR")
    assert any(data["scores"][s] > 0 for s in ("FL", "FR", "RL", "RR"))


# ---------------------------------------------------------------------------
# Edge case 4 – Invalid origin/destination → clean 400 error
#
# Google Directions API returns ZERO_RESULTS or NOT_FOUND for invalid
# addresses. The endpoint must respond with 400 and a helpful message.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_origin_returns_400(monkeypatch):
    """Unresolvable address → Maps API ZERO_RESULTS → HTTP 400 with message."""
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "fake-key")

    error_json = {
        "status": "ZERO_RESULTS",
        "geocoded_waypoints": [],
        "routes": [],
    }

    with _patch_httpx(error_json):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/recommend",
                json={
                    "origin": "xyzzy nowhere 99999",
                    "destination": "aaaaa bbbbb 00000",
                    "departure_time": "2024-06-21T10:00:00Z",
                },
            )

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "ZERO_RESULTS" in detail


@pytest.mark.asyncio
async def test_invalid_destination_not_found_returns_400(monkeypatch):
    """Maps API NOT_FOUND status → HTTP 400 with message."""
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "fake-key")

    error_json = {
        "status": "NOT_FOUND",
        "error_message": "Origin and/or destination not found.",
    }

    with _patch_httpx(error_json):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/recommend",
                json={
                    "origin": "New York, NY",
                    "destination": "!!invalid!!",
                    "departure_time": "2024-06-21T10:00:00Z",
                },
            )

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "NOT_FOUND" in detail
