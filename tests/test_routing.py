"""Unit tests for core.routing helpers and get_route_segments."""
import math
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.routing import _bearing, _decode_polyline, get_route_segments

# ---------------------------------------------------------------------------
# Polyline documented by Google:
#   https://developers.google.com/maps/documentation/utilities/polylinealgorithm
# Decodes to: [(38.5, -120.2), (40.7, -120.95), (43.252, -126.453)]
# ---------------------------------------------------------------------------
_GOOGLE_POLYLINE = "_p~iF~ps|U_ulLnnqC_mqNvxq`@"
_GOOGLE_POINTS = [(38.5, -120.2), (40.7, -120.95), (43.252, -126.453)]

_DEPARTURE = datetime(2024, 6, 21, 8, 0, 0, tzinfo=timezone.utc)
_DEPARTURE_TS = _DEPARTURE.timestamp()


# ---------------------------------------------------------------------------
# _decode_polyline
# ---------------------------------------------------------------------------

def test_decode_polyline_google_example_count():
    points = _decode_polyline(_GOOGLE_POLYLINE)
    assert len(points) == 3


def test_decode_polyline_google_example_values():
    points = _decode_polyline(_GOOGLE_POLYLINE)
    for (got_lat, got_lng), (exp_lat, exp_lng) in zip(points, _GOOGLE_POINTS):
        assert abs(got_lat - exp_lat) < 1e-4, f"lat mismatch: {got_lat} vs {exp_lat}"
        assert abs(got_lng - exp_lng) < 1e-4, f"lng mismatch: {got_lng} vs {exp_lng}"


def test_decode_polyline_single_point():
    # A single point should decode cleanly without IndexError.
    # Encode (0.0, 0.0): diff=0 → result=0 → no sign flip → delta=0
    # The encoding of (0,0) is "??" (63+0=63='?' for both lat and lng)
    # But actually an empty delta encodes as '?' (63): let's use the empty string edge case
    # Instead, use a known single-point encoding
    # (1.0, 1.0): lat_e5=100000, lng_e5=100000
    # lat: 100000<<1=200000, chunks in 5-bit: 200000=0b110000110101000000
    #   from LSB: 00000,01010,00011,00011,0110 → add 63, set continue bit on all but last
    # This is complex to hand-compute; just verify decode(encode(p)) roundtrip via
    # the Google example truncated to the first point.
    first_point_encoded = "_p~iF~ps|U"  # first two coord-groups only
    points = _decode_polyline(first_point_encoded)
    assert len(points) == 1
    assert abs(points[0][0] - 38.5) < 1e-4
    assert abs(points[0][1] - (-120.2)) < 1e-4


# ---------------------------------------------------------------------------
# _bearing
# ---------------------------------------------------------------------------

def test_bearing_due_north():
    # Moving from equator northward → bearing ≈ 0°
    b = _bearing(0.0, 0.0, 1.0, 0.0)
    assert abs(b - 0.0) < 0.5


def test_bearing_due_east():
    # Moving eastward along equator → bearing ≈ 90°
    b = _bearing(0.0, 0.0, 0.0, 1.0)
    assert abs(b - 90.0) < 0.5


def test_bearing_due_south():
    b = _bearing(1.0, 0.0, 0.0, 0.0)
    assert abs(b - 180.0) < 0.5


def test_bearing_due_west():
    b = _bearing(0.0, 1.0, 0.0, 0.0)
    assert abs(b - 270.0) < 0.5


def test_bearing_northeast():
    b = _bearing(0.0, 0.0, 1.0, 1.0)
    assert 0 < b < 90


def test_bearing_result_in_range():
    # Should always be [0, 360)
    for lat1, lng1, lat2, lng2 in [
        (51.5, -0.1, 48.8, 2.3),   # London → Paris
        (40.7, -74.0, 34.0, -118.2),  # NYC → LA
        (-33.9, 151.2, 1.3, 103.8),   # Sydney → Singapore
    ]:
        b = _bearing(lat1, lng1, lat2, lng2)
        assert 0.0 <= b < 360.0, f"bearing out of range: {b}"


# ---------------------------------------------------------------------------
# Fixtures / helpers for get_route_segments
# ---------------------------------------------------------------------------

def _make_mock_response(polyline: str, step_duration: int = 600) -> MagicMock:
    """Build a MagicMock that looks like a successful Directions API response."""
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "status": "OK",
        "routes": [{
            "legs": [{
                "steps": [{
                    "start_location": {"lat": _GOOGLE_POINTS[0][0], "lng": _GOOGLE_POINTS[0][1]},
                    "end_location":   {"lat": _GOOGLE_POINTS[-1][0], "lng": _GOOGLE_POINTS[-1][1]},
                    "duration": {"value": step_duration},
                    "distance": {"value": 999},
                    "polyline": {"points": polyline},
                }]
            }]
        }]
    }
    return response


def _patch_httpx(mock_response: MagicMock):
    """Context manager: patch httpx.AsyncClient to return mock_response."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = AsyncMock(return_value=mock_response)
    return patch("httpx.AsyncClient", return_value=mock_client)


# ---------------------------------------------------------------------------
# get_route_segments — structure and count
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_returns_list_of_dicts():
    resp = _make_mock_response(_GOOGLE_POLYLINE)
    with _patch_httpx(resp):
        segments = await get_route_segments("A", "B", _DEPARTURE, maps_api_key="test")

    assert isinstance(segments, list)
    assert len(segments) > 0
    for seg in segments:
        assert isinstance(seg, dict)


@pytest.mark.asyncio
async def test_segment_count_equals_polyline_intervals():
    # 3-point polyline → 2 sub-segments
    resp = _make_mock_response(_GOOGLE_POLYLINE)
    with _patch_httpx(resp):
        segments = await get_route_segments("A", "B", _DEPARTURE, maps_api_key="test")

    assert len(segments) == 2


@pytest.mark.asyncio
async def test_segment_keys_present():
    resp = _make_mock_response(_GOOGLE_POLYLINE)
    with _patch_httpx(resp):
        segments = await get_route_segments("A", "B", _DEPARTURE, maps_api_key="test")

    required = {"lat", "lng", "timestamp_utc", "heading_degrees", "uv_index", "cloud_cover_pct"}
    for seg in segments:
        assert required <= seg.keys(), f"missing keys in segment: {seg}"


# ---------------------------------------------------------------------------
# get_route_segments — lat/lng values
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_first_segment_lat_lng_matches_polyline():
    resp = _make_mock_response(_GOOGLE_POLYLINE)
    with _patch_httpx(resp):
        segments = await get_route_segments("A", "B", _DEPARTURE, maps_api_key="test")

    assert abs(segments[0]["lat"] - _GOOGLE_POINTS[0][0]) < 1e-4
    assert abs(segments[0]["lng"] - _GOOGLE_POINTS[0][1]) < 1e-4


# ---------------------------------------------------------------------------
# get_route_segments — timestamps
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_first_segment_timestamp_equals_departure():
    resp = _make_mock_response(_GOOGLE_POLYLINE, step_duration=600)
    with _patch_httpx(resp):
        segments = await get_route_segments("A", "B", _DEPARTURE, maps_api_key="test")

    assert segments[0]["timestamp_utc"] == pytest.approx(_DEPARTURE_TS)


@pytest.mark.asyncio
async def test_timestamps_increase_monotonically():
    resp = _make_mock_response(_GOOGLE_POLYLINE, step_duration=600)
    with _patch_httpx(resp):
        segments = await get_route_segments("A", "B", _DEPARTURE, maps_api_key="test")

    for i in range(1, len(segments)):
        assert segments[i]["timestamp_utc"] > segments[i - 1]["timestamp_utc"]


@pytest.mark.asyncio
async def test_timestamps_spaced_by_step_duration_over_intervals():
    # 3 points → 2 intervals → each gets step_duration / 2 seconds
    step_duration = 600
    resp = _make_mock_response(_GOOGLE_POLYLINE, step_duration=step_duration)
    with _patch_httpx(resp):
        segments = await get_route_segments("A", "B", _DEPARTURE, maps_api_key="test")

    expected_gap = step_duration / 2  # 300 s
    gap = segments[1]["timestamp_utc"] - segments[0]["timestamp_utc"]
    assert gap == pytest.approx(expected_gap, rel=1e-6)


@pytest.mark.asyncio
async def test_naive_departure_treated_as_utc():
    naive = datetime(2024, 6, 21, 8, 0, 0)          # no tzinfo
    aware = datetime(2024, 6, 21, 8, 0, 0, tzinfo=timezone.utc)
    resp = _make_mock_response(_GOOGLE_POLYLINE)

    with _patch_httpx(resp):
        segs_naive = await get_route_segments("A", "B", naive, maps_api_key="test")
    with _patch_httpx(resp):
        segs_aware = await get_route_segments("A", "B", aware, maps_api_key="test")

    assert segs_naive[0]["timestamp_utc"] == pytest.approx(segs_aware[0]["timestamp_utc"])


# ---------------------------------------------------------------------------
# get_route_segments — heading
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_heading_in_valid_range():
    resp = _make_mock_response(_GOOGLE_POLYLINE)
    with _patch_httpx(resp):
        segments = await get_route_segments("A", "B", _DEPARTURE, maps_api_key="test")

    for seg in segments:
        assert 0.0 <= seg["heading_degrees"] < 360.0


@pytest.mark.asyncio
async def test_heading_matches_bearing_of_polyline_points():
    # First segment: (38.5, -120.2) → (40.7, -120.95)
    # That's roughly north-northwest; bearing should be < 360 and > 270 (or near 355°)
    resp = _make_mock_response(_GOOGLE_POLYLINE)
    with _patch_httpx(resp):
        segments = await get_route_segments("A", "B", _DEPARTURE, maps_api_key="test")

    from core.routing import _bearing as b
    expected = b(*_GOOGLE_POINTS[0], *_GOOGLE_POINTS[1])
    assert abs(segments[0]["heading_degrees"] - expected) < 0.01


# ---------------------------------------------------------------------------
# get_route_segments — placeholder weather fields
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_uv_and_cloud_are_placeholder_zeros():
    resp = _make_mock_response(_GOOGLE_POLYLINE)
    with _patch_httpx(resp):
        segments = await get_route_segments("A", "B", _DEPARTURE, maps_api_key="test")

    for seg in segments:
        assert seg["uv_index"] == 0.0
        assert seg["cloud_cover_pct"] == 0.0


# ---------------------------------------------------------------------------
# get_route_segments — API key resolution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No Maps API key"):
        await get_route_segments("A", "B", _DEPARTURE, maps_api_key=None)


@pytest.mark.asyncio
async def test_api_key_read_from_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "env-key")
    resp = _make_mock_response(_GOOGLE_POLYLINE)
    captured = {}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    async def capture_get(url, params):
        captured["key"] = params.get("key")
        return resp

    mock_client.get = capture_get

    with patch("httpx.AsyncClient", return_value=mock_client):
        await get_route_segments("A", "B", _DEPARTURE, maps_api_key=None)

    assert captured["key"] == "env-key"


# ---------------------------------------------------------------------------
# get_route_segments — error handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_raises_on_api_error_status():
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "status": "REQUEST_DENIED",
        "error_message": "API key invalid.",
    }
    with _patch_httpx(resp):
        with pytest.raises(ValueError, match="REQUEST_DENIED"):
            await get_route_segments("A", "B", _DEPARTURE, maps_api_key="bad-key")


@pytest.mark.asyncio
async def test_raises_on_zero_results():
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"status": "ZERO_RESULTS"}
    with _patch_httpx(resp):
        with pytest.raises(ValueError, match="ZERO_RESULTS"):
            await get_route_segments("A", "B", _DEPARTURE, maps_api_key="test")
