"""Routing: fetch a Google Maps route and convert it to scorer-ready segments."""
import math
import os
from datetime import datetime, timezone
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _decode_polyline(encoded: str) -> list[tuple[float, float]]:
    """
    Decode a Google Maps encoded polyline string into (lat, lng) pairs.

    Uses the standard 5-bit chunk algorithm documented at
    https://developers.google.com/maps/documentation/utilities/polylinealgorithm
    """
    points: list[tuple[float, float]] = []
    index = 0
    lat = 0
    lng = 0
    n = len(encoded)

    while index < n:
        # Decode one coordinate (lat then lng)
        for is_lng in range(2):
            shift = 0
            result = 0
            while True:
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:        # highest bit clear → last chunk
                    break
            delta = ~(result >> 1) if (result & 1) else (result >> 1)
            if is_lng:
                lng += delta
            else:
                lat += delta

        points.append((lat / 1e5, lng / 1e5))

    return points


def _bearing(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Compute the forward azimuth (0–360°, clockwise from north) from
    point 1 to point 2 using the spherical law of cosines.
    """
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlng_r = math.radians(lng2 - lng1)

    x = math.sin(dlng_r) * math.cos(lat2_r)
    y = (math.cos(lat1_r) * math.sin(lat2_r)
         - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlng_r))

    return (math.degrees(math.atan2(x, y)) + 360) % 360


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def get_route_segments(
    origin: str,
    destination: str,
    departure_time: datetime,
    maps_api_key: Optional[str] = None,
) -> list[dict]:
    """
    Fetch a driving route from Google Maps and return scorer-ready segments.

    Each returned segment dict contains:
        lat             – float, decimal degrees
        lng             – float, decimal degrees
        timestamp_utc   – float, Unix timestamp (UTC) at which the vehicle
                          is expected to be at this point
        heading_degrees – float, bearing to the next point (0 = north)
        uv_index        – float, placeholder 0.0 (populate from weather API)
        cloud_cover_pct – float, placeholder 0.0 (populate from weather API)

    Args:
        origin:         Address or "lat,lng" string for the start point.
        destination:    Address or "lat,lng" string for the end point.
        departure_time: Departure time (naive → assumed UTC).
        maps_api_key:   Google Maps API key. Falls back to the
                        GOOGLE_MAPS_API_KEY environment variable / .env file.

    Raises:
        ValueError: If no API key is available or the Directions API returns
                    an error status.
        httpx.HTTPStatusError: On HTTP-level errors.
    """
    api_key = maps_api_key or os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise ValueError(
            "No Maps API key provided. "
            "Set GOOGLE_MAPS_API_KEY in .env or pass maps_api_key=."
        )

    # Normalise to UTC-aware datetime → Unix timestamp
    if departure_time.tzinfo is None:
        departure_time = departure_time.replace(tzinfo=timezone.utc)
    departure_ts = departure_time.timestamp()

    params = {
        "origin": origin,
        "destination": destination,
        "departure_time": int(departure_ts),
        "key": api_key,
    }

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(_DIRECTIONS_URL, params=params)
        response.raise_for_status()
        data = response.json()

    status = data.get("status")
    if status != "OK":
        raise ValueError(
            f"Directions API error: {status} — "
            f"{data.get('error_message', 'no details')}"
        )

    segments: list[dict] = []
    elapsed = 0.0     # seconds since departure

    for leg in data["routes"][0]["legs"]:
        for step in leg["steps"]:
            step_duration: float = step["duration"]["value"]   # seconds
            points = _decode_polyline(step["polyline"]["points"])

            if len(points) < 2:
                # Degenerate step with a single point: use step endpoints for heading.
                lat, lng = points[0] if points else (
                    step["start_location"]["lat"],
                    step["start_location"]["lng"],
                )
                heading = _bearing(
                    step["start_location"]["lat"], step["start_location"]["lng"],
                    step["end_location"]["lat"],   step["end_location"]["lng"],
                )
                segments.append(_make_segment(lat, lng, departure_ts + elapsed, heading))
                elapsed += step_duration
                continue

            # Distribute the step's duration evenly across sub-segments.
            time_per_sub = step_duration / (len(points) - 1)

            for i in range(len(points) - 1):
                lat1, lng1 = points[i]
                lat2, lng2 = points[i + 1]
                segments.append(_make_segment(
                    lat1, lng1,
                    departure_ts + elapsed,
                    _bearing(lat1, lng1, lat2, lng2),
                ))
                elapsed += time_per_sub

    return segments


def _make_segment(
    lat: float, lng: float, timestamp_utc: float, heading_degrees: float
) -> dict:
    return {
        "lat": lat,
        "lng": lng,
        "timestamp_utc": timestamp_utc,
        "heading_degrees": heading_degrees,
        "uv_index": 0.0,        # caller should populate from weather API
        "cloud_cover_pct": 0.0, # caller should populate from weather API
    }


# ---------------------------------------------------------------------------
# Legacy helper (kept for backwards compatibility with api/routes.py)
# ---------------------------------------------------------------------------

async def get_route(origin: str, destination: str, api_key: str) -> list[dict]:
    """Return a flat list of {lat, lon} waypoints (step start-points only)."""
    params = {"origin": origin, "destination": destination, "key": api_key}
    async with httpx.AsyncClient() as client:
        response = await client.get(_DIRECTIONS_URL, params=params)
        response.raise_for_status()
        data = response.json()

    waypoints = []
    if data.get("routes"):
        legs = data["routes"][0]["legs"]
        for leg in legs:
            for step in leg["steps"]:
                waypoints.append({
                    "lat": step["start_location"]["lat"],
                    "lon": step["start_location"]["lng"],
                })
        last_leg = legs[-1]
        waypoints.append({
            "lat": last_leg["end_location"]["lat"],
            "lon": last_leg["end_location"]["lng"],
        })
    return waypoints
