"""API route definitions."""
import logging
import math
from datetime import datetime, timezone
from typing import Literal

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from core.routing import get_route_segments
from core.scorer import score_seats
from core.solar import get_solar_position, get_sun_position
from core.scoring import score_seat

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    origin: str = Field(..., min_length=1, description="Origin address or 'lat,lng'")
    destination: str = Field(..., min_length=1, description="Destination address or 'lat,lng'")
    departure_time: datetime = Field(
        ..., description="Departure time in ISO 8601 format; naive timestamps assumed UTC"
    )
    drive_side: Literal["LHD", "RHD"] = Field(
        "LHD", description="LHD = driver front-left; RHD = driver front-right"
    )
    vehicle_type: str = Field("sedan", description="Vehicle type (informational)")


class SeatScores(BaseModel):
    FL: float
    FR: float
    RL: float
    RR: float


class RecommendResponse(BaseModel):
    best_seat: str
    worst_seat: str
    scores: SeatScores
    confidence: Literal["high", "moderate", "low"]
    summary: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SEAT_NAMES = {
    "FL": "Front Left",
    "FR": "Front Right",
    "RL": "Rear Left",
    "RR": "Rear Right",
}


_log = logging.getLogger(__name__)


def _estimate_uv_from_time(timestamp_utc: float) -> float:
    """Coarse UV estimate from UTC hour when the solar API is unavailable.

    Returns a bell-curve value peaking at ~6 around solar noon (12 UTC).
    """
    hour = datetime.fromtimestamp(timestamp_utc, tz=timezone.utc).hour
    if 6 <= hour <= 18:
        return round(max(0.0, 6.0 * math.sin(math.pi * (hour - 6) / 12)), 2)
    return 0.0


def _enrich_weather(segments: list[dict]) -> list[dict]:
    """
    Populate uv_index and cloud_cover_pct on each segment.

    Uses a clear-sky UV estimate derived from solar elevation:
        uv_index ≈ elevation / 9  (0 at horizon → ~10 at 90° zenith)
    cloud_cover_pct is set to 0.0 (clear-sky assumption; replace with
    a real weather API call to improve accuracy).

    If the solar position lookup fails for a segment, falls back to a
    coarse time-of-day UV estimate so the pipeline still returns a result.
    """
    enriched = []
    for seg in segments:
        try:
            solar = get_sun_position(seg["lat"], seg["lng"], seg["timestamp_utc"])
            uv = max(0.0, round(solar["elevation"] / 9.0, 2))
        except Exception:
            _log.warning(
                "Solar lookup failed for segment at (%.4f, %.4f); "
                "falling back to time-of-day UV estimate.",
                seg["lat"], seg["lng"],
            )
            uv = _estimate_uv_from_time(seg["timestamp_utc"])
        enriched.append({**seg, "uv_index": uv, "cloud_cover_pct": 0.0})
    return enriched


def _compute_confidence(
    enriched: list[dict],
    scores: dict,
) -> Literal["high", "moderate", "low"]:
    """
    Estimate recommendation confidence from three factors:
      1. Daytime fraction  – what share of the journey has sun above the horizon
      2. Average UV index  – intensity of solar exposure
      3. Relative spread   – how different best/worst scores are (larger = clearer winner)
    """
    if not enriched:
        return "low"

    sunny = [s for s in enriched if s["uv_index"] > 0]
    sun_fraction = len(sunny) / len(enriched)

    if not sunny:
        return "low"

    avg_uv = sum(s["uv_index"] for s in sunny) / len(sunny)

    worst = max(scores.values())
    best = min(scores.values())
    relative_spread = (worst - best) / worst if worst > 0 else 0.0

    if sun_fraction >= 0.7 and avg_uv >= 4.0 and relative_spread >= 0.3:
        return "high"
    if sun_fraction >= 0.3 and avg_uv >= 1.5 and relative_spread >= 0.1:
        return "moderate"
    return "low"


def _build_summary(
    best: str,
    worst: str,
    scores: dict,
    enriched: list[dict],
) -> str:
    """
    Human-readable recommendation line.

    Estimates 'direct sun minutes' for the worst seat by allocating total
    journey time proportionally to each seat's accumulated exposure score.
    Returns a "no significant difference" message when all seats receive
    roughly equal sun exposure (relative spread < 5%).
    """
    worst_score = max(scores.values())
    best_score = min(scores.values())
    relative_spread = (worst_score - best_score) / worst_score if worst_score > 0 else 0.0

    if relative_spread < 0.05:
        return "No significant difference between seats; sun exposure is roughly equal."

    journey_minutes = 0.0
    if len(enriched) >= 2:
        journey_minutes = (
            enriched[-1]["timestamp_utc"] - enriched[0]["timestamp_utc"]
        ) / 60.0

    total_score = sum(scores.values())

    if worst_score > 0 and journey_minutes > 0 and total_score > 0:
        sun_minutes = round((worst_score / total_score) * journey_minutes)
        return (
            f"{_SEAT_NAMES[best]} recommended. "
            f"{_SEAT_NAMES[worst]} gets ~{sun_minutes} min of direct sun."
        )
    return f"{_SEAT_NAMES[best]} recommended as the shadiest seat."


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/sun-position")
def sun_position(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    dt: datetime = Query(default=None, description="ISO datetime (UTC); defaults to now"),
):
    if dt is None:
        dt = datetime.now(timezone.utc)
    return get_solar_position(lat, lon, dt)


@router.get("/seat-score")
def seat_score(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    heading: float = Query(..., description="Vehicle heading in degrees (0=North, clockwise)"),
    dt: datetime = Query(default=None, description="ISO datetime (UTC); defaults to now"),
):
    if dt is None:
        dt = datetime.now(timezone.utc)
    solar = get_solar_position(lat, lon, dt)
    score = score_seat(
        solar_azimuth=solar["azimuth"],
        vehicle_heading=heading,
    )
    return {**solar, **score}


# ---------------------------------------------------------------------------
# POST /recommend
# ---------------------------------------------------------------------------

@router.post("/recommend", response_model=RecommendResponse)
async def recommend(body: RecommendRequest) -> RecommendResponse:
    """
    Full seat-recommendation pipeline:
      1. Fetch route segments from Google Maps Directions API
      2. Enrich each segment with a clear-sky UV estimate
      3. Score all four seats (FL / FR / RL / RR) over the route
      4. Return the best/worst seat with confidence and a human-readable summary
    """
    dt = body.departure_time
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # --- Step 1: routing ---
    try:
        segments = await get_route_segments(body.origin, body.destination, dt)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Maps API returned an unexpected HTTP error: {exc.response.status_code}",
        )

    if not segments:
        raise HTTPException(
            status_code=422,
            detail="The route produced no drivable segments. Check origin and destination.",
        )

    # --- Step 2: weather enrichment ---
    enriched = _enrich_weather(segments)

    # --- Step 3: scoring ---
    result = score_seats(enriched, drive_side=body.drive_side)

    # --- Step 4: response assembly ---
    scores_dict = {k: result[k] for k in ("FL", "FR", "RL", "RR")}
    confidence = _compute_confidence(enriched, scores_dict)
    summary = _build_summary(result["best_seat"], result["worst_seat"], scores_dict, enriched)

    return RecommendResponse(
        best_seat=result["best_seat"],
        worst_seat=result["worst_seat"],
        scores=SeatScores(**scores_dict),
        confidence=confidence,
        summary=summary,
    )
