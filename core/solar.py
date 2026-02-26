"""Solar position and irradiance calculations using pvlib."""
import pvlib
import pandas as pd
from datetime import datetime


def get_sun_position(lat: float, lng: float, timestamp_utc: float) -> dict:
    """
    Return solar azimuth and elevation for a given location and UTC timestamp.

    Args:
        lat: Latitude in decimal degrees.
        lng: Longitude in decimal degrees.
        timestamp_utc: Unix timestamp (seconds since epoch, UTC).

    Returns:
        dict with keys:
            azimuth   – degrees clockwise from north (0–360)
            elevation – degrees above horizon (negative when below)
    """
    times = pd.DatetimeIndex([pd.Timestamp(timestamp_utc, unit="s", tz="UTC")])
    location = pvlib.location.Location(latitude=lat, longitude=lng)
    solar_pos = location.get_solarposition(times)
    return {
        "azimuth": round(float(solar_pos["azimuth"].iloc[0]), 4),
        "elevation": round(float(solar_pos["apparent_elevation"].iloc[0]), 4),
    }


def get_solar_position(lat: float, lon: float, dt: datetime) -> dict:
    """Return solar azimuth and elevation for a given location and time."""
    times = pd.DatetimeIndex([dt])
    location = pvlib.location.Location(latitude=lat, longitude=lon)
    solar_pos = location.get_solarposition(times)
    return {
        "azimuth": solar_pos["azimuth"].iloc[0],
        "elevation": solar_pos["apparent_elevation"].iloc[0],
    }


def get_irradiance(lat: float, lon: float, dt: datetime) -> dict:
    """Return estimated GHI, DNI, DHI for a location and time."""
    times = pd.DatetimeIndex([dt])
    location = pvlib.location.Location(latitude=lat, longitude=lon)
    solar_pos = location.get_solarposition(times)
    clearsky = location.get_clearsky(times)
    return {
        "ghi": clearsky["ghi"].iloc[0],
        "dni": clearsky["dni"].iloc[0],
        "dhi": clearsky["dhi"].iloc[0],
        "elevation": solar_pos["apparent_elevation"].iloc[0],
    }
