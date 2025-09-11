"""Utilities for converting and formatting activity metrics."""

from typing import Optional


def format_duration(seconds: Optional[float]) -> Optional[str]:
    """Convert duration in seconds to HH:MM:SS format."""
    if seconds is None:
        return None
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def meters_to_km(meters: Optional[float]) -> Optional[float]:
    """Convert meters to kilometers."""
    if meters is None:
        return None
    return round(meters / 1000, 2)


def meters_to_miles(meters: Optional[float]) -> Optional[float]:
    """Convert meters to miles."""
    if meters is None:
        return None
    return round(meters * 0.000621371, 2)


def mps_to_kmh(mps: Optional[float]) -> Optional[float]:
    """Convert meters per second to kilometers per hour."""
    if mps is None:
        return None
    return round(mps * 3.6, 1)


def mps_to_mph(mps: Optional[float]) -> Optional[float]:
    """Convert meters per second to miles per hour."""
    if mps is None:
        return None
    return round(mps * 2.23694, 1)


def pace_per_km(mps: Optional[float]) -> Optional[str]:
    """Convert speed (m/s) to pace per kilometer (MM:SS)."""
    if mps is None or mps == 0:
        return None
    
    # Time in seconds to cover 1 km
    seconds_per_km = 1000 / mps
    minutes = int(seconds_per_km // 60)
    seconds = int(seconds_per_km % 60)
    
    return f"{minutes:02d}:{seconds:02d}"


def pace_per_mile(mps: Optional[float]) -> Optional[str]:
    """Convert speed (m/s) to pace per mile (MM:SS)."""
    if mps is None or mps == 0:
        return None
    
    # Time in seconds to cover 1 mile (1609.34 meters)
    seconds_per_mile = 1609.34 / mps
    minutes = int(seconds_per_mile // 60)
    seconds = int(seconds_per_mile % 60)
    
    return f"{minutes:02d}:{seconds:02d}"


def meters_to_feet(meters: Optional[float]) -> Optional[float]:
    """Convert meters to feet."""
    if meters is None:
        return None
    return round(meters * 3.28084, 0)