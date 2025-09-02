"""Activity-related Pydantic schemas."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class ActivityBase(BaseModel):
    """Base activity schema with common fields."""

    activity_name: Optional[str] = Field(None, max_length=255, description="Activity name")
    activity_type: str = Field(..., max_length=100, description="Activity type")
    sport_type: Optional[str] = Field(None, max_length=100, description="Sport type")
    start_time: Optional[datetime] = Field(None, description="Activity start time")
    duration: Optional[float] = Field(None, ge=0, description="Activity duration in seconds")
    distance: Optional[float] = Field(None, ge=0, description="Activity distance in meters")
    calories: Optional[int] = Field(None, ge=0, description="Calories burned")


class ActivityCreate(BaseModel):
    """Schema for creating an activity."""

    garmin_activity_id: str = Field(..., description="Garmin activity ID")
    user_id: str = Field(..., description="User ID")
    activity_name: Optional[str] = Field(None, description="Activity name")
    activity_type: str = Field(..., description="Activity type")
    sport_type: Optional[str] = Field(None, description="Sport type")
    start_time: Optional[datetime] = Field(None, description="Activity start time")
    duration: Optional[float] = Field(None, ge=0, description="Duration in seconds")
    distance: Optional[float] = Field(None, ge=0, description="Distance in meters")
    calories: Optional[int] = Field(None, ge=0, description="Calories burned")
    raw_data: Optional[Dict[str, Any]] = Field(None, description="Raw activity data")
    summary_data: Optional[Dict[str, Any]] = Field(None, description="Activity summary data")

    # Performance metrics
    average_speed: Optional[float] = Field(None, ge=0, description="Average speed in m/s")
    max_speed: Optional[float] = Field(None, ge=0, description="Maximum speed in m/s")
    average_heart_rate: Optional[int] = Field(None, ge=0, le=250, description="Average heart rate")
    max_heart_rate: Optional[int] = Field(None, ge=0, le=250, description="Maximum heart rate")
    
    # Elevation
    elevation_gain: Optional[float] = Field(None, description="Elevation gain in meters")
    elevation_loss: Optional[float] = Field(None, description="Elevation loss in meters")
    min_elevation: Optional[float] = Field(None, description="Minimum elevation in meters")
    max_elevation: Optional[float] = Field(None, description="Maximum elevation in meters")
    
    # Power
    average_power: Optional[float] = Field(None, ge=0, description="Average power in watts")
    max_power: Optional[float] = Field(None, ge=0, description="Maximum power in watts")
    normalized_power: Optional[float] = Field(None, ge=0, description="Normalized power in watts")
    
    # Cycling
    average_cadence: Optional[float] = Field(None, ge=0, description="Average cadence in RPM")
    max_cadence: Optional[float] = Field(None, ge=0, description="Maximum cadence in RPM")
    
    # Swimming
    pool_length: Optional[float] = Field(None, ge=0, description="Pool length in meters")
    strokes: Optional[int] = Field(None, ge=0, description="Total strokes")
    swim_stroke_type: Optional[str] = Field(None, description="Swimming stroke type")
    
    # Training
    training_stress_score: Optional[float] = Field(None, ge=0, description="Training stress score")
    intensity_factor: Optional[float] = Field(None, ge=0, le=2, description="Intensity factor")
    vo2_max: Optional[float] = Field(None, ge=0, description="VO2 Max")
    
    # Location
    start_latitude: Optional[float] = Field(None, ge=-90, le=90, description="Start latitude")
    start_longitude: Optional[float] = Field(None, ge=-180, le=180, description="Start longitude")
    
    # Weather
    temperature: Optional[float] = Field(None, description="Temperature in Celsius")
    weather_condition: Optional[str] = Field(None, description="Weather condition")


class ActivityUpdate(BaseModel):
    """Schema for updating an activity."""

    activity_name: Optional[str] = Field(None, description="Activity name")
    activity_type: Optional[str] = Field(None, description="Activity type")
    sport_type: Optional[str] = Field(None, description="Sport type")
    raw_data: Optional[Dict[str, Any]] = Field(None, description="Raw activity data")
    summary_data: Optional[Dict[str, Any]] = Field(None, description="Activity summary data")


class ActivityResponse(BaseModel):
    """Schema for activity responses."""

    id: str = Field(..., description="Activity unique identifier")
    garmin_activity_id: str = Field(..., description="Garmin activity ID")
    user_id: str = Field(..., description="User ID")
    activity_name: Optional[str] = Field(None, description="Activity name")
    activity_type: str = Field(..., description="Activity type")
    sport_type: Optional[str] = Field(None, description="Sport type")
    start_time: Optional[datetime] = Field(None, description="Activity start time")
    duration: Optional[float] = Field(None, description="Duration in seconds")
    distance: Optional[float] = Field(None, description="Distance in meters")
    calories: Optional[int] = Field(None, description="Calories burned")
    
    # Performance metrics
    average_speed: Optional[float] = Field(None, description="Average speed in m/s")
    max_speed: Optional[float] = Field(None, description="Maximum speed in m/s")
    average_heart_rate: Optional[int] = Field(None, description="Average heart rate")
    max_heart_rate: Optional[int] = Field(None, description="Maximum heart rate")
    
    # Elevation
    elevation_gain: Optional[float] = Field(None, description="Elevation gain in meters")
    elevation_loss: Optional[float] = Field(None, description="Elevation loss in meters")
    
    # Power
    average_power: Optional[float] = Field(None, description="Average power in watts")
    max_power: Optional[float] = Field(None, description="Maximum power in watts")
    
    # Cycling
    average_cadence: Optional[float] = Field(None, description="Average cadence in RPM")
    max_cadence: Optional[float] = Field(None, description="Maximum cadence in RPM")
    
    # Metadata
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        from_attributes = True


class NormalizedMetrics(BaseModel):
    """Normalized metrics in human-readable units."""
    
    duration_formatted: Optional[str] = Field(None, description="Duration formatted (HH:MM:SS)")
    distance_km: Optional[float] = Field(None, description="Distance in kilometers")
    distance_miles: Optional[float] = Field(None, description="Distance in miles")
    average_speed_kmh: Optional[float] = Field(None, description="Average speed in km/h")
    average_speed_mph: Optional[float] = Field(None, description="Average speed in mph")
    max_speed_kmh: Optional[float] = Field(None, description="Maximum speed in km/h")
    max_speed_mph: Optional[float] = Field(None, description="Maximum speed in mph")
    average_pace_per_km: Optional[str] = Field(None, description="Average pace per km (MM:SS)")
    average_pace_per_mile: Optional[str] = Field(None, description="Average pace per mile (MM:SS)")
    elevation_gain_ft: Optional[float] = Field(None, description="Elevation gain in feet")


class ActivitySummary(BaseModel):
    """Activity summary schema for list views."""

    id: str = Field(..., description="Activity unique identifier")
    garmin_activity_id: str = Field(..., description="Garmin activity ID")
    activity_name: Optional[str] = Field(None, description="Activity name")
    activity_type: str = Field(..., description="Activity type")
    start_time: Optional[datetime] = Field(None, description="Activity start time")
    
    # Raw metrics (Garmin native format)
    duration: Optional[float] = Field(None, description="Duration in seconds")
    distance: Optional[float] = Field(None, description="Distance in meters")
    calories: Optional[int] = Field(None, description="Calories burned")
    average_speed: Optional[float] = Field(None, description="Average speed in m/s")
    average_heart_rate: Optional[int] = Field(None, description="Average heart rate")
    
    # Power metrics (cycling)
    average_power: Optional[float] = Field(None, description="Average power in watts")
    max_power: Optional[float] = Field(None, description="Maximum power in watts")
    normalized_power: Optional[float] = Field(None, description="Normalized power in watts")
    
    # Cadence (cycling/running)
    average_cadence: Optional[float] = Field(None, description="Average cadence in RPM")
    max_cadence: Optional[float] = Field(None, description="Maximum cadence in RPM")
    
    # Elevation
    elevation_gain: Optional[float] = Field(None, description="Elevation gain in meters")
    
    # Normalized human-readable metrics
    normalized: NormalizedMetrics = Field(default_factory=NormalizedMetrics, description="Human-readable metrics")

    class Config:
        from_attributes = True


class ActivityFilter(BaseModel):
    """Schema for filtering activities."""

    activity_type: Optional[str] = Field(None, description="Filter by activity type")
    start_date: Optional[datetime] = Field(None, description="Filter activities after this date")
    end_date: Optional[datetime] = Field(None, description="Filter activities before this date")
    min_distance: Optional[float] = Field(None, ge=0, description="Minimum distance in meters")
    max_distance: Optional[float] = Field(None, ge=0, description="Maximum distance in meters")
    min_duration: Optional[float] = Field(None, ge=0, description="Minimum duration in seconds")
    max_duration: Optional[float] = Field(None, ge=0, description="Maximum duration in seconds")

    @validator("end_date")
    def validate_date_range(cls, v, values):
        """Validate that end_date is after start_date."""
        if v and values.get("start_date") and v < values["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v

    @validator("max_distance")
    def validate_distance_range(cls, v, values):
        """Validate that max_distance is greater than min_distance."""
        if v and values.get("min_distance") and v < values["min_distance"]:
            raise ValueError("max_distance must be greater than min_distance")
        return v

    @validator("max_duration")
    def validate_duration_range(cls, v, values):
        """Validate that max_duration is greater than min_duration."""
        if v and values.get("min_duration") and v < values["min_duration"]:
            raise ValueError("max_duration must be greater than min_duration")
        return v


class ActivityListResponse(BaseModel):
    """Schema for paginated activity list responses."""

    items: List[ActivitySummary] = Field(..., description="List of activities")
    total: int = Field(..., description="Total number of activities")
    page: int = Field(..., description="Current page number")
    pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_prev: bool = Field(..., description="Whether there is a previous page")