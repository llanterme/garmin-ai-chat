"""Database models for the application."""

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.mysql import CHAR
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .base import Base


def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())


class User(Base):
    """User model for storing user information and Garmin credentials."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        CHAR(36), primary_key=True, default=generate_uuid, index=True
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    
    # Encrypted Garmin credentials
    garmin_username: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    garmin_password: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    garmin_session_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_garmin_sync: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email})>"


class Activity(Base):
    """Activity model for storing Garmin activity data."""

    __tablename__ = "activities"

    id: Mapped[str] = mapped_column(
        CHAR(36), primary_key=True, default=generate_uuid, index=True
    )
    user_id: Mapped[str] = mapped_column(
        CHAR(36), index=True, nullable=False
    )  # Foreign key to users table
    
    # Garmin-specific identifiers
    garmin_activity_id: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    
    # Basic activity information
    activity_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    activity_type: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    sport_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Timing information
    start_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    duration: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # seconds
    
    # Distance and location
    distance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # meters
    start_latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    start_longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Performance metrics
    calories: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    average_speed: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # m/s
    max_speed: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # m/s
    average_heart_rate: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # bpm
    max_heart_rate: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # bpm
    
    # Elevation data
    elevation_gain: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # meters
    elevation_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # meters
    min_elevation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # meters
    max_elevation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # meters
    
    # Power data (cycling/running)
    average_power: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # watts
    max_power: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # watts
    normalized_power: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # watts
    
    # Cycling-specific metrics
    average_cadence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # rpm
    max_cadence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # rpm
    
    # Swimming-specific metrics
    pool_length: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # meters
    strokes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    swim_stroke_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Training metrics
    training_stress_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    intensity_factor: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    vo2_max: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Weather data
    temperature: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # celsius
    weather_condition: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Raw data storage
    raw_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    summary_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<Activity(id={self.id}, type={self.activity_type}, start_time={self.start_time})>"


class SyncHistory(Base):
    """Track synchronization history for users."""

    __tablename__ = "sync_history"

    id: Mapped[str] = mapped_column(
        CHAR(36), primary_key=True, default=generate_uuid, index=True
    )
    user_id: Mapped[str] = mapped_column(
        CHAR(36), index=True, nullable=False
    )  # Foreign key to users table
    
    # Sync parameters
    sync_type: Mapped[str] = mapped_column(String(50), nullable=False)  # 'activities', 'profile', etc.
    start_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    end_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Sync results
    status: Mapped[str] = mapped_column(String(20), nullable=False)  # 'success', 'partial', 'failed'
    activities_synced: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    activities_failed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Timing
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<SyncHistory(id={self.id}, user_id={self.user_id}, status={self.status})>"


class BackgroundTask(Base):
    """Track background task execution and status."""

    __tablename__ = "background_tasks"

    id: Mapped[str] = mapped_column(
        CHAR(36), primary_key=True, default=generate_uuid, index=True
    )
    user_id: Mapped[str] = mapped_column(
        CHAR(36), index=True, nullable=False
    )  # Foreign key to users table
    
    # Task details
    task_type: Mapped[str] = mapped_column(String(50), nullable=False)  # 'sync', 'ingestion', 'cleanup'
    task_name: Mapped[str] = mapped_column(String(100), nullable=False)  # Descriptive name
    status: Mapped[str] = mapped_column(String(20), nullable=False)  # 'pending', 'running', 'completed', 'failed'
    
    # Progress tracking
    progress_percentage: Mapped[Optional[float]] = mapped_column(Float, default=0.0, nullable=True)
    progress_message: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Results
    result_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Timing
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    
    def __repr__(self) -> str:
        return f"<BackgroundTask(id={self.id}, task_type={self.task_type}, status={self.status})>"