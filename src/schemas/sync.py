"""Synchronization-related Pydantic schemas."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class SyncRequest(BaseModel):
    """Schema for sync request."""

    days: int = Field(..., ge=1, le=365, description="Number of days to sync (1-365)")
    sync_type: str = Field(default="activities", description="Type of data to sync")
    force_resync: bool = Field(default=False, description="Force re-sync even if already synced")

    @validator("sync_type")
    def validate_sync_type(cls, v):
        """Validate sync type."""
        allowed_types = ["activities", "profile", "all"]
        if v not in allowed_types:
            raise ValueError(f"sync_type must be one of {allowed_types}")
        return v


class SyncStatus(BaseModel):
    """Schema for sync status response."""

    sync_id: str = Field(..., description="Sync operation ID")
    status: str = Field(..., description="Sync status")
    started_at: datetime = Field(..., description="Sync start time")
    completed_at: Optional[datetime] = Field(None, description="Sync completion time")
    duration_seconds: Optional[float] = Field(None, description="Sync duration in seconds")
    activities_synced: int = Field(default=0, description="Number of activities synced")
    activities_failed: int = Field(default=0, description="Number of activities that failed to sync")
    error_message: Optional[str] = Field(None, description="Error message if sync failed")
    progress_percentage: Optional[float] = Field(None, ge=0, le=100, description="Sync progress percentage")

    class Config:
        from_attributes = True


class SyncHistoryResponse(BaseModel):
    """Schema for sync history response."""

    id: str = Field(..., description="Sync history ID")
    sync_type: str = Field(..., description="Type of sync")
    status: str = Field(..., description="Sync status")
    start_date: Optional[datetime] = Field(None, description="Sync start date range")
    end_date: Optional[datetime] = Field(None, description="Sync end date range")
    started_at: datetime = Field(..., description="Sync start time")
    completed_at: Optional[datetime] = Field(None, description="Sync completion time")
    duration_seconds: Optional[float] = Field(None, description="Sync duration in seconds")
    activities_synced: int = Field(..., description="Number of activities synced")
    activities_failed: int = Field(..., description="Number of activities that failed")
    error_message: Optional[str] = Field(None, description="Error message if failed")

    class Config:
        from_attributes = True


class SyncHistoryListResponse(BaseModel):
    """Schema for paginated sync history list."""

    items: List[SyncHistoryResponse] = Field(..., description="List of sync history records")
    total: int = Field(..., description="Total number of sync records")
    page: int = Field(..., description="Current page number")
    pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_prev: bool = Field(..., description="Whether there is a previous page")


class SyncStats(BaseModel):
    """Schema for sync statistics."""

    total_syncs: int = Field(..., description="Total number of syncs")
    successful_syncs: int = Field(..., description="Number of successful syncs")
    failed_syncs: int = Field(..., description="Number of failed syncs")
    last_sync: Optional[datetime] = Field(None, description="Last sync timestamp")
    total_activities_synced: int = Field(..., description="Total activities synced")
    average_sync_duration: Optional[float] = Field(None, description="Average sync duration in seconds")


class GarminConnectionTest(BaseModel):
    """Schema for testing Garmin connection."""

    success: bool = Field(..., description="Whether connection test was successful")
    message: str = Field(..., description="Connection test message")
    user_info: Optional[dict] = Field(None, description="Garmin user info if successful")
    test_timestamp: datetime = Field(..., description="Test timestamp")