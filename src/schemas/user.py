"""User-related Pydantic schemas."""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, EmailStr, Field


class UserBase(BaseModel):
    """Base user schema with common fields."""

    email: EmailStr = Field(..., description="User email address")
    full_name: Optional[str] = Field(None, max_length=255, description="User full name")
    is_active: bool = Field(default=True, description="Whether user is active")


class UserCreate(UserBase):
    """Schema for creating a user."""

    password: str = Field(..., min_length=6, description="User password")


class UserUpdate(BaseModel):
    """Schema for updating a user."""

    email: Optional[EmailStr] = Field(None, description="User email address")
    full_name: Optional[str] = Field(None, max_length=255, description="User full name")
    is_active: Optional[bool] = Field(None, description="Whether user is active")


class UserGarminUpdate(BaseModel):
    """Schema for updating user's Garmin credentials."""

    garmin_username: str = Field(..., description="Garmin Connect username")
    garmin_password: str = Field(..., description="Garmin Connect password")


class UserResponse(UserBase):
    """Schema for user responses."""

    id: str = Field(..., description="User unique identifier")
    created_at: datetime = Field(..., description="User creation timestamp")
    updated_at: datetime = Field(..., description="User last update timestamp")
    last_login: Optional[datetime] = Field(None, description="User last login timestamp")
    last_garmin_sync: Optional[datetime] = Field(None, description="Last Garmin sync timestamp")
    has_garmin_credentials: bool = Field(..., description="Whether user has Garmin credentials")

    class Config:
        from_attributes = True

    @classmethod
    def from_db_model(cls, user) -> "UserResponse":
        """Create UserResponse from database model."""
        return cls(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            is_active=user.is_active,
            created_at=user.created_at,
            updated_at=user.updated_at,
            last_login=user.last_login,
            last_garmin_sync=user.last_garmin_sync,
            has_garmin_credentials=bool(user.garmin_username and user.garmin_password),
        )


class UserProfile(BaseModel):
    """Complete user profile schema."""

    id: str = Field(..., description="User unique identifier")
    email: EmailStr = Field(..., description="User email address")
    full_name: Optional[str] = Field(None, description="User full name")
    is_active: bool = Field(..., description="Whether user is active")
    created_at: datetime = Field(..., description="User creation timestamp")
    updated_at: datetime = Field(..., description="User last update timestamp")
    last_login: Optional[datetime] = Field(None, description="User last login timestamp")
    last_garmin_sync: Optional[datetime] = Field(None, description="Last Garmin sync timestamp")
    has_garmin_credentials: bool = Field(..., description="Whether user has Garmin credentials")
    total_activities: int = Field(default=0, description="Total number of activities")
    activity_types: list[str] = Field(default_factory=list, description="List of activity types")

    class Config:
        from_attributes = True