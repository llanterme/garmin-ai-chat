"""Authentication-related Pydantic schemas."""

from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class UserLogin(BaseModel):
    """User login schema."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=6, description="User password")


class UserRegister(BaseModel):
    """User registration schema."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=6, description="User password")
    full_name: Optional[str] = Field(None, max_length=255, description="User full name")


class GarminCredentials(BaseModel):
    """Garmin Connect credentials schema."""

    username: str = Field(..., min_length=1, description="Garmin Connect username")
    password: str = Field(..., min_length=1, description="Garmin Connect password")


class TokenResponse(BaseModel):
    """Token response schema."""

    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")


class TokenRefresh(BaseModel):
    """Token refresh schema."""

    refresh_token: str = Field(..., description="Refresh token")


class PasswordChange(BaseModel):
    """Password change schema."""

    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=6, description="New password")


class PasswordReset(BaseModel):
    """Password reset schema."""

    email: EmailStr = Field(..., description="User email address")


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation schema."""

    token: str = Field(..., description="Password reset token")
    new_password: str = Field(..., min_length=6, description="New password")