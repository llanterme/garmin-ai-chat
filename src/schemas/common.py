"""Common Pydantic schemas used across the application."""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standard error response schema."""

    error: str = Field(..., description="Error type or code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")
    path: Optional[str] = Field(None, description="Request path that caused the error")


class SuccessResponse(BaseModel):
    """Standard success response schema."""

    success: bool = Field(default=True, description="Success status")
    message: str = Field(..., description="Success message")
    data: Optional[Union[Dict[str, Any], List[Any], str, int, bool]] = Field(
        None, description="Response data"
    )


class PaginationParams(BaseModel):
    """Pagination parameters schema."""

    page: int = Field(default=1, ge=1, description="Page number (starting from 1)")
    size: int = Field(default=20, ge=1, le=100, description="Number of items per page")

    @property
    def skip(self) -> int:
        """Calculate the number of items to skip."""
        return (self.page - 1) * self.size

    @property
    def limit(self) -> int:
        """Get the limit value."""
        return self.size


class PaginationResponse(BaseModel):
    """Base pagination response schema."""

    page: int = Field(..., description="Current page number")
    size: int = Field(..., description="Number of items per page")
    total: int = Field(..., description="Total number of items")
    pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_prev: bool = Field(..., description="Whether there is a previous page")

    @classmethod
    def create(cls, page: int, size: int, total: int) -> "PaginationResponse":
        """Create pagination response."""
        pages = (total + size - 1) // size  # Ceiling division
        return cls(
            page=page,
            size=size,
            total=total,
            pages=pages,
            has_next=page < pages,
            has_prev=page > 1,
        )


class HealthCheck(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    timestamp: str = Field(..., description="Health check timestamp")
    database: str = Field(..., description="Database connection status")
    garmin_service: str = Field(..., description="Garmin service status")


class BulkOperationResponse(BaseModel):
    """Response schema for bulk operations."""

    total_requested: int = Field(..., description="Total number of items requested for operation")
    successful: int = Field(..., description="Number of items processed successfully")
    failed: int = Field(..., description="Number of items that failed processing")
    errors: List[str] = Field(default_factory=list, description="List of error messages")
    success_rate: float = Field(..., description="Success rate as a percentage")

    @classmethod
    def create(cls, total: int, successful: int, errors: List[str]) -> "BulkOperationResponse":
        """Create bulk operation response."""
        failed = total - successful
        success_rate = (successful / total * 100) if total > 0 else 0
        return cls(
            total_requested=total,
            successful=successful,
            failed=failed,
            errors=errors,
            success_rate=success_rate,
        )


class ConfigInfo(BaseModel):
    """Configuration information schema."""

    app_name: str = Field(..., description="Application name")
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Environment (development, staging, production)")
    features: Dict[str, bool] = Field(..., description="Enabled features")
    limits: Dict[str, int] = Field(..., description="Application limits")


class ApiInfo(BaseModel):
    """API information schema."""

    title: str = Field(..., description="API title")
    description: str = Field(..., description="API description")
    version: str = Field(..., description="API version")
    contact: Dict[str, str] = Field(..., description="Contact information")
    license: Dict[str, str] = Field(..., description="License information")
    docs_url: str = Field(..., description="Documentation URL")
    redoc_url: str = Field(..., description="ReDoc URL")