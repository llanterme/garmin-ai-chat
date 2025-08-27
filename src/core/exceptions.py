"""Custom exception classes for the application."""

from typing import Any, Dict, Optional


class GarminAIChatException(Exception):
    """Base exception for the application."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class AuthenticationError(GarminAIChatException):
    """Authentication related errors."""

    def __init__(self, message: str = "Authentication failed", details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, status_code=401, details=details)


class AuthorizationError(GarminAIChatException):
    """Authorization related errors."""

    def __init__(self, message: str = "Access denied", details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, status_code=403, details=details)


class ValidationError(GarminAIChatException):
    """Data validation errors."""

    def __init__(self, message: str = "Validation error", details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, status_code=422, details=details)


class NotFoundError(GarminAIChatException):
    """Resource not found errors."""

    def __init__(self, message: str = "Resource not found", details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, status_code=404, details=details)


class ConflictError(GarminAIChatException):
    """Resource conflict errors."""

    def __init__(self, message: str = "Resource conflict", details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, status_code=409, details=details)


class DatabaseError(GarminAIChatException):
    """Database operation errors."""

    def __init__(self, message: str = "Database error", details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, status_code=500, details=details)


class GarminConnectError(GarminAIChatException):
    """Garmin Connect API errors."""

    def __init__(self, message: str = "Garmin Connect error", details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, status_code=502, details=details)


class RateLimitError(GarminAIChatException):
    """Rate limit exceeded errors."""

    def __init__(self, message: str = "Rate limit exceeded", details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, status_code=429, details=details)