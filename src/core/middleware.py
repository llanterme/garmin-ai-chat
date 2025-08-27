"""Custom middleware for the application."""

import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .logging import get_logger

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details."""
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log response
        logger.info(
            f"Response: {response.status_code} for {request.method} {request.url.path} "
            f"in {duration:.4f}s"
        )
        
        # Add custom headers
        response.headers["X-Process-Time"] = str(duration)
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for additional error handling."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with error handling."""
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(
                f"Unhandled error in middleware for {request.method} {request.url.path}: {str(e)}",
                exc_info=True
            )
            # Re-raise to let FastAPI handle it
            raise