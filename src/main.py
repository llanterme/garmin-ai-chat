"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import activities, auth, chat, health, sync
from .core.config import settings
from .core.exceptions import GarminAIChatException
from .core.logging import configure_logging, get_logger
from .db.base import init_db

# Configure logging
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Garmin AI Chat API...")
    
    try:
        # Initialize database
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Garmin AI Chat API...")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="FastAPI backend for Garmin AI Chat with activity sync and analysis",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    **settings.cors_config,
)


# Global exception handler
@app.exception_handler(GarminAIChatException)
async def garmin_exception_handler(request: Request, exc: GarminAIChatException):
    """Handle custom application exceptions."""
    logger.error(f"Application error: {exc.message} - Details: {exc.details}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "details": exc.details,
            "timestamp": str(request.url),
            "path": str(request.url.path),
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "details": {} if not settings.debug else {"exception": str(exc)},
            "timestamp": str(request.url),
            "path": str(request.url.path),
        }
    )


# Include routers
app.include_router(health.router)
app.include_router(auth.router, prefix="/api/v1")
app.include_router(activities.router, prefix="/api/v1")
app.include_router(sync.router, prefix="/api/v1")
app.include_router(chat.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs_url": "/docs" if settings.debug else None,
    }


@app.get("/api/info")
async def api_info():
    """API information endpoint."""
    return {
        "title": settings.app_name,
        "description": "FastAPI backend for Garmin AI Chat with activity sync and analysis",
        "version": settings.app_version,
        "environment": settings.environment,
        "debug": settings.debug,
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload and settings.debug,
        log_level=settings.log_level.lower(),
    )