"""Health check API endpoints."""

from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..db.base import get_db
from ..schemas.common import HealthCheck

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/", response_model=HealthCheck)
async def health_check(db: AsyncSession = Depends(get_db)) -> HealthCheck:
    """Health check endpoint."""
    # Test database connection
    try:
        await db.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"
    
    # For Garmin service, we'll just report as available since it's external
    garmin_status = "available"
    
    return HealthCheck(
        status="healthy" if db_status == "healthy" else "unhealthy",
        version=settings.app_version,
        timestamp=datetime.utcnow().isoformat(),
        database=db_status,
        garmin_service=garmin_status,
    )


@router.get("/ready")
async def readiness_check(db: AsyncSession = Depends(get_db)):
    """Readiness check endpoint for Kubernetes."""
    try:
        await db.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception:
        return {"status": "not ready"}, 503


@router.get("/live")
async def liveness_check():
    """Liveness check endpoint for Kubernetes."""
    return {"status": "alive"}