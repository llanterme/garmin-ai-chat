"""Synchronization API endpoints."""

from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.exceptions import GarminConnectError
from ..db.base import get_db
from ..db.models import User
from ..schemas.common import PaginationParams, PaginationResponse, SuccessResponse
from ..schemas.sync import (
    SyncHistoryListResponse,
    SyncHistoryResponse,
    SyncRequest,
    SyncStats,
    SyncStatus,
)
from ..services.sync import SyncService
from .dependencies import get_current_active_user

router = APIRouter(prefix="/sync", tags=["Synchronization"])


async def background_sync_activities(user_id: str, days: int, force_resync: bool, db: AsyncSession):
    """Background task for syncing activities."""
    sync_service = SyncService(db)
    try:
        await sync_service.sync_user_activities(user_id, days, force_resync)
    except Exception as e:
        # Error handling is done within the sync service
        pass


@router.post("/activities", response_model=SyncStatus, status_code=status.HTTP_202_ACCEPTED)
async def sync_activities(
    sync_request: SyncRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> SyncStatus:
    """Start activity synchronization from Garmin Connect."""
    try:
        sync_service = SyncService(db)
        
        # Start sync in background
        sync_id = await sync_service.sync_user_activities(
            user_id=current_user.id,
            days=sync_request.days,
            force_resync=sync_request.force_resync,
        )
        
        # Get initial status
        sync_status = await sync_service.get_sync_status(sync_id)
        
        if not sync_status:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to start synchronization"
            )
            
        return SyncStatus(**sync_status)
        
    except GarminConnectError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start synchronization: {str(e)}"
        )


@router.get("/status/{sync_id}", response_model=SyncStatus)
async def get_sync_status(
    sync_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> SyncStatus:
    """Get synchronization status."""
    try:
        sync_service = SyncService(db)
        sync_status = await sync_service.get_sync_status(sync_id)
        
        if not sync_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Sync record not found"
            )
            
        return SyncStatus(**sync_status)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get sync status: {str(e)}"
        )


@router.get("/history", response_model=SyncHistoryListResponse)
async def get_sync_history(
    sync_type: Optional[str] = Query(None, description="Filter by sync type"),
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Items per page"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> SyncHistoryListResponse:
    """Get synchronization history for the current user."""
    try:
        sync_service = SyncService(db)
        
        # Calculate pagination
        pagination = PaginationParams(page=page, size=size)
        
        # Get history
        history = await sync_service.get_user_sync_history(
            user_id=current_user.id,
            sync_type=sync_type,
            skip=pagination.skip,
            limit=pagination.limit,
        )
        
        # For simplicity, we'll estimate total from the returned results
        # In a real implementation, you'd want a separate count query
        total = len(history)
        if len(history) == pagination.limit:
            total = pagination.skip + pagination.limit + 1  # Estimate there's more
            
        history_responses = [SyncHistoryResponse(**record) for record in history]
        pagination_response = PaginationResponse.create(page, size, total)
        
        return SyncHistoryListResponse(
            items=history_responses,
            total=pagination_response.total,
            page=pagination_response.page,
            pages=pagination_response.pages,
            has_next=pagination_response.has_next,
            has_prev=pagination_response.has_prev,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sync history: {str(e)}"
        )


@router.get("/stats", response_model=SyncStats)
async def get_sync_stats(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> SyncStats:
    """Get synchronization statistics for the current user."""
    try:
        sync_service = SyncService(db)
        stats = await sync_service.get_sync_stats(current_user.id)
        return SyncStats(**stats)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get sync statistics: {str(e)}"
        )