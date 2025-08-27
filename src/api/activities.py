"""Activity API endpoints."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.exceptions import NotFoundError
from ..db.base import get_db
from ..db.models import User
from ..db.repositories import ActivityRepository
from ..schemas.activity import (
    ActivityFilter,
    ActivityListResponse,
    ActivityResponse,
    ActivitySummary,
)
from ..schemas.common import PaginationParams, PaginationResponse
from .dependencies import get_current_active_user

router = APIRouter(prefix="/activities", tags=["Activities"])


@router.get("/", response_model=ActivityListResponse)
async def get_activities(
    activity_type: Optional[str] = Query(None, description="Filter by activity type"),
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Items per page"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> ActivityListResponse:
    """Get user's activities with optional filtering and pagination."""
    try:
        activity_repo = ActivityRepository(db)
        
        # Calculate pagination
        pagination = PaginationParams(page=page, size=size)
        
        # Get activities
        activities = await activity_repo.get_user_activities(
            user_id=current_user.id,
            activity_type=activity_type,
            skip=pagination.skip,
            limit=pagination.limit,
        )
        
        # Get total count
        total = await activity_repo.count_user_activities(
            user_id=current_user.id,
            activity_type=activity_type,
        )
        
        # Create response
        activity_summaries = [ActivitySummary.model_validate(activity) for activity in activities]
        pagination_response = PaginationResponse.create(page, size, total)
        
        return ActivityListResponse(
            items=activity_summaries,
            total=pagination_response.total,
            page=pagination_response.page,
            pages=pagination_response.pages,
            has_next=pagination_response.has_next,
            has_prev=pagination_response.has_prev,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve activities: {str(e)}"
        )


@router.get("/{activity_id}", response_model=ActivityResponse)
async def get_activity(
    activity_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> ActivityResponse:
    """Get a specific activity by ID."""
    try:
        activity_repo = ActivityRepository(db)
        activity = await activity_repo.get_by_id(activity_id)
        
        if not activity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Activity not found"
            )
            
        # Check if activity belongs to current user
        if activity.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
            
        return ActivityResponse.model_validate(activity)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve activity: {str(e)}"
        )


@router.get("/types/", response_model=list[str])
async def get_activity_types(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> list[str]:
    """Get unique activity types for the current user."""
    try:
        activity_repo = ActivityRepository(db)
        activity_types = await activity_repo.get_activity_types_for_user(current_user.id)
        return activity_types
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve activity types: {str(e)}"
        )


@router.delete("/{activity_id}")
async def delete_activity(
    activity_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a specific activity."""
    try:
        activity_repo = ActivityRepository(db)
        activity = await activity_repo.get_by_id(activity_id)
        
        if not activity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Activity not found"
            )
            
        # Check if activity belongs to current user
        if activity.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
            
        success = await activity_repo.delete(activity_id)
        
        if success:
            return {"message": "Activity deleted successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete activity"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete activity: {str(e)}"
        )