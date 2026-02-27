"""Activity API endpoints."""

from datetime import date, datetime, timedelta
from typing import Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
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
    DateWindow,
    NormalizedMetrics,
    ViewType,
)
from ..schemas.common import PaginationParams, PaginationResponse
from ..services.activity_views import ActivityViewService
from ..utils.metrics import (
    format_duration,
    meters_to_km,
    meters_to_miles,
    mps_to_kmh,
    mps_to_mph,
    pace_per_km,
    pace_per_mile,
    meters_to_feet,
)
from .dependencies import get_current_active_user

router = APIRouter(prefix="/activities", tags=["Activities"])


@router.get("/")
async def get_activities(
    # --- existing params (backward compatible) ---
    activity_type: Optional[str] = Query(None, description="Filter by activity type"),
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Items per page"),
    # --- new params ---
    days_back: Optional[int] = Query(
        None, ge=1, le=365, description="Fetch last N days of activities"
    ),
    start_date: Optional[date] = Query(
        None, description="Window start date (YYYY-MM-DD)"
    ),
    end_date: Optional[date] = Query(
        None, description="Window end date (YYYY-MM-DD)"
    ),
    view: Optional[ViewType] = Query(
        None, description="Response view: raw | structured | agent"
    ),
    # --- dependencies ---
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Get user's activities.

    When called WITHOUT date/view params, returns the legacy paginated response.
    When date params or view are provided, returns the specified view
    (defaults to structured).
    """
    try:
        has_new_params = (
            days_back is not None
            or start_date is not None
            or end_date is not None
            or view is not None
        )

        # Legacy path — exact backward compatibility
        if not has_new_params:
            return await _legacy_paginated_response(
                current_user.id, activity_type, page, size, db
            )

        # New view path
        resolved_view = view or ViewType.structured
        window_start, window_end = _resolve_window(days_back, start_date, end_date)

        activity_repo = ActivityRepository(db)
        view_service = ActivityViewService()

        if resolved_view == ViewType.raw:
            # Raw view uses exclusive end boundary:
            #   startDate inclusive at 00:00:00
            #   endDate exclusive at 00:00:00 (activities before this instant)
            raw_start_dt = datetime.combine(window_start, datetime.min.time())
            raw_end_dt = datetime.combine(window_end, datetime.min.time())
            raw_window = DateWindow(
                daysBack=days_back,
                startDate=window_start,
                endDate=window_end,
                endDateExclusive=True,
            )
            activities = await activity_repo.get_user_activities_in_range(
                user_id=current_user.id,
                start_date=raw_start_dt,
                end_date=raw_end_dt,
                activity_type=activity_type,
                end_exclusive=True,
            )
            return view_service.build_raw_response(
                activities, raw_window
            ).model_dump(mode="json")

        # Structured and agent views use inclusive end boundary
        window_start_dt = datetime.combine(window_start, datetime.min.time())
        window_end_dt = datetime.combine(window_end, datetime.max.time())
        window = DateWindow(
            daysBack=days_back,
            startDate=window_start,
            endDate=window_end,
        )

        if resolved_view == ViewType.structured:
            activities = await activity_repo.get_user_activities_in_range(
                user_id=current_user.id,
                start_date=window_start_dt,
                end_date=window_end_dt,
                activity_type=activity_type,
            )
            return view_service.build_structured_response(
                activities, window
            ).model_dump(mode="json")

        if resolved_view == ViewType.agent:
            # Always fetch 28 days from window end for ACR computation
            acr_start_dt = datetime.combine(
                window_end - timedelta(days=28), datetime.min.time()
            )
            all_28d = await activity_repo.get_user_activities_in_range(
                user_id=current_user.id,
                start_date=acr_start_dt,
                end_date=window_end_dt,
                activity_type=activity_type,
            )
            return view_service.build_agent_response(all_28d, window).model_dump(
                mode="json"
            )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve activities: {str(e)}",
        )


def _resolve_window(
    days_back: Optional[int],
    start_date: Optional[date],
    end_date: Optional[date],
) -> tuple[date, date]:
    """Resolve date window from query params.

    Explicit start_date/end_date take priority over days_back.
    """
    today = date.today()

    if start_date or end_date:
        s = start_date or (today - timedelta(days=30))
        e = end_date or today
        if s > e:
            raise ValueError("start_date must be on or before end_date")
        return s, e

    if days_back is not None:
        return today - timedelta(days=days_back), today

    return today - timedelta(days=30), today


async def _legacy_paginated_response(
    user_id: str,
    activity_type: Optional[str],
    page: int,
    size: int,
    db: AsyncSession,
) -> ActivityListResponse:
    """Original paginated behavior, preserved verbatim."""
    activity_repo = ActivityRepository(db)
    pagination = PaginationParams(page=page, size=size)

    activities = await activity_repo.get_user_activities(
        user_id=user_id,
        activity_type=activity_type,
        skip=pagination.skip,
        limit=pagination.limit,
    )

    total = await activity_repo.count_user_activities(
        user_id=user_id,
        activity_type=activity_type,
    )

    activity_summaries = []
    for activity in activities:
        summary = ActivitySummary.model_validate(activity)
        summary.normalized = NormalizedMetrics(
            duration_formatted=format_duration(activity.duration),
            distance_km=meters_to_km(activity.distance),
            distance_miles=meters_to_miles(activity.distance),
            average_speed_kmh=mps_to_kmh(activity.average_speed),
            average_speed_mph=mps_to_mph(activity.average_speed),
            max_speed_kmh=mps_to_kmh(activity.max_speed),
            max_speed_mph=mps_to_mph(activity.max_speed),
            average_pace_per_km=pace_per_km(activity.average_speed),
            average_pace_per_mile=pace_per_mile(activity.average_speed),
            elevation_gain_ft=meters_to_feet(activity.elevation_gain),
        )
        activity_summaries.append(summary)

    pagination_response = PaginationResponse.create(page, size, total)
    return ActivityListResponse(
        items=activity_summaries,
        total=pagination_response.total,
        page=pagination_response.page,
        pages=pagination_response.pages,
        has_next=pagination_response.has_next,
        has_prev=pagination_response.has_prev,
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