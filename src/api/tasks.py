"""Background task API endpoints."""

from typing import Optional
import time

from fastapi import APIRouter, Depends, HTTPException, Query, status, Response
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.base import get_db
from ..db.models import User
from ..schemas.background_task import (
    BackgroundTaskListResponse,
    BackgroundTaskResponse,
    BackgroundTaskSummary,
)
from ..services.background_task import BackgroundTaskService
from .dependencies import get_current_user

router = APIRouter(prefix="/tasks", tags=["Background Tasks"])


@router.get("/{task_id}", response_model=BackgroundTaskResponse)
async def get_task_status(
    task_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    response: Response = None
):
    """Get status of a specific background task."""
    try:
        # Direct database query for better performance
        from sqlalchemy import select
        from ..db.models import BackgroundTask
        
        # Start timing for performance monitoring
        start_time = time.time()
        
        result = await db.execute(
            select(BackgroundTask).where(BackgroundTask.id == task_id)
        )
        task = result.scalar_one_or_none()
        
        # Add performance header
        query_time = (time.time() - start_time) * 1000  # Convert to ms
        if response:
            response.headers["X-DB-Query-Time"] = f"{query_time:.2f}ms"
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found"
            )
        
        # Verify user ownership
        if task.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this task"
            )
        
        # Add cache headers for completed tasks
        if task.status == "completed":
            if response:
                response.headers["Cache-Control"] = "public, max-age=300"  # 5 minutes
        else:
            if response:
                response.headers["Cache-Control"] = "no-cache"
        
        # Convert to response format
        task_data = {
            "task_id": task.id,
            "user_id": task.user_id,
            "task_type": task.task_type,
            "task_name": task.task_name,
            "status": task.status,
            "progress_percentage": task.progress_percentage or 0.0,
            "progress_message": task.progress_message,
            "result_data": task.result_data,
            "error_message": task.error_message,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "duration_seconds": (
                (task.completed_at - task.started_at).total_seconds()
                if task.started_at and task.completed_at else None
            )
        }
        
        return BackgroundTaskResponse(**task_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task status: {str(e)}"
        )


@router.get("/", response_model=BackgroundTaskListResponse)
async def get_user_tasks(
    task_type: Optional[str] = Query(None, description="Filter by task type"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get all background tasks for the current user."""
    try:
        task_service = BackgroundTaskService(db)
        
        skip = (page - 1) * page_size
        tasks = await task_service.get_user_tasks(
            user_id=current_user.id,
            task_type=task_type,
            skip=skip,
            limit=page_size,
            session=db
        )
        
        # Convert to summary format
        task_summaries = [BackgroundTaskSummary(**task) for task in tasks]
        
        return BackgroundTaskListResponse(
            tasks=task_summaries,
            total_count=len(task_summaries),  # This is simplified - in production you'd want a proper count
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user tasks: {str(e)}"
        )