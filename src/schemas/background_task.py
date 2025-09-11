"""Background task schemas."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BackgroundTaskResponse(BaseModel):
    """Response schema for background task status."""
    
    task_id: str = Field(..., description="Unique task identifier")
    user_id: str = Field(..., description="User ID who owns the task")
    task_type: str = Field(..., description="Type of task (sync, ingestion, cleanup)")
    task_name: str = Field(..., description="Descriptive task name")
    status: str = Field(..., description="Task status (pending, running, completed, failed)")
    progress_percentage: float = Field(default=0.0, description="Task progress percentage (0-100)")
    progress_message: Optional[str] = Field(None, description="Current progress message")
    result_data: Optional[Dict[str, Any]] = Field(None, description="Task result data")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    created_at: str = Field(..., description="Task creation timestamp")
    started_at: Optional[str] = Field(None, description="Task start timestamp")
    completed_at: Optional[str] = Field(None, description="Task completion timestamp")
    duration_seconds: Optional[float] = Field(None, description="Task duration in seconds")


class BackgroundTaskSummary(BaseModel):
    """Summary schema for background task listing."""
    
    task_id: str = Field(..., description="Unique task identifier")
    task_type: str = Field(..., description="Type of task")
    task_name: str = Field(..., description="Descriptive task name")
    status: str = Field(..., description="Task status")
    progress_percentage: float = Field(default=0.0, description="Task progress percentage")
    progress_message: Optional[str] = Field(None, description="Current progress message")
    created_at: str = Field(..., description="Task creation timestamp")
    started_at: Optional[str] = Field(None, description="Task start timestamp")
    completed_at: Optional[str] = Field(None, description="Task completion timestamp")


class BackgroundTaskListResponse(BaseModel):
    """Response schema for listing background tasks."""
    
    tasks: List[BackgroundTaskSummary] = Field(..., description="List of background tasks")
    total_count: int = Field(..., description="Total number of tasks")
    page: int = Field(default=1, description="Current page number")
    page_size: int = Field(default=50, description="Number of items per page")


class TaskCreationResponse(BaseModel):
    """Response schema for task creation."""
    
    task_id: str = Field(..., description="Created task ID")
    message: str = Field(..., description="Success message")
    status_url: str = Field(..., description="URL to check task status")