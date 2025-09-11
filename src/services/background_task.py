"""Background task management service."""

import asyncio
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.logging import get_logger
from ..db.base import get_db
from ..db.repositories import BackgroundTaskRepository

logger = get_logger(__name__)


class BackgroundTaskService:
    """Service for managing background tasks with progress tracking."""
    
    def __init__(self, session: Optional[AsyncSession] = None):
        self.session = session
        if session:
            self.task_repo = BackgroundTaskRepository(session)
    
    async def create_task(
        self,
        user_id: str,
        task_type: str,
        task_name: str,
        session: Optional[AsyncSession] = None
    ) -> str:
        """Create a new background task and return task_id."""
        if session:
            task_repo = BackgroundTaskRepository(session)
        else:
            task_repo = self.task_repo
            
        task_data = {
            "user_id": user_id,
            "task_type": task_type,
            "task_name": task_name,
            "status": "pending",
            "progress_percentage": 0.0,
            "progress_message": "Task created"
        }
        
        task = await task_repo.create(task_data)
        logger.info(f"Created background task {task.id} for user {user_id}: {task_name}")
        return task.id
    
    async def start_task(self, task_id: str, session: Optional[AsyncSession] = None) -> bool:
        """Mark task as started."""
        if session:
            task_repo = BackgroundTaskRepository(session)
        else:
            task_repo = self.task_repo
            
        update_data = {
            "status": "running",
            "started_at": datetime.utcnow(),
            "progress_message": "Task started"
        }
        
        task = await task_repo.update(task_id, update_data)
        if task:
            logger.info(f"Started background task {task_id}")
            return True
        return False
    
    async def update_progress(
        self,
        task_id: str,
        progress_percentage: float,
        progress_message: str,
        session: Optional[AsyncSession] = None
    ) -> bool:
        """Update task progress."""
        if session:
            task_repo = BackgroundTaskRepository(session)
        else:
            task_repo = self.task_repo
            
        update_data = {
            "progress_percentage": max(0.0, min(100.0, progress_percentage)),
            "progress_message": progress_message
        }
        
        task = await task_repo.update(task_id, update_data)
        if task:
            logger.debug(f"Updated task {task_id}: {progress_percentage}% - {progress_message}")
            return True
        return False
    
    async def complete_task(
        self,
        task_id: str,
        result_data: Optional[Dict[str, Any]] = None,
        session: Optional[AsyncSession] = None
    ) -> bool:
        """Mark task as completed."""
        if session:
            task_repo = BackgroundTaskRepository(session)
        else:
            task_repo = self.task_repo
            
        update_data = {
            "status": "completed",
            "completed_at": datetime.utcnow(),
            "progress_percentage": 100.0,
            "progress_message": "Task completed successfully"
        }
        
        if result_data:
            update_data["result_data"] = result_data
        
        task = await task_repo.update(task_id, update_data)
        if task:
            logger.info(f"Completed background task {task_id}")
            return True
        return False
    
    async def fail_task(
        self,
        task_id: str,
        error_message: str,
        session: Optional[AsyncSession] = None
    ) -> bool:
        """Mark task as failed."""
        if session:
            task_repo = BackgroundTaskRepository(session)
        else:
            task_repo = self.task_repo
            
        update_data = {
            "status": "failed",
            "completed_at": datetime.utcnow(),
            "error_message": error_message,
            "progress_message": f"Task failed: {error_message}"
        }
        
        task = await task_repo.update(task_id, update_data)
        if task:
            logger.error(f"Failed background task {task_id}: {error_message}")
            return True
        return False
    
    async def get_task_status(self, task_id: str, session: Optional[AsyncSession] = None) -> Optional[Dict[str, Any]]:
        """Get task status and progress."""
        if session:
            task_repo = BackgroundTaskRepository(session)
        else:
            task_repo = self.task_repo
            
        task = await task_repo.get_by_id(task_id)
        if not task:
            return None
        
        return {
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
    
    async def get_user_tasks(
        self,
        user_id: str,
        task_type: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
        session: Optional[AsyncSession] = None
    ) -> List[Dict[str, Any]]:
        """Get all tasks for a user."""
        if session:
            task_repo = BackgroundTaskRepository(session)
        else:
            task_repo = self.task_repo
            
        tasks = await task_repo.get_user_tasks(user_id, task_type, skip, limit)
        
        return [
            {
                "task_id": task.id,
                "task_type": task.task_type,
                "task_name": task.task_name,
                "status": task.status,
                "progress_percentage": task.progress_percentage or 0.0,
                "progress_message": task.progress_message,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            }
            for task in tasks
        ]
    
    async def run_background_task(
        self,
        task_id: str,
        task_func: Callable,
        *args,
        **kwargs
    ) -> None:
        """Run a task function in the background with progress tracking."""
        async def task_wrapper():
            # Get a fresh database session for the background task
            async for db_session in get_db():
                task_service = BackgroundTaskService(db_session)
                
                try:
                    # Start the task
                    await task_service.start_task(task_id, db_session)
                    
                    # Run the actual task function
                    result = await task_func(task_id, db_session, *args, **kwargs)
                    
                    # Complete the task
                    await task_service.complete_task(task_id, result, db_session)
                    
                except Exception as e:
                    # Handle task failure
                    await task_service.fail_task(task_id, str(e), db_session)
                    logger.error(f"Background task {task_id} failed: {str(e)}")
                
                break  # Exit the async generator
        
        # Run the task in the background
        asyncio.create_task(task_wrapper())
    
    async def cleanup_old_tasks(self, days_old: int = 30, session: Optional[AsyncSession] = None) -> int:
        """Clean up old completed/failed tasks."""
        if session:
            task_repo = BackgroundTaskRepository(session)
        else:
            task_repo = self.task_repo
            
        deleted_count = await task_repo.delete_old_tasks(days_old)
        logger.info(f"Cleaned up {deleted_count} old background tasks")
        return deleted_count