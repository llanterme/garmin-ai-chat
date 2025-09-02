"""Service for cleaning up all user data."""

from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.logging import get_logger
from ..db.repositories import ActivityRepository, SyncHistoryRepository
from .vector_db import VectorDBService

logger = get_logger(__name__)


class UserCleanupService:
    """Service for comprehensive user data cleanup."""
    
    def __init__(self):
        self.vector_db = VectorDBService()
    
    async def delete_all_user_data(
        self,
        user_id: str,
        session: AsyncSession
    ) -> Dict[str, Any]:
        """
        Delete all user data from database and vector store.
        
        This includes:
        - All activities from the database
        - All sync history records
        - All vectors from Pinecone
        """
        logger.info(f"Starting complete data deletion for user {user_id}")
        
        results = {
            "user_id": user_id,
            "activities_deleted": 0,
            "sync_history_deleted": 0,
            "vectors_deleted": False,
            "errors": []
        }
        
        try:
            # Initialize repositories
            activity_repo = ActivityRepository(session)
            sync_repo = SyncHistoryRepository(session)
            
            # Delete activities from database
            try:
                activities_count = await activity_repo.delete_user_activities(user_id)
                results["activities_deleted"] = activities_count
                logger.info(f"Deleted {activities_count} activities for user {user_id}")
            except Exception as e:
                error_msg = f"Failed to delete activities: {str(e)}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
            
            # Delete sync history from database
            try:
                sync_count = await sync_repo.delete_user_sync_history(user_id)
                results["sync_history_deleted"] = sync_count
                logger.info(f"Deleted {sync_count} sync history records for user {user_id}")
            except Exception as e:
                error_msg = f"Failed to delete sync history: {str(e)}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
            
            # Delete vectors from Pinecone
            try:
                await self.vector_db.delete_user_activities(user_id)
                results["vectors_deleted"] = True
                logger.info(f"Deleted all vectors for user {user_id}")
            except Exception as e:
                error_msg = f"Failed to delete vectors: {str(e)}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
            
            # Determine overall status
            if not results["errors"]:
                results["status"] = "success"
                logger.info(f"Successfully deleted all data for user {user_id}")
            else:
                results["status"] = "partial"
                logger.warning(f"Partial deletion for user {user_id}: {len(results['errors'])} errors")
            
            return results
            
        except Exception as e:
            error_msg = f"Critical error during user data deletion: {str(e)}"
            logger.error(error_msg)
            results["status"] = "failed"
            results["errors"].append(error_msg)
            return results