"""Synchronization service for Garmin Connect data."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from ..core.exceptions import GarminConnectError
from ..core.logging import get_logger
from ..db.repositories import ActivityRepository, SyncHistoryRepository, UserRepository
from ..schemas.activity import ActivityCreate
from .garmin import GarminService

logger = get_logger(__name__)


class SyncService:
    """Service for synchronizing Garmin Connect data."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.user_repo = UserRepository(session)
        self.activity_repo = ActivityRepository(session)
        self.sync_history_repo = SyncHistoryRepository(session)
        self.garmin_service = GarminService()

    async def sync_user_activities(
        self,
        user_id: str,
        days: int,
        force_resync: bool = False,
    ) -> str:
        """
        Sync user's activities from Garmin Connect.
        
        Returns sync_id for tracking progress.
        """
        # Create sync history record
        sync_data = {
            "user_id": user_id,
            "sync_type": "activities",
            "status": "started",
            "started_at": datetime.utcnow(),
            "end_date": datetime.utcnow(),
            "start_date": datetime.utcnow() - timedelta(days=days),
        }
        sync_history = await self.sync_history_repo.create(sync_data)
        sync_id = sync_history.id

        try:
            # Get user's Garmin credentials
            user = await self.user_repo.get_by_id(user_id)
            if not user or not user.garmin_username or not user.garmin_password:
                raise GarminConnectError("Garmin credentials not found")

            # Authenticate with Garmin Connect
            success, session_data = await self.garmin_service.authenticate(
                user.garmin_username,
                user.garmin_password,
                user.garmin_session_data,
            )

            if not success:
                raise GarminConnectError("Failed to authenticate with Garmin Connect")

            # Update session data if changed
            if session_data != user.garmin_session_data:
                await self.user_repo.update(user_id, {"garmin_session_data": session_data})

            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)

            logger.info(f"Starting activity sync for user {user_id} from {start_date.date()} to {end_date.date()}")

            # Get activities from Garmin Connect
            raw_activities = await self.garmin_service.get_activities(start_date, end_date)

            # Process activities
            activities_synced = 0
            activities_failed = 0
            failed_activities = []

            for raw_activity in raw_activities:
                try:
                    # Parse activity data
                    parsed_activity = self.garmin_service.parse_activity_data(raw_activity)
                    parsed_activity["user_id"] = user_id

                    # Check if activity already exists
                    garmin_activity_id = parsed_activity.get("garmin_activity_id")
                    if not garmin_activity_id:
                        logger.warning("Activity missing garmin_activity_id, skipping")
                        activities_failed += 1
                        continue

                    existing_activity = await self.activity_repo.get_by_garmin_id(garmin_activity_id)

                    if existing_activity and not force_resync:
                        logger.debug(f"Activity {garmin_activity_id} already exists, skipping")
                        continue

                    # Create or update activity
                    if existing_activity and force_resync:
                        # Update existing activity
                        update_data = {k: v for k, v in parsed_activity.items() 
                                     if k not in ["id", "user_id", "garmin_activity_id", "created_at"]}
                        await self.activity_repo.update(existing_activity.id, update_data)
                        logger.debug(f"Updated existing activity {garmin_activity_id}")
                    else:
                        # Create new activity
                        await self.activity_repo.create(parsed_activity)
                        logger.debug(f"Created new activity {garmin_activity_id}")

                    activities_synced += 1

                    # Rate limiting
                    await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(f"Failed to sync activity {raw_activity.get('activityId', 'unknown')}: {str(e)}")
                    activities_failed += 1
                    failed_activities.append(str(e))

            # Update sync history
            completion_data = {
                "status": "success" if activities_failed == 0 else "partial",
                "completed_at": datetime.utcnow(),
                "duration_seconds": (datetime.utcnow() - sync_history.started_at).total_seconds(),
                "activities_synced": activities_synced,
                "activities_failed": activities_failed,
            }

            if failed_activities:
                completion_data["error_message"] = f"Failed activities: {'; '.join(failed_activities[:5])}"

            await self.sync_history_repo.update(sync_id, completion_data)

            # Update user's last sync time
            await self.user_repo.update(user_id, {"last_garmin_sync": datetime.utcnow()})

            logger.info(
                f"Sync completed for user {user_id}: {activities_synced} synced, {activities_failed} failed"
            )

            return sync_id

        except Exception as e:
            logger.error(f"Sync failed for user {user_id}: {str(e)}")
            
            # Update sync history with error
            error_data = {
                "status": "failed",
                "completed_at": datetime.utcnow(),
                "duration_seconds": (datetime.utcnow() - sync_history.started_at).total_seconds(),
                "error_message": str(e),
            }
            await self.sync_history_repo.update(sync_id, error_data)
            
            raise

    async def get_sync_status(self, sync_id: str) -> Optional[Dict]:
        """Get sync status by sync ID."""
        try:
            sync_history = await self.sync_history_repo.get_by_id(sync_id)
            if not sync_history:
                return None

            # Calculate progress percentage
            progress_percentage = None
            if sync_history.status in ["success", "failed", "partial"]:
                progress_percentage = 100.0
            elif sync_history.status == "started":
                # Estimate progress based on time elapsed (rough estimate)
                elapsed = (datetime.utcnow() - sync_history.started_at).total_seconds()
                # Assume sync takes roughly 1 minute per week of data
                estimated_total = max(60, (sync_history.end_date - sync_history.start_date).days * 8.6)
                progress_percentage = min(95.0, (elapsed / estimated_total) * 100)

            return {
                "sync_id": sync_history.id,
                "status": sync_history.status,
                "started_at": sync_history.started_at,
                "completed_at": sync_history.completed_at,
                "duration_seconds": sync_history.duration_seconds,
                "activities_synced": sync_history.activities_synced,
                "activities_failed": sync_history.activities_failed,
                "error_message": sync_history.error_message,
                "progress_percentage": progress_percentage,
            }

        except Exception as e:
            logger.error(f"Failed to get sync status: {str(e)}")
            return None

    async def test_garmin_connection(self, user_id: str) -> Tuple[bool, str, Optional[Dict]]:
        """Test Garmin Connect connection for a user."""
        try:
            # Get user's Garmin credentials
            user = await self.user_repo.get_by_id(user_id)
            if not user or not user.garmin_username or not user.garmin_password:
                return False, "Garmin credentials not configured", None

            # Test authentication
            success, session_data = await self.garmin_service.authenticate(
                user.garmin_username,
                user.garmin_password,
                user.garmin_session_data,
            )

            if not success:
                return False, "Failed to authenticate with Garmin Connect", None

            # Get user profile to test connection
            profile = await self.garmin_service.get_user_profile()

            # Update session data if changed
            if session_data != user.garmin_session_data:
                await self.user_repo.update(user_id, {"garmin_session_data": session_data})

            return True, "Connection successful", profile

        except Exception as e:
            logger.error(f"Garmin connection test failed: {str(e)}")
            return False, f"Connection test failed: {str(e)}", None

    async def get_user_sync_history(
        self, user_id: str, sync_type: Optional[str] = None, skip: int = 0, limit: int = 50
    ) -> List[Dict]:
        """Get sync history for a user."""
        try:
            history = await self.sync_history_repo.get_user_sync_history(
                user_id, sync_type, skip, limit
            )

            return [
                {
                    "id": record.id,
                    "sync_type": record.sync_type,
                    "status": record.status,
                    "start_date": record.start_date,
                    "end_date": record.end_date,
                    "started_at": record.started_at,
                    "completed_at": record.completed_at,
                    "duration_seconds": record.duration_seconds,
                    "activities_synced": record.activities_synced,
                    "activities_failed": record.activities_failed,
                    "error_message": record.error_message,
                }
                for record in history
            ]

        except Exception as e:
            logger.error(f"Failed to get sync history: {str(e)}")
            return []

    async def get_sync_stats(self, user_id: str) -> Dict:
        """Get synchronization statistics for a user."""
        try:
            # Get all sync history for user
            all_syncs = await self.sync_history_repo.get_user_sync_history(
                user_id, sync_type="activities", skip=0, limit=1000
            )

            total_syncs = len(all_syncs)
            successful_syncs = len([s for s in all_syncs if s.status == "success"])
            failed_syncs = len([s for s in all_syncs if s.status == "failed"])
            partial_syncs = len([s for s in all_syncs if s.status == "partial"])

            last_sync = None
            if all_syncs:
                last_sync = max(all_syncs, key=lambda s: s.started_at).completed_at

            total_activities_synced = sum(s.activities_synced for s in all_syncs)

            # Calculate average sync duration
            completed_syncs = [s for s in all_syncs if s.duration_seconds is not None]
            average_sync_duration = None
            if completed_syncs:
                average_sync_duration = sum(s.duration_seconds for s in completed_syncs) / len(completed_syncs)

            return {
                "total_syncs": total_syncs,
                "successful_syncs": successful_syncs,
                "failed_syncs": failed_syncs,
                "partial_syncs": partial_syncs,
                "last_sync": last_sync,
                "total_activities_synced": total_activities_synced,
                "average_sync_duration": average_sync_duration,
            }

        except Exception as e:
            logger.error(f"Failed to get sync stats: {str(e)}")
            return {
                "total_syncs": 0,
                "successful_syncs": 0,
                "failed_syncs": 0,
                "partial_syncs": 0,
                "last_sync": None,
                "total_activities_synced": 0,
                "average_sync_duration": None,
            }