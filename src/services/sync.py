"""Synchronization service for Garmin Connect data."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from ..core.exceptions import GarminConnectError
from ..core.logging import get_logger
from ..db.repositories import ActivityRepository, SyncHistoryRepository, UserRepository
from ..schemas.activity import ActivityCreate
from .activity_ingestion import ActivityIngestionService
from .background_task import BackgroundTaskService
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

    async def start_background_sync(
        self,
        user_id: str,
        days: int,
        session: AsyncSession,
        force_resync: bool = False,
        force_reingest: bool = False,
        batch_size: int = 10
    ) -> str:
        """
        Start a background sync task that includes Pinecone ingestion.
        
        Returns task_id for tracking progress.
        """
        # Create background task
        task_service = BackgroundTaskService(session)
        task_id = await task_service.create_task(
            user_id=user_id,
            task_type="sync",
            task_name=f"Sync & ingest activities ({days} days)",
            session=session
        )
        
        # Start the background task
        await task_service.run_background_task(
            task_id,
            self._background_sync_activities,
            user_id=user_id,
            days=days,
            force_resync=force_resync,
            force_reingest=force_reingest,
            batch_size=batch_size
        )
        
        return task_id
    
    async def _background_sync_activities(
        self,
        task_id: str,
        session: AsyncSession,
        user_id: str,
        days: int,
        force_resync: bool = False,
        force_reingest: bool = False,
        batch_size: int = 10
    ) -> Dict:
        """
        Run sync activities and Pinecone ingestion in background with progress tracking.
        
        This method runs the actual sync followed by ingestion with progress updates.
        """
        task_service = BackgroundTaskService(session)
        
        try:
            # Initialize repositories with the background session
            user_repo = UserRepository(session)
            activity_repo = ActivityRepository(session)
            sync_history_repo = SyncHistoryRepository(session)
            
            # Update progress: Starting
            await task_service.update_progress(
                task_id, 5.0, "Initializing sync...", session
            )
            
            # Create sync history record
            sync_data = {
                "user_id": user_id,
                "sync_type": "activities",
                "status": "started",
                "started_at": datetime.utcnow(),
                "end_date": datetime.utcnow(),
                "start_date": datetime.utcnow() - timedelta(days=days),
            }
            sync_history = await sync_history_repo.create(sync_data)
            sync_id = sync_history.id
            
            # Update progress: Authenticating
            await task_service.update_progress(
                task_id, 10.0, "Authenticating with Garmin Connect...", session
            )
            
            # Get user's Garmin credentials
            user = await user_repo.get_by_id(user_id)
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
                await user_repo.update(user_id, {"garmin_session_data": session_data})

            # Update progress: Fetching activities
            await task_service.update_progress(
                task_id, 20.0, "Fetching activities from Garmin Connect...", session
            )

            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)

            logger.info(f"Starting background sync for user {user_id} from {start_date.date()} to {end_date.date()}")

            # Get activities from Garmin Connect (no limit, get all in date range)
            raw_activities = await self.garmin_service.get_activities(start_date, end_date)
            total_activities = len(raw_activities)
            
            # Debug: Check actual date range of returned activities
            if raw_activities:
                first_activity_date = raw_activities[0].get('startTimeLocal', 'Unknown')
                last_activity_date = raw_activities[-1].get('startTimeLocal', 'Unknown')
                logger.info(f"Retrieved {total_activities} activities from Garmin for {days}-day period")
                logger.info(f"Date range of returned activities: {last_activity_date} to {first_activity_date}")

            if total_activities == 0:
                await task_service.update_progress(
                    task_id, 90.0, "No activities found in date range", session
                )
            else:
                await task_service.update_progress(
                    task_id, 25.0, f"Processing {total_activities} activities from last {days} days...", session
                )

            # Process activities
            activities_synced = 0
            activities_already_exist = 0
            activities_failed = 0
            failed_activities = []

            for idx, raw_activity in enumerate(raw_activities):
                try:
                    # Update progress
                    progress = 25.0 + (idx / total_activities) * 60.0  # 25% to 85%
                    await task_service.update_progress(
                        task_id, 
                        progress, 
                        f"Processing activity {idx + 1} of {total_activities}...", 
                        session
                    )

                    # Parse activity data
                    parsed_activity = self.garmin_service.parse_activity_data(raw_activity)
                    parsed_activity["user_id"] = user_id

                    # Check if activity already exists
                    garmin_activity_id = parsed_activity.get("garmin_activity_id")
                    if not garmin_activity_id:
                        logger.warning("Activity missing garmin_activity_id, skipping")
                        activities_failed += 1
                        continue

                    existing_activity = await activity_repo.get_by_garmin_id(garmin_activity_id)

                    if existing_activity and not force_resync:
                        logger.debug(f"Activity {garmin_activity_id} already exists, skipping")
                        activities_already_exist += 1
                        continue

                    # Create or update activity
                    if existing_activity:
                        await activity_repo.update(existing_activity.id, parsed_activity)
                        logger.debug(f"Updated existing activity {garmin_activity_id}")
                    else:
                        await activity_repo.create(parsed_activity)
                        logger.debug(f"Created new activity {garmin_activity_id}")

                    activities_synced += 1

                    # Small delay to prevent overwhelming the system
                    await asyncio.sleep(0.01)  # Reduced from 0.1 to 0.01 seconds

                except Exception as e:
                    logger.error(f"Failed to sync activity {raw_activity.get('activityId', 'unknown')}: {str(e)}")
                    activities_failed += 1
                    failed_activities.append(str(e))

            # Update progress: Sync completed, starting ingestion
            await task_service.update_progress(
                task_id, 60.0, "Sync completed, starting Pinecone ingestion...", session
            )

            # Initialize ingestion service
            ingestion_service = ActivityIngestionService(session)
            
            # Run ingestion as part of the sync process
            ingestion_result = await self._run_ingestion_with_progress(
                ingestion_service=ingestion_service,
                task_service=task_service,
                task_id=task_id,
                user_id=user_id,
                session=session,
                batch_size=batch_size,
                force_reingest=force_reingest,
                progress_start=60.0,
                progress_end=95.0
            )

            # Update progress: Finalizing
            await task_service.update_progress(
                task_id, 95.0, "Finalizing sync and ingestion...", session
            )

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

            await sync_history_repo.update(sync_id, completion_data)

            # Update user's last sync time
            await user_repo.update(user_id, {"last_garmin_sync": datetime.utcnow()})

            logger.info(
                f"Background sync completed for user {user_id}: {activities_synced} synced, {activities_already_exist} already exist, {activities_failed} failed"
            )
            logger.info(f"Ingestion completed: {ingestion_result.get('ingested_count', 0)} activities ingested")

            # Return result data with clear explanations
            return {
                "sync_id": sync_id,
                "activities_in_date_range": total_activities,  # Activities actually within the requested date range
                "activities_synced": activities_synced,  # New activities added to database
                "activities_already_in_db": activities_already_exist,  # Activities that already existed
                "activities_failed": activities_failed,  # Activities that failed to sync
                "ingested_count": ingestion_result.get("ingested_count", 0),
                "vectorized_activities": ingestion_result.get("vectorized_activities", 0),
                "status": "success" if activities_failed == 0 else "partial"
            }

        except Exception as e:
            logger.error(f"Background sync failed for user {user_id}: {str(e)}")
            
            # Update sync history with error if we have sync_id
            if 'sync_id' in locals():
                error_data = {
                    "status": "failed",
                    "completed_at": datetime.utcnow(),
                    "error_message": str(e),
                }
                try:
                    await sync_history_repo.update(sync_id, error_data)
                except:
                    pass  # Don't fail task completion due to sync_history update failure
            
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
    
    async def _run_ingestion_with_progress(
        self,
        ingestion_service: ActivityIngestionService,
        task_service: BackgroundTaskService,
        task_id: str,
        user_id: str,
        session: AsyncSession,
        batch_size: int,
        force_reingest: bool,
        progress_start: float,
        progress_end: float
    ) -> Dict:
        """
        Run ingestion with progress updates within a given progress range.
        
        Maps ingestion progress from progress_start to progress_end.
        """
        try:
            # Create a custom progress callback
            async def progress_callback(current: int, total: int, message: str):
                if total > 0:
                    ingestion_progress = (current / total) * 100
                    # Map ingestion progress to our range
                    overall_progress = progress_start + (ingestion_progress / 100) * (progress_end - progress_start)
                    await task_service.update_progress(task_id, overall_progress, message, session)
            
            # Run ingestion with progress tracking
            result = await ingestion_service.ingest_user_activities(
                user_id=user_id,
                session=session,
                batch_size=batch_size,
                force_reingest=force_reingest,
                progress_callback=progress_callback
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Ingestion failed during sync: {str(e)}")
            # Return partial results
            return {
                "status": "failed",
                "ingested_count": 0,
                "vectorized_activities": 0,
                "error_message": str(e)
            }