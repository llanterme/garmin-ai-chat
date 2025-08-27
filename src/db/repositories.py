"""Repository classes for database operations."""

from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import and_, desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..core.exceptions import DatabaseError, NotFoundError
from .models import Activity, SyncHistory, User


class BaseRepository:
    """Base repository with common database operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def commit(self) -> None:
        """Commit the current transaction."""
        try:
            await self.session.commit()
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to commit transaction: {str(e)}") from e

    async def rollback(self) -> None:
        """Rollback the current transaction."""
        await self.session.rollback()


class UserRepository(BaseRepository):
    """Repository for user operations."""

    async def create(self, user_data: Dict[str, Any]) -> User:
        """Create a new user."""
        user = User(**user_data)
        self.session.add(user)
        await self.commit()
        await self.session.refresh(user)
        return user

    async def get_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        result = await self.session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        result = await self.session.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def update(self, user_id: str, update_data: Dict[str, Any]) -> User:
        """Update user data."""
        user = await self.get_by_id(user_id)
        if not user:
            raise NotFoundError(f"User with ID {user_id} not found")

        for key, value in update_data.items():
            if hasattr(user, key):
                setattr(user, key, value)

        await self.commit()
        await self.session.refresh(user)
        return user

    async def update_garmin_credentials(
        self, user_id: str, username: str, password: str, session_data: Optional[Dict[str, Any]] = None
    ) -> User:
        """Update user's Garmin credentials."""
        update_data = {
            "garmin_username": username,
            "garmin_password": password,
            "garmin_session_data": session_data,
        }
        return await self.update(user_id, update_data)

    async def delete(self, user_id: str) -> bool:
        """Delete a user."""
        user = await self.get_by_id(user_id)
        if not user:
            return False

        await self.session.delete(user)
        await self.commit()
        return True

    async def list_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        """List users with pagination."""
        result = await self.session.execute(
            select(User).offset(skip).limit(limit).order_by(User.created_at.desc())
        )
        return list(result.scalars().all())


class ActivityRepository(BaseRepository):
    """Repository for activity operations."""

    async def create(self, activity_data: Dict[str, Any]) -> Activity:
        """Create a new activity."""
        activity = Activity(**activity_data)
        self.session.add(activity)
        await self.commit()
        await self.session.refresh(activity)
        return activity

    async def get_by_id(self, activity_id: str) -> Optional[Activity]:
        """Get activity by ID."""
        result = await self.session.execute(select(Activity).where(Activity.id == activity_id))
        return result.scalar_one_or_none()

    async def get_by_garmin_id(self, garmin_activity_id: str) -> Optional[Activity]:
        """Get activity by Garmin activity ID."""
        result = await self.session.execute(
            select(Activity).where(Activity.garmin_activity_id == garmin_activity_id)
        )
        return result.scalar_one_or_none()

    async def get_user_activities(
        self,
        user_id: str,
        activity_type: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Activity]:
        """Get activities for a user with optional filtering."""
        query = select(Activity).where(Activity.user_id == user_id)

        if activity_type:
            query = query.where(Activity.activity_type == activity_type)

        query = query.offset(skip).limit(limit).order_by(desc(Activity.start_time))

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def bulk_create(self, activities_data: List[Dict[str, Any]]) -> List[Activity]:
        """Create multiple activities in bulk."""
        activities = [Activity(**data) for data in activities_data]
        self.session.add_all(activities)
        await self.commit()

        # Refresh all activities to get their IDs
        for activity in activities:
            await self.session.refresh(activity)

        return activities

    async def update(self, activity_id: str, update_data: Dict[str, Any]) -> Activity:
        """Update activity data."""
        activity = await self.get_by_id(activity_id)
        if not activity:
            raise NotFoundError(f"Activity with ID {activity_id} not found")

        for key, value in update_data.items():
            if hasattr(activity, key):
                setattr(activity, key, value)

        await self.commit()
        await self.session.refresh(activity)
        return activity

    async def delete(self, activity_id: str) -> bool:
        """Delete an activity."""
        activity = await self.get_by_id(activity_id)
        if not activity:
            return False

        await self.session.delete(activity)
        await self.commit()
        return True

    async def get_activity_types_for_user(self, user_id: str) -> List[str]:
        """Get unique activity types for a user."""
        result = await self.session.execute(
            select(Activity.activity_type)
            .where(Activity.user_id == user_id)
            .distinct()
            .order_by(Activity.activity_type)
        )
        return [row[0] for row in result.all()]

    async def count_user_activities(
        self, user_id: str, activity_type: Optional[str] = None
    ) -> int:
        """Count activities for a user."""
        query = select(Activity).where(Activity.user_id == user_id)

        if activity_type:
            query = query.where(Activity.activity_type == activity_type)

        result = await self.session.execute(query)
        return len(list(result.scalars().all()))


class SyncHistoryRepository(BaseRepository):
    """Repository for sync history operations."""

    async def create(self, sync_data: Dict[str, Any]) -> SyncHistory:
        """Create a new sync history record."""
        sync_history = SyncHistory(**sync_data)
        self.session.add(sync_history)
        await self.commit()
        await self.session.refresh(sync_history)
        return sync_history

    async def get_by_id(self, sync_id: str) -> Optional[SyncHistory]:
        """Get sync history by ID."""
        result = await self.session.execute(select(SyncHistory).where(SyncHistory.id == sync_id))
        return result.scalar_one_or_none()

    async def get_user_sync_history(
        self, user_id: str, sync_type: Optional[str] = None, skip: int = 0, limit: int = 50
    ) -> List[SyncHistory]:
        """Get sync history for a user."""
        query = select(SyncHistory).where(SyncHistory.user_id == user_id)

        if sync_type:
            query = query.where(SyncHistory.sync_type == sync_type)

        query = query.offset(skip).limit(limit).order_by(desc(SyncHistory.started_at))

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def update(self, sync_id: str, update_data: Dict[str, Any]) -> SyncHistory:
        """Update sync history data."""
        sync_history = await self.get_by_id(sync_id)
        if not sync_history:
            raise NotFoundError(f"Sync history with ID {sync_id} not found")

        for key, value in update_data.items():
            if hasattr(sync_history, key):
                setattr(sync_history, key, value)

        await self.commit()
        await self.session.refresh(sync_history)
        return sync_history

    async def get_last_successful_sync(
        self, user_id: str, sync_type: str
    ) -> Optional[SyncHistory]:
        """Get the last successful sync for a user and sync type."""
        result = await self.session.execute(
            select(SyncHistory)
            .where(
                and_(
                    SyncHistory.user_id == user_id,
                    SyncHistory.sync_type == sync_type,
                    SyncHistory.status == "success",
                )
            )
            .order_by(desc(SyncHistory.completed_at))
            .limit(1)
        )
        return result.scalar_one_or_none()