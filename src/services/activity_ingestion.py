"""Activity ingestion service for processing Garmin activities into vector embeddings."""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.logging import get_logger
from ..db.repositories import ActivityRepository
from .background_task import BackgroundTaskService
from .embedding import EmbeddingService
from .temporal_processor import QueryContext, TemporalQueryProcessor
from .vector_db import VectorDBService

logger = get_logger(__name__)


class ActivityIngestionService:
    """Service for processing and ingesting activities into vector database."""
    
    def __init__(self, session: Optional[AsyncSession] = None):
        self.session = session
        self.embedding_service = EmbeddingService()
        self.vector_db = VectorDBService()
        self.temporal_processor = TemporalQueryProcessor()
        if session:
            self.activity_repo = ActivityRepository(session)
    
    async def ingest_user_activities(
        self,
        user_id: str,
        session: AsyncSession,
        batch_size: int = 10,
        force_reingest: bool = False,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Ingest all activities for a user into vector database."""
        start_time = datetime.now()
        activity_repo = ActivityRepository(session)
        
        try:
            logger.info(f"Starting activity ingestion for user {user_id}")
            
            # Get all activities for user
            activities = await activity_repo.get_user_activities(
                user_id=user_id,
                skip=0,
                limit=10000  # Large limit to get all activities
            )
            
            if not activities:
                logger.info(f"No activities found for user {user_id}")
                return {
                    "status": "completed",
                    "total_activities": 0,
                    "processed_activities": 0,
                    "failed_activities": 0,
                    "duration_seconds": 0,
                    "error_message": None
                }
            
            logger.info(f"Found {len(activities)} activities for user {user_id}")
            
            # Clear existing vectors if force reingest
            if force_reingest:
                await self.vector_db.delete_user_activities(user_id)
                logger.info(f"Cleared existing vectors for user {user_id}")
            
            # Process activities in batches
            processed_count = 0
            failed_count = 0
            failed_activities = []
            total_activities = len(activities)
            
            # Initial progress callback
            if progress_callback:
                await progress_callback(0, total_activities, "Starting activity ingestion...")
            
            for i in range(0, len(activities), batch_size):
                batch = activities[i:i + batch_size]
                batch_num = i//batch_size + 1
                total_batches = (len(activities) + batch_size - 1)//batch_size
                
                logger.info(f"Processing batch {batch_num}/{total_batches}")
                
                # Progress callback for batch start
                if progress_callback:
                    await progress_callback(
                        processed_count + failed_count, 
                        total_activities, 
                        f"Processing batch {batch_num}/{total_batches}..."
                    )
                
                # Process batch with concurrency
                batch_results = await asyncio.gather(
                    *[self._process_single_activity(user_id, activity) for activity in batch],
                    return_exceptions=True
                )
                
                # Count results
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        failed_count += 1
                        activity_id = batch[j].garmin_activity_id
                        failed_activities.append(f"Activity {activity_id}: {str(result)}")
                        logger.error(f"Failed to process activity {activity_id}: {str(result)}")
                    else:
                        processed_count += 1
                
                # Progress callback for batch completion
                if progress_callback:
                    await progress_callback(
                        processed_count + failed_count, 
                        total_activities, 
                        f"Completed batch {batch_num}/{total_batches}"
                    )
                
                # Rate limiting between batches
                await asyncio.sleep(0.5)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Final progress callback
            if progress_callback:
                await progress_callback(
                    processed_count + failed_count, 
                    total_activities, 
                    "Ingestion completed"
                )
            
            result = {
                "status": "completed" if failed_count == 0 else "partial",
                "total_activities": len(activities),
                "processed_activities": processed_count,
                "ingested_count": processed_count,
                "vectorized_activities": processed_count, 
                "failed_activities": failed_count,
                "duration_seconds": duration,
                "error_message": "; ".join(failed_activities[:5]) if failed_activities else None
            }
            
            logger.info(f"Completed ingestion for user {user_id}: {processed_count}/{len(activities)} activities processed")
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Activity ingestion failed for user {user_id}: {str(e)}")
            return {
                "status": "failed",
                "total_activities": 0,
                "processed_activities": 0,
                "failed_activities": 0,
                "duration_seconds": duration,
                "error_message": str(e)
            }
    
    async def start_background_ingestion(
        self,
        user_id: str,
        session: AsyncSession,
        batch_size: int = 10,
        force_reingest: bool = False
    ) -> str:
        """
        Start a background ingestion task.
        
        Returns task_id for tracking progress.
        """
        # Create background task
        task_service = BackgroundTaskService(session)
        task_id = await task_service.create_task(
            user_id=user_id,
            task_type="ingestion",
            task_name="Ingest activities to vector database",
            session=session
        )
        
        # Start the background task
        await task_service.run_background_task(
            task_id,
            self._background_ingest_activities,
            user_id=user_id,
            batch_size=batch_size,
            force_reingest=force_reingest
        )
        
        return task_id
    
    async def _background_ingest_activities(
        self,
        task_id: str,
        session: AsyncSession,
        user_id: str,
        batch_size: int = 10,
        force_reingest: bool = False
    ) -> Dict:
        """
        Run activity ingestion in background with progress tracking.
        """
        task_service = BackgroundTaskService(session)
        start_time = datetime.now()
        
        try:
            # Initialize repository with background session
            activity_repo = ActivityRepository(session)
            
            # Update progress: Starting
            await task_service.update_progress(
                task_id, 5.0, "Initializing ingestion...", session
            )
            
            logger.info(f"Starting background activity ingestion for user {user_id}")
            
            # Get all activities for user
            activities = await activity_repo.get_user_activities(
                user_id=user_id,
                skip=0,
                limit=10000  # Large limit to get all activities
            )
            
            if not activities:
                await task_service.update_progress(
                    task_id, 90.0, "No activities found", session
                )
                logger.info(f"No activities found for user {user_id}")
                return {
                    "status": "completed",
                    "total_activities": 0,
                    "processed_activities": 0,
                    "failed_activities": 0,
                    "duration_seconds": 0,
                    "error_message": None
                }
            
            total_activities = len(activities)
            logger.info(f"Found {total_activities} activities for user {user_id}")
            
            # Update progress: Clear existing vectors if force reingest
            if force_reingest:
                await task_service.update_progress(
                    task_id, 10.0, "Clearing existing vectors...", session
                )
                await self.vector_db.delete_user_activities(user_id)
                logger.info(f"Cleared existing vectors for user {user_id}")
            
            # Update progress: Processing activities
            await task_service.update_progress(
                task_id, 15.0, f"Processing {total_activities} activities in batches...", session
            )
            
            # Process activities in batches
            processed_count = 0
            failed_count = 0
            failed_activities = []
            
            total_batches = (total_activities + batch_size - 1) // batch_size
            
            for batch_idx in range(0, total_activities, batch_size):
                batch = activities[batch_idx:batch_idx + batch_size]
                batch_num = (batch_idx // batch_size) + 1
                
                # Update progress
                progress = 15.0 + (batch_idx / total_activities) * 70.0  # 15% to 85%
                await task_service.update_progress(
                    task_id,
                    progress,
                    f"Processing batch {batch_num}/{total_batches} ({len(batch)} activities)...",
                    session
                )
                
                logger.info(f"Processing batch {batch_num}/{total_batches}")
                
                # Process batch with concurrency
                batch_results = await asyncio.gather(
                    *[self._process_single_activity(user_id, activity) for activity in batch],
                    return_exceptions=True
                )
                
                # Count results
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        failed_count += 1
                        activity_id = batch[j].garmin_activity_id
                        failed_activities.append(f"Activity {activity_id}: {str(result)}")
                        logger.error(f"Failed to process activity {activity_id}: {str(result)}")
                    else:
                        processed_count += 1
                
                # Update progress within batch
                batch_progress = 15.0 + ((batch_idx + len(batch)) / total_activities) * 70.0
                await task_service.update_progress(
                    task_id,
                    batch_progress,
                    f"Completed batch {batch_num}/{total_batches} - {processed_count} processed, {failed_count} failed",
                    session
                )
                
                # Rate limiting between batches
                await asyncio.sleep(0.5)
            
            # Update progress: Finalizing
            await task_service.update_progress(
                task_id, 90.0, "Finalizing ingestion...", session
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            result = {
                "status": "completed" if failed_count == 0 else "partial",
                "total_activities": len(activities),
                "processed_activities": processed_count,
                "failed_activities": failed_count,
                "duration_seconds": duration,
                "error_message": "; ".join(failed_activities[:5]) if failed_activities else None
            }
            
            logger.info(f"Completed background ingestion for user {user_id}: {processed_count}/{len(activities)} activities processed")
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Background activity ingestion failed for user {user_id}: {str(e)}")
            return {
                "status": "failed",
                "total_activities": 0,
                "processed_activities": 0,
                "failed_activities": 0,
                "duration_seconds": duration,
                "error_message": str(e)
            }
    
    async def _process_single_activity(self, user_id: str, activity) -> bool:
        """Process a single activity into vector embeddings."""
        try:
            # Convert SQLAlchemy model to dict
            activity_data = self._activity_to_dict(activity)
            
            # Generate embeddings
            embedding_results = await self.embedding_service.process_activity_embeddings(activity_data)
            
            # Prepare embeddings and summaries for vector DB
            embeddings = {vtype: result.embedding for vtype, result in embedding_results.items()}
            summaries = {vtype: result.text for vtype, result in embedding_results.items()}
            
            # Store in vector database
            await self.vector_db.upsert_activity_multi_vector(
                user_id=user_id,
                activity_data=activity_data,
                embeddings=embeddings,
                summaries=summaries
            )
            
            logger.debug(f"Successfully processed activity {activity_data.get('garmin_activity_id')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process activity: {str(e)}")
            raise
    
    def _activity_to_dict(self, activity) -> Dict[str, Any]:
        """Convert SQLAlchemy activity model to dictionary."""
        return {
            "garmin_activity_id": activity.garmin_activity_id,
            "activity_name": activity.activity_name,
            "activity_type": activity.activity_type,
            "sport_type": activity.sport_type,
            "start_time": activity.start_time,
            "duration": activity.duration,
            "distance": activity.distance,
            "start_latitude": activity.start_latitude,
            "start_longitude": activity.start_longitude,
            "calories": activity.calories,
            "average_speed": activity.average_speed,
            "max_speed": activity.max_speed,
            "average_heart_rate": activity.average_heart_rate,
            "max_heart_rate": activity.max_heart_rate,
            "elevation_gain": activity.elevation_gain,
            "elevation_loss": activity.elevation_loss,
            "min_elevation": activity.min_elevation,
            "max_elevation": activity.max_elevation,
            "average_power": activity.average_power,
            "max_power": activity.max_power,
            "normalized_power": activity.normalized_power,
            "average_cadence": activity.average_cadence,
            "max_cadence": activity.max_cadence,
            "training_stress_score": activity.training_stress_score,
            "intensity_factor": activity.intensity_factor,
            "vo2_max": activity.vo2_max,
            "temperature": activity.temperature,
            "weather_condition": activity.weather_condition,
            "raw_data": activity.raw_data,
            "summary_data": activity.summary_data,
            "created_at": activity.created_at,
            "updated_at": activity.updated_at
        }
    
    async def search_user_activities(
        self,
        user_id: str,
        query: str,
        top_k: int = 15,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search user activities using semantic vector search."""
        try:
            # Process query to extract temporal and metric context
            query_context = self.temporal_processor.process_query(query)
            
            # Override top_k based on query type if not explicitly set
            search_limit = self.temporal_processor.get_search_limit(query_context)
            if top_k == 15:  # Default value
                top_k = search_limit
            
            # Generate query embeddings for all vector types
            summaries = {
                "main": query_context.enhanced_query,
                "metrics": f"Query about fitness metrics and performance data: {query}",
                "temporal": f"Temporal fitness query with date context: {query_context.enhanced_query}",
                "performance": f"Performance analysis query for training data: {query}"
            }
            
            texts = list(summaries.values())
            query_embeddings_list = await self.embedding_service.generate_embeddings_batch(texts)
            
            query_embeddings = {
                vector_type: embedding 
                for vector_type, embedding in zip(summaries.keys(), query_embeddings_list)
            }
            
            # Create metadata filter from query context
            metadata_filter = self.temporal_processor.create_pinecone_filter(query_context)
            
            # Merge with any additional filter metadata
            if filter_metadata:
                if metadata_filter:
                    metadata_filter.update(filter_metadata)
                else:
                    metadata_filter = filter_metadata
            
            # Perform hybrid search
            vector_results = await self.vector_db.search_activities_hybrid(
                user_id=user_id,
                query_embeddings=query_embeddings,
                metadata_filter=metadata_filter,
                top_k=top_k,
                boost_recent=True
            )
            
            if not vector_results:
                logger.info(f"No vector results found for user {user_id} query: {query}")
                return []
            
            # Convert to activity dictionaries with relevance scores
            activities = []
            for result in vector_results:
                activity_dict = dict(result.metadata)
                activity_dict["relevance_score"] = result.score
                activity_dict["summary_text"] = result.text
                activities.append(activity_dict)
            
            logger.info(f"Found {len(activities)} activities for query: {query[:50]}...")
            return activities
            
        except Exception as e:
            logger.error(f"Failed to search user activities: {str(e)}")
            return []
    
    async def get_user_activity_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about user's activities in vector database."""
        try:
            stats = await self.vector_db.get_index_stats(user_id)
            
            namespace_key = f"user_{user_id}"
            user_stats = stats.get("namespaces", {}).get(namespace_key, {})
            
            # Each activity has 4 vectors, so divide by 4 to get activity count
            vector_count = user_stats.get("vector_count", 0)
            estimated_activity_count = vector_count // 4
            
            return {
                "total_vectors": vector_count,
                "estimated_activities": estimated_activity_count,
                "namespace": namespace_key,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get user activity stats: {str(e)}")
            return {
                "error": str(e),
                "total_vectors": 0,
                "estimated_activities": 0
            }
    
    async def process_new_activity(self, user_id: str, activity_data: Dict[str, Any]) -> bool:
        """Process a single new activity (for real-time ingestion during sync)."""
        try:
            # Generate embeddings
            embedding_results = await self.embedding_service.process_activity_embeddings(activity_data)
            
            # Prepare embeddings and summaries
            embeddings = {vtype: result.embedding for vtype, result in embedding_results.items()}
            summaries = {vtype: result.text for vtype, result in embedding_results.items()}
            
            # Store in vector database
            await self.vector_db.upsert_activity_multi_vector(
                user_id=user_id,
                activity_data=activity_data,
                embeddings=embeddings,
                summaries=summaries
            )
            
            logger.info(f"Successfully processed new activity {activity_data.get('garmin_activity_id')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process new activity: {str(e)}")
            return False
    
    async def delete_user_activity_vectors(self, user_id: str, activity_id: str) -> bool:
        """Delete vectors for a specific activity."""
        try:
            await self.vector_db.delete_activity(user_id, activity_id)
            logger.info(f"Deleted vectors for activity {activity_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete activity vectors: {str(e)}")
            return False
    
    async def get_activity_embeddings_status(
        self,
        user_id: str,
        session: AsyncSession
    ) -> Dict[str, Any]:
        """Get status of embeddings vs database activities."""
        try:
            # Get activity count from database
            result = await session.execute(
                text("SELECT COUNT(*) as count FROM activities WHERE user_id = :user_id"),
                {"user_id": user_id}
            )
            db_count = result.scalar()
            
            # Try to get vector count using search query (workaround for serverless limitations)
            try:
                namespace = f"user_{user_id}"
                # Query for any vectors in the user namespace to count them
                dummy_vector = [0.0] * 1536  # Create dummy vector for counting
                search_results = self.vector_db.index.query(
                    vector=dummy_vector,
                    filter={"user_id": user_id, "vector_type": "main"},
                    top_k=10000,  # Large number to get all activities
                    include_metadata=True,
                    namespace=namespace
                )
                vector_activity_count = len(search_results.matches)
                logger.info(f"Found {vector_activity_count} vectorized activities for user {user_id}")
            except Exception as vector_error:
                logger.warning(f"Could not count vectors directly: {str(vector_error)}")
                # Fallback to sync_history for successful syncs
                sync_result = await session.execute(
                    text("""
                        SELECT activities_synced 
                        FROM sync_history 
                        WHERE user_id = :user_id 
                        AND sync_type = 'activities' 
                        AND status = 'success'
                        ORDER BY completed_at DESC 
                        LIMIT 1
                    """),
                    {"user_id": user_id}
                )
                sync_activities = sync_result.scalar()
                vector_activity_count = sync_activities if sync_activities is not None else 0
                logger.info(f"Using sync_history count for user {user_id}: {vector_activity_count} activities")
            
            return {
                "database_activities": db_count,
                "vectorized_activities": vector_activity_count,
                "sync_needed": db_count != vector_activity_count,
                "sync_percentage": (vector_activity_count / db_count * 100) if db_count > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get embeddings status: {str(e)}")
            return {
                "error": str(e),
                "database_activities": 0,
                "vectorized_activities": 0,
                "sync_needed": True,
                "sync_percentage": 0
            }