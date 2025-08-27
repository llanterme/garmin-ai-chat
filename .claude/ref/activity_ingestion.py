"""
Activity ingestion service that orchestrates Strava data fetching, 
embedding generation, and vector database storage.
"""

import asyncio
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from pydantic import BaseModel

from .strava_client import StravaClient
from .embeddings import EmbeddingService
from .vector_db import VectorDBService
from .metrics_service import MetricsService, TSSCalculation, FitnessMetrics
from .threshold_estimator import ThresholdEstimator
from .mysql_service import mysql_service


class IngestionStatus(BaseModel):
    """Status of an ingestion process."""
    user_id: str
    status: str  # "in_progress", "completed", "failed"
    total_activities: int
    processed_activities: int
    failed_activities: int
    error_message: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None


class ActivityIngestionService:
    """
    Service for ingesting Strava activities into the vector database.
    """
    
    def __init__(self):
        self.strava_client = StravaClient()
        self.embedding_service = EmbeddingService()
        self.vector_db = VectorDBService()
        self.metrics_service = MetricsService()
        self.threshold_estimator = ThresholdEstimator()
        self.logger = logging.getLogger(__name__)
        
        # Track ingestion status per user
        self.ingestion_status: Dict[str, IngestionStatus] = {}
        
        # Track TSS calculations per user (in-memory cache)
        self.user_tss_calculations: Dict[str, List[TSSCalculation]] = {}
    
    async def _save_tss_data(self, user_id: str) -> None:
        """Save TSS calculations for a user to MySQL database."""
        try:
            if user_id not in self.user_tss_calculations:
                return
                
            tss_calculations = self.user_tss_calculations[user_id]
            
            # Save to MySQL database
            saved_count = await mysql_service.save_tss_calculations(user_id, tss_calculations)
            self.logger.info(f"Saved {saved_count} TSS calculations to MySQL for user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to save TSS data to MySQL for user {user_id}: {str(e)}")
    
    async def _load_tss_data(self, user_id: str) -> None:
        """Load TSS data for a user from MySQL database."""
        try:
            # Load from MySQL database
            tss_data = await mysql_service.get_user_tss_data(user_id)
            
            # Convert to TSSCalculation objects
            tss_calculations = []
            for calc_data in tss_data:
                try:
                    # Convert MySQL data back to TSSCalculation format
                    tss_calc = TSSCalculation(
                        activity_id=calc_data['activity_id'],
                        tss_value=calc_data['tss_value'],
                        calculation_method=calc_data['calculation_method'],
                        sport_type=calc_data['sport_type'],
                        intensity_factor=calc_data['intensity_factor'],
                        duration_hours=calc_data['duration_hours'],
                        activity_date=calc_data['activity_date'],
                        notes=calc_data['notes'],
                        original_activity_type=calc_data['original_activity_type']
                    )
                    tss_calculations.append(tss_calc)
                except Exception as e:
                    self.logger.warning(f"Failed to parse TSS calculation: {str(e)}")
                    continue
            
            self.user_tss_calculations[user_id] = tss_calculations
            self.logger.info(f"Loaded {len(tss_calculations)} TSS calculations from MySQL for user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to load TSS data from MySQL for user {user_id}: {str(e)}")
    
    async def ingest_user_activities(
        self,
        user_id: str,
        access_token: str,
        refresh_token: str,
        full_sync: bool = False,
        sync_days: Optional[int] = None
    ) -> IngestionStatus:
        """
        Ingest all activities for a user into the vector database.
        
        Args:
            user_id: Strava athlete ID
            access_token: User's access token
            refresh_token: User's refresh token
            full_sync: Whether to delete existing data and resync all activities
            sync_days: Number of days to sync (defaults to STRAVA_SYNC_DAYS env var)
            
        Returns:
            Ingestion status
        """
        # Initialize status
        status = IngestionStatus(
            user_id=user_id,
            status="in_progress",
            total_activities=0,
            processed_activities=0,
            failed_activities=0,
            started_at=datetime.now()
        )
        
        self.ingestion_status[user_id] = status
        
        try:
            # If full sync, delete existing activities
            if full_sync:
                self.logger.info(f"Performing full sync for user {user_id} - deleting existing activities")
                try:
                    self.vector_db.delete_user_activities(user_id)
                    self.logger.info(f"Successfully deleted existing activities for user {user_id}")
                except Exception as e:
                    self.logger.error(f"Error deleting existing activities for user {user_id}: {str(e)}")
                    # Continue anyway - the delete might fail if there are no activities
            
            # Fetch activities from Strava
            # Use provided sync_days or fall back to environment variable
            if sync_days is None:
                sync_days = int(os.getenv('STRAVA_SYNC_DAYS', '365'))
                sync_source = "STRAVA_SYNC_DAYS env var"
            else:
                sync_source = "API parameter"
            
            self.logger.info(f"Starting to fetch activities for user {user_id} (last {sync_days} days from {sync_source})")
            try:
                activities = await self.strava_client.get_activities(access_token, sync_days=sync_days)
                self.logger.info(f"Successfully fetched {len(activities)} activities for user {user_id} from the last {sync_days} days")
            except Exception as e:
                self.logger.error(f"Error fetching activities from Strava for user {user_id}: {str(e)}")
                self.logger.error(f"Error type: {type(e).__name__}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                raise
            
            await self._update_ingestion_status(user_id, total_activities=len(activities))
            self.logger.info(f"Retrieved {len(activities)} activities for user {user_id}")
            
            if not activities:
                self.logger.info(f"No activities found for user {user_id}")
                await self._update_ingestion_status(user_id, status="completed", completed_at=datetime.now())
                return status
            
            # Estimate thresholds for TSS calculations
            self.logger.info(f"Estimating thresholds for user {user_id}")
            try:
                thresholds = self.threshold_estimator.estimate_all_thresholds(activities)
                validated_thresholds = self.threshold_estimator.validate_thresholds(thresholds)
                self.logger.info(f"Estimated thresholds for user {user_id}: FTP={validated_thresholds.ftp}W, "
                               f"Running={validated_thresholds.running_threshold_pace} min/km, "
                               f"Swimming={validated_thresholds.swimming_threshold_pace} min/100m, "
                               f"LTHR={validated_thresholds.lthr} bpm")
            except Exception as e:
                self.logger.error(f"Error estimating thresholds for user {user_id}: {str(e)}")
                # Use default thresholds
                from .metrics_service import ThresholdEstimates
                validated_thresholds = ThresholdEstimates(ftp=200, running_threshold_pace=4.0, 
                                                        swimming_threshold_pace=1.5, lthr=165)
            
            # Calculate TSS and advanced metrics for all activities
            self.logger.info(f"DEBUG: About to calculate TSS for {len(activities)} activities")
            self.logger.info(f"Calculating TSS and advanced metrics for {len(activities)} activities")
            tss_calculations = []
            for activity in activities:
                try:
                    tss_calc = self.metrics_service.calculate_multi_sport_tss(activity, validated_thresholds)
                    # Add date information to TSS calculation
                    start_date = activity.get('start_date_local', '').split('T')[0]
                    tss_calc.activity_date = start_date
                    tss_calculations.append(tss_calc)
                    
                    # Calculate and add advanced metrics
                    np = self.metrics_service.calculate_normalized_power(activity)
                    if np:
                        activity['normalized_power'] = np
                        if validated_thresholds.ftp:
                            activity['intensity_factor'] = self.metrics_service.calculate_intensity_factor(np, validated_thresholds.ftp)
                        activity['variability_index'] = self.metrics_service.calculate_variability_index(activity)
                    
                    activity['efficiency_factor'] = self.metrics_service.calculate_efficiency_factor(activity)
                    activity['grade_adjusted_pace'] = self.metrics_service.calculate_grade_adjusted_pace(activity)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to calculate metrics for activity {activity.get('id', 'unknown')}: {str(e)}")
            
            # Store TSS calculations for this user
            self.user_tss_calculations[user_id] = tss_calculations
            self.logger.info(f"Calculated TSS for {len(tss_calculations)} activities")
            
            # Save TSS data to MySQL database
            await self._save_tss_data(user_id)
            
            # Update cached thresholds with new activity data for future use
            self.logger.info(f"Updating cached thresholds with new activity data for user {user_id}")
            try:
                await self._update_cached_thresholds_after_ingestion(user_id, validated_thresholds)
                self.logger.info(f"Successfully updated cached thresholds for user {user_id}")
            except Exception as e:
                self.logger.warning(f"Failed to update cached thresholds for user {user_id}: {str(e)}")
            
            # Process activities in batches with async processing
            batch_size = 10
            concurrent_activities = 5  # Process 5 activities concurrently within each batch
            self.logger.info(f"Processing {len(activities)} activities in batches of {batch_size} with {concurrent_activities} concurrent activities")
            
            for i in range(0, len(activities), batch_size):
                batch = activities[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(activities) + batch_size - 1) // batch_size
                
                self.logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch)} activities")
                
                try:
                    # Process activities in smaller concurrent groups within the batch
                    for j in range(0, len(batch), concurrent_activities):
                        concurrent_batch = batch[j:j + concurrent_activities]
                        
                        # Process concurrent activities in parallel
                        tasks = []
                        for activity in concurrent_batch:
                            task = self._process_single_activity(user_id, activity)
                            tasks.append(task)
                        
                        # Wait for all concurrent activities to complete
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        # Handle results and errors
                        for k, result in enumerate(results):
                            if isinstance(result, Exception):
                                self.logger.error(f"Failed to process activity {concurrent_batch[k].get('id', 'unknown')}: {result}")
                                await self._update_ingestion_status(user_id, failed_activities=status.failed_activities + 1)
                            else:
                                await self._update_ingestion_status(user_id, processed_activities=status.processed_activities + 1)
                    
                    # Log progress
                    progress = (status.processed_activities / status.total_activities) * 100
                    self.logger.info(f"User {user_id}: {status.processed_activities}/{status.total_activities} activities processed ({progress:.1f}%)")
                    
                except Exception as e:
                    self.logger.error(f"Failed to process batch {batch_num}/{total_batches} for user {user_id}: {str(e)}")
                    self.logger.error(f"Batch contained activities: {[a.get('id', 'unknown') for a in batch]}")
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    await self._update_ingestion_status(user_id, failed_activities=status.failed_activities + len(batch))
            
            # Mark as completed
            await self._update_ingestion_status(user_id, status="completed", completed_at=datetime.now())
            
            self.logger.info(f"Ingestion completed for user {user_id}: {status.processed_activities} processed, {status.failed_activities} failed")
            
        except Exception as e:
            self.logger.error(f"Ingestion failed for user {user_id}: {str(e)}")
            await self._update_ingestion_status(user_id, status="failed", error_message=str(e), completed_at=datetime.now())
        
        return status
    
    async def _update_ingestion_status(self, user_id: str, **updates) -> None:
        """
        Update ingestion status in memory.
        
        Args:
            user_id: User identifier
            **updates: Status fields to update
        """
        if user_id in self.ingestion_status:
            status = self.ingestion_status[user_id]
            
            # Update in-memory status
            for key, value in updates.items():
                if hasattr(status, key):
                    setattr(status, key, value)
    
    async def _process_single_activity(self, user_id: str, activity: Dict[str, Any]) -> None:
        """
        Process a single activity: generate summaries, embeddings, and store.
        
        Args:
            user_id: Strava athlete ID
            activity: Activity data from Strava API
        """
        activity_id = activity.get("id", "unknown")
        activity_name = activity.get("name", "Unnamed Activity")
        activity_type = activity.get("type", "Unknown")
        
        self.logger.info(f"Processing activity: {activity_id} ({activity_type}: {activity_name})")
        
        try:
            activity_id_str = str(activity_id)
            
            # Generate multiple summaries for different aspects
            self.logger.info(f"Generating multi-vector embeddings for activity {activity_id}")
            summaries_dict = self.embedding_service.create_multi_vector_embeddings(activity)
            self.logger.info(f"Generated {len(summaries_dict)} different summaries for activity {activity_id}")
            
            # Extract rich metadata
            self.logger.info(f"Extracting metadata for activity {activity_id}")
            metadata = self.embedding_service.extract_activity_metadata(activity)
            self.logger.info(f"Extracted {len(metadata)} metadata fields for activity {activity_id}")
            
            # Generate embeddings for all summary types in batch
            self.logger.info(f"Generating embeddings for activity {activity_id}")
            summary_texts = list(summaries_dict.values())
            vector_types = list(summaries_dict.keys())
            
            # Batch generate embeddings (4 summaries in one API call)
            embeddings_list = self.embedding_service.generate_embeddings_batch(summary_texts)
            
            # Map embeddings back to vector types
            embeddings_dict = {}
            for i, vector_type in enumerate(vector_types):
                embeddings_dict[vector_type] = embeddings_list[i]
            
            self.logger.info(f"Generated {len(embeddings_dict)} embeddings in single batch for activity {activity_id}")
            
            # Store in vector database with multi-vector approach
            self.logger.info(f"Storing activity {activity_id} in vector database")
            self.vector_db.upsert_activity_multi_vector(
                user_id=user_id,
                activity_id=activity_id_str,
                embeddings=embeddings_dict,
                summaries=summaries_dict,
                activity_metadata=metadata
            )
            self.logger.info(f"Successfully stored activity {activity_id} with {len(embeddings_dict)} vector types")
            
        except Exception as e:
            self.logger.warning(f"Failed to process activity {activity_id}: {str(e)}")
            self.logger.warning(f"Error type: {type(e).__name__}")
            import traceback
            self.logger.warning(f"Traceback: {traceback.format_exc()}")
            
            # Fallback to basic storage for failed activities
            self.logger.info(f"Attempting fallback processing for activity {activity_id}")
            try:
                summary = self.embedding_service.create_activity_summary(activity)
                embedding = self.embedding_service.generate_embedding(summary)
                basic_metadata = {
                    "activity_type": activity.get("type", "unknown").lower(),
                    "name": activity.get("name", ""),
                    "start_date": activity.get("start_date_local", ""),
                }
                
                self.vector_db.upsert_activity(
                    user_id=user_id,
                    activity_id=str(activity_id),
                    activity_summary=summary,
                    embedding=embedding,
                    activity_metadata=basic_metadata
                )
                self.logger.info(f"Successfully stored activity {activity_id} using fallback method")
            except Exception as fallback_error:
                self.logger.error(f"Even fallback processing failed for activity {activity_id}: {str(fallback_error)}")
                self.logger.error(f"Fallback error type: {type(fallback_error).__name__}")
                import traceback
                self.logger.error(f"Fallback traceback: {traceback.format_exc()}")
                raise  # Re-raise to be handled by the calling code
    
    async def _process_activity_batch(self, user_id: str, activities: List[Dict[str, Any]]) -> None:
        """
        Process a batch of activities: generate summaries, embeddings, and store.
        
        Args:
            user_id: Strava athlete ID
            activities: List of activity data from Strava API
        """
        # Process each activity with multi-vector approach
        self.logger.info(f"Starting to process {len(activities)} activities for user {user_id}")
        
        for i, activity in enumerate(activities):
            activity_id = activity.get("id", "unknown")
            activity_name = activity.get("name", "Unnamed Activity")
            activity_type = activity.get("type", "Unknown")
            
            self.logger.info(f"Processing activity {i+1}/{len(activities)}: {activity_id} ({activity_type}: {activity_name})")
            
            try:
                activity_id_str = str(activity_id)
                
                # Generate multiple summaries for different aspects
                self.logger.info(f"Generating multi-vector embeddings for activity {activity_id}")
                summaries_dict = self.embedding_service.create_multi_vector_embeddings(activity)
                self.logger.info(f"Generated {len(summaries_dict)} different summaries for activity {activity_id}")
                
                # Extract rich metadata
                self.logger.info(f"Extracting metadata for activity {activity_id}")
                metadata = self.embedding_service.extract_activity_metadata(activity)
                self.logger.info(f"Extracted {len(metadata)} metadata fields for activity {activity_id}")
                
                # Generate embeddings for all summary types in batch
                self.logger.info(f"Generating embeddings for activity {activity_id}")
                summary_texts = list(summaries_dict.values())
                vector_types = list(summaries_dict.keys())
                
                # Batch generate embeddings (4 summaries in one API call)
                embeddings_list = self.embedding_service.generate_embeddings_batch(summary_texts)
                
                # Map embeddings back to vector types
                embeddings_dict = {}
                for i, vector_type in enumerate(vector_types):
                    embeddings_dict[vector_type] = embeddings_list[i]
                
                self.logger.info(f"Generated {len(embeddings_dict)} embeddings in single batch for activity {activity_id}")
                
                # Store in vector database with multi-vector approach
                self.logger.info(f"Storing activity {activity_id} in vector database")
                self.vector_db.upsert_activity_multi_vector(
                    user_id=user_id,
                    activity_id=activity_id_str,
                    embeddings=embeddings_dict,
                    summaries=summaries_dict,
                    activity_metadata=metadata
                )
                self.logger.info(f"Successfully stored activity {activity_id} with {len(embeddings_dict)} vector types")
                
            except Exception as e:
                self.logger.warning(f"Failed to process activity {activity_id}: {str(e)}")
                self.logger.warning(f"Error type: {type(e).__name__}")
                import traceback
                self.logger.warning(f"Traceback: {traceback.format_exc()}")
                
                # Fallback to basic storage for failed activities
                self.logger.info(f"Attempting fallback processing for activity {activity_id}")
                try:
                    summary = self.embedding_service.create_activity_summary(activity)
                    embedding = self.embedding_service.generate_embedding(summary)
                    basic_metadata = {
                        "activity_type": activity.get("type", "unknown").lower(),
                        "name": activity.get("name", ""),
                        "start_date": activity.get("start_date_local", ""),
                    }
                    
                    self.vector_db.upsert_activity(
                        user_id=user_id,
                        activity_id=str(activity_id),
                        activity_summary=summary,
                        embedding=embedding,
                        activity_metadata=basic_metadata
                    )
                    self.logger.info(f"Successfully stored activity {activity_id} using fallback method")
                except Exception as fallback_error:
                    self.logger.error(f"Even fallback processing failed for activity {activity_id}: {str(fallback_error)}")
                    self.logger.error(f"Fallback error type: {type(fallback_error).__name__}")
                    import traceback
                    self.logger.error(f"Fallback traceback: {traceback.format_exc()}")
        
        self.logger.info(f"Completed processing {len(activities)} activities for user {user_id}")
    
    async def search_user_activities(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search user's activities using natural language query with temporal and metric awareness.
        
        Args:
            user_id: Strava athlete ID
            query: Natural language search query
            top_k: Number of results to return
            filter_metadata: Additional metadata filters
            
        Returns:
            List of matching activities with scores
        """
        try:
            # Import temporal processor here to avoid circular imports
            from .temporal_processor import TemporalQueryProcessor
            
            # Process query for temporal and metric context
            temporal_processor = TemporalQueryProcessor()
            query_context = temporal_processor.process_query(query)
            
            # Generate embeddings for different aspects of the query
            query_embeddings = {}
            
            # Main query embedding
            query_embeddings["main"] = self.embedding_service.generate_embedding(query_context.enhanced_query)
            
            # Generate additional embeddings based on query context
            if query_context.has_metric_context:
                # Create metrics-focused query
                metrics_query = f"Activity metrics and performance data: {query}"
                query_embeddings["metrics"] = self.embedding_service.generate_embedding(metrics_query)
            
            if query_context.has_temporal_context:
                # Create temporal-focused query
                temporal_query = f"Activity timing and date information: {query_context.enhanced_query}"
                query_embeddings["temporal"] = self.embedding_service.generate_embedding(temporal_query)
            
            # Create Pinecone metadata filter from temporal/metric context
            metadata_filter = temporal_processor.create_pinecone_filter(query_context)
            
            # Combine with additional filters if provided
            if filter_metadata:
                metadata_filter.update(filter_metadata)
            
            # Determine which vector types to search based on query context
            vector_types = ["main"]
            if query_context.has_metric_context:
                vector_types.append("metrics")
            if query_context.has_temporal_context:
                vector_types.append("temporal")
            
            # Perform hybrid search
            results = self.vector_db.search_activities_hybrid(
                user_id=user_id,
                query_embeddings=query_embeddings,
                top_k=top_k,
                metadata_filter=metadata_filter if metadata_filter else None,
                vector_types=vector_types,
                boost_recent=query_context.has_temporal_context
            )
            
            # Convert to response format
            search_results = []
            for result in results:
                search_results.append({
                    "activity_id": result.metadata.get("main_activity_id") or result.metadata.get("activity_id"),
                    "summary": result.metadata.get("summary"),
                    "score": result.score,
                    "activity_type": result.metadata.get("activity_type"),
                    "start_date": result.metadata.get("start_date"),
                    "date": result.metadata.get("date"),
                    "distance": result.metadata.get("distance"),
                    "distance_km": result.metadata.get("distance_km"),
                    "moving_time": result.metadata.get("moving_time"),
                    "duration_minutes": result.metadata.get("duration_minutes"),
                    "avg_power_watts": result.metadata.get("avg_power_watts"),
                    "avg_heartrate": result.metadata.get("avg_heartrate"),
                    "avg_speed_kmh": result.metadata.get("avg_speed_kmh"),
                    "vector_type": result.metadata.get("vector_type"),
                    "metadata": result.metadata
                })
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Search failed for user {user_id}: {str(e)}")
            raise
    
    async def get_ingestion_status(self, user_id: str) -> Optional[IngestionStatus]:
        """
        Get current ingestion status for a user from memory.
        
        Args:
            user_id: Strava athlete ID
            
        Returns:
            Ingestion status or None if not found
        """
        return self.ingestion_status.get(user_id)
    
    async def get_user_activity_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics about user's stored activities.
        
        Args:
            user_id: Strava athlete ID
            
        Returns:
            Activity statistics
        """
        try:
            activity_count = self.vector_db.get_user_activity_count(user_id)
            
            return {
                "total_activities": activity_count,
                "ingestion_status": await self.get_ingestion_status(user_id)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get stats for user {user_id}: {str(e)}")
            return {"total_activities": 0, "ingestion_status": None}
    
    async def delete_user_data(self, user_id: str) -> None:
        """
        Delete all data for a user from all storage systems.
        
        Args:
            user_id: Strava athlete ID
        """
        try:
            # Delete from Pinecone vector database
            self.vector_db.delete_user_activities(user_id)
            self.logger.info(f"Deleted Pinecone data for user {user_id}")
            
            # Delete from MySQL database
            try:
                success = await mysql_service.delete_user_data(user_id)
                if success:
                    self.logger.info(f"Deleted MySQL data for user {user_id}")
                else:
                    self.logger.warning(f"MySQL deletion returned false for user {user_id}")
            except Exception as e:
                self.logger.error(f"Failed to delete MySQL data for user {user_id}: {str(e)}")
                raise
            
            # Remove from in-memory caches
            if user_id in self.ingestion_status:
                del self.ingestion_status[user_id]
                
            if user_id in self.user_tss_calculations:
                del self.user_tss_calculations[user_id]
                
            # Delete local storage file (legacy support)
            try:
                tss_file_path = self._get_tss_file_path(user_id)
                if os.path.exists(tss_file_path):
                    os.remove(tss_file_path)
                    self.logger.info(f"Deleted legacy TSS storage file for user {user_id}")
            except Exception as e:
                self.logger.warning(f"Failed to delete legacy TSS storage file for user {user_id}: {str(e)}")
                
            self.logger.info(f"Successfully deleted all data for user {user_id} from all storage systems")
            
        except Exception as e:
            self.logger.error(f"Failed to delete data for user {user_id}: {str(e)}")
            raise
    
    async def get_user_tss_calculations(self, user_id: str) -> List[TSSCalculation]:
        """
        Get TSS calculations for a user from MySQL database.
        
        Args:
            user_id: Strava athlete ID
            
        Returns:
            List of TSS calculations
        """
        # First try to get from memory cache
        if user_id in self.user_tss_calculations:
            return self.user_tss_calculations[user_id]
        
        # Get TSS data from MySQL database
        try:
            tss_data = await mysql_service.get_user_tss_data(user_id)
            if not tss_data:
                self.logger.info(f"No TSS data found in MySQL for user {user_id}")
                return []
            
            # Convert MySQL data to TSSCalculation objects
            tss_calculations = []
            for record in tss_data:
                tss_calc = TSSCalculation(
                    activity_id=record['activity_id'],
                    tss_value=record['tss_value'],
                    calculation_method=record['calculation_method'],
                    sport_type=record['sport_type'],
                    intensity_factor=record['intensity_factor'],
                    duration_hours=record['duration_hours'],
                    activity_date=record['activity_date'],
                    notes=record['notes'],
                    original_activity_type=record['original_activity_type']
                )
                tss_calculations.append(tss_calc)
            
            # Cache the results
            self.user_tss_calculations[user_id] = tss_calculations
            self.logger.info(f"Retrieved {len(tss_calculations)} TSS calculations from MySQL for user {user_id}")
            
            return tss_calculations
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve TSS data from MySQL for user {user_id}: {str(e)}")
            return []
    
    def _convert_metadata_to_activity(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert activity metadata back to activity format for TSS calculation.
        
        Args:
            metadata: Activity metadata from vector database
            
        Returns:
            Activity dictionary compatible with TSS calculation
        """
        return {
            'id': metadata.get('main_activity_id', metadata.get('activity_id')),
            'type': metadata.get('activity_type', 'unknown'),
            'name': metadata.get('name', ''),
            'distance': metadata.get('distance', 0),
            'moving_time': metadata.get('moving_time', 0),
            'start_date_local': metadata.get('start_date', metadata.get('date', '')),
            'average_watts': metadata.get('avg_power_watts', 0),
            'average_heartrate': metadata.get('avg_heartrate', 0),
            'elapsed_time': metadata.get('elapsed_time', metadata.get('moving_time', 0)),
            'total_elevation_gain': metadata.get('total_elevation_gain', 0),
            'avg_speed': metadata.get('avg_speed_ms', 0),
            'max_speed': metadata.get('max_speed_ms', 0),
            'max_heartrate': metadata.get('max_heartrate', 0),
            'max_watts': metadata.get('max_power_watts', 0),
        }
    
    async def get_user_fitness_metrics(self, user_id: str, force_recalculate: bool = False) -> List[FitnessMetrics]:
        """
        Get fitness metrics (CTL, ATL, TSB) for a user, with MySQL caching.
        
        Args:
            user_id: Strava athlete ID
            force_recalculate: If True, bypass cache and recalculate
            
        Returns:
            List of daily fitness metrics
        """
        # Try to get cached fitness metrics from MySQL first (unless forcing recalculation)
        if not force_recalculate:
            try:
                cached_trends = await mysql_service.get_fitness_trends(user_id, days=90)
                if cached_trends:
                    self.logger.info(f"Retrieved {len(cached_trends)} cached fitness metrics from MySQL for user {user_id}")
                    # Convert cached data to FitnessMetrics objects
                    fitness_metrics = []
                    for trend in cached_trends:
                        fitness_metrics.append(FitnessMetrics(
                            date=trend['date'],
                            total_tss=trend['total_tss'] or 0,
                            cycling_tss=0,  # Not stored in fitness_metrics table
                            running_tss=0,
                            swimming_tss=0,
                            other_tss=0,
                            ctl=trend['ctl'] or 0,
                            atl=trend['atl'] or 0,
                            tsb=trend['tsb'] or 0
                        ))
                    return fitness_metrics
            except Exception as e:
                self.logger.warning(f"Failed to retrieve cached fitness metrics for user {user_id}, will recalculate: {str(e)}")
        
        # If no cache or force recalculation, calculate from TSS data
        tss_calculations = await self.get_user_tss_calculations(user_id)
        if not tss_calculations:
            return []
        
        # Group TSS by date
        daily_tss = {}
        for calc in tss_calculations:
            date = getattr(calc, 'activity_date', 'unknown')
            if date == 'unknown':
                continue
                
            if date not in daily_tss:
                daily_tss[date] = {
                    'total': 0,
                    'cycling': 0,
                    'running': 0,
                    'swimming': 0,
                    'other': 0
                }
            
            daily_tss[date]['total'] += calc.tss_value
            daily_tss[date][calc.sport_type] += calc.tss_value
        
        # Calculate CTL/ATL
        fitness_metrics = self.metrics_service._calculate_ctl_atl(daily_tss)
        
        # Cache the calculated metrics to MySQL for future use
        await self._cache_fitness_metrics(user_id, fitness_metrics)
        
        return fitness_metrics
    
    async def _cache_fitness_metrics(self, user_id: str, fitness_metrics: List[FitnessMetrics]) -> None:
        """
        Cache calculated fitness metrics to MySQL database.
        
        Args:
            user_id: Strava athlete ID
            fitness_metrics: List of calculated fitness metrics
        """
        try:
            # Convert FitnessMetrics to MySQL FitnessMetrics format and save
            from datetime import date
            from app.services.mysql_service import FitnessMetrics as MySQLFitnessMetrics
            
            cached_count = 0
            for metric in fitness_metrics:
                mysql_metric = MySQLFitnessMetrics(
                    user_id=user_id,
                    metric_date=date.fromisoformat(metric.date),
                    ctl=metric.ctl,
                    atl=metric.atl,
                    tsb=metric.tsb,
                    total_tss=metric.total_tss
                )
                
                success = await mysql_service.save_fitness_metrics(mysql_metric)
                if success:
                    cached_count += 1
            
            self.logger.info(f"Cached {cached_count}/{len(fitness_metrics)} fitness metrics to MySQL for user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to cache fitness metrics for user {user_id}: {str(e)}")
    
    async def get_user_fitness_status(self, user_id: str) -> Dict[str, Any]:
        """
        Get current fitness status for a user.
        
        Args:
            user_id: Strava athlete ID
            
        Returns:
            Dictionary with current fitness status and recommendations
        """
        try:
            fitness_metrics = await self.get_user_fitness_metrics(user_id)
            if not fitness_metrics:
                return {
                    "status": "no_data",
                    "status_color": "#6b7280",
                    "recommendation": "Complete a Full Sync to analyze your fitness data.",
                    "fitness_level": "unknown",
                    "ctl": 0,
                    "atl": 0,
                    "tsb": 0
                }
            
            # Get latest metrics for CTL/ATL/TSB
            latest_metrics = fitness_metrics[0]
            
            # Calculate total sport breakdown across all activities
            tss_calculations = await self.get_user_tss_calculations(user_id)
            total_sport_breakdown = {"cycling": 0, "running": 0, "swimming": 0, "other": 0}
            
            for calc in tss_calculations:
                total_sport_breakdown[calc.sport_type] += calc.tss_value
            
            # Get fitness status with corrected sport breakdown
            fitness_status = self.metrics_service.get_fitness_status(latest_metrics)
            fitness_status["sport_breakdown"] = total_sport_breakdown
            
            return fitness_status
            
        except Exception as e:
            self.logger.error(f"Failed to get fitness status for user {user_id}: {str(e)}")
            return {
                "status": "error",
                "status_color": "#ef4444",
                "recommendation": "Error calculating fitness status. Please try syncing again.",
                "fitness_level": "unknown",
                "ctl": 0,
                "atl": 0,
                "tsb": 0
            }
    
    async def get_user_threshold_estimates(self, user_id: str, force_recalculate: bool = False) -> Dict[str, Any]:
        """
        Get threshold estimates for a user with MySQL caching.
        
        Args:
            user_id: Strava athlete ID
            force_recalculate: If True, bypass cache and recalculate
            
        Returns:
            Dictionary with threshold estimates and confidence levels
        """
        # Try to get cached thresholds from MySQL first (unless forcing recalculation)
        if not force_recalculate:
            try:
                cached_data = await mysql_service.get_user_thresholds(user_id)
                if cached_data:
                    self.logger.info(f"Retrieved cached thresholds from MySQL for user {user_id}")
                    # Convert to expected format
                    thresholds = cached_data['thresholds']
                    confidence_levels = cached_data['confidence_levels']
                    
                    return {
                        "ftp": thresholds.get('ftp'),
                        "running_threshold_pace": thresholds.get('running_threshold_pace'),
                        "swimming_threshold_pace": thresholds.get('swimming_threshold_pace'),
                        "lthr": thresholds.get('lthr'),
                        "max_hr": thresholds.get('max_hr'),
                        "confidence": confidence_levels,
                        "updated_at": cached_data['updated_at']
                    }
            except Exception as e:
                self.logger.warning(f"Failed to retrieve cached thresholds for user {user_id}, will recalculate: {str(e)}")
        
        # If no cache or force recalculation, calculate from activities
        try:
            # Get user's activities from vector DB
            self.logger.info(f"Getting activities from vector DB for user {user_id}")
            activities = self.vector_db.get_user_activities(user_id)
            self.logger.info(f"Retrieved {len(activities) if activities else 0} activities from vector DB")
            
            if not activities:
                self.logger.info(f"No activities found for user {user_id}, returning no_data response")
                return {
                    "ftp": None,
                    "running_threshold_pace": None,
                    "swimming_threshold_pace": None,
                    "lthr": None,
                    "confidence": {
                        "ftp": "no_data",
                        "running_threshold_pace": "no_data",
                        "swimming_threshold_pace": "no_data",
                        "lthr": "no_data"
                    }
                }
            
            # Convert metadata back to activity format for threshold estimation
            self.logger.info(f"Converting {len(activities)} activities for threshold estimation")
            converted_activities = []
            for activity_metadata in activities:
                try:
                    activity = self._convert_metadata_to_activity_for_threshold(activity_metadata)
                    converted_activities.append(activity)
                except Exception as e:
                    self.logger.warning(f"Failed to convert activity metadata: {str(e)}")
                    continue
            
            self.logger.info(f"Successfully converted {len(converted_activities)} activities")
            
            # Estimate thresholds from actual activities
            self.logger.info(f"Estimating thresholds from {len(converted_activities)} activities")
            thresholds = self.threshold_estimator.estimate_all_thresholds(converted_activities)
            self.logger.info(f"Threshold estimates: FTP={thresholds.ftp}, Running={thresholds.running_threshold_pace}, "
                           f"Swimming={thresholds.swimming_threshold_pace}, LTHR={thresholds.lthr}")
            
            # Calculate confidence levels based on data availability
            confidence = self._calculate_threshold_confidence(converted_activities, thresholds)
            self.logger.info(f"Confidence levels: {confidence}")
            
            # Prepare threshold data for return and caching - fix None comparison errors
            threshold_data = {
                "ftp": thresholds.ftp if thresholds.ftp and thresholds.ftp > 0 else None,
                "running_threshold_pace": thresholds.running_threshold_pace if thresholds.running_threshold_pace and thresholds.running_threshold_pace > 0 else None,
                "swimming_threshold_pace": thresholds.swimming_threshold_pace if thresholds.swimming_threshold_pace and thresholds.swimming_threshold_pace > 0 else None,
                "lthr": thresholds.lthr if thresholds.lthr and thresholds.lthr > 0 else None,
                "max_hr": thresholds.max_hr if hasattr(thresholds, 'max_hr') and thresholds.max_hr and thresholds.max_hr > 0 else None,
                "confidence": confidence
            }
            
            # Cache the calculated thresholds to MySQL for future use
            await self._cache_user_thresholds(user_id, threshold_data)
            
            return threshold_data
            
        except Exception as e:
            self.logger.error(f"Failed to get threshold estimates for user {user_id}: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "ftp": None,
                "running_threshold_pace": None,
                "swimming_threshold_pace": None,
                "lthr": None,
                "confidence": {
                    "ftp": "error",
                    "running_threshold_pace": "error",
                    "swimming_threshold_pace": "error",
                    "lthr": "error"
                }
            }
    
    async def _cache_user_thresholds(self, user_id: str, threshold_data: Dict[str, Any]) -> None:
        """
        Cache calculated user thresholds to MySQL database.
        
        Args:
            user_id: Strava athlete ID
            threshold_data: Dictionary with threshold values and confidence levels
        """
        try:
            # Extract thresholds and confidence levels
            thresholds = {
                'ftp': threshold_data.get('ftp'),
                'running_threshold_pace': threshold_data.get('running_threshold_pace'),
                'swimming_threshold_pace': threshold_data.get('swimming_threshold_pace'),
                'lthr': threshold_data.get('lthr'),
                'max_hr': threshold_data.get('max_hr')
            }
            
            confidence_levels = threshold_data.get('confidence', {})
            
            success = await mysql_service.save_user_thresholds(user_id, thresholds, confidence_levels)
            if success:
                self.logger.info(f"Cached user thresholds to MySQL for user {user_id}")
            else:
                self.logger.warning(f"Failed to cache user thresholds for user {user_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to cache user thresholds for user {user_id}: {str(e)}")
    
    async def _update_cached_thresholds_after_ingestion(self, user_id: str, thresholds) -> None:
        """
        Update cached thresholds in MySQL after ingesting new activities.
        
        Args:
            user_id: Strava athlete ID
            thresholds: ThresholdEstimates object from ingestion
        """
        try:
            # Convert ThresholdEstimates to threshold data format
            threshold_data = {
                "ftp": thresholds.ftp if thresholds.ftp and thresholds.ftp > 0 else None,
                "running_threshold_pace": thresholds.running_threshold_pace if thresholds.running_threshold_pace and thresholds.running_threshold_pace > 0 else None,
                "swimming_threshold_pace": thresholds.swimming_threshold_pace if thresholds.swimming_threshold_pace and thresholds.swimming_threshold_pace > 0 else None,
                "lthr": thresholds.lthr if thresholds.lthr and thresholds.lthr > 0 else None,
                "max_hr": thresholds.max_hr if hasattr(thresholds, 'max_hr') and thresholds.max_hr and thresholds.max_hr > 0 else None,
            }
            
            # Calculate proper confidence levels based on activity data
            activities = self.user_activities.get(user_id, [])
            confidence_levels = self.threshold_estimator.get_threshold_confidence(activities)
            
            # Cache to MySQL
            success = await mysql_service.save_user_thresholds(user_id, threshold_data, confidence_levels)
            if success:
                self.logger.info(f"Updated cached thresholds in MySQL for user {user_id}")
            else:
                self.logger.warning(f"Failed to update cached thresholds for user {user_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to update cached thresholds after ingestion for user {user_id}: {str(e)}")
            
    def _convert_metadata_to_activity_for_threshold(self, activity_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert activity metadata back to activity format for threshold estimation.
        
        Args:
            activity_metadata: Activity metadata from vector DB
            
        Returns:
            Activity dict in format expected by threshold estimator
        """
        # Handle potential None values and convert to appropriate types
        def safe_float(value, default=0):
            if value is None:
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        def safe_int(value, default=0):
            if value is None:
                return default
            try:
                return int(value)
            except (ValueError, TypeError):
                return default
                
        return {
            "id": activity_metadata.get("activity_id") or activity_metadata.get("main_activity_id"),
            "type": activity_metadata.get("activity_type", "unknown"),
            "start_date_local": activity_metadata.get("start_date") or activity_metadata.get("date", ""),
            "distance": safe_float(activity_metadata.get("distance", 0)),
            "moving_time": safe_int(activity_metadata.get("moving_time", 0)),
            "total_elevation_gain": safe_float(activity_metadata.get("total_elevation_gain", 0)),
            "average_speed": safe_float(activity_metadata.get("average_speed") or activity_metadata.get("avg_speed_ms", 0)),
            "max_speed": safe_float(activity_metadata.get("max_speed") or activity_metadata.get("max_speed_ms", 0)),
            "average_heartrate": safe_float(activity_metadata.get("avg_heartrate") or activity_metadata.get("average_heartrate", 0)),
            "max_heartrate": safe_float(activity_metadata.get("max_heartrate", 0)),
            "average_watts": safe_float(activity_metadata.get("avg_power_watts") or activity_metadata.get("average_watts", 0)),
            "weighted_average_watts": safe_float(activity_metadata.get("weighted_average_watts", 0)),
            "normalized_power": safe_float(activity_metadata.get("normalized_power", 0)),
            "max_watts": safe_float(activity_metadata.get("max_watts") or activity_metadata.get("max_power_watts", 0)),
            "kilojoules": safe_float(activity_metadata.get("kilojoules", 0)),
            "average_cadence": safe_float(activity_metadata.get("average_cadence", 0)),
            "name": str(activity_metadata.get("name", "")),
            "trainer": bool(activity_metadata.get("trainer", False)),
            "commute": bool(activity_metadata.get("commute", False)),
        }
    
    async def get_activity_by_id(self, activity_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a single activity by its Strava ID.
        
        Args:
            activity_id: Strava activity ID
            
        Returns:
            Activity data or None if not found
        """
        try:
            # First, try to get from vector DB metadata
            results = self.vector_db.search_by_activity_id(str(activity_id))
            
            if results and len(results) > 0:
                # Convert metadata back to activity format
                metadata = results[0].metadata
                activity = self._convert_metadata_to_activity_for_threshold(metadata)
                
                # Include computed metrics if available
                activity['normalized_power'] = metadata.get('normalized_power')
                activity['intensity_factor'] = metadata.get('intensity_factor')
                activity['efficiency_factor'] = metadata.get('efficiency_factor')
                activity['grade_adjusted_pace'] = metadata.get('grade_adjusted_pace')
                activity['variability_index'] = metadata.get('variability_index')
                
                return activity
                
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get activity {activity_id}: {str(e)}")
            return None
    
    async def get_user_activities(self, user_id: str, days: int = 90, sport_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get user's activities for a specific time period.
        
        Args:
            user_id: Strava athlete ID
            days: Number of days to look back
            sport_type: Optional filter by sport type
            
        Returns:
            List of activities
        """
        try:
            # Get activities from vector DB
            activities = self.vector_db.get_user_activities(user_id, limit=1000)
            
            if not activities:
                return []
            
            # Filter by date range
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=days)
            
            filtered_activities = []
            for activity_metadata in activities:
                # Parse activity date
                date_str = activity_metadata.get('start_date') or activity_metadata.get('date', '')
                if date_str:
                    try:
                        activity_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        if activity_date >= cutoff_date:
                            # Convert to activity format
                            activity = self._convert_metadata_to_activity_for_threshold(activity_metadata)
                            
                            # Filter by sport type if specified
                            if sport_type:
                                activity_type = activity.get('type', '').lower()
                                sport_category = self.metrics_service._categorize_sport(activity_type)
                                if sport_category != sport_type.lower():
                                    continue
                            
                            # Include computed metrics
                            activity['normalized_power'] = activity_metadata.get('normalized_power')
                            activity['intensity_factor'] = activity_metadata.get('intensity_factor')
                            activity['efficiency_factor'] = activity_metadata.get('efficiency_factor')
                            activity['grade_adjusted_pace'] = activity_metadata.get('grade_adjusted_pace')
                            activity['variability_index'] = activity_metadata.get('variability_index')
                            
                            filtered_activities.append(activity)
                    except Exception as e:
                        self.logger.warning(f"Failed to parse date {date_str}: {str(e)}")
            
            # Sort by date (newest first)
            filtered_activities.sort(key=lambda x: x.get('start_date_local', ''), reverse=True)
            
            return filtered_activities
            
        except Exception as e:
            self.logger.error(f"Failed to get user activities: {str(e)}")
            return []
    
    def _calculate_threshold_confidence(self, activities: List[Dict[str, Any]], thresholds) -> Dict[str, str]:
        """
        Calculate confidence levels for threshold estimates based on data availability.
        
        Args:
            activities: List of activities
            thresholds: Estimated thresholds
            
        Returns:
            Dictionary with confidence levels for each threshold
        """
        try:
            # Count activities by sport type
            cycling_count = len([a for a in activities if a.get("type", "").lower() in ["ride", "virtualride", "ebikeride"]])
            running_count = len([a for a in activities if a.get("type", "").lower() in ["run", "virtualrun", "treadmill"]])
            swimming_count = len([a for a in activities if a.get("type", "").lower() in ["swim"]])
            
            # Count activities with power data
            power_activities = len([a for a in activities if a.get("average_watts", 0) > 0])
            
            # Count activities with heart rate data
            hr_activities = len([a for a in activities if a.get("average_heartrate", 0) > 0])
            
            confidence = {}
            
            # FTP confidence
            if cycling_count == 0 or power_activities == 0:
                confidence["ftp"] = "no_data"
            elif power_activities >= 10 and cycling_count >= 15:
                confidence["ftp"] = "high"
            elif power_activities >= 5 and cycling_count >= 8:
                confidence["ftp"] = "medium"
            else:
                confidence["ftp"] = "low"
            
            # Running threshold confidence
            if running_count == 0:
                confidence["running_threshold_pace"] = "no_data"
            elif running_count >= 15:
                confidence["running_threshold_pace"] = "high"
            elif running_count >= 8:
                confidence["running_threshold_pace"] = "medium"
            else:
                confidence["running_threshold_pace"] = "low"
            
            # Swimming threshold confidence
            if swimming_count == 0:
                confidence["swimming_threshold_pace"] = "no_data"
            elif swimming_count >= 10:
                confidence["swimming_threshold_pace"] = "high"
            elif swimming_count >= 5:
                confidence["swimming_threshold_pace"] = "medium"
            else:
                confidence["swimming_threshold_pace"] = "low"
            
            # LTHR confidence
            if hr_activities == 0:
                confidence["lthr"] = "no_data"
            elif hr_activities >= 20:
                confidence["lthr"] = "high"
            elif hr_activities >= 10:
                confidence["lthr"] = "medium"
            else:
                confidence["lthr"] = "low"
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Failed to calculate threshold confidence: {str(e)}")
            return {
                "ftp": "error",
                "running_threshold_pace": "error", 
                "swimming_threshold_pace": "error",
                "lthr": "error"
            }