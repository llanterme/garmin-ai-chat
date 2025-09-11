"""Vector database service for Pinecone integration with multi-vector storage and hybrid search."""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel

from ..core.logging import get_logger

logger = get_logger(__name__)


class VectorDocument(BaseModel):
    """Document to be stored in vector database."""
    id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]


class VectorSearchResult(BaseModel):
    """Search result from vector database."""
    id: str
    score: float
    metadata: Dict[str, Any]
    text: Optional[str] = None


class VectorDBConfig(BaseModel):
    """Configuration for vector database."""
    index_name: str = "garmin-fitness-activities"
    dimension: int = 1536
    metric: str = "cosine"
    cloud: str = "aws"
    region: str = "us-east-1"


class VectorDBService:
    """Service for managing vector database operations with Pinecone."""
    
    def __init__(self):
        from ..core.config import settings
        
        self.api_key = settings.pinecone_api_key
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        self.pc = Pinecone(api_key=self.api_key)
        self.config = VectorDBConfig()
        
        # Override config from settings
        self.config.index_name = settings.pinecone_index_name
        self.config.region = settings.pinecone_environment
        
        # Initialize index
        self._ensure_index_exists()
        self.index = self.pc.Index(self.config.index_name)
    
    def _ensure_index_exists(self):
        """Create Pinecone index if it doesn't exist."""
        try:
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if self.config.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.config.index_name}")
                self.pc.create_index(
                    name=self.config.index_name,
                    dimension=self.config.dimension,
                    metric=self.config.metric,
                    spec=ServerlessSpec(
                        cloud=self.config.cloud,
                        region=self.config.region
                    )
                )
                logger.info(f"Successfully created index: {self.config.index_name}")
            else:
                logger.info(f"Using existing index: {self.config.index_name}")
        except Exception as e:
            logger.error(f"Failed to ensure index exists: {str(e)}")
            raise
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean metadata to ensure compatibility with Pinecone."""
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif isinstance(value, list) and all(isinstance(v, str) for v in value):
                cleaned[key] = value
            elif isinstance(value, datetime):
                cleaned[key] = value.isoformat()
            else:
                cleaned[key] = str(value)
        return cleaned
    
    def _create_activity_metadata(
        self, 
        user_id: str,
        activity_data: Dict[str, Any],
        vector_type: str
    ) -> Dict[str, Any]:
        """Create comprehensive metadata for activity vector."""
        metadata = {
            # Core identifiers
            "user_id": user_id,
            "activity_id": activity_data.get("garmin_activity_id"),
            "main_activity_id": activity_data.get("garmin_activity_id"), # For grouping vectors
            "vector_type": vector_type,
            "data_source": "garmin"
        }
        
        # Temporal data
        start_time = activity_data.get("start_time")
        if start_time:
            if isinstance(start_time, str):
                try:
                    dt = datetime.fromisoformat(start_time.replace('T', ' ').replace('Z', ''))
                except:
                    dt = datetime.now()
            else:
                dt = start_time
            
            metadata.update({
                "date": dt.strftime("%Y-%m-%d"),
                "timestamp": int(dt.timestamp()),
                "day_of_week": dt.strftime("%A").lower(),
                "month": dt.strftime("%B").lower(),
                "year": dt.year
            })
        
        # Activity classification
        activity_type = activity_data.get("activity_type", "unknown")
        metadata["activity_type"] = activity_type.lower()
        
        # Performance metrics (convert to appropriate types)
        numeric_fields = [
            "distance", "duration", "average_speed", "max_speed",
            "average_heart_rate", "max_heart_rate", "average_power", "max_power",
            "elevation_gain", "elevation_loss", "calories"
        ]
        
        for field in numeric_fields:
            value = activity_data.get(field)
            if value is not None:
                try:
                    if field == "distance" and value > 0:
                        metadata["distance_km"] = round(float(value) / 1000, 2)
                    elif field == "duration" and value > 0:
                        metadata["duration_minutes"] = round(float(value) / 60, 1)
                    elif field in ["average_speed", "max_speed"] and value > 0:
                        metadata[f"{field}_kmh"] = round(float(value) * 3.6, 1)
                    elif "heart_rate" in field and value > 0:
                        metadata[field] = int(value)
                    elif "power" in field and value > 0:
                        metadata[field] = int(value)
                    elif "elevation" in field:
                        metadata[field] = round(float(value), 1)
                    elif field == "calories" and value > 0:
                        metadata[field] = int(value)
                except (ValueError, TypeError):
                    continue
        
        # Performance indicators
        has_hr = activity_data.get("average_heart_rate") is not None
        has_power = activity_data.get("average_power") is not None
        
        metadata.update({
            "has_heartrate": has_hr,
            "has_power_data": has_power
        })
        
        # Calculate efficiency score if we have power and distance
        if has_power and activity_data.get("distance") and activity_data.get("duration"):
            try:
                power = float(activity_data["average_power"])
                distance_km = float(activity_data["distance"]) / 1000
                duration_hours = float(activity_data["duration"]) / 3600
                if duration_hours > 0:
                    speed_kmh = distance_km / duration_hours
                    if speed_kmh > 0:
                        efficiency = power / speed_kmh
                        metadata["efficiency_score"] = round(efficiency, 2)
            except (ValueError, ZeroDivisionError):
                pass
        
        # Training stress score if available
        if activity_data.get("training_stress_score"):
            try:
                metadata["intensity_score"] = float(activity_data["training_stress_score"])
            except ValueError:
                pass
        
        return self._clean_metadata(metadata)
    
    async def upsert_activity_multi_vector(
        self,
        user_id: str,
        activity_data: Dict[str, Any],
        embeddings: Dict[str, List[float]],
        summaries: Dict[str, str]
    ) -> None:
        """Store multiple embeddings for a single activity."""
        try:
            namespace = f"user_{user_id}"
            activity_id = activity_data.get("garmin_activity_id")
            
            if not activity_id:
                raise ValueError("Activity must have garmin_activity_id")
            
            # Prepare vectors for upsert
            vectors = []
            for vector_type in ["main", "metrics", "temporal", "performance"]:
                if vector_type not in embeddings:
                    logger.warning(f"Missing {vector_type} embedding for activity {activity_id}")
                    continue
                
                vector_id = f"{activity_id}_{vector_type}"
                metadata = self._create_activity_metadata(user_id, activity_data, vector_type)
                metadata["summary_text"] = summaries.get(vector_type, "")
                
                vectors.append({
                    "id": vector_id,
                    "values": embeddings[vector_type],
                    "metadata": metadata
                })
            
            if vectors:
                self.index.upsert(vectors=vectors, namespace=namespace)
                logger.info(f"Upserted {len(vectors)} vectors for activity {activity_id} in namespace {namespace}")
            else:
                logger.warning(f"No vectors to upsert for activity {activity_id}")
                
        except Exception as e:
            logger.error(f"Failed to upsert activity vectors: {str(e)}")
            raise
    
    async def search_activities_hybrid(
        self,
        user_id: str,
        query_embeddings: Dict[str, List[float]],
        metadata_filter: Optional[Dict[str, Any]] = None,
        vector_types: Optional[List[str]] = None,
        top_k: int = 10,
        boost_recent: bool = True
    ) -> List[VectorSearchResult]:
        """Perform hybrid search with multi-vector approach and metadata filtering."""
        try:
            namespace = f"user_{user_id}"
            
            # Default to all vector types
            if vector_types is None:
                vector_types = ["main", "metrics", "temporal", "performance"]
            
            all_results = []
            
            # Search each vector type
            for vector_type in vector_types:
                if vector_type not in query_embeddings:
                    continue
                
                # Prepare filter
                search_filter = {"vector_type": vector_type}
                if metadata_filter:
                    search_filter.update(metadata_filter)
                
                try:
                    # Perform search
                    search_results = self.index.query(
                        vector=query_embeddings[vector_type],
                        filter=search_filter,
                        top_k=top_k * 2,  # Get more results for deduplication
                        include_metadata=True,
                        namespace=namespace
                    )
                    
                    # Apply vector-type specific boosts
                    boost_factor = self._get_vector_boost_factor(vector_type)
                    
                    for match in search_results.matches:
                        result = VectorSearchResult(
                            id=match.id,
                            score=match.score * boost_factor,
                            metadata=match.metadata,
                            text=match.metadata.get("summary_text", "")
                        )
                        all_results.append(result)
                        
                except Exception as e:
                    logger.warning(f"Search failed for vector type {vector_type}: {str(e)}")
                    continue
            
            # Deduplicate by main_activity_id and get top results
            deduped_results = self._deduplicate_results(all_results, boost_recent)
            
            # Sort by score and return top_k
            deduped_results.sort(key=lambda x: x.score, reverse=True)
            return deduped_results[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to perform hybrid search: {str(e)}")
            return []
    
    def _get_vector_boost_factor(self, vector_type: str) -> float:
        """Get boost factor for different vector types."""
        boost_factors = {
            "main": 1.0,          # Baseline
            "metrics": 1.1,       # Boost for performance queries
            "temporal": 1.2,      # Boost for time-based queries
            "performance": 1.05   # Slight boost for training analysis
        }
        return boost_factors.get(vector_type, 1.0)
    
    def _deduplicate_results(
        self, 
        results: List[VectorSearchResult], 
        boost_recent: bool = True
    ) -> List[VectorSearchResult]:
        """Deduplicate results by activity ID, keeping highest scoring match."""
        activity_results = {}
        
        for result in results:
            main_activity_id = result.metadata.get("main_activity_id")
            if not main_activity_id:
                continue
            
            # Apply recency boost if enabled
            final_score = result.score
            if boost_recent:
                final_score = self._apply_recency_boost(result)
            
            # Keep result with highest score for each activity
            if main_activity_id not in activity_results or final_score > activity_results[main_activity_id].score:
                result.score = final_score
                activity_results[main_activity_id] = result
        
        return list(activity_results.values())
    
    def _apply_recency_boost(self, result: VectorSearchResult) -> float:
        """Apply recency boost to results within 30 days."""
        try:
            timestamp = result.metadata.get("timestamp")
            if timestamp:
                now = datetime.now().timestamp()
                days_ago = (now - timestamp) / (24 * 3600)
                
                # Boost activities within 30 days
                if days_ago <= 30:
                    boost = 1.0 + (0.1 * (30 - days_ago) / 30)
                    return result.score * boost
            
            return result.score
            
        except (ValueError, TypeError):
            return result.score
    
    async def delete_user_activities(self, user_id: str) -> None:
        """Delete all activities for a user."""
        try:
            namespace = f"user_{user_id}"
            
            # Delete entire namespace
            self.index.delete(delete_all=True, namespace=namespace)
            logger.info(f"Deleted all activities for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete user activities: {str(e)}")
            raise
    
    async def delete_activity(self, user_id: str, activity_id: str) -> None:
        """Delete all vectors for a specific activity."""
        try:
            namespace = f"user_{user_id}"
            vector_types = ["main", "metrics", "temporal", "performance"]
            
            # Delete all vector types for this activity
            vector_ids = [f"{activity_id}_{vtype}" for vtype in vector_types]
            self.index.delete(ids=vector_ids, namespace=namespace)
            
            logger.info(f"Deleted activity {activity_id} vectors for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete activity vectors: {str(e)}")
            raise
    
    async def get_index_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about the index."""
        try:
            if user_id:
                namespace = f"user_{user_id}"
                stats = self.index.describe_index_stats(filter={"user_id": user_id})
            else:
                stats = self.index.describe_index_stats()
            
            return {
                "total_vector_count": stats.total_vector_count,
                "namespaces": dict(stats.namespaces) if stats.namespaces else {}
            }
            
        except Exception as e:
            logger.error(f"Failed to get index stats: {str(e)}")
            return {"error": str(e)}