"""
Vector database service for Pinecone integration.
Handles embeddings storage and retrieval with per-user namespacing.
"""

import os
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel
from datetime import datetime


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


class VectorDBService:
    """Service for managing vector database operations with Pinecone."""
    
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        self.pc = Pinecone(api_key=self.api_key)
        self.index_name = "athlete-iq-activities"
        self.dimension = 1536  # OpenAI text-embedding-3-small dimension
        
        # Initialize index if it doesn't exist
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)
    
    def _ensure_index_exists(self):
        """Create Pinecone index if it doesn't exist."""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean metadata to ensure compatibility with Pinecone.
        Removes null values and converts incompatible types.
        """
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
                # Skip null values
                continue
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif isinstance(value, list) and all(isinstance(v, str) for v in value):
                cleaned[key] = value
            else:
                # Convert other types to string
                cleaned[key] = str(value)
        return cleaned
    
    def upsert_activity_multi_vector(
        self,
        user_id: str,
        activity_id: str,
        embeddings: Dict[str, List[float]],
        summaries: Dict[str, str],
        activity_metadata: Dict[str, Any],
        data_source: str = "strava"
    ) -> None:
        """
        Store multiple embeddings for a single activity in the vector database.
        
        Args:
            user_id: User ID for namespacing
            activity_id: Unique activity identifier
            embeddings: Dictionary of embedding vectors for different aspects
            summaries: Dictionary of summaries corresponding to embeddings
            activity_metadata: Rich metadata about the activity
            data_source: Data source ("strava" or "garmin")
        """
        namespace = f"athlete_{user_id}"
        
        # Prepare base metadata
        base_metadata = {
            "user_id": user_id,
            "activity_id": activity_id,
            "data_source": data_source,
            "created_at": datetime.utcnow().isoformat(),
            **activity_metadata
        }
        
        # Clean metadata to remove null values
        base_metadata = self._clean_metadata(base_metadata)
        
        # Create vectors for each embedding type
        vectors = []
        for vector_type, embedding in embeddings.items():
            doc_id = f"{activity_id}_{vector_type}"
            
            # Add vector-specific metadata
            metadata = {
                **base_metadata,
                "vector_type": vector_type,
                "summary": summaries.get(vector_type, ""),
                "main_activity_id": activity_id  # For grouping
            }
            
            vectors.append({
                "id": doc_id,
                "values": embedding,
                "metadata": metadata
            })
        
        # Batch upsert all vectors for this activity
        self.index.upsert(vectors=vectors, namespace=namespace)
    
    def upsert_activity(
        self,
        user_id: str,
        activity_id: str,
        activity_summary: str,
        embedding: List[float],
        activity_metadata: Dict[str, Any]
    ) -> None:
        """
        Store a single activity embedding in the vector database (legacy support).
        
        Args:
            user_id: Strava athlete ID for namespacing
            activity_id: Unique activity identifier
            activity_summary: Human-readable activity summary
            embedding: Vector embedding of the activity
            activity_metadata: Additional metadata about the activity
        """
        namespace = f"athlete_{user_id}"
        
        # Create document ID
        doc_id = f"activity_{activity_id}_main"
        
        # Prepare and clean metadata
        metadata = {
            "user_id": user_id,
            "activity_id": activity_id,
            "main_activity_id": activity_id,
            "vector_type": "main",
            "summary": activity_summary,
            "created_at": datetime.utcnow().isoformat(),
            **activity_metadata
        }
        
        # Clean metadata to remove null values
        metadata = self._clean_metadata(metadata)
        
        # Upsert to Pinecone
        self.index.upsert(
            vectors=[{
                "id": doc_id,
                "values": embedding,
                "metadata": metadata
            }],
            namespace=namespace
        )
    
    def upsert_activities_batch(
        self,
        user_id: str,
        activities: List[Dict[str, Any]]
    ) -> None:
        """
        Store multiple activities in batch for efficiency.
        
        Args:
            user_id: Strava athlete ID for namespacing
            activities: List of activity dictionaries with required fields
        """
        namespace = f"athlete_{user_id}"
        
        vectors = []
        for activity in activities:
            doc_id = f"activity_{activity['activity_id']}"
            
            metadata = {
                "user_id": user_id,
                "activity_id": activity["activity_id"],
                "summary": activity["summary"],
                "created_at": datetime.utcnow().isoformat(),
                **activity.get("metadata", {})
            }
            
            # Clean metadata to remove null values
            metadata = self._clean_metadata(metadata)
            
            vectors.append({
                "id": doc_id,
                "values": activity["embedding"],
                "metadata": metadata
            })
        
        # Batch upsert to Pinecone
        self.index.upsert(vectors=vectors, namespace=namespace)
    
    def search_activities_hybrid(
        self,
        user_id: str,
        query_embeddings: Dict[str, List[float]],
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        vector_types: Optional[List[str]] = None,
        boost_recent: bool = True,
        data_sources: Optional[List[str]] = None
    ) -> List[VectorSearchResult]:
        """
        Hybrid search combining multiple vector types and metadata filtering.
        
        Args:
            user_id: User ID for namespacing
            query_embeddings: Dictionary of embeddings for different query aspects
            top_k: Number of results to return
            metadata_filter: Pinecone metadata filters
            vector_types: Which vector types to search (default: all)
            boost_recent: Whether to boost more recent activities
            data_sources: Which data sources to include ("strava", "garmin", or both)
            
        Returns:
            List of search results with scores and metadata
        """
        namespace = f"athlete_{user_id}"
        
        try:
            # Default vector types to search
            if vector_types is None:
                vector_types = ["main", "metrics", "temporal", "performance"]
            
            # Prepare base filter
            filter_dict = {"user_id": user_id}
            if metadata_filter:
                filter_dict.update(metadata_filter)
            
            # Add data source filtering if specified
            if data_sources:
                if len(data_sources) == 1:
                    filter_dict["data_source"] = data_sources[0]
                else:
                    # For multiple data sources, we'll need to do separate queries
                    # and combine results (Pinecone doesn't support OR filters directly)
                    pass
            
            # Collect results from different vector types
            all_results = {}  # activity_id -> best result
            
            for vector_type in vector_types:
                if vector_type not in query_embeddings:
                    continue
                
                # Add vector type filter
                current_filter = {**filter_dict, "vector_type": vector_type}
                
                # Search this vector type
                results = self.index.query(
                    vector=query_embeddings[vector_type],
                    top_k=top_k * 2,  # Get more results to merge
                    namespace=namespace,
                    filter=current_filter,
                    include_metadata=True
                )
                
                # Process results
                for match in results.matches:
                    activity_id = match.metadata.get("main_activity_id", match.metadata.get("activity_id"))
                    
                    if activity_id not in all_results or match.score > all_results[activity_id].score:
                        # Boost score based on vector type relevance
                        boosted_score = self._calculate_boosted_score(
                            match.score, vector_type, match.metadata, boost_recent
                        )
                        
                        all_results[activity_id] = VectorSearchResult(
                            id=match.id,
                            score=boosted_score,
                            metadata=match.metadata
                        )
            
            # Sort by boosted score and return top_k
            sorted_results = sorted(all_results.values(), key=lambda x: x.score, reverse=True)
            return sorted_results[:top_k]
            
        except Exception as e:
            # If namespace doesn't exist, return empty results
            if "Namespace not found" in str(e) or "404" in str(e):
                return []
            else:
                raise e
    
    def search_activities(
        self,
        user_id: str,
        query_embedding: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        vector_type: str = "main"
    ) -> List[VectorSearchResult]:
        """
        Search for similar activities using vector similarity (legacy support).
        
        Args:
            user_id: Strava athlete ID for namespacing
            query_embedding: Vector embedding of the search query
            top_k: Number of results to return
            filter_metadata: Additional metadata filters
            vector_type: Which vector type to search
            
        Returns:
            List of search results with scores and metadata
        """
        namespace = f"athlete_{user_id}"
        
        try:
            # Prepare filter (always include user_id and vector_type)
            filter_dict = {"user_id": user_id, "vector_type": vector_type}
            if filter_metadata:
                filter_dict.update(filter_metadata)
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                filter=filter_dict,
                include_metadata=True
            )
            
            # Convert to VectorSearchResult objects
            search_results = []
            for match in results.matches:
                search_results.append(VectorSearchResult(
                    id=match.id,
                    score=match.score,
                    metadata=match.metadata
                ))
            
            return search_results
        
        except Exception as e:
            # If namespace doesn't exist, return empty results
            if "Namespace not found" in str(e) or "404" in str(e):
                return []
            else:
                raise e
    
    def _calculate_boosted_score(
        self, 
        base_score: float, 
        vector_type: str, 
        metadata: Dict[str, Any],
        boost_recent: bool
    ) -> float:
        """Calculate boosted score based on vector type and recency."""
        score = base_score
        
        # Vector type boosts
        vector_boosts = {
            "main": 1.0,      # Base score
            "metrics": 1.1,   # Slight boost for metric queries
            "temporal": 1.2,  # Higher boost for temporal queries
            "performance": 1.05  # Small boost for performance queries
        }
        
        score *= vector_boosts.get(vector_type, 1.0)
        
        # Recency boost
        if boost_recent and metadata.get("timestamp"):
            try:
                import time
                current_time = time.time()
                activity_time = float(metadata["timestamp"])
                days_ago = (current_time - activity_time) / (24 * 3600)
                
                # Boost recent activities (decay over 30 days)
                if days_ago < 30:
                    recency_boost = 1.0 + (0.1 * (30 - days_ago) / 30)
                    score *= recency_boost
                    
            except (ValueError, TypeError):
                pass  # Skip recency boost if timestamp is invalid
        
        return min(score, 1.0)  # Cap at 1.0
    
    def get_mixed_source_activities(self, user_id: str, top_k: int = 100) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get activities from all data sources for a user, grouped by source.
        
        Args:
            user_id: User ID
            top_k: Maximum number of activities to return per source
            
        Returns:
            Dictionary with data sources as keys and activity lists as values
        """
        namespace = f"athlete_{user_id}"
        sources = {}
        
        for data_source in ["strava", "garmin"]:
            try:
                # Query for each data source separately
                result = self.index.query(
                    vector=[0] * self.dimension,  # Placeholder vector
                    top_k=top_k,
                    namespace=namespace,
                    include_metadata=True,
                    filter={"user_id": user_id, "vector_type": "main", "data_source": data_source}
                )
                
                activities = []
                for match in result.matches:
                    if match.metadata:
                        activities.append(match.metadata)
                
                sources[data_source] = activities
                
            except Exception as e:
                if "Namespace not found" in str(e) or "404" in str(e):
                    sources[data_source] = []
                else:
                    raise e
        
        return sources
    
    def get_data_source_stats(self, user_id: str) -> Dict[str, int]:
        """
        Get statistics about activities by data source for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with data source counts
        """
        mixed_activities = self.get_mixed_source_activities(user_id)
        
        stats = {}
        for source, activities in mixed_activities.items():
            # Count unique activities (not vectors)
            unique_activities = set()
            for activity in activities:
                activity_id = activity.get("main_activity_id", activity.get("activity_id"))
                if activity_id:
                    unique_activities.add(activity_id)
            stats[source] = len(unique_activities)
        
        return stats
    
    def delete_user_activities(self, user_id: str) -> None:
        """
        Delete all activities for a specific user.
        
        Args:
            user_id: Strava athlete ID
        """
        namespace = f"athlete_{user_id}"
        
        try:
            # Delete all vectors in the user's namespace
            self.index.delete(delete_all=True, namespace=namespace)
        except Exception as e:
            # If namespace doesn't exist, that's fine - nothing to delete
            if "Namespace not found" in str(e) or "404" in str(e):
                pass  # Namespace doesn't exist, which is fine for deletion
            else:
                raise e
    
    def get_user_activity_count(self, user_id: str) -> int:
        """
        Get the number of activities stored for a user.
        
        Args:
            user_id: Strava athlete ID
            
        Returns:
            Number of activities stored (counting unique activities, not vectors)
        """
        namespace = f"athlete_{user_id}"
        
        try:
            # Query for distinct activity IDs only
            # This is more accurate than using vector_count, which counts all vector types
            result = self.index.query(
                vector=[0] * self.dimension,  # Placeholder vector
                top_k=10000,  # Large limit to get all possible activities
                namespace=namespace,
                include_metadata=True,
                filter={"user_id": user_id}
            )
            
            # Count unique main_activity_id values
            unique_activity_ids = set()
            for match in result.matches:
                activity_id = match.metadata.get("main_activity_id", match.metadata.get("activity_id"))
                if activity_id:
                    unique_activity_ids.add(activity_id)
            
            return len(unique_activity_ids)
        except Exception as e:
            # If namespace doesn't exist, return 0
            if "Namespace not found" in str(e) or "404" in str(e):
                return 0
            else:
                raise e
    
    def get_user_activities(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all activities for a user from the vector database.
        
        Args:
            user_id: Strava athlete ID
            
        Returns:
            List of activity metadata dictionaries
        """
        namespace = f"athlete_{user_id}"
        
        try:
            # Query for all activity metadata
            result = self.index.query(
                vector=[0] * self.dimension,  # Placeholder vector
                top_k=10000,  # Large limit to get all possible activities
                namespace=namespace,
                include_metadata=True,
                filter={"user_id": user_id, "vector_type": "main"}  # Only get main vectors to avoid duplicates
            )
            
            # Extract activity metadata
            activities = []
            for match in result.matches:
                if match.metadata:
                    activities.append(match.metadata)
            
            return activities
        except Exception as e:
            # If namespace doesn't exist, return empty list
            if "Namespace not found" in str(e) or "404" in str(e):
                return []
            else:
                raise e
    
    def search_by_activity_id(self, activity_id: str) -> List[VectorSearchResult]:
        """
        Search for a specific activity by ID across all namespaces.
        
        Args:
            activity_id: The Strava activity ID to search for
            
        Returns:
            List of search results (should typically be one)
        """
        try:
            # Search across all namespaces by using activity ID in the filter
            # We need to do a dummy vector search with a filter
            results = self.index.query(
                vector=[0] * self.dimension,  # Dummy vector
                top_k=10,
                filter={"activity_id": activity_id},
                include_metadata=True
            )
            
            # Convert to VectorSearchResult objects
            search_results = []
            for match in results.matches:
                if match.metadata and match.metadata.get('activity_id') == activity_id:
                    search_results.append(VectorSearchResult(
                        id=match.id,
                        score=match.score,
                        metadata=match.metadata
                    ))
            
            # If not found by activity_id, try main_activity_id
            if not search_results:
                results = self.index.query(
                    vector=[0] * self.dimension,
                    top_k=10,
                    filter={"main_activity_id": activity_id},
                    include_metadata=True
                )
                
                for match in results.matches:
                    if match.metadata and match.metadata.get('main_activity_id') == activity_id:
                        search_results.append(VectorSearchResult(
                            id=match.id,
                            score=match.score,
                            metadata=match.metadata
                        ))
            
            return search_results
            
        except Exception as e:
            # If activity not found, return empty list
            if "404" in str(e) or "not found" in str(e).lower():
                return []
            else:
                raise e