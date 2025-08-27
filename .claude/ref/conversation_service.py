"""
Conversational Q&A service that combines vector search with LLM responses.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from pydantic import BaseModel

from .activity_ingestion import ActivityIngestionService
from .llm_service import LLMService, ChatMessage


class ConversationContext(BaseModel):
    """Context for maintaining conversation state."""
    user_id: str
    messages: List[ChatMessage]
    created_at: datetime
    updated_at: datetime


class ConversationResponse(BaseModel):
    """Response from conversational Q&A."""
    response: str
    relevant_activities: List[Dict[str, Any]]
    follow_up_questions: List[str]
    conversation_id: str
    timestamp: datetime


class ConversationService:
    """
    Service for handling conversational queries about fitness activities.
    """
    
    def __init__(self):
        self.ingestion_service = ActivityIngestionService()
        self.llm_service = LLMService()
        self.logger = logging.getLogger(__name__)
        
        # Simple in-memory storage for conversation contexts
        # In production, this would be stored in a database
        self.conversations: Dict[str, ConversationContext] = {}
    
    async def process_query(
        self,
        user_id: str,
        query: str,
        conversation_id: Optional[str] = None,
        search_limit: Optional[int] = None
    ) -> ConversationResponse:
        """
        Process a conversational query about fitness activities.
        
        Args:
            user_id: Strava athlete ID
            query: User's natural language query
            conversation_id: Optional conversation ID for context
            search_limit: Number of activities to retrieve for context
            
        Returns:
            Conversational response with activities and follow-ups
        """
        try:
            # Get or create conversation context
            context = self._get_or_create_context(user_id, conversation_id)
            
            # Add user message to context
            user_message = ChatMessage(
                role="user",
                content=query,
                timestamp=datetime.now()
            )
            context.messages.append(user_message)
            
            # Determine appropriate search limit based on query type
            if search_limit is None:
                search_limit = self._determine_search_limit(query)
            
            # Create temporal filter for recent queries
            filter_metadata = self._create_temporal_filter(query)
            
            # Perform semantic search for relevant activities
            self.logger.info(f"Searching activities for user {user_id} with query: {query} (limit: {search_limit})")
            if filter_metadata:
                self.logger.info(f"Applied temporal filter: {filter_metadata}")
            
            relevant_activities = await self.ingestion_service.search_user_activities(
                user_id=user_id,
                query=query,
                top_k=search_limit,
                filter_metadata=filter_metadata
            )
            
            self.logger.info(f"Found {len(relevant_activities)} relevant activities")
            
            # Add performance context for better accuracy
            enhanced_activities = await self._add_performance_context(user_id, relevant_activities, query)
            
            # Generate conversational response using LLM
            response_text = self.llm_service.generate_conversational_response(
                user_query=query,
                relevant_activities=enhanced_activities,
                conversation_history=context.messages[:-1]  # Exclude current message
            )
            
            # Add assistant response to context
            assistant_message = ChatMessage(
                role="assistant",
                content=response_text,
                timestamp=datetime.now()
            )
            context.messages.append(assistant_message)
            
            # Generate follow-up questions
            follow_up_questions = self.llm_service.generate_follow_up_questions(
                user_query=query,
                response=response_text,
                activities=relevant_activities
            )
            
            # Update context
            context.updated_at = datetime.now()
            
            return ConversationResponse(
                response=response_text,
                relevant_activities=relevant_activities,
                follow_up_questions=follow_up_questions,
                conversation_id=context.user_id,  # Using user_id as conversation_id for simplicity
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error processing query for user {user_id}: {str(e)}")
            raise Exception(f"Failed to process query: {str(e)}")
    
    def _determine_search_limit(self, query: str) -> int:
        """
        Determine appropriate search limit based on query type.
        
        Args:
            query: User's natural language query
            
        Returns:
            Appropriate search limit for the query type
        """
        query_lower = query.lower()
        
        # Keywords that indicate temporal/recent queries
        temporal_recent_keywords = [
            'today', 'recent', 'lately', 'this week', 'past few days',
            'yesterday', 'last few', 'currently', 'now'
        ]
        
        # Keywords that indicate aggregation/calculation queries
        aggregation_keywords = [
            'average', 'mean', 'total', 'sum', 'calculate', 'compute',
            'all my', 'every', 'total distance', 'total time', 'overall',
            'across all', 'throughout', 'over all', 'statistics', 'stats',
            'this year', 'this month', 'past year', 'past month', 'last year',
            'compare', 'comparison', 'trend', 'trends', 'progress', 'improvement'
        ]
        
        # Keywords that indicate comprehensive analysis queries
        comprehensive_keywords = [
            'analyze', 'analysis', 'insights', 'patterns', 'summary',
            'overview', 'breakdown', 'distribution', 'frequency',
            'performance', 'training', 'fitness level', 'progress'
        ]
        
        # Check for temporal/recent queries - need more context for accuracy
        if any(keyword in query_lower for keyword in temporal_recent_keywords):
            self.logger.info("Detected temporal/recent query, using enhanced search limit")
            return 20   # Higher limit for recent context and comparison
        
        # Check for aggregation queries - need lots of data
        elif any(keyword in query_lower for keyword in aggregation_keywords):
            self.logger.info("Detected aggregation query, using high search limit")
            return 100  # High limit for calculations
        
        # Check for comprehensive analysis queries - need moderate data
        elif any(keyword in query_lower for keyword in comprehensive_keywords):
            self.logger.info("Detected comprehensive analysis query, using medium search limit")
            return 50   # Medium limit for analysis
        
        # Default for specific/conversational queries - increased from 5 to 15
        else:
            self.logger.info("Detected specific query, using enhanced default search limit")
            return 15   # Enhanced default limit for better accuracy
    
    def _create_temporal_filter(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Create temporal metadata filter for recent/today queries.
        
        Args:
            query: User's natural language query
            
        Returns:
            Metadata filter dictionary or None
        """
        query_lower = query.lower()
        now = datetime.now()
        
        # Today queries
        if 'today' in query_lower:
            today_str = now.strftime("%Y-%m-%d")
            self.logger.info(f"Filtering for today's activities: {today_str}")
            return {"date": today_str}
        
        # Yesterday queries
        elif 'yesterday' in query_lower:
            yesterday = now - timedelta(days=1)
            yesterday_str = yesterday.strftime("%Y-%m-%d")
            self.logger.info(f"Filtering for yesterday's activities: {yesterday_str}")
            return {"date": yesterday_str}
        
        # This week queries
        elif 'this week' in query_lower:
            days_since_monday = now.weekday()
            start_of_week = now - timedelta(days=days_since_monday)
            start_date = start_of_week.strftime("%Y-%m-%d")
            self.logger.info(f"Filtering for this week's activities since: {start_date}")
            # Use a range filter for this week
            return {"date": {"$gte": start_date}}
        
        # Recent/lately queries (last 7 days)
        elif any(keyword in query_lower for keyword in ['recent', 'lately', 'past few days', 'last few']):
            cutoff_date = now - timedelta(days=7)
            cutoff_str = cutoff_date.strftime("%Y-%m-%d")
            self.logger.info(f"Filtering for recent activities since: {cutoff_str}")
            return {"date": {"$gte": cutoff_str}}
        
        # No temporal filter needed
        return None
    
    async def _add_performance_context(self, user_id: str, activities: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Add performance context to activities for better AI responses.
        
        Args:
            user_id: Strava athlete ID
            activities: List of relevant activities
            query: Original user query
            
        Returns:
            Activities enhanced with performance context
        """
        if not activities:
            return activities
        
        try:
            # Get user's fitness status for context
            fitness_status = await self.ingestion_service.get_user_fitness_status(user_id)
            
            # Calculate user averages for comparison (last 30 activities)
            user_averages = await self._calculate_user_averages(user_id)
            
            # Check if query indicates performance judgment
            performance_keywords = ['sucked', 'bad', 'terrible', 'awful', 'good', 'great', 'amazing', 'best', 'worst']
            has_performance_sentiment = any(keyword in query.lower() for keyword in performance_keywords)
            
            enhanced_activities = []
            for activity in activities:
                enhanced_activity = activity.copy()
                
                # Add performance context if sentiment detected
                if has_performance_sentiment:
                    context_notes = []
                    
                    # Compare to personal averages
                    if user_averages:
                        sport_type = activity.get('activity_type', '').lower()
                        if sport_type in user_averages:
                            avg_data = user_averages[sport_type]
                            
                            # Distance comparison
                            if activity.get('distance_km') and avg_data.get('avg_distance'):
                                distance_ratio = activity['distance_km'] / avg_data['avg_distance']
                                if distance_ratio > 1.2:
                                    context_notes.append("longer than usual")
                                elif distance_ratio < 0.8:
                                    context_notes.append("shorter than usual")
                            
                            # Pace comparison for running
                            if sport_type in ['run', 'running'] and activity.get('avg_pace_min_per_km') and avg_data.get('avg_pace'):
                                pace_diff = activity['avg_pace_min_per_km'] - avg_data['avg_pace']
                                if pace_diff < -0.3:
                                    context_notes.append("faster pace than average")
                                elif pace_diff > 0.3:
                                    context_notes.append("slower pace than average")
                            
                            # Power comparison for cycling
                            if sport_type in ['ride', 'cycling'] and activity.get('avg_power_watts') and avg_data.get('avg_power'):
                                power_ratio = activity['avg_power_watts'] / avg_data['avg_power']
                                if power_ratio > 1.1:
                                    context_notes.append("higher power than average")
                                elif power_ratio < 0.9:
                                    context_notes.append("lower power than average")
                    
                    # Add intensity context using new metrics
                    if activity.get('intensity_factor'):
                        if_value = activity['intensity_factor']
                        if if_value > 1.0:
                            context_notes.append("very high intensity effort")
                        elif if_value > 0.85:
                            context_notes.append("moderate-high intensity")
                        elif if_value < 0.65:
                            context_notes.append("easy recovery effort")
                    
                    # Add efficiency context
                    if activity.get('efficiency_factor') and user_averages.get('all', {}).get('avg_efficiency'):
                        ef_ratio = activity['efficiency_factor'] / user_averages['all']['avg_efficiency']
                        if ef_ratio < 0.9:
                            context_notes.append("lower efficiency than usual")
                        elif ef_ratio > 1.1:
                            context_notes.append("higher efficiency than usual")
                    
                    # Add context to activity
                    if context_notes:
                        enhanced_activity['performance_context'] = f"This activity was {', '.join(context_notes)}"
                
                # Add current fitness context
                enhanced_activity['current_fitness_status'] = fitness_status.get('status', 'unknown')
                enhanced_activity['fitness_recommendation'] = fitness_status.get('recommendation', '')
                
                enhanced_activities.append(enhanced_activity)
            
            self.logger.info(f"Enhanced {len(enhanced_activities)} activities with performance context")
            return enhanced_activities
            
        except Exception as e:
            self.logger.warning(f"Failed to add performance context: {str(e)}")
            return activities  # Return original activities if enhancement fails
    
    async def _calculate_user_averages(self, user_id: str) -> Dict[str, Dict[str, float]]:
        """Calculate user's average performance metrics for comparison."""
        try:
            # Get recent activities for baseline (last 30 activities)
            baseline_activities = await self.ingestion_service.search_user_activities(
                user_id=user_id,
                query="recent training activities",
                top_k=30
            )
            
            if not baseline_activities:
                return {}
            
            # Group by sport type and calculate averages
            sport_groups = {}
            all_activities = []
            
            for activity in baseline_activities:
                sport_type = activity.get('activity_type', '').lower()
                if sport_type not in sport_groups:
                    sport_groups[sport_type] = []
                sport_groups[sport_type].append(activity)
                all_activities.append(activity)
            
            averages = {}
            
            # Calculate per-sport averages
            for sport_type, activities in sport_groups.items():
                if len(activities) >= 3:  # Only if we have enough data
                    distances = [a.get('distance_km', 0) for a in activities if a.get('distance_km')]
                    paces = [a.get('avg_pace_min_per_km', 0) for a in activities if a.get('avg_pace_min_per_km')]
                    powers = [a.get('avg_power_watts', 0) for a in activities if a.get('avg_power_watts')]
                    
                    averages[sport_type] = {}
                    if distances:
                        averages[sport_type]['avg_distance'] = sum(distances) / len(distances)
                    if paces:
                        averages[sport_type]['avg_pace'] = sum(paces) / len(paces)
                    if powers:
                        averages[sport_type]['avg_power'] = sum(powers) / len(powers)
            
            # Calculate overall averages
            if all_activities:
                efficiencies = [a.get('efficiency_factor', 0) for a in all_activities if a.get('efficiency_factor')]
                if efficiencies:
                    averages['all'] = {'avg_efficiency': sum(efficiencies) / len(efficiencies)}
            
            return averages
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate user averages: {str(e)}")
            return {}
    
    def _get_or_create_context(
        self,
        user_id: str,
        conversation_id: Optional[str] = None
    ) -> ConversationContext:
        """Get existing conversation context or create a new one."""
        context_id = conversation_id or user_id
        
        if context_id not in self.conversations:
            self.conversations[context_id] = ConversationContext(
                user_id=user_id,
                messages=[],
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        
        return self.conversations[context_id]
    
    async def get_fitness_insights(
        self,
        user_id: str,
        time_period: str = "month",
        activity_limit: int = 50
    ) -> str:
        """
        Generate comprehensive fitness insights for a user.
        
        Args:
            user_id: Strava athlete ID
            time_period: Time period for analysis (week, month, year)
            activity_limit: Maximum number of activities to analyze
            
        Returns:
            Fitness insights and trends
        """
        try:
            # Get recent activities for analysis
            activities = await self.ingestion_service.search_user_activities(
                user_id=user_id,
                query="recent activities training fitness",
                top_k=activity_limit
            )
            
            # Generate trend analysis
            insights = self.llm_service.analyze_fitness_trends(
                activities=activities,
                time_period=time_period
            )
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating insights for user {user_id}: {str(e)}")
            raise Exception(f"Failed to generate insights: {str(e)}")
    
    def get_conversation_history(
        self,
        user_id: str,
        conversation_id: Optional[str] = None,
        limit: int = 50
    ) -> List[ChatMessage]:
        """
        Get conversation history for a user.
        
        Args:
            user_id: Strava athlete ID
            conversation_id: Optional conversation ID
            limit: Maximum number of messages to return
            
        Returns:
            List of chat messages
        """
        context_id = conversation_id or user_id
        
        if context_id not in self.conversations:
            return []
        
        context = self.conversations[context_id]
        return context.messages[-limit:]
    
    def clear_conversation(
        self,
        user_id: str,
        conversation_id: Optional[str] = None
    ) -> None:
        """
        Clear conversation history for a user.
        
        Args:
            user_id: Strava athlete ID
            conversation_id: Optional conversation ID
        """
        context_id = conversation_id or user_id
        
        if context_id in self.conversations:
            del self.conversations[context_id]
    
    async def ask_follow_up(
        self,
        user_id: str,
        follow_up_question: str,
        conversation_id: Optional[str] = None
    ) -> ConversationResponse:
        """
        Process a follow-up question from suggested questions.
        
        Args:
            user_id: Strava athlete ID
            follow_up_question: The follow-up question to process
            conversation_id: Optional conversation ID for context
            
        Returns:
            Conversational response
        """
        return await self.process_query(
            user_id=user_id,
            query=follow_up_question,
            conversation_id=conversation_id
        )
    
    def get_conversation_summary(
        self,
        user_id: str,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get a summary of the conversation.
        
        Args:
            user_id: Strava athlete ID
            conversation_id: Optional conversation ID
            
        Returns:
            Conversation summary
        """
        context_id = conversation_id or user_id
        
        if context_id not in self.conversations:
            return {
                "exists": False,
                "message_count": 0,
                "created_at": None,
                "updated_at": None
            }
        
        context = self.conversations[context_id]
        
        return {
            "exists": True,
            "message_count": len(context.messages),
            "created_at": context.created_at,
            "updated_at": context.updated_at,
            "last_message": context.messages[-1].content if context.messages else None
        }