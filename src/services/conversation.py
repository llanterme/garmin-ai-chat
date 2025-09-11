"""Conversation service for managing chat sessions and orchestrating RAG responses."""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ..core.logging import get_logger
from .activity_ingestion import ActivityIngestionService
from .llm import ChatMessage, LLMService

logger = get_logger(__name__)


class ConversationContext:
    """Context for maintaining conversation state."""
    
    def __init__(self, user_id: str, conversation_id: str):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.messages: List[ChatMessage] = []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.metadata: Dict = {}
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation."""
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now()
        )
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def get_recent_messages(self, limit: int = 10) -> List[ChatMessage]:
        """Get recent messages for context."""
        return self.messages[-limit:] if self.messages else []


class ConversationService:
    """Service for handling conversational queries about fitness activities."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.ingestion_service = ActivityIngestionService(session)
        self.llm_service = LLMService()
        
        # In-memory storage for conversation contexts
        # In production, this would be stored in Redis or database
        self.conversations: Dict[str, ConversationContext] = {}
        
        # Cleanup old conversations periodically (24 hour TTL)
        self.conversation_ttl_hours = 24
    
    async def process_query(
        self,
        user_id: str,
        query: str,
        conversation_id: Optional[str] = None,
        search_limit: Optional[int] = None,
        include_follow_ups: bool = True
    ) -> Dict:
        """Process a conversational query and return response with activities."""
        try:
            # Get or create conversation context
            context = self._get_or_create_context(user_id, conversation_id)
            
            # Add user message to context
            context.add_message("user", query)
            
            # Clean up old conversations
            self._cleanup_old_conversations()
            
            logger.info(f"Processing query for user {user_id}: {query[:100]}")
            
            # Search for relevant activities
            relevant_activities = await self.ingestion_service.search_user_activities(
                user_id=user_id,
                query=query,
                top_k=search_limit or 15
            )
            
            logger.info(f"Found {len(relevant_activities)} relevant activities")
            
            # Add performance context for better responses
            enhanced_activities = await self._add_performance_context(
                user_id, relevant_activities, query
            )
            
            # Generate conversational response using LLM
            conversation_history = context.get_recent_messages(limit=8)  # Exclude current user message
            
            response = await self.llm_service.generate_conversational_response(
                user_query=query,
                relevant_activities=enhanced_activities,
                conversation_history=conversation_history[:-1],  # Exclude current user message
                user_context={"user_id": user_id}
            )
            
            # Add assistant response to context
            context.add_message("assistant", response)
            
            # Generate follow-up questions if requested
            follow_up_questions = []
            if include_follow_ups:
                try:
                    follow_up_questions = await self.llm_service.generate_follow_up_questions(
                        user_query=query,
                        response=response,
                        relevant_activities=enhanced_activities
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate follow-up questions: {str(e)}")
                    follow_up_questions = self._get_default_follow_ups(query)
            
            result = {
                "response": response,
                "relevant_activities": self._format_activities_for_response(enhanced_activities),
                "follow_up_questions": follow_up_questions,
                "conversation_id": context.conversation_id,
                "timestamp": datetime.now().isoformat(),
                "activity_count": len(enhanced_activities)
            }
            
            logger.info(f"Successfully processed query for user {user_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process query for user {user_id}: {str(e)}")
            return {
                "response": self._get_error_response(query, str(e)),
                "relevant_activities": [],
                "follow_up_questions": ["Can you try rephrasing your question?", "Would you like me to check your data sync status?"],
                "conversation_id": conversation_id or str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "activity_count": 0,
                "error": str(e)
            }
    
    async def get_conversation_history(
        self, 
        conversation_id: str, 
        limit: int = 50
    ) -> Optional[Dict]:
        """Get conversation history by ID."""
        context = self.conversations.get(conversation_id)
        if not context:
            return None
        
        messages = context.get_recent_messages(limit)
        return {
            "conversation_id": conversation_id,
            "user_id": context.user_id,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None
                }
                for msg in messages
            ],
            "created_at": context.created_at.isoformat(),
            "updated_at": context.updated_at.isoformat(),
            "message_count": len(context.messages)
        }
    
    async def clear_conversation(self, conversation_id: str) -> bool:
        """Clear a conversation by ID."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Cleared conversation {conversation_id}")
            return True
        return False
    
    async def get_user_conversations(self, user_id: str) -> List[Dict]:
        """Get all conversations for a user."""
        user_conversations = []
        for conv_id, context in self.conversations.items():
            if context.user_id == user_id:
                last_message = context.messages[-1] if context.messages else None
                user_conversations.append({
                    "conversation_id": conv_id,
                    "created_at": context.created_at.isoformat(),
                    "updated_at": context.updated_at.isoformat(),
                    "message_count": len(context.messages),
                    "last_message": last_message.content[:100] + "..." if last_message and len(last_message.content) > 100 else last_message.content if last_message else None,
                    "last_message_role": last_message.role if last_message else None
                })
        
        # Sort by updated_at descending
        user_conversations.sort(key=lambda x: x["updated_at"], reverse=True)
        return user_conversations
    
    def _get_or_create_context(
        self, 
        user_id: str, 
        conversation_id: Optional[str] = None
    ) -> ConversationContext:
        """Get existing conversation context or create new one."""
        if conversation_id and conversation_id in self.conversations:
            context = self.conversations[conversation_id]
            # Verify user ownership
            if context.user_id != user_id:
                logger.warning(f"User {user_id} attempted to access conversation {conversation_id} owned by {context.user_id}")
                conversation_id = None
        
        if not conversation_id or conversation_id not in self.conversations:
            conversation_id = str(uuid.uuid4())
            context = ConversationContext(user_id, conversation_id)
            self.conversations[conversation_id] = context
            logger.info(f"Created new conversation {conversation_id} for user {user_id}")
        else:
            context = self.conversations[conversation_id]
            logger.debug(f"Using existing conversation {conversation_id}")
        
        return context
    
    async def _add_performance_context(
        self, 
        user_id: str, 
        activities: List[Dict], 
        query: str
    ) -> List[Dict]:
        """Add performance context and user averages to activities."""
        if not activities:
            return activities
        
        try:
            # Calculate basic averages from the activities found
            total_activities = len(activities)
            
            # Calculate averages for numeric fields
            numeric_fields = [
                "distance_km", "duration_minutes", "average_speed_kmh", 
                "average_heart_rate", "average_power", "calories", "elevation_gain"
            ]
            
            averages = {}
            for field in numeric_fields:
                values = [
                    float(activity.get(field, 0)) 
                    for activity in activities 
                    if activity.get(field) is not None and float(activity.get(field, 0)) > 0
                ]
                if values:
                    averages[f"avg_{field}"] = round(sum(values) / len(values), 2)
            
            # Add context to each activity
            enhanced_activities = []
            for activity in activities:
                enhanced_activity = dict(activity)
                
                # Add performance indicators
                self._add_performance_indicators(enhanced_activity, averages)
                
                enhanced_activities.append(enhanced_activity)
            
            # Add summary statistics as metadata to first activity
            if enhanced_activities:
                enhanced_activities[0]["_context"] = {
                    "total_activities_found": total_activities,
                    "user_averages": averages,
                    "query_type": self._classify_query_type(query)
                }
            
            return enhanced_activities
            
        except Exception as e:
            logger.warning(f"Failed to add performance context: {str(e)}")
            return activities
    
    def _add_performance_indicators(self, activity: Dict, averages: Dict) -> None:
        """Add performance indicators to an activity."""
        try:
            # Compare to averages
            for field in ["distance_km", "duration_minutes", "average_speed_kmh", "average_heart_rate", "average_power"]:
                activity_value = activity.get(field)
                avg_value = averages.get(f"avg_{field}")
                
                if activity_value is not None and avg_value is not None:
                    try:
                        activity_val = float(activity_value)
                        if activity_val > avg_value * 1.1:
                            activity[f"{field}_vs_avg"] = "above_average"
                        elif activity_val < avg_value * 0.9:
                            activity[f"{field}_vs_avg"] = "below_average"
                        else:
                            activity[f"{field}_vs_avg"] = "average"
                    except (ValueError, TypeError):
                        continue
        
        except Exception as e:
            logger.warning(f"Failed to add performance indicators: {str(e)}")
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query type for context."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["average", "total", "sum", "count"]):
            return "aggregation"
        elif any(word in query_lower for word in ["best", "worst", "fastest", "slowest"]):
            return "comparison"
        elif any(word in query_lower for word in ["yesterday", "today", "last week", "this month"]):
            return "temporal"
        elif any(word in query_lower for word in ["trend", "progress", "improvement"]):
            return "analysis"
        else:
            return "general"
    
    def _format_activities_for_response(self, activities: List[Dict]) -> List[Dict]:
        """Format activities for API response (remove internal fields)."""
        formatted = []
        for activity in activities:
            # Create clean copy without internal fields
            clean_activity = {
                k: v for k, v in activity.items() 
                if not k.startswith("_") and k not in ["summary_text"]
            }
            formatted.append(clean_activity)
        
        return formatted
    
    def _get_default_follow_ups(self, query: str) -> List[str]:
        """Get default follow-up questions based on query."""
        query_lower = query.lower()
        
        if "run" in query_lower or "running" in query_lower:
            return [
                "How does your running pace compare to last month?",
                "What's your average weekly running distance?",
                "Would you like to see your heart rate trends during runs?"
            ]
        elif "ride" in query_lower or "cycling" in query_lower:
            return [
                "How's your cycling power output trending?",
                "What's your longest ride this month?",
                "Would you like to analyze your cycling efficiency?"
            ]
        else:
            return [
                "How has your overall training volume changed recently?",
                "What activities have you been most consistent with?",
                "Would you like to see your performance trends?"
            ]
    
    def _get_error_response(self, query: str, error: str) -> str:
        """Generate error response for failed queries."""
        return f"""I encountered an issue while processing your query about "{query}". 

This might be due to:
• A temporary connection issue with the AI services
• Your activities may need to be processed first
• The query format might need adjustment

Please try again in a moment, or rephrase your question. If the problem persists, you might need to sync your Garmin data first."""
    
    def _cleanup_old_conversations(self) -> None:
        """Clean up old conversations that exceed TTL."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.conversation_ttl_hours)
            
            # Find conversations to remove
            to_remove = []
            for conv_id, context in self.conversations.items():
                if context.updated_at < cutoff_time:
                    to_remove.append(conv_id)
            
            # Remove old conversations
            for conv_id in to_remove:
                del self.conversations[conv_id]
            
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old conversations")
                
        except Exception as e:
            logger.warning(f"Failed to cleanup old conversations: {str(e)}")