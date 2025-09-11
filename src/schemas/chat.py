"""Schemas for chat and conversation endpoints."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request schema for chat queries."""
    query: str = Field(..., min_length=1, max_length=2000, description="User's fitness question")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for context")
    search_limit: Optional[int] = Field(15, ge=1, le=100, description="Number of activities to search")
    include_follow_ups: bool = Field(True, description="Whether to generate follow-up questions")


class ActivitySummary(BaseModel):
    """Summary of a relevant activity."""
    garmin_activity_id: Optional[str] = None
    activity_type: Optional[str] = None
    activity_name: Optional[str] = None
    date: Optional[str] = None
    distance_km: Optional[float] = None
    duration_minutes: Optional[float] = None
    average_speed_kmh: Optional[float] = None
    average_heart_rate: Optional[int] = None
    average_power: Optional[int] = None
    elevation_gain: Optional[float] = None
    calories: Optional[int] = None
    relevance_score: Optional[float] = Field(None, description="Search relevance score")


class ChatResponse(BaseModel):
    """Response schema for chat queries."""
    response: str = Field(..., description="AI assistant's response")
    relevant_activities: List[ActivitySummary] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    conversation_id: str = Field(..., description="Conversation ID for follow-up queries")
    timestamp: str = Field(..., description="Response timestamp")
    activity_count: int = Field(..., description="Number of relevant activities found")
    error: Optional[str] = Field(None, description="Error message if any")


class ConversationMessage(BaseModel):
    """A message in a conversation."""
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(None, description="Message timestamp")


class ConversationHistory(BaseModel):
    """Conversation history response."""
    conversation_id: str
    user_id: str
    messages: List[ConversationMessage]
    created_at: str
    updated_at: str
    message_count: int


class ConversationSummary(BaseModel):
    """Summary of a conversation for listing."""
    conversation_id: str
    created_at: str
    updated_at: str
    message_count: int
    last_message: Optional[str] = None
    last_message_role: Optional[str] = None


class UserConversationsResponse(BaseModel):
    """Response for user's conversations list."""
    conversations: List[ConversationSummary]
    total_count: int


class ActivityStats(BaseModel):
    """Statistics about user's vectorized activities."""
    total_vectors: int = Field(..., description="Total vectors stored")
    estimated_activities: int = Field(..., description="Estimated number of activities")
    namespace: str = Field(..., description="User's vector namespace")
    last_updated: str = Field(..., description="Last update timestamp")


class SuggestionRequest(BaseModel):
    """Request for follow-up suggestions."""
    recent_query: Optional[str] = Field(None, description="Recent user query for context")
    activity_type: Optional[str] = Field(None, description="Focus on specific activity type")


class SuggestionsResponse(BaseModel):
    """Response with suggested questions."""
    suggestions: List[str] = Field(..., description="Suggested questions")
    category: str = Field(..., description="Suggestion category")


class HealthCheckResponse(BaseModel):
    """Health check response for chat services."""
    status: str = Field(..., description="Service status")
    services: Dict[str, Any] = Field(..., description="Individual service statuses")
    timestamp: str = Field(..., description="Check timestamp")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")